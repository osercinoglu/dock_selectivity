#!/usr/bin/env python3
"""Drop-in stand-in for the gnina binary that runs it on a remote GPU host.

Why a shim rather than a pipeline stage
---------------------------------------

``consensus_docker.py`` invokes gnina as one self-contained command::

    gnina -r <receptor.pdb> -l <ligand.sdf> \\
          --center_x/y/z --size_x/y/z -o <out.sdf> --log <out.log> --cpu N

and does everything else -- splitting ``out.sdf`` into per-pose files, building
``complex_*.pdb``, writing ``results.csv`` -- **locally, after gnina returns**. So
only that one call has to happen on the GPU box. This script presents the same
command line, ships the two inputs over SSH, runs gnina in a container on the
remote, and writes ``out.sdf`` and ``out.log`` back to the exact local paths it
was given.

Point ``GNINA_PATH`` at this file and nothing else changes::

    GNINA_PATH = "/home/onur/repos/dock_selectivity/gnina_remote.py"

Surviving a flaky link
----------------------

A campaign is thousands of jobs over days, so the connection *will* drop. Three
things make that survivable:

1. **The remote job runs detached.** It is launched with ``setsid nohup`` and
   writes its exit code to a marker file when it finishes, so losing the
   connection mid-run does not kill the container -- we reconnect and collect.
   Without this, an outage during a 2.5-minute gnina run destroys the work.
2. **Every SSH operation retries** with exponential backoff, but only for
   *transient* failures. A missing input or a non-zero gnina exit is permanent and
   fails immediately rather than burning five retries and burying the real error.
3. **The remote directory is content-addressed** -- a hash of the inputs and
   arguments. Re-running the same job finds the finished result and just downloads
   it, so a local crash, a Ctrl-C or a pipeline restart costs nothing. This is the
   same resume discipline the rest of the pipeline uses: work already done is not
   done again.

The three modes (``GNINA_REMOTE_MODE``)
---------------------------------------

``sync`` (the default, and byte-for-byte what this shim has always done) blocks
until the remote job finishes and then downloads it. That is correct but it
pins a local process to a remote GPU for ~2.5 minutes per job.

The other two modes exist so ``batch_dock.py`` can dock on the GPU while it is
busy locally with smina/GOLD/LeDock. They **split this script at the point where
it waits**, and nothing else moves:

``submit``
    upload, take the remote GPU mutex, launch the detached job, write a ticket,
    and exit 3 *without* producing ``out.sdf``. ``consensus_docker`` then fails
    its own ``split_mol`` on the absent file and writes nothing -- which is the
    intended outcome. The **ticket**, not the exit code, is the signal.
``collect``
    download ``out.sdf``/``out.log`` for an already-finished remote job and exit
    0, so the *same* ``consensus_docker`` command then runs its own
    ``split_mol`` / ``make_complex`` / ``parse_gnina`` over them.

The consequence worth stating plainly: the poses and the CNN scores are produced
by exactly the same lines of code as a local run. This shim never parses a pose
or a score, in any mode. See section 1 of the decoupling specification.

Configuration (environment, so no secrets live in the repo)::

    GNINA_REMOTE_HOST     user@host          required
    GNINA_REMOTE_PORT     ssh port           default 22
    GNINA_REMOTE_IMAGE    container image    default gnina/gnina:v1.3.3
    GNINA_REMOTE_DIR      remote scratch     default gnina_jobs
    GNINA_REMOTE_KEY      private key path   optional
    GNINA_REMOTE_NATIVE   native gnina path on the remote; skips the container
    GNINA_REMOTE_RETRIES  transient-failure attempts, default 5
    GNINA_REMOTE_TIMEOUT  seconds to wait for the remote run, default 7200
    GNINA_REMOTE_KEEP     set to 1 to keep the remote job directory after success

    GNINA_REMOTE_MODE       sync (default) | submit | collect
    GNINA_REMOTE_TICKET     where to write the ticket JSON
    GNINA_REMOTE_JOB_DIR    absolute remote dir; overrides content addressing
    GNINA_REMOTE_LOCK       1 to honour the GPU mutex in sync mode too (default 0)
    GNINA_REMOTE_LOCK_STALE seconds before a lock with no live owner is broken
    GNINA_REMOTE_LOCK_WAIT  seconds a submit may wait for the mutex (default 0)

Authentication is SSH keys only. Passwords are deliberately unsupported: this runs
unattended once per job, and the alternatives (``sshpass``, a stored secret) put
the password in the process list or on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import shlex
import subprocess
import sys
import time

DEFAULT_IMAGE = "gnina/gnina:v1.3.3"
DEFAULT_REMOTE_DIR = "gnina_jobs"
DEFAULT_RETRIES = 5
DEFAULT_RUN_TIMEOUT = 7200      # seconds to wait for the detached remote job
POLL_INTERVAL = 10              # seconds between completion checks

#: Modes. ``sync`` is the historical behaviour and the default: an unset
#: GNINA_REMOTE_MODE must never change what an existing caller gets.
MODE_SYNC = "sync"
MODE_SUBMIT = "submit"
MODE_COLLECT = "collect"
MODES = (MODE_SYNC, MODE_SUBMIT, MODE_COLLECT)

#: Exit codes. 0/1/2 keep exactly their historical meaning; 3/4/5 are new and
#: are *not* errors -- they are outcomes the queue expects and acts on.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_SUBMITTED = 3        # submit: launched (or already finished); see the ticket
EXIT_NOT_READY = 4        # collect: the remote job has not finished yet
EXIT_LOCK_BUSY = 5        # the remote GPU mutex is held by someone else

#: Name of the mutex directory inside the remote scratch root.
LOCK_DIRNAME = "gpu.lock"

#: Default seconds before a lock whose owner cannot be found is broken. The
#: floor matters: a lock must never look stale while the job that holds it could
#: still be running, so this is kept above any plausible job deadline.
DEFAULT_LOCK_STALE = 1800

#: Keep-alives so a long, quiet gnina run is not culled by a NAT or firewall idle
#: timeout, and so a dead link is noticed in ~60s instead of hanging indefinitely.
_SSH_COMMON = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=20",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=4",
]

#: Message fragments meaning "the network misbehaved" rather than "what you asked
#: for is wrong". Only these are retried: retrying a permanent failure wastes
#: minutes per job across thousands of jobs and buries the real message.
_TRANSIENT_MARKERS = (
    "connection refused", "connection closed", "connection reset",
    "connection timed out", "timed out", "broken pipe", "network is unreachable",
    "no route to host", "temporary failure in name resolution",
    "kex_exchange_identification", "banner exchange", "operation timed out",
    "host is down", "software caused connection abort",
)


class RemoteGninaError(RuntimeError):
    """The remote leg failed. Never swallowed into a fake success."""


class TransientError(RemoteGninaError):
    """A network-level failure worth retrying."""


def ssh_base(host, port, key):
    cmd = ["ssh"] + list(_SSH_COMMON)
    if port:
        cmd += ["-p", str(port)]
    if key:
        cmd += ["-i", key]
    return cmd + [host]


def scp_base(port, key):
    cmd = ["scp", "-q"] + list(_SSH_COMMON)
    if port:
        cmd += ["-P", str(port)]
    if key:
        cmd += ["-i", key]
    return cmd


def is_transient(returncode, stderr):
    """True if this failure is worth retrying."""
    text = (stderr or "").lower()
    if any(marker in text for marker in _TRANSIENT_MARKERS):
        return True
    # 255 is ssh's own "connection failed" code. scp uses 1 for both transfer and
    # usage errors, so 255 is the only bare code safe to treat as transient.
    return returncode == 255


def run_resilient(cmd, what, retries, timeout=None, log=None):
    """Run an SSH/SCP command, retrying transient failures with exponential backoff."""
    delay = 5.0
    last = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            last = TransientError("%s timed out after %ss" % (what, timeout))
        else:
            if proc.returncode == 0:
                return proc
            err = (proc.stderr or "").strip()
            if is_transient(proc.returncode, err):
                last = TransientError("%s: %s" % (what, err[:300] or "exit 255"))
            else:
                raise RemoteGninaError(
                    "%s failed (exit %d)\n  cmd: %s\n  stderr: %s"
                    % (what, proc.returncode,
                       " ".join(shlex.quote(c) for c in cmd), err[:2000]))
        if attempt < max(1, retries):
            if log:
                log("%s -- transient, retry %d/%d in %.0fs"
                    % (last, attempt, max(1, retries) - 1, delay))
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
    raise last or RemoteGninaError("%s failed" % what)


def parse_args(argv):
    """Split gnina's command line into the paths we handle and the rest.

    Unknown flags pass through untouched, so a gnina option this shim has never
    heard of still reaches the binary.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-r", "--receptor")
    parser.add_argument("-l", "--ligand")
    parser.add_argument("-o", "--out")
    parser.add_argument("--log")
    known, rest = parser.parse_known_args(argv)
    return known, rest


def job_key(receptor, ligand, passthrough):
    """Content address for this job: same inputs and arguments -> same remote dir.

    This is what makes a re-run free. If a previous attempt finished but the link
    died before the download, the next invocation finds the completed job and
    collects it instead of docking again.
    """
    digest = hashlib.sha256()
    for path in (receptor, ligand):
        digest.update(os.path.basename(path).encode())
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    digest.update(("\x00".join(passthrough)).encode())
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# the remote GPU mutex
# ---------------------------------------------------------------------------
#
# One GPU means one gnina at a time. The mutex lives on the REMOTE host, not
# here, because that is the only place all the contenders can see: a local lock
# would not stop a second machine, a restarted pump or an operator's manual run.
# ``mkdir`` is the primitive -- it is atomic on POSIX and needs no daemon.
#
# The holder is the *job*, not the submitter. The submitter is a short-lived
# process that cannot renew a lease, so release is wired into the detached job's
# own command line (see build_remote_command): it removes the lock directory
# after writing its exit code. That is what keeps the mutex correct when the
# local machine disappears mid-run.

def lock_dir_for(remote_root):
    """The mutex directory for a scratch root. One definition, used everywhere."""
    root = (remote_root or "").rstrip("/") or "."
    return "%s/%s" % (root, LOCK_DIRNAME)


def build_lock_command(remote_root, owner, stale_seconds):
    """One round trip that either takes the mutex or explains who has it.

    Four answers, and the distinction between the last three is the whole point:

    ``ACQUIRED``  it is ours.
    ``BUSY pid``  a live process holds it. Never break this one.
    ``STALE ...`` the holder is provably gone: either it recorded a pid that is
                  no longer alive, or it recorded none and the lock is older than
                  *stale_seconds*. Broken here, and the caller retries the whole
                  snippet rather than assuming it now owns it -- two racers can
                  both break a stale lock, but only one wins the next ``mkdir``.
    ``YOUNG age`` we cannot prove the owner is gone. Wait.

    TWO INDEPENDENT MECHANISMS, and the first is why the pid is recorded at all:

    * a **recorded pid that is dead** (the wrapper was SIGKILLed, the remote
      OOM-killer took it, the box was reset) is conclusive, so the lock is broken
      immediately -- no waiting;
    * **no pid recorded** (a crash in the sub-second window between ``mkdir`` and
      ``echo $$``, or a reboot) leaves nothing to test, so the lock is broken only
      once it is older than *stale_seconds*.

    Deliberately asymmetric where the evidence is weak. An ``acquired_at`` that is
    missing or unparseable counts as *now*, i.e. YOUNG, and a **live** pid is BUSY
    however old the lock is. Breaking a lock is the one operation that can put two
    gnina processes on one GPU; it happens only on proof, never on doubt. The
    residual is a stalled campaign if a pid is recycled onto an unrelated live
    process -- visible as endless "busy", cured by removing the directory by hand.
    """
    root = shlex.quote((remote_root or "").rstrip("/") or ".")
    lock = '"$ROOT"/%s' % LOCK_DIRNAME
    return (
        'ROOT=%s; '
        'mkdir -p "$ROOT"; '
        'if mkdir %s 2>/dev/null; then '
        'printf %%s\\\\n %s > %s/owner; '
        'date +%%s > %s/acquired_at; echo ACQUIRED; '
        'else '
        'pid=$(cat %s/pid 2>/dev/null || echo ""); '
        'if [ -n "$pid" ]; then '
        'if kill -0 "$pid" 2>/dev/null; then echo "BUSY $pid"; '
        'else rm -rf %s; echo "STALE dead_pid_$pid"; fi; '
        'else '
        'acq=$(cat %s/acquired_at 2>/dev/null || echo ""); '
        'case "$acq" in \'\'|*[!0-9]*) acq=$(date +%%s);; esac; '
        'age=$(( $(date +%%s) - acq )); '
        'if [ "$age" -gt %d ]; then rm -rf %s; echo "STALE $age"; '
        'else echo "YOUNG $age"; fi; '
        'fi; '
        'fi'
        % (root, lock, shlex.quote(str(owner)), lock, lock, lock, lock,
           lock, int(stale_seconds), lock)
    )


def build_unlock_command(remote_root):
    """Release the mutex. Used only for cleanup after a launch that never happened."""
    return "rm -rf %s" % shlex.quote(lock_dir_for(remote_root))


def parse_lock_result(stdout):
    """Classify :func:`build_lock_command` output. Never raises.

    Anything unrecognised is reported as BUSY. The failure modes are not
    symmetric: mistaking "held" for "free" puts two jobs on one GPU and corrupts
    both runs, while mistaking "free" for "held" costs one poll interval.
    """
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    if not lines:
        return "BUSY"
    head = lines[-1].split()[0].upper()
    if head in ("ACQUIRED", "BUSY", "STALE", "YOUNG"):
        return head
    return "BUSY"


def build_remote_command(remote_dir, r_name, l_name, passthrough, image, native,
                         lock_root=None):
    """The command run on the GPU host: detached, recording its own exit code.

    ``setsid nohup`` means the job outlives the SSH session, so an outage mid-run
    costs latency rather than the whole computation. The exit code lands in a
    marker file, which doubles as the completion signal we poll for.

    With *lock_root* (the remote scratch **root**, whose ``gpu.lock`` directory is
    the mutex) two statements are added *inside* the detached shell:

    * ``echo $$ > <lock>/pid`` first, so ``kill -0`` can tell a live holder from
      a dead one. ``$$`` is the detached shell, which lives exactly as long as
      the gnina run it wraps;
    * ``rm -rf <lock>`` last, **after** the exit code is written, so the next job
      can never start while this one's result is still being recorded.

    With ``lock_root=None`` the output is byte-identical to the pre-mutex shim.
    """
    if native:
        inner = [native, "-r", "%s/%s" % (remote_dir, r_name),
                 "-l", "%s/%s" % (remote_dir, l_name),
                 "-o", "%s/out.sdf" % remote_dir,
                 "--log", "%s/out.log" % remote_dir] + list(passthrough)
    else:
        inner = ["docker", "run", "--rm", "--gpus", "all",
                 "-v", "%s:/w" % remote_dir, image,
                 "gnina", "-r", "/w/%s" % r_name, "-l", "/w/%s" % l_name,
                 "-o", "/w/out.sdf", "--log", "/w/out.log"] + list(passthrough)
    payload = " ".join(shlex.quote(c) for c in inner)
    d = shlex.quote(remote_dir)
    script = "%s > %s/run.log 2>&1; echo $? > %s/exit_code" % (payload, d, d)
    if lock_root is not None:
        lock = shlex.quote(lock_dir_for(lock_root))
        script = "echo $$ > %s/pid; %s; rm -rf %s" % (lock, script, lock)
    return ("setsid nohup sh -c %s > /dev/null 2>&1 < /dev/null & echo launched"
            % shlex.quote(script))


def build_job_probe_command(remote_dir):
    """One round trip that distinguishes GONE / LAUNCHED / RUNNING / DONE."""
    d = shlex.quote(remote_dir)
    return (
        'if [ ! -d %s ]; then echo GONE; '
        'elif [ -f %s/exit_code ]; then echo "DONE $(cat %s/exit_code 2>/dev/null)"; '
        'elif [ -s %s/run.log ]; then echo RUNNING; '
        'else echo LAUNCHED; fi' % (d, d, d, d)
    )


def parse_job_state(text):
    """Parse :func:`build_job_probe_command` output. Never raises.

    Unrecognised output means "we do not know", which must not be read as "the
    job is gone" -- that would re-dock a job that is still running. It maps to
    LAUNCHED, the only answer whose consequence is to wait.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "LAUNCHED", None
    parts = lines[-1].split()
    head = parts[0].upper()
    if head == "DONE":
        code = None
        if len(parts) > 1:
            try:
                code = int(parts[1])
            except ValueError:
                code = None
        if code is None:
            # "DONE" with an unreadable marker: still running, rather than a
            # success we cannot substantiate.
            return "RUNNING", None
        return "DONE", code
    if head in ("GONE", "RUNNING", "LAUNCHED"):
        return head, None
    return "LAUNCHED", None


def probe_job_state(host, port, key, remote_dir, retries=3, ssh_runner=None):
    """Ask the remote what state a detached job is in.

    Returns ``(state, exit_code)`` with state in GONE / LAUNCHED / RUNNING /
    DONE; *exit_code* is set only for DONE. Raises :class:`RemoteGninaError` if
    the question could not be asked -- "I could not reach the host" and "the job
    is gone" are different answers and must not be collapsed.
    """
    runner = ssh_runner or run_resilient
    cmd = ssh_base(host, port, key) + [build_job_probe_command(remote_dir)]
    proc = runner(cmd, "probe %s" % remote_dir, retries, 180, None)
    return parse_job_state(getattr(proc, "stdout", "") or "")


# ---------------------------------------------------------------------------
# tickets
# ---------------------------------------------------------------------------

def write_ticket(path, payload):
    """Write a ticket atomically. The queue reads this to decide what happened.

    Same discipline as the queue's own records: a unique temp file in the target
    directory, fsync, then ``os.replace``. A reader can see the old ticket or the
    new one, never a half-written one -- and a torn ticket is exactly the input
    that would make the queue guess.
    """
    if not path:
        return
    body = dict(payload or {})
    body.setdefault("schema_version", 1)
    body.setdefault("at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, ".%s.tmp.%d.%d"
                       % (os.path.basename(path), os.getpid(), time.monotonic_ns()))
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(body, sort_keys=True, indent=1) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class _Ticket(object):
    """Accumulates the ticket for one run so every exit path writes exactly one."""

    def __init__(self, path, mode, argv):
        self.path = path
        self.body = {
            "schema_version": 1, "mode": mode, "outcome": None,
            "host": None, "port": None, "remote_dir": None, "job_key": None,
            "image": None, "lock": "not_required", "remote_pid_file": None,
            "exit_code": None, "out": None, "log": None,
            "argv": list(argv), "error": None,
        }

    def set(self, **fields):
        self.body.update(fields)

    def finish(self, outcome, **fields):
        self.body.update(fields)
        self.body["outcome"] = outcome
        self.body["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_ticket(self.path, self.body)


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------

class _Config(object):
    """Everything the three modes agree on, resolved once."""

    def __init__(self, known, passthrough, log):
        self.known = known
        self.passthrough = passthrough
        self.log = log
        self.host = os.environ.get("GNINA_REMOTE_HOST")
        self.port = os.environ.get("GNINA_REMOTE_PORT")
        self.key = os.environ.get("GNINA_REMOTE_KEY")
        self.image = os.environ.get("GNINA_REMOTE_IMAGE", DEFAULT_IMAGE)
        self.native = os.environ.get("GNINA_REMOTE_NATIVE")
        self.retries = int(os.environ.get("GNINA_REMOTE_RETRIES", DEFAULT_RETRIES))
        self.run_timeout = float(os.environ.get("GNINA_REMOTE_TIMEOUT",
                                                DEFAULT_RUN_TIMEOUT))
        self.keep_remote = os.environ.get("GNINA_REMOTE_KEEP") == "1"
        self.job_dir_override = os.environ.get("GNINA_REMOTE_JOB_DIR") or None
        self.lock_wait = float(os.environ.get("GNINA_REMOTE_LOCK_WAIT", "0") or 0)
        self.lock_stale = int(float(os.environ.get("GNINA_REMOTE_LOCK_STALE",
                                                   DEFAULT_LOCK_STALE) or
                                    DEFAULT_LOCK_STALE))
        remote_root = os.environ.get("GNINA_REMOTE_DIR", DEFAULT_REMOTE_DIR)
        # A quoted "~" is literal to the remote shell: shlex.quote("~/x") makes it
        # create a directory actually named "~". Keep the path relative (ssh resolves
        # it against $HOME) or absolute, never tilde-prefixed.
        if remote_root.startswith("~/"):
            remote_root = remote_root[2:]
        elif remote_root == "~":
            remote_root = ""
        self.remote_root = remote_root
        self.ssh = ssh_base(self.host, self.port, self.key)
        self.scp = scp_base(self.port, self.key)
        self.remote_dir = None
        self.job_key = None

    def resolve_remote_dir(self):
        """Absolute remote job directory, created if need be.

        Absolute matters: ``docker -v`` rejects a relative path (exit 125) and a
        tilde cannot be quoted through the remote shell.
        """
        if self.job_dir_override:
            # An explicit directory is authoritative and is NOT re-derived from a
            # hash: collection must not depend on the inputs still hashing to the
            # same value, or a finished result would become unreachable.
            self.remote_dir = self.job_dir_override
            self.job_key = posixpath.basename(self.remote_dir).replace("gnina_", "", 1)
            return self.remote_dir
        self.job_key = job_key(self.known.receptor, self.known.ligand,
                               self.passthrough)
        rel_dir = "%s/gnina_%s" % (self.remote_root.rstrip("/") or ".", self.job_key)
        made = run_resilient(
            self.ssh + ["mkdir -p %s && cd %s && pwd"
                        % (shlex.quote(rel_dir), shlex.quote(rel_dir))],
            "remote mkdir", self.retries, timeout=180, log=self.log)
        self.remote_dir = made.stdout.strip().splitlines()[-1]
        return self.remote_dir

    def absolute_root(self):
        """The scratch root that holds the mutex, as an absolute remote path.

        Derived from the resolved job directory so it costs no extra round trip
        and cannot disagree with it.
        """
        if self.remote_dir and self.remote_dir.startswith("/"):
            return posixpath.dirname(self.remote_dir)
        return self.remote_root.rstrip("/") or "."

    def prior_exit_code(self, quoted_dir):
        probe = run_resilient(
            self.ssh + ["cat %s/exit_code 2>/dev/null || echo NONE" % quoted_dir],
            "resume probe", self.retries, timeout=180, log=self.log)
        return (probe.stdout.strip().splitlines() or ["NONE"])[-1]

    def upload(self):
        run_resilient(self.scp + [self.known.receptor, self.known.ligand,
                                  "%s:%s/" % (self.host, self.remote_dir)],
                      "upload", self.retries, timeout=1800, log=self.log)

    def acquire_lock(self, owner):
        """Take the remote GPU mutex. Returns True, or False if someone has it.

        A broken lock is the only operation here that could double-book the GPU,
        so it is never silent: every break is reported with its age.
        """
        root = self.absolute_root()
        deadline = time.time() + max(0.0, self.lock_wait)
        while True:
            for _attempt in range(3):
                proc = run_resilient(
                    self.ssh + [build_lock_command(root, owner, self.lock_stale)],
                    "acquire gpu mutex", self.retries, timeout=180, log=self.log)
                result = parse_lock_result(proc.stdout)
                if result == "ACQUIRED":
                    return True
                if result == "STALE":
                    self.log("broke a stale GPU mutex on %s (%s); the previous "
                             "owner left no live process. Retrying the mkdir -- "
                             "if another submitter broke it too, only one of us "
                             "wins." % (self.host, proc.stdout.strip()))
                    continue
                break
            if time.time() >= deadline:
                return False
            time.sleep(min(5.0, max(0.5, deadline - time.time())))

    def stream_run_log(self, quoted_dir):
        """gnina's own output belongs to the caller verbatim.

        consensus_docker parses and logs it; hiding it would make a remote
        failure look like a local one.
        """
        try:
            out = run_resilient(self.ssh + ["cat %s/run.log 2>/dev/null" % quoted_dir],
                                "fetch run log", self.retries, timeout=300, log=self.log)
            sys.stdout.write(out.stdout)
        except RemoteGninaError:
            pass

    def download(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.known.out)) or ".",
                    exist_ok=True)
        run_resilient(self.scp + ["%s:%s/out.sdf" % (self.host, self.remote_dir),
                                  self.known.out],
                      "download out.sdf", self.retries, timeout=1800, log=self.log)
        if self.known.log:
            os.makedirs(os.path.dirname(os.path.abspath(self.known.log)) or ".",
                        exist_ok=True)
            try:
                run_resilient(self.scp + ["%s:%s/out.log" % (self.host, self.remote_dir),
                                          self.known.log],
                              "download out.log", self.retries, timeout=600,
                              log=self.log)
            except RemoteGninaError:
                pass  # advisory; out.sdf is the result
        if not os.path.exists(self.known.out) or os.path.getsize(self.known.out) == 0:
            raise RemoteGninaError(
                "out.sdf is missing or empty after download: %s. Treating this as a "
                "failure rather than letting an empty pose file look like a result."
                % self.known.out)

    def tail_failure(self, quoted_dir, code):
        try:
            tail = run_resilient(self.ssh + ["tail -40 %s/run.log 2>/dev/null" % quoted_dir],
                                 "fetch run log", self.retries, timeout=180, log=self.log)
            sys.stderr.write(tail.stdout)
        except RemoteGninaError:
            pass
        return RemoteGninaError("remote gnina exited %s" % code)


def _run_sync(cfg, ticket):
    """Block until the remote job finishes, then download it. The original path."""
    started = time.time()
    ok = False
    use_lock = os.environ.get("GNINA_REMOTE_LOCK") == "1"
    try:
        cfg.resolve_remote_dir()
        q = shlex.quote(cfg.remote_dir)
        ticket.set(remote_dir=cfg.remote_dir, job_key=cfg.job_key)

        prior = cfg.prior_exit_code(q)
        if prior == "0":
            cfg.log("found a completed run for these inputs -- collecting, not re-docking")
        else:
            if prior != "NONE":
                cfg.log("previous attempt exited %s; re-running" % prior)
                run_resilient(cfg.ssh + ["rm -f %s/exit_code %s/out.sdf" % (q, q)],
                              "clear stale markers", cfg.retries, timeout=180,
                              log=cfg.log)
            r_name = os.path.basename(cfg.known.receptor)
            l_name = os.path.basename(cfg.known.ligand)
            cfg.upload()
            lock_root = None
            if use_lock:
                if not cfg.acquire_lock("sync:%d" % os.getpid()):
                    cfg.log("the remote GPU mutex is held by another job; not "
                            "starting a second gnina on one GPU")
                    ticket.finish("busy", lock="busy")
                    return EXIT_LOCK_BUSY
                lock_root = cfg.absolute_root()
            run_resilient(
                cfg.ssh + [build_remote_command(cfg.remote_dir, r_name, l_name,
                                                cfg.passthrough, cfg.image,
                                                cfg.native, lock_root=lock_root)],
                "launch", cfg.retries, timeout=300, log=cfg.log)

            deadline = time.time() + cfg.run_timeout
            code = None
            while time.time() < deadline:
                time.sleep(POLL_INTERVAL)
                try:
                    poll = run_resilient(
                        cfg.ssh + ["cat %s/exit_code 2>/dev/null || echo RUNNING" % q],
                        "poll", cfg.retries, timeout=180, log=cfg.log)
                except TransientError as exc:
                    # The link is down; the detached job is not. Keep waiting.
                    cfg.log("poll unreachable (%s) -- job is detached, still waiting"
                            % exc)
                    continue
                text = (poll.stdout.strip().splitlines() or ["RUNNING"])[-1]
                if text != "RUNNING":
                    code = text
                    break
            if code is None:
                raise RemoteGninaError(
                    "remote job did not finish within %.0fs. It is detached and may "
                    "still be running; re-running this job will collect its result "
                    "from %s" % (cfg.run_timeout, cfg.remote_dir))
            if code != "0":
                raise cfg.tail_failure(q, code)

        cfg.stream_run_log(q)
        cfg.download()
        ok = True
        cfg.log("%s -> %d bytes in %.1fs" % (cfg.host, os.path.getsize(cfg.known.out),
                                             time.time() - started))
        ticket.finish("collected", exit_code=0)
        return EXIT_OK
    except RemoteGninaError as exc:
        cfg.log("%s" % exc)
        if cfg.remote_dir:
            # Deliberately kept: completed or partial work stays so the next
            # attempt at this job resumes instead of repeating it.
            cfg.log("remote job directory kept for resume: %s" % cfg.remote_dir)
        ticket.finish("error", error=str(exc))
        return EXIT_FAILED
    finally:
        if ok and cfg.remote_dir and not cfg.keep_remote:
            subprocess.run(cfg.ssh + ["rm -rf %s" % shlex.quote(cfg.remote_dir)],
                           capture_output=True, text=True)


def _run_submit(cfg, ticket):
    """Launch the job on the GPU and return immediately. Never downloads.

    The order is deliberate: upload happens **before** the mutex is taken. A
    submit that loses the race has still staged its inputs into the
    content-addressed directory, so the next attempt reuses them; holding the
    GPU during a 300 KB upload would idle it for nothing.
    """
    try:
        cfg.resolve_remote_dir()
        q = shlex.quote(cfg.remote_dir)
        ticket.set(remote_dir=cfg.remote_dir, job_key=cfg.job_key)

        prior = cfg.prior_exit_code(q)
        if prior == "0":
            cfg.log("a completed run for these inputs already exists on %s -- "
                    "nothing to launch; it will be collected" % cfg.host)
            ticket.finish("already_done", exit_code=0)
            return EXIT_SUBMITTED
        if prior != "NONE":
            cfg.log("previous attempt exited %s; re-running" % prior)
            run_resilient(cfg.ssh + ["rm -f %s/exit_code %s/out.sdf" % (q, q)],
                          "clear stale markers", cfg.retries, timeout=180, log=cfg.log)

        # Remove any local leftovers BEFORE launching. If an out.sdf from an
        # earlier attempt survived here, consensus_docker's split_mol would
        # succeed on those stale bytes and write a results.csv from the previous
        # parameter set -- a wrong score that nothing downstream could detect.
        for path in (cfg.known.out, cfg.known.log):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError as exc:
                    raise RemoteGninaError(
                        "could not remove the stale %s before submitting: %s. "
                        "Leaving it would let this job's results.csv be built "
                        "from the previous run's poses." % (path, exc))

        cfg.upload()

        owner = "%s:%s" % (os.environ.get("GNINA_REMOTE_OWNER") or "submit",
                           cfg.job_key)
        if not cfg.acquire_lock(owner):
            cfg.log("the remote GPU is busy; this job stays queued and is retried")
            ticket.finish("busy", lock="busy")
            return EXIT_LOCK_BUSY

        lock_root = cfg.absolute_root()
        r_name = os.path.basename(cfg.known.receptor)
        l_name = os.path.basename(cfg.known.ligand)
        try:
            run_resilient(
                cfg.ssh + [build_remote_command(cfg.remote_dir, r_name, l_name,
                                                cfg.passthrough, cfg.image,
                                                cfg.native, lock_root=lock_root)],
                "launch", cfg.retries, timeout=300, log=cfg.log)
        except RemoteGninaError:
            # We hold a mutex for a job that never started. Nothing else will
            # release it: the release is wired into the job's own command line.
            try:
                run_resilient(cfg.ssh + [build_unlock_command(lock_root)],
                              "release gpu mutex", cfg.retries, timeout=180,
                              log=cfg.log)
            except RemoteGninaError as release_exc:
                cfg.log("could not release the GPU mutex after a failed launch "
                        "(%s); it will be broken as stale in %ds"
                        % (release_exc, cfg.lock_stale))
            raise

        cfg.log("launched on %s in %s; not waiting" % (cfg.host, cfg.remote_dir))
        ticket.finish("launched", lock="acquired",
                      remote_pid_file="%s/pid" % lock_dir_for(lock_root))
        return EXIT_SUBMITTED
    except RemoteGninaError as exc:
        cfg.log("%s" % exc)
        ticket.finish("error", error=str(exc))
        return EXIT_FAILED


def _run_collect(cfg, ticket):
    """Download a finished remote result. Never launches, never takes the mutex.

    THE REMOTE DIRECTORY IS DELIBERATELY NOT DELETED HERE, even on success --
    unlike sync mode, which downloads and post-processes in one process and can
    safely clean up at the end. Here the post-processing happens after this
    process exits, and when it fails the retry policy re-runs the *collect*
    rather than the docking, which only works while the poses are still on the
    GPU box. Deleting them here would turn a five-second rdkit failure into
    another two and a half minutes of GPU time, and would also throw away the
    content-addressed resume that makes a re-run of the campaign free.

    The cost is roughly 1-2 MB per job left in the remote scratch root, which
    someone has to reclaim once the results are known good; the cost of the
    alternative is paid in GPU hours.
    """
    try:
        cfg.resolve_remote_dir()
        q = shlex.quote(cfg.remote_dir)
        ticket.set(remote_dir=cfg.remote_dir, job_key=cfg.job_key)

        state, code = probe_job_state(cfg.host, cfg.port, cfg.key, cfg.remote_dir,
                                      retries=cfg.retries)
        if state == "GONE":
            cfg.log("the remote job directory %s no longer exists; there is "
                    "nothing to collect and the job has to be docked again"
                    % cfg.remote_dir)
            ticket.finish("gone")
            return EXIT_NOT_READY
        if state != "DONE":
            cfg.log("the remote job has not finished yet (%s); nothing to collect"
                    % state.lower())
            ticket.finish("not_ready")
            return EXIT_NOT_READY
        if code != 0:
            raise cfg.tail_failure(q, code)

        cfg.stream_run_log(q)
        cfg.download()
        cfg.log("%s -> %d bytes collected from %s"
                % (cfg.host, os.path.getsize(cfg.known.out), cfg.remote_dir))
        ticket.finish("collected", exit_code=0)
        return EXIT_OK
    except RemoteGninaError as exc:
        cfg.log("%s" % exc)
        if cfg.remote_dir:
            cfg.log("remote job directory kept for resume: %s" % cfg.remote_dir)
        ticket.finish("error", error=str(exc))
        return EXIT_FAILED


#: Modes this build implements. --capabilities prints them so a caller can check
#: what the shim can actually DO rather than what it is called: the filename test
#: it replaces passed against a build that had no submit/collect mode at all,
#: which is precisely the failure the check exists to catch.
CAPABILITIES = ("sync", "submit", "collect")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--capabilities" in argv:
        sys.stdout.write("gnina_remote modes: %s\n" % " ".join(CAPABILITIES))
        return 0
    known, passthrough = parse_args(argv)

    def log(msg):
        sys.stderr.write("gnina_remote: %s\n" % msg)
        sys.stderr.flush()

    mode = (os.environ.get("GNINA_REMOTE_MODE") or MODE_SYNC).strip().lower()
    if mode not in MODES:
        log("GNINA_REMOTE_MODE=%r is not one of %s. Refusing to guess: submit and "
            "collect do very different things and picking the wrong one either "
            "double-books the GPU or throws away a finished result."
            % (mode, "/".join(MODES)))
        return EXIT_USAGE

    ticket_path = os.environ.get("GNINA_REMOTE_TICKET")
    if not ticket_path and known.out and mode != MODE_SYNC:
        ticket_path = os.path.join(
            os.path.dirname(os.path.abspath(known.out)) or ".",
            ".gnina_remote_ticket.json")
    ticket = _Ticket(ticket_path, mode, argv)

    host = os.environ.get("GNINA_REMOTE_HOST")
    if not host:
        log("GNINA_REMOTE_HOST is not set. This is a stand-in for the gnina binary "
            "that runs it on a remote GPU host; silently falling back to a local "
            "gnina would hide where the CNN scores came from.")
        return EXIT_USAGE

    for flag, path in (("-r/--receptor", known.receptor), ("-l/--ligand", known.ligand)):
        if not path:
            log("%s is required" % flag)
            return EXIT_USAGE
        if not os.path.exists(path):
            log("%s not found: %s" % (flag, path))
            return EXIT_USAGE
    if not known.out:
        log("-o/--out is required")
        return EXIT_USAGE

    cfg = _Config(known, passthrough, log)
    ticket.set(host=cfg.host, port=cfg.port, image=cfg.image,
               out=os.path.abspath(known.out),
               log=os.path.abspath(known.log) if known.log else None)

    if mode == MODE_SUBMIT:
        return _run_submit(cfg, ticket)
    if mode == MODE_COLLECT:
        return _run_collect(cfg, ticket)
    return _run_sync(cfg, ticket)


if __name__ == "__main__":
    sys.exit(main())
