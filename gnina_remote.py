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

Authentication is SSH keys only. Passwords are deliberately unsupported: this runs
unattended once per job, and the alternatives (``sshpass``, a stored secret) put
the password in the process list or on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
import time

DEFAULT_IMAGE = "gnina/gnina:v1.3.3"
DEFAULT_REMOTE_DIR = "gnina_jobs"
DEFAULT_RETRIES = 5
DEFAULT_RUN_TIMEOUT = 7200      # seconds to wait for the detached remote job
POLL_INTERVAL = 10              # seconds between completion checks

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


def build_remote_command(remote_dir, r_name, l_name, passthrough, image, native):
    """The command run on the GPU host: detached, recording its own exit code.

    ``setsid nohup`` means the job outlives the SSH session, so an outage mid-run
    costs latency rather than the whole computation. The exit code lands in a
    marker file, which doubles as the completion signal we poll for.
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
    return ("setsid nohup sh -c %s > /dev/null 2>&1 < /dev/null & echo launched"
            % shlex.quote(script))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    known, passthrough = parse_args(argv)

    def log(msg):
        sys.stderr.write("gnina_remote: %s\n" % msg)
        sys.stderr.flush()

    host = os.environ.get("GNINA_REMOTE_HOST")
    if not host:
        log("GNINA_REMOTE_HOST is not set. This is a stand-in for the gnina binary "
            "that runs it on a remote GPU host; silently falling back to a local "
            "gnina would hide where the CNN scores came from.")
        return 2
    port = os.environ.get("GNINA_REMOTE_PORT")
    key = os.environ.get("GNINA_REMOTE_KEY")
    image = os.environ.get("GNINA_REMOTE_IMAGE", DEFAULT_IMAGE)
    native = os.environ.get("GNINA_REMOTE_NATIVE")
    retries = int(os.environ.get("GNINA_REMOTE_RETRIES", DEFAULT_RETRIES))
    run_timeout = float(os.environ.get("GNINA_REMOTE_TIMEOUT", DEFAULT_RUN_TIMEOUT))
    keep_remote = os.environ.get("GNINA_REMOTE_KEEP") == "1"

    remote_root = os.environ.get("GNINA_REMOTE_DIR", DEFAULT_REMOTE_DIR)
    # A quoted "~" is literal to the remote shell: shlex.quote("~/x") makes it
    # create a directory actually named "~". Keep the path relative (ssh resolves
    # it against $HOME) or absolute, never tilde-prefixed.
    if remote_root.startswith("~/"):
        remote_root = remote_root[2:]
    elif remote_root == "~":
        remote_root = ""

    for flag, path in (("-r/--receptor", known.receptor), ("-l/--ligand", known.ligand)):
        if not path:
            log("%s is required" % flag)
            return 2
        if not os.path.exists(path):
            log("%s not found: %s" % (flag, path))
            return 2
    if not known.out:
        log("-o/--out is required")
        return 2

    ssh = ssh_base(host, port, key)
    scp = scp_base(port, key)
    rel_dir = "%s/gnina_%s" % (remote_root.rstrip("/") or ".",
                              job_key(known.receptor, known.ligand, passthrough))
    started = time.time()
    remote_dir = None
    ok = False

    try:
        # Resolve to an ABSOLUTE remote path: `docker -v` rejects a relative one
        # (exit 125), and the tilde cannot be quoted through the remote shell.
        made = run_resilient(
            ssh + ["mkdir -p %s && cd %s && pwd"
                   % (shlex.quote(rel_dir), shlex.quote(rel_dir))],
            "remote mkdir", retries, timeout=180, log=log)
        remote_dir = made.stdout.strip().splitlines()[-1]
        q = shlex.quote(remote_dir)

        # Resume: a previous attempt for these exact inputs may already have run.
        probe = run_resilient(
            ssh + ["cat %s/exit_code 2>/dev/null || echo NONE" % q],
            "resume probe", retries, timeout=180, log=log)
        prior = (probe.stdout.strip().splitlines() or ["NONE"])[-1]

        if prior == "0":
            log("found a completed run for these inputs -- collecting, not re-docking")
        else:
            if prior != "NONE":
                log("previous attempt exited %s; re-running" % prior)
                run_resilient(ssh + ["rm -f %s/exit_code %s/out.sdf" % (q, q)],
                              "clear stale markers", retries, timeout=180, log=log)

            r_name = os.path.basename(known.receptor)
            l_name = os.path.basename(known.ligand)
            run_resilient(scp + [known.receptor, known.ligand,
                                 "%s:%s/" % (host, remote_dir)],
                          "upload", retries, timeout=1800, log=log)
            run_resilient(
                ssh + [build_remote_command(remote_dir, r_name, l_name,
                                            passthrough, image, native)],
                "launch", retries, timeout=300, log=log)

            deadline = time.time() + run_timeout
            code = None
            while time.time() < deadline:
                time.sleep(POLL_INTERVAL)
                try:
                    poll = run_resilient(
                        ssh + ["cat %s/exit_code 2>/dev/null || echo RUNNING" % q],
                        "poll", retries, timeout=180, log=log)
                except TransientError as exc:
                    # The link is down; the detached job is not. Keep waiting.
                    log("poll unreachable (%s) -- job is detached, still waiting" % exc)
                    continue
                text = (poll.stdout.strip().splitlines() or ["RUNNING"])[-1]
                if text != "RUNNING":
                    code = text
                    break
            if code is None:
                raise RemoteGninaError(
                    "remote job did not finish within %.0fs. It is detached and may "
                    "still be running; re-running this job will collect its result "
                    "from %s" % (run_timeout, remote_dir))
            if code != "0":
                try:
                    tail = run_resilient(ssh + ["tail -40 %s/run.log 2>/dev/null" % q],
                                         "fetch run log", retries, timeout=180, log=log)
                    sys.stderr.write(tail.stdout)
                except RemoteGninaError:
                    pass
                raise RemoteGninaError("remote gnina exited %s" % code)

        # gnina's own output belongs to the caller verbatim: consensus_docker parses
        # and logs it, and hiding it would make a remote failure look local.
        try:
            out = run_resilient(ssh + ["cat %s/run.log 2>/dev/null" % q],
                                "fetch run log", retries, timeout=300, log=log)
            sys.stdout.write(out.stdout)
        except RemoteGninaError:
            pass

        os.makedirs(os.path.dirname(os.path.abspath(known.out)) or ".", exist_ok=True)
        run_resilient(scp + ["%s:%s/out.sdf" % (host, remote_dir), known.out],
                      "download out.sdf", retries, timeout=1800, log=log)
        if known.log:
            os.makedirs(os.path.dirname(os.path.abspath(known.log)) or ".", exist_ok=True)
            try:
                run_resilient(scp + ["%s:%s/out.log" % (host, remote_dir), known.log],
                              "download out.log", retries, timeout=600, log=log)
            except RemoteGninaError:
                pass  # advisory; out.sdf is the result

        if not os.path.exists(known.out) or os.path.getsize(known.out) == 0:
            raise RemoteGninaError(
                "out.sdf is missing or empty after download: %s. Treating this as a "
                "failure rather than letting an empty pose file look like a result."
                % known.out)

        ok = True
        log("%s -> %d bytes in %.1fs" % (host, os.path.getsize(known.out),
                                         time.time() - started))
        return 0

    except RemoteGninaError as exc:
        log("%s" % exc)
        if remote_dir:
            # Deliberately kept: completed or partial work stays so the next
            # attempt at this job resumes instead of repeating it.
            log("remote job directory kept for resume: %s" % remote_dir)
        return 1
    finally:
        if ok and remote_dir and not keep_remote:
            subprocess.run(ssh + ["rm -rf %s" % shlex.quote(remote_dir)],
                           capture_output=True, text=True)


if __name__ == "__main__":
    sys.exit(main())
