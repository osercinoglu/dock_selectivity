#!/usr/bin/env python3
"""Run gnina on a Google Colab GPU runtime, as a second backend for ``gnina_remote.py``.

Why this exists
---------------

``gnina_remote.py`` ran gnina on one SSH-reachable host. When that host is busy --
it has a single GPU, and somebody else is using it -- the whole gnina stage
stalls. A Colab Pro+ subscription is a second GPU that is already paid for, so
this module makes it a drop-in alternative selected with
``GNINA_REMOTE_BACKEND=colab``.

What drives it
--------------

The ``colab`` CLI (``uv tool install google-colab-cli``), *not* ``gcloud``.
``gcloud colab`` manages **Colab Enterprise**, a Vertex AI product billed to a
GCP project; it does not touch a consumer Pro/Pro+ subscription's compute units.
The two are unrelated services with similar names. This module shells out to
``colab``, which authenticates as the subscriber and spends their compute units.

How gnina gets onto the runtime
-------------------------------

Colab has no Docker, so the container the SSH backend uses is not an option.
Instead the release binary is fetched **on the runtime** (2.06 GB at ~74 MB/s
from Google's network, about 28 s -- far faster than uploading 2 GB from a home
connection).

It is called ``gnina.cuda12.8.static`` but is not fully static: ``ldd`` reports
``libcudart.so.12``, ``libcufft.so.11``, ``libcublas.so.12`` and
``libcublasLt.so.12`` missing, because current Colab images ship CUDA 13. The
CUDA 12 libraries are nonetheless present as pip wheels
(``nvidia-cuda-runtime-cu12`` and friends), so :func:`gnina_install_source`
locates each missing soname and composes an ``LD_LIBRARY_PATH``. That is
discovered at run time rather than hardcoded, because the paths carry the
runtime's Python version (``/usr/local/lib/python3.13/dist-packages/...``) and
Colab upgrades it without notice.

Why the job is detached
-----------------------

Exactly as in the SSH backend: ``setsid nohup`` with the exit code written to a
marker file. A docking run is minutes long, and neither a dropped websocket nor
an interrupted local process may kill it or orphan it invisibly. Polling the
marker is what makes a lost connection survivable and what makes resume free.

Billing is the thing to get right
---------------------------------

A Colab runtime bills for wall-clock time it is *allocated*, not time it is
computing: a measured 1.07 compute units per hour for a T4. An idle session
therefore costs exactly as much as a busy one, which is why
:func:`stop_session` exists and why the caller is expected to use it.
"""

from __future__ import annotations

import json
import logging
import os
import posixpath
import shlex
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence

LOG = logging.getLogger("colab_backend")

#: The release asset. Not actually static -- see the module docstring.
GNINA_RELEASE_URL = (
    "https://github.com/gnina/gnina/releases/download/v1.3.3/gnina.cuda12.8.static"
)

#: Where things live on the runtime. ``/content`` is the Colab working directory.
REMOTE_BINARY = "/content/gnina"
REMOTE_ENV_FILE = "/content/gnina_env.json"
DEFAULT_REMOTE_ROOT = "/content/gnina_jobs"

#: Accelerators the CLI accepts. An unrecognised value silently falls back to
#: A100, which then fails allocation on most accounts, so the value is checked
#: here instead of being passed through.
GPUS = ("T4", "L4", "G4", "A100", "H100")

DEFAULT_SESSION = "gnina"
DEFAULT_GPU = "T4"
#: Core-seconds per exhaustiveness unit, measured on a Colab T4's 2.00 GHz Xeon
#: vCPUs. ``remote_caps`` carries 112.5, calibrated on the SSH host's 24 faster
#: cores; on this hardware that under-predicts by about 3x, and an
#: under-predicting deadline model is the dangerous direction -- it lets a job
#: through and then misses the deadline, instead of refusing up front.
#: Override with GNINA_COLAB_CORE_SECONDS after measuring another accelerator.
COLAB_CORE_SECONDS_PER_EXH = float(
    os.environ.get("GNINA_COLAB_CORE_SECONDS") or 344.0)

DEFAULT_RETRIES = 4
DEFAULT_POLL_INTERVAL = 10.0
DEFAULT_RUN_TIMEOUT = 7200.0

#: Marks a runtime as already carrying a working gnina, so a second job in the
#: same session skips the 2 GB download.
READY_MARKER = "/content/.gnina_ready"

#: Substrings in a failed ``colab`` invocation that mean "try again" rather than
#: "this will never work". A pruned session is transient in the useful sense:
#: re-creating it is the fix, and the caller's retry loop does exactly that.
_TRANSIENT_MARKERS = (
    "session not found",
    "connection reset",
    "connection aborted",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "502",
    "503",
    "504",
    "429",
    "websocket",
    "remote end closed",
)

#: A missing CLI is not transient and the message says what to do about it.
_CLI_MISSING = (
    "the 'colab' CLI is not on PATH. Install it with "
    "'uv tool install google-colab-cli' (or 'pip install google-colab-cli'); "
    "note that 'gcloud colab' is a different product (Colab Enterprise) and "
    "cannot spend a Pro+ subscription's compute units."
)


class ColabError(RuntimeError):
    """A Colab operation failed in a way the caller must not paper over."""


class ColabTransientError(ColabError):
    """A failure worth retrying: a dropped socket, a pruned session, a 503."""


class ColabUnavailable(ColabError):
    """The backend cannot be used at all: no CLI, no auth, no GPU entitlement."""


class ColabRefused(ColabError):
    """The operator asked for something this runtime cannot do.

    Distinct from :class:`ColabError` because nothing ran: there is no partial
    work to resume and retrying changes nothing. The caller reports it as a usage
    error rather than a failed job.
    """


def is_transient(returncode: int, output: str) -> bool:
    """Whether a failed ``colab`` call should be retried.

    Unknown failures are treated as permanent. Retrying a genuine error wastes
    an allocated GPU's billed minutes, and silently retrying a configuration
    mistake hides it.
    """
    if returncode == 0:
        return False
    haystack = (output or "").lower()
    return any(marker in haystack for marker in _TRANSIENT_MARKERS)


class ColabSession(object):
    """One named Colab runtime, driven through the ``colab`` CLI.

    Reads its configuration from the environment so that ``gnina_remote.py`` can
    hand control over without threading a dozen arguments through three modes:

    ========================================  ====================================
    ``GNINA_COLAB_SESSION``                   session name (default ``gnina``)
    ``GNINA_COLAB_GPU``                       ``T4`` / ``L4`` / ``G4`` / ``A100`` / ``H100``
    ``GNINA_COLAB_CONFIG``                    CLI state file, for isolating runs
    ``GNINA_COLAB_AUTH``                      ``adc`` (default) or ``oauth2``
    ``GNINA_COLAB_BINARY_URL``                override the gnina release asset
    ``GNINA_COLAB_ROOT``                      job root (default ``/content/gnina_jobs``)
    ``GNINA_COLAB_KEEP``                      ``1`` leaves the job directory behind
    ========================================  ====================================
    """

    def __init__(self,
                 session: Optional[str] = None,
                 gpu: Optional[str] = None,
                 config_path: Optional[str] = None,
                 auth: Optional[str] = None,
                 retries: int = DEFAULT_RETRIES,
                 log: Optional[Any] = None) -> None:
        self.session = session or os.environ.get("GNINA_COLAB_SESSION", DEFAULT_SESSION)
        self.gpu = (gpu or os.environ.get("GNINA_COLAB_GPU", DEFAULT_GPU) or "").upper()
        if self.gpu and self.gpu not in GPUS:
            raise ColabError(
                "GNINA_COLAB_GPU=%r is not one of %s. The CLI silently falls back "
                "to A100 for an unrecognised value, which then fails to allocate "
                "on most accounts." % (self.gpu, ", ".join(GPUS)))
        self.config_path = config_path or os.environ.get("GNINA_COLAB_CONFIG") or None
        self.auth = auth or os.environ.get("GNINA_COLAB_AUTH", "adc")
        self.retries = int(retries)
        self.log = log or (lambda message: LOG.info("%s", message))
        self.binary_url = os.environ.get("GNINA_COLAB_BINARY_URL", GNINA_RELEASE_URL)
        self.remote_root = os.environ.get("GNINA_COLAB_ROOT", DEFAULT_REMOTE_ROOT)

    # -- the CLI ----------------------------------------------------------

    def base(self) -> List[str]:
        """The invariant prefix of every ``colab`` invocation.

        ``--auth`` and ``--config`` are global flags and must precede the
        subcommand; the CLI rejects them afterwards.
        """
        command = ["colab", "--auth=%s" % self.auth]
        if self.config_path:
            command += ["--config", self.config_path]
        return command

    def run(self, args: Sequence[str], what: str, timeout: Optional[float] = 300.0,
            check: bool = True) -> subprocess.CompletedProcess:
        """Run one ``colab`` subcommand, retrying only transient failures."""
        command = self.base() + list(args)
        attempt = 0
        while True:
            attempt += 1
            try:
                proc = subprocess.run(command, capture_output=True, text=True,
                                      timeout=timeout)
            except FileNotFoundError:
                raise ColabUnavailable(_CLI_MISSING) from None
            except subprocess.TimeoutExpired:
                if attempt > self.retries:
                    raise ColabTransientError(
                        "%s timed out after %.0fs on attempt %d"
                        % (what, timeout or 0, attempt))
                self.log("%s timed out; retry %d/%d" % (what, attempt, self.retries))
                time.sleep(min(2 ** attempt, 30))
                continue

            if proc.returncode == 0 or not check:
                return proc

            output = (proc.stderr or "") + (proc.stdout or "")
            if is_transient(proc.returncode, output) and attempt <= self.retries:
                self.log("%s failed transiently (%s); retry %d/%d"
                         % (what, output.strip().splitlines()[-1:] or "?",
                            attempt, self.retries))
                time.sleep(min(2 ** attempt, 30))
                continue
            raise ColabError("%s failed (exit %d): %s"
                             % (what, proc.returncode, output.strip()[-800:]))

    # -- session lifecycle -------------------------------------------------

    def exists(self) -> bool:
        """Whether the backend believes this session is live.

        Asks the server rather than the local state file: the backend prunes
        runtimes on its own schedule, and a stale local entry is exactly how a
        job would be submitted into nothing.
        """
        proc = self.run(["status", "-s", self.session], "session status",
                        timeout=120, check=False)
        text = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            return False
        return self.session in text and "not found" not in text.lower()

    def ensure(self) -> bool:
        """Create the session if it is not already up. Returns True if created."""
        if self.exists():
            self.log("colab session %r already up" % self.session)
            return False
        args = ["new", "-s", self.session]
        if self.gpu:
            args += ["--gpu", self.gpu]
        started = time.time()
        self.run(args, "colab new", timeout=900)
        self.log("colab session %r allocated (%s) in %.1fs"
                 % (self.session, self.gpu or "CPU", time.time() - started))
        return True

    def stop(self) -> None:
        """Release the runtime. Never raises: a failed stop must not fail a job.

        It is still reported at WARNING, because the consequence of a silently
        failed stop is a GPU billing by the hour against the subscription.
        """
        try:
            self.run(["stop", "-s", self.session], "colab stop", timeout=300,
                     check=False)
            self.log("colab session %r stopped" % self.session)
        except ColabError as exc:  # pragma: no cover - defensive
            LOG.warning("could not stop colab session %r: %s -- it may still be "
                        "allocated and billing", self.session, exc)

    def usage(self) -> Dict[str, Any]:
        """Compute-unit balance and burn rate, as the CLI reports them."""
        proc = self.run(["usage"], "colab usage", timeout=120, check=False)
        text = (proc.stdout or "") + (proc.stderr or "")
        out: Dict[str, Any] = {"raw": text.strip()}
        for line in text.splitlines():
            low = line.lower()
            if "balance" in low:
                out["balance"] = _first_float(line)
            elif "rate" in low:
                out["rate_per_hour"] = _first_float(line)
            elif "active" in low:
                out["active"] = _first_float(line)
        return out

    # -- files and code ----------------------------------------------------

    def upload(self, local: str, remote: str) -> None:
        if not os.path.exists(local):
            raise ColabError("cannot upload %s: it does not exist" % local)
        self.run(["upload", "-s", self.session, local, remote],
                 "upload %s" % os.path.basename(local), timeout=1800)

    def download(self, remote: str, local: str) -> None:
        parent = os.path.dirname(os.path.abspath(local))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.run(["download", "-s", self.session, remote, local],
                 "download %s" % posixpath.basename(remote), timeout=1800)

    def exec_python(self, code: str, what: str = "exec",
                    timeout: Optional[float] = 900.0) -> str:
        """Run Python on the runtime and return its stdout.

        The code goes in through stdin rather than a temporary file: it keeps the
        caller free of cleanup, and the CLI documents piped stdin as equivalent
        to ``-f``.
        """
        command = self.base() + ["exec", "-s", self.session]
        attempt = 0
        while True:
            attempt += 1
            try:
                proc = subprocess.run(command, input=code, capture_output=True,
                                      text=True, timeout=timeout)
            except FileNotFoundError:
                raise ColabUnavailable(_CLI_MISSING) from None
            except subprocess.TimeoutExpired:
                raise ColabTransientError("%s timed out after %.0fs"
                                          % (what, timeout or 0))
            if proc.returncode == 0:
                return proc.stdout
            output = (proc.stderr or "") + (proc.stdout or "")
            if is_transient(proc.returncode, output) and attempt <= self.retries:
                self.log("%s failed transiently; retry %d/%d"
                         % (what, attempt, self.retries))
                time.sleep(min(2 ** attempt, 30))
                continue
            raise ColabError("%s failed (exit %d): %s"
                             % (what, proc.returncode, output.strip()[-800:]))

    # -- gnina on the runtime ---------------------------------------------

    def ensure_gnina(self) -> Dict[str, Any]:
        """Put a working gnina on the runtime and return how it was reached.

        Idempotent: a marker file records success, so the second job in a session
        pays nothing. Verified by actually running ``gnina --version`` -- a
        downloaded file that cannot execute is not an installation.
        """
        report = self.exec_python(
            gnina_install_source(self.binary_url), "install gnina", timeout=1800)
        payload = last_json(report)
        if not payload or not payload.get("ok"):
            raise ColabError(
                "gnina is not runnable on the Colab runtime: %s"
                % (payload.get("error") if payload else report.strip()[-600:]))
        self.log("gnina ready on %r: %s (%s)"
                 % (self.session, payload.get("version", "?"),
                    "cached" if payload.get("cached") else "installed in %.1fs"
                    % payload.get("install_seconds", 0.0)))
        return payload


def _fill(template: str, **values: str) -> str:
    """Substitute ``@@NAME@@`` markers in a remote snippet.

    Deliberately not ``%``- or ``str.format``-based: the snippets themselves
    contain ``%s`` and ``{}``, so either mechanism would consume them and raise
    ``not enough arguments for format string`` at the worst possible moment --
    the one where a docking job is already running. Same trap as the unescaped
    ``%`` that once broke ``--help`` in this repository.
    """
    out = template
    for name, value in values.items():
        marker = "@@%s@@" % name
        if marker not in out:
            raise ValueError("snippet has no %s to fill" % marker)
        out = out.replace(marker, value)
    leftover = [tok for tok in out.split("@@") if tok.isupper() and tok.isidentifier()]
    if leftover:
        raise ValueError("unfilled snippet placeholders: %s" % ", ".join(leftover))
    return out


def fit_to_runtime(passthrough: Sequence[str], cores: int,
                   deadline_seconds: float = DEFAULT_RUN_TIMEOUT,
                   log: Optional[Any] = None) -> List[str]:
    """Adapt ``--cpu`` to the runtime, and refuse an exhaustiveness that cannot fit.

    The same two rules the SSH backend applies, for the same reason, because a
    Colab runtime is the more constrained machine of the two: it hands out a GPU
    with only a couple of vCPUs, and gnina's search is CPU-parallel -- only the
    CNN rescoring runs on the card.

    * ``--cpu`` is **clamped, loudly**. ``consensus_docker.py`` passes ``--cpu
      60``; asking 60 threads of a 2-vCPU runtime does not go faster, it
      thrashes. The clamp names both numbers so a log never leaves it ambiguous
      which thread count produced a result.
    * ``exhaustiveness`` is **never clamped -- it is refused**. It changes what
      the docking *is*, so quietly lowering it would make two jobs in one
      campaign incomparable while both look fine. ``remote_caps`` already owns
      this policy and its calibration (112.5 core-seconds per exhaustiveness
      unit, measured); this reuses it rather than inventing a second model.
    """
    say = log or (lambda message: LOG.warning("%s", message))
    cores = max(1, int(cores or 1))
    out: List[str] = []
    index = 0
    requested_exh = None
    while index < len(passthrough):
        token = passthrough[index]
        value = passthrough[index + 1] if index + 1 < len(passthrough) else None
        if token in ("--cpu", "--num_threads") and value is not None:
            try:
                asked = int(float(value))
            except ValueError:
                out += [token, value]
                index += 2
                continue
            if asked > cores:
                say("clamping %s %d -> %d: this Colab runtime has %d vCPU(s). "
                    "The thread count changes speed, not the result."
                    % (token, asked, cores, cores))
                out += [token, str(cores)]
            else:
                out += [token, value]
            index += 2
            continue
        if token == "--exhaustiveness" and value is not None:
            try:
                requested_exh = int(float(value))
            except ValueError:
                requested_exh = None
            out += [token, value]
            index += 2
            continue
        out.append(token)
        index += 1

    if requested_exh:
        # The shape of the model is remote_caps'; only the constant differs, so
        # the two cannot drift apart in their reasoning about deadlines.
        core_seconds = float(os.environ.get("GNINA_COLAB_CORE_SECONDS")
                             or COLAB_CORE_SECONDS_PER_EXH)
        predicted = core_seconds * float(requested_exh) / max(1, cores)
        if predicted > deadline_seconds:
            ceiling = int(deadline_seconds * cores / core_seconds)
            while ceiling > 0 and (core_seconds * ceiling / max(1, cores)
                                   > deadline_seconds):
                ceiling -= 1
            raise ColabRefused(
                "--exhaustiveness %d on %d vCPU(s) is predicted to take %.0fs, "
                "past the %.0fs deadline. Exhaustiveness is never clamped "
                "silently: it changes what the docking is, and a campaign whose "
                "jobs used different values is not comparable. Either lower it "
                "to <= %d, raise GNINA_REMOTE_TIMEOUT, or dock this job on a "
                "host with more cores."
                % (requested_exh, cores, predicted, deadline_seconds, ceiling))
        say("exhaustiveness %d on %d vCPU(s): predicted %.0fs (%.1f min) per job"
            % (requested_exh, cores, predicted, predicted / 60.0))
    return out


def _first_float(line: str) -> Optional[float]:
    for token in line.replace(",", " ").replace("/hr", " ").split():
        try:
            return float(token.strip(":"))
        except ValueError:
            continue
    return None


def last_json(text: str) -> Optional[Dict[str, Any]]:
    """The last JSON object printed by a remote snippet.

    The runtime prints progress as well as its result, and ``colab exec``
    interleaves kernel chatter, so the contract is "the result is the last line
    that parses as a JSON object".
    """
    for line in reversed((text or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


def gnina_install_source(url: str = GNINA_RELEASE_URL) -> str:
    """Python to run on the runtime that installs gnina and proves it works.

    Returned as source rather than executed here because it runs *there*. It
    prints one JSON object as its last line; see :func:`last_json`.
    """
    return _fill(_INSTALL_TEMPLATE, URL=repr(url), BINARY=repr(REMOTE_BINARY),
                 ENV_FILE=repr(REMOTE_ENV_FILE), MARKER=repr(READY_MARKER))


_INSTALL_TEMPLATE = r'''
import glob, json, os, subprocess, sys, time

URL = @@URL@@
BINARY = @@BINARY@@
ENV_FILE = @@ENV_FILE@@
MARKER = @@MARKER@@

# Sonames the release binary needs that current Colab images do not provide at
# system level, because they ship CUDA 13. All exist as pip wheels; the search
# finds them rather than hardcoding a path that carries the Python version.
#
# LINKED are what ldd reports missing. DLOPENED are loaded at RUN time by the
# libtorch gnina bundles, so ldd never mentions them -- which is exactly how
# libnvrtc.so.12 got missed: gnina --version passed, and the job then aborted 23
# minutes later with "Error in dlopen for library libnvrtc.so.12" once CNN
# scoring started. Anything loaded lazily has to be checked explicitly.
LINKED = ["libcudart.so.12", "libcufft.so.11", "libcublas.so.12", "libcublasLt.so.12"]
DLOPENED = ["libnvrtc.so.12", "libcudnn.so.9", "libcurand.so.10",
            "libcusparse.so.12", "libcusolver.so.11"]
ROOTS = ["/usr/local/lib/python3*/dist-packages/nvidia",
         "/usr/local/cuda-12*/targets/x86_64-linux/lib",
         "/usr/lib/x86_64-linux-gnu"]
#: Excluded from the search: it holds the CUDA 13 build of several of the same
#: sonames, and picking one of those back up defeats the whole exercise.
EXCLUDE = "/cu13/"


def library_path():
    # Seeded with whatever the runtime already set -- /usr/lib64-nvidia, which
    # holds libcuda.so.1, the NVIDIA DRIVER library. Replacing LD_LIBRARY_PATH
    # instead of extending it makes gnina load, find no driver, print "No GPU
    # detected" and finish on CPU: 12x slower, on a runtime billed for its GPU.
    dirs = [d for d in (os.environ.get("LD_LIBRARY_PATH") or "").split(":") if d]
    missing = []
    for soname in LINKED + DLOPENED:
        found = None
        for root in ROOTS:
            matches = [m for m in glob.glob(root + "/**/" + soname, recursive=True)
                       if EXCLUDE not in m]
            if matches:
                found = os.path.dirname(matches[0])
                break
        if found is None:
            missing.append(soname)
        elif found not in dirs:
            # Prepended: the CUDA 12 wheels must win over any CUDA 13 copy
            # already on the path.
            dirs.insert(0, found)
    return ":".join(dirs), missing


def report(**kw):
    print(json.dumps(kw))


cached = os.path.exists(MARKER) and os.path.exists(BINARY)
install_seconds = 0.0
if not cached:
    t0 = time.time()
    os.makedirs(os.path.dirname(BINARY) or "/content", exist_ok=True)
    fetch = subprocess.run(["curl", "-sSL", "--retry", "3", "-o", BINARY, URL],
                           capture_output=True, text=True)
    if fetch.returncode != 0:
        report(ok=False, error="download failed: " + (fetch.stderr or "")[-400:])
        raise SystemExit(0)
    os.chmod(BINARY, 0o755)
    install_seconds = time.time() - t0

ldpath, missing = library_path()
if missing:
    report(ok=False, error="no CUDA 12 libraries on this runtime for: "
                           + ", ".join(missing))
    raise SystemExit(0)

with open(ENV_FILE, "w") as handle:
    json.dump({"LD_LIBRARY_PATH": ldpath}, handle)

env = dict(os.environ, LD_LIBRARY_PATH=ldpath)
probe = subprocess.run([BINARY, "--version"], capture_output=True, text=True, env=env)
if probe.returncode != 0:
    report(ok=False, error="gnina --version exited %d: %s"
                           % (probe.returncode, (probe.stderr or "")[-400:]))
    raise SystemExit(0)

# Prove every lazily-loaded library actually loads, in a CHILD process with the
# composed environment -- the exact conditions gnina will run under. Setting
# os.environ here would not do: the dynamic loader reads LD_LIBRARY_PATH once,
# when a process starts. This is the check that --version cannot make, because
# --version never reaches the CNN code that dlopens these.
unloadable = []
for soname in DLOPENED:
    check = subprocess.run(
        [sys.executable, "-c",
         "import ctypes, sys; ctypes.CDLL(sys.argv[1])", soname],
        capture_output=True, text=True, env=env)
    if check.returncode != 0:
        unloadable.append(soname)
if unloadable:
    report(ok=False, error="present on disk but not loadable with this "
                           "LD_LIBRARY_PATH: " + ", ".join(unloadable)
                           + ". gnina would abort mid-job when CNN scoring "
                             "dlopens them. Path was " + ldpath)
    raise SystemExit(0)

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True)

# gnina announces a missing GPU on stderr and then runs on CPU anyway. That is
# the one outcome this backend must never accept silently: the operator asked
# for a GPU backend and is billed by the hour for one.
banner = (probe.stdout or "") + (probe.stderr or "")
if "No GPU detected" in banner:
    report(ok=False, error="gnina starts but reports 'No GPU detected' -- it would "
                           "run on CPU at about 12x the wall time while the runtime "
                           "bills for its GPU. LD_LIBRARY_PATH was " + ldpath)
    raise SystemExit(0)
# The vCPU count, because it -- not the GPU -- is what bounds a gnina run: the
# Monte Carlo search is CPU-parallel and only the CNN rescoring is on the card.
cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
with open(MARKER, "w") as handle:
    handle.write(ldpath)

report(ok=True,
       version=(probe.stdout or "").strip().splitlines()[0] if probe.stdout else "",
       gpu=(gpu.stdout or "").strip(),
       cores=cores,
       library_path=ldpath,
       size_mb=round(os.path.getsize(BINARY) / 1e6, 1),
       cached=cached,
       install_seconds=round(install_seconds, 1))
'''


def launch_source(job_dir: str, receptor: str, ligand: str,
                  passthrough: Sequence[str],
                  lock_root: Optional[str] = None) -> str:
    """Python that launches gnina **detached** and returns at once.

    The run is wrapped in ``setsid nohup`` with its exit code written to a marker
    file, for the same reason the SSH backend does it: a docking run is minutes
    long, and a dropped websocket or an interrupted local process must not kill
    it. The caller then polls :func:`probe_source`.
    """
    args = [REMOTE_BINARY, "-r", posixpath.join(job_dir, receptor),
            "-l", posixpath.join(job_dir, ligand),
            "-o", posixpath.join(job_dir, "out.sdf"),
            "--log", posixpath.join(job_dir, "out.log")]
    args += list(passthrough)
    inner = " ".join(shlex.quote(a) for a in args)
    quoted_dir = shlex.quote(job_dir)
    # The detached job releases the mutex itself, as the last thing it does and
    # AFTER the exit code is written. Releasing it from the local process instead
    # would hand the GPU to the next job while this one is still on it, and a
    # local crash would leave it held until the staleness timeout.
    release = ""
    if lock_root:
        release = " ; rm -rf %s" % shlex.quote(posixpath.join(lock_root, LOCK_DIRNAME))
    return _fill(_LAUNCH_TEMPLATE, JOB_DIR=repr(job_dir), QUOTED_DIR=quoted_dir,
                 INNER=inner, ENV_FILE=repr(REMOTE_ENV_FILE),
                 MARKER=repr(READY_MARKER), RELEASE=release)


_LAUNCH_TEMPLATE = r'''
import json, os, subprocess

JOB_DIR = @@JOB_DIR@@
ENV_FILE = @@ENV_FILE@@
MARKER = @@MARKER@@

os.makedirs(JOB_DIR, exist_ok=True)

ldpath = ""
if os.path.exists(ENV_FILE):
    ldpath = json.load(open(ENV_FILE)).get("LD_LIBRARY_PATH", "")
elif os.path.exists(MARKER):
    ldpath = open(MARKER).read().strip()

# setsid detaches from the kernel's process group, so the run survives the
# websocket closing, a kernel restart, and this cell being interrupted. The exit
# code lands in a file because that is the only completion signal that is still
# there after a disconnection.
# EXTENDED, never replaced. The login shell's own LD_LIBRARY_PATH is
# /usr/lib64-nvidia, which holds libcuda.so.1 -- the NVIDIA driver library.
# Overwriting it lets gnina load, find no driver, print "No GPU detected" and
# finish on CPU: measured at better than 12x the wall time, on a runtime that
# bills for its GPU either way.
quoted_ldpath = '"' + ldpath + ':$LD_LIBRARY_PATH"'
script = (
    "cd @@QUOTED_DIR@@ && "
    "export LD_LIBRARY_PATH=" + quoted_ldpath + " && "
    "@@INNER@@ > stdout.txt 2> stderr.txt; echo $? > exit_code@@RELEASE@@"
)
subprocess.Popen(["setsid", "nohup", "bash", "-lc", script],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                 start_new_session=True)
print(json.dumps({"launched": True, "job_dir": JOB_DIR}))
'''


#: The GPU mutex, as a directory because ``os.mkdir`` is atomic. One runtime has
#: one GPU, and ``gnina_queue`` sets ``GNINA_REMOTE_LOCK=1`` on every submit
#: expecting the shim to honour it: two detached gnina processes sharing one T4
#: contend for VRAM and each run slower than if they had queued.
LOCK_DIRNAME = "gpu.lock"
DEFAULT_LOCK_STALE = 1800


def lock_source(root: str, owner: str, stale_seconds: int = DEFAULT_LOCK_STALE) -> str:
    """Python that takes the GPU mutex, or reports who holds it.

    A stale lock is taken over rather than waited on forever: the holder may have
    been a runtime that was pruned mid-job, and a mutex that outlives its owner
    stops the campaign rather than protecting it.
    """
    return _fill(_LOCK_TEMPLATE, ROOT=repr(root), OWNER=repr(owner),
                 LOCK_DIRNAME=repr(LOCK_DIRNAME), STALE=repr(int(stale_seconds)))


_LOCK_TEMPLATE = r'''
import json, os, shutil, time

ROOT = @@ROOT@@
OWNER = @@OWNER@@
LOCK = os.path.join(ROOT, @@LOCK_DIRNAME@@)
STALE = @@STALE@@

os.makedirs(ROOT, exist_ok=True)
info = os.path.join(LOCK, "owner")


def held_by():
    try:
        with open(info) as handle:
            return json.load(handle)
    except Exception:
        return {}


try:
    os.mkdir(LOCK)
except FileExistsError:
    current = held_by()
    age = time.time() - float(current.get("at") or 0)
    if age > STALE:
        # The holder is gone or wedged. Taking over is the lesser evil: the
        # alternative is a mutex nobody will ever release.
        shutil.rmtree(LOCK, ignore_errors=True)
        try:
            os.mkdir(LOCK)
        except OSError:
            print(json.dumps({"acquired": False, "owner": current, "age": age}))
            raise SystemExit(0)
    else:
        print(json.dumps({"acquired": False, "owner": current, "age": age}))
        raise SystemExit(0)

with open(info, "w") as handle:
    json.dump({"owner": OWNER, "at": time.time()}, handle)
print(json.dumps({"acquired": True, "lock": LOCK}))
'''


def unlock_source(root: str) -> str:
    """Python that releases the GPU mutex. Safe to run when it is not held."""
    return _fill(_UNLOCK_TEMPLATE, ROOT=repr(root),
                 LOCK_DIRNAME=repr(LOCK_DIRNAME))


_UNLOCK_TEMPLATE = r'''
import json, os, shutil

shutil.rmtree(os.path.join(@@ROOT@@, @@LOCK_DIRNAME@@), ignore_errors=True)
print(json.dumps({"released": True}))
'''


def probe_source(job_dir: str) -> str:
    """Python that reports one job's state without disturbing it."""
    return _fill(_PROBE_TEMPLATE, JOB_DIR=repr(job_dir))


_PROBE_TEMPLATE = r'''
import json, os

JOB_DIR = @@JOB_DIR@@


def read(name, limit=None):
    path = os.path.join(JOB_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as handle:
        text = handle.read()
    return text[-limit:] if limit else text


code = read("exit_code")
out = os.path.join(JOB_DIR, "out.sdf")
log_text = read("out.log") or ""
print(json.dumps({
    "exit_code": code.strip() if code else None,
    "out_bytes": os.path.getsize(out) if os.path.exists(out) else 0,
    "stderr_tail": read("stderr.txt", 1200),
    "log_tail": log_text[-1200:],
    # Surfaced so a run that fell back to CPU is caught while it is still going,
    # rather than after it has spent an hour of billed GPU time pretending.
    "no_gpu": "No GPU detected" in log_text,
}))
'''


#: Kept so existing callers of the private name keep working.
_last_json = last_json
