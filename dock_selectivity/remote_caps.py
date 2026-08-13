"""Remote GPU capability probe and docking-configuration validation.

Why this module exists
----------------------

gnina runs on another machine (see ``gnina_remote.py``).  The local pipeline was
sized for *this* box -- ``NUM_THREADS = 60`` -- and the GPU host has 24 cores.
Nothing in the stack notices: gnina accepts ``--cpu 60`` on a 24-core machine,
oversubscribes, and runs slower.  Worse, a request the remote genuinely cannot
serve (an exhaustiveness that will not finish inside the per-job deadline, a
thread count whose VRAM footprint exceeds the card) fails *per job*, thousands of
times, four hours into a campaign, after the smina stage has already run.

So: probe once, judge the request against what was measured, and refuse **before
stage 1** rather than discovering it job by job.

Clamp or refuse?  A decision per parameter
------------------------------------------

A silent clamp that changes a scientific setting is the exact failure this
project exists to prevent, so every parameter gets an explicit, documented
policy:

``--cpu`` (``NUM_THREADS``) -- **clamped, loudly, once.**
    Asking for 60 threads on 24 cores is not an error; it is oversubscription,
    which costs time and nothing else.  Refusing the campaign over it would be
    theatre.  But gnina spreads its Monte-Carlo chains over threads, so the
    thread count is *not* strictly result-neutral: it must be ONE value for the
    whole campaign.  Hence the clamp is a ``WARN`` naming both numbers, is
    resolved once, is frozen into ``caps.json``/``params.json``, and any later
    run that would resolve a *different* effective cpu is ``CAPS_CHANGED``
    (FATAL).  Loud, once, and then immutable.

``exhaustiveness`` -- **never clamped.  Refused.**
    Exhaustiveness is the search budget; it decides which poses exist and
    therefore what the cross-tool RMSD consensus is measuring.  Halving it to
    fit a deadline would change the science invisibly.  This is the user's rule
    -- *"if exhaustiveness required for gnina is too much for remote device we
    shall warn and exit"* -- implemented as :data:`EXHAUSTIVENESS_TOO_HIGH`.

``num_modes`` -- **never touched, never judged.**  It is the pose count; it has
    no capacity dimension worth guessing about.

VRAM -- **refused, never clamped.**  The only lever on VRAM is the thread count,
    and lowering the thread count is the operator's decision to make (see above),
    not ours to make silently.

The per-job deadline -- **operational, not scientific.**  It is the reference the
    exhaustiveness rule is measured against, never itself adjusted.

Never optimistic
----------------

If the host cannot be reached, that is :class:`RemoteCapsError` with the ssh
error attached -- never a defaulted, hopeful :class:`RemoteCaps`.  A guessed
capability record would be used to authorise a campaign that then fails on every
job.  :func:`parse_probe_output` is the one deliberately forgiving piece: it
never raises, because a remote missing ``nvidia-smi`` must produce
``gpu_present=False`` (which :func:`validate_request` turns into a FATAL with a
usable message), not a parse traceback in the middle of the pipeline.

Cache
-----

One probe per campaign, not one per job.  :func:`get_caps` reads and writes
``<queue_root>/caps.json`` (``$GNINA_QUEUE_DIR`` or ``<run_dir>/gnina_queue``,
see the queue spec section 3) and re-probes only when the record is older than
``ttl_seconds`` (6 h) or was taken against a different host/port.  ``--caps-file``
loads a canned record and never touches the network at all.

Departures from the specification (section 5), and why
------------------------------------------------------

Every name and signature in section 7(a) is implemented as frozen.  Three
*behavioural* additions were necessary; all three are additive, none changes a
documented rule, and none fires on the intended campaign:

1. ``DISK_UNKNOWN`` (new, WARN).  The specified probe runs
   ``df -Pk <root> ... | awk 'NR==2{print $4}'``.  Verified locally: when the
   scratch root does not exist -- which is the normal state before the first job
   creates it -- that prints nothing, not ``0``.  Parsing it as "0 MB free"
   would raise ``DISK_LOW`` FATAL and refuse a healthy campaign on its first
   run.  Unknown is therefore distinguished from zero: unknown warns, ``0``
   still refuses.
2. ``CORES_UNKNOWN`` (new, WARN).  With ``nproc`` missing, the spec's
   ``effective_cpu`` falls back to the requested value unclamped.  That is a
   silent pass-through of the exact parameter this module exists to police, so
   it is announced.
3. The VRAM rules are skipped when ``gpu_present`` is false.  Otherwise a
   host with no GPU emits a second FATAL reading "over the 0 MB usable of the
   0 MB on the remote GPU", which buries ``NO_GPU`` -- the finding the operator
   actually has to act on.

Also note: the section 5 sample text for ``VRAM_INSUFFICIENT`` ("--cpu 48 needs
~6480 MB ... over the 6963 MB usable") is arithmetically inconsistent -- 6480 is
*under* 6963 and that case is really ``VRAM_TIGHT``.  The **rule** is
implemented as specified (fatal when the estimate exceeds usable); only the
sample's numbers were wrong.  On the measured card the threshold is ``--cpu 53``.

Python 3.9 compatible.  stdlib only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# gnina_remote.py lives at the repo root, beside this package rather than
# inside it.  Import it defensively: the pure half of this module (parsing,
# validation, reporting) must stay importable and testable even if the shim is
# missing or momentarily broken, because only the SSH paths need it.
# --------------------------------------------------------------------------
_GNINA_REMOTE = None
_GNINA_REMOTE_IMPORT_ERROR = None  # type: Optional[BaseException]
try:  # pragma: no cover - exercised implicitly by every import in the suite
    import gnina_remote as _GNINA_REMOTE  # type: ignore[no-redef]
except Exception:  # noqa: BLE001 - a SyntaxError here must not sink the module
    try:
        _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        import gnina_remote as _GNINA_REMOTE  # type: ignore[no-redef]
    except Exception as _exc:  # noqa: BLE001
        _GNINA_REMOTE = None
        _GNINA_REMOTE_IMPORT_ERROR = _exc

#: Container image, mirrored from the shim so the two cannot drift apart. The
#: literal is only used when the shim could not be imported at all.
DEFAULT_IMAGE = getattr(_GNINA_REMOTE, "DEFAULT_IMAGE", "gnina/gnina:v1.3.3")
DEFAULT_REMOTE_ROOT = getattr(_GNINA_REMOTE, "DEFAULT_REMOTE_DIR", "gnina_jobs")

_LOG = logging.getLogger(__name__)

#: Six hours. A GPU host's core count and card do not change during a campaign;
#: its free VRAM and free disk do, which is why the record expires at all.
DEFAULT_CAPS_TTL_SECONDS = 21600


def _env_number(name: str, default: Any, cast: Any) -> Any:
    """Read a model-calibration override from the environment, once, at import.

    Deliberately resolved here rather than inside :func:`validate_request`: that
    function must stay pure so every rule is a table-driven test.  A malformed
    or non-positive override is reported and ignored -- silently reverting to
    the default would make a mistyped calibration look like it took effect.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = cast(raw.strip())
    except (TypeError, ValueError):
        _LOG.warning("ignoring %s=%r: not a number; using %s", name, raw, default)
        return default
    if value <= 0:
        _LOG.warning("ignoring %s=%r: must be positive; using %s",
                     name, raw, default)
        return default
    return value


#: Time model. Calibrated from the measured ~150 s per job at exhaustiveness 32
#: on 24 cores at ``--num_modes 20`` on an RTX 3060 Ti: 150 * 24 / 32 = 112.5.
#: It is a straight-line model in exhaustiveness and inverse in threads, which
#: is how gnina's Monte-Carlo search actually scales, and it is only ever used
#: to answer "is this obviously impossible?", never to promise a runtime.
#: Re-calibrate with GNINA_CORE_SECONDS_PER_EXH after measuring another card.
CORE_SECONDS_PER_EXH_UNIT = _env_number("GNINA_CORE_SECONDS_PER_EXH", 112.5, float)

#: VRAM model: a fixed CNN/context footprint plus a per-thread working set.
#: WARNING -- THESE DEFAULTS ARE KNOWN TO OVER-PREDICT BY ~30x. MEASURED
#: 2026-08-13 by sampling nvidia-smi at 1 Hz on the real RTX 3060 Ti during
#: production runs:
#:
#:     --cpu 24 --exhaustiveness 32  ->  546 MiB peak (243 MiB over idle)
#:     --cpu 60 --exhaustiveness 32  ->  545 MiB peak (241 MiB over idle)
#:
#: VRAM is essentially FLAT in the thread count, so the per-thread term is not
#: merely miscalibrated -- it is the wrong shape. The model predicts 7800 MB for
#: the --cpu 60 run, which actually used 545 MiB and completed normally in 2m23s.
#:
#: NOT CHANGED HERE, deliberately: on this host effective_cpu is clamped to 24, so
#: the estimate never exceeds 3840 MB against 6659 MB usable and the rule cannot
#: fire -- nothing is currently blocked. The calibration also rests on a single
#: ligand/receptor pair. It WOULD false-refuse a workable campaign on a many-core
#: host with the same card (64 cores at --cpu 60 -> 7800 MB predicted vs ~241 MiB
#: actual), so before running anywhere else, override:
#:
#:     export GNINA_VRAM_BASE_MB=800 GNINA_VRAM_PER_THREAD_MB=0
#:
#: and re-measure on a second ligand pair before making that the default. The
#: tests currently pin the old numbers, so changing the defaults means updating
#: them too -- they assert the model's arithmetic, not the hardware's behaviour.
VRAM_BASE_MB = _env_number("GNINA_VRAM_BASE_MB", 1200, int)
VRAM_PER_THREAD_MB = _env_number("GNINA_VRAM_PER_THREAD_MB", 110, int)
#: Never plan to fill the card. The display server, the CUDA context and
#: allocator fragmentation all live in the gap.
VRAM_SAFETY_FRACTION = 0.85

#: ``EX_CONFIG``. Distinct from batch_dock's 2 (usage) and 3 (nothing to do), so
#: "the remote cannot serve this configuration" is distinguishable in a log.
EXIT_CAPS_FATAL = 78

SEVERITIES = ("INFO", "WARN", "FATAL")
_SEVERITY_RANK = {"FATAL": 0, "WARN": 1, "INFO": 2}

#: Codes that ``--strict-caps`` promotes from WARN to FATAL.
STRICT_FATAL_CODES = frozenset({"CPU_OVERSUBSCRIBED"})

#: Basename of the cached record inside the queue root.
DEFAULT_CACHE_BASENAME = "caps.json"

#: Keys the probe emits. Anything else in the output is ignored; anything
#: missing here is defaulted. Both directions are deliberate -- a remote with an
#: MOTD or a chatty profile must not break the probe.
_PROBE_KEYS = ("cores", "mem_total_kb", "mem_avail_kb", "gpu", "docker",
               "image", "native", "disk_free_kb", "uptime")

#: ``df`` could not be read for the scratch root (it usually does not exist
#: until the first job creates it). Distinguished from "0 MB free", which is a
#: full disk and genuinely fatal.
DISK_UNKNOWN = -1


class RemoteCapsError(RuntimeError):
    """The remote could not be characterised. Never softened into a default."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _utcnow_iso() -> str:
    """ISO-8601 UTC with a trailing ``Z``, microsecond precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _iso_from_epoch(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_iso_utc(text: Optional[str]) -> Optional[float]:
    """Epoch seconds for a timestamp written by :func:`_utcnow_iso`.

    Returns ``None`` rather than raising: an unreadable timestamp means "age
    unknown", which the cache treats as stale.
    """
    if not text:
        return None
    raw = str(text).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _as_int(value: Any, default: int = 0) -> int:
    """Best-effort integer. ``[N/A]``, ``""`` and rubbish all become *default*."""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip() in ("1", "true", "True", "yes")


def _optional_int(value: Any) -> Optional[int]:
    """Ports arrive from the environment as strings; keep ``None`` as ``None``."""
    if value is None or value == "":
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    """Write JSON so a reader sees either the old file or the new one.

    Same discipline as the queue's state store (spec section 3): a temp file in
    the same directory, ``O_EXCL``, fsync, ``os.replace``, then a best-effort
    directory fsync.  No hardlinks -- the run directory can be a fuseblk mount
    where ``os.link`` is unavailable and a directory fsync raises.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(
        directory,
        "%s.tmp.%d.%d" % (os.path.basename(path), os.getpid(), time.monotonic_ns()),
    )
    payload = (json.dumps(obj, sort_keys=True, indent=1) + "\n").encode("utf-8")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        _LOG.debug("directory fsync unsupported on %s; continuing", directory)


def cache_path_for(queue_root: str) -> str:
    """Where the cached record lives: ``<queue_root>/caps.json``."""
    return os.path.join(queue_root, DEFAULT_CACHE_BASENAME)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RemoteCaps:
    """What the remote actually has. Measured, never assumed."""

    host: str
    port: Optional[int] = None
    key: Optional[str] = None
    cores: int = 0
    mem_total_mb: int = 0
    mem_available_mb: int = 0
    gpu_present: bool = False
    gpu_name: str = ""
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    compute_capability: str = ""
    docker_available: bool = False
    image_present: bool = False
    native_present: bool = False
    disk_free_mb: int = DISK_UNKNOWN
    uptime_seconds: float = 0.0
    probed_at: str = ""
    source: str = "probe"          # "probe" | "cache" | "file"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "key": self.key,
            "cores": self.cores,
            "mem_total_mb": self.mem_total_mb,
            "mem_available_mb": self.mem_available_mb,
            "gpu_present": self.gpu_present,
            "gpu_name": self.gpu_name,
            "vram_total_mb": self.vram_total_mb,
            "vram_used_mb": self.vram_used_mb,
            "compute_capability": self.compute_capability,
            "docker_available": self.docker_available,
            "image_present": self.image_present,
            "native_present": self.native_present,
            "disk_free_mb": self.disk_free_mb,
            "uptime_seconds": self.uptime_seconds,
            "probed_at": self.probed_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RemoteCaps":
        """Tolerant of missing and unknown keys, so an old record still loads."""
        if not isinstance(d, dict):
            raise RemoteCapsError("capability record is not a JSON object")
        host = d.get("host")
        if not host:
            raise RemoteCapsError("capability record has no host")
        return cls(
            host=str(host),
            port=_optional_int(d.get("port")),
            key=d.get("key") or None,
            cores=_as_int(d.get("cores")),
            mem_total_mb=_as_int(d.get("mem_total_mb")),
            mem_available_mb=_as_int(d.get("mem_available_mb")),
            gpu_present=_as_bool(d.get("gpu_present")),
            gpu_name=str(d.get("gpu_name") or ""),
            vram_total_mb=_as_int(d.get("vram_total_mb")),
            vram_used_mb=_as_int(d.get("vram_used_mb")),
            compute_capability=str(d.get("compute_capability") or ""),
            docker_available=_as_bool(d.get("docker_available")),
            image_present=_as_bool(d.get("image_present")),
            native_present=_as_bool(d.get("native_present")),
            disk_free_mb=_as_int(d.get("disk_free_mb"), DISK_UNKNOWN),
            uptime_seconds=_as_float(d.get("uptime_seconds")),
            probed_at=str(d.get("probed_at") or ""),
            source=str(d.get("source") or "probe"),
        )

    @property
    def usable_vram_mb(self) -> float:
        """VRAM we are willing to plan against, after the safety margin."""
        return self.vram_total_mb * VRAM_SAFETY_FRACTION - self.vram_used_mb

    @property
    def endpoint(self) -> str:
        return "%s:%s" % (self.host, self.port) if self.port else self.host


@dataclass(frozen=True)
class DockRequest:
    """What this run wants to ask the remote to do."""

    cpu: int
    exhaustiveness: int
    num_modes: Optional[int] = None
    job_deadline_seconds: float = 1800.0
    image: str = DEFAULT_IMAGE
    native: Optional[str] = None
    #: Whatever ``GNINA_PATH`` points at. Async mode requires the shim.
    gnina_path: str = ""
    async_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu": self.cpu,
            "exhaustiveness": self.exhaustiveness,
            "num_modes": self.num_modes,
            "job_deadline_seconds": self.job_deadline_seconds,
            "image": self.image,
            "native": self.native,
            "gnina_path": self.gnina_path,
            "async_mode": self.async_mode,
        }


@dataclass(frozen=True)
class CapsFinding:
    """One judgement about the request. The message is the deliverable."""

    code: str
    severity: str
    message: str
    remedy: str = ""
    observed: Any = None
    limit: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "remedy": self.remedy,
            "observed": self.observed,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class CapsReport:
    """The verdict: the numbers behind it and every finding, in one object."""

    caps: RemoteCaps
    request: DockRequest
    effective_cpu: int
    estimated_seconds: float
    estimated_vram_mb: float
    findings: Tuple[CapsFinding, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.fatal

    @property
    def fatal(self) -> List[CapsFinding]:
        return [f for f in self.findings if f.severity == "FATAL"]

    @property
    def warnings(self) -> List[CapsFinding]:
        return [f for f in self.findings if f.severity == "WARN"]

    def format_report(self) -> str:
        return _format_report(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "caps": self.caps.to_dict(),
            "request": self.request.to_dict(),
            "effective_cpu": self.effective_cpu,
            "estimated_seconds": self.estimated_seconds,
            "estimated_vram_mb": self.estimated_vram_mb,
            "usable_vram_mb": self.caps.usable_vram_mb,
            "findings": [f.to_dict() for f in self.findings],
        }


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------

def probe_command(image: str, native: Optional[str], remote_root: str) -> str:
    """One shell command, one round trip, every field independently degradable.

    Each value is wrapped so that a missing tool yields ``0``/empty instead of
    failing the whole command: a remote without ``nvidia-smi`` must still report
    its core count, or one missing binary blinds the entire probe.
    """
    root = remote_root or "."
    # A quoted "~" is literal to the remote shell (the same trap gnina_remote
    # documents): keep the path relative or absolute, never tilde-prefixed.
    if root.startswith("~/"):
        root = root[2:] or "."
    elif root == "~":
        root = "."
    q_root = shlex.quote(root)
    q_image = shlex.quote(image or DEFAULT_IMAGE)
    # With no native binary requested, probe the plain name: knowing whether a
    # native gnina exists is useful in the report even when unused.
    q_native = shlex.quote(native or "gnina")
    lines = [
        'echo "cores=$(nproc 2>/dev/null || echo 0)"',
        'echo "mem_total_kb=$(awk \'/MemTotal/{print $2}\' /proc/meminfo '
        '2>/dev/null || echo 0)"',
        'echo "mem_avail_kb=$(awk \'/MemAvailable/{print $2}\' /proc/meminfo '
        '2>/dev/null || echo 0)"',
        'echo "gpu=$(nvidia-smi --query-gpu=name,memory.total,memory.used,'
        'compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)"',
        'echo "docker=$(command -v docker >/dev/null 2>&1 && echo 1 || echo 0)"',
        'echo "image=$(docker image inspect %s >/dev/null 2>&1 && echo 1 '
        '|| echo 0)"' % q_image,
        'echo "native=$(command -v %s >/dev/null 2>&1 && echo 1 || echo 0)"'
        % q_native,
        'echo "disk_free_kb=$(df -Pk %s 2>/dev/null | awk \'NR==2{print $4}\' '
        '|| echo 0)"' % q_root,
        'echo "uptime=$(cut -d\' \' -f1 /proc/uptime 2>/dev/null || echo 0)"',
    ]
    return "; ".join(lines)


def _parse_gpu_field(text: str) -> Tuple[str, int, int, str]:
    """``name, total, used, compute_cap`` from one ``nvidia-smi`` CSV row.

    The name is taken as *everything but the last three fields* so a card whose
    marketing name contains a comma cannot shift the numbers.
    """
    raw = (text or "").strip()
    if not raw:
        return "", 0, 0, ""
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) >= 4:
        name = ", ".join(parts[:-3]).strip()
        return name, _as_int(parts[-3]), _as_int(parts[-2]), parts[-1]
    return raw, 0, 0, ""


def parse_probe_output(text: str, host: str, port: Optional[int] = None,
                       key: Optional[str] = None,
                       probed_at: Optional[str] = None) -> RemoteCaps:
    """Pure ``key=value`` parse. **Never raises.**

    Unknown keys are ignored and missing keys are defaulted, so a login banner,
    an MOTD or a future extra field cannot break a campaign.  A remote with no
    GPU parses cleanly to ``gpu_present=False`` -- judging that is
    :func:`validate_request`'s job, not the parser's.
    """
    values = {}  # type: Dict[str, str]
    for line in (text or "").splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name in _PROBE_KEYS:
            values[name] = value.strip()

    gpu_name, vram_total, vram_used, compute = _parse_gpu_field(values.get("gpu", ""))
    disk_raw = values.get("disk_free_kb", "")
    disk_mb = _as_int(disk_raw, DISK_UNKNOWN)
    if disk_mb > 0:
        disk_mb = disk_mb // 1024
    elif disk_raw.strip() == "":
        # df said nothing at all: the scratch root probably does not exist yet.
        disk_mb = DISK_UNKNOWN

    return RemoteCaps(
        host=host,
        port=_optional_int(port),
        key=key or None,
        cores=_as_int(values.get("cores")),
        mem_total_mb=_as_int(values.get("mem_total_kb")) // 1024,
        mem_available_mb=_as_int(values.get("mem_avail_kb")) // 1024,
        gpu_present=bool(gpu_name),
        gpu_name=gpu_name,
        vram_total_mb=vram_total,
        vram_used_mb=vram_used,
        compute_capability=compute,
        docker_available=_as_bool(values.get("docker")),
        image_present=_as_bool(values.get("image")),
        native_present=_as_bool(values.get("native")),
        disk_free_mb=disk_mb,
        uptime_seconds=_as_float(values.get("uptime")),
        probed_at=probed_at or _utcnow_iso(),
        source="probe",
    )


def probe_remote(host: str, port: Optional[int] = None, key: Optional[str] = None,
                 image: str = DEFAULT_IMAGE, native: Optional[str] = None,
                 remote_root: str = DEFAULT_REMOTE_ROOT, timeout: float = 120,
                 retries: int = 3,
                 ssh_runner: Optional[Callable[..., Any]] = None) -> RemoteCaps:
    """Measure the remote over SSH. Raises rather than guessing.

    *ssh_runner* is the only seam: ``runner(cmd_list, what, retries, timeout,
    log)`` returning an object with ``.stdout``/``.stderr``/``.returncode``.  It
    defaults to :func:`gnina_remote.run_resilient`, which already retries
    transient network failures with backoff and refuses to retry real errors.
    """
    if not host:
        raise RemoteCapsError("no remote host given; set GNINA_REMOTE_HOST")

    runner = ssh_runner
    if runner is None:
        if _GNINA_REMOTE is None:
            raise RemoteCapsError(
                "cannot probe %s: gnina_remote.py could not be imported (%s). "
                "It provides the resilient SSH runner; pass ssh_runner= to use "
                "another one." % (host, _GNINA_REMOTE_IMPORT_ERROR))
        runner = _GNINA_REMOTE.run_resilient

    if _GNINA_REMOTE is not None:
        base = _GNINA_REMOTE.ssh_base(host, port, key)
    else:  # pragma: no cover - only when the shim is unimportable
        base = ["ssh"] + (["-p", str(port)] if port else []) + \
               (["-i", key] if key else []) + [host]

    cmd = list(base) + [probe_command(image, native, remote_root)]

    def _log(msg: str) -> None:
        # run_resilient's message already names the operation; do not stutter.
        _LOG.warning("%s", msg)

    try:
        proc = runner(cmd, "remote capability probe", retries, timeout, _log)
    except Exception as exc:  # noqa: BLE001 - re-raised as our own error below
        transient = _GNINA_REMOTE is not None and isinstance(
            exc, getattr(_GNINA_REMOTE, "TransientError", ()))
        raise RemoteCapsError(
            "cannot reach %s to measure its capacity: %s%s\nThe remote GPU host "
            "must be reachable before a gnina campaign starts; guessing its "
            "capacity here would authorise a run that then fails on every job."
            % (host, exc, " (network-level; the link, not the request)"
               if transient else "")) from exc

    returncode = getattr(proc, "returncode", 0)
    stdout = getattr(proc, "stdout", "") or ""
    stderr = getattr(proc, "stderr", "") or ""
    if returncode != 0:
        raise RemoteCapsError(
            "capability probe on %s exited %s: %s"
            % (host, returncode, (stderr.strip() or "no stderr")[:500]))

    if not any(("%s=" % k) in stdout for k in _PROBE_KEYS):
        raise RemoteCapsError(
            "capability probe on %s produced no recognisable output. stdout=%r "
            "stderr=%r. Something ate the command (a login shell, an MOTD, a "
            "restricted shell); treating that as 'no GPU' would report the "
            "wrong problem." % (host, stdout[:300], stderr[:300]))

    return parse_probe_output(stdout, host, port=port, key=key)


# --------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------

def load_caps(path: str) -> RemoteCaps:
    """Load a canned/cached record. Loud on absence or corruption.

    ``--caps-file`` is an explicit operator instruction; a missing or unreadable
    file there is a mistake to report, not to paper over with a probe.
    """
    try:
        with open(path, "r") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RemoteCapsError("cannot read capability record %s: %s" % (path, exc))
    caps = RemoteCaps.from_dict(data)
    return _replace(caps, source="file")


def save_caps(caps: RemoteCaps, path: str) -> None:
    """Persist the record atomically. Also the anchor for ``CAPS_CHANGED``."""
    _atomic_write_json(path, caps.to_dict())


def get_caps(host: str, port: Optional[int] = None, key: Optional[str] = None,
             cache_path: Optional[str] = None,
             ttl_seconds: float = DEFAULT_CAPS_TTL_SECONDS, force: bool = False,
             now: Optional[Any] = None, **probe_kw: Any) -> RemoteCaps:
    """Cached :func:`probe_remote`: one probe per campaign, not one per job.

    The cache is used only when it is younger than *ttl_seconds* **and** was
    taken against the same host and port -- a record from another machine says
    nothing about this one.  *now* is epoch seconds (or a zero-argument callable
    returning them) so tests can age the cache without sleeping; when it is
    given, the record this call writes is stamped from that same clock, because
    a clock that applies to only half of an ageing comparison is a bug waiting
    to happen.
    """
    injected_clock = now is not None
    now_value = now() if callable(now) else now
    if now_value is None:
        now_value = time.time()

    if cache_path and not force:
        cached = _read_cache(cache_path)
        if cached is not None:
            fresh, why = _cache_is_usable(cached, host, port, ttl_seconds, now_value)
            if fresh:
                _LOG.debug("using cached remote capabilities from %s", cache_path)
                return _replace(cached, source="cache")
            _LOG.info("re-probing %s: cached capabilities %s", host, why)

    caps = probe_remote(host, port=port, key=key, **probe_kw)
    if injected_clock:
        caps = _replace(caps, probed_at=_iso_from_epoch(now_value))
    if cache_path:
        try:
            save_caps(caps, cache_path)
        except OSError as exc:
            # A cache that cannot be written is a slower campaign, not a failed
            # one. The freeze that matters (CAPS_CHANGED) lives in params.json.
            _LOG.warning("could not cache remote capabilities to %s: %s",
                         cache_path, exc)
    return caps


def _read_cache(path: str) -> Optional[RemoteCaps]:
    """A corrupt or absent cache is a cache miss, never an error."""
    try:
        with open(path, "r") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    try:
        return RemoteCaps.from_dict(data)
    except RemoteCapsError:
        return None


def _cache_is_usable(caps: RemoteCaps, host: str, port: Optional[int],
                     ttl_seconds: float, now_value: float) -> Tuple[bool, str]:
    if caps.host != host:
        return False, "were taken against %s, not %s" % (caps.host, host)
    if _optional_int(caps.port) != _optional_int(port):
        return False, "were taken against port %s, not %s" % (caps.port, port)
    probed = _parse_iso_utc(caps.probed_at)
    if probed is None:
        return False, "carry no readable timestamp"
    age = now_value - probed
    if age < 0:
        return False, "are stamped in the future (%.0f s ahead)" % (-age)
    if age > ttl_seconds:
        return False, "are %.0f s old (ttl %.0f s)" % (age, ttl_seconds)
    return True, ""


def _replace(caps: RemoteCaps, **changes: Any) -> RemoteCaps:
    data = caps.to_dict()
    data.update(changes)
    return RemoteCaps.from_dict(data)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def max_exhaustiveness_for(effective_cpu: int, deadline_seconds: float) -> int:
    """Largest exhaustiveness the model says fits inside *deadline_seconds*.

    The refusal message quotes this number as "lower it to <= N", so N must
    actually pass the predicate that produced the refusal.  Truncating the
    division is not enough on its own -- float error could put N a hair over --
    so the result is verified against the same estimate and stepped down.  A
    remedy that does not work is worse than no remedy.
    """
    if deadline_seconds <= 0 or effective_cpu <= 0:
        return 0
    ceiling = int(deadline_seconds * effective_cpu / CORE_SECONDS_PER_EXH_UNIT)
    while ceiling > 0 and estimate_seconds(ceiling, effective_cpu) > deadline_seconds:
        ceiling -= 1
    return ceiling


def estimate_seconds(exhaustiveness: int, effective_cpu: int) -> float:
    return CORE_SECONDS_PER_EXH_UNIT * float(exhaustiveness) / max(1, effective_cpu)


def estimate_vram_mb(effective_cpu: int) -> float:
    return float(VRAM_BASE_MB) + float(VRAM_PER_THREAD_MB) * max(0, effective_cpu)


def shim_modes(gnina_path, timeout=20):
    """Modes the configured shim reports via ``--capabilities``.

    Returns an empty set when the path is missing, is not this shim, or predates
    the ``--capabilities`` query -- all of which mean "cannot be trusted to run
    the async protocol". Asking what it CAN DO replaces comparing its filename to
    "gnina_remote.py", a test that passed against a build with no submit/collect
    mode at all -- exactly the failure the rule exists to catch.
    """
    if not gnina_path or not os.path.exists(gnina_path):
        return set()
    try:
        proc = subprocess.run([sys.executable, gnina_path, "--capabilities"],
                              capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return set()
    if proc.returncode != 0:
        return set()
    text = (proc.stdout or "").strip().lower()
    if "modes:" not in text:
        return set()
    return {w for w in text.split("modes:", 1)[1].replace(",", " ").split()}


def effective_cpu_for(caps: RemoteCaps, request: DockRequest) -> int:
    """The one definition of the clamp.

    Exposed because the ``CAPS_CHANGED`` check needs the value a *previous*
    ``caps.json`` implied, and a caller re-deriving ``min(cpu, cores)`` by hand
    would be a second definition of the rule that freezes a campaign's thread
    count -- exactly the kind of drift this module exists to catch.
    """
    if caps.cores > 0:
        return max(1, min(request.cpu, caps.cores))
    return max(1, request.cpu)


def validate_request(caps: RemoteCaps, request: DockRequest, strict: bool = False,
                     previous_effective_cpu: Optional[int] = None) -> CapsReport:
    """Judge *request* against *caps*. **Pure**: no I/O, no clock, no environment.

    Every rule is therefore a table-driven unit test, which is the point: the
    predicate that decides whether a campaign starts must be checkable without a
    GPU host in the loop.
    """
    findings = []  # type: List[CapsFinding]

    def add(code, severity, message, remedy="", observed=None, limit=None):
        findings.append(CapsFinding(code=code, severity=severity, message=message,
                                    remedy=remedy, observed=observed, limit=limit))

    # ---- the clamp (see the module docstring: loud, once, then frozen) -----
    effective_cpu = effective_cpu_for(caps, request)
    if caps.cores <= 0:
        add("CORES_UNKNOWN", "WARN",
            "could not read a core count from %s (nproc reported nothing), so "
            "--cpu %d is passed through unclamped and the time estimate below "
            "is a guess rather than a measurement."
            % (caps.endpoint, request.cpu),
            remedy="check that the remote is a Linux host with nproc, or set "
                   "NUM_THREADS to a value you know it can serve.",
            observed=caps.cores, limit=None)
    effective_cpu = max(1, effective_cpu)

    est_seconds = estimate_seconds(request.exhaustiveness, effective_cpu)
    est_vram = estimate_vram_mb(effective_cpu)
    usable_vram = caps.usable_vram_mb

    # ---- can it run gnina at all? -----------------------------------------
    if not caps.gpu_present:
        add("NO_GPU", "FATAL",
            "%s reports no NVIDIA GPU (nvidia-smi produced nothing). gnina here "
            "is GPU-only; there is no CPU fallback and a CPU run would silently "
            "produce different CNN scores." % caps.host,
            remedy="fix the driver/toolkit on the remote, or drop gnina from "
                   "--tools.",
            observed=caps.gpu_name or None, limit="an NVIDIA GPU")

    if request.native:
        if not caps.native_present:
            add("NO_GNINA", "FATAL",
                "native gnina %s is not on %s's PATH, so every gnina job would "
                "fail on the remote." % (request.native, caps.host),
                remedy="install gnina there, unset GNINA_REMOTE_NATIVE to use "
                       "the %s container, or drop gnina from --tools."
                       % request.image,
                observed=request.native, limit="an executable on PATH")
    elif not caps.docker_available or not caps.image_present:
        if not caps.docker_available:
            reason = "docker is not installed"
        else:
            reason = "the image %s is not present" % request.image
        add("NO_GNINA", "FATAL",
            "%s cannot run the gnina container: %s. Every gnina job would fail "
            "on the remote." % (caps.host, reason),
            remedy="install docker and `docker pull %s` on the remote, or set "
                   "GNINA_REMOTE_NATIVE to a native gnina binary."
                   % request.image,
            observed={"docker": caps.docker_available,
                      "image": caps.image_present},
            limit="docker + %s" % request.image)

    if request.async_mode:
        # Ask what it CAN DO, not what it is called. The previous test compared
        # os.path.basename(...) to "gnina_remote.py" and so passed against a build
        # of the shim that had no submit/collect mode -- exactly the failure this
        # rule is named for, confirmed live. A filename is not a capability.
        modes = shim_modes(request.gnina_path)
        missing = [m for m in ("submit", "collect") if m not in modes]
        if missing:
            add("ASYNC_NEEDS_SHIM", "FATAL",
                "GNINA_ASYNC=1 needs a gnina_remote.py that implements the "
                "submit/collect modes, but %r reports %s. The decoupled queue "
                "drives gnina through those modes; without them the pump blocks "
                "and never writes a ticket."
                % (request.gnina_path or "",
                   ("modes: " + ", ".join(sorted(modes))) if modes
                   else "no --capabilities support at all"),
                remedy="point GNINA_PATH at an up-to-date <repo>/gnina_remote.py "
                       "(check: `python3 gnina_remote.py --capabilities`), or "
                       "unset GNINA_ASYNC.",
                observed=sorted(modes) or None, limit="submit+collect")

    # ---- VRAM: refused, never clamped -------------------------------------
    # Only meaningful when there is a card to measure against. On a host with no
    # GPU the numbers are all zero, and a second FATAL reading "over the 0 MB
    # usable of the 0 MB on the remote GPU" would bury NO_GPU, which is the
    # finding the operator actually has to act on.
    if not caps.gpu_present:
        pass
    elif est_vram > usable_vram:
        add("VRAM_INSUFFICIENT", "FATAL",
            "--cpu %d needs ~%.0f MB of VRAM, over the %.0f MB usable of the "
            "%d MB on %s (%d%% safety margin). gnina would fail with a CUDA "
            "out-of-memory part-way through the campaign."
            % (effective_cpu, est_vram, usable_vram, caps.vram_total_mb,
               caps.gpu_name or "the remote GPU", int(VRAM_SAFETY_FRACTION * 100)),
            remedy="lower NUM_THREADS, or use a larger GPU.",
            observed=est_vram, limit=usable_vram)
    elif est_vram > 0.70 * usable_vram:
        add("VRAM_TIGHT", "WARN",
            "--cpu %d needs ~%.0f MB of VRAM, %.0f%% of the %.0f MB usable on "
            "%s. Anything else landing on this GPU would push gnina into a CUDA "
            "out-of-memory."
            % (effective_cpu, est_vram,
               100.0 * est_vram / usable_vram if usable_vram > 0 else 100.0,
               usable_vram, caps.gpu_name or "the remote GPU"),
            remedy="keep other GPU work off %s, or lower NUM_THREADS." % caps.host,
            observed=est_vram, limit=0.70 * usable_vram)

    # ---- exhaustiveness: never clamped, refused ---------------------------
    deadline = request.job_deadline_seconds
    if est_seconds > deadline:
        ceiling = max_exhaustiveness_for(effective_cpu, deadline)
        add("EXHAUSTIVENESS_TOO_HIGH", "FATAL",
            "exhaustiveness=%d at --cpu %d needs ~%.0f s per job on %s "
            "(%d cores), over the %.0f s per-job deadline. Every job would "
            "time out."
            % (request.exhaustiveness, effective_cpu, est_seconds, caps.host,
               caps.cores, deadline),
            remedy="lower GNINA_EXHAUSTIVENESS to <= %d, or raise "
                   "GNINA_JOB_DEADLINE above %.0f s." % (ceiling, est_seconds),
            observed=est_seconds, limit=deadline)
    elif est_seconds > 0.50 * deadline:
        add("EXHAUSTIVENESS_SLOW", "WARN",
            "exhaustiveness=%d at --cpu %d needs ~%.0f s per job on %s, more "
            "than half the %.0f s per-job deadline. A job that runs slower than "
            "the model would time out and be retried, doubling its GPU time."
            % (request.exhaustiveness, effective_cpu, est_seconds, caps.host,
               deadline),
            remedy="lower GNINA_EXHAUSTIVENESS, or raise GNINA_JOB_DEADLINE.",
            observed=est_seconds, limit=0.50 * deadline)

    # ---- cpu oversubscription: the clamp, announced -----------------------
    if caps.cores > 0 and request.cpu > caps.cores:
        if strict:
            # --strict-caps means "do not touch my parameters at all": refuse
            # rather than clamp, and say so instead of describing a clamp that
            # is not going to happen.
            add("CPU_OVERSUBSCRIBED", "FATAL",
                "--cpu %d requested but %s has %d cores, and --strict-caps "
                "refuses to change a docking parameter on your behalf. The "
                "clamp to --cpu %d is available without it."
                % (request.cpu, caps.host, caps.cores, effective_cpu),
                remedy="set NUM_THREADS=%d, or drop --strict-caps to accept the "
                       "clamp as a warning." % effective_cpu,
                observed=request.cpu, limit=caps.cores)
        else:
            add("CPU_OVERSUBSCRIBED", "WARN",
                "--cpu %d requested but %s has %d cores. Clamping to --cpu %d "
                "for every gnina job in this run and recording it in "
                "gnina_queue/caps.json; a mid-campaign change is refused."
                % (request.cpu, caps.host, caps.cores, effective_cpu),
                remedy="set NUM_THREADS=%d, or pass --strict-caps to make this "
                       "fatal." % effective_cpu,
                observed=request.cpu, limit=caps.cores)

    # ---- headroom ---------------------------------------------------------
    if caps.mem_total_mb < 4096:
        add("RAM_TIGHT", "WARN",
            "%s reports %d MB of RAM (%d MB available). gnina's grids plus the "
            "container runtime want more headroom than that, and the remote may "
            "start killing jobs under memory pressure."
            % (caps.host, caps.mem_total_mb, caps.mem_available_mb),
            remedy="free memory on the remote, or use a larger machine.",
            observed=caps.mem_total_mb, limit=4096)

    if caps.disk_free_mb == DISK_UNKNOWN or caps.disk_free_mb < 0:
        add("DISK_UNKNOWN", "WARN",
            "could not read the free space on %s (df reported nothing for the "
            "gnina scratch root, which usually just means it does not exist "
            "yet). Proceeding: a genuinely full disk will surface as a failed "
            "job rather than as a refusal here." % caps.host,
            remedy="re-probe once the scratch root exists, or point "
                   "GNINA_REMOTE_DIR at a path that does.",
            observed=None, limit=5000)
    elif caps.disk_free_mb < 1000:
        add("DISK_LOW", "FATAL",
            "%s has %d MB free on the filesystem holding the gnina scratch "
            "directory. Each job leaves a receptor, a ligand and ~2 MB of poses "
            "there until it is collected; the campaign would stall on a full "
            "disk." % (caps.host, caps.disk_free_mb),
            remedy="free space on the remote, or set GNINA_REMOTE_DIR to a "
                   "larger filesystem.",
            observed=caps.disk_free_mb, limit=1000)
    elif caps.disk_free_mb < 5000:
        add("DISK_LOW", "WARN",
            "%s has %d MB free on the filesystem holding the gnina scratch "
            "directory. Each job leaves ~2 MB there until it is collected."
            % (caps.host, caps.disk_free_mb),
            remedy="free space on the remote, or set GNINA_REMOTE_DIR to a "
                   "larger filesystem.",
            observed=caps.disk_free_mb, limit=5000)

    # ---- the freeze -------------------------------------------------------
    if previous_effective_cpu is not None and previous_effective_cpu != effective_cpu:
        add("CAPS_CHANGED", "FATAL",
            "this run resolves --cpu %d for gnina, but the campaign was started "
            "with --cpu %d (recorded in gnina_queue/caps.json). gnina spreads "
            "its Monte-Carlo chains over threads, so changing the thread count "
            "mid-campaign changes the search and the results stop being "
            "comparable." % (effective_cpu, previous_effective_cpu),
            remedy="restore the original NUM_THREADS, or pass --allow-caps-change "
                   "if the campaign really is being re-parameterised.",
            observed=effective_cpu, limit=previous_effective_cpu)

    ordered = tuple(sorted(findings, key=lambda f: _SEVERITY_RANK.get(f.severity, 9)))
    return CapsReport(caps=caps, request=request, effective_cpu=effective_cpu,
                      estimated_seconds=est_seconds, estimated_vram_mb=est_vram,
                      findings=ordered)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _wrap(text: str, initial: str = "", subsequent: str = "",
          width: int = 96) -> List[str]:
    """Wrap for the terminal without ever splitting a token.

    ``break_long_words``/``break_on_hyphens`` are off so a path, a code or a
    number is never cut in half -- these reports get grepped.
    """
    return textwrap.wrap(text, width=width, initial_indent=initial,
                         subsequent_indent=subsequent, break_long_words=False,
                         break_on_hyphens=False) or [initial + text]


def _format_finding(finding: CapsFinding) -> List[str]:
    head = "  %-5s [%s] " % (finding.severity, finding.code)
    lines = _wrap(head + finding.message, subsequent="         ")
    if finding.remedy:
        lines += _wrap("Remedy: " + finding.remedy, initial="         ",
                       subsequent="                 ")
    return lines


def _format_report(report: CapsReport) -> str:
    caps = report.caps
    req = report.request
    out = ["gnina remote capability report -- %s" % caps.endpoint]

    def row(label, value):
        out.append("  %-14s %s" % (label, value))

    row("probed", "%s (%s)" % (caps.probed_at or "unknown", caps.source))
    row("cores", "%s  (requested --cpu %d -> effective --cpu %d)"
        % (caps.cores if caps.cores > 0 else "unknown", req.cpu,
           report.effective_cpu))
    row("memory", "%d MB total, %d MB available"
        % (caps.mem_total_mb, caps.mem_available_mb))
    if caps.gpu_present:
        row("gpu", "%s, %d MB VRAM (%d MB in use), compute capability %s"
            % (caps.gpu_name, caps.vram_total_mb, caps.vram_used_mb,
               caps.compute_capability or "unknown"))
    else:
        row("gpu", "NONE DETECTED")
    row("runtime", "docker=%s image(%s)=%s native(%s)=%s"
        % ("yes" if caps.docker_available else "no", req.image,
           "yes" if caps.image_present else "no", req.native or "gnina",
           "yes" if caps.native_present else "no"))
    row("disk", "unknown" if caps.disk_free_mb == DISK_UNKNOWN
        else "%d MB free" % caps.disk_free_mb)
    row("time model", "exhaustiveness %d on %d threads -> ~%.0f s per job "
        "(deadline %.0f s, ceiling exhaustiveness %d)"
        % (req.exhaustiveness, report.effective_cpu, report.estimated_seconds,
           req.job_deadline_seconds,
           max_exhaustiveness_for(report.effective_cpu, req.job_deadline_seconds)))
    row("vram model", "~%.0f MB needed of ~%.0f MB usable (%d%% of %d MB, "
        "%d MB in use)"
        % (report.estimated_vram_mb, caps.usable_vram_mb,
           int(VRAM_SAFETY_FRACTION * 100), caps.vram_total_mb, caps.vram_used_mb))

    if report.findings:
        out.append("")
        for finding in report.findings:
            out.extend(_format_finding(finding))
    else:
        out.append("")
        out.append("  no findings: the remote can serve this configuration.")

    out.append("")
    n_fatal = len(report.fatal)
    n_warn = len(report.warnings)
    if n_fatal:
        out.append("verdict: REFUSED -- %d fatal, %d warning(s). exit %d"
                   % (n_fatal, n_warn, EXIT_CAPS_FATAL))
    else:
        out.append("verdict: OK -- %d warning(s)." % n_warn)
    return "\n".join(out)


def enforce(report: CapsReport, logger: Optional[Any] = None,
            strict: bool = False) -> int:
    """Log the report and return the exit code: 0, or 78 when anything is FATAL.

    Deliberately does not exit: batch_dock owns process exit.  *strict* here is
    belt-and-braces -- it promotes the same codes :func:`validate_request` would
    have promoted, so ``enforce(strict=True)`` on a strictly-validated report is
    idempotent and on a permissively-validated one is not a silent downgrade.
    """
    log = logger or _LOG
    text = report.format_report()

    fatal_codes = {f.code for f in report.fatal}
    promoted = [f for f in report.warnings
                if strict and f.code in STRICT_FATAL_CODES]

    # The whole report goes to the log as well as to stderr. The findings alone
    # are not enough six months later: "why is this campaign running at --cpu
    # 24" is answered by the measured numbers, and they belong in
    # docking_automation.log, not only on a terminal nobody kept.
    log.info("gnina remote capability report:\n%s", text)

    for finding in report.fatal:
        log.critical("gnina caps FATAL [%s] %s Remedy: %s",
                     finding.code, finding.message, finding.remedy)
    for finding in report.warnings:
        if finding in promoted:
            log.critical("gnina caps FATAL [%s] (--strict-caps) %s Remedy: %s",
                         finding.code, finding.message, finding.remedy)
        else:
            log.warning("gnina caps WARN [%s] %s Remedy: %s",
                        finding.code, finding.message, finding.remedy)

    sys.stderr.write(text + "\n")
    sys.stderr.flush()

    if fatal_codes or promoted:
        log.critical("remote %s cannot serve this gnina configuration; "
                     "refusing before stage 1 rather than failing job by job.",
                     report.caps.host)
        return EXIT_CAPS_FATAL
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dock_selectivity.remote_caps",
        description="Probe a remote GPU host and judge a gnina configuration "
                    "against it. Exits %d when the remote cannot serve it."
                    % EXIT_CAPS_FATAL)
    parser.add_argument("--host", help="user@host of the remote GPU machine")
    parser.add_argument("--port", type=int, default=None,
                        help="ssh port (default: 22)")
    parser.add_argument("--key", default=None, help="ssh private key")
    parser.add_argument("--cpu", type=int, default=60,
                        help="requested gnina --cpu (default: NUM_THREADS, 60)")
    parser.add_argument("--exhaustiveness", type=int, default=32,
                        help="requested gnina exhaustiveness (default: %(default)s)")
    parser.add_argument("--num-modes", type=int, default=None,
                        help="omit to keep consensus_docker's default of 20")
    parser.add_argument("--deadline", type=float, default=1800.0,
                        help="per-job deadline in seconds (default 1800)")
    parser.add_argument("--image", default=DEFAULT_IMAGE,
                        help="container image checked for on the remote (default: %(default)s)")
    parser.add_argument("--native", default=None,
                        help="native gnina path on the remote; skips the container")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT,
                        help="remote scratch directory, for the free-space check")
    parser.add_argument("--gnina-path", default="",
                        help="what GNINA_PATH points at (checked when --async)")
    parser.add_argument("--async", dest="async_mode", action="store_true",
                        help="validate for the decoupled queue (requires the shim)")
    parser.add_argument("--caps-file", default=None,
                        help="load a canned capability record; never probes")
    parser.add_argument("--cache", default=None,
                        help="cache path (default: no caching)")
    parser.add_argument("--ttl", type=float, default=DEFAULT_CAPS_TTL_SECONDS,
                        help="seconds a cached probe stays valid (default: %(default)s)")
    parser.add_argument("--json", action="store_true",
                        help="print the report as JSON instead of text")
    parser.add_argument("--strict", action="store_true",
                        help="oversubscribing the remote's cores becomes fatal")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.caps_file and not args.host:
        parser.error("--host is required unless --caps-file is given")

    try:
        if args.caps_file:
            caps = load_caps(args.caps_file)
        else:
            caps = get_caps(args.host, port=args.port, key=args.key,
                            cache_path=args.cache, ttl_seconds=args.ttl,
                            image=args.image, native=args.native,
                            remote_root=args.remote_root)
    except RemoteCapsError as exc:
        sys.stderr.write("FATAL [PROBE_FAILED] %s\n" % exc)
        return EXIT_CAPS_FATAL

    request = DockRequest(cpu=args.cpu, exhaustiveness=args.exhaustiveness,
                          num_modes=args.num_modes,
                          job_deadline_seconds=args.deadline, image=args.image,
                          native=args.native, gnina_path=args.gnina_path,
                          async_mode=args.async_mode)
    report = validate_request(caps, request, strict=args.strict)

    if args.json:
        sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
        return EXIT_CAPS_FATAL if not report.ok else 0

    return enforce(report, strict=args.strict)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
