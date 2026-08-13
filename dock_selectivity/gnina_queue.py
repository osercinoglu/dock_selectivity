"""Durable job-state store and submitter for decoupled remote gnina.

Why this module exists
----------------------

``batch_dock.py`` runs five stages in sequence and gnina is the only one that
does not run on this machine: it goes over SSH to a single-GPU box.  Blocking the
whole local pipeline on a serial remote GPU wastes hours, so gnina is decoupled --
enqueued before stage 1, worked by a separate pump process
(``gnina_collect.py``), and waited for by a barrier immediately before stage 5.

That turns one blocking call into a distributed protocol with three writers (the
pump, the barrier, an operator) and two crash domains (this machine and the GPU
box).  Everything in this file exists to make that protocol survive a crash on
either side **without ever docking a job twice or losing a finished one**:

* one JSON file per job, written with a temp-file-plus-``os.replace`` discipline,
  so a torn write is impossible and readers need no lock;
* an explicit legal-transition table -- an illegal transition raises rather than
  quietly corrupting the record;
* :meth:`GninaQueue.reconcile`, which rebuilds state from *disk first, remote
  second, record last*, because the record is the only one of the three that a
  crash can truncate;
* a local pump mutex with stale-lock recovery, so a crashed pump does not
  deadlock the campaign forever;
* and the invariant that at most one job is ever outstanding on the remote,
  which is what "one GPU" means.

Split-at-the-binary
-------------------

``consensus_docker.py`` is invoked **twice** per gnina job and is never modified.
The split point is the gnina *binary* boundary, inside ``gnina_remote.py``,
selected by ``GNINA_REMOTE_MODE``:

``submit``
    ``consensus_docker`` builds the argv (search box from the pocket, ``--cpu``,
    ``--exhaustiveness``) and calls ``gnina_remote``, which uploads, takes the
    remote GPU mutex, launches the detached container, writes a ticket and exits
    without producing ``out.sdf``.  ``consensus_docker`` then fails on the absent
    file and writes nothing.  The **ticket** is the signal; its exit code is not.

``collect``
    The identical command runs again.  ``gnina_remote`` downloads
    ``out.sdf``/``out.log`` and exits 0, and ``consensus_docker`` runs
    ``split_mol`` / ``make_complex`` / ``parse_gnina`` -- the *same lines of code*
    that produce today's poses and CNN scores.  No pose or scoring logic is
    reimplemented here, so the scientific output cannot drift.

Three rules make that safe, and each has a test:

R1  Both phases pass ``--skip_rmsd``, always.  ``consensus_docker`` computes a
    consensus whenever >= 2 tools have a ``results.csv``; without the flag the
    submit phase would write a gnina-less ``final_results.csv`` that
    ``check_final_results_completion`` would then report as finished.
R2  ``<job>/gnina/`` outputs are purged before submit *and* before collect.  A
    surviving ``out.sdf`` would let the submit phase's ``split_mol`` succeed on
    stale bytes and write a ``results.csv`` from the previous parameter set --
    a wrong result that nothing downstream can detect.
R3  After every submit, ``<job>/gnina/results.csv`` must not exist.  If it does,
    the exit contract or the purge is broken; the job is abandoned loudly.  This
    catches a future change to ``consensus_docker`` on job #1, not job #5000.

Deliberate deviations from the design spec (all disclosed, none silent)
-----------------------------------------------------------------------

The spec's transition table and its reconciliation rules disagree in three
places.  The reconciliation rules win, because they are the ones that describe
crash recovery, and the extra edges are collected in
:data:`_RECONCILE_TRANSITIONS` so the divergence is auditable:

* ``PENDING -> REMOTE_DONE`` (reconcile rule 6: a PENDING record with a prior
  remote directory that has finished must not be re-docked);
* ``SUBMITTED/RUNNING -> PENDING`` (reconcile rules 4 and 5: no remote directory
  was ever resolved, or it is GONE);
* ``SUBMITTING -> RUNNING`` (reconcile rule 5 applies to SUBMITTING too, and a
  probe that finds the remote process alive must be able to record that);
* ``* -> COLLECTED`` from every non-terminal state (reconcile rule 1: complete
  outputs on disk outrank any recorded state -- "never lose a finished job").

The record also carries an ``error_counts`` map that the spec's schema omits but
its failure table requires: three error classes are documented as "one retry,
then abandon regardless of ``max_attempts``", and that cannot be decided without
counting occurrences per code.

This module imports nothing from ``batch_dock`` -- importing it configures
logging and writes ``docking_automation.log`` as a side effect -- and nothing
from the science stack (pandas, rdkit, openbabel), so the pump starts in well
under a second.
"""

from __future__ import annotations

import errno
import glob as _glob
import json
import logging
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

try:                                   # Linux everywhere in this pipeline; the
    import fcntl                       # fallback path exists for mounts that do
except ImportError:                    # not implement flock at all (some FUSE).
    fcntl = None                       # type: ignore

LOG = logging.getLogger(__name__)

SCHEMA_VERSION = 1
QUEUE_DIRNAME = "gnina_queue"

#: Where job directories live under a run directory.  Used only to re-derive a
#: job directory for a record whose file was truncated by a crash (reconcile
#: rule 2), never to *find* jobs -- those always arrive through ``enqueue``.
DOCKING_RESULTS_DIRNAME = "docking_results"

ALL_STATES = (
    "PENDING",
    "SUBMITTING",
    "SUBMITTED",
    "RUNNING",
    "REMOTE_DONE",
    "COLLECTING",
    "COLLECTED",
    "FAILED",
    "ABANDONED",
)
TERMINAL_STATES = frozenset({"COLLECTED", "ABANDONED"})

#: The pseudo-state a record occupies before its file exists.  Keeping it in the
#: same table as the real states means ``enqueue`` is validated by exactly the
#: same code path as every other transition.
NEW_STATE = "(new)"

#: The transition table exactly as specified.  Anything not in here raises.
_SPEC_TRANSITIONS = {
    NEW_STATE: frozenset({"PENDING", "COLLECTED"}),
    "PENDING": frozenset({"SUBMITTING", "COLLECTED"}),
    "SUBMITTING": frozenset({"SUBMITTED", "REMOTE_DONE", "PENDING", "FAILED",
                             "ABANDONED"}),
    "SUBMITTED": frozenset({"RUNNING", "REMOTE_DONE", "FAILED"}),
    "RUNNING": frozenset({"REMOTE_DONE", "FAILED"}),
    "REMOTE_DONE": frozenset({"COLLECTING"}),
    "COLLECTING": frozenset({"COLLECTED", "REMOTE_DONE", "FAILED"}),
    "COLLECTED": frozenset(),
    "FAILED": frozenset({"PENDING", "REMOTE_DONE", "ABANDONED"}),
    "ABANDONED": frozenset(),
}

#: Edges the spec's *reconciliation rules* require but its transition table
#: omits.  Kept separate so a reader can see precisely where this implementation
#: goes beyond the table, and so a test can assert the table above is intact.
_RECONCILE_TRANSITIONS = {
    # rule 6: a PENDING record whose recorded remote directory has finished must
    # be collected, never re-docked.
    "PENDING": frozenset({"REMOTE_DONE"}),
    # rules 4 and 5: the intent record exists but no remote directory was ever
    # resolved, or the remote scratch is gone.
    "SUBMITTED": frozenset({"PENDING"}),
    "RUNNING": frozenset({"PENDING"}),
    # rule 1: complete outputs on disk outrank every recorded state.  Rule 5 also
    # applies to SUBMITTING, and a probe that finds the process alive has to be
    # able to say so -- otherwise a record whose ticket was lost mid-launch would
    # sit in SUBMITTING while the GPU is visibly working.
    "SUBMITTING": frozenset({"COLLECTED", "RUNNING"}),
    "REMOTE_DONE": frozenset({"COLLECTED"}),
    "FAILED": frozenset({"COLLECTED"}),
}
for _state in ("SUBMITTED", "RUNNING"):
    _RECONCILE_TRANSITIONS[_state] = _RECONCILE_TRANSITIONS[_state] | frozenset({"COLLECTED"})

LEGAL_TRANSITIONS = {}  # type: Dict[str, FrozenSet[str]]
for _state in list(_SPEC_TRANSITIONS):
    LEGAL_TRANSITIONS[_state] = _SPEC_TRANSITIONS[_state] | _RECONCILE_TRANSITIONS.get(
        _state, frozenset()
    )
del _state

#: Error classes a retry can plausibly fix (spec section 11).  Anything not in
#: here is permanent: the record goes straight to ABANDONED with the error kept.
RETRIABLE_ERRORS = frozenset({
    "SSH_TRANSIENT",
    "REMOTE_BUSY",
    "REMOTE_GNINA_FAILED",
    "STALLED",
    "REMOTE_DIR_GONE",
    "EMPTY_OUTPUT",
    "POSTPROCESS_FAILED",
    "COLLECT_TIMEOUT",
    "SUBMIT_NO_TICKET",
})

#: "yes (1)" in the failure table: one retry, then abandon regardless of
#: ``max_attempts``.  A second identical remote failure is a property of the
#: inputs, not of the link, and re-docking it burns GPU time to learn nothing.
ONE_RETRY_ERRORS = frozenset({
    "REMOTE_GNINA_FAILED",
    "EMPTY_OUTPUT",
    "POSTPROCESS_FAILED",
})

#: Where a retry resumes.  The default is PENDING (re-dock).  The exceptions are
#: the failures that happened *after* the GPU produced a good result: re-docking
#: those would throw away work the remote already did.
RETRY_TARGET_STATE = {
    "POSTPROCESS_FAILED": "REMOTE_DONE",
    "COLLECT_TIMEOUT": "REMOTE_DONE",
}

#: Backoff after the remote GPU mutex was held by someone else.  A busy submit is
#: not a failure and must not consume an attempt, or a long-running neighbour
#: would abandon the whole queue.
BUSY_BACKOFF_SECONDS = (5.0, 15.0, 30.0, 60.0)
#: Backoff before re-docking a failed job, indexed by attempts already consumed.
RETRY_BACKOFF_SECONDS = (0.0, 30.0, 120.0, 300.0)

#: The largest gap that can legitimately separate ``now`` from a record's
#: ``next_attempt_at``.  It exists to catch a clock-domain mismatch: this module
#: always writes that field using the queue's own clock, so a caller that passes
#: ``time.monotonic()`` to :meth:`GninaQueue.next_pending` while the queue was
#: built with the default ``time.time`` would compare an uptime against an epoch
#: and never see a backoff expire -- every job that hit a busy GPU or a retry
#: would sit PENDING for the rest of the campaign, silently.  Ten minutes past
#: the longest backoff this module ever sets.
MAX_BACKOFF_SECONDS = 900.0

#: Transient SSH failures are capped separately from docking attempts: a flaky
#: link must not exhaust the (small) number of times we are willing to re-dock.
MAX_TRANSIENT_ATTEMPTS = 10

#: Grace on top of ``job_deadline_seconds`` before a launched-but-silent remote
#: job is declared STALLED.  Generous on purpose: killing a job that is merely
#: slow costs a full re-dock.
STALL_GRACE_SECONDS = 600.0

#: A pump lock whose owner cannot be judged (a different host, or a body a crash
#: truncated) is broken only after this long.  Deliberately a full day: the
#: fallback path is only correct for same-host pumps, so breaking a foreign lock
#: is a last resort against a permanent deadlock, not a routine event.
PUMP_LOCK_STALE_SECONDS = float(os.environ.get("GNINA_PUMP_LOCK_STALE", "86400"))

#: History is diagnostic, not an audit log; an unbounded list would grow the
#: record on a mount where small files are the whole point.
HISTORY_LIMIT = 50
#: Job-id lists in ``summary.json`` are capped so a 10^4-job campaign does not
#: write a megabyte of JSON on every barrier poll.
SUMMARY_ID_LIMIT = 200

#: The files ``purge_gnina_outputs`` removes, and nothing else, inside
#: ``<job>/gnina/`` only.
PURGE_EXACT_NAMES = ("out.sdf", "out.log", "results.csv")
PURGE_GLOBS = ("out_*.sdf", "complex_*.pdb", "complex_*.pdb.gz")

#: Params that define *what is docked*.  Changing one mid-campaign would make
#: half the results incomparable, so it is refused unless explicitly allowed.
#: Operational knobs (deadlines, attempt counts) are deliberately excluded.
SIGNIFICANT_PARAM_KEYS = (
    "cpu",
    "exhaustiveness",
    "num_modes",
    "cutoff_value",
    "consensus_docker_script",
    "gnina_path",
    "image",
    "native",
    "host",
    "port",
    "remote_root",
)

PROBE_STATES = ("GONE", "LAUNCHED", "RUNNING", "DONE")


class QueueError(RuntimeError):
    """The queue was asked to do something it cannot do safely."""


class IllegalTransition(QueueError):
    """A state change that is not in :data:`LEGAL_TRANSITIONS`.

    Raised rather than accepted: a queue that silently allows
    ``COLLECTED -> PENDING`` would re-dock finished work, and a queue that allows
    ``PENDING -> SUBMITTED`` would record a launch that never happened.
    """


# ---------------------------------------------------------------------------
# time, JSON and small filesystem helpers
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    """ISO-8601 UTC with a trailing ``Z`` and microseconds.

    One helper, used everywhere, so records from the pump, the barrier and an
    operator's ``--requeue`` sort correctly against each other as plain strings.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    """Write *obj* to *path* so a reader sees the old file or the new one, never a mix.

    The discipline is temp-file-in-the-same-directory, ``fsync``, ``os.replace``.
    ``os.replace`` is atomic on POSIX, which is what lets every reader (the
    barrier, ``--status``, another queue object) work with no lock at all.

    Two constraints come from the deployment rather than from taste: the run
    directory may sit on a fuseblk external drive, so the directory ``fsync`` is
    best-effort (some FUSE mounts refuse it) and no hardlink is ever used
    (``os.link`` is not universally available there).
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    tmp = "%s.tmp.%d.%d" % (path, os.getpid(), time.monotonic_ns())
    payload = (json.dumps(obj, sort_keys=True, indent=1) + "\n").encode("utf-8")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(directory)


def _fsync_dir(directory: str) -> None:
    """Best-effort directory fsync; a FUSE mount that refuses it is not an error."""
    try:
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        LOG.debug("directory fsync unsupported on %s; continuing", directory)
    finally:
        os.close(fd)


def read_json_retry(path: str, attempts: int = 3, delay: float = 0.05
                    ) -> Optional[Dict[str, Any]]:
    """Read JSON, tolerating the microscopic window where a record is being created.

    ``enqueue`` claims a record name with ``O_EXCL`` before replacing it with the
    real bytes, so a reader can land on a zero-byte file.  That is "not yet",
    not "corrupt" -- retry briefly, then give up and let reconciliation re-derive
    the record.  Returning ``None`` rather than raising is what keeps every
    reader lock-free.
    """
    for attempt in range(max(1, attempts)):
        try:
            with open(path, "rb") as handle:
                raw = handle.read()
        except FileNotFoundError:
            return None
        except OSError as exc:
            LOG.debug("cannot read %s: %s", path, exc)
            return None
        if raw.strip():
            try:
                loaded = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                loaded = None
            if isinstance(loaded, dict):
                return loaded
        if attempt < max(1, attempts) - 1:
            time.sleep(delay)
    return None


def _write_create_if_absent(path: str, obj: Dict[str, Any]) -> bool:
    """Create *path* only if a readable record is not already there.  True if we did.

    Never ``os.replace`` alone: that would reset a live SUBMITTED record to
    PENDING every time a campaign is restarted, and the job would be docked
    twice.  The name is claimed with ``O_EXCL`` first; the loser of the race
    reads what the winner wrote.

    The one case where an existing file *is* overwritten is a file that cannot
    be parsed after retries -- the zero-byte remains of a crash between claiming
    the name and writing the bytes.  That is not a record anyone can lose, and
    leaving it would make the job invisible to the pump forever.
    """
    if os.path.exists(path):
        if read_json_retry(path) is not None:
            return False
        LOG.warning("completing an abandoned record claim at %s: the file exists "
                    "but holds no readable JSON, so a previous process died "
                    "between claiming the name and writing it.", path)
        atomic_write_json(path, obj)
        return True
    directory = os.path.dirname(os.path.abspath(path)) or "."
    tmp = "%s.tmp.%d.%d" % (path, os.getpid(), time.monotonic_ns())
    payload = (json.dumps(obj, sort_keys=True, indent=1) + "\n").encode("utf-8")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
    except FileExistsError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)
    _fsync_dir(directory)
    return True


def _makedirs(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise QueueError("cannot create queue directory %s: %s" % (path, exc))


# ---------------------------------------------------------------------------
# job-directory predicates
# ---------------------------------------------------------------------------

def queue_root(run_dir: str, queue_dir: Optional[str] = None) -> str:
    """Absolute path of the queue directory for *run_dir*.

    An explicit argument wins over ``GNINA_QUEUE_DIR``, which wins over
    ``<run_dir>/gnina_queue``.  The environment variable exists because the run
    directory is often on the external fuseblk drive, where thousands of small
    files are slow; pointing the queue at ext4 costs nothing and is much faster.
    """
    if queue_dir:
        return os.path.abspath(queue_dir)
    env = os.environ.get("GNINA_QUEUE_DIR")
    if env:
        return os.path.abspath(env)
    return os.path.join(os.path.abspath(run_dir), QUEUE_DIRNAME)


def job_id_for(job_dir: str) -> str:
    """The job directory's basename, e.g. ``DB00035__P30518__AVPR2__F1__cav1``.

    Already unique per job (it is built by ``dock_selectivity.jobid``) and
    filesystem-safe, so it doubles as the record filename with no encoding.
    """
    return os.path.basename(os.path.normpath(job_dir))


def _gnina_dir(job_dir: str) -> str:
    return os.path.join(job_dir, "gnina")


def _nonempty(path: str) -> bool:
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


def _csv_data_rows(path: str, limit_bytes: int = 1 << 20) -> int:
    """Number of data rows (lines after the header) in a small CSV. -1 if unreadable."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read(limit_bytes)
    except OSError:
        return -1
    lines = [ln for ln in raw.decode("utf-8", "replace").splitlines() if ln.strip()]
    if not lines:
        return 0
    return len(lines) - 1


def outputs_complete(job_dir: str) -> bool:
    """True if this job's gnina result is on local disk and usable.

    Deliberately does **not** *require* ``complex_*.pdb``: the campaign's default
    cleanup deletes those after a successful stage 5, and requiring them would
    make the pump re-dock every finished job on the second run of a campaign --
    thousands of GPU-hours to reproduce files it is about to delete again.
    ``results.csv`` (the CNN scores) plus a non-empty ``out.sdf`` (the poses) is
    what "gnina is done for this job" means.

    But when complexes ARE present they must agree with the poses and the score
    rows.  A crash part-way through ``parse_gnina``, or part-way through the R2
    purge, leaves a directory with (say) 20 complexes and 3 score rows; without
    this check reconciliation promotes it to COLLECTED, nothing ever looks at it
    again, and stage 5 computes a consensus from a fraction of the poses -- a
    wrong scientific answer that no exit code reports.  Re-collecting instead is
    cheap and idempotent: the poses are still on the GPU box, so it costs a
    download rather than a re-dock.

    The two conditions do not conflict, because cleanup removes *all* the
    complexes: zero complexes means "cleaned up", not "torn".
    """
    gnina = _gnina_dir(job_dir)
    results = os.path.join(gnina, "results.csv")
    if not _nonempty(results):
        return False
    if _csv_data_rows(results) < 1:          # header alone is not a result
        return False
    if not _nonempty(os.path.join(gnina, "out.sdf")):
        return False
    detail = outputs_detail(job_dir)
    if detail["n_complexes"]:
        return (detail["n_complexes"] == detail["n_result_rows"]
                == detail["n_poses"])
    return True


def outputs_detail(job_dir: str) -> Dict[str, Any]:
    """Counts used for verification and for the record's ``outputs`` block."""
    gnina = _gnina_dir(job_dir)
    poses = [p for p in _glob.glob(os.path.join(gnina, "out_*.sdf"))]
    complexes = [p for p in _glob.glob(os.path.join(gnina, "complex_*.pdb"))]
    complexes += [p for p in _glob.glob(os.path.join(gnina, "complex_*.pdb.gz"))]
    out_sdf = os.path.join(gnina, "out.sdf")
    results = os.path.join(gnina, "results.csv")
    try:
        out_bytes = os.path.getsize(out_sdf)
    except OSError:
        out_bytes = 0
    return {
        "n_poses": len(poses),
        "n_complexes": len(complexes),
        "n_result_rows": _csv_data_rows(results) if os.path.exists(results) else 0,
        "out_sdf_bytes": out_bytes,
    }


def outputs_verified_at_collect(job_dir: str) -> Tuple[bool, Dict[str, Any]]:
    """The strict predicate, used only on the COLLECTING -> COLLECTED edge.

    At that moment nothing has cleaned up yet, so every artefact
    ``consensus_docker`` produces must be present and self-consistent: one
    ``complex_N.pdb`` per split pose, and one ``results.csv`` row per pose.  A
    mismatch means ``split_mol`` / ``make_complex`` / ``parse_gnina`` disagreed
    about how many poses there are, which is exactly the silent half-write that
    would otherwise reach the consensus RMSD.
    """
    detail = outputs_detail(job_dir)
    reasons = []
    if not outputs_complete(job_dir):
        reasons.append("results.csv or out.sdf missing/empty")
    if detail["n_complexes"] < 1:
        reasons.append("no complex_*.pdb was written")
    if detail["n_complexes"] != detail["n_poses"]:
        reasons.append("%d complex_*.pdb vs %d out_*.sdf"
                       % (detail["n_complexes"], detail["n_poses"]))
    if detail["n_result_rows"] != detail["n_complexes"]:
        reasons.append("%d results.csv rows vs %d complex_*.pdb"
                       % (detail["n_result_rows"], detail["n_complexes"]))
    detail = dict(detail)
    detail["reasons"] = reasons
    return (not reasons), detail


def purge_gnina_outputs(job_dir: str) -> int:
    """Remove stale gnina artefacts before a submit or a collect (rule R2).

    This is the single most important line of defence in the design.  If
    ``<job>/gnina/out.sdf`` survives from an earlier attempt, the *submit*
    phase's ``split_mol`` succeeds on those stale bytes and ``parse_gnina``
    writes a ``results.csv`` from the previous parameter set.  Nothing
    downstream can tell: ``check_tool_completion`` only checks that the file
    exists and looks like CSV.

    Scope is deliberately narrow -- named files inside ``<job>/gnina/`` only.
    Never the directory itself, never ``input/``, never ``final_results.csv``,
    never another tool's directory.  Returns the number of files removed.
    """
    gnina = _gnina_dir(job_dir)
    if not os.path.isdir(gnina):
        return 0
    victims = []  # type: List[str]
    for name in PURGE_EXACT_NAMES:
        candidate = os.path.join(gnina, name)
        if os.path.isfile(candidate):
            victims.append(candidate)
    for pattern in PURGE_GLOBS:
        for candidate in _glob.glob(os.path.join(gnina, pattern)):
            if os.path.isfile(candidate):
                victims.append(candidate)
    removed = 0
    for path in sorted(set(victims)):
        try:
            os.unlink(path)
            removed += 1
        except OSError as exc:
            LOG.warning("could not purge %s: %s", path, exc)
    if removed:
        LOG.debug("purged %d stale gnina file(s) from %s", removed, gnina)
    return removed


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

def _default_inputs() -> Dict[str, Any]:
    return {
        "receptor_pdb": None,
        "ligand_sdf": None,
        "pocket_pdb": None,
        "receptor_pdbqt": None,
        "skip_pdb_to_pdbqt": False,
        "receptor_sha256": None,
        "ligand_sha256": None,
    }


def _default_remote() -> Dict[str, Any]:
    return {
        "host": None,
        "port": None,
        "dir": None,
        "job_key": None,
        "image": None,
        "launched_at": None,
        "exit_code": None,
        "lock_pid": None,
        "remote_seconds": None,
    }


def _default_collect() -> Dict[str, Any]:
    return {"pid": None, "host": None, "started_at": None, "finished_at": None}


def _default_outputs() -> Dict[str, Any]:
    return {"n_poses": None, "n_complexes": None, "out_sdf_bytes": None}


@dataclass
class JobRecord:
    """One gnina job's durable state.  One writer at a time, many lock-free readers."""

    job_id: str
    job_dir: str
    state: str = "PENDING"
    schema_version: int = SCHEMA_VERSION
    attempts: int = 0
    transient_attempts: int = 0
    next_attempt_at: Optional[float] = None
    created_at: str = ""
    updated_at: str = ""
    inputs: Dict[str, Any] = field(default_factory=_default_inputs)
    params: Dict[str, Any] = field(default_factory=dict)
    remote: Dict[str, Any] = field(default_factory=_default_remote)
    collect: Dict[str, Any] = field(default_factory=_default_collect)
    outputs: Dict[str, Any] = field(default_factory=_default_outputs)
    error: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    #: Occurrences per error code.  Not in the spec's schema, but its failure
    #: table needs it: three classes are "one retry, then abandon regardless of
    #: max_attempts", which cannot be decided from ``attempts`` alone.
    error_counts: Dict[str, int] = field(default_factory=dict)
    #: Unknown top-level keys, preserved verbatim on rewrite so a newer writer's
    #: fields survive an older reader.
    extra: Dict[str, Any] = field(default_factory=dict)

    _KNOWN = (
        "job_id", "job_dir", "state", "schema_version", "attempts",
        "transient_attempts", "next_attempt_at", "created_at", "updated_at",
        "inputs", "params", "remote", "collect", "outputs", "error", "history",
        "error_counts",
    )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JobRecord":
        known = dict(d)
        extra = {k: v for k, v in d.items() if k not in cls._KNOWN}
        for key in list(extra):
            known.pop(key, None)
        inputs = dict(_default_inputs())
        inputs.update(known.get("inputs") or {})
        remote = dict(_default_remote())
        remote.update(known.get("remote") or {})
        collect = dict(_default_collect())
        collect.update(known.get("collect") or {})
        outputs = dict(_default_outputs())
        outputs.update(known.get("outputs") or {})
        return cls(
            job_id=known.get("job_id") or "",
            job_dir=known.get("job_dir") or "",
            state=known.get("state") or "PENDING",
            schema_version=int(known.get("schema_version") or SCHEMA_VERSION),
            attempts=int(known.get("attempts") or 0),
            transient_attempts=int(known.get("transient_attempts") or 0),
            next_attempt_at=known.get("next_attempt_at"),
            created_at=known.get("created_at") or "",
            updated_at=known.get("updated_at") or "",
            inputs=inputs,
            params=dict(known.get("params") or {}),
            remote=remote,
            collect=collect,
            outputs=outputs,
            error=known.get("error"),
            history=list(known.get("history") or []),
            error_counts=dict(known.get("error_counts") or {}),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.extra)
        out.update({
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "job_dir": self.job_dir,
            "state": self.state,
            "attempts": self.attempts,
            "transient_attempts": self.transient_attempts,
            "next_attempt_at": self.next_attempt_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "inputs": dict(self.inputs),
            "params": dict(self.params),
            "remote": dict(self.remote),
            "collect": dict(self.collect),
            "outputs": dict(self.outputs),
            "error": None if self.error is None else dict(self.error),
            "history": list(self.history),
            "error_counts": dict(self.error_counts),
        })
        return out

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def is_outstanding(self) -> bool:
        """FAILED counts as outstanding: the pump may still retry it."""
        return self.state not in TERMINAL_STATES

    @property
    def error_code(self) -> Optional[str]:
        if not self.error:
            return None
        return self.error.get("code")


def make_error(code: str, message: str, phase: str,
               retriable: Optional[bool] = None) -> Dict[str, Any]:
    """Build an error block.  ``retriable`` defaults to the failure table."""
    if retriable is None:
        retriable = code in RETRIABLE_ERRORS
    return {
        "code": code,
        "message": message,
        "phase": phase,
        "retriable": bool(retriable),
        "at": _utcnow_iso(),
    }


# ---------------------------------------------------------------------------
# the pump lock
# ---------------------------------------------------------------------------

class PumpLock:
    """Local single-pump mutex.  Release is idempotent and safe to call twice.

    Two implementations, chosen by what the filesystem supports:

    ``flock``
        The kernel drops the lock when the holder dies, so a crashed pump leaves
        nothing stale to clean up.  The file is never unlinked -- unlinking a
        file another process holds open would let two pumps both "hold" it.

    ``excl``
        An ``O_EXCL`` lock file, for mounts with no ``flock``.  Correct only for
        same-host pumps, which is the deployment; staleness is judged by
        ``kill(pid, 0)`` plus a hostname match.
    """

    def __init__(self, path: str, fd: int, mode: str, body: Dict[str, Any]):
        self.path = path
        self.mode = mode
        self.body = body
        self._fd = fd
        self._released = False

    def __enter__(self) -> "PumpLock":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self.mode == "flock":
            # Leave a "released" marker rather than an empty file so a later
            # fallback-mode pump can tell a clean handover from a truncated body.
            try:
                os.lseek(self._fd, 0, os.SEEK_SET)
                os.ftruncate(self._fd, 0)
                marker = dict(self.body)
                marker["released"] = True
                marker["released_at"] = _utcnow_iso()
                os.write(self._fd, (json.dumps(marker, sort_keys=True) + "\n").encode())
            except OSError:
                pass
            try:
                if fcntl is not None:
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self._fd)
            except OSError:
                pass
        else:
            try:
                os.close(self._fd)
            except OSError:
                pass
            try:
                os.unlink(self.path)
            except OSError:
                pass


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                      # exists, owned by someone else
    except (OSError, ValueError, TypeError):
        return False
    return True


def _hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


def _lock_body(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    return {
        "pid": os.getpid(),
        "host": _hostname(),
        "started_at": _utcnow_iso(),
        "argv": list(argv if argv is not None else sys.argv),
    }


def _read_lock_body(path: str) -> Optional[Dict[str, Any]]:
    body = read_json_retry(path, attempts=1)
    return body if isinstance(body, dict) else None


def _lock_is_breakable(path: str, body: Optional[Dict[str, Any]],
                       stale_seconds: float, now: float) -> Tuple[bool, str]:
    """Decide whether an existing fallback lock file may be removed.

    Never deadlocking forever is a hard requirement -- a pump that was SIGKILLed
    must not strand the campaign -- but breaking a live pump's lock would put two
    submitters on one GPU.  The judgement, most trustworthy evidence first:

    1. the holder said it released cleanly            -> break;
    2. same host and its pid is gone                  -> break, immediately;
    3. same host and its pid is alive                 -> keep;
    4. anything else (foreign host, truncated body)   -> break only after
       ``stale_seconds``, and say so loudly, because that is the one case where
       the evidence is genuinely inconclusive.
    """
    if body and body.get("released"):
        return True, "previous holder released it"
    host = (body or {}).get("host")
    pid = (body or {}).get("pid")
    if body and host == _hostname() and pid:
        if _pid_alive(pid):
            return False, "held by live pid %s on this host" % pid
        return True, "holder pid %s on this host is gone" % pid
    try:
        age = now - os.path.getmtime(path)
    except OSError:
        return True, "lock file vanished"
    if age > stale_seconds:
        return True, ("unjudgeable owner (%s) and %.0fs old, over the %.0fs stale "
                      "threshold" % (host or "unknown", age, stale_seconds))
    return False, ("held by %s, %.0fs old; not stale until %.0fs"
                   % (host or "unknown", age, stale_seconds))


# ---------------------------------------------------------------------------
# gnina_remote seam
# ---------------------------------------------------------------------------

_GNINA_REMOTE = None


def _gnina_remote():
    """Import ``gnina_remote`` lazily, from the repo root if it is not on the path.

    Lazy on purpose: importing this module must never require the shim to be
    importable, so ``batch_dock`` and the test suite keep working while the shim
    is being extended in parallel.
    """
    global _GNINA_REMOTE
    if _GNINA_REMOTE is not None:
        return _GNINA_REMOTE
    try:
        import gnina_remote as module          # type: ignore
    except ImportError:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import gnina_remote as module          # type: ignore
    _GNINA_REMOTE = module
    return module


def build_probe_command(remote_dir: str) -> str:
    """One round trip that distinguishes GONE / LAUNCHED / RUNNING / DONE.

    Used only when ``gnina_remote.probe_job_state`` is not available; see
    :func:`default_prober`.
    """
    d = shlex.quote(remote_dir)
    return (
        'if [ ! -d %s ]; then echo GONE; '
        'elif [ -f %s/exit_code ]; then echo "DONE $(cat %s/exit_code 2>/dev/null)"; '
        'elif [ -s %s/run.log ]; then echo RUNNING; '
        'else echo LAUNCHED; fi' % (d, d, d, d)
    )


def parse_probe_state(text: str) -> Tuple[str, Optional[int]]:
    """Parse :func:`build_probe_command` output.  Never raises.

    Unrecognised output means "we do not know", which must not be mistaken for
    "the job is gone" -- that would re-dock a job that is still running.  It maps
    to LAUNCHED, the state that only waits.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "LAUNCHED", None
    last = lines[-1]
    parts = last.split()
    head = parts[0].upper()
    if head == "DONE":
        code = None
        if len(parts) > 1:
            try:
                code = int(parts[1])
            except ValueError:
                code = None
        if code is None:
            # "DONE" with an unreadable marker: treat as still running rather
            # than as a success we cannot substantiate.
            return "RUNNING", None
        return "DONE", code
    if head in ("GONE", "RUNNING", "LAUNCHED"):
        return head, None
    return "LAUNCHED", None


def default_prober(record: "JobRecord", retries: int = 3, timeout: float = 180.0
                   ) -> Tuple[str, Optional[int]]:
    """Ask the remote what happened to *record*'s job.

    Prefers ``gnina_remote.probe_job_state`` -- the shim owns the remote protocol
    and its version of this is authoritative.  The inline fallback exists only so
    the queue is usable before that function lands; it is built from
    ``ssh_base``/``run_resilient``, which the shim has always exported, and it
    becomes dead code the moment the shim grows ``probe_job_state``.
    """
    remote_dir = (record.remote or {}).get("dir")
    if not remote_dir:
        return "GONE", None
    module = _gnina_remote()
    host = (record.remote or {}).get("host") or os.environ.get("GNINA_REMOTE_HOST")
    port = (record.remote or {}).get("port") or os.environ.get("GNINA_REMOTE_PORT")
    key = os.environ.get("GNINA_REMOTE_KEY")
    native = getattr(module, "probe_job_state", None)
    if native is not None:
        return native(host, port, key, remote_dir, retries=retries)
    ssh = module.ssh_base(host, port, key)
    try:
        proc = module.run_resilient(ssh + [build_probe_command(remote_dir)],
                                    "probe %s" % record.job_id, retries,
                                    timeout=timeout)
    except Exception as exc:                     # noqa: BLE001 - classified below
        raise QueueError("probe failed for %s: %s" % (record.job_id, exc))
    return parse_probe_state(proc.stdout)


# ---------------------------------------------------------------------------
# the subprocess seam
# ---------------------------------------------------------------------------

def default_runner(cmd: List[str], env: Dict[str, str], timeout: float,
                   log_path: str, cwd: str) -> Tuple[int, str, str]:
    """Run one ``consensus_docker.py`` phase, appending its output to *log_path*.

    ``start_new_session=True`` puts the child in its own process group so a
    timeout can kill the whole tree.  Killing it does **not** kill the remote
    gnina job -- that is ``setsid``-detached on the GPU box, and its work
    continues; the next poll or collect picks it up.  That is intended: an
    aborted local process must never cost a finished GPU run.

    On timeout this raises :class:`subprocess.TimeoutExpired` after killing the
    group, rather than inventing a return code that could collide with a real
    one.  Callers distinguish "the phase failed" (a tuple) from "the phase never
    finished" (the exception).
    """
    _makedirs(os.path.dirname(os.path.abspath(log_path)) or ".")
    proc = subprocess.Popen(
        cmd, cwd=cwd or None, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        try:
            out, err = proc.communicate(timeout=10)
        except Exception:                        # noqa: BLE001
            out, err = "", ""
        _append_log(log_path, cmd, out, err, "TIMEOUT after %.0fs" % timeout)
        raise
    _append_log(log_path, cmd, out, err, "exit %s" % proc.returncode)
    return proc.returncode, out or "", err or ""


def _kill_group(proc) -> None:
    import signal
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except OSError:
        return
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:
        pass


def _append_log(log_path: str, cmd: Sequence[str], out: str, err: str,
                status: str) -> None:
    try:
        with open(log_path, "a", errors="replace") as handle:
            handle.write("\n===== %s  %s =====\n" % (_utcnow_iso(), status))
            handle.write("cmd: %s\n" % " ".join(cmd))
            if out:
                handle.write("--- stdout ---\n%s\n" % out)
            if err:
                handle.write("--- stderr ---\n%s\n" % err)
    except OSError as exc:
        LOG.warning("could not append to %s: %s", log_path, exc)


def _log_tail(path: str, lines: int = 40) -> str:
    try:
        with open(path, "r", errors="replace") as handle:
            content = handle.readlines()
    except OSError:
        return ""
    return "".join(content[-lines:]).strip()


# ---------------------------------------------------------------------------
# command and environment construction
# ---------------------------------------------------------------------------

def build_consensus_command(record: JobRecord, params: Dict[str, Any]) -> List[str]:
    """The one place a ``consensus_docker.py`` argv is built for a gnina job.

    Both phases run the *identical* command; only the environment differs.  That
    is what makes the collect phase reproduce the submit phase's search box: the
    argv is rebuilt from the same pocket file by the same code.

    ``--skip_rmsd`` is unconditional (rule R1).  ``consensus_docker`` computes a
    consensus RMSD whenever two tools have a ``results.csv``, so without the flag
    the submit phase would write a ``final_results.csv`` from smina + GOLD alone
    and the job would look finished with no gnina in it.  RMSD is computed
    exactly once, in stage 5, after the barrier.

    ``--num_modes`` is omitted when ``params["num_modes"]`` is None, preserving
    ``consensus_docker``'s default of 20 -- which is what today's runs use.
    Passing it explicitly would be a silent change to the science.
    """
    inputs = record.inputs or {}
    for key in ("receptor_pdb", "ligand_sdf", "pocket_pdb"):
        if not inputs.get(key):
            raise QueueError("job %s has no %s recorded; cannot build a command"
                             % (record.job_id, key))
    cmd = [
        str(params.get("python_executable") or "python"),
        str(params["consensus_docker_script"]),
        "--use_gnina",
        "--overwrite",
        "--skip_rmsd",
        "--outfolder", record.job_dir,
        "--gnina_path", str(params["gnina_path"]),
        "--ligand_sdf", str(inputs["ligand_sdf"]),
        "--receptor_pdb", str(inputs["receptor_pdb"]),
        "--pocket_pdb", str(inputs["pocket_pdb"]),
        "--num_threads", str(params["cpu"]),
        "--cutoff_value", str(params.get("cutoff_value", -7.0)),
        "--exhaustiveness", str(params["exhaustiveness"]),
    ]
    if params.get("num_modes") is not None:
        cmd += ["--num_modes", str(params["num_modes"])]
    if inputs.get("receptor_pdbqt"):
        cmd += ["--receptor_pdbqt", str(inputs["receptor_pdbqt"])]
    if inputs.get("skip_pdb_to_pdbqt"):
        cmd.append("--skip_pdb_to_pdbqt")
    return cmd


def build_phase_env(record: JobRecord, params: Dict[str, Any], phase: str,
                    base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Environment for one phase.  The phase is selected here and nowhere else.

    Only variables the queue actually decides are set; everything else is
    inherited, so an operator's existing ``GNINA_REMOTE_*`` settings still apply
    where the queue has no opinion.

    The collect phase pins ``GNINA_REMOTE_JOB_DIR`` to the directory recorded at
    submit time.  Collection must not depend on re-deriving the content address:
    the ligand file could have been touched, or the hash algorithm could change,
    and the result would be a "job not found" for work that is sitting finished
    on the GPU box.
    """
    if phase not in ("submit", "collect"):
        raise QueueError("phase must be 'submit' or 'collect', got %r" % (phase,))
    env = dict(os.environ if base_env is None else base_env)
    env["GNINA_REMOTE_MODE"] = phase
    root = params.get("queue_root")
    if root:
        env["GNINA_REMOTE_TICKET"] = ticket_path(root, record.job_id, phase)
    for key, value in (
        ("GNINA_REMOTE_HOST", params.get("host")),
        ("GNINA_REMOTE_PORT", params.get("port")),
        ("GNINA_REMOTE_KEY", params.get("key")),
        ("GNINA_REMOTE_IMAGE", params.get("image")),
        ("GNINA_REMOTE_NATIVE", params.get("native")),
        ("GNINA_REMOTE_DIR", params.get("remote_root")),
    ):
        if value:
            env[key] = str(value)
    if phase == "submit":
        # Belt and braces: submit mode takes the remote GPU mutex by definition,
        # and this makes that explicit for a shim that reads the flag.
        env["GNINA_REMOTE_LOCK"] = "1"
        deadline = float(params.get("job_deadline_seconds") or 1800.0)
        env.setdefault("GNINA_REMOTE_LOCK_STALE", str(int(max(1800.0, deadline + 600.0))))
        env.pop("GNINA_REMOTE_JOB_DIR", None)
    else:
        env.pop("GNINA_REMOTE_LOCK", None)
        remote_dir = (record.remote or {}).get("dir")
        if remote_dir:
            env["GNINA_REMOTE_JOB_DIR"] = str(remote_dir)
    return env


def ticket_path(root: str, job_id: str, phase: str) -> str:
    return os.path.join(root, "tickets", "%s.%s.json" % (job_id, phase))


def phase_log_path(root: str, job_id: str, phase: str) -> str:
    return os.path.join(root, "logs", "%s.%s.log" % (job_id, phase))


# ---------------------------------------------------------------------------
# the queue
# ---------------------------------------------------------------------------

class GninaQueue:
    """The per-run state store.  Many lock-free readers, ONE writer.

    THE WRITER CONTRACT, which nothing in this class can enforce for you:
    :meth:`transition` and :meth:`update_fields` are read-modify-write.  Each
    individual write is atomic, so a reader can never see a torn record, but two
    processes writing the *same* record concurrently will lose one of the two
    updates.  Every writer must therefore hold :meth:`acquire_pump_lock` --
    the pump does, the barrier does before it abandons anything, and an
    operator's ``--requeue`` must too.  Readers (``--status``, the barrier's
    polling, ``read_all``) need no lock and must not take one.
    """

    def __init__(self, run_dir: str, queue_dir: Optional[str] = None,
                 create: bool = True, clock=None):
        self.run_dir = os.path.abspath(run_dir)
        self.root = queue_root(run_dir, queue_dir)
        self.jobs_dir = os.path.join(self.root, "jobs")
        self.logs_dir = os.path.join(self.root, "logs")
        self.tickets_dir = os.path.join(self.root, "tickets")
        self.params_path = os.path.join(self.root, "params.json")
        self.caps_path = os.path.join(self.root, "caps.json")
        self.pump_lock_path = os.path.join(self.root, "pump.lock")
        self.stop_path = os.path.join(self.root, "STOP")
        self.summary_path = os.path.join(self.root, "summary.json")
        self._clock = clock or time.time
        # job_id -> (stat_key, record).  The collector collects in threads, so
        # this is shared mutable state and is guarded.
        self._cache = {}                       # type: Dict[str, Tuple[Any, JobRecord]]
        self._lock = threading.RLock()
        self._params_cache = None              # type: Optional[Dict[str, Any]]
        self._params_stat = None               # type: Optional[Tuple[int, int]]
        if create:
            for path in (self.root, self.jobs_dir, self.logs_dir, self.tickets_dir):
                _makedirs(path)

    # -- params ----------------------------------------------------------

    def set_params(self, params: Dict[str, Any], allow_change: bool = False
                   ) -> Dict[str, Any]:
        """Freeze the docking parameters for this queue.

        A campaign whose exhaustiveness or thread count changes halfway produces
        two incomparable halves and no record of where the seam is.  Changing a
        significant parameter is therefore refused, loudly and by name, unless
        the operator says otherwise.  Operational knobs (deadlines, attempt
        counts) are not significant and may change freely.
        """
        existing = self.get_params()
        merged = dict(params)
        merged.setdefault("schema_version", SCHEMA_VERSION)
        merged.setdefault("created_at", _utcnow_iso())
        if existing:
            changed = [
                k for k in SIGNIFICANT_PARAM_KEYS
                if k in params and existing.get(k) != params.get(k)
            ]
            if changed and not allow_change:
                raise QueueError(
                    "gnina queue %s was created with different docking parameters: %s. "
                    "Half a campaign at one setting and half at another cannot be "
                    "compared, and nothing downstream records where the change "
                    "happened. Re-run with --allow-params-change to override, or "
                    "point --queue-dir at a fresh queue."
                    % (self.root, ", ".join(
                        "%s %r -> %r" % (k, existing.get(k), params.get(k))
                        for k in changed))
                )
            merged["created_at"] = existing.get("created_at", merged["created_at"])
        _makedirs(self.root)
        atomic_write_json(self.params_path, merged)
        self._params_cache = None
        self._params_stat = None
        return merged

    def get_params(self) -> Dict[str, Any]:
        """The frozen parameters.

        Cached against the file's mtime and size: ``enqueue`` would otherwise
        re-read it once per job, and a stale cache would hide a params.json
        another process has just written.
        """
        try:
            stat = os.stat(self.params_path)
            key = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            key = None
        if self._params_cache is None or key != self._params_stat:
            self._params_cache = read_json_retry(self.params_path) or {}
            self._params_stat = key
        return dict(self._params_cache)

    def _runtime_params(self) -> Dict[str, Any]:
        """Params plus the queue root, which the phase environment needs."""
        params = self.get_params()
        params["queue_root"] = self.root
        return params

    # -- paths -----------------------------------------------------------

    def record_path(self, job_id: str) -> str:
        return os.path.join(self.jobs_dir, "%s.json" % job_id)

    # -- writes ----------------------------------------------------------

    def enqueue(self, job_dir: str, inputs: Dict[str, Any],
                job_id: Optional[str] = None) -> JobRecord:
        """Create a record if there is not one already.  Idempotent by design.

        Re-running a campaign must not disturb work in flight: a job that is
        already SUBMITTED stays SUBMITTED.  A job whose outputs are already on
        disk is created directly as COLLECTED, which is how today's
        skip-if-done behaviour survives the move to a queue.
        """
        job_dir = os.path.abspath(job_dir)
        jid = job_id or job_id_for(job_dir)
        if not jid:
            raise QueueError("cannot derive a job id from %r" % (job_dir,))
        _makedirs(self.jobs_dir)
        path = self.record_path(jid)
        existing = read_json_retry(path)
        if existing:
            record = JobRecord.from_dict(existing)
            if os.path.abspath(record.job_dir or "") != job_dir:
                raise QueueError(
                    "job id %s already maps to %s; refusing to point it at %s. "
                    "Two different jobs with the same directory name would "
                    "overwrite each other's results."
                    % (jid, record.job_dir, job_dir))
            return self._fill_missing_inputs(record, inputs)

        now = _utcnow_iso()
        merged_inputs = dict(_default_inputs())
        for key, value in (inputs or {}).items():
            merged_inputs[key] = value
        for key in ("receptor_pdb", "ligand_sdf", "pocket_pdb", "receptor_pdbqt"):
            if merged_inputs.get(key):
                merged_inputs[key] = os.path.abspath(str(merged_inputs[key]))
        already = outputs_complete(job_dir)
        state = "COLLECTED" if already else "PENDING"
        _validate_transition(NEW_STATE, state)
        record = JobRecord(
            job_id=jid,
            job_dir=job_dir,
            state=state,
            created_at=now,
            updated_at=now,
            inputs=merged_inputs,
            params=dict(self.get_params()),
            history=[{"at": now, "from": NEW_STATE, "to": state,
                      "why": "already complete at enqueue" if already else "enqueued"}],
        )
        if already:
            detail = outputs_detail(job_dir)
            record.outputs.update(detail)
            record.outputs["already_complete_at_enqueue"] = True
        with self._lock:
            created = _write_create_if_absent(path, record.to_dict())
            self._cache.pop(jid, None)
        if not created:
            loser = read_json_retry(path)
            if loser:
                return JobRecord.from_dict(loser)
        return record

    def _fill_missing_inputs(self, record: JobRecord,
                             inputs: Optional[Dict[str, Any]]) -> JobRecord:
        """Repair a PENDING record whose input paths are missing.

        Reconciliation re-derives an unreadable record from its job id alone, so
        the replacement has a job_dir but no receptor/ligand/pocket.  ``enqueue``
        would then return it untouched -- its contract is not to disturb work in
        flight -- and the first submit would ABANDON it with MISSING_INPUT, with
        ``--requeue`` restoring the same empty record.  Nothing short of deleting
        the file by hand recovered the job.

        Only PENDING records are touched, and only fields that are missing: that
        is the one state where no launch has happened, so nothing that has run
        can be contradicted.  Re-running batch_dock is then the repair path it
        was always meant to be.
        """
        if record.state != "PENDING" or not inputs:
            return record
        updates = {}
        for key in ("receptor_pdb", "ligand_sdf", "pocket_pdb", "receptor_pdbqt"):
            value = inputs.get(key)
            if value and not (record.inputs or {}).get(key):
                updates[key] = os.path.abspath(str(value))
        if not updates:
            return record
        LOG.info("filling in %s for PENDING job %s (its record was re-derived "
                 "without them)", ", ".join(sorted(updates)), record.job_id)
        return self.update_fields(record.job_id, inputs=updates)

    def enqueue_many(self, job_data_list: List[Dict[str, Any]],
                     params: Dict[str, Any],
                     allow_params_change: bool = False) -> Dict[str, int]:
        """Enqueue a whole campaign.  Returns enqueued / already_present / already_complete.

        Accepts ``batch_dock``'s job dicts verbatim (``current_outfolder``,
        ``receptor_pdb``, ...) so the caller does not have to reshape them.
        """
        self.set_params(params, allow_change=allow_params_change)
        stats = {"enqueued": 0, "already_present": 0, "already_complete": 0}
        for job in job_data_list or []:
            job_dir = job.get("current_outfolder") or job.get("job_dir")
            if not job_dir:
                raise QueueError(
                    "job entry has neither 'current_outfolder' nor 'job_dir': %r"
                    % (sorted(job.keys()),))
            jid = job.get("job_id") or job_id_for(job_dir)
            existed = os.path.exists(self.record_path(jid))
            record = self.enqueue(job_dir, {
                "receptor_pdb": job.get("receptor_pdb"),
                "ligand_sdf": job.get("ligand_sdf"),
                "pocket_pdb": job.get("pocket_pdb"),
                "receptor_pdbqt": job.get("receptor_pdbqt"),
                "skip_pdb_to_pdbqt": bool(job.get("skip_pdb_to_pdbqt", False)),
                "receptor_sha256": job.get("receptor_sha256"),
                "ligand_sha256": job.get("ligand_sha256"),
            }, job_id=jid)
            if existed:
                stats["already_present"] += 1
            else:
                stats["enqueued"] += 1
                if record.state == "COLLECTED":
                    stats["already_complete"] += 1
        return stats

    def transition(self, job_id: str, to_state: str, why: str = "",
                   **fields: Any) -> JobRecord:
        """Move a record to *to_state*, validating the edge first.

        Read-modify-write: safe against a crash (the write itself is atomic),
        not against a second concurrent writer.  See the class docstring.

        Dict-valued fields (``inputs``, ``remote``, ``collect``, ``outputs``,
        ``params``) are *merged* into what is there, so a caller recording a
        remote directory does not have to restate the host and port.  ``error``
        is *replaced* -- it is one whole object, and a stale ``code`` merged
        under a new ``message`` would be worse than useless.
        """
        record = self.read(job_id)
        if record is None:
            raise QueueError("no queue record for job %s" % job_id)
        _validate_transition(record.state, to_state)
        previous = record.state
        record.state = to_state
        self._apply_fields(record, fields)
        now = _utcnow_iso()
        record.updated_at = now
        record.history.append({"at": now, "from": previous, "to": to_state,
                               "why": why or ""})
        if len(record.history) > HISTORY_LIMIT:
            record.history = record.history[-HISTORY_LIMIT:]
        self._write(record)
        LOG.debug("%s: %s -> %s (%s)", job_id, previous, to_state, why)
        return record

    def update_fields(self, job_id: str, **fields: Any) -> JobRecord:
        """Update a record without changing its state.  A no-op does not write.

        The run directory may be on a fuseblk mount where every write costs;
        rewriting an identical record on every poll would be pure churn and would
        defeat the barrier's mtime-based cache.
        """
        record = self.read(job_id)
        if record is None:
            raise QueueError("no queue record for job %s" % job_id)
        before = record.to_dict()
        self._apply_fields(record, fields)
        after = record.to_dict()
        before.pop("updated_at", None)
        after.pop("updated_at", None)
        if before == after:
            return record
        record.updated_at = _utcnow_iso()
        self._write(record)
        return record

    @staticmethod
    def _apply_fields(record: JobRecord, fields: Dict[str, Any]) -> None:
        for key, value in fields.items():
            if key in ("inputs", "remote", "collect", "outputs", "params"):
                if value is None:
                    continue
                if not isinstance(value, dict):
                    raise QueueError("%s must be a dict, got %r" % (key, type(value)))
                getattr(record, key).update(value)
            elif key == "error":
                record.error = None if value is None else dict(value)
                if value and value.get("code"):
                    code = value["code"]
                    record.error_counts[code] = record.error_counts.get(code, 0) + 1
            elif key == "history":
                raise QueueError("history is append-only; pass why= instead")
            elif key in JobRecord._KNOWN:
                setattr(record, key, value)
            else:
                record.extra[key] = value

    def _write(self, record: JobRecord) -> None:
        with self._lock:
            atomic_write_json(self.record_path(record.job_id), record.to_dict())
            self._cache.pop(record.job_id, None)

    # -- reads -----------------------------------------------------------

    def read(self, job_id: str) -> Optional[JobRecord]:
        data = read_json_retry(self.record_path(job_id))
        if data is None:
            return None
        return JobRecord.from_dict(data)

    def read_all(self, refresh: bool = True) -> List[JobRecord]:
        """Every record, sorted by job id.

        Only records whose mtime or size changed are re-read.  A campaign is
        10^3-10^4 jobs and the barrier polls every 15 s; re-parsing every file
        each time would turn a cheap wait into a busy loop on the external drive.
        """
        if not os.path.isdir(self.jobs_dir):
            return []
        with self._lock:
            if not refresh and self._cache:
                return sorted((rec for (_key, rec) in self._cache.values()),
                              key=lambda r: r.job_id)
            try:
                names = os.listdir(self.jobs_dir)
            except OSError as exc:
                LOG.warning("cannot list %s: %s", self.jobs_dir, exc)
                return []
            records = []
            fresh_cache = {}
            for name in names:
                if not name.endswith(".json"):
                    continue
                job_id = name[:-len(".json")]
                path = os.path.join(self.jobs_dir, name)
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                # The inode is in the key because every write commits through
                # os.replace, which installs a NEW inode. On a filesystem with
                # coarse mtime granularity -- the external fuseblk drive has it --
                # two writes inside one tick would otherwise look unchanged.
                key = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                cached = self._cache.get(job_id)
                if cached is not None and cached[0] == key:
                    fresh_cache[job_id] = cached
                    records.append(cached[1])
                    continue
                data = read_json_retry(path)
                if data is None:
                    continue
                record = JobRecord.from_dict(data)
                if not record.job_id:
                    record.job_id = job_id
                elif record.job_id != job_id:
                    # The FILENAME is the address: read() and transition() find a
                    # record at <job_id>.json. Yielding a record whose body names
                    # a different job would put a job in front of the pump and the
                    # barrier that read() cannot find and transition() cannot
                    # write -- outstanding forever, advanceable by nothing. It is
                    # reported as unreadable instead, which is exactly what it is.
                    LOG.warning(
                        "queue record %s names job_id %r; the filename is the "
                        "address, so this record can be read but never written. "
                        "Treating it as unreadable -- rename it to %s.json or "
                        "delete it and re-run batch_dock.",
                        path, record.job_id, record.job_id)
                    continue
                fresh_cache[job_id] = (key, record)
                records.append(record)
            self._cache = fresh_cache
        records.sort(key=lambda r: r.job_id)
        return records

    def unreadable_job_ids(self) -> List[str]:
        """Record files that exist but cannot be parsed (a crash mid-create).

        Derived by difference from :meth:`read_all` rather than by re-parsing
        every file, so the cost is one directory listing even for 10^4 jobs.
        """
        if not os.path.isdir(self.jobs_dir):
            return []
        readable = set(r.job_id for r in self.read_all())
        bad = []
        try:
            names = sorted(os.listdir(self.jobs_dir))
        except OSError:
            return []
        for name in names:
            if not name.endswith(".json"):
                continue
            job_id = name[:-len(".json")]
            if job_id not in readable:
                bad.append(job_id)
        return bad

    def counts(self) -> Dict[str, int]:
        counts = dict((state, 0) for state in ALL_STATES)
        for record in self.read_all():
            counts[record.state] = counts.get(record.state, 0) + 1
        return counts

    def by_state(self, *states: str) -> List[JobRecord]:
        wanted = set(states)
        return [r for r in self.read_all() if r.state in wanted]

    def next_pending(self, now: Optional[float] = None) -> Optional[JobRecord]:
        """The oldest PENDING record whose backoff has expired.

        *now* must come from the same clock the queue was built with
        (:meth:`now`).  A mismatch is detected rather than obeyed: see
        :data:`MAX_BACKOFF_SECONDS`.
        """
        moment = self._clock() if now is None else now
        ready = []
        for record in self.read_all():
            if record.state != "PENDING":
                continue
            due = record.next_attempt_at
            if due is None or float(due) <= moment:
                ready.append(record)
                continue
            if float(due) - moment > MAX_BACKOFF_SECONDS:
                LOG.error(
                    "%s has next_attempt_at %.1f, %.0f s beyond the current time "
                    "%.1f -- further than any backoff this queue sets. The value "
                    "was almost certainly written in a different clock domain "
                    "(time.time vs time.monotonic); pass GninaQueue.now() to "
                    "next_pending(). Treating the job as due so the campaign is "
                    "not stalled by it.",
                    record.job_id, float(due), float(due) - moment, moment)
                ready.append(record)
        if not ready:
            return None
        ready.sort(key=lambda r: (r.created_at or "", r.job_id))
        return ready[0]

    def now(self) -> float:
        """The queue's clock.  Every scheduling comparison must use this one."""
        return self._clock()

    def active_remote(self) -> Optional[JobRecord]:
        """The one job outstanding on the GPU, if any.

        There must never be two.  If there are, something bypassed the pump lock
        or the remote mutex; say so, and return the oldest so the caller makes
        progress rather than stalling.
        """
        active = self.by_state("SUBMITTING", "SUBMITTED", "RUNNING")
        if not active:
            return None
        if len(active) > 1:
            LOG.warning(
                "%d jobs are outstanding on the remote at once (%s). One GPU means "
                "one job; this points at two pumps or a bypassed remote mutex.",
                len(active), ", ".join(r.job_id for r in active[:5]))
        active.sort(key=lambda r: (r.updated_at or "", r.job_id))
        return active[0]

    # -- locks -----------------------------------------------------------

    def acquire_pump_lock(self, argv: Optional[Sequence[str]] = None
                          ) -> Optional[PumpLock]:
        """Take the local single-pump mutex, or return None if someone else has it.

        Never waits.  A second pump that cannot get the lock must exit, not
        queue up behind the first -- two pumps would double-book the GPU.
        """
        _makedirs(self.root)
        body = _lock_body(argv)
        if fcntl is not None:
            fd = os.open(self.pump_lock_path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                os.close(fd)
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    return None
                LOG.debug("flock unavailable on %s (%s); using the O_EXCL fallback",
                          self.pump_lock_path, exc)
            else:
                try:
                    os.ftruncate(fd, 0)
                    os.lseek(fd, 0, os.SEEK_SET)
                    os.write(fd, (json.dumps(body, sort_keys=True) + "\n").encode())
                    os.fsync(fd)
                except OSError:
                    pass
                return PumpLock(self.pump_lock_path, fd, "flock", body)
        return self._acquire_pump_lock_excl(body)

    def _acquire_pump_lock_excl(self, body: Dict[str, Any]) -> Optional[PumpLock]:
        for attempt in (0, 1):
            try:
                fd = os.open(self.pump_lock_path,
                             os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
            except FileExistsError:
                existing = _read_lock_body(self.pump_lock_path)
                breakable, why = _lock_is_breakable(
                    self.pump_lock_path, existing, PUMP_LOCK_STALE_SECONDS,
                    self._clock())
                if not breakable or attempt:
                    LOG.debug("pump lock not acquired: %s", why)
                    return None
                LOG.warning(
                    "breaking a stale gnina pump lock at %s: %s. If a pump really "
                    "is alive elsewhere, two submitters would share one GPU -- "
                    "check before continuing.", self.pump_lock_path, why)
                try:
                    os.unlink(self.pump_lock_path)
                except OSError:
                    return None
                continue
            try:
                os.write(fd, (json.dumps(body, sort_keys=True) + "\n").encode())
                os.fsync(fd)
            except OSError:
                pass
            return PumpLock(self.pump_lock_path, fd, "excl", body)
        return None

    def pump_lock_is_free(self) -> bool:
        """True if no pump holds the lock.  Never disturbs a live holder.

        The barrier calls this to tell a dead collector from a slow GPU, so it
        must be read-only: acquiring the lock to test it would make a pump that
        was merely starting up lose the race and exit.
        """
        if not os.path.exists(self.pump_lock_path):
            return True
        if fcntl is not None:
            try:
                fd = os.open(self.pump_lock_path, os.O_RDWR)
            except OSError:
                return True
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    if exc.errno in (errno.EACCES, errno.EAGAIN):
                        return False
                    # flock is not supported here; judge by the body instead.
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    return True
            finally:
                os.close(fd)
        body = _read_lock_body(self.pump_lock_path)
        breakable, _why = _lock_is_breakable(
            self.pump_lock_path, body, PUMP_LOCK_STALE_SECONDS, self._clock())
        return breakable

    # -- recovery --------------------------------------------------------

    def reconcile(self, prober=None, logger=None,
                  now: Optional[float] = None) -> Dict[str, Tuple[str, str]]:
        """Rebuild state after a crash.  Disk first, remote second, record last.

        The record is the least trustworthy of the three because it is the only
        one a crash can truncate.  Rules are applied in order and the first match
        wins; the whole point of the ordering is that rule 1 -- complete outputs
        on local disk -- beats every recorded state, so a finished job can never
        be re-docked no matter what the queue believes.

        Returns ``{job_id: (from_state, to_state)}`` for records it changed.
        """
        log = logger or LOG
        moment = self._clock() if now is None else now
        probe = prober or default_prober
        changed = {}  # type: Dict[str, Tuple[str, str]]

        for job_id in self.unreadable_job_ids():
            derived = os.path.join(self.run_dir, DOCKING_RESULTS_DIRNAME, job_id)
            if os.path.isdir(derived):
                log.warning(
                    "queue record for %s is unreadable (a crash between claiming "
                    "the name and writing it); re-deriving it as PENDING from %s",
                    job_id, derived)
                record = JobRecord(
                    job_id=job_id, job_dir=derived, state="PENDING",
                    created_at=_utcnow_iso(), updated_at=_utcnow_iso(),
                    history=[{"at": _utcnow_iso(), "from": NEW_STATE,
                              "to": "PENDING", "why": "re-derived: record unreadable"}],
                )
                self._write(record)
                changed[job_id] = ("(unreadable)", "PENDING")
            else:
                log.warning(
                    "queue record for %s is unreadable and %s does not exist; "
                    "leaving it alone rather than guessing what job it was.",
                    job_id, derived)

        for record in self.read_all():
            if record.state in TERMINAL_STATES:
                continue
            before = record.state
            try:
                after = self._reconcile_one(record, probe, log, moment)
            except QueueError as exc:
                # One unclassifiable record must cost one job, not the campaign.
                # read_all is sorted, so without this guard a single record with
                # an unknown state (an operator's edit, a half-rolled-back
                # deployment) aborts the loop and every job after it loses crash
                # recovery -- permanently, because the pump retries the same
                # failing pass forever.
                log.error("reconcile: leaving %s alone, it cannot be classified "
                          "(%s). Every other record is still reconciled; this one "
                          "needs an operator (gnina_collect.py --requeue %s).",
                          record.job_id, exc, record.job_id)
                continue
            if after and after != before:
                changed[record.job_id] = (before, after)
        return changed

    def _reconcile_one(self, record: JobRecord, probe, log, moment: float
                       ) -> Optional[str]:
        job_id = record.job_id

        # Rule 1: never lose a finished job. Outputs on disk outrank everything.
        if outputs_complete(record.job_dir):
            detail = outputs_detail(record.job_dir)
            self.transition(job_id, "COLLECTED",
                            why="reconcile: outputs already complete on disk",
                            outputs=detail, error=None)
            return "COLLECTED"

        # Rule 3: a collect whose process is gone must be redone. Collecting is
        # idempotent -- purge, then split/complex/parse all overwrite -- so this
        # cannot leave stale poses behind.
        if record.state == "COLLECTING":
            pid = (record.collect or {}).get("pid")
            host = (record.collect or {}).get("host")
            if host and host != _hostname():
                alive = False
            else:
                alive = _pid_alive(pid)
            if not alive:
                self.transition(job_id, "REMOTE_DONE",
                                why="reconcile: collect pid %s is gone" % pid,
                                collect={"pid": None, "finished_at": _utcnow_iso()})
                return "REMOTE_DONE"
            # A live pid is not proof of a live collect: pids are recycled, and a
            # wedged download answers kill -0 for as long as the machine is up.
            # Without a deadline such a record is outstanding for the rest of the
            # campaign and the barrier waits on it forever. Re-collecting is
            # idempotent, so being wrong here costs one repeated download.
            deadline = float((record.params or {}).get("job_deadline_seconds")
                             or (self.get_params() or {}).get("job_deadline_seconds")
                             or 1800.0) + STALL_GRACE_SECONDS
            elapsed = _elapsed_since((record.collect or {}).get("started_at"))
            if elapsed is not None and elapsed > deadline:
                log.warning(
                    "reconcile: %s has been COLLECTING for %.0fs (deadline %.0fs) "
                    "with pid %s still answering. Sending it back to REMOTE_DONE "
                    "for another attempt; the remote result is untouched.",
                    job_id, elapsed, deadline, pid)
                self.transition(job_id, "REMOTE_DONE",
                                why="reconcile: collect exceeded its deadline",
                                collect={"pid": None, "finished_at": _utcnow_iso()})
                return "REMOTE_DONE"
            return None

        # Rule 4: an intent record with no remote directory. Nothing was ever
        # launched that we can name, so start over rather than probe blindly.
        if record.state in ("SUBMITTING", "SUBMITTED", "RUNNING"):
            if not (record.remote or {}).get("dir"):
                self.transition(job_id, "PENDING",
                                why="reconcile: no remote directory was recorded")
                return "PENDING"
            # Rule 5: probe the directory we do know about.
            try:
                state, code = probe(record)
            except Exception as exc:                      # noqa: BLE001
                log.warning("reconcile: probe failed for %s (%s); leaving it in %s",
                            job_id, exc, record.state)
                return None
            return self._apply_probe(record, state, code, moment, log,
                                     source="reconcile")

        # Rule 6: a PENDING record that already has a remote directory. If the
        # GPU finished it, collect -- never re-dock work that exists.
        if record.state == "PENDING" and (record.remote or {}).get("dir"):
            try:
                state, code = probe(record)
            except Exception as exc:                      # noqa: BLE001
                log.debug("reconcile: probe failed for pending %s: %s", job_id, exc)
                return None
            if state == "DONE" and code == 0:
                self.transition(job_id, "REMOTE_DONE",
                                why="reconcile: a prior remote run finished",
                                remote={"exit_code": 0})
                return "REMOTE_DONE"
            return None

        # Rule 7: FAILED is left to the pump's retry policy.
        return None

    def _apply_probe(self, record: JobRecord, state: str, code: Optional[int],
                     moment: float, log, source: str) -> Optional[str]:
        job_id = record.job_id
        current = record.state
        if state == "DONE" and code == 0:
            target = "REMOTE_DONE"
            if target not in LEGAL_TRANSITIONS.get(current, frozenset()):
                return None
            remote = {"exit_code": 0}
            seconds = _elapsed_since((record.remote or {}).get("launched_at"))
            if seconds is not None:
                remote["remote_seconds"] = seconds
            self.transition(job_id, target, why="%s: remote exit_code 0" % source,
                            remote=remote)
            return target
        if state == "DONE":
            return self._fail(record, "REMOTE_GNINA_FAILED",
                              "remote gnina exited %s" % code, "poll")
        if state == "GONE":
            if current == "PENDING":
                return None
            if source == "reconcile":
                self.transition(job_id, "PENDING",
                                why="reconcile: remote directory is gone",
                                remote={"dir": None, "exit_code": None})
                return "PENDING"
            return self._fail(record, "REMOTE_DIR_GONE",
                              "remote job directory %s no longer exists"
                              % (record.remote or {}).get("dir"), "poll")
        # LAUNCHED or RUNNING: still working, unless it has blown the deadline.
        deadline = float((record.params or {}).get("job_deadline_seconds")
                         or (self.get_params().get("job_deadline_seconds") or 1800.0))
        # Fall back to updated_at when no launch time was recorded: without a
        # clock to measure against, a job whose probe answers LAUNCHED forever
        # would never be declared stalled and would hold the GPU slot for good.
        elapsed = _elapsed_since((record.remote or {}).get("launched_at")
                                 or record.updated_at)
        if elapsed is not None and elapsed > deadline + STALL_GRACE_SECONDS:
            return self._fail(
                record, "STALLED",
                "no exit_code after %.0fs, past the %.0fs deadline plus %.0fs grace"
                % (elapsed, deadline, STALL_GRACE_SECONDS), "poll")
        target = "RUNNING" if state == "RUNNING" else "SUBMITTED"
        if target == current:
            return None
        if target not in LEGAL_TRANSITIONS.get(current, frozenset()):
            return None
        self.transition(job_id, target, why="%s: remote is %s" % (source, state))
        return target

    def _fail(self, record: JobRecord, code: str, message: str, phase: str
              ) -> Optional[str]:
        if "FAILED" not in LEGAL_TRANSITIONS.get(record.state, frozenset()):
            LOG.warning("cannot fail %s from %s: %s", record.job_id, record.state,
                        message)
            return None
        LOG.warning("%s failed [%s]: %s", record.job_id, code, message)
        self.transition(record.job_id, "FAILED", why="%s: %s" % (code, message),
                        error=make_error(code, message, phase))
        return "FAILED"

    # -- reporting -------------------------------------------------------

    def summary(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        records = self.read_all()
        counts = dict((state, 0) for state in ALL_STATES)
        errors = {}  # type: Dict[str, int]
        remote_seconds = []  # type: List[float]
        already_complete = 0
        outstanding_ids = []  # type: List[str]
        abandoned_ids = []  # type: List[str]
        active = None
        for record in records:
            counts[record.state] = counts.get(record.state, 0) + 1
            if record.error_code:
                errors[record.error_code] = errors.get(record.error_code, 0) + 1
            seconds = (record.remote or {}).get("remote_seconds")
            if isinstance(seconds, (int, float)) and seconds > 0:
                remote_seconds.append(float(seconds))
            if (record.outputs or {}).get("already_complete_at_enqueue"):
                already_complete += 1
            if record.is_outstanding:
                outstanding_ids.append(record.job_id)
            if record.state == "ABANDONED":
                abandoned_ids.append(record.job_id)
            if record.state in ("SUBMITTING", "SUBMITTED", "RUNNING") and active is None:
                active = record
        # A record file that cannot be parsed is still a job. Leaving it out of
        # the totals is how a campaign that lost one reports a clean drain: the
        # barrier stops waiting, the pump exits 0, and the job is neither
        # collected nor abandoned nor named anywhere. Count it as outstanding --
        # which is what it is -- and name it, so every consumer inherits both.
        unreadable = self.unreadable_job_ids()
        outstanding_ids = outstanding_ids + unreadable
        total = len(records) + len(unreadable)
        mean_remote = (sum(remote_seconds) / len(remote_seconds)
                       if remote_seconds else None)
        out = {
            "schema_version": SCHEMA_VERSION,
            "at": _utcnow_iso(),
            "queue_root": self.root,
            "run_dir": self.run_dir,
            "total": total,
            "counts": counts,
            "collected": counts["COLLECTED"],
            "abandoned": counts["ABANDONED"],
            "failed": counts["FAILED"],
            "pending": counts["PENDING"],
            "outstanding": total - counts["COLLECTED"] - counts["ABANDONED"],
            "already_complete": already_complete,
            "mean_remote_seconds": mean_remote,
            "errors": errors,
            "outstanding_job_ids": outstanding_ids[:SUMMARY_ID_LIMIT],
            "outstanding_job_ids_truncated": max(
                0, len(outstanding_ids) - SUMMARY_ID_LIMIT),
            "abandoned_job_ids": abandoned_ids[:SUMMARY_ID_LIMIT],
            "unreadable_job_ids": unreadable[:SUMMARY_ID_LIMIT],
            "active_job_id": active.job_id if active else None,
            "active_state": active.state if active else None,
            "active_seconds": (_elapsed_since((active.remote or {}).get("launched_at"))
                               if active else None),
        }
        if extra:
            out.update(extra)
        return out

    def write_summary(self, extra: Optional[Dict[str, Any]] = None) -> str:
        _makedirs(self.root)
        atomic_write_json(self.summary_path, self.summary(extra))
        return self.summary_path


def _validate_transition(from_state: str, to_state: str) -> None:
    if to_state not in ALL_STATES:
        raise QueueError("%r is not a known state; the nine states are %s"
                         % (to_state, ", ".join(ALL_STATES)))
    allowed = LEGAL_TRANSITIONS.get(from_state)
    if allowed is None:
        raise QueueError("%r is not a known state" % (from_state,))
    if to_state not in allowed:
        raise IllegalTransition(
            "%s -> %s is not a legal transition (legal from %s: %s). Accepting it "
            "would either re-dock finished work or record a launch that never "
            "happened." % (from_state, to_state, from_state,
                           ", ".join(sorted(allowed)) or "nothing: it is terminal"))


def _elapsed_since(iso: Optional[str]) -> Optional[float]:
    if not iso:
        return None
    try:
        stamp = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None
    return max(0.0, (datetime.now(timezone.utc) - stamp).total_seconds())


# ---------------------------------------------------------------------------
# submit / collect / poll
# ---------------------------------------------------------------------------

def submit_job(queue: GninaQueue, record: JobRecord, runner=None,
               timeout: float = 900.0) -> JobRecord:
    """Launch *record* on the remote GPU.  One job at a time, no exceptions.

    The sequence, and why it is that sequence:

    1. refuse if anything is already outstanding on the remote -- one GPU;
    2. write SUBMITTING **before** anything is launched, so a crash mid-launch
       is resolved by probing rather than by re-launching blind;
    3. purge stale gnina outputs (rule R2), or the submit phase would
       "succeed" on the previous run's poses;
    4. run ``consensus_docker`` in submit mode;
    5. check rule R3 -- a ``results.csv`` after a submit means the contract is
       broken, and that must stop the job rather than be papered over;
    6. read the ticket.  The ticket, not the exit code, is the signal: the
       submit phase exits non-zero *by design*.
    """
    run = runner or default_runner
    params = queue._runtime_params()
    if record.state != "PENDING":
        LOG.warning("refusing to submit %s from state %s; only PENDING jobs are "
                    "submitted", record.job_id, record.state)
        return record
    active = queue.active_remote()
    if active is not None and active.job_id != record.job_id:
        LOG.warning(
            "refusing to submit %s: %s is already outstanding on the remote (%s). "
            "One GPU means one gnina at a time.",
            record.job_id, active.job_id, active.state)
        return record

    record = queue.transition(record.job_id, "SUBMITTING",
                              why="pump picked it up", error=None)

    missing = [k for k in ("receptor_pdb", "ligand_sdf", "pocket_pdb")
               if not (record.inputs.get(k) and os.path.exists(record.inputs[k]))]
    if missing:
        message = ("input file(s) missing: %s"
                   % ", ".join("%s=%s" % (k, record.inputs.get(k)) for k in missing))
        LOG.error("%s: %s", record.job_id, message)
        return queue.transition(record.job_id, "ABANDONED", why=message,
                                error=make_error("MISSING_INPUT", message, "submit",
                                                 retriable=False))

    purge_gnina_outputs(record.job_dir)
    tickets = ticket_path(queue.root, record.job_id, "submit")
    _forget_ticket(tickets)
    log_path = phase_log_path(queue.root, record.job_id, "submit")
    try:
        command = build_consensus_command(record, params)
    except QueueError as exc:
        return queue.transition(record.job_id, "ABANDONED", why=str(exc),
                                error=make_error("MISSING_INPUT", str(exc),
                                                 "submit", retriable=False))
    env = build_phase_env(record, params, "submit")

    returncode = None
    why = None
    try:
        returncode, _out, _err = run(command, env, timeout, log_path, record.job_dir)
    except subprocess.TimeoutExpired:
        why = ("did not return within %.0fs; the remote job may still have been "
               "launched, so the next attempt probes before re-docking" % timeout)
        LOG.warning("%s: submit phase %s", record.job_id, why)
        returncode = None
    except Exception as exc:                                  # noqa: BLE001
        why = "could not be run: %s: %s" % (type(exc).__name__, exc)
        LOG.warning("%s: submit phase %s", record.job_id, why)
        returncode = None

    # Rule R3, checked before anything else the run produced is believed.
    results_csv = os.path.join(_gnina_dir(record.job_dir), "results.csv")
    if os.path.exists(results_csv):
        message = (
            "the submit phase wrote %s. It must not: submit does not download "
            "poses, so a results.csv here came from stale bytes or from a "
            "consensus_docker that no longer fails on the missing out.sdf. Either "
            "way the scores could be from the wrong run." % results_csv)
        if outputs_complete(record.job_dir):
            # The likeliest cause during integration, and the cheapest to check.
            message += (
                " The outputs are in fact complete, which is the signature of a "
                "gnina_remote.py that ignores GNINA_REMOTE_MODE and ran the whole "
                "job synchronously. Check that the shim implements submit/collect "
                "mode; until it does, run with GNINA_ASYNC=0.")
        LOG.critical("%s: %s", record.job_id, message)
        return queue.transition(record.job_id, "ABANDONED", why="SUBMIT_WROTE_RESULTS",
                                error=make_error("SUBMIT_WROTE_RESULTS", message,
                                                 "submit", retriable=False))

    ticket = read_json_retry(tickets)
    outcome = (ticket or {}).get("outcome")
    remote_fields = _remote_fields_from_ticket(ticket)

    if outcome == "busy" or (ticket is None and returncode == 5):
        busy_count = int((record.remote or {}).get("busy_count") or 0) + 1
        backoff = BUSY_BACKOFF_SECONDS[min(busy_count - 1,
                                           len(BUSY_BACKOFF_SECONDS) - 1)]
        remote_fields["busy_count"] = busy_count
        LOG.info("%s: remote GPU is busy; retrying in %.0fs (attempt not consumed)",
                 record.job_id, backoff)
        return queue.transition(
            record.job_id, "PENDING", why="remote GPU mutex held by someone else",
            next_attempt_at=queue._clock() + backoff, remote=remote_fields)

    if outcome == "already_done":
        LOG.info("%s: a completed remote run for these inputs already exists; "
                 "collecting rather than re-docking", record.job_id)
        remote_fields.setdefault("exit_code", 0)
        return queue.transition(record.job_id, "REMOTE_DONE",
                                why="resume hit: remote result already exists",
                                attempts=record.attempts + 1, remote=remote_fields,
                                error=None)

    if outcome == "launched":
        remote_fields.setdefault("launched_at", _utcnow_iso())
        return queue.transition(record.job_id, "SUBMITTED",
                                why="detached remote job launched",
                                attempts=record.attempts + 1, remote=remote_fields,
                                error=None)

    if outcome == "error":
        detail = (ticket or {}).get("error") or "the submit phase reported an error"
        return _fail_or_abandon(queue, record, "SUBMIT_NO_TICKET",
                                "submit ticket reports an error: %s" % detail,
                                "submit", remote_fields)

    message = ("no usable submit ticket at %s (outcome=%r, exit=%s%s). Last log "
               "lines:\n%s" % (tickets, outcome, returncode,
                               "; the phase %s" % why if why else "",
                               _log_tail(log_path, 40) or "<log empty>"))
    return _fail_or_abandon(queue, record, "SUBMIT_NO_TICKET", message, "submit",
                            remote_fields)


def collect_job(queue: GninaQueue, record: JobRecord, runner=None,
                timeout: float = 1800.0) -> JobRecord:
    """Download the remote result and let ``consensus_docker`` post-process it.

    This is where the poses and the CNN scores are actually produced, by
    ``split_mol`` / ``make_complex`` / ``parse_gnina`` -- the same code as a
    local run.  It is idempotent: outputs are purged first and every step
    overwrites, so a collect interrupted anywhere can simply be run again.

    A failure here is classified by *where* it happened, because that decides
    whether the GPU has to repeat 2.5 minutes of work: a download or
    post-processing failure returns to REMOTE_DONE (the poses on the GPU box are
    still good), while an empty or unverifiable result goes back to PENDING.
    """
    run = runner or default_runner
    params = queue._runtime_params()
    if record.state != "REMOTE_DONE":
        LOG.warning("refusing to collect %s from state %s", record.job_id, record.state)
        return record

    record = queue.transition(
        record.job_id, "COLLECTING", why="collect phase starting",
        collect={"pid": os.getpid(), "host": _hostname(),
                 "started_at": _utcnow_iso(), "finished_at": None})

    purge_gnina_outputs(record.job_dir)
    tickets = ticket_path(queue.root, record.job_id, "collect")
    _forget_ticket(tickets)
    log_path = phase_log_path(queue.root, record.job_id, "collect")
    try:
        command = build_consensus_command(record, params)
    except QueueError as exc:
        return queue.transition(record.job_id, "FAILED", why=str(exc),
                                error=make_error("POSTPROCESS_FAILED", str(exc),
                                                 "collect"))
    env = build_phase_env(record, params, "collect")

    try:
        returncode, _out, _err = run(command, env, timeout, log_path, record.job_dir)
    except subprocess.TimeoutExpired:
        message = ("collect phase exceeded %.0fs and was killed. The remote job is "
                   "detached and unaffected; its result is still there to collect."
                   % timeout)
        LOG.warning("%s: %s", record.job_id, message)
        return queue.transition(record.job_id, "FAILED", why="COLLECT_TIMEOUT",
                                collect={"finished_at": _utcnow_iso()},
                                error=make_error("COLLECT_TIMEOUT", message, "collect"))
    except Exception as exc:                                   # noqa: BLE001
        message = "collect phase could not be run: %s" % exc
        LOG.warning("%s: %s", record.job_id, message)
        return queue.transition(record.job_id, "FAILED", why="POSTPROCESS_FAILED",
                                collect={"finished_at": _utcnow_iso()},
                                error=make_error("POSTPROCESS_FAILED", message,
                                                 "collect"))

    ticket = read_json_retry(tickets)
    outcome = (ticket or {}).get("outcome")
    queue.update_fields(record.job_id, collect={"finished_at": _utcnow_iso()})
    record = queue.read(record.job_id) or record

    if outcome in ("not_ready", "gone"):
        message = ("the remote no longer has a finished result for this job "
                   "(ticket outcome %r); it has to be docked again" % outcome)
        return queue.transition(record.job_id, "FAILED", why="REMOTE_DIR_GONE",
                                error=make_error("REMOTE_DIR_GONE", message, "collect"))

    if returncode != 0:
        if outcome == "collected":
            code = "POSTPROCESS_FAILED"
            message = ("the result downloaded but post-processing failed (exit %s). "
                       "The remote poses are still good, so this retries the "
                       "collect rather than the docking. Last log lines:\n%s"
                       % (returncode, _log_tail(log_path, 40) or "<log empty>"))
        else:
            code = "SSH_TRANSIENT"
            message = ("collect phase exited %s before the result was downloaded "
                       "(ticket outcome %r). Last log lines:\n%s"
                       % (returncode, outcome,
                          _log_tail(log_path, 40) or "<log empty>"))
        if code == "SSH_TRANSIENT":
            transient = record.transient_attempts + 1
            if transient <= MAX_TRANSIENT_ATTEMPTS:
                return queue.transition(
                    record.job_id, "REMOTE_DONE",
                    why="transient failure during download; will retry",
                    transient_attempts=transient,
                    error=make_error(code, message, "collect"))
            message = ("%s (giving up after %d transient failures)"
                       % (message, transient))
        return queue.transition(record.job_id, "FAILED", why=code,
                                error=make_error(code, message, "collect"))

    ok, detail = outputs_verified_at_collect(record.job_dir)
    if ok:
        return queue.transition(
            record.job_id, "COLLECTED", why="collected and verified",
            outputs={k: detail[k] for k in
                     ("n_poses", "n_complexes", "n_result_rows", "out_sdf_bytes")},
            error=None)

    if detail.get("out_sdf_bytes", 0) == 0:
        message = ("out.sdf is empty after a successful collect: %s. An empty pose "
                   "file must never look like a result."
                   % "; ".join(detail.get("reasons") or []))
        return queue.transition(record.job_id, "FAILED", why="EMPTY_OUTPUT",
                                outputs=_detail_fields(detail),
                                error=make_error("EMPTY_OUTPUT", message, "collect"))
    message = ("the collect phase exited 0 but its outputs do not agree: %s. That "
               "is a half-written result, and it would reach the consensus RMSD "
               "unnoticed." % "; ".join(detail.get("reasons") or []))
    return queue.transition(record.job_id, "FAILED", why="POSTPROCESS_FAILED",
                            outputs=_detail_fields(detail),
                            error=make_error("POSTPROCESS_FAILED", message, "collect"))


def poll_job(queue: GninaQueue, record: JobRecord, prober=None,
             now: Optional[float] = None) -> JobRecord:
    """Ask the remote how the outstanding job is doing and record the answer.

    Only SUBMITTED and RUNNING are pollable.  SUBMITTING is not: its remote
    directory may not exist yet, and probing a directory we have not been told
    about could only produce a wrong answer.
    """
    if record.state not in ("SUBMITTED", "RUNNING"):
        return record
    if not (record.remote or {}).get("dir"):
        return record
    probe = prober or default_prober
    moment = queue._clock() if now is None else now
    try:
        state, code = probe(record)
    except Exception as exc:                                   # noqa: BLE001
        transient = record.transient_attempts + 1
        LOG.info("%s: probe failed (%s); the remote job is detached and unaffected",
                 record.job_id, exc)
        if transient > MAX_TRANSIENT_ATTEMPTS:
            return queue.transition(
                record.job_id, "FAILED", why="SSH_TRANSIENT",
                error=make_error("SSH_TRANSIENT",
                                 "probe failed %d times: %s" % (transient, exc),
                                 "poll"))
        return queue.update_fields(record.job_id, transient_attempts=transient)
    queue._apply_probe(record, state, code, moment, LOG, source="poll")
    return queue.read(record.job_id) or record


def _detail_fields(detail: Dict[str, Any]) -> Dict[str, Any]:
    return dict((k, detail[k]) for k in
                ("n_poses", "n_complexes", "n_result_rows", "out_sdf_bytes")
                if k in detail)


def _forget_ticket(path: str) -> None:
    """Remove a previous attempt's ticket so it cannot be read as this one's."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        LOG.warning("could not remove stale ticket %s: %s", path, exc)


def _remote_fields_from_ticket(ticket: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not ticket:
        return {}
    fields = {}
    for src, dst in (("remote_dir", "dir"), ("job_key", "job_key"),
                     ("host", "host"), ("port", "port"), ("image", "image"),
                     ("remote_pid_file", "remote_pid_file"),
                     ("exit_code", "exit_code")):
        if ticket.get(src) is not None:
            fields[dst] = ticket[src]
    return fields


def _fail_or_abandon(queue: GninaQueue, record: JobRecord, code: str,
                     message: str, phase: str,
                     remote_fields: Optional[Dict[str, Any]] = None) -> JobRecord:
    LOG.warning("%s failed [%s]: %s", record.job_id, code, message)
    return queue.transition(record.job_id, "FAILED", why=code,
                            attempts=record.attempts + 1,
                            remote=remote_fields or {},
                            error=make_error(code, message, phase))


# ---------------------------------------------------------------------------
# retry policy
# ---------------------------------------------------------------------------

def retry_decision(record: JobRecord, max_attempts: int = 3
                   ) -> Tuple[Optional[str], str]:
    """Where a FAILED record goes next.  Pure, so the policy is table-testable.

    Returns ``(next_state, why)``; ``next_state`` None means abandon.  Two rules
    beyond "count the attempts":

    * some failures resume at REMOTE_DONE rather than PENDING, because the GPU's
      output is still good and re-docking it would waste 2.5 minutes to produce
      the same poses;
    * three classes get exactly one retry regardless of ``max_attempts`` -- a
      second identical remote failure is a property of the inputs, not the link.
    """
    code = record.error_code
    if code is None:
        return "PENDING", "no error recorded; retrying"
    if code not in RETRIABLE_ERRORS:
        return None, "%s is not retriable" % code
    if code in ONE_RETRY_ERRORS and record.error_counts.get(code, 0) > 1:
        return None, ("%s happened %d times; a second identical failure is a "
                      "property of the inputs, not of the link"
                      % (code, record.error_counts.get(code, 0)))
    if code == "SSH_TRANSIENT":
        if record.transient_attempts > MAX_TRANSIENT_ATTEMPTS:
            return None, "the link failed %d times" % record.transient_attempts
        return RETRY_TARGET_STATE.get(code, "PENDING"), "transient failure"
    if record.attempts >= max_attempts:
        return None, ("%d of %d docking attempts consumed"
                      % (record.attempts, max_attempts))
    return (RETRY_TARGET_STATE.get(code, "PENDING"),
            "retrying after %s (attempt %d of %d)"
            % (code, record.attempts + 1, max_attempts))


def apply_retry_policy(queue: GninaQueue, record: JobRecord, max_attempts: int = 3,
                       now: Optional[float] = None) -> JobRecord:
    """Move a FAILED record on, per :func:`retry_decision`."""
    if record.state != "FAILED":
        return record
    target, why = retry_decision(record, max_attempts)
    if target is None:
        return queue.transition(record.job_id, "ABANDONED", why=why)
    moment = queue._clock() if now is None else now
    backoff = RETRY_BACKOFF_SECONDS[min(record.attempts,
                                        len(RETRY_BACKOFF_SECONDS) - 1)]
    fields = {}
    if target == "PENDING":
        fields["next_attempt_at"] = moment + backoff
    return queue.transition(record.job_id, target, why=why, **fields)


# ---------------------------------------------------------------------------
# the barrier
# ---------------------------------------------------------------------------

def wait_for_completion(run_dir: str, timeout: Optional[float] = None,
                        poll: float = 15.0, logger=None, on_progress=None,
                        restart_collector=None, queue_dir: Optional[str] = None,
                        progress_interval: float = 30.0,
                        clock=None, sleeper=None,
                        max_restarts: int = 3) -> Dict[str, Any]:
    """Wait for the gnina queue to drain, then return a summary.

    Placed immediately before stage 5, so gnina has been running on the GPU for
    the whole local pipeline.

    It *waits, reports and proceeds*.  A timeout is not a run failure: stage 5's
    existing filter already excludes any job whose enabled tools are not all
    complete, so a partial run still produces a usable dataset -- every job with
    all four tools gets its consensus and the rest are named, here and in
    ``summary.json``, so a partial campaign is never analysed as a complete one.

    A silently dead collector must never look like a slow one, so a free pump
    lock with work outstanding triggers a restart (up to *max_restarts*).  After
    that the barrier takes the lock itself, abandons anything still FAILED with
    PUMP_UNAVAILABLE, and stops waiting rather than hanging until the timeout.

    ``clock`` and ``sleeper`` are injection points for tests; production uses
    ``time.time`` and ``time.sleep``.
    """
    log = logger or LOG
    now_fn = clock or time.time
    sleep_fn = sleeper or time.sleep
    queue = GninaQueue(run_dir, queue_dir=queue_dir, create=False,
                       clock=now_fn)
    started = now_fn()

    if not os.path.isdir(queue.jobs_dir):
        log.info("no gnina queue at %s; nothing to wait for", queue.root)
        return _finish_summary(queue, started, now_fn, timed_out=False, waited=False)

    snapshot = queue.summary()
    outstanding = snapshot["outstanding"]
    mean_remote = snapshot["mean_remote_seconds"] or 180.0
    if timeout is None:
        timeout = max(3600.0, outstanding * mean_remote * 1.5 + 1800.0)
        log.info("gnina barrier: %d job(s) outstanding, waiting up to %.0f s "
                 "(computed from %.0f s mean remote time, serial execution)",
                 outstanding, timeout, mean_remote)
    else:
        timeout = float(timeout)
        log.info("gnina barrier: %d job(s) outstanding, waiting up to %.0f s",
                 outstanding, timeout)

    restarts = 0
    free_lock_observations = 0
    last_progress = 0.0
    timed_out = False

    while True:
        snapshot = queue.summary()
        outstanding = snapshot["outstanding"]
        if outstanding <= 0:
            log.info("gnina barrier: queue drained (%d collected, %d abandoned)",
                     snapshot["collected"], snapshot["abandoned"])
            break

        elapsed = now_fn() - started
        if timeout <= 0:
            log.info("gnina barrier: not waiting (timeout=0); %d job(s) outstanding",
                     outstanding)
            timed_out = True
            break
        if elapsed >= timeout:
            timed_out = True
            ids = snapshot["outstanding_job_ids"]
            log.warning("gnina barrier timed out after %.0fs with %d job(s) "
                        "outstanding", elapsed, outstanding)
            log.warning("outstanding: %s%s", ", ".join(ids[:20]),
                        (" +%d more" % (outstanding - 20)) if outstanding > 20 else "")
            log.warning("These jobs have no gnina result, so stage 5 will skip them. "
                        "Re-running batch_dock resumes them: the queue and the "
                        "content-addressed remote directories make that cheap.")
            break

        if now_fn() - last_progress >= progress_interval:
            last_progress = now_fn()
            log.info("%s", _progress_line(snapshot, mean_remote))
            if on_progress is not None:
                try:
                    on_progress(snapshot)
                except Exception as exc:                       # noqa: BLE001
                    log.debug("on_progress callback raised: %s", exc)

        if queue.pump_lock_is_free():
            free_lock_observations += 1
        else:
            free_lock_observations = 0
        # Two consecutive observations: a pump that is merely restarting can look
        # free for an instant, and restarting a healthy pump would double-book.
        if free_lock_observations >= 2:
            free_lock_observations = 0
            if restart_collector is not None and restarts < max_restarts:
                restarts += 1
                log.warning("gnina collector is not holding the pump lock while %d "
                            "job(s) are outstanding; restarting it (%d/%d)",
                            outstanding, restarts, max_restarts)
                try:
                    restart_collector()
                except Exception as exc:                       # noqa: BLE001
                    log.warning("restarting the gnina collector failed: %s", exc)
            else:
                log.warning("no gnina collector is running and it cannot be "
                            "restarted (%d attempt(s) made). Abandoning the %d "
                            "failed job(s) and proceeding.", restarts,
                            snapshot["failed"])
                _abandon_without_pump(queue, log)
                timed_out = True
                break

        sleep_fn(poll)

    summary = _finish_summary(queue, started, now_fn, timed_out=timed_out,
                              waited=True, restarts=restarts)
    try:
        queue.write_summary(summary)
    except (OSError, QueueError) as exc:
        log.warning("could not write %s: %s", queue.summary_path, exc)
    return summary


def _abandon_without_pump(queue: GninaQueue, log) -> None:
    lock = queue.acquire_pump_lock()
    if lock is None:
        log.warning("could not take the pump lock to abandon failed jobs")
        return
    try:
        for record in queue.by_state("FAILED"):
            queue.transition(record.job_id, "ABANDONED",
                             why="PUMP_UNAVAILABLE: no collector is running",
                             error=make_error(
                                 "PUMP_UNAVAILABLE",
                                 "the gnina collector could not be kept alive; this "
                                 "job's last error was %s"
                                 % (record.error_code or "unrecorded"),
                                 "barrier", retriable=False))
    finally:
        lock.release()


def _finish_summary(queue: GninaQueue, started: float, now_fn,
                    timed_out: bool, waited: bool,
                    restarts: int = 0) -> Dict[str, Any]:
    return queue.summary({
        "elapsed_seconds": max(0.0, now_fn() - started),
        "timed_out": bool(timed_out),
        "waited": bool(waited),
        "collector_restarts": restarts,
    })


def _progress_line(snapshot: Dict[str, Any], mean_remote: float) -> str:
    counts = snapshot["counts"]
    running = counts["SUBMITTING"] + counts["SUBMITTED"] + counts["RUNNING"]
    eta = snapshot["outstanding"] * (snapshot["mean_remote_seconds"] or mean_remote)
    active = ""
    if snapshot.get("active_job_id"):
        seconds = snapshot.get("active_seconds")
        active = (" | GPU busy on %s%s"
                  % (snapshot["active_job_id"],
                     (" for %.0f s" % seconds) if seconds is not None else ""))
    return ("GNINA QUEUE: %d total | %d collected, %d running, %d pending, "
            "%d failed(retrying), %d abandoned%s | ETA %.1f m"
            % (snapshot["total"], snapshot["collected"], running,
               counts["PENDING"], counts["FAILED"], counts["ABANDONED"],
               active, eta / 60.0))
