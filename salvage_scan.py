#!/usr/bin/env python3
"""Phase 2 of ``REBUILD_PLAN.md`` -- classify every old job under correct identity.

What this does
--------------
Reads the 24,173 job directories of the 2025-06-29 campaign **read-only**, and
answers one question per directory: *under corrected identity, what is this job
actually a measurement of, and does the corrected design want it?*

Four outcomes, written to ``salvage_map.csv`` (one row per directory, always):

=========================  ===================================================
``true_positive``          the drug really does bind the accession that was
                           physically docked (per ``interactions_resolved.csv``)
``valid_non_interaction``  a real measurement with no annotated interaction --
                           a candidate for the balanced negative pool
``incomplete``             no ``final_results.csv``: nothing to salvage
``unparseable``            the directory name yielded no identity
=========================  ===================================================

The ``valid_non_interaction`` rows are additionally emitted to
``negatives_pool.csv``, which is the candidate pool the re-run balanced sampler
draws from in Phase 3.

Two further files sit beside the map: ``salvage_map.evidence.csv`` (the per-job
evidence the ``salvage_map`` schema does not declare -- the old campaign's own
sample_type and where each derived value came from, keyed 1:1 on
``old_job_dir``) and ``salvage_scan_summary.json`` (counts, invariant reports,
every unparseable name and every non-directory entry).

The three rules this script exists to honour
--------------------------------------------
1. **The accession in the folder name is physical truth; the gene token is
   not.** ``batch_dock.py:2211`` built the name from opposite sides of the
   mapping bug: the gene came from the TSV's own ``node_2_name`` (a true
   symbol) while the accession came from a last-wins
   ``dict(zip(Mapped_Gene_Name, Entry))`` -- and the accession is what was
   loaded into the docking engines. So the accession is kept, the gene token is
   recorded as *evidence* (``old_gene_label``) and never used to select
   anything. The ``gene`` column is rewritten by
   ``IdentityIndex.label(accession)``.

2. **Nothing is dropped silently.** Every listed entry is accounted for:
   directories become rows (``n_in == n_resolved + n_rejected`` is asserted),
   and a directory whose name cannot be parsed becomes an ``unparseable`` row
   carrying the parser's message -- never a ``continue``. Non-directory entries
   (the campaign has one stray ``docking_automation.log``) are listed by name in
   the summary JSON.

3. **Nothing is written under ``$OLD``.** The old run directory is the only
   copy of 14,893 core-hours. Every output path is checked against it before a
   single byte is written, and the scan itself only ever calls ``scandir`` and
   ``stat``.

Resumability
------------
The listing is one ``scandir``; the cost is the ~24k ``stat`` calls on a slow
USB disk. Those observations -- and only those, because they are the expensive,
immutable part -- are appended to a JSONL checkpoint as they are made, so an
interrupted scan resumes where it stopped. Classification is always recomputed
from the current index and interaction set, so a re-run after rebuilding the
identity index yields fresh answers rather than stale cached ones.

Idempotence
-----------
Rows are ordered by directory name, every tie-break is total, and the writes are
atomic. Two runs over the same tree produce byte-identical files, including
under different ``PYTHONHASHSEED`` values.

Usage (``REBUILD_PLAN.md`` Phase 2)::

    python $REPO/salvage_scan.py \\
        --old-results  $OLD/consensus_docking_results \\
        --index        $NEW/identity/identity_index.parquet \\
        --interactions $NEW/manifests/interactions_resolved.csv \\
        --out-map      $NEW/manifests/salvage_map.csv \\
        --out-pool     $NEW/manifests/negatives_pool.csv

Python 3.9 compatible: no PEP 604 unions at runtime, no ``match``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from dock_selectivity import schemas
from dock_selectivity.identity import IdentityIndex, file_sha256
# dock_selectivity.identity currently declares its OWN InvariantViolation, so
# it is NOT the class schemas/joins/invariants raise. Both are caught at the
# CLI boundary; everything raised here uses the canonical one from invariants.
from dock_selectivity.identity import InvariantViolation as IdentityInvariantViolation
from dock_selectivity.invariants import (
    InvariantViolation,
    assert_collapse_factor_one,
    assert_no_row_growth,
    assert_total_accounting,
    assert_unique_key,
)
from dock_selectivity.jobid import (
    JobIdError,
    JobIdParseError,
    format_job_dir,
    normalize_fragment_id,
    parse_legacy_job_dir,
)
from dock_selectivity.joins import checked_merge_with_report

__all__ = [
    "CLASSIFICATIONS",
    "EVIDENCE_COLUMNS",
    "SALVAGE_MAP_COLUMNS",
    "SALVAGE_MAP_EXTRA_COLUMNS",
    "FATAL_ERRORS",
    "FINAL_RESULTS_FILENAME",
    "DirObservation",
    "ScanResult",
    "list_entries",
    "observe_job_dir",
    "observe_job_dirs",
    "load_interaction_pairs",
    "load_old_plan",
    "classify_jobs",
    "negatives_pool_frame",
    "run_scan",
    "main",
]

logger = logging.getLogger("salvage_scan")

# --------------------------------------------------------------------------
# vocabulary -- kept in lockstep with dock_selectivity.schemas
# --------------------------------------------------------------------------

TRUE_POSITIVE = "true_positive"
VALID_NON_INTERACTION = "valid_non_interaction"
INCOMPLETE = "incomplete"
UNPARSEABLE = "unparseable"

#: Fixed order, used for every count report so the JSON is byte-stable.
CLASSIFICATIONS = (TRUE_POSITIVE, VALID_NON_INTERACTION, INCOMPLETE, UNPARSEABLE)

if set(CLASSIFICATIONS) != set(schemas.CLASSIFICATIONS):  # pragma: no cover
    raise ImportError(
        "salvage_scan classifications {0} disagree with schemas.CLASSIFICATIONS "
        "{1}; the manifest contract and its producer must use one "
        "vocabulary".format(sorted(CLASSIFICATIONS), sorted(schemas.CLASSIFICATIONS))
    )

#: The payload that makes a job salvageable at all.
FINAL_RESULTS_FILENAME = "final_results.csv"

#: Per-job evidence this stage is required to record that the ``salvage_map``
#: schema in ``dock_selectivity/schemas.py`` does not declare.
#:
#: By default these go to a **companion file** (``salvage_map.evidence.csv``,
#: one row per job directory, keyed 1:1 on ``old_job_dir``) so that
#: ``salvage_map.csv`` stays strictly conformant and every reader --
#: ``curate_jobs.py:1204`` calls ``read_checked(path, "salvage_map")`` with no
#: ``allow_extra`` -- keeps working. ``--inline-extras`` folds them into
#: ``salvage_map.csv`` instead, for anyone who would rather declare them in
#: ``schemas.py`` and read with ``allow_extra=True``.
SALVAGE_MAP_EXTRA_COLUMNS = (
    "old_sample_type",
    "old_sample_type_source",
    "is_known_interaction",
    "fragment_source",
    "has_final_results",
)

#: The companion file's columns: the salvage_map key plus the evidence.
EVIDENCE_COLUMNS = ("old_job_dir",) + SALVAGE_MAP_EXTRA_COLUMNS

_SALVAGE_MAP_DTYPES: Tuple[Tuple[str, str], ...] = (
    ("old_job_dir", "string"),
    ("classification", "string"),
    ("drugbank_id", "string"),
    ("accession", "string"),
    ("gene", "string"),
    ("fragment_id", "string"),
    ("cavity_index", "Int64"),
    ("old_gene_label", "string"),
    ("label_was_correct", "boolean"),
    ("new_job_dir", "string"),
    ("excluded_reason", "string"),
    # extras
    ("old_sample_type", "string"),
    ("old_sample_type_source", "string"),
    ("is_known_interaction", "boolean"),
    ("fragment_source", "string"),
    ("has_final_results", "boolean"),
)

SALVAGE_MAP_COLUMNS = tuple(name for name, _ in _SALVAGE_MAP_DTYPES)

#: Why a job is not carried over. Recorded per row; never a bare drop.
POOL_PENDING_REASON = (
    "valid non-interaction: negative-pool candidate. Carried over only if the "
    "balanced sampler draws it (REBUILD_PLAN Phase 3); see negatives_pool.csv"
)
INCOMPLETE_REASON = (
    "no {0} in the job directory: the consensus payload was never written, so "
    "there is nothing to salvage".format(FINAL_RESULTS_FILENAME)
)
UNPARSEABLE_REASON = "unparseable directory name: {0}"
NAME_BUILD_FAILED_REASON = "cannot build a new job directory name: {0}"

#: Where ``old_sample_type`` came from, per row.
SOURCE_OLD_PLAN = "old_plan"
SOURCE_GENE_TOKEN = "gene_token"
SOURCE_UNPARSEABLE = "unparseable"

#: Where ``fragment_id`` came from, per row.
FRAGMENT_FROM_DIR_NAME = "dir_name"
FRAGMENT_FROM_DEFAULT = "default"

#: Columns the old campaign's plan (``required_structures_with_negatives.csv``)
#: must carry for ``--old-plan`` to be usable.
OLD_PLAN_COLUMNS = ("drugbank_id", "UniProt_ID", "Fragment_ID", "Cavity_Index", "sample_type")

#: Failures the CLI reports and exits non-zero on, rather than tracebacking.
#: Both ``InvariantViolation`` classes are listed on purpose: ``identity.py``
#: defines its own, so ``schemas.SchemaViolation`` (which derives from
#: ``invariants.InvariantViolation``) and ``IdentityIndex.load``'s SHA-mismatch
#: error are *different* hierarchies. Catching only one would let half the
#: guard rails escape as tracebacks.
FATAL_ERRORS = (
    InvariantViolation,
    IdentityInvariantViolation,
    JobIdError,
    FileNotFoundError,
    NotADirectoryError,
    RuntimeError,
)

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_TOOL = "salvage_scan"
DEFAULT_CHECKPOINT_NAME = ".salvage_scan_checkpoint.jsonl"
DEFAULT_SUMMARY_NAME = "salvage_scan_summary.json"

_MAX_REPORTED = 10


# --------------------------------------------------------------------------
# value types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DirObservation:
    """The only thing this scan reads off the slow disk, per job directory.

    ``has_final_results`` means the file exists **and is non-empty**. The file
    is never opened: 24,173 x 184 KB is 4.4 GB of USB reads for a fact that a
    ``stat`` already establishes.
    """

    name: str
    has_final_results: bool


@dataclass(frozen=True)
class ScanResult:
    """Everything a caller (or a test) needs to inspect a completed scan."""

    salvage_map: pd.DataFrame
    negatives_pool: pd.DataFrame
    summary: Dict[str, Any]
    out_map: Optional[str] = None
    out_pool: Optional[str] = None
    out_summary: Optional[str] = None
    out_evidence: Optional[str] = None

    def counts(self) -> Dict[str, int]:
        """Classification counts, all four keys always present."""
        return dict(self.summary["counts"]["by_classification"])


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _norm_accession(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def _norm_drugbank_id(value: Any) -> Optional[str]:
    """Normalise a DrugBank id exactly as :class:`JobIdentity` does.

    ``DB\\d+`` is upper-cased; a name slug (222 ``node_1`` rows in
    ``1_drug_to_protein.tsv`` are slugs, not ids) is left alone, so the two
    sides of the membership test agree either way.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text[:2].upper() == "DB" and text[2:].isdigit():
        return text.upper()
    return text


def _series(values: Sequence[Any], dtype: str) -> pd.Series:
    """Build a typed column; ``None`` becomes ``pd.NA``, never the string 'None'."""
    return pd.Series(list(values), dtype=dtype)


def _real(path: Any) -> str:
    return os.path.realpath(os.path.abspath(os.fspath(path)))


def assert_not_under(path: Any, root: Any, label: str) -> str:
    """Refuse to write *path* if it resolves inside *root*.

    ``$OLD`` holds the only copy of 14,893 core-hours of docking and is
    read-only for the whole rebuild (``REBUILD_PLAN.md`` §4.1). This is checked
    before anything is opened for writing, not after.
    """
    target = _real(path)
    base = _real(root)
    if target == base or target.startswith(base.rstrip(os.sep) + os.sep):
        raise InvariantViolation(
            "refusing to write {0} to {1}: it is inside the read-only old run "
            "directory {2}. $OLD is the only copy of the campaign's docking "
            "output; point this output at $NEW.".format(label, target, base),
            name="salvage_scan.read_only_guard",
        )
    return target


def _git_revision(start: str) -> Optional[str]:
    """Best-effort ``git rev-parse HEAD`` for provenance; ``None`` if unavailable."""
    try:
        completed = subprocess.run(
            ["git", "-C", start, "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        return None
    if completed.returncode != 0:
        return None
    revision = completed.stdout.decode("utf-8", "replace").strip()
    return revision or None


def _write_csv(frame: pd.DataFrame, path: str) -> str:
    """Atomic, index-free, LF-terminated CSV. Same rows in, same bytes out."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=parent or ".", delete=False, suffix=".tmp"
    )
    handle.close()
    try:
        frame.to_csv(handle.name, index=False, lineterminator="\n")
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    return path


def _write_json(payload: Any, path: str) -> str:
    """Atomic, sorted-key JSON. Two identical runs produce identical bytes."""
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=parent or ".", delete=False, suffix=".tmp"
    )
    try:
        with handle:
            handle.write(text)
            handle.write("\n")
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    return path


# --------------------------------------------------------------------------
# the read-only scan
# --------------------------------------------------------------------------


def list_entries(old_results: Any) -> Tuple[List[str], List[str]]:
    """Single-level listing of ``$OLD/consensus_docking_results``.

    Returns ``(job_dirs, non_directory_entries)``, both sorted. **One level
    only** -- the campaign has 24,173 job directories on a slow USB disk and
    walking them recursively costs hours for information already in the names.

    Non-directory entries are returned rather than skipped: the campaign
    directory contains one stray ``docking_automation.log``, and "24,174 listed
    = 24,173 dirs + 1 file" is an accounting statement the summary makes
    explicitly.
    """
    root = os.fspath(old_results)
    if not os.path.isdir(root):
        raise NotADirectoryError(
            "--old-results {0} is not a directory. It must point at "
            "$OLD/consensus_docking_results. A missing prerequisite is an "
            "error, never a fallback.".format(root)
        )
    job_dirs: List[str] = []
    others: List[str] = []
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_dir():
                job_dirs.append(entry.name)
            else:
                others.append(entry.name)
    return sorted(job_dirs), sorted(others)


def observe_job_dir(root: Any, name: str) -> DirObservation:
    """One ``stat``: does this job carry a non-empty ``final_results.csv``?"""
    path = os.path.join(os.fspath(root), name, FINAL_RESULTS_FILENAME)
    try:
        size = os.stat(path).st_size
    except FileNotFoundError:
        return DirObservation(name=name, has_final_results=False)
    except NotADirectoryError:
        return DirObservation(name=name, has_final_results=False)
    except OSError as exc:
        # A read error on the campaign drive must be loud: silently calling it
        # "incomplete" would relabel a finished 0.62 core-hour job as garbage.
        raise RuntimeError(
            "could not stat {0}: {1}. Refusing to guess whether this job is "
            "complete. Fix the read error and re-run -- the checkpoint keeps "
            "the work already done.".format(path, exc)
        )
    return DirObservation(name=name, has_final_results=size > 0)


def observe_job_dirs(
    root: Any,
    names: Sequence[str],
    cached: Optional[Dict[str, bool]] = None,
    on_new: Optional[Any] = None,
    progress_every: int = 2000,
) -> List[DirObservation]:
    """Observe every directory in *names*, reusing *cached* observations.

    *on_new* is called with each freshly made :class:`DirObservation` (the
    checkpoint writer) so an interrupted scan does not lose the ``stat`` calls
    it already paid for.
    """
    cached = cached or {}
    observations: List[DirObservation] = []
    n_new = 0
    for position, name in enumerate(names, start=1):
        if name in cached:
            observations.append(
                DirObservation(name=name, has_final_results=bool(cached[name]))
            )
        else:
            observation = observe_job_dir(root, name)
            observations.append(observation)
            n_new += 1
            if on_new is not None:
                on_new(observation)
        if progress_every and position % progress_every == 0:
            logger.info(
                "scanned %d/%d directories (%d new stat calls)",
                position,
                len(names),
                n_new,
            )
    return observations


# --------------------------------------------------------------------------
# checkpoint (resumability)
# --------------------------------------------------------------------------


def read_checkpoint(path: Any, old_results: Any) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    """Load a checkpoint written by a previous run of this scan.

    A checkpoint recorded against a *different* ``--old-results`` directory
    raises rather than being reused: silently mixing observations from two
    trees is exactly the class of universe-widening fallback ``FIX_PLAN.md`` G8
    bans. Delete the file or pass ``--no-resume``.

    A single malformed *final* line is tolerated (the signature of a process
    killed mid-write) and counted; a malformed line anywhere else raises.
    """
    path = os.fspath(path)
    report: Dict[str, Any] = {
        "path": path,
        "existed": False,
        "n_entries": 0,
        "n_truncated_lines": 0,
    }
    if not os.path.exists(path):
        return {}, report
    report["existed"] = True
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    if not lines:
        return {}, report
    try:
        header = json.loads(lines[0])
    except ValueError:
        raise InvariantViolation(
            "checkpoint {0} has an unreadable header line. Delete it or pass "
            "--no-resume; it will not be silently ignored.".format(path),
            name="salvage_scan.checkpoint",
        )
    expected = _real(old_results)
    if (
        not isinstance(header, dict)
        or header.get("tool") != CHECKPOINT_TOOL
        or header.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or header.get("old_results") != expected
    ):
        raise InvariantViolation(
            "checkpoint {0} was written by {1!r} (schema {2}) for old_results "
            "{3!r}, but this run is {4!r} (schema {5}) over {6!r}. Reusing it "
            "would mix observations from two different trees. Delete the "
            "checkpoint or pass --no-resume.".format(
                path,
                header.get("tool") if isinstance(header, dict) else None,
                header.get("schema_version") if isinstance(header, dict) else None,
                header.get("old_results") if isinstance(header, dict) else None,
                CHECKPOINT_TOOL,
                CHECKPOINT_SCHEMA_VERSION,
                expected,
            ),
            name="salvage_scan.checkpoint",
        )
    cached: Dict[str, bool] = {}
    for position, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            if position == len(lines):
                report["n_truncated_lines"] = 1
                logger.warning(
                    "checkpoint %s ends with a truncated record (line %d); "
                    "that directory will simply be re-observed",
                    path,
                    position,
                )
                continue
            raise InvariantViolation(
                "checkpoint {0} line {1} is not valid JSON. Delete the file or "
                "pass --no-resume; a partially readable checkpoint is not "
                "silently accepted.".format(path, position),
                name="salvage_scan.checkpoint",
            )
        cached[str(record["d"])] = bool(record["f"])
    report["n_entries"] = len(cached)
    return cached, report


def _open_checkpoint(path: str, old_results: str, entries: Dict[str, bool]) -> Any:
    """Rewrite the checkpoint from *entries*, then open it for appending.

    Rewriting rather than blindly appending is what makes a *crashed* run
    resumable: the previous file may end mid-record, and appending after a
    truncated line would corrupt it permanently. The rewrite is atomic and its
    bytes are deterministic (header + entries sorted by name).
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    header = json.dumps(
        {
            "tool": CHECKPOINT_TOOL,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "old_results": _real(old_results),
        },
        sort_keys=True,
    )
    staging = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=parent or ".", delete=False, suffix=".tmp"
    )
    try:
        with staging:
            staging.write(header)
            staging.write("\n")
            for name in sorted(entries):
                staging.write(
                    json.dumps({"d": name, "f": int(entries[name])}, sort_keys=True)
                )
                staging.write("\n")
        os.replace(staging.name, path)
    except BaseException:
        try:
            os.unlink(staging.name)
        except OSError:
            pass
        raise
    return open(path, "a", encoding="utf-8")


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def load_interaction_pairs(
    path: Any,
    index: Optional[IdentityIndex] = None,
) -> Tuple[frozenset, Dict[str, Any]]:
    """Read ``interactions_resolved.csv`` into a ``(drugbank_id, accession)`` set.

    Membership in this set is the *only* input to the positive/negative
    decision. The gene column is never read for anything but reporting: the
    2025 campaign's whole failure was a symbol deciding which protein a row
    referred to.

    When *index* is supplied, every accession in the edge list must exist in it.
    A mismatch means the edge list and the index were built from different
    identity artifacts, and every downstream count would be quietly wrong.
    """
    frame = schemas.read_checked(path, "interactions_resolved", allow_extra=True)
    drugs = [_norm_drugbank_id(v) for v in frame["drugbank_id"].tolist()]
    accessions = [_norm_accession(v) for v in frame["accession"].tolist()]
    pairs = frozenset(
        (drug, accession)
        for drug, accession in zip(drugs, accessions)
        if drug is not None and accession is not None
    )
    distinct_accessions = sorted({a for a in accessions if a is not None})
    report: Dict[str, Any] = {
        "path": os.fspath(path),
        "sha256": file_sha256(path),
        "n_rows": int(len(frame)),
        "n_pairs": len(pairs),
        "n_drugs": len({d for d in drugs if d is not None}),
        "n_accessions": len(distinct_accessions),
    }
    if index is not None:
        unknown = [a for a in distinct_accessions if a not in index]
        report["n_accessions_not_in_index"] = len(unknown)
        if unknown:
            raise InvariantViolation(
                "{0} of {1} accessions in {2} are absent from the identity "
                "index ({3}...). The edge list and the index were built from "
                "different identity artifacts; every classification below "
                "would inherit the disagreement. Rebuild "
                "interactions_resolved.csv against this index.".format(
                    len(unknown),
                    len(distinct_accessions),
                    os.fspath(path),
                    ", ".join(unknown[:_MAX_REPORTED]),
                ),
                name="salvage_scan.interactions",
            )
    return pairs, report


def _coerce_cavity_index(value: Any) -> Optional[int]:
    """Cavity index from a CSV cell. Missing -> ``None`` (the row is then
    dropped by the null-key guard and counted); malformed -> raises."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "<na>"):
        return None
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit():
        raise InvariantViolation(
            "cavity index {0!r} is not a non-negative integer".format(value),
            name="salvage_scan.old_plan",
        )
    return int(text)


def load_old_plan(path: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read the old campaign's plan, for the ``old_sample_type`` evidence column.

    The plan is ``required_structures_with_negatives.csv``: columns
    ``drugbank_id, UniProt_ID, Fragment_ID, Cavity_Index, sample_type``. It is
    used **only** to recover the label the old campaign gave each job. It never
    influences the corrected classification -- that comes from
    ``interactions_resolved.csv`` alone.

    Two hazards are handled explicitly:

    * ``Fragment_ID`` is ``'F1'`` on positives and a bare ``'1'`` on negatives
      (``ASSESSMENT`` §2 Step 6). Both sides are normalised through
      :func:`normalize_fragment_id` before anything is joined.
    * Exact duplicate rows are collapsed; a key claiming *both* labels is an
      unresolvable contradiction and raises, naming the offenders, rather than
      letting a ``dict``/merge pick the last one.
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "--old-plan {0} does not exist. Omit the flag to derive "
            "old_sample_type from the directory name instead.".format(path)
        )
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    missing = [c for c in OLD_PLAN_COLUMNS if c not in raw.columns]
    if missing:
        raise InvariantViolation(
            "--old-plan {0} is missing column(s) {1}; it must be the campaign's "
            "required_structures_with_negatives.csv, which declares {2}. "
            "Present: {3}".format(
                path, missing, list(OLD_PLAN_COLUMNS), list(raw.columns)
            ),
            name="salvage_scan.old_plan",
        )
    sample_types = [str(v).strip().lower() for v in raw["sample_type"].tolist()]
    bad = sorted({s for s in sample_types if s not in schemas.SAMPLE_TYPES})
    if bad:
        raise InvariantViolation(
            "--old-plan {0} carries sample_type value(s) {1} outside {2}. A "
            "label outside the declared vocabulary is not coerced.".format(
                path, bad[:_MAX_REPORTED], list(schemas.SAMPLE_TYPES)
            ),
            name="salvage_scan.old_plan",
        )
    plan = pd.DataFrame(
        {
            "drugbank_id": _series(
                [_norm_drugbank_id(v) for v in raw["drugbank_id"].tolist()], "string"
            ),
            "accession": _series(
                [_norm_accession(v) for v in raw["UniProt_ID"].tolist()], "string"
            ),
            "fragment_id": _series(
                [normalize_fragment_id(v) for v in raw["Fragment_ID"].tolist()],
                "string",
            ),
            "cavity_index": _series(
                [_coerce_cavity_index(v) for v in raw["Cavity_Index"].tolist()],
                "Int64",
            ),
            "old_sample_type": _series(sample_types, "string"),
        }
    )
    n_raw = len(plan)
    # A plan row with a null key would join to the null keys of the unparseable
    # job rows (pandas matches NaN to NaN) and fabricate a label. Drop those
    # rows and say how many, rather than letting them match anything.
    complete_key = plan[
        ["drugbank_id", "accession", "fragment_id", "cavity_index"]
    ].notna().all(axis=1)
    n_null_key = int((~complete_key).sum())
    plan = plan[complete_key]
    plan = plan.drop_duplicates().reset_index(drop=True)
    key = (
        plan["drugbank_id"].astype(str)
        + "|"
        + plan["accession"].astype(str)
        + "|"
        + plan["fragment_id"].astype(str)
        + "|"
        + plan["cavity_index"].astype(str)
    )
    collapse = assert_collapse_factor_one(
        pd.DataFrame({"key": key, "old_sample_type": plan["old_sample_type"]}),
        name="salvage_scan old_plan (drug, accession, fragment, cavity) -> sample_type",
    )
    report = {
        "path": path,
        "sha256": file_sha256(path),
        "n_rows": int(n_raw),
        "n_dropped_null_key": n_null_key,
        "n_distinct_rows": int(len(plan)),
        "collapse": collapse,
    }
    if n_null_key:
        logger.warning(
            "--old-plan %s: %d row(s) have an incomplete "
            "(drug, accession, fragment, cavity) key and cannot label any job; "
            "they are excluded from the join and counted in the summary",
            path,
            n_null_key,
        )
    return plan, report


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


def _row_template(name: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {column: None for column in SALVAGE_MAP_COLUMNS}
    row["old_job_dir"] = name
    return row


def classify_jobs(
    observations: Iterable[DirObservation],
    index: IdentityIndex,
    interaction_pairs: frozenset,
    default_fragment: str = "F1",
) -> pd.DataFrame:
    """Classify observed job directories. Pure: no filesystem, no network.

    One row out per observation in, in the order given. The accession decides
    everything; ``old_gene_label`` is carried as evidence only.

    ``default_fragment`` is an **explicit, recorded** choice. 22,430 of the
    campaign's directories carry no fragment token because their receptor was
    the single-fragment ``AF-{accession}-F1-model_v1`` model.
    :func:`parse_legacy_job_dir` deliberately refuses to infer that, so the
    fragment is supplied here and every row records whether it came from the
    name (``fragment_source == 'dir_name'``) or from this default.
    """
    fragment_default = normalize_fragment_id(default_fragment)
    if fragment_default is None:
        raise JobIdError(
            "default_fragment must be a fragment id such as 'F1', got "
            "{0!r}. A job with no fragment cannot be named, and inferring one "
            "silently is what fanned a merge out by 410,087 rows.".format(
                default_fragment
            )
        )
    accessions = index.accessions()

    rows: List[Dict[str, Any]] = []
    for observation in observations:
        row = _row_template(observation.name)
        row["has_final_results"] = bool(observation.has_final_results)
        try:
            identity = parse_legacy_job_dir(observation.name, accessions)
        except (JobIdParseError, JobIdError) as exc:
            row["classification"] = UNPARSEABLE
            row["excluded_reason"] = UNPARSEABLE_REASON.format(exc)
            row["old_sample_type_source"] = SOURCE_UNPARSEABLE
            rows.append(row)
            continue

        # The accession token was matched against index.accessions(), so this
        # lookup cannot fail -- the label is derived FROM the key, never used
        # to find it.
        protein = index.by_accession(identity.accession)
        fragment_id = identity.fragment_id
        fragment_source = FRAGMENT_FROM_DIR_NAME
        if fragment_id is None:
            fragment_id = fragment_default
            fragment_source = FRAGMENT_FROM_DEFAULT

        is_known = (identity.drugbank_id, identity.accession) in interaction_pairs

        row.update(
            {
                "drugbank_id": identity.drugbank_id,
                "accession": identity.accession,
                "gene": protein.primary_symbol,
                "fragment_id": fragment_id,
                "cavity_index": identity.cavity_index,
                "old_gene_label": identity.gene,
                "label_was_correct": (
                    None if identity.gene is None else protein.has_symbol(identity.gene)
                ),
                "is_known_interaction": bool(is_known),
                "fragment_source": fragment_source,
            }
        )

        if not observation.has_final_results:
            row["classification"] = INCOMPLETE
            row["excluded_reason"] = INCOMPLETE_REASON
        elif is_known:
            row["classification"] = TRUE_POSITIVE
            try:
                row["new_job_dir"] = format_job_dir(
                    identity.drugbank_id,
                    identity.accession,
                    protein.primary_symbol,
                    fragment_id,
                    identity.cavity_index,
                )
            except JobIdError as exc:
                row["excluded_reason"] = NAME_BUILD_FAILED_REASON.format(exc)
        else:
            row["classification"] = VALID_NON_INTERACTION
            row["excluded_reason"] = POOL_PENDING_REASON
        rows.append(row)

    frame = pd.DataFrame(
        {
            column: _series([row[column] for row in rows], dtype)
            for column, dtype in _SALVAGE_MAP_DTYPES
        },
        columns=list(SALVAGE_MAP_COLUMNS),
    )
    return frame


def attach_old_sample_type(
    frame: pd.DataFrame,
    plan: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fill ``old_sample_type`` -- the label the *old* campaign gave each job.

    Two sources, one recorded per row in ``old_sample_type_source``:

    ``old_plan``
        matched in ``required_structures_with_negatives.csv`` on the full
        ``(drug, accession, fragment, cavity)`` key.
    ``gene_token``
        derived from the directory name: ``generate_negative_samples.py:375``
        read a ``Gene_Name`` column that does not exist in
        ``cavity_mapping.csv``, so **every** negative got a blank gene and every
        negative directory carries the literal token ``nan``. 11,806 directories
        have that token and 11,806 jobs were labelled negative -- the rule is
        exact for this campaign, and it is applied explicitly rather than
        assumed.

    This column is evidence. It never touches the classification.
    """
    n_before = len(frame)
    report: Dict[str, Any] = {"n_rows": int(n_before), "merge": None}
    parsed = frame["classification"] != UNPARSEABLE

    if plan is not None and len(plan):
        merged, merge_report = checked_merge_with_report(
            frame.drop(columns=["old_sample_type"]),
            plan,
            on=["drugbank_id", "accession", "fragment_id", "cavity_index"],
            how="left",
            name="salvage_scan old_sample_type <- old plan",
            allow_growth=False,
        )
        report["merge"] = merge_report
        frame = merged[list(SALVAGE_MAP_COLUMNS)]
        assert_no_row_growth(
            n_before,
            frame,
            name="salvage_scan old_plan merge",
            allow_shrinkage=False,
        )

    from_plan = frame["old_sample_type"].notna()
    # The gene token is missing (written as 'nan') on exactly the jobs the old
    # sampler produced; a real symbol means the old campaign called it positive.
    token_rule = frame["old_gene_label"].isna().map(
        {True: "negative", False: "positive"}
    )
    filled = frame["old_sample_type"].where(from_plan, token_rule.where(parsed))
    source = pd.Series(
        [SOURCE_UNPARSEABLE] * len(frame), index=frame.index, dtype="string"
    )
    source = source.where(~parsed, SOURCE_GENE_TOKEN)
    source = source.where(~from_plan, SOURCE_OLD_PLAN)

    frame = frame.copy()
    frame["old_sample_type"] = _series(filled.tolist(), "string")
    frame["old_sample_type_source"] = _series(source.tolist(), "string")
    report["n_from_old_plan"] = int(from_plan.sum())
    report["n_from_gene_token"] = int(((~from_plan) & parsed).sum())
    report["n_unparseable"] = int((~parsed).sum())
    return frame, report


def negatives_pool_frame(salvage_map: pd.DataFrame) -> pd.DataFrame:
    """The negative candidate pool: exactly the ``valid_non_interaction`` rows.

    Emitted separately (``REBUILD_PLAN.md`` §2 ``manifests/negatives_pool.csv``)
    and strictly conformant to the ``negatives_pool`` schema, so the balanced
    sampler can read it with ``read_checked`` and nothing else.

    Note what this pool is *not*: it is not a set of proven non-binders, and it
    is not all drawable. A target with zero positives in the corrected design
    must receive zero negatives, so a large part of the pool is structurally
    ineligible. Eligibility is the sampler's judgement, not this scan's.
    """
    rows = salvage_map[salvage_map["classification"] == VALID_NON_INTERACTION]
    pool = pd.DataFrame(
        {
            "old_job_dir": rows["old_job_dir"].astype("string"),
            "drugbank_id": rows["drugbank_id"].astype("string"),
            "accession": rows["accession"].astype("string"),
            "gene": rows["gene"].astype("string"),
            "fragment_id": rows["fragment_id"].astype("string"),
            "cavity_index": rows["cavity_index"].astype("int64"),
            "old_sample_type": rows["old_sample_type"].astype("string"),
            "old_gene_label": rows["old_gene_label"].astype("string"),
            "classification": rows["classification"].astype("string"),
            "has_final_results": rows["has_final_results"].astype("boolean"),
        }
    ).reset_index(drop=True)
    return pool


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def _count(frame: pd.DataFrame, column: str, vocabulary: Sequence[str]) -> Dict[str, int]:
    """Value counts with a fixed key order, zeros included."""
    observed = frame[column].value_counts(dropna=False)
    counts = {key: int(observed.get(key, 0)) for key in vocabulary}
    extra = {
        str(key): int(value)
        for key, value in observed.items()
        if not pd.isna(key) and str(key) not in counts
    }
    counts.update(extra)
    n_null = int(frame[column].isna().sum())
    if n_null:
        counts["<null>"] = n_null
    return counts


def run_scan(
    old_results: Any,
    index_path: Any,
    interactions_path: Any,
    out_map: Optional[Any] = None,
    out_pool: Optional[Any] = None,
    out_summary: Optional[Any] = None,
    out_evidence: Optional[Any] = None,
    inline_extras: bool = False,
    checkpoint: Optional[Any] = None,
    resume: bool = True,
    old_plan: Optional[Any] = None,
    default_fragment: str = "F1",
    index_sha256: Optional[str] = None,
    progress_every: int = 2000,
    code_revision: Optional[str] = None,
) -> ScanResult:
    """Run the whole Phase-2 scan and (optionally) write its artifacts."""
    old_results = os.fspath(old_results)

    # Guard before anything is opened for writing.
    targets = {
        "--out-map": out_map,
        "--out-pool": out_pool,
        "--out-summary": out_summary,
        "--out-evidence": out_evidence,
        "--checkpoint": checkpoint,
    }
    for label, target in targets.items():
        if target is not None:
            assert_not_under(target, old_results, label)
    if out_map is not None and out_pool is not None and _real(out_map) == _real(out_pool):
        raise InvariantViolation(
            "--out-map and --out-pool resolve to the same file ({0}); one would "
            "overwrite the other".format(_real(out_map)),
            name="salvage_scan.outputs",
        )

    index = IdentityIndex.load(index_path, expect_sha256=index_sha256)
    index_report = {
        "path": os.fspath(index_path),
        "sha256": index.source_sha256,
        "n_accessions": len(index),
    }
    logger.info(
        "identity index: %d accessions from %s (sha256 %s)",
        len(index),
        index_report["path"],
        (index.source_sha256 or "")[:12],
    )

    interaction_pairs, interactions_report = load_interaction_pairs(
        interactions_path, index=index
    )
    logger.info(
        "interactions: %d (drug, accession) pairs over %d drugs and %d accessions",
        interactions_report["n_pairs"],
        interactions_report["n_drugs"],
        interactions_report["n_accessions"],
    )

    plan_frame = None
    plan_report = None
    if old_plan is not None:
        plan_frame, plan_report = load_old_plan(old_plan)
        logger.info(
            "old plan: %d rows (%d distinct) from %s",
            plan_report["n_rows"],
            plan_report["n_distinct_rows"],
            plan_report["path"],
        )
    else:
        logger.warning(
            "no --old-plan given: old_sample_type is derived from the directory "
            "name (gene token 'nan' == the old campaign's negative). The column "
            "is evidence only and never affects classification; every row "
            "records its source in old_sample_type_source."
        )

    job_dirs, non_dirs = list_entries(old_results)
    logger.info(
        "%s: %d entries = %d job directories + %d non-directory entries",
        old_results,
        len(job_dirs) + len(non_dirs),
        len(job_dirs),
        len(non_dirs),
    )
    if non_dirs:
        logger.info(
            "non-directory entries (not jobs, listed in the summary): %s",
            ", ".join(non_dirs[:_MAX_REPORTED]),
        )

    cached: Dict[str, bool] = {}
    checkpoint_report: Dict[str, Any] = {
        "path": None,
        "existed": False,
        "n_entries": 0,
        "n_truncated_lines": 0,
    }
    handle = None
    try:
        if checkpoint is not None:
            checkpoint = os.fspath(checkpoint)
            if resume:
                cached, checkpoint_report = read_checkpoint(checkpoint, old_results)
            else:
                checkpoint_report = {
                    "path": checkpoint,
                    "existed": os.path.exists(checkpoint),
                    "n_entries": 0,
                    "n_truncated_lines": 0,
                }
            handle = _open_checkpoint(checkpoint, old_results, cached)
            if cached:
                logger.info(
                    "resuming: %d of %d directories already observed",
                    len(cached),
                    len(job_dirs),
                )

        def record(observation: DirObservation) -> None:
            if handle is None:
                return
            handle.write(
                json.dumps(
                    {"d": observation.name, "f": int(observation.has_final_results)},
                    sort_keys=True,
                )
            )
            handle.write("\n")

        observations = observe_job_dirs(
            old_results,
            job_dirs,
            cached=cached,
            on_new=record,
            progress_every=progress_every,
        )
    finally:
        if handle is not None:
            handle.close()

    n_cached_used = sum(1 for name in job_dirs if name in cached)

    salvage_map = classify_jobs(
        observations,
        index=index,
        interaction_pairs=interaction_pairs,
        default_fragment=default_fragment,
    )
    salvage_map, sample_type_report = attach_old_sample_type(salvage_map, plan_frame)

    # ---- invariants -------------------------------------------------------
    invariants: List[Dict[str, Any]] = []
    invariants.append(
        assert_no_row_growth(
            len(job_dirs),
            salvage_map,
            name="salvage_scan one row per job directory",
            allow_shrinkage=False,
        )
    )
    n_unparseable = int((salvage_map["classification"] == UNPARSEABLE).sum())
    invariants.append(
        assert_total_accounting(
            n_in=len(job_dirs),
            n_resolved=len(salvage_map) - n_unparseable,
            n_rejected=n_unparseable,
            name="salvage_scan directories -> classified + unparseable",
        )
    )
    invariants.append(
        assert_unique_key(
            salvage_map, ["old_job_dir"], name="salvage_scan salvage_map key"
        )
    )
    # The failure this whole rebuild exists for: a gene symbol standing in for
    # an accession collapses many proteins onto one. Every directory yields
    # exactly one accession, and that mapping must be a function.
    parsed = salvage_map[salvage_map["classification"] != UNPARSEABLE]
    invariants.append(
        assert_collapse_factor_one(
            parsed[["old_job_dir", "accession"]],
            name="salvage_scan old_job_dir -> accession",
        )
    )
    carried = salvage_map[salvage_map["new_job_dir"].notna()]
    invariants.append(
        assert_unique_key(
            carried, ["new_job_dir"], name="salvage_scan new_job_dir uniqueness"
        )
    )
    if plan_report is not None:
        invariants.append(plan_report["collapse"])
    if sample_type_report.get("merge") is not None:
        invariants.append(sample_type_report["merge"])

    pool = negatives_pool_frame(salvage_map)

    # ---- summary ----------------------------------------------------------
    label_known = salvage_map["label_was_correct"].notna()
    counts = {
        "n_entries_listed": len(job_dirs) + len(non_dirs),
        "n_directories": len(job_dirs),
        "n_non_directory_entries": len(non_dirs),
        "n_rows": int(len(salvage_map)),
        "by_classification": _count(salvage_map, "classification", CLASSIFICATIONS),
        "n_carried": int(salvage_map["new_job_dir"].notna().sum()),
        "n_excluded": int(salvage_map["excluded_reason"].notna().sum()),
        "n_pool_rows": int(len(pool)),
        "n_with_final_results": int(salvage_map["has_final_results"].sum()),
        "n_without_final_results": int((~salvage_map["has_final_results"]).sum()),
        "n_cached_observations": n_cached_used,
        "n_new_observations": len(job_dirs) - n_cached_used,
        "n_stale_checkpoint_entries": len(cached) - n_cached_used,
        "label_was_correct": {
            "true": int((salvage_map["label_was_correct"] == True).sum()),  # noqa: E712
            "false": int((salvage_map["label_was_correct"] == False).sum()),  # noqa: E712
            "unknown_no_label": int((~label_known).sum()),
        },
        "old_sample_type": _count(salvage_map, "old_sample_type", schemas.SAMPLE_TYPES),
        "old_sample_type_source": _count(
            salvage_map,
            "old_sample_type_source",
            (SOURCE_OLD_PLAN, SOURCE_GENE_TOKEN, SOURCE_UNPARSEABLE),
        ),
        "fragment_source": _count(
            salvage_map,
            "fragment_source",
            (FRAGMENT_FROM_DIR_NAME, FRAGMENT_FROM_DEFAULT),
        ),
        "n_distinct_accessions": int(parsed["accession"].nunique()),
        "n_distinct_drugs": int(parsed["drugbank_id"].nunique()),
        "n_non_drugbank_drug_tokens": int(
            (~parsed["drugbank_id"].str.match(r"^DB\d+$", na=False)).sum()
        ),
    }
    summary: Dict[str, Any] = {
        "tool": "salvage_scan.py",
        "schema_version": 1,
        "code_revision": (
            code_revision
            if code_revision is not None
            else _git_revision(os.path.dirname(os.path.abspath(__file__)))
        ),
        "old_results": _real(old_results),
        "default_fragment": normalize_fragment_id(default_fragment),
        "identity_index": index_report,
        "interactions": interactions_report,
        "old_plan": plan_report,
        "checkpoint": checkpoint_report,
        "counts": counts,
        "non_directory_entries": non_dirs,
        "unparseable": [
            {"old_job_dir": row.old_job_dir, "reason": row.excluded_reason}
            for row in salvage_map[
                salvage_map["classification"] == UNPARSEABLE
            ].itertuples()
        ],
        "invariants": invariants,
        "outputs": {
            "salvage_map": None if out_map is None else _real(out_map),
            "negatives_pool": None if out_pool is None else _real(out_pool),
            "salvage_map_evidence": (
                None if (out_evidence is None or inline_extras) else _real(out_evidence)
            ),
            "extra_columns": (
                "inline" if inline_extras else "companion_file"
            ),
        },
    }

    logger.info(
        "classified %d directories: %s",
        len(salvage_map),
        ", ".join(
            "{0}={1}".format(key, counts["by_classification"][key])
            for key in CLASSIFICATIONS
        ),
    )
    if n_unparseable:
        logger.warning(
            "%d directory name(s) did not parse; REBUILD_PLAN Phase 2 expects 0. "
            "They are recorded in salvage_map.csv as 'unparseable' with a "
            "reason, never dropped.",
            n_unparseable,
        )

    written_map = written_pool = written_summary = written_evidence = None
    if out_map is not None:
        declared = list(schemas.get_schema("salvage_map").column_names)
        written_map = schemas.write_checked(
            salvage_map if inline_extras else salvage_map[declared],
            os.fspath(out_map),
            "salvage_map",
            allow_extra=inline_extras,
        )
        logger.info("wrote %d rows to %s", len(salvage_map), written_map)
        if not inline_extras and out_evidence is not None:
            written_evidence = _write_csv(
                salvage_map[list(EVIDENCE_COLUMNS)], os.fspath(out_evidence)
            )
            logger.info(
                "wrote %d rows to %s (per-job evidence the salvage_map schema "
                "does not declare, keyed 1:1 on old_job_dir)",
                len(salvage_map),
                written_evidence,
            )
    if out_pool is not None:
        written_pool = schemas.write_checked(
            pool, os.fspath(out_pool), "negatives_pool"
        )
        logger.info("wrote %d rows to %s", len(pool), written_pool)
    if out_summary is not None:
        written_summary = _write_json(summary, os.fspath(out_summary))
        logger.info("wrote %s", written_summary)

    return ScanResult(
        salvage_map=salvage_map,
        negatives_pool=pool,
        summary=summary,
        out_map=written_map,
        out_pool=written_pool,
        out_summary=written_summary,
        out_evidence=written_evidence,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="salvage_scan.py",
        description=(
            "REBUILD_PLAN Phase 2: classify every old job directory under "
            "corrected identity, read-only. Writes salvage_map.csv (one row per "
            "directory) and negatives_pool.csv (the valid non-interactions)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Nothing under --old-results is written, moved or renamed; output "
            "paths that resolve inside it are refused before any file is "
            "opened. The scan is resumable: filesystem observations are "
            "checkpointed as they are made."
        ),
    )
    parser.add_argument(
        "--old-results",
        required=True,
        help="$OLD/consensus_docking_results (read-only). Listed one level deep.",
    )
    parser.add_argument(
        "--index",
        required=True,
        help="identity_index.parquet. The accession set is the parser's authority.",
    )
    parser.add_argument(
        "--interactions",
        required=True,
        help=(
            "manifests/interactions_resolved.csv. Membership of "
            "(drugbank_id, accession) is the only input to the positive decision."
        ),
    )
    parser.add_argument("--out-map", required=True, help="salvage_map.csv to write")
    parser.add_argument("--out-pool", required=True, help="negatives_pool.csv to write")
    parser.add_argument(
        "--out-summary",
        default=None,
        help=(
            "scan summary JSON (counts, invariant reports, every unparseable "
            "name and every non-directory entry). Default: {0} beside "
            "--out-map.".format(DEFAULT_SUMMARY_NAME)
        ),
    )
    parser.add_argument(
        "--out-evidence",
        default=None,
        help=(
            "companion CSV carrying the per-job evidence the salvage_map schema "
            "does not declare ({0}), keyed 1:1 on old_job_dir. Default: "
            "<--out-map>.evidence.csv".format(", ".join(SALVAGE_MAP_EXTRA_COLUMNS))
        ),
    )
    parser.add_argument(
        "--inline-extras",
        action="store_true",
        help=(
            "write those columns into salvage_map.csv instead of the companion "
            "file. Readers must then pass allow_extra=True; curate_jobs.py:1204 "
            "does not, which is why this is off by default."
        ),
    )
    parser.add_argument(
        "--index-sha256",
        default=None,
        help=(
            "pin the identity index: refuse to run against any other build "
            "(FIX_PLAN §1)."
        ),
    )
    parser.add_argument(
        "--old-plan",
        default=None,
        help=(
            "the campaign's required_structures_with_negatives.csv, used only "
            "to recover each job's ORIGINAL sample_type. Without it that column "
            "is derived from the directory name (gene token 'nan' == negative) "
            "and every row records which source it used. It never affects the "
            "corrected classification."
        ),
    )
    parser.add_argument(
        "--default-fragment",
        default="F1",
        help=(
            "fragment assigned to directories whose name carries no fragment "
            "token (22,430 of 24,173; their receptor was the single-fragment "
            "AF-{accession}-F1-model_v1). Recorded per row as "
            "fragment_source='default'. Default: F1"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "JSONL checkpoint of filesystem observations, for resuming an "
            "interrupted scan. Default: {0} beside --out-map.".format(
                DEFAULT_CHECKPOINT_NAME
            )
        ),
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="ignore and rewrite the checkpoint; re-stat every directory.",
    )
    parser.add_argument(
        "--no-checkpoint",
        dest="use_checkpoint",
        action="store_false",
        help="do not read or write a checkpoint at all.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        help="log progress every N directories (0 disables). Default: 2000",
    )
    parser.add_argument(
        "--code-revision",
        default=None,
        help="override the git revision recorded in the summary.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Default: INFO",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    manifests_dir = os.path.dirname(os.path.abspath(args.out_map))
    summary_path = args.out_summary or os.path.join(manifests_dir, DEFAULT_SUMMARY_NAME)
    evidence_path = args.out_evidence or (
        os.path.splitext(os.path.abspath(args.out_map))[0] + ".evidence.csv"
    )
    checkpoint_path = None
    if args.use_checkpoint:
        checkpoint_path = args.checkpoint or os.path.join(
            manifests_dir, DEFAULT_CHECKPOINT_NAME
        )
    try:
        run_scan(
            old_results=args.old_results,
            index_path=args.index,
            interactions_path=args.interactions,
            out_map=args.out_map,
            out_pool=args.out_pool,
            out_summary=summary_path,
            out_evidence=evidence_path,
            inline_extras=args.inline_extras,
            checkpoint=checkpoint_path,
            resume=args.resume,
            old_plan=args.old_plan,
            default_fragment=args.default_fragment,
            index_sha256=args.index_sha256,
            progress_every=args.progress_every,
            code_revision=args.code_revision,
        )
    except FATAL_ERRORS as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
