#!/usr/bin/env python3
"""Materialise the curated docking directory -- ``REBUILD_PLAN.md`` Phase 5.

What this does
--------------
Reads three manifests and copies the subset of ``$OLD/consensus_docking_results``
that the *corrected* design actually wants into ``$NEW/docking_results``, under
the new naming scheme, with an authoritative ``job_identity.json`` in every
directory.

* ``--job-classification``   one row per old job dir, classified under correct identity
* ``--positives``     the positive (drug, accession, fragment, cavity) jobs wanted
* ``--drawn``         what the balanced sampler drew; ``source='pool'`` rows name
                      the old directory they are reused from

  positives carried  = reuse rows classified ``true_positive`` whose pocket key
                       appears in ``positives_planned.csv``
  negatives carried  = ``negatives_drawn.csv`` rows with ``source == 'pool'``

Everything else in ``$OLD`` is surplus and is simply not copied.  It is not
deleted, moved, or touched -- see "read-only" below.

The three structural fixes it applies
-------------------------------------
1. **The name can no longer disagree with the receptor.**  The old scheme
   ``DB00035_AVPR2_Q8NCT1_cavity_2`` put the gene and the accession in one
   string, sourced from opposite sides of the mapping bug (Q8NCT1 is ARRDC4;
   the real AVPR2 is P30518).  Here the directory name is built by
   ``jobid.format_job_dir`` with ``gene = IdentityIndex.label(accession)``, so
   the label is a *function of* the accession rather than an independent claim.
   ``--index`` is therefore required: there is no path in which the gene comes
   from a manifest column.

2. **The name is no longer the data contract.**  ``job_identity.json`` is
   written into every curated directory and carries ``provenance`` and
   ``source_job``, so the relabelling is auditable rather than a leap of faith.

3. **The 2.4% rule.**  ``--evidence-only`` copies the measurements
   (``final_results.csv``, the four ``results.csv``, ``out.sdf`` / ``out.dok`` /
   the GOLD poses, ``input/*.mol2``) and skips the ~45 MB of ``complex_*.pdb``
   per job, which are derived (receptor + pose) and reconstructible.  Retry logs
   are skipped too; provenance lives in ``job_identity.json`` now.

   One measured correction to ``REBUILD_PLAN.md`` section 1.3: taking the whole
   ``gold/output/`` tree costs **3.09 MB**, not the ~180 KB the plan's table
   assumes, because it also holds two further copies of the receptor
   (``gold_protein.mol2`` + ``plp_protein.mol2``, 1.47 MB, already carried as
   ``input/receptor.mol2``), the cavity fitting points, 424 KB of GOLD logs and
   a byte-duplicate pose set (``ranked_*.sdf``).  ``--gold-output poses``
   (the default) takes ``gold_soln_*.sdf`` + rankings = 363 KB, which lands
   total evidence at ~1.21 MB/job -- the plan's ~1.14 MB / 2.4%.
   ``--gold-output all`` takes the tree and costs ~4.52 MB/job (9.1%): 62 GB
   over 13,730 reused jobs instead of 19 GB.  Measured on six real job
   directories, read-only.

Read-only source, safe posture
------------------------------
``--dry-run`` prints the plan and writes **nothing**; run it first, every time.
The source tree is only ever *read*: this module never calls ``os.rename``,
``os.link``, ``shutil.move`` or any unlink on a path under ``--src``, and it
refuses to run if ``--dst`` is inside ``--src`` (or the reverse).  Copies are
verified by size after write.  ``$OLD`` is 14,893 core-hours and is the only
copy until Phase 8 passes.

Resumable
---------
``job_identity.json`` is written **last**, atomically, and records the evidence
file list with sizes.  A re-run therefore skips a job iff its destination is
complete and self-consistent, and a killed run leaves incomplete directories
that the next run redoes.  Nothing about the resume decision requires reading
the slow source drive.

Usage::

    python3 curate_jobs.py --dry-run \\
        --index       $NEW/identity/identity_index.parquet \\
        --job-classification $NEW/manifests/job_classification.csv \\
        --drawn       $NEW/manifests/negatives_drawn.csv \\
        --positives   $NEW/manifests/positives_planned.csv \\
        --src $OLD/consensus_docking_results \\
        --dst $NEW/docking_results \\
        --evidence-only

    python3 curate_jobs.py ...same flags, minus --dry-run...

The shared runtime flags apply (``--out-dir``, ``--log-file``, ``--log-level``,
``--quiet``, ``--notify``, ``--notify-on``, ``--notify-title``).  The log lands in
``<run-dir>/steps/curate_jobs/curate_jobs.log``, where the run directory is the
parent of ``--dst``; ``--dry-run`` writes no log either, because it writes
nothing.  A copy of 13,730 job directories is worth a notification::

    PUSHOVER_TOKEN=... PUSHOVER_USER_KEY=... \\
        python3 curate_jobs.py --notify --notify-on all ...

Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import fnmatch
import glob
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:  # `python $REPO/curate_jobs.py` from any cwd
    sys.path.insert(0, _REPO_ROOT)

from dock_selectivity.identity import IdentityIndex, InvariantViolation  # noqa: E402
from dock_selectivity.invariants import (  # noqa: E402
    assert_collapse_factor_one,
    assert_total_accounting,
    write_invariant_report,
)
from dock_selectivity.jobid import (  # noqa: E402
    JOB_IDENTITY_FILENAME,
    JobIdError,
    JobIdentity,
    format_job_dir,
    normalize_fragment_id,
    read_job_identity_record,
    write_job_identity,
)
from dock_selectivity.runtime import (  # noqa: E402
    add_runtime_arguments,
    resolve_out_dir,
    step,
)
from dock_selectivity.schemas import read_checked, write_checked  # noqa: E402

__all__ = [
    "EvidenceRule",
    "EVIDENCE_RULES",
    "GOLD_OUTPUT_MODES",
    "evidence_rules",
    "EXCLUDED_BASENAME_PATTERNS",
    "CurationError",
    "JobPlan",
    "Reject",
    "CurationPlan",
    "CopyOutcome",
    "is_excluded_basename",
    "select_evidence_files",
    "select_all_files",
    "build_plan",
    "copy_job",
    "job_provenance",
    "execute_plan",
    "main",
    "STEP_NAME",
]

#: Names the logger, the log file and the step directory
#: (``<run-dir>/steps/curate_jobs/curate_jobs.log``). One string, so they cannot drift.
STEP_NAME = "curate_jobs"

logger = logging.getLogger(STEP_NAME)

#: Classification a reuse row must carry to be carried over as a positive.
TRUE_POSITIVE = "true_positive"
#: ...and as a negative.
VALID_NON_INTERACTION = "valid_non_interaction"

#: The pocket key every planning manifest is keyed on.
POCKET_KEY = ("drugbank_id", "accession", "fragment_id", "cavity_index")
#: ...minus ``fragment_id``, which is the one component a legacy directory may
#: legitimately lack and which therefore gets its own, actionable rejection.
IDENTITY_COLUMNS = tuple(c for c in POCKET_KEY if c != "fragment_id")

RETENTION_EVIDENCE = "evidence_only"
RETENTION_FULL = "full"

#: Severity of a rejected candidate.  ``by_design`` rows are the design saying
#: "not wanted"; ``error`` rows are two stages disagreeing, or the source not
#: being what the manifest says it is.  A run with any ``error`` row fails.
SEVERITY_BY_DESIGN = "by_design"
SEVERITY_ERROR = "error"


class CurationError(Exception):
    """The curation cannot proceed as specified."""


# --------------------------------------------------------------------------
# the 2.4% rule -- which files constitute the evidence
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceRule:
    """One selector for files worth carrying over.

    ``kind`` is one of:

    ``exact``
        *pattern* is a single path relative to the job directory.
    ``glob``
        *pattern* is a non-recursive glob relative to the job directory.
    ``tree``
        *pattern* is a subdirectory; every file beneath it is selected.
    """

    name: str
    kind: str
    pattern: str
    required: bool
    why: str

    def __post_init__(self) -> None:
        if self.kind not in ("exact", "glob", "tree"):
            raise ValueError("unknown evidence rule kind {0!r}".format(self.kind))


#: ``REBUILD_PLAN.md`` section 1.3, minus the GOLD rule, which is a choice.
_COMMON_RULES: Tuple[EvidenceRule, ...] = (
    EvidenceRule(
        "final_results",
        "exact",
        "final_results.csv",
        True,
        "the consensus payload: Tool1/Tool2/Score1/Score2/RMSD",
    ),
    EvidenceRule("smina_results", "exact", "smina/results.csv", False, "per-tool scores"),
    EvidenceRule("gnina_results", "exact", "gnina/results.csv", False, "per-tool scores"),
    EvidenceRule("gexisting_results", "exact", "gold/results.csv", False, "per-tool scores"),
    EvidenceRule("ledock_results", "exact", "ledock/results.csv", False, "per-tool scores"),
    EvidenceRule("smina_poses", "exact", "smina/out.sdf", False, "native poses, unrounded scores"),
    EvidenceRule(
        "gnina_poses",
        "exact",
        "gnina/out.sdf",
        False,
        "sole carrier of CNN_VS, CNNaffinity_variance and unrounded scores",
    ),
    EvidenceRule("ledock_poses", "exact", "ledock/out.dok", False, "native poses"),
    EvidenceRule(
        "input_mol2",
        "glob",
        "input/*.mol2",
        False,
        "ligand + receptor; makes the job self-describing",
    ),
)

#: What to take out of ``gold/output/``.  This is a real choice, measured on
#: six real job directories:
#:
#: ``poses`` (default)
#:     ``gold_soln_*.sdf`` + the rankings = **363 KB**.  Total evidence
#:     ~1.21 MB/job, i.e. the ~1.14 MB / 2.4% that ``REBUILD_PLAN.md`` section
#:     1.3 budgets, and exactly what its table row says ("gold/output/
#:     (gold_soln poses) ~180 KB").
#: ``all``
#:     the whole tree = **3.09 MB**, because it also holds two more copies of
#:     the receptor (``gold_protein.mol2`` 741 KB + ``plp_protein.mol2`` 730 KB,
#:     already carried as ``input/receptor.mol2``), the cavity fitting points
#:     (``fit_pts*.mol2`` 402 KB), 424 KB of GOLD logs and a duplicate pose set
#:     (``ranked_*.sdf``, byte-for-byte the ``gold_soln`` poses re-sorted).
#:     Total evidence ~4.52 MB/job = 9.1%: **62 GB** for 13,730 reused jobs
#:     rather than 19 GB.
GOLD_OUTPUT_MODES = ("poses", "all")

_GOLD_POSE_RULES: Tuple[EvidenceRule, ...] = (
    EvidenceRule(
        "gold_poses", "glob", "gold/output/gold_soln_*.sdf", False, "GOLD solution poses"
    ),
    EvidenceRule(
        "gold_ranking",
        "glob",
        "gold/output/bestranking.lst",
        False,
        "GOLD fitness ranking",
    ),
    EvidenceRule("gold_rnk", "glob", "gold/output/*.rnk", False, "per-solution ranks"),
)

_GOLD_TREE_RULES: Tuple[EvidenceRule, ...] = (
    EvidenceRule(
        "gold_output",
        "tree",
        "gold/output",
        False,
        "the whole GOLD output tree (2.7x the rest of the evidence combined)",
    ),
)


def evidence_rules(gold_output: str = "poses") -> Tuple[EvidenceRule, ...]:
    """The evidence selectors for a given ``gold/output`` policy."""
    if gold_output not in GOLD_OUTPUT_MODES:
        raise ValueError(
            "gold_output must be one of {0}, got {1!r}".format(
                GOLD_OUTPUT_MODES, gold_output
            )
        )
    gold = _GOLD_POSE_RULES if gold_output == "poses" else _GOLD_TREE_RULES
    return _COMMON_RULES + gold


#: The default rule set (``gold_output='poses'``).
EVIDENCE_RULES: Tuple[EvidenceRule, ...] = evidence_rules()

#: Never copied under ``--evidence-only``, and never copied *at all* by the
#: evidence path even if a rule's tree happens to contain one.  ``complex_*.pdb``
#: is receptor+pose (derived, 99.75% byte-identical between poses); ``log_*.log``
#: is retry noise whose only useful content is now in ``job_identity.json``.
EXCLUDED_BASENAME_PATTERNS: Tuple[str, ...] = ("complex_*.pdb", "log_*.log")


def is_excluded_basename(path: str) -> bool:
    """True iff *path*'s basename matches an excluded pattern."""
    base = os.path.basename(path)
    return any(fnmatch.fnmatch(base, pat) for pat in EXCLUDED_BASENAME_PATTERNS)


def _stat_size(path: str) -> int:
    return int(os.stat(path).st_size)


def _rel(job_dir: str, path: str) -> str:
    return os.path.relpath(path, job_dir).replace(os.sep, "/")


def select_evidence_files(
    job_dir: str, gold_output: str = "poses"
) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Evidence files of *job_dir* as ``[(relpath, size)]``, plus missing rules.

    Deterministic: the result is sorted by relative path, never left in
    ``os.scandir`` order.  ``complex_*.pdb`` and ``log_*.log`` are filtered out
    unconditionally, after rule expansion, so no rule can ever leak one.
    """
    selected: Dict[str, int] = {}
    missing: List[str] = []
    for rule in evidence_rules(gold_output):
        hits: List[str] = []
        if rule.kind == "exact":
            candidate = os.path.join(job_dir, rule.pattern)
            if os.path.isfile(candidate):
                hits.append(candidate)
        elif rule.kind == "glob":
            hits.extend(
                p for p in glob.glob(os.path.join(job_dir, rule.pattern)) if os.path.isfile(p)
            )
        else:  # tree
            root = os.path.join(job_dir, rule.pattern)
            if os.path.isdir(root):
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames.sort()
                    for filename in sorted(filenames):
                        full = os.path.join(dirpath, filename)
                        if os.path.isfile(full):
                            hits.append(full)
        kept = [p for p in hits if not is_excluded_basename(p)]
        if not kept:
            missing.append(rule.name)
        for path in kept:
            selected[_rel(job_dir, path)] = _stat_size(path)
    return sorted(selected.items()), missing


def select_all_files(job_dir: str) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Every file under *job_dir* (the ``--evidence-only``-off path)."""
    selected: Dict[str, int] = {}
    for dirpath, dirnames, filenames in os.walk(job_dir):
        dirnames.sort()
        for filename in sorted(filenames):
            full = os.path.join(dirpath, filename)
            if os.path.isfile(full):
                selected[_rel(job_dir, full)] = _stat_size(full)
    missing = [] if "final_results.csv" in selected else ["final_results"]
    return sorted(selected.items()), missing


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JobPlan:
    """One old directory that will become one new directory."""

    old_job_dir: str
    new_job_dir: str
    identity: JobIdentity
    sample_type: str
    selected_by: str
    fragment_source: str  # 'manifest' | 'assumed'


@dataclass(frozen=True)
class Reject:
    """A candidate that will NOT be carried over, and why.

    Every candidate ends up in the plan or here.  Nothing is dropped silently --
    that is how 2,829 edges went missing from the old campaign.
    """

    old_job_dir: str
    stage: str
    severity: str
    reason: str


@dataclass
class CurationPlan:
    jobs: List[JobPlan] = field(default_factory=list)
    rejects: List[Reject] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    invariants: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def n_errors(self) -> int:
        return sum(1 for r in self.rejects if r.severity == SEVERITY_ERROR)

    def counts_by_reason(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for reject in self.rejects:
            key = "{0}/{1}: {2}".format(reject.severity, reject.stage, reject.reason)
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))


def _key_tuple(row: Dict[str, Any], fragment_id: Optional[str] = None) -> Tuple[str, str, str, str]:
    """The pocket key ``(drug, accession, fragment, cavity)`` as strings.

    ``fragment_id`` overrides the row's own value, so a legacy row whose
    fragment had to be supplied explicitly is matched on the fragment it
    actually has, not on a null.
    """
    return (
        str(row["drugbank_id"]),
        str(row["accession"]),
        str(row["fragment_id"] if fragment_id is None else fragment_id),
        str(int(row["cavity_index"])),
    )


#: The message a null fragment gets.  It names the escape hatch, because the
#: inference (missing token == F1, since the receptor was AF-{acc}-F1-model_v1)
#: is legitimate -- it just has to be a decision someone made, not a default.
_NULL_FRAGMENT_REASON = (
    "fragment_id is null. Refusing to infer it -- an inferred fragment is what "
    "made one pocket key match several pockets (410,087-row fan-out). Pass "
    "--assume-fragment F1 to make the inference explicit and logged."
)


def _resolve_fragment(
    row: Dict[str, Any], assume_fragment: Optional[str]
) -> Tuple[Optional[str], str, Optional[str]]:
    """``(fragment_id, source, error)``.  Never guesses."""
    value = row.get("fragment_id")
    if not _is_null(value):
        return str(value), "manifest", None
    if assume_fragment is None:
        return None, "", _NULL_FRAGMENT_REASON
    return (
        assume_fragment,
        "assumed:{0} (--assume-fragment)".format(assume_fragment),
        None,
    )


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(value != value)  # NaN
    except (TypeError, ValueError):
        return True


def build_plan(
    reuse: "Any",
    positives: "Any",
    drawn: "Any",
    index: IdentityIndex,
    src: str,
    assume_fragment: Optional[str] = None,
    on_name_collision: str = "fail",
) -> CurationPlan:
    """Decide, deterministically, which old job becomes which new job.

    Selection is a *set* operation on pocket keys, never a dict lookup that
    could last-wins-collapse: both directions of ``old_job_dir <-> new_job_dir``
    are asserted to have collapse factor exactly 1 before the plan is returned.
    """
    plan = CurationPlan()

    reuse_rows = reuse.to_dict("records")
    by_old_dir: Dict[str, Dict[str, Any]] = {}
    duplicate_old_dirs: List[str] = []
    for row in reuse_rows:
        name = str(row["old_job_dir"])
        if name in by_old_dir:
            duplicate_old_dirs.append(name)
        else:
            by_old_dir[name] = row
    if duplicate_old_dirs:
        raise CurationError(
            "reuse map lists {0} old_job_dir value(s) more than once ({1}); it "
            "is keyed on old_job_dir and a dict would silently keep the last "
            "row.".format(len(duplicate_old_dirs), sorted(set(duplicate_old_dirs))[:10])
        )

    positive_keys = set(_key_tuple(r) for r in positives.to_dict("records"))
    positive_selected_by = {
        _key_tuple(r): str(r["selected_by"]) for r in positives.to_dict("records")
    }

    # ---- candidates -----------------------------------------------------
    # (old_job_dir, sample_type, selected_by, row, fragment_id, fragment_source)
    candidates: List[Tuple[str, str, str, Dict[str, Any], str, str]] = []

    for row in reuse_rows:
        if str(row["classification"]) != TRUE_POSITIVE:
            continue
        name = str(row["old_job_dir"])
        if any(_is_null(row[c]) for c in IDENTITY_COLUMNS):
            plan.rejects.append(
                Reject(
                    name,
                    "positive",
                    SEVERITY_ERROR,
                    "classified true_positive but its identity is incomplete "
                    "(drugbank_id/accession/cavity_index)",
                )
            )
            continue
        fragment_id, fragment_source, error = _resolve_fragment(row, assume_fragment)
        if error is not None:
            plan.rejects.append(Reject(name, "positive", SEVERITY_ERROR, error))
            continue
        key = _key_tuple(row, fragment_id)
        if key not in positive_keys:
            plan.rejects.append(
                Reject(
                    name,
                    "positive",
                    SEVERITY_BY_DESIGN,
                    "true_positive whose pocket key is not in positives_planned",
                )
            )
            continue
        candidates.append(
            (name, "positive", positive_selected_by[key], row, fragment_id, fragment_source)
        )

    n_new_negatives = 0
    for row in drawn.to_dict("records"):
        source = str(row["source"])
        if source != "pool":
            n_new_negatives += 1
            continue
        name = "" if _is_null(row["old_job_dir"]) else str(row["old_job_dir"])
        if not name:
            plan.rejects.append(
                Reject(
                    "<none>",
                    "negative",
                    SEVERITY_ERROR,
                    "negatives_drawn row has source='pool' but no old_job_dir",
                )
            )
            continue
        reuse_row = by_old_dir.get(name)
        if reuse_row is None:
            plan.rejects.append(
                Reject(
                    name,
                    "negative",
                    SEVERITY_ERROR,
                    "drawn from the pool but absent from the reuse map",
                )
            )
            continue
        if str(reuse_row["classification"]) != VALID_NON_INTERACTION:
            plan.rejects.append(
                Reject(
                    name,
                    "negative",
                    SEVERITY_ERROR,
                    "drawn as a negative but the reuse map classifies it "
                    "{0!r}".format(str(reuse_row["classification"])),
                )
            )
            continue
        if any(_is_null(reuse_row[c]) for c in IDENTITY_COLUMNS):
            plan.rejects.append(
                Reject(
                    name,
                    "negative",
                    SEVERITY_ERROR,
                    "drawn from the pool but its reuse row's identity is "
                    "incomplete (drugbank_id/accession/cavity_index)",
                )
            )
            continue
        fragment_id, fragment_source, error = _resolve_fragment(
            reuse_row, assume_fragment
        )
        if error is not None:
            plan.rejects.append(Reject(name, "negative", SEVERITY_ERROR, error))
            continue
        if _key_tuple(row) != _key_tuple(reuse_row, fragment_id):
            plan.rejects.append(
                Reject(
                    name,
                    "negative",
                    SEVERITY_ERROR,
                    "drawn pocket key {0} != reuse map pocket key {1}".format(
                        _key_tuple(row), _key_tuple(reuse_row, fragment_id)
                    ),
                )
            )
            continue
        candidates.append(
            (name, "negative", str(row["selected_by"]), row, fragment_id, fragment_source)
        )

    n_candidates = len(candidates) + plan.n_errors + sum(
        1 for r in plan.rejects if r.severity == SEVERITY_BY_DESIGN
    )

    # ---- turn each surviving candidate into a rename --------------------
    seen_old: Dict[str, str] = {}
    built: List[JobPlan] = []
    for name, sample_type, selected_by, row, fragment_id, fragment_source in candidates:
        if name in seen_old:
            plan.rejects.append(
                Reject(
                    name,
                    "rename",
                    SEVERITY_ERROR,
                    "selected twice: once as {0}, once as {1}. A job cannot be "
                    "both classes.".format(seen_old[name], sample_type),
                )
            )
            continue
        source_dir = os.path.join(src, name)
        if not os.path.isdir(source_dir):
            plan.rejects.append(
                Reject(name, "rename", SEVERITY_ERROR, "source job directory not found")
            )
            continue

        accession = str(row["accession"])
        try:
            gene = index.label(accession)
        except KeyError as exc:
            plan.rejects.append(
                Reject(
                    name,
                    "rename",
                    SEVERITY_ERROR,
                    "accession {0} is not in the identity index: {1}".format(
                        accession, exc
                    ),
                )
            )
            continue

        try:
            identity = JobIdentity(
                drugbank_id=str(row["drugbank_id"]),
                accession=accession,
                gene=gene,
                fragment_id=fragment_id,
                cavity_index=row["cavity_index"],
            )
            new_name = format_job_dir(
                identity.drugbank_id,
                identity.accession,
                identity.gene,
                identity.fragment_id,
                identity.cavity_index,
            )
        except JobIdError as exc:
            plan.rejects.append(
                Reject(name, "rename", SEVERITY_ERROR, "cannot build a job name: {0}".format(exc))
            )
            continue

        seen_old[name] = sample_type
        built.append(
            JobPlan(
                old_job_dir=name,
                new_job_dir=new_name,
                identity=identity,
                sample_type=sample_type,
                selected_by=selected_by,
                fragment_source=fragment_source,
            )
        )

    # ---- collisions: two old directories claiming one new name ----------
    by_new: Dict[str, List[JobPlan]] = {}
    for job in built:
        by_new.setdefault(job.new_job_dir, []).append(job)
    colliding = {k: v for k, v in by_new.items() if len(v) > 1}
    if colliding:
        detail = "; ".join(
            "{0} <- {1}".format(new, sorted(j.old_job_dir for j in jobs))
            for new, jobs in sorted(colliding.items())[:10]
        )
        if on_name_collision == "fail":
            raise CurationError(
                "{0} new job name(s) are claimed by more than one old directory: "
                "{1}. Picking one would be the last-wins collapse this rebuild "
                "exists to eliminate -- decide which measurement to keep, or pass "
                "--on-name-collision skip to exclude all of them with a recorded "
                "reason.".format(len(colliding), detail)
            )
        logger.warning(
            "excluding %d colliding new name(s) (all sides): %s", len(colliding), detail
        )
        for new, jobs in sorted(colliding.items()):
            for job in jobs:
                plan.rejects.append(
                    Reject(
                        job.old_job_dir,
                        "rename",
                        SEVERITY_ERROR,
                        "new name {0} is claimed by {1} old directories ({2}); "
                        "excluded rather than arbitrarily chosen".format(
                            new, len(jobs), sorted(j.old_job_dir for j in jobs)
                        ),
                    )
                )
        built = [j for j in built if j.new_job_dir not in colliding]

    # G3: total order, independent of manifest row order and hash seed.
    plan.jobs = sorted(built, key=lambda j: (j.new_job_dir, j.old_job_dir))

    # ---- invariants -----------------------------------------------------
    pairs = [(j.old_job_dir, j.new_job_dir) for j in plan.jobs]
    plan.invariants.append(
        assert_collapse_factor_one(pairs, name="curate_jobs old_job_dir -> new_job_dir")
    )
    plan.invariants.append(
        assert_collapse_factor_one(
            [(new, old) for old, new in pairs],
            name="curate_jobs new_job_dir -> old_job_dir",
        )
    )
    plan.invariants.append(
        assert_total_accounting(
            n_in=n_candidates,
            n_resolved=len(plan.jobs),
            n_rejected=len(plan.rejects),
            name="curate_jobs candidate accounting",
        )
    )

    n_positive = sum(1 for j in plan.jobs if j.sample_type == "positive")
    n_assumed = sum(1 for j in plan.jobs if j.fragment_source != "manifest")
    plan.stats = {
        "n_reuse_rows": int(len(reuse_rows)),
        "n_positives_planned": int(len(positive_keys)),
        "n_drawn_rows": int(len(drawn)),
        "n_drawn_from_pool": int(len(drawn)) - n_new_negatives,
        "n_drawn_new": n_new_negatives,
        "n_candidates": n_candidates,
        "n_planned": len(plan.jobs),
        "n_planned_positive": n_positive,
        "n_planned_negative": len(plan.jobs) - n_positive,
        "n_rejected": len(plan.rejects),
        "n_rejected_error": plan.n_errors,
        "n_fragment_assumed": n_assumed,
        "n_positives_planned_not_reuseable": int(
            len(positive_keys)
            - sum(1 for j in plan.jobs if j.sample_type == "positive")
        ),
    }
    if n_assumed:
        logger.warning(
            "%d job(s) had no fragment_id in the manifest and were assumed %s. "
            "This is recorded per job in job_identity.json (fragment_source).",
            n_assumed,
            assume_fragment,
        )
    return plan


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


@dataclass
class CopyOutcome:
    new_job_dir: str
    old_job_dir: str
    status: str  # 'copied' | 'skipped_complete' | 'failed'
    n_files: int = 0
    n_bytes: int = 0
    reason: str = ""


def _assert_disjoint(src: str, dst: str) -> None:
    src_abs = os.path.abspath(src)
    dst_abs = os.path.abspath(dst)
    if src_abs == dst_abs:
        raise CurationError("--src and --dst are the same directory: {0}".format(src_abs))
    if dst_abs.startswith(src_abs + os.sep):
        raise CurationError(
            "--dst ({0}) is inside --src ({1}). The source campaign is read-only "
            "and must stay an independent copy.".format(dst_abs, src_abs)
        )
    if src_abs.startswith(dst_abs + os.sep):
        raise CurationError(
            "--src ({0}) is inside --dst ({1}); refusing to write into an "
            "ancestor of the read-only source.".format(src_abs, dst_abs)
        )


def _copy_one_file(src_file: str, dst_file: str, expected_size: int) -> int:
    """Copy one file and verify its size.  Never touches *src_file*."""
    parent = os.path.dirname(dst_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    shutil.copyfile(src_file, dst_file)
    try:  # mtimes are provenance; failing to preserve them is not fatal
        shutil.copystat(src_file, dst_file)
    except OSError:  # pragma: no cover - filesystem dependent
        pass
    written = _stat_size(dst_file)
    if written != expected_size:
        try:
            os.unlink(dst_file)  # destination only; src is never unlinked
        except OSError:  # pragma: no cover
            pass
        raise CurationError(
            "size mismatch after copying {0}: source {1} bytes, destination {2} "
            "bytes. The destination file was removed; re-run to retry.".format(
                src_file, expected_size, written
            )
        )
    return written


def _identity_record(job: JobPlan, provenance: Dict[str, Any]) -> Dict[str, Any]:
    record = dict(job.identity.to_dict())
    record.update(provenance)
    return record


def copy_job(
    job: JobPlan,
    src: str,
    dst: str,
    evidence_only: bool,
    provenance: Dict[str, Any],
    force: bool = False,
    gold_output: str = "poses",
) -> CopyOutcome:
    """Materialise one job.  Idempotent; ``job_identity.json`` is written last."""
    source_dir = os.path.join(src, job.old_job_dir)
    target_dir = os.path.join(dst, job.new_job_dir)

    expected = _identity_record(job, provenance)
    if not force:
        complete, _why = _destination_is_complete(target_dir, expected)
        if complete:
            return CopyOutcome(job.new_job_dir, job.old_job_dir, "skipped_complete")

    if not os.path.isdir(source_dir):
        return CopyOutcome(
            job.new_job_dir, job.old_job_dir, "failed", reason="source directory not found"
        )

    if evidence_only:
        files, missing_rules = select_evidence_files(source_dir, gold_output=gold_output)
    else:
        files, missing_rules = select_all_files(source_dir)
    if "final_results" in missing_rules:
        return CopyOutcome(
            job.new_job_dir,
            job.old_job_dir,
            "failed",
            reason="no final_results.csv; incomplete, not reuseable",
        )
    if evidence_only:
        # Defended twice on purpose: select_evidence_files already filters, and
        # this refuses to write even if a future rule stops doing so.
        leaked = [p for p, _ in files if is_excluded_basename(p)]
        if leaked:  # pragma: no cover
            return CopyOutcome(
                job.new_job_dir,
                job.old_job_dir,
                "failed",
                reason="excluded artifact(s) reached the copy list: {0}".format(leaked[:5]),
            )

    os.makedirs(target_dir, exist_ok=True)
    n_bytes = 0
    for relpath, size in files:
        source_file = os.path.join(source_dir, relpath)
        target_file = os.path.join(target_dir, relpath)
        if not force and os.path.isfile(target_file) and _stat_size(target_file) == size:
            n_bytes += size
            continue
        n_bytes += _copy_one_file(source_file, target_file, size)

    record = dict(expected)
    record["evidence"] = [{"path": p, "bytes": b} for p, b in files]
    record["missing_evidence_rules"] = sorted(missing_rules)
    core = {"drugbank_id", "accession", "gene", "fragment_id", "cavity_index"}
    write_job_identity(
        target_dir,
        job.identity,
        **{k: v for k, v in record.items() if k not in core},
    )
    return CopyOutcome(
        job.new_job_dir, job.old_job_dir, "copied", n_files=len(files), n_bytes=n_bytes
    )


def _destination_is_complete(
    target_dir: str, expected: Dict[str, Any]
) -> Tuple[bool, str]:
    """Is this destination already a finished copy of exactly this job?

    Decided entirely from the destination, so a resume does not re-read the
    slow source drive.  ``job_identity.json`` is the completion marker precisely
    because it is written last and atomically.
    """
    identity_path = os.path.join(target_dir, JOB_IDENTITY_FILENAME)
    if not os.path.isfile(identity_path):
        return False, "no {0}".format(JOB_IDENTITY_FILENAME)
    try:
        record = read_job_identity_record(target_dir)
    except (JobIdError, OSError, ValueError) as exc:
        return False, "unreadable {0}: {1}".format(JOB_IDENTITY_FILENAME, exc)
    for key, value in sorted(expected.items()):
        if record.get(key) != value:
            return False, "{0} disagrees on {1!r}".format(JOB_IDENTITY_FILENAME, key)
    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        return False, "{0} carries no evidence manifest".format(JOB_IDENTITY_FILENAME)
    for entry in evidence:
        path = os.path.join(target_dir, str(entry.get("path", "")))
        if not os.path.isfile(path):
            return False, "missing evidence file {0!r}".format(entry.get("path"))
        if _stat_size(path) != int(entry.get("bytes", -1)):
            return False, "size mismatch on {0!r}".format(entry.get("path"))
    return True, "complete"


def job_provenance(job: JobPlan, base: Dict[str, Any]) -> Dict[str, Any]:
    """The provenance block written into one job's ``job_identity.json``.

    Four fields vary per job (``sample_type``, ``source_job``, ``selected_by``,
    ``fragment_source``); the rest -- retention, index SHA256, code revision --
    are constant for the run.  No timestamp: the curated tree must be
    byte-identical across two runs of the same plan.
    """
    record = dict(base)
    record["sample_type"] = job.sample_type
    record["provenance"] = "reused_from"
    record["source_job"] = job.old_job_dir
    record["selected_by"] = job.selected_by
    record["fragment_source"] = job.fragment_source
    return record


def execute_plan(
    plan: CurationPlan,
    src: str,
    dst: str,
    evidence_only: bool,
    base_provenance: Dict[str, Any],
    workers: int = 1,
    force: bool = False,
    gold_output: str = "poses",
) -> List[CopyOutcome]:
    """Copy every planned job.  Results come back in plan order, always."""
    _assert_disjoint(src, dst)
    if gold_output not in GOLD_OUTPUT_MODES:
        raise ValueError(
            "gold_output must be one of {0}, got {1!r}".format(
                GOLD_OUTPUT_MODES, gold_output
            )
        )
    os.makedirs(dst, exist_ok=True)
    # The retention policy describes what this run actually did, so it is set
    # here rather than trusted from the caller's provenance dict.  It is part
    # of the resume key: changing the policy correctly re-copies.
    base_provenance = dict(base_provenance)
    base_provenance["retention"] = (
        RETENTION_EVIDENCE if evidence_only else RETENTION_FULL
    )
    base_provenance["evidence_gold_output"] = gold_output if evidence_only else None
    outcomes: List[Optional[CopyOutcome]] = [None] * len(plan.jobs)

    def run(position: int) -> Tuple[int, CopyOutcome]:
        job = plan.jobs[position]
        try:
            return position, copy_job(
                job,
                src,
                dst,
                evidence_only,
                job_provenance(job, base_provenance),
                force=force,
                gold_output=gold_output,
            )
        except (CurationError, OSError, JobIdError) as exc:
            return position, CopyOutcome(
                job.new_job_dir, job.old_job_dir, "failed", reason=str(exc)
            )

    if workers <= 1:
        for position in range(len(plan.jobs)):
            _, outcome = run(position)
            outcomes[position] = outcome
            _log_progress(position + 1, len(plan.jobs))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run, position) for position in range(len(plan.jobs))]
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                position, outcome = future.result()
                outcomes[position] = outcome
                _log_progress(done, len(plan.jobs))
    return [o for o in outcomes if o is not None]


def _log_progress(done: int, total: int, every: int = 250) -> None:
    if total and (done % every == 0 or done == total):
        logger.info("… %d/%d jobs", done, total)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _reject_frame(rejects: Sequence[Reject]) -> "Any":
    import pandas as pd

    return pd.DataFrame(
        {
            "old_job_dir": [r.old_job_dir for r in rejects],
            "stage": [r.stage for r in rejects],
            "severity": [r.severity for r in rejects],
            "reason": [r.reason for r in rejects],
        },
        columns=["old_job_dir", "stage", "severity", "reason"],
    )


def _manifest_frame(
    plan: CurationPlan, outcomes: Sequence[CopyOutcome]
) -> "Any":
    """``run_manifest`` rows for the jobs that are actually on disk."""
    import pandas as pd

    materialised = {
        o.new_job_dir for o in outcomes if o.status in ("copied", "skipped_complete")
    }
    rows = [j for j in plan.jobs if j.new_job_dir in materialised]
    return pd.DataFrame(
        {
            "job_id": [j.new_job_dir for j in rows],
            "drugbank_id": [j.identity.drugbank_id for j in rows],
            "accession": [j.identity.accession for j in rows],
            "gene": [j.identity.gene for j in rows],
            "fragment_id": [j.identity.fragment_id for j in rows],
            "cavity_index": [j.identity.cavity_index for j in rows],
            "sample_type": [j.sample_type for j in rows],
            "provenance": ["reused_from"] * len(rows),
            "source_job": [j.old_job_dir for j in rows],
            "selected_by": [j.selected_by for j in rows],
        }
    )


def _print_plan(
    plan: CurationPlan,
    dst: str,
    evidence_only: bool,
    gold_output: str = "poses",
    limit: int = 20,
) -> None:
    print("")
    print("curate_jobs --dry-run — NOTHING WILL BE WRITTEN")
    print("=" * 78)
    for key, value in sorted(plan.stats.items()):
        print("  {0:<40} {1}".format(key, value))
    print("  {0:<40} {1}".format("retention", "evidence_only" if evidence_only else "full"))
    if evidence_only:
        print("  {0:<40} {1}".format("gold/output policy", gold_output))
    print("  {0:<40} {1}".format("destination", dst))
    print("")
    if plan.rejects:
        print("not carried over ({0} candidate(s)):".format(len(plan.rejects)))
        for reason, count in plan.counts_by_reason().items():
            print("  {0:>8}  {1}".format(count, reason))
        print("")
    print("first {0} of {1} renames:".format(min(limit, len(plan.jobs)), len(plan.jobs)))
    for job in plan.jobs[:limit]:
        print(
            "  {0:<9} {1}  ->  {2}".format(job.sample_type, job.old_job_dir, job.new_job_dir)
        )
    if len(plan.jobs) > limit:
        print("  … {0} more".format(len(plan.jobs) - limit))
    print("")


def _git_revision(path: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return "unknown"
    if completed.returncode != 0:  # pragma: no cover
        return "unknown"
    return completed.stdout.decode("utf-8", "replace").strip() or "unknown"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="curate_jobs.py",
        description=(
            "Materialise the curated docking directory (REBUILD_PLAN Phase 5). "
            "Copies -- never moves, never hardlinks, never deletes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with --dry-run first. It writes nothing.",
    )
    parser.add_argument("--job-classification", required=True, help="manifests/job_classification.csv")
    parser.add_argument("--drawn", required=True, help="manifests/negatives_drawn.csv")
    parser.add_argument("--positives", required=True, help="manifests/positives_planned.csv")
    parser.add_argument(
        "--index",
        required=True,
        help=(
            "identity/identity_index.parquet. Required: the gene in every new "
            "directory name is IdentityIndex.label(accession), so the label "
            "cannot disagree with the receptor."
        ),
    )
    parser.add_argument("--index-sha256", default=None, help="pin the identity index")
    parser.add_argument("--src", required=True, help="$OLD/consensus_docking_results (read-only)")
    parser.add_argument("--dst", required=True, help="$NEW/docking_results")
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="apply the 2.4%% rule: skip complex_*.pdb and log_*.log (~47 MB -> ~1.2 MB/job)",
    )
    parser.add_argument(
        "--gold-output",
        choices=GOLD_OUTPUT_MODES,
        default="poses",
        help=(
            "what to take from gold/output/ under --evidence-only. 'poses' "
            "(default) = gold_soln_*.sdf + rankings, 363 KB, which is the "
            "2.4%% budget. 'all' = the whole tree, 3.1 MB, because it also "
            "holds two more receptor copies, the cavity fitting points, 424 KB "
            "of GOLD logs and a duplicate pose set (62 GB vs 19 GB overall)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
    parser.add_argument("--workers", type=int, default=1, help="parallel copy workers (default 1)")
    parser.add_argument(
        "--assume-fragment",
        default=None,
        help=(
            "explicitly assume this fragment id (e.g. F1) for reuse rows whose "
            "fragment_id is null. Off by default: an inferred fragment is what "
            "made one pocket key match several pockets."
        ),
    )
    parser.add_argument(
        "--on-name-collision",
        choices=("fail", "skip"),
        default="fail",
        help="two old dirs claiming one new name: fail (default) or exclude both",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="exit 0 even if some candidates could not be resolved (they are still recorded)",
    )
    parser.add_argument("--force", action="store_true", help="re-copy even if complete")
    parser.add_argument("--report", default=None, help="write a JSON run report here")
    parser.add_argument("--rejects", default=None, help="write the full reject ledger CSV here")
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="write run_manifest rows for the curated jobs here (schema: run_manifest)",
    )
    parser.add_argument("--code-revision", default=None, help="override the recorded git revision")
    add_runtime_arguments(parser, STEP_NAME)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.dry_run and (args.report or args.rejects or args.manifest_out):
        _build_parser().error(
            "--dry-run writes nothing, so it cannot be combined with --report, "
            "--rejects or --manifest-out."
        )
    if args.workers < 1:
        _build_parser().error("--workers must be >= 1")

    # The run directory is the parent of --dst ($NEW/docking_results), so the log
    # sits beside the tree it describes instead of in whatever directory the
    # operator happened to be standing in.
    resolve_out_dir(args, STEP_NAME, os.path.dirname(os.path.abspath(args.dst)))
    if args.dry_run and not args.log_file:
        # "--dry-run writes nothing" has to include this program's own log, or
        # the one command an operator is told to run first stops being harmless.
        args.log_file = os.devnull

    with step(STEP_NAME, args) as ctx:
        return _curate(args, ctx)


def _curate(args: Any, ctx: Any) -> int:
    """The body of :func:`main`, inside the step's logging and notifications."""
    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        _assert_disjoint(args.src, args.dst)
        if not os.path.isdir(args.src):
            raise CurationError("--src is not a directory: {0}".format(args.src))
        index = IdentityIndex.load(args.index, expect_sha256=args.index_sha256)
        reuse = read_checked(args.job_classification, "job_classification")
        positives = read_checked(args.positives, "positives_planned")
        drawn = read_checked(args.drawn, "negatives_drawn")
        assume_fragment = (
            normalize_fragment_id(args.assume_fragment) if args.assume_fragment else None
        )
        plan = build_plan(
            reuse=reuse,
            positives=positives,
            drawn=drawn,
            index=index,
            src=args.src,
            assume_fragment=assume_fragment,
            on_name_collision=args.on_name_collision,
        )
    except (CurationError, InvariantViolation, FileNotFoundError, JobIdError) as exc:
        logger.error("%s", exc)
        ctx.notify("failure", "curate_jobs: could not plan: {0}".format(exc), priority=1)
        return 1

    logger.info(
        "plan: %d job(s) (%d positive, %d negative), %d candidate(s) not carried "
        "(%d error)",
        plan.stats["n_planned"],
        plan.stats["n_planned_positive"],
        plan.stats["n_planned_negative"],
        plan.stats["n_rejected"],
        plan.stats["n_rejected_error"],
    )

    if args.dry_run:
        _print_plan(plan, args.dst, args.evidence_only, args.gold_output)
        return 0 if (plan.n_errors == 0 or args.allow_errors) else 1

    base_provenance = {
        # retention / evidence_gold_output are stamped by execute_plan()
        "identity_index_sha256": index.source_sha256,
        "code_revision": args.code_revision or _git_revision(_REPO_ROOT),
        "curated_by": "curate_jobs.py",
    }

    try:
        outcomes = execute_plan(
            plan,
            src=args.src,
            dst=args.dst,
            evidence_only=args.evidence_only,
            base_provenance=base_provenance,
            workers=args.workers,
            force=args.force,
            gold_output=args.gold_output,
        )
    except CurationError as exc:
        logger.error("%s", exc)
        ctx.notify("failure", "curate_jobs: copy aborted: {0}".format(exc), priority=1)
        return 1

    counts: Dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    n_failed = counts.get("failed", 0)
    total_bytes = sum(o.n_bytes for o in outcomes)
    total_files = sum(o.n_files for o in outcomes)
    logger.info(
        "copied %d, skipped %d already-complete, failed %d; %d file(s), %.2f MB",
        counts.get("copied", 0),
        counts.get("skipped_complete", 0),
        n_failed,
        total_files,
        total_bytes / (1024.0 * 1024.0),
    )
    for outcome in outcomes:
        if outcome.status == "failed":
            logger.error("FAILED %s: %s", outcome.old_job_dir, outcome.reason)

    report = {
        "tool": "curate_jobs.py",
        "started_at": started_at,
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "src": os.path.abspath(args.src),
        "dst": os.path.abspath(args.dst),
        "retention": RETENTION_EVIDENCE if args.evidence_only else RETENTION_FULL,
        "evidence_gold_output": args.gold_output if args.evidence_only else None,
        "identity_index": {
            "path": os.path.abspath(args.index),
            "sha256": index.source_sha256,
            "n_records": len(index),
        },
        "code_revision": base_provenance["code_revision"],
        "plan": plan.stats,
        "outcomes": {
            "copied": counts.get("copied", 0),
            "skipped_complete": counts.get("skipped_complete", 0),
            "failed": n_failed,
            "files": total_files,
            "bytes": total_bytes,
        },
        "failures": [
            {"old_job_dir": o.old_job_dir, "new_job_dir": o.new_job_dir, "reason": o.reason}
            for o in outcomes
            if o.status == "failed"
        ],
        "rejects_by_reason": plan.counts_by_reason(),
        "invariants": plan.invariants,
        "ok": bool(n_failed == 0 and plan.n_errors == 0),
    }
    if args.report:
        write_invariant_report(report, args.report)
    if args.rejects:
        frame = _reject_frame(plan.rejects)
        parent = os.path.dirname(os.path.abspath(args.rejects))
        if parent:
            os.makedirs(parent, exist_ok=True)
        frame.to_csv(args.rejects, index=False, lineterminator="\n")
        logger.info("wrote %d reject row(s) to %s", len(frame), args.rejects)
    if args.manifest_out:
        write_checked(_manifest_frame(plan, outcomes), args.manifest_out, "run_manifest")

    if n_failed or (plan.n_errors and not args.allow_errors):
        ctx.notify("failure",
                   "curate_jobs: {0} job(s) failed to copy, {1} unresolved "
                   "candidate(s)".format(n_failed, plan.n_errors),
                   priority=1)
        return 1
    ctx.notify("success",
               "curate_jobs: {0} copied, {1} already complete, {2:.1f} GB".format(
                   counts.get("copied", 0), counts.get("skipped_complete", 0),
                   total_bytes / (1024.0 ** 3)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
