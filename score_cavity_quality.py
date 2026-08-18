#!/usr/bin/env python3
"""Phase 2b -- score every CavitySpace pocket's AlphaFold confidence (pLDDT).

Sits between the cavity manifest (phase 2) and positive planning (phase 3), and
writes ``manifests/cavity_quality.csv`` (schema ``cavity_quality``): one row per
``(accession, fragment_id, cavity_index)`` carrying that pocket's per-residue
pLDDT distribution.

Why this stage exists
---------------------

A CavitySpace cavity archive holds the pocket-**lining residues** of an AlphaFold
model, and those residues keep the model's pLDDT in the B-factor column::

    ATOM     61  CG  TYR A   8       3.860   6.337  -9.914  1.00 98.75           C
                                                                 ^^^^^ pLDDT

Mean pocket pLDDT ranges from 34 to 99 across the set, so a docking campaign that
does not look at it is docking into confidently-modelled pockets and into
disordered guesses with equal weight, and cannot say afterwards which was which.

What it computes, exactly
-------------------------

Only ``ATOM`` records are read -- never ``HETATM``, because the ``surface_*.pdb``
and ``vacant_*.pdb`` members of the same archive are HETATM grid points with no
pLDDT at all. A residue is keyed ``(chain_id, resSeq+iCode)`` = ``(line[21],
line[22:27].strip())`` and its pLDDT is the **mean** of its atoms' B-factors.
AlphaFold writes one value per residue on every one of its atoms, so the mean
equals that value; taking the mean is simply robust to an odd file.

``residue_plddt`` stores all of those values, 2dp, ``;``-separated. That column is
the point of the artifact: it lets any later cutoff be evaluated
(:func:`dock_selectivity.quality.fraction_at`) without re-reading 7.0 GB of
archives, so a cutoff sweep costs a string split. The ``mean/median/min/max`` and
``frac_ge_50/70/90`` columns are all derived from those same rounded values, so
every column in the file is reproducible from ``residue_plddt`` alone.

Nothing is skipped silently
---------------------------

A cavity the manifest names but that cannot be scored -- archive absent, archive
unreadable, no ``*_cavity_<n>.pdb`` member, a member with zero ``ATOM`` records --
is written to the reject ledger with a reason, and **not** written to the quality
table as a zero-filled row. A zero row would read downstream as "a terrible
pocket" rather than "no data" and would fail every cutoff without saying why.
``n_targets == n_scored + n_rejected`` is asserted before anything is written.

Threads
-------

``--threads`` defaults to **1**, and that is a measurement, not an oversight.
Reading one archive costs ~0.7 ms warm / ~1.1 ms cold because this module stops
tar iteration at the cavity member instead of calling ``getmembers()`` (which
decompresses the whole archive, including the 0.4-1.3 MB ``vacant_*.pdb`` nobody
wants). At that size the pure-Python tar-header parsing dominates, so a thread
pool only adds GIL hand-off: measured over 400-500 archives, 1 thread 0.7 ms,
2 threads 1.6 ms, 4 threads 1.8 ms, 8 threads 2.2 ms, 16 threads 2.3 ms per
archive. All 38,709 cavities take about 40 seconds single-threaded. The flag is
kept because a different filesystem (a network mount, where each open blocks) can
reverse that; raise it only after measuring on the machine in question.

Inputs:
    - a folder of CavitySpace ``*.tar.gz`` cavity archives (--tarballs)
    - manifests/cavity_manifest.csv (--manifest), from extract_cavities.py --manifest-only
    - optionally a required set (--required): redock_queue.csv, positives_planned.csv,
      or any frame with an accession column and a cavity index

Outputs:
    - --out:     manifests/cavity_quality.csv     (schema ``cavity_quality``)
    - --rejects: the unscoreable-cavity ledger    (default <out>.rejects.csv)
    - <run-dir>/steps/score_cavity_quality/score_cavity_quality.log

Usage:
    # every cavity in the manifest (~40 s, 38,709 archives)
    python3 score_cavity_quality.py \\
        --tarballs /opt/data/cavity_space/af_strong_cavities_data_all \\
        --manifest $RUN/manifests/cavity_manifest.csv \\
        --out      $RUN/manifests/cavity_quality.csv \\
        --run-dir  $RUN

    # only the cavities a plan actually names (seconds)
    python3 score_cavity_quality.py \\
        --manifest $RUN/manifests/cavity_manifest.csv \\
        --required $RUN/manifests/redock_queue.csv \\
        --out      $RUN/manifests/cavity_quality.csv \\
        --run-dir  $RUN --resume
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
from tqdm import tqdm

from dock_selectivity import jobid, schemas
from dock_selectivity.invariants import InvariantViolation, assert_total_accounting
from dock_selectivity.runtime import (
    add_runtime_arguments,
    resolve_out_dir,
    step,
    write_step_summary,
)

#: Names the logger, the step directory and the log file. Must stay equal to the
#: logger name below, or ``step()`` would configure a logger this module never uses.
STEP_NAME = "score_cavity_quality"

LOG = logging.getLogger(STEP_NAME)

#: Machine-wide reference dataset, not a per-run location. Overridable with
#: ``--tarballs`` like every other ``/opt/data`` default in the repo.
DEFAULT_TARBALL_DIR = "/opt/data/cavity_space/af_strong_cavities_data_all"

#: Measured, not guessed -- see the module docstring. Threads make this slower.
DEFAULT_THREADS = 1

#: The pLDDT bands the artifact precomputes. Any other cutoff is recomputed from
#: ``residue_plddt`` by :func:`dock_selectivity.quality.fraction_at`; these three
#: exist because 70 is AlphaFold's "confident" line and the other two bracket it.
PRECOMPUTED_CUTOFFS: Tuple[int, int, int] = (50, 70, 90)

#: Directories the fixed run layout puts artifacts in; their parent is the run
#: directory. Consulted only to place this step's own log when ``--run-dir`` is
#: absent -- the log must land in the campaign, never in the invocation directory.
_LAYOUT_DIRS = ("manifests", "identity", "analysis", "structures", "docking_results")

#: Every reason a named cavity can fail to be scored. Exhaustive by design:
#: :func:`score_cavities` asserts the ledger it writes uses only these.
REJECT_REASONS: Tuple[str, ...] = (
    "NOT_IN_MANIFEST",
    "MISSING_ARCHIVE",
    "UNREADABLE_ARCHIVE",
    "NO_CAVITY_MEMBER",
    "NO_ATOM_RECORDS",
    "BAD_B_FACTOR",
)

#: Columns of the reject ledger, in order.
REJECT_COLUMNS: Tuple[str, ...] = (
    "accession",
    "fragment_id",
    "cavity_index",
    "tarball",
    "reason",
    "detail",
)


class CavityQualityError(InvariantViolation):
    """Raised when pocket quality cannot be produced without losing a cavity."""


class PocketParseError(Exception):
    """A pocket PDB could not be turned into per-residue pLDDT values."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__("{0}: {1}".format(reason, detail))
        self.reason = reason
        self.detail = detail


def infer_run_dir(artifact_path: str) -> str:
    """Best guess at the run directory, from an artifact this step writes.

    ``--run-dir`` is optional, but the step log may not fall back to the
    invocation directory -- that is how the repository working tree filled up
    with stray ``*.log`` files.
    """
    parent = os.path.dirname(os.path.abspath(artifact_path))
    if os.path.basename(parent) in _LAYOUT_DIRS:
        return os.path.dirname(parent)
    return parent


# ---------------------------------------------------------------------------
# Parsing one pocket
# ---------------------------------------------------------------------------


def residue_plddt_from_pdb(text: str) -> Tuple[List[float], int]:
    """``(per-residue pLDDT in file order, n_atoms)`` for one pocket PDB.

    Only ``ATOM`` records count. A residue is ``(line[21], line[22:27].strip())``
    -- chain plus residue number plus insertion code -- and its value is the mean
    of its atoms' B-factors (columns 61-66).

    Raises :class:`PocketParseError` when the member holds no ``ATOM`` record at
    all, or when one carries a B-factor that is not a number. Both are ledger
    entries for the caller, never a pocket scored 0.
    """
    order: List[Tuple[str, str, str]] = []
    sums: Dict[Tuple[str, str, str], float] = {}
    counts: Dict[Tuple[str, str, str], int] = {}
    n_atoms = 0

    for line in text.splitlines():
        if not line.startswith("ATOM"):
            continue
        n_atoms += 1
        field = line[60:66].strip()
        try:
            b_factor = float(field)
        except ValueError:
            raise PocketParseError(
                "BAD_B_FACTOR",
                "atom record {0} has B-factor field {1!r}, which is not a "
                "number".format(n_atoms, field),
            ) from None
        # (chain, resSeq, iCode) as three fields, not a concatenated string:
        # "8" with insertion code "1" and residue "81" both flatten to "81",
        # which would average two residues into one. No AlphaFold model has a
        # numeric insertion code, but this parser is one repoint away from
        # experimental structures that do.
        key = (line[21:22], line[22:26].strip(), line[26:27].strip())
        if key not in counts:
            order.append(key)
            sums[key] = 0.0
            counts[key] = 0
        sums[key] += b_factor
        counts[key] += 1

    if not n_atoms:
        raise PocketParseError(
            "NO_ATOM_RECORDS",
            "the pocket member holds no ATOM record (surface/vacant members are "
            "HETATM grid points and carry no pLDDT)",
        )
    return [sums[key] / counts[key] for key in order], n_atoms


def summarize_residues(values: Sequence[float], n_atoms: int) -> Dict[str, object]:
    """The metric columns for one pocket, all derived from the same rounded values.

    Each residue value is formatted to 2dp once and every statistic is computed
    from that formatted number, so ``residue_plddt`` is not merely a record of the
    row -- it *is* the row. :func:`dock_selectivity.quality.fraction_at` asked for
    50, 70 or 90 therefore reproduces ``frac_ge_50/70/90`` exactly.
    """
    if not values:
        raise CavityQualityError(
            "summarize_residues() was given no residues; an unscoreable cavity "
            "belongs in the reject ledger, never in the quality table."
        )
    texts = ["{0:.2f}".format(float(v)) for v in values]
    rounded = [float(t) for t in texts]
    n = len(rounded)
    row: Dict[str, object] = {
        "n_residues": n,
        "n_atoms": int(n_atoms),
        "mean_plddt": sum(rounded) / n,
        "median_plddt": float(statistics.median(rounded)),
        "min_plddt": min(rounded),
        "max_plddt": max(rounded),
        "residue_plddt": ";".join(texts),
    }
    for cutoff in PRECOMPUTED_CUTOFFS:
        row["frac_ge_{0}".format(cutoff)] = (
            sum(1 for v in rounded if v >= cutoff) / float(n)
        )
    return row


def read_cavity_member(tarball_path: str, cavity_index: int) -> str:
    """The text of ``*_cavity_<index>.pdb`` inside ``tarball_path``.

    Iterates the archive sequentially and stops at the member, rather than
    calling ``getmembers()``: the cavity PDB is the first entry, and the archive
    also holds a ``surface_*.pdb`` and a ``vacant_*.pdb`` of 0.4-1.3 MB that
    ``getmembers()`` would decompress for nothing. That single change is what
    makes a full pass ~40 s instead of ~4 minutes.

    Raises :class:`PocketParseError` if the archive is unreadable or holds no such
    member -- ``_cavity_1.pdb`` never matches ``_cavity_11.pdb``, the character
    before the index is ``_``.
    """
    suffix = "_cavity_{0}.pdb".format(int(cavity_index))
    try:
        with tarfile.open(tarball_path, "r:gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(suffix):
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                return handle.read().decode("utf-8", "replace")
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise PocketParseError(
            "UNREADABLE_ARCHIVE",
            "{0}: {1}".format(type(exc).__name__, exc),
        ) from None
    raise PocketParseError(
        "NO_CAVITY_MEMBER",
        "archive holds no member ending in {0!r}".format(suffix),
    )


def score_one(tarball_dir: str, target: Dict[str, object]) -> Dict[str, object]:
    """Score one manifest row. Returns either a quality row or a reject row.

    Never raises for a data problem: a cavity that cannot be scored comes back as
    ``{'reason': ..., 'detail': ...}`` so the caller can write it to the ledger,
    which is what keeps ``n_targets == n_scored + n_rejected`` true.
    """
    tarball = str(target["tarball"])
    identity = {
        "accession": target["accession"],
        "fragment_id": target["fragment_id"],
        "cavity_index": int(target["cavity_index"]),
    }
    path = os.path.join(tarball_dir, tarball)
    if not os.path.isfile(path):
        return dict(
            identity,
            tarball=tarball,
            reason="MISSING_ARCHIVE",
            detail="no such archive under --tarballs",
        )
    try:
        text = read_cavity_member(path, identity["cavity_index"])
        values, n_atoms = residue_plddt_from_pdb(text)
        metrics = summarize_residues(values, n_atoms)
    except PocketParseError as exc:
        return dict(identity, tarball=tarball, reason=exc.reason, detail=exc.detail)
    return dict(identity, **metrics)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _fragment(value: object) -> str:
    """Normalise a fragment id, defaulting to ``'F1'`` like the other stages."""
    try:
        return jobid.normalize_fragment_id(value) or "F1"
    except jobid.JobIdError:
        return "F1"


def tarball_name(accession: str, fragment_id: str, cavity_index: int) -> str:
    """The canonical CavitySpace archive name for one cavity.

    Used only when the manifest has no ``Tarball`` column (the legacy
    ``cavity_mapping.csv`` does not). Deriving the name is deterministic; if the
    derived archive is absent the cavity becomes a ``MISSING_ARCHIVE`` reject
    naming it, so a wrong guess is visible rather than silent.
    """
    return "AF-{0}-{1}-model_v1_cavity_result_{2}.tar.gz".format(
        accession, fragment_id, int(cavity_index)
    )


def load_manifest_targets(path: str) -> List[Dict[str, object]]:
    """Manifest rows as ``{accession, fragment_id, cavity_index, tarball}`` dicts.

    Accepts ``cavity_manifest.csv`` (``UniProt_ID``, ``Fragment_ID``,
    ``Cavity_Index``, ``Tarball``) and the legacy ``cavity_mapping.csv`` without
    the ``Tarball`` column. Rows are de-duplicated and sorted, so the output is
    byte-identical across runs and ``PYTHONHASHSEED`` values.
    """
    if not os.path.isfile(path):
        raise CavityQualityError(
            "--manifest {0!r} does not exist. A missing prerequisite is an error, "
            "never a fallback: produce it with "
            "'extract_cavities.py --manifest-only' first.".format(path)
        )
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    columns = {c.lower(): c for c in df.columns}
    accession_col = columns.get("uniprot_id") or columns.get("accession")
    cavity_col = columns.get("cavity_index")
    fragment_col = columns.get("fragment_id")
    if not accession_col or not cavity_col:
        raise CavityQualityError(
            "--manifest {0!r} needs an accession column (UniProt_ID or accession) "
            "and a Cavity_Index column; it has {1}".format(path, list(df.columns))
        )
    tarball_col = columns.get("tarball")

    seen: Set[Tuple[str, str, int]] = set()
    targets: List[Dict[str, object]] = []
    for row in df.to_dict("records"):
        accession = row[accession_col]
        cavity = row[cavity_col]
        if pd.isna(accession) or pd.isna(cavity):
            continue
        accession = str(accession).strip()
        fragment = _fragment(row[fragment_col]) if fragment_col else "F1"
        try:
            cavity_index = int(cavity)
        except (TypeError, ValueError):
            continue
        key = (accession, fragment, cavity_index)
        if key in seen:
            continue
        seen.add(key)
        tarball = None
        if tarball_col and not pd.isna(row[tarball_col]):
            tarball = str(row[tarball_col]).strip()
        targets.append(
            {
                "accession": accession,
                "fragment_id": fragment,
                "cavity_index": cavity_index,
                "tarball": tarball or tarball_name(accession, fragment, cavity_index),
            }
        )
    targets.sort(key=lambda t: (t["accession"], t["fragment_id"], t["cavity_index"]))
    return targets


def load_required_keys(path: str) -> Set[Tuple[str, str, int]]:
    """``{(accession, fragment_id, cavity_index)}`` from any manifest-shaped frame.

    ``redock_queue.csv``, ``positives_planned.csv``, ``negatives_drawn.csv`` and
    ``cavity_manifest.csv`` all qualify: the accession column may be spelled
    ``accession`` or ``UniProt_ID`` and the case of the rest does not matter.
    """
    if not os.path.isfile(path):
        raise CavityQualityError(
            "--required {0!r} does not exist. Omit the flag to score every cavity "
            "in the manifest; do not point it at a file that is not there.".format(path)
        )
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    columns = {c.lower(): c for c in df.columns}
    accession_col = columns.get("accession") or columns.get("uniprot_id")
    cavity_col = columns.get("cavity_index")
    fragment_col = columns.get("fragment_id")
    if not accession_col or not cavity_col:
        raise CavityQualityError(
            "--required {0!r} needs an accession column (accession or UniProt_ID) "
            "and a cavity_index column; it has {1}".format(path, list(df.columns))
        )
    keys: Set[Tuple[str, str, int]] = set()
    for row in df.to_dict("records"):
        accession = row[accession_col]
        cavity = row[cavity_col]
        if pd.isna(accession) or pd.isna(cavity):
            continue
        fragment = _fragment(row[fragment_col]) if fragment_col else "F1"
        try:
            keys.add((str(accession).strip(), fragment, int(cavity)))
        except (TypeError, ValueError):
            continue
    return keys


def select_targets(
    targets: Sequence[Dict[str, object]],
    required: Optional[Set[Tuple[str, str, int]]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Restrict ``targets`` to ``required``; report what the manifest cannot supply.

    Returns ``(selected, rejects)``. A cavity the required set names but the
    manifest does not carry becomes a ``NOT_IN_MANIFEST`` reject: it would
    otherwise be absent from ``cavity_quality.csv`` with no record of why, and
    the downstream filter would have to guess whether it failed or was never
    looked at.
    """
    if required is None:
        return list(targets), []
    by_key = {
        (t["accession"], t["fragment_id"], int(t["cavity_index"])): t for t in targets
    }
    selected = [by_key[key] for key in sorted(by_key) if key in required]
    rejects = [
        {
            "accession": key[0],
            "fragment_id": key[1],
            "cavity_index": int(key[2]),
            "tarball": tarball_name(key[0], key[1], key[2]),
            "reason": "NOT_IN_MANIFEST",
            "detail": "named by --required but absent from --manifest",
        }
        for key in sorted(required - set(by_key))
    ]
    return selected, rejects


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_cavities(
    tarball_dir: str,
    targets: Sequence[Dict[str, object]],
    threads: int = DEFAULT_THREADS,
    progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score every target. Returns ``(quality, rejects)``.

    Every target lands in exactly one of the two frames;
    :func:`assert_total_accounting` enforces it before returning, so a cavity
    cannot vanish between the manifest and the quality table.

    ``threads`` above 1 uses a thread pool; see the module docstring for why the
    default is 1. Results keep manifest order regardless, because
    ``Executor.map`` yields in submission order.
    """
    if not os.path.isdir(tarball_dir):
        raise CavityQualityError(
            "--tarballs {0!r} is not a directory. A missing prerequisite is an "
            "error, never a fallback.".format(tarball_dir)
        )
    threads = max(1, int(threads))
    targets = list(targets)

    def work(target: Dict[str, object]) -> Dict[str, object]:
        return score_one(tarball_dir, target)

    bar = tqdm(
        total=len(targets), desc="Scoring cavities", unit="cavity", disable=not progress
    )
    results: List[Dict[str, object]] = []
    try:
        if threads == 1:
            for target in targets:
                results.append(work(target))
                bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=threads) as pool:
                for result in pool.map(work, targets):
                    results.append(result)
                    bar.update(1)
    finally:
        bar.close()

    quality_rows = [r for r in results if "reason" not in r]
    reject_rows = [r for r in results if "reason" in r]

    quality = (
        schemas.empty_frame("cavity_quality")
        if not quality_rows
        else pd.DataFrame(quality_rows)[list(schemas.get_schema("cavity_quality").column_names)]
    )
    rejects = pd.DataFrame(reject_rows, columns=list(REJECT_COLUMNS))

    bad = set(rejects["reason"]) - set(REJECT_REASONS)
    if bad:
        raise CavityQualityError(
            "reject ledger used undeclared reason(s) {0}; declared reasons are "
            "{1}".format(", ".join(sorted(bad)), ", ".join(REJECT_REASONS))
        )
    assert_total_accounting(
        n_in=len(targets),
        n_resolved=len(quality),
        n_rejected=len(rejects),
        name=STEP_NAME,
    )
    return quality, rejects


def load_existing(path: str) -> pd.DataFrame:
    """Already-scored cavities from a previous pass, validated. Empty if absent."""
    if not os.path.isfile(path):
        return schemas.empty_frame("cavity_quality")
    return schemas.read_checked(path, "cavity_quality")


def _sorted_quality(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values(
        ["accession", "fragment_id", "cavity_index"], kind="mergesort"
    ).reset_index(drop=True)


def _distribution(series: pd.Series) -> str:
    """``min / p25 / median / p75 / max`` of a numeric column, as one string."""
    if series.empty:
        return "no rows"
    return "min {0:.1f}  p25 {1:.1f}  median {2:.1f}  p75 {3:.1f}  max {4:.1f}".format(
        float(series.min()),
        float(series.quantile(0.25)),
        float(series.median()),
        float(series.quantile(0.75)),
        float(series.max()),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score each CavitySpace pocket's AlphaFold pLDDT into "
            "manifests/cavity_quality.csv, so a later stage can filter pockets "
            "on model confidence without re-reading the archives."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tarballs",
        default=DEFAULT_TARBALL_DIR,
        help=(
            "folder of *.tar.gz CavitySpace cavity archives; a machine-wide "
            "reference dataset, not a per-run location (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help=(
            "cavity_manifest.csv from 'extract_cavities.py --manifest-only' "
            "(a legacy cavity_mapping.csv also works; its archive names are then "
            "derived rather than read)"
        ),
    )
    parser.add_argument(
        "--required",
        default=None,
        help=(
            "score ONLY the cavities this file names -- redock_queue.csv, "
            "positives_planned.csv, or any frame with an accession and a "
            "cavity_index column. Omit to score every cavity in the manifest "
            "(~40 s for 38,709 archives; a candidate subset takes seconds)."
        ),
    )
    parser.add_argument("--out", required=True, help="cavity_quality.csv to write")
    parser.add_argument(
        "--rejects",
        default=None,
        help=(
            "ledger of cavities that could not be scored, one row per cavity "
            "with a reason. Defaults to <out>.rejects.csv. An unscoreable cavity "
            "is recorded here, never written to --out as a zero-filled row."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=(
            "thread-pool size (default: %(default)s). Measured on local disk, "
            "threads make this SLOWER -- 0.7 ms/archive at 1 thread against "
            "2.3 ms at 16 -- because the tar header parsing is pure Python and "
            "each archive is small. Raise it only for a filesystem where an "
            "open blocks, and only after measuring."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "extend an existing --out instead of rebuilding it: cavities already "
            "present are not rescored, so an interrupted pass costs nothing. "
            "Cavities in the reject ledger ARE retried, since a reject can be a "
            "transient filesystem failure."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress the tqdm progress bar (it is already off when --quiet)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "campaign run directory. Optional: it only places this step's own log "
            "and scratch state at <run-dir>/steps/score_cavity_quality/. The "
            "artifacts keep their explicit --out/--rejects paths."
        ),
    )
    add_runtime_arguments(parser, STEP_NAME)
    return parser


def _score(args: argparse.Namespace) -> int:
    """The stage itself. ``main()`` runs this inside :func:`step`."""
    rejects_path = args.rejects or (args.out + ".rejects.csv")

    try:
        targets = load_manifest_targets(args.manifest)
        required = load_required_keys(args.required) if args.required else None
    except (InvariantViolation, OSError, ValueError) as exc:
        LOG.error("%s", exc)
        return 2

    selected, missing_rejects = select_targets(targets, required)
    if required is None:
        LOG.info("scoring every cavity in the manifest: %d", len(selected))
    else:
        LOG.info(
            "--required names %d cavities; %d of them are in the manifest "
            "(of %d), %d are not",
            len(required),
            len(selected),
            len(targets),
            len(missing_rejects),
        )

    existing = schemas.empty_frame("cavity_quality")
    if args.resume:
        try:
            existing = load_existing(args.out)
        except InvariantViolation as exc:
            LOG.error(
                "--resume cannot read the existing %s: %s. Delete it to rebuild, "
                "or fix it; this step will not silently overwrite a file it "
                "cannot validate.",
                args.out,
                exc,
            )
            return 2
        done = set(
            zip(existing["accession"], existing["fragment_id"], existing["cavity_index"])
        )
        before = len(selected)
        selected = [
            t
            for t in selected
            if (t["accession"], t["fragment_id"], int(t["cavity_index"])) not in done
        ]
        LOG.info(
            "--resume: %d already scored in %s, %d of %d left to do",
            len(done),
            args.out,
            len(selected),
            before,
        )

    try:
        scored, rejects = score_cavities(
            tarball_dir=args.tarballs,
            targets=selected,
            threads=args.threads,
            progress=not (args.no_progress or args.quiet),
        )
    except InvariantViolation as exc:
        LOG.error("%s", exc)
        return 2

    if missing_rejects:
        rejects = pd.concat(
            [pd.DataFrame(missing_rejects, columns=list(REJECT_COLUMNS)), rejects],
            ignore_index=True,
        )
    quality = _sorted_quality(
        pd.concat([existing, scored], ignore_index=True) if len(existing) else scored
    )

    schemas.write_checked(quality, args.out, "cavity_quality")
    rejects.to_csv(rejects_path, index=False)

    LOG.info("scored %d cavities this pass, %d rows in %s", len(scored), len(quality), args.out)
    LOG.info("rejected %d cavities -> %s", len(rejects), rejects_path)
    if not rejects.empty:
        for reason, count in rejects["reason"].value_counts().items():
            LOG.info("  %-20s %d", reason, count)
            example = rejects.loc[rejects["reason"] == reason].iloc[0]
            LOG.info("      e.g. %s: %s", example["tarball"], example["detail"])

    if not quality.empty:
        LOG.info("mean pocket pLDDT:  %s", _distribution(quality["mean_plddt"]))
        LOG.info("pocket residues:    %s", _distribution(quality["n_residues"]))
        for cutoff in PRECOMPUTED_CUTOFFS:
            column = quality["frac_ge_{0}".format(cutoff)]
            LOG.info(
                "frac_ge_%-2d:         %s  (%d pockets at >= 0.80, %.1f%% of %d)",
                cutoff,
                _distribution(column),
                int((column >= 0.80).sum()),
                100.0 * float((column >= 0.80).mean()),
                len(quality),
            )

    write_step_summary(
        args.out_dir,
        STEP_NAME,
        {
            "manifest": args.manifest,
            "required": args.required,
            "tarballs": args.tarballs,
            "out": args.out,
            "rejects": rejects_path,
            "threads": int(args.threads),
            "resume": bool(args.resume),
            "n_manifest": len(targets),
            "n_scored_this_pass": int(len(scored)),
            "n_rows": int(len(quality)),
            "n_rejected": int(len(rejects)),
            "reject_reasons": (
                {} if rejects.empty else rejects["reason"].value_counts().to_dict()
            ),
        },
    )
    LOG.info("wrote %s", args.out)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    resolve_out_dir(args, STEP_NAME, args.run_dir or infer_run_dir(args.out))
    try:
        with step(STEP_NAME, args):
            code = _score(args)
            if code:
                # step() reports a non-zero SystemExit as a failure (and notifies
                # about it); main() still owes its caller an int, so catch it back.
                raise SystemExit(code)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
