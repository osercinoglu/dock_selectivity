#!/usr/bin/env python3
"""REBUILD_PLAN Phase 2b -- expand the corrected edge list into planned positive jobs.

Reads ``manifests/interactions_resolved.csv`` and emits
``manifests/positives_planned.csv`` (schema ``positives_planned``): one row per
``(drugbank_id, accession, fragment_id, cavity_index)`` the corrected design wants
docked as a positive, **whether or not it already exists on disk**.

Why this stage exists
---------------------

``REBUILD_PLAN.md`` named ``positives_planned.csv`` in its directory layout, three
downstream stages consume it (``sample_negatives.py --positives``,
``curate_jobs.py --positives``, ``audit.py``), and **nothing wrote it**. The
end-to-end test had to synthesise it. This module is that producer.

It is also where the *denominator* of the whole study is decided, which makes it
the one stage where a silent drop does the most damage. An edge disappears here
for three legitimate reasons -- the drug is not a small molecule, its ligand SDF is
absent, or the protein has no CavitySpace cavity -- and each of those shrinks the
positive set, which under balanced sampling shrinks the *negative* set by exactly
as much (a target with zero positives may take zero negatives). So every rejected
edge is written to a ledger with its reason, and the totals are asserted to
reconcile: ``n_edges_in == n_planned_pairs + n_rejected``.

The three filters, and what each costs
--------------------------------------

Measured against the 2025-06-29 inputs, starting from 8,285 resolvable edges:

===============================  =======  ===========================================
Filter                           Pairs    Rationale
===============================  =======  ===========================================
resolvable edges                 9,623    ``interactions_resolved.csv``
+ small molecule                          DrugBank ``small_molecule_drug_links.csv``
+ ligand SDF present on disk              a drug with no 3D ligand cannot be docked
+ >=1 CavitySpace cavity         5,287    a protein with no pocket cannot be docked
``POS_DOCKABLE`` x every cavity  18,166   3.44 cavities per pair
===============================  =======  ===========================================

The cavity expansion is *not* a detail. Positives are one row per cavity of the
target, and the original ``generate_negative_samples.py`` balanced **jobs**, not
pairs -- so a target with eight cavities demanded eight negatives. Getting the
expansion wrong changes the negative demand by the same factor.

The optional fourth filter: pocket confidence
---------------------------------------------

``score_cavity_quality.py`` writes ``manifests/cavity_quality.csv``, the pLDDT the
AlphaFold model carries in the pocket PDB's B-factor column. Passing
``--cavity-quality`` **and** ``--min-plddt-fraction`` drops a cavity whose
confident fraction is too low, *before* the pair expansion -- so a target whose
every pocket fails leaves the plan as one clearly-reasoned rejected pair
(``pocket_quality``) rather than as a pair that quietly expands to zero rows.

The filter is opt-in in both directions: without ``--min-plddt-fraction`` nothing
is filtered at all, and a cavity the manifest names but the quality file does not
carry is an error (exit 2) rather than a silent pass or a silent drop, unless
``--allow-missing-quality`` downgrades it to a warning and treats it as failing.
Filtering that quietly changes the size of a scientific dataset is precisely the
class of defect this repository was rebuilt to eliminate.

Identity discipline
-------------------

``accession`` is the key and arrives already resolved. ``gene`` is written by
``IdentityIndex.label(accession)`` and is a label derived from the key, never an
input to selection -- the property whose absence let ``AVPR2`` select ``Q8NCT1``.
Cavities are matched on ``(accession, fragment_id)``, never on a filename
substring, and ``fragment_id`` is normalised through :func:`jobid.normalize_fragment_id`
because the old campaign wrote ``'F1'`` for positives and bare ``'1'`` for
negatives -- the mismatch that made the sample-type merge non-unique and duplicated
410,087 rows.

Usage
-----

::

    python3 plan_positives.py \\
        --run-dir      $RUN \\
        --interactions $RUN/manifests/interactions_resolved.csv \\
        --index        $RUN/identity/identity_index.parquet \\
        --cavities     $RUN/manifests/cavity_manifest.csv \\
        --ligands      $RUN/ligands/ \\
        --drugs        /opt/data/drugbank/small_molecule_drug_links.csv \\
        --out          $RUN/manifests/positives_planned.csv \\
        --rejects      $RUN/manifests/positives_rejected.csv

    # ...and the same run, keeping only pockets where >=80% of the lining
    # residues reach AlphaFold's "confident" band:
    python3 plan_positives.py \\
        ... \\
        --cavity-quality      $RUN/manifests/cavity_quality.csv \\
        --plddt-min           70 \\
        --min-plddt-fraction  0.80

``--run-dir`` is optional and moves only this step's own log and scratch state to
``$RUN/steps/plan_positives/``; the artifacts keep their explicit ``--out`` paths.
Without it the step directory is inferred from ``--out``, so the log still lands in
the campaign rather than in the invocation directory.  ``--notify`` (with
``PUSHOVER_TOKEN`` / ``PUSHOVER_USER_KEY`` in the environment) sends a Pushover
message when the stage finishes or fails.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import pandas as pd

from dock_selectivity import jobid, quality, schemas
from dock_selectivity.identity import IdentityIndex
from dock_selectivity.invariants import InvariantViolation, assert_total_accounting
from dock_selectivity.runtime import add_runtime_arguments, resolve_out_dir, step

#: Names the logger, the step directory and the log file. Must stay equal to the
#: logger name below, or ``step()`` would configure a logger this module never uses.
STEP_NAME = "plan_positives"

LOG = logging.getLogger(STEP_NAME)

#: Directories the fixed run layout puts artifacts in; their parent is the run
#: directory. Consulted only to place this step's own log when ``--run-dir`` is
#: absent -- the log must land in the campaign, never in the invocation directory.
_LAYOUT_DIRS = ("manifests", "identity", "analysis", "structures", "docking_results")

#: Matches ``resolve_interactions.py`` so ``selected_by`` provenance concatenates.
_SELECTED_BY_SEP = "; "

#: File extensions that count as a prepared ligand. Overridable with
#: ``--ligand-extensions``: which formats the ligand folder holds is a property of
#: the run, not of this module.
DEFAULT_LIGAND_EXTENSIONS: Tuple[str, ...] = (".sdf", ".mol2", ".mol", ".pdb")

#: pLDDT a pocket residue must reach to count as confident. AlphaFold's own
#: "confident" band starts at 70; it is a default only for *where* the bar sits,
#: never for whether the filter runs -- that is ``--min-plddt-fraction``, which
#: has no default anywhere in the codebase.
DEFAULT_PLDDT_MIN = 70.0

#: Every reason an edge can fail to become a planned job. Exhaustive by design:
#: :func:`plan_positives` asserts the rejects it writes use only these.
#:
#: ``pocket_quality`` is lower-case on purpose: it is the reason string the
#: cavity-quality contract fixes across ``plan_positives.py`` and
#: ``sample_negatives.py``, so a downstream reader can count pockets lost to the
#: pLDDT filter across both ledgers with one comparison.
REJECT_REASONS: Tuple[str, ...] = (
    "NOT_SMALL_MOLECULE",
    "NO_LIGAND_SDF",
    "NO_CAVITY",
    "UNKNOWN_ACCESSION",
    "pocket_quality",
)


class PlanningError(InvariantViolation):
    """Raised when the planned set cannot be produced without losing an edge."""


def infer_run_dir(artifact_path: str) -> str:
    """Best guess at the run directory, from an artifact this step writes.

    ``--run-dir`` is optional so every existing command line (including the ones
    ``run_full_workflow.py`` builds) keeps working unchanged, but the step log may
    not fall back to the invocation directory -- that is how the repository working
    tree filled up with stray ``*.log`` files.
    """
    parent = os.path.dirname(os.path.abspath(artifact_path))
    if os.path.basename(parent) in _LAYOUT_DIRS:
        return os.path.dirname(parent)
    return parent


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_cavity_index(path: str) -> Dict[Tuple[str, str], List[int]]:
    """Map ``(accession, fragment_id)`` -> sorted cavity indices.

    Accepts either a ``cavity_manifest.csv`` produced by the fixed
    ``extract_cavities.py --manifest-only`` or the legacy ``cavity_mapping.csv``;
    both carry ``UniProt_ID``, ``Fragment_ID``, ``Cavity_Index``.

    Fragment ids are normalised, so ``1`` and ``'F1'`` land in the same bucket.
    Cavity indices are returned sorted so the emitted plan is deterministic
    regardless of input row order -- the property whose absence made the original
    mapping non-reproducible.
    """
    df = pd.read_csv(path)
    required = {"UniProt_ID", "Fragment_ID", "Cavity_Index"}
    missing = required - set(df.columns)
    if missing:
        raise PlanningError(
            "cavity file {!r} is missing column(s) {}; expected a cavity_manifest.csv "
            "or cavity_mapping.csv with {}".format(
                path, ", ".join(sorted(missing)), ", ".join(sorted(required))
            )
        )

    out: Dict[Tuple[str, str], Set[int]] = {}
    for acc, frag, cav in zip(df["UniProt_ID"], df["Fragment_ID"], df["Cavity_Index"]):
        if pd.isna(acc) or pd.isna(cav):
            continue
        fragment = jobid.normalize_fragment_id(frag) or "F1"
        try:
            cavity = int(cav)
        except (TypeError, ValueError):
            continue
        out.setdefault((str(acc).strip(), fragment), set()).add(cavity)
    return {key: sorted(values) for key, values in out.items()}


def load_available_ligands(
    ligand_dir: Optional[str],
    extensions: Optional[Sequence[str]] = None,
) -> Optional[Set[str]]:
    """DrugBank ids that have a ligand file on disk.

    ``None`` disables the filter entirely (and is reported as such), so a run
    without a prepared ligand folder produces a plan that is explicitly larger
    than what can be docked, rather than a silently empty one.

    ``extensions`` defaults to :data:`DEFAULT_LIGAND_EXTENSIONS`; a ligand set
    written in some other format would otherwise reject every pair with
    ``NO_LIGAND_SDF`` and look like a data problem.
    """
    if ligand_dir is None:
        return None
    if not os.path.isdir(ligand_dir):
        raise PlanningError(
            "--ligands {!r} is not a directory. Pass the prepared SDF folder, or omit "
            "--ligands to plan without the ligand-availability filter.".format(ligand_dir)
        )
    wanted = tuple(
        e.lower() if e.startswith(".") else "." + e.lower()
        for e in (extensions if extensions is not None else DEFAULT_LIGAND_EXTENSIONS)
    )
    ids: Set[str] = set()
    for entry in os.listdir(ligand_dir):
        stem, ext = os.path.splitext(entry)
        if ext.lower() in wanted:
            ids.add(stem.strip().upper())
    return ids


def load_small_molecules(path: Optional[str]) -> Optional[Set[str]]:
    """DrugBank ids classed as small molecules. ``None`` disables the filter."""
    if path is None:
        return None
    df = pd.read_csv(path)
    for candidate in ("DrugBank ID", "drugbank_id", "DrugBank_ID"):
        if candidate in df.columns:
            return {str(v).strip().upper() for v in df[candidate] if not pd.isna(v)}
    raise PlanningError(
        "{!r} has no DrugBank ID column (looked for 'DrugBank ID', 'drugbank_id', "
        "'DrugBank_ID'); columns are {}".format(path, list(df.columns))
    )


# ---------------------------------------------------------------------------
# Pocket confidence
# ---------------------------------------------------------------------------


#: ``<rejects>.quality_rejects.csv`` -- one row per *pocket* the pLDDT filter
#: dropped, with the numbers that decided it. ``positives_rejected.csv`` stays a
#: one-row-per-PAIR ledger (``assert_total_accounting`` is a pair-level
#: invariant), so a pair that keeps some pockets and loses others would otherwise
#: have its losses nowhere but the log. The house rule is that a rejected row is
#: recorded with a reason, and a dropped pocket is a dropped docking job.
QUALITY_REJECT_COLUMNS: Tuple[str, ...] = (
    "accession",
    "fragment_id",
    "cavity_index",
    "reason",
    "detail",
)


class QualityFilterResult(NamedTuple):
    """What :func:`filter_cavities_by_quality` did, in numbers an operator reads.

    ``cavities`` is the surviving cavity index, in the same shape
    :func:`load_cavity_index` returns, so the pair expansion never learns that a
    filter ran. ``failures`` maps an accession to one human-readable line per
    pocket it lost, which is what a pair that loses *every* pocket is rejected
    with -- the numbers travel into the ledger instead of only into the log.
    ``rejected_pockets`` is the same losses one row per pocket, for the
    ``quality_rejects`` sidecar, so a pocket dropped from a pair that survives on
    its other pockets is still recorded rather than only counted.
    """

    cavities: Dict[Tuple[str, str], List[int]]
    failures: Dict[str, List[str]]
    rejected_pockets: pd.DataFrame
    n_considered: int
    n_passed: int
    n_failed: int
    n_missing: int
    n_accessions_before: int
    n_accessions_after: int


def _quality_key(
    accession: object, fragment: object, cavity: object
) -> Tuple[str, str, int]:
    """The pocket identity, normalised the same way on both sides of the join.

    ``score_cavity_quality.py`` already writes ``'F1'``, but normalising here too
    means a hand-built or legacy quality file cannot half-match: ``'1'`` and
    ``'F1'`` are one pocket, never two.
    """
    return (
        str(accession).strip(),
        jobid.normalize_fragment_id(fragment) or "F1",
        int(cavity),
    )


def filter_cavities_by_quality(
    cavities: Dict[Tuple[str, str], List[int]],
    quality_frame: pd.DataFrame,
    plddt_min: float = DEFAULT_PLDDT_MIN,
    min_fraction: Optional[float] = None,
    allow_missing: bool = False,
) -> QualityFilterResult:
    """Drop cavities whose confident fraction is below ``min_fraction``.

    Applied to the cavity index *before* the pair expansion, so a target whose
    every pocket fails leaves the plan as a rejected pair carrying the actual
    numbers, not as a pair that silently expands to zero rows.

    ``min_fraction`` is required by :func:`dock_selectivity.quality.passes` and
    has no default here either: supplying it is what turns the filter on.

    A cavity named by ``cavities`` but absent from ``quality_frame`` raises
    :class:`PlanningError` naming how many and showing examples -- unless
    ``allow_missing``, which treats those pockets as failing and records them in
    ``failures`` like any other loss.
    """
    keep = quality.passes(quality_frame, plddt_min, min_fraction)
    failing = quality_frame.loc[~keep]
    failed_fraction = (
        quality.fraction_at(failing, plddt_min)
        if len(failing)
        else pd.Series([], dtype="float64")
    )

    passing_keys: Set[Tuple[str, str, int]] = {
        _quality_key(row.accession, row.fragment_id, row.cavity_index)
        for row in quality_frame.loc[keep].itertuples(index=False)
    }
    failed_detail: Dict[Tuple[str, str, int], str] = {}
    for row, fraction in zip(failing.itertuples(index=False), failed_fraction):
        failed_detail[_quality_key(row.accession, row.fragment_id, row.cavity_index)] = (
            "frac_ge_%.0f=%.2f < %.2f (n_residues=%d)"
            % (float(plddt_min), float(fraction), float(min_fraction), int(row.n_residues))
        )

    kept: Dict[Tuple[str, str], List[int]] = {}
    failures: Dict[str, List[str]] = {}
    rejected_rows: List[Dict[str, object]] = []
    missing: List[str] = []
    n_considered = 0
    n_failed = 0

    def record(accession: str, fragment: str, cavity: int, reason: str, detail: str) -> None:
        """One ledger row per pocket, so a partial loss is never only a log line."""
        rejected_rows.append(
            {
                "accession": str(accession),
                "fragment_id": str(fragment),
                "cavity_index": int(cavity),
                "reason": reason,
                "detail": detail,
            }
        )

    for (accession, fragment), indices in sorted(cavities.items()):
        survivors: List[int] = []
        for cavity in indices:
            n_considered += 1
            key = _quality_key(accession, fragment, cavity)
            if key in passing_keys:
                survivors.append(cavity)
                continue
            if key in failed_detail:
                n_failed += 1
                failures.setdefault(accession, []).append(
                    "%s/cav%d %s" % (fragment, cavity, failed_detail[key])
                )
                record(
                    accession, fragment, cavity, "pocket_quality", failed_detail[key]
                )
                continue
            missing.append("%s/%s/cav%d" % (accession, fragment, cavity))
            if allow_missing:
                n_failed += 1
                unscored = (
                    "absent from the cavity-quality file "
                    "(--allow-missing-quality treats an unscored pocket as failing)"
                )
                failures.setdefault(accession, []).append(
                    "%s/cav%d %s" % (fragment, cavity, unscored)
                )
                record(accession, fragment, cavity, "pocket_quality", unscored)
        if survivors:
            kept[(accession, fragment)] = survivors

    if missing and not allow_missing:
        raise PlanningError(
            "{0} cavity/cavities named by --cavities are absent from the "
            "cavity-quality file, e.g. {1}. Score them with "
            "score_cavity_quality.py, or pass --allow-missing-quality to treat "
            "an unscored pocket as failing. Filtering on a quality file that "
            "does not cover the manifest would change the size of the dataset "
            "without saying so.".format(len(missing), ", ".join(sorted(missing)[:5]))
        )

    rejected_pockets = pd.DataFrame(
        rejected_rows, columns=list(QUALITY_REJECT_COLUMNS)
    )
    if len(rejected_pockets):
        rejected_pockets = rejected_pockets.sort_values(
            ["accession", "fragment_id", "cavity_index"], kind="mergesort"
        ).reset_index(drop=True)

    return QualityFilterResult(
        cavities=kept,
        failures=failures,
        rejected_pockets=rejected_pockets,
        n_considered=n_considered,
        n_passed=sum(len(values) for values in kept.values()),
        n_failed=n_failed,
        n_missing=len(missing),
        n_accessions_before=len({accession for accession, _ in cavities}),
        n_accessions_after=len({accession for accession, _ in kept}),
    )


def _quality_reject_detail(failures: Sequence[str], max_shown: int = 3) -> str:
    """One ledger cell describing every pocket a pair lost, numbers included."""
    shown = list(failures[:max_shown])
    text = "; ".join(shown)
    if len(failures) > len(shown):
        text += "; ... (%d more)" % (len(failures) - len(shown))
    return "all %d pocket(s) failed the pLDDT filter: %s" % (len(failures), text)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan_positives(
    interactions: pd.DataFrame,
    index: IdentityIndex,
    cavities: Dict[Tuple[str, str], List[int]],
    small_molecules: Optional[Set[str]] = None,
    available_ligands: Optional[Set[str]] = None,
    quality_failures: Optional[Dict[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Expand resolved edges into planned positive jobs.

    Returns ``(planned, rejected)``. Every input pair appears in exactly one of
    them; :func:`assert_total_accounting` enforces it before returning, so a pair
    cannot vanish between the edge list and the plan.

    ``cavities`` is expected to be already filtered (see
    :func:`filter_cavities_by_quality`); ``quality_failures`` is that filter's
    per-accession record, and it is what turns "this target has no cavity left"
    into a ``pocket_quality`` rejection carrying the numbers rather than an
    indistinguishable ``NO_CAVITY``.

    Rows are emitted in sorted ``(drugbank_id, accession, fragment_id,
    cavity_index)`` order, making the output byte-identical across runs and across
    ``PYTHONHASHSEED`` values.
    """
    planned_rows: List[Dict[str, object]] = []
    reject_rows: List[Dict[str, object]] = []

    pairs = interactions.drop_duplicates(subset=["drugbank_id", "accession"])
    n_in = len(pairs)

    # Group the cavity index by accession once, rather than per row.
    by_accession: Dict[str, List[Tuple[str, List[int]]]] = {}
    for (acc, fragment), indices in cavities.items():
        by_accession.setdefault(acc, []).append((fragment, indices))
    for acc in by_accession:
        by_accession[acc].sort(key=lambda item: item[0])

    for row in pairs.itertuples(index=False):
        drug = str(getattr(row, "drugbank_id")).strip()
        accession = str(getattr(row, "accession")).strip()
        selected_by = getattr(row, "selected_by", "") or ""

        def reject(reason: str, detail: str) -> None:
            reject_rows.append(
                {
                    "drugbank_id": drug,
                    "accession": accession,
                    "reason": reason,
                    "detail": detail,
                    "selected_by": selected_by,
                }
            )

        if small_molecules is not None and drug.upper() not in small_molecules:
            reject("NOT_SMALL_MOLECULE", "not in the DrugBank small-molecule list")
            continue
        if available_ligands is not None and drug.upper() not in available_ligands:
            reject("NO_LIGAND_SDF", "no prepared ligand file on disk")
            continue

        try:
            gene = index.label(accession)
        except KeyError:
            reject("UNKNOWN_ACCESSION", "accession absent from the identity index")
            continue

        fragments = by_accession.get(accession)
        if not fragments:
            lost = (quality_failures or {}).get(accession)
            if lost:
                # The target HAS cavities; the pLDDT filter took all of them.
                # Saying NO_CAVITY here would blame the structure set for a
                # threshold the operator chose.
                reject("pocket_quality", _quality_reject_detail(lost))
            else:
                reject("NO_CAVITY", "no CavitySpace cavity for this accession")
            continue

        for fragment, indices in fragments:
            for cavity in indices:
                planned_rows.append(
                    {
                        "drugbank_id": drug,
                        "accession": accession,
                        "gene": gene,
                        "fragment_id": fragment,
                        "cavity_index": int(cavity),
                        "sample_type": "positive",
                        "selected_by": selected_by,
                    }
                )

    planned = _frame(planned_rows, "positives_planned")
    if not planned.empty:
        planned = planned.sort_values(
            ["drugbank_id", "accession", "fragment_id", "cavity_index"],
            kind="mergesort",
        ).reset_index(drop=True)

    rejected = pd.DataFrame(
        reject_rows,
        columns=["drugbank_id", "accession", "reason", "detail", "selected_by"],
    )
    if not rejected.empty:
        rejected = rejected.sort_values(
            ["drugbank_id", "accession"], kind="mergesort"
        ).reset_index(drop=True)
        bad = set(rejected["reason"]) - set(REJECT_REASONS)
        if bad:
            raise PlanningError(
                "reject ledger used undeclared reason(s) {}; declared reasons are {}".format(
                    ", ".join(sorted(bad)), ", ".join(REJECT_REASONS)
                )
            )

    n_planned_pairs = (
        0
        if planned.empty
        else len(planned.drop_duplicates(subset=["drugbank_id", "accession"]))
    )
    assert_total_accounting(
        n_in=n_in,
        n_resolved=n_planned_pairs,
        n_rejected=len(rejected),
        name="plan_positives",
    )
    return planned, rejected


def _frame(rows: Sequence[Dict[str, object]], schema_name: str) -> pd.DataFrame:
    if not rows:
        return schemas.empty_frame(schema_name)
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Expand interactions_resolved.csv into positives_planned.csv -- one row "
            "per (drug, accession, fragment, cavity) positive job the design wants."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--interactions", required=True, help="interactions_resolved.csv")
    parser.add_argument("--index", required=True, help="identity_index.parquet")
    parser.add_argument(
        "--cavities",
        required=True,
        help="cavity_manifest.csv (or legacy cavity_mapping.csv)",
    )
    parser.add_argument(
        "--ligands",
        default=None,
        help=(
            "Prepared ligand folder. Omit to skip the ligand-availability filter "
            "(the plan is then larger than what can actually be docked, and says so)."
        ),
    )
    parser.add_argument(
        "--drugs",
        default=None,
        help="small_molecule_drug_links.csv. Omit to skip the small-molecule filter.",
    )
    parser.add_argument("--out", required=True, help="positives_planned.csv to write")
    parser.add_argument(
        "--rejects",
        default=None,
        help=(
            "Rejected-pair ledger. Defaults to <out>.rejects.csv. Must NOT point at "
            "identity/unresolved_targets.csv -- that is Phase 0's ledger and "
            "audit.py's G2 check asserts its row count."
        ),
    )
    parser.add_argument(
        "--ligand-extensions",
        default=",".join(DEFAULT_LIGAND_EXTENSIONS),
        help=(
            "comma-separated file extensions that count as a prepared ligand when "
            "--ligands is given (default: %(default)s). A ligand set written in a "
            "format not listed here would silently reject every pair."
        ),
    )
    parser.add_argument(
        "--cavity-quality",
        default=None,
        help=(
            "manifests/cavity_quality.csv from score_cavity_quality.py: each "
            "pocket's AlphaFold pLDDT. Without it, no pocket-quality filtering "
            "happens at all."
        ),
    )
    parser.add_argument(
        "--plddt-min",
        type=float,
        default=DEFAULT_PLDDT_MIN,
        help=(
            "pLDDT a pocket residue must reach to count as confident "
            "(default: %(default)s, AlphaFold's 'confident' band). Only has an "
            "effect together with --min-plddt-fraction."
        ),
    )
    parser.add_argument(
        "--min-plddt-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of a pocket's residues that must reach --plddt-min for the "
            "pocket to be planned, e.g. 0.80. NO DEFAULT: supplying it is what "
            "turns pocket-quality filtering on; without it nothing is filtered."
        ),
    )
    parser.add_argument(
        "--quality-rejects",
        default=None,
        help=(
            "where to write the per-pocket pLDDT reject ledger (one row per "
            "cavity the filter dropped, with the numbers that decided it). "
            "Default <rejects>.quality_rejects.csv; written only when "
            "--min-plddt-fraction is given. positives_rejected.csv stays one row "
            "per PAIR, so without this file a pair that loses some but not all "
            "of its pockets would have those losses nowhere but the log."
        ),
    )
    parser.add_argument(
        "--allow-missing-quality",
        action="store_true",
        help=(
            "Downgrade 'cavity named by --cavities but absent from "
            "--cavity-quality' from an error to a warning, and treat those "
            "pockets as FAILING. Without it such a cavity exits 2: filtering on "
            "a quality file that does not cover the manifest would change the "
            "size of the dataset without saying so."
        ),
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "campaign run directory. Optional: it only places this step's own log "
            "and scratch state at <run-dir>/steps/plan_positives/. The artifacts "
            "keep their explicit --out/--rejects paths."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="debug-level logging (same as --log-level DEBUG)")
    add_runtime_arguments(parser, STEP_NAME)
    return parser


def _plan(args: argparse.Namespace) -> int:
    """The stage itself. ``main()`` runs this inside :func:`step`."""
    rejects_path = args.rejects or (args.out + ".rejects.csv")
    quality_rejects_path = args.quality_rejects or (
        os.path.splitext(rejects_path)[0] + ".quality_rejects.csv"
    )
    if os.path.basename(rejects_path) == "unresolved_targets.csv":
        LOG.error(
            "--rejects must not be identity/unresolved_targets.csv: that is Phase 0's "
            "ledger and audit.py's G2 check asserts its row count equals the index "
            "build's n_rejected. Use a distinct filename."
        )
        return 2

    if args.min_plddt_fraction is not None and not args.cavity_quality:
        LOG.error(
            "--min-plddt-fraction %.2f was given without --cavity-quality: there "
            "is no pLDDT to filter on. Run score_cavity_quality.py and pass its "
            "cavity_quality.csv, or drop --min-plddt-fraction to plan without "
            "the pocket filter.",
            float(args.min_plddt_fraction),
        )
        return 2

    try:
        index = IdentityIndex.load(args.index)
        interactions = schemas.read_checked(args.interactions, "interactions_resolved")
        cavities = load_cavity_index(args.cavities)
        small_molecules = load_small_molecules(args.drugs)
        ligands = load_available_ligands(
            args.ligands,
            extensions=[e for e in str(args.ligand_extensions).split(",") if e.strip()],
        )
        quality_frame = (
            quality.load_cavity_quality(args.cavity_quality)
            if args.cavity_quality
            else None
        )
    except (InvariantViolation, OSError, ValueError) as exc:
        LOG.error("%s", exc)
        return 2

    if ligands is None:
        LOG.warning(
            "--ligands not given: planning WITHOUT the ligand-availability filter. "
            "The plan will contain pairs whose ligand may not exist on disk."
        )
    if small_molecules is None:
        LOG.warning("--drugs not given: planning WITHOUT the small-molecule filter.")

    quality_failures: Optional[Dict[str, List[str]]] = None
    quality_rejects: Optional[pd.DataFrame] = None
    if quality_frame is not None and args.min_plddt_fraction is None:
        LOG.warning(
            "--cavity-quality given without --min-plddt-fraction: the %d scored "
            "pockets are NOT filtered. Supplying --min-plddt-fraction is what "
            "turns the pocket filter on.",
            len(quality_frame),
        )
    elif quality_frame is not None:
        try:
            result = filter_cavities_by_quality(
                cavities=cavities,
                quality_frame=quality_frame,
                plddt_min=args.plddt_min,
                min_fraction=args.min_plddt_fraction,
                allow_missing=args.allow_missing_quality,
            )
        except InvariantViolation as exc:
            LOG.error("%s", exc)
            return 2
        if result.n_missing:
            LOG.warning(
                "%d cavity/cavities are absent from %s and were treated as "
                "FAILING (--allow-missing-quality).",
                result.n_missing,
                args.cavity_quality,
            )
        LOG.info(
            "pocket quality: keeping pockets with frac_ge_%.0f >= %.2f (from %s)",
            float(args.plddt_min),
            float(args.min_plddt_fraction),
            args.cavity_quality,
        )
        LOG.info(
            "  cavities %d -> %d (%d failed the pLDDT filter, %d unscored)",
            result.n_considered,
            result.n_passed,
            result.n_failed,
            result.n_missing,
        )
        LOG.info(
            "  accessions with >=1 pocket %d -> %d",
            result.n_accessions_before,
            result.n_accessions_after,
        )
        cavities = result.cavities
        quality_failures = result.failures
        quality_rejects = result.rejected_pockets

    try:
        planned, rejected = plan_positives(
            interactions=interactions,
            index=index,
            cavities=cavities,
            small_molecules=small_molecules,
            available_ligands=ligands,
            quality_failures=quality_failures,
        )
    except InvariantViolation as exc:
        LOG.error("%s", exc)
        return 3

    schemas.write_checked(planned, args.out, "positives_planned")
    rejected.to_csv(rejects_path, index=False)
    if quality_rejects is not None:
        # Written whenever the filter ran, empty file included: an absent file
        # means "no filtering was asked for", an empty one "filtered, dropped
        # nothing". The two must not look alike.
        quality_rejects.to_csv(quality_rejects_path, index=False, lineterminator="\n")

    n_pairs = (
        0
        if planned.empty
        else len(planned.drop_duplicates(subset=["drugbank_id", "accession"]))
    )
    LOG.info("planned %d positive jobs over %d pairs", len(planned), n_pairs)
    LOG.info(
        "  %d drugs, %d target accessions, %.2f cavities/pair",
        0 if planned.empty else planned["drugbank_id"].nunique(),
        0 if planned.empty else planned["accession"].nunique(),
        (len(planned) / n_pairs) if n_pairs else 0.0,
    )
    LOG.info("rejected %d pairs -> %s", len(rejected), rejects_path)
    if not rejected.empty:
        for reason, count in rejected["reason"].value_counts().items():
            LOG.info("  %-20s %d", reason, count)
    if quality_failures is not None:
        n_lost = (
            0 if rejected.empty else int((rejected["reason"] == "pocket_quality").sum())
        )
        LOG.info(
            "pocket quality: %d pairs planned, %d pairs lost every pocket to the "
            "filter (recorded as 'pocket_quality' in %s)",
            n_pairs,
            n_lost,
            rejects_path,
        )
        if quality_rejects is not None:
            LOG.info(
                "  %d pocket(s) dropped for quality, each recorded in %s",
                len(quality_rejects),
                quality_rejects_path,
            )
    LOG.info("wrote %s", args.out)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose:
        # -v predates --log-level and run_full_workflow may still pass it.
        args.log_level = "DEBUG"
    resolve_out_dir(args, STEP_NAME, args.run_dir or infer_run_dir(args.out))
    try:
        with step(STEP_NAME, args):
            code = _plan(args)
            if code:
                # step() reports a non-zero SystemExit as a failure (and notifies
                # about it); main() still owes its caller an int, so catch it back.
                raise SystemExit(code)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
