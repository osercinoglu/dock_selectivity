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
        --interactions manifests/interactions_resolved.csv \\
        --index        identity/identity_index.parquet \\
        --cavities     manifests/cavity_manifest.csv \\
        --ligands      ligands/ \\
        --drugs        /opt/data/drugbank/small_molecule_drug_links.csv \\
        --out          manifests/positives_planned.csv \\
        --rejects      manifests/positives_rejected.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from dock_selectivity import jobid, schemas
from dock_selectivity.identity import IdentityIndex
from dock_selectivity.invariants import InvariantViolation, assert_total_accounting

LOG = logging.getLogger("plan_positives")

#: Matches ``resolve_interactions.py`` so ``selected_by`` provenance concatenates.
_SELECTED_BY_SEP = "; "

#: Every reason an edge can fail to become a planned job. Exhaustive by design:
#: :func:`plan_positives` asserts the rejects it writes use only these.
REJECT_REASONS: Tuple[str, ...] = (
    "NOT_SMALL_MOLECULE",
    "NO_LIGAND_SDF",
    "NO_CAVITY",
    "UNKNOWN_ACCESSION",
)


class PlanningError(InvariantViolation):
    """Raised when the planned set cannot be produced without losing an edge."""


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


def load_available_ligands(ligand_dir: Optional[str]) -> Optional[Set[str]]:
    """DrugBank ids that have a ligand file on disk.

    ``None`` disables the filter entirely (and is reported as such), so a run
    without a prepared ligand folder produces a plan that is explicitly larger
    than what can be docked, rather than a silently empty one.
    """
    if ligand_dir is None:
        return None
    if not os.path.isdir(ligand_dir):
        raise PlanningError(
            "--ligands {!r} is not a directory. Pass the prepared SDF folder, or omit "
            "--ligands to plan without the ligand-availability filter.".format(ligand_dir)
        )
    ids: Set[str] = set()
    for entry in os.listdir(ligand_dir):
        stem, ext = os.path.splitext(entry)
        if ext.lower() in (".sdf", ".mol2", ".mol", ".pdb"):
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
# Planning
# ---------------------------------------------------------------------------


def plan_positives(
    interactions: pd.DataFrame,
    index: IdentityIndex,
    cavities: Dict[Tuple[str, str], List[int]],
    small_molecules: Optional[Set[str]] = None,
    available_ligands: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Expand resolved edges into planned positive jobs.

    Returns ``(planned, rejected)``. Every input pair appears in exactly one of
    them; :func:`assert_total_accounting` enforces it before returning, so a pair
    cannot vanish between the edge list and the plan.

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
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    rejects_path = args.rejects or (args.out + ".rejects.csv")
    if os.path.basename(rejects_path) == "unresolved_targets.csv":
        LOG.error(
            "--rejects must not be identity/unresolved_targets.csv: that is Phase 0's "
            "ledger and audit.py's G2 check asserts its row count equals the index "
            "build's n_rejected. Use a distinct filename."
        )
        return 2

    try:
        index = IdentityIndex.load(args.index)
        interactions = schemas.read_checked(args.interactions, "interactions_resolved")
        cavities = load_cavity_index(args.cavities)
        small_molecules = load_small_molecules(args.drugs)
        ligands = load_available_ligands(args.ligands)
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

    try:
        planned, rejected = plan_positives(
            interactions=interactions,
            index=index,
            cavities=cavities,
            small_molecules=small_molecules,
            available_ligands=ligands,
        )
    except InvariantViolation as exc:
        LOG.error("%s", exc)
        return 3

    schemas.write_checked(planned, args.out, "positives_planned")
    rejected.to_csv(rejects_path, index=False)

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
    LOG.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
