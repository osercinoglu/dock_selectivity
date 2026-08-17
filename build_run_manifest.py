#!/usr/bin/env python3
"""REBUILD_PLAN Phase 4 -- assemble run_manifest.csv and the complete redock queue.

Emits ``manifests/run_manifest.csv`` (schema ``run_manifest``): **every job the
corrected design says should exist**, whether it is reused from the old campaign
or has yet to be docked. Optionally emits ``manifests/redock_queue.csv`` (schema
``redock_queue``): the subset that must be docked fresh.

Why this stage exists
---------------------

Two holes, both found by running the chain end to end.

**1. Nothing wrote the full manifest.** ``curate_jobs.py --manifest-out`` writes
``run_manifest`` rows only for jobs it actually copied -- i.e. reused ones. But
the manifest's whole purpose is to be *the authority on what should exist*, so
that ``parse_prepare.py`` can check discovery against it and a truncated pass
becomes detectable. The 2025-06-29 campaign lost 14,173 completed jobs (58%) to
``max_files_per_pattern = 10000`` and nothing noticed, because there was nothing
to notice against. A manifest listing only the reused half would reproduce that
blind spot for every newly-docked job.

**2. Nothing queued the positive re-dock gap.** ``sample_negatives.py`` emits a
redock queue for the *negative* side. The positive side has a far larger gap --
only 2,100 of 8,285 intended pairs were ever docked against the correct protein --
and no stage turned it into a queue. ``REBUILD_PLAN`` Phase 4 said "redock_queue.csv
now exists"; for positives it did not. This module closes both halves.

How a planned job is matched to an existing one
-----------------------------------------------

On the full pocket key ``(drugbank_id, accession, fragment_id, cavity_index)`` --
never on a directory-name substring, and never on a partial key. Dropping
``fragment_id`` from a key of this shape is exactly what made the old sample-type
merge non-unique and duplicated 410,087 rows, so the match is asserted to be
one-to-one: a planned job resolves to at most one reused directory, and
:func:`assert_unique_key` enforces that ``job_id`` is unique across the whole
manifest before it is written.

Fragment ids are normalised on both sides, because the old campaign wrote ``'F1'``
for positives and bare ``'1'`` for negatives.

Usage
-----

::

    python3 build_run_manifest.py \\
        --run-dir      $RUN \\
        --positives    $RUN/manifests/positives_planned.csv \\
        --drawn        $RUN/manifests/negatives_drawn.csv \\
        --job-classification  $RUN/manifests/job_classification.csv \\
        --out          $RUN/manifests/run_manifest.csv \\
        --redock-out   $RUN/manifests/redock_queue.csv

``--run-dir`` is optional and moves only this step's own log and scratch state to
``$RUN/steps/build_run_manifest/``; the manifests keep their explicit ``--out`` /
``--redock-out`` paths.  Without it the step directory is inferred from ``--out``,
so the log still lands in the campaign rather than in the invocation directory.
``--notify`` (with ``PUSHOVER_TOKEN`` / ``PUSHOVER_USER_KEY`` in the environment)
sends a Pushover message when the stage finishes or fails.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from dock_selectivity import jobid, schemas
from dock_selectivity.invariants import InvariantViolation, assert_unique_key
from dock_selectivity.runtime import add_runtime_arguments, resolve_out_dir, step

#: Names the logger, the step directory and the log file. Must stay equal to the
#: logger name below, or ``step()`` would configure a logger this module never uses.
STEP_NAME = "build_run_manifest"

LOG = logging.getLogger(STEP_NAME)

#: Directories the fixed run layout puts artifacts in; their parent is the run
#: directory. Consulted only to place this step's own log when ``--run-dir`` is
#: absent -- the log must land in the campaign, never in the invocation directory.
_LAYOUT_DIRS = ("manifests", "identity", "analysis", "structures", "docking_results")

#: Reuse classification that makes a job reusable as a positive.
_TRUE_POSITIVE = "true_positive"

#: ``negatives_drawn.source`` value meaning "already docked, reuse it".
_FROM_POOL = "pool"


class ManifestError(InvariantViolation):
    """Raised when the manifest cannot be assembled without ambiguity or loss."""


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


def _norm_fragment(value: Any) -> str:
    return jobid.normalize_fragment_id(value) or "F1"


def _pocket_key(row: Any) -> Tuple[str, str, str, int]:
    """The full four-part pocket key. Partial keys are how the old merge broke."""
    return (
        str(row["drugbank_id"]).strip(),
        str(row["accession"]).strip(),
        _norm_fragment(row.get("fragment_id")),
        int(row["cavity_index"]),
    )


def index_reused_positives(job_classification: pd.DataFrame) -> Dict[Tuple[str, str, str, int], str]:
    """Pocket key -> old directory, for jobs reusable as positives.

    Raises if two reused directories claim the same pocket: that would make the
    manifest ambiguous about which one a planned positive resolves to, and silently
    picking one is the class of mistake this whole rebuild exists to eliminate.
    """
    out: Dict[Tuple[str, str, str, int], str] = {}
    if job_classification.empty:
        return out
    rows = job_classification[job_classification["classification"] == _TRUE_POSITIVE]
    for _, row in rows.iterrows():
        if pd.isna(row.get("cavity_index")):
            continue
        key = _pocket_key(row)
        old_dir = str(row["old_job_dir"]).strip()
        if key in out and out[key] != old_dir:
            raise ManifestError(
                "two reused directories claim pocket {}: {!r} and {!r}. The manifest "
                "cannot say which a planned positive resolves to.".format(
                    key, out[key], old_dir
                )
            )
        out[key] = old_dir
    return out


def build_manifest(
    positives: pd.DataFrame,
    drawn: pd.DataFrame,
    job_classification: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble ``(run_manifest, redock_queue)``.

    Every row of ``positives`` and ``drawn`` becomes exactly one manifest row.
    Rows whose provenance is ``newly_docked`` also become a ``redock_queue`` row,
    so the two outputs reconcile by construction:
    ``len(redock_queue) == (manifest.provenance == 'newly_docked').sum()``.
    """
    reused = index_reused_positives(job_classification)
    rows: List[Dict[str, Any]] = []

    for _, row in positives.iterrows():
        key = _pocket_key(row)
        old_dir = reused.get(key)
        rows.append(
            {
                "job_id": jobid.format_job_dir(
                    drugbank_id=key[0],
                    accession=key[1],
                    gene=str(row["gene"]),
                    fragment_id=key[2],
                    cavity_index=key[3],
                ),
                "drugbank_id": key[0],
                "accession": key[1],
                "gene": str(row["gene"]),
                "fragment_id": key[2],
                "cavity_index": key[3],
                "sample_type": "positive",
                "provenance": "reused_from" if old_dir else "newly_docked",
                "source_job": old_dir,
                "selected_by": row.get("selected_by", "") or "",
            }
        )

    for _, row in drawn.iterrows():
        key = _pocket_key(row)
        from_pool = str(row.get("source", "")).strip() == _FROM_POOL
        old_dir = row.get("old_job_dir")
        old_dir = None if (pd.isna(old_dir) or not str(old_dir).strip()) else str(old_dir).strip()
        if from_pool and not old_dir:
            raise ManifestError(
                "negatives_drawn row {} says source='pool' but names no old_job_dir; "
                "it cannot be reused from nowhere.".format(key)
            )
        rows.append(
            {
                "job_id": jobid.format_job_dir(
                    drugbank_id=key[0],
                    accession=key[1],
                    gene=str(row["gene"]),
                    fragment_id=key[2],
                    cavity_index=key[3],
                ),
                "drugbank_id": key[0],
                "accession": key[1],
                "gene": str(row["gene"]),
                "fragment_id": key[2],
                "cavity_index": key[3],
                "sample_type": "negative",
                "provenance": "reused_from" if from_pool else "newly_docked",
                "source_job": old_dir if from_pool else None,
                "selected_by": row.get("selected_by", "") or "",
            }
        )

    manifest = (
        schemas.empty_frame("run_manifest")
        if not rows
        else pd.DataFrame(rows).sort_values("job_id", kind="mergesort").reset_index(drop=True)
    )

    if not manifest.empty:
        # A duplicate job_id would mean two design rows want the same directory --
        # e.g. the same pocket planned as both a positive and a negative.
        assert_unique_key(manifest, ["job_id"], name="run_manifest")

    queue_rows = (
        manifest[manifest["provenance"] == "newly_docked"]
        if not manifest.empty
        else manifest
    )
    if queue_rows.empty:
        redock = schemas.empty_frame("redock_queue")
    else:
        redock = queue_rows[
            [
                "job_id",
                "drugbank_id",
                "accession",
                "gene",
                "fragment_id",
                "cavity_index",
                "sample_type",
                "selected_by",
            ]
        ].reset_index(drop=True)

    return manifest, redock


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble run_manifest.csv (every job the design says should exist) and "
            "the complete redock_queue.csv (the subset that must be docked fresh)."
        )
    )
    parser.add_argument("--positives", required=True, help="positives_planned.csv")
    parser.add_argument("--drawn", required=True, help="negatives_drawn.csv")
    parser.add_argument("--job-classification", default=None,
                        help=("job_classification.csv, when an output directory was "
                              "pre-populated. Omit for a fresh run: nothing is reused, "
                              "so every planned job is newly docked."))
    parser.add_argument("--out", required=True, help="run_manifest.csv to write")
    parser.add_argument(
        "--redock-out",
        default=None,
        help="redock_queue.csv to write. Omit to skip; the manifest still marks the rows.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "campaign run directory. Optional: it only places this step's own log "
            "and scratch state at <run-dir>/steps/build_run_manifest/. The manifests "
            "keep their explicit --out/--redock-out paths."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="debug-level logging (same as --log-level DEBUG)")
    add_runtime_arguments(parser, STEP_NAME)
    return parser


def _assemble(args: argparse.Namespace) -> int:
    """The stage itself. ``main()`` runs this inside :func:`step`."""
    try:
        positives = schemas.read_checked(args.positives, "positives_planned")
        drawn = schemas.read_checked(args.drawn, "negatives_drawn")
        if args.job_classification:
            job_classification = schemas.read_checked(
                args.job_classification, "job_classification")
        else:
            job_classification = schemas.empty_frame("job_classification")
            LOG.info("no --job-classification: every planned job is newly docked")
    except (InvariantViolation, OSError, ValueError) as exc:
        LOG.error("%s", exc)
        return 2

    try:
        manifest, redock = build_manifest(positives, drawn, job_classification)
    except InvariantViolation as exc:
        LOG.error("%s", exc)
        return 3

    schemas.write_checked(manifest, args.out, "run_manifest")
    if args.redock_out:
        schemas.write_checked(redock, args.redock_out, "redock_queue")

    n_salv = int((manifest["provenance"] == "reused_from").sum()) if not manifest.empty else 0
    n_new = len(manifest) - n_salv
    LOG.info("run_manifest: %d jobs (%d reused, %d to dock)", len(manifest), n_salv, n_new)
    if not manifest.empty:
        for sample_type, group in manifest.groupby("sample_type"):
            salv = int((group["provenance"] == "reused_from").sum())
            LOG.info(
                "  %-9s %6d  (%d reused, %d new)",
                sample_type, len(group), salv, len(group) - salv,
            )
    LOG.info("wrote %s", args.out)
    if args.redock_out:
        LOG.info("redock_queue: %d jobs -> %s", len(redock), args.redock_out)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose:
        # -v predates --log-level and existing command lines still pass it.
        args.log_level = "DEBUG"
    resolve_out_dir(args, STEP_NAME, args.run_dir or infer_run_dir(args.out))
    try:
        with step(STEP_NAME, args):
            code = _assemble(args)
            if code:
                # step() reports a non-zero SystemExit as a failure (and notifies
                # about it); main() still owes its caller an int, so catch it back.
                raise SystemExit(code)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
