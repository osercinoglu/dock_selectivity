#!/usr/bin/env python3
"""Run the corrected rebuild pipeline, REBUILD_PLAN.md Phases 0-8.

What this replaces
------------------

The previous runner invoked seven scripts and had three structural problems, all
of which this version fixes:

1. **It never called the negative-sampling stage.** A repo-wide grep found
   ``generate_negative_samples.py`` referenced only in two ``batch_dock.py``
   comments. So the automated workflow could not produce the balanced dataset the
   study depends on -- whether negatives existed at all depended on someone having
   run that script by hand.
2. **The docking mode was inferred from file presence.** ``batch_dock.py`` silently
   switched between "dock the listed triples" and "dock every cavity of every known
   target" depending on whether ``required_structures_with_negatives.csv`` happened
   to exist. Behaviour determined by leftovers is not behaviour anyone chose.
3. **It started with the corrupted identity stage.** Step 1 was
   ``prepare_uniprot_mapping.py``, whose free-text search produced the mapping in
   which 57.73% of rows name a different protein.

This runner drives the new stages, gates on the identity index, and makes every
choice explicit. Each phase is skippable, and every phase verifies its own
prerequisite rather than degrading when one is missing.

Usage
-----

::

    python3 run_full_workflow.py --run-dir $NEW \\
        --idmapping HUMAN_9606_idmapping.dat.gz \\
        --proteome  UP000005640_9606.fasta.gz \\
        --edges     1_drug_to_protein.tsv \\
        --tarballs  /opt/data/cavity_space/af_strong_cavities_data_all \\
        --ligands   $NEW/ligands \\
        --mode      salvage-and-redock

    python3 run_full_workflow.py --run-dir $NEW --dry-run     # print the plan only
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time

LOG = logging.getLogger("workflow")

REPO = os.path.dirname(os.path.abspath(__file__))

#: Docking modes. Explicit -- never inferred from which files happen to exist.
MODES = ("positives-only", "salvage-and-redock", "full-redock")


class PhaseFailed(RuntimeError):
    """A phase exited non-zero and the pipeline cannot honestly continue."""


class Phase:
    """One runnable step, with the prerequisite it refuses to run without."""

    def __init__(self, key, title, script, args, produces, requires=(), optional=False):
        self.key = key
        self.title = title
        self.script = script
        self.args = list(args)
        self.produces = list(produces)
        self.requires = list(requires)
        self.optional = optional

    def command(self):
        return [sys.executable, os.path.join(REPO, self.script)] + self.args

    def missing_prerequisites(self):
        return [p for p in self.requires if not os.path.exists(p)]


def build_phases(args):
    """Assemble the phase list for the requested mode."""
    run = args.run_dir
    j = os.path.join
    identity = j(run, "identity", "identity_index.parquet")
    manifests = j(run, "manifests")

    interactions = j(manifests, "interactions_resolved.csv")
    cavity_manifest = j(manifests, "cavity_manifest.csv")
    positives = j(manifests, "positives_planned.csv")
    salvage_map = j(manifests, "salvage_map.csv")
    pool = j(manifests, "negatives_pool.csv")
    drawn = j(manifests, "negatives_drawn.csv")
    manifest = j(manifests, "run_manifest.csv")
    queue = j(manifests, "redock_queue.csv")

    phases = [
        Phase("identity", "Phase 0 - build the identity index",
              "build_identity_index.py",
              ["--idmapping", args.idmapping or "", "--proteome", args.proteome or "",
               "--out", identity],
              produces=[identity]),
        Phase("interactions", "Phase 1 - resolve the drug-target edges",
              "resolve_interactions.py",
              ["--edges", args.edges or "", "--index", identity,
               "--drugs", args.drugs, "--out", interactions,
               "--rejects", j(manifests, "interactions_rejected.csv")],
              produces=[interactions], requires=[identity]),
        Phase("cavities", "Phase 2 - cavity manifest (names only, no unpacking)",
              "extract_cavities.py",
              ["--manifest-only", "--tarballs", args.tarballs or "", "--out", cavity_manifest],
              produces=[cavity_manifest]),
        Phase("positives", "Phase 3 - plan the positive jobs",
              "plan_positives.py",
              ["--interactions", interactions, "--index", identity,
               "--cavities", cavity_manifest, "--drugs", args.drugs,
               "--out", positives, "--rejects", j(manifests, "positives_rejected.csv")]
              + (["--ligands", args.ligands] if args.ligands else []),
              produces=[positives], requires=[interactions, cavity_manifest]),
    ]

    if args.mode in ("salvage-and-redock",):
        phases.append(
            Phase("salvage", "Phase 4 - classify the old campaign (read-only)",
                  "salvage_scan.py",
                  ["--old-results", args.old_results or "", "--index", identity,
                   "--interactions", interactions,
                   "--out-map", salvage_map, "--out-pool", pool],
                  produces=[salvage_map, pool], requires=[interactions]))

    sample_args = ["--index", identity, "--interactions", interactions,
                   "--positives", positives, "--out", drawn,
                   "--redock-queue", j(manifests, "negatives_redock_queue.csv")]
    if args.mode == "salvage-and-redock":
        sample_args += ["--pool", pool, "--prefer-pool"]
    if args.mode != "positives-only":
        phases.append(
            # The stage the old runner never called at all.
            Phase("negatives", "Phase 5 - balanced negative sampling",
                  "sample_negatives.py", sample_args,
                  produces=[drawn], requires=[positives]))

    phases.append(
        Phase("manifest", "Phase 6 - assemble the run manifest and redock queue",
              "build_run_manifest.py",
              ["--positives", positives,
               "--drawn", drawn if args.mode != "positives-only" else positives,
               "--salvage-map", salvage_map,
               "--out", manifest, "--redock-out", queue],
              produces=[manifest, queue], requires=[positives]))

    phases.append(
        Phase("dock", "Phase 7 - dock the queue",
              "batch_dock.py",
              ["--queue", queue, "--run-dir", run,
               "--complex-pdb", args.complex_pdb]
              + (["--ligands", args.ligands] if args.ligands else []),
              produces=[], requires=[queue]))

    phases.append(
        Phase("parse", "Phase 8 - parse results into the analysis table",
              "parse_prepare.py", ["--run-dir", run],
              produces=[], requires=[manifest], optional=True))

    return phases


def run_phase(phase, dry_run=False, timeout=None, will_exist=()):
    """Run one phase.

    ``will_exist`` names artifacts that earlier phases in this plan produce. In a
    dry run nothing is actually created, so without it the prerequisite check
    stops at phase 2 and the operator never sees the rest of the plan -- which
    defeats the purpose of a dry run.
    """
    missing = [p for p in phase.missing_prerequisites() if p not in set(will_exist)]
    if missing:
        raise PhaseFailed(
            "%s cannot run: missing prerequisite(s) %s. A missing prerequisite is "
            "an error, never a fallback -- run the earlier phase, or pass its "
            "--skip flag only if you know the artifact exists elsewhere."
            % (phase.title, ", ".join(missing)))

    command = phase.command()
    LOG.info("%s", phase.title)
    LOG.info("  $ %s", " ".join(command))
    if dry_run:
        return 0

    start = time.time()
    proc = subprocess.run(command, timeout=timeout)
    elapsed = time.time() - start

    if proc.returncode != 0:
        raise PhaseFailed("%s exited %d after %.1fs"
                          % (phase.title, proc.returncode, elapsed))

    absent = [p for p in phase.produces if not os.path.exists(p)]
    if absent:
        raise PhaseFailed(
            "%s exited 0 but did not produce %s. Exit code 0 with no output is how "
            "a silent no-op looks like success." % (phase.title, ", ".join(absent)))

    LOG.info("  ok (%.1fs)%s", elapsed,
             "".join("\n    -> %s" % p for p in phase.produces))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="the new experiment directory")
    parser.add_argument("--mode", choices=MODES, default="salvage-and-redock",
                        help=("EXPLICIT docking mode (default: %(default)s). The old "
                              "runner inferred this from whether a file happened to "
                              "exist."))
    parser.add_argument("--idmapping", default=None, help="HUMAN_9606_idmapping.dat.gz")
    parser.add_argument("--proteome", default=None, help="UP000005640_9606.fasta.gz")
    parser.add_argument("--edges", default=None, help="1_drug_to_protein.tsv")
    parser.add_argument("--drugs", default="/opt/data/drugbank/small_molecule_drug_links.csv")
    parser.add_argument("--tarballs", default="/opt/data/cavity_space/af_strong_cavities_data_all")
    parser.add_argument("--ligands", default=None, help="prepared ligand SDF folder")
    parser.add_argument("--old-results", default=None,
                        help="the OLD consensus_docking_results tree (read-only)")
    parser.add_argument("--complex-pdb", choices=("keep", "delete", "compress"),
                        default="delete")
    parser.add_argument("--skip", action="append", default=[],
                        metavar="PHASE", help="phase key to skip (repeatable)")
    parser.add_argument("--only", action="append", default=[],
                        metavar="PHASE", help="run only these phase keys (repeatable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without running anything")
    parser.add_argument("--timeout", type=float, default=None,
                        help="per-phase timeout in seconds")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    phases = build_phases(args)
    if args.only:
        phases = [p for p in phases if p.key in set(args.only)]
    if args.skip:
        phases = [p for p in phases if p.key not in set(args.skip)]

    if not phases:
        LOG.error("no phases selected")
        return 2

    LOG.info("run directory : %s", args.run_dir)
    LOG.info("mode          : %s", args.mode)
    LOG.info("phases        : %s", ", ".join(p.key for p in phases))
    if args.dry_run:
        LOG.info("--dry-run: printing commands only, running nothing")

    started = time.time()
    planned_outputs = []
    for phase in phases:
        try:
            run_phase(phase, dry_run=args.dry_run, timeout=args.timeout,
                      will_exist=planned_outputs if args.dry_run else ())
            planned_outputs.extend(phase.produces)
        except PhaseFailed as exc:
            if phase.optional:
                LOG.warning("%s (optional) -- continuing", exc)
                continue
            LOG.error("%s", exc)
            LOG.error("STOPPING. Later phases would consume an artifact this phase "
                      "did not produce.")
            return 1
        except subprocess.TimeoutExpired:
            LOG.error("%s timed out after %.0fs", phase.title, args.timeout or 0)
            return 1

    LOG.info("all %d phase(s) completed in %.1fs", len(phases), time.time() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
