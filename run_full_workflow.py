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
4. **It planned jobs and then docked them without building the receptors.** The
   four structure-preparation scripts were left to be run by hand between the
   manifest and the docking phase. Run the chain straight through and Phase 7
   started with no receptor for any job in the queue.

This runner drives the new stages, gates on the identity index, and makes every
choice explicit. Each phase is skippable, and every phase verifies its own
prerequisite rather than degrading when one is missing.

Phase groups
------------

``--skip`` and ``--only`` take a phase key, or one of the group names in
:data:`PHASE_GROUPS`. ``--only structures`` is the four structure-preparation
phases; ``--skip structures`` is everything else.

Where everything is written
---------------------------

The runner has its own log and gives every phase one too, so a failure at hour 40
is a file you can open rather than terminal scrollback you no longer have::

    <run-dir>/logs/workflow.log                 this runner: plan, phase start,
                                                exit code, elapsed time
    <run-dir>/steps/<phase>/<script>.log        the phase script's own log
    <run-dir>/steps/<phase>/<phase>.console.log everything the phase printed

Artifacts do not move: ``manifests/``, ``structures/``, ``docking_results/`` and
the analysis parquet keep their paths, because ``batch_dock.py`` resolves each
mapping as ``<run-dir>/manifests/<basename>``.

Notifications
-------------

``--notify`` sends Pushover messages for the workflow: one at the start, one per
failed phase naming the phase and its exit code, and one at the end with the
total elapsed time and a per-phase summary. Credentials come from
``PUSHOVER_TOKEN`` / ``PUSHOVER_USER_KEY`` in the environment only, and a
``--notify`` run with neither exits 2 *before* the first phase starts.
``--notify-steps`` additionally passes ``--notify`` down to each phase script;
without it a 12-phase run sends notifications about the workflow, not twelve
separate conversations.

Usage
-----

::

    python3 run_full_workflow.py --run-dir $NEW \\
        --idmapping HUMAN_9606_idmapping.dat.gz \\
        --proteome  UP000005640_9606.fasta.gz \\
        --edges     1_drug_to_protein.tsv \\
        --tarballs  /opt/data/cavity_space/af_strong_cavities_data_all \\
        --ligands   $NEW/ligands \\
        --mode      balanced

    python3 run_full_workflow.py --run-dir $NEW --dry-run     # print the plan only
    python3 run_full_workflow.py --run-dir $NEW --notify      # tell me when it ends
"""

# Phase 6 is four scripts rather than one because each writes a distinct mapping
# CSV that batch_dock.py resolves independently (receptor priority: fixed >
# alphafold > cavity, PDBQT substituted when present). Collapsing them into a
# single phase would hide which mapping is missing when one step fails.

from __future__ import annotations

import argparse
import collections
import logging
import os
import subprocess
import sys
import time

from dock_selectivity.runtime import (
    LOGS_DIRNAME,
    WORKFLOW_LOG_BASENAME,
    add_runtime_arguments,
    resolve_notify_events,
    step,
    step_out_dir,
)

#: The step name the shared runtime knows this program by: it names the logger,
#: the log file (``<run-dir>/logs/workflow.log``) and the notification title.
STEP_NAME = "workflow"

LOG = logging.getLogger(STEP_NAME)

REPO = os.path.dirname(os.path.abspath(__file__))

#: Docking modes. Explicit -- never inferred from which files happen to exist.
MODES = ("positives-only", "balanced")

#: Names --skip/--only accept in place of a single phase key.
PHASE_GROUPS = {
    "structures": ("cavity-structures", "alphafold", "fix", "pdbqt"),
    "planning": ("identity", "interactions", "cavities", "positives",
                 "negatives", "manifest"),
}

#: Default AlphaFold archive, matching extract_alphafold_models.py's own default.
DEFAULT_AF_ARCHIVE = "/opt/data/alphafold2/UP000005640_9606_HUMAN_v4.tar"

#: Phases whose script means *the artifact destination* by ``--out-dir``.
#:
#: Everywhere else in the repo ``--out-dir`` is the shared runtime's flag: the
#: step's own folder for its log. The four structure scripts owned that spelling
#: first, and it is where the receptors land -- this runner already points it at
#: ``<run-dir>/structures/{cavities,alphafold,fixed,pdbqt}``, and ``extract_cavities
#: --manifest-only`` writes it into the manifest's ``Pocket_PDB`` column. Handing
#: them a step directory there would silently move the structures. They take
#: ``--log-file`` instead, which the same runtime declares and which wins over
#: ``--out-dir`` for the log.
ARTIFACT_OUT_DIR_PHASES = frozenset(
    {"cavities", "cavity-structures", "alphafold", "fix", "pdbqt"})

#: Lines of a failed phase's captured output echoed into the workflow log.
#: Capturing output must not mean hiding it: a failure has to stay diagnosable
#: from the console the operator is already looking at.
CONSOLE_TAIL_LINES = 40


def phase_step_dir(run_dir, key):
    """``<run-dir>/steps/<phase-key>`` -- one phase's own folder.

    Keyed on the *phase* rather than the script because two phases run
    ``extract_cavities.py`` (the manifest, then the extraction) and their logs
    must not land in one file.
    """
    return step_out_dir(run_dir, key)


def phase_console_log(run_dir, key):
    """Where a phase's stdout and stderr are captured."""
    return os.path.join(phase_step_dir(run_dir, key), key + ".console.log")


def runtime_flags(key, args):
    """The shared-runtime flags this runner appends to one phase's command line.

    Every phase script now takes the runtime flags, so each gets its own
    directory under ``<run-dir>/steps/``. ``--notify`` is forwarded only under
    ``--notify-steps``: the workflow reporting on itself is one conversation,
    twelve scripts reporting on themselves is twelve.
    """
    directory = phase_step_dir(args.run_dir, key)
    if key in ARTIFACT_OUT_DIR_PHASES:
        flags = ["--log-file", os.path.join(directory, key + ".log")]
    else:
        flags = ["--out-dir", directory]
    if getattr(args, "notify_steps", False):
        flags.append("--notify")
        for event in getattr(args, "_notify_on_given", ()) or ():
            flags += ["--notify-on", event]
        if getattr(args, "notify_title", None):
            flags += ["--notify-title", args.notify_title]
    return flags


def expand_phase_keys(keys):
    """Expand group names in a --skip/--only list; pass anything else through.

    An unknown name is deliberately *not* an error here: ``--only nonsense``
    selects no phases and :func:`main` exits 2 saying so, which is the same
    outcome by a clearer route than a parser error listing 12 valid keys.
    """
    out = []
    for key in keys:
        out.extend(PHASE_GROUPS.get(key, (key,)))
    return out


class PhaseFailed(RuntimeError):
    """A phase exited non-zero and the pipeline cannot honestly continue.

    ``returncode`` carries the child's exit code when there was one -- it is
    ``None`` for a missing prerequisite or a phase that exited 0 without writing
    its output -- so the failure notification can name it.
    """

    def __init__(self, message, returncode=None):
        super().__init__(message)
        self.returncode = returncode


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
    pool = j(manifests, "negatives_pool.csv")
    drawn = j(manifests, "negatives_drawn.csv")
    manifest = j(manifests, "run_manifest.csv")
    queue = j(manifests, "redock_queue.csv")

    structures = j(run, "structures")
    cavity_mapping = j(manifests, "cavity_mapping.csv")
    alphafold_mapping = j(manifests, "alphafold_mapping.csv")
    fixed_mapping = j(manifests, "fixed_mapping.csv")
    pdbqt_mapping = j(manifests, "pdbqt_mapping.csv")

    # batch_dock.py resolves each mapping as <run-dir>/manifests/<basename>, so the
    # structure phases must write there and nowhere else.
    threads = ["--threads", str(args.threads)] if args.threads else []

    phases = [
        Phase("identity", "Phase 0 - build the identity index",
              "build_identity_index.py",
              ["--idmapping", args.idmapping or "", "--proteome", args.proteome or "",
               "--out", identity],
              produces=[identity]),
        Phase("interactions", "Phase 1 - resolve the drug-target edges",
              "resolve_interactions.py",
              ["--edges", args.edges or "", "--index", identity,
               "--drugs", args.drugs or "", "--out", interactions,
               "--rejects", j(manifests, "interactions_rejected.csv")],
              produces=[interactions], requires=[identity]),
        Phase("cavities", "Phase 2 - cavity manifest (names only, no unpacking)",
              "extract_cavities.py",
              ["--manifest-only", "--tarballs", args.tarballs or "", "--out", cavity_manifest],
              produces=[cavity_manifest]),
        Phase("positives", "Phase 3 - plan the positive jobs",
              "plan_positives.py",
              ["--interactions", interactions, "--index", identity,
               "--cavities", cavity_manifest, "--drugs", args.drugs or "",
               "--out", positives, "--rejects", j(manifests, "positives_rejected.csv")]
              + (["--ligands", args.ligands] if args.ligands else []),
              produces=[positives], requires=[interactions, cavity_manifest]),
    ]

    sample_args = ["--index", identity, "--interactions", interactions,
                   "--positives", positives, "--out", drawn,
                   "--redock-queue", j(manifests, "negatives_redock_queue.csv")]
    # Exactly one of --pool / --no-pool: the sampler refuses to guess whether
    # prior results exist, and the runner must not guess on its behalf either.
    if args.pool:
        sample_args += ["--pool", args.pool, "--prefer-pool"]
    else:
        sample_args += ["--no-pool"]
    if args.mode != "positives-only":
        phases.append(
            # The stage the old runner never called at all.
            Phase("negatives", "Phase 4 - balanced negative sampling",
                  "sample_negatives.py", sample_args,
                  produces=[drawn], requires=[positives]))

    phases.append(
        Phase("manifest", "Phase 5 - assemble the run manifest and job queue",
              "build_run_manifest.py",
              ["--positives", positives,
               "--drawn", drawn if args.mode != "positives-only" else positives,
               "--out", manifest, "--redock-out", queue]
              + (["--job-classification", args.job_classification]
                 if args.job_classification else []),
              produces=[manifest, queue], requires=[positives]))

    phases.extend([
        Phase("cavity-structures", "Phase 6a - unpack the cavities the queue names",
              "extract_cavities.py",
              ["--manifest", cavity_manifest, "--required", queue,
               "--tarballs", args.tarballs or "",
               "--out-dir", j(structures, "cavities"), "--out", cavity_mapping],
              produces=[cavity_mapping], requires=[cavity_manifest, queue]),
        Phase("alphafold", "Phase 6b - extract the AlphaFold models the queue names",
              "extract_alphafold_models.py",
              ["--required", queue, "--archive", args.af_archive,
               "--out-dir", j(structures, "alphafold"), "--out", alphafold_mapping]
              + threads,
              produces=[alphafold_mapping], requires=[queue]),
        Phase("fix", "Phase 6c - add hydrogens to the receptors",
              "fix_required_pdbs.py",
              ["--required", queue, "--alphafold-mapping", alphafold_mapping,
               "--out-dir", j(structures, "fixed"), "--out", fixed_mapping]
              + threads,
              produces=[fixed_mapping], requires=[queue, alphafold_mapping]),
        Phase("pdbqt", "Phase 6d - convert the receptors to PDBQT",
              "convert_pdb_to_pdbqt.py",
              ["--input-mapping", fixed_mapping, "--required", queue,
               "--out-dir", j(structures, "pdbqt"), "--out", pdbqt_mapping]
              + threads,
              produces=[pdbqt_mapping], requires=[queue, fixed_mapping]),
    ])

    phases.append(
        # Requires the receptors, not just the queue: docking a job list with no
        # structures behind it is the failure this phase group was added to prevent.
        Phase("dock", "Phase 7 - dock the queue (skips jobs already complete in the output dir)",
              "batch_dock.py",
              ["--queue", queue, "--run-dir", run,
               "--complex-pdb", args.complex_pdb]
              + (["--ligands", args.ligands] if args.ligands else []),
              produces=[], requires=[queue, fixed_mapping, cavity_mapping]))

    phases.append(
        # --manifest turns discovery into a checked step: a parse that finds fewer
        # job directories than the manifest declares is the failure that silently
        # dropped most of the last campaign, and without this flag it is only a
        # warning.
        Phase("parse", "Phase 8 - parse results into the analysis table",
              "parse_prepare.py", ["--run-dir", run, "--manifest", manifest],
              produces=[], requires=[manifest], optional=True))

    # Every phase gets its own folder under <run-dir>/steps/, so a phase's log is
    # never in the invocation cwd and never beside the artifacts another stage
    # enumerates.
    for phase in phases:
        phase.args += runtime_flags(phase.key, args)

    return phases


def tail(path, lines=CONSOLE_TAIL_LINES):
    """The last ``lines`` lines of a captured-output file, for an error message."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            kept = collections.deque(handle, maxlen=lines)
    except OSError as exc:
        return "(could not read %s: %s)" % (path, exc)
    return "".join(kept).rstrip() or "(the phase printed nothing)"


def run_phase(phase, dry_run=False, timeout=None, will_exist=(), console_log=None):
    """Run one phase.

    ``will_exist`` names artifacts that earlier phases in this plan produce. In a
    dry run nothing is actually created, so without it the prerequisite check
    stops at phase 2 and the operator never sees the rest of the plan -- which
    defeats the purpose of a dry run.

    ``console_log`` captures the phase's stdout and stderr into a file instead of
    letting twelve scripts interleave their progress bars in one terminal. It is
    a capture, not a muzzle: a failure quotes the tail of that file into the
    error, so the operator still sees what went wrong without going looking. With
    no ``console_log`` the child inherits this process's streams, as before.
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
    if console_log:
        directory = os.path.dirname(console_log)
        if directory:
            os.makedirs(directory, exist_ok=True)
        LOG.info("  output -> %s", console_log)
        with open(console_log, "wb") as sink:
            proc = subprocess.run(command, timeout=timeout, stdout=sink,
                                  stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(command, timeout=timeout)
    elapsed = time.time() - start

    if proc.returncode != 0:
        detail = "\n  last %d line(s) of %s:\n%s" % (
            CONSOLE_TAIL_LINES, console_log, tail(console_log)) if console_log else ""
        raise PhaseFailed("%s exited %d after %.1fs%s"
                          % (phase.title, proc.returncode, elapsed, detail),
                          returncode=proc.returncode)

    absent = [p for p in phase.produces if not os.path.exists(p)]
    if absent:
        raise PhaseFailed(
            "%s exited 0 but did not produce %s. Exit code 0 with no output is how "
            "a silent no-op looks like success." % (phase.title, ", ".join(absent)))

    LOG.info("  ok (%.1fs)%s", elapsed,
             "".join("\n    -> %s" % p for p in phase.produces))
    return 0


#: Where each input lives inside an --inputs-dir. Explicit, so a typo is a
#: missing-file error rather than a stage quietly running on the wrong data.
INPUT_LAYOUT = {
    "edges": "1_drug_to_protein.tsv",
    "drugs": "small_molecule_drug_links.csv",
    "ligands": "ligands",
    "idmapping": os.path.join("uniprot", "HUMAN_9606_idmapping.dat.gz"),
    "proteome": os.path.join("uniprot", "UP000005640_9606.fasta.gz"),
}


#: Last-resort location for the DrugBank vocabulary if neither --drugs nor
#: --inputs-dir supplies one.
DEFAULT_DRUGS = "/opt/data/drugbank/small_molecule_drug_links.csv"


def resolve_inputs(args):
    """Fill unset input paths from ``--inputs-dir``.

    An explicit flag always wins. Anything still unset after this is reported by
    the phase that needs it, rather than being silently defaulted -- the whole
    point of separating inputs from outputs is that you can see which is which.
    """
    if not args.inputs_dir:
        if not args.drugs:
            args.drugs = DEFAULT_DRUGS
        return args
    for attr, relative in INPUT_LAYOUT.items():
        if getattr(args, attr, None):
            continue
        candidate = os.path.join(args.inputs_dir, relative)
        if os.path.exists(candidate):
            setattr(args, attr, candidate)
            LOG.info("input %-10s <- %s", attr, candidate)
        else:
            LOG.warning("input %-10s not found at %s", attr, candidate)
    if not args.drugs:
        args.drugs = DEFAULT_DRUGS
    return args


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True,
                        help="OUTPUT directory: manifests, structures, docking results, analysis")
    parser.add_argument("--inputs-dir", default=None,
                        help=("INPUT directory holding the small reusable inputs "
                              "(edges TSV, drug links, ligands/, uniprot/). Kept separate "
                              "from --run-dir so inputs are shared and versioned while "
                              "outputs are per-run. Individual --edges/--drugs/--ligands/"
                              "--idmapping/--proteome override what is found here."))
    parser.add_argument("--mode", choices=MODES, default="balanced",
                        help=("EXPLICIT docking mode (default: %(default)s). The old "
                              "runner inferred this from whether a file happened to "
                              "exist."))
    parser.add_argument("--idmapping", default=None, help="HUMAN_9606_idmapping.dat.gz")
    parser.add_argument("--proteome", default=None, help="UP000005640_9606.fasta.gz")
    parser.add_argument("--edges", default=None, help="1_drug_to_protein.tsv")
    parser.add_argument("--drugs", default=None,
                        help=("small_molecule_drug_links.csv. Resolved from --inputs-dir "
                              "when unset, else %s" % DEFAULT_DRUGS))
    parser.add_argument("--tarballs", default="/opt/data/cavity_space/af_strong_cavities_data_all",
                        help="folder of CavitySpace cavity tarballs (default: %(default)s)")
    parser.add_argument("--af-archive", default=DEFAULT_AF_ARCHIVE,
                        help="AlphaFold tar archive to extract models from (default: %(default)s)")
    parser.add_argument("--ligands", default=None, help="prepared ligand SDF folder")
    parser.add_argument("--threads", type=int, default=None,
                        help=("worker threads for the structure phases (extraction, "
                              "hydrogen addition, PDBQT conversion). Unset leaves each "
                              "script on its own default."))
    parser.add_argument("--pool", default=None,
                        help=("optional negatives_pool.csv: already-docked "
                              "non-interaction jobs the sampler may draw from before "
                              "queueing new work"))
    parser.add_argument("--job-classification", default=None,
                        help="optional job_classification.csv from bootstrap_output_dir.py")
    parser.add_argument("--complex-pdb", choices=("keep", "delete", "compress"),
                        default="keep",
                        help=("consensus_docker.py computes RMSD from complex_*.pdb, "
                              "so removing them makes RMSD non-recomputable. Keep "
                              "during a run; delete/compress as an archival step "
                              "afterwards. Default: %(default)s"))
    parser.add_argument("--skip", action="append", default=[], metavar="PHASE",
                        help=("phase key to skip (repeatable). Also accepts a group "
                              "name: %s" % ", ".join(sorted(PHASE_GROUPS))))
    parser.add_argument("--only", action="append", default=[], metavar="PHASE",
                        help=("run only these phase keys (repeatable). Also accepts a "
                              "group name: %s" % ", ".join(sorted(PHASE_GROUPS))))
    parser.add_argument("--dry-run", action="store_true",
                        help=("print the plan and exit without running anything. "
                              "Writes nothing at all -- not even this runner's log"))
    parser.add_argument("--timeout", type=float, default=None,
                        help="per-phase timeout in seconds")
    parser.add_argument("--notify-steps", action="store_true",
                        help=("also pass --notify to every phase script, so each "
                              "one notifies for itself. Off by default: a 12-phase "
                              "run would otherwise send 12 sets of notifications "
                              "on top of the workflow's own"))
    # Last, so the shared flags read as a block in --help. This runner's log is
    # <run-dir>/logs/workflow.log rather than a steps/ directory; main() settles
    # that once --run-dir is known.
    add_runtime_arguments(parser, STEP_NAME)
    return parser


def resolve_runtime_paths(args):
    """Settle where this runner's own log goes, and return the log path.

    The workflow is the one step whose log is not under ``steps/``: it belongs to
    the whole run, so it goes to ``<run-dir>/logs/workflow.log``. An explicit
    ``--out-dir`` or ``--log-file`` still wins.

    A dry run is pointed at the null device instead. Printing a plan must leave
    nothing behind -- including in a ``--run-dir`` that does not exist yet, which
    is exactly when an operator asks for the plan.
    """
    if args.dry_run:
        args.out_dir = os.path.dirname(os.devnull) or "."
        args.log_file = os.devnull
        return args.log_file

    default_out_dir = getattr(args, "_runtime_out_dir_default", None)
    if not args.out_dir or args.out_dir == default_out_dir:
        args.out_dir = os.path.join(os.path.abspath(args.run_dir), LOGS_DIRNAME)
    args.out_dir = os.path.abspath(args.out_dir)
    if not args.log_file:
        args.log_file = os.path.join(args.out_dir, WORKFLOW_LOG_BASENAME)
    return args.log_file


def notify_once(ctx, event, message, priority=0):
    """Send one notification and stop :func:`step` sending its generic twin.

    ``step()`` posts a terse "workflow: finished in 40h" / "workflow: FAILED" of
    its own when the block ends. The workflow's message names the phase, the exit
    code and every phase's elapsed time, so the generic one would be a second
    buzz saying less.
    """
    sent = ctx.notify(event, message, priority=priority)
    ctx.notifier.events = tuple(e for e in ctx.notifier.events if e != event)
    return sent


def select_phases(args):
    """The phase list after ``--only`` and ``--skip``, in run order."""
    phases = build_phases(args)
    if args.only:
        selected = set(expand_phase_keys(args.only))
        phases = [p for p in phases if p.key in selected]
    if args.skip:
        skipped = set(expand_phase_keys(args.skip))
        phases = [p for p in phases if p.key not in skipped]
    return phases


def run_workflow(args, ctx=None):
    """Run the selected phases in order; returns the process exit code.

    ``ctx`` is the :class:`~dock_selectivity.runtime.StepContext` the
    notifications go through. Without one (a dry run) nothing is notified.
    """
    resolve_inputs(args)
    phases = select_phases(args)
    if not phases:
        LOG.error("no phases selected")
        return 2

    LOG.info("run directory : %s", args.run_dir)
    LOG.info("mode          : %s", args.mode)
    LOG.info("phases        : %s", ", ".join(p.key for p in phases))
    if args.dry_run:
        LOG.info("--dry-run: printing commands only, running nothing")

    if ctx is not None and ctx.notifier.enabled:
        # The resolved-configuration block above prints notify_on without 'start'
        # (see main()), so state the events that are actually in force.
        LOG.info("notifications : %s", ", ".join(ctx.notifier.events))
    if ctx is not None:
        notify_once(ctx, "start", "workflow: %d phase(s) started in %s\n%s"
                    % (len(phases), args.run_dir, ", ".join(p.key for p in phases)))

    started = time.time()
    planned_outputs = []
    #: (phase key, outcome, seconds) for the completion notification.
    outcomes = []
    for phase in phases:
        phase_started = time.time()
        console_log = None if args.dry_run else phase_console_log(args.run_dir, phase.key)
        try:
            run_phase(phase, dry_run=args.dry_run, timeout=args.timeout,
                      will_exist=planned_outputs if args.dry_run else (),
                      console_log=console_log)
            planned_outputs.extend(phase.produces)
            outcomes.append((phase.key, "ok", time.time() - phase_started))
        except (PhaseFailed, subprocess.TimeoutExpired) as exc:
            if isinstance(exc, subprocess.TimeoutExpired):
                message = "%s timed out after %.0fs" % (phase.title, args.timeout or 0)
                code = "timeout"
                if console_log:
                    message += "\n  last %d line(s) of %s:\n%s" % (
                        CONSOLE_TAIL_LINES, console_log, tail(console_log))
            else:
                message = str(exc)
                # No exit code for a missing prerequisite or a phase that exited
                # 0 without writing its output -- the message says which.
                code = exc.returncode if exc.returncode is not None else "n/a"
            if phase.optional and not isinstance(exc, subprocess.TimeoutExpired):
                LOG.warning("%s (optional) -- continuing", message)
                outcomes.append((phase.key, "failed (optional)",
                                 time.time() - phase_started))
                continue
            LOG.error("%s", message)
            LOG.error("STOPPING. Later phases would consume an artifact this phase "
                      "did not produce.")
            outcomes.append((phase.key, "FAILED", time.time() - phase_started))
            if ctx is not None:
                notify_once(
                    ctx, "failure",
                    "workflow FAILED in phase '%s' (exit %s) after %s\nrun-dir: %s\n"
                    "log: %s" % (phase.key, code, _elapsed(time.time() - started),
                                 args.run_dir,
                                 console_log or "<not captured>"),
                    priority=1)
            return 1

    total = time.time() - started
    LOG.info("all %d phase(s) completed in %.1fs", len(phases), total)
    if ctx is not None:
        notify_once(ctx, "success",
                    "workflow: %d phase(s) finished in %s\nrun-dir: %s\n%s"
                    % (len(phases), _elapsed(total), args.run_dir,
                       "\n".join("  %-17s %-18s %s" % (key, state, _elapsed(seconds))
                                 for key, state, seconds in outcomes)))
    return 0


def _elapsed(seconds):
    """Human-readable duration; a three-day run is not readable in seconds."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return "%dh%02dm" % (hours, minutes)
    if minutes:
        return "%dm%02ds" % (minutes, secs)
    return "%.1fs" % seconds


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Remembered before the default below overwrites it: only events the operator
    # asked for are forwarded to the phases under --notify-steps.
    args._notify_on_given = list(args.notify_on or [])
    # The runtime's default is failure-only, which is right for one step. The
    # workflow is the thing an operator watches: it reports that it started, that
    # it finished, and what broke. An explicit --notify-on still wins.
    if args.notify and not args.notify_on:
        args.notify_on = ["all"]
    wanted_events = resolve_notify_events(args.notify_on) if args.notify else ()
    # step() sends its own terse "started" the instant the block is entered --
    # before the body can say which phases are about to run. Withhold 'start'
    # from it and restore the operator's real set inside, so the one start
    # message that goes out is the one naming the run directory and the plan.
    args.notify_on = [e for e in wanted_events if e != "start"] or None
    resolve_runtime_paths(args)

    try:
        with step(STEP_NAME, args) as ctx:
            ctx.notifier.events = wanted_events
            # A dry run gets no phase-level notifications: nothing ran, so there
            # is nothing to report about. --notify is still validated on the way
            # in, which makes --dry-run --notify a way to check the credentials
            # before committing to three days.
            code = run_workflow(args, None if args.dry_run else ctx)
            if code:
                # step() reads a plain return as success, so an exit code has to
                # be raised or a failed campaign is announced as a finished one.
                raise SystemExit(code)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
