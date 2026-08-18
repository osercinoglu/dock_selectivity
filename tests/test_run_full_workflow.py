"""Tests for the rebuilt workflow runner.

Three structural defects in the previous runner, each with a test here:

* it never invoked the negative-sampling stage at all, so the automated workflow
  could not produce the balanced dataset the study depends on;
* the docking mode was inferred from whether a file happened to exist, so
  behaviour was decided by leftovers rather than by anyone;
* it began with ``prepare_uniprot_mapping.py``, the corrupted identity stage.

Plus the rule this whole session has been about: a missing prerequisite is an
error, never a fallback -- and a phase that exits 0 without producing its output
is a silent no-op wearing a success code.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.parse
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import run_full_workflow as wf  # noqa: E402
from dock_selectivity.runtime import Pushover  # noqa: E402


def plan(**overrides):
    argv = ["--run-dir", "/tmp/does-not-exist-run"]
    for key, value in overrides.items():
        argv += ["--" + key.replace("_", "-"), value]
    return wf.build_phases(wf.build_parser().parse_args(argv))


class TestPhaseComposition(unittest.TestCase):
    def test_identity_is_phase_zero(self):
        """Nothing may run before the index; everything downstream pins it."""
        self.assertEqual(plan()[0].key, "identity")
        self.assertEqual(plan()[0].script, "build_identity_index.py")

    def test_negative_sampling_is_actually_invoked(self):
        """The old runner never called it -- verified by repo-wide grep."""
        keys = [p.key for p in plan()]
        self.assertIn("negatives", keys)
        scripts = [p.script for p in plan()]
        self.assertIn("sample_negatives.py", scripts)

    def test_positives_only_mode_omits_negative_sampling(self):
        keys = [p.key for p in plan(mode="positives-only")]
        self.assertNotIn("negatives", keys)

    def test_there_is_no_reuse_phase(self):
        """Populating the output directory is a one-time bootstrap, not a phase.

        Every stage already skips work that is already complete on disk (verified:
        an evidence-only job directory with no complex_*.pdb reports complete on
        all four tools and on final_results). So a pre-populated output directory
        needs no special handling -- you just run the pipeline.
        """
        for mode in wf.MODES:
            with self.subTest(mode=mode):
                self.assertNotIn("reuse", [p.key for p in plan(mode=mode)])

    def test_the_retired_script_is_never_invoked(self):
        for mode in wf.MODES:
            with self.subTest(mode=mode):
                self.assertNotIn("prepare_uniprot_mapping.py",
                                 [p.script for p in plan(mode=mode)])

    def test_every_phase_names_a_script_that_exists(self):
        for mode in wf.MODES:
            for phase in plan(mode=mode):
                with self.subTest(mode=mode, phase=phase.key):
                    self.assertTrue(
                        os.path.exists(os.path.join(REPO, phase.script)),
                        "%s invokes %s which does not exist" % (phase.key, phase.script))

    def test_mode_is_explicit_not_inferred(self):
        self.assertEqual(set(wf.MODES), {"positives-only", "balanced"})
        parser = wf.build_parser()
        with self.assertRaises(SystemExit):  # an unknown mode is rejected outright
            parser.parse_args(["--run-dir", "/tmp/x", "--mode", "whatever-is-lying-around"])


class TestStructurePhases(unittest.TestCase):
    """Phase 6 builds the receptors.

    Before it existed the runner planned a queue and then handed it to
    ``batch_dock.py`` with nothing to dock into: the four structure-preparation
    scripts were documented in the pilot section of SETUP.md and left to be run by
    hand. A chain that plans work it cannot execute is the same class of defect as
    a phase exiting 0 without producing its output.
    """

    STRUCTURE_KEYS = ("cavity-structures", "alphafold", "fix", "pdbqt")

    def test_the_four_structure_scripts_are_invoked(self):
        scripts = [p.script for p in plan()]
        for script in ("extract_cavities.py", "extract_alphafold_models.py",
                       "fix_required_pdbs.py", "convert_pdb_to_pdbqt.py"):
            with self.subTest(script=script):
                self.assertIn(script, scripts)

    def test_structures_come_after_the_queue_and_before_docking(self):
        keys = [p.key for p in plan()]
        for key in self.STRUCTURE_KEYS:
            with self.subTest(phase=key):
                self.assertLess(keys.index("manifest"), keys.index(key))
                self.assertLess(keys.index(key), keys.index("dock"))

    def test_structure_phases_are_ordered_by_their_dependencies(self):
        """Hydrogens need the extracted model; PDBQT needs the hydrogenated one."""
        keys = [p.key for p in plan()]
        self.assertLess(keys.index("alphafold"), keys.index("fix"))
        self.assertLess(keys.index("fix"), keys.index("pdbqt"))

    def test_mappings_land_where_batch_dock_looks_for_them(self):
        """batch_dock resolves each mapping as <run-dir>/manifests/<basename>.

        A structure phase writing anywhere else produces a run that plans, preps,
        and then silently docks against whatever older mapping is on the path.
        """
        import batch_dock

        expected = {
            "cavity-structures": batch_dock.CAVITY_MAPPING_CSV,
            "alphafold": batch_dock.ALPHAFOLD_MAPPING_CSV,
            "fix": batch_dock.FIXED_MAPPING_CSV,
            "pdbqt": batch_dock.PDBQT_MAPPING_CSV,
        }
        by_key = {p.key: p for p in plan()}
        for key, mapping_name in expected.items():
            with self.subTest(phase=key):
                produced = by_key[key].produces
                self.assertEqual(len(produced), 1)
                self.assertEqual(os.path.basename(produced[0]),
                                 os.path.basename(mapping_name))
                self.assertEqual(os.path.basename(os.path.dirname(produced[0])),
                                 "manifests")

    def test_every_structure_phase_is_restricted_to_the_queue(self):
        """--required is what keeps a run from materialising all 38,709 cavities."""
        for phase in plan():
            if phase.key not in self.STRUCTURE_KEYS:
                continue
            with self.subTest(phase=phase.key):
                self.assertIn("--required", phase.args)
                target = phase.args[phase.args.index("--required") + 1]
                self.assertTrue(target.endswith("redock_queue.csv"), target)

    def test_docking_requires_the_receptors_not_just_the_queue(self):
        dock = [p for p in plan() if p.key == "dock"][0]
        basenames = {os.path.basename(r) for r in dock.requires}
        self.assertIn("fixed_mapping.csv", basenames)
        self.assertIn("redock_queue.csv", basenames)

    def test_threads_is_passed_only_when_set(self):
        by_key = {p.key: p for p in plan()}
        self.assertNotIn("--threads", by_key["fix"].args)
        by_key = {p.key: p for p in plan(threads="8")}
        self.assertEqual(by_key["fix"].args[by_key["fix"].args.index("--threads") + 1], "8")
        # extract_cavities.py has no --threads; passing one would be a parser error.
        self.assertNotIn("--threads", by_key["cavity-structures"].args)


class TestPocketQualityPhase(unittest.TestCase):
    """Phase 2b scores every cavity; filtering on the score is opt-in.

    The scoring pass is unconditional because it is cheap (~6 CPU-minutes for all
    38,709 archives) and because it is the only place a pocket's model confidence
    is ever measured. What is *not* unconditional is acting on it: a filter that
    quietly changes the size of a scientific dataset is the defect class this
    repository was rebuilt to eliminate, so ``--min-plddt-fraction`` is the single
    switch that turns it on, and without it phases 3 and 4 are handed no quality
    flags at all.
    """

    QUALITY_FLAGS = ("--cavity-quality", "--plddt-min", "--min-plddt-fraction")

    def test_the_phase_exists_and_runs_the_scoring_script(self):
        by_key = {p.key: p for p in plan()}
        self.assertIn("quality", by_key)
        self.assertEqual(by_key["quality"].script, "score_cavity_quality.py")

    def test_it_sits_between_the_cavity_manifest_and_the_positives(self):
        keys = [p.key for p in plan()]
        self.assertLess(keys.index("cavities"), keys.index("quality"))
        self.assertLess(keys.index("quality"), keys.index("positives"))

    def test_it_is_part_of_the_planning_group(self):
        self.assertIn("quality", wf.PHASE_GROUPS["planning"])
        self.assertIn("quality", wf.expand_phase_keys(["planning"]))

    def test_it_produces_cavity_quality_csv_from_the_cavity_manifest(self):
        phase = {p.key: p for p in plan()}["quality"]
        self.assertEqual([os.path.basename(p) for p in phase.produces],
                         ["cavity_quality.csv"])
        self.assertEqual(os.path.basename(os.path.dirname(phase.produces[0])),
                         "manifests")
        self.assertEqual([os.path.basename(r) for r in phase.requires],
                         ["cavity_manifest.csv"])

    def test_it_runs_even_when_nothing_is_being_filtered(self):
        """The numbers are worth having whether or not this run acts on them."""
        for mode in wf.MODES:
            with self.subTest(mode=mode):
                self.assertIn("quality", [p.key for p in plan(mode=mode)])

    def _flags(self, phase):
        return {flag: phase.args[phase.args.index(flag) + 1]
                for flag in self.QUALITY_FLAGS if flag in phase.args}

    def test_the_thresholds_reach_both_planning_stages(self):
        """One threshold, both classes.

        Filtering the positives and not the negatives would put a positive class
        full of low-confidence pockets against a high-confidence negative class --
        a classifier could then separate them on pLDDT alone.
        """
        by_key = {p.key: p for p in plan(plddt_min="80", min_plddt_fraction="0.75")}
        for key in ("positives", "negatives"):
            with self.subTest(phase=key):
                flags = self._flags(by_key[key])
                self.assertEqual(sorted(flags), sorted(self.QUALITY_FLAGS))
                self.assertEqual(float(flags["--plddt-min"]), 80.0)
                self.assertEqual(float(flags["--min-plddt-fraction"]), 0.75)
                self.assertEqual(os.path.basename(flags["--cavity-quality"]),
                                 "cavity_quality.csv")

    def test_the_file_the_stages_filter_on_is_the_one_phase_2b_wrote(self):
        by_key = {p.key: p for p in plan(min_plddt_fraction="0.8")}
        written = by_key["quality"].produces[0]
        for key in ("positives", "negatives"):
            with self.subTest(phase=key):
                self.assertEqual(self._flags(by_key[key])["--cavity-quality"], written)
                self.assertIn(written, by_key[key].requires)

    def test_the_filtered_command_lines_are_accepted_by_the_real_parsers(self):
        """The generic check only ever parses the unfiltered plan."""
        import importlib

        by_key = {p.key: p for p in plan(plddt_min="80", min_plddt_fraction="0.75")}
        for key in ("quality", "positives", "negatives"):
            phase = by_key[key]
            module = importlib.import_module(phase.script[: -len(".py")])
            builder = (getattr(module, "build_parser", None)
                       or getattr(module, "build_arg_parser"))
            with self.subTest(phase=key):
                try:
                    builder().parse_args(phase.args)
                except SystemExit as exc:
                    self.fail("%s does not accept %s (%s)"
                              % (phase.script, phase.args, exc))

    def test_without_a_fraction_neither_stage_is_given_a_quality_flag(self):
        """No fraction, no filtering -- and no new arguments to today's commands."""
        for overrides in ({}, {"plddt_min": "90"}):
            by_key = {p.key: p for p in plan(**overrides)}
            for key in ("positives", "negatives"):
                with self.subTest(phase=key, overrides=overrides):
                    self.assertEqual(self._flags(by_key[key]), {})
                    self.assertNotIn(by_key["quality"].produces[0], by_key[key].requires)


class TestPhaseGroups(unittest.TestCase):
    def test_only_structures_selects_exactly_the_structure_phases(self):
        self.assertEqual(wf.expand_phase_keys(["structures"]),
                         list(TestStructurePhases.STRUCTURE_KEYS))

    def test_groups_name_phases_that_exist(self):
        keys = {p.key for p in plan()}
        for group, members in wf.PHASE_GROUPS.items():
            for member in members:
                with self.subTest(group=group, phase=member):
                    self.assertIn(member, keys)

    def test_the_groups_partition_everything_except_dock_and_parse(self):
        grouped = {k for members in wf.PHASE_GROUPS.values() for k in members}
        self.assertEqual({p.key for p in plan()} - grouped, {"dock", "parse"})

    def test_an_unrecognised_key_passes_through_unchanged(self):
        self.assertEqual(wf.expand_phase_keys(["dock", "nonsense"]), ["dock", "nonsense"])


class TestPrerequisiteDiscipline(unittest.TestCase):
    def test_a_missing_prerequisite_stops_the_run(self):
        phase = wf.Phase("x", "X", "build_identity_index.py", [],
                         produces=[], requires=["/nonexistent/artifact.csv"])
        with self.assertRaises(wf.PhaseFailed) as ctx:
            wf.run_phase(phase, dry_run=True)
        self.assertIn("never a fallback", str(ctx.exception))

    def test_dry_run_shows_the_whole_plan(self):
        """Without will_exist the dry run halts at phase 2 and hides the rest."""
        phases = plan()
        produced = []
        for phase in phases:
            wf.run_phase(phase, dry_run=True, will_exist=produced)
            produced.extend(phase.produces)
        self.assertGreaterEqual(len(phases), 8)

    def test_chain_is_connected_every_requirement_is_produced_upstream(self):
        """A phase may only require what an earlier phase actually produces."""
        for mode in wf.MODES:
            produced = set()
            for phase in plan(mode=mode):
                with self.subTest(mode=mode, phase=phase.key):
                    for requirement in phase.requires:
                        self.assertIn(
                            requirement, produced,
                            "%s requires %s which no earlier phase produces"
                            % (phase.key, requirement))
                produced.update(phase.produces)


class TestCli(unittest.TestCase):
    def test_only_and_skip_select_phases(self):
        argv = ["--run-dir", "/tmp/x", "--only", "identity", "--dry-run"]
        args = wf.build_parser().parse_args(argv)
        self.assertEqual(args.only, ["identity"])
        args = wf.build_parser().parse_args(
            ["--run-dir", "/tmp/x", "--skip", "dock", "--skip", "parse"])
        self.assertEqual(args.skip, ["dock", "parse"])

    def test_complex_pdb_defaults_to_keep(self):
        """consensus_docker.py computes RMSD FROM complex_*.pdb.

        It discovers poses with glob("complex_*.pdb") and needs two tools to have
        them. Deleting them by default breaks the RMSD stage -- verified in a live
        pilot: an --only_rmsd run then logs "Not enough tools with valid results"
        and exits 0 having written no final_results.csv. They are still 89% of the
        old campaign's 1.4 TB, so delete/compress remain available as an archival
        step once the analysis table is built.
        """
        self.assertEqual(
            wf.build_parser().parse_args(["--run-dir", "/tmp/x"]).complex_pdb, "keep")

    def test_no_phases_selected_exits_nonzero(self):
        rc = wf.main(["--run-dir", "/tmp/x", "--only", "nonexistent-phase", "--dry-run"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()


class TestInputOutputSeparation(unittest.TestCase):
    """Inputs are shared and versioned; outputs are per-run. Tangling them is how
    the 2025-06-29 directory ended up holding 1.55 TB of mixed provenance."""

    def _inputs(self, tmp):
        os.makedirs(os.path.join(tmp, "uniprot"))
        os.makedirs(os.path.join(tmp, "ligands"))
        for rel in ("1_drug_to_protein.tsv", "small_molecule_drug_links.csv",
                    os.path.join("uniprot", "HUMAN_9606_idmapping.dat.gz"),
                    os.path.join("uniprot", "UP000005640_9606.fasta.gz")):
            open(os.path.join(tmp, rel), "w").close()
        return tmp

    def test_inputs_dir_populates_every_declared_input(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self._inputs(tmp)
            args = wf.build_parser().parse_args(
                ["--run-dir", "/tmp/out", "--inputs-dir", tmp])
            wf.resolve_inputs(args)
            for attr in wf.INPUT_LAYOUT:
                with self.subTest(input=attr):
                    self.assertTrue(getattr(args, attr), "%s unresolved" % attr)
                    self.assertTrue(getattr(args, attr).startswith(tmp))

    def test_an_explicit_flag_beats_the_inputs_dir(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self._inputs(tmp)
            args = wf.build_parser().parse_args(
                ["--run-dir", "/tmp/out", "--inputs-dir", tmp,
                 "--edges", "/elsewhere/edges.tsv"])
            wf.resolve_inputs(args)
            self.assertEqual(args.edges, "/elsewhere/edges.tsv")

    def test_a_missing_input_is_reported_not_silently_defaulted(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "uniprot"))
            args = wf.build_parser().parse_args(
                ["--run-dir", "/tmp/out", "--inputs-dir", tmp])
            wf.resolve_inputs(args)
            self.assertIsNone(args.edges)  # absent stays absent; the phase will say so

    def test_run_dir_is_output_only_and_distinct_from_inputs(self):
        args = wf.build_parser().parse_args(
            ["--run-dir", "/data/runs/r1", "--inputs-dir", "/data/inputs"])
        self.assertNotEqual(args.run_dir, args.inputs_dir)
        for phase in wf.build_phases(args):
            for produced in phase.produces:
                with self.subTest(phase=phase.key):
                    self.assertTrue(
                        produced.startswith(args.run_dir),
                        "%s writes %s outside the output directory"
                        % (phase.key, produced))


class LoggingIsolation(unittest.TestCase):
    """Keep a run's log handlers from outliving the temporary directory.

    ``configure_logging`` attaches a FileHandler to the step logger *and* to the
    shared ``dock_selectivity`` package logger. A test that calls ``main()``
    inside a TemporaryDirectory and walks away leaves that handler attached to a
    path that no longer exists, and the next test anywhere in the process that
    logs through ``dock_selectivity.*`` gets a FileNotFoundError from logging
    itself. Restore both loggers exactly, and drop any handler whose stream the
    runtime legitimately closed while we were away.
    """

    LOGGERS = (wf.STEP_NAME, "dock_selectivity")

    def setUp(self):
        super().setUp()
        self._saved = []
        for name in self.LOGGERS:
            logger = logging.getLogger(name)
            self._saved.append((logger, list(logger.handlers), logger.level))
        self._stderr = sys.stderr
        sys.stderr = io.StringIO()

    def tearDown(self):
        sys.stderr = self._stderr
        for logger, handlers, level in self._saved:
            for handler in list(logger.handlers):
                if handler not in handlers:
                    logger.removeHandler(handler)
                    try:
                        handler.close()
                    except Exception:
                        pass
            logger.handlers[:] = [h for h in handlers
                                  if getattr(h, "stream", True) is not None]
            logger.setLevel(level)
        super().tearDown()


class TestPerPhaseOutputDirectories(unittest.TestCase):
    """Every phase writes its log into its own folder under <run-dir>/steps/.

    Before this, twelve scripts logged into the invocation directory or into the
    artifact tree, and their console output went nowhere at all once the terminal
    was closed.
    """

    RUN = "/tmp/does-not-exist-run"

    def test_every_phase_is_given_its_own_step_directory(self):
        for phase in plan():
            with self.subTest(phase=phase.key):
                expected = os.path.join(self.RUN, "steps", phase.key)
                if phase.key in wf.ARTIFACT_OUT_DIR_PHASES:
                    self.assertIn("--log-file", phase.args)
                    target = phase.args[phase.args.index("--log-file") + 1]
                    self.assertEqual(os.path.dirname(target), expected)
                else:
                    self.assertIn("--out-dir", phase.args)
                    self.assertEqual(
                        phase.args[phase.args.index("--out-dir") + 1], expected)

    def test_the_structure_phases_keep_out_dir_meaning_the_structures(self):
        """--out-dir is the receptor destination for those four, not a log dir.

        Handing them a step directory there would silently move where the
        receptors land, and batch_dock.py would then dock against whatever older
        structures were on the path.
        """
        by_key = {p.key: p for p in plan()}
        for key, tail in (("cavity-structures", "cavities"), ("alphafold", "alphafold"),
                          ("fix", "fixed"), ("pdbqt", "pdbqt")):
            with self.subTest(phase=key):
                args = by_key[key].args
                destination = args[args.index("--out-dir") + 1]
                self.assertEqual(destination,
                                 os.path.join(self.RUN, "structures", tail))

    def test_the_two_extract_cavities_phases_do_not_share_a_log(self):
        """Phase 2 and phase 6a run the same script; one log would interleave them."""
        by_key = {p.key: p for p in plan()}
        first = by_key["cavities"].args[by_key["cavities"].args.index("--log-file") + 1]
        second = by_key["cavity-structures"].args[
            by_key["cavity-structures"].args.index("--log-file") + 1]
        self.assertNotEqual(first, second)

    def test_every_phase_script_accepts_the_flag_it_is_handed(self):
        """The flags are real, not hopeful: parse each command with its own parser."""
        import importlib

        for phase in plan():
            if phase.script == "batch_dock.py":
                continue        # its parser is built under __main__, not importable
            module = importlib.import_module(phase.script[:-len(".py")])
            builder = (getattr(module, "build_parser", None)
                       or getattr(module, "build_arg_parser"))
            with self.subTest(phase=phase.key):
                try:
                    builder().parse_args(phase.args)
                except SystemExit as exc:      # argparse rejected something
                    self.fail("%s does not accept %s (%s)"
                              % (phase.script, phase.args, exc))


class TestWorkflowLogLocation(LoggingIsolation):
    def test_the_workflow_log_is_under_run_dir_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = wf.build_parser().parse_args(["--run-dir", tmp])
            path = wf.resolve_runtime_paths(args)
            self.assertEqual(path, os.path.join(tmp, "logs", "workflow.log"))
            self.assertEqual(args.out_dir, os.path.join(tmp, "logs"))

    def test_an_explicit_log_file_wins(self):
        args = wf.build_parser().parse_args(
            ["--run-dir", "/tmp/x", "--log-file", "/tmp/elsewhere.log"])
        self.assertEqual(wf.resolve_runtime_paths(args), "/tmp/elsewhere.log")

    def test_a_real_run_writes_the_workflow_log_and_the_phase_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "run")
            with mock.patch.object(wf.subprocess, "run", _fake_phase()):
                rc = wf.main(["--run-dir", run, "--only", "identity"])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(os.path.join(run, "logs", "workflow.log")))
            self.assertTrue(os.path.exists(
                os.path.join(run, "steps", "identity", "identity.console.log")))


class TestPhaseOutputIsCapturedNotSwallowed(LoggingIsolation):
    def test_stdout_and_stderr_land_in_the_phase_console_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "run")
            with mock.patch.object(wf.subprocess, "run",
                                   _fake_phase(output="hello from the phase\n")):
                self.assertEqual(wf.main(["--run-dir", run, "--only", "identity"]), 0)
            with open(os.path.join(run, "steps", "identity",
                                   "identity.console.log")) as handle:
                self.assertIn("hello from the phase", handle.read())

    def test_a_failure_still_shows_the_operator_what_went_wrong(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "run")
            with mock.patch.object(wf.subprocess, "run",
                                   _fake_phase(returncode=3, output="TRACEBACK HERE\n")):
                self.assertEqual(wf.main(["--run-dir", run, "--only", "identity"]), 1)
            with open(os.path.join(run, "logs", "workflow.log")) as handle:
                workflow_log = handle.read()
            self.assertIn("TRACEBACK HERE", workflow_log)   # not swallowed
            self.assertIn("exited 3", workflow_log)


class TestNotifications(LoggingIsolation):
    NO_CREDENTIALS = {k: v for k, v in os.environ.items()
                      if k not in (Pushover.ENV_TOKEN, Pushover.ENV_USER_KEY)}
    CREDENTIALS = dict(NO_CREDENTIALS,
                       **{Pushover.ENV_TOKEN: "t0ken", Pushover.ENV_USER_KEY: "us3r"})

    def test_notify_parses_without_credentials(self):
        """Parsing must not consult the environment; only running may."""
        with mock.patch.dict(os.environ, self.NO_CREDENTIALS, clear=True):
            args = wf.build_parser().parse_args(["--run-dir", "/tmp/x", "--notify"])
        self.assertTrue(args.notify)

    def test_notify_without_credentials_refuses_at_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, self.NO_CREDENTIALS, clear=True):
                with mock.patch.object(wf.subprocess, "run", _fake_phase()) as runner:
                    rc = wf.main(["--run-dir", tmp, "--only", "identity", "--notify"])
            self.assertEqual(rc, 2)
            # Before the first phase, not at hour 61.
            self.assertEqual(runner.call_count, 0)

    def test_a_failed_phase_notifies_with_its_name_and_exit_code(self):
        sent = self._run_notified(["--only", "identity", "--notify"], returncode=4)
        failures = [m for m in sent if int(m["priority"][0]) == 1]
        self.assertEqual(len(failures), 1)
        self.assertIn("identity", failures[0]["message"][0])
        self.assertIn("exit 4", failures[0]["message"][0])

    def test_completion_notifies_with_the_elapsed_time_and_a_per_phase_summary(self):
        sent = self._run_notified(
            ["--only", "identity", "--only", "cavities", "--notify"])
        self.assertEqual(len(sent), 2, [m["message"][0] for m in sent])
        start, done = sent
        self.assertIn("started", start["message"][0])
        self.assertIn("finished in", done["message"][0])
        for key in ("identity", "cavities"):
            self.assertIn(key, done["message"][0])

    def test_notify_steps_forwards_the_flag_and_plain_notify_does_not(self):
        with mock.patch.dict(os.environ, self.CREDENTIALS, clear=True):
            plain = wf.build_parser().parse_args(["--run-dir", "/tmp/x", "--notify"])
            forwarded = wf.build_parser().parse_args(
                ["--run-dir", "/tmp/x", "--notify", "--notify-steps"])
        for phase in wf.build_phases(plain):
            with self.subTest(phase=phase.key, forwarded=False):
                self.assertNotIn("--notify", phase.args)
        for phase in wf.build_phases(forwarded):
            with self.subTest(phase=phase.key, forwarded=True):
                self.assertIn("--notify", phase.args)

    def test_forwarded_phases_get_only_the_events_the_operator_asked_for(self):
        """The workflow defaults itself to every event; twelve phases must not."""
        args = wf.build_parser().parse_args(
            ["--run-dir", "/tmp/x", "--notify", "--notify-steps"])
        args._notify_on_given = []          # what main() records for a bare --notify
        for phase in wf.build_phases(args):
            with self.subTest(phase=phase.key):
                self.assertNotIn("--notify-on", phase.args)

        args = wf.build_parser().parse_args(
            ["--run-dir", "/tmp/x", "--notify", "--notify-steps", "--notify-on", "start"])
        args._notify_on_given = ["start"]
        for phase in wf.build_phases(args):
            with self.subTest(phase=phase.key, events=True):
                self.assertIn("--notify-on", phase.args)

    def _run_notified(self, extra, returncode=0):
        """Run the workflow with credentials present and Pushover intercepted."""
        sent = []

        class _Response:
            status = 200

            def close(self):
                pass

        def _urlopen(request, timeout=None):
            sent.append(urllib.parse.parse_qs(request.data.decode("utf-8")))
            return _Response()

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, self.CREDENTIALS, clear=True), \
                    mock.patch("urllib.request.urlopen", _urlopen), \
                    mock.patch.object(wf.subprocess, "run",
                                      _fake_phase(returncode=returncode)):
                wf.main(["--run-dir", os.path.join(tmp, "run")] + list(extra))
        return sent


class TestDryRunWritesNothing(LoggingIsolation):
    def test_the_whole_plan_is_printed_and_nothing_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "run")     # deliberately absent
            with mock.patch.object(wf.subprocess, "run") as runner:
                with self.assertLogs(wf.LOG, level="INFO") as captured:
                    rc = wf.main(["--run-dir", run, "--dry-run"])
            self.assertEqual(rc, 0)
            self.assertEqual(runner.call_count, 0)
            self.assertFalse(os.path.exists(run), "a dry run created %s" % run)
            self.assertEqual(os.listdir(tmp), [])
            printed = "\n".join(captured.output)
            for key in [p.key for p in plan()]:
                with self.subTest(phase=key):
                    self.assertIn(key, printed)

    def test_a_dry_run_writes_no_log_of_its_own(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = wf.build_parser().parse_args(["--run-dir", tmp, "--dry-run"])
            self.assertEqual(wf.resolve_runtime_paths(args), os.devnull)
            wf.main(["--run-dir", tmp, "--dry-run"])
            self.assertEqual(os.listdir(tmp), [])


def _fake_phase(returncode=0, output=""):
    """A stand-in for ``subprocess.run`` that satisfies the produces-check.

    Each phase writes the paths it was given with ``--out`` / ``--redock-out``,
    which is exactly what ``run_phase`` verifies afterwards -- so a fake that
    touches them exercises the real control flow without running twelve scripts.
    """
    def runner(command, timeout=None, stdout=None, stderr=None):
        if stdout is not None and output:
            stdout.write(output.encode("utf-8"))
        if returncode == 0:
            for index, token in enumerate(command[:-1]):
                if token in ("--out", "--redock-out"):
                    target = command[index + 1]
                    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
                    open(target, "a").close()
        return subprocess.CompletedProcess(command, returncode)

    return mock.Mock(side_effect=runner)
