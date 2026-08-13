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

import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import run_full_workflow as wf  # noqa: E402


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

    def test_salvage_runs_only_in_salvage_mode(self):
        self.assertIn("salvage", [p.key for p in plan(mode="salvage-and-redock")])
        self.assertNotIn("salvage", [p.key for p in plan(mode="full-redock")])

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
        self.assertEqual(
            set(wf.MODES), {"positives-only", "salvage-and-redock", "full-redock"})
        parser = wf.build_parser()
        with self.assertRaises(SystemExit):  # an unknown mode is rejected outright
            parser.parse_args(["--run-dir", "/tmp/x", "--mode", "whatever-is-lying-around"])


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

    def test_complex_pdb_defaults_to_delete(self):
        """Otherwise the new run directory grows back to 1.4 TB of derived files."""
        self.assertEqual(
            wf.build_parser().parse_args(["--run-dir", "/tmp/x"]).complex_pdb, "delete")

    def test_no_phases_selected_exits_nonzero(self):
        rc = wf.main(["--run-dir", "/tmp/x", "--only", "nonexistent-phase", "--dry-run"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
