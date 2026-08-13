"""batch_dock.py's configuration is command-line arguments, not edited constants.

Every setting used to be a module constant you edited in place between runs. That
is a large part of why the 2025-06-29 campaign cannot be reproduced: the
parameters of a run lived in a file that was modified between runs and never
recorded alongside the output, so the artifacts do not say what produced them.

The load-bearing test here is test_every_config_constant_has_a_flag: adding a new
constant without adding a flag fails the suite, so the old habit cannot come back
one constant at a time.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import batch_dock as bd  # noqa: E402

#: Constants that are deliberately NOT flags, with the reason. Anything else must
#: have one. Keeping this list explicit means an exemption is a decision someone
#: made, not an oversight nobody noticed.
NOT_PARAMETERS = {
    "MODE_SYNC", "MODE_SUBMIT", "MODE_COLLECT", "MODES",       # gnina_remote protocol
    "PARAMETERS", "NOT_PARAMETERS",                            # this machinery
    "EXIT_OK", "EXIT_FAILED", "EXIT_USAGE",                    # exit codes
    "TEST_MODE", "MAX_TEST_JOBS",                              # --test-mode / --max-jobs
    "AVAILABLE_CPUS",                                          # measured, not chosen
}

#: Env-driven async gnina settings. They are configuration, but they belong to the
#: queue rather than to a docking stage, and are set for a whole campaign rather
#: than per invocation. Listed so the count below is honest about them.
ASYNC_ENV_SETTINGS = {
    "GNINA_ASYNC", "GNINA_VALIDATE_CAPS", "GNINA_STRICT_CAPS", "GNINA_BARRIER_TIMEOUT",
    "GNINA_JOB_DEADLINE", "GNINA_MAX_ATTEMPTS", "GNINA_COLLECTOR_AUTOSTART",
    "GNINA_LEAVE_COLLECTOR", "GNINA_PARTIAL_RMSD", "GNINA_QUEUE_DIR", "GNINA_CAPS_FILE",
}


def config_block_constants():
    """UPPER_CASE assignments in batch_dock's configuration block."""
    source = open(os.path.join(REPO, "batch_dock.py"), encoding="utf-8").read()
    start = source.index("# --- Configuration ---")
    end = source.index("# ---------------------------------------------------------------------------\n# Command-line configuration")
    found = []
    for line in source[start:end].splitlines():
        match = re.match(r"^([A-Z][A-Z0-9_]*)\s*=", line)
        if match:
            found.append(match.group(1))
    return found


class TestEveryConstantIsReachable(unittest.TestCase):
    def test_every_config_constant_has_a_flag(self):
        """The regression guard: a new constant without a flag fails here."""
        flagged = {const for _f, const, _k, _h in bd.PARAMETERS}
        missing = [c for c in config_block_constants()
                   if c not in flagged
                   and c not in NOT_PARAMETERS
                   and c not in ASYNC_ENV_SETTINGS]
        self.assertEqual(
            missing, [],
            "these constants can still only be changed by editing the file, which "
            "is how a run becomes unreproducible: %s. Add them to PARAMETERS, or "
            "to NOT_PARAMETERS with a reason." % missing)

    def test_the_table_has_no_duplicate_flags_or_constants(self):
        flags = [f for f, _c, _k, _h in bd.PARAMETERS]
        consts = [c for _f, c, _k, _h in bd.PARAMETERS]
        self.assertEqual(len(flags), len(set(flags)), "duplicate flag")
        self.assertEqual(len(consts), len(set(consts)), "duplicate constant")

    def test_every_table_entry_names_a_real_constant(self):
        for _flag, const, _kind, _help in bd.PARAMETERS:
            with self.subTest(const=const):
                self.assertTrue(hasattr(bd, const),
                                "%s is in PARAMETERS but does not exist" % const)

    def test_every_entry_has_help_text(self):
        for flag, const, _kind, helptext in bd.PARAMETERS:
            with self.subTest(flag=flag):
                self.assertTrue(helptext and helptext.strip(),
                                "--%s (%s) has no help" % (flag, const))


class TestOverrideMechanics(unittest.TestCase):
    def setUp(self):
        self._saved = {c: getattr(bd, c) for _f, c, _k, _h in bd.PARAMETERS}

    def tearDown(self):
        for const, value in self._saved.items():
            setattr(bd, const, value)

    def _parse(self, argv):
        parser = bd.argparse.ArgumentParser()
        bd._add_parameter_arguments(parser)
        return parser.parse_args(argv)

    def test_an_unset_flag_leaves_the_default_alone(self):
        before = bd.SMINA_EXHAUSTIVENESS
        changed = bd.apply_parameter_arguments(self._parse([]))
        self.assertEqual(changed, {})
        self.assertEqual(bd.SMINA_EXHAUSTIVENESS, before)

    def test_a_numeric_override_is_applied_and_typed(self):
        bd.apply_parameter_arguments(self._parse(["--smina-exhaustiveness", "8"]))
        self.assertEqual(bd.SMINA_EXHAUSTIVENESS, 8)
        self.assertIsInstance(bd.SMINA_EXHAUSTIVENESS, int)

    def test_a_float_override_is_a_float(self):
        bd.apply_parameter_arguments(self._parse(["--cutoff", "-8.5"]))
        self.assertEqual(bd.CUTOFF_VALUE, -8.5)
        self.assertIsInstance(bd.CUTOFF_VALUE, float)

    def test_boolean_flags_go_both_ways(self):
        """--force-rmsd/--no-force-rmsd matters: on a curated evidence-only
        directory the recalculation cannot succeed, so it must be switchable."""
        bd.apply_parameter_arguments(self._parse(["--no-force-rmsd"]))
        self.assertIs(bd.FORCE_RMSD_RECALCULATION, False)
        bd.apply_parameter_arguments(self._parse(["--force-rmsd"]))
        self.assertIs(bd.FORCE_RMSD_RECALCULATION, True)

    def test_changed_settings_are_reported_for_the_log(self):
        changed = bd.apply_parameter_arguments(
            self._parse(["--num-threads", "24", "--smina-jobs", "4"]))
        self.assertIn("NUM_THREADS", changed)
        self.assertIn("SMINA_MAX_PARALLEL_JOBS", changed)
        old, new = changed["NUM_THREADS"]
        self.assertNotEqual(old, new)
        self.assertEqual(new, 24)

    def test_setting_a_value_equal_to_the_default_is_not_reported_as_changed(self):
        changed = bd.apply_parameter_arguments(
            self._parse(["--num-threads", str(bd.NUM_THREADS)]))
        self.assertNotIn("NUM_THREADS", changed)


class TestCli(unittest.TestCase):
    def _run(self, *args, **kwargs):
        return subprocess.run([sys.executable, os.path.join(REPO, "batch_dock.py")] + list(args),
                              cwd=kwargs.get("cwd", REPO), capture_output=True,
                              text=True, timeout=180)

    def test_help_lists_the_flags_and_their_defaults(self):
        proc = self._run("--help")
        self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
        for flag in ("--smina-exhaustiveness", "--num-threads", "--gnina-jobs",
                     "--output-base", "--force-rmsd", "--gnina-path"):
            self.assertIn(flag, proc.stdout, "%s missing from --help" % flag)

    def test_print_config_exits_zero_without_docking(self):
        proc = self._run("--print-config")
        self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
        self.assertIn("resolved configuration", proc.stdout)
        self.assertIn("NUM_THREADS", proc.stdout)

    def test_print_config_marks_what_was_overridden(self):
        """The record the edit-the-file workflow never produced."""
        proc = self._run("--print-config", "--num-threads", "7")
        self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
        line = [l for l in proc.stdout.splitlines() if "NUM_THREADS" in l][0]
        self.assertIn("= 7", line)
        self.assertIn("<- --num-threads", line)

    def test_an_unknown_flag_is_rejected(self):
        proc = self._run("--print-config", "--not-a-real-setting", "1")
        self.assertNotEqual(proc.returncode, 0)

    def test_a_bad_numeric_value_is_rejected(self):
        proc = self._run("--print-config", "--num-threads", "not-a-number")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("invalid", (proc.stderr + proc.stdout).lower())


class TestImportIsQuiet(unittest.TestCase):
    """Importing must not log the configuration or write a log file.

    The banner used to run at import, so it reported the constants before any
    command line was parsed: a run with `--tools smina,ledock` printed
    "STAGE 1: SMINA - DISABLED" and then, seconds later, that smina was enabled.
    """

    def test_import_does_not_emit_the_configuration_banner(self):
        proc = subprocess.run(
            [sys.executable, "-c", "import batch_dock"],
            cwd=REPO, capture_output=True, text=True, timeout=180)
        self.assertEqual(proc.returncode, 0, proc.stderr[-1000:])
        combined = proc.stdout + proc.stderr
        for marker in ("RUNNING IN PRODUCTION MODE", "STAGE 1: SMINA",
                       "BATCH DOCK CONFIGURATION"):
            self.assertNotIn(marker, combined,
                             "importing batch_dock still logs %r" % marker)

    def test_importing_does_not_create_a_log_file(self):
        """`import batch_dock` used to drop docking_automation.log into the cwd."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, "-c", "import sys; sys.path.insert(0, %r); import batch_dock" % REPO],
                cwd=tmp, capture_output=True, text=True, timeout=180)
            self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
            self.assertEqual(
                os.listdir(tmp), [],
                "importing batch_dock created %s in the working directory"
                % os.listdir(tmp))

    def test_print_config_creates_no_log_file(self):
        """A query must not have side effects."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, os.path.join(REPO, "batch_dock.py"), "--print-config"],
                cwd=tmp, capture_output=True, text=True, timeout=180)
            self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
            self.assertEqual(os.listdir(tmp), [],
                             "--print-config created %s" % os.listdir(tmp))

    def test_log_configuration_is_callable_and_reports_the_resolved_values(self):
        self.assertTrue(callable(bd.log_configuration))
        self.assertTrue(callable(bd.configure_logging))


if __name__ == "__main__":
    unittest.main()
