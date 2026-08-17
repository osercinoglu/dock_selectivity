"""The operational tools' command line: every setting reachable, nothing hardcoded.

The six scripts here (audit, curate_jobs, classify_existing, gnina_collect,
gnina_remote, batch_dock) are what an operator actually runs during a campaign,
so they are the ones where a stale default costs the most: batch_dock's
``OUTPUT_BASE_FOLDER`` named the 2025-06-29 campaign, which lives on a read-only
drive, and ``run_docking``'s own default named it a second time -- so an
invocation that forgot ``--run-dir`` did not fail, it wrote somewhere plausible.

What is pinned here:

* the parser builds, ``--help`` exits 0, and the shared runtime flags are on it;
* importing the module configures no logging and writes no file;
* no module constant still points at a past run directory or a home directory;
* the values that used to be editable constants are reachable as flags;
* a smoke run of each entry point in a temp directory puts its log under
  ``<run-dir>/steps/<step>/`` and nowhere else.
"""

from __future__ import annotations

import ast
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import audit  # noqa: E402
import classify_existing  # noqa: E402
import curate_jobs  # noqa: E402
import gnina_collect  # noqa: E402
import gnina_remote  # noqa: E402
from dock_selectivity import runtime  # noqa: E402

#: The scripts this file owns.
OPS_SCRIPTS = (
    "audit.py",
    "curate_jobs.py",
    "classify_existing.py",
    "gnina_collect.py",
    "gnina_remote.py",
    "batch_dock.py",
)

#: Substrings that mean "this default belongs to one past run or one person's
#: account". /opt is allowed: it is a machine-wide install location, and the
#: work order says a machine-wide reference path may stay as a default.
DEAD_PATH_MARKERS = ("2025_06_29", "cavity_space_consensus_docking", "/home/")

#: Flags dock_selectivity.runtime adds to every step's parser.
RUNTIME_FLAGS = ("--out-dir", "--log-file", "--log-level", "--quiet",
                 "--notify", "--notify-on", "--notify-title")


def source_of(script):
    with open(os.path.join(REPO, script), encoding="utf-8") as handle:
        return handle.read()


def string_constants(script):
    """Every string literal assigned to a module-level name, with its name.

    AST rather than import, so a constant computed at import time (batch_dock
    resolves its engine paths through $PATH) is judged on what the *source*
    says -- which is where a hardcoded path would be written.
    """
    found = []
    tree = ast.parse(source_of(script))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not names:
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                found.append((names[0], sub.value))
    return found


def run_script(script, *args, **kwargs):
    return subprocess.run([sys.executable, os.path.join(REPO, script)] + list(args),
                          cwd=kwargs.get("cwd", REPO), capture_output=True,
                          text=True, timeout=180)


class LoggerIsolation(unittest.TestCase):
    """Put every logger this test touched back exactly as it was.

    The step logger and the ``dock_selectivity`` package logger are shared with
    the rest of the suite, and a FileHandler left pointing into a deleted temp
    directory is a failure in somebody else's test half an hour later.
    """

    STEP_LOGGERS = ("audit", "curate_jobs", "classify_existing", "gnina_collect",
                    "gnina_remote", "batch_dock", "dock_selectivity")

    def setUp(self):
        self._saved = {}
        for name in self.STEP_LOGGERS:
            log = logging.getLogger(name)
            self._saved[name] = (list(log.handlers), log.level)
        self.addCleanup(self._restore)

    @staticmethod
    def _alive(handler):
        """A FileHandler the runtime has already replaced has ``stream = None``.

        ``logging`` reopens such a handler's path on the next record, so putting
        one back is how a deleted temp directory becomes a FileNotFoundError in
        an unrelated test file half an hour later.
        """
        return not (isinstance(handler, logging.FileHandler)
                    and handler.stream is None)

    def _restore(self):
        library = logging.getLogger("dock_selectivity")
        for name in self.STEP_LOGGERS:
            log = logging.getLogger(name)
            handlers, level = self._saved[name]
            for handler in list(log.handlers):
                if handler not in handlers:
                    log.removeHandler(handler)
                    library.removeHandler(handler)
                    try:
                        handler.close()
                    except Exception:  # pragma: no cover - already closed
                        pass
            log.handlers[:] = [h for h in handlers if self._alive(h)]
            log.setLevel(level)


class TestNoDeadDefaults(unittest.TestCase):
    def test_no_ops_script_names_a_past_run_directory_or_a_home_directory(self):
        for script in OPS_SCRIPTS:
            for name, value in string_constants(script):
                for marker in DEAD_PATH_MARKERS:
                    with self.subTest(script=script, const=name):
                        self.assertNotIn(
                            marker, value,
                            "%s: %s carries %r, which names one past campaign or "
                            "one person's account rather than this run" % (
                                script, name, value))

    def test_batch_dock_has_no_output_default_at_all(self):
        import batch_dock
        self.assertIsNone(
            batch_dock.OUTPUT_BASE_FOLDER,
            "a default output directory is how a run writes into a previous "
            "campaign; --run-dir or --output-base must be given")

    def test_run_docking_requires_an_output_folder(self):
        """It used to default to the 2025-06-29 tree, so a forgotten argument
        was silently a different experiment."""
        import inspect

        import batch_dock
        signature = inspect.signature(batch_dock.run_docking)
        parameter = signature.parameters["output_base_folder"]
        self.assertIs(parameter.default, inspect.Parameter.empty)

    def test_batch_dock_refuses_to_dock_without_an_output_directory(self):
        proc = run_script("batch_dock.py", "--tools", "smina")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("--run-dir", proc.stdout + proc.stderr)


class TestParsersBuild(unittest.TestCase):
    def test_help_exits_zero_for_every_ops_script(self):
        # gnina_remote is not in the list: it presents gnina's command line, and
        # gnina has no --help of ours to print. --capabilities is its query.
        for script in [s for s in OPS_SCRIPTS if s != "gnina_remote.py"]:
            with self.subTest(script=script):
                proc = run_script(script, "--help")
                self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
                self.assertTrue(proc.stdout.strip())

    def test_gnina_remote_answers_the_query_it_does_have(self):
        proc = run_script("gnina_remote.py", "--capabilities")
        self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
        for mode in gnina_remote.MODES:
            self.assertIn(mode, proc.stdout)

    def test_the_runtime_flags_are_on_every_step_parser(self):
        # gnina_remote is exempt: consensus_docker.py execs it as if it were the
        # gnina binary, and every argument it does not recognise is forwarded to
        # the remote gnina, so a --log-file here would be sent to gnina.
        for script in ("audit.py", "curate_jobs.py", "classify_existing.py",
                       "gnina_collect.py", "batch_dock.py"):
            proc = run_script(script, "--help")
            for flag in RUNTIME_FLAGS:
                with self.subTest(script=script, flag=flag):
                    self.assertIn(flag, proc.stdout)

    def test_parsing_sets_the_runtime_values(self):
        args = classify_existing.build_parser().parse_args([
            "--existing-results", "/nowhere", "--index", "i.parquet",
            "--interactions", "x.csv", "--out-map", "m.csv", "--out-pool", "p.csv",
            "--out-dir", "/tmp/somewhere", "--log-level", "DEBUG", "--quiet",
            "--notify-on", "success",
        ])
        self.assertEqual(args.out_dir, "/tmp/somewhere")
        self.assertEqual(args.log_level, "DEBUG")
        self.assertTrue(args.quiet)
        self.assertEqual(args.notify_on, ["success"])

    def test_gnina_collect_keeps_log_as_a_spelling_of_log_file(self):
        """batch_dock spawns the collector with --log; that command line is in
        running campaigns and must not become an unknown-argument error."""
        args = gnina_collect.build_parser().parse_args(
            ["--run-dir", "/tmp/x", "--log", "/tmp/x/collector.log"])
        self.assertEqual(args.log_file, "/tmp/x/collector.log")

    def test_the_collectors_retry_tunables_are_flags(self):
        """They used to be editable constants: a flaky link could only be
        survived by changing the source between runs."""
        args = gnina_collect.build_parser().parse_args(
            ["--run-dir", "/tmp/x", "--reconcile-interval", "30",
             "--max-transient-attempts", "42"])
        self.assertEqual(args.reconcile_interval, 30.0)
        self.assertEqual(args.max_transient_attempts, 42)

    def test_audit_takes_the_runtime_flags_before_the_subcommand(self):
        args = audit._build_parser().parse_args(
            ["--quiet", "--log-level", "ERROR", "run", "--run-dir", "/tmp/r"])
        self.assertTrue(args.quiet)
        self.assertEqual(args.log_level, "ERROR")
        self.assertEqual(audit.infer_run_dir(args), "/tmp/r")

    def test_audit_infers_the_run_directory_from_the_artifact_it_is_given(self):
        args = audit._build_parser().parse_args(
            ["curated", "--dir", "/tmp/r/docking_results",
             "--manifest", "/tmp/r/manifests/run_manifest.csv",
             "--index", "/tmp/r/identity/identity_index.parquet"])
        self.assertEqual(audit.infer_run_dir(args), "/tmp/r")


class TestBatchDockFlagsForEveryConstant(unittest.TestCase):
    """The five decoupled-gnina settings that had only an environment variable."""

    NEWLY_FLAGGED = (
        ("--gnina-validate-caps", "GNINA_VALIDATE_CAPS", None),
        ("--gnina-job-deadline", "GNINA_JOB_DEADLINE", "900"),
        ("--gnina-max-attempts", "GNINA_MAX_ATTEMPTS", "5"),
        ("--leave-collector", "GNINA_LEAVE_COLLECTOR", None),
        ("--caps-file", "GNINA_CAPS_FILE", "/tmp/caps.json"),
    )

    def test_each_is_in_the_parameters_table(self):
        import batch_dock
        flagged = {const for _f, const, _k, _h in batch_dock.PARAMETERS}
        for _flag, const, _value in self.NEWLY_FLAGGED:
            with self.subTest(const=const):
                self.assertIn(const, flagged)

    def test_each_flag_actually_overrides_the_constant(self):
        import batch_dock
        saved = {c: getattr(batch_dock, c) for _f, c, _v in self.NEWLY_FLAGGED}
        self.addCleanup(lambda: [setattr(batch_dock, k, v) for k, v in saved.items()])
        parser = batch_dock.argparse.ArgumentParser()
        batch_dock._add_parameter_arguments(parser)
        argv = []
        for flag, _const, value in self.NEWLY_FLAGGED:
            argv.append(flag)
            if value is not None:
                argv.append(value)
        batch_dock.apply_parameter_arguments(parser.parse_args(argv))
        self.assertIs(batch_dock.GNINA_VALIDATE_CAPS, True)
        self.assertEqual(batch_dock.GNINA_JOB_DEADLINE, 900.0)
        self.assertEqual(batch_dock.GNINA_MAX_ATTEMPTS, 5)
        self.assertIs(batch_dock.GNINA_LEAVE_COLLECTOR, True)
        self.assertEqual(batch_dock.GNINA_CAPS_FILE, "/tmp/caps.json")

    def test_an_enabled_engine_with_no_binary_is_named(self):
        import batch_dock
        saved = (batch_dock.USE_SMINA, batch_dock.SMINA_PATH,
                 batch_dock.CONSENSUS_DOCKER_SCRIPT)

        def restore():
            (batch_dock.USE_SMINA, batch_dock.SMINA_PATH,
             batch_dock.CONSENSUS_DOCKER_SCRIPT) = saved

        self.addCleanup(restore)
        batch_dock.USE_SMINA = True
        batch_dock.SMINA_PATH = "/definitely/not/here/smina"
        batch_dock.CONSENSUS_DOCKER_SCRIPT = os.path.abspath(__file__)
        missing = batch_dock.missing_engine_binaries()
        self.assertEqual([m[0] for m in missing], ["smina"])
        self.assertEqual(missing[0][3], "--smina-path")


class TestImportingConfiguresNothing(unittest.TestCase):
    SNIPPET = (
        "import sys, os, json, logging; sys.path.insert(0, {repo!r}); "
        "before = list(logging.getLogger().handlers); "
        "import {module}; "
        "after = list(logging.getLogger().handlers); "
        "print(json.dumps([len(before) == len(after), sorted(os.listdir('.'))]))"
    )

    def test_importing_touches_neither_the_root_logger_nor_the_cwd(self):
        for module in ("audit", "curate_jobs", "classify_existing",
                       "gnina_collect", "gnina_remote", "batch_dock"):
            with self.subTest(module=module), tempfile.TemporaryDirectory() as tmp:
                proc = subprocess.run(
                    [sys.executable, "-c",
                     self.SNIPPET.format(repo=REPO, module=module)],
                    cwd=tmp, capture_output=True, text=True, timeout=180)
                self.assertEqual(proc.returncode, 0, proc.stderr[-1200:])
                unchanged, listing = json.loads(proc.stdout.strip().splitlines()[-1])
                self.assertTrue(unchanged,
                                "importing %s added a root handler" % module)
                self.assertEqual(listing, [],
                                 "importing %s wrote %s into the cwd"
                                 % (module, listing))


class TestLogsLandInTheStepDirectory(LoggerIsolation):
    def test_audit_writes_its_log_under_the_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = audit.main(["run", "--run-dir", tmp, "--quiet", "--allow-skips"])
            self.assertIn(code, (0, 1, 2))
            log = os.path.join(tmp, "steps", "audit", "audit.log")
            self.assertTrue(os.path.exists(log), sorted(os.listdir(tmp)))
            self.assertTrue(os.path.getsize(log) > 0)

    def test_gnina_collect_writes_its_log_under_the_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            # No queue there, so the pump refuses -- after configuring logging.
            code = gnina_collect.main(["--run-dir", tmp, "--once", "--quiet"])
            self.assertEqual(code, gnina_collect.EXIT_USAGE)
            log = os.path.join(tmp, "steps", "gnina_collect", "gnina_collect.log")
            self.assertTrue(os.path.exists(log))
            with open(log, encoding="utf-8") as handle:
                self.assertIn("no gnina queue at", handle.read())

    def test_a_query_creates_no_directory_and_no_log(self):
        """--status is a question about a running pump, not a run of one."""
        with tempfile.TemporaryDirectory() as tmp:
            code = gnina_collect.main(["--run-dir", tmp, "--status"])
            self.assertEqual(code, gnina_collect.EXIT_USAGE)
            self.assertEqual(os.listdir(tmp), [])

    def test_curate_jobs_dry_run_still_writes_absolutely_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "old")
            os.makedirs(src)
            code = curate_jobs.main([
                "--job-classification", os.path.join(tmp, "missing.csv"),
                "--drawn", os.path.join(tmp, "missing.csv"),
                "--positives", os.path.join(tmp, "missing.csv"),
                "--index", os.path.join(tmp, "missing.parquet"),
                "--src", src, "--dst", os.path.join(tmp, "new", "docking_results"),
                "--dry-run", "--quiet",
            ])
            self.assertEqual(code, 1)
            files = [os.path.join(d, f)
                     for d, _dirs, names in os.walk(tmp) for f in names]
            self.assertEqual(files, [], "--dry-run wrote %s" % files)


class TestGninaRemoteStaysAStandIn(LoggerIsolation):
    def test_it_declares_no_runtime_flags_of_its_own(self):
        """Its argv belongs to gnina: unknown flags are forwarded to the remote
        binary, so a runtime flag here would be sent to gnina instead."""
        _known, passthrough = gnina_remote.parse_args(
            ["-r", "r.pdb", "-l", "l.sdf", "-o", "o.sdf", "--out-dir", "/tmp/x"])
        self.assertIn("--out-dir", passthrough)

    def test_the_stderr_contract_is_unchanged_and_the_file_is_extra(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shim.log")
            os.environ["GNINA_REMOTE_LOG"] = path
            host = os.environ.pop("GNINA_REMOTE_HOST", None)
            if host is not None:
                self.addCleanup(os.environ.__setitem__, "GNINA_REMOTE_HOST", host)
            self.addCleanup(os.environ.pop, "GNINA_REMOTE_LOG", None)
            stderr = io.StringIO()
            saved = sys.stderr
            sys.stderr = stderr
            try:
                # No GNINA_REMOTE_HOST: it refuses, and says so on stderr.
                code = gnina_remote.main(["-r", "r.pdb", "-l", "l.sdf", "-o", "o.sdf"])
            finally:
                sys.stderr = saved
            self.assertEqual(code, gnina_remote.EXIT_USAGE)
            self.assertIn("GNINA_REMOTE_HOST", stderr.getvalue())
            with open(path, encoding="utf-8") as handle:
                self.assertIn("GNINA_REMOTE_HOST", handle.read())

    def test_the_poll_interval_is_configurable(self):
        os.environ["GNINA_REMOTE_POLL"] = "0.25"
        os.environ["GNINA_REMOTE_HOST"] = "user@nowhere"
        self.addCleanup(os.environ.pop, "GNINA_REMOTE_POLL", None)
        self.addCleanup(os.environ.pop, "GNINA_REMOTE_HOST", None)
        known, rest = gnina_remote.parse_args(
            ["-r", "r.pdb", "-l", "l.sdf", "-o", "o.sdf"])
        cfg = gnina_remote._Config(known, rest, lambda _msg: None)
        self.assertEqual(cfg.poll_interval, 0.25)


class TestSharedRuntimeHandlerHygiene(LoggerIsolation):
    def test_configuring_a_second_step_does_not_break_the_first(self):
        """A handler the library logger merely borrows must not be closed there:
        logging reopens a closed FileHandler, and the step's directory may be
        long gone by then."""
        with tempfile.TemporaryDirectory() as one, \
                tempfile.TemporaryDirectory() as two:
            first = runtime.configure_logging("audit", out_dir=one, quiet=True)
            runtime.configure_logging("curate_jobs", out_dir=two, quiet=True)
            first.warning("the first step is still alive")
            with open(os.path.join(one, "audit.log"), encoding="utf-8") as handle:
                self.assertIn("still alive", handle.read())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
