"""CLI and shared-runtime contract for the five planning stages (phases 0-5).

The stages are ``build_identity_index``, ``resolve_interactions``,
``plan_positives``, ``sample_negatives`` and ``build_run_manifest``.  What these
tests pin is not what each stage computes -- their own test files do that -- but
the properties that made the previous generation of scripts unoperable:

* importing a module configured logging, so the first import won the one
  ``FileHandler`` and the test suite dropped megabytes of ``*.log`` into the
  repository working tree;
* a step's log had no home, so it landed in whatever directory the operator
  happened to be standing in;
* real configuration (the repair-scan budget, the ligand extensions, the
  determinism seeds) lived in module constants no command line could reach;
* ``--log-level`` was silently ignored whenever the root logger already had
  handlers.

They are hermetic: no network, no ``/media``, no docking, and every artifact is
written under a ``TemporaryDirectory``.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import inspect
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

import build_identity_index as bii
import build_run_manifest as brm
import plan_positives as pp
import resolve_interactions as ri
import sample_negatives as sn
from dock_selectivity import schemas

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: (module, parser factory).  ``build_arg_parser`` versus ``build_parser`` is a
#: pre-existing spelling difference; both names are pinned by other test files.
STAGES = (
    (bii, bii.build_parser),
    (ri, ri.build_arg_parser),
    (pp, pp.build_parser),
    (sn, sn.build_arg_parser),
    (brm, brm.build_parser),
)

RUNTIME_FLAGS = (
    "--out-dir",
    "--log-file",
    "--log-level",
    "--quiet",
    "--notify",
    "--notify-on",
    "--notify-title",
)

#: Paths that must never appear in one of these files again: a dead campaign
#: directory on a read-only drive, and two specific users' home directories.
DEAD_PATH_TOKENS = ("2025_06_29", "/media/onur", "/home/onur", "/home/ssahin")


def flags_of(parser: argparse.ArgumentParser):
    out = set()
    for action in parser._actions:  # argparse exposes no public accessor
        out.update(action.option_strings)
    return out


class TestRuntimeWiring(unittest.TestCase):
    def test_every_stage_declares_the_shared_runtime_flags(self):
        for module, factory in STAGES:
            with self.subTest(stage=module.STEP_NAME):
                flags = flags_of(factory())
                for flag in RUNTIME_FLAGS + ("--run-dir",):
                    self.assertIn(flag, flags, module.STEP_NAME)

    def test_step_name_is_the_logger_name(self):
        """step() configures logging.getLogger(STEP_NAME); a mismatch loses every record."""
        for module, _ in STAGES:
            with self.subTest(stage=module.STEP_NAME):
                own = getattr(module, "LOG", None) or getattr(module, "logger")
                self.assertEqual(own.name, module.STEP_NAME)

    def test_help_exits_zero_and_names_the_step(self):
        for module, factory in STAGES:
            with self.subTest(stage=module.STEP_NAME):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    with self.assertRaises(SystemExit) as ctx:
                        factory().parse_args(["--help"])
                self.assertEqual(ctx.exception.code, 0)
                self.assertIn("--notify", buffer.getvalue())

    def test_main_runs_inside_the_step_context_manager(self):
        """A stage outside step() gets no log file and no failure notification."""
        for module, _ in STAGES:
            entry = getattr(module, "run", None) or module.main
            source = inspect.getsource(entry)
            with self.subTest(stage=module.STEP_NAME):
                self.assertIn("with step(", source)
                self.assertIn("resolve_out_dir(", source)

    def test_no_module_calls_basic_config(self):
        for module, _ in STAGES:
            path = os.path.join(REPO, module.__name__ + ".py")
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            with self.subTest(stage=module.STEP_NAME):
                self.assertNotIn("basicConfig", source)


class TestImportIsQuiet(unittest.TestCase):
    """Import time must configure nothing and create nothing."""

    def test_importing_every_stage_leaves_root_logging_and_the_cwd_alone(self):
        snippet = (
            "import json, logging, os, sys\n"
            "sys.path.insert(0, {repo!r})\n"
            "before = list(logging.getLogger().handlers)\n"
            "import build_identity_index, resolve_interactions, plan_positives\n"
            "import sample_negatives, build_run_manifest\n"
            "after = list(logging.getLogger().handlers)\n"
            "print(json.dumps({{'same': before == after,\n"
            "                  'n': len(after),\n"
            "                  'cwd': sorted(os.listdir('.'))}}))\n"
        ).format(repo=REPO)
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, "-c", snippet],
                cwd=tmp, capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(proc.stdout.strip().splitlines()[-1])
            self.assertTrue(report["same"], "importing changed the root handlers")
            self.assertEqual(report["n"], 0)
            self.assertEqual(report["cwd"], [], "an import created a file")


class TestNoDeadPaths(unittest.TestCase):
    def test_no_module_level_constant_points_at_a_dead_run_directory(self):
        for module, _ in STAGES:
            path = os.path.join(REPO, module.__name__ + ".py")
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            tree = ast.parse(source)
            for node in tree.body:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                    continue
                with self.subTest(stage=module.STEP_NAME, line=node.lineno):
                    for token in DEAD_PATH_TOKENS:
                        self.assertNotIn(token, value.value)

    def test_no_dead_path_anywhere_in_the_source(self):
        """Including docstrings and help text: an example nobody can run is a lie."""
        for module, _ in STAGES:
            path = os.path.join(REPO, module.__name__ + ".py")
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            with self.subTest(stage=module.STEP_NAME):
                for token in DEAD_PATH_TOKENS:
                    self.assertNotIn(token, source)

    def test_reference_data_defaults_stay_overridable(self):
        """/opt/data is a machine-wide dataset, so it may be a default -- but a flag must exist."""
        self.assertTrue(ri.DEFAULT_VOCABULARY.startswith("/opt/data"))
        self.assertIn("--vocabulary", flags_of(ri.build_arg_parser()))


class TestPreviouslyHardcodedValuesAreFlags(unittest.TestCase):
    def test_repair_scan_budget_is_a_flag_and_reaches_the_sampler(self):
        args = sn.build_arg_parser().parse_args(
            ["--index", "i", "--interactions", "x", "--positives", "p",
             "--out", "o", "--no-pool", "--max-repair-scans", "7"]
        )
        self.assertEqual(args.max_repair_scans, 7)
        default = sn.build_arg_parser().parse_args(
            ["--index", "i", "--interactions", "x", "--positives", "p",
             "--out", "o", "--no-pool"]
        )
        self.assertEqual(default.max_repair_scans, sn.DEFAULT_MAX_REPAIR_SCANS)
        self.assertIn(
            "max_repair_scans=args.max_repair_scans", inspect.getsource(sn._sample)
        )

    def test_ligand_extensions_are_a_flag_and_change_what_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "DB00035.mol2"), "w").close()
            self.assertEqual(pp.load_available_ligands(tmp), {"DB00035"})
            self.assertEqual(
                pp.load_available_ligands(tmp, extensions=[".sdf"]), set()
            )
            # extensions may be given with or without the leading dot
            self.assertEqual(
                pp.load_available_ligands(tmp, extensions=["mol2"]), {"DB00035"}
            )
        args = pp.build_parser().parse_args(
            ["--interactions", "i", "--index", "x", "--cavities", "c",
             "--out", "o", "--ligand-extensions", ".sdf,.pdb"]
        )
        self.assertEqual(args.ligand_extensions, ".sdf,.pdb")

    def test_determinism_seeds_are_a_flag(self):
        args = bii.build_parser().parse_args(
            ["--out", "o", "--determinism-seeds", "3", "5"]
        )
        self.assertEqual(args.determinism_seeds, [3, 5])
        self.assertEqual(
            bii.build_parser().parse_args(["--out", "o"]).determinism_seeds,
            list(bii.DEFAULT_DETERMINISM_SEEDS),
        )

    def test_build_identity_index_still_accepts_critical(self):
        """The suite silences intentionally-failing builds with --log-level CRITICAL."""
        args = bii.build_parser().parse_args(
            ["--out", "o", "--log-level", "CRITICAL"]
        )
        self.assertEqual(args.log_level, "CRITICAL")


def _fixture_inputs(tmp: str):
    """Two one-row manifests: the cheapest real input build_run_manifest accepts."""
    manifests = os.path.join(tmp, "manifests")
    os.makedirs(manifests)
    positives = os.path.join(manifests, "positives_planned.csv")
    drawn = os.path.join(manifests, "negatives_drawn.csv")
    schemas.write_checked(
        pd.DataFrame([{
            "drugbank_id": "DB00035", "accession": "P30518", "gene": "AVPR2",
            "fragment_id": "F1", "cavity_index": 1, "sample_type": "positive",
            "selected_by": "entrez",
        }]),
        positives, "positives_planned",
    )
    schemas.write_checked(
        pd.DataFrame([{
            "drugbank_id": "DB00157", "accession": "P68363", "gene": "TUBA1B",
            "fragment_id": "F1", "cavity_index": 2, "sample_type": "negative",
            "source": "new", "old_job_dir": None, "selected_by": "balanced",
        }]),
        drawn, "negatives_drawn",
    )
    return manifests, positives, drawn


class TestStepDirectoryAndLog(unittest.TestCase):
    """Where the log goes, end to end, on a real (tiny) run."""

    def test_run_dir_puts_the_log_in_the_fixed_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifests, positives, drawn = _fixture_inputs(tmp)
            code = brm.main([
                "--run-dir", tmp,
                "--positives", positives, "--drawn", drawn,
                "--out", os.path.join(manifests, "run_manifest.csv"),
                "--quiet",
            ])
            self.assertEqual(code, 0)
            log = os.path.join(tmp, "steps", "build_run_manifest",
                               "build_run_manifest.log")
            self.assertTrue(os.path.exists(log), os.listdir(tmp))
            with open(log, encoding="utf-8") as handle:
                text = handle.read()
            self.assertIn("resolved configuration", text)
            self.assertIn("run_manifest", text)

    def test_without_run_dir_the_log_follows_the_artifact_not_the_cwd(self):
        """<run>/manifests/x.csv implies <run>; the invocation directory is never used."""
        with tempfile.TemporaryDirectory() as tmp:
            manifests, positives, drawn = _fixture_inputs(tmp)
            cwd = os.path.join(tmp, "elsewhere")
            os.makedirs(cwd)
            here = os.getcwd()
            os.chdir(cwd)
            try:
                code = brm.main([
                    "--positives", positives, "--drawn", drawn,
                    "--out", os.path.join(manifests, "run_manifest.csv"),
                    "--quiet",
                ])
            finally:
                os.chdir(here)
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(os.path.join(
                tmp, "steps", "build_run_manifest", "build_run_manifest.log")))
            self.assertEqual(os.listdir(cwd), [])

    def test_explicit_out_dir_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifests, positives, drawn = _fixture_inputs(tmp)
            chosen = os.path.join(tmp, "chosen")
            code = brm.main([
                "--run-dir", tmp, "--out-dir", chosen,
                "--positives", positives, "--drawn", drawn,
                "--out", os.path.join(manifests, "run_manifest.csv"),
                "--quiet",
            ])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(
                os.path.join(chosen, "build_run_manifest.log")))

    def test_verbose_still_means_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifests, positives, drawn = _fixture_inputs(tmp)
            code = brm.main([
                "--run-dir", tmp, "-v", "--quiet",
                "--positives", positives, "--drawn", drawn,
                "--out", os.path.join(manifests, "run_manifest.csv"),
            ])
            self.assertEqual(code, 0)
            with open(os.path.join(tmp, "steps", "build_run_manifest",
                                   "build_run_manifest.log"), encoding="utf-8") as h:
                self.assertIn("'DEBUG'", h.read())

    def test_a_failing_stage_still_returns_its_exit_code(self):
        """step() re-raises SystemExit; main() must hand the caller the int back."""
        with tempfile.TemporaryDirectory() as tmp:
            code = brm.main([
                "--run-dir", tmp,
                "--positives", os.path.join(tmp, "nope.csv"),
                "--drawn", os.path.join(tmp, "nope2.csv"),
                "--out", os.path.join(tmp, "out.csv"),
                "--quiet",
            ])
            self.assertEqual(code, 2)
            with open(os.path.join(tmp, "steps", "build_run_manifest",
                                   "build_run_manifest.log"), encoding="utf-8") as h:
                self.assertIn("exited with code 2", h.read())


class TestNotifications(unittest.TestCase):
    def test_notify_without_credentials_refuses_at_startup(self):
        """Exit 2 before any work, not a surprise at hour 61 of a long run."""
        with tempfile.TemporaryDirectory() as tmp:
            manifests, positives, drawn = _fixture_inputs(tmp)
            out = os.path.join(manifests, "run_manifest.csv")
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith("PUSHOVER_")}
            with mock.patch.dict(os.environ, env, clear=True):
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    code = brm.main([
                        "--run-dir", tmp, "--notify", "--quiet",
                        "--positives", positives, "--drawn", drawn, "--out", out,
                    ])
            self.assertEqual(code, 2)
            self.assertIn("PUSHOVER_TOKEN", err.getvalue())
            self.assertFalse(os.path.exists(out), "work ran despite the refusal")

    def test_notify_events_are_selectable(self):
        for module, factory in STAGES:
            with self.subTest(stage=module.STEP_NAME):
                action = [a for a in factory()._actions
                          if "--notify-on" in a.option_strings][0]
                self.assertIn("failure", action.choices)
                self.assertIn("all", action.choices)


class TestLoggingIsolation(unittest.TestCase):
    """A stage may configure its own logger; it may not touch anyone else's."""

    def setUp(self):
        root = logging.getLogger()
        self._handlers = list(root.handlers)
        self._level = root.level

    def tearDown(self):
        root = logging.getLogger()
        root.handlers = self._handlers
        root.setLevel(self._level)

    def test_a_run_does_not_add_or_remove_root_handlers(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifests, positives, drawn = _fixture_inputs(tmp)
            before = list(logging.getLogger().handlers)
            brm.main([
                "--run-dir", tmp, "--quiet",
                "--positives", positives, "--drawn", drawn,
                "--out", os.path.join(manifests, "run_manifest.csv"),
            ])
            self.assertEqual(list(logging.getLogger().handlers), before)

    def test_two_runs_do_not_duplicate_log_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifests, positives, drawn = _fixture_inputs(tmp)
            argv = [
                "--run-dir", tmp, "--quiet",
                "--positives", positives, "--drawn", drawn,
                "--out", os.path.join(manifests, "run_manifest.csv"),
            ]
            brm.main(argv)
            brm.main(argv)
            log = os.path.join(tmp, "steps", "build_run_manifest",
                               "build_run_manifest.log")
            with open(log, encoding="utf-8") as handle:
                lines = [l for l in handle if "=== build_run_manifest: start ===" in l]
            # two runs, two start lines -- not four, which is what a stacked
            # handler would produce.
            self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
