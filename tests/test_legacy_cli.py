"""CLI and runtime contract for the legacy group.

Three scripts, two lifecycles. ``identify_required_structures.py`` and
``generate_negative_samples.py`` are superseded and must stay refusable: a bare
run exits 2, says why, and leaves nothing behind. ``get_PM_scores.py`` is the
live pocket-similarity side branch.

What all three now share: every path, count and timeout is a flag, no default
names a past run directory or a specific user's home, importing the module
configures no logging and writes no file, and main() runs inside the shared
runtime's step() so the log lands in the step's own directory.
"""

from __future__ import annotations

import ast
import io
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import generate_negative_samples as gns  # noqa: E402
import get_PM_scores as pm  # noqa: E402
import identify_required_structures as irs  # noqa: E402

#: A default naming one of these is a default that only ever worked on one
#: machine, on one day.
DEAD_PATH_FRAGMENTS = (
    "2025_06_29_batch_dock",
    "/home/onur/experiments",
    "/media/onur/Elements",
    "/home/ssahin",
)

MODULES = (
    (irs, "identify_required_structures.py"),
    (gns, "generate_negative_samples.py"),
    (pm, "get_PM_scores.py"),
)


def source_of(name):
    with open(os.path.join(REPO, name), encoding="utf-8") as handle:
        return handle.read()


class LoggingIsolation(unittest.TestCase):
    """Restore the logging tree after every test.

    The suite imports seven prep-era scripts that still stack handlers on the
    root logger; these tests must observe only what they configure themselves,
    and must hand back exactly what they found.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        root = logging.getLogger()
        self._root_handlers = list(root.handlers)
        self._root_level = root.level
        self._touched = {}
        self.addCleanup(self._restore_logging)
        for name in ("dock_selectivity", irs.STEP_NAME, gns.STEP_NAME, pm.STEP_NAME):
            self._touched[name] = (logging.getLogger(name).level,
                                   list(logging.getLogger(name).handlers))
        root.handlers[:] = []
        self.console = io.StringIO()
        silencer = mock.patch.object(sys, "stderr", self.console)
        silencer.start()
        self.addCleanup(silencer.stop)

    def _restore_logging(self):
        for name, (level, handlers) in self._touched.items():
            target = logging.getLogger(name)
            for handler in list(target.handlers):
                if handler in handlers:
                    continue
                target.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass
            target.handlers[:] = handlers
            target.setLevel(level)
        root = logging.getLogger()
        root.handlers[:] = self._root_handlers
        root.setLevel(self._root_level)


# --- parsers ------------------------------------------------------------------


class TestParsersBuild(unittest.TestCase):
    def test_each_module_exposes_build_parser(self):
        for module, name in MODULES:
            with self.subTest(script=name):
                self.assertTrue(callable(getattr(module, "build_parser", None)),
                                "%s has no build_parser()" % name)

    def test_help_exits_zero(self):
        """--help must render: an unescaped %% in a help string raises here."""
        for _, name in MODULES:
            with self.subTest(script=name):
                proc = subprocess.run([sys.executable, os.path.join(REPO, name), "--help"],
                                      cwd=REPO, capture_output=True, text=True, timeout=120)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn("--out-dir", proc.stdout)
                self.assertIn("--notify", proc.stdout)

    def test_runtime_flags_are_present_exactly_once(self):
        for module, name in MODULES:
            with self.subTest(script=name):
                parser = module.build_parser()
                flags = [option
                         for action in parser._actions
                         for option in action.option_strings]
                for expected in ("--out-dir", "--log-file", "--log-level",
                                 "--quiet", "--notify", "--notify-on", "--notify-title"):
                    self.assertEqual(flags.count(expected), 1,
                                     "%s: %s appears %d times"
                                     % (name, expected, flags.count(expected)))


# --- every previously hardcoded value is now a flag ---------------------------


class TestSupersededStagesTakeTheirPaths(unittest.TestCase):
    """The seven/eight path constants each stage carried are now flags."""

    def test_identify_required_structures_paths_are_flags(self):
        args = irs.build_parser().parse_args([
            "--edges", "/e.tsv", "--drugs", "/d.csv",
            "--uniprot-mapping", "/u.csv", "--cavity-mapping", "/c.csv",
            "--ligands", "/lig", "--out", "/o.csv", "--run-dir", "/run",
        ])
        self.assertEqual(
            (args.edges, args.drugs, args.uniprot_mapping, args.cavity_mapping,
             args.ligands, args.out, args.run_dir),
            ("/e.tsv", "/d.csv", "/u.csv", "/c.csv", "/lig", "/o.csv", "/run"))

    def test_generate_negative_samples_paths_and_seed_are_flags(self):
        args = gns.build_parser().parse_args([
            "--edges", "/e.tsv", "--drugs", "/d.csv",
            "--uniprot-mapping", "/u.csv", "--cavity-mapping", "/c.csv",
            "--ligands", "/lig", "--out", "/o.csv",
            "--metadata-out", "/m.json", "--seed", "7",
        ])
        self.assertEqual(
            (args.edges, args.drugs, args.uniprot_mapping, args.cavity_mapping,
             args.ligands, args.out, args.metadata_out, args.seed),
            ("/e.tsv", "/d.csv", "/u.csv", "/c.csv", "/lig", "/o.csv", "/m.json", 7))

    def test_the_seed_is_no_longer_a_buried_literal(self):
        """A seed nailed into the source is not a reproducible draw, it is a rumour.

        Checked on the AST, not the text: the comment that explains the old
        global np.random.seed(42) is documentation, and must stay legal.
        """
        tree = ast.parse(source_of("generate_negative_samples.py"))
        seeded = [node.lineno for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and getattr(node.func, "attr", None) == "seed"]
        self.assertEqual(seeded, [], "a global seed call survives at %r" % seeded)
        self.assertEqual(gns.build_parser().parse_args([]).seed, gns.DEFAULT_SEED)


class TestPocketMatchTunablesAreFlags(unittest.TestCase):
    BASE = ["--cavities", "/cav", "--output-dir", "/out", "--pm-dir", "/pm"]

    def test_every_timeout_is_reachable(self):
        args = pm.build_parser().parse_args(self.BASE + [
            "--residue-timeout", "11", "--cabbage-timeout", "12",
            "--pm-timeout", "13", "--block-timeout", "14",
            "--block-phase-timeout", "15", "--comparison-timeout", "16",
            "--comparison-phase-timeout", "17", "--io-timeout", "18",
            "--thread-cap", "19",
        ])
        self.assertEqual(
            [args.residue_timeout, args.cabbage_timeout, args.pm_timeout,
             args.block_timeout, args.block_phase_timeout, args.comparison_timeout,
             args.comparison_phase_timeout, args.io_timeout, args.thread_cap],
            [11, 12, 13, 14, 15, 16, 17, 18, 19])

    def test_apply_runtime_tunables_rebinds_the_module_constants(self):
        before = [pm.PM_TIMEOUT, pm.BLOCK_TIMEOUT, pm.DEFAULT_THREAD_CAP]
        self.addCleanup(_restore_pm_constants, before)
        args = pm.build_parser().parse_args(
            self.BASE + ["--pm-timeout", "42", "--block-timeout", "43", "--thread-cap", "2"])
        changed = pm.apply_runtime_tunables(args)
        self.assertEqual(pm.PM_TIMEOUT, 42)
        self.assertEqual(pm.BLOCK_TIMEOUT, 43)
        self.assertEqual(pm.DEFAULT_THREAD_CAP, 2)
        self.assertEqual(changed["PM_TIMEOUT"], 42)

    def test_typeB_default_follows_the_flag(self):
        """The default parameter must read the tunable, not a frozen 600."""
        before = [pm.PM_TIMEOUT, pm.BLOCK_TIMEOUT, pm.DEFAULT_THREAD_CAP]
        self.addCleanup(_restore_pm_constants, before)
        pm.PM_TIMEOUT = 5
        with tempfile.TemporaryDirectory() as tmp:
            chunk = os.path.join(tmp, "a.cabbage")
            full = os.path.join(tmp, "b.cabbage")
            for path in (chunk, full):
                with open(path, "w") as handle:
                    handle.write("")
            with mock.patch.object(pm.subprocess, "run") as run:
                run.side_effect = pm.subprocess.TimeoutExpired(cmd="x", timeout=5)
                with self.assertRaises(RuntimeError) as caught:
                    pm.run_pocketmatch_typeB(tmp, chunk, full)
            self.assertEqual(run.call_args[1]["timeout"], 5)
        self.assertIn("5s", str(caught.exception))

    def test_single_dash_spellings_still_work(self):
        args = pm.build_parser().parse_args(
            ["-cavities", "/cav", "-output", "/out", "-pm_dir", "/pm",
             "-af_structures", "/af", "-threads", "4"])
        self.assertEqual((args.cavities, args.output, args.pm_dir,
                          args.af_structures, args.threads),
                         ("/cav", "/out", "/pm", "/af", 4))


def _restore_pm_constants(values):
    pm.PM_TIMEOUT, pm.BLOCK_TIMEOUT, pm.DEFAULT_THREAD_CAP = values


# --- no dead defaults ---------------------------------------------------------


class TestNoDeadPathsRemain(unittest.TestCase):
    def test_no_module_attribute_names_a_dead_run_directory(self):
        for module, name in MODULES:
            for attr in dir(module):
                value = getattr(module, attr)
                if not isinstance(value, str):
                    continue
                for fragment in DEAD_PATH_FRAGMENTS:
                    with self.subTest(script=name, attr=attr):
                        self.assertNotIn(fragment, value)

    def test_no_string_constant_in_the_source_names_one_either(self):
        """Including argparse defaults and epilogs, which dir() cannot see."""
        for _, name in MODULES:
            tree = ast.parse(source_of(name), filename=name)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for fragment in DEAD_PATH_FRAGMENTS:
                        with self.subTest(script=name, line=node.lineno):
                            self.assertNotIn(fragment, node.value)

    def test_defaults_that_remain_are_machine_wide_reference_data(self):
        for module in (irs, gns):
            for attr in ("DEFAULT_DRUG_TO_PROTEIN_TSV",
                         "DEFAULT_SMALL_MOLECULE_DRUGS_CSV",
                         "DEFAULT_LIGAND_SDF_FOLDER"):
                with self.subTest(module=module.STEP_NAME, attr=attr):
                    self.assertTrue(getattr(module, attr).startswith("/opt/data/"))


# --- import is silent ---------------------------------------------------------


IMPORT_PROBE = """
import json, logging, os, sys
sys.path.insert(0, {repo!r})
before = list(logging.getLogger().handlers)
import {module}
after = list(logging.getLogger().handlers)
print(json.dumps({{"added": len(after) - len(before), "files": sorted(os.listdir("."))}}))
"""


class TestImportIsQuiet(unittest.TestCase):
    """A module that configures logging on import is a module you cannot import."""

    def _probe(self, module_name):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, "-c", IMPORT_PROBE.format(repo=REPO, module=module_name)],
                cwd=tmp, capture_output=True, text=True, timeout=180)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            import json
            return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_importing_adds_no_root_handler_and_writes_no_file(self):
        for module_name in ("identify_required_structures", "generate_negative_samples",
                            "get_PM_scores"):
            with self.subTest(module=module_name):
                result = self._probe(module_name)
                self.assertEqual(result["added"], 0,
                                 "%s configured the root logger on import" % module_name)
                self.assertEqual(result["files"], [],
                                 "%s created %s on import"
                                 % (module_name, result["files"]))


# --- the refusal still refuses, and still leaves nothing -----------------------


class TestRefusalLeavesNothingBehind(unittest.TestCase):
    def test_bare_run_exits_two_and_creates_nothing(self):
        for name in ("identify_required_structures.py", "generate_negative_samples.py"):
            with self.subTest(script=name), tempfile.TemporaryDirectory() as tmp:
                proc = subprocess.run([sys.executable, os.path.join(REPO, name)],
                                      cwd=tmp, capture_output=True, text=True, timeout=180)
                self.assertEqual(proc.returncode, 2)
                self.assertIn("superseded", proc.stdout + proc.stderr)
                self.assertEqual(os.listdir(tmp), [],
                                 "a refused run must not create a log or an out-dir")

    def test_the_module_docstring_carries_the_deprecation_banner(self):
        for module, name in ((irs, "identify_required_structures.py"),
                             (gns, "generate_negative_samples.py")):
            with self.subTest(script=name):
                doc = module.__doc__ or ""
                self.assertIn("DEPRECATED", doc)
                self.assertIn("SUPERSEDED", doc)
                for replacement in module.REPLACEMENT_STAGES:
                    self.assertIn(replacement, doc)


# --- smoke runs on tiny fixtures ----------------------------------------------


def write_legacy_fixture(root):
    """Two drugs, two genes, one cavity each, one SDF each."""
    ligands = os.path.join(root, "ligands")
    os.makedirs(ligands)
    for drug in ("DB00001", "DB00002"):
        with open(os.path.join(ligands, drug + ".sdf"), "w") as handle:
            handle.write("$$$$\n")

    edges = os.path.join(root, "edges.tsv")
    with open(edges, "w") as handle:
        handle.write("node_1\tnode_2_name\tnode_2\n")
        handle.write("DB00001\tGENEA\t111\n")
        handle.write("DB00002\tGENEB\t222\n")

    drugs = os.path.join(root, "drugs.csv")
    with open(drugs, "w") as handle:
        handle.write("DrugBank ID\nDB00001\nDB00002\n")

    uniprot = os.path.join(root, "uniprot_gene_mapping.csv")
    with open(uniprot, "w") as handle:
        handle.write("Entry,Mapped_Gene_Name\nP11111,GENEA\nP22222,GENEB\n")

    cavities = os.path.join(root, "cavity_mapping.csv")
    with open(cavities, "w") as handle:
        handle.write("UniProt_ID,Fragment_ID,Cavity_Index,Pocket_PDB,Gene_Name\n")
        handle.write("P11111,F1,1,/tmp/p1.pdb,GENEA\n")
        handle.write("P22222,F1,1,/tmp/p2.pdb,GENEB\n")
    return dict(ligands=ligands, edges=edges, drugs=drugs,
                uniprot=uniprot, cavities=cavities)


class TestLegacySmokeRuns(LoggingIsolation):
    def test_identify_required_structures_runs_and_logs_into_its_out_dir(self):
        fx = write_legacy_fixture(self.dir)
        out_csv = os.path.join(self.dir, "required_structures.csv")
        out_dir = os.path.join(self.dir, "step")
        code = irs.main([
            "--legacy-gene-resolution",
            "--edges", fx["edges"], "--drugs", fx["drugs"],
            "--uniprot-mapping", fx["uniprot"], "--cavity-mapping", fx["cavities"],
            "--ligands", fx["ligands"], "--out", out_csv, "--out-dir", out_dir,
        ])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(out_csv))
        log = os.path.join(out_dir, "identify_required_structures.log")
        self.assertTrue(os.path.exists(log), "the step log must live in --out-dir")
        with open(log, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("CORRUPTED symbol resolver", text)

        import pandas as pd
        frame = pd.read_csv(out_csv)
        self.assertEqual(sorted(frame["UniProt_ID"]), ["P11111", "P22222"])

    def test_a_missing_ligand_folder_is_an_error_not_an_empty_result(self):
        fx = write_legacy_fixture(self.dir)
        out_dir = os.path.join(self.dir, "step2")
        with self.assertRaises(SystemExit) as caught:
            irs.main([
                "--legacy-gene-resolution",
                "--edges", fx["edges"], "--drugs", fx["drugs"],
                "--uniprot-mapping", fx["uniprot"], "--cavity-mapping", fx["cavities"],
                "--ligands", os.path.join(self.dir, "nope"),
                "--out", os.path.join(self.dir, "r.csv"), "--out-dir", out_dir,
            ])
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "r.csv")))

    def test_generate_negative_samples_runs_and_honours_the_seed(self):
        fx = write_legacy_fixture(self.dir)
        out_csv = os.path.join(self.dir, "with_negatives.csv")
        meta = os.path.join(self.dir, "meta.json")
        out_dir = os.path.join(self.dir, "gns")
        code = gns.main([
            "--legacy-gene-resolution",
            "--edges", fx["edges"], "--drugs", fx["drugs"],
            "--uniprot-mapping", fx["uniprot"], "--cavity-mapping", fx["cavities"],
            "--ligands", fx["ligands"], "--out", out_csv,
            "--metadata-out", meta, "--seed", "7", "--out-dir", out_dir,
        ])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(
            os.path.join(out_dir, "generate_negative_samples.log")))

        import json
        import pandas as pd
        with open(meta, encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.assertEqual(metadata["seed"], 7)
        self.assertEqual(metadata["strategy"]["superseded_by"],
                         list(gns.REPLACEMENT_STAGES))

        frame = pd.read_csv(out_csv)
        self.assertEqual((frame["sample_type"] == "positive").sum(), 2)
        negatives = frame[frame["sample_type"] == "negative_balanced"]
        self.assertEqual(len(negatives), 2)
        # A negative must never be a pair the edges file already asserts.
        drawn = set(zip(negatives["drugbank_id"], negatives["UniProt_ID"]))
        self.assertNotIn(("DB00001", "P11111"), drawn)
        self.assertNotIn(("DB00002", "P22222"), drawn)


class TestPocketMatchRefusesMissingInputs(LoggingIsolation):
    def test_a_missing_cavities_directory_exits_two_and_still_writes_a_log(self):
        out_dir = os.path.join(self.dir, "pmstep")
        with self.assertRaises(SystemExit) as caught:
            pm.main(["--cavities", os.path.join(self.dir, "absent"),
                     "--output-dir", os.path.join(self.dir, "results"),
                     "--pm-dir", self.dir,
                     "--out-dir", out_dir])
        self.assertEqual(caught.exception.code, 2)
        log = os.path.join(out_dir, "get_PM_scores.log")
        self.assertTrue(os.path.exists(log))
        with open(log, encoding="utf-8") as handle:
            self.assertIn("Cavities directory not found", handle.read())

    def test_the_log_defaults_under_the_output_directory_not_the_cwd(self):
        results = os.path.join(self.dir, "results")
        with self.assertRaises(SystemExit):
            pm.main(["--cavities", os.path.join(self.dir, "absent"),
                     "--output-dir", results, "--pm-dir", self.dir])
        self.assertTrue(os.path.exists(
            os.path.join(results, "steps", "get_PM_scores", "get_PM_scores.log")))


if __name__ == "__main__":
    unittest.main()
