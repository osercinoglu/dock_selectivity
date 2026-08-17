"""CLI and runtime-wiring tests for the analysis group.

The two scripts covered here -- ``parse_prepare.py`` (phase 8) and
``pocketmatch_clustering.py`` (the optional pocket-similarity side branch) -- both
used to carry absolute paths to the dead ``2025_06_29_batch_dock`` campaign as
module constants, on a drive that is now read only. These tests pin the things
that made that possible:

* no module-level constant, and no default in the source, names a specific past
  run directory or a specific user's home;
* every value that used to be a constant is reachable as a flag, and parsing sets it;
* importing either module configures no logging and writes no file;
* the whole pass runs on a tiny fixture in a temporary directory.

Nothing here dockes, downloads, or touches ``/media``.
"""

from __future__ import annotations

import ast
import logging
import os
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import parse_prepare as pp  # noqa: E402
import pocketmatch_clustering as pmc  # noqa: E402

PYTHON = sys.executable

#: Substrings that must not appear as a default anywhere in these scripts. The
#: first two are the dead campaign, the last two are one workstation's home.
DEAD_PATH_MARKERS = (
    "/media/onur",
    "2025_06_29_batch_dock",
    "/home/madinasu",
    "/home/onur/experiments",
)


class RuntimeLoggingIsolation(unittest.TestCase):
    """Undo the handlers ``step()`` installs before the temp directory vanishes.

    ``configure_logging`` attaches a FileHandler to the step logger *and* to the
    shared ``dock_selectivity`` logger. Left attached after a tmpdir is deleted,
    that handler raises FileNotFoundError inside every later ``logger.info`` in
    the same interpreter -- including other test files' -- so each test that runs
    a ``main()`` restores exactly the handler list and level it found.
    """

    ISOLATED_LOGGERS = ("parse_prepare", "pocketmatch_clustering", "dock_selectivity")

    def setUp(self):
        super().setUp()
        self._saved_logging = {}
        for name in self.ISOLATED_LOGGERS:
            logger = logging.getLogger(name)
            self._saved_logging[name] = (logger.level, list(logger.handlers))

    def tearDown(self):
        for name, (level, handlers) in self._saved_logging.items():
            logger = logging.getLogger(name)
            for handler in list(logger.handlers):
                if handler not in handlers:
                    logger.removeHandler(handler)
                    try:
                        handler.close()
                    except Exception:  # pragma: no cover - already-dead stream
                        pass
            logger.handlers[:] = handlers
            logger.setLevel(level)
        super().tearDown()


def _string_constants(path):
    """Every string literal in a module, with its line number."""
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


class TestNoDeadRunDirectories(unittest.TestCase):
    """A default pointing at one past campaign is how --run-dir got ignored."""

    def test_sources_contain_no_dead_campaign_path(self):
        for script in ("parse_prepare.py", "pocketmatch_clustering.py"):
            path = os.path.join(REPO_ROOT, script)
            for lineno, text in _string_constants(path):
                for marker in DEAD_PATH_MARKERS:
                    self.assertNotIn(
                        marker, text,
                        "%s:%d still names %s" % (script, lineno, marker))

    def test_no_module_attribute_points_at_a_dead_directory(self):
        for module in (pp, pmc):
            for name in dir(module):
                value = getattr(module, name)
                if not isinstance(value, str):
                    continue
                for marker in DEAD_PATH_MARKERS:
                    self.assertNotIn(marker, value,
                                     "%s.%s = %r" % (module.__name__, name, value))

    def test_base_dir_is_gone_entirely(self):
        # BASE_DIR was read at import time by three derived constants and rebound
        # by main() for only two of them, which is why --run-dir moved half the
        # paths and left the rest on the dead drive.
        self.assertFalse(hasattr(pp, "BASE_DIR"))
        self.assertFalse(hasattr(pp, "DTA_ATLAS_DIR"))
        self.assertFalse(hasattr(pp, "POCKET_CLUSTERS_DIR"))


class TestImportIsQuiet(unittest.TestCase):
    """A module that configures logging on import cannot be used as a library."""

    def test_importing_configures_no_logging_and_writes_no_file(self):
        code = (
            "import logging, os, sys\n"
            "sys.path.insert(0, %r)\n"
            "before = list(logging.getLogger().handlers)\n"
            "import parse_prepare, pocketmatch_clustering\n"
            "after = list(logging.getLogger().handlers)\n"
            "assert before == after, after\n"
            "assert not logging.getLogger('parse_prepare').handlers\n"
            "assert not logging.getLogger('pocketmatch_clustering').handlers\n"
            "assert os.listdir('.') == [], os.listdir('.')\n"
        ) % REPO_ROOT
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([PYTHON, "-c", code], cwd=tmp,
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_root_logger_untouched_by_import_in_this_process(self):
        before = list(logging.getLogger().handlers)
        import importlib
        importlib.import_module("parse_prepare")
        importlib.import_module("pocketmatch_clustering")
        self.assertEqual(before, list(logging.getLogger().handlers))


class TestParsePrepareParser(unittest.TestCase):

    def test_help_exits_zero(self):
        proc = subprocess.run([PYTHON, os.path.join(REPO_ROOT, "parse_prepare.py"),
                               "--help"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--run-dir", proc.stdout)

    def test_run_dir_is_required(self):
        proc = subprocess.run([PYTHON, os.path.join(REPO_ROOT, "parse_prepare.py")],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--run-dir", proc.stderr)

    def test_every_former_constant_is_now_a_flag(self):
        args = pp.build_parser().parse_args([
            "--run-dir", "/runs/r1",
            "--results-dir", "/runs/r1/docking",
            "--out", "/runs/r1/out.parquet",
            "--manifest", "/runs/r1/manifests/run_manifest.csv",
            "--allow-discovery-shortfall",
            "--sample-metadata", "/runs/r1/labels.csv",
            "--cavity-clusters", "/data/clusters.csv",
            "--ic50-mapping", "/runs/r1/ic50.csv",
            "--pocket-clusters", "/runs/r1/pockets",
            "--dta-atlas-dir", "/data/dta",
            "--dta-batch-size", "17",
            "--dta-workers", "3",
            "--no-dta-parallel",
            "--search-workers", "5",
            "--load-workers", "4",
            "--file-cap", "99",
        ])
        self.assertEqual(args.results_dir, "/runs/r1/docking")
        self.assertEqual(args.out, "/runs/r1/out.parquet")
        self.assertEqual(args.manifest, "/runs/r1/manifests/run_manifest.csv")
        self.assertTrue(args.allow_discovery_shortfall)
        self.assertEqual(args.sample_metadata, "/runs/r1/labels.csv")
        self.assertEqual(args.cavity_clusters, "/data/clusters.csv")
        self.assertEqual(args.ic50_mapping, "/runs/r1/ic50.csv")
        self.assertEqual(args.pocket_clusters, "/runs/r1/pockets")
        self.assertEqual(args.dta_atlas_dir, "/data/dta")
        self.assertEqual(args.dta_batch_size, 17)
        self.assertEqual(args.dta_workers, 3)
        self.assertTrue(args.no_dta_parallel)
        self.assertEqual(args.search_workers, 5)
        self.assertEqual(args.load_workers, 4)
        self.assertEqual(args.file_cap, 99)

    def test_defaults_follow_the_run_directory(self):
        args = pp.build_parser().parse_args(["--run-dir", "/runs/r2"])
        pp.resolve_paths(args)
        self.assertEqual(args.results_dir, "/runs/r2/docking_results")
        self.assertEqual(args.out, os.path.join("/runs/r2", pp.OUTPUT_PARQUET_BASENAME))
        self.assertEqual(args.sample_metadata,
                         "/runs/r2/manifests/run_manifest.csv")
        self.assertEqual(args.ic50_mapping, "/runs/r2/ic50_mapping.csv")
        self.assertEqual(args.pocket_clusters,
                         "/runs/r2/pocketmatch_results/pocket_clusters")
        # Optional annotations stay unset rather than guessing a location.
        self.assertIsNone(args.dta_atlas_dir)

    def test_file_cap_defaults_to_no_cap(self):
        args = pp.build_parser().parse_args(["--run-dir", "/runs/r3"])
        self.assertIsNone(args.file_cap,
                          "a cap here silently truncates the analysis")

    def test_runtime_flags_are_present(self):
        args = pp.build_parser().parse_args(
            ["--run-dir", "/runs/r4", "--log-level", "DEBUG", "--quiet",
             "--notify-on", "success"])
        self.assertEqual(args.log_level, "DEBUG")
        self.assertTrue(args.quiet)
        self.assertEqual(args.notify_on, ["success"])


JOB_A = "DB00035_AVPR2_P30518_cavity_1"
JOB_B = "DB00035_ATP1A1_P05023_cavity_1"
JOB_C = "DB00091_CAMLG_P08F94_cavity_3"


class TestDiscoveryAgainstManifest(unittest.TestCase):
    """The check the old --manifest flag promised and never performed."""

    def _manifest(self, tmp, job_ids):
        path = os.path.join(tmp, "run_manifest.csv")
        pd.DataFrame({"job_id": job_ids}).to_csv(path, index=False)
        return path

    def test_shortfall_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            found = {"final_results.csv": [os.path.join(tmp, JOB_A, "final_results.csv")]}
            manifest = self._manifest(tmp, [JOB_A, JOB_B, JOB_C])
            with self.assertRaises(pp.DiscoveryShortfall) as ctx:
                pp.check_discovery_against_manifest(found, manifest)
            self.assertIn("2 jobs are missing", str(ctx.exception))

    def test_shortfall_can_be_allowed_deliberately(self):
        with tempfile.TemporaryDirectory() as tmp:
            found = {"final_results.csv": [os.path.join(tmp, JOB_A, "final_results.csv")]}
            manifest = self._manifest(tmp, [JOB_A, JOB_B])
            discovered, expected = pp.check_discovery_against_manifest(
                found, manifest, allow_shortfall=True)
            self.assertEqual((discovered, expected), (1, 2))

    def test_complete_discovery_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            found = {"final_results.csv": [
                os.path.join(tmp, JOB_A, "final_results.csv"),
                os.path.join(tmp, JOB_B, "final_results.csv"),
            ]}
            manifest = self._manifest(tmp, [JOB_A, JOB_B])
            self.assertEqual(
                pp.check_discovery_against_manifest(found, manifest), (2, 2))

    def test_per_tool_results_count_towards_their_job_not_their_tool_folder(self):
        # <job>/smina/results.csv must be attributed to <job>, or the discovered
        # set fills up with the string "smina" and the check is meaningless.
        with tempfile.TemporaryDirectory() as tmp:
            found = {"results.csv": [
                os.path.join(tmp, JOB_A, "smina", "results.csv"),
                os.path.join(tmp, JOB_A, "gnina", "results.csv"),
                os.path.join(tmp, JOB_B, "smina", "results.csv"),
            ]}
            manifest = self._manifest(tmp, [JOB_A, JOB_B])
            self.assertEqual(
                pp.check_discovery_against_manifest(found, manifest), (2, 2))


class TestSampleMetadataVocabularies(unittest.TestCase):
    """--sample-metadata defaults to run_manifest.csv, so both spellings must work."""

    def test_run_manifest_vocabulary(self):
        frame = pd.DataFrame({
            "job_id": ["j1"], "drugbank_id": ["DB00035"], "accession": ["P30518"],
            "gene": ["AVPR2"], "fragment_id": ["F1"], "cavity_index": [1],
            "sample_type": ["positive"],
        })
        out = pp.normalize_sample_metadata(frame)
        self.assertEqual(set(out.columns),
                         {"uniprot_id", "cavity_index", "drugbank_id",
                          "sample_type", "Gene_Name"})
        self.assertEqual(out.iloc[0]["uniprot_id"], "P30518")
        self.assertEqual(out.iloc[0]["Gene_Name"], "AVPR2")

    def test_legacy_vocabulary(self):
        frame = pd.DataFrame({
            "UniProt_ID": ["P30518"], "drugbank_id": ["DB00035"],
            "Cavity_Index": [1], "sample_type": ["positive"], "Gene_Name": ["AVPR2"],
        })
        out = pp.normalize_sample_metadata(frame)
        self.assertEqual(out.iloc[0]["uniprot_id"], "P30518")

    def test_an_unknown_vocabulary_is_an_error_not_a_shrug(self):
        frame = pd.DataFrame({"something": [1]})
        with self.assertRaises(KeyError):
            pp.normalize_sample_metadata(frame)


class TestParsePrepareSmokeRun(RuntimeLoggingIsolation):
    """A whole pass over two job directories, entirely inside a tmpdir."""

    TOOLS = ("SMINA", "GNINA", "GOLD", "LEDOCK")

    def _write_run(self, tmp):
        results = os.path.join(tmp, "docking_results")
        jobs = ["DB00035_AVPR2_P30518_cavity_1", "DB00035_ATP1A1_P05023_cavity_1"]
        for job in jobs:
            job_dir = os.path.join(results, job)
            os.makedirs(job_dir)
            rows = []
            for i, tool_a in enumerate(self.TOOLS):
                tool_b = self.TOOLS[(i + 1) % len(self.TOOLS)]
                rows.append({"Tool1": tool_a, "Tool2": tool_b, "RMSD": 1.0 + i,
                             "Score1": -7.0 - i, "Score2": -6.5 - i})
            pd.DataFrame(rows).to_csv(
                os.path.join(job_dir, "final_results.csv"), index=False)

        labels = os.path.join(tmp, "manifests", "run_manifest.csv")
        os.makedirs(os.path.dirname(labels))
        pd.DataFrame({
            "job_id": jobs,
            "drugbank_id": ["DB00035", "DB00035"],
            "accession": ["P30518", "P05023"],
            "gene": ["AVPR2", "ATP1A1"],
            "fragment_id": ["F1", "F1"],
            "cavity_index": [1, 1],
            "sample_type": ["positive", "negative_balanced"],
        }).to_csv(labels, index=False)
        return results, labels

    def test_runs_end_to_end_and_writes_the_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            results, labels = self._write_run(tmp)
            out = os.path.join(tmp, "analysis", "combined.parquet")
            rc = pp.main([
                "--run-dir", tmp,
                "--results-dir", results,
                "--sample-metadata", labels,
                "--manifest", labels,
                "--out", out,
                "--cavity-clusters", os.path.join(tmp, "no-such-clusters.csv"),
                "--search-workers", "2",
                "--load-workers", "2",
                "--quiet",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(out), os.listdir(tmp))
            frame = pd.read_parquet(out)
            self.assertGreater(len(frame), 0)
            self.assertIn("sample_type", frame.columns)
            # The step wrapper gave the run its own folder and log.
            log = os.path.join(tmp, "steps", "parse_prepare", "parse_prepare.log")
            self.assertTrue(os.path.exists(log))
            self.assertGreater(os.path.getsize(log), 0)

    def test_a_missing_results_tree_is_an_error_not_an_empty_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as ctx:
                pp.main(["--run-dir", tmp, "--quiet"])
            self.assertEqual(ctx.exception.code, 2)


class TestPocketmatchClusteringParser(RuntimeLoggingIsolation):

    def test_module_parses(self):
        # It once shipped a stray 'tmux' token that stopped the file compiling.
        proc = subprocess.run(
            [PYTHON, "-m", "py_compile",
             os.path.join(REPO_ROOT, "pocketmatch_clustering.py")],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_help_exits_zero(self):
        proc = subprocess.run(
            [PYTHON, os.path.join(REPO_ROOT, "pocketmatch_clustering.py"), "--help"],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--csv-dir", proc.stdout)

    def test_run_dir_is_required(self):
        proc = subprocess.run(
            [PYTHON, os.path.join(REPO_ROOT, "pocketmatch_clustering.py")],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--run-dir", proc.stderr)

    def test_every_former_constant_is_now_a_flag(self):
        args = pmc.build_parser().parse_args([
            "--run-dir", "/runs/r1",
            "--csv-dir", "/runs/r1/sim",
            "--parquet-dir", "/runs/r1/pq",
            "--output-dir", "/runs/r1/clusters",
            "--max-memory-mb", "111",
            "--min-available-mb", "22",
            "--csv-chunk-rows", "33",
            "--max-files", "4",
            "--resolution", "0.9",
            "--pmax-thresholds", "0.4", "0.6",
        ])
        self.assertEqual(args.csv_dir, "/runs/r1/sim")
        self.assertEqual(args.parquet_dir, "/runs/r1/pq")
        self.assertEqual(args.output_dir, "/runs/r1/clusters")
        self.assertEqual(args.max_memory_mb, 111)
        self.assertEqual(args.min_available_mb, 22)
        self.assertEqual(args.csv_chunk_rows, 33)
        self.assertEqual(args.max_files, 4)
        self.assertAlmostEqual(args.resolution, 0.9)
        self.assertEqual(args.pmax_thresholds, [0.4, 0.6])

    def test_paths_default_to_none_so_the_run_dir_supplies_them(self):
        args = pmc.build_parser().parse_args(["--run-dir", "/runs/r2"])
        self.assertIsNone(args.csv_dir)
        self.assertIsNone(args.parquet_dir)
        self.assertIsNone(args.output_dir)

    def test_missing_similarity_directory_is_a_refusal(self):
        # SystemExit, not a non-zero return: step() only reports a failure (and
        # only notifies) when the body raises.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as ctx:
                pmc.main(["--run-dir", tmp, "--quiet"])
            self.assertEqual(ctx.exception.code, 2)


class TestPocketmatchClusteringSmokeRun(RuntimeLoggingIsolation):

    def test_clusters_a_tiny_similarity_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_dir = os.path.join(tmp, "pocketmatch_results",
                                   "PocketMatch_similarities")
            os.makedirs(csv_dir)
            pd.DataFrame({
                "Pocket1": ["a", "b", "a", "c", "d"],
                "Pocket2": ["b", "a", "a", "d", "c"],
                "Pmax": [0.9, 0.9, 1.0, 0.8, 0.8],
            }).to_csv(os.path.join(csv_dir, "similarities_part_001.csv"), index=False)

            rc = pmc.main(["--run-dir", tmp, "--pmax-thresholds", "0.5", "--quiet"])
            self.assertEqual(rc, 0)

            clusters = os.path.join(tmp, "pocketmatch_results", "pocket_clusters",
                                    "pocket_louvain_clusters_0p5.csv")
            self.assertTrue(os.path.exists(clusters))
            frame = pd.read_csv(clusters)
            self.assertEqual(set(frame["Pocket"]), {"a", "b", "c", "d"})
            # The parquet cache is a named directory, not one derived from the
            # (possibly read-only) input directory's parent.
            self.assertTrue(os.path.isdir(
                os.path.join(tmp, "pocketmatch_results", "parquet_data")))
            log = os.path.join(tmp, "steps", "pocketmatch_clustering",
                               "pocketmatch_clustering.log")
            self.assertTrue(os.path.exists(log))


if __name__ == "__main__":
    unittest.main()
