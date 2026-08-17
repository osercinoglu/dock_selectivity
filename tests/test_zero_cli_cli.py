"""The three scripts that used to have no CLI at all now have a complete one.

``create_ic50_mapping.py``, ``split_sdf.py`` and ``download_cavityspace_structures.py``
configured themselves by module constants: paths (two of them pointing into a
dead, read-only campaign directory), thread counts, a service URL, the ETKDG seed
that fixes every ligand's geometry. These tests pin the replacement: every one of
those values is reachable as a flag, nothing still names a past run directory,
importing the modules configures no logging and writes no file, and each script
runs end to end on a fixture small enough to build in a temp directory.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import create_ic50_mapping as cim  # noqa: E402
import download_cavityspace_structures as dcs  # noqa: E402
import split_sdf as ssdf  # noqa: E402

MODULES = (cim, ssdf, dcs)

#: Strings that must never reappear in these files: a specific past campaign, and
#: a specific user's home. Both were live defaults before this change.
FORBIDDEN_PATH_FRAGMENTS = ("/media/", "2025_06_29", "/home/onur", "/home/ssahin")


class LoggingIsolation(unittest.TestCase):
    """Restore every logger these scripts configure, so other tests are unaffected.

    ``step()`` attaches handlers to the step logger and to the ``dock_selectivity``
    package logger and sets their levels; a test that leaves those behind changes
    what the rest of the suite prints.
    """

    LOGGERS = ("create_ic50_mapping", "split_sdf", "download_cavityspace_structures",
               "dock_selectivity")

    def setUp(self):
        self._saved = []
        for name in self.LOGGERS:
            logger = logging.getLogger(name)
            self._saved.append((logger, logger.level, list(logger.handlers)))
        self.addCleanup(self._restore)

    def _restore(self):
        for logger, level, handlers in self._saved:
            for handler in list(logger.handlers):
                if handler not in handlers:
                    logger.removeHandler(handler)
                    try:
                        handler.close()
                    except Exception:
                        pass
            logger.handlers = list(handlers)
            logger.setLevel(level)


def parse(module, argv):
    """Parse ``argv`` with a module's own parser, keeping argparse output quiet."""
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return module.build_parser().parse_args(argv)


class TestParsersBuild(unittest.TestCase):
    def test_every_module_exposes_a_parser_and_a_main(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                self.assertTrue(callable(module.build_parser))
                self.assertTrue(callable(module.main))

    def test_help_exits_zero(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                proc = subprocess.run(
                    [sys.executable, os.path.join(REPO, module.__name__ + ".py"), "--help"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=REPO)
                self.assertEqual(proc.returncode, 0, proc.stderr.decode())
                self.assertIn(b"--out-dir", proc.stdout)

    def test_runtime_flags_are_wired_in(self):
        """Every script must carry the shared runtime surface, not its own."""
        expected = ("out_dir", "log_file", "log_level", "quiet", "notify",
                    "notify_on", "notify_title")
        cases = (
            (cim, ["--uniprot-mapping", "u.csv", "--out", "o.csv"]),
            (ssdf, ["--in-sdf", "in.sdf", "--out", "ligands"]),
            (dcs, []),
        )
        for module, argv in cases:
            args = parse(module, argv)
            for name in expected:
                with self.subTest(module=module.__name__, flag=name):
                    self.assertTrue(hasattr(args, name))


class TestFormerConstantsAreFlags(unittest.TestCase):
    def test_ic50_paths_are_all_flags(self):
        args = parse(cim, ["--uniprot-mapping", "/tmp/idx.parquet", "--out", "/tmp/ic50.csv",
                           "--ttd-targets", "/tmp/t.txt", "--ttd-activity", "/tmp/a.txt",
                           "--drug-links", "/tmp/links.csv"])
        self.assertEqual(args.uniprot_mapping, "/tmp/idx.parquet")
        self.assertEqual(args.out, "/tmp/ic50.csv")
        self.assertEqual(args.ttd_targets, "/tmp/t.txt")
        self.assertEqual(args.ttd_activity, "/tmp/a.txt")
        self.assertEqual(args.drug_links, "/tmp/links.csv")

    def test_ic50_requires_its_input_and_its_output(self):
        for argv in ([], ["--out", "x.csv"], ["--uniprot-mapping", "u.csv"]):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    parse(cim, argv)

    def test_split_sdf_exposes_geometry_parameters(self):
        args = parse(ssdf, ["--in-sdf", "a.sdf", "--out", "ligands",
                            "--seed", "7", "--max-iters", "25", "--remove-hydrogens"])
        self.assertEqual(args.in_sdf, "a.sdf")
        self.assertEqual(args.out, "ligands")
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.max_iters, 25)
        self.assertTrue(args.remove_hydrogens)

    def test_split_sdf_defaults_match_the_old_literals(self):
        args = parse(ssdf, ["--in-sdf", "a.sdf", "--out", "ligands"])
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.max_iters, 1000)
        self.assertFalse(args.remove_hydrogens)

    def test_split_sdf_requires_input_and_output(self):
        for argv in ([], ["--in-sdf", "a.sdf"], ["--out", "ligands"]):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    parse(ssdf, argv)

    def test_cavityspace_settings_are_all_flags(self):
        args = parse(dcs, ["--cavity-mapping", "cm.csv", "--structures-dir", "structs",
                           "--out", "map.csv", "--threads", "3", "--db-version", "v2",
                           "--request-delay", "0.25", "--timeout", "5",
                           "--max-retries", "1", "--base-url", "http://example.org/x"])
        self.assertEqual(args.cavity_mapping, "cm.csv")
        self.assertEqual(args.structures_dir, "structs")
        self.assertEqual(args.out, "map.csv")
        self.assertEqual(args.threads, 3)
        self.assertEqual(args.db_version, "v2")
        self.assertEqual(args.request_delay, 0.25)
        self.assertEqual(args.timeout, 5.0)
        self.assertEqual(args.max_retries, 1)
        self.assertEqual(args.base_url, "http://example.org/x")

    def test_cavityspace_defaults_match_the_old_literals(self):
        args = parse(dcs, [])
        self.assertEqual(args.cavity_mapping, "cavity_mapping.csv")
        self.assertEqual(args.structures_dir, "cavityspace_structures")
        self.assertEqual(args.out, "cavityspace_mapping.csv")
        self.assertEqual(args.threads, 8)
        self.assertEqual(args.db_version, "v1")
        self.assertEqual(args.request_delay, 1.0)

    def test_base_url_and_db_version_reach_the_request(self):
        """--base-url / --db-version must shape both the URL and the local path."""
        alphafold_id, url, path = dcs.model_paths(
            "P12345", 1, "/tmp/out", "v9", "http://example.org/item/")
        self.assertEqual(alphafold_id, "AF-P12345-F1")
        self.assertEqual(
            url,
            "http://example.org/item/P12345/structure/alphafold/"
            "AF-P12345-F1-model_v9/AF-P12345-F1-model_v9.pdb")
        self.assertEqual(path, "/tmp/out/P12345/F1/AF-P12345-F1-model_v9.pdb")


class TestNoDeadPaths(unittest.TestCase):
    def test_no_source_names_a_past_run_directory_or_a_users_home(self):
        offenders = []
        for module in MODULES:
            path = os.path.join(REPO, module.__name__ + ".py")
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            for fragment in FORBIDDEN_PATH_FRAGMENTS:
                if fragment in source:
                    offenders.append("%s.py contains %r" % (module.__name__, fragment))
        self.assertEqual(offenders, [], "; ".join(offenders))

    def test_no_module_constant_points_at_a_dead_path(self):
        offenders = []
        for module in MODULES:
            for name in dir(module):
                if not name.isupper():
                    continue
                value = getattr(module, name)
                if not isinstance(value, str):
                    continue
                for fragment in FORBIDDEN_PATH_FRAGMENTS:
                    if fragment in value:
                        offenders.append("%s.%s = %r" % (module.__name__, name, value))
        self.assertEqual(offenders, [], "; ".join(offenders))

    def test_reference_dataset_defaults_stay_under_opt_data(self):
        """Machine-wide reference data may default; a campaign directory may not."""
        for constant in (cim.DEFAULT_TTD_TARGET_FILE, cim.DEFAULT_TTD_ACTIVITY_FILE,
                         cim.DEFAULT_DRUGBANK_LINKS_FILE):
            self.assertTrue(constant.startswith("/opt/data/"), constant)


class TestImportIsQuiet(unittest.TestCase):
    def test_importing_configures_no_logging_and_writes_no_file(self):
        program = (
            "import logging, os, sys\n"
            "sys.path.insert(0, %r)\n"
            "before = list(logging.getLogger().handlers)\n"
            "import create_ic50_mapping, split_sdf, download_cavityspace_structures\n"
            "assert list(logging.getLogger().handlers) == before, "
            "'import touched the root logger'\n"
            "assert os.listdir('.') == [], os.listdir('.')\n"
            "print('ok')\n" % REPO
        )
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([sys.executable, "-c", program],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr.decode())
            self.assertIn(b"ok", proc.stdout)


def write(path, text):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


class TestIc50SmokeRun(LoggingIsolation):
    def _fixture(self, tmp, name_column="Entry Name", id_column="Entry"):
        targets = write(os.path.join(tmp, "targets.txt"),
                        "TTD - header line\n"
                        "T00001\tUNIPROID\tKCNH2_HUMAN\n"
                        "T00001\tTARGNAME\tSomething\n"
                        "T00002\tUNIPROID\tUNKNOWN_HUMAN\n")
        activity = write(os.path.join(tmp, "activity.txt"),
                         "TTDTargetID\tTTDDrugID\tPubchemCID\tActivity\n"
                         "T00001\tD00001\t2244\tIC50 = 3270000 nM\n"
                         "T00001\tD00002\t5090\tKi > 10 uM\n"
                         "T00002\tD00003\t2244\tIC50 = 5 nM\n"
                         "T00001\tD00004\t9999999\tIC50 = 5 nM\n"
                         "T00001\tD00005\t2244\tno measurement here\n")
        links = write(os.path.join(tmp, "links.csv"),
                      "DrugBank ID,PubChem Compound ID\nDB00001,2244\nDB00002,5090\n")
        mapping = write(os.path.join(tmp, "identity.csv"),
                        "%s,%s\nKCNH2_HUMAN,Q12809\n" % (name_column, id_column))
        return targets, activity, links, mapping

    def test_end_to_end_on_a_tiny_fixture(self):
        import polars as pl
        with tempfile.TemporaryDirectory() as tmp:
            targets, activity, links, mapping = self._fixture(tmp)
            out = os.path.join(tmp, "ic50_mapping.csv")
            code = cim.main(["--uniprot-mapping", mapping, "--out", out,
                             "--ttd-targets", targets, "--ttd-activity", activity,
                             "--drug-links", links, "--run-dir", tmp, "--quiet"])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(out))
            frame = pl.read_csv(out)
            # Only the two rows whose target, drug and measurement all resolve.
            self.assertEqual(frame.height, 2)
            self.assertEqual(sorted(frame["drugbank_id"].to_list()), ["DB00001", "DB00002"])
            self.assertEqual(set(frame["uniprot_id"].to_list()), {"Q12809"})
            self.assertEqual(sorted(frame["measurement_type"].to_list()), ["IC50", "Ki"])
            # The step's log lives under the run directory, never in the cwd.
            log = os.path.join(tmp, "steps", "create_ic50_mapping",
                               "create_ic50_mapping.log")
            self.assertTrue(os.path.exists(log))
            self.assertGreater(os.path.getsize(log), 0)

    def test_identity_index_vocabulary_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            targets, activity, links, mapping = self._fixture(
                tmp, name_column="entry_name", id_column="accession")
            out = os.path.join(tmp, "ic50_mapping.csv")
            code = cim.main(["--uniprot-mapping", mapping, "--out", out,
                             "--ttd-targets", targets, "--ttd-activity", activity,
                             "--drug-links", links, "--run-dir", tmp, "--quiet"])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(out))

    def test_unusable_identity_map_is_an_error_not_an_empty_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            targets, activity, links, _ = self._fixture(tmp)
            bad = write(os.path.join(tmp, "bad.csv"), "foo,bar\n1,2\n")
            with self.assertRaises(ValueError):
                cim.main(["--uniprot-mapping", bad, "--out", os.path.join(tmp, "o.csv"),
                          "--ttd-targets", targets, "--ttd-activity", activity,
                          "--drug-links", links, "--run-dir", tmp, "--quiet"])

    def test_missing_input_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = cim.main(["--uniprot-mapping", os.path.join(tmp, "nope.csv"),
                             "--out", os.path.join(tmp, "o.csv"),
                             "--ttd-targets", os.path.join(tmp, "nope.txt"),
                             "--ttd-activity", os.path.join(tmp, "nope2.txt"),
                             "--drug-links", os.path.join(tmp, "nope3.csv"),
                             "--run-dir", tmp, "--quiet"])
            self.assertEqual(code, 2)
            self.assertFalse(os.path.exists(os.path.join(tmp, "o.csv")))

    def test_notify_without_credentials_fails_at_startup(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("PUSHOVER_TOKEN", "PUSHOVER_USER_KEY")}
        with tempfile.TemporaryDirectory() as tmp:
            targets, activity, links, mapping = self._fixture(tmp)
            out = os.path.join(tmp, "ic50_mapping.csv")
            with mock.patch.dict(os.environ, env, clear=True), \
                    redirect_stderr(io.StringIO()):
                code = cim.main(["--uniprot-mapping", mapping, "--out", out,
                                 "--ttd-targets", targets, "--ttd-activity", activity,
                                 "--drug-links", links, "--run-dir", tmp,
                                 "--quiet", "--notify"])
            self.assertEqual(code, 2)
            self.assertFalse(os.path.exists(out))


class TestSplitSdfSmokeRun(LoggingIsolation):
    def _write_sdf(self, path, ids=("DB00001", "DB00002")):
        from rdkit import Chem
        writer = Chem.SDWriter(path)
        for index, drugbank_id in enumerate(ids):
            mol = Chem.MolFromSmiles("CCO" if index % 2 == 0 else "CCN")
            mol.SetProp("DRUGBANK_ID", drugbank_id)
            writer.write(mol)
        writer.close()
        return path

    def test_end_to_end_on_a_two_molecule_sdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdf = self._write_sdf(os.path.join(tmp, "in.sdf"))
            ligands = os.path.join(tmp, "ligands")
            code = ssdf.main(["--in-sdf", sdf, "--out", ligands,
                              "--seed", "7", "--max-iters", "50",
                              "--run-dir", tmp, "--quiet"])
            self.assertEqual(code, 0)
            self.assertEqual(sorted(os.listdir(ligands)), ["DB00001.sdf", "DB00002.sdf"])
            log = os.path.join(tmp, "steps", "split_sdf", "split_sdf.log")
            self.assertTrue(os.path.exists(log))

    def test_the_seed_is_actually_used(self):
        """Same seed, same coordinates; a different seed is allowed to differ."""
        with tempfile.TemporaryDirectory() as tmp:
            sdf = self._write_sdf(os.path.join(tmp, "in.sdf"), ids=("DB00001",))
            outputs = {}
            for label, seed in (("a", "7"), ("b", "7"), ("c", "1234")):
                out = os.path.join(tmp, "ligands_" + label)
                self.assertEqual(0, ssdf.main(
                    ["--in-sdf", sdf, "--out", out, "--seed", seed,
                     "--run-dir", tmp, "--quiet"]))
                with open(os.path.join(out, "DB00001.sdf"), encoding="utf-8") as handle:
                    outputs[label] = handle.read()
            self.assertEqual(outputs["a"], outputs["b"])

    def test_missing_input_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = ssdf.main(["--in-sdf", os.path.join(tmp, "nope.sdf"),
                              "--out", os.path.join(tmp, "ligands"),
                              "--run-dir", tmp, "--quiet"])
            self.assertEqual(code, 2)


class TestCavitySpaceSmokeRun(LoggingIsolation):
    def test_resume_path_needs_no_network(self):
        """With every structure already on disk the run does no HTTP at all."""
        with tempfile.TemporaryDirectory() as tmp:
            mapping = write(os.path.join(tmp, "cavity_mapping.csv"),
                            "UniProt_ID,Fragment_ID,Cavity_Index,Pocket_PDB\n"
                            "P12345,1,1,/tmp/pocket.pdb\n")
            structures = os.path.join(tmp, "structs")
            _id, _url, model = dcs.model_paths("P12345", 1, structures, "v1",
                                               dcs.DEFAULT_BASE_URL)
            write(model, "ATOM      1  N   MET A   1\n")
            out = os.path.join(tmp, "cavityspace_mapping.csv")
            with mock.patch.object(dcs.requests, "get",
                                   side_effect=AssertionError("no network expected")):
                code = dcs.main(["--cavity-mapping", mapping, "--structures-dir", structures,
                                 "--out", out, "--run-dir", tmp, "--quiet"])
            self.assertEqual(code, 0)
            import pandas as pd
            frame = pd.read_csv(out)
            self.assertEqual(len(frame), 1)
            self.assertTrue(bool(frame["CavitySpace_Available"].iloc[0]))
            self.assertTrue(os.path.exists(os.path.join(
                tmp, "steps", "download_cavityspace_structures",
                "download_cavityspace_structures.log")))

    def test_missing_cavity_mapping_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = dcs.main(["--cavity-mapping", os.path.join(tmp, "nope.csv"),
                             "--structures-dir", os.path.join(tmp, "structs"),
                             "--out", os.path.join(tmp, "out.csv"),
                             "--run-dir", tmp, "--quiet"])
            self.assertEqual(code, 2)
            self.assertFalse(os.path.exists(os.path.join(tmp, "out.csv")))

    def test_download_honours_the_retry_and_delay_budget(self):
        """--max-retries and --request-delay must reach the HTTP loop."""
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "structs", "P1", "F1", "model.pdb")
            with mock.patch.object(dcs.requests, "get",
                                   side_effect=OSError("boom")) as get, \
                    mock.patch.object(dcs.time, "sleep") as sleep:
                ok = dcs.download_file("http://example.org/x", target,
                                       delay=0.5, timeout=3.0, max_retries=2)
            self.assertFalse(ok)
            self.assertEqual(get.call_count, 2)
            self.assertEqual(get.call_args.kwargs["timeout"], 3.0)
            self.assertIn(mock.call(0.5), sleep.call_args_list)


if __name__ == "__main__":
    unittest.main()
