"""Tests for the four structure-prep CLIs (phases 6a-6d of run_full_workflow.py).

The original defect these tests were written for: no way to pass a required set,
and a chain that silently widened the input universe when a preferred file was
missing. ``convert_pdb_to_pdbqt.load_best_cavity_mapping()`` fell through
fixed -> alphafold -> cavity_mapping, and that last step is the full CavitySpace
frame (38,709 rows) rather than the ~980 receptors actually used.

fix_required_pdbs is the more expensive of the two to widen: it runs PDBFixer per
structure, so a widened set costs CPU hours as well as disk.

The second half of this file covers the shared-runtime migration of all four
scripts: every genuine configuration constant is now a flag, no constant names a
past run directory, importing a module configures nothing and writes nothing, and
each step's log lands under ``<run-dir>/steps/<step>/`` rather than in whatever
directory the operator happened to be standing in.
"""

from __future__ import annotations

import ast
import contextlib
import io
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest

import pandas as pd

import convert_pdb_to_pdbqt as cvt
import extract_alphafold_models as eam
import extract_cavities as ec
import fix_required_pdbs as frp

from tests.support import run_in_temp_cwd

#: A step with no --run-dir logs into ./<step-name>/; tests must not
#: create that inside the checkout.
setUpModule, tearDownModule = run_in_temp_cwd()


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: (module, source basename, step name). These four are phases 6a-6d.
PREP_MODULES = (
    (ec, "extract_cavities.py", "extract_cavities"),
    (eam, "extract_alphafold_models.py", "extract_alphafold_models"),
    (frp, "fix_required_pdbs.py", "fix_required_pdbs"),
    (cvt, "convert_pdb_to_pdbqt.py", "convert_pdb_to_pdbqt"),
)

#: Flags every migrated step must carry, from dock_selectivity.runtime.
RUNTIME_FLAGS = ("--out-dir", "--log-file", "--log-level", "--quiet",
                 "--notify", "--notify-on", "--notify-title")


class LoggingIsolation(unittest.TestCase):
    """Detach the step loggers for the duration of a test, then put them back.

    ``dock_selectivity.runtime.configure_logging`` attaches the SAME handler
    object to the step logger and to the shared ``dock_selectivity`` package
    logger, and reconfiguring for any step removes-and-CLOSES every tagged
    handler on that shared logger -- including one another step is still using.
    Verified: configure_logging("a"); configure_logging("b") leaves step a's
    FileHandler attached to logger "a" with ``stream is None``, so the next record
    reopens the path and raises FileNotFoundError once the directory is gone.

    That only bites in-process chaining (tests), but it means an earlier test file
    can poison this one and vice versa. Detaching first makes each test start from
    handlers it owns, and restores the originals untouched afterwards.
    """

    def setUp(self):
        names = [name for _, _, name in PREP_MODULES] + ["dock_selectivity"]
        self._saved = []
        for name in names:
            logger = logging.getLogger(name)
            self._saved.append((logger, logger.level, list(logger.handlers)))
            logger.handlers[:] = []

    def tearDown(self):
        for logger, level, handlers in self._saved:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                with contextlib.suppress(Exception):
                    handler.close()
            logger.handlers[:] = handlers
            logger.setLevel(level)


class TestFixRequiredPdbs(LoggingIsolation):
    def test_missing_required_set_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(frp.MissingRequiredSet):
                frp.load_required_structures(os.path.join(tmp, "absent.csv"),
                                             os.path.join(tmp, "af.csv"))

    def test_missing_alphafold_mapping_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            req = os.path.join(tmp, "req.csv")
            pd.DataFrame([{"UniProt_ID": "P30518", "Fragment_ID": 1}]).to_csv(req, index=False)
            with self.assertRaises(frp.MissingRequiredSet) as ctx:
                frp.load_required_structures(req, os.path.join(tmp, "absent_af.csv"))
            self.assertIn("AlphaFold mapping not found", str(ctx.exception))

    def test_cli_without_required_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = frp.main(["--out-dir", os.path.join(tmp, "fixed"),
                           "--out", os.path.join(tmp, "m.csv")])
            self.assertEqual(rc, 2)
            self.assertFalse(os.path.exists(os.path.join(tmp, "fixed")))

    def test_normalize_accepts_both_column_vocabularies(self):
        legacy = pd.DataFrame([{"UniProt_ID": "P30518", "Fragment_ID": 1}])
        modern = pd.DataFrame([{"accession": "P30518", "fragment_id": "F1"}])
        a = frp.normalize_required_frame(legacy, "legacy")
        b = frp.normalize_required_frame(modern, "modern")
        pd.testing.assert_frame_equal(a, b)

    def test_normalize_deduplicates_cavity_rows_to_structures(self):
        df = pd.DataFrame([
            {"accession": "P30518", "fragment_id": "F1", "cavity_index": 1},
            {"accession": "P30518", "fragment_id": "F1", "cavity_index": 2},
        ])
        self.assertEqual(len(frp.normalize_required_frame(df, "x")), 1)

    def test_normalize_without_accession_column_raises(self):
        with self.assertRaises(frp.MissingRequiredSet):
            frp.normalize_required_frame(pd.DataFrame([{"nope": 1}]), "x")

    def test_parser_exposes_documented_flags(self):
        args = frp.build_parser().parse_args(
            ["--required", "r.csv", "--alphafold-mapping", "af.csv",
             "--out-dir", "d", "--out", "m.csv", "--threads", "3"]
        )
        self.assertEqual((args.required, args.threads), ("r.csv", 3))


class TestConvertPdbToPdbqt(LoggingIsolation):
    def _mapping(self, path, accessions):
        pd.DataFrame(
            [{"UniProt_ID": a, "Cavity_Index": 1,
              "Receptor_PDB": "s/%s.pdb" % a, "Pocket_PDB": "p/%s.pdb" % a}
             for a in accessions]
        ).to_csv(path, index=False)

    def test_no_mapping_at_all_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with self.assertRaises(cvt.MissingInputMapping):
                    cvt.load_best_cavity_mapping()
            finally:
                os.chdir(cwd)

    def test_cavity_fallback_is_refused_unless_asked_for(self):
        """The 38,709-row path must be opt-in, not the automatic last resort."""
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                self._mapping(cvt.CAVITY_MAPPING_CSV, ["P30518", "Q8NCT1"])
                with self.assertRaises(cvt.MissingInputMapping) as ctx:
                    cvt.load_best_cavity_mapping()
                self.assertIn("allow-cavity-fallback", str(ctx.exception))
                df, src = cvt.load_best_cavity_mapping(allow_cavity_fallback=True)
                self.assertEqual(len(df), 2)
                self.assertEqual(src, cvt.CAVITY_MAPPING_CSV)
            finally:
                os.chdir(cwd)

    def test_explicit_mapping_wins_and_missing_one_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "explicit.csv")
            self._mapping(path, ["P30518"])
            df, src = cvt.load_best_cavity_mapping(explicit=path)
            self.assertEqual(src, path)
            self.assertEqual(len(df), 1)
            with self.assertRaises(cvt.MissingInputMapping):
                cvt.load_best_cavity_mapping(explicit=os.path.join(tmp, "nope.csv"))

    def test_required_filter_restricts_to_named_accessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = os.path.join(tmp, "m.csv")
            self._mapping(mapping, ["P30518", "Q8NCT1", "P68363"])
            required = os.path.join(tmp, "r.csv")
            pd.DataFrame([{"accession": "P30518"}, {"accession": "P68363"}]).to_csv(
                required, index=False
            )
            df, _ = cvt.load_best_cavity_mapping(explicit=mapping)
            filtered = cvt.filter_to_required(df, required)
            self.assertEqual(sorted(filtered["UniProt_ID"]), ["P30518", "P68363"])

    def test_required_file_without_accession_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = os.path.join(tmp, "m.csv")
            self._mapping(mapping, ["P30518"])
            bad = os.path.join(tmp, "bad.csv")
            pd.DataFrame([{"nope": 1}]).to_csv(bad, index=False)
            df, _ = cvt.load_best_cavity_mapping(explicit=mapping)
            with self.assertRaises(cvt.MissingInputMapping):
                cvt.filter_to_required(df, bad)

    def test_cli_with_absent_mapping_exits_two_without_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "pdbqt")
            rc = cvt.main(["--input-mapping", os.path.join(tmp, "nope.csv"),
                           "--out-dir", out_dir, "--out", os.path.join(tmp, "m.csv")])
            self.assertEqual(rc, 2)
            self.assertFalse(os.path.exists(out_dir))

    def test_parser_exposes_documented_flags(self):
        args = cvt.build_parser().parse_args(
            ["--input-mapping", "f.csv", "--required", "r.csv",
             "--allow-cavity-fallback", "--out-dir", "d", "--out", "m.csv", "--threads", "8"]
        )
        self.assertTrue(args.allow_cavity_fallback)
        self.assertEqual((args.input_mapping, args.required, args.threads), ("f.csv", "r.csv", 8))


def _flags(parser):
    """Every option string the parser accepts."""
    out = set()
    for action in parser._get_optional_actions():
        out.update(action.option_strings)
    return out


def make_cavity_tarball(folder, accession, fragment, cavity):
    """Write one realistic CavitySpace archive and return its filename."""
    stem = "AF-%s-F%d-model_v1" % (accession, fragment)
    name = "%s_cavity_result_%d.tar.gz" % (stem, cavity)
    inner = "%s_cavity_%d.pdb" % (stem, cavity)
    payload = b"ATOM      1  N   ALA A   1       0.000   0.000   0.000\n"
    with tarfile.open(os.path.join(folder, name), "w:gz") as tar:
        info = tarfile.TarInfo(inner)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return name


class TestPrepParsers(LoggingIsolation):
    """Every prep script builds a parser, prints help, and carries the runtime flags."""

    def test_help_exits_zero_for_every_prep_script(self):
        for module, filename, _ in PREP_MODULES:
            with self.subTest(script=filename):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    with self.assertRaises(SystemExit) as ctx:
                        module.build_parser().parse_args(["--help"])
                self.assertEqual(ctx.exception.code, 0)
                self.assertIn("--out-dir", buffer.getvalue())

    def test_every_prep_script_carries_the_shared_runtime_flags(self):
        for module, filename, _ in PREP_MODULES:
            with self.subTest(script=filename):
                flags = _flags(module.build_parser())
                for flag in RUNTIME_FLAGS + ("--run-dir",):
                    self.assertIn(flag, flags)

    def test_out_dir_still_means_the_artifact_destination(self):
        """batch_dock resolves these mappings by name; --out/--out-dir may not move."""
        cases = (
            (ec, ["--out-dir", "d", "--out", "cavity_mapping.csv"]),
            (eam, ["--out-dir", "d", "--out", "alphafold_mapping.csv"]),
            (frp, ["--out-dir", "d", "--out", "fixed_mapping.csv"]),
            (cvt, ["--out-dir", "d", "--out", "pdbqt_mapping.csv"]),
        )
        for module, argv in cases:
            with self.subTest(script=module.__name__):
                args = module.build_parser().parse_args(argv)
                self.assertEqual(args.out_dir, "d")
                self.assertEqual(args.out, argv[-1])


class TestFormerlyHardcodedValues(LoggingIsolation):
    """Each constant that was only editable in the source is now a flag."""

    def test_extract_cavities_tarball_folder_is_a_flag(self):
        args = ec.build_parser().parse_args(["--tarballs", "/data/cavities"])
        self.assertEqual(args.tarballs, "/data/cavities")
        self.assertEqual(ec.build_parser().parse_args([]).tarballs,
                         ec.CAVITY_TARBALL_FOLDER)

    def test_extract_cavities_manifest_honours_the_extraction_destination(self):
        """The manifest advertises the paths step two will write, not a constant."""
        with tempfile.TemporaryDirectory() as tmp:
            make_cavity_tarball(tmp, "B4DS77", 1, 1)
            manifest, _ = ec.build_cavity_manifest(tmp, extract_base_dir="/s/cav")
            self.assertTrue(manifest.iloc[0]["Pocket_PDB"].startswith("/s/cav/"))

    def test_alphafold_database_version_is_a_flag(self):
        self.assertEqual(eam.build_parser().parse_args(["--db-version", "v5"]).db_version,
                         "v5")
        self.assertEqual(eam.build_parser().parse_args([]).db_version,
                         eam.DATABASE_VERSION)

    def test_alphafold_mapping_paths_use_the_requested_db_version(self):
        """--archive may hold v3 or v5; the filenames must not still claim v4."""
        with tempfile.TemporaryDirectory() as tmp:
            wanted = os.path.join(tmp, "P30518", "F1", "AF-P30518-F1-model_v9.pdb")
            os.makedirs(os.path.dirname(wanted))
            with open(wanted, "w") as handle:
                handle.write("ATOM\n")
            frame = pd.DataFrame([{"UniProt_ID": "P30518", "Fragment_ID": 1}])
            mapping = eam.create_alphafold_mapping(frame, tmp, db_version="v9")
            self.assertEqual(mapping.iloc[0]["Receptor_PDB"], wanted)
            self.assertTrue(bool(mapping.iloc[0]["AlphaFold_Available"]))

    def test_fix_required_pdbs_ph_and_db_version_are_flags(self):
        args = frp.build_parser().parse_args(["--ph", "6.5", "--db-version", "v3"])
        self.assertEqual(args.ph, 6.5)
        self.assertEqual(args.db_version, "v3")
        self.assertEqual(frp.build_parser().parse_args([]).ph, frp.PH_VALUE)

    def test_fix_required_pdbs_db_version_reaches_the_output_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            hydrogenated = os.path.join(tmp, "P30518", "F1",
                                        "AF-P30518-F1-model_v3_H.pdb")
            os.makedirs(os.path.dirname(hydrogenated))
            with open(hydrogenated, "w") as handle:
                handle.write("ATOM\n")
            frame = pd.DataFrame([{"UniProt_ID": "P30518", "Fragment_ID": 1}])
            mapping = frp.create_fixed_mapping(frame, tmp, db_version="v3")
            self.assertEqual(mapping.iloc[0]["Receptor_PDB"], hydrogenated)
            self.assertTrue(bool(mapping.iloc[0]["Receptor_Fixed"]))

    def test_fix_required_pdbs_carries_the_ph_into_every_worker_task(self):
        frame = pd.DataFrame(
            [{"UniProt_ID": "P30518", "Fragment_ID": 1, "Receptor_PDB": __file__}]
        )
        tasks, _, _ = frp.prepare_fixing_tasks(frame, "out", ph=6.5, db_version="v3")
        self.assertEqual([t["ph"] for t in tasks], [6.5])
        self.assertIn("model_v3_H.pdb", tasks[0]["output_pdb"])

    def test_convert_timeout_and_ph_are_flags_not_environment_only(self):
        args = cvt.build_parser().parse_args(["--timeout", "45", "--ph", "6.0"])
        self.assertEqual(args.timeout, 45)
        self.assertEqual(args.ph, 6.0)
        defaults = cvt.build_parser().parse_args([])
        self.assertEqual(defaults.timeout, cvt.CONVERSION_TIMEOUT)
        self.assertEqual(defaults.ph, cvt.PH_VALUE)

    def test_convert_carries_the_ph_into_every_worker_task(self):
        """The workers run in another process, so the pH must travel in the task."""
        with tempfile.TemporaryDirectory() as tmp:
            pdb = os.path.join(tmp, "P30518.pdb")
            with open(pdb, "w") as handle:
                handle.write("ATOM      1  N   ALA A   1       0.0   0.0   0.0\n")
            frame = pd.DataFrame([{"UniProt_ID": "P30518", "Receptor_PDB": pdb,
                                   "Pocket_PDB": "p.pdb"}])
            tasks, _ = cvt.prepare_conversion_tasks(frame, os.path.join(tmp, "out"),
                                                    ph=6.0)
            self.assertEqual([t["ph"] for t in tasks], [6.0])


class TestNoDeadRunDirectories(unittest.TestCase):
    """No prep script may name a past campaign or a specific user's home."""

    FORBIDDEN = ("/media/", "2025_06_29", "/home/onur", "/home/ssahin",
                 "/home/madinasu", "/home/experiments")

    def test_no_string_literal_names_a_dead_run_directory(self):
        for _, filename, _ in PREP_MODULES:
            with self.subTest(script=filename):
                path = os.path.join(REPO_ROOT, filename)
                with open(path, encoding="utf-8") as handle:
                    tree = ast.parse(handle.read(), filename=path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        for needle in self.FORBIDDEN:
                            self.assertNotIn(
                                needle, node.value,
                                "%s:%s still names %s" % (filename, node.lineno, needle))

    def test_module_level_path_constants_are_relative_or_reference_data(self):
        """An absolute default may only be a machine-wide dataset under /opt/data."""
        for module, filename, _ in PREP_MODULES:
            for name in dir(module):
                if not name.isupper():
                    continue
                value = getattr(module, name)
                if isinstance(value, str) and value.startswith("/"):
                    with self.subTest(script=filename, constant=name):
                        self.assertTrue(value.startswith("/opt/data"), value)


class TestImportIsQuiet(unittest.TestCase):
    """Importing a prep module must configure nothing and write nothing.

    Before the migration, importing these modules called
    ``logging.basicConfig(filename=...)`` at module scope: the first one imported
    won the process's only FileHandler, the other five never wrote a file at all,
    and every import stacked another stdout handler so a single record printed
    six times. Running the test suite left 1.7 MB of *.log in the repository.
    """

    def test_importing_all_four_leaves_the_root_logger_and_the_cwd_alone(self):
        script = (
            "import logging, os, sys\n"
            "before = list(logging.getLogger().handlers)\n"
            "import extract_cavities, extract_alphafold_models\n"
            "import fix_required_pdbs, convert_pdb_to_pdbqt\n"
            "after = list(logging.getLogger().handlers)\n"
            "assert before == after, 'root logger changed: %r -> %r' % (before, after)\n"
            "left = sorted(os.listdir('.'))\n"
            "assert left == [], 'import created %r' % (left,)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, PYTHONPATH=REPO_ROOT)
            proc = subprocess.run([sys.executable, "-c", script], cwd=tmp, env=env,
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])


class TestStepLogging(LoggingIsolation):
    """A step's log lands in its own folder, never in the artifact tree or cwd."""

    def test_manifest_only_writes_its_log_under_the_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "tarballs")
            os.makedirs(src)
            make_cavity_tarball(src, "B4DS77", 1, 1)
            run = os.path.join(tmp, "run")
            os.makedirs(os.path.join(run, "manifests"))
            out = os.path.join(run, "manifests", "cavity_manifest.csv")

            rc = ec.main(["--manifest-only", "--tarballs", src, "--out", out,
                          "--run-dir", run])
            self.assertEqual(rc, 0)
            log = os.path.join(run, "steps", "extract_cavities", "extract_cavities.log")
            self.assertTrue(os.path.exists(log), sorted(os.listdir(run)))
            self.assertGreater(os.path.getsize(log), 0)
            with open(log, encoding="utf-8") as handle:
                self.assertIn("resolved configuration", handle.read())

    def test_without_a_run_dir_the_log_goes_to_a_step_folder_not_the_cwd(self):
        """These four take --out-dir to mean "where the structures land", so the
        step folder was derived from --out instead. An unqualified --out
        ("cavity_manifest.csv") made that the invocation cwd, which is how running
        the test suite left extract_cavities.log in the repository root.
        """
        for module, argv, step_name in (
            (ec, ["--manifest-only", "--out", "cavity_manifest.csv"], "extract_cavities"),
        ):
            with self.subTest(step=step_name), tempfile.TemporaryDirectory() as tmp:
                src = os.path.join(tmp, "tarballs")
                os.makedirs(src)
                make_cavity_tarball(src, "B4DS77", 1, 1)
                cwd = os.getcwd()
                os.chdir(tmp)
                try:
                    self.assertEqual(module.main(argv + ["--tarballs", src]), 0)
                    self.assertEqual([e for e in os.listdir(tmp) if e.endswith(".log")], [],
                                     "a log landed in the invocation cwd")
                    log = os.path.join(tmp, step_name, step_name + ".log")
                    self.assertTrue(os.path.exists(log), sorted(os.listdir(tmp)))
                finally:
                    os.chdir(cwd)

    def test_the_log_never_lands_in_the_extraction_directory(self):
        """The extraction tree is counted by other stages; a log there is a defect."""
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "tarballs")
            os.makedirs(src)
            make_cavity_tarball(src, "B4DS77", 1, 1)
            run = os.path.join(tmp, "run")
            os.makedirs(os.path.join(run, "manifests"))
            manifest = os.path.join(run, "manifests", "cavity_manifest.csv")
            self.assertEqual(
                ec.main(["--manifest-only", "--tarballs", src, "--out", manifest,
                         "--run-dir", run]), 0)

            required = os.path.join(tmp, "required.csv")
            pd.DataFrame([{"accession": "B4DS77", "fragment_id": "F1",
                           "cavity_index": 1}]).to_csv(required, index=False)
            structures = os.path.join(run, "structures", "cavities")
            rc = ec.main(["--tarballs", src, "--manifest", manifest,
                          "--required", required, "--out-dir", structures,
                          "--out", os.path.join(run, "manifests", "cavity_mapping.csv"),
                          "--run-dir", run])
            self.assertEqual(rc, 0)
            self.assertEqual([e for e in os.listdir(structures)
                              if e.endswith(".log")], [])

    def test_a_failing_step_still_returns_its_exit_code(self):
        """step() sees a clean return, so the code must still reach the caller."""
        with tempfile.TemporaryDirectory() as tmp:
            rc = eam.main(["--out", os.path.join(tmp, "m.csv"),
                           "--out-dir", os.path.join(tmp, "af"),
                           "--run-dir", os.path.join(tmp, "run")])
            self.assertEqual(rc, 2)
            self.assertFalse(os.path.exists(os.path.join(tmp, "af")))

    def test_notify_without_credentials_refuses_at_startup(self):
        """--notify with no environment credentials must fail before any work."""
        with tempfile.TemporaryDirectory() as tmp:
            saved = {k: os.environ.pop(k) for k in
                     ("PUSHOVER_TOKEN", "PUSHOVER_USER_KEY") if k in os.environ}
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as ctx:
                        ec.main(["--manifest-only", "--tarballs", tmp,
                                 "--out", os.path.join(tmp, "m.csv"),
                                 "--run-dir", os.path.join(tmp, "run"), "--notify"])
                self.assertEqual(ctx.exception.code, 2)
                self.assertFalse(os.path.exists(os.path.join(tmp, "m.csv")))
            finally:
                os.environ.update(saved)


if __name__ == "__main__":
    unittest.main()
