"""Tests for the required-set CLIs on fix_required_pdbs.py and convert_pdb_to_pdbqt.py.

Both carried the same defect as the two extraction scripts: no way to pass a
required set, and a chain that silently widened the input universe when a
preferred file was missing. ``convert_pdb_to_pdbqt.load_best_cavity_mapping()``
fell through fixed -> alphafold -> cavity_mapping, and that last step is the full
CavitySpace frame (38,709 rows) rather than the ~980 receptors actually used.

fix_required_pdbs is the more expensive of the two to widen: it runs PDBFixer per
structure, so a widened set costs CPU hours as well as disk.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

import convert_pdb_to_pdbqt as cvt
import fix_required_pdbs as frp


class TestFixRequiredPdbs(unittest.TestCase):
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


class TestConvertPdbToPdbqt(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
