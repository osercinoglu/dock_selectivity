"""Tests for extract_alphafold_models.py's removed fallback.

The defect: ``load_required_structures()`` degraded to ``cavity_mapping.csv`` on
three separate triggers -- the required file absent, present but missing columns,
or unreadable. Each widened the job from 979 accessions / 1,005 fragments to
13,600 / 13,975. The 2025-07-02 log records exactly that happening twice before
the corrected 14:47 run, leaving 5,661 accession directories on disk for a
campaign that used 979.

It was not silent -- it logged a warning -- which is worse, because a warning in
a multi-thousand-line log is not a decision anybody makes. All three triggers are
now fatal.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

import extract_alphafold_models as eam


class TestFallbackRemoved(unittest.TestCase):
    def test_no_path_raises(self):
        with self.assertRaises(eam.MissingRequiredSet) as ctx:
            eam.load_required_structures(None)
        self.assertIn("never a fallback", str(ctx.exception))

    def test_missing_file_raises_rather_than_widening(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(eam.MissingRequiredSet):
                eam.load_required_structures(os.path.join(tmp, "absent.csv"))

    def test_file_without_an_accession_column_raises(self):
        """Trigger 2: present but unusable used to fall through to cavity_mapping."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            pd.DataFrame([{"nope": 1}]).to_csv(path, index=False)
            with self.assertRaises(eam.MissingRequiredSet) as ctx:
                eam.load_required_structures(path)
            self.assertIn("accession column", str(ctx.exception))

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.csv")
            pd.DataFrame(columns=["UniProt_ID", "Fragment_ID"]).to_csv(path, index=False)
            with self.assertRaises(eam.MissingRequiredSet):
                eam.load_required_structures(path)

    def test_allow_fallback_is_refused_loudly(self):
        """The old call site's behaviour must fail, not mysteriously do the old thing."""
        with self.assertRaises(eam.MissingRequiredSet) as ctx:
            eam.load_required_structures("anything.csv", allow_fallback=True)
        self.assertIn("13,975", str(ctx.exception))


class TestFragmentNormalisation(unittest.TestCase):
    """prepare_extraction_tasks builds AF-{acc}-F{fragment}; 'F1' would give 'FF1'."""

    def test_accepts_int_str_and_prefixed(self):
        for value in (1, "1", "F1", "f1", " F1 "):
            self.assertEqual(eam.normalize_fragment_number(value), 1)

    def test_multi_digit(self):
        self.assertEqual(eam.normalize_fragment_number("F23"), 23)

    def test_garbage_raises(self):
        with self.assertRaises(ValueError):
            eam.normalize_fragment_number("not-a-fragment")

    def test_prefixed_fragments_survive_the_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "req.csv")
            pd.DataFrame(
                [{"accession": "P30518", "fragment_id": "F13"}]
            ).to_csv(path, index=False)
            out = eam.load_required_structures(path)
            self.assertEqual(int(out.iloc[0]["Fragment_ID"]), 13)
            # the id the archive scan will look for
            self.assertEqual(
                "AF-%s-F%s" % (out.iloc[0]["UniProt_ID"], out.iloc[0]["Fragment_ID"]),
                "AF-P30518-F13",
            )


class TestLoader(unittest.TestCase):
    def test_accepts_legacy_column_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "req.csv")
            pd.DataFrame(
                [{"UniProt_ID": "P30518", "Fragment_ID": 1, "Cavity_Index": 3}]
            ).to_csv(path, index=False)
            out = eam.load_required_structures(path)
            self.assertEqual(list(out.columns), ["UniProt_ID", "Fragment_ID"])
            self.assertEqual(len(out), 1)

    def test_accepts_new_schema_column_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "queue.csv")
            pd.DataFrame(
                [{"drugbank_id": "DB00035", "accession": "P30518",
                  "fragment_id": "F1", "cavity_index": 2}]
            ).to_csv(path, index=False)
            out = eam.load_required_structures(path)
            self.assertEqual(out.iloc[0]["UniProt_ID"], "P30518")
            self.assertEqual(int(out.iloc[0]["Fragment_ID"]), 1)

    def test_deduplicates_to_structure_level(self):
        """Many cavities of one fragment need the structure extracted once."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "q.csv")
            pd.DataFrame(
                [
                    {"accession": "P30518", "fragment_id": "F1", "cavity_index": 1},
                    {"accession": "P30518", "fragment_id": "F1", "cavity_index": 2},
                    {"accession": "P30518", "fragment_id": "F1", "cavity_index": 5},
                ]
            ).to_csv(path, index=False)
            self.assertEqual(len(eam.load_required_structures(path)), 1)

    def test_missing_fragment_column_defaults_to_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "q.csv")
            pd.DataFrame([{"accession": "P30518"}]).to_csv(path, index=False)
            out = eam.load_required_structures(path)
            self.assertEqual(int(out.iloc[0]["Fragment_ID"]), 1)


class TestCli(unittest.TestCase):
    def test_without_required_exits_two_and_extracts_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "af")
            rc = eam.main(["--out-dir", out_dir, "--out", os.path.join(tmp, "m.csv")])
            self.assertEqual(rc, 2)
            self.assertFalse(os.path.exists(out_dir))  # nothing created

    def test_required_set_is_read_before_the_archive_is_touched(self):
        """Ordering matters: a bad required set must not depend on archive presence."""
        with tempfile.TemporaryDirectory() as tmp:
            rc = eam.main([
                "--required", os.path.join(tmp, "absent.csv"),
                "--archive", os.path.join(tmp, "also-absent.tar"),
                "--out-dir", os.path.join(tmp, "af"),
            ])
            self.assertEqual(rc, 2)

    def test_missing_archive_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            req = os.path.join(tmp, "req.csv")
            pd.DataFrame([{"accession": "P30518", "fragment_id": "F1"}]).to_csv(req, index=False)
            rc = eam.main([
                "--required", req,
                "--archive", os.path.join(tmp, "nope.tar"),
                "--out-dir", os.path.join(tmp, "af"),
                "--out", os.path.join(tmp, "m.csv"),
            ])
            self.assertEqual(rc, 2)

    def test_parser_exposes_the_documented_flags(self):
        parser = eam.build_parser()
        args = parser.parse_args([
            "--required", "r.csv", "--archive", "a.tar",
            "--out-dir", "d", "--out", "m.csv", "--threads", "4",
        ])
        self.assertEqual(args.required, "r.csv")
        self.assertEqual(args.threads, 4)


if __name__ == "__main__":
    unittest.main()
