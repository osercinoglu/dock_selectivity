"""Tests for the five FIX_PLAN §3.4 defects in parse_prepare.py.

Each test names the production consequence it prevents. All five were measured
against the real 2025-06-29 artifacts before being fixed.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import parse_prepare as pp  # noqa: E402


class TestDefect1FileDiscoveryCap(unittest.TestCase):
    """The cap dropped 14,173 of 24,173 completed jobs -- 58% -- in silence."""

    def test_default_is_no_cap(self):
        self.assertIsNone(
            pp.FILE_DISCOVERY_CAP,
            "a non-None default silently truncates discovery; that is the defect",
        )

    def test_walk_returns_everything_when_uncapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(50):
                d = os.path.join(tmp, "job_%03d" % i)
                os.makedirs(d)
                open(os.path.join(d, "final_results.csv"), "w").close()
            found = pp.optimized_walk_search((tmp, ["final_results.csv"], None, None))
            self.assertEqual(len(found["final_results.csv"]), 50)

    def test_an_explicit_cap_still_works_when_asked_for(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(50):
                d = os.path.join(tmp, "job_%03d" % i)
                os.makedirs(d)
                open(os.path.join(d, "final_results.csv"), "w").close()
            found = pp.optimized_walk_search((tmp, ["final_results.csv"], None, 10))
            self.assertLessEqual(len(found["final_results.csv"]), 10)


class TestDefect2IdentityParsing(unittest.TestCase):
    """The old regex rejected 1,777 of 24,173 real directories (verified 7.35%)."""

    def test_plain_legacy_name(self):
        got = pp.parse_job_dir_name("DB00035_AVPR2_P30518_cavity_2")
        self.assertEqual(got["drugbank_id"], "DB00035")
        self.assertEqual(got["uniprot_id"], "P30518")
        self.assertEqual(got["cavity_index"], 2)

    def test_hyphenated_gene_symbol(self):
        """34 real dirs; [A-Za-z0-9]+ cannot match a hyphen."""
        got = pp.parse_job_dir_name("DB00157_MT-ND1_P03886_cavity_1")
        self.assertEqual(got["gene_name"], "MT-ND1")
        self.assertEqual(got["uniprot_id"], "P03886")

    def test_legacy_multi_fragment_name(self):
        """1,743 real dirs; the old pattern had no field for a fragment at all."""
        got = pp.parse_job_dir_name("DB00091_CAMLG_P08F94_-F13-model_v1_cavity_10")
        self.assertEqual(got["uniprot_id"], "P08F94")
        self.assertEqual(got["fragment_id"], "F13")
        self.assertEqual(got["cavity_index"], 10)

    def test_new_scheme_is_preferred(self):
        got = pp.parse_job_dir_name("DB00035__P30518__AVPR2__F1__cav2")
        self.assertEqual((got["uniprot_id"], got["gene_name"]), ("P30518", "AVPR2"))

    def test_nan_gene_token_becomes_none_not_the_string(self):
        """11,806 real negative dirs carry the literal token 'nan'."""
        self.assertIsNone(pp.parse_job_dir_name("DB00035_nan_P30518_cavity_1")["gene_name"])

    def test_unparseable_returns_none_rather_than_a_partial_guess(self):
        self.assertIsNone(pp.parse_job_dir_name("not-a-job-directory"))

    def test_every_shape_yields_a_fragment(self):
        """Fragment absence is what fanned the sample-type merge out by 410,087 rows."""
        for name in ("DB00035_AVPR2_P30518_cavity_2",
                     "DB00157_MT-ND1_P03886_cavity_1",
                     "DB00091_CAMLG_P08F94_-F13-model_v1_cavity_10"):
            self.assertTrue(pp.parse_job_dir_name(name)["fragment_id"].startswith("F"))


class TestDefects3And4ToolNames(unittest.TestCase):
    """Data says SMINA/GNINA/LEDOCK/GOLD; the code said Smina/Gnina/LeDock."""

    def test_canonical_tool_folds_every_spelling(self):
        for spelling in ("LeDock", "ledock", "LEDOCK", " LeDock "):
            self.assertEqual(pp.canonical_tool(spelling), "LEDOCK")
        self.assertEqual(pp.canonical_tool("Smina"), "SMINA")
        self.assertEqual(pp.canonical_tool("Gnina"), "GNINA")
        self.assertIsNone(pp.canonical_tool(None))
        self.assertIsNone(pp.canonical_tool("  "))

    def test_canonical_tools_match_the_data_vocabulary(self):
        self.assertEqual(set(pp.CANONICAL_TOOLS), {"GOLD", "SMINA", "LEDOCK", "GNINA"})

    def test_ledock_score_column_is_populated(self):
        """At HEAD this column was entirely null because 'LeDock' != 'LEDOCK'."""
        df = pd.DataFrame({
            "Tool1": ["SMINA", "LEDOCK"], "Tool2": ["LEDOCK", "GOLD"],
            "Score1": [-7.0, -5.0], "Score2": [-4.0, 50.0],
        })
        out = pp.add_tool_specific_scores(df.copy())
        self.assertEqual(out["LEDOCK_Score"].notna().sum(), 2)
        self.assertEqual(out["SMINA_Score"].notna().sum(), 1)

    def test_complete_tool_coverage_actually_filters(self):
        """It has NEVER filtered: only 'GOLD' could match, so it was a silent no-op."""
        complete = [{"drugbank_id": "DB1", "uniprot_id": "P1", "source_type": "final_results",
                     "Tool1": a, "Tool2": b, "Score1": -1.0, "Score2": -2.0}
                    for a, b in (("SMINA", "GNINA"), ("LEDOCK", "GOLD"))]
        partial = [{"drugbank_id": "DB2", "uniprot_id": "P2", "source_type": "final_results",
                    "Tool1": "SMINA", "Tool2": "GNINA", "Score1": -1.0, "Score2": -2.0}]
        out = pp.filter_complete_tool_coverage(pd.DataFrame(complete + partial))
        self.assertEqual(set(out["drugbank_id"]), {"DB1"},
                         "the incomplete pair should have been filtered out")


class TestDefect5MergeFanOut(unittest.TestCase):
    """The key omitted Fragment_ID and duplicated 410,087 rows across 179 jobs."""

    def test_a_fanning_merge_raises_instead_of_inflating(self):
        df = pd.DataFrame({
            "uniprot_id": ["P30518"], "drugbank_id": ["DB00035"],
            "extracted_cavity_index": [1], "Score1": [-7.0],
        })
        # Two metadata rows for one (uniprot, drug, cavity) differing only by fragment
        meta = pd.DataFrame({
            "UniProt_ID": ["P30518", "P30518"],
            "drugbank_id": ["DB00035", "DB00035"],
            "Cavity_Index": [1, 1],
            "Fragment_ID": ["F1", "F2"],
            "sample_type": ["positive", "positive"],
            "Gene_Name": ["AVPR2", "AVPR2"],
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "meta.csv")
            meta.to_csv(path, index=False)
            with self.assertRaises(pp.MergeFanOut) as ctx:
                pp.annotate_sample_types(df, metadata_file=path)
            self.assertIn("row count", str(ctx.exception))
            self.assertIn("Fragment_ID", str(ctx.exception))

    def test_a_unique_key_merges_without_growth(self):
        df = pd.DataFrame({
            "uniprot_id": ["P30518"], "drugbank_id": ["DB00035"],
            "extracted_cavity_index": [1], "Score1": [-7.0],
        })
        meta = pd.DataFrame({
            "UniProt_ID": ["P30518"], "drugbank_id": ["DB00035"], "Cavity_Index": [1],
            "Fragment_ID": ["F1"], "sample_type": ["positive"], "Gene_Name": ["AVPR2"],
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "meta.csv")
            meta.to_csv(path, index=False)
            out = pp.annotate_sample_types(df, metadata_file=path)
            self.assertEqual(len(out), 1)
            self.assertEqual(out.iloc[0]["sample_type"], "positive")


if __name__ == "__main__":
    unittest.main()
