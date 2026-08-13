"""Tests for dock_selectivity.joins.

The fixtures reproduce ``parse_prepare.py:1171`` at 1/100,000 scale: docking
results carry ``Fragment_ID``, the annotation frame is keyed on
``(uniprot_id, drugbank_id, cavity_index)`` only, and the join therefore fans
out. In production that added **410,087 rows across 179 jobs**, one job 12x,
and nothing failed.

Hermetic: pure in-memory frames, no I/O.
"""

from __future__ import annotations

import unittest

import pandas as pd

from dock_selectivity.invariants import InvariantViolation, deterministic_fingerprint
from dock_selectivity.joins import (
    checked_merge,
    checked_merge_with_report,
    duplicate_key_report,
    unmatched_left_frame,
)


def docking_results(n_fragments=2):
    """One job's rows: same (accession, drug, cavity), different fragments."""
    return pd.DataFrame(
        {
            "uniprot_id": ["P30518"] * n_fragments,
            "drugbank_id": ["DB00035"] * n_fragments,
            "cavity_index": [2] * n_fragments,
            "fragment_id": ["F{0}".format(i + 1) for i in range(n_fragments)],
            "rmsd": [0.5 + i for i in range(n_fragments)],
        }
    )


def sample_types(with_fragment):
    """The annotation frame — non-unique unless it carries the fragment."""
    frame = pd.DataFrame(
        {
            "uniprot_id": ["P30518", "P30518"],
            "drugbank_id": ["DB00035", "DB00035"],
            "cavity_index": [2, 2],
            "sample_type": ["positive", "negative"],
        }
    )
    if with_fragment:
        frame["fragment_id"] = ["F1", "F2"]
    return frame


class TestFanOut(unittest.TestCase):
    def test_merge_on_a_non_unique_key_raises_with_the_row_delta(self):
        left = docking_results()
        right = sample_types(with_fragment=False)
        with self.assertRaises(InvariantViolation) as ctx:
            checked_merge(
                left,
                right,
                on=["uniprot_id", "drugbank_id", "cavity_index"],
                how="left",
                name="annotate_sample_types",
            )
        message = str(ctx.exception)
        self.assertIn("annotate_sample_types", message)
        self.assertIn("2 -> 4", message)
        self.assertIn("+2", message)  # the row delta, in the message
        self.assertIn("duplicated key value", message)
        report = ctx.exception.report
        self.assertEqual(report["n_left"], 2)
        self.assertEqual(report["n_out"], 4)
        self.assertEqual(report["row_delta"], 2)
        self.assertEqual(report["n_duplicate_right_keys"], 1)
        self.assertEqual(report["max_right_key_multiplicity"], 2)

    def test_merge_on_the_corrected_key_passes_and_preserves_row_count(self):
        left = docking_results()
        right = sample_types(with_fragment=True)
        out, report = checked_merge_with_report(
            left,
            right,
            on=["uniprot_id", "drugbank_id", "cavity_index", "fragment_id"],
            how="left",
            name="annotate_sample_types",
        )
        self.assertEqual(len(out), len(left))
        self.assertEqual(report["row_delta"], 0)
        self.assertEqual(report["match_rate"], 1.0)
        self.assertEqual(
            out.sort_values("fragment_id")["sample_type"].tolist(),
            ["positive", "negative"],
        )
        self.assertNotIn("_merge", out.columns)
        self.assertNotIn("_checked_merge_indicator", out.columns)

    def test_the_worst_job_multiplicity_is_reported(self):
        """One 2025 job appeared 12x in the parquet; the report must say 12."""
        left = docking_results(n_fragments=1)
        right = pd.DataFrame(
            {
                "uniprot_id": ["P30518"] * 12,
                "drugbank_id": ["DB00035"] * 12,
                "cavity_index": [2] * 12,
                "sample_type": ["positive"] * 12,
            }
        )
        with self.assertRaises(InvariantViolation) as ctx:
            checked_merge(
                left,
                right,
                on=["uniprot_id", "drugbank_id", "cavity_index"],
                name="fan_out_12x",
            )
        self.assertEqual(ctx.exception.report["max_right_key_multiplicity"], 12)
        self.assertEqual(ctx.exception.report["n_out"], 12)

    def test_expensive_left_hand_diagnostics_are_computed_only_on_failure(self):
        """The left frame is the 16.5M-row one; do not group it for nothing."""
        left = docking_results()
        clean = sample_types(with_fragment=True)
        key = ["uniprot_id", "drugbank_id", "cavity_index", "fragment_id"]
        _, ok_report = checked_merge_with_report(left, clean, on=key, name="clean")
        self.assertIsNone(ok_report["n_duplicate_left_keys"])
        self.assertIsNone(ok_report["n_unmatched_right"])

        with self.assertRaises(InvariantViolation) as ctx:
            checked_merge(
                left,
                sample_types(with_fragment=False),
                on=["uniprot_id", "drugbank_id", "cavity_index"],
                name="dirty",
            )
        self.assertEqual(ctx.exception.report["n_duplicate_left_keys"], 1)

    def test_growth_is_allowed_only_when_asked_for(self):
        left = docking_results()
        right = sample_types(with_fragment=False)
        out, report = checked_merge_with_report(
            left,
            right,
            on=["uniprot_id", "drugbank_id", "cavity_index"],
            name="deliberate_cross",
            allow_growth=True,
        )
        self.assertEqual(len(out), 4)
        self.assertEqual(report["row_delta"], 2)
        self.assertTrue(report["ok"])

    def test_warn_only_logs_instead_of_raising(self):
        left = docking_results()
        right = sample_types(with_fragment=False)
        with self.assertLogs("dock_selectivity.invariants", level="WARNING") as logs:
            out = checked_merge(
                left,
                right,
                on=["uniprot_id", "drugbank_id", "cavity_index"],
                name="triage_only",
                warn_only=True,
            )
        self.assertEqual(len(out), 4)
        self.assertIn("row count", "".join(logs.output))

    def test_null_keys_fan_out_because_pandas_matches_nan_to_nan(self):
        left = pd.DataFrame({"acc": [None, None], "rmsd": [1.0, 2.0]})
        right = pd.DataFrame({"acc": [None, None], "sample_type": ["p", "n"]})
        with self.assertRaises(InvariantViolation) as ctx:
            checked_merge(left, right, on="acc", name="null_key_join")
        self.assertEqual(ctx.exception.report["n_null_key_rows_left"], 2)
        self.assertEqual(ctx.exception.report["n_out"], 4)

    def test_forbid_null_keys_fails_before_the_merge(self):
        left = pd.DataFrame({"acc": ["P30518", None], "rmsd": [1.0, 2.0]})
        right = pd.DataFrame({"acc": ["P30518"], "sample_type": ["positive"]})
        with self.assertRaises(InvariantViolation) as ctx:
            checked_merge(
                left, right, on="acc", name="strict_keys", forbid_null_keys=True
            )
        self.assertIn("null", str(ctx.exception))


class TestMatchRate(unittest.TestCase):
    def test_unmatched_left_rows_are_always_counted(self):
        left = pd.DataFrame(
            {"acc": ["P30518", "P68363", "Q8NCT1"], "rmsd": [1.0, 2.0, 3.0]}
        )
        right = pd.DataFrame({"acc": ["P30518"], "gene": ["AVPR2"]})
        out, report = checked_merge_with_report(left, right, on="acc", name="label")
        self.assertEqual(len(out), 3)
        self.assertEqual(report["n_unmatched_left"], 2)
        self.assertAlmostEqual(report["match_rate"], 1 / 3, places=6)
        self.assertEqual(report["n_left_only"], 2)
        self.assertEqual(
            sorted(k[0] for k in report["unmatched_left_examples"]),
            ["P68363", "Q8NCT1"],
        )

    def test_require_match_raises_below_the_threshold(self):
        left = pd.DataFrame({"acc": ["P30518", "P68363"], "rmsd": [1.0, 2.0]})
        right = pd.DataFrame({"acc": ["P30518"], "gene": ["AVPR2"]})
        with self.assertRaises(InvariantViolation) as ctx:
            checked_merge(
                left, right, on="acc", name="label_every_row", require_match=1.0
            )
        message = str(ctx.exception)
        self.assertIn("match rate", message)
        self.assertIn("P68363", message)

    def test_require_match_passes_when_satisfied(self):
        left = pd.DataFrame({"acc": ["P30518", "P68363"], "rmsd": [1.0, 2.0]})
        right = pd.DataFrame(
            {"acc": ["P30518", "P68363"], "gene": ["AVPR2", "TUBA1B"]}
        )
        out = checked_merge(
            left, right, on="acc", name="label_every_row", require_match=1.0
        )
        self.assertEqual(out["gene"].tolist(), ["AVPR2", "TUBA1B"])

    def test_unmatched_rows_can_be_recovered_for_the_rejects_ledger(self):
        """I5 — 1,174,786 rows entered the 2025 parquet with a null identity."""
        left = pd.DataFrame(
            {"acc": ["P30518", "P68363", "Q8NCT1"], "rmsd": [1.0, 2.0, 3.0]}
        )
        right = pd.DataFrame({"acc": ["P30518"], "gene": ["AVPR2"]})
        rejects = unmatched_left_frame(left, right, on="acc")
        self.assertEqual(rejects["acc"].tolist(), ["P68363", "Q8NCT1"])
        self.assertEqual(len(rejects) + len(left) - len(rejects), len(left))


class TestSemantics(unittest.TestCase):
    def test_inner_join_may_shrink_but_not_grow(self):
        left = pd.DataFrame({"acc": ["P30518", "P68363"], "rmsd": [1.0, 2.0]})
        right = pd.DataFrame({"acc": ["P30518"], "gene": ["AVPR2"]})
        out = checked_merge(left, right, on="acc", how="inner", name="inner_ok")
        self.assertEqual(len(out), 1)

        right_dup = pd.DataFrame(
            {"acc": ["P30518", "P30518"], "gene": ["AVPR2", "V2R"]}
        )
        with self.assertRaises(InvariantViolation):
            checked_merge(left, right_dup, on="acc", how="inner", name="inner_fanout")

    def test_outer_join_union_is_allowed_but_fan_out_is_not(self):
        left = pd.DataFrame({"acc": ["P30518", "P68363"], "rmsd": [1.0, 2.0]})
        right = pd.DataFrame(
            {"acc": ["P30518", "Q8NCT1"], "gene": ["AVPR2", "ARRDC4"]}
        )
        out, report = checked_merge_with_report(
            left, right, on="acc", how="outer", name="union"
        )
        self.assertEqual(len(out), 3)  # 2 left + 1 unmatched right
        self.assertEqual(report["expected_rows"], 3)

        right_dup = pd.DataFrame(
            {"acc": ["P30518", "P30518"], "gene": ["AVPR2", "V2R"]}
        )
        with self.assertRaises(InvariantViolation):
            checked_merge(left, right_dup, on="acc", how="outer", name="outer_fanout")

    def test_pandas_validate_failure_is_reraised_as_invariant_violation(self):
        left = docking_results()
        right = sample_types(with_fragment=False)
        with self.assertRaises(InvariantViolation) as ctx:
            checked_merge(
                left,
                right,
                on=["uniprot_id", "drugbank_id", "cavity_index"],
                name="validated",
                validate="m:1",
            )
        self.assertIn("validate='m:1'", str(ctx.exception))
        self.assertEqual(ctx.exception.report["n_duplicate_right_keys"], 1)

    def test_a_join_key_is_mandatory(self):
        left = pd.DataFrame({"acc": ["P30518"]})
        right = pd.DataFrame({"acc": ["P30518"]})
        with self.assertRaises(ValueError) as ctx:
            checked_merge(left, right, name="no_key")
        self.assertIn("join key is required", str(ctx.exception))

    def test_missing_join_column_names_the_side(self):
        left = pd.DataFrame({"uniprot_id": ["P30518"]})
        right = pd.DataFrame({"accession": ["P30518"]})
        with self.assertRaises(ValueError) as ctx:
            checked_merge(left, right, on="accession", name="wrong_column")
        self.assertIn("left frame is missing join column", str(ctx.exception))

    def test_left_on_right_on_is_supported(self):
        left = pd.DataFrame({"uniprot_id": ["P30518"], "rmsd": [1.0]})
        right = pd.DataFrame({"accession": ["P30518"], "gene": ["AVPR2"]})
        out = checked_merge(
            left,
            right,
            left_on="uniprot_id",
            right_on="accession",
            name="renamed_key",
            require_match=1.0,
        )
        self.assertEqual(out["gene"].tolist(), ["AVPR2"])

    def test_indicator_column_can_be_kept_on_request(self):
        left = pd.DataFrame({"acc": ["P30518", "P68363"]})
        right = pd.DataFrame({"acc": ["P30518"], "gene": ["AVPR2"]})
        out = checked_merge(left, right, on="acc", name="keep", indicator=True)
        self.assertIn("_merge", out.columns)
        self.assertEqual(sorted(out["_merge"].astype(str)), ["both", "left_only"])

    def test_empty_frames_are_handled(self):
        left = pd.DataFrame({"acc": pd.Series([], dtype="string")})
        right = pd.DataFrame({"acc": pd.Series([], dtype="string"), "gene": []})
        out, report = checked_merge_with_report(left, right, on="acc", name="empty")
        self.assertEqual(len(out), 0)
        self.assertEqual(report["match_rate"], 1.0)

    def test_a_filtered_frame_with_a_gappy_index_is_handled(self):
        """Stages routinely pass ``df[mask].copy()``; the index has holes."""
        left = pd.DataFrame(
            {"acc": ["P30518", "Q8NCT1", "P68363"], "rmsd": [1.0, 2.0, 3.0]}
        )
        left = left[left["rmsd"] > 1.0]  # index becomes [1, 2]
        right = pd.DataFrame({"acc": ["Q8NCT1"], "gene": ["ARRDC4"]})
        out, report = checked_merge_with_report(left, right, on="acc", name="gappy")
        self.assertEqual(len(out), 2)
        self.assertEqual(report["n_unmatched_left"], 1)
        self.assertEqual(
            unmatched_left_frame(left, right, on="acc")["acc"].tolist(), ["P68363"]
        )

    def test_unmatched_left_frame_on_empty_input(self):
        left = pd.DataFrame({"acc": pd.Series([], dtype="string")})
        right = pd.DataFrame({"acc": pd.Series([], dtype="string")})
        self.assertEqual(len(unmatched_left_frame(left, right, on="acc")), 0)

    def test_duplicate_key_report_never_raises(self):
        report = duplicate_key_report(
            sample_types(with_fragment=False),
            ["uniprot_id", "drugbank_id", "cavity_index"],
            name="right_frame",
        )
        self.assertEqual(report["n_duplicate_keys"], 1)
        self.assertEqual(report["max_multiplicity"], 2)

    def test_extension_dtype_keys_from_read_checked_behave(self):
        """Manifests come back from schemas.read_checked with 'string' dtype."""
        left = pd.DataFrame(
            {
                "accession": pd.array(["P30518", "Q8NCT1"], dtype="string"),
                "fragment_id": pd.array(["F1", "F2"], dtype="string"),
                "cavity_index": pd.array([2, 1], dtype="int64"),
            }
        )
        right = pd.DataFrame(
            {
                "accession": pd.array(["P30518"], dtype="string"),
                "fragment_id": pd.array(["F1"], dtype="string"),
                "cavity_index": pd.array([2], dtype="int64"),
                "gene": pd.array(["AVPR2"], dtype="string"),
            }
        )
        key = ["accession", "fragment_id", "cavity_index"]
        out, report = checked_merge_with_report(left, right, on=key, name="typed")
        self.assertEqual(len(out), 2)
        self.assertEqual(report["n_unmatched_left"], 1)
        # a plain-object right-hand frame must behave identically
        out2 = checked_merge(
            left, right.astype({"accession": object}), on=key, name="mixed"
        )
        self.assertEqual(len(out2), 2)

    def test_merge_result_is_deterministic(self):
        left = docking_results()
        right = sample_types(with_fragment=True)
        key = ["uniprot_id", "drugbank_id", "cavity_index", "fragment_id"]
        first = checked_merge(left, right, on=key, name="run1")
        second = checked_merge(left, right, on=key, name="run2")
        self.assertEqual(
            deterministic_fingerprint(first), deterministic_fingerprint(second)
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
