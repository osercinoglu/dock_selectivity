"""Tests for build_run_manifest.py.

Two properties carry the weight here:

* the manifest must list **every** job the design wants, salvaged or not --
  a manifest covering only the salvaged half would reproduce the blind spot that
  let ``max_files_per_pattern = 10000`` drop 14,173 completed jobs unnoticed;
* matching a planned job to an existing directory happens on the **full** pocket
  key. Dropping ``fragment_id`` from a key of this shape is what made the old
  sample-type merge non-unique and duplicated 410,087 rows.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

import build_run_manifest as brm
from dock_selectivity import schemas
from dock_selectivity.invariants import InvariantViolation


def positives(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "drugbank_id": d,
                "accession": a,
                "gene": g,
                "fragment_id": f,
                "cavity_index": c,
                "sample_type": "positive",
                "selected_by": "entrez",
            }
            for (d, a, g, f, c) in rows
        ]
    )


def drawn(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "drugbank_id": d,
                "accession": a,
                "gene": g,
                "fragment_id": f,
                "cavity_index": c,
                "sample_type": "negative",
                "source": src,
                "old_job_dir": old,
                "selected_by": "balanced",
            }
            for (d, a, g, f, c, src, old) in rows
        ]
    )


def salvage(rows) -> pd.DataFrame:
    """salvage_map rows.

    The schema requires exactly one of new_job_dir / excluded_reason, so a kept
    row names its destination and an excluded row names its reason.
    """
    out = []
    for (old, cls, d, a, g, f, c, old_label) in rows:
        kept = cls in ("true_positive", "valid_non_interaction")
        out.append(
            {
                "old_job_dir": old,
                "classification": cls,
                "drugbank_id": d,
                "accession": a,
                "gene": g,
                "fragment_id": f,
                "cavity_index": c,
                "old_gene_label": old_label,
                "label_was_correct": old_label == g,
                "new_job_dir": "{0}__{1}__{2}__{3}__cav{4}".format(d, a, g, f, c) if kept else None,
                "excluded_reason": None if kept else "not_reusable",
            }
        )
    return pd.DataFrame(out)


class TestCoverage(unittest.TestCase):
    def test_every_planned_job_appears_salvaged_or_not(self):
        manifest, _ = brm.build_manifest(
            positives([("DB00035", "P30518", "AVPR2", "F1", 1),
                       ("DB00035", "P30518", "AVPR2", "F1", 2)]),
            drawn([("DB00157", "P68363", "TUBA1B", "F1", 3, "new", None)]),
            salvage([("DB00035_AVPR2_P30518_cavity_1", "true_positive",
                      "DB00035", "P30518", "AVPR2", "F1", 1, "AVPR2")]),
        )
        self.assertEqual(len(manifest), 3)
        self.assertEqual(
            sorted(manifest["provenance"]),
            ["newly_docked", "newly_docked", "salvaged_from"],
        )

    def test_newly_docked_rows_reconcile_with_the_redock_queue(self):
        manifest, redock = brm.build_manifest(
            positives([("DB00035", "P30518", "AVPR2", "F1", 1),
                       ("DB00157", "P68363", "TUBA1B", "F1", 1)]),
            drawn([("DB00157", "Q8NCT1", "ARRDC4", "F1", 2, "new", None)]),
            salvage([]),
        )
        self.assertEqual(len(redock), int((manifest["provenance"] == "newly_docked").sum()))
        self.assertEqual(len(redock), 3)

    def test_positive_gap_is_queued(self):
        """Nothing previously queued the positive side; only 2,100/8,285 were docked."""
        _, redock = brm.build_manifest(
            positives([("DB00035", "P30518", "AVPR2", "F1", 1)]),
            drawn([]),
            salvage([]),
        )
        self.assertEqual(len(redock), 1)
        self.assertEqual(redock.iloc[0]["sample_type"], "positive")


class TestPocketKeyMatching(unittest.TestCase):
    def test_match_requires_the_full_key_including_fragment(self):
        """A salvaged F2 job must not satisfy a planned F1 job."""
        manifest, _ = brm.build_manifest(
            positives([("DB00035", "P30518", "AVPR2", "F1", 1)]),
            drawn([]),
            salvage([("old_f2", "true_positive", "DB00035", "P30518", "AVPR2", "F2", 1, "AVPR2")]),
        )
        self.assertEqual(manifest.iloc[0]["provenance"], "newly_docked")

    def test_bare_and_prefixed_fragment_ids_match_each_other(self):
        manifest, _ = brm.build_manifest(
            positives([("DB00035", "P30518", "AVPR2", "F1", 1)]),
            drawn([]),
            salvage([("old_1", "true_positive", "DB00035", "P30518", "AVPR2", "1", 1, "AVPR2")]),
        )
        self.assertEqual(manifest.iloc[0]["provenance"], "salvaged_from")
        self.assertEqual(manifest.iloc[0]["source_job"], "old_1")

    def test_non_true_positive_salvage_rows_do_not_satisfy_a_positive(self):
        manifest, _ = brm.build_manifest(
            positives([("DB00035", "P30518", "AVPR2", "F1", 1)]),
            drawn([]),
            salvage([("old", "valid_non_interaction",
                      "DB00035", "P30518", "AVPR2", "F1", 1, "AVPR2")]),
        )
        self.assertEqual(manifest.iloc[0]["provenance"], "newly_docked")

    def test_two_salvaged_dirs_claiming_one_pocket_raises(self):
        with self.assertRaises(InvariantViolation) as ctx:
            brm.build_manifest(
                positives([("DB00035", "P30518", "AVPR2", "F1", 1)]),
                drawn([]),
                salvage([
                    ("dir_a", "true_positive", "DB00035", "P30518", "AVPR2", "F1", 1, "AVPR2"),
                    ("dir_b", "true_positive", "DB00035", "P30518", "AVPR2", "F1", 1, "nan"),
                ]),
            )
        self.assertIn("dir_a", str(ctx.exception))


class TestIntegrity(unittest.TestCase):
    def test_duplicate_job_id_raises(self):
        """The same pocket planned as both a positive and a negative is a design bug."""
        with self.assertRaises(InvariantViolation):
            brm.build_manifest(
                positives([("DB00035", "P30518", "AVPR2", "F1", 1)]),
                drawn([("DB00035", "P30518", "AVPR2", "F1", 1, "new", None)]),
                salvage([]),
            )

    def test_pool_source_without_an_old_dir_raises(self):
        with self.assertRaises(InvariantViolation) as ctx:
            brm.build_manifest(
                positives([]),
                drawn([("DB00035", "P30518", "AVPR2", "F1", 1, "pool", None)]),
                salvage([]),
            )
        self.assertIn("nowhere", str(ctx.exception))

    def test_job_id_uses_the_new_naming_scheme(self):
        manifest, _ = brm.build_manifest(
            positives([("DB00035", "P30518", "AVPR2", "F1", 2)]), drawn([]), salvage([])
        )
        self.assertEqual(manifest.iloc[0]["job_id"], "DB00035__P30518__AVPR2__F1__cav2")

    def test_row_order_is_deterministic(self):
        rows = [("DB00157", "P68363", "TUBA1B", "F1", 1),
                ("DB00035", "P30518", "AVPR2", "F1", 1)]
        a, _ = brm.build_manifest(positives(rows), drawn([]), salvage([]))
        b, _ = brm.build_manifest(positives(list(reversed(rows))), drawn([]), salvage([]))
        pd.testing.assert_frame_equal(a, b)

    def test_outputs_validate_against_their_schemas(self):
        manifest, redock = brm.build_manifest(
            positives([("DB00035", "P30518", "AVPR2", "F1", 1)]),
            drawn([("DB00157", "P68363", "TUBA1B", "F1", 2, "pool", "old_dir")]),
            salvage([]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            m = os.path.join(tmp, "run_manifest.csv")
            r = os.path.join(tmp, "redock_queue.csv")
            schemas.write_checked(manifest, m, "run_manifest")
            schemas.write_checked(redock, r, "redock_queue")
            self.assertEqual(len(schemas.read_checked(m, "run_manifest")), 2)
            self.assertEqual(len(schemas.read_checked(r, "redock_queue")), 1)

    def test_empty_inputs_produce_valid_empty_outputs(self):
        manifest, redock = brm.build_manifest(positives([]), drawn([]), salvage([]))
        self.assertTrue(manifest.empty)
        self.assertTrue(redock.empty)
        with tempfile.TemporaryDirectory() as tmp:
            schemas.write_checked(manifest, os.path.join(tmp, "m.csv"), "run_manifest")


class TestCli(unittest.TestCase):
    def test_end_to_end_through_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "pos.csv")
            d = os.path.join(tmp, "drawn.csv")
            s = os.path.join(tmp, "salv.csv")
            schemas.write_checked(
                positives([("DB00035", "P30518", "AVPR2", "F1", 1)]), p, "positives_planned"
            )
            schemas.write_checked(
                drawn([("DB00157", "P68363", "TUBA1B", "F1", 2, "pool", "old_dir")]),
                d, "negatives_drawn",
            )
            schemas.write_checked(
                salvage([("old_dir2", "true_positive",
                          "DB00035", "P30518", "AVPR2", "F1", 1, "AVPR2")]),
                s, "salvage_map",
            )
            out = os.path.join(tmp, "run_manifest.csv")
            queue = os.path.join(tmp, "redock_queue.csv")
            rc = brm.main(
                ["--positives", p, "--drawn", d, "--salvage-map", s,
                 "--out", out, "--redock-out", queue]
            )
            self.assertEqual(rc, 0)
            manifest = schemas.read_checked(out, "run_manifest")
            self.assertEqual(len(manifest), 2)
            self.assertEqual(
                manifest.set_index("sample_type")["provenance"].to_dict(),
                {"positive": "salvaged_from", "negative": "salvaged_from"},
            )
            self.assertEqual(len(schemas.read_checked(queue, "redock_queue")), 0)

    def test_missing_input_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = brm.main(
                ["--positives", os.path.join(tmp, "nope.csv"),
                 "--drawn", os.path.join(tmp, "nope2.csv"),
                 "--salvage-map", os.path.join(tmp, "nope3.csv"),
                 "--out", os.path.join(tmp, "out.csv")]
            )
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
