"""Tests for dock_selectivity.invariants.

Every test here reproduces, in miniature, a failure the 2025-06-29 campaign
actually shipped. Fixtures use the real identifiers from the incident so a
failing assertion reads like the incident report:

* ``Q8NCT1`` is ARRDC4. The real AVPR2 (vasopressin V2 receptor) is ``P30518``.
  ``dict(zip(Mapped_Gene_Name, Entry))`` returned Q8NCT1 for AVPR2 because it
  was last, and Q8NCT1 is the protein that got docked.
* ``TUBA1B`` (real accession P68363) was claimed by TUBB3 (Q13509), TUBGCP6
  (Q96RT7) and TUBGCP3 (Q96CW5) as well.

Hermetic: no network, no external drive, no /opt/data. The only subprocesses
are ``sys.executable -c '<snippet>'`` for the PYTHONHASHSEED check.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import tempfile
import unittest

import pandas as pd

from dock_selectivity.invariants import (
    InvariantViolation,
    assert_collapse_factor_one,
    assert_deterministic,
    assert_deterministic_across_hash_seeds,
    assert_no_row_growth,
    assert_self_consistent_labels,
    assert_total_accounting,
    assert_unique_key,
    cardinality_report,
    deterministic_fingerprint,
    write_invariant_report,
)

# The real symbol -> accession pairs behind the incident. Building a dict from
# these keeps only the LAST accession per symbol.
CORRUPT_PAIRS = [
    ("AVPR2", "P30518"),  # the real V2 receptor
    ("AVPR2", "Q8NCT1"),  # ARRDC4 — free-text hit, and the one dict() kept
    ("TUBA1B", "P68363"),  # the real tubulin alpha-1B
    ("TUBA1B", "Q13509"),  # TUBB3
    ("TUBA1B", "Q96RT7"),  # TUBGCP6
    ("TUBA1B", "Q96CW5"),  # TUBGCP3
    ("IL2", "P60568"),  # the real interleukin-2
    ("IL2", "P01589"),  # IL2RA
]

# Resolution on the clean key: Entrez -> accession, strictly 1:1.
CLEAN_PAIRS = [
    (554, "P30518"),  # AVPR2
    (10970, "Q8NCT1"),  # ARRDC4
    (10376, "P68363"),  # TUBA1B
    (3558, "P60568"),  # IL2
]

# Independent ground truth: accession -> its own symbols (UniProt 'Gene Names').
SYMBOLS = {
    "P30518": ("AVPR2", "ADHR", "DIR", "DIR3", "V2R"),
    "Q8NCT1": ("ARRDC4",),
    "P68363": ("TUBA1B", "TUBA1", "TUBAL1"),
    "Q13509": ("TUBB3", "TUBB4"),
    "P60568": ("IL2",),
    "P01589": ("IL2RA", "IL2R"),
    "Q7Z2W7": ("MT-ND1",),  # hyphenated symbol, as in the 34 hyphenated job dirs
}


class TestCollapseFactor(unittest.TestCase):
    """G5 / I2 — a key -> value map must be a function."""

    def test_many_to_one_map_raises(self):
        with self.assertRaises(InvariantViolation) as ctx:
            assert_collapse_factor_one(CORRUPT_PAIRS, "symbol_to_accession")
        message = str(ctx.exception)
        self.assertIn("symbol_to_accession", message)
        self.assertIn("max_collapse=4", message)  # TUBA1B claimed by 4 accessions
        self.assertIn("TUBA1B", message)
        self.assertIn("keeps the LAST value", message)
        self.assertFalse(ctx.exception.report["ok"])
        self.assertEqual(ctx.exception.report["max_collapse"], 4)

    def test_one_to_one_map_passes_with_collapse_factor_one(self):
        report = assert_collapse_factor_one(CLEAN_PAIRS, "entrez_to_accession")
        self.assertTrue(report["ok"])
        self.assertEqual(report["max_collapse"], 1)
        self.assertEqual(report["mean_collapse"], 1.0)
        self.assertEqual(report["n_keys"], 4)
        self.assertEqual(report["n_keys_collapsed"], 0)

    def test_cardinality_report_reproduces_the_incident_shape(self):
        report = cardinality_report(CORRUPT_PAIRS, name="uniprot_gene_mapping")
        self.assertEqual(report["n_pairs"], 8)
        self.assertEqual(report["n_keys"], 3)
        self.assertEqual(report["n_distinct_values"], 8)
        self.assertEqual(report["max_collapse"], 4)
        self.assertGreater(report["mean_collapse"], 2.0)  # the real file: 2.55
        self.assertEqual(report["n_keys_collapsed"], 3)
        self.assertFalse(report["ok"])
        # worst-first, deterministic ordering
        self.assertEqual(report["worst_keys"][0]["key"], "TUBA1B")

    def test_a_built_dict_is_refused_because_it_has_already_collapsed(self):
        collapsed = dict(CORRUPT_PAIRS)  # exactly the 2025 bug
        self.assertEqual(collapsed["AVPR2"], "Q8NCT1")  # last-wins, the docked one
        with self.assertRaises(ValueError) as ctx:
            cardinality_report(collapsed, name="already_collapsed")
        self.assertIn("already collapsed", str(ctx.exception))
        self.assertIn("Pass the (key, value) pairs", str(ctx.exception))

    def test_grouped_mapping_and_dataframe_and_series_inputs_agree(self):
        grouped = {"TUBA1B": ["P68363", "Q13509", "Q96RT7", "Q96CW5"]}
        frame = pd.DataFrame(CORRUPT_PAIRS, columns=["symbol", "accession"])
        series = pd.Series(
            [v for _, v in CORRUPT_PAIRS], index=[k for k, _ in CORRUPT_PAIRS]
        )
        self.assertEqual(cardinality_report(grouped)["max_collapse"], 4)
        self.assertEqual(
            cardinality_report(frame, key="symbol", value="accession")["max_collapse"],
            4,
        )
        self.assertEqual(cardinality_report(frame)["max_collapse"], 4)
        self.assertEqual(cardinality_report(series)["max_collapse"], 4)

    def test_nullable_integer_keys_keep_their_identity_in_the_report(self):
        """An Entrez ID must be reported as 554, never as '554.0'.

        The identity index resolves on a nullable Int64 Entrez column; naive
        numpy conversion floats it as soon as one value is missing, and the
        report then names a gene id that does not exist.
        """
        frame = pd.DataFrame(
            {
                "entrez_id": pd.array([554, 554, 10970, None], dtype="Int64"),
                "accession": pd.array(
                    ["P30518", "Q8NCT1", "Q8NCT1", None], dtype="string"
                ),
            }
        )
        report = cardinality_report(
            frame, key="entrez_id", value="accession", name="entrez_to_accession"
        )
        self.assertEqual(report["worst_keys"][0]["key"], "554")
        self.assertEqual(report["n_null_keys"], 1)
        self.assertEqual(report["max_collapse"], 2)

    def test_duplicate_pairs_are_not_a_collapse(self):
        # The same (key, value) twice is redundancy, not ambiguity.
        report = assert_collapse_factor_one(
            [("AVPR2", "P30518"), ("AVPR2", "P30518")], "duplicated_edge"
        )
        self.assertEqual(report["max_collapse"], 1)

    def test_warn_only_reports_without_raising(self):
        with self.assertLogs("dock_selectivity.invariants", level="WARNING") as logs:
            report = assert_collapse_factor_one(
                CORRUPT_PAIRS, "symbol_to_accession", warn_only=True
            )
        self.assertFalse(report["ok"])
        self.assertIn("many-to-one", "".join(logs.output))


class TestSelfConsistentLabels(unittest.TestCase):
    """I1 — the check that would have caught the incident on day one."""

    def _mapping_frame(self):
        """A miniature uniprot_gene_mapping.csv, corrupt exactly as shipped."""
        return pd.DataFrame(
            {
                "Entry": ["P30518", "Q8NCT1", "P68363", "Q13509", "P60568", "P01589"],
                "Mapped_Gene_Name": [
                    "AVPR2",
                    "AVPR2",  # <- Q8NCT1 (ARRDC4) relabelled as AVPR2
                    "TUBA1B",
                    "TUBA1B",  # <- Q13509 (TUBB3) relabelled as TUBA1B
                    "IL2",
                    "IL2",  # <- P01589 (IL2RA) relabelled as IL2
                ],
            }
        )

    def test_flags_the_row_whose_label_belongs_to_a_different_protein(self):
        frame = self._mapping_frame()
        with self.assertRaises(InvariantViolation) as ctx:
            assert_self_consistent_labels(
                frame, "Entry", "Mapped_Gene_Name", SYMBOLS, "uniprot_gene_mapping"
            )
        message = str(ctx.exception)
        self.assertIn("Q8NCT1", message)
        self.assertIn("AVPR2", message)
        self.assertIn("ARRDC4", message)  # what Q8NCT1 really is
        report = ctx.exception.report
        self.assertEqual(report["n_rows"], 6)
        self.assertEqual(report["n_violations"], 3)
        self.assertEqual(report["n_ok"], 3)
        self.assertAlmostEqual(report["violation_rate"], 0.5)  # real file: 0.577

    def test_corrected_mapping_passes(self):
        corrected = pd.DataFrame(
            {
                "Entry": ["P30518", "Q8NCT1", "P68363", "Q13509", "Q7Z2W7"],
                "gene": ["AVPR2", "ARRDC4", "TUBA1B", "TUBB3", "MT-ND1"],
            }
        )
        report = assert_self_consistent_labels(
            corrected, "Entry", "gene", SYMBOLS, "identity_index"
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["n_violations"], 0)
        self.assertEqual(report["n_ok"], 5)

    def test_synonym_counts_as_consistent(self):
        # V2R is a listed synonym of P30518; a label need not be the primary.
        frame = pd.DataFrame({"acc": ["P30518"], "gene": ["V2R"]})
        report = assert_self_consistent_labels(
            frame, "acc", "gene", SYMBOLS, "synonym_ok"
        )
        self.assertTrue(report["ok"])

    def test_callable_truth_lookup_from_an_index_like_object(self):
        def label_of(accession):
            return SYMBOLS[accession]  # raises KeyError like IdentityIndex

        frame = pd.DataFrame({"acc": ["P30518", "Q99999"], "gene": ["AVPR2", "FAKE1"]})
        with self.assertRaises(InvariantViolation) as ctx:
            assert_self_consistent_labels(frame, "acc", "gene", label_of, "index_check")
        self.assertEqual(ctx.exception.report["n_missing_keys"], 1)
        self.assertIn("absent from the truth source", str(ctx.exception))

    def test_null_label_is_a_violation_not_a_pass(self):
        # 13,452/13,452 negatives shipped with a blank gene name in 2025.
        frame = pd.DataFrame({"acc": ["P30518", "P68363"], "gene": ["AVPR2", None]})
        with self.assertRaises(InvariantViolation) as ctx:
            assert_self_consistent_labels(frame, "acc", "gene", SYMBOLS, "blank_gene")
        self.assertEqual(ctx.exception.report["n_null_labels"], 1)

    def test_null_key_is_a_violation(self):
        frame = pd.DataFrame({"acc": [None], "gene": ["AVPR2"]})
        with self.assertRaises(InvariantViolation) as ctx:
            assert_self_consistent_labels(frame, "acc", "gene", SYMBOLS, "null_key")
        self.assertEqual(ctx.exception.report["n_null_keys"], 1)

    def test_missing_column_is_a_programming_error(self):
        frame = pd.DataFrame({"acc": ["P30518"]})
        with self.assertRaises(ValueError):
            assert_self_consistent_labels(frame, "acc", "gene", SYMBOLS, "missing_col")

    def test_truth_derived_from_the_frame_can_never_fail(self):
        """Documents the governing principle, with evidence.

        ``validate_negatives()`` re-checked negatives against the same
        ``known_pairs`` object that produced them and reported "All validations
        passed" while emitting 5 real interactions as negatives. Here the same
        mistake — deriving ``truth_lookup`` from the frame under test — makes
        the corrupt mapping pass. ``truth_lookup`` must come from a primary
        source.
        """
        frame = self._mapping_frame()
        self_derived = dict(zip(frame["Entry"], frame["Mapped_Gene_Name"]))
        report = assert_self_consistent_labels(
            frame, "Entry", "Mapped_Gene_Name", self_derived, "self_referential"
        )
        self.assertTrue(report["ok"])  # passes, and means nothing


class TestTotalAccounting(unittest.TestCase):
    """I5 — no silent drops."""

    def test_balanced_ledger_passes(self):
        report = assert_total_accounting(8568, 8331, 237, "resolve_interactions")
        self.assertTrue(report["ok"])
        self.assertEqual(report["n_unaccounted"], 0)

    def test_dropped_rows_raise_with_the_count(self):
        # 237 symbol-less edges and 222 slug drug rows vanished in 2025.
        with self.assertRaises(InvariantViolation) as ctx:
            assert_total_accounting(8568, 8331, 0, "resolve_interactions")
        message = str(ctx.exception)
        self.assertIn("237", message)
        self.assertIn("rejects", message)
        self.assertEqual(ctx.exception.report["n_unaccounted"], 237)

    def test_double_counting_is_also_caught(self):
        with self.assertRaises(InvariantViolation):
            assert_total_accounting(10, 8, 5, "double_counted")

    def test_negative_counts_are_a_programming_error(self):
        with self.assertRaises(ValueError):
            assert_total_accounting(10, -1, 11, "bad")


class TestNoRowGrowth(unittest.TestCase):
    """I4 — the 410,087-row fan-out."""

    def test_growth_raises_with_the_delta(self):
        with self.assertRaises(InvariantViolation) as ctx:
            assert_no_row_growth(16_600_000, 17_010_087, "annotate_sample_types")
        self.assertIn("410087", str(ctx.exception).replace(",", ""))
        self.assertEqual(ctx.exception.report["row_delta"], 410_087)

    def test_frames_are_accepted_as_well_as_counts(self):
        before = pd.DataFrame({"a": [1, 2, 3]})
        after = pd.DataFrame({"a": [1, 1, 2, 3]})
        with self.assertRaises(InvariantViolation):
            assert_no_row_growth(before, after, "frame_growth")
        self.assertTrue(assert_no_row_growth(before, before, "same")["ok"])

    def test_shrinkage_allowed_by_default_and_catchable_on_request(self):
        self.assertTrue(assert_no_row_growth(100, 90, "filtered")["ok"])
        with self.assertRaises(InvariantViolation) as ctx:
            assert_no_row_growth(100, 90, "must_not_shrink", allow_shrinkage=False)
        self.assertIn("shrank by 10", str(ctx.exception))


class TestUniqueKey(unittest.TestCase):
    """The 1,428 duplicate merge keys that made the fan-out possible."""

    def test_duplicate_key_raises_and_names_the_worst_offender(self):
        frame = pd.DataFrame(
            {
                "uniprot_id": ["P30518"] * 12 + ["P68363"],
                "drugbank_id": ["DB00035"] * 12 + ["DB00091"],
                "cavity_index": [2] * 12 + [1],
                "fragment_id": ["F{0}".format(i) for i in range(1, 13)] + ["F1"],
            }
        )
        with self.assertRaises(InvariantViolation) as ctx:
            assert_unique_key(
                frame,
                ["uniprot_id", "drugbank_id", "cavity_index"],
                "sample_type_key",
            )
        report = ctx.exception.report
        self.assertEqual(report["n_duplicate_keys"], 1)
        self.assertEqual(report["max_multiplicity"], 12)  # the real worst job
        self.assertIn("Fragment_ID", str(ctx.exception))

    def test_adding_the_fragment_makes_the_key_unique(self):
        frame = pd.DataFrame(
            {
                "uniprot_id": ["P30518", "P30518"],
                "drugbank_id": ["DB00035", "DB00035"],
                "cavity_index": [2, 2],
                "fragment_id": ["F1", "F2"],
            }
        )
        report = assert_unique_key(
            frame,
            ["uniprot_id", "drugbank_id", "cavity_index", "fragment_id"],
            "sample_type_key",
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["n_distinct_keys"], 2)

    def test_null_keys_are_counted_not_ignored(self):
        frame = pd.DataFrame({"acc": ["P30518", None, None]})
        with self.assertRaises(InvariantViolation) as ctx:
            assert_unique_key(frame, "acc", "null_keys")
        self.assertEqual(ctx.exception.report["n_null_key_rows"], 2)


class TestDeterminism(unittest.TestCase):
    """G3 / I6 — the same input must give byte-identical output."""

    def test_nondeterministic_callable_is_caught(self):
        counter = itertools.count()

        with self.assertRaises(InvariantViolation) as ctx:
            assert_deterministic(lambda: next(counter), "counter")
        self.assertIn("not reproducible", str(ctx.exception))
        self.assertEqual(ctx.exception.report["n_distinct_fingerprints"], 2)

    def test_unseeded_rng_is_caught(self):
        rng = random.Random()
        with self.assertRaises(InvariantViolation):
            assert_deterministic(lambda: [rng.random() for _ in range(5)], "unseeded")

    def test_deterministic_callable_passes(self):
        report = assert_deterministic(
            lambda: sorted(["Q96RT7", "P68363", "Q13509"]), "sorted_accessions"
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["n_distinct_fingerprints"], 1)

    def test_dataframe_results_are_compared_by_content(self):
        def build():
            return pd.DataFrame({"accession": ["P30518", "P68363"], "n": [1, 2]})

        self.assertTrue(assert_deterministic(build, "frame")["ok"])

        flip = itertools.count()

        def unstable():
            order = ["P30518", "P68363"]
            if next(flip) % 2:
                order.reverse()  # same set, different order -> must be caught
            return pd.DataFrame({"accession": order})

        with self.assertRaises(InvariantViolation):
            assert_deterministic(unstable, "unstable_order")

    def test_fingerprint_is_content_addressed_and_order_sensitive(self):
        self.assertEqual(
            deterministic_fingerprint(["P30518", "Q8NCT1"]),
            deterministic_fingerprint(["P30518", "Q8NCT1"]),
        )
        self.assertNotEqual(
            deterministic_fingerprint(["P30518", "Q8NCT1"]),
            deterministic_fingerprint(["Q8NCT1", "P30518"]),
        )

    def test_fingerprint_of_a_path_hashes_file_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "identity_index.parquet")
            second = os.path.join(tmp, "copy.parquet")
            frame = pd.DataFrame({"accession": ["P30518"], "symbol": ["AVPR2"]})
            frame.to_parquet(first)
            frame.to_parquet(second)
            import pathlib

            self.assertEqual(
                deterministic_fingerprint(pathlib.Path(first)),
                deterministic_fingerprint(pathlib.Path(second)),
            )
            with open(second, "ab") as handle:
                handle.write(b"\x00")
            self.assertNotEqual(
                deterministic_fingerprint(pathlib.Path(first)),
                deterministic_fingerprint(pathlib.Path(second)),
            )

    def test_hash_seed_dependent_output_is_caught(self):
        """prepare_uniprot_mapping.py:51 — ``list(set(...))``.

        Its query order was hash-randomised and a first-wins
        ``drop_duplicates`` then let the interpreter's hash seed decide which
        protein owned an accession. Invisible in-process; visible here.
        """
        with self.assertRaises(InvariantViolation) as ctx:
            assert_deterministic_across_hash_seeds(
                "print(hash('AVPR2'))", "hash_seed_dependent", seeds=(0, 1, 2)
            )
        self.assertIn("PYTHONHASHSEED", str(ctx.exception))
        self.assertEqual(len(ctx.exception.report["runs"]), 3)

    def test_hash_seed_independent_output_passes(self):
        report = assert_deterministic_across_hash_seeds(
            "print(sorted(['Q96RT7', 'P68363', 'Q13509']))",
            "sorted_output",
            seeds=(0, 1, 2),
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["n_distinct_outputs"], 1)

    def test_a_failing_snippet_is_reported_not_silently_passed(self):
        with self.assertRaises(InvariantViolation) as ctx:
            assert_deterministic_across_hash_seeds(
                "raise SystemExit(3)", "broken_snippet", seeds=(0, 1)
            )
        self.assertIn("exited 3", str(ctx.exception))


class TestExceptionHierarchy(unittest.TestCase):
    """One exception type per package, or ``except`` clauses silently miss."""

    def test_schema_violation_is_an_invariant_violation(self):
        from dock_selectivity.schemas import SchemaViolation

        self.assertTrue(issubclass(SchemaViolation, InvariantViolation))

    def test_identity_module_shares_the_canonical_exception(self):
        """``invariants`` is the canonical home of ``InvariantViolation``.

        Skips (rather than fails) while ``identity.py`` still defines its own,
        because that file belongs to another change. The fix is one line in
        ``dock_selectivity/identity.py``::

            from .invariants import InvariantViolation

        Until then, ``except InvariantViolation`` catches only the half of the
        package the name was imported from.
        """
        try:
            from dock_selectivity import identity
        except ImportError:  # pragma: no cover - identity.py not present yet
            self.skipTest("dock_selectivity.identity is not importable yet")
        other = getattr(identity, "InvariantViolation", None)
        if other is None:  # pragma: no cover
            self.skipTest("identity.py does not expose InvariantViolation")
        if other is not InvariantViolation:
            self.skipTest(
                "identity.py defines its own InvariantViolation; it should do "
                "'from .invariants import InvariantViolation' instead, so that "
                "one except clause covers the whole package"
            )
        self.assertIs(other, InvariantViolation)


class TestInvariantReport(unittest.TestCase):
    def test_report_file_is_deterministic_and_json(self):
        reports = [
            assert_collapse_factor_one(CLEAN_PAIRS, "entrez_to_accession"),
            assert_total_accounting(8568, 8331, 237, "resolve_interactions"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "nested", "stage.invariants.json")
            second = os.path.join(tmp, "nested", "stage2.invariants.json")
            write_invariant_report(reports, first)
            write_invariant_report(reports, second)
            with open(first, "rb") as handle:
                first_bytes = handle.read()
            with open(second, "rb") as handle:
                second_bytes = handle.read()
            self.assertEqual(first_bytes, second_bytes)
            payload = json.loads(first_bytes.decode("utf-8"))
            self.assertEqual(len(payload), 2)
            self.assertTrue(all(entry["ok"] for entry in payload))

    def test_failing_report_can_be_recorded_without_raising(self):
        with self.assertLogs("dock_selectivity.invariants", level="WARNING"):
            report = assert_collapse_factor_one(
                CORRUPT_PAIRS, "symbol_to_accession", warn_only=True
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stage.invariants.json")
            write_invariant_report(report, path)
            with open(path) as handle:
                payload = json.load(handle)
        self.assertFalse(payload[0]["ok"])
        self.assertEqual(payload[0]["max_collapse"], 4)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
