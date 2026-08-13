"""Tests for dock_selectivity.identity.

These are regression tests for a specific incident, not toy cases.  Every
fixture row below is a real row from the campaign's
``uniprot_gene_mapping.csv`` (values copied as literals -- no file is read, no
drive is touched):

    P30518 V2R_HUMAN   AVPR2 ADHR DIR DIR3 V2R      mislabelled map said AVPR2 -> Q8NCT1
    Q8NCT1 ARRD4_HUMAN ARRDC4                       ... which is ARRDC4, a different protein
    P68363 TBA1B_HUMAN TUBA1B                       TUBA1B -> Q13509 in the corrupt map
    Q13509 TBB3_HUMAN  TUBB3 TUBB4                  ... which is TUBB3
    P60568 IL2_HUMAN   IL2                          IL2 -> P01589 in the corrupt map
    P01589 IL2RA_HUMAN IL2RA                        ... which is IL2RA

Entrez ids for AVPR2 (554), TUBA1B (10376) and IL2 (3558) are the real ones;
the others are fixture values and are only ever asserted against themselves.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dock_selectivity.identity import (  # noqa: E402
    IDENTITY_INDEX_COLUMNS,
    IdentityIndex,
    InvariantViolation,
    Protein,
    Resolution,
    file_sha256,
    is_accession,
)

# (accession, entry_name, primary_symbol, symbols, entrez_id, reviewed)
FIXTURE_ROWS = [
    ("P30518", "V2R_HUMAN", "AVPR2", ["AVPR2", "ADHR", "DIR", "DIR3", "V2R"], 554, True),
    ("Q8NCT1", "ARRD4_HUMAN", "ARRDC4", ["ARRDC4"], 91947, True),
    ("P68363", "TBA1B_HUMAN", "TUBA1B", ["TUBA1B"], 10376, True),
    ("Q13509", "TBB3_HUMAN", "TUBB3", ["TUBB3", "TUBB4"], 10381, True),
    ("P60568", "IL2_HUMAN", "IL2", ["IL2"], 3558, True),
    ("P01589", "IL2RA_HUMAN", "IL2RA", ["IL2RA"], 3559, True),
    ("P03886", "NU1M_HUMAN", "MT-ND1", ["MT-ND1", "MTND1", "NADH1", "ND1"], 4535, True),
    ("P08F94", "PKHD1_HUMAN", "PKHD1", ["PKHD1", "FCYT", "TIGM1"], 5314, True),
    ("Q96RT7", "GCP6_HUMAN", "TUBGCP6", ["TUBGCP6", "GCP6", "KIAA1669"], 85378, True),
    # 10-character accession, unreviewed, no Entrez id: the shapes a builder
    # must carry without special-casing.
    ("A0A0B4J2D5", "GAT3B_HUMAN", "GATD3B", ["GATD3B"], None, False),
]


def make_frame(rows=None):
    """Build an identity-index DataFrame from ``FIXTURE_ROWS``-shaped tuples."""
    rows = FIXTURE_ROWS if rows is None else rows
    return pd.DataFrame(
        {
            "accession": [r[0] for r in rows],
            "entry_name": [r[1] for r in rows],
            "primary_symbol": [r[2] for r in rows],
            "symbols": [list(r[3]) for r in rows],
            "entrez_id": pd.array([r[4] for r in rows], dtype="Int64"),
            "reviewed": [r[5] for r in rows],
        },
        columns=list(IDENTITY_INDEX_COLUMNS),
    )


def make_index(rows=None, **kwargs):
    return IdentityIndex.from_frame(make_frame(rows), **kwargs)


class TestAcceptanceCriteria(unittest.TestCase):
    """FIX_PLAN section 1 'Acceptance criteria', verbatim."""

    def setUp(self):
        self.index = make_index()

    def test_entrez_resolves_to_the_right_accession(self):
        self.assertEqual(self.index.by_entrez(554).accession, "P30518")  # AVPR2
        self.assertEqual(self.index.by_entrez(10376).accession, "P68363")  # TUBA1B
        self.assertEqual(self.index.by_entrez(3558).accession, "P60568")  # IL2

    def test_the_corrupt_map_answers_are_now_impossible(self):
        # The retired map said AVPR2 -> Q8NCT1, TUBA1B -> Q13509, IL2 -> P01589.
        # Those accessions exist, and are other proteins.  Labels only ever
        # flow accession -> symbol, so the wrong answer has nowhere to come from.
        self.assertEqual(self.index.label("Q8NCT1"), "ARRDC4")
        self.assertEqual(self.index.label("Q13509"), "TUBB3")
        self.assertEqual(self.index.label("P01589"), "IL2RA")
        self.assertNotEqual(self.index.label("Q8NCT1"), "AVPR2")

    def test_max_collapse_is_exactly_one(self):
        report = self.index.cardinality_report()
        self.assertEqual(report["max_collapse"], 1)
        self.assertEqual(report["n_entrez_collapsing"], 0)
        self.assertEqual(report["mean_collapse"], 1.0)

    def test_known_issue_regression_assertion_passes(self):
        # "Mapped_Gene_Name must be a symbol or synonym of the entry it labels"
        for protein in self.index:
            self.assertTrue(
                protein.has_symbol(protein.primary_symbol),
                "{0}: {1} not in {2}".format(
                    protein.accession, protein.primary_symbol, protein.symbols
                ),
            )

    def test_cardinality_report_is_json_serialisable_and_complete(self):
        report = self.index.cardinality_report()
        json.dumps(report)  # must not raise
        for key in (
            "n_accessions",
            "n_entrez",
            "max_collapse",
            "n_unreviewed",
            "n_records",
            "n_symbols",
        ):
            self.assertIn(key, report)
        self.assertEqual(report["n_accessions"], len(FIXTURE_ROWS))
        self.assertEqual(report["n_records"], len(FIXTURE_ROWS))
        self.assertEqual(report["n_unreviewed"], 1)
        self.assertEqual(report["n_records_without_entrez"], 1)
        self.assertEqual(report["n_entrez"], len(FIXTURE_ROWS) - 1)


class TestCollapseFactorEnforcement(unittest.TestCase):
    """G5: entrez -> accession collapse factor must be exactly 1.

    A last-wins dict collapse must be impossible to construct silently.
    """

    def colliding_rows(self):
        rows = list(FIXTURE_ROWS)
        # Q8NCT1 (ARRDC4) claims AVPR2's Entrez id -- exactly the shape of the
        # retired map, where AVPR2 resolved to Q8NCT1 because it was written last.
        rows[1] = ("Q8NCT1", "ARRD4_HUMAN", "ARRDC4", ["ARRDC4"], 554, True)
        return rows

    def test_from_frame_refuses_a_colliding_index(self):
        with self.assertRaises(InvariantViolation) as ctx:
            make_index(self.colliding_rows())
        message = str(ctx.exception)
        self.assertIn("collapse factor must be exactly 1", message)
        self.assertIn("554", message)
        self.assertIn("P30518", message)
        self.assertIn("Q8NCT1", message)

    def test_assert_self_consistent_fires_on_an_unvalidated_index(self):
        index = make_index(self.colliding_rows(), validate=False)
        self.assertEqual(index.cardinality_report()["max_collapse"], 2)
        with self.assertRaises(InvariantViolation):
            index.assert_self_consistent()

    def test_by_entrez_refuses_to_pick_a_winner(self):
        index = make_index(self.colliding_rows(), validate=False)
        # Not KeyError: a caller's `except KeyError` must not swallow a broken
        # index, and returning either accession is the original bug.
        with self.assertRaises(InvariantViolation):
            index.by_entrez(554)

    def test_resolve_entrez_reports_ambiguity_with_candidates(self):
        index = make_index(self.colliding_rows(), validate=False)
        resolution = index.resolve_entrez(554)
        self.assertEqual(resolution.status, "AMBIGUOUS")
        self.assertIsNone(resolution.protein)
        self.assertTrue(resolution.reason)
        self.assertEqual(resolution.candidates, ("P30518", "Q8NCT1"))
        self.assertEqual(resolution.as_reject_row()["candidates"], "P30518;Q8NCT1")

    def test_duplicate_accession_rows_are_not_swallowed_by_a_dict(self):
        rows = list(FIXTURE_ROWS)
        rows.append(("P30518", "V2R_HUMAN", "AVPR2", ["AVPR2"], 554, False))
        with self.assertRaises(InvariantViolation) as ctx:
            make_index(rows)
        message = str(ctx.exception)
        self.assertIn("one row per accession", message)
        self.assertIn("P30518 x2", message)


class TestOtherInvariants(unittest.TestCase):
    def test_primary_symbol_must_be_one_of_the_symbols(self):
        rows = list(FIXTURE_ROWS)
        # The literal corruption: Q96RT7 is TUBGCP6, labelled TUBA1B.
        rows[8] = (
            "Q96RT7",
            "GCP6_HUMAN",
            "TUBA1B",
            ["TUBGCP6", "GCP6", "KIAA1669"],
            85378,
            True,
        )
        with self.assertRaises(InvariantViolation) as ctx:
            make_index(rows)
        message = str(ctx.exception)
        self.assertIn("primary_symbol in symbols", message)
        self.assertIn("Q96RT7", message)

    def test_isoform_accessions_are_rejected(self):
        rows = list(FIXTURE_ROWS)
        rows.append(("P30518-2", "V2R_HUMAN", "AVPR2", ["AVPR2"], 55400, True))
        with self.assertRaises(InvariantViolation) as ctx:
            make_index(rows)
        self.assertIn("no isoform accessions", str(ctx.exception))

    def test_a_symbol_in_the_accession_column_is_rejected(self):
        rows = list(FIXTURE_ROWS)
        rows.append(("AVPR2", "V2R_HUMAN", "AVPR2", ["AVPR2"], 55401, True))
        with self.assertRaises(InvariantViolation) as ctx:
            make_index(rows)
        self.assertIn("not valid UniProtKB accessions", str(ctx.exception))

    def test_all_problems_are_reported_together(self):
        rows = list(FIXTURE_ROWS)
        rows.append(("P30518-2", "V2R_HUMAN", "AVPR2", ["AVPR2"], 554, True))
        with self.assertRaises(InvariantViolation) as ctx:
            make_index(rows)
        message = str(ctx.exception)
        self.assertIn("no isoform accessions", message)
        self.assertIn("collapse factor must be exactly 1", message)
        self.assertIn("failed 2 invariant(s)", message)

    def test_missing_column_raises_instead_of_defaulting(self):
        frame = make_frame().drop(columns=["entrez_id"])
        with self.assertRaises(InvariantViolation) as ctx:
            IdentityIndex.from_frame(frame)
        message = str(ctx.exception)
        self.assertIn("entrez_id", message)
        self.assertIn("Refusing to default", message)

    def test_symbols_as_a_bare_string_raises_not_split_into_letters(self):
        frame = make_frame()
        frame.loc[0, "symbols"] = "AVPR2"
        with self.assertRaises(InvariantViolation) as ctx:
            IdentityIndex.from_frame(frame)
        self.assertIn("list of strings", str(ctx.exception))

    def test_null_primary_symbol_raises(self):
        rows = list(FIXTURE_ROWS)
        rows[0] = ("P30518", "V2R_HUMAN", None, ["AVPR2"], 554, True)
        with self.assertRaises(InvariantViolation) as ctx:
            make_index(rows)
        self.assertIn("primary_symbol", str(ctx.exception))

    def test_non_numeric_entrez_raises_at_build(self):
        frame = make_frame()
        frame["entrez_id"] = frame["entrez_id"].astype(object)
        frame.loc[0, "entrez_id"] = "AVPR2"
        with self.assertRaises(InvariantViolation) as ctx:
            IdentityIndex.from_frame(frame)
        self.assertIn("not numeric", str(ctx.exception))


class TestResolution(unittest.TestCase):
    """An unresolved input is recorded with a reason, never dropped."""

    def setUp(self):
        self.index = make_index(obsolete_entrez={11111: "merged into Entrez 554"})

    def test_resolve_entrez_never_raises_and_always_explains(self):
        bad_inputs = [
            None,
            float("nan"),
            "",
            "nan",
            "ABC",
            "AVPR2",
            -1,
            0,
            3.5,
            999999999,
            11111,
            pd.NA,
            [],
        ]
        for value in bad_inputs:
            with self.subTest(value=repr(value)):
                resolution = self.index.resolve_entrez(value)  # must not raise
                self.assertIsInstance(resolution, Resolution)
                self.assertNotEqual(resolution.status, "OK")
                self.assertFalse(resolution.ok)
                self.assertIsNone(resolution.protein)
                self.assertTrue(
                    resolution.reason.strip(),
                    "status {0} came back with no reason".format(resolution.status),
                )
                row = resolution.as_reject_row()
                self.assertEqual(set(row), {"key", "status", "reason", "candidates"})
                self.assertTrue(row["reason"])

    def test_status_discriminates_no_key_from_unknown_from_obsolete(self):
        self.assertEqual(self.index.resolve_entrez(None).status, "NO_KEY")
        self.assertEqual(self.index.resolve_entrez("").status, "NO_KEY")
        self.assertEqual(self.index.resolve_entrez("not-a-number").status, "NO_KEY")
        self.assertEqual(self.index.resolve_entrez(999999999).status, "UNKNOWN")
        self.assertEqual(self.index.resolve_entrez(11111).status, "OBSOLETE")
        self.assertIn("retired", self.index.resolve_entrez(11111).reason)
        self.assertEqual(self.index.resolve_entrez(554).status, "OK")

    def test_ok_resolution_carries_the_protein_and_echoes_the_key(self):
        resolution = self.index.resolve_entrez("554")
        self.assertTrue(resolution.ok)
        self.assertEqual(resolution.unwrap().accession, "P30518")
        self.assertEqual(resolution.key, "554")

    def test_numeric_shapes_a_dataframe_actually_produces(self):
        for value in (554, "554", 554.0, "554.0"):
            with self.subTest(value=repr(value)):
                self.assertEqual(
                    self.index.resolve_entrez(value).unwrap().accession, "P30518"
                )

    def test_a_rejection_cannot_be_constructed_without_a_reason(self):
        with self.assertRaises(ValueError):
            Resolution(status="UNKNOWN", protein=None, reason="")
        with self.assertRaises(ValueError):
            Resolution(status="UNKNOWN", protein=None, reason="   ")
        with self.assertRaises(ValueError):
            Resolution(status="OK", protein=None, reason="")
        with self.assertRaises(ValueError):
            Resolution(status="WAT", protein=None, reason="x")

    def test_unwrap_on_a_rejection_raises_rather_than_returning_none(self):
        resolution = self.index.resolve_entrez(999999999)
        with self.assertRaises(InvariantViolation):
            resolution.unwrap()


class TestLookups(unittest.TestCase):
    def setUp(self):
        self.index = make_index()

    def test_by_accession_and_label(self):
        protein = self.index.by_accession("P30518")
        self.assertIsInstance(protein, Protein)
        self.assertEqual(protein.entry_name, "V2R_HUMAN")
        self.assertEqual(protein.symbols[0], "AVPR2")
        self.assertEqual(self.index.label("P30518"), "AVPR2")
        self.assertEqual(self.index.label(" p30518 "), "AVPR2")

    def test_unknown_accession_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.index.by_accession("P99999")
        with self.assertRaises(KeyError):
            self.index.label("P99999")
        with self.assertRaises(KeyError):
            self.index.by_accession(None)

    def test_isoform_lookup_raises_with_an_actionable_message(self):
        with self.assertRaises(KeyError) as ctx:
            self.index.by_accession("P30518-2")
        self.assertIn("isoform", str(ctx.exception))

    def test_by_entrez_unknown_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.index.by_entrez(999999999)
        with self.assertRaises(KeyError):
            self.index.by_entrez(None)

    def test_by_symbol_ambiguous_is_audit_only_and_returns_a_tuple(self):
        hits = self.index.by_symbol_ambiguous("TUBA1B")
        self.assertIsInstance(hits, tuple)
        self.assertEqual([p.accession for p in hits], ["P68363"])
        # A synonym resolves to its own entry, not to the queried name's entry.
        self.assertEqual(
            [p.accession for p in self.index.by_symbol_ambiguous("TUBB4")], ["Q13509"]
        )
        self.assertEqual(self.index.by_symbol_ambiguous("NOT_A_GENE"), ())
        self.assertEqual(self.index.by_symbol_ambiguous(None), ())

    def test_a_shared_symbol_returns_every_candidate_deterministically(self):
        rows = list(FIXTURE_ROWS)
        rows.append(("P0DTC1", "SHARE_HUMAN", "SHARED", ["SHARED", "AVPR2"], 77777, False))
        rows.append(("P0DTC2", "SHARE2_HUMAN", "SHARED2", ["SHARED2", "AVPR2"], 77778, True))
        index = make_index(rows)
        hits = index.by_symbol_ambiguous("AVPR2")
        # reviewed first, then accession -- a total order, never set iteration.
        self.assertEqual(
            [p.accession for p in hits], ["P0DTC2", "P30518", "P0DTC1"]
        )
        self.assertEqual(index.cardinality_report()["max_symbol_collapse"], 3)
        # ... and the accession key is still unambiguous.
        self.assertEqual(index.by_entrez(554).accession, "P30518")

    def test_membership_and_accession_listing(self):
        self.assertIn("P30518", self.index)
        self.assertIn("p30518", self.index)
        self.assertNotIn("P99999", self.index)
        self.assertEqual(len(self.index), len(FIXTURE_ROWS))
        self.assertEqual(list(self.index.accessions()), sorted(self.index.accessions()))

    def test_has_symbol_is_the_pair_validity_check(self):
        protein = self.index.by_accession("P03886")
        self.assertTrue(protein.has_symbol("MT-ND1"))
        self.assertTrue(protein.has_symbol("nd1"))
        self.assertFalse(protein.has_symbol("AVPR2"))
        self.assertFalse(protein.has_symbol(None))
        self.assertFalse(protein.has_symbol(float("nan")))

    def test_is_accession_helper(self):
        self.assertTrue(is_accession("P30518"))
        self.assertTrue(is_accession("A0A0B4J2D5"))
        self.assertFalse(is_accession("AVPR2"))
        self.assertFalse(is_accession("P30518-2"))
        self.assertFalse(is_accession(None))


class TestLoadFromDisk(unittest.TestCase):
    def test_missing_index_raises_and_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "identity_index.parquet")
            with self.assertRaises(FileNotFoundError) as ctx:
                IdentityIndex.load(missing)
            message = str(ctx.exception)
            self.assertIn("build_identity_index.py", message)
            self.assertIn("no fallback", message)

    def test_round_trip_through_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "identity_index.parquet")
            make_index().to_frame().to_parquet(path, index=False)
            index = IdentityIndex.load(path)
            self.assertEqual(len(index), len(FIXTURE_ROWS))
            self.assertEqual(index.by_entrez(554).accession, "P30518")
            self.assertEqual(index.by_accession("A0A0B4J2D5").entrez_id, None)
            self.assertFalse(index.by_accession("A0A0B4J2D5").reviewed)
            self.assertEqual(
                index.by_accession("P03886").symbols,
                ("MT-ND1", "MTND1", "NADH1", "ND1"),
            )
            self.assertEqual(index.source_sha256, file_sha256(path))

    def test_sha256_pin_refuses_an_unexpected_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "identity_index.parquet")
            make_index().to_frame().to_parquet(path, index=False)
            digest = file_sha256(path)
            IdentityIndex.load(path, expect_sha256=digest)  # must not raise
            with self.assertRaises(InvariantViolation) as ctx:
                IdentityIndex.load(path, expect_sha256="0" * 64)
            self.assertIn("SHA256 mismatch", str(ctx.exception))


class TestDeterminism(unittest.TestCase):
    """G3: no set iteration, no insertion-order dependence."""

    def test_row_order_does_not_change_anything(self):
        forward = make_index()
        reversed_frame = make_frame().iloc[::-1].reset_index(drop=True)
        backward = IdentityIndex.from_frame(reversed_frame)
        self.assertEqual(forward.records, backward.records)
        self.assertEqual(
            json.dumps(forward.cardinality_report(), sort_keys=False),
            json.dumps(backward.cardinality_report(), sort_keys=False),
        )
        self.assertEqual(
            forward.to_frame().to_json(), backward.to_frame().to_json()
        )

    def test_byte_identical_output_under_three_python_hash_seeds(self):
        script = r'''
import hashlib, json, os, sys
sys.path.insert(0, os.environ["REPO_ROOT"])
import pandas as pd
from dock_selectivity.identity import IdentityIndex, IDENTITY_INDEX_COLUMNS

ROWS = json.load(open(os.environ["ROWS_JSON"]))
# Iterate a set of strings: this order is PYTHONHASHSEED-dependent, which is
# exactly how prepare_uniprot_mapping.py:51 (`list(set(...))`) became
# irreproducible.  The index must be immune to it.
order = sorted({r[0] for r in ROWS}, key=hash)
by_acc = {r[0]: r for r in ROWS}
rows = [by_acc[a] for a in order]
frame = pd.DataFrame({
    "accession": [r[0] for r in rows],
    "entry_name": [r[1] for r in rows],
    "primary_symbol": [r[2] for r in rows],
    "symbols": [list(r[3]) for r in rows],
    "entrez_id": pd.array([r[4] for r in rows], dtype="Int64"),
    "reviewed": [r[5] for r in rows],
}, columns=list(IDENTITY_INDEX_COLUMNS))
index = IdentityIndex.from_frame(frame)
out = os.path.join(os.environ["OUT_DIR"], "identity_index_%s.parquet" % os.environ["TAG"])
index.to_frame().to_parquet(out, index=False)
print("INPUT_ORDER " + ",".join(order))
print("PARQUET_SHA " + hashlib.sha256(open(out, "rb").read()).hexdigest())
print("REPORT " + json.dumps(index.cardinality_report()))
print("ACCESSIONS " + ",".join(index.accessions()))
print("SYMBOL_HITS " + ",".join(p.accession for p in index.by_symbol_ambiguous("AVPR2")))
'''
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "build_index.py")
            rows_path = os.path.join(tmp, "rows.json")
            with open(script_path, "w") as handle:
                handle.write(script)
            with open(rows_path, "w") as handle:
                json.dump(FIXTURE_ROWS, handle)

            outputs = []
            input_orders = []
            for seed in ("0", "1", "42"):
                env = dict(os.environ)
                env.update(
                    {
                        "PYTHONHASHSEED": seed,
                        "REPO_ROOT": REPO_ROOT,
                        "ROWS_JSON": rows_path,
                        "OUT_DIR": tmp,
                        "TAG": seed,
                    }
                )
                proc = subprocess.run(
                    [sys.executable, script_path],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
                self.assertEqual(
                    proc.returncode, 0, "seed {0} failed:\n{1}".format(seed, proc.stderr)
                )
                lines = proc.stdout.strip().splitlines()
                input_orders.append(lines[0])
                outputs.append("\n".join(lines[1:]))

            self.assertGreater(
                len(set(input_orders)),
                1,
                "hash seeds produced identical input orders; the test proved "
                "nothing. Orders seen: {0}".format(input_orders),
            )
            self.assertEqual(
                len(set(outputs)),
                1,
                "output differs across PYTHONHASHSEED values:\n{0}".format(
                    "\n---\n".join(sorted(set(outputs)))
                ),
            )


if __name__ == "__main__":
    unittest.main()
