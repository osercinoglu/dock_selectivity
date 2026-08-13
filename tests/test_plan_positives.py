"""Tests for plan_positives.py -- the positives_planned.csv producer.

The failure modes under test are the ones this project actually suffered:

* an edge that cannot be planned must be recorded with a reason, never dropped
  (``if not uniprot_id: continue`` lost 237 edges with no ledger);
* fragment ids must normalise, because the old campaign wrote ``'F1'`` for
  positives and bare ``'1'`` for negatives, and that mismatch made the sample-type
  merge non-unique and duplicated 410,087 rows;
* the cavity expansion must be right, because the original sampler balanced
  **jobs** rather than pairs, so a target's cavity count scales negative demand;
* output must be byte-identical across runs and hash seeds.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

import plan_positives
from dock_selectivity import schemas
from dock_selectivity.identity import IdentityIndex
from dock_selectivity.invariants import InvariantViolation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_index() -> IdentityIndex:
    """A small index carrying the real AVPR2/ARRDC4 collision."""
    return IdentityIndex.from_frame(
        pd.DataFrame(
            [
                {
                    "accession": "P30518",
                    "entry_name": "V2R_HUMAN",
                    "primary_symbol": "AVPR2",
                    "symbols": ["AVPR2", "ADHR", "DIR", "DIR3", "V2R"],
                    "entrez_id": 554,
                    "reviewed": True,
                },
                {
                    "accession": "Q8NCT1",
                    "entry_name": "ARRD4_HUMAN",
                    "primary_symbol": "ARRDC4",
                    "symbols": ["ARRDC4"],
                    "entrez_id": 91947,
                    "reviewed": True,
                },
                {
                    "accession": "P68363",
                    "entry_name": "TBA1B_HUMAN",
                    "primary_symbol": "TUBA1B",
                    "symbols": ["TUBA1B"],
                    "entrez_id": 10376,
                    "reviewed": True,
                },
            ]
        )
    )


def make_interactions(pairs) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "drugbank_id": d,
                "entrez_id": e,
                "accession": a,
                "gene": g,
                "selected_by": "entrez",
            }
            for (d, e, a, g) in pairs
        ]
    )


class TestCavityExpansion(unittest.TestCase):
    def test_one_row_per_cavity_of_the_target(self):
        """Positives are per-cavity; this factor scales the whole negative demand."""
        planned, rejected = plan_positives.plan_positives(
            interactions=make_interactions([("DB00035", 554, "P30518", "AVPR2")]),
            index=make_index(),
            cavities={("P30518", "F1"): [1, 2, 5]},
        )
        self.assertEqual(len(planned), 3)
        self.assertEqual(sorted(planned["cavity_index"]), [1, 2, 5])
        self.assertTrue(rejected.empty)

    def test_multiple_fragments_all_expand(self):
        planned, _ = plan_positives.plan_positives(
            interactions=make_interactions([("DB00035", 554, "P30518", "AVPR2")]),
            index=make_index(),
            cavities={("P30518", "F1"): [1], ("P30518", "F2"): [1, 3]},
        )
        self.assertEqual(len(planned), 3)
        self.assertEqual(sorted(set(planned["fragment_id"])), ["F1", "F2"])

    def test_bare_and_prefixed_fragment_ids_are_the_same_bucket(self):
        """'1' and 'F1' must not become two pockets -- that bug duplicated 410,087 rows."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cav.csv")
            pd.DataFrame(
                [
                    {"UniProt_ID": "P30518", "Fragment_ID": 1, "Cavity_Index": 1},
                    {"UniProt_ID": "P30518", "Fragment_ID": "F1", "Cavity_Index": 2},
                ]
            ).to_csv(path, index=False)
            cavities = plan_positives.load_cavity_index(path)
        self.assertEqual(list(cavities.keys()), [("P30518", "F1")])
        self.assertEqual(cavities[("P30518", "F1")], [1, 2])


class TestRejectLedger(unittest.TestCase):
    """Nothing may vanish between the edge list and the plan."""

    def test_no_cavity_is_recorded_not_dropped(self):
        planned, rejected = plan_positives.plan_positives(
            interactions=make_interactions([("DB00035", 554, "P30518", "AVPR2")]),
            index=make_index(),
            cavities={},
        )
        self.assertTrue(planned.empty)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected.iloc[0]["reason"], "NO_CAVITY")

    def test_missing_ligand_is_recorded(self):
        planned, rejected = plan_positives.plan_positives(
            interactions=make_interactions([("DB00035", 554, "P30518", "AVPR2")]),
            index=make_index(),
            cavities={("P30518", "F1"): [1]},
            available_ligands=set(),
        )
        self.assertTrue(planned.empty)
        self.assertEqual(rejected.iloc[0]["reason"], "NO_LIGAND_SDF")

    def test_non_small_molecule_is_recorded(self):
        planned, rejected = plan_positives.plan_positives(
            interactions=make_interactions([("DB00035", 554, "P30518", "AVPR2")]),
            index=make_index(),
            cavities={("P30518", "F1"): [1]},
            small_molecules=set(),
        )
        self.assertTrue(planned.empty)
        self.assertEqual(rejected.iloc[0]["reason"], "NOT_SMALL_MOLECULE")

    def test_unknown_accession_is_recorded(self):
        planned, rejected = plan_positives.plan_positives(
            interactions=make_interactions([("DB00035", 999, "Q99999", "NOPE")]),
            index=make_index(),
            cavities={("Q99999", "F1"): [1]},
        )
        self.assertTrue(planned.empty)
        self.assertEqual(rejected.iloc[0]["reason"], "UNKNOWN_ACCESSION")

    def test_totals_reconcile_exactly(self):
        """n_in == n_planned_pairs + n_rejected, asserted inside the function."""
        planned, rejected = plan_positives.plan_positives(
            interactions=make_interactions(
                [
                    ("DB00035", 554, "P30518", "AVPR2"),
                    ("DB00035", 91947, "Q8NCT1", "ARRDC4"),
                    ("DB00157", 10376, "P68363", "TUBA1B"),
                ]
            ),
            index=make_index(),
            cavities={("P30518", "F1"): [1, 2], ("P68363", "F1"): [1]},
        )
        n_pairs = len(planned.drop_duplicates(subset=["drugbank_id", "accession"]))
        self.assertEqual(n_pairs + len(rejected), 3)
        self.assertEqual(len(planned), 3)  # 2 cavities + 1 cavity

    def test_reject_reasons_are_from_the_declared_vocabulary(self):
        _, rejected = plan_positives.plan_positives(
            interactions=make_interactions([("DB00035", 554, "P30518", "AVPR2")]),
            index=make_index(),
            cavities={},
        )
        self.assertTrue(set(rejected["reason"]) <= set(plan_positives.REJECT_REASONS))


class TestIdentityDiscipline(unittest.TestCase):
    def test_gene_comes_from_the_index_not_the_edge_list(self):
        """The property whose absence let AVPR2 select Q8NCT1."""
        lying = make_interactions([("DB00035", 91947, "Q8NCT1", "AVPR2")])  # wrong gene
        planned, _ = plan_positives.plan_positives(
            interactions=lying,
            index=make_index(),
            cavities={("Q8NCT1", "F1"): [1]},
        )
        self.assertEqual(planned.iloc[0]["gene"], "ARRDC4")


class TestDeterminism(unittest.TestCase):
    def test_row_order_is_independent_of_input_order(self):
        pairs = [
            ("DB00157", 10376, "P68363", "TUBA1B"),
            ("DB00035", 554, "P30518", "AVPR2"),
        ]
        cav = {("P30518", "F1"): [2, 1], ("P68363", "F1"): [1]}
        a, _ = plan_positives.plan_positives(
            make_interactions(pairs), make_index(), cav
        )
        b, _ = plan_positives.plan_positives(
            make_interactions(list(reversed(pairs))), make_index(), cav
        )
        pd.testing.assert_frame_equal(a, b)

    def test_output_is_byte_identical_across_hash_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            outs = []
            for seed in ("0", "1", "12345"):
                out = self._run_cli(tmp, seed)
                with open(out, "rb") as handle:
                    outs.append(handle.read())
            self.assertEqual(outs[0], outs[1])
            self.assertEqual(outs[1], outs[2])

    def _run_cli(self, tmp: str, seed: str) -> str:
        idx = os.path.join(tmp, "index.parquet")
        if not os.path.exists(idx):
            make_index().to_frame().to_parquet(idx, index=False)
            inter = os.path.join(tmp, "inter.csv")
            schemas.write_checked(
                make_interactions(
                    [
                        ("DB00035", 554, "P30518", "AVPR2"),
                        ("DB00157", 10376, "P68363", "TUBA1B"),
                    ]
                ),
                inter,
                "interactions_resolved",
            )
            pd.DataFrame(
                [
                    {"UniProt_ID": "P30518", "Fragment_ID": "F1", "Cavity_Index": 2},
                    {"UniProt_ID": "P30518", "Fragment_ID": "F1", "Cavity_Index": 1},
                    {"UniProt_ID": "P68363", "Fragment_ID": "F1", "Cavity_Index": 1},
                ]
            ).to_csv(os.path.join(tmp, "cav.csv"), index=False)
        out = os.path.join(tmp, "planned_%s.csv" % seed)
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=REPO)
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(REPO, "plan_positives.py"),
                "--interactions", os.path.join(tmp, "inter.csv"),
                "--index", idx,
                "--cavities", os.path.join(tmp, "cav.csv"),
                "--out", out,
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return out


class TestCli(unittest.TestCase):
    def test_rejects_may_not_overwrite_phase_0_ledger(self):
        """audit.py's G2 check asserts that file's row count; clobbering it fails the gate."""
        with tempfile.TemporaryDirectory() as tmp:
            rc = plan_positives.main(
                [
                    "--interactions", os.path.join(tmp, "missing.csv"),
                    "--index", os.path.join(tmp, "missing.parquet"),
                    "--cavities", os.path.join(tmp, "missing_cav.csv"),
                    "--out", os.path.join(tmp, "out.csv"),
                    "--rejects", os.path.join(tmp, "unresolved_targets.csv"),
                ]
            )
            self.assertEqual(rc, 2)

    def test_output_validates_against_its_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            planned, _ = plan_positives.plan_positives(
                interactions=make_interactions([("DB00035", 554, "P30518", "AVPR2")]),
                index=make_index(),
                cavities={("P30518", "F1"): [1]},
            )
            path = os.path.join(tmp, "planned.csv")
            schemas.write_checked(planned, path, "positives_planned")
            back = schemas.read_checked(path, "positives_planned")
            self.assertEqual(len(back), 1)

    def test_bad_cavity_file_raises_actionably(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            pd.DataFrame([{"wrong": 1}]).to_csv(path, index=False)
            with self.assertRaises(InvariantViolation) as ctx:
                plan_positives.load_cavity_index(path)
            self.assertIn("UniProt_ID", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
