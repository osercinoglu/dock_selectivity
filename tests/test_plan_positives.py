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


def quality_row(accession, cavity, residues, fragment="F1"):
    """One ``cavity_quality`` row, every column derived from ``residues``.

    Mirrors ``score_cavity_quality.py``'s guarantee that ``residue_plddt`` *is*
    the row: the schema's row checks reject anything else.
    """
    values = [round(float(v), 2) for v in residues]
    ordered = sorted(values)

    def frac(threshold):
        return sum(1 for v in values if v >= threshold) / float(len(values))

    return {
        "accession": accession,
        "fragment_id": fragment,
        "cavity_index": int(cavity),
        "n_residues": len(values),
        "n_atoms": len(values) * 4,
        "mean_plddt": sum(values) / float(len(values)),
        "median_plddt": ordered[len(ordered) // 2],
        "min_plddt": min(values),
        "max_plddt": max(values),
        "frac_ge_50": frac(50.0),
        "frac_ge_70": frac(70.0),
        "frac_ge_90": frac(90.0),
        "residue_plddt": ";".join("%.2f" % v for v in values),
    }


def make_quality(rows) -> pd.DataFrame:
    return schemas.coerce(pd.DataFrame(rows), "cavity_quality")


#: 21 confident residues out of 50 -> frac_ge_70 = 0.42 exactly.
FAILING_POCKET = [90.0] * 21 + [30.0] * 29
#: 45 of 50 -> frac_ge_70 = 0.90.
PASSING_POCKET = [90.0] * 45 + [30.0] * 5


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


class TestPocketQualityFilter(unittest.TestCase):
    """The pLDDT filter: opt-in, applied per cavity, and never silent.

    Every drop here shrinks the positive set, and under balanced sampling it
    shrinks the negative set by exactly as much -- so the numbers have to reach
    the ledger, not only the log.
    """

    def test_a_pocket_below_the_fraction_is_dropped(self):
        result = plan_positives.filter_cavities_by_quality(
            cavities={("P30518", "F1"): [1]},
            quality_frame=make_quality([quality_row("P30518", 1, FAILING_POCKET)]),
            plddt_min=70.0,
            min_fraction=0.80,
        )
        self.assertEqual(result.cavities, {})
        self.assertEqual(result.n_considered, 1)
        self.assertEqual(result.n_passed, 0)
        self.assertEqual(result.n_failed, 1)
        self.assertEqual(result.n_missing, 0)

    def test_the_reject_detail_carries_the_actual_numbers(self):
        result = plan_positives.filter_cavities_by_quality(
            cavities={("P30518", "F1"): [1]},
            quality_frame=make_quality([quality_row("P30518", 1, FAILING_POCKET)]),
            plddt_min=70.0,
            min_fraction=0.80,
        )
        planned, rejected = plan_positives.plan_positives(
            interactions=make_interactions([("DB00035", 554, "P30518", "AVPR2")]),
            index=make_index(),
            cavities=result.cavities,
            quality_failures=result.failures,
        )
        self.assertTrue(planned.empty)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected.iloc[0]["reason"], "pocket_quality")
        self.assertIn(
            "frac_ge_70=0.42 < 0.80 (n_residues=50)", rejected.iloc[0]["detail"]
        )
        self.assertIn("F1/cav1", rejected.iloc[0]["detail"])
        self.assertTrue(
            set(rejected["reason"]) <= set(plan_positives.REJECT_REASONS),
            "pocket_quality must be a declared reject reason",
        )

    def test_a_pair_losing_every_pocket_is_a_rejected_pair_not_a_vanished_one(self):
        cavities = {("P30518", "F1"): [1, 2], ("P68363", "F1"): [1]}
        result = plan_positives.filter_cavities_by_quality(
            cavities=cavities,
            quality_frame=make_quality(
                [
                    quality_row("P30518", 1, FAILING_POCKET),
                    quality_row("P30518", 2, FAILING_POCKET),
                    quality_row("P68363", 1, PASSING_POCKET),
                ]
            ),
            plddt_min=70.0,
            min_fraction=0.80,
        )
        planned, rejected = plan_positives.plan_positives(
            interactions=make_interactions(
                [
                    ("DB00035", 554, "P30518", "AVPR2"),
                    ("DB00157", 10376, "P68363", "TUBA1B"),
                ]
            ),
            index=make_index(),
            cavities=result.cavities,
            quality_failures=result.failures,
        )
        # the surviving pair keeps its pocket, the losing pair is on the ledger
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned.iloc[0]["accession"], "P68363")
        self.assertEqual(list(rejected["reason"]), ["pocket_quality"])
        self.assertEqual(rejected.iloc[0]["accession"], "P30518")
        self.assertIn("all 2 pocket(s) failed", rejected.iloc[0]["detail"])
        # and the totals still reconcile: 2 in == 1 planned pair + 1 rejected
        n_pairs = len(planned.drop_duplicates(subset=["drugbank_id", "accession"]))
        self.assertEqual(n_pairs + len(rejected), 2)

    def test_a_target_without_any_cavity_is_still_NO_CAVITY(self):
        """The filter must not relabel a structural absence as a quality drop."""
        result = plan_positives.filter_cavities_by_quality(
            cavities={("P68363", "F1"): [1]},
            quality_frame=make_quality([quality_row("P68363", 1, PASSING_POCKET)]),
            plddt_min=70.0,
            min_fraction=0.80,
        )
        _, rejected = plan_positives.plan_positives(
            interactions=make_interactions([("DB00035", 554, "P30518", "AVPR2")]),
            index=make_index(),
            cavities=result.cavities,
            quality_failures=result.failures,
        )
        self.assertEqual(rejected.iloc[0]["reason"], "NO_CAVITY")

    def test_the_filter_is_per_cavity_two_of_five_pockets_survive(self):
        result = plan_positives.filter_cavities_by_quality(
            cavities={("P30518", "F1"): [1, 2, 3, 4, 5]},
            quality_frame=make_quality(
                [
                    quality_row("P30518", 1, PASSING_POCKET),
                    quality_row("P30518", 2, FAILING_POCKET),
                    quality_row("P30518", 3, FAILING_POCKET),
                    quality_row("P30518", 4, PASSING_POCKET),
                    quality_row("P30518", 5, FAILING_POCKET),
                ]
            ),
            plddt_min=70.0,
            min_fraction=0.80,
        )
        self.assertEqual(result.cavities, {("P30518", "F1"): [1, 4]})
        self.assertEqual((result.n_considered, result.n_passed), (5, 2))
        planned, rejected = plan_positives.plan_positives(
            interactions=make_interactions([("DB00035", 554, "P30518", "AVPR2")]),
            index=make_index(),
            cavities=result.cavities,
            quality_failures=result.failures,
        )
        self.assertEqual(list(planned["cavity_index"]), [1, 4])
        self.assertTrue(rejected.empty)

    def test_bare_fragment_ids_in_the_quality_file_still_match(self):
        """'1' and 'F1' are one pocket on both sides of the join, never two."""
        result = plan_positives.filter_cavities_by_quality(
            cavities={("P30518", "F1"): [1]},
            quality_frame=make_quality(
                [quality_row("P30518", 1, PASSING_POCKET, fragment="1")]
            ),
            plddt_min=70.0,
            min_fraction=0.80,
        )
        self.assertEqual(result.cavities, {("P30518", "F1"): [1]})

    def test_an_unscored_pocket_raises_naming_the_count_and_the_flag(self):
        with self.assertRaises(InvariantViolation) as ctx:
            plan_positives.filter_cavities_by_quality(
                cavities={("P30518", "F1"): [1, 2]},
                quality_frame=make_quality([quality_row("P30518", 1, PASSING_POCKET)]),
                plddt_min=70.0,
                min_fraction=0.80,
            )
        message = str(ctx.exception)
        self.assertIn("1 cavity", message)
        self.assertIn("P30518/F1/cav2", message)
        self.assertIn("--allow-missing-quality", message)

    def test_allow_missing_treats_an_unscored_pocket_as_failing(self):
        result = plan_positives.filter_cavities_by_quality(
            cavities={("P30518", "F1"): [1, 2]},
            quality_frame=make_quality([quality_row("P30518", 1, PASSING_POCKET)]),
            plddt_min=70.0,
            min_fraction=0.80,
            allow_missing=True,
        )
        self.assertEqual(result.cavities, {("P30518", "F1"): [1]})
        self.assertEqual(result.n_missing, 1)
        self.assertIn("F1/cav2", result.failures["P30518"][0])


class TestPocketQualityCli(unittest.TestCase):
    """Same filter through the command line, where the policy flags live."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.index = os.path.join(self.root, "index.parquet")
        make_index().to_frame().to_parquet(self.index, index=False)
        self.interactions = os.path.join(self.root, "inter.csv")
        schemas.write_checked(
            make_interactions(
                [
                    ("DB00035", 554, "P30518", "AVPR2"),
                    ("DB00157", 10376, "P68363", "TUBA1B"),
                ]
            ),
            self.interactions,
            "interactions_resolved",
        )
        self.cavities = os.path.join(self.root, "cav.csv")
        pd.DataFrame(
            [
                {"UniProt_ID": "P30518", "Fragment_ID": "F1", "Cavity_Index": 1},
                {"UniProt_ID": "P30518", "Fragment_ID": "F1", "Cavity_Index": 2},
                {"UniProt_ID": "P68363", "Fragment_ID": "F1", "Cavity_Index": 1},
            ]
        ).to_csv(self.cavities, index=False)

    def write_quality(self, rows, name="cavity_quality.csv") -> str:
        path = os.path.join(self.root, name)
        schemas.write_checked(make_quality(rows), path, "cavity_quality")
        return path

    def full_quality(self) -> str:
        """Every cavity scored: P30518's two fail, P68363's one passes."""
        return self.write_quality(
            [
                quality_row("P30518", 1, FAILING_POCKET),
                quality_row("P30518", 2, FAILING_POCKET),
                quality_row("P68363", 1, PASSING_POCKET),
            ]
        )

    def run_cli(self, tag, *extra):
        out = os.path.join(self.root, "planned_%s.csv" % tag)
        rejects = os.path.join(self.root, "rejected_%s.csv" % tag)
        code = plan_positives.main(
            [
                "--interactions", self.interactions,
                "--index", self.index,
                "--cavities", self.cavities,
                "--out", out,
                "--rejects", rejects,
                "--run-dir", self.root,
                "--quiet",
            ]
            + list(extra)
        )
        return code, out, rejects

    def test_without_min_fraction_nothing_is_filtered(self):
        """--cavity-quality alone must leave the plan byte-for-byte unchanged."""
        base_code, base_out, base_rejects = self.run_cli("base")
        code, out, rejects = self.run_cli(
            "loaded", "--cavity-quality", self.full_quality()
        )
        self.assertEqual((base_code, code), (0, 0))
        pd.testing.assert_frame_equal(
            schemas.read_checked(base_out, "positives_planned"),
            schemas.read_checked(out, "positives_planned"),
        )
        with open(base_out, "rb") as handle:
            base_bytes = handle.read()
        with open(out, "rb") as handle:
            self.assertEqual(handle.read(), base_bytes)
        self.assertEqual(len(pd.read_csv(base_rejects)), len(pd.read_csv(rejects)))
        # ...and every cavity is still there: 2 + 1
        self.assertEqual(len(schemas.read_checked(out, "positives_planned")), 3)

    def test_filtering_drops_the_failing_pockets_and_records_the_pair(self):
        code, out, rejects = self.run_cli(
            "filtered",
            "--cavity-quality", self.full_quality(),
            "--plddt-min", "70",
            "--min-plddt-fraction", "0.80",
        )
        self.assertEqual(code, 0)
        planned = schemas.read_checked(out, "positives_planned")
        self.assertEqual(list(planned["accession"]), ["P68363"])
        rejected = pd.read_csv(rejects)
        self.assertEqual(list(rejected["reason"]), ["pocket_quality"])
        self.assertIn("frac_ge_70=0.42 < 0.80", rejected.iloc[0]["detail"])

    def test_every_dropped_pocket_lands_in_the_quality_reject_ledger(self):
        """A pair that survives on one pocket must still account for the other.

        ``positives_rejected.csv`` is one row per PAIR, so a pair that keeps some
        pockets and loses others has no row there at all. Without the per-pocket
        sidecar those losses exist only as a log line, and on the real campaign
        that is the majority of what the filter removes (6,806 of 9,202 dropped
        positive jobs at pLDDT>=70 / frac>=0.80).
        """
        quality_path = self.write_quality(
            [
                quality_row("P30518", 1, PASSING_POCKET),   # kept
                quality_row("P30518", 2, FAILING_POCKET),   # dropped, pair survives
                quality_row("P68363", 1, FAILING_POCKET),   # dropped, pair dies
            ],
            name="mixed_quality.csv",
        )
        code, out, rejects = self.run_cli(
            "mixed",
            "--cavity-quality", quality_path,
            "--plddt-min", "70",
            "--min-plddt-fraction", "0.80",
        )
        self.assertEqual(code, 0)
        planned = schemas.read_checked(out, "positives_planned")
        # P30518 keeps cavity 1 only; P68363 leaves the plan entirely.
        self.assertEqual(
            sorted(zip(planned["accession"], planned["cavity_index"])),
            [("P30518", 1)],
        )
        # The pair ledger names only the pair that lost everything.
        rejected = pd.read_csv(rejects)
        self.assertEqual(list(rejected["reason"]), ["pocket_quality"])

        sidecar = os.path.splitext(rejects)[0] + ".quality_rejects.csv"
        self.assertTrue(os.path.exists(sidecar), "the per-pocket ledger must exist")
        ledger = pd.read_csv(sidecar)
        self.assertEqual(
            list(ledger.columns), list(plan_positives.QUALITY_REJECT_COLUMNS)
        )
        # BOTH dropped pockets are here, including the one whose pair survived.
        self.assertEqual(
            sorted(zip(ledger["accession"], ledger["cavity_index"])),
            [("P30518", 2), ("P68363", 1)],
        )
        self.assertEqual(set(ledger["reason"]), {"pocket_quality"})
        for detail in ledger["detail"]:
            self.assertIn("frac_ge_70=0.42 < 0.80 (n_residues=50)", detail)

    def test_the_pocket_ledger_is_absent_unless_the_filter_ran(self):
        """Absent means 'no filtering asked for'; empty means 'dropped nothing'."""
        _, _, rejects = self.run_cli("noflag")
        self.assertFalse(
            os.path.exists(os.path.splitext(rejects)[0] + ".quality_rejects.csv")
        )
        _, _, loaded = self.run_cli("loadedonly", "--cavity-quality", self.full_quality())
        self.assertFalse(
            os.path.exists(os.path.splitext(loaded)[0] + ".quality_rejects.csv")
        )
        # Filter on, but a threshold nothing fails: the file exists and is empty.
        passing = self.write_quality(
            [
                quality_row("P30518", 1, PASSING_POCKET),
                quality_row("P30518", 2, PASSING_POCKET),
                quality_row("P68363", 1, PASSING_POCKET),
            ],
            name="all_pass.csv",
        )
        _, _, kept = self.run_cli(
            "allpass",
            "--cavity-quality", passing,
            "--plddt-min", "70",
            "--min-plddt-fraction", "0.80",
        )
        sidecar = os.path.splitext(kept)[0] + ".quality_rejects.csv"
        self.assertTrue(os.path.exists(sidecar))
        self.assertEqual(len(pd.read_csv(sidecar)), 0)

    def test_unscored_pockets_are_named_in_the_ledger_under_allow_missing(self):
        quality_path = self.write_quality(
            [quality_row("P30518", 1, PASSING_POCKET)], name="sparse.csv"
        )
        code, out, rejects = self.run_cli(
            "allow",
            "--cavity-quality", quality_path,
            "--plddt-min", "70",
            "--min-plddt-fraction", "0.80",
            "--allow-missing-quality",
        )
        self.assertEqual(code, 0)
        ledger = pd.read_csv(os.path.splitext(rejects)[0] + ".quality_rejects.csv")
        self.assertEqual(
            sorted(zip(ledger["accession"], ledger["cavity_index"])),
            [("P30518", 2), ("P68363", 1)],
        )
        for detail in ledger["detail"]:
            self.assertIn("absent from the cavity-quality file", detail)

    def test_a_manifest_pocket_missing_from_the_quality_file_exits_2(self):
        quality_path = self.write_quality(
            [
                quality_row("P30518", 1, PASSING_POCKET),
                quality_row("P68363", 1, PASSING_POCKET),
            ],
            name="partial_quality.csv",
        )
        code, out, _ = self.run_cli(
            "missing",
            "--cavity-quality", quality_path,
            "--min-plddt-fraction", "0.80",
        )
        self.assertEqual(code, 2)
        self.assertFalse(
            os.path.exists(out), "a refused run must create nothing on the way out"
        )

    def test_allow_missing_quality_downgrades_it_and_fails_the_pocket(self):
        quality_path = self.write_quality(
            [
                quality_row("P30518", 1, PASSING_POCKET),
                quality_row("P68363", 1, PASSING_POCKET),
            ],
            name="partial_quality2.csv",
        )
        code, out, rejects = self.run_cli(
            "allowed",
            "--cavity-quality", quality_path,
            "--min-plddt-fraction", "0.80",
            "--allow-missing-quality",
        )
        self.assertEqual(code, 0)
        planned = schemas.read_checked(out, "positives_planned")
        # P30518 keeps cavity 1 (scored, passing) and loses the unscored cavity 2
        self.assertEqual(
            sorted(zip(planned["accession"], planned["cavity_index"])),
            [("P30518", 1), ("P68363", 1)],
        )
        self.assertTrue(pd.read_csv(rejects).empty)

    def test_min_fraction_without_a_quality_file_exits_2(self):
        code, out, _ = self.run_cli("nofile", "--min-plddt-fraction", "0.80")
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
