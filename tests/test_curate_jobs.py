"""Tests for ``curate_jobs.py`` -- REBUILD_PLAN Phase 5.

Every test is hermetic: fixtures live in ``tempfile.TemporaryDirectory()``, no
network, no external drive, no ``/opt/data``.  The fixture job directories
reproduce the real layout of ``$OLD/consensus_docking_results/<job>`` (verified
against ``DB00035_AVPR2_Q8NCT1_cavity_2``): ``final_results.csv``, four tool
directories each with ``results.csv`` and ~20 ``complex_*.pdb``, ``gold/output``,
``input/*.mol2`` and a dozen retry logs.

The failures these tests exist to make impossible are the ones the 2025 campaign
actually suffered:

* the directory name claimed a gene that belonged to a different protein
  (``DB00035_AVPR2_Q8NCT1_...`` -- Q8NCT1 is ARRDC4);
* a many-to-one map silently kept its last entry;
* output depended on hash-seed / iteration order;
* an input that could not be resolved was dropped without a record;
* a missing prerequisite was quietly replaced by a wider default.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd  # noqa: E402

import curate_jobs  # noqa: E402
from curate_jobs import (  # noqa: E402
    EXCLUDED_BASENAME_PATTERNS,
    CurationError,
    build_plan,
    execute_plan,
    is_excluded_basename,
    job_provenance,
    select_all_files,
    select_evidence_files,
)
from dock_selectivity.identity import IdentityIndex, Protein  # noqa: E402
from dock_selectivity.jobid import (  # noqa: E402
    JOB_IDENTITY_FILENAME,
    read_job_identity_record,
)
from dock_selectivity.schemas import read_checked, write_checked  # noqa: E402

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

#: Real accessions and their real symbols.  P30518/Q8NCT1, P68363/Q13509 and
#: P60568/P01589 are the three pairs the corrupt map confused.
FIXTURE_PROTEINS = (
    ("P30518", "V2R_HUMAN", "AVPR2", ("AVPR2",), 554),
    ("Q8NCT1", "ARRD4_HUMAN", "ARRDC4", ("ARRDC4",), 91947),
    ("P68363", "TBA1B_HUMAN", "TUBA1B", ("TUBA1B", "TUBA1B1"), 10376),
    ("Q13509", "TBB3_HUMAN", "TUBB3", ("TUBB3", "TUBB4"), 10381),
    ("P60568", "IL2_HUMAN", "IL2", ("IL2",), 3558),
    ("P01589", "IL2RA_HUMAN", "IL2RA", ("IL2RA",), 3559),
    ("P03886", "NU1M_HUMAN", "MT-ND1", ("MT-ND1", "ND1"), 4535),
)

FINAL_RESULTS = "Tool1,Tool2,Score1,Score2,RMSD\nSMINA,GNINA,-7.1,-7.4,0.82\n"

#: Exactly what ``--evidence-only`` must copy out of a complete job.
EXPECTED_EVIDENCE = sorted(
    [
        "final_results.csv",
        "gnina/out.sdf",
        "gnina/results.csv",
        "gold/output/DB00035_m1.rnk",
        "gold/output/bestranking.lst",
        "gold/output/gold_soln_DB00035_m1_1.sdf",
        "gold/output/gold_soln_DB00035_m1_2.sdf",
        "gold/results.csv",
        "input/DB00035.mol2",
        "input/receptor.mol2",
        "ledock/out.dok",
        "ledock/results.csv",
        "smina/out.sdf",
        "smina/results.csv",
    ]
)


def write_index(path):
    """Write a valid ``identity_index.parquet`` and return the loaded index."""
    records = [
        Protein(acc, entry, primary, symbols, entrez, True)
        for acc, entry, primary, symbols, entrez in FIXTURE_PROTEINS
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    IdentityIndex(records).to_frame().to_parquet(path, index=False)
    return IdentityIndex.load(path)


def make_job_dir(root, name, with_complex=True, with_logs=True, with_final=True):
    """Build one fixture job directory in the real ``$OLD`` layout."""
    files = {}
    if with_final:
        files["final_results.csv"] = FINAL_RESULTS
    for tool in ("smina", "gnina", "gold", "ledock"):
        files["{0}/results.csv".format(tool)] = "pose,score\n1,-7.1\n2,-6.8\n"
    files["smina/out.sdf"] = "smina 20 poses\n" * 3
    files["smina/out.log"] = "smina stdout\n"
    files["smina/out_1.sdf"] = "single pose\n"
    files["gnina/out.sdf"] = "gnina 20 poses with CNN_VS\n" * 3
    files["gnina/out_1.sdf"] = "single pose\n"
    files["ledock/out.dok"] = "ledock 20 poses\n" * 3
    files["ledock/out_1.dok"] = "single pose\n"
    files["ledock/pro.pdb"] = "receptor copy, derived\n" * 50
    files["ledock/dock.in"] = "config\n"
    files["gold/gold.in"] = "config\n"
    files["gold/output/gold_soln_DB00035_m1_1.sdf"] = "gold pose 1\n"
    files["gold/output/gold_soln_DB00035_m1_2.sdf"] = "gold pose 2\n"
    files["gold/output/bestranking.lst"] = "rank fitness\n"
    files["gold/output/DB00035_m1.rnk"] = "solution ranks\n"
    # the bulk of a real gold/output: a third receptor copy and GOLD's own logs
    files["gold/output/gold_protein.mol2"] = "@<TRIPOS>MOLECULE\nreceptor again\n" * 40
    files["gold/output/plp_protein.mol2"] = "@<TRIPOS>MOLECULE\nreceptor again\n" * 40
    files["gold/output/gold_DB00035_m1.log"] = "GOLD log\n" * 30
    files["gold/output/ranked_DB00035_m1_1.sdf"] = "gold pose 1\n"
    files["input/DB00035.mol2"] = "@<TRIPOS>MOLECULE\nligand\n"
    files["input/receptor.mol2"] = "@<TRIPOS>MOLECULE\nreceptor\n" * 20
    if with_complex:
        for tool in ("smina", "gnina", "gold", "ledock"):
            for pose in range(1, 4):
                files["{0}/complex_{1}.pdb".format(tool, pose)] = "ATOM\n" * 200
        # a complex_*.pdb *inside* a tree rule: the deny-list must still win
        files["gold/output/complex_99.pdb"] = "ATOM\n" * 200
    if with_logs:
        files["log_20250704_071438_c57a.log"] = "retry 1\n"
        files["log_20251222_000201_1609.log"] = "retry 12\n"
    job = os.path.join(root, name)
    for relpath, content in files.items():
        full = os.path.join(job, relpath)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as handle:
            handle.write(content)
    return job


def salvage_row(old_dir, classification, drug, accession, gene, fragment, cavity, **extra):
    row = {
        "old_job_dir": old_dir,
        "classification": classification,
        "drugbank_id": drug,
        "accession": accession,
        "gene": gene,
        "fragment_id": fragment,
        "cavity_index": cavity,
        "old_gene_label": gene,
        "label_was_correct": True,
        "new_job_dir": None,
        "excluded_reason": "not selected",
    }
    row.update(extra)
    return row


def positive_row(drug, accession, gene, fragment, cavity, why="entrez route"):
    return {
        "drugbank_id": drug,
        "accession": accession,
        "gene": gene,
        "fragment_id": fragment,
        "cavity_index": cavity,
        "sample_type": "positive",
        "selected_by": why,
    }


def drawn_row(drug, accession, gene, fragment, cavity, source, old_dir=None, why="balanced draw"):
    return {
        "drugbank_id": drug,
        "accession": accession,
        "gene": gene,
        "fragment_id": fragment,
        "cavity_index": cavity,
        "sample_type": "negative",
        "source": source,
        "old_job_dir": old_dir,
        "selected_by": why,
    }


def snapshot_tree(root):
    """(relpath -> (size, mtime_ns, mode)) for every file under *root*."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            full = os.path.join(dirpath, filename)
            stat = os.stat(full)
            out[os.path.relpath(full, root)] = (
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_mode,
            )
    return out


def digest_tree(root):
    """SHA256 over every relative path and every byte under *root*."""
    digest = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            full = os.path.join(dirpath, filename)
            digest.update(os.path.relpath(full, root).encode("utf-8"))
            digest.update(b"\0")
            with open(full, "rb") as handle:
                digest.update(handle.read())
            digest.update(b"\0")
    return digest.hexdigest()


class CurationFixture(unittest.TestCase):
    """A miniature ``$OLD`` + manifests that mirror the real campaign's shapes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.addCleanup(self.tmp.cleanup)

        self.src = os.path.join(self.root, "old", "consensus_docking_results")
        self.dst = os.path.join(self.root, "new", "docking_results")
        os.makedirs(self.src)

        self.index_path = os.path.join(self.root, "new", "identity", "identity_index.parquet")
        self.index = write_index(self.index_path)

        # Two salvaged positives and two salvaged negatives.  The first one is
        # the incident in miniature: the folder says AVPR2 but the receptor is
        # Q8NCT1 (ARRDC4).  The label is a lie; the accession is physical truth.
        self.jobs = {
            "DB00035_AVPR2_Q8NCT1_cavity_2": ("Q8NCT1", 2, "true_positive"),
            "DB00035_nan_P30518_cavity_1": ("P30518", 1, "true_positive"),
            "DB00157_MT-ND1_P03886_cavity_1": ("P03886", 1, "valid_non_interaction"),
            "DB00157_nan_P68363_cavity_3": ("P68363", 3, "valid_non_interaction"),
            # in the pool but never drawn -> surplus, must not be copied
            "DB00157_nan_Q13509_cavity_1": ("Q13509", 1, "valid_non_interaction"),
        }
        for name in self.jobs:
            make_job_dir(self.src, name)

        self.manifests = os.path.join(self.root, "new", "manifests")
        os.makedirs(self.manifests)
        self.write_manifests()

    # -- manifest construction (overridable per test) ---------------------

    def salvage_rows(self):
        return [
            salvage_row(
                "DB00035_AVPR2_Q8NCT1_cavity_2",
                "true_positive",
                "DB00035",
                "Q8NCT1",
                "AVPR2",  # the corrupt label, as it appears in the old campaign
                "F1",
                2,
            ),
            salvage_row(
                "DB00035_nan_P30518_cavity_1",
                "true_positive",
                "DB00035",
                "P30518",
                "AVPR2",
                "F1",
                1,
            ),
            salvage_row(
                "DB00157_MT-ND1_P03886_cavity_1",
                "valid_non_interaction",
                "DB00157",
                "P03886",
                "MT-ND1",
                "F1",
                1,
            ),
            salvage_row(
                "DB00157_nan_P68363_cavity_3",
                "valid_non_interaction",
                "DB00157",
                "P68363",
                "TUBA1B",
                "F2",
                3,
            ),
            salvage_row(
                "DB00157_nan_Q13509_cavity_1",
                "valid_non_interaction",
                "DB00157",
                "Q13509",
                "TUBB3",
                "F1",
                1,
            ),
        ]

    def positive_rows(self):
        return [
            positive_row("DB00035", "Q8NCT1", "ARRDC4", "F1", 2),
            positive_row("DB00035", "P30518", "AVPR2", "F1", 1),
            # planned but never docked -> must be reported, not silently absent
            positive_row("DB00035", "P60568", "IL2", "F1", 4),
        ]

    def drawn_rows(self):
        return [
            drawn_row("DB00157", "P03886", "MT-ND1", "F1", 1, "pool",
                      "DB00157_MT-ND1_P03886_cavity_1"),
            drawn_row("DB00157", "P68363", "TUBA1B", "F2", 3, "pool",
                      "DB00157_nan_P68363_cavity_3"),
            drawn_row("DB00157", "P01589", "IL2RA", "F1", 2, "new", None),
        ]

    def write_manifests(self):
        self.salvage_path = os.path.join(self.manifests, "salvage_map.csv")
        self.positives_path = os.path.join(self.manifests, "positives_planned.csv")
        self.drawn_path = os.path.join(self.manifests, "negatives_drawn.csv")
        write_checked(pd.DataFrame(self.salvage_rows()), self.salvage_path, "salvage_map")
        write_checked(
            pd.DataFrame(self.positive_rows()), self.positives_path, "positives_planned"
        )
        write_checked(pd.DataFrame(self.drawn_rows()), self.drawn_path, "negatives_drawn")

    # -- helpers -----------------------------------------------------------

    def make_plan(self, **kwargs):
        return build_plan(
            salvage=read_checked(self.salvage_path, "salvage_map"),
            positives=read_checked(self.positives_path, "positives_planned"),
            drawn=read_checked(self.drawn_path, "negatives_drawn"),
            index=self.index,
            src=self.src,
            **kwargs
        )

    def base_provenance(self):
        return {
            "retention": curate_jobs.RETENTION_EVIDENCE,
            "identity_index_sha256": self.index.source_sha256,
            "code_revision": "test-revision",
            "curated_by": "curate_jobs.py",
        }

    def cli(self, *extra):
        return curate_jobs.main(
            [
                "--salvage-map", self.salvage_path,
                "--drawn", self.drawn_path,
                "--positives", self.positives_path,
                "--index", self.index_path,
                "--src", self.src,
                "--dst", self.dst,
                "--code-revision", "test-revision",
                "--quiet",
            ]
            + list(extra)
        )


# --------------------------------------------------------------------------
# the 2.4% rule
# --------------------------------------------------------------------------


class TestEvidenceSelection(CurationFixture):
    def job(self):
        return os.path.join(self.src, "DB00035_AVPR2_Q8NCT1_cavity_2")

    def test_evidence_only_selects_exactly_the_intended_file_set(self):
        files, missing = select_evidence_files(self.job())
        self.assertEqual([p for p, _ in files], EXPECTED_EVIDENCE)
        self.assertEqual(missing, [])

    def test_no_complex_pdb_and_no_retry_log_is_ever_selected(self):
        files, _ = select_evidence_files(self.job())
        leaked = [p for p, _ in files if is_excluded_basename(p)]
        self.assertEqual(leaked, [], "the 45 MB of derived poses must not be carried")
        self.assertFalse(any(p.endswith(".log") for p, _ in files))

    def test_the_tree_rule_cannot_leak_an_excluded_file(self):
        # gold/output/complex_99.pdb exists on disk and is matched by the
        # 'gold/output' tree rule; the deny-list runs after rule expansion.
        self.assertTrue(
            os.path.exists(os.path.join(self.job(), "gold", "output", "complex_99.pdb"))
        )
        files, _ = select_evidence_files(self.job())
        self.assertNotIn("gold/output/complex_99.pdb", [p for p, _ in files])

    def test_selection_is_sorted_and_repeatable(self):
        first, _ = select_evidence_files(self.job())
        second, _ = select_evidence_files(self.job())
        self.assertEqual(first, second)
        self.assertEqual([p for p, _ in first], sorted(p for p, _ in first))

    def test_sizes_are_the_real_sizes(self):
        files, _ = select_evidence_files(self.job())
        for relpath, size in files:
            self.assertEqual(os.path.getsize(os.path.join(self.job(), relpath)), size)

    def test_gold_output_all_takes_the_whole_tree(self):
        """The literal ``gold/output/**`` rule -- 3.09 MB/job on real data."""
        files, _ = select_evidence_files(self.job(), gold_output="all")
        names = [p for p, _ in files]
        self.assertIn("gold/output/gold_protein.mol2", names)
        self.assertIn("gold/output/plp_protein.mol2", names)
        self.assertIn("gold/output/gold_DB00035_m1.log", names)
        self.assertIn("gold/output/ranked_DB00035_m1_1.sdf", names)
        self.assertNotIn("gold/output/complex_99.pdb", names)

    def test_gold_output_poses_is_the_default_and_drops_the_bulk(self):
        """...which is why the default keeps only the poses and the rankings."""
        default, _ = select_evidence_files(self.job())
        poses, _ = select_evidence_files(self.job(), gold_output="poses")
        self.assertEqual(default, poses)
        names = [p for p, _ in poses]
        self.assertIn("gold/output/gold_soln_DB00035_m1_1.sdf", names)
        self.assertIn("gold/output/bestranking.lst", names)
        for dropped in (
            "gold/output/gold_protein.mol2",
            "gold/output/plp_protein.mol2",
            "gold/output/gold_DB00035_m1.log",
            "gold/output/ranked_DB00035_m1_1.sdf",
        ):
            self.assertNotIn(dropped, names)
        self.assertLess(
            sum(b for _, b in poses),
            sum(b for _, b in select_evidence_files(self.job(), gold_output="all")[0]),
        )

    def test_an_unknown_gold_output_policy_is_refused(self):
        with self.assertRaises(ValueError):
            select_evidence_files(self.job(), gold_output="everything")

    def test_full_copy_includes_what_evidence_only_excludes(self):
        everything, _ = select_all_files(self.job())
        names = [p for p, _ in everything]
        self.assertIn("smina/complex_1.pdb", names)
        self.assertIn("log_20250704_071438_c57a.log", names)
        self.assertGreater(len(everything), len(EXPECTED_EVIDENCE))

    def test_a_job_without_final_results_reports_the_missing_rule(self):
        make_job_dir(self.src, "DB00035_nan_P60568_cavity_9", with_final=False)
        _, missing = select_evidence_files(
            os.path.join(self.src, "DB00035_nan_P60568_cavity_9")
        )
        self.assertIn("final_results", missing)

    def test_exclusion_patterns_are_the_documented_ones(self):
        self.assertEqual(EXCLUDED_BASENAME_PATTERNS, ("complex_*.pdb", "log_*.log"))
        self.assertTrue(is_excluded_basename("gold/complex_20.pdb"))
        self.assertTrue(is_excluded_basename("log_20250704_071438_c57a.log"))
        self.assertFalse(is_excluded_basename("smina/out.sdf"))
        self.assertFalse(is_excluded_basename("smina/out.log"))


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------


class TestPlan(CurationFixture):
    def test_selects_wanted_positives_and_drawn_pool_negatives_only(self):
        plan = self.make_plan()
        self.assertEqual(plan.stats["n_planned"], 4)
        self.assertEqual(plan.stats["n_planned_positive"], 2)
        self.assertEqual(plan.stats["n_planned_negative"], 2)
        carried = {job.old_job_dir for job in plan.jobs}
        self.assertNotIn(
            "DB00157_nan_Q13509_cavity_1",
            carried,
            "a pool job the sampler did not draw is surplus and is not copied",
        )

    def test_the_gene_in_the_new_name_comes_from_the_index_not_the_manifest(self):
        # The salvage row carries the corrupt label AVPR2 on accession Q8NCT1.
        # Q8NCT1 is ARRDC4.  The new name must say ARRDC4.
        plan = self.make_plan()
        job = next(j for j in plan.jobs if j.old_job_dir == "DB00035_AVPR2_Q8NCT1_cavity_2")
        self.assertEqual(job.new_job_dir, "DB00035__Q8NCT1__ARRDC4__F1__cav2")
        self.assertEqual(job.identity.gene, self.index.label("Q8NCT1"))
        self.assertNotIn("AVPR2", job.new_job_dir)

    def test_every_new_name_agrees_with_the_index(self):
        for job in self.make_plan().jobs:
            self.assertEqual(job.identity.gene, self.index.label(job.identity.accession))
            self.assertIn(job.identity.accession, job.new_job_dir)
            self.assertIn(job.identity.gene, job.new_job_dir)

    def test_hyphenated_gene_symbols_survive_the_new_scheme(self):
        job = next(
            j for j in self.make_plan().jobs if j.identity.accession == "P03886"
        )
        self.assertEqual(job.new_job_dir, "DB00157__P03886__MT-ND1__F1__cav1")

    def test_plan_order_is_total_and_repeatable(self):
        first = [(j.old_job_dir, j.new_job_dir) for j in self.make_plan().jobs]
        second = [(j.old_job_dir, j.new_job_dir) for j in self.make_plan().jobs]
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first, key=lambda p: (p[1], p[0])))

    def test_a_last_wins_collapse_is_impossible(self):
        plan = self.make_plan()
        collapse = [
            report
            for report in plan.invariants
            if report.get("check") == "collapse_factor_one"
        ]
        self.assertEqual(len(collapse), 2, "both directions must be asserted")
        for report in collapse:
            self.assertEqual(report["max_collapse"], 1)
            self.assertTrue(report["ok"])

    def test_every_candidate_is_either_planned_or_recorded_with_a_reason(self):
        plan = self.make_plan()
        accounting = next(
            r for r in plan.invariants if r.get("check") == "total_accounting"
        )
        self.assertTrue(accounting["ok"])
        self.assertEqual(
            accounting["n_in"], accounting["n_resolved"] + accounting["n_rejected"]
        )
        for reject in plan.rejects:
            self.assertTrue(reject.reason.strip(), "a rejection must carry a reason")

    def test_a_docked_positive_the_design_does_not_want_is_recorded_not_dropped(self):
        plan = self.make_plan()
        # every true_positive salvage row is in positives_planned here, so add one
        rows = self.salvage_rows() + [
            salvage_row(
                "DB00035_nan_P60568_cavity_9",
                "true_positive",
                "DB00035",
                "P60568",
                "IL2",
                "F1",
                9,
            )
        ]
        write_checked(pd.DataFrame(rows), self.salvage_path, "salvage_map")
        plan = self.make_plan()
        reasons = [r.reason for r in plan.rejects if r.old_job_dir == "DB00035_nan_P60568_cavity_9"]
        self.assertEqual(len(reasons), 1)
        self.assertIn("positives_planned", reasons[0])
        self.assertEqual(plan.stats["n_planned"], 4)

    def test_a_pool_draw_absent_from_the_salvage_map_is_an_error(self):
        rows = self.drawn_rows() + [
            drawn_row("DB00157", "P01589", "IL2RA", "F1", 7, "pool", "DB00157_nan_P01589_cavity_7")
        ]
        write_checked(pd.DataFrame(rows), self.drawn_path, "negatives_drawn")
        plan = self.make_plan()
        self.assertEqual(plan.n_errors, 1)
        self.assertIn(
            "absent from the salvage map",
            next(r.reason for r in plan.rejects if r.severity == "error"),
        )

    def test_drawing_a_true_positive_job_as_a_negative_is_an_error(self):
        rows = self.drawn_rows() + [
            drawn_row("DB00035", "P30518", "AVPR2", "F1", 1, "pool", "DB00035_nan_P30518_cavity_1")
        ]
        write_checked(pd.DataFrame(rows), self.drawn_path, "negatives_drawn")
        plan = self.make_plan()
        self.assertEqual(plan.n_errors, 1)
        self.assertIn("classifies it", plan.rejects[-1].reason)

    def test_a_missing_source_directory_is_recorded_not_invented(self):
        shutil.rmtree(os.path.join(self.src, "DB00157_MT-ND1_P03886_cavity_1"))
        plan = self.make_plan()
        self.assertEqual(plan.stats["n_planned"], 3)
        self.assertEqual(
            [r.reason for r in plan.rejects if r.severity == "error"],
            ["source job directory not found"],
        )

    def test_an_accession_absent_from_the_index_is_recorded_not_guessed(self):
        rows = self.salvage_rows()
        rows[0]["accession"] = "P99999"
        write_checked(pd.DataFrame(rows), self.salvage_path, "salvage_map")
        positives = self.positive_rows()
        positives[0]["accession"] = "P99999"
        write_checked(pd.DataFrame(positives), self.positives_path, "positives_planned")
        plan = self.make_plan()
        errors = [r for r in plan.rejects if r.severity == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("not in the identity index", errors[0].reason)

    def test_a_null_fragment_is_never_inferred_silently(self):
        rows = self.salvage_rows()
        rows[0]["fragment_id"] = None
        write_checked(pd.DataFrame(rows), self.salvage_path, "salvage_map")
        plan = self.make_plan()
        errors = [r for r in plan.rejects if r.severity == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("Refusing to infer", errors[0].reason)
        self.assertIn("--assume-fragment", errors[0].reason)

    def test_assume_fragment_makes_the_inference_explicit_and_recorded(self):
        rows = self.salvage_rows()
        rows[0]["fragment_id"] = None
        write_checked(pd.DataFrame(rows), self.salvage_path, "salvage_map")
        plan = self.make_plan(assume_fragment="F1")
        job = next(j for j in plan.jobs if j.old_job_dir == "DB00035_AVPR2_Q8NCT1_cavity_2")
        self.assertEqual(job.identity.fragment_id, "F1")
        self.assertIn("assumed", job.fragment_source)
        self.assertEqual(plan.stats["n_fragment_assumed"], 1)

    def test_two_old_dirs_claiming_one_new_name_fail_loudly(self):
        rows = self.salvage_rows() + [
            salvage_row(
                "DB00035_ARRDC4_Q8NCT1_cavity_2",  # same pocket, different label
                "true_positive",
                "DB00035",
                "Q8NCT1",
                "ARRDC4",
                "F1",
                2,
            )
        ]
        write_checked(pd.DataFrame(rows), self.salvage_path, "salvage_map")
        make_job_dir(self.src, "DB00035_ARRDC4_Q8NCT1_cavity_2")
        with self.assertRaises(CurationError) as caught:
            self.make_plan()
        self.assertIn("claimed by more than one", str(caught.exception))
        self.assertIn("last-wins", str(caught.exception))

    def test_a_collision_may_be_skipped_but_never_arbitrarily_resolved(self):
        rows = self.salvage_rows() + [
            salvage_row(
                "DB00035_ARRDC4_Q8NCT1_cavity_2",
                "true_positive",
                "DB00035",
                "Q8NCT1",
                "ARRDC4",
                "F1",
                2,
            )
        ]
        write_checked(pd.DataFrame(rows), self.salvage_path, "salvage_map")
        make_job_dir(self.src, "DB00035_ARRDC4_Q8NCT1_cavity_2")
        plan = self.make_plan(on_name_collision="skip")
        carried = {j.old_job_dir for j in plan.jobs}
        self.assertNotIn("DB00035_AVPR2_Q8NCT1_cavity_2", carried)
        self.assertNotIn("DB00035_ARRDC4_Q8NCT1_cavity_2", carried)
        self.assertEqual(
            sum(1 for r in plan.rejects if "claimed by" in r.reason),
            2,
            "both sides of a collision must be recorded",
        )

    def test_a_duplicated_salvage_key_is_refused_before_a_dict_is_built(self):
        rows = self.salvage_rows() + [self.salvage_rows()[0]]
        frame = pd.DataFrame(rows)
        # write past the schema's own primary-key check to simulate a producer
        # that emitted a duplicate; build_plan must still refuse it.
        frame.to_csv(self.salvage_path, index=False, lineterminator="\n")
        with self.assertRaises(Exception) as caught:
            self.make_plan()
        self.assertIn("old_job_dir", str(caught.exception))

    def test_planned_positives_with_no_docked_job_are_counted(self):
        plan = self.make_plan()
        self.assertEqual(plan.stats["n_positives_planned"], 3)
        self.assertEqual(plan.stats["n_positives_planned_not_salvageable"], 1)


# --------------------------------------------------------------------------
# copying
# --------------------------------------------------------------------------


class TestCopy(CurationFixture):
    def run_copy(self, evidence_only=True, workers=1, force=False, gold_output="poses"):
        plan = self.make_plan()
        outcomes = execute_plan(
            plan,
            src=self.src,
            dst=self.dst,
            evidence_only=evidence_only,
            base_provenance=self.base_provenance(),
            workers=workers,
            force=force,
            gold_output=gold_output,
        )
        return plan, outcomes

    def test_evidence_only_copies_exactly_the_intended_files(self):
        plan, outcomes = self.run_copy()
        self.assertEqual([o.status for o in outcomes], ["copied"] * 4)
        for job in plan.jobs:
            target = os.path.join(self.dst, job.new_job_dir)
            copied = sorted(
                os.path.relpath(os.path.join(dirpath, f), target).replace(os.sep, "/")
                for dirpath, _, filenames in os.walk(target)
                for f in filenames
            )
            self.assertEqual(copied, sorted(EXPECTED_EVIDENCE + [JOB_IDENTITY_FILENAME]))

    def test_no_complex_pdb_reaches_the_destination(self):
        self.run_copy()
        leaked = [
            os.path.join(dirpath, f)
            for dirpath, _, filenames in os.walk(self.dst)
            for f in filenames
            if is_excluded_basename(f)
        ]
        self.assertEqual(leaked, [])

    def test_full_copy_mode_carries_the_complex_pdbs(self):
        self.run_copy(evidence_only=False)
        found = [
            f
            for _, _, filenames in os.walk(self.dst)
            for f in filenames
            if f.startswith("complex_")
        ]
        self.assertTrue(found)

    def test_every_copied_file_matches_its_source_byte_for_byte(self):
        plan, _ = self.run_copy()
        for job in plan.jobs:
            source = os.path.join(self.src, job.old_job_dir)
            target = os.path.join(self.dst, job.new_job_dir)
            for relpath in EXPECTED_EVIDENCE:
                with open(os.path.join(source, relpath), "rb") as handle:
                    original = handle.read()
                with open(os.path.join(target, relpath), "rb") as handle:
                    copied = handle.read()
                self.assertEqual(original, copied)
                self.assertEqual(
                    os.path.getsize(os.path.join(source, relpath)), len(copied)
                )

    def test_job_identity_json_is_the_authority_and_records_its_source(self):
        plan, _ = self.run_copy()
        job = next(j for j in plan.jobs if j.old_job_dir == "DB00035_AVPR2_Q8NCT1_cavity_2")
        record = read_job_identity_record(os.path.join(self.dst, job.new_job_dir))
        self.assertEqual(record["accession"], "Q8NCT1")
        self.assertEqual(record["gene"], "ARRDC4")
        self.assertEqual(record["fragment_id"], "F1")
        self.assertEqual(record["cavity_index"], 2)
        self.assertEqual(record["sample_type"], "positive")
        self.assertEqual(record["provenance"], "salvaged_from")
        self.assertEqual(record["source_job"], "DB00035_AVPR2_Q8NCT1_cavity_2")
        self.assertEqual(record["retention"], "evidence_only")
        self.assertEqual(record["identity_index_sha256"], self.index.source_sha256)
        self.assertEqual(record["dir_name"], job.new_job_dir)
        self.assertTrue(record["selected_by"])
        self.assertEqual(
            sorted(e["path"] for e in record["evidence"]), EXPECTED_EVIDENCE
        )

    def test_source_tree_is_byte_for_byte_unmodified_by_a_real_run(self):
        before = snapshot_tree(self.src)
        self.run_copy()
        after = snapshot_tree(self.src)
        self.assertEqual(before, after, "$OLD must remain an untouched, independent copy")

    def test_source_tree_is_unmodified_even_when_the_run_fails(self):
        os.unlink(
            os.path.join(self.src, "DB00035_AVPR2_Q8NCT1_cavity_2", "final_results.csv")
        )
        before = snapshot_tree(self.src)
        _, outcomes = self.run_copy()
        self.assertIn("failed", [o.status for o in outcomes])
        self.assertEqual(snapshot_tree(self.src), before)

    def test_nothing_is_hardlinked(self):
        plan, _ = self.run_copy()
        job = plan.jobs[0]
        source = os.path.join(self.src, job.old_job_dir, "final_results.csv")
        target = os.path.join(self.dst, job.new_job_dir, "final_results.csv")
        self.assertNotEqual(os.stat(source).st_ino, os.stat(target).st_ino)
        self.assertEqual(os.stat(source).st_nlink, 1)

    def test_rerunning_is_a_no_op(self):
        self.run_copy()
        before = snapshot_tree(self.dst)
        _, outcomes = self.run_copy()
        self.assertEqual([o.status for o in outcomes], ["skipped_complete"] * 4)
        self.assertEqual(snapshot_tree(self.dst), before)

    def test_an_interrupted_job_is_redone_not_reported_complete(self):
        plan, _ = self.run_copy()
        target = os.path.join(self.dst, plan.jobs[0].new_job_dir)
        os.unlink(os.path.join(target, JOB_IDENTITY_FILENAME))
        _, outcomes = self.run_copy()
        statuses = sorted(o.status for o in outcomes)
        self.assertEqual(statuses, ["copied", "skipped_complete", "skipped_complete", "skipped_complete"])
        self.assertTrue(os.path.exists(os.path.join(target, JOB_IDENTITY_FILENAME)))

    def test_a_truncated_evidence_file_is_detected_on_resume(self):
        plan, _ = self.run_copy()
        target = os.path.join(self.dst, plan.jobs[0].new_job_dir)
        with open(os.path.join(target, "smina", "out.sdf"), "w", encoding="utf-8") as handle:
            handle.write("truncated")
        _, outcomes = self.run_copy()
        self.assertIn("copied", [o.status for o in outcomes])
        with open(os.path.join(target, "smina", "out.sdf"), "rb") as handle:
            restored = handle.read()
        with open(
            os.path.join(self.src, plan.jobs[0].old_job_dir, "smina", "out.sdf"), "rb"
        ) as handle:
            self.assertEqual(restored, handle.read())

    def test_a_job_missing_final_results_fails_rather_than_being_half_copied(self):
        job_dir = os.path.join(self.src, "DB00035_AVPR2_Q8NCT1_cavity_2")
        os.unlink(os.path.join(job_dir, "final_results.csv"))
        _, outcomes = self.run_copy()
        failed = [o for o in outcomes if o.status == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertIn("final_results.csv", failed[0].reason)
        self.assertFalse(
            os.path.exists(os.path.join(self.dst, "DB00035__Q8NCT1__ARRDC4__F1__cav2"))
        )

    def test_a_size_mismatch_removes_the_destination_and_never_the_source(self):
        source = os.path.join(self.src, "DB00035_AVPR2_Q8NCT1_cavity_2", "final_results.csv")
        target = os.path.join(self.dst, "x", "final_results.csv")
        with self.assertRaises(CurationError) as caught:
            curate_jobs._copy_one_file(source, target, expected_size=999999)
        self.assertIn("size mismatch", str(caught.exception))
        self.assertFalse(os.path.exists(target))
        self.assertTrue(os.path.exists(source))

    def test_parallel_workers_produce_the_same_tree(self):
        self.run_copy(workers=1)
        serial = digest_tree(self.dst)
        shutil.rmtree(self.dst)
        self.run_copy(workers=4)
        self.assertEqual(digest_tree(self.dst), serial)

    def test_dst_inside_src_is_refused(self):
        with self.assertRaises(CurationError) as caught:
            execute_plan(
                self.make_plan(),
                src=self.src,
                dst=os.path.join(self.src, "curated"),
                evidence_only=True,
                base_provenance=self.base_provenance(),
            )
        self.assertIn("read-only", str(caught.exception))

    def test_changing_the_gold_policy_invalidates_a_previous_copy(self):
        self.run_copy(gold_output="poses")
        _, outcomes = self.run_copy(gold_output="all")
        self.assertEqual(
            [o.status for o in outcomes],
            ["copied"] * 4,
            "a different retention policy is a different artifact",
        )
        record = read_job_identity_record(
            os.path.join(self.dst, sorted(os.listdir(self.dst))[0])
        )
        self.assertEqual(record["evidence_gold_output"], "all")

    def test_provenance_carries_no_timestamp(self):
        plan = self.make_plan()
        record = job_provenance(plan.jobs[0], self.base_provenance())
        joined = json.dumps(record)
        self.assertNotIn("timestamp", joined)
        self.assertNotIn("20", record.get("code_revision", ""))


# --------------------------------------------------------------------------
# CLI behaviour
# --------------------------------------------------------------------------


class TestCli(CurationFixture):
    def test_dry_run_writes_absolutely_nothing(self):
        before = snapshot_tree(self.root)
        code = self.cli("--evidence-only", "--dry-run")
        self.assertEqual(code, 0)
        self.assertFalse(os.path.exists(self.dst))
        self.assertEqual(snapshot_tree(self.root), before)

    def test_dry_run_refuses_to_be_combined_with_an_output_flag(self):
        with self.assertRaises(SystemExit):
            self.cli("--dry-run", "--report", os.path.join(self.root, "r.json"))

    def test_real_run_materialises_the_tree_and_the_manifest(self):
        report_path = os.path.join(self.root, "report.json")
        manifest_path = os.path.join(self.manifests, "run_manifest.csv")
        rejects_path = os.path.join(self.root, "rejects.csv")
        code = self.cli(
            "--evidence-only",
            "--report", report_path,
            "--manifest-out", manifest_path,
            "--rejects", rejects_path,
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(os.listdir(self.dst)), 4)

        manifest = read_checked(manifest_path, "run_manifest")
        self.assertEqual(len(manifest), 4)
        self.assertEqual(set(manifest["provenance"]), {"salvaged_from"})
        self.assertEqual(set(manifest["job_id"]), set(os.listdir(self.dst)))
        for row in manifest.to_dict("records"):
            self.assertEqual(row["gene"], self.index.label(row["accession"]))

        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)[0]
        self.assertTrue(report["ok"])
        self.assertEqual(report["outcomes"]["copied"], 4)
        self.assertEqual(report["retention"], "evidence_only")
        self.assertTrue(os.path.exists(rejects_path))
        rejects = pd.read_csv(rejects_path)
        self.assertEqual(list(rejects.columns), ["old_job_dir", "stage", "severity", "reason"])

    def test_a_run_with_unresolvable_candidates_exits_non_zero(self):
        rows = self.drawn_rows() + [
            drawn_row("DB00157", "P01589", "IL2RA", "F1", 7, "pool", "not_a_real_dir")
        ]
        write_checked(pd.DataFrame(rows), self.drawn_path, "negatives_drawn")
        self.assertEqual(self.cli("--evidence-only"), 1)
        self.assertEqual(self.cli("--evidence-only", "--allow-errors"), 0)

    def test_a_missing_identity_index_is_an_error_never_a_fallback(self):
        os.unlink(self.index_path)
        self.assertEqual(self.cli("--evidence-only", "--dry-run"), 1)

    def test_a_wrong_index_pin_refuses_to_run(self):
        code = curate_jobs.main(
            [
                "--salvage-map", self.salvage_path,
                "--drawn", self.drawn_path,
                "--positives", self.positives_path,
                "--index", self.index_path,
                "--index-sha256", "0" * 64,
                "--src", self.src,
                "--dst", self.dst,
                "--dry-run",
                "--quiet",
            ]
        )
        self.assertEqual(code, 1)
        self.assertFalse(os.path.exists(self.dst))

    def test_the_curated_tree_is_identical_across_hash_seeds(self):
        """G3: no ``set`` iteration, no dict order, no timestamp in the output."""
        digests = []
        for seed in ("0", "1", "12345"):
            dst = os.path.join(self.root, "dst_seed_{0}".format(seed))
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = seed
            snippet = (
                "import sys; sys.path.insert(0, {repo!r}); import curate_jobs; "
                "sys.exit(curate_jobs.main([{args}]))".format(
                    repo=REPO_ROOT,
                    args=", ".join(
                        repr(a)
                        for a in [
                            "--salvage-map", self.salvage_path,
                            "--drawn", self.drawn_path,
                            "--positives", self.positives_path,
                            "--index", self.index_path,
                            "--src", self.src,
                            "--dst", dst,
                            "--evidence-only",
                            "--code-revision", "test-revision",
                            "--quiet",
                        ]
                    ),
                )
            )
            completed = subprocess.run(
                [sys.executable, "-c", snippet],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(
                completed.returncode, 0, completed.stderr.decode("utf-8", "replace")
            )
            digests.append(digest_tree(dst))
        self.assertEqual(len(set(digests)), 1, "output depends on PYTHONHASHSEED")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
