"""Tests for ``audit.py`` -- the read-only STOP-gate runner.

An auditor is only worth having if it *fails*.  Every check below is therefore
tested twice: once on a clean artifact (must PASS) and once on an artifact into
which the 2025 defect has been deliberately planted (must FAIL).  The planted
defects are the real ones:

* ``Q8NCT1`` labelled ``AVPR2`` (Q8NCT1 is ARRDC4; the real AVPR2 is P30518) --
  the row that decided which receptor got docked;
* an Entrez id claiming two accessions (collapse factor 9 in the retired map);
* a curated directory whose name disagrees with its own ``job_identity.json``;
* ``complex_*.pdb`` leaking into an evidence-only tree;
* a manifest and a directory tree that disagree on how many jobs exist -- the
  shape of the 10,000-file cap that silently dropped 58% of the campaign.

Hermetic: ``tempfile.TemporaryDirectory()`` only, no network, no external drive.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd  # noqa: E402

import audit  # noqa: E402
import curate_jobs  # noqa: E402
from audit import (  # noqa: E402
    FAIL,
    PASS,
    SKIP,
    audit_curated,
    audit_identity,
    audit_run,
    exit_code,
    format_table,
)
from dock_selectivity.identity import IdentityIndex, Protein  # noqa: E402
from dock_selectivity.jobid import (  # noqa: E402
    JOB_IDENTITY_FILENAME,
    format_job_dir,
)
from dock_selectivity.schemas import write_checked  # noqa: E402

# --------------------------------------------------------------------------
# fixtures (kept local so this file runs on its own)
# --------------------------------------------------------------------------

CLEAN_PROTEINS = (
    ("P30518", "V2R_HUMAN", "AVPR2", ("AVPR2",), 554),
    ("Q8NCT1", "ARRD4_HUMAN", "ARRDC4", ("ARRDC4",), 91947),
    ("P68363", "TBA1B_HUMAN", "TUBA1B", ("TUBA1B", "TUBA1B1"), 10376),
    ("Q13509", "TBB3_HUMAN", "TUBB3", ("TUBB3", "TUBB4"), 10381),
    ("P60568", "IL2_HUMAN", "IL2", ("IL2",), 3558),
    ("P01589", "IL2RA_HUMAN", "IL2RA", ("IL2RA",), 3559),
    ("P03886", "NU1M_HUMAN", "MT-ND1", ("MT-ND1", "ND1"), 4535),
)


def proteins(records=CLEAN_PROTEINS):
    return [
        Protein(acc, entry, primary, symbols, entrez, True)
        for acc, entry, primary, symbols, entrez in records
    ]


def write_index_frame(path, records, validate=False):
    """Write an index parquet, bypassing validation so defects can be planted."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    IdentityIndex(records, validate=validate).to_frame().to_parquet(path, index=False)
    return path


def make_job_dir(root, name, with_complex=True, with_final=True):
    files = {}
    if with_final:
        files["final_results.csv"] = "Tool1,Tool2,Score1,Score2,RMSD\nSMINA,GNINA,-7.1,-7.4,0.82\n"
    for tool in ("smina", "gnina", "gold", "ledock"):
        files["{0}/results.csv".format(tool)] = "pose,score\n1,-7.1\n"
    files["smina/out.sdf"] = "smina poses\n"
    files["gnina/out.sdf"] = "gnina poses\n"
    files["ledock/out.dok"] = "ledock poses\n"
    files["gold/output/gold_soln_DB00035_m1_1.sdf"] = "gold pose 1\n"
    files["input/DB00035.mol2"] = "@<TRIPOS>MOLECULE\n"
    files["input/receptor.mol2"] = "@<TRIPOS>MOLECULE\n" * 10
    if with_complex:
        for tool in ("smina", "gnina"):
            files["{0}/complex_1.pdb".format(tool)] = "ATOM\n" * 100
        files["log_20250704_071438_c57a.log"] = "retry\n"
    for relpath, content in files.items():
        full = os.path.join(root, name, relpath)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as handle:
            handle.write(content)
    return os.path.join(root, name)


def by_name(results):
    return {r.name: r for r in results}


class RunDirFixture(unittest.TestCase):
    """A miniature ``$NEW`` built by actually running ``curate_jobs.py``."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.addCleanup(self.tmp.cleanup)

        self.src = os.path.join(self.root, "old", "consensus_docking_results")
        self.new = os.path.join(self.root, "new")
        self.manifests = os.path.join(self.new, "manifests")
        self.curated = os.path.join(self.new, "docking_results")
        os.makedirs(self.src)
        os.makedirs(self.manifests)

        self.index_path = os.path.join(self.new, "identity", "identity_index.parquet")
        write_index_frame(self.index_path, proteins(), validate=True)
        self.index = IdentityIndex.load(self.index_path)

        self.old_dirs = [
            "DB00035_AVPR2_Q8NCT1_cavity_2",
            "DB00035_nan_P30518_cavity_1",
            "DB00157_MT-ND1_P03886_cavity_1",
            "DB00157_nan_P68363_cavity_3",
            "DB00157_nan_Q13509_cavity_1",
        ]
        for name in self.old_dirs:
            make_job_dir(self.src, name)

        self.write_manifests()
        self.run_curation()

    # -- manifests ---------------------------------------------------------

    def reuse_rows(self):
        common = dict(old_gene_label="whatever", label_was_correct=False,
                      new_job_dir=None, excluded_reason="not selected")
        return [
            dict(old_job_dir="DB00035_AVPR2_Q8NCT1_cavity_2", classification="true_positive",
                 drugbank_id="DB00035", accession="Q8NCT1", gene="ARRDC4",
                 fragment_id="F1", cavity_index=2, **common),
            dict(old_job_dir="DB00035_nan_P30518_cavity_1", classification="true_positive",
                 drugbank_id="DB00035", accession="P30518", gene="AVPR2",
                 fragment_id="F1", cavity_index=1, **common),
            dict(old_job_dir="DB00157_MT-ND1_P03886_cavity_1",
                 classification="valid_non_interaction", drugbank_id="DB00157",
                 accession="P03886", gene="MT-ND1", fragment_id="F1", cavity_index=1, **common),
            dict(old_job_dir="DB00157_nan_P68363_cavity_3",
                 classification="valid_non_interaction", drugbank_id="DB00157",
                 accession="P68363", gene="TUBA1B", fragment_id="F2", cavity_index=3, **common),
            dict(old_job_dir="DB00157_nan_Q13509_cavity_1",
                 classification="valid_non_interaction", drugbank_id="DB00157",
                 accession="Q13509", gene="TUBB3", fragment_id="F1", cavity_index=1, **common),
        ]

    def positive_rows(self):
        return [
            dict(drugbank_id="DB00035", accession="Q8NCT1", gene="ARRDC4",
                 fragment_id="F1", cavity_index=2, sample_type="positive",
                 selected_by="entrez=91947"),
            dict(drugbank_id="DB00035", accession="P30518", gene="AVPR2",
                 fragment_id="F1", cavity_index=1, sample_type="positive",
                 selected_by="entrez=554"),
        ]

    def drawn_rows(self):
        return [
            dict(drugbank_id="DB00157", accession="P03886", gene="MT-ND1",
                 fragment_id="F1", cavity_index=1, sample_type="negative",
                 source="pool", old_job_dir="DB00157_MT-ND1_P03886_cavity_1",
                 selected_by="balanced draw"),
            dict(drugbank_id="DB00157", accession="P68363", gene="TUBA1B",
                 fragment_id="F2", cavity_index=3, sample_type="negative",
                 source="pool", old_job_dir="DB00157_nan_P68363_cavity_3",
                 selected_by="balanced draw"),
            dict(drugbank_id="DB00157", accession="P01589", gene="IL2RA",
                 fragment_id="F1", cavity_index=2, sample_type="negative",
                 source="new", old_job_dir=None, selected_by="balanced draw"),
        ]

    def interaction_rows(self):
        return [
            dict(drugbank_id="DB00035", entrez_id=91947, accession="Q8NCT1",
                 gene="ARRDC4", selected_by="edge (DB00035,91947)"),
            dict(drugbank_id="DB00035", entrez_id=554, accession="P30518",
                 gene="AVPR2", selected_by="edge (DB00035,554)"),
        ]

    def pool_rows(self):
        return [
            dict(old_job_dir=r["old_job_dir"], drugbank_id=r["drugbank_id"],
                 accession=r["accession"], gene=r["gene"], fragment_id=r["fragment_id"],
                 cavity_index=r["cavity_index"], old_sample_type="negative",
                 old_gene_label=r["old_gene_label"],
                 classification="valid_non_interaction", has_final_results=True)
            for r in self.reuse_rows()
            if r["classification"] == "valid_non_interaction"
        ]

    def redock_rows(self):
        return [
            dict(job_id=format_job_dir("DB00157", "P01589", "IL2RA", "F1", 2),
                 drugbank_id="DB00157", accession="P01589", gene="IL2RA",
                 fragment_id="F1", cavity_index=2, sample_type="negative",
                 selected_by="balanced draw"),
        ]

    def path(self, filename):
        return os.path.join(self.manifests, filename)

    def write_manifests(self):
        write_checked(pd.DataFrame(self.reuse_rows()), self.path("job_classification.csv"), "job_classification")
        write_checked(pd.DataFrame(self.positive_rows()), self.path("positives_planned.csv"),
                      "positives_planned")
        write_checked(pd.DataFrame(self.drawn_rows()), self.path("negatives_drawn.csv"),
                      "negatives_drawn")
        write_checked(pd.DataFrame(self.interaction_rows()),
                      self.path("interactions_resolved.csv"), "interactions_resolved")
        write_checked(pd.DataFrame(self.pool_rows()), self.path("negatives_pool.csv"),
                      "negatives_pool")
        write_checked(pd.DataFrame(self.redock_rows()), self.path("redock_queue.csv"),
                      "redock_queue")

    def run_curation(self):
        reused_manifest = os.path.join(self.root, "reused_manifest.csv")
        code = curate_jobs.main(
            [
                "--job-classification", self.path("job_classification.csv"),
                "--drawn", self.path("negatives_drawn.csv"),
                "--positives", self.path("positives_planned.csv"),
                "--index", self.index_path,
                "--src", self.src,
                "--dst", self.curated,
                "--evidence-only",
                "--manifest-out", reused_manifest,
                "--code-revision", "test-revision",
                "--quiet",
            ]
        )
        self.assertEqual(code, 0)
        reused = pd.read_csv(reused_manifest)
        newly = pd.DataFrame(
            [
                dict(job_id=r["job_id"], drugbank_id=r["drugbank_id"],
                     accession=r["accession"], gene=r["gene"], fragment_id=r["fragment_id"],
                     cavity_index=r["cavity_index"], sample_type=r["sample_type"],
                     provenance="newly_docked", source_job=None,
                     selected_by=r["selected_by"])
                for r in self.redock_rows()
            ]
        )
        write_checked(
            pd.concat([reused, newly], ignore_index=True),
            self.path("run_manifest.csv"),
            "run_manifest",
        )
        self.manifest_path = self.path("run_manifest.csv")

    # -- helpers -----------------------------------------------------------

    def curated_results(self, **kwargs):
        return audit_curated(self.curated, self.manifest_path, self.index_path, **kwargs)

    def any_job(self):
        return sorted(os.listdir(self.curated))[0]

    def rewrite_identity(self, job, **changes):
        path = os.path.join(self.curated, job, JOB_IDENTITY_FILENAME)
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        record.update(changes)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)


# --------------------------------------------------------------------------
# audit identity
# --------------------------------------------------------------------------


class TestAuditIdentity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.root, "identity_index.parquet")

    def run_audit(self, records, validate=False, **kwargs):
        write_index_frame(self.path, records, validate=validate)
        kwargs.setdefault("skip_determinism", True)
        return by_name(audit_identity(self.path, **kwargs))

    def test_a_clean_index_passes_every_invariant(self):
        results = self.run_audit(proteins(), validate=True)
        for key in (
            "G1_one_row_per_accession",
            "G4_primary_symbol_in_symbols",
            "G5_entrez_collapse_factor_one",
            "G7_no_isoform_accessions",
            "accession_syntax",
            "index_self_consistent",
            "spot_AVPR2",
            "spot_TUBA1B",
            "spot_IL2",
        ):
            self.assertEqual(results[key].status, PASS, results[key].detail)
        self.assertEqual(results["G5_entrez_collapse_factor_one"].data["max_collapse"], 1)

    def test_G5_fails_when_one_entrez_id_claims_two_accessions(self):
        records = proteins() + [
            Protein("Q96RT7", "GCP6_HUMAN", "TUBGCP6", ("TUBGCP6",), 10376, True)
        ]
        results = self.run_audit(records)
        check = results["G5_entrez_collapse_factor_one"]
        self.assertEqual(check.status, FAIL)
        self.assertEqual(check.data["max_collapse"], 2)
        self.assertIn("receptor that got docked", check.detail)

    def test_G4_fails_on_the_exact_shape_of_the_incident(self):
        # Q8NCT1 (ARRDC4) carrying the primary symbol AVPR2: the broadcast bug.
        records = [p for p in proteins() if p.accession != "Q8NCT1"] + [
            Protein("Q8NCT1", "ARRD4_HUMAN", "AVPR2", ("ARRDC4",), 91947, True)
        ]
        results = self.run_audit(records)
        self.assertEqual(results["G4_primary_symbol_in_symbols"].status, FAIL)
        self.assertIn("KNOWN_ISSUE", results["G4_primary_symbol_in_symbols"].detail)

    def test_G1_fails_on_a_duplicated_accession(self):
        records = proteins() + [
            Protein("P30518", "V2R_HUMAN", "AVPR2", ("AVPR2",), 554, True)
        ]
        results = self.run_audit(records)
        self.assertEqual(results["G1_one_row_per_accession"].status, FAIL)
        self.assertEqual(results["G1_one_row_per_accession"].data["n_duplicate_accessions"], 1)

    def test_G7_fails_on_an_isoform_accession(self):
        records = proteins() + [
            Protein("P30518-2", "V2R_HUMAN", "AVPR2", ("AVPR2",), 5540, True)
        ]
        results = self.run_audit(records)
        self.assertEqual(results["G7_no_isoform_accessions"].status, FAIL)

    def test_a_gene_symbol_in_the_accession_column_is_rejected(self):
        records = proteins() + [
            Protein("AVPR2", "V2R_HUMAN", "AVPR2", ("AVPR2",), 5541, True)
        ]
        results = self.run_audit(records)
        self.assertEqual(results["accession_syntax"].status, FAIL)

    def test_the_spot_check_catches_an_impostor_that_still_claims_the_symbol(self):
        records = [p for p in proteins() if p.accession != "Q8NCT1"] + [
            Protein("Q8NCT1", "ARRD4_HUMAN", "ARRDC4", ("ARRDC4", "AVPR2"), 91947, True)
        ]
        results = self.run_audit(records)
        self.assertEqual(results["spot_AVPR2"].status, FAIL)
        self.assertIn("impostor Q8NCT1", results["spot_AVPR2"].detail)
        self.assertEqual(results["spot_TUBA1B"].status, PASS)

    def test_the_spot_check_fails_when_the_correct_accession_is_missing(self):
        records = [p for p in proteins() if p.accession != "P68363"]
        results = self.run_audit(records)
        self.assertEqual(results["spot_TUBA1B"].status, FAIL)
        self.assertIn("P68363 is not in the index", results["spot_TUBA1B"].detail)

    def test_a_missing_index_is_a_failure_not_a_fallback(self):
        results = by_name(audit_identity(os.path.join(self.root, "absent.parquet")))
        self.assertEqual(results["index_exists"].status, FAIL)
        self.assertIn("no fallback identity source", results["index_exists"].detail)

    def test_a_sha256_mismatch_refuses_the_index(self):
        results = self.run_audit(proteins(), validate=True, expect_sha256="0" * 64)
        self.assertEqual(results["index_sha256_pinned"].status, FAIL)

    def test_a_matching_sha256_passes(self):
        write_index_frame(self.path, proteins(), validate=True)
        index = IdentityIndex.load(self.path)
        results = by_name(
            audit_identity(
                self.path, expect_sha256=index.source_sha256, skip_determinism=True
            )
        )
        self.assertEqual(results["index_sha256_pinned"].status, PASS)

    def test_accounting_is_skipped_not_passed_when_the_meta_file_is_absent(self):
        results = self.run_audit(proteins(), validate=True)
        self.assertEqual(results["G2_total_accounting"].status, SKIP)
        self.assertNotEqual(
            exit_code(list(results.values())),
            0,
            "an unrun check must not clear the gate",
        )

    def test_accounting_passes_when_the_ledger_balances(self):
        write_index_frame(self.path, proteins(), validate=True)
        meta = os.path.join(self.root, "identity_index.meta.json")
        rejects = os.path.join(self.root, "unresolved_targets.csv")
        with open(rejects, "w", encoding="utf-8") as handle:
            handle.write("key,status,reason,candidates\n99,UNKNOWN,retired,\n")
        with open(meta, "w", encoding="utf-8") as handle:
            json.dump({"n_in": 8, "n_resolved": 7, "n_rejected": 1}, handle)
        results = by_name(audit_identity(self.path, skip_determinism=True))
        self.assertEqual(results["G2_total_accounting"].status, PASS)

    def test_accounting_fails_when_rows_vanish(self):
        write_index_frame(self.path, proteins(), validate=True)
        with open(
            os.path.join(self.root, "identity_index.meta.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump({"n_in": 100, "n_resolved": 7, "n_rejected": 1}, handle)
        results = by_name(audit_identity(self.path, skip_determinism=True))
        self.assertEqual(results["G2_total_accounting"].status, FAIL)
        self.assertIn("unaccounted", results["G2_total_accounting"].detail)

    def test_determinism_check_runs_and_passes_on_a_real_index(self):
        write_index_frame(self.path, proteins(), validate=True)
        results = by_name(audit_identity(self.path, skip_determinism=False))
        self.assertEqual(
            results["G3_deterministic_ordering"].status,
            PASS,
            results["G3_deterministic_ordering"].detail,
        )

    def test_skipping_determinism_is_reported_as_skip(self):
        results = self.run_audit(proteins(), validate=True)
        self.assertEqual(results["G3_deterministic_ordering"].status, SKIP)


# --------------------------------------------------------------------------
# audit curated
# --------------------------------------------------------------------------


class TestAuditCurated(RunDirFixture):
    def test_a_clean_curated_tree_passes(self):
        results = self.curated_results()
        failures = [r for r in results if r.status == FAIL]
        self.assertEqual(failures, [], format_table(results))
        self.assertEqual(exit_code(results), 0, format_table(results))

    def test_a_planted_gene_mismatch_fails(self):
        # the incident, planted: gene AVPR2 on accession Q8NCT1 (ARRDC4)
        job = "DB00035__Q8NCT1__ARRDC4__F1__cav2"
        self.rewrite_identity(job, gene="AVPR2")
        results = by_name(self.curated_results())
        self.assertEqual(results["gene_equals_index_label"].status, FAIL)
        self.assertIn("Q8NCT1", results["gene_equals_index_label"].detail)
        self.assertEqual(exit_code(list(results.values())), 1)

    def test_one_accession_carrying_two_gene_labels_fails(self):
        # The signature of the retired map: 4,585 accessions collapsed onto
        # 1,795 symbols, so one accession could appear under several names.
        original = "DB00035__Q8NCT1__ARRDC4__F1__cav2"
        clone = "DB00035__Q8NCT1__AVPR2__F1__cav3"
        shutil.copytree(
            os.path.join(self.curated, original), os.path.join(self.curated, clone)
        )
        self.rewrite_identity(clone, gene="AVPR2", cavity_index=3, dir_name=clone)
        results = by_name(self.curated_results())
        check = results["accession_gene_collapse_factor_one"]
        self.assertEqual(check.status, FAIL)
        self.assertEqual(check.data["max_collapse"], 2)

    def test_a_missing_job_identity_fails(self):
        os.unlink(os.path.join(self.curated, self.any_job(), JOB_IDENTITY_FILENAME))
        results = by_name(self.curated_results())
        self.assertEqual(results["job_identity_present"].status, FAIL)

    def test_a_missing_final_results_fails(self):
        os.unlink(os.path.join(self.curated, self.any_job(), "final_results.csv"))
        results = by_name(self.curated_results())
        self.assertEqual(results["final_results_present"].status, FAIL)

    def test_a_leaked_complex_pdb_fails(self):
        leaked = os.path.join(self.curated, self.any_job(), "smina", "complex_7.pdb")
        with open(leaked, "w", encoding="utf-8") as handle:
            handle.write("ATOM\n")
        results = by_name(self.curated_results())
        self.assertEqual(results["no_excluded_artifacts"].status, FAIL)
        self.assertIn("complex_7.pdb", results["no_excluded_artifacts"].detail)

    def test_a_leaked_retry_log_fails(self):
        with open(
            os.path.join(self.curated, self.any_job(), "log_20250704_071438_c57a.log"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write("retry\n")
        results = by_name(self.curated_results())
        self.assertEqual(results["no_excluded_artifacts"].status, FAIL)

    def test_a_directory_name_disagreeing_with_its_own_identity_fails(self):
        job = self.any_job()
        os.rename(
            os.path.join(self.curated, job),
            os.path.join(self.curated, "DB00035__P30518__WRONG__F1__cav1"),
        )
        results = by_name(self.curated_results())
        self.assertEqual(results["dir_name_matches_identity"].status, FAIL)

    def test_a_missing_directory_fails_the_count_and_the_set(self):
        shutil.rmtree(os.path.join(self.curated, self.any_job()))
        results = by_name(self.curated_results())
        self.assertEqual(results["dir_count_equals_manifest"].status, FAIL)
        self.assertEqual(results["dir_set_equals_manifest"].status, FAIL)
        self.assertIn("truncated", results["dir_count_equals_manifest"].detail)

    def test_a_directory_absent_from_the_manifest_fails(self):
        shutil.copytree(
            os.path.join(self.curated, self.any_job()),
            os.path.join(self.curated, "DB00035__P60568__IL2__F1__cav9"),
        )
        results = by_name(self.curated_results())
        self.assertEqual(results["dir_set_equals_manifest"].status, FAIL)

    def test_expect_exact_requires_the_not_yet_docked_rows(self):
        subset = by_name(self.curated_results(expect="subset"))
        self.assertEqual(subset["dir_count_equals_manifest"].status, PASS)
        exact = by_name(self.curated_results(expect="exact"))
        self.assertEqual(exact["dir_count_equals_manifest"].status, FAIL)
        self.assertEqual(exact["dir_set_equals_manifest"].status, FAIL)

    def test_subset_mode_still_requires_every_reused_row(self):
        """A partial Phase 7 must not mask a missing reused job."""
        # materialise the newly-docked row: subset must still pass ...
        shutil.copytree(
            os.path.join(self.curated, "DB00035__P30518__AVPR2__F1__cav1"),
            os.path.join(self.curated, "DB00157__P01589__IL2RA__F1__cav2"),
        )
        self.rewrite_identity(
            "DB00157__P01589__IL2RA__F1__cav2",
            drugbank_id="DB00157",
            accession="P01589",
            gene="IL2RA",
            cavity_index=2,
            sample_type="negative",
            provenance="newly_docked",
            source_job=None,
            dir_name="DB00157__P01589__IL2RA__F1__cav2",
        )
        results = by_name(self.curated_results())
        self.assertEqual(results["dir_count_equals_manifest"].status, PASS)
        self.assertEqual(results["dir_set_equals_manifest"].status, PASS)
        # ... and now exact passes too, because the design is fully materialised
        self.assertEqual(
            by_name(self.curated_results(expect="exact"))["dir_count_equals_manifest"].status,
            PASS,
        )
        # but a missing REUSED job is a failure in either mode
        shutil.rmtree(os.path.join(self.curated, "DB00035__P30518__AVPR2__F1__cav1"))
        results = by_name(self.curated_results())
        self.assertEqual(results["dir_set_equals_manifest"].status, FAIL)
        self.assertEqual(results["dir_count_equals_manifest"].status, FAIL)

    def test_a_truncated_evidence_file_is_detected(self):
        job = self.any_job()
        with open(
            os.path.join(self.curated, job, "smina", "out.sdf"), "w", encoding="utf-8"
        ) as handle:
            handle.write("")
        results = by_name(self.curated_results())
        self.assertEqual(results["evidence_complete"].status, FAIL)
        self.assertIn("size mismatch", results["evidence_complete"].detail)

    def test_an_identity_disagreeing_with_its_manifest_row_fails(self):
        self.rewrite_identity(
            "DB00035__Q8NCT1__ARRDC4__F1__cav2", sample_type="negative"
        )
        results = by_name(self.curated_results())
        self.assertEqual(results["identity_matches_manifest"].status, FAIL)
        self.assertIn("sample_type", results["identity_matches_manifest"].detail)

    def test_a_reused_job_with_no_source_is_not_traceable_and_fails(self):
        self.rewrite_identity("DB00035__Q8NCT1__ARRDC4__F1__cav2", source_job="")
        results = by_name(self.curated_results())
        self.assertEqual(results["reused_jobs_name_their_source"].status, FAIL)

    def test_a_stray_file_in_the_tree_is_reported(self):
        with open(os.path.join(self.curated, "notes.txt"), "w", encoding="utf-8") as handle:
            handle.write("hello\n")
        results = by_name(self.curated_results())
        self.assertEqual(results["no_stray_files"].status, FAIL)

    def test_limit_reports_that_the_tree_was_not_fully_scanned(self):
        results = by_name(self.curated_results(limit=1))
        self.assertEqual(results["full_tree_scanned"].status, SKIP)
        self.assertNotEqual(exit_code(list(results.values())), 0)

    def test_an_invalid_manifest_stops_the_gate_rather_than_passing_it(self):
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            handle.write("job_id\nnot,a,manifest\n")
        results = by_name(self.curated_results())
        self.assertEqual(results["manifest_valid"].status, FAIL)
        self.assertEqual(len(results), 1, "no further check may claim to have run")


# --------------------------------------------------------------------------
# audit run
# --------------------------------------------------------------------------


class TestAuditRun(RunDirFixture):
    def results(self, **kwargs):
        kwargs.setdefault("skip_determinism", True)
        return by_name(audit_run(self.new, **kwargs))

    def test_a_complete_clean_run_directory_passes(self):
        results = self.results()
        failures = [r for r in results.values() if r.status == FAIL]
        self.assertEqual(
            failures, [], format_table(sorted(results.values(), key=lambda r: r.name))
        )

    def test_every_manifest_is_validated(self):
        results = self.results()
        for _, schema in audit.MANIFESTS:
            self.assertEqual(results["manifest.{0}".format(schema)].status, PASS)

    def test_a_missing_manifest_is_skipped_never_passed(self):
        os.unlink(self.path("redock_queue.csv"))
        results = self.results()
        self.assertEqual(results["manifest.redock_queue"].status, SKIP)
        self.assertNotEqual(exit_code(list(results.values())), 0)

    def test_a_contaminated_negative_fails(self):
        rows = self.interaction_rows() + [
            dict(drugbank_id="DB00157", entrez_id=10376, accession="P68363",
                 gene="TUBA1B", selected_by="edge (DB00157,10376)")
        ]
        write_checked(
            pd.DataFrame(rows), self.path("interactions_resolved.csv"), "interactions_resolved"
        )
        results = self.results()
        check = results["xstage.negatives_not_known_interactions"]
        self.assertEqual(check.status, FAIL)
        self.assertEqual(check.data["n_contaminated"], 1)

    def test_a_pocket_in_both_classes_fails(self):
        rows = self.positive_rows() + [
            dict(drugbank_id="DB00157", accession="P68363", gene="TUBA1B",
                 fragment_id="F2", cavity_index=3, sample_type="positive",
                 selected_by="entrez=10376")
        ]
        write_checked(pd.DataFrame(rows), self.path("positives_planned.csv"), "positives_planned")
        results = self.results()
        self.assertEqual(results["xstage.positive_negative_disjoint"].status, FAIL)

    def test_a_pool_draw_that_names_a_true_positive_job_fails(self):
        rows = self.drawn_rows()
        rows[0]["old_job_dir"] = "DB00035_nan_P30518_cavity_1"  # a true positive
        rows[0]["accession"] = "P30518"
        rows[0]["gene"] = "AVPR2"
        rows[0]["cavity_index"] = 1
        write_checked(pd.DataFrame(rows), self.path("negatives_drawn.csv"), "negatives_drawn")
        results = self.results()
        self.assertEqual(results["xstage.pool_draws_reference_pool_jobs"].status, FAIL)
        self.assertIn("true_positive", results["xstage.pool_draws_reference_pool_jobs"].detail)

    def test_a_manifest_gene_that_is_not_the_index_label_fails(self):
        rows = self.positive_rows()
        rows[0]["gene"] = "AVPR2"  # on Q8NCT1, which is ARRDC4
        write_checked(pd.DataFrame(rows), self.path("positives_planned.csv"), "positives_planned")
        results = self.results()
        self.assertEqual(results["xstage.gene_labels"].status, FAIL)
        self.assertIn("positives_planned", results["xstage.gene_labels"].detail)

    def test_a_job_id_that_is_not_derivable_from_its_own_row_fails(self):
        manifest = pd.read_csv(self.manifest_path)
        manifest.loc[0, "job_id"] = "DB00035_AVPR2_Q8NCT1_cavity_2"  # the old scheme
        manifest.to_csv(self.manifest_path, index=False, lineterminator="\n")
        results = self.results()
        self.assertEqual(results["xstage.run_manifest_job_id_derivable"].status, FAIL)

    def test_a_design_job_missing_from_the_run_manifest_fails(self):
        manifest = pd.read_csv(self.manifest_path)
        manifest.iloc[1:].to_csv(self.manifest_path, index=False, lineterminator="\n")
        results = self.results()
        self.assertEqual(results["xstage.run_manifest_covers_design"].status, FAIL)

    def test_the_curated_tree_is_audited_as_part_of_the_run(self):
        results = self.results()
        self.assertEqual(results["curated.gene_equals_index_label"].status, PASS)
        self.rewrite_identity("DB00035__P30518__AVPR2__F1__cav1", gene="TUBA1B")
        self.assertEqual(self.results()["curated.gene_equals_index_label"].status, FAIL)

    def test_a_broken_identity_index_does_not_silently_skip_the_run(self):
        os.unlink(self.index_path)
        results = self.results()
        self.assertEqual(results["identity.index_exists"].status, FAIL)
        self.assertEqual(results["xstage.gene_labels"].status, SKIP)


# --------------------------------------------------------------------------
# output and exit codes
# --------------------------------------------------------------------------


class TestOutput(RunDirFixture):
    def test_the_table_lists_every_check_and_repeats_the_failures(self):
        self.rewrite_identity("DB00035__P30518__AVPR2__F1__cav1", gene="TUBA1B")
        results = self.curated_results()
        table = format_table(results)
        for result in results:
            self.assertIn(result.name[:44], table)
        self.assertIn("FAILURES", table)
        self.assertIn("check(s):", table)

    def test_exit_code_semantics(self):
        clean = [audit.CheckResult("a", PASS, "")]
        skipped = clean + [audit.CheckResult("b", SKIP, "")]
        broken = skipped + [audit.CheckResult("c", FAIL, "")]
        self.assertEqual(exit_code(clean), 0)
        self.assertEqual(exit_code(skipped), 2)
        self.assertEqual(exit_code(skipped, allow_skips=True), 0)
        self.assertEqual(exit_code(broken), 1)
        self.assertEqual(exit_code(broken, allow_skips=True), 1)

    def test_cli_curated_exits_zero_on_a_clean_tree_and_one_on_a_planted_defect(self):
        argv = [
            "curated",
            "--dir", self.curated,
            "--manifest", self.manifest_path,
            "--index", self.index_path,
            "--quiet",
        ]
        self.assertEqual(audit.main(argv), 0)
        self.rewrite_identity("DB00035__P30518__AVPR2__F1__cav1", gene="TUBA1B")
        self.assertEqual(audit.main(argv), 1)

    def test_cli_writes_a_json_report_and_never_touches_the_tree(self):
        report = os.path.join(self.root, "audit.json")
        before = sorted(
            (os.path.relpath(os.path.join(d, f), self.curated), os.path.getsize(os.path.join(d, f)))
            for d, _, files in os.walk(self.curated)
            for f in files
        )
        audit.main(
            [
                "--json", report,
                "curated",
                "--dir", self.curated,
                "--manifest", self.manifest_path,
                "--index", self.index_path,
                "--quiet",
            ]
        )
        with open(report, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertTrue(payload)
        self.assertTrue(all("status" in entry for entry in payload))
        after = sorted(
            (os.path.relpath(os.path.join(d, f), self.curated), os.path.getsize(os.path.join(d, f)))
            for d, _, files in os.walk(self.curated)
            for f in files
        )
        self.assertEqual(before, after)

    def test_cli_identity_exits_two_when_a_check_could_not_run(self):
        code = audit.main(
            ["identity", "--index", self.index_path, "--skip-determinism", "--quiet"]
        )
        self.assertEqual(code, 2)
        code = audit.main(
            [
                "--allow-skips",
                "identity",
                "--index", self.index_path,
                "--skip-determinism",
                "--quiet",
            ]
        )
        self.assertEqual(code, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
