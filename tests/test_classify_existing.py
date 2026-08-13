"""Tests for ``classify_existing.py`` -- REBUILD_PLAN Phase 2.

Hermetic: every fixture is built in a ``tempfile.TemporaryDirectory()``. Nothing
here touches ``/media/onur/Elements``, ``/opt/data`` or the network. Real
directory names, accessions and gene symbols from the 2025-06-29 campaign appear
as **literals**, copied out of the assessment -- never read from the drive.

Each test names the campaign failure it reproduces:

* the gene token ``AVPR2`` sitting on ``Q8NCT1`` (which is ``ARRDC4``) -- the
  row that decided which receptor got docked;
* ``TUBA1B`` collapsing onto both P68363 and Q13509;
* the 1,777 directories the old regex rejected (1,743 ``-F<k>-model_v1``
  multi-fragment names, 34 hyphenated symbols such as ``MT-ND1``);
* the 11,806 directories whose gene token is the literal string ``nan``;
* ``Fragment_ID`` written as ``'F1'`` on positives and bare ``'1'`` on
  negatives, which made a later merge match neither;
* ``if not uniprot_id: continue`` -- a drop with no ledger.
"""

from __future__ import annotations

import hashlib
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

import classify_existing  # noqa: E402
from dock_selectivity import schemas  # noqa: E402
from dock_selectivity.identity import IdentityIndex, Protein  # noqa: E402
from dock_selectivity.identity import (  # noqa: E402
    InvariantViolation as IdentityInvariantViolation,
)
from dock_selectivity.invariants import (  # noqa: E402
    InvariantViolation,
    assert_collapse_factor_one,
    assert_deterministic_across_hash_seeds,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

#: Real accessions with their real symbols. Q8NCT1 is ARRDC4, not AVPR2, and
#: Q13509 is TUBB3, not TUBA1B -- the two mislabellings the campaign made.
PROTEINS = (
    Protein("P30518", "V2R_HUMAN", "AVPR2", ("AVPR2",), 554, True),
    Protein("Q8NCT1", "ARRD4_HUMAN", "ARRDC4", ("ARRDC4",), 91947, True),
    Protein("P68363", "TBA1B_HUMAN", "TUBA1B", ("TUBA1B", "K-ALPHA-1"), 10376, True),
    Protein("Q13509", "TBB3_HUMAN", "TUBB3", ("TUBB3", "TUBB4"), 10381, True),
    Protein("P03886", "NU1M_HUMAN", "MT-ND1", ("MT-ND1", "ND1"), 4535, True),
    Protein("P08F94", "PKHD1_HUMAN", "PKHD1", ("PKHD1", "FCYT", "TIGM1"), 5314, False),
    Protein("P08173", "ACM4_HUMAN", "CHRM4", ("CHRM4",), 1132, True),
    Protein("P47901", "V1BR_HUMAN", "AVPR1B", ("AVPR1B", "AVPR3"), 553, True),
)

#: ``directory name -> has a non-empty final_results.csv``.
#: 9 directories + 1 stray file, mirroring the campaign's 24,173 + 1.
CAMPAIGN_DIRS = {
    # true positive: DB00035 really does target AVPR2 = P30518
    "DB00035_AVPR2_P30518_cavity_1": True,
    # the incident itself: label AVPR2, receptor Q8NCT1 (= ARRDC4)
    "DB00035_AVPR2_Q8NCT1_cavity_2": True,
    # gene token is the literal string 'nan' (11,806 real directories)
    "DB00035_nan_P08173_cavity_3": True,
    # hyphenated symbol (34 real directories); a true positive
    "DB00157_MT-ND1_P03886_cavity_1": True,
    # multi-fragment legacy name (1,743 real directories)
    "DB00091_CAMLG_P08F94_-F13-model_v1_cavity_10": True,
    # same corrupt gene token on two different accessions
    "DB00570_TUBA1B_P68363_cavity_1": True,
    "DB00570_TUBA1B_Q13509_cavity_1": True,
    # complete directory, no final_results.csv -> incomplete
    "DB00035_AVPR1B_P47901_cavity_8": False,
    # a new-scheme name is NOT a legacy name: unparseable, recorded
    "DB00035__P30518__AVPR2__F1__cav1": True,
}

STRAY_ENTRY = "docking_automation.log"

#: (drugbank_id, entrez_id, accession, gene) -- the corrected edge list.
INTERACTIONS = (
    ("DB00035", 554, "P30518", "AVPR2"),
    ("DB00570", 10376, "P68363", "TUBA1B"),
    ("DB00157", 4535, "P03886", "MT-ND1"),
)


def make_index(proteins=PROTEINS) -> IdentityIndex:
    return IdentityIndex(list(proteins))


def write_index(path, proteins=PROTEINS) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    make_index(proteins).to_frame().to_parquet(path, index=False)
    return path


def write_interactions(path, edges=INTERACTIONS) -> str:
    frame = pd.DataFrame(
        [
            {
                "drugbank_id": drug,
                "entrez_id": entrez,
                "accession": accession,
                "gene": gene,
                "selected_by": "entrez={0} via edge ({1},{0})".format(entrez, drug),
            }
            for drug, entrez, accession, gene in edges
        ],
        columns=["drugbank_id", "entrez_id", "accession", "gene", "selected_by"],
    )
    return schemas.write_checked(frame, path, "interactions_resolved")


def build_old_tree(root, dirs=CAMPAIGN_DIRS, stray=STRAY_ENTRY) -> str:
    """A miniature ``$OLD/consensus_docking_results``."""
    old = os.path.join(root, "old_run", "consensus_docking_results")
    os.makedirs(old)
    for name, complete in dirs.items():
        job = os.path.join(old, name)
        os.makedirs(job)
        # every job carries the engine subdirectories; only some finished
        os.makedirs(os.path.join(job, "smina"))
        if complete:
            with open(os.path.join(job, classify_existing.FINAL_RESULTS_FILENAME), "w") as fh:
                fh.write("Tool1,Tool2,Score1,Score2,RMSD\nSMINA,GNINA,-6.5,-6.4,1.23\n")
    if stray:
        with open(os.path.join(old, stray), "w") as fh:
            fh.write("2025-07-04 docking automation started\n")
    return old


def observations(dirs=CAMPAIGN_DIRS):
    return [
        classify_existing.DirObservation(name=name, has_final_results=complete)
        for name, complete in sorted(dirs.items())
    ]


def interaction_pairs(edges=INTERACTIONS):
    return frozenset((drug, accession) for drug, _, accession, _ in edges)


def classify(dirs=CAMPAIGN_DIRS, proteins=PROTEINS, edges=INTERACTIONS, **kwargs):
    return classify_existing.classify_jobs(
        observations(dirs),
        index=make_index(proteins),
        interaction_pairs=interaction_pairs(edges),
        **kwargs
    )


def row_for(frame, old_job_dir):
    matches = frame[frame["old_job_dir"] == old_job_dir]
    assert len(matches) == 1, "{0} appears {1} times".format(old_job_dir, len(matches))
    return matches.iloc[0]


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def snapshot_tree(root):
    """Every path under ``root`` with its size and mtime -- the read-only proof."""
    state = {}
    for directory, subdirs, files in os.walk(root):
        subdirs.sort()
        for name in sorted(files) + sorted(subdirs):
            path = os.path.join(directory, name)
            info = os.stat(path)
            state[os.path.relpath(path, root)] = (info.st_size, info.st_mtime_ns)
    return state


class ScanFixture:
    """A complete Phase-2 input set in a temporary directory."""

    def __init__(self, root, dirs=CAMPAIGN_DIRS, proteins=PROTEINS, edges=INTERACTIONS):
        self.root = root
        self.old = build_old_tree(root, dirs)
        self.index = write_index(
            os.path.join(root, "new", "identity", "identity_index.parquet"), proteins
        )
        self.manifests = os.path.join(root, "new", "manifests")
        self.interactions = write_interactions(
            os.path.join(self.manifests, "interactions_resolved.csv"), edges
        )
        self.out_map = os.path.join(self.manifests, "job_classification.csv")
        self.out_pool = os.path.join(self.manifests, "negatives_pool.csv")
        self.out_summary = os.path.join(self.manifests, "classify_existing_summary.json")
        self.out_evidence = os.path.join(self.manifests, "job_classification.evidence.csv")

    def argv(self, *extra):
        return [
            "--existing-results", self.old,
            "--index", self.index,
            "--interactions", self.interactions,
            "--out-map", self.out_map,
            "--out-pool", self.out_pool,
            "--out-summary", self.out_summary,
            "--log-level", "ERROR",
        ] + list(extra)

    def run(self, *extra, **kwargs):
        expect = kwargs.pop("expect", 0)
        assert not kwargs, kwargs
        code = classify_existing.main(self.argv(*extra))
        assert code == expect, "expected exit {0}, got {1}".format(expect, code)
        return code

    def read_map(self, allow_extra=False):
        """Strict by default: this is the contract curate_jobs.py reads with."""
        frame = schemas.read_checked(self.out_map, "job_classification", allow_extra=allow_extra)
        if os.path.exists(self.out_evidence) and not allow_extra:
            evidence = pd.read_csv(
                self.out_evidence, keep_default_na=False, na_values=[""]
            )
            frame = frame.merge(evidence, on="old_job_dir", how="left", validate="1:1")
        return frame

    def read_evidence(self):
        return pd.read_csv(self.out_evidence, keep_default_na=False, na_values=[""])

    def read_pool(self):
        return schemas.read_checked(self.out_pool, "negatives_pool")

    def read_summary(self):
        with open(self.out_summary) as handle:
            return json.load(handle)


class ReuseScanTestCase(unittest.TestCase):
    """Base class providing a temporary directory and a built fixture."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name

    def fixture(self, **kwargs):
        return ScanFixture(self.root, **kwargs)


# --------------------------------------------------------------------------
# listing and observation
# --------------------------------------------------------------------------


class TestListing(ReuseScanTestCase):
    def test_directories_and_non_directories_are_separated_and_sorted(self):
        old = build_old_tree(self.root)
        job_dirs, others = classify_existing.list_entries(old)
        self.assertEqual(job_dirs, sorted(CAMPAIGN_DIRS))
        self.assertEqual(others, [STRAY_ENTRY])
        # the campaign's own arithmetic: 24,174 listed = 24,173 dirs + 1 file
        self.assertEqual(
            len(job_dirs) + len(others), len(CAMPAIGN_DIRS) + 1
        )

    def test_missing_existing_results_raises_rather_than_scanning_nothing(self):
        with self.assertRaises(NotADirectoryError):
            classify_existing.list_entries(os.path.join(self.root, "does_not_exist"))

    def test_final_results_must_be_present_and_non_empty(self):
        old = build_old_tree(self.root)
        complete = classify_existing.observe_job_dir(old, "DB00035_AVPR2_P30518_cavity_1")
        self.assertTrue(complete.has_final_results)
        missing = classify_existing.observe_job_dir(old, "DB00035_AVPR1B_P47901_cavity_8")
        self.assertFalse(missing.has_final_results)
        # a zero-byte file is not a payload
        empty = os.path.join(old, "DB00035_AVPR1B_P47901_cavity_8", "final_results.csv")
        open(empty, "w").close()
        self.assertFalse(
            classify_existing.observe_job_dir(
                old, "DB00035_AVPR1B_P47901_cavity_8"
            ).has_final_results
        )


# --------------------------------------------------------------------------
# classification -- the four outcomes
# --------------------------------------------------------------------------


class TestClassification(ReuseScanTestCase):
    def setUp(self):
        super().setUp()
        self.frame = classify()

    def test_every_directory_produces_exactly_one_row(self):
        self.assertEqual(len(self.frame), len(CAMPAIGN_DIRS))
        self.assertEqual(
            sorted(self.frame["old_job_dir"].tolist()), sorted(CAMPAIGN_DIRS)
        )

    def test_true_positive(self):
        row = row_for(self.frame, "DB00035_AVPR2_P30518_cavity_1")
        self.assertEqual(row["classification"], classify_existing.TRUE_POSITIVE)
        self.assertEqual(row["accession"], "P30518")
        self.assertEqual(row["gene"], "AVPR2")
        self.assertTrue(row["is_known_interaction"])
        self.assertEqual(row["new_job_dir"], "DB00035__P30518__AVPR2__F1__cav1")
        self.assertTrue(pd.isna(row["excluded_reason"]))

    def test_valid_non_interaction_is_the_incident_row(self):
        # DB00035_AVPR2_Q8NCT1: labelled AVPR2, but Q8NCT1 is ARRDC4, and
        # DB00035 has no annotated interaction with ARRDC4.
        row = row_for(self.frame, "DB00035_AVPR2_Q8NCT1_cavity_2")
        self.assertEqual(row["classification"], classify_existing.VALID_NON_INTERACTION)
        self.assertEqual(row["accession"], "Q8NCT1")
        self.assertEqual(row["gene"], "ARRDC4")  # relabelled from the index
        self.assertEqual(row["old_gene_label"], "AVPR2")  # kept as evidence
        self.assertFalse(row["label_was_correct"])
        self.assertFalse(row["is_known_interaction"])
        self.assertTrue(pd.isna(row["new_job_dir"]))
        self.assertIn("negative-pool candidate", row["excluded_reason"])

    def test_incomplete_is_recorded_not_silently_skipped(self):
        row = row_for(self.frame, "DB00035_AVPR1B_P47901_cavity_8")
        self.assertEqual(row["classification"], classify_existing.INCOMPLETE)
        self.assertFalse(row["has_final_results"])
        # identity is still recovered, so the job is auditable
        self.assertEqual(row["accession"], "P47901")
        self.assertEqual(row["gene"], "AVPR1B")
        self.assertIn("final_results.csv", row["excluded_reason"])
        self.assertTrue(pd.isna(row["new_job_dir"]))

    def test_unparseable_is_recorded_with_a_reason(self):
        row = row_for(self.frame, "DB00035__P30518__AVPR2__F1__cav1")
        self.assertEqual(row["classification"], classify_existing.UNPARSEABLE)
        for column in ("drugbank_id", "accession", "gene", "fragment_id", "cavity_index"):
            self.assertTrue(pd.isna(row[column]), column)
        self.assertTrue(row["excluded_reason"].startswith("unparseable directory name"))
        self.assertIn("cavity", row["excluded_reason"])

    def test_an_accession_outside_the_index_is_unparseable_not_guessed(self):
        # The old regex trusted the third token positionally. Here the accession
        # set is the authority: an unknown token yields a recorded rejection.
        frame = classify(dirs={"DB00035_FOO_P99999_cavity_1": True})
        row = row_for(frame, "DB00035_FOO_P99999_cavity_1")
        self.assertEqual(row["classification"], classify_existing.UNPARSEABLE)
        self.assertIn("accession set is the authority", row["excluded_reason"])

    def test_classification_covers_all_four_outcomes(self):
        self.assertEqual(
            set(self.frame["classification"]),
            set(classify_existing.CLASSIFICATIONS),
        )

    def test_totals_reconcile_to_the_directory_count(self):
        counts = self.frame["classification"].value_counts()
        self.assertEqual(int(counts.sum()), len(CAMPAIGN_DIRS))
        self.assertEqual(len(self.frame), len(CAMPAIGN_DIRS))


class TestNameParsing(ReuseScanTestCase):
    """The 1,777 directories the old regex rejected."""

    def test_multi_fragment_legacy_name(self):
        row = row_for(classify(), "DB00091_CAMLG_P08F94_-F13-model_v1_cavity_10")
        self.assertEqual(row["accession"], "P08F94")
        self.assertEqual(row["fragment_id"], "F13")
        self.assertEqual(row["fragment_source"], classify_existing.FRAGMENT_FROM_DIR_NAME)
        self.assertEqual(row["cavity_index"], 10)
        self.assertEqual(row["old_gene_label"], "CAMLG")
        self.assertEqual(row["gene"], "PKHD1")  # P08F94 is PKHD1, not CAMLG
        self.assertFalse(row["label_was_correct"])

    def test_hyphenated_gene_symbol(self):
        row = row_for(classify(), "DB00157_MT-ND1_P03886_cavity_1")
        self.assertEqual(row["accession"], "P03886")
        self.assertEqual(row["old_gene_label"], "MT-ND1")
        self.assertEqual(row["gene"], "MT-ND1")
        self.assertTrue(row["label_was_correct"])
        self.assertEqual(row["classification"], classify_existing.TRUE_POSITIVE)
        self.assertEqual(row["new_job_dir"], "DB00157__P03886__MT-ND1__F1__cav1")

    def test_nan_gene_token_becomes_a_null_label_never_the_string_nan(self):
        row = row_for(classify(), "DB00035_nan_P08173_cavity_3")
        self.assertTrue(pd.isna(row["old_gene_label"]))
        self.assertEqual(row["gene"], "CHRM4")
        # 'was the old label right?' is unanswerable when there was no label
        self.assertTrue(pd.isna(row["label_was_correct"]))

    def test_label_was_correct_is_null_only_when_there_was_no_label(self):
        frame = classify()
        labelled = frame["old_gene_label"].notna()
        parsed = frame["classification"] != classify_existing.UNPARSEABLE
        self.assertTrue(frame.loc[labelled & parsed, "label_was_correct"].notna().all())
        self.assertTrue(frame.loc[~labelled, "label_was_correct"].isna().all())

    def test_missing_fragment_token_uses_the_explicit_default_and_says_so(self):
        frame = classify()
        row = row_for(frame, "DB00035_AVPR2_P30518_cavity_1")
        self.assertEqual(row["fragment_id"], "F1")
        self.assertEqual(row["fragment_source"], classify_existing.FRAGMENT_FROM_DEFAULT)

        other = classify(default_fragment="F7")
        self.assertEqual(
            row_for(other, "DB00035_AVPR2_P30518_cavity_1")["fragment_id"], "F7"
        )
        # a name that carries its own fragment ignores the default
        self.assertEqual(
            row_for(other, "DB00091_CAMLG_P08F94_-F13-model_v1_cavity_10")["fragment_id"],
            "F13",
        )

    def test_a_fragment_may_not_be_silently_absent(self):
        from dock_selectivity.jobid import JobIdError

        with self.assertRaises(JobIdError):
            classify(default_fragment="")


# --------------------------------------------------------------------------
# the failure mode this rebuild exists for
# --------------------------------------------------------------------------


class TestNoIdentityCollapse(ReuseScanTestCase):
    """A gene symbol may never select a protein, and may never collapse two."""

    def test_one_corrupt_gene_token_on_two_accessions_yields_two_independent_rows(self):
        frame = classify()
        p68363 = row_for(frame, "DB00570_TUBA1B_P68363_cavity_1")
        q13509 = row_for(frame, "DB00570_TUBA1B_Q13509_cavity_1")
        # identical old gene token...
        self.assertEqual(p68363["old_gene_label"], q13509["old_gene_label"])
        # ...different proteins, different corrected labels, different verdicts
        self.assertEqual(p68363["gene"], "TUBA1B")
        self.assertEqual(q13509["gene"], "TUBB3")
        self.assertEqual(p68363["classification"], classify_existing.TRUE_POSITIVE)
        self.assertEqual(q13509["classification"], classify_existing.VALID_NON_INTERACTION)

    def test_directory_to_accession_has_collapse_factor_one(self):
        frame = classify()
        parsed = frame[frame["classification"] != classify_existing.UNPARSEABLE]
        report = assert_collapse_factor_one(
            parsed[["old_job_dir", "accession"]], name="test old_job_dir -> accession"
        )
        self.assertEqual(report["max_collapse"], 1)

    def test_the_old_gene_keyed_mapping_would_have_collapsed(self):
        # Proof the check above is meaningful and not vacuous: the mapping the
        # 2025 pipeline actually used (gene symbol -> accession) collapses on
        # this very fixture, exactly as dict(zip(Mapped_Gene_Name, Entry)) did.
        frame = classify()
        labelled = frame[frame["old_gene_label"].notna()]
        with self.assertRaises(InvariantViolation) as caught:
            assert_collapse_factor_one(
                labelled[["old_gene_label", "accession"]],
                name="test old_gene_label -> accession",
            )
        self.assertIn("many-to-one", str(caught.exception))
        self.assertIn("TUBA1B", str(caught.exception))

    def test_gene_is_always_derived_from_the_accession(self):
        frame = classify()
        index = make_index()
        parsed = frame[frame["classification"] != classify_existing.UNPARSEABLE]
        for _, row in parsed.iterrows():
            self.assertEqual(row["gene"], index.label(row["accession"]))
            self.assertTrue(index.by_accession(row["accession"]).has_symbol(row["gene"]))

    def test_new_job_dir_carries_the_true_symbol_not_the_old_token(self):
        edges = INTERACTIONS + (("DB00035", 91947, "Q8NCT1", "ARRDC4"),)
        frame = classify(edges=edges)
        row = row_for(frame, "DB00035_AVPR2_Q8NCT1_cavity_2")
        self.assertEqual(row["classification"], classify_existing.TRUE_POSITIVE)
        self.assertEqual(row["new_job_dir"], "DB00035__Q8NCT1__ARRDC4__F1__cav2")
        self.assertNotIn("AVPR2", row["new_job_dir"])

    def test_an_unnameable_job_is_recorded_with_a_reason_not_dropped(self):
        # A symbol that cannot go in a directory name (whitespace) must not
        # silently produce a nameless carried-over job.
        proteins = (Protein("P30518", "V2R_HUMAN", "AVPR 2", ("AVPR 2",), 554, True),)
        frame = classify(
            dirs={"DB00035_AVPR2_P30518_cavity_1": True},
            proteins=proteins,
            edges=(("DB00035", 554, "P30518", "AVPR2"),),
        )
        row = row_for(frame, "DB00035_AVPR2_P30518_cavity_1")
        self.assertEqual(row["classification"], classify_existing.TRUE_POSITIVE)
        self.assertTrue(pd.isna(row["new_job_dir"]))
        self.assertIn("cannot build a new job directory name", row["excluded_reason"])


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


class TestEndToEnd(ReuseScanTestCase):
    def setUp(self):
        super().setUp()
        self.fx = self.fixture()
        self.fx.run()

    def test_job_classification_satisfies_its_declared_contract(self):
        frame = self.fx.read_map()  # read_checked validates on read
        self.assertEqual(len(frame), len(CAMPAIGN_DIRS))
        self.assertEqual(frame["old_job_dir"].nunique(), len(CAMPAIGN_DIRS))

    def test_every_row_either_names_its_new_directory_or_gives_a_reason(self):
        frame = self.fx.read_map()
        carried = frame["new_job_dir"].notna()
        excluded = frame["excluded_reason"].notna()
        self.assertTrue(bool((carried ^ excluded).all()))
        self.assertFalse(bool((carried & excluded).any()))

    def test_negatives_pool_is_exactly_the_valid_non_interactions(self):
        frame = self.fx.read_map()
        pool = self.fx.read_pool()
        expected = frame.loc[
            frame["classification"] == classify_existing.VALID_NON_INTERACTION,
            "old_job_dir",
        ]
        self.assertEqual(sorted(pool["old_job_dir"]), sorted(expected))
        self.assertEqual(set(pool["classification"]), {"valid_non_interaction"})
        self.assertTrue(bool(pool["has_final_results"].all()))
        # the pool is a manifest the sampler reads: no nulls in its keys
        for column in ("drugbank_id", "accession", "gene", "fragment_id", "cavity_index"):
            self.assertEqual(int(pool[column].isna().sum()), 0, column)

    def test_summary_accounts_for_every_listed_entry(self):
        summary = self.fx.read_summary()
        counts = summary["counts"]
        self.assertEqual(counts["n_directories"], len(CAMPAIGN_DIRS))
        self.assertEqual(counts["n_non_directory_entries"], 1)
        self.assertEqual(counts["n_entries_listed"], len(CAMPAIGN_DIRS) + 1)
        self.assertEqual(counts["n_rows"], len(CAMPAIGN_DIRS))
        self.assertEqual(sum(counts["by_classification"].values()), len(CAMPAIGN_DIRS))
        self.assertEqual(summary["non_directory_entries"], [STRAY_ENTRY])
        self.assertEqual(counts["n_carried"] + counts["n_excluded"], len(CAMPAIGN_DIRS))

    def test_summary_records_the_total_accounting_invariant(self):
        summary = self.fx.read_summary()
        accounting = [
            report
            for report in summary["invariants"]
            if report["check"] == "total_accounting"
        ]
        self.assertEqual(len(accounting), 1)
        self.assertTrue(accounting[0]["ok"])
        self.assertEqual(accounting[0]["n_in"], len(CAMPAIGN_DIRS))
        self.assertEqual(
            accounting[0]["n_resolved"] + accounting[0]["n_rejected"],
            len(CAMPAIGN_DIRS),
        )
        self.assertTrue(all(report["ok"] for report in summary["invariants"]))

    def test_summary_names_every_unparseable_directory_with_its_reason(self):
        summary = self.fx.read_summary()
        self.assertEqual(len(summary["unparseable"]), 1)
        record = summary["unparseable"][0]
        self.assertEqual(record["old_job_dir"], "DB00035__P30518__AVPR2__F1__cav1")
        self.assertTrue(record["reason"].strip())

    def test_summary_pins_the_identity_index_and_the_interaction_set(self):
        summary = self.fx.read_summary()
        self.assertEqual(summary["identity_index"]["sha256"], sha256(self.fx.index))
        self.assertEqual(summary["identity_index"]["n_accessions"], len(PROTEINS))
        self.assertEqual(summary["interactions"]["sha256"], sha256(self.fx.interactions))
        self.assertEqual(summary["interactions"]["n_pairs"], len(INTERACTIONS))
        self.assertEqual(summary["interactions"]["n_accessions_not_in_index"], 0)

    def test_expected_classification_counts(self):
        counts = self.fx.read_summary()["counts"]["by_classification"]
        self.assertEqual(
            counts,
            {
                "true_positive": 3,
                "valid_non_interaction": 4,
                "incomplete": 1,
                "unparseable": 1,
            },
        )


# --------------------------------------------------------------------------
# determinism and idempotence
# --------------------------------------------------------------------------


class TestDeterminism(ReuseScanTestCase):
    def test_rerunning_produces_byte_identical_manifests(self):
        fx = self.fixture()
        fx.run()
        first = (sha256(fx.out_map), sha256(fx.out_pool), sha256(fx.out_evidence))
        fx.run()
        self.assertEqual(
            (sha256(fx.out_map), sha256(fx.out_pool), sha256(fx.out_evidence)), first
        )

    def test_the_summary_is_byte_identical_for_an_identical_run(self):
        fx = self.fixture()
        fx.run("--no-checkpoint")
        first = sha256(fx.out_summary)
        fx.run("--no-checkpoint")
        self.assertEqual(sha256(fx.out_summary), first)

    def test_output_does_not_depend_on_pythonhashseed(self):
        # prepare_uniprot_mapping.py:51 did list(set(...)); the resulting order
        # decided which protein owned an accession, invisibly, per interpreter.
        fx = self.fixture()
        snippet = (
            "import hashlib, sys;"
            "sys.path.insert(0, {repo!r});"
            "import classify_existing;"
            "rc = classify_existing.main({argv!r});"
            "assert rc == 0, rc;"
            "print(hashlib.sha256(open({m!r},'rb').read()).hexdigest());"
            "print(hashlib.sha256(open({p!r},'rb').read()).hexdigest());"
            "print(hashlib.sha256(open({e!r},'rb').read()).hexdigest());"
            "print(hashlib.sha256(open({s!r},'rb').read()).hexdigest())"
        ).format(
            repo=REPO_ROOT,
            argv=fx.argv("--no-checkpoint"),
            m=fx.out_map,
            p=fx.out_pool,
            e=fx.out_evidence,
            s=fx.out_summary,
        )
        report = assert_deterministic_across_hash_seeds(
            snippet, name="classify_existing outputs", seeds=(0, 1, 42), timeout=300
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["n_distinct_outputs"], 1)

    def test_rows_are_ordered_by_directory_name(self):
        fx = self.fixture()
        fx.run()
        frame = fx.read_map()
        self.assertEqual(
            frame["old_job_dir"].tolist(), sorted(frame["old_job_dir"].tolist())
        )


# --------------------------------------------------------------------------
# the old directory is read-only
# --------------------------------------------------------------------------


class TestReadOnly(ReuseScanTestCase):
    def test_the_old_tree_is_byte_for_byte_unchanged(self):
        fx = self.fixture()
        before = snapshot_tree(fx.old)
        fx.run()
        self.assertEqual(snapshot_tree(fx.old), before)
        self.assertTrue(before)  # the snapshot is not vacuously empty

    def test_writing_a_manifest_into_the_old_tree_is_refused(self):
        fx = self.fixture()
        target = os.path.join(fx.old, "job_classification.csv")
        before = snapshot_tree(fx.old)
        code = classify_existing.main(
            [
                "--existing-results", fx.old,
                "--index", fx.index,
                "--interactions", fx.interactions,
                "--out-map", target,
                "--out-pool", fx.out_pool,
                "--log-level", "ERROR",
            ]
        )
        self.assertEqual(code, 1)
        self.assertFalse(os.path.exists(target))
        self.assertEqual(snapshot_tree(fx.old), before)

    def test_an_evidence_file_inside_the_old_tree_is_refused(self):
        fx = self.fixture()
        before = snapshot_tree(fx.old)
        with self.assertRaises(InvariantViolation):
            classify_existing.run_scan(
                existing_results=fx.old,
                index_path=fx.index,
                interactions_path=fx.interactions,
                out_map=fx.out_map,
                out_pool=fx.out_pool,
                out_evidence=os.path.join(fx.old, "evidence.csv"),
            )
        self.assertEqual(snapshot_tree(fx.old), before)

    def test_a_checkpoint_inside_the_old_tree_is_refused(self):
        fx = self.fixture()
        with self.assertRaises(InvariantViolation):
            classify_existing.run_scan(
                existing_results=fx.old,
                index_path=fx.index,
                interactions_path=fx.interactions,
                out_map=fx.out_map,
                out_pool=fx.out_pool,
                checkpoint=os.path.join(fx.old, "checkpoint.jsonl"),
            )

    def test_the_guard_accepts_a_sibling_directory(self):
        fx = self.fixture()
        sibling = os.path.join(os.path.dirname(fx.old), "consensus_docking_results_new")
        self.assertEqual(
            classify_existing.assert_not_under(
                os.path.join(sibling, "map.csv"), fx.old, "--out-map"
            ),
            os.path.realpath(os.path.join(sibling, "map.csv")),
        )

    def test_out_map_and_out_pool_may_not_be_the_same_file(self):
        fx = self.fixture()
        with self.assertRaises(InvariantViolation):
            classify_existing.run_scan(
                existing_results=fx.old,
                index_path=fx.index,
                interactions_path=fx.interactions,
                out_map=fx.out_map,
                out_pool=fx.out_map,
            )


# --------------------------------------------------------------------------
# resumability
# --------------------------------------------------------------------------


class TestResume(ReuseScanTestCase):
    def checkpoint_path(self, fx):
        return os.path.join(
            os.path.dirname(fx.out_map), classify_existing.DEFAULT_CHECKPOINT_NAME
        )

    def test_a_checkpoint_is_written_and_reused(self):
        fx = self.fixture()
        fx.run()
        path = self.checkpoint_path(fx)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(fx.read_summary()["counts"]["n_new_observations"], len(CAMPAIGN_DIRS))

        fx.run()
        summary = fx.read_summary()
        self.assertEqual(summary["counts"]["n_cached_observations"], len(CAMPAIGN_DIRS))
        self.assertEqual(summary["counts"]["n_new_observations"], 0)
        self.assertEqual(summary["checkpoint"]["n_entries"], len(CAMPAIGN_DIRS))

    def test_the_cached_observation_is_genuinely_reused(self):
        # Proof the cache is consulted rather than merely written: change the
        # tree without touching the checkpoint and the answer must not move.
        fx = self.fixture()
        fx.run()
        os.remove(
            os.path.join(
                fx.old, "DB00035_AVPR2_P30518_cavity_1", classify_existing.FINAL_RESULTS_FILENAME
            )
        )
        fx.run()
        self.assertEqual(
            row_for(fx.read_map(), "DB00035_AVPR2_P30518_cavity_1")["classification"],
            classify_existing.TRUE_POSITIVE,
        )
        # ...and --no-resume re-observes it
        fx.run("--no-resume")
        self.assertEqual(
            row_for(fx.read_map(), "DB00035_AVPR2_P30518_cavity_1")["classification"],
            classify_existing.INCOMPLETE,
        )

    def test_an_interrupted_scan_resumes_from_the_checkpoint(self):
        fx = self.fixture()
        names = sorted(CAMPAIGN_DIRS)[:4]
        path = self.checkpoint_path(fx)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write(
                json.dumps(
                    {
                        "tool": classify_existing.CHECKPOINT_TOOL,
                        "schema_version": classify_existing.CHECKPOINT_SCHEMA_VERSION,
                        "existing_results": os.path.realpath(fx.old),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            for name in names:
                handle.write(json.dumps({"d": name, "f": 1}) + "\n")
        fx.run()
        counts = fx.read_summary()["counts"]
        self.assertEqual(counts["n_cached_observations"], len(names))
        self.assertEqual(counts["n_new_observations"], len(CAMPAIGN_DIRS) - len(names))

    def test_a_checkpoint_from_another_tree_raises_instead_of_being_reused(self):
        fx = self.fixture()
        path = self.checkpoint_path(fx)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write(
                json.dumps(
                    {
                        "tool": classify_existing.CHECKPOINT_TOOL,
                        "schema_version": classify_existing.CHECKPOINT_SCHEMA_VERSION,
                        "existing_results": "/media/onur/Elements/some_other_campaign",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            handle.write(json.dumps({"d": "DB99999_X_P30518_cavity_1", "f": 1}) + "\n")
        with self.assertRaises(InvariantViolation) as caught:
            classify_existing.read_checkpoint(path, fx.old)
        self.assertIn("--no-resume", str(caught.exception))
        # the CLI reports it and exits non-zero rather than scanning anyway
        self.assertEqual(fx.run(expect=1), 1)

    def test_a_truncated_final_record_is_tolerated_and_counted(self):
        fx = self.fixture()
        fx.run()
        path = self.checkpoint_path(fx)
        with open(path) as handle:
            text = handle.read()
        # a process killed mid-write leaves a partial final record
        with open(path, "w") as handle:
            handle.write(text[: text.rstrip("\n").rfind("\n") + 12])
        cached, report = classify_existing.read_checkpoint(path, fx.old)
        self.assertEqual(report["n_truncated_lines"], 1)
        self.assertEqual(len(cached), len(CAMPAIGN_DIRS) - 1)
        fx.run()  # and the run completes, re-observing the lost directory
        self.assertEqual(len(fx.read_map()), len(CAMPAIGN_DIRS))

    def test_a_corrupt_middle_record_raises(self):
        fx = self.fixture()
        fx.run()
        path = self.checkpoint_path(fx)
        with open(path) as handle:
            lines = handle.read().splitlines()
        lines[1] = "{not json"
        with open(path, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        with self.assertRaises(InvariantViolation):
            classify_existing.read_checkpoint(path, fx.old)

    def test_no_checkpoint_writes_none(self):
        fx = self.fixture()
        fx.run("--no-checkpoint")
        self.assertFalse(os.path.exists(self.checkpoint_path(fx)))
        self.assertIsNone(fx.read_summary()["checkpoint"]["path"])


# --------------------------------------------------------------------------
# fail closed: a missing prerequisite is an error, never a fallback
# --------------------------------------------------------------------------


class TestFailClosed(ReuseScanTestCase):
    def test_a_missing_identity_index_is_not_substituted(self):
        fx = self.fixture()
        os.remove(fx.index)
        self.assertEqual(fx.run(expect=1), 1)
        self.assertFalse(os.path.exists(fx.out_map))

    def test_a_missing_interaction_manifest_is_not_substituted(self):
        fx = self.fixture()
        os.remove(fx.interactions)
        self.assertEqual(fx.run(expect=1), 1)
        self.assertFalse(os.path.exists(fx.out_map))

    def test_an_index_sha_mismatch_refuses_to_run(self):
        fx = self.fixture()
        self.assertEqual(fx.run("--index-sha256", "0" * 64, expect=1), 1)
        self.assertFalse(os.path.exists(fx.out_map))
        # the real digest is accepted
        self.assertEqual(fx.run("--index-sha256", sha256(fx.index)), 0)

    def test_both_invariant_hierarchies_are_caught_at_the_cli_boundary(self):
        # ``identity.py`` used to declare a *second* InvariantViolation, so a
        # stage catching only one of the two names let half its guard rails
        # escape as a traceback.  ``invariants`` is now the single canonical
        # home and ``identity`` re-exports it, so one ``except`` covers the
        # package.  Both assertions below must hold: the names are one class,
        # and every error type the stage can raise is in FATAL_ERRORS.  If
        # anyone re-declares a private InvariantViolation, this fails.
        self.assertIs(
            IdentityInvariantViolation,
            InvariantViolation,
            "dock_selectivity.identity must re-export invariants."
            "InvariantViolation, not declare its own",
        )
        self.assertIn(InvariantViolation, classify_existing.FATAL_ERRORS)
        self.assertIn(IdentityInvariantViolation, classify_existing.FATAL_ERRORS)
        self.assertTrue(issubclass(schemas.SchemaViolation, InvariantViolation))
        self.assertTrue(issubclass(schemas.SchemaViolation, IdentityInvariantViolation))

    def test_interactions_naming_an_unknown_accession_raise(self):
        # The edge list and the index must be the same universe; otherwise
        # every count below silently inherits the disagreement.
        fx = self.fixture(edges=INTERACTIONS + (("DB00099", 1, "P99999", "FOO"),))
        self.assertEqual(fx.run(expect=1), 1)
        with self.assertRaises(InvariantViolation) as caught:
            classify_existing.load_interaction_pairs(fx.interactions, index=make_index())
        self.assertIn("P99999", str(caught.exception))

    def test_an_interaction_manifest_missing_a_column_is_rejected(self):
        fx = self.fixture()
        frame = pd.read_csv(fx.interactions).drop(columns=["gene"])
        frame.to_csv(fx.interactions, index=False)
        self.assertEqual(fx.run(expect=1), 1)

    def test_a_missing_existing_results_directory_exits_non_zero(self):
        fx = self.fixture()
        code = classify_existing.main(
            [
                "--existing-results", os.path.join(self.root, "nope"),
                "--index", fx.index,
                "--interactions", fx.interactions,
                "--out-map", fx.out_map,
                "--out-pool", fx.out_pool,
                "--log-level", "ERROR",
            ]
        )
        self.assertEqual(code, 1)


# --------------------------------------------------------------------------
# old_sample_type: evidence only, and the F1-vs-1 merge trap
# --------------------------------------------------------------------------


def write_old_plan(path, rows):
    """The campaign's required_structures_with_negatives.csv."""
    frame = pd.DataFrame(
        rows,
        columns=["UniProt_ID", "Fragment_ID", "Cavity_Index", "Gene_Name", "sample_type", "drugbank_id"],
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frame.to_csv(path, index=False)
    return path


PLAN_ROWS = [
    # positives were written with 'F1'...
    ("P30518", "F1", 1, "AVPR2", "positive", "DB00035"),
    ("Q8NCT1", "F1", 2, "AVPR2", "positive", "DB00035"),
    ("P68363", "F1", 1, "TUBA1B", "positive", "DB00570"),
    ("Q13509", "F1", 1, "TUBA1B", "positive", "DB00570"),
    ("P03886", "F1", 1, "MT-ND1", "positive", "DB00157"),
    ("P08F94", "F13", 10, "CAMLG", "positive", "DB00091"),
    ("P47901", "F1", 8, "AVPR1B", "positive", "DB00035"),
    # ...negatives with a bare integer, which is why the old merge matched neither
    ("P08173", 1, 3, "", "negative", "DB00035"),
]


class TestOldSampleType(ReuseScanTestCase):
    def test_without_a_plan_the_gene_token_rule_is_used_and_recorded(self):
        fx = self.fixture()
        fx.run()
        frame = fx.read_map()
        nan_row = row_for(frame, "DB00035_nan_P08173_cavity_3")
        self.assertEqual(nan_row["old_sample_type"], "negative")
        self.assertEqual(
            nan_row["old_sample_type_source"], classify_existing.SOURCE_GENE_TOKEN
        )
        labelled = row_for(frame, "DB00035_AVPR2_Q8NCT1_cavity_2")
        self.assertEqual(labelled["old_sample_type"], "positive")

    def test_the_plan_supplies_the_label_despite_the_F1_vs_1_mismatch(self):
        fx = self.fixture()
        plan = write_old_plan(os.path.join(self.root, "plan.csv"), PLAN_ROWS)
        fx.run("--old-plan", plan)
        frame = fx.read_map()
        # the negative row's Fragment_ID is the bare integer 1 in the plan
        row = row_for(frame, "DB00035_nan_P08173_cavity_3")
        self.assertEqual(row["old_sample_type"], "negative")
        self.assertEqual(row["old_sample_type_source"], classify_existing.SOURCE_OLD_PLAN)
        self.assertEqual(
            row_for(frame, "DB00091_CAMLG_P08F94_-F13-model_v1_cavity_10")[
                "old_sample_type_source"
            ],
            classify_existing.SOURCE_OLD_PLAN,
        )

    def test_the_plan_merge_does_not_change_the_row_count(self):
        # parse_prepare.py:1171 merged on a key missing Fragment_ID and added
        # 410,087 rows. Here an exactly duplicated plan row must not fan out.
        fx = self.fixture()
        plan = write_old_plan(
            os.path.join(self.root, "plan.csv"), PLAN_ROWS + [PLAN_ROWS[0]]
        )
        fx.run("--old-plan", plan)
        frame = fx.read_map()
        self.assertEqual(len(frame), len(CAMPAIGN_DIRS))
        summary = fx.read_summary()
        merge = [r for r in summary["invariants"] if r["check"] == "checked_merge"]
        self.assertEqual(len(merge), 1)
        self.assertEqual(merge[0]["n_out"], len(CAMPAIGN_DIRS))
        self.assertEqual(merge[0]["row_delta"], 0)
        self.assertTrue(merge[0]["ok"])

    def test_a_plan_key_claiming_both_labels_raises_instead_of_last_wins(self):
        fx = self.fixture()
        contradiction = ("P30518", "F1", 1, "AVPR2", "negative", "DB00035")
        plan = write_old_plan(
            os.path.join(self.root, "plan.csv"), PLAN_ROWS + [contradiction]
        )
        with self.assertRaises(InvariantViolation) as caught:
            classify_existing.load_old_plan(plan)
        self.assertIn("many-to-one", str(caught.exception))
        self.assertEqual(fx.run("--old-plan", plan, expect=1), 1)

    def test_a_plan_with_an_unknown_sample_type_is_not_coerced(self):
        plan = write_old_plan(
            os.path.join(self.root, "plan.csv"),
            PLAN_ROWS + [("P30518", "F1", 4, "AVPR2", "decoy", "DB00035")],
        )
        with self.assertRaises(InvariantViolation) as caught:
            classify_existing.load_old_plan(plan)
        self.assertIn("decoy", str(caught.exception))

    def test_a_plan_missing_a_required_column_is_rejected(self):
        path = os.path.join(self.root, "bad_plan.csv")
        pd.DataFrame({"UniProt_ID": ["P30518"]}).to_csv(path, index=False)
        with self.assertRaises(InvariantViolation):
            classify_existing.load_old_plan(path)

    def test_a_missing_plan_file_raises_rather_than_falling_back(self):
        with self.assertRaises(FileNotFoundError):
            classify_existing.load_old_plan(os.path.join(self.root, "absent.csv"))

    def test_the_old_label_never_decides_the_corrected_classification(self):
        # The plan calls DB00035 x P30518 a positive and DB00035 x Q8NCT1 a
        # positive too; the corrected answer depends only on the edge list.
        fx = self.fixture()
        plan = write_old_plan(os.path.join(self.root, "plan.csv"), PLAN_ROWS)
        fx.run("--old-plan", plan)
        with_plan = fx.read_map().set_index("old_job_dir")["classification"]
        fx.run("--no-resume")
        without_plan = fx.read_map().set_index("old_job_dir")["classification"]
        self.assertTrue(with_plan.equals(without_plan))
        # and both disagree with the old labels, which is the whole point
        frame = fx.read_map()
        self.assertEqual(
            row_for(frame, "DB00035_AVPR2_Q8NCT1_cavity_2")["classification"],
            classify_existing.VALID_NON_INTERACTION,
        )

    def test_a_plan_row_with_a_null_key_cannot_label_an_unparseable_job(self):
        # pandas matches NaN to NaN: a plan row with a blank fragment would
        # otherwise join to the all-null key of an unparseable job row.
        fx = self.fixture()
        plan = write_old_plan(
            os.path.join(self.root, "plan.csv"),
            PLAN_ROWS
            + [
                ("P30518", "", 1, "AVPR2", "negative", ""),  # no fragment, no drug
                ("P30518", "F1", "", "AVPR2", "negative", "DB00035"),  # no cavity
            ],
        )
        fx.run("--old-plan", plan)
        frame = fx.read_map()
        self.assertEqual(len(frame), len(CAMPAIGN_DIRS))
        unparseable = row_for(frame, "DB00035__P30518__AVPR2__F1__cav1")
        self.assertTrue(pd.isna(unparseable["old_sample_type"]))
        self.assertEqual(
            unparseable["old_sample_type_source"], classify_existing.SOURCE_UNPARSEABLE
        )
        self.assertEqual(fx.read_summary()["old_plan"]["n_dropped_null_key"], 2)

    def test_a_malformed_cavity_index_in_the_plan_raises(self):
        plan = write_old_plan(
            os.path.join(self.root, "plan.csv"),
            PLAN_ROWS + [("P30518", "F1", "cavity_two", "AVPR2", "negative", "DB00035")],
        )
        with self.assertRaises(InvariantViolation):
            classify_existing.load_old_plan(plan)


# --------------------------------------------------------------------------
# CLI surface
# --------------------------------------------------------------------------


class TestCommandLine(ReuseScanTestCase):
    def test_the_phase_2_command_from_the_rebuild_plan_runs_as_a_subprocess(self):
        fx = self.fixture()
        completed = subprocess.run(
            [
                sys.executable,
                os.path.join(REPO_ROOT, "classify_existing.py"),
                "--existing-results", fx.old,
                "--index", fx.index,
                "--interactions", fx.interactions,
                "--out-map", fx.out_map,
                "--out-pool", fx.out_pool,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.root,
            timeout=300,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8", "replace")[-2000:],
        )
        self.assertTrue(os.path.exists(fx.out_map))
        self.assertTrue(os.path.exists(fx.out_pool))
        self.assertTrue(
            os.path.exists(
                os.path.join(fx.manifests, classify_existing.DEFAULT_SUMMARY_NAME)
            )
        )

    def test_required_arguments_are_required(self):
        with self.assertRaises(SystemExit):
            classify_existing.build_parser().parse_args([])

    def test_job_classification_is_strictly_conformant_for_curate_jobs(self):
        # curate_jobs.py:1204 calls read_checked(path, "job_classification") with no
        # allow_extra. The default output must satisfy exactly that call.
        fx = self.fixture()
        fx.run()
        declared = schemas.get_schema("job_classification").column_names
        self.assertEqual(list(pd.read_csv(fx.out_map, nrows=0).columns), declared)
        strict = schemas.read_checked(fx.out_map, "job_classification")
        self.assertEqual(len(strict), len(CAMPAIGN_DIRS))

    def test_the_evidence_companion_carries_the_undeclared_columns_one_to_one(self):
        fx = self.fixture()
        fx.run()
        evidence = fx.read_evidence()
        self.assertEqual(
            list(evidence.columns), list(classify_existing.EVIDENCE_COLUMNS)
        )
        self.assertEqual(len(evidence), len(CAMPAIGN_DIRS))
        self.assertEqual(
            sorted(evidence["old_job_dir"]), sorted(fx.read_map()["old_job_dir"])
        )
        self.assertEqual(evidence["old_job_dir"].nunique(), len(CAMPAIGN_DIRS))
        self.assertEqual(
            set(classify_existing.EVIDENCE_COLUMNS)
            - set(schemas.get_schema("job_classification").column_names),
            set(classify_existing.JOB_CLASSIFICATION_EXTRA_COLUMNS),
        )

    def test_inline_extras_folds_them_in_and_then_a_strict_read_fails(self):
        fx = self.fixture()
        fx.run("--inline-extras")
        written = list(pd.read_csv(fx.out_map, nrows=0).columns)
        self.assertEqual(written, list(classify_existing.JOB_CLASSIFICATION_COLUMNS))
        self.assertFalse(os.path.exists(fx.out_evidence))
        with self.assertRaises(schemas.SchemaViolation):
            schemas.read_checked(fx.out_map, "job_classification")
        self.assertEqual(
            len(schemas.read_checked(fx.out_map, "job_classification", allow_extra=True)),
            len(CAMPAIGN_DIRS),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
