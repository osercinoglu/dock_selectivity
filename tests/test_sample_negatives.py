"""Tests for ``sample_negatives.py`` — balanced negatives in accession space.

Every fixture reproduces something the 2025-06-29 campaign actually did:

* ``uniprot_gene_mapping.csv`` collapsed 4,585 accessions onto 1,795 symbols, so
  ``TUBA1B`` named ``Q13509`` (TUBB3) as often as ``P68363``. Here both
  accessions exist, one of them carries the literal corrupt label, and the
  sampler must keep them apart.
* ``generate_negative_samples.py:303-319`` built its avoid-list in symbol space:
  only 50.11% of its 6,664 entries pointed at an accession that really carries
  that gene, and 5 "negatives" were known interactions.
* ``:561-566`` fell back to "any available target" once no target still needed
  negatives, silently abandoning the target-side quota.
* 13,452 negatives were written with a blank gene, producing 10,698 job
  directories named ``*_nan_*``.

Hermetic: everything is built in ``tempfile.TemporaryDirectory()``. No network,
no external drive, no ``/opt/data``. Real accessions and gene symbols appear as
literals, never as reads.

    python3 -m unittest discover -s /home/onur/repos/dock_selectivity/tests \\
        -t /home/onur/repos/dock_selectivity
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import sample_negatives as sn  # noqa: E402
from dock_selectivity.identity import IdentityIndex, Protein  # noqa: E402
from dock_selectivity.invariants import (  # noqa: E402
    InvariantViolation,
    assert_collapse_factor_one,
)
from dock_selectivity.jobid import parse_job_dir  # noqa: E402
from dock_selectivity.schemas import SchemaViolation, read_checked, write_checked  # noqa: E402

SCRIPT = os.path.join(REPO_ROOT, "sample_negatives.py")


def setUpModule() -> None:
    """Silence this step's logging without touching the root logger.

    This used to call ``logging.basicConfig``, on the strength of a comment that
    said ``main()`` did too. It no longer does -- nothing in the repo configures
    logging at import any more -- and the root handler that call installed was
    what made every other step's records reappear on the console.
    """
    logging.getLogger("sample_negatives").setLevel(logging.CRITICAL)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

#: (accession, entry_name, primary_symbol, symbols, entrez_id, reviewed).
#: Real values. ``P68363``/``Q13509`` are the pair the corrupt mapping merged
#: (KNOWN_ISSUE.md); ``P30518``/``Q8NCT1`` are the AVPR2 example from the
#: assessment; ``P03886`` carries the hyphenated symbol ``MT-ND1``.
DEFAULT_PROTEINS: Tuple[Tuple[Any, ...], ...] = (
    ("P30518", "V2R_HUMAN", "AVPR2", ("AVPR2",), 554, True),
    ("Q8NCT1", "AGMO_HUMAN", "AGMO", ("AGMO", "TMEM195"), 118881, True),
    ("P68363", "TBA1B_HUMAN", "TUBA1B", ("TUBA1B", "TUBA1B_HUMAN"), 10376, True),
    ("Q13509", "TBB3_HUMAN", "TUBB3", ("TUBB3", "TUBB4"), 10381, True),
    ("P03886", "NU1M_HUMAN", "MT-ND1", ("MT-ND1", "ND1"), 4535, True),
    ("P35348", "ADA1A_HUMAN", "ADRA1A", ("ADRA1A", "ADRA1C"), 148, True),
    ("P08913", "ADA2A_HUMAN", "ADRA2A", ("ADRA2A",), 150, True),
    ("P14416", "DRD2_HUMAN", "DRD2", ("DRD2",), 1813, True),
    ("P28223", "5HT2A_HUMAN", "HTR2A", ("HTR2A", "HTR2"), 3356, True),
    ("P08172", "ACM2_HUMAN", "CHRM2", ("CHRM2",), 1129, True),
    ("P35367", "HRH1_HUMAN", "HRH1", ("HRH1",), 3269, True),
    ("P23219", "PGH1_HUMAN", "PTGS1", ("PTGS1", "COX1"), 5742, True),
    ("P35354", "PGH2_HUMAN", "PTGS2", ("PTGS2", "COX2"), 5743, True),
    ("P04150", "GCR_HUMAN", "NR3C1", ("NR3C1", "GRL"), 2908, True),
    ("P10275", "ANDR_HUMAN", "AR", ("AR", "DHTR", "NR3C4"), 367, True),
    ("P06401", "PRGR_HUMAN", "PGR", ("PGR", "NR3C3"), 5241, True),
)

ENTREZ_BY_ACCESSION = {row[0]: row[4] for row in DEFAULT_PROTEINS}


def write_index(
    path: str,
    proteins: Sequence[Sequence[Any]] = DEFAULT_PROTEINS,
) -> IdentityIndex:
    """Write an ``identity_index.parquet`` and return the loaded index."""
    index = IdentityIndex([Protein(*row) for row in proteins])
    index.to_frame().to_parquet(path, index=False)
    return IdentityIndex.load(path)


def make_interactions(
    edges: Iterable[Tuple[str, str]],
    index: IdentityIndex,
) -> pd.DataFrame:
    """``interactions_resolved`` from ``(drugbank_id, accession)`` edges."""
    rows = []
    for drug, accession in edges:
        entrez = ENTREZ_BY_ACCESSION.get(accession)
        if entrez is None:
            protein = index.by_accession(accession)
            entrez = protein.entrez_id
        rows.append(
            {
                "drugbank_id": drug,
                "entrez_id": int(entrez),
                "accession": accession,
                "gene": index.label(accession),
                "selected_by": "entrez={0} via edge ({1},{0})".format(entrez, drug),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["drugbank_id", "entrez_id", "accession", "gene", "selected_by"],
    )


def make_positives(
    pairs: Iterable[Tuple[str, str]],
    cavities: Mapping[str, Sequence[Any]],
    index: IdentityIndex,
) -> pd.DataFrame:
    """One positive row per cavity, exactly as ``batch_dock.py`` expanded them.

    ``cavities`` maps accession -> cavity indices (fragment ``F1``) or explicit
    ``(fragment_id, cavity_index)`` tuples.
    """
    rows = []
    for drug, accession in pairs:
        for pocket in cavities[accession]:
            fragment, cavity = (
                pocket if isinstance(pocket, tuple) else ("F1", int(pocket))
            )
            rows.append(
                {
                    "drugbank_id": drug,
                    "accession": accession,
                    "gene": index.label(accession),
                    "fragment_id": fragment,
                    "cavity_index": int(cavity),
                    "sample_type": "positive",
                    "selected_by": "interaction ({0},{1})".format(drug, accession),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "drugbank_id",
            "accession",
            "gene",
            "fragment_id",
            "cavity_index",
            "sample_type",
            "selected_by",
        ],
    )


def pool_row(
    drug: str,
    accession: str,
    index: IdentityIndex,
    fragment: str = "F1",
    cavity: int = 1,
    old_job_dir: Optional[str] = None,
    gene: Optional[str] = None,
    old_gene_label: Optional[str] = None,
    old_sample_type: str = "negative",
    has_final_results: bool = True,
) -> Dict[str, Any]:
    """One ``negatives_pool`` row, named the way the old campaign named jobs."""
    label = index.label(accession) if accession in index else "?"
    return {
        "old_job_dir": old_job_dir
        or "{0}_{1}_{2}_cavity_{3}".format(drug, old_gene_label or label, accession, cavity),
        "drugbank_id": drug,
        "accession": accession,
        "gene": gene if gene is not None else label,
        "fragment_id": fragment,
        "cavity_index": int(cavity),
        "old_sample_type": old_sample_type,
        "old_gene_label": old_gene_label if old_gene_label is not None else label,
        "classification": "valid_non_interaction",
        "has_final_results": bool(has_final_results),
    }


POOL_COLUMNS = [
    "old_job_dir",
    "drugbank_id",
    "accession",
    "gene",
    "fragment_id",
    "cavity_index",
    "old_sample_type",
    "old_gene_label",
    "classification",
    "has_final_results",
]


def make_pool(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=POOL_COLUMNS)


class Fixture:
    """An input directory holding the four artifacts the sampler consumes."""

    def __init__(
        self,
        root: str,
        interactions: pd.DataFrame,
        positives: pd.DataFrame,
        pool: pd.DataFrame,
        index: IdentityIndex,
    ) -> None:
        self.root = root
        self.index = index
        self.index_path = index.source_path
        self.interactions_path = os.path.join(root, "interactions_resolved.csv")
        self.positives_path = os.path.join(root, "positives_planned.csv")
        self.pool_path = os.path.join(root, "negatives_pool.csv")
        write_checked(interactions, self.interactions_path, "interactions_resolved")
        write_checked(positives, self.positives_path, "positives_planned")
        write_checked(pool, self.pool_path, "negatives_pool")

    def argv(self, out_dir: str, *extra: str) -> List[str]:
        os.makedirs(out_dir, exist_ok=True)
        return [
            "--index", self.index_path,
            "--interactions", self.interactions_path,
            "--positives", self.positives_path,
            "--pool", self.pool_path,
            "--out", os.path.join(out_dir, "negatives_drawn.csv"),
            "--log-level", "ERROR",
        ] + list(extra)

    def frames(self):
        return (
            self.index,
            read_checked(self.interactions_path, "interactions_resolved"),
            read_checked(self.positives_path, "positives_planned"),
            read_checked(self.pool_path, "negatives_pool"),
        )

    def sample(self, **kwargs) -> sn.SamplingResult:
        index, interactions, positives, pool = self.frames()
        return sn.sample_negatives(
            index=index,
            interactions=interactions,
            positives=positives,
            pool=pool,
            **kwargs
        )


def build_fixture(
    root: str,
    edges: Iterable[Tuple[str, str]],
    positive_pairs: Iterable[Tuple[str, str]],
    cavities: Mapping[str, Sequence[Any]],
    pool_rows: Sequence[Mapping[str, Any]] = (),
    proteins: Sequence[Sequence[Any]] = DEFAULT_PROTEINS,
) -> Fixture:
    os.makedirs(root, exist_ok=True)
    index = write_index(os.path.join(root, "identity_index.parquet"), proteins)
    return Fixture(
        root,
        make_interactions(edges, index),
        make_positives(positive_pairs, cavities, index),
        make_pool(pool_rows),
        index,
    )


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_plain(path: str) -> pd.DataFrame:
    """Read a sidecar CSV without pandas' NA magic (a gene symbol may be 'NA')."""
    return pd.read_csv(path, keep_default_na=False, na_values=[])


class SamplerTestCase(unittest.TestCase):
    """Base class handing out a temp directory."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def out_dir(self, name: str = "out") -> str:
        path = os.path.join(self.tmp, name)
        os.makedirs(path, exist_ok=True)
        return path

    def drawn_at(self, out_dir: str) -> pd.DataFrame:
        return read_checked(
            os.path.join(out_dir, "negatives_drawn.csv"), "negatives_drawn"
        )


# --------------------------------------------------------------------------
# the constraint: zero positives -> zero negatives
# --------------------------------------------------------------------------


class TestStructuralEligibility(SamplerTestCase):
    """Balanced sampling makes ineligibility structural, not probabilistic."""

    def fixture(self) -> Fixture:
        # DB00001 binds AVPR2(P30518); DB00002 binds ADRA1A(P35348).
        # Q8NCT1 (AGMO) is docked in the pool but has NO positives — it is the
        # class of receptor the 2025 corruption selected: 523 of 979 docked
        # accessions were proteins nobody targets.
        return build_fixture(
            os.path.join(self.tmp, "in"),
            edges=[("DB00001", "P30518"), ("DB00002", "P35348")],
            positive_pairs=[("DB00001", "P30518"), ("DB00002", "P35348")],
            cavities={"P30518": [1], "P35348": [1], "Q8NCT1": [1]},
            pool_rows=[
                pool_row("DB00001", "Q8NCT1", self.index_of(), cavity=1),
                pool_row("DB00001", "P35348", self.index_of(), cavity=1),
                pool_row("DB00404", "P35348", self.index_of(), cavity=1),
            ],
        )

    def index_of(self) -> IdentityIndex:
        if not hasattr(self, "_index"):
            os.makedirs(os.path.join(self.tmp, "in"), exist_ok=True)
            self._index = write_index(
                os.path.join(self.tmp, "in", "identity_index.parquet")
            )
        return self._index

    def test_target_with_zero_positives_receives_zero_negatives(self) -> None:
        result = self.fixture().sample()
        drawn = result.drawn
        self.assertEqual(
            0,
            int((drawn["accession"] == "Q8NCT1").sum()),
            "a target with no positives must receive no negatives; the pool "
            "offered one and it must have been refused",
        )
        reasons = dict(
            zip(
                result.eligibility["old_job_dir"].tolist(),
                result.eligibility["eligibility"].tolist(),
            )
        )
        self.assertEqual(
            sn.REASON_TARGET_NO_POSITIVES,
            reasons["DB00001_AGMO_Q8NCT1_cavity_1"],
            "the refusal must be recorded with its reason, never silent",
        )

    def test_drug_with_zero_positives_receives_zero_negatives(self) -> None:
        result = self.fixture().sample()
        self.assertEqual(0, int((result.drawn["drugbank_id"] == "DB00404").sum()))
        reasons = dict(
            zip(
                result.eligibility["old_job_dir"].tolist(),
                result.eligibility["eligibility"].tolist(),
            )
        )
        self.assertEqual(
            sn.REASON_DRUG_NO_POSITIVES, reasons["DB00404_ADRA1A_P35348_cavity_1"]
        )

    def test_eligibility_reasons_reconcile_with_the_pool_size(self) -> None:
        fixture = self.fixture()
        result = fixture.sample()
        pool = read_checked(fixture.pool_path, "negatives_pool")

        self.assertEqual(len(pool), len(result.eligibility))
        self.assertEqual(
            sorted(pool["old_job_dir"].tolist()),
            sorted(result.eligibility["old_job_dir"].tolist()),
            "every pool row must appear exactly once in the eligibility ledger",
        )
        counts = result.report["pool"]["reasons"]
        self.assertEqual(
            len(pool),
            sum(counts.values()),
            "the reason counts must sum to the pool size: {0}".format(counts),
        )
        self.assertEqual(
            len(pool),
            result.report["pool"]["n_eligible"]
            + result.report["pool"]["n_ineligible"],
        )
        self.assertEqual({sn.ELIGIBLE: 1, sn.REASON_TARGET_NO_POSITIVES: 1,
                          sn.REASON_DRUG_NO_POSITIVES: 1},
                         {k: v for k, v in counts.items() if v})

    def test_ineligible_rows_are_written_out_not_dropped(self) -> None:
        fixture = self.fixture()
        out = self.out_dir()
        self.assertEqual(sn.EXIT_OK, sn.main(fixture.argv(out)))
        ledger = read_plain(os.path.join(out, "negatives_drawn.pool_eligibility.csv"))
        self.assertEqual(3, len(ledger))
        self.assertEqual({"True", "False"}, set(ledger["eligible"].astype(str)))
        self.assertTrue((ledger["eligibility"].astype(str).str.len() > 0).all())

    def test_only_eligible_pool_rows_can_be_reused(self) -> None:
        """Nothing the eligibility pass rejected may reach the output."""
        index = self.index_of()
        fixture = build_fixture(
            os.path.join(self.tmp, "in3"),
            edges=[("DB00001", "P30518"), ("DB00002", "P35348")],
            positive_pairs=[("DB00001", "P30518"), ("DB00002", "P35348")],
            cavities={"P30518": [1, 2], "P35348": [1, 2]},
            pool_rows=[
                pool_row("DB00001", "P35348", index, cavity=1),
                pool_row("DB00001", "P35348", index, cavity=2,
                         has_final_results=False),
                pool_row("DB00404", "P30518", index, cavity=1),
                pool_row("DB00002", "Q8NCT1", index, cavity=1),
            ],
        )
        result = fixture.sample(prefer_pool=True)
        eligible = set(
            result.eligibility.loc[result.eligibility["eligible"], "old_job_dir"]
        )
        reused = set(result.drawn["old_job_dir"].dropna())
        self.assertTrue(reused)
        self.assertTrue(
            reused <= eligible,
            "these reused jobs were marked ineligible: {0}".format(
                sorted(reused - eligible)
            ),
        )
        self.assertEqual(1, len(eligible))

    def test_accession_absent_from_the_index_is_recorded_not_dropped(self) -> None:
        index = self.index_of()
        fixture = build_fixture(
            os.path.join(self.tmp, "in2"),
            edges=[("DB00001", "P30518")],
            positive_pairs=[("DB00001", "P30518")],
            cavities={"P30518": [1]},
            pool_rows=[
                # Q9UNKN is not in the index at all. It must not crash the run
                # and it must not be drawn.
                dict(
                    pool_row("DB00001", "P35348", index, cavity=1),
                    old_job_dir="DB00001_XXX_Q9H0H0_cavity_1",
                    accession="Q9H0H0",
                    gene="ADRA1A",
                    old_gene_label="XXX",
                ),
            ],
        )
        result = fixture.sample()
        self.assertEqual(
            sn.REASON_NOT_IN_INDEX,
            result.eligibility["eligibility"].iloc[0],
        )
        self.assertEqual(0, int((result.drawn["accession"] == "Q9H0H0").sum()))


# --------------------------------------------------------------------------
# the avoid-list
# --------------------------------------------------------------------------


class TestAvoidList(SamplerTestCase):
    def test_drawn_negative_is_never_a_known_interaction(self) -> None:
        fixture = ring_fixture(os.path.join(self.tmp, "in"))
        result = fixture.sample(prefer_pool=True, seed=11)
        interactions = read_checked(fixture.interactions_path, "interactions_resolved")
        avoid = sn.build_avoid_list(interactions)
        drawn_pairs = set(
            zip(result.drawn["drugbank_id"].tolist(), result.drawn["accession"].tolist())
        )
        self.assertTrue(drawn_pairs)
        self.assertEqual(set(), drawn_pairs & avoid)
        self.assertEqual(
            0, result.report["contamination"]["n_drawn_known_interactions"]
        )

    def test_pool_entry_that_is_a_known_interaction_is_refused(self) -> None:
        """The 2025 negatives contained 5 real interactions. Zero here."""
        index = write_index(os.path.join(self.tmp, "identity_index.parquet"))
        fixture = build_fixture(
            os.path.join(self.tmp, "in"),
            edges=[("DB00001", "P30518"), ("DB00002", "P35348")],
            positive_pairs=[("DB00001", "P30518"), ("DB00002", "P35348")],
            cavities={"P30518": [1], "P35348": [1]},
            pool_rows=[pool_row("DB00001", "P30518", index, cavity=1)],
        )
        result = fixture.sample()
        self.assertEqual(
            sn.REASON_KNOWN_INTERACTION, result.eligibility["eligibility"].iloc[0]
        )
        self.assertEqual(
            0,
            int(
                (
                    (result.drawn["drugbank_id"] == "DB00001")
                    & (result.drawn["accession"] == "P30518")
                ).sum()
            ),
        )

    def test_contamination_outranks_the_axis_reason(self) -> None:
        """A contaminated row on a zero-positive target reports contamination."""
        index = write_index(os.path.join(self.tmp, "identity_index.parquet"))
        fixture = build_fixture(
            os.path.join(self.tmp, "in"),
            edges=[("DB00001", "P30518"), ("DB00001", "Q8NCT1"),
                   ("DB00002", "P35348")],
            positive_pairs=[("DB00001", "P30518"), ("DB00002", "P35348")],
            cavities={"P30518": [1], "P35348": [1], "Q8NCT1": [1]},
            pool_rows=[pool_row("DB00001", "Q8NCT1", index, cavity=1)],
        )
        result = fixture.sample()
        self.assertEqual(
            sn.REASON_KNOWN_INTERACTION, result.eligibility["eligibility"].iloc[0]
        )

    def test_avoid_list_covers_interactions_that_never_became_positives(self) -> None:
        """A real interaction with no cavity is still an interaction.

        ``interactions_resolved`` is the avoid-list source, not
        ``positives_planned``: an edge that never became a planned positive
        (no CavitySpace pocket, no ligand SDF) must still block the pair.
        """
        index = write_index(os.path.join(self.tmp, "identity_index.parquet"))
        fixture = build_fixture(
            os.path.join(self.tmp, "in"),
            # DB00002 also binds AVPR2, but AVPR2 positives exist only for
            # DB00001 — say DB00002's ligand file was missing at planning time.
            edges=[("DB00001", "P30518"), ("DB00002", "P35348"),
                   ("DB00002", "P30518")],
            positive_pairs=[("DB00001", "P30518"), ("DB00002", "P35348")],
            cavities={"P30518": [1], "P35348": [1]},
            pool_rows=[pool_row("DB00002", "P30518", index, cavity=1)],
        )
        result = fixture.sample()
        self.assertEqual(
            sn.REASON_KNOWN_INTERACTION, result.eligibility["eligibility"].iloc[0]
        )
        self.assertEqual(
            0,
            int(
                (
                    (result.drawn["drugbank_id"] == "DB00002")
                    & (result.drawn["accession"] == "P30518")
                ).sum()
            ),
        )

    def test_avoid_list_is_accession_space_not_symbol_space(self) -> None:
        """The corrupt label ``TUBA1B`` on Q13509 must not protect Q13509.

        ``uniprot_gene_mapping.csv`` labelled Q13509 (TUBB3) as ``TUBA1B``.
        A symbol-space avoid-list therefore blocked the wrong accession and
        left the right one exposed. Here the interaction is on P68363, and
        Q13509 — carrying the corrupt label in ``old_gene_label`` — is a
        perfectly valid negative.
        """
        index = write_index(os.path.join(self.tmp, "identity_index.parquet"))
        fixture = build_fixture(
            os.path.join(self.tmp, "in"),
            edges=[("DB00001", "P68363"), ("DB00002", "Q13509")],
            positive_pairs=[("DB00001", "P68363"), ("DB00002", "Q13509")],
            cavities={"P68363": [1], "Q13509": [1]},
            pool_rows=[
                dict(
                    pool_row("DB00001", "Q13509", index, cavity=1),
                    old_gene_label="TUBA1B",  # the corruption, preserved
                    old_job_dir="DB00001_TUBA1B_Q13509_cavity_1",
                )
            ],
        )
        result = fixture.sample(prefer_pool=True)
        drawn = result.drawn
        row = drawn[
            (drawn["drugbank_id"] == "DB00001") & (drawn["accession"] == "Q13509")
        ]
        self.assertEqual(1, len(row), "Q13509 is not a target of DB00001; it is drawable")
        self.assertEqual("pool", row["source"].iloc[0])
        self.assertEqual(
            "TUBB3",
            row["gene"].iloc[0],
            "the label must be re-derived from the index, never carried over",
        )
        # ...and P68363 (the real interaction) is nowhere in the negatives.
        self.assertEqual(
            0,
            int(
                (
                    (drawn["drugbank_id"] == "DB00001")
                    & (drawn["accession"] == "P68363")
                ).sum()
            ),
        )

    def test_two_accessions_the_corrupt_map_merged_stay_distinct(self) -> None:
        """Collapse factor exactly 1: one accession, one gene, no last-wins."""
        index = write_index(os.path.join(self.tmp, "identity_index.parquet"))
        fixture = build_fixture(
            os.path.join(self.tmp, "in"),
            edges=[("DB00001", "P68363"), ("DB00002", "Q13509"),
                   ("DB00003", "P35348")],
            positive_pairs=[("DB00001", "P68363"), ("DB00002", "Q13509"),
                            ("DB00003", "P35348")],
            cavities={"P68363": [1], "Q13509": [1], "P35348": [1]},
        )
        result = fixture.sample(seed=3)
        drawn = result.drawn
        self.assertTrue({"P68363", "Q13509"} <= set(drawn["accession"]))
        pairs = drawn[["accession", "gene"]].drop_duplicates()
        report = assert_collapse_factor_one(
            pairs, name="drawn accession -> gene", key="accession", value="gene"
        )
        self.assertEqual(1, report["max_collapse"])
        labels = dict(zip(pairs["accession"], pairs["gene"]))
        self.assertEqual("TUBA1B", labels["P68363"])
        self.assertEqual("TUBB3", labels["Q13509"])


# --------------------------------------------------------------------------
# balance
# --------------------------------------------------------------------------


def ring_fixture(root: str, n: int = 12) -> Fixture:
    """A satisfiable campaign: drug *i* binds targets *i* and *i+1*.

    Every target therefore has exactly two positive pairs and every drug two
    positive targets, so both quota vectors are well defined and the
    non-interaction complement is rich enough to balance exactly. Half the
    targets carry two cavities, so job-space quotas are not all equal — the
    cavity multiplicity that made the 2025 negative class 2.48x wider and 2.5x
    shallower than its positive class.
    """
    accessions = [row[0] for row in DEFAULT_PROTEINS][:n]
    drugs = ["DB{0:05d}".format(i + 1) for i in range(n)]
    cavities = {
        accession: ([1] if i % 2 == 0 else [1, 2])
        for i, accession in enumerate(accessions)
    }
    pairs = []
    for i, drug in enumerate(drugs):
        pairs.append((drug, accessions[i]))
        pairs.append((drug, accessions[(i + 1) % n]))
    index_dir = root
    os.makedirs(index_dir, exist_ok=True)
    index = write_index(os.path.join(index_dir, "identity_index.parquet"))
    # A pool covering a slice of the valid non-interactions, at cavity 1.
    pool_rows = []
    for i, drug in enumerate(drugs):
        target = accessions[(i + 5) % n]  # never i or i+1 => never an interaction
        pool_rows.append(pool_row(drug, target, index, cavity=1))
    return Fixture(
        root,
        make_interactions(pairs, index),
        make_positives(pairs, cavities, index),
        make_pool(pool_rows),
        index,
    )


class TestBalance(SamplerTestCase):
    def assert_balanced(self, result: sn.SamplingResult) -> None:
        deficits = result.deficits
        worst = deficits.loc[deficits["deficit"].abs().idxmax()]
        self.assertEqual(
            0,
            int(deficits["deficit"].abs().max()),
            "max|positives - negatives| must be 0 per drug and per target; "
            "worst: {0}".format(worst.to_dict()),
        )
        self.assertTrue(result.balance_ok)
        self.assertEqual(0, result.report["balance"]["max_abs_imbalance"])

    def test_balance_holds_exactly_on_a_satisfiable_fixture(self) -> None:
        fixture = ring_fixture(os.path.join(self.tmp, "in"))
        result = fixture.sample(seed=5)
        self.assert_balanced(result)
        positives = read_checked(fixture.positives_path, "positives_planned")
        self.assertEqual(len(positives), len(result.drawn))
        # per-entity equality, computed independently of the ledger
        for column in ("drugbank_id", "accession"):
            pos = positives.groupby(column).size()
            neg = result.drawn.groupby(column).size()
            self.assertEqual(sorted(pos.index), sorted(neg.index))
            for key in pos.index:
                self.assertEqual(
                    int(pos[key]),
                    int(neg[key]),
                    "{0}={1} has {2} positives but {3} negatives".format(
                        column, key, int(pos[key]), int(neg[key])
                    ),
                )

    def test_balance_holds_with_prefer_pool_too(self) -> None:
        fixture = ring_fixture(os.path.join(self.tmp, "in"))
        result = fixture.sample(prefer_pool=True, seed=5)
        self.assert_balanced(result)
        self.assertGreater(result.report["drawn"]["n_from_pool"], 0)

    def test_no_fallback_draws_on_an_exhausted_target_quota(self) -> None:
        """``generate_negative_samples.py:561-566``, deliberately absent.

        D1 needs 3 negatives; its only non-interacting target T2 has quota 2.
        The old fallback ("if no candidate needs negatives, use any available
        target") would have given T2 a third negative against 2 positives. Here
        D1 simply carries a deficit of 1, and it is reported.
        """
        index = write_index(os.path.join(self.tmp, "identity_index.parquet"))
        fixture = build_fixture(
            os.path.join(self.tmp, "in"),
            edges=[("DB00001", "P30518"), ("DB00002", "P35348")],
            positive_pairs=[("DB00001", "P30518"), ("DB00002", "P35348")],
            # P30518 has 3 cavities -> DB00001 quota 3; P35348 has 2 -> quota 2
            cavities={"P30518": [1, 2, 3], "P35348": [1, 2]},
        )
        result = fixture.sample(seed=1)
        drawn = result.drawn
        self.assertEqual(
            2,
            int((drawn["accession"] == "P35348").sum()),
            "P35348 has 2 positives and must never receive a 3rd negative",
        )
        deficits = result.deficits.set_index(["axis", "key"])
        self.assertEqual(1, int(deficits.loc[("drug", "DB00001"), "deficit"]))
        self.assertEqual(0, int(deficits.loc[("drug", "DB00002"), "deficit"]))
        self.assertFalse(result.balance_ok)
        # no target ever exceeds its quota
        targets = result.deficits[result.deficits["axis"] == "target"]
        self.assertTrue((targets["drawn"] <= targets["quota"]).all())

    def test_deficit_table_names_the_worst_offenders(self) -> None:
        index = write_index(os.path.join(self.tmp, "identity_index.parquet"))
        fixture = build_fixture(
            os.path.join(self.tmp, "in"),
            edges=[("DB00001", "P30518"), ("DB00002", "P35348")],
            positive_pairs=[("DB00001", "P30518"), ("DB00002", "P35348")],
            cavities={"P30518": [1, 2, 3], "P35348": [1, 2]},
        )
        out = self.out_dir()
        self.assertEqual(sn.EXIT_OK, sn.main(fixture.argv(out)))
        table = read_plain(os.path.join(out, "negatives_drawn.deficits.csv"))
        self.assertEqual(
            {"drug", "target"}, set(table["axis"]), "both axes are reported"
        )
        self.assertEqual(4, len(table), "every entity with a quota appears")
        self.assertEqual(
            list(table.columns),
            ["axis", "key", "gene", "unit", "quota", "drawn", "from_pool",
             "from_new", "deficit"],
        )
        self.assertEqual({"job"}, set(table["unit"]), "the file names its unit")
        worst = table[table["axis"] == "drug"].iloc[0]
        self.assertEqual("DB00001", worst["key"])
        self.assertEqual(1, int(worst["deficit"]))
        self.assertEqual(
            int(table[table["axis"] == "drug"]["deficit"].sum()),
            int(table[table["axis"] == "target"]["deficit"].sum()),
            "the two axes account for the same missing units",
        )

    def test_require_balance_exits_non_zero_when_short(self) -> None:
        index = write_index(os.path.join(self.tmp, "identity_index.parquet"))
        fixture = build_fixture(
            os.path.join(self.tmp, "in"),
            edges=[("DB00001", "P30518"), ("DB00002", "P35348")],
            positive_pairs=[("DB00001", "P30518"), ("DB00002", "P35348")],
            cavities={"P30518": [1, 2, 3], "P35348": [1, 2]},
        )
        self.assertEqual(
            sn.EXIT_UNBALANCED,
            sn.main(fixture.argv(self.out_dir("a"), "--require-balance")),
        )
        # ...and the artifacts are still written, so the deficit is inspectable
        self.assertTrue(
            os.path.exists(os.path.join(self.tmp, "a", "negatives_drawn.deficits.csv"))
        )
        fixture2 = ring_fixture(os.path.join(self.tmp, "in2"))
        self.assertEqual(
            sn.EXIT_OK,
            sn.main(fixture2.argv(self.out_dir("b"), "--require-balance", "--seed", "5")),
        )

    def test_a_drug_never_receives_more_negatives_than_positives(self) -> None:
        fixture = ring_fixture(os.path.join(self.tmp, "in"))
        result = fixture.sample(seed=99)
        self.assertTrue((result.deficits["deficit"] >= 0).all())


# --------------------------------------------------------------------------
# --prefer-pool
# --------------------------------------------------------------------------


def prefer_pool_fixture(root: str) -> Fixture:
    """A campaign where the greedy's first choice is *not* the pool's.

    Quotas: drugs DB1..DB5 = 1 each; targets P35348 = 3, P30518 = 1,
    P14416 = 1. DB1's neediest non-interacting target is P35348 (need 3), but
    the only pool job it has is on P30518 (need 1). Without ``--prefer-pool``
    DB1 draws P35348 and the pool job is wasted; with it, DB1 takes the pool
    job first.
    """
    os.makedirs(root, exist_ok=True)
    index = write_index(os.path.join(root, "identity_index.parquet"))
    edges = [
        ("DB00001", "P14416"),
        ("DB00002", "P35348"),
        ("DB00003", "P35348"),
        ("DB00004", "P35348"),
        ("DB00005", "P30518"),
    ]
    cavities = {"P35348": [1], "P30518": [1], "P14416": [1]}
    pool_rows = [pool_row("DB00001", "P30518", index, cavity=1)]
    return Fixture(
        root,
        make_interactions(edges, index),
        make_positives(edges, cavities, index),
        make_pool(pool_rows),
        index,
    )


class TestPreferPool(SamplerTestCase):
    def test_prefer_pool_draws_from_the_pool_before_queueing_a_redock(self) -> None:
        fixture = ring_fixture(os.path.join(self.tmp, "in"))
        pool = read_checked(fixture.pool_path, "negatives_pool")
        eligible_pairs = set(zip(pool["drugbank_id"], pool["accession"]))

        for seed in (1, 4, 7):
            plain = fixture.sample(prefer_pool=False, seed=seed)
            preferred = fixture.sample(prefer_pool=True, seed=seed)

            self.assertEqual(
                len(pool),
                preferred.report["drawn"]["n_from_pool"],
                "--prefer-pool must reuse every eligible already-docked job "
                "before asking for a fresh one (seed {0})".format(seed),
            )
            self.assertLess(
                plain.report["drawn"]["n_from_pool"],
                preferred.report["drawn"]["n_from_pool"],
                "seed {0}".format(seed),
            )
            self.assertLess(
                preferred.report["drawn"]["n_new"],
                plain.report["drawn"]["n_new"],
                "preferring the pool must shrink the re-dock queue",
            )
            # ...and the reuse is not bought with balance
            self.assertTrue(plain.balance_ok)
            self.assertTrue(preferred.balance_ok)

            reused = preferred.drawn[preferred.drawn["source"] == "pool"]
            self.assertEqual(
                eligible_pairs, set(zip(reused["drugbank_id"], reused["accession"]))
            )
            self.assertTrue(reused["old_job_dir"].notna().all())
            self.assertTrue(
                reused["selected_by"].str.contains("phase=pool_only").all(),
                "a reused job must be drawn in the pool-first phase",
            )

    def test_repair_may_sacrifice_a_reused_job_to_close_a_deficit(self) -> None:
        """Balance is the hard constraint; reuse is the optimisation.

        In this fixture the only way to serve the last drug is to move the one
        drug holding a pool job onto a different target. The sampler takes the
        trade and says so in the report, rather than quietly leaving a drug
        unbalanced to protect a reuse.
        """
        fixture = prefer_pool_fixture(os.path.join(self.tmp, "in"))
        result = fixture.sample(prefer_pool=True, seed=4)
        self.assertGreater(result.report["repair"]["n_repaired"], 0)
        self.assertEqual(0, result.report["drawn"]["n_from_pool"])
        self.assertLess(
            result.report["balance"]["drugs"]["total_deficit"],
            2,
            "the repair must have closed at least one deficit unit",
        )
        self.assertFalse(result.report["repair"]["budget_exhausted"])

    def test_pool_hit_and_redock_counts_are_reported_per_axis(self) -> None:
        fixture = ring_fixture(os.path.join(self.tmp, "in"))
        result = fixture.sample(prefer_pool=True, seed=5)
        ledger = result.deficits
        for axis in ("drug", "target"):
            part = ledger[ledger["axis"] == axis]
            self.assertTrue((part["from_pool"] + part["from_new"] == part["drawn"]).all())
            self.assertEqual(
                result.report["drawn"]["n_from_pool"], int(part["from_pool"].sum())
            )
            self.assertEqual(result.report["drawn"]["n_new"], int(part["from_new"].sum()))
        self.assertEqual(
            result.report["drawn"]["n_rows"],
            result.report["drawn"]["n_from_pool"] + result.report["drawn"]["n_new"],
        )

    def test_redock_queue_holds_exactly_the_new_rows(self) -> None:
        fixture = ring_fixture(os.path.join(self.tmp, "in"))
        out = self.out_dir()
        queue_path = os.path.join(out, "redock_queue.csv")
        self.assertEqual(
            sn.EXIT_OK,
            sn.main(fixture.argv(out, "--prefer-pool", "--seed", "5",
                                 "--redock-queue", queue_path)),
        )
        drawn = self.drawn_at(out)
        queue = read_checked(queue_path, "redock_queue")
        self.assertEqual(int((drawn["source"] == "new").sum()), len(queue))
        self.assertGreater(len(queue), 0)
        self.assertEqual(
            sorted(zip(queue["drugbank_id"], queue["accession"],
                       queue["fragment_id"], queue["cavity_index"])),
            sorted(
                zip(*[drawn.loc[drawn["source"] == "new", c]
                      for c in ("drugbank_id", "accession", "fragment_id", "cavity_index")])
            ),
        )
        self.assertTrue((queue["sample_type"] == "negative").all())

    def test_redock_job_ids_round_trip_including_a_hyphenated_symbol(self) -> None:
        """``MT-ND1`` broke the old regex; the new scheme must round-trip it."""
        index = write_index(os.path.join(self.tmp, "identity_index.parquet"))
        fixture = build_fixture(
            os.path.join(self.tmp, "in"),
            edges=[("DB00001", "P30518"), ("DB00002", "P03886")],
            positive_pairs=[("DB00001", "P30518"), ("DB00002", "P03886")],
            cavities={"P30518": [1], "P03886": [1]},
        )
        out = self.out_dir()
        queue_path = os.path.join(out, "redock_queue.csv")
        self.assertEqual(
            sn.EXIT_OK, sn.main(fixture.argv(out, "--redock-queue", queue_path))
        )
        queue = read_checked(queue_path, "redock_queue")
        self.assertIn("DB00001__P03886__MT-ND1__F1__cav1", set(queue["job_id"]))
        for job_id, accession, gene in zip(
            queue["job_id"], queue["accession"], queue["gene"]
        ):
            identity = parse_job_dir(job_id)
            self.assertEqual(accession, identity.accession)
            self.assertEqual(gene, identity.gene)

    def test_empty_redock_queue_is_still_a_valid_manifest(self) -> None:
        fixture = prefer_pool_fixture(os.path.join(self.tmp, "in"))
        # An all-pool draw is impossible here, so force the empty case by
        # asking for a queue on a run whose draws all come from the pool.
        index = write_index(os.path.join(self.tmp, "identity_index.parquet"))
        tiny = build_fixture(
            os.path.join(self.tmp, "in2"),
            edges=[("DB00001", "P30518"), ("DB00002", "P35348")],
            positive_pairs=[("DB00001", "P30518"), ("DB00002", "P35348")],
            cavities={"P30518": [1], "P35348": [1]},
            pool_rows=[
                pool_row("DB00001", "P35348", index, cavity=1),
                pool_row("DB00002", "P30518", index, cavity=1),
            ],
        )
        out = self.out_dir()
        queue_path = os.path.join(out, "redock_queue.csv")
        self.assertEqual(
            sn.EXIT_OK,
            sn.main(tiny.argv(out, "--prefer-pool", "--redock-queue", queue_path)),
        )
        queue = read_checked(queue_path, "redock_queue")
        self.assertEqual(0, len(queue))
        drawn = self.drawn_at(out)
        self.assertEqual(2, len(drawn))
        self.assertTrue((drawn["source"] == "pool").all())


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


class TestDeterminism(SamplerTestCase):
    ARTIFACTS = (
        "negatives_drawn.csv",
        "negatives_drawn.pool_eligibility.csv",
        "negatives_drawn.deficits.csv",
        "negatives_drawn.report.json",
    )

    def test_same_seed_produces_byte_identical_output(self) -> None:
        fixture = ring_fixture(os.path.join(self.tmp, "in"))
        first, second = self.out_dir("a"), self.out_dir("b")
        for out in (first, second):
            self.assertEqual(
                sn.EXIT_OK,
                sn.main(fixture.argv(out, "--prefer-pool", "--seed", "1234")),
            )
        for name in self.ARTIFACTS:
            self.assertEqual(
                sha256(os.path.join(first, name)),
                sha256(os.path.join(second, name)),
                "{0} differs between two runs with the same seed".format(name),
            )
        self.assertGreater(os.path.getsize(os.path.join(first, self.ARTIFACTS[0])), 200)

    def test_output_is_independent_of_pythonhashseed(self) -> None:
        """G3: no ``set`` iteration, no insertion-order dependence.

        ``prepare_uniprot_mapping.py:51`` did ``list(set(gene_names))`` and its
        output changed with the hash seed. A subprocess per seed is the only
        honest way to test that.
        """
        fixture = ring_fixture(os.path.join(self.tmp, "in"))
        digests = {}
        for seed in ("0", "1", "12345"):
            out = self.out_dir("hs{0}".format(seed))
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = seed
            proc = subprocess.run(
                [sys.executable, SCRIPT] + fixture.argv(out, "--prefer-pool", "--seed", "7"),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.tmp,
            )
            self.assertEqual(
                0, proc.returncode, proc.stderr.decode("utf-8", "replace")
            )
            digests[seed] = tuple(
                sha256(os.path.join(out, name)) for name in self.ARTIFACTS
            )
        self.assertEqual(
            1,
            len(set(digests.values())),
            "output depends on PYTHONHASHSEED: {0}".format(digests),
        )

    def test_the_seed_is_recorded_in_every_artifact(self) -> None:
        fixture = ring_fixture(os.path.join(self.tmp, "in"))
        out = self.out_dir()
        self.assertEqual(sn.EXIT_OK, sn.main(fixture.argv(out, "--seed", "77")))
        with open(os.path.join(out, "negatives_drawn.report.json")) as handle:
            report = json.load(handle)[0]
        self.assertEqual(77, report["seed"])
        drawn = self.drawn_at(out)
        self.assertTrue(drawn["selected_by"].str.contains("seed=77").all())

    def test_no_timestamp_leaks_into_the_artifacts(self) -> None:
        """A timestamp would make byte-identity impossible to assert at all."""
        fixture = ring_fixture(os.path.join(self.tmp, "in"))
        out = self.out_dir()
        self.assertEqual(sn.EXIT_OK, sn.main(fixture.argv(out)))
        for name in self.ARTIFACTS:
            with open(os.path.join(out, name)) as handle:
                text = handle.read()
            for token in ("timestamp", "generated_at", "20" + "26-", self.tmp):
                self.assertNotIn(token, text, "{0} embeds {1!r}".format(name, token))


# --------------------------------------------------------------------------
# input gates — a missing prerequisite is an error, never a fallback (G8)
# --------------------------------------------------------------------------


class TestInputGates(SamplerTestCase):
    def base(self) -> Fixture:
        return ring_fixture(os.path.join(self.tmp, "in"))

    def test_missing_pool_file_is_an_error_not_an_empty_pool(self) -> None:
        fixture = self.base()
        os.remove(fixture.pool_path)
        self.assertEqual(sn.EXIT_INPUT, sn.main(fixture.argv(self.out_dir())))
        self.assertFalse(
            os.path.exists(os.path.join(self.tmp, "out", "negatives_drawn.csv")),
            "nothing may be written when a prerequisite is missing",
        )

    def test_missing_index_is_an_error(self) -> None:
        fixture = self.base()
        os.remove(fixture.index_path)
        self.assertEqual(sn.EXIT_INPUT, sn.main(fixture.argv(self.out_dir())))

    def test_index_sha256_pin_is_enforced(self) -> None:
        fixture = self.base()
        self.assertEqual(
            sn.EXIT_INPUT,
            sn.main(fixture.argv(self.out_dir(), "--index-sha256", "0" * 64)),
        )
        self.assertEqual(
            sn.EXIT_OK,
            sn.main(
                fixture.argv(
                    self.out_dir("ok"), "--index-sha256", sha256(fixture.index_path)
                )
            ),
        )

    def test_empty_positives_raises_instead_of_writing_an_empty_manifest(self) -> None:
        fixture = self.base()
        index, interactions, positives, pool = fixture.frames()
        with self.assertRaises(InvariantViolation) as ctx:
            sn.sample_negatives(
                index=index,
                interactions=interactions,
                positives=positives.iloc[0:0],
                pool=pool,
            )
        self.assertIn("empty", str(ctx.exception).lower())

    def test_pool_may_be_empty_but_must_be_supplied(self) -> None:
        fixture = self.base()
        index, interactions, positives, pool = fixture.frames()
        result = sn.sample_negatives(
            index=index,
            interactions=interactions,
            positives=positives,
            pool=pool.iloc[0:0],
            prefer_pool=True,
            seed=5,
        )
        self.assertEqual(0, result.report["drawn"]["n_from_pool"])
        self.assertEqual(len(positives), result.report["drawn"]["n_new"])
        self.assertTrue(result.balance_ok)

    def test_positive_pair_absent_from_interactions_is_refused(self) -> None:
        """Two manifests from different runs must not be silently combined."""
        fixture = self.base()
        index, interactions, positives, pool = fixture.frames()
        with self.assertRaises(InvariantViolation) as ctx:
            sn.sample_negatives(
                index=index,
                interactions=interactions.iloc[1:],
                positives=positives,
                pool=pool,
            )
        self.assertIn("interactions_resolved", str(ctx.exception))

    def test_positive_accession_absent_from_the_index_is_refused(self) -> None:
        fixture = self.base()
        index, interactions, positives, pool = fixture.frames()
        broken = positives.copy()
        broken.loc[broken.index[0], "accession"] = "P99999"
        with self.assertRaises(InvariantViolation) as ctx:
            sn.sample_negatives(
                index=index, interactions=interactions, positives=broken, pool=pool
            )
        self.assertIn("P99999", str(ctx.exception))

    def test_a_manifest_label_that_names_another_protein_is_refused(self) -> None:
        """The KNOWN_ISSUE.md regression assertion, on an input manifest.

        57.7% of ``uniprot_gene_mapping.csv`` rows failed exactly this check.
        """
        index = write_index(os.path.join(self.tmp, "identity_index.parquet"))
        fixture = build_fixture(
            os.path.join(self.tmp, "in"),
            edges=[("DB00001", "P68363"), ("DB00002", "P35348")],
            positive_pairs=[("DB00001", "P68363"), ("DB00002", "P35348")],
            cavities={"P68363": [1], "P35348": [1]},
            pool_rows=[
                dict(
                    pool_row("DB00001", "P35348", index, cavity=1),
                    gene="TUBA1B",  # P35348 is ADRA1A; this is the 2025 defect
                )
            ],
        )
        index2, interactions, positives, pool = fixture.frames()
        with self.assertRaises(InvariantViolation) as ctx:
            sn.sample_negatives(
                index=index2, interactions=interactions, positives=positives, pool=pool
            )
        self.assertIn("P35348", str(ctx.exception))
        self.assertIn("TUBA1B", str(ctx.exception))

    def test_a_synonym_label_is_accepted_but_not_propagated(self) -> None:
        """``ADRA1C`` is a real synonym of ADRA1A, so the input is legal...

        ...but the emitted label is always the index's primary symbol, so a
        downstream join can never see two spellings of one protein.
        """
        index = write_index(os.path.join(self.tmp, "identity_index.parquet"))
        fixture = build_fixture(
            os.path.join(self.tmp, "in"),
            edges=[("DB00001", "P68363"), ("DB00002", "P35348")],
            positive_pairs=[("DB00001", "P68363"), ("DB00002", "P35348")],
            cavities={"P68363": [1], "P35348": [1]},
            pool_rows=[
                dict(pool_row("DB00001", "P35348", index, cavity=1), gene="ADRA1C")
            ],
        )
        result = fixture.sample(prefer_pool=True)
        row = result.drawn[result.drawn["accession"] == "P35348"]
        self.assertEqual(1, len(row))
        self.assertEqual("ADRA1A", row["gene"].iloc[0])


# --------------------------------------------------------------------------
# output contract
# --------------------------------------------------------------------------


class TestOutputContract(SamplerTestCase):
    def test_output_satisfies_the_negatives_drawn_schema(self) -> None:
        fixture = ring_fixture(os.path.join(self.tmp, "in"))
        out = self.out_dir()
        self.assertEqual(
            sn.EXIT_OK, sn.main(fixture.argv(out, "--prefer-pool", "--seed", "5"))
        )
        drawn = self.drawn_at(out)  # read_checked re-validates on read
        self.assertTrue((drawn["sample_type"] == "negative").all())
        self.assertTrue(drawn["gene"].notna().all(), "no blank gene, ever")
        self.assertTrue((drawn["gene"].str.len() > 0).all())
        self.assertTrue(drawn["selected_by"].notna().all())
        pool_rows = drawn[drawn["source"] == "pool"]
        new_rows = drawn[drawn["source"] == "new"]
        self.assertTrue(pool_rows["old_job_dir"].notna().all())
        self.assertTrue(new_rows["old_job_dir"].isna().all())

    def test_primary_key_is_unique(self) -> None:
        fixture = ring_fixture(os.path.join(self.tmp, "in"))
        result = fixture.sample(prefer_pool=True, seed=5)
        key = result.drawn[
            ["drugbank_id", "accession", "fragment_id", "cavity_index"]
        ]
        self.assertEqual(len(key), len(key.drop_duplicates()))

    def test_duplicate_pool_pockets_cannot_break_the_primary_key(self) -> None:
        """Two old directories for one pocket: keep one, record the other."""
        index = write_index(os.path.join(self.tmp, "identity_index.parquet"))
        fixture = build_fixture(
            os.path.join(self.tmp, "in"),
            edges=[("DB00001", "P30518"), ("DB00002", "P35348")],
            positive_pairs=[("DB00001", "P30518"), ("DB00002", "P35348")],
            cavities={"P30518": [1], "P35348": [1]},
            pool_rows=[
                pool_row("DB00001", "P35348", index, cavity=1,
                         old_job_dir="DB00001_ADRA1A_P35348_cavity_1"),
                pool_row("DB00001", "P35348", index, cavity=1,
                         old_job_dir="DB00001_ADRA1C_P35348_cavity_1"),
            ],
        )
        result = fixture.sample(prefer_pool=True)
        reasons = dict(
            zip(result.eligibility["old_job_dir"], result.eligibility["eligibility"])
        )
        self.assertEqual(sn.ELIGIBLE, reasons["DB00001_ADRA1A_P35348_cavity_1"])
        self.assertEqual(
            sn.REASON_DUPLICATE_POCKET, reasons["DB00001_ADRA1C_P35348_cavity_1"]
        )
        rows = result.drawn[
            (result.drawn["drugbank_id"] == "DB00001")
            & (result.drawn["accession"] == "P35348")
        ]
        self.assertEqual(1, len(rows))
        self.assertEqual("DB00001_ADRA1A_P35348_cavity_1", rows["old_job_dir"].iloc[0])

    def test_a_pool_job_without_final_results_is_re_docked_not_reused(self) -> None:
        index = write_index(os.path.join(self.tmp, "identity_index.parquet"))
        fixture = build_fixture(
            os.path.join(self.tmp, "in"),
            edges=[("DB00001", "P30518"), ("DB00002", "P35348")],
            positive_pairs=[("DB00001", "P30518"), ("DB00002", "P35348")],
            cavities={"P30518": [1], "P35348": [1]},
            pool_rows=[
                pool_row("DB00001", "P35348", index, cavity=1, has_final_results=False)
            ],
        )
        result = fixture.sample(prefer_pool=True)
        self.assertEqual(
            sn.REASON_NO_FINAL_RESULTS, result.eligibility["eligibility"].iloc[0]
        )
        row = result.drawn[
            (result.drawn["drugbank_id"] == "DB00001")
            & (result.drawn["accession"] == "P35348")
        ]
        self.assertEqual(1, len(row))
        self.assertEqual("new", row["source"].iloc[0])

    def test_sidecar_paths_can_be_overridden(self) -> None:
        fixture = ring_fixture(os.path.join(self.tmp, "in"))
        out = self.out_dir()
        paths = {
            "--pool-eligibility": os.path.join(out, "elig.csv"),
            "--deficits": os.path.join(out, "def.csv"),
            "--report": os.path.join(out, "rep.json"),
        }
        extra: List[str] = []
        for flag, path in paths.items():
            extra += [flag, path]
        self.assertEqual(sn.EXIT_OK, sn.main(fixture.argv(out, *extra)))
        for path in paths.values():
            self.assertTrue(os.path.exists(path), path)
        self.assertFalse(
            os.path.exists(os.path.join(out, "negatives_drawn.deficits.csv"))
        )

    def test_a_run_that_can_draw_nothing_still_writes_a_valid_manifest(self) -> None:
        """One drug, one target, and they interact: no negative exists."""
        fixture = build_fixture(
            os.path.join(self.tmp, "in"),
            edges=[("DB00001", "P30518")],
            positive_pairs=[("DB00001", "P30518")],
            cavities={"P30518": [1, 2]},
        )
        out = self.out_dir()
        queue_path = os.path.join(out, "redock_queue.csv")
        self.assertEqual(
            sn.EXIT_OK, sn.main(fixture.argv(out, "--redock-queue", queue_path))
        )
        drawn = self.drawn_at(out)
        self.assertEqual(0, len(drawn))
        self.assertEqual(0, len(read_checked(queue_path, "redock_queue")))
        table = read_plain(os.path.join(out, "negatives_drawn.deficits.csv"))
        self.assertEqual(
            [2, 2], sorted(table["deficit"].tolist()), "the shortfall is reported"
        )
        self.assertEqual(
            sn.EXIT_UNBALANCED,
            sn.main(fixture.argv(self.out_dir("strict"), "--require-balance")),
        )

    def test_the_eligibility_ledger_never_changes_the_pool_row_count(self) -> None:
        """checked_merge on the quota tables: a merge may not add rows."""
        index = write_index(os.path.join(self.tmp, "identity_index.parquet"))
        rows = [
            pool_row("DB00001", "P35348", index, cavity=c,
                     old_job_dir="j{0}".format(c))
            for c in range(1, 9)
        ]
        fixture = build_fixture(
            os.path.join(self.tmp, "in"),
            edges=[("DB00001", "P30518"), ("DB00002", "P35348")],
            positive_pairs=[("DB00001", "P30518"), ("DB00002", "P35348")],
            cavities={"P30518": list(range(1, 9)), "P35348": list(range(1, 9))},
            pool_rows=rows,
        )
        index2, interactions, positives, pool = fixture.frames()
        classified = sn.classify_pool(
            pool,
            *sn.compute_quotas(positives),
            avoid=sn.build_avoid_list(interactions),
            index=index2,
        )
        self.assertEqual(len(pool), len(classified))


# --------------------------------------------------------------------------
# balance unit
# --------------------------------------------------------------------------


class TestBalanceUnit(SamplerTestCase):
    def fixture(self) -> Fixture:
        # P35348 has three cavities, P30518 one. In job space the quotas are
        # 3 and 1; in pair space both are 1.
        return build_fixture(
            os.path.join(self.tmp, "in"),
            edges=[("DB00001", "P35348"), ("DB00002", "P30518")],
            positive_pairs=[("DB00001", "P35348"), ("DB00002", "P30518")],
            cavities={"P35348": [1, 2, 3], "P30518": [1]},
        )

    def test_job_unit_counts_cavity_rows(self) -> None:
        result = self.fixture().sample(balance_unit=sn.UNIT_JOB, seed=2)
        ledger = result.deficits.set_index(["axis", "key"])
        self.assertEqual(3, int(ledger.loc[("drug", "DB00001"), "quota"]))
        self.assertEqual(1, int(ledger.loc[("drug", "DB00002"), "quota"]))
        # DB00001 needs 3 negatives but P30518 has only one cavity => deficit 2
        self.assertEqual(2, int(ledger.loc[("drug", "DB00001"), "deficit"]))

    def test_pair_unit_counts_pairs_and_mirrors_the_positive_cavity_depth(self) -> None:
        result = self.fixture().sample(balance_unit=sn.UNIT_PAIR, seed=2)
        ledger = result.deficits.set_index(["axis", "key"])
        self.assertEqual(1, int(ledger.loc[("drug", "DB00001"), "quota"]))
        self.assertEqual(1, int(ledger.loc[("target", "P35348"), "quota"]))
        self.assertTrue(result.balance_ok, "pair balance is satisfiable here")
        drawn = result.drawn
        # DB00002 draws the pair (DB00002, P35348); P35348 has 3 positive
        # cavities, so the negative pair is docked at the same depth.
        self.assertEqual(
            3, int((drawn["accession"] == "P35348").sum()),
            "a negative pair must get the positives' cavity depth",
        )
        self.assertEqual(1, int((drawn["accession"] == "P30518").sum()))
        self.assertEqual(2, result.report["drawn"]["n_pairs"])

    def test_the_declared_unit_is_recorded_everywhere(self) -> None:
        fixture = self.fixture()
        out = self.out_dir()
        self.assertEqual(
            sn.EXIT_OK, sn.main(fixture.argv(out, "--balance-unit", "pair"))
        )
        drawn = self.drawn_at(out)
        self.assertTrue(drawn["selected_by"].str.contains("unit=pair").all())
        with open(os.path.join(out, "negatives_drawn.report.json")) as handle:
            report = json.load(handle)[0]
        self.assertEqual("pair", report["balance_unit"])
        self.assertEqual("pair", report["balance"]["unit"])

    def test_pair_mode_still_refuses_known_interactions(self) -> None:
        fixture = ring_fixture(os.path.join(self.tmp, "in"))
        result = fixture.sample(balance_unit=sn.UNIT_PAIR, prefer_pool=True, seed=8)
        avoid = sn.build_avoid_list(
            read_checked(fixture.interactions_path, "interactions_resolved")
        )
        drawn_pairs = set(
            zip(result.drawn["drugbank_id"], result.drawn["accession"])
        )
        self.assertEqual(set(), drawn_pairs & avoid)

    def test_an_unknown_balance_unit_is_rejected(self) -> None:
        fixture = ring_fixture(os.path.join(self.tmp, "in"))
        index, interactions, positives, pool = fixture.frames()
        with self.assertRaises(ValueError):
            sn.sample_negatives(
                index=index,
                interactions=interactions,
                positives=positives,
                pool=pool,
                balance_unit="cavity",
            )


# --------------------------------------------------------------------------
# unit-level helpers
# --------------------------------------------------------------------------


class TestHelpers(unittest.TestCase):
    def test_pocket_sort_key_orders_fragments_numerically(self) -> None:
        pockets = [("F10", 1), ("F2", 3), ("F1", 2), ("F2", 1)]
        self.assertEqual(
            [("F1", 2), ("F2", 1), ("F2", 3), ("F10", 1)],
            sorted(pockets, key=sn._pocket_sort_key),
        )

    def test_compute_quotas_job_versus_pair(self) -> None:
        positives = pd.DataFrame(
            {
                "drugbank_id": ["DB00001", "DB00001", "DB00002"],
                "accession": ["P35348", "P35348", "P35348"],
                "fragment_id": ["F1", "F1", "F1"],
                "cavity_index": [1, 2, 1],
            }
        )
        drugs, targets = sn.compute_quotas(positives, sn.UNIT_JOB)
        self.assertEqual({"DB00001": 2, "DB00002": 1}, drugs)
        self.assertEqual({"P35348": 3}, targets)
        drugs, targets = sn.compute_quotas(positives, sn.UNIT_PAIR)
        self.assertEqual({"DB00001": 1, "DB00002": 1}, drugs)
        self.assertEqual({"P35348": 2}, targets)

    def test_pockets_by_accession_unions_its_inputs_deterministically(self) -> None:
        a = pd.DataFrame(
            {"accession": ["P1", "P1"], "fragment_id": ["F2", "F1"],
             "cavity_index": [1, 3]}
        )
        b = pd.DataFrame(
            {"accession": ["P1"], "fragment_id": ["F1"], "cavity_index": [1]}
        )
        self.assertEqual(
            {"P1": (("F1", 1), ("F1", 3), ("F2", 1))},
            sn.pockets_by_accession(a, b),
        )

    def test_need_heap_returns_the_neediest_acceptable_target(self) -> None:
        needs = {"P1": 3, "P2": 5, "P3": 1}
        heap = sn._NeedHeap(needs, {"P1": 0.1, "P2": 0.2, "P3": 0.3})
        self.assertEqual("P2", heap.take_best(lambda t: True))
        self.assertEqual(4, needs["P2"])
        self.assertEqual("P1", heap.take_best(lambda t: t != "P2"))
        # exhaust P3 and confirm it leaves the heap for good
        self.assertEqual("P3", heap.take_best(lambda t: t == "P3"))
        self.assertIsNone(heap.take_best(lambda t: t == "P3"))

    def test_need_heap_has_no_use_any_target_fallback(self) -> None:
        needs = {"P1": 0}
        heap = sn._NeedHeap(needs, {"P1": 0.5})
        self.assertIsNone(
            heap.take_best(lambda t: True),
            "a target at quota must never be offered again",
        )

    def test_repair_releases_a_fresh_unit_before_a_reused_one(self) -> None:
        """Two donors could close the deficit; the reused one must survive.

        ``DB00001`` holds an already-docked job and sorts first, so a
        first-match search would sacrifice it. ``DB00002`` holds a job that has
        not been docked yet and costs nothing to move.
        """
        pockets = {
            "P30518": (("F1", 1),),
            "P35348": (("F1", 1),),
            "P14416": (("F1", 1),),
        }
        eligible_pool = pd.DataFrame(
            [
                {
                    "drugbank_id": "DB00001",
                    "accession": "P30518",
                    "fragment_id": "F1",
                    "cavity_index": 1,
                    "old_job_dir": "keep_me",
                }
            ]
        )
        book = sn._JobBook(eligible_pool, pockets)
        reused = book.take("DB00001", "P30518", allow_new=True)
        fresh = book.take("DB00002", "P35348", allow_new=True)
        self.assertEqual("keep_me", reused.old_job_dir)
        self.assertIsNone(fresh.old_job_dir)

        draws = [
            sn._Draw("DB00001", "P30518", reused, sn.PHASE_OPEN),
            sn._Draw("DB00002", "P35348", fresh, sn.PHASE_OPEN),
        ]
        need_drug = {"DB00001": 0, "DB00002": 0, "DB00003": 1}
        need_target = {"P30518": 0, "P35348": 0, "P14416": 1}
        kept, stats = sn._repair(
            book,
            ["DB00001", "DB00002", "DB00003"],
            need_drug,
            need_target,
            frozenset({("DB00003", "P14416")}),
            allow_new=True,
            draws=draws,
        )
        self.assertEqual(1, stats["n_repaired"])
        placed = {(d.drugbank_id, d.accession): d.unit for d in kept}
        self.assertIn(("DB00001", "P30518"), placed)
        self.assertEqual(
            "keep_me",
            placed[("DB00001", "P30518")].old_job_dir,
            "the already-docked job must not be the one released",
        )
        self.assertIn(("DB00002", "P14416"), placed)
        self.assertIn(("DB00003", "P35348"), placed)
        self.assertEqual({0, 0, 0}, set(need_drug.values()))
        self.assertEqual({0, 0, 0}, set(need_target.values()))

    def test_repair_stops_at_its_scan_budget(self) -> None:
        book = sn._JobBook(
            pd.DataFrame(
                columns=[
                    "drugbank_id",
                    "accession",
                    "fragment_id",
                    "cavity_index",
                    "old_job_dir",
                ]
            ),
            {"P30518": (("F1", 1),), "P35348": (("F1", 1),)},
        )
        unit = book.take("DB00002", "P35348", allow_new=True)
        draws = [sn._Draw("DB00002", "P35348", unit, sn.PHASE_OPEN)]
        need_drug = {"DB00002": 0, "DB00003": 1}
        need_target = {"P35348": 0, "P30518": 1}
        kept, stats = sn._repair(
            book,
            ["DB00002", "DB00003"],
            need_drug,
            need_target,
            frozenset({("DB00003", "P30518")}),
            allow_new=True,
            draws=draws,
            max_scans=0,
        )
        self.assertTrue(stats["budget_exhausted"])
        self.assertEqual(0, stats["n_repaired"])
        self.assertEqual(1, len(kept), "an aborted repair leaves the greedy result")

    def test_build_avoid_list_requires_accession_columns(self) -> None:
        with self.assertRaises(SchemaViolation):
            sn.build_avoid_list(
                pd.DataFrame({"drugbank_id": ["DB1"], "gene": ["AVPR2"]})
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
