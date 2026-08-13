"""Tests for resolve_interactions.py (REBUILD_PLAN Phase 1).

These are regression tests for a specific incident. Every fixture value below is
a real value copied out of the campaign's inputs as a literal -- no file is read,
no drive is touched, no network is used:

``uniprot_gene_mapping.csv`` / the identity axis::

    P30518 V2R_HUMAN   AVPR2 ...  entrez 554     the corrupt map said AVPR2 -> Q8NCT1
    Q8NCT1 ARRD4_HUMAN ARRDC4     entrez 91947   ... which is ARRDC4, a different protein
    P68363 TBA1B_HUMAN TUBA1B     entrez 10376   TUBA1B -> Q13509 in the corrupt map
    Q13509 TBB3_HUMAN  TUBB3 ...  entrez 10381   ... which is TUBB3

``drugbank vocabulary.csv`` / the compound axis::

    DB00035 Desmopressin      accessions BTD00112 | BTD00061 | BIOD00112 | BIOD00061
    DB00125 Arginine          synonyms ... | L-Arg | L-Arginin | L-arginine | R
    DB17289 L-arginine        no synonyms          <- 'L-Arginine' names BOTH of these
    DB00155 Citrulline        synonyms ... | L-Citrulline | ...
    DB11620 Neridronic Acid   <- the slug 'neridronic-acid'
    DB05482 SN-38             <- the slug 'SN-38'
    DB01169 Arsenic trioxide  <- the slug 'arsenic-trioxide', which ALSO appears
                                 as DB01169 elsewhere in the same edge file

``1_drug_to_protein.tsv``::

    8,568 edges; node_2 numeric on 8,568/8,568; node_2_name null on 237;
    node_1 a name slug on 222 rows (110 distinct); 'clorprenaline' matches no
    DrugBank name at all; DB00729 has DrugBank syntax but is not in the
    vocabulary; 38 edges duplicate a (drug, target) pair via slug + id.

Hermetic: every file is written inside ``tempfile.TemporaryDirectory()``.
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

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import resolve_interactions as ri  # noqa: E402
from dock_selectivity.identity import IdentityIndex  # noqa: E402
from dock_selectivity.invariants import (  # noqa: E402
    InvariantViolation,
    assert_collapse_factor_one,
    cardinality_report,
)
from dock_selectivity.schemas import SCHEMAS, read_checked  # noqa: E402


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

#: (accession, entry_name, primary_symbol, symbols, entrez_id, reviewed)
INDEX_ROWS = [
    ("P30518", "V2R_HUMAN", "AVPR2", ["AVPR2", "ADHR", "DIR", "DIR3", "V2R"], 554, True),
    ("Q8NCT1", "ARRD4_HUMAN", "ARRDC4", ["ARRDC4"], 91947, True),
    ("P68363", "TBA1B_HUMAN", "TUBA1B", ["TUBA1B"], 10376, True),
    ("Q13509", "TBB3_HUMAN", "TUBB3", ["TUBB3", "TUBB4"], 10381, True),
    ("P00966", "ASSY_HUMAN", "ASS1", ["ASS1", "ASS"], 445, True),
    ("P05023", "AT1A1_HUMAN", "ATP1A1", ["ATP1A1"], 476, True),
]

#: (DrugBank ID, Accession Numbers, Common name, Synonyms)
VOCABULARY_ROWS = [
    (
        "DB00035",
        "BTD00112 | BTD00061 | BIOD00112 | BIOD00061",
        "Desmopressin",
        "dDAVP | Desmopresina | Desmopressine | Desmopressinum",
    ),
    (
        "DB00125",
        "NUTR00014",
        "Arginine",
        "Arg | Arginina | Arginine | L-(+)-Arginine | L-Arg | L-Arginin | L-arginine | R",
    ),
    ("DB17289", "", "L-arginine", ""),
    (
        "DB00155",
        "NUTR00021",
        "Citrulline",
        "Cit | Citrulina | Citrulline | L-Citrulline | delta-Ureidonorvaline",
    ),
    ("DB11620", "", "Neridronic Acid", ""),
    ("DB05482", "", "SN-38", ""),
    ("DB01169", "APRD00171", "Arsenic trioxide", "Arsenic oxide | White arsenic"),
    # A biotech drug: present in the vocabulary, absent from the small-molecule
    # links file. DB00104 (octreotide) is such a row in the real data.
    ("DB00104", "BTD00034", "Octreotide", "Octreotida | Octreotidum"),
]

SMALL_MOLECULE_IDS = [
    "DB00035",
    "DB00125",
    "DB17289",
    "DB00155",
    "DB11620",
    "DB05482",
    "DB01169",
]


def index_frame(rows=None):
    rows = INDEX_ROWS if rows is None else rows
    return pd.DataFrame(
        {
            "accession": [r[0] for r in rows],
            "entry_name": [r[1] for r in rows],
            "primary_symbol": [r[2] for r in rows],
            "symbols": [list(r[3]) for r in rows],
            "entrez_id": pd.array([r[4] for r in rows], dtype="Int64"),
            "reviewed": [r[5] for r in rows],
        }
    )


def make_index(rows=None, **kwargs):
    return IdentityIndex.from_frame(index_frame(rows), **kwargs)


def vocabulary_frame(rows=None):
    rows = VOCABULARY_ROWS if rows is None else rows
    return pd.DataFrame(
        {
            "DrugBank ID": [r[0] for r in rows],
            "Accession Numbers": [r[1] or None for r in rows],
            "Common name": [r[2] for r in rows],
            "CAS": [None] * len(rows),
            "UNII": [None] * len(rows),
            "Synonyms": [r[3] or None for r in rows],
            "Standard InChI Key": [None] * len(rows),
        }
    )


def make_compounds(rows=None, **kwargs):
    return ri.CompoundIndex.from_frame(
        vocabulary_frame(rows), source_path="<fixture>", **kwargs
    )


def make_small_molecules(ids=None):
    ids = SMALL_MOLECULE_IDS if ids is None else ids
    return pd.DataFrame({"drugbank_id": list(ids)})


#: (node_1, node_2, node_1_name, node_2_name) -- node_*_name are diagnostics
#: only and are deliberately wrong or absent on several rows.
EDGE_ROWS = [
    # 0  the regression case: entrez 554 is AVPR2 -> P30518, never Q8NCT1
    ("DB00035", "554", "Desmopressin", "AVPR2"),
    # 1  node_2_name is null: one of the 237 edges a symbol-keyed resolver could
    #    never reach, because it has no symbol at all
    ("DB00035", "10376", "Desmopressin", None),
    # 2  node_2_name names a DIFFERENT protein than node_2 does. The output must
    #    follow node_2 and label from the index.
    ("DB01169", "10381", "Arsenic trioxide", "ARRDC4"),
    # 3  a name slug matching a DrugBank common name
    ("neridronic-acid", "476", None, "ATP1A1"),
    # 4  a name slug matching a synonym
    ("L-Citrulline", "445", None, "ASS1"),
    # 5  the same (drug, target) as row 4, written as a DrugBank ID: a duplicate,
    #    not a conflict (38 such rows exist in the real file)
    ("DB00155", "445", "Citrulline", "ASS1"),
    # 6  AMBIGUOUS: 'L-Arginine' is DB00125's synonym and DB17289's common name
    ("L-Arginine", "445", None, "ASS1"),
    # 7  a slug matching nothing at all
    ("clorprenaline", "554", None, "AVPR2"),
    # 8  DrugBank syntax, absent from the vocabulary (DB00729 is the real one)
    ("DB00729", "554", None, "AVPR2"),
    # 9  not a small molecule
    ("DB00104", "554", "octreotide", None),
    # 10 no Entrez key at all
    ("DB00035", None, "Desmopressin", "AVPR2"),
    # 11 an Entrez id the index does not carry
    ("DB00035", "999999999", "Desmopressin", "NOSUCHGENE"),
]


def edges_frame(rows=None, order=None):
    """Build an edge frame; ``order`` permutes the file rows before numbering.

    ``edge_row`` is assigned by position, exactly as :func:`ri.read_edges` does,
    so a permuted frame is a genuinely different input file -- which is what the
    determinism tests need.
    """
    rows = EDGE_ROWS if rows is None else rows
    if order is not None:
        rows = [rows[i] for i in order]
    frame = pd.DataFrame(
        {
            "node_1": [r[0] for r in rows],
            "node_2": [r[1] for r in rows],
            "node_1_type": ["drug"] * len(rows),
            "node_2_type": ["protein"] * len(rows),
            "node_1_name": [r[2] for r in rows],
            "node_2_name": [r[3] for r in rows],
        }
    )
    frame.insert(0, "edge_row", range(len(frame)))
    return frame


def run(edges=None, index=None, compounds=None, small=None, **kwargs):
    return ri.resolve_interactions(
        edges_frame() if edges is None else edges,
        make_index() if index is None else index,
        make_compounds() if compounds is None else compounds,
        make_small_molecules() if small is None else small,
        **kwargs
    )


def write_inputs(directory, edges=None, index=None, vocabulary=None, drugs=None):
    """Materialise the four CLI inputs inside ``directory``; return their paths."""
    edge_frame = edges_frame() if edges is None else edges
    edges_path = os.path.join(directory, "1_drug_to_protein.tsv")
    edge_frame.drop(columns=["edge_row"]).to_csv(
        edges_path, sep="\t", index=False, lineterminator="\n"
    )

    index_path = os.path.join(directory, "identity_index.parquet")
    (make_index() if index is None else index).to_frame().to_parquet(
        index_path, index=False
    )

    vocabulary_path = os.path.join(directory, "drugbank vocabulary.csv")
    (vocabulary_frame() if vocabulary is None else vocabulary).to_csv(
        vocabulary_path, index=False, lineterminator="\n"
    )

    drugs_path = os.path.join(directory, "small_molecule_drug_links.csv")
    if drugs is None:
        drugs = pd.DataFrame(
            {
                "DrugBank ID": SMALL_MOLECULE_IDS,
                "Name": ["x"] * len(SMALL_MOLECULE_IDS),
                "Drug Type": ["SmallMoleculeDrug"] * len(SMALL_MOLECULE_IDS),
            }
        )
    drugs.to_csv(drugs_path, index=False, lineterminator="\n")
    return edges_path, index_path, vocabulary_path, drugs_path


def cli(directory, extra=(), **kwargs):
    """Run ``main()`` against fixtures in ``directory``; return (rc, out, rejects)."""
    edges_path, index_path, vocabulary_path, drugs_path = write_inputs(
        directory, **kwargs
    )
    out_path = os.path.join(directory, "interactions_resolved.csv")
    rejects_path = os.path.join(directory, "unresolved_targets.csv")
    argv = [
        "--edges",
        edges_path,
        "--index",
        index_path,
        "--drugs",
        drugs_path,
        "--vocabulary",
        vocabulary_path,
        "--out",
        out_path,
        "--rejects",
        rejects_path,
    ] + list(extra)
    logging.disable(logging.CRITICAL)
    try:
        rc = ri.main(argv)
    finally:
        logging.disable(logging.NOTSET)
    return rc, out_path, rejects_path


def sha256_of(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def setUpModule():
    """Keep the stage's INFO chatter out of the test report.

    ``ri.main()`` calls ``logging.basicConfig``, which installs a root handler
    for the rest of the process; without this, later direct calls to
    ``read_edges``/``load_small_molecule_ids`` print into the runner's output.
    """
    for name in ("resolve_interactions", "dock_selectivity.schemas",
                 "dock_selectivity.invariants", "dock_selectivity.joins"):
        logging.getLogger(name).setLevel(logging.WARNING)


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


class TestNormalizeName(unittest.TestCase):
    """The only transformation applied before an exact name match."""

    def test_real_slugs_from_node_1(self):
        cases = {
            "neridronic-acid": "neridronic acid",
            "SN-38": "sn 38",
            "L-Citrulline": "l citrulline",
            "arsenic-trioxide": "arsenic trioxide",
            "butorphanol-(+)-tartrate": "butorphanol tartrate",
            "cycloserine-(d)": "cycloserine d",
            "5-hydroxytryptophan": "5 hydroxytryptophan",
            "ACT-132577": "act 132577",
            "clorprenaline": "clorprenaline",
        }
        for slug, expected in cases.items():
            self.assertEqual(ri.normalize_name(slug), expected, slug)

    def test_accents_are_folded_but_letters_are_not_deleted(self):
        # 'Cétuximab' is a real DrugBank synonym; folding the accent is what
        # makes it comparable, deleting the letter would not be.
        self.assertEqual(ri.normalize_name("Cétuximab"), "cetuximab")
        self.assertEqual(ri.normalize_name("Cetuximab"), "cetuximab")

    def test_non_latin_letters_survive(self):
        # A [^0-9a-z] filter would turn 'α-tocopherol' into 'tocopherol' and
        # make it collide with a different compound. It must not.
        self.assertEqual(ri.normalize_name("α-tocopherol"), "α tocopherol")

    def test_nothing_left_is_none_not_empty_string(self):
        for value in ("", "   ", "---", "()", None, float("nan"), pd.NA):
            self.assertIsNone(ri.normalize_name(value), repr(value))

    def test_normalisation_is_not_fuzzy(self):
        # Order matters, and a missing token matters. These are the two ways a
        # fuzzy resolver would have "helpfully" matched.
        self.assertNotEqual(
            ri.normalize_name("cycloserine-(d)"), ri.normalize_name("D-Cycloserine")
        )
        self.assertNotEqual(
            ri.normalize_name("dabigatran"), ri.normalize_name("dabigatran-etexilate")
        )

    def test_is_drugbank_id(self):
        self.assertTrue(ri.is_drugbank_id("DB00035"))
        self.assertTrue(ri.is_drugbank_id(" db00035 "))
        for value in ("DB0035", "DB000355", "neridronic-acid", "", None, "BTD00112"):
            self.assertFalse(ri.is_drugbank_id(value), repr(value))


# --------------------------------------------------------------------------
# the compound axis
# --------------------------------------------------------------------------


class TestCompoundIndexResolves(unittest.TestCase):
    def setUp(self):
        self.compounds = make_compounds()

    def test_drugbank_id_is_matched_as_an_identifier(self):
        resolution = self.compounds.resolve("DB00035")
        self.assertTrue(resolution.ok)
        self.assertEqual(resolution.drugbank_id, "DB00035")
        self.assertEqual(resolution.matched_by, "drugbank_id")

    def test_secondary_accession_is_matched(self):
        resolution = self.compounds.resolve("BIOD00112")
        self.assertTrue(resolution.ok)
        self.assertEqual(resolution.drugbank_id, "DB00035")
        self.assertEqual(resolution.matched_by, "accession_number")

    def test_name_slug_via_common_name(self):
        resolution = self.compounds.resolve("neridronic-acid")
        self.assertTrue(resolution.ok)
        self.assertEqual(resolution.drugbank_id, "DB11620")
        self.assertEqual(resolution.matched_by, "common_name")

    def test_name_slug_via_synonym(self):
        resolution = self.compounds.resolve("L-Citrulline")
        self.assertTrue(resolution.ok)
        self.assertEqual(resolution.drugbank_id, "DB00155")
        self.assertEqual(resolution.matched_by, "synonym")

    def test_hyphenated_and_numeric_slug(self):
        resolution = self.compounds.resolve("SN-38")
        self.assertTrue(resolution.ok)
        self.assertEqual(resolution.drugbank_id, "DB05482")


class TestCompoundIndexRefusesToGuess(unittest.TestCase):
    """AMBIGUOUS, UNKNOWN and NO_KEY are outcomes -- never a silent pick."""

    def setUp(self):
        self.compounds = make_compounds()

    def test_ambiguous_name_is_not_resolved(self):
        # THE case: 'L-Arginine' is a synonym of DB00125 and the common name of
        # DB17289. dict(zip(name, id)) would have kept whichever came last.
        resolution = self.compounds.resolve("L-Arginine")
        self.assertEqual(resolution.status, "AMBIGUOUS")
        self.assertIsNone(resolution.drugbank_id)
        self.assertEqual(resolution.candidates, ("DB00125", "DB17289"))
        self.assertIn("refusing to choose", resolution.reason)
        self.assertIn("DB00125 as synonym", resolution.reason)
        self.assertIn("DB17289 as common_name", resolution.reason)
        with self.assertRaises(InvariantViolation):
            resolution.unwrap()

    def test_candidate_order_is_total_not_hash_order(self):
        first = make_compounds().resolve("L-Arginine").candidates
        reversed_rows = list(reversed(VOCABULARY_ROWS))
        second = make_compounds(reversed_rows).resolve("L-Arginine").candidates
        self.assertEqual(first, second)

    def test_unknown_name_is_recorded_with_a_reason(self):
        resolution = self.compounds.resolve("clorprenaline")
        self.assertEqual(resolution.status, "UNKNOWN")
        self.assertIn("no fuzzy fallback", resolution.reason)
        self.assertIn("clorprenaline", resolution.reason)

    def test_drugbank_shaped_but_absent_is_unknown_not_passed_through(self):
        # DB00729 really is in node_1 and really is absent from the vocabulary.
        # Passing it through because it "looks like an id" would put a
        # non-existent compound into the corrected interaction set.
        resolution = self.compounds.resolve("DB00729")
        self.assertEqual(resolution.status, "UNKNOWN")
        self.assertIsNone(resolution.drugbank_id)
        self.assertIn("not in the vocabulary", resolution.reason)

    def test_null_token_is_no_key(self):
        for value in (None, "", "   ", pd.NA):
            resolution = self.compounds.resolve(value)
            self.assertEqual(resolution.status, "NO_KEY", repr(value))
            self.assertTrue(resolution.reason.strip())

    def test_punctuation_only_token_is_no_key(self):
        resolution = self.compounds.resolve("---")
        self.assertEqual(resolution.status, "NO_KEY")
        self.assertIn("normalises to nothing", resolution.reason)

    def test_prefer_common_name_is_opt_in_and_declares_what_it_outranked(self):
        default = make_compounds().resolve("L-Arginine")
        self.assertEqual(default.status, "AMBIGUOUS")
        opted_in = make_compounds(prefer_common_name=True).resolve("L-Arginine")
        self.assertTrue(opted_in.ok)
        self.assertEqual(opted_in.drugbank_id, "DB17289")
        self.assertIn("outranked", opted_in.reason)
        self.assertIn("DB00125", opted_in.reason)


class TestCompoundResolutionType(unittest.TestCase):
    """The type makes an unexplained rejection unconstructable."""

    def test_non_ok_requires_a_reason(self):
        with self.assertRaises(ValueError):
            ri.CompoundResolution(status="UNKNOWN", drugbank_id=None, reason="")

    def test_non_ok_may_not_carry_an_id(self):
        with self.assertRaises(ValueError):
            ri.CompoundResolution(
                status="AMBIGUOUS", drugbank_id="DB00035", reason="two of them"
            )

    def test_ok_requires_an_id_and_a_tier(self):
        with self.assertRaises(ValueError):
            ri.CompoundResolution(status="OK", drugbank_id=None)
        with self.assertRaises(ValueError):
            ri.CompoundResolution(status="OK", drugbank_id="DB00035", matched_by="vibes")

    def test_unknown_status_is_rejected(self):
        with self.assertRaises(ValueError):
            ri.CompoundResolution(status="MAYBE", drugbank_id=None, reason="?")

    def test_reject_row_carries_the_reason(self):
        row = make_compounds().resolve("L-Arginine").as_reject_row()
        self.assertEqual(row["status"], "AMBIGUOUS")
        self.assertEqual(row["candidates"], "DB00125;DB17289")
        self.assertTrue(row["reason"])


class TestCompoundIndexCannotCollapse(unittest.TestCase):
    """A last-wins ``dict(zip(names, ids))`` must be impossible here."""

    def test_the_name_relation_is_preserved_not_collapsed(self):
        compounds = make_compounds()
        candidates = compounds.name_candidates("L-Arginine")
        self.assertEqual(
            candidates, (("DB00125", "synonym"), ("DB17289", "common_name"))
        )
        # The relation is still auditable AFTER the index is built -- which is
        # exactly what dict(zip(...)) destroys. cardinality_report() sees the
        # collapse; a built dict would have reported max_collapse == 1.
        report = cardinality_report(
            {"l arginine": [c[0] for c in candidates]}, name="drug name -> id"
        )
        self.assertEqual(report["max_collapse"], 2)
        with self.assertRaises(InvariantViolation):
            assert_collapse_factor_one(
                {"l arginine": [c[0] for c in candidates]}, name="drug name -> id"
            )

    def test_every_vocabulary_row_survives_index_construction(self):
        compounds = make_compounds()
        self.assertEqual(len(compounds), len(VOCABULARY_ROWS))
        report = compounds.cardinality_report()
        self.assertEqual(report["n_drugbank_ids"], len(VOCABULARY_ROWS))
        self.assertEqual(report["n_duplicate_drugbank_ids"], 0)
        self.assertEqual(report["max_name_collapse"], 2)
        self.assertEqual(report["n_ambiguous_names"], 1)
        self.assertEqual(
            report["worst_names"],
            [{"name": "l arginine", "drugbank_ids": ["DB00125", "DB17289"]}],
        )

    def test_duplicate_drugbank_id_is_an_error_not_a_last_wins_overwrite(self):
        rows = list(VOCABULARY_ROWS) + [("DB00035", "", "Something else", "")]
        with self.assertRaises(InvariantViolation) as ctx:
            make_compounds(rows)
        self.assertIn("DB00035 x2", str(ctx.exception))

    def test_non_drugbank_id_in_the_id_column_is_rejected(self):
        rows = list(VOCABULARY_ROWS) + [("Desmopressin", "", "Desmopressin", "")]
        with self.assertRaises(InvariantViolation) as ctx:
            make_compounds(rows)
        self.assertIn("not DB+5", str(ctx.exception))


#: Two real within-tier collisions, both created by normalisation itself:
#: DB11344's common name is '(+)-menthol' and DB14123's is 'Menthol' -- both
#: normalise to 'menthol'; DB01001 lists the synonym 'Albuterol' and DB13139
#: lists '(-)-Albuterol' -- both normalise to 'albuterol'.
COLLIDING_VOCABULARY_ROWS = [
    ("DB11344", "", "(+)-menthol", "D-menthol"),
    ("DB14123", "", "Menthol", "DL-menthol | Menthol racemic"),
    ("DB01001", "", "Salbutamol", "Albuterol | Salbutamol | Salbutamolum"),
    ("DB13139", "", "Levosalbutamol", "(-)-Albuterol | Levalbuterol"),
]


class TestWithinTierCollapse(unittest.TestCase):
    """A collision *inside* one tier must survive index construction.

    Storing ``name -> id`` instead of ``name -> (ids...)`` keeps the last entry
    per name and the loss becomes unobservable -- ``batch_dock.py:2034``
    verbatim, on the compound axis. These two collisions are created by
    normalisation itself, which is the case a per-tier scalar map hides best.
    """

    def setUp(self):
        self.compounds = make_compounds(COLLIDING_VOCABULARY_ROWS)

    def test_two_common_names_normalising_alike_are_ambiguous(self):
        resolution = self.compounds.resolve("menthol")
        self.assertEqual(resolution.status, "AMBIGUOUS")
        self.assertEqual(resolution.candidates, ("DB11344", "DB14123"))

    def test_two_synonyms_normalising_alike_are_ambiguous(self):
        resolution = self.compounds.resolve("albuterol")
        self.assertEqual(resolution.status, "AMBIGUOUS")
        self.assertEqual(resolution.candidates, ("DB01001", "DB13139"))

    def test_the_collision_is_visible_in_the_cardinality_report(self):
        report = self.compounds.cardinality_report()
        self.assertEqual(report["max_common_name_collapse"], 2)
        self.assertEqual(report["max_synonym_collapse"], 2)
        self.assertEqual(report["n_ambiguous_names"], 2)
        self.assertEqual(report["n_drugbank_ids"], len(COLLIDING_VOCABULARY_ROWS))

    def test_prefer_common_name_cannot_break_a_tie_inside_the_common_tier(self):
        compounds = make_compounds(COLLIDING_VOCABULARY_ROWS, prefer_common_name=True)
        self.assertEqual(compounds.resolve("menthol").status, "AMBIGUOUS")
        self.assertEqual(compounds.resolve("albuterol").status, "AMBIGUOUS")

    def test_an_edge_on_a_colliding_name_is_rejected_not_resolved(self):
        edges = edges_frame([("menthol", "554", None, "AVPR2")])
        result = run(
            edges=edges,
            compounds=make_compounds(COLLIDING_VOCABULARY_ROWS),
            small=make_small_molecules(["DB11344", "DB14123"]),
        )
        self.assertEqual(len(result.resolved), 0)
        self.assertEqual(result.rejects.loc[0, "status"], "AMBIGUOUS_DRUG")
        self.assertEqual(result.rejects.loc[0, "candidates"], "DB11344;DB14123")


class TestCompoundIndexLoading(unittest.TestCase):
    def test_a_directory_holding_one_csv_is_accepted(self):
        # The export on disk unpacks to a DIRECTORY named
        # 'drugbank_all_drugbank_vocabulary.csv' containing one CSV.
        with tempfile.TemporaryDirectory() as tmp:
            directory = os.path.join(tmp, "drugbank_all_drugbank_vocabulary.csv")
            os.makedirs(directory)
            path = os.path.join(directory, "drugbank vocabulary.csv")
            vocabulary_frame().to_csv(path, index=False)
            compounds = ri.CompoundIndex.load(directory)
            self.assertEqual(len(compounds), len(VOCABULARY_ROWS))
            self.assertEqual(compounds.source_path, path)

    def test_a_directory_holding_two_csvs_raises_rather_than_choosing(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = os.path.join(tmp, "vocab")
            os.makedirs(directory)
            for name in ("a.csv", "b.csv"):
                vocabulary_frame().to_csv(os.path.join(directory, name), index=False)
            with self.assertRaises(FileNotFoundError) as ctx:
                ri.CompoundIndex.load(directory)
            self.assertIn("Choosing one would be a guess", str(ctx.exception))

    def test_a_missing_vocabulary_raises_and_does_not_degrade(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            ri.CompoundIndex.load("/nonexistent/vocabulary.csv")
        self.assertIn("no fallback name source", str(ctx.exception))

    def test_a_missing_column_is_an_error_not_a_default(self):
        frame = vocabulary_frame().drop(columns=["Common name"])
        with self.assertRaises(InvariantViolation) as ctx:
            ri.CompoundIndex.from_frame(frame)
        self.assertIn("Common name", str(ctx.exception))

    def test_optional_columns_narrow_the_index_they_never_widen_it(self):
        frame = vocabulary_frame().drop(columns=["Synonyms", "Accession Numbers"])
        compounds = ri.CompoundIndex.from_frame(frame)
        self.assertTrue(compounds.resolve("neridronic-acid").ok)
        # Without the synonym column, 'L-Citrulline' is UNKNOWN -- recorded with
        # a reason. It is never resolved to something else.
        self.assertEqual(compounds.resolve("L-Citrulline").status, "UNKNOWN")
        # ... and 'L-Arginine' stops being ambiguous, because only DB17289's
        # common name remains. Narrower, never wider.
        self.assertEqual(compounds.resolve("L-Arginine").drugbank_id, "DB17289")


# --------------------------------------------------------------------------
# the protein axis -- the regression the whole rebuild exists for
# --------------------------------------------------------------------------


class TestProteinAxisRegression(unittest.TestCase):
    def test_entrez_554_resolves_to_P30518_and_never_to_Q8NCT1(self):
        result = run()
        avpr2 = result.resolved[result.resolved["entrez_id"] == 554]
        self.assertEqual(sorted(set(avpr2["accession"])), ["P30518"])
        self.assertEqual(sorted(set(avpr2["gene"])), ["AVPR2"])
        self.assertNotIn("Q8NCT1", set(result.resolved["accession"]))

    def test_node_2_name_is_never_consulted(self):
        # Edge 2 says node_2_name='ARRDC4' while node_2=10381 (TUBB3). Following
        # the name would dock Q8NCT1; following the key gives Q13509.
        result = run()
        row = result.resolved[
            (result.resolved["drugbank_id"] == "DB01169")
            & (result.resolved["entrez_id"] == 10381)
        ]
        self.assertEqual(list(row["accession"]), ["Q13509"])
        self.assertEqual(list(row["gene"]), ["TUBB3"])

    def test_corrupting_every_symbol_column_changes_nothing(self):
        clean = run().resolved
        corrupted_rows = [(r[0], r[1], r[2], "ARRDC4") for r in EDGE_ROWS]
        corrupted = run(edges=edges_frame(corrupted_rows)).resolved
        pd.testing.assert_frame_equal(clean, corrupted)

    def test_an_edge_with_a_null_symbol_still_resolves(self):
        # 237 real edges have no node_2_name at all. A symbol-keyed resolver
        # could not reach them even in principle; this one resolves them from
        # node_2 alone.
        result = run()
        row = result.resolved[
            (result.resolved["drugbank_id"] == "DB00035")
            & (result.resolved["entrez_id"] == 10376)
        ]
        self.assertEqual(list(row["accession"]), ["P68363"])
        self.assertEqual(list(row["gene"]), ["TUBA1B"])
        self.assertEqual(result.summary["n_edges_resolved_with_null_node_2_name"], 1)

    def test_the_gene_column_is_a_label_derived_from_the_key(self):
        # The KNOWN_ISSUE.md regression assertion, on our own output: every
        # (accession, gene) pair must satisfy gene in index.by_accession(acc).symbols.
        index = make_index()
        result = run(index=index)
        for row in result.resolved.to_dict("records"):
            self.assertIn(row["gene"], index.by_accession(row["accession"]).symbols)
            self.assertEqual(row["gene"], index.label(row["accession"]))

    def test_entrez_to_accession_collapse_factor_is_exactly_one(self):
        resolved = run().resolved
        report = assert_collapse_factor_one(
            resolved[["entrez_id", "accession"]].drop_duplicates(),
            name="entrez -> accession",
            key="entrez_id",
            value="accession",
        )
        self.assertEqual(report["max_collapse"], 1)


class TestProteinAxisRejects(unittest.TestCase):
    def rejects(self, **kwargs):
        return run(**kwargs).rejects.set_index("edge_row")

    def test_missing_entrez_is_no_key(self):
        self.assertEqual(self.rejects().loc[10, "status"], "NO_KEY")

    def test_unknown_entrez_is_recorded_not_dropped(self):
        row = self.rejects().loc[11]
        self.assertEqual(row["status"], "UNKNOWN_ENTREZ")
        self.assertIn("999999999", row["reason"])

    def test_non_numeric_entrez_is_no_key_not_a_crash(self):
        rows = [("DB00035", "ATP1A1", "Desmopressin", "ATP1A1")]
        result = run(edges=edges_frame(rows))
        self.assertEqual(len(result.resolved), 0)
        self.assertEqual(result.rejects.loc[0, "status"], "NO_KEY")
        self.assertIn("not numeric", result.rejects.loc[0, "reason"])

    def test_obsolete_entrez_is_distinguished_from_unknown(self):
        index = make_index(obsolete_entrez={999999999: "withdrawn by NCBI"})
        row = self.rejects(index=index).loc[11]
        self.assertEqual(row["status"], "OBSOLETE_ENTREZ")
        self.assertIn("withdrawn by NCBI", row["reason"])

    def test_a_collapsed_index_produces_AMBIGUOUS_ENTREZ_not_a_pick(self):
        # An index in which one Entrez id carries two accessions is the exact
        # shape of the 2025 map (mean collapse 2.55, max 9). Resolution must
        # refuse, not choose -- choosing is what selected the receptor.
        rows = list(INDEX_ROWS) + [
            ("Q9BQE5", "APOL2_HUMAN", "APOL2", ["APOL2"], 554, True)
        ]
        index = make_index(rows, validate=False)
        result = run(index=index)
        rejects = result.rejects.set_index("edge_row")
        self.assertEqual(rejects.loc[0, "status"], "AMBIGUOUS_ENTREZ")
        self.assertEqual(rejects.loc[0, "candidates"], "P30518;Q9BQE5")
        self.assertNotIn("Q9BQE5", set(result.resolved["accession"]))
        self.assertNotIn(554, set(result.resolved["entrez_id"]))


# --------------------------------------------------------------------------
# total accounting -- nothing vanishes
# --------------------------------------------------------------------------


class TestTotalAccounting(unittest.TestCase):
    def test_n_in_equals_n_resolved_plus_n_rejected(self):
        result = run()
        summary = result.summary
        self.assertEqual(summary["n_edges_in"], len(EDGE_ROWS))
        self.assertEqual(
            summary["n_edges_in"],
            summary["n_resolved_edges"] + summary["n_rejected_edges"],
        )
        self.assertTrue(summary["ok"])

    def test_every_input_row_is_traceable_to_an_output_row_or_a_reject(self):
        result = run()
        accounted = set(int(v) for v in result.rejects["edge_row"])
        self.assertEqual(len(accounted), len(result.rejects))
        for selected_by in result.resolved["selected_by"]:
            marker = "edge_rows="
            rows = selected_by.split(marker, 1)[1]
            for value in rows.split(","):
                edge_row = int(value)
                self.assertNotIn(edge_row, accounted, "edge counted twice")
                accounted.add(edge_row)
        self.assertEqual(accounted, set(range(len(EDGE_ROWS))))

    def test_every_reject_carries_a_reason_and_a_declared_status(self):
        result = run()
        self.assertTrue(len(result.rejects))
        for row in result.rejects.to_dict("records"):
            self.assertTrue(str(row["reason"]).strip(), row)
            self.assertIn(row["status"], ri.REJECT_PRECEDENCE, row)
            self.assertIn(row["axis"], ("compound", "protein", "both", "edge"), row)

    def test_an_edge_failing_on_both_axes_reports_both(self):
        rows = [("clorprenaline", "999999999", None, None)]
        result = run(edges=edges_frame(rows))
        reject = result.rejects.loc[0]
        self.assertEqual(reject["axis"], "both")
        self.assertIn("node_1:", reject["reason"])
        self.assertIn("node_2:", reject["reason"])
        # The primary status is the first applicable member of the declared
        # precedence, so it is a total function of the failures, not of order.
        self.assertEqual(reject["status"], "UNKNOWN_ENTREZ")

    def test_the_accounting_assertion_actually_fires(self):
        # Prove the guard is live: an edge frame whose edge_row column lies
        # about its size must not slip through.
        edges = edges_frame()
        with self.assertRaises(InvariantViolation) as ctx:
            ri.resolve_interactions(
                edges.drop(columns=["edge_row"]),
                make_index(),
                make_compounds(),
                make_small_molecules(),
            )
        self.assertIn("edge_row", str(ctx.exception))

    def test_an_edge_of_the_wrong_node_type_is_rejected(self):
        edges = edges_frame()
        edges.loc[0, "node_2_type"] = "drug"
        result = run(edges=edges)
        self.assertEqual(result.rejects.set_index("edge_row").loc[0, "status"],
                         "WRONG_EDGE_TYPE")
        self.assertEqual(
            result.summary["n_edges_in"],
            result.summary["n_resolved_edges"] + result.summary["n_rejected_edges"],
        )


# --------------------------------------------------------------------------
# the small-molecule gate
# --------------------------------------------------------------------------


class TestSmallMoleculeGate(unittest.TestCase):
    def test_a_non_small_molecule_is_rejected_with_a_reason(self):
        result = run()
        reject = result.rejects.set_index("edge_row").loc[9]
        self.assertEqual(reject["status"], "NOT_SMALL_MOLECULE")
        self.assertIn("DB00104", reject["reason"])
        self.assertNotIn("DB00104", set(result.resolved["drugbank_id"]))

    def test_the_gate_may_not_duplicate_edges(self):
        # A duplicated DrugBank ID on the right-hand side of a left merge fans
        # out -- the parse_prepare.py:1171 failure mode (+410,087 rows). It must
        # raise, naming the duplicated key, not silently double the edges.
        duplicated = make_small_molecules(SMALL_MOLECULE_IDS + ["DB00035"])
        with self.assertRaises(InvariantViolation) as ctx:
            run(small=duplicated)
        message = str(ctx.exception)
        self.assertIn("merge changed the row count", message)
        self.assertIn("DB00035", message)

    def test_row_count_is_preserved_when_the_gate_passes(self):
        before = run(small=make_small_molecules(SMALL_MOLECULE_IDS + ["DB00104"]))
        # With DB00104 admitted, exactly one more edge survives and no edge is
        # duplicated.
        baseline = run()
        self.assertEqual(
            before.summary["n_resolved_edges"],
            baseline.summary["n_resolved_edges"] + 1,
        )

    def test_the_drug_links_file_must_declare_the_drug_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "links.csv")
            pd.DataFrame({"DrugBank ID": SMALL_MOLECULE_IDS}).to_csv(path, index=False)
            with self.assertRaises(InvariantViolation) as ctx:
                ri.load_small_molecule_ids(path)
            self.assertIn("Drug Type", str(ctx.exception))

    def test_only_small_molecule_rows_pass_the_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "links.csv")
            pd.DataFrame(
                {
                    "DrugBank ID": ["DB00035", "DB00104"],
                    "Drug Type": ["SmallMoleculeDrug", "BiotechDrug"],
                }
            ).to_csv(path, index=False)
            ids, report = ri.load_small_molecule_ids(path)
            self.assertEqual(list(ids["drugbank_id"]), ["DB00035"])
            self.assertEqual(report["n_other_drug_type_rows"], 1)

    def test_a_missing_drug_links_file_raises(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            ri.load_small_molecule_ids("/nonexistent/links.csv")
        self.assertIn("no fallback", str(ctx.exception))


# --------------------------------------------------------------------------
# collapsing onto the declared primary key
# --------------------------------------------------------------------------


class TestDeduplication(unittest.TestCase):
    def test_a_slug_and_an_id_for_the_same_pair_collapse_to_one_row(self):
        # 38 real edges are this shape: node_1 is 'L-Citrulline' on one row and
        # 'DB00155' on another, both pointing at the same Entrez id.
        result = run()
        rows = result.resolved[
            (result.resolved["drugbank_id"] == "DB00155")
            & (result.resolved["accession"] == "P00966")
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(result.summary["n_duplicate_edges_collapsed"], 1)
        selected_by = rows.iloc[0]["selected_by"]
        self.assertIn("edge_rows=4,5", selected_by)
        self.assertIn("DB00155,L-Citrulline", selected_by)

    def test_the_primary_key_is_unique(self):
        resolved = run().resolved
        self.assertFalse(resolved.duplicated(["drugbank_id", "accession"]).any())

    def test_selected_by_records_why_the_row_exists(self):
        resolved = run().resolved
        row = resolved[
            (resolved["drugbank_id"] == "DB00035")
            & (resolved["accession"] == "P30518")
        ].iloc[0]
        self.assertEqual(
            row["selected_by"],
            "entrez=554 -> P30518; node_1=DB00035 via drugbank_id; edge_rows=0",
        )
        self.assertTrue((resolved["selected_by"].str.len() > 0).all())

    def test_two_entrez_ids_on_one_accession_is_a_conflict_not_a_pick(self):
        # Only reachable through an index that violates G1/G5. It must raise
        # rather than keep whichever row sorted first.
        rows = list(INDEX_ROWS) + [
            ("P30518", "V2R_HUMAN", "AVPR2", ["AVPR2"], 12345, True)
        ]
        index = make_index(rows, validate=False)
        edges = edges_frame(
            [
                ("DB00035", "554", "Desmopressin", "AVPR2"),
                ("DB00035", "12345", "Desmopressin", "AVPR2"),
            ]
        )
        with self.assertRaises(InvariantViolation) as ctx:
            run(edges=edges, index=index)
        self.assertIn("collapse factor 1", str(ctx.exception))


# --------------------------------------------------------------------------
# the output contract
# --------------------------------------------------------------------------


class TestOutputContract(unittest.TestCase):
    def test_the_output_satisfies_the_declared_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out_path, _ = cli(tmp)
            self.assertEqual(rc, 0)
            frame = read_checked(out_path, "interactions_resolved")
            self.assertEqual(
                list(frame.columns), SCHEMAS["interactions_resolved"].column_names
            )
            self.assertEqual(str(frame["entrez_id"].dtype), "int64")
            self.assertFalse(frame.isna().any().any())

    def test_the_rejects_ledger_is_written_even_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean = [("DB00035", "554", "Desmopressin", "AVPR2")]
            rc, out_path, rejects_path = cli(tmp, edges=edges_frame(clean))
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(rejects_path))
            frame = pd.read_csv(rejects_path)
            self.assertEqual(len(frame), 0)
            self.assertEqual(list(frame.columns), list(ri.REJECT_COLUMNS))

    def test_the_invariant_report_records_that_the_checks_ran(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out_path, _ = cli(tmp)
            self.assertEqual(rc, 0)
            report_path = os.path.splitext(out_path)[0] + ".invariants.json"
            with open(report_path, "r", encoding="utf-8") as handle:
                reports = json.load(handle)
            checks = {r.get("check") for r in reports}
            for expected in (
                "total_accounting",
                "collapse_factor_one",
                "self_consistent_labels",
                "unique_key",
                "checked_merge",
                "schema",
                "compound_index_cardinality",
                "resolve_interactions summary",
            ):
                self.assertIn(expected, checks)
            self.assertTrue(all(r.get("ok", True) for r in reports))

    def test_the_rejects_ledger_names_the_line_it_came_from(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, _, rejects_path = cli(tmp)
            self.assertEqual(rc, 0)
            frame = pd.read_csv(rejects_path)
            self.assertEqual(list(frame["edge_row"]), sorted(frame["edge_row"]))
            self.assertTrue(set(frame["edge_row"]).issubset(set(range(len(EDGE_ROWS)))))


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


class TestDeterminism(unittest.TestCase):
    def test_shuffling_the_input_rows_does_not_change_the_output(self):
        order = [11, 3, 7, 0, 9, 2, 6, 10, 4, 1, 8, 5]
        shuffled = edges_frame(order=order)
        self.assertNotEqual(
            list(edges_frame()["node_1"]), list(shuffled["node_1"])
        )  # the input really is different
        baseline = run().resolved.drop(columns=["selected_by"])
        permuted = run(edges=shuffled).resolved.drop(columns=["selected_by"])
        pd.testing.assert_frame_equal(baseline, permuted)

    def test_the_output_is_ordered_by_its_primary_key(self):
        # The declared row order. Any other order is still a valid manifest but
        # is no longer byte-reproducible from a re-ordered input file.
        resolved = run().resolved
        expected = resolved.sort_values(
            ["drugbank_id", "accession"], kind="mergesort"
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(resolved, expected)

    def test_two_runs_produce_byte_identical_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out_path, rejects_path = cli(tmp)
            self.assertEqual(rc, 0)
            first = (sha256_of(out_path), sha256_of(rejects_path))
            rc, out_path, rejects_path = cli(tmp)
            self.assertEqual(rc, 0)
            self.assertEqual(first, (sha256_of(out_path), sha256_of(rejects_path)))

    def test_output_is_identical_across_PYTHONHASHSEED_values(self):
        # prepare_uniprot_mapping.py:51 did list(set(...)) and let the resulting
        # order decide which protein owned an accession. That is invisible in a
        # single process and trivially visible here.
        #
        # All three subprocesses share ONE fixture directory on purpose: reject
        # reasons quote the index and vocabulary paths (deliberately -- "which
        # index said this?"), so a per-process temp directory would make the
        # ledgers differ for a reason that has nothing to do with hash order.
        digests = []
        seeds = ("0", "1", "42")
        hash_probes = []
        with tempfile.TemporaryDirectory() as shared:
            for seed in seeds:
                env = dict(os.environ)
                env["PYTHONHASHSEED"] = seed
                env["REPO_ROOT"] = REPO_ROOT
                env["FIXTURE_DIR"] = shared
                completed = subprocess.run(
                    [sys.executable, "-c", _HASH_SEED_SNIPPET],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    completed.stderr.decode("utf-8", "replace")[-2000:],
                )
                payload = json.loads(
                    completed.stdout.decode("utf-8").strip().splitlines()[-1]
                )
                digests.append((payload["out"], payload["rejects"]))
                hash_probes.append(payload["hash_probe"])
        # The seeds really did differ, so equal output means order-independence
        # rather than an inert test.
        self.assertEqual(len(set(hash_probes)), len(seeds), hash_probes)
        self.assertEqual(len(set(digests)), 1, digests)

    def test_the_reject_ledger_is_stable_for_a_fixed_set_of_paths(self):
        # Companion to the hash-seed test: with the paths held fixed, two runs
        # in the same directory must produce identical ledgers.
        with tempfile.TemporaryDirectory() as tmp:
            rc, _, rejects_path = cli(tmp)
            self.assertEqual(rc, 0)
            first = sha256_of(rejects_path)
            rc, _, rejects_path = cli(tmp)
            self.assertEqual(rc, 0)
            self.assertEqual(first, sha256_of(rejects_path))


_HASH_SEED_SNIPPET = r'''
import hashlib, json, os, sys, logging
sys.path.insert(0, os.environ["REPO_ROOT"])
sys.path.insert(0, os.path.join(os.environ["REPO_ROOT"], "tests"))
logging.disable(logging.CRITICAL)
import test_resolve_interactions as t

rc, out_path, rejects_path = t.cli(os.environ["FIXTURE_DIR"])
assert rc == 0, rc
print(json.dumps({
    "out": t.sha256_of(out_path),
    "rejects": t.sha256_of(rejects_path),
    "hash_probe": hash("AVPR2"),
}))
'''


# --------------------------------------------------------------------------
# the CLI: a missing prerequisite is an error, never a fallback
# --------------------------------------------------------------------------


class TestCliFailsClosed(unittest.TestCase):
    def _argv(self, tmp, **overrides):
        edges_path, index_path, vocabulary_path, drugs_path = write_inputs(tmp)
        argv = {
            "--edges": edges_path,
            "--index": index_path,
            "--drugs": drugs_path,
            "--vocabulary": vocabulary_path,
            "--out": os.path.join(tmp, "interactions_resolved.csv"),
            "--rejects": os.path.join(tmp, "unresolved_targets.csv"),
        }
        argv.update(overrides)
        flat = []
        for key, value in argv.items():
            flat.extend([key, value])
        return flat

    def _run(self, argv):
        logging.disable(logging.CRITICAL)
        try:
            return ri.main(argv)
        finally:
            logging.disable(logging.NOTSET)

    def test_a_missing_identity_index_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = self._argv(tmp, **{"--index": os.path.join(tmp, "absent.parquet")})
            self.assertEqual(self._run(argv), 2)
            self.assertFalse(
                os.path.exists(os.path.join(tmp, "interactions_resolved.csv"))
            )

    def test_a_missing_vocabulary_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = self._argv(tmp, **{"--vocabulary": os.path.join(tmp, "absent.csv")})
            self.assertEqual(self._run(argv), 2)

    def test_a_missing_edge_file_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = self._argv(tmp, **{"--edges": os.path.join(tmp, "absent.tsv")})
            self.assertEqual(self._run(argv), 2)

    def test_an_index_sha256_mismatch_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = self._argv(tmp) + ["--index-sha256", "0" * 64]
            self.assertEqual(self._run(argv), 2)

    def test_both_InvariantViolation_classes_are_caught(self):
        # identity.py defines its OWN InvariantViolation instead of importing
        # the canonical one from invariants.py, so the two are different
        # classes and one `except` clause catches only one of them. A pinned
        # SHA256 mismatch raises identity's; checked_merge and the schemas raise
        # invariants'. Both must reach the same non-zero exit, or a stage
        # crashes with a traceback where it should report a failed gate.
        #
        # This assertion is written so that it keeps passing (and stays
        # meaningful) once identity.py adopts the canonical exception.
        from dock_selectivity import identity as identity_module
        from dock_selectivity import invariants as invariants_module

        self.assertIn(identity_module.InvariantViolation, ri.INVARIANT_ERRORS)
        self.assertIn(invariants_module.InvariantViolation, ri.INVARIANT_ERRORS)
        with tempfile.TemporaryDirectory() as tmp:
            # identity's exception (pinned SHA mismatch)
            self.assertEqual(
                self._run(self._argv(tmp) + ["--index-sha256", "0" * 64]), 2
            )
        with tempfile.TemporaryDirectory() as tmp:
            # invariants' exception (a duplicated key in the drug links file
            # would fan the merge out)
            duplicated = pd.DataFrame(
                {
                    "DrugBank ID": SMALL_MOLECULE_IDS + ["DB00035"],
                    "Drug Type": ["SmallMoleculeDrug"] * (len(SMALL_MOLECULE_IDS) + 1),
                }
            )
            edges_path, index_path, vocabulary_path, drugs_path = write_inputs(
                tmp, drugs=duplicated
            )
            out = os.path.join(tmp, "interactions_resolved.csv")
            self.assertEqual(
                self._run(
                    [
                        "--edges", edges_path,
                        "--index", index_path,
                        "--drugs", drugs_path,
                        "--vocabulary", vocabulary_path,
                        "--out", out,
                    ]
                ),
                2,
            )
            self.assertFalse(os.path.exists(out))

    def test_the_pinned_index_sha256_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = self._argv(tmp)
            digest = sha256_of(argv[argv.index("--index") + 1])
            self.assertEqual(self._run(argv + ["--index-sha256", digest]), 0)

    def test_rejects_may_not_overwrite_the_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "interactions_resolved.csv")
            argv = self._argv(tmp, **{"--rejects": out})
            self.assertEqual(self._run(argv), 2)

    def test_every_input_flag_is_required(self):
        parser = ri.build_arg_parser()
        for missing in ("--edges", "--index", "--drugs", "--out"):
            argv = []
            for flag in ("--edges", "--index", "--drugs", "--out"):
                if flag != missing:
                    argv.extend([flag, "x"])
            with self.assertRaises(SystemExit):
                with open(os.devnull, "w") as devnull:
                    stderr, sys.stderr = sys.stderr, devnull
                    try:
                        parser.parse_args(argv)
                    finally:
                        sys.stderr = stderr

    def test_a_broken_vocabulary_aborts_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken = vocabulary_frame(
                list(VOCABULARY_ROWS) + [("DB00035", "", "Duplicate", "")]
            )
            edges_path, index_path, vocabulary_path, drugs_path = write_inputs(
                tmp, vocabulary=broken
            )
            out = os.path.join(tmp, "interactions_resolved.csv")
            argv = [
                "--edges", edges_path,
                "--index", index_path,
                "--drugs", drugs_path,
                "--vocabulary", vocabulary_path,
                "--out", out,
            ]
            self.assertEqual(self._run(argv), 2)
            self.assertFalse(os.path.exists(out))


class TestReadEdges(unittest.TestCase):
    def test_literal_NA_and_nan_symbols_are_not_read_as_missing(self):
        # pandas' defaults turn 'NA', 'nan', 'null' and 'None' into NaN. 'NA'
        # is a real gene symbol, and 'nan' is a real token in 11,806 old job
        # directory names.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "edges.tsv")
            pd.DataFrame(
                {
                    "node_1": ["DB00035", "DB00035"],
                    "node_2": ["554", "10376"],
                    "node_2_name": ["NA", "nan"],
                }
            ).to_csv(path, sep="\t", index=False)
            frame = ri.read_edges(path)
            self.assertEqual(list(frame["node_2_name"]), ["NA", "nan"])
            self.assertEqual(list(frame["edge_row"]), [0, 1])

    def test_optional_name_columns_may_be_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "edges.tsv")
            pd.DataFrame(
                {"node_1": ["DB00035"], "node_2": ["554"]}
            ).to_csv(path, sep="\t", index=False)
            frame = ri.read_edges(path)
            self.assertIn("node_2_name", frame.columns)
            self.assertTrue(frame["node_2_name"].isna().all())

    def test_a_missing_key_column_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "edges.tsv")
            pd.DataFrame(
                {"node_1": ["DB00035"], "node_2_name": ["AVPR2"]}
            ).to_csv(path, sep="\t", index=False)
            with self.assertRaises(InvariantViolation) as ctx:
                ri.read_edges(path)
            self.assertIn("node_2", str(ctx.exception))

    def test_a_missing_edge_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            ri.read_edges("/nonexistent/1_drug_to_protein.tsv")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
