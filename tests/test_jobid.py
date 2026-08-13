"""Tests for dock_selectivity.jobid.

The job-directory name was the pipeline's only carrier of identity and it broke
in three ways at once.  Every legacy name in ``REAL_LEGACY_DIRS`` below is a
verbatim directory name from
``$OLD/consensus_docking_results`` (copied as a literal -- this test never
touches the drive), chosen to cover all the shapes the old regex
``r'(DB\\d+)_([A-Za-z0-9]+)_([A-Z0-9]+)_cavity_(\\d+)'`` rejected:

* 1,743 multi-fragment names ``..._-F13-model_v1_cavity_10``;
* 34 hyphenated symbols ``MT-ND1``;
* 11,806 names whose gene token is the literal string ``nan``.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dock_selectivity import jobid  # noqa: E402
from dock_selectivity.jobid import (  # noqa: E402
    DELIM,
    JOB_IDENTITY_FILENAME,
    JOB_IDENTITY_SCHEMA_VERSION,
    NULL_GENE,
    JobIdentity,
    JobIdError,
    JobIdParseError,
    format_job_dir,
    normalize_fragment_id,
    parse_job_dir,
    parse_legacy_job_dir,
    read_job_identity,
    read_job_identity_record,
    write_job_identity,
)

# Accessions that appear in the real directory names below, plus a few more.
KNOWN_ACCESSIONS = (
    "P30518",
    "Q8NCT1",
    "P68363",
    "Q13509",
    "P60568",
    "P01589",
    "P03886",
    "P03905",
    "Q9NWH2",
    "P08F94",
    "Q96PZ7",
    "P08173",
    "P21359",
    "P98164",
    "P04150",
    "P47901",
    "P08588",
    "A0A0B4J2D5",
)

# name -> (drugbank_id, accession, gene, fragment_id, cavity_index)
REAL_LEGACY_DIRS = {
    # the plain majority
    "DB00035_AVPR2_Q8NCT1_cavity_2": ("DB00035", "Q8NCT1", "AVPR2", None, 2),
    "DB00253_NR3C1_P04150_cavity_7": ("DB00253", "P04150", "NR3C1", None, 7),
    "DB00035_AVPR1B_P47901_cavity_1": ("DB00035", "P47901", "AVPR1B", None, 1),
    "DB00397_ADRB1_P08588_cavity_1": ("DB00397", "P08588", "ADRB1", None, 1),
    # gene token is literally 'nan' (11,806 dirs)
    "DB00035_nan_P08173_cavity_3": ("DB00035", "P08173", None, None, 3),
    # hyphenated symbols (34 dirs)
    "DB00157_MT-ND1_P03886_cavity_1": ("DB00157", "P03886", "MT-ND1", None, 1),
    "DB00157_MT-ND4_P03905_cavity_2": ("DB00157", "P03905", "MT-ND4", None, 2),
    "DB00157_MT-ND2_Q9NWH2_cavity_1": ("DB00157", "Q9NWH2", "MT-ND2", None, 1),
    # legacy multi-fragment names (1,743 dirs), incl. two-digit fragment and cavity
    "DB00091_CAMLG_P08F94_-F13-model_v1_cavity_10": (
        "DB00091",
        "P08F94",
        "CAMLG",
        "F13",
        10,
    ),
    "DB00091_CAMLG_P08F94_-F4-model_v1_cavity_1": (
        "DB00091",
        "P08F94",
        "CAMLG",
        "F4",
        1,
    ),
    "DB00091_nan_Q96PZ7_-F7-model_v1_cavity_6": ("DB00091", "Q96PZ7", None, "F7", 6),
    "DB00114_nan_P21359_-F2-model_v1_cavity_3": ("DB00114", "P21359", None, "F2", 3),
    "DB00890_SHBG_P98164_-F2-model_v1_cavity_11": (
        "DB00890",
        "P98164",
        "SHBG",
        "F2",
        11,
    ),
    # the other documented AlphaFold form (batch_dock.py:2207 only stripped -F1-)
    "DB00091_CAMLG_P08F94_AF-P08F94-F13-model_v1_cavity_10": (
        "DB00091",
        "P08F94",
        "CAMLG",
        "F13",
        10,
    ),
}


class TestNewSchemeFormat(unittest.TestCase):
    def test_the_specified_name(self):
        self.assertEqual(
            format_job_dir("DB00035", "P30518", "AVPR2", "F1", 2),
            "DB00035__P30518__AVPR2__F1__cav2",
        )

    def test_accession_comes_before_the_gene(self):
        name = format_job_dir("DB00035", "P30518", "AVPR2", "F1", 2)
        self.assertLess(name.index("P30518"), name.index("AVPR2"))

    def test_the_accession_slot_cannot_hold_a_gene_symbol(self):
        # The structural guarantee: a symbol is not an identity key, so it can
        # never be written where an accession belongs.
        with self.assertRaises(JobIdError) as ctx:
            format_job_dir("DB00035", "AVPR2", "AVPR2", "F1", 2)
        self.assertIn("not a canonical UniProtKB accession", str(ctx.exception))
        with self.assertRaises(JobIdError):
            format_job_dir("DB00035", "P30518-2", "AVPR2", "F1", 2)
        with self.assertRaises(JobIdError):
            format_job_dir("DB00035", "V2R_HUMAN", "AVPR2", "F1", 2)

    def test_fragment_is_required_and_never_inferred(self):
        with self.assertRaises(JobIdError) as ctx:
            format_job_dir("DB00035", "P30518", "AVPR2", None, 2)
        message = str(ctx.exception)
        self.assertIn("fragment_id is required", message)
        self.assertIn("F1", message)

    def test_the_string_nan_is_never_written_into_a_name(self):
        for gene in (None, "nan", "NaN", "None", "", float("nan"), NULL_GENE):
            with self.subTest(gene=repr(gene)):
                name = format_job_dir("DB00035", "P30518", gene, "F1", 2)
                self.assertEqual(name, "DB00035__P30518__-__F1__cav2")
                self.assertIsNone(parse_job_dir(name).gene)

    def test_fragment_normalisation_closes_the_f1_vs_1_join_trap(self):
        self.assertEqual(normalize_fragment_id("F1"), "F1")
        self.assertEqual(normalize_fragment_id("1"), "F1")
        self.assertEqual(normalize_fragment_id(1), "F1")
        self.assertEqual(normalize_fragment_id("f1"), "F1")
        self.assertEqual(normalize_fragment_id(1.0), "F1")
        self.assertIsNone(normalize_fragment_id(None))
        self.assertIsNone(normalize_fragment_id("nan"))
        self.assertIsNone(normalize_fragment_id(float("nan")))
        with self.assertRaises(JobIdError):
            normalize_fragment_id("FX")
        names = {
            format_job_dir("DB00035", "P30518", "AVPR2", fragment, 2)
            for fragment in ("F1", "1", 1, "f1")
        }
        self.assertEqual(len(names), 1)

    def test_fields_that_would_not_round_trip_are_rejected(self):
        with self.assertRaises(JobIdError):
            format_job_dir("DB0__1", "P30518", "AVPR2", "F1", 2)
        with self.assertRaises(JobIdError):
            format_job_dir("DB00035", "P30518", "A__B", "F1", 2)
        with self.assertRaises(JobIdError):
            format_job_dir("DB00035", "P30518", "A_", "F1", 2)
        with self.assertRaises(JobIdError):
            format_job_dir("DB00035", "P30518", "_A", "F1", 2)
        with self.assertRaises(JobIdError):
            format_job_dir("DB00035", "P30518", "A B", "F1", 2)
        with self.assertRaises(JobIdError):
            format_job_dir("", "P30518", "AVPR2", "F1", 2)
        with self.assertRaises(JobIdError):
            format_job_dir("DB00035", "P30518", "AVPR2", "F1", -1)
        with self.assertRaises(JobIdError):
            format_job_dir("DB00035", "P30518", "AVPR2", "F1", "cav2")

    def test_pocket_key_distinguishes_fragments(self):
        # Omitting Fragment_ID from the pocket key fanned one merge out by
        # 410,087 rows across 179 jobs.
        first = JobIdentity("DB00091", "P08F94", "PKHD1", "F4", 1)
        second = JobIdentity("DB00091", "P08F94", "PKHD1", "F13", 1)
        self.assertNotEqual(first.pocket_key, second.pocket_key)
        self.assertNotEqual(first.dir_name(), second.dir_name())
        self.assertEqual(first.pair_key, second.pair_key)


class TestRoundTrip(unittest.TestCase):
    """Property: parse(format(x)) == x, over adversarial generated inputs."""

    DRUGS = ["DB00035", "DB1", "DB99999999", "db00035", "drug-slug", "a_b"]
    ACCESSIONS = ["P30518", "Q8NCT1", "P08F94", "A0A0B4J2D5"]
    GENES = [None, "AVPR2", "MT-ND6", "A_B", "nan", "NaN", NULL_GENE, "C1QTNF9B-AS1"]
    FRAGMENTS = ["F1", "1", 1, 13, "F13", "F0"]
    CAVITIES = [0, 1, 9, 10, 12345, "7"]

    def test_round_trip_over_the_full_product(self):
        cases = 0
        for drug, accession, gene, fragment, cavity in itertools.product(
            self.DRUGS, self.ACCESSIONS, self.GENES, self.FRAGMENTS, self.CAVITIES
        ):
            name = format_job_dir(drug, accession, gene, fragment, cavity)
            parsed = parse_job_dir(name)
            expected = JobIdentity(drug, accession, gene, fragment, cavity)
            self.assertEqual(parsed, expected, name)
            self.assertEqual(parsed.dir_name(), name)
            self.assertEqual(name.count(DELIM), 4, name)
            self.assertNotIn("nan", name.lower())
            cases += 1
        self.assertEqual(cases, 6 * 4 * 8 * 6 * 6)

    def test_dict_round_trip(self):
        identity = JobIdentity("DB00035", "P30518", "AVPR2", "F1", 2)
        self.assertEqual(JobIdentity.from_dict(identity.to_dict()), identity)
        with self.assertRaises(JobIdError) as ctx:
            JobIdentity.from_dict({"drugbank_id": "DB1", "accession": "P30518"})
        self.assertIn("missing field", str(ctx.exception))

    def test_parse_accepts_a_path_and_rejects_a_legacy_name(self):
        name = "DB00035__P30518__AVPR2__F1__cav2"
        self.assertEqual(
            parse_job_dir(os.path.join("/tmp/run/docking_results", name)).accession,
            "P30518",
        )
        # No silent fallback to the legacy parser: guessing the scheme is how
        # identity gets fabricated.
        with self.assertRaises(JobIdParseError) as ctx:
            parse_job_dir("DB00035_AVPR2_Q8NCT1_cavity_2")
        self.assertIn("parse_legacy_job_dir", str(ctx.exception))
        for bad in ("", "a__b__c", "DB1__P30518__AVPR2__X1__cav2",
                    "DB1__P30518__AVPR2__F1__cavity2",
                    "DB1__P30518__AVPR2__F1__cav2__extra"):
            with self.subTest(name=bad):
                with self.assertRaises(JobIdParseError):
                    parse_job_dir(bad)


class TestLegacyParse(unittest.TestCase):
    def test_every_real_shape_parses(self):
        for name, expected in sorted(REAL_LEGACY_DIRS.items()):
            with self.subTest(name=name):
                identity = parse_legacy_job_dir(name, KNOWN_ACCESSIONS)
                self.assertEqual(
                    (
                        identity.drugbank_id,
                        identity.accession,
                        identity.gene,
                        identity.fragment_id,
                        identity.cavity_index,
                    ),
                    expected,
                )

    def test_the_old_regex_would_have_failed_on_these(self):
        import re

        old = re.compile(r"(DB\d+)_([A-Za-z0-9]+)_([A-Z0-9]+)_cavity_(\d+)")
        rejected = [n for n in REAL_LEGACY_DIRS if not old.search(n)]
        self.assertTrue(rejected)
        for name in rejected:
            with self.subTest(name=name):
                # ... and the right-anchored parser resolves every one of them.
                self.assertIn(
                    parse_legacy_job_dir(name, KNOWN_ACCESSIONS).accession,
                    KNOWN_ACCESSIONS,
                )

    def test_the_accession_wins_when_the_gene_disagrees(self):
        # DB00091_CAMLG_P08F94_...: P08F94 is PKHD1, not CAMLG.  The accession is
        # physical truth (it selected the receptor file); the gene is decoration.
        identity = parse_legacy_job_dir(
            "DB00091_CAMLG_P08F94_-F13-model_v1_cavity_10", KNOWN_ACCESSIONS
        )
        self.assertEqual(identity.accession, "P08F94")
        self.assertEqual(identity.gene, "CAMLG")

        from dock_selectivity.identity import IdentityIndex

        import pandas as pd

        index = IdentityIndex.from_frame(
            pd.DataFrame(
                {
                    "accession": ["P08F94", "Q8NCT1", "P30518"],
                    "entry_name": ["PKHD1_HUMAN", "ARRD4_HUMAN", "V2R_HUMAN"],
                    "primary_symbol": ["PKHD1", "ARRDC4", "AVPR2"],
                    "symbols": [
                        ["PKHD1", "FCYT", "TIGM1"],
                        ["ARRDC4"],
                        ["AVPR2", "V2R"],
                    ],
                    "entrez_id": pd.array([5314, 91947, 554], dtype="Int64"),
                    "reviewed": [True, True, True],
                }
            )
        )
        protein = index.by_accession(identity.accession)
        self.assertFalse(protein.has_symbol(identity.gene))
        self.assertEqual(index.label(identity.accession), "PKHD1")
        # Relabelling is what carries the job forward, and it cannot disagree.
        relabelled = identity.with_gene(index.label(identity.accession))
        self.assertEqual(relabelled.gene, "PKHD1")
        self.assertEqual(
            relabelled.with_fragment("F13").dir_name(),
            "DB00091__P08F94__PKHD1__F13__cav10",
        )

        # The same for the headline case: AVPR2 in the name, ARRDC4 in reality.
        other = parse_legacy_job_dir("DB00035_AVPR2_Q8NCT1_cavity_2", KNOWN_ACCESSIONS)
        self.assertEqual(other.accession, "Q8NCT1")
        self.assertEqual(index.label(other.accession), "ARRDC4")
        self.assertFalse(index.by_accession(other.accession).has_symbol(other.gene))

    def test_an_empty_accession_set_raises_instead_of_guessing(self):
        for empty in ((), [], set(), ("", "  ")):
            with self.subTest(value=repr(empty)):
                with self.assertRaises(JobIdError) as ctx:
                    parse_legacy_job_dir("DB00035_AVPR2_Q8NCT1_cavity_2", empty)
                self.assertIn("Refusing to fall back", str(ctx.exception))

    def test_a_bare_string_accession_set_raises(self):
        # Iterating a string would test single characters and match nothing --
        # or, worse, match something.
        with self.assertRaises(JobIdError) as ctx:
            parse_legacy_job_dir("DB00035_AVPR2_Q8NCT1_cavity_2", "Q8NCT1")
        self.assertIn("bare string", str(ctx.exception))

    def test_an_unknown_accession_is_reported_not_positionally_guessed(self):
        with self.assertRaises(JobIdParseError) as ctx:
            parse_legacy_job_dir("DB00035_AVPR2_Z99999_cavity_2", KNOWN_ACCESSIONS)
        message = str(ctx.exception)
        self.assertIn("accession set is the authority", message)
        self.assertIn("Z99999", message)

    def test_malformed_legacy_names_raise(self):
        for name in (
            "DB00035_AVPR2_Q8NCT1",  # no cavity suffix
            "DB00035_AVPR2_Q8NCT1_cavity_x",
            "Q8NCT1_cavity_2",  # accession is the first token: no drug
            "",
        ):
            with self.subTest(name=name):
                with self.assertRaises(JobIdParseError):
                    parse_legacy_job_dir(name, KNOWN_ACCESSIONS)

    def test_a_path_is_accepted(self):
        identity = parse_legacy_job_dir(
            "/media/x/consensus_docking_results/DB00157_MT-ND1_P03886_cavity_1/",
            KNOWN_ACCESSIONS,
        )
        self.assertEqual(identity.accession, "P03886")
        self.assertEqual(identity.gene, "MT-ND1")

    def test_the_rightmost_accession_wins(self):
        # A gene token that is itself an accession must not shadow the real
        # accession slot -- the parse is right-anchored.
        identity = parse_legacy_job_dir(
            "DB00035_P30518_Q8NCT1_cavity_2", KNOWN_ACCESSIONS
        )
        self.assertEqual(identity.accession, "Q8NCT1")
        self.assertEqual(identity.gene, "P30518")

    def test_legacy_identities_are_renamed_into_the_new_scheme(self):
        identity = parse_legacy_job_dir(
            "DB00157_MT-ND1_P03886_cavity_1", KNOWN_ACCESSIONS
        )
        # A legacy name without a fragment token cannot be renamed until the
        # fragment is supplied explicitly.
        with self.assertRaises(JobIdError):
            identity.dir_name()
        self.assertEqual(
            identity.with_fragment("F1").dir_name(),
            "DB00157__P03886__MT-ND1__F1__cav1",
        )
        self.assertEqual(
            parse_job_dir(identity.with_fragment(1).dir_name()).gene, "MT-ND1"
        )


class TestJobIdentityJson(unittest.TestCase):
    """job_identity.json is the authority; the folder name is a convenience."""

    def setUp(self):
        self.identity = JobIdentity("DB00035", "P30518", "AVPR2", "F1", 2)

    def test_write_then_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = os.path.join(tmp, self.identity.dir_name())
            path = write_job_identity(
                job_dir,
                self.identity,
                sample_type="positive",
                identity_index_sha256="a" * 64,
                code_revision="cad1ccb",
                provenance="reused_from",
                source_job="DB00035_AVPR2_Q8NCT1_cavity_2",
            )
            self.assertEqual(os.path.basename(path), JOB_IDENTITY_FILENAME)
            self.assertEqual(read_job_identity(job_dir), self.identity)
            self.assertEqual(read_job_identity(path), self.identity)

            record = read_job_identity_record(job_dir)
            self.assertEqual(record["schema_version"], JOB_IDENTITY_SCHEMA_VERSION)
            self.assertEqual(record["dir_name"], self.identity.dir_name())
            self.assertEqual(record["source_job"], "DB00035_AVPR2_Q8NCT1_cavity_2")
            self.assertEqual(record["provenance"], "reused_from")
            self.assertEqual(record["sample_type"], "positive")

            # An auditor reconstructs identity without parsing any name.
            self.assertEqual(record["accession"], "P30518")
            self.assertEqual(record["fragment_id"], "F1")
            self.assertEqual(record["cavity_index"], 2)

    def test_writes_are_atomic_and_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = os.path.join(tmp, "job")
            kwargs = {"sample_type": "positive", "code_revision": "cad1ccb"}
            path = write_job_identity(job_dir, self.identity, **kwargs)
            with open(path, "rb") as handle:
                first = handle.read()
            write_job_identity(job_dir, self.identity, **kwargs)
            with open(path, "rb") as handle:
                second = handle.read()
            self.assertEqual(first, second)
            self.assertEqual(
                sorted(os.listdir(job_dir)), [JOB_IDENTITY_FILENAME]
            )  # no leftover temp files
            self.assertTrue(first.endswith(b"\n"))

    def test_reading_a_directory_without_the_file_raises_no_name_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = os.path.join(tmp, "DB00035_AVPR2_Q8NCT1_cavity_2")
            os.makedirs(job_dir)
            with self.assertRaises(FileNotFoundError) as ctx:
                read_job_identity(job_dir)
            message = str(ctx.exception)
            self.assertIn("parse_legacy_job_dir", message)
            self.assertIn(JOB_IDENTITY_FILENAME, message)
            # ... and a job directory that does not exist at all names the
            # missing file too, rather than the directory.
            with self.assertRaises(FileNotFoundError) as ctx:
                read_job_identity(os.path.join(tmp, "no_such_job"))
            self.assertIn(JOB_IDENTITY_FILENAME, str(ctx.exception))

    def test_unknown_schema_version_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = os.path.join(tmp, "job")
            path = write_job_identity(job_dir, self.identity)
            with open(path, "r") as handle:
                record = json.load(handle)
            record["schema_version"] = 99
            with open(path, "w") as handle:
                json.dump(record, handle)
            with self.assertRaises(JobIdError) as ctx:
                read_job_identity(job_dir)
            self.assertIn("schema_version", str(ctx.exception))

    def test_corrupt_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = os.path.join(tmp, "job")
            os.makedirs(job_dir)
            with open(os.path.join(job_dir, JOB_IDENTITY_FILENAME), "w") as handle:
                handle.write("{not json")
            with self.assertRaises(JobIdError):
                read_job_identity(job_dir)

    def test_provenance_may_not_shadow_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(JobIdError) as ctx:
                write_job_identity(
                    os.path.join(tmp, "job"), self.identity, accession="Q8NCT1"
                )
            self.assertIn("may not shadow", str(ctx.exception))

    def test_non_serialisable_provenance_raises_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = os.path.join(tmp, "job")
            with self.assertRaises(JobIdError):
                write_job_identity(job_dir, self.identity, extra=object())
            self.assertFalse(
                os.path.exists(os.path.join(job_dir, JOB_IDENTITY_FILENAME))
            )

    def test_writing_without_a_fragment_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity = parse_legacy_job_dir(
                "DB00035_AVPR2_Q8NCT1_cavity_2", KNOWN_ACCESSIONS
            )
            with self.assertRaises(JobIdError) as ctx:
                write_job_identity(os.path.join(tmp, "job"), identity)
            self.assertIn("fragment_id is required", str(ctx.exception))

    def test_write_rejects_a_non_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(JobIdError):
                write_job_identity(tmp, {"accession": "P30518"})


class TestDeterminism(unittest.TestCase):
    def test_names_are_byte_identical_across_python_hash_seeds(self):
        script = r'''
import hashlib, json, os, sys
sys.path.insert(0, os.environ["REPO_ROOT"])
from dock_selectivity.jobid import parse_legacy_job_dir, format_job_dir

payload = json.load(open(os.environ["PAYLOAD"]))
# A set of accessions: iteration order here is PYTHONHASHSEED-dependent.
accessions = set(payload["accessions"])
lines = []
for name in payload["names"]:
    identity = parse_legacy_job_dir(name, accessions)
    lines.append("{0}|{1}".format(name, identity.with_fragment(
        identity.fragment_id or "F1").dir_name()))
print("SET_ORDER " + ",".join(accessions))
print("DIGEST " + hashlib.sha256("\n".join(lines).encode()).hexdigest())
print("\n".join(lines))
'''
        payload = {
            "accessions": list(KNOWN_ACCESSIONS),
            "names": sorted(REAL_LEGACY_DIRS),
        }
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "parse_names.py")
            payload_path = os.path.join(tmp, "payload.json")
            with open(script_path, "w") as handle:
                handle.write(script)
            with open(payload_path, "w") as handle:
                json.dump(payload, handle)

            set_orders = []
            outputs = []
            for seed in ("0", "1", "42"):
                env = dict(os.environ)
                env.update(
                    {
                        "PYTHONHASHSEED": seed,
                        "REPO_ROOT": REPO_ROOT,
                        "PAYLOAD": payload_path,
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
                set_orders.append(lines[0])
                outputs.append("\n".join(lines[1:]))

            self.assertGreater(
                len(set(set_orders)),
                1,
                "hash seeds produced identical set orders; the test proved nothing",
            )
            self.assertEqual(
                len(set(outputs)),
                1,
                "parsed identities differ across PYTHONHASHSEED values:\n{0}".format(
                    "\n---\n".join(sorted(set(outputs)))
                ),
            )


if __name__ == "__main__":
    unittest.main()


class TestAccessionSetNormalisationIsHoistable(unittest.TestCase):
    """Regression: parse_legacy_job_dir must not renormalise a big set per call.

    Found by running the real chain, not by a fixture. parse_legacy_job_dir
    normalised its known_accessions argument on every call, making a scan over N
    directories O(N * A). At production scale -- 24,173 real job directories
    against the 210,709-accession authoritative index -- that measured 35 minutes
    of pure CPU. Passing a pre-normalised frozenset is 10,173x faster (0.21s).

    Hermetic tests could never catch this: they use ~10 directories and a
    handful of accessions, where both paths are instant.
    """

    def test_normalize_accession_set_returns_a_frozenset(self):
        out = jobid.normalize_accession_set([" p30518 ", "q8nct1", "", "   "])
        self.assertIsInstance(out, frozenset)
        self.assertEqual(out, frozenset({"P30518", "Q8NCT1"}))

    def test_normalize_rejects_a_bare_string(self):
        with self.assertRaises(jobid.JobIdError):
            jobid.normalize_accession_set("P30518")

    def test_prenormalised_frozenset_is_reused_not_rebuilt(self):
        """The fast path must be taken for a frozenset, and give identical results."""
        raw = {"P30518", "Q8NCT1"}
        norm = jobid.normalize_accession_set(raw)
        name = "DB00035_AVPR2_P30518_cavity_2"
        self.assertEqual(
            jobid.parse_legacy_job_dir(name, raw),
            jobid.parse_legacy_job_dir(name, norm),
        )

    def test_frozenset_path_does_not_scale_with_set_size(self):
        """The whole point: parse cost must be independent of |accessions|."""
        import time

        small = jobid.normalize_accession_set({"P30518"})
        large = jobid.normalize_accession_set(
            {"P%05d" % i for i in range(60000)} | {"P30518"}
        )
        name = "DB00035_AVPR2_P30518_cavity_1"

        def timed(accs, n=300):
            start = time.time()
            for _ in range(n):
                jobid.parse_legacy_job_dir(name, accs)
            return time.time() - start

        timed(small, 50)  # warm up
        t_small, t_large = timed(small), timed(large)
        # A per-call rebuild would make the large set thousands of times slower.
        self.assertLess(
            t_large, max(t_small * 20, 0.5),
            "parse scales with accession-set size: %.4fs vs %.4fs -- the "
            "normalisation is being rebuilt per call again" % (t_large, t_small),
        )
