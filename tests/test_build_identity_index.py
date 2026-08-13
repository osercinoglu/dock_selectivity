"""Tests for ``build_identity_index.py``.

Every test here reproduces, in miniature, a failure the 2025-06-29 campaign
actually shipped, using the real identifiers from the incident so that a failing
assertion reads like the incident report:

* ``AVPR2`` is ``P30518``.  ``dict(zip(Mapped_Gene_Name, Entry))`` returned
  ``Q8NCT1`` (which is ARRDC4) because it happened to be last, **and Q8NCT1 is
  the protein that got docked**.
* ``TUBA1B`` is ``P68363``.  ``Q13509`` (TUBB3), ``Q96RT7`` (TUBGCP6) and
  ``Q96CW5`` (TUBGCP3) were all relabelled ``TUBA1B`` by the free-text search.
* ``IL2`` is ``P60568``; ``P01589`` is IL2RA.
* ``A0A024R1R8`` is the unreviewed TrEMBL entry that also carries GeneID 10376 —
  exactly the shape of collision that the old code resolved by hash order.

Hermetic: no network, no external drive, no ``/opt/data``.  Fixtures are written
into ``tempfile.TemporaryDirectory()``.  The ``--fetch`` path is exercised with
``_urlopen`` replaced by a local stub, and the offline path is exercised with
``_urlopen`` replaced by a stub that *fails the test if it is called at all*.
The only subprocesses are ``sys.executable -c '<snippet>'`` for the
``PYTHONHASHSEED`` check.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import build_identity_index as bii  # noqa: E402
from dock_selectivity.identity import IdentityIndex  # noqa: E402
from dock_selectivity.invariants import InvariantViolation  # noqa: E402


# --------------------------------------------------------------------------
# fixtures — real identifiers, synthetic files
# --------------------------------------------------------------------------

#: ``HUMAN_9606_idmapping.dat`` rows: (accession, id_type, value).
IDMAPPING_ROWS = [
    ("P30518", "UniProtKB-ID", "V2R_HUMAN"),
    ("P30518", "GeneID", "554"),
    ("P30518", "Gene_Name", "AVPR2"),
    ("P30518", "Gene_Synonym", "ADHR"),
    ("P30518", "Gene_Synonym", "DIR"),
    ("P30518", "Gene_Synonym", "V2R"),
    ("P30518", "RefSeq", "NP_000045.1"),          # an id type we never read
    ("P30518", "Ensembl", "ENSG00000126895"),     # ditto
    # the isoform must be excluded (G7), not mangled into the canonical row
    ("P30518-2", "UniProtKB-ID", "V2R_HUMAN"),
    ("P30518-2", "GeneID", "554"),
    ("P30518-2", "Gene_Name", "AVPR2"),
    ("Q8NCT1", "UniProtKB-ID", "ARRD4_HUMAN"),
    ("Q8NCT1", "GeneID", "91947"),
    ("Q8NCT1", "Gene_Name", "ARRDC4"),
    ("P68363", "UniProtKB-ID", "TBA1B_HUMAN"),
    ("P68363", "GeneID", "10376"),
    ("P68363", "Gene_Name", "TUBA1B"),
    # the TrEMBL twin: same GeneID, unreviewed. Must lose, must be kept.
    ("A0A024R1R8", "UniProtKB-ID", "A0A024R1R8_HUMAN"),
    ("A0A024R1R8", "GeneID", "10376"),
    ("A0A024R1R8", "Gene_Name", "TUBA1B"),
    ("Q13509", "UniProtKB-ID", "TBB3_HUMAN"),
    ("Q13509", "GeneID", "10381"),
    ("Q13509", "Gene_Name", "TUBB3"),
    ("Q13509", "Gene_Synonym", "TUBB4"),
    ("P60568", "UniProtKB-ID", "IL2_HUMAN"),
    ("P60568", "GeneID", "3558"),
    ("P60568", "Gene_Name", "IL2"),
    ("P01589", "UniProtKB-ID", "IL2RA_HUMAN"),
    ("P01589", "GeneID", "3559"),
    ("P01589", "Gene_Name", "IL2RA"),
    # a hyphenated gene symbol — must survive intact, and must NOT be confused
    # with the hyphen that marks an isoform accession
    ("P03886", "UniProtKB-ID", "NU1M_HUMAN"),
    ("P03886", "GeneID", "4535"),
    ("P03886", "Gene_Name", "MT-ND1"),
    # non-human entry name -> excluded (G7)
    ("P12345", "UniProtKB-ID", "AATM_PIG"),
    ("P12345", "GeneID", "999999"),
    ("P12345", "Gene_Name", "GOT2"),
]

#: Reference-proteome FASTA headers: (db, accession, entry_name, GN symbol).
PROTEOME_HEADERS = [
    ("sp", "P30518", "V2R_HUMAN", "AVPR2"),
    ("sp", "Q8NCT1", "ARRD4_HUMAN", "ARRDC4"),
    ("sp", "P68363", "TBA1B_HUMAN", "TUBA1B"),
    ("tr", "A0A024R1R8", "A0A024R1R8_HUMAN", "TUBA1B"),
    ("sp", "Q13509", "TBB3_HUMAN", "TUBB3"),
    ("sp", "P60568", "IL2_HUMAN", "IL2"),
    ("sp", "P01589", "IL2RA_HUMAN", "IL2RA"),
    ("sp", "P03886", "NU1M_HUMAN", "MT-ND1"),
]

LEGACY_CSV = (
    "Entry,Entry Name,Gene Names,Organism,Mapped_Gene_Name\n"
    # the four rows below are literally from uniprot_gene_mapping.csv; the
    # Mapped_Gene_Name column is the corrupt one and must never be read
    "P68363,TBA1B_HUMAN,TUBA1B,Homo sapiens (Human),TUBA1B\n"
    "Q96RT7,GCP6_HUMAN,TUBGCP6 GCP6 KIAA1669,Homo sapiens (Human),TUBA1B\n"
    "Q13509,TBB3_HUMAN,TUBB3 TUBB4,Homo sapiens (Human),TUBA1B\n"
    "P01589,IL2RA_HUMAN,IL2RA,Homo sapiens (Human),IL2\n"
    "P30518,V2R_HUMAN,AVPR2 ADHR DIR DIR3 V2R,Homo sapiens (Human),AVPR2\n"
    "Q8NCT1,ARRD4_HUMAN,ARRDC4,Homo sapiens (Human),AVPR2\n"
    "P60568,IL2_HUMAN,IL2,Homo sapiens (Human),IL2\n"
    "P03886,NU1M_HUMAN,MT-ND1,Homo sapiens (Human),MT-ND1\n"
    "Q9BQE5,APOL2_HUMAN,APOL2,Homo sapiens (Human),SLC18A1\n"
    "Q9XYZ9,FAKE_MOUSE,Fake1,Mus musculus (Mouse),FAKE1\n"
)

HGNC_HEADER = "hgnc_id\tsymbol\tstatus\tentrez_id\tuniprot_ids\tprev_symbol\talias_symbol\n"
HGNC_ROWS = [
    "HGNC:897\tAVPR2\tApproved\t554\tP30518\tDIR\tV2R|ADHR\n",
    "HGNC:20214\tARRDC4\tApproved\t91947\tQ8NCT1\t\t\n",
    "HGNC:12412\tTUBA1B\tApproved\t10376\tP68363\tTUBA1\tK-ALPHA-1\n",
    "HGNC:6001\tIL2\tApproved\t3558\tP60568\t\tTCGF\n",
    "HGNC:6008\tIL2RA\tApproved\t3559\tP01589\t\tCD25\n",
    "HGNC:7455\tMT-ND1\tApproved\t4535\tP03886\tMTND1\tND1\n",
    "HGNC:99999\tWITHDRAWN1\tEntry Withdrawn\t111111\tP68363\t\t\n",
]


def write_idmapping(path, rows=None, shuffle_seed=None, gzipped=False):
    """Write an idmapping .dat (optionally .gz, optionally row-shuffled)."""
    rows = list(IDMAPPING_ROWS if rows is None else rows)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)
    text = "".join("\t".join(r) + "\n" for r in rows)
    if gzipped:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    return path


def write_proteome(path, headers=None, gzipped=False):
    headers = PROTEOME_HEADERS if headers is None else headers
    text = "".join(
        ">{0}|{1}|{2} Some protein OS=Homo sapiens OX=9606 GN={3} PE=1 SV=1\n"
        "MRECISIHVGQAG\n".format(db, acc, name, symbol)
        for db, acc, name, symbol in headers
    )
    if gzipped:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    return path


def write_hgnc(path, rows=None):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(HGNC_HEADER)
        handle.writelines(HGNC_ROWS if rows is None else rows)
    return path


def write_legacy(path, text=None):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(LEGACY_CSV if text is None else text)
    return path


def sha256_of(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_text(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def read_notes(path_or_out):
    """Read identity_index.notes.csv, given the notes path or the index path."""
    if str(path_or_out).endswith(".parquet"):
        path_or_out = bii.default_notes_path(path_or_out)
    return pd.read_csv(path_or_out, dtype=str, keep_default_na=False)


def read_unresolved(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


class TempCaseMixin(unittest.TestCase):
    """A temp dir per test, plus a network stub that fails the test if used."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bii_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

        # run() prints a CLI summary; capture it so the suite stays readable
        # (and so a test can assert on the REDUCED_LOCAL banner).
        self.stdout = io.StringIO()
        self._real_stdout = sys.stdout
        sys.stdout = self.stdout
        self.addCleanup(setattr, sys, "stdout", self._real_stdout)

        self._real_urlopen = bii._urlopen
        self.addCleanup(setattr, bii, "_urlopen", self._real_urlopen)

        def _forbidden(url, timeout=300):  # pragma: no cover - only on failure
            raise AssertionError(
                "the offline path opened a socket for {0}; "
                "build_identity_index must never touch the network without "
                "--fetch".format(url)
            )

        bii._urlopen = _forbidden

    def path(self, *parts):
        return os.path.join(self.tmp, *parts)

    def standard_inputs(self, **kwargs):
        return (
            write_idmapping(self.path("idmapping.dat"), **kwargs),
            write_proteome(self.path("proteome.fasta")),
        )

    def _assert_leak_is_caught_by(self, leak, directory, expected_check):
        """Inject a silent loss and require *expected_check* to be what fires.

        Naming the check matters: if some other invariant happens to catch the
        leak, the accounting check could quietly have become a tautology
        (n_in computed as n_resolved + n_rejected balances by construction) and
        nobody would notice.
        """
        idmapping, proteome = self.standard_inputs()
        real = bii.assemble_records

        def leaky(*args, **kwargs):
            assembly = real(*args, **kwargs)
            leak(assembly)
            return assembly

        bii.assemble_records = leaky
        self.addCleanup(setattr, bii, "assemble_records", real)
        out = self.path(directory, "identity_index.parquet")
        code = bii.run(
            ["--idmapping", idmapping, "--proteome", proteome, "--out", out,
             "--log-level", "CRITICAL"]
        )
        self.assertEqual(code, 3)
        self.assertFalse(os.path.exists(out), "a leaking build must write no index")
        report = load_json(os.path.splitext(out)[0] + ".invariants.json")
        self.assertFalse(report[0]["ok"])
        self.assertEqual(report[0]["name"], expected_check)

    def build_standard(self, out_name="out/identity_index.parquet", extra=()):
        idmapping, proteome = self.standard_inputs()
        out = self.path(out_name)
        code = bii.run(
            ["--idmapping", idmapping, "--proteome", proteome, "--out", out,
             "--log-level", "ERROR"] + list(extra)
        )
        self.assertEqual(code, 0)
        return out


# ==========================================================================
# 1. the offline contract: a missing input is an error, never a fallback
# ==========================================================================


class TestOfflineContract(TempCaseMixin):
    def test_missing_proteome_without_fetch_raises_actionably(self):
        idmapping = write_idmapping(self.path("idmapping.dat"))
        with self.assertRaises(bii.BuildInputError) as ctx:
            bii.resolve_sources(
                {"idmapping": idmapping, "proteome": None},
                required_roles=("idmapping", "proteome"),
                fetch=False,
            )
        message = str(ctx.exception)
        # actionable: names the flag, the expected filename, the URL, and the
        # documented alternative — and says out loud that it will not degrade
        self.assertIn("--proteome", message)
        self.assertIn("UP000005640_9606.fasta.gz", message)
        self.assertIn("https://", message)
        self.assertIn("--fetch", message)
        self.assertIn("will NOT fall back", message)
        self.assertIn("--from-legacy-gene-names", message)

    def test_missing_input_writes_nothing_and_exits_nonzero(self):
        idmapping = write_idmapping(self.path("idmapping.dat"))
        out = self.path("out/identity_index.parquet")
        code = bii.run(["--idmapping", idmapping, "--out", out, "--log-level", "CRITICAL"])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out))
        self.assertFalse(os.path.exists(bii.default_meta_path(out)))

    def test_missing_idmapping_without_fetch_raises(self):
        proteome = write_proteome(self.path("proteome.fasta"))
        out = self.path("identity_index.parquet")
        code = bii.run(["--proteome", proteome, "--out", out, "--log-level", "CRITICAL"])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out))

    def test_nonexistent_path_is_an_error_not_a_skip(self):
        with self.assertRaises(bii.BuildInputError) as ctx:
            bii.resolve_sources(
                {"idmapping": self.path("nope.dat")},
                required_roles=("idmapping",),
                fetch=False,
            )
        self.assertIn("does not exist", str(ctx.exception))

    def test_offline_build_never_opens_a_socket(self):
        # bii._urlopen is the failing stub installed in setUp; a successful
        # build therefore proves no network access happened.
        out = self.build_standard()
        self.assertTrue(os.path.exists(out))

    def test_reduced_mode_never_fetches_even_with_fetch_flag(self):
        legacy = write_legacy(self.path("legacy.csv"))
        out = self.path("identity_index.parquet")
        code = bii.run(
            ["--from-legacy-gene-names", legacy, "--out", out, "--fetch",
             "--cache-dir", self.path("cache"), "--log-level", "ERROR"]
        )
        self.assertEqual(code, 0)

    def test_reduced_and_idmapping_together_is_refused(self):
        legacy = write_legacy(self.path("legacy.csv"))
        idmapping = write_idmapping(self.path("idmapping.dat"))
        out = self.path("identity_index.parquet")
        code = bii.run(
            ["--from-legacy-gene-names", legacy, "--idmapping", idmapping,
             "--out", out, "--log-level", "CRITICAL"]
        )
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out))


class TestFetchPath(TempCaseMixin):
    """``--fetch`` is the only network access, is cached, and is recorded."""

    def _install_stub(self, fail_hgnc=False):
        calls = []
        idmapping_text = "".join("\t".join(r) + "\n" for r in IDMAPPING_ROWS)
        proteome_text = read_text(write_proteome(self.path("src.fasta")))
        hgnc_text = read_text(write_hgnc(self.path("src_hgnc.txt")))

        def gzipped(text):
            buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
                handle.write(text.encode("utf-8"))
            return io.BytesIO(buffer.getvalue())

        def stub(url, timeout=300):
            calls.append(url)
            if url.endswith("idmapping.dat.gz"):
                return gzipped(idmapping_text)
            if url.endswith(".fasta.gz"):
                return gzipped(proteome_text)
            if url.endswith("hgnc_complete_set.txt"):
                if fail_hgnc:
                    raise OSError("HTTP Error 404: Not Found")
                return io.BytesIO(hgnc_text.encode("utf-8"))
            if url.endswith("reldate.txt"):
                return io.BytesIO(b"UniProt Knowledgebase Release 2026_02 ...\n")
            raise AssertionError("unexpected URL " + url)  # pragma: no cover

        bii._urlopen = stub
        return calls

    def test_fetch_downloads_caches_and_records_url_and_sha(self):
        calls = self._install_stub()
        cache = self.path("cache")
        out = self.path("identity_index.parquet")
        self.assertEqual(
            0,
            bii.run(["--fetch", "--cache-dir", cache, "--out", out, "--log-level", "ERROR"]),
        )
        self.assertTrue(os.path.exists(os.path.join(cache, "HUMAN_9606_idmapping.dat.gz")))
        n_first = len(calls)
        self.assertGreaterEqual(n_first, 2)

        meta = load_json(bii.default_meta_path(out))
        roles = {s["role"]: s for s in meta["sources"]}
        self.assertEqual(roles["idmapping"]["url"], bii.SOURCE_SPECS["idmapping"].url)
        self.assertEqual(
            roles["idmapping"]["sha256"],
            sha256_of(os.path.join(cache, "HUMAN_9606_idmapping.dat.gz")),
        )
        self.assertTrue(roles["idmapping"]["fetched_now"])
        self.assertIn("2026_02", meta["uniprot_release"] or "")

        # second run: cached, so no new download without --refresh
        del calls[:]
        out2 = self.path("again/identity_index.parquet")
        self.assertEqual(
            0,
            bii.run(["--fetch", "--cache-dir", cache, "--out", out2, "--log-level", "ERROR"]),
        )
        self.assertEqual(calls, [], "a cached input was re-downloaded without --refresh")
        # and the cached build is byte-identical to the fetched one
        self.assertEqual(sha256_of(out), sha256_of(out2))

        # --refresh downloads again
        del calls[:]
        out3 = self.path("refreshed/identity_index.parquet")
        self.assertEqual(
            0,
            bii.run(
                ["--fetch", "--refresh", "--cache-dir", cache, "--out", out3,
                 "--log-level", "ERROR"]
            ),
        )
        self.assertTrue(calls, "--refresh did not re-download")

    def test_explicit_local_path_is_never_fetched_over(self):
        calls = self._install_stub()
        idmapping, proteome = self.standard_inputs()
        out = self.path("identity_index.parquet")
        self.assertEqual(
            0,
            bii.run(
                ["--fetch", "--cache-dir", self.path("cache"), "--idmapping", idmapping,
                 "--proteome", proteome, "--out", out, "--log-level", "ERROR"]
            ),
        )
        for url in calls:
            self.assertNotIn("idmapping", url, "a supplied --idmapping was fetched over")
            self.assertNotIn("UP000005640", url, "a supplied --proteome was fetched over")
        meta = load_json(bii.default_meta_path(out))
        roles = {s["role"]: s for s in meta["sources"]}
        self.assertFalse(roles["idmapping"]["fetched_now"])
        self.assertEqual(roles["idmapping"]["path"], os.path.abspath(idmapping))

    def test_optional_hgnc_download_failure_is_noted_not_fatal(self):
        # HGNC is declared optional: it adds synonyms and a tie-break hint and
        # cannot add or remove an accession, so its absence is recorded rather
        # than fatal. A REQUIRED input behaves the opposite way (below).
        self._install_stub(fail_hgnc=True)
        out = self.path("identity_index.parquet")
        self.assertEqual(
            0,
            bii.run(["--fetch", "--cache-dir", self.path("cache"), "--out", out,
                     "--log-level", "CRITICAL"]),
        )
        meta = load_json(bii.default_meta_path(out))
        self.assertNotIn("hgnc", {s["role"] for s in meta["sources"]})
        self.assertTrue(any("hgnc" in note for note in meta["notes"]))

    def test_required_download_failure_is_fatal(self):
        def only_hgnc(url, timeout=300):
            if url.endswith("hgnc_complete_set.txt"):
                return io.BytesIO(b"hgnc_id\tsymbol\tstatus\tentrez_id\tuniprot_ids\n")
            raise OSError("HTTP Error 500")

        bii._urlopen = only_hgnc
        out = self.path("identity_index.parquet")
        self.assertEqual(
            2,
            bii.run(["--fetch", "--cache-dir", self.path("cache"), "--out", out,
                     "--log-level", "CRITICAL"]),
        )
        self.assertFalse(os.path.exists(out))

    def test_download_failure_is_actionable_and_writes_nothing(self):
        def broken(url, timeout=300):
            raise OSError("connection reset by peer")

        bii._urlopen = broken
        out = self.path("identity_index.parquet")
        code = bii.run(
            ["--fetch", "--cache-dir", self.path("cache"), "--out", out,
             "--log-level", "CRITICAL"]
        )
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out))


# ==========================================================================
# 2. the authoritative build and the FIX_PLAN acceptance criteria
# ==========================================================================


class TestAuthoritativeBuild(TempCaseMixin):
    def setUp(self):
        super().setUp()
        self.out = self.build_standard()
        self.index = IdentityIndex.load(self.out)
        self.meta = load_json(bii.default_meta_path(self.out))
        self.rejects = read_unresolved(bii.default_unresolved_path(self.out))

    def test_fix_plan_acceptance_criteria(self):
        # FIX_PLAN.md section 1 "Acceptance criteria", verbatim.
        self.assertEqual(self.index.by_entrez(554).accession, "P30518")     # AVPR2
        self.assertEqual(self.index.by_entrez(10376).accession, "P68363")   # TUBA1B
        self.assertEqual(self.index.by_entrez(3558).accession, "P60568")    # IL2
        self.assertNotEqual(self.index.by_entrez(554).accession, "Q8NCT1")
        self.assertEqual(self.index.cardinality_report()["max_collapse"], 1)

    def test_label_direction_is_accession_to_symbol_only(self):
        self.assertEqual(self.index.label("P30518"), "AVPR2")
        self.assertEqual(self.index.label("Q8NCT1"), "ARRDC4")
        self.assertEqual(self.index.label("P68363"), "TUBA1B")

    def test_known_issue_regression_assertion_passes_on_the_emitted_index(self):
        # KNOWN_ISSUE_uniprot_gene_mapping.md "Regression check": every label
        # must be a symbol or synonym of the entry it labels. It failed on
        # 2,647/4,585 rows of the old file.
        frame = pd.read_parquet(self.out)
        bad = [
            (row.accession, row.primary_symbol, list(row.symbols))
            for row in frame.itertuples()
            if row.primary_symbol not in list(row.symbols)
        ]
        self.assertEqual(bad, [])

    def test_synonyms_are_kept_and_ordered_deterministically(self):
        protein = self.index.by_accession("P30518")
        self.assertEqual(protein.primary_symbol, protein.symbols[0])
        self.assertEqual(protein.symbols, ("AVPR2", "ADHR", "DIR", "V2R"))
        self.assertTrue(protein.has_symbol("V2R"))
        self.assertFalse(protein.has_symbol("ARRDC4"))

    def test_hyphenated_gene_symbol_survives_intact(self):
        # MT-ND1 has a hyphen; an isoform accession also has a hyphen. Only the
        # accession's hyphen means "isoform".
        protein = self.index.by_accession("P03886")
        self.assertEqual(protein.primary_symbol, "MT-ND1")
        self.assertEqual(self.index.by_entrez(4535).accession, "P03886")

    def test_isoform_accession_is_excluded_and_recorded(self):
        self.assertNotIn("P30518-2", self.index)
        row = self.rejects[self.rejects["key"] == "P30518-2"]
        self.assertEqual(len(row), 1)
        self.assertEqual(row.iloc[0]["disposition"], "excluded")
        self.assertIn("isoform", row.iloc[0]["reason"])
        self.assertIn("P30518", row.iloc[0]["reason"])

    def test_non_human_entry_is_excluded_and_recorded(self):
        self.assertNotIn("P12345", self.index)
        row = self.rejects[self.rejects["key"] == "P12345"]
        self.assertEqual(len(row), 1)
        self.assertEqual(row.iloc[0]["disposition"], "excluded")
        self.assertIn("not human", row.iloc[0]["reason"])

    def test_gene_symbol_in_the_accession_column_is_rejected(self):
        rows = list(IDMAPPING_ROWS) + [
            ("AVPR2", "UniProtKB-ID", "V2R_HUMAN"),
            ("AVPR2", "GeneID", "554"),
            ("AVPR2", "Gene_Name", "AVPR2"),
        ]
        idmapping = write_idmapping(self.path("bad.dat"), rows=rows)
        proteome = write_proteome(self.path("proteome2.fasta"))
        out = self.path("bad/identity_index.parquet")
        self.assertEqual(
            0,
            bii.run(["--idmapping", idmapping, "--proteome", proteome, "--out", out,
                     "--log-level", "ERROR"]),
        )
        index = IdentityIndex.load(out)
        self.assertNotIn("AVPR2", index)
        self.assertEqual(index.by_entrez(554).accession, "P30518")
        rejects = read_unresolved(bii.default_unresolved_path(out))
        row = rejects[rejects["key"] == "AVPR2"]
        self.assertEqual(len(row), 1)
        self.assertIn("not a valid UniProtKB accession", row.iloc[0]["reason"])

    def test_proteome_cross_check_is_reported_by_default_and_fatal_on_demand(self):
        # UniProt's GN= and its Gene_Name record legitimately disagree for a
        # handful of entries, so this independent cross-check is REPORTED by
        # default and only fatal under --strict-cross-check.
        rows = [
            ("P68363", "UniProtKB-ID", "TBA1B_HUMAN"),
            ("P68363", "GeneID", "10376"),
            ("P68363", "Gene_Name", "TUBA1B"),
        ]
        headers = [("sp", "P68363", "TBA1B_HUMAN", "TUBA1BX")]  # GN disagrees
        idmapping = write_idmapping(self.path("cc.dat"), rows=rows)
        proteome = write_proteome(self.path("cc.fasta"), headers=headers)

        lenient = self.path("lenient/identity_index.parquet")
        self.assertEqual(
            0,
            bii.run(["--idmapping", idmapping, "--proteome", proteome, "--out", lenient,
                     "--log-level", "CRITICAL"]),
        )
        meta = load_json(bii.default_meta_path(lenient))
        cross = {r.get("name"): r for r in meta["invariants"]}[
            "cross-check: primary_symbol vs reference-proteome GN="
        ]
        self.assertFalse(cross["ok"])
        self.assertEqual(cross["n_violations"], 1)
        self.assertEqual(IdentityIndex.load(lenient).label("P68363"), "TUBA1B")

        strict = self.path("strict_cc/identity_index.parquet")
        self.assertEqual(
            3,
            bii.run(["--idmapping", idmapping, "--proteome", proteome, "--out", strict,
                     "--strict-cross-check", "--log-level", "CRITICAL"]),
        )
        self.assertFalse(os.path.exists(strict))

    def test_meta_records_sources_counts_and_invariants(self):
        self.assertEqual(self.meta["provenance"], bii.PROVENANCE_UNIPROT)
        self.assertIsNone(self.meta["warning"])
        roles = {s["role"]: s for s in self.meta["sources"]}
        self.assertEqual(set(roles), {"idmapping", "proteome"})
        for source in roles.values():
            self.assertEqual(len(source["sha256"]), 64)
            self.assertGreater(source["bytes"], 0)
        self.assertEqual(self.meta["outputs"]["index_sha256"], sha256_of(self.out))
        self.assertEqual(self.meta["cardinality_report"]["max_collapse"], 1)
        self.assertTrue(all(r.get("ok") for r in self.meta["invariants"]))
        names = [r.get("name") for r in self.meta["invariants"]]
        for expected in (
            "G1 one row per accession",
            "G2 accession accounting",
            "G2 entrez accounting",
            "G5 entrez -> accession",
            "G4 entrez(accession(e)) == e",
        ):
            self.assertIn(expected, names)
        self.assertIsNotNone(self.meta["builder"]["builder_version"])

    def test_parquet_carries_the_provenance_stamp(self):
        self.assertEqual(bii.read_index_provenance(self.out), bii.PROVENANCE_UNIPROT)

    def test_gzipped_inputs_produce_the_same_index(self):
        idmapping = write_idmapping(self.path("gz.dat.gz"), gzipped=True)
        proteome = write_proteome(self.path("gz.fasta.gz"), gzipped=True)
        out = self.path("gz/identity_index.parquet")
        self.assertEqual(
            0,
            bii.run(["--idmapping", idmapping, "--proteome", proteome, "--out", out,
                     "--log-level", "ERROR"]),
        )
        self.assertEqual(sha256_of(out), sha256_of(self.out))

    def test_unknown_entrez_resolves_with_a_reason_and_never_raises(self):
        resolution = self.index.resolve_entrez(123456789)
        self.assertEqual(resolution.status, "UNKNOWN")
        self.assertTrue(resolution.reason)
        self.assertEqual(resolution.as_reject_row()["status"], "UNKNOWN")


# ==========================================================================
# 3. a last-wins collapse must be impossible
# ==========================================================================


class TestCollapseIsImpossible(TempCaseMixin):
    def test_collision_is_resolved_by_the_documented_tie_break(self):
        out = self.build_standard()
        index = IdentityIndex.load(out)
        # reviewed Swiss-Prot P68363 beats unreviewed TrEMBL A0A024R1R8
        self.assertEqual(index.by_entrez(10376).accession, "P68363")
        self.assertEqual(index.cardinality_report()["max_collapse"], 1)
        self.assertEqual(index.cardinality_report()["n_entrez_collapsing"], 0)

    def test_the_loser_is_retained_not_dropped_and_is_recorded(self):
        out = self.build_standard()
        index = IdentityIndex.load(out)
        # still in the index, still correctly labelled, simply not canonical
        loser = index.by_accession("A0A024R1R8")
        self.assertIsNone(loser.entrez_id)
        self.assertEqual(loser.primary_symbol, "TUBA1B")
        # a demotion is a recorded decision about a row that IS in the index,
        # so it belongs in the notes ledger, not in "unresolved targets"
        self.assertNotIn(
            "A0A024R1R8", list(read_unresolved(bii.default_unresolved_path(out))["key"])
        )
        notes = read_notes(out)
        row = notes[(notes["key"] == "A0A024R1R8") & (notes["status"] == "AMBIGUOUS")]
        self.assertEqual(len(row), 1)
        self.assertEqual(row.iloc[0]["disposition"], "retained")
        self.assertIn("P68363", row.iloc[0]["reason"])
        self.assertIn("10376", row.iloc[0]["reason"])
        self.assertTrue(row.iloc[0]["reason"].strip())

    def test_winner_does_not_depend_on_input_row_order(self):
        # The 2025 bug's signature: which accession survived depended on the
        # order rows arrived in. Reverse the file and the answer must not move.
        forward = write_idmapping(self.path("fwd.dat"))
        backward = write_idmapping(self.path("bwd.dat"), rows=list(reversed(IDMAPPING_ROWS)))
        self.assertNotEqual(sha256_of(forward), sha256_of(backward))
        proteome = write_proteome(self.path("proteome.fasta"))
        digests = []
        for name, source in (("f", forward), ("b", backward)):
            out = self.path(name, "identity_index.parquet")
            self.assertEqual(
                0,
                bii.run(["--idmapping", source, "--proteome", proteome, "--out", out,
                         "--log-level", "ERROR"]),
            )
            self.assertEqual(IdentityIndex.load(out).by_entrez(10376).accession, "P68363")
            digests.append(sha256_of(out))
        self.assertEqual(digests[0], digests[1])

    def _collision_index(self, rows, headers, name):
        idmapping = write_idmapping(self.path(name + ".dat"), rows=rows)
        proteome = write_proteome(self.path(name + ".fasta"), headers=headers)
        out = self.path(name, "identity_index.parquet")
        self.assertEqual(
            0,
            bii.run(["--idmapping", idmapping, "--proteome", proteome, "--out", out,
                     "--log-level", "ERROR"]),
        )
        return IdentityIndex.load(out)

    def test_tie_break_level_2_is_reference_proteome_membership(self):
        # both unreviewed: the one in the reference proteome wins
        rows = [
            ("A0A024R1R8", "UniProtKB-ID", "A0A024R1R8_HUMAN"),
            ("A0A024R1R8", "GeneID", "10376"),
            ("A0A024R1R8", "Gene_Name", "TUBA1B"),
            ("A0A096LP49", "UniProtKB-ID", "A0A096LP49_HUMAN"),
            ("A0A096LP49", "GeneID", "10376"),
            ("A0A096LP49", "Gene_Name", "TUBA1B"),
        ]
        headers = [("tr", "A0A024R1R8", "A0A024R1R8_HUMAN", "TUBA1B")]
        index = self._collision_index(rows, headers, "lvl2")
        self.assertEqual(index.by_entrez(10376).accession, "A0A024R1R8")
        self.assertIsNone(index.by_accession("A0A096LP49").entrez_id)

    def test_tie_break_level_3_is_lowest_accession_lexicographically(self):
        # neither reviewed nor in the proteome: a total, explicit last resort
        rows = [
            ("B2RXH2", "UniProtKB-ID", "B2RXH2_HUMAN"),
            ("B2RXH2", "GeneID", "10376"),
            ("B2RXH2", "Gene_Name", "TUBA1B"),
            ("A0A096LP49", "UniProtKB-ID", "A0A096LP49_HUMAN"),
            ("A0A096LP49", "GeneID", "10376"),
            ("A0A096LP49", "Gene_Name", "TUBA1B"),
        ]
        index = self._collision_index(rows, [], "lvl3")
        self.assertEqual(index.by_entrez(10376).accession, "A0A096LP49")

    def test_on_entrez_collision_fail_refuses_to_build(self):
        idmapping, proteome = self.standard_inputs()
        out = self.path("strict/identity_index.parquet")
        code = bii.run(
            ["--idmapping", idmapping, "--proteome", proteome, "--out", out,
             "--on-entrez-collision", "fail", "--log-level", "CRITICAL"]
        )
        self.assertEqual(code, 3)
        self.assertFalse(os.path.exists(out), "a rejected build must write no index")
        # the diagnostic is written, and it is unmistakably a report
        report_path = self.path("strict/identity_index.invariants.json")
        self.assertTrue(os.path.exists(report_path))
        report = load_json(report_path)
        self.assertFalse(report[0]["ok"])
        self.assertIn("10376", report[0]["message"])

    def test_the_g5_gate_fires_if_a_collision_ever_reaches_the_index(self):
        # The tie-break makes a collapse structurally impossible, so G5 can only
        # be shown to work by injecting one. Without this, deleting the G5 check
        # would pass the whole suite.
        def inject_collision(assembly):
            assembly.entrez_pairs.append((554, "Q8NCT1"))  # AVPR2 -> ARRDC4

        self._assert_leak_is_caught_by(
            inject_collision, "g5", "G5 entrez -> accession"
        )

    def test_the_g4_gate_fires_on_the_exact_incident_shape(self):
        # Q8NCT1 (ARRDC4) carrying the label 'AVPR2' is the row that decided
        # which receptor got docked. Note the injected record is internally
        # self-consistent (primary_symbol IS in symbols) — only a check against
        # an INDEPENDENT truth source can see it.
        from dock_selectivity.identity import Protein

        def mislabel(assembly):
            for i, protein in enumerate(assembly.records):
                if protein.accession == "Q8NCT1":
                    assembly.records[i] = Protein(
                        accession="Q8NCT1",
                        entry_name=protein.entry_name,
                        primary_symbol="AVPR2",
                        symbols=("AVPR2", "ARRDC4"),
                        entrez_id=protein.entrez_id,
                        reviewed=protein.reviewed,
                    )

        self._assert_leak_is_caught_by(
            mislabel, "g4", "G4 primary_symbol belongs to its own accession"
        )

    def test_assemble_records_raises_on_collision_when_asked(self):
        entries = {
            "P68363": bii.RawEntry("P68363", "TBA1B_HUMAN", ["TUBA1B"], [], [10376], True, True),
            "A0A024R1R8": bii.RawEntry(
                "A0A024R1R8", "A0A024R1R8_HUMAN", ["TUBA1B"], [], [10376], False, True
            ),
        }
        with self.assertRaises(InvariantViolation) as ctx:
            bii.assemble_records(entries, {}, on_entrez_collision="fail")
        self.assertIn("collapse factor must be exactly 1", str(ctx.exception))

    def test_a_failed_build_does_not_overwrite_an_existing_index(self):
        good = self.build_standard(out_name="live/identity_index.parquet")
        before = sha256_of(good)
        idmapping, proteome = self.standard_inputs()
        code = bii.run(
            ["--idmapping", idmapping, "--proteome", proteome, "--out", good,
             "--on-entrez-collision", "fail", "--log-level", "CRITICAL"]
        )
        self.assertEqual(code, 3)
        self.assertEqual(sha256_of(good), before)


# ==========================================================================
# 4. determinism (G3)
# ==========================================================================


class TestDeterminism(TempCaseMixin):
    def test_two_builds_to_two_paths_are_byte_identical(self):
        idmapping, proteome = self.standard_inputs()
        digests = []
        for name in ("first", "second"):
            out = self.path(name, "identity_index.parquet")
            self.assertEqual(
                0,
                bii.run(["--idmapping", idmapping, "--proteome", proteome, "--out", out,
                         "--log-level", "ERROR"]),
            )
            digests.append(sha256_of(out))
        self.assertEqual(digests[0], digests[1])

    def test_shuffled_input_rows_produce_identical_bytes(self):
        proteome = write_proteome(self.path("proteome.fasta"))
        sources = [
            write_idmapping(self.path("s0.dat"), shuffle_seed=0),
            write_idmapping(self.path("s1.dat"), shuffle_seed=12345),
        ]
        # the premise: the two inputs really are different files
        self.assertNotEqual(sha256_of(sources[0]), sha256_of(sources[1]))
        digests = []
        for i, source in enumerate(sources):
            out = self.path("shuf{0}".format(i), "identity_index.parquet")
            self.assertEqual(
                0,
                bii.run(["--idmapping", source, "--proteome", proteome, "--out", out,
                         "--log-level", "ERROR"]),
            )
            digests.append(sha256_of(out))
        self.assertEqual(digests[0], digests[1])

    def test_unresolved_csv_is_also_deterministic(self):
        proteome = write_proteome(self.path("proteome.fasta"))
        digests = []
        for i, seed in enumerate((3, 99)):
            source = write_idmapping(self.path("u{0}.dat".format(i)), shuffle_seed=seed)
            out = self.path("u{0}".format(i), "identity_index.parquet")
            bii.run(["--idmapping", source, "--proteome", proteome, "--out", out,
                     "--log-level", "ERROR"])
            digests.append(
                sha256_of(bii.default_unresolved_path(out))
            )
        self.assertEqual(digests[0], digests[1])

    def _run_under_seed(self, snippet, seed):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = str(seed)
        completed = subprocess.run(
            [sys.executable, "-c", snippet],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=300,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8", "replace")[-2000:],
        )
        return completed.stdout.decode("utf-8").strip()

    def test_output_is_identical_across_pythonhashseed(self):
        idmapping, proteome = self.standard_inputs()

        # First prove the harness is sensitive: string hashing really does move
        # between these seeds, so an identical build result is evidence.
        control = [
            self._run_under_seed("print(hash('AVPR2'), hash('TUBA1B'))", seed)
            for seed in (0, 1, 42)
        ]
        self.assertEqual(len(set(control)), 3, "PYTHONHASHSEED had no effect")

        snippet = (
            "import json, sys\n"
            "sys.path.insert(0, {repo!r})\n"
            "import build_identity_index as b\n"
            "print(b.parquet_fingerprint_for_argv(json.loads({argv!r})))\n"
        ).format(
            repo=REPO_ROOT,
            argv=json.dumps(
                ["--idmapping", idmapping, "--proteome", proteome,
                 "--out", self.path("probe.parquet"), "--log-level", "ERROR"]
            ),
        )
        digests = [self._run_under_seed(snippet, seed) for seed in (0, 1, 42)]
        self.assertEqual(len(set(digests)), 1, "output depends on PYTHONHASHSEED")
        self.assertEqual(len(digests[0]), 64)

    def test_check_determinism_flag_runs_and_passes(self):
        idmapping, proteome = self.standard_inputs()
        out = self.path("det/identity_index.parquet")
        self.assertEqual(
            0,
            bii.run(["--idmapping", idmapping, "--proteome", proteome, "--out", out,
                     "--check-determinism", "--log-level", "ERROR"]),
        )

    def test_meta_is_reproducible_when_source_date_epoch_is_set(self):
        idmapping, proteome = self.standard_inputs()
        previous = os.environ.get("SOURCE_DATE_EPOCH")
        os.environ["SOURCE_DATE_EPOCH"] = "1700000000"
        try:
            metas = []
            for name in ("m1", "m2"):
                out = self.path(name, "identity_index.parquet")
                bii.run(["--idmapping", idmapping, "--proteome", proteome, "--out", out,
                         "--log-level", "ERROR"])
                meta = load_json(bii.default_meta_path(out))
                meta.pop("outputs")  # paths differ by construction
                metas.append(json.dumps(meta, sort_keys=True))
        finally:
            if previous is None:
                os.environ.pop("SOURCE_DATE_EPOCH", None)
            else:
                os.environ["SOURCE_DATE_EPOCH"] = previous
        self.assertEqual(metas[0], metas[1])
        self.assertIn("2023-11-14", json.loads(metas[0])["build_timestamp_utc"])

    def test_unique_sorted_is_order_and_seed_independent(self):
        self.assertEqual(
            bii.unique_sorted(["TUBB3", "TUBA1B", "TUBB3", "AVPR2"]),
            ["AVPR2", "TUBA1B", "TUBB3"],
        )
        self.assertEqual(
            bii.unique_sorted(["AVPR2", "TUBB3", "TUBA1B"]),
            bii.unique_sorted(["TUBB3", "TUBA1B", "AVPR2"]),
        )


# ==========================================================================
# 5. total accounting: nothing is dropped without a reason
# ==========================================================================


class TestTotalAccounting(TempCaseMixin):
    def test_every_source_accession_is_indexed_or_rejected_with_a_reason(self):
        out = self.build_standard()
        index = IdentityIndex.load(out)
        rejects = read_unresolved(bii.default_unresolved_path(out))
        excluded = rejects[
            (rejects["key_type"] == "accession") & (rejects["disposition"] == "excluded")
        ]
        source_accessions = bii.unique_sorted([r[0] for r in IDMAPPING_ROWS])
        self.assertEqual(len(index) + len(excluded), len(source_accessions))
        for reason in rejects["reason"]:
            self.assertTrue(str(reason).strip(), "a reject row carries no reason")

    def test_meta_accounting_is_cross_checkable_against_index_and_ledger(self):
        # This is exactly what audit.py's G2 gate does: it reads n_in /
        # n_resolved / n_rejected out of the meta and checks them against two
        # OTHER files. A check that reads only one artifact proves nothing.
        out = self.build_standard()
        meta = load_json(bii.default_meta_path(out))
        accounting = meta["accounting"]
        index = IdentityIndex.load(out)
        n_csv_rows = len(read_unresolved(bii.default_unresolved_path(out)))
        self.assertEqual(
            accounting["n_in"], accounting["n_resolved"] + accounting["n_rejected"]
        )
        self.assertEqual(accounting["n_resolved"], len(index))
        self.assertEqual(accounting["n_rejected"], n_csv_rows)
        self.assertEqual(accounting["n_in"], len(bii.unique_sorted(
            [r[0] for r in IDMAPPING_ROWS]
        )))

    def test_the_two_ledgers_are_disjoint_and_mean_different_things(self):
        out = self.build_standard()
        unresolved = read_unresolved(bii.default_unresolved_path(out))
        notes = read_notes(out)
        index = IdentityIndex.load(out)
        # unresolved_targets.csv: NOT in the index
        for key in unresolved["key"]:
            self.assertNotIn(key, index)
        self.assertEqual(set(unresolved["disposition"]), {"excluded"})
        # notes: recorded decisions about rows that ARE in the index (or about
        # an Entrez id that lost its accession)
        self.assertTrue(len(notes) > 0)
        for _, row in notes.iterrows():
            if row["key_type"] == "accession":
                self.assertIn(row["key"], index)
            self.assertTrue(row["reason"].strip())
        self.assertEqual(
            set(unresolved["key"]) & set(notes[notes["key_type"] == "accession"]["key"]),
            set(),
        )

    def test_notes_ledger_is_deterministic(self):
        proteome = write_proteome(self.path("proteome.fasta"))
        digests = []
        for i, seed in enumerate((5, 77)):
            source = write_idmapping(self.path("n{0}.dat".format(i)), shuffle_seed=seed)
            out = self.path("n{0}".format(i), "identity_index.parquet")
            bii.run(["--idmapping", source, "--proteome", proteome, "--out", out,
                     "--log-level", "ERROR"])
            digests.append(sha256_of(bii.default_notes_path(out)))
        self.assertEqual(digests[0], digests[1])

    def test_accounting_reports_are_present_and_non_tautological(self):
        out = self.build_standard()
        meta = load_json(bii.default_meta_path(out))
        reports = {r.get("name"): r for r in meta["invariants"]}
        accession_report = reports["G2 accession accounting"]
        self.assertEqual(accession_report["n_unaccounted"], 0)
        # n_in must be counted from the SOURCE. If it were computed as
        # n_resolved + n_rejected the check would be a tautology that can never
        # fail, which is how validate_negatives() reported "All validations
        # passed" over a corrupt set.
        n_source_accessions = len(bii.unique_sorted([r[0] for r in IDMAPPING_ROWS]))
        self.assertEqual(accession_report["n_in"], n_source_accessions)
        self.assertGreater(accession_report["n_in"], accession_report["n_resolved"])

        entrez_report = reports["G2 entrez accounting"]
        self.assertEqual(entrez_report["n_unaccounted"], 0)
        n_source_gids = len(
            bii.unique_sorted(
                [int(r[2]) for r in IDMAPPING_ROWS
                 if r[1] == "GeneID" and "-" not in r[0] and r[0] != "P12345"]
            )
        )
        self.assertEqual(entrez_report["n_in"], n_source_gids)

    def test_the_g1_gate_fires_if_an_accession_is_ever_duplicated(self):
        def duplicate_a_row(assembly):
            assembly.records.append(assembly.records[0])

        self._assert_leak_is_caught_by(
            duplicate_a_row, "g1", "G1 one row per accession"
        )

    def test_a_silent_accession_drop_is_caught_by_the_accounting_invariant(self):
        self._assert_leak_is_caught_by(
            lambda assembly: assembly.records.pop(),
            "leak_acc",
            "G2 accession accounting",
        )

    def test_a_silent_entrez_drop_is_caught_by_the_accounting_invariant(self):
        def lose_a_gene_id(assembly):
            # one more GeneID came in than the ledger can explain
            assembly.stats["n_entrez_ids_raw"] += 1

        self._assert_leak_is_caught_by(
            lose_a_gene_id, "leak_gid", "G2 entrez accounting"
        )

    def test_an_orphaned_gene_id_is_recorded_on_the_entrez_axis(self):
        # P0DTC2 lists two GeneIDs; only one can be kept, and the other has no
        # other accession, so it becomes unresolvable — and must be recorded.
        rows = [
            ("P0DTC2", "UniProtKB-ID", "SPIKE_HUMAN"),
            ("P0DTC2", "GeneID", "100"),
            ("P0DTC2", "GeneID", "200"),
            ("P0DTC2", "Gene_Name", "GENEA"),
        ]
        idmapping = write_idmapping(self.path("orphan.dat"), rows=rows)
        proteome = write_proteome(self.path("orphan.fasta"), headers=[])
        out = self.path("orphan/identity_index.parquet")
        self.assertEqual(
            0,
            bii.run(["--idmapping", idmapping, "--proteome", proteome, "--out", out,
                     "--log-level", "ERROR"]),
        )
        index = IdentityIndex.load(out)
        self.assertEqual(index.by_entrez(100).accession, "P0DTC2")
        self.assertEqual(index.resolve_entrez(200).status, "UNKNOWN")
        notes = read_notes(out)
        entrez_rows = notes[notes["key_type"] == "entrez"]
        self.assertEqual(list(entrez_rows["key"]), ["200"])
        self.assertIn("no canonical accession", entrez_rows.iloc[0]["reason"])
        # and the discarded (accession, GeneID) pair is recorded too
        pair_rows = notes[(notes["key"] == "P0DTC2") & (notes["status"] == "AMBIGUOUS")]
        self.assertEqual(len(pair_rows), 1)
        self.assertIn("lists 2 GeneIDs", pair_rows.iloc[0]["reason"])

    def test_an_accession_with_no_symbol_gets_a_label_not_a_blank(self):
        rows = [
            ("P0DTC2", "UniProtKB-ID", "SPIKE_HUMAN"),
            ("P0DTC2", "GeneID", "100"),
        ]
        idmapping = write_idmapping(self.path("nosym.dat"), rows=rows)
        proteome = write_proteome(self.path("nosym.fasta"), headers=[])
        out = self.path("nosym/identity_index.parquet")
        self.assertEqual(
            0,
            bii.run(["--idmapping", idmapping, "--proteome", proteome, "--out", out,
                     "--log-level", "ERROR"]),
        )
        index = IdentityIndex.load(out)
        protein = index.by_accession("P0DTC2")
        # never a blank a downstream join could match
        self.assertEqual(protein.primary_symbol, "P0DTC2")
        self.assertEqual(protein.symbols, ("P0DTC2",))
        notes = read_notes(out)
        row = notes[(notes["key"] == "P0DTC2") & (notes["disposition"] == "retained")]
        self.assertEqual(len(row), 1)
        self.assertIn("no gene symbol", row.iloc[0]["reason"])

    def test_entry_name_falls_back_to_the_accession_rather_than_a_blank(self):
        rows = [("P0DTC2", "GeneID", "100"), ("P0DTC2", "Gene_Name", "GENEA")]
        idmapping = write_idmapping(self.path("noname.dat"), rows=rows)
        proteome = write_proteome(self.path("noname.fasta"), headers=[])
        out = self.path("noname/identity_index.parquet")
        bii.run(["--idmapping", idmapping, "--proteome", proteome, "--out", out,
                 "--log-level", "ERROR"])
        self.assertEqual(IdentityIndex.load(out).by_accession("P0DTC2").entry_name, "P0DTC2")

    def test_empty_input_is_refused_rather_than_producing_an_empty_index(self):
        idmapping = write_idmapping(self.path("empty.dat"), rows=[])
        proteome = write_proteome(self.path("empty.fasta"), headers=[])
        out = self.path("empty/identity_index.parquet")
        code = bii.run(
            ["--idmapping", idmapping, "--proteome", proteome, "--out", out,
             "--log-level", "CRITICAL"]
        )
        self.assertEqual(code, 3)
        self.assertFalse(os.path.exists(out))

    def test_min_records_gate_refuses_a_suspiciously_small_index(self):
        idmapping, proteome = self.standard_inputs()
        out = self.path("small/identity_index.parquet")
        code = bii.run(
            ["--idmapping", idmapping, "--proteome", proteome, "--out", out,
             "--min-records", "20000", "--log-level", "CRITICAL"]
        )
        self.assertEqual(code, 3)
        self.assertFalse(os.path.exists(out))


# ==========================================================================
# 6. the REDUCED_LOCAL fallback
# ==========================================================================


class TestReducedLocalMode(TempCaseMixin):
    def build_reduced(self, text=None, name="red", extra=()):
        legacy = write_legacy(self.path(name + ".csv"), text=text)
        out = self.path(name, "identity_index.parquet")
        code = bii.run(
            ["--from-legacy-gene-names", legacy, "--out", out, "--log-level", "ERROR"]
            + list(extra)
        )
        self.assertEqual(code, 0)
        return out

    def test_produces_a_valid_index_and_stamps_its_provenance(self):
        out = self.build_reduced()
        index = IdentityIndex.load(out)
        index.assert_self_consistent()
        self.assertEqual(len(index), 9)  # 10 rows, 1 mouse row excluded
        meta = load_json(bii.default_meta_path(out))
        self.assertEqual(meta["provenance"], bii.PROVENANCE_REDUCED)
        self.assertIn("REDUCED_LOCAL", meta["warning"])
        self.assertIn("not from the authoritative", meta["warning"].lower())
        # stamped in the parquet too, because downstream code loads the parquet
        # and may never open the sidecar
        self.assertEqual(bii.read_index_provenance(out), bii.PROVENANCE_REDUCED)
        self.assertTrue(any("REDUCED_LOCAL" in n for n in meta["notes"]))

    def test_labels_come_from_gene_names_not_from_the_corrupt_column(self):
        out = self.build_reduced()
        index = IdentityIndex.load(out)
        # The fixture's Mapped_Gene_Name says Q8NCT1 is 'AVPR2' and Q96RT7 is
        # 'TUBA1B'. Both are wrong, and both are what the campaign believed.
        self.assertEqual(index.label("Q8NCT1"), "ARRDC4")
        self.assertEqual(index.label("Q96RT7"), "TUBGCP6")
        self.assertEqual(index.label("P30518"), "AVPR2")
        self.assertEqual(index.label("Q13509"), "TUBB3")
        self.assertEqual(index.label("P01589"), "IL2RA")

    def test_the_corrupt_column_is_never_read(self):
        # Same rows, but Mapped_Gene_Name replaced with garbage, and a second
        # copy with the column removed entirely. All three must agree byte for
        # byte, which is only possible if the column is not read.
        lines = LEGACY_CSV.strip().split("\n")
        header = lines[0].split(",")
        garbage_rows = []
        for line in lines[1:]:
            fields = line.split(",")
            fields[-1] = "TOTALLY_WRONG"
            garbage_rows.append(",".join(fields))
        garbage = ",".join(header) + "\n" + "\n".join(garbage_rows) + "\n"
        without = ",".join(header[:-1]) + "\n" + "\n".join(
            ",".join(line.split(",")[:-1]) for line in lines[1:]
        ) + "\n"

        digests = [
            sha256_of(self.build_reduced(text=text, name=name))
            for text, name in ((None, "base"), (garbage, "garbage"), (without, "without"))
        ]
        self.assertEqual(len(set(digests)), 1)

    def test_missing_organism_column_is_an_error_not_a_skipped_filter(self):
        text = "Entry,Entry Name,Gene Names\nP30518,V2R_HUMAN,AVPR2\n"
        legacy = write_legacy(self.path("noorg.csv"), text=text)
        with self.assertRaises(bii.BuildInputError) as ctx:
            bii.load_legacy_gene_names(legacy)
        message = str(ctx.exception)
        self.assertIn("Organism", message)
        self.assertIn("widen the universe", message)
        out = self.path("noorg/identity_index.parquet")
        self.assertEqual(
            2,
            bii.run(["--from-legacy-gene-names", legacy, "--out", out,
                     "--log-level", "CRITICAL"]),
        )
        self.assertFalse(os.path.exists(out))

    def test_isoform_entry_in_the_legacy_file_is_excluded_and_recorded(self):
        # The legacy loader does NOT pre-filter isoforms the way the idmapping
        # parser does, so the assembly-level G7 guard is the only thing standing
        # between P30518-2 and the index.
        text = LEGACY_CSV + "P30518-2,V2R_HUMAN,AVPR2,Homo sapiens (Human),AVPR2\n"
        out = self.build_reduced(text=text, name="iso")
        index = IdentityIndex.load(out)
        self.assertNotIn("P30518-2", index)
        self.assertEqual(len(index), 9)
        rejects = read_unresolved(bii.default_unresolved_path(out))
        row = rejects[rejects["key"] == "P30518-2"]
        self.assertEqual(len(row), 1)
        self.assertIn("isoform", row.iloc[0]["reason"])
        self.assertIn("P30518", row.iloc[0]["reason"])

    def test_assemble_records_excludes_an_isoform_key_directly(self):
        entries = {
            "P30518": bii.RawEntry("P30518", "V2R_HUMAN", ["AVPR2"], [], [554], True, True),
            "P30518-2": bii.RawEntry("P30518-2", "V2R_HUMAN", ["AVPR2"], [], [554], True, True),
        }
        assembly = bii.assemble_records(entries, {})
        self.assertEqual([p.accession for p in assembly.records], ["P30518"])
        excluded = [
            r for r in assembly.rejects
            if r.key == "P30518-2" and r.disposition == "excluded"
        ]
        self.assertEqual(len(excluded), 1)
        self.assertIn("isoform", excluded[0].reason)

    def test_empty_cells_do_not_crash_and_are_recorded(self):
        # Regression: pandas returns float('nan') for an empty field, and NaN is
        # TRUTHY, so `row.get(col) or ""` never fires and `.split()` explodes.
        # Two of the 4,585 real rows have an empty 'Gene Names'; one has an
        # empty 'Entry Name'.
        text = LEGACY_CSV + (
            "P0DOX5,IGG1_HUMAN,,Homo sapiens (Human),IGHG1\n"
            "Q96FB2,,,Homo sapiens (Human),SOMETHING\n"
        )
        out = self.build_reduced(text=text, name="blanks")
        index = IdentityIndex.load(out)
        self.assertEqual(len(index), 11)
        # never a blank label a downstream join could match
        self.assertEqual(index.label("P0DOX5"), "P0DOX5")
        self.assertEqual(index.by_accession("Q96FB2").primary_symbol, "Q96FB2")
        self.assertEqual(index.by_accession("Q96FB2").entry_name, "Q96FB2")
        notes = read_notes(out)
        recorded = set(notes[notes["disposition"] == "retained"]["key"])
        self.assertEqual({"P0DOX5", "Q96FB2"} & recorded, {"P0DOX5", "Q96FB2"})

    def test_non_human_rows_are_excluded_and_recorded(self):
        out = self.build_reduced()
        self.assertNotIn("Q9XYZ9", IdentityIndex.load(out))
        rejects = read_unresolved(bii.default_unresolved_path(out))
        row = rejects[rejects["key"] == "Q9XYZ9"]
        self.assertEqual(len(row), 1)
        self.assertIn("non-human", row.iloc[0]["reason"])
        self.assertEqual(row.iloc[0]["disposition"], "excluded")

    def test_disagreeing_duplicate_accessions_are_refused(self):
        text = LEGACY_CSV + "P30518,V2R_HUMAN,SOMETHINGELSE,Homo sapiens (Human),AVPR2\n"
        legacy = write_legacy(self.path("dup.csv"), text=text)
        with self.assertRaises(bii.BuildInputError) as ctx:
            bii.load_legacy_gene_names(legacy)
        self.assertIn("last-duplicate-wins", str(ctx.exception))
        out = self.path("dup/identity_index.parquet")
        self.assertEqual(
            2,
            bii.run(["--from-legacy-gene-names", legacy, "--out", out,
                     "--log-level", "CRITICAL"]),
        )
        self.assertFalse(os.path.exists(out))

    def test_identical_duplicate_rows_collapse_to_one(self):
        text = LEGACY_CSV + "P30518,V2R_HUMAN,AVPR2 ADHR DIR DIR3 V2R,Homo sapiens (Human),AVPR2\n"
        out = self.build_reduced(text=text, name="dup2")
        self.assertEqual(len(IdentityIndex.load(out)), 9)

    def test_hyphenated_symbol_and_synonyms_survive(self):
        index = IdentityIndex.load(self.build_reduced())
        self.assertEqual(index.by_accession("P03886").primary_symbol, "MT-ND1")
        self.assertEqual(
            index.by_accession("P30518").symbols,
            ("AVPR2", "ADHR", "DIR", "DIR3", "V2R"),
        )

    def test_without_hgnc_there_are_no_entrez_ids_and_g5_still_holds(self):
        index = IdentityIndex.load(self.build_reduced())
        report = index.cardinality_report()
        self.assertEqual(report["n_entrez"], 0)
        self.assertLessEqual(report["max_collapse"], 1)
        self.assertEqual(index.resolve_entrez(554).status, "UNKNOWN")

    def test_with_hgnc_entrez_ids_are_attached_by_accession(self):
        legacy = write_legacy(self.path("hg.csv"))
        hgnc = write_hgnc(self.path("hgnc.txt"))
        out = self.path("hg", "identity_index.parquet")
        self.assertEqual(
            0,
            bii.run(["--from-legacy-gene-names", legacy, "--hgnc", hgnc, "--out", out,
                     "--log-level", "ERROR"]),
        )
        index = IdentityIndex.load(out)
        self.assertEqual(index.by_entrez(554).accession, "P30518")
        self.assertEqual(index.by_entrez(10376).accession, "P68363")
        self.assertEqual(index.cardinality_report()["max_collapse"], 1)
        # HGNC synonyms are added, but the primary symbol still comes from the
        # accession's own UniProt gene names
        self.assertEqual(index.by_accession("P68363").primary_symbol, "TUBA1B")
        self.assertIn("K-ALPHA-1", index.by_accession("P68363").symbols)

    def test_reduced_index_passes_the_known_issue_regression_assertion(self):
        frame = pd.read_parquet(self.build_reduced())
        bad = [
            row.accession
            for row in frame.itertuples()
            if row.primary_symbol not in list(row.symbols)
        ]
        self.assertEqual(bad, [])


# ==========================================================================
# 7. enrichment may not change the row count
# ==========================================================================


class TestEnrichmentDoesNotFanOut(TempCaseMixin):
    def test_hgnc_rows_referencing_one_accession_do_not_add_index_rows(self):
        # Two HGNC rows both name P68363. A careless merge on uniprot_ids would
        # duplicate the accession — that is the parse_prepare.py:1171 fan-out
        # (+410,087 rows) in miniature.
        idmapping, proteome = self.standard_inputs()
        hgnc = write_hgnc(
            self.path("hgnc.txt"),
            rows=HGNC_ROWS + [
                "HGNC:12413\tTUBA1BP\tApproved\t777777\tP68363\t\tTUBA1B-DUP\n",
            ],
        )
        out = self.path("hg/identity_index.parquet")
        self.assertEqual(
            0,
            bii.run(["--idmapping", idmapping, "--proteome", proteome, "--hgnc", hgnc,
                     "--out", out, "--log-level", "ERROR"]),
        )
        without = self.build_standard(out_name="plain/identity_index.parquet")
        self.assertEqual(len(IdentityIndex.load(out)), len(IdentityIndex.load(without)))
        meta = load_json(bii.default_meta_path(out))
        report = {r.get("name"): r for r in meta["invariants"]}[
            "enrichment (proteome + HGNC) must not add rows"
        ]
        self.assertEqual(report["row_delta"], 0)
        self.assertTrue(report["ok"])
        # the extra HGNC alias is present as a synonym, without a new row
        self.assertIn("TUBA1B-DUP", IdentityIndex.load(out).by_accession("P68363").symbols)

    def test_withdrawn_hgnc_rows_are_ignored(self):
        legacy = write_legacy(self.path("legacy.csv"))
        hgnc = write_hgnc(self.path("hgnc.txt"))  # contains an "Entry Withdrawn" row
        out = self.path("wd/identity_index.parquet")
        bii.run(["--from-legacy-gene-names", legacy, "--hgnc", hgnc, "--out", out,
                 "--log-level", "ERROR"])
        index = IdentityIndex.load(out)
        self.assertNotIn("WITHDRAWN1", index.by_accession("P68363").symbols)
        self.assertEqual(index.resolve_entrez(111111).status, "UNKNOWN")


# ==========================================================================
# 8. parsers
# ==========================================================================


class TestParsers(TempCaseMixin):
    def test_parse_idmapping_reads_only_the_four_needed_types(self):
        path = write_idmapping(self.path("idmapping.dat"))
        entries, excluded, stats = bii.parse_idmapping(path)
        self.assertEqual(stats["n_lines"], len(IDMAPPING_ROWS))
        self.assertEqual(stats["n_isoform_rows_skipped"], 3)
        self.assertNotIn("P30518-2", entries)
        self.assertIn("P30518-2", excluded)
        entry = entries["P30518"]
        self.assertEqual(entry.entry_name, "V2R_HUMAN")
        self.assertEqual(entry.entrez_ids, [554])
        self.assertEqual(entry.gene_names, ["AVPR2"])
        self.assertEqual(sorted(entry.synonyms), ["ADHR", "DIR", "V2R"])

    def test_parse_idmapping_counts_malformed_lines_instead_of_crashing(self):
        path = self.path("broken.dat")
        with open(path, "w") as handle:
            handle.write("P30518\tGene_Name\tAVPR2\n")
            handle.write("this line has no tabs at all\n")
            handle.write("P30518\tGeneID\tnot-a-number\n")
            handle.write("\n")
        entries, _, stats = bii.parse_idmapping(path)
        self.assertEqual(stats["n_rows_malformed"], 2)
        self.assertEqual(entries["P30518"].entrez_ids, [])

    def test_parse_proteome_fasta_reads_review_status_and_gn(self):
        path = write_proteome(self.path("p.fasta"))
        entries, stats = bii.parse_proteome_fasta(path)
        self.assertEqual(stats["n_accessions"], len(PROTEOME_HEADERS))
        self.assertTrue(entries["P30518"]["reviewed"])
        self.assertFalse(entries["A0A024R1R8"]["reviewed"])
        self.assertEqual(entries["P30518"]["symbol"], "AVPR2")
        self.assertEqual(entries["P03886"]["symbol"], "MT-ND1")
        self.assertEqual(entries["P30518"]["entry_name"], "V2R_HUMAN")

    def test_parse_hgnc_splits_pipe_separated_fields(self):
        table = parse = bii.parse_hgnc(write_hgnc(self.path("h.txt")))
        self.assertEqual(table.entrez_to_symbol[554], "AVPR2")
        self.assertEqual(table.accession_to_entrez["P30518"], [554])
        self.assertIn("V2R", table.accession_to_symbols["P30518"])
        self.assertIn("ADHR", table.accession_to_symbols["P30518"])
        self.assertIn("DIR", table.accession_to_symbols["P30518"])
        self.assertNotIn(111111, table.entrez_to_symbol)  # withdrawn
        self.assertEqual(parse.stats["n_approved"], len(HGNC_ROWS) - 1)


# ==========================================================================
# 9. CLI plumbing
# ==========================================================================


class TestCli(TempCaseMixin):
    def test_default_output_paths(self):
        out = self.build_standard(out_name="run/identity_index.parquet")
        directory = os.path.dirname(out)
        for name in (
            "identity_index.parquet",
            "identity_index.meta.json",
            "identity_index.invariants.json",
            "identity_index.notes.csv",
            "unresolved_targets.csv",
        ):
            self.assertTrue(os.path.exists(os.path.join(directory, name)), name)

    def test_explicit_meta_and_unresolved_paths_are_honoured(self):
        idmapping, proteome = self.standard_inputs()
        out = self.path("x/index.parquet")
        meta = self.path("y/meta.json")
        unresolved = self.path("z/rejects.csv")
        notes = self.path("z/notes.csv")
        self.assertEqual(
            0,
            bii.run(["--idmapping", idmapping, "--proteome", proteome, "--out", out,
                     "--meta", meta, "--unresolved", unresolved, "--notes", notes,
                     "--log-level", "ERROR"]),
        )
        self.assertTrue(os.path.exists(meta))
        self.assertTrue(os.path.exists(unresolved))
        self.assertTrue(os.path.exists(notes))
        outputs = load_json(meta)["outputs"]
        self.assertEqual(outputs["unresolved_path"], unresolved)
        self.assertEqual(outputs["notes_path"], notes)

    def test_out_is_required(self):
        stderr, sys.stderr = sys.stderr, io.StringIO()
        try:
            with self.assertRaises(SystemExit):
                bii.build_parser().parse_args(["--idmapping", "x"])
        finally:
            sys.stderr = stderr

    def test_index_loads_with_the_pinned_sha256(self):
        out = self.build_standard()
        meta = load_json(bii.default_meta_path(out))
        index = IdentityIndex.load(out, expect_sha256=meta["outputs"]["index_sha256"])
        self.assertEqual(len(index), 8)
        with self.assertRaises(Exception):
            IdentityIndex.load(out, expect_sha256="0" * 64)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
