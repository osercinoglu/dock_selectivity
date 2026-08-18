"""Tests for score_cavity_quality.py and dock_selectivity.quality.

Pocket confidence is a *filter*, and this repository exists because a filter that
quietly changed the size of a scientific dataset went unnoticed for a year. So the
cases under test are the ones where a wrong answer would look plausible:

* a pocket whose archive is missing, whose cavity member is absent, or whose
  member holds no ATOM record must become a **reject with a reason**, never a
  zero-filled row -- a zero row reads downstream as "a terrible pocket" and fails
  every cutoff without saying why;
* ``HETATM`` must be ignored, because the same archive carries ``surface_*.pdb``
  and ``vacant_*.pdb`` members that are HETATM grid points with no pLDDT;
* a residue's pLDDT is the *mean* of its atoms' B-factors, so an odd file with
  differing values per atom must average rather than take the first;
* ``fraction_at(frame, 70)`` must reproduce ``frac_ge_70`` exactly, because
  recomputing from ``residue_plddt`` is the entire reason the per-residue values
  are stored;
* ``--required`` must restrict the work and ``--resume`` must not redo it.

Hermetic: every archive, manifest and output lives inside a
``tempfile.TemporaryDirectory()``. No network, no /opt/data.
"""

from __future__ import annotations

import io
import os
import tarfile
import tempfile
import unittest

import pandas as pd

import score_cavity_quality as scq
from dock_selectivity import quality, schemas
from tests.support import run_in_temp_cwd

setUpModule, tearDownModule = run_in_temp_cwd()


# ---------------------------------------------------------------------------
# synthetic PDB / tarball fixtures
# ---------------------------------------------------------------------------


def atom_line(
    record="ATOM",
    serial=1,
    name="CA",
    resname="ALA",
    chain="A",
    resseq=1,
    icode=" ",
    b_factor=90.0,
    element="C",
):
    """One PDB coordinate line with exact column placement.

    The metric contract keys a residue on ``(line[21], line[22:27].strip())`` and
    reads the B-factor from ``line[60:66]``; this builder is asserted against
    those offsets in :class:`TestFixtureColumns` so a broken fixture cannot make
    a broken parser look right.
    """
    return "{0:<6}{1:>5} {2:<4}{3:1}{4:>3} {5:1}{6:>4}{7:1}   {8:8.3f}{9:8.3f}{10:8.3f}{11:6.2f}{12:6.2f}          {13:>2}".format(
        record, serial, name, " ", resname, chain, resseq, icode,
        1.0, 2.0, 3.0, 1.00, b_factor, element,
    )


def pocket_pdb(residues, atoms_per_residue=1, chain="A", record="ATOM"):
    """A pocket PDB whose residue *k* carries pLDDT ``residues[k]``.

    ``residues`` may hold a float (all atoms of that residue share it) or a list
    of floats (one per atom, so the residue mean is the average of them).
    """
    lines = ["HEADER    SYNTHETIC", "MODEL        1"]
    serial = 0
    for index, value in enumerate(residues, start=1):
        values = value if isinstance(value, (list, tuple)) else [value] * atoms_per_residue
        for atom, b_factor in enumerate(values):
            serial += 1
            lines.append(
                atom_line(
                    record=record,
                    serial=serial,
                    name="C{0}".format(atom),
                    chain=chain,
                    resseq=index,
                    b_factor=b_factor,
                )
            )
    lines += ["TER", "ENDMDL", "END", ""]
    return "\n".join(lines)


def _add_member(tar, name, text):
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def make_archive(
    directory,
    accession="P30518",
    fragment="F1",
    cavity=1,
    pdb_text=None,
    include_cavity_member=True,
):
    """Write one CavitySpace-shaped archive and return its file name.

    Always carries a ``surface_*.pdb`` decoy of HETATM grid points, exactly as the
    real archives do -- picking "the first .pdb member" instead of the cavity one
    would score those instead, and they have no pLDDT.
    """
    stem = "AF-{0}-{1}-model_v1".format(accession, fragment)
    name = "{0}_cavity_result_{1}.tar.gz".format(stem, cavity)
    surface = "\n".join(
        [
            "HEADER    POCKET SURFACE",
            "HETATM    1 C   POK   1         3.000   3.500   6.500  1.00",
            "",
        ]
    )
    with tarfile.open(os.path.join(directory, name), "w:gz") as tar:
        if include_cavity_member:
            _add_member(
                tar,
                "{0}_cavity_{1}.pdb".format(stem, cavity),
                pdb_text if pdb_text is not None else pocket_pdb([90.0, 80.0]),
            )
        _add_member(tar, "{0}_surface_{1}.pdb".format(stem, cavity), surface)
    return name


def write_manifest(path, entries):
    """A cavity_manifest.csv over ``(accession, fragment, cavity, tarball)`` rows."""
    pd.DataFrame(
        [
            {
                "UniProt_ID": accession,
                "Fragment_ID": fragment,
                "Cavity_Index": cavity,
                "Tarball": tarball,
                "Pocket_PDB": "extracted/{0}".format(tarball),
            }
            for accession, fragment, cavity, tarball in entries
        ],
        columns=["UniProt_ID", "Fragment_ID", "Cavity_Index", "Tarball", "Pocket_PDB"],
    ).to_csv(path, index=False)


def run_cli(*args):
    """``score_cavity_quality.main`` with progress off, returning the exit code."""
    return scq.main(list(args) + ["--no-progress", "--quiet"])


class Fixture:
    """A temp directory holding tarballs/, a manifest and the output paths."""

    def __init__(self, stack):
        self.root = stack
        self.tarballs = os.path.join(stack, "tarballs")
        self.run = os.path.join(stack, "run")
        os.makedirs(self.tarballs)
        os.makedirs(os.path.join(self.run, "manifests"))
        self.manifest = os.path.join(self.run, "manifests", "cavity_manifest.csv")
        self.out = os.path.join(self.run, "manifests", "cavity_quality.csv")
        self.rejects = os.path.join(self.run, "manifests", "cavity_quality_rejects.csv")

    def cli(self, *extra):
        return run_cli(
            "--tarballs", self.tarballs,
            "--manifest", self.manifest,
            "--out", self.out,
            "--rejects", self.rejects,
            "--run-dir", self.run,
            *extra
        )

    def quality(self):
        return quality.load_cavity_quality(self.out)

    def reject_frame(self):
        return pd.read_csv(self.rejects, keep_default_na=False, na_values=[""])


# ---------------------------------------------------------------------------
# the fixture itself
# ---------------------------------------------------------------------------


class TestFixtureColumns(unittest.TestCase):
    """The synthetic PDB must place its fields where the contract reads them."""

    def test_chain_residue_and_bfactor_land_in_the_declared_columns(self):
        line = atom_line(chain="B", resseq=137, icode="A", b_factor=76.25)
        self.assertEqual(line[21], "B")
        self.assertEqual(line[22:27].strip(), "137A")
        self.assertEqual(line[60:66].strip(), "76.25")
        self.assertTrue(line.startswith("ATOM"))

    def test_a_real_cavitySpace_line_parses(self):
        real = (
            "ATOM     61  CG  TYR A   8       3.860   6.337  -9.914  "
            "1.00 98.75           C"
        )
        values, n_atoms = scq.residue_plddt_from_pdb(real)
        self.assertEqual(n_atoms, 1)
        self.assertEqual(values, [98.75])


# ---------------------------------------------------------------------------
# the metric
# ---------------------------------------------------------------------------


class TestResidueParsing(unittest.TestCase):
    def test_known_bfactors_give_exactly_the_expected_fractions(self):
        plddt = [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 98.0, 99.0]
        values, n_atoms = scq.residue_plddt_from_pdb(pocket_pdb(plddt))
        self.assertEqual(values, plddt)
        self.assertEqual(n_atoms, 10)

        row = scq.summarize_residues(values, n_atoms)
        self.assertEqual(row["n_residues"], 10)
        self.assertEqual(row["n_atoms"], 10)
        self.assertAlmostEqual(row["mean_plddt"], 71.2)
        self.assertEqual(row["median_plddt"], 75.0)
        self.assertEqual(row["min_plddt"], 30.0)
        self.assertEqual(row["max_plddt"], 99.0)
        self.assertEqual(row["frac_ge_50"], 0.8)
        self.assertEqual(row["frac_ge_70"], 0.6)
        self.assertEqual(row["frac_ge_90"], 0.4)
        self.assertEqual(
            row["residue_plddt"],
            "30.00;40.00;50.00;60.00;70.00;80.00;90.00;95.00;98.00;99.00",
        )

    def test_a_residue_whose_atoms_differ_averages(self):
        """AlphaFold repeats one value per atom; the mean is robust to a file that does not."""
        text = pocket_pdb([[70.0, 80.0], [60.0, 60.0, 90.0]])
        values, n_atoms = scq.residue_plddt_from_pdb(text)
        self.assertEqual(n_atoms, 5)
        self.assertEqual(values, [75.0, 70.0])
        row = scq.summarize_residues(values, n_atoms)
        self.assertEqual(row["n_residues"], 2)
        self.assertEqual(row["residue_plddt"], "75.00;70.00")
        self.assertEqual(row["frac_ge_70"], 1.0)

    def test_hetatm_is_ignored(self):
        atoms = pocket_pdb([90.0, 92.0])
        grid = pocket_pdb([10.0, 10.0, 10.0], record="HETATM")
        values, n_atoms = scq.residue_plddt_from_pdb(atoms + "\n" + grid)
        self.assertEqual(n_atoms, 2)
        self.assertEqual(values, [90.0, 92.0])

    def test_a_member_with_no_atom_records_raises_rather_than_scoring_zero(self):
        with self.assertRaises(scq.PocketParseError) as ctx:
            scq.residue_plddt_from_pdb("HEADER  x\nHETATM    1 C   POK   1  1.0 2.0\n")
        self.assertEqual(ctx.exception.reason, "NO_ATOM_RECORDS")

    def test_insertion_codes_separate_residues(self):
        text = "\n".join(
            [
                atom_line(serial=1, chain="A", resseq=52, icode=" ", b_factor=90.0),
                atom_line(serial=2, chain="A", resseq=52, icode="A", b_factor=40.0),
                atom_line(serial=3, chain="B", resseq=52, icode=" ", b_factor=60.0),
            ]
        )
        values, n_atoms = scq.residue_plddt_from_pdb(text)
        self.assertEqual(n_atoms, 3)
        self.assertEqual(values, [90.0, 40.0, 60.0])

    def test_a_numeric_insertion_code_does_not_collide_with_another_residue(self):
        """Residue 8 with insertion code "1" is not residue 81.

        The key was resSeq and iCode concatenated and stripped, so both flattened
        to "81" and the two residues were averaged into one. No AlphaFold model
        carries a numeric insertion code, but nothing about this parser announces
        that it may only be pointed at AlphaFold models.
        """
        text = "\n".join(
            [
                atom_line(serial=1, chain="A", resseq=8, icode="1", b_factor=20.0),
                atom_line(serial=2, chain="A", resseq=81, icode=" ", b_factor=95.0),
            ]
        )
        values, n_atoms = scq.residue_plddt_from_pdb(text)
        self.assertEqual(n_atoms, 2)
        self.assertEqual(values, [20.0, 95.0])

    def test_a_non_numeric_b_factor_is_reported_not_dropped(self):
        broken = atom_line(b_factor=90.0)[:60] + "  abcd" + atom_line()[66:]
        with self.assertRaises(scq.PocketParseError) as ctx:
            scq.residue_plddt_from_pdb(broken)
        self.assertEqual(ctx.exception.reason, "BAD_B_FACTOR")


# ---------------------------------------------------------------------------
# the archive reader
# ---------------------------------------------------------------------------


class TestArchiveReading(unittest.TestCase):
    def test_the_cavity_member_is_read_not_the_surface_decoy(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = make_archive(tmp, pdb_text=pocket_pdb([88.0]))
            text = scq.read_cavity_member(os.path.join(tmp, name), 1)
            self.assertIn("ATOM", text)
            self.assertNotIn("POK", text)

    def test_cavity_1_does_not_match_cavity_11(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = make_archive(tmp, cavity=11, pdb_text=pocket_pdb([88.0]))
            with self.assertRaises(scq.PocketParseError) as ctx:
                scq.read_cavity_member(os.path.join(tmp, name), 1)
            self.assertEqual(ctx.exception.reason, "NO_CAVITY_MEMBER")

    def test_an_unreadable_archive_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "broken.tar.gz")
            with open(path, "wb") as handle:
                handle.write(b"this is not a gzip stream")
            with self.assertRaises(scq.PocketParseError) as ctx:
                scq.read_cavity_member(path, 1)
            self.assertEqual(ctx.exception.reason, "UNREADABLE_ARCHIVE")


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------


class TestScoreCavitiesAccounting(unittest.TestCase):
    def test_every_unscoreable_cavity_is_a_reject_and_none_is_a_zero_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            good = make_archive(
                fixture.tarballs, "P30518", "F1", 1, pocket_pdb([90.0, 40.0])
            )
            no_member = make_archive(
                fixture.tarballs, "Q8NCT1", "F1", 1, include_cavity_member=False
            )
            empty = make_archive(
                fixture.tarballs,
                "P68363",
                "F1",
                1,
                pdb_text="HEADER  empty\nHETATM    1 C   POK   1\n",
            )
            write_manifest(
                fixture.manifest,
                [
                    ("P30518", "F1", 1, good),
                    ("Q8NCT1", "F1", 1, no_member),
                    ("P68363", "F1", 1, empty),
                    ("P11362", "F1", 1, "AF-P11362-F1-model_v1_cavity_result_1.tar.gz"),
                ],
            )

            self.assertEqual(fixture.cli(), 0)

            scored = fixture.quality()
            self.assertEqual(list(scored["accession"]), ["P30518"])
            self.assertEqual(int(scored.loc[0, "n_residues"]), 2)

            rejects = fixture.reject_frame()
            self.assertEqual(
                dict(zip(rejects["accession"], rejects["reason"])),
                {
                    "Q8NCT1": "NO_CAVITY_MEMBER",
                    "P68363": "NO_ATOM_RECORDS",
                    "P11362": "MISSING_ARCHIVE",
                },
            )
            # nothing scored 0: the failures are absent from the table entirely
            self.assertEqual(len(scored) + len(rejects), 4)
            self.assertTrue((scored["n_residues"] >= 1).all())

    def test_required_restricts_the_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            entries = []
            for accession in ("P30518", "Q8NCT1", "P68363"):
                name = make_archive(
                    fixture.tarballs, accession, "F1", 1, pocket_pdb([90.0])
                )
                entries.append((accession, "F1", 1, name))
            write_manifest(fixture.manifest, entries)

            required = os.path.join(tmp, "redock_queue.csv")
            pd.DataFrame(
                [{"accession": "Q8NCT1", "fragment_id": "F1", "cavity_index": 1}]
            ).to_csv(required, index=False)

            self.assertEqual(fixture.cli("--required", required), 0)
            self.assertEqual(list(fixture.quality()["accession"]), ["Q8NCT1"])
            self.assertTrue(fixture.reject_frame().empty)

    def test_required_naming_a_cavity_the_manifest_lacks_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            name = make_archive(fixture.tarballs, "P30518", "F1", 1, pocket_pdb([90.0]))
            write_manifest(fixture.manifest, [("P30518", "F1", 1, name)])

            required = os.path.join(tmp, "queue.csv")
            pd.DataFrame(
                [
                    {"accession": "P30518", "fragment_id": "F1", "cavity_index": 1},
                    {"accession": "P99999", "fragment_id": "F2", "cavity_index": 4},
                ]
            ).to_csv(required, index=False)

            self.assertEqual(fixture.cli("--required", required), 0)
            rejects = fixture.reject_frame()
            self.assertEqual(list(rejects["reason"]), ["NOT_IN_MANIFEST"])
            self.assertEqual(list(rejects["accession"]), ["P99999"])

    def test_a_thread_pool_produces_the_same_rows_in_the_same_order(self):
        """--threads is a throughput knob, never a change of answer or of order."""
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            entries = []
            for index, accession in enumerate(("P30518", "Q8NCT1", "P68363", "P11362")):
                name = make_archive(
                    fixture.tarballs,
                    accession,
                    "F1",
                    1,
                    pocket_pdb([40.0 + index, 90.0, 70.0]),
                )
                entries.append((accession, "F1", 1, name))
            write_manifest(fixture.manifest, entries)

            self.assertEqual(fixture.cli(), 0)
            single = fixture.quality()

            os.unlink(fixture.out)
            self.assertEqual(fixture.cli("--threads", "4"), 0)
            pd.testing.assert_frame_equal(single, fixture.quality())

    def test_fragment_ids_normalise_so_a_bare_1_matches_F1(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            name = make_archive(fixture.tarballs, "P30518", "F1", 2, pocket_pdb([90.0]))
            write_manifest(fixture.manifest, [("P30518", 1, 2, name)])
            required = os.path.join(tmp, "queue.csv")
            pd.DataFrame(
                [{"accession": "P30518", "fragment_id": 1, "cavity_index": 2}]
            ).to_csv(required, index=False)

            self.assertEqual(fixture.cli("--required", required), 0)
            frame = fixture.quality()
            self.assertEqual(list(frame["fragment_id"]), ["F1"])
            self.assertEqual(list(frame["cavity_index"]), [2])


class TestResume(unittest.TestCase):
    def test_resume_does_not_rescore_and_extends_instead_of_rebuilding(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            first = make_archive(fixture.tarballs, "P30518", "F1", 1, pocket_pdb([90.0]))
            second = make_archive(
                fixture.tarballs, "Q8NCT1", "F1", 1, pocket_pdb([40.0, 42.0])
            )
            write_manifest(
                fixture.manifest,
                [("P30518", "F1", 1, first), ("Q8NCT1", "F1", 1, second)],
            )

            only_first = os.path.join(tmp, "first.csv")
            pd.DataFrame(
                [{"accession": "P30518", "fragment_id": "F1", "cavity_index": 1}]
            ).to_csv(only_first, index=False)
            self.assertEqual(fixture.cli("--required", only_first), 0)
            self.assertEqual(list(fixture.quality()["accession"]), ["P30518"])

            # Delete the first archive. A rescore would now report MISSING_ARCHIVE;
            # --resume must not touch it, and must keep its row verbatim.
            os.unlink(os.path.join(fixture.tarballs, first))

            self.assertEqual(fixture.cli("--resume"), 0)
            frame = fixture.quality()
            self.assertEqual(list(frame["accession"]), ["P30518", "Q8NCT1"])
            self.assertEqual(float(frame.loc[0, "mean_plddt"]), 90.0)
            self.assertEqual(int(frame.loc[1, "n_residues"]), 2)
            self.assertTrue(fixture.reject_frame().empty)

    def test_without_resume_the_output_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            first = make_archive(fixture.tarballs, "P30518", "F1", 1, pocket_pdb([90.0]))
            write_manifest(fixture.manifest, [("P30518", "F1", 1, first)])
            self.assertEqual(fixture.cli(), 0)

            os.unlink(os.path.join(fixture.tarballs, first))
            self.assertEqual(fixture.cli(), 0)
            self.assertTrue(fixture.quality().empty)
            self.assertEqual(list(fixture.reject_frame()["reason"]), ["MISSING_ARCHIVE"])


class TestCliRefusals(unittest.TestCase):
    def test_a_missing_manifest_exits_2_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            self.assertFalse(os.path.exists(fixture.manifest))
            self.assertEqual(fixture.cli(), 2)
            self.assertFalse(os.path.exists(fixture.out))
            self.assertFalse(os.path.exists(fixture.rejects))

    def test_a_missing_required_file_exits_2_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            write_manifest(fixture.manifest, [])
            self.assertEqual(fixture.cli("--required", os.path.join(tmp, "nope.csv")), 2)
            self.assertFalse(os.path.exists(fixture.out))
            self.assertFalse(os.path.exists(fixture.rejects))

    def test_a_missing_tarball_folder_exits_2_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            write_manifest(fixture.manifest, [])
            code = run_cli(
                "--tarballs", os.path.join(tmp, "absent"),
                "--manifest", fixture.manifest,
                "--out", fixture.out,
                "--rejects", fixture.rejects,
                "--run-dir", fixture.run,
            )
            self.assertEqual(code, 2)
            self.assertFalse(os.path.exists(fixture.out))

    def test_every_flag_is_documented_and_the_runtime_flags_are_present(self):
        parser = scq.build_parser()
        flags = set()
        for action in parser._actions:  # argparse exposes no public accessor
            flags.update(action.option_strings)
            if action.option_strings:
                self.assertTrue(action.help, action.option_strings)
        for flag in ("--out-dir", "--log-file", "--log-level", "--quiet",
                     "--notify", "--notify-on", "--notify-title", "--run-dir"):
            self.assertIn(flag, flags)

    def test_the_logger_name_is_the_step_name(self):
        self.assertEqual(scq.LOG.name, scq.STEP_NAME)


# ---------------------------------------------------------------------------
# the schema and the loader
# ---------------------------------------------------------------------------


def quality_row(**overrides):
    row = {
        "accession": "P30518",
        "fragment_id": "F1",
        "cavity_index": 1,
        "n_residues": 2,
        "n_atoms": 2,
        "mean_plddt": 85.0,
        "median_plddt": 85.0,
        "min_plddt": 80.0,
        "max_plddt": 90.0,
        "frac_ge_50": 1.0,
        "frac_ge_70": 1.0,
        "frac_ge_90": 0.5,
        "residue_plddt": "80.00;90.00",
    }
    row.update(overrides)
    return row


class TestCavityQualitySchema(unittest.TestCase):
    def test_the_schema_is_registered(self):
        self.assertIn("cavity_quality", schemas.schema_names())
        schema = schemas.get_schema("cavity_quality")
        self.assertEqual(
            schema.column_names,
            [
                "accession", "fragment_id", "cavity_index",
                "n_residues", "n_atoms",
                "mean_plddt", "median_plddt", "min_plddt", "max_plddt",
                "frac_ge_50", "frac_ge_70", "frac_ge_90",
                "residue_plddt",
            ],
        )
        self.assertEqual(
            schema.primary_key, ("accession", "fragment_id", "cavity_index")
        )

    def test_a_zero_residue_row_is_rejected(self):
        frame = pd.DataFrame([quality_row(n_residues=0, residue_plddt="")])
        with self.assertRaises(schemas.SchemaViolation):
            schemas.write_checked(frame, "unused.csv", "cavity_quality")

    def test_a_truncated_residue_list_is_rejected(self):
        frame = pd.DataFrame([quality_row(n_residues=3)])
        with self.assertRaises(schemas.SchemaViolation):
            schemas.write_checked(frame, "unused.csv", "cavity_quality")

    def test_a_duplicate_pocket_key_is_rejected(self):
        frame = pd.DataFrame([quality_row(), quality_row()])
        with self.assertRaises(schemas.SchemaViolation):
            schemas.write_checked(frame, "unused.csv", "cavity_quality")


class TestQualityLoader(unittest.TestCase):
    def test_fraction_at_reproduces_frac_ge_70_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            entries = []
            pockets = {
                "P30518": [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 98.0, 99.0],
                "Q8NCT1": [69.99, 70.00, 70.01],
                "P68363": [34.1, 98.6],
                # 29/30 and 2/31 are the interesting cases: neither survives a
                # CSV round trip through pandas' default (fast, not correctly
                # rounded) float parser, so this is what pins
                # read_checked(float_precision='round_trip').
                "P11362": [95.0] * 29 + [30.0],
                "P00533": [95.0] * 2 + [40.0] * 29,
            }
            for accession, values in pockets.items():
                name = make_archive(
                    fixture.tarballs, accession, "F1", 1, pocket_pdb(values)
                )
                entries.append((accession, "F1", 1, name))
            write_manifest(fixture.manifest, entries)
            self.assertEqual(fixture.cli(), 0)

            frame = fixture.quality()
            for cutoff in (50, 70, 90):
                self.assertEqual(
                    list(quality.fraction_at(frame, float(cutoff))),
                    list(frame["frac_ge_{0}".format(cutoff)]),
                    "fraction_at({0}) must reproduce frac_ge_{0}".format(cutoff),
                )
            # the two ratios that do not survive a naive CSV round trip
            self.assertEqual(float(frame.loc[1, "frac_ge_70"]), 29.0 / 30.0)
            self.assertEqual(float(frame.loc[0, "frac_ge_90"]), 2.0 / 31.0)

            # ...and an arbitrary cutoff the producer never precomputed. Rows are
            # sorted by accession.
            self.assertEqual(
                list(frame["accession"]),
                ["P00533", "P11362", "P30518", "P68363", "Q8NCT1"],
            )
            self.assertEqual(
                list(quality.fraction_at(frame, 70.005)),
                [2.0 / 31.0, 29.0 / 30.0, 0.5, 0.5, 1.0 / 3.0],
            )

    def test_passes_applies_the_fraction_threshold(self):
        frame = schemas.coerce(
            pd.DataFrame(
                [
                    quality_row(accession="A", residue_plddt="90.00;90.00;90.00;30.00",
                                n_residues=4, frac_ge_70=0.75),
                    quality_row(accession="B", residue_plddt="30.00;30.00;30.00;90.00",
                                n_residues=4, frac_ge_70=0.25),
                ]
            ),
            "cavity_quality",
        )
        keep = quality.passes(frame, plddt_min=70.0, min_fraction=0.5)
        self.assertEqual(list(keep), [True, False])
        self.assertEqual(list(keep.index), list(frame.index))
        self.assertEqual(list(quality.passes(frame, 70.0, 0.75)), [True, False])
        self.assertEqual(list(quality.passes(frame, 70.0, 0.76)), [False, False])
        self.assertEqual(list(quality.passes(frame, 20.0, 1.0)), [True, True])

    def test_passes_refuses_an_unset_or_nonsensical_fraction(self):
        frame = schemas.coerce(pd.DataFrame([quality_row()]), "cavity_quality")
        with self.assertRaises(quality.QualityError):
            quality.passes(frame, 70.0, None)
        with self.assertRaises(quality.QualityError):
            quality.passes(frame, 70.0, 80)  # a percentage, not a fraction

    def test_fraction_at_needs_the_residue_column(self):
        frame = schemas.coerce(pd.DataFrame([quality_row()]), "cavity_quality")
        with self.assertRaises(quality.QualityError):
            quality.fraction_at(frame.drop(columns=["residue_plddt"]), 70.0)

    def test_an_empty_frame_yields_empty_results(self):
        frame = schemas.empty_frame("cavity_quality")
        self.assertEqual(len(quality.fraction_at(frame, 70.0)), 0)
        self.assertEqual(len(quality.passes(frame, 70.0, 0.8)), 0)

    def test_load_cavity_quality_validates_on_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cavity_quality.csv")
            pd.DataFrame([quality_row(n_residues=9)]).to_csv(path, index=False)
            with self.assertRaises(schemas.SchemaViolation):
                quality.load_cavity_quality(path)

    def test_load_cavity_quality_refuses_a_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(schemas.SchemaViolation):
                quality.load_cavity_quality(os.path.join(tmp, "absent.csv"))

    def test_parse_residue_plddt_refuses_a_non_numeric_value(self):
        self.assertEqual(quality.parse_residue_plddt("70.00;80.50"), [70.0, 80.5])
        self.assertEqual(quality.parse_residue_plddt(None), [])
        with self.assertRaises(quality.QualityError):
            quality.parse_residue_plddt("70.00;;;80.00;oops")


if __name__ == "__main__":
    unittest.main()
