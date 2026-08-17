"""Tests for extract_cavities.py's manifest/required-set split.

The defect under test: the script iterated the *supply* --
``os.listdir(cavity_tarball_folder)`` over 38,709 archives -- with no way to pass
a required set, because the mapping was believed to require unpacking. It did
not: the tarball NAME carries (accession, fragment, cavity), one archive per
cavity. Building the manifest from names and unpacking only what is required is
worth 76.6 GB on the 2025-06-29 inputs (2,829 of 38,709 needed).

The behaviour that must hold forever after: extraction without an explicit
required set is an ERROR, never a fallback to "everything".
"""

from __future__ import annotations

import io
import os
import tarfile
import tempfile
import unittest

import pandas as pd

import extract_cavities as ec

from tests.support import run_in_temp_cwd

#: A step with no --run-dir logs into ./<step-name>/; tests must not
#: create that inside the checkout.
setUpModule, tearDownModule = run_in_temp_cwd()



def make_tarball(folder: str, accession: str, fragment: int, cavity: int,
                 member_cavity: int = None) -> str:
    """Write a realistic cavity archive. ``member_cavity`` forces a name/content mismatch."""
    stem = "AF-%s-F%d-model_v1" % (accession, fragment)
    name = "%s_cavity_result_%d.tar.gz" % (stem, cavity)
    path = os.path.join(folder, name)
    inner = "%s_cavity_%d.pdb" % (stem, cavity if member_cavity is None else member_cavity)
    payload = b"ATOM      1  N   ALA A   1       0.000   0.000   0.000\n"
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(inner)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return name


class TestNameParsing(unittest.TestCase):
    def test_name_carries_accession_fragment_and_cavity(self):
        self.assertEqual(
            ec.parse_tarball_name("AF-B4DS77-F1-model_v1_cavity_result_2.tar.gz"),
            ("B4DS77", "F1", 2),
        )

    def test_multi_digit_fragment_and_cavity(self):
        self.assertEqual(
            ec.parse_tarball_name("AF-P08F94-F13-model_v1_cavity_result_10.tar.gz"),
            ("P08F94", "F13", 10),
        )

    def test_unparseable_name_returns_none_rather_than_guessing(self):
        self.assertIsNone(ec.parse_tarball_name("not-a-cavity-archive.tar.gz"))


class TestManifest(unittest.TestCase):
    def test_manifest_unpacks_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_tarball(tmp, "B4DS77", 1, 2)
            make_tarball(tmp, "P30518", 1, 1)
            before = sorted(os.listdir(tmp))
            manifest, stats = ec.build_cavity_manifest(tmp)
            self.assertEqual(sorted(os.listdir(tmp)), before)  # nothing created
            self.assertEqual(stats["rows"], 2)
            self.assertEqual(stats["accessions"], 2)

    def test_manifest_columns_and_sort_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_tarball(tmp, "P30518", 1, 3)
            make_tarball(tmp, "B4DS77", 1, 2)
            manifest, _ = ec.build_cavity_manifest(tmp)
            self.assertEqual(
                list(manifest.columns),
                ["UniProt_ID", "Fragment_ID", "Cavity_Index", "Tarball", "Pocket_PDB"],
            )
            self.assertEqual(list(manifest["UniProt_ID"]), ["B4DS77", "P30518"])

    def test_manifest_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            for acc, cav in (("P30518", 2), ("B4DS77", 1), ("P68363", 5)):
                make_tarball(tmp, acc, 1, cav)
            a, _ = ec.build_cavity_manifest(tmp)
            b, _ = ec.build_cavity_manifest(tmp)
            pd.testing.assert_frame_equal(a, b)

    def test_unparseable_tarballs_are_counted_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_tarball(tmp, "B4DS77", 1, 1)
            open(os.path.join(tmp, "junk.tar.gz"), "wb").close()
            _, stats = ec.build_cavity_manifest(tmp)
            self.assertEqual(stats["rows"], 1)
            self.assertEqual(stats["unparseable"], ["junk.tar.gz"])

    def test_verify_members_catches_a_name_content_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_tarball(tmp, "B4DS77", 1, 2, member_cavity=9)
            loose, _ = ec.build_cavity_manifest(tmp, verify_members=False)
            strict, stats = ec.build_cavity_manifest(tmp, verify_members=True)
            self.assertEqual(len(loose), 1)   # trusting the name
            self.assertEqual(len(strict), 0)  # verified away
            self.assertEqual(len(stats["mismatched"]), 1)

    def test_missing_folder_raises(self):
        with self.assertRaises(NotADirectoryError):
            ec.build_cavity_manifest("/nonexistent/cavity/folder")


class TestRequiredSet(unittest.TestCase):
    def test_extraction_requires_an_explicit_set(self):
        """The whole point: no required set is an error, not 'extract everything'."""
        with self.assertRaises(ValueError) as ctx:
            ec.extract_required_cavities("/tmp", pd.DataFrame(), None)
        self.assertIn("never a fallback", str(ctx.exception))

    def test_only_required_cavities_are_unpacked(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "tarballs")
            dst = os.path.join(tmp, "out")
            os.makedirs(src)
            make_tarball(src, "B4DS77", 1, 1)
            make_tarball(src, "B4DS77", 1, 2)
            make_tarball(src, "P30518", 1, 1)
            manifest, _ = ec.build_cavity_manifest(src)

            out, stats = ec.extract_required_cavities(
                src, manifest, {("P30518", "F1", 1)}, extract_base_dir=dst
            )
            self.assertEqual(stats["required"], 1)
            self.assertEqual(stats["extracted"], 1)
            self.assertEqual(len(out), 1)
            # exactly one archive's worth on disk, not three
            self.assertEqual(len(os.listdir(dst)), 1)
            self.assertTrue(os.path.exists(out.iloc[0]["Pocket_PDB"]))

    def test_reruns_reuse_and_do_not_re_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "t")
            dst = os.path.join(tmp, "o")
            os.makedirs(src)
            make_tarball(src, "B4DS77", 1, 1)
            manifest, _ = ec.build_cavity_manifest(src)
            keys = {("B4DS77", "F1", 1)}
            _, first = ec.extract_required_cavities(src, manifest, keys, extract_base_dir=dst)
            _, second = ec.extract_required_cavities(src, manifest, keys, extract_base_dir=dst)
            self.assertEqual((first["extracted"], first["reused"]), (1, 0))
            self.assertEqual((second["extracted"], second["reused"]), (0, 1))

    def test_required_cavity_absent_from_manifest_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "t")
            os.makedirs(src)
            make_tarball(src, "B4DS77", 1, 1)
            manifest, _ = ec.build_cavity_manifest(src)
            _, stats = ec.extract_required_cavities(
                src, manifest, {("Q99999", "F1", 1)}, extract_base_dir=os.path.join(tmp, "o")
            )
            self.assertEqual(stats["missing_from_manifest"], [("Q99999", "F1", 1)])
            self.assertNotEqual(stats["present"], stats["required"])  # invariant trips

    def test_fragment_id_forms_are_interchangeable(self):
        """'1' and 'F1' must select the same cavity."""
        frame = pd.DataFrame(
            [
                {"UniProt_ID": "B4DS77", "Fragment_ID": 1, "Cavity_Index": 1},
                {"accession": None, "Fragment_ID": None, "Cavity_Index": None},
            ][:1]
        )
        self.assertEqual(ec.required_keys_from_frame(frame), {("B4DS77", "F1", 1)})
        frame2 = pd.DataFrame(
            [{"accession": "B4DS77", "fragment_id": "F1", "cavity_index": 1}]
        )
        self.assertEqual(ec.required_keys_from_frame(frame2), {("B4DS77", "F1", 1)})

    def test_required_frame_without_an_accession_column_raises(self):
        with self.assertRaises(ValueError):
            ec.required_keys_from_frame(pd.DataFrame([{"nope": 1}]))


class TestCli(unittest.TestCase):
    def test_extraction_without_required_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = ec.main(["--tarballs", tmp, "--manifest", "x.csv", "--out-dir", tmp])
            self.assertEqual(rc, 2)

    def test_manifest_only_writes_and_unpacks_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "t")
            os.makedirs(src)
            make_tarball(src, "B4DS77", 1, 1)
            out = os.path.join(tmp, "manifest.csv")
            rc = ec.main(["--manifest-only", "--tarballs", src, "--out", out])
            self.assertEqual(rc, 0)
            self.assertEqual(len(pd.read_csv(out)), 1)
            self.assertEqual(sorted(os.listdir(src)),
                             ["AF-B4DS77-F1-model_v1_cavity_result_1.tar.gz"])

    def test_full_two_step_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "t")
            os.makedirs(src)
            make_tarball(src, "B4DS77", 1, 1)
            make_tarball(src, "P30518", 1, 1)
            manifest = os.path.join(tmp, "manifest.csv")
            self.assertEqual(ec.main(["--manifest-only", "--tarballs", src, "--out", manifest]), 0)

            required = os.path.join(tmp, "required.csv")
            pd.DataFrame([{"accession": "P30518", "fragment_id": "F1", "cavity_index": 1}]).to_csv(
                required, index=False
            )
            out = os.path.join(tmp, "cavity_mapping.csv")
            dst = os.path.join(tmp, "cavities")
            rc = ec.main([
                "--tarballs", src, "--manifest", manifest, "--required", required,
                "--out", out, "--out-dir", dst,
            ])
            self.assertEqual(rc, 0)
            self.assertEqual(len(pd.read_csv(out)), 1)
            self.assertEqual(len(os.listdir(dst)), 1)

    def test_unsatisfiable_required_set_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "t")
            os.makedirs(src)
            make_tarball(src, "B4DS77", 1, 1)
            manifest = os.path.join(tmp, "m.csv")
            ec.main(["--manifest-only", "--tarballs", src, "--out", manifest])
            required = os.path.join(tmp, "r.csv")
            pd.DataFrame([{"accession": "Q99999", "fragment_id": "F1", "cavity_index": 1}]).to_csv(
                required, index=False
            )
            rc = ec.main([
                "--tarballs", src, "--manifest", manifest, "--required", required,
                "--out", os.path.join(tmp, "o.csv"), "--out-dir", os.path.join(tmp, "d"),
            ])
            self.assertEqual(rc, 3)


if __name__ == "__main__":
    unittest.main()
