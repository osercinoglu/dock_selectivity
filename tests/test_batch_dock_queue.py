"""Tests for batch_dock.py's queue-driven path and complex_*.pdb cleanup.

Three defects, all FIX_PLAN §3.3:

* ``:2034`` resolved a gene SYMBOL to an accession with
  ``dict(zip(Mapped_Gene_Name, Entry))`` -- last-duplicate-wins over a column
  that is 57.73% wrong. It is the sole reason 633 of 979 docked proteins were not
  the protein their job name claimed. The queue path never resolves anything.
* ``:2211`` encoded identity in a folder-name string whose gene and accession came
  from opposite sides of the bug and could disagree. The name is now built from
  the key, and ``job_identity.json`` is the authority.
* ``complex_*.pdb`` was retained forever: 89% of 1.4 TB, derived from
  ``out.sdf`` + the receptor, 99.75% byte-identical between poses.

Hermetic: no docking engine runs, nothing touches the external drive.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import pandas as pd

import batch_dock
from dock_selectivity import jobid


def queue_frame(rows):
    return pd.DataFrame(
        [
            {
                "job_id": jobid.format_job_dir(
                    drugbank_id=d, accession=a, gene=g, fragment_id=f, cavity_index=c
                ),
                "drugbank_id": d,
                "accession": a,
                "gene": g,
                "fragment_id": f,
                "cavity_index": c,
                "sample_type": st,
                "selected_by": "entrez",
            }
            for (d, a, g, f, c, st) in rows
        ]
    )


def make_pdb_info(tmp, accession, cavities):
    """pdb_info_dict entry: [(receptor_pdb, pocket_pdb, receptor_pdbqt), ...]."""
    entries = []
    for cav in cavities:
        pocket = os.path.join(
            tmp, "AF-%s-F1-model_v1_cavity_%d.pdb" % (accession, cav)
        )
        open(pocket, "w").close()
        receptor = os.path.join(tmp, "%s.pdb" % accession)
        open(receptor, "w").close()
        entries.append((receptor, pocket, None))
    return {accession: entries}


class TestBuildJobsFromQueue(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.ligands = os.path.join(self.dir, "ligands")
        os.makedirs(self.ligands)
        open(os.path.join(self.ligands, "DB00035.sdf"), "w").close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_job_name_is_built_from_the_key_not_a_lookup(self):
        info = make_pdb_info(self.dir, "P30518", [1])
        jobs, rejects = batch_dock.build_jobs_from_queue(
            queue_frame([("DB00035", "P30518", "AVPR2", "F1", 1, "positive")]),
            info, self.ligands, os.path.join(self.dir, "out"),
        )
        self.assertEqual(rejects, [])
        self.assertEqual(len(jobs), 1)
        self.assertTrue(jobs[0]["current_outfolder"].endswith(
            "DB00035__P30518__AVPR2__F1__cav1"))
        self.assertEqual(jobs[0]["uniprot_id"], "P30518")

    def test_receptor_comes_from_the_accession_in_the_row(self):
        """The AVPR2/Q8NCT1 case: queueing Q8NCT1 docks Q8NCT1, labelled ARRDC4."""
        info = make_pdb_info(self.dir, "Q8NCT1", [2])
        jobs, _ = batch_dock.build_jobs_from_queue(
            queue_frame([("DB00035", "Q8NCT1", "ARRDC4", "F1", 2, "negative")]),
            info, self.ligands, os.path.join(self.dir, "out"),
        )
        self.assertIn("Q8NCT1", jobs[0]["receptor_pdb"])
        self.assertIn("Q8NCT1", jobs[0]["pocket_pdb"])
        self.assertTrue(jobs[0]["current_outfolder"].endswith(
            "DB00035__Q8NCT1__ARRDC4__F1__cav2"))

    def test_the_requested_cavity_is_the_one_selected(self):
        info = make_pdb_info(self.dir, "P30518", [1, 2, 5])
        jobs, _ = batch_dock.build_jobs_from_queue(
            queue_frame([("DB00035", "P30518", "AVPR2", "F1", 5, "positive")]),
            info, self.ligands, os.path.join(self.dir, "out"),
        )
        self.assertIn("_cavity_5.pdb", jobs[0]["pocket_pdb"])

    def test_unavailable_cavity_is_rejected_with_a_reason(self):
        info = make_pdb_info(self.dir, "P30518", [1])
        jobs, rejects = batch_dock.build_jobs_from_queue(
            queue_frame([("DB00035", "P30518", "AVPR2", "F1", 9, "positive")]),
            info, self.ligands, os.path.join(self.dir, "out"),
        )
        self.assertEqual(jobs, [])
        self.assertEqual(rejects[0]["reason"], "CAVITY_9_NOT_AVAILABLE")

    def test_missing_ligand_is_rejected_not_dropped(self):
        info = make_pdb_info(self.dir, "P30518", [1])
        _, rejects = batch_dock.build_jobs_from_queue(
            queue_frame([("DB99999", "P30518", "AVPR2", "F1", 1, "positive")]),
            info, self.ligands, os.path.join(self.dir, "out"),
        )
        self.assertEqual(rejects[0]["reason"], "NO_LIGAND_SDF")

    def test_unknown_accession_is_rejected(self):
        _, rejects = batch_dock.build_jobs_from_queue(
            queue_frame([("DB00035", "Q99999", "NOPE", "F1", 1, "positive")]),
            {}, self.ligands, os.path.join(self.dir, "out"),
        )
        self.assertEqual(rejects[0]["reason"], "NO_RECEPTOR_FOR_ACCESSION")

    def test_every_row_is_either_a_job_or_a_reject(self):
        info = make_pdb_info(self.dir, "P30518", [1])
        rows = [
            ("DB00035", "P30518", "AVPR2", "F1", 1, "positive"),
            ("DB00035", "P30518", "AVPR2", "F1", 7, "positive"),
            ("DB99999", "P30518", "AVPR2", "F1", 1, "positive"),
        ]
        jobs, rejects = batch_dock.build_jobs_from_queue(
            queue_frame(rows), info, self.ligands, os.path.join(self.dir, "out")
        )
        self.assertEqual(len(jobs) + len(rejects), len(rows))


class TestJobIdentityRecords(unittest.TestCase):
    def test_identity_json_is_written_and_matches_the_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            ligands = os.path.join(tmp, "lig")
            os.makedirs(ligands)
            open(os.path.join(ligands, "DB00035.sdf"), "w").close()
            info = make_pdb_info(tmp, "P30518", [1])
            jobs, _ = batch_dock.build_jobs_from_queue(
                queue_frame([("DB00035", "P30518", "AVPR2", "F1", 1, "positive")]),
                info, ligands, os.path.join(tmp, "out"),
            )
            self.assertEqual(batch_dock.write_queue_job_identities(jobs), 1)

            path = os.path.join(jobs[0]["current_outfolder"], "job_identity.json")
            self.assertTrue(os.path.exists(path))
            with open(path) as handle:
                record = json.load(handle)
            self.assertEqual(record["accession"], "P30518")
            self.assertEqual(record["gene"], "AVPR2")
            self.assertEqual(record["provenance"], "newly_docked")
            # identity is recoverable without parsing the directory name
            identity = jobid.read_job_identity(jobs[0]["current_outfolder"])
            self.assertEqual(identity.accession, "P30518")


class TestComplexPdbCleanup(unittest.TestCase):
    def _job(self, tmp, with_final=True, n=3):
        job = os.path.join(tmp, "job")
        for tool in ("smina", "gnina", "gold", "ledock"):
            os.makedirs(os.path.join(job, tool))
            for i in range(1, n + 1):
                with open(os.path.join(job, tool, "complex_%d.pdb" % i), "w") as fh:
                    fh.write("ATOM  " * 500)
            with open(os.path.join(job, tool, "results.csv"), "w") as fh:
                fh.write("Tool,Score\n")
            with open(os.path.join(job, tool, "out.sdf"), "w") as fh:
                fh.write("pose\n")
        if with_final:
            with open(os.path.join(job, "final_results.csv"), "w") as fh:
                fh.write("Tool1,Tool2,Score1,Score2,RMSD\n" + "x,y,1,2,3\n" * 20)
        return job

    def test_delete_removes_complexes_and_keeps_the_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(tmp)
            touched, reclaimed = batch_dock.cleanup_complex_pdbs(job, "delete")
            self.assertEqual(touched, 12)
            self.assertGreater(reclaimed, 0)
            for tool in ("smina", "gnina", "gold", "ledock"):
                names = os.listdir(os.path.join(job, tool))
                self.assertNotIn("complex_1.pdb", names)
                self.assertIn("results.csv", names)   # the payload survives
                self.assertIn("out.sdf", names)       # CNN_VS lives here
            self.assertTrue(os.path.exists(os.path.join(job, "final_results.csv")))

    def test_compress_replaces_with_gz(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(tmp)
            touched, _ = batch_dock.cleanup_complex_pdbs(job, "compress")
            self.assertEqual(touched, 12)
            names = os.listdir(os.path.join(job, "smina"))
            self.assertIn("complex_1.pdb.gz", names)
            self.assertNotIn("complex_1.pdb", names)

    def test_keep_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(tmp)
            self.assertEqual(batch_dock.cleanup_complex_pdbs(job, "keep"), (0, 0))
            self.assertIn("complex_1.pdb", os.listdir(os.path.join(job, "smina")))

    def test_refuses_to_clean_a_job_with_no_final_results(self):
        """A crashed job keeps everything needed to diagnose it."""
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(tmp, with_final=False)
            self.assertEqual(batch_dock.cleanup_complex_pdbs(job, "delete"), (0, 0))
            self.assertIn("complex_1.pdb", os.listdir(os.path.join(job, "smina")))

    def test_bad_mode_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                batch_dock.cleanup_complex_pdbs(tmp, "shred")

    def test_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(tmp)
            batch_dock.cleanup_complex_pdbs(job, "delete")
            self.assertEqual(batch_dock.cleanup_complex_pdbs(job, "delete"), (0, 0))


if __name__ == "__main__":
    unittest.main()


class TestResumeOverPrePopulatedOutput(unittest.TestCase):
    """A pre-populated output directory needs no special pipeline handling.

    Every stage skips work already complete on disk. That is what makes a
    populated output directory just work: the queue is docked to completion and
    whatever is already there is left alone. This test pins that behaviour for
    directories carrying only the retained evidence -- no complex_*.pdb -- because
    that is what the curation step produces and a completion check that looked for
    pose complexes would silently re-dock everything.
    """

    def _evidence_only_job(self, tmp, name="DB00035__P30518__AVPR2__F1__cav1"):
        job = os.path.join(tmp, name)
        for tool in ("smina", "gnina", "gold", "ledock"):
            os.makedirs(os.path.join(job, tool))
            with open(os.path.join(job, tool, "results.csv"), "w") as fh:
                fh.write("Tool,Pose,Score\nSMINA,1,-7.2\n")
            with open(os.path.join(job, tool, "out.sdf"), "w") as fh:
                fh.write("pose\n")
        with open(os.path.join(job, "final_results.csv"), "w") as fh:
            fh.write("Tool1,Tool2,PoseNumber1,PoseNumber2,Score1,Score2,File1,File2,RMSD\n")
            fh.write("SMINA,GNINA,1,1,-7.2,-6.9,a,b,1.4\n" * 30)
        return job

    def test_evidence_only_job_counts_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = self._evidence_only_job(tmp)
            self.assertFalse(
                any(f.startswith("complex_")
                    for f in os.listdir(os.path.join(job, "smina"))),
                "fixture must have no complex_*.pdb or it proves nothing")
            for tool in ("smina", "gnina", "gold", "ledock"):
                self.assertTrue(batch_dock.check_tool_completion(job, tool), tool)
            self.assertTrue(batch_dock.check_final_results_completion(job))
            status = batch_dock.get_job_completion_status(job)
            self.assertTrue(all(status.values()), status)

    def test_an_absent_job_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "DB99999__P00000__NOPE__F1__cav1")
            self.assertFalse(batch_dock.check_final_results_completion(missing))
            self.assertFalse(batch_dock.check_tool_completion(missing, "smina"))

    def test_a_half_finished_job_is_not_complete(self):
        """smina done, the rest absent: the queue must still dock it."""
        with tempfile.TemporaryDirectory() as tmp:
            job = os.path.join(tmp, "DB00035__P30518__AVPR2__F1__cav2")
            os.makedirs(os.path.join(job, "smina"))
            with open(os.path.join(job, "smina", "results.csv"), "w") as fh:
                fh.write("Tool,Pose,Score\nSMINA,1,-7.2\n")
            self.assertTrue(batch_dock.check_tool_completion(job, "smina"))
            self.assertFalse(batch_dock.check_tool_completion(job, "gold"))
            self.assertFalse(batch_dock.check_final_results_completion(job))
