"""The decoupled gnina stage, end to end, with a fake GPU host.

WHAT THIS COVERS THAT THE PER-MODULE SUITES DO NOT
--------------------------------------------------

The four modules of the decoupled stage were built concurrently against a written
contract, and each is tested against *its own* reading of that contract with the
neighbouring module mocked out. That is exactly the arrangement in which two
correct-looking modules disagree: a ticket field one writes and the other never
reads, an exit code one returns and the other treats as failure, a remote
directory recorded in one format and parsed in another.

So nothing is mocked here. The chain runs for real --

    capability check -> enqueue -> submit -> (fake) remote gnina finishes
    -> collect -> post-process -> RMSD barrier -> RMSD proceeds

-- with only the two things that genuinely cannot run on this machine replaced:

``ssh``/``scp``
    Real POSIX shell scripts on ``PATH`` that map the "remote host" onto a local
    directory. Every remote command in this design is shell -- ``mkdir`` for the
    GPU mutex, ``kill -0`` for liveness, ``setsid nohup`` for detachment,
    ``echo $? > exit_code`` for the completion marker -- so running it through a
    real shell tests the quoting, the ordering and the detachment semantics
    rather than a Python imitation of them. The jobs really do detach; the
    mutex really is an atomic ``mkdir``; the exit-code marker really is written
    by the job itself.
``gnina`` and ``consensus_docker.py``
    Stand-ins. The gnina one writes a pose file and a log in gnina's own format.
    The consensus_docker one reproduces the contract that matters to this
    design, verified against the real file at
    /home/onur/repos/consensus_docking/consensus_docker.py:

      * ``run_gnina_single`` only *warns* on a non-zero gnina exit (line 1372) --
        which is why the submit phase's exit 3 is harmless;
      * ``split_mol`` then raises on the absent ``out.sdf``, ``run_gnina``
        propagates, and ``consensus_dock`` catches it, logs "gnina docking
        failed" (line 1731) and still exits 0 having written nothing;
      * on the collect phase it splits ``out.sdf`` into ``out_N.sdf``, builds
        ``complex_N.pdb`` per pose, and writes ``results.csv`` with the columns
        ``Pose,GNINA_Affinity,GNINA_CNN_pose,GNINA_CNN_affinity`` (line 594).

FIDELITY LIMIT, stated plainly: the stand-in consensus_docker is a stand-in. It
reproduces that control flow and those file names, not openbabel's SDF parsing or
rdkit's ``CombineMols``. This suite therefore proves that the *plumbing* delivers
the right bytes to the right post-processing at the right time; it cannot prove
that rdkit builds a good complex. That is what the live pilot in SETUP.md is for.

Hermetic: no SSH, no GPU, no network, no writes outside a TemporaryDirectory, and
nothing under /media/onur/Elements.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import batch_dock                                              # noqa: E402
import gnina_collect                                           # noqa: E402
import gnina_remote                                            # noqa: E402
from dock_selectivity import gnina_queue, remote_caps          # noqa: E402


# The measured capacity of the real GPU box, from the brief. Canned on purpose:
# the host has one GPU and re-probing it from a test would disturb whoever is
# using it -- and a test that needs a particular machine to be up is not a test.
MEASURED_PROBE = "\n".join([
    "cores=24",
    "mem_total_kb=32505856",
    "mem_avail_kb=28311552",
    "gpu=NVIDIA GeForce RTX 3060 Ti, 8192, 512, 8.6",
    "docker=1",
    "image=1",
    "native=0",
    "disk_free_kb=419430400",
    "uptime=812345.67",
])


# ---------------------------------------------------------------------------
# fixtures: a fake GPU host you can actually ssh to
# ---------------------------------------------------------------------------

def _write(path, text, executable=False):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)
    if executable:
        os.chmod(path, 0o755)
    return path


def _read(path):
    with open(path) as handle:
        return handle.read()


def _read_json(path):
    with open(path) as handle:
        return json.load(handle)


def build_fake_transport(root):
    """An ``ssh``/``scp`` pair that runs commands against a local directory.

    The point of doing it this way rather than monkeypatching ``run_resilient``:
    every interesting remote operation in this design is a shell snippet, and the
    bugs live in the shell, not in the Python that sends it. A stubbed transport
    would have happily accepted a mutex snippet that never releases.
    """
    host_root = os.path.join(root, "fakehost")
    bin_dir = os.path.join(root, "bin")
    os.makedirs(host_root, exist_ok=True)

    _write(os.path.join(bin_dir, "ssh"), textwrap.dedent("""\
        #!/bin/sh
        # Strip the ssh options, then the user@host, then run the rest as a
        # command inside the fake host's filesystem.
        while [ $# -gt 0 ]; do
          case "$1" in
            -o|-p|-i|-P|-F|-l) shift 2 ;;
            -q|-T|-t|-n) shift ;;
            --) shift; break ;;
            -*) shift ;;
            *) break ;;
          esac
        done
        [ $# -gt 0 ] || exit 255
        shift                      # the user@host argument
        cd %s || exit 255
        exec /bin/sh -c "$*"
        """) % _sh_quote(host_root), executable=True)

    _write(os.path.join(bin_dir, "scp"), textwrap.dedent("""\
        #!/bin/sh
        # Strip options, drop any "user@host:" prefix, copy locally.
        args=""
        while [ $# -gt 0 ]; do
          case "$1" in
            -o|-p|-i|-P|-F|-l) shift 2 ;;
            -q|-r|-C|-B) shift ;;
            --) shift; break ;;
            -*) shift ;;
            *) break ;;
          esac
        done
        for a in "$@"; do
          case "$a" in
            */*) ;;                                  # a plain path
            *:*) a="${a#*:}" ;;                      # strip user@host:
          esac
          case "$a" in
            *:*) a="${a#*:}" ;;
          esac
          args="$args '$a'"
        done
        eval cp $args
        """), executable=True)

    # The "gnina binary" on the GPU box. Behaviour is driven by a control file so
    # a test can make one job fail without touching the environment of a process
    # that has already been detached.
    _write(os.path.join(bin_dir, "gnina_fake"), textwrap.dedent('''\
        #!/usr/bin/env python3
        """Stand-in for the gnina binary: writes poses and a log in gnina's format."""
        import os
        import sys

        args = sys.argv[1:]

        def value(flag, default=None):
            return args[args.index(flag) + 1] if flag in args else default

        out = value("-o") or value("--out")
        log = value("--log")
        control = os.path.join(%s, "behaviour")
        behaviour = "ok"
        if os.path.exists(control):
            with open(control) as handle:
                behaviour = handle.read().strip() or "ok"

        if behaviour == "fail":
            sys.stderr.write("CUDA error: out of memory\\n")
            sys.exit(1)

        poses = 1 if behaviour == "one_pose" else 3
        if behaviour == "empty":
            open(out, "w").close()
            sys.exit(0)

        records = []
        for i in range(1, poses + 1):
            records.append(
                "pose_%%d\\n  fake_gnina\\n\\n"
                "  1  0  0  0  0  0  0  0  0  0999 V2000\\n"
                "    %%.4f    2.0000    3.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\\n"
                "M  END\\n"
                "> <minimizedAffinity>\\n%%.2f\\n\\n$$$$\\n" %% (i, float(i), -7.5 - i))
        with open(out, "w") as handle:
            handle.write("".join(records))

        if log:
            rows = "".join("%%5d      %%7.2f      %%7.3f      %%7.4f    %%7.3f\\n"
                           %% (i, -7.5 - i, 0.0, 0.9 - 0.1 * i, 6.0 + i)
                           for i in range(1, poses + 1))
            with open(log, "w") as handle:
                handle.write(
                    "gnina (fake) for testing\\n"
                    "Using random seed: 42\\n\\n"
                    "mode |  affinity  |  intramol  |    CNN     |   CNN\\n"
                    "     | (kcal/mol) | (kcal/mol) | pose score | affinity\\n"
                    "-----+------------+------------+------------+----------\\n"
                    + rows)
        ''') % repr(host_root), executable=True)

    return {"host_root": host_root, "bin_dir": bin_dir,
            "gnina": os.path.join(bin_dir, "gnina_fake"),
            "behaviour": os.path.join(host_root, "behaviour")}


def _sh_quote(text):
    return "'" + text.replace("'", "'\\''") + "'"


def build_fake_consensus_docker(path):
    """A stand-in for consensus_docker.py that keeps the contract of section 1.

    It is deliberately faithful about the parts this design depends on and
    deliberately dumb about chemistry. See the module docstring.
    """
    return _write(path, textwrap.dedent('''\
        #!/usr/bin/env python3
        """Stand-in for consensus_docker.py (gnina path only)."""
        import argparse
        import glob
        import os
        import subprocess
        import sys

        parser = argparse.ArgumentParser()
        parser.add_argument("--outfolder", required=True)
        parser.add_argument("--gnina_path")
        parser.add_argument("--ligand_sdf")
        parser.add_argument("--receptor_pdb")
        parser.add_argument("--pocket_pdb")
        parser.add_argument("--receptor_pdbqt")
        parser.add_argument("--num_threads")
        parser.add_argument("--cutoff_value")
        parser.add_argument("--exhaustiveness")
        parser.add_argument("--num_modes")
        parser.add_argument("--use_gnina", action="store_true")
        parser.add_argument("--overwrite", action="store_true")
        parser.add_argument("--skip_rmsd", action="store_true")
        parser.add_argument("--skip_pdb_to_pdbqt", action="store_true")
        args = parser.parse_args()

        gnina_dir = os.path.join(args.outfolder, "gnina")
        os.makedirs(gnina_dir, exist_ok=True)
        out_sdf = os.path.join(gnina_dir, "out.sdf")
        out_log = os.path.join(gnina_dir, "out.log")

        # A record of every invocation, so a test can prove that both phases
        # built the SAME search box from the same pocket file.
        def get_pocket_coords(path):
            xs, ys, zs = [], [], []
            with open(path) as handle:
                for line in handle:
                    if line.startswith(("ATOM", "HETATM")):
                        xs.append(float(line[30:38]))
                        ys.append(float(line[38:46]))
                        zs.append(float(line[46:54]))
            centre = [round(sum(v) / len(v), 3) for v in (xs, ys, zs)]
            size = [round(max(v) - min(v) + 8.0, 3) for v in (xs, ys, zs)]
            return centre, size

        centre, size = get_pocket_coords(args.pocket_pdb)
        cmd = ("{gnina} -r {rec} -l {lig} "
               "--center_x {cx} --center_y {cy} --center_z {cz} "
               "--size_x {sx} --size_y {sy} --size_z {sz} "
               "-o {out} --log {log} --cpu {cpu}").format(
            gnina=args.gnina_path, rec=args.receptor_pdb, lig=args.ligand_sdf,
            cx=centre[0], cy=centre[1], cz=centre[2],
            sx=size[0], sy=size[1], sz=size[2],
            out=out_sdf, log=out_log, cpu=args.num_threads)
        if args.num_modes:
            cmd += " --num_modes %s" % args.num_modes
        if args.exhaustiveness:
            cmd += " --exhaustiveness %s" % args.exhaustiveness

        with open(os.path.join(args.outfolder, "invocations.log"), "a") as handle:
            handle.write("%s\\t%s\\n" % (os.environ.get("GNINA_REMOTE_MODE", "?"), cmd))

        def run_gnina_single():
            code = subprocess.call(cmd, shell=True)
            if code != 0:
                # The real one only warns here; the failure surfaces in split_mol.
                sys.stderr.write("Gnina command returned non-zero exit code: %d\\n" % code)

        def split_mol():
            with open(out_sdf) as handle:            # raises if it is not there
                blocks = [b for b in handle.read().split("$$$$\\n") if b.strip()]
            if not blocks:
                raise RuntimeError("no molecules in %s" % out_sdf)
            for i, block in enumerate(blocks, 1):
                with open(os.path.join(gnina_dir, "out_%d.sdf" % i), "w") as handle:
                    handle.write(block + "$$$$\\n")
            return len(blocks)

        def make_complex(n_poses):
            with open(args.receptor_pdb) as handle:
                receptor = handle.read()
            for i in range(1, n_poses + 1):
                with open(os.path.join(gnina_dir, "complex_%d.pdb" % i), "w") as handle:
                    handle.write(receptor)
                    handle.write("HETATM    1  C   LIG A 900       "
                                 "1.000   2.000   3.000  1.00  0.00           C\\n")
                    handle.write("END\\n")

        def parse_gnina():
            if not os.path.exists(out_log):
                raise RuntimeError("Gnina log file not found: %s" % out_log)
            with open(out_log) as handle:
                lines = handle.readlines()
            start = None
            for i, line in enumerate(lines):
                if "mode |" in line and "affinity" in line:
                    start = i + 3
                    break
            if start is None:
                raise RuntimeError("Could not find results in %s" % out_log)
            rows = []
            for line in lines[start:]:
                parts = line.split()
                if len(parts) < 5:
                    break
                try:
                    rows.append((int(parts[0]), float(parts[1]),
                                 float(parts[3]), float(parts[4])))
                except ValueError:
                    continue
            if not rows:
                raise RuntimeError("No valid results found in %s" % out_log)
            with open(os.path.join(gnina_dir, "results.csv"), "w") as handle:
                handle.write("Pose,GNINA_Affinity,GNINA_CNN_pose,GNINA_CNN_affinity\\n")
                for mode, affinity, cnn_pose, cnn_affinity in rows:
                    handle.write("%d,%s,%s,%s\\n" % (mode, affinity, cnn_pose,
                                                     cnn_affinity))

        try:
            run_gnina_single()
            n = split_mol()
            make_complex(n)
            parse_gnina()
        except Exception as exc:
            # Exactly what consensus_dock does: log and carry on, exit 0.
            sys.stderr.write("gnina docking failed: %s\\n" % exc)

        sys.exit(0)
        '''), executable=True)


def build_job(run_dir, job_id):
    """A job directory shaped like batch_dock's, with the three inputs."""
    job_dir = os.path.join(run_dir, "docking_results", job_id)
    inputs = os.path.join(job_dir, "input")
    os.makedirs(inputs, exist_ok=True)
    receptor = _write(os.path.join(inputs, "receptor.pdb"),
                      "ATOM      1  CA  ALA A   1      "
                      "10.000  10.000  10.000  1.00  0.00           C\n"
                      "ATOM      2  CA  ALA A   2      "
                      "12.000  14.000  16.000  1.00  0.00           C\nEND\n")
    ligand = _write(os.path.join(inputs, "ligand.sdf"),
                    "%s\n  fixture\n\n"
                    "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
                    "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
                    "M  END\n$$$$\n" % job_id)
    pocket = _write(os.path.join(inputs, "pocket.pdb"),
                    "ATOM      1  CA  ALA A   1      "
                    "11.000  12.000  13.000  1.00  0.00           C\n"
                    "ATOM      2  CA  ALA A   2      "
                    "15.000  16.000  17.000  1.00  0.00           C\nEND\n")
    return {"current_outfolder": job_dir, "receptor_pdb": receptor,
            "ligand_sdf": ligand, "pocket_pdb": pocket}


def seed_other_tool(job_dir, tool="smina", poses=2):
    """A sibling tool's results, so the job has the two tools RMSD needs."""
    tool_dir = os.path.join(job_dir, tool)
    os.makedirs(tool_dir, exist_ok=True)
    _write(os.path.join(tool_dir, "results.csv"), "Pose,SMINA_Score\n1,-8.1\n2,-7.9\n")
    for i in range(1, poses + 1):
        _write(os.path.join(tool_dir, "complex_%d.pdb" % i), "ATOM\nEND\n")
    return tool_dir


class AsyncGninaFixture(unittest.TestCase):
    """A run directory, a fake GPU host and a queue, torn down per test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.run_dir = os.path.join(self.root, "run")
        os.makedirs(self.run_dir, exist_ok=True)
        self.remote = build_fake_transport(self.root)
        self.consensus = build_fake_consensus_docker(
            os.path.join(self.root, "fake_consensus_docker.py"))
        self.caps_file = os.path.join(self.root, "caps.json")
        remote_caps.save_caps(
            remote_caps.parse_probe_output(MEASURED_PROBE, "onur@fakehost", port=50000),
            self.caps_file)

        self._saved_env = dict(os.environ)
        os.environ["PATH"] = self.remote["bin_dir"] + os.pathsep + os.environ["PATH"]
        os.environ["GNINA_REMOTE_HOST"] = "onur@fakehost"
        os.environ["GNINA_REMOTE_PORT"] = "50000"
        os.environ["GNINA_REMOTE_NATIVE"] = self.remote["gnina"]
        os.environ["GNINA_REMOTE_DIR"] = "gnina_jobs"
        os.environ["GNINA_REMOTE_RETRIES"] = "1"      # fail fast; the link is local
        os.environ.pop("GNINA_REMOTE_MODE", None)
        os.environ.pop("GNINA_REMOTE_JOB_DIR", None)
        os.environ.pop("GNINA_REMOTE_TICKET", None)
        os.environ.pop("GNINA_REMOTE_KEY", None)
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        os.environ.clear()
        os.environ.update(self._saved_env)

    def tearDown(self):
        self._tmp.cleanup()

    # -- helpers ---------------------------------------------------------

    def params(self, **overrides):
        params = {
            "schema_version": 1,
            "cpu": 24,
            "exhaustiveness": 32,
            "num_modes": None,
            "cutoff_value": -7.0,
            "consensus_docker_script": self.consensus,
            "gnina_path": os.path.join(REPO, "gnina_remote.py"),
            "image": gnina_remote.DEFAULT_IMAGE,
            "host": "onur@fakehost",
            "port": 50000,
            "key": None,
            "native": self.remote["gnina"],
            "remote_root": "gnina_jobs",
            "job_deadline_seconds": 1800.0,
            "max_attempts": 3,
            # The suite's interpreter, so the stand-in runs under the same
            # Python the test does rather than whatever "python" is on PATH.
            "python_executable": sys.executable,
            "created_at": "2026-08-13T00:00:00.000000Z",
        }
        params.update(overrides)
        return params

    def enqueue(self, job_ids, **param_overrides):
        jobs = [build_job(self.run_dir, jid) for jid in job_ids]
        queue = gnina_queue.GninaQueue(self.run_dir)
        queue.enqueue_many(jobs, self.params(**param_overrides))
        return queue, jobs

    def drain(self, queue, max_seconds=90.0, **collector_kw):
        """Run a real Collector until the queue drains, or fail loudly."""
        kwargs = {"poll": 0.15, "collect_workers": 2, "max_attempts": 2,
                  "stop_when_drained": True}
        kwargs.update(collector_kw)
        collector = gnina_collect.Collector(self.run_dir, **kwargs)
        deadline = time.time() + max_seconds
        try:
            while time.time() < deadline:
                collector.run_once()
                if not [r for r in queue.read_all() if not r.is_terminal]:
                    return collector
            outstanding = [(r.job_id, r.state)
                           for r in queue.read_all() if not r.is_terminal]
            self.fail("the queue did not drain in %.0fs; outstanding: %s"
                      % (max_seconds, outstanding))
        finally:
            collector.close()


# ---------------------------------------------------------------------------
# the shim's half of the protocol
# ---------------------------------------------------------------------------

class TestShimProtocol(AsyncGninaFixture):
    """gnina_remote's submit/collect modes against the fake host.

    These are the exit codes and ticket outcomes the queue switches on. They are
    tested here rather than in isolation because the queue's reading of them is
    the thing that has to agree.
    """

    def _argv(self, job_dir):
        gnina_dir = os.path.join(job_dir, "gnina")
        os.makedirs(gnina_dir, exist_ok=True)
        job = build_job(self.run_dir, os.path.basename(job_dir))
        return ["-r", job["receptor_pdb"], "-l", job["ligand_sdf"],
                "--center_x", "1.4", "--center_y", "1.09", "--center_z", "-11.5",
                "--size_x", "33.0", "--size_y", "32.0", "--size_z", "33.5",
                "-o", os.path.join(gnina_dir, "out.sdf"),
                "--log", os.path.join(gnina_dir, "out.log"),
                "--cpu", "24", "--exhaustiveness", "32"]

    def _job_dir(self, job_id="DB00035__P30518__AVPR2__F1__cav1"):
        return os.path.join(self.run_dir, "docking_results", job_id)

    def test_submit_launches_and_never_downloads(self):
        job_dir = self._job_dir()
        argv = self._argv(job_dir)
        ticket_path = os.path.join(self.root, "t.submit.json")
        os.environ["GNINA_REMOTE_MODE"] = "submit"
        os.environ["GNINA_REMOTE_TICKET"] = ticket_path

        code = gnina_remote.main(argv)

        self.assertEqual(code, gnina_remote.EXIT_SUBMITTED,
                         "submit must exit 3: it has launched, not finished")
        ticket = _read_json(ticket_path)
        self.assertEqual(ticket["outcome"], "launched")
        self.assertEqual(ticket["lock"], "acquired")
        self.assertTrue(ticket["remote_dir"].startswith(self.remote["host_root"]))
        self.assertFalse(os.path.exists(os.path.join(job_dir, "gnina", "out.sdf")),
                         "submit must not produce out.sdf -- consensus_docker's "
                         "split_mol must fail on its absence, or the submit phase "
                         "would write a results.csv with no poses behind it")

    def test_the_remote_job_releases_the_mutex_when_it_finishes(self):
        """The submitter cannot renew a lease it does not outlive, so the JOB frees it."""
        os.environ["GNINA_REMOTE_MODE"] = "submit"
        os.environ["GNINA_REMOTE_TICKET"] = os.path.join(self.root, "t.json")
        self.assertEqual(gnina_remote.main(self._argv(self._job_dir())),
                         gnina_remote.EXIT_SUBMITTED)
        lock = os.path.join(self.remote["host_root"], "gnina_jobs", "gpu.lock")

        deadline = time.time() + 30
        while os.path.exists(lock) and time.time() < deadline:
            time.sleep(0.05)
        self.assertFalse(os.path.exists(lock),
                         "the detached job must remove the GPU mutex after writing "
                         "its exit code, or one crashed submitter blocks the GPU "
                         "for the whole campaign")

    def test_a_second_submit_while_the_gpu_is_busy_reports_busy(self):
        lock = os.path.join(self.remote["host_root"], "gnina_jobs", "gpu.lock")
        os.makedirs(lock)
        _write(os.path.join(lock, "pid"), "%d\n" % os.getpid())     # a live holder
        _write(os.path.join(lock, "acquired_at"), "%d\n" % int(time.time()))
        ticket_path = os.path.join(self.root, "t.json")
        os.environ["GNINA_REMOTE_MODE"] = "submit"
        os.environ["GNINA_REMOTE_TICKET"] = ticket_path

        code = gnina_remote.main(self._argv(self._job_dir()))

        self.assertEqual(code, gnina_remote.EXIT_LOCK_BUSY)
        self.assertEqual(_read_json(ticket_path)["outcome"], "busy")
        self.assertTrue(os.path.exists(lock), "a busy submit must not steal the lock")

    def test_collect_before_the_job_finishes_says_not_ready(self):
        remote_dir = os.path.join(self.remote["host_root"], "gnina_jobs", "gnina_x")
        os.makedirs(remote_dir)
        ticket_path = os.path.join(self.root, "t.json")
        os.environ["GNINA_REMOTE_MODE"] = "collect"
        os.environ["GNINA_REMOTE_JOB_DIR"] = remote_dir
        os.environ["GNINA_REMOTE_TICKET"] = ticket_path

        code = gnina_remote.main(self._argv(self._job_dir()))

        self.assertEqual(code, gnina_remote.EXIT_NOT_READY)
        self.assertEqual(_read_json(ticket_path)["outcome"], "not_ready")

    def test_collect_uses_the_recorded_directory_not_the_hash(self):
        """Collection must survive an input file that no longer hashes the same."""
        remote_dir = os.path.join(self.remote["host_root"], "gnina_jobs", "elsewhere")
        os.makedirs(remote_dir)
        _write(os.path.join(remote_dir, "out.sdf"), "pose\n$$$$\n")
        _write(os.path.join(remote_dir, "out.log"), "log\n")
        _write(os.path.join(remote_dir, "exit_code"), "0\n")
        job_dir = self._job_dir()
        argv = self._argv(job_dir)
        # Change the ligand *after* the job ran: the content address no longer
        # matches, and only the recorded directory can still find the result.
        _write(argv[3], "TOTALLY DIFFERENT LIGAND\n")
        os.environ["GNINA_REMOTE_MODE"] = "collect"
        os.environ["GNINA_REMOTE_JOB_DIR"] = remote_dir
        os.environ["GNINA_REMOTE_TICKET"] = os.path.join(self.root, "t.json")

        code = gnina_remote.main(argv)

        self.assertEqual(code, gnina_remote.EXIT_OK)
        self.assertEqual(_read(os.path.join(job_dir, "gnina", "out.sdf")),
                         "pose\n$$$$\n")

    def test_a_failed_remote_run_is_never_reported_as_success(self):
        remote_dir = os.path.join(self.remote["host_root"], "gnina_jobs", "bad")
        os.makedirs(remote_dir)
        _write(os.path.join(remote_dir, "exit_code"), "1\n")
        _write(os.path.join(remote_dir, "run.log"), "CUDA error: out of memory\n")
        ticket_path = os.path.join(self.root, "t.json")
        os.environ["GNINA_REMOTE_MODE"] = "collect"
        os.environ["GNINA_REMOTE_JOB_DIR"] = remote_dir
        os.environ["GNINA_REMOTE_TICKET"] = ticket_path

        code = gnina_remote.main(self._argv(self._job_dir()))

        self.assertEqual(code, gnina_remote.EXIT_FAILED)
        self.assertEqual(_read_json(ticket_path)["outcome"], "error")

    def test_an_unknown_mode_refuses_rather_than_guessing(self):
        os.environ["GNINA_REMOTE_MODE"] = "collct"          # a typo
        self.assertEqual(gnina_remote.main(self._argv(self._job_dir())),
                         gnina_remote.EXIT_USAGE)

    def test_sync_mode_still_works_against_a_real_transport(self):
        """The default path must keep working: the pilot and 774 tests rely on it."""
        job_dir = self._job_dir()
        argv = self._argv(job_dir)
        self.assertIsNone(os.environ.get("GNINA_REMOTE_MODE"))
        # Only the wait between completion polls is shortened; the path is the same.
        saved = gnina_remote.POLL_INTERVAL
        gnina_remote.POLL_INTERVAL = 0.1
        self.addCleanup(setattr, gnina_remote, "POLL_INTERVAL", saved)

        code = gnina_remote.main(argv)

        self.assertEqual(code, gnina_remote.EXIT_OK)
        self.assertTrue(os.path.getsize(os.path.join(job_dir, "gnina", "out.sdf")) > 0)
        self.assertTrue(os.path.exists(os.path.join(job_dir, "gnina", "out.log")))


# ---------------------------------------------------------------------------
# the whole chain
# ---------------------------------------------------------------------------

class TestEndToEndChain(AsyncGninaFixture):

    def test_capability_check_then_the_full_dock_and_collect(self):
        """caps -> enqueue -> submit -> remote finishes -> collect -> post-process."""
        # 1. capability check, on the measured capacity of the real box.
        caps = remote_caps.load_caps(self.caps_file)
        request = remote_caps.DockRequest(
            cpu=60, exhaustiveness=32, num_modes=None, job_deadline_seconds=1800.0,
            image=gnina_remote.DEFAULT_IMAGE, native=None,
            gnina_path=os.path.join(REPO, "gnina_remote.py"), async_mode=True)
        report = remote_caps.validate_request(caps, request)
        self.assertTrue(report.ok, "the intended campaign must not be refused")
        self.assertEqual(report.effective_cpu, 24, "60 threads clamped to 24 cores")

        # 2. enqueue, with the cpu the check resolved.
        queue, jobs = self.enqueue(["DB00035__P30518__AVPR2__F1__cav1"],
                                   cpu=report.effective_cpu)
        job_dir = jobs[0]["current_outfolder"]
        self.assertEqual(queue.read(gnina_queue.job_id_for(job_dir)).state, "PENDING")

        # 3. the pump: submit, poll, collect, post-process.
        self.drain(queue)

        record = queue.read(gnina_queue.job_id_for(job_dir))
        self.assertEqual(record.state, "COLLECTED", record.error)

        # 4. the gnina directory holds what a local run would have produced.
        gnina_dir = os.path.join(job_dir, "gnina")
        produced = sorted(os.listdir(gnina_dir))
        self.assertEqual(produced, sorted([
            "out.sdf", "out.log",
            "out_1.sdf", "out_2.sdf", "out_3.sdf",
            "complex_1.pdb", "complex_2.pdb", "complex_3.pdb",
            "results.csv"]),
            "the collected directory must match a local run's, file for file")

        rows = _read(os.path.join(gnina_dir, "results.csv")).splitlines()
        self.assertEqual(rows[0],
                         "Pose,GNINA_Affinity,GNINA_CNN_pose,GNINA_CNN_affinity",
                         "the CNN columns are produced by consensus_docker's own "
                         "parse_gnina and must not be reimplemented anywhere")
        self.assertEqual(len(rows), 4, "one header + one row per pose")
        self.assertEqual(record.outputs["n_poses"], 3)
        self.assertEqual(record.outputs["n_complexes"], 3)

        # 5. both phases derived the SAME search box from the same pocket file.
        invocations = [line.split("\t") for line in
                       _read(os.path.join(job_dir, "invocations.log")).splitlines()]
        modes = [mode for mode, _cmd in invocations]
        self.assertEqual(modes, ["submit", "collect"],
                         "consensus_docker runs exactly twice: once to launch, "
                         "once to post-process")
        self.assertEqual(invocations[0][1], invocations[1][1],
                         "the collect phase must rebuild the identical gnina "
                         "command; a different box would post-process poses that "
                         "were searched somewhere else")

    def test_the_submit_phase_writes_nothing_into_the_job(self):
        """Runtime invariant R3: a results.csv after a submit means broken contract."""
        queue, jobs = self.enqueue(["DB00777__Q9Y5N1__HRH3__F1__cav2"])
        job_dir = jobs[0]["current_outfolder"]
        record = queue.next_pending()

        gnina_queue.submit_job(queue, record)

        record = queue.read(record.job_id)
        self.assertEqual(record.state, "SUBMITTED")
        self.assertFalse(os.path.exists(os.path.join(job_dir, "gnina", "results.csv")),
                         "R3: the submit phase must not write results.csv")
        self.assertFalse(os.path.exists(os.path.join(job_dir, "final_results.csv")),
                         "R1: --skip_rmsd on both phases, or a gnina-less consensus "
                         "would be written and the job would look finished")

    def test_stale_outputs_are_purged_before_a_new_attempt(self):
        """Rule R2 -- the most dangerous bug in the design if it is skipped.

        The debris modelled here is what an *interrupted* attempt leaves: poses
        and complexes but no results.csv, so the job is genuinely unfinished and
        will be docked again. (Debris that looks complete is a different case
        entirely: outputs_complete() is true, reconciliation promotes the job to
        COLLECTED and nothing is re-docked -- deliberately, because that is the
        skip-if-done rule that keeps a campaign's second run cheap.)
        """
        queue, jobs = self.enqueue(["DB00001__P00001__AAA__F1__cav1"])
        job_dir = jobs[0]["current_outfolder"]
        gnina_dir = os.path.join(job_dir, "gnina")
        os.makedirs(gnina_dir, exist_ok=True)
        _write(os.path.join(gnina_dir, "out.sdf"), "STALE POSES FROM AN OLD RUN\n$$$$\n")
        _write(os.path.join(gnina_dir, "out_9.sdf"), "STALE\n")
        _write(os.path.join(gnina_dir, "complex_9.pdb"), "STALE\n")
        self.assertFalse(gnina_queue.outputs_complete(job_dir),
                         "the fixture must be an unfinished job, or this tests "
                         "skip-if-done instead of the purge")

        self.drain(queue)

        content = _read(os.path.join(gnina_dir, "out.sdf"))
        self.assertNotIn("STALE", content,
                         "split_mol would have succeeded on these stale bytes and "
                         "parse_gnina would have written scores for the wrong run")
        self.assertFalse(os.path.exists(os.path.join(gnina_dir, "out_9.sdf")),
                         "a stale pose file left behind would be globbed into the "
                         "consensus RMSD alongside this run's poses")
        self.assertFalse(os.path.exists(os.path.join(gnina_dir, "complex_9.pdb")))
        self.assertEqual(sorted(os.listdir(gnina_dir)), sorted([
            "out.sdf", "out.log", "out_1.sdf", "out_2.sdf", "out_3.sdf",
            "complex_1.pdb", "complex_2.pdb", "complex_3.pdb", "results.csv"]))

    def test_debris_that_looks_complete_is_not_re_docked(self):
        """The other half of R2, and the reason outputs_complete() stays loose.

        A stricter predicate here would re-dock every finished job on a
        campaign's second run -- thousands of GPU-hours to reproduce files that
        are already on disk.
        """
        queue, jobs = self.enqueue(["DB00002__P00002__AAA__F1__cav1"])
        job_dir = jobs[0]["current_outfolder"]
        gnina_dir = os.path.join(job_dir, "gnina")
        os.makedirs(gnina_dir, exist_ok=True)
        _write(os.path.join(gnina_dir, "out.sdf"), "POSES FROM THE FIRST RUN\n$$$$\n")
        _write(os.path.join(gnina_dir, "results.csv"),
               "Pose,GNINA_Affinity,GNINA_CNN_pose,GNINA_CNN_affinity\n1,-8.5,0.8,6.1\n")

        self.drain(queue)

        self.assertEqual(queue.read(gnina_queue.job_id_for(job_dir)).state, "COLLECTED")
        self.assertFalse(os.path.exists(os.path.join(job_dir, "invocations.log")),
                         "consensus_docker must not run at all for a job whose "
                         "outputs are already on disk")

    def test_only_one_job_is_ever_on_the_gpu(self):
        """THE SINGLE-GPU INVARIANT, through the real mutex on the fake host."""
        job_ids = ["DB0000%d__P0000%d__T%d__F1__cav1" % (i, i, i) for i in range(1, 6)]
        queue, _jobs = self.enqueue(job_ids)

        collector = gnina_collect.Collector(self.run_dir, poll=0.05,
                                            collect_workers=2, max_attempts=2)
        try:
            deadline = time.time() + 120
            while time.time() < deadline:
                collector.run_once()
                outstanding = [r for r in queue.read_all()
                               if r.state in ("SUBMITTING", "SUBMITTED", "RUNNING")]
                self.assertLessEqual(
                    len(outstanding), 1,
                    "two gnina jobs on one GPU: %s" % [r.job_id for r in outstanding])
                if not [r for r in queue.read_all() if not r.is_terminal]:
                    break
            else:
                self.fail("the queue did not drain")
        finally:
            collector.close()

        states = dict((r.job_id, r.state) for r in queue.read_all())
        self.assertEqual(sorted(set(states.values())), ["COLLECTED"], states)

        # The sampling above only sees the gaps between passes, and against a
        # fake GPU a whole job can start and finish inside one pass. So the
        # invariant is also checked over the recorded history, which cannot be
        # missed by sampling: every job's time on the GPU is an interval, and no
        # two of those intervals may overlap.
        intervals = []
        for record in queue.read_all():
            occupied = [h["at"] for h in record.history
                        if h["to"] in ("SUBMITTING", "SUBMITTED", "RUNNING")]
            released = [h["at"] for h in record.history
                        if h["to"] in ("REMOTE_DONE", "FAILED", "PENDING")
                        and h["from"] in ("SUBMITTING", "SUBMITTED", "RUNNING")]
            self.assertTrue(occupied, "%s never went to the GPU at all: %s"
                            % (record.job_id, record.history))
            intervals.append((min(occupied), max(released) if released else "9999",
                              record.job_id))
        intervals.sort()
        for (_start, end, job), (next_start, _e, next_job) in zip(intervals,
                                                                  intervals[1:]):
            self.assertLessEqual(
                end, next_start,
                "%s was still on the GPU when %s started: one GPU, one gnina"
                % (job, next_job))

    def test_a_finished_job_is_never_docked_twice(self):
        """Re-running a campaign must cost nothing for work that already exists."""
        queue, jobs = self.enqueue(["DB00042__P00042__BBB__F1__cav1"])
        job_dir = jobs[0]["current_outfolder"]
        self.drain(queue)
        first = _read(os.path.join(job_dir, "invocations.log")).splitlines()

        queue2 = gnina_queue.GninaQueue(self.run_dir)
        stats = queue2.enqueue_many([jobs[0]], self.params())
        self.drain(queue2)

        self.assertEqual(stats["already_present"], 1)
        self.assertEqual(queue2.read(gnina_queue.job_id_for(job_dir)).state, "COLLECTED")
        self.assertEqual(_read(os.path.join(job_dir, "invocations.log")).splitlines(),
                         first, "a second run must not invoke consensus_docker again")

    def test_a_remote_failure_is_retried_then_abandoned_not_silently_dropped(self):
        _write(self.remote["behaviour"], "fail")
        queue, jobs = self.enqueue(["DB00099__P00099__CCC__F1__cav1"])

        self.drain(queue, max_seconds=120)

        record = queue.read(gnina_queue.job_id_for(jobs[0]["current_outfolder"]))
        self.assertEqual(record.state, "ABANDONED")
        self.assertEqual(record.error["code"], "REMOTE_GNINA_FAILED")
        self.assertFalse(
            os.path.exists(os.path.join(jobs[0]["current_outfolder"], "gnina",
                                        "results.csv")),
            "a failed remote run must leave no results.csv behind to be mistaken "
            "for a real one")

    def test_a_crashed_pump_mid_collect_is_recovered_not_lost(self):
        """The expensive part is on the GPU; a local crash must never re-dock it."""
        queue, jobs = self.enqueue(["DB00123__P00123__DDD__F1__cav1"])
        job_dir = jobs[0]["current_outfolder"]
        record = queue.next_pending()
        record = gnina_queue.submit_job(queue, record)

        # Wait for the remote to finish, then simulate a pump that died between
        # starting the collect and finishing it.
        deadline = time.time() + 30
        while time.time() < deadline:
            state, code = gnina_queue.default_prober(queue.read(record.job_id))
            if state == "DONE":
                break
            time.sleep(0.05)
        queue.transition(record.job_id, "REMOTE_DONE", why="test", remote={"exit_code": 0})
        queue.transition(record.job_id, "COLLECTING", why="test",
                         collect={"pid": 999999, "host": None})     # a dead pid
        os.makedirs(os.path.join(job_dir, "gnina"), exist_ok=True)
        _write(os.path.join(job_dir, "gnina", "out.sdf"), "HALF DOWNLOADED")

        self.drain(queue)

        record = queue.read(record.job_id)
        self.assertEqual(record.state, "COLLECTED")
        invocations = _read(os.path.join(job_dir, "invocations.log"))
        self.assertEqual(invocations.count("submit\t"), 1,
                         "recovery must collect, never re-dock: the GPU already "
                         "did this work")
        self.assertNotIn("HALF DOWNLOADED",
                         _read(os.path.join(job_dir, "gnina", "out.sdf")))


# ---------------------------------------------------------------------------
# the barrier, and what stage 5 is then allowed to compute
# ---------------------------------------------------------------------------

class TestBarrierAndRmsd(AsyncGninaFixture):

    def test_the_barrier_returns_when_the_queue_drains_and_rmsd_proceeds(self):
        queue, jobs = self.enqueue(["DB00035__P30518__AVPR2__F1__cav1"])
        job_dir = jobs[0]["current_outfolder"]
        seed_other_tool(job_dir, "smina")
        self.drain(queue)

        summary = batch_dock.wait_for_gnina_stage(self.run_dir, timeout=30)

        self.assertEqual(summary["collected"], 1)
        self.assertEqual(summary["outstanding"], 0)
        self.assertFalse(summary["timed_out"])
        # What stage 5 asks before it spends a job:
        self.assertTrue(batch_dock.check_tool_completion(job_dir, "gnina"))
        self.assertGreaterEqual(
            batch_dock.rmsd_inputs_present(job_dir), 2,
            "consensus RMSD needs complex_*.pdb from two tools; gnina's must be "
            "there by the time the barrier returns")

    def test_a_job_whose_gnina_never_landed_is_excluded_and_named(self):
        """A partial campaign must never be analysable as a complete one."""
        landed, missing = ("DB00001__P00001__AAA__F1__cav1",
                           "DB00002__P00002__BBB__F1__cav1")
        queue, jobs = self.enqueue([landed, missing])
        by_id = dict((os.path.basename(j["current_outfolder"]), j) for j in jobs)
        for job in jobs:
            seed_other_tool(job["current_outfolder"], "smina")

        # Only the first job gets to run; the second is abandoned, as it would be
        # after exhausting its attempts against a broken remote.
        record = queue.read(landed)
        while record.state != "COLLECTED":
            record = self._advance(queue, record)
        # ...the way a real job reaches ABANDONED: submitted, failed, attempts
        # exhausted. (PENDING -> ABANDONED is rejected by the state machine, and
        # rightly: nothing may be written off before it has been tried.)
        queue.transition(missing, "SUBMITTING", why="test")
        queue.transition(missing, "FAILED", why="test",
                         error=gnina_queue.make_error(
                             "REMOTE_GNINA_FAILED", "gnina failed twice", "poll"))
        queue.transition(missing, "ABANDONED", why="test: attempts exhausted")

        summary = batch_dock.wait_for_gnina_stage(self.run_dir, timeout=10)
        self.assertEqual(summary["collected"], 1)
        self.assertEqual(summary["abandoned"], 1)
        self.assertEqual(summary["outstanding"], 0)

        # Stage 5's filter, unchanged: only the complete job is an RMSD job.
        eligible = [jid for jid in (landed, missing)
                    if batch_dock.check_tool_completion(by_id[jid]["current_outfolder"],
                                                        "gnina")]
        self.assertEqual(eligible, [landed])

        with self.assertLogs(level="WARNING") as logs:
            batch_dock._report_gnina_exclusions(self.run_dir, [missing], [])
        self.assertTrue(any(missing in line for line in logs.output),
                        "the excluded job must be named in the log: silence is how "
                        "a partial dataset gets analysed as a complete one")
        summary_path = os.path.join(gnina_queue.queue_root(self.run_dir), "summary.json")
        self.assertIn(missing, _read_json(summary_path)["rmsd_excluded_jobs"])

    def test_a_leftover_stop_file_says_why_it_stopped(self):
        """A pump that stops before doing anything must not look like a broken one.

        batch_dock leaves ``STOP`` behind whenever it did not spawn the collector
        itself, so the next hand-started pump obeys a stop nobody asked it for.
        Obeying is right; being quiet about it is not.
        """
        queue, _jobs = self.enqueue(["DB00004__P00004__DDD__F1__cav1"])
        _write(os.path.join(queue.root, "STOP"), "{}")
        collector = gnina_collect.Collector(self.run_dir, poll=0.05)
        try:
            with self.assertLogs(gnina_collect.LOG, level="WARNING") as logs:
                collector.run_once()
        finally:
            collector.close()

        self.assertTrue(any("STOP" in line for line in logs.output), logs.output)
        self.assertTrue(collector.stopped)

    def test_the_barrier_times_out_without_failing_the_run(self):
        """Stage 5 must still compute what it can; the rest are named, not lost."""
        queue, jobs = self.enqueue(["DB00003__P00003__CCC__F1__cav1"])

        started = time.time()
        summary = batch_dock.wait_for_gnina_stage(self.run_dir, timeout=0)

        self.assertLess(time.time() - started, 30, "timeout=0 must not wait")
        self.assertEqual(summary["outstanding"], 1)
        self.assertIn(gnina_queue.job_id_for(jobs[0]["current_outfolder"]),
                      summary["outstanding_job_ids"])

    def _advance(self, queue, record):
        """One step of the pump, without the pump: used to control the ordering."""
        if record.state == "PENDING":
            return gnina_queue.submit_job(queue, record)
        if record.state in ("SUBMITTED", "RUNNING"):
            time.sleep(0.05)
            return gnina_queue.poll_job(queue, record)
        if record.state == "REMOTE_DONE":
            return gnina_queue.collect_job(queue, record)
        self.fail("cannot advance from %s: %s" % (record.state, record.error))


if __name__ == "__main__":
    unittest.main()
