"""Tests for gnina_collect.py, the pump that files remote gnina results.

Hermetic by construction: **no SSH, no GPU host, no writes outside a
TemporaryDirectory**. Every remote interaction in this design goes through an
injected seam -- ``runner`` for the consensus_docker subprocess, ``prober`` for
the remote state probe, ``clock`` for time -- so the whole protocol runs
in-process. :class:`FakeRemote` below stands in for both the GPU box and the two
consensus_docker invocations, and it is deliberately strict: it asserts the
contract the pump is supposed to honour (``--skip_rmsd`` on both phases, the
purge before each phase, one gnina on the GPU at a time) and records a violation
rather than quietly tolerating one.

What is being protected here is a sync job with several ways to lose or corrupt
scientific output:

* a second collection of the same job must be harmless;
* a crash half way through a collection must leave the job recoverable and must
  never leave a half-written ``gnina/`` that looks finished;
* a truncated download must be rejected, not filed;
* a job that cannot be collected must end up named, with its reason, not gone.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    from dock_selectivity import gnina_queue as gq
    import gnina_collect as gc
except ImportError as exc:                        # pragma: no cover
    raise unittest.SkipTest(
        "dock_selectivity.gnina_queue is not available yet (%s); gnina_collect "
        "is a driver for it and has nothing to test without it." % exc)


# --- fixtures -----------------------------------------------------------------

JOB_IDS = [
    "DB00035__P30518__AVPR2__F1__cav1",
    "DB00091__P62942__FKBP1A__F1__cav2",
    "DB00104__P30874__SSTR2__F1__cav1",
]

#: The exact header consensus_docker.parse_gnina writes. If a future change to
#: the collector ever starts producing results.csv itself, this literal is what
#: catches the fork.
GNINA_RESULTS_HEADER = "Pose,GNINA_Affinity,GNINA_CNN_pose,GNINA_CNN_affinity"


class FakeClock(object):
    """Monotonic-like clock the test drives, so backoff is exercised without waiting."""

    def __init__(self, t=1000.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += float(dt)


class AutoClock(FakeClock):
    """A clock that steps forward every time it is read.

    ``main()`` builds its own Collector, so a test driving the CLI cannot call
    ``advance()`` between passes. Without a clock that moves, a job that has just
    backed off for five seconds never becomes eligible again and ``--daemon``
    loops forever -- which is exactly what a frozen clock means.
    """

    def __init__(self, t=1000.0, step=10.0):
        FakeClock.__init__(self, t)
        self.step = float(step)

    def __call__(self):
        self.t += self.step
        return self.t


class FakeRemote(object):
    """The GPU host and the two consensus_docker phases, in-process.

    ``runner`` is what the queue calls instead of spawning consensus_docker;
    ``prober`` is what it calls instead of an ssh probe. The remote GPU mutex is
    emulated for real: a submit while another job is running returns ``busy``,
    exactly as ``gnina_remote`` does, and the attempt is recorded so a test can
    assert the pump never even contends for it.
    """

    def __init__(self, poses=3, polls_before_done=0):
        self.poses = poses
        self.polls_before_done = polls_before_done
        self.jobs = {}                 # job_id -> {"state","exit_code","polls"}
        self.calls = []                # ("submit"|"collect", job_id)
        self.probe_calls = []
        self.mutex_conflicts = []      # submits attempted while one was running
        self.remote_exit = {}          # job_id -> exit code the GPU will report
        self.collect_mode = {}         # job_id -> "ok"|"truncated"|"empty"|"not_ready"
        self.submit_outcome = {}       # job_id -> forced ticket outcome
        self.raise_on_collect = set()
        self.seen_skip_rmsd = []
        self.purged_before = []        # (phase, job_id, files present in gnina/)

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _arg(cmd, flag):
        return cmd[cmd.index(flag) + 1] if flag in cmd else None

    def _gnina_dir(self, job_dir):
        path = os.path.join(job_dir, "gnina")
        os.makedirs(path, exist_ok=True)
        return path

    def _running(self):
        return [j for j, s in self.jobs.items() if s["state"] == "running"]

    # -- the runner seam -------------------------------------------------------
    def runner(self, cmd, env, timeout, log_path, cwd):
        job_dir = self._arg(cmd, "--outfolder")
        job_id = os.path.basename(os.path.normpath(job_dir))
        mode = env.get("GNINA_REMOTE_MODE")
        ticket_path = env.get("GNINA_REMOTE_TICKET")
        assert ticket_path, (
            "build_phase_env must set GNINA_REMOTE_TICKET: the queue switches on "
            "the ticket, not on the exit code")
        # R1: RMSD is computed exactly once, in stage 5, after the barrier.
        assert "--skip_rmsd" in cmd, "both phases must pass --skip_rmsd (R1)"
        self.seen_skip_rmsd.append((mode, job_id))
        # R2: whatever is in gnina/ now must not be a previous attempt's output.
        gdir = os.path.join(job_dir, "gnina")
        present = sorted(os.listdir(gdir)) if os.path.isdir(gdir) else []
        self.purged_before.append((mode, job_id, present))
        with open(log_path, "a") as handle:
            handle.write("fake consensus_docker %s %s\n" % (mode, job_id))
        if mode == "submit":
            return self._submit(job_id, job_dir, ticket_path)
        if mode == "collect":
            return self._collect(job_id, job_dir, ticket_path, env)
        raise AssertionError("unexpected GNINA_REMOTE_MODE %r" % mode)

    def _submit(self, job_id, job_dir, ticket_path):
        self.calls.append(("submit", job_id))
        remote_dir = "/home/onur/gnina_jobs/gnina_%s" % job_id[:16]
        outcome = self.submit_outcome.get(job_id)
        running = self._running()
        if outcome is None and running and running != [job_id]:
            # The remote mkdir mutex is held by someone else.
            self.mutex_conflicts.append((job_id, list(running)))
            outcome = "busy"
        if outcome is None:
            outcome = "launched"
        if outcome == "launched":
            self.jobs[job_id] = {"state": "running", "polls": 0,
                                 "exit_code": self.remote_exit.get(job_id, 0),
                                 "dir": remote_dir}
        self._write_ticket(ticket_path, {
            "schema_version": 1, "mode": "submit", "outcome": outcome,
            "host": "user@fake", "port": 50000, "remote_dir": remote_dir,
            "job_key": job_id[:16], "image": "gnina/gnina:v1.3.3",
            "lock": "acquired" if outcome == "launched" else "busy",
            "exit_code": None, "error": None,
        })
        # consensus_docker catches "gnina docking failed" and exits 0 having
        # written nothing. The queue must not care either way.
        return 0, "", "gnina docking failed"

    def _collect(self, job_id, job_dir, ticket_path, env):
        self.calls.append(("collect", job_id))
        assert env.get("GNINA_REMOTE_JOB_DIR"), (
            "collect must be pinned to the recorded remote dir, never re-derived "
            "from the content address")
        if job_id in self.raise_on_collect:
            self.raise_on_collect.discard(job_id)
            raise RuntimeError("openbabel segfaulted while splitting out.sdf")
        state = self.jobs.get(job_id)
        mode = self.collect_mode.get(job_id, "ok")
        if state is None or state["state"] != "done" or mode == "not_ready":
            self._write_ticket(ticket_path, {"schema_version": 1, "mode": "collect",
                                             "outcome": "not_ready"})
            return 4, "", "remote not finished"
        gdir = self._gnina_dir(job_dir)
        if mode == "empty":
            open(os.path.join(gdir, "out.sdf"), "w").close()
            self._write_ticket(ticket_path, {"schema_version": 1, "mode": "collect",
                                             "outcome": "collected"})
            return 0, "", ""
        self._write_local_outputs(gdir, truncated=(mode == "truncated"))
        self._write_ticket(ticket_path, {"schema_version": 1, "mode": "collect",
                                         "outcome": "collected"})
        return 0, "gnina output\n", ""

    def _write_local_outputs(self, gdir, truncated=False):
        """Exactly the files consensus_docker leaves in gnina/ after a local run.

        run_gnina() is four steps: the binary (out.sdf/out.log), split_mol
        (out_N.sdf), make_complex (complex_N.pdb) and parse_gnina (results.csv).
        ``truncated`` stops after the download, which is what a killed collect or
        a half-finished post-processing leaves behind.
        """
        with open(os.path.join(gdir, "out.sdf"), "w") as handle:
            handle.write("".join("pose %d\n$$$$\n" % (i + 1)
                                 for i in range(self.poses)))
        with open(os.path.join(gdir, "out.log"), "w") as handle:
            handle.write("mode |  affinity | intramol |    CNN pose |  CNN affinity\n")
            handle.write("-----+-----------+----------+-------------+--------------\n")
            for i in range(self.poses):
                handle.write("%d  %.2f  0.00  %.4f  %.4f\n"
                             % (i + 1, -8.0 - i, 0.9 - 0.1 * i, 6.0 - i))
        if truncated:
            with open(os.path.join(gdir, "results.csv"), "w") as handle:
                handle.write(GNINA_RESULTS_HEADER + "\n")   # header only, no poses
            return
        for i in range(self.poses):
            with open(os.path.join(gdir, "out_%d.sdf" % (i + 1)), "w") as handle:
                handle.write("pose %d\n$$$$\n" % (i + 1))
            with open(os.path.join(gdir, "complex_%d.pdb" % (i + 1)), "w") as handle:
                handle.write("ATOM      1  N   ALA A   1      0.000\nEND\n")
        with open(os.path.join(gdir, "results.csv"), "w") as handle:
            handle.write(GNINA_RESULTS_HEADER + "\n")
            for i in range(self.poses):
                handle.write("%d,%.2f,%.4f,%.4f\n"
                             % (i + 1, -8.0 - i, 0.9 - 0.1 * i, 6.0 - i))

    @staticmethod
    def _write_ticket(path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write(json.dumps(payload))

    # -- the prober seam -------------------------------------------------------
    def prober(self, record):
        job_id = record.job_id
        self.probe_calls.append(job_id)
        state = self.jobs.get(job_id)
        if state is None:
            return ("GONE", None)
        if state["state"] == "running":
            state["polls"] += 1
            if state["polls"] > self.polls_before_done:
                state["state"] = "done"
                return ("DONE", state["exit_code"])
            return ("RUNNING", None)
        if state["state"] == "done":
            return ("DONE", state["exit_code"])
        return ("LAUNCHED", None)                       # pragma: no cover

    def finish(self, job_id, exit_code=0):
        self.jobs.setdefault(job_id, {"dir": "/x"})
        self.jobs[job_id].update({"state": "done", "exit_code": exit_code})


class QueueFixture(object):
    """A run directory with N enqueued jobs, built in a temp dir."""

    def __init__(self, n_jobs=1, job_ids=None, clock=None):
        self.tmp = tempfile.mkdtemp(prefix="gnina_collect_test_")
        self.run_dir = os.path.join(self.tmp, "run")
        self.results = os.path.join(self.run_dir, "docking_results")
        os.makedirs(self.results)
        self.clock = clock or FakeClock()
        self.job_ids = list(job_ids or [])
        if not self.job_ids:
            self.job_ids = [JOB_IDS[i] if i < len(JOB_IDS)
                            else "DB%05d__P%05d__GENE%d__F1__cav1" % (i, i, i)
                            for i in range(n_jobs)]
        self.queue = gq.GninaQueue(self.run_dir, create=True, clock=self.clock)
        self.queue.set_params(self.params(), allow_change=True)
        for job_id in self.job_ids:
            self.queue.enqueue(self.make_job_dir(job_id), self.inputs(job_id))

    def params(self):
        return {
            "schema_version": 1, "cpu": 24, "exhaustiveness": 32,
            "num_modes": None, "cutoff_value": -7.0,
            "consensus_docker_script": "/home/onur/repos/consensus_docking/consensus_docker.py",
            "gnina_path": os.path.join(REPO, "gnina_remote.py"),
            "image": "gnina/gnina:v1.3.3", "host": "user@fake", "port": 50000,
            "key": None, "native": None, "remote_root": "gnina_jobs",
            "job_deadline_seconds": 1800.0, "max_attempts": 3,
        }

    def job_dir(self, job_id):
        return os.path.join(self.results, job_id)

    def make_job_dir(self, job_id):
        path = self.job_dir(job_id)
        for sub in ("input", "gnina", "smina", "gold", "ledock"):
            os.makedirs(os.path.join(path, sub), exist_ok=True)
        for name, body in (("receptor.pdb", "ATOM\n"), ("ligand.sdf", "MOL\n"),
                           ("pocket.pdb", "ATOM\n")):
            with open(os.path.join(path, "input", name), "w") as handle:
                handle.write(body)
        return path

    def add_job(self, job_id):
        """Enqueue one more job after the fixture was built."""
        self.job_ids.append(job_id)
        self.queue.enqueue(self.make_job_dir(job_id), self.inputs(job_id))
        return job_id

    def inputs(self, job_id):
        base = os.path.join(self.job_dir(job_id), "input")
        return {"receptor_pdb": os.path.join(base, "receptor.pdb"),
                "ligand_sdf": os.path.join(base, "ligand.sdf"),
                "pocket_pdb": os.path.join(base, "pocket.pdb"),
                "receptor_pdbqt": None, "skip_pdb_to_pdbqt": False}

    def collector(self, remote, **kw):
        kw.setdefault("poll", 0.0)
        kw.setdefault("clock", self.clock)
        return gc.Collector(self.run_dir, runner=remote.runner,
                            prober=remote.prober, **kw)

    def state(self, job_id):
        record = self.queue.read(job_id)
        return None if record is None else record.state

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class CollectorTestCase(unittest.TestCase):
    def setUp(self):
        self.fixtures = []

    def tearDown(self):
        for fixture in self.fixtures:
            fixture.cleanup()

    def fixture(self, **kw):
        fixture = QueueFixture(**kw)
        self.fixtures.append(fixture)
        return fixture

    def install_cli_seams(self, fixture, remote):
        """Point the CLI's own Collector at the fake remote.

        ``main()`` constructs its Collector itself, and production deliberately
        has no flag for "pretend there is a GPU", so the seams are injected by
        wrapping the constructor for the duration of one test.
        """
        original = gc.Collector.__init__

        def patched(self, *args, **kwargs):
            kwargs.setdefault("runner", remote.runner)
            kwargs.setdefault("prober", remote.prober)
            kwargs.setdefault("clock", fixture.clock)
            original(self, *args, **kwargs)

        gc.Collector.__init__ = patched
        self.addCleanup(setattr, gc.Collector, "__init__", original)


# --- the happy path -----------------------------------------------------------


class TestOnePassPipeline(CollectorTestCase):
    def test_one_pass_drives_a_job_from_pending_to_collected(self):
        """PENDING -> SUBMITTED -> REMOTE_DONE -> COLLECTED, in that order."""
        fx = self.fixture(n_jobs=1)
        remote = FakeRemote()
        job_id = fx.job_ids[0]
        with fx.collector(remote) as collector:
            counts = collector.run_once()
        self.assertEqual(fx.state(job_id), "COLLECTED")
        self.assertEqual(counts["submitted"], 1)
        self.assertEqual(counts["collected"], 1)
        # Order matters: nothing may be collected before it has been submitted
        # and confirmed done on the remote.
        self.assertEqual(
            [t for t in collector.trace if t.split(":")[0] in
             ("submit", "poll", "collect")],
            ["submit:%s->SUBMITTED" % job_id,
             "poll:%s->REMOTE_DONE" % job_id,
             "collect:%s->COLLECTED" % job_id])
        self.assertEqual([c[0] for c in remote.calls], ["submit", "collect"])

    def test_the_submit_phase_exit_code_is_ignored(self):
        """consensus_docker exits 0 having written nothing; the TICKET is the signal."""
        fx = self.fixture(n_jobs=1)
        remote = FakeRemote()
        with fx.collector(remote) as collector:
            collector.run_once()
        self.assertEqual(fx.state(fx.job_ids[0]), "COLLECTED")

    def test_both_phases_pass_skip_rmsd(self):
        """R1: without it the submit phase writes a gnina-less final_results.csv."""
        fx = self.fixture(n_jobs=1)
        remote = FakeRemote()
        with fx.collector(remote) as collector:
            collector.run_once()
        self.assertEqual(sorted(m for m, _ in remote.seen_skip_rmsd),
                         ["collect", "submit"])

    def test_a_job_whose_remote_result_already_exists_is_not_re_docked(self):
        fx = self.fixture(n_jobs=1)
        remote = FakeRemote()
        job_id = fx.job_ids[0]
        remote.submit_outcome[job_id] = "already_done"
        remote.finish(job_id, 0)
        with fx.collector(remote) as collector:
            collector.run_once()
        self.assertEqual(fx.state(job_id), "COLLECTED")
        self.assertEqual([c[0] for c in remote.calls], ["submit", "collect"])


class TestFilesLandingInGninaDir(CollectorTestCase):
    """The result must be indistinguishable from a locally-run gnina."""

    def test_gnina_dir_contains_exactly_what_a_local_run_produces(self):
        fx = self.fixture(n_jobs=1)
        remote = FakeRemote(poses=3)
        job_id = fx.job_ids[0]
        with fx.collector(remote) as collector:
            collector.run_once()
        gdir = os.path.join(fx.job_dir(job_id), "gnina")
        self.assertEqual(
            sorted(os.listdir(gdir)),
            sorted(["out.sdf", "out.log", "results.csv"]
                   + ["out_%d.sdf" % i for i in (1, 2, 3)]
                   + ["complex_%d.pdb" % i for i in (1, 2, 3)]))

    def test_results_csv_has_the_columns_parse_gnina_writes(self):
        """GNINA_Affinity / GNINA_CNN_pose / GNINA_CNN_affinity exist in exactly
        one place in this system: consensus_docker.parse_gnina."""
        fx = self.fixture(n_jobs=1)
        remote = FakeRemote(poses=2)
        with fx.collector(remote) as collector:
            collector.run_once()
        path = os.path.join(fx.job_dir(fx.job_ids[0]), "gnina", "results.csv")
        with open(path) as handle:
            lines = [ln.strip() for ln in handle if ln.strip()]
        self.assertEqual(lines[0], GNINA_RESULTS_HEADER)
        self.assertEqual(len(lines), 3)

    def test_the_result_satisfies_batch_docks_completion_check(self):
        """check_tool_completion only asks for results.csv with a CSV header.

        Replicated rather than imported: importing batch_dock configures logging
        and writes docking_automation.log at import (spec D7).
        """
        fx = self.fixture(n_jobs=1)
        remote = FakeRemote()
        with fx.collector(remote) as collector:
            collector.run_once()
        path = os.path.join(fx.job_dir(fx.job_ids[0]), "gnina", "results.csv")
        self.assertTrue(os.path.exists(path))
        with open(path) as handle:
            self.assertIn(",", handle.readline().strip())

    def test_no_other_tool_directory_is_touched(self):
        fx = self.fixture(n_jobs=1)
        job_dir = fx.job_dir(fx.job_ids[0])
        for tool in ("smina", "gold", "ledock"):
            with open(os.path.join(job_dir, tool, "results.csv"), "w") as handle:
                handle.write("keep me\n")
        final = os.path.join(job_dir, "final_results.csv")
        with open(final, "w") as handle:
            handle.write("consensus\n")
        remote = FakeRemote()
        with fx.collector(remote) as collector:
            collector.run_once()
        for tool in ("smina", "gold", "ledock"):
            with open(os.path.join(job_dir, tool, "results.csv")) as handle:
                self.assertEqual(handle.read(), "keep me\n")
        with open(final) as handle:
            self.assertEqual(handle.read(), "consensus\n")


# --- the single-GPU invariant --------------------------------------------------


class TestSingleGpuInvariant(CollectorTestCase):
    def test_never_more_than_one_job_outstanding_on_the_remote(self):
        """50 jobs, 200 passes: at most one SUBMITTING/SUBMITTED/RUNNING, ever.

        This is the invariant the whole design exists to hold. The fake remote
        emulates the mkdir mutex, so a pump that tried to double-book would show
        up as a mutex conflict even if the record states happened to look fine.
        """
        fx = self.fixture(n_jobs=50)
        remote = FakeRemote(polls_before_done=1)
        seen_max = 0
        with fx.collector(remote) as collector:
            for _ in range(200):
                collector.run_once()
                live = [r for r in fx.queue.read_all()
                        if r.state in ("SUBMITTING", "SUBMITTED", "RUNNING")]
                seen_max = max(seen_max, len(live))
                self.assertLessEqual(
                    len(live), 1,
                    "two jobs on one GPU: %s" % [r.job_id for r in live])
                if all(r.state in gq.TERMINAL_STATES for r in fx.queue.read_all()):
                    break
        self.assertEqual(remote.mutex_conflicts, [],
                         "the pump contended for the remote GPU mutex; its own "
                         "active_remote() guard should have prevented that")
        states = [r.state for r in fx.queue.read_all()]
        self.assertEqual(states.count("COLLECTED"), 50)
        self.assertEqual(seen_max, 1)

    def test_prestage_never_breaks_the_invariant(self):
        """--prestage is the only path that would deliberately touch the remote
        while a job is running. Whether the queue lets it upload and be told
        busy, or refuses it outright (which is what this queue does, so the
        option switches itself off), the second job must stay PENDING and the
        remote must still have exactly one job outstanding."""
        fx = self.fixture(n_jobs=2)
        remote = FakeRemote(polls_before_done=5)
        with fx.collector(remote, prestage=True) as collector:
            collector.run_once()
            disabled = not collector.prestage
        live = [r for r in fx.queue.read_all()
                if r.state in ("SUBMITTING", "SUBMITTED", "RUNNING")]
        self.assertEqual(len(live), 1)
        self.assertEqual(fx.state(fx.job_ids[1]), "PENDING")
        self.assertTrue(disabled or remote.mutex_conflicts,
                        "prestage must either take effect or turn itself off, "
                        "never log a refusal on every poll for a whole campaign")

    def test_two_orphaned_remote_results_collect_in_parallel(self):
        """Only reconciliation of a previous run leaves more than one job in
        REMOTE_DONE. When it does, the collect workers must handle them
        independently -- each writes only its own job directory."""
        fx = self.fixture(n_jobs=2)
        remote = FakeRemote()
        for i, job_id in enumerate(fx.job_ids):
            remote.finish(job_id, 0)
            fx.queue.transition(job_id, "SUBMITTING", why="test setup",
                                remote={"dir": "/home/onur/gnina_jobs/gnina_%d" % i})
            fx.queue.transition(job_id, "SUBMITTED", why="test setup")
            fx.queue.transition(job_id, "REMOTE_DONE", why="test setup")
        with fx.collector(remote, collect_workers=2) as collector:
            collector.run_once()
        for job_id in fx.job_ids:
            self.assertEqual(fx.state(job_id), "COLLECTED")
            self.assertTrue(gq.outputs_complete(fx.job_dir(job_id)))
        self.assertEqual(sorted(c[1] for c in remote.calls if c[0] == "collect"),
                         sorted(fx.job_ids))

    def test_a_busy_gpu_does_not_consume_an_attempt(self):
        fx = self.fixture(n_jobs=1)
        remote = FakeRemote()
        job_id = fx.job_ids[0]
        remote.submit_outcome[job_id] = "busy"
        with fx.collector(remote) as collector:
            counts = collector.run_once()
        record = fx.queue.read(job_id)
        self.assertEqual(record.state, "PENDING")
        self.assertEqual(record.attempts, 0)
        self.assertEqual(counts["busy"], 1)
        self.assertTrue(record.next_attempt_at, "busy must back off")


# --- idempotence and crash safety ----------------------------------------------


class TestIdempotentCollection(CollectorTestCase):
    def test_collecting_the_same_job_twice_is_harmless(self):
        fx = self.fixture(n_jobs=1)
        remote = FakeRemote()
        job_id = fx.job_ids[0]
        with fx.collector(remote) as collector:
            collector.run_once()
        gdir = os.path.join(fx.job_dir(job_id), "gnina")
        first = dict((n, open(os.path.join(gdir, n), "rb").read())
                     for n in sorted(os.listdir(gdir)))
        calls_after_first = len(remote.calls)

        # Force a second collection of an already-COLLECTED job the only way the
        # design allows: reset the record and let the queue re-derive it.
        with fx.collector(remote) as collector:
            counts = collector.run_once()
        self.assertEqual(fx.state(job_id), "COLLECTED")
        self.assertEqual(counts["submitted"], 0,
                         "a finished job must never be re-docked")
        self.assertEqual(len(remote.calls), calls_after_first)
        second = dict((n, open(os.path.join(gdir, n), "rb").read())
                      for n in sorted(os.listdir(gdir)))
        self.assertEqual(first, second)

    def test_stale_output_from_a_previous_attempt_is_purged_not_reused(self):
        """The most dangerous bug in this design (spec risk 2).

        A ``gnina/`` left over from an earlier attempt at different parameters
        would let the submit phase's split_mol succeed on stale bytes, and
        nothing downstream can tell -- check_tool_completion only asks whether
        results.csv exists and looks like CSV. Both phases purge first, so the
        extras from the previous run must be gone.
        """
        fx = self.fixture(n_jobs=1)
        job_id = fx.job_ids[0]
        gdir = os.path.join(fx.job_dir(job_id), "gnina")
        for i in range(1, 6):
            with open(os.path.join(gdir, "out_%d.sdf" % i), "w") as handle:
                handle.write("STALE pose %d from the previous parameter set\n" % i)
            with open(os.path.join(gdir, "complex_%d.pdb" % i), "w") as handle:
                handle.write("STALE\n")
        with open(os.path.join(gdir, "out.log"), "w") as handle:
            handle.write("STALE LOG\n")

        remote = FakeRemote(poses=2)
        with fx.collector(remote) as collector:
            collector.run_once()

        self.assertEqual(fx.state(job_id), "COLLECTED")
        self.assertEqual(sorted(n for n in os.listdir(gdir) if n.startswith("out_")),
                         ["out_1.sdf", "out_2.sdf"],
                         "poses 3-5 of the previous attempt survived")
        self.assertEqual(sorted(n for n in os.listdir(gdir)
                                if n.startswith("complex_")),
                         ["complex_1.pdb", "complex_2.pdb"])
        with open(os.path.join(gdir, "out.log")) as handle:
            self.assertNotIn("STALE", handle.read())
        # And the purge is observable at both phases, not just the last one.
        self.assertEqual([present for phase, _, present in remote.purged_before],
                         [[], []],
                         "gnina/ must be empty when each phase starts")


class TestCrashMidCollect(CollectorTestCase):
    def test_a_crashed_collect_is_recoverable_and_finishes(self):
        """A pump killed mid-collect leaves COLLECTING and a partial gnina/.

        Reconciliation rule 3 demotes it to REMOTE_DONE (collect is idempotent),
        the next pass re-collects, and the partial output is purged rather than
        mistaken for a result.
        """
        fx = self.fixture(n_jobs=1)
        job_id = fx.job_ids[0]
        remote = FakeRemote()
        remote.finish(job_id, 0)

        # Simulate the crash: a COLLECTING record whose collect pid is dead, and
        # a gnina/ directory containing only the downloaded out.sdf.
        proc = subprocess.Popen(["sh", "-c", "exit 0"])
        proc.wait()
        dead_pid = proc.pid
        record = fx.queue.read(job_id)
        data = record.to_dict()
        data["state"] = "COLLECTING"
        data["remote"] = {"dir": "/home/onur/gnina_jobs/gnina_x", "host": "user@fake"}
        data["collect"] = {"pid": dead_pid, "started_at": "2026-08-13T09:00:00.0Z",
                           "finished_at": None}
        gq.atomic_write_json(os.path.join(fx.queue.jobs_dir, "%s.json" % job_id), data)
        gdir = os.path.join(fx.job_dir(job_id), "gnina")
        with open(os.path.join(gdir, "out.sdf"), "w") as handle:
            handle.write("half a download")

        # A half-written job must not look finished to anyone.
        self.assertFalse(gq.outputs_complete(fx.job_dir(job_id)))

        with fx.collector(remote) as collector:
            collector.run_once()
        self.assertEqual(fx.state(job_id), "COLLECTED")
        self.assertTrue(gq.outputs_complete(fx.job_dir(job_id)))
        self.assertTrue(any(t.startswith("reconcile:%s" % job_id)
                            for t in collector.trace), collector.trace)

    def test_a_pump_that_died_mid_launch_does_not_wedge_the_gpu(self):
        """SUBMITTING is the one active state poll_job will not probe, so a
        record orphaned there holds the single GPU slot and nothing else can be
        submitted. Reconciliation is the only thing that can resolve it, and it
        must be pulled forward rather than waited for."""
        fx = self.fixture(n_jobs=1)
        remote = FakeRemote()
        with fx.collector(remote) as collector:
            collector.run_once()                     # the first reconcile happens here
            self.assertEqual(fx.state(fx.job_ids[0]), "COLLECTED")
            # Now a job appears that another pump crashed mid-launch, holding
            # the one GPU slot with a state nothing polls.
            orphan = fx.add_job("DB00104__P30874__SSTR2__F1__cav1")
            fx.queue.transition(orphan, "SUBMITTING", why="test: crashed pump",
                                remote={"dir": "/home/onur/gnina_jobs/gnina_orphan"})
            self.assertIsNotNone(fx.queue.active_remote())
            collector.run_once()                     # notices and pulls reconcile in
            self.assertEqual(fx.state(orphan), "SUBMITTING")
            collector.run_once()                     # reconciles, frees the slot
        self.assertEqual(fx.state(orphan), "COLLECTED")

    def test_an_exception_inside_collect_does_not_lose_the_job(self):
        fx = self.fixture(n_jobs=1)
        job_id = fx.job_ids[0]
        remote = FakeRemote()
        remote.raise_on_collect.add(job_id)
        with fx.collector(remote) as collector:
            collector.run_once()
            record = fx.queue.read(job_id)
            # Recorded, not swallowed, and pointed back at the finished remote
            # result rather than at a fresh 150 s of GPU time.
            self.assertEqual(record.state, "REMOTE_DONE")
            self.assertTrue(any(t == "retry:%s->REMOTE_DONE" % job_id
                                for t in collector.trace), collector.trace)
            fx.clock.advance(300)
            collector.run_once()
        self.assertEqual(fx.state(job_id), "COLLECTED")
        # It never went back to the GPU.
        self.assertEqual([c[0] for c in remote.calls].count("submit"), 1)

    def test_a_crash_in_the_queue_itself_is_recorded_against_the_job(self):
        """If collect_job raises rather than returning a FAILED record, the pump
        still has to leave the job visible with a reason. A job that vanishes
        because of an unhandled exception is the failure mode this catches."""
        fx = self.fixture(n_jobs=1)
        job_id = fx.job_ids[0]
        remote = FakeRemote()
        original = gq.collect_job

        def exploding(*args, **kwargs):
            raise MemoryError("rdkit ran out of memory building complex_3.pdb")

        gq.collect_job = exploding
        self.addCleanup(setattr, gq, "collect_job", original)
        with fx.collector(remote) as collector:
            collector.run_once()
            self.assertTrue(any(t.startswith("fail:%s:" % job_id)
                                for t in collector.trace), collector.trace)
        record = fx.queue.read(job_id)
        self.assertIn(record.state, ("FAILED", "REMOTE_DONE"))
        self.assertIsNotNone(record.error)
        self.assertEqual(record.error["code"], "POSTPROCESS_FAILED")
        self.assertIn("MemoryError", record.error["message"])


# --- rejecting bad results ------------------------------------------------------


class TestPartialDownloadIsRejected(CollectorTestCase):
    def test_a_truncated_result_is_failed_not_filed(self):
        """results.csv with a header and no poses is not a result."""
        fx = self.fixture(n_jobs=1)
        job_id = fx.job_ids[0]
        remote = FakeRemote()
        remote.collect_mode[job_id] = "truncated"
        with fx.collector(remote) as collector:
            collector.run_once()
        record = fx.queue.read(job_id)
        self.assertNotEqual(record.state, "COLLECTED")
        self.assertFalse(gq.outputs_complete(fx.job_dir(job_id)))

    def test_a_postprocess_failure_retries_the_collect_and_never_re_docks(self):
        fx = self.fixture(n_jobs=1)
        job_id = fx.job_ids[0]
        remote = FakeRemote()
        remote.collect_mode[job_id] = "truncated"
        with fx.collector(remote) as collector:
            collector.run_once()
            self.assertEqual(fx.state(job_id), "REMOTE_DONE",
                             "a collect-side failure must go back to REMOTE_DONE, "
                             "not PENDING: the GPU's poses are fine")
            fx.clock.advance(300)
            collector.run_once()            # second failure -> give up
        record = fx.queue.read(job_id)
        self.assertEqual(record.state, "ABANDONED")
        self.assertEqual((record.error or {}).get("code"), "POSTPROCESS_FAILED")
        self.assertEqual([c[0] for c in remote.calls].count("submit"), 1,
                         "the job must never have been re-docked")

    def test_an_empty_out_sdf_is_rejected(self):
        fx = self.fixture(n_jobs=1)
        job_id = fx.job_ids[0]
        remote = FakeRemote()
        remote.collect_mode[job_id] = "empty"
        with fx.collector(remote) as collector:
            collector.run_once()
        record = fx.queue.read(job_id)
        self.assertNotEqual(record.state, "COLLECTED")
        self.assertIn((record.error or {}).get("code"),
                      ("EMPTY_OUTPUT", "POSTPROCESS_FAILED"))


class TestFailuresAreRecordedAndSurfaced(CollectorTestCase):
    def test_a_remote_gnina_failure_is_retried_once_then_abandoned_with_its_reason(self):
        fx = self.fixture(n_jobs=1)
        job_id = fx.job_ids[0]
        remote = FakeRemote()
        remote.remote_exit[job_id] = 1          # gnina died on the GPU
        with fx.collector(remote) as collector:
            collector.run_once()
            record = fx.queue.read(job_id)
            self.assertEqual(record.state, "PENDING")
            self.assertEqual(record.attempts, 1,
                             "the retry itself must not consume a second attempt; "
                             "submit_job charges one when it actually launches")
            fx.clock.advance(300)
            collector.run_once()                # fails again -> give up
            summary = collector._write_summary()
        record = fx.queue.read(job_id)
        self.assertEqual(record.state, "ABANDONED")
        self.assertEqual((record.error or {}).get("code"), "REMOTE_GNINA_FAILED")
        # Surfaced, not merely counted.
        with open(summary["summary_path"]) as handle:
            written = json.load(handle)
        self.assertIn(job_id, [j["job_id"] for j in written["abandoned_jobs"]])
        self.assertEqual([j["code"] for j in written["abandoned_jobs"]],
                         ["REMOTE_GNINA_FAILED"])

    def test_a_transient_failure_is_retried_off_its_own_budget(self):
        """A dropped connection must send the job back to the GPU without
        spending one of its three docking attempts."""
        fx = self.fixture(n_jobs=1)
        job_id = fx.job_ids[0]
        fx.queue.transition(job_id, "SUBMITTING", why="test setup")
        fx.queue.transition(
            job_id, "FAILED", why="test setup",
            error={"code": "SSH_TRANSIENT", "message": "Connection reset by peer",
                   "phase": "submit", "retriable": True,
                   "at": "2026-08-13T09:00:00.000000Z"})
        before = fx.queue.read(job_id)
        remote = FakeRemote()
        with fx.collector(remote) as collector:
            collector.run_once()
            record = fx.queue.read(job_id)
            self.assertEqual(record.state, "PENDING")
            self.assertEqual(record.transient_attempts,
                             before.transient_attempts + 1)
            self.assertEqual(record.attempts, before.attempts,
                             "a network drop is not a docking attempt")
            fx.clock.advance(300)
            collector.run_once()
        self.assertEqual(fx.state(job_id), "COLLECTED")

    def test_a_missing_input_is_abandoned_immediately(self):
        fx = self.fixture(n_jobs=1)
        job_id = fx.job_ids[0]
        os.remove(os.path.join(fx.job_dir(job_id), "input", "ligand.sdf"))
        remote = FakeRemote()
        with fx.collector(remote) as collector:
            collector.run_once()
        record = fx.queue.read(job_id)
        self.assertEqual(record.state, "ABANDONED")
        self.assertEqual((record.error or {}).get("code"), "MISSING_INPUT")
        self.assertEqual(remote.calls, [], "nothing should have reached the GPU")

    def test_one_bad_job_does_not_stop_the_others(self):
        fx = self.fixture(n_jobs=3)
        bad = fx.job_ids[1]
        remote = FakeRemote()
        remote.remote_exit[bad] = 1
        with fx.collector(remote) as collector:
            for _ in range(12):
                collector.run_once()
                fx.clock.advance(300)
        states = dict((r.job_id, r.state) for r in fx.queue.read_all())
        self.assertEqual(states[fx.job_ids[0]], "COLLECTED")
        self.assertEqual(states[fx.job_ids[2]], "COLLECTED")
        self.assertEqual(states[bad], "ABANDONED")


# --- retry policy (pure) --------------------------------------------------------


class TestRetryPolicy(unittest.TestCase):
    """§11 as a table. Pure function, so every row is one assertion."""

    def plan(self, code, attempts=1, transient=0, retries=0, max_attempts=3,
             phase="poll", retriable=None):
        error = {"code": code, "phase": phase}
        if retriable is not None:
            error["retriable"] = retriable
        return gc.plan_retry(error, attempts, transient, retries, max_attempts)

    def test_permanent_errors_are_never_retried(self):
        for code in ("MISSING_INPUT", "SUBMIT_WROTE_RESULTS", "PUMP_UNAVAILABLE",
                     "CAPS_FATAL"):
            with self.subTest(code=code):
                self.assertEqual(self.plan(code).to_state, "ABANDONED")

    def test_a_remote_gnina_failure_gets_exactly_one_retry(self):
        self.assertEqual(self.plan("REMOTE_GNINA_FAILED", attempts=1).to_state,
                         "PENDING")
        self.assertEqual(self.plan("REMOTE_GNINA_FAILED", attempts=2).to_state,
                         "ABANDONED")

    def test_a_postprocess_failure_goes_back_to_remote_done_not_pending(self):
        plan = self.plan("POSTPROCESS_FAILED", phase="collect", retries=0)
        self.assertEqual(plan.to_state, "REMOTE_DONE")
        self.assertIsNone(plan.counter)
        self.assertEqual(self.plan("POSTPROCESS_FAILED", phase="collect",
                                   retries=1).to_state, "ABANDONED")

    def test_no_plan_ever_charges_a_docking_attempt(self):
        """submit_job owns ``attempts``; charging it here too would abandon a job
        one retry early and is exactly the kind of off-by-one that only shows up
        after the campaign has thrown work away."""
        for code in sorted(gc.RETRIABLE_CODES):
            with self.subTest(code=code):
                self.assertIn(self.plan(code, attempts=1, retries=0).counter,
                              (None, "transient"))

    def test_a_stall_re_docks_up_to_max_attempts(self):
        self.assertEqual(self.plan("STALLED", attempts=1).to_state, "PENDING")
        self.assertEqual(self.plan("STALLED", attempts=2).to_state, "PENDING")
        self.assertEqual(self.plan("STALLED", attempts=3).to_state, "ABANDONED")

    def test_transient_failures_have_their_own_budget(self):
        """A thousand dropped connections must not burn a job's three chances."""
        plan = self.plan("SSH_TRANSIENT", attempts=3, transient=0)
        self.assertEqual(plan.to_state, "PENDING",
                         "attempts is exhausted but the transient budget is not")
        self.assertEqual(plan.counter, "transient")
        self.assertEqual(self.plan("SSH_TRANSIENT", transient=10).to_state,
                         "ABANDONED")

    def test_a_transient_failure_during_collect_does_not_re_dock(self):
        self.assertEqual(self.plan("SSH_TRANSIENT", phase="collect").to_state,
                         "REMOTE_DONE")

    def test_an_unknown_code_is_abandoned_unless_it_says_it_is_retriable(self):
        self.assertEqual(self.plan("SOMETHING_NEW").to_state, "ABANDONED")
        self.assertEqual(self.plan("SOMETHING_NEW", retriable=True).to_state,
                         "PENDING")

    def test_backoff_grows(self):
        delays = [self.plan("STALLED", attempts=n, max_attempts=9).delay
                  for n in (0, 1, 2, 3)]
        self.assertEqual(delays, sorted(delays))
        self.assertLess(delays[0], delays[-1])


# --- locking, stopping, exit codes ----------------------------------------------


class TestPumpLock(CollectorTestCase):
    def test_a_second_pump_refuses_rather_than_waits(self):
        fx = self.fixture(n_jobs=1)
        remote = FakeRemote()
        first = fx.collector(remote)
        try:
            first.run_once()
            second = fx.collector(remote)
            with self.assertRaises(gc.PumpLocked):
                second.run_once()
            second.close()
        finally:
            first.close()

    def test_the_lock_is_released_on_close(self):
        fx = self.fixture(n_jobs=1)
        remote = FakeRemote()
        first = fx.collector(remote)
        first.run_once()
        first.close()
        second = fx.collector(remote)
        try:
            second.run_once()       # must not raise
        finally:
            second.close()

    def test_status_takes_no_lock(self):
        fx = self.fixture(n_jobs=1)
        remote = FakeRemote()
        with fx.collector(remote) as collector:
            collector.run_once()
            code = gc.main(["--run-dir", fx.run_dir, "--status"])
        self.assertEqual(code, gc.EXIT_OK)

    def test_cli_returns_five_when_another_pump_holds_the_lock(self):
        fx = self.fixture(n_jobs=1)
        remote = FakeRemote()
        with fx.collector(remote) as collector:
            collector.run_once()
            self.assertEqual(gc.main(["--run-dir", fx.run_dir, "--once"]),
                             gc.EXIT_LOCKED)


class TestStopping(CollectorTestCase):
    def test_the_stop_file_stops_the_loop_and_writes_a_summary(self):
        fx = self.fixture(n_jobs=3)
        remote = FakeRemote()
        open(os.path.join(fx.queue.root, "STOP"), "w").close()
        with fx.collector(remote) as collector:
            summary = collector.run_forever()
        self.assertTrue(collector.stopped)
        self.assertEqual(collector.stop_reason, "STOP file")
        self.assertTrue(os.path.exists(os.path.join(fx.queue.root, "summary.json")))
        self.assertIn("summary_path", summary)

    def test_stop_cli_writes_the_stop_file(self):
        fx = self.fixture(n_jobs=1)
        self.assertEqual(gc.main(["--run-dir", fx.run_dir, "--stop"]), gc.EXIT_OK)
        self.assertTrue(os.path.exists(os.path.join(fx.queue.root, "STOP")))

    def test_sigterm_asks_the_pump_to_stop_without_killing_the_remote(self):
        fx = self.fixture(n_jobs=1)
        remote = FakeRemote()
        previous = signal.getsignal(signal.SIGTERM)
        with fx.collector(remote) as collector:
            try:
                gc._install_signal_handlers(collector, gc.LOG)
                os.kill(os.getpid(), signal.SIGTERM)
            finally:
                signal.signal(signal.SIGTERM, previous)
            self.assertTrue(collector.stopped)
            self.assertIn("signal", collector.stop_reason)

    def test_a_stopped_run_leaves_the_remote_job_alone(self):
        fx = self.fixture(n_jobs=2)
        remote = FakeRemote(polls_before_done=5)
        with fx.collector(remote) as collector:
            collector.run_once()
            collector.request_stop("test")
            collector.run_once()
        live = [j for j, s in remote.jobs.items() if s["state"] == "running"]
        self.assertEqual(len(live), 1, "the detached remote job must keep running")


class TestExitCodes(CollectorTestCase):
    def test_zero_when_everything_collected(self):
        fx = self.fixture(n_jobs=2, clock=AutoClock())
        remote = FakeRemote()
        self.install_cli_seams(fx, remote)
        self.assertEqual(gc.main(["--run-dir", fx.run_dir, "--daemon",
                                  "--poll", "0"]), gc.EXIT_OK)

    def test_three_when_a_job_was_abandoned(self):
        fx = self.fixture(n_jobs=2, clock=AutoClock())
        remote = FakeRemote()
        remote.remote_exit[fx.job_ids[1]] = 1
        self.install_cli_seams(fx, remote)
        self.assertEqual(gc.main(["--run-dir", fx.run_dir, "--daemon",
                                  "--poll", "0"]), gc.EXIT_ABANDONED)
        # Drained, but one job has no gnina result and is named as such.
        with open(os.path.join(fx.queue.root, "summary.json")) as handle:
            summary = json.load(handle)
        self.assertEqual([j["job_id"] for j in summary["abandoned_jobs"]],
                         [fx.job_ids[1]])

    def test_four_when_work_is_left_outstanding(self):
        fx = self.fixture(n_jobs=3, clock=AutoClock())
        remote = FakeRemote(polls_before_done=99)
        self.install_cli_seams(fx, remote)
        self.assertEqual(gc.main(["--run-dir", fx.run_dir, "--once"]),
                         gc.EXIT_STOPPED)

    def test_two_when_there_is_no_queue(self):
        """A mistyped --run-dir must not create an empty queue and then report a
        clean, instantaneous, entirely fictional drain."""
        tmp = tempfile.mkdtemp(prefix="gnina_collect_empty_")
        try:
            self.assertEqual(gc.main(["--run-dir", tmp, "--status"]),
                             gc.EXIT_USAGE)
            self.assertEqual(gc.main(["--run-dir", tmp, "--once"]),
                             gc.EXIT_USAGE)
            self.assertFalse(os.path.exists(os.path.join(tmp, "gnina_queue")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# --- operator commands ----------------------------------------------------------


class TestOperatorCommands(CollectorTestCase):
    def test_requeue_resets_an_abandoned_job(self):
        fx = self.fixture(n_jobs=1)
        job_id = fx.job_ids[0]
        remote = FakeRemote()
        remote.remote_exit[job_id] = 1
        with fx.collector(remote) as collector:
            collector.run_once()
            fx.clock.advance(300)
            collector.run_once()
        self.assertEqual(fx.state(job_id), "ABANDONED")
        self.assertEqual(gc.main(["--run-dir", fx.run_dir, "--requeue", job_id]),
                         gc.EXIT_OK)
        record = fx.queue.read(job_id)
        self.assertEqual(record.state, "PENDING")
        self.assertEqual(record.attempts, 0)
        self.assertIsNone(record.error)
        self.assertEqual(record.history[-1]["from"], "ABANDONED")

    def test_requeue_refuses_a_live_job(self):
        fx = self.fixture(n_jobs=1)
        self.assertEqual(
            gc.main(["--run-dir", fx.run_dir, "--requeue", fx.job_ids[0]]),
            gc.EXIT_USAGE)
        self.assertEqual(fx.state(fx.job_ids[0]), "PENDING")

    def test_requeue_refuses_an_unknown_job(self):
        fx = self.fixture(n_jobs=1)
        self.assertEqual(gc.main(["--run-dir", fx.run_dir, "--requeue", "nope"]),
                         gc.EXIT_USAGE)

    def test_reconcile_only_does_not_submit_anything(self):
        fx = self.fixture(n_jobs=2)
        remote = FakeRemote()
        self.install_cli_seams(fx, remote)
        gc.main(["--run-dir", fx.run_dir, "--reconcile-only"])
        self.assertEqual(remote.calls, [])
        self.assertTrue(os.path.exists(os.path.join(fx.queue.root, "summary.json")))

    def test_max_jobs_bounds_what_one_run_submits(self):
        fx = self.fixture(n_jobs=4)
        remote = FakeRemote()
        with fx.collector(remote, max_jobs=2) as collector:
            collector.run_forever()
        submits = [c for c in remote.calls if c[0] == "submit"]
        self.assertEqual(len(submits), 2)
        self.assertEqual(collector.stop_reason, "max-jobs")


class TestParamsDiscipline(CollectorTestCase):
    def test_a_deadline_change_is_refused_without_the_flag(self):
        fx = self.fixture(n_jobs=1)
        remote = FakeRemote()
        with fx.collector(remote) as collector:
            collector.set_params_if_changed({"job_deadline_seconds": 60.0})
            self.assertEqual(fx.queue.get_params()["job_deadline_seconds"], 1800.0)
            collector.set_params_if_changed({"job_deadline_seconds": 60.0},
                                            allow_change=True)
            self.assertEqual(fx.queue.get_params()["job_deadline_seconds"], 60.0)

    def test_submitting_without_params_is_refused_loudly(self):
        fx = self.fixture(n_jobs=1)
        os.remove(fx.queue.params_path)
        remote = FakeRemote()
        with fx.collector(remote) as collector:
            with self.assertRaises(gc.CollectorError):
                collector.run_once()
        self.assertEqual(remote.calls, [])


#: The measured capacity of the campaign's GPU box, as the probe reports it.
#: A literal, never a read: the spec forbids re-probing the host, and a test that
#: needed a GPU would not be hermetic.
MEASURED_PROBE = "\n".join([
    "cores=24",
    "mem_total_kb=32505856",
    "mem_avail_kb=30000000",
    "gpu=NVIDIA GeForce RTX 3060 Ti, 8192, 0, 8.6",
    "docker=1",
    "image=1",
    "native=0",
    "disk_free_kb=200000000",
    "uptime=100000",
])


class TestCapabilityGate(CollectorTestCase):
    """--caps-file validates against a saved probe and never opens a connection."""

    def caps_file(self, fixture):
        try:
            from dock_selectivity import remote_caps
        except ImportError as exc:                       # pragma: no cover
            self.skipTest("remote_caps not available: %s" % exc)
        caps = remote_caps.parse_probe_output(MEASURED_PROBE, "user@fake", 50000)
        path = os.path.join(fixture.tmp, "caps.json")
        remote_caps.save_caps(caps, path)
        return path

    def test_the_intended_campaign_is_allowed_to_run(self):
        fx = self.fixture(n_jobs=1)
        remote = FakeRemote()
        self.install_cli_seams(fx, remote)
        code = gc.main(["--run-dir", fx.run_dir, "--once",
                        "--caps-file", self.caps_file(fx)])
        self.assertNotEqual(code, gc.EXIT_CAPS_FATAL)
        self.assertEqual(fx.state(fx.job_ids[0]), "COLLECTED")

    def test_an_impossible_exhaustiveness_stops_the_pump_before_it_submits(self):
        fx = self.fixture(n_jobs=1)
        params = fx.queue.get_params()
        params["exhaustiveness"] = 512
        gq.atomic_write_json(fx.queue.params_path, params)
        remote = FakeRemote()
        self.install_cli_seams(fx, remote)
        code = gc.main(["--run-dir", fx.run_dir, "--once",
                        "--caps-file", self.caps_file(fx)])
        self.assertEqual(code, gc.EXIT_CAPS_FATAL)
        self.assertEqual(remote.calls, [],
                         "nothing may be submitted after a fatal caps check")


class TestClockAgreement(CollectorTestCase):
    """The pump and the queue must measure ``next_attempt_at`` on the same clock.

    The field is persisted and then compared by ``GninaQueue.next_pending``, so a
    pump using ``time.monotonic`` (seconds since boot) while the queue uses
    ``time.time`` (seconds since the epoch) does not fail loudly -- it writes a
    deadline that is permanently in the past, every backoff silently becomes
    zero, and a job burns all three of its attempts inside one pass against a
    remote that is still broken.
    """

    def test_a_backoff_written_by_the_pump_is_in_the_future_for_the_queue(self):
        fx = self.fixture(n_jobs=1)
        job_id = fx.job_ids[0]
        fx.queue.transition(job_id, "SUBMITTING", why="test setup")
        fx.queue.transition(
            job_id, "FAILED", why="test setup",
            error={"code": "STALLED", "message": "no exit_code",
                   "phase": "poll", "retriable": True,
                   "at": "2026-08-13T09:00:00.000000Z"})
        remote = FakeRemote()
        # clock=None on purpose: this is the production default path.
        with fx.collector(remote, clock=None) as collector:
            collector._ensure_lock()
            collector._apply_retries(dict((k, 0) for k in
                                          ("retried", "abandoned")))
        record = fx.queue.read(job_id)
        self.assertEqual(record.state, "PENDING")
        self.assertIsNotNone(record.next_attempt_at)

        plain = gq.GninaQueue(fx.run_dir, create=False)   # the queue's own clock
        self.assertIsNone(
            plain.next_pending(),
            "the pump's backoff reads as already expired to the queue: the two "
            "are not measuring time the same way")
        self.assertIsNotNone(
            plain.next_pending(now=record.next_attempt_at + 1.0),
            "and it must still become eligible once the backoff has passed")


class TestProcessHygiene(unittest.TestCase):
    def test_the_collector_does_not_import_batch_dock_or_the_science_stack(self):
        """batch_dock configures logging and writes docking_automation.log at
        import (spec D7); pandas/rdkit/psutil would put the science stack in a
        process whose whole job is to start fast and wait on subprocesses."""
        code = ("import sys; sys.path.insert(0, %r); import gnina_collect; "
                "print(','.join(sorted(m for m in "
                "('batch_dock', 'pandas', 'rdkit', 'psutil', 'numpy') "
                "if m in sys.modules)))" % REPO)
        proc = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "",
                         "gnina_collect pulled in a module it must not")

    def test_repeated_main_calls_do_not_stack_log_handlers(self):
        """A duplicated line per handler is how a campaign log becomes unreadable.

        Logging now belongs to dock_selectivity.runtime, which owns tagged
        handlers on the ``gnina_collect`` logger and never touches root; both
        halves of that are asserted here.
        """
        step_logger = logging.getLogger(gc.STEP_NAME)
        before = list(step_logger.handlers)
        root = logging.getLogger()
        root_before = list(root.handlers)
        tmp = tempfile.mkdtemp(prefix="gnina_collect_log_")
        try:
            # There is no queue under --run-dir, so the pump refuses -- but only
            # after the step has configured its logging, which is the part under
            # test. (--status takes the query path and configures nothing.)
            for _ in range(3):
                self.assertEqual(gc.main(["--run-dir", tmp, "--once", "--quiet"]),
                                 gc.EXIT_USAGE)
            log_path = os.path.join(tmp, "steps", gc.STEP_NAME,
                                    gc.STEP_NAME + ".log")
            ours = [h for h in step_logger.handlers
                    if getattr(h, "baseFilename", None) == log_path]
            self.assertEqual(len(ours), 1)
            self.assertEqual(list(root.handlers), root_before,
                             "the pump must not touch the root logger")
        finally:
            # The same handler object is also borrowed by the dock_selectivity
            # package logger; closing it while it is still attached there would
            # make the next library record try to reopen a deleted temp file.
            library = logging.getLogger("dock_selectivity")
            for handler in list(step_logger.handlers):
                if handler not in before:
                    step_logger.removeHandler(handler)
                    library.removeHandler(handler)
                    handler.close()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_cli_help_works_without_a_run_directory(self):
        proc = subprocess.run([sys.executable,
                               os.path.join(REPO, "gnina_collect.py"), "--help"],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--run-dir", proc.stdout)


class TestSummaryReporting(CollectorTestCase):
    def test_the_summary_names_outstanding_and_abandoned_jobs(self):
        fx = self.fixture(n_jobs=3)
        remote = FakeRemote(polls_before_done=99)
        remote.remote_exit[fx.job_ids[0]] = 0
        with fx.collector(remote) as collector:
            collector.run_once()
            summary = collector._write_summary()
        self.assertIn("state_counts", summary)
        self.assertGreater(summary["state_counts"]["outstanding"], 0)
        self.assertEqual(summary["collector"]["submitted"], 1)


if __name__ == "__main__":
    unittest.main()
