"""Adversarial sync and crash tests for the decoupled remote-gnina pipeline.

This file exists to BREAK the protocol, not to demonstrate it.  Everything here
is written from the position that the interesting bugs in a distributed pump are
the ones that leave a plausible-looking result behind: a job that quietly
disappears, a ``gnina/`` directory assembled from two different runs, a queue
that reports "drained" while work is still outstanding.  None of those raise, so
none of them are caught by a test that only walks the happy path.

Scope and ownership
-------------------

Nothing in this file writes to any module.  Each defect it found was reported
rather than patched -- four implementers own the four modules and a fix belongs
in one place, theirs -- as a test asserting the CORRECT behaviour, marked
``@unittest.expectedFailure`` so that the suite stayed green while it was open
and turned the test into an *unexpected success* (which fails the run) the
moment it was fixed.  ALL SIX were fixed by their owners while this file was
being written, so no marker remains: every ``R`` below is now a live regression
guard, and each one's docstring keeps the failure it was written against,
because that is the thing a future change would silently reintroduce.

Hermetic by construction: no SSH, no GPU host, no ``/media/onur/Elements``, no
docking engine, no ``consensus_docker.py``.  Every remote interaction goes
through the injected ``runner`` / ``prober`` / ``clock`` seams, and every byte
written lands in a ``tempfile.TemporaryDirectory()``.  This module deliberately
defines its own fixtures rather than importing another test module's, so that a
change over there cannot silently weaken the adversarial cases here.

Defect index -- all six found here, all six since fixed by their owners; the
tests named are what stops them coming back (details in each docstring)
-----------------------------------------------------------------------------

R1  an unparseable record file dropped out of the queue entirely: the barrier
    reported a clean drain and the pump exited 0 while the job was neither
    collected nor abandoned.  Fixed in two places, ``GninaQueue.summary`` and
    the pump's own accounting.                     -- TestUnreadableRecords
R2  ``reconcile`` aborted on the first record it could not classify, so one
    poisoned record disabled crash recovery for every other job in the
    campaign -- permanently, since the pump retried the same failing pass.
                                                   -- TestUnknownStateValue
R3  the "outputs are complete on disk" predicate could not see a half-written
    ``gnina/``, so a crash during ``parse_gnina`` or during the R2 purge was
    promoted to COLLECTED with a subset of poses.  ``outputs_complete`` now
    requires complexes, split poses and score rows to agree whenever any
    complex is present.  Two residuals remain, recorded in the class.
                                                   -- TestHalfWrittenOutputs
R4  ``reconcile`` rule 2 rewrote a truncated record without its input paths,
    which destroyed the one repair path there was (a re-run of batch_dock) and
    guaranteed the job would be ABANDONED with MISSING_INPUT.
                                                   -- TestTruncatedRecordRepair
R5  nothing bounded how long a record could stay COLLECTING, so a collect whose
    pid was alive but idle was outstanding forever. -- TestCollectingDeadline
R6  a record whose filename and ``job_id`` disagreed was readable but
    unwritable: permanently outstanding and unreachable by any code.
                                                   -- TestRecordIdentity

Still open, and reported as characterisations rather than as defects, because
each is either working as designed or a design decision this file should not
make:

C1  two writers that bypass ``pump.lock`` both run a whole collect phase into
    one job directory -- the lock is the only defence, not a belt on braces.
                       -- TestTwoCollectors.test_a_bypassed_lock_...
C2  ``next_attempt_at`` is set as a re-dock backoff and never cleared, so it
    later defers that job's *download* too.
                       -- TestOperationalTraps.test_a_docking_backoff_...
C3  the barrier reads records only, never the job directory, so a job finished
    on disk but unrecorded is waited out and then named as missing.
                       -- TestBarrier.test_the_barrier_never_sees_...
C4  a leftover ``STOP`` file turns a fresh pump into a silent no-op.
                       -- TestOperationalTraps.test_a_leftover_stop_file_...
C5  a record from a newer ``schema_version`` is acted on with today's rules,
    ungated and unwarned.
                       -- TestRecordIdentity.test_a_newer_schema_version_...
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from dock_selectivity import gnina_queue as gq        # noqa: E402
import gnina_collect                                  # noqa: E402


# ---------------------------------------------------------------------------
# noise control
# ---------------------------------------------------------------------------

_SILENCERS = []


def setUpModule():
    """Silence the modules under test.

    Almost every case here provokes a WARNING or a CRITICAL on purpose -- that
    loudness is the product, and several tests assert on it with ``assertLogs``,
    which installs its own handler regardless.  Without this, logging's
    last-resort writer buries the suite's own output in the failures we are
    deliberately causing.
    """
    for name in (gq.__name__, "gnina_collect"):
        logger = logging.getLogger(name)
        handler = logging.NullHandler()
        logger.addHandler(handler)
        logger.propagate = False
        _SILENCERS.append((logger, handler))


def tearDownModule():
    for logger, handler in _SILENCERS:
        logger.removeHandler(handler)
        logger.propagate = True
    del _SILENCERS[:]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

JOB_A = "DB00035__P30518__AVPR2__F1__cav1"
JOB_B = "DB00035__P30518__AVPR2__F1__cav2"
JOB_C = "DB01234__Q9Y5N1__HRH3__F1__cav1"

REMOTE_DIR = "/home/onur/gnina_jobs/gnina_ab12cd34ef567890"

PARAMS = {
    "schema_version": 1,
    "cpu": 24,
    "exhaustiveness": 32,
    "num_modes": None,
    "cutoff_value": -7.0,
    "consensus_docker_script": "/abs/consensus_docker.py",
    "gnina_path": "/abs/gnina_remote.py",
    "image": "gnina/gnina:v1.3.3",
    "host": "onur@example.invalid",
    "port": 50000,
    "key": None,
    "native": None,
    "remote_root": "gnina_jobs",
    "job_deadline_seconds": 1800.0,
    "max_attempts": 3,
}


def make_run(tmp):
    """A run directory shaped the way batch_dock lays one out."""
    run_dir = os.path.join(tmp, "run")
    os.makedirs(os.path.join(run_dir, "docking_results"), exist_ok=True)
    return run_dir


def make_job_dir(run_dir, job_id):
    """One job directory with its inputs, and empty per-tool subdirectories."""
    job_dir = os.path.join(run_dir, "docking_results", job_id)
    for sub in ("input", "smina", "gnina", "gold", "ledock"):
        os.makedirs(os.path.join(job_dir, sub), exist_ok=True)
    for name, body in (("receptor.pdb", "ATOM      1  N   ALA A   1\n"),
                       ("ligand.sdf", "ligand\n$$$$\n"),
                       ("pocket.pdb", "ATOM      1  N   ALA A   1\n")):
        with open(os.path.join(job_dir, "input", name), "w") as handle:
            handle.write(body)
    return job_dir


def write_gnina_outputs(job_dir, n_poses=3, tag="runA", n_rows=None,
                        n_complexes=None):
    """Exactly what consensus_docker leaves in ``gnina/`` after a good run.

    ``tag`` is written into every file so a test can prove which run a surviving
    file came from -- that is the only way to tell "one consistent result" from
    "a mixture of two", which is the corruption these tests are hunting.
    """
    gnina = os.path.join(job_dir, "gnina")
    os.makedirs(gnina, exist_ok=True)
    rows = n_poses if n_rows is None else n_rows
    complexes = n_poses if n_complexes is None else n_complexes
    with open(os.path.join(gnina, "out.sdf"), "w") as handle:
        handle.write("%s pose\n$$$$\n" % tag * max(1, n_poses))
    with open(os.path.join(gnina, "out.log"), "w") as handle:
        handle.write("# %s\nmode |   affinity\n" % tag)
    for i in range(1, n_poses + 1):
        with open(os.path.join(gnina, "out_%d.sdf" % i), "w") as handle:
            handle.write("%s pose %d\n$$$$\n" % (tag, i))
    for i in range(1, complexes + 1):
        with open(os.path.join(gnina, "complex_%d.pdb" % i), "w") as handle:
            handle.write("REMARK %s\nATOM      1  N   ALA A   1\n" % tag)
    with open(os.path.join(gnina, "results.csv"), "w") as handle:
        handle.write("Pose,GNINA_Affinity,GNINA_CNN_pose,GNINA_CNN_affinity\n")
        for i in range(1, rows + 1):
            handle.write("%d,-8.%d,0.9,6.1,%s\n" % (i, i, tag))


def job_inputs(job_dir):
    return {
        "receptor_pdb": os.path.join(job_dir, "input", "receptor.pdb"),
        "ligand_sdf": os.path.join(job_dir, "input", "ligand.sdf"),
        "pocket_pdb": os.path.join(job_dir, "input", "pocket.pdb"),
        "receptor_pdbqt": None,
        "skip_pdb_to_pdbqt": False,
    }


def force_state(queue, job_id, state, **fields):
    """Put a record into a state without going through ``transition``.

    A fixture escape hatch, and only that: the whole point of the transition
    table is that production code cannot do this.  It is how a crashed process's
    leftovers are reproduced without crashing a process.
    """
    record = queue.read(job_id)
    record.state = state
    for key, value in fields.items():
        if key in ("inputs", "remote", "collect", "outputs", "params"):
            getattr(record, key).update(value)
        else:
            setattr(record, key, value)
    gq.atomic_write_json(queue.record_path(job_id), record.to_dict())
    queue._cache.pop(job_id, None)
    return queue.read(job_id)


def digest_tree(root):
    """{relative path: sha256} for every file under *root*.

    Byte-level, because "the collector is idempotent" has to mean the bytes are
    identical, not merely that the state string did not change.
    """
    out = {}
    for base, _dirs, names in os.walk(root):
        for name in sorted(names):
            path = os.path.join(base, name)
            with open(path, "rb") as handle:
                out[os.path.relpath(path, root)] = hashlib.sha256(
                    handle.read()).hexdigest()
    return out


def launched_ticket(remote_dir=REMOTE_DIR):
    return {"schema_version": 1, "mode": "submit", "outcome": "launched",
            "host": PARAMS["host"], "port": PARAMS["port"],
            "remote_dir": remote_dir, "job_key": "ab12cd34ef567890",
            "image": PARAMS["image"], "lock": "acquired", "exit_code": None,
            "error": None}


def collected_ticket():
    return {"schema_version": 1, "mode": "collect", "outcome": "collected",
            "host": PARAMS["host"], "remote_dir": REMOTE_DIR,
            "exit_code": 0, "error": None}


class FakeGpu(object):
    """Stands in for gnina_remote's submit/collect modes AND the GPU box.

    It is handed exactly what the real runner is handed -- argv, environment,
    timeout, log path, cwd -- and writes its ticket to the path the queue put in
    ``GNINA_REMOTE_TICKET``, so the ticket handshake is exercised for real.  It
    counts launches and collects per job, which is how "docked twice" and "lost"
    are actually measured rather than assumed.
    """

    def __init__(self, job_dirs, n_poses=3, tag="runA", collect_hook=None,
                 submit_hook=None):
        self.job_dirs = dict(job_dirs)
        self.n_poses = n_poses
        self.tag = tag
        self.collect_hook = collect_hook
        self.submit_hook = submit_hook
        self.launches = {}
        self.collects = {}
        self.modes = []

    def _job_id(self, cwd):
        return os.path.basename(os.path.normpath(cwd))

    def __call__(self, cmd, env, timeout, log_path, cwd):
        job_id = self._job_id(cwd)
        mode = env.get("GNINA_REMOTE_MODE")
        self.modes.append((job_id, mode))
        ticket = env.get("GNINA_REMOTE_TICKET")
        os.makedirs(os.path.dirname(ticket), exist_ok=True)
        if mode == "submit":
            self.launches[job_id] = self.launches.get(job_id, 0) + 1
            if self.submit_hook is not None:
                out = self.submit_hook(self, job_id, env)
                if out is not None:
                    return out
            with open(ticket, "w") as handle:
                json.dump(launched_ticket(REMOTE_DIR + job_id), handle)
            # Exit 3 is the design's "launched, ticket written" code; the queue
            # switches on the ticket, never on this.
            return 3, "", ""
        self.collects[job_id] = self.collects.get(job_id, 0) + 1
        if self.collect_hook is not None:
            out = self.collect_hook(self, job_id, env)
            if out is not None:
                return out
        write_gnina_outputs(self.job_dirs[job_id], self.n_poses, tag=self.tag)
        with open(ticket, "w") as handle:
            json.dump(collected_ticket(), handle)
        return 0, "", ""

    def total_launches(self):
        return sum(self.launches.values())

    def total_collects(self):
        return sum(self.collects.values())


def prober_for(answers):
    """prober(record) -> (state, exit_code) from a job_id -> answer mapping."""
    def prober(record):
        answer = answers.get(record.job_id, ("LAUNCHED", None))
        if isinstance(answer, Exception):
            raise answer
        return answer
    return prober


class TickingClock(object):
    """A clock that advances by *step* on every read.

    The barrier reads the clock several times per iteration, so a fixed clock
    makes a timeout unreachable and a real clock makes the test a race.  This
    makes elapsed time a deterministic function of how much work was done.
    """

    def __init__(self, step=1.0, start=0.0):
        self.now = float(start)
        self.step = float(step)

    def __call__(self):
        self.now += self.step
        return self.now


class RaceCase(unittest.TestCase):
    """A temp run directory, a queue with frozen params, and no stray env."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self._saved_queue_dir = os.environ.pop("GNINA_QUEUE_DIR", None)
        self.run_dir = make_run(self.tmp)
        self.queue = gq.GninaQueue(self.run_dir)
        self.queue.set_params(PARAMS)
        self.job_dirs = {}
        self._collectors = []

    def tearDown(self):
        for collector in self._collectors:
            try:
                collector.close()
            except Exception:                       # pragma: no cover
                pass
        if self._saved_queue_dir is not None:
            os.environ["GNINA_QUEUE_DIR"] = self._saved_queue_dir
        self._tmp.cleanup()

    # -- helpers ---------------------------------------------------------

    def add_job(self, job_id=JOB_A, outputs=False, n_poses=3, tag="runA"):
        job_dir = make_job_dir(self.run_dir, job_id)
        self.job_dirs[job_id] = job_dir
        if outputs:
            write_gnina_outputs(job_dir, n_poses, tag=tag)
        self.queue.enqueue(job_dir, job_inputs(job_dir))
        return job_dir

    def collector(self, **kwargs):
        """A Collector that is always closed in tearDown, lock and all."""
        kwargs.setdefault("runner", FakeGpu(self.job_dirs))
        kwargs.setdefault("prober", prober_for({}))
        collector = gnina_collect.Collector(self.run_dir, **kwargs)
        self._collectors.append(collector)
        return collector

    def drain(self, collector, limit=40):
        """Run passes until nothing is outstanding, or *limit* is reached."""
        for _ in range(limit):
            collector.run_once()
            if all(r.state in gq.TERMINAL_STATES
                   for r in self.queue.read_all()):
                return True
        return False

    def states(self):
        return dict((r.job_id, r.state) for r in self.queue.read_all())


# ---------------------------------------------------------------------------
# 1. two collectors over one queue
# ---------------------------------------------------------------------------

class TestTwoCollectors(RaceCase):
    """Two pumps on one queue would double-book the single GPU.

    The local ``pump.lock`` is the only thing standing between a second
    ``batch_dock`` run (or a hand-started collector, or the barrier's restart
    callback firing on a pump that was merely slow to start) and two processes
    submitting to the same GPU.  These tests hold it to that.
    """

    def test_second_collector_is_refused_and_never_waits(self):
        self.add_job()
        first = self.collector()
        first._ensure_lock()
        second = self.collector()
        with self.assertRaises(gnina_collect.PumpLocked):
            second._ensure_lock()

    def test_second_collector_process_exits_5(self):
        """Exit 5 is "someone else is pumping", not an error to retry on."""
        self.add_job()
        first = self.collector()
        first._ensure_lock()
        with restored_logging():
            code = gnina_collect.main(
                ["--run-dir", self.run_dir, "--once", "--quiet"])
        self.assertEqual(code, gnina_collect.EXIT_LOCKED)

    def test_status_is_readable_while_a_pump_holds_the_lock(self):
        """``--status`` must take no lock: an operator checking on a campaign
        must never be able to disturb it, and must never be blocked by it."""
        self.add_job()
        first = self.collector()
        first._ensure_lock()
        with restored_logging(), restored_stdio():
            code = gnina_collect.main(
                ["--run-dir", self.run_dir, "--status", "--quiet"])
        self.assertEqual(code, gnina_collect.EXIT_OK)
        self.assertFalse(self.queue.pump_lock_is_free(),
                         "--status must not have released the pump lock")

    def test_two_collectors_never_submit_the_same_job(self):
        """Even driven alternately, only the lock holder ever launches."""
        job_dir = self.add_job()
        gpu = FakeGpu({JOB_A: job_dir})
        first = self.collector(runner=gpu, prober=prober_for({JOB_A: ("RUNNING", None)}))
        second = self.collector(runner=gpu, prober=prober_for({JOB_A: ("RUNNING", None)}))
        first.run_once()
        for _ in range(3):
            with self.assertRaises(gnina_collect.PumpLocked):
                second.run_once()
            first.run_once()
        self.assertEqual(gpu.launches.get(JOB_A), 1,
                         "the job must be launched exactly once: %r" % gpu.launches)

    def test_a_bypassed_lock_really_does_double_collect(self):
        """CHARACTERISATION -- the pump lock is load-bearing, not an optimisation.

        ``collect_job`` guards itself with ``if record.state != "REMOTE_DONE"``
        and then with ``transition(..., "COLLECTING")``.  Both are
        read-modify-write, so two writers that both read REMOTE_DONE before
        either writes both pass, both run the ENTIRE collect phase into the same
        ``<job>/gnina/`` -- each purging the other's output first -- and only
        the final transition notices, by which time both consensus_docker
        processes have already written.

        The window is real but narrow, so this test widens it deliberately: one
        queue object's ``_write`` is made to pause between the validation and
        the write.  That changes the timing, not the logic; nothing here would
        be safe if the two writers happened to interleave that way on their own.

        None of this is a bug in ``collect_job`` -- the design says one pump.
        It is here to show that the pump lock is the ONLY defence, so anything
        that weakens it (a mount without ``flock``, an operator starting a
        second pump "just to collect the stragglers") corrupts a job directory
        rather than merely wasting time.
        """
        job_dir = self.add_job()
        force_state(self.queue, JOB_A, "REMOTE_DONE",
                    remote={"dir": REMOTE_DIR, "exit_code": 0})
        other = gq.GninaQueue(self.run_dir)
        phases = []
        outcomes = {}
        # Released only once BOTH writers have validated REMOTE_DONE -> COLLECTING
        # against the file and are about to write it.  That is the window; the
        # barrier holds it open instead of hoping to land in it.
        both_validated = threading.Barrier(2)

        def gated(queue):
            original = queue._write
            state = {"gated": False}

            def gated_write(record):
                if record.state == "COLLECTING" and not state["gated"]:
                    state["gated"] = True
                    both_validated.wait(30)
                return original(record)

            queue._write = gated_write

        gated(self.queue)
        gated(other)

        def runner(cmd, env, timeout, log_path, cwd):
            phases.append(env.get("GNINA_REMOTE_MODE"))
            write_gnina_outputs(job_dir, 3)
            with open(env["GNINA_REMOTE_TICKET"], "w") as handle:
                json.dump(collected_ticket(), handle)
            return 0, "", ""

        def work(name, queue):
            try:
                outcomes[name] = gq.collect_job(
                    queue, queue.read(JOB_A), runner=runner).state
            except gq.IllegalTransition:
                outcomes[name] = "IllegalTransition"
            except Exception as exc:            # pragma: no cover
                outcomes[name] = type(exc).__name__

        threads = [threading.Thread(target=work, args=("a", self.queue)),
                   threading.Thread(target=work, args=("b", other))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        for thread in threads:
            self.assertFalse(thread.is_alive(), "a collect thread hung")

        self.assertEqual(len(phases), 2,
                         "both writers ran a full collect phase into one job "
                         "directory: %r" % (phases,))
        self.assertIn("COLLECTED", outcomes.values(), "%r" % outcomes)
        self.assertEqual(self.queue.read(JOB_A).state, "COLLECTED",
                         "and the record cannot tell that it happened twice")


# ---------------------------------------------------------------------------
# 2. the collector writing a job directory a local stage is reading
# ---------------------------------------------------------------------------

class TestConcurrentJobDirectoryAccess(RaceCase):
    """The pump writes ``<job>/gnina/`` while batch_dock's stages touch the
    same job directory.  Stage 5 is the dangerous reader: consensus_docker
    admits a tool to the consensus with

        if complex_files and os.path.exists(results_file)

    so what matters is not whether it can see a partially written directory but
    whether it can ever see one where the ``complex_*.pdb`` and the
    ``results.csv`` come from *different runs*.
    """

    def test_purge_never_touches_another_tool_or_final_results(self):
        job_dir = self.add_job(outputs=True, n_poses=5)
        for tool in ("smina", "gold", "ledock"):
            tool_dir = os.path.join(job_dir, tool)
            with open(os.path.join(tool_dir, "results.csv"), "w") as handle:
                handle.write("Pose,Affinity\n1,-7.0\n")
            with open(os.path.join(tool_dir, "complex_1.pdb"), "w") as handle:
                handle.write("ATOM      1  N   ALA A   1\n")
        with open(os.path.join(job_dir, "final_results.csv"), "w") as handle:
            handle.write("Pose,Consensus\n1,0.4\n")
        before = digest_tree(job_dir)

        removed = gq.purge_gnina_outputs(job_dir)

        self.assertGreater(removed, 0)
        after = digest_tree(job_dir)
        for path, sha in before.items():
            if path.startswith("gnina" + os.sep):
                continue
            self.assertEqual(after.get(path), sha,
                             "purge modified %s, which belongs to another stage"
                             % path)
        self.assertEqual(os.listdir(os.path.join(job_dir, "gnina")), [],
                         "purge must empty gnina/ and nothing else")
        self.assertTrue(os.path.isdir(os.path.join(job_dir, "gnina")),
                        "the directory itself must survive")

    def test_stage5_never_sees_poses_and_scores_from_different_runs(self):
        """A stage-5 reader polling during a re-collect sees all or nothing.

        The safety here is an accident of alphabetical order --
        ``purge_gnina_outputs`` deletes ``complex_*.pdb`` before ``results.csv``,
        and a collect writes ``results.csv`` last -- so the admission gate is
        only ever true for a self-consistent directory.  That is worth pinning:
        renaming one artefact, or purging in a different order, would silently
        start feeding stage 5 a mixture.
        """
        job_dir = self.add_job(outputs=True, n_poses=6, tag="runA")
        gnina_dir = os.path.join(job_dir, "gnina")
        mixtures = []
        stop = threading.Event()

        def stage5_reader():
            while not stop.is_set():
                try:
                    names = os.listdir(gnina_dir)
                except OSError:                  # pragma: no cover
                    continue
                complexes = [n for n in names if n.startswith("complex_")]
                if not (complexes and "results.csv" in names):
                    continue                     # gnina simply not admitted
                tags = set()
                try:
                    for name in complexes:
                        with open(os.path.join(gnina_dir, name)) as handle:
                            tags.add(handle.readline().split()[-1])
                    with open(os.path.join(gnina_dir, "results.csv")) as handle:
                        handle.readline()
                        row = handle.readline().strip()
                    if row:
                        tags.add(row.split(",")[-1])
                except (OSError, IndexError):     # a file vanished mid-read
                    continue
                if len(tags) > 1:
                    mixtures.append(sorted(tags))

        reader = threading.Thread(target=stage5_reader)
        reader.start()
        try:
            for tag in ("runB", "runC", "runD"):
                gq.purge_gnina_outputs(job_dir)
                time.sleep(0.005)
                write_gnina_outputs(job_dir, 6, tag=tag)
                time.sleep(0.005)
        finally:
            stop.set()
            reader.join(10)

        self.assertEqual(mixtures, [],
                         "stage 5 could admit gnina with poses and scores from "
                         "different runs: %r" % (mixtures[:3],))

    def test_a_reader_never_sees_a_torn_record(self):
        """Records are read lock-free while the pump rewrites them constantly."""
        self.add_job()
        seen = []
        problems = []
        stop = threading.Event()

        def reader():
            queue = gq.GninaQueue(self.run_dir, create=False)
            while not stop.is_set():
                for record in queue.read_all():
                    seen.append(record.state)
                    if record.state not in gq.ALL_STATES:
                        problems.append(("state", record.state))
                    if record.job_id != JOB_A:
                        problems.append(("job_id", record.job_id))
                    if not record.inputs.get("receptor_pdb"):
                        problems.append(("inputs", record.job_id))

        thread = threading.Thread(target=reader)
        thread.start()
        try:
            for _ in range(120):
                self.queue.transition(JOB_A, "SUBMITTING", why="race")
                self.queue.transition(JOB_A, "PENDING", why="race")
        finally:
            stop.set()
            thread.join(10)

        self.assertEqual(problems[:3], [], "torn or partial record observed")
        self.assertGreater(len(seen), 0, "the reader never ran")

    def test_two_writers_on_one_record_lose_an_update(self):
        """CHARACTERISATION -- the single-writer contract is real and unenforced.

        Each write is atomic, so no reader ever sees a mixture, but
        ``transition`` and ``update_fields`` are read-modify-write: two holders
        of a record both write a whole document and the later one wins
        wholesale.  Documented in ``GninaQueue``'s docstring; pinned here so
        that nobody later reads "atomic writes" as "safe to write from two
        places".  The pump lock is what makes it true in production.
        """
        self.add_job()
        other = gq.GninaQueue(self.run_dir)
        stale = other.read(JOB_A)                # B's snapshot, taken first

        self.queue.update_fields(JOB_A, transient_attempts=7)

        other._apply_fields(stale, {"attempts": 4})
        stale.updated_at = gq._utcnow_iso()
        other._write(stale)

        final = self.queue.read(JOB_A)
        self.assertEqual(final.attempts, 4)
        self.assertEqual(final.transient_attempts, 0,
                         "if this is 7 the store has grown a merge and this "
                         "characterisation is out of date")


# ---------------------------------------------------------------------------
# 3. crash at every interesting point
# ---------------------------------------------------------------------------

class TestCrashRecoveryPoints(RaceCase):
    """Kill the pump at each step of the protocol and restart it.

    For every point the same two questions are asked: is the job LOST (never
    finishes, or finishes with nothing on disk), and is it DUPLICATED (docked
    twice on a GPU that costs ~2.5 minutes a job).  ``FakeGpu`` counts launches
    and collects per job, so both are measured rather than inferred from the
    state string.
    """

    def _restart(self, gpu, answers):
        """A brand-new pump over the same queue, as a restart really is."""
        collector = self.collector(runner=gpu, prober=prober_for(answers))
        return collector

    def test_crash_after_enqueue_before_submit(self):
        job_dir = self.add_job()
        gpu = FakeGpu({JOB_A: job_dir})
        collector = self._restart(gpu, {JOB_A: ("DONE", 0)})
        self.assertTrue(self.drain(collector))
        self.assertEqual(self.states()[JOB_A], "COLLECTED")
        self.assertEqual(gpu.launches.get(JOB_A), 1)
        self.assertEqual(gpu.collects.get(JOB_A), 1)

    def test_crash_after_submitting_intent_before_launch(self):
        """SUBMITTING with no remote directory: nothing nameable was launched.

        Reconciliation must send it back to PENDING -- and the content-addressed
        remote directory is what makes a re-launch cheap if the crash happened
        to be *after* the real launch.
        """
        job_dir = self.add_job()
        force_state(self.queue, JOB_A, "SUBMITTING")
        gpu = FakeGpu({JOB_A: job_dir})
        collector = self._restart(gpu, {JOB_A: ("DONE", 0)})
        self.assertTrue(self.drain(collector))
        self.assertEqual(self.states()[JOB_A], "COLLECTED")
        self.assertEqual(gpu.launches.get(JOB_A), 1,
                         "exactly one launch after the crash")
        self.assertTrue(gq.outputs_complete(job_dir))

    def test_a_crash_between_the_launch_and_the_ticket_does_not_re_dock(self):
        """The compensating control for the one hole in the state machine.

        Only ``gnina_remote`` can resolve the content address, so SUBMITTING is
        necessarily written with ``remote.dir`` null: a crash between the real
        launch and the ticket write leaves a record that reconciliation can only
        send back to PENDING.  What stops the GPU doing the work twice is the
        shim's resume probe, which answers ``already_done`` for that content
        address -- so the queue must treat that ticket as "collect", never as
        "launched".
        """
        job_dir = self.add_job()
        force_state(self.queue, JOB_A, "SUBMITTING")
        self.queue.reconcile(prober=prober_for({}))
        self.assertEqual(self.queue.read(JOB_A).state, "PENDING")

        def already_done(gpu, job_id, env):
            with open(env["GNINA_REMOTE_TICKET"], "w") as handle:
                json.dump(dict(launched_ticket(REMOTE_DIR + job_id),
                               outcome="already_done", exit_code=0), handle)
            return 3, "", ""

        gpu = FakeGpu({JOB_A: job_dir}, submit_hook=already_done)
        collector = self._restart(gpu, {JOB_A: ("DONE", 0)})
        self.assertTrue(self.drain(collector))

        self.assertEqual(self.states()[JOB_A], "COLLECTED")
        self.assertEqual(gpu.collects.get(JOB_A), 1)
        self.assertEqual(self.queue.read(JOB_A).remote["dir"],
                         REMOTE_DIR + JOB_A,
                         "the resumed remote directory must be recorded so the "
                         "collect pins GNINA_REMOTE_JOB_DIR to it")

    def test_crash_after_launch_before_the_exit_marker(self):
        """The remote is still working.  A restart must WAIT, never re-dock."""
        job_dir = self.add_job()
        force_state(self.queue, JOB_A, "SUBMITTED",
                    remote={"dir": REMOTE_DIR, "launched_at": gq._utcnow_iso()})
        gpu = FakeGpu({JOB_A: job_dir})
        answers = {JOB_A: ("RUNNING", None)}
        collector = self._restart(gpu, answers)
        collector.run_once()
        self.assertEqual(self.states()[JOB_A], "RUNNING")
        self.assertEqual(gpu.total_launches(), 0,
                         "a running remote job must never be re-launched")

        answers[JOB_A] = ("DONE", 0)             # the marker lands
        self.assertTrue(self.drain(collector))
        self.assertEqual(self.states()[JOB_A], "COLLECTED")
        self.assertEqual(gpu.total_launches(), 0)
        self.assertEqual(gpu.collects.get(JOB_A), 1)

    def test_crash_after_the_marker_before_the_download(self):
        job_dir = self.add_job()
        force_state(self.queue, JOB_A, "SUBMITTED",
                    remote={"dir": REMOTE_DIR, "launched_at": gq._utcnow_iso()})
        gpu = FakeGpu({JOB_A: job_dir})
        collector = self._restart(gpu, {JOB_A: ("DONE", 0)})
        self.assertTrue(self.drain(collector))
        self.assertEqual(self.states()[JOB_A], "COLLECTED")
        self.assertEqual(gpu.total_launches(), 0,
                         "the GPU already produced this result")
        self.assertEqual(gpu.collects.get(JOB_A), 1)

    def test_crash_mid_download_purges_the_partial_file(self):
        """A half-downloaded out.sdf must never be mistaken for a result."""
        job_dir = self.add_job()
        gnina_dir = os.path.join(job_dir, "gnina")
        with open(os.path.join(gnina_dir, "out.sdf"), "w") as handle:
            handle.write("half a pose, no terminator")
        force_state(self.queue, JOB_A, "COLLECTING",
                    remote={"dir": REMOTE_DIR, "exit_code": 0},
                    collect={"pid": 2 ** 22 - 1, "host": gq._hostname(),
                             "started_at": gq._utcnow_iso()})
        self.assertFalse(gq.outputs_complete(job_dir),
                         "a partial download must not look complete")

        gpu = FakeGpu({JOB_A: job_dir})
        collector = self._restart(gpu, {JOB_A: ("DONE", 0)})
        self.assertTrue(self.drain(collector))

        self.assertEqual(self.states()[JOB_A], "COLLECTED")
        self.assertEqual(gpu.total_launches(), 0, "no re-dock: the poses exist")
        self.assertEqual(gpu.collects.get(JOB_A), 1)
        with open(os.path.join(gnina_dir, "out.sdf")) as handle:
            self.assertNotIn("half a pose", handle.read(),
                             "the partial download survived the re-collect")

    def test_crash_after_download_before_the_state_update(self):
        """Outputs on disk outrank the record: no second collect, no rewrite."""
        job_dir = self.add_job(outputs=True, n_poses=4, tag="the-real-result")
        force_state(self.queue, JOB_A, "COLLECTING",
                    remote={"dir": REMOTE_DIR, "exit_code": 0},
                    collect={"pid": 2 ** 22 - 1, "host": gq._hostname(),
                             "started_at": gq._utcnow_iso()})
        before = digest_tree(job_dir)

        gpu = FakeGpu({JOB_A: job_dir}, tag="a-second-run")
        collector = self._restart(gpu, {JOB_A: ("DONE", 0)})
        self.assertTrue(self.drain(collector))

        self.assertEqual(self.states()[JOB_A], "COLLECTED")
        self.assertEqual(gpu.total_collects(), 0,
                         "a finished job must not be collected again")
        self.assertEqual(digest_tree(job_dir), before,
                         "the finished job directory was rewritten")

    def test_crash_between_claiming_a_record_and_writing_it(self):
        """The zero-byte remains of a crashed ``enqueue`` are re-derived.

        Only the state is recovered here; see ``TestTruncatedRecordRepair`` for
        what happens to the inputs, which is where this stops being benign.
        """
        job_dir = self.add_job()
        with open(self.queue.record_path(JOB_A), "w") as handle:
            handle.write("")
        self.queue._cache.clear()

        self.assertEqual(self.queue.unreadable_job_ids(), [JOB_A])
        changed = self.queue.reconcile(prober=prober_for({}))
        self.assertEqual(changed.get(JOB_A), ("(unreadable)", "PENDING"))
        self.assertEqual(self.queue.read(JOB_A).state, "PENDING")

    def test_a_crash_never_leaves_a_temp_file_behind(self):
        """A ``.tmp`` left in jobs/ would be picked up by nothing and cleaned by
        nobody, and on the external drive it would accumulate for a campaign."""
        self.add_job()
        path = self.queue.record_path(JOB_A)
        broken = {"job_id": JOB_A, "unserialisable": object()}
        with self.assertRaises(TypeError):
            gq.atomic_write_json(path, broken)
        leftovers = [n for n in os.listdir(self.queue.jobs_dir) if ".tmp." in n]
        self.assertEqual(leftovers, [])
        self.assertEqual(self.queue.read(JOB_A).state, "PENDING",
                         "the failed write must not have damaged the record")


# ---------------------------------------------------------------------------
# 4. idempotency
# ---------------------------------------------------------------------------

class TestIdempotency(RaceCase):
    """Running the pump again over a finished campaign must cost nothing.

    This is the property that makes "just re-run batch_dock" a safe answer to a
    partial campaign.  If it does not hold, the recommended recovery procedure
    is itself the thing that re-docks thousands of finished jobs.
    """

    def test_three_more_runs_leave_the_job_directory_byte_identical(self):
        job_dir = self.add_job()
        gpu = FakeGpu({JOB_A: job_dir})
        collector = self.collector(runner=gpu, prober=prober_for({JOB_A: ("DONE", 0)}))
        self.assertTrue(self.drain(collector))
        self.assertEqual(self.states()[JOB_A], "COLLECTED")
        snapshot = digest_tree(job_dir)

        for run in range(3):
            collector.close()
            collector = self.collector(
                runner=gpu, prober=prober_for({JOB_A: ("DONE", 0)}))
            collector.run_once()
            self.assertEqual(self.states()[JOB_A], "COLLECTED")
            self.assertEqual(digest_tree(job_dir), snapshot,
                             "run %d rewrote the job directory" % (run + 1))
        self.assertEqual(gpu.total_launches(), 1)
        self.assertEqual(gpu.total_collects(), 1)

    def test_reconcile_is_idempotent_over_a_finished_queue(self):
        job_dir = self.add_job(outputs=True)
        force_state(self.queue, JOB_A, "COLLECTED")
        with open(self.queue.record_path(JOB_A)) as handle:
            record_before = json.load(handle)
        for _ in range(3):
            self.assertEqual(self.queue.reconcile(prober=prober_for({})), {})
        with open(self.queue.record_path(JOB_A)) as handle:
            self.assertEqual(json.load(handle), record_before,
                             "reconcile rewrote a terminal record")
        self.assertEqual(digest_tree(job_dir), digest_tree(job_dir))

    def test_re_enqueueing_a_live_campaign_disturbs_nothing(self):
        """A second ``batch_dock`` run enqueues the same jobs while a pump works."""
        job_dir = self.add_job()
        other = self.add_job(JOB_B)
        force_state(self.queue, JOB_A, "SUBMITTED",
                    remote={"dir": REMOTE_DIR}, attempts=1)
        write_gnina_outputs(other, 3)
        force_state(self.queue, JOB_B, "COLLECTED")

        stats = self.queue.enqueue_many([
            dict(current_outfolder=job_dir, **job_inputs(job_dir)),
            dict(current_outfolder=other, **job_inputs(other)),
        ], PARAMS)

        self.assertEqual(stats["already_present"], 2)
        self.assertEqual(stats["enqueued"], 0)
        self.assertEqual(self.queue.read(JOB_A).state, "SUBMITTED",
                         "a live remote job was reset by a re-run")
        self.assertEqual(self.queue.read(JOB_A).attempts, 1)
        self.assertEqual(self.queue.read(JOB_B).state, "COLLECTED")

    def test_two_processes_enqueueing_at_once_produce_one_record(self):
        """Two batch_dock runs started together race on the same record file.

        ``enqueue`` claims the name with ``O_EXCL`` and the loser reads what the
        winner wrote, so the outcome must be one record and no leftover temp
        files -- a ``.tmp`` in ``jobs/`` is read by nothing and cleaned by
        nobody.
        """
        job_dir = make_job_dir(self.run_dir, JOB_C)
        self.job_dirs[JOB_C] = job_dir
        queues = [gq.GninaQueue(self.run_dir) for _ in range(6)]
        start = threading.Barrier(len(queues))
        results = []

        def enqueue(queue):
            start.wait(30)
            results.append(queue.enqueue(job_dir, job_inputs(job_dir)).state)

        threads = [threading.Thread(target=enqueue, args=(q,)) for q in queues]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)

        self.assertEqual(results, ["PENDING"] * len(queues))
        names = os.listdir(self.queue.jobs_dir)
        self.assertEqual(sorted(names), ["%s.json" % JOB_C])
        record = self.queue.read(JOB_C)
        self.assertTrue(record.inputs["receptor_pdb"],
                        "the surviving record must be the complete one")


# ---------------------------------------------------------------------------
# 5. stale locks
# ---------------------------------------------------------------------------

class TestStaleLocks(RaceCase):
    """A crashed pump must not strand the campaign; a live one must not be
    evicted.  Those two requirements pull in opposite directions, so both
    directions get a test."""

    def test_flock_is_released_when_the_holder_is_killed(self):
        script = textwrap.dedent("""
            import sys, time
            sys.path.insert(0, %r)
            from dock_selectivity import gnina_queue as gq
            queue = gq.GninaQueue(%r)
            lock = queue.acquire_pump_lock()
            print("ACQUIRED" if lock else "DENIED", flush=True)
            time.sleep(300)
        """ % (REPO, self.run_dir))
        proc = subprocess.Popen([sys.executable, "-c", script],
                                stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(proc.stdout.readline().strip(), "ACQUIRED")
            self.assertFalse(self.queue.pump_lock_is_free())
            self.assertIsNone(self.queue.acquire_pump_lock(),
                              "a live pump's lock was stolen")
        finally:
            proc.kill()
            proc.wait(timeout=30)
            proc.stdout.close()
        deadline = time.time() + 10
        while time.time() < deadline and not self.queue.pump_lock_is_free():
            time.sleep(0.05)
        self.assertTrue(self.queue.pump_lock_is_free(),
                        "a SIGKILLed pump left the campaign deadlocked")
        lock = self.queue.acquire_pump_lock()
        self.assertIsNotNone(lock)
        lock.release()

    def test_fallback_lock_from_a_dead_pid_is_broken_loudly(self):
        """Without flock the only evidence is the recorded pid, and breaking a
        lock is the one act that could put two submitters on one GPU -- so it
        must never be silent."""
        saved = gq.fcntl
        gq.fcntl = None
        try:
            with open(self.queue.pump_lock_path, "w") as handle:
                json.dump({"pid": 2 ** 22 - 1, "host": gq._hostname(),
                           "started_at": "2020-01-01T00:00:00.000000Z",
                           "argv": ["old-pump"]}, handle)
            with self.assertLogs(gq.__name__, level="WARNING") as caught:
                lock = self.queue.acquire_pump_lock()
            self.assertIsNotNone(lock)
            lock.release()
            self.assertTrue(any("stale" in line for line in caught.output),
                            "breaking a lock must say so: %r" % caught.output)
        finally:
            gq.fcntl = saved

    def test_fallback_lock_from_a_live_pid_is_never_broken(self):
        """A live same-host pid keeps the lock forever, with no age escape.

        That is the right call for a real pump and the wrong one for a recycled
        pid -- see this module's ``risks`` note.  Pinned so the rule is a
        decision rather than an accident.
        """
        saved = gq.fcntl
        gq.fcntl = None
        try:
            with open(self.queue.pump_lock_path, "w") as handle:
                json.dump({"pid": os.getpid(), "host": gq._hostname(),
                           "started_at": "1970-01-01T00:00:00.000000Z",
                           "argv": ["not-really-a-pump"]}, handle)
            old = time.time() - (gq.PUMP_LOCK_STALE_SECONDS * 10)
            os.utime(self.queue.pump_lock_path, (old, old))
            self.assertIsNone(self.queue.acquire_pump_lock())
            self.assertFalse(self.queue.pump_lock_is_free())
        finally:
            gq.fcntl = saved

    def test_a_stale_remote_job_does_not_hold_the_gpu_slot(self):
        """A SUBMITTED record whose remote scratch is gone must free the GPU."""
        job_dir = self.add_job()
        self.add_job(JOB_B)
        force_state(self.queue, JOB_A, "SUBMITTED",
                    remote={"dir": REMOTE_DIR, "launched_at": gq._utcnow_iso()})
        self.assertIsNotNone(self.queue.active_remote())

        changed = self.queue.reconcile(prober=prober_for({JOB_A: ("GONE", None)}))

        self.assertEqual(changed.get(JOB_A), ("SUBMITTED", "PENDING"))
        self.assertIsNone(self.queue.active_remote(),
                          "a vanished remote job still occupied the one GPU slot")
        self.assertIsNotNone(self.queue.next_pending())

    def test_a_silent_remote_job_is_stalled_rather_than_waited_on_forever(self):
        """LAUNCHED forever past the deadline is a failure, not patience."""
        self.add_job()
        long_ago = "2020-01-01T00:00:00.000000Z"
        force_state(self.queue, JOB_A, "SUBMITTED",
                    remote={"dir": REMOTE_DIR, "launched_at": long_ago})
        self.queue.reconcile(prober=prober_for({JOB_A: ("LAUNCHED", None)}))
        record = self.queue.read(JOB_A)
        self.assertEqual(record.state, "FAILED")
        self.assertEqual(record.error_code, "STALLED")


# ---------------------------------------------------------------------------
# 6. the state store under corruption
# ---------------------------------------------------------------------------

class TestUnreadableRecords(RaceCase):
    """DEFECT R1 -- an unparseable record file disappears from the campaign.

    ``read_all`` skips a record it cannot parse, and every consumer of the queue
    counted from ``read_all``: ``counts()``, ``summary()``, the barrier's
    outstanding total, ``Collector._drained()`` and the pump's exit code.  The
    job was therefore not merely un-run but un-mentioned -- the barrier reported
    a clean drain, the pump exited 0 ("everything COLLECTED"), and the only
    trace was a filename in ``gnina_queue/jobs/`` that nothing read.

    HALF FIXED while this file was being written.  ``GninaQueue.summary`` now
    folds ``unreadable_job_ids()`` into ``total`` / ``outstanding`` and names
    them under an ``unreadable_job_ids`` key, so the barrier is honest.  The
    pump's own accounting -- ``Collector._drained`` and ``_exit_code``, which
    count from ``read_all`` directly rather than from ``summary`` -- still is
    not; see the remaining expected failure below.
    """

    def _one_lost_one_finished(self):
        lost = self.add_job(JOB_A)
        done = self.add_job(JOB_B, outputs=True)
        force_state(self.queue, JOB_B, "COLLECTED")
        with open(self.queue.record_path(JOB_A), "w") as handle:
            handle.write("")                     # crashed between claim and write
        self.queue._cache.clear()
        return lost, done

    def test_a_lost_record_is_counted_and_named(self):
        """The guard on the fix: a record nobody can parse is still a job.

        ``read_all`` cannot return it -- there is nothing to return -- so the
        accounting has to come from the filename.  What must never happen again
        is a total that quietly drops by one.
        """
        self._one_lost_one_finished()
        self.assertEqual([r.job_id for r in self.queue.read_all()], [JOB_B])
        summary = self.queue.summary()
        self.assertEqual(summary["total"], 2, "the lost job is still a job")
        self.assertEqual(summary["outstanding"], 1)
        self.assertIn(JOB_A, summary["outstanding_job_ids"])
        self.assertIn(JOB_A, summary["unreadable_job_ids"])
        self.assertEqual(self.queue.unreadable_job_ids(), [JOB_A])

    def test_the_barrier_must_not_report_a_clean_drain(self):
        """DEFECT R1, FIXED WHILE THIS FILE WAS BEING WRITTEN -- now the guard.

        A job whose record cannot be read used to be silently forgotten:
        ``outstanding == 0``, ``timed_out is False``, and the job id nowhere in
        the summary -- a campaign that lost a job reporting complete success.
        """
        self._one_lost_one_finished()
        summary = gq.wait_for_completion(
            self.run_dir, timeout=5.0, poll=0.01,
            clock=TickingClock(0.5), sleeper=lambda _s: None)
        self.assertTrue(
            summary["outstanding"] >= 1 or JOB_A in json.dumps(summary),
            "the barrier reported a clean drain while %s was unaccounted for: "
            "%r" % (JOB_A, summary))

    def test_the_pump_must_not_exit_0_with_a_lost_record(self):
        """DEFECT R1, FIXED -- the pump's half, in ``gnina_collect.py``.

        ``Collector._drained`` and ``_exit_code`` counted from
        ``queue.read_all()`` directly, so they could not see a record the store
        reports as unreadable.  The pump logged "queue drained" and exited 0 --
        "everything COLLECTED" -- for a campaign that dropped a job, which is
        precisely the ``$?`` an operator trusts.  Worse, the barrier had already
        been fixed, so the two contradicted each other.

        Both now account for unreadable records: ``_drained`` keeps working
        while reconciliation can still re-derive them (their job directory
        exists, so this cannot spin forever on one that cannot), and
        ``_exit_code`` refuses EXIT_OK whenever any record was unreadable.
        """
        self._one_lost_one_finished()
        collector = self.collector()
        self.assertFalse(collector._drained(),
                         "the pump considers a queue with an unreadable record "
                         "drained")
        self.assertNotEqual(gnina_collect._exit_code(self.queue),
                            gnina_collect.EXIT_OK)

    def test_a_garbage_record_is_at_least_recoverable_by_reconcile(self):
        """The one consumer that does look: reconcile re-derives the record."""
        self.add_job()
        with open(self.queue.record_path(JOB_A), "w") as handle:
            handle.write("{ this is not json")
        self.queue._cache.clear()
        changed = self.queue.reconcile(prober=prober_for({}))
        self.assertEqual(changed.get(JOB_A), ("(unreadable)", "PENDING"))

    def test_a_record_holding_a_json_list_is_treated_as_unreadable(self):
        """Valid JSON that is not an object must not be coerced into a record."""
        self.add_job()
        with open(self.queue.record_path(JOB_A), "w") as handle:
            json.dump(["PENDING"], handle)
        self.queue._cache.clear()
        self.assertIsNone(gq.read_json_retry(self.queue.record_path(JOB_A)))
        self.assertEqual(self.queue.unreadable_job_ids(), [JOB_A])


class TestUnknownStateValue(RaceCase):
    """DEFECT R2 -- one record with an unrecognised ``state`` disables
    reconciliation for the whole queue.

    ``_validate_transition`` correctly refuses to move a record out of a state
    it does not know, but ``reconcile`` runs its rules in a bare loop with no
    per-record guard, so that refusal escapes as a ``QueueError`` and every
    record after the poisoned one (``read_all`` is sorted by job id) is left
    unreconciled.  In the pump, ``_reconcile`` swallows the exception and never
    sets ``_last_reconcile``, so the failure repeats on every pass forever and
    crash recovery is permanently off for that campaign.

    Where an unknown state comes from: an operator editing a record, a
    half-rolled-back deployment, or a queue written by a future version -- the
    schema has no gate against any of those.
    """

    def _poisoned_queue(self):
        """One unknown-state record, and one healthy record only reconcile can fix.

        The healthy job is left COLLECTING with a dead pid: reconcile rule 3 is
        the only thing in the system that can advance it, so if reconciliation
        is broken the job is stuck, and nothing else in the pump can mask that.
        ``JOB_C`` sorts after ``JOB_A``, which is the order ``read_all`` walks.
        """
        poisoned = self.add_job(JOB_A)
        healthy = self.add_job(JOB_C)
        write_gnina_outputs(poisoned, 3)              # makes rule 1 the raiser
        force_state(self.queue, JOB_A, "QUANTUM_SUPERPOSITION")
        force_state(self.queue, JOB_C, "COLLECTING",
                    remote={"dir": REMOTE_DIR, "exit_code": 0},
                    collect={"pid": 2 ** 22 - 1, "host": gq._hostname(),
                             "started_at": gq._utcnow_iso()})
        return poisoned, healthy

    def test_an_unknown_state_is_outstanding_forever(self):
        """CHARACTERISATION, and still true: the record itself is unrescuable.

        R2 stopped one such record from taking the rest of the queue down with
        it, but nothing rescues the record itself, by design -- guessing what an
        unrecognised state meant is how a queue re-docks work it does not
        understand.  It stays outstanding, is named in the summary, and needs an
        operator.  Worth pinning because it is the one job class that no amount
        of waiting resolves: a barrier with one of these in the queue always
        runs to its full timeout.
        """
        self._poisoned_queue()
        record = self.queue.read(JOB_A)
        self.assertTrue(record.is_outstanding)
        self.assertFalse(record.is_terminal)
        self.assertIsNone(self.queue.active_remote(),
                          "an unknown state is invisible to the submitter")
        self.assertIsNone(self.queue.next_pending())
        self.assertIn(JOB_A, self.queue.summary()["outstanding_job_ids"],
                      "outstanding forever, and unreachable by any rule")
        self.assertEqual(self.queue.counts().get("QUANTUM_SUPERPOSITION"), 1,
                         "counts() invents a bucket rather than refusing")

    def test_one_poisoned_record_must_not_block_the_others(self):
        """DEFECT R2, FIXED WHILE THIS FILE WAS BEING WRITTEN -- now the guard.

        ``reconcile`` used to run its rules in a bare loop, so the (correct)
        refusal to transition a record out of a state it does not recognise
        escaped as a ``QueueError`` and every record after the poisoned one --
        ``read_all`` is sorted -- lost crash recovery.  It now catches
        ``QueueError`` per record, logs the job id and carries on.

        The healthy job here is COLLECTING with a dead pid, so reconcile rule 3
        is the ONLY thing in the system that can advance it: if the isolation
        regresses, this fails rather than merely looking slower.
        """
        self._poisoned_queue()
        try:
            self.queue.reconcile(prober=prober_for({JOB_C: ("DONE", 0)}))
        except gq.QueueError as exc:
            self.fail("reconcile aborted the whole queue on one bad record: %s"
                      % exc)
        self.assertEqual(self.queue.read(JOB_C).state, "REMOTE_DONE")

    def test_the_pump_reconciles_the_rest_of_the_queue_around_the_poison(self):
        """FIXED with R2: one bad record no longer disables crash recovery.

        Was: ``_last_reconcile`` is only set on success, so a queue with a
        poisoned record re-attempted (and re-failed) reconciliation on every
        single pass for the life of the campaign, and the healthy job stayed
        stuck because reconcile is the only thing that could rescue it.
        """
        self._poisoned_queue()
        collector = self.collector(prober=prober_for({JOB_C: ("DONE", 0)}))
        collector._pass(wait_for_collects=True)
        self.assertIsNotNone(collector._last_reconcile,
                             "reconciliation completed, so it is rate-limited "
                             "again rather than retried every pass")
        self.assertNotEqual(self.queue.read(JOB_C).state, "COLLECTING",
                            "the healthy job was rescued despite the poisoned "
                            "record beside it")
        self.assertEqual(self.queue.read(JOB_A).state, "QUANTUM_SUPERPOSITION",
                         "and the record nobody can classify is left untouched, "
                         "not guessed at")

    def test_an_unknown_state_is_never_transitioned_out_of(self):
        """The refusal itself is right and must stay: guessing what an unknown
        state meant is how a queue re-docks work it does not understand."""
        self._poisoned_queue()
        with self.assertRaises(gq.QueueError):
            self.queue.transition(JOB_A, "PENDING", why="guess")


class TestHalfWrittenOutputs(RaceCase):
    """DEFECT R3 (FIXED WHILE THIS FILE WAS BEING WRITTEN) -- a half-written
    ``gnina/`` must not be mistaken for a finished one.

    ``outputs_complete`` is deliberately weak -- ``results.csv`` plus a
    non-empty ``out.sdf``, with no requirement that ``complex_*.pdb`` exist --
    because the campaign's default cleanup deletes the complexes after stage 5
    and a stricter predicate would re-dock every finished job on a campaign's
    second run.  Reconcile rule 1 promotes straight to COLLECTED from that
    predicate, so anything it cannot see becomes final and is never looked at
    again.

    Two ordinary crashes used to land inside that gap:

    * a crash while ``parse_gnina`` writes ``results.csv`` -- fewer score rows
      than poses;
    * a crash part-way through the R2 purge -- ``complex_*.pdb`` are deleted
      first and ``results.csv`` last, so an interrupted purge leaves a subset of
      the complexes beside a full ``results.csv``.

    Both produced a directory that passed the predicate, was promoted to
    COLLECTED, and was then fed to the consensus RMSD with a fraction of its
    poses.  ``outputs_complete`` now additionally requires that, *when any
    complex is present*, the complexes, the split poses and the score rows all
    agree -- which is exactly the shape that distinguishes "cleaned up" (no
    complexes at all) from "torn" (some).

    These tests are now the guard on that fix.  The first three are the ones to
    keep whatever else changes: the two crashes must stay detected, and a
    cleaned-up job must stay complete, because the cheapest way to break this
    fix is to make it re-dock every finished job in the campaign.
    """

    def _torn_results_csv(self):
        """20 poses and complexes, 3 score rows: parse_gnina died mid-write."""
        job_dir = self.add_job(outputs=True, n_poses=20)
        with open(os.path.join(job_dir, "gnina", "results.csv"), "w") as handle:
            handle.write("Pose,GNINA_Affinity,GNINA_CNN_pose,GNINA_CNN_affinity\n")
            for i in (1, 2, 3):
                handle.write("%d,-8.%d,0.9,6.1,runA\n" % (i, i))
        return job_dir

    def _interrupted_purge(self):
        """A purge killed after five deletions, all of them complexes."""
        job_dir = self.add_job(outputs=True, n_poses=20)
        gnina_dir = os.path.join(job_dir, "gnina")
        victims = sorted(os.path.join(gnina_dir, n)
                         for n in os.listdir(gnina_dir))
        for path in victims[:5]:
            self.assertIn("complex_", os.path.basename(path),
                          "the purge order changed; this test's premise is stale")
            os.unlink(path)
        return job_dir

    def test_a_torn_results_csv_is_not_complete(self):
        job_dir = self._torn_results_csv()
        detail = gq.outputs_detail(job_dir)
        self.assertEqual((detail["n_result_rows"], detail["n_complexes"]), (3, 20))
        self.assertFalse(gq.outputs_complete(job_dir),
                         "3 score rows for 20 poses is not a finished job")
        self.assertFalse(gq.outputs_verified_at_collect(job_dir)[0])

    def test_an_interrupted_purge_is_not_complete(self):
        job_dir = self._interrupted_purge()
        detail = gq.outputs_detail(job_dir)
        self.assertEqual((detail["n_complexes"], detail["n_poses"]), (15, 20))
        self.assertFalse(gq.outputs_complete(job_dir),
                         "a half-purged directory is not a finished job")

    def test_a_cleaned_up_job_is_still_complete(self):
        """THE regression guard on the fix above.

        ``cleanup_complex_pdbs`` is the CLI default after a successful stage 5
        and removes every ``complex_*.pdb`` in the job.  If "complete" ever
        starts requiring complexes, the pump re-docks every finished job on the
        campaign's second run -- thousands of GPU-hours to reproduce files that
        are deleted again immediately.  Zero complexes means cleaned up; a
        *subset* means torn.
        """
        job_dir = self.add_job(outputs=True, n_poses=20)
        gnina_dir = os.path.join(job_dir, "gnina")
        for name in list(os.listdir(gnina_dir)):
            if name.startswith("complex_"):
                os.unlink(os.path.join(gnina_dir, name))
        self.assertEqual(gq.outputs_detail(job_dir)["n_complexes"], 0)
        self.assertTrue(gq.outputs_complete(job_dir),
                        "a job cleaned up after stage 5 must stay complete")

        # ... and the compressed variant of the same cleanup, which keeps one
        # complex_N.pdb.gz per pose.
        job_dir_b = self.add_job(JOB_B, outputs=True, n_poses=4)
        gnina_b = os.path.join(job_dir_b, "gnina")
        for name in list(os.listdir(gnina_b)):
            if name.startswith("complex_") and name.endswith(".pdb"):
                os.rename(os.path.join(gnina_b, name),
                          os.path.join(gnina_b, name + ".gz"))
        self.assertTrue(gq.outputs_complete(job_dir_b),
                        "compressed complexes must count the same as plain ones")

    def test_reconcile_refuses_to_finalise_a_torn_result(self):
        """The consequence of the fix, at the edge that used to swallow it."""
        job_dir = self._torn_results_csv()
        force_state(self.queue, JOB_A, "COLLECTING",
                    remote={"dir": REMOTE_DIR, "exit_code": 0},
                    collect={"pid": 2 ** 22 - 1, "host": gq._hostname()})
        self.queue.reconcile(prober=prober_for({JOB_A: ("DONE", 0)}))
        self.assertEqual(self.queue.read(JOB_A).state, "REMOTE_DONE",
                         "a torn result must go back for another (cheap, "
                         "download-only) collect, not be declared final")

    def test_the_re_collect_actually_repairs_the_torn_directory(self):
        """End to end: torn on disk, pump restarted, consistent afterwards."""
        job_dir = self._torn_results_csv()
        force_state(self.queue, JOB_A, "COLLECTING",
                    remote={"dir": REMOTE_DIR, "exit_code": 0},
                    collect={"pid": 2 ** 22 - 1, "host": gq._hostname()})
        gpu = FakeGpu({JOB_A: job_dir}, n_poses=20, tag="repaired")
        collector = self.collector(runner=gpu,
                                   prober=prober_for({JOB_A: ("DONE", 0)}))
        self.assertTrue(self.drain(collector))
        self.assertEqual(self.states()[JOB_A], "COLLECTED")
        self.assertEqual(gpu.total_launches(), 0, "a re-collect, not a re-dock")
        detail = gq.outputs_detail(job_dir)
        self.assertEqual(detail["n_result_rows"], detail["n_complexes"])
        self.assertEqual(detail["n_complexes"], detail["n_poses"])

    def test_an_interrupted_cleanup_now_costs_one_extra_collect(self):
        """CHARACTERISATION -- the price of the fix, so it is a known price.

        ``cleanup_complex_pdbs`` deletes complexes one at a time.  Interrupted,
        it leaves a subset -- indistinguishable from a torn collect -- so the
        next pump run re-collects the job.  That is a download and a local
        post-process per affected job, never a re-dock, and it self-heals; it is
        the right trade against silently consuming a fraction of the poses, but
        it is not free and it should not be a surprise.
        """
        job_dir = self.add_job(outputs=True, n_poses=20)
        gnina_dir = os.path.join(job_dir, "gnina")
        force_state(self.queue, JOB_A, "COLLECTED")
        complexes = sorted(n for n in os.listdir(gnina_dir)
                           if n.startswith("complex_"))
        for name in complexes[:12]:            # cleanup killed part-way through
            os.unlink(os.path.join(gnina_dir, name))
        self.assertFalse(gq.outputs_complete(job_dir))

        # A COLLECTED record is terminal, so this costs nothing until the job is
        # re-enqueued by a fresh batch_dock run -- and then it is one collect.
        fresh = gq.GninaQueue(os.path.join(self.tmp, "run2"))
        os.makedirs(os.path.join(self.tmp, "run2", "docking_results"),
                    exist_ok=True)
        record = fresh.enqueue(job_dir, job_inputs(job_dir))
        self.assertEqual(record.state, "PENDING",
                         "the half-cleaned job is enqueued for another collect")

    def test_a_mixture_of_two_runs_is_visible_but_never_re_examined(self):
        """RESIDUAL of R3 -- how the mixture is produced, and what still misses it.

        ``default_runner`` starts consensus_docker with ``start_new_session``,
        so a SIGKILLed pump leaves its collect subprocess running.  Reconcile
        rule 3 then judges the collect by ``collect.pid`` -- which is the
        *pump's* pid, not the subprocess's -- finds it dead, and hands the job
        to a new collect that purges and rewrites the same directory while the
        orphan is still writing into it.

        The strengthened ``outputs_complete`` can now see the result (five poses
        and complexes from the new run, twenty score rows from the old one), but
        two things still do not:

        * the record is already COLLECTED, and reconcile skips terminal states,
          so nothing re-examines it;
        * ``check_tool_completion``, which is what stage 5 filters on, accepts
          any ``results.csv`` that exists.

        Fixing rule 3 to record and check the *writer's* pid would close the
        window itself.
        """
        job_dir = self.add_job()
        gnina_dir = os.path.join(job_dir, "gnina")
        write_gnina_outputs(job_dir, 20, tag="orphan")
        os.unlink(os.path.join(gnina_dir, "results.csv"))   # orphan not finished
        force_state(self.queue, JOB_A, "COLLECTING",
                    remote={"dir": REMOTE_DIR, "exit_code": 0},
                    collect={"pid": 2 ** 22 - 1, "host": gq._hostname()})

        changed = self.queue.reconcile(prober=prober_for({JOB_A: ("DONE", 0)}))
        self.assertEqual(changed.get(JOB_A), ("COLLECTING", "REMOTE_DONE"),
                         "rule 3 judged the pump's pid, not the writer's")

        gpu = FakeGpu({JOB_A: job_dir}, n_poses=5, tag="newrun")
        collector = self.collector(runner=gpu,
                                   prober=prober_for({JOB_A: ("DONE", 0)}))
        collector.run_once()
        self.assertEqual(self.queue.read(JOB_A).state, "COLLECTED")

        # ... and now the orphan finally writes its results.csv.
        with open(os.path.join(gnina_dir, "results.csv"), "w") as handle:
            handle.write("Pose,GNINA_Affinity,GNINA_CNN_pose,GNINA_CNN_affinity\n")
            for i in range(1, 21):
                handle.write("%d,-9.%d,0.9,6.1,orphan\n" % (i, i))

        detail = gq.outputs_detail(job_dir)
        self.assertEqual((detail["n_poses"], detail["n_complexes"],
                          detail["n_result_rows"]), (5, 5, 20))
        self.assertFalse(gq.outputs_complete(job_dir),
                         "the predicate can see the mixture")
        self.queue.reconcile(prober=prober_for({JOB_A: ("DONE", 0)}))
        self.assertEqual(self.queue.read(JOB_A).state, "COLLECTED",
                         "but the record is terminal and is never re-examined")

    def test_the_two_completion_gates_disagree_about_a_torn_result(self):
        """RESIDUAL -- stage 5's gate is blinder than the queue's, in the
        direction that matters.

        ``check_tool_completion`` accepts any ``results.csv`` whose first line
        contains a comma, so a directory the queue considers unfinished (and is
        about to re-collect) is one stage 5 will consume if the barrier times
        out first.  The two gates should agree, or stage 5 should use
        ``gnina_ready``, which batch_dock already defines for exactly this.
        """
        job_dir = self._torn_results_csv()
        with quiet_root_logging():
            try:
                import batch_dock
            except Exception as exc:                # pragma: no cover
                self.skipTest("batch_dock is not importable here: %s" % exc)
            admitted = batch_dock.check_tool_completion(job_dir, "gnina")
            ready = batch_dock.gnina_ready(job_dir)
        self.assertTrue(admitted, "stage 5 admits any results.csv that exists")
        self.assertFalse(gq.outputs_complete(job_dir))
        self.assertFalse(ready, "gnina_ready agrees with the queue, and is the "
                                "predicate stage 5 should be using")


class TestTruncatedRecordRepair(RaceCase):
    """DEFECT R4 (since fixed) -- reconciliation destroyed the only repair path
    for a truncated record.

    ``enqueue`` knows how to fix a record file that holds no readable JSON: it
    overwrites it with a complete record, input paths and all.  So a crash
    between the ``O_EXCL`` name claim and the write costs nothing as long as
    batch_dock is re-run.

    Reconcile rule 2 gets there first.  It re-derives the record from the job
    id alone -- ``JobRecord(job_id=..., job_dir=..., state="PENDING")`` with the
    default, empty ``inputs`` -- and once that readable-but-input-less record
    exists, ``enqueue`` returns it untouched (its whole contract is not to
    disturb work in flight).  ``submit_job`` then abandons the job with
    MISSING_INPUT, and ``--requeue`` only restores it to the same PENDING record
    with the same empty inputs, so the loop is closed: nothing short of deleting
    the record file by hand recovers the job.

    A pump reconciles at startup, so in a real campaign rule 2 always wins.
    """

    def _truncate(self, job_dir):
        with open(self.queue.record_path(JOB_A), "w") as handle:
            handle.write("")
        self.queue._cache.clear()

    def test_enqueue_alone_repairs_a_truncated_record(self):
        """The path that works, so the regression below is unambiguous."""
        job_dir = self.add_job()
        self._truncate(job_dir)
        record = self.queue.enqueue(job_dir, job_inputs(job_dir))
        self.assertEqual(record.state, "PENDING")
        self.assertTrue(record.inputs["receptor_pdb"])

    def test_reconcile_first_is_repaired_by_the_next_enqueue(self):
        """FIXED with R4: a pump always reconciles first, so this is the real order.

        Was: rule 2 re-derived the record without its input paths, ``enqueue``
        returned it untouched, and the first submit ABANDONED it with
        MISSING_INPUT -- a closed loop that only deleting the file by hand broke.
        """
        job_dir = self.add_job()
        self._truncate(job_dir)
        self.queue.reconcile(prober=prober_for({}))
        record = self.queue.enqueue(job_dir, job_inputs(job_dir))
        self.assertTrue(record.inputs["receptor_pdb"],
                        "re-running batch_dock must restore the input paths")
        after = gq.submit_job(self.queue, self.queue.read(JOB_A),
                              runner=FakeGpu({JOB_A: job_dir}))
        self.assertEqual(after.state, "SUBMITTED")

    def test_only_a_pending_record_has_its_inputs_repaired(self):
        """The boundary on the repair, which is what makes it safe.

        Filling in the inputs of a SUBMITTED job would let a re-run silently
        re-point it at different files while the GPU is working on the old ones,
        and the collect phase would then post-process one job's poses against
        another job's receptor.  PENDING is the only state where nothing has
        been launched that could be contradicted.
        """
        job_dir = self.add_job()
        force_state(self.queue, JOB_A, "SUBMITTED", remote={"dir": REMOTE_DIR},
                    inputs={"receptor_pdb": None})
        record = self.queue.enqueue(job_dir, job_inputs(job_dir))
        self.assertEqual(record.state, "SUBMITTED")
        self.assertIsNone(record.inputs["receptor_pdb"],
                          "a live record must not be edited by an enqueue")

    def test_a_re_run_of_batch_dock_must_repair_a_re_derived_record(self):
        """DEFECT R4, FIXED: re-running batch_dock repairs a re-derived record.

        Was: enqueue returned the input-less record untouched and the job was
        ABANDONED with MISSING_INPUT on its first submit, with ``--requeue``
        restoring the same empty record -- a closed loop.

        Fixed in ``GninaQueue.enqueue``, which now fills in input paths that are
        missing from an existing **PENDING** record.  PENDING is the only state
        where that is safe: nothing has been launched, so no record of something
        that already ran can be contradicted.
        """
        job_dir = self.add_job()
        self._truncate(job_dir)
        self.queue.reconcile(prober=prober_for({}))
        self.queue.enqueue(job_dir, job_inputs(job_dir))
        gpu = FakeGpu({JOB_A: job_dir})
        collector = self.collector(runner=gpu,
                                   prober=prober_for({JOB_A: ("DONE", 0)}))
        self.drain(collector)
        self.assertEqual(self.queue.read(JOB_A).state, "COLLECTED")


class TestCollectingDeadline(RaceCase):
    """DEFECT R5 (since fixed) -- a record could stay COLLECTING forever.

    Reconcile rule 3 rescues a collect whose ``collect.pid`` is dead.  If the
    pid was *alive* it returned immediately, and there was no other rule, no
    deadline and no grace period anywhere for COLLECTING -- unlike SUBMITTED and
    RUNNING, which are declared STALLED past ``job_deadline_seconds +
    STALL_GRACE_SECONDS``.

    A pid is alive but not collecting whenever the number has been recycled by
    an unrelated process (records survive reboots; pids do not), or when the
    pump process lives on with its worker thread dead.  The job was then
    outstanding for the rest of the campaign: the barrier waited its entire
    timeout on it and stage 5 skipped it.  Rule 3 now also demotes a collect
    past its deadline whatever the pid says, which is safe because collecting
    is idempotent -- being wrong costs one repeated download, never a re-dock.

    Both halves of that rule are tested: believed inside the deadline, overruled
    outside it.
    """

    def test_a_live_pid_holds_the_record_only_until_the_deadline(self):
        """FIXED with R5, but only after the deadline: a live pid is still respected.

        Inside the deadline the pid is believed, which is what stops a healthy
        slow download from being restarted underneath itself.
        """
        job_dir = self.add_job()
        force_state(self.queue, JOB_A, "COLLECTING",
                    remote={"dir": REMOTE_DIR, "exit_code": 0},
                    collect={"pid": os.getpid(),      # alive, and not a collect
                             "host": gq._hostname(),
                             "started_at": gq._utcnow_iso()})
        for _ in range(3):
            self.assertEqual(self.queue.reconcile(prober=prober_for({})), {})
            self.assertEqual(self.queue.read(JOB_A).state, "COLLECTING")
        self.assertEqual(self.queue.summary()["outstanding"], 1)

    def test_a_collect_older_than_the_deadline_must_be_rescued(self):
        """DEFECT R5, FIXED WHILE THIS FILE WAS BEING WRITTEN -- now the guard.

        COLLECTING used to have no timeout of any kind: reconcile rule 3
        rescued a collect whose pid was dead and returned immediately if it was
        alive, so a recycled pid (records outlive reboots; pids do not) or a
        wedged download pinned the record for the rest of the campaign and the
        barrier waited its whole timeout on it.  Rule 3 now also demotes a
        collect older than ``job_deadline_seconds + STALL_GRACE_SECONDS``
        whatever the pid says -- safe because re-collecting is idempotent, so
        being wrong costs one repeated download and never a re-dock.
        """
        job_dir = self.add_job()
        force_state(self.queue, JOB_A, "COLLECTING",
                    remote={"dir": REMOTE_DIR, "exit_code": 0},
                    collect={"pid": os.getpid(), "host": gq._hostname(),
                             "started_at": "2020-01-01T00:00:00.000000Z"})
        self.queue.reconcile(prober=prober_for({JOB_A: ("DONE", 0)}))
        self.assertNotEqual(self.queue.read(JOB_A).state, "COLLECTING")

    def test_a_collecting_record_does_not_hold_the_gpu_slot(self):
        """The damage is bounded: COLLECTING is not in the active-remote set, so
        the GPU keeps working through the rest of the queue."""
        self.add_job(JOB_A)
        self.add_job(JOB_B)
        force_state(self.queue, JOB_A, "COLLECTING",
                    collect={"pid": os.getpid(), "host": gq._hostname()})
        self.assertIsNone(self.queue.active_remote())
        self.assertEqual(self.queue.next_pending().job_id, JOB_B)


class TestRecordIdentity(RaceCase):
    """DEFECT R6 (FIXED WHILE THIS FILE WAS BEING WRITTEN) -- a record is
    addressed by its filename for writes and by its embedded ``job_id`` for
    reads, and nothing checked that the two agreed.

    ``read_all`` used to yield whatever ``job_id`` the document contained, while
    ``read`` / ``transition`` / ``_write`` all derive the path from the id.  A
    record whose filename did not match its content was therefore visible in
    every count and in the barrier's outstanding total, but no write could reach
    it -- ``transition`` raised "no queue record" -- so it was outstanding
    forever, and a write that did land would have gone to a *different* file,
    forking the job in two.  ``read_all`` now treats a mismatch as unreadable,
    which it is.

    Reachable by an operator renaming or copying a record file, which is the
    obvious way to try to "pause" a job by hand.
    """

    def test_a_renamed_record_is_reported_rather_than_half_readable(self):
        job_dir = self.add_job()
        os.rename(self.queue.record_path(JOB_A),
                  self.queue.record_path("renamed_by_an_operator"))
        self.queue._cache.clear()

        self.assertEqual([r.job_id for r in self.queue.read_all()], [],
                         "a record that cannot be written back must not be "
                         "yielded as if it could")
        self.assertIn("renamed_by_an_operator", self.queue.unreadable_job_ids())
        self.assertIn("renamed_by_an_operator",
                      self.queue.summary()["unreadable_job_ids"])

    def test_every_record_read_must_be_writable(self):
        """The invariant itself, which is what the fix above buys.

        If ``read_all`` can yield a record ``read`` cannot find, the pump and
        the barrier can both see a job that no code in the system can advance.
        """
        job_dir = self.add_job()
        os.rename(self.queue.record_path(JOB_A),
                  self.queue.record_path("renamed_by_an_operator"))
        self.queue._cache.clear()
        for record in self.queue.read_all():
            self.assertIsNotNone(
                self.queue.read(record.job_id),
                "read_all yielded %s, which read() cannot find" % record.job_id)

    def test_a_renamed_record_leaves_the_job_recoverable_by_re_enqueue(self):
        """The job itself is not lost: its record file is simply not its record
        any more, so a batch_dock re-run enqueues it again from scratch."""
        job_dir = self.add_job()
        os.rename(self.queue.record_path(JOB_A),
                  self.queue.record_path("renamed_by_an_operator"))
        self.queue._cache.clear()
        record = self.queue.enqueue(job_dir, job_inputs(job_dir))
        self.assertEqual(record.state, "PENDING")
        self.assertTrue(record.inputs["receptor_pdb"])

    def test_a_newer_schema_version_is_carried_through_untouched(self):
        """Forward compatibility works -- and is completely ungated.

        Unknown keys and a future ``schema_version`` both survive a rewrite,
        which is what the design asks for.  What is missing is any refusal: a
        record written by a future version is acted on with today's rules, and
        if that version changed what a field means, the damage is silent.  See
        this module's ``risks`` note; it is deliberately not asserted as a
        defect because the right gate is a design decision.
        """
        self.add_job()
        path = self.queue.record_path(JOB_A)
        with open(path) as handle:
            data = json.load(handle)
        data["schema_version"] = 99
        data["gpu_affinity"] = {"device": 3}
        gq.atomic_write_json(path, data)
        self.queue._cache.clear()

        self.queue.transition(JOB_A, "SUBMITTING", why="acted on regardless")

        with open(path) as handle:
            rewritten = json.load(handle)
        self.assertEqual(rewritten["schema_version"], 99)
        self.assertEqual(rewritten["gpu_affinity"], {"device": 3})
        self.assertEqual(rewritten["state"], "SUBMITTING")


# ---------------------------------------------------------------------------
# 7. the RMSD barrier
# ---------------------------------------------------------------------------

class TestBarrier(RaceCase):
    """The barrier decides what stage 5 is allowed to compute.

    It must never fail the run, never hang forever, and never let a partial
    campaign pass for a complete one.  Every test here holds the pump lock,
    because a free lock means "no collector" and takes a different branch --
    which is itself the last test in this class.
    """

    def setUp(self):
        super(TestBarrier, self).setUp()
        self.pump_lock = self.queue.acquire_pump_lock()
        self.assertIsNotNone(self.pump_lock)

    def tearDown(self):
        if self.pump_lock is not None:
            self.pump_lock.release()
        super(TestBarrier, self).tearDown()

    def test_a_result_arriving_during_the_wait_releases_the_barrier(self):
        job_dir = self.add_job()
        force_state(self.queue, JOB_A, "SUBMITTED", remote={"dir": REMOTE_DIR})
        clock = TickingClock(1.0)
        landed = {"done": False}

        def sleeper(_seconds):
            if clock.now > 4 and not landed["done"]:
                write_gnina_outputs(job_dir, 3)
                force_state(self.queue, JOB_A, "COLLECTED")
                landed["done"] = True

        summary = gq.wait_for_completion(
            self.run_dir, timeout=10000.0, poll=1.0, clock=clock,
            sleeper=sleeper)

        self.assertTrue(landed["done"])
        self.assertEqual(summary["outstanding"], 0)
        self.assertEqual(summary["collected"], 1)
        self.assertFalse(summary["timed_out"])

    def test_a_result_arriving_after_the_timeout_is_reported_as_outstanding(self):
        self.add_job()
        force_state(self.queue, JOB_A, "SUBMITTED", remote={"dir": REMOTE_DIR})
        summary = gq.wait_for_completion(
            self.run_dir, timeout=5.0, poll=1.0, clock=TickingClock(1.0),
            sleeper=lambda _s: None)
        self.assertTrue(summary["timed_out"])
        self.assertEqual(summary["outstanding"], 1)
        self.assertIn(JOB_A, summary["outstanding_job_ids"])

    def test_a_timeout_names_the_outstanding_jobs_on_disk(self):
        """summary.json is what ``_report_gnina_exclusions`` points an operator
        at, so the names have to survive the process, not just the log."""
        self.add_job()
        force_state(self.queue, JOB_A, "SUBMITTED", remote={"dir": REMOTE_DIR})
        gq.wait_for_completion(self.run_dir, timeout=3.0, poll=1.0,
                               clock=TickingClock(1.0), sleeper=lambda _s: None)
        with open(self.queue.summary_path) as handle:
            on_disk = json.load(handle)
        self.assertTrue(on_disk["timed_out"])
        self.assertIn(JOB_A, on_disk["outstanding_job_ids"])

    def test_a_result_that_never_arrives_does_not_hang_or_raise(self):
        for job_id in (JOB_A, JOB_B, JOB_C):
            self.add_job(job_id)
        summary = gq.wait_for_completion(
            self.run_dir, timeout=6.0, poll=1.0, clock=TickingClock(1.0),
            sleeper=lambda _s: None)
        self.assertTrue(summary["timed_out"])
        self.assertEqual(summary["outstanding"], 3)
        self.assertEqual(summary["collected"], 0)

    def test_timeout_zero_does_not_wait_at_all(self):
        """"Enqueue now, collect later" has to be expressible."""
        self.add_job()
        sleeps = []
        summary = gq.wait_for_completion(
            self.run_dir, timeout=0.0, poll=1.0, clock=TickingClock(1.0),
            sleeper=sleeps.append)
        self.assertEqual(sleeps, [])
        self.assertTrue(summary["timed_out"])
        self.assertEqual(summary["outstanding"], 1)

    def test_an_abandoned_job_is_not_outstanding_and_is_named(self):
        self.add_job(JOB_A)
        self.add_job(JOB_B, outputs=True)
        force_state(self.queue, JOB_B, "COLLECTED")
        force_state(self.queue, JOB_A, "ABANDONED",
                    error=gq.make_error("REMOTE_GNINA_FAILED", "CUDA OOM", "poll"))
        summary = gq.wait_for_completion(
            self.run_dir, timeout=60.0, poll=1.0, clock=TickingClock(1.0),
            sleeper=lambda _s: None)
        self.assertEqual(summary["outstanding"], 0)
        self.assertFalse(summary["timed_out"])
        self.assertEqual(summary["abandoned"], 1)
        self.assertIn(JOB_A, summary["abandoned_job_ids"])

    def test_the_barrier_never_sees_a_finished_job_that_only_exists_on_disk(self):
        """CHARACTERISATION -- the barrier reads records, never the job directory.

        A pump that wrote the outputs and died before updating the record leaves
        a job that is finished on disk and outstanding in the queue.  The
        barrier waits its whole timeout for it and then names it as missing,
        even though stage 5 (which checks the disk) will happily use it.  Not
        wrong, but it is why a timed-out barrier's list is an upper bound on
        what is really missing -- and why the barrier should reconcile before it
        gives up.
        """
        job_dir = self.add_job()
        write_gnina_outputs(job_dir, 3)
        summary = gq.wait_for_completion(
            self.run_dir, timeout=4.0, poll=1.0, clock=TickingClock(1.0),
            sleeper=lambda _s: None)
        self.assertTrue(gq.outputs_complete(job_dir))
        self.assertTrue(summary["timed_out"])
        self.assertIn(JOB_A, summary["outstanding_job_ids"])


class TestBarrierWithoutAPump(RaceCase):
    """A silently dead collector must never look like a slow GPU."""

    def test_a_free_pump_lock_triggers_restarts_then_gives_up(self):
        self.add_job(JOB_A)
        self.add_job(JOB_B)
        force_state(self.queue, JOB_B, "FAILED",
                    error=gq.make_error("SSH_TRANSIENT", "link died", "poll"))
        restarts = []

        summary = gq.wait_for_completion(
            self.run_dir, timeout=100000.0, poll=1.0, clock=TickingClock(1.0),
            sleeper=lambda _s: None, restart_collector=lambda: restarts.append(1),
            max_restarts=3)

        self.assertEqual(len(restarts), 3, "the barrier must try to revive it")
        self.assertEqual(summary["collector_restarts"], 3)
        self.assertTrue(summary["timed_out"])
        self.assertEqual(self.queue.read(JOB_B).state, "ABANDONED")
        self.assertEqual(self.queue.read(JOB_B).error_code, "PUMP_UNAVAILABLE")
        self.assertEqual(self.queue.read(JOB_A).state, "PENDING",
                         "only FAILED jobs may be abandoned this way")

    def test_a_restart_callback_that_raises_does_not_break_the_barrier(self):
        self.add_job()

        def explode():
            raise RuntimeError("spawn failed: no such file")

        summary = gq.wait_for_completion(
            self.run_dir, timeout=100000.0, poll=1.0, clock=TickingClock(1.0),
            sleeper=lambda _s: None, restart_collector=explode, max_restarts=2)

        self.assertTrue(summary["timed_out"])
        self.assertEqual(summary["outstanding"], 1)

    def test_one_transient_free_observation_does_not_evict_a_starting_pump(self):
        """The debounce matters: a pump that is a moment from taking the lock
        must not be "restarted" into a second pump."""
        self.add_job()
        lock = self.queue.acquire_pump_lock()
        restarts = []
        try:
            gq.wait_for_completion(
                self.run_dir, timeout=5.0, poll=1.0, clock=TickingClock(1.0),
                sleeper=lambda _s: None,
                restart_collector=lambda: restarts.append(1), max_restarts=3)
        finally:
            lock.release()
        self.assertEqual(restarts, [], "a live pump was restarted underneath")


# ---------------------------------------------------------------------------
# 8. the single-GPU invariant under sustained abuse
# ---------------------------------------------------------------------------

class TestSingleGpuInvariantUnderCrashes(RaceCase):
    """One GPU means at most one job in SUBMITTING/SUBMITTED/RUNNING, ever.

    A long simulation with the pump killed and restarted at random points, jobs
    failing on the GPU, and the remote answering slowly.  The invariant is
    sampled from *inside* the fake GPU as well as between passes, so a violation
    that exists only during a pass is still caught.
    """

    def test_a_crash_ridden_campaign_never_doubles_or_loses_a_job(self):
        import random

        rng = random.Random(20260813)
        job_ids = ["DB%05d__P30518__AVPR2__F1__cav1" % i for i in range(24)]
        for job_id in job_ids:
            self.add_job(job_id)

        max_active = {"n": 0}
        failing = set(job_ids[:3])        # these fail on the GPU, once each
        failed_once = set()
        # The remote box, modelled honestly: a launch writes an exit_code that
        # then stays put until the job is launched again. A prober that simply
        # started answering DONE 0 after a failure would let reconcile rule 6
        # "collect" a result that never existed.
        remote_exit = {}
        polls_since_launch = {}
        # One clock for the queue and every restarted pump, advancing a second
        # per read: retry backoffs then expire deterministically instead of
        # being waited out in real time.
        clock = TickingClock(1.0)

        def sample():
            active = self.queue.by_state("SUBMITTING", "SUBMITTED", "RUNNING")
            max_active["n"] = max(max_active["n"], len(active))

        def collect_hook(gpu, job_id, env):
            sample()
            if remote_exit.get(job_id) != 0:
                raise AssertionError(
                    "%s was collected while the remote said %r"
                    % (job_id, remote_exit.get(job_id)))
            return None

        def submit_hook(gpu, job_id, env):
            sample()
            polls_since_launch[job_id] = 0
            if job_id in failing and job_id not in failed_once:
                failed_once.add(job_id)
                remote_exit[job_id] = 1          # CUDA OOM, say
            else:
                remote_exit[job_id] = 0
            return None

        gpu = FakeGpu(self.job_dirs, collect_hook=collect_hook,
                      submit_hook=submit_hook)

        def prober(record):
            sample()
            job_id = record.job_id
            if job_id not in remote_exit:
                return ("GONE", None)
            polls_since_launch[job_id] = polls_since_launch.get(job_id, 0) + 1
            if polls_since_launch[job_id] < 2:
                return ("RUNNING", None)
            return ("DONE", remote_exit[job_id])

        collector = self.collector(runner=gpu, prober=prober, collect_workers=2,
                                   clock=clock)
        for step in range(400):
            collector.run_once()
            sample()
            if rng.random() < 0.2:         # the pump dies and is restarted
                collector.close()
                collector = self.collector(runner=gpu, prober=prober,
                                           collect_workers=2, clock=clock)
            if all(r.state in gq.TERMINAL_STATES
                   for r in self.queue.read_all()):
                break

        states = self.states()
        self.assertEqual(max_active["n"], 1,
                         "more than one job was outstanding on a single GPU")
        self.assertEqual(sorted(states), sorted(job_ids), "a job disappeared")
        self.assertTrue(all(s == "COLLECTED" for s in states.values()),
                        "not every job finished: %r"
                        % {k: v for k, v in states.items() if v != "COLLECTED"})
        for job_id in job_ids:
            self.assertEqual(gpu.collects.get(job_id), 1,
                             "%s was collected %r times"
                             % (job_id, gpu.collects.get(job_id)))
            expected = 2 if job_id in failing else 1
            self.assertEqual(gpu.launches.get(job_id), expected,
                             "%s was docked %r times, expected %d"
                             % (job_id, gpu.launches.get(job_id), expected))


# ---------------------------------------------------------------------------
# 9. operational traps
# ---------------------------------------------------------------------------

class TestOperationalTraps(RaceCase):
    """Ways to switch the pump off by accident.

    These are all "working as designed"; they are here because the design's
    couplings are invisible at the call site, and a campaign that silently
    collects nothing looks exactly like a campaign with nothing to collect.
    """

    def test_a_leftover_stop_file_makes_a_fresh_pump_a_no_op(self):
        """batch_dock writes ``STOP`` on exit and only deletes it again if it
        spawned the collector itself.  A hand-started pump then stops on its
        first pass having collected nothing, and exits 4."""
        job_dir = self.add_job()
        force_state(self.queue, JOB_A, "REMOTE_DONE",
                    remote={"dir": REMOTE_DIR, "exit_code": 0})
        with open(os.path.join(self.queue.root, "STOP"), "w") as handle:
            handle.write("{}")

        with restored_logging():
            code = gnina_collect.main(
                ["--run-dir", self.run_dir, "--once", "--quiet"])

        self.assertEqual(code, gnina_collect.EXIT_STOPPED)
        self.assertEqual(self.queue.read(JOB_A).state, "REMOTE_DONE",
                         "the STOP file silently prevented every collect")

    def test_a_pump_without_params_json_refuses_rather_than_guessing(self):
        """Docking parameters are never improvised: an unparameterised queue
        must not submit anything at all."""
        job_dir = self.add_job()
        os.unlink(self.queue.params_path)
        collector = self.collector(runner=FakeGpu({JOB_A: job_dir}))
        with self.assertRaises(gnina_collect.CollectorError):
            collector.run_once()
        self.assertEqual(self.queue.read(JOB_A).state, "PENDING")

    def test_changing_the_docking_parameters_mid_campaign_is_refused(self):
        """Half a campaign at exhaustiveness 32 and half at 8 is two campaigns
        with no record of where the seam is."""
        job_dir = self.add_job()
        changed = dict(PARAMS, exhaustiveness=8)
        with self.assertRaises(gq.QueueError) as caught:
            self.queue.enqueue_many(
                [dict(current_outfolder=job_dir, **job_inputs(job_dir))], changed)
        self.assertIn("exhaustiveness", str(caught.exception))
        self.assertEqual(self.queue.get_params()["exhaustiveness"], 32)

    def test_a_docking_backoff_also_delays_the_later_collect(self):
        """CHARACTERISATION -- ``next_attempt_at`` is set but never cleared.

        ``apply_retry_policy`` stamps a re-dock backoff (up to 60 s) on the
        record, and nothing on the way through SUBMITTING / SUBMITTED /
        REMOTE_DONE removes it.  ``Collector._schedule_collects`` honours the
        same field, so a job that once failed on the GPU has its *download*
        deferred by a delay that was meant for its *docking*.  Only throughput,
        never correctness -- but it is invisible, and it lengthens exactly the
        jobs that are already behind.
        """
        job_dir = self.add_job()
        force_state(self.queue, JOB_A, "FAILED", attempts=1,
                    remote={"dir": REMOTE_DIR},
                    error=gq.make_error("REMOTE_GNINA_FAILED", "CUDA OOM", "poll"))
        record = gq.apply_retry_policy(self.queue, self.queue.read(JOB_A),
                                       max_attempts=3)
        self.assertEqual(record.state, "PENDING")
        backoff_until = record.next_attempt_at
        self.assertGreater(backoff_until, self.queue.now())

        force_state(self.queue, JOB_A, "REMOTE_DONE")   # the re-dock succeeded
        self.assertEqual(self.queue.read(JOB_A).next_attempt_at, backoff_until,
                         "the docking backoff survived into REMOTE_DONE")

        collector = self.collector(runner=FakeGpu({JOB_A: job_dir}))
        self.assertEqual(collector._schedule_collects(), 0,
                         "the collect was deferred by a docking backoff")
        self.assertEqual(self.queue.read(JOB_A).state, "REMOTE_DONE")

    def test_requeue_only_touches_terminal_failures(self):
        """The one operator write that takes no lock is safe because ABANDONED
        is the one state a running pump never writes over."""
        job_dir = self.add_job()
        force_state(self.queue, JOB_A, "SUBMITTED", remote={"dir": REMOTE_DIR})
        args = gnina_collect.build_parser().parse_args(
            ["--run-dir", self.run_dir, "--requeue", JOB_A])
        with restored_stdio():
            code = gnina_collect.cmd_requeue(args)
        self.assertEqual(code, gnina_collect.EXIT_USAGE)
        self.assertEqual(self.queue.read(JOB_A).state, "SUBMITTED")

    def test_requeue_of_an_abandoned_job_keeps_the_remote_directory(self):
        """So the retry collects the finished remote result instead of paying
        for another 2.5 minutes of GPU time."""
        job_dir = self.add_job()
        force_state(self.queue, JOB_A, "ABANDONED", attempts=3,
                    remote={"dir": REMOTE_DIR, "exit_code": 0},
                    error=gq.make_error("STALLED", "silent", "poll"))
        args = gnina_collect.build_parser().parse_args(
            ["--run-dir", self.run_dir, "--requeue", JOB_A])
        with restored_stdio():
            code = gnina_collect.cmd_requeue(args)
        self.assertEqual(code, gnina_collect.EXIT_OK)
        record = self.queue.read(JOB_A)
        self.assertEqual(record.state, "PENDING")
        self.assertEqual(record.attempts, 0)
        self.assertEqual(record.remote["dir"], REMOTE_DIR)
        self.assertIsNone(record.next_attempt_at)


# ---------------------------------------------------------------------------
# small context managers used above
# ---------------------------------------------------------------------------

class restored_logging(object):
    """Put the root logger back after ``gnina_collect.main`` reconfigures it.

    ``main`` calls ``_configure_logging``, which sets the root level and adds a
    stderr handler -- reasonable for a process, ruinous for the rest of a shared
    test suite.  Signal handlers get the same treatment: ``main`` installs
    SIGINT/SIGTERM handlers that outlive it.
    """

    def __enter__(self):
        root = logging.getLogger()
        self._handlers = list(root.handlers)
        self._level = root.level
        self._signals = {}
        for name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    self._signals[sig] = signal.getsignal(sig)
                except (ValueError, OSError):    # pragma: no cover
                    pass
        return self

    def __exit__(self, *exc):
        root = logging.getLogger()
        for handler in list(root.handlers):
            if handler not in self._handlers:
                root.removeHandler(handler)
                try:
                    handler.close()
                except Exception:                # pragma: no cover
                    pass
        for handler in self._handlers:
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(self._level)
        for sig, previous in self._signals.items():
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError, TypeError):   # pragma: no cover
                pass
        return False


class quiet_root_logging(object):
    """Mute the root logger for the duration.

    ``batch_dock`` configures logging and logs a banner at IMPORT, and
    ``check_tool_completion`` narrates every file it looks at.  Neither belongs
    in this suite's output.
    """

    def __enter__(self):
        root = logging.getLogger()
        self._level = root.level
        self._disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        return self

    def __exit__(self, *exc):
        logging.disable(self._disable)
        logging.getLogger().setLevel(self._level)
        return False


class restored_stdio(object):
    """Swallow a CLI helper's ``print`` output without losing it on failure."""

    def __enter__(self):
        import io
        self._out, self._err = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self._out, self._err
        return False


if __name__ == "__main__":                        # pragma: no cover
    unittest.main(verbosity=2)
