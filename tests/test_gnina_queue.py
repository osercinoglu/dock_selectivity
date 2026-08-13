"""Tests for dock_selectivity.gnina_queue -- the durable gnina job store.

This is a sync protocol with two crash domains (this machine and the GPU box)
and three writers (the pump, the barrier, an operator).  Nothing here talks to
a network, a GPU or an SSH host: every remote interaction goes through an
injected seam (``runner``, ``prober``, ``clock``), which is what makes the whole
protocol testable in-process.  Everything is built inside
``tempfile.TemporaryDirectory()``.

What is actually being defended, in order of how much a failure would cost:

* **a finished job is never re-docked** -- outputs on local disk outrank every
  recorded state, so a lost or stale queue costs a re-scan, not GPU-months;
* **a job is never docked twice at once** -- one GPU means at most one record in
  SUBMITTING/SUBMITTED/RUNNING, and one pump per queue;
* **a crash never corrupts the store** -- readers see the old record or the new
  one, and a record truncated by a crash is re-derived rather than believed;
* **a stale lock never deadlocks the campaign** -- and never gets broken while
  its owner is alive;
* **stale poses never reach the consensus** -- the purge is exact, and the
  submit phase writing a ``results.csv`` stops the job loudly.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from dock_selectivity import gnina_queue as gq  # noqa: E402

_SILENCER = logging.NullHandler()


def setUpModule():
    """Keep the module's (deliberately loud) warnings out of the suite's output.

    Several tests provoke WARNING and CRITICAL messages on purpose.  Without a
    handler, logging's last-resort writer puts them on stderr and buries the
    real test results.  The tests that care that a message is emitted assert on
    it with ``assertLogs``, which installs its own handler regardless.
    """
    logger = logging.getLogger(gq.__name__)
    logger.addHandler(_SILENCER)
    logger.propagate = False


def tearDownModule():
    logger = logging.getLogger(gq.__name__)
    logger.removeHandler(_SILENCER)
    logger.propagate = True


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

JOB_A = "DB00035__P30518__AVPR2__F1__cav1"
JOB_B = "DB00035__P30518__AVPR2__F1__cav2"
JOB_C = "DB01234__Q9Y5N1__HRH3__F1__cav1"

PARAMS = {
    "schema_version": 1,
    "cpu": 24,
    "exhaustiveness": 32,
    "num_modes": None,
    "cutoff_value": -7.0,
    "consensus_docker_script": "/abs/consensus_docker.py",
    "gnina_path": "/abs/gnina_remote.py",
    "image": "gnina/gnina:v1.3.3",
    "host": "onur@headscale.bio-cloud.site",
    "port": 50000,
    "key": None,
    "native": None,
    "remote_root": "gnina_jobs",
    "job_deadline_seconds": 1800.0,
    "max_attempts": 3,
}


def make_run(tmp):
    """A run directory with a docking_results/ tree, as batch_dock lays it out."""
    run_dir = os.path.join(tmp, "run")
    os.makedirs(os.path.join(run_dir, "docking_results"), exist_ok=True)
    return run_dir


def make_job_dir(run_dir, job_id, outputs=False, n_poses=3):
    """One job directory, optionally with a finished gnina result in it."""
    job_dir = os.path.join(run_dir, "docking_results", job_id)
    for sub in ("input", "smina", "gnina", "gold", "ledock"):
        os.makedirs(os.path.join(job_dir, sub), exist_ok=True)
    for name, body in (("receptor.pdb", "ATOM      1  N   ALA A   1\n"),
                       ("ligand.sdf", "ligand\n$$$$\n"),
                       ("pocket.pdb", "ATOM      1  N   ALA A   1\n")):
        with open(os.path.join(job_dir, "input", name), "w") as handle:
            handle.write(body)
    if outputs:
        write_gnina_outputs(job_dir, n_poses)
    return job_dir


def write_gnina_outputs(job_dir, n_poses=3, n_result_rows=None, n_complexes=None):
    """Exactly what consensus_docker leaves behind after a successful gnina run."""
    gnina = os.path.join(job_dir, "gnina")
    os.makedirs(gnina, exist_ok=True)
    rows = n_poses if n_result_rows is None else n_result_rows
    complexes = n_poses if n_complexes is None else n_complexes
    with open(os.path.join(gnina, "out.sdf"), "w") as handle:
        handle.write("pose\n$$$$\n" * max(1, n_poses))
    with open(os.path.join(gnina, "out.log"), "w") as handle:
        handle.write("mode |   affinity\n")
    for i in range(1, n_poses + 1):
        with open(os.path.join(gnina, "out_%d.sdf" % i), "w") as handle:
            handle.write("pose %d\n$$$$\n" % i)
    for i in range(1, complexes + 1):
        with open(os.path.join(gnina, "complex_%d.pdb" % i), "w") as handle:
            handle.write("ATOM      1  N   ALA A   1\n")
    with open(os.path.join(gnina, "results.csv"), "w") as handle:
        handle.write("Pose,GNINA_Affinity,GNINA_CNN_pose,GNINA_CNN_affinity\n")
        for i in range(1, rows + 1):
            handle.write("%d,-8.%d,0.9,6.1\n" % (i, i))


def job_inputs(job_dir):
    return {
        "receptor_pdb": os.path.join(job_dir, "input", "receptor.pdb"),
        "ligand_sdf": os.path.join(job_dir, "input", "ligand.sdf"),
        "pocket_pdb": os.path.join(job_dir, "input", "pocket.pdb"),
        "receptor_pdbqt": None,
        "skip_pdb_to_pdbqt": False,
    }


def force_state(queue, job_id, state, **fields):
    """Put a record into *state* without going through transition().

    Only a fixture helper: the point of the transition table is that production
    code cannot do this.
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


class FakeRunner(object):
    """Stands in for the ``consensus_docker.py`` subprocess.

    It receives exactly what the real runner receives -- argv, environment,
    timeout, log path, cwd -- and it writes the ticket to the path the queue put
    in ``GNINA_REMOTE_TICKET``, so the ticket handshake is exercised for real.
    """

    def __init__(self, ticket=None, returncode=0, side_effect=None):
        self.ticket = ticket
        self.returncode = returncode
        self.side_effect = side_effect
        self.calls = []

    def __call__(self, cmd, env, timeout, log_path, cwd):
        self.calls.append({"cmd": list(cmd), "env": dict(env), "timeout": timeout,
                           "log_path": log_path, "cwd": cwd,
                           "mode": env.get("GNINA_REMOTE_MODE")})
        if self.side_effect is not None:
            result = self.side_effect(self, cmd, env, log_path, cwd)
            if result is not None:
                return result
        if self.ticket is not None:
            path = env.get("GNINA_REMOTE_TICKET")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as handle:
                json.dump(self.ticket, handle)
        return self.returncode, "", ""


def launched_ticket(remote_dir="/home/onur/gnina_jobs/gnina_ab12cd34ef567890"):
    return {"schema_version": 1, "mode": "submit", "outcome": "launched",
            "host": PARAMS["host"], "port": PARAMS["port"],
            "remote_dir": remote_dir, "job_key": "ab12cd34ef567890",
            "image": PARAMS["image"], "lock": "acquired", "exit_code": None,
            "error": None}


def busy_ticket():
    return {"schema_version": 1, "mode": "submit", "outcome": "busy",
            "host": PARAMS["host"], "remote_dir": None, "lock": "busy",
            "error": None}


def collected_ticket():
    return {"schema_version": 1, "mode": "collect", "outcome": "collected",
            "host": PARAMS["host"], "remote_dir": "/home/onur/gnina_jobs/gnina_ab12",
            "exit_code": 0, "error": None}


def fake_prober(mapping):
    """prober(record) -> (state, exit_code), driven by a job_id -> answer map."""
    def prober(record):
        answer = mapping.get(record.job_id, ("LAUNCHED", None))
        if isinstance(answer, Exception):
            raise answer
        return answer
    return prober


class QueueCase(unittest.TestCase):
    """Shared setup: a run directory, a queue and one enqueued job."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.run_dir = make_run(self.tmp)
        self._saved_env = os.environ.pop("GNINA_QUEUE_DIR", None)
        self.queue = gq.GninaQueue(self.run_dir)
        self.queue.set_params(PARAMS)

    def tearDown(self):
        if self._saved_env is not None:
            os.environ["GNINA_QUEUE_DIR"] = self._saved_env
        self._tmp.cleanup()

    def enqueue(self, job_id=JOB_A, outputs=False):
        job_dir = make_job_dir(self.run_dir, job_id, outputs=outputs)
        return self.queue.enqueue(job_dir, job_inputs(job_dir)), job_dir


# ---------------------------------------------------------------------------
# the transition table
# ---------------------------------------------------------------------------

class TestTransitionTable(unittest.TestCase):
    """The table is the contract; both halves of it must stay visible."""

    def test_the_specified_table_is_intact(self):
        """Every edge the design specifies is present, unmodified."""
        for state, targets in gq._SPEC_TRANSITIONS.items():
            for target in targets:
                self.assertIn(target, gq.LEGAL_TRANSITIONS[state],
                              "%s -> %s is specified and must stay legal"
                              % (state, target))

    def test_extra_edges_are_declared_not_smuggled(self):
        """Anything beyond the spec table lives in _RECONCILE_TRANSITIONS."""
        for state in gq.LEGAL_TRANSITIONS:
            extra = gq.LEGAL_TRANSITIONS[state] - gq._SPEC_TRANSITIONS.get(
                state, frozenset())
            self.assertEqual(
                extra, gq._RECONCILE_TRANSITIONS.get(state, frozenset()),
                "undeclared extra edges from %s" % state)

    def test_terminal_states_have_no_exits(self):
        """COLLECTED and ABANDONED are terminal; only an operator resets them."""
        for state in gq.TERMINAL_STATES:
            self.assertEqual(gq.LEGAL_TRANSITIONS[state], frozenset())

    def test_every_state_is_reachable_and_named(self):
        self.assertEqual(len(gq.ALL_STATES), 9)
        for state in gq.ALL_STATES:
            self.assertIn(state, gq.LEGAL_TRANSITIONS)


class TestTransitions(QueueCase):
    def test_every_legal_transition_is_accepted(self):
        for state in gq.ALL_STATES:
            for target in sorted(gq.LEGAL_TRANSITIONS[state]):
                with self.subTest(edge="%s->%s" % (state, target)):
                    queue = gq.GninaQueue(self.run_dir,
                                          queue_dir=os.path.join(self.tmp, "q1"))
                    job_dir = make_job_dir(self.run_dir, JOB_A)
                    queue.enqueue(job_dir, job_inputs(job_dir))
                    force_state(queue, JOB_A, state)
                    record = queue.transition(JOB_A, target, why="test")
                    self.assertEqual(record.state, target)
                    self.assertEqual(record.history[-1]["from"], state)
                    self.assertEqual(record.history[-1]["to"], target)
                    import shutil
                    shutil.rmtree(os.path.join(self.tmp, "q1"))

    def test_every_illegal_transition_raises(self):
        """All 81 ordered pairs are checked, not a representative sample."""
        self.enqueue()
        checked = 0
        for state in gq.ALL_STATES:
            for target in gq.ALL_STATES:
                if target in gq.LEGAL_TRANSITIONS[state]:
                    continue
                with self.subTest(edge="%s->%s" % (state, target)):
                    force_state(self.queue, JOB_A, state)
                    with self.assertRaises(gq.IllegalTransition):
                        self.queue.transition(JOB_A, target, why="test")
                    self.assertEqual(self.queue.read(JOB_A).state, state,
                                     "a rejected transition must not be written")
                    checked += 1
        self.assertEqual(checked + sum(len(v) for k, v in
                                       gq.LEGAL_TRANSITIONS.items()
                                       if k != gq.NEW_STATE),
                         len(gq.ALL_STATES) ** 2,
                         "every ordered pair must be either legal or rejected")

    def test_collected_is_terminal(self):
        """The edge that would re-dock finished work."""
        self.enqueue()
        force_state(self.queue, JOB_A, "COLLECTED")
        with self.assertRaises(gq.IllegalTransition):
            self.queue.transition(JOB_A, "PENDING", why="no")

    def test_abandoned_is_terminal(self):
        self.enqueue()
        force_state(self.queue, JOB_A, "ABANDONED")
        for target in gq.ALL_STATES:
            with self.assertRaises(gq.IllegalTransition):
                self.queue.transition(JOB_A, target, why="no")

    def test_unknown_state_is_rejected(self):
        self.enqueue()
        with self.assertRaises(gq.QueueError):
            self.queue.transition(JOB_A, "ALMOST_DONE")

    def test_transition_on_a_missing_record_raises(self):
        with self.assertRaises(gq.QueueError):
            self.queue.transition("nope", "PENDING")

    def test_dict_fields_merge_and_error_replaces(self):
        """A caller recording a remote dir must not wipe the host and port."""
        self.enqueue()
        self.queue.update_fields(JOB_A, remote={"host": "h", "port": 50000})
        self.queue.update_fields(JOB_A, remote={"dir": "/d"})
        record = self.queue.read(JOB_A)
        self.assertEqual(record.remote["host"], "h")
        self.assertEqual(record.remote["dir"], "/d")
        self.queue.update_fields(JOB_A, error=gq.make_error("STALLED", "m", "poll"))
        self.queue.update_fields(JOB_A, error=gq.make_error("EMPTY_OUTPUT", "m2",
                                                            "collect"))
        self.assertEqual(self.queue.read(JOB_A).error["code"], "EMPTY_OUTPUT")
        self.queue.update_fields(JOB_A, error=None)
        self.assertIsNone(self.queue.read(JOB_A).error)

    def test_history_is_capped_but_keeps_the_recent_end(self):
        self.enqueue()
        for _ in range(40):
            self.queue.transition(JOB_A, "SUBMITTING", why="a")
            self.queue.transition(JOB_A, "PENDING", why="b")
        record = self.queue.read(JOB_A)
        self.assertLessEqual(len(record.history), gq.HISTORY_LIMIT)
        self.assertEqual(record.history[-1]["to"], "PENDING")

    def test_update_fields_that_change_nothing_do_not_write(self):
        """The run directory may be on a slow fuseblk mount; churn is not free."""
        self.enqueue()
        path = self.queue.record_path(JOB_A)
        before = os.stat(path).st_mtime_ns
        time.sleep(0.01)
        self.queue.update_fields(JOB_A, attempts=0)
        self.assertEqual(os.stat(path).st_mtime_ns, before)

    def test_history_cannot_be_set_directly(self):
        self.enqueue()
        with self.assertRaises(gq.QueueError):
            self.queue.update_fields(JOB_A, history=[])


# ---------------------------------------------------------------------------
# atomicity
# ---------------------------------------------------------------------------

class TestAtomicWrite(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.path = os.path.join(self.tmp, "record.json")

    def tearDown(self):
        self._tmp.cleanup()

    def _tmp_files(self):
        return [n for n in os.listdir(self.tmp) if ".tmp." in n]

    def test_success_leaves_no_temp_file(self):
        gq.atomic_write_json(self.path, {"a": 1})
        self.assertEqual(self._tmp_files(), [])
        with open(self.path) as handle:
            self.assertEqual(json.load(handle), {"a": 1})

    def test_a_crash_mid_write_leaves_the_old_file_intact_and_no_temp(self):
        """os.replace is the commit point; a failure before it changes nothing."""
        gq.atomic_write_json(self.path, {"generation": 1})
        original = os.replace

        def exploding_replace(src, dst):
            raise OSError(28, "No space left on device")

        gq.os.replace = exploding_replace
        try:
            with self.assertRaises(OSError):
                gq.atomic_write_json(self.path, {"generation": 2})
        finally:
            gq.os.replace = original
        with open(self.path) as handle:
            self.assertEqual(json.load(handle), {"generation": 1},
                             "a failed write must not damage the old record")
        self.assertEqual(self._tmp_files(), [],
                         "a failed write must not leave a temp file behind")

    def test_a_reader_never_observes_a_partial_document(self):
        """200 rewrites under a concurrent reader; every read is a whole record."""
        gq.atomic_write_json(self.path, {"generation": 0})
        seen = []
        errors = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                data = gq.read_json_retry(self.path, attempts=1)
                if data is None:
                    errors.append("unreadable")
                    continue
                if "generation" not in data:
                    errors.append("partial: %r" % data)
                else:
                    seen.append(data["generation"])

        thread = threading.Thread(target=reader)
        thread.start()
        try:
            for generation in range(1, 201):
                gq.atomic_write_json(self.path,
                                     {"generation": generation, "pad": "x" * 4000})
        finally:
            stop.set()
            thread.join(timeout=10)
        self.assertEqual(errors, [], "a reader saw a torn or missing record")
        self.assertTrue(seen, "the reader never ran")
        self.assertEqual(self._tmp_files(), [])

    def test_read_json_retry_tolerates_the_zero_byte_claim_window(self):
        """enqueue claims the name before writing; a reader may land in between."""
        open(self.path, "w").close()
        self.assertIsNone(gq.read_json_retry(self.path, attempts=2, delay=0.001))

    def test_read_json_retry_returns_none_for_a_truncated_record(self):
        with open(self.path, "w") as handle:
            handle.write('{"job_id": "DB1__P1__G__F1__cav1", "sta')
        self.assertIsNone(gq.read_json_retry(self.path, attempts=1))

    def test_read_json_retry_returns_none_for_a_missing_file(self):
        self.assertIsNone(gq.read_json_retry(os.path.join(self.tmp, "nope.json")))


# ---------------------------------------------------------------------------
# enqueue
# ---------------------------------------------------------------------------

class TestEnqueue(QueueCase):
    def test_a_new_job_is_pending(self):
        record, _ = self.enqueue()
        self.assertEqual(record.state, "PENDING")
        self.assertEqual(record.attempts, 0)
        self.assertEqual(record.job_id, JOB_A)

    def test_a_job_whose_outputs_exist_is_enqueued_as_collected(self):
        """Today's skip-if-done behaviour, preserved: it is never re-docked."""
        record, _ = self.enqueue(outputs=True)
        self.assertEqual(record.state, "COLLECTED")
        self.assertTrue(record.outputs["already_complete_at_enqueue"])
        self.assertIsNone(self.queue.next_pending())

    def test_enqueue_is_idempotent_and_does_not_reset_a_live_record(self):
        """Re-running a campaign must not re-dock what is already on the GPU."""
        _, job_dir = self.enqueue()
        force_state(self.queue, JOB_A, "SUBMITTED",
                    remote={"dir": "/remote/x"}, attempts=1)
        again = self.queue.enqueue(job_dir, job_inputs(job_dir))
        self.assertEqual(again.state, "SUBMITTED")
        self.assertEqual(again.attempts, 1)
        self.assertEqual(again.remote["dir"], "/remote/x")

    def test_the_same_id_pointing_at_a_different_directory_is_refused(self):
        _, job_dir = self.enqueue()
        other = os.path.join(self.tmp, "elsewhere", JOB_A)
        os.makedirs(other)
        with self.assertRaises(gq.QueueError):
            self.queue.enqueue(other, job_inputs(job_dir), job_id=JOB_A)

    def test_enqueue_many_reports_what_it_did(self):
        jobs = []
        for job_id, outputs in ((JOB_A, False), (JOB_B, True), (JOB_C, False)):
            job_dir = make_job_dir(self.run_dir, job_id, outputs=outputs)
            entry = {"current_outfolder": job_dir}
            entry.update(job_inputs(job_dir))
            jobs.append(entry)
        stats = self.queue.enqueue_many(jobs, PARAMS)
        self.assertEqual(stats, {"enqueued": 3, "already_present": 0,
                                 "already_complete": 1})
        again = self.queue.enqueue_many(jobs, PARAMS)
        self.assertEqual(again["already_present"], 3)
        self.assertEqual(again["enqueued"], 0)

    def test_enqueue_many_rejects_an_entry_with_no_directory(self):
        with self.assertRaises(gq.QueueError):
            self.queue.enqueue_many([{"receptor_pdb": "/x"}], PARAMS)

    def test_input_paths_are_stored_absolute(self):
        job_dir = make_job_dir(self.run_dir, JOB_A)
        inputs = job_inputs(job_dir)
        inputs["receptor_pdb"] = os.path.relpath(inputs["receptor_pdb"])
        record = self.queue.enqueue(job_dir, inputs)
        self.assertTrue(os.path.isabs(record.inputs["receptor_pdb"]))

    def test_a_record_survives_a_round_trip_with_unknown_keys(self):
        """A newer writer's fields must not be dropped by an older reader."""
        _, _ = self.enqueue()
        path = self.queue.record_path(JOB_A)
        with open(path) as handle:
            data = json.load(handle)
        data["future_field"] = {"added": "by a later version"}
        gq.atomic_write_json(path, data)
        self.queue._cache.clear()
        self.queue.transition(JOB_A, "SUBMITTING", why="test")
        with open(path) as handle:
            self.assertEqual(json.load(handle)["future_field"],
                             {"added": "by a later version"})


class TestParams(QueueCase):
    def test_changing_a_docking_parameter_midway_is_refused(self):
        changed = dict(PARAMS)
        changed["exhaustiveness"] = 8
        with self.assertRaises(gq.QueueError) as ctx:
            self.queue.set_params(changed)
        self.assertIn("exhaustiveness", str(ctx.exception))

    def test_the_change_can_be_allowed_explicitly(self):
        changed = dict(PARAMS)
        changed["exhaustiveness"] = 8
        self.queue.set_params(changed, allow_change=True)
        self.assertEqual(self.queue.get_params()["exhaustiveness"], 8)

    def test_operational_knobs_may_change_freely(self):
        """A longer deadline is not a scientific change."""
        changed = dict(PARAMS)
        changed["job_deadline_seconds"] = 3600.0
        changed["max_attempts"] = 5
        self.queue.set_params(changed)
        self.assertEqual(self.queue.get_params()["job_deadline_seconds"], 3600.0)


# ---------------------------------------------------------------------------
# job-directory predicates
# ---------------------------------------------------------------------------

class TestOutputPredicates(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.run_dir = make_run(self.tmp)
        self.job_dir = make_job_dir(self.run_dir, JOB_A)

    def tearDown(self):
        self._tmp.cleanup()

    def test_complete_outputs_are_recognised(self):
        write_gnina_outputs(self.job_dir, 3)
        self.assertTrue(gq.outputs_complete(self.job_dir))

    def test_complete_does_not_require_complex_pdbs(self):
        """The default cleanup deletes them; requiring them would re-dock the campaign."""
        write_gnina_outputs(self.job_dir, 3)
        for name in os.listdir(os.path.join(self.job_dir, "gnina")):
            if name.startswith("complex_"):
                os.unlink(os.path.join(self.job_dir, "gnina", name))
        self.assertTrue(gq.outputs_complete(self.job_dir))
        ok, _detail = gq.outputs_verified_at_collect(self.job_dir)
        self.assertFalse(ok, "the strict predicate still wants them at collect time")

    def test_a_header_only_results_csv_is_not_a_result(self):
        write_gnina_outputs(self.job_dir, 3)
        with open(os.path.join(self.job_dir, "gnina", "results.csv"), "w") as handle:
            handle.write("Pose,GNINA_Affinity,GNINA_CNN_pose,GNINA_CNN_affinity\n")
        self.assertFalse(gq.outputs_complete(self.job_dir))

    def test_an_empty_out_sdf_is_not_a_result(self):
        write_gnina_outputs(self.job_dir, 3)
        open(os.path.join(self.job_dir, "gnina", "out.sdf"), "w").close()
        self.assertFalse(gq.outputs_complete(self.job_dir))

    def test_a_missing_job_directory_is_not_complete(self):
        self.assertFalse(gq.outputs_complete(os.path.join(self.tmp, "absent")))

    def test_verification_catches_a_pose_count_mismatch(self):
        write_gnina_outputs(self.job_dir, n_poses=5, n_complexes=3)
        ok, detail = gq.outputs_verified_at_collect(self.job_dir)
        self.assertFalse(ok)
        self.assertTrue(any("complex" in r for r in detail["reasons"]))

    def test_verification_catches_a_results_row_mismatch(self):
        write_gnina_outputs(self.job_dir, n_poses=5, n_result_rows=2)
        ok, detail = gq.outputs_verified_at_collect(self.job_dir)
        self.assertFalse(ok)
        self.assertIn(("2 results.csv rows vs 5 complex_*.pdb"), detail["reasons"])

    def test_job_id_is_the_directory_basename(self):
        self.assertEqual(gq.job_id_for("/a/b/%s/" % JOB_A), JOB_A)
        self.assertEqual(gq.job_id_for("/a/b/%s" % JOB_A), JOB_A)


class TestPurge(unittest.TestCase):
    """Stale output is the most dangerous bug in this design."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.run_dir = make_run(self.tmp)
        self.job_dir = make_job_dir(self.run_dir, JOB_A)
        write_gnina_outputs(self.job_dir, 3)
        gnina = os.path.join(self.job_dir, "gnina")
        with open(os.path.join(gnina, "complex_1.pdb.gz"), "wb") as handle:
            handle.write(b"\x1f\x8b")
        # the bystanders
        for tool in ("smina", "gold", "ledock"):
            with open(os.path.join(self.job_dir, tool, "results.csv"), "w") as handle:
                handle.write("Pose,Score\n1,-9.0\n")
            with open(os.path.join(self.job_dir, tool, "complex_1.pdb"), "w") as handle:
                handle.write("ATOM\n")
        with open(os.path.join(self.job_dir, "final_results.csv"), "w") as handle:
            handle.write("consensus_rmsd\n1.2\n")
        with open(os.path.join(gnina, "keepme.txt"), "w") as handle:
            handle.write("not ours to delete\n")

    def tearDown(self):
        self._tmp.cleanup()

    def test_removes_exactly_the_named_patterns(self):
        removed = gq.purge_gnina_outputs(self.job_dir)
        gnina = os.path.join(self.job_dir, "gnina")
        left = sorted(os.listdir(gnina))
        self.assertEqual(left, ["keepme.txt"])
        self.assertEqual(removed, 3 + 3 + 1 + 3)   # out_*, complex_*, .gz, exacts

    def test_touches_nothing_outside_the_gnina_directory(self):
        gq.purge_gnina_outputs(self.job_dir)
        for tool in ("smina", "gold", "ledock"):
            self.assertTrue(os.path.exists(
                os.path.join(self.job_dir, tool, "results.csv")))
            self.assertTrue(os.path.exists(
                os.path.join(self.job_dir, tool, "complex_1.pdb")))
        self.assertTrue(os.path.exists(
            os.path.join(self.job_dir, "final_results.csv")))
        self.assertTrue(os.path.isdir(os.path.join(self.job_dir, "input")))
        self.assertTrue(os.path.isdir(os.path.join(self.job_dir, "gnina")))

    def test_is_a_no_op_when_there_is_nothing_to_purge(self):
        gq.purge_gnina_outputs(self.job_dir)
        self.assertEqual(gq.purge_gnina_outputs(self.job_dir), 0)

    def test_a_job_with_no_gnina_directory_is_fine(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        self.assertEqual(gq.purge_gnina_outputs(empty), 0)


# ---------------------------------------------------------------------------
# the consensus_docker command and environment
# ---------------------------------------------------------------------------

class TestCommandConstruction(QueueCase):
    def setUp(self):
        super(TestCommandConstruction, self).setUp()
        self.record, self.job_dir = self.enqueue()

    def test_skip_rmsd_is_always_present(self):
        """Rule R1. Without it the submit phase writes a gnina-less consensus."""
        cmd = gq.build_consensus_command(self.record, PARAMS)
        self.assertIn("--skip_rmsd", cmd)
        self.assertIn("--use_gnina", cmd)
        self.assertIn("--overwrite", cmd)

    def test_num_modes_is_omitted_when_none(self):
        """Passing it explicitly would silently change consensus_docker's default of 20."""
        self.assertNotIn("--num_modes", gq.build_consensus_command(self.record, PARAMS))

    def test_num_modes_is_passed_when_set(self):
        params = dict(PARAMS)
        params["num_modes"] = 5
        cmd = gq.build_consensus_command(self.record, params)
        self.assertEqual(cmd[cmd.index("--num_modes") + 1], "5")

    def test_the_effective_cpu_is_what_reaches_num_threads(self):
        """24, not the local NUM_THREADS of 60: the remote has 24 cores."""
        cmd = gq.build_consensus_command(self.record, PARAMS)
        self.assertEqual(cmd[cmd.index("--num_threads") + 1], "24")

    def test_optional_inputs_appear_only_when_present(self):
        cmd = gq.build_consensus_command(self.record, PARAMS)
        self.assertNotIn("--receptor_pdbqt", cmd)
        self.assertNotIn("--skip_pdb_to_pdbqt", cmd)
        self.queue.update_fields(JOB_A, inputs={"receptor_pdbqt": "/x/r.pdbqt",
                                                "skip_pdb_to_pdbqt": True})
        cmd = gq.build_consensus_command(self.queue.read(JOB_A), PARAMS)
        self.assertIn("--receptor_pdbqt", cmd)
        self.assertIn("--skip_pdb_to_pdbqt", cmd)

    def test_a_job_with_no_receptor_cannot_produce_a_command(self):
        self.queue.update_fields(JOB_A, inputs={"receptor_pdb": None})
        with self.assertRaises(gq.QueueError):
            gq.build_consensus_command(self.queue.read(JOB_A), PARAMS)

    def test_both_phases_run_the_identical_command(self):
        """Only the environment differs; that is what reproduces the search box."""
        submit = gq.build_consensus_command(self.record, PARAMS)
        collect = gq.build_consensus_command(self.record, PARAMS)
        self.assertEqual(submit, collect)

    def test_the_phase_environment_selects_the_mode(self):
        params = self.queue._runtime_params()
        env = gq.build_phase_env(self.record, params, "submit", base_env={})
        self.assertEqual(env["GNINA_REMOTE_MODE"], "submit")
        self.assertTrue(env["GNINA_REMOTE_TICKET"].endswith(
            "%s.submit.json" % JOB_A))
        self.assertNotIn("GNINA_REMOTE_JOB_DIR", env)

    def test_collect_pins_the_recorded_remote_directory(self):
        """Collection must not depend on re-deriving the content address."""
        self.queue.update_fields(JOB_A, remote={"dir": "/home/onur/gnina_jobs/g_ab"})
        params = self.queue._runtime_params()
        env = gq.build_phase_env(self.queue.read(JOB_A), params, "collect",
                                 base_env={})
        self.assertEqual(env["GNINA_REMOTE_MODE"], "collect")
        self.assertEqual(env["GNINA_REMOTE_JOB_DIR"], "/home/onur/gnina_jobs/g_ab")
        self.assertNotIn("GNINA_REMOTE_LOCK", env)

    def test_an_unknown_phase_is_refused(self):
        with self.assertRaises(gq.QueueError):
            gq.build_phase_env(self.record, PARAMS, "dock", base_env={})


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

class TestSubmit(QueueCase):
    def setUp(self):
        super(TestSubmit, self).setUp()
        self.record, self.job_dir = self.enqueue()

    def test_a_launched_ticket_moves_the_job_to_submitted(self):
        runner = FakeRunner(ticket=launched_ticket(), returncode=3)
        record = gq.submit_job(self.queue, self.record, runner=runner)
        self.assertEqual(record.state, "SUBMITTED")
        self.assertEqual(record.attempts, 1)
        self.assertEqual(record.remote["dir"],
                         "/home/onur/gnina_jobs/gnina_ab12cd34ef567890")
        self.assertEqual(record.remote["job_key"], "ab12cd34ef567890")
        self.assertIsNotNone(record.remote["launched_at"])

    def test_exit_code_three_is_not_treated_as_a_failure(self):
        """The submit phase exits non-zero by design; the ticket is the signal."""
        runner = FakeRunner(ticket=launched_ticket(), returncode=3)
        self.assertEqual(gq.submit_job(self.queue, self.record,
                                       runner=runner).state, "SUBMITTED")

    def test_a_busy_remote_returns_to_pending_without_consuming_an_attempt(self):
        """A long-running neighbour must not abandon the queue."""
        runner = FakeRunner(ticket=busy_ticket(), returncode=5)
        record = gq.submit_job(self.queue, self.record, runner=runner)
        self.assertEqual(record.state, "PENDING")
        self.assertEqual(record.attempts, 0)
        self.assertIsNotNone(record.next_attempt_at)
        self.assertGreater(record.next_attempt_at, time.time())

    def test_repeated_busy_backs_off_further(self):
        runner = FakeRunner(ticket=busy_ticket(), returncode=5)
        first = gq.submit_job(self.queue, self.record, runner=runner)
        self.queue.update_fields(JOB_A, next_attempt_at=None)
        second = gq.submit_job(self.queue, self.queue.read(JOB_A), runner=runner)
        self.assertEqual(second.state, "PENDING")
        self.assertEqual(second.attempts, 0)
        self.assertGreater(second.next_attempt_at - time.time(),
                           first.next_attempt_at - time.time())

    def test_a_resume_hit_goes_straight_to_remote_done(self):
        """Never re-dock work the GPU has already finished."""
        ticket = launched_ticket()
        ticket["outcome"] = "already_done"
        runner = FakeRunner(ticket=ticket, returncode=3)
        record = gq.submit_job(self.queue, self.record, runner=runner)
        self.assertEqual(record.state, "REMOTE_DONE")
        self.assertEqual(record.remote["exit_code"], 0)

    def test_no_ticket_is_a_retriable_failure_with_the_log_tail(self):
        def side_effect(runner, cmd, env, log_path, cwd):
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w") as handle:
                handle.write("Traceback (most recent call last):\nBoom\n")
            return 1, "", ""
        runner = FakeRunner(side_effect=side_effect)
        record = gq.submit_job(self.queue, self.record, runner=runner)
        self.assertEqual(record.state, "FAILED")
        self.assertEqual(record.error["code"], "SUBMIT_NO_TICKET")
        self.assertIn("Boom", record.error["message"])
        self.assertTrue(record.error["retriable"])

    def test_a_submit_that_wrote_results_abandons_the_job_loudly(self):
        """Rule R3: the exit contract or the purge is broken; stop on job #1."""
        def side_effect(runner, cmd, env, log_path, cwd):
            write_gnina_outputs(self.job_dir, 3)
            return 0, "", ""
        runner = FakeRunner(ticket=launched_ticket(), side_effect=side_effect)
        record = gq.submit_job(self.queue, self.record, runner=runner)
        self.assertEqual(record.state, "ABANDONED")
        self.assertEqual(record.error["code"], "SUBMIT_WROTE_RESULTS")
        self.assertFalse(record.error["retriable"])

    def test_a_shim_that_ignores_the_mode_is_named_as_the_likely_cause(self):
        """Today's gnina_remote.py has no submit mode: it would run the job whole.

        That is safe -- nothing is docked twice and no wrong score is written --
        but the abandonment must point straight at the cause, or an integrator
        loses an afternoon to it.
        """
        def side_effect(runner, cmd, env, log_path, cwd):
            write_gnina_outputs(self.job_dir, 3)      # a full synchronous run
            return 0, "", ""
        runner = FakeRunner(ticket=launched_ticket(), side_effect=side_effect)
        record = gq.submit_job(self.queue, self.record, runner=runner)
        self.assertEqual(record.state, "ABANDONED")
        self.assertIn("GNINA_REMOTE_MODE", record.error["message"])
        self.assertIn("GNINA_ASYNC=0", record.error["message"])

    def test_a_broken_contract_is_reported_at_critical(self):
        """R3 exists to catch a consensus_docker change on job #1, not job #5000."""
        def side_effect(runner, cmd, env, log_path, cwd):
            write_gnina_outputs(self.job_dir, 3)
            return 0, "", ""
        runner = FakeRunner(ticket=launched_ticket(), side_effect=side_effect)
        with self.assertLogs(gq.__name__, level="CRITICAL") as captured:
            gq.submit_job(self.queue, self.record, runner=runner)
        self.assertIn("results.csv", "\n".join(captured.output))

    def test_stale_outputs_are_purged_before_the_submit_runs(self):
        """Rule R2: split_mol would otherwise succeed on the previous run's poses."""
        write_gnina_outputs(self.job_dir, 3)
        seen = {}

        def side_effect(runner, cmd, env, log_path, cwd):
            seen["out_sdf"] = os.path.exists(
                os.path.join(self.job_dir, "gnina", "out.sdf"))
            seen["results"] = os.path.exists(
                os.path.join(self.job_dir, "gnina", "results.csv"))
            return 3, "", ""
        runner = FakeRunner(ticket=launched_ticket(), side_effect=side_effect)
        gq.submit_job(self.queue, self.record, runner=runner)
        self.assertFalse(seen["out_sdf"], "a stale out.sdf survived into submit")
        self.assertFalse(seen["results"], "a stale results.csv survived into submit")

    def test_a_missing_input_is_abandoned_not_retried(self):
        os.unlink(os.path.join(self.job_dir, "input", "ligand.sdf"))
        runner = FakeRunner(ticket=launched_ticket())
        record = gq.submit_job(self.queue, self.record, runner=runner)
        self.assertEqual(record.state, "ABANDONED")
        self.assertEqual(record.error["code"], "MISSING_INPUT")
        self.assertEqual(runner.calls, [], "nothing should have been launched")

    def test_a_stale_ticket_from_a_previous_attempt_is_not_believed(self):
        path = gq.ticket_path(self.queue.root, JOB_A, "submit")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(launched_ticket("/remote/from/last/time"), handle)
        runner = FakeRunner(side_effect=lambda *a: (1, "", ""))
        record = gq.submit_job(self.queue, self.record, runner=runner)
        self.assertEqual(record.state, "FAILED")
        self.assertEqual(record.error["code"], "SUBMIT_NO_TICKET")

    def test_only_one_job_may_be_outstanding_on_the_remote(self):
        """One GPU. The pump enforces this too; so does the queue."""
        force_state(self.queue, JOB_A, "SUBMITTED", remote={"dir": "/r/a"})
        job_b = make_job_dir(self.run_dir, JOB_B)
        other = self.queue.enqueue(job_b, job_inputs(job_b))
        runner = FakeRunner(ticket=launched_ticket())
        record = gq.submit_job(self.queue, other, runner=runner)
        self.assertEqual(record.state, "PENDING")
        self.assertEqual(runner.calls, [])

    def test_a_non_pending_record_is_not_submitted(self):
        force_state(self.queue, JOB_A, "REMOTE_DONE")
        runner = FakeRunner(ticket=launched_ticket())
        record = gq.submit_job(self.queue, self.queue.read(JOB_A), runner=runner)
        self.assertEqual(record.state, "REMOTE_DONE")
        self.assertEqual(runner.calls, [])

    def test_submitting_is_written_before_anything_is_launched(self):
        """A crash mid-launch must be resolvable by probing, not by re-launching."""
        observed = {}

        def side_effect(runner, cmd, env, log_path, cwd):
            observed["state"] = self.queue.read(JOB_A).state
            return 3, "", ""
        runner = FakeRunner(ticket=launched_ticket(), side_effect=side_effect)
        gq.submit_job(self.queue, self.record, runner=runner)
        self.assertEqual(observed["state"], "SUBMITTING")

    def test_a_timeout_during_submit_does_not_lose_the_job(self):
        def side_effect(runner, cmd, env, log_path, cwd):
            raise subprocess.TimeoutExpired(cmd, 900)
        runner = FakeRunner(side_effect=side_effect)
        record = gq.submit_job(self.queue, self.record, runner=runner)
        self.assertEqual(record.state, "FAILED")
        self.assertTrue(record.error["retriable"])


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def successful_collect(job_dir, n_poses=3, **kwargs):
    """A fake collect that produces what consensus_docker produces."""
    def side_effect(runner, cmd, env, log_path, cwd):
        write_gnina_outputs(job_dir, n_poses, **kwargs)
        return None                                   # fall through to the ticket
    return side_effect


class TestCollect(QueueCase):
    def setUp(self):
        super(TestCollect, self).setUp()
        self.record, self.job_dir = self.enqueue()
        force_state(self.queue, JOB_A, "REMOTE_DONE",
                    remote={"dir": "/home/onur/gnina_jobs/gnina_ab12",
                            "exit_code": 0, "launched_at": gq._utcnow_iso()},
                    attempts=1)
        self.record = self.queue.read(JOB_A)

    def test_a_verified_collect_is_terminal_success(self):
        runner = FakeRunner(ticket=collected_ticket(),
                            side_effect=successful_collect(self.job_dir, 3))
        record = gq.collect_job(self.queue, self.record, runner=runner)
        self.assertEqual(record.state, "COLLECTED")
        self.assertEqual(record.outputs["n_poses"], 3)
        self.assertEqual(record.outputs["n_complexes"], 3)
        self.assertIsNone(record.error)
        self.assertEqual(runner.calls[0]["mode"], "collect")

    def test_a_results_row_mismatch_fails_and_retries_the_collect_not_the_docking(self):
        """The poses are fine; rdkit/openbabel are not. Never re-dock for that."""
        runner = FakeRunner(ticket=collected_ticket(),
                            side_effect=successful_collect(self.job_dir, 5,
                                                           n_result_rows=2))
        record = gq.collect_job(self.queue, self.record, runner=runner)
        self.assertEqual(record.state, "FAILED")
        self.assertEqual(record.error["code"], "POSTPROCESS_FAILED")
        target, _why = gq.retry_decision(record, max_attempts=3)
        self.assertEqual(target, "REMOTE_DONE")
        retried = gq.apply_retry_policy(self.queue, record, max_attempts=3)
        self.assertEqual(retried.state, "REMOTE_DONE")

    def test_an_empty_out_sdf_goes_back_to_pending_to_be_re_docked(self):
        def side_effect(runner, cmd, env, log_path, cwd):
            gnina = os.path.join(self.job_dir, "gnina")
            os.makedirs(gnina, exist_ok=True)
            open(os.path.join(gnina, "out.sdf"), "w").close()
            return None
        runner = FakeRunner(ticket=collected_ticket(), side_effect=side_effect)
        record = gq.collect_job(self.queue, self.record, runner=runner)
        self.assertEqual(record.state, "FAILED")
        self.assertEqual(record.error["code"], "EMPTY_OUTPUT")
        self.assertEqual(gq.retry_decision(record, 3)[0], "PENDING")

    def test_a_download_failure_returns_to_remote_done(self):
        """The GPU result is still there; only the link failed."""
        runner = FakeRunner(ticket={"outcome": "error", "error": "ssh timeout"},
                            returncode=1)
        record = gq.collect_job(self.queue, self.record, runner=runner)
        self.assertEqual(record.state, "REMOTE_DONE")
        self.assertEqual(record.error["code"], "SSH_TRANSIENT")
        self.assertEqual(record.transient_attempts, 1)
        self.assertEqual(record.attempts, 1, "a flaky link is not a docking attempt")

    def test_a_post_processing_failure_is_distinguished_from_a_download_failure(self):
        runner = FakeRunner(ticket=collected_ticket(), returncode=1)
        record = gq.collect_job(self.queue, self.record, runner=runner)
        self.assertEqual(record.state, "FAILED")
        self.assertEqual(record.error["code"], "POSTPROCESS_FAILED")

    def test_a_vanished_remote_directory_means_re_dock(self):
        runner = FakeRunner(ticket={"outcome": "not_ready"}, returncode=4)
        record = gq.collect_job(self.queue, self.record, runner=runner)
        self.assertEqual(record.state, "FAILED")
        self.assertEqual(record.error["code"], "REMOTE_DIR_GONE")
        self.assertEqual(gq.retry_decision(record, 3)[0], "PENDING")

    def test_a_collect_timeout_keeps_the_remote_result(self):
        def side_effect(runner, cmd, env, log_path, cwd):
            raise subprocess.TimeoutExpired(cmd, 1800)
        runner = FakeRunner(side_effect=side_effect)
        record = gq.collect_job(self.queue, self.record, runner=runner)
        self.assertEqual(record.state, "FAILED")
        self.assertEqual(record.error["code"], "COLLECT_TIMEOUT")
        self.assertEqual(gq.retry_decision(record, 3)[0], "REMOTE_DONE")

    def test_stale_outputs_are_purged_before_the_collect_runs(self):
        write_gnina_outputs(self.job_dir, 9)
        seen = {}

        def side_effect(runner, cmd, env, log_path, cwd):
            seen["poses"] = len(
                [n for n in os.listdir(os.path.join(self.job_dir, "gnina"))])
            write_gnina_outputs(self.job_dir, 3)
            return None
        runner = FakeRunner(ticket=collected_ticket(), side_effect=side_effect)
        record = gq.collect_job(self.queue, self.record, runner=runner)
        self.assertEqual(seen["poses"], 0, "the previous run's poses survived")
        self.assertEqual(record.outputs["n_poses"], 3)

    def test_collect_is_only_entered_from_remote_done(self):
        force_state(self.queue, JOB_A, "PENDING")
        runner = FakeRunner(ticket=collected_ticket())
        record = gq.collect_job(self.queue, self.queue.read(JOB_A), runner=runner)
        self.assertEqual(record.state, "PENDING")
        self.assertEqual(runner.calls, [])

    def test_collecting_records_the_pid_so_a_dead_pump_is_detectable(self):
        observed = {}

        def side_effect(runner, cmd, env, log_path, cwd):
            record = self.queue.read(JOB_A)
            observed["state"] = record.state
            observed["pid"] = record.collect["pid"]
            return None
        runner = FakeRunner(ticket=collected_ticket(), side_effect=side_effect)
        gq.collect_job(self.queue, self.record, runner=runner)
        self.assertEqual(observed["state"], "COLLECTING")
        self.assertEqual(observed["pid"], os.getpid())


# ---------------------------------------------------------------------------
# poll
# ---------------------------------------------------------------------------

class TestPoll(QueueCase):
    def setUp(self):
        super(TestPoll, self).setUp()
        self.record, self.job_dir = self.enqueue()
        force_state(self.queue, JOB_A, "SUBMITTED",
                    remote={"dir": "/r/a", "launched_at": gq._utcnow_iso()},
                    attempts=1, params={"job_deadline_seconds": 1800.0})
        self.record = self.queue.read(JOB_A)

    def test_a_finished_remote_job_becomes_remote_done(self):
        record = gq.poll_job(self.queue, self.record,
                             prober=fake_prober({JOB_A: ("DONE", 0)}))
        self.assertEqual(record.state, "REMOTE_DONE")
        self.assertEqual(record.remote["exit_code"], 0)

    def test_a_live_remote_job_becomes_running(self):
        record = gq.poll_job(self.queue, self.record,
                             prober=fake_prober({JOB_A: ("RUNNING", None)}))
        self.assertEqual(record.state, "RUNNING")

    def test_a_nonzero_remote_exit_fails_the_job(self):
        record = gq.poll_job(self.queue, self.record,
                             prober=fake_prober({JOB_A: ("DONE", 1)}))
        self.assertEqual(record.state, "FAILED")
        self.assertEqual(record.error["code"], "REMOTE_GNINA_FAILED")

    def test_a_vanished_remote_directory_fails_the_job(self):
        record = gq.poll_job(self.queue, self.record,
                             prober=fake_prober({JOB_A: ("GONE", None)}))
        self.assertEqual(record.state, "FAILED")
        self.assertEqual(record.error["code"], "REMOTE_DIR_GONE")

    def test_a_job_past_its_deadline_is_stalled(self):
        old = "2020-01-01T00:00:00.000000Z"
        self.queue.update_fields(JOB_A, remote={"launched_at": old})
        record = gq.poll_job(self.queue, self.queue.read(JOB_A),
                             prober=fake_prober({JOB_A: ("LAUNCHED", None)}))
        self.assertEqual(record.state, "FAILED")
        self.assertEqual(record.error["code"], "STALLED")

    def test_a_job_with_no_launch_time_still_stalls_eventually(self):
        """Without this fallback a lost launch time holds the GPU slot forever."""
        self.queue.update_fields(JOB_A, remote={"launched_at": None})
        record = self.queue.read(JOB_A)
        record.updated_at = "2020-01-01T00:00:00.000000Z"
        gq.atomic_write_json(self.queue.record_path(JOB_A), record.to_dict())
        self.queue._cache.clear()
        record = gq.poll_job(self.queue, self.queue.read(JOB_A),
                             prober=fake_prober({JOB_A: ("LAUNCHED", None)}))
        self.assertEqual(record.state, "FAILED")
        self.assertEqual(record.error["code"], "STALLED")

    def test_a_probe_failure_is_transient_and_does_not_touch_the_job(self):
        """The remote job is detached; a dead link says nothing about it."""
        prober = fake_prober({JOB_A: OSError("connection reset")})
        record = gq.poll_job(self.queue, self.record, prober=prober)
        self.assertEqual(record.state, "SUBMITTED")
        self.assertEqual(record.transient_attempts, 1)
        self.assertEqual(record.attempts, 1)

    def test_repeated_probe_failures_eventually_fail_the_job(self):
        self.queue.update_fields(JOB_A,
                                 transient_attempts=gq.MAX_TRANSIENT_ATTEMPTS)
        prober = fake_prober({JOB_A: OSError("connection reset")})
        record = gq.poll_job(self.queue, self.queue.read(JOB_A), prober=prober)
        self.assertEqual(record.state, "FAILED")
        self.assertEqual(record.error["code"], "SSH_TRANSIENT")

    def test_states_that_are_not_pollable_are_left_alone(self):
        for state in ("PENDING", "SUBMITTING", "REMOTE_DONE", "COLLECTING"):
            force_state(self.queue, JOB_A, state)
            record = gq.poll_job(self.queue, self.queue.read(JOB_A),
                                 prober=fake_prober({JOB_A: ("DONE", 0)}))
            self.assertEqual(record.state, state)

    def test_a_job_with_no_remote_directory_is_not_polled(self):
        self.queue.update_fields(JOB_A, remote={"dir": None})
        called = []

        def prober(record):
            called.append(record.job_id)
            return ("DONE", 0)
        gq.poll_job(self.queue, self.queue.read(JOB_A), prober=prober)
        self.assertEqual(called, [])


class TestProbeParsing(unittest.TestCase):
    def test_the_four_states_parse(self):
        self.assertEqual(gq.parse_probe_state("GONE"), ("GONE", None))
        self.assertEqual(gq.parse_probe_state("LAUNCHED"), ("LAUNCHED", None))
        self.assertEqual(gq.parse_probe_state("RUNNING"), ("RUNNING", None))
        self.assertEqual(gq.parse_probe_state("DONE 0"), ("DONE", 0))
        self.assertEqual(gq.parse_probe_state("DONE 137"), ("DONE", 137))

    def test_a_login_banner_before_the_answer_is_ignored(self):
        self.assertEqual(gq.parse_probe_state("Welcome to Ubuntu\nDONE 0"),
                         ("DONE", 0))

    def test_unrecognised_output_never_reads_as_gone(self):
        """"Gone" would re-dock a job that is still running. Unknown means wait."""
        for text in ("", "   ", "sh: not found", "DONE", "DONE nonsense"):
            state, _code = gq.parse_probe_state(text)
            self.assertIn(state, ("LAUNCHED", "RUNNING"))
            self.assertNotEqual(state, "GONE")

    def test_the_probe_command_names_the_directory(self):
        cmd = gq.build_probe_command("/home/onur/gnina_jobs/gnina_ab12")
        self.assertIn("exit_code", cmd)
        self.assertIn("run.log", cmd)
        self.assertIn("gnina_ab12", cmd)


# ---------------------------------------------------------------------------
# reconciliation
# ---------------------------------------------------------------------------

class TestReconcile(QueueCase):
    """Crash recovery: disk first, remote second, record last."""

    def setUp(self):
        super(TestReconcile, self).setUp()
        self.record, self.job_dir = self.enqueue()

    def test_rule1_complete_outputs_beat_every_recorded_state(self):
        """The rule that makes a lost queue cost a re-scan, not GPU-months."""
        for state in ("PENDING", "SUBMITTING", "SUBMITTED", "RUNNING",
                      "REMOTE_DONE", "COLLECTING", "FAILED"):
            with self.subTest(state=state):
                force_state(self.queue, JOB_A, state, remote={"dir": "/r/a"},
                            collect={"pid": os.getpid(), "host": gq._hostname()})
                write_gnina_outputs(self.job_dir, 3)
                changed = self.queue.reconcile(
                    prober=fake_prober({JOB_A: ("GONE", None)}))
                self.assertEqual(self.queue.read(JOB_A).state, "COLLECTED")
                self.assertEqual(changed[JOB_A], (state, "COLLECTED"))
                gq.purge_gnina_outputs(self.job_dir)

    def test_rule2_an_unreadable_record_is_re_derived_as_pending(self):
        """The zero-byte remains of a crash between claiming a name and writing it."""
        open(self.queue.record_path(JOB_A), "w").close()
        self.queue._cache.clear()
        changed = self.queue.reconcile(prober=fake_prober({}))
        self.assertEqual(self.queue.read(JOB_A).state, "PENDING")
        self.assertEqual(changed[JOB_A][1], "PENDING")

    def test_rule2_leaves_a_record_alone_when_its_job_directory_is_gone(self):
        """Guessing what job a truncated record was would be worse than waiting."""
        import shutil
        shutil.rmtree(self.job_dir)
        open(self.queue.record_path(JOB_A), "w").close()
        self.queue._cache.clear()
        self.queue.reconcile(prober=fake_prober({}))
        self.assertIsNone(self.queue.read(JOB_A))

    def test_rule3_a_collect_whose_process_died_returns_to_remote_done(self):
        force_state(self.queue, JOB_A, "COLLECTING",
                    collect={"pid": 999999, "host": gq._hostname()},
                    remote={"dir": "/r/a"})
        self.queue.reconcile(prober=fake_prober({JOB_A: ("DONE", 0)}))
        self.assertEqual(self.queue.read(JOB_A).state, "REMOTE_DONE")

    def test_rule3_a_live_collect_is_left_running(self):
        force_state(self.queue, JOB_A, "COLLECTING",
                    collect={"pid": os.getpid(), "host": gq._hostname()},
                    remote={"dir": "/r/a"})
        self.queue.reconcile(prober=fake_prober({JOB_A: ("DONE", 0)}))
        self.assertEqual(self.queue.read(JOB_A).state, "COLLECTING")

    def test_rule4_an_intent_with_no_remote_directory_returns_to_pending(self):
        for state in ("SUBMITTING", "SUBMITTED", "RUNNING"):
            with self.subTest(state=state):
                force_state(self.queue, JOB_A, state, remote={"dir": None})
                self.queue.reconcile(prober=fake_prober({}))
                self.assertEqual(self.queue.read(JOB_A).state, "PENDING")

    def test_rule5_probes_the_recorded_directory(self):
        cases = {
            ("DONE", 0): "REMOTE_DONE",
            ("DONE", 1): "FAILED",
            ("RUNNING", None): "RUNNING",
            ("LAUNCHED", None): "SUBMITTED",
            ("GONE", None): "PENDING",
        }
        for answer, expected in cases.items():
            with self.subTest(answer=answer):
                force_state(self.queue, JOB_A, "SUBMITTING",
                            remote={"dir": "/r/a", "launched_at": gq._utcnow_iso()},
                            error=None)
                self.queue.reconcile(prober=fake_prober({JOB_A: answer}))
                self.assertEqual(self.queue.read(JOB_A).state, expected)

    def test_rule6_a_pending_job_with_a_finished_remote_is_never_re_docked(self):
        force_state(self.queue, JOB_A, "PENDING", remote={"dir": "/r/a"})
        self.queue.reconcile(prober=fake_prober({JOB_A: ("DONE", 0)}))
        self.assertEqual(self.queue.read(JOB_A).state, "REMOTE_DONE")

    def test_rule6_leaves_a_pending_job_pending_when_the_remote_is_not_done(self):
        force_state(self.queue, JOB_A, "PENDING", remote={"dir": "/r/a"})
        self.queue.reconcile(prober=fake_prober({JOB_A: ("GONE", None)}))
        self.assertEqual(self.queue.read(JOB_A).state, "PENDING")

    def test_rule7_failed_is_left_to_the_retry_policy(self):
        force_state(self.queue, JOB_A, "FAILED",
                    error=gq.make_error("STALLED", "m", "poll"))
        self.queue.reconcile(prober=fake_prober({}))
        self.assertEqual(self.queue.read(JOB_A).state, "FAILED")

    def test_terminal_records_are_never_touched(self):
        for state in ("COLLECTED", "ABANDONED"):
            force_state(self.queue, JOB_A, state)
            self.queue.reconcile(prober=fake_prober({JOB_A: ("DONE", 0)}))
            self.assertEqual(self.queue.read(JOB_A).state, state)

    def test_a_probe_failure_during_reconcile_leaves_the_record_alone(self):
        force_state(self.queue, JOB_A, "SUBMITTED", remote={"dir": "/r/a"})
        self.queue.reconcile(prober=fake_prober({JOB_A: OSError("no route to host")}))
        self.assertEqual(self.queue.read(JOB_A).state, "SUBMITTED")

    def test_reconcile_never_double_submits_a_running_job(self):
        """The whole point: a restart finds the running job, it does not relaunch."""
        force_state(self.queue, JOB_A, "SUBMITTED",
                    remote={"dir": "/r/a", "launched_at": gq._utcnow_iso()})
        fresh = gq.GninaQueue(self.run_dir)
        fresh.reconcile(prober=fake_prober({JOB_A: ("RUNNING", None)}))
        self.assertEqual(fresh.read(JOB_A).state, "RUNNING")
        self.assertIsNone(fresh.next_pending(),
                          "a running job must not be offered for submission")
        self.assertIsNotNone(fresh.active_remote())


# ---------------------------------------------------------------------------
# scheduling
# ---------------------------------------------------------------------------

class TestScheduling(QueueCase):
    def test_next_pending_respects_the_backoff(self):
        record, _ = self.enqueue()
        self.queue.update_fields(JOB_A, next_attempt_at=1000.0)
        self.assertIsNone(self.queue.next_pending(now=999.0))
        self.assertIsNotNone(self.queue.next_pending(now=1001.0))

    def test_a_clock_domain_mismatch_is_detected_instead_of_stalling(self):
        """time.monotonic vs time.time would strand every backed-off job forever.

        The pump injects a clock; if the queue and the caller ever disagree about
        which one, a busy GPU (a constant condition with one GPU) would push
        every job past a deadline that never arrives, silently, for the rest of
        the campaign. Detected and overridden, loudly.
        """
        self.enqueue()
        self.queue.update_fields(JOB_A, next_attempt_at=time.time() + 5.0)
        monotonic_ish = 12345.0                    # uptime, not epoch
        with self.assertLogs(gq.__name__, level="ERROR") as captured:
            record = self.queue.next_pending(now=monotonic_ish)
        self.assertIsNotNone(record, "the job must not be stranded")
        self.assertIn("clock domain", "\n".join(captured.output))

    def test_a_legitimate_backoff_is_still_respected(self):
        self.enqueue()
        self.queue.update_fields(JOB_A, next_attempt_at=1060.0)
        self.assertIsNone(self.queue.next_pending(now=1000.0))
        self.assertIsNotNone(self.queue.next_pending(now=1061.0))

    def test_the_queue_exposes_its_own_clock(self):
        clock = FakeClock(500.0)
        queue = gq.GninaQueue(self.run_dir, queue_dir=os.path.join(self.tmp, "qc"),
                              clock=clock)
        self.assertEqual(queue.now(), 500.0)

    def test_next_pending_is_oldest_first(self):
        for job_id in (JOB_B, JOB_A):
            job_dir = make_job_dir(self.run_dir, job_id)
            self.queue.enqueue(job_dir, job_inputs(job_dir))
            time.sleep(0.002)
        self.assertEqual(self.queue.next_pending().job_id, JOB_B)

    def test_active_remote_finds_the_one_outstanding_job(self):
        self.enqueue(JOB_A)
        self.enqueue(JOB_B)
        self.assertIsNone(self.queue.active_remote())
        force_state(self.queue, JOB_B, "RUNNING")
        self.assertEqual(self.queue.active_remote().job_id, JOB_B)

    def test_counts_cover_every_state(self):
        self.enqueue()
        counts = self.queue.counts()
        self.assertEqual(set(counts), set(gq.ALL_STATES))
        self.assertEqual(counts["PENDING"], 1)

    def test_read_all_re_reads_only_what_changed(self):
        """The barrier polls every 15 s; re-parsing 10^4 files each time is a busy loop."""
        self.enqueue(JOB_A)
        self.enqueue(JOB_B)
        self.queue.read_all()
        reads = []
        original = gq.read_json_retry

        def counting_read(path, *args, **kwargs):
            reads.append(os.path.basename(path))
            return original(path, *args, **kwargs)

        gq.read_json_retry = counting_read
        try:
            self.queue.read_all()
            self.assertEqual(reads, [], "nothing changed, nothing should be re-read")
            gq.read_json_retry = original
            self.queue.transition(JOB_A, "SUBMITTING", why="x")
            gq.read_json_retry = counting_read
            reads[:] = []
            self.queue.read_all()
            self.assertEqual(reads, ["%s.json" % JOB_A],
                             "only the changed record should be re-read")
        finally:
            gq.read_json_retry = original


class TestRetryPolicy(QueueCase):
    def setUp(self):
        super(TestRetryPolicy, self).setUp()
        self.record, self.job_dir = self.enqueue()

    def _failed(self, code, attempts=1, count=1, transient=0):
        force_state(self.queue, JOB_A, "FAILED", attempts=attempts,
                    transient_attempts=transient,
                    error=gq.make_error(code, "m", "poll"),
                    error_counts={code: count})
        return self.queue.read(JOB_A)

    def test_a_permanent_error_is_abandoned_immediately(self):
        record = self._failed("MISSING_INPUT")
        self.assertIsNone(gq.retry_decision(record, 3)[0])
        self.assertEqual(gq.apply_retry_policy(self.queue, record, 3).state,
                         "ABANDONED")

    def test_exhausted_attempts_are_abandoned(self):
        record = self._failed("STALLED", attempts=3)
        self.assertIsNone(gq.retry_decision(record, 3)[0])

    def test_one_retry_errors_abandon_on_the_second_occurrence(self):
        """A second identical remote failure is about the inputs, not the link."""
        self.assertEqual(gq.retry_decision(self._failed("REMOTE_GNINA_FAILED",
                                                        count=1), 3)[0], "PENDING")
        self.assertIsNone(gq.retry_decision(self._failed("REMOTE_GNINA_FAILED",
                                                         count=2), 3)[0])

    def test_a_retry_gets_a_backoff(self):
        record = self._failed("STALLED", attempts=2)
        retried = gq.apply_retry_policy(self.queue, record, 3, now=1000.0)
        self.assertEqual(retried.state, "PENDING")
        self.assertGreater(retried.next_attempt_at, 1000.0)

    def test_the_retriable_table_matches_the_failure_taxonomy(self):
        for code in ("SSH_TRANSIENT", "REMOTE_GNINA_FAILED", "STALLED",
                     "REMOTE_DIR_GONE", "EMPTY_OUTPUT", "POSTPROCESS_FAILED",
                     "COLLECT_TIMEOUT", "SUBMIT_NO_TICKET"):
            self.assertIn(code, gq.RETRIABLE_ERRORS)
        for code in ("MISSING_INPUT", "SUBMIT_WROTE_RESULTS", "CAPS_FATAL",
                     "PUMP_UNAVAILABLE"):
            self.assertNotIn(code, gq.RETRIABLE_ERRORS)

    def test_only_failed_records_are_retried(self):
        force_state(self.queue, JOB_A, "RUNNING")
        record = self.queue.read(JOB_A)
        self.assertEqual(gq.apply_retry_policy(self.queue, record, 3).state,
                         "RUNNING")


# ---------------------------------------------------------------------------
# the pump lock
# ---------------------------------------------------------------------------

class TestPumpLock(QueueCase):
    def test_a_second_pump_cannot_take_the_lock(self):
        """Two pumps would double-book the single GPU."""
        first = self.queue.acquire_pump_lock()
        self.assertIsNotNone(first)
        try:
            other = gq.GninaQueue(self.run_dir)
            self.assertIsNone(other.acquire_pump_lock())
        finally:
            first.release()

    def test_the_lock_is_reusable_after_release(self):
        first = self.queue.acquire_pump_lock()
        first.release()
        second = gq.GninaQueue(self.run_dir).acquire_pump_lock()
        self.assertIsNotNone(second)
        second.release()

    def test_release_is_idempotent(self):
        lock = self.queue.acquire_pump_lock()
        lock.release()
        lock.release()

    def test_it_works_as_a_context_manager(self):
        with self.queue.acquire_pump_lock() as lock:
            self.assertIsNotNone(lock)
            self.assertFalse(self.queue.pump_lock_is_free())
        self.assertTrue(self.queue.pump_lock_is_free())

    def test_probing_the_lock_does_not_steal_it(self):
        """The barrier must be able to ask without making a starting pump lose."""
        lock = self.queue.acquire_pump_lock()
        try:
            for _ in range(5):
                self.assertFalse(gq.GninaQueue(self.run_dir).pump_lock_is_free())
            self.assertIsNone(gq.GninaQueue(self.run_dir).acquire_pump_lock())
        finally:
            lock.release()

    def test_a_free_lock_reads_as_free(self):
        self.assertTrue(self.queue.pump_lock_is_free())


class TestStaleLockRecovery(QueueCase):
    """A crashed pump must not deadlock the campaign; a live one must not be evicted."""

    def setUp(self):
        super(TestStaleLockRecovery, self).setUp()
        self._fcntl = gq.fcntl
        gq.fcntl = None                    # exercise the no-flock fallback path

    def tearDown(self):
        gq.fcntl = self._fcntl
        super(TestStaleLockRecovery, self).tearDown()

    def _write_lock(self, body, age_seconds=0.0):
        with open(self.queue.pump_lock_path, "w") as handle:
            json.dump(body, handle)
        if age_seconds:
            when = time.time() - age_seconds
            os.utime(self.queue.pump_lock_path, (when, when))

    def test_a_dead_holder_on_this_host_is_evicted_at_once(self):
        self._write_lock({"pid": 999999, "host": gq._hostname(),
                          "started_at": gq._utcnow_iso()})
        lock = self.queue.acquire_pump_lock()
        self.assertIsNotNone(lock, "a crashed pump must not deadlock the campaign")
        lock.release()

    def test_breaking_a_lock_is_never_silent(self):
        """It is the one operation that could put two gnina jobs on one GPU."""
        self._write_lock({"pid": 999999, "host": gq._hostname()})
        with self.assertLogs(gq.__name__, level="WARNING") as captured:
            lock = self.queue.acquire_pump_lock()
        lock.release()
        joined = "\n".join(captured.output)
        self.assertIn("breaking a stale gnina pump lock", joined)
        self.assertIn("999999", joined, "the previous owner must be named")

    def test_a_live_holder_on_this_host_is_never_evicted(self):
        self._write_lock({"pid": os.getpid(), "host": gq._hostname(),
                          "started_at": gq._utcnow_iso()})
        self.assertIsNone(self.queue.acquire_pump_lock())

    def test_a_cleanly_released_lock_is_taken_immediately(self):
        self._write_lock({"pid": 999999, "host": "other-host", "released": True})
        lock = self.queue.acquire_pump_lock()
        self.assertIsNotNone(lock)
        lock.release()

    def test_a_fresh_foreign_lock_is_respected(self):
        self._write_lock({"pid": 4242, "host": "some-other-machine",
                          "started_at": gq._utcnow_iso()})
        self.assertIsNone(self.queue.acquire_pump_lock())

    def test_an_ancient_foreign_lock_is_eventually_broken(self):
        self._write_lock({"pid": 4242, "host": "some-other-machine"},
                         age_seconds=gq.PUMP_LOCK_STALE_SECONDS + 60)
        lock = self.queue.acquire_pump_lock()
        self.assertIsNotNone(lock, "an abandoned foreign lock must not be forever")
        lock.release()

    def test_a_truncated_lock_body_is_judged_by_age(self):
        open(self.queue.pump_lock_path, "w").close()
        self.assertIsNone(self.queue.acquire_pump_lock())
        when = time.time() - (gq.PUMP_LOCK_STALE_SECONDS + 60)
        os.utime(self.queue.pump_lock_path, (when, when))
        lock = self.queue.acquire_pump_lock()
        self.assertIsNotNone(lock)
        lock.release()

    def test_the_decision_function_explains_itself(self):
        breakable, why = gq._lock_is_breakable(
            self.queue.pump_lock_path,
            {"pid": os.getpid(), "host": gq._hostname()}, 3600.0, time.time())
        self.assertFalse(breakable)
        self.assertIn("live pid", why)

    def test_the_fallback_lock_file_is_removed_on_release(self):
        lock = self.queue.acquire_pump_lock()
        self.assertTrue(os.path.exists(self.queue.pump_lock_path))
        lock.release()
        self.assertFalse(os.path.exists(self.queue.pump_lock_path))


class TestFlockCrashRecovery(QueueCase):
    """With flock, the kernel cleans up after a killed pump. Prove it end to end."""

    def test_a_killed_pump_leaves_the_lock_available(self):
        if gq.fcntl is None:
            self.skipTest("flock is unavailable on this platform")
        script = (
            "import sys, time\n"
            "sys.path.insert(0, %r)\n"
            "from dock_selectivity import gnina_queue as gq\n"
            "q = gq.GninaQueue(%r)\n"
            "lock = q.acquire_pump_lock()\n"
            "print('ACQUIRED' if lock else 'DENIED', flush=True)\n"
            "time.sleep(60)\n" % (REPO, self.run_dir)
        )
        child = subprocess.Popen([sys.executable, "-c", script],
                                 stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), "ACQUIRED")
            self.assertIsNone(self.queue.acquire_pump_lock(),
                              "the child holds the lock")
            child.kill()
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
            child.stdout.close()
        deadline = time.time() + 10
        lock = None
        while time.time() < deadline and lock is None:
            lock = self.queue.acquire_pump_lock()
            if lock is None:
                time.sleep(0.1)
        self.assertIsNotNone(lock, "a SIGKILLed pump must not hold the lock forever")
        lock.release()


# ---------------------------------------------------------------------------
# two processes
# ---------------------------------------------------------------------------

CHILD_ENQUEUE = """
import os, sys, time
sys.path.insert(0, {repo!r})
from dock_selectivity import gnina_queue as gq

run_dir = {run_dir!r}
job_ids = {job_ids!r}
go = {go!r}
while not os.path.exists(go):
    time.sleep(0.005)
queue = gq.GninaQueue(run_dir)
for job_id in job_ids:
    job_dir = os.path.join(run_dir, "docking_results", job_id)
    queue.enqueue(job_dir, {{"receptor_pdb": os.path.join(job_dir, "input", "receptor.pdb"),
                            "ligand_sdf": os.path.join(job_dir, "input", "ligand.sdf"),
                            "pocket_pdb": os.path.join(job_dir, "input", "pocket.pdb")}})
print("DONE", flush=True)
"""

CHILD_LOCK = """
import sys, time
sys.path.insert(0, {repo!r})
from dock_selectivity import gnina_queue as gq
queue = gq.GninaQueue({run_dir!r})
lock = queue.acquire_pump_lock()
print("ACQUIRED" if lock else "DENIED", flush=True)
time.sleep({hold})
"""


class TestTwoProcesses(QueueCase):
    """The store is written by a pump and read by a barrier, concurrently."""

    def test_two_processes_enqueueing_the_same_jobs_do_not_corrupt_the_store(self):
        job_ids = ["DB%05d__P30518__AVPR2__F1__cav1" % i for i in range(40)]
        for job_id in job_ids:
            make_job_dir(self.run_dir, job_id)
        go = os.path.join(self.tmp, "GO")
        children = []
        for _ in range(2):
            script = CHILD_ENQUEUE.format(repo=REPO, run_dir=self.run_dir,
                                          job_ids=job_ids, go=go)
            children.append(subprocess.Popen([sys.executable, "-c", script],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, text=True))
        open(go, "w").close()
        # Read the store the whole time the two writers are working. A reader
        # takes no lock at all, which is only safe because every write commits
        # with os.replace.
        partial = []
        observed = 0
        while any(c.poll() is None for c in children):
            for record in self.queue.read_all():
                observed += 1
                if record.state not in gq.ALL_STATES or not record.job_dir:
                    partial.append(record.job_id)
            time.sleep(0.005)
        for child in children:
            out, err = child.communicate(timeout=30)
            self.assertEqual(child.returncode, 0, err)
            self.assertIn("DONE", out)
        self.assertEqual(partial, [], "a reader observed an incomplete record")
        self.assertGreater(observed, 0, "the concurrent reader never ran")
        records = self.queue.read_all()
        self.assertEqual(len(records), len(job_ids))
        self.assertEqual(sorted(r.job_id for r in records), sorted(job_ids))
        for record in records:
            self.assertEqual(record.state, "PENDING")
        leftovers = [n for n in os.listdir(self.queue.jobs_dir) if ".tmp." in n]
        self.assertEqual(leftovers, [], "temp files were left behind")

    def test_only_one_of_two_processes_gets_the_pump_lock(self):
        script = CHILD_LOCK.format(repo=REPO, run_dir=self.run_dir, hold=8)
        children = [subprocess.Popen([sys.executable, "-c", script],
                                     stdout=subprocess.PIPE, text=True)
                    for _ in range(2)]
        try:
            answers = sorted(c.stdout.readline().strip() for c in children)
            self.assertEqual(answers, ["ACQUIRED", "DENIED"],
                             "exactly one pump may hold the lock")
        finally:
            for child in children:
                child.kill()
                child.wait(timeout=10)
                child.stdout.close()


# ---------------------------------------------------------------------------
# never re-dock finished work
# ---------------------------------------------------------------------------

class TestThreadSafety(QueueCase):
    """The pump polls on the main thread while a pool collects. One object, many threads."""

    def test_concurrent_readers_and_disjoint_writers_stay_consistent(self):
        job_ids = ["DB%05d__P30518__AVPR2__F1__cav1" % i for i in range(24)]
        for job_id in job_ids:
            job_dir = make_job_dir(self.run_dir, job_id)
            self.queue.enqueue(job_dir, job_inputs(job_dir))
        errors = []
        stop = threading.Event()

        def reader():
            try:
                while not stop.is_set():
                    for record in self.queue.read_all():
                        if record.state not in gq.ALL_STATES:
                            errors.append("bad state %r" % record.state)
            except BaseException as exc:                       # noqa: BLE001
                errors.append("reader: %r" % (exc,))

        def writer(mine):
            # Each collect thread owns one job, so writers never share a record.
            try:
                for _ in range(6):
                    for job_id in mine:
                        self.queue.transition(job_id, "SUBMITTING", why="t")
                        self.queue.transition(job_id, "PENDING", why="t")
            except BaseException as exc:                       # noqa: BLE001
                errors.append("writer: %r" % (exc,))

        readers = [threading.Thread(target=reader) for _ in range(2)]
        writers = [threading.Thread(target=writer, args=(job_ids[i::3],))
                   for i in range(3)]
        for thread in readers + writers:
            thread.start()
        for thread in writers:
            thread.join(timeout=60)
        stop.set()
        for thread in readers:
            thread.join(timeout=30)
        self.assertEqual(errors, [])
        records = self.queue.read_all()
        self.assertEqual(len(records), len(job_ids))
        for record in records:
            self.assertEqual(record.state, "PENDING")

    def test_a_rewrite_is_never_missed_by_the_cache(self):
        """os.replace installs a new inode, so a coarse mtime cannot hide a write."""
        record, _ = self.enqueue()
        self.queue.read_all()
        path = self.queue.record_path(JOB_A)
        data = gq.read_json_retry(path)
        data["state"] = "SUBMITTING"
        gq.atomic_write_json(path, data)          # same size class, same instant
        self.assertEqual(self.queue.read_all()[0].state, "SUBMITTING")


class TestFinishedWorkIsNeverRepeated(QueueCase):
    """The single most expensive mistake this design can make."""

    def test_a_completed_job_is_never_offered_for_submission(self):
        self.enqueue(JOB_A, outputs=True)
        self.enqueue(JOB_B)
        self.assertEqual(self.queue.next_pending().job_id, JOB_B)

    def test_a_second_campaign_run_adopts_finished_jobs_instead_of_re_docking(self):
        """The realistic case: batch_dock is re-run over a half-finished campaign."""
        for job_id, done in ((JOB_A, True), (JOB_B, False), (JOB_C, True)):
            make_job_dir(self.run_dir, job_id, outputs=done)
        jobs = []
        for job_id in (JOB_A, JOB_B, JOB_C):
            job_dir = os.path.join(self.run_dir, "docking_results", job_id)
            entry = {"current_outfolder": job_dir}
            entry.update(job_inputs(job_dir))
            jobs.append(entry)
        stats = self.queue.enqueue_many(jobs, PARAMS)
        self.assertEqual(stats["already_complete"], 2)
        pending = [r.job_id for r in self.queue.by_state("PENDING")]
        self.assertEqual(pending, [JOB_B])

    def test_a_job_finished_by_another_process_is_adopted_not_re_docked(self):
        """Someone ran batch_dock in sync mode while the pump was thinking."""
        record, job_dir = self.enqueue()
        write_gnina_outputs(job_dir, 3)
        self.queue.reconcile(prober=fake_prober({}))
        self.assertEqual(self.queue.read(JOB_A).state, "COLLECTED")
        self.assertIsNone(self.queue.next_pending())

    def test_a_resubmitted_job_reuses_its_content_addressed_remote_directory(self):
        """gnina_remote's job key is the resume mechanism; the record carries it."""
        record, _ = self.enqueue()
        runner = FakeRunner(ticket=launched_ticket(), returncode=3)
        record = gq.submit_job(self.queue, record, runner=runner)
        remote_dir = record.remote["dir"]
        gq.poll_job(self.queue, record, prober=fake_prober({JOB_A: ("DONE", 0)}))
        record = self.queue.read(JOB_A)
        self.assertEqual(record.state, "REMOTE_DONE")
        params = self.queue._runtime_params()
        env = gq.build_phase_env(record, params, "collect", base_env={})
        self.assertEqual(env["GNINA_REMOTE_JOB_DIR"], remote_dir)

    def test_a_restart_mid_flight_collects_rather_than_re_docks(self):
        """The end-to-end crash story, with a brand-new queue object."""
        record, job_dir = self.enqueue()
        runner = FakeRunner(ticket=launched_ticket(), returncode=3)
        gq.submit_job(self.queue, record, runner=runner)
        # ... the machine dies here ...
        fresh = gq.GninaQueue(self.run_dir)
        fresh.reconcile(prober=fake_prober({JOB_A: ("DONE", 0)}))
        record = fresh.read(JOB_A)
        self.assertEqual(record.state, "REMOTE_DONE")
        self.assertEqual(record.attempts, 1, "the crash must not cost an attempt")
        collector = FakeRunner(ticket=collected_ticket(),
                               side_effect=successful_collect(job_dir, 3))
        record = gq.collect_job(fresh, record, runner=collector)
        self.assertEqual(record.state, "COLLECTED")
        self.assertEqual(len(runner.calls), 1, "the job was docked exactly once")


# ---------------------------------------------------------------------------
# the barrier
# ---------------------------------------------------------------------------

class FakeClock(object):
    """A clock the sleeper advances, so a 3600 s wait costs no wall time."""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class TestBarrier(QueueCase):
    def setUp(self):
        super(TestBarrier, self).setUp()
        self.clock = FakeClock()

    def _wait(self, **kwargs):
        kwargs.setdefault("clock", self.clock)
        kwargs.setdefault("sleeper", self.clock.sleep)
        kwargs.setdefault("poll", 15.0)
        return gq.wait_for_completion(self.run_dir, **kwargs)

    def test_a_drained_queue_returns_at_once(self):
        self.enqueue(JOB_A, outputs=True)
        summary = self._wait()
        self.assertEqual(summary["outstanding"], 0)
        self.assertEqual(summary["collected"], 1)
        self.assertFalse(summary["timed_out"])

    def test_an_absent_queue_is_not_an_error(self):
        summary = gq.wait_for_completion(os.path.join(self.tmp, "no-such-run"))
        self.assertEqual(summary["total"], 0)
        self.assertFalse(summary["timed_out"])

    def test_a_timeout_reports_the_outstanding_jobs_and_proceeds(self):
        """A partial run must produce a usable dataset, with the gaps named."""
        self.enqueue(JOB_A)
        self.enqueue(JOB_B, outputs=True)
        lock = self.queue.acquire_pump_lock()       # a pump is alive, just slow
        try:
            summary = self._wait(timeout=60.0)
        finally:
            lock.release()
        self.assertTrue(summary["timed_out"])
        self.assertEqual(summary["outstanding"], 1)
        self.assertIn(JOB_A, summary["outstanding_job_ids"])
        self.assertTrue(os.path.exists(self.queue.summary_path))
        with open(self.queue.summary_path) as handle:
            written = json.load(handle)
        self.assertIn(JOB_A, written["outstanding_job_ids"])

    def test_timeout_zero_does_not_wait(self):
        self.enqueue(JOB_A)
        lock = self.queue.acquire_pump_lock()
        try:
            summary = self._wait(timeout=0.0)
        finally:
            lock.release()
        self.assertEqual(self.clock.now, 1000.0, "it must not have slept")
        self.assertEqual(summary["outstanding"], 1)

    def test_a_dead_collector_is_restarted_not_waited_on(self):
        """A silently dead pump must never look like a slow one."""
        self.enqueue(JOB_A)
        restarts = []
        summary = self._wait(timeout=600.0, max_restarts=2,
                             restart_collector=lambda: restarts.append(1))
        self.assertEqual(len(restarts), 2)
        self.assertTrue(summary["timed_out"])
        self.assertEqual(summary["collector_restarts"], 2)

    def test_the_barrier_gives_up_and_abandons_failed_jobs(self):
        self.enqueue(JOB_A)
        force_state(self.queue, JOB_A, "FAILED",
                    error=gq.make_error("STALLED", "m", "poll"))
        summary = self._wait(timeout=600.0, max_restarts=0,
                             restart_collector=lambda: None)
        record = self.queue.read(JOB_A)
        self.assertEqual(record.state, "ABANDONED")
        self.assertEqual(record.error["code"], "PUMP_UNAVAILABLE")
        self.assertEqual(summary["abandoned"], 1)

    def test_abandoned_jobs_are_reported_not_hidden(self):
        self.enqueue(JOB_A)
        force_state(self.queue, JOB_A, "ABANDONED")
        summary = self._wait()
        self.assertEqual(summary["abandoned"], 1)
        self.assertEqual(summary["outstanding"], 0)
        self.assertIn(JOB_A, summary["abandoned_job_ids"])

    def test_progress_is_reported_to_the_callback(self):
        self.enqueue(JOB_A)
        seen = []
        lock = self.queue.acquire_pump_lock()
        try:
            self._wait(timeout=120.0, progress_interval=1.0,
                       on_progress=seen.append)
        finally:
            lock.release()
        self.assertTrue(seen)
        self.assertIn("outstanding", seen[0])

    def test_the_computed_timeout_is_used_when_none_is_given(self):
        self.enqueue(JOB_A)
        lock = self.queue.acquire_pump_lock()
        try:
            summary = self._wait(timeout=None)
        finally:
            lock.release()
        self.assertTrue(summary["timed_out"])
        self.assertGreaterEqual(self.clock.now - 1000.0, 3600.0,
                                "the computed floor is one hour")

    def test_the_summary_carries_what_the_final_report_needs(self):
        self.enqueue(JOB_A, outputs=True)
        summary = self._wait()
        for key in ("collected", "already_complete", "abandoned", "outstanding",
                    "elapsed_seconds", "total", "counts"):
            self.assertIn(key, summary)
        self.assertEqual(summary["already_complete"], 1)

    def test_the_progress_line_names_the_active_job(self):
        self.enqueue(JOB_A)
        force_state(self.queue, JOB_A, "RUNNING",
                    remote={"dir": "/r/a", "launched_at": gq._utcnow_iso()})
        line = gq._progress_line(self.queue.summary(), 180.0)
        self.assertIn("GNINA QUEUE", line)
        self.assertIn(JOB_A, line)


class TestSummary(QueueCase):
    def test_failed_jobs_count_as_outstanding(self):
        """The pump may still retry them; the barrier must keep waiting."""
        self.enqueue(JOB_A)
        force_state(self.queue, JOB_A, "FAILED",
                    error=gq.make_error("STALLED", "m", "poll"))
        summary = self.queue.summary()
        self.assertEqual(summary["outstanding"], 1)
        self.assertEqual(summary["failed"], 1)

    def test_error_codes_are_tallied(self):
        self.enqueue(JOB_A)
        self.enqueue(JOB_B)
        force_state(self.queue, JOB_A, "FAILED",
                    error=gq.make_error("STALLED", "m", "poll"))
        force_state(self.queue, JOB_B, "FAILED",
                    error=gq.make_error("STALLED", "m", "poll"))
        self.assertEqual(self.queue.summary()["errors"]["STALLED"], 2)

    def test_mean_remote_seconds_drives_the_eta(self):
        self.enqueue(JOB_A)
        force_state(self.queue, JOB_A, "COLLECTED",
                    remote={"remote_seconds": 150.0})
        self.assertEqual(self.queue.summary()["mean_remote_seconds"], 150.0)

    def test_write_summary_is_atomic_json(self):
        self.enqueue(JOB_A)
        path = self.queue.write_summary({"note": "hello"})
        with open(path) as handle:
            data = json.load(handle)
        self.assertEqual(data["note"], "hello")
        self.assertEqual(data["total"], 1)


class TestDefaultRunner(unittest.TestCase):
    """The one seam that really spawns consensus_docker.py."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.log = os.path.join(self.tmp, "phase.log")

    def tearDown(self):
        self._tmp.cleanup()

    def test_it_returns_the_exit_code_and_records_both_streams(self):
        script = ("import sys; sys.stdout.write('to stdout'); "
                  "sys.stderr.write('to stderr'); sys.exit(7)")
        rc, out, err = gq.default_runner(
            [sys.executable, "-c", script], dict(os.environ), 60, self.log, self.tmp)
        self.assertEqual(rc, 7)
        self.assertIn("to stdout", out)
        self.assertIn("to stderr", err)
        with open(self.log) as handle:
            logged = handle.read()
        self.assertIn("to stdout", logged)
        self.assertIn("to stderr", logged)
        self.assertIn("exit 7", logged)

    def test_the_phase_environment_reaches_the_child(self):
        env = dict(os.environ)
        env["GNINA_REMOTE_MODE"] = "submit"
        env["GNINA_REMOTE_TICKET"] = "/x/t.json"
        rc, out, _err = gq.default_runner(
            [sys.executable, "-c",
             "import os; print(os.environ['GNINA_REMOTE_MODE'],"
             " os.environ['GNINA_REMOTE_TICKET'])"],
            env, 60, self.log, self.tmp)
        self.assertEqual(rc, 0)
        self.assertIn("submit /x/t.json", out)

    def test_the_log_is_appended_not_truncated(self):
        """Both phases of a retried job must leave their evidence behind."""
        for _ in range(2):
            gq.default_runner([sys.executable, "-c", "print('hello')"],
                              dict(os.environ), 60, self.log, self.tmp)
        with open(self.log) as handle:
            logged = handle.read()
        self.assertEqual(logged.count("--- stdout ---"), 2)
        self.assertEqual(logged.count("exit 0"), 2)

    def test_a_timeout_kills_the_whole_process_tree_and_raises(self):
        """Killing the local process must not orphan its children."""
        marker = os.path.join(self.tmp, "grandchild.pid")
        script = (
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(120)'])\n"
            "open(%r, 'w').write(str(child.pid))\n"
            "time.sleep(120)\n" % marker
        )
        with self.assertRaises(subprocess.TimeoutExpired):
            gq.default_runner([sys.executable, "-c", script], dict(os.environ),
                              3, self.log, self.tmp)
        deadline = time.time() + 10
        with open(marker) as handle:
            grandchild = int(handle.read().strip())
        while time.time() < deadline and gq._pid_alive(grandchild):
            time.sleep(0.1)
        self.assertFalse(gq._pid_alive(grandchild),
                         "a timed-out phase left an orphaned child behind")
        with open(self.log) as handle:
            self.assertIn("TIMEOUT", handle.read())


class TestQueueRoot(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("GNINA_QUEUE_DIR", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["GNINA_QUEUE_DIR"] = self._saved
        else:
            os.environ.pop("GNINA_QUEUE_DIR", None)

    def test_the_default_is_under_the_run_directory(self):
        self.assertEqual(gq.queue_root("/runs/campaign"),
                         "/runs/campaign/gnina_queue")

    def test_the_environment_can_move_it_off_the_external_drive(self):
        os.environ["GNINA_QUEUE_DIR"] = "/fast/ext4/queue"
        self.assertEqual(gq.queue_root("/media/x/run"), "/fast/ext4/queue")

    def test_an_explicit_argument_wins(self):
        os.environ["GNINA_QUEUE_DIR"] = "/fast/ext4/queue"
        self.assertEqual(gq.queue_root("/media/x/run", "/tmp/other"), "/tmp/other")


class TestNoBatchDockImport(unittest.TestCase):
    def test_the_module_does_not_import_batch_dock(self):
        """batch_dock configures logging and writes a log file at import."""
        script = ("import sys; sys.path.insert(0, %r);"
                  "from dock_selectivity import gnina_queue;"
                  "assert 'batch_dock' not in sys.modules, sorted(sys.modules);"
                  "assert 'pandas' not in sys.modules;"
                  "assert 'rdkit' not in sys.modules;"
                  "print('clean')" % REPO)
        proc = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("clean", proc.stdout)

    def test_importing_it_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = ("import sys; sys.path.insert(0, %r);"
                      "from dock_selectivity import gnina_queue; print('ok')" % REPO)
            subprocess.run([sys.executable, "-c", script], cwd=tmp,
                           capture_output=True, text=True)
            self.assertEqual(os.listdir(tmp), [])


if __name__ == "__main__":
    unittest.main()
