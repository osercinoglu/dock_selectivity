"""Tests for the decoupled (async) gnina stage wired into ``batch_dock.py``.

gnina is GPU-only and the GPU is on another machine, so stage 2 does not need the
local cores at all -- but today the whole pipeline stops and waits for it. With
``GNINA_ASYNC=1`` the stage enqueues its jobs before stage 1, one remote GPU works
through them while smina/GOLD/LeDock keep the local cores busy, and a barrier
before stage 5 waits for whatever is still outstanding.

The delicate parts, and therefore what is tested here:

* **The default must not move.** ``GNINA_ASYNC=0`` is today's blocking stage,
  byte for byte. A reproduction run must not silently take a different path, so
  the sync tests patch every async entry point to raise and assert the run still
  works.
* **The barrier decides what stage 5 may compute.** consensus RMSD is globbed from
  ``complex_*.pdb`` per tool, so a job whose gnina never landed is not an RMSD job.
  That exclusion already existed; what is new is that the excluded jobs are NAMED.
  A partial campaign analysed as if it were complete is the failure mode here.
* **Capability validation happens before job one.** "If exhaustiveness required for
  gnina is too much for remote device we shall warn and exit" is worth nothing if
  it fires after four hours of smina.

Hermetic: no SSH, no GPU, no docking engine, no external drive. The queue and the
capability modules are stand-ins installed into ``sys.modules`` -- they are owned
by other implementers and may not exist yet, and these tests are about what
batch_dock does with them, not about what they do.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import types
import unittest
from unittest import mock

import batch_dock


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MISSING = object()


# ---------------------------------------------------------------------------
# scaffolding
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def patched(**overrides):
    """Temporarily set batch_dock module globals, then put them back."""
    saved = {name: getattr(batch_dock, name, _MISSING) for name in overrides}
    for name, value in overrides.items():
        setattr(batch_dock, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is _MISSING:
                delattr(batch_dock, name)
            else:
                setattr(batch_dock, name, value)


@contextlib.contextmanager
def fake_submodule(name, module):
    """Install a stand-in for ``dock_selectivity.<name>``.

    batch_dock imports these lazily on purpose, so a stand-in in ``sys.modules``
    (plus the package attribute, which is what ``from X import Y`` actually reads)
    is enough to exercise every call site without the real module existing.
    """
    import dock_selectivity

    full = "dock_selectivity." + name
    saved_sys = sys.modules.get(full, _MISSING)
    saved_attr = getattr(dock_selectivity, name, _MISSING)
    sys.modules[full] = module
    setattr(dock_selectivity, name, module)
    try:
        yield module
    finally:
        if saved_sys is _MISSING:
            sys.modules.pop(full, None)
        else:
            sys.modules[full] = saved_sys
        if saved_attr is _MISSING:
            try:
                delattr(dock_selectivity, name)
            except AttributeError:
                pass
        else:
            setattr(dock_selectivity, name, saved_attr)


def make_queue_module(summary=None, counts=None, enqueue_raises=None,
                      barrier_raises=None, restart_collector_wanted=False,
                      outputs_complete=False):
    """A stand-in for ``dock_selectivity.gnina_queue`` that records its calls."""
    module = types.ModuleType("dock_selectivity.gnina_queue")
    calls = []
    module.calls = calls

    class QueueError(RuntimeError):
        pass

    module.QueueError = QueueError

    def queue_root(run_dir, queue_dir=None):
        if queue_dir:
            return os.path.abspath(queue_dir)
        return os.path.join(os.path.abspath(run_dir), "gnina_queue")

    module.queue_root = queue_root

    class FakeQueue(object):
        def __init__(self, run_dir, queue_dir=None, create=True, clock=None):
            self.run_dir = run_dir
            self.root = queue_root(run_dir, queue_dir)
            calls.append(("GninaQueue", run_dir, queue_dir, create))
            if create:
                os.makedirs(self.root, exist_ok=True)

        def enqueue_many(self, job_data_list, params):
            calls.append(("enqueue_many",
                          [job["current_outfolder"] for job in job_data_list],
                          params))
            if enqueue_raises is not None:
                raise enqueue_raises
            return {"enqueued": len(job_data_list), "already_present": 0,
                    "already_complete": 0}

        def counts(self):
            calls.append(("counts",))
            return dict(counts or {"PENDING": 0})

        def write_summary(self, extra=None):
            calls.append(("write_summary", extra))
            os.makedirs(self.root, exist_ok=True)
            path = os.path.join(self.root, "summary.json")
            with open(path, "w") as handle:
                json.dump({"extra": extra}, handle)
            return path

    module.GninaQueue = FakeQueue

    def _outputs_complete(job_dir):
        calls.append(("outputs_complete", job_dir))
        return outputs_complete

    module.outputs_complete = _outputs_complete

    def wait_for_completion(run_dir, timeout=None, poll=15.0, logger=None,
                            on_progress=None, restart_collector=None,
                            queue_dir=None, progress_interval=30.0, **kwargs):
        # Same signature as the real one (gnina_queue.wait_for_completion), so a
        # call that would fail there fails here too.
        calls.append(("wait_for_completion", run_dir, timeout, queue_dir))
        if barrier_raises is not None:
            raise barrier_raises
        if restart_collector_wanted and restart_collector is not None:
            restart_collector()
        return dict(summary if summary is not None else {
            "total": 1, "collected": 1, "abandoned": 0, "outstanding": 0,
            "already_complete": 0, "elapsed_seconds": 1.5})

    module.wait_for_completion = wait_for_completion
    return module


def make_caps_module(effective_cpu=24, exit_code=0, get_caps_raises=None):
    """A stand-in for ``dock_selectivity.remote_caps``."""
    module = types.ModuleType("dock_selectivity.remote_caps")
    calls = []
    module.calls = calls

    class RemoteCapsError(RuntimeError):
        pass

    module.RemoteCapsError = RemoteCapsError

    class DockRequest(object):
        def __init__(self, **fields):
            self.__dict__.update(fields)

    module.DockRequest = DockRequest

    class Report(object):
        def __init__(self, request):
            self.request = request
            self.effective_cpu = effective_cpu
            self.estimated_seconds = 150.0
            self.estimated_vram_mb = 3840.0
            self.findings = ()

        @property
        def ok(self):
            return exit_code == 0

        def format_report(self):
            return "fake caps report"

    def get_caps(host, port=None, key=None, cache_path=None, **kwargs):
        calls.append(("get_caps", host, port, cache_path, kwargs))
        if get_caps_raises is not None:
            raise get_caps_raises
        return {"host": host, "cores": 24}

    def load_caps(path):
        calls.append(("load_caps", path))
        return {"host": "canned", "cores": 24}

    def validate_request(caps, request, strict=False, previous_effective_cpu=None):
        calls.append(("validate_request", strict, previous_effective_cpu))
        return Report(request)

    def enforce(report, logger=None, strict=False):
        calls.append(("enforce", strict))
        return exit_code

    module.get_caps = get_caps
    module.load_caps = load_caps
    module.validate_request = validate_request
    module.enforce = enforce
    return module


STUB_COLLECTOR = textwrap.dedent(
    """
    # A stand-in for gnina_collect.py: record the argv batch_dock spawned us with,
    # then wait for the STOP file exactly the way the real pump does.
    import os, sys, time

    args = sys.argv[1:]
    run_dir = args[args.index("--run-dir") + 1]
    queue = os.path.join(run_dir, "gnina_queue")
    if "--queue-dir" in args:
        queue = args[args.index("--queue-dir") + 1]
    os.makedirs(queue, exist_ok=True)
    with open(os.path.join(queue, "stub_argv.txt"), "w") as handle:
        handle.write("\\n".join(args))
    deadline = time.time() + 60
    while time.time() < deadline:
        if os.path.exists(os.path.join(queue, "STOP")):
            sys.exit(4)
        time.sleep(0.05)
    sys.exit(1)
    """
)


def stage_stats(n):
    return {"completed": n, "skipped": 0, "failed": 0, "timeout": 0,
            "elapsed_time": 0.0, "unique_processes": 1}


class AsyncGninaTestCase(unittest.TestCase):
    """Common fixture: a run directory, quiet logging, and a recorded pipeline."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.run_dir = os.path.join(self.dir, "run")
        self.out_base = os.path.join(self.run_dir, "docking_results")
        self.ligands = os.path.join(self.dir, "ligands")
        os.makedirs(self.out_base)
        os.makedirs(self.ligands)
        self.cwd = os.getcwd()
        os.chdir(self.dir)          # preview/summary CSVs are written to the CWD
        self.events = []
        root = logging.getLogger()
        self._saved_handlers = root.handlers[:]
        root.handlers = [logging.NullHandler()]

    def tearDown(self):
        logging.getLogger().handlers = self._saved_handlers
        os.chdir(self.cwd)
        self.tmp.cleanup()

    # -- fixtures ---------------------------------------------------------
    def build_job(self, job_id, tools=(), complexes=(), final=False):
        """Create a job directory with the given tools' results.csv."""
        job_dir = os.path.join(self.out_base, job_id)
        os.makedirs(job_dir, exist_ok=True)
        for tool in tools:
            tool_dir = os.path.join(job_dir, tool)
            os.makedirs(tool_dir, exist_ok=True)
            with open(os.path.join(tool_dir, "results.csv"), "w") as handle:
                handle.write("Pose,Score\n1,-8.1\n")
        for tool in complexes:
            tool_dir = os.path.join(job_dir, tool)
            os.makedirs(tool_dir, exist_ok=True)
            with open(os.path.join(tool_dir, "complex_1.pdb"), "w") as handle:
                handle.write("ATOM      1  N   ALA A   1       0.0 0.0 0.0\n")
        if final:
            with open(os.path.join(job_dir, "final_results.csv"), "w") as handle:
                handle.write("pose,consensus_rmsd\n" + "1,0.5\n" * 40)
        ligand = os.path.join(self.ligands, "%s.sdf" % job_id.split("__")[0])
        open(ligand, "a").close()
        receptor = os.path.join(self.dir, "receptor_%s.pdb" % job_id)
        open(receptor, "a").close()
        pocket = os.path.join(self.dir, "AF-X-F1-model_v1_cavity_1_%s.pdb" % job_id)
        open(pocket, "a").close()
        parts = job_id.split("__")
        return {
            "drugbank_id": parts[0],
            "uniprot_id": parts[1] if len(parts) > 1 else "P00000",
            "gene_name": parts[2] if len(parts) > 2 else "GENE",
            "ligand_sdf": ligand,
            "receptor_pdb": receptor,
            "receptor_pdbqt": None,
            "pocket_pdb": pocket,
            "current_outfolder": job_dir,
            "job_signature": job_id,
        }

    def recorded_stages(self):
        """Replacements for the two stage runners that record what they were given."""
        def fake_stage(job_data_list, stage_name, job_function, max_workers=None,
                       timeout=None, exhaustiveness=None):
            self.events.append(
                (stage_name, [os.path.basename(j["current_outfolder"])
                              for j in job_data_list]))
            return stage_stats(len(job_data_list))

        def fake_adaptive(stage_name, jobs, job_function, resource_monitor=None):
            self.events.append(
                (stage_name, [os.path.basename(j["current_outfolder"]) for j in jobs]))
            return stage_stats(len(jobs))

        return fake_stage, fake_adaptive

    def run_docking(self, jobs):
        with contextlib.redirect_stdout(io.StringIO()):
            batch_dock.run_docking(None, None, {}, self.ligands,
                                   output_base_folder=self.out_base,
                                   prebuilt_jobs=list(jobs))

    def stage_names(self):
        return [name for name, _jobs in self.events]

    def jobs_for(self, needle):
        for name, job_ids in self.events:
            if needle in name:
                return job_ids
        return None


# ---------------------------------------------------------------------------
# the default must not move
# ---------------------------------------------------------------------------

class TestSyncModeIsUnchanged(AsyncGninaTestCase):
    def test_async_is_off_by_default(self):
        """Opt-in, always. Changing the default changes what a re-run does."""
        self.assertFalse(batch_dock.GNINA_ASYNC)
        self.assertFalse(batch_dock.GNINA_VALIDATE_CAPS)
        self.assertFalse(batch_dock.GNINA_PARTIAL_RMSD)
        self.assertIsNone(batch_dock.GNINA_BARRIER_TIMEOUT)
        with patched(GNINA_ASYNC=False):
            self.assertFalse(batch_dock.gnina_async_enabled())
        with patched(GNINA_ASYNC=True):
            self.assertTrue(batch_dock.gnina_async_enabled())

    def test_sync_run_never_touches_the_queue(self):
        """Every async entry point raises; the blocking pipeline must not care."""
        def explode(*args, **kwargs):
            raise AssertionError("async path taken with GNINA_ASYNC=0")

        job = self.build_job("DB00035__P30518__AVPR2__F1__cav1",
                             tools=("smina", "gnina", "gold"))
        fake_stage, fake_adaptive = self.recorded_stages()
        with patched(GNINA_ASYNC=False, GNINA_VALIDATE_CAPS=False,
                     USE_SMINA=True, USE_GNINA=True, USE_GOLD=True,
                     USE_LEDOCK=False, USE_RMSD_CALCULATION=True,
                     run_stage_jobs=fake_stage,
                     run_stage_jobs_adaptive=fake_adaptive,
                     resolve_gnina_caps=explode, enqueue_gnina_jobs=explode,
                     start_gnina_collector=explode, wait_for_gnina_stage=explode,
                     stop_gnina_collector=explode):
            self.run_docking([job])

        names = self.stage_names()
        self.assertIn("Stage 2: Gnina Docking", names)
        self.assertEqual(["DB00035__P30518__AVPR2__F1__cav1"],
                         self.jobs_for("Stage 2: Gnina"))
        self.assertEqual(["DB00035__P30518__AVPR2__F1__cav1"],
                         self.jobs_for("Stage 5: RMSD"))

    def test_sync_run_does_not_clamp_threads_without_a_caps_report(self):
        """No capability report -> job_data is handed to stage 2 untouched."""
        job = self.build_job("DB00035__P30518__AVPR2__F1__cav1", tools=("smina",))
        seen = {}

        def fake_stage(job_data_list, stage_name, job_function, max_workers=None,
                       timeout=None, exhaustiveness=None):
            if "Gnina" in stage_name:
                seen["num_threads"] = [j["num_threads"] for j in job_data_list]
                seen["identical"] = job_data_list is not None
            return stage_stats(len(job_data_list))

        with patched(GNINA_ASYNC=False, GNINA_VALIDATE_CAPS=False,
                     USE_SMINA=False, USE_GNINA=True, USE_GOLD=False,
                     USE_LEDOCK=False, USE_RMSD_CALCULATION=False,
                     run_stage_jobs=fake_stage):
            self.run_docking([job])
        self.assertEqual([batch_dock.NUM_THREADS], seen["num_threads"])


# ---------------------------------------------------------------------------
# async mode: submit and keep going
# ---------------------------------------------------------------------------

class TestAsyncSubmission(AsyncGninaTestCase):
    def test_async_stage_enqueues_before_stage_one_and_never_blocks(self):
        jobs = [self.build_job("DB00035__P30518__AVPR2__F1__cav1", tools=("smina",)),
                self.build_job("DB00040__P12345__XYZ__F1__cav2", tools=("smina",))]
        queue_module = make_queue_module()
        fake_stage, fake_adaptive = self.recorded_stages()

        def fake_enqueue(job_data_list, run_dir, params):
            self.events.append(("ENQUEUE", [os.path.basename(j["current_outfolder"])
                                            for j in job_data_list]))
            return {"enqueued": len(job_data_list), "already_present": 0,
                    "already_complete": 0}

        def fake_collector(run_dir, logger=None):
            self.events.append(("COLLECTOR", [run_dir]))
            return None

        def fake_barrier(run_dir, timeout=None):
            self.events.append(("BARRIER", [run_dir]))
            # What the collector does while the local stages run: the gnina result
            # lands in the job directory. Stage 5 can only see it because the
            # barrier is between them.
            for job in jobs:
                tool_dir = os.path.join(job["current_outfolder"], "gnina")
                os.makedirs(tool_dir, exist_ok=True)
                with open(os.path.join(tool_dir, "results.csv"), "w") as handle:
                    handle.write("Pose,GNINA_CNN_affinity\n1,6.4\n")
            return {"total": 2, "collected": 2, "abandoned": 0, "outstanding": 0,
                    "already_complete": 0, "elapsed_seconds": 3.0}

        with fake_submodule("gnina_queue", queue_module), \
                patched(GNINA_ASYNC=True, USE_SMINA=True, USE_GNINA=True,
                        USE_GOLD=False, USE_LEDOCK=False, USE_RMSD_CALCULATION=True,
                        run_stage_jobs=fake_stage,
                        run_stage_jobs_adaptive=fake_adaptive,
                        resolve_gnina_caps=lambda run_dir, strict=False: None,
                        enqueue_gnina_jobs=fake_enqueue,
                        start_gnina_collector=fake_collector,
                        wait_for_gnina_stage=fake_barrier,
                        stop_gnina_collector=lambda *a, **k: None):
            self.run_docking(jobs)

        order = self.stage_names()
        self.assertEqual("ENQUEUE", order[0], "gnina must be queued before stage 1")
        self.assertEqual("COLLECTOR", order[1])
        self.assertIn("Stage 1: Smina Docking", order)
        self.assertLess(order.index("Stage 1: Smina Docking"), order.index("BARRIER"))
        self.assertLess(order.index("BARRIER"),
                        order.index("Stage 5: RMSD Calculation"))
        # The blocking gnina stage must not run at all.
        self.assertNotIn("Stage 2: Gnina Docking", order)
        self.assertEqual(["DB00035__P30518__AVPR2__F1__cav1",
                          "DB00040__P12345__XYZ__F1__cav2"],
                         sorted(self.jobs_for("ENQUEUE")))
        # Both jobs reach RMSD, and only because the barrier ran first.
        self.assertEqual(["DB00035__P30518__AVPR2__F1__cav1",
                          "DB00040__P12345__XYZ__F1__cav2"],
                         sorted(self.jobs_for("Stage 5: RMSD")))

    def test_barrier_result_feeds_the_final_summary(self):
        """stage2_stats keys must be exactly what the summary block already reads."""
        job = self.build_job("DB00035__P30518__AVPR2__F1__cav1",
                             tools=("smina", "gnina"))
        summary = {"total": 7, "collected": 5, "abandoned": 1, "outstanding": 1,
                   "already_complete": 2, "elapsed_seconds": 12.5}
        queue_module = make_queue_module(summary=summary)
        fake_stage, fake_adaptive = self.recorded_stages()
        with fake_submodule("gnina_queue", queue_module), \
                patched(GNINA_ASYNC=True, USE_SMINA=True, USE_GNINA=True,
                        USE_GOLD=False, USE_LEDOCK=False, USE_RMSD_CALCULATION=True,
                        run_stage_jobs=fake_stage,
                        run_stage_jobs_adaptive=fake_adaptive,
                        resolve_gnina_caps=lambda run_dir, strict=False: None,
                        enqueue_gnina_jobs=lambda *a, **k: {"enqueued": 1},
                        start_gnina_collector=lambda *a, **k: None,
                        stop_gnina_collector=lambda *a, **k: None):
            with self.assertLogs(level="INFO") as captured:
                self.run_docking([job])

        text = "\n".join(captured.output)
        self.assertIn("Gnina jobs completed successfully: 5", text)
        self.assertIn("Gnina jobs skipped (already completed): 2", text)
        self.assertIn("Gnina jobs failed: 1", text)
        self.assertIn("Stage 2 execution time: 12.50 seconds", text)
        # A job still outstanding when the barrier gave up is reported, not hidden:
        # it lands in the summary's "timed out" column.
        self.assertRegex(text, r"Gnina jobs timed out \([^)]*\): 1")
        # And a barrier that gave up is NOT a failed run.
        self.assertIn("Multi-stage docking workflow finished.", text)

    def test_queue_params_are_frozen_from_the_capability_report(self):
        report = types.SimpleNamespace(effective_cpu=24)
        with patched(GNINA_EXHAUSTIVENESS=32, GNINA_JOB_DEADLINE=1800.0,
                     GNINA_MAX_ATTEMPTS=3, NUM_THREADS=60):
            params = batch_dock.gnina_params_from_config(report)
        self.assertEqual(24, params["cpu"], "the clamp must be frozen, not NUM_THREADS")
        self.assertIsNone(params["num_modes"],
                          "passing --num_modes would override consensus_docker's 20")
        self.assertEqual(32, params["exhaustiveness"])
        self.assertEqual(1800.0, params["job_deadline_seconds"])
        self.assertEqual(3, params["max_attempts"])
        self.assertTrue(os.path.isabs(params["consensus_docker_script"]))
        self.assertTrue(os.path.isabs(params["gnina_path"]))
        self.assertEqual(1, params["schema_version"])

    def test_params_fall_back_to_num_threads_without_a_report(self):
        with patched(NUM_THREADS=60):
            self.assertEqual(60, batch_dock.gnina_params_from_config(None)["cpu"])

    def test_enqueue_uses_the_queue_and_reports_counts(self):
        job = self.build_job("DB00035__P30518__AVPR2__F1__cav1")
        queue_module = make_queue_module()
        with fake_submodule("gnina_queue", queue_module), patched(GNINA_QUEUE_DIR=None):
            stats = batch_dock.enqueue_gnina_jobs([job], self.run_dir, {"cpu": 24})
        self.assertEqual(1, stats["enqueued"])
        enqueued = [c for c in queue_module.calls if c[0] == "enqueue_many"]
        self.assertEqual(1, len(enqueued))
        self.assertEqual([job["current_outfolder"]], enqueued[0][1])

    def test_a_queue_that_refuses_the_params_stops_the_run(self):
        """A mid-campaign parameter change is refused, loudly, not applied."""
        job = self.build_job("DB00035__P30518__AVPR2__F1__cav1")
        queue_module = make_queue_module()
        queue_module_err = make_queue_module(
            enqueue_raises=queue_module.QueueError("params.json differs"))
        queue_module_err.QueueError = queue_module.QueueError
        with fake_submodule("gnina_queue", queue_module_err):
            with self.assertRaises(SystemExit) as caught:
                batch_dock.enqueue_gnina_jobs([job], self.run_dir, {"cpu": 24})
        self.assertEqual(2, caught.exception.code)

    def test_run_dir_is_the_parent_of_the_output_base(self):
        self.assertEqual(self.run_dir, batch_dock.gnina_run_dir(self.out_base))
        self.assertEqual("/a/b", batch_dock.gnina_run_dir("/a/b/docking_results"))

    def test_queue_root_honours_the_override(self):
        with patched(GNINA_QUEUE_DIR=None):
            self.assertEqual(os.path.join(self.run_dir, "gnina_queue"),
                             batch_dock._gnina_queue_root(self.run_dir))
        elsewhere = os.path.join(self.dir, "ext4_queue")
        with patched(GNINA_QUEUE_DIR=elsewhere):
            self.assertEqual(elsewhere, batch_dock._gnina_queue_root(self.run_dir))


# ---------------------------------------------------------------------------
# the barrier
# ---------------------------------------------------------------------------

class TestBarrier(AsyncGninaTestCase):
    def test_barrier_waits_then_returns_the_summary(self):
        queue_module = make_queue_module(summary={
            "total": 3, "collected": 3, "abandoned": 0, "outstanding": 0,
            "already_complete": 0, "elapsed_seconds": 2.0})
        with fake_submodule("gnina_queue", queue_module), \
                patched(GNINA_QUEUE_DIR=None):
            summary = batch_dock.wait_for_gnina_stage(self.run_dir, timeout=42.0)
        self.assertEqual(3, summary["collected"])
        waited = [c for c in queue_module.calls if c[0] == "wait_for_completion"]
        self.assertEqual([("wait_for_completion", self.run_dir, 42.0, None)], waited)

    def test_barrier_restarts_a_dead_collector(self):
        """A silently dead pump must never look like a slow GPU."""
        restarts = []
        queue_module = make_queue_module(restart_collector_wanted=True)
        with fake_submodule("gnina_queue", queue_module), \
                patched(start_gnina_collector=lambda run_dir, logger=None:
                        restarts.append(run_dir)):
            with self.assertLogs(level="WARNING") as captured:
                batch_dock.wait_for_gnina_stage(self.run_dir)
        self.assertEqual([self.run_dir], restarts)
        self.assertIn("restarting it", "\n".join(captured.output))

    def test_a_broken_barrier_does_not_lose_stage_five(self):
        """If the barrier itself fails, the jobs that DID land still get RMSD."""
        queue_module = make_queue_module(barrier_raises=RuntimeError("queue on fire"))
        with fake_submodule("gnina_queue", queue_module):
            with self.assertLogs(level="CRITICAL") as captured:
                summary = batch_dock.wait_for_gnina_stage(self.run_dir)
        self.assertEqual("queue on fire", summary["error"])
        self.assertEqual(0, summary["collected"])
        self.assertIn("queue on fire", "\n".join(captured.output))

    def test_barrier_is_skipped_when_rmsd_is_disabled_but_says_so(self):
        job = self.build_job("DB00035__P30518__AVPR2__F1__cav1")
        queue_module = make_queue_module()
        barrier_calls = []
        with fake_submodule("gnina_queue", queue_module), \
                patched(GNINA_ASYNC=True, USE_SMINA=False, USE_GNINA=True,
                        USE_GOLD=False, USE_LEDOCK=False,
                        USE_RMSD_CALCULATION=False,
                        resolve_gnina_caps=lambda run_dir, strict=False: None,
                        enqueue_gnina_jobs=lambda *a, **k: {"enqueued": 1},
                        start_gnina_collector=lambda *a, **k: None,
                        stop_gnina_collector=lambda *a, **k: None,
                        wait_for_gnina_stage=lambda *a, **k: barrier_calls.append(a)):
            with self.assertLogs(level="WARNING") as captured:
                self.run_docking([job])
        self.assertEqual([], barrier_calls)
        self.assertIn("RMSD is disabled", "\n".join(captured.output))


# ---------------------------------------------------------------------------
# what stage 5 may compute, and what it must say about the rest
# ---------------------------------------------------------------------------

class TestStageFiveExclusions(AsyncGninaTestCase):
    def test_job_whose_gnina_never_landed_is_skipped_and_named(self):
        landed = self.build_job("DB00035__P30518__AVPR2__F1__cav1",
                                tools=("smina", "gnina"))
        missing = self.build_job("DB00040__P12345__XYZ__F1__cav2", tools=("smina",))
        queue_module = make_queue_module()
        fake_stage, fake_adaptive = self.recorded_stages()
        with fake_submodule("gnina_queue", queue_module), \
                patched(GNINA_ASYNC=True, USE_SMINA=True, USE_GNINA=True,
                        USE_GOLD=False, USE_LEDOCK=False, USE_RMSD_CALCULATION=True,
                        GNINA_PARTIAL_RMSD=False,
                        run_stage_jobs=fake_stage,
                        run_stage_jobs_adaptive=fake_adaptive,
                        resolve_gnina_caps=lambda run_dir, strict=False: None,
                        enqueue_gnina_jobs=lambda *a, **k: {"enqueued": 2},
                        start_gnina_collector=lambda *a, **k: None,
                        stop_gnina_collector=lambda *a, **k: None):
            with self.assertLogs(level="WARNING") as captured:
                self.run_docking([landed, missing])

        self.assertEqual(["DB00035__P30518__AVPR2__F1__cav1"],
                         self.jobs_for("Stage 5: RMSD"))
        warning = "\n".join(captured.output)
        self.assertIn("DB00040__P12345__XYZ__F1__cav2", warning)
        self.assertIn("gnina result never landed", warning)
        self.assertIn("re-running batch_dock", warning)
        # ... and recorded where a later analysis can find it.
        recorded = [c for c in queue_module.calls if c[0] == "write_summary"]
        self.assertEqual(1, len(recorded))
        self.assertEqual(["DB00040__P12345__XYZ__F1__cav2"],
                         recorded[0][1]["rmsd_excluded_jobs"])
        self.assertEqual([], recorded[0][1]["partial_rmsd_jobs"])

    def test_partial_rmsd_admits_a_two_tool_job_and_records_it(self):
        """Off by default; when on, every admitted job is named as different."""
        job = self.build_job("DB00040__P12345__XYZ__F1__cav2",
                             tools=("smina", "gold"), complexes=("smina", "gold"))
        queue_module = make_queue_module()
        fake_stage, fake_adaptive = self.recorded_stages()
        common = dict(GNINA_ASYNC=True, USE_SMINA=True, USE_GNINA=True,
                      USE_GOLD=True, USE_LEDOCK=False, USE_RMSD_CALCULATION=True,
                      run_stage_jobs=fake_stage,
                      run_stage_jobs_adaptive=fake_adaptive,
                      resolve_gnina_caps=lambda run_dir, strict=False: None,
                      enqueue_gnina_jobs=lambda *a, **k: {"enqueued": 1},
                      start_gnina_collector=lambda *a, **k: None,
                      stop_gnina_collector=lambda *a, **k: None)

        with fake_submodule("gnina_queue", queue_module), \
                patched(GNINA_PARTIAL_RMSD=False, **common):
            self.run_docking([job])
        self.assertIsNone(self.jobs_for("Stage 5: RMSD"),
                          "default must NOT compute a consensus without gnina")

        self.events = []
        with fake_submodule("gnina_queue", queue_module), \
                patched(GNINA_PARTIAL_RMSD=True, **common):
            with self.assertLogs(level="WARNING") as captured:
                self.run_docking([job])
        self.assertEqual(["DB00040__P12345__XYZ__F1__cav2"],
                         self.jobs_for("Stage 5: RMSD"))
        self.assertIn("WITHOUT gnina", "\n".join(captured.output))
        recorded = [c for c in queue_module.calls if c[0] == "write_summary"][-1]
        self.assertEqual(["DB00040__P12345__XYZ__F1__cav2"],
                         recorded[1]["partial_rmsd_jobs"])
        self.assertEqual([], recorded[1]["rmsd_excluded_jobs"],
                         "an admitted job was not excluded; do not report it twice")

    def test_partial_rmsd_still_refuses_a_job_with_one_tool(self):
        """rmsd_inputs_present() is the gate, and one tool is not enough."""
        job = self.build_job("DB00041__P12345__XYZ__F1__cav3",
                             tools=("smina",), complexes=("smina",))
        queue_module = make_queue_module()
        fake_stage, fake_adaptive = self.recorded_stages()
        with fake_submodule("gnina_queue", queue_module), \
                patched(GNINA_ASYNC=True, GNINA_PARTIAL_RMSD=True, USE_SMINA=True,
                        USE_GNINA=True, USE_GOLD=False, USE_LEDOCK=False,
                        USE_RMSD_CALCULATION=True, run_stage_jobs=fake_stage,
                        run_stage_jobs_adaptive=fake_adaptive,
                        resolve_gnina_caps=lambda run_dir, strict=False: None,
                        enqueue_gnina_jobs=lambda *a, **k: {"enqueued": 1},
                        start_gnina_collector=lambda *a, **k: None,
                        stop_gnina_collector=lambda *a, **k: None):
            self.run_docking([job])
        self.assertIsNone(self.jobs_for("Stage 5: RMSD"))

    def test_sync_mode_exclusion_is_silent_as_before(self):
        """No new reporting on the blocking path: D6 keeps it byte-for-byte."""
        missing = self.build_job("DB00040__P12345__XYZ__F1__cav2", tools=("smina",))
        fake_stage, fake_adaptive = self.recorded_stages()
        with patched(GNINA_ASYNC=False, USE_SMINA=True, USE_GNINA=True,
                     USE_GOLD=False, USE_LEDOCK=False, USE_RMSD_CALCULATION=True,
                     run_stage_jobs=fake_stage,
                     run_stage_jobs_adaptive=fake_adaptive):
            with self.assertLogs(level="INFO") as captured:
                self.run_docking([missing])
        self.assertIsNone(self.jobs_for("Stage 5: RMSD"))
        self.assertNotIn("gnina result never landed", "\n".join(captured.output))


# ---------------------------------------------------------------------------
# capability validation: warn and exit, before anything is submitted
# ---------------------------------------------------------------------------

class TestCapabilityValidation(AsyncGninaTestCase):
    def test_fatal_capabilities_exit_78_before_any_job_is_submitted(self):
        job = self.build_job("DB00035__P30518__AVPR2__F1__cav1")
        caps_module = make_caps_module(exit_code=78)
        queue_module = make_queue_module()
        submitted = []

        def explode(*args, **kwargs):
            submitted.append(args)
            raise AssertionError("a job was submitted after a FATAL capability check")

        with fake_submodule("remote_caps", caps_module), \
                fake_submodule("gnina_queue", queue_module), \
                mock.patch.dict(os.environ, {"GNINA_REMOTE_HOST": "user@gpu"}), \
                patched(GNINA_ASYNC=True, USE_SMINA=True, USE_GNINA=True,
                        USE_GOLD=False, USE_LEDOCK=False, USE_RMSD_CALCULATION=True,
                        GNINA_CAPS_FILE=None,
                        run_stage_jobs=explode, run_stage_jobs_adaptive=explode,
                        enqueue_gnina_jobs=explode, start_gnina_collector=explode,
                        wait_for_gnina_stage=explode):
            with self.assertRaises(SystemExit) as caught:
                self.run_docking([job])
        self.assertEqual(78, caught.exception.code)
        self.assertEqual([], submitted)
        self.assertEqual([], [c for c in queue_module.calls if c[0] == "enqueue_many"])

    def test_a_healthy_report_is_returned_and_the_previous_cpu_is_offered(self):
        caps_module = make_caps_module(effective_cpu=24, exit_code=0)
        queue_root = os.path.join(self.run_dir, "gnina_queue")
        os.makedirs(queue_root)
        with open(os.path.join(queue_root, "params.json"), "w") as handle:
            json.dump({"cpu": 24}, handle)
        with fake_submodule("remote_caps", caps_module), \
                mock.patch.dict(os.environ, {"GNINA_REMOTE_HOST": "user@gpu",
                                             "GNINA_REMOTE_PORT": "50000"}), \
                patched(GNINA_ASYNC=True, GNINA_CAPS_FILE=None, GNINA_QUEUE_DIR=None):
            report = batch_dock.resolve_gnina_caps(self.run_dir)
        self.assertEqual(24, report.effective_cpu)
        validated = [c for c in caps_module.calls if c[0] == "validate_request"]
        self.assertEqual([("validate_request", False, 24)], validated)
        probed = [c for c in caps_module.calls if c[0] == "get_caps"]
        self.assertEqual(50000, probed[0][2])

    def test_a_canned_caps_file_is_used_instead_of_probing_the_gpu_host(self):
        caps_module = make_caps_module()
        canned = os.path.join(self.dir, "caps.json")
        with open(canned, "w") as handle:
            json.dump({"cores": 24}, handle)
        with fake_submodule("remote_caps", caps_module), \
                mock.patch.dict(os.environ, {"GNINA_REMOTE_HOST": "user@gpu"}), \
                patched(GNINA_ASYNC=True, GNINA_CAPS_FILE=canned):
            batch_dock.resolve_gnina_caps(self.run_dir)
        self.assertEqual([("load_caps", canned)],
                         [c for c in caps_module.calls if c[0] == "load_caps"])
        self.assertEqual([], [c for c in caps_module.calls if c[0] == "get_caps"])

    def test_async_without_a_remote_host_exits_78(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GNINA_REMOTE_HOST", None)
            with patched(GNINA_ASYNC=True):
                with self.assertRaises(SystemExit) as caught:
                    batch_dock.resolve_gnina_caps(self.run_dir)
        self.assertEqual(78, caught.exception.code)

    def test_an_unreachable_host_stops_an_async_run_rather_than_starting_blind(self):
        caps_module = make_caps_module(get_caps_raises=OSError("no route to host"))
        with fake_submodule("remote_caps", caps_module), \
                mock.patch.dict(os.environ, {"GNINA_REMOTE_HOST": "user@gpu"}), \
                patched(GNINA_ASYNC=True, GNINA_CAPS_FILE=None):
            with self.assertRaises(SystemExit) as caught:
                batch_dock.resolve_gnina_caps(self.run_dir)
        self.assertEqual(78, caught.exception.code)

    def test_sync_mode_survives_a_probe_failure(self):
        """Validation is advisory when gnina still runs the old blocking way."""
        caps_module = make_caps_module(get_caps_raises=OSError("no route to host"))
        with fake_submodule("remote_caps", caps_module), \
                mock.patch.dict(os.environ, {"GNINA_REMOTE_HOST": "user@gpu"}), \
                patched(GNINA_ASYNC=False, GNINA_VALIDATE_CAPS=True,
                        GNINA_CAPS_FILE=None):
            with self.assertLogs(level="WARNING"):
                self.assertIsNone(batch_dock.resolve_gnina_caps(self.run_dir))

    def test_sync_stage_two_clamps_threads_to_the_remote_core_count(self):
        """--cpu 60 on a 24-core box is the mismatch this whole check exists for."""
        job = self.build_job("DB00035__P30518__AVPR2__F1__cav1")
        seen = {}

        def fake_stage(job_data_list, stage_name, job_function, max_workers=None,
                       timeout=None, exhaustiveness=None):
            if "Gnina" in stage_name:
                seen["threads"] = sorted({j["num_threads"] for j in job_data_list})
            return stage_stats(len(job_data_list))

        with patched(GNINA_ASYNC=False, GNINA_VALIDATE_CAPS=True, NUM_THREADS=60,
                     USE_SMINA=False, USE_GNINA=True, USE_GOLD=False,
                     USE_LEDOCK=False, USE_RMSD_CALCULATION=False,
                     run_stage_jobs=fake_stage,
                     resolve_gnina_caps=lambda run_dir, strict=False:
                         types.SimpleNamespace(effective_cpu=24)):
            with self.assertLogs(level="WARNING") as captured:
                self.run_docking([job])
        self.assertEqual([24], seen["threads"])
        self.assertIn("clamping gnina --num_threads from 60 to 24",
                      "\n".join(captured.output))
        self.assertEqual(60, batch_dock.NUM_THREADS,
                         "the global belongs to the local engines and must not move")


# ---------------------------------------------------------------------------
# the collector process
# ---------------------------------------------------------------------------

class TestCollectorProcess(AsyncGninaTestCase):
    def stub_script(self):
        path = os.path.join(self.dir, "stub_collect.py")
        with open(path, "w") as handle:
            handle.write(STUB_COLLECTOR)
        return path

    def wait_for(self, path, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if os.path.exists(path):
                return True
            time.sleep(0.05)
        return False

    def test_collector_is_spawned_with_the_run_directory_and_stops_on_request(self):
        queue_root = os.path.join(self.run_dir, "gnina_queue")
        with patched(GNINA_COLLECTOR_SCRIPT=self.stub_script(),
                     GNINA_COLLECTOR_AUTOSTART=True, GNINA_QUEUE_DIR=None,
                     GNINA_LEAVE_COLLECTOR=False, GNINA_CAPS_FILE=None,
                     GNINA_MAX_ATTEMPTS=3, GNINA_JOB_DEADLINE=1800.0):
            proc = batch_dock.start_gnina_collector(self.run_dir)
            self.assertIsNotNone(proc)
            try:
                self.assertTrue(self.wait_for(os.path.join(queue_root,
                                                           "stub_argv.txt")))
                with open(os.path.join(queue_root, "stub_argv.txt")) as handle:
                    argv = handle.read().split("\n")
                self.assertIn("--run-dir", argv)
                self.assertEqual(self.run_dir, argv[argv.index("--run-dir") + 1])
                self.assertIn("--daemon", argv)
                self.assertEqual("3", argv[argv.index("--max-attempts") + 1])
                batch_dock.stop_gnina_collector(self.run_dir, proc, timeout=30.0)
            finally:
                if proc.poll() is None:      # never leave a stray process behind
                    proc.kill()
                    proc.wait(timeout=10)
        self.assertEqual(4, proc.returncode, "the pump should stop gracefully")
        self.assertFalse(os.path.exists(os.path.join(queue_root, "STOP")),
                         "a leftover STOP would stop the NEXT run's collector")

    def test_a_leftover_stop_file_is_cleared_before_starting(self):
        queue_root = os.path.join(self.run_dir, "gnina_queue")
        os.makedirs(queue_root)
        with open(os.path.join(queue_root, "STOP"), "w") as handle:
            handle.write("from a previous run\n")
        with patched(GNINA_COLLECTOR_SCRIPT=self.stub_script(),
                     GNINA_COLLECTOR_AUTOSTART=True, GNINA_QUEUE_DIR=None,
                     GNINA_CAPS_FILE=None, GNINA_LEAVE_COLLECTOR=False):
            proc = batch_dock.start_gnina_collector(self.run_dir)
            try:
                self.assertFalse(os.path.exists(os.path.join(queue_root, "STOP")))
                self.assertTrue(self.wait_for(os.path.join(queue_root,
                                                           "stub_argv.txt")))
            finally:
                batch_dock.stop_gnina_collector(self.run_dir, proc, timeout=30.0)
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=10)

    def test_autostart_off_returns_none_and_says_how_to_start_it(self):
        with patched(GNINA_COLLECTOR_AUTOSTART=False,
                     GNINA_COLLECTOR_SCRIPT=self.stub_script()):
            with self.assertLogs(level="INFO") as captured:
                self.assertIsNone(batch_dock.start_gnina_collector(self.run_dir))
        self.assertIn("--daemon", "\n".join(captured.output))

    def test_a_missing_collector_script_is_reported_not_ignored(self):
        with patched(GNINA_COLLECTOR_AUTOSTART=True,
                     GNINA_COLLECTOR_SCRIPT=os.path.join(self.dir, "nope.py")):
            with self.assertLogs(level="ERROR") as captured:
                self.assertIsNone(batch_dock.start_gnina_collector(self.run_dir))
        self.assertIn("nothing will run them", "\n".join(captured.output))

    def test_leave_collector_keeps_it_running(self):
        with patched(GNINA_LEAVE_COLLECTOR=True):
            with self.assertLogs(level="INFO"):
                batch_dock.stop_gnina_collector(self.run_dir, None)
        self.assertFalse(os.path.exists(
            os.path.join(self.run_dir, "gnina_queue", "STOP")))

    def test_stopping_a_collector_we_did_not_start_leaves_the_stop_file(self):
        """We cannot see it exit, so the file must stay -- and be named."""
        with patched(GNINA_LEAVE_COLLECTOR=False, GNINA_QUEUE_DIR=None,
                     GNINA_COLLECTOR_AUTOSTART=True):
            with self.assertLogs(level="WARNING") as captured:
                batch_dock.stop_gnina_collector(self.run_dir, None)
        stop_file = os.path.join(self.run_dir, "gnina_queue", "STOP")
        self.assertTrue(os.path.exists(stop_file))
        self.assertIn(stop_file, "\n".join(captured.output))

    def test_no_collector_mode_does_not_stop_someone_elses_pump(self):
        """--no-collector means the pump is the operator's, running elsewhere."""
        with patched(GNINA_LEAVE_COLLECTOR=False, GNINA_QUEUE_DIR=None,
                     GNINA_COLLECTOR_AUTOSTART=False):
            with self.assertLogs(level="INFO"):
                batch_dock.stop_gnina_collector(self.run_dir, None)
        self.assertFalse(os.path.exists(
            os.path.join(self.run_dir, "gnina_queue", "STOP")))

    def test_the_finally_stops_the_collector_even_when_a_stage_explodes(self):
        job = self.build_job("DB00035__P30518__AVPR2__F1__cav1")
        stopped = []

        def boom(*args, **kwargs):
            raise RuntimeError("stage 1 died")

        with fake_submodule("gnina_queue", make_queue_module()), \
                patched(GNINA_ASYNC=True, USE_SMINA=True, USE_GNINA=True,
                        USE_GOLD=False, USE_LEDOCK=False, USE_RMSD_CALCULATION=True,
                        run_stage_jobs_adaptive=boom,
                        resolve_gnina_caps=lambda run_dir, strict=False: None,
                        enqueue_gnina_jobs=lambda *a, **k: {"enqueued": 1},
                        start_gnina_collector=lambda *a, **k: "proc",
                        stop_gnina_collector=lambda *a, **k: stopped.append(a)):
            with self.assertRaises(RuntimeError):
                self.run_docking([job])
        self.assertEqual(1, len(stopped))
        self.assertEqual((self.run_dir, "proc"), stopped[0])


# ---------------------------------------------------------------------------
# odds and ends
# ---------------------------------------------------------------------------

class TestMisc(AsyncGninaTestCase):
    def test_gnina_ready_delegates_to_the_queue(self):
        module = make_queue_module(outputs_complete=True)
        with fake_submodule("gnina_queue", module):
            self.assertTrue(batch_dock.gnina_ready("/some/job"))
        self.assertIn(("outputs_complete", "/some/job"), module.calls)

    def test_previous_effective_cpu_reads_the_frozen_params(self):
        queue_root = os.path.join(self.run_dir, "gnina_queue")
        os.makedirs(queue_root)
        with patched(GNINA_QUEUE_DIR=None):
            self.assertIsNone(batch_dock._previous_effective_cpu(self.run_dir))
            with open(os.path.join(queue_root, "params.json"), "w") as handle:
                json.dump({"cpu": 24}, handle)
            self.assertEqual(24, batch_dock._previous_effective_cpu(self.run_dir))
            with open(os.path.join(queue_root, "params.json"), "w") as handle:
                handle.write("{ truncated")
            self.assertIsNone(batch_dock._previous_effective_cpu(self.run_dir))

    def test_cli_exposes_the_async_switches(self):
        """The flags are the operator's interface; --help must show them."""
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "batch_dock.py"), "--help"],
            cwd=self.dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=300)
        text = result.stdout.decode("utf-8", "replace")
        self.assertEqual(0, result.returncode, text)
        for flag in ("--gnina-async", "--gnina-barrier-timeout", "--strict-caps",
                     "--no-collector", "--gnina-partial-rmsd", "--queue-dir"):
            self.assertIn(flag, text)


# ---------------------------------------------------------------------------
# the same wiring, against the real queue rather than a stand-in
# ---------------------------------------------------------------------------

try:
    from dock_selectivity import gnina_queue as real_gnina_queue
except ImportError:                              # pragma: no cover
    real_gnina_queue = None

try:
    from dock_selectivity import remote_caps as real_remote_caps
except ImportError:                              # pragma: no cover
    real_remote_caps = None


# The measured capacity of the GPU host, verbatim, as a canned probe. Using it as
# a fixture is deliberate: these tests must never touch that machine (one GPU,
# several agents), and the numbers are what the production run will meet.
MEASURED_PROBE = (
    "cores=24\n"
    "mem_total_kb=32505856\n"
    "mem_avail_kb=30000000\n"
    "gpu=NVIDIA GeForce RTX 3060 Ti, 8192, 1, 8.6\n"
    "docker=1\n"
    "image=1\n"
    "native=0\n"
    "disk_free_kb=200000000\n"
    "uptime=100000\n"
)


@unittest.skipIf(real_gnina_queue is None, "dock_selectivity.gnina_queue not present")
class TestAgainstTheRealQueue(AsyncGninaTestCase):
    """Stand-ins prove batch_dock calls the right things; this proves they fit.

    Still hermetic: the queue is local JSON state, nothing here reaches SSH, the
    GPU host or a docking engine. No collector is started, and the barrier is
    told not to wait -- the point is the handover, not the waiting.
    """

    def test_enqueue_barrier_and_exclusion_report_against_the_real_modules(self):
        done = self.build_job("DB00035__P30518__AVPR2__F1__cav1",
                              tools=("smina", "gnina"))
        with open(os.path.join(done["current_outfolder"], "gnina", "out.sdf"),
                  "w") as handle:
            handle.write("mol\n$$$$\n")          # outputs_complete() wants poses too
        todo = self.build_job("DB00040__P12345__XYZ__F1__cav2", tools=("smina",))
        fake_stage, fake_adaptive = self.recorded_stages()

        with patched(GNINA_ASYNC=True, GNINA_QUEUE_DIR=None,
                     GNINA_COLLECTOR_AUTOSTART=False, GNINA_LEAVE_COLLECTOR=False,
                     GNINA_BARRIER_TIMEOUT=0.0, GNINA_PARTIAL_RMSD=False,
                     USE_SMINA=True, USE_GNINA=True, USE_GOLD=False,
                     USE_LEDOCK=False, USE_RMSD_CALCULATION=True,
                     NUM_THREADS=60, GNINA_EXHAUSTIVENESS=32,
                     run_stage_jobs=fake_stage, run_stage_jobs_adaptive=fake_adaptive,
                     resolve_gnina_caps=lambda run_dir, strict=False:
                         types.SimpleNamespace(effective_cpu=24)):
            with self.assertLogs(level="WARNING") as captured:
                self.run_docking([done, todo])

        queue_root = os.path.join(self.run_dir, "gnina_queue")
        with open(os.path.join(queue_root, "params.json")) as handle:
            params = json.load(handle)
        self.assertEqual(24, params["cpu"], "the clamped --cpu is what got frozen")
        self.assertIsNone(params["num_modes"])

        queue = real_gnina_queue.GninaQueue(self.run_dir, create=False)
        states = {record.job_id: record.state for record in queue.read_all()}
        self.assertEqual({"DB00035__P30518__AVPR2__F1__cav1": "COLLECTED",
                          "DB00040__P12345__XYZ__F1__cav2": "PENDING"}, states)

        # Stage 5 runs for the finished job only, and says so about the other.
        self.assertEqual(["DB00035__P30518__AVPR2__F1__cav1"],
                         self.jobs_for("Stage 5: RMSD"))
        self.assertIn("DB00040__P12345__XYZ__F1__cav2", "\n".join(captured.output))

        with open(os.path.join(queue_root, "summary.json")) as handle:
            summary = json.load(handle)
        self.assertEqual(["DB00040__P12345__XYZ__F1__cav2"],
                         summary["rmsd_excluded_jobs"])
        self.assertEqual(1, summary["outstanding"])
        self.assertEqual(1, summary["collected"])

    @unittest.skipIf(real_remote_caps is None, "remote_caps not present")
    def test_todays_production_config_passes_the_real_capability_check(self):
        """cpu 60 -> 24, exhaustiveness 32, deadline 1800: warn, clamp, proceed.

        The rules must not kill the campaign they exist to protect. Uses the
        recorded probe -- the GPU host is never contacted.
        """
        caps_file = os.path.join(self.dir, "caps.json")
        real_remote_caps.save_caps(
            real_remote_caps.parse_probe_output(MEASURED_PROBE, "user@gpu", 50000),
            caps_file)
        shim = os.path.join(REPO_ROOT, "gnina_remote.py")
        with mock.patch.dict(os.environ, {"GNINA_REMOTE_HOST": "user@gpu"}), \
                patched(GNINA_ASYNC=True, GNINA_CAPS_FILE=caps_file,
                        GNINA_QUEUE_DIR=None, NUM_THREADS=60,
                        GNINA_EXHAUSTIVENESS=32, GNINA_JOB_DEADLINE=1800.0,
                        GNINA_PATH=shim):
            report = batch_dock.resolve_gnina_caps(self.run_dir)
        self.assertIsNotNone(report)
        self.assertEqual(24, report.effective_cpu)
        self.assertTrue(report.ok)
        self.assertEqual(["CPU_OVERSUBSCRIBED"],
                         [f.code for f in report.findings])

    @unittest.skipIf(real_remote_caps is None, "remote_caps not present")
    def test_an_impossible_exhaustiveness_exits_78_for_real(self):
        caps_file = os.path.join(self.dir, "caps.json")
        real_remote_caps.save_caps(
            real_remote_caps.parse_probe_output(MEASURED_PROBE, "user@gpu", 50000),
            caps_file)
        shim = os.path.join(REPO_ROOT, "gnina_remote.py")
        with mock.patch.dict(os.environ, {"GNINA_REMOTE_HOST": "user@gpu"}), \
                patched(GNINA_ASYNC=True, GNINA_CAPS_FILE=caps_file,
                        GNINA_QUEUE_DIR=None, NUM_THREADS=60,
                        GNINA_EXHAUSTIVENESS=512, GNINA_JOB_DEADLINE=1800.0,
                        GNINA_PATH=shim):
            with self.assertLogs(level="CRITICAL") as captured:
                with self.assertRaises(SystemExit) as caught:
                    batch_dock.resolve_gnina_caps(self.run_dir)
        self.assertEqual(78, caught.exception.code)
        text = "\n".join(captured.output)
        self.assertIn("EXHAUSTIVENESS_TOO_HIGH", text)
        self.assertIn("nothing has been enqueued", text)

    @unittest.skipIf(real_remote_caps is None, "remote_caps not present")
    def test_async_with_the_wrong_gnina_path_is_refused(self):
        """Async mode goes through gnina_remote.py; a native binary cannot queue."""
        caps_file = os.path.join(self.dir, "caps.json")
        real_remote_caps.save_caps(
            real_remote_caps.parse_probe_output(MEASURED_PROBE, "user@gpu", 50000),
            caps_file)
        with mock.patch.dict(os.environ, {"GNINA_REMOTE_HOST": "user@gpu"}), \
                patched(GNINA_ASYNC=True, GNINA_CAPS_FILE=caps_file,
                        GNINA_QUEUE_DIR=None, NUM_THREADS=24,
                        GNINA_EXHAUSTIVENESS=32, GNINA_JOB_DEADLINE=1800.0,
                        GNINA_PATH="/home/onur/software/gnina.1.3.2"):
            with self.assertRaises(SystemExit) as caught:
                batch_dock.resolve_gnina_caps(self.run_dir)
        self.assertEqual(78, caught.exception.code)

    def test_run_dir_and_queue_root_agree_with_the_queue_module(self):
        with patched(GNINA_QUEUE_DIR=None):
            self.assertEqual(real_gnina_queue.queue_root(self.run_dir),
                             batch_dock._gnina_queue_root(self.run_dir))
        elsewhere = os.path.join(self.dir, "ext4_queue")
        with patched(GNINA_QUEUE_DIR=elsewhere):
            self.assertEqual(real_gnina_queue.queue_root(self.run_dir, elsewhere),
                             batch_dock._gnina_queue_root(self.run_dir))


if __name__ == "__main__":
    unittest.main()
