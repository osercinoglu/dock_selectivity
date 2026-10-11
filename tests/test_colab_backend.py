"""Tests for the Colab GPU backend of gnina_remote.py.

Hermetic: no ``colab`` CLI, no Google account, no GPU, no network. Every test
either inspects generated source or drives ``_run_colab`` against a stub session,
because the live path costs compute units and cannot run in CI.

What matters here, in order:

* **The remote snippets are real Python.** They are built as strings and executed
  on a machine this process cannot see, so a syntax error in them surfaces as a
  docking failure ten minutes into a run. Each one is compiled here.
* **The job must be detached.** A docking run is minutes long and the websocket
  will drop; ``setsid`` plus an exit-code marker is what makes that survivable,
  exactly as in the SSH backend.
* **Resume must not re-dock.** A finished job is collected. GPU minutes are the
  scarce resource and they are billed against a subscription.
* **A failure must never look like a success.** consensus_docker.py reads exit
  codes; the Colab backend reuses the SSH backend's, so the async queue and the
  retry policy need no special case.

The live path was verified by hand against a real T4 and is written up in
SETUP.md.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import gnina_remote as gr  # noqa: E402
from dock_selectivity import colab_backend as cb  # noqa: E402

#: The exact box consensus_docker.py passes, from the pilot's own log.
PASSTHROUGH = [
    "--center_x", "1.0795002", "--center_y", "-0.015500069", "--center_z", "-12.14",
    "--size_x", "30.035", "--size_y", "29.029", "--size_z", "30.516",
    "--cpu", "60", "--num_modes", "20", "--exhaustiveness", "32",
]


def _env(**values):
    """Context-managed environment, restoring whatever was there."""
    import contextlib

    @contextlib.contextmanager
    def manager():
        saved = {k: os.environ.get(k) for k in values}
        try:
            for key, value in values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            yield
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    return manager()


class StubSession(object):
    """A Colab session that records what was asked of it and invents nothing.

    Mirrors only the surface ``_run_colab`` uses. ``probe_states`` is consumed one
    call at a time so a test can script "running, running, done".
    """

    def __init__(self, probe_states=None, exists=True, out_bytes=4096):
        self.session = "stub"
        self.gpu = "T4"
        self.remote_root = cb.DEFAULT_REMOTE_ROOT
        self.calls = []
        self.uploaded = []
        self.downloaded = []
        self.stopped = False
        self._exists = exists
        self._probe_states = list(probe_states or [{"exit_code": None, "out_bytes": 0}])
        self._out_bytes = out_bytes
        self.lock_free = True
        self.log = lambda message: None

    def exists(self):
        self.calls.append("exists")
        return self._exists

    def ensure(self):
        self.calls.append("ensure")
        return True

    def ensure_gnina(self):
        self.calls.append("ensure_gnina")
        return {"ok": True, "version": "gnina v1.3.3", "gpu": "Tesla T4, 15360 MiB"}

    def __init_lock__(self):
        pass

    def exec_python(self, code, what="exec", timeout=None):
        self.calls.append("exec:%s" % what)
        if "gpu lock" in what:
            return json.dumps({"acquired": self.lock_free,
                               "owner": {"owner": "other:1"}, "age": 5})
        if "probe" in what:
            state = self._probe_states.pop(0) if len(self._probe_states) > 1 \
                else self._probe_states[0]
            return json.dumps(state)
        return "{}"

    def upload(self, local, remote):
        self.calls.append("upload")
        self.uploaded.append((local, remote))

    def download(self, remote, local):
        self.calls.append("download")
        self.downloaded.append((remote, local))
        with open(local, "w") as handle:
            handle.write("x" * self._out_bytes)

    def stop(self):
        self.calls.append("stop")
        self.stopped = True


class _Ticket(object):
    def __init__(self):
        self.data = {}
        self.outcome = None

    def set(self, **kw):
        self.data.update(kw)

    def finish(self, outcome, **kw):
        self.outcome = outcome
        self.data.update(kw)


class TestGeneratedRemoteSource(unittest.TestCase):
    """These strings run on a machine we cannot see. Compile them here."""

    def test_every_snippet_is_valid_python(self):
        snippets = {
            "install": cb.gnina_install_source(),
            "launch": cb.launch_source("/content/gnina_jobs/gnina_abc", "r.pdb",
                                       "l.sdf", PASSTHROUGH),
            "probe": cb.probe_source("/content/gnina_jobs/gnina_abc"),
        }
        for name, source in snippets.items():
            with self.subTest(snippet=name):
                compile(source, "<%s>" % name, "exec")

    def test_the_installer_hunts_every_soname_colab_lacks(self):
        """Colab ships CUDA 13; the binary needs these four from CUDA 12 wheels."""
        source = cb.gnina_install_source()
        for soname in ("libcudart.so.12", "libcufft.so.11", "libcublas.so.12",
                       "libcublasLt.so.12"):
            self.assertIn(soname, source)

    def test_the_installer_does_not_hardcode_a_python_version(self):
        """Colab upgrades Python without notice; the wheel path carries it."""
        source = cb.gnina_install_source()
        self.assertIn("python3*", source)
        self.assertNotIn("python3.13/dist-packages/nvidia/cuda_runtime", source)

    def test_the_installer_verifies_rather_than_assumes(self):
        """A downloaded file that cannot execute is not an installation."""
        source = cb.gnina_install_source()
        self.assertIn("--version", source)
        self.assertIn("ok=False", source)

    def test_a_custom_release_url_reaches_the_snippet(self):
        source = cb.gnina_install_source("https://example.invalid/gnina")
        self.assertIn("https://example.invalid/gnina", source)

    def test_the_launch_detaches_and_records_its_exit_code(self):
        """A dropped websocket must not kill or orphan a ten-minute run."""
        source = cb.launch_source("/content/gnina_jobs/gnina_abc", "r.pdb", "l.sdf",
                                  PASSTHROUGH)
        self.assertIn("setsid", source)
        self.assertIn("nohup", source)
        self.assertIn("exit_code", source)
        self.assertIn("start_new_session=True", source)

    def test_the_launch_passes_the_docking_box_through_unchanged(self):
        source = cb.launch_source("/content/gnina_jobs/gnina_abc", "r.pdb", "l.sdf",
                                  PASSTHROUGH)
        for token in ("--center_x", "1.0795002", "--exhaustiveness", "32",
                      "--num_modes", "20"):
            self.assertIn(token, source)

    def test_the_launch_quotes_paths(self):
        """A job directory is content-addressed, but the names come from outside."""
        source = cb.launch_source("/content/jobs/weird dir", "r e c.pdb", "l.sdf", [])
        self.assertIn("'/content/jobs/weird dir'", source)
        self.assertIn("'/content/jobs/weird dir/r e c.pdb'", source)

    def test_the_probe_reports_the_fields_the_caller_branches_on(self):
        source = cb.probe_source("/content/gnina_jobs/gnina_abc")
        for field in ("exit_code", "out_bytes", "stderr_tail"):
            self.assertIn(field, source)


class TestTheGpuIsActuallyUsed(unittest.TestCase):
    """A silent CPU fallback is the expensive failure, not a crash.

    Measured on a real T4: replacing LD_LIBRARY_PATH instead of extending it
    dropped /usr/lib64-nvidia, which holds libcuda.so.1. gnina then loaded
    happily, logged "No GPU detected", and finished the job on 2 CPU cores -- 30+
    minutes against a predicted 2.5 on a GPU, on a runtime billed by the hour for
    its card. Valid chemistry, wrong by every other measure, and nothing in the
    stack noticed.
    """

    def test_the_installer_seeds_from_the_runtimes_own_library_path(self):
        self.assertIn('os.environ.get("LD_LIBRARY_PATH")', cb.gnina_install_source())

    def test_the_launch_extends_rather_than_replaces_the_library_path(self):
        source = cb.launch_source("/content/gnina_jobs/gnina_abc", "r.pdb", "l.sdf", [])
        self.assertIn(":$LD_LIBRARY_PATH", source)

    def test_the_installer_refuses_a_runtime_whose_gnina_sees_no_gpu(self):
        source = cb.gnina_install_source()
        self.assertIn("No GPU detected", source)
        self.assertIn("ok=False", source)

    def test_the_probe_reports_a_cpu_fallback(self):
        self.assertIn("no_gpu", cb.probe_source("/content/gnina_jobs/gnina_abc"))


class TestTransientClassification(unittest.TestCase):
    """Retry a dropped socket; never retry a configuration mistake."""

    def test_dropped_sockets_and_server_errors_are_transient(self):
        for text in ("Session not found", "connection reset by peer", "HTTP 503",
                     "websocket closed", "429 Too Many Requests", "read timed out"):
            with self.subTest(text=text):
                self.assertTrue(cb.is_transient(1, text))

    def test_real_errors_are_not_transient(self):
        for text in ("no GPU entitlement for this account",
                     "invalid accelerator", "permission denied", "quota exceeded"):
            with self.subTest(text=text):
                self.assertFalse(cb.is_transient(1, text))

    def test_success_is_never_transient(self):
        self.assertFalse(cb.is_transient(0, "Session not found"))


class TestSessionConfiguration(unittest.TestCase):
    def test_global_flags_precede_the_subcommand(self):
        """The CLI rejects --auth/--config after the subcommand."""
        with _env(GNINA_COLAB_CONFIG="/tmp/state.json", GNINA_COLAB_AUTH="adc"):
            base = cb.ColabSession().base()
        self.assertEqual(base[0], "colab")
        self.assertIn("--auth=adc", base)
        self.assertEqual(base[-2:], ["--config", "/tmp/state.json"])

    def test_an_unknown_gpu_is_refused_rather_than_passed_on(self):
        """The CLI silently falls back to A100, which then fails to allocate."""
        with _env(GNINA_COLAB_GPU="RTX4090"):
            with self.assertRaises(cb.ColabError) as ctx:
                cb.ColabSession()
        self.assertIn("A100", str(ctx.exception))

    def test_every_documented_gpu_is_accepted(self):
        for gpu in cb.GPUS:
            with self.subTest(gpu=gpu), _env(GNINA_COLAB_GPU=gpu):
                self.assertEqual(cb.ColabSession().gpu, gpu)

    def test_defaults_are_a_named_t4_session(self):
        with _env(GNINA_COLAB_SESSION=None, GNINA_COLAB_GPU=None):
            session = cb.ColabSession()
        self.assertEqual(session.session, cb.DEFAULT_SESSION)
        self.assertEqual(session.gpu, "T4")

    def test_a_missing_cli_names_the_install_command_and_the_gcloud_trap(self):
        session = cb.ColabSession()
        session.base = lambda: ["definitely-not-a-real-binary-xyz"]
        with self.assertRaises(cb.ColabUnavailable) as ctx:
            session.run(["usage"], "usage", timeout=10)
        message = str(ctx.exception)
        self.assertIn("google-colab-cli", message)
        self.assertIn("Colab Enterprise", message)


class TestResultParsing(unittest.TestCase):
    def test_the_last_json_object_wins_over_kernel_chatter(self):
        text = ("Downloading...\n"
                '{"ok": false, "error": "early"}\n'
                "more progress\n"
                '{"ok": true, "version": "gnina v1.3.3"}\n')
        self.assertEqual(cb._last_json(text)["version"], "gnina v1.3.3")

    def test_no_json_at_all_is_none_not_a_crash(self):
        self.assertIsNone(cb._last_json("nothing structured here"))
        self.assertIsNone(cb._last_json(""))

    def test_malformed_json_is_skipped_not_raised(self):
        self.assertEqual(cb._last_json('{"broken": \n{"ok": 1}')["ok"], 1)

    def test_usage_is_parsed_into_numbers(self):
        session = cb.ColabSession()
        session.run = lambda *a, **k: type(
            "P", (), {"stdout": "Current balance: 1784.33 compute units\n"
                                "Usage rate: 1.07/hr\nActive assignments: 1\n",
                      "stderr": "", "returncode": 0})()
        usage = session.usage()
        self.assertAlmostEqual(usage["balance"], 1784.33, places=2)
        self.assertAlmostEqual(usage["rate_per_hour"], 1.07, places=2)


class TestFittingToTheRuntime(unittest.TestCase):
    """A Colab GPU comes with very few vCPUs, and gnina's search is CPU-parallel.

    Measured: exhaustiveness 32 on the 24-core SSH host takes 2m23s; the same job
    on a 2-vCPU Colab T4 is ~30 min by the repo's own calibration. The GPU is not
    the bottleneck, so the thread count has to follow the runtime.
    """

    def test_cpu_is_clamped_to_the_runtime_and_says_so(self):
        said = []
        out = cb.fit_to_runtime(["--cpu", "60", "--exhaustiveness", "8"], cores=2,
                                deadline_seconds=7200, log=said.append)
        self.assertEqual(out[out.index("--cpu") + 1], "2")
        self.assertTrue(any("60" in m and "2" in m for m in said),
                        "the clamp must name both numbers: %s" % said)

    def test_a_cpu_request_below_the_core_count_is_left_alone(self):
        out = cb.fit_to_runtime(["--cpu", "2", "--exhaustiveness", "8"], cores=8,
                                deadline_seconds=7200, log=lambda m: None)
        self.assertEqual(out[out.index("--cpu") + 1], "2")

    def test_num_threads_is_clamped_too(self):
        out = cb.fit_to_runtime(["--num_threads", "60"], cores=4,
                                deadline_seconds=7200, log=lambda m: None)
        self.assertEqual(out[out.index("--num_threads") + 1], "4")

    def test_exhaustiveness_is_refused_not_quietly_lowered(self):
        """Lowering it would make two jobs in one campaign incomparable."""
        with self.assertRaises(cb.ColabRefused) as ctx:
            cb.fit_to_runtime(["--cpu", "60", "--exhaustiveness", "32"], cores=2,
                              deadline_seconds=600, log=lambda m: None)
        message = str(ctx.exception)
        self.assertIn("never clamped", message)
        self.assertIn("lower it to", message)

    def test_the_refusal_quotes_a_ceiling_that_actually_fits(self):
        """A remedy that does not work is worse than no remedy.

        Checked against the COLAB constant, not remote_caps': the SSH host's
        112.5 core-seconds per exhaustiveness unit under-predicts this hardware
        by about 3x, and an under-predicting deadline model lets a job through
        and then misses the deadline.
        """
        deadline, cores = 600.0, 2
        with self.assertRaises(cb.ColabRefused) as ctx:
            cb.fit_to_runtime(["--exhaustiveness", "32"], cores=cores,
                              deadline_seconds=deadline, log=lambda m: None)
        ceiling = int(str(ctx.exception).split("lower it to <= ")[1].split(",")[0])
        predicted = cb.COLAB_CORE_SECONDS_PER_EXH * ceiling / cores
        self.assertLessEqual(predicted, deadline)

    def test_the_colab_model_is_more_conservative_than_the_ssh_one(self):
        """Measured: 2.00 GHz Colab vCPUs are far slower than the host's cores."""
        from dock_selectivity import remote_caps
        self.assertGreater(cb.COLAB_CORE_SECONDS_PER_EXH,
                           remote_caps.CORE_SECONDS_PER_EXH_UNIT)

    def test_an_affordable_exhaustiveness_passes_with_a_prediction(self):
        said = []
        out = cb.fit_to_runtime(["--exhaustiveness", "8"], cores=8,
                                deadline_seconds=7200, log=said.append)
        self.assertEqual(out, ["--exhaustiveness", "8"])
        self.assertTrue(any("predicted" in m for m in said), said)

    def test_the_box_survives_fitting_untouched(self):
        """Fitting may only touch thread counts; the geometry is the experiment."""
        out = cb.fit_to_runtime(PASSTHROUGH, cores=2, deadline_seconds=36000,
                                log=lambda m: None)
        for index, token in enumerate(PASSTHROUGH):
            if token.startswith("--center") or token.startswith("--size"):
                self.assertEqual(out[out.index(token) + 1], PASSTHROUGH[index + 1])
        self.assertEqual(out[out.index("--num_modes") + 1], "20")

    def test_a_non_numeric_thread_count_is_passed_through_not_crashed_on(self):
        out = cb.fit_to_runtime(["--cpu", "all"], cores=2, deadline_seconds=7200,
                                log=lambda m: None)
        self.assertEqual(out, ["--cpu", "all"])


class TestGpuMutex(unittest.TestCase):
    """One runtime has one GPU, and gnina_queue sets GNINA_REMOTE_LOCK on submit.

    Two detached gnina processes on one T4 contend for VRAM and both run slower
    than if they had queued, so a shim that ignores the flag quietly undoes the
    sequencing the queue is asking for.
    """

    def test_the_lock_is_a_directory_because_mkdir_is_atomic(self):
        source = cb.lock_source("/content/gnina_jobs", "submit:1")
        self.assertIn("os.mkdir", source)
        self.assertIn(cb.LOCK_DIRNAME, source)

    def test_a_stale_lock_is_taken_over_not_waited_on_forever(self):
        """Its holder may have been a runtime that was pruned mid-job."""
        source = cb.lock_source("/content/gnina_jobs", "submit:1", stale_seconds=900)
        self.assertIn("900", source)
        self.assertIn("rmtree", source)

    def test_the_detached_job_releases_the_lock_after_writing_its_exit_code(self):
        source = cb.launch_source("/content/gnina_jobs/gnina_abc", "r.pdb", "l.sdf",
                                  [], lock_root="/content/gnina_jobs")
        self.assertIn(cb.LOCK_DIRNAME, source)
        self.assertLess(source.index("exit_code"), source.index(cb.LOCK_DIRNAME))

    def test_no_lock_is_touched_when_none_was_taken(self):
        source = cb.launch_source("/content/gnina_jobs/gnina_abc", "r.pdb", "l.sdf", [])
        self.assertNotIn(cb.LOCK_DIRNAME, source)


class TestBackendSelection(unittest.TestCase):
    def test_ssh_is_the_default(self):
        with _env(GNINA_REMOTE_BACKEND=None):
            self.assertEqual(gr.resolve_backend(), gr.BACKEND_SSH)

    def test_an_unknown_backend_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            receptor = os.path.join(tmp, "r.pdb")
            ligand = os.path.join(tmp, "l.sdf")
            for path in (receptor, ligand):
                open(path, "w").close()
            with _env(GNINA_REMOTE_BACKEND="aws"):
                rc = gr.main(["-r", receptor, "-l", ligand,
                              "-o", os.path.join(tmp, "o.sdf")])
        self.assertEqual(rc, gr.EXIT_USAGE)

    def test_the_colab_backend_does_not_require_an_ssh_host(self):
        """The whole point: it is the alternative for when that host is busy."""
        calls = []

        class Unavailable(StubSession):
            def ensure(self):
                calls.append("ensure")
                raise cb.ColabUnavailable("no cli")

        with tempfile.TemporaryDirectory() as tmp:
            receptor = os.path.join(tmp, "r.pdb")
            ligand = os.path.join(tmp, "l.sdf")
            for path in (receptor, ligand):
                with open(path, "w") as handle:
                    handle.write("x")
            original = cb.ColabSession
            cb.ColabSession = lambda **kw: Unavailable()
            try:
                with _env(GNINA_REMOTE_BACKEND="colab", GNINA_REMOTE_HOST=None):
                    rc = gr.main(["-r", receptor, "-l", ligand,
                                  "-o", os.path.join(tmp, "o.sdf")])
            finally:
                cb.ColabSession = original
        # It got as far as trying to use Colab rather than refusing for a missing
        # GNINA_REMOTE_HOST, and a missing CLI is a usage error, not a failure.
        self.assertEqual(calls, ["ensure"])
        self.assertEqual(rc, gr.EXIT_USAGE)


class TestColabJobLifecycle(unittest.TestCase):
    """``_run_colab`` against a stub: the decisions, not the transport."""

    def setUp(self):
        # The directory outlives the call: an earlier version of this helper
        # asserted on known.out after the TemporaryDirectory context had already
        # removed it, so a genuine download looked like a missing file.
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _job(self, stub, mode=gr.MODE_SYNC, out_name="out.sdf", **env):
        ticket = _Ticket()
        receptor = os.path.join(self.tmp.name, "r.pdb")
        ligand = os.path.join(self.tmp.name, "l.sdf")
        with open(receptor, "w") as handle:
            handle.write("ATOM\n")
        with open(ligand, "w") as handle:
            handle.write("lig\n")
        known = type("K", (), {
            "receptor": receptor, "ligand": ligand,
            "out": os.path.join(self.tmp.name, out_name), "log": None})()
        defaults = {"GNINA_REMOTE_POLL": "0", "GNINA_COLAB_STOP": None,
                    "GNINA_COLAB_KEEP": "1"}
        defaults.update(env)
        original = cb.ColabSession
        cb.ColabSession = lambda **kw: stub
        try:
            with _env(**defaults):
                rc = gr._run_colab(known, PASSTHROUGH, lambda m: None, ticket, mode)
        finally:
            cb.ColabSession = original
        return rc, ticket, known

    def test_a_finished_job_is_collected_not_redocked(self):
        """GPU minutes are billed; re-docking a finished job spends them twice."""
        stub = StubSession(probe_states=[{"exit_code": "0", "out_bytes": 4096}])
        rc, ticket, _ = self._job(stub)
        self.assertEqual(rc, gr.EXIT_OK)
        self.assertEqual(ticket.outcome, "collected")
        self.assertNotIn("upload", stub.calls)
        self.assertNotIn("exec:launch gnina", stub.calls)
        self.assertIn("download", stub.calls)

    def test_a_fresh_job_uploads_launches_polls_then_downloads(self):
        stub = StubSession(probe_states=[
            {"exit_code": None, "out_bytes": 0},
            {"exit_code": None, "out_bytes": 128},
            {"exit_code": "0", "out_bytes": 4096},
        ])
        rc, ticket, known = self._job(stub)
        self.assertEqual(rc, gr.EXIT_OK)
        self.assertEqual(len(stub.uploaded), 2)
        self.assertIn("exec:launch gnina", stub.calls)
        self.assertTrue(os.path.exists(known.out))
        self.assertLess(stub.calls.index("exec:mkdir job dir"),
                        stub.calls.index("upload"))

    def test_the_job_directory_is_content_addressed(self):
        stub = StubSession(probe_states=[{"exit_code": "0", "out_bytes": 4096}])
        _, ticket, _ = self._job(stub)
        self.assertTrue(ticket.data["job_dir"].startswith(cb.DEFAULT_REMOTE_ROOT))
        self.assertIn("gnina_", ticket.data["job_dir"])
        self.assertEqual(ticket.data["backend"], "colab")

    def test_a_nonzero_exit_fails_the_job_and_keeps_the_directory(self):
        """Never a successful-looking empty result: consensus_docker reads codes."""
        stub = StubSession(probe_states=[
            {"exit_code": "1", "out_bytes": 0,
             "stderr_tail": "CUDA error: out of memory"}])
        rc, ticket, known = self._job(stub)
        self.assertEqual(rc, gr.EXIT_FAILED)
        self.assertEqual(ticket.outcome, "error")
        self.assertIn("out of memory", ticket.data["error"])
        self.assertFalse(os.path.exists(known.out))

    def test_submit_reports_launched_the_word_the_queue_matches_on(self):
        """gnina_queue moves a record to SUBMITTED on outcome == "launched".

        Any other word falls through to SUBMIT_NO_TICKET, so a job that really
        did launch is recorded as failed and the GPU work is thrown away.
        """
        stub = StubSession(probe_states=[{"exit_code": None, "out_bytes": 0}])
        rc, ticket, _ = self._job(stub, mode=gr.MODE_SUBMIT)
        self.assertEqual(rc, gr.EXIT_SUBMITTED)
        self.assertEqual(ticket.outcome, "launched")
        self.assertIn("exec:launch gnina", stub.calls)
        self.assertNotIn("download", stub.calls)

    def test_submit_on_a_finished_job_reports_already_done_and_does_not_download(self):
        """The queue takes this to REMOTE_DONE; collect is a separate phase."""
        stub = StubSession(probe_states=[{"exit_code": "0", "out_bytes": 4096}])
        rc, ticket, _ = self._job(stub, mode=gr.MODE_SUBMIT)
        self.assertEqual(rc, gr.EXIT_SUBMITTED)
        self.assertEqual(ticket.outcome, "already_done")
        self.assertNotIn("download", stub.calls)
        self.assertNotIn("exec:launch gnina", stub.calls)

    def test_a_stale_local_out_sdf_is_removed_before_launching(self):
        """split_mol would succeed on stale bytes and score the previous run."""
        stub = StubSession(probe_states=[{"exit_code": None, "out_bytes": 0}])
        stale = os.path.join(self.tmp.name, "out.sdf")
        with open(stale, "w") as handle:
            handle.write("PREVIOUS RUN POSES")
        self._job(stub, mode=gr.MODE_SUBMIT)
        self.assertFalse(os.path.exists(stale),
                         "a previous attempt's poses survived into this job")

    def test_the_outcome_vocabulary_is_exactly_what_the_queue_knows(self):
        """Pinned against the queue itself, so a rename there breaks here first."""
        import inspect
        from dock_selectivity import gnina_queue
        source = inspect.getsource(gnina_queue.submit_job)
        for word in ("busy", "already_done", "launched", "error"):
            self.assertIn('"%s"' % word, source,
                          "the queue no longer recognises %r" % word)

    def test_submit_never_stops_the_session_it_just_filled(self):
        """Stopping after a submit would discard the job that was just launched."""
        stub = StubSession(probe_states=[{"exit_code": None, "out_bytes": 0}])
        self._job(stub, mode=gr.MODE_SUBMIT, GNINA_COLAB_STOP="1")
        self.assertFalse(stub.stopped)

    def test_collect_on_an_unfinished_job_is_not_ready(self):
        stub = StubSession(probe_states=[{"exit_code": None, "out_bytes": 512}])
        rc, ticket, _ = self._job(stub, mode=gr.MODE_COLLECT)
        self.assertEqual(rc, gr.EXIT_NOT_READY)
        self.assertEqual(ticket.outcome, "not_ready")

    def test_collect_against_a_pruned_session_reports_gone(self):
        """The runtime took the job directory with it; it must be re-docked."""
        stub = StubSession(exists=False)
        rc, ticket, _ = self._job(stub, mode=gr.MODE_COLLECT)
        self.assertEqual(rc, gr.EXIT_NOT_READY)
        self.assertEqual(ticket.outcome, "gone")

    def test_collect_downloads_a_finished_job(self):
        stub = StubSession(probe_states=[{"exit_code": "0", "out_bytes": 4096}])
        rc, ticket, known = self._job(stub, mode=gr.MODE_COLLECT)
        self.assertEqual(rc, gr.EXIT_OK)
        self.assertTrue(os.path.exists(known.out))
        self.assertNotIn("ensure", stub.calls)

    def test_stop_is_opt_in(self):
        """A campaign shares one session; stopping per job pays allocation each time."""
        stub = StubSession(probe_states=[{"exit_code": "0", "out_bytes": 4096}])
        self._job(stub)
        self.assertFalse(stub.stopped)

        stub = StubSession(probe_states=[{"exit_code": "0", "out_bytes": 4096}])
        self._job(stub, GNINA_COLAB_STOP="1")
        self.assertTrue(stub.stopped)

    def test_a_busy_gpu_mutex_yields_rather_than_double_booking(self):
        stub = StubSession(probe_states=[{"exit_code": None, "out_bytes": 0}])
        stub.lock_free = False
        rc, ticket, _ = self._job(stub, mode=gr.MODE_SUBMIT,
                                  GNINA_REMOTE_LOCK="1")
        self.assertEqual(rc, gr.EXIT_LOCK_BUSY)
        self.assertEqual(ticket.outcome, "busy")
        self.assertNotIn("exec:launch gnina", stub.calls)

    def test_the_mutex_is_taken_after_the_upload_not_before(self):
        """A job that loses the race has still staged its inputs for the next try."""
        stub = StubSession(probe_states=[{"exit_code": None, "out_bytes": 0}])
        self._job(stub, mode=gr.MODE_SUBMIT, GNINA_REMOTE_LOCK="1")
        self.assertLess(stub.calls.index("upload"),
                        stub.calls.index("exec:take gpu lock"))

    def test_without_the_flag_no_mutex_is_taken(self):
        stub = StubSession(probe_states=[{"exit_code": None, "out_bytes": 0}])
        self._job(stub, mode=gr.MODE_SUBMIT, GNINA_REMOTE_LOCK=None)
        self.assertNotIn("exec:take gpu lock", stub.calls)

    def test_an_unaffordable_exhaustiveness_is_a_usage_error_not_a_failed_job(self):
        """Nothing ran, so there is nothing to resume and nothing to retry."""
        stub = StubSession(probe_states=[{"exit_code": None, "out_bytes": 0}])
        stub.ensure_gnina = lambda: {"ok": True, "version": "v1.3.3", "cores": 2}
        rc, ticket, _ = self._job(stub, GNINA_REMOTE_TIMEOUT="600")
        self.assertEqual(rc, gr.EXIT_USAGE)
        self.assertEqual(ticket.outcome, "refused")
        self.assertIn("never clamped", ticket.data["error"])
        self.assertNotIn("upload", stub.calls)

    def test_a_runtime_that_can_afford_it_proceeds(self):
        stub = StubSession(probe_states=[{"exit_code": "0", "out_bytes": 4096}])
        stub.ensure_gnina = lambda: {"ok": True, "version": "v1.3.3", "cores": 24}
        rc, _, _ = self._job(stub, GNINA_REMOTE_TIMEOUT="7200")
        self.assertEqual(rc, gr.EXIT_OK)

    def test_a_finished_job_is_collected_even_if_its_exhaustiveness_would_be_refused(self):
        """A result already on the runtime needs no docking, so the deadline it
        will never spend must not stand between the operator and their poses.

        Found live: a completed job was refused for an exhaustiveness it was not
        going to run.
        """
        stub = StubSession(probe_states=[{"exit_code": "0", "out_bytes": 4096}])
        stub.ensure_gnina = lambda: {"ok": True, "version": "v1.3.3", "cores": 2}
        rc, ticket, known = self._job(stub, GNINA_REMOTE_TIMEOUT="600")
        self.assertEqual(rc, gr.EXIT_OK)
        self.assertEqual(ticket.outcome, "collected")
        self.assertTrue(os.path.exists(known.out))

    def test_an_empty_pose_file_is_a_failure_not_a_negative_result(self):
        """split_mol on empty bytes yields an empty results.csv, which reads
        downstream as "this pair does not bind" rather than "this job broke"."""
        stub = StubSession(probe_states=[{"exit_code": "0", "out_bytes": 4096}],
                           out_bytes=0)
        rc, ticket, _ = self._job(stub)
        self.assertEqual(rc, gr.EXIT_FAILED)
        self.assertIn("not a negative result", ticket.data["error"])

    def test_a_stale_remote_exit_code_is_cleared_before_relaunching(self):
        """Otherwise the poll reads the PREVIOUS attempt's failure instantly.

        Observed: a job aborted with 134, and the next attempt was reported failed
        before gnina had started, because exit_code was still on the runtime.
        """
        stub = StubSession(probe_states=[
            {"exit_code": "134", "out_bytes": 0},     # the stale marker
            {"exit_code": "0", "out_bytes": 4096},    # the fresh run
        ])
        rc, _, _ = self._job(stub)
        self.assertEqual(rc, gr.EXIT_OK)
        self.assertIn("exec:clear stale markers", stub.calls)
        self.assertLess(stub.calls.index("exec:clear stale markers"),
                        stub.calls.index("exec:launch gnina"))

    def test_nothing_is_cleared_when_there_was_no_previous_attempt(self):
        stub = StubSession(probe_states=[
            {"exit_code": None, "out_bytes": 0},
            {"exit_code": "0", "out_bytes": 4096},
        ])
        self._job(stub)
        self.assertNotIn("exec:clear stale markers", stub.calls)

    def test_a_cpu_fallback_stops_the_job_instead_of_paying_for_it(self):
        """12x the wall time on a GPU-billed runtime, and the deadline model is
        calibrated for a GPU, so a fallback breaks the budget and the prediction
        at once."""
        stub = StubSession(probe_states=[
            {"exit_code": None, "out_bytes": 0, "no_gpu": True}])
        rc, ticket, _ = self._job(stub)
        self.assertEqual(rc, gr.EXIT_FAILED)
        self.assertIn("No GPU detected", ticket.data["error"])
        self.assertIn("libcuda.so.1", ticket.data["error"])

    def test_a_timeout_does_not_discard_the_detached_run(self):
        stub = StubSession(probe_states=[{"exit_code": None, "out_bytes": 0}])
        rc, ticket, _ = self._job(stub, GNINA_REMOTE_TIMEOUT="0")
        self.assertEqual(rc, gr.EXIT_FAILED)
        self.assertIn("still be running", ticket.data["error"])
        self.assertIn("gnina_", ticket.data["error"])


if __name__ == "__main__":
    unittest.main()
