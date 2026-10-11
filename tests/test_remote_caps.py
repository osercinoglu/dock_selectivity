"""Tests for dock_selectivity/remote_caps.py -- the remote GPU capability gate.

Hermetic by construction: **no SSH, no GPU host, no network**. The one seam that
touches the outside world is ``ssh_runner``, and every test that needs it injects
a fake. The measured capacity of the real host (24 cores, RTX 3060 Ti, 8192 MiB)
is a canned fixture below, taken from the pilot measurements -- re-probing the
box from a test would both be non-hermetic and collide with the live-validation
agent, which owns that connection.

What is being protected:

* the rules must **not kill the intended campaign** (cpu 60 -> 24, exhaustiveness
  32, deadline 1800 must produce exactly one warning and start);
* an impossible exhaustiveness must be refused *before* stage 1, with the number
  in the message;
* an unreachable host must be an error, never a defaulted optimistic record;
* the cache must stop a thousand jobs from becoming a thousand probes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from dock_selectivity import remote_caps as rc  # noqa: E402

_SILENCED = []


def setUpModule():
    """Keep the suite's output clean.

    enforce() deliberately logs and writes its report to stderr -- that is the
    behaviour under test -- but a hundred capability reports scrolling past the
    other 774 tests is noise. Silenced here, not weakened: TestEnforce still
    asserts on what was logged, through an injected logger.
    """
    import logging

    logger = logging.getLogger("dock_selectivity.remote_caps")
    _SILENCED.append((logger, logger.propagate, list(logger.handlers)))
    logger.propagate = False
    logger.addHandler(logging.NullHandler())


def tearDownModule():
    for logger, propagate, handlers in _SILENCED:
        logger.propagate = propagate
        logger.handlers = handlers
    del _SILENCED[:]


def enforce_quietly(report, **kwargs):
    """enforce() with its stderr copy of the report swallowed."""
    saved = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        return rc.enforce(report, **kwargs)
    finally:
        sys.stderr.close()
        sys.stderr = saved


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

#: Exactly what the probe printed on the real GPU host during the pilot.
REAL_PROBE_OUTPUT = """cores=24
mem_total_kb=32505856
mem_avail_kb=29142000
gpu=NVIDIA GeForce RTX 3060 Ti, 8192, 1, 8.6
docker=1
image=1
native=0
disk_free_kb=212345678
uptime=98765.43
"""

HOST = "user@gpu-host"
PORT = 50000
SHIM = "/home/onur/repos/dock_selectivity/gnina_remote.py"
STAMP = "2026-08-13T09:00:00.000000Z"


def real_caps(**overrides):
    """The measured host, with optional single-field overrides."""
    caps = rc.parse_probe_output(REAL_PROBE_OUTPUT, HOST, port=PORT, probed_at=STAMP)
    if not overrides:
        return caps
    data = caps.to_dict()
    data.update(overrides)
    return rc.RemoteCaps.from_dict(data)


def production_request(**overrides):
    """Today's batch_dock configuration: NUM_THREADS=60, exhaustiveness 32."""
    kwargs = dict(cpu=60, exhaustiveness=32, num_modes=None,
                  job_deadline_seconds=1800.0, image=rc.DEFAULT_IMAGE,
                  native=None, gnina_path=SHIM, async_mode=True)
    kwargs.update(overrides)
    return rc.DockRequest(**kwargs)


class FakeRunner(object):
    """Stands in for gnina_remote.run_resilient. Records every call."""

    class Result(object):
        def __init__(self, returncode, stdout, stderr):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def __init__(self, stdout="", stderr="", returncode=0, raises=None):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.raises = raises
        self.calls = []

    def __call__(self, cmd, what, retries, timeout, log):
        self.calls.append({"cmd": list(cmd), "what": what, "retries": retries,
                           "timeout": timeout})
        if self.raises is not None:
            raise self.raises
        return self.Result(self.returncode, self.stdout, self.stderr)


# --------------------------------------------------------------------------
# the probe command
# --------------------------------------------------------------------------

class TestProbeCommand(unittest.TestCase):
    """One round trip, and every field degrades on its own."""

    def test_every_field_is_asked_for(self):
        cmd = rc.probe_command("img:tag", None, "gnina_jobs")
        for key in ("cores=", "mem_total_kb=", "mem_avail_kb=", "gpu=",
                    "docker=", "image=", "native=", "disk_free_kb=", "uptime="):
            self.assertIn(key, cmd, key)

    def test_a_missing_tool_cannot_blind_the_whole_probe(self):
        """Each substitution has its own error suppression and its own
        fallback, so nvidia-smi's absence does not cost us the core count."""
        segments = rc.probe_command("img:tag", None, "gnina_jobs").split("; echo ")
        self.assertEqual(len(segments), 9, "one segment per probed field")
        for segment in segments:
            with self.subTest(segment=segment[:20]):
                self.assertTrue("2>/dev/null" in segment or "2>&1" in segment,
                                "stderr must not leak into the parsed output")
                self.assertTrue("|| echo 0" in segment or "head -1" in segment,
                                "a failed field must degrade, not fail the probe")

    def test_untrusted_values_cannot_inject_a_remote_command(self):
        """GNINA_REMOTE_DIR and GNINA_REMOTE_IMAGE come from the environment
        and are interpolated into a string the remote shell executes.

        Run the generated command through a real local shell and check the
        injected payload did not fire. Local sh only -- no network."""
        with tempfile.TemporaryDirectory() as tmp:
            cmd = rc.probe_command("img:tag", None, "scratch; touch pwned")
            subprocess.run(["sh", "-c", cmd], cwd=tmp, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=60)
            self.assertFalse(os.path.exists(os.path.join(tmp, "pwned")),
                             "the scratch root was interpolated unquoted")

    def test_the_image_is_quoted(self):
        cmd = rc.probe_command("img; rm -rf /", None, "gnina_jobs")
        self.assertIn("'img; rm -rf /'", cmd)

    def test_tilde_is_stripped_not_quoted(self):
        """shlex.quote('~/x') is literal to the remote shell -- it would make a
        directory actually named '~'. Same trap gnina_remote documents."""
        cmd = rc.probe_command("img", None, "~/gnina_jobs")
        self.assertIn("df -Pk gnina_jobs", cmd)
        self.assertNotIn("'~", cmd)

    def test_it_is_a_single_command(self):
        self.assertNotIn("\n", rc.probe_command("img", None, "gnina_jobs"))

    def test_native_probe_uses_the_requested_binary(self):
        cmd = rc.probe_command("img", "/opt/gnina/bin/gnina", "gnina_jobs")
        self.assertIn("command -v /opt/gnina/bin/gnina", cmd)


# --------------------------------------------------------------------------
# parsing -- must never raise
# --------------------------------------------------------------------------

class TestParseProbeOutput(unittest.TestCase):

    def test_the_real_host(self):
        caps = rc.parse_probe_output(REAL_PROBE_OUTPUT, HOST, port=PORT,
                                     probed_at=STAMP)
        self.assertEqual(caps.cores, 24)
        self.assertEqual(caps.mem_total_mb, 31744)          # 31 GB
        self.assertTrue(caps.gpu_present)
        self.assertEqual(caps.gpu_name, "NVIDIA GeForce RTX 3060 Ti")
        self.assertEqual(caps.vram_total_mb, 8192)
        self.assertEqual(caps.vram_used_mb, 1)
        self.assertEqual(caps.compute_capability, "8.6")
        self.assertTrue(caps.docker_available)
        self.assertTrue(caps.image_present)
        self.assertFalse(caps.native_present)
        self.assertEqual(caps.disk_free_mb, 212345678 // 1024)
        self.assertEqual(caps.uptime_seconds, 98765.43)
        self.assertEqual(caps.port, PORT)

    def test_missing_nvidia_smi_is_no_gpu_not_a_traceback(self):
        text = REAL_PROBE_OUTPUT.replace(
            "gpu=NVIDIA GeForce RTX 3060 Ti, 8192, 1, 8.6", "gpu=")
        caps = rc.parse_probe_output(text, HOST)
        self.assertFalse(caps.gpu_present)
        self.assertEqual(caps.vram_total_mb, 0)
        self.assertEqual(caps.cores, 24, "one missing tool must not blind the rest")

    def test_garbage_never_raises(self):
        for text in ("", "\x00\x01 not even text", "Last login: Tue Aug 13\n",
                     "cores=\ngpu=[N/A]\nmem_total_kb=[Not Supported]\n",
                     "cores=abc\ndisk_free_kb=-\nuptime=NaNish\n",
                     "=\n=\n===\n"):
            with self.subTest(text=text[:24]):
                caps = rc.parse_probe_output(text, HOST)
                self.assertIsInstance(caps, rc.RemoteCaps)
                self.assertEqual(caps.host, HOST)

    def test_unknown_keys_are_ignored(self):
        """A future field, or a chatty login shell, must not break a campaign."""
        caps = rc.parse_probe_output(
            REAL_PROBE_OUTPUT + "quantum_cores=99\nMOTD=welcome\n", HOST)
        self.assertEqual(caps.cores, 24)

    def test_a_comma_in_the_gpu_name_does_not_shift_the_numbers(self):
        caps = rc.parse_probe_output(
            "gpu=Weird, Branded GPU, 4096, 512, 7.5\n", HOST)
        self.assertEqual(caps.gpu_name, "Weird, Branded GPU")
        self.assertEqual(caps.vram_total_mb, 4096)
        self.assertEqual(caps.vram_used_mb, 512)

    def test_an_unreadable_df_is_unknown_not_zero(self):
        """`df -Pk gnina_jobs` on a host where the scratch root does not exist
        yet prints nothing. Calling that '0 MB free' would refuse a perfectly
        healthy campaign on its very first run."""
        text = REAL_PROBE_OUTPUT.replace("disk_free_kb=212345678",
                                         "disk_free_kb=")
        caps = rc.parse_probe_output(text, HOST)
        self.assertEqual(caps.disk_free_mb, rc.DISK_UNKNOWN)
        report = rc.validate_request(caps, production_request())
        self.assertTrue(report.ok, "unknown disk must not be fatal")
        self.assertIn("DISK_UNKNOWN", [f.code for f in report.findings])

    def test_a_genuinely_full_disk_is_still_fatal(self):
        text = REAL_PROBE_OUTPUT.replace("disk_free_kb=212345678",
                                         "disk_free_kb=0")
        caps = rc.parse_probe_output(text, HOST)
        self.assertEqual(caps.disk_free_mb, 0)
        report = rc.validate_request(caps, production_request())
        self.assertFalse(report.ok)
        self.assertIn("DISK_LOW", [f.code for f in report.fatal])

    def test_roundtrip_through_dict(self):
        caps = real_caps()
        again = rc.RemoteCaps.from_dict(json.loads(json.dumps(caps.to_dict())))
        self.assertEqual(caps, again)

    def test_from_dict_needs_a_host(self):
        with self.assertRaises(rc.RemoteCapsError):
            rc.RemoteCaps.from_dict({"cores": 24})


# --------------------------------------------------------------------------
# validation -- the table
# --------------------------------------------------------------------------

class TestValidateProductionConfig(unittest.TestCase):
    """THE REGRESSION TEST: the rules must not kill the intended campaign."""

    def test_todays_config_starts_with_exactly_one_warning(self):
        report = rc.validate_request(real_caps(), production_request())
        self.assertEqual([f.code for f in report.findings],
                         ["CPU_OVERSUBSCRIBED"])
        self.assertEqual(report.effective_cpu, 24)
        self.assertEqual(report.estimated_seconds, 150.0)
        self.assertEqual(report.estimated_vram_mb, 3840.0)
        self.assertTrue(report.ok)
        self.assertEqual(enforce_quietly(report), 0)

    def test_the_warning_names_both_numbers(self):
        """A clamp that changes a docking parameter must be loud, never silent."""
        report = rc.validate_request(real_caps(), production_request())
        message = report.warnings[0].message
        self.assertIn("--cpu 60", message)
        self.assertIn("24 cores", message)
        self.assertIn("--cpu 24", message)
        self.assertIn("NUM_THREADS=24", report.warnings[0].remedy)

    def test_a_request_that_fits_produces_no_findings_at_all(self):
        report = rc.validate_request(real_caps(),
                                     production_request(cpu=24))
        self.assertEqual(report.findings, ())
        self.assertTrue(report.ok)

    def test_validate_request_is_pure(self):
        """No I/O, no clock, no environment: the same inputs must give the same
        report, or the predicate that decides whether a campaign starts is not
        testable."""
        caps, request = real_caps(), production_request()
        first = rc.validate_request(caps, request)
        second = rc.validate_request(caps, request)
        self.assertEqual(first.to_dict(), second.to_dict())


class TestExhaustivenessRule(unittest.TestCase):
    """'If exhaustiveness required for gnina is too much for remote device we
    shall warn and exit.' -- refused, never quietly reduced."""

    def test_excessive_exhaustiveness_is_fatal_and_exits_78(self):
        report = rc.validate_request(real_caps(),
                                     production_request(exhaustiveness=512))
        codes = [f.code for f in report.fatal]
        self.assertIn("EXHAUSTIVENESS_TOO_HIGH", codes)
        self.assertFalse(report.ok)
        self.assertEqual(enforce_quietly(report), rc.EXIT_CAPS_FATAL)
        self.assertEqual(rc.EXIT_CAPS_FATAL, 78)

    def test_the_message_carries_the_numbers_that_justify_it(self):
        report = rc.validate_request(real_caps(),
                                     production_request(exhaustiveness=512))
        finding = [f for f in report.fatal
                   if f.code == "EXHAUSTIVENESS_TOO_HIGH"][0]
        self.assertIn("exhaustiveness=512", finding.message)
        self.assertIn("2400 s", finding.message)
        self.assertIn("1800 s", finding.message)
        self.assertIn("<= 384", finding.remedy, "must say what would fit")

    def test_the_ceiling_it_recommends_actually_passes(self):
        """The remedy has to be true, not merely encouraging."""
        ceiling = rc.max_exhaustiveness_for(24, 1800.0)
        self.assertEqual(ceiling, 384)
        ok = rc.validate_request(real_caps(),
                                 production_request(exhaustiveness=ceiling))
        self.assertNotIn("EXHAUSTIVENESS_TOO_HIGH", [f.code for f in ok.fatal])
        too_much = rc.validate_request(real_caps(),
                                       production_request(exhaustiveness=ceiling + 1))
        self.assertIn("EXHAUSTIVENESS_TOO_HIGH", [f.code for f in too_much.fatal])

    def test_the_recommended_ceiling_is_never_a_hair_too_high(self):
        """A remedy that does not work is worse than no remedy, and the naive
        truncated division can land one over on the wrong deadline."""
        for cores in (1, 3, 7, 16, 24, 60):
            for deadline in (37.0, 100.0, 599.5, 1800.0, 7200.0):
                with self.subTest(cores=cores, deadline=deadline):
                    ceiling = rc.max_exhaustiveness_for(cores, deadline)
                    self.assertLessEqual(rc.estimate_seconds(ceiling, cores),
                                         deadline)
                    self.assertGreater(rc.estimate_seconds(ceiling + 1, cores),
                                       deadline, "and not needlessly low")

    def test_a_zero_or_negative_deadline_has_no_ceiling(self):
        self.assertEqual(rc.max_exhaustiveness_for(24, 0.0), 0)
        self.assertEqual(rc.max_exhaustiveness_for(0, 1800.0), 0)

    def test_exhaustiveness_is_never_clamped(self):
        """The refusal must not quietly become a reduction: the request that
        comes out is the request that went in."""
        report = rc.validate_request(real_caps(),
                                     production_request(exhaustiveness=512))
        self.assertEqual(report.request.exhaustiveness, 512)

    def test_more_than_half_the_deadline_warns(self):
        report = rc.validate_request(real_caps(),
                                     production_request(exhaustiveness=200))
        codes = [f.code for f in report.findings]
        self.assertIn("EXHAUSTIVENESS_SLOW", codes)
        self.assertTrue(report.ok, "slow is a warning, not a refusal")

    def test_a_longer_deadline_makes_room(self):
        report = rc.validate_request(
            real_caps(), production_request(exhaustiveness=512,
                                            job_deadline_seconds=3600.0))
        self.assertNotIn("EXHAUSTIVENESS_TOO_HIGH", [f.code for f in report.fatal])


class TestCpuRule(unittest.TestCase):

    def test_clamping_happens_and_is_reported(self):
        report = rc.validate_request(real_caps(), production_request(cpu=60))
        self.assertEqual(report.effective_cpu, 24)
        self.assertEqual([f.code for f in report.findings], ["CPU_OVERSUBSCRIBED"])

    def test_strict_upgrades_it_to_fatal(self):
        report = rc.validate_request(real_caps(), production_request(cpu=60),
                                     strict=True)
        self.assertIn("CPU_OVERSUBSCRIBED", [f.code for f in report.fatal])
        self.assertFalse(report.ok)
        self.assertEqual(enforce_quietly(report, strict=True), rc.EXIT_CAPS_FATAL)

    def test_enforce_strict_on_a_permissive_report_is_still_fatal(self):
        """Otherwise --strict-caps would depend on which call site built the
        report, which is exactly the kind of silent downgrade this module is
        supposed to prevent."""
        report = rc.validate_request(real_caps(), production_request(cpu=60),
                                     strict=False)
        self.assertTrue(report.ok)
        self.assertEqual(enforce_quietly(report, strict=False), 0)
        self.assertEqual(enforce_quietly(report, strict=True), rc.EXIT_CAPS_FATAL)

    def test_fewer_threads_than_cores_is_not_a_finding(self):
        report = rc.validate_request(real_caps(), production_request(cpu=8))
        self.assertEqual(report.effective_cpu, 8)
        self.assertEqual(report.findings, ())

    def test_an_unknown_core_count_warns_instead_of_clamping_to_zero(self):
        """nproc missing must not divide by zero or clamp --cpu to 0. The
        request passes through unclamped and the operator is told so."""
        report = rc.validate_request(real_caps(cores=0), production_request())
        self.assertEqual(report.effective_cpu, 60)
        self.assertIn("CORES_UNKNOWN", [f.code for f in report.findings])

    def test_an_unknown_core_count_alone_is_not_a_refusal(self):
        report = rc.validate_request(real_caps(cores=0),
                                     production_request(cpu=24))
        self.assertEqual([f.code for f in report.findings], ["CORES_UNKNOWN"])
        self.assertTrue(report.ok)

    def test_unclamped_60_threads_do_not_fit_the_card(self):
        """The clamp is what keeps this campaign inside 8 GB of VRAM: without a
        core count to clamp against, --cpu 60 needs 7800 MB and is refused."""
        report = rc.validate_request(real_caps(cores=0), production_request())
        self.assertEqual(report.estimated_vram_mb, 7800.0)
        self.assertIn("VRAM_INSUFFICIENT", [f.code for f in report.fatal])

    def test_a_nonsense_cpu_request_cannot_divide_by_zero(self):
        report = rc.validate_request(real_caps(cores=0),
                                     production_request(cpu=0))
        self.assertEqual(report.effective_cpu, 1)
        self.assertGreater(report.estimated_seconds, 0)

    def test_effective_cpu_for_is_the_only_definition_of_the_clamp(self):
        """The CAPS_CHANGED check needs the value a previous caps.json implied.
        If a caller re-derives min(cpu, cores) by hand, the rule that freezes a
        campaign's thread count has two definitions and they will drift."""
        caps, request = real_caps(), production_request()
        self.assertEqual(rc.effective_cpu_for(caps, request),
                         rc.validate_request(caps, request).effective_cpu)
        self.assertEqual(rc.effective_cpu_for(real_caps(cores=0), request), 60)
        self.assertEqual(rc.effective_cpu_for(caps, production_request(cpu=8)), 8)


class TestGpuRules(unittest.TestCase):

    def test_no_gpu_is_fatal(self):
        report = rc.validate_request(real_caps(gpu_present=False, gpu_name="",
                                               vram_total_mb=0),
                                     production_request())
        codes = [f.code for f in report.fatal]
        self.assertIn("NO_GPU", codes)
        self.assertFalse(report.ok)
        self.assertEqual(enforce_quietly(report), rc.EXIT_CAPS_FATAL)

    def test_no_gpu_does_not_also_emit_a_confusing_vram_fatal(self):
        """'over the 0 MB usable of the 0 MB on the remote GPU' would bury the
        finding the operator actually has to act on."""
        report = rc.validate_request(real_caps(gpu_present=False, gpu_name="",
                                               vram_total_mb=0, vram_used_mb=0),
                                     production_request())
        self.assertEqual([f.code for f in report.fatal], ["NO_GPU"])

    def test_the_no_gpu_message_explains_why_there_is_no_fallback(self):
        report = rc.validate_request(real_caps(gpu_present=False, gpu_name=""),
                                     production_request())
        finding = [f for f in report.fatal if f.code == "NO_GPU"][0]
        self.assertIn(HOST, finding.message)
        self.assertIn("CNN", finding.message)
        self.assertTrue(finding.remedy)

    # The VRAM rule can no longer fire on the real 8 GB card: the measured
    # requirement is ~550 MiB flat, so 800 MB never approaches the 6659 MB usable.
    # These tests therefore exercise the RULE against a deliberately small card.
    # Before the recalibration they passed on the real card only because the model
    # over-predicted by ~30x -- they were testing the bug, not the rule.
    SMALL_CARD_MB = 1024

    def test_vram_insufficient_is_fatal(self):
        # 1200 + 110*60 = 7800 MB against 0.85*8192 - 1 = 6962 MB usable.
        report = rc.validate_request(real_caps(cores=64), production_request())
        self.assertEqual(report.effective_cpu, 60)
        self.assertIn("VRAM_INSUFFICIENT", [f.code for f in report.fatal])

    def test_vram_tight_is_only_a_warning(self):
        # 1200 + 110*48 = 6480 MB: 93% of usable -- tight, but it fits.
        report = rc.validate_request(real_caps(cores=64),
                                     production_request(cpu=48))
        self.assertIn("VRAM_TIGHT", [f.code for f in report.findings])
        self.assertTrue(report.ok)

    def test_the_documented_vram_threshold_is_where_the_docstring_says(self):
        """The module docstring claims --cpu 53 is where this card runs out.
        A documented number nobody checks is a number that drifts."""
        card = real_caps(cores=64, vram_used_mb=0)      # 0.85 * 8192 = 6963.2
        for cpu, expected in ((52, "VRAM_TIGHT"), (53, "VRAM_INSUFFICIENT")):
            with self.subTest(cpu=cpu):
                codes = [f.code for f in rc.validate_request(
                    card, production_request(cpu=cpu)).findings]
                self.assertIn(expected, codes)

    def test_the_production_thread_count_is_comfortable(self):
        report = rc.validate_request(real_caps(), production_request())
        self.assertNotIn("VRAM_TIGHT", [f.code for f in report.findings])
        self.assertNotIn("VRAM_INSUFFICIENT", [f.code for f in report.findings])

    def test_the_vram_model_matches_the_documented_constants(self):
        self.assertEqual(rc.estimate_vram_mb(24), 3840.0)
        self.assertAlmostEqual(rc.RemoteCaps(host="h", vram_total_mb=8192,
                                             vram_used_mb=0).usable_vram_mb,
                               6963.2, places=1)


class TestRuntimeRules(unittest.TestCase):

    def test_no_docker_is_fatal(self):
        report = rc.validate_request(real_caps(docker_available=False),
                                     production_request())
        self.assertIn("NO_GNINA", [f.code for f in report.fatal])

    def test_a_missing_image_is_fatal_and_says_which_one(self):
        report = rc.validate_request(real_caps(image_present=False),
                                     production_request())
        finding = [f for f in report.fatal if f.code == "NO_GNINA"][0]
        self.assertIn(rc.DEFAULT_IMAGE, finding.message)

    def test_a_missing_native_binary_is_fatal(self):
        report = rc.validate_request(
            real_caps(native_present=False),
            production_request(native="/opt/gnina"))
        self.assertIn("NO_GNINA", [f.code for f in report.fatal])

    def test_native_mode_does_not_care_about_docker(self):
        report = rc.validate_request(
            real_caps(native_present=True, docker_available=False,
                      image_present=False),
            production_request(native="/opt/gnina"))
        self.assertNotIn("NO_GNINA", [f.code for f in report.fatal])

    def test_async_without_the_shim_is_fatal(self):
        """The queue drives gnina through gnina_remote's submit/collect modes;
        a real gnina binary would block the pump and never write a ticket."""
        report = rc.validate_request(
            real_caps(), production_request(gnina_path="/home/onur/software/gnina.1.3.2"))
        self.assertIn("ASYNC_NEEDS_SHIM", [f.code for f in report.fatal])

    def test_sync_mode_may_point_at_a_real_binary(self):
        report = rc.validate_request(
            real_caps(),
            production_request(gnina_path="/home/onur/software/gnina.1.3.2",
                               async_mode=False))
        self.assertNotIn("ASYNC_NEEDS_SHIM", [f.code for f in report.fatal])

    def test_ram_tight_warns(self):
        report = rc.validate_request(real_caps(mem_total_mb=2048),
                                     production_request())
        self.assertIn("RAM_TIGHT", [f.code for f in report.findings])
        self.assertTrue(report.ok)


class TestCapsChanged(unittest.TestCase):
    """Uniform parameters across a campaign beat squeezing one extra thread."""

    def test_a_mid_campaign_change_is_fatal(self):
        report = rc.validate_request(real_caps(), production_request(),
                                     previous_effective_cpu=16)
        self.assertIn("CAPS_CHANGED", [f.code for f in report.fatal])
        finding = [f for f in report.fatal if f.code == "CAPS_CHANGED"][0]
        self.assertIn("24", finding.message)
        self.assertIn("16", finding.message)

    def test_the_same_resolution_is_not_a_finding(self):
        report = rc.validate_request(real_caps(), production_request(),
                                     previous_effective_cpu=24)
        self.assertNotIn("CAPS_CHANGED", [f.code for f in report.findings])

    def test_no_previous_value_means_no_check(self):
        report = rc.validate_request(real_caps(), production_request(),
                                     previous_effective_cpu=None)
        self.assertNotIn("CAPS_CHANGED", [f.code for f in report.findings])


class TestReportShape(unittest.TestCase):

    def test_fatal_findings_sort_first(self):
        report = rc.validate_request(
            real_caps(mem_total_mb=1024),
            production_request(exhaustiveness=512))
        severities = [f.severity for f in report.findings]
        self.assertEqual(severities, sorted(severities,
                                            key=lambda s: rc.SEVERITIES.index(s),
                                            reverse=True))
        self.assertEqual(report.findings[0].severity, "FATAL")

    def test_the_formatted_report_shows_the_verdict_and_the_codes(self):
        report = rc.validate_request(real_caps(),
                                     production_request(exhaustiveness=512))
        text = report.format_report()
        self.assertIn("EXHAUSTIVENESS_TOO_HIGH", text)
        self.assertIn("REFUSED", text)
        self.assertIn("Remedy:", text)
        self.assertIn(HOST, text)

    def test_an_ok_report_says_so(self):
        text = rc.validate_request(real_caps(), production_request()).format_report()
        self.assertIn("verdict: OK", text)
        self.assertIn("CPU_OVERSUBSCRIBED", text)

    def test_no_report_line_is_absurdly_long(self):
        report = rc.validate_request(real_caps(mem_total_mb=1024),
                                     production_request(exhaustiveness=512))
        for line in report.format_report().splitlines():
            self.assertLessEqual(len(line), 120, line)

    def test_to_dict_is_json_serialisable(self):
        report = rc.validate_request(real_caps(), production_request())
        text = json.dumps(report.to_dict(), sort_keys=True)
        again = json.loads(text)
        self.assertEqual(again["effective_cpu"], 24)
        self.assertTrue(again["ok"])
        self.assertEqual(again["findings"][0]["code"], "CPU_OVERSUBSCRIBED")


# --------------------------------------------------------------------------
# the probe over the injected seam
# --------------------------------------------------------------------------

class TestProbeRemote(unittest.TestCase):

    def test_a_successful_probe_parses(self):
        runner = FakeRunner(stdout=REAL_PROBE_OUTPUT)
        caps = rc.probe_remote(HOST, port=PORT, ssh_runner=runner)
        self.assertEqual(caps.cores, 24)
        self.assertEqual(caps.source, "probe")
        self.assertEqual(len(runner.calls), 1, "one round trip, not nine")

    def test_the_command_goes_to_the_right_host_and_port(self):
        runner = FakeRunner(stdout=REAL_PROBE_OUTPUT)
        rc.probe_remote(HOST, port=PORT, key="/k/id_rsa", ssh_runner=runner)
        cmd = runner.calls[0]["cmd"]
        self.assertIn(HOST, cmd)
        self.assertIn("-p", cmd)
        self.assertIn(str(PORT), cmd)
        self.assertIn("/k/id_rsa", cmd)
        self.assertIn("nvidia-smi", cmd[-1])

    def test_an_unreachable_host_is_an_error_not_an_optimistic_guess(self):
        """The whole point: a guessed capability record would authorise a
        campaign that then fails on every single job."""
        import gnina_remote

        runner = FakeRunner(raises=gnina_remote.TransientError(
            "ssh: connect to host x port 50000: Connection refused"))
        with self.assertRaises(rc.RemoteCapsError) as ctx:
            rc.probe_remote(HOST, port=PORT, ssh_runner=runner)
        message = str(ctx.exception)
        self.assertIn(HOST, message)
        self.assertIn("Connection refused", message)
        self.assertIn("network-level", message)

    def test_a_permanent_ssh_failure_is_also_an_error(self):
        import gnina_remote

        runner = FakeRunner(raises=gnina_remote.RemoteGninaError(
            "Permission denied (publickey)"))
        with self.assertRaises(rc.RemoteCapsError) as ctx:
            rc.probe_remote(HOST, ssh_runner=runner)
        self.assertIn("Permission denied", str(ctx.exception))

    def test_a_nonzero_exit_is_an_error(self):
        runner = FakeRunner(returncode=127, stderr="sh: nproc: not found")
        with self.assertRaises(rc.RemoteCapsError) as ctx:
            rc.probe_remote(HOST, ssh_runner=runner)
        self.assertIn("127", str(ctx.exception))

    def test_unrecognisable_output_is_an_error_not_a_no_gpu_diagnosis(self):
        """A restricted shell that eats the command must not be reported as a
        missing GPU -- that sends the operator to fix the wrong thing."""
        runner = FakeRunner(stdout="-bash: line 1: syntax error\n")
        with self.assertRaises(rc.RemoteCapsError) as ctx:
            rc.probe_remote(HOST, ssh_runner=runner)
        self.assertIn("no recognisable output", str(ctx.exception))

    def test_a_gpu_less_host_probes_fine_and_is_judged_later(self):
        """Parsing stays forgiving; refusing is validate_request's job."""
        runner = FakeRunner(stdout=REAL_PROBE_OUTPUT.replace(
            "gpu=NVIDIA GeForce RTX 3060 Ti, 8192, 1, 8.6", "gpu="))
        caps = rc.probe_remote(HOST, ssh_runner=runner)
        self.assertFalse(caps.gpu_present)
        self.assertFalse(rc.validate_request(caps, production_request()).ok)

    def test_no_host_is_refused_before_any_ssh(self):
        runner = FakeRunner(stdout=REAL_PROBE_OUTPUT)
        with self.assertRaises(rc.RemoteCapsError):
            rc.probe_remote("", ssh_runner=runner)
        self.assertEqual(runner.calls, [])


# --------------------------------------------------------------------------
# the cache -- one probe per campaign, not one per job
# --------------------------------------------------------------------------

class TestCache(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "gnina_queue", "caps.json")

    def test_a_fresh_cache_prevents_a_second_probe(self):
        """A thousand jobs must not become a thousand probes."""
        runner = FakeRunner(stdout=REAL_PROBE_OUTPUT)
        first = rc.get_caps(HOST, port=PORT, cache_path=self.path,
                            ssh_runner=runner, now=1000.0)
        self.assertEqual(len(runner.calls), 1)

        for _ in range(50):
            again = rc.get_caps(HOST, port=PORT, cache_path=self.path,
                                ssh_runner=runner, now=1000.0)
        self.assertEqual(len(runner.calls), 1, "the cache did not hold")
        self.assertEqual(again.cores, first.cores)
        self.assertEqual(again.source, "cache")

    def test_the_cache_expires(self):
        runner = FakeRunner(stdout=REAL_PROBE_OUTPUT)
        rc.get_caps(HOST, port=PORT, cache_path=self.path, ssh_runner=runner,
                    ttl_seconds=100.0, now=1000.0)
        rc.get_caps(HOST, port=PORT, cache_path=self.path, ssh_runner=runner,
                    ttl_seconds=100.0, now=1099.0)
        self.assertEqual(len(runner.calls), 1, "still fresh at 99 s of a 100 s ttl")
        rc.get_caps(HOST, port=PORT, cache_path=self.path, ssh_runner=runner,
                    ttl_seconds=100.0, now=1101.0)
        self.assertEqual(len(runner.calls), 2, "stale at 101 s of a 100 s ttl")

    def test_an_injected_clock_stamps_the_record_it_writes(self):
        """Otherwise the ageing comparison mixes an injected clock with a real
        one and the cache looks like it was taken in 2026 or in 1970."""
        runner = FakeRunner(stdout=REAL_PROBE_OUTPUT)
        caps = rc.get_caps(HOST, cache_path=self.path, ssh_runner=runner,
                           now=1000.0)
        self.assertAlmostEqual(rc._parse_iso_utc(caps.probed_at), 1000.0, places=3)

    def test_force_ignores_a_fresh_cache(self):
        runner = FakeRunner(stdout=REAL_PROBE_OUTPUT)
        rc.get_caps(HOST, cache_path=self.path, ssh_runner=runner)
        rc.get_caps(HOST, cache_path=self.path, ssh_runner=runner, force=True)
        self.assertEqual(len(runner.calls), 2)

    def test_a_record_for_another_host_is_not_reused(self):
        """A different machine's core count says nothing about this one."""
        runner = FakeRunner(stdout=REAL_PROBE_OUTPUT)
        rc.get_caps("someone@elsewhere", cache_path=self.path, ssh_runner=runner)
        rc.get_caps(HOST, cache_path=self.path, ssh_runner=runner)
        self.assertEqual(len(runner.calls), 2)

    def test_a_record_for_another_port_is_not_reused(self):
        runner = FakeRunner(stdout=REAL_PROBE_OUTPUT)
        rc.get_caps(HOST, port=22, cache_path=self.path, ssh_runner=runner)
        rc.get_caps(HOST, port=50000, cache_path=self.path, ssh_runner=runner)
        self.assertEqual(len(runner.calls), 2)

    def test_a_corrupt_cache_is_a_miss_not_a_crash(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as handle:
            handle.write("{ this is not json")
        runner = FakeRunner(stdout=REAL_PROBE_OUTPUT)
        caps = rc.get_caps(HOST, cache_path=self.path, ssh_runner=runner)
        self.assertEqual(caps.cores, 24)

    def test_a_truncated_cache_is_a_miss_not_a_crash(self):
        """The queue root can be on a fuseblk mount; a torn write is plausible."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        open(self.path, "w").close()
        runner = FakeRunner(stdout=REAL_PROBE_OUTPUT)
        self.assertEqual(rc.get_caps(HOST, cache_path=self.path,
                                     ssh_runner=runner).cores, 24)

    def test_probe_settings_reach_the_probe(self):
        """image/native/remote_root are what the probe asks about; a get_caps
        that dropped them would report on the wrong image."""
        runner = FakeRunner(stdout=REAL_PROBE_OUTPUT)
        rc.get_caps(HOST, ssh_runner=runner, image="other/img:9",
                    native="/opt/gnina", remote_root="/scratch/gnina")
        command = runner.calls[0]["cmd"][-1]
        self.assertIn("other/img:9", command)
        self.assertIn("/opt/gnina", command)
        self.assertIn("/scratch/gnina", command)

    def test_without_a_cache_path_every_call_probes(self):
        runner = FakeRunner(stdout=REAL_PROBE_OUTPUT)
        rc.get_caps(HOST, ssh_runner=runner)
        rc.get_caps(HOST, ssh_runner=runner)
        self.assertEqual(len(runner.calls), 2)

    def test_an_unreachable_host_is_not_papered_over_by_the_cache(self):
        import gnina_remote

        runner = FakeRunner(raises=gnina_remote.TransientError("Connection refused"))
        with self.assertRaises(rc.RemoteCapsError):
            rc.get_caps(HOST, cache_path=self.path, ssh_runner=runner)
        self.assertFalse(os.path.exists(self.path))

    def test_the_write_is_atomic_and_leaves_no_temp_files(self):
        runner = FakeRunner(stdout=REAL_PROBE_OUTPUT)
        rc.get_caps(HOST, cache_path=self.path, ssh_runner=runner)
        leftovers = [n for n in os.listdir(os.path.dirname(self.path))
                     if ".tmp." in n]
        self.assertEqual(leftovers, [])

    def test_a_cache_that_cannot_be_written_does_not_fail_the_run(self):
        """The cache is an optimisation. The freeze that matters is params.json."""
        runner = FakeRunner(stdout=REAL_PROBE_OUTPUT)
        blocked = os.path.join(self.tmp.name, "not_a_dir")
        open(blocked, "w").close()
        caps = rc.get_caps(HOST, cache_path=os.path.join(blocked, "caps.json"),
                           ssh_runner=runner)
        self.assertEqual(caps.cores, 24)

    def test_a_timestamp_from_the_future_is_treated_as_stale(self):
        runner = FakeRunner(stdout=REAL_PROBE_OUTPUT)
        rc.save_caps(real_caps(probed_at="2099-01-01T00:00:00.000000Z"), self.path)
        rc.get_caps(HOST, port=PORT, cache_path=self.path, ssh_runner=runner)
        self.assertEqual(len(runner.calls), 1)

    def test_load_caps_marks_the_source_and_never_probes(self):
        rc.save_caps(real_caps(), self.path)
        caps = rc.load_caps(self.path)
        self.assertEqual(caps.source, "file")
        self.assertEqual(caps.cores, 24)

    def test_load_caps_is_loud_about_a_missing_file(self):
        """--caps-file is an explicit instruction; silently probing instead
        would contact a host the operator told us to leave alone."""
        with self.assertRaises(rc.RemoteCapsError):
            rc.load_caps(os.path.join(self.tmp.name, "nope.json"))

    def test_cache_path_for_is_the_documented_location(self):
        self.assertEqual(rc.cache_path_for("/run/gnina_queue"),
                         "/run/gnina_queue/caps.json")


# --------------------------------------------------------------------------
# enforce + CLI
# --------------------------------------------------------------------------

class _CapturingLogger(object):
    def __init__(self):
        self.critical_calls = []
        self.warning_calls = []
        self.info_calls = []

    def critical(self, msg, *args):
        self.critical_calls.append(msg % args if args else msg)

    def warning(self, msg, *args):
        self.warning_calls.append(msg % args if args else msg)

    def info(self, msg, *args):
        self.info_calls.append(msg % args if args else msg)


class TestEnforce(unittest.TestCase):

    def setUp(self):
        self._stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        self.addCleanup(self._restore)

    def _restore(self):
        sys.stderr.close()
        sys.stderr = self._stderr

    def test_fatal_is_logged_critical_and_returns_78(self):
        logger = _CapturingLogger()
        report = rc.validate_request(real_caps(),
                                     production_request(exhaustiveness=512))
        self.assertEqual(rc.enforce(report, logger=logger), 78)
        joined = "\n".join(logger.critical_calls)
        self.assertIn("EXHAUSTIVENESS_TOO_HIGH", joined)
        self.assertIn("2400 s", joined)

    def test_warn_is_logged_warning_and_returns_zero(self):
        logger = _CapturingLogger()
        report = rc.validate_request(real_caps(), production_request())
        self.assertEqual(rc.enforce(report, logger=logger), 0)
        self.assertEqual(logger.critical_calls, [])
        self.assertIn("CPU_OVERSUBSCRIBED", "\n".join(logger.warning_calls))

    def test_the_measured_numbers_reach_the_log_not_only_stderr(self):
        """'Why is this campaign running at --cpu 24' has to be answerable from
        docking_automation.log months later."""
        logger = _CapturingLogger()
        report = rc.validate_request(real_caps(), production_request())
        rc.enforce(report, logger=logger)
        logged = "\n".join(logger.info_calls)
        self.assertIn("NVIDIA GeForce RTX 3060 Ti", logged)
        self.assertIn("effective --cpu 24", logged)
        self.assertIn("~150 s per job", logged)

    def test_enforce_does_not_exit_the_process(self):
        """batch_dock owns process exit; a library that calls sys.exit cannot
        be tested and cannot be embedded."""
        report = rc.validate_request(real_caps(),
                                     production_request(exhaustiveness=512))
        self.assertEqual(rc.enforce(report), 78)   # would raise SystemExit


class TestCli(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.caps_file = os.path.join(self.tmp.name, "caps.json")
        rc.save_caps(real_caps(), self.caps_file)
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = open(os.path.join(self.tmp.name, "out"), "w")
        sys.stderr = open(os.path.join(self.tmp.name, "err"), "w")
        self.addCleanup(self._restore)

    def _restore(self):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def _read(self, name):
        sys.stdout.flush()
        sys.stderr.flush()
        with open(os.path.join(self.tmp.name, name)) as handle:
            return handle.read()

    def test_caps_file_never_touches_the_network(self):
        """--caps-file is how the tests and the other implementers avoid the
        GPU box entirely."""
        def explode(*_a, **_k):
            self.fail("the CLI probed the network with --caps-file")

        saved = rc.probe_remote
        rc.probe_remote = explode
        try:
            code = rc.main(["--caps-file", self.caps_file, "--cpu", "60",
                            "--exhaustiveness", "32"])
        finally:
            rc.probe_remote = saved
        self.assertEqual(code, 0)
        self.assertIn("CPU_OVERSUBSCRIBED", self._read("err"))

    def test_an_impossible_request_exits_78(self):
        code = rc.main(["--caps-file", self.caps_file, "--exhaustiveness", "512"])
        self.assertEqual(code, rc.EXIT_CAPS_FATAL)

    def test_json_output_is_machine_readable(self):
        code = rc.main(["--caps-file", self.caps_file, "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(self._read("out"))
        self.assertEqual(payload["effective_cpu"], 24)
        self.assertEqual(payload["caps"]["gpu_name"], "NVIDIA GeForce RTX 3060 Ti")

    def test_strict_turns_the_clamp_into_a_refusal(self):
        code = rc.main(["--caps-file", self.caps_file, "--cpu", "60", "--strict"])
        self.assertEqual(code, rc.EXIT_CAPS_FATAL)

    def test_a_missing_host_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as ctx:
            rc.main(["--cpu", "60"])
        self.assertEqual(ctx.exception.code, 2)

    def test_an_unreadable_caps_file_exits_78_not_a_traceback(self):
        code = rc.main(["--caps-file", os.path.join(self.tmp.name, "absent.json")])
        self.assertEqual(code, rc.EXIT_CAPS_FATAL)
        self.assertIn("PROBE_FAILED", self._read("err"))


class TestSurvivesWithoutTheShim(unittest.TestCase):
    """The pure half must stay importable when gnina_remote.py is not.

    Only the SSH paths need the shim. If a broken or absent gnina_remote.py
    could stop this module from importing, a mid-edit syntax error in a file
    another author owns would take the capability gate -- and every test in
    this file -- down with it.
    """

    def _import_in_isolation(self, snippet):
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            package = os.path.join(tmp, "dock_selectivity")
            os.makedirs(package)
            open(os.path.join(package, "__init__.py"), "w").close()
            shutil.copy(os.path.join(REPO, "dock_selectivity", "remote_caps.py"),
                        package)
            # No gnina_remote.py anywhere this copy can reach.
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            return subprocess.run([sys.executable, "-c", snippet], cwd=tmp,
                                  env=env, capture_output=True, text=True,
                                  timeout=120)

    def test_it_imports_and_validates_without_gnina_remote(self):
        out = self._import_in_isolation(
            "from dock_selectivity import remote_caps as rc;"
            "assert rc._GNINA_REMOTE is None, 'the shim should be unreachable';"
            "caps = rc.parse_probe_output('cores=24\\n"
            "gpu=NVIDIA GeForce RTX 3060 Ti, 8192, 1, 8.6\\n"
            "docker=1\\nimage=1\\ndisk_free_kb=999999999\\n', 'u@h');"
            "r = rc.validate_request(caps, rc.DockRequest(cpu=24, exhaustiveness=32));"
            "print('OK', r.ok, rc.DEFAULT_IMAGE)")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("OK True gnina/gnina:v1.3.3", out.stdout)

    def test_probing_without_the_shim_is_a_clear_error_not_a_guess(self):
        out = self._import_in_isolation(
            "from dock_selectivity import remote_caps as rc\n"
            "try:\n"
            "    rc.probe_remote('u@h')\n"
            "except rc.RemoteCapsError as exc:\n"
            "    print('RAISED', 'gnina_remote.py' in str(exc))\n")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("RAISED True", out.stdout)


class TestConstantsAgreeWithTheShim(unittest.TestCase):
    """Two definitions of the image tag would drift silently."""

    def test_default_image_matches_gnina_remote(self):
        import gnina_remote

        self.assertEqual(rc.DEFAULT_IMAGE, gnina_remote.DEFAULT_IMAGE)
        self.assertEqual(rc.DEFAULT_REMOTE_ROOT, gnina_remote.DEFAULT_REMOTE_DIR)

    def test_the_time_model_reproduces_the_measurement(self):
        """~150 s per job at exhaustiveness 32 on 24 cores, measured."""
        self.assertAlmostEqual(rc.estimate_seconds(32, 24), 150.0, places=6)

    def _model_in_a_fresh_interpreter(self, env):
        """Import remote_caps in a subprocess and report its model constants.

        A subprocess rather than importlib.reload(): the constants are resolved
        at import, and reloading in-process would hand every other test module
        in this shared suite a *new* RemoteCaps class object while they still
        hold the old one. Isolation is worth 100 ms.
        """
        child = dict(os.environ)
        child.update(env)
        child["PYTHONPATH"] = REPO
        code = ("from dock_selectivity import remote_caps as rc;"
                "print(rc.CORE_SECONDS_PER_EXH_UNIT, rc.VRAM_BASE_MB,"
                " rc.VRAM_PER_THREAD_MB, rc.estimate_seconds(32, 24),"
                " rc.estimate_vram_mb(24))")
        out = subprocess.run([sys.executable, "-c", code], env=child,
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        return [float(x) for x in out.stdout.split()]

    def test_the_model_constants_are_env_overridable(self):
        """Another card means another calibration. Resolved once at import so
        validate_request stays pure -- see the module docstring."""
        values = self._model_in_a_fresh_interpreter({
            "GNINA_CORE_SECONDS_PER_EXH": "56.25",
            "GNINA_VRAM_BASE_MB": "600",
            "GNINA_VRAM_PER_THREAD_MB": "55"})
        self.assertEqual(values, [56.25, 600.0, 55.0, 75.0, 1920.0])

    # NOTE: these pin the model's arithmetic, not the hardware's behaviour. The
    # VRAM constants were measured on 2026-08-13 to over-predict by ~30x; see the
    # warning above VRAM_BASE_MB in remote_caps.py. Changing the defaults means
    # changing these two assertions with them.
    def test_the_defaults_are_the_measured_calibration(self):
        values = self._model_in_a_fresh_interpreter({})
        self.assertEqual(values, [112.5, 1200.0, 110.0, 150.0, 3840.0])

    def test_a_malformed_override_is_ignored_not_obeyed(self):
        """A mistyped calibration must not silently become 0 s per job -- that
        would make every exhaustiveness look affordable."""
        for bad in ("not-a-number", "0", "-5"):
            with self.subTest(bad=bad):
                values = self._model_in_a_fresh_interpreter(
                    {"GNINA_CORE_SECONDS_PER_EXH": bad})
                self.assertEqual(values[0], 112.5)

    def test_severities_cover_every_severity_emitted(self):
        reports = [
            rc.validate_request(real_caps(), production_request()),
            rc.validate_request(real_caps(gpu_present=False), production_request()),
            rc.validate_request(real_caps(cores=0, mem_total_mb=512),
                                production_request(exhaustiveness=512)),
        ]
        for report in reports:
            for finding in report.findings:
                self.assertIn(finding.severity, rc.SEVERITIES, finding.code)


if __name__ == "__main__":
    unittest.main()
