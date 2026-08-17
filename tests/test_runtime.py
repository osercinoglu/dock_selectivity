"""Tests for dock_selectivity.runtime -- the shared step runtime.

Every assertion here corresponds to a failure the repo actually shipped, or to a
promise six pipeline scripts are about to depend on:

* import-time logging that wrote a file into whatever directory the process
  started in, and stacked one stdout handler per imported module;
* ``--log-level`` silently ignored because ``basicConfig`` is a no-op once the
  root logger has handlers;
* ``batch_dock.configure_logging`` closing handlers it did not own;
* and the new promise: a notification failure must never end a three-day run,
  and the Pushover token must never reach a log record.

Hermetic: no network (``urllib.request.urlopen`` is always patched), no external
drive, no ``/opt/data``. Log files are written into ``tempfile.TemporaryDirectory``.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import socket
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dock_selectivity import runtime  # noqa: E402
from dock_selectivity.runtime import (  # noqa: E402
    NOTIFY_EVENTS,
    NotifyConfigurationError,
    Pushover,
    RuntimeArgumentConflict,
    add_runtime_arguments,
    configure_logging,
    log_resolved_configuration,
    resolve_notify_events,
    step,
    step_out_dir,
    workflow_log_path,
)

TOKEN = "aBcDeFgHiJkLmNoPqRsTuVwXyZ0123"
USER_KEY = "uSeR9876543210zyxwvutsrqponml"


def parser_for(step_name="unit_step", **kwargs):
    parser = argparse.ArgumentParser(prog=step_name + ".py")
    return add_runtime_arguments(parser, step_name, **kwargs)


def args_for(step_name, out_dir, extra=()):
    """Parse a realistic argv for ``step_name`` into a namespace."""
    argv = ["--out-dir", out_dir] + list(extra)
    return parser_for(step_name).parse_args(argv)


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


class LoggingIsolation(unittest.TestCase):
    """Base class: restore the logging tree after every test.

    Without this, one test's handlers leak into the next -- which is the exact
    class of bug this module exists to prevent, so the tests must not commit it.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        root = logging.getLogger()
        self._root_handlers = list(root.handlers)
        self._root_level = root.level
        self._touched = {}
        self.addCleanup(self._restore_logging)
        self.track("dock_selectivity")
        # Seven prep-era scripts still configure the ROOT logger at import, so by
        # the time this file runs under `discover` the root has half a dozen
        # stacked stdout handlers. Detach them for the duration (restored in
        # cleanup) so these tests observe only the handlers under test.
        root.handlers[:] = []
        # Console handlers bind sys.stderr at construction; silence it so a test
        # run does not print the fixtures' log records. Tests that assert on the
        # console patch sys.stderr again, before they configure logging.
        self.console = io.StringIO()
        silencer = mock.patch.object(sys, "stderr", self.console)
        silencer.start()
        self.addCleanup(silencer.stop)

    def track(self, *names):
        """Restore this logger's level and handlers exactly, after the test.

        Exactly, not to NOTSET: other test modules in the suite deliberately
        quieten loggers such as ``dock_selectivity.schemas`` for the rest of the
        process, and this file must not undo that.
        """
        for name in names:
            if name not in self._touched:
                target = logging.getLogger(name)
                self._touched[name] = (target.level, list(target.handlers))

    def _restore_logging(self):
        self.track("dock_selectivity")
        for name, (level, handlers) in self._touched.items():
            target = logging.getLogger(name)
            for handler in list(target.handlers):
                if handler in handlers:
                    continue
                target.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass
            target.handlers[:] = handlers
            target.setLevel(level)
        root = logging.getLogger()
        root.handlers[:] = self._root_handlers
        root.setLevel(self._root_level)


# --- add_runtime_arguments ----------------------------------------------------


class TestRuntimeArguments(unittest.TestCase):
    def test_the_exact_flag_set_is_added_with_the_documented_defaults(self):
        args = parser_for("cavities").parse_args([])
        self.assertEqual(args.out_dir, os.path.join(".", "cavities"))
        self.assertIsNone(args.log_file)
        self.assertEqual(args.log_level, "INFO")
        self.assertFalse(args.quiet)
        self.assertFalse(args.notify)
        self.assertIsNone(args.notify_on)
        self.assertIsNone(args.notify_title)

    def test_default_out_dir_overrides_the_step_named_default(self):
        args = parser_for("dock", default_out_dir="/runs/r1/steps/dock").parse_args([])
        self.assertEqual(args.out_dir, "/runs/r1/steps/dock")

    def test_notify_on_is_repeatable_and_validated(self):
        args = parser_for().parse_args(["--notify-on", "start",
                                        "--notify-on", "success"])
        self.assertEqual(args.notify_on, ["start", "success"])
        with self.assertRaises(SystemExit):
            silence = io.StringIO()
            with mock.patch.object(sys, "stderr", silence):
                parser_for().parse_args(["--notify-on", "carrier-pigeon"])

    def test_every_added_flag_carries_help_text(self):
        parser = parser_for("helped")
        actions = {a.dest: a for a in parser._actions if a.dest != "help"}
        for dest in ("out_dir", "log_file", "log_level", "quiet", "notify",
                     "notify_on", "notify_title"):
            self.assertIn(dest, actions)
            self.assertTrue(actions[dest].help,
                            "%s has no help= and the repo lint forbids that" % dest)

    def test_a_parser_that_already_owns_a_runtime_flag_is_an_explicit_error(self):
        parser = argparse.ArgumentParser(prog="legacy.py")
        parser.add_argument("--log-level", default="INFO", help="pre-existing flag")
        with self.assertRaises(RuntimeArgumentConflict) as ctx:
            add_runtime_arguments(parser, "legacy")
        self.assertIn("--log-level", str(ctx.exception))

    def test_resolve_notify_events(self):
        self.assertEqual(resolve_notify_events(None), ("failure",))
        self.assertEqual(resolve_notify_events([]), ("failure",))
        self.assertEqual(resolve_notify_events(["success", "start"]),
                         ("start", "success"))
        self.assertEqual(resolve_notify_events(["all"]),
                         ("failure", "start", "success"))
        self.assertEqual(resolve_notify_events("failure"), ("failure",))
        with self.assertRaises(ValueError):
            resolve_notify_events(["nope"])
        self.assertEqual(set(NOTIFY_EVENTS),
                         {"start", "success", "failure", "all"})


class TestResolveOutDir(unittest.TestCase):
    """--out-dir is settled after parsing, because --run-dir is only known then."""

    def test_an_untouched_out_dir_moves_under_the_run_directory(self):
        args = parser_for("plan_positives").parse_args([])
        resolved = runtime.resolve_out_dir(args, "plan_positives", "/runs/r1")
        expected = os.path.abspath("/runs/r1/steps/plan_positives")
        self.assertEqual(resolved, expected)
        self.assertEqual(args.out_dir, expected)

    def test_an_operator_supplied_out_dir_is_kept(self):
        args = parser_for("plan_positives").parse_args(["--out-dir", "/scratch/here"])
        resolved = runtime.resolve_out_dir(args, "plan_positives", "/runs/r1")
        self.assertEqual(resolved, os.path.abspath("/scratch/here"))

    def test_without_a_run_dir_the_parsed_value_is_only_made_absolute(self):
        args = parser_for("plan_positives").parse_args([])
        resolved = runtime.resolve_out_dir(args, "plan_positives")
        self.assertEqual(resolved,
                         os.path.abspath(os.path.join(".", "plan_positives")))


class TestDirectoryConvention(unittest.TestCase):
    def test_the_layout_is_the_fixed_one(self):
        self.assertEqual(step_out_dir("/runs/r1", "extract_cavities"),
                         os.path.join("/runs/r1", "steps", "extract_cavities"))
        self.assertEqual(workflow_log_path("/runs/r1"),
                         os.path.join("/runs/r1", "logs", "workflow.log"))


# --- configure_logging --------------------------------------------------------


class TestConfigureLogging(LoggingIsolation):
    def test_it_creates_the_directory_and_writes_to_the_expected_path(self):
        self.track("step_a")
        out_dir = os.path.join(self.dir, "steps", "step_a")
        self.assertFalse(os.path.exists(out_dir))
        log = configure_logging("step_a", out_dir=out_dir)
        expected = os.path.join(out_dir, "step_a.log")
        self.assertTrue(os.path.isdir(out_dir))
        self.assertTrue(os.path.exists(expected))
        log.info("hello from step_a")
        self.assertIn("hello from step_a", read(expected))
        self.assertEqual(log.name, "step_a")
        self.assertIsNot(log, logging.getLogger())

    def test_an_explicit_log_file_wins_over_the_default(self):
        self.track("step_b")
        out_dir = os.path.join(self.dir, "b")
        target = os.path.join(self.dir, "elsewhere", "custom.log")
        log = configure_logging("step_b", out_dir=out_dir, log_file=target)
        log.info("into the override")
        self.assertTrue(os.path.exists(target))
        self.assertIn("into the override", read(target))
        self.assertFalse(os.path.exists(os.path.join(out_dir, "step_b.log")))

    def test_configuring_twice_duplicates_neither_handlers_nor_lines(self):
        self.track("step_c")
        out_dir = os.path.join(self.dir, "c")
        configure_logging("step_c", out_dir=out_dir)
        log = configure_logging("step_c", out_dir=out_dir)
        self.assertEqual(len(log.handlers), 2)  # one file, one console
        self.assertEqual(
            len([h for h in logging.getLogger("dock_selectivity").handlers
                 if getattr(h, runtime._HANDLER_TAG, None) == "step_c"]), 2)
        log.warning("exactly once please")
        text = read(os.path.join(out_dir, "step_c.log"))
        self.assertEqual(text.count("exactly once please"), 1)

    def test_reconfiguring_a_second_step_does_not_double_library_records(self):
        self.track("step_d1", "step_d2")
        configure_logging("step_d1", out_dir=os.path.join(self.dir, "d1"))
        configure_logging("step_d2", out_dir=os.path.join(self.dir, "d2"))
        library = logging.getLogger("dock_selectivity")
        tags = [getattr(h, runtime._HANDLER_TAG, None) for h in library.handlers]
        self.assertEqual(sorted(tags), ["step_d2", "step_d2"])
        # Other test modules quieten this logger for the rest of the process;
        # this test is about handler wiring, so give it a level of its own.
        self.track("dock_selectivity.schemas")
        schemas = logging.getLogger("dock_selectivity.schemas")
        schemas.setLevel(logging.NOTSET)
        schemas.info("wrote a manifest")
        second = read(os.path.join(self.dir, "d2", "step_d2.log"))
        self.assertEqual(second.count("wrote a manifest"), 1)
        first = read(os.path.join(self.dir, "d1", "step_d1.log"))
        self.assertNotIn("wrote a manifest", first)

    def test_it_never_touches_the_root_loggers_handlers(self):
        self.track("step_e")
        root = logging.getLogger()
        sentinel = logging.StreamHandler(io.StringIO())
        root.addHandler(sentinel)
        self.addCleanup(root.removeHandler, sentinel)
        before = list(root.handlers)
        before_level = root.level

        configure_logging("step_e", out_dir=os.path.join(self.dir, "e"))
        configure_logging("step_e", out_dir=os.path.join(self.dir, "e"))

        self.assertEqual(root.handlers, before)
        self.assertEqual(root.level, before_level)
        self.assertIn(sentinel, root.handlers)
        self.assertFalse(getattr(sentinel.stream, "closed", False))

    def test_it_never_calls_basic_config(self):
        self.track("step_f")
        with mock.patch.object(logging, "basicConfig") as basic:
            configure_logging("step_f", out_dir=os.path.join(self.dir, "f"))
        basic.assert_not_called()

    def test_importing_the_module_configures_nothing(self):
        # In a fresh interpreter, in an empty directory: importing the module
        # must leave no handler anywhere and no file behind. This is the failure
        # the seven prep-era scripts shipped -- a log file created by an import.
        # A subprocess, not importlib.reload: reloading would rebind the classes
        # this test module already imported.
        import subprocess

        program = (
            "import json, logging, os, sys\n"
            "sys.path.insert(0, %r)\n"
            "from dock_selectivity import runtime\n"
            "print(json.dumps({'root': len(logging.getLogger().handlers),\n"
            "                  'pkg': len(logging.getLogger('dock_selectivity').handlers),\n"
            "                  'files': sorted(os.listdir('.'))}))\n" % REPO_ROOT)
        out = subprocess.check_output([sys.executable, "-c", program],
                                      cwd=self.dir)
        import json as _json

        report = _json.loads(out.decode("utf-8").strip().splitlines()[-1])
        self.assertEqual(report["root"], 0)
        self.assertEqual(report["pkg"], 0)
        self.assertEqual(report["files"], [])

    def test_quiet_suppresses_the_console_but_still_writes_the_file(self):
        self.track("step_g")
        out_dir = os.path.join(self.dir, "g")
        captured = io.StringIO()
        with mock.patch.object(sys, "stderr", captured):
            log = configure_logging("step_g", out_dir=out_dir, quiet=True)
            log.error("silent on the console")
        self.assertEqual(captured.getvalue(), "")
        self.assertEqual([type(h).__name__ for h in log.handlers], ["FileHandler"])
        self.assertIn("silent on the console",
                      read(os.path.join(out_dir, "step_g.log")))

    def test_the_console_handler_is_on_stderr_so_stdout_stays_pipeable(self):
        self.track("step_h")
        out = io.StringIO()
        err = io.StringIO()
        with mock.patch.object(sys, "stdout", out), mock.patch.object(sys, "stderr", err):
            log = configure_logging("step_h", out_dir=os.path.join(self.dir, "h"))
            log.warning("to stderr")
        self.assertEqual(out.getvalue(), "")
        self.assertIn("to stderr", err.getvalue())

    def test_the_level_is_applied_even_when_the_root_logger_already_has_handlers(self):
        self.track("step_i")
        root = logging.getLogger()
        sentinel = logging.StreamHandler(io.StringIO())
        root.addHandler(sentinel)
        self.addCleanup(root.removeHandler, sentinel)

        out_dir = os.path.join(self.dir, "i")
        log = configure_logging("step_i", out_dir=out_dir, level="DEBUG", quiet=True)
        self.assertEqual(log.level, logging.DEBUG)
        log.debug("a debug record")
        self.assertIn("a debug record", read(os.path.join(out_dir, "step_i.log")))

        log = configure_logging("step_i", out_dir=out_dir, level="ERROR", quiet=True)
        self.assertEqual(log.level, logging.ERROR)
        log.warning("must not appear")
        self.assertNotIn("must not appear", read(os.path.join(out_dir, "step_i.log")))

    def test_noisy_third_party_loggers_are_clamped_to_warning(self):
        self.track("step_j", "matplotlib", "numexpr", "urllib3", "PIL", "fontTools")
        logging.getLogger("matplotlib").setLevel(logging.DEBUG)
        configure_logging("step_j", out_dir=os.path.join(self.dir, "j"),
                          level="DEBUG")
        for name in ("matplotlib", "numexpr", "urllib3", "PIL", "fontTools"):
            self.assertGreaterEqual(logging.getLogger(name).level, logging.WARNING,
                                    "%s would flood a DEBUG run" % name)

    def test_the_file_format_carries_an_iso_timestamp_level_and_logger_name(self):
        self.track("step_k")
        out_dir = os.path.join(self.dir, "k")
        log = configure_logging("step_k", out_dir=out_dir, quiet=True)
        log.warning("formatted")
        line = read(os.path.join(out_dir, "step_k.log")).strip()
        self.assertRegex(line, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
                               r"[.,]?\d*\s+WARNING\s+step_k\s+formatted$")

    def test_an_unwritable_log_file_warns_and_leaves_the_step_running(self):
        self.track("step_l")
        blocker = os.path.join(self.dir, "blocked")
        with open(blocker, "w", encoding="utf-8") as handle:
            handle.write("I am a file, not a directory\n")
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            log = configure_logging("step_l", out_dir=os.path.join(blocker, "deep"))
            log.info("the step still runs")
        self.assertNotIn("FileHandler", [type(h).__name__ for h in log.handlers])
        self.assertIn("cannot write the log file", err.getvalue())

    def test_an_unknown_level_is_rejected_rather_than_defaulted(self):
        with self.assertRaises(ValueError):
            configure_logging("step_m", out_dir=self.dir, level="CHATTY")


class TestLogResolvedConfiguration(LoggingIsolation):
    def test_it_prints_every_value_and_marks_what_the_command_line_set(self):
        self.track("step_n")
        out_dir = os.path.join(self.dir, "n")
        args = args_for("step_n", out_dir, ["--log-level", "DEBUG"])
        log = configure_logging("step_n", out_dir=out_dir, quiet=True)
        log_resolved_configuration(log, args,
                                   argv=["--out-dir", out_dir,
                                         "--log-level", "DEBUG"])
        text = read(os.path.join(out_dir, "step_n.log"))
        self.assertIn("resolved configuration", text)
        self.assertIn("log_level", text)
        self.assertIn("<- --log-level", text)
        self.assertIn("<- --out-dir", text)
        # not given on the command line, so no marker
        self.assertNotIn("<- --notify-title", text)

    def test_marked_defaults_false_drops_the_markers(self):
        self.track("step_o")
        out_dir = os.path.join(self.dir, "o")
        args = args_for("step_o", out_dir)
        log = configure_logging("step_o", out_dir=out_dir, quiet=True)
        log_resolved_configuration(log, args, marked_defaults=False,
                                   argv=["--out-dir", out_dir])
        self.assertNotIn("<- --", read(os.path.join(out_dir, "step_o.log")))

    def test_a_credential_looking_argument_is_redacted(self):
        self.track("step_p")
        out_dir = os.path.join(self.dir, "p")
        log = configure_logging("step_p", out_dir=out_dir, quiet=True)
        log_resolved_configuration(log, argparse.Namespace(api_token=TOKEN),
                                   argv=[])
        text = read(os.path.join(out_dir, "step_p.log"))
        self.assertNotIn(TOKEN, text)
        self.assertIn("<redacted>", text)


# --- Pushover -----------------------------------------------------------------


class FakeResponse:
    """Minimal stand-in for the object urlopen returns."""

    def __init__(self, status=200, body=b'{"status":1}'):
        self.status = status
        self._body = body
        self.closed = False

    def getcode(self):
        return self.status

    def read(self):
        return self._body

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class TestPushoverConfiguration(unittest.TestCase):
    def test_disabled_by_default_and_send_is_a_no_op_returning_false(self):
        notifier = Pushover()
        self.assertFalse(notifier.enabled)
        with mock.patch.object(runtime.urllib.request, "urlopen") as urlopen:
            self.assertFalse(notifier.send("title", "message"))
        urlopen.assert_not_called()

    def test_from_env_reads_only_the_environment(self):
        env = {"PUSHOVER_TOKEN": TOKEN, "PUSHOVER_USER_KEY": USER_KEY,
               "PUSHOVER_DEVICE": "phone"}
        notifier = Pushover.from_env(True, ["all"], title="campaign", environ=env)
        self.assertTrue(notifier.enabled)
        self.assertEqual(notifier.token, TOKEN)
        self.assertEqual(notifier.user_key, USER_KEY)
        self.assertEqual(notifier.device, "phone")
        self.assertEqual(notifier.title, "campaign")
        self.assertEqual(notifier.events, ("failure", "start", "success"))
        notifier.validate()

    def test_validate_raises_when_enabled_without_credentials(self):
        notifier = Pushover.from_env(True, ["failure"], environ={})
        with self.assertRaises(NotifyConfigurationError) as ctx:
            notifier.validate()
        message = str(ctx.exception)
        self.assertIn("PUSHOVER_TOKEN", message)
        self.assertIn("PUSHOVER_USER_KEY", message)

    def test_validate_raises_when_only_one_credential_is_present(self):
        notifier = Pushover.from_env(True, None, environ={"PUSHOVER_TOKEN": TOKEN})
        with self.assertRaises(NotifyConfigurationError) as ctx:
            notifier.validate()
        self.assertIn("PUSHOVER_USER_KEY", str(ctx.exception))
        self.assertNotIn(TOKEN, str(ctx.exception))

    def test_validate_is_silent_when_disabled_even_without_credentials(self):
        self.assertIsNone(Pushover.from_env(False, None, environ={}).validate())

    def test_wants_respects_the_requested_events(self):
        notifier = Pushover(token=TOKEN, user_key=USER_KEY, enabled=True,
                            events=["failure"])
        self.assertTrue(notifier.wants("failure"))
        self.assertFalse(notifier.wants("success"))
        self.assertFalse(notifier.wants("start"))
        self.assertFalse(Pushover(events=["all"]).wants("failure"))


def notifier_for_tests(logger=None, **kwargs):
    kwargs.setdefault("token", TOKEN)
    kwargs.setdefault("user_key", USER_KEY)
    kwargs.setdefault("enabled", True)
    kwargs.setdefault("events", ["all"])
    kwargs.setdefault("backoff", (0.0, 0.0))  # tests never sleep
    return Pushover(logger=logger, **kwargs)


class TestPushoverSend(LoggingIsolation):
    def capture(self, name="notify_step"):
        self.track(name)
        out_dir = os.path.join(self.dir, name)
        log = configure_logging(name, out_dir=out_dir, quiet=True, level="DEBUG")
        return log, os.path.join(out_dir, name + ".log")

    def test_a_successful_send_posts_the_documented_body_to_the_api_url(self):
        notifier = notifier_for_tests()
        seen = {}

        def fake_urlopen(request, timeout=None):
            seen["url"] = request.full_url
            seen["data"] = request.data
            seen["timeout"] = timeout
            seen["method"] = request.get_method()
            return FakeResponse()

        with mock.patch.object(runtime.urllib.request, "urlopen", fake_urlopen):
            self.assertTrue(notifier.send("Docking", "campaign finished"))

        self.assertEqual(seen["url"], "https://api.pushover.net/1/messages.json")
        self.assertEqual(seen["method"], "POST")
        self.assertEqual(seen["timeout"], 10.0)
        body = urllib.parse.parse_qs(seen["data"].decode("utf-8"))
        self.assertEqual(body["token"], [TOKEN])
        self.assertEqual(body["user"], [USER_KEY])
        self.assertEqual(body["title"], ["Docking"])
        self.assertEqual(body["message"], ["campaign finished"])
        self.assertEqual(body["priority"], ["0"])
        self.assertNotIn("device", body)

    def test_the_device_and_priority_are_forwarded_when_set(self):
        notifier = notifier_for_tests(device="workstation")
        seen = {}

        def fake_urlopen(request, timeout=None):
            seen["data"] = request.data
            return FakeResponse()

        with mock.patch.object(runtime.urllib.request, "urlopen", fake_urlopen):
            self.assertTrue(notifier.send("t", "m", priority=1))
        body = urllib.parse.parse_qs(seen["data"].decode("utf-8"))
        self.assertEqual(body["device"], ["workstation"])
        self.assertEqual(body["priority"], ["1"])

    def test_the_message_is_truncated_to_the_pushover_limit(self):
        notifier = notifier_for_tests()
        seen = {}

        def fake_urlopen(request, timeout=None):
            seen["data"] = request.data
            return FakeResponse()

        long_message = "x" * 5000
        with mock.patch.object(runtime.urllib.request, "urlopen", fake_urlopen):
            self.assertTrue(notifier.send("t" * 900, long_message))
        body = urllib.parse.parse_qs(seen["data"].decode("utf-8"))
        self.assertEqual(len(body["message"][0]), runtime.PUSHOVER_MESSAGE_LIMIT)
        self.assertTrue(body["message"][0].endswith("... [truncated]"))
        self.assertEqual(len(body["title"][0]), runtime.PUSHOVER_TITLE_LIMIT)

    def test_a_message_at_the_limit_is_left_alone(self):
        notifier = notifier_for_tests()
        seen = {}

        def fake_urlopen(request, timeout=None):
            seen["data"] = request.data
            return FakeResponse()

        exact = "y" * runtime.PUSHOVER_MESSAGE_LIMIT
        with mock.patch.object(runtime.urllib.request, "urlopen", fake_urlopen):
            notifier.send("t", exact)
        body = urllib.parse.parse_qs(seen["data"].decode("utf-8"))
        self.assertEqual(body["message"][0], exact)

    def test_a_url_error_never_raises_and_returns_false(self):
        log, _ = self.capture("notify_urlerror")
        notifier = notifier_for_tests(logger=log)
        with mock.patch.object(runtime.urllib.request, "urlopen",
                               side_effect=urllib.error.URLError("no route to host")):
            self.assertFalse(notifier.send("t", "m"))

    def test_a_socket_timeout_never_raises_and_returns_false(self):
        log, _ = self.capture("notify_timeout")
        notifier = notifier_for_tests(logger=log)
        with mock.patch.object(runtime.urllib.request, "urlopen",
                               side_effect=socket.timeout("timed out")):
            self.assertFalse(notifier.send("t", "m"))

    def test_an_http_500_response_object_returns_false(self):
        log, _ = self.capture("notify_500_response")
        notifier = notifier_for_tests(logger=log)
        with mock.patch.object(runtime.urllib.request, "urlopen",
                               return_value=FakeResponse(status=500)) as urlopen:
            self.assertFalse(notifier.send("t", "m"))
        self.assertEqual(urlopen.call_count, 3)  # one try plus two retries

    def test_an_http_500_error_is_retried_then_reported(self):
        log, _ = self.capture("notify_500_error")
        notifier = notifier_for_tests(logger=log)
        error = urllib.error.HTTPError(runtime.PUSHOVER_API_URL, 500,
                                       "Server Error", {}, None)
        with mock.patch.object(runtime.urllib.request, "urlopen",
                               side_effect=error) as urlopen:
            self.assertFalse(notifier.send("t", "m"))
        self.assertEqual(urlopen.call_count, 3)

    def test_an_http_400_is_not_retried_because_retrying_cannot_fix_it(self):
        log, _ = self.capture("notify_400")
        notifier = notifier_for_tests(logger=log)
        error = urllib.error.HTTPError(runtime.PUSHOVER_API_URL, 400,
                                       "Bad Request", {}, None)
        with mock.patch.object(runtime.urllib.request, "urlopen",
                               side_effect=error) as urlopen:
            self.assertFalse(notifier.send("t", "m"))
        self.assertEqual(urlopen.call_count, 1)

    def test_a_transient_failure_followed_by_success_returns_true(self):
        log, _ = self.capture("notify_recovers")
        notifier = notifier_for_tests(logger=log)
        outcomes = [urllib.error.URLError("flaky"), FakeResponse()]

        def fake_urlopen(request, timeout=None):
            item = outcomes.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with mock.patch.object(runtime.urllib.request, "urlopen", fake_urlopen):
            self.assertTrue(notifier.send("t", "m"))
        self.assertEqual(outcomes, [])

    def test_the_token_never_appears_in_any_log_record_of_a_failed_send(self):
        log, path = self.capture("notify_secrecy")
        notifier = notifier_for_tests(logger=log)
        # The failure text itself carries the secret, the way a real urllib error
        # quoting the request URL would.
        boom = urllib.error.URLError(
            "connection refused while posting token=%s user=%s" % (TOKEN, USER_KEY))
        with mock.patch.object(runtime.urllib.request, "urlopen", side_effect=boom):
            self.assertFalse(notifier.send("t", "m"))
        text = read(path)
        self.assertIn("not delivered", text)
        self.assertNotIn(TOKEN, text)
        self.assertNotIn(USER_KEY, text)
        self.assertIn("<redacted>", text)

    def test_an_enabled_notifier_without_credentials_warns_instead_of_raising(self):
        log, path = self.capture("notify_uncredentialed")
        notifier = Pushover(enabled=True, events=["all"], logger=log)
        with mock.patch.object(runtime.urllib.request, "urlopen") as urlopen:
            self.assertFalse(notifier.send("t", "m"))
        urlopen.assert_not_called()
        self.assertIn("notification skipped", read(path))

    def test_retries_sleep_with_backoff_between_attempts(self):
        log, _ = self.capture("notify_backoff")
        notifier = notifier_for_tests(logger=log, backoff=(1.0, 3.0))
        with mock.patch.object(runtime.time, "sleep") as sleep, \
                mock.patch.object(runtime.urllib.request, "urlopen",
                                  side_effect=urllib.error.URLError("down")):
            self.assertFalse(notifier.send("t", "m"))
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1.0, 3.0])


# --- step() -------------------------------------------------------------------


class TestStep(LoggingIsolation):
    def setUp(self):
        super().setUp()
        # No ambient credentials: every notification test states its own.
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("PUSHOVER_TOKEN", "PUSHOVER_USER_KEY", "PUSHOVER_DEVICE"):
            os.environ.pop(name, None)

    def credentials(self):
        os.environ["PUSHOVER_TOKEN"] = TOKEN
        os.environ["PUSHOVER_USER_KEY"] = USER_KEY

    def test_it_creates_the_out_dir_logs_the_configuration_and_the_elapsed_time(self):
        self.track("phase")
        out_dir = os.path.join(self.dir, "steps", "phase")
        args = args_for("phase", out_dir, ["--quiet"])
        with step("phase", args) as ctx:
            self.assertTrue(os.path.isdir(out_dir))
            self.assertEqual(ctx.out_dir, os.path.abspath(out_dir))
            self.assertEqual(ctx.step_name, "phase")
            ctx.logger.info("doing the work")
        text = read(os.path.join(out_dir, "phase.log"))
        self.assertIn("phase: start", text)
        self.assertIn("resolved configuration", text)
        self.assertIn("doing the work", text)
        self.assertIn("phase: done in", text)

    def test_an_exception_is_logged_with_its_traceback_and_re_raised(self):
        self.track("failing")
        out_dir = os.path.join(self.dir, "failing")
        args = args_for("failing", out_dir, ["--quiet", "--log-level", "DEBUG"])
        with self.assertRaises(RuntimeError):
            with step("failing", args):
                raise RuntimeError("ledock is not on this machine")
        text = read(os.path.join(out_dir, "failing.log"))
        self.assertIn("FAILED", text)
        self.assertIn("ledock is not on this machine", text)
        self.assertIn("Traceback (most recent call last)", text)

    def test_notify_without_credentials_exits_2_at_startup_before_any_work(self):
        self.track("early")
        out_dir = os.path.join(self.dir, "early")
        args = args_for("early", out_dir, ["--quiet", "--notify"])
        ran = []
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            with self.assertRaises(SystemExit) as ctx:
                with step("early", args):
                    ran.append(True)
        self.assertEqual(ctx.exception.code, 2)
        self.assertEqual(ran, [])
        self.assertIn("PUSHOVER_TOKEN", err.getvalue())

    def test_the_startup_refusal_creates_nothing_on_the_way_out(self):
        """"Exit 2 for a missing prerequisite, and create nothing on the way out."

        The credential check used to run *after* configure_logging(), so a refusal
        left behind the step directory and an empty log file it had already
        decided not to write to.
        """
        self.track("untouched")
        out_dir = os.path.join(self.dir, "untouched")
        args = args_for("untouched", out_dir, ["--quiet", "--notify"])
        with mock.patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                with step("untouched", args):
                    pass
        self.assertFalse(os.path.exists(out_dir),
                         "a refused step created %s" % out_dir)

    def test_failure_is_notified_by_default_and_success_is_not(self):
        self.credentials()
        self.track("noisy_fail")
        out_dir = os.path.join(self.dir, "noisy_fail")
        args = args_for("noisy_fail", out_dir, ["--quiet", "--notify"])
        with mock.patch.object(Pushover, "send", return_value=True) as send:
            with self.assertRaises(ValueError):
                with step("noisy_fail", args):
                    raise ValueError("gnina queue is empty")
        self.assertEqual(send.call_count, 1)
        title, message = send.call_args.args[0], send.call_args.args[1]
        self.assertEqual(title, "noisy_fail")
        self.assertIn("FAILED", message)
        self.assertIn("gnina queue is empty", message)
        self.assertEqual(send.call_args.kwargs.get("priority"), 1)

    def test_a_clean_run_sends_nothing_under_the_default_failure_only_policy(self):
        self.credentials()
        self.track("quiet_success")
        out_dir = os.path.join(self.dir, "quiet_success")
        args = args_for("quiet_success", out_dir, ["--quiet", "--notify"])
        with mock.patch.object(Pushover, "send", return_value=True) as send:
            with step("quiet_success", args):
                pass
        send.assert_not_called()

    def test_notify_on_success_sends_on_a_clean_exit(self):
        self.credentials()
        self.track("won")
        out_dir = os.path.join(self.dir, "won")
        args = args_for("won", out_dir,
                        ["--quiet", "--notify", "--notify-on", "success"])
        with mock.patch.object(Pushover, "send", return_value=True) as send:
            with step("won", args):
                pass
        self.assertEqual(send.call_count, 1)
        self.assertIn("finished", send.call_args.args[1])

    def test_notify_on_all_sends_start_and_then_failure(self):
        self.credentials()
        self.track("all_events")
        out_dir = os.path.join(self.dir, "all_events")
        args = args_for("all_events", out_dir,
                        ["--quiet", "--notify", "--notify-on", "all",
                         "--notify-title", "campaign 2026-08"])
        with mock.patch.object(Pushover, "send", return_value=True) as send:
            with self.assertRaises(KeyError):
                with step("all_events", args):
                    raise KeyError("uniprot")
        self.assertEqual(send.call_count, 2)
        self.assertIn("started", send.call_args_list[0].args[1])
        self.assertIn("FAILED", send.call_args_list[1].args[1])
        for call in send.call_args_list:
            self.assertEqual(call.args[0], "campaign 2026-08")

    def test_notifications_are_off_unless_notify_is_given(self):
        self.credentials()
        self.track("silent")
        out_dir = os.path.join(self.dir, "silent")
        args = args_for("silent", out_dir, ["--quiet", "--notify-on", "all"])
        with mock.patch.object(Pushover, "send", return_value=True) as send:
            with self.assertRaises(RuntimeError):
                with step("silent", args):
                    raise RuntimeError("boom")
        send.assert_not_called()

    def test_a_failed_notification_does_not_change_the_outcome_of_the_step(self):
        self.credentials()
        self.track("resilient")
        out_dir = os.path.join(self.dir, "resilient")
        args = args_for("resilient", out_dir,
                        ["--quiet", "--notify", "--notify-on", "success"])
        with mock.patch.object(runtime.urllib.request, "urlopen",
                               side_effect=urllib.error.URLError("offline")), \
                mock.patch.object(runtime.time, "sleep"):
            with step("resilient", args) as ctx:
                ctx.logger.info("three days of docking")
        text = read(os.path.join(out_dir, "resilient.log"))
        self.assertIn("three days of docking", text)
        self.assertIn("not delivered", text)
        self.assertNotIn(TOKEN, text)

    def test_a_zero_exit_inside_the_block_counts_as_success(self):
        self.credentials()
        self.track("exit_zero")
        out_dir = os.path.join(self.dir, "exit_zero")
        args = args_for("exit_zero", out_dir,
                        ["--quiet", "--notify", "--notify-on", "all"])
        with mock.patch.object(Pushover, "send", return_value=True) as send:
            with self.assertRaises(SystemExit):
                with step("exit_zero", args):
                    raise SystemExit(0)
        messages = [c.args[1] for c in send.call_args_list]
        self.assertEqual(len(messages), 2)
        self.assertIn("started", messages[0])
        self.assertIn("finished", messages[1])

    def test_a_nonzero_exit_inside_the_block_counts_as_failure(self):
        self.credentials()
        self.track("exit_two")
        out_dir = os.path.join(self.dir, "exit_two")
        args = args_for("exit_two", out_dir, ["--quiet", "--notify"])
        with mock.patch.object(Pushover, "send", return_value=True) as send:
            with self.assertRaises(SystemExit) as ctx:
                with step("exit_two", args):
                    raise SystemExit(2)
        self.assertEqual(ctx.exception.code, 2)
        self.assertEqual(send.call_count, 1)
        self.assertIn("exit code 2", send.call_args.args[1])

    def test_a_supplied_logger_is_used_as_is(self):
        self.track("borrowed")
        out_dir = os.path.join(self.dir, "borrowed")
        own = configure_logging("borrowed", out_dir=out_dir, quiet=True)
        args = args_for("borrowed", out_dir, ["--quiet"])
        with step("borrowed", args, logger=own) as ctx:
            self.assertIs(ctx.logger, own)
        self.assertEqual(len(own.handlers), 1)

    def test_the_context_exposes_an_ad_hoc_notify_helper(self):
        self.credentials()
        self.track("adhoc")
        out_dir = os.path.join(self.dir, "adhoc")
        args = args_for("adhoc", out_dir,
                        ["--quiet", "--notify", "--notify-on", "success"])
        with mock.patch.object(Pushover, "send", return_value=True) as send:
            with step("adhoc", args) as ctx:
                self.assertFalse(ctx.notify("failure", "not requested"))
                self.assertTrue(ctx.notify("success", "halfway there"))
                self.assertGreaterEqual(ctx.elapsed, 0.0)
        self.assertIn("halfway there", [c.args[1] for c in send.call_args_list])

    def test_the_step_writes_its_log_under_the_fixed_run_directory_layout(self):
        self.track("extract_cavities")
        run_dir = os.path.join(self.dir, "run")
        out_dir = step_out_dir(run_dir, "extract_cavities")
        args = args_for("extract_cavities", out_dir, ["--quiet"])
        with step("extract_cavities", args) as ctx:
            ctx.logger.info("ok")
        expected = os.path.join(run_dir, "steps", "extract_cavities",
                                "extract_cavities.log")
        self.assertTrue(os.path.exists(expected))
        self.assertEqual(ctx.log_file, os.path.abspath(expected))


class TestWriteStepSummary(unittest.TestCase):
    def test_it_writes_deterministic_json_beside_the_step_log(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "steps", "audit")
            path = runtime.write_step_summary(out_dir, "audit",
                                              {"jobs": 3, "ok": True})
            self.assertEqual(path, os.path.join(out_dir, "audit_summary.json"))
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"jobs": 3, "ok": True})


if __name__ == "__main__":
    unittest.main()
