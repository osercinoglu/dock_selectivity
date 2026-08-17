"""One runtime contract for every pipeline step: flags, logging, notifications.

Why this module exists
----------------------

Before it, the repo had three incompatible generations of logging and no
notification path at all:

* Seven prep-era scripts called ``logging.basicConfig(filename=...)`` at
  **module import**, so the first module imported won the single ``FileHandler``
  and the other six never wrote a file at all; every additional import stacked
  another stdout handler, so a single record printed six times.  The file landed
  in whatever directory the process happened to start in -- which is why running
  the test suite left 1.7 MB of ``*.log`` in the repository working tree, and why
  a stray ``docking_automation.log`` sits beside the campaign's job directories
  and has to be special-cased by name in ``classify_existing.py``.
* The newer stage scripts do the right thing (named logger, configured in
  ``main()``) but write **no file at all**, and their ``--log-level`` is silently
  ignored whenever the root logger already has handlers, because
  ``logging.basicConfig`` is a no-op in that case.
* ``batch_dock.py`` cleared *every* root handler -- including handlers a caller
  or a test harness installed -- and defaulted the root level to ``DEBUG``, where
  ``matplotlib.font_manager`` alone contributes 600+ lines per run.

And a three-day docking run that dies at hour 61 tells nobody.

The rules this module implements
--------------------------------

1. **Import configures nothing.**  Logging is set up from ``main()``, after the
   command line is parsed, never at import time.
2. **A step's log lives in the step's own directory**, never in the invocation
   cwd and never beside artifacts another stage enumerates::

       <run-dir>/steps/<step_key>/<step_key>.log   one step
       <run-dir>/logs/workflow.log                 the whole workflow

   Artifact locations do not move: ``manifests/``, ``structures/``,
   ``docking_results/`` and ``analysis/`` keep their paths and their ``--out``
   flags.
3. **Handlers are owned, tagged and idempotent.**  Re-configuring a step removes
   only the handlers this module installed for that step; a handler someone else
   owns is never closed, and the root logger's handlers are never touched.
4. **Notifications never kill the run.**  A failed Pushover POST is a WARNING and
   ``False``; losing a notification must not lose a campaign.  The credentials
   come from the environment only, and are redacted before anything is logged.
5. **Bad notification configuration fails at startup, not at hour 61.**
   ``--notify`` with no credentials in the environment exits 2 from :func:`step`
   before any work begins.

Usage::

    import argparse
    from dock_selectivity.runtime import add_runtime_arguments, resolve_out_dir, step

    def build_parser():
        parser = argparse.ArgumentParser(prog="extract_cavities.py")
        parser.add_argument("--archive", required=True, help="CavitySpace tarball")
        parser.add_argument("--run-dir", required=True, help="campaign run directory")
        return add_runtime_arguments(parser, "extract_cavities")

    def main(argv=None):
        args = build_parser().parse_args(argv)
        resolve_out_dir(args, "extract_cavities", args.run_dir)
        with step("extract_cavities", args) as ctx:
            ctx.logger.info("extracting from %s", args.archive)
        return 0

Environment (notifications only, never a flag, never a file in the repo)::

    PUSHOVER_TOKEN       application token
    PUSHOVER_USER_KEY    user or group key
    PUSHOVER_DEVICE      optional device name

This module is standard library only: no third-party import, so importing it
costs nothing and cannot fail on a machine that lacks pandas.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "NotifyConfigurationError",
    "RuntimeArgumentConflict",
    "NOTIFY_EVENTS",
    "DEFAULT_NOTIFY_EVENTS",
    "PUSHOVER_API_URL",
    "PUSHOVER_MESSAGE_LIMIT",
    "PUSHOVER_TITLE_LIMIT",
    "STEPS_DIRNAME",
    "LOGS_DIRNAME",
    "WORKFLOW_LOG_BASENAME",
    "Pushover",
    "StepContext",
    "add_runtime_arguments",
    "configure_logging",
    "log_resolved_configuration",
    "resolve_log_path",
    "resolve_notify_events",
    "resolve_out_dir",
    "step",
    "step_out_dir",
    "workflow_log_path",
    "write_step_summary",
]

#: This module's own logger, and the fallback a Pushover gets when the caller
#: supplies none. It is deliberately never configured here -- see rule 1.
_LOG = logging.getLogger(__name__)


class NotifyConfigurationError(Exception):
    """Notifications were requested but cannot be delivered.

    Raised by :meth:`Pushover.validate` when ``--notify`` is set and
    ``PUSHOVER_TOKEN`` / ``PUSHOVER_USER_KEY`` are absent from the environment.
    :func:`step` turns it into exit code 2 *at startup*: an operator who asked to
    be told when the run ends must not discover at hour 61 that the credentials
    were never there.
    """


class RuntimeArgumentConflict(ValueError):
    """A parser already declares one of the runtime flags.

    :func:`add_runtime_arguments` owns ``--out-dir``, ``--log-file``,
    ``--log-level``, ``--quiet``, ``--notify``, ``--notify-on`` and
    ``--notify-title``.  A script that declares its own copy gets this error
    rather than a half-migrated CLI where two flags mean almost the same thing:
    delete the script's own declaration and let the shared helper add it.
    """


#: The notification events ``--notify-on`` accepts. ``all`` means all three.
NOTIFY_EVENTS = ("start", "success", "failure", "all")

#: Default when ``--notify-on`` is not given: tell me when it breaks.
DEFAULT_NOTIFY_EVENTS = ("failure",)

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"

#: Pushover truncates silently past these; we truncate visibly instead.
PUSHOVER_MESSAGE_LIMIT = 1024
PUSHOVER_TITLE_LIMIT = 250

#: The fixed run-directory layout (see rule 2).
STEPS_DIRNAME = "steps"
LOGS_DIRNAME = "logs"
WORKFLOW_LOG_BASENAME = "workflow.log"

#: Third-party loggers raised to WARNING so a ``--log-level DEBUG`` run stays
#: readable. ``matplotlib.font_manager`` alone emits 600+ DEBUG lines per import.
NOISY_LOGGERS = (
    "matplotlib",
    "matplotlib.font_manager",
    "numexpr",
    "numexpr.utils",
    "urllib3",
    "PIL",
    "fontTools",
    "asyncio",
    "parso",
)

#: Marks the handlers this module installed. Reconfiguration removes only these,
#: so a handler installed by a caller or a test harness is never closed.
_HANDLER_TAG = "_dock_selectivity_runtime_step"

#: Records from ``dock_selectivity.*`` (schemas' read/write provenance, the
#: invariant reports, the join statistics) also reach the step log, because they
#: are part of what the step did. Attached to the package logger, never to root.
_LIBRARY_LOGGER = __name__.split(".")[0]

_FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
_FILE_DATEFMT = "%Y-%m-%dT%H:%M:%S"
_CONSOLE_FORMAT = "%(levelname)-8s %(message)s"

#: Argument names whose value is never printed in full by
#: :func:`log_resolved_configuration`, on the off chance a script ever grows one.
_SECRET_HINTS = ("token", "secret", "password", "passwd", "user_key", "apikey",
                 "api_key")


# --- run-directory layout -----------------------------------------------------


def step_out_dir(run_dir: str, step_name: str) -> str:
    """``<run-dir>/steps/<step_name>`` -- a step's own directory.

    The one place a step may write logs and scratch state. Artifacts keep their
    existing homes (``manifests/``, ``structures/``, ``docking_results/``,
    ``analysis/``) and their existing ``--out`` flags.
    """
    return os.path.join(run_dir, STEPS_DIRNAME, step_name)


def workflow_log_path(run_dir: str) -> str:
    """``<run-dir>/logs/workflow.log`` -- the orchestrator's own log."""
    return os.path.join(run_dir, LOGS_DIRNAME, WORKFLOW_LOG_BASENAME)


def artifact_step_dir(step_name: str, run_dir: Optional[str] = None) -> str:
    """The step directory for a script whose ``--out-dir`` means something else.

    Four structure-preparation scripts owned ``--out-dir`` before the shared
    runtime existed, and for them it names *where the structures land* -- a
    contract ``run_full_workflow.py`` and ``batch_dock.py`` depend on. They still
    need somewhere to put their own log:

    * with ``--run-dir``, the standard ``<run-dir>/steps/<step_name>``;
    * without one, ``./<step_name>`` -- the same default
      :func:`add_runtime_arguments` gives every other script.

    Deriving it from ``--out`` instead looks tidier and is wrong twice over: an
    unqualified ``--out`` puts the log in the invocation cwd, which is the thing
    this whole layout exists to prevent, and a qualified one drops it into
    ``manifests/`` beside artifacts a later stage enumerates.
    """
    if run_dir:
        return step_out_dir(run_dir, step_name)
    return os.path.abspath(os.path.join(".", step_name))


# --- command-line arguments ---------------------------------------------------


def add_runtime_arguments(parser: argparse.ArgumentParser,
                          step_name: str,
                          default_out_dir: Optional[str] = None
                          ) -> argparse.ArgumentParser:
    """Add the shared runtime flags to ``parser`` and return it.

    Adds, with exactly these spellings::

        --out-dir DIR      this step's own output directory (created if absent)
        --log-file PATH    override the log file (default <out-dir>/<step>.log)
        --log-level LEVEL  DEBUG | INFO | WARNING | ERROR   (default INFO)
        --quiet            file log only; nothing on the console
        --notify           opt in to Pushover notifications (off by default)
        --notify-on EVENT  repeatable: start | success | failure | all
        --notify-title T   override the notification title

    ``default_out_dir`` overrides the ``./<step_name>`` default -- pass
    :func:`step_out_dir` when the step already knows its run directory.

    Raises :class:`RuntimeArgumentConflict` if the parser already declares one of
    these flags, rather than leaving two flags that almost agree.
    """
    if not step_name:
        raise ValueError("step_name is required; it names the logger and the log file")
    out_default = default_out_dir if default_out_dir else os.path.join(".", step_name)

    def add(*flags: str, **kwargs: Any) -> None:
        try:
            parser.add_argument(*flags, **kwargs)
        except argparse.ArgumentError as exc:
            raise RuntimeArgumentConflict(
                "{0} already declares {1} ({2}). The shared runtime owns this "
                "flag: delete the script's own declaration and call "
                "add_runtime_arguments() instead.".format(
                    getattr(parser, "prog", "the parser"), flags[0], exc))

    # Remembered so resolve_out_dir() can tell "the operator chose this" from
    # "nobody said, so put it under the run directory".
    parser.set_defaults(_runtime_out_dir_default=out_default)
    add("--out-dir", default=out_default,
        help=("this step's own directory for its log and scratch state, created "
              "if absent. Artifacts keep their own --out flags. Default: "
              + out_default))
    add("--log-file", default=None,
        help=("write the log here instead of <out-dir>/" + step_name + ".log. "
              "The log never lands in the invocation directory."))
    add("--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="console and file verbosity (default: INFO)")
    add("--quiet", action="store_true",
        help="log to the file only; print nothing on the console")
    add("--notify", action="store_true",
        help=("send Pushover notifications for this run. Credentials come from "
              "PUSHOVER_TOKEN / PUSHOVER_USER_KEY in the environment only; the "
              "run refuses to start if they are missing."))
    add("--notify-on", action="append", default=None, choices=list(NOTIFY_EVENTS),
        metavar="EVENT",
        help=("which events to notify about; repeatable. One of: "
              + " | ".join(NOTIFY_EVENTS) + " (default: failure)"))
    add("--notify-title", default=None,
        help="notification title (default: the step name)")
    return parser


def resolve_out_dir(args: Any, step_name: str,
                    run_dir: Optional[str] = None) -> str:
    """Settle ``--out-dir`` once the run directory is known, and return it.

    A script learns its ``--run-dir`` only *after* parsing, so the parser cannot
    default ``--out-dir`` to ``<run-dir>/steps/<step>``.  Call this straight
    after ``parse_args``: an ``--out-dir`` the operator actually typed is kept,
    otherwise the fixed layout wins.  ``args.out_dir`` is updated in place so the
    value :func:`step` uses and the value logged are the same one.
    """
    chosen = getattr(args, "out_dir", None)
    default = getattr(args, "_runtime_out_dir_default", os.path.join(".", step_name))
    if run_dir and (not chosen or chosen == default):
        resolved = step_out_dir(run_dir, step_name)
    else:
        resolved = chosen or step_out_dir(run_dir or ".", step_name)
    resolved = os.path.abspath(resolved)
    if hasattr(args, "out_dir"):
        args.out_dir = resolved
    return resolved


def resolve_notify_events(values: Optional[Iterable[str]]) -> Tuple[str, ...]:
    """Normalise ``--notify-on`` into a sorted tuple of concrete events.

    ``None`` (the flag was never given) means ``("failure",)``; ``all`` expands
    to every event. Unknown names raise :class:`ValueError` -- argparse already
    rejects them, so reaching here means a caller built the list by hand.
    """
    if values is None:
        return tuple(DEFAULT_NOTIFY_EVENTS)
    if isinstance(values, str):
        values = [values]
    wanted = set()
    for value in values:
        name = str(value).strip().lower()
        if name not in NOTIFY_EVENTS:
            raise ValueError(
                "unknown notify event {0!r}; expected one of {1}".format(
                    value, ", ".join(NOTIFY_EVENTS)))
        if name == "all":
            wanted.update(e for e in NOTIFY_EVENTS if e != "all")
        else:
            wanted.add(name)
    if not wanted:
        return tuple(DEFAULT_NOTIFY_EVENTS)
    return tuple(sorted(wanted))


# --- logging ------------------------------------------------------------------


def _as_level(level: Any) -> int:
    """Accept ``"DEBUG"``, ``logging.DEBUG`` or ``10``; default to INFO."""
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(str(level).strip().upper())
    if not isinstance(resolved, int):
        raise ValueError(
            "unknown log level {0!r}; expected DEBUG, INFO, WARNING or ERROR".format(
                level))
    return resolved


def _drop_our_handlers(target: logging.Logger, step_name: Optional[str],
                       close: bool = True) -> None:
    """Remove only the handlers this module installed, closing ours as we go.

    ``step_name=None`` means "any step" -- used on the shared library logger,
    whose handlers belong to whichever step configured most recently.

    ``close=False`` detaches without closing, and the library logger needs it:
    a handler tagged for a *different* step is still attached to that step's own
    logger, and closing it there leaves a closed handler live somewhere else.
    ``logging`` reopens a closed FileHandler on the next record, so the symptom
    is a FileNotFoundError from a step that did nothing wrong, hours later,
    whenever its directory has since gone away.
    """
    for handler in list(target.handlers):
        tag = getattr(handler, _HANDLER_TAG, None)
        if tag is None:
            continue
        if step_name is not None and tag != step_name:
            continue
        target.removeHandler(handler)
        if not close:
            continue
        try:
            handler.close()
        except Exception:  # pragma: no cover - closing a dead stream
            pass


def resolve_log_path(step_name: str,
                     out_dir: Optional[str] = None,
                     log_file: Optional[str] = None) -> str:
    """Where :func:`configure_logging` would write, without configuring anything.

    An explicit ``log_file`` always wins; otherwise
    ``<out-dir or ./<step_name>>/<step_name>.log``.
    """
    if log_file:
        return os.path.abspath(log_file)
    base = out_dir if out_dir else os.path.join(".", step_name)
    return os.path.abspath(os.path.join(base, step_name + ".log"))


def configure_logging(step_name: str,
                      out_dir: Optional[str] = None,
                      log_file: Optional[str] = None,
                      level: Any = "INFO",
                      quiet: bool = False) -> logging.Logger:
    """Configure and return the named logger for one step.

    Attaches exactly one ``FileHandler`` (at ``<out-dir>/<step_name>.log`` unless
    ``log_file`` overrides it) and, unless ``quiet``, one ``StreamHandler`` on
    **stderr** -- stderr so that piping a step's stdout stays clean and so log
    records share a stream with the tqdm bars rather than interleaving across two.

    The handlers are also attached to the ``dock_selectivity`` package logger, so
    the library's provenance records (schemas' reads and writes, invariant
    reports, join statistics) land in the step log where they belong.

    Guarantees:

    * **idempotent** -- calling it twice for the same step replaces this module's
      handlers instead of stacking a second copy of every line;
    * **the root logger is never touched**, so a caller's or a test harness's
      handlers survive;
    * ``logging.basicConfig`` is never called, and import time configures nothing;
    * a log file that cannot be opened degrades to console-only with a WARNING --
      never a crash, never a silent skip.
    """
    if not step_name:
        raise ValueError("step_name is required; it names the logger and the log file")
    resolved_level = _as_level(level)
    target = os.path.abspath(out_dir) if out_dir else os.path.abspath(
        os.path.join(".", step_name))

    step_logger = logging.getLogger(step_name)
    library_logger = logging.getLogger(_LIBRARY_LOGGER)
    # A step logger that already lives under dock_selectivity.* propagates INTO
    # the package logger, so attaching the same handler to both would print every
    # record twice.
    attach_library = not (step_logger.name == _LIBRARY_LOGGER
                          or step_logger.name.startswith(_LIBRARY_LOGGER + "."))
    # The step logger owns its handlers, so they are closed here; the library
    # logger merely borrows them, so they are only detached (see _drop_our_handlers).
    _drop_our_handlers(step_logger, step_name)
    if attach_library:
        _drop_our_handlers(library_logger, None, close=False)

    # basicConfig's level argument is a no-op once handlers exist anywhere; set
    # the level explicitly so a second main() in one interpreter still honours
    # --log-level.
    step_logger.setLevel(resolved_level)
    if attach_library:
        library_logger.setLevel(resolved_level)
        # This module owns the step's output, so a record must not ALSO reach the
        # root logger: whenever anything has configured root (a harness, a
        # notebook, a library that called basicConfig), propagation re-prints
        # every line through root's handlers -- which is the duplicated-console
        # defect this module exists to remove, and which makes --quiet ("file log
        # only, nothing on the console") a promise the runtime cannot keep.
        # Root's own handlers are still never touched.
        step_logger.propagate = False
        library_logger.propagate = False

    handlers: List[logging.Handler] = []
    file_error = None
    path = resolve_log_path(step_name, out_dir=target, log_file=log_file)
    try:
        os.makedirs(target, exist_ok=True)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, _FILE_DATEFMT))
        file_handler.setLevel(resolved_level)
        handlers.append(file_handler)
    except OSError as exc:
        file_error = exc

    if not quiet:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
        console.setLevel(resolved_level)
        handlers.append(console)

    for handler in handlers:
        setattr(handler, _HANDLER_TAG, step_name)
        step_logger.addHandler(handler)
        if attach_library:
            library_logger.addHandler(handler)

    for name in NOISY_LOGGERS:
        noisy = logging.getLogger(name)
        if noisy.level < logging.WARNING:
            noisy.setLevel(logging.WARNING)

    if file_error is not None:
        step_logger.warning(
            "cannot write the log file %s (%s); this run leaves no log on disk",
            path, file_error)
    else:
        setattr(step_logger, "log_file_path", path)
    return step_logger


def log_resolved_configuration(logger: logging.Logger,
                               args: Any,
                               marked_defaults: bool = True,
                               argv: Optional[Sequence[str]] = None,
                               redact: Optional[Any] = None) -> None:
    """Log every resolved argument, marking the ones the command line set.

    The format mirrors ``batch_dock.py --print-config``, which already prints
    ``NAME = value <- --flag`` for overridden settings, so one operator reads one
    format everywhere.  ``marked_defaults=False`` prints the values without the
    marker.  ``argv`` defaults to ``sys.argv[1:]`` and exists for tests.

    A value whose argument name looks like a credential is redacted, so a script
    that ever grows one cannot leak it into the run log.  ``redact`` is an
    additional value-based scrubber (:meth:`Pushover.redact`): name-based
    redaction cannot catch a credential the operator typed into some *other*
    flag, and the dump is the one place every argument's value is written down.
    """
    if args is None:
        return
    raw = vars(args) if hasattr(args, "__dict__") else dict(args)
    # Names starting with "_" are this module's own bookkeeping, not settings.
    values = {k: v for k, v in raw.items() if not k.startswith("_")}
    given = set(argv if argv is not None else sys.argv[1:])
    logger.info("resolved configuration (%d settings):", len(values))
    for name in sorted(values):
        value = values[name]
        flag = "--" + name.replace("_", "-")
        mark = ""
        if marked_defaults:
            on_cli = flag in given or any(
                a.startswith(flag + "=") for a in given if isinstance(a, str))
            mark = " <- " + flag if on_cli else ""
        if any(hint in name.lower() for hint in _SECRET_HINTS) and value:
            shown = "<redacted>"
        elif redact is not None and value:
            shown = redact(repr(value))
        else:
            shown = repr(value)
        logger.info("  %-28s = %s%s", name, shown, mark)


# --- notifications ------------------------------------------------------------


class Pushover:
    """Best-effort Pushover notifier: it tells you, or it tells the log.

    A notification is never worth a run.  :meth:`send` therefore catches
    everything -- network errors, timeouts, HTTP failures, malformed responses --
    logs one WARNING and returns ``False``.  The only thing that ever raises is
    :meth:`validate`, and it raises at startup, before any work has been done.

    Credentials are read from the environment by :meth:`from_env` and never from
    a flag, a file in the repo, or a log record: :meth:`redact` scrubs them out of
    anything on its way to the log.
    """

    ENV_TOKEN = "PUSHOVER_TOKEN"
    ENV_USER_KEY = "PUSHOVER_USER_KEY"
    ENV_DEVICE = "PUSHOVER_DEVICE"

    def __init__(self,
                 token: Optional[str] = None,
                 user_key: Optional[str] = None,
                 device: Optional[str] = None,
                 enabled: bool = False,
                 events: Optional[Iterable[str]] = None,
                 title: Optional[str] = None,
                 timeout: float = 10.0,
                 retries: int = 2,
                 backoff: Sequence[float] = (1.0, 3.0),
                 logger: Optional[logging.Logger] = None) -> None:
        self.token = token or None
        self.user_key = user_key or None
        self.device = device or None
        self.enabled = bool(enabled)
        self.events = resolve_notify_events(events)
        self.title = title or None
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.backoff = tuple(backoff)
        self.log = logger if logger is not None else _LOG

    @classmethod
    def from_env(cls,
                 enabled: bool,
                 events: Optional[Iterable[str]] = None,
                 title: Optional[str] = None,
                 environ: Optional[Dict[str, str]] = None,
                 logger: Optional[logging.Logger] = None) -> "Pushover":
        """Build a notifier from the environment.

        ``PUSHOVER_TOKEN`` and ``PUSHOVER_USER_KEY`` are required when
        ``enabled``; ``PUSHOVER_DEVICE`` is optional.  Missing credentials are
        *not* an error here -- :meth:`validate` reports them, so the caller
        decides when to fail (:func:`step` does it at startup).
        """
        env = os.environ if environ is None else environ
        return cls(token=env.get(cls.ENV_TOKEN),
                   user_key=env.get(cls.ENV_USER_KEY),
                   device=env.get(cls.ENV_DEVICE),
                   enabled=bool(enabled),
                   events=events,
                   title=title,
                   logger=logger)

    # -- configuration --

    def validate(self) -> None:
        """Raise :class:`NotifyConfigurationError` if enabled but unusable."""
        if not self.enabled:
            return
        missing = []
        if not self.token:
            missing.append(self.ENV_TOKEN)
        if not self.user_key:
            missing.append(self.ENV_USER_KEY)
        if missing:
            raise NotifyConfigurationError(
                "notifications were requested (--notify) but {0} {1} not set in "
                "the environment. Export them before starting the run: "
                "notification credentials are never a command-line flag and are "
                "never stored in the repository.".format(
                    " and ".join(missing),
                    "is" if len(missing) == 1 else "are"))

    def wants(self, event: str) -> bool:
        """Is ``event`` (start | success | failure) one we were asked to send?"""
        return self.enabled and event in self.events

    def redact(self, text: Any) -> str:
        """Replace the credentials with ``<redacted>`` wherever they appear."""
        out = text if isinstance(text, str) else str(text)
        for secret in (self.token, self.user_key):
            if secret:
                out = out.replace(secret, "<redacted>")
        return out

    # -- sending --

    @staticmethod
    def _truncate(text: Any, limit: int) -> str:
        out = text if isinstance(text, str) else str(text)
        if len(out) <= limit:
            return out
        marker = "... [truncated]"
        return out[: max(0, limit - len(marker))] + marker

    def send(self, title: str, message: str, priority: int = 0) -> bool:
        """POST one notification. Returns whether it was delivered; never raises.

        Retries transient failures (network errors, timeouts, HTTP 5xx) with
        backoff; a 4xx is the operator's configuration problem, so it is reported
        once and not retried.  A disabled notifier is a no-op returning ``False``.
        """
        if not self.enabled:
            return False
        try:
            self.validate()
        except NotifyConfigurationError as exc:
            self.log.warning("notification skipped: %s", self.redact(exc))
            return False

        payload = {
            "token": self.token,
            "user": self.user_key,
            "title": self._truncate(title or self.title or "", PUSHOVER_TITLE_LIMIT),
            "message": self._truncate(message, PUSHOVER_MESSAGE_LIMIT),
            "priority": str(int(priority)),
        }
        if self.device:
            payload["device"] = self.device
        body = urllib.parse.urlencode(payload).encode("utf-8")

        attempts = max(1, self.retries + 1)
        last_error = "unknown error"
        for attempt in range(attempts):
            retry, last_error = self._attempt(body)
            if retry is False and last_error is None:
                return True
            if not retry:
                break
            if attempt + 1 < attempts:
                delay = self.backoff[min(attempt, len(self.backoff) - 1)] \
                    if self.backoff else 0
                if delay:
                    time.sleep(delay)
        self.log.warning(
            "Pushover notification was not delivered (%s); the run is unaffected",
            self.redact(last_error))
        return False

    def _attempt(self, body: bytes) -> Tuple[bool, Optional[str]]:
        """One POST. Returns ``(should_retry, error)``; ``(False, None)`` is success."""
        request = urllib.request.Request(
            PUSHOVER_API_URL, data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            code = getattr(exc, "code", 0) or 0
            # 4xx is a bad token or a bad user key: retrying cannot fix it.
            return (500 <= code < 600, "HTTP {0} from Pushover".format(code))
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            return True, "{0}: {1}".format(type(exc).__name__, exc)
        except Exception as exc:  # pragma: no cover - defensive: never kill a run
            return False, "{0}: {1}".format(type(exc).__name__, exc)

        try:
            status = getattr(response, "status", None)
            if status is None and hasattr(response, "getcode"):
                status = response.getcode()
            status = int(status) if status is not None else 200
            if status >= 500:
                return True, "HTTP {0} from Pushover".format(status)
            if status >= 400:
                return False, "HTTP {0} from Pushover".format(status)
            return False, None
        except Exception as exc:  # pragma: no cover - defensive
            return False, "{0}: {1}".format(type(exc).__name__, exc)
        finally:
            try:
                response.close()
            except Exception:  # pragma: no cover
                pass


# --- the step wrapper ---------------------------------------------------------


class StepContext:
    """What :func:`step` hands to the body of a ``with`` block."""

    __slots__ = ("step_name", "logger", "notifier", "out_dir", "log_file",
                 "started_at")

    def __init__(self, step_name: str, logger: logging.Logger,
                 notifier: "Pushover", out_dir: str, log_file: str,
                 started_at: float) -> None:
        self.step_name = step_name
        self.logger = logger
        self.notifier = notifier
        self.out_dir = out_dir
        self.log_file = log_file
        self.started_at = started_at

    @property
    def elapsed(self) -> float:
        """Seconds since the step started."""
        return time.time() - self.started_at

    def notify(self, event: str, message: str, priority: int = 0) -> bool:
        """Send an ad-hoc notification if ``event`` was requested."""
        if not self.notifier.wants(event):
            return False
        title = self.notifier.title or self.step_name
        return self.notifier.send(title, message, priority=priority)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<StepContext {0} out_dir={1}>".format(self.step_name, self.out_dir)


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return "{0:.1f}s".format(seconds)
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return "{0}h{1:02d}m{2:02d}s".format(hours, minutes, secs)
    return "{0}m{1:02d}s".format(minutes, secs)


@contextmanager
def step(step_name: str, args: Any, logger: Optional[logging.Logger] = None):
    """Run one pipeline step: logging, the output directory, notifications.

    On entry it validates the notification configuration (exiting **2 at startup**
    if ``--notify`` was asked for and the environment has no credentials -- never
    mid-run, and before creating anything), then creates ``--out-dir``, configures
    logging into it, logs the resolved configuration and sends the ``start``
    notification if requested.

    On a clean exit it logs the elapsed time and sends ``success``.  On an
    exception it logs the traceback, sends ``failure`` with the exception text,
    and **re-raises**: this wrapper reports failures, it never swallows them.

    ``logger`` lets a caller supply an already-configured logger (the workflow
    orchestrator does this for its phases); logging is then left alone.
    """
    out_dir = getattr(args, "out_dir", None) or os.path.join(".", step_name)
    out_dir = os.path.abspath(out_dir)
    log_file = getattr(args, "log_file", None)
    level = getattr(args, "log_level", "INFO")
    quiet = bool(getattr(args, "quiet", False))

    # Validate the notification configuration FIRST, before anything touches the
    # disk. Two rules meet here: an operator who asked to be told when a three-day
    # job ends must not find out at hour 61 that they will not be, and a step that
    # refuses must create nothing on the way out -- configure_logging() would
    # already have made the out-dir and opened the log file. The refusal goes to
    # stderr only, because at this point there is deliberately no log to write to.
    notifier = Pushover.from_env(
        enabled=bool(getattr(args, "notify", False)),
        events=getattr(args, "notify_on", None),
        title=getattr(args, "notify_title", None))
    try:
        notifier.validate()
    except NotifyConfigurationError as exc:
        sys.stderr.write("{0}: {1}\n".format(step_name, exc))
        raise SystemExit(2)

    if logger is None:
        step_logger = configure_logging(step_name, out_dir=out_dir,
                                        log_file=log_file, level=level, quiet=quiet)
    else:
        step_logger = logger
        os.makedirs(out_dir, exist_ok=True)
    notifier.log = step_logger

    title = notifier.title or step_name
    started_at = time.time()
    context = StepContext(step_name, step_logger, notifier, out_dir,
                          resolve_log_path(step_name, out_dir, log_file), started_at)

    step_logger.info("=== %s: start ===", step_name)
    step_logger.info("out-dir: %s", out_dir)
    step_logger.info("log-file: %s", context.log_file)
    log_resolved_configuration(step_logger, args, redact=notifier.redact)
    if notifier.wants("start"):
        notifier.send(title, "{0}: started".format(step_name))

    try:
        yield context
    except SystemExit as exc:
        code = exc.code if exc.code is not None else 0
        if code in (0, "0"):
            elapsed = _format_elapsed(time.time() - started_at)
            step_logger.info("=== %s: done in %s ===", step_name, elapsed)
            if notifier.wants("success"):
                notifier.send(title, "{0}: finished in {1}".format(step_name, elapsed))
        else:
            elapsed = _format_elapsed(time.time() - started_at)
            step_logger.error("=== %s: exited with code %s after %s ===",
                              step_name, code, elapsed)
            if notifier.wants("failure"):
                notifier.send(
                    title,
                    "{0}: FAILED after {1} (exit code {2})".format(
                        step_name, elapsed, code),
                    priority=1)
        raise
    except BaseException as exc:
        elapsed = _format_elapsed(time.time() - started_at)
        step_logger.error("=== %s: FAILED after %s: %s: %s ===",
                          step_name, elapsed, type(exc).__name__, exc)
        step_logger.debug("%s", traceback.format_exc())
        if notifier.wants("failure"):
            notifier.send(
                title,
                "{0}: FAILED after {1}\n{2}: {3}".format(
                    step_name, elapsed, type(exc).__name__, exc),
                priority=1)
        raise
    else:
        elapsed = _format_elapsed(time.time() - started_at)
        step_logger.info("=== %s: done in %s ===", step_name, elapsed)
        if notifier.wants("success"):
            notifier.send(title, "{0}: finished in {1}".format(step_name, elapsed))


def write_step_summary(out_dir: str, step_name: str, summary: Dict[str, Any]) -> str:
    """Write ``<out-dir>/<step_name>_summary.json`` and return its path.

    A passing run should leave evidence that it passed, not just an absence of
    errors (the rule :mod:`dock_selectivity.invariants` already follows).
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "{0}_summary.json".format(step_name))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return path
