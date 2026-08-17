#!/usr/bin/env python3
"""The gnina pump: keeps one remote GPU fed and files each result where a local
run would have put it.

What this process is for
------------------------

``batch_dock.py`` used to run gnina as stage 2 of five, blocking the whole job
list until every gnina call returned. gnina now runs on a remote GPU box that has
exactly **one** GPU, so stage 2 was both the slowest stage and the one with the
least local work to do -- the local machine sat idle waiting for a network job.

With ``GNINA_ASYNC=1`` batch_dock enqueues the gnina jobs and starts this
process, then gets on with smina/GOLD/LeDock. This pump then, on its own clock:

* submits **one** job at a time to the GPU (§4: a local ``pump.lock`` so only one
  pump runs, plus the remote ``gpu.lock`` mkdir mutex so only one gnina runs);
* polls the detached remote job for its ``exit_code`` marker;
* collects finished jobs -- downloading ``out.sdf``/``out.log`` and running
  consensus_docker's own splitting, complex-building and log-parsing so the
  contents of ``<job>/gnina/`` are produced by exactly the same lines of code as
  a local run (§1);
* retries what is worth retrying and **abandons loudly** what is not.

It deliberately does not compute RMSD and does not touch any other tool's
directory. Stage 5 in batch_dock does that, after its barrier
(``gnina_queue.wait_for_completion``) has waited for this pump.

What it must never do
---------------------

* **Import batch_dock.** batch_dock calls ``logging.basicConfig`` and logs at
  import; importing it here would hijack this process's logging and append to
  ``docking_automation.log`` (spec D7).
* **Import pandas / rdkit / psutil.** The pump must start in well under a second
  and must not depend on the science stack. The science stack is loaded by the
  ``consensus_docker.py`` subprocess, where it belongs.
* **Reimplement pose splitting or score parsing.** There is exactly one
  definition of "the CNN affinity of pose 3" in this system and it lives in
  ``consensus_docker.parse_gnina``. This process shells out to it.
* **Kill a remote job.** The remote leg is ``setsid``-detached on purpose: when
  a local timeout fires, the GPU work continues and the next poll collects it.

Usage
-----

::

    # what batch_dock spawns
    gnina_collect.py --run-dir $RUN --daemon --log $RUN/gnina_queue/collector.log

    # one pass, collect whatever is ready, exit (what the tests drive)
    gnina_collect.py --run-dir $RUN --once

    # tell me on my phone when the pump abandons a job or drains the queue
    PUSHOVER_TOKEN=... PUSHOVER_USER_KEY=... \\
        gnina_collect.py --run-dir $RUN --daemon --notify --notify-on all

    # read-only introspection; takes no lock, safe while a pump runs
    gnina_collect.py --run-dir $RUN --status

    # ask a running pump to stop after its in-flight collect
    gnina_collect.py --run-dir $RUN --stop

    # operator reset of a terminal failure
    gnina_collect.py --run-dir $RUN --requeue DB00035__P30518__AVPR2__F1__cav1

Exit codes
----------

===  ============================================================
  0  the queue drained and every job is COLLECTED
  3  the queue drained but some jobs are ABANDONED
  4  stopped (signal, STOP file, ``--once``, ``--max-jobs``) with work left
  5  another collector holds the pump lock -- this one never waits
 78  the remote capability check is fatal (``--caps-file``)
  2  usage / the queue is not there / no params.json
  1  an unexpected error
===  ============================================================

Runtime flags
-------------
The shared runtime supplies ``--out-dir``, ``--log-file``, ``--log-level``,
``--quiet``, ``--notify``, ``--notify-on`` and ``--notify-title``; the pump's log
defaults to ``<run-dir>/steps/gnina_collect/gnina_collect.log`` and ``--log`` is
kept as the spelling batch_dock spawns the collector with.  ``--status``,
``--stop`` and ``--requeue`` are queries and deliberately configure nothing: a
question about a running pump must not create a directory or a log file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
import traceback

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:                                             # pragma: no cover - trivial
    from dock_selectivity import gnina_queue as gq
except ImportError:                              # pragma: no cover - trivial
    gq = None

# Standard library only, like the rest of this module: importing the shared
# runtime cannot pull in the science stack and cannot slow the pump's start.
from dock_selectivity.runtime import (  # noqa: E402
    add_runtime_arguments,
    resolve_out_dir,
    step,
)

#: Names the logger, the log file and the step directory
#: (``<run-dir>/steps/gnina_collect/gnina_collect.log``).
STEP_NAME = "gnina_collect"

LOG = logging.getLogger(STEP_NAME)

DEFAULT_POLL_SECONDS = 15.0
DEFAULT_COLLECT_WORKERS = 2
DEFAULT_MAX_ATTEMPTS = 3

#: How often the pump re-derives state from disk and the remote (§2 reconcile).
#: Reconciliation is only needed after a crash, so it is cheap to do rarely; it
#: always runs once at start, which is the case that matters.
RECONCILE_INTERVAL_SECONDS = 300.0

#: One pass may submit more than one job, but only when the previous one has
#: already finished -- that is the "keep the GPU fed" rule of §4, which says to
#: submit again immediately on a transition into REMOTE_DONE/FAILED rather than
#: idling a whole poll interval. The cap only guarantees the pass terminates.
MAX_SUBMIT_ROUNDS_PER_PASS = 16

#: Backoff after a retriable failure or a busy GPU, indexed by attempts so far.
BACKOFF_SECONDS = (5.0, 15.0, 30.0, 60.0)

#: SSH-level flakiness is capped separately from docking attempts: a thousand
#: dropped connections should not burn a job's three chances at the GPU.
MAX_TRANSIENT_ATTEMPTS = 10

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_USAGE = 2
EXIT_ABANDONED = 3
EXIT_STOPPED = 4
EXIT_LOCKED = 5
EXIT_CAPS_FATAL = 78

# --- the §11 failure taxonomy, as policy -------------------------------------
#
# Every code the queue can record maps to exactly one decision. Anything not in
# these tables is decided by the record's own ``error.retriable`` flag, and if
# that is absent the job is ABANDONED rather than retried forever -- an
# abandoned job is visible in summary.json, a silently looping one is not.

#: Retry lands here instead of PENDING: the GPU already produced good poses, the
#: local post-processing is what failed. Re-docking would throw away a finished
#: remote result and take another ~150 s of GPU time to reproduce it.
RETRY_TO_REMOTE_DONE = frozenset({"POSTPROCESS_FAILED", "COLLECT_TIMEOUT"})

#: "yes (1)" in §11: one retry, then ABANDONED regardless of --max-attempts,
#: because a second identical remote failure is a property of the inputs rather
#: than of the link.
ONE_RETRY_ONLY = frozenset({"REMOTE_GNINA_FAILED", "EMPTY_OUTPUT", "POSTPROCESS_FAILED"})

#: Permanent by construction. Retrying cannot change the outcome.
NEVER_RETRY = frozenset({
    "MISSING_INPUT", "SUBMIT_WROTE_RESULTS", "CAPS_FATAL", "PUMP_UNAVAILABLE",
})

#: Network flakiness: capped by transient_attempts, not by docking attempts.
TRANSIENT_CODES = frozenset({"SSH_TRANSIENT"})

RETRIABLE_CODES = frozenset({
    "SSH_TRANSIENT", "REMOTE_GNINA_FAILED", "STALLED", "REMOTE_DIR_GONE",
    "EMPTY_OUTPUT", "POSTPROCESS_FAILED", "COLLECT_TIMEOUT", "SUBMIT_NO_TICKET",
})

#: The state machine has no PENDING -> FAILED or REMOTE_DONE -> FAILED edge, so
#: recording a crash means walking the legal intent state first. Doing it this
#: way keeps every write to a record a legal transition, which is what makes
#: reconciliation able to trust the record at all.
_HOPS_TO_FAILED = {
    "PENDING": ("SUBMITTING",),
    "REMOTE_DONE": ("COLLECTING",),
}


class CollectorError(RuntimeError):
    """The pump cannot run against this queue. Reported, never worked around."""


class PumpLocked(CollectorError):
    """Another collector holds the pump lock. Two pumps would double-book the GPU."""


class RetryPlan(object):
    """What to do with one FAILED record. Immutable, produced by ``plan_retry``.

    ``counter`` names the budget this retry spends, and it is deliberately never
    ``"attempts"``: a docking attempt is consumed by ``submit_job`` when it
    actually launches something on the GPU, so incrementing it here as well
    would charge every failure twice and abandon jobs a retry early. The only
    counter the pump owns is ``transient_attempts``, because nothing else
    increments it.
    """

    __slots__ = ("to_state", "delay", "why", "counter")

    def __init__(self, to_state, delay, why, counter=None):
        self.to_state = to_state
        self.delay = float(delay)
        self.why = why
        self.counter = counter

    @property
    def gives_up(self):
        return self.to_state == "ABANDONED"

    def __repr__(self):                                  # pragma: no cover
        return "RetryPlan(%s, %.0fs, %r)" % (self.to_state, self.delay, self.why)

    def __eq__(self, other):                             # pragma: no cover
        return (isinstance(other, RetryPlan)
                and (self.to_state, self.delay, self.counter)
                == (other.to_state, other.delay, other.counter))


def _utcnow_iso():
    """ISO-8601 UTC with a trailing Z, matching the record schema (§3)."""
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ")


def _backoff(n):
    return BACKOFF_SECONDS[min(max(int(n), 0), len(BACKOFF_SECONDS) - 1)]


def postprocess_retries(record):
    """How many times this job has already been sent back for re-collection.

    Counted from the record's own history rather than a counter, because the
    FAILED -> REMOTE_DONE edge exists for exactly this purpose and cannot be
    taken for any other reason. A collect-side failure must not consume one of
    the job's docking attempts -- the GPU work is fine -- so it needs its own
    bound, and the history already is one.
    """
    n = 0
    for entry in (getattr(record, "history", None) or []):
        try:
            if entry.get("from") == "FAILED" and entry.get("to") == "REMOTE_DONE":
                n += 1
        except AttributeError:                            # pragma: no cover
            continue
    return n


def plan_retry(error, attempts, transient_attempts, retries_taken, max_attempts):
    """Decide what happens to a FAILED record. Pure: no I/O, no clock, no queue.

    ``retries_taken`` is ``postprocess_retries(record)`` -- the number of
    collect-side retries already spent, which is bounded separately from
    ``attempts`` because it never re-docks anything.

    Returns a :class:`RetryPlan`. ``to_state == "ABANDONED"`` means give up; the
    record keeps its error, so the reason survives into summary.json.
    """
    error = error or {}
    code = error.get("code") or "UNKNOWN"
    phase = error.get("phase") or ""
    max_attempts = max(1, int(max_attempts))

    if code in NEVER_RETRY:
        return RetryPlan("ABANDONED", 0.0,
                         "%s is permanent; retrying cannot change it" % code)

    if code in TRANSIENT_CODES:
        # The link misbehaved, not the job. Go back to whichever side of the
        # split we were on: a collect-side drop must not re-dock.
        if transient_attempts + 1 > MAX_TRANSIENT_ATTEMPTS:
            return RetryPlan("ABANDONED", 0.0,
                             "%d transient failures on this job exceeds the cap of %d"
                             % (transient_attempts, MAX_TRANSIENT_ATTEMPTS))
        target = "REMOTE_DONE" if phase == "collect" else "PENDING"
        return RetryPlan(target, _backoff(transient_attempts),
                         "transient %s, retry %d/%d"
                         % (code, transient_attempts + 1, MAX_TRANSIENT_ATTEMPTS),
                         counter="transient")

    known = code in RETRIABLE_CODES
    if not known and not error.get("retriable"):
        return RetryPlan("ABANDONED", 0.0,
                         "%s is not a retriable failure" % code)

    if code in RETRY_TO_REMOTE_DONE:
        # No re-dock: the poses on the remote (or already downloaded) are good.
        limit = 1 if code in ONE_RETRY_ONLY else max_attempts
        if retries_taken >= limit:
            return RetryPlan("ABANDONED", 0.0,
                             "%s again after %d collect retr%s"
                             % (code, retries_taken, "y" if retries_taken == 1 else "ies"))
        return RetryPlan("REMOTE_DONE", _backoff(retries_taken),
                         "%s, re-collecting (%d/%d) without re-docking"
                         % (code, retries_taken + 1, limit))

    # Everything else re-docks, and so is bounded by the docking attempts --
    # which submit_job increments when it launches, not here.
    limit = 2 if code in ONE_RETRY_ONLY else max_attempts
    if attempts >= limit:
        return RetryPlan("ABANDONED", 0.0,
                         "%s after %d attempt%s (limit %d)"
                         % (code, attempts, "" if attempts == 1 else "s", limit))
    return RetryPlan("PENDING", _backoff(attempts),
                     "%s, re-docking (attempt %d/%d)" % (code, attempts + 1, limit))


def _require_queue():
    if gq is None:                                        # pragma: no cover
        raise CollectorError(
            "dock_selectivity.gnina_queue is not importable. The collector is a "
            "driver for that state store and cannot do anything without it; it "
            "will not improvise a second implementation of the queue.")
    return gq


def _state_counts(records):
    """State histogram derived from the records themselves.

    Deliberately not ``queue.counts()``: this function is also used to compute
    the process exit code, and deriving it from the same list the summary is
    built from means the two can never disagree.
    """
    states = list(getattr(gq, "ALL_STATES", ()) or
                  ("PENDING", "SUBMITTING", "SUBMITTED", "RUNNING", "REMOTE_DONE",
                   "COLLECTING", "COLLECTED", "FAILED", "ABANDONED"))
    counts = dict((s, 0) for s in states)
    for rec in records:
        counts[rec.state] = counts.get(rec.state, 0) + 1
    counts["total"] = len(records)
    counts["outstanding"] = (len(records) - counts.get("COLLECTED", 0)
                             - counts.get("ABANDONED", 0))
    return counts


class Collector(object):
    """One pass of the pump, or a loop of them. One per queue, ever (§4).

    Every remote interaction goes through an injected seam -- ``runner`` for the
    consensus_docker subprocess, ``prober`` for the remote state probe, ``clock``
    for time -- so the whole protocol is exercised in-process by the tests. There
    is deliberately no code path here that calls ssh or subprocess directly.
    """

    def __init__(self, run_dir, queue_dir=None, poll=DEFAULT_POLL_SECONDS,
                 collect_workers=DEFAULT_COLLECT_WORKERS,
                 max_attempts=DEFAULT_MAX_ATTEMPTS, job_deadline=None,
                 max_jobs=None, stop_when_drained=True, logger=None,
                 runner=None, prober=None, clock=None, prestage=False):
        queue_mod = _require_queue()
        self.log = logger or LOG
        self.run_dir = os.path.abspath(run_dir)
        self.poll = float(poll)
        self.collect_workers = max(1, int(collect_workers))
        self.max_attempts = max(1, int(max_attempts))
        self.job_deadline = float(job_deadline) if job_deadline else None
        self.max_jobs = int(max_jobs) if max_jobs else None
        self.stop_when_drained = bool(stop_when_drained)
        self.prestage = bool(prestage)
        self.runner = runner
        self.prober = prober

        self.queue = queue_mod.GninaQueue(self.run_dir, queue_dir=queue_dir,
                                          create=True, clock=clock)
        # ONE clock for both, taken from the queue rather than chosen here.
        # ``next_attempt_at`` is written by this process and compared by
        # ``GninaQueue.next_pending``; if the pump measured monotonic seconds
        # (since boot) while the queue measured epoch seconds, every backoff
        # would land permanently in the past. Nothing would raise -- the
        # retries would simply stop being spaced out, and a job would spend all
        # three attempts inside one pass against a remote that is still broken.
        # ``GninaQueue.now`` is the public accessor for exactly this: bound to it,
        # the pump cannot drift into a different clock domain even if the queue's
        # default changes, because there is only one clock and the queue owns it.
        self._clock = self.queue.now
        self.params = self._read_params()

        # The local consensus_docker call is bounded so a wedged local process
        # cannot stall the pump. Killing it does NOT kill the remote job (§ risk
        # 12) -- the work continues and the next pass collects it.
        deadline = self.job_deadline or float(
            self.params.get("job_deadline_seconds") or 1800.0)
        self.submit_timeout = 900.0
        self.collect_timeout = max(600.0, deadline)

        self._lock = None
        self._stop = threading.Event()
        self._stop_why = None
        self._pool_obj = None
        self._in_flight = {}                 # job_id -> Future, owned by workers
        self._last_reconcile = None
        self._passes = 0
        self._submitted = 0
        self._collected = 0
        self._started_at = self._clock()
        self._last_heartbeat = 0.0
        #: Append-only record of what the pump did, in order. Tests assert on it;
        #: in production it is the cheapest possible trace of a delicate protocol.
        self.trace = []

    # ---- construction helpers ------------------------------------------------

    def _read_params(self):
        try:
            params = self.queue.get_params() or {}
        except Exception as exc:
            self.log.warning("could not read %s/params.json: %s",
                             self.queue.root, exc)
            return {}
        if not params:
            self.log.warning(
                "no params.json in %s. batch_dock writes it when it enqueues; "
                "without it this pump can read and reconcile the queue but "
                "cannot build a consensus_docker command, so it will refuse to "
                "submit rather than guess the docking parameters.",
                self.queue.root)
        return params

    def set_params_if_changed(self, updates, allow_change=False):
        """Apply operator overrides (``--job-deadline``) to params.json.

        A campaign that silently changes its docking parameters half way through
        is not one campaign, so this refuses unless explicitly allowed.
        """
        if not updates:
            return self.params
        changed = dict((k, v) for k, v in updates.items()
                       if self.params.get(k) != v)
        if not changed:
            return self.params
        if not allow_change:
            self.log.warning(
                "ignoring %s: params.json already says %s. Pass "
                "--allow-params-change to change the parameters of a campaign "
                "that is already running.",
                changed,
                dict((k, self.params.get(k)) for k in changed))
            return self.params
        merged = dict(self.params)
        merged.update(changed)
        self.params = self.queue.set_params(merged, allow_change=True) or merged
        self.log.warning("params.json changed by operator request: %s", changed)
        return self.params

    # ---- lifecycle -----------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        """Release the pump lock and the collect pool. Idempotent."""
        if self._pool_obj is not None:
            self._pool_obj.shutdown(wait=True)
            self._pool_obj = None
        if self._in_flight:
            # shutdown(wait=True) has already finished them; harvesting is what
            # turns a finished future into a logged, counted outcome. Dropping
            # them here is precisely how a collected job would go unreported.
            try:
                self._harvest(dict((k, 0) for k in ("collected", "failed")),
                              wait=True)
            except Exception as exc:                       # pragma: no cover
                self.log.warning("could not harvest the last collect(s): %s", exc)
        if self._lock is not None:
            try:
                self._lock.release()
            except Exception as exc:                       # pragma: no cover
                self.log.warning("releasing the pump lock failed: %s", exc)
            self._lock = None

    def request_stop(self, why="signal"):
        """Ask the loop to finish its in-flight collect and exit.

        Never touches the remote job: it is detached, its result is still coming,
        and the next pump run collects it for free.
        """
        if not self._stop.is_set():
            self._stop_why = why
            self.log.info("stop requested (%s); finishing the in-flight collect, "
                          "leaving the remote job alone", why)
        self._stop.set()

    @property
    def stopped(self):
        return self._stop.is_set()

    @property
    def stop_reason(self):
        return self._stop_why

    def _ensure_lock(self):
        if self._lock is not None:
            return self._lock
        lock = self.queue.acquire_pump_lock()
        if lock is None:
            raise PumpLocked(
                "another collector already holds %s. Two pumps on one queue "
                "would double-book the single remote GPU, so this one exits "
                "rather than waiting." % os.path.join(self.queue.root, "pump.lock"))
        self._lock = lock
        return lock

    def _pool(self):
        if self._pool_obj is None:
            self._pool_obj = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.collect_workers,
                thread_name_prefix="gnina-collect")
        return self._pool_obj

    # ---- the pass ------------------------------------------------------------

    def run_once(self):
        """One full pass, waiting for the collects it starts. Returns counters.

        Waiting is what makes ``--once`` and the tests deterministic. The daemon
        loop uses the same body without the wait, so a slow local
        post-processing step never delays the next submit to the GPU.
        """
        return self._pass(wait_for_collects=True)

    def _pass(self, wait_for_collects):
        counts = dict((k, 0) for k in
                      ("reconciled", "submitted", "busy", "polled",
                       "collect_started", "collected", "failed", "abandoned",
                       "retried", "prestaged"))
        self._ensure_lock()
        self._passes += 1
        self._check_stop_file()

        if self._reconcile_due():
            counts["reconciled"] = self._reconcile()

        # §4 "keeping the GPU fed": submit, poll, and if that poll freed the GPU
        # submit again in the same pass rather than idling a poll interval.
        for _ in range(MAX_SUBMIT_ROUNDS_PER_PASS):
            if self._stop.is_set():
                break
            self._maybe_submit(counts)
            if not self._poll_active(counts):
                break

        if self.prestage and not self._stop.is_set():
            self._maybe_prestage(counts)

        # A stopping pump finishes what it started and starts nothing new. The
        # remaining REMOTE_DONE jobs are safe on the remote and cost one cheap
        # collect on the next run.
        if not self._stop.is_set():
            counts["collect_started"] = self._schedule_collects()
        self._harvest(counts, wait=wait_for_collects)
        self._apply_retries(counts)
        self._heartbeat()
        return counts

    def run_forever(self):
        """Poll until the queue drains or someone asks us to stop.

        Returns the summary dict (also written to ``summary.json``), so a caller
        that spawned this in-process gets the same picture the barrier reads off
        disk.
        """
        self._ensure_lock()
        self.log.info("gnina collector up: queue=%s poll=%.0fs workers=%d "
                      "max-attempts=%d", self.queue.root, self.poll,
                      self.collect_workers, self.max_attempts)
        try:
            while True:
                self._pass(wait_for_collects=False)
                if self._stop.is_set():
                    break
                if self._drained():
                    if self.stop_when_drained:
                        self._stop_why = "drained"
                        self.log.info("queue drained")
                        break
                elif self._submit_budget_spent() and not self._work_in_progress():
                    self._stop_why = "max-jobs"
                    self.log.info("submitted %d job(s), the --max-jobs budget; "
                                  "leaving the rest PENDING for the next run",
                                  self._submitted)
                    break
                self._sleep(self.poll)
            # Finish what we started. A collect is idempotent, but abandoning one
            # half way costs a whole re-run of the local post-processing.
            if self._in_flight:
                self.log.info("finishing %d in-flight collect(s) before exit",
                              len(self._in_flight))
                self._harvest(dict((k, 0) for k in ("collected", "failed")),
                              wait=True)
            return self._write_summary()
        finally:
            self.close()

    def _sleep(self, seconds):
        # Waiting on the stop event rather than time.sleep means a signal or a
        # STOP file takes effect at once instead of up to a poll interval later.
        if seconds > 0:
            self._stop.wait(seconds)

    def _check_stop_file(self):
        try:
            path = os.path.join(self.queue.root, "STOP")
            if os.path.exists(path):
                if self._passes <= 1 and not self._stop.is_set():
                    # A STOP that predates this pump stops it before it has done
                    # anything, which looks exactly like a pump that is broken.
                    # Obeying it is right -- somebody asked for a stop and this
                    # process cannot know they have changed their mind -- but it
                    # must never be quiet about it. batch_dock leaves the file
                    # behind whenever it did not spawn the collector itself.
                    LOG.warning(
                        "%s exists, so this pump is stopping before collecting "
                        "anything. It is left over from a previous run unless "
                        "you just created it; delete it to start collecting.",
                        path)
                self.request_stop("STOP file")
        except OSError:                                    # pragma: no cover
            pass

    def _drained(self):
        if self._in_flight:
            return False
        for rec in self.queue.read_all():
            if rec.state not in gq.TERMINAL_STATES:
                return False
        # A record file that cannot be parsed is a job too. If reconciliation can
        # still repair it -- which it can whenever the job directory is there --
        # the queue is NOT drained, or the pump would exit 0 on a campaign that
        # quietly dropped a job. One that cannot be repaired is not worth
        # spinning on; it ends the loop but not with a clean exit code, so an
        # operator's `$?` still shows the job was lost.
        for job_id in _repairable_unreadable(self.queue):
            LOG.warning("queue record for %s is unreadable; not drained until "
                        "reconciliation re-derives it", job_id)
            return False
        return True

    def _work_in_progress(self):
        return bool(self._in_flight) or self.queue.active_remote() is not None

    def _submit_budget_spent(self):
        return self.max_jobs is not None and self._submitted >= self.max_jobs

    # ---- reconciliation ------------------------------------------------------

    def _reconcile_due(self):
        if self._in_flight:
            # Rule 3 demotes COLLECTING to REMOTE_DONE when the collect pid is
            # not alive. Our own collect is a subprocess of a worker thread and
            # is briefly pid-less while it starts, so reconciling now could
            # demote a collect that is running fine. Reconciliation exists for
            # crash recovery; it can always wait for our own work to finish.
            return False
        if self._last_reconcile is None:
            return True
        return (self._clock() - self._last_reconcile) >= RECONCILE_INTERVAL_SECONDS

    def _reconcile(self):
        try:
            changes = self.queue.reconcile(prober=self.prober, logger=self.log) or {}
        except Exception as exc:
            self.log.error("reconcile failed: %s", exc)
            self.log.debug("%s", traceback.format_exc())
            return 0
        self._last_reconcile = self._clock()
        for job_id in sorted(changes):
            change = changes[job_id]
            try:
                before, after = change
            except (TypeError, ValueError):                 # pragma: no cover
                before, after = "?", change
            self.log.info("reconciled %s: %s -> %s", job_id, before, after)
            self.trace.append("reconcile:%s->%s" % (job_id, after))
        if changes:
            self.log.info("reconciliation adjusted %d record(s)", len(changes))
        return len(changes)

    # ---- submit --------------------------------------------------------------

    def _maybe_submit(self, counts):
        if self._submit_budget_spent():
            return None
        if self.queue.active_remote() is not None:
            return None                      # the single-GPU invariant, guard (a)
        record = self.queue.next_pending(now=self._clock())
        if record is None:
            return None
        return self._submit(record, counts)

    def _submit(self, record, counts, prestage=False):
        if not self.params:
            raise CollectorError(
                "refusing to submit %s: %s has no params.json, so the "
                "consensus_docker command cannot be built. Re-run batch_dock "
                "with GNINA_ASYNC=1 to enqueue properly."
                % (record.job_id, self.queue.root))
        job_id = record.job_id
        self.log.info("submitting %s%s", job_id, " (prestage)" if prestage else "")
        try:
            record = gq.submit_job(self.queue, record, runner=self.runner,
                                   timeout=self.submit_timeout)
        except Exception as exc:
            # A crash inside submit is itself retriable: nothing was launched
            # that we know of, and the content-addressed remote directory means
            # a repeat either resumes or re-docks, never duplicates.
            self.log.error("submit of %s raised %s: %s", job_id,
                           type(exc).__name__, exc)
            self.log.debug("%s", traceback.format_exc())
            self._force_failed(job_id, "SUBMIT_NO_TICKET",
                               "%s: %s" % (type(exc).__name__, exc), "submit")
            counts["failed"] += 1
            return None

        state = record.state
        self.trace.append("submit:%s->%s" % (job_id, state))
        if state == "PENDING":
            # The remote GPU mutex was held. Not an error and not an attempt:
            # the upload is already paid for and the next pass tries again.
            counts["busy"] += 1
            if prestage:
                counts["prestaged"] += 1
            self.log.debug("%s: GPU busy, staying PENDING", job_id)
            return record
        if state in ("SUBMITTED", "SUBMITTING", "RUNNING", "REMOTE_DONE"):
            self._submitted += 1
            counts["submitted"] += 1
            if state == "REMOTE_DONE":
                self.log.info("%s: a finished remote run already existed; "
                              "collecting instead of re-docking", job_id)
            return record
        if state == "ABANDONED":
            counts["abandoned"] += 1
            self._log_terminal_failure(record)
            return record
        if state == "FAILED":
            counts["failed"] += 1
            self._log_failure(record)
            return record
        return record                                      # pragma: no cover

    def _maybe_prestage(self, counts):
        """Upload the next job's inputs while the GPU is busy with this one.

        Optional, off by default, and it disables itself when it cannot work.
        The spec's optimisation relies on ``submit_job`` uploading *before* it
        tries the remote mutex, so a losing submit still pays the transfer
        early. The queue in this repo instead refuses a second outstanding job
        outright -- a stronger guarantee, and one that makes prestaging a no-op.
        Rather than log a refusal every poll for the whole campaign, the first
        attempt that changes nothing turns the option off with one warning.

        It stays off by default regardless: it is the one path that would put
        two records momentarily in the SUBMITTING/SUBMITTED/RUNNING set, and a
        crash inside that window leaves a job reconciliation must time out.
        """
        if self.queue.active_remote() is None:
            return None
        if self._submit_budget_spent():
            return None
        record = self.queue.next_pending(now=self._clock())
        if record is None:
            return None
        before = getattr(record, "updated_at", None)
        result = self._submit(record, counts, prestage=True)
        after = self.queue.read(record.job_id)
        if (after is not None and after.state == "PENDING"
                and getattr(after, "updated_at", None) == before):
            self.prestage = False
            self.log.warning(
                "--prestage has no effect on this queue: submit_job refuses a "
                "second outstanding job outright instead of uploading and then "
                "finding the GPU mutex held. That is the stronger guarantee, so "
                "prestaging is simply switched off for this run.")
        return result

    # ---- poll ----------------------------------------------------------------

    def _poll_active(self, counts):
        """Poll the one outstanding remote job. True if the GPU is now free."""
        record = self.queue.active_remote()
        if record is None:
            return False
        job_id = record.job_id
        before = record.state
        if before == "SUBMITTING":
            # poll_job will not probe a SUBMITTING record, and rightly so: its
            # remote directory may not exist yet, so a probe could only produce
            # a wrong answer. But submit_job never returns while a record is
            # SUBMITTING and it runs synchronously here, so one seen at this
            # point is the residue of a crashed pump -- and it holds the GPU
            # slot. Only reconciliation can resolve it, so pull the next one
            # forward instead of idling the GPU for the full 300 s interval.
            if self._last_reconcile is not None:
                self.log.info(
                    "%s is stuck in SUBMITTING (a pump died mid-launch) and is "
                    "holding the GPU slot; reconciling on the next pass rather "
                    "than waiting for the interval", job_id)
                self._last_reconcile = None
            return False
        try:
            record = gq.poll_job(self.queue, record, prober=self.prober,
                                 now=self._clock())
        except Exception as exc:
            self.log.error("poll of %s raised %s: %s", job_id,
                           type(exc).__name__, exc)
            self.log.debug("%s", traceback.format_exc())
            # The remote job is detached and unharmed; treat the probe failure
            # as transient so the next pass looks again instead of re-docking.
            self._force_failed(job_id, "SSH_TRANSIENT",
                               "%s: %s" % (type(exc).__name__, exc), "poll")
            counts["failed"] += 1
            return True
        counts["polled"] += 1
        if record.state != before:
            self.trace.append("poll:%s->%s" % (job_id, record.state))
            self.log.info("%s: %s -> %s", job_id, before, record.state)
        if record.state == "FAILED":
            counts["failed"] += 1
            self._log_failure(record)
        return record.state not in ("SUBMITTING", "SUBMITTED", "RUNNING")

    # ---- collect -------------------------------------------------------------

    def _schedule_collects(self):
        free = self.collect_workers - len(self._in_flight)
        if free <= 0:
            return 0
        now = self._clock()
        started = 0
        for record in self.queue.by_state("REMOTE_DONE"):
            if started >= free:
                break
            if record.job_id in self._in_flight:
                continue
            nxt = getattr(record, "next_attempt_at", None)
            if nxt and float(nxt) > now:
                continue                       # backing off after a failed collect
            self.log.info("collecting %s", record.job_id)
            self.trace.append("collect-start:%s" % record.job_id)
            future = self._pool().submit(self._collect_one, record)
            self._in_flight[record.job_id] = future
            started += 1
        return started

    def _collect_one(self, record):
        """Worker-thread body. Owns this record for the duration.

        ``collect_job`` purges ``<job>/gnina/`` before it runs the collect phase
        (spec R2) and verifies the outputs afterwards, so a crash here leaves
        either the previous state or a verified one -- never a half-filled
        directory that looks finished.
        """
        job_id = record.job_id
        try:
            return gq.collect_job(self.queue, record, runner=self.runner,
                                  timeout=self.collect_timeout)
        except BaseException as exc:                       # noqa: BLE001
            name = type(exc).__name__
            timeout_ish = ("timeout" in name.lower()
                           or "timeout" in str(exc).lower())
            code = "COLLECT_TIMEOUT" if timeout_ish else "POSTPROCESS_FAILED"
            self.log.error("collect of %s raised %s: %s", job_id, name, exc)
            self.log.debug("%s", traceback.format_exc())
            # Both codes route the retry back to REMOTE_DONE, never to PENDING:
            # whatever went wrong here, the GPU's poses are not implicated.
            self._force_failed(job_id, code, "%s: %s" % (name, exc), "collect")
            return self.queue.read(job_id)

    def _harvest(self, counts, wait):
        if not self._in_flight:
            return
        if wait:
            concurrent.futures.wait(list(self._in_flight.values()))
        for job_id in list(self._in_flight):
            future = self._in_flight[job_id]
            if not future.done():
                continue
            del self._in_flight[job_id]
            record = None
            try:
                record = future.result()
            except BaseException as exc:                    # pragma: no cover
                self.log.error("collect worker for %s died: %s: %s", job_id,
                               type(exc).__name__, exc)
                self._force_failed(job_id, "POSTPROCESS_FAILED",
                                   "%s: %s" % (type(exc).__name__, exc), "collect")
            if record is None:
                record = self.queue.read(job_id)
            if record is None:                             # pragma: no cover
                self.log.error("collect of %s finished but its record is gone; "
                               "the next reconcile will re-derive it", job_id)
                continue
            self.trace.append("collect:%s->%s" % (job_id, record.state))
            if record.state == "COLLECTED":
                self._collected += 1
                counts["collected"] = counts.get("collected", 0) + 1
                detail = getattr(record, "outputs", None) or {}
                self.log.info("collected %s (%s pose(s))", job_id,
                              detail.get("n_poses", "?"))
            elif record.state == "FAILED":
                counts["failed"] = counts.get("failed", 0) + 1
                self._log_failure(record)
            else:
                self.log.info("%s left in %s after collect", job_id, record.state)

    # ---- retry policy --------------------------------------------------------

    def _apply_retries(self, counts):
        now = self._clock()
        for record in self.queue.by_state("FAILED"):
            if record.job_id in self._in_flight:
                continue
            nxt = getattr(record, "next_attempt_at", None)
            if nxt and float(nxt) > now:
                continue
            plan = plan_retry(record.error,
                              int(getattr(record, "attempts", 0) or 0),
                              int(getattr(record, "transient_attempts", 0) or 0),
                              postprocess_retries(record),
                              self.max_attempts)
            fields = {}
            if plan.gives_up:
                self.log.error("ABANDONING %s: %s", record.job_id, plan.why)
            else:
                fields["next_attempt_at"] = now + plan.delay
                if plan.counter == "transient":
                    fields["transient_attempts"] = int(
                        getattr(record, "transient_attempts", 0) or 0) + 1
                self.log.warning("retrying %s: %s", record.job_id, plan.why)
            try:
                self.queue.transition(record.job_id, plan.to_state,
                                      why=plan.why, **fields)
            except Exception as exc:
                self.log.error("could not move %s to %s (%s); leaving it FAILED "
                               "so it is visible rather than lost",
                               record.job_id, plan.to_state, exc)
                continue
            self.trace.append("retry:%s->%s" % (record.job_id, plan.to_state))
            if plan.gives_up:
                counts["abandoned"] += 1
            else:
                counts["retried"] += 1

    # ---- failure recording ---------------------------------------------------

    def _force_failed(self, job_id, code, message, phase, retriable=None):
        """Record a crash against the job, walking only legal transitions.

        A job that cannot be run must end up visible in its record and in
        summary.json. Swallowing the exception here is how jobs vanish.
        """
        record = self.queue.read(job_id)
        if record is None:
            self.log.error("cannot record %s against %s: its record is "
                           "unreadable; reconciliation will re-derive it",
                           code, job_id)
            return None
        if record.state in gq.TERMINAL_STATES:
            self.log.warning("%s already %s; not recording %s",
                             job_id, record.state, code)
            return record
        if record.state == "FAILED":
            self.log.warning("%s is already FAILED; keeping the first error and "
                             "discarding %s: %s", job_id, code, message[:200])
            return record
        error = {
            "code": code,
            "message": str(message)[:2000],
            "phase": phase,
            "retriable": (bool(retriable) if retriable is not None
                          else code in RETRIABLE_CODES),
            "at": _utcnow_iso(),
        }
        for hop in _HOPS_TO_FAILED.get(record.state, ()):
            try:
                record = self.queue.transition(
                    job_id, hop, why="recording a %s failure" % code)
            except Exception as exc:
                self.log.error("could not step %s to %s before failing it: %s",
                               job_id, hop, exc)
                return record
        try:
            record = self.queue.transition(
                job_id, "FAILED", why="%s: %s" % (code, str(message)[:200]),
                error=error)
        except Exception as exc:
            self.log.error("could not mark %s FAILED (%s). The record still "
                           "says %s and reconciliation will pick it up.",
                           job_id, exc, record.state)
            return record
        self.trace.append("fail:%s:%s" % (job_id, code))
        self._log_failure(record)
        return record

    def _log_failure(self, record):
        error = getattr(record, "error", None) or {}
        self.log.warning("%s FAILED [%s] during %s: %s", record.job_id,
                         error.get("code", "UNKNOWN"), error.get("phase", "?"),
                         str(error.get("message", ""))[:400])

    def _log_terminal_failure(self, record):
        error = getattr(record, "error", None) or {}
        self.log.error("%s ABANDONED [%s]: %s", record.job_id,
                       error.get("code", "UNKNOWN"),
                       str(error.get("message", ""))[:400])

    # ---- reporting -----------------------------------------------------------

    def _heartbeat(self):
        now = self._clock()
        if now - self._last_heartbeat < 60.0:
            return
        self._last_heartbeat = now
        counts = _state_counts(self.queue.read_all())
        active = self.queue.active_remote()
        self.log.info(
            "GNINA PUMP: %d total | %d collected, %d outstanding, %d failed, "
            "%d abandoned | GPU: %s",
            counts["total"], counts.get("COLLECTED", 0), counts["outstanding"],
            counts.get("FAILED", 0), counts.get("ABANDONED", 0),
            active.job_id if active is not None else "idle")

    def summary_extra(self):
        records = self.queue.read_all()
        counts = _state_counts(records)
        failed = [r.job_id for r in records if r.state == "FAILED"]
        abandoned = [(r.job_id, (r.error or {}).get("code", "UNKNOWN"))
                     for r in records if r.state == "ABANDONED"]
        return {
            "collector": {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "argv": list(sys.argv),
                "finished_at": _utcnow_iso(),
                "elapsed_seconds": round(self._clock() - self._started_at, 3),
                "passes": self._passes,
                "submitted": self._submitted,
                "collected": self._collected,
                "poll_seconds": self.poll,
                "collect_workers": self.collect_workers,
                "max_attempts": self.max_attempts,
                "stopped": self._stop.is_set(),
                "stop_reason": self._stop_why,
            },
            # Named, not merely counted: a partial campaign that is analysed as
            # if it were complete is the failure mode this list exists to stop.
            "failed_job_ids": sorted(failed)[:200],
            "abandoned_jobs": [{"job_id": j, "code": c}
                               for j, c in sorted(abandoned)][:200],
            "state_counts": counts,
        }

    def _write_summary(self):
        extra = self.summary_extra()
        summary = {}
        try:
            summary = dict(self.queue.summary(extra) or {})
        except Exception as exc:                           # pragma: no cover
            self.log.warning("could not build the queue summary: %s", exc)
            summary = dict(extra)
        try:
            summary["summary_path"] = self.queue.write_summary(extra)
        except Exception as exc:                           # pragma: no cover
            self.log.warning("could not write summary.json: %s", exc)
        counts = extra["state_counts"]
        if counts.get("ABANDONED"):
            self.log.error(
                "%d job(s) ABANDONED and will have no gnina result. They are "
                "named in %s; stage 5 will exclude them and re-running "
                "batch_dock resumes them cheaply.",
                counts["ABANDONED"], summary.get("summary_path", "summary.json"))
        return summary


# --- CLI ---------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        prog="gnina_collect.py",
        description="Pump remote gnina jobs: submit one at a time, collect what "
                    "finishes, file it where a local run would have.")
    parser.add_argument("--run-dir", required=True,
                        help="batch_dock run directory (job dirs live in "
                             "<run-dir>/docking_results/)")
    parser.add_argument("--queue-dir", default=None,
                        help="queue root; default <run-dir>/gnina_queue or "
                             "$GNINA_QUEUE_DIR")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true",
                      help="one pass: submit if the GPU is free, collect what is "
                           "ready, exit")
    mode.add_argument("--daemon", action="store_true",
                      help="poll until the queue drains or a stop is requested "
                           "(the default)")
    parser.add_argument("--poll", type=float, default=DEFAULT_POLL_SECONDS,
                        help="seconds between passes (default %(default)s)")
    parser.add_argument("--collect-workers", type=int,
                        default=DEFAULT_COLLECT_WORKERS,
                        help="parallel local post-processing jobs "
                             "(default %(default)s)")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                        help="docking attempts per job (default %(default)s)")
    parser.add_argument("--job-deadline", type=float, default=None,
                        help="per-job remote deadline in seconds; only written to "
                             "params.json with --allow-params-change")
    parser.add_argument("--max-jobs", type=int, default=None,
                        help="stop after submitting this many jobs")
    parser.add_argument("--caps-file", default=None,
                        help="validate against a saved caps.json instead of "
                             "probing the remote host")
    parser.add_argument("--prestage", action="store_true",
                        help="upload the next job's inputs while the GPU is busy "
                             "(experimental; see the module docstring)")
    parser.add_argument("--reconcile-only", action="store_true",
                        help="re-derive every record from disk and the remote, "
                             "then exit")
    parser.add_argument("--status", action="store_true",
                        help="print the queue state and exit; takes no lock")
    parser.add_argument("--stop", action="store_true",
                        help="ask a running collector to stop, then exit")
    parser.add_argument("--requeue", default=None, metavar="JOB_ID",
                        help="reset one ABANDONED job to PENDING")
    parser.add_argument("--allow-params-change", action="store_true",
                        help="permit --job-deadline to rewrite params.json")
    parser.add_argument("--strict-caps", action="store_true",
                        help="treat capability warnings as fatal")
    parser.add_argument("--reconcile-interval", type=float,
                        default=RECONCILE_INTERVAL_SECONDS,
                        help="seconds between full re-derivations of the queue "
                             "state from disk and the remote (default %(default)s)")
    parser.add_argument("--max-transient-attempts", type=int,
                        default=MAX_TRANSIENT_ATTEMPTS,
                        help="SSH-level failures one job may survive before it is "
                             "abandoned; capped separately from docking attempts "
                             "so a flaky link does not burn them (default %(default)s)")
    add_runtime_arguments(parser, STEP_NAME)
    # Same dest as --log-file, so it is one setting with two spellings rather
    # than two settings that almost agree: batch_dock has always spawned the
    # collector with --log <queue_root>/collector.log and that command line
    # lives in running campaigns and in operators' shell history.
    parser.add_argument("--log", dest="log_file", default=None,
                        help="compatibility spelling of --log-file")
    return parser


def _open_queue(args, create):
    queue_mod = _require_queue()
    root = queue_mod.queue_root(os.path.abspath(args.run_dir), args.queue_dir)
    if not create and not os.path.isdir(root):
        raise CollectorError(
            "no gnina queue at %s. batch_dock creates it when it enqueues the "
            "gnina stage (GNINA_ASYNC=1); there is nothing here to collect."
            % root)
    return queue_mod.GninaQueue(os.path.abspath(args.run_dir),
                                queue_dir=args.queue_dir, create=create)


def cmd_stop(args):
    """Ask a running pump to stop. A file, not a signal: it works across users,
    survives a lost pid, and is the same mechanism whether the pump is local or
    was started by batch_dock in another process group."""
    queue = _open_queue(args, create=False)
    path = os.path.join(queue.root, "STOP")
    with open(path, "w") as handle:
        handle.write(json.dumps({"at": _utcnow_iso(), "by": os.getpid()}) + "\n")
    print("stop requested: %s" % path)
    return EXIT_OK


def cmd_status(args):
    """Read-only. Takes no lock, so it is safe while the pump is running."""
    queue = _open_queue(args, create=False)
    records = queue.read_all()
    counts = _state_counts(records)
    print("queue: %s" % queue.root)
    print("jobs : %d" % counts["total"])
    for state in (getattr(gq, "ALL_STATES", ()) or ()):
        if counts.get(state):
            print("  %-12s %d" % (state, counts[state]))
    print("outstanding: %d" % counts["outstanding"])
    try:
        free = queue.pump_lock_is_free()
        print("pump lock  : %s" % ("free" if free else "held"))
    except Exception:                                      # pragma: no cover
        pass
    active = queue.active_remote()
    if active is not None:
        remote = getattr(active, "remote", None) or {}
        print("on the GPU : %s (%s)" % (active.job_id, remote.get("dir", "?")))
    for record in records:
        if record.state == "ABANDONED":
            error = record.error or {}
            print("ABANDONED %s [%s] %s" % (record.job_id,
                                            error.get("code", "UNKNOWN"),
                                            str(error.get("message", ""))[:160]))
        elif record.state == "FAILED":
            error = record.error or {}
            print("FAILED    %s [%s] attempt %s" % (
                record.job_id, error.get("code", "UNKNOWN"),
                getattr(record, "attempts", "?")))
    return EXIT_OK


def cmd_requeue(args):
    """Operator reset of a terminal failure.

    ABANDONED has no outgoing edge in the state machine on purpose, so this
    rewrites the record rather than transitioning it -- and only for a record
    that is ABANDONED, which is precisely the set of records no running pump
    ever writes. That is what makes it safe without taking the pump lock.
    """
    queue = _open_queue(args, create=False)
    job_id = args.requeue
    record = queue.read(job_id)
    if record is None:
        print("no such job in %s: %s" % (queue.jobs_dir, job_id), file=sys.stderr)
        return EXIT_USAGE
    if record.state != "ABANDONED":
        print("%s is %s, not ABANDONED. Only a terminal failure needs an "
              "operator to reset it; a live job is the pump's business."
              % (job_id, record.state), file=sys.stderr)
        return EXIT_USAGE
    data = record.to_dict()
    history = list(data.get("history") or [])
    history.append({"at": _utcnow_iso(), "from": "ABANDONED", "to": "PENDING",
                    "why": "operator requeue (gnina_collect.py --requeue)"})
    data["history"] = history[-50:]
    data["state"] = "PENDING"
    data["attempts"] = 0
    data["transient_attempts"] = 0
    data["next_attempt_at"] = None
    data["error"] = None
    data["updated_at"] = _utcnow_iso()
    gq.atomic_write_json(os.path.join(queue.jobs_dir, "%s.json" % job_id), data)
    print("requeued %s (was ABANDONED [%s])"
          % (job_id, (record.error or {}).get("code", "UNKNOWN")))
    return EXIT_OK


def _check_caps(args, params, logger):
    """Validate a saved caps.json against the campaign's frozen parameters.

    Never probes: the point of ``--caps-file`` is that a collector started by
    hand on a busy campaign does not open a second connection to the one GPU box.
    """
    try:
        from dock_selectivity import remote_caps
    except ImportError as exc:
        raise CollectorError(
            "--caps-file needs dock_selectivity.remote_caps, which is not "
            "importable (%s). Drop the flag to run without the check." % exc)
    caps = remote_caps.load_caps(args.caps_file)
    request = remote_caps.DockRequest(
        cpu=int(params.get("cpu") or 0),
        exhaustiveness=int(params.get("exhaustiveness") or 0),
        num_modes=params.get("num_modes"),
        job_deadline_seconds=float(params.get("job_deadline_seconds") or 1800.0),
        image=params.get("image") or remote_caps.DEFAULT_IMAGE,
        native=params.get("native"),
        gnina_path=params.get("gnina_path") or "gnina_remote.py",
        async_mode=True,
    )
    report = remote_caps.validate_request(caps, request, strict=args.strict_caps)
    return remote_caps.enforce(report, logger=logger, strict=args.strict_caps)


def _repairable_unreadable(queue):
    """Unreadable records whose job directory still exists, so reconcile can fix them.

    The distinction matters to the pump's stopping condition: a record it can
    recover is outstanding work, while one whose job directory has gone is
    unrecoverable and must not become an infinite loop.
    """
    try:
        job_ids = queue.unreadable_job_ids()
    except Exception:                                     # noqa: BLE001
        return []
    root = os.path.join(queue.run_dir,
                        getattr(gq, "DOCKING_RESULTS_DIRNAME", "docking_results"))
    return [j for j in job_ids if os.path.isdir(os.path.join(root, j))]


def _exit_code(queue):
    """0 drained and clean, 3 drained with abandoned jobs, 4 work left over."""
    counts = _state_counts(queue.read_all())
    if counts["outstanding"] > 0:
        return EXIT_STOPPED
    # A record nobody could parse is a job nobody collected. Reporting success
    # for it would tell an operator the campaign is complete when a job is
    # missing from it -- the one thing an exit code must never do here.
    try:
        lost = queue.unreadable_job_ids()
    except Exception:                                     # noqa: BLE001
        lost = []
    if lost:
        LOG.error("%d queue record(s) could not be read and their jobs are "
                  "unaccounted for: %s. Re-run batch_dock to re-derive them.",
                  len(lost), ", ".join(lost[:20]))
        return EXIT_STOPPED
    if counts.get("ABANDONED", 0) > 0:
        return EXIT_ABANDONED
    return EXIT_OK


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # --status / --stop / --requeue are questions about a pump, not a pump run.
    # A question must not create a step directory or a log file, for the same
    # reason batch_dock's --print-config does not: a query with side effects is
    # not a query. They print to stdout and take no lock.
    if args.stop or args.requeue or args.status:
        try:
            if gq is None:
                _require_queue()
            if args.stop:
                return cmd_stop(args)
            if args.requeue:
                return cmd_requeue(args)
            return cmd_status(args)
        except CollectorError as exc:
            LOG.error("%s", exc)
            print("gnina_collect: %s" % exc, file=sys.stderr)
            return EXIT_USAGE

    resolve_out_dir(args, STEP_NAME, os.path.abspath(args.run_dir))
    with step(STEP_NAME, args) as ctx:
        code = _pump(args, ctx)
        if code not in (EXIT_OK,):
            ctx.notify("failure",
                       "gnina_collect: exit {0} (0 drained clean, 3 abandoned "
                       "jobs, 4 work left, 5 locked, 78 caps)".format(code),
                       priority=1)
        return code


def _pump(args, ctx):
    """The pump itself, inside the step's logging and notifications."""
    logger = ctx.logger
    # Both are read as module globals from inside the retry policy and the
    # reconcile clock, so the resolved values are written back over them once,
    # here, rather than threaded through every call site.
    globals()["RECONCILE_INTERVAL_SECONDS"] = float(args.reconcile_interval)
    globals()["MAX_TRANSIENT_ATTEMPTS"] = int(args.max_transient_attempts)

    try:
        if gq is None:
            _require_queue()

        # Refuse rather than create. GninaQueue(create=True) would happily make
        # an empty queue under a mistyped --run-dir, and the pump would then
        # report a clean, instantaneous, entirely fictional drain.
        root = gq.queue_root(os.path.abspath(args.run_dir), args.queue_dir)
        if not os.path.isdir(root):
            raise CollectorError(
                "no gnina queue at %s. batch_dock creates it when it enqueues "
                "the gnina stage (GNINA_ASYNC=1); there is nothing here to "
                "collect, and creating an empty queue would look like success."
                % root)

        collector = Collector(
            args.run_dir, queue_dir=args.queue_dir, poll=args.poll,
            collect_workers=args.collect_workers, max_attempts=args.max_attempts,
            job_deadline=args.job_deadline, max_jobs=args.max_jobs,
            stop_when_drained=True, logger=logger, prestage=args.prestage)
    except PumpLocked as exc:
        logger.error("%s", exc)
        return EXIT_LOCKED
    except CollectorError as exc:
        logger.error("%s", exc)
        return EXIT_USAGE
    except KeyboardInterrupt:                              # pragma: no cover
        return EXIT_STOPPED
    except Exception as exc:                               # noqa: BLE001
        logger.critical("could not start the collector: %s: %s",
                        type(exc).__name__, exc)
        logger.critical("%s", traceback.format_exc())
        return EXIT_UNEXPECTED

    try:
        if args.job_deadline:
            collector.set_params_if_changed(
                {"job_deadline_seconds": float(args.job_deadline)},
                allow_change=args.allow_params_change)
        if args.caps_file:
            code = _check_caps(args, collector.params, logger)
            if code:
                logger.critical("remote capability check is fatal; not starting "
                                "the pump")
                return EXIT_CAPS_FATAL

        _install_signal_handlers(collector, logger)

        if args.reconcile_only:
            collector._ensure_lock()
            changed = collector._reconcile()
            summary = collector._write_summary()
            logger.info("reconcile-only: %d record(s) adjusted, summary at %s",
                        changed, summary.get("summary_path", "?"))
            return _exit_code(collector.queue)

        if args.once:
            collector.run_once()
            collector._write_summary()
        else:
            collector.run_forever()
        return _exit_code(collector.queue)
    except PumpLocked as exc:
        logger.error("%s", exc)
        return EXIT_LOCKED
    except CollectorError as exc:
        logger.error("%s", exc)
        return EXIT_USAGE
    except KeyboardInterrupt:
        logger.warning("interrupted; the remote job keeps running and the next "
                       "collector run picks it up")
        return EXIT_STOPPED
    except Exception as exc:                               # pragma: no cover
        logger.critical("unexpected collector failure: %s: %s",
                        type(exc).__name__, exc)
        logger.critical("%s", traceback.format_exc())
        return EXIT_UNEXPECTED
    finally:
        collector.close()


def _install_signal_handlers(collector, logger):
    """Only in ``main``: installing handlers is a main-thread-only operation and
    a library that does it behind the caller's back is a bug."""
    def handler(signum, _frame):
        collector.request_stop("signal %d" % signum)

    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:                                    # pragma: no cover
            continue
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError) as exc:               # pragma: no cover
            logger.debug("cannot install a %s handler (%s); the STOP file still "
                         "works", name, exc)


if __name__ == "__main__":
    sys.exit(main())
