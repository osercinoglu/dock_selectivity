"""Shared helpers for the test suite. Not a test module (discovery matches test*.py).

``run_in_temp_cwd`` exists because a step with no ``--run-dir`` now creates
``./<step-name>/`` for its log -- correct behaviour, but a test that calls such a
``main()`` while standing in the repository leaves a directory behind in the
working tree. Tests should not write into the checkout at all, so the three
modules that exercise the standalone path chdir into a temporary directory for
their duration.
"""

from __future__ import annotations

import os
import tempfile
from typing import Callable, Tuple


def run_in_temp_cwd() -> Tuple[Callable[[], None], Callable[[], None]]:
    """Return ``(setUpModule, tearDownModule)`` that chdir into a temp directory.

    Fixtures in these modules build their own absolute ``TemporaryDirectory``
    paths, so moving the process cwd changes nothing except where a step with no
    ``--run-dir`` puts its own folder.
    """
    state = {}

    def set_up() -> None:
        state["cwd"] = os.getcwd()
        state["tmp"] = tempfile.TemporaryDirectory()
        os.chdir(state["tmp"].name)

    def tear_down() -> None:
        os.chdir(state.pop("cwd", os.getcwd()))
        tmp = state.pop("tmp", None)
        if tmp is not None:
            tmp.cleanup()

    return set_up, tear_down
