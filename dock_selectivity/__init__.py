"""dock_selectivity -- shared library for the consensus-docking pipeline.

The package sits beside the pipeline scripts, so ``python $REPO/script.py``
imports it via ``sys.path[0]`` with no install step.

Its reason to exist is one rule (``FIX_PLAN.md`` section 1): **a protein is a
UniProt accession, resolved once, from an Entrez Gene ID.**  A gene symbol, an
entry name or any ``*_name`` column is a label, never a key.  The 2025-06-29
campaign resolved receptors through a last-wins ``dict(zip(symbol, accession))``
and therefore docked the wrong protein for 45.7% of its job directories.

Modules:

* :mod:`dock_selectivity.identity` -- ``Protein``, ``Resolution``,
  ``IdentityIndex``: the resolver and its asserted cardinality invariants.
* :mod:`dock_selectivity.jobid` -- the job-directory naming codec and
  ``job_identity.json``, which replaces the folder name as the carrier of
  identity.
* :mod:`dock_selectivity.runtime` -- the shared step runtime every script's
  ``main()`` uses: the common ``--out-dir/--log-*/--notify*`` flags, logging
  configured *after* parsing into ``<run-dir>/steps/<step>/<step>.log``, and
  Pushover notifications that can fail without failing the run.
* :mod:`dock_selectivity.colab_backend` -- the second GPU backend for
  ``gnina_remote.py``: gnina on a Google Colab runtime, for when the SSH host's
  one GPU is in use. Selected with ``GNINA_REMOTE_BACKEND=colab``.

Submodules are imported lazily (PEP 562) so that importing the package costs
nothing and never depends on optional third-party libraries.
"""

from __future__ import annotations

from typing import Any, List

__version__ = "0.1.0"

__all__ = [
    "identity",
    "jobid",
    "IdentityIndex",
    "InvariantViolation",
    "Protein",
    "Resolution",
    "JobIdentity",
    "JobIdError",
    "JobIdParseError",
    "format_job_dir",
    "parse_job_dir",
    "parse_legacy_job_dir",
    "read_job_identity",
    "write_job_identity",
]

_LAZY = {
    "IdentityIndex": "identity",
    "InvariantViolation": "identity",
    "Protein": "identity",
    "Resolution": "identity",
    "JobIdentity": "jobid",
    "JobIdError": "jobid",
    "JobIdParseError": "jobid",
    "format_job_dir": "jobid",
    "parse_job_dir": "jobid",
    "parse_legacy_job_dir": "jobid",
    "read_job_identity": "jobid",
    "write_job_identity": "jobid",
}


def __getattr__(name: str) -> Any:
    """Resolve public names on first use (PEP 562)."""
    import importlib

    if name in ("identity", "jobid"):
        return importlib.import_module("." + name, __name__)
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(
            "module {0!r} has no attribute {1!r}".format(__name__, name)
        )
    return getattr(importlib.import_module("." + module_name, __name__), name)


def __dir__() -> List[str]:
    return sorted(__all__)
