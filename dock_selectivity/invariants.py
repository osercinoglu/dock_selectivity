"""Named, reusable invariant checks for the dock_selectivity rebuild.

Every check in this module exists because its absence caused a concrete,
measured failure in the 2025-06-29 campaign (see
``ASSESSMENT_blast_radius_2026_08_12.md`` §3 and ``KNOWN_ISSUE_uniprot_gene_mapping.md``):

===============================  ==============================================
Check                            What it would have caught, and when
===============================  ==============================================
``assert_collapse_factor_one``   ``dict(zip(Mapped_Gene_Name, Entry))`` at
                                 ``batch_dock.py:2034`` — 4,585 accessions
                                 collapsed onto 1,795 symbols (mean 2.55,
                                 max 9), last-duplicate-wins, and the surviving
                                 accession *became the docked receptor*.
``assert_self_consistent_labels``2,647/4,585 rows (57.7%) whose label named a
                                 different protein than the row's own
                                 accession — e.g. the label ``AVPR2`` sitting
                                 on ``Q8NCT1`` (which is ``ARRDC4``).
``assert_total_accounting``      237 symbol-less edges, 222 slug drug rows and
                                 163 genes that were dropped with no ledger.
``assert_no_row_growth``         the ``parse_prepare.py:1171`` merge whose key
                                 omitted ``Fragment_ID`` and silently
                                 duplicated 410,087 rows across 179 jobs.
``assert_deterministic``         ``prepare_uniprot_mapping.py:51``
                                 ``list(set(...))`` feeding a first-wins
                                 ``drop_duplicates`` — a non-reproducible
                                 identity map.
``assert_unique_key``            the 1,428 duplicate merge keys that made the
                                 fan-out possible in the first place.
===============================  ==============================================

Design rules followed here, all learned from the incident:

1. **Every check returns a JSON-serialisable report dict** *and* raises
   :class:`InvariantViolation` on failure. Stages collect the reports and write
   them next to their output with :func:`write_invariant_report`, so a passing
   run leaves evidence that it passed, not just an absence of errors.
2. **Every failure message is actionable**: it names the check, states the
   measured numbers, and shows the worst offending keys.
3. **No check consumes the object it is validating as its own ground truth.**
   ``validate_negatives()`` re-checked negatives against the same corrupt
   ``known_pairs`` set that produced them and reported "All validations
   passed". :func:`assert_self_consistent_labels` therefore takes an
   independent ``truth_lookup``.
4. **A collapsed container cannot be audited.** A built ``dict`` has already
   lost its duplicates, so :func:`cardinality_report` refuses one and tells the
   caller to pass the pairs instead.

``InvariantViolation`` is defined here and this module is its canonical home;
``identity.py``, ``joins.py`` and ``schemas.py`` import it from here.

This module imports pandas but never touches the filesystem except in
:func:`write_invariant_report` (and :func:`assert_deterministic_across_hash_seeds`,
which spawns subprocesses of the current interpreter).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd

__all__ = [
    "InvariantViolation",
    "cardinality_report",
    "assert_collapse_factor_one",
    "assert_total_accounting",
    "assert_no_row_growth",
    "assert_deterministic",
    "assert_deterministic_across_hash_seeds",
    "assert_self_consistent_labels",
    "assert_unique_key",
    "deterministic_fingerprint",
    "write_invariant_report",
]

logger = logging.getLogger(__name__)

class _NullSentinel:
    """Stand-in for a missing key/value.

    Nulls must group together deterministically: two ``float('nan')`` objects
    are neither equal nor necessarily the same object, so grouping on raw
    values would scatter them. A dedicated singleton also cannot collide with
    real data the way a magic string could.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "<NULL>"


#: Singleton used in place of a missing/NaN key or value.
_NULL = _NullSentinel()

#: Default number of offending examples embedded in a report/message.
MAX_EXAMPLES = 5


class InvariantViolation(Exception):
    """Raised when a named invariant does not hold.

    Parameters
    ----------
    message:
        Actionable description: what was measured, what was expected, and what
        to do about it.
    name:
        Name of the check site (e.g. ``"annotate_sample_types merge"``). It is
        prefixed onto ``str(exc)`` so a traceback identifies the stage.
    report:
        The check's report dict, attached for programmatic inspection and for
        writing into ``<stage>.invariants.json``.
    """

    def __init__(
        self,
        message: str,
        name: Optional[str] = None,
        report: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.report = report if report is not None else {}
        self.message = message
        super().__init__(message if not name else "[{0}] {1}".format(name, message))


# --------------------------------------------------------------------------
# internal helpers
# --------------------------------------------------------------------------


def _is_null(value: Any) -> bool:
    """``pd.isna`` that never raises on non-scalars."""
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, bool):
        return result
    return False


def _norm(value: Any) -> Any:
    """Normalise a key/value for grouping: nulls collapse to one sentinel."""
    if _is_null(value):
        return _NULL
    return value


def _sort_key(value: Any) -> Tuple[str, str]:
    """Total, type-safe ordering used to make report examples deterministic."""
    return (type(value).__name__, repr(value))


def _fail(
    message: str,
    name: Optional[str],
    report: Dict[str, Any],
    warn_only: bool,
) -> Dict[str, Any]:
    """Raise or warn, depending on ``warn_only``; always return the report."""
    report["ok"] = False
    if warn_only:
        logger.warning("INVARIANT (warn-only) [%s] %s", name, message)
        return report
    raise InvariantViolation(message, name=name, report=report)


#: Internal column names for the normalised (key, value) frame.
_K = "__key__"
_V = "__value__"


def _as_pair_frame(
    mapping: Any,
    key: Optional[str] = None,
    value: Optional[str] = None,
) -> pd.DataFrame:
    """Normalise many input shapes into a 2-column ``(key, value)`` frame.

    Accepted:

    * ``pandas.DataFrame`` — ``key``/``value`` column names, or exactly two
      columns.
    * ``pandas.Series`` — index supplies the keys, values the values.
    * ``Mapping[key, Collection[value]]`` — the *grouped* form, e.g.
      ``{'TUBA1B': ['P68363', 'Q13509']}``.
    * any iterable of 2-tuples.

    Deliberately **rejected**: a ``Mapping`` whose values are scalars. That is
    a map that has *already* collapsed — ``dict(zip(symbols, accessions))``
    keeps only the last accession per symbol, so auditing it can never reveal
    the loss. This refusal is the point: it forces the caller to check the
    pairs *before* building the dict, which is exactly where the 2025 bug hid.
    """
    if isinstance(mapping, pd.DataFrame):
        if key is not None or value is not None:
            if key is None or value is None:
                raise ValueError(
                    "pass both key= and value= column names, or neither"
                )
            missing = [c for c in (key, value) if c not in mapping.columns]
            if missing:
                raise ValueError(
                    "column(s) {0} not in frame; available: {1}".format(
                        missing, list(mapping.columns)
                    )
                )
            key_col, value_col = key, value
        elif mapping.shape[1] == 2:
            key_col, value_col = list(mapping.columns)
        else:
            raise ValueError(
                "DataFrame has {0} columns; pass key=<col> and value=<col> to "
                "say which is which".format(mapping.shape[1])
            )
        # reset_index rather than to_numpy: a nullable Int64 column must not be
        # cast to float, or Entrez 554 is reported as '554.0'.
        return pd.DataFrame(
            {
                _K: mapping[key_col].reset_index(drop=True),
                _V: mapping[value_col].reset_index(drop=True),
            }
        )

    if isinstance(mapping, pd.Series):
        return pd.DataFrame(
            {
                _K: pd.Series(mapping.index),
                _V: mapping.reset_index(drop=True),
            }
        )

    if isinstance(mapping, _Mapping):
        pairs: List[Tuple[Any, Any]] = []
        scalar_keys: List[Any] = []
        for k, v in mapping.items():
            if isinstance(v, (str, bytes)) or not isinstance(v, _Iterable):
                scalar_keys.append(k)
                continue
            for item in v:
                pairs.append((k, item))
        if scalar_keys:
            raise ValueError(
                "refusing to audit a built mapping with scalar values: a dict "
                "has already collapsed its duplicates (dict(zip(k, v)) keeps "
                "the LAST value per key), so a collapse factor computed from "
                "it is always 1 and always meaningless. Pass the (key, value) "
                "pairs, a 2-column DataFrame, or a key -> list-of-values map "
                "instead. Offending key(s): {0}".format(
                    sorted(scalar_keys, key=_sort_key)[:MAX_EXAMPLES]
                )
            )
        return pd.DataFrame(pairs, columns=[_K, _V])

    if isinstance(mapping, _Iterable):
        pairs = []
        for i, item in enumerate(mapping):
            if isinstance(item, (tuple, list)) and len(item) == 2:
                pairs.append((item[0], item[1]))
            else:
                raise ValueError(
                    "element {0} is not a (key, value) pair: {1!r}".format(i, item)
                )
        return pd.DataFrame(pairs, columns=[_K, _V])

    raise TypeError(
        "unsupported mapping type {0}; expected DataFrame, Series, "
        "key -> collection Mapping, or an iterable of (key, value) "
        "pairs".format(type(mapping).__name__)
    )


# --------------------------------------------------------------------------
# I2 — cardinality / collapse
# --------------------------------------------------------------------------


def cardinality_report(
    mapping: Any,
    name: Optional[str] = None,
    key: Optional[str] = None,
    value: Optional[str] = None,
    max_examples: int = MAX_EXAMPLES,
) -> Dict[str, Any]:
    """Describe the shape of a key -> value mapping without asserting anything.

    ``max_collapse`` is the headline number: the largest number of *distinct*
    values claimed by a single key. ``max_collapse == 1`` means the mapping is
    a function and ``dict(...)`` over it is lossless. The 2025 identity map had
    ``mean_collapse == 2.55`` and ``max_collapse == 9``.

    See :func:`_as_pair_frame` for accepted input shapes (a built ``dict`` with
    scalar values is refused on purpose).

    Vectorised: ``HUMAN_9606_idmapping.dat`` is millions of rows, so nothing
    here iterates the input in Python.
    """
    frame = _as_pair_frame(mapping, key=key, value=value)
    n_pairs = int(len(frame))
    base = {
        "check": "cardinality_report",
        "name": name,
        "n_pairs": n_pairs,
        "n_keys": 0,
        "n_distinct_values": 0,
        "n_null_keys": 0,
        "n_null_values": 0,
        "max_collapse": 0,
        "mean_collapse": 0.0,
        "n_keys_collapsed": 0,
        "worst_keys": [],
        "ok": True,
    }
    if n_pairs == 0:
        return base

    n_null_keys = int(frame[_K].isna().sum())
    n_null_values = int(frame[_V].isna().sum())
    # drop_duplicates treats NaN == NaN, which is the grouping we want: all
    # missing values are one value, not one per row.
    distinct_pairs = frame.drop_duplicates()
    per_key = distinct_pairs.groupby(_K, dropna=False, sort=False).size()
    n_keys = int(len(per_key))
    max_collapse = int(per_key.max())
    mean_collapse = float(per_key.sum()) / n_keys
    collapsed = per_key[per_key > 1]

    worst_frame = collapsed.rename("__n__").reset_index()
    try:
        worst_frame = worst_frame.sort_values(
            by=["__n__", _K], ascending=[False, True], kind="mergesort"
        )
    except TypeError:  # pragma: no cover - mixed key types
        worst_frame = worst_frame.sort_values(
            by="__n__", ascending=False, kind="mergesort"
        )
    worst_frame = worst_frame.head(max_examples)

    worst_keys = []
    for _, row in worst_frame.iterrows():
        key_value = row[_K]
        if _is_null(key_value):
            members = distinct_pairs[distinct_pairs[_K].isna()]
        else:
            members = distinct_pairs[distinct_pairs[_K] == key_value]
        values = sorted(
            (_norm(v) for v in members[_V].tolist()), key=_sort_key
        )
        worst_keys.append(
            {
                "key": _stringify(_norm(key_value)),
                "n_distinct_values": int(row["__n__"]),
                "values": [_stringify(v) for v in values][:max_examples],
            }
        )

    base.update(
        {
            "n_keys": n_keys,
            "n_distinct_values": int(frame[_V].nunique(dropna=False)),
            "n_null_keys": n_null_keys,
            "n_null_values": n_null_values,
            "max_collapse": max_collapse,
            "mean_collapse": round(mean_collapse, 6),
            "n_keys_collapsed": int(len(collapsed)),
            "worst_keys": worst_keys,
            "ok": max_collapse <= 1,
        }
    )
    return base


def assert_collapse_factor_one(
    mapping: Any,
    name: str,
    key: Optional[str] = None,
    value: Optional[str] = None,
    warn_only: bool = False,
    max_examples: int = MAX_EXAMPLES,
) -> Dict[str, Any]:
    """Assert a key -> value mapping is a *function* (collapse factor exactly 1).

    This is guarantee **G5**, the specific guarantee whose absence caused the
    incident: ``entrez -> accession`` must never be many-to-one, because the
    accession that survives ``dict()`` construction is the receptor that gets
    docked.

    Raises :class:`InvariantViolation` naming the worst offending keys and all
    the distinct values fighting over them.
    """
    report = cardinality_report(
        mapping, name=name, key=key, value=value, max_examples=max_examples
    )
    report["check"] = "collapse_factor_one"
    if report["max_collapse"] > 1:
        worst = report["worst_keys"]
        detail = "; ".join(
            "{0} -> {1} distinct values {2}".format(
                w["key"], w["n_distinct_values"], w["values"]
            )
            for w in worst
        )
        message = (
            "mapping is many-to-one: {n_keys_collapsed} of {n_keys} keys carry "
            "more than one distinct value (max_collapse={max_collapse}, "
            "mean_collapse={mean_collapse}) over {n_pairs} pairs. Building a "
            "dict from this silently keeps the LAST value per key. Worst: "
            "{detail}. Fix the resolution direction (resolve on a key that is "
            "1:1, e.g. Entrez ID, never a gene symbol) or de-duplicate with an "
            "explicit, total tie-break before building the map.".format(
                detail=detail, **report
            )
        )
        return _fail(message, name, report, warn_only)
    report["ok"] = True
    return report


# --------------------------------------------------------------------------
# I5 — total accounting / no silent drops
# --------------------------------------------------------------------------


def assert_total_accounting(
    n_in: int,
    n_resolved: int,
    n_rejected: int,
    name: str,
    warn_only: bool = False,
) -> Dict[str, Any]:
    """Assert ``n_in == n_resolved + n_rejected`` — nothing vanishes silently.

    Every stage must be able to say where each input row went. The 2025
    pipeline dropped 237 symbol-less edges, 222 slug drug rows and 163 genes
    with ``if not uniprot_id: continue`` and no ledger.
    """
    for label, count in (
        ("n_in", n_in),
        ("n_resolved", n_resolved),
        ("n_rejected", n_rejected),
    ):
        if int(count) != count or count < 0:
            raise ValueError(
                "{0} must be a non-negative integer, got {1!r}".format(label, count)
            )
    n_in, n_resolved, n_rejected = int(n_in), int(n_resolved), int(n_rejected)
    unaccounted = n_in - (n_resolved + n_rejected)
    report = {
        "check": "total_accounting",
        "name": name,
        "n_in": n_in,
        "n_resolved": n_resolved,
        "n_rejected": n_rejected,
        "n_unaccounted": unaccounted,
        "ok": unaccounted == 0,
    }
    if unaccounted != 0:
        direction = "vanished" if unaccounted > 0 else "appeared from nowhere"
        message = (
            "{0} of {1} input rows {2}: n_in ({1}) != n_resolved ({3}) + "
            "n_rejected ({4}). Every input row must end up either in the output "
            "or in the rejects ledger with a reason. Write the missing rows to "
            "<stage>.rejects.csv instead of dropping them.".format(
                abs(unaccounted), n_in, direction, n_resolved, n_rejected
            )
        )
        return _fail(message, name, report, warn_only)
    return report


# --------------------------------------------------------------------------
# I4 — row growth
# --------------------------------------------------------------------------


def _n_rows(obj: Any, label: str) -> int:
    if isinstance(obj, (pd.DataFrame, pd.Series, pd.Index)):
        return int(len(obj))
    if isinstance(obj, bool):
        raise TypeError("{0} must be a row count or a frame, got bool".format(label))
    if isinstance(obj, int):
        if obj < 0:
            raise ValueError("{0} must be non-negative, got {1}".format(label, obj))
        return obj
    if hasattr(obj, "__len__"):
        return int(len(obj))
    raise TypeError(
        "{0} must be an int or something with a length, got {1}".format(
            label, type(obj).__name__
        )
    )


def assert_no_row_growth(
    before: Any,
    after: Any,
    name: str,
    allow_shrinkage: bool = True,
    warn_only: bool = False,
) -> Dict[str, Any]:
    """Assert an operation did not add rows (and optionally did not lose any).

    ``before``/``after`` may be row counts or anything with ``len()``
    (DataFrame, Series, Index, list).

    A merge that grows the frame is the signature of a non-unique join key —
    ``parse_prepare.py:1171`` merged on ``(uniprot_id, drugbank_id,
    cavity_index)`` while ``Fragment_ID`` distinguished the rows, and added
    410,087 duplicates (one job 12x).
    """
    n_before = _n_rows(before, "before")
    n_after = _n_rows(after, "after")
    delta = n_after - n_before
    report = {
        "check": "no_row_growth",
        "name": name,
        "n_before": n_before,
        "n_after": n_after,
        "row_delta": delta,
        "allow_shrinkage": bool(allow_shrinkage),
        "ok": True,
    }
    if delta > 0:
        message = (
            "row count grew by +{0} ({1} -> {2}, +{3:.2f}%). An operation that "
            "was not meant to add rows did. The usual cause is a non-unique "
            "join key: add the missing key column (Fragment_ID was the one that "
            "bit us), de-duplicate the right-hand frame, or pass "
            "allow_growth=True if fan-out is genuinely intended.".format(
                delta, n_before, n_after, 100.0 * delta / n_before if n_before else 0.0
            )
        )
        return _fail(message, name, report, warn_only)
    if delta < 0 and not allow_shrinkage:
        message = (
            "row count shrank by {0} ({1} -> {2}). Rows were dropped where none "
            "were expected; record them in a rejects ledger or pass "
            "allow_shrinkage=True.".format(abs(delta), n_before, n_after)
        )
        return _fail(message, name, report, warn_only)
    return report


# --------------------------------------------------------------------------
# I6 — determinism
# --------------------------------------------------------------------------


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is _NULL:
        return "<NULL>"
    return repr(value)


def deterministic_fingerprint(obj: Any) -> str:
    """Return a stable SHA-256 fingerprint of ``obj``.

    Container iteration order is preserved rather than sorted, on purpose: the
    failure mode being hunted is *order* instability (``list(set(...))``), so
    two equal-but-differently-ordered results must fingerprint differently.

    ``pathlib.Path``/``os.PathLike`` pointing at an existing file fingerprints
    the file's **contents**, which is what "two builds are byte-identical"
    means for an artifact like ``identity_index.parquet``.

    A DataFrame is fingerprinted over its column names, dtypes, index and
    values — so a frame that changes only its row order, only its index, or
    only a dtype still shows up as different.
    """
    hasher = hashlib.sha256()
    _feed(hasher, obj)
    return hasher.hexdigest()


def _feed(hasher: "hashlib._Hash", obj: Any) -> None:
    def tag(text: str) -> None:
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\x1f")

    if obj is None:
        tag("none")
        return
    if isinstance(obj, bool):
        tag("bool:" + repr(obj))
        return
    if isinstance(obj, (bytes, bytearray)):
        tag("bytes:{0}".format(len(obj)))
        hasher.update(bytes(obj))
        return
    if isinstance(obj, str):
        tag("str")
        hasher.update(obj.encode("utf-8"))
        hasher.update(b"\x1e")
        return
    if isinstance(obj, (int, float)):
        tag("num:" + repr(obj))
        return
    if isinstance(obj, pd.DataFrame):
        tag("dataframe")
        tag(",".join(str(c) for c in obj.columns))
        tag(",".join(str(d) for d in obj.dtypes))
        hasher.update(obj.to_csv(index=True, lineterminator="\n").encode("utf-8"))
        return
    if isinstance(obj, pd.Series):
        tag("series:" + str(obj.name))
        _feed(hasher, obj.to_frame())
        return
    if isinstance(obj, pd.Index):
        tag("index")
        _feed(hasher, list(obj))
        return
    if isinstance(obj, _Mapping):
        tag("mapping:{0}".format(len(obj)))
        for k, v in obj.items():
            _feed(hasher, k)
            _feed(hasher, v)
        return
    if isinstance(obj, (os.PathLike,)):
        path = os.fspath(obj)
        if os.path.isfile(path):
            tag("file")
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    hasher.update(chunk)
        else:
            tag("path")
            _feed(hasher, str(path))
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        tag("{0}:{1}".format(type(obj).__name__, len(obj)))
        for item in obj:
            _feed(hasher, item)
        return
    tag("repr")
    hasher.update(repr(obj).encode("utf-8"))


def assert_deterministic(
    func: Callable[..., Any],
    name: str,
    args: Sequence[Any] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    runs: int = 2,
    warn_only: bool = False,
) -> Dict[str, Any]:
    """Call ``func`` ``runs`` times and require byte-identical results.

    Catches in-process nondeterminism: unseeded RNG, timestamps, UUIDs, and
    any result whose order depends on iteration over an unordered container
    that is rebuilt each call.

    Cross-*process* hash-order instability (``PYTHONHASHSEED``) cannot be seen
    from inside one interpreter — use
    :func:`assert_deterministic_across_hash_seeds` for that.
    """
    if runs < 2:
        raise ValueError("runs must be >= 2 to compare anything")
    kwargs = dict(kwargs or {})
    fingerprints = []
    for _ in range(runs):
        fingerprints.append(deterministic_fingerprint(func(*args, **kwargs)))
    distinct = sorted(set(fingerprints))
    report = {
        "check": "deterministic",
        "name": name,
        "runs": runs,
        "fingerprints": fingerprints,
        "n_distinct_fingerprints": len(distinct),
        "ok": len(distinct) == 1,
    }
    if len(distinct) > 1:
        first_diff = next(
            i for i, fp in enumerate(fingerprints) if fp != fingerprints[0]
        )
        message = (
            "output is not reproducible: run 1 and run {0} of {1} differ "
            "({2} distinct fingerprints, e.g. {3}... vs {4}...). Look for "
            "list(set(...)) or dict/set iteration over identifiers, an "
            "unseeded RNG, a timestamp, or drop_duplicates over an unordered "
            "concat. Every tie-break must be total and explicit.".format(
                first_diff + 1,
                runs,
                len(distinct),
                fingerprints[0][:12],
                fingerprints[first_diff][:12],
            )
        )
        return _fail(message, name, report, warn_only)
    return report


def assert_deterministic_across_hash_seeds(
    snippet: str,
    name: str,
    seeds: Sequence[int] = (0, 1, 2),
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 600,
    warn_only: bool = False,
) -> Dict[str, Any]:
    """Run a Python ``snippet`` under several ``PYTHONHASHSEED`` values and
    require identical stdout.

    This is guarantee **G3/I6**. ``prepare_uniprot_mapping.py:51`` did
    ``list(set(gene_names_list))``; the resulting query order was
    hash-randomised and a first-wins ``drop_duplicates`` then made *which
    protein owned an accession* depend on the interpreter's hash seed. That is
    invisible inside a single process and trivially visible here.

    The snippet should print a digest of the artifact under test, e.g.::

        assert_deterministic_across_hash_seeds(
            "import hashlib,sys;"
            "sys.path.insert(0, '/home/onur/repos/dock_selectivity');"
            "from build_identity_index import build;"
            "print(hashlib.sha256(build().to_parquet()).hexdigest())",
            name="identity_index",
        )

    Runs ``sys.executable``; no network, no shell.
    """
    if len(seeds) < 2:
        raise ValueError("need at least two seeds to compare anything")
    outputs: List[str] = []
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        child_env = dict(os.environ if env is None else env)
        child_env["PYTHONHASHSEED"] = str(seed)
        completed = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=cwd,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        stdout = completed.stdout.decode("utf-8", "replace")
        per_seed.append(
            {
                "seed": int(seed),
                "returncode": completed.returncode,
                "stdout_sha256": hashlib.sha256(
                    completed.stdout
                ).hexdigest(),
                "stdout_head": stdout[:200],
                "stderr_tail": completed.stderr.decode("utf-8", "replace")[-400:],
            }
        )
        if completed.returncode != 0:
            report = {
                "check": "deterministic_across_hash_seeds",
                "name": name,
                "seeds": [int(s) for s in seeds],
                "runs": per_seed,
                "ok": False,
            }
            message = (
                "snippet exited {0} under PYTHONHASHSEED={1}; determinism is "
                "untestable until it runs cleanly. stderr tail: {2}".format(
                    completed.returncode, seed, per_seed[-1]["stderr_tail"]
                )
            )
            return _fail(message, name, report, warn_only)
        outputs.append(stdout)

    distinct = sorted(set(outputs))
    report = {
        "check": "deterministic_across_hash_seeds",
        "name": name,
        "seeds": [int(s) for s in seeds],
        "runs": per_seed,
        "n_distinct_outputs": len(distinct),
        "ok": len(distinct) == 1,
    }
    if len(distinct) > 1:
        message = (
            "output depends on PYTHONHASHSEED: {0} distinct results over seeds "
            "{1}. Something iterates a set or a dict keyed by strings and lets "
            "that order decide the result (this is exactly "
            "prepare_uniprot_mapping.py:51 + :130). Sort explicitly with a "
            "total key.".format(len(distinct), list(seeds))
        )
        return _fail(message, name, report, warn_only)
    return report


# --------------------------------------------------------------------------
# I1 — label self-consistency  (the check that would have caught the bug)
# --------------------------------------------------------------------------


def assert_self_consistent_labels(
    df: pd.DataFrame,
    key_col: str,
    label_col: str,
    truth_lookup: Union[Callable[[Any], Any], Mapping[Any, Any]],
    name: str,
    max_examples: int = MAX_EXAMPLES,
    max_violation_rate: float = 0.0,
    warn_only: bool = False,
) -> Dict[str, Any]:
    """Assert every row's label really belongs to that row's own key.

    **This is the check that would have caught the incident on day one.** On
    ``uniprot_gene_mapping.csv`` it prints 2,647/4,585 = 57.7% violations,
    including the row that decided which receptor got docked::

        accession Q8NCT1 labelled 'AVPR2', but Q8NCT1's own symbols are
        ('ARRDC4',)   -- the real AVPR2 is P30518

    Parameters
    ----------
    df:
        Frame to audit.
    key_col:
        Column holding the canonical key (a UniProt accession).
    label_col:
        Column holding the display label (a gene symbol).
    truth_lookup:
        **Independent** source of truth: either a callable
        ``key -> allowed label(s)`` (e.g.
        ``lambda acc: index.by_accession(acc).symbols``) or a mapping
        ``key -> allowed label(s)``. A single string is accepted as a
        one-element collection. It must NOT be derived from ``df`` — an
        assertion that consumes the object it validates cannot fail
        (``validate_negatives()`` reported "All validations passed" that way).
    max_violation_rate:
        Tolerance, default ``0.0`` (no violation allowed).

    A key missing from ``truth_lookup``, a null key, and a null label are all
    violations — never silent passes.
    """
    for col in (key_col, label_col):
        if col not in df.columns:
            raise ValueError(
                "column {0!r} not in frame; available: {1}".format(
                    col, list(df.columns)
                )
            )
    if callable(truth_lookup):
        getter = truth_lookup
        is_callable = True
    elif isinstance(truth_lookup, _Mapping):
        getter = truth_lookup
        is_callable = False
    else:
        raise TypeError(
            "truth_lookup must be a callable or a Mapping, got {0}".format(
                type(truth_lookup).__name__
            )
        )

    missing = object()

    def allowed_for(key: Any) -> Any:
        if is_callable:
            try:
                return getter(key)
            except KeyError:
                return missing
        return getter.get(key, missing)

    violations: List[Dict[str, str]] = []
    n_missing_keys = 0
    n_null_keys = 0
    n_null_labels = 0
    n_ok = 0

    for key, label in zip(df[key_col].tolist(), df[label_col].tolist()):
        if _is_null(key):
            n_null_keys += 1
            violations.append(
                {
                    "key": "<NULL>",
                    "label": _stringify(label),
                    "reason": "null key: this row cannot be attributed to any protein",
                    "allowed": "",
                }
            )
            continue
        if _is_null(label):
            n_null_labels += 1
            violations.append(
                {
                    "key": _stringify(key),
                    "label": "<NULL>",
                    "reason": "null label",
                    "allowed": "",
                }
            )
            continue
        allowed = allowed_for(key)
        if allowed is missing:
            n_missing_keys += 1
            violations.append(
                {
                    "key": _stringify(key),
                    "label": _stringify(label),
                    "reason": "key absent from the truth source",
                    "allowed": "",
                }
            )
            continue
        if isinstance(allowed, str):
            allowed_set = {allowed}
        elif isinstance(allowed, _Iterable):
            allowed_set = {a for a in allowed if not _is_null(a)}
        else:
            allowed_set = {allowed}
        if label in allowed_set:
            n_ok += 1
            continue
        violations.append(
            {
                "key": _stringify(key),
                "label": _stringify(label),
                "reason": "label is not a symbol of this key",
                "allowed": ", ".join(
                    sorted(_stringify(a) for a in allowed_set)[: max_examples * 2]
                ),
            }
        )

    n_rows = int(len(df))
    n_violations = len(violations)
    rate = (n_violations / float(n_rows)) if n_rows else 0.0
    report = {
        "check": "self_consistent_labels",
        "name": name,
        "key_col": key_col,
        "label_col": label_col,
        "n_rows": n_rows,
        "n_ok": n_ok,
        "n_violations": n_violations,
        "violation_rate": round(rate, 6),
        "n_missing_keys": n_missing_keys,
        "n_null_keys": n_null_keys,
        "n_null_labels": n_null_labels,
        "max_violation_rate": max_violation_rate,
        "examples": violations[:max_examples],
        "ok": rate <= max_violation_rate,
    }
    if rate > max_violation_rate:
        detail = "; ".join(
            "{0}={1} labelled {2!r} but {3}".format(
                key_col,
                v["key"],
                v["label"],
                (
                    "its own labels are ({0})".format(v["allowed"])
                    if v["allowed"]
                    else v["reason"]
                ),
            )
            for v in violations[:max_examples]
        )
        message = (
            "{0} of {1} rows ({2:.2%}) carry a label that does not belong to "
            "their own {3} (tolerance {4:.2%}). A label must be derived FROM "
            "the key (accession -> primary_symbol), never used to select it. "
            "Offenders: {5}".format(
                n_violations,
                n_rows,
                rate,
                key_col,
                max_violation_rate,
                detail,
            )
        )
        return _fail(message, name, report, warn_only)
    return report


# --------------------------------------------------------------------------
# key uniqueness (used by joins.checked_merge and schemas.validate)
# --------------------------------------------------------------------------


def assert_unique_key(
    df: pd.DataFrame,
    cols: Union[str, Sequence[str]],
    name: str,
    max_examples: int = MAX_EXAMPLES,
    warn_only: bool = False,
) -> Dict[str, Any]:
    """Assert ``cols`` uniquely identify the rows of ``df``.

    A non-unique key is what turns an innocent ``.merge()`` into a fan-out.
    The 2025 parquet had 1,428 duplicate ``(uniprot_id, drugbank_id,
    cavity_index)`` keys because ``Fragment_ID`` was left out.
    """
    key_cols = [cols] if isinstance(cols, str) else list(cols)
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "key column(s) {0} not in frame; available: {1}".format(
                missing, list(df.columns)
            )
        )
    dup_report = _duplicate_key_stats(df, key_cols, max_examples=max_examples)
    report = {
        "check": "unique_key",
        "name": name,
        "key": key_cols,
        "n_rows": int(len(df)),
        "n_distinct_keys": dup_report["n_distinct_keys"],
        "n_duplicate_keys": dup_report["n_duplicate_keys"],
        "n_duplicate_rows": dup_report["n_duplicate_rows"],
        "max_multiplicity": dup_report["max_multiplicity"],
        "worst_keys": dup_report["worst_keys"],
        "n_null_key_rows": dup_report["n_null_key_rows"],
        "ok": dup_report["n_duplicate_keys"] == 0,
    }
    if dup_report["n_duplicate_keys"]:
        message = (
            "key {0} is not unique: {1} duplicated key value(s) over {2} rows "
            "(max multiplicity {3}). Worst: {4}. Either the key is missing a "
            "column (Fragment_ID, historically) or the frame needs an explicit "
            "de-duplication with a total tie-break.".format(
                key_cols,
                dup_report["n_duplicate_keys"],
                dup_report["n_duplicate_rows"],
                dup_report["max_multiplicity"],
                dup_report["worst_keys"],
            )
        )
        return _fail(message, name, report, warn_only)
    return report


def _duplicate_key_stats(
    df: pd.DataFrame,
    key_cols: Sequence[str],
    max_examples: int = MAX_EXAMPLES,
) -> Dict[str, Any]:
    """Duplicate statistics for ``key_cols`` in ``df`` (nulls grouped, not dropped).

    Vectorised on purpose: this runs against the 16.5M-row docking frame in
    ``parse_prepare.py``, so it must not materialise a Python object per row.
    """
    key_cols = list(key_cols)
    empty = {
        "n_distinct_keys": 0,
        "n_duplicate_keys": 0,
        "n_duplicate_rows": 0,
        "max_multiplicity": 0,
        "worst_keys": [],
        "n_null_key_rows": 0,
    }
    if len(df) == 0:
        return empty

    counts = df.groupby(key_cols, dropna=False, sort=False).size()
    if len(counts) == 0:  # pragma: no cover - defensive
        return empty
    frame = counts.rename("__n_rows__").reset_index()
    duplicated = frame[frame["__n_rows__"] > 1]
    try:
        duplicated = duplicated.sort_values(
            by=["__n_rows__"] + key_cols,
            ascending=[False] + [True] * len(key_cols),
            kind="mergesort",
        )
    except TypeError:  # mixed types in a key column: fall back to a stable sort
        duplicated = duplicated.sort_values(
            by="__n_rows__", ascending=False, kind="mergesort"
        )
    worst = duplicated.head(max_examples)
    n_null_key_rows = int(df[key_cols].isna().any(axis=1).sum())
    return {
        "n_distinct_keys": int(len(frame)),
        "n_duplicate_keys": int(len(duplicated)),
        "n_duplicate_rows": int(duplicated["__n_rows__"].sum()),
        "max_multiplicity": int(frame["__n_rows__"].max()),
        "worst_keys": [
            {
                "key": [_stringify(row[c]) for c in key_cols],
                "n_rows": int(row["__n_rows__"]),
            }
            for _, row in worst.iterrows()
        ],
        "n_null_key_rows": n_null_key_rows,
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    try:
        import numpy as np  # local import: keeps the module importable without numpy
    except ImportError:  # pragma: no cover - numpy is a hard dep of pandas
        np = None
    if np is not None:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


def write_invariant_report(
    reports: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    path: Any,
) -> str:
    """Write invariant reports to ``path`` as deterministic JSON.

    A stage writes ``<stage>.invariants.json`` next to its output so that a
    passing run leaves positive evidence, not merely an absence of exceptions.
    Keys are sorted and the file is written atomically, so two identical runs
    produce byte-identical files.
    """
    payload = list(reports) if isinstance(reports, (list, tuple)) else [reports]
    text = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default
    )
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=parent or ".", delete=False, suffix=".tmp"
    )
    try:
        handle.write(text)
        handle.write("\n")
        handle.close()
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    n_failed = sum(1 for r in payload if not r.get("ok", True))
    logger.info(
        "wrote %d invariant report(s) (%d failing) to %s", len(payload), n_failed, path
    )
    return path
