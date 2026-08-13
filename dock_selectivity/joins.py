"""Checked joins — a merge that cannot silently change the shape of the data.

Why this module exists
----------------------

``parse_prepare.py:1171`` merged sample-type annotations onto the docking
results with::

    df = df.merge(sample_types, on=['uniprot_id', 'drugbank_id', 'cavity_index'],
                  how='left')

The key omitted ``Fragment_ID``, so 1,428 key values were non-unique on the
right-hand side and the "annotation" **duplicated 410,087 rows** across 179
jobs — one job appearing 12 times. Nothing failed, nothing warned; the
inflation was found a year later by dividing parquet rows by
``final_results.csv`` rows and getting exactly 12.00.

The same file also carried 1,174,786 rows (7.08%) whose identity columns were
null: unmatched left rows that no one counted.

:func:`checked_merge` is a drop-in replacement for ``DataFrame.merge`` that
makes both failures impossible to ship:

* a merge that adds rows raises, and the message names the duplicated keys on
  the right that caused the fan-out;
* unmatched left rows are always counted and reported, and can be required to
  stay under a declared budget with ``require_match``;
* :func:`unmatched_left_frame` hands back the unmatched rows so a stage can
  write them to its rejects ledger instead of dropping them.

Usage::

    from dock_selectivity.joins import checked_merge

    out = checked_merge(
        results, sample_types,
        on=['uniprot_id', 'drugbank_id', 'fragment_id', 'cavity_index'],
        how='left', name='annotate_sample_types', require_match=0.99,
    )

Note on nulls: pandas (unlike SQL) *matches* NaN keys to NaN keys, so a frame
with many null keys can fan out against every null key on the other side. Null
key counts are always in the report, and ``forbid_null_keys=True`` turns them
into an error.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .invariants import (
    MAX_EXAMPLES,
    InvariantViolation,
    _duplicate_key_stats,
    _fail,
    _stringify,
    assert_unique_key,
)

__all__ = [
    "checked_merge",
    "checked_merge_with_report",
    "unmatched_left_frame",
    "duplicate_key_report",
    "InvariantViolation",
    "assert_unique_key",
]

logger = logging.getLogger(__name__)

_VALID_HOWS = ("left", "right", "inner", "outer")

#: Internal indicator column name — chosen so it cannot collide with a real
#: column or with pandas' own default ``_merge``.
_INDICATOR = "_checked_merge_indicator"


def _as_key_list(value: Union[str, Sequence[str], None]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _require_columns(df: pd.DataFrame, cols: Sequence[str], side: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            "{0} frame is missing join column(s) {1}; available: {2}".format(
                side, missing, list(df.columns)
            )
        )


def duplicate_key_report(
    df: pd.DataFrame,
    keys: Union[str, Sequence[str]],
    name: Optional[str] = None,
    max_examples: int = MAX_EXAMPLES,
) -> Dict[str, Any]:
    """Duplicate-key statistics for ``keys`` in ``df`` (never raises).

    Use this before designing a join key: if the right-hand frame has duplicate
    keys, a left merge on those keys *will* fan out.
    """
    key_cols = _as_key_list(keys) or []
    _require_columns(df, key_cols, "frame")
    stats = _duplicate_key_stats(df, key_cols, max_examples=max_examples)
    stats.update({"check": "duplicate_key_report", "name": name, "key": key_cols})
    return stats


def _key_index(df: pd.DataFrame, cols: Sequence[str]):
    """Join key as an Index/MultiIndex, for vectorised membership tests.

    Kept vectorised deliberately: these helpers run against the 16.5M-row
    docking frame, so nothing may build one Python object per row.
    ``Index.isin`` treats NaN as matching NaN, which is also how
    ``DataFrame.merge`` behaves — the two agree by construction.
    """
    cols = list(cols)
    if len(cols) == 1:
        return pd.Index(df[cols[0]])
    return pd.MultiIndex.from_frame(df[cols])


def _matched_mask(left: pd.DataFrame, left_cols, right: pd.DataFrame, right_cols):
    """Boolean array over ``left`` rows: does this key exist in ``right``?"""
    if len(left) == 0:
        return np.zeros(0, dtype=bool)
    if len(right) == 0:
        return np.zeros(len(left), dtype=bool)
    return np.asarray(_key_index(left, left_cols).isin(_key_index(right, right_cols)))


def unmatched_left_frame(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Union[str, Sequence[str], None] = None,
    left_on: Union[str, Sequence[str], None] = None,
    right_on: Union[str, Sequence[str], None] = None,
) -> pd.DataFrame:
    """Return the rows of ``left`` whose join key is absent from ``right``.

    This is the ledger half of :func:`checked_merge`: a stage writes these rows
    to ``<stage>.rejects.csv`` with a reason instead of letting a left merge
    turn them into silent nulls (7.08% of the 2025 parquet).
    """
    left_cols, right_cols = _resolve_keys(on, left_on, right_on)
    _require_columns(left, left_cols, "left")
    _require_columns(right, right_cols, "right")
    matched = _matched_mask(left, left_cols, right, right_cols)
    return left[~matched].copy()


def _resolve_keys(
    on: Union[str, Sequence[str], None],
    left_on: Union[str, Sequence[str], None],
    right_on: Union[str, Sequence[str], None],
) -> Tuple[List[str], List[str]]:
    on_list = _as_key_list(on)
    left_list = _as_key_list(left_on)
    right_list = _as_key_list(right_on)
    if on_list is not None:
        if left_list is not None or right_list is not None:
            raise ValueError("pass either on=, or left_on=/right_on=, not both")
        return list(on_list), list(on_list)
    if left_list is None or right_list is None:
        raise ValueError(
            "a join key is required: pass on=, or both left_on= and right_on=. "
            "An implicit index join is never acceptable in this pipeline."
        )
    if len(left_list) != len(right_list):
        raise ValueError(
            "left_on ({0}) and right_on ({1}) must have the same length".format(
                left_list, right_list
            )
        )
    return left_list, right_list


def _expected_rows(
    how: str, n_left: int, n_right: int, n_matched_left: int, n_unmatched_right: int
) -> int:
    """Row count the merge produces when neither side duplicates the other.

    Anything above this is fan-out. Note ``inner`` is ``n_matched_left``, not
    ``n_left``: an inner join that drops one row while duplicating another can
    land back on ``n_left`` and would slip past a naive "did it get bigger?"
    check.
    """
    if how == "left":
        return n_left
    if how == "inner":
        return n_matched_left
    if how == "right":
        return n_right
    return n_left + n_unmatched_right  # outer


def checked_merge_with_report(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Union[str, Sequence[str], None] = None,
    how: str = "left",
    name: Optional[str] = None,
    allow_growth: bool = False,
    require_match: Optional[float] = None,
    validate: Optional[str] = None,
    left_on: Union[str, Sequence[str], None] = None,
    right_on: Union[str, Sequence[str], None] = None,
    suffixes: Tuple[str, str] = ("_x", "_y"),
    indicator: Union[bool, str] = False,
    forbid_null_keys: bool = False,
    max_examples: int = MAX_EXAMPLES,
    warn_only: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """:func:`checked_merge`, additionally returning the merge report dict.

    The report is JSON-serialisable and is meant to be collected by the stage
    and written with :func:`dock_selectivity.invariants.write_invariant_report`.

    Two fields are ``None`` unless they were needed, because computing them
    costs a full pass over the left (16.5M-row) frame:
    ``n_duplicate_left_keys`` is filled in only when a check fails, and
    ``n_unmatched_right`` only for ``how='outer'``. Everything else is always
    populated. Total overhead over a bare ``DataFrame.merge`` is roughly 2x.
    """
    if how not in _VALID_HOWS:
        raise ValueError("how must be one of {0}, got {1!r}".format(_VALID_HOWS, how))
    if require_match is not None and not (0.0 <= float(require_match) <= 1.0):
        raise ValueError("require_match must be a fraction in [0, 1]")
    if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
        raise TypeError("checked_merge operates on two DataFrames")

    left_cols, right_cols = _resolve_keys(on, left_on, right_on)
    _require_columns(left, left_cols, "left")
    _require_columns(right, right_cols, "right")

    label = name or "merge {0} on {1}".format(how, left_cols)
    n_left, n_right = int(len(left)), int(len(right))

    # The right frame is the fan-out source (and in this pipeline it is the
    # small annotation table), so its key statistics are computed eagerly. The
    # LEFT frame is the 16.5M-row one: only its null count is computed up
    # front; its duplicate statistics cost a full groupby and are computed
    # lazily, in the failure path, where the diagnosis is actually needed.
    right_dups = _duplicate_key_stats(right, right_cols, max_examples=max_examples)
    n_null_left = int(left[left_cols].isna().any(axis=1).sum()) if n_left else 0
    n_null_right = right_dups["n_null_key_rows"]

    report: Dict[str, Any] = {
        "check": "checked_merge",
        "name": label,
        "how": how,
        "left_on": left_cols,
        "right_on": right_cols,
        "n_left": n_left,
        "n_right": n_right,
        "n_out": None,
        "row_delta": None,
        "expected_rows": None,
        "n_matched_left": None,
        "n_unmatched_left": None,
        "n_unmatched_right": None,  # only computed for how='outer'
        "match_rate": None,
        "require_match": require_match,
        "allow_growth": bool(allow_growth),
        "unmatched_left_examples": [],
        "n_duplicate_right_keys": right_dups["n_duplicate_keys"],
        "max_right_key_multiplicity": right_dups["max_multiplicity"],
        "worst_right_keys": right_dups["worst_keys"],
        "n_duplicate_left_keys": None,  # computed only when a check fails
        "n_null_key_rows_left": n_null_left,
        "n_null_key_rows_right": n_null_right,
        "ok": True,
    }

    if forbid_null_keys and (n_null_left or n_null_right):
        message = (
            "join key contains nulls ({0} left row(s), {1} right row(s)) on {2}. "
            "pandas matches NaN to NaN, so null keys join to each other and can "
            "fan out. Resolve or reject those rows before joining.".format(
                n_null_left, n_null_right, left_cols
            )
        )
        _fail(message, label, report, warn_only)

    merge_kwargs: Dict[str, Any] = {
        "how": how,
        "suffixes": suffixes,
        "indicator": _INDICATOR,
    }
    if on is not None:
        merge_kwargs["on"] = left_cols
    else:
        merge_kwargs["left_on"] = left_cols
        merge_kwargs["right_on"] = right_cols
    if validate is not None:
        merge_kwargs["validate"] = validate

    try:
        merged = left.merge(right, **merge_kwargs)
    except pd.errors.MergeError as exc:
        report["ok"] = False
        report["pandas_validate"] = validate
        message = (
            "pandas rejected validate={0!r}: {1}. Right-hand duplicate keys: "
            "{2} (max multiplicity {3}); worst {4}.".format(
                validate,
                exc,
                right_dups["n_duplicate_keys"],
                right_dups["max_multiplicity"],
                right_dups["worst_keys"],
            )
        )
        raise InvariantViolation(message, name=label, report=report) from exc

    n_out = int(len(merged))
    report["n_out"] = n_out
    report["row_delta"] = n_out - n_left

    indicator_counts = merged[_INDICATOR].value_counts()
    n_left_only = int(indicator_counts.get("left_only", 0))
    n_right_only = int(indicator_counts.get("right_only", 0))
    report["n_left_only"] = n_left_only
    report["n_right_only"] = n_right_only
    report["n_both"] = int(indicator_counts.get("both", 0))

    # For 'left'/'outer' the indicator already counts the unmatched left rows
    # exactly (each appears once), so no extra pass over the big frame is
    # needed. 'inner'/'right' drop them, so there the membership test is run.
    if how in ("left", "outer"):
        n_unmatched_left = n_left_only
    else:
        n_unmatched_left = n_left - int(
            _matched_mask(left, left_cols, right, right_cols).sum()
        )
    n_matched_left = n_left - n_unmatched_left
    n_unmatched_right = n_right_only if how == "outer" else None
    match_rate = (n_matched_left / float(n_left)) if n_left else 1.0

    report["n_matched_left"] = n_matched_left
    report["n_unmatched_left"] = n_unmatched_left
    report["n_unmatched_right"] = n_unmatched_right
    report["match_rate"] = round(match_rate, 6)
    report["expected_rows"] = _expected_rows(
        how, n_left, n_right, n_matched_left, n_unmatched_right or 0
    )

    if n_unmatched_left:
        # Diagnostic only, so the second pass is paid for only when something
        # actually failed to match. First occurrences, in input order.
        unmatched = left.loc[
            ~_matched_mask(left, left_cols, right, right_cols), left_cols
        ]
        report["unmatched_left_examples"] = [
            [_stringify(v) for v in record]
            for record in unmatched.drop_duplicates()
            .head(max_examples)
            .itertuples(index=False, name=None)
        ]

    if indicator is False:
        merged = merged.drop(columns=[_INDICATOR])
    else:
        merged = merged.rename(
            columns={_INDICATOR: "_merge" if indicator is True else str(indicator)}
        )

    expected = report["expected_rows"]
    if n_out != expected and not allow_growth:
        # Now that something is wrong, pay for the full left-hand diagnosis.
        left_dups = _duplicate_key_stats(left, left_cols, max_examples=max_examples)
        report["n_duplicate_left_keys"] = left_dups["n_duplicate_keys"]
        cause = (
            "the right frame has {0} duplicated key value(s) (max multiplicity "
            "{1}); worst: {2}".format(
                right_dups["n_duplicate_keys"],
                right_dups["max_multiplicity"],
                right_dups["worst_keys"],
            )
            if right_dups["n_duplicate_keys"]
            else "no duplicated right keys were found, so check for null keys "
            "({0} left / {1} right; pandas matches NaN to NaN) or a "
            "many-to-many key ({2} duplicated left keys)".format(
                n_null_left, n_null_right, left_dups["n_duplicate_keys"]
            )
        )
        message = (
            "merge changed the row count: {0} -> {1} (delta {2:+d}; {3} rows "
            "expected for how={4!r} without fan-out) on key={5}. This is the "
            "parse_prepare.py:1171 failure mode (+410,087 rows from a key "
            "missing Fragment_ID). Cause: {6}. Add the missing key column, "
            "de-duplicate the right frame, or pass allow_growth=True if "
            "fan-out is intended.".format(
                n_left, n_out, n_out - n_left, expected, how, left_cols, cause
            )
        )
        _fail(message, label, report, warn_only)

    if require_match is not None and match_rate < require_match:
        message = (
            "match rate {0:.4%} is below the required {1:.4%}: {2} of {3} left "
            "rows found no match on {4}. Unmatched key examples: {5}. Record "
            "them with unmatched_left_frame() and a reason, or fix the key "
            "(null keys: {6} left / {7} right).".format(
                match_rate,
                require_match,
                n_unmatched_left,
                n_left,
                left_cols,
                report["unmatched_left_examples"],
                n_null_left,
                n_null_right,
            )
        )
        _fail(message, label, report, warn_only)

    logger.debug(
        "checked_merge[%s]: %d x %d -> %d rows, match_rate=%.4f",
        label,
        n_left,
        n_right,
        n_out,
        match_rate,
    )
    return merged, report


def checked_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Union[str, Sequence[str], None] = None,
    how: str = "left",
    name: Optional[str] = None,
    allow_growth: bool = False,
    require_match: Optional[float] = None,
    validate: Optional[str] = None,
    left_on: Union[str, Sequence[str], None] = None,
    right_on: Union[str, Sequence[str], None] = None,
    suffixes: Tuple[str, str] = ("_x", "_y"),
    indicator: Union[bool, str] = False,
    forbid_null_keys: bool = False,
    max_examples: int = MAX_EXAMPLES,
    warn_only: bool = False,
) -> pd.DataFrame:
    """Merge ``left`` and ``right``, asserting the result's shape.

    Parameters
    ----------
    left, right:
        Frames to join.
    on:
        Join key (str or list). Use ``left_on``/``right_on`` when the names
        differ. There is no index join: a key must always be named.
    how:
        ``'left'`` (default), ``'right'``, ``'inner'`` or ``'outer'``.
    name:
        Call-site label used in messages, logs and the report — make it the
        stage name (``'annotate_sample_types'``).
    allow_growth:
        ``False`` (default) means neither side may duplicate the other. The
        output must have exactly the row count a fan-out-free merge produces:
        ``len(left)`` for ``how='left'``, the number of matched left rows for
        ``'inner'``, ``len(right)`` for ``'right'``, and
        ``len(left) + unmatched right rows`` for ``'outer'``. Set ``True`` only
        when fan-out is the intent, and say so in the ``name``.
    require_match:
        Optional minimum fraction of *left* rows that must find a match
        (``1.0`` = every row). Below it, raises with the unmatched key
        examples.
    validate:
        Passed through to pandas (``'1:1'``, ``'m:1'``, ...); a ``MergeError``
        is re-raised as :class:`InvariantViolation` with the duplicate keys
        named.
    forbid_null_keys:
        Raise if either side has null join keys (pandas joins NaN to NaN).
    warn_only:
        Log instead of raising. For triage only — never for a production stage.

    Raises
    ------
    InvariantViolation
        If the row count grew, the match rate is below ``require_match``, or
        pandas' ``validate`` failed.
    """
    merged, _ = checked_merge_with_report(
        left,
        right,
        on=on,
        how=how,
        name=name,
        allow_growth=allow_growth,
        require_match=require_match,
        validate=validate,
        left_on=left_on,
        right_on=right_on,
        suffixes=suffixes,
        indicator=indicator,
        forbid_null_keys=forbid_null_keys,
        max_examples=max_examples,
        warn_only=warn_only,
    )
    return merged
