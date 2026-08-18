"""Pocket-confidence lookup: read ``cavity_quality.csv`` and apply a pLDDT cutoff.

``score_cavity_quality.py`` writes the artifact; this module is the *only* way any
downstream stage should read it. Three functions, and the reason for each:

``load_cavity_quality(path)``
    Read and validate against the ``cavity_quality`` schema, on read as well as on
    write. A hand-edited or stale file fails here rather than quietly filtering a
    different set of pockets than the operator asked for.

``fraction_at(frame, plddt_min)``
    Recompute "fraction of pocket residues at or above this pLDDT" **from the
    ``residue_plddt`` column**, never from the three precomputed ``frac_ge_*``
    columns. That is the whole reason the per-residue values are stored: a cutoff
    sweep costs a string split, not a second pass over 7.0 GB of tarballs. Asked
    for 50/70/90 it reproduces the stored columns exactly, because the producer
    derives those columns from the same rounded values.

``passes(frame, plddt_min, min_fraction)``
    The filter predicate itself: ``fraction_at(frame, plddt_min) >= min_fraction``.

What this module deliberately does **not** do
---------------------------------------------

It does not decide what happens to a pocket the manifest names but the quality
file does not carry. That is a policy question with a scientific cost -- treating
it as a pass silently admits unscored pockets, treating it as a drop silently
shrinks the dataset -- so it belongs in the stage, with its own flag
(``--allow-missing-quality``) and its own reject rows, not hidden in a loader.

Usage::

    from dock_selectivity import quality

    frame = quality.load_cavity_quality(args.cavity_quality)
    fraction = quality.fraction_at(frame, plddt_min=70.0)
    keep = quality.passes(frame, plddt_min=70.0, min_fraction=0.8)

    for row, frac in zip(frame.loc[~keep].itertuples(), fraction[~keep]):
        reject("pocket_quality", "frac_ge_70=%.2f < 0.80 (n_residues=%d)"
                                 % (frac, row.n_residues))
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple, Union

import pandas as pd

from .invariants import InvariantViolation
from .schemas import read_checked

__all__ = [
    "QualityError",
    "QUALITY_KEY_COLUMNS",
    "SCHEMA_NAME",
    "load_cavity_quality",
    "parse_residue_plddt",
    "fraction_at",
    "passes",
]

#: The schema :func:`load_cavity_quality` validates against.
SCHEMA_NAME = "cavity_quality"

#: The identity of a pocket, everywhere. Downstream stages join on exactly these
#: three columns -- ``accession`` is the only protein key, ``fragment_id`` is
#: normalised ``'F<k>'``, and ``cavity_index`` is an int.
QUALITY_KEY_COLUMNS: Tuple[str, str, str] = (
    "accession",
    "fragment_id",
    "cavity_index",
)

#: The column every cutoff is recomputed from.
RESIDUE_COLUMN = "residue_plddt"


class QualityError(InvariantViolation):
    """A cavity-quality frame cannot answer the question that was asked of it."""


def load_cavity_quality(path: Union[str, "os.PathLike[str]"]) -> pd.DataFrame:
    """Read ``cavity_quality.csv`` (or ``.parquet``), validated against its schema.

    Returns a frame with the declared columns at their declared dtypes, indexed
    ``0..n-1``. A missing file, a missing column, a duplicated
    ``(accession, fragment_id, cavity_index)`` or a ``residue_plddt`` whose value
    count disagrees with ``n_residues`` all raise
    :class:`~dock_selectivity.schemas.SchemaViolation` here, before anything has
    been filtered on it.
    """
    return read_checked(os.fspath(path), SCHEMA_NAME)


def parse_residue_plddt(value: Any) -> List[float]:
    """The ``;``-separated per-residue pLDDT string, as floats.

    Empty fields are ignored (a trailing separator is harmless); a value that is
    not a number raises, because a pocket whose residue list cannot be read must
    not quietly become a pocket with fewer residues.
    """
    if value is None or (isinstance(value, float) and value != value):
        return []
    out: List[float] = []
    for part in str(value).split(";"):
        text = part.strip()
        if not text:
            continue
        try:
            out.append(float(text))
        except ValueError:
            raise QualityError(
                "residue_plddt contains {0!r}, which is not a number. The "
                "per-residue list is what every later cutoff is recomputed "
                "from; it may not be repaired by discarding the value.".format(text)
            ) from None
    return out


def _residue_series(frame: pd.DataFrame, caller: str) -> pd.Series:
    if not isinstance(frame, pd.DataFrame):
        raise QualityError(
            "{0}() expects a DataFrame, got {1}".format(caller, type(frame).__name__)
        )
    if RESIDUE_COLUMN not in frame.columns:
        raise QualityError(
            "{0}() needs the {1!r} column, and the frame has {2}. Load the file "
            "with load_cavity_quality() -- an arbitrary cutoff cannot be "
            "evaluated from the precomputed frac_ge_* columns alone.".format(
                caller, RESIDUE_COLUMN, list(frame.columns)
            )
        )
    return frame[RESIDUE_COLUMN]


def fraction_at(frame: pd.DataFrame, plddt_min: float) -> pd.Series:
    """Fraction of each pocket's residues with pLDDT >= ``plddt_min``.

    Recomputed from ``residue_plddt`` for every row, so *any* cutoff works, not
    only the three the producer precomputed. Returned as ``float64``, indexed
    like ``frame``, so it can be used directly as a mask or assigned back as a
    column.

    A row with no residue values raises: the schema forbids one, and silently
    scoring it 0.0 would report "a terrible pocket" where the truth is "no data".
    """
    values = _residue_series(frame, "fraction_at")
    threshold = float(plddt_min)
    fractions: List[float] = []
    for raw in values:
        residues = parse_residue_plddt(raw)
        if not residues:
            raise QualityError(
                "a cavity_quality row carries no residue_plddt values. An "
                "unscoreable cavity belongs in score_cavity_quality.py's reject "
                "ledger, never in the quality table as a zero-filled row."
            )
        n_at_or_above = sum(1 for value in residues if value >= threshold)
        fractions.append(n_at_or_above / float(len(residues)))
    return pd.Series(fractions, index=frame.index, dtype="float64")


def passes(
    frame: pd.DataFrame,
    plddt_min: float,
    min_fraction: Optional[float],
) -> pd.Series:
    """``True`` where at least ``min_fraction`` of the pocket reaches ``plddt_min``.

    ``min_fraction`` is required and has no default here on purpose: "how much of
    the pocket must be confident" is the scientific choice that turns the filter
    on, and the CLI makes supplying it the opt-in. Passing ``None`` raises rather
    than defaulting to a threshold nobody chose.

    Returned as a plain ``bool`` Series indexed like ``frame``.
    """
    if min_fraction is None:
        raise QualityError(
            "passes() requires an explicit min_fraction. There is deliberately "
            "no default: supplying it is what turns pocket-quality filtering on, "
            "and a filter that quietly changes the size of a scientific dataset "
            "is the defect class this repository was rebuilt to eliminate."
        )
    fraction = float(min_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise QualityError(
            "min_fraction must be a fraction in [0, 1]; got {0!r}. (A percentage "
            "such as 80 would pass nothing.)".format(min_fraction)
        )
    return fraction_at(frame, plddt_min) >= fraction
