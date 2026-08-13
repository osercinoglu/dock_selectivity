"""Canonical protein identity for the dock_selectivity rebuild.

Why this module exists
----------------------
The 2025-06-29 campaign resolved proteins with

    gene_to_uniprot = dict(zip(uniprot_map_df['Mapped_Gene_Name'],
                               uniprot_map_df['Entry']))      # batch_dock.py:2034

which is *last-duplicate-wins*: 4,585 accessions collapsed onto 1,795 symbols
(mean collapse 2.55, max 9).  The accession that survived the collapse was the
structure that got docked, so a labelling bug selected the receptor.  See
``KNOWN_ISSUE_uniprot_gene_mapping.md`` and ``ASSESSMENT_blast_radius_2026_08_12.md``.

The law this module enforces (``FIX_PLAN.md`` section 1):

===========  ================================  ===========================
Axis         Canonical key                     Never a key
===========  ================================  ===========================
Protein      UniProt accession                 gene symbol, entry name, ``*_name``
===========  ================================  ===========================

1. Resolve once, at the ingest edge, from **Entrez Gene ID** (``node_2`` in
   ``1_drug_to_protein.tsv`` -- numeric on 8,568/8,568 edges, 1:1 with symbol).
2. Labels flow the other way only: ``accession -> primary_symbol``, for display.
   A label is never an input to selection.
3. A many-to-one map may not be inverted.

``resolve_entrez`` returns a :class:`Resolution` and never raises.  That is
deliberate: it makes ``gene_to_uniprot.get(gene)`` followed by
``if not uniprot_id: continue`` (``identify_required_structures.py:170``,
``generate_negative_samples.py:190`` -- both silent droppers) impossible to
write without also deciding where the rejected row is recorded.  Feed
:meth:`Resolution.as_reject_row` straight into ``unresolved_targets.csv``.

Invariants (``FIX_PLAN.md`` G1/G3/G4/G5/G7), all asserted by
:meth:`IdentityIndex.assert_self_consistent`, which runs by default at load:

* **G1** one row per accession;
* **G3** deterministic -- records are ordered by accession, never by ``set``
  iteration or input row order, so output is byte-identical across
  ``PYTHONHASHSEED`` values;
* **G4** ``primary_symbol in symbols``;
* **G5** ``entrez -> accession`` collapse factor **exactly 1** (the guarantee
  whose absence caused the incident);
* **G7** no isoform accessions (``P12345-2``).

Python 3.9 compatible: no PEP 604 unions at runtime, no ``match``.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

try:  # pragma: no cover - typing only
    from typing import Literal

    ResolutionStatus = Literal["OK", "AMBIGUOUS", "UNKNOWN", "OBSOLETE", "NO_KEY"]
except ImportError:  # pragma: no cover - Python < 3.8
    ResolutionStatus = str  # type: ignore[misc,assignment]

#: ``invariants`` is the canonical home of ``InvariantViolation``.  This module
#: re-exports it rather than declaring a second class of the same name: a stage
#: that writes ``except InvariantViolation`` must catch the whole package, not
#: the half it happened to import the name from.  ``checked_merge``, the schema
#: validators and ``IdentityIndex.load``'s SHA256 pin now raise one type.
from .invariants import InvariantViolation

__all__ = [
    "ACCESSION_RE",
    "IDENTITY_INDEX_COLUMNS",
    "RESOLUTION_STATUSES",
    "InvariantViolation",
    "Protein",
    "Resolution",
    "IdentityIndex",
    "is_accession",
    "file_sha256",
]

#: Official UniProtKB accession syntax, anchored.  Isoform suffixes
#: (``P12345-2``) and gene symbols do not match, by construction.
#: Verified against all 4,585 accessions in the campaign's mapping file: 0 misses.
ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)

#: Required parquet columns, in canonical order.
IDENTITY_INDEX_COLUMNS = (
    "accession",
    "entry_name",
    "primary_symbol",
    "symbols",
    "entrez_id",
    "reviewed",
)

RESOLUTION_STATUSES = ("OK", "AMBIGUOUS", "UNKNOWN", "OBSOLETE", "NO_KEY")

#: Strings that mean "no value" when they arrive from a CSV/TSV round trip.
#: ``'nan'`` is here because 11,806 job directories literally contain the token
#: ``nan`` where a gene symbol should be.  ``'NA'`` is deliberately *not* here:
#: treating a real value as missing is the same class of bug in the other
#: direction.
_MISSING_TOKENS = frozenset({"", "nan", "none", "null", "<na>", "nat", "-"})

_MAX_REPORTED = 10


# ``InvariantViolation`` is imported from :mod:`dock_selectivity.invariants`
# above.  It is raised here instead of ``assert`` so that ``python -O`` cannot
# switch the guarantee off; the message always names the offending values and
# the action to take.


# --------------------------------------------------------------------------
# small helpers (this module imports no pandas of its own; the frame paths
# import it locally.  ``.invariants`` above does import pandas, so importing
# this module is no longer free -- the helpers below are still written without
# it because they run per row.)
# --------------------------------------------------------------------------


def is_accession(value: Any) -> bool:
    """True iff *value* is a canonical UniProtKB accession (no isoform suffix)."""
    return isinstance(value, str) and bool(ACCESSION_RE.match(value.strip().upper()))


def file_sha256(path: Union[str, "os.PathLike[str]"]) -> str:
    """SHA256 of a file, streamed.  Used to pin the identity index."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_missing(value: Any) -> bool:
    """True for ``None``, NaN, ``pd.NA``/``pd.NaT`` and missing-ish strings.

    Written without importing pandas so this module stays cheap to import.
    ``pd.NA != pd.NA`` returns ``pd.NA``; truth-testing it raises ``TypeError``,
    which is caught here.  Containers (lists, numpy arrays) are never missing --
    they are checked before the NaN test, because ``array != array`` is an
    array, not a bool.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _MISSING_TOKENS
    if hasattr(value, "__len__"):  # list/tuple/dict/ndarray/Series
        return False
    try:
        if value != value:  # NaN is the only value unequal to itself
            return True
    except (TypeError, ValueError):  # pandas NA-likes are not boolean-comparable
        return True
    return False


def _clean_str(value: Any) -> Optional[str]:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _coerce_entrez(value: Any) -> Tuple[Optional[int], Optional[str]]:
    """Coerce *value* to a positive Entrez Gene ID.

    Returns ``(gene_id, error)``; exactly one of the two is ``None``.  Never
    raises -- callers include :meth:`IdentityIndex.resolve_entrez`, whose whole
    contract is that it does not raise.
    """
    if _is_missing(value):
        return None, "no Entrez gene id supplied (null/empty)"
    if isinstance(value, bool):
        return None, "Entrez gene id must be an integer, got a bool"
    if isinstance(value, int):
        gene_id = int(value)
    elif isinstance(value, float):
        if float(value).is_integer():
            gene_id = int(value)
        else:
            return None, "Entrez gene id {0!r} is not an integer".format(value)
    else:
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        if not re.match(r"^[+-]?\d+$", text):
            return None, "Entrez gene id {0!r} is not numeric".format(value)
        gene_id = int(text)
    if gene_id <= 0:
        return None, "Entrez gene id {0!r} is not a positive integer".format(value)
    return gene_id, None


# --------------------------------------------------------------------------
# value types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Protein:
    """One canonical protein.  ``accession`` is the only key."""

    accession: str
    entry_name: str
    primary_symbol: str
    symbols: Tuple[str, ...]
    entrez_id: Optional[int]
    reviewed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbols", tuple(self.symbols))

    def has_symbol(self, symbol: Any) -> bool:
        """True iff *symbol* is this protein's primary symbol or a synonym.

        This is the ``KNOWN_ISSUE.md`` regression check in method form: a
        ``(accession, gene)`` pair emitted by any stage must satisfy it.
        """
        text = _clean_str(symbol)
        if text is None:
            return False
        return text.upper() in {s.upper() for s in self.symbols}


@dataclass(frozen=True)
class Resolution:
    """The outcome of resolving one key.  Never ``None``, never a bare accession.

    ``status`` is one of :data:`RESOLUTION_STATUSES`.  ``reason`` is always
    populated when ``status != 'OK'`` -- enforced in ``__post_init__`` so an
    unexplained rejection cannot be constructed.
    """

    status: ResolutionStatus
    protein: Optional[Protein]
    reason: str = ""
    key: Optional[str] = None
    candidates: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if self.status not in RESOLUTION_STATUSES:
            raise ValueError(
                "Resolution.status must be one of {0}, got {1!r}".format(
                    ", ".join(RESOLUTION_STATUSES), self.status
                )
            )
        if self.status == "OK":
            if self.protein is None:
                raise ValueError("Resolution(status='OK') requires a protein")
        else:
            if self.protein is not None:
                raise ValueError(
                    "Resolution(status={0!r}) must not carry a protein; a "
                    "non-OK resolution is not resolvable".format(self.status)
                )
            if not self.reason.strip():
                raise ValueError(
                    "Resolution(status={0!r}) requires a non-empty reason; an "
                    "unexplained rejection is how 2,829 edges were lost "
                    "silently".format(self.status)
                )

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def unwrap(self) -> Protein:
        """The protein, or :class:`InvariantViolation` if this is not ``OK``."""
        if self.protein is None:
            raise InvariantViolation(
                "unwrap() on a {0} resolution for key {1!r}: {2}".format(
                    self.status, self.key, self.reason
                )
            )
        return self.protein

    def as_reject_row(self) -> Dict[str, Any]:
        """A row for ``unresolved_targets.csv``.  Never drop, always record."""
        return {
            "key": self.key,
            "status": self.status,
            "reason": self.reason,
            "candidates": ";".join(self.candidates),
        }


# --------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------


class IdentityIndex:
    """Accession-keyed protein index with asserted cardinality guarantees.

    Construct with :meth:`load` (parquet) or :meth:`from_frame` (DataFrame).
    Records are stored sorted by accession, so every derived structure and
    every report is independent of input row order and of ``PYTHONHASHSEED``.
    """

    def __init__(
        self,
        records: Iterable[Protein],
        source_path: Optional[str] = None,
        source_sha256: Optional[str] = None,
        obsolete_entrez: Optional[Mapping[int, str]] = None,
        validate: bool = True,
    ) -> None:
        # G3: total, input-order-independent ordering.  Never sort by anything
        # that a set or a dict could have reordered.
        self._records: Tuple[Protein, ...] = tuple(
            sorted(records, key=lambda p: p.accession)
        )
        self.source_path = source_path
        self.source_sha256 = source_sha256
        self._obsolete_entrez: Dict[int, str] = {
            int(k): str(v) for k, v in dict(obsolete_entrez or {}).items()
        }

        by_accession: Dict[str, Protein] = {}
        duplicate_accessions: Dict[str, int] = {}
        entrez_to_accessions: Dict[int, List[str]] = {}
        symbol_to_accessions: Dict[str, List[str]] = {}

        for protein in self._records:
            if protein.accession in by_accession:
                # Keep the collision instead of letting a dict swallow it:
                # last-wins over duplicates is the defect this file exists for.
                duplicate_accessions[protein.accession] = (
                    duplicate_accessions.get(protein.accession, 1) + 1
                )
            else:
                by_accession[protein.accession] = protein
            if protein.entrez_id is not None:
                entrez_to_accessions.setdefault(protein.entrez_id, []).append(
                    protein.accession
                )
            for symbol in protein.symbols:
                symbol_to_accessions.setdefault(symbol.upper(), []).append(
                    protein.accession
                )

        self._by_accession = by_accession
        self._duplicate_accessions = duplicate_accessions
        self._entrez_to_accessions: Dict[int, Tuple[str, ...]] = {
            key: tuple(value) for key, value in entrez_to_accessions.items()
        }
        self._symbol_to_accessions: Dict[str, Tuple[str, ...]] = {
            key: tuple(value) for key, value in symbol_to_accessions.items()
        }

        if validate:
            self.assert_self_consistent()

    # -- construction ------------------------------------------------------

    @classmethod
    def from_frame(
        cls,
        df: "Any",
        source_path: Optional[str] = None,
        source_sha256: Optional[str] = None,
        obsolete_entrez: Optional[Mapping[int, str]] = None,
        validate: bool = True,
    ) -> "IdentityIndex":
        """Build from a DataFrame carrying :data:`IDENTITY_INDEX_COLUMNS`.

        Missing columns raise :class:`InvariantViolation` -- they are never
        defaulted, because a defaulted identity column is a fabricated
        identity.  ``symbols`` must be a list/tuple/array per row; a bare
        string raises rather than being split into characters.
        """
        missing = [c for c in IDENTITY_INDEX_COLUMNS if c not in list(df.columns)]
        if missing:
            raise InvariantViolation(
                "identity index is missing required column(s): {0}. Present: {1}. "
                "Refusing to default them -- rebuild the index with "
                "build_identity_index.py.".format(
                    ", ".join(missing), ", ".join(str(c) for c in df.columns)
                )
            )
        records = [
            cls._record_from_row(row, position)
            for position, row in enumerate(df[list(IDENTITY_INDEX_COLUMNS)].to_dict("records"))
        ]
        return cls(
            records,
            source_path=source_path,
            source_sha256=source_sha256,
            obsolete_entrez=obsolete_entrez,
            validate=validate,
        )

    @staticmethod
    def _record_from_row(row: Mapping[str, Any], position: int) -> Protein:
        where = "identity index row {0}".format(position)

        accession = _clean_str(row["accession"])
        if accession is None:
            raise InvariantViolation("{0}: accession is null/empty".format(where))
        accession = accession.upper()

        entry_name = _clean_str(row["entry_name"])
        if entry_name is None:
            raise InvariantViolation(
                "{0} ({1}): entry_name is null/empty".format(where, accession)
            )

        raw_symbols = row["symbols"]
        if isinstance(raw_symbols, str):
            raise InvariantViolation(
                "{0} ({1}): symbols must be a list of strings, got the bare string "
                "{2!r}. A string would be silently split into characters; write "
                "the column as list<str>.".format(where, accession, raw_symbols)
            )
        if _is_missing(raw_symbols):
            raw_symbols = []
        try:
            symbol_list = [
                s for s in (_clean_str(item) for item in list(raw_symbols)) if s
            ]
        except TypeError:
            raise InvariantViolation(
                "{0} ({1}): symbols is not iterable: {2!r}".format(
                    where, accession, raw_symbols
                )
            )

        primary_symbol = _clean_str(row["primary_symbol"])
        if primary_symbol is None:
            raise InvariantViolation(
                "{0} ({1}): primary_symbol is null/empty; an accession with no "
                "displayable label must still carry one (use the accession "
                "itself) -- never a blank that a join can match".format(
                    where, accession
                )
            )

        entrez_id, entrez_error = _coerce_entrez(row["entrez_id"])
        if entrez_error is not None and not _is_missing(row["entrez_id"]):
            raise InvariantViolation(
                "{0} ({1}): {2}".format(where, accession, entrez_error)
            )

        reviewed = row["reviewed"]
        if _is_missing(reviewed):
            raise InvariantViolation(
                "{0} ({1}): reviewed is null; review status is a tie-break input "
                "and may not be guessed".format(where, accession)
            )
        if isinstance(reviewed, str):
            reviewed_bool = reviewed.strip().lower() in {"true", "t", "yes", "y", "1"}
        else:
            reviewed_bool = bool(reviewed)

        return Protein(
            accession=accession,
            entry_name=entry_name,
            primary_symbol=primary_symbol,
            symbols=tuple(symbol_list),
            entrez_id=entrez_id,
            reviewed=reviewed_bool,
        )

    @classmethod
    def load(
        cls,
        path: Union[str, "os.PathLike[str]"],
        expect_sha256: Optional[str] = None,
        obsolete_entrez: Optional[Mapping[int, str]] = None,
        validate: bool = True,
    ) -> "IdentityIndex":
        """Read ``identity_index.parquet``.

        A missing file raises ``FileNotFoundError``; there is no fallback to
        another mapping source.  ``expect_sha256`` pins the artifact: a
        mismatch raises :class:`InvariantViolation` so a stage cannot run
        against an index other than the one its plan was built from.
        """
        path_str = os.fspath(path)
        if not os.path.exists(path_str):
            raise FileNotFoundError(
                "identity index not found: {0}. Build it with "
                "build_identity_index.py; there is no fallback identity source "
                "(a stage that picks its own universe is a different "
                "experiment).".format(path_str)
            )
        digest = file_sha256(path_str)
        if expect_sha256 is not None and digest != expect_sha256.strip().lower():
            raise InvariantViolation(
                "identity index SHA256 mismatch for {0}: expected {1}, found {2}. "
                "Downstream artifacts were planned against the expected index; "
                "rebuild them or point at the pinned index.".format(
                    path_str, expect_sha256.strip().lower(), digest
                )
            )
        import pandas as pd  # local import: keeps module import cheap

        frame = pd.read_parquet(path_str)
        return cls.from_frame(
            frame,
            source_path=path_str,
            source_sha256=digest,
            obsolete_entrez=obsolete_entrez,
            validate=validate,
        )

    def to_frame(self) -> "Any":
        """Canonical DataFrame: :data:`IDENTITY_INDEX_COLUMNS`, sorted by accession.

        Deterministic by construction, so serialising it twice under different
        ``PYTHONHASHSEED`` values yields identical bytes.
        """
        import pandas as pd

        return pd.DataFrame(
            {
                "accession": [p.accession for p in self._records],
                "entry_name": [p.entry_name for p in self._records],
                "primary_symbol": [p.primary_symbol for p in self._records],
                "symbols": [list(p.symbols) for p in self._records],
                "entrez_id": pd.array(
                    [p.entrez_id for p in self._records], dtype="Int64"
                ),
                "reviewed": [p.reviewed for p in self._records],
            },
            columns=list(IDENTITY_INDEX_COLUMNS),
        )

    # -- lookups -----------------------------------------------------------

    @property
    def records(self) -> Tuple[Protein, ...]:
        """All proteins, ordered by accession."""
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[Protein]:
        return iter(self._records)

    def __contains__(self, accession: Any) -> bool:
        text = _clean_str(accession)
        return text is not None and text.upper() in self._by_accession

    def accessions(self) -> Tuple[str, ...]:
        """Every accession, sorted.  This is the token set for legacy job-name parsing."""
        return tuple(p.accession for p in self._records)

    def by_accession(self, acc: Any) -> Protein:
        """Look up by accession.  Raises ``KeyError`` if absent."""
        text = _clean_str(acc)
        if text is None:
            raise KeyError(
                "by_accession() called with a null/empty accession; the caller "
                "must record the rejected row rather than look up nothing"
            )
        key = text.upper()
        if key in self._by_accession:
            return self._by_accession[key]
        if "-" in key:
            raise KeyError(
                "{0!r} looks like an isoform accession; the index is keyed on "
                "canonical accessions only. Strip the isoform suffix at ingest, "
                "explicitly.".format(text)
            )
        raise KeyError(
            "accession {0!r} is not in the identity index ({1} accessions){2}".format(
                text,
                len(self._by_accession),
                "" if self.source_path is None else " loaded from " + self.source_path,
            )
        )

    def by_entrez(self, gene_id: Any) -> Protein:
        """Look up by Entrez Gene ID.  Raises ``KeyError`` if absent.

        An Entrez ID carrying more than one accession raises
        :class:`InvariantViolation`, not ``KeyError``: returning one of them
        would be exactly the last-wins collapse this package exists to prevent,
        and a caller's ``except KeyError`` must not swallow a broken index.
        """
        key, error = _coerce_entrez(gene_id)
        if key is None:
            raise KeyError("cannot look up Entrez gene id {0!r}: {1}".format(gene_id, error))
        accessions = self._entrez_to_accessions.get(key)
        if not accessions:
            raise KeyError(
                "Entrez gene id {0} is not in the identity index "
                "({1} Entrez ids)".format(key, len(self._entrez_to_accessions))
            )
        if len(set(accessions)) > 1:
            raise InvariantViolation(
                "Entrez {0} maps to {1} accessions ({2}); collapse factor must be "
                "exactly 1. Rebuild the index -- do not pick one.".format(
                    key, len(set(accessions)), ", ".join(sorted(set(accessions)))
                )
            )
        return self._by_accession[accessions[0]]

    def resolve_entrez(self, gene_id: Any) -> Resolution:
        """Resolve an Entrez Gene ID.  **Never raises**; the caller must branch.

        Statuses: ``NO_KEY`` (null/non-numeric input), ``OBSOLETE`` (retired id,
        supplied via ``obsolete_entrez``), ``UNKNOWN`` (not in the index),
        ``AMBIGUOUS`` (more than one accession -- only possible on an index that
        skipped validation), ``OK``.
        """
        key_repr = "" if gene_id is None else str(gene_id)
        key, error = _coerce_entrez(gene_id)
        if key is None:
            return Resolution(
                status="NO_KEY",
                protein=None,
                reason=error or "no Entrez gene id supplied",
                key=key_repr,
            )
        key_repr = str(key)
        if key in self._obsolete_entrez:
            return Resolution(
                status="OBSOLETE",
                protein=None,
                reason="Entrez {0} is retired: {1}".format(
                    key, self._obsolete_entrez[key]
                ),
                key=key_repr,
            )
        accessions = self._entrez_to_accessions.get(key)
        if not accessions:
            return Resolution(
                status="UNKNOWN",
                protein=None,
                reason=(
                    "Entrez {0} is not in the identity index ({1} Entrez ids{2}); "
                    "it is either non-human, retired, or the index needs "
                    "rebuilding".format(
                        key,
                        len(self._entrez_to_accessions),
                        ""
                        if self.source_path is None
                        else " from " + self.source_path,
                    )
                ),
                key=key_repr,
            )
        distinct = sorted(set(accessions))
        if len(distinct) > 1:
            return Resolution(
                status="AMBIGUOUS",
                protein=None,
                reason=(
                    "Entrez {0} maps to {1} accessions; collapse factor must be 1 "
                    "(G5). Rebuild the index; do not choose one here.".format(
                        key, len(distinct)
                    )
                ),
                key=key_repr,
                candidates=tuple(distinct),
            )
        return Resolution(
            status="OK",
            protein=self._by_accession[distinct[0]],
            reason="",
            key=key_repr,
            candidates=(distinct[0],),
        )

    def label(self, acc: Any) -> str:
        """``accession -> primary symbol``, for display only.

        The only direction a symbol is ever allowed to travel.  Raises
        ``KeyError`` for an unknown accession rather than returning a blank
        that a downstream join could match.
        """
        return self.by_accession(acc).primary_symbol

    def by_symbol_ambiguous(self, symbol: Any) -> Tuple[Protein, ...]:
        """Every protein carrying *symbol* as primary symbol or synonym.

        **Audit only.**  The name is deliberately awkward: a symbol is not a
        key, and this method returns a tuple precisely so that a caller cannot
        pretend it got one answer.  Never use it to choose a structure.

        Ordering is total (reviewed first, then accession), so the result does
        not depend on input row order or hash seed.  Unknown symbol -> ``()``.
        """
        text = _clean_str(symbol)
        if text is None:
            return ()
        accessions = self._symbol_to_accessions.get(text.upper())
        if not accessions:
            return ()
        proteins = [self._by_accession[a] for a in sorted(set(accessions))]
        proteins.sort(key=lambda p: (0 if p.reviewed else 1, p.accession))
        return tuple(proteins)

    # -- invariants --------------------------------------------------------

    def assert_self_consistent(self) -> None:
        """Enforce G1, G4, G5, G7.  Raises :class:`InvariantViolation`.

        All problems are collected and reported together -- fixing an index one
        exception at a time is how a build loop turns into a guessing game.
        """
        problems: List[str] = []

        if self._duplicate_accessions:
            worst = sorted(
                self._duplicate_accessions.items(), key=lambda kv: (-kv[1], kv[0])
            )
            problems.append(
                "G1 one row per accession: {0} accession(s) appear more than once "
                "({1}). A dict keyed on accession would silently keep the last "
                "one. De-duplicate in the builder with an explicit, total "
                "tie-break.".format(
                    len(self._duplicate_accessions),
                    ", ".join(
                        "{0} x{1}".format(a, n) for a, n in worst[:_MAX_REPORTED]
                    ),
                )
            )

        bad_shape = [
            p.accession for p in self._records if not ACCESSION_RE.match(p.accession)
        ]
        isoforms = [p.accession for p in self._records if "-" in p.accession]
        if isoforms:
            problems.append(
                "G7 no isoform accessions: {0} row(s) carry an isoform-style "
                "accession ({1}). Collapse isoforms onto their canonical "
                "accession at build time.".format(
                    len(isoforms), ", ".join(sorted(isoforms)[:_MAX_REPORTED])
                )
            )
        non_isoform_bad = sorted(set(bad_shape) - set(isoforms))
        if non_isoform_bad:
            problems.append(
                "accession syntax: {0} row(s) are not valid UniProtKB accessions "
                "({1}). A gene symbol or entry name in the accession column is "
                "the original defect.".format(
                    len(non_isoform_bad), ", ".join(non_isoform_bad[:_MAX_REPORTED])
                )
            )

        no_symbols = [p.accession for p in self._records if not p.symbols]
        if no_symbols:
            problems.append(
                "G4 symbols: {0} row(s) have an empty symbols list ({1}); every "
                "row must at least list its own primary symbol.".format(
                    len(no_symbols), ", ".join(sorted(no_symbols)[:_MAX_REPORTED])
                )
            )
        mismatched = [
            "{0} (primary_symbol={1!r} not in {2})".format(
                p.accession, p.primary_symbol, list(p.symbols)
            )
            for p in self._records
            if p.symbols and p.primary_symbol.upper() not in {s.upper() for s in p.symbols}
        ]
        if mismatched:
            problems.append(
                "G4 primary_symbol in symbols: {0} row(s) fail the KNOWN_ISSUE "
                "regression assertion ({1}). This is the exact signature of "
                "broadcasting a queried name onto unrelated hits.".format(
                    len(mismatched), "; ".join(mismatched[:_MAX_REPORTED])
                )
            )

        collapsing = {
            gene_id: sorted(set(accs))
            for gene_id, accs in self._entrez_to_accessions.items()
            if len(set(accs)) > 1
        }
        if collapsing:
            worst = sorted(collapsing.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            max_collapse = len(worst[0][1])
            problems.append(
                "G5 entrez -> accession collapse factor must be exactly 1: {0} "
                "Entrez id(s) map to more than one accession, max collapse {1} "
                "({2}). The retired mapping had mean collapse 2.55 / max 9 and it "
                "chose the receptor that got docked.".format(
                    len(collapsing),
                    max_collapse,
                    "; ".join(
                        "{0} -> {1}".format(gid, ", ".join(accs))
                        for gid, accs in worst[:_MAX_REPORTED]
                    ),
                )
            )

        if problems:
            raise InvariantViolation(
                "identity index{0} failed {1} invariant(s):\n  - {2}".format(
                    "" if self.source_path is None else " " + self.source_path,
                    len(problems),
                    "\n  - ".join(problems),
                )
            )

    def cardinality_report(self) -> Dict[str, Any]:
        """Cardinality summary.  ``max_collapse`` is the headline number.

        Keys are emitted in a fixed order and every value is a plain Python
        scalar, so ``json.dumps(report)`` is byte-stable across runs.
        """
        entrez_sizes = [
            len(set(accs)) for accs in self._entrez_to_accessions.values()
        ]
        symbol_sizes = [
            len(set(accs)) for accs in self._symbol_to_accessions.values()
        ]
        n_with_entrez = sum(1 for p in self._records if p.entrez_id is not None)
        mean_collapse = (
            round(sum(entrez_sizes) / float(len(entrez_sizes)), 4)
            if entrez_sizes
            else 0.0
        )
        return {
            "n_records": len(self._records),
            "n_accessions": len(self._by_accession),
            "n_duplicate_accessions": len(self._duplicate_accessions),
            "n_isoform_accessions": sum(
                1 for p in self._records if "-" in p.accession
            ),
            "n_entrez": len(self._entrez_to_accessions),
            "n_records_with_entrez": n_with_entrez,
            "n_records_without_entrez": len(self._records) - n_with_entrez,
            "max_collapse": max(entrez_sizes) if entrez_sizes else 0,
            "mean_collapse": mean_collapse,
            "n_entrez_collapsing": sum(1 for size in entrez_sizes if size > 1),
            "n_symbols": len(self._symbol_to_accessions),
            "n_ambiguous_symbols": sum(1 for size in symbol_sizes if size > 1),
            "max_symbol_collapse": max(symbol_sizes) if symbol_sizes else 0,
            "n_reviewed": sum(1 for p in self._records if p.reviewed),
            "n_unreviewed": sum(1 for p in self._records if not p.reviewed),
            "source_sha256": self.source_sha256,
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return "IdentityIndex(n={0}, n_entrez={1}, source={2!r})".format(
            len(self._records), len(self._entrez_to_accessions), self.source_path
        )
