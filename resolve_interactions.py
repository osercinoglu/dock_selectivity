#!/usr/bin/env python3
"""REBUILD_PLAN Phase 1 -- resolve the corrected drug-target interaction set.

Reads ``1_drug_to_protein.tsv`` and emits ``manifests/interactions_resolved.csv``
(schema ``interactions_resolved``) plus a rejects ledger in which **every** input
edge that did not resolve is recorded with a reason.

What this stage exists to prevent
---------------------------------

The 2025-06-29 campaign resolved proteins through the gene *symbol* column::

    gene_to_uniprot = dict(zip(uniprot_map_df['Mapped_Gene_Name'],
                               uniprot_map_df['Entry']))     # batch_dock.py:2034
    uniprot_id = gene_to_uniprot.get(gene)                    # ...:170, ...:190
    if not uniprot_id:
        continue                                              # silent drop

Three separate defects in five lines:

1. ``dict(zip(...))`` is **last-duplicate-wins**. 4,585 accessions collapsed onto
   1,795 symbols (mean 2.55, max 9), and the accession that survived the collapse
   became the receptor that was docked. ``AVPR2`` resolved to ``Q8NCT1``, which is
   ``ARRDC4`` -- a different protein.
2. Resolution ran on a **name**. ``node_2`` of the same file is the NCBI Entrez
   Gene ID, numeric on 8,568/8,568 edges and strictly 1:1 with the symbol. It was
   never used.
3. ``if not uniprot_id: continue`` dropped rows with no ledger. The 237 edges whose
   ``node_2_name`` is null were *structurally unreachable* by a symbol-keyed
   resolver: they have no symbol at all, yet every one of them carries an Entrez id.

This module resolves the protein axis **only** through
:meth:`dock_selectivity.identity.IdentityIndex.resolve_entrez`, which returns a
:class:`~dock_selectivity.identity.Resolution` rather than an ``Optional[str]``, so
a rejected edge cannot be dropped without also deciding where it is recorded.
``node_2_name`` is read for the rejects ledger and for nothing else; the ``gene``
column of the output is written by ``index.label(accession)`` and is therefore a
label *derived from* the key, never an input to selection.

The compound axis has the same shape and was never verified at all: 222 of the
8,568 rows carry a name slug (``clorprenaline``, ``neridronic-acid``, ``SN-38``)
in ``node_1`` instead of a DrugBank ID. :class:`CompoundIndex` resolves those by
**normalised exact** matching against DrugBank's own vocabulary, and returns
``AMBIGUOUS`` -- never a pick -- when a name belongs to more than one entry.
``L-Arginine`` is the live example: it is the common name of ``DB17289`` and a
synonym of ``DB00125``. Choosing either would be a guess, so its 7 edges are
rejected with a reason.

Guarantees asserted before anything is written
----------------------------------------------

===============================  ==========================================
``n_in == n_resolved + n_rejected``   nothing vanishes (FIX_PLAN G2)
``entrez -> accession`` collapse 1    the incident's signature (G5)
``gene in index.by_accession(acc).symbols`` for 100% of rows
                                      the ``KNOWN_ISSUE.md`` regression check
``(drugbank_id, accession)`` unique   the output's declared primary key
the small-molecule join adds no rows  ``checked_merge`` (the +410,087 fan-out)
byte-identical output across runs     no ``set`` iteration, total sort keys
===============================  ==========================================

Usage
-----

::

    python resolve_interactions.py \\
        --edges /opt/data/multiscale_interactome_data/1_drug_to_protein.tsv \\
        --index $NEW/identity/identity_index.parquet \\
        --drugs /opt/data/drugbank/small_molecule_drug_links.csv \\
        --out   $NEW/manifests/interactions_resolved.csv \\
        --rejects $NEW/identity/unresolved_targets.csv

Python 3.9 compatible: no PEP 604 unions at runtime, no ``match`` statements.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:  # `python $REPO/resolve_interactions.py` from anywhere
    sys.path.insert(0, _REPO_ROOT)

from dock_selectivity.identity import IdentityIndex, Resolution  # noqa: E402
from dock_selectivity.identity import (  # noqa: E402
    InvariantViolation as _IdentityInvariantViolation,
)
from dock_selectivity.invariants import (  # noqa: E402
    InvariantViolation,
    assert_collapse_factor_one,
    assert_self_consistent_labels,
    assert_total_accounting,
    assert_unique_key,
    write_invariant_report,
)
from dock_selectivity.joins import checked_merge_with_report  # noqa: E402
from dock_selectivity.schemas import SchemaViolation, coerce, validate, write_checked  # noqa: E402

__all__ = [
    "DEFAULT_VOCABULARY",
    "DRUGBANK_ID_RE",
    "COMPOUND_STATUSES",
    "COMPOUND_MATCH_TIERS",
    "REJECT_PRECEDENCE",
    "REJECT_COLUMNS",
    "INVARIANT_ERRORS",
    "SMALL_MOLECULE_ID_COLUMN",
    "SMALL_MOLECULE_TYPE_COLUMN",
    "SMALL_MOLECULE_TYPE_VALUE",
    "normalize_name",
    "is_drugbank_id",
    "Compound",
    "CompoundResolution",
    "CompoundIndex",
    "ResolveResult",
    "read_edges",
    "load_small_molecule_ids",
    "resolve_interactions",
    "write_rejects",
    "build_arg_parser",
    "main",
]

logger = logging.getLogger("resolve_interactions")

#: DrugBank's own vocabulary export. On this machine the path is a *directory*
#: holding ``drugbank vocabulary.csv``; :meth:`CompoundIndex.load` accepts either
#: the directory or the file, and refuses to guess when a directory holds more
#: than one CSV.
DEFAULT_VOCABULARY = "/opt/data/drugbank/drugbank_all_drugbank_vocabulary.csv"

#: A DrugBank accession is ``DB`` + exactly five digits (verified: 16,581/16,581
#: vocabulary entries match, 0 exceptions).
DRUGBANK_ID_RE = re.compile(r"^DB\d{5}$")

#: Column names in ``1_drug_to_protein.tsv``.
EDGE_DRUG_COLUMN = "node_1"
EDGE_ENTREZ_COLUMN = "node_2"
EDGE_DRUG_NAME_COLUMN = "node_1_name"
EDGE_SYMBOL_COLUMN = "node_2_name"
EDGE_DRUG_TYPE_COLUMN = "node_1_type"
EDGE_PROTEIN_TYPE_COLUMN = "node_2_type"

#: Required on input. ``node_1_name``/``node_2_name`` are diagnostics only: they
#: appear in the rejects ledger and are *never* consulted for resolution.
REQUIRED_EDGE_COLUMNS = (EDGE_DRUG_COLUMN, EDGE_ENTREZ_COLUMN)
OPTIONAL_EDGE_COLUMNS = (
    EDGE_DRUG_NAME_COLUMN,
    EDGE_SYMBOL_COLUMN,
    EDGE_DRUG_TYPE_COLUMN,
    EDGE_PROTEIN_TYPE_COLUMN,
)

EXPECTED_DRUG_TYPE = "drug"
EXPECTED_PROTEIN_TYPE = "protein"

#: Columns of ``small_molecule_drug_links.csv`` this stage depends on. Both are
#: required: a missing column is an error, never a default (``batch_dock.py:448``
#: defaulted one to ``''`` and produced 10,698 ``*_nan_*`` job directories).
SMALL_MOLECULE_ID_COLUMN = "DrugBank ID"
SMALL_MOLECULE_TYPE_COLUMN = "Drug Type"
SMALL_MOLECULE_TYPE_VALUE = "SmallMoleculeDrug"

COMPOUND_STATUSES = ("OK", "AMBIGUOUS", "UNKNOWN", "NO_KEY")

#: Where a compound token matched, in the order the tiers are tried. The first two
#: are *identifier* lookups (exact, no normalisation beyond case); the last two are
#: normalised-exact *name* lookups and are pooled, so a token that names two
#: different entries is AMBIGUOUS rather than resolved by tier precedence.
COMPOUND_MATCH_TIERS = ("drugbank_id", "accession_number", "common_name", "synonym")

#: The two tiers that match a *name* rather than an identifier. They are pooled,
#: never ranked (see :class:`CompoundIndex`).
_NAME_TIERS = ("common_name", "synonym")

#: Every way an edge can fail, in a total precedence order. An edge that fails on
#: both axes reports the first of these that applies as its ``status`` and names
#: every failure in its ``reason``.
REJECT_PRECEDENCE = (
    "WRONG_EDGE_TYPE",
    "NO_KEY",
    "NO_DRUG_KEY",
    "UNKNOWN_ENTREZ",
    "OBSOLETE_ENTREZ",
    "AMBIGUOUS_ENTREZ",
    "UNKNOWN_DRUG",
    "AMBIGUOUS_DRUG",
    "NOT_SMALL_MOLECULE",
)

#: ``Resolution.status`` (protein axis) -> reject status. ``NO_KEY`` keeps its
#: name because that is the vocabulary FIX_PLAN §1 declares.
_PROTEIN_STATUS_TO_REJECT = {
    "NO_KEY": "NO_KEY",
    "UNKNOWN": "UNKNOWN_ENTREZ",
    "OBSOLETE": "OBSOLETE_ENTREZ",
    "AMBIGUOUS": "AMBIGUOUS_ENTREZ",
}

#: ``CompoundResolution.status`` -> reject status.
_COMPOUND_STATUS_TO_REJECT = {
    "NO_KEY": "NO_DRUG_KEY",
    "UNKNOWN": "UNKNOWN_DRUG",
    "AMBIGUOUS": "AMBIGUOUS_DRUG",
}

#: Columns of the rejects ledger (``identity/unresolved_targets.csv``).
REJECT_COLUMNS = (
    "edge_row",
    "node_1",
    "node_2",
    "node_1_name",
    "node_2_name",
    "axis",
    "status",
    "reason",
    "candidates",
)

#: ``selected_by`` uses ';' internally so the field never needs CSV quoting.
_SELECTED_BY_SEP = "; "

_SEPARATOR_RE = re.compile(r"[\W_]+", re.UNICODE)

_MAX_REPORTED = 10

#: ``identity.py`` currently defines its own ``InvariantViolation`` rather than
#: importing the canonical one from ``invariants.py``, so
#: ``identity.InvariantViolation is not invariants.InvariantViolation`` and a
#: single ``except`` clause catches only one of them. A pinned-SHA256 mismatch
#: raises identity's; ``checked_merge`` and the schemas raise the canonical one.
#: Both are caught here. When ``identity.py`` adopts
#: ``from .invariants import InvariantViolation`` this tuple collapses to one
#: entry and nothing else changes.
INVARIANT_ERRORS = (InvariantViolation, _IdentityInvariantViolation)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _text(value: Any) -> Optional[str]:
    """Strip to a non-empty string, or ``None``.

    Deliberately does *not* treat the literal strings ``'nan'``/``'NA'`` as
    missing: ``NA`` is a real gene symbol and ``nan`` is a real token in 11,806
    old job directories. Only a true null or an empty field is missing.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):  # pragma: no cover - non-scalar input
            pass
        value = str(value)
    text = value.strip()
    return text or None


def normalize_name(value: Any) -> Optional[str]:
    """Casefold, de-accent and collapse separators; ``None`` if nothing remains.

    This is the *only* transformation applied before a compound name is matched,
    and matching is exact afterwards. There is no edit distance, no prefix match
    and no token-set comparison anywhere in this module -- a fuzzy resolver is
    how ``service.get_df(gene_name)`` broadcast one queried name onto every
    unrelated hit.

    >>> normalize_name('neridronic-acid')
    'neridronic acid'
    >>> normalize_name('SN-38')
    'sn 38'
    >>> normalize_name('butorphanol-(+)-tartrate')
    'butorphanol tartrate'
    >>> normalize_name('Cétuximab')
    'cetuximab'
    >>> normalize_name('---') is None
    True
    """
    text = _text(value)
    if text is None:
        return None
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    collapsed = _SEPARATOR_RE.sub(" ", stripped.casefold()).strip()
    return collapsed or None


def is_drugbank_id(value: Any) -> bool:
    """True iff *value* has DrugBank accession syntax (``DB`` + five digits)."""
    text = _text(value)
    return text is not None and bool(DRUGBANK_ID_RE.match(text.upper()))


def _join_reason(parts: Sequence[str]) -> str:
    return " | ".join(p for p in parts if p)


_REJECT_ORDER = {status: i for i, status in enumerate(REJECT_PRECEDENCE)}


def _primary_status(statuses: Sequence[str]) -> str:
    """Lowest-index member of :data:`REJECT_PRECEDENCE` present in *statuses*.

    A total function of the failure set, so an edge that fails on both axes
    always reports the same primary status regardless of the order the checks
    happened to run in.
    """
    return sorted(
        statuses, key=lambda s: (_REJECT_ORDER.get(s, len(_REJECT_ORDER)), s)
    )[0]


# --------------------------------------------------------------------------
# compound axis
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Compound:
    """One DrugBank vocabulary entry. ``drugbank_id`` is the only key."""

    drugbank_id: str
    common_name: str
    accession_numbers: Tuple[str, ...] = ()
    synonyms: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "accession_numbers", tuple(self.accession_numbers))
        object.__setattr__(self, "synonyms", tuple(self.synonyms))


@dataclass(frozen=True)
class CompoundResolution:
    """Outcome of resolving one compound token. Never ``None``, never a bare id.

    Mirrors :class:`dock_selectivity.identity.Resolution` on purpose: the two
    axes of an edge are resolved by the same discipline, so neither can be the
    one that quietly uses ``.get()`` and ``continue``.
    """

    status: str
    drugbank_id: Optional[str]
    reason: str = ""
    key: Optional[str] = None
    matched_by: Optional[str] = None
    candidates: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if self.status not in COMPOUND_STATUSES:
            raise ValueError(
                "CompoundResolution.status must be one of {0}, got {1!r}".format(
                    ", ".join(COMPOUND_STATUSES), self.status
                )
            )
        if self.status == "OK":
            if self.drugbank_id is None:
                raise ValueError("CompoundResolution(status='OK') requires a drugbank_id")
            if self.matched_by not in COMPOUND_MATCH_TIERS:
                raise ValueError(
                    "CompoundResolution(status='OK') must say how it matched; "
                    "matched_by must be one of {0}, got {1!r}".format(
                        ", ".join(COMPOUND_MATCH_TIERS), self.matched_by
                    )
                )
        else:
            if self.drugbank_id is not None:
                raise ValueError(
                    "CompoundResolution(status={0!r}) must not carry a "
                    "drugbank_id".format(self.status)
                )
            if not self.reason.strip():
                raise ValueError(
                    "CompoundResolution(status={0!r}) requires a non-empty "
                    "reason; an unexplained rejection is how 222 name-slug rows "
                    "were lost without a ledger".format(self.status)
                )

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def unwrap(self) -> str:
        if self.drugbank_id is None:
            raise InvariantViolation(
                "unwrap() on a {0} compound resolution for key {1!r}: {2}".format(
                    self.status, self.key, self.reason
                ),
                name="CompoundResolution.unwrap",
            )
        return self.drugbank_id

    def as_reject_row(self) -> Dict[str, Any]:
        """A row for the rejects ledger. Never drop, always record."""
        return {
            "key": self.key,
            "status": self.status,
            "reason": self.reason,
            "candidates": ";".join(self.candidates),
        }


class CompoundIndex:
    """DrugBank vocabulary as a resolver that refuses to guess.

    Four match tiers, tried in the order of :data:`COMPOUND_MATCH_TIERS`:

    1. ``drugbank_id`` -- the token *is* a primary DrugBank accession;
    2. ``accession_number`` -- the token is a secondary/legacy accession
       (``BIOD00024``), i.e. still an identifier;
    3. ``common_name`` / 4. ``synonym`` -- **pooled**, and matched only after
       :func:`normalize_name`.

    The two name tiers are pooled rather than ranked, so a token that names one
    entry *and* is a synonym of another resolves to ``AMBIGUOUS`` rather than to
    whichever tier was tried first. ``L-Arginine`` is exactly that case
    (``DB17289`` by common name, ``DB00125`` by synonym) and it is a real row of
    ``1_drug_to_protein.tsv``. Pass ``prefer_common_name=True`` to declare, as an
    explicit and logged choice, that DrugBank's own preferred label outranks an
    alias; it is off by default because ranking is a choice and choices about
    identity are what this rebuild exists to make visible.

    Every internal mapping is ``name -> tuple of ids``, never ``name -> id``: a
    ``dict(zip(names, ids))`` would keep the last entry per name and the loss
    would be unobservable afterwards (``batch_dock.py:2034``, verbatim).
    """

    def __init__(
        self,
        records: Iterable[Compound],
        source_path: Optional[str] = None,
        prefer_common_name: bool = False,
        validate: bool = True,
    ) -> None:
        self._records: Tuple[Compound, ...] = tuple(
            sorted(records, key=lambda c: c.drugbank_id)
        )
        self.source_path = source_path
        self.prefer_common_name = bool(prefer_common_name)

        by_id: Dict[str, Compound] = {}
        duplicate_ids: Dict[str, int] = {}
        accession_to_ids: Dict[str, List[str]] = {}
        common_to_ids: Dict[str, List[str]] = {}
        synonym_to_ids: Dict[str, List[str]] = {}

        for compound in self._records:
            if compound.drugbank_id in by_id:
                # Capture the collision instead of letting a dict swallow it.
                duplicate_ids[compound.drugbank_id] = (
                    duplicate_ids.get(compound.drugbank_id, 1) + 1
                )
            else:
                by_id[compound.drugbank_id] = compound
            for accession in compound.accession_numbers:
                accession_to_ids.setdefault(accession.upper(), []).append(
                    compound.drugbank_id
                )
            name = normalize_name(compound.common_name)
            if name:
                common_to_ids.setdefault(name, []).append(compound.drugbank_id)
            for synonym in compound.synonyms:
                normalised = normalize_name(synonym)
                if normalised:
                    synonym_to_ids.setdefault(normalised, []).append(
                        compound.drugbank_id
                    )

        self._by_id = by_id
        self._duplicate_ids = duplicate_ids
        # Sorted, de-duplicated tuples: every report and every AMBIGUOUS
        # candidate list is then independent of input order and of PYTHONHASHSEED.
        self._accession_to_ids = _freeze(accession_to_ids)
        self._common_to_ids = _freeze(common_to_ids)
        self._synonym_to_ids = _freeze(synonym_to_ids)

        if validate:
            self.assert_self_consistent()

    # -- construction ------------------------------------------------------

    @staticmethod
    def resolve_vocabulary_path(path: Union[str, "os.PathLike[str]"]) -> str:
        """Accept the vocabulary file, or a directory holding exactly one CSV.

        The export on this machine unpacks to a *directory* named
        ``drugbank_all_drugbank_vocabulary.csv`` containing
        ``drugbank vocabulary.csv``. A directory with zero or several CSVs raises
        and names what it found -- picking one would be a guess.
        """
        path_str = os.fspath(path)
        if not os.path.exists(path_str):
            raise FileNotFoundError(
                "DrugBank vocabulary not found: {0}. It is required to resolve "
                "the 222 name-slug rows of node_1; there is no fallback name "
                "source. Pass --vocabulary explicitly.".format(path_str)
            )
        if not os.path.isdir(path_str):
            return path_str
        candidates = sorted(
            name for name in os.listdir(path_str) if name.lower().endswith(".csv")
        )
        if len(candidates) == 1:
            return os.path.join(path_str, candidates[0])
        raise FileNotFoundError(
            "{0} is a directory holding {1} CSV file(s) ({2}); it must hold "
            "exactly one, or --vocabulary must name the file directly. "
            "Choosing one would be a guess.".format(
                path_str, len(candidates), candidates[:_MAX_REPORTED]
            )
        )

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        source_path: Optional[str] = None,
        prefer_common_name: bool = False,
        validate: bool = True,
    ) -> "CompoundIndex":
        """Build from a DrugBank vocabulary frame.

        Required columns: ``DrugBank ID``, ``Common name``. ``Accession Numbers``
        and ``Synonyms`` are optional ``|``-separated lists; when absent, that
        match tier is simply empty (it is a *narrower* index, never a wider one).
        """
        required = ["DrugBank ID", "Common name"]
        missing = [c for c in required if c not in list(df.columns)]
        if missing:
            raise InvariantViolation(
                "DrugBank vocabulary is missing required column(s) {0}; present: "
                "{1}. Refusing to default them -- a defaulted identity column is "
                "a fabricated identity.".format(missing, list(df.columns)),
                name="CompoundIndex.from_frame",
            )
        has_accessions = "Accession Numbers" in df.columns
        has_synonyms = "Synonyms" in df.columns

        records: List[Compound] = []
        for position, row in enumerate(df.to_dict("records")):
            drugbank_id = _text(row["DrugBank ID"])
            if drugbank_id is None:
                raise InvariantViolation(
                    "DrugBank vocabulary row {0}: DrugBank ID is null/empty".format(
                        position
                    ),
                    name="CompoundIndex.from_frame",
                )
            drugbank_id = drugbank_id.upper()
            common_name = _text(row["Common name"])
            if common_name is None:
                raise InvariantViolation(
                    "DrugBank vocabulary row {0} ({1}): Common name is "
                    "null/empty".format(position, drugbank_id),
                    name="CompoundIndex.from_frame",
                )
            records.append(
                Compound(
                    drugbank_id=drugbank_id,
                    common_name=common_name,
                    accession_numbers=_split_pipe(
                        row["Accession Numbers"] if has_accessions else None
                    ),
                    synonyms=_split_pipe(row["Synonyms"] if has_synonyms else None),
                )
            )
        return cls(
            records,
            source_path=source_path,
            prefer_common_name=prefer_common_name,
            validate=validate,
        )

    @classmethod
    def load(
        cls,
        path: Union[str, "os.PathLike[str]"],
        prefer_common_name: bool = False,
        validate: bool = True,
    ) -> "CompoundIndex":
        """Read the DrugBank vocabulary CSV (or its containing directory)."""
        resolved = cls.resolve_vocabulary_path(path)
        frame = pd.read_csv(
            resolved, dtype=str, keep_default_na=False, na_values=[""]
        )
        return cls.from_frame(
            frame,
            source_path=resolved,
            prefer_common_name=prefer_common_name,
            validate=validate,
        )

    # -- lookups -----------------------------------------------------------

    @property
    def records(self) -> Tuple[Compound, ...]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, drugbank_id: Any) -> bool:
        text = _text(drugbank_id)
        return text is not None and text.upper() in self._by_id

    def by_id(self, drugbank_id: Any) -> Compound:
        """Look up by DrugBank ID. Raises ``KeyError`` if absent."""
        text = _text(drugbank_id)
        if text is None:
            raise KeyError(
                "by_id() called with a null/empty DrugBank ID; the caller must "
                "record the rejected row rather than look up nothing"
            )
        key = text.upper()
        if key not in self._by_id:
            raise KeyError(
                "DrugBank ID {0!r} is not in the vocabulary ({1} entries{2})".format(
                    text,
                    len(self._by_id),
                    "" if self.source_path is None else " from " + self.source_path,
                )
            )
        return self._by_id[key]

    def name_candidates(self, token: Any) -> Tuple[Tuple[str, str], ...]:
        """``((drugbank_id, tier), ...)`` for a token, sorted. Audit helper.

        Returns every entry the *name* tiers can reach, with the tier that
        reached it. A caller cannot mistake this for an answer, which is the
        point: names are not keys.
        """
        name = normalize_name(token)
        if name is None:
            return ()
        tiers: Dict[str, str] = {}
        for drug_id in self._common_to_ids.get(name, ()):
            tiers[drug_id] = "common_name"
        for drug_id in self._synonym_to_ids.get(name, ()):
            tiers.setdefault(drug_id, "synonym")
        return tuple(sorted(tiers.items()))

    def resolve(self, token: Any) -> CompoundResolution:
        """Resolve one ``node_1`` token. **Never raises**; the caller must branch."""
        text = _text(token)
        if text is None:
            return CompoundResolution(
                status="NO_KEY",
                drugbank_id=None,
                reason="no compound identifier supplied (null/empty node_1)",
                key="" if token is None else str(token),
            )
        upper = text.upper()

        # Tier 1/2: identifier lookups. Exact, no normalisation beyond case.
        if upper in self._by_id:
            return CompoundResolution(
                status="OK",
                drugbank_id=upper,
                key=text,
                matched_by="drugbank_id",
                candidates=(upper,),
            )
        secondary = self._accession_to_ids.get(upper, ())
        if len(secondary) == 1:
            return CompoundResolution(
                status="OK",
                drugbank_id=secondary[0],
                key=text,
                matched_by="accession_number",
                candidates=secondary,
            )
        if len(secondary) > 1:
            return CompoundResolution(
                status="AMBIGUOUS",
                drugbank_id=None,
                reason=(
                    "{0!r} is a secondary accession of {1} DrugBank entries "
                    "({2}); refusing to choose".format(
                        text, len(secondary), ", ".join(secondary)
                    )
                ),
                key=text,
                candidates=secondary,
            )
        if DRUGBANK_ID_RE.match(upper):
            # DrugBank-shaped but unknown: a withdrawn or mistyped id. It is NOT
            # passed through as if it were valid -- DB00729 is such a row.
            return CompoundResolution(
                status="UNKNOWN",
                drugbank_id=None,
                reason=(
                    "{0} has DrugBank accession syntax but is not in the "
                    "vocabulary ({1} entries{2}); it is withdrawn, merged or "
                    "mistyped".format(
                        upper,
                        len(self._by_id),
                        "" if self.source_path is None else " from " + self.source_path,
                    )
                ),
                key=text,
            )

        # Tiers 3/4: normalised-exact name lookup.
        name = normalize_name(text)
        if name is None:
            return CompoundResolution(
                status="NO_KEY",
                drugbank_id=None,
                reason=(
                    "node_1 {0!r} normalises to nothing (no alphanumeric "
                    "characters); it is not an identifier and not a name".format(text)
                ),
                key=text,
            )
        common = self._common_to_ids.get(name, ())
        synonyms = self._synonym_to_ids.get(name, ())

        outranked: Tuple[str, ...] = ()
        if self.prefer_common_name and common:
            pooled: Tuple[str, ...] = common
            tier_of = {drug_id: "common_name" for drug_id in common}
            outranked = tuple(sorted(set(synonyms) - set(common)))
        else:
            tier_of = {drug_id: "common_name" for drug_id in common}
            for drug_id in synonyms:
                tier_of.setdefault(drug_id, "synonym")
            pooled = tuple(sorted(tier_of))

        if not pooled:
            return CompoundResolution(
                status="UNKNOWN",
                drugbank_id=None,
                reason=(
                    "{0!r} (normalised {1!r}) matches no DrugBank common name or "
                    "synonym exactly; {2} entries searched. Normalised-exact "
                    "matching only -- no fuzzy fallback".format(
                        text, name, len(self._by_id)
                    )
                ),
                key=text,
            )
        if len(pooled) > 1:
            detail = ", ".join(
                "{0} as {1}".format(drug_id, tier_of[drug_id]) for drug_id in pooled
            )
            return CompoundResolution(
                status="AMBIGUOUS",
                drugbank_id=None,
                reason=(
                    "{0!r} (normalised {1!r}) names {2} DrugBank entries ({3}); "
                    "refusing to choose. Resolve it upstream by writing the "
                    "DrugBank ID in node_1".format(text, name, len(pooled), detail)
                ),
                key=text,
                candidates=pooled,
            )
        only = pooled[0]
        return CompoundResolution(
            status="OK",
            drugbank_id=only,
            # An OK resolution may still carry a note; a NON-ok one must.
            reason=(
                "--prefer-common-name outranked {0} synonym match(es): {1}".format(
                    len(outranked), ", ".join(outranked)
                )
                if outranked
                else ""
            ),
            key=text,
            matched_by=tier_of[only],
            candidates=pooled,
        )

    # -- invariants --------------------------------------------------------

    def assert_self_consistent(self) -> None:
        """One row per DrugBank ID, and every id has DrugBank syntax."""
        problems: List[str] = []
        if self._duplicate_ids:
            worst = sorted(self._duplicate_ids.items(), key=lambda kv: (-kv[1], kv[0]))
            problems.append(
                "{0} DrugBank ID(s) appear more than once ({1}). A dict keyed on "
                "the id would silently keep the last row.".format(
                    len(self._duplicate_ids),
                    ", ".join(
                        "{0} x{1}".format(k, n) for k, n in worst[:_MAX_REPORTED]
                    ),
                )
            )
        bad = sorted(
            c.drugbank_id
            for c in self._records
            if not DRUGBANK_ID_RE.match(c.drugbank_id)
        )
        if bad:
            problems.append(
                "{0} vocabulary row(s) carry a DrugBank ID that is not DB+5 "
                "digits ({1}); a drug NAME in the id column is the compound-axis "
                "form of the original defect.".format(
                    len(bad), ", ".join(bad[:_MAX_REPORTED])
                )
            )
        if problems:
            raise InvariantViolation(
                "DrugBank vocabulary{0} failed {1} invariant(s):\n  - {2}".format(
                    "" if self.source_path is None else " " + self.source_path,
                    len(problems),
                    "\n  - ".join(problems),
                ),
                name="CompoundIndex.assert_self_consistent",
            )

    def ambiguous_names(self, limit: int = _MAX_REPORTED) -> Tuple[Dict[str, Any], ...]:
        """Normalised names reaching more than one entry, worst first."""
        pooled: Dict[str, set] = {}
        for name, ids in self._common_to_ids.items():
            pooled.setdefault(name, set()).update(ids)
        for name, ids in self._synonym_to_ids.items():
            pooled.setdefault(name, set()).update(ids)
        rows = [
            {"name": name, "drugbank_ids": sorted(ids)}
            for name, ids in pooled.items()
            if len(ids) > 1
        ]
        rows.sort(key=lambda r: (-len(r["drugbank_ids"]), r["name"]))
        return tuple(rows[:limit])

    def cardinality_report(self) -> Dict[str, Any]:
        """Shape of every match tier. ``max_name_collapse`` is the headline."""
        pooled: Dict[str, set] = {}
        for name, ids in self._common_to_ids.items():
            pooled.setdefault(name, set()).update(ids)
        for name, ids in self._synonym_to_ids.items():
            pooled.setdefault(name, set()).update(ids)
        pooled_sizes = [len(v) for v in pooled.values()] or [0]
        common_sizes = [len(v) for v in self._common_to_ids.values()] or [0]
        synonym_sizes = [len(v) for v in self._synonym_to_ids.values()] or [0]
        collisions = sum(
            1
            for accession, ids in self._accession_to_ids.items()
            if accession in self._by_id and set(ids) != {accession}
        )
        return {
            "check": "compound_index_cardinality",
            "name": "compound_index",
            "n_records": len(self._records),
            "n_drugbank_ids": len(self._by_id),
            "n_duplicate_drugbank_ids": len(self._duplicate_ids),
            "n_secondary_accessions": len(self._accession_to_ids),
            "n_secondary_accessions_colliding_with_a_primary_id": collisions,
            "n_common_names": len(self._common_to_ids),
            "n_ambiguous_common_names": sum(1 for s in common_sizes if s > 1),
            "max_common_name_collapse": max(common_sizes),
            "n_synonyms": len(self._synonym_to_ids),
            "n_ambiguous_synonyms": sum(1 for s in synonym_sizes if s > 1),
            "max_synonym_collapse": max(synonym_sizes),
            "n_names": len(pooled),
            "n_ambiguous_names": sum(1 for s in pooled_sizes if s > 1),
            "max_name_collapse": max(pooled_sizes),
            "prefer_common_name": self.prefer_common_name,
            "source_path": self.source_path,
            "worst_names": list(self.ambiguous_names()),
            # A collapsed name is not a failure of the index -- it is a fact
            # about DrugBank. It is a failure only if something resolves it.
            "ok": True,
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return "CompoundIndex(n={0}, source={1!r})".format(
            len(self._records), self.source_path
        )


def _split_pipe(value: Any) -> Tuple[str, ...]:
    """Split DrugBank's ``a | b | c`` list columns; empty for a null."""
    text = _text(value)
    if text is None:
        return ()
    return tuple(part for part in (p.strip() for p in text.split("|")) if part)


def _freeze(mapping: Mapping[str, List[str]]) -> Dict[str, Tuple[str, ...]]:
    """``{key: sorted distinct values}`` -- a total order, never set order."""
    return {key: tuple(sorted(set(values))) for key, values in mapping.items()}


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def read_edges(path: Union[str, "os.PathLike[str]"]) -> pd.DataFrame:
    """Read ``1_drug_to_protein.tsv`` verbatim, adding a stable ``edge_row``.

    ``keep_default_na=False, na_values=['']`` so only an empty field is null:
    pandas' defaults turn the literal strings ``NA``, ``null``, ``nan`` and
    ``None`` into NaN, and ``NA`` is a real gene symbol.

    ``edge_row`` is the 0-based row number in the file and is the identity of an
    input row throughout this stage, so the rejects ledger points at a line the
    operator can open.
    """
    path_str = os.fspath(path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(
            "edge file not found: {0}. A missing prerequisite is an error, never "
            "a fallback.".format(path_str)
        )
    frame = pd.read_csv(
        path_str, sep="\t", dtype=str, keep_default_na=False, na_values=[""]
    )
    missing = [c for c in REQUIRED_EDGE_COLUMNS if c not in frame.columns]
    if missing:
        raise InvariantViolation(
            "edge file {0} is missing required column(s) {1}; present: {2}. "
            "node_2 (the Entrez Gene ID) is the only key the protein axis "
            "resolves on.".format(path_str, missing, list(frame.columns)),
            name="read_edges",
        )
    for column in OPTIONAL_EDGE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame.reset_index(drop=True)
    frame.insert(0, "edge_row", range(len(frame)))
    logger.info("read %d edges from %s", len(frame), path_str)
    return frame


def load_small_molecule_ids(
    path: Union[str, "os.PathLike[str]"],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read ``small_molecule_drug_links.csv``; return its ids and a report.

    Returns the ids as a one-column frame and **does not de-duplicate it**: a
    duplicated DrugBank ID must surface as a fan-out in
    :func:`checked_merge_with_report`, not be absorbed here.

    Both :data:`SMALL_MOLECULE_ID_COLUMN` and :data:`SMALL_MOLECULE_TYPE_COLUMN`
    are required. Accepting the file without a ``Drug Type`` column would silently
    widen the compound universe to every drug type, which is the ``G8`` pattern
    (``extract_alphafold_models.py:214`` widened 979 accessions to 13,600 the
    same way).
    """
    path_str = os.fspath(path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(
            "small-molecule drug links not found: {0}. It is the compound-axis "
            "gate; there is no fallback that admits every drug type.".format(path_str)
        )
    frame = pd.read_csv(
        path_str, dtype=str, keep_default_na=False, na_values=[""]
    )
    missing = [
        c
        for c in (SMALL_MOLECULE_ID_COLUMN, SMALL_MOLECULE_TYPE_COLUMN)
        if c not in frame.columns
    ]
    if missing:
        raise InvariantViolation(
            "{0} is missing required column(s) {1}; present: {2}. Both are "
            "required: without {3!r} this stage cannot tell a small molecule "
            "from a biotech drug, and defaulting it would admit every drug "
            "type.".format(
                path_str, missing, list(frame.columns), SMALL_MOLECULE_TYPE_COLUMN
            ),
            name="load_small_molecule_ids",
        )
    n_rows = int(len(frame))
    small = frame[frame[SMALL_MOLECULE_TYPE_COLUMN] == SMALL_MOLECULE_TYPE_VALUE]
    ids = small[[SMALL_MOLECULE_ID_COLUMN]].copy()
    ids.columns = ["drugbank_id"]
    ids["drugbank_id"] = ids["drugbank_id"].astype("string").str.strip().str.upper()
    n_null = int(ids["drugbank_id"].isna().sum())
    ids = ids[ids["drugbank_id"].notna()]
    report = {
        "check": "small_molecule_gate_input",
        "name": "small_molecule_drug_links",
        "path": path_str,
        "n_rows": n_rows,
        "n_small_molecule_rows": int(len(small)),
        "n_other_drug_type_rows": n_rows - int(len(small)),
        "n_null_ids_dropped": n_null,
        "n_ids": int(len(ids)),
        "n_distinct_ids": int(ids["drugbank_id"].nunique()),
        "type_column": SMALL_MOLECULE_TYPE_COLUMN,
        "type_value": SMALL_MOLECULE_TYPE_VALUE,
        "ok": True,
    }
    logger.info(
        "read %d small-molecule ids from %s (%d row(s) of another drug type)",
        report["n_ids"],
        path_str,
        report["n_other_drug_type_rows"],
    )
    return ids.reset_index(drop=True), report


def load_obsolete_entrez(
    path: Optional[Union[str, "os.PathLike[str]"]],
) -> Dict[int, str]:
    """Read an optional ``{entrez_id: reason}`` JSON map of retired gene ids.

    Without it a retired Entrez id resolves as ``UNKNOWN`` -- still recorded with
    a reason, never dropped, but less informative.
    """
    if path is None:
        return {}
    path_str = os.fspath(path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(
            "obsolete-entrez map not found: {0}".format(path_str)
        )
    with open(path_str, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise InvariantViolation(
            "{0} must contain a JSON object mapping entrez id -> reason, got "
            "{1}".format(path_str, type(payload).__name__),
            name="load_obsolete_entrez",
        )
    return {int(key): str(value) for key, value in payload.items()}


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolveResult:
    """Everything Phase 1 produces, in memory and already asserted.

    ``resolved`` satisfies the ``interactions_resolved`` schema; ``rejects``
    carries one row per input edge that did not survive, with a reason; the two
    together account for every input row.
    """

    resolved: pd.DataFrame
    rejects: pd.DataFrame
    summary: Dict[str, Any]
    reports: List[Dict[str, Any]] = field(default_factory=list)


def _resolution_reject(resolution: Resolution) -> Tuple[str, str, Tuple[str, ...]]:
    status = _PROTEIN_STATUS_TO_REJECT.get(resolution.status)
    if status is None:  # pragma: no cover - defensive
        raise InvariantViolation(
            "unhandled protein resolution status {0!r}".format(resolution.status),
            name="resolve_interactions",
        )
    return status, resolution.reason, tuple(resolution.candidates)


def _compound_reject(resolution: CompoundResolution) -> Tuple[str, str, Tuple[str, ...]]:
    status = _COMPOUND_STATUS_TO_REJECT.get(resolution.status)
    if status is None:  # pragma: no cover - defensive
        raise InvariantViolation(
            "unhandled compound resolution status {0!r}".format(resolution.status),
            name="resolve_interactions",
        )
    return status, resolution.reason, tuple(resolution.candidates)


def resolve_interactions(
    edges: pd.DataFrame,
    index: IdentityIndex,
    compounds: CompoundIndex,
    small_molecule_ids: pd.DataFrame,
    check_edge_types: bool = True,
) -> ResolveResult:
    """Resolve every edge on ``node_2`` (Entrez) and ``node_1`` (compound).

    Pure over already-loaded inputs: no file is read and none is written, so the
    whole stage is testable without a filesystem.

    Parameters
    ----------
    edges:
        Frame from :func:`read_edges` (must carry ``edge_row``).
    index:
        The pinned :class:`~dock_selectivity.identity.IdentityIndex`. The protein
        axis is resolved **only** through ``index.resolve_entrez(node_2)``;
        ``node_2_name`` is copied into the rejects ledger and never consulted.
    compounds:
        :class:`CompoundIndex` over the DrugBank vocabulary.
    small_molecule_ids:
        One-column (``drugbank_id``) frame from :func:`load_small_molecule_ids`.
    check_edge_types:
        Reject rows whose ``node_1_type``/``node_2_type`` are not
        ``drug``/``protein`` (only when those columns are present).
    """
    if "edge_row" not in edges.columns:
        raise InvariantViolation(
            "edges frame has no 'edge_row' column; build it with read_edges() so "
            "every reject can name the line it came from",
            name="resolve_interactions",
        )
    missing = [c for c in REQUIRED_EDGE_COLUMNS if c not in edges.columns]
    if missing:
        raise InvariantViolation(
            "edges frame is missing required column(s) {0}".format(missing),
            name="resolve_interactions",
        )
    n_in = int(len(edges))

    # Resolve each DISTINCT key once, in sorted order. Distinct-key caching keeps
    # the pass linear in the number of keys rather than of rows, and sorting the
    # keys means nothing downstream can depend on row order or on hash order.
    entrez_keys = sorted(
        {k for k in (_text(v) for v in edges[EDGE_ENTREZ_COLUMN]) if k is not None}
    )
    protein_cache: Dict[str, Resolution] = {
        key: index.resolve_entrez(key) for key in entrez_keys
    }
    compound_keys = sorted(
        {k for k in (_text(v) for v in edges[EDGE_DRUG_COLUMN]) if k is not None}
    )
    compound_cache: Dict[str, CompoundResolution] = {
        key: compounds.resolve(key) for key in compound_keys
    }

    null_protein = index.resolve_entrez(None)
    null_compound = compounds.resolve(None)

    has_types = check_edge_types and all(
        c in edges.columns for c in (EDGE_DRUG_TYPE_COLUMN, EDGE_PROTEIN_TYPE_COLUMN)
    )

    ok_rows: List[Dict[str, Any]] = []
    reject_rows: List[Dict[str, Any]] = []

    for row in edges.to_dict("records"):
        edge_row = int(row["edge_row"])
        drug_token = _text(row[EDGE_DRUG_COLUMN])
        entrez_token = _text(row[EDGE_ENTREZ_COLUMN])
        node_1_name = _text(row.get(EDGE_DRUG_NAME_COLUMN))
        node_2_name = _text(row.get(EDGE_SYMBOL_COLUMN))

        statuses: List[str] = []
        reasons: List[str] = []
        candidates: List[str] = []
        axes: List[str] = []

        if has_types:
            drug_type = _text(row.get(EDGE_DRUG_TYPE_COLUMN))
            protein_type = _text(row.get(EDGE_PROTEIN_TYPE_COLUMN))
            if drug_type != EXPECTED_DRUG_TYPE or protein_type != EXPECTED_PROTEIN_TYPE:
                statuses.append("WRONG_EDGE_TYPE")
                axes.append("edge")
                reasons.append(
                    "edge is {0!r} -> {1!r}, expected {2!r} -> {3!r}; this file "
                    "must contain drug-protein edges only".format(
                        drug_type,
                        protein_type,
                        EXPECTED_DRUG_TYPE,
                        EXPECTED_PROTEIN_TYPE,
                    )
                )

        protein = (
            protein_cache[entrez_token] if entrez_token is not None else null_protein
        )
        if not protein.ok:
            status, reason, cands = _resolution_reject(protein)
            statuses.append(status)
            axes.append("protein")
            reasons.append("node_2: " + reason)
            candidates.extend(cands)

        compound = (
            compound_cache[drug_token] if drug_token is not None else null_compound
        )
        if not compound.ok:
            status, reason, cands = _compound_reject(compound)
            statuses.append(status)
            axes.append("compound")
            reasons.append("node_1: " + reason)
            candidates.extend(cands)

        if statuses:
            reject_rows.append(
                {
                    "edge_row": edge_row,
                    "node_1": drug_token,
                    "node_2": entrez_token,
                    "node_1_name": node_1_name,
                    "node_2_name": node_2_name,
                    "axis": "both" if len(set(axes)) > 1 else axes[0],
                    "status": _primary_status(statuses),
                    "reason": _join_reason(reasons),
                    "candidates": ";".join(candidates),
                }
            )
            continue

        protein_record = protein.unwrap()
        ok_rows.append(
            {
                "edge_row": edge_row,
                "node_1": drug_token,
                "node_2": entrez_token,
                "node_1_name": node_1_name,
                "node_2_name": node_2_name,
                "drugbank_id": compound.unwrap(),
                "entrez_id": int(protein.key),
                "accession": protein_record.accession,
                # The ONLY direction a symbol travels: accession -> label.
                "gene": index.label(protein_record.accession),
                "matched_by": compound.matched_by,
                # Diagnostics counted AFTER the small-molecule gate, so they
                # describe what actually survived rather than what merely passed
                # identity resolution.
                "node_2_name_was_null": node_2_name is None,
                "node_1_was_a_name": compound.matched_by in _NAME_TIERS,
            }
        )

    ok_frame = pd.DataFrame(ok_rows, columns=_OK_COLUMNS)
    reject_frame = pd.DataFrame(reject_rows, columns=list(REJECT_COLUMNS))
    reports: List[Dict[str, Any]] = []

    # ---- small-molecule gate -------------------------------------------
    ok_frame, gate_rejects, gate_report = _apply_small_molecule_gate(
        ok_frame, small_molecule_ids
    )
    reports.append(gate_report)
    if len(gate_rejects):
        parts = [f for f in (reject_frame, gate_rejects) if len(f)]
        reject_frame = pd.concat(parts, ignore_index=True)
    if len(reject_frame):
        reject_frame["edge_row"] = reject_frame["edge_row"].astype(int)

    n_resolved_edges = int(len(ok_frame))
    n_rejected_edges = int(len(reject_frame))
    n_null_symbol_edges_resolved = (
        int(ok_frame["node_2_name_was_null"].sum()) if n_resolved_edges else 0
    )
    n_slug_edges_resolved = (
        int(ok_frame["node_1_was_a_name"].sum()) if n_resolved_edges else 0
    )
    reports.append(
        assert_total_accounting(
            n_in, n_resolved_edges, n_rejected_edges, name="resolve_interactions edges"
        )
    )

    # ---- collapse to the declared primary key ---------------------------
    resolved, dedup_report = _deduplicate(ok_frame)
    reports.append(dedup_report)

    # ---- assert the guarantees before anything is written ---------------
    reports.append(
        assert_unique_key(
            resolved,
            ["drugbank_id", "accession"],
            name="interactions_resolved primary key",
        )
    )
    if len(resolved):
        reports.append(
            assert_collapse_factor_one(
                resolved[["entrez_id", "accession"]].drop_duplicates(),
                name="entrez -> accession",
                key="entrez_id",
                value="accession",
            )
        )
        reports.append(
            assert_collapse_factor_one(
                resolved[["accession", "entrez_id"]].drop_duplicates(),
                name="accession -> entrez",
                key="accession",
                value="entrez_id",
            )
        )
        # The KNOWN_ISSUE.md regression assertion, applied to our own output.
        # truth_lookup is the index, which is independent of `resolved`.
        reports.append(
            assert_self_consistent_labels(
                resolved,
                key_col="accession",
                label_col="gene",
                truth_lookup=lambda acc: index.by_accession(acc).symbols,
                name="interactions_resolved gene labels",
            )
        )

    resolved = coerce(resolved, "interactions_resolved")
    reports.append(validate(resolved, "interactions_resolved"))

    summary = _build_summary(
        n_in=n_in,
        n_resolved_edges=n_resolved_edges,
        n_rejected_edges=n_rejected_edges,
        resolved=resolved,
        rejects=reject_frame,
        dedup_report=dedup_report,
        n_null_symbol_edges_resolved=n_null_symbol_edges_resolved,
        n_slug_edges_resolved=n_slug_edges_resolved,
    )
    reports.append(summary)

    reject_frame = reject_frame.sort_values("edge_row", kind="mergesort").reset_index(
        drop=True
    )
    _assert_every_reject_has_a_reason(reject_frame)
    return ResolveResult(
        resolved=resolved, rejects=reject_frame, summary=summary, reports=reports
    )


_OK_COLUMNS = [
    "edge_row",
    "node_1",
    "node_2",
    "node_1_name",
    "node_2_name",
    "drugbank_id",
    "entrez_id",
    "accession",
    "gene",
    "matched_by",
    "node_2_name_was_null",
    "node_1_was_a_name",
]


def _apply_small_molecule_gate(
    ok_frame: pd.DataFrame,
    small_molecule_ids: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Keep only edges whose drug is a small molecule; reject the rest by name.

    Uses :func:`checked_merge_with_report`, so a duplicated DrugBank ID in the
    links file raises instead of duplicating edges -- the ``parse_prepare.py:1171``
    failure mode (+410,087 rows), caught at its own scale.
    """
    if "drugbank_id" not in small_molecule_ids.columns:
        raise InvariantViolation(
            "small_molecule_ids must have a 'drugbank_id' column; got {0}".format(
                list(small_molecule_ids.columns)
            ),
            name="small_molecule_gate",
        )
    flags = small_molecule_ids[["drugbank_id"]].copy()
    flags["drugbank_id"] = flags["drugbank_id"].astype(object)
    flags["_is_small_molecule"] = True
    if ok_frame.empty:
        empty_rejects = pd.DataFrame(columns=list(REJECT_COLUMNS))
        report = {
            "check": "checked_merge",
            "name": "small_molecule_gate",
            "n_left": 0,
            "n_right": int(len(flags)),
            "n_out": 0,
            "ok": True,
        }
        return ok_frame, empty_rejects, report

    merged, report = checked_merge_with_report(
        ok_frame,
        flags,
        on="drugbank_id",
        how="left",
        name="small_molecule_gate",
    )
    is_small = merged["_is_small_molecule"].fillna(False).astype(bool)
    kept = merged[is_small].drop(columns=["_is_small_molecule"]).reset_index(drop=True)
    dropped = merged[~is_small].reset_index(drop=True)

    rejects = pd.DataFrame(
        [
            {
                "edge_row": int(row["edge_row"]),
                "node_1": row["node_1"],
                "node_2": row["node_2"],
                "node_1_name": row["node_1_name"],
                "node_2_name": row["node_2_name"],
                "axis": "compound",
                "status": "NOT_SMALL_MOLECULE",
                "reason": (
                    "node_1: {0} resolved but is not listed as a {1} in the drug "
                    "links file; it cannot be docked and is excluded from the "
                    "corrected interaction set".format(
                        row["drugbank_id"], SMALL_MOLECULE_TYPE_VALUE
                    )
                ),
                "candidates": row["drugbank_id"],
            }
            for row in dropped.to_dict("records")
        ],
        columns=list(REJECT_COLUMNS),
    )

    report["n_not_small_molecule"] = int(len(dropped))
    report["n_small_molecule"] = int(len(kept))
    return kept, rejects, report


def _deduplicate(ok_frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Collapse resolved edges onto the ``(drugbank_id, accession)`` primary key.

    Two input edges legitimately land on the same pair: 38 rows of
    ``1_drug_to_protein.tsv`` carry both a DrugBank ID and a name slug for the
    same compound (``DB00155`` and ``L-Citrulline``), each pointing at the same
    target. That is a duplicate, not a conflict, and it is *accounted for* rather
    than silently absorbed: the surviving row's ``selected_by`` names every edge
    that produced it, and the count is reported.

    Two edges with the same pair but different Entrez ids WOULD be a conflict --
    it would mean the index maps two gene ids onto one accession, contradicting
    G5 -- so that raises.
    """
    columns = ["drugbank_id", "entrez_id", "accession", "gene", "selected_by"]
    if ok_frame.empty:
        return (
            pd.DataFrame(columns=columns),
            {
                "check": "deduplicate",
                "name": "interactions_resolved dedup",
                "n_resolved_edges": 0,
                "n_rows": 0,
                "n_duplicate_edges": 0,
                "n_keys_with_duplicates": 0,
                "worst_keys": [],
                "ok": True,
            },
        )

    conflicting = (
        ok_frame.groupby(["drugbank_id", "accession"], sort=True)["entrez_id"]
        .nunique()
        .pipe(lambda s: s[s > 1])
    )
    if len(conflicting):
        raise InvariantViolation(
            "{0} (drugbank_id, accession) pair(s) resolved from more than one "
            "Entrez id, e.g. {1}. That contradicts G5 (entrez -> accession "
            "collapse factor 1): two gene ids cannot own one accession. Rebuild "
            "the identity index; do not pick one.".format(
                len(conflicting),
                [list(k) for k in list(conflicting.index)[:_MAX_REPORTED]],
            ),
            name="interactions_resolved dedup",
        )

    # Total sort: the surviving row is the lowest edge_row of the pair, always.
    ordered = ok_frame.sort_values(
        ["drugbank_id", "accession", "edge_row"], kind="mergesort"
    )
    grouped = ordered.groupby(["drugbank_id", "accession"], sort=True)

    rows: List[Dict[str, Any]] = []
    worst: List[Dict[str, Any]] = []
    n_duplicate_edges = 0
    n_keys_with_duplicates = 0
    for (drugbank_id, accession), block in grouped:
        first = block.iloc[0]
        edge_rows = [int(v) for v in block["edge_row"].tolist()]
        tokens = [str(v) for v in block["node_1"].tolist()]
        matched = [str(v) for v in block["matched_by"].tolist()]
        if len(block) > 1:
            n_duplicate_edges += len(block) - 1
            n_keys_with_duplicates += 1
            if len(worst) < _MAX_REPORTED:
                worst.append(
                    {
                        "drugbank_id": drugbank_id,
                        "accession": accession,
                        "n_edges": int(len(block)),
                        "node_1_tokens": tokens,
                    }
                )
        rows.append(
            {
                "drugbank_id": drugbank_id,
                "entrez_id": int(first["entrez_id"]),
                "accession": accession,
                "gene": first["gene"],
                "selected_by": _selected_by(
                    entrez_id=int(first["entrez_id"]),
                    accession=accession,
                    tokens=tokens,
                    matched_by=matched,
                    edge_rows=edge_rows,
                ),
            }
        )

    resolved = pd.DataFrame(rows, columns=columns)
    # Sorted output => byte-identical file for the same input, in any row order.
    resolved = resolved.sort_values(
        ["drugbank_id", "accession"], kind="mergesort"
    ).reset_index(drop=True)
    report = {
        "check": "deduplicate",
        "name": "interactions_resolved dedup",
        "n_resolved_edges": int(len(ok_frame)),
        "n_rows": int(len(resolved)),
        "n_duplicate_edges": n_duplicate_edges,
        "n_keys_with_duplicates": n_keys_with_duplicates,
        "worst_keys": worst,
        "ok": int(len(ok_frame)) - n_duplicate_edges == int(len(resolved)),
    }
    if not report["ok"]:  # pragma: no cover - defensive
        raise InvariantViolation(
            "deduplication lost rows: {0} resolved edges - {1} duplicates != {2} "
            "output rows".format(len(ok_frame), n_duplicate_edges, len(resolved)),
            name="interactions_resolved dedup",
            report=report,
        )
    return resolved, report


def _selected_by(
    entrez_id: int,
    accession: str,
    tokens: Sequence[str],
    matched_by: Sequence[str],
    edge_rows: Sequence[int],
) -> str:
    """Invariant I7: a human-readable statement of why this row exists.

    Had ``required_structures.csv`` carried one, ``AVPR2 -> Q8NCT1`` would have
    been legible in the plan file itself.
    """
    unique_tokens = sorted(set(tokens))
    unique_matched = sorted(set(matched_by))
    return _SELECTED_BY_SEP.join(
        [
            "entrez={0} -> {1}".format(entrez_id, accession),
            "node_1={0} via {1}".format(
                ",".join(unique_tokens), ",".join(unique_matched)
            ),
            "edge_rows={0}".format(",".join(str(r) for r in sorted(edge_rows))),
        ]
    )


def _assert_every_reject_has_a_reason(rejects: pd.DataFrame) -> None:
    """No unexplained rejection. A dropped row without a reason is a lost row."""
    if rejects.empty:
        return
    blank = rejects[
        rejects["reason"].isna() | (rejects["reason"].astype(str).str.strip() == "")
    ]
    if len(blank):
        raise InvariantViolation(
            "{0} reject row(s) carry no reason (edge rows {1}). Every rejected "
            "edge must say why, or it is indistinguishable from a silent "
            "drop.".format(len(blank), blank["edge_row"].tolist()[:_MAX_REPORTED]),
            name="resolve_interactions rejects",
        )
    unknown = sorted(set(rejects["status"]) - set(REJECT_PRECEDENCE))
    if unknown:
        raise InvariantViolation(
            "reject status(es) {0} are not in the declared vocabulary {1}".format(
                unknown, list(REJECT_PRECEDENCE)
            ),
            name="resolve_interactions rejects",
        )


def _build_summary(
    n_in: int,
    n_resolved_edges: int,
    n_rejected_edges: int,
    resolved: pd.DataFrame,
    rejects: pd.DataFrame,
    dedup_report: Dict[str, Any],
    n_null_symbol_edges_resolved: int,
    n_slug_edges_resolved: int,
) -> Dict[str, Any]:
    by_status = (
        {str(k): int(v) for k, v in rejects["status"].value_counts().items()}
        if len(rejects)
        else {}
    )
    return {
        "check": "resolve_interactions summary",
        "name": "resolve_interactions",
        "n_edges_in": n_in,
        "n_resolved_edges": n_resolved_edges,
        "n_rejected_edges": n_rejected_edges,
        "n_output_rows": int(len(resolved)),
        "n_duplicate_edges_collapsed": int(dedup_report["n_duplicate_edges"]),
        "n_drugs": int(resolved["drugbank_id"].nunique()) if len(resolved) else 0,
        "n_accessions": int(resolved["accession"].nunique()) if len(resolved) else 0,
        "n_entrez": int(resolved["entrez_id"].nunique()) if len(resolved) else 0,
        # Edges with no gene symbol at all: unreachable by the retired
        # symbol-keyed resolver, resolved here through node_2 alone.
        "n_edges_resolved_with_null_node_2_name": n_null_symbol_edges_resolved,
        # node_1 rows that were a name slug rather than a DrugBank ID: 222 of
        # the 8,568 edges, a population the old pipeline never examined at all.
        "n_edges_resolved_from_a_node_1_name_slug": n_slug_edges_resolved,
        "rejects_by_status": {k: by_status.get(k, 0) for k in REJECT_PRECEDENCE},
        "ok": n_in == n_resolved_edges + n_rejected_edges,
    }


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def write_rejects(rejects: pd.DataFrame, path: Union[str, "os.PathLike[str]"]) -> str:
    """Write the rejects ledger atomically and deterministically.

    Always written, even with zero rows: an empty file with a header is evidence
    that nothing was rejected, while a missing file is indistinguishable from a
    stage that never checked.
    """
    path_str = os.fspath(path)
    frame = rejects.reindex(columns=list(REJECT_COLUMNS))
    if len(frame):
        frame = frame.sort_values("edge_row", kind="mergesort")
    frame = frame.reset_index(drop=True)
    parent = os.path.dirname(os.path.abspath(path_str))
    if parent:
        os.makedirs(parent, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=parent or ".", delete=False, suffix=".tmp")
    handle.close()
    try:
        frame.to_csv(handle.name, index=False, lineterminator="\n")
        os.replace(handle.name, path_str)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    logger.info("wrote %d reject row(s) to %s", len(frame), path_str)
    return path_str


def _format_summary(summary: Dict[str, Any]) -> str:
    lines = [
        "edges in ............................. {0}".format(summary["n_edges_in"]),
        "  resolved ........................... {0}".format(
            summary["n_resolved_edges"]
        ),
        "  rejected ........................... {0}".format(
            summary["n_rejected_edges"]
        ),
        "output rows (drugbank_id, accession) . {0}".format(summary["n_output_rows"]),
        "  duplicate edges collapsed .......... {0}".format(
            summary["n_duplicate_edges_collapsed"]
        ),
        "  distinct drugs ..................... {0}".format(summary["n_drugs"]),
        "  distinct accessions ................ {0}".format(summary["n_accessions"]),
        "resolved despite a null node_2_name .. {0}   (unreachable by a symbol-keyed resolver)".format(
            summary["n_edges_resolved_with_null_node_2_name"]
        ),
        "resolved from a node_1 name slug ..... {0}".format(
            summary["n_edges_resolved_from_a_node_1_name_slug"]
        ),
        "rejects by reason:",
    ]
    for status, count in summary["rejects_by_status"].items():
        if count:
            lines.append("  {0:<22} {1}".format(status, count))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resolve_interactions.py",
        description=(
            "REBUILD_PLAN Phase 1: resolve 1_drug_to_protein.tsv into "
            "manifests/interactions_resolved.csv. The protein axis resolves on "
            "node_2 (Entrez Gene ID) only -- never on node_2_name."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Every input edge ends up either in --out or in --rejects with a "
            "reason; the stage asserts n_edges_in == n_resolved + n_rejected "
            "before writing anything."
        ),
    )
    parser.add_argument(
        "--edges",
        required=True,
        help="1_drug_to_protein.tsv (columns node_1, node_2, ...).",
    )
    parser.add_argument(
        "--index",
        required=True,
        help="identity/identity_index.parquet built by build_identity_index.py.",
    )
    parser.add_argument(
        "--drugs",
        required=True,
        help=(
            "small_molecule_drug_links.csv; requires the columns {0!r} and "
            "{1!r}.".format(SMALL_MOLECULE_ID_COLUMN, SMALL_MOLECULE_TYPE_COLUMN)
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        help="manifests/interactions_resolved.csv (schema interactions_resolved).",
    )
    parser.add_argument(
        "--rejects",
        default=None,
        help=(
            "Ledger of every edge that did not resolve. Default: "
            "<out>.rejects.csv. Do NOT point it at "
            "identity/unresolved_targets.csv: that file is Phase 0's own "
            "ledger, and 'audit.py identity' asserts its row count equals the "
            "index build's n_rejected (G2). Overwriting it fails that gate. "
            "Use a distinct name such as "
            "identity/unresolved_interactions.csv."
        ),
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Invariant report JSON. Default: <out>.invariants.json.",
    )
    parser.add_argument(
        "--vocabulary",
        default=DEFAULT_VOCABULARY,
        help=(
            "DrugBank vocabulary CSV (or a directory holding exactly one). "
            "Required: 222 node_1 rows are name slugs. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--index-sha256",
        default=None,
        help=(
            "Pin the identity index. A mismatch aborts, so a stage cannot run "
            "against an index other than the one its plan was built from."
        ),
    )
    parser.add_argument(
        "--obsolete-entrez",
        default=None,
        help=(
            "Optional JSON object {entrez_id: reason} of retired gene ids, so "
            "they are reported as OBSOLETE_ENTREZ rather than UNKNOWN_ENTREZ."
        ),
    )
    parser.add_argument(
        "--prefer-common-name",
        action="store_true",
        help=(
            "Declare that a DrugBank common name outranks a synonym when a "
            "name slug matches both. OFF by default: pooling the two tiers is "
            "what makes 'L-Arginine' AMBIGUOUS instead of a silent pick."
        ),
    )
    parser.add_argument(
        "--no-edge-type-check",
        dest="check_edge_types",
        action="store_false",
        help="Do not reject rows whose node_1_type/node_2_type are not drug/protein.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug-level logging.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    out_path = os.fspath(args.out)
    base = os.path.splitext(out_path)[0]
    rejects_path = args.rejects or (base + ".rejects.csv")
    report_path = args.report or (base + ".invariants.json")
    if os.path.abspath(rejects_path) == os.path.abspath(out_path):
        logger.error("--rejects and --out must be different files")
        return 2

    try:
        index = IdentityIndex.load(
            args.index,
            expect_sha256=args.index_sha256,
            obsolete_entrez=load_obsolete_entrez(args.obsolete_entrez),
        )
        logger.info(
            "identity index: %d accessions, sha256=%s",
            len(index),
            index.source_sha256,
        )
        compounds = CompoundIndex.load(
            args.vocabulary, prefer_common_name=args.prefer_common_name
        )
        logger.info(
            "DrugBank vocabulary: %d entries from %s",
            len(compounds),
            compounds.source_path,
        )
        edges = read_edges(args.edges)
        small_molecule_ids, drugs_report = load_small_molecule_ids(args.drugs)

        result = resolve_interactions(
            edges,
            index,
            compounds,
            small_molecule_ids,
            check_edge_types=args.check_edge_types,
        )

        write_checked(result.resolved, out_path, "interactions_resolved")
        write_rejects(result.rejects, rejects_path)
        reports = [
            {
                "check": "identity_index",
                "name": "identity_index",
                "path": os.fspath(args.index),
                "sha256": index.source_sha256,
                "ok": True,
                "cardinality": index.cardinality_report(),
            },
            compounds.cardinality_report(),
            drugs_report,
        ] + list(result.reports)
        write_invariant_report(reports, report_path)
    except INVARIANT_ERRORS + (SchemaViolation, FileNotFoundError) as exc:
        # A failed invariant is a stage failure, not a stack trace to decode:
        # log the actionable message and exit non-zero with nothing written.
        logger.error("%s", exc)
        return 2

    logger.info(
        "resolve_interactions complete\n%s", _format_summary(result.summary)
    )
    logger.info("wrote %s", out_path)
    logger.info("wrote %s", rejects_path)
    logger.info("wrote %s", report_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
