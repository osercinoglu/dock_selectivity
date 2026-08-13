"""Job-directory naming codec -- the fix for the load-bearing-string defect.

The old scheme
--------------
``batch_dock.py:2211`` wrote ``f"{drugbank_id}_{gene_name}_{uniprot_id}_{pocket_suffix}"``
and ``parse_prepare.py:359`` read identity back out with
``r'(DB\\d+)_([A-Za-z0-9]+)_([A-Z0-9]+)_cavity_(\\d+)'``.  Three structural sins:

* the delimiter ``_`` is a legal character in the payload;
* a nullable *display* field sits inside a positional key -- the gene token is
  literally ``nan`` in 11,806 of 24,173 directories;
* the two halves came from opposite sides of the mapping bug, so the gene and
  the accession in one name can name different proteins
  (``DB00091_CAMLG_P08F94_...`` -- P08F94 is PKHD1, not CAMLG).

The regex rejected 1,777 directories (1,743 legacy multi-fragment names, 34
hyphenated symbols such as ``MT-ND1``), producing 1,174,786 rows with null
identity that shared one fabricated ``<NA>_<NA>`` group.

The new scheme
--------------
``DB00035__P30518__AVPR2__F1__cav2``

* ``__`` delimiter -- hyphenated symbols and odd characters cannot break it;
* **accession before gene**, and the gene is written by
  ``IdentityIndex.label(accession)`` at creation time, so the two cannot
  disagree by construction;
* fragment is **explicit** (``F1``), closing the gap that caused the
  410,087-row merge fan-out;
* the name is a convenience.  The authority is ``job_identity.json`` inside the
  directory -- see :func:`write_job_identity` / :func:`read_job_identity`.

Legacy directories are recovered with :func:`parse_legacy_job_dir`, which is
**right-anchored** and requires the accession set: strip ``_cavity_<N>``, strip
an optional ``_-F<k>-model_v<n>``, then take the rightmost token that is a
member of the supplied accessions.  That resolves 24,173/24,173 real
directories.  The accession is authoritative; the gene token is an untrusted
label.

Python 3.9 compatible.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, replace as _dc_replace
from typing import Any, Collection, Dict, Optional, Tuple, Union

from .identity import ACCESSION_RE

__all__ = [
    "DELIM",
    "NULL_GENE",
    "JOB_IDENTITY_FILENAME",
    "JOB_IDENTITY_SCHEMA_VERSION",
    "JobIdError",
    "JobIdParseError",
    "JobIdentity",
    "format_job_dir",
    "parse_job_dir",
    "parse_legacy_job_dir",
    "write_job_identity",
    "read_job_identity",
    "read_job_identity_record",
    "normalize_fragment_id",
]

#: Double underscore.  A single underscore is a legal payload character; this
#: one is not (fields containing ``__`` are rejected at construction).
DELIM = "__"

#: Written in the gene slot when no symbol is known.  Never ``nan``.
NULL_GENE = "-"

JOB_IDENTITY_FILENAME = "job_identity.json"
JOB_IDENTITY_SCHEMA_VERSION = 1

_CAVITY_PREFIX = "cav"
_FRAGMENT_RE = re.compile(r"^F\d+$")
_NEW_CAVITY_RE = re.compile(r"^cav(\d+)$")
_LEGACY_CAVITY_RE = re.compile(r"_cavity_(\d+)$")
#: ``_-F13-model_v1`` (1,743 real dirs) and the ``_AF-P12345-F2-model_v1`` form.
_LEGACY_FRAGMENT_RE = re.compile(r"_-?(?:AF-[A-Z0-9]+-)?F(\d+)-model_v\d+$")
_DRUGBANK_RE = re.compile(r"^DB\d+$", re.IGNORECASE)

#: Gene tokens that mean "no symbol".  ``nan`` is here because the old scheme
#: wrote the string ``nan`` into 11,806 directory names.
_NULL_GENE_TOKENS = frozenset({"", "nan", "none", "null", "<na>", NULL_GENE})

_CORE_FIELDS = (
    "drugbank_id",
    "accession",
    "gene",
    "fragment_id",
    "cavity_index",
)


class JobIdError(ValueError):
    """A job identity is malformed, or a name cannot be built from it."""


class JobIdParseError(JobIdError):
    """A directory name cannot be parsed into a job identity."""


def _basename(name: Any) -> str:
    """Final path component of *name*, accepting ``str`` or ``os.PathLike``."""
    text = os.fspath(name) if hasattr(name, "__fspath__") else str(name)
    text = text.strip().rstrip("/")
    return os.path.basename(text)


def _reject_field(value: str, field_name: str) -> None:
    if not value:
        raise JobIdError("{0} must be non-empty".format(field_name))
    if DELIM in value:
        raise JobIdError(
            "{0}={1!r} contains the field delimiter {2!r}; the name would not "
            "round-trip".format(field_name, value, DELIM)
        )
    if value.startswith("_") or value.endswith("_"):
        raise JobIdError(
            "{0}={1!r} may not start or end with '_'; adjacent to the {2!r} "
            "delimiter it would split ambiguously".format(field_name, value, DELIM)
        )
    if any(ch.isspace() for ch in value) or "/" in value:
        raise JobIdError(
            "{0}={1!r} contains whitespace or a path separator; it is used as a "
            "directory name".format(field_name, value)
        )


def normalize_fragment_id(value: Any) -> Optional[str]:
    """Normalise a fragment to ``'F<k>'``.

    Accepts ``'F1'``, ``'f1'``, ``'1'``, ``1``.  ``None``/``nan``/``''`` ->
    ``None``.  The old pipeline wrote ``'F1'`` on positives and bare ``'1'`` on
    negatives, which is why every join on ``Fragment_ID`` must normalise first.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise JobIdError("fragment_id must not be a bool, got {0!r}".format(value))
    if isinstance(value, int):
        if value < 0:
            raise JobIdError("fragment_id must be non-negative, got {0!r}".format(value))
        return "F{0}".format(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        if not float(value).is_integer():
            raise JobIdError("fragment_id {0!r} is not an integer".format(value))
        return "F{0}".format(int(value))
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>", "-"}:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit():
        return "F{0}".format(int(text))
    candidate = text.upper()
    if _FRAGMENT_RE.match(candidate):
        return "F{0}".format(int(candidate[1:]))
    raise JobIdError(
        "fragment_id {0!r} is not a fragment id; expected 'F<k>', '<k>' or an "
        "int".format(value)
    )


def _normalize_gene(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    text = str(value).strip()
    if text.lower() in _NULL_GENE_TOKENS:
        return None
    _reject_field(text, "gene")
    return text


def _normalize_cavity(value: Any) -> int:
    if isinstance(value, bool):
        raise JobIdError("cavity_index must not be a bool, got {0!r}".format(value))
    if isinstance(value, int):
        index = int(value)
    elif isinstance(value, float):
        if not float(value).is_integer():
            raise JobIdError("cavity_index {0!r} is not an integer".format(value))
        index = int(value)
    else:
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        if not re.match(r"^\d+$", text):
            raise JobIdError(
                "cavity_index {0!r} is not a non-negative integer".format(value)
            )
        index = int(text)
    if index < 0:
        raise JobIdError("cavity_index must be non-negative, got {0!r}".format(value))
    return index


@dataclass(frozen=True)
class JobIdentity:
    """What one docking job is a measurement of.

    ``accession`` is the identity; ``gene`` is a display label that may be
    ``None`` and must never be used to select anything.  Fields are normalised
    and validated on construction, so an instance is always encodable.
    """

    drugbank_id: str
    accession: str
    gene: Optional[str]
    fragment_id: Optional[str]
    cavity_index: int

    def __post_init__(self) -> None:
        drug = "" if self.drugbank_id is None else str(self.drugbank_id).strip()
        if _DRUGBANK_RE.match(drug):
            drug = drug.upper()
        _reject_field(drug, "drugbank_id")
        object.__setattr__(self, "drugbank_id", drug)

        accession = (
            "" if self.accession is None else str(self.accession).strip().upper()
        )
        _reject_field(accession, "accession")
        if not ACCESSION_RE.match(accession):
            raise JobIdError(
                "accession={0!r} is not a canonical UniProtKB accession. A gene "
                "symbol, entry name or isoform suffix is not an identity key -- "
                "resolve it through IdentityIndex before building a job "
                "name.".format(accession)
            )
        object.__setattr__(self, "accession", accession)

        object.__setattr__(self, "gene", _normalize_gene(self.gene))

        fragment = normalize_fragment_id(self.fragment_id)
        if fragment is not None:
            _reject_field(fragment, "fragment_id")
        object.__setattr__(self, "fragment_id", fragment)

        object.__setattr__(self, "cavity_index", _normalize_cavity(self.cavity_index))

    # -- derived views -----------------------------------------------------

    @property
    def pocket_key(self) -> Tuple[str, Optional[str], int]:
        """``(accession, fragment_id, cavity_index)`` -- the pocket identity."""
        return (self.accession, self.fragment_id, self.cavity_index)

    @property
    def pair_key(self) -> Tuple[str, str]:
        """``(drugbank_id, accession)`` -- the interaction identity."""
        return (self.drugbank_id, self.accession)

    def dir_name(self) -> str:
        """This identity as a directory name (new scheme)."""
        return format_job_dir(
            self.drugbank_id,
            self.accession,
            self.gene,
            self.fragment_id,
            self.cavity_index,
        )

    def with_fragment(self, fragment_id: Any) -> "JobIdentity":
        """Copy carrying an explicitly supplied fragment id."""
        return _dc_replace(self, fragment_id=normalize_fragment_id(fragment_id))

    def with_gene(self, gene: Any) -> "JobIdentity":
        """Copy carrying a display label -- use ``IdentityIndex.label(accession)``."""
        return _dc_replace(self, gene=_normalize_gene(gene))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drugbank_id": self.drugbank_id,
            "accession": self.accession,
            "gene": self.gene,
            "fragment_id": self.fragment_id,
            "cavity_index": self.cavity_index,
        }

    @classmethod
    def from_dict(cls, record: Dict[str, Any]) -> "JobIdentity":
        missing = [k for k in _CORE_FIELDS if k not in record]
        if missing:
            raise JobIdError(
                "job identity record is missing field(s): {0}. Present: {1}".format(
                    ", ".join(missing), ", ".join(sorted(record))
                )
            )
        return cls(
            drugbank_id=record["drugbank_id"],
            accession=record["accession"],
            gene=record["gene"],
            fragment_id=record["fragment_id"],
            cavity_index=record["cavity_index"],
        )


# --------------------------------------------------------------------------
# codec
# --------------------------------------------------------------------------


def format_job_dir(
    drugbank_id: Any,
    accession: Any,
    gene: Any,
    fragment_id: Any,
    cavity_index: Any,
) -> str:
    """Build a job directory name: ``DB00035__P30518__AVPR2__F1__cav2``.

    *gene* may be ``None`` (written as ``-``); *fragment_id* may not.  A
    missing fragment used to be inferred downstream, which is how one pocket
    key matched several pockets and fanned a merge out by 410,087 rows -- pass
    ``'F1'`` explicitly for single-fragment AlphaFold models.
    """
    identity = JobIdentity(
        drugbank_id=drugbank_id,
        accession=accession,
        gene=gene,
        fragment_id=fragment_id,
        cavity_index=cavity_index,
    )
    if identity.fragment_id is None:
        raise JobIdError(
            "fragment_id is required to name a job directory (accession={0}, "
            "cavity={1}). Pass 'F1' explicitly for single-fragment AlphaFold "
            "models; do not let it be inferred.".format(
                identity.accession, identity.cavity_index
            )
        )
    return DELIM.join(
        [
            identity.drugbank_id,
            identity.accession,
            identity.gene if identity.gene is not None else NULL_GENE,
            identity.fragment_id,
            "{0}{1}".format(_CAVITY_PREFIX, identity.cavity_index),
        ]
    )


def parse_job_dir(name: Any) -> JobIdentity:
    """Parse a new-scheme job directory name (or a path to one).

    Round-trips with :func:`format_job_dir` for every input that function
    accepts.  Anything else raises :class:`JobIdParseError` -- there is no
    fallback to the legacy parser, because guessing which scheme a name uses is
    how identity gets fabricated.  Use :func:`parse_legacy_job_dir` explicitly.
    """
    base = _basename(name)
    parts = base.split(DELIM)
    if len(parts) != 5:
        raise JobIdParseError(
            "{0!r} is not a job directory name: expected 5 {1!r}-separated "
            "fields (drugbank_id, accession, gene, fragment, cavity), got "
            "{2}. Legacy names must go through parse_legacy_job_dir().".format(
                base, DELIM, len(parts)
            )
        )
    drug, accession, gene, fragment, cavity = parts
    if not _FRAGMENT_RE.match(fragment):
        raise JobIdParseError(
            "{0!r}: fragment field {1!r} is not 'F<k>'".format(base, fragment)
        )
    cavity_match = _NEW_CAVITY_RE.match(cavity)
    if not cavity_match:
        raise JobIdParseError(
            "{0!r}: cavity field {1!r} is not '{2}<n>'".format(
                base, cavity, _CAVITY_PREFIX
            )
        )
    try:
        return JobIdentity(
            drugbank_id=drug,
            accession=accession,
            gene=None if gene == NULL_GENE else gene,
            fragment_id=fragment,
            cavity_index=int(cavity_match.group(1)),
        )
    except JobIdError as exc:
        raise JobIdParseError("{0!r}: {1}".format(base, exc))


def parse_legacy_job_dir(
    name: Any,
    known_accessions: Collection[str],
) -> JobIdentity:
    """Parse an old-scheme name ``{drug}_{gene}_{accession}_cavity_{n}``.

    Right-anchored, and *the accession set is the authority*:

    1. strip a trailing ``_cavity_<N>``;
    2. strip an optional ``_-F<k>-model_v<n>`` fragment suffix;
    3. take the **rightmost** remaining ``_``-token that is a member of
       *known_accessions*.

    Handles hyphenated symbols (``DB00157_MT-ND1_P03886_cavity_1``), the
    literal gene ``nan`` (``DB00035_nan_P08173_cavity_3`` -> ``gene=None``) and
    the multi-fragment form
    (``DB00091_CAMLG_P08F94_-F13-model_v1_cavity_10`` -> fragment ``F13``).
    Verified: 24,173/24,173 real directories resolve, 0 ambiguous.

    The returned ``accession`` is authoritative (it is the receptor that was
    physically docked); ``gene`` is the untrusted label from the old name and
    may name a different protein entirely.  Relabel with
    ``IdentityIndex.label(accession)``.

    *known_accessions* is required and must be non-empty: without it the third
    token would have to be trusted positionally, which is the assumption that
    failed on 1,777 directories.
    """
    if isinstance(known_accessions, str):
        raise JobIdError(
            "known_accessions must be a collection of accessions, got the bare "
            "string {0!r}; iterating it would test single characters".format(
                known_accessions
            )
        )
    accessions = {
        str(a).strip().upper() for a in known_accessions if str(a).strip()
    }
    if not accessions:
        raise JobIdError(
            "parse_legacy_job_dir() requires a non-empty known_accessions set "
            "(e.g. IdentityIndex.accessions()). Refusing to fall back to a "
            "positional parse -- that is what failed on 1,777 directories."
        )

    base = _basename(name)
    cavity_match = _LEGACY_CAVITY_RE.search(base)
    if not cavity_match:
        raise JobIdParseError(
            "{0!r}: no trailing '_cavity_<N>'; not a legacy job directory "
            "name".format(base)
        )
    cavity_index = int(cavity_match.group(1))
    rest = base[: cavity_match.start()]

    fragment_id = None
    fragment_match = _LEGACY_FRAGMENT_RE.search(rest)
    if fragment_match:
        fragment_id = "F{0}".format(int(fragment_match.group(1)))
        rest = rest[: fragment_match.start()]

    tokens = rest.split("_")
    accession_position = None
    for position in range(len(tokens) - 1, -1, -1):
        if tokens[position].strip().upper() in accessions:
            accession_position = position
            break
    if accession_position is None:
        raise JobIdParseError(
            "{0!r}: no token in {1} is a member of the {2}-accession set. The "
            "accession set is the authority; a name that carries no known "
            "accession must be recorded as unresolved, not "
            "guessed.".format(base, tokens, len(accessions))
        )
    if accession_position == 0:
        raise JobIdParseError(
            "{0!r}: the accession {1!r} is the first token, so the name carries "
            "no drug id".format(base, tokens[0])
        )

    accession = tokens[accession_position].strip().upper()
    drugbank_id = tokens[0]
    gene_tokens = tokens[1:accession_position]
    gene = "_".join(gene_tokens) if gene_tokens else None

    try:
        return JobIdentity(
            drugbank_id=drugbank_id,
            accession=accession,
            gene=gene,
            fragment_id=fragment_id,
            cavity_index=cavity_index,
        )
    except JobIdError as exc:
        raise JobIdParseError("{0!r}: {1}".format(base, exc))


# --------------------------------------------------------------------------
# job_identity.json -- the authority
# --------------------------------------------------------------------------


def write_job_identity(
    directory: Union[str, "os.PathLike[str]"],
    identity: JobIdentity,
    **provenance: Any,
) -> str:
    """Write ``job_identity.json`` into *directory*; return its path.

    Provenance keys are free-form and go in alongside the core fields --
    ``sample_type``, ``identity_index_sha256``, ``code_revision``,
    ``provenance``, ``source_job`` are the ones ``REBUILD_PLAN.md`` expects.
    They may not shadow a core field.

    The write is atomic (temp file + ``os.replace``) so a killed docking run
    cannot leave a half-written identity, and the bytes are deterministic
    (fixed core order, provenance keys sorted).
    """
    if not isinstance(identity, JobIdentity):
        raise JobIdError(
            "identity must be a JobIdentity, got {0!r}".format(type(identity).__name__)
        )
    clashes = sorted(set(provenance) & (set(_CORE_FIELDS) | {"schema_version", "dir_name"}))
    if clashes:
        raise JobIdError(
            "provenance may not shadow core identity field(s): {0}".format(
                ", ".join(clashes)
            )
        )
    try:
        dir_name = identity.dir_name()
    except JobIdError as exc:
        raise JobIdError(
            "cannot write {0}: {1}".format(JOB_IDENTITY_FILENAME, exc)
        )
    record: Dict[str, Any] = {"schema_version": JOB_IDENTITY_SCHEMA_VERSION}
    record.update(identity.to_dict())
    record["dir_name"] = dir_name
    for key in sorted(provenance):
        record[key] = provenance[key]

    try:
        payload = json.dumps(record, indent=2, sort_keys=False) + "\n"
    except TypeError as exc:
        raise JobIdError(
            "provenance is not JSON-serialisable: {0}".format(exc)
        )

    directory_str = os.fspath(directory)
    os.makedirs(directory_str, exist_ok=True)
    target = os.path.join(directory_str, JOB_IDENTITY_FILENAME)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=directory_str,
        prefix=".job_identity.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, target)
    except BaseException:
        if os.path.exists(handle.name):
            os.unlink(handle.name)
        raise
    return target


def read_job_identity_record(
    directory: Union[str, "os.PathLike[str]"],
) -> Dict[str, Any]:
    """Read the full ``job_identity.json`` record (identity + provenance).

    Accepts the job directory or the JSON file itself.  A missing file raises
    ``FileNotFoundError`` -- it never falls back to parsing the directory name,
    because a name is a convenience and this file is the authority.
    """
    path = os.fspath(directory)
    if os.path.isdir(path) or not path.endswith(JOB_IDENTITY_FILENAME):
        path = os.path.join(path, JOB_IDENTITY_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "{0} not found. It is the authority on job identity; for a legacy "
            "directory call parse_legacy_job_dir(name, index.accessions()) "
            "explicitly rather than trusting the folder name.".format(path)
        )
    with open(path, "r", encoding="utf-8") as handle:
        try:
            record = json.load(handle)
        except ValueError as exc:
            raise JobIdError("{0} is not valid JSON: {1}".format(path, exc))
    if not isinstance(record, dict):
        raise JobIdError(
            "{0} must contain a JSON object, got {1}".format(
                path, type(record).__name__
            )
        )
    version = record.get("schema_version")
    if version != JOB_IDENTITY_SCHEMA_VERSION:
        raise JobIdError(
            "{0}: unsupported schema_version {1!r} (this code understands {2}). "
            "Refusing to read fields whose meaning may have "
            "changed.".format(path, version, JOB_IDENTITY_SCHEMA_VERSION)
        )
    return record


def read_job_identity(directory: Union[str, "os.PathLike[str]"]) -> JobIdentity:
    """Read ``job_identity.json`` and return its :class:`JobIdentity`."""
    return JobIdentity.from_dict(read_job_identity_record(directory))
