#!/usr/bin/env python3
"""build_identity_index.py -- build the ONE authoritative protein identity index.

This script replaces ``prepare_uniprot_mapping.py`` (``FIX_PLAN.md`` section 2),
which resolved proteins by *free-text gene-symbol search* and broadcast the
queried symbol onto every hit.  4,585 accessions collapsed onto 1,795 symbols
(mean 2.55, max 9) and, because ``dict(zip(Mapped_Gene_Name, Entry))``
(``batch_dock.py:2034``) is last-duplicate-wins, **the surviving accession became
the receptor that got docked**.  A labelling bug selected the structure.

What this script is
-------------------
* the **sole network consumer** in the repository, and only with ``--fetch``;
* fully offline otherwise -- a missing input is an *error*, never a fallback to
  some other universe (``FIX_PLAN.md`` G8);
* deterministic to the byte: no ``set`` iteration, no dict-insertion-order
  dependence, no ``drop_duplicates`` over an unordered concat.  Every tie-break
  is total and explicit.

Resolution direction
--------------------
The only sanctioned key is the **UniProt accession**, resolved from the
**Entrez Gene ID**.  A gene symbol is a *label* produced from an accession and
is never an input to selection.  This script therefore never searches by
symbol; it reads accession-keyed records out of authoritative files.

Tie-breaks (total, in this order)
---------------------------------
When several accessions claim one Entrez Gene ID, the canonical accession is:

1. ``reviewed`` (Swiss-Prot ``sp|``) before unreviewed (TrEMBL ``tr|``);
2. present in the AlphaFold/UniProt reference proteome before absent;
3. lowest accession lexicographically.

The accessions that lose stay in the index (retrievable by accession, correctly
labelled) with ``entrez_id`` null, and every demotion is written to
``unresolved_targets.csv``.  Nothing is ever dropped without a reason.

Guarantees enforced at build time (``FIX_PLAN.md`` section 1.4).  The build
**refuses to write** if any of them fails:

===  =========================================================================
G1   one row per accession                        (``assert_unique_key``)
G2   ``n_in == n_resolved + n_rejected``, rejects written with a reason
     (``assert_total_accounting``, both on the accession and the Entrez axis)
G3   deterministic: byte-identical output for the same inputs, independent of
     input row order and ``PYTHONHASHSEED``   (``--check-determinism``; the
     unit tests assert it directly)
G4   ``primary_symbol in symbols`` and ``entrez(accession(e)) == e``
     (``assert_self_consistent_labels`` against the raw source symbols, plus
     ``IdentityIndex.assert_self_consistent``)
G5   ``entrez -> accession`` collapse factor **exactly 1**
     (``assert_collapse_factor_one``)
G7   isoform accessions (``P12345-2``) and non-human entries excluded
===  =========================================================================

Modes
-----
**Authoritative** (``--idmapping`` + ``--proteome`` [+ ``--hgnc``]) -- the real
thing.  Provenance ``UNIPROT_AUTHORITATIVE``.

**Reduced** (``--from-legacy-gene-names``) -- the documented fallback for
*today*, before anyone has fetched the UniProt files.  It rebuilds a smaller
index from the **trustworthy** ``Gene Names`` column of the campaign's
``uniprot_gene_mapping.csv`` (4,585 accessions; ``Entry`` <-> ``Entry Name`` is
a true bijection).  The corrupt ``Mapped_Gene_Name`` column is never read -- it
is not even loaded.  The result is stamped ``REDUCED_LOCAL`` in
``identity_index.meta.json`` *and* in the parquet's own key/value metadata, so
no stage can mistake it for the authoritative index.  It satisfies every
invariant above; it is simply smaller, and (without ``--hgnc``) carries no
Entrez ids.

Examples
--------
::

    # authoritative, offline, from files already on disk
    python build_identity_index.py \\
        --idmapping /opt/data/uniprot/HUMAN_9606_idmapping.dat.gz \\
        --proteome  /opt/data/uniprot/UP000005640_9606.fasta.gz \\
        --hgnc      /opt/data/hgnc/hgnc_complete_set.txt \\
        --out       $NEW/identity/identity_index.parquet

    # one-time fetch (the only network access in this repository)
    python build_identity_index.py --fetch --out $NEW/identity/identity_index.parquet

    # today's unblocker: reduced index from the campaign's own Gene Names column
    python build_identity_index.py \\
        --from-legacy-gene-names $OLD/uniprot_gene_mapping.csv \\
        --out $NEW/identity/identity_index_reduced.parquet

Set ``SOURCE_DATE_EPOCH`` to make ``identity_index.meta.json`` byte-reproducible
too (the parquet always is).

Python 3.9 compatible: no PEP 604 unions at runtime, no ``match``.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:  # so `python $REPO/build_identity_index.py` works
    sys.path.insert(0, REPO_ROOT)

from dock_selectivity.identity import (  # noqa: E402
    IdentityIndex,
    Protein,
    file_sha256,
    is_accession,
)
from dock_selectivity.identity import (  # noqa: E402
    InvariantViolation as _IdentityInvariantViolation,
)
from dock_selectivity.invariants import (  # noqa: E402
    InvariantViolation,
    assert_collapse_factor_one,
    assert_deterministic,
    assert_deterministic_across_hash_seeds,
    assert_no_row_growth,
    assert_self_consistent_labels,
    assert_total_accounting,
    assert_unique_key,
    write_invariant_report,
)

__all__ = [
    "BUILDER_VERSION",
    "PROVENANCE_UNIPROT",
    "PROVENANCE_REDUCED",
    "PROVENANCE_METADATA_KEY",
    "SOURCE_SPECS",
    "UNRESOLVED_COLUMNS",
    "Assembly",
    "BuildInputError",
    "BuildResult",
    "HgncTable",
    "RawEntry",
    "RejectRow",
    "ResolvedSource",
    "SourceSpec",
    "assemble_records",
    "build",
    "build_from_options",
    "build_parser",
    "default_cache_dir",
    "default_meta_path",
    "default_notes_path",
    "default_unresolved_path",
    "load_legacy_gene_names",
    "main",
    "parquet_fingerprint_for_argv",
    "parse_hgnc",
    "parse_idmapping",
    "parse_proteome_fasta",
    "read_index_provenance",
    "resolve_sources",
    "run",
    "unique_sorted",
    "write_outputs",
]

logger = logging.getLogger("build_identity_index")

#: Bumped whenever the *content* of the emitted index could change for identical
#: inputs.  Recorded in the meta so a stale index is identifiable.
BUILDER_VERSION = "1.0.0"

#: Meta/parquet provenance stamps.  ``REDUCED_LOCAL`` must never be mistaken for
#: the authoritative index -- that is the whole point of stamping it.
PROVENANCE_UNIPROT = "UNIPROT_AUTHORITATIVE"
PROVENANCE_REDUCED = "REDUCED_LOCAL"

#: Parquet key/value metadata key carrying the provenance stamp.
PROVENANCE_METADATA_KEY = b"dock_selectivity.provenance"
BUILDER_METADATA_KEY = b"dock_selectivity.builder"

#: ``unresolved_targets.csv`` columns.  The first four are exactly
#: ``identity.Resolution.as_reject_row()``, so a downstream consumer that reads
#: only those keeps working.
UNRESOLVED_COLUMNS = ("key", "key_type", "status", "disposition", "reason", "candidates")

#: The four ``HUMAN_9606_idmapping.dat`` record types this build needs.  Every
#: other type (RefSeq, Ensembl, PDB, ...) is skipped at parse time.
IDMAPPING_TYPES = ("UniProtKB-ID", "GeneID", "Gene_Name", "Gene_Synonym")

#: Columns required of a legacy ``uniprot_gene_mapping.csv``.  ``Mapped_Gene_Name``
#: is deliberately absent: it is the corrupt column and is never loaded.
LEGACY_REQUIRED_COLUMNS = ("Entry", "Entry Name", "Gene Names", "Organism")

#: Substring that marks a human row in the legacy file's ``Organism`` column.
HUMAN_ORGANISM_TOKEN = "homo sapiens"

#: Suffix of a human UniProtKB entry name.  Used only to *exclude* obviously
#: non-human rows (G7); an unknown entry name is never guessed at.
HUMAN_ENTRY_NAME_SUFFIX = "_HUMAN"


@dataclass(frozen=True)
class SourceSpec:
    """A fetchable input file: where it comes from and what it provides."""

    role: str
    filename: str
    url: str
    required_for_uniprot_mode: bool
    provides: str


SOURCE_SPECS: Dict[str, SourceSpec] = {
    "idmapping": SourceSpec(
        role="idmapping",
        filename="HUMAN_9606_idmapping.dat.gz",
        url=(
            "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
            "knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz"
        ),
        required_for_uniprot_mode=True,
        provides="accession <-> GeneID (Entrez), Gene_Name, Gene_Synonym, UniProtKB-ID",
    ),
    "proteome": SourceSpec(
        role="proteome",
        filename="UP000005640_9606.fasta.gz",
        url=(
            "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
            "knowledgebase/reference_proteomes/Eukaryota/UP000005640/"
            "UP000005640_9606.fasta.gz"
        ),
        required_for_uniprot_mode=True,
        provides="sp|/tr| review status and reference-proteome membership (tie-break)",
    ),
    "hgnc": SourceSpec(
        role="hgnc",
        filename="hgnc_complete_set.txt",
        url=(
            "https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/"
            "hgnc_complete_set.txt"
        ),
        required_for_uniprot_mode=False,
        provides="prev/alias symbols, approved symbol per Entrez id, cross-check",
    ),
}

#: UniProt release marker, fetched opportunistically so the meta can name it.
RELEASE_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/complete/reldate.txt"
)

#: Both flavours of the invariant exception.  ``identity.py`` still defines its
#: own class (see the note in the module docstring of ``invariants.py``); until
#: it adopts the canonical one, catching a single class would miss half the
#: failures.
INVARIANT_ERRORS = (InvariantViolation, _IdentityInvariantViolation)


class BuildInputError(Exception):
    """An input file is missing, unreadable, or structurally wrong.

    Deliberately **not** an :class:`InvariantViolation`: nothing was measured
    and found wanting, the build simply cannot start.  It is also deliberately
    not a silent fallback -- ``extract_alphafold_models.py:214`` warned and then
    processed 13,975 structures instead of 1,005, which is the pattern
    ``FIX_PLAN.md`` G8 bans.
    """


# ==========================================================================
# small deterministic helpers
# ==========================================================================


def unique_sorted(values: Sequence[Any]) -> List[Any]:
    """De-duplicate and sort.  The only de-duplication primitive used here.

    ``sorted(dict.fromkeys(...))`` rather than ``list(set(...))``: the result is
    a total order, so it cannot depend on ``PYTHONHASHSEED``.  That single
    substitution is the fix for ``prepare_uniprot_mapping.py:51``.
    """
    return sorted(dict.fromkeys(values))


def _open_text(path: str) -> Any:
    """Open a text file, transparently handling ``.gz``."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return open(path, "rt", encoding="utf-8", errors="replace", newline="")


def _atomic_write_bytes(path: str, payload: bytes) -> str:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "wb", dir=parent, delete=False, suffix=".tmp"
    )
    try:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    return path


def _atomic_write_text(path: str, text: str) -> str:
    return _atomic_write_bytes(path, text.encode("utf-8"))


def _build_timestamp() -> str:
    """UTC build timestamp; honours ``SOURCE_DATE_EPOCH`` for reproducibility."""
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch:
        try:
            stamp = datetime.datetime.utcfromtimestamp(int(epoch))
        except (TypeError, ValueError):
            raise BuildInputError(
                "SOURCE_DATE_EPOCH={0!r} is not an integer unix timestamp".format(epoch)
            )
    else:
        stamp = datetime.datetime.utcnow()
    return stamp.replace(microsecond=0).isoformat() + "Z"


def _git_revision() -> Optional[str]:
    """HEAD of the repository holding this script, or ``None``.  Local only."""
    try:
        completed = subprocess.run(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git
        return None
    if completed.returncode != 0:  # pragma: no cover - not a repo
        return None
    return completed.stdout.decode("utf-8", "replace").strip() or None


# ==========================================================================
# input resolution:  local files, or --fetch.  Never a silent substitution.
# ==========================================================================


@dataclass(frozen=True)
class ResolvedSource:
    """An input file that exists, with everything needed to pin it."""

    role: str
    path: str
    sha256: str
    n_bytes: int
    url: Optional[str]
    fetched: bool

    def as_meta(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "path": os.path.abspath(self.path),
            "sha256": self.sha256,
            "bytes": self.n_bytes,
            "url": self.url,
            "fetched_now": self.fetched,
        }


def _urlopen(url: str, timeout: int = 300) -> Any:
    """Indirection so tests can prove the offline path never opens a socket."""
    from urllib.request import Request, urlopen  # local import: offline-safe

    request = Request(url, headers={"User-Agent": "dock_selectivity/build_identity_index"})
    return urlopen(request, timeout=timeout)  # noqa: S310 - fixed https URLs only


def _download(url: str, destination: str, timeout: int = 300) -> str:
    """Download ``url`` to ``destination`` atomically.  Only reachable via ``--fetch``."""
    logger.info("fetching %s -> %s", url, destination)
    try:
        response = _urlopen(url, timeout=timeout)
        try:
            payload = response.read()
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        raise BuildInputError(
            "download failed for {0}: {1}. Fetch it by hand and pass the local "
            "path instead -- this script never substitutes a different input "
            "when one is missing.".format(url, exc)
        )
    if not payload:
        raise BuildInputError(
            "download of {0} returned 0 bytes; refusing to build an index from "
            "an empty source".format(url)
        )
    return _atomic_write_bytes(destination, payload)


def _fetch_release(cache_dir: str, refresh: bool, timeout: int = 60) -> Optional[str]:
    """Best-effort UniProt release string.  Never fatal."""
    path = os.path.join(cache_dir, "reldate.txt")
    if refresh or not os.path.exists(path):
        try:
            _download(RELEASE_URL, path, timeout=timeout)
        except BuildInputError as exc:  # pragma: no cover - network shape
            logger.warning("could not record UniProt release: %s", exc)
            return None
    try:
        with open(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "Release" in line:
                    return line.strip()
    except OSError:  # pragma: no cover - unreadable cache
        return None
    return None


def resolve_sources(
    roles_to_paths: Mapping[str, Optional[str]],
    required_roles: Sequence[str],
    fetch: bool = False,
    refresh: bool = False,
    cache_dir: Optional[str] = None,
    notes: Optional[List[str]] = None,
) -> Dict[str, ResolvedSource]:
    """Turn requested roles into existing, hashed files.

    An explicitly supplied path always wins and is **never** fetched over.  A
    **required** file that is absent raises :class:`BuildInputError` naming the
    flag, the expected filename and the download URL -- it does not quietly
    proceed with fewer inputs, because that would change the scientific
    universe.

    A **declared-optional** file (only ``hgnc``) that cannot be downloaded is a
    recorded note rather than a failure.  That is not the banned fallback
    pattern: HGNC contributes synonyms, a tie-break hint and a cross-check, and
    cannot add or remove a single accession from the index.
    """
    resolved: Dict[str, ResolvedSource] = {}
    for role in sorted(roles_to_paths):
        spec = SOURCE_SPECS.get(role)
        given = roles_to_paths.get(role)
        required = role in required_roles
        path = given
        fetched = False

        if path is not None and not os.path.exists(path):
            raise BuildInputError(
                "--{0} points at a file that does not exist: {1}".format(
                    role.replace("_", "-"), path
                )
            )

        if path is None and fetch and spec is not None:
            target_dir = cache_dir or default_cache_dir()
            os.makedirs(target_dir, exist_ok=True)
            path = os.path.join(target_dir, spec.filename)
            if os.path.exists(path) and not refresh:
                logger.info("using cached %s (%s); pass --refresh to re-download",
                            role, path)
            else:
                try:
                    _download(spec.url, path)
                except BuildInputError:
                    if required:
                        raise
                    logger.warning(
                        "optional input %r could not be downloaded; continuing "
                        "without it", role
                    )
                    if notes is not None:
                        notes.append(
                            "optional input {0!r} was not available ({1}); the "
                            "index was built without it. It contributes {2} and "
                            "cannot change which accessions exist.".format(
                                role, spec.url, spec.provides
                            )
                        )
                    continue
                fetched = True

        if path is None:
            if not required:
                continue
            raise BuildInputError(_missing_input_message(role, spec))

        resolved[role] = ResolvedSource(
            role=role,
            path=path,
            sha256=file_sha256(path),
            n_bytes=os.path.getsize(path),
            url=(spec.url if (spec is not None and fetched) else None),
            fetched=fetched,
        )
    return resolved


def _missing_input_message(role: str, spec: Optional[SourceSpec]) -> str:
    flag = "--" + role.replace("_", "-")
    if spec is None:  # pragma: no cover - all roles have specs
        return "{0} is required but was not supplied".format(flag)
    return (
        "required input {0!r} is missing and --fetch was not given, so this "
        "build refuses to run.\n"
        "  It provides: {1}\n"
        "  Either pass  {2} /path/to/{3}\n"
        "  or re-run with --fetch to download it once from\n"
        "      {4}\n"
        "  (a cached copy is reused; --refresh forces a re-download).\n"
        "  This script will NOT fall back to a smaller or different identity "
        "source: choosing your own universe when an input is missing is the "
        "defect that extracted 13,975 structures instead of 1,005.\n"
        "  If you have no UniProt files today, build the clearly-labelled "
        "reduced index instead:\n"
        "      --from-legacy-gene-names $OLD/uniprot_gene_mapping.csv".format(
            role, spec.provides, flag, spec.filename, spec.url
        )
    )


def default_cache_dir() -> str:
    """Where ``--fetch`` puts downloads (``$XDG_CACHE_HOME`` aware)."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "dock_selectivity", "uniprot")


# ==========================================================================
# parsers -- accession-keyed, never symbol-keyed
# ==========================================================================


@dataclass
class RawEntry:
    """Everything one accession claims, straight from the source files."""

    accession: str
    entry_name: Optional[str] = None
    gene_names: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    entrez_ids: List[int] = field(default_factory=list)
    reviewed: bool = False
    in_proteome: bool = False
    proteome_symbol: Optional[str] = None


def _note_excluded(excluded: Dict[str, str], accession: str, reason: str) -> None:
    """Record an accession the build refuses to index.  First reason wins."""
    excluded.setdefault(accession, reason)


def parse_idmapping(
    path: str,
    excluded: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, RawEntry], Dict[str, str], Dict[str, Any]]:
    """Parse ``HUMAN_9606_idmapping.dat[.gz]``.

    The file is a 3-column TSV ``accession<TAB>id_type<TAB>id_value`` with one
    row per fact.  Only :data:`IDMAPPING_TYPES` are read.

    Isoform accessions (``P30518-2``) are excluded here rather than mangled --
    guarantee G7 -- and every one is recorded, so ``n_in == n_resolved +
    n_rejected`` still balances.

    Returns ``(entries, excluded, stats)``; ``entries`` is keyed by accession.
    """
    entries: Dict[str, RawEntry] = {}
    excluded = {} if excluded is None else excluded
    n_lines = 0
    n_used = 0
    n_malformed = 0
    n_isoform_rows = 0
    wanted = frozenset(IDMAPPING_TYPES)

    with _open_text(path) as handle:
        for raw_line in handle:
            n_lines += 1
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                n_malformed += 1
                continue
            accession, id_type, value = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if id_type not in wanted or not accession or not value:
                continue
            if "-" in accession:
                n_isoform_rows += 1
                canonical = accession.split("-", 1)[0]
                _note_excluded(
                    excluded,
                    accession,
                    "isoform accession excluded (G7); canonical form is "
                    "{0}".format(canonical),
                )
                continue
            n_used += 1
            entry = entries.get(accession)
            if entry is None:
                entry = RawEntry(accession=accession)
                entries[accession] = entry
            if id_type == "UniProtKB-ID":
                if entry.entry_name is None:
                    entry.entry_name = value
            elif id_type == "GeneID":
                try:
                    gene_id = int(value)
                except ValueError:
                    n_malformed += 1
                    continue
                if gene_id > 0:
                    entry.entrez_ids.append(gene_id)
            elif id_type == "Gene_Name":
                entry.gene_names.append(value)
            elif id_type == "Gene_Synonym":
                entry.synonyms.append(value)

    stats = {
        "n_lines": n_lines,
        "n_rows_used": n_used,
        "n_rows_malformed": n_malformed,
        "n_isoform_rows_skipped": n_isoform_rows,
        "n_accessions": len(entries),
    }
    logger.info(
        "idmapping: %d lines, %d usable rows, %d accessions, %d isoform rows skipped",
        n_lines, n_used, len(entries), n_isoform_rows,
    )
    return entries, excluded, stats


def parse_proteome_fasta(path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Parse a UniProt proteome FASTA for review status and membership.

    Only the ``>`` headers are read::

        >sp|P30518|V2R_HUMAN Vasopressin V2 receptor OS=Homo sapiens GN=AVPR2 ...

    ``sp|`` is Swiss-Prot (reviewed), ``tr|`` is TrEMBL (unreviewed).  Both facts
    -- review status and *presence in the reference proteome* -- are tie-break
    inputs for the canonical-accession choice, and both must come from data
    rather than be guessed.
    """
    entries: Dict[str, Dict[str, Any]] = {}
    n_headers = 0
    n_unparsed = 0
    n_isoform = 0
    with _open_text(path) as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            n_headers += 1
            header = line[1:].rstrip("\r\n")
            fields = header.split("|", 2)
            if len(fields) != 3:
                n_unparsed += 1
                continue
            database, accession, remainder = fields[0].strip(), fields[1].strip(), fields[2]
            if "-" in accession:
                n_isoform += 1
                continue
            entry_name = remainder.split(None, 1)[0].strip() if remainder.strip() else None
            symbol = None
            for token in remainder.split():
                if token.startswith("GN="):
                    symbol = token[3:].strip() or None
                    break
            existing = entries.get(accession)
            record = {
                "reviewed": database == "sp",
                "entry_name": entry_name,
                "symbol": symbol,
            }
            if existing is None:
                entries[accession] = record
            else:
                # A duplicated accession in one FASTA is not expected; merge
                # conservatively rather than let the last line win silently.
                existing["reviewed"] = existing["reviewed"] or record["reviewed"]
                existing["entry_name"] = existing["entry_name"] or record["entry_name"]
                existing["symbol"] = existing["symbol"] or record["symbol"]
    stats = {
        "n_headers": n_headers,
        "n_accessions": len(entries),
        "n_reviewed": sum(1 for r in entries.values() if r["reviewed"]),
        "n_headers_unparsed": n_unparsed,
        "n_isoform_headers_skipped": n_isoform,
    }
    logger.info(
        "proteome: %d headers, %d accessions, %d reviewed",
        n_headers, len(entries), stats["n_reviewed"],
    )
    return entries, stats


@dataclass
class HgncTable:
    """The optional HGNC cross-reference, keyed the only two legal ways."""

    entrez_to_symbol: Dict[int, str] = field(default_factory=dict)
    accession_to_entrez: Dict[str, List[int]] = field(default_factory=dict)
    accession_to_symbols: Dict[str, List[str]] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)


def parse_hgnc(path: str) -> HgncTable:
    """Parse ``hgnc_complete_set.txt`` (TSV) for symbol/Entrez cross-reference.

    Used for three things and no others: extra *synonyms* (``prev_symbol``,
    ``alias_symbol``), an ``entrez -> approved symbol`` map used only as a
    tie-break, and an accession-keyed Entrez cross-check.  It is never used to
    turn a symbol into an accession.
    """
    table = HgncTable()
    n_rows = 0
    n_approved = 0
    n_multi_uniprot = 0
    with _open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            n_rows += 1
            if (row.get("status") or "").strip() != "Approved":
                continue
            n_approved += 1
            symbol = (row.get("symbol") or "").strip()
            entrez_raw = (row.get("entrez_id") or "").strip()
            entrez_id: Optional[int] = None
            if entrez_raw:
                try:
                    entrez_id = int(float(entrez_raw))
                except ValueError:
                    entrez_id = None
            if entrez_id is not None and symbol:
                table.entrez_to_symbol.setdefault(entrez_id, symbol)
            accessions = [
                a.strip()
                for a in (row.get("uniprot_ids") or "").replace(",", "|").split("|")
                if a.strip()
            ]
            if len(accessions) > 1:
                n_multi_uniprot += 1
            extra_symbols = [symbol] if symbol else []
            for column in ("prev_symbol", "alias_symbol"):
                for token in (row.get(column) or "").replace(",", "|").split("|"):
                    token = token.strip()
                    if token:
                        extra_symbols.append(token)
            for accession in accessions:
                if "-" in accession or not accession:
                    continue
                if entrez_id is not None:
                    table.accession_to_entrez.setdefault(accession, []).append(entrez_id)
                if extra_symbols:
                    table.accession_to_symbols.setdefault(accession, []).extend(
                        extra_symbols
                    )
    table.stats = {
        "n_rows": n_rows,
        "n_approved": n_approved,
        "n_entrez": len(table.entrez_to_symbol),
        "n_accessions": len(table.accession_to_symbols),
        "n_rows_with_multiple_uniprot_ids": n_multi_uniprot,
    }
    logger.info(
        "hgnc: %d rows, %d approved, %d accessions cross-referenced",
        n_rows, n_approved, len(table.accession_to_symbols),
    )
    return table


def load_legacy_gene_names(
    path: str,
    excluded: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, RawEntry], Dict[str, str], Dict[str, Any]]:
    """REDUCED_LOCAL loader: the campaign's ``uniprot_gene_mapping.csv``.

    Reads **only** ``Entry``, ``Entry Name``, ``Gene Names`` and ``Organism``.
    ``Mapped_Gene_Name`` -- the column that is wrong on 57.7% of rows and that
    selected the docked receptor -- is not in the column list, so it is never
    even loaded.  ``Gene Names`` is UniProt's own symbol list for the accession
    on that row and is trustworthy; ``Entry`` <-> ``Entry Name`` is a verified
    bijection over the 4,585 rows.

    Missing required columns raise :class:`BuildInputError`.  In particular a
    missing ``Organism`` column is an error and not a "skip the filter"
    fallback: silently widening the universe is the pattern G8 bans.
    """
    import pandas as pd  # local import: the module stays cheap to import

    excluded = {} if excluded is None else excluded
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception as exc:  # noqa: BLE001
        raise BuildInputError(
            "cannot read legacy mapping {0}: {1}".format(path, exc)
        )
    present = [str(c) for c in header.columns]
    missing = [c for c in LEGACY_REQUIRED_COLUMNS if c not in present]
    if missing:
        raise BuildInputError(
            "legacy mapping {0} is missing required column(s): {1}. Present: {2}. "
            "All four are required; in particular 'Organism' is not optional -- "
            "dropping the human filter because the column is absent would widen "
            "the universe silently.".format(path, ", ".join(missing), ", ".join(present))
        )
    frame = pd.read_csv(
        path,
        usecols=list(LEGACY_REQUIRED_COLUMNS),
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )

    entries: Dict[str, RawEntry] = {}
    n_rows = int(len(frame))
    n_non_human = 0
    n_blank_accession = 0
    n_no_symbols = 0
    conflicts: List[str] = []

    def cell(row: Mapping[str, Any], column: str) -> str:
        """Read a CSV cell as a stripped string.

        ``x or ''`` is wrong here: an empty field comes back as ``float('nan')``
        and **NaN is truthy**, so the fallback never fires.  163 of the 4,585
        real rows have an empty ``Gene Names``, which is how this was found.
        """
        value = row.get(column)
        if value is None or not isinstance(value, str):
            if value is None or value != value:  # None or NaN
                return ""
            value = str(value)
        return value.strip()

    for row in frame.to_dict("records"):
        accession = cell(row, "Entry")
        if not accession:
            n_blank_accession += 1
            continue
        organism = cell(row, "Organism")
        if HUMAN_ORGANISM_TOKEN not in organism.lower():
            n_non_human += 1
            _note_excluded(
                excluded,
                accession,
                "non-human organism {0!r} excluded (G7)".format(organism),
            )
            continue
        entry_name = cell(row, "Entry Name") or None
        symbols = [s for s in cell(row, "Gene Names").split() if s]
        if not symbols:
            n_no_symbols += 1
        existing = entries.get(accession)
        if existing is None:
            entries[accession] = RawEntry(
                accession=accession,
                entry_name=entry_name,
                gene_names=symbols[:1],
                synonyms=symbols[1:],
            )
            continue
        # G1: a duplicate accession may not be resolved by "last row wins".
        same = (
            existing.entry_name == entry_name
            and existing.gene_names + existing.synonyms == symbols
        )
        if not same:
            conflicts.append(
                "{0}: {1!r}/{2} vs {3!r}/{4}".format(
                    accession,
                    existing.entry_name,
                    existing.gene_names + existing.synonyms,
                    entry_name,
                    symbols,
                )
            )

    if conflicts:
        raise BuildInputError(
            "legacy mapping {0} has {1} accession(s) appearing twice with "
            "DIFFERENT content: {2}. Refusing to pick one -- last-duplicate-wins "
            "over disagreeing rows is precisely the bug this index replaces. "
            "De-duplicate the file explicitly first.".format(
                path, len(conflicts), "; ".join(sorted(conflicts)[:10])
            )
        )

    stats = {
        "n_rows": n_rows,
        "n_accessions": len(entries),
        "n_non_human_rows_excluded": n_non_human,
        "n_blank_accession_rows": n_blank_accession,
        "n_rows_without_gene_names": n_no_symbols,
        "columns_read": list(LEGACY_REQUIRED_COLUMNS),
        "mapped_gene_name_read": False,
    }
    logger.info(
        "legacy gene names: %d rows -> %d accessions (%d non-human excluded)",
        n_rows, len(entries), n_non_human,
    )
    return entries, excluded, stats


# ==========================================================================
# assembly -- the total, explicit tie-breaks
# ==========================================================================


@dataclass
class RejectRow:
    """One row of ``unresolved_targets.csv``.  Nothing is dropped without one."""

    key: str
    key_type: str
    status: str
    disposition: str
    reason: str
    candidates: str = ""

    def as_row(self) -> Dict[str, str]:
        return {
            "key": self.key,
            "key_type": self.key_type,
            "status": self.status,
            "disposition": self.disposition,
            "reason": self.reason,
            "candidates": self.candidates,
        }


@dataclass
class Assembly:
    """The output of :func:`assemble_records`, before any invariant is checked."""

    records: List[Protein]
    rejects: List[RejectRow]
    entrez_pairs: List[Tuple[int, str]]
    source_symbols: Dict[str, Tuple[str, ...]]
    stats: Dict[str, Any]


def _accession_rank(entry: RawEntry) -> Tuple[int, int, str]:
    """Total tie-break for "which accession owns this Entrez id".

    ``reviewed`` > unreviewed, then in-proteome > not, then lowest accession
    lexicographically.  Every component is a decidable fact from a source file;
    none of them is an iteration order.
    """
    return (
        0 if entry.reviewed else 1,
        0 if entry.in_proteome else 1,
        entry.accession,
    )


def _primary_symbol_for(
    entry: RawEntry,
    hgnc: Optional[HgncTable],
) -> Tuple[str, str]:
    """Choose the display symbol for one accession.  Returns ``(symbol, how)``.

    Order of preference, all total:

    1. the HGNC approved symbol for this accession, when it is also one of the
       accession's own UniProt gene names;
    2. the reference proteome's ``GN=`` value, when it is one of them;
    3. the lexicographically smallest UniProt ``Gene_Name``;
    4. the proteome ``GN=`` value, then the HGNC approved symbol, when UniProt
       lists no gene name at all;
    5. the accession itself -- never a blank, which a join could match.
    """
    names = unique_sorted(entry.gene_names)
    if names:
        if hgnc is not None:
            approved = unique_sorted(hgnc.accession_to_symbols.get(entry.accession, []))
            for symbol in approved:
                if symbol in names:
                    return symbol, "hgnc_approved"
        if entry.proteome_symbol and entry.proteome_symbol in names:
            return entry.proteome_symbol, "proteome_gn"
        return names[0], "uniprot_gene_name_lexicographic_min"
    if entry.proteome_symbol:
        return entry.proteome_symbol, "proteome_gn_only"
    if hgnc is not None:
        approved = unique_sorted(hgnc.accession_to_symbols.get(entry.accession, []))
        if approved:
            return approved[0], "hgnc_only"
    return entry.accession, "accession_fallback"


def assemble_records(
    raw_entries: Mapping[str, RawEntry],
    excluded: Mapping[str, str],
    hgnc: Optional[HgncTable] = None,
    on_entrez_collision: str = "resolve",
) -> Assembly:
    """Turn raw per-accession facts into the final record set.

    Two selection passes, both total and both recorded:

    * **accession -> Entrez.**  An accession listing several GeneIDs keeps one:
      preferring the GeneID whose HGNC approved symbol equals this accession's
      own primary symbol, then a GeneID no other accession claims, then the
      lowest GeneID.  Each discarded pair is written to the reject ledger.
    * **Entrez -> accession.**  Among the accessions that kept a given GeneID,
      the canonical one is chosen by :func:`_accession_rank`.  The others stay
      in the index with ``entrez_id`` null (still retrievable, still correctly
      labelled) and each demotion is recorded.  This is what makes G5 hold *by
      construction* rather than by hope.
    """
    if on_entrez_collision not in ("resolve", "fail"):
        raise ValueError(
            "on_entrez_collision must be 'resolve' or 'fail', got "
            "{0!r}".format(on_entrez_collision)
        )

    rejects: List[RejectRow] = []
    kept: Dict[str, RawEntry] = {}
    excluded_local: Dict[str, str] = dict(excluded)

    # ---- exclusions (G7 + accession syntax) ------------------------------
    for accession in sorted(raw_entries):
        entry = raw_entries[accession]
        if "-" in accession:
            excluded_local.setdefault(
                accession,
                "isoform accession excluded (G7); canonical form is "
                "{0}".format(accession.split("-", 1)[0]),
            )
            continue
        if not is_accession(accession):
            excluded_local.setdefault(
                accession,
                "not a valid UniProtKB accession; a gene symbol or entry name "
                "in the accession column is the original defect",
            )
            continue
        if (
            entry.entry_name
            and not entry.entry_name.upper().endswith(HUMAN_ENTRY_NAME_SUFFIX)
        ):
            excluded_local.setdefault(
                accession,
                "entry name {0!r} is not human (G7: non-human excluded)".format(
                    entry.entry_name
                ),
            )
            continue
        kept[accession] = entry

    for accession in sorted(excluded_local):
        rejects.append(
            RejectRow(
                key=accession,
                key_type="accession",
                status="UNKNOWN",
                disposition="excluded",
                reason=excluded_local[accession],
            )
        )

    # ---- symbols and primary symbol --------------------------------------
    primary: Dict[str, str] = {}
    symbols: Dict[str, Tuple[str, ...]] = {}
    source_symbols: Dict[str, Tuple[str, ...]] = {}
    primary_sources: Dict[str, int] = {}
    n_accession_fallback = 0

    for accession in sorted(kept):
        entry = kept[accession]
        chosen, how = _primary_symbol_for(entry, hgnc)
        primary_sources[how] = primary_sources.get(how, 0) + 1
        from_source = unique_sorted(entry.gene_names + entry.synonyms)
        source_symbols[accession] = tuple(from_source)
        extra: List[str] = []
        if hgnc is not None:
            extra = list(hgnc.accession_to_symbols.get(accession, []))
        others = [s for s in unique_sorted(from_source + extra) if s != chosen]
        primary[accession] = chosen
        symbols[accession] = tuple([chosen] + others)
        if how == "accession_fallback":
            n_accession_fallback += 1
            rejects.append(
                RejectRow(
                    key=accession,
                    key_type="accession",
                    status="UNKNOWN",
                    disposition="retained",
                    reason=(
                        "no gene symbol in any source; primary_symbol set to the "
                        "accession itself rather than left blank"
                    ),
                )
            )

    # ---- pass 1: accession -> one Entrez id ------------------------------
    raw_pairs: List[Tuple[str, int]] = []
    for accession in sorted(kept):
        candidates = unique_sorted(kept[accession].entrez_ids)
        if not candidates and hgnc is not None:
            candidates = unique_sorted(hgnc.accession_to_entrez.get(accession, []))
        for gene_id in candidates:
            raw_pairs.append((accession, gene_id))

    accessions_per_gid: Dict[int, int] = {}
    for _, gene_id in raw_pairs:
        accessions_per_gid[gene_id] = accessions_per_gid.get(gene_id, 0) + 1

    candidates_by_accession: Dict[str, List[int]] = {}
    for accession, gene_id in raw_pairs:
        candidates_by_accession.setdefault(accession, []).append(gene_id)

    chosen_gid: Dict[str, int] = {}
    n_multi_gid = 0
    for accession in sorted(candidates_by_accession):
        candidates = unique_sorted(candidates_by_accession[accession])
        if len(candidates) == 1:
            chosen_gid[accession] = candidates[0]
            continue
        n_multi_gid += 1

        def preference(gene_id: int, _accession: str = accession) -> Tuple[int, int, int]:
            symbol_match = 1
            if hgnc is not None:
                approved = hgnc.entrez_to_symbol.get(gene_id)
                if approved is not None and approved == primary.get(_accession):
                    symbol_match = 0
            sole_claim = 0 if accessions_per_gid.get(gene_id, 0) == 1 else 1
            return (symbol_match, sole_claim, gene_id)

        winner = min(candidates, key=preference)
        chosen_gid[accession] = winner
        for loser in candidates:
            if loser == winner:
                continue
            rejects.append(
                RejectRow(
                    key=accession,
                    key_type="accession",
                    status="AMBIGUOUS",
                    disposition="retained",
                    reason=(
                        "accession lists {0} GeneIDs; kept {1} (accession -> entrez "
                        "must be single-valued). GeneID {2} was not assigned to "
                        "this accession.".format(len(candidates), winner, loser)
                    ),
                    candidates=";".join(str(c) for c in candidates),
                )
            )

    # ---- pass 2: Entrez id -> one canonical accession ---------------------
    claimants: Dict[int, List[str]] = {}
    for accession in sorted(chosen_gid):
        claimants.setdefault(chosen_gid[accession], []).append(accession)

    colliding = {gid: accs for gid, accs in claimants.items() if len(accs) > 1}
    if colliding and on_entrez_collision == "fail":
        worst = sorted(colliding.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:10]
        raise InvariantViolation(
            "G5 entrez -> accession collapse factor must be exactly 1, but {0} "
            "GeneID(s) are claimed by more than one accession and "
            "--on-entrez-collision=fail was requested: {1}. Re-run with "
            "--on-entrez-collision=resolve to apply the documented tie-break "
            "(reviewed > unreviewed > in-proteome > lowest accession), or fix "
            "the source files.".format(
                len(colliding),
                "; ".join(
                    "{0} -> {1}".format(gid, ", ".join(sorted(accs)))
                    for gid, accs in worst
                ),
            ),
            name="G5 entrez -> accession (raw, pre-tie-break)",
        )

    entrez_of: Dict[str, Optional[int]] = {a: None for a in sorted(kept)}
    n_demoted = 0
    for gene_id in sorted(claimants):
        contenders = sorted(claimants[gene_id])
        winner = min(contenders, key=lambda a: _accession_rank(kept[a]))
        entrez_of[winner] = gene_id
        for loser in contenders:
            if loser == winner:
                continue
            n_demoted += 1
            rejects.append(
                RejectRow(
                    key=loser,
                    key_type="accession",
                    status="AMBIGUOUS",
                    disposition="retained",
                    reason=(
                        "GeneID {0} is claimed by {1} accessions; canonical is {2} "
                        "(reviewed={3}, in_proteome={4}). {5} stays in the index "
                        "with entrez_id=null -- it is still retrievable by "
                        "accession and still correctly labelled.".format(
                            gene_id,
                            len(contenders),
                            winner,
                            kept[winner].reviewed,
                            kept[winner].in_proteome,
                            loser,
                        )
                    ),
                    candidates=";".join(contenders),
                )
            )

    # GeneIDs that lost every candidate accession in pass 1.
    all_gids = unique_sorted([gid for _, gid in raw_pairs])
    resolved_gids = set(claimants)
    lost_gids = [gid for gid in all_gids if gid not in resolved_gids]
    for gene_id in lost_gids:
        owners = unique_sorted([a for a, g in raw_pairs if g == gene_id])
        rejects.append(
            RejectRow(
                key=str(gene_id),
                key_type="entrez",
                status="UNKNOWN",
                disposition="excluded",
                reason=(
                    "every accession listing this GeneID was assigned to a "
                    "different GeneID, so it has no canonical accession; edges "
                    "keyed on it will resolve as UNKNOWN"
                ),
                candidates=";".join(owners),
            )
        )

    # ---- records ---------------------------------------------------------
    records: List[Protein] = []
    for accession in sorted(kept):
        entry = kept[accession]
        records.append(
            Protein(
                accession=accession,
                entry_name=entry.entry_name or accession,
                primary_symbol=primary[accession],
                symbols=symbols[accession],
                entrez_id=entrez_of.get(accession),
                reviewed=bool(entry.reviewed),
            )
        )

    entrez_pairs = sorted(
        (gid, acc) for acc, gid in entrez_of.items() if gid is not None
    )

    stats = {
        "n_accessions_seen": len(raw_entries) + len(excluded),
        "n_accessions_indexed": len(records),
        "n_accessions_excluded": len(excluded_local),
        "n_entrez_pairs_raw": len(raw_pairs),
        "n_entrez_ids_raw": len(all_gids),
        "n_entrez_ids_indexed": len(entrez_pairs),
        "n_entrez_ids_lost": len(lost_gids),
        "n_accessions_with_multiple_geneids": n_multi_gid,
        "n_accessions_demoted_to_null_entrez": n_demoted,
        "n_entrez_collisions_resolved": len(colliding),
        "n_primary_symbol_from_accession_fallback": n_accession_fallback,
        "primary_symbol_sources": dict(sorted(primary_sources.items())),
        "n_entry_name_fallback": sum(
            1 for a in sorted(kept) if not kept[a].entry_name
        ),
        "n_reviewed": sum(1 for r in records if r.reviewed),
        "n_in_proteome": sum(1 for a in sorted(kept) if kept[a].in_proteome),
        "n_reject_rows": len(rejects),
    }
    return Assembly(
        records=records,
        rejects=rejects,
        entrez_pairs=entrez_pairs,
        source_symbols=source_symbols,
        stats=stats,
    )


# ==========================================================================
# the build
# ==========================================================================


@dataclass
class BuildResult:
    """Everything a successful build produced, before anything is written."""

    index: IdentityIndex
    rejects: List[RejectRow]
    reports: List[Dict[str, Any]]
    stats: Dict[str, Any]
    provenance: str
    sources: Dict[str, ResolvedSource]
    uniprot_release: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def build(
    sources: Mapping[str, ResolvedSource],
    provenance: str,
    on_entrez_collision: str = "resolve",
    strict_cross_check: bool = False,
    min_records: int = 1,
    uniprot_release: Optional[str] = None,
) -> BuildResult:
    """Parse, assemble, and check.  Writes nothing.

    Every guarantee in ``FIX_PLAN.md`` section 1.4 is enforced here, so a caller
    that gets a :class:`BuildResult` back is holding an index that already
    passed.  A failure raises before any file is created.
    """
    if provenance not in (PROVENANCE_UNIPROT, PROVENANCE_REDUCED):
        raise ValueError("unknown provenance {0!r}".format(provenance))

    excluded: Dict[str, str] = {}
    stats: Dict[str, Any] = {"provenance": provenance}
    notes: List[str] = []

    # ---- parse -----------------------------------------------------------
    if provenance == PROVENANCE_UNIPROT:
        raw_entries, excluded, idmapping_stats = parse_idmapping(
            sources["idmapping"].path, excluded
        )
        stats["idmapping"] = idmapping_stats
    else:
        raw_entries, excluded, legacy_stats = load_legacy_gene_names(
            sources["legacy"].path, excluded
        )
        stats["legacy_gene_names"] = legacy_stats
        notes.append(
            "REDUCED_LOCAL: built from the 'Gene Names' column of a legacy "
            "uniprot_gene_mapping.csv. The corrupt 'Mapped_Gene_Name' column was "
            "not read. This index is smaller than the authoritative one and is "
            "NOT a substitute for it."
        )

    n_accessions_before_enrichment = len(raw_entries)

    proteome: Dict[str, Dict[str, Any]] = {}
    if "proteome" in sources:
        proteome, proteome_stats = parse_proteome_fasta(sources["proteome"].path)
        stats["proteome"] = proteome_stats
    elif provenance == PROVENANCE_REDUCED:
        notes.append(
            "no --proteome supplied: review status is UNKNOWN and recorded as "
            "reviewed=False for every row. It is only a tie-break input, and "
            "with no Entrez ids there is nothing to tie-break, but do not read "
            "the 'reviewed' column of this index as a claim about Swiss-Prot."
        )
        stats["review_status_source"] = "unknown (REDUCED_LOCAL without --proteome)"

    for accession in sorted(raw_entries):
        info = proteome.get(accession)
        if info is None:
            continue
        entry = raw_entries[accession]
        entry.reviewed = bool(info["reviewed"])
        entry.in_proteome = True
        entry.proteome_symbol = info.get("symbol")
        if entry.entry_name is None:
            entry.entry_name = info.get("entry_name")

    hgnc: Optional[HgncTable] = None
    if "hgnc" in sources:
        hgnc = parse_hgnc(sources["hgnc"].path)
        stats["hgnc"] = hgnc.stats

    n_accessions_after_enrichment = len(raw_entries)

    # ---- assemble --------------------------------------------------------
    assembly = assemble_records(
        raw_entries,
        excluded,
        hgnc=hgnc,
        on_entrez_collision=on_entrez_collision,
    )
    stats["assembly"] = assembly.stats

    index = IdentityIndex(
        assembly.records,
        source_path=None,
        source_sha256=None,
        validate=False,  # validated explicitly below, with named reports
    )
    frame = index.to_frame()

    reports: List[Dict[str, Any]] = []

    # ---- G1: one row per accession ---------------------------------------
    reports.append(assert_unique_key(frame, ["accession"], name="G1 one row per accession"))

    # ---- an enrichment may not add rows ----------------------------------
    # The proteome and HGNC passes are *lookups*: they annotate accessions that
    # are already there.  A join that quietly fanned them out is the
    # parse_prepare.py:1171 failure (+410,087 rows), so it is asserted, not
    # assumed.
    reports.append(
        assert_no_row_growth(
            n_accessions_before_enrichment,
            n_accessions_after_enrichment,
            name="enrichment (proteome + HGNC) must not add rows",
            allow_shrinkage=False,
        )
    )

    # ---- G2: total accounting, both axes ---------------------------------
    # n_in is the number of DISTINCT accession strings the sources offered,
    # counted independently of the output, so this cannot be a tautology.
    n_excluded_accessions = sum(
        1
        for r in assembly.rejects
        if r.key_type == "accession" and r.disposition == "excluded"
    )
    reports.append(
        assert_total_accounting(
            n_in=assembly.stats["n_accessions_seen"],
            n_resolved=len(assembly.records),
            n_rejected=n_excluded_accessions,
            name="G2 accession accounting",
        )
    )
    reports.append(
        assert_total_accounting(
            n_in=assembly.stats["n_entrez_ids_raw"],
            n_resolved=assembly.stats["n_entrez_ids_indexed"],
            n_rejected=assembly.stats["n_entrez_ids_lost"],
            name="G2 entrez accounting",
        )
    )

    # ---- G5: entrez -> accession collapse factor exactly 1 ----------------
    reports.append(
        assert_collapse_factor_one(
            assembly.entrez_pairs,
            name="G5 entrez -> accession",
        )
    )

    # ---- G4a: primary_symbol really belongs to its own accession ----------
    # Truth comes from the raw symbols parsed out of the source file, NOT from
    # the frame under test: a check that consumes its own subject cannot fail.
    checkable = frame[
        frame["accession"].isin(
            [a for a, syms in sorted(assembly.source_symbols.items()) if syms]
        )
    ]
    label_report = assert_self_consistent_labels(
        checkable,
        key_col="accession",
        label_col="primary_symbol",
        truth_lookup={a: syms for a, syms in assembly.source_symbols.items() if syms},
        name="G4 primary_symbol belongs to its own accession",
    )
    label_report["n_rows_without_source_symbol"] = int(len(frame) - len(checkable))
    reports.append(label_report)

    # ---- G4b: entrez(accession(e)) == e -----------------------------------
    reports.append(_assert_entrez_round_trip(index, assembly.entrez_pairs))

    # ---- G1/G4/G5/G7 again, through identity.py itself --------------------
    index.assert_self_consistent()
    reports.append(
        {
            "check": "identity_index_self_consistent",
            "name": "G1/G4/G5/G7 via IdentityIndex.assert_self_consistent",
            "n_records": len(index),
            "ok": True,
        }
    )

    # ---- G3: in-process determinism of the assembly ------------------------
    reports.append(
        assert_deterministic(
            lambda: [
                (p.accession, p.entry_name, p.primary_symbol, p.symbols,
                 p.entrez_id, p.reviewed)
                for p in assemble_records(
                    raw_entries, excluded, hgnc=hgnc,
                    on_entrez_collision=on_entrez_collision,
                ).records
            ],
            name="G3 assembly is deterministic in-process",
            runs=2,
        )
    )

    # ---- optional independent cross-check against the proteome GN= --------
    if proteome:
        reports.append(
            _cross_check_against_proteome(frame, proteome, strict=strict_cross_check)
        )

    # ---- an empty index is a failed build, not a valid answer -------------
    if len(index) < max(1, int(min_records)):
        raise InvariantViolation(
            "the build produced {0} record(s), below the required minimum of "
            "{1}. An empty or near-empty identity index would silently shrink "
            "the study universe to nothing; check that the input files are the "
            "ones you meant.".format(len(index), max(1, int(min_records))),
            name="minimum record count",
        )

    stats["cardinality_report"] = index.cardinality_report()
    stats["n_records"] = len(index)
    stats["n_reject_rows"] = len(assembly.rejects)

    return BuildResult(
        index=index,
        rejects=assembly.rejects,
        reports=reports,
        stats=stats,
        provenance=provenance,
        sources=dict(sources),
        uniprot_release=uniprot_release,
        notes=notes,
    )


def _assert_entrez_round_trip(
    index: IdentityIndex,
    entrez_pairs: Sequence[Tuple[int, str]],
) -> Dict[str, Any]:
    """G4, second half: ``entrez(accession(e)) == e`` for every resolved id."""
    problems: List[str] = []
    for gene_id, accession in entrez_pairs:
        try:
            protein = index.by_accession(accession)
        except KeyError:
            # Report it as the named invariant failure rather than letting a
            # KeyError escape: a caller's `except KeyError` must never be able
            # to swallow a broken index.
            problems.append(
                "{0} -> {1} -> <accession absent from the index>".format(
                    gene_id, accession
                )
            )
            continue
        if protein.entrez_id != gene_id:
            problems.append(
                "{0} -> {1} -> {2}".format(gene_id, accession, protein.entrez_id)
            )
    report = {
        "check": "entrez_round_trip",
        "name": "G4 entrez(accession(e)) == e",
        "n_pairs": len(entrez_pairs),
        "n_violations": len(problems),
        "examples": problems[:5],
        "ok": not problems,
    }
    if problems:
        raise InvariantViolation(
            "G4 round trip failed for {0} Entrez id(s): {1}. The index's "
            "entrez -> accession and accession -> entrez views disagree.".format(
                len(problems), "; ".join(problems[:5])
            ),
            name=report["name"],
            report=report,
        )
    return report


def _cross_check_against_proteome(
    frame: Any,
    proteome: Mapping[str, Mapping[str, Any]],
    strict: bool = False,
) -> Dict[str, Any]:
    """Compare each ``primary_symbol`` with the proteome FASTA's ``GN=`` value.

    A genuinely independent source, so it is a real cross-check rather than a
    tautology -- but UniProt's ``GN=`` and its ``Gene_Name`` record legitimately
    differ for a handful of entries, so a mismatch is *reported* by default and
    only fatal under ``--strict-cross-check``.
    """
    truth = {
        accession: (info.get("symbol"),)
        for accession, info in sorted(proteome.items())
        if info.get("symbol")
    }
    subset = frame[frame["accession"].isin(sorted(truth))]
    report = assert_self_consistent_labels(
        subset,
        key_col="accession",
        label_col="primary_symbol",
        truth_lookup=truth,
        name="cross-check: primary_symbol vs reference-proteome GN=",
        warn_only=not strict,
    )
    report["strict"] = bool(strict)
    report["n_compared"] = int(len(subset))
    return report


# ==========================================================================
# writing
# ==========================================================================


def _write_parquet(frame: Any, path: str, provenance: str, builder: Mapping[str, Any]) -> str:
    """Write the index parquet atomically, stamping provenance in its metadata.

    The stamp lives in the parquet's own key/value metadata as well as in the
    meta JSON, because downstream code loads the parquet and may never look at
    the sidecar.  A ``REDUCED_LOCAL`` index must announce itself wherever it is
    opened.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[PROVENANCE_METADATA_KEY] = provenance.encode("utf-8")
    metadata[BUILDER_METADATA_KEY] = json.dumps(
        dict(builder), sort_keys=True
    ).encode("utf-8")
    table = table.replace_schema_metadata(metadata)

    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("wb", dir=parent, delete=False, suffix=".tmp")
    handle.close()
    try:
        pq.write_table(table, handle.name)
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    return path


def read_index_provenance(path: str) -> Optional[str]:
    """Read the provenance stamp out of an ``identity_index.parquet``.

    ``None`` means the file carries no stamp -- treat that as untrusted, not as
    authoritative.
    """
    import pyarrow.parquet as pq

    metadata = pq.read_schema(path).metadata or {}
    value = metadata.get(PROVENANCE_METADATA_KEY)
    return value.decode("utf-8") if value is not None else None


def _write_reject_csv(rejects: Sequence[RejectRow], path: str) -> str:
    """Write a reject/notes ledger -- deterministically ordered, atomically."""
    rows = sorted(
        (r.as_row() for r in rejects),
        key=lambda r: (r["key_type"], r["key"], r["status"], r["disposition"], r["reason"]),
    )
    buffer: List[str] = []
    writer = csv.DictWriter(
        _StringSink(buffer), fieldnames=list(UNRESOLVED_COLUMNS), lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return _atomic_write_text(path, "".join(buffer))


class _StringSink:
    """Minimal write-only file object, so the CSV bytes are built in memory."""

    def __init__(self, sink: List[str]) -> None:
        self._sink = sink

    def write(self, text: str) -> int:
        self._sink.append(text)
        return len(text)


def write_outputs(
    result: BuildResult,
    out_path: str,
    meta_path: Optional[str] = None,
    unresolved_path: Optional[str] = None,
    invariants_path: Optional[str] = None,
    notes_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Write parquet + meta JSON + two ledgers + the invariant report.

    Called only after :func:`build` returned, i.e. only after every guarantee
    held.  Every write is atomic.

    Two ledgers, because "unresolved" must mean what it says:

    * ``unresolved_targets.csv`` -- accessions that are **not** in the index and
      why.  Its row count is exactly ``accounting['n_rejected']``, so
      ``audit.py`` can cross-check meta against index against ledger.
    * ``identity_index.notes.csv`` -- decisions recorded about rows that *are*
      in the index (an accession demoted to a null ``entrez_id``, a discarded
      ``(accession, GeneID)`` pair, an Entrez id left with no canonical
      accession, a primary symbol that fell back to the accession).  These are
      provenance, not rejections, and counting them as rejects would make the
      accounting lie.
    """
    out_path = os.path.abspath(out_path)
    meta_path = os.path.abspath(meta_path or default_meta_path(out_path))
    unresolved_path = os.path.abspath(
        unresolved_path or default_unresolved_path(out_path)
    )
    notes_path = os.path.abspath(notes_path or default_notes_path(out_path))
    invariants_path = os.path.abspath(
        invariants_path or (os.path.splitext(out_path)[0] + ".invariants.json")
    )

    builder = _builder_record()
    frame = result.index.to_frame()
    _write_parquet(frame, out_path, result.provenance, builder)
    unresolved_rows: List[RejectRow] = []
    notes_rows: List[RejectRow] = []
    for reject in result.rejects:
        target = (
            unresolved_rows
            if (reject.key_type == "accession" and reject.disposition == "excluded")
            else notes_rows
        )
        target.append(reject)
    _write_reject_csv(unresolved_rows, unresolved_path)
    _write_reject_csv(notes_rows, notes_path)
    write_invariant_report(result.reports, invariants_path)

    index_sha = file_sha256(out_path)
    meta = {
        "schema_version": 1,
        "provenance": result.provenance,
        "warning": (
            None
            if result.provenance == PROVENANCE_UNIPROT
            else (
                "REDUCED_LOCAL index: rebuilt from a legacy uniprot_gene_mapping.csv "
                "'Gene Names' column, not from the authoritative UniProt id-mapping "
                "files. Smaller coverage; Entrez ids only if --hgnc was supplied. "
                "Do not present results built on it as using the authoritative index."
            )
        ),
        "build_timestamp_utc": _build_timestamp(),
        "builder": builder,
        "uniprot_release": result.uniprot_release,
        "sources": [
            result.sources[role].as_meta() for role in sorted(result.sources)
        ],
        "outputs": {
            "index_path": out_path,
            "index_sha256": index_sha,
            "index_bytes": os.path.getsize(out_path),
            "meta_path": meta_path,
            "unresolved_path": unresolved_path,
            "unresolved_sha256": file_sha256(unresolved_path),
            "n_unresolved_rows": len(unresolved_rows),
            "notes_path": notes_path,
            "notes_sha256": file_sha256(notes_path),
            "n_notes_rows": len(notes_rows),
            "invariants_path": invariants_path,
        },
        # G2 in the shape audit.py reads: n_in is counted from the sources,
        # n_resolved is the index, n_rejected is exactly the row count of
        # unresolved_targets.csv. All three are cross-checkable against
        # different files, which is the point.
        "accounting": {
            "n_in": int(result.stats["assembly"]["n_accessions_seen"]),
            "n_resolved": len(result.index),
            "n_rejected": len(unresolved_rows),
            "axis": "accession",
            "note": (
                "the Entrez axis is accounted separately in "
                "row_counts.assembly and in the 'G2 entrez accounting' "
                "invariant; its rejects are listed in notes_path"
            ),
        },
        "row_counts": result.stats,
        "cardinality_report": result.index.cardinality_report(),
        "invariants": result.reports,
        "notes": result.notes,
    }
    _atomic_write_text(
        meta_path,
        json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
    )
    return meta


def default_meta_path(out_path: str) -> str:
    """``identity_index.parquet`` -> ``identity_index.meta.json``."""
    return os.path.splitext(out_path)[0] + ".meta.json"


def default_unresolved_path(out_path: str) -> str:
    """``unresolved_targets.csv``, beside the index."""
    return os.path.join(os.path.dirname(os.path.abspath(out_path)), "unresolved_targets.csv")


def default_notes_path(out_path: str) -> str:
    """``identity_index.notes.csv`` -- recorded decisions about indexed rows."""
    return os.path.splitext(out_path)[0] + ".notes.csv"


def _builder_record() -> Dict[str, Any]:
    record = {
        "script": os.path.basename(__file__),
        "builder_version": BUILDER_VERSION,
        "code_revision": _git_revision(),
        "python": sys.version.split()[0],
    }
    try:
        import pandas as pd

        record["pandas"] = pd.__version__
    except ImportError:  # pragma: no cover - pandas is a hard dep
        record["pandas"] = None
    try:
        import pyarrow

        record["pyarrow"] = pyarrow.__version__
    except ImportError:  # pragma: no cover - pyarrow is a hard dep
        record["pyarrow"] = None
    return record


# ==========================================================================
# CLI
# ==========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_identity_index.py",
        description=(
            "Build the authoritative protein identity index "
            "(identity_index.parquet + identity_index.meta.json + "
            "unresolved_targets.csv). The only script permitted to touch the "
            "network, and only with --fetch."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  authoritative : --idmapping FILE --proteome FILE [--hgnc FILE]\n"
            "  reduced       : --from-legacy-gene-names uniprot_gene_mapping.csv\n"
            "                  (stamped REDUCED_LOCAL; unblocks the reuse phases "
            "today)\n"
        ),
    )
    parser.add_argument("--idmapping", help="HUMAN_9606_idmapping.dat[.gz]")
    parser.add_argument("--proteome", help="UP000005640_9606.fasta[.gz]")
    parser.add_argument("--hgnc", help="hgnc_complete_set.txt (optional, recommended)")
    parser.add_argument(
        "--from-legacy-gene-names",
        dest="from_legacy_gene_names",
        help=(
            "REDUCED fallback: build from the trustworthy 'Gene Names' column of "
            "a legacy uniprot_gene_mapping.csv. The corrupt 'Mapped_Gene_Name' "
            "column is never read. Output is stamped REDUCED_LOCAL."
        ),
    )
    parser.add_argument("--out", required=True, help="output identity_index.parquet")
    parser.add_argument("--meta", help="output meta JSON (default: <out>.meta.json)")
    parser.add_argument(
        "--unresolved",
        help=(
            "output rejects CSV -- accessions NOT in the index, with reasons "
            "(default: unresolved_targets.csv beside --out)"
        ),
    )
    parser.add_argument(
        "--notes",
        help=(
            "output notes CSV -- recorded decisions about rows that ARE in the "
            "index, e.g. an accession demoted to a null entrez_id "
            "(default: <out>.notes.csv)"
        ),
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="download missing inputs (THE only network access in this repo)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="with --fetch, re-download even if a cached copy exists",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="where --fetch stores downloads (default: {0})".format(default_cache_dir()),
    )
    parser.add_argument(
        "--on-entrez-collision",
        choices=("resolve", "fail"),
        default="resolve",
        help=(
            "'resolve' (default) applies the documented total tie-break and "
            "records every demotion; 'fail' refuses to build when any GeneID is "
            "claimed by more than one accession"
        ),
    )
    parser.add_argument(
        "--strict-cross-check",
        action="store_true",
        help="make the proteome GN= cross-check fatal instead of reported",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=1,
        help="refuse to write an index smaller than this (default: 1)",
    )
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help=(
            "G3 gate: rebuild under several PYTHONHASHSEED values in "
            "subprocesses and require byte-identical parquet output"
        ),
    )
    parser.add_argument(
        "--uniprot-release",
        default=None,
        help="record this UniProt release string in the meta (offline builds)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return parser


def build_from_options(args: argparse.Namespace) -> BuildResult:
    """Resolve inputs for the selected mode and run :func:`build`."""
    reduced = bool(args.from_legacy_gene_names)
    if reduced and args.idmapping:
        raise BuildInputError(
            "--from-legacy-gene-names and --idmapping select different identity "
            "sources; pass exactly one. The reduced mode exists only for when "
            "the UniProt files are not available."
        )
    if not reduced and not args.idmapping and not args.fetch:
        raise BuildInputError(_missing_input_message("idmapping", SOURCE_SPECS["idmapping"]))

    uniprot_release = args.uniprot_release
    fetch_notes: List[str] = []
    if reduced:
        legacy = args.from_legacy_gene_names
        if not os.path.exists(legacy):
            raise BuildInputError(
                "--from-legacy-gene-names points at a file that does not exist: "
                "{0}".format(legacy)
            )
        sources: Dict[str, ResolvedSource] = {
            "legacy": ResolvedSource(
                role="legacy",
                path=legacy,
                sha256=file_sha256(legacy),
                n_bytes=os.path.getsize(legacy),
                url=None,
                fetched=False,
            )
        }
        optional = resolve_sources(
            {"proteome": args.proteome, "hgnc": args.hgnc},
            required_roles=(),
            fetch=False,  # reduced mode never fetches: it exists for offline use
            refresh=False,
            cache_dir=args.cache_dir,
            notes=fetch_notes,
        )
        sources.update(optional)
        provenance = PROVENANCE_REDUCED
    else:
        sources = resolve_sources(
            {"idmapping": args.idmapping, "proteome": args.proteome, "hgnc": args.hgnc},
            required_roles=("idmapping", "proteome"),
            fetch=bool(args.fetch),
            refresh=bool(args.refresh),
            cache_dir=args.cache_dir,
            notes=fetch_notes,
        )
        provenance = PROVENANCE_UNIPROT
        if args.fetch and uniprot_release is None:
            uniprot_release = _fetch_release(
                args.cache_dir or default_cache_dir(), bool(args.refresh)
            )

    result = build(
        sources,
        provenance=provenance,
        on_entrez_collision=args.on_entrez_collision,
        strict_cross_check=bool(args.strict_cross_check),
        min_records=int(args.min_records),
        uniprot_release=uniprot_release,
    )
    result.notes.extend(fetch_notes)
    return result


def parquet_fingerprint_for_argv(argv: Sequence[str]) -> str:
    """SHA256 of the parquet these CLI args would produce.  Writes to a temp dir.

    Used by ``--check-determinism`` from inside subprocesses running under
    different ``PYTHONHASHSEED`` values.
    """
    parser = build_parser()
    filtered = [a for a in argv if a != "--check-determinism"]
    args = parser.parse_args(filtered)
    result = build_from_options(args)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "identity_index.parquet")
        _write_parquet(
            result.index.to_frame(), path, result.provenance, _builder_record()
        )
        return file_sha256(path)


def _check_determinism(argv: Sequence[str], seeds: Sequence[int] = (0, 1, 42)) -> Dict[str, Any]:
    snippet = (
        "import json, sys\n"
        "sys.path.insert(0, {repo!r})\n"
        "import build_identity_index as b\n"
        "print(b.parquet_fingerprint_for_argv(json.loads({argv!r})))\n"
    ).format(repo=REPO_ROOT, argv=json.dumps(list(argv)))
    return assert_deterministic_across_hash_seeds(
        snippet,
        name="G3 identity_index.parquet is byte-identical across PYTHONHASHSEED",
        seeds=seeds,
    )


def run(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.  Returns a process exit code; raises nothing expected."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # basicConfig is a no-op once the root logger has handlers, so set the level
    # explicitly: a second call to run() in the same process must still honour
    # --log-level.
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        result = build_from_options(args)
    except BuildInputError as exc:
        logger.error("%s", exc)
        return 2
    except INVARIANT_ERRORS as exc:
        logger.error("refusing to write: %s", exc)
        report = getattr(exc, "report", None)
        try:
            write_invariant_report(
                [
                    {
                        "check": "build",
                        "name": getattr(exc, "name", None) or "build",
                        "ok": False,
                        "message": str(exc),
                        "detail": report or {},
                    }
                ],
                os.path.splitext(os.path.abspath(args.out))[0] + ".invariants.json",
            )
        except OSError:  # pragma: no cover - unwritable directory
            pass
        if os.path.exists(args.out):
            logger.error(
                "note: %s already exists and was NOT overwritten by this failed build",
                args.out,
            )
        return 3

    meta = write_outputs(
        result,
        out_path=args.out,
        meta_path=args.meta,
        unresolved_path=args.unresolved,
        notes_path=args.notes,
    )

    if args.check_determinism:
        report = _check_determinism(raw_argv)
        logger.info("determinism check: %s", "OK" if report.get("ok") else "FAILED")

    banner = ""
    if result.provenance == PROVENANCE_REDUCED:
        banner = (
            "\n"
            "  ############################################################\n"
            "  #  REDUCED_LOCAL identity index                            #\n"
            "  #  Built from a legacy uniprot_gene_mapping.csv, NOT from  #\n"
            "  #  the authoritative UniProt id-mapping files.             #\n"
            "  #  Provenance is stamped in the parquet and the meta.      #\n"
            "  ############################################################\n"
        )
    print(
        "{0}\nidentity index written: {1}\n"
        "  provenance      : {2}\n"
        "  records         : {3}\n"
        "  entrez ids      : {4}\n"
        "  max_collapse    : {5}   (G5: must not exceed 1)\n"
        "  sha256          : {6}\n"
        "  meta            : {7}\n"
        "  unresolved rows : {8} -> {9}\n"
        "  recorded notes  : {10} -> {11}".format(
            banner,
            meta["outputs"]["index_path"],
            meta["provenance"],
            meta["cardinality_report"]["n_records"],
            meta["cardinality_report"]["n_entrez"],
            meta["cardinality_report"]["max_collapse"],
            meta["outputs"]["index_sha256"],
            meta["outputs"]["meta_path"],
            meta["outputs"]["n_unresolved_rows"],
            meta["outputs"]["unresolved_path"],
            meta["outputs"]["n_notes_rows"],
            meta["outputs"]["notes_path"],
        )
    )
    return 0


def main() -> None:  # pragma: no cover - thin wrapper
    sys.exit(run())


if __name__ == "__main__":  # pragma: no cover
    main()
