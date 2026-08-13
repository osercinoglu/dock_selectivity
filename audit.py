#!/usr/bin/env python3
"""Read-only invariant runner for the clean rebuild -- every STOP gate lives here.

``REBUILD_PLAN.md`` puts a **STOP** gate at the end of each phase.  This is the
program those gates run.  It reads; it never writes anything except the optional
``--json`` report, and it never touches ``$OLD``.

    python3 audit.py identity --index $NEW/identity/identity_index.parquet
    python3 audit.py curated  --dir   $NEW/docking_results \\
                              --manifest $NEW/manifests/run_manifest.csv \\
                              --index  $NEW/identity/identity_index.parquet
    python3 audit.py run      --run-dir $NEW

Why an auditor at all
---------------------
The 2025 campaign had no gate that could have caught its own failure.  The one
assertion that would have -- *"the gene attached to an accession must be one of
that accession's own symbols"* -- costs three lines and is now ``gene ==
index.label(accession)``, checked here for 100% of curated jobs.  The other
lesson is subtler: ``validate_negatives()`` reported "All validations passed"
because it re-derived its ground truth from the object under test.  Everything
below therefore checks against a source *independent* of the thing being
checked: the curated tree is checked against the identity index, the manifests
against each other, and the index against its own arithmetic.

Exit codes
----------
``0``  every check ran and passed
``1``  at least one check FAILED
``2``  nothing failed, but at least one check could not run (SKIP).  A check
       that did not run is not a check that passed, so a STOP gate must treat
       this as "not cleared".  ``--allow-skips`` collapses it to 0.

Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dock_selectivity.identity import (  # noqa: E402
    ACCESSION_RE,
    IdentityIndex,
    InvariantViolation,
)
from dock_selectivity.invariants import (  # noqa: E402
    assert_collapse_factor_one,
    assert_deterministic_across_hash_seeds,
    assert_self_consistent_labels,
    write_invariant_report,
)
from dock_selectivity.jobid import (  # noqa: E402
    JOB_IDENTITY_FILENAME,
    JobIdError,
    JobIdentity,
    format_job_dir,
    read_job_identity_record,
)
from dock_selectivity.schemas import SchemaViolation, read_checked  # noqa: E402

__all__ = [
    "CheckResult",
    "PASS",
    "FAIL",
    "SKIP",
    "SPOT_CHECKS",
    "EXCLUDED_BASENAME_PATTERNS",
    "audit_identity",
    "audit_curated",
    "audit_run",
    "format_table",
    "exit_code",
    "main",
]

logger = logging.getLogger("audit")

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

#: ``(symbol, correct accession, accession the corrupt map returned)``.
#: ``REBUILD_PLAN.md`` Phase 0 names the first three; the impostor column is the
#: half that matters -- the old map did not merely mislabel these, it handed the
#: impostor's structure to the docking engines.
SPOT_CHECKS: Tuple[Tuple[str, str, str], ...] = (
    ("AVPR2", "P30518", "Q8NCT1"),
    ("TUBA1B", "P68363", "Q13509"),
    ("IL2", "P60568", "P01589"),
)

#: Must not appear in a curated tree built with ``--evidence-only``.
EXCLUDED_BASENAME_PATTERNS: Tuple[str, ...] = ("complex_*.pdb", "log_*.log")

#: Manifests ``REBUILD_PLAN.md`` section 2 expects, and their schema names.
MANIFESTS: Tuple[Tuple[str, str], ...] = (
    ("interactions_resolved.csv", "interactions_resolved"),
    ("positives_planned.csv", "positives_planned"),
    ("negatives_pool.csv", "negatives_pool"),
    ("negatives_drawn.csv", "negatives_drawn"),
    ("salvage_map.csv", "salvage_map"),
    ("redock_queue.csv", "redock_queue"),
    ("run_manifest.csv", "run_manifest"),
)

_MAX_EXAMPLES = 5


# --------------------------------------------------------------------------
# result type
# --------------------------------------------------------------------------


@dataclass
class CheckResult:
    """One check, its verdict, and enough detail to act on it."""

    name: str
    status: str
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.name,
            "status": self.status,
            "detail": self.detail,
            "data": self.data,
        }


def _result(name: str, status: str, detail: str, **data: Any) -> CheckResult:
    return CheckResult(name=name, status=status, detail=detail, data=dict(data))


def _verdict(name: str, ok: bool, pass_detail: str, fail_detail: str, **data: Any) -> CheckResult:
    return _result(name, PASS if ok else FAIL, pass_detail if ok else fail_detail, **data)


def _data(report: Dict[str, Any], *drop: str) -> Dict[str, Any]:
    """An invariant report as check ``data``.

    ``name``, ``check`` and ``ok`` are dropped: they are this module's own
    vocabulary and would collide with :func:`_verdict`'s parameters.
    """
    excluded = {"name", "check", "ok"} | set(drop)
    return {k: v for k, v in report.items() if k not in excluded}


def _examples(values: Sequence[Any], limit: int = _MAX_EXAMPLES) -> str:
    values = list(values)
    shown = ", ".join(str(v) for v in values[:limit])
    if len(values) > limit:
        shown += ", … (+{0} more)".format(len(values) - limit)
    return shown


# --------------------------------------------------------------------------
# audit identity  (REBUILD_PLAN Phase 0 STOP gate)
# --------------------------------------------------------------------------


def _determinism_snippet(index_path: str) -> str:
    """Snippet printing a digest of the index's *content and ordering*."""
    return (
        "import hashlib, sys;"
        "sys.path.insert(0, {repo!r});"
        "from dock_selectivity.identity import IdentityIndex;"
        "ix = IdentityIndex.load({path!r}, validate=False);"
        "blob = '\\n'.join('{{0}}\\t{{1}}\\t{{2}}'.format("
        "p.accession, p.primary_symbol, ';'.join(p.symbols)) for p in ix.records);"
        "print(hashlib.sha256(blob.encode('utf-8')).hexdigest())"
    ).format(repo=_REPO_ROOT, path=os.path.abspath(index_path))


def _accounting_from_meta(meta: Dict[str, Any]) -> Optional[Dict[str, int]]:
    source = meta.get("accounting") if isinstance(meta.get("accounting"), dict) else meta
    keys = ("n_in", "n_resolved", "n_rejected")
    if not all(k in source for k in keys):
        return None
    try:
        return {k: int(source[k]) for k in keys}
    except (TypeError, ValueError):
        return None


def audit_identity(
    index_path: str,
    expect_sha256: Optional[str] = None,
    meta_path: Optional[str] = None,
    unresolved_path: Optional[str] = None,
    skip_determinism: bool = False,
    prefix: str = "",
) -> List[CheckResult]:
    """G1-G7 plus the AVPR2 / TUBA1B / IL2 spot checks.

    The index is loaded with ``validate=False`` on purpose: this program's job
    is to *report* which invariant failed, not to die on the first one.
    """
    name = lambda s: prefix + s  # noqa: E731
    results: List[CheckResult] = []

    if not os.path.exists(index_path):
        return [
            _result(
                name("index_exists"),
                FAIL,
                "identity index not found: {0}. Build it with "
                "build_identity_index.py -- there is no fallback identity "
                "source.".format(index_path),
                path=index_path,
            )
        ]
    try:
        index = IdentityIndex.load(index_path, validate=False)
    except Exception as exc:  # noqa: BLE001 - report, never crash the gate
        return [
            _result(
                name("index_loads"),
                FAIL,
                "{0}: {1}".format(type(exc).__name__, exc),
                path=index_path,
            )
        ]

    report = index.cardinality_report()
    results.append(
        _result(
            name("index_loads"),
            PASS,
            "{0} accession(s), {1} Entrez id(s), sha256 {2}".format(
                report["n_records"], report["n_entrez"], (index.source_sha256 or "?")[:12]
            ),
            **_data(report)
        )
    )

    if expect_sha256 is not None:
        got = (index.source_sha256 or "").lower()
        want = expect_sha256.strip().lower()
        results.append(
            _verdict(
                name("index_sha256_pinned"),
                got == want,
                "matches the pinned digest",
                "index is {0}, plan was built against {1}. Downstream artifacts "
                "were planned against a different index.".format(got, want),
                expected=want,
                actual=got,
            )
        )

    # -- G1 ---------------------------------------------------------------
    results.append(
        _verdict(
            name("G1_one_row_per_accession"),
            report["n_duplicate_accessions"] == 0,
            "{0} distinct accession(s), no duplicates".format(report["n_accessions"]),
            "{0} accession(s) appear more than once; a dict keyed on accession "
            "would silently keep the last".format(report["n_duplicate_accessions"]),
            n_duplicate_accessions=report["n_duplicate_accessions"],
        )
    )

    # -- G7 + accession syntax --------------------------------------------
    results.append(
        _verdict(
            name("G7_no_isoform_accessions"),
            report["n_isoform_accessions"] == 0,
            "no isoform-style accessions",
            "{0} isoform accession(s); collapse them onto their canonical "
            "accession at build time".format(report["n_isoform_accessions"]),
            n_isoform_accessions=report["n_isoform_accessions"],
        )
    )
    bad_syntax = [
        p.accession for p in index.records if not ACCESSION_RE.match(p.accession)
    ]
    results.append(
        _verdict(
            name("accession_syntax"),
            not bad_syntax,
            "every accession matches the UniProtKB syntax",
            "{0} row(s) are not valid accessions ({1}); a gene symbol or entry "
            "name in the accession column is the original defect".format(
                len(bad_syntax), _examples(sorted(bad_syntax))
            ),
            n_bad=len(bad_syntax),
            examples=sorted(bad_syntax)[:_MAX_EXAMPLES],
        )
    )

    # -- G4 ---------------------------------------------------------------
    no_symbols = [p.accession for p in index.records if not p.symbols]
    mismatched = [
        "{0}(primary={1} not in {2})".format(p.accession, p.primary_symbol, list(p.symbols))
        for p in index.records
        if p.symbols and p.primary_symbol.upper() not in {s.upper() for s in p.symbols}
    ]
    results.append(
        _verdict(
            name("G4_primary_symbol_in_symbols"),
            not mismatched and not no_symbols,
            "every row's primary_symbol is one of its own symbols "
            "(the KNOWN_ISSUE regression assertion)",
            "{0} row(s) fail the KNOWN_ISSUE regression assertion, {1} row(s) "
            "have no symbols at all. Offenders: {2}".format(
                len(mismatched), len(no_symbols), _examples(mismatched + no_symbols)
            ),
            n_mismatched=len(mismatched),
            n_without_symbols=len(no_symbols),
        )
    )

    # -- G5, the guarantee whose absence caused everything ----------------
    results.append(
        _verdict(
            name("G5_entrez_collapse_factor_one"),
            report["max_collapse"] <= 1,
            "max_collapse = {0}, mean_collapse = {1} over {2} Entrez id(s)".format(
                report["max_collapse"], report["mean_collapse"], report["n_entrez"]
            ),
            "max_collapse = {0} (mean {1}); {2} Entrez id(s) claim more than one "
            "accession. The retired map had mean 2.55 / max 9 and it chose the "
            "receptor that got docked.".format(
                report["max_collapse"],
                report["mean_collapse"],
                report["n_entrez_collapsing"],
            ),
            max_collapse=report["max_collapse"],
            mean_collapse=report["mean_collapse"],
            n_entrez_collapsing=report["n_entrez_collapsing"],
        )
    )

    # -- spot checks -------------------------------------------------------
    for symbol, correct, impostor in SPOT_CHECKS:
        problems: List[str] = []
        try:
            protein = index.by_accession(correct)
        except KeyError:
            problems.append("{0} is not in the index".format(correct))
            protein = None
        if protein is not None:
            if protein.primary_symbol != symbol:
                problems.append(
                    "{0}'s primary_symbol is {1!r}, expected {2!r}".format(
                        correct, protein.primary_symbol, symbol
                    )
                )
            if not protein.has_symbol(symbol):
                problems.append(
                    "{0} does not list {1} among its symbols".format(correct, symbol)
                )
        if impostor in index:
            other = index.by_accession(impostor)
            if other.has_symbol(symbol):
                problems.append(
                    "the impostor {0} still claims {1} (its own symbols are "
                    "{2}) -- this is the exact corruption".format(
                        impostor, symbol, list(other.symbols)
                    )
                )
        carriers = [p.accession for p in index.by_symbol_ambiguous(symbol)]
        if correct not in carriers:
            problems.append(
                "{0} resolves to {1}, which does not include {2}".format(
                    symbol, carriers or "nothing", correct
                )
            )
        results.append(
            _verdict(
                name("spot_{0}".format(symbol)),
                not problems,
                "{0} -> {1} (not {2}); carriers: {3}".format(
                    symbol, correct, impostor, carriers
                ),
                "; ".join(problems),
                symbol=symbol,
                expected=correct,
                impostor=impostor,
                carriers=carriers,
            )
        )

    # -- G3: determinism ---------------------------------------------------
    if skip_determinism:
        results.append(
            _result(
                name("G3_deterministic_ordering"),
                SKIP,
                "--skip-determinism: hash-seed determinism was NOT verified",
            )
        )
    else:
        try:
            det = assert_deterministic_across_hash_seeds(
                _determinism_snippet(index_path),
                name="identity index ordering",
                seeds=(0, 1, 42),
                warn_only=True,
            )
            results.append(
                _verdict(
                    name("G3_deterministic_ordering"),
                    bool(det.get("ok")),
                    "identical content digest under PYTHONHASHSEED 0/1/42 "
                    "(checks the artifact's ordering; rebuilding it under three "
                    "seeds is build_identity_index.py's own test)",
                    "the index digest depends on PYTHONHASHSEED: {0}".format(
                        det.get("digests") or det
                    ),
                    **_data(det)
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                _result(
                    name("G3_deterministic_ordering"),
                    SKIP,
                    "could not run the subprocess check: {0}: {1}".format(
                        type(exc).__name__, exc
                    ),
                )
            )

    # -- G2: total accounting ---------------------------------------------
    meta_path = meta_path or os.path.join(
        os.path.dirname(os.path.abspath(index_path)), "identity_index.meta.json"
    )
    unresolved_path = unresolved_path or os.path.join(
        os.path.dirname(os.path.abspath(index_path)), "unresolved_targets.csv"
    )
    if not os.path.exists(meta_path):
        results.append(
            _result(
                name("G2_total_accounting"),
                SKIP,
                "{0} not found, so n_in == n_resolved + n_rejected could not be "
                "checked. build_identity_index.py must record n_in, n_resolved "
                "and n_rejected there.".format(meta_path),
                meta_path=meta_path,
            )
        )
    else:
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            accounting = _accounting_from_meta(meta if isinstance(meta, dict) else {})
        except (OSError, ValueError) as exc:
            accounting = None
            meta = {}
            logger.warning("could not read %s: %s", meta_path, exc)
        if accounting is None:
            results.append(
                _result(
                    name("G2_total_accounting"),
                    SKIP,
                    "{0} carries no n_in/n_resolved/n_rejected".format(meta_path),
                    meta_path=meta_path,
                )
            )
        else:
            unaccounted = accounting["n_in"] - (
                accounting["n_resolved"] + accounting["n_rejected"]
            )
            problems = []
            if unaccounted != 0:
                problems.append(
                    "{0} row(s) unaccounted: n_in {1} != n_resolved {2} + "
                    "n_rejected {3}".format(
                        abs(unaccounted),
                        accounting["n_in"],
                        accounting["n_resolved"],
                        accounting["n_rejected"],
                    )
                )
            if accounting["n_resolved"] != len(index):
                problems.append(
                    "meta says n_resolved {0} but the index holds {1} row(s)".format(
                        accounting["n_resolved"], len(index)
                    )
                )
            if os.path.exists(unresolved_path):
                n_rows = _count_csv_rows(unresolved_path)
                if n_rows is not None and n_rows != accounting["n_rejected"]:
                    problems.append(
                        "meta says n_rejected {0} but {1} holds {2} row(s)".format(
                            accounting["n_rejected"], unresolved_path, n_rows
                        )
                    )
            elif accounting["n_rejected"]:
                problems.append(
                    "{0} rejected row(s) but {1} does not exist -- rejects must "
                    "be recorded, never dropped".format(
                        accounting["n_rejected"], unresolved_path
                    )
                )
            results.append(
                _verdict(
                    name("G2_total_accounting"),
                    not problems,
                    "n_in {0} == n_resolved {1} + n_rejected {2}".format(
                        accounting["n_in"],
                        accounting["n_resolved"],
                        accounting["n_rejected"],
                    ),
                    "; ".join(problems),
                    **_data(accounting)
                )
            )

    # -- catch-all: anything the enumeration above missed -------------------
    try:
        index.assert_self_consistent()
        results.append(
            _result(name("index_self_consistent"), PASS, "no invariant violations")
        )
    except InvariantViolation as exc:
        results.append(_result(name("index_self_consistent"), FAIL, str(exc)))

    return results


def _count_csv_rows(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except OSError:  # pragma: no cover
        return None


# --------------------------------------------------------------------------
# audit curated  (REBUILD_PLAN Phase 5 STOP gate)
# --------------------------------------------------------------------------


@dataclass
class _JobScan:
    dir_name: str
    has_identity: bool = False
    identity_error: str = ""
    record: Dict[str, Any] = field(default_factory=dict)
    identity: Optional[JobIdentity] = None
    has_final_results: bool = False
    excluded_files: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)


def _scan_job(job_dir: str, check_excluded: bool) -> _JobScan:
    scan = _JobScan(dir_name=os.path.basename(job_dir.rstrip("/")))
    scan.has_final_results = os.path.isfile(os.path.join(job_dir, "final_results.csv"))
    identity_path = os.path.join(job_dir, JOB_IDENTITY_FILENAME)
    scan.has_identity = os.path.isfile(identity_path)
    if scan.has_identity:
        try:
            scan.record = read_job_identity_record(job_dir)
            scan.identity = JobIdentity.from_dict(scan.record)
        except (JobIdError, OSError, ValueError) as exc:
            scan.identity_error = "{0}: {1}".format(type(exc).__name__, exc)

    retention = str(scan.record.get("retention", "")) if scan.record else ""
    look_for_excluded = check_excluded and retention != "full"
    evidence = scan.record.get("evidence") if scan.record else None

    if look_for_excluded:
        for dirpath, dirnames, filenames in os.walk(job_dir):
            dirnames.sort()
            for filename in sorted(filenames):
                if any(fnmatch.fnmatch(filename, p) for p in EXCLUDED_BASENAME_PATTERNS):
                    scan.excluded_files.append(
                        os.path.relpath(os.path.join(dirpath, filename), job_dir)
                    )

    if isinstance(evidence, list):
        for entry in evidence:
            if not isinstance(entry, dict):
                scan.missing_evidence.append(str(entry))
                continue
            relpath = str(entry.get("path", ""))
            full = os.path.join(job_dir, relpath)
            if not os.path.isfile(full):
                scan.missing_evidence.append(relpath + " (absent)")
            elif int(os.stat(full).st_size) != int(entry.get("bytes", -1)):
                scan.missing_evidence.append(relpath + " (size mismatch)")
    return scan


def audit_curated(
    curated_dir: str,
    manifest_path: str,
    index_path: str,
    limit: Optional[int] = None,
    check_excluded: bool = True,
    expect: str = "subset",
    prefix: str = "",
) -> List[CheckResult]:
    """Every check ``REBUILD_PLAN.md`` Phase 5 asks for, plus the ones it implies.

    ``expect='subset'`` (default) requires every manifest row with
    ``provenance == 'salvaged_from'`` to be on disk and forbids any directory
    that is not in the manifest -- the invariant that holds from Phase 5 until
    Phase 7 finishes docking the gap.  ``expect='exact'`` additionally requires
    the newly-docked rows, i.e. the end-state gate ("directory count equals
    run_manifest.csv row count exactly").
    """
    if expect not in ("subset", "exact"):
        raise ValueError("expect must be 'subset' or 'exact', got {0!r}".format(expect))
    import pandas as pd

    name = lambda s: prefix + s  # noqa: E731
    results: List[CheckResult] = []

    if not os.path.isdir(curated_dir):
        return [
            _result(
                name("curated_dir_exists"),
                FAIL,
                "not a directory: {0}".format(curated_dir),
            )
        ]
    try:
        index = IdentityIndex.load(index_path, validate=False)
    except Exception as exc:  # noqa: BLE001
        return [
            _result(
                name("index_loads"),
                FAIL,
                "cannot load the identity index {0}: {1}: {2}. Every check below "
                "needs it -- none were run.".format(index_path, type(exc).__name__, exc),
            )
        ]
    try:
        manifest = read_checked(manifest_path, "run_manifest")
    except (SchemaViolation, InvariantViolation, OSError) as exc:
        return [
            _result(
                name("manifest_valid"),
                FAIL,
                "{0} does not satisfy the run_manifest contract: {1}. Every check "
                "below needs it -- none were run.".format(manifest_path, exc),
            )
        ]
    results.append(
        _result(
            name("manifest_valid"),
            PASS,
            "{0} row(s) satisfy the run_manifest contract".format(len(manifest)),
            n_rows=int(len(manifest)),
        )
    )

    entries = sorted(os.listdir(curated_dir))
    dirs = [e for e in entries if os.path.isdir(os.path.join(curated_dir, e))]
    strays = [e for e in entries if e not in dirs]
    results.append(
        _verdict(
            name("no_stray_files"),
            not strays,
            "no non-directory entries in the curated tree",
            "{0} stray entr(ies) that are not job directories: {1}".format(
                len(strays), _examples(strays)
            ),
            strays=strays[:_MAX_EXAMPLES],
        )
    )

    truncated = False
    scanned_dirs = dirs
    if limit is not None and len(dirs) > limit:
        scanned_dirs = dirs[:limit]
        truncated = True

    scans = [
        _scan_job(os.path.join(curated_dir, d), check_excluded=check_excluded)
        for d in scanned_dirs
    ]

    # -- counts ------------------------------------------------------------
    manifest_ids = set(manifest["job_id"].astype(str))
    salvaged_ids = set(
        manifest.loc[manifest["provenance"].astype(str) == "salvaged_from", "job_id"]
        .astype(str)
        .tolist()
    )
    disk_ids = set(dirs)
    if expect == "exact":
        expected_ids = manifest_ids
    else:
        # every salvaged row must be here; a newly-docked row may not be yet,
        # but if it is, it counts. Anything on disk and not in the manifest is
        # caught by the set check below.
        expected_ids = salvaged_ids | (manifest_ids & disk_ids)
    pending = sorted(manifest_ids - expected_ids)
    results.append(
        _verdict(
            name("dir_count_equals_manifest"),
            len(dirs) == len(expected_ids),
            "{0} director(ies) == {1} expected manifest row(s) (--expect {2}"
            "{3})".format(
                len(dirs),
                len(expected_ids),
                expect,
                "; {0} newly_docked row(s) not yet materialised".format(len(pending))
                if pending
                else "",
            ),
            "{0} director(ies) on disk vs {1} expected manifest row(s) (of {2} "
            "total, --expect {3}); a truncated pass must be detectable, not "
            "silent".format(len(dirs), len(expected_ids), len(manifest), expect),
            n_dirs=len(dirs),
            n_manifest_rows=int(len(manifest)),
            n_expected=len(expected_ids),
            expect=expect,
        )
    )
    missing_on_disk = sorted(expected_ids - disk_ids)
    not_in_manifest = sorted(disk_ids - manifest_ids)
    results.append(
        _verdict(
            name("dir_set_equals_manifest"),
            not missing_on_disk and not not_in_manifest,
            "every expected manifest job is on disk and no directory is absent "
            "from the manifest",
            "{0} expected manifest job(s) absent from disk ({1}); {2} "
            "director(ies) absent from the manifest ({3})".format(
                len(missing_on_disk),
                _examples(missing_on_disk),
                len(not_in_manifest),
                _examples(not_in_manifest),
            ),
            n_missing_on_disk=len(missing_on_disk),
            n_not_in_manifest=len(not_in_manifest),
        )
    )

    # -- per-directory presence -------------------------------------------
    no_identity = [s.dir_name for s in scans if not s.has_identity]
    unreadable = [
        "{0} ({1})".format(s.dir_name, s.identity_error) for s in scans if s.identity_error
    ]
    results.append(
        _verdict(
            name("job_identity_present"),
            not no_identity and not unreadable,
            "all {0} scanned director(ies) carry a readable {1}".format(
                len(scans), JOB_IDENTITY_FILENAME
            ),
            "{0} without {1} ({2}); {3} unreadable ({4})".format(
                len(no_identity),
                JOB_IDENTITY_FILENAME,
                _examples(no_identity),
                len(unreadable),
                _examples(unreadable),
            ),
            n_missing=len(no_identity),
            n_unreadable=len(unreadable),
        )
    )
    no_results = [s.dir_name for s in scans if not s.has_final_results]
    results.append(
        _verdict(
            name("final_results_present"),
            not no_results,
            "all {0} scanned director(ies) carry final_results.csv".format(len(scans)),
            "{0} director(ies) have no final_results.csv ({1}); those jobs carry "
            "no measurement".format(len(no_results), _examples(no_results)),
            n_missing=len(no_results),
        )
    )

    usable = [s for s in scans if s.identity is not None]

    # -- the check the old pipeline never had ------------------------------
    if not usable:
        results.append(
            _result(
                name("gene_equals_index_label"),
                SKIP,
                "no readable job identity to check",
            )
        )
    else:
        frame = pd.DataFrame(
            {
                "accession": [s.identity.accession for s in usable],
                "gene": [s.identity.gene for s in usable],
                "job": [s.dir_name for s in usable],
            }
        )
        label_report = assert_self_consistent_labels(
            frame,
            key_col="accession",
            label_col="gene",
            truth_lookup=lambda acc: [index.label(acc)],
            name="curated gene == index.label(accession)",
            warn_only=True,
        )
        results.append(
            _verdict(
                name("gene_equals_index_label"),
                bool(label_report["ok"]),
                "gene == index.label(accession) for 100% of {0} job(s)".format(len(usable)),
                "{0} of {1} job(s) carry a gene that is not "
                "index.label(accession): {2}".format(
                    label_report["n_violations"],
                    label_report["n_rows"],
                    label_report["examples"],
                ),
                **_data(label_report)
            )
        )

        collapse = assert_collapse_factor_one(
            list(zip(frame["accession"].tolist(), frame["gene"].tolist())),
            name="curated accession -> gene",
            warn_only=True,
        )
        results.append(
            _verdict(
                name("accession_gene_collapse_factor_one"),
                bool(collapse["ok"]),
                "every accession carries exactly one gene label across the tree",
                "{0} accession(s) carry more than one gene label (max {1}): "
                "{2}".format(
                    collapse["n_keys_collapsed"],
                    collapse["max_collapse"],
                    collapse["worst_keys"],
                ),
                max_collapse=collapse["max_collapse"],
                n_keys_collapsed=collapse["n_keys_collapsed"],
            )
        )

        mismatched_names = [
            "{0} != {1}".format(s.dir_name, s.identity.dir_name())
            for s in usable
            if s.identity.dir_name() != s.dir_name
        ]
        results.append(
            _verdict(
                name("dir_name_matches_identity"),
                not mismatched_names,
                "every directory name is exactly what its own identity encodes",
                "{0} director(ies) disagree with their own {1}: {2}".format(
                    len(mismatched_names), JOB_IDENTITY_FILENAME, _examples(mismatched_names)
                ),
                n_mismatched=len(mismatched_names),
            )
        )

        # -- identity vs manifest row --------------------------------------
        manifest_by_id = {
            str(row["job_id"]): row for row in manifest.to_dict("records")
        }
        disagreements: List[str] = []
        for scan in usable:
            row = manifest_by_id.get(scan.dir_name)
            if row is None:
                continue  # covered by dir_set_equals_manifest
            expected = (
                str(row["drugbank_id"]),
                str(row["accession"]),
                str(row["gene"]),
                str(row["fragment_id"]),
                int(row["cavity_index"]),
            )
            actual = (
                scan.identity.drugbank_id,
                scan.identity.accession,
                scan.identity.gene or "",
                scan.identity.fragment_id or "",
                scan.identity.cavity_index,
            )
            if expected != actual:
                disagreements.append("{0}: manifest {1} vs disk {2}".format(
                    scan.dir_name, expected, actual
                ))
            manifest_sample = str(row["sample_type"])
            disk_sample = str(scan.record.get("sample_type", ""))
            if disk_sample and manifest_sample != disk_sample:
                disagreements.append(
                    "{0}: sample_type manifest {1!r} vs disk {2!r}".format(
                        scan.dir_name, manifest_sample, disk_sample
                    )
                )
        results.append(
            _verdict(
                name("identity_matches_manifest"),
                not disagreements,
                "every scanned job's identity matches its manifest row",
                "{0} disagreement(s): {1}".format(
                    len(disagreements), _examples(disagreements, 3)
                ),
                n_disagreements=len(disagreements),
            )
        )

        # -- provenance ----------------------------------------------------
        missing_provenance = [
            s.dir_name
            for s in usable
            if s.record.get("provenance") == "salvaged_from" and not s.record.get("source_job")
        ]
        results.append(
            _verdict(
                name("salvaged_jobs_name_their_source"),
                not missing_provenance,
                "every salvaged job records the old directory it came from",
                "{0} salvaged job(s) have no source_job; the relabelling is not "
                "traceable: {1}".format(
                    len(missing_provenance), _examples(missing_provenance)
                ),
                n_missing=len(missing_provenance),
            )
        )

    # -- the 2.4% rule -----------------------------------------------------
    if not check_excluded:
        results.append(
            _result(
                name("no_excluded_artifacts"),
                SKIP,
                "--no-check-excluded: the tree was not scanned for complex_*.pdb "
                "or log_*.log",
            )
        )
    else:
        leaked = [
            "{0}/{1}".format(s.dir_name, f) for s in scans for f in s.excluded_files
        ]
        n_jobs_leaking = sum(1 for s in scans if s.excluded_files)
        results.append(
            _verdict(
                name("no_excluded_artifacts"),
                not leaked,
                "no {0} in {1} scanned job(s)".format(
                    " or ".join(EXCLUDED_BASENAME_PATTERNS), len(scans)
                ),
                "{0} excluded artifact(s) across {1} job(s) leaked into the "
                "curated tree: {2}".format(len(leaked), n_jobs_leaking, _examples(leaked)),
                n_leaked=len(leaked),
                n_jobs=n_jobs_leaking,
            )
        )

    # -- evidence integrity ------------------------------------------------
    with_manifest = [s for s in scans if isinstance(s.record.get("evidence"), list)]
    if not with_manifest:
        results.append(
            _result(
                name("evidence_complete"),
                SKIP,
                "no job records an evidence manifest, so per-file integrity was "
                "not checked",
            )
        )
    else:
        broken = [
            "{0}: {1}".format(s.dir_name, _examples(s.missing_evidence, 2))
            for s in with_manifest
            if s.missing_evidence
        ]
        results.append(
            _verdict(
                name("evidence_complete"),
                not broken,
                "every file recorded in {0} of {1} scanned job(s) is present at "
                "its recorded size".format(len(with_manifest), len(scans)),
                "{0} job(s) are missing recorded evidence: {1}".format(
                    len(broken), _examples(broken, 3)
                ),
                n_broken=len(broken),
            )
        )

    if truncated:
        results.append(
            _result(
                name("full_tree_scanned"),
                SKIP,
                "--limit {0}: only the first {0} of {1} director(ies) were "
                "scanned".format(limit, len(dirs)),
                n_scanned=len(scanned_dirs),
                n_total=len(dirs),
            )
        )
    return results


# --------------------------------------------------------------------------
# audit run  (cross-stage consistency)
# --------------------------------------------------------------------------


def _read_manifest(run_dir: str, filename: str, schema: str):
    path = os.path.join(run_dir, "manifests", filename)
    if not os.path.exists(path):
        return None, path, "not present"
    try:
        return read_checked(path, schema), path, ""
    except (SchemaViolation, InvariantViolation, OSError) as exc:
        return None, path, str(exc)


def _keys(df, columns: Sequence[str]):
    return set(
        zip(*[df[c].astype(str) if c != "cavity_index" else df[c].astype("int64").astype(str)
              for c in columns])
    )


def audit_run(
    run_dir: str,
    limit: Optional[int] = None,
    skip_determinism: bool = False,
    check_excluded: bool = True,
    expect: str = "subset",
) -> List[CheckResult]:
    """Cross-stage consistency for a whole ``$NEW`` run directory."""
    results: List[CheckResult] = []
    index_path = os.path.join(run_dir, "identity", "identity_index.parquet")

    results.extend(
        audit_identity(index_path, skip_determinism=skip_determinism, prefix="identity.")
    )
    try:
        index = IdentityIndex.load(index_path, validate=False)
    except Exception:  # noqa: BLE001 - already reported above
        index = None

    frames: Dict[str, Any] = {}
    for filename, schema in MANIFESTS:
        frame, path, problem = _read_manifest(run_dir, filename, schema)
        if frame is None:
            status = SKIP if problem == "not present" else FAIL
            results.append(
                _result(
                    "manifest.{0}".format(schema),
                    status,
                    "{0}: {1}".format(path, problem),
                    path=path,
                )
            )
            continue
        frames[schema] = frame
        results.append(
            _result(
                "manifest.{0}".format(schema),
                PASS,
                "{0} row(s) satisfy the contract".format(len(frame)),
                path=path,
                n_rows=int(len(frame)),
            )
        )

    # -- labels in every manifest must be index.label(accession) -----------
    if index is None:
        results.append(
            _result(
                "xstage.gene_labels",
                SKIP,
                "no identity index; manifest gene labels were not checked",
            )
        )
    else:
        offenders: List[str] = []
        n_checked = 0
        for schema, frame in sorted(frames.items()):
            if "accession" not in frame.columns or "gene" not in frame.columns:
                continue
            subset = frame[frame["accession"].notna() & frame["gene"].notna()]
            if not len(subset):
                continue
            n_checked += len(subset)
            report = assert_self_consistent_labels(
                subset,
                key_col="accession",
                label_col="gene",
                truth_lookup=lambda acc: [index.label(acc)],
                name="{0} gene == index.label(accession)".format(schema),
                warn_only=True,
            )
            if not report["ok"]:
                offenders.append(
                    "{0}: {1}/{2} bad, e.g. {3}".format(
                        schema, report["n_violations"], report["n_rows"], report["examples"][:2]
                    )
                )
        results.append(
            _verdict(
                "xstage.gene_labels",
                not offenders,
                "every (accession, gene) pair in {0} manifest row(s) satisfies "
                "gene == index.label(accession)".format(n_checked),
                "; ".join(offenders),
                n_checked=n_checked,
            )
        )

    # -- positives and negatives may not overlap ---------------------------
    if "positives_planned" in frames and "negatives_drawn" in frames:
        key = ("drugbank_id", "accession", "fragment_id", "cavity_index")
        overlap = sorted(
            _keys(frames["positives_planned"], key) & _keys(frames["negatives_drawn"], key)
        )
        results.append(
            _verdict(
                "xstage.positive_negative_disjoint",
                not overlap,
                "no pocket is both a planned positive and a drawn negative",
                "{0} pocket(s) are in both classes: {1}".format(
                    len(overlap), _examples(overlap)
                ),
                n_overlap=len(overlap),
            )
        )
    else:
        results.append(
            _result(
                "xstage.positive_negative_disjoint",
                SKIP,
                "needs positives_planned.csv and negatives_drawn.csv",
            )
        )

    # -- contamination: a drawn negative may not be a known interaction ----
    if "interactions_resolved" in frames and "negatives_drawn" in frames:
        known = _keys(frames["interactions_resolved"], ("drugbank_id", "accession"))
        drawn = _keys(frames["negatives_drawn"], ("drugbank_id", "accession"))
        contaminated = sorted(known & drawn)
        results.append(
            _verdict(
                "xstage.negatives_not_known_interactions",
                not contaminated,
                "0 drawn negatives are known interactions (the old campaign had 5)",
                "{0} drawn negative pair(s) ARE known interactions: {1}".format(
                    len(contaminated), _examples(contaminated)
                ),
                n_contaminated=len(contaminated),
            )
        )
    else:
        results.append(
            _result(
                "xstage.negatives_not_known_interactions",
                SKIP,
                "needs interactions_resolved.csv and negatives_drawn.csv",
            )
        )

    # -- a pool draw must name a pool job ----------------------------------
    if "negatives_drawn" in frames and "salvage_map" in frames:
        drawn = frames["negatives_drawn"]
        pool_rows = drawn[drawn["source"].astype(str) == "pool"]
        salvage = frames["salvage_map"]
        by_dir = {
            str(r["old_job_dir"]): str(r["classification"])
            for r in salvage.to_dict("records")
        }
        problems = []
        for value in pool_rows["old_job_dir"].astype(str).tolist():
            classification = by_dir.get(value)
            if classification is None:
                problems.append("{0} (absent from salvage_map)".format(value))
            elif classification != "valid_non_interaction":
                problems.append("{0} (classified {1})".format(value, classification))
        results.append(
            _verdict(
                "xstage.pool_draws_reference_pool_jobs",
                not problems,
                "all {0} pool draw(s) name a salvage-map job classified "
                "valid_non_interaction".format(len(pool_rows)),
                "{0} problem(s): {1}".format(len(problems), _examples(problems)),
                n_problems=len(problems),
            )
        )
    else:
        results.append(
            _result(
                "xstage.pool_draws_reference_pool_jobs",
                SKIP,
                "needs negatives_drawn.csv and salvage_map.csv",
            )
        )

    # -- run_manifest job_id must be derivable, not free text --------------
    if "run_manifest" in frames:
        run_manifest = frames["run_manifest"]
        bad = []
        for row in run_manifest.to_dict("records"):
            try:
                expected = format_job_dir(
                    row["drugbank_id"],
                    row["accession"],
                    row["gene"],
                    row["fragment_id"],
                    row["cavity_index"],
                )
            except JobIdError as exc:
                bad.append("{0}: {1}".format(row["job_id"], exc))
                continue
            if expected != str(row["job_id"]):
                bad.append("{0} != {1}".format(row["job_id"], expected))
        results.append(
            _verdict(
                "xstage.run_manifest_job_id_derivable",
                not bad,
                "every job_id is exactly format_job_dir() of its own columns",
                "{0} job_id(s) are not derivable from their own row: {1}".format(
                    len(bad), _examples(bad)
                ),
                n_bad=len(bad),
            )
        )
    else:
        results.append(
            _result("xstage.run_manifest_job_id_derivable", SKIP, "needs run_manifest.csv")
        )

    # -- run_manifest must cover exactly the design ------------------------
    if (
        "run_manifest" in frames
        and "positives_planned" in frames
        and "negatives_drawn" in frames
    ):
        key = ("drugbank_id", "accession", "fragment_id", "cavity_index")
        wanted = _keys(frames["positives_planned"], key) | _keys(
            frames["negatives_drawn"], key
        )
        have = _keys(frames["run_manifest"], key)
        missing = sorted(wanted - have)
        extra = sorted(have - wanted)
        results.append(
            _verdict(
                "xstage.run_manifest_covers_design",
                not missing and not extra,
                "run_manifest is exactly positives_planned + negatives_drawn "
                "({0} pocket(s))".format(len(wanted)),
                "{0} designed job(s) missing from run_manifest ({1}); {2} extra "
                "({3})".format(
                    len(missing), _examples(missing), len(extra), _examples(extra)
                ),
                n_missing=len(missing),
                n_extra=len(extra),
            )
        )
    else:
        results.append(
            _result(
                "xstage.run_manifest_covers_design",
                SKIP,
                "needs run_manifest.csv, positives_planned.csv and negatives_drawn.csv",
            )
        )

    # -- the curated tree --------------------------------------------------
    curated_dir = os.path.join(run_dir, "docking_results")
    manifest_path = os.path.join(run_dir, "manifests", "run_manifest.csv")
    if os.path.isdir(curated_dir) and os.path.exists(manifest_path):
        results.extend(
            audit_curated(
                curated_dir,
                manifest_path,
                index_path,
                limit=limit,
                check_excluded=check_excluded,
                expect=expect,
                prefix="curated.",
            )
        )
    else:
        results.append(
            _result(
                "curated.present",
                SKIP,
                "{0} or {1} does not exist yet".format(curated_dir, manifest_path),
            )
        )
    return results


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def format_table(results: Sequence[CheckResult], width: int = 96) -> str:
    """The check -> PASS/FAIL -> detail table, plus full text for failures."""
    if not results:
        return "no checks ran\n"
    name_width = min(max(len(r.name) for r in results), 44)
    lines = ["", "{0:<6} {1:<{2}}  {3}".format("STATUS", "CHECK", name_width, "DETAIL")]
    lines.append("-" * (8 + name_width + 2 + 40))
    for result in results:
        detail = " ".join(str(result.detail).split())
        room = max(width - name_width - 9, 20)
        if len(detail) > room:
            detail = detail[: room - 1] + "…"
        lines.append(
            "{0:<6} {1:<{2}}  {3}".format(result.status, result.name[:name_width], name_width, detail)
        )
    failures = [r for r in results if r.status == FAIL]
    skips = [r for r in results if r.status == SKIP]
    if failures:
        lines.append("")
        lines.append("FAILURES")
        lines.append("-" * 40)
        for result in failures:
            lines.append("  {0}".format(result.name))
            for chunk in _wrap(" ".join(str(result.detail).split()), width - 6):
                lines.append("      {0}".format(chunk))
    lines.append("")
    lines.append(
        "{0} check(s): {1} passed, {2} failed, {3} skipped".format(
            len(results), len(results) - len(failures) - len(skips), len(failures), len(skips)
        )
    )
    if skips and not failures:
        lines.append(
            "A check that did not run is not a check that passed — this gate is "
            "NOT cleared. Pass --allow-skips to exit 0 anyway."
        )
    lines.append("")
    return "\n".join(lines)


def _wrap(text: str, width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = word if not current else current + " " + word
    if current:
        lines.append(current)
    return lines or [""]


def exit_code(results: Sequence[CheckResult], allow_skips: bool = False) -> int:
    if any(r.status == FAIL for r in results):
        return 1
    if not allow_skips and any(r.status == SKIP for r in results):
        return 2
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser, suppress: bool) -> None:
    """Options accepted before *and* after the subcommand.

    The subcommand copies default to ``argparse.SUPPRESS`` so that omitting them
    leaves the top-level parser's value in place instead of overwriting it with
    a fresh default -- the standard argparse subparser trap.
    """
    common = {"default": argparse.SUPPRESS} if suppress else {}
    parser.add_argument(
        "--json",
        dest="json_out",
        help="write the results here as JSON",
        **(common if suppress else {"default": None})
    )
    parser.add_argument(
        "--allow-skips",
        action="store_true",
        help="exit 0 even if some checks could not run",
        **common
    )
    parser.add_argument(
        "--quiet", action="store_true", help="warnings and errors only", **common
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit.py",
        description="Read-only invariant runner for the dock_selectivity rebuild.",
        epilog=(
            "Exit codes: 0 all checks ran and passed; 1 a check FAILED; "
            "2 nothing failed but a check could not run."
        ),
    )
    _add_common(parser, suppress=False)
    sub = parser.add_subparsers(dest="command", required=True)

    identity = sub.add_parser("identity", help="G1-G7 + spot checks on the identity index")
    identity.add_argument("--index", required=True)
    identity.add_argument("--expect-sha256", default=None)
    identity.add_argument("--meta", default=None, help="identity_index.meta.json")
    identity.add_argument("--unresolved", default=None, help="unresolved_targets.csv")
    identity.add_argument(
        "--skip-determinism",
        action="store_true",
        help="skip the PYTHONHASHSEED subprocess check (reported as SKIP)",
    )

    curated = sub.add_parser("curated", help="checks on a materialised docking tree")
    curated.add_argument("--dir", required=True, dest="curated_dir")
    curated.add_argument("--manifest", required=True)
    curated.add_argument(
        "--index",
        required=True,
        help="identity_index.parquet; gene == index.label(accession) is checked against it",
    )
    curated.add_argument("--limit", type=int, default=None, help="scan only the first N dirs")
    curated.add_argument(
        "--no-check-excluded",
        action="store_true",
        help="do not walk each job looking for complex_*.pdb / log_*.log",
    )
    curated.add_argument(
        "--expect",
        choices=("subset", "exact"),
        default="subset",
        help=(
            "subset (default): every salvaged manifest row must be on disk and "
            "no directory may be missing from the manifest. exact: the "
            "newly-docked rows must be there too (the Phase 7/8 end state)."
        ),
    )

    run = sub.add_parser("run", help="cross-stage consistency for a whole run directory")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--skip-determinism", action="store_true")
    run.add_argument("--no-check-excluded", action="store_true")
    run.add_argument("--expect", choices=("subset", "exact"), default="subset")

    for subparser in (identity, curated, run):
        _add_common(subparser, suppress=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "identity":
        results = audit_identity(
            args.index,
            expect_sha256=args.expect_sha256,
            meta_path=args.meta,
            unresolved_path=args.unresolved,
            skip_determinism=args.skip_determinism,
        )
    elif args.command == "curated":
        results = audit_curated(
            args.curated_dir,
            args.manifest,
            args.index,
            limit=args.limit,
            check_excluded=not args.no_check_excluded,
            expect=args.expect,
        )
    else:
        results = audit_run(
            args.run_dir,
            limit=args.limit,
            skip_determinism=args.skip_determinism,
            check_excluded=not args.no_check_excluded,
            expect=args.expect,
        )

    sys.stdout.write(format_table(results))
    if args.json_out:
        write_invariant_report([r.to_dict() for r in results], args.json_out)
    return exit_code(results, allow_skips=args.allow_skips)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
