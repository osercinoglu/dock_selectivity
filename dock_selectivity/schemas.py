"""Declarative column contracts for every inter-stage manifest.

Invariant **I9**: *every* CSV that crosses a stage boundary declares its
columns, dtypes, nullability, allowed values and primary key, and is validated
on **write** and again on **read**. The 2025 pipeline had no such contract, and
paid for it:

* ``generate_negative_samples.py:375`` read ``row.get('Gene_Name', '')`` from
  ``cavity_mapping.csv``, which has no ``Gene_Name`` column — 13,452/13,452
  negatives got a blank gene and produced 10,698 directories named ``*_nan_*``.
  Here ``gene`` is declared non-nullable, so that write fails immediately.
* ``batch_dock.py:448`` patched a missing column with ``df['Gene_Name'] = ''``
  instead of failing. A schema makes the absence an error, not a default.
* Fragment ids were written ``'F1'`` on positives and bare ``'1'`` on
  negatives, so a later join matched neither. One ``fragment_id`` column,
  declared once, is normalised by its producer and checked by every reader.
* ``pandas`` reads the literal gene symbol ``NA`` (and the literal string
  ``nan`` the old folder names contain) as *missing* by default.
  :func:`read_checked` disables that: only an empty field is null.

The schemas below cover the manifests of ``REBUILD_PLAN.md`` §2::

    manifests/interactions_resolved.csv   corrected DTI edge list
    manifests/positives_planned.csv       positives the design wants
    manifests/negatives_pool.csv          reuseable non-interaction jobs
    manifests/negatives_drawn.csv         what the balanced sampler selected
    manifests/job_classification.csv             old job dir -> new job dir, or a reason
    manifests/redock_queue.csv            jobs that must be docked fresh
    manifests/run_manifest.csv            every job that SHOULD exist

Usage::

    from dock_selectivity.schemas import read_checked, write_checked, validate

    write_checked(df, new / "manifests/run_manifest.csv", "run_manifest")
    manifest = read_checked(new / "manifests/run_manifest.csv", "run_manifest")

Two conventions run through all of them:

``accession`` is the only protein key. ``gene`` is a *label*, written by
``IdentityIndex.label(accession)``, and never a join key.

``selected_by`` is invariant **I7** (selection provenance): a non-null,
human-readable statement of why the row exists, e.g.
``"entrez=554 via edge (DB00035,554)"`` or ``"pool job DB00035_AVPR2_Q8NCT1_cavity_2"``.
Had ``required_structures.csv`` carried it, ``AVPR2 -> Q8NCT1`` would have been
legible in the plan file itself.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from .invariants import InvariantViolation, assert_unique_key

__all__ = [
    "SchemaViolation",
    "Column",
    "RowCheck",
    "Schema",
    "SCHEMAS",
    "SAMPLE_TYPES",
    "CLASSIFICATIONS",
    "PROVENANCES",
    "NEGATIVE_SOURCES",
    "schema_names",
    "get_schema",
    "validate",
    "coerce",
    "empty_frame",
    "write_checked",
    "read_checked",
    "validate_file",
]

logger = logging.getLogger(__name__)


class SchemaViolation(InvariantViolation):
    """A frame or file does not satisfy its declared contract."""


#: The only dtypes a manifest column may declare. Deliberately small: each one
#: round-trips through CSV exactly (see :func:`read_checked`).
ALLOWED_DTYPES = ("string", "int64", "Int64", "float64", "boolean")

#: Vocabulary shared across manifests.
SAMPLE_TYPES = ("positive", "negative")
CLASSIFICATIONS = (
    "true_positive",  # drug really does bind the accession that was docked
    "valid_non_interaction",  # real measurement, no annotated interaction
    "unparseable",  # directory name yielded no identity
    "incomplete",  # no final_results.csv; not reuseable
)
PROVENANCES = ("reused_from", "newly_docked")
NEGATIVE_SOURCES = ("pool", "new")


@dataclass(frozen=True)
class Column:
    """One declared column of a manifest."""

    name: str
    dtype: str
    nullable: bool = False
    allowed: Optional[Tuple[str, ...]] = None
    description: str = ""


@dataclass(frozen=True)
class RowCheck:
    """A cross-column rule.

    ``predicate(df)`` returns a boolean Series that must be ``True`` for every
    row. Used for rules a per-column contract cannot express, such as "a
    reuse row either names its new job directory or gives a reason for
    exclusion — never neither".
    """

    name: str
    description: str
    predicate: Callable[[pd.DataFrame], Any]


@dataclass(frozen=True)
class Schema:
    """Column contract for one inter-stage artifact."""

    name: str
    description: str
    columns: Tuple[Column, ...]
    primary_key: Tuple[str, ...] = ()
    checks: Tuple[RowCheck, ...] = ()

    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    @property
    def dtypes(self) -> Dict[str, str]:
        return {c.name: c.dtype for c in self.columns}

    def column(self, name: str) -> Column:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(
            "{0} has no column {1!r}; it has {2}".format(
                self.name, name, self.column_names
            )
        )

    def describe(self) -> str:
        lines = ["{0} — {1}".format(self.name, self.description)]
        if self.primary_key:
            lines.append("  primary key: {0}".format(list(self.primary_key)))
        for col in self.columns:
            lines.append(
                "  {0:<20} {1:<8} {2}{3}{4}".format(
                    col.name,
                    col.dtype,
                    "NULL-able " if col.nullable else "required  ",
                    ("in {0} ".format(list(col.allowed)) if col.allowed else ""),
                    col.description,
                )
            )
        for check in self.checks:
            lines.append("  check {0}: {1}".format(check.name, check.description))
        return "\n".join(lines)


# --------------------------------------------------------------------------
# shared column definitions — identical names and dtypes across all manifests
# --------------------------------------------------------------------------


def _nullable(col: Column, description: Optional[str] = None) -> Column:
    if description is None:
        return replace(col, nullable=True)
    return replace(col, nullable=True, description=description)


DRUGBANK_ID = Column(
    "drugbank_id",
    "string",
    description="DrugBank accession, e.g. DB00035. Never a drug name slug.",
)
ENTREZ_ID = Column(
    "entrez_id",
    "int64",
    description="NCBI Entrez Gene ID — the clean 1:1 key resolution runs on.",
)
ACCESSION = Column(
    "accession",
    "string",
    description="UniProt accession, e.g. P30518. The only protein key.",
)
GENE = Column(
    "gene",
    "string",
    description=(
        "Display label only, written by IdentityIndex.label(accession). "
        "Never a join key, never an input to selection."
    ),
)
FRAGMENT_ID = Column(
    "fragment_id",
    "string",
    description="AlphaFold fragment, normalised 'F<k>' — never bare '1'.",
)
CAVITY_INDEX = Column(
    "cavity_index", "int64", description="CavitySpace cavity index within the fragment."
)
SAMPLE_TYPE = Column(
    "sample_type",
    "string",
    allowed=SAMPLE_TYPES,
    description="Design role of the job under corrected identity.",
)
SELECTED_BY = Column(
    "selected_by",
    "string",
    description="I7 selection provenance: why this row exists. Never null.",
)
JOB_ID = Column(
    "job_id",
    "string",
    description="New-scheme job directory name, e.g. DB00035__P30518__AVPR2__F1__cav2.",
)
OLD_JOB_DIR = Column(
    "old_job_dir",
    "string",
    description="Directory name in $OLD/consensus_docking_results (read-only).",
)
OLD_GENE_LABEL = Column(
    "old_gene_label",
    "string",
    nullable=True,
    description=(
        "Gene token as it appears in the old directory name. Corrupt by "
        "construction; kept as evidence. May be the literal string 'nan'."
    ),
)
CLASSIFICATION = Column(
    "classification",
    "string",
    allowed=CLASSIFICATIONS,
    description="What the old job really is under corrected identity.",
)


# --------------------------------------------------------------------------
# row checks
# --------------------------------------------------------------------------


def _reuse_outcome_recorded(df: pd.DataFrame) -> Any:
    """Exactly one of ``new_job_dir`` / ``excluded_reason`` must be set."""
    return df["new_job_dir"].notna() ^ df["excluded_reason"].notna()


def _pool_rows_name_their_source(df: pd.DataFrame) -> Any:
    """``source == 'pool'`` iff ``old_job_dir`` is set."""
    return (df["source"] == "pool") == df["old_job_dir"].notna()


def _reused_rows_name_their_source(df: pd.DataFrame) -> Any:
    """``provenance == 'reused_from'`` iff ``source_job`` is set."""
    return (df["provenance"] == "reused_from") == df["source_job"].notna()


# --------------------------------------------------------------------------
# the schemas
# --------------------------------------------------------------------------

_SCHEMA_LIST: Tuple[Schema, ...] = (
    Schema(
        name="interactions_resolved",
        description=(
            "Corrected drug-target edge list. One row per (drugbank_id, "
            "accession) resolved from 1_drug_to_protein.tsv on node_2 (Entrez), "
            "never on node_2_name. Edges that did not resolve belong in "
            "identity/unresolved_targets.csv, never dropped."
        ),
        columns=(DRUGBANK_ID, ENTREZ_ID, ACCESSION, GENE, SELECTED_BY),
        primary_key=("drugbank_id", "accession"),
    ),
    Schema(
        name="positives_planned",
        description=(
            "Positive (drug, protein, fragment, cavity) jobs the corrected "
            "design wants, whether or not they already exist on disk."
        ),
        columns=(
            DRUGBANK_ID,
            ACCESSION,
            GENE,
            FRAGMENT_ID,
            CAVITY_INDEX,
            replace(SAMPLE_TYPE, allowed=("positive",)),
            SELECTED_BY,
        ),
        primary_key=("drugbank_id", "accession", "fragment_id", "cavity_index"),
    ),
    Schema(
        name="negatives_pool",
        description=(
            "Already-docked jobs that are valid non-interactions under "
            "corrected identity (expected 16,868 = 5,072 mislabelled positives "
            "+ 11,796 clean negatives). Candidates only — the balanced sampler "
            "decides which are drawn."
        ),
        columns=(
            OLD_JOB_DIR,
            DRUGBANK_ID,
            ACCESSION,
            GENE,
            FRAGMENT_ID,
            CAVITY_INDEX,
            Column(
                "old_sample_type",
                "string",
                allowed=SAMPLE_TYPES,
                description="Label the old campaign gave this job.",
            ),
            OLD_GENE_LABEL,
            replace(CLASSIFICATION, allowed=("valid_non_interaction",)),
            Column(
                "has_final_results",
                "boolean",
                description="final_results.csv exists and is non-empty.",
            ),
        ),
        primary_key=("old_job_dir",),
    ),
    Schema(
        name="negatives_drawn",
        description=(
            "Negatives the balanced sampler actually selected, in accession "
            "space. source='pool' rows are reused from an existing job; "
            "source='new' rows become redock_queue entries."
        ),
        columns=(
            DRUGBANK_ID,
            ACCESSION,
            GENE,
            FRAGMENT_ID,
            CAVITY_INDEX,
            replace(SAMPLE_TYPE, allowed=("negative",)),
            Column(
                "source",
                "string",
                allowed=NEGATIVE_SOURCES,
                description="'pool' = reused from an old job, 'new' = must be docked.",
            ),
            _nullable(
                OLD_JOB_DIR,
                "Old directory this negative is reused from; null when source='new'.",
            ),
            SELECTED_BY,
        ),
        primary_key=("drugbank_id", "accession", "fragment_id", "cavity_index"),
        checks=(
            RowCheck(
                "pool_rows_name_their_source",
                "source == 'pool' if and only if old_job_dir is set",
                _pool_rows_name_their_source,
            ),
        ),
    ),
    Schema(
        name="job_classification",
        description=(
            "One row per directory in $OLD/consensus_docking_results (expected "
            "24,173). Every row either names the new directory it becomes or "
            "gives the reason it is excluded — never neither, so a job can "
            "never disappear unremarked."
        ),
        columns=(
            OLD_JOB_DIR,
            CLASSIFICATION,
            _nullable(DRUGBANK_ID, "Null only when classification == 'unparseable'."),
            _nullable(ACCESSION, "Null only when classification == 'unparseable'."),
            _nullable(GENE, "Label from IdentityIndex; null when unparseable."),
            _nullable(FRAGMENT_ID, "Null when unparseable."),
            Column(
                "cavity_index",
                "Int64",
                nullable=True,
                description="Nullable integer: unparseable rows have no cavity.",
            ),
            OLD_GENE_LABEL,
            Column(
                "label_was_correct",
                "boolean",
                nullable=True,
                description=(
                    "Did the old folder's gene token really name the docked "
                    "accession? False on the 45.8% that did not."
                ),
            ),
            Column(
                "new_job_dir",
                "string",
                nullable=True,
                description="New-scheme directory this job becomes, if carried over.",
            ),
            Column(
                "excluded_reason",
                "string",
                nullable=True,
                description="Why this job is NOT carried over. Required if not carried.",
            ),
        ),
        primary_key=("old_job_dir",),
        checks=(
            RowCheck(
                "outcome_recorded",
                "exactly one of new_job_dir / excluded_reason is non-null",
                _reuse_outcome_recorded,
            ),
        ),
    ),
    Schema(
        name="redock_queue",
        description=(
            "Jobs that must be docked fresh because the reuse pool cannot "
            "supply them. Also the required-set argument for every structure "
            "preparation step (REBUILD_PLAN Phase 6)."
        ),
        columns=(
            JOB_ID,
            DRUGBANK_ID,
            ACCESSION,
            GENE,
            FRAGMENT_ID,
            CAVITY_INDEX,
            SAMPLE_TYPE,
            SELECTED_BY,
        ),
        primary_key=("job_id",),
    ),
    Schema(
        name="run_manifest",
        description=(
            "The authority on which jobs SHOULD exist. Discovery in "
            "parse_prepare.py is checked against this, so a truncated pass "
            "(the 10,000-file cap that silently dropped 58% of the campaign) "
            "is detectable instead of silent."
        ),
        columns=(
            JOB_ID,
            DRUGBANK_ID,
            ACCESSION,
            GENE,
            FRAGMENT_ID,
            CAVITY_INDEX,
            SAMPLE_TYPE,
            Column(
                "provenance",
                "string",
                allowed=PROVENANCES,
                description="Mirrors job_identity.json: reused_from | newly_docked.",
            ),
            Column(
                "source_job",
                "string",
                nullable=True,
                description="Old directory a reused job came from; null if newly docked.",
            ),
            SELECTED_BY,
        ),
        primary_key=("job_id",),
        checks=(
            RowCheck(
                "reused_rows_name_their_source",
                "provenance == 'reused_from' if and only if source_job is set",
                _reused_rows_name_their_source,
            ),
        ),
    ),
)

SCHEMAS: Dict[str, Schema] = {s.name: s for s in _SCHEMA_LIST}


def _self_check(schemas: Tuple[Schema, ...] = _SCHEMA_LIST) -> None:
    """Validate the schema declarations themselves, at import time.

    Takes the list as an argument so a test can prove a malformed declaration
    (nullable primary key, dtype outside the vocabulary, nullable numpy int)
    is rejected rather than merely trusting that the shipped ones are fine.
    """
    for schema in schemas:
        names = schema.column_names
        if len(names) != len(set(names)):
            raise SchemaViolation(
                "schema {0} declares a duplicate column".format(schema.name),
                name="schemas._self_check",
            )
        for col in schema.columns:
            if col.dtype not in ALLOWED_DTYPES:
                raise SchemaViolation(
                    "schema {0} column {1} declares dtype {2!r}; allowed: "
                    "{3}".format(schema.name, col.name, col.dtype, ALLOWED_DTYPES),
                    name="schemas._self_check",
                )
            if col.dtype == "int64" and col.nullable:
                raise SchemaViolation(
                    "schema {0} column {1} is nullable but declares numpy "
                    "'int64', which cannot hold nulls; use 'Int64'".format(
                        schema.name, col.name
                    ),
                    name="schemas._self_check",
                )
            if col.allowed is not None and col.dtype != "string":
                raise SchemaViolation(
                    "schema {0} column {1}: allowed= is only supported for "
                    "string columns".format(schema.name, col.name),
                    name="schemas._self_check",
                )
        for key_col in schema.primary_key:
            col = schema.column(key_col)  # raises if absent
            if col.nullable:
                raise SchemaViolation(
                    "schema {0} primary key column {1} may not be "
                    "nullable".format(schema.name, key_col),
                    name="schemas._self_check",
                )


_self_check()


def schema_names() -> List[str]:
    """Names of every declared schema, sorted."""
    return sorted(SCHEMAS)


def get_schema(schema: Union[str, Schema]) -> Schema:
    """Look up a schema by name.

    An unknown name is an error listing the known ones — never a permissive
    "no schema, skip validation" fallback (that class of fallback is what
    widened the universe in ``extract_alphafold_models.py``).
    """
    if isinstance(schema, Schema):
        return schema
    try:
        return SCHEMAS[schema]
    except KeyError:
        raise SchemaViolation(
            "unknown schema {0!r}; known schemas: {1}".format(schema, schema_names()),
            name="get_schema",
        ) from None


def empty_frame(schema: Union[str, Schema]) -> pd.DataFrame:
    """An empty frame with exactly this schema's columns and dtypes.

    Useful as an accumulator seed so a stage that produces nothing still writes
    a valid, well-typed manifest instead of a zero-byte file.
    """
    sch = get_schema(schema)
    return pd.DataFrame(
        {c.name: pd.Series([], dtype=c.dtype) for c in sch.columns},
        columns=sch.column_names,
    )


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def _check_columns(
    df: pd.DataFrame,
    sch: Schema,
    allow_extra: bool,
    label: str,
) -> Tuple[List[str], List[str]]:
    """Missing/extra column gate shared by validate, coerce and the I/O pair."""
    declared = sch.column_names
    present = list(df.columns)
    missing = [c for c in declared if c not in present]
    extra = [c for c in present if c not in declared]
    if missing:
        raise SchemaViolation(
            "missing required column(s) {0}. Schema {1} declares {2}; the frame "
            "has {3}. A missing column is an error, never a default value "
            "(batch_dock.py:448 filled one in with '' and produced 10,698 "
            "'*_nan_*' job directories).".format(missing, sch.name, declared, present),
            name=label,
        )
    if extra and not allow_extra:
        raise SchemaViolation(
            "undeclared column(s) {0} for schema {1}. Declare them in "
            "schemas.py, or drop them explicitly rather than letting a writer "
            "discard data; pass allow_extra=True only for a deliberate "
            "superset.".format(extra, sch.name),
            name=label,
        )
    return declared, extra


def validate(
    df: pd.DataFrame,
    schema: Union[str, Schema],
    allow_extra: bool = False,
    check_key: bool = True,
    check_dtypes: bool = True,
    where: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate ``df`` against a declared schema; raise :class:`SchemaViolation`.

    Checks, in order: missing columns, extra columns, dtypes, nullability,
    allowed values, primary-key uniqueness, cross-column row checks.

    Returns a JSON-serialisable report so a stage can record that validation
    ran and what it saw.
    """
    sch = get_schema(schema)
    label = "{0}{1}".format(sch.name, " @ {0}".format(where) if where else "")
    if not isinstance(df, pd.DataFrame):
        raise SchemaViolation(
            "expected a DataFrame for schema {0}, got {1}".format(
                sch.name, type(df).__name__
            ),
            name=label,
        )

    declared, extra = _check_columns(df, sch, allow_extra, label)

    problems: List[str] = []
    for col in sch.columns:
        series = df[col.name]
        actual = str(series.dtype)
        if check_dtypes and actual != col.dtype:
            problems.append(
                "{0}: declared {1}, found {2}".format(col.name, col.dtype, actual)
            )
            continue
        n_null = int(series.isna().sum())
        if n_null and not col.nullable:
            examples = df.index[series.isna()].tolist()[:5]
            problems.append(
                "{0}: {1} null value(s) in a non-nullable column (rows {2})".format(
                    col.name, n_null, examples
                )
            )
        if col.allowed is not None:
            bad = series.dropna()
            bad = bad[~bad.isin(list(col.allowed))]
            if len(bad):
                problems.append(
                    "{0}: {1} value(s) outside {2}, e.g. {3}".format(
                        col.name,
                        len(bad),
                        list(col.allowed),
                        sorted({str(v) for v in bad.tolist()})[:5],
                    )
                )
    if problems:
        raise SchemaViolation(
            "schema {0} violated by {1} column(s): {2}".format(
                sch.name, len(problems), "; ".join(problems)
            ),
            name=label,
        )

    key_report = None
    if check_key and sch.primary_key:
        try:
            key_report = assert_unique_key(df, list(sch.primary_key), name=label)
        except InvariantViolation as exc:
            raise SchemaViolation(
                "primary key {0} of schema {1} is not unique: {2}".format(
                    list(sch.primary_key), sch.name, exc.message
                ),
                name=label,
                report=exc.report,
            ) from None

    check_results = []
    for check in sch.checks:
        mask = check.predicate(df)
        mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
        n_bad = int((~mask).sum())
        check_results.append({"name": check.name, "n_violations": n_bad})
        if n_bad:
            offenders = df.loc[~mask, list(sch.primary_key) or declared[:1]]
            raise SchemaViolation(
                "row check {0!r} failed on {1} of {2} rows ({3}). Offending "
                "keys: {4}".format(
                    check.name,
                    n_bad,
                    len(df),
                    check.description,
                    offenders.head(5).to_dict("records"),
                ),
                name=label,
            )

    return {
        "check": "schema",
        "name": label,
        "schema": sch.name,
        "n_rows": int(len(df)),
        "columns": declared,
        "extra_columns": extra,
        "primary_key": list(sch.primary_key),
        "key_report": key_report,
        "row_checks": check_results,
        "ok": True,
    }


def _coerce_series(series: pd.Series, col: Column, schema_name: str) -> pd.Series:
    target = col.dtype
    actual = str(series.dtype)
    if actual == target:
        return series
    try:
        if target in ("int64", "Int64"):
            if str(series.dtype).startswith("float"):
                fractional = series.dropna()
                fractional = fractional[fractional % 1 != 0]
                if len(fractional):
                    raise SchemaViolation(
                        "column {0} of schema {1} declares {2} but holds "
                        "non-integral value(s) {3}; casting would silently "
                        "truncate".format(
                            col.name,
                            schema_name,
                            target,
                            fractional.head(3).tolist(),
                        ),
                        name="coerce",
                    )
            if target == "int64" and int(series.isna().sum()):
                raise SchemaViolation(
                    "column {0} of schema {1} declares non-nullable int64 but "
                    "holds {2} null(s)".format(
                        col.name, schema_name, int(series.isna().sum())
                    ),
                    name="coerce",
                )
            if series.dtype == object or str(series.dtype) == "string":
                return series.astype("Int64").astype(target)
            return series.astype(target)
        if target == "boolean":
            if series.dtype == object or str(series.dtype) == "string":
                mapping = {
                    "True": True,
                    "true": True,
                    "TRUE": True,
                    "1": True,
                    "False": False,
                    "false": False,
                    "FALSE": False,
                    "0": False,
                }
                mapped = series.map(lambda v: mapping.get(v, v) if isinstance(v, str) else v)
                unknown = [
                    v for v in mapped.dropna().tolist() if not isinstance(v, bool)
                ]
                if unknown:
                    raise SchemaViolation(
                        "column {0} of schema {1} declares boolean but holds "
                        "{2!r}".format(col.name, schema_name, sorted(set(map(str, unknown)))[:5]),
                        name="coerce",
                    )
                return mapped.astype("boolean")
            return series.astype("boolean")
        return series.astype(target)
    except SchemaViolation:
        raise
    except (TypeError, ValueError) as exc:
        raise SchemaViolation(
            "column {0} of schema {1} cannot be cast from {2} to {3}: "
            "{4}".format(col.name, schema_name, actual, target, exc),
            name="coerce",
        ) from None


def coerce(
    df: pd.DataFrame,
    schema: Union[str, Schema],
    allow_extra: bool = False,
) -> pd.DataFrame:
    """Return ``df`` with the schema's columns, in schema order, at declared dtypes.

    Casts are **lossless or they raise**: a float column holding 1.5 will not
    silently become 1, an ``int64`` column will not silently accept nulls, and
    an unparseable value raises with the column named. Missing columns are an
    error, and so are undeclared ones unless ``allow_extra=True`` — in which
    case they are passed through untouched, after the declared ones. Coercion
    never invents or discards a column.
    """
    sch = get_schema(schema)
    if not isinstance(df, pd.DataFrame):
        raise SchemaViolation(
            "expected a DataFrame for schema {0}, got {1}".format(
                sch.name, type(df).__name__
            ),
            name="coerce",
        )
    declared, extra = _check_columns(df, sch, allow_extra, "coerce " + sch.name)
    data = {
        col.name: _coerce_series(df[col.name], col, sch.name) for col in sch.columns
    }
    out = pd.DataFrame(data, columns=declared, index=df.index)
    if extra:
        out = pd.concat([out, df[extra]], axis=1)
    return out


# --------------------------------------------------------------------------
# checked I/O
# --------------------------------------------------------------------------


def _suffix(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def write_checked(
    df: pd.DataFrame,
    path: Any,
    schema: Union[str, Schema],
    coerce_dtypes: bool = True,
    allow_extra: bool = False,
) -> str:
    """Validate ``df`` against ``schema`` and write it to ``path``.

    Nothing reaches disk unless the contract holds, so a malformed manifest can
    never become the next stage's input. The write is atomic (temp file +
    ``os.replace``), column order is the schema's, and no index is written —
    so two runs producing the same rows produce byte-identical files.

    ``coerce_dtypes`` (default ``True``) casts the frame to the declared dtypes
    first, which is what lets a producer build rows from dicts; the cast is
    lossless or it raises. ``allow_extra`` writes undeclared columns after the
    declared ones instead of rejecting them.

    Supported extensions: ``.csv``, ``.tsv``, ``.parquet``.
    """
    sch = get_schema(schema)
    path = os.fspath(path)
    frame = coerce(df, sch, allow_extra=allow_extra) if coerce_dtypes else df
    validate(frame, sch, allow_extra=allow_extra, where=path)
    # declared columns first, in schema order; any allowed extras keep theirs
    frame = frame[
        sch.column_names + [c for c in frame.columns if c not in sch.column_names]
    ]

    suffix = _suffix(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=parent or ".", delete=False, suffix=".tmp"
    )
    handle.close()
    tmp_path = handle.name
    try:
        if suffix in (".csv", ".tsv"):
            frame.to_csv(
                tmp_path,
                index=False,
                sep="\t" if suffix == ".tsv" else ",",
                lineterminator="\n",
            )
        elif suffix == ".parquet":
            frame.to_parquet(tmp_path, index=False)
        else:
            raise SchemaViolation(
                "unsupported extension {0!r} for {1}; use .csv, .tsv or "
                ".parquet".format(suffix, path),
                name=sch.name,
            )
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    logger.info("wrote %d rows to %s (schema %s)", len(frame), path, sch.name)
    return path


def read_checked(
    path: Any,
    schema: Union[str, Schema],
    allow_extra: bool = False,
) -> pd.DataFrame:
    """Read ``path`` and validate it against ``schema`` before returning it.

    Validating on read as well as on write is the point: it catches a file
    edited by hand, produced by an older revision, or written by a stage that
    was not itself checked.

    CSV/TSV files are read with ``keep_default_na=False, na_values=['']`` so
    only an empty field is missing. Without that, pandas turns the literal
    strings ``NA``, ``N/A``, ``null``, ``nan`` and ``None`` into NaN — and
    ``nan`` is exactly what 10,698 old job directories carry as their gene
    token.
    """
    sch = get_schema(schema)
    path = os.fspath(path)
    if not os.path.exists(path):
        raise SchemaViolation(
            "{0} does not exist. A missing prerequisite is an error, never a "
            "fallback: the stage that should have produced this "
            "{1} manifest must run first.".format(path, sch.name),
            name=sch.name,
        )
    suffix = _suffix(path)
    if suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        header = pd.read_csv(path, sep=sep, nrows=0)
        dtypes = {
            name: dtype
            for name, dtype in sch.dtypes.items()
            if name in header.columns
        }
        # int64 columns are read as Int64 first so a stray empty field fails
        # in validate() with a useful message instead of a bare pandas error.
        read_dtypes = {
            k: ("Int64" if v == "int64" else v) for k, v in dtypes.items()
        }
        df = pd.read_csv(
            path,
            sep=sep,
            dtype=read_dtypes,
            keep_default_na=False,
            na_values=[""],
        )
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise SchemaViolation(
            "unsupported extension {0!r} for {1}; use .csv, .tsv or "
            ".parquet".format(suffix, path),
            name=sch.name,
        )

    coerced = coerce(df, sch, allow_extra=allow_extra)
    validate(coerced, sch, allow_extra=allow_extra, where=path)
    logger.info("read %d rows from %s (schema %s)", len(coerced), path, sch.name)
    return coerced


def validate_file(
    path: Any,
    schema: Union[str, Schema],
    allow_extra: bool = False,
) -> Dict[str, Any]:
    """Read and validate a file, returning the validation report only."""
    sch = get_schema(schema)
    df = read_checked(path, sch, allow_extra=allow_extra)
    return validate(df, sch, allow_extra=allow_extra, where=os.fspath(path))
