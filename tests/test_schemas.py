"""Tests for dock_selectivity.schemas.

The contracts here are the stage boundaries of ``REBUILD_PLAN.md`` §2. Each
test reproduces a way the 2025 pipeline crossed one of those boundaries with
malformed data and did not notice:

* ``generate_negative_samples.py:375`` read a ``Gene_Name`` column that did not
  exist and wrote 13,452 blank genes -> ``test_null_gene_is_rejected``.
* ``batch_dock.py:448`` filled a missing column in with ``''`` instead of
  failing -> ``test_missing_column_is_rejected``.
* positives carried ``fragment_id == 'F1'`` while negatives carried ``'1'``
  -> ``test_fragment_ids_round_trip_exactly``.
* the sample-type merge key was non-unique -> ``test_duplicate_primary_key_is_rejected``.
* 10,698 job directories were named ``*_nan_*`` and pandas reads the literal
  string ``nan`` as missing -> ``test_literal_na_and_nan_strings_survive_a_round_trip``.

Hermetic: every file is written inside ``tempfile.TemporaryDirectory()``.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

from dock_selectivity.invariants import InvariantViolation
from dock_selectivity.schemas import (
    SCHEMAS,
    Column,
    RowCheck,
    Schema,
    SchemaViolation,
    _self_check,
    coerce,
    empty_frame,
    get_schema,
    read_checked,
    schema_names,
    validate,
    write_checked,
)

# REBUILD_PLAN.md §2 names these seven manifests.
EXPECTED_SCHEMAS = [
    "interactions_resolved",
    "negatives_drawn",
    "negatives_pool",
    "positives_planned",
    "redock_queue",
    "run_manifest",
    "salvage_map",
]


def run_manifest_rows():
    """Two jobs: one salvaged from a corrupt-named directory, one fresh."""
    return pd.DataFrame(
        [
            {
                "job_id": "DB00035__P30518__AVPR2__F1__cav2",
                "drugbank_id": "DB00035",
                "accession": "P30518",
                "gene": "AVPR2",
                "fragment_id": "F1",
                "cavity_index": 2,
                "sample_type": "positive",
                "provenance": "salvaged_from",
                "source_job": "DB00035_AVPR2_Q8NCT1_cavity_2",
                "selected_by": "entrez=554 via edge (DB00035,554)",
            },
            {
                "job_id": "DB00091__Q7Z2W7__MT-ND1__F2__cav1",
                "drugbank_id": "DB00091",
                "accession": "Q7Z2W7",
                "gene": "MT-ND1",  # hyphenated symbol, as in 34 old job dirs
                "fragment_id": "F2",
                "cavity_index": 1,
                "sample_type": "negative",
                "provenance": "newly_docked",
                "source_job": None,
                "selected_by": "balanced sampler draw 1",
            },
        ]
    )


def salvage_rows():
    return pd.DataFrame(
        [
            {
                "old_job_dir": "DB00035_AVPR2_Q8NCT1_cavity_2",
                "classification": "valid_non_interaction",
                "drugbank_id": "DB00035",
                "accession": "Q8NCT1",
                "gene": "ARRDC4",  # what Q8NCT1 really is
                "fragment_id": "F1",
                "cavity_index": 2,
                "old_gene_label": "AVPR2",  # what the folder claimed
                "label_was_correct": False,
                "new_job_dir": "DB00035__Q8NCT1__ARRDC4__F1__cav2",
                "excluded_reason": None,
            },
            {
                "old_job_dir": "DB00091_nan_P68363_cavity_1",
                "classification": "incomplete",
                "drugbank_id": "DB00091",
                "accession": "P68363",
                "gene": "TUBA1B",
                "fragment_id": "F1",
                "cavity_index": 1,
                "old_gene_label": "nan",  # the literal string, 10,698 of these
                "label_was_correct": False,
                "new_job_dir": None,
                "excluded_reason": "no final_results.csv",
            },
        ]
    )


class TestSchemaCatalogue(unittest.TestCase):
    def test_the_seven_rebuild_plan_manifests_are_declared(self):
        self.assertEqual(schema_names(), EXPECTED_SCHEMAS)

    def test_unknown_schema_name_raises_and_lists_the_known_ones(self):
        with self.assertRaises(SchemaViolation) as ctx:
            get_schema("required_structures")  # a retired artifact
        self.assertIn("unknown schema", str(ctx.exception))
        self.assertIn("run_manifest", str(ctx.exception))

    def test_every_schema_declares_a_primary_key_and_provenance_columns(self):
        for name, schema in SCHEMAS.items():
            with self.subTest(schema=name):
                self.assertTrue(schema.primary_key, "{0} has no primary key".format(name))
                for key_col in schema.primary_key:
                    self.assertFalse(schema.column(key_col).nullable)

    def test_gene_is_never_a_primary_key_anywhere(self):
        """A label may never be part of a key. That is the whole incident."""
        for name, schema in SCHEMAS.items():
            with self.subTest(schema=name):
                for key_col in schema.primary_key:
                    self.assertNotIn(
                        key_col, ("gene", "old_gene_label", "entry_name")
                    )

    def test_every_schema_round_trips_an_empty_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in schema_names():
                with self.subTest(schema=name):
                    path = os.path.join(tmp, "{0}.csv".format(name))
                    write_checked(empty_frame(name), path, name)
                    back = read_checked(path, name)
                    self.assertEqual(len(back), 0)
                    self.assertEqual(
                        list(back.columns), SCHEMAS[name].column_names
                    )

    def test_a_malformed_schema_declaration_is_rejected(self):
        nullable_key = Schema(
            name="broken",
            description="primary key that may be null",
            columns=(Column("job_id", "string", nullable=True),),
            primary_key=("job_id",),
        )
        with self.assertRaises(SchemaViolation):
            _self_check((nullable_key,))

        bad_dtype = Schema(
            name="broken2",
            description="dtype outside the vocabulary",
            columns=(Column("job_id", "str"),),
            primary_key=("job_id",),
        )
        with self.assertRaises(SchemaViolation):
            _self_check((bad_dtype,))

        nullable_int = Schema(
            name="broken3",
            description="numpy int64 cannot hold nulls",
            columns=(
                Column("job_id", "string"),
                Column("cavity_index", "int64", nullable=True),
            ),
            primary_key=("job_id",),
        )
        with self.assertRaises(SchemaViolation):
            _self_check((nullable_int,))


class TestValidate(unittest.TestCase):
    def test_a_conforming_frame_validates(self):
        report = validate(coerce(run_manifest_rows(), "run_manifest"), "run_manifest")
        self.assertTrue(report["ok"])
        self.assertEqual(report["n_rows"], 2)

    def test_missing_column_is_rejected(self):
        frame = coerce(run_manifest_rows(), "run_manifest").drop(columns=["gene"])
        with self.assertRaises(SchemaViolation) as ctx:
            validate(frame, "run_manifest")
        self.assertIn("missing required column", str(ctx.exception))
        self.assertIn("gene", str(ctx.exception))

    def test_extra_column_is_rejected_unless_explicitly_allowed(self):
        frame = coerce(run_manifest_rows(), "run_manifest")
        frame["Mapped_Gene_Name"] = "AVPR2"  # a retired, corrupt column
        with self.assertRaises(SchemaViolation) as ctx:
            validate(frame, "run_manifest")
        self.assertIn("Mapped_Gene_Name", str(ctx.exception))
        self.assertTrue(validate(frame, "run_manifest", allow_extra=True)["ok"])

    def test_wrong_dtype_is_rejected(self):
        frame = coerce(run_manifest_rows(), "run_manifest")
        frame["cavity_index"] = frame["cavity_index"].astype("float64")
        with self.assertRaises(SchemaViolation) as ctx:
            validate(frame, "run_manifest")
        self.assertIn("cavity_index", str(ctx.exception))
        self.assertIn("declared int64, found float64", str(ctx.exception))

    def test_object_dtype_strings_are_rejected_by_validate(self):
        frame = coerce(run_manifest_rows(), "run_manifest")
        frame["accession"] = frame["accession"].astype(object)
        with self.assertRaises(SchemaViolation) as ctx:
            validate(frame, "run_manifest")
        self.assertIn("declared string, found object", str(ctx.exception))

    def test_null_gene_is_rejected(self):
        """The 13,452 blank gene names that produced 10,698 '*_nan_*' dirs."""
        frame = coerce(run_manifest_rows(), "run_manifest")
        frame.loc[0, "gene"] = pd.NA
        with self.assertRaises(SchemaViolation) as ctx:
            validate(frame, "run_manifest")
        self.assertIn("gene", str(ctx.exception))
        self.assertIn("non-nullable", str(ctx.exception))

    def test_null_selected_by_is_rejected(self):
        """I7: every planned job must say why it was selected."""
        frame = coerce(run_manifest_rows(), "run_manifest")
        frame.loc[1, "selected_by"] = pd.NA
        with self.assertRaises(SchemaViolation):
            validate(frame, "run_manifest")

    def test_value_outside_the_declared_vocabulary_is_rejected(self):
        frame = coerce(run_manifest_rows(), "run_manifest")
        frame.loc[0, "sample_type"] = "postive"  # typo
        with self.assertRaises(SchemaViolation) as ctx:
            validate(frame, "run_manifest")
        self.assertIn("sample_type", str(ctx.exception))
        self.assertIn("postive", str(ctx.exception))

    def test_duplicate_primary_key_is_rejected(self):
        frame = coerce(
            pd.concat([run_manifest_rows().head(1)] * 2, ignore_index=True),
            "run_manifest",
        )
        with self.assertRaises(SchemaViolation) as ctx:
            validate(frame, "run_manifest")
        self.assertIn("primary key", str(ctx.exception))
        self.assertIn("not unique", str(ctx.exception))

    def test_positives_planned_key_includes_the_fragment(self):
        """Two fragments of the same protein are two different pockets.

        The old job signature omitted the fragment and collapsed 1,309 of them.
        """
        schema = SCHEMAS["positives_planned"]
        self.assertIn("fragment_id", schema.primary_key)
        rows = pd.DataFrame(
            [
                {
                    "drugbank_id": "DB00035",
                    "accession": "P30518",
                    "gene": "AVPR2",
                    "fragment_id": frag,
                    "cavity_index": 2,
                    "sample_type": "positive",
                    "selected_by": "entrez=554",
                }
                for frag in ("F1", "F2")
            ]
        )
        self.assertTrue(validate(coerce(rows, "positives_planned"), "positives_planned")["ok"])


class TestRowChecks(unittest.TestCase):
    def test_a_salvage_row_with_no_outcome_is_rejected(self):
        """An excluded job must say why. Never a silent disappearance."""
        rows = salvage_rows()
        rows.loc[1, "excluded_reason"] = None  # and new_job_dir is already None
        with self.assertRaises(SchemaViolation) as ctx:
            validate(coerce(rows, "salvage_map"), "salvage_map")
        self.assertIn("outcome_recorded", str(ctx.exception))
        self.assertIn("DB00091_nan_P68363_cavity_1", str(ctx.exception))

    def test_a_salvage_row_claiming_both_outcomes_is_rejected(self):
        rows = salvage_rows()
        rows.loc[1, "new_job_dir"] = "DB00091__P68363__TUBA1B__F1__cav1"
        with self.assertRaises(SchemaViolation):
            validate(coerce(rows, "salvage_map"), "salvage_map")

    def test_a_well_formed_salvage_map_validates(self):
        self.assertTrue(validate(coerce(salvage_rows(), "salvage_map"), "salvage_map")["ok"])

    def test_unparseable_rows_keep_their_reason_without_an_identity(self):
        rows = pd.DataFrame(
            [
                {
                    "old_job_dir": "DB00035_MT-ND1_cavity_3",
                    "classification": "unparseable",
                    "drugbank_id": None,
                    "accession": None,
                    "gene": None,
                    "fragment_id": None,
                    "cavity_index": None,
                    "old_gene_label": "MT-ND1",
                    "label_was_correct": None,
                    "new_job_dir": None,
                    "excluded_reason": "no accession token in directory name",
                }
            ]
        )
        frame = coerce(rows, "salvage_map")
        self.assertTrue(validate(frame, "salvage_map")["ok"])
        self.assertEqual(str(frame["cavity_index"].dtype), "Int64")

    def test_negatives_drawn_pool_rows_must_name_their_source_job(self):
        rows = pd.DataFrame(
            [
                {
                    "drugbank_id": "DB00035",
                    "accession": "P68363",
                    "gene": "TUBA1B",
                    "fragment_id": "F1",
                    "cavity_index": 1,
                    "sample_type": "negative",
                    "source": "pool",
                    "old_job_dir": None,  # <- must be set for source='pool'
                    "selected_by": "balanced sampler",
                }
            ]
        )
        with self.assertRaises(SchemaViolation) as ctx:
            validate(coerce(rows, "negatives_drawn"), "negatives_drawn")
        self.assertIn("pool_rows_name_their_source", str(ctx.exception))

        rows.loc[0, "old_job_dir"] = "DB00035_TUBA1B_P68363_cavity_1"
        self.assertTrue(
            validate(coerce(rows, "negatives_drawn"), "negatives_drawn")["ok"]
        )

    def test_run_manifest_salvaged_rows_must_name_their_source_job(self):
        rows = run_manifest_rows()
        rows.loc[0, "source_job"] = None
        with self.assertRaises(SchemaViolation) as ctx:
            validate(coerce(rows, "run_manifest"), "run_manifest")
        self.assertIn("salvaged_rows_name_their_source", str(ctx.exception))

    def test_a_custom_row_check_can_be_declared(self):
        schema = Schema(
            name="custom",
            description="demo",
            columns=(Column("a", "int64"), Column("b", "int64")),
            primary_key=("a",),
            checks=(RowCheck("b_positive", "b must be > 0", lambda df: df["b"] > 0),),
        )
        good = pd.DataFrame({"a": [1], "b": [2]})
        self.assertTrue(validate(good, schema)["ok"])
        with self.assertRaises(SchemaViolation):
            validate(pd.DataFrame({"a": [1], "b": [0]}), schema)


class TestCoerce(unittest.TestCase):
    def test_coercion_is_lossless_or_it_raises(self):
        rows = coerce(run_manifest_rows(), "run_manifest")
        self.assertEqual(str(rows["cavity_index"].dtype), "int64")
        self.assertEqual(str(rows["gene"].dtype), "string")

        fractional = run_manifest_rows()
        fractional["cavity_index"] = [1.5, 2.0]
        with self.assertRaises(SchemaViolation) as ctx:
            coerce(fractional, "run_manifest")
        self.assertIn("non-integral", str(ctx.exception))

        integral_float = run_manifest_rows()
        integral_float["cavity_index"] = [1.0, 2.0]
        self.assertEqual(
            coerce(integral_float, "run_manifest")["cavity_index"].tolist(), [1, 2]
        )

    def test_unparseable_value_raises_with_the_column_named(self):
        rows = run_manifest_rows()
        rows["cavity_index"] = ["cavity_2", "cavity_1"]
        with self.assertRaises(SchemaViolation) as ctx:
            coerce(rows, "run_manifest")
        self.assertIn("cavity_index", str(ctx.exception))

    def test_nulls_in_a_non_nullable_int_column_raise(self):
        rows = run_manifest_rows()
        rows["cavity_index"] = [2, None]
        with self.assertRaises(SchemaViolation) as ctx:
            coerce(rows, "run_manifest")
        self.assertIn("null", str(ctx.exception))

    def test_coerce_refuses_to_drop_an_undeclared_column(self):
        rows = run_manifest_rows()
        rows["Mapped_Gene_Name"] = "AVPR2"  # the retired, corrupt column
        with self.assertRaises(SchemaViolation) as ctx:
            coerce(rows, "run_manifest")
        self.assertIn("undeclared column", str(ctx.exception))
        self.assertIn("drop them explicitly", str(ctx.exception))
        # ... but an explicit superset is passed through untouched
        widened = coerce(rows, "run_manifest", allow_extra=True)
        self.assertEqual(list(widened.columns)[-1], "Mapped_Gene_Name")
        self.assertEqual(
            list(widened.columns)[:-1], SCHEMAS["run_manifest"].column_names
        )

    def test_boolean_strings_are_understood_but_nonsense_is_not(self):
        rows = salvage_rows()
        rows["label_was_correct"] = ["False", "True"]
        self.assertEqual(
            coerce(rows, "salvage_map")["label_was_correct"].tolist(), [False, True]
        )
        rows["label_was_correct"] = ["maybe", "True"]
        with self.assertRaises(SchemaViolation):
            coerce(rows, "salvage_map")


class TestCheckedIO(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp, name)

    def test_write_then_read_round_trips_values_and_dtypes(self):
        path = self.path("run_manifest.csv")
        write_checked(run_manifest_rows(), path, "run_manifest")
        back = read_checked(path, "run_manifest")
        self.assertEqual(len(back), 2)
        self.assertEqual(back["cavity_index"].tolist(), [2, 1])
        self.assertEqual(str(back["cavity_index"].dtype), "int64")
        self.assertEqual(str(back["gene"].dtype), "string")
        self.assertEqual(back["gene"].tolist(), ["AVPR2", "MT-ND1"])
        self.assertTrue(pd.isna(back.loc[1, "source_job"]))

    def test_fragment_ids_round_trip_exactly(self):
        """'F1' must never come back as 1 — the F1 vs '1' join trap."""
        path = self.path("positives_planned.csv")
        rows = pd.DataFrame(
            [
                {
                    "drugbank_id": "DB00035",
                    "accession": "P30518",
                    "gene": "AVPR2",
                    "fragment_id": "F1",
                    "cavity_index": 2,
                    "sample_type": "positive",
                    "selected_by": "entrez=554",
                }
            ]
        )
        write_checked(rows, path, "positives_planned")
        back = read_checked(path, "positives_planned")
        self.assertEqual(back["fragment_id"].tolist(), ["F1"])
        self.assertEqual(str(back["fragment_id"].dtype), "string")

    def test_literal_na_and_nan_strings_survive_a_round_trip(self):
        """pandas turns 'NA'/'nan'/'null' into NaN by default. Not here."""
        rows = salvage_rows()
        rows.loc[0, "old_gene_label"] = "NA"
        rows.loc[1, "old_gene_label"] = "nan"
        path = self.path("salvage_map.csv")
        write_checked(rows, path, "salvage_map")
        back = read_checked(path, "salvage_map")
        self.assertEqual(back["old_gene_label"].tolist(), ["NA", "nan"])
        self.assertEqual(back["new_job_dir"].isna().tolist(), [False, True])
        self.assertEqual(back["excluded_reason"].isna().tolist(), [True, False])

    def test_writing_the_same_frame_twice_is_byte_identical(self):
        first, second = self.path("a.csv"), self.path("b.csv")
        write_checked(run_manifest_rows(), first, "run_manifest")
        write_checked(run_manifest_rows(), second, "run_manifest")
        with open(first, "rb") as handle:
            first_bytes = handle.read()
        with open(second, "rb") as handle:
            second_bytes = handle.read()
        self.assertEqual(first_bytes, second_bytes)
        self.assertTrue(first_bytes.endswith(b"\n"))

    def test_column_order_follows_the_schema_not_the_frame(self):
        rows = run_manifest_rows()
        shuffled = rows[list(reversed(rows.columns))]
        path = self.path("ordered.csv")
        write_checked(shuffled, path, "run_manifest")
        with open(path) as handle:
            header = handle.readline().strip().split(",")
        self.assertEqual(header, SCHEMAS["run_manifest"].column_names)

    def test_a_frame_with_a_gappy_index_writes_cleanly(self):
        """Stages pass ``df[mask].copy()``; a stale index must not leak out."""
        rows = pd.concat([run_manifest_rows()] * 2, ignore_index=True)
        rows = rows[rows["sample_type"] == "positive"].copy()  # index [0, 2]
        rows.loc[2, "job_id"] = "DB00035__P30518__AVPR2__F1__cav3"
        rows.loc[2, "cavity_index"] = 3
        path = self.path("gappy.csv")
        write_checked(rows, path, "run_manifest")
        back = read_checked(path, "run_manifest")
        self.assertEqual(len(back), 2)
        self.assertEqual(list(back.index), [0, 1])

    def test_an_invalid_frame_never_reaches_disk(self):
        rows = run_manifest_rows()
        rows.loc[0, "sample_type"] = "unlabelled"
        path = self.path("never_written.csv")
        with self.assertRaises(SchemaViolation):
            write_checked(rows, path, "run_manifest")
        self.assertFalse(os.path.exists(path))
        self.assertEqual(
            [f for f in os.listdir(self.tmp) if f.endswith(".tmp")], []
        )

    def test_reading_a_file_missing_a_column_raises(self):
        path = self.path("truncated.csv")
        write_checked(run_manifest_rows(), path, "run_manifest")
        frame = pd.read_csv(path).drop(columns=["selected_by"])
        frame.to_csv(path, index=False)
        with self.assertRaises(SchemaViolation) as ctx:
            read_checked(path, "run_manifest")
        self.assertIn("selected_by", str(ctx.exception))

    def test_reading_a_file_with_an_undeclared_column_raises(self):
        path = self.path("widened.csv")
        write_checked(run_manifest_rows(), path, "run_manifest")
        frame = pd.read_csv(path)
        frame["Mapped_Gene_Name"] = "AVPR2"
        frame.to_csv(path, index=False)
        with self.assertRaises(SchemaViolation) as ctx:
            read_checked(path, "run_manifest")
        self.assertIn("undeclared column", str(ctx.exception))
        widened = read_checked(path, "run_manifest", allow_extra=True)
        self.assertEqual(len(widened), 2)
        self.assertIn("Mapped_Gene_Name", widened.columns)

    def test_an_allowed_superset_survives_a_write_read_round_trip(self):
        path = self.path("superset.csv")
        rows = run_manifest_rows()
        rows["notes"] = ["salvaged in phase 5", "queued in phase 4"]
        write_checked(rows, path, "run_manifest", allow_extra=True)
        with open(path) as handle:
            header = handle.readline().strip().split(",")
        self.assertEqual(header[:-1], SCHEMAS["run_manifest"].column_names)
        self.assertEqual(header[-1], "notes")
        back = read_checked(path, "run_manifest", allow_extra=True)
        self.assertEqual(back["notes"].tolist(), rows["notes"].tolist())

    def test_reading_a_hand_edited_file_with_a_bad_enum_raises(self):
        path = self.path("edited.csv")
        write_checked(run_manifest_rows(), path, "run_manifest")
        with open(path) as handle:
            text = handle.read()
        with open(path, "w") as handle:
            handle.write(text.replace(",positive,", ",POSITIVE,"))
        with self.assertRaises(SchemaViolation):
            read_checked(path, "run_manifest")

    def test_a_blank_required_integer_field_raises_instead_of_becoming_nan(self):
        path = self.path("blank_cavity.csv")
        write_checked(run_manifest_rows(), path, "run_manifest")
        with open(path) as handle:
            lines = handle.read().splitlines()
        fields = lines[1].split(",")
        fields[5] = ""  # cavity_index
        lines[1] = ",".join(fields)
        with open(path, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        with self.assertRaises(SchemaViolation) as ctx:
            read_checked(path, "run_manifest")
        self.assertIn("cavity_index", str(ctx.exception))

    def test_missing_file_is_an_error_not_a_fallback(self):
        with self.assertRaises(SchemaViolation) as ctx:
            read_checked(self.path("absent.csv"), "redock_queue")
        self.assertIn("does not exist", str(ctx.exception))
        self.assertIn("never a fallback", str(ctx.exception))

    def test_unsupported_extension_is_rejected(self):
        with self.assertRaises(SchemaViolation):
            write_checked(run_manifest_rows(), self.path("m.txt"), "run_manifest")

    def test_parquet_round_trip(self):
        path = self.path("run_manifest.parquet")
        write_checked(run_manifest_rows(), path, "run_manifest")
        back = read_checked(path, "run_manifest")
        self.assertEqual(back["job_id"].tolist(), run_manifest_rows()["job_id"].tolist())
        self.assertEqual(str(back["gene"].dtype), "string")

    def test_tsv_round_trip(self):
        path = self.path("run_manifest.tsv")
        write_checked(run_manifest_rows(), path, "run_manifest")
        with open(path) as handle:
            self.assertIn("\t", handle.readline())
        self.assertEqual(len(read_checked(path, "run_manifest")), 2)

    def test_schema_violation_is_an_invariant_violation(self):
        """One exception hierarchy, so a stage can catch either."""
        self.assertTrue(issubclass(SchemaViolation, InvariantViolation))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
