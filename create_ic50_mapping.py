#!/usr/bin/env python3
"""Build the DrugBank-UniProt IC50 table that ``parse_prepare.py`` annotates with.

Joins four sources into one row per (drug, target, measurement):

1. TTD target download   -- TTD Target ID  -> UniProt entry name (``UNIPROID``)
2. A UniProt identity map -- UniProt entry name -> UniProt accession
3. DrugBank drug links   -- PubChem CID    -> DrugBank ID
4. TTD activity data     -- IC50/Ki values per (TTD target, TTD drug)

The identity map is read for ``entry name -> accession`` only. That direction is
a verified bijection (see ``build_identity_index.py``), so this stage is *not*
affected by the gene-symbol corruption described in
``KNOWN_ISSUE_uniprot_gene_mapping.md`` -- it never reads ``Mapped_Gene_Name``.
Both vocabularies are accepted: the legacy ``Entry`` / ``Entry Name`` columns and
the rebuilt index's ``accession`` / ``entry_name`` (CSV or Parquet).

Inputs (all overridable; the reference datasets default to ``/opt/data``):
    --ttd-targets, --ttd-activity, --drug-links, --uniprot-mapping

Output:
    --out, a CSV with columns: drugbank_id, uniprot_id, ttd_target_id,
    ttd_drug_id, pubchem_cid, activity, measurement_type, operator, ic50_value,
    ic50_unit. This is the file ``parse_prepare.py --ic50-mapping`` consumes.

Usage::

    python3 create_ic50_mapping.py \
        --uniprot-mapping <run-dir>/manifests/identity_index.parquet \
        --out <run-dir>/ic50_mapping.csv \
        --run-dir <run-dir>

    # with an explicit reference-dataset location and a start/finish ping
    python3 create_ic50_mapping.py --uniprot-mapping index.parquet --out ic50.csv \
        --ttd-targets /opt/data/ttd/P1-01-TTD_target_download.txt \
        --notify --notify-on all
"""

import argparse
import csv
import logging
import os
import re

import polars as pl

from dock_selectivity.runtime import add_runtime_arguments, resolve_out_dir, step

STEP_NAME = "create_ic50_mapping"

# --- Configuration ------------------------------------------------------------
# Defaults only: every one of these is a flag. The /opt/data paths are the
# machine-wide reference datasets, not a particular campaign's run directory.
DEFAULT_TTD_TARGET_FILE = "/opt/data/ttd/P1-01-TTD_target_download.txt"
DEFAULT_TTD_ACTIVITY_FILE = "/opt/data/ttd/P1-09-Target_compound_activity.txt"
DEFAULT_DRUGBANK_LINKS_FILE = "/opt/data/drugbank/small_molecule_drug_links.csv"

# --- Implementation detail (file format, not configuration) -------------------
# Header/prose lines TTD ships above the tab-separated body.
_TTD_PREAMBLE = ('TTD -', 'Title', 'Version', 'Provided', 'Any question', 'Dr.',
                 '---', 'Abbreviations')
# 'IC50 = 3270000 nM', 'Ki > 10 uM', ... -> (type, operator, value, unit).
_ACTIVITY_PATTERN = r'(IC50|Ki)\s*([>=<]+)\s*([\d.]+)\s*([nuµμ]?M)'
# The two column vocabularies an identity map may use, entry-name column first.
_IDENTITY_COLUMNS = (("Entry Name", "Entry"), ("entry_name", "accession"))
OUTPUT_COLUMNS = ['drugbank_id', 'uniprot_id', 'ttd_target_id', 'ttd_drug_id',
                  'pubchem_cid', 'activity', 'measurement_type', 'operator',
                  'ic50_value', 'ic50_unit']

LOG = logging.getLogger(STEP_NAME)


def parse_ttd_targets(filepath):
    """
    Parse TTD target download file to extract TTD Target ID -> UniProt Entry Name mapping.

    The file format is:
    TTD_TARGET_ID   FIELD_TYPE   VALUE

    Returns:
        dict: {ttd_target_id: uniprot_entry_name}
    """
    LOG.info("Parsing TTD targets from: %s", filepath)

    ttd_to_uniprot = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(_TTD_PREAMBLE):
                continue

            parts = line.split('\t')
            if len(parts) >= 3:
                ttd_id = parts[0]
                field_type = parts[1]
                value = parts[2] if len(parts) > 2 else ''

                # Look for UNIPROID field
                if field_type == 'UNIPROID':
                    ttd_to_uniprot[ttd_id] = value

    LOG.info("   Found %s TTD targets with UniProt mappings", format(len(ttd_to_uniprot), ','))
    return ttd_to_uniprot


def load_uniprot_mapping(filepath):
    """
    Load a UniProt identity map and return {entry_name: accession}.

    Accepts either vocabulary -- the legacy ``Entry Name`` / ``Entry`` columns or
    the rebuilt index's ``entry_name`` / ``accession`` -- and either format (a
    ``.parquet`` index or a CSV). Only that one direction is read; it is a
    bijection, unlike the gene-symbol columns beside it.
    """
    LOG.info("Loading UniProt identity map from: %s", filepath)

    if str(filepath).endswith(".parquet"):
        df = pl.read_parquet(filepath)
    else:
        df = pl.read_csv(filepath)

    for name_column, id_column in _IDENTITY_COLUMNS:
        if name_column in df.columns and id_column in df.columns:
            break
    else:
        # An unusable map must stop the run: silently producing zero mappings
        # here would look like "TTD had no matching targets".
        raise ValueError(
            "{0} has none of the expected identity column pairs {1}; it has {2}".format(
                filepath, list(_IDENTITY_COLUMNS), df.columns))

    mapping = dict(zip(df[name_column].to_list(), df[id_column].to_list()))

    LOG.info("   Loaded %s UniProt mappings from columns %s -> %s",
             format(len(mapping), ','), name_column, id_column)
    return mapping


def load_drugbank_pubchem_mapping(drugbank_file):
    """
    Load mapping from DrugBank file to get PubChem Compound ID -> DrugBank ID mapping.
    """
    pubchem_to_drugbank = {}

    with open(drugbank_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            drugbank_id = row.get('DrugBank ID', '').strip()
            pubchem_id = row.get('PubChem Compound ID', '').strip()

            if drugbank_id and pubchem_id:
                # Store the mapping (PubChem ID may map to multiple DrugBank IDs, keep first)
                if pubchem_id not in pubchem_to_drugbank:
                    pubchem_to_drugbank[pubchem_id] = drugbank_id

    return pubchem_to_drugbank


def parse_ic50_value(activity_string):
    """
    Parse IC50/Ki value from activity string.

    Examples:
        'IC50 = 3270000 nM' -> (3270000.0, 'nM', 'IC50', '=')
        'IC50 > 10000 nM' -> (10000.0, 'nM', 'IC50', '>')
        'Ki = 2390 nM' -> (2390.0, 'nM', 'Ki', '=')

    Returns:
        tuple: (value, unit, measurement_type, operator) or (None, None, None, None)
    """
    match = re.search(_ACTIVITY_PATTERN, activity_string, re.IGNORECASE)
    if match:
        measurement_type = match.group(1)
        operator = match.group(2)
        value = float(match.group(3))
        unit = match.group(4)

        return value, unit, measurement_type, operator

    return None, None, None, None


def parse_activity_data(filepath, ttd_target_to_uniprot, uniprot_name_to_id, pubchem_to_drugbank):
    """
    Parse the TTD activity file and create final mapping table.

    Args:
        filepath: Path to P1-09-Target_compound_activity.txt
        ttd_target_to_uniprot: Mapping from TTD Target ID to UniProt Entry Name
        uniprot_name_to_id: Mapping from UniProt Entry Name to UniProt Entry ID
        pubchem_to_drugbank: Mapping from PubChem Compound ID to DrugBank ID

    Returns:
        list of dicts with keys: drugbank_id, uniprot_id, activity, pubchem_cid
    """
    LOG.info("Parsing activity data from: %s", filepath)

    results = []
    total_lines = 0
    mapped_lines = 0
    skipped_no_drugbank = 0
    skipped_no_uniprot = 0
    skipped_no_ic50 = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        # Skip header line
        next(f, None)

        for line in f:
            total_lines += 1
            parts = line.strip().split('\t')

            if len(parts) < 4:
                continue

            ttd_target_id = parts[0]
            ttd_drug_id = parts[1]
            pubchem_cid = parts[2] if len(parts) > 2 else ''
            activity = parts[3] if len(parts) > 3 else ''

            # Parse IC50 value
            ic50_value, ic50_unit, measurement_type, operator = parse_ic50_value(activity)

            if ic50_value is None:
                skipped_no_ic50 += 1
                continue

            # Map TTD Target ID to UniProt Entry Name
            uniprot_entry_name = ttd_target_to_uniprot.get(ttd_target_id)
            if not uniprot_entry_name:
                skipped_no_uniprot += 1
                continue

            # Map UniProt Entry Name to UniProt Entry ID
            uniprot_id = uniprot_name_to_id.get(uniprot_entry_name)
            if not uniprot_id:
                skipped_no_uniprot += 1
                continue

            # Map PubChem CID to DrugBank ID
            drugbank_id = pubchem_to_drugbank.get(pubchem_cid)
            if not drugbank_id:
                skipped_no_drugbank += 1
                continue

            # Successfully mapped!
            results.append({
                'drugbank_id': drugbank_id,
                'uniprot_id': uniprot_id,
                'ttd_target_id': ttd_target_id,
                'ttd_drug_id': ttd_drug_id,
                'pubchem_cid': pubchem_cid,
                'activity': activity,
                'measurement_type': measurement_type,
                'operator': operator,
                'ic50_value': ic50_value,
                'ic50_unit': ic50_unit
            })
            mapped_lines += 1

    mapped_pct = (mapped_lines / total_lines * 100) if total_lines else 0.0
    LOG.info("Activity data processing summary:")
    LOG.info("   Total activity records: %s", format(total_lines, ','))
    LOG.info("   Successfully mapped: %s (%.1f%%)", format(mapped_lines, ','), mapped_pct)
    LOG.info("   Skipped - no IC50 value: %s", format(skipped_no_ic50, ','))
    LOG.info("   Skipped - no DrugBank mapping: %s", format(skipped_no_drugbank, ','))
    LOG.info("   Skipped - no UniProt mapping: %s", format(skipped_no_uniprot, ','))

    return results


def write_results(results, output_file):
    """Write results to CSV file and log the resulting distributions."""
    LOG.info("Writing results to: %s", output_file)

    if not results:
        LOG.warning("No results to write!")
        return None

    # Convert to Polars DataFrame for efficient writing
    df = pl.DataFrame(results).select(OUTPUT_COLUMNS)

    parent = os.path.dirname(os.path.abspath(output_file))
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.write_csv(output_file)

    LOG.info("   Wrote %s records to %s", format(len(results), ','), output_file)

    LOG.info("Sample records (first 5):\n%s", df.head(5))
    LOG.info("Dataset statistics:")
    LOG.info("   Unique DrugBank IDs: %s", format(df['drugbank_id'].n_unique(), ','))
    LOG.info("   Unique UniProt IDs: %s", format(df['uniprot_id'].n_unique(), ','))
    LOG.info("   Unique drug-target pairs: %s",
             format(df.select(['drugbank_id', 'uniprot_id']).unique().height, ','))

    ic50_stats = df.select([
        pl.col('ic50_value').min().alias('min'),
        pl.col('ic50_value').max().alias('max'),
        pl.col('ic50_value').mean().alias('mean'),
        pl.col('ic50_value').median().alias('median')
    ])
    LOG.info("IC50 value statistics:\n%s", ic50_stats)
    LOG.info("Measurement type distribution:\n%s",
             df.group_by('measurement_type').agg(pl.len()).sort('measurement_type'))
    LOG.info("Operator distribution:\n%s",
             df.group_by('operator').agg(pl.len()).sort('operator'))
    LOG.info("Unit distribution:\n%s",
             df.group_by('ic50_unit').agg(pl.len()).sort('ic50_unit'))
    return df


def build_parser():
    """Build the command-line parser for the IC50 mapping stage."""
    parser = argparse.ArgumentParser(
        prog="create_ic50_mapping.py",
        description=("Join TTD activity data, TTD targets, DrugBank links and a "
                     "UniProt identity map into one IC50 table keyed by "
                     "(drugbank_id, uniprot_id)."))
    parser.add_argument('--uniprot-mapping', required=True,
                        help=('UniProt identity map providing entry name -> accession: '
                              'identity_index.parquet, or a legacy CSV with the '
                              '"Entry Name" and "Entry" columns. MANDATORY'))
    parser.add_argument('--out', required=True,
                        help='IC50 mapping CSV to write (consumed by parse_prepare.py). MANDATORY')
    parser.add_argument('--ttd-targets', default=DEFAULT_TTD_TARGET_FILE,
                        help='TTD target download file, for TTD ID -> UniProt entry name '
                             '(default: %(default)s)')
    parser.add_argument('--ttd-activity', default=DEFAULT_TTD_ACTIVITY_FILE,
                        help='TTD target-compound activity file holding the IC50/Ki strings '
                             '(default: %(default)s)')
    parser.add_argument('--drug-links', default=DEFAULT_DRUGBANK_LINKS_FILE,
                        help='DrugBank small-molecule drug links CSV, for PubChem CID -> '
                             'DrugBank ID (default: %(default)s)')
    parser.add_argument('--run-dir', default=None,
                        help='campaign run directory; when given, this step logs to '
                             '<run-dir>/steps/' + STEP_NAME + '/ (default: ./' + STEP_NAME + ')')
    return add_runtime_arguments(parser, STEP_NAME)


def main(argv=None):
    """Build the IC50 mapping table; returns a process exit code."""
    args = build_parser().parse_args(argv)
    resolve_out_dir(args, STEP_NAME, args.run_dir)

    try:
        with step(STEP_NAME, args) as ctx:
            missing = [path for path in (args.ttd_targets, args.ttd_activity,
                                         args.drug_links, args.uniprot_mapping)
                       if not os.path.exists(path)]
            if missing:
                # A missing source silently becomes "no IC50 values found", which
                # downstream reads as "this drug-target pair has no evidence".
                ctx.logger.error("input file(s) not found: %s", ", ".join(missing))
                raise SystemExit(2)

            ttd_to_uniprot = parse_ttd_targets(args.ttd_targets)
            uniprot_name_to_id = load_uniprot_mapping(args.uniprot_mapping)
            pubchem_to_drugbank = load_drugbank_pubchem_mapping(args.drug_links)
            ctx.logger.info("   Found %s PubChem IDs with DrugBank mappings",
                            format(len(pubchem_to_drugbank), ','))

            results = parse_activity_data(args.ttd_activity, ttd_to_uniprot,
                                          uniprot_name_to_id, pubchem_to_drugbank)
            write_results(results, args.out)
            ctx.logger.info("IC50 mapping complete: %s rows -> %s",
                            format(len(results), ','), args.out)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
