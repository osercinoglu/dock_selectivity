#!/usr/bin/env python3
"""
Negative Sample Generator -- SUPERSEDED, DO NOT USE FOR NEW RUNS
===============================================================

    ############################################################
    #  DEPRECATED.  This stage is replaced by                  #
    #      sample_negatives.py                                 #
    #  It builds its positive set through Mapped_Gene_Name, a  #
    #  column in which 57.73% of rows name a DIFFERENT         #
    #  protein (KNOWN_ISSUE_uniprot_gene_mapping.md), so both  #
    #  the positives it balances against and the "known        #
    #  interactions" it avoids are wrong.  It refuses to run   #
    #  unless --legacy-gene-resolution is given, and it is     #
    #  retained only so an old campaign can be reproduced      #
    #  bug-for-bug.                                            #
    ############################################################

Implemented strategy: Balanced Negative Sampling (Najm et al.)
--------------------------------------------------------------
Each drug appears in the same number of positive and negative samples, and so
does each target, which stops a model learning "drug X binds everything" or
"target Y binds nothing". A greedy algorithm satisfies both constraints at
once: count each drug's and each target's positives, then draw, for every drug,
that many non-interacting targets, always preferring the target that is
furthest from its own quota.

It replicated batch_dock.py's pairing logic rather than reading
required_structures.csv, so that the negatives sat in the same universe as the
positives -- and it inherited batch_dock.py's identity defect along with it.

Why it is wrong: the symbol -> accession step is
``dict(zip(Mapped_Gene_Name, Entry))``, last-duplicate-wins over a corrupted
column. sample_negatives.py resolves on the Entrez key through IdentityIndex
(collapse factor 1), draws reproducibly from --seed, and reports its deficits.

Every input path is now a flag with no machine-specific default, and the draw's
random seed is a flag too: reproducing an old run means stating that run's
paths and seed, not inheriting a dead workstation's layout.

Usage:
    # refuses, exit code 2 -- this is the intended outcome
    python3 generate_negative_samples.py

    # deliberate legacy reproduction
    python3 generate_negative_samples.py --legacy-gene-resolution \\
        --edges /opt/data/multiscale_interactome_data/1_drug_to_protein.tsv \\
        --drugs /opt/data/drugbank/small_molecule_drug_links.csv \\
        --uniprot-mapping ./uniprot_gene_mapping.csv \\
        --cavity-mapping ./cavity_mapping.csv \\
        --ligands /opt/data/drugbank/drugbank_approved_split \\
        --out ./required_structures_with_negatives.csv \\
        --metadata-out ./negative_samples_metadata.json \\
        --seed 42 --run-dir /path/to/campaign

Input:
    --edges            drug-protein interaction edges (TSV)
    --drugs            small-molecule DrugBank links (CSV)
    --uniprot-mapping  the corrupted UniProt/gene mapping (CSV)
    --cavity-mapping   extracted cavity structures (CSV, from extract_cavities.py)
    --ligands          directory of per-drug SDF files

Output:
    --out              required_structures_with_negatives.csv
    --metadata-out     negative_samples_metadata.json
    <out-dir>/generate_negative_samples.log   (see --out-dir / --log-file)
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from dock_selectivity.runtime import add_runtime_arguments, resolve_out_dir, step

STEP_NAME = "generate_negative_samples"

# --- Configuration ---
# Defaults only; every one of these is a flag. Reference datasets under
# /opt/data are machine-wide and may default; run-specific files may not.
DEFAULT_DRUG_TO_PROTEIN_TSV = "/opt/data/multiscale_interactome_data/1_drug_to_protein.tsv"
DEFAULT_SMALL_MOLECULE_DRUGS_CSV = "/opt/data/drugbank/small_molecule_drug_links.csv"
DEFAULT_UNIPROT_MAPPING_CSV = "uniprot_gene_mapping.csv"
DEFAULT_CAVITY_MAPPING_CSV = "cavity_mapping.csv"
DEFAULT_LIGAND_SDF_FOLDER = "/opt/data/drugbank/drugbank_approved_split"
DEFAULT_OUTPUT_CSV = "required_structures_with_negatives.csv"
DEFAULT_METADATA_JSON = "negative_samples_metadata.json"
DEFAULT_SEED = 42

#: Guard against a greedy loop that can never satisfy its constraints. An
#: implementation detail of the loop, not a knob anyone tunes per run.
MAX_GREEDY_ATTEMPTS = 1000000

#: The columns batch_dock.py reads back out of --out. A contract, not a setting.
REQUIRED_MAPPING_COLUMNS = ['Entry', 'Mapped_Gene_Name']

logger = logging.getLogger(STEP_NAME)


class MissingInput(RuntimeError):
    """A declared input does not exist. A missing prerequisite is not a default."""


def _read_or_fail(path, what, **kwargs):
    """Load a table, or fail naming the flag that would fix it."""
    if not path or not os.path.exists(path):
        raise MissingInput("%s not found: %r" % (what, path))
    frame = pd.read_csv(path, **kwargs)
    logger.info("Loaded %d rows from %s (%s)", len(frame), path, what)
    return frame


def _gene_to_uniprot(uniprot_df):
    """THE DEFECT: last-duplicate-wins over a column that is 57.73% wrong."""
    missing = [c for c in REQUIRED_MAPPING_COLUMNS if c not in uniprot_df.columns]
    if missing:
        raise MissingInput(
            "UniProt mapping is missing required columns %s; found %s"
            % (missing, list(uniprot_df.columns)))
    return dict(zip(uniprot_df['Mapped_Gene_Name'], uniprot_df['Entry']))


def load_positive_samples(edges_path, drugs_path, uniprot_mapping_path,
                          cavity_mapping_path, ligand_folder):
    """Load positive samples by replicating batch_dock.py's (corrupted) logic.

    1. drug-protein interactions, filtered to small molecules
    2. gene symbol -> UniProt accession through the corrupted mapping
    3. one sample per available cavity of each mapped accession
    """
    logger.info("LOADING POSITIVE SAMPLES (replicating batch_dock.py logic)")

    drug_to_protein_df = _read_or_fail(edges_path, "drug-protein edges (--edges)", sep='\t')
    small_molecules_df = _read_or_fail(drugs_path, "small-molecule drug links (--drugs)")
    small_molecules = set(small_molecules_df['DrugBank ID'].unique())

    initial_rows = len(drug_to_protein_df)
    drug_to_protein_df = drug_to_protein_df[drug_to_protein_df['node_1'].isin(small_molecules)]
    logger.info("Filtered to %d small molecule drug-protein pairs (from %d)",
                len(drug_to_protein_df), initial_rows)

    uniprot_df = _read_or_fail(uniprot_mapping_path,
                               "UniProt/gene mapping (--uniprot-mapping)")
    gene_to_uniprot = _gene_to_uniprot(uniprot_df)
    logger.info("Loaded UniProt mapping for %d genes", len(gene_to_uniprot))

    cavity_df = _read_or_fail(cavity_mapping_path, "cavity mapping (--cavity-mapping)")
    logger.info("Unique proteins in cavity mapping: %d", cavity_df['UniProt_ID'].nunique())

    if not ligand_folder or not os.path.isdir(ligand_folder):
        raise MissingInput("ligand SDF folder not found: %r (--ligands)" % ligand_folder)

    positive_samples = []
    skipped_no_ligand = 0
    skipped_no_uniprot = 0
    skipped_no_cavity = 0

    logger.info("Creating positive drug-target-pocket combinations from %d interactions...",
                len(drug_to_protein_df))

    for idx, row in drug_to_protein_df.iterrows():
        drugbank_id = row['node_1']
        gene_name = row['node_2_name']

        if (idx + 1) % 1000 == 0:
            logger.info("  Processed %d/%d interactions, created %d samples...",
                        idx + 1, len(drug_to_protein_df), len(positive_samples))

        ligand_sdf_path = os.path.join(ligand_folder, "%s.sdf" % drugbank_id)
        if not os.path.exists(ligand_sdf_path):
            skipped_no_ligand += 1
            continue

        uniprot_id = gene_to_uniprot.get(gene_name)
        if not uniprot_id:
            skipped_no_uniprot += 1
            continue

        target_cavities = cavity_df[cavity_df['UniProt_ID'] == uniprot_id]
        if target_cavities.empty:
            skipped_no_cavity += 1
            continue

        for _, cavity_row in target_cavities.iterrows():
            # Fragment_ID arrives as an int from some producers and as "F1" from
            # others; batch_dock.py only understands the F-prefixed spelling.
            fragment_id = cavity_row.get('Fragment_ID', 'F1')
            if isinstance(fragment_id, (int, float)):
                fragment_id = "F%d" % int(fragment_id)
            elif not isinstance(fragment_id, str):
                fragment_id = 'F1'

            positive_samples.append({
                'UniProt_ID': uniprot_id,
                'Fragment_ID': fragment_id,
                'Cavity_Index': cavity_row.get('Cavity_Index', 1),
                'Pocket_PDB': cavity_row.get('Pocket_PDB', ''),
                'Gene_Name': gene_name,
                'Interaction_Count': 1,  # This is a known interaction
                'sample_type': 'positive',
                'drugbank_id': drugbank_id,
            })

    positive_df = pd.DataFrame(positive_samples)

    if positive_df.empty:
        raise MissingInput(
            "no positive samples could be created "
            "(%d skipped for no ligand SDF, %d for no UniProt mapping, "
            "%d for no cavity structures)"
            % (skipped_no_ligand, skipped_no_uniprot, skipped_no_cavity))

    logger.info("Created %d positive drug-target-pocket combinations", len(positive_df))
    logger.info("   Unique drugs: %d", positive_df['drugbank_id'].nunique())
    logger.info("   Unique proteins: %d", positive_df['UniProt_ID'].nunique())
    logger.info("   Unique drug-target pairs: %d",
                len(set(zip(positive_df['drugbank_id'], positive_df['UniProt_ID']))))

    for count, reason in ((skipped_no_ligand, "no ligand SDF"),
                          (skipped_no_uniprot, "no UniProt mapping"),
                          (skipped_no_cavity, "no cavity structures")):
        if count:
            logger.warning("Skipped %d interactions (%s)", count, reason)

    return positive_df


def load_known_interactions(edges_path, drugs_path, uniprot_mapping_path):
    """Load known drug-protein interactions to avoid generating false negatives.

    This reads ALL interactions, not just those with available structures or
    ligands, so a pair that is genuinely known never turns up as a negative.
    """
    logger.info("LOADING KNOWN INTERACTIONS")

    interactions_df = _read_or_fail(edges_path, "drug-protein edges (--edges)", sep='\t')
    small_molecules_df = _read_or_fail(drugs_path, "small-molecule drug links (--drugs)")
    small_molecules = set(small_molecules_df['DrugBank ID'].unique())

    uniprot_df = _read_or_fail(uniprot_mapping_path,
                               "UniProt/gene mapping (--uniprot-mapping)")
    gene_to_uniprot = _gene_to_uniprot(uniprot_df)

    known_pairs = set()
    for _, row in interactions_df.iterrows():
        drugbank_id = row['node_1']
        gene_name = row['node_2_name']
        if drugbank_id not in small_molecules:
            continue
        uniprot_id = gene_to_uniprot.get(gene_name)
        if uniprot_id:
            known_pairs.add((drugbank_id, uniprot_id))

    logger.info("Loaded %d known small molecule drug-target interactions", len(known_pairs))
    return known_pairs


def load_available_resources(positive_df, cavity_mapping_path, ligand_folder):
    """Load available drugs, proteins, and pockets.

    Only drugs and proteins that appear in the interaction database are loaded,
    so negatives are sampled from the same universe as the positives.
    """
    logger.info("LOADING AVAILABLE RESOURCES")

    drugs_in_db = set(positive_df['drugbank_id'].unique())
    proteins_in_db = set(positive_df['UniProt_ID'].unique())
    logger.info("Drugs in interaction database: %d", len(drugs_in_db))
    logger.info("Proteins in interaction database: %d", len(proteins_in_db))

    if not ligand_folder or not os.path.isdir(ligand_folder):
        raise MissingInput("ligand SDF folder not found: %r (--ligands)" % ligand_folder)

    drugs_with_sdf = set()
    for drug_id in drugs_in_db:
        if os.path.exists(os.path.join(ligand_folder, "%s.sdf" % drug_id)):
            drugs_with_sdf.add(drug_id)
    logger.info("   Drugs with SDF files: %d", len(drugs_with_sdf))

    cavity_df = _read_or_fail(cavity_mapping_path, "cavity mapping (--cavity-mapping)")

    protein_cavities = defaultdict(list)
    for _, row in cavity_df.iterrows():
        uniprot_id = row['UniProt_ID']
        if uniprot_id not in proteins_in_db:
            continue
        protein_cavities[uniprot_id].append({
            'Fragment_ID': row.get('Fragment_ID', 'F1'),
            'Cavity_Index': row.get('Cavity_Index', 1),
            'Pocket_PDB': row.get('Pocket_PDB', ''),
            'Gene_Name': row.get('Gene_Name', ''),
        })

    total_pockets = sum(len(cavities) for cavities in protein_cavities.values())
    logger.info("Available proteins (from interaction DB): %d", len(protein_cavities))
    logger.info("   Total pockets: %d", total_pockets)

    return drugs_with_sdf, protein_cavities


def validate_negatives(negatives_df, positive_df, known_pairs):
    """Validate that no negatives are actually positive samples."""
    logger.info("VALIDATING NEGATIVE SAMPLES")

    issues = []
    positive_pairs = set(zip(positive_df['drugbank_id'], positive_df['UniProt_ID']))
    negative_pairs = set(zip(negatives_df['drugbank_id'], negatives_df['UniProt_ID']))

    overlap = positive_pairs & negative_pairs
    if overlap:
        issues.append("%d negative samples overlap with positive samples!" % len(overlap))

    false_negatives = negative_pairs & known_pairs
    if false_negatives:
        issues.append("%d negatives are actually known interactions!" % len(false_negatives))

    duplicates = negatives_df.duplicated(
        subset=['drugbank_id', 'UniProt_ID', 'Cavity_Index']).sum()
    if duplicates > 0:
        issues.append("%d duplicate negative samples found!" % duplicates)

    if issues:
        for issue in issues:
            logger.warning(issue)
        return False
    logger.info("All validations passed: no overlap, no known interactions, no duplicates")
    return True


def generate_metadata(positive_df, negatives_df, known_pairs, seed):
    """Generate comprehensive metadata about the negative sampling process."""
    sample_type_counts = {}
    if 'sample_type' in negatives_df.columns:
        sample_type_counts = negatives_df['sample_type'].value_counts().to_dict()

    return {
        'timestamp': datetime.now().isoformat(),
        'seed': seed,
        'positive_samples': {
            'total': len(positive_df),
            'unique_drugs': int(positive_df['drugbank_id'].nunique()),
            'unique_proteins': int(positive_df['UniProt_ID'].nunique()),
            'unique_pairs': len(set(zip(positive_df['drugbank_id'],
                                        positive_df['UniProt_ID']))),
        },
        'negative_samples': {
            'total': len(negatives_df),
            'unique_drugs': int(negatives_df['drugbank_id'].nunique()),
            'unique_proteins': int(negatives_df['UniProt_ID'].nunique()),
            'unique_pairs': len(set(zip(negatives_df['drugbank_id'],
                                        negatives_df['UniProt_ID']))),
            'sample_type_breakdown': sample_type_counts,
        },
        'validation': {
            'known_interactions_total': len(known_pairs),
            'positive_negative_ratio': "1:%.2f" % (len(negatives_df) / len(positive_df)),
        },
    }


def save_results(combined_df, metadata, output_csv, metadata_json):
    """Save combined dataset and metadata."""
    logger.info("SAVING RESULTS")

    for path in (output_csv, metadata_json):
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    combined_df.to_csv(output_csv, index=False)
    logger.info("Saved combined dataset: %s", output_csv)
    logger.info("   Total samples: %d", len(combined_df))
    logger.info("   Positive: %d", (combined_df['sample_type'] == 'positive').sum())
    logger.info("   Negative: %d",
                combined_df['sample_type'].str.startswith('negative').sum())

    with open(metadata_json, 'w') as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("Saved metadata: %s", metadata_json)

    return output_csv, metadata_json


def generate_balanced_negatives(positive_df, known_pairs, drugs_with_sdf,
                                protein_cavities, seed=DEFAULT_SEED):
    """Generate balanced negative samples using the Najm et al. greedy algorithm.

    For each drug, draw as many negative targets as it has positives, always
    preferring the target that is furthest from its own quota, so drugs and
    targets end up equally represented on both sides.

    ``seed`` seeds both the drug shuffle and the per-target cavity choice: it is
    the only thing that makes a repeat of this draw reproducible.
    """
    logger.info("GENERATING BALANCED NEGATIVE SAMPLES (Najm et al.), seed=%s", seed)

    drug_positive_counts = positive_df.groupby('drugbank_id').size().to_dict()
    target_positive_counts = positive_df.groupby('UniProt_ID').size().to_dict()

    logger.info("Unique drugs in positives: %d", len(drug_positive_counts))
    logger.info("Unique targets in positives: %d", len(target_positive_counts))
    logger.info("Average positive interactions per drug: %.2f",
                np.mean(list(drug_positive_counts.values())))
    logger.info("Average positive interactions per target: %.2f",
                np.mean(list(target_positive_counts.values())))

    drug_negative_counts = defaultdict(int)
    target_negative_counts = defaultdict(int)

    available_drugs = [d for d in drug_positive_counts.keys() if d in drugs_with_sdf]
    available_targets = list(protein_cavities.keys())

    logger.info("Available drugs with SDF files: %d", len(available_drugs))
    logger.info("Available targets with structures: %d", len(available_targets))

    negatives = []
    attempts = 0

    rng = np.random.RandomState(seed)
    drugs_to_process = available_drugs.copy()
    rng.shuffle(drugs_to_process)

    logger.info("Starting greedy negative sample generation...")

    while drugs_to_process and attempts < MAX_GREEDY_ATTEMPTS:
        attempts += 1

        drug_found = False
        for drug_id in drugs_to_process[:]:
            needed_negatives = drug_positive_counts[drug_id] - drug_negative_counts[drug_id]
            if needed_negatives <= 0:
                drugs_to_process.remove(drug_id)
                continue

            drug_found = True

            # Prefer targets that still owe negatives; fall back to any target
            # that is not a known interaction of this drug.
            candidate_targets = [
                t for t in available_targets
                if target_negative_counts[t] < target_positive_counts.get(t, 0)
                and (drug_id, t) not in known_pairs
            ]
            if not candidate_targets:
                candidate_targets = [
                    t for t in available_targets if (drug_id, t) not in known_pairs
                ]
            if not candidate_targets:
                drugs_to_process.remove(drug_id)
                logger.warning("Drug %s cannot find valid negative targets", drug_id)
                continue

            target_id = max(
                candidate_targets,
                key=lambda t: target_positive_counts.get(t, 0) - target_negative_counts[t]
            )

            # rng.shuffle/rng.choice on a fresh RandomState(seed) draw exactly the
            # stream the old global np.random.seed(42) did, so an old run with
            # --seed 42 reproduces bit for bit.
            cavity_info = rng.choice(protein_cavities[target_id])

            negatives.append({
                'UniProt_ID': target_id,
                'Fragment_ID': cavity_info['Fragment_ID'],
                'Cavity_Index': cavity_info['Cavity_Index'],
                'Pocket_PDB': cavity_info['Pocket_PDB'],
                'Gene_Name': cavity_info['Gene_Name'],
                'Interaction_Count': 0,
                'sample_type': 'negative_balanced',
                'drugbank_id': drug_id,
            })
            drug_negative_counts[drug_id] += 1
            target_negative_counts[target_id] += 1

            if len(negatives) % 1000 == 0:
                logger.info("  Generated %d negatives...", len(negatives))

            break

        if not drug_found:
            break

    logger.info("Generated %d balanced negative samples in %d attempts",
                len(negatives), attempts)
    logger.info("  Drugs fully balanced: %d/%d",
                sum(1 for d in drug_positive_counts
                    if drug_negative_counts[d] == drug_positive_counts[d]),
                len(drug_positive_counts))
    logger.info("  Targets fully balanced: %d/%d",
                sum(1 for t in target_positive_counts
                    if target_negative_counts[t] == target_positive_counts[t]),
                len(target_positive_counts))

    underbalanced_drugs = [
        (d, drug_positive_counts[d], drug_negative_counts[d])
        for d in drug_positive_counts
        if drug_negative_counts[d] < drug_positive_counts[d]
    ]
    if underbalanced_drugs:
        logger.warning("%d drugs are underbalanced:", len(underbalanced_drugs))
        for drug_id, needed, got in underbalanced_drugs[:5]:
            logger.warning("  %s: needed %d, got %d", drug_id, needed, got)
        if len(underbalanced_drugs) > 5:
            logger.warning("  ... and %d more", len(underbalanced_drugs) - 5)

    return negatives


def run_legacy_sampling(args):
    """The whole superseded workflow, with every path stated by the caller."""
    start_time = datetime.now()

    positive_df = load_positive_samples(
        edges_path=args.edges,
        drugs_path=args.drugs,
        uniprot_mapping_path=args.uniprot_mapping,
        cavity_mapping_path=args.cavity_mapping,
        ligand_folder=args.ligands,
    )
    known_pairs = load_known_interactions(
        edges_path=args.edges,
        drugs_path=args.drugs,
        uniprot_mapping_path=args.uniprot_mapping,
    )
    drugs_with_sdf, protein_cavities = load_available_resources(
        positive_df, cavity_mapping_path=args.cavity_mapping, ligand_folder=args.ligands)

    logger.info("RESOURCES LOADED - READY FOR NEGATIVE SAMPLING")
    logger.info("Positive samples: %d", len(positive_df))
    logger.info("Available drugs with SDF: %d", len(drugs_with_sdf))
    logger.info("Available proteins: %d", len(protein_cavities))
    logger.info("Known interactions to avoid: %d", len(known_pairs))

    negatives = generate_balanced_negatives(
        positive_df, known_pairs, drugs_with_sdf, protein_cavities, seed=args.seed)

    negatives_df = pd.DataFrame(negatives)
    if negatives_df.empty:
        raise MissingInput("failed to generate any negative samples")

    validate_negatives(negatives_df, positive_df, known_pairs)

    combined_df = pd.concat([positive_df, negatives_df], ignore_index=True)

    metadata = generate_metadata(positive_df, negatives_df, known_pairs, args.seed)
    metadata['strategy'] = {
        'name': 'Balanced Negative Sampling',
        'reference': 'Najm et al.',
        'description': 'Each drug and target appears in equal number of '
                       'positive and negative samples',
        'superseded_by': list(REPLACEMENT_STAGES),
    }

    save_results(combined_df, metadata, args.out, args.metadata_out)

    duration = (datetime.now() - start_time).total_seconds()
    logger.info("NEGATIVE SAMPLE GENERATION COMPLETE in %.2f seconds", duration)
    logger.info("Total samples: %d (positive %d, negative %d)",
                len(combined_df),
                (combined_df['sample_type'] == 'positive').sum(),
                combined_df['sample_type'].str.startswith('negative').sum())
    logger.info("Actual ratio: 1:%.2f", len(negatives_df) / len(positive_df))
    return combined_df


# ---------------------------------------------------------------------------
# Superseded -- FIX_PLAN 3.2
# ---------------------------------------------------------------------------

#: The stages that replace this one. Both resolve on ``node_2`` -- the Entrez
#: Gene ID that has been present and unused in 1_drug_to_protein.tsv all along --
#: through IdentityIndex, which guarantees a collapse factor of exactly 1.
REPLACEMENT_STAGES = ('sample_negatives.py',)


class CorruptedResolverRefused(RuntimeError):
    """Raised when this script's gene-symbol resolver is invoked.

    ``dict(zip(uniprot_df['Mapped_Gene_Name'], uniprot_df['Entry']))`` is
    last-duplicate-wins over a column in which 57.73% of rows name a different
    protein. It collapses 4,585 accessions onto 1,795 symbols, silently
    discarding 2,790, and the accession that survives each collapse became the
    receptor that was docked -- 633 of the campaign's 979 proteins (64.66%) are
    not the protein their job name claims.
    """


LEGACY_WARNING = (
    "*** --legacy-gene-resolution: using the CORRUPTED symbol resolver. "
    "57.73% of the mapping rows name a different protein. Nothing this run "
    "produces may be trusted as identity. ***")


def assert_resolver_not_used(allow_legacy=False):
    """Refuse the corrupted path unless explicitly opted into."""
    if allow_legacy:
        logger.warning(LEGACY_WARNING)
        return
    raise CorruptedResolverRefused(
        "generate_negative_samples.py resolves identity with dict(zip(Mapped_Gene_Name, Entry)), "
        "which is last-duplicate-wins over a column that is 57.73% wrong. "
        "This stage is superseded by: "
        + ", ".join(REPLACEMENT_STAGES)
        + ". They resolve on the Entrez key via IdentityIndex (collapse factor 1). "
        "Pass --legacy-gene-resolution to run this anyway."
    )


def build_parser():
    """The CLI. Nothing here is required, so a bare run reaches the refusal."""
    parser = argparse.ArgumentParser(
        prog="generate_negative_samples.py",
        description="SUPERSEDED by %s -- refuses unless --legacy-gene-resolution."
                    % ", ".join(REPLACEMENT_STAGES),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--legacy-gene-resolution', action='store_true',
                        help='run the superseded, corrupted symbol resolver anyway')
    parser.add_argument('--edges', default=DEFAULT_DRUG_TO_PROTEIN_TSV,
                        help='drug-protein interaction edges TSV (default: %(default)s)')
    parser.add_argument('--drugs', default=DEFAULT_SMALL_MOLECULE_DRUGS_CSV,
                        help='small-molecule DrugBank links CSV (default: %(default)s)')
    parser.add_argument('--uniprot-mapping', default=DEFAULT_UNIPROT_MAPPING_CSV,
                        help='UniProt/gene mapping CSV, resolved against the invocation '
                             'directory (default: %(default)s). This is the 57.73%%-wrong '
                             'file; see KNOWN_ISSUE_uniprot_gene_mapping.md')
    parser.add_argument('--cavity-mapping', default=DEFAULT_CAVITY_MAPPING_CSV,
                        help='cavity mapping CSV from extract_cavities.py (default: %(default)s)')
    parser.add_argument('--ligands', default=DEFAULT_LIGAND_SDF_FOLDER,
                        help='directory of per-drug SDF files; a drug without one is '
                             'never sampled (default: %(default)s)')
    parser.add_argument('--out', default=DEFAULT_OUTPUT_CSV,
                        help='combined positives+negatives CSV to write (default: %(default)s)')
    parser.add_argument('--metadata-out', default=DEFAULT_METADATA_JSON,
                        help='sampling metadata JSON to write (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='random seed for the greedy draw; the only thing that makes '
                             'a repeat of this draw reproducible (default: %(default)s)')
    parser.add_argument('--run-dir', default=None,
                        help='campaign run directory; the step log goes to '
                             '<run-dir>/steps/%s/ unless --out-dir says otherwise' % STEP_NAME)
    add_runtime_arguments(parser, STEP_NAME)
    return parser


def main(argv=None):
    """Refuse by default; run the legacy sampler only when told to, twice."""
    args = build_parser().parse_args(argv)
    try:
        assert_resolver_not_used(args.legacy_gene_resolution)
    except CorruptedResolverRefused as exc:
        # Logging is deliberately not configured yet: a refused run creates no
        # directory and no log file. The operator still has to be told why.
        sys.stderr.write("%s\n" % exc)
        return 2

    resolve_out_dir(args, STEP_NAME, args.run_dir)
    with step(STEP_NAME, args) as ctx:
        ctx.logger.warning(LEGACY_WARNING)
        try:
            run_legacy_sampling(args)
        except MissingInput as exc:
            ctx.logger.error("%s", exc)
            # SystemExit inside step() is reported as a failure, not a quiet return.
            raise SystemExit(2)
        ctx.logger.info("Next step: extract_alphafold_models.py, which will use %s",
                        args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
