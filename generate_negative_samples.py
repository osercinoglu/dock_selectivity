#!/usr/bin/env python3
"""
Negative Sample Generator for Consensus Docking
================================================

This script generates negative samples (non-interacting drug-target pairs) to complement
the positive samples identified in the docking workflow. Negative samples are essential
for training and evaluating machine learning models on docking data.

Implemented Strategy: Balanced Negative Sampling (Najm et al.)
--------------------------------------------------------------
This strategy ensures balanced representation by maintaining equal numbers of positive
and negative samples for each drug and each target.

Key principles:
- Each drug appears in the same number of positive and negative samples
- Each target appears in the same number of positive and negative samples
- Prevents the model from learning spurious patterns like:
  * "Drug X interacts with everything"
  * "Target Y never interacts"

Algorithm (Greedy Approach):
1. Replicate batch_dock.py logic to identify actual positive drug-target-pocket combinations
2. Count positive interactions for each drug (n_drug) and target (n_target)
3. For each drug: select (n_drug) negative targets
4. For each target: ensure it appears (n_target) times in negatives
5. Use a greedy algorithm to satisfy both constraints simultaneously

This addresses a specific concern: drugs with 20 targets will have 20 negative samples,
while drugs with 1 target will have 1 negative sample, maintaining equal representation.

Important: This script replicates the same drug-target pairing logic as batch_dock.py
to ensure consistency. It doesn't just read required_structures.csv (which only lists
structures to extract), but actually determines which drugs interact with which targets
by consulting the drug-protein interaction database.

The script provides infrastructure for:
1. Loading actual positive drug-target-pocket combinations (like batch_dock.py)
2. Loading known drug-target interactions to avoid false negatives
3. Identifying available drugs (with SDF files) and protein structures
4. Validating generated negatives
5. Combining positive and negative samples into a single dataset
6. Generating metadata about the sampling process

Prerequisites:
- drug_to_protein.tsv (known interactions)
- small_molecule_drug_links.csv (available drugs)
- uniprot_gene_mapping.csv (UniProt to gene mapping, from prepare_uniprot_mapping.py)
- cavity_mapping.csv (available protein structures/pockets, from extract_cavities.py)
- Processed ligand SDF folder with individual drug files

Usage:
    python generate_negative_samples.py

Output:
    - required_structures_with_negatives.csv: Augmented structure list with negatives
    - negative_samples_metadata.json: Statistics and metadata about generated negatives
    - negative_samples.log: Processing log
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# --- Configuration ---
# Input files (same paths as batch_dock.py for consistency)
DRUG_TO_PROTEIN_TSV = "/opt/data/multiscale_interactome_data/1_drug_to_protein.tsv"
SMALL_MOLECULE_DRUGS_CSV = "/opt/data/drugbank/small_molecule_drug_links.csv"
UNIPROT_MAPPING_CSV = "uniprot_gene_mapping.csv"  # From prepare_uniprot_mapping.py
CAVITY_MAPPING_CSV = "cavity_mapping.csv"  # From extract_cavities.py
PROCESSED_LIGAND_SDF_FOLDER = "/opt/data/drugbank/drugbank_approved_split"

OUTPUT_CSV = "required_structures_with_negatives.csv"
METADATA_JSON = "negative_samples_metadata.json"
LOG_FILE = "negative_samples.log"

# Set up logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


def load_positive_samples():
    """
    Load positive samples by replicating batch_dock.py's logic.
    
    This function:
    1. Loads drug-protein interactions from DRUG_TO_PROTEIN_TSV
    2. Filters for small molecules only
    3. Maps gene names to UniProt IDs
    4. Finds available pockets for each target
    5. Creates explicit drug-target-pocket combinations (the actual positive samples)
    """
    logging.info("=" * 60)
    logging.info("LOADING POSITIVE SAMPLES (replicating batch_dock.py logic)")
    logging.info("=" * 60)
    
    # Step 1: Load drug-protein interactions
    if not os.path.exists(DRUG_TO_PROTEIN_TSV):
        logging.error(f"❌ Drug-protein interaction file not found: {DRUG_TO_PROTEIN_TSV}")
        sys.exit(1)
    
    drug_to_protein_df = pd.read_csv(DRUG_TO_PROTEIN_TSV, sep='\t')
    logging.info(f"✅ Loaded {len(drug_to_protein_df):,} drug-protein interactions")
    
    # Step 2: Load small molecule drugs
    if not os.path.exists(SMALL_MOLECULE_DRUGS_CSV):
        logging.error(f"❌ Small molecule drugs file not found: {SMALL_MOLECULE_DRUGS_CSV}")
        sys.exit(1)
    
    small_molecules_df = pd.read_csv(SMALL_MOLECULE_DRUGS_CSV)
    small_molecules = set(small_molecules_df['DrugBank ID'].unique())
    logging.info(f"✅ Loaded {len(small_molecules):,} small molecule drugs")
    
    # Step 3: Filter drug-protein interactions for small molecules only
    initial_rows = len(drug_to_protein_df)
    drug_to_protein_df = drug_to_protein_df[drug_to_protein_df['node_1'].isin(small_molecules)]
    logging.info(f"✅ Filtered to {len(drug_to_protein_df):,} small molecule drug-protein pairs (from {initial_rows:,})")
    
    # Step 4: Load UniProt mapping
    if not os.path.exists(UNIPROT_MAPPING_CSV):
        logging.error(f"❌ UniProt mapping file not found: {UNIPROT_MAPPING_CSV}")
        logging.error("Please run prepare_uniprot_mapping.py first!")
        sys.exit(1)
    
    uniprot_df = pd.read_csv(UNIPROT_MAPPING_CSV)
    
    # Debug: show what columns we actually have
    logging.info(f"   UniProt mapping columns: {list(uniprot_df.columns)}")
    
    # Check for required columns (consistent with batch_dock.py)
    required_columns = ['Entry', 'Mapped_Gene_Name']
    missing_columns = [col for col in required_columns if col not in uniprot_df.columns]
    
    if missing_columns:
        logging.error(f"❌ Required columns missing from UniProt mapping: {missing_columns}")
        logging.error(f"   Found columns: {list(uniprot_df.columns)}")
        logging.error("Please regenerate the mapping file using prepare_uniprot_mapping.py")
        sys.exit(1)
    
    # Create gene to UniProt mapping (same as batch_dock.py: Mapped_Gene_Name -> Entry)
    gene_to_uniprot = dict(zip(uniprot_df['Mapped_Gene_Name'], uniprot_df['Entry']))
    logging.info(f"✅ Loaded UniProt mapping for {len(gene_to_uniprot):,} genes")
    
    # Step 5: Load cavity mapping
    if not os.path.exists(CAVITY_MAPPING_CSV):
        logging.error(f"❌ Cavity mapping file not found: {CAVITY_MAPPING_CSV}")
        logging.error("Please run extract_cavities.py first!")
        sys.exit(1)
    
    cavity_df = pd.read_csv(CAVITY_MAPPING_CSV)
    logging.info(f"✅ Loaded cavity mapping with {len(cavity_df):,} structures")
    logging.info(f"   Cavity mapping columns: {list(cavity_df.columns)}")
    logging.info(f"   Unique proteins in cavity mapping: {cavity_df['UniProt_ID'].nunique():,}")
    
    # Step 6: Create positive samples by replicating batch_dock.py logic
    positive_samples = []
    skipped_no_ligand = 0
    skipped_no_uniprot = 0
    skipped_no_cavity = 0
    
    logging.info("\nCreating positive drug-target-pocket combinations...")
    logging.info(f"Processing {len(drug_to_protein_df):,} drug-protein interactions...")
    
    for idx, row in drug_to_protein_df.iterrows():
        drugbank_id = row['node_1']
        gene_name = row['node_2_name']
        
        # Log progress every 1000 interactions
        if (idx + 1) % 1000 == 0:
            logging.info(f"  Processed {idx + 1:,}/{len(drug_to_protein_df):,} interactions, created {len(positive_samples):,} samples...")
        
        # Check if ligand SDF exists (like batch_dock.py does)
        ligand_sdf_path = os.path.join(PROCESSED_LIGAND_SDF_FOLDER, f"{drugbank_id}.sdf")
        if not os.path.exists(ligand_sdf_path):
            skipped_no_ligand += 1
            continue
        
        # Map gene to UniProt ID
        uniprot_id = gene_to_uniprot.get(gene_name)
        if not uniprot_id:
            skipped_no_uniprot += 1
            continue
        
        # Find available cavities for this UniProt ID
        target_cavities = cavity_df[cavity_df['UniProt_ID'] == uniprot_id]
        if target_cavities.empty:
            skipped_no_cavity += 1
            continue
        
        # Create one positive sample for each cavity (like batch_dock.py does)
        for _, cavity_row in target_cavities.iterrows():
            # Handle Fragment_ID - convert integer to F-string format if needed
            fragment_id = cavity_row.get('Fragment_ID', 'F1')
            if isinstance(fragment_id, (int, float)):
                fragment_id = f"F{int(fragment_id)}"
            elif not isinstance(fragment_id, str):
                fragment_id = 'F1'
            
            positive_sample = {
                'UniProt_ID': uniprot_id,
                'Fragment_ID': fragment_id,
                'Cavity_Index': cavity_row.get('Cavity_Index', 1),
                'Pocket_PDB': cavity_row.get('Pocket_PDB', ''),
                'Gene_Name': gene_name,
                'Interaction_Count': 1,  # This is a known interaction
                'sample_type': 'positive',
                'drugbank_id': drugbank_id
            }
            positive_samples.append(positive_sample)
    
    # Convert to DataFrame
    positive_df = pd.DataFrame(positive_samples)
    
    # Check if we have any positive samples
    if positive_df.empty:
        logging.error("\n❌ No positive samples could be created!")
        logging.error("Reasons:")
        if skipped_no_ligand > 0:
            logging.error(f"  - {skipped_no_ligand:,} interactions skipped (no ligand SDF)")
        if skipped_no_uniprot > 0:
            logging.error(f"  - {skipped_no_uniprot:,} interactions skipped (no UniProt mapping)")
        if skipped_no_cavity > 0:
            logging.error(f"  - {skipped_no_cavity:,} interactions skipped (no cavity structures)")
        logging.error("\nPlease check:")
        logging.error("1. Drug-protein interaction file has valid data")
        logging.error("2. Small molecule drugs are available")
        logging.error("3. UniProt mapping is complete")
        logging.error("4. Cavity structures have been extracted")
        logging.error("5. Ligand SDF files are in the correct location")
        sys.exit(1)
    
    # Report statistics
    logging.info(f"\n✅ Created {len(positive_df):,} positive drug-target-pocket combinations")
    logging.info(f"   Unique drugs: {positive_df['drugbank_id'].nunique():,}")
    logging.info(f"   Unique proteins: {positive_df['UniProt_ID'].nunique():,}")
    logging.info(f"   Unique drug-target pairs: {len(set(zip(positive_df['drugbank_id'], positive_df['UniProt_ID']))):,}")
    
    if skipped_no_ligand > 0:
        logging.warning(f"⚠️  Skipped {skipped_no_ligand:,} interactions (no ligand SDF)")
    if skipped_no_uniprot > 0:
        logging.warning(f"⚠️  Skipped {skipped_no_uniprot:,} interactions (no UniProt mapping)")
    if skipped_no_cavity > 0:
        logging.warning(f"⚠️  Skipped {skipped_no_cavity:,} interactions (no cavity structures)")
    
    return positive_df


def load_known_interactions():
    """
    Load known drug-protein interactions to avoid generating false negatives.
    
    This function loads ALL drug-protein interactions (not just those with available
    structures/ligands) to ensure we don't accidentally create negatives for 
    interactions that are actually known, even if they're not in our positive set
    due to missing structures or ligands.
    """
    logging.info("\n" + "=" * 60)
    logging.info("LOADING KNOWN INTERACTIONS")
    logging.info("=" * 60)
    
    if not os.path.exists(DRUG_TO_PROTEIN_TSV):
        logging.error(f"❌ Drug-protein interaction file not found: {DRUG_TO_PROTEIN_TSV}")
        sys.exit(1)
    
    # Load all interactions
    interactions_df = pd.read_csv(DRUG_TO_PROTEIN_TSV, sep='\t')
    
    # Load small molecule filter
    if not os.path.exists(SMALL_MOLECULE_DRUGS_CSV):
        logging.error(f"❌ Small molecule drugs file not found: {SMALL_MOLECULE_DRUGS_CSV}")
        sys.exit(1)
    
    small_molecules_df = pd.read_csv(SMALL_MOLECULE_DRUGS_CSV)
    small_molecules = set(small_molecules_df['DrugBank ID'].unique())
    
    # Load UniProt mapping to convert gene names to UniProt IDs
    if not os.path.exists(UNIPROT_MAPPING_CSV):
        logging.error(f"❌ UniProt mapping file not found: {UNIPROT_MAPPING_CSV}")
        sys.exit(1)
    
    uniprot_df = pd.read_csv(UNIPROT_MAPPING_CSV)
    
    # Check for required columns (consistent with batch_dock.py)
    required_columns = ['Entry', 'Mapped_Gene_Name']
    missing_columns = [col for col in required_columns if col not in uniprot_df.columns]
    
    if missing_columns:
        logging.error(f"❌ Required columns missing from UniProt mapping: {missing_columns}")
        sys.exit(1)
    
    # Create gene to UniProt mapping (same as batch_dock.py: Mapped_Gene_Name -> Entry)
    gene_to_uniprot = dict(zip(uniprot_df['Mapped_Gene_Name'], uniprot_df['Entry']))
    
    # Create set of known interactions for fast lookup
    known_pairs = set()
    
    for _, row in interactions_df.iterrows():
        drugbank_id = row['node_1']  # DrugBank ID
        gene_name = row['node_2_name']  # Gene name
        
        # Only include small molecules
        if drugbank_id not in small_molecules:
            continue
            
        # Map gene name to UniProt ID
        uniprot_id = gene_to_uniprot.get(gene_name)
        if uniprot_id:
            known_pairs.add((drugbank_id, uniprot_id))
    
    logging.info(f"✅ Loaded {len(known_pairs):,} known small molecule drug-target interactions")
    
    return known_pairs


def load_available_resources(positive_df):
    """
    Load available drugs, proteins, and pockets.
    
    IMPORTANT: Only loads drugs and proteins that appear in the known interaction
    database (DRUG_TO_PROTEIN_TSV). This ensures negatives are sampled from the
    same universe as the positives.
    """
    logging.info("\n" + "=" * 60)
    logging.info("LOADING AVAILABLE RESOURCES")
    logging.info("=" * 60)
    
    # Get drugs and proteins that appear in the interaction database (from positives)
    drugs_in_db = set(positive_df['drugbank_id'].unique())
    proteins_in_db = set(positive_df['UniProt_ID'].unique())
    
    logging.info(f"✅ Drugs in interaction database: {len(drugs_in_db):,}")
    logging.info(f"✅ Proteins in interaction database: {len(proteins_in_db):,}")
    
    # Filter drugs that have SDF files
    drugs_with_sdf = set()
    if os.path.exists(PROCESSED_LIGAND_SDF_FOLDER):
        for drug_id in drugs_in_db:
            sdf_path = os.path.join(PROCESSED_LIGAND_SDF_FOLDER, f"{drug_id}.sdf")
            if os.path.exists(sdf_path):
                drugs_with_sdf.add(drug_id)
    
    logging.info(f"   Drugs with SDF files: {len(drugs_with_sdf):,}")
    
    # Load available protein structures and pockets (only for proteins in DB)
    if not os.path.exists(CAVITY_MAPPING_CSV):
        logging.error(f"❌ Cavity mapping file not found: {CAVITY_MAPPING_CSV}")
        sys.exit(1)
    
    cavity_df = pd.read_csv(CAVITY_MAPPING_CSV)
    
    # Create mapping of UniProt_ID to available cavities (only for proteins in DB)
    protein_cavities = defaultdict(list)
    for _, row in cavity_df.iterrows():
        uniprot_id = row['UniProt_ID']
        
        # Only include proteins that appear in the interaction database
        if uniprot_id not in proteins_in_db:
            continue
            
        cavity_info = {
            'Fragment_ID': row.get('Fragment_ID', 'F1'),
            'Cavity_Index': row.get('Cavity_Index', 1),
            'Pocket_PDB': row.get('Pocket_PDB', ''),
            'Gene_Name': row.get('Gene_Name', '')
        }
        protein_cavities[uniprot_id].append(cavity_info)
    
    available_proteins = set(protein_cavities.keys())
    total_pockets = sum(len(cavities) for cavities in protein_cavities.values())
    
    logging.info(f"✅ Available proteins (from interaction DB): {len(available_proteins):,}")
    logging.info(f"   Total pockets: {total_pockets:,}")
    
    return drugs_with_sdf, protein_cavities


def validate_negatives(negatives_df, positive_df, known_pairs):
    """Validate that no negatives are actually positive samples"""
    logging.info("\n" + "=" * 60)
    logging.info("VALIDATING NEGATIVE SAMPLES")
    logging.info("=" * 60)
    
    issues = []
    
    # Check for overlap with positive samples
    positive_pairs = set(zip(positive_df['drugbank_id'], positive_df['UniProt_ID']))
    negative_pairs = set(zip(negatives_df['drugbank_id'], negatives_df['UniProt_ID']))
    
    overlap = positive_pairs & negative_pairs
    if overlap:
        issues.append(f"⚠️  {len(overlap)} negative samples overlap with positive samples!")
    
    # Check for known interactions
    false_negatives = negative_pairs & known_pairs
    if false_negatives:
        issues.append(f"⚠️  {len(false_negatives)} negatives are actually known interactions!")
    
    # Check for duplicates within negatives
    duplicates = negatives_df.duplicated(subset=['drugbank_id', 'UniProt_ID', 'Cavity_Index']).sum()
    if duplicates > 0:
        issues.append(f"⚠️  {duplicates} duplicate negative samples found!")
    
    if issues:
        for issue in issues:
            logging.warning(issue)
        return False
    else:
        logging.info("✅ All validations passed!")
        logging.info(f"   No overlap with positive samples")
        logging.info(f"   No known interactions included")
        logging.info(f"   No duplicates found")
        return True


def generate_metadata(positive_df, negatives_df, known_pairs):
    """Generate comprehensive metadata about the negative sampling process"""
    
    # Count different sample types in negatives
    sample_type_counts = {}
    if 'sample_type' in negatives_df.columns:
        sample_type_counts = negatives_df['sample_type'].value_counts().to_dict()
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'positive_samples': {
            'total': len(positive_df),
            'unique_drugs': int(positive_df['drugbank_id'].nunique()),
            'unique_proteins': int(positive_df['UniProt_ID'].nunique()),
            'unique_pairs': len(set(zip(positive_df['drugbank_id'], positive_df['UniProt_ID'])))
        },
        'negative_samples': {
            'total': len(negatives_df),
            'unique_drugs': int(negatives_df['drugbank_id'].nunique()),
            'unique_proteins': int(negatives_df['UniProt_ID'].nunique()),
            'unique_pairs': len(set(zip(negatives_df['drugbank_id'], negatives_df['UniProt_ID']))),
            'sample_type_breakdown': sample_type_counts
        },
        'validation': {
            'known_interactions_total': len(known_pairs),
            'positive_negative_ratio': f"1:{len(negatives_df) / len(positive_df):.2f}"
        }
    }
    
    return metadata


def save_results(combined_df, metadata):
    """Save combined dataset and metadata"""
    logging.info("\n" + "=" * 60)
    logging.info("SAVING RESULTS")
    logging.info("=" * 60)
    
    # Save combined CSV
    combined_df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"✅ Saved combined dataset: {OUTPUT_CSV}")
    logging.info(f"   Total samples: {len(combined_df):,}")
    logging.info(f"   Positive: {(combined_df['sample_type'] == 'positive').sum():,}")
    logging.info(f"   Negative: {(combined_df['sample_type'].str.startswith('negative')).sum():,}")
    
    # Save metadata
    with open(METADATA_JSON, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"✅ Saved metadata: {METADATA_JSON}")
    
    return OUTPUT_CSV, METADATA_JSON


def generate_balanced_negatives(positive_df, known_pairs, drugs_with_sdf, protein_cavities):
    """
    Generate balanced negative samples using the Najm et al. greedy algorithm.
    
    Strategy:
    - For each drug: ensure it appears in equal number of positive and negative samples
    - For each target: ensure it appears in equal number of positive and negative samples
    
    This prevents the model from learning spurious patterns like:
    - "Drug X interacts with everything"
    - "Target Y never interacts"
    
    Algorithm:
    1. Count positive interactions for each drug (n_drug) and target (n_target)
    2. For each drug: select (n_drug) negative targets
    3. For each target: ensure it appears (n_target) times in negatives
    4. Use greedy algorithm to satisfy both constraints simultaneously
    
    Reference: Najm et al. - Balanced negative sampling approach
    """
    logging.info("\n" + "=" * 60)
    logging.info("GENERATING BALANCED NEGATIVE SAMPLES (Najm et al.)")
    logging.info("=" * 60)
    
    # Step 1: Count positive interactions for each drug and target
    drug_positive_counts = positive_df.groupby('drugbank_id').size().to_dict()
    target_positive_counts = positive_df.groupby('UniProt_ID').size().to_dict()
    
    logging.info(f"Unique drugs in positives: {len(drug_positive_counts)}")
    logging.info(f"Unique targets in positives: {len(target_positive_counts)}")
    logging.info(f"Average positive interactions per drug: {np.mean(list(drug_positive_counts.values())):.2f}")
    logging.info(f"Average positive interactions per target: {np.mean(list(target_positive_counts.values())):.2f}")
    
    # Step 2: Initialize tracking dictionaries
    drug_negative_counts = defaultdict(int)  # Track how many negatives each drug has
    target_negative_counts = defaultdict(int)  # Track how many negatives each target has
    
    # Get list of available drugs and targets
    available_drugs = [d for d in drug_positive_counts.keys() if d in drugs_with_sdf]
    available_targets = list(protein_cavities.keys())
    
    logging.info(f"Available drugs with SDF files: {len(available_drugs)}")
    logging.info(f"Available targets with structures: {len(available_targets)}")
    
    # Step 3: Greedy algorithm to generate negatives
    negatives = []
    max_attempts = 1000000  # Prevent infinite loops
    attempts = 0
    
    # Create a shuffled list of drugs to process
    np.random.seed(42)  # For reproducibility
    drugs_to_process = available_drugs.copy()
    np.random.shuffle(drugs_to_process)
    
    logging.info("\nStarting greedy negative sample generation...")
    
    while drugs_to_process and attempts < max_attempts:
        attempts += 1
        
        # Get next drug that needs more negatives
        drug_found = False
        for drug_id in drugs_to_process[:]:
            needed_negatives = drug_positive_counts[drug_id] - drug_negative_counts[drug_id]
            
            if needed_negatives <= 0:
                drugs_to_process.remove(drug_id)
                continue
            
            drug_found = True
            
            # Find a target that:
            # 1. Doesn't have a known interaction with this drug
            # 2. Still needs more negative samples
            # 3. Has available cavity structure
            
            # Create list of candidate targets (need more negatives)
            candidate_targets = [
                t for t in available_targets
                if target_negative_counts[t] < target_positive_counts.get(t, 0)
                and (drug_id, t) not in known_pairs
            ]
            
            if not candidate_targets:
                # If no candidates need negatives, use any available target
                candidate_targets = [
                    t for t in available_targets
                    if (drug_id, t) not in known_pairs
                ]
            
            if not candidate_targets:
                # This drug can't find any valid negative targets
                drugs_to_process.remove(drug_id)
                logging.warning(f"Drug {drug_id} cannot find valid negative targets")
                continue
            
            # Select target that needs the most negatives (greedy choice)
            target_id = max(
                candidate_targets,
                key=lambda t: target_positive_counts.get(t, 0) - target_negative_counts[t]
            )
            
            # Randomly select a cavity for this target
            cavity_info = np.random.choice(protein_cavities[target_id])
            
            # Create negative entry
            negative_entry = {
                'UniProt_ID': target_id,
                'Fragment_ID': cavity_info['Fragment_ID'],
                'Cavity_Index': cavity_info['Cavity_Index'],
                'Pocket_PDB': cavity_info['Pocket_PDB'],
                'Gene_Name': cavity_info['Gene_Name'],
                'Interaction_Count': 0,
                'sample_type': 'negative_balanced',
                'drugbank_id': drug_id
            }
            
            negatives.append(negative_entry)
            drug_negative_counts[drug_id] += 1
            target_negative_counts[target_id] += 1
            
            # Log progress periodically
            if len(negatives) % 1000 == 0:
                logging.info(f"  Generated {len(negatives):,} negatives...")
            
            break
        
        if not drug_found:
            break
    
    logging.info(f"\n✅ Generated {len(negatives):,} balanced negative samples")
    logging.info(f"   Attempts: {attempts:,}")
    
    # Step 4: Report statistics
    logging.info("\nBalance Statistics:")
    logging.info(f"  Drugs fully balanced: {sum(1 for d in drug_positive_counts if drug_negative_counts[d] == drug_positive_counts[d])}/{len(drug_positive_counts)}")
    logging.info(f"  Targets fully balanced: {sum(1 for t in target_positive_counts if target_negative_counts[t] == target_positive_counts[t])}/{len(target_positive_counts)}")
    
    # Report drugs that couldn't be fully balanced
    underbalanced_drugs = [
        (d, drug_positive_counts[d], drug_negative_counts[d])
        for d in drug_positive_counts
        if drug_negative_counts[d] < drug_positive_counts[d]
    ]
    
    if underbalanced_drugs:
        logging.warning(f"\n⚠️  {len(underbalanced_drugs)} drugs are underbalanced:")
        for drug_id, needed, got in underbalanced_drugs[:5]:  # Show first 5
            logging.warning(f"  {drug_id}: needed {needed}, got {got}")
        if len(underbalanced_drugs) > 5:
            logging.warning(f"  ... and {len(underbalanced_drugs) - 5} more")
    
    return negatives



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


def assert_resolver_not_used(allow_legacy=False):
    """Refuse the corrupted path unless explicitly opted into."""
    if allow_legacy:
        logging.warning(
            "*** --legacy-gene-resolution: using the CORRUPTED symbol resolver. "
            "57.73%% of the mapping rows name a different protein. ***")
        return
    raise CorruptedResolverRefused(
        "generate_negative_samples.py resolves identity with dict(zip(Mapped_Gene_Name, Entry)), "
        "which is last-duplicate-wins over a column that is 57.73% wrong. "
        "This stage is superseded by: "
        + ", ".join(REPLACEMENT_STAGES)
        + ". They resolve on the Entrez key via IdentityIndex (collapse factor 1). "
        "Pass --legacy-gene-resolution to run this anyway."
    )



def main():
    import argparse as _ap
    _p = _ap.ArgumentParser(description="superseded stage; see REPLACEMENT_STAGES")
    _p.add_argument('--legacy-gene-resolution', action='store_true',
                    help='run the superseded, corrupted symbol resolver anyway')
    _a, _ = _p.parse_known_args()
    try:
        assert_resolver_not_used(_a.legacy_gene_resolution)
    except CorruptedResolverRefused as _exc:
        logging.error('%s', _exc)
        return 2
    """Main execution function"""
    start_time = datetime.now()
    
    logging.info("=" * 60)
    logging.info("NEGATIVE SAMPLE GENERATOR FOR CONSENSUS DOCKING")
    logging.info("=" * 60)
    logging.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("Strategy: Balanced sampling (Najm et al.)")
    
    # Step 1: Load positive samples
    positive_df = load_positive_samples()
    n_positives = len(positive_df)
    
    # Step 2: Load known interactions
    known_pairs = load_known_interactions()
    
    # Step 3: Load available resources (restricted to drugs/proteins in interaction DB)
    drugs_with_sdf, protein_cavities = load_available_resources(positive_df)
    
    logging.info("\n" + "=" * 60)
    logging.info("RESOURCES LOADED - READY FOR NEGATIVE SAMPLING")
    logging.info("=" * 60)
    logging.info(f"Positive samples: {n_positives:,}")
    logging.info(f"Available drugs with SDF: {len(drugs_with_sdf):,}")
    logging.info(f"Available proteins: {len(protein_cavities):,}")
    logging.info(f"Known interactions to avoid: {len(known_pairs):,}")
    
    # Step 4: Generate balanced negative samples
    negatives = generate_balanced_negatives(
        positive_df, known_pairs, drugs_with_sdf, protein_cavities
    )
    
    # Step 5: Create DataFrame and validate
    negatives_df = pd.DataFrame(negatives)
    
    if negatives_df.empty:
        logging.error("❌ Failed to generate any negative samples!")
        sys.exit(1)
    
    validate_negatives(negatives_df, positive_df, known_pairs)
    
    # Step 6: Combine positive and negative samples
    combined_df = pd.concat([positive_df, negatives_df], ignore_index=True)
    
    # Step 7: Generate metadata
    metadata = generate_metadata(positive_df, negatives_df, known_pairs)
    
    # Add strategy information to metadata
    metadata['strategy'] = {
        'name': 'Balanced Negative Sampling',
        'reference': 'Najm et al.',
        'description': 'Each drug and target appears in equal number of positive and negative samples'
    }
    
    # Step 8: Save results
    save_results(combined_df, metadata)
    
    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logging.info("\n" + "=" * 60)
    logging.info("NEGATIVE SAMPLE GENERATION COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Duration: {duration:.2f} seconds")
    logging.info(f"Total samples: {len(combined_df):,}")
    logging.info(f"  Positive: {(combined_df['sample_type'] == 'positive').sum():,}")
    logging.info(f"  Negative: {(combined_df['sample_type'].str.startswith('negative')).sum():,}")
    logging.info(f"Actual ratio: 1:{len(negatives_df) / len(positive_df):.2f}")
    logging.info("\n✅ Next step: Run extract_alphafold_models.py with the augmented structure list")
    logging.info(f"   (It will use {OUTPUT_CSV} if available)")



if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        logging.warning("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\n❌ Error: {str(e)}", exc_info=True)
        sys.exit(1)
