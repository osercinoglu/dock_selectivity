#!/usr/bin/env python3
"""
Required Structures Identifier
Identifies which AlphaFold structures are actually needed based on small molecule drug-protein interactions.

This script analyzes the drug-protein interaction data and determines which UniProt IDs
and their corresponding AlphaFold structures are required for docking. It filters for
small molecule drugs only, as the docking workflow is optimized for small molecules.

The output includes both cavity information (from extract_cavities.py) and placeholder
paths for AlphaFold receptor structures that need to be extracted later.

Usage:
    python identify_required_structures.py

Input:
    - drug_to_protein.tsv: Drug-protein interaction data
    - small_molecule_drug_links.csv: Small molecule drugs from DrugBank
    - uniprot_gene_mapping.csv: UniProt to gene mapping
    - cavity_mapping.csv: Available cavity structures (pocket PDBs only)

Output:
    - required_structures.csv: List of required structures for small molecule docking
    - required_structures.log: Processing log
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path

# --- Configuration ---
DRUG_TO_PROTEIN_TSV = "/opt/data/multiscale_interactome_data/1_drug_to_protein.tsv"
SMALL_MOLECULE_DRUGS_CSV = "/opt/data/drugbank/small_molecule_drug_links.csv"  # Small molecule drugs only
UNIPROT_MAPPING_CSV = "uniprot_gene_mapping.csv"
CAVITY_MAPPING_CSV = "cavity_mapping.csv"
PROCESSED_LIGAND_SDF_FOLDER = "/home/onur/experiments/cavity_space_consensus_docking/drugbank_approved_split"
OUTPUT_CSV = "required_structures.csv"
LOG_FILE = "required_structures.log"

# Set up logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

def load_drug_protein_interactions():
    """Load drug-protein interaction data."""
    if not os.path.exists(DRUG_TO_PROTEIN_TSV):
        logging.error(f"Drug-protein TSV not found: {DRUG_TO_PROTEIN_TSV}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(DRUG_TO_PROTEIN_TSV, sep='\t')
        logging.info(f"Loaded {len(df)} drug-protein interactions")
        return df
    except Exception as e:
        logging.error(f"Error loading drug-protein data: {e}")
        sys.exit(1)

def load_small_molecule_drugs():
    """Load small molecule drugs list from DrugBank."""
    if not os.path.exists(SMALL_MOLECULE_DRUGS_CSV):
        logging.error(f"Small molecule drugs CSV not found: {SMALL_MOLECULE_DRUGS_CSV}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(SMALL_MOLECULE_DRUGS_CSV)
        small_molecule_ids = set(df['DrugBank ID'].unique())
        logging.info(f"Loaded {len(small_molecule_ids)} small molecule drug IDs")
        return small_molecule_ids
    except Exception as e:
        logging.error(f"Error loading small molecule drugs: {e}")
        sys.exit(1)

def load_uniprot_mapping():
    """Load UniProt to gene mapping."""
    if not os.path.exists(UNIPROT_MAPPING_CSV):
        logging.error(f"UniProt mapping not found: {UNIPROT_MAPPING_CSV}")
        logging.error("Run prepare_uniprot_mapping.py first")
        sys.exit(1)
    
    try:
        df = pd.read_csv(UNIPROT_MAPPING_CSV)
        logging.info(f"Loaded {len(df)} UniProt mappings")
        return df
    except Exception as e:
        logging.error(f"Error loading UniProt mapping: {e}")
        sys.exit(1)

def load_cavity_mapping():
    """Load cavity structure mapping."""
    if not os.path.exists(CAVITY_MAPPING_CSV):
        logging.error(f"Cavity mapping not found: {CAVITY_MAPPING_CSV}")
        logging.error("Run extract_cavities.py first")
        sys.exit(1)
    
    try:
        df = pd.read_csv(CAVITY_MAPPING_CSV)
        logging.info(f"Loaded {len(df)} cavity mappings")
        return df
    except Exception as e:
        logging.error(f"Error loading cavity mapping: {e}")
        sys.exit(1)

def check_ligand_availability(drugbank_ids):
    """Check which ligands are available as SDF files."""
    available_ligands = set()
    missing_ligands = set()
    
    for drugbank_id in drugbank_ids:
        sdf_path = os.path.join(PROCESSED_LIGAND_SDF_FOLDER, f"{drugbank_id}.sdf")
        if os.path.exists(sdf_path):
            available_ligands.add(drugbank_id)
        else:
            missing_ligands.add(drugbank_id)
    
    logging.info(f"Available ligands: {len(available_ligands)}/{len(drugbank_ids)}")
    if missing_ligands:
        logging.warning(f"Missing ligands: {len(missing_ligands)}")
    
    return available_ligands

def identify_required_structures():
    """Identify which structures are actually needed for docking."""
    logging.info("Starting required structures identification...")
    
    # Load all input data
    drug_protein_df = load_drug_protein_interactions()
    small_molecule_ids = load_small_molecule_drugs()
    uniprot_df = load_uniprot_mapping()
    cavity_df = load_cavity_mapping()
    
    # Filter interactions to only include small molecule drugs
    initial_interactions = len(drug_protein_df)
    drug_protein_df = drug_protein_df[
        drug_protein_df['node_1'].isin(small_molecule_ids)
    ]
    logging.info(f"Filtered to small molecules: {len(drug_protein_df)}/{initial_interactions} interactions")
    
    # Create mappings for efficient lookup
    gene_to_uniprot = dict(zip(uniprot_df['Mapped_Gene_Name'], uniprot_df['Entry']))
    
    # Get unique drug and gene combinations from interactions
    unique_drugs = drug_protein_df['node_1'].unique()
    unique_genes = drug_protein_df['node_2_name'].unique()
    
    logging.info(f"Unique small molecule drugs in interactions: {len(unique_drugs)}")
    logging.info(f"Unique genes in interactions: {len(unique_genes)}")
    
    # Check ligand availability
    available_drugs = check_ligand_availability(unique_drugs)
    
    # Filter interactions to only those with available ligands
    filtered_interactions = drug_protein_df[
        drug_protein_df['node_1'].isin(available_drugs)
    ]
    logging.info(f"Small molecule interactions with available ligands: {len(filtered_interactions)}")
    
    # Identify required UniProt IDs
    required_genes = filtered_interactions['node_2_name'].unique()
    required_uniprot_ids = set()
    missing_gene_mappings = set()
    
    for gene in required_genes:
        uniprot_id = gene_to_uniprot.get(gene)
        if uniprot_id:
            required_uniprot_ids.add(uniprot_id)
        else:
            missing_gene_mappings.add(gene)
    
    logging.info(f"Required UniProt IDs: {len(required_uniprot_ids)}")
    if missing_gene_mappings:
        logging.warning(f"Genes without UniProt mapping: {len(missing_gene_mappings)}")
    
    # Find available structures for required UniProt IDs
    available_structures = cavity_df[
        cavity_df['UniProt_ID'].isin(required_uniprot_ids)
    ].copy()
    
    logging.info(f"Available structures for required targets: {len(available_structures)}")
    
    # Note: Receptor_PDB paths will be generated by extract_alphafold_models.py after extraction
    # This avoids path inconsistencies between structure identification and actual extraction
    
    # Add priority information
    interaction_counts = filtered_interactions['node_2_name'].value_counts()
    gene_to_interaction_count = dict(interaction_counts)
    
    # Map back to UniProt IDs and add interaction counts
    available_structures['Gene_Name'] = available_structures['UniProt_ID'].map(
        {v: k for k, v in gene_to_uniprot.items()}
    )
    available_structures['Interaction_Count'] = available_structures['Gene_Name'].map(
        gene_to_interaction_count
    ).fillna(0)
    
    # Sort by interaction count (most interactions first) and fragment ID
    available_structures = available_structures.sort_values(
        ['Interaction_Count', 'UniProt_ID', 'Fragment_ID'], 
        ascending=[False, True, True]
    )
    
    # Save results (exclude Receptor_PDB column as it will be generated by extract_alphafold_models.py)
    columns_to_save = ['UniProt_ID', 'Fragment_ID', 'Cavity_Index', 'Pocket_PDB', 'Gene_Name', 'Interaction_Count']
    available_structures[columns_to_save].to_csv(OUTPUT_CSV, index=False)
    
    # Summary statistics
    unique_targets = len(available_structures['UniProt_ID'].unique())
    total_structures = len(available_structures)
    total_interactions = len(filtered_interactions)
    
    logging.info(f"\n--- Summary ---")
    logging.info(f"Total small molecule drug-protein interactions (with available ligands): {total_interactions}")
    logging.info(f"Unique target proteins with structures: {unique_targets}")
    logging.info(f"Total structures to process: {total_structures}")
    logging.info(f"Results saved to: {OUTPUT_CSV}")
    
    # Show top targets by interaction count
    top_targets = available_structures.groupby(['UniProt_ID', 'Gene_Name'])['Interaction_Count'].first().sort_values(ascending=False).head(10)
    logging.info(f"\nTop 10 targets by small molecule interaction count:")
    for (uniprot_id, gene_name), count in top_targets.items():
        logging.info(f"  {gene_name} ({uniprot_id}): {count} interactions")
    
    return available_structures


# ---------------------------------------------------------------------------
# Superseded -- FIX_PLAN 3.1
# ---------------------------------------------------------------------------

#: The stages that replace this one. Both resolve on ``node_2`` -- the Entrez
#: Gene ID that has been present and unused in 1_drug_to_protein.tsv all along --
#: through IdentityIndex, which guarantees a collapse factor of exactly 1.
REPLACEMENT_STAGES = ('resolve_interactions.py', 'plan_positives.py')


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
        "identify_required_structures.py resolves identity with dict(zip(Mapped_Gene_Name, Entry)), "
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
    """Main function."""
    logging.info("Starting required structures identification workflow")
    
    try:
        required_structures = identify_required_structures()
        logging.info("Required structures identification completed successfully")
        return 0
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
