#!/usr/bin/env python3
"""
Required Structures Identifier -- SUPERSEDED, DO NOT USE FOR NEW RUNS
=====================================================================

    ############################################################
    #  DEPRECATED.  This stage is replaced by                  #
    #      resolve_interactions.py  +  plan_positives.py       #
    #  It resolves drug targets through Mapped_Gene_Name, a    #
    #  column in which 57.73% of rows name a DIFFERENT         #
    #  protein (KNOWN_ISSUE_uniprot_gene_mapping.md).  It      #
    #  refuses to run unless --legacy-gene-resolution is       #
    #  given, and it is retained only so an old campaign can   #
    #  be reproduced bug-for-bug.                              #
    ############################################################

What it did: read the drug-protein interaction edges, keep the small-molecule
drugs, map each edge's gene symbol to a UniProt accession, and emit the cavity
rows of the accessions that a docking run would therefore need.

Why it is wrong: the symbol -> accession step is
``dict(zip(Mapped_Gene_Name, Entry))``, last-duplicate-wins over a corrupted
column. It collapses 4,585 accessions onto 1,795 symbols and the survivor of
each collapse became the docked receptor. The replacement stages resolve on the
Entrez key through IdentityIndex, whose collapse factor is exactly 1.

Every input path is now a flag with no machine-specific default: reproducing an
old run means stating that run's paths, not inheriting a dead workstation's
layout.

Usage:
    # refuses, exit code 2 -- this is the intended outcome
    python3 identify_required_structures.py

    # deliberate legacy reproduction
    python3 identify_required_structures.py --legacy-gene-resolution \\
        --edges /opt/data/multiscale_interactome_data/1_drug_to_protein.tsv \\
        --drugs /opt/data/drugbank/small_molecule_drug_links.csv \\
        --uniprot-mapping ./uniprot_gene_mapping.csv \\
        --cavity-mapping ./cavity_mapping.csv \\
        --ligands /opt/data/drugbank/drugbank_approved_split \\
        --out ./required_structures.csv \\
        --run-dir /path/to/campaign

Input:
    --edges            drug-protein interaction edges (TSV)
    --drugs            small-molecule DrugBank links (CSV)
    --uniprot-mapping  the corrupted UniProt/gene mapping (CSV)
    --cavity-mapping   extracted cavity structures (CSV, from extract_cavities.py)
    --ligands          directory of per-drug SDF files

Output:
    --out              required_structures.csv
    <out-dir>/identify_required_structures.log   (see --out-dir / --log-file)
"""

import argparse
import logging
import os
import sys

import pandas as pd

from dock_selectivity.runtime import add_runtime_arguments, resolve_out_dir, step

STEP_NAME = "identify_required_structures"

# --- Configuration ---
# Defaults only; every one of these is a flag. Reference datasets under
# /opt/data are machine-wide and may default; run-specific files may not.
DEFAULT_DRUG_TO_PROTEIN_TSV = "/opt/data/multiscale_interactome_data/1_drug_to_protein.tsv"
DEFAULT_SMALL_MOLECULE_DRUGS_CSV = "/opt/data/drugbank/small_molecule_drug_links.csv"
DEFAULT_UNIPROT_MAPPING_CSV = "uniprot_gene_mapping.csv"
DEFAULT_CAVITY_MAPPING_CSV = "cavity_mapping.csv"
DEFAULT_LIGAND_SDF_FOLDER = "/opt/data/drugbank/drugbank_approved_split"
DEFAULT_OUTPUT_CSV = "required_structures.csv"

#: Columns written to --out. Implementation detail of the downstream contract
#: (extract_alphafold_models.py adds Receptor_PDB itself), not configuration.
OUTPUT_COLUMNS = ['UniProt_ID', 'Fragment_ID', 'Cavity_Index', 'Pocket_PDB',
                  'Gene_Name', 'Interaction_Count']

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


def load_drug_protein_interactions(path):
    """Load drug-protein interaction data."""
    return _read_or_fail(path, "drug-protein edges (--edges)", sep='\t')


def load_small_molecule_drugs(path):
    """Load small molecule drugs list from DrugBank."""
    frame = _read_or_fail(path, "small-molecule drug links (--drugs)")
    return set(frame['DrugBank ID'].unique())


def load_uniprot_mapping(path):
    """Load the (corrupted) UniProt to gene mapping."""
    return _read_or_fail(path, "UniProt/gene mapping (--uniprot-mapping)")


def load_cavity_mapping(path):
    """Load cavity structure mapping."""
    return _read_or_fail(path, "cavity mapping (--cavity-mapping)")


def check_ligand_availability(drugbank_ids, ligand_folder):
    """Check which ligands are available as SDF files.

    An absent or empty ligand folder used to yield an empty result set and a
    zero exit code -- a run that "succeeded" with nothing in it. It is an error.
    """
    if not ligand_folder or not os.path.isdir(ligand_folder):
        raise MissingInput("ligand SDF folder not found: %r (--ligands)" % ligand_folder)
    if not any(name.endswith('.sdf') for name in os.listdir(ligand_folder)):
        raise MissingInput(
            "ligand SDF folder %r contains no .sdf files (--ligands)" % ligand_folder)

    available_ligands = set()
    missing_ligands = set()
    for drugbank_id in drugbank_ids:
        sdf_path = os.path.join(ligand_folder, "%s.sdf" % drugbank_id)
        if os.path.exists(sdf_path):
            available_ligands.add(drugbank_id)
        else:
            missing_ligands.add(drugbank_id)

    logger.info("Available ligands: %d/%d", len(available_ligands), len(drugbank_ids))
    if missing_ligands:
        logger.warning("Missing ligands: %d", len(missing_ligands))
    return available_ligands


def identify_required_structures(edges_path, drugs_path, uniprot_mapping_path,
                                 cavity_mapping_path, ligand_folder, output_csv):
    """Identify which structures are actually needed for docking."""
    logger.info("Starting required structures identification...")

    drug_protein_df = load_drug_protein_interactions(edges_path)
    small_molecule_ids = load_small_molecule_drugs(drugs_path)
    uniprot_df = load_uniprot_mapping(uniprot_mapping_path)
    cavity_df = load_cavity_mapping(cavity_mapping_path)

    # Filter interactions to only include small molecule drugs
    initial_interactions = len(drug_protein_df)
    drug_protein_df = drug_protein_df[drug_protein_df['node_1'].isin(small_molecule_ids)]
    logger.info("Filtered to small molecules: %d/%d interactions",
                len(drug_protein_df), initial_interactions)

    # THE DEFECT: last-duplicate-wins over a column that is 57.73% wrong.
    gene_to_uniprot = dict(zip(uniprot_df['Mapped_Gene_Name'], uniprot_df['Entry']))

    unique_drugs = drug_protein_df['node_1'].unique()
    unique_genes = drug_protein_df['node_2_name'].unique()
    logger.info("Unique small molecule drugs in interactions: %d", len(unique_drugs))
    logger.info("Unique genes in interactions: %d", len(unique_genes))

    available_drugs = check_ligand_availability(unique_drugs, ligand_folder)

    filtered_interactions = drug_protein_df[drug_protein_df['node_1'].isin(available_drugs)]
    logger.info("Small molecule interactions with available ligands: %d",
                len(filtered_interactions))

    required_genes = filtered_interactions['node_2_name'].unique()
    required_uniprot_ids = set()
    missing_gene_mappings = set()
    for gene in required_genes:
        uniprot_id = gene_to_uniprot.get(gene)
        if uniprot_id:
            required_uniprot_ids.add(uniprot_id)
        else:
            missing_gene_mappings.add(gene)

    logger.info("Required UniProt IDs: %d", len(required_uniprot_ids))
    if missing_gene_mappings:
        logger.warning("Genes without UniProt mapping: %d", len(missing_gene_mappings))

    available_structures = cavity_df[cavity_df['UniProt_ID'].isin(required_uniprot_ids)].copy()
    logger.info("Available structures for required targets: %d", len(available_structures))

    # Note: Receptor_PDB paths will be generated by extract_alphafold_models.py after
    # extraction. This avoids path inconsistencies between structure identification
    # and actual extraction.

    interaction_counts = filtered_interactions['node_2_name'].value_counts()
    gene_to_interaction_count = dict(interaction_counts)

    available_structures['Gene_Name'] = available_structures['UniProt_ID'].map(
        {v: k for k, v in gene_to_uniprot.items()}
    )
    available_structures['Interaction_Count'] = available_structures['Gene_Name'].map(
        gene_to_interaction_count
    ).fillna(0)

    available_structures = available_structures.sort_values(
        ['Interaction_Count', 'UniProt_ID', 'Fragment_ID'],
        ascending=[False, True, True]
    )

    out_parent = os.path.dirname(os.path.abspath(output_csv))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    available_structures[OUTPUT_COLUMNS].to_csv(output_csv, index=False)

    unique_targets = len(available_structures['UniProt_ID'].unique())
    logger.info("--- Summary ---")
    logger.info("Total small molecule drug-protein interactions (with available ligands): %d",
                len(filtered_interactions))
    logger.info("Unique target proteins with structures: %d", unique_targets)
    logger.info("Total structures to process: %d", len(available_structures))
    logger.info("Results saved to: %s", output_csv)

    top_targets = available_structures.groupby(
        ['UniProt_ID', 'Gene_Name'])['Interaction_Count'].first().sort_values(
        ascending=False).head(10)
    logger.info("Top 10 targets by small molecule interaction count:")
    for (uniprot_id, gene_name), count in top_targets.items():
        logger.info("  %s (%s): %s interactions", gene_name, uniprot_id, count)

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
        "identify_required_structures.py resolves identity with dict(zip(Mapped_Gene_Name, Entry)), "
        "which is last-duplicate-wins over a column that is 57.73% wrong. "
        "This stage is superseded by: "
        + ", ".join(REPLACEMENT_STAGES)
        + ". They resolve on the Entrez key via IdentityIndex (collapse factor 1). "
        "Pass --legacy-gene-resolution to run this anyway."
    )


def build_parser():
    """The CLI. Nothing here is required, so a bare run reaches the refusal."""
    parser = argparse.ArgumentParser(
        prog="identify_required_structures.py",
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
                        help='directory of per-drug SDF files; must exist and contain '
                             '.sdf files (default: %(default)s)')
    parser.add_argument('--out', default=DEFAULT_OUTPUT_CSV,
                        help='required-structures CSV to write (default: %(default)s)')
    parser.add_argument('--run-dir', default=None,
                        help='campaign run directory; the step log goes to '
                             '<run-dir>/steps/%s/ unless --out-dir says otherwise' % STEP_NAME)
    add_runtime_arguments(parser, STEP_NAME)
    return parser


def main(argv=None):
    """Refuse by default; run the legacy resolver only when told to, twice."""
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
            identify_required_structures(
                edges_path=args.edges,
                drugs_path=args.drugs,
                uniprot_mapping_path=args.uniprot_mapping,
                cavity_mapping_path=args.cavity_mapping,
                ligand_folder=args.ligands,
                output_csv=args.out,
            )
        except MissingInput as exc:
            ctx.logger.error("%s", exc)
            # SystemExit inside step() is reported as a failure, not a quiet return.
            raise SystemExit(2)
        ctx.logger.info("Required structures identification completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
