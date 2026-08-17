#!/usr/bin/env python3
"""
PocketMatch Similarity Calculator for Cavity Files (Parallelized)
=================================================================

Runs PocketMatch on the cavity PDB files extracted from AlphaFold structures and
computes pairwise pocket similarity scores (Pmin and Pmax).

Pockets are divided into blocks; each block becomes one cabbage file, and
Step3-PM_typeB compares block against block inside isolated temporary
workspaces, so an all-pairs comparison over tens of thousands of pockets stays
tractable and ``--resume`` can continue from the blocks and chunks already on
disk.

This is the optional pocket-similarity side branch: it is not part of
run_full_workflow.py's phase chain. Its output feeds pocketmatch_clustering.py,
whose pocket_clusters/ directory parse_prepare.py joins back onto the docking
results.

Every timeout in the branch is a flag: on slower storage the defaults below are
the first thing that needs changing, and editing the source to change them is
not a run configuration.

Usage:
    python3 get_PM_scores.py \\
        --cavities <run-dir>/extracted_cavities \\
        --output-dir <run-dir>/pocketmatch_results \\
        --pm-dir /home/onur/software/PocketMatch_v2.1/PocketMatch_v2.1 \\
        --af-structures <run-dir>/alphafold_structures \\
        --threads 8 --resume

    # a quick shape check that touches only the first few blocks
    python3 get_PM_scores.py --cavities ... --output-dir ... --pm-dir ... \\
        --test-mode --test-blocks 2

    The original single-dash spellings (-cavities, -output, -pm_dir,
    -af_structures, -threads) still work, so shell history keeps running.

Input:
    --cavities       directory of cavity PDB files in per-structure subdirectories
    --pm-dir         PocketMatch v2.1 installation (Step0-cabbage.sh, Step3-PM_typeB)
    --af-structures  AlphaFold models, used to restore complete residues

Output:
    <output-dir>/PocketMatch_similarities/similarities_part_*.csv
    <out-dir>/get_PM_scores.log   (see --out-dir / --log-file)
"""

import pandas as pd
import numpy as np
import subprocess
import os
import sys
import glob
import shutil
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import tempfile
from functools import partial
from tqdm import tqdm
import concurrent.futures
import json
import logging
import faulthandler
import signal
import re
from biopandas.pdb import PandasPdb

from dock_selectivity.runtime import add_runtime_arguments, resolve_out_dir, step

faulthandler.register(signal.SIGUSR1)

STEP_NAME = "get_PM_scores"

# Module-level logger. Configured by the shared runtime in main(); importing
# this module configures nothing.
logger = logging.getLogger(STEP_NAME)

# --- Configuration ---
# Defaults only; every one of these is a flag (see build_parser). They are
# module-level so the worker functions can read them without threading eight
# timeouts through six call layers; main() rebinds them from the command line
# before any work starts.
RESIDUE_TIMEOUT = 30          # --residue-timeout   : one cavity's residue extraction
CABBAGE_TIMEOUT = 300         # --cabbage-timeout   : one Step0-cabbage.sh run
PM_TIMEOUT = 600              # --pm-timeout        : one Step3-PM_typeB run
BLOCK_TIMEOUT = 1200          # --block-timeout     : one cabbage block, end to end
BLOCK_PHASE_TIMEOUT = 7200    # --block-phase-timeout      : all blocks together
COMPARISON_TIMEOUT = 300      # --comparison-timeout       : one block-vs-block pair
COMPARISON_PHASE_TIMEOUT = 14400  # --comparison-phase-timeout : all pairs together
IO_TIMEOUT = 300              # --io-timeout        : one background chunk save
DEFAULT_THREAD_CAP = 20       # --thread-cap        : ceiling when --threads is absent

#: PocketMatch's own cabbage limit. A property of the tool, not a run setting.
MAX_BLOCK_SIZE = 1000

#: Which settings main() may rebind, mapped to their argparse destination.
RUNTIME_TUNABLES = (
    ('RESIDUE_TIMEOUT', 'residue_timeout'),
    ('CABBAGE_TIMEOUT', 'cabbage_timeout'),
    ('PM_TIMEOUT', 'pm_timeout'),
    ('BLOCK_TIMEOUT', 'block_timeout'),
    ('BLOCK_PHASE_TIMEOUT', 'block_phase_timeout'),
    ('COMPARISON_TIMEOUT', 'comparison_timeout'),
    ('COMPARISON_PHASE_TIMEOUT', 'comparison_phase_timeout'),
    ('IO_TIMEOUT', 'io_timeout'),
    ('DEFAULT_THREAD_CAP', 'thread_cap'),
)


def apply_runtime_tunables(args):
    """Rebind the module tunables from parsed arguments; return what changed.

    The forked workers inherit these globals, so this must run before any pool
    is created -- main() calls it immediately after parsing.
    """
    changed = {}
    for constant, dest in RUNTIME_TUNABLES:
        value = getattr(args, dest, None)
        if value is None:
            continue
        if globals()[constant] != value:
            changed[constant] = value
        globals()[constant] = value
    return changed


def collect_cavity_files(cavities_dir):
    """
    Collect all cavity PDB files from subdirectories.
    Uses os.scandir() for much faster directory traversal on large directories.
    
    Args:
        cavities_dir: Path to extracted_cavities directory
        
    Returns:
        List of paths to cavity PDB files
    """
    import re
    
    logger.info(f"Collecting cavity PDB files from: {cavities_dir}")
    
    # Use os.scandir for fast directory iteration (much faster than glob for large dirs)
    # Pattern: AF-*_cavity_*.pdb (excluding vacant files)
    cavity_pattern = re.compile(r'^AF-.*_cavity_\d+\.pdb$')
    cavity_files = []
    
    try:
        # First level: iterate through subdirectories
        with os.scandir(cavities_dir) as entries:
            subdirs = [entry.path for entry in entries if entry.is_dir()]
        
        logger.info(f"Found {len(subdirs):,} subdirectories to scan")
        
        # Second level: iterate through files in each subdirectory
        for i, subdir in enumerate(subdirs):
            try:
                with os.scandir(subdir) as files:
                    for f in files:
                        if f.is_file() and cavity_pattern.match(f.name):
                            cavity_files.append(Path(f.path))
            except (PermissionError, OSError) as e:
                logger.warning(f"Could not scan {subdir}: {e}")
            
            # Progress update every 5000 directories
            if (i + 1) % 5000 == 0:
                logger.info(f"  Scanned {i + 1:,}/{len(subdirs):,} directories, found {len(cavity_files):,} cavity files so far...")
    
    except Exception as e:
        logger.error(f"Error scanning directory {cavities_dir}: {e}")
        raise
    
    logger.info(f"Found {len(cavity_files):,} cavity PDB files")
    
    return cavity_files


def create_isolated_pocketmatch_workspace(pm_base_dir, work_dir, workspace_id):
    """
    Create an isolated PocketMatch workspace by copying only essential files.
    Since workspaces are created sequentially, we can safely copy files without
    hitting "too many open files" errors.
    
    Args:
        pm_base_dir: Base PocketMatch installation directory
        work_dir: Working directory for this process
        workspace_id: Unique identifier for this workspace
        
    Returns:
        Path to the isolated PocketMatch directory
    """
    isolated_pm_dir = os.path.join(work_dir, f"pm_workspace_{workspace_id}")
    
    # Create workspace directory
    os.makedirs(isolated_pm_dir, exist_ok=True)
    
    # Copy executable file (Step3-PM_typeB is needed for comparisons)
    step3_typeB = os.path.join(pm_base_dir, "Step3-PM_typeB")
    if os.path.exists(step3_typeB):
        dest_step3 = os.path.join(isolated_pm_dir, "Step3-PM_typeB")
        if not os.path.exists(dest_step3):
            shutil.copy2(step3_typeB, dest_step3)
    
    # Create cabbage-file_maker directory structure
    cabbage_dir = os.path.join(isolated_pm_dir, "cabbage-file_maker")
    os.makedirs(cabbage_dir, exist_ok=True)
    
    # Copy essential cabbage-file_maker executables and scripts
    base_cabbage_dir = os.path.join(pm_base_dir, "cabbage-file_maker")
    for item in ["Step0-cabbage.sh", "Step0-cabbage_core", "Step0-END-FILE"]:
        src = os.path.join(base_cabbage_dir, item)
        if os.path.exists(src):
            dest = os.path.join(cabbage_dir, item)
            if not os.path.exists(dest):
                shutil.copy2(src, dest)
    
    # Create empty Sample_pockets directory (will be populated later)
    sample_pockets_dir = os.path.join(cabbage_dir, "Sample_pockets")
    os.makedirs(sample_pockets_dir, exist_ok=True)
    
    # Verify the structure was created successfully
    if not os.path.isdir(cabbage_dir):
        raise RuntimeError(f"Failed to create cabbage-file_maker directory: {cabbage_dir}")
    if not os.path.isdir(sample_pockets_dir):
        raise RuntimeError(f"Failed to create Sample_pockets directory: {sample_pockets_dir}")
    
    return isolated_pm_dir


def extract_complete_residues_from_cavity(args):
    """
    Extract complete residues from AlphaFold structure for a single cavity file.
    Worker function for parallel processing.
    
    Uses BioPandas to properly parse PDB files and ensure complete residue extraction.
    
    Args:
        args: Tuple of (idx, pocket_file, af_base_dir, dest_file)
        
    Returns:
        Tuple of (simple_name, original_name, success_status)
    """
    idx, pocket_file, af_base_dir, dest_file = args
    
    simple_name = f"p{idx+1}.pdb"
    original_name = os.path.basename(pocket_file)
    
    # If af_base_dir is None, we cannot extract complete residues
    # Just copy the cavity file as-is (may cause issues with PocketMatch)
    if af_base_dir is None:
        try:
            shutil.copy2(pocket_file, dest_file)
            return (simple_name, original_name, True)
        except Exception as e:
            logger.warning(f"Failed to copy {original_name}: {e}")
            return (simple_name, original_name, False)
    
    try:
        # Parse cavity file using BioPandas to get residue information
        cavity_ppdb = PandasPdb().read_pdb(str(pocket_file))
        cavity_atoms = cavity_ppdb.df['ATOM']
        
        if len(cavity_atoms) == 0:
            return (simple_name, original_name, False)
        
        # Get unique residues from cavity file (chain + residue number)
        cavity_residues = set(
            zip(cavity_atoms['chain_id'], cavity_atoms['residue_number'])
        )
        
        # Parse filename to extract protein ID and fragment
        # Format: AF-PROTEINID-FX-model_vY_cavity_N.pdb
        pocket_basename = os.path.basename(pocket_file)
        if not (pocket_basename.startswith('AF-') and '_cavity_' in pocket_basename):
            # Not an AlphaFold cavity file - skip it
            return (simple_name, original_name, False)
        
        parts = pocket_basename.split('-')
        if len(parts) < 4:
            return (simple_name, original_name, False)
        
        protein_id = parts[1]  # Extract PROTEINID
        fragment_id = parts[2]  # Extract FX (F1, F2, F3, etc.)
        
        # Find AlphaFold PDB file for the specific fragment
        af_dir = os.path.join(af_base_dir, protein_id, fragment_id)
        af_pdb_pattern = f"AF-{protein_id}-{fragment_id}-model_v*.pdb"
        af_pdb_files = list(Path(af_dir).glob(af_pdb_pattern))
        
        if not af_pdb_files:
            # AlphaFold structure not found - skip this cavity
            return (simple_name, original_name, False)
        
        af_pdb_path = af_pdb_files[0]
        
        # Load AlphaFold structure using BioPandas
        af_ppdb = PandasPdb().read_pdb(str(af_pdb_path))
        af_atoms = af_ppdb.df['ATOM']
        
        if len(af_atoms) == 0:
            logger.debug(f"No ATOM records in AlphaFold structure for {original_name}")
            return (simple_name, original_name, False)
        
        # CRITICAL VALIDATION: Check that ALL cavity residues have CA atoms in AlphaFold structure
        # PocketMatch Step0-cabbage_core allocates memory based on CA count
        # Missing CA atoms cause heap corruption and malloc crashes
        
        # Get residues with CA atoms in AlphaFold structure
        ca_atoms = af_atoms[af_atoms['atom_name'] == 'CA']
        af_residues_with_ca = set(
            zip(ca_atoms['chain_id'], ca_atoms['residue_number'])
        )
        
        # Find which cavity residues have CA atoms in AlphaFold
        valid_residues = cavity_residues & af_residues_with_ca
        
        if len(valid_residues) == 0:
            # No valid residues with CA atoms - this shouldn't happen with AlphaFold
            # but could happen if cavity file is corrupted
            return (simple_name, original_name, False)
        
        if len(valid_residues) < len(cavity_residues):
            # Some cavity residues don't have CA atoms in AlphaFold structure
            # This indicates data corruption - skip this file
            missing = cavity_residues - valid_residues
            logger.warning(f"{original_name} has residues without CA atoms: {missing}")
            return (simple_name, original_name, False)
        
        # Extract all atoms for valid residues from AlphaFold structure
        mask = af_atoms.apply(
            lambda row: (row['chain_id'], row['residue_number']) in valid_residues,
            axis=1
        )
        extracted_atoms = af_atoms[mask].copy()
        
        if len(extracted_atoms) == 0:
            return (simple_name, original_name, False)
        
        # Write extracted atoms to destination file
        # Create a new PandasPdb object and write properly formatted PDB
        output_ppdb = PandasPdb()
        output_ppdb.df['ATOM'] = extracted_atoms
        
        # Write to file with proper PDB formatting (80 character lines)
        output_ppdb.to_pdb(path=dest_file, records=['ATOM'], gz=False)
        
        return (simple_name, original_name, True)
        
    except Exception as e:
        logger.warning(f"Failed to process {original_name}: {e}")
        import traceback
        traceback.print_exc()
        return (simple_name, original_name, False)


def generate_cabbage_file(pm_dir, pocket_files, output_name="outfile.cabbage", af_base_dir=None, show_progress=False, n_threads=None):
    """
    Generate a cabbage file for a set of pocket files.
    
    Args:
        pm_dir: PocketMatch directory
        pocket_files: List of pocket PDB file paths
        output_name: Name for the cabbage output file
        af_base_dir: Base directory containing AlphaFold structures (required for extracting complete residues)
        show_progress: Whether to show progress bar
        n_threads: Number of threads for parallel residue extraction (default: CPU count)
        
    Returns:
        Path to generated cabbage file, dict mapping simplified names to original names
    """
    # Verify workspace structure exists
    cabbage_dir = os.path.join(pm_dir, "cabbage-file_maker")
    if not os.path.exists(cabbage_dir):
        raise RuntimeError(f"cabbage-file_maker directory does not exist: {cabbage_dir}. Workspace may not be properly initialized.")
    
    sample_pockets_dir = os.path.join(cabbage_dir, "Sample_pockets")
    
    # Clean and prepare Sample_pockets directory
    if os.path.exists(sample_pockets_dir):
        for file in Path(sample_pockets_dir).glob("*.pdb"):
            file.unlink()
    else:
        os.makedirs(sample_pockets_dir)
    
    # Extract COMPLETE RESIDUES from AlphaFold structures in parallel
    # Use simple numeric names to avoid issues with long names or special characters
    # If af_base_dir is None, just copy cavity files directly (faster for resume mode)
    name_mapping = {}
    
    # Prepare arguments for parallel processing
    if n_threads is None:
        n_threads = cpu_count()
    
    # Use the provided n_threads value directly
    # When called from parallel block processing, this will be 1 (sequential)
    max_workers = n_threads
    
    args_list = [
        (idx, pocket_file, af_base_dir, os.path.join(sample_pockets_dir, f"p{idx+1}.pdb"))
        for idx, pocket_file in enumerate(pocket_files)
    ]
    
    # Process residue extraction in parallel with timeout
    # Use ProcessPoolExecutor for true parallelism (bypasses Python GIL)
    # If af_base_dir is None, this just copies files (WARNING: may not work with PocketMatch)
    results = []
    if af_base_dir is None:
        logger.warning(f"No AlphaFold base directory provided - copying cavity files as-is (may cause PocketMatch errors)")
    else:
        logger.debug(f"Extracting complete residues from {len(pocket_files)} cavity files using {max_workers} workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and track futures
        future_to_idx = {executor.submit(extract_complete_residues_from_cavity, args): i for i, args in enumerate(args_list)}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            try:
                result = future.result(timeout=RESIDUE_TIMEOUT)
                results.append(result)
                idx = future_to_idx[future]
                logger.debug(f"Extracted residues from file {idx+1}/{len(pocket_files)}")
            except concurrent.futures.TimeoutError:
                idx = future_to_idx[future]
                logger.error(f"Timeout processing file index {idx}")
                results.append((f"p{idx+1}.pdb", "TIMEOUT", False))
            except Exception as e:
                idx = future_to_idx[future]
                logger.error(f"Error processing file index {idx}: {e}")
                results.append((f"p{idx+1}.pdb", "ERROR", False))
    
    # Build name mapping from results
    for simple_name, original_name, _ in results:
        name_mapping[simple_name] = original_name
    
    # Generate cabbage file using Step0-cabbage.sh
    cabbage_dir = os.path.join(pm_dir, "cabbage-file_maker")
    
    # Verify outfile.cabbage does not exist (should have been cleaned in create_isolated_pocketmatch_workspace)
    outfile_path = os.path.join(cabbage_dir, "outfile.cabbage")
    if os.path.exists(outfile_path):
        raise RuntimeError(f"outfile.cabbage still exists at {outfile_path} - cleanup failed")
    
    # Run Step0-cabbage.sh with Sample_pockets directory as argument
    sample_pockets_dir = os.path.join(cabbage_dir, "Sample_pockets")
    
    if show_progress:
        logger.info("Generating cabbage file...")
    
    logger.debug(f"Running Step0-cabbage.sh with Sample_pockets: {sample_pockets_dir}")
    cmd = ["./Step0-cabbage.sh", sample_pockets_dir]
    # Use cwd parameter instead of os.chdir() to avoid threading issues
    # Add timeout to prevent hanging (300s = 5 min should be enough for cabbage generation)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cabbage_dir, timeout=CABBAGE_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Cabbage generation timed out after {CABBAGE_TIMEOUT}s for {output_name}")
    
    if result.returncode != 0:
        raise RuntimeError(f"Cabbage generation failed: {result.stderr}")
    
    if not os.path.exists(outfile_path):
        raise FileNotFoundError(f"Cabbage file not created: {outfile_path}")
    
    # Rename if needed
    if output_name != "outfile.cabbage":
        new_path = os.path.join(cabbage_dir, output_name)
        shutil.move(outfile_path, new_path)
        cabbage_file = new_path
    else:
        cabbage_file = outfile_path
    
    return cabbage_file, name_mapping


def run_pocketmatch_typeB(pm_dir, cabbage_chunk, cabbage_full, timeout=None):
    """
    Run PocketMatch typeB comparison (chunk vs full set).
    
    Args:
        pm_dir: PocketMatch directory
        cabbage_chunk: Path to cabbage file for chunk
        cabbage_full: Path to cabbage file for full set
        timeout: Maximum seconds to wait for comparison (default: --pm-timeout)

    Returns:
        Path to output file
    """
    if timeout is None:
        timeout = PM_TIMEOUT

    # Copy cabbage files to PM directory if not already there
    chunk_local = os.path.join(pm_dir, os.path.basename(cabbage_chunk))
    full_local = os.path.join(pm_dir, os.path.basename(cabbage_full))
    
    if cabbage_chunk != chunk_local:
        shutil.copy2(cabbage_chunk, chunk_local)
    if cabbage_full != full_local:
        shutil.copy2(cabbage_full, full_local)
    
    # Run PocketMatch typeB with timeout to prevent hanging
    cmd = ["./Step3-PM_typeB", os.path.basename(chunk_local), os.path.basename(full_local)]
    # Use cwd parameter instead of os.chdir() to avoid threading issues
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=pm_dir, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"PocketMatch typeB timed out after {timeout}s for {os.path.basename(chunk_local)} vs {os.path.basename(full_local)}")
    
    if result.returncode != 0:
        raise RuntimeError(f"PocketMatch typeB failed with code {result.returncode}. stderr: {result.stderr}, stdout: {result.stdout}")
    
    output_file = os.path.join(pm_dir, "PocketMatch_score.txt")
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"PocketMatch output not created: {output_file}")
    
    return output_file


def process_chunk(chunk_id, chunk_files, all_files, pm_base_dir, work_dir, af_base_dir=None, debug=False):
    """
    Process a single chunk of pocket files in parallel.
    
    Args:
        chunk_id: Unique identifier for this chunk
        chunk_files: List of pocket files in this chunk
        all_files: List of all pocket files
        pm_base_dir: Base PocketMatch directory
        work_dir: Working directory for temporary files
        af_base_dir: Base directory for AlphaFold structures
        debug: Whether to keep temporary files
        
    Returns:
        DataFrame with similarity results for this chunk
    """
    try:
        logger.info(f"Worker {chunk_id}: Processing {len(chunk_files)} pockets...")
        
        # Create isolated workspace
        isolated_pm = create_isolated_pocketmatch_workspace(pm_base_dir, work_dir, chunk_id)
        
        # Generate cabbage file for chunk
        cabbage_chunk, chunk_mapping = generate_cabbage_file(isolated_pm, chunk_files, f"chunk_{chunk_id}.cabbage", af_base_dir)
        
        # Generate cabbage file for full set (reuse if exists)
        cabbage_full_path = os.path.join(work_dir, "full_set.cabbage")
        mapping_path = os.path.join(work_dir, "name_mapping.json")
        
        if not os.path.exists(cabbage_full_path):
            # Only generate once (first worker to need it)
            cabbage_full, full_mapping = generate_cabbage_file(isolated_pm, all_files, "full_set.cabbage", af_base_dir)
            shutil.copy2(cabbage_full, cabbage_full_path)
            # Save mapping for reuse
            import json
            with open(mapping_path, 'w') as f:
                json.dump(full_mapping, f)
        else:
            # Copy existing full set cabbage and load mapping
            cabbage_full_dest = os.path.join(isolated_pm, "cabbage-file_maker", "full_set.cabbage")
            shutil.copy2(cabbage_full_path, cabbage_full_dest)
            cabbage_full = cabbage_full_dest
            import json
            with open(mapping_path, 'r') as f:
                full_mapping = json.load(f)
        
        # Run PocketMatch typeB
        output_file = run_pocketmatch_typeB(isolated_pm, cabbage_chunk, cabbage_full)
        
        # Parse results with name mapping
        df_chunk = parse_pocketmatch_output(output_file, name_mapping2=full_mapping)
        
        logger.info(f"Worker {chunk_id}: Completed - {len(df_chunk)} comparisons")
        
        # Cleanup isolated workspace
        if not debug:
            shutil.rmtree(isolated_pm, ignore_errors=True)
        
        return df_chunk
        
    except Exception as e:
        logger.error(f"Worker {chunk_id}: Error - {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def build_cabbage_block(block_id, pocket_files, pm_base_dir, work_dir, af_base_dir, max_internal_threads=1, show_progress=False, verbose=False, isolated_pm=None):
    """
    Build a cabbage file for a block (<=100 pockets) in an isolated workspace.
    Returns (cabbage_path, name_mapping, tmp_workspace_path).
    The workspace is cleaned by the caller after consuming outputs if needed.
    """
    logger.info(f"Block {block_id}: Building cabbage with {len(pocket_files)} pockets")
    if verbose:
        logger.debug(f"Block {block_id}: Starting build with {len(pocket_files)} pockets")
    # Use pre-created workspace if provided, otherwise create new one
    if isolated_pm is None:
        isolated_pm = create_isolated_pocketmatch_workspace(pm_base_dir, work_dir, f"blk_{block_id}")
    try:
        cabbage_path, name_mapping = generate_cabbage_file(
            isolated_pm,
            pocket_files,
            output_name=f"block_{block_id}.cabbage",
            af_base_dir=af_base_dir,
            show_progress=show_progress,
            n_threads=max_internal_threads,
        )
        # Move outputs to a shared blocks dir for later comparisons
        blocks_dir = os.path.join(work_dir, "cabbage_blocks")
        os.makedirs(blocks_dir, exist_ok=True)
        final_cabbage = os.path.join(blocks_dir, f"block_{block_id}.cabbage")
        shutil.copy2(cabbage_path, final_cabbage)
        # Persist mapping
        mapping_path = os.path.join(blocks_dir, f"block_{block_id}.mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(name_mapping, f)
        logger.info(f"Block {block_id}: Finished -> {final_cabbage}")
        if verbose:
            logger.debug(f"Block {block_id}: Completed successfully")
        return final_cabbage, name_mapping, isolated_pm
    except Exception as e:
        # Don't delete workspace here - will be cleaned up by caller after all blocks finish
        # Deleting during parallel execution causes race conditions where other blocks
        # try to access files that have been deleted
        logger.error(f"Block {block_id}: Error during build - {e}")
        raise


def run_block_comparison(pair_id, cabbage_a, cabbage_b, mapping_a, mapping_b, pm_base_dir, work_dir, debug=False, verbose=False, isolated_pm=None):
    """
    Run Step3-PM_typeB for two cabbage blocks and parse with separate mappings.
    Returns DataFrame of results for this pair.
    """
    try:
        logger.info(f"Pair {pair_id}: Starting comparison {os.path.basename(cabbage_a)} vs {os.path.basename(cabbage_b)}")
        if verbose:
            logger.debug(f"Pair {pair_id}: Starting comparison")
        # Use pre-created workspace if provided, otherwise create new one
        if isolated_pm is None:
            isolated_pm = create_isolated_pocketmatch_workspace(pm_base_dir, work_dir, f"cmp_{pair_id}")
        
        # Copy cabbage files for this comparison into workspace
        local_a = os.path.join(isolated_pm, os.path.basename(cabbage_a))
        local_b = os.path.join(isolated_pm, os.path.basename(cabbage_b))
        shutil.copy2(cabbage_a, local_a)
        shutil.copy2(cabbage_b, local_b)

        # Run comparison
        output_file = run_pocketmatch_typeB(isolated_pm, local_a, local_b)

        # Parse using separate mappings: first set uses mapping_a, second uses mapping_b
        df = parse_pocketmatch_output(output_file, name_mapping1=mapping_a, name_mapping2=mapping_b)
        
        # Optionally save raw results to a separate folder (only if debug mode)
        if debug:
            results_dir = os.path.join(work_dir, "raw_results")
            os.makedirs(results_dir, exist_ok=True)
            result_file = os.path.join(results_dir, f"result_{pair_id}.txt")
            shutil.copy2(output_file, result_file)
        
        # Report completion to log
        completion_msg = f"Pair {pair_id}: Completed with {len(df)} comparisons"
        logger.info(completion_msg)

        # IMMEDIATELY clean up workspace (don't wait for finally block)
        # This prevents disk from getting clogged with thousands of workspaces
        if isolated_pm and not debug:
            try:
                shutil.rmtree(isolated_pm, ignore_errors=False)
                logger.debug(f"Pair {pair_id}: Cleaned up workspace")
            except Exception as cleanup_error:
                logger.warning(f"Pair {pair_id}: Failed to cleanup workspace: {cleanup_error}")
        
        return df
        
    except Exception as e:
        logger.error(f"Pair {pair_id}: Error - {e}")
        import traceback
        import sys
        # Log full traceback to file
        tb_str = ''.join(traceback.format_exception(*sys.exc_info()))
        logger.error(f"Pair {pair_id}: Traceback:\n{tb_str}")
        return pd.DataFrame()
    finally:
        # Final cleanup attempt (in case immediate cleanup above failed)
        if isolated_pm and not debug:
            if os.path.exists(isolated_pm):
                try:
                    shutil.rmtree(isolated_pm, ignore_errors=True)
                except:
                    pass  # Silent fail in finally block


def parse_previous_run_config(output_dir):
    """
    Parse the previous run's log file to extract exact configuration parameters.
    
    Args:
        output_dir: Output directory containing pocketmatch_progress.log
        
    Returns:
        Dict with config parameters or None if log not found
    """
    log_path = os.path.join(output_dir, 'pocketmatch_progress.log')
    
    if not os.path.exists(log_path):
        logger.warning(f"No previous log file found at {log_path}")
        return None
    
    config = {
        'num_cavities': None,
        'num_blocks': None,
        'block_size': None,
        'test_mode': False,
        'test_blocks': None,
        'total_pairs': None,
        'cavities_dir': None
    }
    
    logger.info(f"Parsing previous run configuration from {log_path}")
    
    with open(log_path, 'r') as f:
        for line in f:
            # Extract cavity count: "Found 38,709 cavity PDB files"
            if 'Found' in line and 'cavity PDB files' in line:
                match = re.search(r'Found ([0-9,]+) cavity PDB files', line)
                if match:
                    config['num_cavities'] = int(match.group(1).replace(',', ''))
            
            # Extract block info: "Preparing 388 cabbage blocks (size <= 100)"
            elif 'Preparing' in line and 'cabbage blocks' in line:
                match = re.search(r'Preparing (\d+) cabbage blocks \(size <= (\d+)\)', line)
                if match:
                    config['num_blocks'] = int(match.group(1))
                    config['block_size'] = int(match.group(2))
            
            # Extract test mode: "TEST MODE: Processing only first 3 blocks"
            elif 'TEST MODE:' in line:
                config['test_mode'] = True
                match = re.search(r'Processing only first (\d+) blocks', line)
                if match:
                    config['test_blocks'] = int(match.group(1))
            
            # Extract total pairs: "Submitting 75,466 block comparisons"
            elif 'Submitting' in line and 'block comparisons' in line:
                match = re.search(r'Submitting ([0-9,]+) block comparisons', line)
                if match:
                    config['total_pairs'] = int(match.group(1).replace(',', ''))
            
            # Extract cavities directory: "Collecting cavity PDB files from: /path/to/cavities"
            elif 'Collecting cavity PDB files from:' in line:
                match = re.search(r'Collecting cavity PDB files from: (.+)$', line)
                if match:
                    config['cavities_dir'] = match.group(1).strip()
    
    # Validate we got essential info
    if config['num_blocks'] is None or config['block_size'] is None:
        logger.error("Could not extract essential config from log file")
        return None
    
    logger.info(f"Previous run config: {config['num_blocks']} blocks (size <= {config['block_size']})")
    if config['test_mode']:
        logger.info(f"Previous run was in TEST MODE with {config['test_blocks']} blocks")
    
    return config


def reconstruct_pocket_to_block_mapping(cavity_files, block_size):
    """
    Reconstruct which block each pocket belongs to based on sorted order and block size.
    This recreates the exact same partitioning as the original run.
    
    Args:
        cavity_files: Sorted list of cavity file paths (same order as original run)
        block_size: Number of pockets per block
        
    Returns:
        Dict mapping pocket filename -> block_index
    """
    pocket_to_block = {}
    
    for idx, cavity_file in enumerate(cavity_files):
        block_idx = idx // block_size
        pocket_name = os.path.basename(cavity_file)
        pocket_to_block[pocket_name] = block_idx
    
    return pocket_to_block


def identify_completed_block_pairs(output_dir, pocket_to_block_map, n_threads=None):
    """
    Identify which block pairs have already been compared by analyzing result chunks in parallel.
    
    Args:
        output_dir: Output directory containing result_chunks folder
        pocket_to_block_map: Dict mapping pocket filename -> block_index
        n_threads: Number of parallel threads (default: CPU count)
        
    Returns:
        Tuple of (completed_pairs set, pockets_with_results set)
        - completed_pairs: Set of pair IDs like "0_1", "2_3"
        - pockets_with_results: Set of pocket filenames that appear in results
    """
    chunk_dir = os.path.join(output_dir, 'result_chunks')
    completed_pairs = set()
    pockets_with_results = set()
    
    if not os.path.exists(chunk_dir):
        logger.info("No result_chunks directory found - no completed pairs")
        return completed_pairs, pockets_with_results
    
    # Read all existing chunk files to find completed comparisons
    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, 'chunk_*.csv')))
    
    if not chunk_files:
        logger.info("No chunk files found - no completed pairs")
        return completed_pairs, pockets_with_results
    
    logger.info(f"Analyzing {len(chunk_files)} existing chunk files to identify completed block pairs...")
    
    if n_threads is None:
        n_threads = cpu_count()
    
    def process_chunk_file(chunk_file):
        """Process a single chunk file and return local results using vectorized operations."""
        local_pairs = {}  # (i, j) -> count
        local_pockets = set()
        
        try:
            df = pd.read_csv(chunk_file)
            if len(df) == 0 or 'Pocket1' not in df.columns or 'Pocket2' not in df.columns:
                return local_pairs, local_pockets
            
            # Vectorized operations - 100-1000x faster than iterrows()
            pocket1_series = df['Pocket1']
            pocket2_series = df['Pocket2']
            
            # Track pockets that have results (vectorized)
            local_pockets.update(pocket1_series.unique())
            local_pockets.update(pocket2_series.unique())
            
            # Map pockets to blocks (vectorized)
            block1_series = pocket1_series.map(pocket_to_block_map)
            block2_series = pocket2_series.map(pocket_to_block_map)
            
            # Filter out rows where mapping failed
            valid_mask = block1_series.notna() & block2_series.notna()
            if not valid_mask.any():
                return local_pairs, local_pockets
            
            block1_valid = block1_series[valid_mask].astype(int).values
            block2_valid = block2_series[valid_mask].astype(int).values
            
            # Create normalized pairs (i <= j) - vectorized
            block_min = np.minimum(block1_valid, block2_valid)
            block_max = np.maximum(block1_valid, block2_valid)
            
            # Create DataFrame for grouping
            pairs_df = pd.DataFrame({'i': block_min, 'j': block_max})
            
            # Count occurrences of each pair
            pair_counts = pairs_df.groupby(['i', 'j']).size()
            
            # Convert to dict
            for (i, j), count in pair_counts.items():
                local_pairs[(i, j)] = count
                    
        except Exception as e:
            logger.warning(f"Error reading {os.path.basename(chunk_file)}: {e}")
        
        return local_pairs, local_pockets
    
    # Process chunk files in parallel
    block_pair_counts = {}  # (i, j) -> count of comparisons
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(process_chunk_file, cf) for cf in chunk_files]
        
        # Collect results with progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="   Analyzing chunks"):
            try:
                local_pairs, local_pockets = future.result()
                
                # Merge local results into global counts
                for pair_key, count in local_pairs.items():
                    block_pair_counts[pair_key] = block_pair_counts.get(pair_key, 0) + count
                
                pockets_with_results.update(local_pockets)
            except Exception as e:
                logger.warning(f"Error processing chunk future: {e}")
    
    logger.info(f"Found {len(pockets_with_results):,} unique pockets in results")
    logger.info(f"Found results for {len(block_pair_counts)} block pairs")
    
    # Mark any pair with results as "completed"
    for pair_key, count in block_pair_counts.items():
        i, j = pair_key
        pair_id = f"{i}_{j}"
        completed_pairs.add(pair_id)
        logger.debug(f"Block pair {pair_id}: {count} comparisons found")
    
    logger.info(f"Identified {len(completed_pairs)} completed block pairs")
    return completed_pairs, pockets_with_results


def identify_blocks_to_build(total_blocks, completed_pairs):
    """
    Identify which blocks need to be built based on remaining comparisons.
    
    Args:
        total_blocks: Total number of blocks
        completed_pairs: Set of completed pair IDs (e.g., "0_1")
        
    Returns:
        Set of block indices that need to be built
    """
    needed_blocks = set()
    
    # Check all possible pairs in upper triangle
    for i in range(total_blocks):
        for j in range(i, total_blocks):
            pair_id = f"{i}_{j}"
            if pair_id not in completed_pairs:
                # This pair needs to be compared, so we need both blocks
                needed_blocks.add(i)
                needed_blocks.add(j)
    
    return needed_blocks


def get_next_chunk_counter(output_dir):
    """
    Determine the next chunk counter to use based on existing chunks.
    
    Args:
        output_dir: Output directory containing result_chunks folder
        
    Returns:
        Next chunk counter to use
    """
    chunk_dir = os.path.join(output_dir, 'result_chunks')
    
    if not os.path.exists(chunk_dir):
        return 0
    
    chunk_files = glob.glob(os.path.join(chunk_dir, 'chunk_*.csv'))
    
    if not chunk_files:
        return 0
    
    # Extract counter from each filename
    counters = []
    for chunk_file in chunk_files:
        basename = os.path.basename(chunk_file)
        # Extract number from chunk_XXXX.csv
        match = re.match(r'chunk_(\d+)\.csv', basename)
        if match:
            counters.append(int(match.group(1)))
    
    if counters:
        next_counter = max(counters) + 1
        logger.info(f"Resuming chunk numbering from {next_counter}")
        return next_counter
    
    return 0


def validate_resume_parameters(args, prev_config):
    """
    Validate that current parameters match previous run for safe resumption.
    
    Args:
        args: Current command-line arguments
        prev_config: Previous run configuration from log
        
    Returns:
        True if parameters are compatible, False otherwise
    """
    errors = []
    
    # Block size must match
    if prev_config['block_size'] != args.block_size:
        errors.append(f"Block size mismatch: previous={prev_config['block_size']}, current={args.block_size}")
    
    # Test mode consistency
    if prev_config['test_mode'] != args.test_mode:
        if prev_config['test_mode']:
            errors.append(f"Cannot resume from test mode run in full mode")
        else:
            errors.append(f"Cannot resume from full run in test mode")
    
    # If both test mode, test_blocks must match
    if prev_config['test_mode'] and args.test_mode:
        if prev_config['test_blocks'] != args.test_blocks:
            errors.append(f"Test blocks mismatch: previous={prev_config['test_blocks']}, current={args.test_blocks}")
    
    # Cavities directory should match (if we have it)
    if prev_config['cavities_dir'] and prev_config['cavities_dir'] != args.cavities:
        logger.warning(f"Cavities directory changed: previous={prev_config['cavities_dir']}, current={args.cavities}")
        logger.warning("This may cause issues if cavity files have changed")
    
    if errors:
        logger.error("Resume parameter validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("✅ Resume parameters validated successfully")
    return True


def parse_pocketmatch_output(output_file, name_mapping=None, name_mapping1=None, name_mapping2=None):
    """
    Parse PocketMatch output file into a DataFrame.
    
    Args:
        output_file: Path to PocketMatch_score.txt
        name_mapping: Dict mapping simplified names (p1.pdb) to original names
        
    Returns:
        DataFrame with columns: Pocket1, Pocket2, Pmin, Pmax
    """
    valid_lines = []
    total_lines = 0
    
    with open(output_file, 'r') as f:
        for line in f:
            total_lines += 1
            try:
                fields = line.strip().split()
                if len(fields) >= 4:
                    valid_lines.append(fields[:4])
            except:
                pass
    
    if not valid_lines:
        return pd.DataFrame(columns=['Pocket1', 'Pocket2', 'Pmin', 'Pmax'])
    
    # Create DataFrame
    df = pd.DataFrame(valid_lines, columns=['Pocket1', 'Pocket2', 'Pmin', 'Pmax'])
    
    # Clean up data
    df = df[df['Pmax'] != 'NULL____']
    df = df[df['Pmin'] != 'NULL____']
    
    # Convert to numeric
    df['Pmin'] = pd.to_numeric(df['Pmin'], errors='coerce')
    df['Pmax'] = pd.to_numeric(df['Pmax'], errors='coerce')
    
    # Remove rows with invalid scores
    df = df.dropna(subset=['Pmin', 'Pmax'])
    
    # Remove trailing underscores from pocket names
    df['Pocket1'] = df['Pocket1'].str.rstrip('_')
    df['Pocket2'] = df['Pocket2'].str.rstrip('_')
    
    # Map simplified names back to original names if mapping provided
    # Backward-compat: if name_mapping is provided, use it for both sides
    if name_mapping and not (name_mapping1 or name_mapping2):
        name_mapping1 = name_mapping
        name_mapping2 = name_mapping

    def make_mapper(mapping):
        if not mapping:
            return lambda x: x
        def map_name(x):
            if x in mapping:
                return mapping[x]
            if (x + '.pdb') in mapping:
                return mapping[x + '.pdb']
            if x.endswith('.pdb'):
                base = x[:-4]
                if base in mapping:
                    return mapping[base]
            return x
        return map_name

    df['Pocket1'] = df['Pocket1'].map(make_mapper(name_mapping1))
    df['Pocket2'] = df['Pocket2'].map(make_mapper(name_mapping2))
    
    return df


def get_pm_similarities_parallel(cavities_dir, output_dir, pm_base_dir, af_base_dir=None, n_threads=None, debug=False, verbose=False, test_mode=False, test_blocks=3, chunk_rows=100000, block_size=100, resume=False):
    """
    Main function to compute PocketMatch similarities with parallelization.
    
    Args:
        cavities_dir: Directory containing extracted cavity files
        output_dir: Directory to save output CSV
        pm_base_dir: Base PocketMatch installation directory
        af_base_dir: Base directory for AlphaFold structures (optional, for complete residue extraction)
        n_threads: Number of parallel threads (default: CPU count)
        debug: Whether to keep temporary files
        verbose: Whether to print detailed progress for each block/comparison
        test_mode: Whether to run in test mode (limited blocks)
        test_blocks: Number of blocks to process in test mode (default: 3)
        chunk_rows: Number of rows per chunk file (default: 100000)
        block_size: Number of pockets per block (default: 100, max: 1000)
        resume: Whether to resume from existing progress (default: False)
        
    Returns:
        DataFrame with pocket similarity scores
    """
    logger.info("="*80)
    logger.info("POCKETMATCH SIMILARITY ANALYSIS (PARALLEL)")
    logger.info("="*80)
    
    # Validate paths
    if not os.path.exists(cavities_dir):
        raise FileNotFoundError(f"Cavities directory not found: {cavities_dir}")
    
    if not os.path.exists(pm_base_dir):
        raise FileNotFoundError(f"PocketMatch directory not found: {pm_base_dir}")
    
    # Create output directory FIRST
    os.makedirs(output_dir, exist_ok=True)
    
    # Logging belongs to the caller: main() configures it through the shared
    # runtime before this function runs, so a library caller keeps its own.
    logger.info("Starting PocketMatch analysis")
    if resume:
        logger.info("RESUME MODE: continuing from existing blocks and chunks")

    # Determine number of threads (a conservative cap instead of all CPUs to avoid
    # resource exhaustion; raise it with --thread-cap)
    if n_threads is None:
        n_threads = min(DEFAULT_THREAD_CAP, cpu_count())
    
    logger.info(f"Configuration:")
    logger.info(f"Parallel threads: {n_threads}")
    
    # Collect cavity files
    cavity_files = collect_cavity_files(cavities_dir)
    
    if len(cavity_files) == 0:
        raise ValueError("No cavity PDB files found!")
    
    # Validate block size
    if block_size < 1 or block_size > MAX_BLOCK_SIZE:
        raise ValueError(f"Block size must be between 1 and {MAX_BLOCK_SIZE}, got {block_size}")
    
    logger.info(f"Using block size: {block_size} pockets per block")
    
    # Divide files into fixed-size blocks for cabbage generation
    blocks = [cavity_files[i:i + block_size] for i in range(0, len(cavity_files), block_size)]
    
    # Test mode: limit to first N blocks for quick testing
    if test_mode:
        blocks = blocks[:test_blocks]
        logger.warning(f"TEST MODE: Processing only first {test_blocks} blocks (up to {test_blocks * block_size} pockets)")
    
    logger.info(f"Preparing {len(blocks)} cabbage blocks (size <= {block_size})")
    
    # Create temporary working directory
    work_dir = os.path.join(output_dir, "temp_workspaces")
    os.makedirs(work_dir, exist_ok=True)
    
    try:
        # RESUME MODE: Parse previous run and validate parameters
        if resume:
            logger.info("="*80)
            logger.info("RESUME MODE ACTIVATED")
            logger.info("="*80)
            
            # Parse previous run configuration
            prev_config = parse_previous_run_config(output_dir)
            if prev_config is None:
                logger.error("Cannot resume: Failed to parse previous run configuration")
                logger.error("Make sure the output directory contains a valid pocketmatch_progress.log file")
                return pd.DataFrame()
            
            # Validate current parameters match previous run
            current_params = argparse.Namespace(
                block_size=block_size,
                test_mode=test_mode,
                test_blocks=test_blocks,
                cavities=cavities_dir
            )
            
            if not validate_resume_parameters(current_params, prev_config):
                logger.error("Resume aborted due to parameter mismatch")
                logger.error("To resume, use the same parameters as the previous run:")
                logger.error(f"  --block-size {prev_config['block_size']}")
                if prev_config['test_mode']:
                    logger.error(f"  --test-mode --test-blocks {prev_config['test_blocks']}")
                return pd.DataFrame()
            
            # Reconstruct pocket-to-block mapping from current cavity files
            logger.info("Reconstructing block structure from cavity files...")
            pocket_to_block_map = reconstruct_pocket_to_block_mapping(cavity_files, block_size)
            logger.info(f"Mapped {len(pocket_to_block_map):,} pockets to {len(blocks)} blocks")
            
            # Identify which block pairs have already been completed
            logger.info("Analyzing existing result chunks to identify completed comparisons...")
            completed_pairs, pockets_with_results = identify_completed_block_pairs(output_dir, pocket_to_block_map, n_threads)
            
            total_pairs = len(blocks) * (len(blocks) + 1) // 2
            remaining_pairs = total_pairs - len(completed_pairs)
            
            logger.info(f"Resume summary:")
            logger.info(f"  - Total blocks: {len(blocks)}")
            logger.info(f"  - Total possible pairs: {total_pairs:,}")
            logger.info(f"  - Completed pairs: {len(completed_pairs):,}")
            logger.info(f"  - Remaining pairs: {remaining_pairs:,}")
            
            if remaining_pairs == 0:
                logger.info("\u2705 All comparisons already completed! Nothing to do.")
                logger.info(f"Results are in: {os.path.join(output_dir, 'result_chunks')}")
                return pd.DataFrame()
            
            # Identify which blocks need to be built for remaining comparisons
            logger.info("Identifying which blocks need to be built...")
            blocks_to_build = identify_blocks_to_build(len(blocks), completed_pairs)
            logger.info(f"Need to build {len(blocks_to_build)} blocks (out of {len(blocks)} total)")
            
            # Build only the needed blocks
            logger.info("Building required cabbage blocks...")
            built_blocks = [None] * len(blocks)  # Placeholder for all blocks
            
            # Pre-create workspaces only for needed blocks
            block_workspaces = {}
            for i in tqdm(sorted(blocks_to_build), desc="   Creating workspaces"):
                ws = create_isolated_pocketmatch_workspace(pm_base_dir, work_dir, f"blk_{i}")
                block_workspaces[i] = ws
            
            # Build needed blocks in parallel
            # In resume mode: use more aggressive parallelism within each block
            # Since we're building fewer blocks, we can afford more workers per block
            workers_per_block = max(1, n_threads // max(1, len(blocks_to_build)))
            workers_per_block = min(workers_per_block, cpu_count())  # Cap at CPU count
            
            logger.info(f"Resume mode: Using {n_threads} parallel block builders, each with {workers_per_block} internal workers")
            logger.info(f"This parallelizes AlphaFold residue extraction more aggressively")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
                futures = {
                    ex.submit(build_cabbage_block, i, blocks[i], pm_base_dir, work_dir, af_base_dir, workers_per_block, False, verbose, block_workspaces[i]): i
                    for i in blocks_to_build
                }
                completed = tqdm(concurrent.futures.as_completed(futures, timeout=BLOCK_PHASE_TIMEOUT), total=len(futures), desc="   Blocks")
                for fut in completed:
                    try:
                        result = fut.result(timeout=BLOCK_TIMEOUT)
                        block_idx = futures[fut]
                        built_blocks[block_idx] = result
                    except concurrent.futures.TimeoutError:
                        block_id = futures[fut]
                        logger.error(f"Block {block_id}: Timeout after 20 minutes")
                    except Exception as e:
                        block_id = futures[fut]
                        logger.error(f"Block {block_id}: Error - {e}")
            
            logger.info(f"Built {len(blocks_to_build)} cabbage blocks successfully")
            
            # Clean up temp workspaces
            for i in blocks_to_build:
                if built_blocks[i] is not None:
                    _, _, ws = built_blocks[i]
                    shutil.rmtree(ws, ignore_errors=True)
            
        else:
            completed_pairs = set()
            built_blocks = None
        
        # Generate ALL blocks if fresh start
        if built_blocks is None:
            # PRE-CREATE all workspaces sequentially to avoid "too many open files" errors
            logger.info(f"Pre-creating {len(blocks)} workspaces sequentially...")
            block_workspaces = {}
            for i in tqdm(range(len(blocks)), desc="   Creating workspaces"):
                ws = create_isolated_pocketmatch_workspace(pm_base_dir, work_dir, f"blk_{i}")
                block_workspaces[i] = ws
            
            # Generate all cabbage blocks in parallel using pre-created workspaces
            logger.info("Generating cabbage blocks in parallel...")
            built_blocks = []  # list of (cabbage_path, mapping, tmp_workspace)
        
            # Each block processes files with 1 worker internally (sequential per block)
            # But we run n_threads blocks in parallel
            logger.info(f"Using {n_threads} parallel blocks (each processes files with 1 worker)")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
                futures = {
                    ex.submit(build_cabbage_block, i, blk, pm_base_dir, work_dir, af_base_dir, 1, False, verbose, block_workspaces[i]): i
                    for i, blk in enumerate(blocks)
                }
                completed = tqdm(concurrent.futures.as_completed(futures, timeout=BLOCK_PHASE_TIMEOUT), total=len(futures), desc="   Blocks")
                for fut in completed:
                    try:
                        result = fut.result(timeout=BLOCK_TIMEOUT)
                        built_blocks.append(result)
                    except concurrent.futures.TimeoutError:
                        block_id = futures[fut]
                        logger.error(f"Block {block_id}: Timeout after 20 minutes")
                    except Exception as e:
                        block_id = futures[fut]
                        logger.error(f"Block {block_id}: Error - {e}")
                logger.info(f"Built {len(built_blocks)} cabbage blocks successfully")

                # Clean up temp workspaces used for building (keep only block files)
                for _, _, ws in built_blocks:
                    shutil.rmtree(ws, ignore_errors=True)

        # Prepare all pairwise comparisons between blocks (including self)
        # Handle both fresh start (list of tuples) and resume (list with None placeholders)
        if resume:
            # Resume mode: built_blocks is a list with None placeholders
            # Only iterate over non-None entries
            total_pairs = len(blocks) * (len(blocks) + 1) // 2
            pair_jobs = []
            
            for i in range(len(blocks)):
                for j in range(i, len(blocks)):
                    pair_id = f"{i}_{j}"
                    
                    # Skip if this pair was already completed
                    if pair_id in completed_pairs:
                        logger.debug(f"Skipping completed pair {pair_id}")
                        continue
                    
                    # Only add if both blocks were built
                    if built_blocks[i] is not None and built_blocks[j] is not None:
                        cab_i, map_i, _ = built_blocks[i]
                        cab_j, map_j, _ = built_blocks[j]
                        pair_jobs.append((pair_id, cab_i, cab_j, map_i, map_j))
                    else:
                        logger.error(f"Cannot compare pair {pair_id}: block(s) not built")
            
            logger.info(f"After filtering completed pairs: {len(pair_jobs)} comparisons remaining (out of {total_pairs} total)")
        else:
            # Fresh start: built_blocks is a list of tuples
            total_pairs = (len(built_blocks) * (len(built_blocks) + 1)) // 2
            pair_jobs = []
            
            for i, (cab_i, map_i, _) in enumerate(built_blocks):
                for j in range(i, len(built_blocks)):
                    cab_j, map_j, _ = built_blocks[j]
                    pair_id = f"{i}_{j}"
                    pair_jobs.append((pair_id, cab_i, cab_j, map_i, map_j))
            
            logger.info(f"Submitting {total_pairs} block comparisons (upper triangle)...")
        
        if len(pair_jobs) == 0:
            logger.info("\u2705 All comparisons already completed! Nothing to do.")
            logger.info(f"Results are in: {os.path.join(output_dir, 'result_chunks')}")
            return pd.DataFrame()  # Return empty, user should merge existing chunks

        # Use ON-DEMAND workspace creation to prevent disk clogging
        # Workspaces will be created, used, and deleted immediately by each comparison
        logger.info(f"Using on-demand workspace creation (prevents disk issues with {len(pair_jobs)} workspaces)...")

        # Run comparisons with bounded concurrency and timeouts
        # Memory-efficient processing: save results in batches to avoid RAM overflow
        BATCH_SIZE = chunk_rows  # Use configurable chunk size
        all_pair_results = []
        
        # In resume mode, start from next available chunk number
        if resume:
            batch_counter = get_next_chunk_counter(output_dir)
            logger.info(f"Resume mode: Starting from chunk {batch_counter}")
        else:
            batch_counter = 0
        
        saved_chunks = []
        chunk_dir = os.path.join(output_dir, 'result_chunks')
        
        # Progress tracking
        completed_count = 0
        empty_result_count = 0
        last_progress_log = 0
        PROGRESS_LOG_INTERVAL = 100
        
        # Background I/O executor for non-blocking chunk saves
        io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='io_saver')
        io_futures = []  # Track background save operations
        
        def save_batch_background(batch_df_copy, chunk_file_path, batch_id):
            """Save batch to disk in background thread to prevent I/O blocking."""
            try:
                batch_df_copy.to_csv(chunk_file_path, index=False)
                file_size = os.path.getsize(chunk_file_path) / (1024 * 1024)
                logger.info(f"Background save completed: batch {batch_id} ({file_size:.1f} MB)")
                return chunk_file_path
            except Exception as e:
                logger.error(f"Background save failed for batch {batch_id}: {e}")
                return None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
            futs = {
                ex.submit(run_block_comparison, pid, cab_i, cab_j, map_i, map_j, pm_base_dir, work_dir, debug, verbose, None): pid
                for (pid, cab_i, cab_j, map_i, map_j) in pair_jobs
            }
            logger.info(f"Submitted {len(futs)} comparison jobs (on-demand workspaces)")
            
            try:
                completed = tqdm(concurrent.futures.as_completed(futs, timeout=COMPARISON_PHASE_TIMEOUT), total=len(futs), desc="   Comparisons")
                for fut in completed:
                    completed_count += 1
                    
                    try:
                        result = fut.result(timeout=COMPARISON_TIMEOUT)
                        all_pair_results.append(result)
                        if result is None or len(result) == 0:
                            empty_result_count += 1
                    except concurrent.futures.TimeoutError:
                        pair_id = futs[fut]
                        logger.error(f"Pair {pair_id}: Timeout after 5 minutes")
                        all_pair_results.append(pd.DataFrame())
                        empty_result_count += 1
                    except Exception as e:
                        pair_id = futs[fut]
                        logger.error(f"Pair {pair_id}: Error - {e}")
                        all_pair_results.append(pd.DataFrame())
                        empty_result_count += 1
                    
                    # Periodic progress logging
                    if completed_count - last_progress_log >= PROGRESS_LOG_INTERVAL:
                        logger.info(f"Progress: {completed_count}/{len(futs)} comparisons, "
                                  f"{len(all_pair_results)} in memory, "
                                  f"{empty_result_count} empty, "
                                  f"{batch_counter} batches saved, "
                                  f"{len([f for f in io_futures if not f.done()])} saves pending")
                        last_progress_log = completed_count
                    
                    # Check if we should save a batch to disk
                    valid_results = [df for df in all_pair_results if df is not None and len(df) > 0]
                    if valid_results:
                        total_rows = sum(len(df) for df in valid_results)
                        logger.debug(f"Batch check: {len(valid_results)} DataFrames, {total_rows:,} rows (threshold: {BATCH_SIZE:,})")
                        
                        if total_rows >= BATCH_SIZE:
                            logger.info(f"Batch threshold reached: {total_rows:,} rows >= {BATCH_SIZE:,}")
                            os.makedirs(chunk_dir, exist_ok=True)
                            chunk_file = os.path.join(chunk_dir, f'chunk_{batch_counter:04d}.csv')
                            
                            # Prepare batch for background save (copy to avoid race conditions)
                            batch_df = pd.concat(valid_results, ignore_index=True)
                            
                            # Submit to background I/O thread
                            logger.info(f"Submitting batch {batch_counter} to background I/O thread...")
                            io_future = io_executor.submit(save_batch_background, batch_df.copy(), chunk_file, batch_counter)
                            io_futures.append(io_future)
                            saved_chunks.append(chunk_file)
                            batch_counter += 1
                            
                            # Clear memory immediately (background thread has its own copy)
                            rows_cleared = len(batch_df)
                            all_pair_results = []
                            del batch_df
                            logger.debug(f"Memory cleared: {rows_cleared:,} rows freed")
            
            except concurrent.futures.TimeoutError:
                # Global timeout on as_completed - save what we have and continue
                incomplete_count = sum(1 for f in futs if not f.done())
                logger.warning(f"Global timeout reached! {incomplete_count} futures still pending.")
                logger.warning(f"Saving completed results and continuing...")
                
                # Cancel pending futures (note: this may not stop running subprocesses)
                for fut in futs:
                    if not fut.done():
                        fut.cancel()
                
                # Log which pairs were incomplete for resume
                incomplete_pairs = [futs[f] for f in futs if not f.done()]
                if incomplete_pairs:
                    logger.info(f"Incomplete pairs (first 20): {incomplete_pairs[:20]}")
                    incomplete_log = os.path.join(output_dir, 'incomplete_pairs.txt')
                    with open(incomplete_log, 'w') as f:
                        for pair_id in incomplete_pairs:
                            f.write(f"{pair_id}\n")
                    logger.info(f"Full list of incomplete pairs saved to: {incomplete_log}")
        
        # Wait for all background I/O operations to complete
        logger.info(f"Waiting for {len(io_futures)} background save operations to complete...")
        for i, io_fut in enumerate(io_futures):
            try:
                result = io_fut.result(timeout=IO_TIMEOUT)
                if result:
                    logger.debug(f"I/O operation {i+1}/{len(io_futures)} completed: {os.path.basename(result)}")
            except Exception as e:
                logger.error(f"I/O operation {i+1}/{len(io_futures)} failed: {e}")
        io_executor.shutdown(wait=True)
        logger.info(f"All background saves completed")
        
        logger.info(f"Completed {len(futs)} block comparisons")

        # Combine final results (remaining in-memory results + saved chunks)
        logger.info(f"Combining results from {batch_counter} saved batches + remaining in-memory results...")
        final_dfs = []
        
        # Add any remaining in-memory results
        valid_results = [df for df in all_pair_results if df is not None and len(df) > 0]
        if valid_results:
            logger.info(f"Processing {len(valid_results)} remaining in-memory results...")
            final_dfs.extend(valid_results)
        
        # Load saved chunks
        if saved_chunks:
            logger.info(f"Loading {len(saved_chunks)} saved chunk files...")
            for chunk_file in saved_chunks:
                chunk_df = pd.read_csv(chunk_file)
                final_dfs.append(chunk_df)
                logger.debug(f"Loaded {len(chunk_df):,} rows from {os.path.basename(chunk_file)}")
        
        if not final_dfs:
            raise ValueError("No valid results from any comparison. Check error messages above for details.")
        
        # Process and save results in chunks to avoid memory overflow
        logger.info(f"Processing {len(final_dfs)} dataframes in batches...")
        
        # Create final output directory for result files
        results_dir = os.path.join(output_dir, 'PocketMatch_similarities')
        os.makedirs(results_dir, exist_ok=True)
        
        total_rows = 0
        total_duplicates_removed = 0
        output_files = []
        
        # Process in batches to manage memory
        FINAL_BATCH_SIZE = 100000
        current_batch = []
        current_batch_rows = 0
        file_counter = 0
        
        for df in final_dfs:
            current_batch.append(df)
            current_batch_rows += len(df)
            
            if current_batch_rows >= FINAL_BATCH_SIZE:
                # Process and save this batch
                batch_df = pd.concat(current_batch, ignore_index=True)
                initial_len = len(batch_df)
                batch_df = batch_df.drop_duplicates(subset=['Pocket1', 'Pocket2'])
                duplicates_removed = initial_len - len(batch_df)
                total_duplicates_removed += duplicates_removed
                
                # Sort by Pmax
                batch_df = batch_df.sort_values(by='Pmax', ascending=False).reset_index(drop=True)
                
                # Save to file
                output_file = os.path.join(results_dir, f'similarities_part_{file_counter:04d}.csv')
                batch_df.to_csv(output_file, index=False)
                output_files.append(output_file)
                
                logger.info(f"Saved part {file_counter}: {len(batch_df):,} rows to {os.path.basename(output_file)}")
                total_rows += len(batch_df)
                file_counter += 1
                
                # Clear memory
                current_batch = []
                current_batch_rows = 0
                del batch_df
        
        # Process remaining data
        if current_batch:
            batch_df = pd.concat(current_batch, ignore_index=True)
            initial_len = len(batch_df)
            batch_df = batch_df.drop_duplicates(subset=['Pocket1', 'Pocket2'])
            duplicates_removed = initial_len - len(batch_df)
            total_duplicates_removed += duplicates_removed
            
            batch_df = batch_df.sort_values(by='Pmax', ascending=False).reset_index(drop=True)
            
            output_file = os.path.join(results_dir, f'similarities_part_{file_counter:04d}.csv')
            batch_df.to_csv(output_file, index=False)
            output_files.append(output_file)
            
            logger.info(f"Saved part {file_counter}: {len(batch_df):,} rows to {os.path.basename(output_file)}")
            total_rows += len(batch_df)
            del batch_df
        
        # Clean up temporary chunk files
        if saved_chunks and os.path.exists(chunk_dir):
            logger.info(f"Cleaning up temporary chunk files...")
            shutil.rmtree(chunk_dir, ignore_errors=True)
        
        # Log summary statistics
        logger.info(f"="*80)
        logger.info(f"Summary Statistics:")
        logger.info(f"Total output files: {len(output_files)}")
        logger.info(f"Results directory: {results_dir}")
        logger.info(f"Total comparisons: {total_rows:,}")
        if total_duplicates_removed > 0:
            logger.info(f"Removed {total_duplicates_removed:,} duplicate comparisons")
        
        # Calculate overall statistics by reading first file as sample
        if output_files:
            sample_df = pd.read_csv(output_files[0])
            logger.info(f"Sample statistics from first file:")
            logger.info(f"  Pmax range: {sample_df['Pmax'].min():.3f} - {sample_df['Pmax'].max():.3f}")
            logger.info(f"  Pmin range: {sample_df['Pmin'].min():.3f} - {sample_df['Pmin'].max():.3f}")
            logger.info(f"  Mean Pmax: {sample_df['Pmax'].mean():.3f}")
            logger.info(f"  Mean Pmin: {sample_df['Pmin'].mean():.3f}")
            del sample_df
        
        return output_files
        
    finally:
        # Cleanup temporary workspaces
        if not debug:
            logger.info(f"Cleaning up temporary workspaces...")
            shutil.rmtree(work_dir, ignore_errors=True)
            logger.info(f"Cleanup complete")
        else:
            logger.info(f"Debug mode: Keeping temporary workspaces at: {work_dir}")


def build_parser():
    """The CLI: canonical kebab-case flags, single-dash spellings kept as aliases."""
    parser = argparse.ArgumentParser(
        prog='get_PM_scores.py',
        description='Run PocketMatch similarity analysis on cavity files (parallelized).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python3 get_PM_scores.py \\
    --cavities <run-dir>/extracted_cavities \\
    --output-dir <run-dir>/pocketmatch_results \\
    --pm-dir /home/onur/software/PocketMatch_v2.1/PocketMatch_v2.1 \\
    --af-structures <run-dir>/alphafold_structures \\
    --threads 8 --resume

Note: Uses PocketMatch Step3-PM_typeB for parallelization.
      Each thread processes a chunk of pockets against the full set.
      Temporary isolated workspaces are created to avoid conflicts.
        """
    )

    # Paths. The single-dash spellings are the ones in the operator's shell
    # history and in CLAUDE.md; they stay as aliases so nothing breaks.
    parser.add_argument(
        '--cavities', '-cavities',
        type=str,
        required=True,
        help='Path to extracted_cavities directory containing cavity PDB files'
    )
    parser.add_argument(
        '--output-dir', '-output',
        type=str,
        required=True,
        dest='output',
        help='Path to output directory for results (similarity CSVs and chunks)'
    )
    parser.add_argument(
        '--pm-dir', '-pm_dir',
        type=str,
        required=True,
        dest='pm_dir',
        help='Path to the PocketMatch v2.1 base directory (Step0-cabbage.sh, Step3-PM_typeB)'
    )
    parser.add_argument(
        '--af-structures', '-af_structures',
        type=str,
        default=None,
        dest='af_structures',
        help='Path to AlphaFold structures directory, used to restore complete '
             'residues around each cavity. Without it the cavity files are compared '
             'as-is, which PocketMatch may reject'
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        default=None,
        help='campaign run directory; the step log goes to <run-dir>/steps/%s/ '
             'instead of under --output-dir' % STEP_NAME
    )

    # Parallelism and batching
    parser.add_argument(
        '--threads', '-threads',
        type=int,
        default=None,
        help='Number of parallel workers (default: min(--thread-cap, %d CPUs))' % cpu_count()
    )
    parser.add_argument(
        '--thread-cap',
        type=int,
        default=DEFAULT_THREAD_CAP,
        help='Ceiling applied when --threads is not given; all CPUs at once '
             'exhausts file handles on large runs (default: %(default)s)'
    )
    parser.add_argument(
        '--chunk-rows',
        type=int,
        default=100000,
        help='Number of rows per chunk file when saving intermediate results (default: %(default)s)'
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=100,
        help='Number of pockets per block for cabbage generation (default: %(default)s, '
             'max: {0})'.format(MAX_BLOCK_SIZE)
    )

    # Timeouts. Every one of these was an inline literal; on slower storage they
    # are the first thing that has to change.
    parser.add_argument(
        '--residue-timeout', type=int, default=RESIDUE_TIMEOUT,
        help='seconds to extract complete residues for ONE cavity file (default: %(default)s)')
    parser.add_argument(
        '--cabbage-timeout', type=int, default=CABBAGE_TIMEOUT,
        help='seconds for ONE Step0-cabbage.sh run (default: %(default)s)')
    parser.add_argument(
        '--pm-timeout', type=int, default=PM_TIMEOUT,
        help='seconds for ONE Step3-PM_typeB comparison (default: %(default)s)')
    parser.add_argument(
        '--block-timeout', type=int, default=BLOCK_TIMEOUT,
        help='seconds to build ONE cabbage block, end to end (default: %(default)s)')
    parser.add_argument(
        '--block-phase-timeout', type=int, default=BLOCK_PHASE_TIMEOUT,
        help='seconds for the whole block-building phase (default: %(default)s)')
    parser.add_argument(
        '--comparison-timeout', type=int, default=COMPARISON_TIMEOUT,
        help='seconds for ONE block-vs-block comparison job (default: %(default)s)')
    parser.add_argument(
        '--comparison-phase-timeout', type=int, default=COMPARISON_PHASE_TIMEOUT,
        help='seconds for the whole comparison phase; on expiry the completed '
             'pairs are saved and the incomplete ones listed for --resume '
             '(default: %(default)s)')
    parser.add_argument(
        '--io-timeout', type=int, default=IO_TIMEOUT,
        help='seconds to wait for ONE background chunk save (default: %(default)s)')

    # Behaviour
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode: keep temporary workspaces for inspection'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose mode: log detailed progress for each block and comparison (implies DEBUG level)'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Test mode: process only the first N blocks (see --test-blocks)'
    )
    parser.add_argument(
        '--test-blocks',
        type=int,
        default=3,
        help='Number of blocks to process in test mode (default: %(default)s)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing progress (reuses existing blocks and continues from last chunk)'
    )

    add_runtime_arguments(parser, STEP_NAME)
    return parser


def main(argv=None):
    """Parse, configure the run, and compute the pocket similarities."""
    args = build_parser().parse_args(argv)
    apply_runtime_tunables(args)

    cavities_dir = os.path.abspath(args.cavities)
    output_dir = os.path.abspath(args.output)
    pm_base_dir = os.path.abspath(args.pm_dir)
    af_base_dir = os.path.abspath(args.af_structures) if args.af_structures else None

    # The step's log lives under --run-dir when there is one, and beside the
    # results otherwise -- never in the invocation directory.
    if args.verbose and args.log_level == 'INFO':
        args.log_level = 'DEBUG'
    resolve_out_dir(args, STEP_NAME, args.run_dir or output_dir)

    with step(STEP_NAME, args) as ctx:
        try:
            get_pm_similarities_parallel(
                cavities_dir,
                output_dir,
                pm_base_dir,
                af_base_dir=af_base_dir,
                n_threads=args.threads,
                debug=args.debug,
                verbose=args.verbose,
                test_mode=args.test_mode,
                test_blocks=args.test_blocks,
                chunk_rows=args.chunk_rows,
                block_size=args.block_size,
                resume=args.resume,
            )
        except (FileNotFoundError, ValueError) as exc:
            # A missing cavity tree or PocketMatch installation is an operator
            # error, not a crash to read a traceback for.
            ctx.logger.error("%s", exc)
            raise SystemExit(2)
        ctx.logger.info("ANALYSIS COMPLETE")
    return 0


if __name__ == '__main__':
    sys.exit(main())
