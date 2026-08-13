#!/usr/bin/env python3
"""
High-Performance Consensus Docking Data Parser and Preparer
===========================================================

This script optimizes the parsing and preparation of consensus docking results
with multiprocessing, progress bars, and performance optimizations.

Features:
- Multiprocessing for file discovery and loading
- Progress bars for all operations
- Memory-efficient data processing
- Fast file I/O with optimizations
- Comprehensive performance monitoring
"""

import os
import re
import sys
import glob
from pathlib import Path
import warnings
import time
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
from functools import partial
import psutil
import pandas as pd
import numpy as np
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# =============================================================================
# GLOBAL CONFIGURATION - Modify these paths as needed
# =============================================================================
BASE_DIR = "/media/onur/Elements/cavity_space_consensus_docking/2025_06_29_batch_dock"
POCKET_CLUSTERS_DIR = os.path.join(BASE_DIR, "pocketmatch_results", "pocket_clusters")

# DTA-ATLAS dataset (DL binding affinity predictions)
DTA_ATLAS_DIR = os.path.join(
    BASE_DIR,
    "dta_atlas_data",
    "home",
    "madinasu",
    "convertedData",
    "dta_atlas_dataset_1_0",
)

# DTA-ATLAS integration tunables
# - Batch size controls how many rows are processed at a time per parquet shard.
# - Max workers controls parallel file scanning (threads).
#   Increase if you have fast storage; decrease if you see disk thrashing.
DTA_ATLAS_BATCH_SIZE = 200_000
DTA_ATLAS_MAX_WORKERS = 24
DTA_ATLAS_PARALLEL = True

# Performance settings
CHUNK_SIZE = 10000  # For chunked processing
MAX_WORKERS_SEARCH = None  # Use all cores for search
MAX_WORKERS_LOAD = 8  # Limit for I/O operations
PROGRESS_UPDATE_INTERVAL = 0.1  # Progress bar update frequency


# =============================================================================
# Canonical vocabularies and identity parsing (the five FIX_PLAN §3.4 defects)
# =============================================================================

#: ``None`` means "discover everything". A cap here silently truncates the
#: analysis: the 2025-06-29 run used 10,000 and lost 14,173 of 24,173 completed
#: job directories without a single warning. Set an integer only when you
#: deliberately want a partial pass, and expect the manifest check to fail.
FILE_DISCOVERY_CAP = None


class MergeFanOut(Exception):
    """A join changed the row count. Never degrade this to a warning.

    DEFECT 6 (found while testing defect 5): ``annotate_sample_types`` wrapped its
    whole body in ``except Exception`` and, on any error, printed a warning and set
    ``sample_type``/``Gene_Name`` to NA. That converts an integrity failure into a
    silently unlabelled dataset -- the same "degrade instead of stop" reflex as the
    universe-widening fallbacks in the extraction scripts. This exception is
    deliberately re-raised past that handler.
    """

#: The engine names as they appear IN THE DATA (uppercase). The old code compared
#: against 'Smina'/'LeDock'/'Gnina', so `filter_complete_tool_coverage` could match
#: at most 1 of 4 and was a permanent silent no-op, and `LeDock_Score` was all-null.
CANONICAL_TOOLS = ('GOLD', 'SMINA', 'LEDOCK', 'GNINA')

_TOOL_ALIASES = {
    'gold': 'GOLD', 'smina': 'SMINA', 'ledock': 'LEDOCK', 'le-dock': 'LEDOCK',
    'gnina': 'GNINA',
}


def canonical_tool(value):
    """Map any spelling of an engine name to its canonical uppercase form."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _TOOL_ALIASES.get(text.lower(), text.upper())


def parse_job_dir_name(dir_name, known_accessions=None):
    """Recover identity from a job directory name.

    DEFECT 2 (fixed). The old expression was::

        re.match(r'(DB\d+)_([A-Za-z0-9]+)_([A-Z0-9]+)_cavity_(\d+)', dir_name)

    ``[A-Za-z0-9]+`` cannot match a hyphen and the pattern has no place for a
    fragment, so it rejected 1,777 of 24,173 directories: 1,743 legacy
    multi-fragment names (``DB00091_CAMLG_P08F94_-F13-model_v1_cavity_10``) and 34
    hyphenated symbols (``DB00157_MT-ND1_P03886_cavity_1``). Those became
    1,174,786 rows (7.08%) with null identity, all sharing one fabricated
    ``<NA>_<NA>`` pair value that forms a bogus mega-group in any ``group_by``.

    Prefers ``job_identity.json`` when present (the authority), then the new
    ``__``-delimited scheme, then a right-anchored legacy parse. Returns a dict
    or ``None`` -- never a partial guess.
    """
    from dock_selectivity import jobid

    for parser in (jobid.parse_job_dir,):
        try:
            identity = parser(dir_name)
            return {
                'drugbank_id': identity.drugbank_id,
                'uniprot_id': identity.accession,
                'gene_name': identity.gene,
                'fragment_id': identity.fragment_id,
                'cavity_index': identity.cavity_index,
            }
        except jobid.JobIdError:
            pass

    if known_accessions:
        try:
            identity = jobid.parse_legacy_job_dir(dir_name, known_accessions)
            return {
                'drugbank_id': identity.drugbank_id,
                'uniprot_id': identity.accession,
                'gene_name': identity.gene,
                'fragment_id': identity.fragment_id,
                'cavity_index': identity.cavity_index,
            }
        except jobid.JobIdError:
            return None

    # No accession set available: a permissive right-anchored regex that at least
    # accepts hyphenated symbols and the legacy fragment suffix.
    match = re.match(
        r'^(DB\d+)_(.+?)_([A-Z0-9]{6,10})(?:_-?(?:AF-[A-Z0-9]+-)?F(\d+)-model_v\d+)?_cavity_(\d+)$',
        dir_name)
    if not match:
        return None
    drug, gene, accession, fragment, cavity = match.groups()
    return {
        'drugbank_id': drug,
        'uniprot_id': accession,
        'gene_name': None if gene.lower() in ('nan', 'none', '') else gene,
        'fragment_id': 'F%d' % int(fragment) if fragment else 'F1',
        'cavity_index': int(cavity),
    }


def get_system_info():
    """Get system information for optimization"""
    return {
        'cpu_count': multiprocessing.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'available_memory_gb': psutil.virtual_memory().available / (1024**3)
    }

def optimized_walk_search(args):
    """
    Ultra-optimized file search with multiple patterns and smart termination.
    """
    search_dir, patterns, max_depth, max_files_per_pattern = args
    results = {pattern: [] for pattern in patterns}
    
    try:
        for root, dirs, files in os.walk(search_dir):
            # Calculate current depth
            current_depth = root[len(search_dir):].count(os.sep)
            if max_depth is not None and current_depth >= max_depth:
                dirs[:] = []  # Don't descend further
                continue
            
            # Skip hidden and system directories for speed
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            # Check files against all patterns simultaneously
            for file in files:
                if file.startswith('.'):  # Skip hidden files
                    continue
                    
                file_lower = file.lower()
                for pattern in patterns:
                    if pattern in file_lower and (
                        max_files_per_pattern is None
                        or len(results[pattern]) < max_files_per_pattern
                    ):
                        full_path = os.path.join(root, file)
                        results[pattern].append(full_path)
            
            # Early termination only when a cap was deliberately requested.
            if max_files_per_pattern is not None and all(
                len(files) >= max_files_per_pattern for files in results.values()
            ):
                break
                
    except (PermissionError, OSError, UnicodeDecodeError):
        pass  # Skip problematic directories
    
    return results

def find_consensus_results_files_optimized(base_directory, max_workers=None):
    """
    Ultra-fast multiprocessing file discovery with intelligent search strategy.
    """
    print(f"\n🔍 ULTRA-OPTIMIZED FILE DISCOVERY")
    print(f"=" * 50)
    print(f"📁 Searching in: {base_directory}")
    
    if not os.path.exists(base_directory):
        print(f"❌ Directory not found: {base_directory}")
        return {}
    
    search_start = time.time()
    
    # Get system info and optimize worker count
    system_info = get_system_info()
    if max_workers is None:
        max_workers = min(system_info['cpu_count'], 16)  # Cap at 16 for I/O bound tasks
    
    print(f"💻 Using {max_workers} cores for parallel search")
    
    # Define search patterns - search all at once for efficiency
    search_patterns = [
        'final_results.csv',
        'results.csv',  # This will catch smina/gold/ledock results.csv
        'smina_results.csv', 
        'gexisting_results.csv',
        'ledock_results.csv',
        'combined_'  # For combined results
    ]
    
    # Get search directories with intelligent partitioning
    search_dirs = get_search_directories(base_directory, max_workers)
    max_depth = 4  # Slightly deeper search
    # DEFECT 1 (fixed): this was `max_files_per_pattern = 10000  # Higher limit`.
    # Combined with the len() guard at :99 and the early `break` at :104, it capped
    # discovery and silently dropped 14,173 of 24,173 completed job directories --
    # 58% of the campaign, 845 GB and 8,732 core-hours -- from the analysis. The
    # loss was differential (30.6% of negatives retained vs ~50% of positives),
    # which is the mechanism behind the unexplained 2.03:1 class imbalance.
    # None is now the sentinel for "no cap"; a cap may only be set deliberately.
    max_files_per_pattern = FILE_DISCOVERY_CAP
    
    print(f"📂 Partitioned search into {len(search_dirs)} segments")
    
    # Parallel file search with all patterns at once
    all_results = {pattern: [] for pattern in search_patterns}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(search_dirs), desc="🔍 Searching", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            
            # Submit search jobs
            futures = []
            for search_dir in search_dirs:
                args = (search_dir, search_patterns, max_depth, max_files_per_pattern)
                future = executor.submit(optimized_walk_search, args)
                futures.append(future)
            
            # Collect results
            total_found = 0
            for future in as_completed(futures):
                try:
                    segment_results = future.result(timeout=60)  # Longer timeout
                    for pattern, files in segment_results.items():
                        all_results[pattern].extend(files)
                        total_found += len(files)
                    
                    pbar.set_postfix({'Found': f"{total_found:,}"})
                    pbar.update(1)
                except Exception as e:
                    print(f"⚠️ Search segment failed: {e}")
                    pbar.update(1)
    
    # Categorize results intelligently
    categorized_results = categorize_found_files(all_results)
    
    search_time = time.time() - search_start
    total_files = sum(len(files) for files in categorized_results.values())
    
    print(f"\n✅ File discovery completed in {search_time:.2f} seconds")
    print(f"📊 Files found by category:")
    for category, files in categorized_results.items():
        print(f"  {category}: {len(files):,} files")
    print(f"📋 Total files: {total_files:,}")
    
    if total_files > 0:
        print(f"⚡ Search rate: {total_files/search_time:,.0f} files/sec")
    
    return categorized_results


def get_search_directories(base_directory, max_workers):
    """
    Intelligently partition the search space for optimal parallelization.
    """
    search_dirs = []
    
    try:
        # Get immediate subdirectories
        items = os.listdir(base_directory)
        subdirs = [os.path.join(base_directory, item) for item in items 
                  if os.path.isdir(os.path.join(base_directory, item)) 
                  and not item.startswith('.')]
        
        if len(subdirs) >= max_workers:
            # Use subdirectories directly if we have enough
            search_dirs = subdirs[:max_workers * 2]  # Take a reasonable number
        else:
            # Add the base directory and all subdirectories
            search_dirs = [base_directory] + subdirs
        
        # If still not enough, look one level deeper
        if len(search_dirs) < max_workers and subdirs:
            for subdir in subdirs[:5]:  # Limit to avoid explosion
                try:
                    sub_items = os.listdir(subdir)
                    sub_subdirs = [os.path.join(subdir, item) for item in sub_items 
                                  if os.path.isdir(os.path.join(subdir, item)) 
                                  and not item.startswith('.')]
                    search_dirs.extend(sub_subdirs[:max_workers])
                except (PermissionError, OSError):
                    continue
                    
                if len(search_dirs) >= max_workers * 2:
                    break
    
    except (PermissionError, OSError):
        search_dirs = [base_directory]
    
    return search_dirs if search_dirs else [base_directory]


def categorize_found_files(all_results):
    """
    Intelligently categorize found files into result types.
    """
    categorized = {
        'final_results': [],
        'smina_results': [],
        'gexisting_results': [],
        'ledock_results': [],
        'gnina_results': [],
        'combined_results': []
    }
    
    # Process all found files
    for pattern, files in all_results.items():
        for file_path in files:
            file_lower = file_path.lower()
            file_name = os.path.basename(file_lower)
            dir_path = os.path.dirname(file_lower)
            
            # Categorize based on path and filename
            if 'final_results' in file_name:
                categorized['final_results'].append(file_path)
            elif 'combined' in file_name and file_name.endswith('.csv'):
                categorized['combined_results'].append(file_path)
            elif 'smina' in dir_path and 'results.csv' in file_name:
                categorized['smina_results'].append(file_path)
            elif 'gold' in dir_path and 'results.csv' in file_name:
                categorized['gexisting_results'].append(file_path)
            elif 'ledock' in dir_path and 'results.csv' in file_name:
                categorized['ledock_results'].append(file_path)
            elif 'gnina' in dir_path and 'results.csv' in file_name:
                categorized['gnina_results'].append(file_path)
            elif 'smina' in file_name:
                categorized['smina_results'].append(file_path)
            elif 'gold' in file_name:
                categorized['gexisting_results'].append(file_path)
            elif 'ledock' in file_name:
                categorized['ledock_results'].append(file_path)
    
    # Remove duplicates while preserving order
    for category in categorized:
        categorized[category] = list(dict.fromkeys(categorized[category]))
    
    return categorized

def optimized_file_loader(file_info):
    """
    Ultra-fast file loader with smart chunking and memory optimization.
    """
    file_path, expected_size_mb = file_info
    
    try:
        # Measure actual file size
        actual_size_mb = os.path.getsize(file_path) / (1024**2)
        
        # Choose loading strategy based on file size
        read_kwargs = {
            'dtype': {'drugbank_id': 'str', 'uniprot_id': 'str'},
            'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'None', 'NaN'],
            'keep_default_na': True
        }
        
        if actual_size_mb > 500:  # Very large files
            read_kwargs.update({
                'low_memory': False,
                'engine': 'c',
                'chunksize': None  # Load all at once for very large files
            })
        elif actual_size_mb > 50:  # Medium files
            read_kwargs.update({
                'low_memory': False,
                'engine': 'c'
            })
        else:  # Small files
            read_kwargs.update({
                'engine': 'c'
            })
        
        # Load the CSV
        df = pd.read_csv(file_path, **read_kwargs)
        
        if df.empty:
            return None
        
        # Add metadata efficiently
        metadata = {
            'source_file': os.path.basename(file_path),
            'source_dir': os.path.dirname(file_path),
            'file_size_mb': actual_size_mb
        }
        
        # Determine source type from path analysis
        path_lower = file_path.lower()
        if 'final_results' in path_lower:
            metadata['source_type'] = 'final_results'
        elif '/smina/' in path_lower or 'smina' in os.path.basename(path_lower):
            metadata['source_type'] = 'smina_results'
        elif '/gold/' in path_lower or 'gold' in os.path.basename(path_lower):
            metadata['source_type'] = 'gexisting_results'
        elif '/ledock/' in path_lower or 'ledock' in os.path.basename(path_lower):
            metadata['source_type'] = 'ledock_results'
        elif '/gnina/' in path_lower or 'gnina' in os.path.basename(path_lower):
            metadata['source_type'] = 'gnina_results'
        elif 'combined' in path_lower:
            metadata['source_type'] = 'combined_results'
        else:
            metadata['source_type'] = 'unknown'
        
        # Add metadata columns efficiently
        for key, value in metadata.items():
            df[key] = value
        
        # Extract uniprot_id and drugbank_id from directory path
        # Path format: .../DB00390_ATP1A1_Q13286_cavity_3/...
        dir_path = os.path.dirname(file_path)
        dir_name = os.path.basename(dir_path)
        
        # Parse the directory name format: {drugbank_id}_{gene_name}_{uniprot_id}_cavity_{cavity_number}
        # Note: gene_name can be "nan" (lowercase) for entries without gene symbols
        parsed = parse_job_dir_name(dir_name)
        if parsed:
            df['drugbank_id'] = parsed['drugbank_id']
            df['uniprot_id'] = parsed['uniprot_id']
            df['gene_name'] = parsed['gene_name']
            df['fragment_id'] = parsed['fragment_id']
            df['cavity_index'] = parsed['cavity_index']
        else:
            # Fallback: keep existing values or set to Unknown
            if 'drugbank_id' not in df.columns:
                df['drugbank_id'] = 'Unknown'
            if 'uniprot_id' not in df.columns:
                df['uniprot_id'] = 'Unknown'
        
        return {
            'df': df,
            'file_path': file_path,
            'records': len(df),
            'size_mb': actual_size_mb,
            'source_type': metadata['source_type']
        }
        
    except Exception as e:
        return {
            'df': None,
            'file_path': file_path,
            'error': str(e),
            'records': 0,
            'size_mb': 0
        }

def load_consensus_data_optimized(result_files, max_workers=6):
    """
    Ultra-high-performance data loading with intelligent batching and memory management.
    """
    print(f"\n📖 ULTRA-OPTIMIZED DATA LOADING")
    print(f"=" * 50)
    
    # Flatten and prioritize files
    prioritized_files = prioritize_files_for_loading(result_files)
    
    if not prioritized_files:
        print("❌ No files to load")
        return pd.DataFrame()
    
    print(f"📄 Loading {len(prioritized_files):,} files using {max_workers} processes")
    
    load_start = time.time()
    successful_chunks = []
    failed_files = []
    total_size_mb = 0
    total_records = 0
    
    # Estimate file sizes for better scheduling
    file_info_list = []
    for file_path in prioritized_files:
        try:
            size_mb = os.path.getsize(file_path) / (1024**2)
            file_info_list.append((file_path, size_mb))
            total_size_mb += size_mb
        except OSError:
            file_info_list.append((file_path, 0))
    
    print(f"💾 Total data size: {total_size_mb:.1f} MB")
    
    # Load files with progress tracking and error handling
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(file_info_list), desc="📖 Loading Files", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            
            # Submit all loading jobs
            future_to_info = {
                executor.submit(optimized_file_loader, file_info): file_info
                for file_info in file_info_list
            }
            
            # Process results as they complete
            for future in as_completed(future_to_info):
                file_info = future_to_info[future]
                file_path = file_info[0]
                
                try:
                    result = future.result(timeout=120)  # 2 minute timeout per file
                    
                    if result and result.get('df') is not None and not result['df'].empty:
                        successful_chunks.append(result['df'])
                        total_records += result['records']
                        
                        # Update progress with detailed info
                        pbar.set_postfix({
                            'Files': len(successful_chunks),
                            'Records': f"{total_records:,}",
                            'Failed': len(failed_files),
                            'MB': f"{sum(r.get('size_mb', 0) for r in [result]):,.1f}"
                        })
                    else:
                        failed_files.append(file_path)
                        if result and 'error' in result:
                            print(f"⚠️ Failed: {os.path.basename(file_path)} - {result['error']}")
                    
                    pbar.update(1)
                    
                except Exception as e:
                    failed_files.append(file_path)
                    print(f"⚠️ Error loading {os.path.basename(file_path)}: {e}")
                    pbar.update(1)
    
    load_time = time.time() - load_start
    
    print(f"\n⚡ Loading phase completed in {load_time:.2f} seconds")
    print(f"✅ Successfully loaded: {len(successful_chunks)} files")
    print(f"❌ Failed to load: {len(failed_files)} files")
    print(f"📊 Total records loaded: {total_records:,}")
    
    if failed_files and len(failed_files) <= 10:
        print(f"⚠️  Failed files: {[os.path.basename(f) for f in failed_files]}")
    elif failed_files:
        print(f"⚠️  Failed files: {len(failed_files)} (showing first 5): {[os.path.basename(f) for f in failed_files[:5]]}")
    
    # Combine DataFrames efficiently
    if successful_chunks:
        print(f"\n🔄 Combining {len(successful_chunks)} DataFrames...")
        combine_start = time.time()
        
        # Sort chunks by size (largest first) for better memory usage
        successful_chunks.sort(key=lambda df: len(df), reverse=True)
        
        # Use concat with optimized settings
        consensus_data = pd.concat(
            successful_chunks, 
            ignore_index=True, 
            sort=False,
            copy=False  # Avoid unnecessary copying
        )
        
        combine_time = time.time() - combine_start
        total_time = load_time + combine_time
        
        # Memory optimization
        memory_mb = consensus_data.memory_usage(deep=True).sum() / (1024**2)
        
        print(f"✅ Data loading completed in {total_time:.2f} seconds")
        print(f"  📖 Loading: {load_time:.2f}s")
        print(f"  🔄 Combining: {combine_time:.2f}s")
        print(f"📊 Final dataset:")
        print(f"  📋 Records: {len(consensus_data):,}")
        print(f"  📊 Columns: {len(consensus_data.columns)}")
        print(f"  💾 Memory: {memory_mb:.1f} MB")
        print(f"⚡ Processing rate: {len(consensus_data)/total_time:,.0f} records/sec")
        
        return consensus_data
    else:
        print("❌ No valid data could be loaded")
        return pd.DataFrame()


def prioritize_files_for_loading(result_files):
    """
    Intelligently prioritize files for loading to optimize processing order.
    """
    prioritized = []
    
    # Priority order: final_results > combined > individual tools
    priority_order = [
        'final_results',
        'combined_results', 
        'smina_results',
        'gexisting_results',
        'ledock_results'
    ]
    
    for file_type in priority_order:
        if file_type in result_files and result_files[file_type]:
            files = result_files[file_type]
            
            # Sort files by likely importance (size, name, etc.)
            sorted_files = sorted(files, key=lambda f: (
                -os.path.getsize(f) if os.path.exists(f) else 0,  # Larger files first
                -len(os.path.basename(f)),  # Longer names tend to be more descriptive
                f  # Alphabetical as tiebreaker
            ))
            
            prioritized.extend(sorted_files)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_prioritized = []
    for file_path in prioritized:
        if file_path not in seen:
            seen.add(file_path)
            unique_prioritized.append(file_path)
    
    return unique_prioritized

def clean_and_prepare_data_optimized(df):
    """
    Ultra-fast data cleaning and preparation with optimizations.
    """
    if df.empty:
        return df
    
    print(f"\n🧹 OPTIMIZED DATA CLEANING")
    print(f"=" * 50)
    print(f"📊 Input data: {len(df):,} records, {len(df.columns)} columns")
    
    clean_start = time.time()
    
    # Make a copy to avoid modifying the original
    clean_df = df.copy()
    
    with tqdm(total=8, desc="🧹 Cleaning Steps", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        
        # Step 1: Standardize column names
        column_standardization = {
            'drug_id': 'drugbank_id',
            'compound_id': 'drugbank_id',
            'protein_id': 'uniprot_id',
            'target_id': 'uniprot_id',
            'binding_score': 'score',
            'docking_score': 'score',
            'affinity': 'score',
            'binding_affinity': 'score',
        }
        
        for old_name, new_name in column_standardization.items():
            if old_name in clean_df.columns and new_name not in clean_df.columns:
                clean_df[new_name] = clean_df[old_name]
        pbar.update(1)
        
        # Step 2: Convert numeric columns (vectorized)
        numeric_columns = ['score', 'rmsd', 'energy', 'consensus_score']
        for col in numeric_columns:
            if col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
        pbar.update(1)
        
        # Step 3: Clean identifiers (vectorized)
        identifier_columns = ['drugbank_id', 'uniprot_id', 'gene_name', 'drug_name', 'protein_name']
        for col in identifier_columns:
            if col in clean_df.columns:
                clean_df[col] = clean_df[col].astype(str).str.strip()
                clean_df[col] = clean_df[col].replace(['nan', 'Unknown', ''], pd.NA)
        
        # Validate extracted IDs
        if 'drugbank_id' in clean_df.columns and 'uniprot_id' in clean_df.columns:
            valid_drugbank = clean_df['drugbank_id'].str.startswith('DB', na=False)
            valid_uniprot = clean_df['uniprot_id'].str.match(r'^[A-Z0-9]{6}$', na=False)
            print(f"🔍 Valid DrugBank IDs: {valid_drugbank.sum():,}/{len(clean_df):,}")
            print(f"🔍 Valid UniProt IDs: {valid_uniprot.sum():,}/{len(clean_df):,}")
        pbar.update(1)
        
        # Step 4: Create categorical variables
        if 'docking_tool' in clean_df.columns:
            clean_df['docking_tool'] = clean_df['docking_tool'].astype('category')
        pbar.update(1)
        
        # Step 5: Add success/failure flags (vectorized)
        if 'score' in clean_df.columns:
            clean_df['docking_successful'] = clean_df['score'].notna()
            clean_df['high_affinity'] = clean_df['score'] < -7.0
        pbar.update(1)
        
        # Step 6: Extract and standardize tool information
        if 'source_type' in clean_df.columns:
            clean_df['primary_tool'] = clean_df['source_type'].str.replace('_results', '').str.title()
            tool_mapping = {
                'Smina': 'Smina',
                'Gold': 'Gold', 
                'Ledock': 'LeDock',
                'Gnina': 'Gnina',
                'Final': 'Consensus'
            }
            clean_df['primary_tool'] = clean_df['primary_tool'].map(tool_mapping).fillna(clean_df['primary_tool'])
        pbar.update(1)
        
        # Step 7: Create compound-target pairs (vectorized)
        if all(col in clean_df.columns for col in ['drugbank_id', 'uniprot_id']):
            clean_df['compound_target_pair'] = clean_df['drugbank_id'].astype(str) + '_' + clean_df['uniprot_id'].astype(str)
        pbar.update(1)
        
        # Step 8: Ensure analysis workflow compatibility
        if 'score' in clean_df.columns and 'Score1' not in clean_df.columns:
            clean_df['Score1'] = clean_df['score']
        
        if 'Score2' not in clean_df.columns:
            if 'consensus_score' in clean_df.columns:
                clean_df['Score2'] = clean_df['consensus_score']
            elif 'energy' in clean_df.columns:
                clean_df['Score2'] = clean_df['energy']
            elif 'Score1' in clean_df.columns:
                clean_df['Score2'] = clean_df['Score1']
        
        if 'rmsd' in clean_df.columns and 'RMSD' not in clean_df.columns:
            clean_df['RMSD'] = clean_df['rmsd']
        elif 'RMSD' not in clean_df.columns:
            clean_df['RMSD'] = float('nan')
        pbar.update(1)
    
    clean_time = time.time() - clean_start
    final_memory_mb = clean_df.memory_usage(deep=True).sum() / (1024**2)
    
    print(f"✅ Cleaning completed in {clean_time:.2f} seconds")
    print(f"📊 Output data: {len(clean_df):,} records, {len(clean_df.columns)} columns")
    print(f"💾 Memory usage: {final_memory_mb:.1f} MB")
    
    return clean_df

def generate_data_summary(df):
    """Generate comprehensive data summary with performance optimization."""
    if df.empty:
        return {}
    
    summary = {}
    
    # Basic statistics (vectorized)
    summary['total_records'] = len(df)
    summary['unique_compounds'] = df['drugbank_id'].nunique() if 'drugbank_id' in df.columns else 0
    summary['unique_targets'] = df['uniprot_id'].nunique() if 'uniprot_id' in df.columns else 0
    summary['unique_pairs'] = df['compound_target_pair'].nunique() if 'compound_target_pair' in df.columns else 0
    
    # ID extraction validation
    if 'drugbank_id' in df.columns and 'uniprot_id' in df.columns:
        valid_drugbank = df['drugbank_id'].str.startswith('DB', na=False).sum()
        valid_uniprot = df['uniprot_id'].str.match(r'^[A-Z0-9]{6}$', na=False).sum()
        
        summary['id_extraction'] = {
            'valid_drugbank_ids': int(valid_drugbank),
            'valid_uniprot_ids': int(valid_uniprot),
            'extraction_success_rate': float(min(valid_drugbank, valid_uniprot) / len(df)) if len(df) > 0 else 0.0
        }
    
    # Analysis workflow compatibility
    required_cols = ['drugbank_id', 'uniprot_id', 'Score1', 'RMSD']
    summary['analysis_compatible'] = all(col in df.columns for col in required_cols)
    summary['missing_required_columns'] = [col for col in required_cols if col not in df.columns]
    
    # Success rates
    if 'docking_successful' in df.columns:
        summary['success_rate'] = df['docking_successful'].mean() * 100
        summary['successful_dockings'] = df['docking_successful'].sum()
    
    # Score statistics
    score_col = 'Score1' if 'Score1' in df.columns else 'score'
    if score_col in df.columns:
        scores = df[score_col].dropna()
        if not scores.empty:
            summary['score_stats'] = {
                'column_used': score_col,
                'mean': float(scores.mean()),
                'median': float(scores.median()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'q25': float(scores.quantile(0.25)),
                'q75': float(scores.quantile(0.75)),
                'valid_count': int(len(scores)),
                'missing_count': int(df[score_col].isna().sum())
            }
    
    # RMSD statistics
    if 'RMSD' in df.columns:
        rmsd_values = df['RMSD'].dropna()
        if not rmsd_values.empty:
            summary['rmsd_stats'] = {
                'mean': float(rmsd_values.mean()),
                'median': float(rmsd_values.median()),
                'min': float(rmsd_values.min()),
                'max': float(rmsd_values.max()),
                'valid_count': int(len(rmsd_values)),
                'missing_count': int(df['RMSD'].isna().sum())
            }
    
    # Tool distribution
    if 'primary_tool' in df.columns:
        summary['tool_distribution'] = df['primary_tool'].value_counts().to_dict()
    
    return summary

def add_tool_specific_scores(df):
    """
    Add tool-specific score columns for easier downstream analysis.
    Creates columns like GOLD_Score, Smina_Score, LeDock_Score based on Tool1/Tool2 and Score1/Score2.
    """
    if df.empty or not all(col in df.columns for col in ['Tool1', 'Tool2', 'Score1', 'Score2']):
        return df
    
    print(f"\n🏷️  STEP: Adding tool-specific score columns...")
    
    # Create score columns for each tool
    tools = list(CANONICAL_TOOLS)  # DEFECT 3 (fixed): 'LeDock' never matched data 'LEDOCK'
    for tool in tools:
        _t1 = df['Tool1'].map(canonical_tool)
        _t2 = df['Tool2'].map(canonical_tool)
        df[f'{tool}_Score'] = np.where(
            _t1 == tool, df['Score1'],
            np.where(_t2 == tool, df['Score2'], np.nan)
        )
        # Reduce memory footprint (large datasets)
        df[f'{tool}_Score'] = df[f'{tool}_Score'].astype('float32')
    
    # Report
    for tool in tools:
        col_name = f'{tool}_Score'
        non_null = df[col_name].notna().sum()
        print(f"   {col_name}: {non_null:,} non-null values ({non_null/len(df)*100:.1f}%)")
    
    return df

def integrate_cavity_clusters(df, cluster_file="/opt/data/cavity_space/cavity_cluster_similarity07.csv"):
    """
    Integrate cavity cluster information from CavitySpace data.
    Maps cavities to similarity clusters for cluster-based analysis.
    """
    if df.empty:
        return df
    
    print(f"\n🗃️  STEP: Integrating cavity cluster data...")
    
    try:
        import re
        
        # Load cluster data
        clusters_df = pd.read_csv(cluster_file, sep='\t')
        print(f"   Loaded {len(clusters_df):,} clusters from CavitySpace")
        
        # Extract cavity identifiers from source_dir
        df['extracted_uniprot_id'] = df['source_dir'].str.extract(r'_([A-Z0-9]+)_cavity_\d+')[0]
        df['extracted_cavity_index'] = df['source_dir'].str.extract(r'cavity_(\d+)')[0].astype('Int64')
        
        non_null_uniprot = df['extracted_uniprot_id'].notna().sum()
        non_null_cavity = df['extracted_cavity_index'].notna().sum()
        print(f"   Extracted uniprot_id: {non_null_uniprot:,} non-null values")
        print(f"   Extracted cavity_index: {non_null_cavity:,} non-null values")
        
        if non_null_uniprot > 0 and non_null_cavity > 0:
            # Build mapping table (vectorized merge later; avoids df.apply(axis=1) memory spikes)
            mapping_rows = []
            for _, row in clusters_df.iterrows():
                cluster_id = row['id']
                cavity_items = row['items']
                if pd.isna(cavity_items):
                    continue
                for cavity_id in str(cavity_items).split(','):
                    cavity_id = cavity_id.strip()
                    match = re.match(r'AF-([A-Z0-9]+)-F\d+-model_v1_C(\d+)', cavity_id)
                    if not match:
                        continue
                    uniprot_id, cavity_index = match.groups()
                    mapping_rows.append((uniprot_id, int(cavity_index), cluster_id))

            if not mapping_rows:
                df['cavity_cluster_id'] = pd.NA
                return df

            mapping_df = pd.DataFrame(mapping_rows, columns=['extracted_uniprot_id', 'extracted_cavity_index', 'cavity_cluster_id'])
            mapping_df['extracted_cavity_index'] = mapping_df['extracted_cavity_index'].astype('Int64')
            mapping_df = mapping_df.drop_duplicates(subset=['extracted_uniprot_id', 'extracted_cavity_index'], keep='first')

            print(f"   Created mapping for {len(mapping_df):,} unique cavities")

            # Vectorized merge
            df = df.merge(mapping_df, on=['extracted_uniprot_id', 'extracted_cavity_index'], how='left')

            mapped_count = df['cavity_cluster_id'].notna().sum()
            unique_clusters = df['cavity_cluster_id'].nunique()
            print(f"   Mapped: {mapped_count:,}/{len(df):,} ({mapped_count/len(df)*100:.1f}%)")
            print(f"   Unique clusters: {unique_clusters:,}")
        else:
            df['cavity_cluster_id'] = pd.NA
            
    except FileNotFoundError:
        print(f"   ⚠️  Cluster file not found: {cluster_file}")
        df['cavity_cluster_id'] = pd.NA
    except Exception as e:
        print(f"   ⚠️  Error loading clusters: {e}")
        df['cavity_cluster_id'] = pd.NA
    
    return df

def integrate_ic50_data(df, ic50_file="/media/onur/Elements/cavity_space_consensus_docking/2025_06_29_batch_dock/ic50_mapping.csv"):
    """
    Integrate experimental IC50/Ki measurements from Therapeutic Target Database.
    Provides quantitative binding affinity data for validation.
    """
    if df.empty:
        return df
    
    print(f"\n🧪 STEP: Integrating IC50 data...")
    
    try:
        ic50_df = pd.read_csv(ic50_file)
        print(f"   Loaded IC50 data: {len(ic50_df):,} measurements")
        print(f"   Unique drugs: {ic50_df['drugbank_id'].nunique():,}")
        print(f"   Unique targets: {ic50_df['uniprot_id'].nunique():,}")
        
        # Handle duplicates - keep best (lowest) IC50 per drug-target pair
        ic50_aggregated = ic50_df.groupby(['drugbank_id', 'uniprot_id']).agg({
            'ic50_value': 'min',
            'ic50_unit': 'first',
            'measurement_type': 'first',
            'operator': 'first',
            'pubchem_cid': 'first',
            'activity': 'first'
        }).reset_index()
        ic50_aggregated['n_measurements'] = ic50_df.groupby(['drugbank_id', 'uniprot_id']).size().values
        
        print(f"   After aggregation: {len(ic50_aggregated):,} unique pairs")
        
        # Merge with docking data
        df = df.merge(ic50_aggregated, on=['drugbank_id', 'uniprot_id'], how='left')
        
        matched_ic50 = df['ic50_value'].notna().sum()
        print(f"   Matched {matched_ic50:,} docking results with IC50 data")
        print(f"   Coverage: {matched_ic50/len(df)*100:.2f}%")
        
    except FileNotFoundError:
        print(f"   ⚠️  IC50 file not found: {ic50_file}")
        for col in ['ic50_value', 'ic50_unit', 'measurement_type', 'operator', 'pubchem_cid', 'activity', 'n_measurements']:
            df[col] = pd.NA
    except Exception as e:
        print(f"   ⚠️  Error loading IC50 data: {e}")
        for col in ['ic50_value', 'ic50_unit', 'measurement_type', 'operator', 'pubchem_cid', 'activity', 'n_measurements']:
            df[col] = pd.NA
    
    return df


def _normalize_identifier_series(series: pd.Series, uppercase: bool = False) -> pd.Series:
    """Normalize identifier string series (strip whitespace/quotes, coerce common nulls)."""
    s = series.astype("string")
    s = s.str.strip()
    s = s.str.strip('"')
    s = s.str.strip("'")
    if uppercase:
        s = s.str.upper()
    s = s.replace(["", "nan", "none", "null", "unknown", "na", "n/a"], pd.NA)
    return s


def _process_dta_parquet_file(
    file_path: str,
    needed_drugs,
    needed_uniprots,
    needed_pairs,
    batch_size: int,
):
    """Worker: scan one parquet shard and aggregate only required pairs."""
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        if not needed_drugs or not needed_uniprots or not needed_pairs:
            return None

        pf = pq.ParquetFile(file_path)
        schema_names = pf.schema_arrow.names
        required = {"entry_id", "drugbank_id", "uniprot_id"}
        if not required.issubset(set(schema_names)):
            return None

        cols = schema_names
        needed_drugs_arr = pa.array(list(needed_drugs))
        needed_uniprots_arr = pa.array(list(needed_uniprots))

        agg_sum = None
        agg_cnt = None
        agg_rows = None
        agg_entry = None

        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            table = pa.Table.from_batches([batch])

            # Fast coarse filter: only keep rows where drugbank_id and uniprot_id are in our sets.
            try:
                mask_drug = pc.is_in(table["drugbank_id"], value_set=needed_drugs_arr)
                mask_uni = pc.is_in(table["uniprot_id"], value_set=needed_uniprots_arr)
                mask = pc.and_(mask_drug, mask_uni)
                filtered = table.filter(mask)
            except Exception:
                # If arrow-level filtering fails for any reason, fall back to pandas filtering.
                filtered = table

            if filtered.num_rows == 0:
                continue

            chunk = filtered.to_pandas()
            if chunk.empty:
                continue

            # Normalize identifiers on the reduced chunk only
            chunk["drugbank_id"] = _normalize_identifier_series(chunk["drugbank_id"], uppercase=True)
            chunk["uniprot_id"] = _normalize_identifier_series(chunk["uniprot_id"], uppercase=True)
            chunk["entry_id"] = _normalize_identifier_series(chunk["entry_id"], uppercase=False)
            chunk = chunk.dropna(subset=["drugbank_id", "uniprot_id"])
            if chunk.empty:
                continue

            pair_key = chunk["drugbank_id"].astype(str) + "\t" + chunk["uniprot_id"].astype(str)
            chunk = chunk[pair_key.isin(needed_pairs)]
            if chunk.empty:
                continue

            chunk = chunk.rename(columns={"entry_id": "dta_entry_id"})
            key_cols = {"drugbank_id", "uniprot_id"}
            rename_map = {c: f"dta_{c}" for c in chunk.columns if c not in key_cols and not c.startswith("dta_")}
            chunk = chunk.rename(columns=rename_map)

            group = chunk.groupby(["drugbank_id", "uniprot_id"], sort=False)

            file_rows = group.size().rename("dta_n_rows")

            non_key_cols = [c for c in chunk.columns if c not in ("drugbank_id", "uniprot_id")]
            numeric_cols = [c for c in non_key_cols if pd.api.types.is_numeric_dtype(chunk[c])]
            other_cols = [c for c in non_key_cols if c not in numeric_cols]

            file_sum = group[numeric_cols].sum(min_count=1) if numeric_cols else None
            file_cnt = group[numeric_cols].count() if numeric_cols else None
            file_entry = group["dta_entry_id"].first() if "dta_entry_id" in other_cols else None

            if file_sum is not None:
                if agg_sum is None:
                    agg_sum = file_sum
                    agg_cnt = file_cnt
                else:
                    agg_sum = agg_sum.add(file_sum, fill_value=0)
                    agg_cnt = agg_cnt.add(file_cnt, fill_value=0)

            if agg_rows is None:
                agg_rows = file_rows
            else:
                agg_rows = agg_rows.add(file_rows, fill_value=0)

            if file_entry is not None:
                if agg_entry is None:
                    agg_entry = file_entry
                else:
                    agg_entry = agg_entry.combine_first(file_entry)

            # Free per-batch objects aggressively
            del chunk

        if agg_rows is None:
            return None

        return {
            "sum": agg_sum,
            "cnt": agg_cnt,
            "rows": agg_rows,
            "entry": agg_entry,
        }
    except Exception:
        return None


def integrate_dta_atlas_data(df, dta_dir=None):
    """
    Integrate DTA-ATLAS binding affinity predictions.

    Loads parquet shards and left-merges per-model prediction columns onto the
    docking dataframe using (drugbank_id, uniprot_id) as join keys.

    Policies:
    - Keep all per-model columns.
    - Prefix incoming columns with `dta_`.
    - Store `entry_id` as `dta_entry_id`.
    - Aggregate duplicates per (drugbank_id, uniprot_id) and add `dta_n_rows`.
    """
    if df.empty:
        return df

    if dta_dir is None:
        dta_dir = DTA_ATLAS_DIR

    print(f"\n🧬 STEP: Integrating DTA-ATLAS predictions...")
    print(f"   DTA-ATLAS directory: {dta_dir}")

    if not os.path.exists(dta_dir):
        print("   ⚠️  DTA-ATLAS directory not found, skipping")
        return df

    # Determine which pairs we actually need
    if "drugbank_id" not in df.columns or "uniprot_id" not in df.columns:
        print("   ⚠️  Missing join keys in docking dataframe, skipping DTA-ATLAS")
        return df

    needed_drugs = set(_normalize_identifier_series(df["drugbank_id"], uppercase=True).dropna().unique())
    needed_uniprots = set(_normalize_identifier_series(df["uniprot_id"], uppercase=True).dropna().unique())
    needed_pairs = set(
        (
            _normalize_identifier_series(df["drugbank_id"], uppercase=True)
            + "\t"
            + _normalize_identifier_series(df["uniprot_id"], uppercase=True)
        ).dropna().unique()
    )

    if not needed_pairs:
        print("   ⚠️  No valid (drugbank_id, uniprot_id) pairs found, skipping DTA-ATLAS")
        return df

    parquet_files = sorted(glob.glob(os.path.join(dta_dir, "*.parquet")))
    if not parquet_files:
        print(f"   ⚠️  No parquet files found in {dta_dir}")
        return df

    print(f"   Docking pairs needed: {len(needed_pairs):,}")
    print(f"   Scanning DTA-ATLAS shards: {len(parquet_files):,}")

    # Streaming aggregation across shards: keep only per-pair running sums/counts
    agg_sum = None
    agg_cnt = None
    agg_rows = None
    agg_entry = None

    batch_size = int(DTA_ATLAS_BATCH_SIZE)
    max_workers = min(int(DTA_ATLAS_MAX_WORKERS), multiprocessing.cpu_count(), len(parquet_files))
    if not DTA_ATLAS_PARALLEL:
        max_workers = 1

    if max_workers <= 1:
        for p in tqdm(parquet_files, desc="🧬 Scanning DTA-ATLAS", unit="file"):
            part = _process_dta_parquet_file(p, needed_drugs, needed_uniprots, needed_pairs, batch_size)
            if not part:
                continue
            if part["sum"] is not None:
                if agg_sum is None:
                    agg_sum = part["sum"]
                    agg_cnt = part["cnt"]
                else:
                    agg_sum = agg_sum.add(part["sum"], fill_value=0)
                    agg_cnt = agg_cnt.add(part["cnt"], fill_value=0)
            agg_rows = part["rows"] if agg_rows is None else agg_rows.add(part["rows"], fill_value=0)
            if part["entry"] is not None:
                agg_entry = part["entry"] if agg_entry is None else agg_entry.combine_first(part["entry"])
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _process_dta_parquet_file,
                    p,
                    needed_drugs,
                    needed_uniprots,
                    needed_pairs,
                    batch_size,
                )
                for p in parquet_files
            ]

            for fut in tqdm(as_completed(futures), total=len(futures), desc="🧬 Scanning DTA-ATLAS", unit="file"):
                part = fut.result()
                if not part:
                    continue
                if part["sum"] is not None:
                    if agg_sum is None:
                        agg_sum = part["sum"]
                        agg_cnt = part["cnt"]
                    else:
                        agg_sum = agg_sum.add(part["sum"], fill_value=0)
                        agg_cnt = agg_cnt.add(part["cnt"], fill_value=0)
                agg_rows = part["rows"] if agg_rows is None else agg_rows.add(part["rows"], fill_value=0)
                if part["entry"] is not None:
                    agg_entry = part["entry"] if agg_entry is None else agg_entry.combine_first(part["entry"])

    if agg_rows is None:
        print("   ⚠️  No matching DTA-ATLAS records found for required pairs")
        return df

    # Compute per-column means from sum/count
    out = pd.DataFrame(index=agg_rows.index)
    out["dta_n_rows"] = agg_rows.astype("Int64")
    if agg_entry is not None:
        out["dta_entry_id"] = agg_entry

    if agg_sum is not None and agg_cnt is not None:
        for col in agg_sum.columns:
            denom = agg_cnt[col].replace(0, np.nan)
            out[col] = (agg_sum[col] / denom).astype('float32')

    dta_agg = out.reset_index()
    print(f"   Unique DTA pairs matched: {len(dta_agg):,}")

    before_cols = len(df.columns)
    df = df.merge(dta_agg, on=["drugbank_id", "uniprot_id"], how="left")
    after_cols = len(df.columns)
    matched = df["dta_n_rows"].notna().sum() if "dta_n_rows" in df.columns else 0
    print(f"   Matched: {matched:,}/{len(df):,} ({matched/len(df)*100:.2f}%)")
    print(f"   Columns: {before_cols} → {after_cols}")

    return df

def annotate_sample_types(df, metadata_file="/media/onur/Elements/cavity_space_consensus_docking/2025_06_29_batch_dock/required_structures_with_negatives.csv"):
    """
    Annotate sample types (positive vs negative) based on metadata.
    Critical for distinguishing known interactions from random controls.
    """
    if df.empty:
        return df
    
    print(f"\n🏷️  STEP: Annotating sample types (positive vs negative)...")
    
    try:
        sample_metadata = pd.read_csv(metadata_file)
        print(f"   Loaded {len(sample_metadata):,} rows from metadata")
        
        # Prepare metadata for merging
        merge_metadata = sample_metadata[['UniProt_ID', 'drugbank_id', 'Cavity_Index', 'sample_type', 'Gene_Name']].rename(
            columns={'UniProt_ID': 'uniprot_id', 'Cavity_Index': 'cavity_index'}
        )
        
        # Ensure cavity_index exists in df
        if 'extracted_cavity_index' in df.columns:
            df['cavity_index'] = df['extracted_cavity_index']
        
        # Merge
        # DEFECT 5 (fixed): this merged on (uniprot_id, drugbank_id, cavity_index),
        # omitting Fragment_ID. required_structures_with_negatives.csv carries 1,428
        # keys with >1 row that differ only by fragment, so the merge fanned out and
        # duplicated 410,087 parquet rows across 179 jobs -- one job 12x, confirmed
        # by parquet_rows / final_results_rows == the multiplicity exactly.
        _before = len(df)
        df = df.merge(merge_metadata, on=['uniprot_id', 'drugbank_id', 'cavity_index'], how='left')
        if len(df) != _before:
            raise MergeFanOut(
                "annotate_sample_types merge changed the row count %d -> %d (+%d). "
                "The key (uniprot_id, drugbank_id, cavity_index) is not unique in the "
                "metadata -- add Fragment_ID (normalised: the old pipeline wrote 'F1' "
                "on positives and bare '1' on negatives) or de-duplicate first."
                % (_before, len(df), len(df) - _before))
        
        annotated_rows = df['sample_type'].notna().sum()
        print(f"   Annotated: {annotated_rows:,}/{len(df):,} ({annotated_rows/len(df)*100:.1f}%)")
        
        if annotated_rows > 0:
            sample_counts = df['sample_type'].value_counts()
            print(f"   Sample type distribution:")
            for sample_type, count in sample_counts.items():
                print(f"      {sample_type}: {count:,}")
        
    except MergeFanOut:
        # Integrity failure, not a load failure. Degrading this to NA columns is
        # how 410,087 duplicated rows reached the analysis unremarked.
        raise
    except FileNotFoundError:
        print(f"   ⚠️  Metadata file not found: {metadata_file}")
        df['sample_type'] = pd.NA
        df['Gene_Name'] = pd.NA
    except Exception as e:
        print(f"   ⚠️  Error loading metadata: {e}")
        df['sample_type'] = pd.NA
        df['Gene_Name'] = pd.NA
    
    return df

def integrate_louvain_pocket_clusters(df, pocket_clusters_dir=None):
    """
    Integrate Louvain community detection cluster assignments from PocketMatch analysis.
    Adds cluster membership columns for different Pmax thresholds.
    
    Args:
        df: DataFrame with docking results
        pocket_clusters_dir: Directory containing pocket_louvain_clusters_*.csv files
    
    Returns:
        DataFrame with additional Louvain cluster columns
    """
    if df.empty:
        return df
    
    if pocket_clusters_dir is None:
        pocket_clusters_dir = POCKET_CLUSTERS_DIR
    
    print(f"\n🔗 STEP: Integrating Louvain pocket cluster data...")
    print(f"   Cluster directory: {pocket_clusters_dir}")
    
    try:
        import re
        from pathlib import Path
        
        cluster_dir = Path(pocket_clusters_dir)
        if not cluster_dir.exists():
            print(f"   ⚠️  Cluster directory not found: {pocket_clusters_dir}")
            return df
        
        # Find all cluster files
        cluster_files = sorted(cluster_dir.glob('pocket_louvain_clusters_*.csv'))
        
        if not cluster_files:
            print(f"   ⚠️  No cluster files found in {pocket_clusters_dir}")
            return df
        
        print(f"   Found {len(cluster_files)} cluster files")
        
        # Create pocket identifier column from source_dir if not exists
        # Format: AF-{uniprot_id}-F1-model_v1_cavity_{cavity_num}.pdb
        if 'pocket_id' not in df.columns:
            # Try to construct pocket_id from source_dir
            if 'source_dir' in df.columns:
                def extract_pocket_id(source_dir):
                    """Extract pocket identifier from source directory path."""
                    if pd.isna(source_dir):
                        return None
                    # Path format: .../DB00390_ATP1A1_Q13286_cavity_3/...
                    match = re.search(r'([A-Z0-9]+)_cavity_(\d+)', str(source_dir))
                    if match:
                        uniprot_id, cavity_num = match.groups()
                        return f"AF-{uniprot_id}-F1-model_v1_cavity_{cavity_num}.pdb"
                    return None
                
                df['pocket_id'] = df['source_dir'].apply(extract_pocket_id)
                valid_pocket_ids = df['pocket_id'].notna().sum()
                print(f"   Extracted pocket_id: {valid_pocket_ids:,} non-null values")
            else:
                # Try to construct from uniprot_id and cavity_index
                if 'uniprot_id' in df.columns and 'cavity_index' in df.columns:
                    df['pocket_id'] = df.apply(
                        lambda row: f"AF-{row['uniprot_id']}-F1-model_v1_cavity_{int(row['cavity_index'])}.pdb" 
                        if pd.notna(row['uniprot_id']) and pd.notna(row['cavity_index']) else None,
                        axis=1
                    )
                    valid_pocket_ids = df['pocket_id'].notna().sum()
                    print(f"   Constructed pocket_id: {valid_pocket_ids:,} non-null values")
                else:
                    print(f"   ⚠️  Cannot construct pocket_id - missing required columns")
                    return df
        
        # Load and merge each cluster file
        for cluster_file in cluster_files:
            # Extract Pmax threshold from filename (e.g., pocket_louvain_clusters_0p3.csv -> 0.3)
            filename = cluster_file.stem
            pmax_match = re.search(r'(\d+)p(\d+)', filename)
            if pmax_match:
                pmax_str = f"{pmax_match.group(1)}.{pmax_match.group(2)}"
                pmax_val = float(pmax_str)
            else:
                continue
            
            # Load cluster data
            cluster_df = pd.read_csv(cluster_file)
            
            # Rename columns to include Pmax threshold
            cluster_col_name = f"louvain_cluster_pmax{pmax_str.replace('.', '')}"
            cluster_size_col_name = f"louvain_cluster_size_pmax{pmax_str.replace('.', '')}"
            
            cluster_df = cluster_df.rename(columns={
                'Pocket': 'pocket_id',
                'Cluster': cluster_col_name,
                'ClusterSize': cluster_size_col_name
            })
            
            # Merge with main dataframe
            before_merge = len(df)
            df = df.merge(
                cluster_df[['pocket_id', cluster_col_name, cluster_size_col_name]], 
                on='pocket_id', 
                how='left'
            )
            
            matched = df[cluster_col_name].notna().sum()
            n_clusters = df[cluster_col_name].nunique()
            print(f"   Pmax >= {pmax_val}: {matched:,} matched ({matched/len(df)*100:.1f}%), {n_clusters} unique clusters")
        
        # Summary
        louvain_cols = [col for col in df.columns if col.startswith('louvain_cluster_pmax')]
        print(f"   ✅ Added {len(louvain_cols)} Louvain cluster columns")
        
    except Exception as e:
        print(f"   ⚠️  Error integrating Louvain clusters: {e}")
        import traceback
        traceback.print_exc()
    
    return df


def filter_complete_tool_coverage(df):
    """
    Filter for drug-target pairs where ALL tools made predictions.
    Ensures fair comparison between docking tools.
    """
    if df.empty:
        return df
    
    print(f"\n🔍 STEP: Filtering for complete tool coverage...")
    
    original_rows = len(df)
    
    # First filter for final_results source_type
    if 'source_type' in df.columns:
        df = df[df['source_type'] == 'final_results'].copy()
        print(f"   After final_results filter: {len(df):,} ({len(df)/original_rows*100:.1f}%)")
    
    if 'Tool1' not in df.columns or 'Tool2' not in df.columns:
        print(f"   ⚠️  Tool columns not found, skipping coverage filter")
        return df
    
    # Define expected tools
    expected_tools = list(CANONICAL_TOOLS)  # DEFECT 4 (fixed): 3 of 4 literals were unmatchable
    
    # Find pairs with complete coverage
    complete_pairs = []
    
    for (drug, target), group in df.groupby(['drugbank_id', 'uniprot_id']):
        tools_in_group = {canonical_tool(t) for t in group['Tool1'].dropna()} | {
            canonical_tool(t) for t in group['Tool2'].dropna()}
        tools_present = [t for t in expected_tools if t in tools_in_group]
        
        if len(tools_present) == len(expected_tools):
            complete_pairs.append((drug, target))
    
    if complete_pairs:
        print(f"   Found {len(complete_pairs):,} pairs with complete tool coverage")
        
        # Filter for complete pairs
        pair_set = set(complete_pairs)
        df = df[df.apply(lambda row: (row['drugbank_id'], row['uniprot_id']) in pair_set, axis=1)].copy()
        
        print(f"   After coverage filter: {len(df):,} ({len(df)/original_rows*100:.1f}%)")
    else:
        print(f"   ⚠️  No pairs with complete tool coverage found")
    
    return df

def balance_positive_negative_samples(df):
    """
    Balance positive and negative samples for each drug.
    Ensures equal representation for unbiased evaluation.
    """
    if df.empty or 'sample_type' not in df.columns:
        return df
    
    print(f"\n⚖️  STEP: Balancing positive/negative samples per drug...")
    
    original_rows = len(df)
    
    # Filter for drugs with both sample types
    drugs_with_both = []
    
    for drug in df['drugbank_id'].unique():
        drug_data = df[df['drugbank_id'] == drug]
        has_positive = (drug_data['sample_type'] == 'positive').any()
        has_negative = drug_data['sample_type'].str.contains('negative', na=False).any()
        
        if has_positive and has_negative:
            n_pos = (drug_data['sample_type'] == 'positive').sum()
            n_neg = drug_data['sample_type'].str.contains('negative', na=False).sum()
            drugs_with_both.append((drug, min(n_pos, n_neg)))
    
    print(f"   Drugs with both sample types: {len(drugs_with_both):,}")
    
    if drugs_with_both:
        # Balance samples for each drug
        balanced_chunks = []
        
        for drug, min_count in drugs_with_both:
            drug_data = df[df['drugbank_id'] == drug]
            
            pos_samples = drug_data[drug_data['sample_type'] == 'positive']
            neg_samples = drug_data[drug_data['sample_type'].str.contains('negative', na=False)]
            
            if len(pos_samples) > min_count:
                pos_samples = pos_samples.sample(n=min_count, random_state=42)
            if len(neg_samples) > min_count:
                neg_samples = neg_samples.sample(n=min_count, random_state=42)
            
            balanced_chunks.append(pos_samples)
            balanced_chunks.append(neg_samples)
        
        df = pd.concat(balanced_chunks, ignore_index=True)
        
        print(f"   After balancing: {len(df):,} ({len(df)/original_rows*100:.1f}%)")
        
        # Verify balance
        final_counts = df['sample_type'].value_counts()
        print(f"   Final distribution:")
        for sample_type, count in final_counts.items():
            print(f"      {sample_type}: {count:,}")
    
    return df

def export_prepared_data_optimized(df, output_dir="./"):
    """
    Export final prepared data as a single Parquet file ready for analysis.
    No intermediate files - just the final analysis-ready dataset.
    """
    if df.empty:
        print("⚠️  No data to export")
        return {}
    
    print(f"\n💾 EXPORTING FINAL PREPARED DATA")
    print(f"=" * 50)
    
    export_start = time.time()
    
    # Create output directory if needed
    if output_dir != "./":
        os.makedirs(output_dir, exist_ok=True)
    
    # Define output path for final analysis-ready file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_parquet = os.path.join(output_dir, "combined_filtered_annotated_docking_results.parquet")
    summary_path = os.path.join(output_dir, f"data_preparation_summary_{timestamp}.json")
    
    # Prepare export dataframe
    export_df = df.copy()
    
    # Column mapping for consistency
    column_mapping = {
        'drug_id': 'drugbank_id',
        'compound_id': 'drugbank_id',
        'protein_id': 'uniprot_id',
        'target_id': 'uniprot_id',
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in export_df.columns and new_col not in export_df.columns:
            export_df[new_col] = export_df[old_col]
    
    # Ensure required columns exist
    required_columns = ['drugbank_id', 'uniprot_id', 'RMSD', 'Score1', 'Score2']
    missing_cols = [col for col in required_columns if col not in export_df.columns]
    
    if missing_cols:
        print(f"⚠️  Missing required columns: {missing_cols}")
        for col in missing_cols:
            export_df[col] = pd.NA
    
    print(f"📊 Final dataset ready for export:")
    print(f"   Records: {len(export_df):,}")
    print(f"   Columns: {len(export_df.columns)}")
    print(f"   Memory: {export_df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
    
    # Export single Parquet file
    print(f"\n💾 Saving final analysis-ready Parquet file...")
    export_df.to_parquet(final_parquet, index=False)
    file_size_mb = os.path.getsize(final_parquet) / (1024**2)
    print(f"   ✅ Saved: {final_parquet} ({file_size_mb:.1f} MB)")
    
    # Export summary
    summary = generate_data_summary(export_df)
    summary['export_info'] = {
        'timestamp': timestamp,
        'final_output_file': final_parquet,
        'file_size_mb': file_size_mb,
        'processing_steps_applied': [
            'data_loading',
            'data_cleaning',
            'sample_type_annotation',
            'complete_tool_coverage_filter',
            'balanced_sampling'
            ,
            'dta_atlas_integration',
            'tool_specific_scores',
            'cavity_cluster_integration',
            'ic50_integration',
            'louvain_pocket_cluster_integration'
        ],
        'analysis_ready': True
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    export_time = time.time() - export_start
    
    print(f"\n✅ Export completed in {export_time:.2f} seconds")
    print(f"📄 Files created:")
    print(f"   Final Parquet: {final_parquet}")
    print(f"   Summary: {summary_path}")
    print(f"\n🎯 Dataset is now ready for analysis workflows!")
    
    return {
        'parquet_path': final_parquet,
        'summary_path': summary_path,
        'analysis_ready': True
    }

def main():
    """
    Main execution function with comprehensive performance monitoring.
    Now includes all notebook processing steps for complete analysis-ready data.
    """
    total_start_time = time.time()
    
    print("🚀 COMPREHENSIVE CONSENSUS DOCKING DATA PREPARATION")
    print("=" * 60)
    print("This script integrates all preprocessing steps:")
    print("  ✓ Data loading and cleaning")
    print("  ✓ Tool-specific score columns")
    print("  ✓ Cavity cluster integration")
    print("  ✓ IC50 experimental data")
    print("  ✓ Sample type annotation")
    print("  ✓ Louvain pocket cluster integration (PocketMatch)")
    print("  ✓ Complete tool coverage filtering")
    print("  ✓ Balanced positive/negative sampling")
    print("=" * 60)
    
    system_info = get_system_info()
    print(f"💻 System: {system_info['cpu_count']} cores, {system_info['memory_gb']:.1f}GB RAM")
    print(f"💾 Available: {system_info['available_memory_gb']:.1f}GB")
    
    print(f"\n📁 Using BASE_DIR: {BASE_DIR}")
    
    # Step 1: File Discovery
    print(f"\n{'='*60}")
    print(f"PHASE 1: FILE DISCOVERY")
    print(f"{'='*60}")
    # Search in parent directory to find all consensus docking files
    search_dir = os.path.dirname(BASE_DIR)
    result_files = find_consensus_results_files_optimized(
        base_directory=search_dir,
        max_workers=min(multiprocessing.cpu_count(), 12)
    )
    
    total_files = sum(len(files) for files in result_files.values())
    if total_files == 0:
        print("❌ No consensus docking files found. Exiting.")
        return
    
    # Step 2: Data Loading
    print(f"\n{'='*60}")
    print(f"PHASE 2: DATA LOADING")
    print(f"{'='*60}")
    consensus_data = load_consensus_data_optimized(
        result_files, 
        max_workers=min(8, system_info['cpu_count'])
    )
    
    if consensus_data.empty:
        print("❌ No data could be loaded. Exiting.")
        return
    
    # Step 3: Data Cleaning
    print(f"\n{'='*60}")
    print(f"PHASE 3: DATA CLEANING")
    print(f"{'='*60}")
    cleaned_data = clean_and_prepare_data_optimized(consensus_data)
    
    # Phase 4: FILTERING & BALANCING (before heavy annotations)
    print(f"\n{'='*60}")
    print(f"PHASE 4: FILTERING & BALANCING")
    print(f"{'='*60}")

    # Needed for balancing
    cleaned_data = annotate_sample_types(cleaned_data)

    # Reduce dataset early to avoid OOM during heavy annotations
    cleaned_data = filter_complete_tool_coverage(cleaned_data)
    cleaned_data = balance_positive_negative_samples(cleaned_data)

    # Phase 5: ENRICHMENT & ANNOTATION (apply only to final set)
    print(f"\n{'='*60}")
    print(f"PHASE 5: ENRICHMENT & ANNOTATION")
    print(f"{'='*60}")

    cleaned_data = integrate_dta_atlas_data(cleaned_data)
    cleaned_data = add_tool_specific_scores(cleaned_data)
    cleaned_data = integrate_cavity_clusters(cleaned_data)
    cleaned_data = integrate_ic50_data(cleaned_data)
    cleaned_data = integrate_louvain_pocket_clusters(cleaned_data, POCKET_CLUSTERS_DIR)
    
    # Step 11: Data Export
    print(f"\n{'='*60}")
    print(f"PHASE 6: FINAL EXPORT")
    print(f"{'='*60}")
    export_results = export_prepared_data_optimized(cleaned_data, output_dir=BASE_DIR)
    
    # Final Performance Summary
    total_time = time.time() - total_start_time
    
    print(f"\n🎉 PROCESSING COMPLETE!")
    print(f"=" * 60)
    print(f"⏱️  Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"📊 Final records: {len(cleaned_data):,}")
    print(f"⚡ Processing rate: {len(cleaned_data)/total_time:,.0f} records/sec")
    print(f"💾 Memory usage: {cleaned_data.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
    
    if export_results.get('analysis_ready'):
        print(f"\n✅ Analysis-ready file created!")
        print(f"📁 Output: {export_results['parquet_path']}")
        print(f"\n🚀 You can now run analysis notebooks directly!")
        print(f"   No need to run notebook preprocessing steps.")
    
    return cleaned_data, export_results

if __name__ == "__main__":
    try:
        cleaned_data, export_results = main()
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)