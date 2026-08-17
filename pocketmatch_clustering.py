#!/usr/bin/env python3
"""
PocketMatch clustering -- Louvain communities over pocket similarity scores.

Optional side branch of the campaign: it is not one of ``run_full_workflow.py``'s
phases. ``parse_prepare.py`` is nevertheless a *consumer* of its output -- it joins
``pocket_louvain_clusters_*.csv`` onto the docking table -- so this stage must run
before phase 8 if you want the ``louvain_cluster_*`` columns populated.

Prerequisites
    ``get_PM_scores.py`` has produced ``similarities_part_*.csv`` under
    ``<run-dir>/pocketmatch_results/PocketMatch_similarities``.

Inputs
    --csv-dir       PocketMatch similarity CSVs (default <run-dir>/pocketmatch_results/PocketMatch_similarities)
    --parquet-dir   where the CSV->Parquet conversion is cached (default <run-dir>/pocketmatch_results/parquet_data)

Outputs
    --output-dir    pocket_louvain_clusters_<pmax>.csv and pocket_network_louvain_<pmax>.gexf
                    (default <run-dir>/pocketmatch_results/pocket_clusters -- the path
                    parse_prepare.py --pocket-clusters defaults to)
    <out-dir>/pocketmatch_clustering.log  the run log (see --out-dir)

Usage::

    python3 pocketmatch_clustering.py --run-dir /data/runs/2026_08_run

    # explicit locations, a single threshold, notify on failure
    python3 pocketmatch_clustering.py \
        --csv-dir  ./pocketmatch_results/PocketMatch_similarities \
        --parquet-dir ./pocketmatch_results/parquet_data \
        --output-dir ./pocketmatch_results/pocket_clusters \
        --pmax-thresholds 0.5 --resolution 1.2 --notify
"""

import argparse
import gc
import logging
import os
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from community import community_louvain
from tqdm import tqdm

from dock_selectivity.runtime import add_runtime_arguments, resolve_out_dir, step

warnings.filterwarnings('ignore')

STEP_NAME = "pocketmatch_clustering"

#: Configured by ``main()`` through the shared runtime; a bare import attaches no
#: handler, so importing this module still logs nowhere.
LOG = logging.getLogger(STEP_NAME)


# --- Configuration ------------------------------------------------------------
# Every value below is a default for a flag of the same name, never a fixed
# setting: a run directory belongs on the command line, not in the source.

#: Layout under --run-dir. get_PM_scores.py writes the similarities here and
#: parse_prepare.py --pocket-clusters reads the clusters back from here.
SIMILARITIES_SUBDIR = os.path.join("pocketmatch_results", "PocketMatch_similarities")
PARQUET_SUBDIR = os.path.join("pocketmatch_results", "parquet_data")
CLUSTERS_SUBDIR = os.path.join("pocketmatch_results", "pocket_clusters")

DEFAULT_MAX_MEMORY_MB = 8000
DEFAULT_MIN_AVAILABLE_MB = 2000
DEFAULT_CSV_CHUNK_ROWS = 100_000
DEFAULT_RESOLUTION = 1.5
DEFAULT_PMAX_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

#: Implementation detail, not configuration: the filename get_PM_scores.py emits.
#: Changing it here without changing it there just finds nothing.
SIMILARITY_GLOB = 'similarities_part_*.csv'


# =============================================================================
# Memory Monitoring Utilities
# =============================================================================

def get_memory_usage_mb():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_available_memory_mb():
    """Get available system memory in MB"""
    return psutil.virtual_memory().available / 1024 / 1024


# =============================================================================
# CSV to Parquet Conversion
# =============================================================================

def convert_csv_to_parquet_chunked(csv_file, parquet_file,
                                    chunk_size=DEFAULT_CSV_CHUNK_ROWS,
                                    max_memory_mb=DEFAULT_MAX_MEMORY_MB,
                                    min_available_mb=DEFAULT_MIN_AVAILABLE_MB):
    """
    Convert CSV to Parquet using chunked reading to minimize memory usage.
    
    Args:
        csv_file: Path to input CSV file
        parquet_file: Path to output Parquet file
        chunk_size: Number of rows to process at a time
        max_memory_mb: Stop if process uses more than this (MB)
        min_available_mb: Stop if less than this available (MB)
    
    Returns:
        True if successful, False if memory limit exceeded
    """
    if get_memory_usage_mb() > max_memory_mb:
        LOG.warning(f"⚠️  Memory limit exceeded before starting: {get_memory_usage_mb():.1f} MB")
        return False
    
    if get_available_memory_mb() < min_available_mb:
        LOG.warning(f"⚠️  Low available memory: {get_available_memory_mb():.1f} MB")
        return False
    
    parquet_writer = None
    
    try:
        for chunk_idx, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size)):
            table = pa.Table.from_pandas(chunk)
            
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(parquet_file, table.schema, compression='snappy')
            
            parquet_writer.write_table(table)
            
            del chunk, table
            gc.collect()
            
            if get_memory_usage_mb() > max_memory_mb:
                LOG.warning(f"⚠️  Memory limit exceeded during processing: {get_memory_usage_mb():.1f} MB")
                if parquet_writer:
                    parquet_writer.close()
                return False
        
        if parquet_writer:
            parquet_writer.close()
        
        return True
        
    except Exception as e:
        LOG.info(f"Error converting {csv_file.name}: {e}")
        if parquet_writer:
            parquet_writer.close()
        return False


def ensure_parquet_files(csv_dir, parquet_dir,
                         max_memory_mb=DEFAULT_MAX_MEMORY_MB,
                         min_available_mb=DEFAULT_MIN_AVAILABLE_MB,
                         chunk_size=DEFAULT_CSV_CHUNK_ROWS):
    """
    Ensure all CSV files are converted to Parquet format.

    Args:
        csv_dir: Directory containing CSV files
        parquet_dir: Directory to store Parquet files
        max_memory_mb: Maximum memory usage allowed (MB)
        min_available_mb: Minimum available memory required (MB)
        chunk_size: Rows read per chunk during conversion

    Returns:
        List of Parquet file paths
    """
    csv_dir = Path(csv_dir)
    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(csv_dir.glob(SIMILARITY_GLOB))
    
    if not csv_files:
        LOG.info(f"No CSV files found in {csv_dir}")
        return []
    
    LOG.info(f"Found {len(csv_files)} CSV files")
    LOG.info(f"Total size: {sum(f.stat().st_size for f in csv_files) / 1024**3:.2f} GB")
    
    converted_files = []
    needs_conversion = False
    
    # Check which files need conversion
    for csv_file in csv_files:
        parquet_file = parquet_dir / csv_file.name.replace('.csv', '.parquet')
        if parquet_file.exists():
            converted_files.append(parquet_file)
        else:
            needs_conversion = True
    
    if not needs_conversion:
        LOG.info(f"✅ All {len(converted_files)} Parquet files already exist")
        return sorted(converted_files)
    
    # Convert remaining files
    LOG.info("\nConverting CSV files to Parquet format...")
    LOG.info(f"Memory safety: Max process usage = {max_memory_mb} MB, Min available = {min_available_mb} MB\n")
    
    converted_files = []
    for csv_file in tqdm(csv_files, desc="Converting files"):
        parquet_file = parquet_dir / csv_file.name.replace('.csv', '.parquet')
        
        if parquet_file.exists():
            converted_files.append(parquet_file)
            continue
        
        success = convert_csv_to_parquet_chunked(
            csv_file, parquet_file,
            chunk_size=chunk_size,
            max_memory_mb=max_memory_mb,
            min_available_mb=min_available_mb
        )
        
        if not success:
            LOG.error(f"\n❌ Stopped conversion due to memory constraints")
            LOG.info(f"Converted {len(converted_files)} out of {len(csv_files)} files")
            break
        
        converted_files.append(parquet_file)
        
        if len(converted_files) % 10 == 0:
            LOG.info(f"  Memory: {get_memory_usage_mb():.1f} MB used, {get_available_memory_mb():.1f} MB available")
    
    LOG.info(f"\n✅ Converted {len(converted_files)} files to Parquet format")
    
    # Calculate compression ratio
    if converted_files:
        csv_size = sum(f.stat().st_size for f in csv_files[:len(converted_files)]) / 1024**3
        parquet_size = sum(f.stat().st_size for f in converted_files) / 1024**3
        LOG.info(f"CSV size: {csv_size:.2f} GB")
        LOG.info(f"Parquet size: {parquet_size:.2f} GB")
        LOG.info(f"Compression ratio: {csv_size/parquet_size:.1f}x")
    
    return sorted(converted_files)


# =============================================================================
# Data Loading
# =============================================================================

def load_parquet_files(parquet_files, max_files=None,
                       max_memory_mb=DEFAULT_MAX_MEMORY_MB,
                       min_available_mb=DEFAULT_MIN_AVAILABLE_MB):
    """
    Load Parquet files into a single DataFrame with memory monitoring.
    
    Args:
        parquet_files: List of Parquet file paths
        max_files: Maximum number of files to load (None = all)
        max_memory_mb: Maximum memory usage allowed (MB)
        min_available_mb: Minimum available memory required (MB)
    
    Returns:
        Combined DataFrame or None if memory limit exceeded
    """
    dfs = []
    files_to_load = parquet_files[:max_files] if max_files else parquet_files
    
    LOG.info(f"Loading {len(files_to_load)} Parquet files...")
    LOG.info(f"Initial memory: {get_memory_usage_mb():.1f} MB\n")
    
    for i, parquet_file in enumerate(tqdm(files_to_load, desc="Loading files")):
        current_memory = get_memory_usage_mb()
        available_memory = get_available_memory_mb()
        
        if current_memory > max_memory_mb:
            LOG.warning(f"\n⚠️  Memory limit exceeded: {current_memory:.1f} MB")
            LOG.info(f"Loaded {i} out of {len(files_to_load)} files")
            break
        
        if available_memory < min_available_mb:
            LOG.warning(f"\n⚠️  Low available memory: {available_memory:.1f} MB")
            LOG.info(f"Loaded {i} out of {len(files_to_load)} files")
            break
        
        df = pd.read_parquet(parquet_file)
        dfs.append(df)
        
        if (i + 1) % 20 == 0:
            LOG.info(f"  Memory: {current_memory:.1f} MB used, {available_memory:.1f} MB available")
    
    if not dfs:
        LOG.info("No files loaded!")
        return None
    
    LOG.info(f"\nCombining {len(dfs)} DataFrames...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    del dfs
    gc.collect()
    
    LOG.info(f"✅ Loaded {len(combined_df):,} rows")
    LOG.info(f"Final memory usage: {get_memory_usage_mb():.1f} MB")
    
    return combined_df


# =============================================================================
# Louvain Clustering
# =============================================================================

def cluster_pockets_louvain(df_similarities, Pmax_min=0.3, resolution=DEFAULT_RESOLUTION):
    """
    Cluster pockets using Louvain community detection on a similarity graph.
    
    Args:
        df_similarities: DataFrame with columns ['Pocket1', 'Pocket2', 'Pmax']
        Pmax_min: Minimum Pmax threshold for including edges (default: 0.3)
        resolution: Louvain resolution parameter (higher = more clusters, default: 1.5)
    
    Returns:
        G: NetworkX graph with cluster attributes on nodes
        df_clusters: DataFrame with columns ['Pocket', 'Cluster', 'ClusterSize']
    """
    # Step 1: Filter by Pmax threshold
    LOG.info(f"Step 1: Filtering edges with Pmax >= {Pmax_min}...")
    df_filtered = df_similarities[df_similarities['Pmax'] >= Pmax_min].copy()
    LOG.info(f"  Kept {len(df_filtered):,} / {len(df_similarities):,} edges ({100*len(df_filtered)/len(df_similarities):.1f}%)")
    
    # Step 2: Build graph
    LOG.info(f"\nStep 2: Building graph...")
    G = nx.from_pandas_edgelist(
        df_filtered,
        source='Pocket1',
        target='Pocket2',
        edge_attr='Pmax',
    )
    # Rename edge attribute to 'weight' for Louvain
    nx.set_edge_attributes(G, nx.get_edge_attributes(G, 'Pmax'), 'weight')
    LOG.info(f"  ✅ Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Step 3: Run Louvain community detection
    LOG.info(f"\nStep 3: Running Louvain community detection (resolution={resolution})...")
    partition = community_louvain.best_partition(G, weight='weight', resolution=resolution)
    cluster_sizes = Counter(partition.values())
    LOG.info(f"  ✅ Found {len(cluster_sizes)} communities")
    LOG.info(f"  Largest 5: {cluster_sizes.most_common(5)}")
    
    # Step 4: Add cluster attributes to nodes
    LOG.info(f"\nStep 4: Adding cluster attributes to graph nodes...")
    for node in G.nodes():
        G.nodes[node]['cluster'] = partition.get(node, -1)
        G.nodes[node]['cluster_size'] = cluster_sizes.get(partition.get(node, -1), 0)
    LOG.info(f"  ✅ Added 'cluster' and 'cluster_size' attributes to all nodes")
    
    # Step 5: Create cluster membership DataFrame
    LOG.info(f"\nStep 5: Creating cluster membership DataFrame...")
    df_clusters = pd.DataFrame([
        {'Pocket': pocket, 'Cluster': cluster_id}
        for pocket, cluster_id in partition.items()
    ])
    df_clusters['ClusterSize'] = df_clusters['Cluster'].map(cluster_sizes)
    df_clusters = df_clusters.sort_values(
        ['ClusterSize', 'Cluster', 'Pocket'], 
        ascending=[False, True, True]
    ).reset_index(drop=True)
    LOG.info(f"  ✅ Created DataFrame with {len(df_clusters):,} pockets in {df_clusters['Cluster'].nunique()} clusters")
    
    # Summary
    LOG.info(f"\n{'='*50}")
    LOG.info(f"SUMMARY")
    LOG.info(f"{'='*50}")
    LOG.info(f"  Pmax threshold: >= {Pmax_min}")
    LOG.info(f"  Total pockets: {len(df_clusters):,}")
    LOG.info(f"  Total clusters: {df_clusters['Cluster'].nunique()}")
    LOG.info(f"  Largest cluster: {cluster_sizes.most_common(1)[0][1]} pockets")
    LOG.info(f"  Smallest cluster: {min(cluster_sizes.values())} pockets")
    LOG.info(f"  Median cluster size: {np.median(list(cluster_sizes.values())):.0f} pockets")
    
    return G, df_clusters


def sanitize_node_name(name):
    """Sanitize node name for XML/GEXF export"""
    name = str(name)
    name = re.sub(r'[<>&"\'\n\r\t]', '_', name)
    return name


def export_graph_gexf(G, output_path):
    """
    Export graph to GEXF format with sanitized node names.
    
    Args:
        G: NetworkX graph
        output_path: Path to output GEXF file
    """
    # Create a copy with sanitized node names
    G_clean = nx.Graph()
    
    # Create mapping
    node_mapping = {node: sanitize_node_name(node) for node in G.nodes()}
    
    # Add nodes with sanitized names
    for node in G.nodes():
        clean_name = node_mapping[node]
        G_clean.add_node(
            clean_name,
            cluster=int(G.nodes[node].get('cluster', -1)),
            cluster_size=int(G.nodes[node].get('cluster_size', 0)),
            original_name=str(node)
        )
    
    # Add edges
    for u, v, d in G.edges(data=True):
        G_clean.add_edge(
            node_mapping[u], 
            node_mapping[v], 
            weight=float(d.get('weight', d.get('Pmax', 1.0)))
        )
    
    nx.write_gexf(G_clean, output_path)
    LOG.info(f"  Exported: {output_path}")


def run_clustering_analysis(df_similarities, output_dir, pmax_thresholds=None,
                            resolution=DEFAULT_RESOLUTION):
    """
    Run Louvain clustering at multiple Pmax thresholds and export results.
    
    Args:
        df_similarities: DataFrame with columns ['Pocket1', 'Pocket2', 'Pmax']
        output_dir: Directory to save output files
        pmax_thresholds: List of Pmax thresholds to test (default: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        resolution: Louvain resolution parameter
    
    Returns:
        Dictionary mapping threshold to (graph, df_clusters) tuples
    """
    if pmax_thresholds is None:
        pmax_thresholds = list(DEFAULT_PMAX_THRESHOLDS)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for pmax_min in pmax_thresholds:
        LOG.info(f"\n{'='*60}")
        LOG.info(f"CLUSTERING WITH Pmax >= {pmax_min}")
        LOG.info(f"{'='*60}\n")
        
        G, df_clusters = cluster_pockets_louvain(df_similarities, Pmax_min=pmax_min, resolution=resolution)
        
        # Generate filenames
        pmax_str = f"{pmax_min:.1f}".replace('.', 'p')
        gexf_path = output_dir / f'pocket_network_louvain_{pmax_str}.gexf'
        csv_path = output_dir / f'pocket_louvain_clusters_{pmax_str}.csv'
        
        # Export results
        LOG.info(f"\nExporting results...")
        export_graph_gexf(G, gexf_path)
        df_clusters.to_csv(csv_path, index=False)
        LOG.info(f"  Exported: {csv_path}")
        
        results[pmax_min] = (G, df_clusters)
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def build_parser():
    """The CLI. Every path is a flag: the old --csv-dir default named one dead
    campaign directory on a read-only drive, and derived the *write* locations
    from that input path's parent."""
    parser = argparse.ArgumentParser(
        description='Perform Louvain community detection on PocketMatch similarity data'
    )
    parser.add_argument(
        '--run-dir',
        required=True,
        help=('campaign run directory. --csv-dir, --parquet-dir and --output-dir '
              'default to the standard layout underneath it')
    )
    parser.add_argument(
        '--csv-dir',
        type=str,
        default=None,
        help=('directory of get_PM_scores.py similarity CSVs (%s). '
              'Default: <run-dir>/%s' % (SIMILARITY_GLOB, SIMILARITIES_SUBDIR))
    )
    parser.add_argument(
        '--parquet-dir',
        type=str,
        default=None,
        help=('where the CSV->Parquet conversion is cached; written to, so it is '
              'named rather than derived from an input directory. '
              'Default: <run-dir>/%s' % PARQUET_SUBDIR)
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=('directory to save clustering results. Default: <run-dir>/%s -- the '
              'path parse_prepare.py --pocket-clusters reads back'
              % CLUSTERS_SUBDIR)
    )
    parser.add_argument(
        '--max-memory-mb',
        type=int,
        default=DEFAULT_MAX_MEMORY_MB,
        help='stop before this process exceeds this RSS, in MB (default: %d)' % DEFAULT_MAX_MEMORY_MB
    )
    parser.add_argument(
        '--min-available-mb',
        type=int,
        default=DEFAULT_MIN_AVAILABLE_MB,
        help='stop when system free memory falls below this, in MB (default: %d)' % DEFAULT_MIN_AVAILABLE_MB
    )
    parser.add_argument(
        '--csv-chunk-rows',
        type=int,
        default=DEFAULT_CSV_CHUNK_ROWS,
        help='rows read per chunk when converting a similarity CSV to Parquet (default: %d)' % DEFAULT_CSV_CHUNK_ROWS
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help=('load at most this many Parquet shards -- a deliberate partial pass '
              'for a smoke test. Default: all of them')
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=DEFAULT_RESOLUTION,
        help='Louvain resolution; higher means more, smaller communities (default: %s)' % DEFAULT_RESOLUTION
    )
    parser.add_argument(
        '--pmax-thresholds',
        type=float,
        nargs='+',
        default=list(DEFAULT_PMAX_THRESHOLDS),
        help=('Pmax edge thresholds to cluster at; one output pair per threshold '
              '(default: %s)' % ' '.join(str(t) for t in DEFAULT_PMAX_THRESHOLDS))
    )
    add_runtime_arguments(parser, STEP_NAME)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    resolve_out_dir(args, STEP_NAME, args.run_dir)
    with step(STEP_NAME, args):
        return _run(args)


def _run(args):
    """The clustering pass itself, with logging and notifications already set up."""
    run_dir = Path(args.run_dir)
    csv_dir = Path(args.csv_dir) if args.csv_dir else run_dir / SIMILARITIES_SUBDIR
    parquet_dir = Path(args.parquet_dir) if args.parquet_dir else run_dir / PARQUET_SUBDIR
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / CLUSTERS_SUBDIR

    # A missing input directory is a prerequisite failure, not an empty result.
    # SystemExit rather than `return 2` so step() reports it as a failure and the
    # --notify path fires; a plain non-zero return reads as a clean exit.
    if not csv_dir.is_dir():
        LOG.error("❌ Similarity directory not found: %s", csv_dir)
        LOG.error("   Run get_PM_scores.py first, or pass --csv-dir explicitly.")
        raise SystemExit(2)

    LOG.info("="*60)
    LOG.info("PocketMatch Louvain Clustering Analysis")
    LOG.info("="*60)
    LOG.info(f"\nConfiguration:")
    LOG.info(f"  CSV directory: {csv_dir}")
    LOG.info(f"  Parquet directory: {parquet_dir}")
    LOG.info(f"  Output directory: {output_dir}")
    LOG.info(f"  Max memory: {args.max_memory_mb} MB")
    LOG.info(f"  Min available memory: {args.min_available_mb} MB")
    LOG.info(f"  Resolution: {args.resolution}")
    LOG.info(f"  Pmax thresholds: {args.pmax_thresholds}")
    LOG.info(f"\nInitial memory usage: {get_memory_usage_mb():.1f} MB")
    LOG.info(f"Available memory: {get_available_memory_mb():.1f} MB")
    
    # Step 1: Ensure Parquet files exist
    LOG.info(f"\n{'='*60}")
    LOG.info("STEP 1: ENSURE PARQUET FILES EXIST")
    LOG.info("="*60 + "\n")
    
    parquet_files = ensure_parquet_files(
        csv_dir, parquet_dir,
        max_memory_mb=args.max_memory_mb,
        min_available_mb=args.min_available_mb,
        chunk_size=args.csv_chunk_rows
    )
    
    if not parquet_files:
        LOG.error("❌ No parquet files available. Exiting.")
        raise SystemExit(1)
    
    # Step 2: Load data
    LOG.info(f"\n{'='*60}")
    LOG.info("STEP 2: LOAD DATA")
    LOG.info("="*60 + "\n")
    
    df_similarities = load_parquet_files(
        parquet_files,
        max_files=args.max_files,
        max_memory_mb=args.max_memory_mb,
        min_available_mb=args.min_available_mb
    )
    
    if df_similarities is None:
        LOG.error("❌ Failed to load data. Exiting.")
        raise SystemExit(1)
    
    # Remove self-comparisons
    original_len = len(df_similarities)
    df_similarities = df_similarities[df_similarities['Pocket1'] != df_similarities['Pocket2']]
    LOG.info(f"\nRemoved {original_len - len(df_similarities):,} self-comparisons")
    LOG.info(f"Final dataset: {len(df_similarities):,} rows")
    
    # Print data summary
    LOG.info(f"\nData Summary:")
    LOG.info(f"  Total rows: {len(df_similarities):,}")
    unique_pockets = set(df_similarities['Pocket1'].unique()) | set(df_similarities['Pocket2'].unique())
    LOG.info(f"  Total unique pockets: {len(unique_pockets):,}")
    LOG.info(f"  Pmax range: [{df_similarities['Pmax'].min():.3f}, {df_similarities['Pmax'].max():.3f}]")
    LOG.info(f"  Pmax mean: {df_similarities['Pmax'].mean():.3f}")
    LOG.info(f"  Pmax median: {df_similarities['Pmax'].median():.3f}")
    
    # Step 3: Run clustering
    LOG.info(f"\n{'='*60}")
    LOG.info("STEP 3: LOUVAIN CLUSTERING")
    LOG.info("="*60)
    
    results = run_clustering_analysis(
        df_similarities,
        output_dir,
        pmax_thresholds=args.pmax_thresholds,
        resolution=args.resolution
    )
    
    # Final summary
    LOG.info(f"\n{'='*60}")
    LOG.info("FINAL SUMMARY")
    LOG.info("="*60)
    LOG.info(f"\nGenerated {len(results)} clustering results:")
    for pmax_min, (G, df_clusters) in results.items():
        LOG.info(f"  Pmax >= {pmax_min}: {G.number_of_nodes():,} pockets, {df_clusters['Cluster'].nunique()} clusters")
    
    LOG.info(f"\nOutput files saved to: {output_dir}")
    LOG.info(f"Final memory usage: {get_memory_usage_mb():.1f} MB")
    LOG.info("\n✅ Analysis complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
