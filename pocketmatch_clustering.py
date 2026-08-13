#!/usr/bin/env python3
"""
PocketMatch Clustering Script

This script performs Louvain community detection on pocket similarity data.
It handles:
1. Converting CSV files to Parquet format (if not already done)
2. Loading Parquet files into memory
3. Running Louvain community detection at multiple Pmax thresholds
4. Exporting results as GEXF (for Gephi) and CSV files

Usage:
    python pocketmatch_clustering.py [--csv-dir PATH] [--output-dir PATH]
"""

import argparse
import gc
import os
import re
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

warnings.filterwarnings('ignore')


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

def convert_csv_to_parquet_chunked(csv_file, parquet_file, chunk_size=100000,
                                    max_memory_mb=8000, min_available_mb=2000):
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
        print(f"⚠️  Memory limit exceeded before starting: {get_memory_usage_mb():.1f} MB")
        return False
    
    if get_available_memory_mb() < min_available_mb:
        print(f"⚠️  Low available memory: {get_available_memory_mb():.1f} MB")
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
                print(f"⚠️  Memory limit exceeded during processing: {get_memory_usage_mb():.1f} MB")
                if parquet_writer:
                    parquet_writer.close()
                return False
        
        if parquet_writer:
            parquet_writer.close()
        
        return True
        
    except Exception as e:
        print(f"Error converting {csv_file.name}: {e}")
        if parquet_writer:
            parquet_writer.close()
        return False


def ensure_parquet_files(csv_dir, parquet_dir, max_memory_mb=8000, min_available_mb=2000):
    """
    Ensure all CSV files are converted to Parquet format.
    
    Args:
        csv_dir: Directory containing CSV files
        parquet_dir: Directory to store Parquet files
        max_memory_mb: Maximum memory usage allowed (MB)
        min_available_mb: Minimum available memory required (MB)
    
    Returns:
        List of Parquet file paths
    """
    csv_dir = Path(csv_dir)
    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted(csv_dir.glob('similarities_part_*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return []
    
    print(f"Found {len(csv_files)} CSV files")
    print(f"Total size: {sum(f.stat().st_size for f in csv_files) / 1024**3:.2f} GB")
    
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
        print(f"✅ All {len(converted_files)} Parquet files already exist")
        return sorted(converted_files)
    
    # Convert remaining files
    print("\nConverting CSV files to Parquet format...")
    print(f"Memory safety: Max process usage = {max_memory_mb} MB, Min available = {min_available_mb} MB\n")
    
    converted_files = []
    for csv_file in tqdm(csv_files, desc="Converting files"):
        parquet_file = parquet_dir / csv_file.name.replace('.csv', '.parquet')
        
        if parquet_file.exists():
            converted_files.append(parquet_file)
            continue
        
        success = convert_csv_to_parquet_chunked(
            csv_file, parquet_file, 
            chunk_size=100000,
            max_memory_mb=max_memory_mb,
            min_available_mb=min_available_mb
        )
        
        if not success:
            print(f"\n❌ Stopped conversion due to memory constraints")
            print(f"Converted {len(converted_files)} out of {len(csv_files)} files")
            break
        
        converted_files.append(parquet_file)
        
        if len(converted_files) % 10 == 0:
            print(f"  Memory: {get_memory_usage_mb():.1f} MB used, {get_available_memory_mb():.1f} MB available")
    
    print(f"\n✅ Converted {len(converted_files)} files to Parquet format")
    
    # Calculate compression ratio
    if converted_files:
        csv_size = sum(f.stat().st_size for f in csv_files[:len(converted_files)]) / 1024**3
        parquet_size = sum(f.stat().st_size for f in converted_files) / 1024**3
        print(f"CSV size: {csv_size:.2f} GB")
        print(f"Parquet size: {parquet_size:.2f} GB")
        print(f"Compression ratio: {csv_size/parquet_size:.1f}x")
    
    return sorted(converted_files)


# =============================================================================
# Data Loading
# =============================================================================

def load_parquet_files(parquet_files, max_files=None, max_memory_mb=8000, min_available_mb=2000):
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
    
    print(f"Loading {len(files_to_load)} Parquet files...")
    print(f"Initial memory: {get_memory_usage_mb():.1f} MB\n")
    
    for i, parquet_file in enumerate(tqdm(files_to_load, desc="Loading files")):
        current_memory = get_memory_usage_mb()
        available_memory = get_available_memory_mb()
        
        if current_memory > max_memory_mb:
            print(f"\n⚠️  Memory limit exceeded: {current_memory:.1f} MB")
            print(f"Loaded {i} out of {len(files_to_load)} files")
            break
        
        if available_memory < min_available_mb:
            print(f"\n⚠️  Low available memory: {available_memory:.1f} MB")
            print(f"Loaded {i} out of {len(files_to_load)} files")
            break
        
        df = pd.read_parquet(parquet_file)
        dfs.append(df)
        
        if (i + 1) % 20 == 0:
            print(f"  Memory: {current_memory:.1f} MB used, {available_memory:.1f} MB available")
    
    if not dfs:
        print("No files loaded!")
        return None
    
    print(f"\nCombining {len(dfs)} DataFrames...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    del dfs
    gc.collect()
    
    print(f"✅ Loaded {len(combined_df):,} rows")
    print(f"Final memory usage: {get_memory_usage_mb():.1f} MB")
    
    return combined_df


# =============================================================================
# Louvain Clustering
# =============================================================================

def cluster_pockets_louvain(df_similarities, Pmax_min=0.3, resolution=1.5):
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
    print(f"Step 1: Filtering edges with Pmax >= {Pmax_min}...")
    df_filtered = df_similarities[df_similarities['Pmax'] >= Pmax_min].copy()
    print(f"  Kept {len(df_filtered):,} / {len(df_similarities):,} edges ({100*len(df_filtered)/len(df_similarities):.1f}%)")
    
    # Step 2: Build graph
    print(f"\nStep 2: Building graph...")
    G = nx.from_pandas_edgelist(
        df_filtered,
        source='Pocket1',
        target='Pocket2',
        edge_attr='Pmax',
    )
    # Rename edge attribute to 'weight' for Louvain
    nx.set_edge_attributes(G, nx.get_edge_attributes(G, 'Pmax'), 'weight')
    print(f"  ✅ Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Step 3: Run Louvain community detection
    print(f"\nStep 3: Running Louvain community detection (resolution={resolution})...")
    partition = community_louvain.best_partition(G, weight='weight', resolution=resolution)
    cluster_sizes = Counter(partition.values())
    print(f"  ✅ Found {len(cluster_sizes)} communities")
    print(f"  Largest 5: {cluster_sizes.most_common(5)}")
    
    # Step 4: Add cluster attributes to nodes
    print(f"\nStep 4: Adding cluster attributes to graph nodes...")
    for node in G.nodes():
        G.nodes[node]['cluster'] = partition.get(node, -1)
        G.nodes[node]['cluster_size'] = cluster_sizes.get(partition.get(node, -1), 0)
    print(f"  ✅ Added 'cluster' and 'cluster_size' attributes to all nodes")
    
    # Step 5: Create cluster membership DataFrame
    print(f"\nStep 5: Creating cluster membership DataFrame...")
    df_clusters = pd.DataFrame([
        {'Pocket': pocket, 'Cluster': cluster_id}
        for pocket, cluster_id in partition.items()
    ])
    df_clusters['ClusterSize'] = df_clusters['Cluster'].map(cluster_sizes)
    df_clusters = df_clusters.sort_values(
        ['ClusterSize', 'Cluster', 'Pocket'], 
        ascending=[False, True, True]
    ).reset_index(drop=True)
    print(f"  ✅ Created DataFrame with {len(df_clusters):,} pockets in {df_clusters['Cluster'].nunique()} clusters")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"  Pmax threshold: >= {Pmax_min}")
    print(f"  Total pockets: {len(df_clusters):,}")
    print(f"  Total clusters: {df_clusters['Cluster'].nunique()}")
    print(f"  Largest cluster: {cluster_sizes.most_common(1)[0][1]} pockets")
    print(f"  Smallest cluster: {min(cluster_sizes.values())} pockets")
    print(f"  Median cluster size: {np.median(list(cluster_sizes.values())):.0f} pockets")
    
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
    print(f"  Exported: {output_path}")


def run_clustering_analysis(df_similarities, output_dir, pmax_thresholds=None, resolution=1.5):
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
        pmax_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for pmax_min in pmax_thresholds:
        print(f"\n{'='*60}")
        print(f"CLUSTERING WITH Pmax >= {pmax_min}")
        print(f"{'='*60}\n")
        
        G, df_clusters = cluster_pockets_louvain(df_similarities, Pmax_min=pmax_min, resolution=resolution)
        
        # Generate filenames
        pmax_str = f"{pmax_min:.1f}".replace('.', 'p')
        gexf_path = output_dir / f'pocket_network_louvain_{pmax_str}.gexf'
        csv_path = output_dir / f'pocket_louvain_clusters_{pmax_str}.csv'
        
        # Export results
        print(f"\nExporting results...")
        export_graph_gexf(G, gexf_path)
        df_clusters.to_csv(csv_path, index=False)
        print(f"  Exported: {csv_path}")
        
        results[pmax_min] = (G, df_clusters)
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Perform Louvain community detection on PocketMatch similarity data'
    )
    parser.add_argument(
        '--csv-dir',
        type=str,
        default='/media/onur/Elements/cavity_space_consensus_docking/2025_06_29_batch_dock/pocketmatch_results/PocketMatch_similarities',
        help='Directory containing CSV similarity files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save clustering results (default: <csv-dir>/../pocket_clusters)'
    )
    parser.add_argument(
        '--max-memory-mb',
        type=int,
        default=8000,
        help='Maximum memory usage in MB (default: 8000)'
    )
    parser.add_argument(
        '--min-available-mb',
        type=int,
        default=2000,
        help='Minimum available memory in MB (default: 2000)'
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=1.5,
        help='Louvain resolution parameter (default: 1.5)'
    )
    parser.add_argument(
        '--pmax-thresholds',
        type=float,
        nargs='+',
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        help='Pmax thresholds for clustering (default: 0.3 0.4 0.5 0.6 0.7 0.8)'
    )
    
    args = parser.parse_args()
    
    csv_dir = Path(args.csv_dir)
    parquet_dir = csv_dir.parent / 'parquet_data'
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = csv_dir.parent / 'pocket_clusters'
    
    print("="*60)
    print("PocketMatch Louvain Clustering Analysis")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  CSV directory: {csv_dir}")
    print(f"  Parquet directory: {parquet_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Max memory: {args.max_memory_mb} MB")
    print(f"  Min available memory: {args.min_available_mb} MB")
    print(f"  Resolution: {args.resolution}")
    print(f"  Pmax thresholds: {args.pmax_thresholds}")
    print(f"\nInitial memory usage: {get_memory_usage_mb():.1f} MB")
    print(f"Available memory: {get_available_memory_mb():.1f} MB")
    
    # Step 1: Ensure Parquet files exist
    print(f"\n{'='*60}")
    print("STEP 1: ENSURE PARQUET FILES EXIST")
    print("="*60 + "\n")
    
    parquet_files = ensure_parquet_files(
        csv_dir, parquet_dir,
        max_memory_mb=args.max_memory_mb,
        min_available_mb=args.min_available_mb
    )
    
    if not parquet_files:
        print("❌ No parquet files available. Exiting.")
        return 1
    
    # Step 2: Load data
    print(f"\n{'='*60}")
    print("STEP 2: LOAD DATA")
    print("="*60 + "\n")
    
    df_similarities = load_parquet_files(
        parquet_files,
        max_memory_mb=args.max_memory_mb,
        min_available_mb=args.min_available_mb
    )
    
    if df_similarities is None:
        print("❌ Failed to load data. Exiting.")
        return 1
    
    # Remove self-comparisons
    original_len = len(df_similarities)
    df_similarities = df_similarities[df_similarities['Pocket1'] != df_similarities['Pocket2']]
    print(f"\nRemoved {original_len - len(df_similarities):,} self-comparisons")
    print(f"Final dataset: {len(df_similarities):,} rows")
    
    # Print data summary
    print(f"\nData Summary:")
    print(f"  Total rows: {len(df_similarities):,}")
    unique_pockets = set(df_similarities['Pocket1'].unique()) | set(df_similarities['Pocket2'].unique())
    print(f"  Total unique pockets: {len(unique_pockets):,}")
    print(f"  Pmax range: [{df_similarities['Pmax'].min():.3f}, {df_similarities['Pmax'].max():.3f}]")
    print(f"  Pmax mean: {df_similarities['Pmax'].mean():.3f}")
    print(f"  Pmax median: {df_similarities['Pmax'].median():.3f}")
    
    # Step 3: Run clustering
    print(f"\n{'='*60}")
    print("STEP 3: LOUVAIN CLUSTERING")
    print("="*60)
    
    results = run_clustering_analysis(
        df_similarities,
        output_dir,
        pmax_thresholds=args.pmax_thresholds,
        resolution=args.resolution
    )
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\nGenerated {len(results)} clustering results:")
    for pmax_min, (G, df_clusters) in results.items():
        print(f"  Pmax >= {pmax_min}: {G.number_of_nodes():,} pockets, {df_clusters['Cluster'].nunique()} clusters")
    
    print(f"\nOutput files saved to: {output_dir}")
    print(f"Final memory usage: {get_memory_usage_mb():.1f} MB")
    print("\n✅ Analysis complete!")
    
    return 0


if __name__ == '__main__':
    exit(main())
