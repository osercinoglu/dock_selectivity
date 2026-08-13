#!/usr/bin/env python3
"""
Cavity Extraction Script
This script extracts cavity/pocket PDB files from AlphaFold cavity tarballs.

This is a separate preprocessing step that can be run independently of the main
docking workflow, allowing for better organization and the ability to cache
extracted cavity files. The receptor PDB information comes from AlphaFold 
structures, not from the cavity tarballs.

Usage:
    python extract_cavities.py

Input:
    - Cavity tarball folder containing .tar.gz files
    
Output:
    - extracted_cavities/ folder with extracted cavity PDB files
    - cavity_mapping.csv: CSV file mapping UniProt IDs to extracted cavities
"""

import argparse
import os
import sys
import pandas as pd
import tarfile
from tqdm import tqdm
from pathlib import Path
import re
import time
import logging

# --- Configuration ---
# --- IMPORTANT: ADJUST THESE PATHS AS PER YOUR SYSTEM ---
CAVITY_TARBALL_FOLDER = "/opt/data/cavity_space/af_strong_cavities_data_all"
EXTRACT_BASE_DIR = "extracted_cavities"  # Will be created in current working directory
OUTPUT_CSV = "cavity_mapping.csv"
LOG_FILE = "cavity_extraction.log"

# Set up logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

def extract_cavity_paths_from_tarballs(cavity_tarball_folder, extract_base_dir="extracted_cavities"):
    """
    Iterates through tar.gz files, extracts cavity/pocket PDBs only,
    and returns a dictionary mapping UniProt ID to lists of (pocket_path, fragment_id, cavity_index).
    
    Args:
        cavity_tarball_folder (str): Path to folder containing .tar.gz files
        extract_base_dir (str): Base directory for extraction
        
    Returns:
        dict: Dictionary mapping UniProt IDs to lists of (pocket_pdb, fragment_id, cavity_index) tuples
    """
    logging.info(f"Extracting cavity paths from tarballs in '{cavity_tarball_folder}'...")
    cavity_info = {} # { "Q00577": [(cavity_pdb1, fragment_id1, cavity_index1), (cavity_pdb2, fragment_id2, cavity_index2)], ... }

    os.makedirs(extract_base_dir, exist_ok=True)

    tar_files = [f for f in os.listdir(cavity_tarball_folder) if f.endswith(".tar.gz")]
    
    if not tar_files:
        logging.warning(f"No .tar.gz files found in '{cavity_tarball_folder}'.")
        return cavity_info

    logging.info(f"Found {len(tar_files)} tarball files to process.")
    
    # Statistics tracking
    extracted_count = 0
    skipped_count = 0

    for tar_gz_filename in tqdm(tar_files, desc="Extracting Tarballs", unit="file"):
        full_tar_path = os.path.join(cavity_tarball_folder, tar_gz_filename)
        
        # Extract UniProt ID and fragment info from filename
        # Pattern: AF-Q96RW7-F23-model_v1_cavity_result_1.tar.gz
        match = re.search(r'AF-([A-Z0-9]+)-F(\d+)-model_v1', tar_gz_filename)
        if match:
            uniprot_id = match.group(1)
            fragment_id = int(match.group(2))
        else:
            tqdm.write(f"Warning: Could not extract UniProt ID and fragment from '{tar_gz_filename}'. Skipping.")
            continue

        extract_folder = os.path.join(extract_base_dir, Path(tar_gz_filename).stem.replace(".tar", ""))
        os.makedirs(extract_folder, exist_ok=True)

        # Dictionary to store cavity files: {cavity_num: cavity_pdb}
        cavity_files = {}

        # Check if extraction folder already exists and has files
        existing_cavity_files = []
        if os.path.exists(extract_folder):
            for file_path in Path(extract_folder).glob("*_cavity_*.pdb"):
                cavity_match = re.search(r'_cavity_(\d+)\.pdb$', file_path.name)
                if cavity_match:
                    cavity_num = int(cavity_match.group(1))
                    existing_cavity_files.append((cavity_num, str(file_path)))
        
        if existing_cavity_files:
            # Use existing files instead of extracting
            tqdm.write(f"Found {len(existing_cavity_files)} existing cavity files for '{tar_gz_filename}'. Skipping extraction.")
            for cavity_num, file_path in existing_cavity_files:
                cavity_files[cavity_num] = file_path
            skipped_count += 1
        else:
            # Extract cavity files
            try:
                with tarfile.open(full_tar_path, "r:gz") as tar:
                    # Find and extract only cavity files
                    for member in tar.getmembers():
                        # Match cavity files with any number: _cavity_1.pdb, _cavity_2.pdb, etc.
                        cavity_match = re.search(r'_cavity_(\d+)\.pdb$', member.name)
                        if cavity_match:
                            cavity_num = int(cavity_match.group(1))
                            tar.extract(member, path=extract_folder)
                            pocket_pdb = os.path.join(extract_folder, member.name)
                            cavity_files[cavity_num] = pocket_pdb
                extracted_count += 1
            
            except tarfile.ReadError as e:
                tqdm.write(f"Error reading tar.gz file '{tar_gz_filename}': {e}. Skipping.")
                continue
            except Exception as e:
                tqdm.write(f"An unexpected error occurred with '{tar_gz_filename}': {e}. Skipping.")
                continue
        # Add all cavity files to cavity_info
        cavities_found = 0
        for cavity_num, pocket_pdb in cavity_files.items():
            if uniprot_id not in cavity_info:
                cavity_info[uniprot_id] = []
            cavity_info[uniprot_id].append((pocket_pdb, fragment_id, cavity_num))
            cavities_found += 1
        
        if cavities_found == 0:
            tqdm.write(f"Warning: No cavity files found in '{tar_gz_filename}'. Skipping.")
        else:
            extraction_status = "existing" if existing_cavity_files else "extracted"
            tqdm.write(f"Found {cavities_found} cavity file(s) ({extraction_status}) for fragment F{fragment_id} in '{tar_gz_filename}'.")

    logging.info(f"Finished extracting cavity paths. Found info for {len(cavity_info)} UniProt IDs.")
    logging.info(f"Extraction statistics: {extracted_count} tarballs extracted, {skipped_count} tarballs skipped (already present)")
    return cavity_info

TARBALL_RE = re.compile(
    r'^AF-(?P<accession>[A-Z0-9]+)-F(?P<fragment>\d+)-model_v\d+_cavity_result_(?P<cavity>\d+)\.tar\.gz$'
)


def parse_tarball_name(filename):
    """Derive ``(accession, fragment_id, cavity_index)`` from a tarball's NAME.

    ``AF-B4DS77-F1-model_v1_cavity_result_2.tar.gz`` describes cavity 2 of
    fragment F1 of B4DS77, and the member it holds is
    ``AF-B4DS77-F1-model_v1_cavity_2.pdb``. The campaign's own log confirms the
    relation is one-to-one: 38,709 tarballs produced 38,709 cavity rows.

    That is what makes :func:`build_cavity_manifest` possible without unpacking,
    which is the whole 76.6 GB saving -- the old code unpacked all 38,709 tarballs
    because the required set could not be computed until the mapping existed, and
    the mapping was (wrongly) believed to require unpacking.

    Returns ``None`` for a name that does not match, so the caller can count and
    report unparseable tarballs rather than silently skipping them.
    """
    match = TARBALL_RE.match(os.path.basename(filename))
    if not match:
        return None
    return (
        match.group('accession'),
        'F%s' % int(match.group('fragment')),
        int(match.group('cavity')),
    )


def build_cavity_manifest(cavity_tarball_folder, verify_members=False):
    """Build the cavity manifest from tarball NAMES. Unpacks nothing.

    Args:
        cavity_tarball_folder: folder of ``*.tar.gz`` cavity archives.
        verify_members: open each archive and confirm it really holds the cavity
            the name advertises. Correct but slow (38,709 gzip opens); off by
            default because the naming relation is one-to-one and verified.

    Returns:
        ``(DataFrame, stats)`` where the frame carries UniProt_ID, Fragment_ID,
        Cavity_Index, Tarball, Pocket_PDB (the path extraction *would* produce),
        sorted so the output is byte-identical across runs and hash seeds.
    """
    if not os.path.isdir(cavity_tarball_folder):
        raise NotADirectoryError(
            "cavity tarball folder not found: %r" % cavity_tarball_folder
        )

    names = sorted(f for f in os.listdir(cavity_tarball_folder) if f.endswith('.tar.gz'))
    rows = []
    unparseable = []
    mismatched = []

    for name in names:
        parsed = parse_tarball_name(name)
        if parsed is None:
            unparseable.append(name)
            continue
        accession, fragment_id, cavity_index = parsed
        stem = name[: -len('.tar.gz')]
        member = '%s_cavity_%d.pdb' % (stem[: stem.rindex('_cavity_result_')], cavity_index)

        if verify_members:
            try:
                with tarfile.open(os.path.join(cavity_tarball_folder, name), 'r:gz') as tar:
                    holds = [
                        m.name for m in tar.getmembers()
                        if re.search(r'_cavity_(\d+)\.pdb$', m.name)
                    ]
                if not any(m.endswith('_cavity_%d.pdb' % cavity_index) for m in holds):
                    mismatched.append((name, holds))
                    continue
            except (tarfile.ReadError, OSError) as exc:
                mismatched.append((name, str(exc)))
                continue

        rows.append({
            'UniProt_ID': accession,
            'Fragment_ID': fragment_id,
            'Cavity_Index': cavity_index,
            'Tarball': name,
            'Pocket_PDB': os.path.join(EXTRACT_BASE_DIR, stem, member),
        })

    df = pd.DataFrame(
        rows,
        columns=['UniProt_ID', 'Fragment_ID', 'Cavity_Index', 'Tarball', 'Pocket_PDB'],
    )
    if not df.empty:
        df = df.sort_values(
            ['UniProt_ID', 'Fragment_ID', 'Cavity_Index'], kind='mergesort'
        ).reset_index(drop=True)

    stats = {
        'tarballs_seen': len(names),
        'rows': len(df),
        'accessions': int(df['UniProt_ID'].nunique()) if not df.empty else 0,
        'unparseable': unparseable,
        'mismatched': mismatched,
    }
    return df, stats


def _normalize_fragment(value):
    """``1``, ``'1'`` and ``'F1'`` all mean fragment F1.

    The old campaign wrote ``'F1'`` for positives and bare ``'1'`` for negatives;
    that mismatch is what made the sample-type merge non-unique.
    """
    text = str(value).strip().upper()
    if text.startswith('F'):
        text = text[1:]
    try:
        return 'F%d' % int(text)
    except (TypeError, ValueError):
        return 'F1'


def required_keys_from_frame(df):
    """``{(accession, fragment_id, cavity_index)}`` from any manifest-shaped frame.

    Accepts either the new column names or the schema's lowercase ones, so a
    ``redock_queue.csv`` and a ``cavity_manifest.csv`` can both be passed as the
    required set.
    """
    cols = {c.lower(): c for c in df.columns}
    acc = cols.get('uniprot_id') or cols.get('accession')
    frag = cols.get('fragment_id')
    cav = cols.get('cavity_index')
    if not acc or not cav:
        raise ValueError(
            'required-set frame needs an accession column (UniProt_ID or accession) '
            'and a Cavity_Index column; got %s' % list(df.columns)
        )
    keys = set()
    for _, row in df.iterrows():
        if pd.isna(row[acc]) or pd.isna(row[cav]):
            continue
        fragment = _normalize_fragment(row[frag]) if frag else 'F1'
        keys.add((str(row[acc]).strip(), fragment, int(row[cav])))
    return keys


def extract_required_cavities(cavity_tarball_folder, manifest, required_keys,
                              extract_base_dir=EXTRACT_BASE_DIR):
    """Unpack ONLY the cavities in ``required_keys``.

    ``required_keys`` is mandatory. There is deliberately no default and no
    fallback to "everything": a stage that silently substitutes a wider universe
    when its input is missing is the exact defect that produced 5,661 extracted
    AlphaFold directories for 979 needed ones, and 38,709 cavities for 2,829.

    Asserts on return that the number of cavities present equals the number
    required, so a partial extraction is loud rather than silent.
    """
    if required_keys is None:
        raise ValueError(
            'extract_required_cavities() requires an explicit required_keys set. '
            'A missing prerequisite is an error, never a fallback.'
        )
    required_keys = set(required_keys)
    os.makedirs(extract_base_dir, exist_ok=True)

    wanted = manifest[
        manifest.apply(
            lambda r: (str(r['UniProt_ID']), _normalize_fragment(r['Fragment_ID']),
                       int(r['Cavity_Index'])) in required_keys,
            axis=1,
        )
    ] if not manifest.empty else manifest

    missing_from_manifest = required_keys - (
        set() if manifest.empty else {
            (str(r['UniProt_ID']), _normalize_fragment(r['Fragment_ID']), int(r['Cavity_Index']))
            for _, r in manifest.iterrows()
        }
    )

    rows = []
    extracted = 0
    reused = 0
    failed = []

    iterator = wanted.iterrows() if not wanted.empty else iter([])
    for _, row in tqdm(list(iterator), desc='Extracting required cavities', unit='cavity'):
        stem = row['Tarball'][: -len('.tar.gz')]
        target_dir = os.path.join(extract_base_dir, stem)
        pocket = os.path.join(target_dir, os.path.basename(row['Pocket_PDB']))

        if os.path.exists(pocket):
            reused += 1
        else:
            os.makedirs(target_dir, exist_ok=True)
            try:
                with tarfile.open(
                    os.path.join(cavity_tarball_folder, row['Tarball']), 'r:gz'
                ) as tar:
                    for member in tar.getmembers():
                        if re.search(
                            r'_cavity_%d\.pdb$' % int(row['Cavity_Index']), member.name
                        ):
                            tar.extract(member, path=target_dir)
                            break
                extracted += 1
            except (tarfile.ReadError, OSError) as exc:
                failed.append((row['Tarball'], str(exc)))
                continue

        rows.append({
            'UniProt_ID': row['UniProt_ID'],
            'Fragment_ID': row['Fragment_ID'],
            'Cavity_Index': int(row['Cavity_Index']),
            'Pocket_PDB': pocket,
        })

    out = pd.DataFrame(
        rows, columns=['UniProt_ID', 'Fragment_ID', 'Cavity_Index', 'Pocket_PDB']
    )
    stats = {
        'required': len(required_keys),
        'extracted': extracted,
        'reused': reused,
        'present': len(out),
        'failed': failed,
        'missing_from_manifest': sorted(missing_from_manifest),
    }
    return out, stats


def save_cavity_mapping_to_csv(cavity_info_dict, output_csv):
    """
    Save the cavity mapping information to a CSV file for later use.
    
    Args:
        cavity_info_dict (dict): Dictionary mapping UniProt IDs to cavity files
        output_csv (str): Path to output CSV file
    """
    logging.info(f"Saving cavity mapping to '{output_csv}'...")
    
    rows = []
    for uniprot_id, cavity_list in cavity_info_dict.items():
        for pocket_pdb, fragment_id, cavity_index in cavity_list:
            rows.append({
                'UniProt_ID': uniprot_id,
                'Fragment_ID': fragment_id,
                'Cavity_Index': cavity_index,
                'Pocket_PDB': pocket_pdb
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    
    logging.info(f"Saved {len(rows)} cavity files for {len(cavity_info_dict)} UniProt IDs to '{output_csv}'.")
    return df

def load_cavity_mapping_from_csv(input_csv):
    """
    Load cavity mapping from a previously generated CSV file.
    
    Args:
        input_csv (str): Path to input CSV file
        
    Returns:
        dict: Dictionary mapping UniProt IDs to lists of (pocket_pdb, fragment_id, cavity_index) tuples
    """
    if not os.path.exists(input_csv):
        logging.error(f"Cavity mapping file not found: {input_csv}")
        return {}
    
    try:
        df = pd.read_csv(input_csv)
        required_columns = ['UniProt_ID', 'Fragment_ID', 'Pocket_PDB', 'Cavity_Index']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logging.error(f"Required columns missing from cavity mapping file: {missing_columns}")
            return {}
        
        cavity_info = {}
        for _, row in df.iterrows():
            uniprot_id = row['UniProt_ID']
            fragment_id = row['Fragment_ID']
            pocket_pdb = row['Pocket_PDB']
            cavity_index = row['Cavity_Index']
            
            if uniprot_id not in cavity_info:
                cavity_info[uniprot_id] = []
            cavity_info[uniprot_id].append((pocket_pdb, fragment_id, cavity_index))
        
        logging.info(f"Loaded cavity mapping for {len(cavity_info)} UniProt IDs from '{input_csv}'.")
        return cavity_info
        
    except Exception as e:
        logging.error(f"Error loading cavity mapping file: {e}")
        return {}


# ---------------------------------------------------------------------------
# CLI -- REBUILD_PLAN Phase 6
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Build the cavity manifest from tarball names, and unpack only the '
            'cavities an explicit required set asks for.'
        ),
        epilog=(
            'Two-step usage:\n'
            '  1. --manifest-only --tarballs DIR --out cavity_manifest.csv\n'
            '  2. --manifest cavity_manifest.csv --required redock_queue.csv '
            '--tarballs DIR --out-dir structures/cavities\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--tarballs', default=CAVITY_TARBALL_FOLDER,
                        help='folder of *.tar.gz cavity archives (default: %(default)s)')
    parser.add_argument('--manifest-only', action='store_true',
                        help='derive the manifest from tarball NAMES and exit. Unpacks nothing.')
    parser.add_argument('--verify-members', action='store_true',
                        help='with --manifest-only, open each archive to confirm its cavity (slow)')
    parser.add_argument('--manifest', default=None,
                        help='cavity_manifest.csv produced by --manifest-only')
    parser.add_argument('--required', default=None,
                        help=(
                            'CSV naming the cavities to unpack (redock_queue.csv, '
                            'positives_planned.csv, or any frame with an accession + '
                            'Cavity_Index column). MANDATORY for extraction.'
                        ))
    parser.add_argument('--out', default=OUTPUT_CSV,
                        help='where to write the manifest (default: %(default)s)')
    parser.add_argument('--out-dir', default=EXTRACT_BASE_DIR,
                        help='extraction destination (default: %(default)s)')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    start = time.time()

    if args.manifest_only:
        manifest, stats = build_cavity_manifest(args.tarballs, verify_members=args.verify_members)
        manifest.to_csv(args.out, index=False)
        logging.info('manifest: %d rows over %d accessions from %d tarballs (nothing unpacked)',
                     stats['rows'], stats['accessions'], stats['tarballs_seen'])
        if stats['unparseable']:
            logging.warning('%d tarball name(s) did not parse, e.g. %s',
                            len(stats['unparseable']), stats['unparseable'][:3])
        if stats['mismatched']:
            logging.warning('%d tarball(s) did not hold the cavity their name advertises',
                            len(stats['mismatched']))
        logging.info('wrote %s in %.1fs', args.out, time.time() - start)
        return 0

    if not args.required:
        logging.error(
            '--required is mandatory for extraction. Without it this script would '
            'unpack every tarball in the folder -- 38,709 archives and 76.6 GB for a '
            'campaign that needed 2,829. Run with --manifest-only first, compute the '
            'required set, then pass it here. A missing prerequisite is an error, '
            'never a fallback.'
        )
        return 2
    if not args.manifest:
        logging.error('--manifest is required for extraction; produce it with --manifest-only.')
        return 2

    try:
        manifest = pd.read_csv(args.manifest)
        required = required_keys_from_frame(pd.read_csv(args.required))
    except (OSError, ValueError) as exc:
        logging.error('%s', exc)
        return 2

    out, stats = extract_required_cavities(
        args.tarballs, manifest, required, extract_base_dir=args.out_dir
    )
    out.to_csv(args.out, index=False)

    logging.info('required %d cavities: %d newly extracted, %d already present, %d on disk',
                 stats['required'], stats['extracted'], stats['reused'], stats['present'])
    if stats['missing_from_manifest']:
        logging.error('%d required cavit(ies) are not in the manifest at all, e.g. %s',
                      len(stats['missing_from_manifest']), stats['missing_from_manifest'][:3])
    if stats['failed']:
        logging.error('%d tarball(s) failed to extract', len(stats['failed']))
    if stats['present'] != stats['required']:
        logging.error('INVARIANT: %d cavities required but %d present on disk',
                      stats['required'], stats['present'])
        return 3
    logging.info('wrote %s in %.1fs', args.out, time.time() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
