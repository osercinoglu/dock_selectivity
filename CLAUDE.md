# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A large-scale **consensus docking selectivity study**: dock DrugBank small molecules into AlphaFold-derived
cavities (CavitySpace) with 4 docking engines, measure cross-tool pose agreement (RMSD), and test whether
that agreement discriminates true drug–target interactions from balanced negatives. Two halves:

1. **Structure prep + docking** (`prepare_uniprot_mapping.py` → `batch_dock.py`) — documented step by step in `README.md`.
2. **Results parsing + analysis** (`parse_prepare.py`, `get_PM_scores.py`, `pocketmatch_clustering.py`, notebooks) — not in the README; see below.

## Critical: the repo is not the working directory

Scripts hold only code. All inputs, intermediates, outputs, and logs live in a separate **run directory**
on an external drive, currently:

```
/media/onur/Elements/cavity_space_consensus_docking/2025_06_29_batch_dock
```

Most scripts use **relative paths** for their mapping CSVs, so they must be invoked with that directory as
cwd. Some scripts additionally hardcode that path absolutely (`parse_prepare.py:BASE_DIR`,
`batch_dock.py:OUTPUT_BASE_FOLDER`, `create_ic50_mapping.py`, `pocketmatch_clustering.py --csv-dir` default,
notebook `DATA_FILE` constants) — when the run directory changes, all of these must be updated together.

Reference datasets live under `/opt/data/` (`drugbank/`, `ttd/`, `cavity_space/`, `alphafold2/`).

## Configuration model

**There is no config file and (mostly) no CLI.** Every pipeline script carries a `# --- Configuration ---`
block of module-level constants at the top, and that block is the only supported way to change behavior.
Changing a run means editing the script. Only `run_full_workflow.py`, `get_PM_scores.py`, and
`pocketmatch_clustering.py` take arguments.

Consequence: never assume a flag exists. Read the constants at the top of the file first.

## Environment

Conda env `teachopencadd` (Python 3.9). External binaries are hardcoded in `batch_dock.py` and must exist:

| Tool | Path constant |
|---|---|
| consensus docker driver | `CONSENSUS_DOCKER_SCRIPT` → `/home/onur/repos/consensus_docking/consensus_docker.py` (separate repo) |
| smina / gnina / gold / ledock+lepro | `SMINA_PATH`, `GNINA_PATH`, `GOLD_PATH`, `LEDOCK_PATH`, `LEPRO_PATH` |
| PocketMatch v2.1 | passed via `get_PM_scores.py -pm_dir` |

`batch_dock.py` never calls docking engines directly — it shells out to `consensus_docker.py` once per
tool per job. Behavior changes to a *single* docking run belong in that other repo, not here.

## Commands

```bash
# --- run everything from the run directory, not the repo ---
cd /media/onur/Elements/cavity_space_consensus_docking/2025_06_29_batch_dock
REPO=/home/onur/repos/dock_selectivity

# Docking pipeline, in order (see README.md for what each produces)
python $REPO/prepare_uniprot_mapping.py       # ⚠ known-broken, see Known issues
python $REPO/extract_cavities.py
python $REPO/identify_required_structures.py
python $REPO/generate_negative_samples.py     # optional; required for the ML framing
python $REPO/extract_alphafold_models.py
python $REPO/fix_required_pdbs.py
python $REPO/convert_pdb_to_pdbqt.py
python $REPO/batch_dock.py                    # long-running; use nohup

# Or the whole chain with skips
python $REPO/run_full_workflow.py --skip-uniprot --skip-extract

# Pocket similarity branch
python $REPO/get_PM_scores.py -cavities ./extracted_cavities -output ./pocketmatch_results \
    -pm_dir /home/onur/software/PocketMatch_v2.1/PocketMatch_v2.1 \
    -af_structures ./alphafold_structures -threads 8 --resume
python $REPO/pocketmatch_clustering.py --csv-dir ./pocketmatch_results/PocketMatch_similarities

# Aggregate all docking outputs into the analysis-ready parquet
python $REPO/parse_prepare.py
```

**Smoke-testing:** there is no test suite and no linter. The dry-run mechanism is `TEST_MODE = True` /
`MAX_TEST_JOBS` in `batch_dock.py`, `--test-mode --test-blocks N` in `get_PM_scores.py`. Prefer these over
running a full pass — a full `batch_dock.py` run is on the order of days.

## Architecture

### The mapping-CSV chain

Each prep step reads the previous step's CSV and writes a new one with additional resolved file-path
columns. This chain is the pipeline's actual interface between stages:

```
uniprot_gene_mapping.csv     (uniprot ↔ gene)
cavity_mapping.csv           (+ extracted cavity/vacant PDB paths)
  └─ required_structures.csv           (subset actually needed for docking)
       └─ required_structures_with_negatives.csv   (+ balanced negative pairs)
            ├─ alphafold_mapping.csv   (+ full AlphaFold model paths)
            │    └─ fixed_mapping.csv  (+ PDBFixer/hydrogen-added paths)
            │         └─ pdbqt_mapping.csv  (+ PDBQT paths)
            └─ consumed directly by batch_dock.py for the job list
```

Two behavioral rules fall out of this and are easy to break:

- **Receptor priority in `batch_dock.py`** is `fixed_mapping > alphafold_mapping > cavity_mapping`, with
  PDBQT substituted when `pdbqt_mapping.csv` has an entry (`load_best_mapping_with_pdbqt`).
- **Presence of `required_structures_with_negatives.csv` silently changes the docking mode.** With it,
  `batch_dock.py` docks only the exact (drug, target, cavity index) triples listed, including negatives.
  Without it, it falls back to `DRUG_TO_PROTEIN_TSV` and docks every cavity of every known target.

### batch_dock.py — staged execution

Jobs run in ordered stages (smina → gnina → gold → ledock → RMSD), each independently toggled by
`USE_SMINA` / `USE_GNINA` / `USE_GOLD` / `USE_LEDOCK` / `USE_RMSD_CALCULATION` and each with its own
parallelism and timeout constants. Parallelism differs by engine because smina/gnina consume
`EXHAUSTIVENESS` cores per job while gold/ledock are single-threaded (hence 2 vs 60 concurrent jobs).

Special mode: when `USE_RMSD_CALCULATION` is true and **all four docking flags are false**, the script
skips job planning entirely and runs `run_rmsd_only_workflow_from_existing_outputs(OUTPUT_BASE_FOLDER)` —
re-deriving consensus RMSD over already-docked output folders. This is the mode the current committed
config is in.

Resume is universal and implicit: every stage checks for valid existing output
(`check_tool_completion`, `check_final_results_completion`) and skips. `FORCE_RMSD_RECALCULATION` and
`UPDATE_SMINA_ONLY` are the escape hatches that delete selected outputs before re-running.

### Job folder naming is a load-bearing contract

`batch_dock.py` writes one directory per job as:

```
{drugbank_id}_{gene_name}_{uniprot_id}_cavity_{n}/
```

`parse_prepare.py` re-parses drug/target/cavity identity back out of that name with
`r'(DB\d+)_([A-Za-z0-9]+)_([A-Z0-9]+)_cavity_(\d+)'`. Gene name may be the literal string `nan`. Changing
the naming on either side without the other silently drops records.

### parse_prepare.py — phase order matters

Discovers `final_results.csv` / per-tool `results.csv` across the whole output tree (multiprocess walk),
then: load → clean → **annotate sample types → filter to complete tool coverage → balance pos/neg** →
*then* the expensive annotations (DTA-ATLAS affinity predictions, cavity clusters, IC50 from TTD, Louvain
pocket clusters) → export.

Filtering/balancing deliberately happens **before** enrichment to avoid OOM on the full result set; moving
an annotation earlier will blow memory. Output is a single
`combined_filtered_annotated_docking_results.parquet` plus a timestamped JSON summary.

### Pocket similarity branch

`get_PM_scores.py` runs PocketMatch `Step3-PM_typeB` in isolated temp workspaces, blocked (`--block-size`
pockets per cabbage) and chunked (`--chunk-rows` per output CSV) so an all-pairs comparison over tens of
thousands of pockets stays tractable; `--resume` continues from existing blocks/chunks.
`pocketmatch_clustering.py` then converts those CSVs to parquet and runs Louvain community detection at
several Pmax thresholds, writing `pocket_clusters/`, which `parse_prepare.py`'s
`integrate_louvain_pocket_clusters` joins back onto the docking results.

### Notebooks

All consume `combined_filtered_annotated_docking_results.parquet` (Polars) — they are downstream of
`parse_prepare.py` and do no aggregation of raw docking output themselves. Roughly newest-first:
`analysis_2025_12_24.ipynb` (classification metrics: PR/ROC over positive vs `negative_balanced`),
`research_questions_analysis.ipynb` (the RQ set: tool reliability, cavity preference, discrimination),
`analysis_pocketmatch.ipynb` (pocket similarity/cluster visualization). The dated
`analysis_2025_05_*` / `analysis_2025_06_10` / `pdb_vs_alphafold` notebooks are an earlier 90-ligand pilot
under `/home/ssahin/...` paths and are historical.

## Known issues

- **`prepare_uniprot_mapping.py` produces a corrupted `uniprot_gene_mapping.csv`** — 57.7% of rows carry a
  gene name belonging to a different protein. Read `KNOWN_ISSUE_uniprot_gene_mapping.md` before touching
  anything that joins on `Mapped_Gene_Name` (`batch_dock.py`, `create_ic50_mapping.py`,
  `generate_negative_samples.py`, `identify_required_structures.py`). Unfixed as of 2026-08-12.
- **`pocketmatch_clustering.py:426` has a stray `tmux` token after a string literal** — the file does not
  parse. Fix before running it.
- `fix_pdb_files.py` is empty (0 bytes); the working script is `fix_required_pdbs.py`.

## Conventions

- Commit messages follow `feat: ...` / `fix: ...`, and reference issues as `- fixes #N.`
- New pipeline scripts follow the house pattern: shebang, module docstring listing prerequisites/inputs/
  outputs, `# --- Configuration ---` constants, logging to both a named `*.log` file and stdout, `tqdm`
  progress, and resume-by-checking-for-existing-output.
