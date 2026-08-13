# dock_selectivity

Consensus docking of DrugBank small molecules into AlphaFold/CavitySpace pockets.
Four engines (smina, gnina, GOLD, LeDock) dock each ligand–pocket pair, and
cross-tool pose agreement (RMSD) is tested as a discriminator of drug–target
interactions against balanced negatives.

Environment and external programs: **`SETUP.md`**.

## Quick start

```bash
DATA=/media/onur/Elements/cavity_space_consensus_docking

python3 run_full_workflow.py \
    --inputs-dir $DATA/inputs \
    --run-dir    $DATA/runs/2026_08_13 \
    --mode       balanced \
    --dry-run                       # prints the plan; drop to execute
```

Inputs are shared and versioned; outputs are per-run:

```
$DATA/inputs/     1_drug_to_protein.tsv, small_molecule_drug_links.csv,
                  ligands/, uniprot/
$DATA/runs/<date>/  identity/ manifests/ structures/ docking_results/ analysis/
```

Every stage skips work already complete on disk, so a run can be interrupted and
resumed, and a partially populated output directory is docked to completion
rather than redone.

## The pipeline, phase by phase

`run_full_workflow.py` runs these in order. Each is a normal script you can run
alone — useful when a phase needs different arguments, or when re-running one.

| # | Script | Reads | Writes |
|---|---|---|---|
| 0 | `build_identity_index.py` | UniProt idmapping + proteome | `identity/identity_index.parquet` |
| 1 | `resolve_interactions.py` | edges TSV, index | `manifests/interactions_resolved.csv` |
| 2 | `extract_cavities.py --manifest-only` | cavity tarball **names** | `manifests/cavity_manifest.csv` |
| 3 | `plan_positives.py` | resolved edges, cavities, ligands | `manifests/positives_planned.csv` |
| 4 | `sample_negatives.py` | positives, interactions | `manifests/negatives_drawn.csv` |
| 5 | `build_run_manifest.py` | positives, negatives | `manifests/run_manifest.csv`, `redock_queue.csv` |
| 6 | `extract_alphafold_models.py`, `fix_required_pdbs.py`, `convert_pdb_to_pdbqt.py`, `extract_cavities.py` | queue | `structures/` |
| 7 | `batch_dock.py --queue` | queue, structures, ligands | `docking_results/<job>/` |
| 8 | `parse_prepare.py` | docking results | `analysis/combined_results.parquet` |

**0 — identity.** Builds the accession ↔ Entrez ↔ symbol index everything else
keys on. Run once; downstream stages pin its SHA256.

**1 — interactions.** Resolves each drug–target edge to a UniProt accession *via
the Entrez gene id*. Unresolvable edges go to a rejects file with a reason.

**2 — cavity manifest.** Derives every cavity's identity from the tarball
filenames without unpacking anything (0.4 s for 38,709 archives).

**3 — positives.** Expands each resolved pair across every cavity of its target.
Drops pairs with no ligand SDF, no cavity, or a non-small-molecule drug — each
recorded with its reason.

**4 — negatives.** Balanced sampling: each drug and each target appears equally
in both classes. `--prefer-pool` reuses already-docked non-interactions.

**5 — manifest.** Merges both into the authority on what should exist, plus the
queue of what must actually be docked.

**6 — structures.** Extracts cavities, AlphaFold models, adds hydrogens, converts
to PDBQT. Each takes `--required` and materialises only what the queue asks for.

**7 — docking.** Five stages in order: smina → gnina → GOLD → LeDock → consensus
RMSD. `consensus_docker.py` is invoked once per tool per job; the RMSD stage
compares every pose of every tool pair into `final_results.csv`.

**8 — analysis.** Walks the job directories into one parquet, joining sample
types, IC50s, cavity clusters and DTA predictions.

## Running one phase

Each script documents its own arguments:

```bash
python3 <script>.py --help
```

`batch_dock.py` additionally has `--print-config`, which prints all 50 resolved
settings and marks the ones your command line changed:

```bash
python3 batch_dock.py --print-config --num-threads 24 --smina-exhaustiveness 8
```

Common docking invocation:

```bash
python3 batch_dock.py \
    --queue   $RUN/manifests/redock_queue.csv \
    --run-dir $RUN \
    --ligands $DATA/inputs/ligands \
    --tools   smina,gnina,ledock \
    --num-threads 24 --smina-exhaustiveness 32
```

Useful flags: `--test-mode --max-jobs N` (smoke test), `--no-force-rmsd` (skip
recomputation when poses were cleaned away), `--complex-pdb keep|delete|compress`.

## gnina on a remote GPU

gnina needs a GPU. `gnina_remote.py` stands in for the binary and runs it
elsewhere over SSH — `consensus_docker.py` cannot tell the difference:

```bash
export GNINA_PATH=$PWD/gnina_remote.py
export GNINA_REMOTE_HOST=user@gpu-host GNINA_REMOTE_PORT=50000
```

With `GNINA_ASYNC=1`, gnina jobs are queued to the GPU while smina/GOLD/LeDock
keep local cores busy; `gnina_collect.py` files results as they land and a
barrier before the RMSD stage waits for them. Details in `SETUP.md`.

## Checking a run

```bash
python3 audit.py identity --index $RUN/identity/identity_index.parquet
python3 audit.py curated  --dir $RUN/docking_results --manifest $RUN/manifests/run_manifest.csv
python3 audit.py run      --run-dir $RUN
```

Run these between phases; each exits non-zero on a failed invariant.

## Identity

Proteins are keyed on **UniProt accession**, resolved from the **Entrez gene id**
in `1_drug_to_protein.tsv`. Gene symbols are display labels derived from the
accession, never resolution inputs. Compounds are keyed on DrugBank ID; pockets on
`(accession, fragment, cavity)`.

Job directories are named `DB00035__P30518__AVPR2__F1__cav2`; the authority is
`job_identity.json` inside each one.

## Conventions

- A missing prerequisite is an error, never a fallback — no stage widens its input
  set when something is absent.
- Stages validate inputs and outputs against `dock_selectivity.schemas` and record
  rejected rows with a reason rather than dropping them.
- Joins go through `checked_merge()`, which fails if a merge changes the row count.
- `run_manifest.csv` is the authority on which jobs should exist; discovery is
  checked against it.

## Tests

```bash
python3 -m unittest discover -s tests -t .
```
