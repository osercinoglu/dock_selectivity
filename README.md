# dock_selectivity

Consensus docking of DrugBank small molecules into AlphaFold/CavitySpace pockets.
Four engines (smina, gnina, GOLD, LeDock) dock each ligand–pocket pair, and
cross-tool pose agreement (RMSD) is tested as a discriminator of drug–target
interactions against balanced negatives.

## Layout

Inputs are small, shared and versioned. Outputs are per-run.

```
<data>/inputs/                     1_drug_to_protein.tsv
                                   small_molecule_drug_links.csv
                                   ligands/          ligand SDFs
                                   uniprot/          idmapping + reference proteome
<data>/runs/<YYYY_MM_DD>/          identity/ manifests/ structures/
                                   docking_results/ analysis/ logs/
```

Large reference stores stay in `/opt/data`: `cavity_space/af_strong_cavities_data_all`
(cavity tarballs) and `alphafold2/UP000005640_9606_HUMAN_v4.tar`.

## Run

```bash
python3 run_full_workflow.py \
    --inputs-dir <data>/inputs \
    --run-dir    <data>/runs/2026_08_13 \
    --mode       balanced \
    --dry-run                      # prints the plan; drop to execute
```

Every stage skips work already complete in the output directory, so a run can be
interrupted and resumed, and a partially populated output directory is docked to
completion rather than redone.

| Phase | Stage | Produces |
|---|---|---|
| 0 | `build_identity_index.py` | `identity/identity_index.parquet` |
| 1 | `resolve_interactions.py` | `manifests/interactions_resolved.csv` |
| 2 | `extract_cavities.py --manifest-only` | `manifests/cavity_manifest.csv` |
| 3 | `plan_positives.py` | `manifests/positives_planned.csv` |
| 4 | `sample_negatives.py` | `manifests/negatives_drawn.csv` |
| 5 | `build_run_manifest.py` | `manifests/run_manifest.csv`, `redock_queue.csv` |
| 6 | `batch_dock.py --queue` | `docking_results/<job>/` |
| 7 | `parse_prepare.py` | `analysis/combined_results.parquet` |

Structure preparation (`extract_cavities.py`, `extract_alphafold_models.py`,
`fix_required_pdbs.py`, `convert_pdb_to_pdbqt.py`) each take `--required` and
materialise only what the queue asks for.

`audit.py` runs the invariant checks; use it between phases.

## Identity

Proteins are keyed on **UniProt accession**, resolved from the **Entrez Gene ID**
in `1_drug_to_protein.tsv` via `dock_selectivity.identity.IdentityIndex`. Gene
symbols are display labels derived from the accession, never resolution inputs.
Compounds are keyed on DrugBank ID; pockets on `(accession, fragment, cavity)`.

Job directories are named `DB00035__P30518__AVPR2__F1__cav2`. The authority is
`job_identity.json` inside each directory; the name is a convenience.

Phase 0 needs the UniProt id-mapping and reference proteome
(`inputs/uniprot/`). `build_identity_index.py --fetch` downloads them once.

## Conventions

- A missing prerequisite is an error, never a fallback. No stage widens its input
  set when something is absent.
- Every stage validates its inputs and outputs against `dock_selectivity.schemas`,
  and records rejected rows with a reason rather than dropping them.
- Joins go through `checked_merge()`, which fails if a merge changes the row count.
- `run_manifest.csv` is the authority on which jobs should exist; discovery is
  checked against it.

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

## Environment

conda env `teachopencadd` (Python 3.9): pandas, polars, pyarrow, numpy, tqdm,
rdkit, biopandas, pdbfixer/openmm, OpenBabel. Docking engines and
`consensus_docker.py` paths are set at the top of `batch_dock.py`.
