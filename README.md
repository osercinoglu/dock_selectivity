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
$DATA/runs/<date>/  identity/ manifests/ structures/ docking_results/
                    steps/ logs/          ← logs, one folder per phase
```

Every stage skips work already complete on disk, so a run can be interrupted and
resumed, and a partially populated output directory is docked to completion
rather than redone.

## Where the logs are

Nothing writes a log file loose into the directory you started it from:

```
$RUN/logs/workflow.log                    the runner: plan, phase, exit code, elapsed
$RUN/steps/<phase>/<script>.log           that phase's own log
$RUN/steps/<phase>/<phase>.console.log    everything that phase printed
```

Run a script standalone with no `--run-dir` and it makes `./<step-name>/` and logs
there. The four structure scripts name that log after the *phase* rather than the
script, because the runner sets it with `--log-file`.

The runner captures each phase's stdout and stderr into that console log instead
of twelve scripts interleaving progress bars in one terminal — but a failed phase
still quotes the tail of its output into `workflow.log` and onto your screen.

Every script takes `--out-dir` (its own folder), `--log-file`, `--log-level` and
`--quiet`. The four structure scripts are the exception: they owned `--out-dir`
first and it means *where the structures land*, so their log is set with
`--log-file` — which is what `run_full_workflow.py` passes them.

## The pipeline, phase by phase

`run_full_workflow.py` runs these in order. Each is a normal script you can run
alone — useful when a phase needs different arguments, or when re-running one.

| # | Script | Reads | Writes |
|---|---|---|---|
| 0 | `build_identity_index.py` | UniProt idmapping + proteome | `identity/identity_index.parquet` |
| 1 | `resolve_interactions.py` | edges TSV, index | `manifests/interactions_resolved.csv` |
| 2 | `extract_cavities.py --manifest-only` | cavity tarball **names** | `manifests/cavity_manifest.csv` |
| 2b | `score_cavity_quality.py` | cavity manifest, cavity tarballs | `manifests/cavity_quality.csv` |
| 3 | `plan_positives.py` | resolved edges, cavities, ligands | `manifests/positives_planned.csv` |
| 4 | `sample_negatives.py` | positives, interactions | `manifests/negatives_drawn.csv` |
| 5 | `build_run_manifest.py` | positives, negatives | `manifests/run_manifest.csv`, `redock_queue.csv` |
| 6a | `extract_cavities.py` | cavity manifest, queue | `structures/cavities/`, `cavity_mapping.csv` |
| 6b | `extract_alphafold_models.py` | queue | `structures/alphafold/`, `alphafold_mapping.csv` |
| 6c | `fix_required_pdbs.py` | queue, alphafold mapping | `structures/fixed/`, `fixed_mapping.csv` |
| 6d | `convert_pdb_to_pdbqt.py` | queue, fixed mapping | `structures/pdbqt/`, `pdbqt_mapping.csv` |
| 7 | `batch_dock.py --queue` | queue, structures, ligands | `docking_results/<job>/` |
| 8 | `parse_prepare.py` | docking results, manifest | `combined_filtered_annotated_docking_results.parquet` |

**0 — identity.** Builds the accession ↔ Entrez ↔ symbol index everything else
keys on. Run once; downstream stages pin its SHA256.

**1 — interactions.** Resolves each drug–target edge to a UniProt accession *via
the Entrez gene id*. Unresolvable edges go to a rejects file with a reason.

**2 — cavity manifest.** Derives every cavity's identity from the tarball
filenames without unpacking anything (0.4 s for 38,709 archives).

**2b — pocket quality.** A CavitySpace cavity file is the pocket-lining residues
of an AlphaFold model, and those residues keep the model's pLDDT in the B-factor
column. This phase reads it: one row per pocket with `n_residues`, the pLDDT
distribution, and `residue_plddt` — every residue's value, so any later cutoff is
a string split rather than a second pass over 7 GB of archives. Mean pocket pLDDT
runs from 34 to 99, so a run that never looks docks into confident pockets and
into disordered guesses with equal weight. About 6 CPU-minutes for all 38,709
archives; a cavity that cannot be scored is recorded in
`cavity_quality_rejected.csv` with a reason, never written as a zero row.

**3 — positives.** Expands each resolved pair across every cavity of its target.
Drops pairs with no ligand SDF, no cavity, or a non-small-molecule drug — each
recorded with its reason.

**4 — negatives.** Balanced sampling: each drug and each target appears equally
in both classes. `--prefer-pool` reuses already-docked non-interactions.

**5 — manifest.** Merges both into the authority on what should exist, plus the
queue of what must actually be docked.

**6 — structures.** Extracts cavities and AlphaFold models, adds hydrogens,
converts to PDBQT. Each takes `--required` and materialises only what the queue
asks for — the four steps run in order because each consumes the previous one's
mapping. `batch_dock.py` picks receptors in the order fixed > alphafold > cavity,
substituting PDBQT where one exists.

**7 — docking.** Five stages in order: smina → gnina → GOLD → LeDock → consensus
RMSD. `consensus_docker.py` is invoked once per tool per job; the RMSD stage
compares every pose of every tool pair into `final_results.csv`.

**8 — analysis.** Walks the job directories into one parquet, joining sample
types, IC50s, cavity clusters and DTA predictions. Discovery is checked against
`run_manifest.csv`: finding fewer jobs than the manifest declares stops the parse
instead of quietly producing a smaller table.

## Filtering pockets on model confidence

Scoring always happens; **filtering is opt-in.** A pocket *passes* when

```
frac_ge(--plddt-min) >= --min-plddt-fraction
```

where `frac_ge(t)` is the fraction of the pocket's residues whose pLDDT is at
least `t`. `--plddt-min` defaults to 70 (AlphaFold's "confident" band);
`--min-plddt-fraction` has **no default**, and supplying it is what turns the
filter on:

```bash
python3 run_full_workflow.py --run-dir $RUN --plddt-min 70 --min-plddt-fraction 0.80
```

Without `--min-plddt-fraction`, phases 3 and 4 are handed no quality arguments at
all and plan exactly what they would have planned before this phase existed. With
it, `plan_positives.py` and `sample_negatives.py` both get the same thresholds —
one class filtered and the other not is a confound, not a filter, because a
classifier could then separate the classes on pLDDT alone. Each stage records
every pocket it drops with reason `pocket_quality` and the numbers behind it
(`frac_ge_70=0.42 < 0.80 (n_residues=61)`), and a pocket the manifest names but
`cavity_quality.csv` does not carry stops the stage rather than silently passing
or silently vanishing (`--allow-missing-quality` downgrades that to a warning and
treats those pockets as failing).

`audit.py run` closes the loop: it checks that every pocket in `run_manifest.csv`
is scored and clears the thresholds recorded in `negatives_drawn.report.json`, so
a disagreement between what was filtered and what was docked surfaces there rather
than in the analysis.

## Being told when it breaks

A three-day run that dies at hour 40 should not wait for you to notice. Set two
environment variables — never a flag, never a file in this repo — and pass
`--notify`:

```bash
export PUSHOVER_TOKEN=<application token>     # pushover.net → Create an Application
export PUSHOVER_USER_KEY=<your user key>      # shown on your pushover.net dashboard
export PUSHOVER_DEVICE=<device name>          # optional; all devices otherwise

python3 run_full_workflow.py --run-dir $RUN --notify
```

You get one message when the run starts, one naming the phase and its exit code
if anything fails, and one at the end with the total time and a per-phase
summary. `--notify` with the variables unset exits 2 **before** the first phase,
so a broken notification setup is discovered in seconds rather than at hour 40;
`--dry-run --notify` is a way to check them without running anything.

`--notify-steps` additionally passes `--notify` down to every phase script, which
then reports on itself (failures only). Off by default — otherwise a 12-phase run
notifies twelve more times. Any individual script takes `--notify`,
`--notify-on start|success|failure|all` and `--notify-title` the same way. A
notification that cannot be delivered is logged as a warning and never stops the
run. See `SETUP.md` for the Pushover account setup.

## Running part of the pipeline

`--only` and `--skip` take a phase key from the table, or a group name —
`planning` (phases 0–5, including 2b) and `structures` (phase 6):

```bash
python3 run_full_workflow.py --run-dir $RUN --only planning     # plan, don't dock
python3 run_full_workflow.py --run-dir $RUN --only structures --threads 16
python3 run_full_workflow.py --run-dir $RUN --skip parse
```

Each script also runs standalone and documents its own arguments:

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
