# dock_selectivity

A consensus docking pipeline. It docks small molecules into protein cavities with
up to four engines — smina, gnina, GOLD, LeDock — and measures how far apart the
engines place each ligand. Cross-tool pose agreement (RMSD) is the output
quantity, computed for a set of drug–target pairs and for a balanced set of
non-interacting pairs.

Inputs are DrugBank ligands, a drug–target edge list, and AlphaFold/CavitySpace
pockets. The output is one parquet table with a row per cross-tool pose pair,
labelled positive or negative.

Requirements, external programs and reference data: **`SETUP.md`**.
Architecture and the rules new code must not break: **`CLAUDE.md`**.

## Quick start

```bash
conda activate teachopencadd

DATA=/media/onur/Elements/cavity_space_consensus_docking
RUN=$DATA/runs/2026_10_11

python3 run_full_workflow.py \
    --inputs-dir $DATA/inputs \
    --run-dir    $RUN \
    --mode       balanced \
    --dry-run                       # prints every phase's command line; drop to execute
```

A full run takes days, almost all of it docking. Plan first and look at the queue
before committing to it:

```bash
python3 run_full_workflow.py --inputs-dir $DATA/inputs --run-dir $RUN --only planning
wc -l $RUN/manifests/redock_queue.csv      # jobs; budget ~57 MB each while docking
```

Every stage skips work already complete on disk, so a run can be interrupted and
resumed, and a partly finished directory is taken to completion rather than redone.

## Directory layout

Inputs are shared between runs; outputs belong to one run.

```
$DATA/inputs/            1_drug_to_protein.tsv, small_molecule_drug_links.csv,
                         ligands/, uniprot/
$DATA/runs/<date>/
    identity/            the accession index everything keys on
    manifests/           one CSV per planning stage; the chain between phases
    structures/          receptors: cavities, alphafold, fixed, pdbqt
    docking_results/     one directory per job
    steps/<phase>/       that phase's log and captured output
    logs/workflow.log    the runner's own log
```

## The pipeline

`run_full_workflow.py` runs these in order. Each is a normal script that also runs
standalone.

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

Phases 2 and 6a both run `extract_cavities.py`: the first derives pocket identity
from tarball filenames without unpacking anything (0.4 s for 38,709 archives), the
second unpacks only the pockets the queue names. Negatives are drawn so that each
drug and each target appears equally often in both classes.

## Running part of it

`--only` and `--skip` take a phase key from the table, or a group name: `planning`
(0–5, including 2b) or `structures` (6).

```bash
python3 run_full_workflow.py --run-dir $RUN --only planning
python3 run_full_workflow.py --run-dir $RUN --only structures --threads 16
python3 run_full_workflow.py --run-dir $RUN --skip parse
```

Every script documents its own arguments; there is no config file and nothing to
edit before a run.

```bash
python3 <script>.py --help
python3 batch_dock.py --print-config --num-threads 24    # resolved settings, no run
```

A typical standalone docking call:

```bash
python3 batch_dock.py \
    --queue   $RUN/manifests/redock_queue.csv \
    --run-dir $RUN \
    --ligands $DATA/inputs/ligands \
    --tools   smina,gnina,ledock \
    --num-threads 24 --smina-exhaustiveness 32
```

Useful flags: `--test-mode --max-jobs N` to smoke-test, `--complex-pdb
keep|delete|compress` (keep during a run — RMSD is computed from those files;
they are ~89% of the output size, so delete or compress once the parquet is built).

## Pocket quality

Phase 2b always runs and records every pocket's pLDDT. **Filtering on it is
opt-in.** A pocket passes when

```
frac_ge(--plddt-min) >= --min-plddt-fraction
```

where `frac_ge(t)` is the fraction of the pocket's residues with pLDDT ≥ t.
`--plddt-min` defaults to 70; `--min-plddt-fraction` has no default, and giving it
is what switches the filter on:

```bash
python3 run_full_workflow.py --run-dir $RUN --plddt-min 70 --min-plddt-fraction 0.70
```

The same thresholds reach positives and negatives — filtering one class and not
the other would let a classifier separate them on pocket confidence alone. Every
dropped pocket is recorded with the numbers behind it.
`POCKET_QUALITY_SWEEP.md` tabulates how many drug–target pairs survive at 25
threshold combinations.

## Notifications

Export two variables and pass `--notify`:

```bash
export PUSHOVER_TOKEN=<application token>
export PUSHOVER_USER_KEY=<user key>

python3 run_full_workflow.py --run-dir $RUN --notify
```

You get a message at the start, one naming the phase and exit code if anything
fails, and one at the end with timings. `--notify` with the variables unset exits
immediately rather than at hour 40, and `--dry-run --notify` checks the setup
without running anything. `--notify-steps` forwards `--notify` to each phase.
Account setup is in `SETUP.md`.

## gnina on a remote GPU

gnina needs a GPU. `gnina_remote.py` stands in for the binary and runs it over
SSH; `consensus_docker.py` cannot tell the difference.

```bash
export GNINA_PATH=$PWD/gnina_remote.py
export GNINA_REMOTE_HOST=user@gpu-host GNINA_REMOTE_PORT=50000
```

With `GNINA_ASYNC=1`, gnina jobs queue to the GPU while the other engines use
local cores, and `gnina_collect.py` files results as they land.

When that host is busy, the same shim runs the jobs on a Google Colab GPU against
a Colab Pro/Pro+ subscription:

```bash
export GNINA_REMOTE_BACKEND=colab GNINA_COLAB_GPU=T4
```

It needs the Colab CLI (`uv tool install google-colab-cli`) — not `gcloud`, which
manages a different product. A Colab GPU comes with very few vCPUs and gnina's
search is CPU-parallel, so it is much slower per job at high `--exhaustiveness`;
the backend clamps `--cpu` to the runtime and refuses an exhaustiveness that
cannot finish in time rather than lowering it silently. Both backends and the
measured timings are in `SETUP.md`.

## Logs

```
$RUN/logs/workflow.log                    the runner: plan, phase, exit code, elapsed
$RUN/steps/<phase>/<script>.log           that phase's own log
$RUN/steps/<phase>/<phase>.console.log    everything that phase printed
```

Nothing writes a log loose into the directory you started it from; a script run
with no `--run-dir` makes `./<step-name>/`. Every script takes `--out-dir`,
`--log-file`, `--log-level` and `--quiet`. For the four structure scripts
`--out-dir` is where the structures land, so their log is set with `--log-file`.

A failed phase still prints the tail of its output to the terminal.

## Checking a run

```bash
python3 audit.py identity --index $RUN/identity/identity_index.parquet
python3 audit.py curated  --dir $RUN/docking_results --manifest $RUN/manifests/run_manifest.csv
python3 audit.py run      --run-dir $RUN
```

Each exits non-zero on a failed invariant. Run them between phases.

## Identity and naming

Proteins are keyed on **UniProt accession**, resolved from the **Entrez gene id**.
Gene symbols are display labels derived from the accession, never resolution
inputs. Compounds are keyed on DrugBank ID, pockets on
`(accession, fragment, cavity)`.

Job directories are named `DB00035__P30518__AVPR2__F1__cav2`, and each one holds a
`job_identity.json` that is the authority on what was docked.

## Conventions

- A missing prerequisite is an error, never a fallback. No stage widens its input
  set when something is absent.
- Stages validate against `dock_selectivity.schemas` and record rejected rows with
  a reason instead of dropping them.
- Joins go through `checked_merge()`, which fails if a merge changes the row count.

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

Standard library only, no network, no GPU.
