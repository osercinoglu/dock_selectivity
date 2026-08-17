# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A large-scale **consensus docking selectivity study**: dock DrugBank small molecules into AlphaFold-derived
cavities (CavitySpace) with 4 docking engines, measure cross-tool pose agreement (RMSD), and test whether
that agreement discriminates true drug–target interactions from balanced negatives. Two halves:

1. **Planning → structure prep → docking** (`build_identity_index.py` → `batch_dock.py`), driven by
   `run_full_workflow.py` — documented phase by phase in `README.md`.
2. **Results parsing + analysis** (`parse_prepare.py`, `get_PM_scores.py`, `pocketmatch_clustering.py`,
   notebooks) — phase 8 is in the README; the pocket-similarity branch is only here.

## Critical: the repo is not the working directory

Scripts hold only code. Inputs, intermediates, outputs and logs live in a **run directory** passed as
`--run-dir`; the small reusable inputs live in a separate `--inputs-dir`, so inputs are shared and
versioned while outputs are per-run. Nothing is resolved relative to the invocation cwd any more, and no
script names a run directory of its own — the 2025-06-29 campaign path that used to be hardcoded in
`parse_prepare.py`, `batch_dock.py`, `create_ic50_mapping.py` and `pocketmatch_clustering.py` is gone, and
repo-wide AST lints in `tests/` fail if one comes back.

```
<run-dir>/identity/  manifests/  structures/  docking_results/  steps/  logs/
```

That older campaign directory still exists and is **read-only reference material**, not a working tree.

Reference datasets live under `/opt/data/` (`drugbank/`, `ttd/`, `cavity_space/`, `alphafold2/`). They are
the only absolute defaults left in the repo, and each one has a flag that overrides it.

## Configuration model

**Everything is a command-line argument.** There is no config file, and editing a script is no longer how
a run is configured — every path, threshold, worker count and toggle that used to be a module constant is
now a flag with a `help=` string. `python3 <script>.py --help` is the authority on what a script accepts;
`batch_dock.py --print-config` additionally prints all resolved settings and marks the ones your command
line changed.

Constants that remain in the source are **implementation detail, not configuration**: column vocabularies,
regexes, schema contracts, loop-termination guards. Turning one into a flag would let an operator break a
join, not tune a run.

Every step shares one runtime contract, `dock_selectivity/runtime.py`:

| Flag | Meaning |
|---|---|
| `--out-dir DIR` | that step's own directory for its log (see the exception below) |
| `--log-file PATH` | override the log file; default `<out-dir>/<step>.log` |
| `--log-level LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR`, default INFO |
| `--quiet` | file log only, nothing on the console |
| `--notify` | opt in to Pushover for this run (off by default) |
| `--notify-on EVENT` | repeatable: `start` / `success` / `failure` / `all`, default `failure` |
| `--notify-title T` | notification title |

Rules that fall out of it, and that new code must not break:

- **Import configures nothing.** Logging is set up in `main()` after parsing, never at import. No
  `logging.basicConfig()` anywhere; scripts log through `logging.getLogger(STEP_NAME)`, never
  `logging.info(...)`, because the runtime attaches handlers to the step logger and the `dock_selectivity`
  package logger and never to root.
- **A step's log lives in the step's own directory**, never in the invocation cwd and never beside the
  artifacts another stage enumerates:
  ```
  <run-dir>/logs/workflow.log                   run_full_workflow.py itself
  <run-dir>/steps/<phase>/<script>.log          each phase's own log
  <run-dir>/steps/<phase>/<phase>.console.log   what that phase printed
  ```
  Standalone with no `--run-dir`, a step makes `./<step-name>/` and logs there — `artifact_step_dir()`
  in the runtime, because the four structure scripts cannot derive it from `--out-dir`. Deriving it
  from `--out` instead is what used to drop `extract_cavities.log` into the repository root on every
  test run. Note the basename: those four log as `<phase>.log`, not `<script>.log`, since the runner
  sets it with `--log-file`.
- **Artifacts did not move.** `manifests/`, `structures/`, `docking_results/` and the analysis parquet keep
  their paths and their own `--out` flags. `batch_dock.py` resolves each mapping as
  `<run-dir>/manifests/<basename>`.
- **Exception, and it bites:** `extract_cavities.py`, `extract_alphafold_models.py`, `fix_required_pdbs.py`
  and `convert_pdb_to_pdbqt.py` owned `--out-dir` first, and for them it means *where the structures land*.
  Those four take `--log-file` for their step log; `run_full_workflow.py` does exactly that.
- **Notification credentials come from the environment only** — `PUSHOVER_TOKEN`, `PUSHOVER_USER_KEY`,
  optional `PUSHOVER_DEVICE`. Never a flag, never a file in the repo. `--notify` with the variables unset
  exits 2 at startup rather than at hour 61, and a failed notification is a WARNING, never a lost run.
- **A missing prerequisite is an error, never a fallback.** Scripts refuse and exit 2 instead of widening
  their input set or defaulting to a reference dataset.

## Environment

Conda env `teachopencadd` (Python 3.9). External binaries are resolved per run, in this order: the
environment variable, then `$PATH`, then `~/software/<binary>` — no account's home directory is baked in
any more. Each also has a flag (`--smina-path`, `--gnina-path`, …), and an **enabled** engine whose binary
is absent aborts the run with exit 2 naming the engine, the variable and the flag.

| Tool | Variable / flag |
|---|---|
| consensus docker driver | `CONSENSUS_DOCKER_SCRIPT` → `consensus_docker.py` (separate repo) |
| smina / gnina / gold / ledock+lepro | `SMINA_PATH`, `GNINA_PATH`, `GOLD_PATH`, `LEDOCK_PATH`, `LEPRO_PATH` |
| PocketMatch v2.1 | `get_PM_scores.py --pm-dir` |
| notifications (optional) | `PUSHOVER_TOKEN`, `PUSHOVER_USER_KEY`, `PUSHOVER_DEVICE` |

`batch_dock.py` never calls docking engines directly — it shells out to `consensus_docker.py` once per
tool per job. Behavior changes to a *single* docking run belong in that other repo, not here.

## Commands

```bash
REPO=/home/onur/repos/dock_selectivity
RUN=/media/onur/Elements/cavity_space_consensus_docking/runs/<date>

# The whole chain (README.md says what each phase produces)
python3 $REPO/run_full_workflow.py --inputs-dir $DATA/inputs --run-dir $RUN --mode balanced --dry-run
python3 $REPO/run_full_workflow.py --inputs-dir $DATA/inputs --run-dir $RUN --mode balanced --notify

# Part of it: --only/--skip take a phase key or the groups 'planning' / 'structures'
python3 $REPO/run_full_workflow.py --run-dir $RUN --only planning
python3 $REPO/run_full_workflow.py --run-dir $RUN --skip dock --skip parse

# Any phase standalone; every script documents itself
python3 $REPO/batch_dock.py --queue $RUN/manifests/redock_queue.csv --run-dir $RUN \
    --ligands $DATA/inputs/ligands --tools smina,ledock --out-dir $RUN/steps/dock
python3 $REPO/batch_dock.py --print-config --num-threads 24     # resolved settings, no run

# Invariant checks between phases; each exits non-zero on a failure
python3 $REPO/audit.py run --run-dir $RUN

# Pocket similarity branch (not part of run_full_workflow.py)
python3 $REPO/get_PM_scores.py --cavities $RUN/structures/cavities \
    --output-dir $RUN/pocketmatch_results --pm-dir ~/software/PocketMatch_v2.1/PocketMatch_v2.1 \
    --af-structures $RUN/structures/alphafold --threads 8 --resume
python3 $REPO/pocketmatch_clustering.py --run-dir $RUN
```

**Testing:** `python3 -m unittest discover -s tests -t .` — ~1,490 tests, standard library only, no
network. Two of them are repo-wide AST lints: every argparse argument must carry a `help=` string
(`tests/test_argument_help_coverage.py`) and no source file may name a dead run directory or resolve an
identity by free-text search (`tests/test_g6_no_freetext_resolution.py`).

**Smoke-testing a real run:** `--dry-run` on the workflow prints every phase's exact command line and
writes nothing; `batch_dock.py --test-mode --max-jobs N` and `get_PM_scores.py --test-mode
--test-blocks N` cap the work. Prefer these over a full pass — a full `batch_dock.py` run is on the order
of days.

## Architecture

### The manifest chain

Each phase reads the previous phase's file and writes a new one with more resolved columns. This chain is
the pipeline's actual interface between stages, and `run_full_workflow.py` is its only ordering authority:

```
identity/identity_index.parquet          (accession ↔ Entrez ↔ symbol; downstream stages pin its SHA256)
manifests/interactions_resolved.csv      (+ resolved drug–target edges, rejects recorded with a reason)
manifests/cavity_manifest.csv            (every cavity's identity, from tarball names, nothing unpacked)
  └─ positives_planned.csv               (each pair × every cavity of its target)
       └─ negatives_drawn.csv            (balanced negative sampling)
            └─ run_manifest.csv          (authority on which jobs should exist)
                 └─ redock_queue.csv     (what must actually be docked; every later phase's --required)
                      ├─ cavity_mapping.csv    → alphafold_mapping.csv → fixed_mapping.csv
                      │                                              └─ pdbqt_mapping.csv
                      └─ consumed by batch_dock.py --queue for the job list
```

Rules that fall out of this and are easy to break:

- **Receptor priority in `batch_dock.py`** is `fixed_mapping > alphafold_mapping > cavity_mapping`, with
  PDBQT substituted when `pdbqt_mapping.csv` has an entry (`load_best_mapping_with_pdbqt`). The mappings
  are resolved as `<run-dir>/manifests/<basename>`, so a structure phase writing anywhere else silently
  docks against whatever older mapping is on the path.
- **The docking mode is explicit.** `batch_dock.py --queue` docks exactly the listed
  (drug, target, cavity) triples, and that is what the workflow passes. The old behaviour — infer the mode
  from whether `required_structures_with_negatives.csv` happened to exist, else fall back to
  `DRUG_TO_PROTEIN_TSV` and dock every cavity of every known target — survives only on the legacy planner
  path taken when no `--queue` is given.
- **`required_structures*.csv` is the legacy vocabulary.** `identify_required_structures.py` and
  `generate_negative_samples.py` produce it, are superseded (see Known issues), and refuse to run without
  an explicit opt-in.

### batch_dock.py — staged execution

Jobs run in ordered stages (smina → gnina → gold → ledock → RMSD), selected with `--tools`, each with its
own parallelism and timeout flag. Parallelism differs by engine because smina/gnina consume
`EXHAUSTIVENESS` cores per job while gold/ledock are single-threaded (hence 2 vs 60 concurrent jobs).

Special mode: `--tools rmsd` (no docking engine, RMSD on) skips job planning entirely and runs
`run_rmsd_only_workflow_from_existing_outputs()` over the resolved output base — re-deriving consensus
RMSD across already-docked job folders. The output base comes from `--run-dir` (results go to
`<run-dir>/docking_results`) or `--output-base`; there is deliberately **no default**, and giving neither
exits 2 rather than writing into a past campaign.

Resume is universal and implicit: every stage checks for valid existing output
(`check_tool_completion`, `check_final_results_completion`) and skips. `FORCE_RMSD_RECALCULATION` and
`UPDATE_SMINA_ONLY` are the escape hatches that delete selected outputs before re-running.

### Job folder naming is a load-bearing contract

In queue mode the job directory name is chosen upstream by `build_run_manifest.py` and encoded by
`dock_selectivity/jobid.py`:

```
DB00035__P30518__AVPR2__F1__cav2/       drug __ accession __ gene __ fragment __ cavity
```

The authority for a job's identity is `job_identity.json` inside the directory, not the name; `jobid.py`
round-trips `format_job_dir` / `parse_job_dir` and has no fallback to the legacy scheme, because guessing
which scheme a name uses is how one pocket key matched several pockets.

The legacy planner path still writes `{drugbank_id}_{gene_name}_{uniprot_id}_cavity_{n}/`, which
`parse_prepare.py` re-parses with `r'(DB\d+)_([A-Za-z0-9]+)_([A-Z0-9]+)_cavity_(\d+)'` and where the gene
name may be the literal string `nan`. Changing the naming on either side without the other silently drops
records.

### parse_prepare.py — phase order matters

Discovers `final_results.csv` / per-tool `results.csv` under `--results-dir` (default
`<run-dir>/docking_results`, a multiprocess walk), then: load → clean → **annotate sample types → filter to
complete tool coverage → balance pos/neg** → *then* the expensive annotations (DTA-ATLAS affinity
predictions, cavity clusters, IC50 from TTD, Louvain pocket clusters) → export.

Filtering/balancing deliberately happens **before** enrichment to avoid OOM on the full result set; moving
an annotation earlier will blow memory. Output is a single
`combined_filtered_annotated_docking_results.parquet` plus a timestamped JSON summary beside it.

Two things that used to be silent are now checked: discovery is compared against `--manifest`
(`run_manifest.csv`) and a shortfall raises unless `--allow-discovery-shortfall`, and `--file-cap`
defaults to no cap — the 10,000 that used to be hardcoded lost most of a campaign without saying so.

### Pocket similarity branch

`get_PM_scores.py` runs PocketMatch `Step3-PM_typeB` in isolated temp workspaces, blocked (`--block-size`
pockets per cabbage) and chunked (`--chunk-rows` per output CSV) so an all-pairs comparison over tens of
thousands of pockets stays tractable; `--resume` continues from existing blocks/chunks, and every timeout
in it is a flag. Its old single-dash spellings (`-cavities`, `-output`, `-pm_dir`, `-af_structures`,
`-threads`) still work as aliases of the kebab-case ones. `pocketmatch_clustering.py --run-dir` then
converts those CSVs to parquet and runs Louvain community detection at several Pmax thresholds, writing
`pocket_clusters/`, which `parse_prepare.py --pocket-clusters` joins back onto the docking results.

### Notebooks

All consume `combined_filtered_annotated_docking_results.parquet` (Polars) — they are downstream of
`parse_prepare.py` and do no aggregation of raw docking output themselves. Roughly newest-first:
`analysis_2025_12_24.ipynb` (classification metrics: PR/ROC over positive vs `negative_balanced`),
`research_questions_analysis.ipynb` (the RQ set: tool reliability, cavity preference, discrimination),
`analysis_pocketmatch.ipynb` (pocket similarity/cluster visualization). The dated
`analysis_2025_05_*` / `analysis_2025_06_10` / `pdb_vs_alphafold` notebooks are an earlier 90-ligand pilot
under `/home/ssahin/...` paths and are historical.

## Known issues

- **The identity corruption that motivated the rebuild.** `prepare_uniprot_mapping.py` resolved proteins by
  free-text search and produced a `uniprot_gene_mapping.csv` in which 57.73% of rows carry a gene name
  belonging to a different protein. The script is deleted and `build_identity_index.py` replaces it, but
  the corrupted CSV still exists in the 2025-06-29 campaign and anything that joins on `Mapped_Gene_Name`
  inherits the defect. Read `KNOWN_ISSUE_uniprot_gene_mapping.md` before reusing that campaign's results.
- **Two superseded stages are kept runnable on purpose.** `identify_required_structures.py` and
  `generate_negative_samples.py` consume that corrupted mapping; they are retained only to reproduce the
  old campaign bug-for-bug, carry a DEPRECATED banner, and refuse to run without an explicit opt-in
  (`tests/test_superseded_stages.py` pins the refusal). Use `plan_positives.py` / `sample_negatives.py`.
- `download_cavityspace_structures.py` still imports `requests` — the only third-party HTTP client left in
  the repo. It is superseded by `extract_alphafold_models.py` (phase 6b) and untested; porting it to
  `urllib` is a clean follow-up.
- Documentation citing *line numbers* in other modules (e.g. `sample_negatives.py:10` naming
  `generate_negative_samples.py:190`) went stale when those files were rewritten. The file names are still
  right; the offsets are not.

## Conventions

- Commit messages follow `feat: ...` / `fix: ...`, and reference issues as `- fixes #N.`
- New pipeline scripts follow the house pattern: shebang; module docstring listing purpose, inputs,
  outputs and a real `Usage` block; `build_parser()` with a `help=` string on **every** argument, ending
  with `add_runtime_arguments(parser, STEP_NAME)`; `resolve_out_dir(...)` immediately after `parse_args`;
  the body inside `with step(STEP_NAME, args) as ctx`; `LOG = logging.getLogger(STEP_NAME)` at module
  level; `tqdm` progress; and resume-by-checking-for-existing-output.
- `main()` returns an exit code, but a non-zero one must be raised as `SystemExit(n)` *inside* the `step()`
  block and converted back in `main()` — `step()` reads a plain return as success, so a `return 2` would be
  announced to Pushover as a finished run.
- A step that fails must say what it needed. Exit 2 for a missing prerequisite, and create nothing on the
  way out.
