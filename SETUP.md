# Setup

## Python environment

Conda, Python 3.9 (`environment.yml` in this repo builds it):

```bash
conda env create -f environment.yml
conda activate dock_selectivity
```

Core packages: `pandas`, `polars`, `pyarrow`, `numpy`, `tqdm`, `psutil`,
`networkx`, `python-louvain`, `rdkit`, `biopandas`, `pdbfixer`, `openmm`,
`openbabel`, `requests`, `bioservices` is **not** used and must not be added
(see the G6 lint in `tests/`).

Tests use the standard library `unittest` only:

```bash
python3 -m unittest discover -s tests -t .
```

### Notifications (optional)

Any script, and the workflow runner, will send Pushover notifications with
`--notify`. The credentials come from the environment only — never a command-line
flag, never a file in this repository, never a log record:

1. Create an account at [pushover.net](https://pushover.net) and install the app
   on the device you want woken up. Your **user key** is on the dashboard.
2. On the same dashboard, *Create an Application/API Token* (any name, e.g.
   `dock_selectivity`). That gives the **application token**.
3. Export both, in the shell you start runs from — put them in `~/.bashrc` or a
   file you keep outside the repo, not in `environment.yml`:

```bash
export PUSHOVER_TOKEN=<application token>
export PUSHOVER_USER_KEY=<user key>
export PUSHOVER_DEVICE=<device name>       # optional; all devices otherwise
```

4. Check them without running anything:

```bash
python3 run_full_workflow.py --run-dir $RUN --dry-run --notify
```

`--notify` with either variable unset exits 2 at startup rather than at hour 40.
A notification that fails to send is logged as a warning and never stops a run.

## External programs

The pipeline shells out to `consensus_docker.py`, which in turn drives the
docking engines. Each binary is found in this order: the environment variable
below, then `$PATH`, then `~/software/<binary>`. Each also has a flag
(`--smina-path`, `--gnina-path`, …), and an *enabled* engine whose binary is
missing aborts the run with exit 2 naming the engine, the variable and the flag —
rather than failing inside a subprocess an hour later.

| Program | Variable | Notes |
|---|---|---|
| `consensus_docker.py` | `CONSENSUS_DOCKER_SCRIPT` | separate repo, `github.com/…/consensus_docking` |
| smina | `SMINA_PATH` | conda: `conda install -c conda-forge smina` |
| gnina | `GNINA_PATH` | **needs CUDA 12 + cuDNN 9 and a GPU** |
| GOLD | `GOLD_PATH` | CCDC licence required |
| LeDock / LePro | `LEDOCK_PATH`, `LEPRO_PATH` | static binaries from the LeDock site |

Check what is actually available before a run:

```bash
python3 - <<'EOF'
import os, batch_dock as b
for name in ("CONSENSUS_DOCKER_SCRIPT","SMINA_PATH","GNINA_PATH","GOLD_PATH",
             "LEDOCK_PATH","LEPRO_PATH"):
    p = getattr(b, name)
    print("%-24s %s  %s" % (name, "OK     " if os.path.exists(p) else "MISSING", p))
EOF
```

Select engines per run with `--tools`, e.g. `--tools smina,ledock`. RMSD is always
computed.

### gnina — local or remote GPU

gnina is GPU-only. On a machine with no CUDA runtime the binary fails at load
(`libcudart.so.12` / `libcudnn.so.9` not found) and there is no CPU fallback.

`gnina_remote.py` runs it on a remote GPU host instead. It presents gnina's exact
command line, ships the receptor and ligand over SSH, runs the container there and
writes `out.sdf` / `out.log` back to the local paths it was given — so
`consensus_docker.py` cannot tell the difference and nothing else changes:

```bash
export GNINA_PATH=/home/onur/repos/dock_selectivity/gnina_remote.py
export GNINA_REMOTE_HOST=user@gpu-host
export GNINA_REMOTE_PORT=50000
python3 batch_dock.py --queue … --tools smina,gnina,ledock
```

| Variable | Default | Meaning |
|---|---|---|
| `GNINA_REMOTE_HOST` | — | `user@host`, **required** |
| `GNINA_REMOTE_PORT` | 22 | ssh port |
| `GNINA_REMOTE_IMAGE` | `gnina/gnina:v1.3.3` | container to run |
| `GNINA_REMOTE_DIR` | `gnina_jobs` | remote scratch (relative to `$HOME`) |
| `GNINA_REMOTE_KEY` | — | private key, if not the default |
| `GNINA_REMOTE_NATIVE` | — | path to a native remote gnina; skips the container |

Auth is SSH keys only — `ssh-copy-id -p <port> ~/.ssh/id_ed25519.pub user@host`
once. Passwords are unsupported on purpose: this runs unattended per job, and
`sshpass` would put the secret in the process list or on disk.

Remote host requirements: an NVIDIA GPU with a current driver, Docker, and
`nvidia-container-toolkit` so `docker run --gpus all` works. Verify with:

```bash
ssh -p <port> user@host 'docker run --rm --gpus all \
    nvidia/cuda:12.9.0-base-ubuntu24.04 nvidia-smi --query-gpu=name --format=csv,noheader'
```

Then `docker pull gnina/gnina:v1.3.3` (23.6 GB download, ~37.5 GB on disk).

Measured on an RTX 3060 Ti (8 GB): ~60 s per job at `--num_modes 5
--exhaustiveness 8`, ~2.5 min at production settings, with ~300 KB up and ~1–2 MB
down.

**Surviving a flaky link.** A campaign is thousands of jobs over days, so the
connection will drop. Three mechanisms handle it:

- The remote job runs **detached** (`setsid nohup`) and writes its exit code to a
  marker file, so losing the connection mid-run does not kill the container. The
  shim polls for that marker and keeps waiting through outages.
- Every SSH operation **retries with exponential backoff**, but only for transient
  failures (connection refused/reset/timeout, DNS, ssh exit 255). A missing input
  or a non-zero gnina exit fails immediately instead of burning five retries.
- The remote directory is **content-addressed** from the inputs and arguments, so
  re-running a job finds finished work and collects it rather than docking again.

Verified against the live host:

| Scenario | Result |
|---|---|
| Cold run | 77.5 s |
| Re-run, same inputs | **6.2 s**, byte-identical output |
| Local process `SIGKILL`ed mid-run | remote `docker run` kept going; re-run collected the result in 43.4 s |

Tuning: `GNINA_REMOTE_RETRIES` (default 5), `GNINA_REMOTE_TIMEOUT` (default 7200 s
to wait for the detached job), `GNINA_REMOTE_KEEP=1` to leave remote job
directories in place.

If a job exceeds `GNINA_REMOTE_TIMEOUT` the shim exits non-zero but leaves the
remote directory intact, and says so — re-running that job collects the result.
Remote directories are removed only after a successful download.

### gnina — on a Colab GPU instead

When the SSH host's single GPU is busy, `GNINA_REMOTE_BACKEND=colab` runs the
same jobs on a Google Colab runtime against a Colab Pro/Pro+ subscription. Same
shim, same command line, same exit codes, same content-addressed resume — only
the transport and how gnina gets there differ.

**This is not `gcloud`.** `gcloud colab` manages *Colab Enterprise*, a Vertex AI
product billed to a GCP project; it cannot spend a consumer subscription's
compute units. The tool that can is the separate Colab CLI:

```bash
uv tool install google-colab-cli          # or: pip install google-colab-cli

# ADC is the headless path. The colaboratory scope matters; re-mint if a call 401s.
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory

colab --auth=adc usage                    # confirms entitlement and balance
```

Then point the shim at it:

```bash
export GNINA_PATH=/home/onur/repos/dock_selectivity/gnina_remote.py
export GNINA_REMOTE_BACKEND=colab
export GNINA_COLAB_GPU=T4                 # T4 | L4 | G4 | A100 | H100
python3 batch_dock.py --queue … --tools smina,gnina,ledock
```

| Variable | Default | Meaning |
|---|---|---|
| `GNINA_REMOTE_BACKEND` | `ssh` | `colab` selects this backend |
| `GNINA_COLAB_SESSION` | `gnina` | session name; jobs share one runtime |
| `GNINA_COLAB_GPU` | `T4` | an unlisted value is refused, not passed on |
| `GNINA_COLAB_AUTH` | `adc` | or `oauth2` (needs a browser) |
| `GNINA_COLAB_CONFIG` | — | CLI state file, to isolate parallel runs |
| `GNINA_COLAB_STOP` | — | `1` releases the runtime after each job |
| `GNINA_COLAB_KEEP` | — | `1` leaves the job directory on the runtime |
| `GNINA_COLAB_BINARY_URL` | v1.3.3 release | override the gnina asset |

**How gnina gets there.** Colab has no Docker, so the container is not an option.
The release binary is fetched *on the runtime* — 2.06 GB at ~74 MB/s, about 28 s,
far quicker than uploading 2 GB from here. It is called
`gnina.cuda12.8.static` but is not fully static: current Colab images ship CUDA
13, so `libcudart.so.12`, `libcufft.so.11`, `libcublas.so.12` and
`libcublasLt.so.12` are missing at the system level. They are present as pip
wheels, and the backend finds each one and composes `LD_LIBRARY_PATH` at run
time rather than hardcoding a path that carries the runtime's Python version.
The download happens once per session; a marker file makes later jobs free.

**The bottleneck is vCPUs, not the GPU.** gnina's Monte Carlo search is
CPU-parallel and only the CNN rescoring uses the card, and a Colab GPU runtime
comes with very few vCPUs. Measured on the same job (DB00035 / P09211, box
30×29×31 Å):

| Host | vCPUs | `--exhaustiveness` | Wall time |
|---|---|---|---|
| SSH host, RTX 3060 Ti | 24 | 32 | 2 m 23 s |
| Colab T4, GPU active | 2 | 8 | 22 m 17 s (1337 s) |
| Colab T4, **CPU fallback** | 2 | 32 | > 32 min |

From the middle row: 1337 x 2 / 8 = **334 core-seconds per exhaustiveness unit**,
against `remote_caps`' 112.5 measured on the SSH host. The Colab backend carries
its own constant (344, slightly conservative) because an under-predicting deadline
model admits a job and then misses the deadline, which is the worse direction.
Override with `GNINA_COLAB_CORE_SECONDS` after measuring another accelerator.

That second row is a bug, not a benchmark, and it is recorded because it was
expensive to spot. Composing `LD_LIBRARY_PATH` from the CUDA 12 wheel directories
alone *replaced* the runtime's own value, `/usr/lib64-nvidia` — which is where
`libcuda.so.1`, the driver library, lives. gnina then started normally, logged
`No GPU detected` to its run log, and finished the job on two CPU cores, on a
runtime billed by the hour for a GPU it never touched. Nothing else in the stack
noticed; the poses would have been perfectly valid.

Two guards now exist because of it: the library path is always *extended*, never
replaced, and both the installer and the per-job probe treat `No GPU detected` as
a hard failure rather than a warning.

So the backend applies the same two rules `remote_caps` applies to the SSH host:
`--cpu` is **clamped loudly** to the runtime's vCPU count (`consensus_docker.py`
passes `--cpu 60`; asking 60 threads of 2 vCPUs thrashes), and an
`--exhaustiveness` the model says cannot finish inside `GNINA_REMOTE_TIMEOUT` is
**refused, never quietly lowered** — lowering it changes what the docking is, and
a campaign whose jobs used different values is not comparable.

**Billing.** A runtime bills for wall-clock time it is *allocated*, not time it
computes: a measured **1.07 compute units per hour** for a T4. An idle session
costs exactly what a busy one does. Jobs therefore share one named session, and
`GNINA_COLAB_STOP=1` is for one-shot runs; a campaign stops the session once at
the end (`colab stop -s gnina`). Check the balance with `colab --auth=adc usage`.
At 1.07 CU/hr a 600-unit monthly Pro+ allowance is about 560 GPU-hours, which is
why exhaustiveness matters here in a way it does not on a host you own. What that
buys, for a queue of 13,222 gnina jobs and the measured 2-vCPU runtime:

| `--exhaustiveness` | Per job (2 vCPU) | CU per job | Jobs per 600 CU | Jobs per 1,784 CU |
|---|---|---|---|---|
| 32 (production) | ~89 min | 1.59 | ~380 | ~1,100 |
| 16 | ~45 min | 0.79 | ~760 | ~2,240 |
| 8 (measured) | 22 min | 0.40 | ~1,510 | ~4,500 |

Against a queue of 13,222 gnina jobs, Colab at production exhaustiveness is about
**8% of a campaign for the whole current balance**. It is a way to keep a few
hundred to a few thousand jobs moving while the SSH host is occupied, not a
replacement for it — one RTX 3060 Ti with 24 cores does a production job in 2 m 23
s, which is 37x faster than this T4 with 2 vCPUs.

Lowering exhaustiveness to fit is a scientific decision, not a throughput one:
jobs docked at different values are not comparable, which is why the backend
refuses rather than adjusting it.

**One GPU, one job.** `gnina_queue` sets `GNINA_REMOTE_LOCK=1` on every submit;
this backend honours it with a `mkdir`-atomic mutex in the job root, released by
the detached job itself after it writes its exit code. Two gnina processes on one
T4 contend for VRAM and both finish later than if they had queued.

**What survives a disconnection.** The run is launched with `setsid nohup` and
its exit code goes to a marker file, exactly as on the SSH host — so a dropped
websocket, an interrupted local process or a `colab exec` timeout loses the
connection, not the run. Observed while measuring the table above: the local
`colab exec` client was `SIGTERM`ed 30 minutes into a run and the runtime's
kernel carried on executing, with `colab status` still reporting `BUSY`. That run
was *not* detached — it was a foreground `subprocess.run` inside the cell — so
the detached path this backend uses is strictly more robust than what that
already demonstrated. Polling resumes against the same content-addressed
directory, and a finished job is collected rather than re-docked. What does *not*
survive is the runtime being pruned by the backend: the job directory goes with
it, and `collect` reports `gone` so the job is docked again.

## Reference data

Large, shared, not copied per run:

```
/opt/data/cavity_space/af_strong_cavities_data_all     cavity tarballs (7.0 GB)
/opt/data/alphafold2/UP000005640_9606_HUMAN_v4.tar     AlphaFold archive (4.8 GB)
/opt/data/drugbank/small_molecule_drug_links.csv       DrugBank vocabulary
```

Per-project inputs (~72 MB), in `<data>/inputs/`:

```
1_drug_to_protein.tsv          drug–target edges; node_2 is the Entrez Gene ID
small_molecule_drug_links.csv
ligands/                       one SDF per DrugBank ID
uniprot/HUMAN_9606_idmapping.dat.gz
uniprot/UP000005640_9606.fasta.gz
```

The two UniProt files are fetched once:

```bash
python3 build_identity_index.py --fetch --out <run>/identity/identity_index.parquet
```

or downloaded manually from
`ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/`
(`idmapping/by_organism/HUMAN_9606_idmapping.dat.gz` and
`reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz`).

## Storage

`complex_*.pdb` files dominate output size (~89% of a job). **Keep them during a
run**: `consensus_docker.py` computes RMSD from them, so removing them makes RMSD
non-recomputable for that job. `batch_dock.py --complex-pdb delete|compress` is an
archival step for after the analysis table is built.

Budget roughly 45 MB per job with complexes, ~1 MB without.

Phase 2b (`score_cavity_quality.py`) reads all 38,709 cavity archives once: about
six CPU-minutes, and the CSV it writes is about 20 MB (measured at ~0.5 KB per
pocket, most of it the per-residue column). Leave `--threads` at 1 — threads were
measured and make it *slower* on local disk (0.7 ms/archive at 1 thread against
2.3 ms at 16), because each archive is small and the tar-header parsing is pure
Python. Raise it only for a filesystem where an open blocks, and only after
measuring there.

## Pilot

A two-drug, two-target run exercises the whole chain including the external
programs, in a few minutes:

```bash
P=<data>/pilot
python3 run_full_workflow.py --inputs-dir $P/inputs --run-dir $P/run --mode balanced \
    --skip dock --skip parse            # phases 0-6: plan and build the receptors
python3 batch_dock.py --queue $P/run/manifests/redock_queue.csv --run-dir $P/run \
    --ligands $P/inputs/ligands --tools smina,ledock --test-mode --max-jobs 1
```

Four jobs are planned and their two receptors materialised in about 40 s; the
docking step is what takes the remaining minutes.

Expected: `docking_results/<job>/final_results.csv` with one row per cross-tool
pose pair (400 rows for a 20×20 smina/LeDock comparison), and `job_identity.json`
naming the accession that was docked.
