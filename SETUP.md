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

## External programs

The pipeline shells out to `consensus_docker.py`, which in turn drives the
docking engines. Paths are set at the top of `batch_dock.py`.

| Program | Constant | Notes |
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
