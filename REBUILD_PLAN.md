# REBUILD PLAN — clean-slate experiment directory with only correct data

**Supersedes** the "what do we do about it" half of `ASSESSMENT_blast_radius_2026_08_12.md`.
**Depends on** `FIX_PLAN.md` for the code changes; this document says *when to apply each one and what
to run next*. Written for manual, step-by-step execution.

**Old run directory** — `/media/onur/Elements/cavity_space_consensus_docking/2025_06_29_batch_dock`
(1.55 TB). It is **read-only from here on**. Nothing in this plan deletes or modifies it. It is the
only copy of 14,893 core-hours of docking and it stays intact until you personally decide otherwise.

**New run directory** — `/media/onur/Elements/cavity_space_consensus_docking/2026_08_13_clean_rebuild`
Referred to below as `$NEW`. Free space on the drive: 5.3 TB.

---

## 1. The governing decision: what is allowed to carry over

### 1.1 What the old campaign actually contains

Every one of the 24,173 job directories is *physically* valid — a real ligand docked into a real cavity
of a real protein — and the **accession in the folder name is the receptor that was actually loaded**
(verified: 0 of 24,173 jobs have a receptor path that disagrees with their accession). Only the
*identity claim* is wrong. So the question is never "is this job good?" but "what is this job actually
a measurement of, and does the corrected design want it?"

Re-derived under correct identity:

| Class | Jobs | What it really is |
|---|---:|---|
| Labelled positive, label matches accession, genuine DTI | 6,987 | **true positive** |
| Labelled positive, label wrong, but drug *does* bind the docked protein | 308 | **true positive** (accidentally) |
| Labelled positive, label wrong, drug does *not* bind the docked protein | 5,072 | **valid non-interaction** |
| Labelled negative, not a known interaction | 11,796 | **valid non-interaction** |
| Labelled negative, actually a known interaction | 5 | **true positive** |

**Positives: 7,300 jobs / 2,086 genuine pairs. Valid non-interactions: 16,868 jobs.**

### 1.2 The correction to make about the 5,072

The instinct that "docking output that doesn't classify as a balanced negative may not be needed" is
right in direction but resolves the other way. Those 5,072 jobs are *not* a third category — under
correct identity they are exactly the same kind of measurement as the 11,796 sampled negatives: a drug
docked into a protein it has no annotated interaction with. What distinguishes them is only that
**nobody chose them** — they are an accident of the corruption rather than a draw from a designed
sampler.

That matters for the *design*, not for the *validity*. So:

> **Do not classify them up front. Put all 16,868 valid non-interactions into a negative candidate
> pool, re-run the balanced sampler against the corrected positive set, and let it draw. Whatever it
> draws is carried over for free. Whatever it does not draw is surplus and simply is not copied.**

This is strictly better than discarding them: the sampler gets a 16,868-job pool to satisfy balance
from instead of 11,796, which reduces how much new docking is needed. It is also strictly safer than
keeping them: nothing enters `$NEW` that the design did not ask for.

**Why the old negatives cannot simply be reused as-is.** They were drawn to balance a positive set that
was 45.8% wrong, so the balance guarantee holds only in corrupted-label space. In true protein space
only 30.8% of drugs and 39.1% of targets are balanced. The sampler **must** be re-run.

### 1.3 The 2.4% rule — what a carried-over job consists of

A job directory is ~47 MB. Measured composition of a representative job:

| Component | Size | Carry over? |
|---|---:|---|
| `final_results.csv` (consensus Tool1/Tool2/Score1/Score2/RMSD) | 184 KB | **yes** — the payload |
| `{smina,gnina,gold,ledock}/results.csv` | ~100 KB | **yes** |
| `smina/out.sdf`, `gnina/out.sdf` | ~440 KB | **yes** — sole carrier of gnina `CNN_VS`, `CNNaffinity_variance`, and unrounded scores |
| `gold/output/` (gold_soln poses) | ~180 KB | **yes** |
| `ledock/out.dok` | ~37 KB | **yes** |
| `input/{DB*.mol2, receptor.mol2}` | 198 KB | **yes** — cheap, makes each job self-describing |
| `{tool}/complex_*.pdb` (~20 per tool) | **~45 MB** | **no** — derived (receptor + pose), 99.75% byte-identical between poses, ligand coords reproduce `out.sdf` exactly |
| `log_*.log` (~13 retry logs per job) | 60 KB | **no** — provenance goes in `job_identity.json` |

**Evidence per job ≈ 1.14 MB, i.e. 2.4%.** Carrying ~14,600 jobs costs **~17 GB** instead of ~690 GB.
The `complex_*.pdb` files are not lost — they remain in the old directory, and are reconstructible from
`receptor.mol2` + `out.sdf` if ever needed.

### 1.4 Folder naming — make the old failure structurally impossible

Old: `DB00035_AVPR2_Q8NCT1_cavity_2` — a single-underscore string in which the gene and the accession
came from opposite sides of the bug and could disagree; it had no fragment field (causing the 410,087-row
merge fan-out) and its parser broke on 1,777 names.

New: **`DB00035__P30518__AVPR2__F1__cav2`**

- `__` double-underscore delimiter — hyphenated symbols (`MT-ND1`) and odd characters can no longer break the parse.
- **Accession before gene**, and the gene is written by `IdentityIndex.label(accession)` at creation time, so it cannot disagree by construction.
- **Fragment is explicit** (`F1`), closing the multi-fragment gap.
- The name is a convenience. The authority is `job_identity.json` inside every directory:

```json
{"drugbank_id":"DB00035","accession":"P30518","gene":"AVPR2","fragment_id":"F1",
 "cavity_index":2,"sample_type":"positive","identity_index_sha256":"…",
 "code_revision":"…","provenance":"salvaged_from|newly_docked",
 "source_job":"DB00035_AVPR2_Q8NCT1_cavity_2"}
```

`provenance` and `source_job` make every carried-over job traceable to the directory it came from, so
the relabelling is auditable rather than a leap of faith.

---

## 2. New run directory layout

```
$NEW/
  identity/
    identity_index.parquet          # THE root artifact — build first, nothing runs without it
    identity_index.meta.json        # source URLs, UniProt release, SHA256s, invariant report
    unresolved_targets.csv          # every edge that did NOT resolve, with a reason. Never silent.
  manifests/
    interactions_resolved.csv       # (drugbank_id, accession, gene) — corrected DTI edge list
    positives_planned.csv           # (drug, accession, fragment, cavity) positives wanted
    negatives_pool.csv              # 16,868 salvageable non-interaction jobs
    negatives_drawn.csv             # what the balanced sampler actually selected
    salvage_map.csv                 # old_job_dir -> new_job_dir, or a reason for exclusion
    redock_queue.csv                # jobs that must be docked fresh
    run_manifest.csv                # every job that SHOULD exist. Discovery is checked against this.
  structures/
    cavities/                       # only the required cavities (~2,900, not 38,709)
    alphafold/                      # only the required accessions (~890, not 5,661)
    fixed/  pdbqt/
  ligands/                          # only the required SDFs
  docking_results/                  # curated + newly docked jobs, new naming scheme
  analysis/
    combined_results.parquet
    provenance.json                 # index SHA256 + code revision that produced it
  logs/
```

Two rules the layout enforces: **`manifests/run_manifest.csv` is the authority on what should exist**
(so a truncated discovery pass is detectable rather than silent), and **`identity/` is built once and
pinned by SHA256 everywhere** (so no stage can invent its own identity).

---

## 3. Runbook

Conventions: `$REPO=/home/onur/repos/dock_selectivity`,
`$OLD=/media/onur/Elements/cavity_space_consensus_docking/2025_06_29_batch_dock`,
`$NEW=/media/onur/Elements/cavity_space_consensus_docking/2026_08_13_clean_rebuild`.
Every phase has a **STOP** gate — do not proceed until its checks pass.

---

### Phase 0 — Identity. Nothing else may run first.

**Code prerequisite:** `FIX_PLAN.md` §1 (`dock_selectivity/identity.py`, `build_identity_index.py`) and §2
(delete `prepare_uniprot_mapping.py`).

```bash
mkdir -p $NEW/{identity,manifests,structures,ligands,docking_results,analysis,logs}

# One-time network fetch, checksummed, then offline forever
python $REPO/build_identity_index.py \
    --fetch \
    --out $NEW/identity/identity_index.parquet
```

**STOP — verify before proceeding:**
```bash
python $REPO/audit.py identity --index $NEW/identity/identity_index.parquet
```
must report:
- `max_collapse == 1` (the old map was 2.55 mean, 9 max — this is the guarantee whose absence caused everything)
- `AVPR2 -> P30518`, `TUBA1B -> P68363`, `IL2 -> P60568`
- `primary_symbol ∈ symbols` for every row
- two builds under different `PYTHONHASHSEED` are byte-identical
- `n_in == n_resolved + n_rejected`, with rejects listed in `unresolved_targets.csv`

If any check fails, stop. Everything downstream inherits this file.

---

### Phase 1 — Resolve the corrected interaction set

**Code prerequisite:** `FIX_PLAN.md` §3.1 (`identify_required_structures.py` resolves via Entrez, not symbol).

```bash
python $REPO/resolve_interactions.py \
    --edges /opt/data/multiscale_interactome_data/1_drug_to_protein.tsv \
    --index $NEW/identity/identity_index.parquet \
    --drugs /opt/data/drugbank/small_molecule_drug_links.csv \
    --out   $NEW/manifests/interactions_resolved.csv
```

Resolution is on `node_2` (**Entrez ID**, numeric on 8,568/8,568 edges), never `node_2_name`.

**STOP — verify:** `n_edges_in == n_resolved + n_rejected`; spot-check that `AVPR2` edges now carry
P30518 and not Q8NCT1; confirm the 237 null-symbol edges are now resolvable via Entrez (they never were
before) and that the 222 name-slug `node_1` rows are either resolved or explicitly rejected.

---

### Phase 2 — Classify every old job under correct identity (read-only)

```bash
python $REPO/salvage_scan.py \
    --old-results $OLD/consensus_docking_results \
    --index       $NEW/identity/identity_index.parquet \
    --interactions $NEW/manifests/interactions_resolved.csv \
    --out-map     $NEW/manifests/salvage_map.csv \
    --out-pool    $NEW/manifests/negatives_pool.csv
```

Parses all 24,173 directory names **right-anchored** (strip trailing `_cavity_<N>`, strip optional
`_-F<k>-model_v1`, then test each token against the accession set — this resolves 24,173/24,173, unlike
the old regex which failed on 1,777). Reads nothing but names and `final_results.csv`.

**STOP — verify** `salvage_map.csv` reproduces the expected classification:

| Expected | Count |
|---|---:|
| true positives | 7,300 |
| valid non-interactions (negative pool) | 16,868 |
| unparseable | 0 |
| jobs missing `final_results.csv` | report it; these are incomplete, not salvageable |

---

### Phase 3 — Re-run the balanced sampler against the corrected positives

**Code prerequisite:** `FIX_PLAN.md` §3.2 (`generate_negative_samples.py` builds its avoid-list in
accession space).

```bash
python $REPO/generate_negative_samples.py \
    --index        $NEW/identity/identity_index.parquet \
    --interactions $NEW/manifests/interactions_resolved.csv \
    --positives    $NEW/manifests/positives_planned.csv \
    --pool         $NEW/manifests/negatives_pool.csv \
    --prefer-pool \
    --out          $NEW/manifests/negatives_drawn.csv
```

`--prefer-pool` is the whole point: satisfy the Najm balance constraint from already-docked jobs first,
and only emit a re-dock request when the pool cannot supply a needed `(drug, accession, cavity)`.

**STOP — verify:** balance holds **in accession space** (not label space): every drug and every target
appears in equal numbers of positives and negatives, `max|pos-neg| <= 1`. Contamination check returns
**0** negatives that are known interactions. Report what fraction of drawn negatives came from the pool
(expect a large majority — 62% of old negatives already sit on accessions in the corrected pool).

---

### Phase 4 — DECISION POINT: how much to re-dock

`redock_queue.csv` now exists. Three options; **B is recommended**:

| | A — salvage only | **B — salvage + fill the gap** | C — full re-run |
|---|---|---|---|
| New docking jobs | 0 | ~13,900 | ~36,000 |
| Wall-clock (64 cores) | 0 | **6–8 days** | 16–20 days |
| New storage (evidence-only retention) | ~17 GB | ~33 GB | ~45 GB |
| Intended-pair coverage | 25.4% | **63–69%** | 63–69% — *identical* |
| Existing compute reused | 100% | 100% | ~0% |

**C buys no coverage over B** — it re-docks the 2,100 pairs that are already correctly docked and
re-buys ~7,300 reusable negatives at full price. Its only gain is uniform provenance, which
`job_identity.json` already records per job.

A hard ceiling to know about: 2,030 intended pairs have **no CavitySpace cavity for their correct
protein**. No amount of re-docking reaches them; 69.1% is the realistic maximum.

---

### Phase 5 — Materialise the curated directory

**Code prerequisite:** the new naming scheme + `job_identity.json` writer (`FIX_PLAN.md` §3.3).

```bash
# Dry run FIRST — prints what it would copy, writes nothing
python $REPO/curate_jobs.py --dry-run \
    --salvage-map $NEW/manifests/salvage_map.csv \
    --drawn       $NEW/manifests/negatives_drawn.csv \
    --positives   $NEW/manifests/positives_planned.csv \
    --src $OLD/consensus_docking_results \
    --dst $NEW/docking_results \
    --evidence-only

python $REPO/curate_jobs.py  ...same flags, minus --dry-run...
```

`--evidence-only` applies the 2.4% rule from §1.3: copies `final_results.csv`, the four `results.csv`,
`out.sdf`/`out.dok`/gold poses and `input/*.mol2`; **skips `complex_*.pdb` and the retry logs**. Writes
`job_identity.json` into every new directory, with `source_job` naming the old directory.

Copy — never move, never hardlink. `$OLD` must remain a complete, independent, untouched copy.

**STOP — verify:**
```bash
python $REPO/audit.py curated --dir $NEW/docking_results --manifest $NEW/manifests/run_manifest.csv
```
- every directory contains `job_identity.json` and `final_results.csv`
- `gene == index.label(accession)` for **100%** of jobs (this is the check the old pipeline never had)
- directory count equals `run_manifest.csv` row count exactly
- `$OLD` byte count and file count unchanged from before Phase 5

---

### Phase 6 — Prepare structures for the re-dock (minimal sets only)

**Code prerequisite:** `FIX_PLAN.md` §3.5 and §3.6 — `extract_cavities.py` must take a required-set
argument, and `extract_alphafold_models.py` must have its `cavity_mapping.csv` fallback deleted.

```bash
python $REPO/extract_cavities.py --manifest-only \
    --tarballs /opt/data/cavity_space/af_strong_cavities_data_all \
    --out $NEW/manifests/cavity_manifest.csv          # names + member listings, NO unpacking

python $REPO/extract_cavities.py \
    --manifest $NEW/manifests/cavity_manifest.csv \
    --required $NEW/manifests/redock_queue.csv \
    --out-dir  $NEW/structures/cavities

python $REPO/extract_alphafold_models.py --required $NEW/manifests/redock_queue.csv --out-dir $NEW/structures/alphafold
python $REPO/fix_required_pdbs.py       --required $NEW/manifests/redock_queue.csv --out-dir $NEW/structures/fixed
python $REPO/convert_pdb_to_pdbqt.py    --required $NEW/manifests/redock_queue.csv --out-dir $NEW/structures/pdbqt
```

**STOP — verify the minimality that was the point of all this:**

| | Old campaign | Expected now |
|---|---:|---:|
| AlphaFold accession dirs | 5,661 | **~890** |
| Cavity dirs | 38,709 | **~2,900** |
| Disk for structure prep | 84 GB | **~6 GB** |

If either count comes out near the old figure, a fallback path survived — stop and fix it. Also confirm
`extract_alphafold_models.py` **exits non-zero** when `--required` is absent (it used to warn and then
extract 13,975 structures).

---

### Phase 7 — Dock the gap

**Code prerequisite:** `FIX_PLAN.md` §3.3 — `batch_dock.py` consumes a plan carrying accessions and
never resolves identity itself; plus a post-job hook that discards or `xz`-compresses `complex_*.pdb`
once `final_results.csv` is written, so `$NEW` never accumulates the 1.4 TB again.

```bash
# Smoke test FIRST — 20 jobs, all four engines
python $REPO/batch_dock.py --queue $NEW/manifests/redock_queue.csv --run-dir $NEW --test-mode --max-jobs 20
```

**STOP — verify on the 20:** all four engines produced `results.csv`; `final_results.csv` has RMSD;
`job_identity.json` written with the correct accession; `gene == index.label(accession)`; the
receptor path in the job log contains that accession; `complex_*.pdb` cleaned up as configured.

Then run for real, staged by engine as the config already does:
```bash
nohup python $REPO/batch_dock.py --queue $NEW/manifests/redock_queue.csv --run-dir $NEW > $NEW/logs/batch_dock.log 2>&1 &
```
Expect 6–8 days. It is resumable — every stage skips jobs with valid existing output.

---

### Phase 8 — Parse and analyse

**Code prerequisite:** `FIX_PLAN.md` §3.4 — all five `parse_prepare.py` defects. The 10,000-file cap is
the one that silently dropped 58% of the last campaign; do not run this phase until it is gone and
discovery is checked against `run_manifest.csv`.

```bash
python $REPO/parse_prepare.py --run-dir $NEW --out $NEW/analysis/combined_results.parquet
```

**STOP — verify:**
- discovered job count **equals** `run_manifest.csv` (the check that would have caught the 10,000 cap)
- rows with null `drugbank_id`/`uniprot_id`: **0** (was 1,174,786)
- no merge changed the row count (`checked_merge` asserts `len(out) == len(left)`; was +410,087)
- `filter_complete_tool_coverage` actually filters — it has **never** filtered, and must now fail loudly rather than no-op
- every `*_Score` column is populated for its tool (`LeDock_Score` was unmatchable at HEAD)
- `analysis/provenance.json` records the index SHA256 and code revision

---

## 4. Guardrails

1. **`$OLD` is read-only, permanently.** Nothing in this plan writes to it. Before any future cleanup there, re-read `ASSESSMENT_blast_radius_2026_08_12.md` §"Reclaim plan" — in particular, the native `.sdf`/`.dok` poses and the legacy `docking_results/` are **not** safely deletable (63 of its 786 dirs hold smina poses absent from the re-run).
2. **Copy, never move.** Until Phase 8 passes its checks, `$OLD` is the only complete copy.
3. **No stage picks its own universe.** Every extraction step takes an explicit required-set argument and exits non-zero without it. A missing prerequisite is an error, never a fallback — that single rule covers the AlphaFold fallback, the unconditional cavity unpack, and the mode-switch-on-file-presence in `batch_dock.py`.
4. **Never re-introduce symbol→accession lookup.** Keep the grep lint (`FIX_PLAN.md` G6) in CI.
5. **`run_full_workflow.py` currently never calls `generate_negative_samples.py`** — do not use it for this rebuild until §3.8 lands. Run the phases above by hand.
6. **HEAD is not the code that produced the old parquet** (`FIX_PLAN.md` F-B). Do not expect any rebuild to reproduce it. Record the producing revision in `provenance.json` from now on.

---

## 5. Expected end state

| | Old | New |
|---|---:|---:|
| Run directory size | 1,550 GB | **~35 GB** |
| Job dirs | 24,173 | ~28,500 (14,600 salvaged + ~13,900 fresh) |
| Jobs with correct identity | 13,128 (54.3%) | **100%** |
| Jobs reaching the analysis | 10,000 (41.4%) | **100%** |
| False positives in the positive class | 45.8% | **0%** |
| Contaminated negatives | 5 | **0** |
| Balance holds in accession space | no | **yes** |
| Intended-pair coverage | 25.4% | **63–69%** |
| AlphaFold structures materialised | 5,661 | ~890 |
| Cavities materialised | 38,709 | ~2,900 |

The single most valuable outcome is not the disk saving. It is that the positive class stops being 45.8%
fabricated, which is what made the headline RMSD result an artifact — the mislabelled half supplied 115%
of the observed effect, and the corrected contrast was −0.061 Å (p = 0.675). After this rebuild the
measurement means what it says.
