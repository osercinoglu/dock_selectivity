# Blast-Radius Assessment — `uniprot_gene_mapping` Corruption

**Date:** 2026-08-12  **Repository:** `dock_selectivity`  **Run directory:** `/media/onur/Elements/cavity_space_consensus_docking/2025_06_29_batch_dock`

**Method:** 14 agents — 6 parallel forensic finders, 5 adversarial verifiers that independently re-derived every headline number with their own code, 3 synthesizers. All analysis read-only; nothing on the external drive was modified. Where a finder and its verifier disagree, the verifier's figure is used and flagged.

---

## Executive summary

The corruption documented in `KNOWN_ISSUE_uniprot_gene_mapping.md` is real, reproduces exactly, and is **worse than documented in four independent ways**. It did not merely mislabel results — it **selected the receptor**, so 43.5% of jobs that assert a drug–target pair docked the ligand into a genuinely different protein. The study's headline finding is **artifact-dominated rather than attenuated**: the mislabelled half supplies 115% of the observed positive-vs-negative effect, and removing it leaves no signal.

**However — and this is the decisive fact — none of the compute is lost.** All 24,173 job directories are physically valid: a real ligand docked into a real cavity of a real protein, and the accession in each folder name is the receptor that was actually loaded. The defect is *identity*, not *physics*, and is repairable in software at zero docking cost.

### The five things that matter

| # | Finding | Number |
|---|---|---|
| 1 | Rows in `uniprot_gene_mapping.csv` mislabelled | **57.73%** (2,647/4,585) — survives 5 looser definitions; charitable floor 57.32% |
| 2 | Jobs asserting a pair that docked the **wrong protein** | **43.50%** (5,380/12,367); 64.66% of the 979 docked targets |
| 3 | Intended drug–target pairs ever docked correctly | **25.35%** (2,100/8,285) — the study's true coverage |
| 4 | Completed jobs that **never reached the analysis** | **58.6%** (14,173/24,173) — an unrelated bug, 845 GB / 8,732 core-h invisible |
| 5 | Docking compute recoverable by relabelling, zero re-docking | **100%** (24,173/24,173) |

### Four defects beyond the documented one

1. **Non-reproducibility.** `prepare_uniprot_mapping.py:51` uses `list(set(...))` (hash-randomised order) and line 130 `drop_duplicates(subset=['Entry'])` keeps the first occurrence. Whichever gene queries first permanently claims an accession; the correct later query is deleted as a duplicate. **1,992 rows destroyed, 370 genes stripped of their correct accession** — BRAF's P15056 filed under `MAP2K2`, BCL2's P10415 under `PPIF`. Re-running produces a *different* corruption pattern. Verified across 4 fresh processes.

2. **The 10,000-file cap.** `parse_prepare.py:146` sets `max_files_per_pattern = 10000` (comment: `# Higher limit`), enforced at line 99 with a hard `break` at line 104. Only **10,000 of 24,173 job directories** entered the parquet. A 500-dir sample of the excluded set: 98.6% have a complete `final_results.csv`. Loss is differential — 30.6% of negatives retained vs ~50% of positives — which is the mechanism behind the unexplained 2.03:1 class imbalance.

3. **Identity-less rows.** `parse_prepare.py:359`'s regex rejects 1,777 job directories (multi-fragment AlphaFold names, hyphenated symbols like `MT-ND1`), producing **1,174,786 rows (7.08%) with null identity**, all sharing one fabricated `compound_target_pair` value `<NA>_<NA>` — a bogus 1.17M-row mega-group in any `group_by`.

4. **Merge fan-out.** `annotate_sample_types` keys on `(uniprot_id, drugbank_id, cavity_index)`, omitting `Fragment_ID`, silently **duplicating 410,087 rows** across 179 jobs (one 12×). Confirmed physically: parquet rows ÷ `final_results.csv` rows = exactly the multiplicity.

### The mechanism

The only symbol→accession converter in the pipeline is `dict(zip(Mapped_Gene_Name, Entry))` — a **last-duplicate-wins** dict — at `identify_required_structures.py:146`, `generate_negative_samples.py:152` and `:303`, `batch_dock.py:2034`. It collapses 4,585 rows to 1,795 keys, **silently discarding 2,790 accessions**. That accession then flows into structure extraction, PDB fixing, PDBQT conversion, and the literal receptor file handed to the docking engines. **0 of 24,173 planned jobs have a receptor path disagreeing with their accession** — the corrupt accession *is* the structure that was loaded.

The AVPR2 case reproduces mechanically: the file holds P30518 (`AVPR2 ADHR DIR DIR3 V2R`) and Q8NCT1 (`ARRDC4`); the dict returns Q8NCT1 because it is last. The real V2 receptor has **zero job directories**.

**The clean key was present and never read.** `1_drug_to_protein.tsv` carries `node_2` = Entrez Gene ID, 100% numeric across all 8,568 edges, strictly 1:1 with the symbol. No script references it.

### The scientific consequence

| Contrast | Mean RMSD | Interpretation |
|---|---|---|
| Positives as run | 11.163 Å | agree **worse** than negatives — opposite to the hypothesis |
| Negatives | 10.917 Å | |
| Observed gap | **+0.246 Å** | decomposes to −0.037 Å (true positives) **+0.283 Å (false positives)** |
| True positives vs negatives | −0.061 Å | protein+cavity-matched −0.046 Å, p = 0.675 — **no signal** |

Consensus AUC as run: **0.4789**, below chance. The mislabelled half supplies **115%** of the observed gap.

**Independent experimental confirmation.** TTD IC50/Ki hit rate by group: correctly-labelled positive pairs **19.6%**, wrong-protein "positives" **0.45%**, negatives **0.38%** — a **43-fold collapse**. External assay data independently says the wrong-protein positives are indistinguishable from the negative controls.

**The negatives survived**: 5 contaminated rows of 13,452 (0.04%), structurally capped at 2.26%. But for an uncomfortable reason — 53.4% of the docked pool aren't drug targets in the interaction database at all, so they *cannot* contaminate a negative. The corruption bought label purity by docking the wrong proteins.

### Minimal data — the direct answer

**No, you did not need all the AlphaFold structures.** You needed 979 of 5,661 (17.3%). But that waste is 2.15 GB and is the least interesting part:

| Resource | Extracted | Needed | Waste | Cause |
|---|---:|---:|---:|---|
| AlphaFold accessions | 5,661 | 979 (17.3%) | 2.15 GB | `extract_alphafold_models.py` silently falls back to `cavity_mapping.csv` (13,600 accessions); two aborted bulk runs left the residue |
| Cavity directories | 38,709 | 2,829 (7.3%) | **76.6 GB** | `extract_cavities.py:63-71` unpacks *every* tarball; takes no "required" input at all |
| CavitySpace downloads | 3,100 | 0 | 1.03 GB | **zero consumers**; 30/30 sampled are coordinate-identical to the AF models already on disk |

The dominant term is elsewhere: **`consensus_docking_results/` is ~1.4 TB, of which 89% is `complex_*.pdb`** — the receptor re-serialized into every pose file, **99.75% byte-identical** between poses of the same job, and *derived* (ligand coords match `out_N.sdf` in 50/50 checks). `xz -9e` per (job, tool) compresses **146.9×**.

**~1,240 GB is reclaimable by lossless recompression, deleting nothing.** A further 25.9 GB is byte-proven duplication — full MD5 confirms `combined_consensus_docking_results.csv` and `backup_consensus_data_20251117_171312.csv` are identical 8.36 GB copies, both superseded by a 257 MB parquet with identical schema, row order and content sitting beside them.

To reproduce the final parquet: **2.95 GB**. To drive a corrected re-run: **~29 MB**.

**Do not delete** the native `.sdf`/`.dok` poses — they are the sole carrier of gnina's `CNN_VS` and `CNNaffinity_variance` and of unrounded scores (`results.csv` rounds −6.53531 → −6.54). Do not delete legacy `docking_results/` — 63 of its 786 directories hold smina poses absent from the re-run.

### Recommendation

**Option B, staged.** Relabel first (zero compute, recovers 100% of 1.44 TB / 14,893 core-h including the 14,173 jobs currently invisible to the analysis), then a targeted 13,903-job re-dock (6–8 days on 64 cores) to lift intended-pair coverage from 25.4% to 63–69%. A full clean re-run costs 2.6× more and buys **identical coverage**.

---


# Salvage & Re-Run Design

**Bottom line:** nothing needs to be re-docked to fix what already exists. All 24,173 completed job directories are physically valid — a real ligand docked into a real cavity of a real protein — and the accession embedded in every folder name is the receptor that was actually loaded. The corruption is an *identity* defect, not a *physics* defect, so it is repairable in software. The one thing software cannot fix is coverage: only **2,100 of 8,285** intended drug–target pairs (25.35%) were ever docked against the correct protein. Closing that gap costs **13,903 new jobs ≈ 6–8 days** on the 64-core box, and needs **888 AlphaFold accessions and 2,878 cavities** — 6.4× and 13.4× less than what was originally extracted.

**Recommendation: Option B, staged (A now, B next).** A full clean re-run (Option C) is explicitly *not* recommended: it costs 2.6× more than B and buys **zero additional scientific coverage** (proven below).

---

## 0. Provenance and conflict resolution

Every number below is either (a) re-derived by me in this session against the read-only run directory, or (b) taken from the adversarially-verified findings. Where a finder and its verifier disagree, **I use the verifier's figure** and flag it.

| Disputed figure | Finder claimed | Verifier corrected | Used here |
|---|---|---|---|
| Job dirs analysed in the final parquet | (not checked) | **10,000 of 24,173** | 10,000 — **I independently confirmed** (distinct `source_dir` basenames over all 16,593,933 rows) |
| Intended pair denominator | 8,507 | 8,331 clean / **8,285 judgeable** (176 phantom `(drug,'nan')` pairs) | 8,285 — **I re-derived exactly** |
| AVPR2 smoking gun | 6 dirs / 3 drugs | **8 dirs / 4 drugs** (missed DB09059) | 8 / 4 |
| Corruption rate | 57.73% | 57.73% strict, **57.64% best-defensible** | 57.7% (robust to ±0.4 pp) |
| "100% of mislabels provably wrong / bucket (c) empty" | asserted | **refuted as circular**; 2,645 of 2,647 refutable | 2,645/2,647 |
| Bug #2 magnitude | 229 accessions stolen | **1,992 rows destroyed, 370 genes lost their correct row** (from `uniprot_preparation.log`) | 1,992 / 370 |
| `consensus_docking_results/` size | 1,440.8 GB | 1,404.1 GB (n=220); `df` bound 1,656 GB allocated | **1.40–1.44 TB** |
| RMSD artifact size | +0.72 Å marginal | **+0.29 Å target-matched** (p=0.017) | +0.29 Å |
| Native `.sdf`/`.dok` files | "regenerable, deletable" | **REFUTED — carry CNN_VS/CNNaffinity_variance absent from results.csv** | must not be deleted |

New measurements I made this session: real per-engine throughput from the campaign's own logs, the re-dock gap sizing, the reusable-negative count, and per-class byte/CPU magnitudes.

---

## 1. SALVAGE LEDGER

### 1a. The docking compute — 24,173 job directories

Classification re-derived by me (parsed right-anchored, token-matched against the 4,585-accession set; 24,173/24,173 resolved to exactly one accession). Magnitudes at the measured 59.6 MB/job and **0.6161 core-h/job** (derivation in §3).

| Class | Jobs | % | Disk | CPU embodied | Verdict |
|---|---:|---:|---:|---:|---|
| **Positive, correctly labelled** | 6,987 | 28.90% | 416 GB | 4,305 core-h | **VALID AS-IS** — keep label and identity |
| **Positive, mislabelled → still a true DTI** | 308 | 1.27% | 18 GB | 190 core-h | **VALID, RELABEL** — stays a positive under its true gene |
| **Positive, mislabelled → not a known DTI** | 5,072 | 20.98% | 302 GB | 3,125 core-h | **VALID, RELABEL** — becomes an unlabelled/decoy pair |
| **Negative, uncontaminated** | 11,801 | 48.82% | 703 GB | 7,271 core-h | **VALID AS-IS** |
| **Negative, actually a true DTI** | 5 | 0.02% | 0.3 GB | 3 core-h | **RELABEL to positive** |
| **TOTAL** | **24,173** | **100%** | **1,441 GB** | **14,893 core-h** | **0 jobs require re-docking** |

Two facts make this the whole story: **0 of 24,173** planned rows have a receptor path that omits its own accession, and **0** physically invalid computations exist. Every mismatched job is mislabelled, never wasted.

**The single largest recoverable asset is not in this table.** The analysis parquet contains only **10,000 of the 24,173** job directories. The other **14,173 completed jobs — 845 GB, 8,732 core-h, 6.2 box-days of compute — never entered the analysis at all**, because `parse_prepare.py:146` caps file discovery at `max_files_per_pattern = 10000`. Sampling confirms these are finished jobs (493/500 have `final_results.csv`; 150/150 have all four engine subdirectories). Loss is differential and biases the class balance:

| Class | In parquet | Absent | Retention |
|---|---:|---:|---:|
| Positive, correctly labelled | 3,463 | 3,524 | 49.6% |
| Positive, mislabelled → decoy | 2,734 | 2,338 | 53.9% |
| Positive, mislabelled → still true | 196 | 112 | 63.6% |
| **Negative** | **3,606** | **8,195** | **30.6%** |

That 30.6% vs ~50% split is the mechanism behind the unexplained 2.03:1 positive:negative row imbalance in the parquet. Raising one constant recovers it.

### 1b. Derived artifacts

| Artifact | Size | Verdict | Why |
|---|---:|---|---|
| `final_results.csv` + per-tool `results.csv` | 2.96 GB | **VALID** | Pure physics: Tool1/Tool2/Score1/Score2/RMSD, 0 nulls, no identifier inside. Identity comes only from the enclosing folder name. |
| Native poses `.sdf`/`.dok`/`gold_soln_*` | 19.7 GB | **VALID — DO NOT DELETE** | Sole carrier of gnina `CNN_VS`/`CNNaffinity_variance` and unrounded scores (results.csv rounds −6.53531→−6.54). |
| `complex_*.pdb` | ~1,251 GB | **VALID but 99.75% redundant** | Derived = receptor + pose; ligand coords identical to `out_N.sdf` (50/50 verified). xz −9e per (job,tool) = **146.9×** → ~8.5 GB. **Recompress, delete nothing.** |
| `combined_filtered_annotated_docking_results.parquet` | 152.6 MB | **REBUILD** | 41% job coverage; 42.76% of positive rows carry a gene naming a different protein; 1,174,786 identity-less rows; 410,087 rows duplicated by a non-unique merge key; 45.8% of its positive labels false. |
| `ic50_mapping.csv` | 0.73 MB | **VALID AS-IS** | `create_ic50_mapping.py:84` uses the clean `Entry Name → Entry` bijection and never touches `Mapped_Gene_Name`. Only inherits a coverage ceiling (1,672/3,685 TTD targets). |
| DTA-Atlas annotations (24 cols) | 32.6 GB source | **VALID VALUES, RE-JOIN** | Keys on `(drugbank_id, uniprot_id)`; `entry_id` check passed on all 14,964,143 rows. Clean key joined onto corrupt row selection. |
| `cavity_cluster_id`, `louvain_cluster_pmax*` | 12.8 GB (gexf unread) | **VALID — structure-keyed** | Key on (accession, cavity) and `pocket_id`; both joins re-derived with 0 disagreements. Fix hardcoded `F1` (1,992/13,330 pockets unreachable). |
| `combined_consensus_docking_results.csv` + 4 backups | 29.6 GB | **REDUNDANT** | A 0.257 GB parquet with identical schema, row order and 28,487,256 rows already sits beside them. |

### 1c. Inputs and structures

| Artifact | Verdict |
|---|---|
| `uniprot_gene_mapping.csv` | **IRRECOVERABLE as a gene→accession map.** Only 80.03% (1,567/1,958) repairable by row-filtering; 391 genes need external re-derivation; 12 absent entirely; the file is non-reproducible (hash-randomised `list(set(...))` + `drop_duplicates`). Its `Gene Names` column remains **valid ground truth** and is what the relabel path uses. |
| `required_structures.csv` (65.11% corrupt rows), `required_structures_with_negatives.csv` (48.06%) | **REGENERATE** — seconds of compute |
| `alphafold_structures/` (5,661 dirs, 2.2–2.5 GB), `extracted_cavities/` (38,709 dirs, 82 GB) | **VALID but 85%/93% never needed.** Keep; do not re-extract (§4). |
| `1_drug_to_protein.tsv` `node_2` (Entrez) | **THE CLEAN KEY, never read by any script.** 100% numeric on 8,568/8,568 edges, strictly 1:1 with the symbol. This is the fix. |

---

## 2. THE RELABEL PATH

### Why the accession is trustworthy and the gene is not

`batch_dock.py:2211` builds the folder as `f"{drugbank_id}_{gene_name}_{uniprot_id}_{pocket_suffix}"`, drawing the two halves from opposite sides of the bug: `gene_name` traces to the TSV's own `node_2_name` (**true symbol**), while `uniprot_id` traces to `gene_to_uniprot.get(gene_name)` (**corrupt accession**). The same job record supplies `receptor_pdbqt`. Therefore **the accession is the physical truth** and the gene is decoration. Discard the gene token; keep the accession.

### Procedure

**Step 0 — rebuild the identifier map from the clean key (prerequisite for Steps 4–7).**
Resolve `node_2` (Entrez) → UniProt from a downloaded `HUMAN_9606_idmapping.dat`, not by free-text symbol search. Add the regression assertion from `KNOWN_ISSUE.md` plus a cardinality report (`max_collapse_factor` must be 1). This also recovers the 23 Entrez IDs with no symbol (237 edges, 2.77%) that were never queryable.

**Step 1 — parse identity from directory names, right-anchored.**
Do **not** use `r'(DB\d+)_([A-Za-z0-9]+)_([A-Z0-9]+)_cavity_(\d+)'` (`parse_prepare.py:359`) — it fails on 1,777 directories. Instead: strip trailing `_cavity_<N>`, strip optional `_-F<k>-model_v1`, then test each underscore token for membership in the accession set. **Result: 24,173/24,173 resolve to exactly one accession, 0 ambiguous, 0 unresolved.** (I re-verified this.)

**Step 2 — take the accession, drop the gene token.** For all 24,173 jobs.

**Step 3 — derive `true_gene`.** `true_gene = Gene Names.split()[0]` for that accession — the primary symbol from UniProt's own column. All 979 docked accessions are present, so this is 100% resolvable locally with no network.

**Step 4 — derive `true_label`.** `(drug, accession)` is **positive** iff any symbol of that accession is a target of that drug in `1_drug_to_protein.tsv`; otherwise **unlabelled/decoy**. Join via Entrez from Step 0; the symbol route is the local fallback.

**Step 5 — rebuild the results table over ALL job dirs.** Raise `max_files_per_pattern` (`parse_prepare.py:146`) above 24,173. This alone recovers 14,173 jobs / 845 GB / 8,732 core-h.

**Step 6 — fix the fan-out.** The sample-type merge keys on `(uniprot_id, drugbank_id, cavity_index)`, which is non-unique because `Fragment_ID` is omitted — 1,428 duplicate keys inflating 410,087 rows across 179 jobs (physically confirmed: parquet rows ÷ `final_results.csv` rows = exactly 12.00, 7.00, 2.00). Either add `Fragment_ID` to the key or derive the label from Step 4 and drop the merge entirely. Note `Fragment_ID` is `'F1'` on positives but bare `'1'` on negatives — normalise before joining.

**Step 7 — re-join the clean-keyed annotations** on the corrected identity: IC50 and DTA on `(drugbank_id, accession)`, cavity clusters on `(accession, cavity)`, Louvain on `pocket_id` (remove the hardcoded `F1`; add `Fragment_ID` to the cavity-cluster key to stop 421 items being silently dropped by `drop_duplicates`).

**Output schema:** `(drugbank_id, accession, fragment_id, cavity_index, true_gene, true_label, tool1, tool2, score1, score2, rmsd, cavity_cluster_id, louvain_*, ic50_*, dta_*)`.

### What this rescues

| Measure | Before | After relabel |
|---|---:|---:|
| Job dirs with valid physics | 24,173 | 24,173 (**100%**) |
| Job dirs with correct identity | 13,128 (54.3%) | 24,173 (**100%**) |
| Job dirs reaching the analysis | 10,000 (41.4%) | 24,173 (**100%**) |
| Job dirs carrying a usable label | — | **19,101 (79.0%)** = 7,300 positive + 11,801 negative |
| Job dirs → unlabelled decoys | — | 5,072 (21.0%) |
| Genuine positive (drug, accession) pairs | 3,982 claimed (45.8% false) | **2,086 real** |
| Negative/decoy class size | 11,801 jobs | **16,873 jobs** (decoys absorbed) |
| False-positive rate in the positive class | 45.77% | **0%** |

**Rescue fraction: 24,173 / 24,173 = 100% of the docking compute, at zero docking cost.** The positive class shrinks to 7,300 jobs / 2,086 pairs; the loss is real but was always illusory. The 5,072 decoys are a genuine asset — an unbiased drug × non-target set, exactly what a decoy class needs.

---

## 3. THE RE-DOCK GAP

### What was never docked correctly

| Bucket | Pairs | Disposition |
|---|---:|---|
| Intended (drug, gene) pairs, clean | 8,331 | 8,568 edges − 237 null-symbol rows |
| …locally judgeable | 8,285 | 46 symbols absent from the mapping file |
| **Docked against the correct protein** | **2,100 (25.35%)** | keep — 2,083 already on the canonical receptor, only 17 not |
| **GAP — never docked correctly** | **6,185 (74.65%)** | |
| ├ canonical accession has ≥1 cavity **and** a ligand SDF | **3,144** | **RE-DOCK NOW** |
| ├ has cavity, ligand SDF missing | 1,011 | 149 drugs are small molecules → **+482 pairs recoverable** after ligand prep; 230 drugs are not small molecules → legitimately undockable |
| └ no cavity for the canonical accession | 2,030 | needs new pocket detection; hard ceiling |

Realistic coverage ceiling with this structural resource: **5,726 / 8,285 = 69.1%**.

### Cost model — measured from this campaign's own logs

Rates are **all-in** (they include the timeout tax: the Nov 6–10 pass alone absorbed 1,975 gold and 3,145 ledock timeouts).

| Stage | Concurrency as configured | Evidence | s/job wall | core-h/job |
|---|---|---|---:|---:|
| smina | 2 jobs × exh 32 = 64 cores | 12,006 jobs / 128,381.7 s (Nov 6–8) | 10.69 | 0.190 |
| gnina | 2 jobs × exh 32 = 64 cores | **no gnina stage in any log — modelled at smina rate** | 10.69–21.4 | 0.190–0.380 |
| gold | 60 × 1 core | 11,194 jobs / 60,517.5 s (Nov 8–9) | 5.41 | 0.090 |
| ledock | 60 × 1 core | 11,078 jobs / 87,230.3 s (Nov 9–10) | 7.87 | 0.131 |
| RMSD consensus | 16 × 1 core | 24,173 jobs / 78,985.5 s (Dec 21–22, full pass) | 3.27 | 0.015 |
| **TOTAL** | | | **37.9 – 48.6** | **0.616 – 0.806** |

The gnina band is the only assumption: gnina never ran as a logged stage, and its configured timeout is 2× smina's (1200 s vs 600 s). The upper bound assumes gnina is 2× slower than smina. **Anchor check:** 24,173 × 0.616 = 14,893 core-h = 254.7 box-hours = 10.6 days for one clean pass — consistent with a campaign that actually spanned July–December because of restarts (Nov 6–10 alone: 80 box-h; Nov 16–17: 32 box-h; Dec RMSD: 22 box-h; Jul 7–8 ledock: 23 box-h).

### Re-dock sizing

Docking all cavities of one canonical accession per gene (mean 3.24 cavities/protein, matching the original design):

| Scenario | Jobs | Box wall-clock | Core-hours |
|---|---:|---:|---:|
| Gap positives only | 10,610 | 4.7 – 6.0 d | 6,537 – 8,554 |
| Gap positives, top-3 cavities/protein | 7,571 | 3.3 – 4.3 d | 4,664 – 6,104 |
| **Gap positives + negative top-up (recommended)** | **13,903** | **6.1 – 7.8 d** | **8,566 – 11,209** |
| …including the 149 newly prepared ligands | 17,153 | 7.5 – 9.7 d | 10,568 – 13,830 |

**The negative top-up is small because 7,317 of the 11,806 existing negative jobs (62.0%) already sit on accessions in the corrected receptor pool and are directly reusable.** Only 3,293 new negative jobs are needed for 1:1 balance. Storage: 13,903 × 59.6 MB ≈ **829 GB** (5.3 TB free — not a constraint).

---

## 4. MINIMAL INPUT SET — "do we really need ALL the AlphaFold structures?"

**No. Not remotely.**

| Resource | Originally extracted | Option B needs | Reduction | Option C needs | Reduction |
|---|---:|---:|---:|---:|---:|
| AlphaFold accession dirs | 5,661 | **888** | **6.4×** | 1,274 | 4.4× |
| Cavity dirs | 38,709 | **2,878** | **13.4×** | 4,026 | 9.6× |
| Ligand SDFs | 2,599 on disk | **867** (all present) | 3.0× | 1,484 | 1.8× |
| Disk to prepare | 84 GB extracted | **~6 GB** | ~14× | ~9 GB | ~9× |

Even a **full corrected re-run touches 1,274–1,388 accessions — 22–25% of what was extracted, and 9–10% of the 13,600 accessions in `cavity_mapping.csv`.** Negatives need no additional structures: they are drawn from the same corrected receptor pool.

**Why the over-extraction happened (both causes are one-line fixes):**
- `extract_alphafold_models.py` silently falls back to `cavity_mapping.csv` (13,600 accessions / 13,975 fragments) when `required_structures.csv` is absent. Two aborted bulk runs on 2025-07-02 left 5,661 dirs; the corrected 14:47 run then needed only 623 new extractions. **Fix: remove the fallback, fail loudly.**
- `extract_cavities.py:63-71` unpacks *every* tarball unconditionally (`os.listdir` over 38,709 files) with no "required" input, because `cavity_mapping.csv` must exist before the required set can be computed. **Fix: build `cavity_mapping.csv` from tarball *filenames* without unpacking, then unpack only the required 2,878.** This alone avoids 76.6 GB of waste.

Bytes actually needed to drive a corrected re-run from this drive: **~29 MB** (`1_drug_to_protein.tsv`, `small_molecule_drug_links.csv`, `cavity_mapping.csv`, ligand SDFs), because both structure sets are re-derivable from `/opt/data/`. Keeping the already-extracted required cavities to skip re-extraction adds 5.4 GB.

---

## 5. DECISION TABLE

| | **A — Relabel only** | **B — Relabel + targeted re-dock** | **C — Full clean re-run** |
|---|---|---|---|
| New docking jobs | **0** | 13,903 | 35,998 |
| Core-hours | **0** | 8,566 – 11,209 | 22,178 – 29,022 |
| Box wall-clock (64 cores) | **0** | **6.1 – 7.8 days** | 15.8 – 20.3 days |
| Engineering effort | 3–5 days | +2 days | +1 week |
| New storage | ~0.2 GB | ~829 GB | ~2,145 GB |
| Structures to prepare | **0** | 888 acc / 2,878 cav | 1,274 acc / 4,026 cav |
| Existing compute reused | **100%** | 100% | ~0% (re-docks everything) |
| **Intended-pair coverage** | **2,100 / 8,285 = 25.4%** | **5,244 = 63.3%** | **5,244 = 63.3% — identical** |
| …with ligand prep | n/a | 5,726 = 69.1% | 69.1% |
| Genuine positive pairs analysable | 2,086 | ~5,230 | ~5,230 |
| Total analysable jobs | 24,173 | 38,076 | 35,998 |

**Scientific claims each option supports**

- **A** supports: a corrected, artifact-free measurement of cross-tool pose agreement on 2,086 genuine positive pairs vs 16,873 negatives/decoys; an honest restatement of the headline result. This matters because the current result is *artifact-dominated*, not merely attenuated: the observed +0.246 Å positive–negative gap decomposes into −0.037 Å from the genuine half and +0.283 Å from the mislabelled half, i.e. the mislabelled half supplies **115%** of the gap, and the null-by-construction FP-vs-NEG contrast (AUC 0.4731) sits 1.5× further from 0.5 than the truth-corrected signal (0.4824). A does **not** support a general claim about DTIs, because the positive set is a biased 25% sample of intent and 523 of 979 receptors are proteins nobody meant to study.
- **B** additionally supports: a properly powered discriminator claim on a correctly resolved, balanced set covering 63–69% of intended pairs, with uniform receptor selection and an unbiased decoy class. This is the minimum that makes the study's central hypothesis testable.
- **C** adds **no coverage over B** — it re-docks the 2,100 pairs that are already validly docked and re-buys 7,317 reusable negatives at full price. Its only gain is uniform provenance (one mapping, one code version, one timeout policy).

### Recommendation: **B, staged — do A immediately, then B.**

1. **Week 1 (zero compute):** run the relabel path (§2). Recovers 100% of 1.44 TB / 14,893 core-h of docking, including the 14,173 jobs / 8,732 core-h currently invisible to the analysis. Re-run the discrimination analysis on the corrected labels and publish the corrected effect size.
2. **Week 1, in parallel:** rebuild the mapping from the Entrez key (§2 Step 0) with the regression assertion. This is the actual bug fix and gates everything downstream.
3. **Weeks 2–3:** extract only the 888 accessions / 2,878 cavities, prepare the 149 missing small-molecule ligands, and run the 13,903-job targeted re-dock (6–8 days wall-clock).
4. **Housekeeping, any time:** recompress `complex_*.pdb` per (job, tool) at 146.9× to reclaim ~1,243 GB **deleting nothing**, and drop the 29.6 GB CSV family that is already superseded by a 0.257 GB parquet on the same disk. Do **not** delete the native `.sdf`/`.dok` poses or the legacy `docking_results/` (63 of its 786 dirs hold smina poses absent from the re-run).

**Residual limitations to state in any write-up.** Ground truth is the `Gene Names` column of the very file that is corrupted in a different column; it covers 4,585 accessions, all 979 docked ones included, so the relabel path is fully resolvable locally — but the re-dock gap's canonical-accession choice should be made from an authoritative idmapping file, since 391 of 1,958 genes (19.97%) cannot be resolved from this file at all. "Non-interaction" means "not annotated in `1_drug_to_protein.tsv`", so the 5,072 relabelled decoys are *unlabelled*, not proven non-binders. And 2,030 gap pairs have no CavitySpace cavity for their correct protein — that ceiling is structural and no amount of re-docking moves it.

---


# Minimal-Data Doctrine

**Standing principle: we shall only use what's needed.**

This section converts the storage audit into an operating doctrine. It is written against the adversarially-corrected classifications, not the original finder's, because the verification pass reversed the two most consequential calls in both directions — it found ~1,240 GB of reclaim the finder classed as irreplaceable, and it found ~34 GB of irreplaceable compute the finder had queued for deletion.

**Nothing in this document has been executed.** Every deletion below is a recommendation for you to run or approve, gated behind a verification command that must pass first. No file under `/media/onur/Elements` or `/home/onur/repos/dock_selectivity` was written, moved, or modified in producing it.

---

## 1. The over-extraction pattern

### The flaw has a name: **supply-driven iteration**

Three separate scripts produced three separate mountains of unused data by making the same structural mistake. Each one loops over *everything the source offers* and then, at best, filters afterwards — instead of looping over *what the next step will actually consume* and using the source only as a lookup table.

| Instance | Extracted | Actually needed | Waste ratio | Wasted bytes |
|---|---:|---:|---:|---:|
| `alphafold_structures/` | 5,661 accession dirs | 979 | 5.78× | ~1.90 GB |
| `extracted_cavities/` | 38,709 cavity dirs | 2,829 | 13.68× | ~76.6 GB |
| `cavityspace_structures/` | 3,100 dirs | 0 (zero consumers) | ∞ | 1.495 GB |

### Root cause, per script — the collection each one iterates over

**`extract_cavities.py` — no demand input exists at all.**
```python
# line 65
tar_files = [f for f in os.listdir(cavity_tarball_folder) if f.endswith(".tar.gz")]
# line 77
for tar_gz_filename in tqdm(tar_files, desc="Extracting Tarballs", unit="file"):
```
The loop variable is the raw directory listing of `/opt/data/cavity_space/af_strong_cavities_data_all`. The function signature accepts no "required" argument, so there is no way to ask it for less. It unpacked all 38,709 tarballs to satisfy 2,829 pockets. Note the ordering trap that made this feel inevitable: `identify_required_structures.py` consumes `cavity_mapping.csv`, so the mapping must exist before demand can be computed — but the mapping only needs *tarball filenames*, which are readable from the archive index without unpacking a single payload.

**`extract_alphafold_models.py` — the demand set is substituted for the supply set on a silent fallback.**
```python
# line 199  (preferred path)
df = pd.read_csv(REQUIRED_STRUCTURES_CSV)      # 979 accessions / 1,005 fragments
# line 221  (fallback path)
df = pd.read_csv(CAVITY_MAPPING_CSV)           # 13,600 accessions / 13,975 fragments
# line 230
logging.warning("Using all cavity structures - for efficiency, run identify_required_structures.py first")
```
This is the worst variant, because the code *knows* the right answer and degrades to the wrong one behind a `warning`. The `alphafold_extraction.log` shows exactly this: the 2025-07-02 00:52 and 00:59 runs took the fallback and queued 13,975 extractions; the corrected 14:47 run needed only 623 new extractions because 382 of its 1,005 were already on disk from the aborted bulk run. A 14× over-extraction was logged at severity `WARNING` and nobody saw it.

**`download_cavityspace_structures.py` — supply-driven, over the network, with no consumer downstream.**
```python
# line 173
df = pd.read_csv(mapping_file)                                   # cavity_mapping.csv = supply
# line 200
unique_combinations = cavity_df[['UniProt_ID','Fragment_ID']].drop_duplicates()
# line 203
for _, row in tqdm(unique_combinations.iterrows(), ...):
```
It never reads `required_structures.csv`. It downloaded 3,100 AlphaFold **v1** models when the pipeline docks **v4** models from a local archive. I confirmed 30/30 sampled accessions have bit-identical CA coordinates between the v1 and v4 copies, that all 3,100 are already present in `alphafold_structures/`, and that no script or notebook in the repo reads the directory. `run_full_workflow.py` does not even invoke the downloader.

### The counter-example that proves the rule

`identify_required_structures.py` is the one script that gets this right, and it is worth reading as the template:

```python
# lines 164-172 — start from DEMAND
required_genes = filtered_interactions['node_2_name'].unique()
required_uniprot_ids = set()
for gene in required_genes:
    uniprot_id = gene_to_uniprot.get(gene)
    if uniprot_id:
        required_uniprot_ids.add(uniprot_id)

# lines 181-183 — supply appears ONLY as a lookup, never as the loop variable
available_structures = cavity_df[
    cavity_df['UniProt_ID'].isin(required_uniprot_ids)
].copy()
```

Demand is the iterator. Supply is an `.isin()`. That single inversion is the whole difference between 979 and 13,600.

### The general rule

> **Every extraction, download, or conversion step must accept an explicit demand manifest as a required argument and iterate over that manifest. The source catalogue may only be consulted as a lookup (`.isin()`, `dict.get()`, archive index). A missing or empty manifest is a fatal error — never a fallback to the full catalogue, and never a warning.**

Three corollaries, each of which independently would have prevented one of the three instances:

1. **No `os.listdir()` / `glob()` of a source tree may be a loop variable in an extraction step.** It may only build an index that demand is looked up against. (Kills the `extract_cavities.py` instance.)
2. **A demand manifest that fails to load must `sys.exit(1)`.** Substituting the supply set is never the safe default; it is the maximally expensive default. Emit `CRITICAL`, not `WARNING`. (Kills the `extract_alphafold_models.py` instance.)
3. **No artifact may be produced that no downstream step reads.** A step whose output has no consumer in `run_full_workflow.py` should not be in the workflow. (Kills the `cavityspace_structures/` instance.)

A fourth pattern, adjacent and worth naming because it cost the most: **do not classify a large derived artifact as irreplaceable without testing whether it is derived.** The 1,251–1,283 GB of `complex_*.pdb` files were treated as primary output for the life of the project. They are receptor coordinates re-serialized once per pose — ~99.75% redundant against their own siblings, and their ligand coordinates are identical to the `out_N.sdf` poses sitting beside them.

---

## 2. Retention table

All sizes in decimal GB (10⁹ B). **Exact** = `du -sb` or `os.stat`. **Sampled** = stratified random sample with the stated uncertainty. Rows marked **⬇ DOWNGRADED** were moved *out* of a reclaimable class by the verification pass; rows marked **⬆ UPGRADED** were moved *into* one.

### IRREPLACEABLE — cannot be regenerated at any price (~46.7 GB)

| Artifact | Size | Basis | Note |
|---|---:|---|---|
| `consensus_docking_results/**/final_results.csv` + per-tool `results.csv` | 2.93 GB | sampled n=400 | The only parseable results; 0.21% of the directory holding them |
| `consensus_docking_results/**/*.sdf`, `gold_soln_*.sdf`, `*.dok` | ~19.7 GB | sampled | **⬇ DOWNGRADED** from "regenerable input" |
| `docking_results/` (legacy July-2025 smina pass) | ~14.4 GB | sampled n=786 exhaustive | **⬇ DOWNGRADED** from "obsolete" |
| `merged_pocketmatch_results.txt` | 7.753 GB | exact | **Unresolved** — holds ~10.6M pairs *more* than `parquet_data`; convert, do not delete |
| `pocketmatch_results/parquet_data/` (163 parts, 87,323,086 rows) | 1.472 GB | exact | Canonical PocketMatch output |
| `combined_consensus_docking_results.parquet` (28,487,256 rows) | 0.257 GB | exact | The replacement for the 8.36 GB CSV family |
| `combined_filtered_annotated_docking_results.parquet` | 0.153 GB | exact | The analysis artifact |
| `pocket_louvain_clusters_*.csv` (6 files) | 0.003 GB | exact | The only *consumed* clustering output |
| Mapping CSVs, `1_drug_to_protein.tsv`, `uniprot_preparation.log`, summary JSONs | ~0.02 GB | exact | Forensic evidence of the mapping bug — see §3 never-delete |

**Why the two downgrades matter more than anything else in this table:**

*Native pose files.* The finder's Tier 3 deleted `.sdf` and `.dok` as "regenerable job inputs." They are the native output of smina, gnina, GOLD and LeDock. I verified directly that the gnina SDFs carry fields that exist nowhere else:

```
gnina/out_10.sdf        gnina/results.csv
  minimizedAffinity        -6.26447          GNINA_Affinity      -6.54
  CNNscore                  0.1231943741     GNINA_CNN_pose       0.4047
  CNNaffinity               6.7710175514     GNINA_CNN_affinity   7.26
  CNN_VS                    0.8341512680     (absent)
  CNNaffinity_variance      1.9565788507     (absent)
```
`CNN_VS` and `CNNaffinity_variance` are lost outright; every retained score loses six decimal places. The claim "with every docking pose still intact" was false as written.

*Legacy `docking_results/`.* The finder checked one directory and generalised. An exhaustive scan of all 786 found 63 (8.0%) holding smina poses absent from the consensus re-run. I confirmed the worked example myself:

```
docking_results/DB00253_NR3C1_P04150_cavity_7/smina/           -> complex_1.pdb complex_2.pdb complex_3.pdb
consensus_docking_results/DB00253_NR3C1_P04150_cavity_7/smina/ -> complex_1.pdb
```
Deleting the legacy tree destroys real poses. It is also *live input* — `parse_prepare.py:1545` sets `search_dir = os.path.dirname(BASE_DIR)` and matches `results.csv`, so legacy results already feed the parquet.

### CHEAPLY REGENERABLE — reproducible from a confirmed-present source (~1,512 GB)

| Artifact | Size | Basis | Regenerable from |
|---|---:|---|---|
| `consensus_docking_results/**/complex_*.pdb` | ~1,251 GB | sampled, 89.1% of dir | **⬆ UPGRADED** from irreplaceable — receptor + `out_N.sdf` pose |
| Per-job inputs: `.mol2`, receptor `.pdb` copies, misc | ~124.5 GB | sampled, 8.86% | `converted_pdbqt/` + `extracted_cavities/` |
| `extracted_cavities/` — the 35,880 never required | ~76.6 GB | sampled ±5.5 | `/opt/data/cavity_space/af_strong_cavities_data_all` ✔ present |
| `dta_atlas_data/` (240 parquet parts) | 32.615 GB | exact | Third-party DTA-Atlas 1.0 re-download |
| `pocket_network_louvain_*.gexf` (6 files) | 12.783 GB | exact | `parquet_data/` via `pocketmatch_clustering.py` |
| `converted_pdbqt/` | 6.077 GB | exact | **not classified by the finder at all** |
| `extracted_cavities/` — the 2,829 required | ~5.40 GB | sampled | Same tarballs; keep to skip re-extraction |
| `alphafold_structures/` — the 4,682 unused | ~1.90 GB | sampled n=800 | `/opt/data/alphafold2/UP000005640_9606_HUMAN_v4.tar` ✔ present |
| `fixed_structures/` | 0.704 GB | exact | **not classified by the finder at all** |
| `alphafold_structures/` — the 979 used | ~0.33 GB | sampled | Same tar; keep, trivially small |

The upgrade of `complex_*.pdb` is the single largest finding in the audit. I measured xz -9e compression on seven independent (job, tool) groups:

| Group | Ratio |
|---|---:|
| `DB00035_AVPR1B_P47901_cavity_1/gnina` | 108.6× |
| `DB00397_ADRB1_P08588_cavity_1/ledock` | 113.2× |
| `DB00035_AVPR1B_P47901_cavity_1/gold` | 118.5× |
| `DB00035_AVPR1B_P47901_cavity_1/smina` | 127.7× |
| `DB00035_AVPR1B_P47901_cavity_1/ledock` | 132.3× |
| `DB00397_ADRB1_P08588_cavity_1/smina` | 147.6× |
| `DB00890_SHBG_P98164_-F2-model_v1_cavity_11/ledock` | 153.2× |

Range 108.6–153.2×, median ~128× (verifier's pooled figure over 10 groups: 146.9×). At a deliberately conservative **120:1** planning ratio, 1,251 GB becomes ~10.4 GB.

### REDUNDANT — a byte-identical or schema-identical copy already exists on disk (~38.8 GB)

| Artifact | Size | Duplicate of | Proof status |
|---|---:|---|---|
| `combined_consensus_docking_results.csv` | 8.365 GB | `…_results.parquet` (0.257 GB) | **Verified by me**: 28,487,256 rows both, identical 30-col schema |
| `backup_consensus_data_20251117_171312.csv` | 8.365 GB | the combined CSV | Full md5 identical: `cedbcc587ace0a6e9979b3e9e3fd1aff` |
| `pocketmatch_results/PocketMatch_similarities/` (163 CSVs) | 7.693 GB | `parquet_data/` (1.472 GB) | Verified 6 random parts: CSV lines = parquet rows + 1 |
| `backup_consensus_data_2025111{2,2,3}*.csv` (3 files) | 12.890 GB | subsets of backup 4 | **Inferred only** — see Stage 3 gate |
| `cavityspace_structures/` | 1.495 GB | `alphafold_structures/` | 30/30 CA coords identical; zero consumers; **finder sized this 45% low (1.028 GB)** |

The finder's own tier 0 was labelled "provably redundant, zero information loss" yet included backups 1–3, whose containment its own limitations section called "inferred, not exhaustively proven." That contradiction is resolved here by moving them to a gated stage.

### OBSOLETE — superseded diagnostics (~5.3 GB)

| Artifact | Size | Basis |
|---|---:|---|
| `consensus_docking_results/**/log_*.log` (per-job) | ~4.58 GB | sampled, 0.318% of dir |
| Top-level `*.log` + `nohup.out` (25 files) | 0.746 GB | **exact** — `docking_automation.log` alone is 448.2 MB |
| `consensus_docking_results/docking_automation.log` | 0.014 GB | exact |

**Class totals:** IRREPLACEABLE 46.7 · REGENERABLE 1,511.9 · REDUNDANT 38.8 · OBSOLETE 5.3 = **~1,602.7 GB**, consistent with the independent `df`-difference figure of 1,656.2 GB allocated (the gap is filesystem allocation overhead across ~4.1M small files).

**Reclassification summary vs. the finder:** IRREPLACEABLE fell from 1,295.95 GB (79.3%) to ~46.7 GB (2.9%). Achievable reclaim rose from 338.54 GB (20.7%) to ~1,537 GB (~96%).

---

## 3. Reclaim plan

Ordered safest → riskiest. **Each stage is a recommendation. Run the verification first; only proceed if it passes; act only with your approval.** Every stage is independent — you may stop after any one.

### Stage 0 — Lossless recompression. Reclaims ~1,240 GB. Deletes nothing.

The largest win, and the only one with literally zero risk of information loss, since the original bytes are recoverable by decompression.

**Verify first** (per (job, tool) group, on a handful before committing):
```
cat <job>/<tool>/complex_*.pdb | wc -c                       # raw bytes
cat <job>/<tool>/complex_*.pdb | xz -9e -T0 -c | wc -c       # compressed bytes
```
Expect ≥100:1. If any group returns <50:1, stop and investigate that group before proceeding.

**Then, for each group**: create `<tool>/complex.tar.xz`, verify the archive lists exactly the expected filenames and that `xz -t` passes, verify extraction of one member is byte-identical to the original, and only then remove the loose files. Recommend keeping the archive per (job, tool) rather than one global archive — a single corrupt global archive would lose the whole campaign.

**Recommended guard:** run this on 100 jobs, confirm round-trip integrity, and leave it a week before doing the remaining 24,073.

### Stage 1 — Byte-proven duplicates. Reclaims 25.92 GB.

| Delete | Size | Verification to run BEFORE deleting |
|---|---:|---|
| `combined_consensus_docking_results.csv` | 8.365 GB | `wc -l` on the CSV = 28,487,257; `pq.ParquetFile(parquet).metadata.num_rows` = 28,487,256; confirm the 30 column names match the CSV header exactly; spot-compare 5 random rows |
| `backup_consensus_data_20251117_171312.csv` | 8.365 GB | `md5sum` both files in full — must both be `cedbcc587ace0a6e9979b3e9e3fd1aff`. Do not accept a head/tail sample here; the files are large enough to warrant the full read |
| `pocketmatch_results/PocketMatch_similarities/` | 7.693 GB | For all 163 parts: `wc -l similarities_part_NNNN.csv` = parquet part rows + 1; header = `Pocket1,Pocket2,Pmin,Pmax`; row 0 identical |
| `cavityspace_structures/` | 1.495 GB | `grep -rl cavityspace_structures` over `*.py` and `*.ipynb` returns nothing; `set(cavityspace dirs) - set(alphafold dirs)` is empty; CA coords match v4 on ≥25 sampled accessions |

### Stage 2 — Log compression + rotation. Reclaims ~4.9 GB.

**Verification:** confirm `uniprot_preparation.log` is on the never-delete list (see below) before touching anything in this class.

Recommend `zstd -19` (or `xz`) in place rather than deletion for top-level logs — these are the run's provenance record and compress 10–20×. Per-job `log_*.log` files can be tarred per job alongside the Stage 0 archive.

### Stage 3 — Backup chain. Reclaims 12.89 GB. **Gated.**

`backup_consensus_data_20251112_172814.csv`, `…20251112_203531.csv`, `…20251113_020050.csv`.

The containment argument is strong but circumstantial: the `data_preparation_summary_*.json` chain shows 7,136,562 → 14,273,124 → 21,409,686 → 28,487,256 with each file's `Combined` bucket exactly equal to the previous file's total, and identical byte deltas (2,043,814,169 twice). But no row-level set comparison was ever run.

**Verification to run BEFORE deleting:** build a key set (e.g. `source_dir` + `File1` + `File2` + `Tool1` + `Tool2`) from each backup and confirm each is a strict subset of `combined_consensus_docking_results.parquet`. If any backup contains keys absent from the parquet, that backup is **not** redundant and must be reclassified. This costs one pass over ~13 GB and closes the finder's own stated gap.

### Stage 4 — Regenerable non-docking artifacts. Reclaims ~130.2 GB.

| Delete | Size | Verification BEFORE deleting |
|---|---:|---|
| `extracted_cavities/` minus the 2,829 required | 76.6 GB | Confirm `/opt/data/cavity_space/af_strong_cavities_data_all` exists and holds 38,709 tarballs; confirm all 2,829 `required_structures.csv` Pocket_PDB paths resolve on disk and are in the keep-set |
| `dta_atlas_data/` | 32.6 GB | Confirm the DTA-Atlas 1.0 release is still retrievable **and** that you accept a re-download cost; the annotation columns are already baked into the final parquet, so this is only needed to re-run `parse_prepare.py` |
| `pocket_network_louvain_*.gexf` | 12.78 GB | `grep -rn gexf *.py` shows only writes (`pocketmatch_clustering.py:334,366,400,405`), zero reads; confirm the six `pocket_louvain_clusters_*.csv` (3.07 MB) are present and are what `parse_prepare.py:1224` globs |
| `converted_pdbqt/` | 6.08 GB | Only after docking is finished for good — this is receptor input. Confirm no pending re-dock is planned |
| `alphafold_structures/` minus the 979 used | 1.90 GB | Confirm `/opt/data/alphafold2/UP000005640_9606_HUMAN_v4.tar` exists; confirm `set(required) - set(on-disk)` is empty |
| `fixed_structures/` | 0.70 GB | Same gate as `converted_pdbqt/` |

### Stage 5 — Per-job regenerable inputs. Reclaims ~124.5 GB. **Riskiest.**

`.mol2`, receptor `.pdb` copies and miscellaneous per-job input files — **explicitly excluding `.sdf`, `gold_soln_*.sdf` and `.dok`.**

**Verification BEFORE deleting:** enumerate by extension within a job and confirm your delete-list contains no `.sdf` or `.dok`; confirm `final_results.csv` and each `<tool>/results.csv` are present and non-empty; confirm the Stage 0 pose archive for that job passes `xz -t`. Recommend doing this only after the corrected pipeline has been re-run and you are confident no re-dock from existing inputs is needed.

### Cumulative

| After stage | Reclaimed | % of ~1,603 GB |
|---|---:|---:|
| 0 — recompress poses | 1,240 GB | 77.4% |
| 1 — byte-proven duplicates | 1,266 GB | 79.0% |
| 2 — logs | 1,271 GB | 79.3% |
| 3 — backups (gated) | 1,284 GB | 80.1% |
| 4 — regenerable non-docking | 1,414 GB | 88.2% |
| 5 — per-job inputs | **1,538 GB** | **95.9%** |

### NEVER DELETE

1. **`consensus_docking_results/**/*.sdf`, `gold_soln_*.sdf`, `*.dok`** — native engine output. Carries `CNN_VS`, `CNNaffinity_variance` and full float precision found nowhere else. ~19.7 GB.
2. **`consensus_docking_results/**/final_results.csv` and `<tool>/results.csv`** — the only parseable results; 0.21% of the directory they live in. ~2.93 GB.
3. **`docking_results/` (legacy)** — 63 of 786 dirs hold smina poses absent from the re-run, and `parse_prepare.py` already ingests its `results.csv`. Deleting it silently changes the parquet's row count. ~14.4 GB.
4. **`merged_pocketmatch_results.txt`** — unresolved. Its ~97,869,263 pairs exceed `parquet_data`'s 87,323,086, so neither file is a superset of the other. Convert to parquet (recovers ~6.3 GB), do not delete. 7.75 GB.
5. **`combined_consensus_docking_results.parquet`** and **`combined_filtered_annotated_docking_results.parquet`** — 0.41 GB total; the entire reason the CSV family is deletable.
6. **`pocketmatch_results/parquet_data/`** and **`pocket_louvain_clusters_*.csv`** — canonical PocketMatch output and the only consumed clustering result. 1.475 GB.
7. **All mapping CSVs, `1_drug_to_protein.tsv`, `uniprot_gene_mapping.csv`, `required_structures*.csv`, `cavity_mapping.csv`, `*_mapping.csv`, all `data_preparation_summary_*.json`, `job_preparation_summary.csv`, `docking_jobs_preview.csv`, `negative_samples_metadata.json`.** ~20 MB total and they are the entire provenance chain.
8. **`uniprot_preparation.log`** (314 KB) — the complete per-gene query trace. It is the only artifact that proves the accession-theft mechanism, establishes first-query-wins ordering (1,795/1,795 match), and records that 1,992 of 6,577 retrieved rows were destroyed by `drop_duplicates`. A previous pass declared this log did not exist. It does. **Back it up off-drive before touching anything.**
9. **`alphafold_extraction.log`, `cavity_extraction.log`, `required_structures.log`, `negative_samples.log`** — the record of *how* the over-extraction happened. A few hundred KB.

---

## 4. Pipeline changes that enforce minimality

### A manifest contract for every extraction step

Each producing step must emit, and each consuming step must require, a **manifest**: a small CSV that is the sole authority on what the next step may touch.

```csv
# manifest schema — <step>_manifest.csv
item_id,source_ref,reason,requested_by,run_id
Q8NCT1_F1,/opt/data/alphafold2/...,required_structures.csv:row_412,extract_alphafold_models,2026-08-12T20:00:00Z
```

- `item_id` is the natural key of the thing to be materialised.
- `source_ref` locates it in the supply *without materialising it* (archive member name, tarball path, URL).
- `reason` is a traceable pointer back to the demand row that justified it. **An item with no `reason` must not be extracted.**
- The manifest is written *before* extraction, so it is auditable in advance and diffable between runs.

### Per-script changes

| Script | Change | Line reference |
|---|---|---|
| `extract_cavities.py` | Add a required `--manifest` argument. Replace `os.listdir(...)` at :65 with a read of the manifest; iterate the manifest at :77 and use the tarball listing only to resolve `source_ref`. Build `cavity_mapping.csv` from **tarball filenames and archive indices**, without unpacking payloads — this breaks the ordering trap that forced supply-driven extraction. | :65, :77 |
| `extract_alphafold_models.py` | Delete the fallback branch entirely. If `required_structures.csv` is missing → log `CRITICAL` and `sys.exit(1)`. Never substitute `cavity_mapping.csv`. | :215–230 |
| `download_cavityspace_structures.py` | Remove from the workflow, or gate behind an explicit `--enable` flag with a named downstream consumer. Its output has zero consumers today. | :173, :200, :203 |
| `identify_required_structures.py` | Keep the demand-driven pattern (:164–183) as the reference implementation. Add manifest emission. Separately — and this is the correctness fix, not a minimality fix — replace the corrupted `dict(zip(Mapped_Gene_Name, Entry))` resolver at :146 with a join on the clean Entrez key `node_2`, which is 100% populated and strictly 1:1 in `1_drug_to_protein.tsv` and is currently read by nothing. | :146, :164–183 |
| `fix_required_pdbs.py`, `convert_pdb_to_pdbqt.py` | Take the manifest as input rather than scanning their input directory; refuse to run on an unmanifested directory. | :52 |
| `batch_dock.py` | Include `Fragment_ID` in `job_signature` at :2089 — the current `{drug}_{uniprot}_{cavity}` key silently collapsed 1,309 fragment-resolved pockets. Fix the pocket-suffix regex at :2206, which only strips the `-F1-` form and produced 1,743 malformed directory names. | :2089, :2206 |
| `run_full_workflow.py` | Add a post-step assertion: **materialised item count must equal manifest row count.** Fail the workflow on any mismatch. A 5.78× or 13.68× overrun would have halted the run in 2025 instead of being discovered a year later. |  |

### The assertion that would have caught all three instances

```python
# after every extraction step
n_manifest = len(manifest)
n_materialised = count_on_disk(output_dir)
if n_materialised != n_manifest:
    logging.critical(f"OVER-EXTRACTION: {n_materialised} materialised vs {n_manifest} manifested")
    sys.exit(1)
```

Cheap, and it converts a silent 76 GB overrun into an immediate stop.

### Identity by key, never by filename

A related minimality violation worth fixing in the same pass: `parse_prepare.py:359` reconstructs `drugbank_id / gene_name / uniprot_id / cavity_index` by **regex-parsing the job directory name**, with no join back to a keyed table. That regex rejects 1,777 of 24,173 directories, which is why 1,174,786 parquet rows (7.08%) carry null identity. Job directories should carry a `job_manifest.json` written at creation time, and `parse_prepare.py` should read it rather than parse the folder string. Filenames are a display format, not a database.

---

## 5. Streaming and format policy

Four artifacts on this drive are the same failure repeated: **a row-oriented text serialisation kept alongside, or instead of, a columnar one.** Together they account for ~58 GB.

| Artifact | Size | Columnar equivalent | Overhead |
|---|---:|---|---:|
| `combined_consensus_docking_results.csv` | 8.365 GB | `…parquet`, same 28,487,256 rows | **32.6×** |
| `backup_*.csv` × 4 | 21.255 GB | same parquet | — |
| `PocketMatch_similarities/` (163 CSVs) | 7.693 GB | `parquet_data/`, same 87,323,086 rows | **5.2×** |
| `merged_pocketmatch_results.txt` | 7.753 GB | none yet — should be parquet | ~6.3 GB recoverable |
| `docking_automation.log` | 0.448 GB | rotated + compressed | ~10–20× |

### Policy

**1. Parquet is the only tabular storage format.** Any dataset over ~10⁵ rows is written as parquet with a defined schema, snappy or zstd compression, and row groups sized for streaming (~500k rows). CSV is permitted only for files a human will open by hand — manifests, mapping tables, summary reports, all well under 10 MB.

**2. Never keep both.** If a parquet exists, the CSV twin is deleted the moment the parquet passes a row-count and schema check. The 8.36 GB CSV and its 257 MB parquet twin coexisted on this drive for nine months.

**3. No append-and-re-ingest backups.** The four `backup_consensus_data_*.csv` files were produced by re-reading the entire previous file and appending a fresh re-parse. Generation 4 is ~75% self-duplication: 21,409,686 of its 28,487,256 rows are re-ingested copies of ~7.08M unique rows. Replace with either (a) a single parquet dataset partitioned by ingest date, where a new run adds a partition, or (b) content-addressed snapshots. Retain **N=2** generations, not four.

**4. Log rotation is mandatory.** No log file may exceed 100 MB uncompressed. Rotate at 100 MB, compress rotated segments with zstd, retain the most recent 5 segments plus any log designated as provenance. `docking_automation.log` reached 448 MB and a second copy of 13.8 MB sits inside `consensus_docking_results/`, where it also breaks directory-listing arithmetic (24,174 entries = 24,173 job dirs + 1 log file — an off-by-one that propagated into three separate analyses).

**5. Provenance logs are exempt from rotation and from deletion.** `uniprot_preparation.log`, `alphafold_extraction.log`, `cavity_extraction.log`, `required_structures.log` and `negative_samples.log` are small, are the only record of how the artifacts came to exist, and in the case of `uniprot_preparation.log` are the decisive forensic evidence for the mapping bug. Tag them and back them up off-drive.

**6. Graph serialisations are derived, not stored.** The six `.gexf` network files total 12.78 GB and nothing reads them; the six cluster-assignment CSVs that *are* consumed total 3.07 MB — a 4,161× ratio. Write the assignment table; regenerate the graph on demand if a visualisation is needed.

**7. Streaming is the default read pattern.** Use `pyarrow.ParquetFile.iter_batches` with an explicit column projection. Row-group statistics answer schema, null-count and min/max questions with zero data read. One practical note for this dataset: polars' streaming engine panics on `combined_filtered_annotated_docking_results.parquet` (`validity must be equal to the array's length` in binview) — use pyarrow.

**8. Compress at rest what you cannot delete.** The doctrine's largest single win is not a deletion at all: 1,251 GB of pose complexes compress ~120–150:1 because ~99.75% of each file is a receptor re-serialised from its own siblings. Any artifact where the same content is repeated across files should be stored as a per-group archive, not as loose files.

---

## Method and confidence

Everything above was verified read-only. My own checks this session: parquet metadata for both combined files; xz compression on seven independent (job, tool) groups; SDF-vs-`results.csv` field comparison on gnina output; a direct pose-count comparison between the legacy and consensus copies of `DB00253_NR3C1_P04150_cavity_7`; exact `du -sb` on `converted_pdbqt`, `fixed_structures`, `cavityspace_structures`, the three `pocketmatch_results` subdirectories, `dta_atlas_data` and `drugbank_approved_split`; an exact sum of all 25 top-level log files (745,534,762 B); and a line-by-line read of the iteration logic in `extract_cavities.py`, `extract_alphafold_models.py`, `download_cavityspace_structures.py` and `identify_required_structures.py`. All figures I could check exactly matched the verifier's, including the three the finder got wrong.

**Confidence is high for exact figures and moderate for sampled ones.** `consensus_docking_results/` is a sampled estimate — 1,404 GB [1,308–1,500] by one method, 1,441 GB [1,368–1,514] by another — so every percentage of that directory (the 1,251 GB pose figure, the 124.5 GB input figure, the 19.7 GB native-pose figure) inherits ±5%. The composition split rests on a 60–220 directory walk. This does not change any ordering or any keep/delete decision, but do not quote the GB figures to three significant digits.

**One open question I could not close:** `merged_pocketmatch_results.txt` contains ~10.6M more pairs than `parquet_data/` while carrying only the Pmax column. Until someone runs a pair-key comparison, it stays on the never-delete list. Resolving it would recover ~6.3 GB by conversion.

No file under `/media/onur/Elements` or `/home/onur/repos/dock_selectivity` was created, modified, or removed. Scratch work and the verification transcript are in `/tmp/claude-1001/-home-onur-repos-dock-selectivity/6b51b73b-17d0-4e12-8d75-c188a82baeaa/scratchpad/doctrine_verification.txt`. Note that `/home/onur/repos/dock_selectivity/CLAUDE.md` appeared as untracked during this session (mtime 19:01, before my first command) — it was not created by me.

---


# Global Repository Fix Plan

**Scope:** `/home/onur/repos/dock_selectivity` (7 implicated scripts + 5 supporting). This is a plan; no file was edited. All line numbers below were read and verified against the working tree at `cad1ccb`.

**The failure in one sentence.** The pipeline used a *display string* (gene symbol) as a *join key*, resolved it in the wrong direction (symbol → free-text search → accession), collapsed the result with a last-wins `dict(zip(...))`, and then let that accession select the physical receptor — so 633 of 979 docked proteins (64.7%) are not the protein their job name claims, while the correct accession sat in the same CSV, correctly labelled, in 100% of the failure cases.

**The class of failure, stated generally.** *Any* identity derived by matching human-readable names, and *any* stage that silently degrades to a different universe when its preferred input is missing. This plan fixes both, not just the UniProt bug. Two further instances found while reading the code confirm the generalisation is warranted:

| Instance | Location | Silent effect |
|---|---|---|
| Drug axis has the same defect | `1_drug_to_protein.tsv` `node_1` holds 222 rows (110 distinct) that are **name slugs**, not DrugBank IDs (`clorprenaline`, `neridronic-acid`, `SN-38`); every stage filters with `node_1.isin(small_molecule_ids)` | 222 edges vanish with no log line. 74/110 resolve uniquely against the local DrugBank vocabulary, 1 is ambiguous, 35 unresolved |
| Tool names are string literals in three places | `parse_prepare.py:748` `'LeDock'`, `:1335` `['GOLD','Smina','LeDock','Gnina']` vs data `'SMINA','GNINA','LEDOCK','GOLD'` | `filter_complete_tool_coverage` can match at most 1 of 4 expected tools, so it finds **zero** complete pairs and is a silent no-op — which is why the artifact's `processing_steps_applied` claim does not describe what ran |

---

## 1. The root fix — exact identifier resolution

### 1.1 Identity model (the rule everything else follows)

| Axis | Canonical key | Resolved from | Never used as a key |
|---|---|---|---|
| Protein | UniProt accession (`P30518`) | **Entrez Gene ID** (`node_2`, present and numeric on 8,568/8,568 edges, strictly 1:1 with the symbol) | gene symbol, entry name, any `*_name` column |
| Compound | DrugBank ID (`DB00035`) | DrugBank ID; name slug only via an explicit, ambiguity-aware resolver | drug name |
| Pocket | `(accession, fragment_id, cavity_index)` | CavitySpace/AlphaFold filename parse | folder-name substring |

Three directional laws:

1. **Resolve once, at the ingest edge.** Symbol/Entrez → accession happens in exactly one module. Downstream stages receive accessions and never re-resolve.
2. **Labels flow the other way only.** `accession → primary_symbol`, computed on demand for display. A label is never an input to selection.
3. **A many-to-one map may not be inverted.** `identify_required_structures.py:196` does `{v: k for k, v in gene_to_uniprot.items()}` — inverting a collapsed dict. Forbidden by construction once labels come from the index.

### 1.2 The new module

New package `dock_selectivity/` sitting next to the scripts (so `python $REPO/script.py` picks it up on `sys.path[0]` with no install step):

```
dock_selectivity/
  identity.py     IdentityIndex, Protein, Resolution      <- the root fix
  compounds.py    CompoundIndex, Compound                 <- the drug axis
  invariants.py   named checks + InvariantViolation
  joins.py        checked_merge()
  jobid.py        JobIdentity codec (format/parse/manifest)
  schemas.py      inter-stage CSV/parquet contracts
  config.py       RunConfig (TOML + env + CLI)
build_identity_index.py   <- only script permitted to touch the network
audit.py                  <- read-only invariant runner over an existing run dir
```

### 1.3 Inputs — the authoritative mapping files to obtain

There is **no** local ID-mapping file today (`ls /opt/data` → `alphafold2, cavity_space, drugbank, multiscale_interactome_data, ttd`; no `idmapping`). One manual fetch, checksummed, then offline forever.

| File | Source | Size | Provides | Status |
|---|---|---|---|---|
| `HUMAN_9606_idmapping.dat.gz` | UniProt FTP `knowledgebase/idmapping/by_organism/` | ~50 MB gz | accession ↔ `GeneID` (Entrez), `Gene_Name`, `Gene_Synonym`, `UniProtKB-ID` | **required** |
| `UP000005640_9606.fasta.gz` | UniProt reference proteomes | ~11 MB | `sp\|`/`tr\|` review status + the canonical one-per-gene set | **required** (canonicalisation tie-break) |
| `hgnc_complete_set.txt` | HGNC | ~15 MB | `prev_symbol`, `alias_symbol`, withdrawn-symbol detection, independent Entrez↔UniProt cross-check | recommended |
| `/opt/data/alphafold2/UP000005640_9606_HUMAN_v4.tar` | already local | — | dockable-accession universe; consistency check | already present |

`build_identity_index.py` consumes these and emits **one** artifact into the run directory: `identity_index.parquet` + `identity_index.meta.json` (source URLs, UniProt release, per-file SHA256, row counts, build timestamp, full invariant report). Every downstream script pins the index's SHA256 in its provenance record and refuses to run against an index it does not expect.

### 1.4 Guarantees (contracts, enforced at build time — not aspirations)

| ID | Guarantee | Enforcement |
|---|---|---|
| G1 | `accession` is unique — one row per accession | assert at build |
| G2 | Total over the study universe: every `node_2` Entrez either resolves or appears in `unresolved_targets.csv` with a reason. No silent drops | assert `n_in == n_resolved + n_rejected` |
| G3 | Deterministic: same inputs → byte-identical output. No `set()` iteration, no insertion-order dependence, no `drop_duplicates` over an unordered concat. All tie-breaks total: reviewed > unreviewed → in-AF-proteome > not → lowest accession lexicographically | subprocess test under 3 `PYTHONHASHSEED` values |
| G4 | Self-consistent: `primary_symbol ∈ symbols` for every row; `entrez(accession(e)) == e` for every resolved Entrez | assert at build |
| G5 | `entrez → accession` is a *function* (collapse factor exactly 1). `accession → entrez` is single-valued or null | assert `max_collapse == 1` |
| G6 | No free-text search exists in the codebase | grep-based lint test in CI |
| G7 | Isoform accessions (`P12345-2`) excluded; non-human excluded | assert at build |

G5 is the specific guarantee whose absence produced this incident: the old map had mean collapse 2.55, max 9.

### 1.5 API

```python
@dataclass(frozen=True)
class Protein:
    accession: str          # 'P30518'
    entry_name: str         # 'V2R_HUMAN'
    primary_symbol: str     # 'AVPR2'
    symbols: tuple[str,...] # ('AVPR2','ADHR','DIR','DIR3','V2R')
    entrez_id: int | None
    reviewed: bool

@dataclass(frozen=True)
class Resolution:
    status: Literal['OK','AMBIGUOUS','UNKNOWN','OBSOLETE','NO_KEY']
    protein: Protein | None
    reason: str             # always populated for non-OK

class IdentityIndex:
    @classmethod
    def load(cls, path) -> 'IdentityIndex': ...

    # exact lookups — the only sanctioned resolution direction
    def by_entrez(self, gene_id: int) -> Protein          # raises KeyError
    def by_accession(self, acc: str) -> Protein           # raises KeyError
    def resolve_entrez(self, gene_id) -> Resolution       # never raises; caller must branch

    # label direction only
    def label(self, acc: str) -> str                      # accession -> primary symbol

    # audit-only; name makes misuse obvious in review
    def by_symbol_ambiguous(self, symbol: str) -> tuple[Protein, ...]

    def assert_self_consistent(self) -> None
    def cardinality_report(self) -> dict
```

`Resolution` exists so that `gene_to_uniprot.get(gene)` followed by `if not uniprot_id: continue` (`identify_required_structures.py:170-174`, `generate_negative_samples.py:190-193`) becomes impossible to write without also deciding where the rejected row is recorded.

`CompoundIndex` mirrors this over `/opt/data/drugbank/drugbank_all_drugbank_vocabulary.csv/drugbank vocabulary.csv` (16,581 rows: DrugBank ID, Common name, Synonyms, InChIKey), with `resolve_name()` doing normalised exact matching and returning `AMBIGUOUS` rather than picking one.

### 1.6 Offline constraint

`build_identity_index.py` is the sole network consumer, run manually, refusing to re-fetch unless `--refresh`. Every other entrypoint asserts the index exists and hashes as expected, and fails with an actionable message otherwise. Consequence for coverage: the old query-list ceiling (1,795 reachable labels, all pre-existing drug targets) disappears — the reachable universe becomes the reference proteome ∩ CavitySpace, i.e. 13,600 accessions with cavities against the 979 actually used.

---

## 2. Per-file change list, in dependency order

### 2.1 `prepare_uniprot_mapping.py` — **retire, do not patch**

Three independent defects, none patchable in place because the direction of resolution is wrong:

| Line | Defect |
|---|---|
| `:73` + `:88-91` | `service.get_df(gene)` is a free-text search; the queried symbol is broadcast to every human row returned |
| `:51` | `list(set(gene_names_list))` — hash-randomised order makes the output non-reproducible across processes |
| `:130` | `pd.concat(dfs).drop_duplicates(subset=['Entry'])` keeps the *first-queried* claimant; combined with `:51`, whichever query happened to run first permanently steals the accession. Per the run log, 1,992 of 6,577 retrieved rows (30.3%) were destroyed and 370 genes lost their own correct row |
| `:243` | `.unique().tolist()` admits `NaN`, queried as gene #1, and the run later exited non-zero on a `str`/`float` comparison — the shipped file comes from a failed run that never wrote its summary |

**Action:** `git mv prepare_uniprot_mapping.py deprecated/`, add a `sys.exit(2)` banner at the top pointing at `build_identity_index.py`. Do **not** delete the existing `uniprot_gene_mapping.csv` in the run directory and do **not** attempt to regenerate it — it is non-reproducible by construction and is now *evidence*. Keep it read-only.
**Invariant established:** no code path can produce a symbol-keyed mapping file again.

### 2.2 `build_identity_index.py` — new

Emits `identity_index.parquet` + meta. **Invariant:** G1–G7 above, all asserted at build, all reported in the meta JSON.

### 2.3 `identify_required_structures.py`

| Line | Change |
|---|---|
| `:140-142` | `node_1.isin(small_molecule_ids)` → `CompoundIndex` resolution; unresolved drug tokens go to a rejects file, not to `/dev/null` |
| `:146` | **delete** `gene_to_uniprot = dict(zip(uniprot_df['Mapped_Gene_Name'], uniprot_df['Entry']))` |
| `:165-175` | iterate `filtered_interactions['node_2']` (Entrez) → `idx.resolve_entrez()`; branch on status; append non-OK to `required_structures.rejects.csv` |
| `:191` | `value_counts()` on `node_2`, not `node_2_name` |
| `:195-197` | **delete** the inverted dict; `Gene_Name = idx.label(accession)` |
| `:209` | add `entrez_id` and `selected_by` columns to the output schema |

**Invariants:** every row's `Gene_Name` is a symbol of its own `UniProt_ID`; `Interaction_Count` equals the edge count of that row's own `entrez_id`; `n_edges_in == n_resolved + n_rejected`.
*Would have caught, day one:* 1,842/2,829 rows (65.1%) mislabelled; `Q14562/DHX8` sitting at the top of the file carrying HRH1's 98 antihistamine interactions.

### 2.4 `generate_negative_samples.py`

| Line | Change |
|---|---|
| `:152`, `:303` | the same corrupt dict built twice — replace both with one shared call; better, extract positive-set construction and known-pair construction into one function so they cannot drift |
| `:190` | `gene_to_uniprot.get(gene_name)` → `idx.resolve_entrez(row['node_2'])` |
| `:303-319` | `known_pairs` built in accession space *from Entrez*. After G5 this class of leak vanishes by construction (the 5 observed false negatives all arose because a drug's real target entered the pool under a *different* gene's resolution) — keep the assertion anyway |
| `:204-208` vs `:372` | Fragment_ID is normalised to `F{n}` for positives but copied raw for negatives → `F1` vs `1`. Normalise in one place |
| `:375` | `row.get('Gene_Name','')` against `cavity_mapping.csv`, which has no such column → **13,452/13,452 negatives have a blank gene name**, producing 10,698 directories named `*_nan_*`. Use `idx.label(uniprot_id)` |
| `:410` | duplicate check widened from `(drug, acc, cavity)` to `(drug, acc)` |
| `:555-566` | add an already-emitted `(drug, target)` guard — 3,596 of 13,452 negative rows are repeats of an existing pair |
| `:518` | `list(protein_cavities.keys())` → `sorted(...)` for explicit determinism |
| `:388-423` | `validate_negatives()` must check against ground truth re-derived from the edge list, **not** against the same `known_pairs` object it was built from (see §3, principle) |

**Invariants:** `Gene_Name` non-null on 100% of rows and always a symbol of its own accession; `known_pairs ∩ negative_pairs == ∅` in ground-truth space; no duplicate `(drug, accession)`.

### 2.5 `extract_alphafold_models.py`, `fix_required_pdbs.py`, `convert_pdb_to_pdbqt.py`, `extract_cavities.py`

Not identity consumers, but they carry the same *silent-fallback* defect:

| File | Line | Change |
|---|---|---|
| `extract_alphafold_models.py` | `:194-235` | `load_required_structures()` silently falls back to `cavity_mapping.csv` — the 2025-07-02 00:52 run took that path (13,600 accessions / 13,975 fragments), was aborted, and left 5,661 accession dirs where 979 were needed (5.8× over-extraction, 2.15 GB). Make the required list **mandatory**; `--all-structures` becomes an explicit opt-in |
| `extract_cavities.py` | `:63-71` | unpacks every tarball in the folder (38,709), of which 2,829 were required (13.7× over-extraction, 76.6 GB waste). Accept an optional required-accession filter; build the mapping from tarball *names* without unpacking payloads |
| `fix_required_pdbs.py`, `convert_pdb_to_pdbqt.py` | — | no logic change; add schema assertions on read and write |

**Invariant:** a stage that cannot obtain its intended input fails; it never proceeds against a larger or different universe.

### 2.6 `batch_dock.py`

| Line | Change |
|---|---|
| `:34-58` | duplicated import block (os, sys, pandas, subprocess, re, time, logging imported twice) — delete |
| `:363-393` | **delete** `load_uniprot_mapping()`; `batch_dock` must not resolve identity at all |
| `:2034` | **delete** `uniprot_gene_map = dict(zip(...))` |
| `:2057-2069` | **delete** the symbol-lookup fallback branch. If `required_structures_with_negatives.csv` is absent → hard fail. This dormant path would re-fire the identical bug verbatim |
| `:448-449` | `if 'Gene_Name' not in df.columns: df['Gene_Name'] = ''` → hard schema assertion |
| `:2089` | `job_signature = f"{drug}_{uniprot}_{cavity}"` → include `fragment_id`. Fragment-blind dedup collapsed 1,309 distinct pockets |
| `:2201-2211` | folder name stops being the data contract; write `job_manifest.json` into each job dir at creation (§5.1). `:2207`'s `re.sub(r'AF-[A-Z0-9]+(?:-F1-model_v1_)?','',…)` strips only the `-F1-` form, producing 1,743 malformed names |
| `:76` | `OUTPUT_BASE_FOLDER` absolute path → `RunConfig` |

**Invariants:** the accession docked equals the accession in the plan row; the receptor path contains that accession; folder name round-trips through the codec; `n_job_dirs == n_planned_jobs` both directions.

### 2.7 `create_ic50_mapping.py`

This is the one consumer that was already correct — `:84` uses the clean `Entry Name → Entry` bijection and never touches `Mapped_Gene_Name`.

| Line | Change |
|---|---|
| `:84` | source the bijection from `identity_index` instead of `uniprot_gene_mapping.csv`. This lifts the coverage ceiling from 1,672/3,685 TTD targets (45.4%) to the full proteome — the 2,013 dropped targets were dropped only because the buggy search never returned them |
| `:30-31` | absolute run-dir paths → `RunConfig` |
| `:34-68` | TTD `UNIPROID` fields: 121–123 are multi-valued (`;`, `,`, `/`, `\|`, spaces) and can never match an exact dict lookup — split, resolve each, and reject explicitly |
| `:110-135` | `parse_ic50_value` is pure — becomes the first unit-tested function in the repo |

**Invariant:** `n_activity_rows == n_mapped + Σ(skip reasons)`, with the ledger written to JSON. Note the emitted column named `ic50_value` also carries 3,202 Ki values and 2,342 inequality-bounded records; rename or add `measurement_type` to the join key downstream.

### 2.8 `parse_prepare.py`

| Line | Change | Why it matters |
|---|---|---|
| `:146` | `max_files_per_pattern = 10000` → unbounded, and assert the discovered job count equals the plan | **Only 10,000 of 24,173 job dirs entered the parquet.** ~13,975 of the 14,173 excluded dirs are complete jobs with `final_results.csv` — 58% of the campaign's compute never reached the analysis |
| `:359` | regex identity extraction → read `job_manifest.json`; keep the regex as a fallback that **raises** with the offending path, never stamps `'Unknown'` | 1,743 fragment dirs + 34 hyphenated-symbol dirs (`MT-ND6`) fail it; 1,174,786 rows (7.08%) carry no identity and share one fabricated `compound_target_pair` value `<NA>_<NA>` |
| `:748` | `tools = ['GOLD','SMINA','LeDock','GNINA']` → `Tool` enum from one constants module | `'LeDock'` never matches the data's `'LEDOCK'` |
| `:1335` | `expected_tools = ['GOLD','Smina','LeDock','Gnina']` → same enum | at most 1 of 4 tools can match, so `filter_complete_tool_coverage` is a **silent no-op** |
| `:783-784`, `:801`, `:813` | cavity-cluster key drops the fragment | 258 `(acc,cavity)` keys come from >1 fragment, 241 land in different clusters, 421 items discarded by `drop_duplicates` |
| `:853-861` | `min('ic50_value')` paired with `first()` of `measurement_type`/`operator` | latent today (168 multi-measurement pairs, 0 where `idxmin != first`), guaranteed to bite on regeneration |
| `:866`, `:818`, `:1171`, `:1291` | every `merge` → `checked_merge(..., validate='m:1', unmatched_budget=…)` | `:1171` merges on a **non-unique** key (1,428 duplicate keys) and silently duplicated 410,087 rows, one job 12× |
| `:1245` | hardcoded `F1` in the constructed `pocket_id` | 1,992 of 13,330 clustered pockets (14.9%) are unreachable |
| `:1359-1414` | `balance_positive_negative_samples` drops all null-`sample_type` rows and equalises classes — the artifact has neither property, further proof HEAD ≠ producing code | reconciliation, not a fix |
| `:1416-1512` | `processing_steps_applied` is a hardcoded literal | must be *recorded* from what actually ran |
| `:39-50`, `:765`, `:836`, `:1147` | absolute paths → `RunConfig` | |

**Invariants:** rows-in == rows-out + ledger; parquet job coverage == plan; `uniprot_id` non-null on 100% of rows; `gene_name` is a symbol of `uniprot_id` on 100% of rows.

### 2.9 `run_full_workflow.py`

| Line | Change |
|---|---|
| `:93-139` | **`generate_negative_samples.py` is never invoked** — the runner and the documented pipeline disagree. Add it as step 4 |
| new step 0 | `build_identity_index.py` (or verify hash) + `validate_run_manifest.py` gate |
| `:110-139` | steps 3–6 only `logging.warning` and continue on failure — precisely how a run proceeds on a fallback-derived structure list. Every step fatal by default; `--allow-partial` to opt out |
| `:42-59` | `subprocess.run(capture_output=True)` swallows all child output unless the child fails (and then only stderr) — stream and tee to the stage log |
| `:145-158` | prerequisite check tests only file *existence* — extend to schema + identity-index hash |
| all | `--run-dir` / `--config`, passed through to every child |

---

## 3. Invariants and assertions

`dock_selectivity/invariants.py` provides named checks that each return a report dict, raise `InvariantViolation` unless `--warn-only`, and write `<stage>.invariants.json` next to the stage output.

**Governing principle, learned from this incident:** *an assertion that consumes the same derived object it is validating cannot fail.* `validate_negatives()` (`generate_negative_samples.py:388-423`) re-checked negatives against the very `known_pairs` set that had been built with the corrupt map, and reported "All validations passed" while emitting real interactions as negatives. Every check below is specified to re-derive its ground truth from a *primary* source.

| ID | Invariant | Where it goes | Value it would have printed on 2025-07-04 |
|---|---|---|---|
| I1 | **Identity self-consistency** — for every emitted `(accession, label)`, `label ∈ symbols(accession)` | `IdentityIndex.assert_self_consistent()`; re-checked in `identify_required_structures`, `generate_negative_samples`, the job manifest writer, and the parquet export | **2,647 / 4,585 = 57.7% violations** |
| I2 | **Cardinality / collapse report** — `{n_input_keys, n_resolved, n_distinct_accessions, mean_collapse, max_collapse, n_unresolved}`, asserting `max_collapse == 1` for a function-typed map | build step + every stage that builds a lookup | mean **2.55**, max **9**, 1,191/1,795 labels with >1 accession |
| I3 | **Round-trip** — `entrez → accession → entrez` is the identity for 100% of resolved rows; `accession → label → resolve(label)` reported with an explicit ambiguity count (not asserted; symbols are legitimately many-to-one) | build step | would have shown 1,056/1,795 labels resolving to a different protein |
| I4 | **Join-key validation** — `checked_merge(left, right, on, how, validate, unmatched_budget, name)`: asserts `validate`, counts unmatched left rows, fails past a declared budget, logs the ledger | replaces every `.merge()` in `parse_prepare.py` (`:818, :866, :1171, :1291`, DTA) and `identify_required_structures.py:195-200` | 1,428 non-unique keys → **410,087 duplicated rows (2.47%)**; 1,174,786 unmatched (7.08%) |
| I5 | **No-silent-drop ledger** — each stage writes `<stage>.rejects.csv` and asserts `n_in == n_out + n_rejected` | every stage | 237 symbol-less edges, 222 slug drug rows, 163 genes with no output, **14,173 job dirs never read** |
| I6 | **Determinism** — no `set()`/dict iteration over identifiers; index rebuilt in subprocesses under differing `PYTHONHASHSEED` must be byte-identical | build step + a unit test | would have failed immediately on `prepare_uniprot_mapping.py:51` |
| I7 | **Selection provenance** — every accession chosen for docking records *why* (`selected_by: entrez=554 via edge (DB00035, 554)`), asserted non-null | `required_structures.csv` schema | would have made `AVPR2 → Q8NCT1` legible in the plan file itself |
| I8 | **Plan ↔ disk ↔ artifact reconciliation** — `set(job_dirs) == set(plan)` both directions; parquet job coverage == plan | end of `batch_dock`, start of `parse_prepare` | 24,173 == 24,173 ✓ but parquet coverage **10,000 / 24,173** ✗ |
| I9 | **Schema contract** — declared columns, dtypes, key uniqueness for every inter-stage CSV, validated on read *and* write | `schemas.py`, called by every loader/writer | `cavity_mapping.csv` has no `Gene_Name` column → would have failed `generate_negative_samples.py:375` at once |
| I10 | **Label/key separation** — a lint test asserting no `merge`/`isin`/`dict(zip(...))` uses a column matching `*gene*`/`*name*`/`*symbol*` | CI grep test | would have flagged all four `dict(zip(Mapped_Gene_Name, Entry))` sites |
| I11 | **Negative-set integrity, re-derived** — `known_pairs ∩ negative_pairs == ∅` computed from the edge list via the index, independently of the object under test | `generate_negative_samples.validate_negatives()` | 5 real interactions emitted as negatives, e.g. `DB06589`/`P07333` (CSF1R) |
| I12 | **Tool-name closure** — every `Tool1`/`Tool2` value is a member of the `Tool` enum | `parse_prepare` cleaning | would have caught `'LeDock'` vs `'LEDOCK'` and the no-op coverage filter |

---

## 4. Test strategy — minimum viable, for a scientific pipeline

Today: no tests, no linter, no `pyproject.toml`, no dependency pin. The proposal is four tiers; tiers 0–2 are the whole ask, tier 3 is the pre-re-run gate.

### Tier 0 — mechanical gates (one afternoon, immediate return)

`pyproject.toml` with pinned deps + `ruff` config; `Makefile` target `make check` = `python -m compileall -q .` + `ruff check .` + `pytest -q`; wired to a `pre-push` hook (and a 10-line GitHub Action if available). This alone would have caught `pocketmatch_clustering.py:426`'s stray `tmux` token the day it was committed — that file has not parsed since, so every `pocket_louvain_clusters_*.csv` in the run directory predates the edit and the Louvain columns in the parquet have unverified provenance.

### Tier 1 — pure-function unit tests (fast, no I/O, no external drive)

Functions that are already pure or need ≤3 lines of extraction:

| Target | Test | Rationale |
|---|---|---|
| `IdentityIndex.resolve_entrez` / `by_accession` / `label` | 20-row in-memory fixture incl. a 2-accession gene, an unreviewed duplicate, an obsolete Entrez | the root fix |
| `jobid.format()` / `jobid.parse()` | **property test** over generated identities: hyphenated symbols (`MT-ND6`), `None` symbol, 10-char accessions, fragment > 9, cavity > 9, symbol containing `_` | highest-value test in the repo — catches both naming bugs at once |
| `CompoundIndex.resolve_name` | normalisation + ambiguity table | recovers 74/110 slugs, flags 1 ambiguous |
| `parse_ic50_value` (`create_ic50_mapping.py:110`) | table-driven: `IC50 > 10000 nM`, `Ki = 2390 nM`, µM/uM/μM, missing unit, malformed | already pure |
| `add_tool_specific_scores` (`:737`) | 6-row frame covering all 6 tool pairs + case variants | catches I12 |
| cavity-id regexes (`extract_cavities.py:82`, `parse_prepare.py:801`, `:1242`) | F1 and non-F1, PDB-derived items that must be excluded | fragment bugs |
| `remove_duplicate_jobs` + signature builder | two jobs differing only by fragment must both survive | the 1,309-pocket collapse |
| `generate_balanced_negatives` — pure core extracted | no known pair emitted; no duplicate `(drug,target)`; balance reached; identical under a fixed seed | negative-set integrity |

### Tier 2 — golden-file fixtures

`tests/fixtures/mini_run/` — ~50 KB, committed, no network, no external drive: a 30-edge `1_drug_to_protein.tsv`, a 40-row `identity_index.parquet`, a 12-row `cavity_mapping.csv`, 3 stub SDFs, 2 canned `final_results.csv`. Golden outputs for `identify_required_structures`, `generate_negative_samples`, and each `parse_prepare` annotation stage, compared with `assert_frame_equal`.

**The fixture must be adversarial, and every adversarial row is a real failure from this run:**

- a gene with 2 accessions where the wrong one sorts first (the last-wins trap);
- a symbol that is a listed synonym of a *different* gene (`TUBA1B` vs `Q13509`/TUBB3);
- a hyphenated symbol (`MT-ND6`) and a null symbol with a valid Entrez;
- a drug supplied as a name slug;
- a multi-fragment protein carrying the same cavity index on F1 and F4;
- a TTD `UNIPROID` with two accessions separated by `;`.

That directory *is* the regression suite for this incident.

### Tier 3 — end-to-end smoke run

`make smoke` → `run_full_workflow.py --config tests/fixtures/mini_run/config.toml` with `CONSENSUS_DOCKER_SCRIPT` pointed at a stub that writes a canned `final_results.csv`. Target < 60 s. Asserts: plan size, folder-name round-trip, every job dir has a manifest, parquet row count and job coverage == plan, all invariants green, index hash recorded. This is the gate that must be green before any re-run is launched.

### Tier 4 — read-only auditor against the real run

`python audit.py --run-dir <path>` re-runs the invariant suite over an existing run directory without writing to it. Two uses: (a) validating the *repaired* run without re-docking; (b) **self-test** — it must reproduce the already-established numbers (57.73% mislabelled rows, 64.66% mislabelled targets, 45.69% of job dirs, 10,000/24,173 parquet coverage). If the auditor cannot reproduce the known corruption, the auditor is wrong and nothing downstream of it can be trusted.

**Explicitly out of scope:** unit-testing docking engines, mocking `subprocess` docking, coverage targets. Aim for ~90% on `identity.py`/`jobid.py`, ~80% on the parsers, and linting only for the staged executor in `batch_dock.py`.

---

## 5. Structural debt, ranked by risk

Ranked by *probability of silently corrupting a future run × cost of that corruption*.

### 5.1 (Highest) The job-folder name as a load-bearing data contract

`batch_dock.py:2211` writes `{drugbank_id}_{gene_name}_{uniprot_id}_{pocket_suffix}`; `parse_prepare.py:359` parses identity back out with `r'(DB\d+)_([A-Za-z0-9]+)_([A-Z0-9]+)_cavity_(\d+)'`. It has already destroyed 1,174,786 rows (7.08%), and it is the *only* carrier of identity into the analysis. Two structural sins: a nullable display field (`gene_name`, literally `nan` for 11,806 dirs) sits inside a positional key, and the delimiter `_` is a legal character in the payload.

**Fix:** write `job_manifest.json` into each job directory at creation — schema-versioned, carrying `drugbank_id`, `uniprot_id`, `entrez_id`, `fragment_id`, `cavity_index`, `sample_type`, `label`, receptor/pocket paths, config hash, index hash, git SHA. `parse_prepare` reads the manifest. The folder name stays human-readable but becomes non-load-bearing; where legacy dirs must be parsed, use the `jobid` codec with a round-trip property test and a hard failure on mismatch.

### 5.2 Silent fallbacks that change the scientific universe

A family, not a single bug: `parse_prepare.py:146` (10,000-file cap), `extract_alphafold_models.py:194-235` (falls back to `cavity_mapping.csv` → 5.8× over-extraction), `batch_dock.py:2057-2069` (falls back to symbol resolution), `load_best_mapping_with_pdbqt` (silently degrades receptor quality), `run_full_workflow.py` (warn-and-continue on 4 of 7 steps). **Principle: a fallback that changes which proteins are studied is not a fallback, it is a different experiment.** Every one becomes explicit opt-in plus a loud provenance record.

### 5.3 Module-level config constants; no CLI, no config file

A run's parameters are recoverable only from the git SHA *plus* the working tree — and the committed state has all four docking engines off, so the file cannot reproduce its own historical runs. **Fix:** `RunConfig` dataclass loaded from `run.toml` in the run directory, env override, `--config` on every entrypoint; existing constants become the schema defaults (a mechanical diff); every script dumps its fully-resolved config next to its output.

### 5.4 Absolute paths hardcoded to one external drive

`parse_prepare.py:39, 765, 836, 1147`; `batch_dock.py:76`; `create_ic50_mapping.py:30-31`; `pocketmatch_clustering.py` `--csv-dir` default; notebook `DATA_FILE` constants. **Fix:** all paths derive from `RunConfig.run_dir` / `data_root`; `DOCK_SELECTIVITY_RUN_DIR` env var; one `paths.py`. Add a startup guard that refuses to write outside the configured run directory.

### 5.5 The repo-vs-run-directory split

Scripts use relative paths and must be invoked with cwd == run directory — undocumented in the scripts themselves and unenforced. **Fix:** entrypoints resolve everything from `RunConfig`, never cwd; `--run-dir`; startup assertion on required inputs *and* their recorded index hash; ship `run.toml.example`. This is also what makes the tier-2 mini-run fixture possible.

### 5.6 `pocketmatch_clustering.py:426` — stray `tmux` after a string literal

`help='Directory containing CSV similarity files'tmux` → the module does not parse (`python -m compileall` confirms; every other `.py` compiles). A one-character fix, but its survival is the proof that nothing here is ever syntax-checked — the real fix is §4 tier 0. **Consequence to record:** the file cannot have run since that edit, so the Louvain cluster CSVs in the run directory predate it and the corresponding parquet columns have unverified provenance; re-derive after the fix.

### 5.7 `fix_pdb_files.py` — 0 bytes

Delete. It shadows the working `fix_required_pdbs.py` in the file listing and in the reader's mind.

### 5.8 `batch_dock.py:34-58` — duplicated import block

`os, sys, pandas, subprocess, re, time, logging` imported twice. Cosmetic, but it is the signal that a 115 KB single-module script has never been reviewed. Extract only the identity/job-planning portion (the part this plan changes); do **not** attempt to rewrite the staged executor.

### 5.9 No dependency pinning or environment lock

`created_by parquet-cpp-arrow 16.1.0` in the artifact is unverifiable against today's environment. Add `pyproject.toml` plus a committed `environment.yml` (`conda env export --no-builds`).

### 5.10 HEAD is not the code that produced the artifacts

Independently established: the parquet contains `gene_name` values like `MT-ATP6` that HEAD's `([A-Za-z0-9]+)` capture cannot produce; the column is `Ledock_Score` where HEAD writes `LeDock_Score`; the summary's `processing_steps_applied` omits steps HEAD hardcodes. **Fix as process, not code:** every artifact carries `provenance.json` with `git rev-parse HEAD`, dirty-tree flag, resolved config, and input hashes; writing into a run directory from a dirty tree requires `--allow-dirty`.

---

## 6. Sequencing

### Phase 0 — foundations (no re-run; invalidates nothing)

| Step | Action |
|---|---|
| P0.1 | `ruff`/`compileall` gate; fix `pocketmatch_clustering.py:426`; delete `fix_pdb_files.py` |
| P0.2 | Land `dock_selectivity/` + unit tests + the adversarial mini-run fixture. Nothing consumes it yet |
| P0.3 | Run `build_identity_index.py` once (**the only network step in this plan**); checksum the sources; the pipeline is offline thereafter |
| P0.4 | Land `audit.py` and run it read-only against the current run directory. **Gate:** it must reproduce 57.73% / 64.66% / 45.69% / 10,000-of-24,173 before the invariant code is trusted to gate anything |

### Phase 1 — re-derive the plan (minutes of compute; invalidates plan artifacts only)

| Step | Action | Invalidates |
|---|---|---|
| P1.1 | Retire `prepare_uniprot_mapping.py`; preserve the existing CSV read-only as evidence (it is non-reproducible — do not attempt regeneration) | `uniprot_gene_mapping.csv` superseded |
| P1.2 | `identify_required_structures.py` on Entrez | `required_structures.csv` + everything downstream |
| P1.3 | `generate_negative_samples.py` | `required_structures_with_negatives.csv`, `negative_samples_metadata.json`. **The new negative set will differ; old negatives must not be mixed with new positives** |
| P1.4 | `create_ic50_mapping.py` on the index | `ic50_mapping.csv` — strictly better, independent of docking, can land immediately and be joined onto existing data for the audit |

**Deliverable of Phase 1:** a corrected target list *and* a diff against the current one, which turns the salvage ledger (2,235 job dirs relabelable, 8,810 on non-targets, 6,185 intended pairs never correctly docked, 4,196 of them re-dockable today) into a reproducible artifact instead of a forensic estimate.

### Phase 2 — structure prep (cheap; existing files largely reusable)

Remove the `cavity_mapping` fallback from `extract_alphafold_models.py`; add the required-accession filter to `extract_cavities.py`; schema assertions in `fix_required_pdbs.py` / `convert_pdb_to_pdbqt.py`. Structures already extracted for surviving accessions are byte-identical and are merely re-indexed; only newly required accessions need extraction. **Invalidates:** `alphafold_mapping.csv`, `fixed_mapping.csv`, `pdbqt_mapping.csv` (index files only, not the PDB/PDBQT payloads).

### Phase 3 — `batch_dock.py` (prerequisite for any re-run)

| Step | Action | Note |
|---|---|---|
| P3.1 | Delete identity resolution; hard-fail without the plan file | |
| P3.2 | Job manifest + fragment in the signature + naming codec | Ship `backfill_job_manifests.py`: writes `job_manifest.json` into each of the 24,173 existing dirs *from the plan*, so existing compute becomes readable by the new `parse_prepare` **without re-docking and without touching a single pose file**. ⚠️ This is the only write into the irreplaceable output tree in this entire plan — additive only, never modifying or deleting, to be reviewed and snapshotted separately |
| P3.3 | Re-dock the **delta only** | 6,185 intended pairs never had their correct protein docked; 4,196 already have a CavitySpace cavity and are re-dockable today; 1,989 need new structure prep. Budget the compute from that queue, not from scratch |

### Phase 4 — parsing and analysis (can follow; runs on existing data once P3.2 lands)

Remove the 10,000 cap; manifest-based identity; `checked_merge` everywhere; fragment-aware cluster keys; tool-name enum; IC50 semantics; recorded (not hardcoded) processing steps. Then regenerate the parquet.

**Invalidates:** `combined_filtered_annotated_docking_results.parquet`, all backups, and every notebook figure derived from them. Expect the row count to *rise* (~2.4×, as the 14,173 dropped job dirs return) while the positive class *shrinks* — **no previously reported metric survives, and old and new numbers must not be reconciled.** Notebooks get a repointed `DATA_FILE` plus a top-of-notebook assertion that the parquet's provenance hash matches the expected index.

### Phase 5 — housekeeping (no artifact impact)

Config/TOML migration, path centralisation, provenance stamping, environment lock, README/CLAUDE.md updates, and `KNOWN_ISSUE_uniprot_gene_mapping.md` marked resolved with a pointer to the audit report.

### Artifact invalidation summary

| Artifact | Fate |
|---|---|
| `uniprot_gene_mapping.csv` | **deleted from the pipeline**, retained read-only as evidence; non-reproducible, never regenerate |
| `required_structures*.csv`, `negative_samples_metadata.json`, `docking_jobs_preview.csv`, `job_preparation_summary.csv` | regenerated (Phase 1/3) |
| `alphafold_mapping.csv`, `fixed_mapping.csv`, `pdbqt_mapping.csv` | re-indexed; underlying structure files reusable |
| `ic50_mapping.csv` | regenerated, strictly better coverage (45.4% → full proteome) |
| `pocket_louvain_clusters_*.csv` | re-derive after the syntax fix — current provenance unverifiable |
| `combined_*.parquet` + backups | regenerated; all downstream figures invalid |
| **`consensus_docking_results/` (~1.4 TB, 24,173 job dirs)** | **RETAINED IN FULL.** Zero physically invalid computations: every job ran a real ligand against a real cavity of a real protein, and the receptor path is keyed on the accession. The labels were wrong, the physics was not. The remedy is relabel + delta-dock, not re-run |
