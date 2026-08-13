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

## 0. Toolchain status — what can actually run today

Verified 2026-08-13 against the working tree:

```bash
python3 -m unittest discover -s /home/onur/repos/dock_selectivity/tests \
                             -t /home/onur/repos/dock_selectivity
# Ran 605 tests in 56.359s -- OK      (no failures, no skips)
```

The suite was still being added to while this section was written (604 → 605 across two runs an hour
apart), so treat the count as a floor and re-run rather than trusting the number. `OK` with **zero
skips** is the part that matters: a skipped check is not a passed check.

Tests are hermetic: `tempfile` fixtures only, no network, no external drive, no `/opt/data`. Among them
`tests/test_end_to_end.py` runs the **whole chain the way this runbook does** — Phase 0 → 1 → 2 → 3 → 5
plus the STOP gate — against a synthetic universe that is the incident in miniature: `AVPR2`/`P30518`
with the impostor `Q8NCT1` (ARRDC4) as a real old job directory, a TrEMBL twin sharing GeneID 10376 with
`TUBA1B`, an isoform and a non-human accession that must be excluded *with a reason*, and old directory
names covering every parse form the campaign actually contains. What it proves is the thing no unit
suite can: that the five hand-offs between stages agree, and that a job the corruption produced cannot
re-enter `$NEW` through any of them.

Everything below is **new code**. No existing pipeline script has been modified; the `FIX_PLAN.md` §3
edits to `batch_dock.py`, `parse_prepare.py`, `extract_*.py`, `identify_required_structures.py` and
`generate_negative_samples.py` have **not** been made.

### 0.1 Built and tested

| Component | Role | Used by phase |
|---|---|---|
| `dock_selectivity/identity.py` | `IdentityIndex` / `Protein` / `Resolution` — the one identifier authority | all |
| `dock_selectivity/invariants.py` | `assert_collapse_factor_one`, `assert_total_accounting`, `assert_no_row_growth`, `assert_deterministic_across_hash_seeds`, `assert_self_consistent_labels`, `assert_unique_key` | all |
| `dock_selectivity/joins.py` | `checked_merge()` — a merge may not change row count | all |
| `dock_selectivity/schemas.py` | the seven manifest schemas, `read_checked` / `write_checked` | all |
| `dock_selectivity/jobid.py` | `__`-delimited job-name codec, `job_identity.json` reader/writer, the right-anchored legacy parser | 2, 5, 7, 8 |
| `build_identity_index.py` | Phase 0 — builds `identity_index.parquet`; the repo's **sole** network consumer | 0 |
| `resolve_interactions.py` | Phase 1 — `interactions_resolved.csv`, resolved on Entrez, never on a symbol | 1 |
| `classify_existing.py` | Phase 2 — classifies all 24,173 old job dirs, read-only | 2 |
| `sample_negatives.py` | Phase 3 — balanced sampler in accession space. **Replaces `generate_negative_samples.py`**; the old script is untouched and is not used by this plan | 3 |
| `curate_jobs.py` | Phase 5 — materialises `$NEW/docking_results` under the new naming scheme | 5 |
| `audit.py` | every STOP gate: `identity`, `curated`, `run` | 0, 5, 8 |

### 0.2 Phase-by-phase readiness

| Phase | Runnable now? | What is missing |
|---|---|---|
| **0 — identity** | **Yes, in `REDUCED_LOCAL` mode** (verified end-to-end: 4,585 records, audit 12/12 PASS, exit 0). | The authoritative build needs a one-time UniProt fetch. `HUMAN_9606_idmapping.dat.gz`, `UP000005640_9606.fasta.gz` and `hgnc_complete_set.txt` are **absent** from `/opt/data` (checked). See §3 Phase 0. |
| **1 — resolve interactions** | **No.** | `resolve_interactions.py` is tested, but it resolves the protein axis *only* through Entrez, and a `REDUCED_LOCAL` index built from the legacy file alone carries **0 Entrez ids**. Measured: **0 of 8,568 edges resolve, all 8,568 reject `UNKNOWN_ENTREZ`, and the script still exits 0 with an empty manifest.** Needs the authoritative files, or at minimum `hgnc_complete_set.txt` passed as `--hgnc` to the reduced build. |
| **1b — `positives_planned.csv`** | **No — no producer exists.** | Nothing in the repo writes this manifest, and Phases 3 and 5 both require it. Either the `FIX_PLAN.md` §3.1 rewrite of `identify_required_structures.py`, or a new `plan_positives.py`. See §3 Phase 1b. |
| **2 — reuse scan** | Script ready; blocked behind Phase 1. | A non-empty `interactions_resolved.csv`. With an empty one every job classifies as `valid_non_interaction` and the "pool" silently becomes 24,173. |
| **3 — balanced sampler** | Script ready; blocked behind Phase 1b. | `positives_planned.csv` — quotas are undefined without it. |
| **4 — decision** | Yes (it is a decision, not code). | — |
| **5 — curate** | Script ready; blocked behind 2 and 3. | Also **5b**: nothing assembles the full `run_manifest.csv`. `curate_jobs.py --manifest-out` emits only the jobs it materialised. |
| **6 — structure prep** | **No.** | `FIX_PLAN.md` §3.5/§3.6 not done. `extract_cavities.py`, `extract_alphafold_models.py`, `fix_required_pdbs.py` and `convert_pdb_to_pdbqt.py` have **zero `add_argument` calls between them** — they take no arguments at all, so none of the `--required` flags in §3 Phase 6 exist yet. |
| **7 — dock the gap** | **No.** | `FIX_PLAN.md` §3.3 not done. `batch_dock.py` has no CLI and still resolves identity itself at `:2034`. |
| **8 — parse and analyse** | **No.** | `FIX_PLAN.md` §3.4 not done. `parse_prepare.py` has no CLI, still caps discovery at 10,000 files (`:146`) and still merges on a non-unique key (`:1171`). |

**Phases 6–8 are blocked on edits to the old pipeline scripts and nothing else.** The new toolchain
carries the rebuild as far as a curated `$NEW/docking_results` (Phase 5); from there the old scripts
have to be fixed before a single new job can be docked or parsed.

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
| Labelled negative, not a known interaction | 11,801 | **valid non-interaction** |
| Labelled negative, actually a known interaction | 5 | **true positive** |

**Positives: 7,300 jobs / 2,091 genuine pairs. Valid non-interactions: 16,873 jobs.**

*(Corrected 2026-08-13. This table previously read 11,796 / 16,868 — an arithmetic slip that propagated
into §1.2, §2 and the Phase 2 STOP gate. `ASSESSMENT_blast_radius_2026_08_12.md` §1a and §2 already
carried the right figures. Independently re-derived: 12,367 labelled-positive jobs + 11,806
labelled-negative jobs = 24,173 exactly; 6,987 + 308 + 5,072 + 11,801 + 5 = 24,173. The pair count is
2,086 from the label-positive jobs alone, 2,091 including the 5 contaminated negatives.)*

### 1.2 The 5,072 mislabelled positives, and what the balanced design can actually use

Under correct identity the 5,072 are the same kind of measurement as the 11,801 sampled negatives: a
drug docked into a protein it has no annotated interaction with. What distinguishes them is only that
**nobody chose them**.

The earlier instinct — "pool all 16,873 and let the sampler draw, which reduces how much new docking is
needed" — is **wrong**, because the design is *balanced* negative sampling (Najm et al.): each drug and
each target must appear in equal numbers of positives and negatives. A target with zero positives in the
corrected set must receive **zero** negatives. That makes a pooled job on such a target **structurally
ineligible**, not merely "maybe not drawn".

**Measured against the corrected positive set (5,287 dockable pairs / 1,101 drugs / 1,283 targets →
18,166 planned positive jobs):**

| | Sampled negatives | Mislabelled positives | Pool |
|---|---:|---:|---:|
| Jobs | 11,801 | 5,072 | 16,873 |
| Structurally eligible | 7,325 (62.1%) | **738 (14.6%)** | 8,063 (47.8%) |
| Ineligible — target has zero positives | 4,476 | 4,334 | 8,810 |
| **Actually drawable under the joint quota (max-flow optimum)** | 6,347 | — | **6,430** |
| **Marginal contribution of the 5,072** | | | **+83 jobs (1.6% of them)** |

The drug axis never binds — all 1,046 docked drugs carry ≥1 corrected positive. **All** ineligibility is
the target axis: 523 of 979 docked accessions have zero positives, because the corruption selected
receptors that nobody targets.

So:

> **Build `negatives_pool.csv` over all 16,873 valid non-interactions, but carry an explicit
> `eligibility` column (`eligible` / `target_has_no_positives` / `drug_has_no_positives` / `both`)
> computed against `positives_planned.csv`. Expect the sampler to draw ~6,430 jobs — 38% of the pool,
> not 100%. The remaining 10,443 are surplus and are not copied.** The 5,072 mislabelled-positive jobs
> are worth **+83 negatives**; do not budget them as a re-dock saving.

**Negative-side re-dock requirement: 11,736 jobs over ~3,887 new (drug, accession) pairs**
(job-balanced), or **3,087 pairs** (pair-balanced). This is a hard requirement of the design, not an
optimisation.

**Balance unit.** `generate_negative_samples.py` balanced **cavity rows (jobs)**, not pairs (`:201-220`,
`:504-505`, `:581-595`). The shipped result is perfect on jobs (1,046/1,046 drugs, 979/979 targets within
±1) and skewed on pairs (338/1,046 and 478/979): 3,982 positive pairs at 3.38 cavities each vs 9,856
negative pairs at 1.36. **The rebuild should balance pairs and dock each negative pair at the same
cavity depth as the positives**, so the two classes are comparable per pair. Both accountings are given
in §1.2.4; Phase 3's STOP gate must state which unit it asserts.

**Why the old negatives cannot be reused as-is** is unchanged and now quantified: they were drawn to
balance a positive set that was 45.8% wrong (in true protein space only 30.8% of drugs and 39.1% of
targets are balanced), and only 62.1% of them are even eligible against the corrected positives. The
sampler **must** be re-run.

The four subsections below are the measurement behind those numbers. Identity throughout is resolved
only through the `Gene Names` column of `uniprot_gene_mapping.csv` — the column `KNOWN_ISSUE` and
`ASSESSMENT` both certify as ground truth. `Mapped_Gene_Name` is never read.

#### 1.2.1 The corrected positive set

`1_drug_to_protein.tsv`: 8,568 edges in → **8,285 resolvable**, 283 rejected and recorded with a reason
(237 `node_2_name` null, 46 symbols absent from every accession's `Gene Names`). Symbol→accession
fan-out is genuinely one-to-many (mean 1.43, max 8), so resolution is a relation, never a dict.

| | pairs | drugs | target accessions | pos/drug (mean, med, max) | pos/target (mean, med, max) |
|---|---:|---:|---:|---|---|
| **`POS_ALL`** — every resolvable edge | 9,623 | 1,640 | 2,355 | 5.87 · 3 · 339 | 4.09 · 1 · 108 |
| **`POS_DOCKABLE`** — + small molecule, SDF on disk, ≥1 CavitySpace cavity | **5,287** | **1,101** | **1,283** | 4.80 · 3 · 240 | 4.12 · 1 · 98 |
| **`positives_planned`** — `POS_DOCKABLE` × every cavity of the target | **18,166 jobs** | 1,101 | 1,283 | — (3.44 cavities/pair) | — |
| …already docked correctly | 7,300 jobs / 2,091 pairs | 767 | 409 | — | — |

`POS_DOCKABLE` is the operative denominator: it is what `manifests/positives_planned.csv` will contain,
and a drug whose only correct targets have no cavity can never acquire a positive, hence can never
acquire a negative.

All 24,173 directory names parse right-anchored, **0 unresolved** — 1,743 legacy `AF-*-F*-model_v1`
forms, 34 hyphenated gene tokens.

#### 1.2.2 Eligibility of the 16,873-job pool

A target with zero positives must receive zero negatives; likewise a drug. Eligibility is therefore
structural, not probabilistic.

**Using `POS_ALL` (most permissive):**

| Status | Sampled negatives (11,801) | Mislabelled positives (5,072) | **Pool (16,873)** |
|---|---:|---:|---:|
| **Eligible** | **7,325 (62.1%)** | **738 (14.6%)** | **8,063 (47.8%)** |
| Ineligible — target has zero positives | 4,476 | 4,334 | 8,810 |
| Ineligible — drug has zero positives | 0 | 0 | 0 |
| Ineligible — both zero | 0 | 0 | 0 |

**Using `POS_DOCKABLE` (operative):**

| Status | Sampled negatives | Mislabelled positives | **Pool** |
|---|---:|---:|---:|
| **Eligible** | **7,190 (60.9%)** | **668 (13.2%)** | **7,858 (46.6%)** |
| Target has zero positives | 4,448 | 4,236 | 8,684 |
| Drug has zero positives | 106 | 41 | 147 |
| Both zero | 57 | 127 | 184 |

All 1,046 docked drugs carry ≥1 corrected positive under `POS_ALL`; under `POS_DOCKABLE` only 49 fall
out. The entire loss is the target axis: **523 of 979 docked accessions have zero positives**, and 8,810
pool jobs (52.2%) sit on them. This is the direct consequence of the corruption having *selected the
receptor* — the wrong-protein receptors are, definitionally, proteins nobody targets.

Eligible pool: **8,063 jobs / 6,249 distinct (drug, accession) pairs / 962 drugs / 456 targets.**
Ineligible remainder: 8,810 jobs spanning 972 drugs and 523 accessions.

**How much of the accidental output survives.** The 5,072 mislabelled-positive jobs contribute **738
eligible jobs (14.6%)** — 85.4% are structurally unusable. Their *marginal* value on top of the sampled
negatives is near zero:

| Pool offered to the sampler | Drawable, balancing 7,300 docked positives | Drawable, balancing 18,166 planned positives |
|---|---:|---:|
| Sampled negatives alone (11,801) | 4,767 | 6,347 |
| Mislabelled positives alone (5,072) | 66 | 325 |
| Both (16,873) | **4,770** | **6,430** |
| **Marginal gain from the 5,072** | **+3 jobs** | **+83 jobs** |

Their 738 eligible jobs cover only 71 targets and **zero targets the clean negatives do not already
cover**, at far lower depth (738 jobs / 193 pairs vs 7,325 jobs / 6,056 pairs). The clean negatives
already saturate those quotas.

#### 1.2.3 Feasibility of the balanced constraint against the eligible pool

Balanced sampling is a degree-constrained bipartite subgraph problem: `src → drug` capped at that drug's
positive count, `drug → target` capped at the pool jobs available for that pair, `target → sink` capped
at that target's positive count. Max-flow gives the exact optimum, verified independently by a scipy
`linprog` re-derivation (identical) and against decoupled per-axis upper bounds. The repo's greedy
reaches within 5–7 jobs of the optimum, so **greedy is not the limitation — the pool is.**

| | Scenario A — balance the 7,300 already-docked positives | Scenario B — balance the full corrected design (18,166) |
|---|---:|---:|
| Negative jobs demanded | 7,300 | 18,166 |
| Pool jobs offered | 16,873 | 16,873 |
| Structurally eligible | 8,063 | 8,063 |
| **Max drawable (optimum)** | **4,770 (65.3%)** | **6,430 (35.4%)** |
| Eligible but over-quota (surplus) | 3,293 | 1,633 |
| **Shortfall — the re-dock requirement** | **2,530 jobs / ~951 new pairs** | **11,736 jobs / ~3,887 new pairs** |
| Drugs not fully balanced | 488 / 767 | 985 / 1,101 |
| Targets not fully balanced | 219 / 409 | 987 / 1,283 |

A and B are not alternatives to choose between. **A** sizes what the pool can do with zero new *positive*
docking; **B** is the actual Phase 3/4 target. **Quote B.**

Worst-20 deficits below are reported as *structural* deficits (`required − pool jobs available on
quota-carrying counterparts`) because that quantity is unique and seed-independent; the max-flow
attribution of a deficit to a specific entity is not (verified: one drug shifts by ±1 across
`PYTHONHASHSEED` values, totals never move).

**Scenario B — worst 20 targets** (`req | max from pool | deficit`). Sixteen of twenty have **zero** pool jobs:

`P35367 HRH1 686|0|686` · `P25100 ADRA1D 552|0|552` · `P35354 PTGS2 300|0|300` · `P08172 CHRM2 280|0|280` · `P14416 DRD2 237|0|237` · `P18825 ADRA2C 232|0|232` · `P21817 RYR1 195|0|195` · `P35348 ADRA1A 609|417|192` · `P34903 GABRA3 188|0|188` · `P34969 HTR7 185|0|185` · `P23219 PTGS1 184|0|184` · `P28221 HTR1D 144|0|144` · `Q09428 ABCC8 144|0|144` · `P08519 LPA 138|0|138` · `P35368 ADRA1B 134|0|134` · `P18089 ADRA2B 124|0|124` · `P78334 GABRE 120|0|120` · `Q8N1C3 GABRG1 120|0|120` · `O75469 NR1I2 117|0|117` · `P03372 ESR1 114|6|108`

**Scenario B — worst 20 drugs:** `DB12010 Fostamatinib 848|231|617` · `DB00201 Caffeine 388|58|330` · `DB00157 NADH 178|30|148` · `DB00513 Aminocaproic acid 143|4|139` · `DB09085 Tetracaine 108|1|107` · `DB00543 Amoxapine 117|14|103` · `DB01151 Desipramine 82|0|82` · `DB13345 Dihydroergocristine 93|13|80` · `DB00334 Olanzapine 123|48|75` · `DB06637 Dalfampridine 135|60|75` · `DB00142 Glutamic acid 123|51|72` · `DB05294 Vandetanib 88|18|70` · `DB00114 Pyridoxal phosphate 85|19|66` · `DB00321 Amitriptyline 123|60|63` · `DB01219 Dantrolene 65|2|63` · `DB00726 Trimipramine 92|31|61` · `DB00408 Loxapine 109|49|60` · `DB00246 Ziprasidone 97|38|59` · `DB00370 Mirtazapine 84|25|59` · `DB00363 Clozapine 103|45|58`

**Scenario A — worst 20 targets:** `P35348 ADRA1A 609|363|246` · `P28335 HTR2C 406|280|126` · `P20309 CHRM3 288|206|82` · `P11229 CHRM1 255|178|77` · `P10275 AR 296|226|70` · `P08913 ADRA2A 280|218|62` · `P28223 HTR2A 234|173|61` · `P21728 DRD1 265|216|49` · `P31645 SLC6A4 200|151|49` · `P08588 ADRB1 177|131|46` · `P08908 HTR1A 177|137|40` · `P07550 ADRB2 124|86|38` · `P41595 HTR2B 185|147|38` · `P04150 NR3C1 205|170|35` · `Q01959 SLC6A3 144|111|33` · `P23975 SLC6A2 165|133|32` · `P08912 CHRM5 144|113|31` · `P08173 CHRM4 171|141|30` · `P06401 PGR 132|110|22` · `P13945 ADRB3 108|87|21`

**Scenario A — worst 20 drugs:** `DB01151 Desipramine 49|0|49` · `DB00543 Amoxapine 61|14|47` · `DB00157 NADH 67|27|40` · `DB00726 Trimipramine 62|27|35` · `DB13345 Dihydroergocristine 47|13|34` · `DB00248 Cabergoline 41|17|24` · `DB00370 Mirtazapine 47|25|22` · `DB00502 Haloperidol 22|0|22` · `DB00934 Maprotiline 37|16|21` · `DB01186 Pergolide 36|16|20` · `DB00142 Glutamic acid 65|46|19` · `DB00780 Phenelzine 19|0|19` · `DB01624 Zuclopenthixol 19|0|19` · `DB00454 Meperidine 31|13|18` · `DB00568 Cinnarizine 18|0|18` · `DB00715 Paroxetine 23|5|18` · `DB01403 Methotrimeprazine 39|21|18` · `DB00413 Pramipexole 29|12|17` · `DB00656 Trazodone 33|16|17` · `DB01149 Nefazodone 35|18|17`

#### 1.2.4 Balance unit: cavity rows or pairs? The old code balanced **jobs**

`generate_negative_samples.py` creates **one positive row per cavity** (`:201-220`, "Create one positive
sample for each cavity"), counts quotas with `positive_df.groupby('drugbank_id').size()` /
`.groupby('UniProt_ID').size()` over those rows (`:504-505`), and emits **exactly one row per greedy
draw** with `np.random.choice(protein_cavities[target_id])` picking a single random cavity
(`:581-595`). The balanced unit is the **cavity row = the docking job**, not the pair.

Consequence in the shipped artifact — perfect on jobs, badly skewed on pairs:

| | Positives | Negatives |
|---|---:|---:|
| Rows (jobs) | 13,452 | 13,452 |
| Distinct (drug, accession) pairs | 3,982 | **9,856** |
| Cavity rows per pair | **3.38** | **1.36** |
| Drugs balanced within ±1 **on rows** | 1,046 / 1,046 | |
| Targets balanced within ±1 **on rows** | 979 / 979 | |
| Drugs balanced within ±1 **on pairs** | **338 / 1,046** | |
| Targets balanced within ±1 **on pairs** | **478 / 979** | |

The negative class spans **2.48× more distinct drug–protein combinations at 2.5× less cavity depth**
than the positive class. That is a real, unintended design asymmetry: cavity multiplicity is a
per-protein property, so job-balance systematically buys negative diversity at the cost of negative
depth, and any per-pair statistic is computed from 3.38 poses on positives and 1.36 on negatives.

Both feasibility numbers for the corrected design:

| Balance unit | Demand | Drawable from pool | Shortfall |
|---|---:|---:|---:|
| **Jobs** (what the old code did) | 18,166 jobs | 6,430 | **11,736 jobs** |
| **Pairs** | 5,287 pairs | 2,200 | **3,087 pairs** ≈ 10,607 jobs at full cavity depth, or 3,087 jobs at one cavity each |
| Jobs, Scenario A | 7,300 | 4,770 | 2,530 |
| Pairs, Scenario A | 2,091 | 1,804 | 287 |

**Recommendation: balance pairs** (`sample_negatives.py --balance-unit pair`), then dock each drawn
negative pair at the same cavity depth as the positives. Job-balance is what produced the 3.38-vs-1.36
asymmetry; pair-balance is both the cheaper constraint (287 pairs short in Scenario A vs 2,530 jobs) and
the one that matches the scientific claim, which is about drug–target pairs, not about pockets.

#### 1.2.5 Method, assumptions, limitations

- **Resolution rule.** An edge (drug, symbol) resolves to **every** accession whose own `Gene Names`
  list contains that symbol. A job (drug, accession) is a true positive iff any symbol of that accession
  is a target of that drug — `ASSESSMENT` §2 Step 4's rule, applied **without** the Entrez route,
  because no id-mapping file exists locally yet. The 237 null-symbol edges and 46 unmapped symbols stay
  unresolved and are recorded with a reason, never dropped.
- **Every figure here is a floor, not a ceiling.** Once the authoritative index exists, the Entrez route
  recovers the 237 null-symbol edges. More resolvable edges → more positives → **more** required
  negatives, so the shortfall can only grow.
- **Job labelling.** 23,508 of 24,173 jobs matched `required_structures_with_negatives.csv` on
  (drug, accession, fragment, cavity); the remaining 665 needed a 3-key fallback because the plan writes
  `Fragment_ID` as `'F1'` on positives and a bare integer on negatives (the normalisation trap
  `ASSESSMENT` §2 Step 6 flags). All 665 resolved to `negative`; 0 unresolved.
- **Feasibility uses the max-flow optimum, not the repo's greedy**, because the question is "is it
  satisfiable", not "does this heuristic find it". Confirmed by an independent `scipy.optimize.linprog`
  re-derivation (the transportation polytope is integral, so LP optimum = integer optimum): 4,770.0 and
  6,430.0 exactly, both under the decoupled per-axis upper bounds. The repo's greedy reaches
  4,763 / 6,425 — within 0.15%.
- **Per-entity deficit attribution is not unique.** Totals are exact and reproducible; which drug absorbs
  a given unit of deficit is one of several optima (verified across `PYTHONHASHSEED` 0/1/12345: totals
  byte-identical, one drug shifted by 1). The worst-20 tables therefore report the *structural* deficit,
  which is unique and a lower bound. Aggregates (8,063 / 6,430 / 11,736 / 2,530) are exact.
- **~3,887 new pairs is a planning estimate, ±10%.** The 11,736 *jobs* figure is exact; how many pairs
  that becomes depends on how the sampler spreads a drug's deficit across targets.
- **The analysis deliberately omits `generate_negative_samples.py:561-566`** — the branch that, when no
  target still needs negatives, falls back to "any available target". That branch silently abandons the
  target-side quota (the universe-widening pattern `FIX_PLAN` G8 bans) and including it would have
  inflated apparent feasibility. `sample_negatives.py` does not have it.
- **"Non-interaction" still means "not annotated in `1_drug_to_protein.tsv`".** Pool jobs are
  *unlabelled*, not proven non-binders. Unchanged from `ASSESSMENT`.
- **Unreconciled.** The claim that "only 388 accessions in the docked pool genuinely carry an intended
  target gene" does not reproduce under any tried definition: 456 carry any symbol appearing in the TSV,
  451 are targets of a corrected positive whose drug has an SDF, 447 have a target symbol as their
  *primary* symbol, 442 both. Nothing above depends on it; **456** is used, with that definition. Where
  388 came from is worth re-checking.

### 1.3 The 2.4% rule — what a carried-over job consists of

A job directory is ~47 MB. Measured composition of a representative job:

| Component | Size | Carry over? |
|---|---:|---|
| `final_results.csv` (consensus Tool1/Tool2/Score1/Score2/RMSD) | 184 KB | **yes** — the payload |
| `{smina,gnina,gold,ledock}/results.csv` | ~100 KB | **yes** |
| `smina/out.sdf`, `gnina/out.sdf` | ~440 KB | **yes** — sole carrier of gnina `CNN_VS`, `CNNaffinity_variance`, and unrounded scores |
| `gold/output/gold_soln_*.sdf` + rankings | 363 KB | **yes** — this is `--gold-output poses`, the default |
| `ledock/out.dok` | ~37 KB | **yes** |
| `input/{DB*.mol2, receptor.mol2}` | 198 KB | **yes** — cheap, makes each job self-describing |
| `{tool}/complex_*.pdb` (~20 per tool) | **~45 MB** | **no** — derived (receptor + pose), 99.75% byte-identical between poses, ligand coords reproduce `out.sdf` exactly |
| `log_*.log` (~13 retry logs per job) | 60 KB | **no** — provenance goes in `job_identity.json` |
| rest of `gold/output/` | 2.73 MB | **no** — see below |

**Evidence per job ≈ 1.14–1.21 MB, i.e. 2.4%.** Carrying **13,730 jobs costs ≈ 15.3 GB** (7,300
positives + 6,430 drawn negatives; §1.2) instead of ~800 GB. The `complex_*.pdb` files are not lost —
they remain in the old directory, and are reconstructible from `receptor.mol2` + `out.sdf` if needed.

*Measured correction (2026-08-13, read-only, six real job directories).* This table originally listed
`gold/output/` at ~180 KB. The **whole tree** is **3.09 MB**: it also holds two further copies of the
receptor (`gold_protein.mol2` + `plp_protein.mol2`, 1.47 MB, already carried as `input/receptor.mol2`),
the cavity fitting points, 424 KB of GOLD logs, and a byte-duplicate pose set (`ranked_*.sdf`). Taking
only the poses is 363 KB and lands total evidence at ~1.21 MB/job, which is the ~1.14 MB / 2.4% the plan
assumed. `curate_jobs.py --gold-output all` takes the literal tree at ~4.52 MB/job (9.1%) — **62 GB**
over 13,730 jobs instead of ~19 GB. The policy used is recorded per job in `job_identity.json`
(`evidence_gold_output`), so a tree is always self-describing, and changing the policy correctly
invalidates a previous copy and re-copies.

### 1.4 Folder naming — make the old failure structurally impossible

Old: `DB00035_AVPR2_Q8NCT1_cavity_2` — a single-underscore string in which the gene and the accession
came from opposite sides of the bug and could disagree; it had no fragment field (causing the 410,087-row
merge fan-out) and its parser broke on 1,777 names.

New: **`DB00035__P30518__AVPR2__F1__cav2`**

- `__` double-underscore delimiter — hyphenated symbols (`MT-ND1`) and odd characters can no longer break the parse.
- **Accession before gene**, and the gene is written by `IdentityIndex.label(accession)` at creation time, so it cannot disagree by construction.
- **Fragment is explicit** (`F1`), closing the multi-fragment gap. A null gene is written as `-`, **never** as the string `nan` (the old blank-gene path produced 10,698 `*_nan_*` directories).
- The name is a convenience. The authority is `job_identity.json` inside every directory:

```json
{"schema_version": 1,
 "drugbank_id": "DB00035", "accession": "P30518", "gene": "AVPR2",
 "fragment_id": "F1", "cavity_index": 2,
 "dir_name": "DB00035__P30518__AVPR2__F1__cav2",
 "sample_type": "positive",
 "provenance": "reused_from", "source_job": "DB00035_AVPR2_Q8NCT1_cavity_2",
 "selected_by": "…", "fragment_source": "manifest",
 "retention": "evidence_only", "evidence_gold_output": "poses",
 "identity_index_sha256": "…", "code_revision": "…", "curated_by": "…",
 "evidence": [{"path": "final_results.csv", "bytes": 188416}],
 "missing_evidence_rules": []}
```

The first six keys plus `dir_name` are the core record written by `jobid.write_job_identity`; everything
after is free-form provenance, sorted, and may not shadow a core field. The write is atomic
(temp file + `os.replace`), so a killed docking run cannot leave a half-written identity.

`provenance` and `source_job` make every carried-over job traceable to the directory it came from, so
the relabelling is auditable rather than a leap of faith. **No timestamp is written**, deliberately: two
runs from the same inputs must produce a byte-identical tree. `retention` and `evidence_gold_output`
record what was actually copied, so a tree is self-describing and changing the policy correctly
invalidates a previous copy. `fragment_source` says whether the fragment came from a manifest or from an
explicit `--assume-fragment` choice.

Real examples from the round-trip tests, showing what the scheme repairs:

```
DB00035_nan_P30518_cavity_1     ->  DB00035__P30518__AVPR2__F1__cav1     (blank gene recovered)
DB00035_AVPR2_Q8NCT1_cavity_2   ->  DB00035__Q8NCT1__ARRDC4__F1__cav2    (the label was the lie)
DB00157_MT-ND1_P03886_cavity_1  ->  DB00157__P03886__MT-ND1__F1__cav1    (hyphenated symbol)
DB00157_nan_P68363_cavity_3     ->  DB00157__P68363__TUBA1B__F2__cav3    (fragment made explicit)
```

---

## 2. Directory layout — inputs separate from outputs

Inputs are small (~72 MB), shared across runs and versioned. Outputs are per-run
and large. Keeping them in one directory is how the 2025-06-29 tree ended up
holding 1.55 TB of mixed provenance with no way to tell what was source and what
was product.

```
/media/onur/Elements/cavity_space_consensus_docking/
├── inputs/                          # 72 MB, shared, read-mostly
│   ├── 1_drug_to_protein.tsv
│   ├── small_molecule_drug_links.csv
│   ├── ligands/                     # 2,599 ligand SDFs
│   └── uniprot/                     # idmapping + reference proteome
└── runs/<YYYY_MM_DD>/               # outputs only, one per run
    ├── identity/  manifests/  structures/  docking_results/  analysis/  logs/
```

Shared reference stores stay in `/opt/data` because they are too large to copy
per run: `cavity_space/af_strong_cavities_data_all` (7.0 GB) and
`alphafold2/UP000005640_9606_HUMAN_v4.tar` (4.8 GB).

`run_full_workflow.py --inputs-dir` resolves every input from that layout; an
explicit `--edges` / `--drugs` / `--ligands` / `--idmapping` / `--proteome` always
overrides. A path that resolves to nothing is reported, never silently defaulted.

### Old run directory layout (per-run internals)

```
$NEW/
  identity/
    identity_index.parquet          # THE root artifact — build first, nothing runs without it
    identity_index.meta.json        # source URLs, UniProt release, SHA256s, accounting, invariant report
    identity_index.notes.csv        # decisions about rows that ARE in the index (e.g. entrez demotions)
    identity_index.invariants.json  # every build-time check, also written when a build is REFUSED
    unresolved_targets.csv          # accessions NOT in the index, with a reason. Never silent.
  manifests/
    interactions_resolved.csv       # (drugbank_id, entrez_id, accession, gene, selected_by)
    interactions_resolved.rejects.csv   # every edge that did NOT resolve, with a reason and an axis
    positives_planned.csv           # (drug, accession, fragment, cavity) positives wanted — NO PRODUCER YET
    negatives_pool.csv              # the reuseable non-interaction jobs (16,873 minus the incomplete)
    negatives_pool_eligibility.csv  # every pool row + eligible(bool) + eligibility(reason)   §1.2
    negatives_drawn.csv             # what the balanced sampler actually selected
    job_classification.csv                 # old_job_dir -> new_job_dir, or a reason for exclusion (11 cols, strict)
    job_classification.evidence.csv        # per-job evidence the job_classification schema does not declare
    redock_queue.negatives.csv      # negative half — written by sample_negatives.py
    redock_queue.positives.csv      # positive half — NO PRODUCER YET
    redock_queue.csv                # the two halves concatenated
    run_manifest.reused.csv       # written by curate_jobs.py --manifest-out (reused jobs only)
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
  logs/                             # classify_existing_summary.json, sample_negatives_report.json,
                                    # negatives_deficits.csv, curate_jobs_report.json, run logs
```

Two rules the layout enforces: **`manifests/run_manifest.csv` is the authority on what should exist**
(so a truncated discovery pass is detectable rather than silent), and **`identity/` is built once and
pinned by SHA256 everywhere** (so no stage can invent its own identity).

Three naming details that matter, because two artifacts would otherwise collide:

- `build_identity_index.py` writes `identity/unresolved_targets.csv` (**accessions** not in the index).
  `resolve_interactions.py --rejects` also defaults to a file called rejects and its help suggests
  pointing it at `identity/unresolved_targets.csv` — **do not**, it would overwrite the index's ledger.
  Keep the edge rejects in `manifests/interactions_resolved.rejects.csv`.
- `sample_negatives.py --redock-queue` writes the **negative half only**. The positive half
  (10,866 jobs, §1.2.3) has no producer yet. Concatenate deliberately; do not let one half claim the
  `redock_queue.csv` name.
- `curate_jobs.py --manifest-out` writes only the jobs it materialised (`provenance='reused_from'`).
  The full `run_manifest.csv` is that file concatenated with the `newly_docked` rows derived from
  `redock_queue.csv`, and `audit.py run` asserts the result equals
  `positives_planned ∪ negatives_drawn` exactly.

---

## 2b. Validation status — run against the real data on 2026-08-13

Phases 0–2 and 6's manifest step were executed against the real inputs, read-only,
with all output to scratch. Results and one correction:

| Stage | Real-data result |
|---|---|
| `build_identity_index.py` (authoritative) | 210,709 accessions, 19,323 Entrez ids, **max_collapse = 1** |
| `resolve_interactions.py` | 8,568 edges → **7,923 resolved**, 645 rejected with reasons; **229 resolved that carry a null `node_2_name`** (unreachable by the old symbol resolver) and 143 from `node_1` name slugs |
| Identity spot checks | AVPR2→P30518, TUBA1B→P68363, IL2→P60568, BRAF→P15056, BCL2→P10415, ALK→Q9UM73 — every documented corruption case correct |
| `extract_cavities.py --manifest-only` | 38,709 rows / 13,600 accessions in **0.4 s, nothing unpacked**; key set **identical** to the `cavity_mapping.csv` the old code produced by unpacking 76 GB |
| `classify_existing.py` | 24,173 dirs in 5.7 s: **7,279 true_positive, 16,600 valid_non_interaction, 294 incomplete, 0 unparseable** |

The reuse split cross-validates the forensic estimate exactly: 7,300 + 16,873 =
24,173 (forensics) versus 7,279 + 16,600 + **294 incomplete** = 24,173 here — the
difference is precisely the incomplete jobs the forensic pass folded into the two
classes. `unparseable = 0` confirms the right-anchored parse handles every real name.

### CORRECTION — `REDUCED_LOCAL` cannot drive Phase 1

An earlier draft of this plan said the reduced index built from the legacy
`uniprot_gene_mapping.csv` was "the way to start today". **It is not.** That file
has no Entrez column, so the index carries **0 Entrez ids**, and since the
corrected design resolves only on Entrez, `resolve_interactions.py` returns
**8,568 in, 0 resolved, 8,568 rejected as UNKNOWN_ENTREZ**. The reduced index is
useful for `accession → label` lookups (the reuse path) and nothing else.

Phase 0 therefore **requires** the authoritative UniProt inputs. Both are public
and were fetched successfully:

```
https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz   # 36 MB
https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz  # 7.7 MB
```

Release 2026-06-10. Store them somewhere durable and pass `--idmapping` /
`--proteome`; the builder refuses to run without them rather than degrading.

---

## 2c. Toolchain status — ALL PHASES EXECUTABLE (2026-08-13)

Every phase of the runbook is now backed by implemented, tested code. The suite is
**743 tests, green under two `PYTHONHASHSEED` values**.

| Phase | Stage | Status |
|---|---|:--:|
| 0 | `build_identity_index.py` | ✅ validated on real UniProt inputs |
| 1 | `resolve_interactions.py` | ✅ 7,923/8,568 real edges resolved |
| 2 | `extract_cavities.py --manifest-only` | ✅ 38,709 rows in 0.4 s, key-identical to the 76 GB unpack |
| 2b | `plan_positives.py` | ✅ |
| 3 | `classify_existing.py` | ✅ all 24,173 real dirs, 0 unparseable |
| 4 | `sample_negatives.py` | ✅ |
| 5 | `curate_jobs.py` | ✅ |
| 6 | `build_run_manifest.py` | ✅ |
| 7 | `batch_dock.py --queue` | ✅ legacy resolver refuses |
| 8 | `parse_prepare.py` | ✅ all six defects fixed |
| — | `run_full_workflow.py` | ✅ drives 0–8, explicit `--mode`, `--dry-run` |
| — | `audit.py` | ✅ STOP-gate runner |

`prepare_uniprot_mapping.py` is **deleted**. An AST-based G6 lint fails the suite if
a free-text identifier search, a `bioservices` import, or a read of
`Mapped_Gene_Name` as a data key reappears anywhere in production code — and it
asserts every production file parses, which is how the long-standing
`pocketmatch_clustering.py:426` stray `tmux` token was finally caught and fixed.

**Still required before Phase 0 can run:** the two authoritative UniProt files
(§2b). They currently live only in scratch and should be moved somewhere durable.

---

## 3. Runbook

Every command below is the **real CLI of the script as it exists today** unless the phase is explicitly
marked *NOT YET IMPLEMENTED*. `python3` is the conda env `teachopencadd` (3.9).

```bash
REPO=/home/onur/repos/dock_selectivity
OLD=/media/onur/Elements/cavity_space_consensus_docking/2025_06_29_batch_dock     # READ-ONLY, always
NEW=/media/onur/Elements/cavity_space_consensus_docking/2026_08_13_clean_rebuild
IDX=$NEW/identity/identity_index.parquet
```

Every phase has a **STOP** gate — do not proceed until its checks pass. `audit.py` exits **0** = every
check ran and passed, **1** = a FAIL, **2** = nothing failed but a check could not run. *A check that did
not run is not a check that passed*: treat 2 as "not cleared". That distinction is the direct lesson of
`validate_negatives()`, which reported "All validations passed" while re-deriving its ground truth from
the object under test.

---

### Phase 0 — Identity. Nothing else may run first.

**Code prerequisite:** none — `dock_selectivity/identity.py` and `build_identity_index.py` exist and are
tested. `FIX_PLAN.md` §2 (delete `prepare_uniprot_mapping.py`) is still outstanding but does not block
this phase.

```bash
mkdir -p $NEW/{identity,manifests,ligands,docking_results,analysis,logs} \
         $NEW/structures/{cavities,alphafold,fixed,pdbqt}
```

#### Route A — authoritative (**required for Phase 1**; needs one manual network fetch)

`ls /opt/data` confirms that **none** of the three authoritative inputs is present locally:
`HUMAN_9606_idmapping.dat.gz`, `UP000005640_9606.fasta.gz`, `hgnc_complete_set.txt`. (What *is* there is
`/opt/data/alphafold2/UP000005640_9606_HUMAN_v4.tar` — AlphaFold structures, not the proteome FASTA.)

```bash
# The ONLY network access in this repository. Downloads to --cache-dir, checksums, then offline forever.
python3 $REPO/build_identity_index.py \
    --fetch \
    --out        $IDX \
    --unresolved $NEW/identity/unresolved_targets.csv \
    --cache-dir  $NEW/identity/cache \
    --check-determinism
```

`--meta`, `--notes` and the invariants JSON default beside `--out`. `--check-determinism` rebuilds under
`PYTHONHASHSEED` 0/1/42 and requires identical bytes (it roughly triples build time; the id-mapping file
is ~600 MB uncompressed). If the files are downloaded by hand instead, pass them explicitly — supplied
paths are never fetched over:

```bash
python3 $REPO/build_identity_index.py \
    --idmapping /path/HUMAN_9606_idmapping.dat.gz \
    --proteome  /path/UP000005640_9606.fasta.gz \
    --hgnc      /path/hgnc_complete_set.txt \
    --out $IDX --unresolved $NEW/identity/unresolved_targets.csv
```

Notes on the flags that matter: `--on-entrez-collision resolve` (default) applies the documented total
tie-break and records every demotion in `identity_index.notes.csv`; `fail` refuses to build instead and
will refuse every real build, because UniProt genuinely contains TrEMBL twins of Swiss-Prot entries.
`--strict-cross-check` makes the proteome `GN=` comparison fatal — run the first real build **without**
it, read the report, then decide.

#### Route B — `REDUCED_LOCAL`, available today, and what it cannot do

```bash
python3 $REPO/build_identity_index.py \
    --from-legacy-gene-names $OLD/uniprot_gene_mapping.csv \
    --out        $IDX \
    --unresolved $NEW/identity/unresolved_targets.csv
```

This reads **only** `Entry`, `Entry Name`, `Gene Names` and `Organism`. The corrupt `Mapped_Gene_Name`
column is not in the column list, so it is never even loaded. Verified end-to-end on 2026-08-13:
**4,585 records, 0 rejects, 2 recorded notes, `max_collapse = 0`**, and the Phase 0 STOP gate below
reports **12/12 PASS, exit 0** — including all three spot checks (`AVPR2 → P30518`, `TUBA1B → P68363`,
`IL2 → P60568`). The output is stamped `REDUCED_LOCAL` inside the parquet itself, so no downstream stage
can mistake it for the authoritative index.

**Its two limitations are hard:**

1. **No Entrez ids at all** (0 of 4,585) unless `--hgnc` is also supplied — reduced mode never fetches.
   `resolve_entrez()` therefore returns `UNKNOWN` for everything, and **Phase 1 resolves nothing**
   (measured: 0/8,568 edges, all `UNKNOWN_ENTREZ`; see Phase 1's STOP gate, which exists because the
   script still exits 0 in that state).
2. **The universe is capped at the 4,585 accessions the corrupt free-text search happened to return.**
   Per `KNOWN_ISSUE`, every one of those came from querying the drug-target gene list, so no other
   protein is reachable through it; per `ASSESSMENT` §1c, only 1,567/1,958 genes (80.03%) are repairable
   from this file, 391 need external re-derivation and 12 are absent entirely. **The re-dock gap's
   canonical-accession choices therefore still require the real files** — a reduced index is fine for
   re-deriving §1.2's reuse arithmetic (which was computed in exactly this universe) and useless for
   deciding which accession a *new* docking job should target.
3. Without `--proteome`, `reviewed` is `False` on every row and means *unknown*, not *TrEMBL*. It is
   recorded as such in `meta["row_counts"]["review_status_source"]`; do not read that column as a
   Swiss-Prot claim.

#### Pin the index, then never let a stage choose another one

```bash
IDX_SHA=$(python3 -c "import sys; sys.path.insert(0, '$REPO')
from dock_selectivity.identity import file_sha256
print(file_sha256('$IDX'))")
echo "$IDX_SHA" > $NEW/identity/identity_index.sha256
```

Every downstream command below passes `--index-sha256 $IDX_SHA`. A mismatch aborts that stage.

**STOP — verify before proceeding:**
```bash
python3 $REPO/audit.py identity \
    --index      $IDX \
    --meta       $NEW/identity/identity_index.meta.json \
    --unresolved $NEW/identity/unresolved_targets.csv \
    --expect-sha256 $IDX_SHA
echo "exit=$?"      # 0 = cleared. 2 = a check did not run = NOT cleared.
```
must report:
- `G5_entrez_collapse_factor_one` — `max_collapse <= 1` (the old map was mean 2.55, max 9; this is the guarantee whose absence caused everything). **A `REDUCED_LOCAL` index reports 0, which passes — that is not evidence of a working Entrez axis, it is evidence there is none.**
- `spot_AVPR2 / spot_TUBA1B / spot_IL2` — `AVPR2 → P30518`, `TUBA1B → P68363`, `IL2 → P60568`, **and** the impostors Q8NCT1 / Q13509 / P01589 no longer claim those symbols
- `G4_primary_symbol_in_symbols` for every row
- `G3_deterministic_ordering` — identical content digest under `PYTHONHASHSEED` 0/1/42. This checks the *artifact*; that the *build* is deterministic is `build_identity_index.py --check-determinism`, a different assertion
- `G2_total_accounting` — `n_in == n_resolved + n_rejected`, rejects listed in `unresolved_targets.csv`. This check **SKIPs** (exit 2) if the meta lacks `n_in`/`n_resolved`/`n_rejected`

If any check fails, stop. Everything downstream inherits this file.

---

### Phase 1 — Resolve the corrected interaction set

**Code prerequisite:** none — `resolve_interactions.py` exists and is tested. It **replaces** the
identity half of `identify_required_structures.py` for this rebuild; `FIX_PLAN.md` §3.1 remains
outstanding for the old script but does not block this phase.

**Requires an index with a working Entrez axis** — see Phase 0 Route B limitation 1.

```bash
python3 $REPO/resolve_interactions.py \
    --edges      /opt/data/multiscale_interactome_data/1_drug_to_protein.tsv \
    --index      $IDX \
    --index-sha256 $IDX_SHA \
    --drugs      /opt/data/drugbank/small_molecule_drug_links.csv \
    --vocabulary /opt/data/drugbank/drugbank_all_drugbank_vocabulary.csv \
    --out        $NEW/manifests/interactions_resolved.csv \
    --rejects    $NEW/manifests/interactions_resolved.rejects.csv \
    --report     $NEW/logs/interactions_resolved.invariants.json
```

Resolution is on `node_2` (**Entrez ID**, numeric on 8,568/8,568 edges), never `node_2_name`.
`node_2_name` is read for the rejects ledger and nothing else; the output's `gene` column is written by
`index.label(accession)` and is a label *derived from* the key, never an input to selection.

Flags worth knowing:
- `--vocabulary` is **required and always loaded**, even when a run has no name slugs. On this machine
  the path is a *directory* named `drugbank_all_drugbank_vocabulary.csv` containing
  `drugbank vocabulary.csv`; the script accepts either, and refuses a directory holding zero or several
  CSVs rather than choosing one.
- `--drugs` needs both `DrugBank ID` **and** `Drug Type`. The type filter is what actually does the work
  (`drug_links.csv` and `small_molecule_drug_links.csv` are byte-identical, md5 `60cd8851…`), and the
  column is mandatory precisely so a plain id list cannot silently widen the compound universe (G8).
- `--prefer-common-name` is **off** by default. Name tiers are pooled, not ranked: `L-Arginine` is
  DB17289's common name *and* DB00125's synonym, so its edges reject as `AMBIGUOUS_DRUG` with both
  candidates rather than resolving to a guess.
- `--obsolete-entrez FILE` turns retired Entrez ids into `OBSOLETE_ENTREZ` instead of `UNKNOWN_ENTREZ`.

**STOP — verify:**

```bash
python3 -c "
import pandas as pd
r = pd.read_csv('$NEW/manifests/interactions_resolved.csv')
assert len(r) > 0, 'EMPTY MANIFEST - do NOT proceed to Phase 2'
print(len(r), 'rows,', r.drugbank_id.nunique(), 'drugs,', r.accession.nunique(), 'accessions')
"
```

- **The manifest must not be empty.** This is not a hypothetical: against a `REDUCED_LOCAL` index built
  without `--hgnc`, the script resolves **0 of 8,568 edges** (all `UNKNOWN_ENTREZ`), writes a 0-row
  `interactions_resolved.csv`, and **exits 0**. Phase 2 would then classify all 24,173 jobs as
  `valid_non_interaction` and the "negative pool" would silently become the entire campaign. Assert a
  non-empty manifest before proceeding, every time.
- `n_edges_in == n_resolved_edges + n_rejected_edges` (asserted internally). Accounting has **three**
  terms, not two: `n_output_rows == n_resolved_edges − n_duplicate_edges_collapsed`. Do not read
  `n_output_rows` as "resolved edges".
- Compound-axis figures measured against the real edges, vocabulary and drug-links (with a synthetic
  full-coverage index, so protein-axis rejects were 0 by construction): **8,568 = 7,934 resolved + 634
  rejected**, `n_output_rows = 7,896` after collapsing **38** duplicate edges — 38 real edges carry both
  a DrugBank ID and a name slug for the same compound pointing at the same target
  (`DB00155` and `L-Citrulline`). 143 slug edges resolved; 35 slugs remain unresolved (salt/ester forms
  DrugBank does not name that way — `fludrocortisone-acetate`, `omeprazole-magnesium`, `gentamycin`).
  **554 edges reject as `NOT_SMALL_MOLECULE`** (156 distinct DrugBank IDs, e.g. octreotide DB00104):
  that is a reject, not a flag, and there is no unfiltered-edge-list output.
- Spot-check that `AVPR2` edges now carry **P30518** and not Q8NCT1.
- Confirm the **237 null-symbol edges** are now resolvable via Entrez (229 of 237 were, in the smoke
  test) — they were *structurally unreachable* by a symbol-keyed resolver.
- Confirm the 222 name-slug `node_1` rows are either resolved or explicitly rejected. Every rejected
  edge is in the rejects ledger with its axis, status, reason and candidates. Nothing is dropped.

---

### Phase 1b — `positives_planned.csv` · **NO PRODUCER EXISTS**

**This is the one genuinely missing piece of the new toolchain.** Phases 3 and 5 both consume
`manifests/positives_planned.csv`, and nothing in the repository writes it. Until it exists the rebuild
stops here.

What it must contain — schema `positives_planned` in `dock_selectivity/schemas.py`, primary key
`(drugbank_id, accession, fragment_id, cavity_index)`:

| column | rule |
|---|---|
| `drugbank_id`, `accession` | one row per **cavity** of each `POS_DOCKABLE` pair |
| `gene` | **required, non-null**, and `= index.label(accession)`. Never part of a key. This is the direct fix for the 13,452 blank gene names that produced 10,698 `*_nan_*` job directories |
| `fragment_id` | string normalised to `F<k>`; the old `'F1'`-vs-bare-`'1'` split must be resolved by the producer, not at join time |
| `cavity_index` | int |
| `sample_type` | `'positive'` |
| `selected_by` | **required, non-null** free-text justification, e.g. `entrez=554 via edge (DB00035,554)`. Had `required_structures.csv` carried this column, `AVPR2 → Q8NCT1` would have been legible in the plan file itself |

Its inputs are `manifests/interactions_resolved.csv` (demand), the CavitySpace cavity manifest
(`$OLD/cavity_mapping.csv` today, or the manifest Phase 6 will build) and the ligand SDF set
(`/opt/data/drugbank/drugbank_approved_split`). Expected output at the §1.2.1 figures: **5,287 pairs /
1,101 drugs / 1,283 targets → 18,166 rows** — the demand side, iterated from the interaction edges,
never from the cavity supply.

Two routes, either is acceptable:
1. the `FIX_PLAN.md` §3.1 rewrite of `identify_required_structures.py`, emitting the new schema; or
2. a new `plan_positives.py`, leaving the old script alone until the wider fix pass.

The same producer should emit `manifests/redock_queue.positives.csv` (schema `redock_queue`) — the
**10,866** planned positive jobs not already correctly docked (18,166 − 7,300).

---

### Phase 2 — Classify every old job under correct identity (read-only)

**Code prerequisite:** none — `classify_existing.py` exists and is tested.

```bash
python3 $REPO/classify_existing.py \
    --existing-results  $OLD/consensus_docking_results \
    --index        $IDX \
    --index-sha256 $IDX_SHA \
    --interactions $NEW/manifests/interactions_resolved.csv \
    --out-map      $NEW/manifests/job_classification.csv \
    --out-pool     $NEW/manifests/negatives_pool.csv \
    --out-summary  $NEW/logs/classify_existing_summary.json \
    --out-evidence $NEW/manifests/job_classification.evidence.csv \
    --old-plan     $OLD/required_structures_with_negatives.csv \
    --default-fragment F1 \
    --checkpoint   $NEW/logs/classify_existing_checkpoint.jsonl
```

Parses all 24,173 directory names **right-anchored** (strip trailing `_cavity_<N>`, strip optional
`_-F<k>-model_v1`, then test each token against the accession set — this resolves 24,173/24,173, unlike
the old regex which failed on 1,777). It reads directory **names** and one `stat` per job; it never opens
`final_results.csv` (24,173 × 184 KB is 4.4 GB of USB reads for a fact a `stat` already establishes).
About 132 s cold, and resumable: the checkpoint caches only the filesystem observations, so a re-run
after rebuilding the index always reclassifies rather than serving stale answers.

Flags worth knowing:
- `--old-plan` supplies `old_sample_type` **as evidence only** — it never touches the corrected
  classification, and a test proves the classification is byte-identical with and without it. Both sides
  of that 4-key merge are normalised through `normalize_fragment_id`, which is what defeats the
  `'F1'`-vs-bare-`'1'` split. A plan key claiming both labels raises rather than silently last-wins.
- `--default-fragment F1` is an explicit, logged, per-row-recorded choice (`fragment_source`) applied to
  the 22,430 names carrying no fragment token. `jobid.py` deliberately refuses to infer it, so someone
  has to; here it is not silent.
- Every output path is checked against `$OLD` before a byte is written. Pointing an output inside the
  old run directory is refused.

**STOP — verify** `job_classification.csv` reproduces the expected classification (one row per directory,
always; `n_in == n_resolved + n_rejected` is asserted):

| Expected | Count |
|---|---:|
| `true_positive` | 7,300 |
| `valid_non_interaction` (→ `negatives_pool.csv`) | 16,579 |
| `incomplete` — no `final_results.csv`, nothing to reuse | 294 |
| `unparseable` | 0 |
| **total** | **24,173** |

**Reconciling 16,579 with §1.2's 16,873.** §1.2's pool arithmetic is over all 16,873 valid
non-interactions. `negatives_pool.csv` carries **294 fewer** because `incomplete` outranks
`valid_non_interaction`: a job with no `final_results.csv` has no payload to reuse whatever its DTI
status (measured on the real campaign: 162 old-negative, 132 old-positive, **0** of the 7,300 true
positives). Their DTI answer is preserved in the evidence file's `is_known_interaction`, and their
pockets remain available as *fresh* jobs, so nothing is lost — but expect the drawable figure in Phase 3
to move by up to those 294.

Also check: `classify_existing_summary.json` lists every non-directory entry by name (the campaign has one
stray `docking_automation.log`) and every unparseable name with the parser's message. The scan refuses to
run if any accession in `interactions_resolved.csv` is absent from the index — the two artifacts must be
one universe, and there is no flag to bypass that.

One contract for Phase 5: `excluded_reason` is populated on every `valid_non_interaction` row (the
schema forces exactly one of `new_job_dir` / `excluded_reason`), and it reads "carried over only if the
balanced sampler draws it". **`curate_jobs.py` must take drawn negatives from `negatives_drawn.csv`, not
by filtering `job_classification` on `new_job_dir.notna()`** — that column is populated only for
`true_positive` rows, and even there it is a proposal.

---

### Phase 3 — Re-run the balanced sampler against the corrected positives

**Code prerequisite:** none — `sample_negatives.py` exists and is tested. It **replaces**
`generate_negative_samples.py`, which is left untouched and is not used by this rebuild;
`FIX_PLAN.md` §3.2 is therefore superseded for this plan's purposes.

```bash
python3 $REPO/sample_negatives.py \
    --index        $IDX \
    --index-sha256 $IDX_SHA \
    --interactions $NEW/manifests/interactions_resolved.csv \
    --positives    $NEW/manifests/positives_planned.csv \
    --pool         $NEW/manifests/negatives_pool.csv \
    --out          $NEW/manifests/negatives_drawn.csv \
    --redock-queue $NEW/manifests/redock_queue.negatives.csv \
    --balance-unit pair \
    --prefer-pool \
    --seed 42 \
    --pool-eligibility $NEW/manifests/negatives_pool_eligibility.csv \
    --deficits         $NEW/logs/negatives_deficits.csv \
    --report           $NEW/logs/sample_negatives_report.json
```

- `--prefer-pool` is reuse-first: phase 1 draws only from already-docked pool candidates, phase 2 tops
  up with fresh jobs. Without it the sampler is design-first and reuses far less (though within a
  chosen pair an existing pool job is always used before a fresh one — re-docking a docked pocket is
  never right). **Balance is the hard constraint and reuse is the optimisation**: the repair pass will
  release a reused unit if that is the only way to close a deficit, and the trade is visible in
  `report['repair']` and `report['drawn']['n_from_pool']`.
- `--balance-unit pair` is the §1.2.4 recommendation. `job` reproduces what the old code did. **The unit
  is recorded in the report and in every `selected_by` string, because the STOP gate has to say which
  one it asserts.** In `pair` mode the deficits ledger counts *pairs* while
  `report['drawn']['n_from_pool']` still counts *rows*; the two numbers legitimately differ.
- **Do not pass `--require-balance`** on the first run: the corrected design is known to be infeasible
  from the pool alone (§1.2.3), and that flag exits **5**. Use it later, against a pool topped up by
  Phase 7's output, when full balance is genuinely expected.
- Exit codes: `0` ok · `2` usage · `3` input contract (missing file, schema violation, SHA256 mismatch) ·
  `4` invariant violation · `5` `--require-balance` and the design could not be balanced.

Two input gates will stop a mismatched pipeline on purpose: every planned positive pair must appear in
`interactions_resolved.csv` (or the avoid-list is not provably complete over the positives), and every
`(accession, gene)` in `positives_planned.csv` and `negatives_pool.csv` must satisfy
`gene ∈ index.by_accession(accession).symbols`. The second accepts synonyms (`ADRA1C` for P35348) but
the emitted label is always `index.label(accession)`, so downstream never sees two spellings of one
protein.

**STOP — verify.** The gate must **state its balance unit** and assert these, not "a large majority":

| Assertion | Job-balanced | Pair-balanced |
|---|---:|---:|
| Demand (from `positives_planned.csv`) | 18,166 jobs | 5,287 pairs |
| Pool offered | 16,873 jobs (16,579 after `incomplete`) | — |
| Structurally eligible | 8,063 (47.8%) | — |
| **Drawn from the pool (max-flow optimum)** | **6,430 = 35.4% of demand** | **2,200 pairs** |
| Shortfall → `redock_queue.negatives.csv` | **11,736 jobs** | **3,087 pairs** |
| Contamination — negatives that are known interactions | **0** | **0** |

- `sample_negatives.py` is greedy + repair, not max-flow (the open phase has ~1.4M candidate edges at
  real scale). Read a residual deficit as *"at most this much is achievable by this heuristic"*, not
  *"this much is impossible"*. The earlier max-flow comparison put the repo's greedy within 0.15% of the
  optimum, so a drawn count materially below 6,430 means an input changed, not that the heuristic gave up.
- Reconcile `negatives_pool_eligibility.csv` against §1.2.2: reasons are assigned in a fixed precedence
  (`accession_not_in_index` > `known_interaction` > the `*_no_positives` axis reasons > `no_final_results`
  > `duplicate_pocket`) and must sum to the pool size exactly. The axis reasons are directly comparable
  with the 8,063 figure.
- Balance holds **in accession space, not label space**: `max|pos−neg| <= 1` per drug and per target on
  the declared unit, over the entities that carry a quota. An entity with zero positives has **no** quota
  key at all — that absence *is* the enforcement.
- `negatives_drawn.csv` cannot contain a duplicate pocket by construction (the old sampler's
  `np.random.choice` over cavities could emit the same job twice). A consequence worth knowing: a pair's
  drawable capacity is capped at `|pockets(accession)|`, a real constraint the old code did not have.
- Determinism: `negatives_drawn.csv`, `negatives_pool_eligibility.csv`, `negatives_deficits.csv` and the
  report JSON are byte-identical across runs and across `PYTHONHASHSEED` values. No timestamp is written
  into any of them.

---

### Phase 4 — DECISION POINT: how much to re-dock

`redock_queue.csv` now exists. Three options; **B is still recommended, at roughly double its previously
budgeted size.**

| | A — reuse only | **B — reuse + fill the gap** | C — full re-run |
|---|---|---|---|
| New **positive** jobs | 0 | **10,866** (18,166 planned − 7,300 docked) | 18,166 |
| New **negative** jobs | 0 | **11,736** job-balanced / ~10,607 pair-balanced | ~18,166 |
| **New docking jobs** | **0** | **22,602** (job-balanced) | ~36,000 |
| Wall-clock (64 cores) | 0 | **~10–13 days** | 16–20 days |
| New storage (evidence-only retention) | ~15.3 GB | **~41 GB** | ~45 GB |
| Intended-pair coverage | 25.4% | **63–69%** | 63–69% — *identical* |
| Existing compute reused | 100% | 100% | ~0% |

**What changed, and why.** The old Option B figure of ~13,900 jobs assumed a **3,293-job** negative
top-up. That number double-counts: it credits the 7,317 "reusable" negatives against *both* the existing
positives and the new ones. Under the joint per-drug **and** per-target quota, only **6,430** of the
16,873 pool jobs are drawable at all (§1.2.3), so the negative side needs **11,736** new jobs, not 3,293.
The positive side is unchanged in kind: 18,166 planned − 7,300 already docked correctly = **10,866**.

Arithmetic behind the two derived cells, at `ASSESSMENT` §3's measured rates:
22,602 × 0.616–0.806 core-h/job = **13,920–18,220 core-h** ≈ 9–12 days on 64 cores (the reuse analysis
quotes 10–13; budget the wider band). Storage at 1.14 MB/job evidence-only:
13,730 reused ≈ 15.3 GB + 22,602 new ≈ 25 GB ≈ **41 GB**.

**Pair-balancing the negatives is the better design and it is also cheaper** (§1.2.4): ~10,607 negative
jobs instead of 11,736, and the two classes become comparable per pair instead of 3.38 poses on
positives against 1.36 on negatives.

**C still buys no coverage over B** — it re-docks the 2,091 pairs that are already correctly docked and
re-buys the 6,430 drawable negatives at full price. Its only gain is uniform provenance, which
`job_identity.json` already records per job.

Two hard ceilings to know about:
- 2,030 intended pairs have **no CavitySpace cavity for their correct protein**. No amount of re-docking
  reaches them; **69.1%** is the realistic maximum intended-pair coverage.
- **Transient disk during Phase 7 is now the binding storage number, not the retained figure.** At the
  measured 59.6 MB/job before cleanup, 22,602 fresh jobs is ~1.3 TB against 5.3 TB free. The Phase 7
  post-job hook that discards or `xz`-compresses `complex_*.pdb` once `final_results.csv` is written is
  load-bearing, not tidiness.

**What is *not* worth budgeting.** The 5,072 mislabelled-positive jobs contribute **+83 negatives**
(1.6% of themselves, 0.7% of the deficit). They are not a re-dock saving. The 4,334 of them that sit on
zero-positive targets remain scientifically real and stay in `$OLD`; if they are wanted, it must be as a
**separately declared, unbalanced decoy set for a secondary analysis** — never as balanced negatives in
the primary dataset.

---

### Phase 5 — Materialise the curated directory

**Code prerequisite:** none — `curate_jobs.py` exists and is tested; it carries the new naming scheme and
the `job_identity.json` writer. (`FIX_PLAN.md` §3.3 remains outstanding for `batch_dock.py`, which is
Phase 7's problem, not this one.)

```bash
# Dry run FIRST — prints what it would copy, writes nothing.
# --dry-run with --report/--rejects/--manifest-out is an argparse error, not a silent no-write.
python3 $REPO/curate_jobs.py --dry-run \
    --job-classification $NEW/manifests/job_classification.csv \
    --drawn       $NEW/manifests/negatives_drawn.csv \
    --positives   $NEW/manifests/positives_planned.csv \
    --index       $IDX \
    --index-sha256 $IDX_SHA \
    --src $OLD/consensus_docking_results \
    --dst $NEW/docking_results \
    --evidence-only --gold-output poses

python3 $REPO/curate_jobs.py \
    --job-classification $NEW/manifests/job_classification.csv \
    --drawn       $NEW/manifests/negatives_drawn.csv \
    --positives   $NEW/manifests/positives_planned.csv \
    --index       $IDX \
    --index-sha256 $IDX_SHA \
    --src $OLD/consensus_docking_results \
    --dst $NEW/docking_results \
    --evidence-only --gold-output poses \
    --manifest-out $NEW/manifests/run_manifest.reused.csv \
    --report       $NEW/logs/curate_jobs_report.json \
    --rejects      $NEW/logs/curate_jobs_rejects.csv \
    --code-revision "$(git -C $REPO rev-parse --short HEAD)"
```

**`--index` is required and is not optional in spirit either.** The whole point of the new scheme is
that the gene is `IdentityIndex.label(accession)` at creation time, so the name cannot disagree with the
receptor. Taking the gene from a manifest column would reintroduce the defect.

`--evidence-only` applies the 2.4% rule from §1.3: copies `final_results.csv`, the four `results.csv`,
`out.sdf`/`out.dok`/`gold_soln_*.sdf` and `input/*.mol2`; **skips `complex_*.pdb` and the retry logs**
(the deny-list is applied *after* rule expansion, so a tree rule cannot leak a `complex_*.pdb`). Writes
`job_identity.json` into every new directory, with `provenance='reused_from'` and `source_job` naming
the old directory, plus the evidence file list, the index SHA256, the code revision and the retention
policy — and **no timestamp**, so two runs produce a byte-identical tree.

Flags worth knowing:
- `--gold-output poses` (default) vs `all` — see the measured correction in §1.3. Recorded per job.
- `--on-name-collision fail` (default) stops the whole run when two old directories claim one new name
  (they can differ only in the gene token: `DB00035_AVPR2_Q8NCT1_cavity_2` vs
  `DB00035_ARRDC4_Q8NCT1_cavity_2`). Picking one would be exactly the last-wins collapse this rebuild
  exists to eliminate. `skip` excludes **all** sides with a recorded reason, never one arbitrarily.
- `--assume-fragment F1` is **off** by default; a null `fragment_id` is an error, not an inference. When
  used, each affected job records `fragment_source='assumed:F1 (--assume-fragment)'` and the count is
  logged at WARNING. The fragment is resolved *before* the pocket key is matched against
  `positives_planned.csv`, so an assumed fragment participates in matching rather than silently failing.
- `--workers` defaults to **1** because the source is a slow USB disk. 4 workers produce a byte-identical
  tree; output ordering is plan order regardless. Expect ~70 evidence file stats per job (~1M over the
  run); a resume run does **zero** source I/O for already-complete jobs.
- Error-severity rejects fail the run (exit 1) unless `--allow-errors`. Five classes: a pool draw absent
  from the reuse map; a pool draw whose reuse row is not `valid_non_interaction`; a drawn-vs-reuse
  pocket-key disagreement; a missing source directory; an accession absent from the index.

Copy — never move, never hardlink. The module never calls `os.rename`, `os.link`, `shutil.move` or any
unlink under `--src`, and refuses to run if `--dst` is inside `--src` or the reverse. `$OLD` must remain
a complete, independent, untouched copy.

Expected result: **13,730 directories ≈ 15.3 GB** — 7,300 reused positives + 6,430 drawn pool
negatives (§1.2).

#### Phase 5b — assemble `run_manifest.csv` · **NO PRODUCER EXISTS**

`--manifest-out` writes only the jobs `curate_jobs.py` materialised, all with
`provenance='reused_from'` and `source_job` set. The authoritative `run_manifest.csv` is that file
concatenated with the `newly_docked` rows derived from `redock_queue.csv` (schema `run_manifest`;
`provenance='newly_docked'` and `source_job` null — the schema's row check enforces
`provenance=='reused_from'` **iff** `source_job` is set). `audit.py run` asserts the union equals
`positives_planned ∪ negatives_drawn` exactly. This concatenation is currently a manual step; the fixture
in `tests/test_audit.py` shows the shape it must have.

**STOP — verify:**
```bash
python3 $REPO/audit.py curated \
    --dir      $NEW/docking_results \
    --manifest $NEW/manifests/run_manifest.csv \
    --index    $IDX \
    --expect   subset
echo "exit=$?"
```
- every directory contains `job_identity.json` and `final_results.csv`
- `gene == index.label(accession)` for **100%** of jobs (the check the old pipeline never had), and
  `accession → gene` collapse factor is exactly 1
- the directory name matches its own `job_identity.json`, and every reused job names its source
- no `complex_*.pdb` or `log_*.log` leaked into the curated tree; no stray files
- `--expect subset` (correct here): every `reused_from` row must be on disk, `newly_docked` rows need
  not be yet, and **no directory may exist that is absent from the manifest**. `--expect exact` is the
  Phase 7/8 end-state gate ("directory count equals `run_manifest.csv` row count exactly").
- `--limit N` gives a quick partial pass but reports SKIP → **exit 2**, so a partial audit can never read
  as a cleared gate
- `$OLD` byte count and file count unchanged from before Phase 5

---

### Phase 6 — Prepare structures for the re-dock (minimal sets only) · **NOT YET IMPLEMENTED**

**Code prerequisite — not done.** `FIX_PLAN.md` §3.5 and §3.6: `extract_cavities.py` must take a
required-set argument, and `extract_alphafold_models.py` must have its `cavity_mapping.csv` fallback
deleted.

> **The commands below are the target CLI, not a working one.** `extract_cavities.py`,
> `extract_alphafold_models.py`, `fix_required_pdbs.py` and `convert_pdb_to_pdbqt.py` contain **zero
> `add_argument` calls between them** — they accept no arguments at all and drive themselves from
> module-level constants. Running these lines today does not fail with a usage error; it runs the old
> supply-driven behaviour and extracts 38,709 cavities and 13,975 structures. Do not paste them until
> §3.5/§3.6 land.

```bash
python3 $REPO/extract_cavities.py --manifest-only \
    --tarballs /opt/data/cavity_space/af_strong_cavities_data_all \
    --out $NEW/manifests/cavity_manifest.csv          # names + member listings, NO unpacking

python3 $REPO/extract_cavities.py \
    --manifest $NEW/manifests/cavity_manifest.csv \
    --required $NEW/manifests/redock_queue.csv \
    --out-dir  $NEW/structures/cavities

python3 $REPO/extract_alphafold_models.py --required $NEW/manifests/redock_queue.csv --out-dir $NEW/structures/alphafold
python3 $REPO/fix_required_pdbs.py       --required $NEW/manifests/redock_queue.csv --out-dir $NEW/structures/fixed
python3 $REPO/convert_pdb_to_pdbqt.py    --required $NEW/manifests/redock_queue.csv --out-dir $NEW/structures/pdbqt
```

`redock_queue.csv` is the required-set argument for all four steps — that is why the schema declares it
that way (§2). Note it must be the **concatenated** queue (positives + negatives), not just the half
`sample_negatives.py` wrote.

**The larger negative budget from §1.2 does not enlarge this phase.** A negative can only be drawn on a
target that carries positives, and only at a pocket that already appears in `positives_planned` or the
pool — so the negative top-up needs **no additional structures**. The counts below are driven by the
positive design alone and are unchanged.

**STOP — verify the minimality that was the point of all this:**

| | Old campaign | Expected now |
|---|---:|---:|
| AlphaFold accession dirs | 5,661 | **~890** |
| Cavity dirs | 38,709 | **~2,900** |
| Disk for structure prep | 84 GB | **~6 GB** |

If either count comes out near the old figure, a fallback path survived — stop and fix it. Also confirm
`extract_alphafold_models.py` **exits non-zero** when `--required` is absent (it used to log a *warning*
and then extract 13,975 structures — a warning-level degradation of the scientific universe, which in a
multi-thousand-line log is operationally indistinguishable from silence).

---

### Phase 7 — Dock the gap · **NOT YET IMPLEMENTED**

**Code prerequisite — not done.** `FIX_PLAN.md` §3.3: `batch_dock.py` must consume a plan carrying
accessions and never resolve identity itself (its `dict(zip(...))` resolver is still live at `:2034`),
plus a post-job hook that discards or `xz`-compresses `complex_*.pdb` once `final_results.csv` is
written, so `$NEW` never accumulates the 1.4 TB again.

> **Target CLI, not a working one.** `batch_dock.py` has no `argparse` at all today, so `--queue`,
> `--run-dir`, `--test-mode` and `--max-jobs` do not exist, and it still switches docking mode on the
> *presence* of a negatives file rather than on an explicit flag.

```bash
# Smoke test FIRST — 20 jobs, all four engines
python3 $REPO/batch_dock.py --queue $NEW/manifests/redock_queue.csv --run-dir $NEW --test-mode --max-jobs 20
```

**STOP — verify on the 20:** all four engines produced `results.csv`; `final_results.csv` has RMSD;
`job_identity.json` written with the correct accession; `gene == index.label(accession)`; the
receptor path in the job log contains that accession; `complex_*.pdb` cleaned up as configured.

Then run for real, staged by engine as the config already does:
```bash
nohup python3 $REPO/batch_dock.py --queue $NEW/manifests/redock_queue.csv --run-dir $NEW > $NEW/logs/batch_dock.log 2>&1 &
```
Expect **~10–13 days** for the full 22,602-job Option B queue (Phase 4), not the 6–8 days budgeted
before the negative shortfall was measured. It is resumable — every stage skips jobs with valid existing
output. Watch transient disk: ~1.3 TB before the `complex_*.pdb` cleanup hook.

---

### Phase 8 — Parse and analyse · **NOT YET IMPLEMENTED**

**Code prerequisite — not done.** `FIX_PLAN.md` §3.4, all five `parse_prepare.py` defects. The
10,000-file cap is the one that silently dropped 58% of the last campaign; do not run this phase until it
is gone and discovery is checked against `run_manifest.csv`.

> **Target CLI, not a working one.** `parse_prepare.py` has no `argparse` at all; `--run-dir` and `--out`
> do not exist, and the paths are module-level constants pointing at the *old* run directory.

```bash
python3 $REPO/parse_prepare.py --run-dir $NEW --out $NEW/analysis/combined_results.parquet
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
5. **`run_full_workflow.py` currently never calls `generate_negative_samples.py`** — do not use it for this rebuild until §3.8 lands. Run the phases above by hand. (It would in any case call the wrong sampler now: this plan uses `sample_negatives.py`.)
6. **HEAD is not the code that produced the old parquet** (`FIX_PLAN.md` F-B). Do not expect any rebuild to reproduce it. Record the producing revision in `provenance.json` from now on.
7. **An empty output is not a passed stage.** `resolve_interactions.py` exits 0 after resolving 0 of 8,568 edges against an Entrez-less index (measured). Every STOP gate asserts a *count*, not merely an exit code.
8. **Exit 2 from `audit.py` means "not cleared".** A check that could not run is not a check that passed. `--allow-skips` exists; using it is a decision to record, not a default.
9. **Say which balance unit you asserted.** Jobs and pairs give different, both-correct answers (§1.2.4). A gate that does not name its unit has not asserted anything.

---

## 5. Expected end state

| | Old | New (Option B, job-balanced) |
|---|---:|---:|
| Run directory size | 1,550 GB | **~41 GB** |
| Job dirs | 24,173 | **~36,300** (13,730 reused + 22,602 fresh) |
| …reused | — | 7,300 positives + 6,430 pool negatives |
| …fresh | — | 10,866 positives + 11,736 negatives |
| Jobs with correct identity | 13,128 (54.3%) | **100%** |
| Jobs reaching the analysis | 10,000 (41.4%) | **100%** |
| False positives in the positive class | 45.8% | **0%** |
| Contaminated negatives | 5 | **0** |
| Balance holds in accession space | no | **yes** (unit declared) |
| Negative pairs per positive pair | 2.48× at 0.40× depth | **1.00× at equal depth** (pair-balanced) |
| Intended-pair coverage | 25.4% | **63–69%** |
| AlphaFold structures materialised | 5,661 | ~890 |
| Cavities materialised | 38,709 | ~2,900 |

Pair-balancing the negatives instead (the §1.2.4 recommendation) trims the fresh negative side to
~10,607 jobs, i.e. ~35,200 job dirs and ~40 GB, and is the design that matches the scientific claim.

The single most valuable outcome is not the disk saving. It is that the positive class stops being 45.8%
fabricated, which is what made the headline RMSD result an artifact — the mislabelled half supplied 115%
of the observed effect, and the corrected contrast was −0.061 Å (p = 0.675). After this rebuild the
measurement means what it says.
