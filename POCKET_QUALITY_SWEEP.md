# Pocket-quality cutoff sweep

How many drug-target pairs survive at different AlphaFold pLDDT cutoffs on the CavitySpace
pocket, measured on the real inputs. Every number below was produced by running the actual
planning stages, not estimated.

Run directory: `/home/onur/.claude/jobs/640c72c3/tmp/sweep`
Cavity archives: `/opt/data/cavity_space/af_strong_cavities_data_all` (38,709 archives, 1 cavity each)
Inputs: `/media/onur/Elements/cavity_space_consensus_docking/inputs` (read-only)

## 1. What was run

Planning built once, then reused by every cell:

```
python3 run_full_workflow.py \
    --inputs-dir /media/onur/Elements/cavity_space_consensus_docking/inputs \
    --run-dir /home/onur/.claude/jobs/640c72c3/tmp/sweep \
    --mode balanced --only planning
```

84.2 s end to end. Phase 2b (`score_cavity_quality.py`) scored **all 38,709 cavities in 41.7 s
single-threaded, with zero rejects** -- `manifests/cavity_quality_rejected.csv` is a header and
nothing else. `manifests/cavity_quality.csv` therefore has exactly 38,709 rows, one per cavity in
`cavity_manifest.csv`, covering 13,600 distinct accessions.

Each of the 25 grid cells then re-ran only the two threshold-dependent stages, into its own
output folder under `<run-dir>/cells/<cell>/`, e.g. for pLDDT>=70 / fraction>=0.80:

```
python3 plan_positives.py \
    --interactions <run>/manifests/interactions_resolved.csv \
    --index <run>/identity/identity_index.parquet \
    --cavities <run>/manifests/cavity_manifest.csv \
    --drugs <inputs>/small_molecule_drug_links.csv \
    --ligands <inputs>/ligands \
    --out <run>/cells/p70_f080/positives_planned.csv \
    --rejects <run>/cells/p70_f080/positives_rejected.csv \
    --out-dir <run>/cells/p70_f080/step_positives --quiet \
    --cavity-quality <run>/manifests/cavity_quality.csv \
    --plddt-min 70.0 --min-plddt-fraction 0.8

python3 sample_negatives.py \
    --index <run>/identity/identity_index.parquet \
    --interactions <run>/manifests/interactions_resolved.csv \
    --positives <run>/cells/p70_f080/positives_planned.csv --no-pool \
    --out <run>/cells/p70_f080/negatives_drawn.csv \
    --report <run>/cells/p70_f080/negatives_drawn.report.json \
    --deficits <run>/cells/p70_f080/negatives_drawn.deficits.csv \
    --out-dir <run>/cells/p70_f080/step_negatives --quiet \
    --cavity-quality <run>/manifests/cavity_quality.csv \
    --plddt-min 70.0 --min-plddt-fraction 0.8
```

About 13 s per cell; the whole 25-cell grid is under 6 minutes. No cell was skipped for cost.

Three checks make the table trustworthy:

* Every `--min-plddt-fraction 0.0` cell reproduces the unfiltered baseline **exactly** -- 38,709
  pockets, 5,460 pairs, 18,842 positive jobs, at all four `--plddt-min` values. The filter is a
  true no-op when the bar is zero, so the machinery is not moving numbers on its own. (Those four
  redundant rows are omitted from the table below; the baseline row is them.)
* `quality.fraction_at(frame, t)` reproduces the stored `frac_ge_50/70/90` column for all 38,709
  rows with **zero** mismatches, so any cutoff can be evaluated from the CSV without re-reading
  7 GB of archives.
* Every dropped pocket is in the cell's `positives_rejected.csv` with reason `pocket_quality` and
  the numbers, e.g.
  `all 1 pocket(s) failed the pLDDT filter: F1/cav1 frac_ge_70=0.62 < 0.80 (n_residues=53)`.
  Nothing is dropped silently.
* The filter reaches both classes equally. Re-checking the pLDDT>=70/frac>=0.70 cell against
  `quality.passes()` directly: 0 of 13,222 positive rows and 0 of 13,222 negative rows sit on a
  failing pocket. A threshold applied to positives only would be a confound rather than a filter.

One caveat on the code, not the numbers: the first run of the test suite
(`python3 -m unittest discover -s tests -t .`) reported `Ran 1571 tests ... FAILED (failures=1)`
while the sweep was still loading the machine, and the pipeline that captured it truncated the
traceback, so **the failing test was not identified**. Five subsequent full runs -- four idle, one
under 4x CPU load -- all reported `OK` on 1571 tests. It is recorded here rather than omitted; the
sweep numbers themselves come from the stages' own artifacts and invariant reports, not from tests.

## 2. Pocket-pLDDT distribution over all 38,709 cavities

| statistic | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| mean pocket pLDDT | 21.74 | 68.78 | 77.87 | 86.08 | 98.86 |
| median pocket pLDDT | 21.59 | 75.00 | 86.34 | 92.07 | 98.88 |
| worst residue in pocket | 7.26 | 28.17 | 35.64 | 53.52 | 98.67 |
| best residue in pocket | 24.57 | 92.28 | 95.91 | 97.94 | 98.97 |
| frac_ge_50 | 0.00 | 0.74 | 0.88 | 1.00 | 1.00 |
| frac_ge_70 | 0.00 | 0.56 | 0.74 | 0.91 | 1.00 |
| frac_ge_90 | 0.00 | 0.07 | 0.34 | 0.61 | 1.00 |
| residues per pocket | 7 | 40 | 60 | 91 | 642 |

Read that carefully, because it is what makes the cutoffs interpretable:

* The pocket set is **already mostly confident**. Median mean-pocket pLDDT is 77.9; the upper
  quartile is above 86. Half of all pockets have >= 74% of their residues in AlphaFold's
  confident band, and a quarter have >= 91%.
* But the tails are real. `min_plddt` has a median of 35.6 -- the *typical* pocket contains at
  least one residue in AlphaFold's "very low" band, so a filter on the pocket *minimum* would be
  catastrophic and is correctly not what the metric does.
* `frac_ge_90` is where the spread lives: p25 = 0.07, median = 0.34, p75 = 0.61. A cutoff at
  pLDDT>=90 is a genuinely different selection, not a stricter version of the same one.

## 3. The sweep

`pockets passing` is over all 38,709 cavities. `study targets` / `drugs` / `pairs` are what
survives in `positives_planned.csv`. `deficit` is the total negative-sampling shortfall across
both the drug and target axes. `docked-pocket mean pLDDT` is the mean, over the positive jobs
actually planned, of each job's pocket mean pLDDT -- the confidence you are buying.

| criterion | pockets passing | % pockets | targets w/ pocket | study targets | drugs | **pairs** | % pairs | pos jobs | neg jobs | deficit | docked-pocket mean pLDDT | % jobs pocket<70 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **none (baseline)** | 38709 | 100.0% | 13600 | 1294 | 1125 | **5460** | 100.0% | 18842 | 18842 | 0 | 80.5 | 16.4% |
| pLDDT>=50, frac>=0.50 | 36066 | 93.2% | 13116 | 1285 | 1124 | **5437** | 99.6% | 18103 | 18103 | 0 | 82.1 | 13.0% |
| pLDDT>=50, frac>=0.70 | 31109 | 80.4% | 12441 | 1260 | 1112 | **5381** | 98.6% | 16268 | 16268 | 0 | 84.0 | 6.3% |
| pLDDT>=50, frac>=0.80 | 24986 | 64.5% | 11288 | 1202 | 1105 | **5191** | 95.1% | 13338 | 13338 | 0 | 86.4 | 2.0% |
| pLDDT>=50, frac>=0.90 | 17774 | 45.9% | 9091 | 1017 | 1091 | **4546** | 83.3% | 9931 | 9931 | 0 | 89.2 | 1.1% |
| pLDDT>=50, frac>=1.00 | 10965 | 28.3% | 6429 | 797 | 1042 | **3938** | 72.1% | 7007 | 7007 | 0 | 91.8 | 0.4% |
| pLDDT>=70, frac>=0.50 | 31017 | 80.1% | 12018 | 1257 | 1116 | **5373** | 98.4% | 16611 | 16611 | 0 | 84.0 | 5.2% |
| pLDDT>=70, frac>=0.70 | 22104 | 57.1% | 10015 | 1168 | 1099 | **5079** | 93.0% | 13222 | 13222 | 0 | 86.9 | 0.0% |
| pLDDT>=70, frac>=0.80 | 15798 | 40.8% | 8125 | 1021 | 1081 | **4571** | 83.7% | 9640 | 9640 | 0 | 90.0 | 0.0% |
| pLDDT>=70, frac>=0.90 | 10326 | 26.7% | 5999 | 832 | 1019 | **3998** | 73.2% | 7167 | 7167 | 0 | 92.2 | 0.0% |
| pLDDT>=70, frac>=1.00 | 5396 | 13.9% | 3691 | 569 | 948 | **3004** | 55.0% | 4579 | 4579 | 0 | 94.1 | 0.0% |
| pLDDT>=80, frac>=0.50 | 26317 | 68.0% | 11110 | 1224 | 1112 | **5212** | 95.5% | 15345 | 15345 | 0 | 85.1 | 1.5% |
| pLDDT>=80, frac>=0.70 | 16301 | 42.1% | 8532 | 1067 | 1091 | **4834** | 88.5% | 10581 | 10581 | 0 | 89.2 | 0.0% |
| pLDDT>=80, frac>=0.80 | 11016 | 28.5% | 6512 | 887 | 1060 | **4251** | 77.9% | 7786 | 7786 | 0 | 91.8 | 0.0% |
| pLDDT>=80, frac>=0.90 | 6708 | 17.3% | 4421 | 673 | 972 | **3413** | 62.5% | 5646 | 5646 | 0 | 93.7 | 0.0% |
| pLDDT>=80, frac>=1.00 | 3142 | 8.1% | 2393 | 408 | 809 | **2253** | 41.3% | 3212 | 3212 | 0 | 95.1 | 0.0% |
| pLDDT>=90, frac>=0.50 | 13657 | 35.3% | 7797 | 1010 | 1080 | **4620** | 84.6% | 9751 | 9751 | 0 | 88.9 | 0.0% |
| pLDDT>=90, frac>=0.70 | 6839 | 17.7% | 4684 | 712 | 1007 | **3698** | 67.7% | 5816 | 5816 | 0 | 92.8 | 0.0% |
| pLDDT>=90, frac>=0.80 | 4291 | 11.1% | 3145 | 509 | 920 | **2764** | 50.6% | 4109 | 4109 | 0 | 94.6 | 0.0% |
| pLDDT>=90, frac>=0.90 | 2486 | 6.4% | 1930 | 349 | 720 | **1877** | 34.4% | 2780 | 2780 | 0 | 95.8 | 0.0% |
| pLDDT>=90, frac>=1.00 | 970 | 2.5% | 829 | 161 | 424 | **845** | 15.5% | 1218 | 1218 | 0 | 96.8 | 0.0% |

## 4. What the table says

**The headline: pair loss lags pocket loss, sharply.** At pLDDT>=70 / frac>=0.70 the filter
throws away 42.9% of all pockets but only 7.0% of drug-target pairs. That asymmetry is the
whole reason the filter is worth having: most targets have several cavities, so discarding the
badly-modelled ones usually costs a cavity, not the target. The gap closes as the criterion
tightens -- at pLDDT>=90 / frac>=1.00 you keep 2.5% of pockets and 15.5% of pairs, which is a
different (and much worse) regime.

**Negative sampling never breaks.** Across all 25 cells the maximum per-entity deficit is **0**,
on both the drug and target axes, and `balance.ok` is true in every report. The negative pool
(13,600 accessions with cavities against 1,294 study targets) is large enough that quality
filtering never starves it. Positive and negative job counts stay exactly equal in every cell.
Whatever the filter costs, it does not cost balance.

**Where the confidence gain actually happens.** The unfiltered dataset docks 16.4% of its
positive jobs into a pocket whose mean pLDDT is below 70, and 3.2% into one below 50. Those are
the jobs whose RMSD agreement is measuring the modelling error rather than the chemistry.
The gain is nearly all captured early:

| criterion | pairs kept | jobs kept | docked-pocket mean pLDDT | % jobs into a pocket<70 | % into a pocket<50 |
|---|---|---|---|---|---|
| none | 100.0% | 100.0% | 80.5 | 16.4% | 3.2% |
| pLDDT>=50, frac>=0.70 | 98.6% | 86.3% | 84.0 | 6.3% | 0.0% |
| pLDDT>=50, frac>=0.80 | 95.1% | 70.8% | 86.4 | 2.0% | 0.0% |
| pLDDT>=70, frac>=0.70 | 93.0% | 70.2% | 86.9 | 0.0% | 0.0% |
| pLDDT>=70, frac>=0.80 | 83.7% | 51.2% | 90.0 | 0.0% | 0.0% |
| pLDDT>=80, frac>=0.80 | 77.9% | 41.3% | 91.8 | 0.0% | 0.0% |
| pLDDT>=90, frac>=0.80 | 50.6% | 21.8% | 94.6 | 0.0% | 0.0% |

By pLDDT>=70 / frac>=0.70 there is **not one** planned job left whose pocket averages below the
confident band. Everything past that row buys pLDDT points inside a range that was already
confident, and pays for them in pairs.

## 5. Recommendation

**Use `--plddt-min 70 --min-plddt-fraction 0.70`.**

The trade-off in numbers, against the unfiltered baseline:

* **Cost:** 381 of 5,460 drug-target pairs (7.0%), 126 of 1,294 targets (9.7%), 26 of 1,125
  drugs (2.3%). The study keeps **5,079 pairs across 1,168 targets and 1,099 drugs**.
* **Gain:** every remaining positive job docks into a pocket that averages at least the confident
  band -- the 16.4% of jobs below mean pLDDT 70, and the 3.2% below 50, are gone entirely. Mean
  docked-pocket pLDDT rises 80.5 -> 86.9, the 25th percentile 74.0 -> 80.7.
* **Bonus:** positive jobs fall 18,842 -> 13,222 and negatives with them, so the campaign is
  **~30% cheaper in GPU/CPU time** while losing 7% of its scientific scope. Given a full
  `batch_dock.py` pass runs for days, that is a real saving, not a rounding error.
* **No balance cost:** zero deficit on both axes, 13,222 positives and 13,222 negatives.

Two honest caveats.

**Past the point where `% jobs pocket<70` hits zero, the filter costs more than it buys.** Going from
pLDDT>=70/0.70 to pLDDT>=70/0.80 costs a further 508 pairs (93.0% -> 83.7% of baseline) and
3,582 jobs to move the mean docked-pocket pLDDT from 86.9 to 90.0 -- a gain entirely inside a
band that is already trustworthy, since `% jobs pocket<70` is already 0.0 at 0.70. The
pLDDT>=90 column is worse still: at frac>=0.80 it discards half the pairs and 60.7% of the
targets for 7.7 pLDDT points nobody needs. For a *selectivity* study, targets are the axis that
matters, and pLDDT>=90 destroys them fastest -- at frac>=1.00 it keeps 15.5% of pairs but only
12.4% of targets, which is no longer the same experiment.

**If 7% of pairs is too much to give up, `--plddt-min 50 --min-plddt-fraction 0.80` is the
frugal alternative:** it keeps 95.1% of pairs and 92.9% of targets, still eliminates every
sub-50 pocket, and leaves only 2.0% of jobs below mean pLDDT 70 -- at almost exactly the same
compute (13,338 vs 13,222 positive jobs). It is the better choice if pair coverage is the
binding constraint; pLDDT>=70/0.70 is the better choice if pocket confidence is.

**What not to do:** running unfiltered. 3.2% of positive jobs -- roughly 600 of them, and 600
matched negatives -- dock into a pocket whose residues average below pLDDT 50, where the backbone
geometry is not meaningfully determined. Cross-tool RMSD agreement on those is noise being scored
as signal, and it costs only 1.4% of pairs (pLDDT>=50/frac>=0.70) to remove all of them.

## 6. Files

* `manifests/cavity_quality.csv` -- 38,709 rows, per-residue pLDDT retained; any cutoff is
  re-evaluable from it in seconds.
* `manifests/cavity_quality_rejected.csv` -- header only; no cavity failed to score.
* `cells/<cell>/positives_planned.csv`, `negatives_drawn.csv`, `positives_rejected.csv`,
  `negatives_drawn.report.json`, `negatives_drawn.deficits.csv` -- one folder per grid cell,
  plus each stage's own console log.
* `cells/baseline/` -- the unfiltered run, with no quality flags passed at all.
