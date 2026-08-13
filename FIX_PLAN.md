# FIX_PLAN — execution-ready global repository fix

**Status:** pass 3 (2026-08-13) — **§1 IMPLEMENTED** (identity module, invariants, checked joins,
schemas, jobid codec, index builder, and the four new stage scripts). §2 and every §3 edit to an
**existing** pipeline script remain **outstanding**; §5's support-module list is now largely done and
its remaining items are named below. `python3 -m unittest discover -s tests -t .` → **605 tests, OK**
(no failures, no skips), including an end-to-end test that chains Phase 0 → 5 of `REBUILD_PLAN.md`.
The suite was still growing while this pass was written; re-run it rather than trusting the count.
**Operational counterpart:** `REBUILD_PLAN.md` — which of these changes must land before each phase of
the clean-slate rebuild, and the manual runbook for executing it in a new experiment directory.
**Source:** the *Global Repository Fix Plan* section of `ASSESSMENT_blast_radius_2026_08_12.md`
**Rule for this document:** nothing is recorded here until its `file:line` citation has been read
against the working tree and the change is specified to the point where it could be handed to someone
who has not read the assessment.

Legend — **V** verified against code · **S** specified (signature + edit site + invariant + acceptance
criterion + invalidation) · **I** implemented and tested in the working tree · **○** not yet done.

---

## Implementation ledger — what exists in the tree as of 2026-08-13

Everything below is **new code**. **No existing pipeline script has been modified.** The strategy that
emerged: rather than patch the old scripts in place for the identity fix, the rebuild path is carried by
new stage scripts that consume and produce declared schemas, and the old scripts are edited later (or
retired) on their own schedule.

| Item | Where | Status |
|---|---|:--:|
| §1 `IdentityIndex` / `Protein` / `Resolution` | `dock_selectivity/identity.py` | **I** |
| §1 index builder, G1–G5/G7 asserted at build, sole network consumer | `build_identity_index.py` | **I** |
| §5 `invariants.py` — collapse-factor, total-accounting, no-row-growth, determinism, self-consistent-labels, unique-key | `dock_selectivity/invariants.py` | **I** |
| §5 `checked_merge()` | `dock_selectivity/joins.py` | **I** |
| §5 `schemas.py` — the seven manifests + `read_checked`/`write_checked` | `dock_selectivity/schemas.py` | **I** |
| §5 `jobid.py` — `__`-delimited codec, `job_identity.json` I/O, right-anchored legacy parser | `dock_selectivity/jobid.py` | **I** |
| §3.1 identity half — Entrez-only resolution of the edge list | **new** `resolve_interactions.py` | **I** (the edit to `identify_required_structures.py` itself is still **○**) |
| §3.2 balanced sampler with an accession-space avoid-list | **new** `sample_negatives.py` | **I** (`generate_negative_samples.py` untouched, **○**) |
| §2 relabel path / job classification | **new** `salvage_scan.py` | **I** |
| §3.3 new naming scheme + `job_identity.json` writer (curation side) | **new** `curate_jobs.py` | **I** (`batch_dock.py` itself is still **○**) |
| §5 read-only auditor (test tier 4) | **new** `audit.py` | **I** |
| §2 delete `prepare_uniprot_mapping.py` + G6 grep lint | — | **○** |
| §3.1 `identify_required_structures.py` | — | **○** |
| §3.2 `generate_negative_samples.py` | — | **○** |
| §3.3 `batch_dock.py` (`:2034`, `:2211`, post-job cleanup hook) | — | **○** |
| §3.4 `parse_prepare.py` (all five defects) | — | **○** |
| §3.5 `extract_cavities.py` — `--manifest-only` (names, no unpacking) + mandatory `--required` | `extract_cavities.py` | **I** |
| §3.6 `extract_alphafold_models.py` | — | **○** |
| §3.7 `create_ic50_mapping.py` (coverage lift only; not corrupt) | — | **○** |
| §3.8 `run_full_workflow.py` | — | **○** |
| `positives_planned.csv` producer | **new** `plan_positives.py` | **I** |
| `run_manifest.csv` assembler + the complete redock queue (incl. the previously unqueued **positive** gap) | **new** `build_run_manifest.py` | **I** |

**Consequence for the rebuild:** `REBUILD_PLAN.md` Phases 0–5 are carried by implemented code; **Phases
6, 7 and 8 are blocked purely on the ○ rows above** — `extract_*.py`, `batch_dock.py` and
`parse_prepare.py` have **zero `add_argument` calls between them** and cannot accept the `--required` /
`--queue` / `--run-dir` arguments those phases pass.

---

## 0. Verification ledger

Every citation below was read at `cad1ccb` during pass 1. All twelve check out; none needed correction.

| # | Citation | Claim | V |
|---|---|---|:--:|
| 1 | `prepare_uniprot_mapping.py:51` | `unique_gene_names = list(set(gene_names_list))` — hash-randomised order | ✅ |
| 2 | `prepare_uniprot_mapping.py:130` | `pd.concat(dfs).drop_duplicates(subset=['Entry'])` — first-wins over that random order | ✅ |
| 3 | `batch_dock.py:2034` | `dict(zip(uniprot_map_df['Mapped_Gene_Name'], uniprot_map_df['Entry']))` — last-wins collapse | ✅ |
| 4 | `batch_dock.py:2211` | `f"{drugbank_id}_{gene_name}_{uniprot_id}_{pocket_suffix}"` — identity encoded as a folder string | ✅ |
| 5 | `identify_required_structures.py:170` | `uniprot_id = gene_to_uniprot.get(gene)` then `if uniprot_id:` / else count-and-drop | ✅ |
| 6 | `identify_required_structures.py:196` | `{v: k for k, v in gene_to_uniprot.items()}` — inverting a many-to-one map | ✅ |
| 7 | `generate_negative_samples.py:190` | `gene_to_uniprot.get(gene_name)` then `if not uniprot_id: continue` | ✅ |
| 8 | `parse_prepare.py:146` | `max_files_per_pattern = 10000`, enforced at `:99`, hard `break` at `:104` | ✅ |
| 9 | `parse_prepare.py:359` | `r'(DB\d+)_([A-Za-z0-9]+)_([A-Z0-9]+)_cavity_(\d+)'` — rejects 1,777 dirs | ✅ |
| 10 | `parse_prepare.py:750` | `tools = ['GOLD', 'SMINA', 'LeDock','GNINA']` — `'LeDock'` ≠ data `'LEDOCK'` | ✅ |
| 11 | `parse_prepare.py:1335` | `expected_tools = ['GOLD', 'Smina', 'LeDock','Gnina']` — 3 of 4 unmatchable | ✅ |
| 12 | `parse_prepare.py:1171` | `.merge(..., on=['uniprot_id','drugbank_id','cavity_index'])` — `Fragment_ID` omitted | ✅ |
| 13 | `extract_cavities.py:65` | `tar_files = [f for f in os.listdir(...)]` — iterates supply, no required-set input | ✅ |
| 14 | `extract_alphafold_models.py:194,214` | `load_required_structures()` degrades to `cavity_mapping.csv` on 3 triggers | ✅ ¹ |
| 15 | `extract_alphafold_models.py:249` | `structures_df[['UniProt_ID','Fragment_ID']].drop_duplicates()` — correct given its input | ✅ |
| 16 | `create_ic50_mapping.py:84` | `dict(zip(df['Entry Name'], df['Entry']))` — clean bijection, not corrupted | ✅ |
| 17 | `run_full_workflow.py:95..167` | seven `run_script()` calls; `generate_negative_samples.py` absent | ✅ ² |
| 18 | `batch_dock.py:34-47` | seven modules imported twice | ✅ |
| 19 | `pocketmatch_clustering.py:426` | stray `tmux` after a string literal; file fails `ast.parse` | ✅ |
| 20 | `fix_pdb_files.py` | 0 bytes | ✅ |

¹ **Correction to the draft plan:** the fallback is *not* silent — `:230` emits a `logging.warning`. It is
a warning-level degradation of the scientific universe, which in a multi-thousand-line log is
operationally equivalent to silence but must be described accurately.

² **New finding F-C**, recorded in §3.8.

### Two findings promoted during verification

**F-A — `filter_complete_tool_coverage` is a silent no-op at HEAD.** At `:1335` the expected-tool
literals are `'Smina'`, `'LeDock'`, `'Gnina'`; the data carries `'SMINA'`, `'LEDOCK'`, `'GNINA'`. Only
`'GOLD'` can ever match, so `len(tools_present)` maxes at 1, never reaches `len(expected_tools) == 4`,
`complete_pairs` stays empty, the function prints `⚠️ No pairs with complete tool coverage found` and
**returns the frame unfiltered**. The step is listed in `processing_steps_applied` in every
`data_preparation_summary_*.json`. It has never filtered anything.

This also removes the memory rationale for the phase ordering in `main()`: the filter that was placed
before the annotations specifically to shrink the frame does not shrink it.

**F-B — HEAD is not the code that produced the artifacts.** At `:750` the literal `'LeDock'` cannot
match `'LEDOCK'`, so `LeDock_Score` would be entirely null. In
`combined_filtered_annotated_docking_results.parquet` the four `*_Score` columns are populated with
non-null counts matching the per-tool totals exactly (GNINA 9,420,360 · SMINA 9,275,147 ·
LEDOCK 7,609,545 · GOLD 6,882,814). The artifact was therefore produced by a different revision than
the one in the tree. **No re-derivation may assume HEAD reproduces the existing parquet**, and the
first act of any rebuild is to record the revision that produced it.

---

## 1. Root fix — `IdentityIndex` **V/S/I — IMPLEMENTED**

### The law

| Axis | Canonical key | Resolved from | Never a key |
|---|---|---|---|
| Protein | UniProt accession | **Entrez Gene ID** (`node_2`; numeric on 8,568/8,568 edges, 1:1 with symbol) | gene symbol, entry name, `*_name` |
| Compound | DrugBank ID | DrugBank ID; name slug only via an ambiguity-aware resolver | drug name |
| Pocket | `(accession, fragment_id, cavity_index)` | filename parse | folder-name substring |

1. Resolve once, at the ingest edge. Downstream stages receive accessions and never re-resolve.
2. Labels flow the other way only: `accession → primary_symbol`, for display. A label is never an input to selection.
3. A many-to-one map may not be inverted (kills item 6).

### Edit site

New package `dock_selectivity/` beside the scripts, so `python $REPO/script.py` picks it up via
`sys.path[0]` with no install step. `identity.py` provides:

```python
@dataclass(frozen=True)
class Protein:
    accession: str; entry_name: str; primary_symbol: str
    symbols: tuple[str, ...]; entrez_id: int | None; reviewed: bool

@dataclass(frozen=True)
class Resolution:
    status: Literal['OK','AMBIGUOUS','UNKNOWN','OBSOLETE','NO_KEY']
    protein: Protein | None
    reason: str                      # always populated when status != 'OK'

class IdentityIndex:
    @classmethod
    def load(cls, path: Path) -> 'IdentityIndex': ...
    def by_entrez(self, gene_id: int) -> Protein         # raises KeyError
    def by_accession(self, acc: str) -> Protein          # raises KeyError
    def resolve_entrez(self, gene_id: int) -> Resolution # never raises; caller must branch
    def label(self, acc: str) -> str                     # accession -> primary symbol
    def by_symbol_ambiguous(self, symbol: str) -> tuple[Protein, ...]   # audit only
    def assert_self_consistent(self) -> None
    def cardinality_report(self) -> dict
```

`Resolution` is the load-bearing type: it makes `.get(gene)` + `if not x: continue` (items 5 and 7)
impossible to write without also deciding where the rejected row is recorded.

**As implemented** (`dock_selectivity/identity.py`), with Python 3.9 syntax (`Optional`/`Tuple`, no
`X | Y` unions) and four additions to the sketch above:

- `Protein.has_symbol(symbol)` — the `(accession, gene)` pair-validity check, i.e. the `KNOWN_ISSUE`
  regression assertion as a method.
- `Resolution` carries `key` (the echoed input, so a reject row identifies itself) and `candidates`
  (the accessions, when `AMBIGUOUS`). Both default, so positional construction per the sketch still
  works. `__post_init__` rejects the incoherent states: non-`OK` without a reason, non-`OK` carrying a
  protein, `OK` without one. `as_reject_row()` emits the `unresolved_targets.csv` row.
- `IdentityIndex.load(path, expect_sha256=...)` implements the pin; `file_sha256()` is exported for the
  builder and the auditor. `from_frame`/`to_frame` give a canonical, order-stable frame.
- `by_entrez` raises **`InvariantViolation`, not `KeyError`**, when an Entrez id carries more than one
  accession — deliberately, so a caller's `except KeyError: continue` cannot swallow a broken index.
  `resolve_entrez` reports the same situation as `status='AMBIGUOUS'` with `candidates` populated.

`OBSOLETE` needs input the parquet schema does not carry: pass `obsolete_entrez={id: reason}` to
`load`/`from_frame`. Without it a retired id resolves as `UNKNOWN` — still recorded with a reason, never
dropped.

**A defect found and closed during this pass.** `identity.py` originally declared its *own*
`class InvariantViolation(Exception)` instead of importing the canonical one from `invariants.py`, so
`identity.InvariantViolation is not invariants.InvariantViolation`: a stage catching only one let the
other escape as a traceback. It is **fixed** — `identity.py:81` now reads
`from .invariants import InvariantViolation`, verified
(`identity.InvariantViolation is invariants.InvariantViolation` → `True`), and the self-clearing test
`tests/test_invariants.py::TestExceptionHierarchy::test_identity_module_shares_the_canonical_exception`
now asserts rather than skips.

Residual, harmless: three stage scripts still carry a two-class `except` tuple written while the split
existed — `resolve_interactions.INVARIANT_ERRORS`, `salvage_scan.FATAL_ERRORS`,
`sample_negatives.IDENTITY_ERRORS`. They are now redundant, not wrong, and each has a test that keeps
passing either way. Collapse them opportunistically; **`invariants.py` is the single canonical home**,
and any new stage should import from there and nowhere else.

### Invariants established (asserted at index build, not aspirations)

| ID | Guarantee | Enforcement | Status |
|---|---|---|:--:|
| G1 | One row per accession | `assert_unique_key(frame, ['accession'])` at build | **I** |
| G2 | No silent drops: `n_in == n_resolved + n_rejected`, rejects written to `unresolved_targets.csv` with a reason | `assert_total_accounting` on **both** axes; `n_in` counted from the *source*, never from the output | **I** |
| G3 | Deterministic — no `set()` iteration, no insertion-order dependence, no `drop_duplicates` over an unordered concat; all tie-breaks total (reviewed > unreviewed → in-AF-proteome > not → lowest accession) | `assert_deterministic` in-process; `--check-determinism` re-runs the whole build under `PYTHONHASHSEED` 0/1/42 | **I** |
| G4 | `primary_symbol ∈ symbols`; `entrez(accession(e)) == e` | `assert_self_consistent_labels` against the **raw source symbols** (independent of the frame) + an explicit round-trip | **I** |
| G5 | `entrez → accession` collapse factor **exactly 1** | `assert_collapse_factor_one` on the *pairs*, before any dict is built | **I** |
| G6 | No free-text identifier search anywhere in the codebase | grep lint in CI | **○** — the new code issues no search and imports no `bioservices`, but the lint itself does not exist |
| G7 | Isoform accessions (`P12345-2`) and non-human excluded | at parse *and* at assembly; plus `IdentityIndex.assert_self_consistent` | **I** |
| G8 | No `except` / `if not exists` branch may widen a stage's input set | per-stage; a missing prerequisite raises | **I** in every new script |

G5 is the specific guarantee whose absence caused the incident: the old map had mean collapse 2.55, max 9.

**Two build-time subtleties worth carrying forward.** (a) An Entrez collision does **not** fail the build
by default — it is *resolved* by the documented total tie-break and every demotion is recorded;
`--on-entrez-collision fail` is the strict mode, and it would refuse every real build, because UniProt
genuinely contains TrEMBL twins of Swiss-Prot entries. (b) The two ledgers are separate on purpose:
`unresolved_targets.csv` lists only what is **not** in the index (so `n_rejected == len(that file)` is
satisfiable), while decisions about rows that **are** in it — an accession demoted to a null
`entrez_id`, an orphaned GeneID, an accession-fallback label — go to `identity_index.notes.csv`.

### Acceptance criteria — all met

- `AVPR2 → P30518` (not Q8NCT1); `TUBA1B → P68363` (not Q13509); `IL2 → P60568` (not P01589). ✅ verified by `audit.py identity`, which also asserts the *impostors no longer claim those symbols*.
- `cardinality_report()['max_collapse'] == 1`. ✅ — note the auditor asserts `<= 1`, because a `REDUCED_LOCAL` index carries no Entrez ids at all and reports 0.
- Two builds under different `PYTHONHASHSEED` produce byte-identical `identity_index.parquet`. ✅ (`--check-determinism`; the unit tests assert it directly).
- The regression assertion from `KNOWN_ISSUE.md` passes on the emitted index. ✅ — and is re-applied downstream to `interactions_resolved.csv`, `positives_planned.csv`, `negatives_pool.csv` and every curated job directory.

### Inputs to obtain (one manual fetch, checksummed, then offline forever)

**Re-checked 2026-08-13: none of the three authoritative inputs is present locally.** `/opt/data`
contains `alphafold2/`, `cavity_space/`, `drugbank/`, `experiments/`, `multiscale_interactome_data/`,
`ttd/` — no id-mapping file, no proteome FASTA, no HGNC table. **The fetch is still the operator's one
manual step, and it is the single thing blocking `REBUILD_PLAN` Phase 1.**

| File | Source | Provides | Status |
|---|---|---|---|
| `HUMAN_9606_idmapping.dat.gz` | UniProt FTP `knowledgebase/idmapping/by_organism/` | accession ↔ GeneID, Gene_Name, Gene_Synonym | **required — ABSENT** |
| `UP000005640_9606.fasta.gz` | UniProt reference proteomes | review status, canonical one-per-gene set | **required** (tie-break) — **ABSENT** |
| `hgnc_complete_set.txt` | HGNC | prev/alias symbols, withdrawn detection, cross-check | recommended — **ABSENT** |
| `/opt/data/alphafold2/UP000005640_9606_HUMAN_v4.tar` | already local | dockable-accession universe | present (structures, **not** the proteome FASTA) |

`build_identity_index.py --fetch` downloads them (URLs are in `SOURCE_SPECS`), or they can be fetched by
hand and passed with `--idmapping` / `--proteome` / `--hgnc`; an explicitly supplied path is never
fetched over, and a cached file is reused unless `--refresh`. A **required** input that cannot be
downloaded is fatal; the declared-optional HGNC is a recorded note, because it contributes only
synonyms, a tie-break hint and a cross-check and cannot add or remove an accession.

`build_identity_index.py` is the **sole network consumer**, run manually, refusing to re-fetch without
`--refresh`. It emits `identity_index.parquet` + `identity_index.meta.json` (source URLs, UniProt
release, per-file SHA256, row counts, accounting, invariant report — and no build timestamp inside the
parquet, so the artifact stays byte-reproducible). Provenance is stamped **inside the parquet itself**,
not only in the sidecar, because downstream code calls `IdentityIndex.load(path)` and may never open the
meta; `read_index_provenance(path)` returning `None` means *untrusted*, not authoritative. Every
downstream entrypoint pins the SHA256 and refuses to run against an unexpected index.

#### `REDUCED_LOCAL` — the offline path, and exactly what it cannot do

`--from-legacy-gene-names $OLD/uniprot_gene_mapping.csv` builds an index from the **trustworthy**
`Gene Names` column of the legacy file. The corrupt `Mapped_Gene_Name` column is not in the loader's
column list, so it is never read. **Verified end-to-end on the real file (2026-08-13): 4,585 records,
0 rejects, 2 recorded notes, `max_collapse = 0`, and `audit.py identity` reports 12/12 PASS, exit 0.**
It is stamped `REDUCED_LOCAL` in the parquet so it cannot be mistaken for the real thing.

Three limitations, the first of which is decisive:

1. **It carries no Entrez ids** (0 of 4,585) unless `--hgnc` is supplied — and reduced mode never
   fetches, so HGNC must be downloaded by hand. Since §3.1's whole point is that resolution runs on
   Entrez, `resolve_interactions.py` against such an index resolves **0 of 8,568 edges** (measured; all
   `UNKNOWN_ENTREZ`) **and still exits 0 with an empty manifest**. Any stage consuming that manifest
   would treat the entire campaign as non-interactions. Assert a non-empty output, not an exit code.
2. **The universe is capped at the 4,585 accessions the corrupt free-text search happened to return.**
   Per `KNOWN_ISSUE`, that query list was the drug-target gene set, so no other protein is reachable
   through it; per `ASSESSMENT` §1c only 1,567/1,958 genes (80.03%) are repairable from this file, 391
   need external re-derivation and 12 are absent entirely. It is therefore adequate for re-deriving the
   salvage arithmetic (which was computed in exactly this universe) and **inadequate for choosing the
   canonical accession of a new docking job** — i.e. for the entire re-dock gap.
3. Without `--proteome`, `reviewed` is `False` on every row and means *unknown*. It is recorded as such
   in `meta["row_counts"]["review_status_source"]`; do not read that column as a Swiss-Prot claim.

163 of the 4,585 legacy rows have an empty `Gene Names`; where no source names the gene, `primary_symbol`
falls back to the accession itself and the row is recorded in `identity_index.notes.csv` (2 rows on the
real file: P0DOX5, Q96FB2). A blank label is what lets a join match the wrong thing, so it is never left
blank.

**Not verified against the real UniProt files.** The idmapping/proteome/HGNC parsers are exercised only
against synthetic fixtures of the documented formats. Run the first authoritative build **without**
`--strict-cross-check`, read `meta["invariants"]`, then decide whether to make the `GN=` cross-check
fatal.

### Invalidates

`uniprot_gene_mapping.csv` (retired), `required_structures.csv`, `required_structures_with_negatives.csv`,
and every artifact derived from them. Does **not** invalidate any docking output — job identity is
recovered from the accession in the folder name, which is physical truth.

---

## 2. `prepare_uniprot_mapping.py` — retire, do not patch **V/S ○ — NOT DONE**

Three independent defects, none patchable in place, because the *direction* of resolution is wrong:

1. `:51` `list(set(...))` — hash-randomised query order (non-reproducible output).
2. `:130` `drop_duplicates(subset=['Entry'])` — first-wins over that random order; destroyed 1,992 rows and stripped 370 genes of their correct accession.
3. The free-text `service.get_df(gene_name)` search itself, whose result is broadcast to every returned row.

**Action:** delete the file; `build_identity_index.py` replaces it. Keep the file's docstring history in
the commit message. Add the G6 grep lint so a free-text resolver cannot reappear.

**Acceptance:** `grep -rn "get_df\|bioservices" *.py` returns nothing outside tests.

---

## 3. Item-by-item change list

Specified in pass 1: items 3, 5, 6, 7 (identity call sites), 8, 9, 10, 11, 12 (parse_prepare defects).
**Pass 3 note:** for the clean-slate rebuild, 3.1's identity half, 3.2 and 3.3's curation half are
delivered as **new scripts** rather than as edits to the old ones (see the implementation ledger). The
edits below remain owed by the old scripts themselves, which are still on the legacy path.

### 3.1 `identify_required_structures.py` **V/S ○** — identity half delivered as `resolve_interactions.py` **I**

- **`:170`** — replace `gene_to_uniprot.get(gene)` with `index.resolve_entrez(row.node_2)`; branch on
  `Resolution.status`; append every non-`OK` to `unresolved_targets.csv` with its reason. Iterate the
  interaction edges (demand), never `cavity_mapping.csv` (supply).
- **`:196`** — delete the dict inversion entirely. `Gene_Name` becomes `index.label(accession)`.
- **Invariant:** `n_edges_in == n_resolved + n_rejected`, logged and asserted.
- **Acceptance:** re-running produces a `required_structures.csv` in which every `(UniProt_ID, Gene_Name)`
  pair satisfies `Gene_Name ∈ index.by_accession(UniProt_ID).symbols` — currently false for 65.11% of rows.
- **Invalidates:** `required_structures.csv` and everything downstream of it.

**Delivered instead (new file, old script untouched):** `resolve_interactions.py` emits
`manifests/interactions_resolved.csv` (schema `interactions_resolved`) plus a rejects ledger with an
`axis` column, resolving the protein axis **only** through `resolve_entrez`. Three things it added
beyond the spec, all of which the old script would also need:

- **The compound axis, which was never verified at all.** 222 of the 8,568 `node_1` values are name
  slugs. A `CompoundIndex` resolves them by *normalised exact* matching against DrugBank's own
  vocabulary and returns `AMBIGUOUS` — never a pick — when a name belongs to more than one entry
  (`L-Arginine` is DB17289's common name *and* DB00125's synonym). Name tiers are **pooled, not ranked**;
  `--prefer-common-name` opts into ranking explicitly.
- **Accounting has three terms, not two.** `n_in == n_resolved + n_rejected` **and**
  `n_output_rows == n_resolved − n_duplicate_edges_collapsed`; 38 real edges name the same compound twice
  (`DB00155` and `L-Citrulline`) and collapse onto the declared primary key, with the survivor's
  `selected_by` naming every contributing edge row.
- **`NOT_SMALL_MOLECULE` is a reject, not a flag** — 554 edges / 156 DrugBank IDs. `--drugs` is required
  and must carry `Drug Type`, so a plain id list cannot silently widen the compound universe.

**Still owed by this item:** the *structure-planning* half — `positives_planned.csv`, the demand-side
expansion of the resolved edges over cavities and ligand availability. See §5.

### 3.2 `generate_negative_samples.py` **V/S ○** — delivered as `sample_negatives.py` **I**

- **`:190`** — same `Resolution` conversion as above.
- **`:303-319`** — the known-interaction avoid-list must be built in accession space from Entrez, not from
  `gene_to_uniprot[node_2_name]`. Only 50.11% of the current avoid-list entries point at an accession that
  really carries that gene.
- **Invariant:** for every emitted negative, `(drug, accession)` ∉ avoid-list, where the avoid-list is
  provably complete over the resolved edge set.
- **Acceptance:** contamination check reports 0 negatives that are known interactions; balance holds in
  *accession* space, not label space.
- **Invalidates:** `required_structures_with_negatives.csv`.

**Delivered instead (new file, old script untouched):** `sample_negatives.py`. Four defects of the old
sampler are structurally absent rather than patched:

- **`:561-566` is gone.** That branch, once no target still needed negatives, drew from *any* available
  target — silently abandoning the target-side quota. It is the same universe-widening pattern G8 bans,
  and including it inflates apparent feasibility. There is no equivalent.
- **Eligibility is computed, not assumed.** A target (or drug) with zero corrected positives must receive
  zero negatives, so the pool is filtered to structurally eligible entries *before* drawing, with every
  exclusion written out with its reason. At real scale only **8,063 of 16,873** pool jobs (47.8%) are
  eligible — see `REBUILD_PLAN.md` §1.2.
- **No duplicate pockets.** `np.random.choice` over cavities could emit the same job twice; a pair is now
  handed each pocket at most once, so the primary key is unique by construction.
- **The balance unit is explicit** (`--balance-unit job|pair`) and recorded in the report and in every
  `selected_by`. The old code balanced cavity rows, which produced 3.38 cavities/pair on positives
  against 1.36 on negatives (`REBUILD_PLAN.md` §1.2.4).

The pocket universe is `positives_planned ∪ pool` per accession — deliberately no `cavity_mapping.csv`
input and no "every cavity on disk" fallback (G8).

### 3.3 `batch_dock.py` **V/S ○** — curation half delivered as `curate_jobs.py` **I**

- **`:2034`** — delete the `dict(zip(...))` fallback resolver. `batch_dock.py` must consume a plan that
  already carries accessions and must never resolve identity itself.
- **`:2211`** — stop encoding identity in a folder-name string. Keep the directory name as a
  *human-readable convenience* but write the authoritative identity to `job_identity.json` inside each
  job directory: `{drugbank_id, accession, fragment_id, cavity_index, index_sha256, code_revision}`.
  This is the structural fix for the load-bearing-string debt and for both parse failures (items 9, F-B).
- **Invariant:** every job directory contains a `job_identity.json` whose accession matches the receptor
  file actually passed to the engine.
- **Acceptance:** an auditor can reconstruct 100% of job identity without parsing a directory name.
- **Invalidates:** nothing existing — new jobs only. Existing jobs get `job_identity.json` backfilled by
  the relabel path (assessment §2), which is where the 24,173 completed jobs are rescued.

**Delivered for the salvage path (new files, `batch_dock.py` untouched):** `dock_selectivity/jobid.py`
is the codec — `format_job_dir` / `parse_job_dir` / `parse_legacy_job_dir` /
`write_job_identity` / `read_job_identity` — and `curate_jobs.py` writes the curated tree. Decisions
`batch_dock.py` must match when §3.3 finally lands:

- Name is `DB00035__P30518__AVPR2__F1__cav2`; the null-gene sentinel is `-`, **never** `nan`. The old
  blank-gene path produced 10,698 `*_nan_*` directories.
- `fragment_id` is **required** to build a name, and `jobid.py` refuses to infer it. The "missing token
  means F1" inference is a caller's explicit, logged, per-row-recorded choice
  (`curate_jobs.py --assume-fragment`, `salvage_scan.py --default-fragment`). An *inferred* fragment is
  the class of bug that caused the 410,087-row fan-out.
- `parse_job_dir` will **not** fall back to `parse_legacy_job_dir` (or vice versa), and
  `read_job_identity` will **not** fall back to parsing the directory name. Callers choose the scheme.
- `job_identity.json` carries no timestamp, so a curated tree is byte-identical across runs.
- A new-name collision fails the run by default. Two old directories can claim one new name when they
  differ only in the gene token (`DB00035_AVPR2_Q8NCT1_cavity_2` vs `DB00035_ARRDC4_Q8NCT1_cavity_2`);
  picking one would be the last-wins collapse this whole plan exists to eliminate.

**Still owed by this item:** everything on the *docking* side — deleting the `:2034` resolver, consuming
a plan that already carries accessions, writing `job_identity.json` at job creation, an actual CLI, and
the post-job hook that discards or `xz`-compresses `complex_*.pdb` once `final_results.csv` exists.
`batch_dock.py` has **no `argparse` at all** today, which is why `REBUILD_PLAN` Phase 7 cannot run.

### 3.4 `parse_prepare.py` **V/S ○**

| Site | Change | Invariant | Acceptance |
|---|---|---|---|
| `:146` | Remove the cap; discovery must be exhaustive or fail loudly. Replace `max_files_per_pattern` with an assertion that the number of discovered job dirs equals the number in the run manifest. | Discovery is total | All 24,173 dirs present; recovers 14,173 jobs / 845 GB / 8,732 core-h |
| `:359` | Delete the regex. Read `job_identity.json`; for legacy jobs use the right-anchored parse (strip trailing `_cavity_<N>`, strip optional `_-F<k>-model_v1`, test tokens against the accession set) which resolves 24,173/24,173 | Identity never derived from a fragile pattern | 0 rows with null `drugbank_id`/`uniprot_id` (currently 1,174,786) |
| `:750` | Replace tool literals with a single normalising `canonical_tool()` used everywhere | One tool-name vocabulary | `LeDock_Score` non-null count equals the LEDOCK row count |
| `:1335` | Same `canonical_tool()`; assert `complete_pairs` is non-empty or fail | A no-op step cannot report success | Filter demonstrably filters; `processing_steps_applied` reflects reality |
| `:1171` | Add `Fragment_ID` to the merge key (normalise `'1'` vs `'F1'` first) **or** derive the label from the identity index and drop the merge | A merge may not change row count unintentionally | `checked_merge()` asserts `len(out) == len(left)`; eliminates the 410,087 duplicated rows |

**Invalidates:** `combined_filtered_annotated_docking_results.parquet` — rebuild required. Note F-B: do
not expect the rebuild to reproduce the current file.

**Tools now available for these five edits, so none of them needs new machinery:**
`dock_selectivity.joins.checked_merge()` for `:1171`; `dock_selectivity.jobid.parse_job_dir` /
`parse_legacy_job_dir` for `:359`; `dock_selectivity.schemas.read_checked` against the `run_manifest`
schema for `:146`; `invariants.assert_no_row_growth` and `assert_unique_key` for the rest. **Do not call
`checked_merge` inside a per-job loop** — it costs roughly 2× a bare merge (measured 3.9 s vs 1.8 s on
2M rows × 4 key columns), which is fine per stage and not fine per job. Likewise
`assert_self_consistent_labels` iterates rows in Python: correct at 8k rows, wrong at 16.5M.

---

### 3.5 `extract_cavities.py` — the 76.6 GB over-extraction **V/S ○**

**Verified:** `:34` `CAVITY_TARBALL_FOLDER = "/opt/data/cavity_space/af_strong_cavities_data_all"`;
`:65` `tar_files = [f for f in os.listdir(cavity_tarball_folder) if f.endswith(".tar.gz")]`.
The loop variable is the *supply* (every tarball on disk, 38,709 of them). There is no "required" input
and no way to pass one. 2,829 were ever needed — 7.3%.

The ordering that forces this: `identify_required_structures.py` consumes `cavity_mapping.csv`, so the
mapping must exist before the required set can be computed. But the mapping needs only *filenames*.

- **Edit site:** split the function in two.
  `build_cavity_manifest(tarball_dir) -> DataFrame` — derives `(UniProt_ID, Fragment_ID, Cavity_Index,
  Pocket_PDB)` from tarball **names and member listings only** (`tarfile.getnames()`, no `extract`).
  `extract_cavities(manifest, required_keys) -> DataFrame` — unpacks only rows whose
  `(accession, fragment_id, cavity_index)` is in `required_keys`, and **requires** that argument.
- **Invariant:** `extracted_count == len(required_keys)`. Extraction never runs over a set it was not given.
- **Acceptance:** a fresh run against the current study produces 2,829 cavity directories, not 38,709;
  `cavity_mapping.csv` is byte-identical to the manifest built without unpacking.
- **Invalidates:** nothing. Existing `extracted_cavities/` is a superset and stays usable.

### 3.6 `extract_alphafold_models.py` — the 5,661-vs-979 over-extraction **V/S ○**

**Verified:** `load_required_structures()` at `:194`. Correction to the draft plan: the fallback is
**not silent** — `:230` logs `logging.warning("Using all cavity structures - for efficiency, run
identify_required_structures.py first")`. It is a *warning-level degradation*, which is worse in
practice: the run continues over 13,600 accessions / 13,975 fragments instead of 979 / 1,005, and a
warning in a multi-thousand-line log is indistinguishable from noise.

Three distinct triggers all fall through to `cavity_mapping.csv`: the file is absent (`:196`), it exists
but lacks `UniProt_ID`/`Fragment_ID` (`:202`), or reading it raises (`:209`). Confirmed at `:249` the
iteration is `structures_df[['UniProt_ID','Fragment_ID']].drop_duplicates()` — correct given its input;
the defect is entirely in which input it accepts.

- **Edit site:** delete the fallback branch (`:214-235`). On any of the three triggers, `sys.exit(1)`
  with the actionable message already drafted at `:217-218`.
- **Invariant:** this script never chooses its own universe. Its input set is supplied or it does not run.
- **Acceptance:** with `required_structures.csv` renamed away, the script exits non-zero and extracts
  nothing (today it extracts 13,975 structures).
- **Invalidates:** nothing. `alphafold_structures/` is a superset and stays usable.

**Generalisation (this is the actual rule).** Items 3.5, 3.6 and F-C below are the same defect: a stage
that *silently substitutes a different scientific universe* when its preferred input is missing. Ban the
pattern — a missing prerequisite is an error, never a fallback. G8: no `except`/`if not exists` branch may
widen a stage's input set.

### 3.7 `create_ic50_mapping.py` — clean, keep **V/S ○**

**Verified `:84`:** `mapping = dict(zip(df['Entry Name'].to_list(), df['Entry'].to_list()))`. It uses the
`Entry Name → Entry` bijection (4,585 → 4,585, zero collisions) and never touches `Mapped_Gene_Name`.
`ic50_mapping.csv` is **not corrupted** and needs no rebuild for correctness.

- **Edit site:** replace the hand-built dict with `IdentityIndex`, purely to lift the coverage ceiling.
- **Invariant:** TTD targets that fail to resolve are written to a rejects file, never dropped silently.
- **Acceptance:** resolvable TTD targets rise from 1,672/3,685 (45.4%) toward the reference-proteome
  ceiling; the 118 entries with multi-value separators are parsed rather than discarded.
- **Invalidates:** nothing. Regeneration is additive — existing rows keep their values.

### 3.8 `run_full_workflow.py` **V/S ○**

**Verified — new finding F-C:** the runner invokes seven scripts (`:95, :103, :111, :119, :127, :135,
:167`). **`generate_negative_samples.py` is not among them.** A repo-wide grep finds it referenced only
in two `batch_dock.py` comments (`:71`, `:420`). So the automated workflow cannot produce the balanced
dataset the study depends on; whether negatives exist at all depends on someone having run that script
by hand, and `batch_dock.py` then silently switches docking mode on the presence of the resulting file.
Same class as 3.5/3.6: behaviour determined by file presence rather than by explicit intent.

- **Edit site:** add `build_identity_index.py` as step 0 (gate: fail closed if the index is absent or its
  SHA256 is unexpected); add `generate_negative_samples.py` as an explicit step with a
  `--positives-only` opt-out; make the docking mode an explicit flag rather than an inferred one.
- **Invariant:** the executed pipeline is fully determined by flags, not by which files happen to exist.
- **Acceptance:** the runner logs a single provenance record — index SHA256, code revision, mode — and
  `batch_dock.py` refuses to start if the requested mode disagrees with the files present.
- **Invalidates:** nothing directly; it is the gate that makes everything else reproducible.

---

## 4. Structural debt — verified **V**

| Item | Citation | Verified | Risk | Status |
|---|---|:--:|---|:--:|
| Job-folder name is a load-bearing data contract | `batch_dock.py:2211` writes, `parse_prepare.py:359` reads | ✅ | **Highest** — fixed by `job_identity.json` (§3.3) | partly: the codec and the curated tree exist; `batch_dock.py`/`parse_prepare.py` still on the old contract **○** |
| Universe-widening fallbacks | `extract_alphafold_models.py:214`, `extract_cavities.py:65`, `run_full_workflow.py` (F-C) | ✅ | High — fixed by G8 | **○** in the old scripts; every new script fails closed |
| `pocketmatch_clustering.py:426` | `help='...'tmux` — stray token; file **does not parse** | ✅ | Medium — the script cannot have run since that edit | **○** |
| `fix_pdb_files.py` | 0 bytes | ✅ | Low — delete; live script is `fix_required_pdbs.py` | **○** |
| Duplicated import block | `batch_dock.py:34-47` — `os, sys, pandas, subprocess, re, time, logging` imported twice | ✅ | Low (cosmetic, but a marker of unreviewed edits) | **○** |
| Module-level config constants, no CLI | every *old* pipeline script — `extract_cavities.py`, `extract_alphafold_models.py`, `fix_required_pdbs.py`, `convert_pdb_to_pdbqt.py`, `batch_dock.py`, `parse_prepare.py`, `identify_required_structures.py`, `generate_negative_samples.py` have **0 `add_argument` calls between them** | ✅ | Medium — `config.py`/`RunConfig` | **○**; every new script has a full CLI, pins the index by SHA256 and exits non-zero on a missing prerequisite |
| Absolute paths to one external drive | `parse_prepare.py:39`, `batch_dock.py:77`, `create_ic50_mapping.py:27-31` | ✅ | Medium | **○** |
| No dependency pinning | no lockfile in repo | ✅ | Medium — compounds F-B | **○** |
| **No `.gitignore`** | repo root | ✅ | Low — `__pycache__/` appears untracked after every test run | **○** (new) |

---

## 5. Remaining work ○

### 5.1 Done since pass 2 (was: "support modules")

`checked_merge()`, `invariants.py`, `schemas.py`, `jobid.py` are **implemented and tested**, and the
read-only auditor (test tier 4) is `audit.py`. **All five test tiers now exist** as 605 hermetic
`unittest` tests (tempfile fixtures only; no network, no external drive, no `/opt/data`) — tier 3, the
e2e smoke, is `tests/test_end_to_end.py`, which chains `build_identity_index` → `resolve_interactions`
→ `salvage_scan` → `sample_negatives` → `curate_jobs` → `audit.py run` over a synthetic universe built
from the incident's real identifiers. They cover the specific regression classes this incident demands:
last-wins collapse impossible, byte-identical output across `PYTHONHASHSEED`, a silent fallback raising
instead of widening, a merge not changing row count, a parse handling hyphenated symbols and legacy
multi-fragment names, and an unresolved input recorded with a reason rather than dropped.

`config.py` / `RunConfig` is **not** done and is the one support module still owed from that list.

### 5.2 New gaps found while implementing — these block the rebuild

1. **Nothing produces `manifests/positives_planned.csv`.** `sample_negatives.py` and `curate_jobs.py`
   both require it; without it Phases 3 and 5 cannot run. It is the demand-side expansion of
   `interactions_resolved.csv` over cavities and ligand availability — schema `positives_planned`, key
   `(drugbank_id, accession, fragment_id, cavity_index)`, `gene` non-null and `= index.label(accession)`,
   `selected_by` non-null. Expected size 5,287 pairs → **18,166 rows**. Either the §3.1 rewrite emits it,
   or a new `plan_positives.py` does. The same producer should emit the **positive half** of
   `redock_queue.csv` (10,866 jobs).
2. **Nothing assembles `manifests/run_manifest.csv`.** `curate_jobs.py --manifest-out` writes only the
   jobs it materialised (`provenance='salvaged_from'`); the `newly_docked` half has to be derived from
   `redock_queue.csv` and concatenated. `audit.py run` already asserts the union equals
   `positives_planned ∪ negatives_drawn`.
3. ~~`identity.py` must adopt the canonical `InvariantViolation`.~~ **Done during this pass** (§1). The
   three redundant two-class `except` tuples can be collapsed opportunistically.
4. **G6's grep lint does not exist.** No new code imports `bioservices` or issues a free-text search, but
   nothing prevents one reappearing. `grep -rn "get_df\|bioservices" *.py` returning nothing outside
   tests is the check; it needs a home.
5. **`schemas.py`'s `negatives_pool` docstring still quotes 16,868 / 11,796.** The correct figures are
   **16,873 / 11,801** (`REBUILD_PLAN.md` §1.1). Cosmetic — the schema itself is unaffected — but it is a
   number a reader will trust.

### 5.3 Still outstanding from pass 2

- Full sequencing with the artifact-invalidation matrix.
- The compound axis: 222 rows in `1_drug_to_protein.tsv` `node_1` are name slugs, not DrugBank IDs.
  **Partly closed** — `resolve_interactions.py` resolves 143 of them against the DrugBank vocabulary and
  rejects the rest with a reason; the 35 unresolved are salt/ester forms (`fludrocortisone-acetate`,
  `omeprazole-magnesium`, `haloperidol-decanoate`, `gentamycin`, `ftorafur`) that need a **curated alias
  table**, not a looser matcher. Every one is in the rejects ledger with its normalised form, which is the
  input to building that table. Note also `jobid.parse_legacy_job_dir` splits a legacy directory name on
  `_` and takes token[0] as the drug; a name-slug drug containing `_` would bleed into the gene label. No
  such directory exists in `$OLD`, but if the compound axis ever produces one, the parser needs a
  `known_drug_ids` longest-prefix match.
