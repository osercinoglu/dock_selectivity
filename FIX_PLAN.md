# FIX_PLAN — execution-ready global repository fix

**Status:** pass 2 of N (self-paced hardening loop) — §1–4 specified, §5 outstanding
**Operational counterpart:** `REBUILD_PLAN.md` — which of these changes must land before each phase of
the clean-slate rebuild, and the manual runbook for executing it in a new experiment directory.
**Source:** the *Global Repository Fix Plan* section of `ASSESSMENT_blast_radius_2026_08_12.md`
**Rule for this document:** nothing is recorded here until its `file:line` citation has been read
against the working tree and the change is specified to the point where it could be handed to someone
who has not read the assessment.

Legend — **V** verified against code · **S** specified (signature + edit site + invariant + acceptance
criterion + invalidation) · **○** not yet done in this loop.

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

## 1. Root fix — `IdentityIndex` **V/S**

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

### Invariants established (asserted at index build, not aspirations)

| ID | Guarantee | Enforcement |
|---|---|---|
| G1 | One row per accession | assert at build |
| G2 | No silent drops: `n_in == n_resolved + n_rejected`, rejects written to `unresolved_targets.csv` with a reason | assert at build |
| G3 | Deterministic — no `set()` iteration, no insertion-order dependence, no `drop_duplicates` over an unordered concat; all tie-breaks total (reviewed > unreviewed → in-AF-proteome > not → lowest accession) | subprocess test under 3 `PYTHONHASHSEED` values |
| G4 | `primary_symbol ∈ symbols`; `entrez(accession(e)) == e` | assert at build |
| G5 | `entrez → accession` collapse factor **exactly 1** | assert `max_collapse == 1` |
| G6 | No free-text identifier search anywhere in the codebase | grep lint in CI |
| G7 | Isoform accessions (`P12345-2`) and non-human excluded | assert at build |

G5 is the specific guarantee whose absence caused the incident: the old map had mean collapse 2.55, max 9.

### Acceptance criteria

- `AVPR2 → P30518` (not Q8NCT1); `TUBA1B → P68363` (not Q13509); `IL2 → P60568` (not P01589).
- `cardinality_report()['max_collapse'] == 1`.
- Two builds under different `PYTHONHASHSEED` produce byte-identical `identity_index.parquet`.
- The regression assertion from `KNOWN_ISSUE.md` passes on the emitted index.

### Inputs to obtain (one manual fetch, checksummed, then offline forever)

`ls /opt/data` confirms no id-mapping file exists locally today.

| File | Source | Provides | Status |
|---|---|---|---|
| `HUMAN_9606_idmapping.dat.gz` | UniProt FTP `knowledgebase/idmapping/by_organism/` | accession ↔ GeneID, Gene_Name, Gene_Synonym | **required** |
| `UP000005640_9606.fasta.gz` | UniProt reference proteomes | review status, canonical one-per-gene set | **required** (tie-break) |
| `hgnc_complete_set.txt` | HGNC | prev/alias symbols, withdrawn detection, cross-check | recommended |
| `/opt/data/alphafold2/UP000005640_9606_HUMAN_v4.tar` | already local | dockable-accession universe | present |

`build_identity_index.py` is the **sole network consumer**, run manually, refusing to re-fetch without
`--refresh`. It emits `identity_index.parquet` + `identity_index.meta.json` (source URLs, UniProt
release, per-file SHA256, row counts, build timestamp, invariant report). Every downstream entrypoint
pins that SHA256 and refuses to run against an unexpected index.

### Invalidates

`uniprot_gene_mapping.csv` (retired), `required_structures.csv`, `required_structures_with_negatives.csv`,
and every artifact derived from them. Does **not** invalidate any docking output — job identity is
recovered from the accession in the folder name, which is physical truth.

---

## 2. `prepare_uniprot_mapping.py` — retire, do not patch **V/S**

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

### 3.1 `identify_required_structures.py` **V/S**

- **`:170`** — replace `gene_to_uniprot.get(gene)` with `index.resolve_entrez(row.node_2)`; branch on
  `Resolution.status`; append every non-`OK` to `unresolved_targets.csv` with its reason. Iterate the
  interaction edges (demand), never `cavity_mapping.csv` (supply).
- **`:196`** — delete the dict inversion entirely. `Gene_Name` becomes `index.label(accession)`.
- **Invariant:** `n_edges_in == n_resolved + n_rejected`, logged and asserted.
- **Acceptance:** re-running produces a `required_structures.csv` in which every `(UniProt_ID, Gene_Name)`
  pair satisfies `Gene_Name ∈ index.by_accession(UniProt_ID).symbols` — currently false for 65.11% of rows.
- **Invalidates:** `required_structures.csv` and everything downstream of it.

### 3.2 `generate_negative_samples.py` **V/S**

- **`:190`** — same `Resolution` conversion as above.
- **`:303-319`** — the known-interaction avoid-list must be built in accession space from Entrez, not from
  `gene_to_uniprot[node_2_name]`. Only 50.11% of the current avoid-list entries point at an accession that
  really carries that gene.
- **Invariant:** for every emitted negative, `(drug, accession)` ∉ avoid-list, where the avoid-list is
  provably complete over the resolved edge set.
- **Acceptance:** contamination check reports 0 negatives that are known interactions; balance holds in
  *accession* space, not label space.
- **Invalidates:** `required_structures_with_negatives.csv`.

### 3.3 `batch_dock.py` **V/S**

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

### 3.4 `parse_prepare.py` **V/S**

| Site | Change | Invariant | Acceptance |
|---|---|---|---|
| `:146` | Remove the cap; discovery must be exhaustive or fail loudly. Replace `max_files_per_pattern` with an assertion that the number of discovered job dirs equals the number in the run manifest. | Discovery is total | All 24,173 dirs present; recovers 14,173 jobs / 845 GB / 8,732 core-h |
| `:359` | Delete the regex. Read `job_identity.json`; for legacy jobs use the right-anchored parse (strip trailing `_cavity_<N>`, strip optional `_-F<k>-model_v1`, test tokens against the accession set) which resolves 24,173/24,173 | Identity never derived from a fragile pattern | 0 rows with null `drugbank_id`/`uniprot_id` (currently 1,174,786) |
| `:750` | Replace tool literals with a single normalising `canonical_tool()` used everywhere | One tool-name vocabulary | `LeDock_Score` non-null count equals the LEDOCK row count |
| `:1335` | Same `canonical_tool()`; assert `complete_pairs` is non-empty or fail | A no-op step cannot report success | Filter demonstrably filters; `processing_steps_applied` reflects reality |
| `:1171` | Add `Fragment_ID` to the merge key (normalise `'1'` vs `'F1'` first) **or** derive the label from the identity index and drop the merge | A merge may not change row count unintentionally | `checked_merge()` asserts `len(out) == len(left)`; eliminates the 410,087 duplicated rows |

**Invalidates:** `combined_filtered_annotated_docking_results.parquet` — rebuild required. Note F-B: do
not expect the rebuild to reproduce the current file.

---

### 3.5 `extract_cavities.py` — the 76.6 GB over-extraction **V/S**

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

### 3.6 `extract_alphafold_models.py` — the 5,661-vs-979 over-extraction **V/S**

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

### 3.7 `create_ic50_mapping.py` — clean, keep **V/S**

**Verified `:84`:** `mapping = dict(zip(df['Entry Name'].to_list(), df['Entry'].to_list()))`. It uses the
`Entry Name → Entry` bijection (4,585 → 4,585, zero collisions) and never touches `Mapped_Gene_Name`.
`ic50_mapping.csv` is **not corrupted** and needs no rebuild for correctness.

- **Edit site:** replace the hand-built dict with `IdentityIndex`, purely to lift the coverage ceiling.
- **Invariant:** TTD targets that fail to resolve are written to a rejects file, never dropped silently.
- **Acceptance:** resolvable TTD targets rise from 1,672/3,685 (45.4%) toward the reference-proteome
  ceiling; the 118 entries with multi-value separators are parsed rather than discarded.
- **Invalidates:** nothing. Regeneration is additive — existing rows keep their values.

### 3.8 `run_full_workflow.py` **V/S**

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

| Item | Citation | Verified | Risk |
|---|---|:--:|---|
| Job-folder name is a load-bearing data contract | `batch_dock.py:2211` writes, `parse_prepare.py:359` reads | ✅ | **Highest** — fixed by `job_identity.json` (§3.3) |
| Universe-widening fallbacks | `extract_alphafold_models.py:214`, `extract_cavities.py:65`, `run_full_workflow.py` (F-C) | ✅ | High — fixed by G8 |
| `pocketmatch_clustering.py:426` | `help='...'tmux` — stray token; file **does not parse** | ✅ | Medium — the script cannot have run since that edit |
| `fix_pdb_files.py` | 0 bytes | ✅ | Low — delete; live script is `fix_required_pdbs.py` |
| Duplicated import block | `batch_dock.py:34-47` — `os, sys, pandas, subprocess, re, time, logging` imported twice | ✅ | Low (cosmetic, but a marker of unreviewed edits) |
| Module-level config constants, no CLI | every pipeline script | ✅ | Medium — `config.py`/`RunConfig` |
| Absolute paths to one external drive | `parse_prepare.py:39`, `batch_dock.py:77`, `create_ic50_mapping.py:27-31` | ✅ | Medium |
| No dependency pinning | no lockfile in repo | ✅ | Medium — compounds F-B |

---

## 5. Remaining for subsequent passes ○

- Support modules: `checked_merge()`, `invariants.py`, `schemas.py`, `config.py`, `jobid.py` — API specs.
- Test tiers 0–4 (mechanical gates → pure-function units → golden files → e2e smoke → read-only auditor).
- Full sequencing with the artifact-invalidation matrix.
- The compound axis: 222 rows in `1_drug_to_protein.tsv` `node_1` are name slugs, not DrugBank IDs — needs its own verification pass.
