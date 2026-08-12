# KNOWN ISSUE — `prepare_uniprot_mapping.py` produces a corrupted gene mapping

**Status:** confirmed, unfixed
**Found:** 2026-08-12
**Severity:** high — silently corrupts every downstream artifact that joins on `Mapped_Gene_Name`
**Affected output:** `uniprot_gene_mapping.csv` (all copies, including
`/media/onur/Elements/multiscale_interactome_data/mi_drug_protein_edges/data/uniprot_gene_mapping.csv`)

---

## Summary

`prepare_uniprot_mapping.py` assigns the **queried** gene name to **every** UniProt entry returned by a free-text search, not just to the entry that actually matches. As a result **57.7% of rows in `uniprot_gene_mapping.csv` carry a gene name belonging to a different protein**, and all 4,585 entries collapse onto only 1,795 distinct gene names.

Any pipeline that maps UniProt accession → gene → protein/node using this file has been assigning data to the wrong protein for the majority of its records.

---

## The bug

`prepare_uniprot_mapping.py`, lines ~71–91:

```python
for i, gene_name in enumerate(tqdm(unique_gene_names, ...)):
    df = service.get_df(gene_name, columns="Entry,Entry Name,Gene Names,Organism")   # (1)
    ...
    df_filtered = df[df['Organism'].str.contains("Homo sapiens", na=False)].copy()

    found_for_this_gene = False
    for gene_name_entry in df_filtered['Gene Names'].dropna():                        # (2)
        if gene_name in str(gene_name_entry).split():
            found_for_this_gene = True
            break

    if found_for_this_gene:
        df_filtered['Mapped_Gene_Name'] = gene_name                                   # (3)  ← BUG
        dfs.append(df_filtered)
```

1. `service.get_df(gene_name)` (bioservices `UniProt`) is a **free-text search**, not an exact gene-symbol lookup. Querying `TUBA1B` returns TUBA1B *and* every other human entry whose text matches — TUBGCP6, TUBGCP3, TUBGCP4, TUBB3, …
2. The guard only checks that **at least one** returned row mentions the queried gene. It does not filter the others.
3. The queried name is then broadcast to **all** surviving rows. Every unrelated hit is relabelled as the queried gene.

The result is not a near-miss or a synonym choice. It assigns entries to genuinely different genes.

---

## Measured impact on `uniprot_gene_mapping.csv`

| Quantity | Value |
|---|---|
| UniProt entries in file | 4,585 |
| Distinct `Mapped_Gene_Name` values | **1,795** |
| Mean entries collapsed per mapped gene | 2.55 (max 9) |
| Mapped genes receiving >1 UniProt entry | 1,191 / 1,795 |
| Rows where `Mapped_Gene_Name` is **not** the entry's own primary symbol | 2,698 (58.8%) |
| Rows where `Mapped_Gene_Name` is **not even a listed synonym** of the entry's own gene | **2,647 (57.7%)** |

Representative errors:

| UniProt | Real gene (`Gene Names`) | Assigned `Mapped_Gene_Name` |
|---|---|---|
| Q96RT7 | TUBGCP6 GCP6 KIAA1669 | TUBA1B |
| Q96CW5 | TUBGCP3 GCP3 | TUBA1B |
| Q13509 | TUBB3 TUBB4 | TUBA1B |
| P01589 | IL2RA | IL2 |
| Q7L0Y3 | TRMT10C MRPP1 RG9MTD1 | HSD17B10 |
| Q14242 | SELPLG | SELE |
| Q9BQE5 | APOL2 | SLC18A1 |
| Q9Y6G1 | TMEM14A C6orf73 PTD011 | SLC18A1 |

A second, structural consequence: the script iterates over a **query list of gene names supplied by the caller**. In the MSI work that list was the curated drug-target gene set, so every one of the 1,795 output genes is a pre-existing drug target and no other protein can ever be reached through the file. Any coverage ceiling observed downstream is imposed here, not by the source dataset.

Reproduce with:

```python
import pandas as pd
m = pd.read_csv("uniprot_gene_mapping.csv")
m["syns"] = m["Gene Names"].astype(str).str.split()
bad = [r.Mapped_Gene_Name not in r.syns for r in m.itertuples()]
print(sum(bad), "/", len(m), "rows mapped to a non-synonymous gene")
```

---

## Downstream consumers to audit

Anything joining on `Mapped_Gene_Name` inherits the corruption. In this repo:

- `batch_dock.py`
- `create_ic50_mapping.py`
- `generate_negative_samples.py`
- `identify_required_structures.py`
- `run_full_workflow.py`

Outside this repo (confirmed affected): `multiscale-interactome/filter_logic.py` and
`multiscale-interactome/1_prepare_filtered_edges.py`, which load the file and use it to resolve
predicted drug–target affinities to protein nodes for both the docking and DTA-Atlas pipelines.

**Note a compounding effect there:** `filter_logic.py` groups by `(drug, Mapped_Gene_Name)` and
aggregates with `max`/`min`, so a target gene absorbing *n* collapsed UniProt entries receives the
best score across all *n* — including the unrelated ones. Mislabelling therefore also distorts the
retained values, not only their assignment. Any consumer that aggregates after joining on this
column has the same problem.

---

## Fix

Do not patch the free-text search — replace the approach.

1. Resolve identifiers with an **exact** mapping source rather than a text query: the UniProt
   ID-mapping service, or a downloaded `HUMAN_9606_idmapping.dat` / `idmapping_selected.tab`.
   Map in the direction the data actually flows (accession → symbol), not symbol → search results.
2. If a per-gene query must be kept, filter the returned frame to rows that actually contain the
   queried symbol before assigning:
   ```python
   hit = df_filtered[df_filtered["Gene Names"].astype(str).str.split().apply(lambda s: gene_name in s)]
   if not hit.empty:
       hit = hit.copy()
       hit["Mapped_Gene_Name"] = gene_name
       dfs.append(hit)
   ```
   This is the minimal correction, but it still yields only the queried genes and inherits the
   query-list ceiling described above.
3. Aggregate on the **resolved identifier**, never on a name column that may be many-to-one.

## Regression check

One assertion catches this class of error and would have caught it here immediately:

```python
assert all(
    r.Mapped_Gene_Name in str(r._asdict()["Gene Names"]).split()
    for r in mapping.itertuples()
), "Mapped_Gene_Name must be a symbol or synonym of the entry it labels"
```

Add an equivalent check to any future identifier-mapping step, plus a cardinality report
(`n_input_accessions`, `n_distinct_output_symbols`, `max_collapse_factor`) — a collapse factor
well above 1 is the signature of this failure mode.

---

## Provenance of this report

Findings verified directly against `prepare_uniprot_mapping.py` and against
`/media/onur/Elements/multiscale_interactome_data/mi_drug_protein_edges/data/uniprot_gene_mapping.csv`
on 2026-08-12. All counts above are recomputed from that file, not estimated. The downstream
consumer list is from a repository-wide grep for `Mapped_Gene_Name`/`uniprot_gene_mapping`; the two
`multiscale-interactome` consumers were confirmed by reading the code, the five in this repo were
not individually traced and should be checked before being trusted or cleared.
