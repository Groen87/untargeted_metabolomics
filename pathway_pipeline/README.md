# Pathway Pipeline

A pathway-mapping pipeline for untargeted metabolomics, built on PathBank. The
current scope is **feature engineering** (per-pathway statistics will be added
in later stages):

1. **Matches** every feature column to an HMDB accession using the HMDB XML
   metabolite database (`hmdb_metabolites.xml`, a streaming name/synonym
   index).
2. **Loads** the PathBank primary-pathways metabolites CSV
   (`pathbank_all_metabolites.csv`), keeping only rows for the configured
   **species** (default `Homo sapiens`).
3. **Maps** the matched HMDB accessions to PathBank pathways and keeps only
   pathways where **at least 20%** (configurable) of the pathway's metabolites
   are mapped to features in the dataset.

## Directory Structure

```
pathway_pipeline/
├── __init__.py
├── main.py                       # Entry point (feature engineering)
├── config/
│   ├── __init__.py
│   ├── config.py                 # Dot-notation Config loader
│   └── config.yaml               # Default configuration
└── pipeline/
    ├── __init__.py
    ├── name_utils.py             # Canonical name normalization (Greek folding)
    ├── hmdb_parser.py            # Streaming HMDB XML -> name index (+ cache)
    ├── pathway_mapping.py        # feature -> HMDB -> PathBank pathway mapping
    └── test_pathway_pipeline.py  # Unit tests (synthetic inputs)
```

## Input Files

- **Feature matrix CSV** (`data/merged_data_with_classification.csv`): rows =
  samples, columns = metabolite features plus the non-feature columns
  (`Oordeel targeted`, `Classification`). Feature columns are named as a plain
  metabolite name, a bare HMDB accession (`HMDB0000063`), or a compound name
  with an `.HMDB########` suffix (`Cortisol.HMDB0000063`).
- **HMDB XML** (`data/hmdb_metabolites.xml`): the HMDB metabolite database.
  Each `<metabolite>` element carries an `<accession>`, a primary `<name>`,
  and zero or more `<synonyms>`. The file is streamed with `iterparse` and the
  resulting name index is cached to disk.
- **PathBank primary-pathways metabolites CSV**
  (`data/pathbank_all_metabolites.csv`): one row per (pathway, metabolite)
  pair, primary pathways only:
  ```
  pathway_id,metabolite_name,metabolite_id,hmdb_id,kegg_id,chebi_id,formula,smiles,iupac_name,inchi_key,species,source,relation,expected_direction,weight,msi_level,plasma_observable,measured
  SMP0000055,Adenosine triphosphate,PW_C000414,HMDB0000538,C00002,...,Homo sapiens,pathbank,direct_member,...
  ```
  Only rows matching the configured `pathbank_species` (default
  `Homo sapiens`) are used. The file carries no pathway name, so pathway
  names come from the PathBank pathways description CSV
  (`pathbank_pathway_names_file`, default `data/pathbank_pathways.csv`:
  `pathway_id,pathbank_id,smpdb_id,name,subject,description,...`); a pathway
  missing from it falls back to its `pathway_id` (e.g. `SMP0000055`) as name.
  A pathway's metabolite set is the distinct set of its `hmdb_id` values. A
  species that matches nothing in the CSV fails fast with a logged error
  listing the available values.

## Feature -> HMDB Matching

Matching priority (first hit wins; the method is recorded in the output):

1. **HMDB tag** — the column contains a trailing HMDB accession (bare
   `HMDB########` or `Name.HMDB########`). The tag is taken authoritatively.
2. **Exact name** — the full normalized column name matches an HMDB primary
   name or synonym.
3. **Loose name** — the non-alphanumeric-stripped column name matches a
   loose-normalized index entry, catching hyphenation/spacing/punctuation
   differences.

Normalization folds Greek symbols and spelled-out Greek words to a single
canonical token and lipid-shorthand `W` between digits to `OMEGA`, so e.g.
`PS(18:2ω6/24:1ω9)` == `PS(18:2W6/24:1W9)` == `PS(18:2omega6)`.

## Pathway Coverage Rule

A pathway is kept only when the fraction of its PathBank-listed metabolites
(distinct HMDB IDs) that are matched to dataset features is at least
`min_pathway_coverage` (default `0.20`, i.e. 20%). Pathways below the
threshold are dropped and logged.

## Metabolite Z-Scores (stage 2)

Values must already be log10-transformed upstream. Every pathway-mapped
feature is turned into a robust z-score against the **normal reference set**
(samples with `Classification == 0` AND `Oordeel targeted == 0`):

```
z_i(s) = (x_i(s) - median_i) / IQR_i      (medians/IQRs over normals only)
```

Features whose reference scale is zero (flat in the normals) or that have no
normal values at all are dropped (`dropped_features.csv`, with the reason) --
they cannot be calibrated and would dilute pathway scores. After the drop,
pathway coverage is recomputed over the surviving features and pathways with
fewer than `min_pathway_features` (default 3) usable matched metabolites are
removed (`pathway_coverage_scored.csv`): a Stouffer score over fewer than 3
metabolites is dominated by a single outlier.

## Outputs

Outputs are written to `outputs/pathway_pipeline/`:

```
outputs/pathway_pipeline/
├── pathway_pipeline.log
├── feature_to_hmdb.csv        # feature -> HMDB accession(s) + match method
├── feature_to_pathway.csv     # (feature, pathway) links
├── pathway_coverage.csv       # per-pathway matched metabolites/features + coverage
├── metabolite_zscores.csv      # per-sample robust z-scores (normals reference)
├── reference_stats.csv         # per-feature median/scale/normal-value counts
├── dropped_features.csv        # features dropped before z-scoring, with reason
├── pathway_coverage_scored.csv # coverage recomputed over calibrated features
├── pathway_stouffer_scores.csv # per (sample, pathway) signed + absolute Stouffer
├── pathway_stouffer_reference.csv # per-pathway normal p50/p95/p99 of |Stouffer|
├── pathway_flags.csv          # per (sample, pathway) threshold/excess/flagged
└── sample_decisions.csv       # per-sample decision + top evidence + review group
```

## Pathway Stouffer Scores (stage 3)

Per pathway and sample, over the pathway's metabolite z-scores (features
mapping to the same HMDB ID are averaged, so one metabolite counts once):

```
z_stouffer     = sum(z_i)  / sqrt(k)     (direction-aware)
z_stouffer_abs = sum(|z_i|) / sqrt(k)    (disturbance regardless of direction)
```

Samples with fewer than `min_stouffer_metabolites` usable metabolites in a
pathway get no score for it. `max_abs_z` (default 10) caps |z| before the
sum so a single artifact feature cannot dominate a whole pathway.
`pathway_stouffer_reference.csv` records each pathway's empirical
p50/p95/p99 of the absolute Stouffer score over the normals -- the
calibration basis for the flagging stage (a noisy pathway automatically
gets a wider normal range instead of being hand-pruned).

## Flagging (stage 4)

A (sample, pathway) pair is flagged when its absolute Stouffer score exceeds
the `flag_threshold_percentile` percentile (default 99) of that pathway's own
**normals** -- empirical per-pathway calibration, so a noisy pathway
automatically gets a wider range. With 200+ pathways, a few chance pathway
flags are expected for *every* sample (~2 of 234 at p99), so the sample
decision combines a count rule (`min_flagged_pathways`, default 1) with a
binomial rule (`use_binomial_sample_rule`, default on): the sample's
flagged-pathway count must be improbable under
Binomial(n_scored_pathways, 1 - percentile) with p <= `max_sample_p`
(default 0.05). The per-sample p-value is reported as `sample_p_value` in
`sample_decisions.csv`, together with the Classification/Oordeel group
(`normal` = Class 0 + Oordeel 0, `imd` = Class 1 + Oordeel 1, `other`) --
a **reporting-only** detection-vs-contamination summary. Thresholds are
never tuned against the IMD labels; tuning would have to happen inside
cross-validation.

## QC Diagnostics

`qc_extremes.py` reports where extreme z-scores concentrate among the
normals (per-feature = unstable feature, per-sample = QC-suspect sample) and
verifies the calibration (per-feature median ~0, IQR ~1):

```bash
python -m pathway_pipeline.qc_extremes \
    --zscores outputs/pathway_pipeline/metabolite_zscores.csv \
    --input data/merged_data_with_classification.csv \
    --threshold 10.0 --top 10
```

## Usage

```bash
python -m pathway_pipeline.main
python -m pathway_pipeline.main --input data/my_data.csv --output outputs/pathway
python -m pathway_pipeline.main --config my_config.yaml
```

## Testing

```bash
python -m pytest pathway_pipeline/pipeline/test_pathway_pipeline.py -q
```
