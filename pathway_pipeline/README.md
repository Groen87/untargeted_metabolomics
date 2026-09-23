# Pathway Pipeline

A pathway-mapping pipeline for untargeted metabolomics, built on PathBank. The
current scope is **feature engineering** (per-pathway statistics will be added
in later stages):

1. **Matches** every feature column to an HMDB accession using the HMDB XML
   metabolite database (`hmdb_metabolites.xml`, a streaming name/synonym
   index).
2. **Loads** the PathBank all-metabolites CSV
   (`pathbank_all_metabolites.csv`), keeping only **Metabolic** and
   **Disease** pathways for **Homo sapiens**.
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
- **PathBank all-metabolites CSV** (`data/pathbank_all_metabolites.csv`): one
  row per (pathway, metabolite) pair:
  ```
  PathBank ID,Pathway Name,Pathway Subject,Species,Metabolite ID,Metabolite Name,HMDB ID,...
  SMP0000055,Alanine Metabolism,Metabolic,Homo sapiens,PW_C000105,L-Alanine,HMDB0000161,...
  ```
  Only rows with `Pathway Subject` in {Metabolic, Disease} and
  `Species == Homo sapiens` are used, and a pathway's metabolite set is the
  distinct set of its `HMDB ID` values.

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

## Outputs

Outputs are written to `outputs/pathway_pipeline/`:

```
outputs/pathway_pipeline/
├── pathway_pipeline.log
├── feature_to_hmdb.csv       # feature -> HMDB accession(s) + match method
├── feature_to_pathway.csv    # (feature, pathway) links
└── pathway_coverage.csv      # per-pathway matched metabolites/features + coverage
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
