# Pathway Pipeline

A pathway-based anomaly-detection pipeline for untargeted metabolomics, built
on PathBank. It maps dataset features to HMDB metabolites and PathBank
pathways, calibrates them against a clean normal reference, and flags samples
whose pathway or metabolite disturbances exceed what normals reach — under a
strict development/validation protocol (label-blind development QC, one-shot
label-aware evaluation).

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
    ├── pathway_stats.py          # z-scores, Stouffer, flagging, summaries
    ├── calibration.py            # Cohort split + leave-one-out reference hygiene
    ├── develop.py                # Label-blind development QC (STEP 9)
    ├── evaluate.py              # One-shot label-aware evaluation (STEP 10)
    ├── test_pathway_pipeline.py  # Unit tests (synthetic inputs)
    ├── test_pathway_stats.py     # Unit tests for the statistics layer
    └── test_calibration.py      # Unit tests for calibration/evaluation
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

### Identity curation (`prefer_tagged_features`)

Feature columns carrying a trailing `.HMDB########` tag are identified by
pure-standard injection upstream, so within a resolved metabolite the
tagged feature is definitely the molecule described. When a plain-named
feature resolves to the same HMDB ID (a co-eluting interloper caught by
name matching, typically with a razor-thin reference IQR -- e.g. plain
`Argininosuccinic acid` at IQR 0.075 vs its tagged twin at 1.13), the
plain twin is dropped from scoring **before any downstream stage sees
it**: the metabolite's z then comes from the confirmed feature alone,
instead of averaging a real measurement with a wrong-compound measurement
at any weight. Tagged-vs-tagged pairs sharing an ID are both kept (both
confirmed; scale² weighting handles their relative noise), and manual
overrides carry the same precedence as a tag. The rule is class-level and
label-blind (column naming and resolved identity only); dropped twins are
written to `superseded_features.csv`. Set `prefer_tagged_features: false`
to restore the old behavior of averaging all twins.

## Pathway Coverage Rule

A pathway is kept only when the fraction of its PathBank-listed metabolites
(distinct HMDB IDs) that are matched to dataset features is at least
`min_pathway_coverage` (default `0.20`, i.e. 20%). Pathways below the
threshold are dropped and logged.

### Chemistry-based pathway curation (`exclude_pathway_keywords`)

Pathways whose **name** contains any of the configured keywords
(case-insensitive substring) are dropped before scoring -- by default any
pathway with "lipid" in its name. The justification is the extraction
chemistry alone (evidence budget #3): a phase-extraction metabolomics
protocol does not recover complex lipids, so lipid-pathway features measure
extraction variability rather than physiology, and leaving them in inflates
the normals' null distributions (which raises the bar for true IMDs under
the `max_excess` rule). The criterion is a class-level keyword block in the
frozen configuration -- never pathway-by-pathway flag performance, and no
disease label is read.

The filter drops **pathways, not features**: a metabolite shared between an
excluded and a kept pathway keeps its z-score and its Stouffer contribution
through the kept pathway. Only features whose *every* pathway is excluded
vanish from scoring entirely -- in practice exactly the complex-lipid
features the extraction cannot recover. `feature_to_pathway.csv` is
written before the filter, so the full link table (including excluded
pathways) is preserved in the outputs for audit.

### Redundancy pruning (`prune_redundant_pathways`, STEP 6)

PathBank disease pathways are mechanism cartoons that share nearly all of
their scored metabolites with their parent metabolic pathway: five pathways
driven by the same three metabolites produce five identical flags and
inflate pathway-test multiplicity under the `max_excess` rule. After the
scored-coverage filter, pathways whose **scored metabolite sets** have
Jaccard >= `redundancy_jaccard` (default `0.8`) are collapsed into connected
groups and each group keeps one representative, chosen by a pre-declared,
composition-based preference (evidence budget #3-style class-level rule --
never flag performance, never disease labels):

1. General metabolic pathway over a disease cartoon (name without
   "deficiency"/"disease"/"aciduria").
2. Larger scored metabolite set (more information).
3. Alphabetical by pathway name (determinism).

Dropped pathways are recorded in `pruned_pathways.csv`
(`smp_id`, `pathway_name`, `represented_by`) for audit. Pruning removes
duplicate **pathway tests**, not metabolites: every scored metabolite keeps
its z-score and remains represented through the surviving pathway, so
sample-level flag counts should barely move -- the reduction is in
pathway-test multiplicity and flag-evidence inflation. The STEP 9 redundancy
report runs on the pruned set, so what it reports at the threshold is what
actually remains.

## Metabolite Z-Scores (stage 2)

Values must already be log10-transformed upstream. Every pathway-mapped
feature is turned into a robust z-score against the **normal reference set**
(samples with `Classification == 0` AND `Oordeel targeted == 0`):

```
z_i(s) = (x_i(s) - median_i) / IQR_i      (medians/IQRs over normals only)
```

The scientific protocol splits the available evidence into three budgets,
enforced by the pipeline structure:

1. **Normal-reference statistics** (medians, IQRs, per-pathway percentile
   thresholds, null distributions): free to use, but computed on
   **development-half normals only** (see the cohort split below).
2. **IMD labels**: evaluation only, read once per frozen configuration
   version on the validation half (STEP 10). Never during calibration.
3. **External chemistry knowledge** (known artifact compounds, HMDB name
   collisions): free to use, documented in `config.yaml`.

### Cohort split (development vs validation, STEP 5)

Every sample is assigned exactly once, deterministically, to a **development**
or **validation** half (`stratified_split`, seeded permutation stratified by
reporting group; `validation_split.seed` and `.fraction` in `config.yaml`,
part of the frozen configuration). Calibration uses development normals
only; validation samples are scored like any other sample but never
contribute to any median, IQR, threshold, or null. The assignment is written
to `cohort_split.csv` and appears as the `validation` column in
`sample_decisions.csv`.

### Automated reference hygiene (STEP 5)

Hand-picked reference exclusions are replaced by a pre-specified,
label-blind rule (`reference_hygiene` in `config.yaml`): a candidate normal
is excluded from the calibration reference when its **leave-one-out** max
metabolite |z| (computed against the other candidate normals' median/IQR,
so it cannot mask its own disturbance by inflating the reference spread)
exceeds `reference_hygiene.max_depth` (default 20, justified from the
ordinary-normal LOO depth distribution: a depth inside that range
cascades and guts the reference). The check iterates: when an exclusion
tightens the peer reference enough to push another borderline candidate
past the threshold, that candidate is excluded in the next round, until no
new candidate crosses (capped at 10 rounds). A pre-declared cap
(`max_excluded_fraction`, default 10%) stops a miscalibrated threshold's
cascade: when the cap is hit, the deepest candidates up to the cap are
excluded, a WARNING is logged, and the reference must not be trusted until
`max_depth` is reconsidered against `reference_hygiene.csv`. Excluded
samples keep their scores and group label everywhere; they only leave the
calibration reference.

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
├── superseded_features.csv   # plain twins dropped in favor of tagged twins
├── feature_to_pathway.csv     # (feature, pathway) links
├── pathway_coverage.csv       # per-pathway matched metabolites/features + coverage
├── cohort_split.csv           # per-sample development/validation assignment
├── reference_hygiene.csv      # per-candidate-normal LOO depth + exclusion
├── metabolite_zscores.csv      # per-sample robust z-scores (normals reference)
├── reference_stats.csv         # per-feature median/scale/normal-value counts
├── dropped_features.csv        # features dropped before z-scoring, with reason
├── pathway_coverage_scored.csv # coverage recomputed over calibrated features
├── pathway_stouffer_scores.csv # per (sample, pathway) signed + absolute Stouffer
├── pathway_stouffer_reference.csv # per-pathway normal p50/p95/p99 of |Stouffer|
├── pathway_flags.csv          # per (sample, pathway) threshold/excess/flagged
├── sample_decisions.csv       # per-sample decision + p-value + evidence + half
├── development_qc_noise_floor.csv # noise-floor features (STEP 9)
├── evaluation_summary.csv     # one-shot validation metrics (STEP 10)
└── metabolite_flags.csv         # per (sample, metabolite) |z| flags
```

## Pathway Stouffer Scores (stage 3)

Per pathway and sample, over the pathway's metabolite z-scores (features
mapping to the same HMDB ID are averaged, so one metabolite counts once):

```
z_stouffer     = sum(z_i)  / sqrt(k)     (direction-aware)
z_stouffer_abs = sum(|z_i|) / sqrt(k)    (disturbance regardless of direction)
```

Samples with fewer than `min_stouffer_metabolites` usable metabolites in a
pathway get no score for it. `max_abs_z` (default 15) caps |z| before the
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
flags are expected for *every* sample, so the sample decision combines a
count rule (`min_flagged_pathways`, default 1) with a null model for the
flagged-pathway count (`sample_rule`):

- `empirical`: the null is the observed flagged-pathway count
  distribution of the normals themselves. PathBank pathways share
  metabolites, so flags are correlated and the binomial null is
  anti-conservative (on real data it flagged 16% of normals at p=0.05);
  the empirical distribution absorbs the correlation automatically.
  Measures **breadth**.
- `max_excess` (default): the null is the observed distribution of each
  normal's *maximum* pathway excess. A sample is flagged when its single
  most extreme pathway exceeds what (1 - `max_sample_p`) of normals reach.
  Measures **depth**: classic IMD blocks a few pathways profoundly, while
  the screened "normals" shift many pathways mildly -- this separates
  exactly those two patterns.
- `binomial`: Binomial(n_scored_pathways, 1 - percentile); only valid when
  pathway flags are near-independent.
- `none`: count rule only.

A sample is flagged when it passes the count rule AND
`sample_p_value <= max_sample_p` (default 0.05) under the chosen null.
`sample_decisions.csv` reports the per-sample p-value together with the
Classification/Oordeel group (`normal` = Class 0 + Oordeel 0, `imd` =
Class 1 + Oordeel 1, `other`) and the development/validation half -- a
**reporting-only** detection-vs-contamination summary. Thresholds are
never tuned against the IMD labels; the development/validation protocol
below is what keeps the flagging method scientifically valid.

### Metabolite-level flags (STEP 8c, report-only)

`sample_decisions.csv` gains `max_metabolite_z`, `n_flagged_metabolites`,
`metabolite_depth_p`, and `top_metabolite` (plus `metabolite_flags.csv` with
every per-metabolite flag). A metabolite is flagged when its |z| exceeds the
`metabolite_flag_percentile` percentile of that metabolite's own |z| among
the reference normals -- per-metabolite calibration absorbs noisy features.
The `metabolite_depth_p` is the fraction of normals whose maximum metabolite
|z| reaches the sample's maximum: the metabolite-level analogue of the
pathway max_excess rule. It catches IMDs with a single grossly elevated
metabolite whose pathway Stouffer score is diluted by the pathway's other
metabolites. These columns are **report-only**; the sample `flagged` decision
still comes from the pathway rules (plus the STEP 8d biomarker channel),
not from this table.

### Biomarker attachment channel (STEP 8d, literature-curated)

PathBank disease pathways are intracellular mechanism cartoons and can omit
the clinically diagnostic biomarkers of the disease they depict (the MCADD
pathway does not list octanoylcarnitine). The biomarker channel injects
curated prior knowledge as a **separate, openly declared channel**: PathBank
pathways are never modified, so the pathway channel stays pure and any
performance difference between the channels is attributable and auditable.

The attachment table (`data/pathway_biomarker_attachments.csv`, columns
`smp_id` OR `pathway_name`, `hmdb_id`, `source`) is user-curated from
systematic literature sources (biomarker tables in reviews / newborn-
screening guidelines), with the citation per row in `source`. Curation rules
evidence budget #3: attachments are chosen from textbook knowledge only,
never from which samples the pipeline flagged or missed, and the table is
frozen before the STEP 10 read. The knowledge is gene/disease-level, so it
is label-blind by construction -- the same table would be declared for any
cohort. A missing file disables the channel with a warning until the table
is provided.

Mechanics: attached biomarkers are z-scored against the same development
normals (even when no kept PathBank pathway maps their feature -- the
z-score stage extends to them). A sample flags the channel when an attached
biomarker exceeds its own normal-percentile threshold (`biomarker_channel.
threshold_percentile`) AND the sample's maximum attached-biomarker |z|
beats the biomarker-restricted depth null of the reference normals at the
same frozen `max_sample_p` as the pathway channel. The channel **ORs into
the sample decision**: `sample_decisions.csv` gains
`flagged_pathway_channel`, `biomarker_flagged`, `n_flagged_biomarkers`,
`biomarker_depth_p`, `max_biomarker_z`, and `top_biomarker`, and every
per-(sample, pathway, biomarker) flag lands in `biomarker_flags.csv` for
audit. Duplicate features of the same biomarker are combined with the same
scale^2 weighting as the Stouffer channel.

## Development QC (STEP 9, label-blind)

`run_development_qc: true` runs every diagnostic on the calibration
reference and measurement/chemistry properties only -- IMD labels are
never read here, so iterating on this output cannot leak disease signal
into any choice:

1. **Calibration verification**: the reference normals' z-scores must have
   median ~0 and IQR ~1 by construction; deviations mean a wrong reference
   subset or a stale reference.
2. **Noise-floor features**: value spikes with razor-thin reference IQR --
   z-magnifiers that manufacture huge z-scores from small absolute changes
   (`development_qc_noise_floor.csv`).
3. **Diverging duplicates**: metabolites backed by multiple dataset features
   whose reference IQRs differ >2x; the averaged z is dominated by the
   noisiest twin, or the features are not the same compound (fix with
   `feature_hmdb_overrides`).
4. **Threshold stability**: bootstrap resampling of the reference shows how
   much each pathway's 99th-percentile threshold wobbles; an unstable
   threshold produces borderline flags that flip on reference resampling.
5. **Pathway redundancy**: pathway pairs whose scored-metabolite sets are
   near-identical (Jaccard >= `redundancy_jaccard`) -- they produce duplicate
   flags and inflate multiplicity. STEP 6 already prunes these label-blind
   (see `prune_redundant_pathways`); this report shows what remains at the
   threshold, e.g. pairs that are similar but below the pruning threshold.

Acting on the output (adding an override or demotion, changing a
threshold) is a human decision that creates a **new frozen configuration
version**.

### Feature demotion (`demoted_features`)

Demoted features keep their z-scores in `metabolite_zscores.csv` for audit
but never contribute to pathway Stouffer sums, metabolite flags, or
downstream threshold calibration. Entries must be **exact feature column
names**; the pipeline logs a warning for any configured name that matches
no scored column. Justification is chemistry/analytical priors only
(evidence budget #3): pre-analytical lability (e.g. Reduced Glutathione,
Urocanic acid), noise-floor compression (thiamine monophosphate), strong
age dependence (creatinine -- pediatric reference ranges span an order of
magnitude, so deviations track maturation, not IMD), and
exogenous/medication-dominated features (caffeine chain, theophylline,
Premarin, tretinoin, etc.) where a minority of exposed normals would
otherwise inflate the reference null distributions and produce
non-specific pathway flags. No group/disease counts inform the list.
Demoting features starves medication pathways (e.g. Caffeine Metabolism)
of coverage so they drop out of scoring on their own.

## Evaluation (STEP 10, label-aware, one-shot)

`run_evaluation` stays `false` during development. When the configuration
is frozen -- no further changes in response to its numbers -- flip it to
`true` for exactly one run:

- Sensitivity and specificity with **exact (Clopper-Pearson)** confidence
  intervals, on the validation half (and optionally the development half
  via `evaluate_dev_half`, for sanity checking only).
- **ROC-AUC** of the continuous anomaly score (max pathway excess) with a
  bootstrap CI, IMD vs normal.
- Flag-resolution evidence per flagged sample, and a per-group breakdown
  (the `other` group is reported but not part of the primary metrics).

Results are written to `evaluation_summary.csv`. Reading the validation
metrics and then changing the configuration invalidates the validation
half -- the act of iterating on validation numbers, not any single
number, is what breaks it. If the numbers motivate changes, freeze a new
configuration version and re-evaluate once.

## Usage

```bash
python -m pathway_pipeline.main
python -m pathway_pipeline.main --input data/my_data.csv --output outputs/pathway
python -m pathway_pipeline.main --config my_config.yaml
```

## Testing

```bash
python -m pytest pathway_pipeline/pipeline/ -q
```
