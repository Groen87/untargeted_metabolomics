# Pathway Pipeline

A pathway-centric alternative to the outlier-detection pipeline. There is **no
outlier-detection model** here -- every layer is a z-based analysis and flagging
rule, escalating in statistical complexity only where the simpler layer fails.
The pipeline:

1. **Matches** every feature column to an HMDB accession using the HMDB XML
   metabolite database (a streaming name/synonym index).
2. **Links** those HMDB accessions to metabolic pathways via an SMPDB-derived
   pathways TSV.
3. **Runs the layered analysis & flagging engine** (see *Layered flagging*
   below): atomic per-metabolite flags → pathway statistics with severity tiers
   → a sample decision rule → an optional global anomaly safety light, plus an
   optional threshold-tuning sweep on the inner IMD split.

## Directory Structure

```
pathway_pipeline/
├── __init__.py
├── main.py                       # Entry point
├── config/
│   ├── __init__.py
│   ├── config.py                 # Dot-notation Config loader
│   └── config.yaml               # Default configuration
└── pipeline/
    ├── __init__.py
    ├── name_utils.py             # Canonical name normalization (Greek folding)
    ├── hmdb_parser.py            # Streaming HMDB XML -> name index (+ cache)
    ├── pathway_mapping.py        # feature -> HMDB -> pathway preprocessing
    └── pathway_stats.py          # Layered analysis: metabolite flags, pathway stats,
                              #   severity tiers, decision rule, global score, tuning
```

## Input Files

- **Feature matrix CSV** (`data/merged_data_with_classification.csv`): rows =
  samples, columns = metabolite features plus the non-feature columns
  (`Oordeel targeted`, `Classification`). Feature columns are named as a plain
  metabolite name, a bare HMDB accession (`HMDB0000063`), or a compound name with
  an `.HMDB########` suffix (`Cortisol.HMDB0000063`).
- **HMDB XML** (`data/hmdb_metabolites.xml`): the HMDB metabolite database. The
  `<metabolite>` elements each carry an `<accession>`, a primary `<name>`, and
  zero or more `<synonyms>`. The file is streamed with `iterparse` and the
  resulting name index is cached to disk.
- **Pathways TSV** (`data/pathways.tsv`):

  ```
  smp_id	pathway_name	n_compounds	hmdb_ids
  SMP0000575	11-beta-Hydroxylase Deficiency (CYP11B1)	41	HMDB0000015;HMDB0000016;...
  ```

  `hmdb_ids` is a `;`-separated list of HMDB accessions.

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
`PS(18:2ω6/24:1ω9)` == `PS(18:2W6/24:1W9)` == `PS(18:2omega6)`. This mirrors
the outlier-detection pipeline's endogenous keep-list, keeping feature
matching consistent across the two pipelines.

## Layered flagging (no outlier-detection model)

The analysis stage runs four layers, each a z-based rule that only fires when
the simpler layers miss:

1. **Per-metabolite z-scores (age-adjusted, robustly scaled)** -- the atomic
   evidence. For each metabolite `z_i(s) = (x_i(s) - median_i(age)) / IQR_i`,
   with the reference estimated over the **normal set only** (optionally
   age-regressed). A single metabolite at z = +9 with a known disease
   association is clinically meaningful even without pathway support, so
   `flag_metabolites` flags `|z_i| > metabolite_override_threshold` and these
   act as single-metabolite overrides in the decision rule.

2. **Pathway-level statistics** -- the primary detector. For each pathway P
   with k metabolites:

   ```
   Z_med(P, s)  = median(z_1 ... z_k)          (direction-aware median)
   F(P, s)      = (1/k) * Σ 1[|z_i| > t_i]      (flagged-fraction breadth)
   Z_up(P, s)   = median of the positive z_i     (signed-extreme guard)
   Z_down(P, s) = median of the negative z_i     (signed-extreme guard)
   ```

   `t_i` is the per-metabolite empirical threshold (99th percentile of `|z_i|`
   over normals). `flag_pathways` assigns a **severity tier**: a pathway is
   *moderate* if any moderate condition holds, *severe* if any severe condition
   holds. The signed extremes catch the cancel-out problem (upstream pileup +
   downstream depletion averaging toward a near-zero `Z_med`).

3. **The decision rule** (the operating point). A sample is flagged when:

   ```
   ≥1 SEVERE pathway flag
   OR  ≥2 MODERATE (non-severe) pathway flags
   OR  any single-metabolite override (|z_i| > override_threshold)
   OR  (optional) global anomaly score > global_threshold
   ```

   The thresholds are tuned on the inner IMD split by `tune_decision_thresholds`
   (`run_threshold_tuning: true`).

4. **Optional global anomaly score** -- the "odd sample" safety light. A
   parameter-light z-aggregate (mean of the top-k `|z|` across metabolites) that
   lights up for globally odd samples the pathway layers miss.

## Per-Pathway Shift Statistics

For each pathway P with k matched metabolites and each sample s:

```
z_i(s)       = (x_i(s) - median_i) / IQR_i        (IQR-scaled, direction-aware)

Z_med(P, s)  = median(z_1 ... z_k)                (direction-aware median)
F(P, s)      = (1/k) * Σ 1[|z_i| > t_i]           (flagged-fraction breadth)
Z_up(P, s)   = median of the positive z_i        (signed-extreme guard)
Z_down(P, s) = median of the negative z_i         (signed-extreme guard)
```

`median_i`, `IQR_i`, and `t_i` are estimated per metabolite over the **normal
reference set only** (no leakage from abnormal samples). `t_i` is the
per-metabolite empirical threshold = the 99th percentile of `|z_i|` over
normals (per-metabolite because after IQR scaling features have different tail
behaviours).

The signed extremes `Z_up` / `Z_down` catch the **cancel-out problem**
(upstream pileup + downstream depletion averaging toward a near-zero
`Z_med`): even when `Z_med ≈ 0`, a large `Z_up` or a large `|Z_down|` flags the
pathway.

A pathway is flagged for a sample when ANY holds:

```
|Z_med(P, s)|   > zmed_threshold              (default 2.0)
F(P, s)         > flagged_fraction_threshold  (default 0.5)
|Z_up(P, s)|    > signed_extreme_threshold     (default 2.5)
|Z_down(P, s)|  > signed_extreme_threshold     (default 2.5)
```

## Output Files

```
outputs/pathway_pipeline/
├── pathway_pipeline.log
├── feature_to_hmdb.csv         # feature, hmdb_id, match_method, n_hmdb_ids
├── feature_to_pathway.csv      # feature, hmdb_id, smp_id, pathway_name, n_compounds
├── pathway_coverage.csv        # smp_id, pathway_name, n_compounds, n_matched_features, matched_features, coverage
├── metabolite_zscores.csv      # sample x metabolite age-adjusted IQR-scaled z-scores
├── metabolite_flags.csv        # atomic single-metabolite overrides (|z| > override_threshold)
├── pathway_statistics.csv      # sample_id, smp_id, pathway_name, n_metabolites, z_med, flagged_fraction, z_up, z_down, threshold_percentile, flagged, severity, flag_reason
├── pathway_zmed_pivot.csv       # sample x pathway Z_med pivot
├── global_anomaly_scores.csv   # per-sample global anomaly score (top-k mean |z|)
├── sample_decisions.csv         # per-sample: flagged, n_severe_pathways, n_moderate_pathways, n_metabolite_overrides, global_anomaly_score, decision_reason
└── threshold_tuning.csv        # (optional) operating-point sweep on the inner IMD split
```

## Usage

```bash
python -m pathway_pipeline.main
python -m pathway_pipeline.main --input data/my_data.csv --output outputs/pathway
python -m pathway_pipeline.main --config my_config.yaml
```

## Tests

```bash
python -m pytest pathway_pipeline/pipeline/test_pathway_pipeline.py -q
```
