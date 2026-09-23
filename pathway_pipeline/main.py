#!/usr/bin/env python3
"""Main entry point for the pathway pipeline.

Feature-engineering stage:

1. Load the feature matrix CSV.
2. Build the HMDB name index from the HMDB XML (``hmdb_metabolites.xml``).
3. Match every feature column to one or more HMDB accessions
   (HMDB tag -> exact name -> loose name).
4. Load the PathBank all-metabolites CSV (``pathbank_all_metabolites.csv``),
   keeping only Metabolic and Disease pathways for Homo sapiens.
5. Map the matched HMDB accessions to PathBank pathways and keep only
   pathways where at least ``min_pathway_coverage`` (default 20%) of the
   pathway's metabolites are mapped to features in the dataset.

The pipeline stops here; the downstream per-pathway statistics will be added
in later stages.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

from pathway_pipeline.config.config import Config
from pathway_pipeline.pipeline.hmdb_parser import build_name_index
from pathway_pipeline.pipeline.pathway_mapping import (
    load_pathbank_pathways,
    match_features_to_hmdb,
    link_features_to_pathways,
    pathway_coverage,
)
from pathway_pipeline.pipeline.pathway_stats import (
    classify_samples,
    compute_metabolite_zscores,
    filter_pathways_for_scoring,
    compute_stouffer_scores,
    flag_pathway_scores,
    summarize_sample_flags,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _setup_logging(output_dir: Path) -> None:
    """Add file logging alongside the stream handler."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "pathway_pipeline.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)


def _log_section(title: str) -> None:
    logger.info(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_feature_matrix(input_file: str,
                         non_feature_columns,
                         patient_id_column: str = None,
                         age_column: str = None,
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the input CSV and split it into (features, metadata, ages).

    Keeps every configured non-feature column in a separate metadata frame
    and, when age_column is set and present, returns a numeric ages Series.
    """
    df = pd.read_csv(input_file, index_col=0 if patient_id_column is None else None)
    if patient_id_column is not None:
        df = df.set_index(patient_id_column)
    logger.info(f"Loaded {input_file}: {df.shape[0]} samples x {df.shape[1]} columns")

    nf = list(non_feature_columns)
    if age_column and age_column not in nf:
        nf = nf + [age_column]
    metadata = df[[c for c in non_feature_columns if c in df.columns]].copy()
    feature_cols = [c for c in df.columns if c not in nf]
    features = df[feature_cols].copy()

    ages = None
    if age_column and age_column in df.columns:
        ages = pd.to_numeric(df[age_column], errors="coerce")
        ages.index = df.index
        logger.info(f"Age column '{age_column}': {int(ages.notna().sum())} usable ages")

    logger.info(f"{len(feature_cols)} feature columns, {metadata.shape[1]} metadata columns")
    return features, metadata, ages


def _label_samples(metadata: pd.DataFrame, normal_mask: pd.Series) -> pd.Series:
    """Assign each sample a review group: normal, imd, or other.

    Normals are the configured reference set (Classification 0 and
    Oordeel 0); IMD samples are Classification 1 and Oordeel 1. The group is
    reporting-only evidence, never used to set thresholds.
    """
    group = pd.Series("other", index=metadata.index)
    if {"Classification", "Oordeel targeted"}.issubset(metadata.columns):
        cls = pd.to_numeric(metadata["Classification"], errors="coerce")
        oor = pd.to_numeric(metadata["Oordeel targeted"], errors="coerce")
        group[(cls == 1) & (oor == 1)] = "imd"
    group[normal_mask.reindex(metadata.index, fill_value=False)] = "normal"
    return group


def run_pipeline(input_file: str,
                  output_dir: str = "outputs/pathway_pipeline",
                  config_path: str = None) -> dict:
    """Run the feature-engineering stage of the pathway pipeline."""
    config = Config(config_path) if config_path else Config()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _setup_logging(out)

    _log_section("PATHWAY PIPELINE -- FEATURE ENGINEERING")
    logger.info(f"Input: {input_file}\nOutput: {out}")

    # ------------------------------------------------------------------
    # Stage 1: preprocessing -- feature -> HMDB -> pathway
    # ------------------------------------------------------------------
    _log_section("STEP 1: Load feature matrix")
    features, metadata, ages = load_feature_matrix(
        input_file,
        non_feature_columns=config.get_list(
            "non_feature_columns", ["Oordeel targeted", "Classification"]
        ),
        patient_id_column=config.get("patient_id_column", None),
        age_column=config.get("age_column", None),
    )

    _log_section("STEP 2: Build HMDB name index")
    hmdb_xml = config.get("hmdb_xml_file", "data/hmdb_metabolites.xml")
    min_name_length = int(config.get("min_name_length", 3))
    use_hmdb_cache = bool(config.get("use_hmdb_cache", True))
    name_index = build_name_index(hmdb_xml, min_name_length=min_name_length,
                                  use_cache=use_hmdb_cache)
    if not name_index:
        logger.warning("HMDB name index is empty (XML missing or unreadable). "
                       "Only HMDB-tagged features will be matched.")

    _log_section("STEP 3: Match features to HMDB accessions")
    feature_to_hmdb = match_features_to_hmdb(
        feature_columns=list(features.columns),
        name_index=name_index,
        min_name_length=min_name_length,
    )

    _log_section("STEP 4: Load PathBank pathways and link features to pathways")
    pathbank_file = config.get("pathbank_file", "data/pathbank_all_metabolites.csv")
    pathbank_species = config.get("pathbank_species", "Homo sapiens")
    pathway_names_file = config.get("pathbank_pathway_names_file",
                                    "data/pathbank_pathways.csv")
    pathways = load_pathbank_pathways(pathbank_file,
                                      species=pathbank_species,
                                      pathway_names_file=pathway_names_file)
    feature_to_pathway = link_features_to_pathways(feature_to_hmdb, pathways)
    min_coverage = float(config.get("min_pathway_coverage", 0.20))
    coverage = pathway_coverage(feature_to_pathway, pathways,
                                min_coverage=min_coverage)

    if bool(config.get("save_mapping_outputs", True)):
        feature_to_hmdb.to_csv(out / "feature_to_hmdb.csv", index=False)
        feature_to_pathway.to_csv(out / "feature_to_pathway.csv", index=False)
        coverage.to_csv(out / "pathway_coverage.csv", index=False)
        logger.info(f"Wrote feature_to_hmdb.csv, feature_to_pathway.csv, "
                     f"pathway_coverage.csv to {out}")

    n_matched = int(feature_to_hmdb.loc[feature_to_hmdb["hmdb_id"].notna(), "feature"].nunique())
    logger.info(f"Matched {n_matched} of {len(features.columns)} features to HMDB; "
                f"{coverage.shape[0]} pathways with >= {min_coverage:.0%} of their "
                f"metabolites mapped to dataset features.")

    if not coverage.empty:
        top = coverage.head(10)[["pathway_name", "n_metabolites",
                                 "n_matched_metabolites", "coverage"]]
        logger.info("Top 10 pathways by coverage:")
        for _, r in top.iterrows():
            logger.info(f"  {r['pathway_name']}: "
                        f"{int(r['n_matched_metabolites'])}/{int(r['n_metabolites'])} "
                        f"metabolites ({r['coverage']:.1%})")

    # ------------------------------------------------------------------
    # Stage 2: z-scores against the normal reference
    # ------------------------------------------------------------------
    if not bool(config.get("run_zscores", True)):
        logger.info("run_zscores is false; stopping after the mapping outputs.")
        return {
            "feature_to_hmdb": feature_to_hmdb,
            "feature_to_pathway": feature_to_pathway,
            "pathway_coverage": coverage,
        }

    _log_section("STEP 5: Compute metabolite z-scores (normals as reference)")
    normal_mask = classify_samples(
        metadata,
        normal_classification=int(config.get("normal_classification", 0)),
        normal_oordeel=int(config.get("normal_oordeel", 0)),
    )
    if not normal_mask.any():
        logger.error("No normal reference samples; cannot compute z-scores.")
        return {
            "feature_to_hmdb": feature_to_hmdb,
            "feature_to_pathway": feature_to_pathway,
            "pathway_coverage": coverage,
        }

    # Only features mapped to a kept pathway can contribute to pathway scores;
    # keep the whole matrix out of scope here.
    pathway_features = sorted(
        set(coverage["matched_features"].str.split(";").explode().dropna())
    ) if not coverage.empty else []
    logger.info(f"Z-scoring {len(pathway_features)} pathway-mapped features "
                f"(of {features.shape[1]} total).")
    features_scored = features[pathway_features]

    zscores, reference_stats, dropped_features = compute_metabolite_zscores(
        features_scored,
        normal_mask=normal_mask,
        iqr_scale=bool(config.get("iqr_scale", True)),
    )

    if bool(config.get("save_zscore_outputs", True)):
        zscores.to_csv(out / "metabolite_zscores.csv")
        reference_stats.to_csv(out / "reference_stats.csv", index=False)
        dropped_features.to_csv(out / "dropped_features.csv", index=False)
        logger.info(f"Wrote metabolite_zscores.csv, reference_stats.csv, "
                    f"dropped_features.csv to {out}")

    # ------------------------------------------------------------------
    # Stage 2b: restrict pathways to calibrated features
    # ------------------------------------------------------------------
    _log_section("STEP 6: Restrict pathways to calibrated features")
    min_pathway_features = int(config.get("min_pathway_features", 3))
    scored_coverage = filter_pathways_for_scoring(
        coverage,
        feature_to_pathway,
        available_features=zscores.columns,
        min_pathway_features=min_pathway_features,
    )
    logger.info(f"{len(scored_coverage)} pathways remain with >= "
                f"{min_pathway_features} usable matched metabolites.")

    if bool(config.get("save_zscore_outputs", True)) and not scored_coverage.empty:
        scored_coverage.to_csv(out / "pathway_coverage_scored.csv", index=False)
        logger.info("Wrote pathway_coverage_scored.csv to "
                    f"{out}")

    # ------------------------------------------------------------------
    # Stage 3: pathway Stouffer scores
    # ------------------------------------------------------------------
    if not bool(config.get("run_stouffer", True)):
        logger.info("run_stouffer is false; stopping after the z-score outputs.")
        return {
            "feature_to_hmdb": feature_to_hmdb,
            "feature_to_pathway": feature_to_pathway,
            "pathway_coverage": coverage,
            "normal_mask": normal_mask,
            "zscores": zscores,
            "reference_stats": reference_stats,
            "dropped_features": dropped_features,
            "pathway_coverage_scored": scored_coverage,
        }

    _log_section("STEP 7: Pathway Stouffer scores")
    min_metabolites = int(config.get("min_stouffer_metabolites", 3))
    pathway_scores, pathway_reference = compute_stouffer_scores(
        zscores,
        feature_to_pathway=feature_to_pathway,
        scored_coverage=scored_coverage,
        normal_mask=normal_mask,
        min_metabolites=min_metabolites,
    )

    if bool(config.get("save_stouffer_outputs", True)):
        pathway_scores.to_csv(out / "pathway_stouffer_scores.csv", index=False)
        pathway_reference.to_csv(out / "pathway_stouffer_reference.csv", index=False)
        logger.info(f"Wrote pathway_stouffer_scores.csv, "
                    f"pathway_stouffer_reference.csv to {out}")

    if not pathway_scores.empty:
        normals_in_scores = normal_mask.reindex(
            pathway_scores["sample_id"].unique(), fill_value=False)
        n_normal = int(normals_in_scores.sum())
        logger.info(f"Stouffer scores cover "
                    f"{pathway_scores['sample_id'].nunique()} samples x "
                    f"{pathway_scores['smp_id'].nunique()} pathways "
                    f"({n_normal} normals in the reference).")

    # ------------------------------------------------------------------
    # Stage 4: flag samples against each pathway's own normal range
    # ------------------------------------------------------------------
    if not bool(config.get("run_flagging", True)):
        logger.info("run_flagging is false; stopping after the Stouffer outputs.")
        return {
            "feature_to_hmdb": feature_to_hmdb,
            "feature_to_pathway": feature_to_pathway,
            "pathway_coverage": coverage,
            "normal_mask": normal_mask,
            "zscores": zscores,
            "reference_stats": reference_stats,
            "dropped_features": dropped_features,
            "pathway_coverage_scored": scored_coverage,
            "pathway_scores": pathway_scores,
            "pathway_reference": pathway_reference,
        }

    _log_section("STEP 8: Flag samples per pathway (normal-percentile thresholds)")
    threshold_percentile = float(config.get("flag_threshold_percentile", 99.0))
    pathway_flags = flag_pathway_scores(
        pathway_scores,
        normal_mask=normal_mask,
        threshold_percentile=threshold_percentile,
    )

    min_flagged_pathways = int(config.get("min_flagged_pathways", 1))
    sample_decisions = summarize_sample_flags(
        pathway_flags,
        min_flagged_pathways=min_flagged_pathways,
    )

    sample_group = _label_samples(metadata, normal_mask)
    decisions_labeled = sample_decisions.merge(
        sample_group.rename("group"), left_on="sample_id", right_index=True,
        how="left")
    decisions_labeled["group"] = decisions_labeled["group"].fillna("other")

    if not decisions_labeled.empty:
        group_counts = (decisions_labeled
                        .groupby("group")
                        .agg(n_samples=("flagged", "size"),
                             n_flagged=("flagged", "sum"))
                        .reset_index())
        for _, r in group_counts.iterrows():
            logger.info(f"Group '{r['group']}': {int(r['n_flagged'])} of "
                        f"{int(r['n_samples'])} samples flagged.")

    if bool(config.get("save_flagging_outputs", True)):
        pathway_flags.to_csv(out / "pathway_flags.csv", index=False)
        decisions_labeled.to_csv(out / "sample_decisions.csv", index=False)
        logger.info(f"Wrote pathway_flags.csv, sample_decisions.csv to {out}")

    return {
        "feature_to_hmdb": feature_to_hmdb,
        "feature_to_pathway": feature_to_pathway,
        "pathway_coverage": coverage,
        "normal_mask": normal_mask,
        "zscores": zscores,
        "reference_stats": reference_stats,
        "dropped_features": dropped_features,
        "pathway_coverage_scored": scored_coverage,
        "pathway_scores": pathway_scores,
        "pathway_reference": pathway_reference,
        "pathway_flags": pathway_flags,
        "sample_decisions": decisions_labeled,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pathway pipeline feature engineering: feature -> HMDB "
                    "-> PathBank pathway mapping with coverage filtering."
    )
    parser.add_argument("--input", default=None,
                        help="Path to the feature matrix CSV.")
    parser.add_argument("--output", default="outputs/pathway_pipeline",
                        help="Output directory (default: outputs/pathway_pipeline).")
    parser.add_argument("--config", default=None,
                        help="Path to config YAML (default: pathway_pipeline/config/config.yaml).")
    args = parser.parse_args()

    if args.input is None:
        config = Config(args.config)
        args.input = config.get("input_file", "data/merged_data_with_classification.csv")

    try:
        run_pipeline(input_file=args.input, output_dir=args.output,
                     config_path=args.config)
        logger.info("\nPathway pipeline feature engineering completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
