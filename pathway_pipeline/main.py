#!/usr/bin/env python3
"""Main entry point for the pathway pipeline.

There is NO outlier-detection model here. The pipeline runs a layered z-based
analysis, escalating in statistical complexity only where the simpler layer
fails:

  Layer 1: per-metabolite z-scores (age-adjusted, robustly scaled) + atomic
           single-metabolite overrides.
  Layer 2: pathway Z_med / F / Z_up / Z_down statistics with moderate/severe
           severity tiers (the primary detector).
  Layer 3: the sample decision rule (>=1 severe OR >=2 moderate pathway flags,
           plus single-metabolite overrides).
  Layer 4: optional global anomaly score (the "odd sample" safety light).

Stage 1 (preprocessing): match every feature column to an HMDB accession from
the HMDB XML file, then link those accessions to pathways via the SMPDB-derived
pathways TSV. Writes the feature->HMDB, feature->pathway, and pathway-coverage
tables.

Stage 2 (statistics, optional): the four layered analysis & flagging steps
above, plus an optional threshold-tuning sweep on the inner IMD split.

Usage:
    python -m pathway_pipeline.main
    python -m pathway_pipeline.main --input data/my_data.csv --output outputs/pathway
    python -m pathway_pipeline.main --config my_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from pathway_pipeline.config.config import Config
from pathway_pipeline.pipeline.hmdb_parser import build_name_index
from pathway_pipeline.pipeline.pathway_mapping import (
    load_pathways_tsv,
    match_features_to_hmdb,
    link_features_to_pathways,
    pathway_coverage,
)
from pathway_pipeline.pipeline.pathway_stats import (
    compute_metabolite_zscores,
    compute_pathway_statistics,
    flag_pathways,
    flag_metabolites,
    compute_global_anomaly_score,
    decide_samples,
    tune_decision_thresholds,
)
from pathway_pipeline.pipeline.pathway_stats_enhanced import (
    compute_enhanced_pathway_statistics,
    flag_pathways_enhanced,
    compute_weighted_decision_score,
    decide_samples_enhanced,
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

    Keeps every configured non-feature column in a separate ``metadata`` frame
    (for the normal-reference definition) and, when ``age_column`` is set and
    present, returns a numeric ``ages`` Series for age-adjusted z-scores. The
    age column is NOT included in the feature matrix even if it is not listed
    in ``non_feature_columns``.
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


def _normal_reference_mask(metadata: pd.DataFrame, scheme: str) -> pd.Series:
    """Boolean Series over the sample index: True = normal reference set.

    The normal reference defines the per-metabolite median/IQR and the
    per-metabolite empirical threshold for the flagged fraction. Mirrors the
    outlier-detection pipeline's label schemes but only needs the normal mask.
    """
    idx = metadata.index
    if metadata.empty:
        return pd.Series(np.ones(len(idx), dtype=bool), index=idx)

    if "Classification" in metadata.columns and "Oordeel targeted" in metadata.columns:
        cls = pd.to_numeric(metadata["Classification"], errors="coerce")
        oor = pd.to_numeric(metadata["Oordeel targeted"], errors="coerce")
        if scheme == "confident_normals":
            return pd.Series((cls == 0) & (oor == 0), index=idx)
        if scheme == "oordeel":
            return pd.Series(oor == 0, index=idx)
        if scheme == "binary_simplified":
            return pd.Series(cls.isin([0, 3]), index=idx)
        if scheme == "class1_imd":
            # Normals = Class 0 AND Oordeel 0.
            # Class 1 = IMD (regardless of Oordeel).
            # Class 2/3 or (Class 0 AND Oordeel 1) = gray (not normal, not IMD).
            return pd.Series((cls == 0) & (oor == 0), index=idx)
        # default: Class 0 (after the pipeline's Oordeel reconciliation)
        return pd.Series(cls == 0, index=idx)
    if "Classification" in metadata.columns:
        cls = pd.to_numeric(metadata["Classification"], errors="coerce")
        return pd.Series(cls == 0, index=idx)
    if "Oordeel targeted" in metadata.columns:
        oor = pd.to_numeric(metadata["Oordeel targeted"], errors="coerce")
        return pd.Series(oor == 0, index=idx)
    # No metadata to define normals: use all samples as the reference.
    logger.warning("No Classification/Oordeel columns found; using all samples "
                   "as the normal reference.")
    return pd.Series(np.ones(len(idx), dtype=bool), index=idx)


def _imd_labels(metadata: pd.DataFrame, scheme: str) -> pd.Series:
    """Binary 0/1 IMD label per sample aligned to ``metadata.index``.

    Used only by the threshold-tuning step (inner IMD split). Mirrors the
    outlier-detection pipeline's label schemes: 1 = IMD (the class the
    decision rule is tuned to detect), 0 = normal reference. Samples with an
    undefined role get 0.
    """
    idx = metadata.index
    if metadata.empty or "Classification" not in metadata.columns:
        return pd.Series(np.zeros(len(idx), dtype=int), index=idx)
    cls = pd.to_numeric(metadata["Classification"], errors="coerce")
    if "Oordeel targeted" in metadata.columns and scheme == "confident_normals":
        oor = pd.to_numeric(metadata["Oordeel targeted"], errors="coerce")
        # Confident normal = (Class 0 AND Oordeel 0); everything else is IMD-ish.
        return pd.Series(np.where((cls == 0) & (oor == 0), 0, 1), index=idx)
    if scheme == "class1_imd":
        # Class 1 = IMD (regardless of Oordeel). Everything else = 0.
        return pd.Series(np.where(cls == 1, 1, 0), index=idx)
    if scheme == "oordeel" and "Oordeel targeted" in metadata.columns:
        oor = pd.to_numeric(metadata["Oordeel targeted"], errors="coerce")
        return pd.Series(np.where(oor == 1, 1, 0), index=idx)
    if scheme == "binary_simplified":
        return pd.Series(np.where(cls == 1, 1, 0), index=idx)
    return pd.Series(np.where(cls == 1, 1, 0), index=idx)


def run_pipeline(input_file: str,
                  output_dir: str = "outputs/pathway_pipeline",
                  config_path: str = None) -> dict:
    """Run the pathway pipeline (preprocessing + optional statistics)."""
    config = Config(config_path) if config_path else Config()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _setup_logging(out)

    _log_section("PATHWAY PIPELINE")
    logger.info(f"Input: {input_file}\nOutput: {out}")

    # ------------------------------------------------------------------
    # Stage 1: preprocessing -- feature -> HMDB -> pathway
    # ------------------------------------------------------------------
    _log_section("STEP 1: Load feature matrix")
    age_column = config.get("age_column", None)
    features, metadata, ages = load_feature_matrix(
        input_file,
        non_feature_columns=config.get_list(
            "non_feature_columns", ["Oordeel trageted", "Classification"]
        ),
        patient_id_column=config.get("patient_id_column", None),
        age_column=age_column,
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

    _log_section("STEP 4: Load pathways and link features to pathways")
    pathways = load_pathways_tsv(config.get("pathways_file", "data/pathways.tsv"))
    feature_to_pathway = link_features_to_pathways(feature_to_hmdb, pathways)
    min_pathway_size = int(config.get("min_pathway_size", 3))
    coverage = pathway_coverage(feature_to_pathway,
                                min_pathway_size=min_pathway_size)

    save_mapping = bool(config.get("save_mapping_outputs", True))
    if save_mapping:
        feature_to_hmdb.to_csv(out / "feature_to_hmdb.csv", index=False)
        feature_to_pathway.to_csv(out / "feature_to_pathway.csv", index=False)
        coverage.to_csv(out / "pathway_coverage.csv", index=False)
        logger.info(f"Wrote feature_to_hmdb.csv, feature_to_pathway.csv, "
                    f"pathway_coverage.csv to {out}")

    n_matched = int(feature_to_hmdb["hmdb_id"].notna().sum())
    logger.info(f"Matched {n_matched} of {len(features.columns)} features to HMDB; "
                f"{coverage.shape[0]} pathways with >={min_pathway_size} matched features.")

    results = {
        "feature_to_hmdb": feature_to_hmdb,
        "feature_to_pathway": feature_to_pathway,
        "pathway_coverage": coverage,
    }

    # ------------------------------------------------------------------
    # Stage 2: layered analysis & flagging (no outlier-detection model)
    #   Layer 1: per-metabolite (age-adjusted, robust) z-scores + atomic flags
    #   Layer 2: pathway Z_med / F / Z_up / Z_down statistics + severity tiers
    #   Layer 3: sample decision rule (>=1 severe OR >=2 moderate + overrides)
    #   Layer 4: optional global anomaly score (the "odd sample" safety light)
    # ------------------------------------------------------------------
    if not bool(config.get("run_stats", True)):
        logger.info("run_stats is false; stopping after the mapping outputs.")
        return results

    _log_section("STEP 5: Layered z-score analysis & flagging")
    normal_mask = _normal_reference_mask(
        metadata, config.get("classification_scheme", "confident_normals")
    )
    n_normal = int(normal_mask.sum())
    logger.info(f"Normal reference set: {n_normal} of {len(normal_mask)} samples")
    if n_normal < 2:
        logger.warning("Fewer than 2 normal reference samples; cannot compute "
                       "robust median/IQR. Skipping statistics.")
        return results

    # Restrict the feature matrix to features that participate in at least one
    # pathway (with >= min_pathway_size members) so the z-score matrix and the
    # pathway-feature map are aligned and the statistics are not wasted on
    # unmatched metabolites.
    pathway_features = sorted(set(coverage.get("matched_features", pd.Series(dtype=str))
                                  .str.split(";").explode().dropna()))
    if not pathway_features:
        logger.warning("No matched pathway features; cannot compute pathway "
                       "statistics.")
        return results
    sub_features = features[pathway_features]
    zscores = compute_metabolite_zscores(
        sub_features, normal_mask=normal_mask,
        iqr_scale=bool(config.get("iqr_scale", True)),
        ages=ages,
        age_adjustment_method=config.get("age_adjustment_method", "ols"),
        age_loess_frac=float(config.get("age_loess_frac", 0.5)),
    )
    zscores.to_csv(out / "metabolite_zscores.csv")

    flag_percentile = float(config.get("flag_percentile", 99))

    # --- Layer 1: atomic per-metabolite flags (single-metabolite overrides) ---
    override_thr = float(config.get("metabolite_override_threshold", 6.0))
    metabolite_flags = flag_metabolites(zscores, normal_mask=normal_mask,
                                        override_threshold=override_thr,
                                        flag_percentile=flag_percentile)
    metabolite_flags.to_csv(out / "metabolite_flags.csv", index=False)
    logger.info(f"Layer 1 (atomic metabolites): flagged {len(metabolite_flags)} "
                f"(sample, metabolite) pairs at |z| > {override_thr}.")

    # --- Check if enhanced statistics are enabled ---
    use_enhanced = bool(config.get("use_enhanced_stats", False))

    if use_enhanced:
        _log_section("STEP 5: Enhanced pathway analysis & flagging")
        logger.info("Running enhanced pipeline with Stouffer's Z, multiple testing "
                    "correction, and weighted decision scores...")

        # Enhanced pathway statistics
        enhanced_stats = compute_enhanced_pathway_statistics(
            zscores=zscores,
            feature_to_pathway=feature_to_pathway,
            normal_mask=normal_mask,
            min_pathway_size=min_pathway_size,
            output_dir=out,
        )

        # Enhanced pathway flagging
        enhanced_flags = flag_pathways_enhanced(
            enhanced_stats,
            zmed_threshold=float(config.get("zmed_threshold", 2.0)),
            stouffer_z_threshold=float(config.get("stouffer_z_threshold", 3.0)),
            p_stouffer_threshold=float(config.get("p_stouffer_threshold", 0.001)),
            p_bonferroni_threshold=float(config.get("p_bonferroni_threshold", 0.05)),
            p_fdr_threshold=float(config.get("p_fdr_threshold", 0.05)),
            use_empirical=bool(config.get("use_empirical_thresholds", True)),
        )
        enhanced_flags.to_csv(out / "enhanced_pathway_flags.csv", index=False)
        logger.info(f"Enhanced Layer 2 (pathways): {int(enhanced_flags['flagged_two_stage'].sum())} "
                    f"flagged via two-stage method.")

        # Weighted decision scores
        weighted_scores = compute_weighted_decision_score(
            enhanced_flags,
            weight_method=config.get("weight_method", "stouffer"),
            use_log=bool(config.get("use_log_weights", True)),
        )

        # Enhanced decision rule
        score_threshold = config.get_float("score_threshold", None)
        min_flagged = int(config.get("min_flagged_pathways", 3))
        min_w = float(config.get("min_weight", 2.0))

        enhanced_decision = decide_samples_enhanced(
            pathway_stats=enhanced_flags,
            weighted_scores=weighted_scores,
            metabolite_flags=metabolite_flags,
            global_scores=None,
            score_threshold=score_threshold,
            min_flagged_pathways=min_flagged,
            min_weight=min_w,
            output_dir=out,
        )
        enhanced_decision.to_csv(out / "enhanced_sample_decisions.csv")
        logger.info(f"Enhanced Layer 3 (decision rule): flagged "
                    f"{int(enhanced_decision['flagged'].sum())} of "
                    f"{len(enhanced_decision)} samples.")

        results["enhanced_pathway_statistics"] = enhanced_stats
        results["enhanced_pathway_flags"] = enhanced_flags
        results["enhanced_weighted_scores"] = weighted_scores
        results["enhanced_sample_decisions"] = enhanced_decision

    # --- Original (or fallback) pipeline ---
    stats = compute_pathway_statistics(
        zscores=zscores,
        feature_to_pathway=feature_to_pathway,
        normal_mask=normal_mask,
        flag_percentile=flag_percentile,
        min_pathway_size=min_pathway_size,
    )
    pathway_flags = flag_pathways(
        stats,
        zmed_threshold=float(config.get("zmed_threshold", 2.0)),
        flagged_fraction_threshold=float(config.get("flagged_fraction_threshold", 0.5)),
        signed_extreme_threshold=float(config.get("signed_extreme_threshold", 2.5)),
        severe_zmed_threshold=float(config.get("severe_zmed_threshold", 3.0)),
        severe_flagged_fraction_threshold=float(
            config.get("severe_flagged_fraction_threshold", 0.7)),
        severe_signed_extreme_threshold=float(
            config.get("severe_signed_extreme_threshold", 4.0)),
    )
    pathway_flags.to_csv(out / "pathway_statistics.csv", index=False)
    if not pathway_flags.empty:
        pivot = (pathway_flags.pivot_table(index="sample_id", columns="pathway_name",
                                            values="z_med", aggfunc="first"))
        pivot.to_csv(out / "pathway_zmed_pivot.csv")
    logger.info(f"Layer 2 (pathways): {int(pathway_flags['flagged'].sum())} flagged "
                f"({int((pathway_flags['severity'] == 'severe').sum())} severe, "
                f"{int((pathway_flags['severity'] == 'moderate').sum())} moderate).")

    # --- Layer 4: optional global anomaly score (safety light) ---
    global_scores = compute_global_anomaly_score(
        zscores, top_k=int(config.get("global_anomaly_top_k", 10))
    )
    global_scores.to_csv(out / "global_anomaly_scores.csv")
    global_threshold = config.get("global_threshold", None)
    if global_threshold is not None:
        global_threshold = float(global_threshold)

    # --- Layer 3: sample decision rule (the operating point) ---
    decision = decide_samples(
        pathway_flags=pathway_flags,
        metabolite_flags=metabolite_flags,
        global_scores=global_scores,
        min_moderate=int(config.get("min_moderate_pathways", 2)),
        min_severe=int(config.get("min_severe_pathways", 1)),
        global_threshold=global_threshold,
        min_severe_zmed=config.get_float("min_severe_zmed", None),
        min_moderate_zmed=config.get_float("min_moderate_zmed", None),
    )
    decision.to_csv(out / "sample_decisions.csv")
    logger.info(f"Layer 3 (decision rule): flagged {int(decision['flagged'].sum())} "
                f"of {len(decision)} samples.")

    # --- Optional: tune the operating point on the inner IMD split ---
    if bool(config.get("run_threshold_tuning", False)):
        _log_section("STEP 6: Threshold tuning (inner IMD split)")
        labels = _imd_labels(metadata,
                             config.get("classification_scheme", "confident_normals"))
        sweep = tune_decision_thresholds(
            pathway_stats=stats,
            zscores=zscores,
            normal_mask=normal_mask,
            labels=labels,
            metabolite_override_grid=config.get_list(
                "tuning_override_grid", [4.0, 5.0, 6.0, 8.0, 10.0]),
            moderate_zmed_grid=config.get_list(
                "tuning_moderate_zmed_grid", [1.5, 2.0, 2.5, 3.0]),
            severe_zmed_grid=config.get_list(
                "tuning_severe_zmed_grid", [2.5, 3.0, 3.5, 4.0]),
            flag_percentile=flag_percentile,
            signed_extreme_grid=config.get_list(
                "tuning_signed_extreme_grid", [2.5, 3.0, 4.0]),
            prevalence=float(config.get("tuning_prevalence", 0.02)),
            metric=config.get("tuning_metric", "f1"),
        )
        sweep.to_csv(out / "threshold_tuning.csv", index=False)
        logger.info(f"Wrote threshold_tuning.csv ({len(sweep)} settings) to {out}")
        results["threshold_tuning"] = sweep

    logger.info(f"Wrote metabolite_zscores.csv, metabolite_flags.csv, "
                f"pathway_statistics.csv, global_anomaly_scores.csv, "
                f"sample_decisions.csv to {out}")
    results["metabolite_zscores"] = zscores
    results["metabolite_flags"] = metabolite_flags
    results["pathway_statistics"] = pathway_flags
    results["global_anomaly_scores"] = global_scores
    results["sample_decisions"] = decision
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Pathway-shift pipeline: feature -> HMDB -> pathway "
                    "matching and direction-aware pathway statistics."
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
        logger.info("\nPathway pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
