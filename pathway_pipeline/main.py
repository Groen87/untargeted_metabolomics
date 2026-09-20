#!/usr/bin/env python3
"""Main entry point for the pathway pipeline.

This is a COMPLETELY REWRITTEN pipeline that implements a clean, simple approach:

1. Map features to pathways, removing unmapped features
2. Calculate Z-scores for all samples using normals as reference
3. Combine Z-scores into compound scores per pathway using absolute Stouffer's Z
4. Find optimal cutoffs empirically from the normal distribution
5. Flag samples based on extreme pathway deviations

Key design decisions:
- Uses absolute Stouffer's Z to detect ANY disturbance regardless of direction
  (critical for IMD detection where a block can cause both accumulation and depletion)
- Only flags samples with 1-2 extremely deviated pathways (IMD pattern)
- Uses empirical thresholds computed from the actual normal distribution
- Filters to only normals + IMDs (Class 1 AND Oordeel 1) for analysis
- Removes all features that don't map to pathways

This replaces the buggy enhanced pipeline completely.
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


def compute_metabolite_zscores(
    features: pd.DataFrame,
    normal_mask: pd.Series,
    iqr_scale: bool = True,
    ages: pd.Series = None,
    age_adjustment_method: str = "ols",
    age_loess_frac: float = 0.5,
) -> pd.DataFrame:
    """Compute robust z-scores for metabolites.
    
    Uses median/IQR scaling on the normal reference set.
    If ages are provided, performs age adjustment first.
    
    Args:
        features: DataFrame with samples as rows, features as columns.
        normal_mask: Boolean Series indicating normal samples.
        iqr_scale: If True, scale by IQR; otherwise by std.
        ages: Optional Series of ages for age adjustment.
        age_adjustment_method: 'ols' or 'loess'
        age_loess_frac: LOESS bandwidth fraction (only used if method='loess')
        
    Returns:
        DataFrame of z-scores with same shape as features.
    """
    if features.empty:
        return pd.DataFrame()
    
    zscores = features.copy()
    
    # Age adjustment if ages are provided
    if ages is not None and age_adjustment_method is not None:
        normal_ages = ages[normal_mask]
        normal_features = features.loc[normal_mask]
        
        if age_adjustment_method == "ols":
            # Per-metabolite linear regression on age
            for col in features.columns:
                y = normal_features[col].to_numpy(dtype=float)
                x = normal_ages.to_numpy(dtype=float)
                
                # Remove NaN pairs
                valid = ~(np.isnan(x) | np.isnan(y))
                if valid.sum() < 2:
                    continue
                    
                x_valid = x[valid]
                y_valid = y[valid]
                
                # Fit OLS: y = a + b*x
                A = np.vstack([np.ones(len(x_valid)), x_valid]).T
                b, a = np.linalg.lstsq(A, y_valid, rcond=None)[0]
                
                # Get residuals for all samples with valid ages
                mask_valid_ages = ~ages.isna()
                x_all = ages[mask_valid_ages].to_numpy(dtype=float)
                y_all = features.loc[mask_valid_ages, col].to_numpy(dtype=float)
                residuals = y_all - (a + b * x_all)
                
                # For samples without age, use original values
                if (~mask_valid_ages).any():
                    y_no_age = features.loc[~mask_valid_ages, col].to_numpy(dtype=float)
                    residuals_full = np.concatenate([residuals, y_no_age])
                else:
                    residuals_full = residuals
                
                zscores[col] = residuals_full
        elif age_adjustment_method == "loess":
            try:
                import statsmodels.api as sm
                for col in features.columns:
                    y = normal_features[col].to_numpy(dtype=float)
                    x = normal_ages.to_numpy(dtype=float)
                    
                    valid = ~(np.isnan(x) | np.isnan(y))
                    if valid.sum() < 2:
                        continue
                        
                    x_valid = x[valid]
                    y_valid = y[valid]
                    
                    # LOESS smoothing
                    lowess = sm.nonparametric.lowess(y_valid, x_valid, frac=age_loess_frac)
                    
                    # Interpolate to get fitted values
                    if len(lowess) > 1:
                        import scipy.interpolate
                        interp = scipy.interpolate.interp1d(
                            lowess[:, 0], lowess[:, 1],
                            bounds_error=False, fill_value="extrapolate"
                        )
                        fitted = interp(x_valid)
                        residuals = y_valid - fitted
                    else:
                        residuals = y_valid - y_valid.mean()
                    
                    # Apply to all samples
                    mask_valid_ages = ~ages.isna()
                    x_all = ages[mask_valid_ages].to_numpy(dtype=float)
                    y_all = features.loc[mask_valid_ages, col].to_numpy(dtype=float)
                    
                    if len(lowess) > 1:
                        fitted_all = interp(x_all)
                        residuals_all = y_all - fitted_all
                    else:
                        residuals_all = y_all - y_all.mean()
                    
                    # For samples without age
                    if (~mask_valid_ages).any():
                        y_no_age = features.loc[~mask_valid_ages, col].to_numpy(dtype=float)
                        residuals_full = np.concatenate([residuals_all, y_no_age])
                    else:
                        residuals_full = residuals_all
                    
                    zscores[col] = residuals_full
            except ImportError:
                logger.warning("statsmodels not available for LOESS; using raw values")
    
    # Robust scaling (median/IQR) on normal reference
    normal_values = zscores.loc[normal_mask]
    
    for col in zscores.columns:
        norm_col = normal_values[col].dropna()
        if len(norm_col) < 1:
            # Not enough normal data; use all data
            all_col = zscores[col].dropna()
            if len(all_col) < 1:
                zscores[col] = 0.0
                continue
            median = np.median(all_col)
            if iqr_scale:
                iqr = np.percentile(all_col, 75) - np.percentile(all_col, 25)
                if iqr == 0:
                    zscores[col] = 0.0
                else:
                    zscores[col] = (zscores[col] - median) / iqr
            else:
                std = np.std(all_col)
                if std == 0:
                    zscores[col] = 0.0
                else:
                    zscores[col] = (zscores[col] - median) / std
        else:
            median = np.median(norm_col)
            if iqr_scale:
                iqr = np.percentile(norm_col, 75) - np.percentile(norm_col, 25)
                if iqr == 0:
                    zscores[col] = 0.0
                else:
                    zscores[col] = (zscores[col] - median) / iqr
            else:
                std = np.std(norm_col)
                if std == 0:
                    zscores[col] = 0.0
                else:
                    zscores[col] = (zscores[col] - median) / std
    
    # Drop features with zero scale
    dropped = []
    for col in zscores.columns:
        if (zscores[col] == 0.0).all():
            dropped.append(col)
    
    if dropped:
        logger.info(f"Dropped {len(dropped)} metabolites with zero scale")
        zscores = zscores.drop(columns=dropped)
    
    return zscores


def run_pipeline(input_file: str,
                  output_dir: str = "outputs/pathway_pipeline",
                  config_path: str = None) -> dict:
    """Run the clean pathway pipeline."""
    config = Config(config_path) if config_path else Config()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _setup_logging(out)

    # Suppress numpy RuntimeWarnings if configured
    if bool(config.get("suppress_numpy_warnings", True)):
        import warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)

    _log_section("CLEAN PATHWAY PIPELINE")
    logger.info(f"Input: {input_file}\nOutput: {out}")

    # ------------------------------------------------------------------
    # Stage 1: preprocessing -- feature -> HMDB -> pathway
    # ------------------------------------------------------------------
    _log_section("STEP 1: Load feature matrix")
    age_column = config.get("age_column", None)
    features, metadata, ages = load_feature_matrix(
        input_file,
        non_feature_columns=config.get_list(
            "non_feature_columns", ["Oordeel targeted", "Classification"]
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
    # Stage 2: Clean pathway analysis
    # ------------------------------------------------------------------
    if not bool(config.get("run_stats", True)):
        logger.info("run_stats is false; stopping after the mapping outputs.")
        return results

    _log_section("STEP 5: Clean pathway analysis")
    
    # Classify samples
    if "Classification" in metadata.columns and "Oordeel targeted" in metadata.columns:
        cls = pd.to_numeric(metadata["Classification"], errors="coerce")
        oor = pd.to_numeric(metadata["Oordeel targeted"], errors="coerce")
        
        # Normals = Class 0 AND Oordeel 0
        # IMD = Class 1 AND Oordeel 1
        # Gray = Everything else
        normal_mask = (cls == 0) & (oor == 0)
        imd_mask = (cls == 1) & (oor == 1)
        gray_mask = ~normal_mask & ~imd_mask
    else:
        logger.error("Classification and Oordeel targeted columns are required")
        return results
    
    normal_sample_ids = metadata.index[normal_mask].tolist()
    imd_sample_ids = metadata.index[imd_mask].tolist()
    gray_sample_ids = metadata.index[gray_mask].tolist()
    
    logger.info(f"Sample classification:")
    logger.info(f"  Normals: {len(normal_sample_ids)}")
    logger.info(f"  IMDs: {len(imd_sample_ids)}")
    logger.info(f"  Gray: {len(gray_sample_ids)}")
    
    # Filter to only normals + IMDs (exclude gray)
    # Use boolean indexing to avoid duplicate sample IDs
    analysis_mask = normal_mask | imd_mask
    analysis_sample_ids = metadata.index[analysis_mask].unique().tolist()
    
    logger.info(f"Analyzing {len(analysis_sample_ids)} samples "
                f"({len(normal_sample_ids)} normals + {len(imd_sample_ids)} IMDs)")
    
    # Filter features to only those that map to pathways
    pathway_features = sorted(set(coverage.get("matched_features", pd.Series(dtype=str))
                                  .str.split(";").explode().dropna()))
    
    if not pathway_features:
        logger.warning("No matched pathway features; cannot compute pathway statistics.")
        return results
    
    # Get only features that are in the feature_to_pathway mapping
    features_filtered = features[pathway_features]
    
    # Filter features and metadata to analysis samples
    features_filtered = features_filtered.loc[analysis_mask]
    metadata_filtered = metadata.loc[analysis_mask]
    normal_mask_filtered = normal_mask.loc[analysis_mask]
    
    logger.info(f"Using {len(pathway_features)} features that map to pathways")
    logger.info(f"Analyzing {len(analysis_sample_ids)} samples "
                f"({len(normal_sample_ids)} normals + {len(imd_sample_ids)} IMDs)")
    
    # Compute z-scores using normals as reference
    zscores = compute_metabolite_zscores(
        features_filtered,
        normal_mask=normal_mask_filtered,
        iqr_scale=bool(config.get("iqr_scale", True)),
        ages=ages.loc[analysis_mask] if ages is not None else None,
        age_adjustment_method=config.get("age_adjustment_method", "ols"),
        age_loess_frac=float(config.get("age_loess_frac", 0.5)),
    )
    
    zscores.to_csv(out / "metabolite_zscores.csv")
    logger.info(f"Computed {len(zscores.columns)} metabolite z-scores over {len(zscores)} samples")

    # ------------------------------------------------------------------
    # Clean pathway analysis with absolute Stouffer's Z
    # ------------------------------------------------------------------
    from pathway_pipeline.pipeline.pathway_analysis_clean import (
        compute_pathway_stouffers_z,
        find_optimal_threshold,
        flag_samples,
        validate_flagging,
    )
    
    # Filter feature_to_pathway to only include features in our zscores
    feature_to_pathway_filtered = feature_to_pathway[
        feature_to_pathway['feature'].isin(zscores.columns)
    ]
    
    # Compute pathway Stouffer's Z-scores
    pathway_stats = compute_pathway_stouffers_z(
        zscores,
        feature_to_pathway_filtered,
        min_pathway_size=min_pathway_size
    )
    
    logger.info(f"Computed Stouffer's Z for {pathway_stats['pathway_name'].nunique()} pathways "
                f"across {pathway_stats['sample_id'].nunique()} samples")
    
    # Get sample IDs from the filtered zscores (which has unique index)
    normal_sample_ids_filtered = metadata_filtered.index[normal_mask_filtered].tolist()
    imd_sample_ids_filtered = metadata_filtered.index[~normal_mask_filtered].tolist()
    
    # Find optimal threshold
    threshold_info = find_optimal_threshold(
        pathway_stats,
        normal_sample_ids_filtered,
        imd_sample_ids_filtered,
        min_detection=float(config.get("min_detection", 0.80)),
        max_contamination=float(config.get("max_contamination", 0.05)),
        min_flagged_pathways=int(config.get("min_flagged_pathways", 1)),
        n_thresholds=int(config.get("n_thresholds", 20)),
    )
    
    logger.info(f"\nOptimal threshold: {threshold_info['optimal_threshold']:.2f} "
                f"(percentile {threshold_info['percentile']:.4f})")
    logger.info(f"Detection rate: {threshold_info['detection_rate']*100:.1f}%")
    logger.info(f"Contamination rate: {threshold_info['contamination_rate']*100:.1f}%")
    logger.info(f"Normals flagged: {threshold_info['n_normals_flagged']}")
    logger.info(f"IMDs flagged: {threshold_info['n_imds_flagged']}")
    
    # Flag samples using optimal threshold
    decisions = flag_samples(
        pathway_stats,
        threshold=threshold_info['optimal_threshold'],
        min_flagged_pathways=int(config.get("min_flagged_pathways", 1))
    )
    
    # Validate
    validation = validate_flagging(
        decisions,
        normal_sample_ids_filtered,
        imd_sample_ids_filtered
    )
    
    logger.info(f"\nValidation results:")
    logger.info(f"  Normals flagged: {validation['normals_flagged']} / {validation['n_normals']} "
                f"({validation['contamination_rate']*100:.1f}%)")
    logger.info(f"  IMDs flagged: {validation['imds_flagged']} / {validation['n_imds']} "
                f"({validation['detection_rate']*100:.1f}%)")
    
    if validation['normals_flagged'] > 0:
        logger.warning(f"WARNING: {validation['normals_flagged']} normal samples were flagged!")
        if validation['contamination_rate'] > 0.05:
            logger.warning(f"CRITICAL: Normal contamination rate ({validation['contamination_rate']*100:.1f}%) "
                          f"exceeds 5% threshold!")
    
    if validation['detection_rate'] < 0.80:
        logger.warning(f"WARNING: Detection rate ({validation['detection_rate']*100:.1f}%) "
                       f"below 80% target")
    
    # Save outputs
    pathway_stats.to_csv(out / "enhanced_pathway_statistics.csv", index=False)
    decisions.reset_index().to_csv(out / "enhanced_sample_decisions.csv", index=False)
    threshold_info['results'].to_csv(out / "threshold_search.csv", index=False)
    
    validation_df = pd.DataFrame([{
        'n_normals': validation['n_normals'],
        'n_imds': validation['n_imds'],
        'normals_flagged': validation['normals_flagged'],
        'imds_flagged': validation['imds_flagged'],
        'detection_rate': validation['detection_rate'],
        'contamination_rate': validation['contamination_rate'],
        'optimal_threshold': threshold_info['optimal_threshold'],
        'optimal_percentile': threshold_info['percentile'],
        'flagged_normal_ids': ','.join(validation['flagged_normal_ids'])
    }])
    validation_df.to_csv(out / "enhanced_validation.csv", index=False)
    
    # Also save as main outputs
    pathway_stats.to_csv(out / "pathway_statistics.csv", index=False)
    decisions.reset_index().to_csv(out / "sample_decisions.csv", index=False)
    
    # ------------------------------------------------------------------
    # NEW: Anomaly Detection on pathway features
    # ------------------------------------------------------------------
    use_anomaly_detection = bool(config.get("use_anomaly_detection", True))
    
    if use_anomaly_detection and len(analysis_sample_ids) > 0:
        from pathway_pipeline.pipeline.pathway_analysis_clean import run_anomaly_detection
        
        _log_section("STEP 6: Anomaly Detection on Pathway Features")
        
        # Recompute pathway stats for NORMALS + IMDs only (exclude gray)
        feature_to_pathway_all = feature_to_pathway[
            feature_to_pathway['feature'].isin(zscores.columns)
        ]
        
        # Compute pathway stats for normals + IMDs only
        pathway_stats_all = compute_pathway_stouffers_z(
            features_filtered,
            feature_to_pathway_all,
            min_pathway_size=min_pathway_size
        )
        
        logger.info(f"Computed pathway Stouffer's Z for {pathway_stats_all['sample_id'].nunique()} samples")
        
        # Run anomaly detection - scores ONLY normals and IMDs (no gray)
        ad_results = run_anomaly_detection(
            pathway_stats=pathway_stats_all,
            normal_sample_ids=normal_sample_ids,
            imd_sample_ids=imd_sample_ids,
            gray_sample_ids=[],  # Empty list - no gray samples
            scorer_name=config.get("anomaly_scorer", "lof"),
            contamination=float(config.get("anomaly_contamination", 0.02)),
            n_neighbors=int(config.get("anomaly_n_neighbors", 20)),
            n_estimators=int(config.get("anomaly_n_estimators", 100)),
            percentile=float(config.get("anomaly_percentile", 95.0)),
        )
        
        # Save anomaly detection results
        ad_results['results'].to_csv(out / "anomaly_scores.csv", index=False)
        
        ad_validation_df = pd.DataFrame([{
            'scorer': ad_results['scorer'],
            'method': ad_results['method'],
            'threshold': ad_results['threshold'],
            'percentile': ad_results['percentile'],
            'n_normals': ad_results['n_normals'],
            'n_imds': ad_results['n_imds'],
            'n_grays': ad_results['n_grays'],
            'normals_flagged': ad_results['normals_flagged'],
            'imds_flagged': ad_results['imds_flagged'],
            'grays_flagged': ad_results['grays_flagged'],
            'detection_rate': ad_results['detection_rate'],
            'contamination_rate': ad_results['contamination_rate'],
            'gray_flag_rate': ad_results['gray_flag_rate'],
            'flagged_normal_ids': ','.join(ad_results['flagged_normal_ids']),
            'flagged_imd_ids': ','.join(ad_results['flagged_imd_ids']),
        }])
        ad_validation_df.to_csv(out / "anomaly_validation.csv", index=False)
        
        results["anomaly_detection"] = ad_results
        
        logger.info(f"\nWrote anomaly_scores.csv, anomaly_validation.csv to {out}")
    
    results["pathway_stats"] = pathway_stats
    results["decisions"] = decisions
    results["threshold_info"] = threshold_info
    results["validation"] = validation
    
    logger.info(f"\nWrote enhanced_pathway_statistics.csv, enhanced_sample_decisions.csv, "
                f"threshold_search.csv, enhanced_validation.csv to {out}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Clean pathway pipeline: feature -> HMDB -> pathway "
                    "matching and absolute Stouffer's Z pathway statistics."
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
        logger.info("\nClean pathway pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
