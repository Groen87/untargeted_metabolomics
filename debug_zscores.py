#!/usr/bin/env python3
"""Debug script to analyze z-score distributions and find optimal thresholds.

This script:
1. Loads the data and computes z-scores
2. Analyzes per-metabolite z-score distributions (normal vs IMD)
3. Analyzes Stouffer's Z distributions
4. Tests thresholds from 10-50 and calculates:
   - Normal contamination rate
   - IMD detection rate
   - F1 score
5. Generates plots for visualization
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import from the pipeline
from pathway_pipeline.config.config import Config
from pathway_pipeline.pipeline.hmdb_parser import build_name_index
from pathway_pipeline.pipeline.pathway_mapping import (
    load_pathways_tsv,
    match_features_to_hmdb,
    link_features_to_pathways,
)
from pathway_pipeline.pipeline.pathway_stats import compute_metabolite_zscores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_feature_matrix(input_file: str,
                         non_feature_columns,
                         patient_id_column: str = None,
                         age_column: str = None,
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the input CSV and split it into (features, metadata, ages)."""
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
    """Boolean Series over the sample index: True = normal reference set."""
    idx = metadata.index
    if metadata.empty:
        return pd.Series(np.ones(len(idx), dtype=bool), index=idx)

    if "Classification" in metadata.columns and "Oordeel targeted" in metadata.columns:
        cls = pd.to_numeric(metadata["Classification"], errors="coerce")
        oor = pd.to_numeric(metadata["Oordeel targeted"], errors="coerce")
        if scheme == "class1_imd":
            return pd.Series((cls == 0) & (oor == 0), index=idx)
    return pd.Series(np.ones(len(idx), dtype=bool), index=idx)


def _imd_labels(metadata: pd.DataFrame, scheme: str) -> pd.Series:
    """Binary 0/1 IMD label per sample aligned to metadata.index."""
    idx = metadata.index
    if metadata.empty or "Classification" not in metadata.columns:
        return pd.Series(np.zeros(len(idx), dtype=int), index=idx)
    cls = pd.to_numeric(metadata["Classification"], errors="coerce")
    if scheme == "class1_imd":
        oor = pd.to_numeric(metadata["Oordeel targeted"], errors="coerce")
        return pd.Series(np.where((cls == 1) & (oor == 1), 1, 0), index=idx)
    return pd.Series(np.where(cls == 0, 0, 1), index=idx)


def analyze_zscore_distributions(zscores: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    """Analyze per-metabolite z-score distributions for normals vs IMDs."""
    cls = pd.to_numeric(metadata["Classification"], errors="coerce")
    oor = pd.to_numeric(metadata["Oordeel targeted"], errors="coerce")
    
    # Classify samples
    normal_mask = (cls == 0) & (oor == 0)
    imd_mask = (cls == 1) & (oor == 1)
    
    n_normal = int(normal_mask.sum())
    n_imd = int(imd_mask.sum())
    
    logger.info(f"Analyzing z-score distributions: {n_normal} normals, {n_imd} IMDs")
    
    # Collect all |z-score| values
    all_abs_z = np.abs(zscores.values).flatten()
    all_abs_z = all_abs_z[~np.isnan(all_abs_z)]
    
    normal_abs_z = np.abs(zscores.loc[normal_mask].values).flatten()
    normal_abs_z = normal_abs_z[~np.isnan(normal_abs_z)]
    
    imd_abs_z = np.abs(zscores.loc[imd_mask].values).flatten()
    imd_abs_z = imd_abs_z[~np.isnan(imd_abs_z)]
    
    results = {
        'all': all_abs_z,
        'normal': normal_abs_z,
        'imd': imd_abs_z,
        'n_normal': n_normal,
        'n_imd': n_imd,
    }
    
    # Print summary statistics
    logger.info(f"\nPer-metabolite |z-score| distribution:")
    logger.info(f"  All samples:   mean={np.mean(all_abs_z):.2f}, median={np.median(all_abs_z):.2f}, max={np.max(all_abs_z):.2f}")
    logger.info(f"  Normals:       mean={np.mean(normal_abs_z):.2f}, median={np.median(normal_abs_z):.2f}, max={np.max(normal_abs_z):.2f}")
    logger.info(f"  IMDs:          mean={np.mean(imd_abs_z):.2f}, median={np.median(imd_abs_z):.2f}, max={np.max(imd_abs_z):.2f}")
    
    return results


def compute_stouffers_z_abs(per_metabolite_zscores: np.ndarray) -> float:
    """Compute absolute Stouffer's Z-score."""
    valid = per_metabolite_zscores[~np.isnan(per_metabolite_zscores)]
    if len(valid) == 0:
        return float("nan")
    n = len(valid)
    z_sum_abs = np.sum(np.abs(valid))
    return float(z_sum_abs / np.sqrt(n))


def analyze_stouffers_distribution(zscores: pd.DataFrame, feature_to_pathway: pd.DataFrame,
                                    metadata: pd.DataFrame, min_pathway_size: int = 3) -> dict:
    """Compute Stouffer's Z for all pathways and analyze distributions."""
    cls = pd.to_numeric(metadata["Classification"], errors="coerce")
    oor = pd.to_numeric(metadata["Oordeel targeted"], errors="coerce")
    
    normal_mask = (cls == 0) & (oor == 0)
    imd_mask = (cls == 1) & (oor == 1)
    
    # Build pathway feature mapping
    available = set(zscores.columns)
    pathway_features = {}
    if not feature_to_pathway.empty:
        for smp_id, grp in feature_to_pathway.groupby("smp_id"):
            feats = sorted(set(grp["feature"]) & available)
            name = grp["pathway_name"].iloc[0] if "pathway_name" in grp.columns else smp_id
            if len(feats) >= min_pathway_size:
                pathway_features[smp_id] = {"pathway_name": name, "features": feats}
    
    if not pathway_features:
        logger.warning("No pathways have enough matched metabolites.")
        return {}
    
    # Compute Stouffer's Z for each sample-pathway combination
    sample_ids = zscores.index
    z_arr = zscores.to_numpy(dtype=float)
    col_index = {c: i for i, c in enumerate(zscores.columns)}
    
    normal_z_stouffers = []
    imd_z_stouffers = []
    all_z_stouffers = []
    
    for smp_id, info in pathway_features.items():
        feat_cols = [col_index[f] for f in info["features"]]
        sub = z_arr[:, feat_cols]
        
        for i, sid in enumerate(sample_ids):
            z_stouffer = compute_stouffers_z_abs(sub[i, :])
            all_z_stouffers.append((sid, info["pathway_name"], z_stouffer))
            
            if normal_mask.loc[sid]:
                normal_z_stouffers.append(z_stouffer)
            elif imd_mask.loc[sid]:
                imd_z_stouffers.append(z_stouffer)
    
    # Convert to arrays
    normal_z_stouffers = np.array([z for z in normal_z_stouffers if not np.isnan(z)])
    imd_z_stouffers = np.array([z for z in imd_z_stouffers if not np.isnan(z)])
    all_z_stouffers_arr = np.array([z for _, _, z in all_z_stouffers if not np.isnan(z)])
    
    results = {
        'all': all_z_stouffers_arr,
        'normal': normal_z_stouffers,
        'imd': imd_z_stouffers,
        'n_pathways': len(pathway_features),
    }
    
    logger.info(f"\nStouffer's Z (absolute) distribution:")
    logger.info(f"  All samples:   mean={np.mean(all_z_stouffers_arr):.2f}, median={np.median(all_z_stouffers_arr):.2f}, max={np.max(all_z_stouffers_arr):.2f}")
    if len(normal_z_stouffers) > 0:
        logger.info(f"  Normals:       mean={np.mean(normal_z_stouffers):.2f}, median={np.median(normal_z_stouffers):.2f}, max={np.max(normal_z_stouffers):.2f}")
    if len(imd_z_stouffers) > 0:
        logger.info(f"  IMDs:          mean={np.mean(imd_z_stouffers):.2f}, median={np.median(imd_z_stouffers):.2f}, max={np.max(imd_z_stouffers):.2f}")
    
    return results


def test_thresholds(zscores: pd.DataFrame, feature_to_pathway: pd.DataFrame,
                   metadata: pd.DataFrame, min_pathway_size: int = 3,
                   threshold_range: Tuple[float, float, int] = (10, 50, 1)) -> pd.DataFrame:
    """Test different thresholds and compute contamination/detection rates.
    
    For each threshold, compute:
    - Normal contamination rate (% of normals with at least one pathway flagged)
    - IMD detection rate (% of IMDs with at least one pathway flagged)
    - F1 score (harmonic mean of precision and recall)
    """
    cls = pd.to_numeric(metadata["Classification"], errors="coerce")
    oor = pd.to_numeric(metadata["Oordeel targeted"], errors="coerce")
    
    normal_mask = (cls == 0) & (oor == 0)
    imd_mask = (cls == 1) & (oor == 1)
    
    n_normal = int(normal_mask.sum())
    n_imd = int(imd_mask.sum())
    
    # Build pathway feature mapping
    available = set(zscores.columns)
    pathway_features = {}
    if not feature_to_pathway.empty:
        for smp_id, grp in feature_to_pathway.groupby("smp_id"):
            feats = sorted(set(grp["feature"]) & available)
            name = grp["pathway_name"].iloc[0] if "pathway_name" in grp.columns else smp_id
            if len(feats) >= min_pathway_size:
                pathway_features[smp_id] = {"pathway_name": name, "features": feats}
    
    if not pathway_features:
        logger.warning("No pathways have enough matched metabolites.")
        return pd.DataFrame()
    
    # Compute Stouffer's Z for each sample-pathway combination
    sample_ids = zscores.index
    z_arr = zscores.to_numpy(dtype=float)
    col_index = {c: i for i, c in enumerate(zscores.columns)}
    
    # For each sample, compute max |Z_stouffer| across all pathways
    sample_max_z_stouffer = {}
    
    for sid in sample_ids:
        max_z = 0
        for smp_id, info in pathway_features.items():
            feat_cols = [col_index[f] for f in info["features"]]
            sub = z_arr[list(sample_ids).index(sid), feat_cols]
            z_stouffer = compute_stouffers_z_abs(sub)
            if not np.isnan(z_stouffer) and z_stouffer > max_z:
                max_z = z_stouffer
        sample_max_z_stouffer[sid] = max_z
    
    # Test thresholds
    start, end, step = threshold_range
    if isinstance(step, int):
        thresholds = list(range(int(start), int(end) + 1, step))
    else:
        thresholds = np.arange(start, end + step, step).tolist()
    
    results = []
    for threshold in thresholds:
        # Count how many normals and IMDs have at least one pathway above threshold
        normal_flagged = sum(1 for sid in sample_ids if normal_mask.loc[sid] and sample_max_z_stouffer[sid] > threshold)
        imd_flagged = sum(1 for sid in sample_ids if imd_mask.loc[sid] and sample_max_z_stouffer[sid] > threshold)
        
        normal_contamination = (normal_flagged / n_normal * 100) if n_normal > 0 else 0
        imd_detection = (imd_flagged / n_imd * 100) if n_imd > 0 else 0
        
        # Calculate precision and recall for F1
        # True positives = IMD flagged
        # False positives = normals flagged
        # True negatives = normals not flagged
        # False negatives = IMD not flagged
        tp = imd_flagged
        fp = normal_flagged
        fn = n_imd - imd_flagged
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'normal_contamination_pct': round(normal_contamination, 2),
            'normal_flagged': normal_flagged,
            'imd_detection_pct': round(imd_detection, 2),
            'imd_flagged': imd_flagged,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
        })
    
    return pd.DataFrame(results)


def generate_plots(zscore_results: dict, stouffer_results: dict,
                   threshold_results: pd.DataFrame, output_dir: Path) -> None:
    """Generate visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Per-metabolite |z-score| distribution KDE
    plt.figure(figsize=(12, 8))
    sns.kdeplot(zscore_results['normal'], label='Normals', color='blue', linewidth=2)
    sns.kdeplot(zscore_results['imd'], label='IMDs', color='red', linewidth=2)
    plt.xlabel('|z-score| per metabolite')
    plt.ylabel('Density')
    plt.title(f'Per-metabolite |z-score| Distribution\n({zscore_results["n_normal"]} normals, {zscore_results["n_imd"]} IMDs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'per_metabolite_zscore_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Stouffer's Z threshold optimization curve
    plt.figure(figsize=(12, 8))
    plt.plot(threshold_results['threshold'], threshold_results['normal_contamination_pct'],
             label='Normal Contamination %', color='blue', linewidth=2, marker='o')
    plt.plot(threshold_results['threshold'], threshold_results['imd_detection_pct'],
             label='IMD Detection %', color='red', linewidth=2, marker='o')
    plt.plot(threshold_results['threshold'], threshold_results['f1_score'] * 100,
             label='F1 Score (%)', color='green', linewidth=2, marker='o', linestyle='--')
    plt.axhline(5, color='gray', linestyle=':', label='5% Contamination Target')
    plt.axhline(80, color='gray', linestyle=':', label='80% Detection Target')
    plt.xlabel('Threshold (|Z_stouffer|)')
    plt.ylabel('Rate (%)')
    plt.title('Threshold Optimization: Normal Contamination vs IMD Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'threshold_optimization_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Max |z_stouffer| per sample distribution
    plt.figure(figsize=(12, 8))
    
    logger.info(f"Generated plots in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug script to analyze z-score distributions and find optimal thresholds."
    )
    parser.add_argument("--input", default=None,
                        help="Path to the feature matrix CSV.")
    parser.add_argument("--output", default="debug_outputs",
                        help="Output directory for debug results.")
    parser.add_argument("--hmdb", default="pathway_pipeline/data/hmdb_metabolites.xml",
                        help="Path to HMDB XML file.")
    parser.add_argument("--pathways", default="pathway_pipeline/data/pathways.tsv",
                        help="Path to pathways TSV file.")
    parser.add_argument("--config", default=None,
                        help="Path to config YAML.")
    args = parser.parse_args()

    # Load config
    config = Config(args.config) if args.config else Config()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("DEBUG: Z-score Distribution Analysis")
    logger.info("=" * 70)

    # Load data
    logger.info("\nLoading feature matrix...")
    age_column = config.get("age_column", None)
    features, metadata, ages = load_feature_matrix(
        args.input or config.get("input_file", "data/merged_data_with_classification.csv"),
        non_feature_columns=config.get_list("non_feature_columns", ["Oordeel targeted", "Classification"]),
        patient_id_column=config.get("patient_id_column", None),
        age_column=age_column,
    )

    # Get normal mask
    normal_mask = _normal_reference_mask(metadata, "class1_imd")
    imd_labels = _imd_labels(metadata, "class1_imd")
    
    n_normal = int(normal_mask.sum())
    n_imd = int(imd_labels.sum())
    n_gray = len(metadata) - n_normal - n_imd
    
    logger.info(f"Sample classification:")
    logger.info(f"  Normals (Class 0 AND Oordeel 0): {n_normal}")
    logger.info(f"  IMDs (Class 1 AND Oordeel 1): {n_imd}")
    logger.info(f"  Gray (others): {n_gray}")

    # Build HMDB index and pathway mapping
    logger.info("\nBuilding HMDB name index...")
    name_index = build_name_index(args.hmdb, min_name_length=3, use_cache=True)

    logger.info("Matching features to HMDB...")
    feature_to_hmdb = match_features_to_hmdb(
        feature_columns=list(features.columns),
        name_index=name_index,
        min_name_length=3,
    )

    logger.info("Loading pathways and linking features...")
    pathways = load_pathways_tsv(args.pathways)
    feature_to_pathway = link_features_to_pathways(feature_to_hmdb, pathways)
    min_pathway_size = int(config.get("min_pathway_size", 3))

    # Compute z-scores
    logger.info("\nComputing metabolite z-scores...")
    pathway_features = sorted(set(feature_to_pathway.get("matched_features", pd.Series(dtype=str))
                                  .str.split(";").explode().dropna()))
    if not pathway_features:
        logger.error("No matched pathway features!")
        sys.exit(1)
    
    sub_features = features[pathway_features]
    zscores = compute_metabolite_zscores(
        sub_features, normal_mask=normal_mask,
        iqr_scale=bool(config.get("iqr_scale", True)),
        ages=ages,
        age_adjustment_method=config.get("age_adjustment_method", "ols"),
        age_loess_frac=float(config.get("age_loess_frac", 0.5)),
    )

    # Analyze distributions
    logger.info("\n" + "=" * 70)
    logger.info("ANALYZING DISTRIBUTIONS")
    logger.info("=" * 70)
    
    zscore_dist = analyze_zscore_distributions(zscores, metadata)
    stouffer_dist = analyze_stouffers_distribution(zscores, feature_to_pathway, metadata, min_pathway_size)

    # Test thresholds
    logger.info("\n" + "=" * 70)
    logger.info("TESTING THRESHOLDS")
    logger.info("=" * 70)
    
    threshold_results = test_thresholds(zscores, feature_to_pathway, metadata, min_pathway_size)
    
    # Save threshold analysis
    threshold_results.to_csv(out / "threshold_analysis.csv", index=False)
    logger.info(f"Wrote threshold_analysis.csv")
    
    # Print threshold results
    logger.info("\nThreshold Analysis:")
    logger.info(threshold_results.to_string(index=False))
    
    # Find optimal threshold (F1 score > 0.8 and contamination < 5%)
    optimal = threshold_results[
        (threshold_results['normal_contamination_pct'] < 5) &
        (threshold_results['imd_detection_pct'] > 80)
    ]
    
    if not optimal.empty:
        # Get threshold with highest F1 score
        optimal = optimal.loc[optimal['f1_score'].idxmax()]
        logger.info(f"\nOPTIMAL THRESHOLD FOUND:")
        logger.info(f"  Threshold: {optimal['threshold']}")
        logger.info(f"  Normal contamination: {optimal['normal_contamination_pct']}%")
        logger.info(f"  IMD detection: {optimal['imd_detection_pct']}%")
        logger.info(f"  F1 score: {optimal['f1_score']}")
        logger.info(f"\nRECOMMENDED config.yaml settings:")
        logger.info(f"  extreme_z_threshold: {optimal['threshold']}")
    else:
        # Find best compromise
        logger.info("\nNo threshold meets both targets (<5% contamination AND >80% detection)")
        logger.info("Finding best compromise...")
        # Sort by F1 score descending
        best = threshold_results.loc[threshold_results['f1_score'].idxmax()]
        logger.info(f"\nBEST COMPROMISE:")
        logger.info(f"  Threshold: {best['threshold']}")
        logger.info(f"  Normal contamination: {best['normal_contamination_pct']}%")
        logger.info(f"  IMD detection: {best['imd_detection_pct']}%")
        logger.info(f"  F1 score: {best['f1_score']}")
    
    # Generate plots
    logger.info("\nGenerating plots...")
    generate_plots(zscore_dist, stouffer_dist, threshold_results, out)
    
    logger.info("\n" + "=" * 70)
    logger.info("DEBUG ANALYSIS COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
