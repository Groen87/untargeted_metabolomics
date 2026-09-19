"""Simple, clean pathway analysis pipeline.

This module implements a streamlined pathway analysis that:
1. Maps features to pathways, removing unmapped features
2. Calculates Z-scores for normals and IMDs only
3. Combines Z-scores into compound scores per pathway using absolute Stouffer's Z
4. Finds optimal cutoffs empirically from the normal distribution
5. Flags samples based on extreme pathway deviations

Key design decisions:
- Uses absolute Stouffer's Z to detect both same-direction and opposite-direction disturbances
- Only flags samples with 1-2 extremely deviated pathways (IMD pattern)
- Uses empirical thresholds computed from the actual normal distribution
- Filters to only 317 samples (217 normals + 100 IMDs) where IMD = Class 1 AND Oordeel 1
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


def compute_stouffers_z_abs(zscores: np.ndarray) -> float:
    """Compute absolute Stouffer's Z-score for a set of z-scores.
    
    This detects ANY disturbance regardless of direction, which is critical
    for IMD detection where a block can cause both accumulation and depletion
    of different metabolites in the same pathway.
    
    Args:
        zscores: 1-D array of z-scores for metabolites in a pathway.
        
    Returns:
        Absolute Stouffer's Z-score: sum(|z_i|) / sqrt(n)
        Returns NaN if all inputs are NaN.
    """
    valid = zscores[~np.isnan(zscores)]
    if len(valid) == 0:
        return float("nan")
    
    z_sum_abs = np.sum(np.abs(valid))
    z_combined_abs = z_sum_abs / np.sqrt(len(valid))
    return float(z_combined_abs)


def compute_pathway_stouffers(
    zscores: pd.DataFrame,
    feature_to_pathway: pd.DataFrame,
    min_pathway_size: int = 3,
) -> pd.DataFrame:
    """Compute Stouffer's Z-score for all pathways × samples.
    
    Args:
        zscores: DataFrame with samples as rows, features as columns.
        feature_to_pathway: DataFrame mapping features to pathways.
        min_pathway_size: Minimum number of features required for a pathway.
        
    Returns:
        DataFrame with columns: sample_id, pathway_name, z_stouffer, n_features
    """
    # Build pathway -> features mapping
    pathway_features = {}
    for _, row in feature_to_pathway.iterrows():
        pathway = row['pathway_name']
        feature = row['feature']
        if pathway not in pathway_features:
            pathway_features[pathway] = []
        pathway_features[pathway].append(feature)
    
    # Filter pathways with enough features
    pathway_features = {p: feats for p, feats in pathway_features.items() 
                       if len(feats) >= min_pathway_size}
    
    if not pathway_features:
        logger.warning("No pathways have enough matched features.")
        return pd.DataFrame(columns=['sample_id', 'pathway_name', 'z_stouffer', 'n_features'])
    
    # Compute Stouffer's Z for each pathway × sample
    rows = []
    for sample_id, sample_zscores in zscores.iterrows():
        for pathway, features in pathway_features.items():
            # Get z-scores for this pathway's features
            pathway_z = sample_zscores[features].to_numpy(dtype=float)
            z_abs = compute_stouffers_z_abs(pathway_z)
            rows.append({
                'sample_id': sample_id,
                'pathway_name': pathway,
                'z_stouffer': z_abs,
                'n_features': len(features)
            })
    
    return pd.DataFrame(rows)


def find_optimal_threshold(
    pathway_stats: pd.DataFrame,
    normal_sample_ids: list,
    imd_sample_ids: list,
    min_detection: float = 0.80,
    max_contamination: float = 0.05,
    min_flagged_pathways: int = 1,
    min_percentile: float = 95.0,
    max_percentile: float = 99.9999,
    n_percentiles: int = 20,
) -> dict:
    """Find optimal threshold empirically.
    
    Tests percentiles from min_percentile to max_percentile, finds the
    threshold that maximizes detection while keeping contamination below target.
    
    Args:
        pathway_stats: DataFrame with z_stouffer per (sample, pathway).
        normal_sample_ids: List of normal sample IDs.
        imd_sample_ids: List of IMD sample IDs.
        min_detection: Minimum IMD detection rate (e.g., 0.80 = 80%).
        max_contamination: Maximum normal contamination rate (e.g., 0.05 = 5%).
        min_flagged_pathways: Minimum number of flagged pathways to flag a sample.
        min_percentile: Minimum percentile to test.
        max_percentile: Maximum percentile to test.
        n_percentiles: Number of percentile steps to test.
        
    Returns:
        Dict with optimal threshold, detection rate, contamination rate, and results.
    """
    # Get normal |Z_stouffer| distribution
    normal_stats = pathway_stats[pathway_stats['sample_id'].isin(normal_sample_ids)]
    normal_z = normal_stats['z_stouffer'].abs().dropna()
    
    if len(normal_z) < 10:
        logger.warning(f"Not enough normal data points ({len(normal_z)}) for empirical thresholding.")
        return {
            'optimal_threshold': 25.0,
            'detection_rate': 0.0,
            'contamination_rate': 0.0,
            'percentile': 99.999,
            'results': pd.DataFrame()
        }
    
    # Generate percentiles to test
    percentiles = np.linspace(min_percentile, max_percentile, n_percentiles)
    
    results = []
    for p in percentiles:
        threshold = float(np.percentile(normal_z, p))
        
        # Flag pathways
        pathway_stats['flagged'] = pathway_stats['z_stouffer'].abs() > threshold
        
        # Count flagged pathways per sample
        flagged_counts = pathway_stats[pathway_stats['flagged']].groupby('sample_id').size()
        flagged_counts = flagged_counts.reindex(pathway_stats['sample_id'].unique(), fill_value=0)
        
        # Flag samples with >= min_flagged_pathways
        flagged_samples = flagged_counts[flagged_counts >= min_flagged_pathways].index.tolist()
        
        # Compute metrics
        n_normals = len(normal_sample_ids)
        n_imds = len(imd_sample_ids)
        
        normals_flagged = len([s for s in flagged_samples if s in normal_sample_ids])
        imds_flagged = len([s for s in flagged_samples if s in imd_sample_ids])
        
        detection_rate = imds_flagged / n_imds if n_imds > 0 else 0.0
        contamination_rate = normals_flagged / n_normals if n_normals > 0 else 0.0
        
        results.append({
            'percentile': p,
            'threshold': threshold,
            'detection_rate': detection_rate,
            'contamination_rate': contamination_rate,
            'normals_flagged': normals_flagged,
            'imds_flagged': imds_flagged,
            'total_flagged': len(flagged_samples)
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal: maximize detection while keeping contamination <= max_contamination
    # Filter to acceptable contamination levels
    acceptable = results_df[results_df['contamination_rate'] <= max_contamination]
    
    if len(acceptable) > 0:
        # Among acceptable, pick the one with highest detection
        optimal = acceptable.loc[acceptable['detection_rate'].idxmax()]
    else:
        # If none meet contamination target, pick the one closest to target
        # that has highest detection
        results_df['contamination_diff'] = results_df['contamination_rate'] - max_contamination
        results_df['score'] = results_df['detection_rate'] - 10 * results_df['contamination_diff']
        optimal = results_df.loc[results_df['score'].idxmax()]
    
    return {
        'optimal_threshold': float(optimal['threshold']),
        'detection_rate': float(optimal['detection_rate']),
        'contamination_rate': float(optimal['contamination_rate']),
        'percentile': float(optimal['percentile']),
        'results': results_df,
        'optimal_row': optimal
    }


def flag_samples_by_pathway(
    pathway_stats: pd.DataFrame,
    threshold: float,
    min_flagged_pathways: int = 1,
) -> pd.DataFrame:
    """Flag samples based on pathway Stouffer's Z-scores.
    
    Args:
        pathway_stats: DataFrame with z_stouffer per (sample, pathway).
        threshold: |Z_stouffer| threshold for flagging a pathway.
        min_flagged_pathways: Minimum number of flagged pathways to flag a sample.
        
    Returns:
        DataFrame with sample_id index and 'flagged' column.
    """
    # Flag pathways
    pathway_stats = pathway_stats.copy()
    pathway_stats['flagged'] = pathway_stats['z_stouffer'].abs() > threshold
    
    # Count flagged pathways per sample
    flagged_counts = pathway_stats[pathway_stats['flagged']].groupby('sample_id').size()
    flagged_counts = flagged_counts.reindex(pathway_stats['sample_id'].unique(), fill_value=0)
    
    # Flag samples
    flagged_samples = flagged_counts >= min_flagged_pathways
    
    return pd.DataFrame({
        'sample_id': flagged_counts.index,
        'flagged': flagged_samples.values,
        'n_flagged_pathways': flagged_counts.values
    }).set_index('sample_id')


def validate_flagging(
    decisions: pd.DataFrame,
    normal_sample_ids: list,
    imd_sample_ids: list,
    gray_sample_ids: list = None,
) -> dict:
    """Validate flagging results.
    
    Args:
        decisions: DataFrame with 'flagged' column indexed by sample_id.
        normal_sample_ids: List of normal sample IDs.
        imd_sample_ids: List of IMD sample IDs.
        gray_sample_ids: Optional list of gray sample IDs.
        
    Returns:
        Dict with validation metrics.
    """
    all_samples = normal_sample_ids + imd_sample_ids
    if gray_sample_ids:
        all_samples += gray_sample_ids
    
    # Get flagged status for each sample
    flagged = decisions['flagged'].reindex(all_samples, fill_value=False)
    
    # Count flagged in each category
    normals_flagged = flagged[flagged.index.isin(normal_sample_ids)].sum()
    imds_flagged = flagged[flagged.index.isin(imd_sample_ids)].sum()
    
    n_normals = len(normal_sample_ids)
    n_imds = len(imd_sample_ids)
    
    detection_rate = imds_flagged / n_imds if n_imds > 0 else 0.0
    contamination_rate = normals_flagged / n_normals if n_normals > 0 else 0.0
    
    # Get list of flagged normals
    flagged_normal_ids = [s for s in normal_sample_ids if flagged.get(s, False)]
    
    return {
        'n_normals': n_normals,
        'n_imds': n_imds,
        'normals_flagged': int(normals_flagged),
        'imds_flagged': int(imds_flagged),
        'detection_rate': detection_rate,
        'contamination_rate': contamination_rate,
        'flagged_normal_ids': flagged_normal_ids
    }


def run_simple_pathway_analysis(
    zscores: pd.DataFrame,
    feature_to_pathway: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Optional[Path] = None,
    min_pathway_size: int = 3,
    classification_scheme: str = "class1_imd",
    min_detection: float = 0.80,
    max_contamination: float = 0.05,
    min_flagged_pathways: int = 1,
    min_percentile: float = 95.0,
    max_percentile: float = 99.9999,
    n_percentiles: int = 20,
) -> dict:
    """Run the complete simple pathway analysis pipeline.
    
    This is the main entry point for the simple, clean implementation.
    
    Args:
        zscores: DataFrame with per-metabolite z-scores (rows=samples, columns=features).
        feature_to_pathway: DataFrame mapping features to pathways.
        metadata: DataFrame with Classification and Oordeel targeted columns.
        output_dir: Optional directory to save outputs.
        min_pathway_size: Minimum number of features per pathway.
        classification_scheme: Scheme for classifying samples.
        min_detection: Minimum IMD detection rate target.
        max_contamination: Maximum normal contamination rate target.
        min_flagged_pathways: Minimum flagged pathways to flag a sample.
        min_percentile: Minimum percentile for threshold testing.
        max_percentile: Maximum percentile for threshold testing.
        n_percentiles: Number of percentile steps to test.
        
    Returns:
        Dict with all results: pathway_stats, decisions, threshold_info, validation.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Classify samples
    cls = pd.to_numeric(metadata['Classification'], errors='coerce')
    oor = pd.to_numeric(metadata['Oordeel targeted'], errors='coerce')
    
    if classification_scheme == "class1_imd":
        normal_mask = (cls == 0) & (oor == 0)
        imd_mask = (cls == 1) & (oor == 1)
        gray_mask = ~normal_mask & ~imd_mask
    else:
        # Default: confident normals
        normal_mask = (cls == 0) & (oor == 0)
        imd_mask = ~normal_mask
        gray_mask = pd.Series(False, index=metadata.index)
    
    normal_sample_ids = metadata.index[normal_mask].tolist()
    imd_sample_ids = metadata.index[imd_mask].tolist()
    gray_sample_ids = metadata.index[gray_mask].tolist()
    
    logger.info(f"Sample classification:")
    logger.info(f"  Normals: {len(normal_sample_ids)}")
    logger.info(f"  IMDs: {len(imd_sample_ids)}")
    logger.info(f"  Gray: {len(gray_sample_ids)}")
    
    # Step 2: Filter to normals + IMDs only (exclude gray)
    analysis_sample_ids = normal_sample_ids + imd_sample_ids
    zscores_filtered = zscores.loc[analysis_sample_ids]
    
    logger.info(f"Analyzing {len(analysis_sample_ids)} samples ({len(normal_sample_ids)} normals + {len(imd_sample_ids)} IMDs)")
    
    # Step 3: Compute pathway Stouffer's Z-scores
    logger.info("Computing pathway Stouffer's Z-scores...")
    pathway_stats = compute_pathway_stouffers(
        zscores_filtered,
        feature_to_pathway,
        min_pathway_size=min_pathway_size
    )
    
    logger.info(f"Computed Stouffer's Z for {len(pathway_stats['pathway_name'].unique())} pathways "
                f"across {len(pathway_stats['sample_id'].unique())} samples")
    
    # Step 4: Find optimal threshold
    logger.info("Finding optimal threshold...")
    threshold_info = find_optimal_threshold(
        pathway_stats,
        normal_sample_ids,
        imd_sample_ids,
        min_detection=min_detection,
        max_contamination=max_contamination,
        min_flagged_pathways=min_flagged_pathways,
        min_percentile=min_percentile,
        max_percentile=max_percentile,
        n_percentiles=n_percentiles
    )
    
    logger.info(f"Optimal threshold: {threshold_info['optimal_threshold']:.2f} "
                f"(percentile {threshold_info['percentile']:.4f})")
    logger.info(f"Detection rate: {threshold_info['detection_rate']*100:.1f}%")
    logger.info(f"Contamination rate: {threshold_info['contamination_rate']*100:.1f}%")
    
    # Step 5: Flag samples using optimal threshold
    logger.info("Flagging samples...")
    decisions = flag_samples_by_pathway(
        pathway_stats,
        threshold=threshold_info['optimal_threshold'],
        min_flagged_pathways=min_flagged_pathways
    )
    
    # Step 6: Validate
    logger.info("Validating...")
    validation = validate_flagging(
        decisions,
        normal_sample_ids,
        imd_sample_ids,
        gray_sample_ids
    )
    
    logger.info(f"\nValidation results:")
    logger.info(f"  Normals flagged: {validation['normals_flagged']} / {validation['n_normals']} "
                f"({validation['contamination_rate']*100:.1f}%)")
    logger.info(f"  IMDs flagged: {validation['imds_flagged']} / {validation['n_imds']} "
                f"({validation['detection_rate']*100:.1f}%)")
    
    # Save outputs
    if output_dir is not None:
        pathway_stats.to_csv(output_dir / "simple_pathway_stats.csv", index=False)
        decisions.reset_index().to_csv(output_dir / "simple_sample_decisions.csv", index=False)
        threshold_info['results'].to_csv(output_dir / "simple_threshold_info.csv", index=False)
        
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
        validation_df.to_csv(output_dir / "simple_validation.csv", index=False)
        
        logger.info(f"Wrote outputs to {output_dir}")
    
    return {
        'pathway_stats': pathway_stats,
        'decisions': decisions,
        'threshold_info': threshold_info,
        'validation': validation
    }
