"""Clean, simple pathway analysis pipeline.

This module implements a streamlined pathway analysis that:
1. Maps features to pathways, removing unmapped features
2. Calculates Z-scores for all samples using normals as reference
3. Combines Z-scores into compound scores per pathway using absolute Stouffer's Z
4. Finds optimal cutoffs empirically from the normal distribution
5. Flags samples based on extreme pathway deviations
6. (NEW) Uses anomaly detection (LOF, IForest, Mahalanobis) trained on normals only

Key design decisions:
- Uses absolute Stouffer's Z to detect both same-direction and opposite-direction disturbances
- Only flags samples with 1-2 extremely deviated pathways (IMD pattern)
- Uses empirical thresholds computed from the actual normal distribution
- Filters to only normals + IMDs (Class 1 AND Oordeel 1) for analysis
- Removes all features that don't map to pathways
- For anomaly detection: train on NORMAL samples only, test on ALL samples

This is a COMPLETELY SEPARATE implementation from the existing pathway_stats_enhanced.py
which has too many bugs and complexity.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


def compute_absolute_stouffers_z(zscores: np.ndarray) -> float:
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


def compute_pathway_stouffers_z(
    zscores: pd.DataFrame,
    feature_to_pathway: pd.DataFrame,
    min_pathway_size: int = 3,
) -> pd.DataFrame:
    """Compute absolute Stouffer's Z-score for all pathways × samples.
    
    Args:
        zscores: DataFrame with samples as rows, features as columns.
        feature_to_pathway: DataFrame mapping features to pathways.
            Must have columns: 'feature', 'pathway_name'
        min_pathway_size: Minimum number of features required for a pathway.
        
    Returns:
        DataFrame with columns: sample_id, pathway_name, z_stouffer_abs, n_features
    """
    # Build pathway -> features mapping
    pathway_features: Dict[str, List[str]] = {}
    for _, row in feature_to_pathway.iterrows():
        pathway = row['pathway_name']
        feature = str(row['feature'])
        if pathway not in pathway_features:
            pathway_features[pathway] = []
        if feature not in pathway_features[pathway]:
            pathway_features[pathway].append(feature)
    
    # Filter pathways with enough features
    pathway_features = {p: feats for p, feats in pathway_features.items() 
                       if len(feats) >= min_pathway_size}
    
    if not pathway_features:
        logger.warning("No pathways have enough matched features.")
        return pd.DataFrame(columns=['sample_id', 'pathway_name', 'z_stouffer_abs', 'n_features'])
    
    logger.info(f"Computing Stouffer's Z for {len(pathway_features)} pathways")
    
    # Compute Stouffer's Z for each pathway × sample
    rows = []
    for sample_id, sample_zscores in zscores.iterrows():
        for pathway, features in pathway_features.items():
            # Get z-scores for this pathway's features
            # Only include features that exist in the zscores DataFrame
            existing_features = [f for f in features if f in sample_zscores.index]
            if len(existing_features) < min_pathway_size:
                continue
            
            pathway_z = sample_zscores[existing_features].to_numpy(dtype=float)
            z_abs = compute_absolute_stouffers_z(pathway_z)
            rows.append({
                'sample_id': sample_id,
                'pathway_name': pathway,
                'z_stouffer_abs': z_abs,
                'n_features': len(existing_features)
            })
    
    return pd.DataFrame(rows)


def find_optimal_threshold(
    pathway_stats: pd.DataFrame,
    normal_sample_ids: List,
    imd_sample_ids: List,
    min_detection: float = 0.80,
    max_contamination: float = 0.05,
    min_flagged_pathways: int = 1,
    n_thresholds: int = 20,
) -> Dict:
    """Find optimal threshold empirically.
    
    Tests various percentiles of the normal |Z_stouffer| distribution, finds the
    threshold that maximizes IMD detection while keeping normal contamination below target.
    
    Args:
        pathway_stats: DataFrame with z_stouffer_abs per (sample, pathway).
        normal_sample_ids: List of normal sample IDs.
        imd_sample_ids: List of IMD sample IDs.
        min_detection: Minimum IMD detection rate target (e.g., 0.80 = 80%).
        max_contamination: Maximum normal contamination rate target (e.g., 0.05 = 5%).
        min_flagged_pathways: Minimum number of flagged pathways to flag a sample.
        n_thresholds: Number of threshold steps to test.
        
    Returns:
        Dict with optimal threshold, detection rate, contamination rate, and full results.
    """
    # Get normal |Z_stouffer| distribution
    normal_stats = pathway_stats[pathway_stats['sample_id'].isin(normal_sample_ids)]
    normal_z = normal_stats['z_stouffer_abs'].dropna()
    
    if len(normal_z) < 10:
        logger.warning(f"Not enough normal data points ({len(normal_z)}) for empirical thresholding.")
        # Return a reasonable default
        return {
            'optimal_threshold': 25.0,
            'detection_rate': 0.0,
            'contamination_rate': 0.0,
            'threshold': 25.0,
            'n_normals_flagged': 0,
            'n_imds_flagged': 0,
            'results': pd.DataFrame()
        }
    
    # Generate percentiles to test (from 95th to 99.9999th)
    percentiles = np.linspace(95, 99.9999, n_thresholds)
    
    results = []
    for p in percentiles:
        threshold = float(np.percentile(normal_z, p))
        
        # Flag pathways
        pathway_stats['flagged'] = pathway_stats['z_stouffer_abs'] > threshold
        
        # Count flagged pathways per sample
        flagged_counts = pathway_stats[pathway_stats['flagged']].groupby('sample_id').size()
        all_samples = pathway_stats['sample_id'].unique()
        flagged_counts = flagged_counts.reindex(all_samples, fill_value=0)
        
        # Flag samples with >= min_flagged_pathways
        flagged_samples = set(flagged_counts[flagged_counts >= min_flagged_pathways].index)
        
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
    # First filter to acceptable contamination levels
    acceptable = results_df[results_df['contamination_rate'] <= max_contamination]
    
    if len(acceptable) > 0:
        # Among acceptable, pick the one with highest detection
        optimal_idx = acceptable['detection_rate'].idxmax()
        optimal = results_df.loc[optimal_idx]
    else:
        # If none meet contamination target, pick the one with highest detection
        # that has contamination closest to (but still above) the target
        results_df['contamination_diff'] = results_df['contamination_rate'] - max_contamination
        results_df['score'] = results_df['detection_rate'] - 10 * results_df['contamination_diff']
        optimal_idx = results_df['score'].idxmax()
        optimal = results_df.loc[optimal_idx]
    
    return {
        'optimal_threshold': float(optimal['threshold']),
        'detection_rate': float(optimal['detection_rate']),
        'contamination_rate': float(optimal['contamination_rate']),
        'threshold': float(optimal['threshold']),
        'n_normals_flagged': int(optimal['normals_flagged']),
        'n_imds_flagged': int(optimal['imds_flagged']),
        'percentile': float(optimal['percentile']),
        'results': results_df,
    }


def flag_samples(
    pathway_stats: pd.DataFrame,
    threshold: float,
    min_flagged_pathways: int = 1,
) -> pd.DataFrame:
    """Flag samples based on pathway Stouffer's Z-scores.
    
    Args:
        pathway_stats: DataFrame with z_stouffer_abs per (sample, pathway).
        threshold: Z_stouffer_abs threshold for flagging a pathway.
        min_flagged_pathways: Minimum number of flagged pathways to flag a sample.
        
    Returns:
        DataFrame with sample_id index and columns: flagged, n_flagged_pathways
    """
    # Flag pathways
    pathway_stats = pathway_stats.copy()
    pathway_stats['flagged'] = pathway_stats['z_stouffer_abs'] > threshold
    
    # Count flagged pathways per sample
    flagged_counts = pathway_stats[pathway_stats['flagged']].groupby('sample_id').size()
    all_samples = pathway_stats['sample_id'].unique()
    flagged_counts = flagged_counts.reindex(all_samples, fill_value=0)
    
    # Flag samples
    flagged = flagged_counts >= min_flagged_pathways
    
    return pd.DataFrame({
        'sample_id': flagged_counts.index,
        'flagged': flagged.values,
        'n_flagged_pathways': flagged_counts.values
    }).set_index('sample_id')


def validate_flagging(
    decisions: pd.DataFrame,
    normal_sample_ids: List,
    imd_sample_ids: List,
) -> Dict:
    """Validate flagging results.
    
    Args:
        decisions: DataFrame with 'flagged' column indexed by sample_id.
        normal_sample_ids: List of normal sample IDs.
        imd_sample_ids: List of IMD sample IDs.
        
    Returns:
        Dict with validation metrics.
    """
    all_samples = normal_sample_ids + imd_sample_ids
    
    # Get flagged status for each sample
    flagged = decisions['flagged'].reindex(all_samples, fill_value=False)
    
    # Count flagged in each category
    normals_flagged = int(flagged[flagged.index.isin(normal_sample_ids)].sum())
    imds_flagged = int(flagged[flagged.index.isin(imd_sample_ids)].sum())
    
    n_normals = len(normal_sample_ids)
    n_imds = len(imd_sample_ids)
    
    detection_rate = imds_flagged / n_imds if n_imds > 0 else 0.0
    contamination_rate = normals_flagged / n_normals if n_normals > 0 else 0.0
    
    # Get list of flagged normals
    # Use at[] to get scalar value instead of loc[] which can return a Series
    flagged_normal_ids = [s for s in normal_sample_ids if s in flagged.index and flagged.at[s]]
    
    return {
        'n_normals': n_normals,
        'n_imds': n_imds,
        'normals_flagged': normals_flagged,
        'imds_flagged': imds_flagged,
        'detection_rate': detection_rate,
        'contamination_rate': contamination_rate,
        'flagged_normal_ids': flagged_normal_ids
    }


def run_clean_pathway_analysis(
    zscores: pd.DataFrame,
    feature_to_pathway: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Optional[Path] = None,
    min_pathway_size: int = 3,
    classification_scheme: str = "class1_imd",
    min_detection: float = 0.80,
    max_contamination: float = 0.05,
    min_flagged_pathways: int = 1,
    n_thresholds: int = 20,
) -> Dict:
    """Run the complete clean pathway analysis pipeline.
    
    This is the main entry point for the clean, simple implementation.
    
    Steps:
    1. Classify samples into normals, IMDs, and gray
    2. Filter to normals + IMDs only (exclude gray)
    3. Compute pathway Stouffer's Z-scores
    4. Find optimal threshold empirically
    5. Flag samples using optimal threshold
    6. Validate results
    
    Args:
        zscores: DataFrame with per-metabolite z-scores (rows=samples, columns=features).
        feature_to_pathway: DataFrame mapping features to pathways.
            Must have columns: 'feature', 'pathway_name'
        metadata: DataFrame with Classification and Oordeel targeted columns.
        output_dir: Optional directory to save outputs.
        min_pathway_size: Minimum number of features per pathway.
        classification_scheme: Scheme for classifying samples ('class1_imd' or 'confident_normals').
        min_detection: Minimum IMD detection rate target.
        max_contamination: Maximum normal contamination rate target.
        min_flagged_pathways: Minimum flagged pathways to flag a sample.
        n_thresholds: Number of threshold steps to test.
        
    Returns:
        Dict with all results: pathway_stats, decisions, threshold_info, validation.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Classify samples
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Classify samples")
    logger.info("="*70)
    
    cls = pd.to_numeric(metadata['Classification'], errors='coerce')
    oor = pd.to_numeric(metadata['Oordeel targeted'], errors='coerce')
    
    if classification_scheme == "class1_imd":
        # Normals = Class 0 AND Oordeel 0
        # IMD = Class 1 AND Oordeel 1
        # Gray = Everything else
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
    
    logger.info(f"\nAnalyzing {len(analysis_sample_ids)} samples "
                f"({len(normal_sample_ids)} normals + {len(imd_sample_ids)} IMDs)")
    
    # Step 3: Compute pathway Stouffer's Z-scores
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Compute pathway Stouffer's Z-scores")
    logger.info("="*70)
    
    pathway_stats = compute_pathway_stouffers_z(
        zscores_filtered,
        feature_to_pathway,
        min_pathway_size=min_pathway_size
    )
    
    logger.info(f"Computed Stouffer's Z for {len(pathway_stats['pathway_name'].unique())} pathways "
                f"across {len(pathway_stats['sample_id'].unique())} samples")
    
    # Step 4: Find optimal threshold
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Find optimal threshold")
    logger.info("="*70)
    
    threshold_info = find_optimal_threshold(
        pathway_stats,
        normal_sample_ids,
        imd_sample_ids,
        min_detection=min_detection,
        max_contamination=max_contamination,
        min_flagged_pathways=min_flagged_pathways,
        n_thresholds=n_thresholds
    )
    
    logger.info(f"Optimal threshold: {threshold_info['optimal_threshold']:.2f} "
                f"(percentile {threshold_info['percentile']:.4f})")
    logger.info(f"Detection rate: {threshold_info['detection_rate']*100:.1f}%")
    logger.info(f"Contamination rate: {threshold_info['contamination_rate']*100:.1f}%")
    
    # Step 5: Flag samples using optimal threshold
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Flag samples")
    logger.info("="*70)
    
    decisions = flag_samples(
        pathway_stats,
        threshold=threshold_info['optimal_threshold'],
        min_flagged_pathways=min_flagged_pathways
    )
    
    # Step 6: Validate
    logger.info("\n" + "="*70)
    logger.info("STEP 5: Validate")
    logger.info("="*70)
    
    validation = validate_flagging(
        decisions,
        normal_sample_ids,
        imd_sample_ids
    )
    
    logger.info(f"\nValidation results:")
    logger.info(f"  Normals flagged: {validation['normals_flagged']} / {validation['n_normals']} "
                f"({validation['contamination_rate']*100:.1f}%)")
    logger.info(f"  IMDs flagged: {validation['imds_flagged']} / {validation['n_imds']} "
                f"({validation['detection_rate']*100:.1f}%)")
    
    if validation['normals_flagged'] > 0:
        logger.warning(f"WARNING: {validation['normals_flagged']} normal samples were flagged!")
        logger.warning(f"Flagged normals: {validation['flagged_normal_ids'][:10]}")
    
    if validation['detection_rate'] < min_detection:
        logger.warning(f"WARNING: Detection rate ({validation['detection_rate']*100:.1f}%) "
                       f"below target ({min_detection*100:.0f}%)")
    
    # Save outputs
    if output_dir is not None:
        pathway_stats.to_csv(output_dir / "pathway_stats.csv", index=False)
        decisions.reset_index().to_csv(output_dir / "sample_decisions.csv", index=False)
        threshold_info['results'].to_csv(output_dir / "threshold_search.csv", index=False)
        
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
        validation_df.to_csv(output_dir / "validation.csv", index=False)
        
        logger.info(f"\nWrote outputs to {output_dir}")
    
    return {
        'pathway_stats': pathway_stats,
        'decisions': decisions,
        'threshold_info': threshold_info,
        'validation': validation
    }


def run_anomaly_detection(
    pathway_stats: pd.DataFrame,
    normal_sample_ids: List,
    imd_sample_ids: List,
    gray_sample_ids: List,
    scorer_name: str = "lof",
    contamination: float = 0.02,
    n_neighbors: int = 20,
    n_estimators: int = 100,
    random_state: int = 42,
    percentile: float = 95.0,
) -> Dict:
    """Run anomaly detection on pathway Stouffer's Z scores.
    
    Trains on NORMAL samples only, then scores ALL samples (normals + IMDs + gray).
    This is the correct approach for low prevalence scenarios.
    
    Args:
        pathway_stats: DataFrame with columns: sample_id, pathway_name, z_stouffer_abs
        normal_sample_ids: List of normal sample IDs (for training)
        imd_sample_ids: List of IMD sample IDs (for validation)
        gray_sample_ids: List of gray sample IDs (for testing)
        scorer_name: Which anomaly detector to use ('lof', 'iforest', 'mahalanobis')
        contamination: Expected contamination rate for thresholding
        n_neighbors: Number of neighbors for LOF
        n_estimators: Number of trees for IForest
        random_state: Random seed
        percentile: Percentile for thresholding (higher = more strict)
        
    Returns:
        Dict with anomaly scores, decisions, and validation metrics
    """
    # Pivot to samples x pathways matrix
    pivot = pathway_stats.pivot(index='sample_id', columns='pathway_name', values='z_stouffer_abs')
    pivot = pivot.fillna(0)  # Fill missing with 0 (no deviation)
    
    # Get all sample IDs in order
    all_sample_ids = pivot.index.tolist()
    
    # Split into train (normals) and test (all)
    X_train = pivot.loc[normal_sample_ids].values
    X_test = pivot.values
    
    logger.info(f"Training anomaly detector on {len(normal_sample_ids)} normal samples")
    logger.info(f"Scoring {len(all_sample_ids)} total samples")
    logger.info(f"Using {pivot.shape[1]} pathway features")
    
    # Train anomaly detector
    if scorer_name == "lof":
        from sklearn.neighbors import LocalOutlierFactor
        
        # Use novelty=True to allow scoring new samples
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            contamination='auto',
            n_jobs=-1
        )
        lof.fit(X_train)
        
        # Score all samples
        scores = -lof.negative_outlier_factor(X_test)  # Higher = more anomalous
        method_name = "Local Outlier Factor"
        
    elif scorer_name == "iforest":
        from sklearn.ensemble import IsolationForest
        
        iforest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        iforest.fit(X_train)
        
        # Score all samples (higher = more anomalous for IF)
        scores = -iforest.score_samples(X_test)  # Negate to make higher = more anomalous
        method_name = "Isolation Forest"
        
    elif scorer_name == "mahalanobis":
        from sklearn.covariance import MinCovDet
        
        # Fit robust covariance on normals
        mcd = MinCovDet(random_state=random_state)
        mcd.fit(X_train)
        
        # Compute Mahalanobis distance for all samples
        distances = mcd.mahalanobis(X_test)
        scores = distances  # Higher = more anomalous
        method_name = "Mahalanobis Distance"
        
    else:
        raise ValueError(f"Unknown scorer: {scorer_name}. Use 'lof', 'iforest', or 'mahalanobis'")
    
    # Create results DataFrame
    results = pd.DataFrame({
        'sample_id': all_sample_ids,
        'anomaly_score': scores
    })
    
    # Compute threshold from normal distribution
    normal_scores = results[results['sample_id'].isin(normal_sample_ids)]['anomaly_score']
    threshold = float(np.percentile(normal_scores, percentile))
    
    # Flag samples above threshold
    results['flagged'] = results['anomaly_score'] > threshold
    
    # Compute validation metrics
    all_test_samples = imd_sample_ids + gray_sample_ids
    
    # Count flagged in each category
    normals_flagged = int(results[results['sample_id'].isin(normal_sample_ids) & results['flagged']].shape[0])
    imds_flagged = int(results[results['sample_id'].isin(imd_sample_ids) & results['flagged']].shape[0])
    grays_flagged = int(results[results['sample_id'].isin(gray_sample_ids) & results['flagged']].shape[0])
    
    n_normals = len(normal_sample_ids)
    n_imds = len(imd_sample_ids)
    n_grays = len(gray_sample_ids)
    
    detection_rate = imds_flagged / n_imds if n_imds > 0 else 0.0
    contamination_rate = normals_flagged / n_normals if n_normals > 0 else 0.0
    gray_flag_rate = grays_flagged / n_grays if n_grays > 0 else 0.0
    
    # Get flagged sample IDs
    flagged_normal_ids = results[results['sample_id'].isin(normal_sample_ids) & results['flagged']]['sample_id'].tolist()
    flagged_imd_ids = results[results['sample_id'].isin(imd_sample_ids) & results['flagged']]['sample_id'].tolist()
    
    logger.info(f"\n{method_name} Results:")
    logger.info(f"  Threshold: {threshold:.4f} (percentile {percentile})")
    logger.info(f"  Normals flagged: {normals_flagged} / {n_normals} ({contamination_rate*100:.1f}%)")
    logger.info(f"  IMDs flagged: {imds_flagged} / {n_imds} ({detection_rate*100:.1f}%)")
    logger.info(f"  Gray flagged: {grays_flagged} / {n_grays} ({gray_flag_rate*100:.1f}%)")
    
    return {
        'scorer': scorer_name,
        'method': method_name,
        'results': results,
        'threshold': threshold,
        'percentile': percentile,
        'n_normals': n_normals,
        'n_imds': n_imds,
        'n_grays': n_grays,
        'normals_flagged': normals_flagged,
        'imds_flagged': imds_flagged,
        'grays_flagged': grays_flagged,
        'detection_rate': detection_rate,
        'contamination_rate': contamination_rate,
        'gray_flag_rate': gray_flag_rate,
        'flagged_normal_ids': flagged_normal_ids,
        'flagged_imd_ids': flagged_imd_ids,
    }
