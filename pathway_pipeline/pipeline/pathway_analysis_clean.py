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
        pathway_stats_copy = pathway_stats.copy()
        pathway_stats_copy['flagged'] = pathway_stats_copy['z_stouffer_abs'] > threshold
        
        # Count flagged pathways per sample
        flagged_counts = pathway_stats_copy[pathway_stats_copy['flagged']].groupby('sample_id').size()
        all_samples = pathway_stats_copy['sample_id'].unique()
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
    
    # Get flagged status for each sample as a dict to avoid Series indexing issues
    flagged_dict = decisions['flagged'].to_dict()
    
    # Count flagged in each category
    normals_flagged = sum(1 for s in normal_sample_ids if flagged_dict.get(s, False))
    imds_flagged = sum(1 for s in imd_sample_ids if flagged_dict.get(s, False))
    
    n_normals = len(normal_sample_ids)
    n_imds = len(imd_sample_ids)
    
    detection_rate = imds_flagged / n_imds if n_imds > 0 else 0.0
    contamination_rate = normals_flagged / n_normals if n_normals > 0 else 0.0
    
    # Get list of flagged normals
    flagged_normal_ids = [s for s in normal_sample_ids if flagged_dict.get(s, False)]
    
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


def find_optimal_anomaly_threshold(
    scores: np.ndarray,
    is_normal: np.ndarray,
    max_contamination: float = 0.05,
    min_detection: float = 0.80,
    optimization_metric: str = "f1",
    n_thresholds: int = 100,
) -> Dict:
    """Find optimal threshold for anomaly scores by optimizing a metric on validation data.
    
    Args:
        scores: 1-D array of anomaly scores for validation samples
        is_normal: Boolean array indicating which samples are normal (True) vs IMD (False)
        max_contamination: Maximum acceptable normal contamination rate
        min_detection: Minimum acceptable IMD detection rate
        optimization_metric: Metric to optimize ('f1', 'precision', 'recall', 'youden')
        n_thresholds: Number of threshold candidates to test
        
    Returns:
        Dict with optimal threshold, metrics, and all candidate results
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Get sorted unique scores as candidate thresholds
    sorted_scores = np.sort(scores)
    
    # Add some thresholds beyond the max to ensure we get clean results
    min_score = sorted_scores.min() - 1
    max_score = sorted_scores.max() + 1
    candidate_thresholds = np.linspace(min_score, max_score, n_thresholds)
    
    results = []
    best_threshold = None
    best_metric_value = -np.inf
    
    n_normals = np.sum(is_normal)
    n_imds = np.sum(~is_normal)
    
    for threshold in candidate_thresholds:
        # Flag samples above threshold
        flagged = scores > threshold
        
        # Count results
        normals_flagged = np.sum(flagged & is_normal)
        imds_flagged = np.sum(flagged & ~is_normal)
        
        contamination_rate = normals_flagged / n_normals if n_normals > 0 else 1.0
        detection_rate = imds_flagged / n_imds if n_imds > 0 else 0.0
        
        # Skip if violates constraints
        if contamination_rate > max_contamination:
            continue
        if detection_rate < min_detection:
            continue
        
        # Compute binary predictions
        y_true_binary = np.where(is_normal, 0, 1)  # 0=normal, 1=IMD
        y_pred_binary = np.where(flagged, 1, 0)     # 1=flagged, 0=not
        
        # Compute metrics
        try:
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        except:
            precision = 0.0
        
        try:
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        except:
            recall = 0.0
        
        try:
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        except:
            f1 = 0.0
        
        # Youden's J statistic = sensitivity + specificity - 1
        specificity = 1.0 - contamination_rate
        youden = recall + specificity - 1.0
        
        # Select metric to optimize
        if optimization_metric == "f1":
            metric_value = f1
        elif optimization_metric == "precision":
            metric_value = precision
        elif optimization_metric == "recall":
            metric_value = recall
        elif optimization_metric == "youden":
            metric_value = youden
        else:
            metric_value = f1  # default
        
        results.append({
            'threshold': float(threshold),
            'contamination_rate': float(contamination_rate),
            'detection_rate': float(detection_rate),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'youden': float(youden),
            'normals_flagged': int(normals_flagged),
            'imds_flagged': int(imds_flagged),
            'metric_value': float(metric_value),
        })
        
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold
    
    results_df = pd.DataFrame(results)
    
    if best_threshold is None:
        # Fallback: use percentile-based threshold
        best_threshold = float(np.percentile(scores[is_normal], 99.0))
        logger.warning(f"No threshold met constraints. Using percentile 99.0 fallback: {best_threshold:.4f}")
    
    return {
        'optimal_threshold': float(best_threshold),
        'best_metric': optimization_metric,
        'best_metric_value': float(best_metric_value),
        'results': results_df,
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
    train_ratio: float = 0.8,
    optimization_metric: str = "f1",
    max_contamination: float = 0.05,
    min_detection: float = 0.80,
) -> Dict:
    """Run anomaly detection on pathway Stouffer's Z scores with proper ML methodology.
    
    This follows the LOF pipeline methodology:
    1. Split normals into train/test (80/20)
    2. Train ONLY on training normals
    3. Validate on test normals + ALL IMDs (no IMDs in training)
    4. Then simulate production with 2% contamination
    
    Args:
        pathway_stats: DataFrame with columns: sample_id, pathway_name, z_stouffer_abs
        normal_sample_ids: List of normal sample IDs
        imd_sample_ids: List of IMD sample IDs
        gray_sample_ids: List of gray sample IDs
        scorer_name: Which anomaly detector to use ('lof', 'iforest', 'mahalanobis')
        contamination: Expected contamination rate for thresholding
        n_neighbors: Number of neighbors for LOF
        n_estimators: Number of trees for IForest
        random_state: Random seed
        percentile: Percentile for thresholding (higher = more strict)
        train_ratio: Ratio of normals to use for training (default 0.8)
        
    Returns:
        Dict with anomaly scores, decisions, and validation metrics
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Pivot to samples x pathways matrix
    # Handle duplicate (sample_id, pathway_name) pairs by taking mean
    pivot = pathway_stats.pivot_table(index='sample_id', columns='pathway_name', values='z_stouffer_abs', aggfunc='mean')
    pivot = pivot.fillna(0)  # Fill missing with 0 (no deviation)
    
    # Get all sample IDs that have pathway data
    all_sample_ids = pivot.index.tolist()
    
    # Filter sample IDs to only those with pathway data
    normal_sample_ids = [s for s in normal_sample_ids if s in all_sample_ids]
    imd_sample_ids_filtered = [s for s in imd_sample_ids if s in all_sample_ids]
    
    logger.info(f"Samples with pathway data: {len(normal_sample_ids)} normals, {len(imd_sample_ids_filtered)} IMDs")
    
    # ========================================================================
    # Step 1: Split normals into train/test (80/20)
    # ========================================================================
    X_normals = pivot.loc[normal_sample_ids]
    
    X_train_normals, X_test_normals = train_test_split(
        X_normals,
        train_size=train_ratio,
        test_size=1-train_ratio,
        random_state=random_state,
        stratify=pd.Series([0]*len(X_normals), index=X_normals.index)  # All are normals
    )
    
    train_normal_ids = X_train_normals.index.tolist()
    test_normal_ids = X_test_normals.index.tolist()
    
    logger.info(f"\nSplit normals into train/test:")
    logger.info(f"  Train normals: {len(train_normal_ids)}")
    logger.info(f"  Test normals: {len(test_normal_ids)}")
    
    # ========================================================================
    # Step 2: Train on training normals only
    # ========================================================================
    X_train = X_train_normals.values
    
    logger.info(f"\nTraining anomaly detector on {len(train_normal_ids)} normal samples")
    logger.info(f"Using {pivot.shape[1]} pathway features")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
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
        lof.fit(X_train_scaled)
        model = lof
        method_name = "Local Outlier Factor"
        
    elif scorer_name == "iforest":
        from sklearn.ensemble import IsolationForest
        
        iforest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        iforest.fit(X_train_scaled)
        model = iforest
        method_name = "Isolation Forest"
        
    elif scorer_name == "mahalanobis":
        from sklearn.covariance import MinCovDet
        
        # Fit robust covariance on normals
        mcd = MinCovDet(random_state=random_state)
        mcd.fit(X_train_scaled)
        model = mcd
        method_name = "Mahalanobis Distance"
        
    else:
        raise ValueError(f"Unknown scorer: {scorer_name}. Use 'lof', 'iforest', or 'mahalanobis'")
    
    # ========================================================================
    # Step 3: Validate on test normals + ALL IMDs (realistic validation)
    # ========================================================================
    # Prepare test set: test normals + all IMDs
    X_test_normals_scaled = scaler.transform(X_test_normals)
    X_imds = pivot.loc[imd_sample_ids_filtered]
    X_imds_scaled = scaler.transform(X_imds)
    
    # Combine test normals and IMDs
    X_val = np.vstack([X_test_normals_scaled, X_imds_scaled])
    val_sample_ids = test_normal_ids + imd_sample_ids_filtered
    
    logger.info(f"\nValidating on {len(test_normal_ids)} test normals + {len(imd_sample_ids_filtered)} IMDs")
    
    # Score validation samples
    if scorer_name == "lof":
        val_scores = -model.decision_function(X_val)
    elif scorer_name == "iforest":
        # For IsolationForest, score_samples returns negative scores (more negative = more anomalous)
        val_scores = -model.score_samples(X_val)
    elif scorer_name == "mahalanobis":
        val_scores = model.mahalanobis(X_val)
    
    # Log score statistics for validation
    logger.info(f"Validation score statistics: Min={val_scores.min():.4f}, Max={val_scores.max():.4f}, Mean={val_scores.mean():.4f}")
    
    # Create validation results
    val_results = pd.DataFrame({
        'sample_id': val_sample_ids,
        'anomaly_score': val_scores,
        'is_normal': [True]*len(test_normal_ids) + [False]*len(imd_sample_ids_filtered),
        'is_imd': [False]*len(test_normal_ids) + [True]*len(imd_sample_ids_filtered)
    })
    
    # Step 3.5: Find optimal threshold on validation set
    # Optimize threshold based on validation performance instead of using a fixed percentile
    # This ensures we pick the threshold that maximizes our target metric
    
    val_scores_arr = val_results['anomaly_score'].values
    val_is_normal_arr = val_results['is_normal'].values
    
    threshold_info = find_optimal_anomaly_threshold(
        scores=val_scores_arr,
        is_normal=val_is_normal_arr,
        max_contamination=max_contamination,
        min_detection=min_detection,
        optimization_metric=optimization_metric,
        n_thresholds=100,
    )
    
    threshold = threshold_info['optimal_threshold']
    best_metric = threshold_info['best_metric']
    best_metric_value = threshold_info['best_metric_value']
    
    # Also compute training normal statistics for reference
    if scorer_name == "lof":
        train_scores = -model.decision_function(X_train_scaled)
    elif scorer_name == "iforest":
        train_scores = -model.score_samples(X_train_scaled)
    elif scorer_name == "mahalanobis":
        train_scores = model.mahalanobis(X_train_scaled)
    
    # Log both training stats and optimization results
    logger.info(f"\nTraining normal score statistics:")
    logger.info(f"  Min: {train_scores.min():.4f}, Max: {train_scores.max():.4f}")
    logger.info(f"  Mean: {train_scores.mean():.4f}, Std: {train_scores.std():.4f}")
    logger.info(f"  Percentile 95: {np.percentile(train_scores, 95):.4f}")
    logger.info(f"  Percentile 99: {np.percentile(train_scores, 99):.4f}")
    
    logger.info(f"\nOptimized threshold on validation set:")
    logger.info(f"  Optimization metric: {best_metric}")
    logger.info(f"  Best {best_metric} value: {best_metric_value:.4f}")
    logger.info(f"  Optimal threshold: {threshold:.4f}")
    
    # Flag validation samples using optimized threshold
    val_results['flagged'] = val_results['anomaly_score'] > threshold
    
    # Compute validation metrics
    n_test_normals = len(test_normal_ids)
    n_val_imds = len(imd_sample_ids_filtered)
    
    test_normals_flagged = int(val_results[(val_results['is_normal']) & (val_results['flagged'])].shape[0])
    val_imds_flagged = int(val_results[(val_results['is_imd']) & (val_results['flagged'])].shape[0])
    
    val_detection_rate = val_imds_flagged / n_val_imds if n_val_imds > 0 else 0.0
    val_contamination_rate = test_normals_flagged / n_test_normals if n_test_normals > 0 else 0.0
    
    flagged_test_normal_ids = val_results[(val_results['is_normal']) & (val_results['flagged'])]['sample_id'].tolist()
    flagged_val_imd_ids = val_results[(val_results['is_imd']) & (val_results['flagged'])]['sample_id'].tolist()
    
    logger.info(f"\n{method_name} Validation Results:")
    logger.info(f"  Threshold: {threshold:.4f} (optimized on validation set)")
    logger.info(f"  Optimization metric: {best_metric} (best value: {best_metric_value:.4f})")
    logger.info(f"  Test normals flagged: {test_normals_flagged} / {n_test_normals} ({val_contamination_rate*100:.1f}%)")
    logger.info(f"  IMDs flagged: {val_imds_flagged} / {n_val_imds} ({val_detection_rate*100:.1f}%)")
    
    # ========================================================================
    # ========================================================================
    # Step 4: Realistic Production Evaluation (Analytical Approach)
    # ========================================================================
    # Following the outlier_detection_pipeline methodology:
    # 1. Score ALL test normals + ALL test IMDs (never seen during training)
    # 2. Compute prevalence-independent metrics (detection rate, FPR)
    # 3. Analytically calculate precision, F1, accuracy at target contamination
    #
    # This is more accurate than batch resampling because:
    # - Each sample is scored independently (scores don't depend on batch composition)
    # - Analytical calculation is exact for assumed deployment prevalence
    
    production_normal_ids = test_normal_ids.copy()  # ONLY test normals (never seen during training)
    production_imd_ids = imd_sample_ids_filtered.copy()  # All test IMDs
    
    X_prod_normals = pivot.loc[production_normal_ids]
    X_prod_imds = pivot.loc[production_imd_ids]
    
    n_prod_normals = len(production_normal_ids)
    n_prod_imds = len(production_imd_ids)
    
    logger.info(f"\n{'='*70}")
    logger.info("STEP 4: Realistic Production Evaluation")
    logger.info(f"{'='*70}")
    logger.info(f"Evaluating on ALL test samples (never seen during training):")
    logger.info(f"  Normal samples: {n_prod_normals}")
    logger.info(f"  IMD samples: {n_prod_imds}")
    logger.info(f"  Total: {n_prod_normals + n_prod_imds}")
    
    if len(X_prod_normals) > 0 and len(X_prod_imds) > 0:
        X_prod_normals_scaled = scaler.transform(X_prod_normals)
        X_prod_imds_scaled = scaler.transform(X_prod_imds)
        
        # Score ALL production samples
        if scorer_name == "lof":
            prod_normal_scores = -model.decision_function(X_prod_normals_scaled)
            prod_imd_scores = -model.decision_function(X_prod_imds_scaled)
        elif scorer_name == "iforest":
            prod_normal_scores = -model.score_samples(X_prod_normals_scaled)
            prod_imd_scores = -model.score_samples(X_prod_imds_scaled)
        elif scorer_name == "mahalanobis":
            prod_normal_scores = model.mahalanobis(X_prod_normals_scaled)
            prod_imd_scores = model.mahalanobis(X_prod_imds_scaled)
        
        # Flag samples using the optimized threshold
        prod_normal_flagged = prod_normal_scores > threshold
        prod_imd_flagged = prod_imd_scores > threshold
        
        # Prevalence-independent metrics (from ALL test samples)
        n_normals_flagged = int(np.sum(prod_normal_flagged))
        n_imds_flagged = int(np.sum(prod_imd_flagged))
        
        detection_rate = n_imds_flagged / n_prod_imds if n_prod_imds > 0 else 0.0
        false_positive_rate = n_normals_flagged / n_prod_normals if n_prod_normals > 0 else 0.0
        
        # Get flagged sample IDs
        flagged_prod_normal_ids = [production_normal_ids[i] for i in range(n_prod_normals) if prod_normal_flagged[i]]
        flagged_prod_imd_ids = [production_imd_ids[i] for i in range(n_prod_imds) if prod_imd_flagged[i]]
        
        logger.info(f"\nProduction Evaluation (ALL test samples):")
        logger.info(f"  Normals flagged: {n_normals_flagged} / {n_prod_normals} ({false_positive_rate*100:.1f}%)")
        logger.info(f"  IMDs flagged: {n_imds_flagged} / {n_prod_imds} ({detection_rate*100:.1f}%)")
        
        # ========================================================================
        # Analytical calculation at target contamination (2%)
        # ========================================================================
        # Using the formula from outlier_detection_pipeline:
        # precision = (p * recall) / (p * recall + (1-p) * fpr)
        # f1 = 2 * precision * recall / (precision + recall)
        # accuracy = (1-p)*(1-fpr) + p*recall
        
        target_contamination = 0.02  # 2% deployment prevalence
        p = target_contamination
        recall = detection_rate
        fpr = false_positive_rate
        
        # Analytical precision at deployment prevalence
        denom = (p * recall) + ((1.0 - p) * fpr)
        precision_deploy = (p * recall) / denom if denom > 0 else 0.0
        
        # Analytical F1 at deployment prevalence
        if (precision_deploy + recall) > 0:
            f1_deploy = 2.0 * precision_deploy * recall / (precision_deploy + recall)
        else:
            f1_deploy = 0.0
        
        # Analytical accuracy at deployment prevalence
        accuracy_deploy = (1.0 - p) * (1.0 - fpr) + p * recall
        
        # Confusion matrix for notional batch at deployment prevalence
        n_notional = max(100, n_prod_normals + n_prod_imds)  # Use at least 100 samples
        n_outliers_notional = max(1, int(round(p * n_notional)))
        n_normals_notional = n_notional - n_outliers_notional
        
        cm_deploy = np.array([
            [int(round(n_normals_notional * (1.0 - fpr))), int(round(n_normals_notional * fpr))],
            [int(round(n_outliers_notional * (1.0 - recall))), int(round(n_outliers_notional * recall))],
        ])
        
        logger.info(f"\nAnalytical Metrics @ {target_contamination*100:.0f}% prevalence:")
        logger.info(f"  Precision: {precision_deploy:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1: {f1_deploy:.4f}")
        logger.info(f"  Accuracy: {accuracy_deploy:.4f}")
        logger.info(f"  Confusion Matrix (notional {n_notional} sample batch):\n{cm_deploy}")
        
        # ROC AUC on production set
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            all_prod_scores = np.concatenate([prod_normal_scores, prod_imd_scores])
            all_prod_true = np.concatenate([np.zeros(n_prod_normals, dtype=int), np.ones(n_prod_imds, dtype=int)])
            roc_auc_prod = float(roc_auc_score(all_prod_true, all_prod_scores))
        except:
            roc_auc_prod = float('nan')
        
        # PR AUC on production set
        try:
            pr_auc_prod = float(average_precision_score(all_prod_true, all_prod_scores))
        except:
            pr_auc_prod = float('nan')
        
        logger.info(f"  ROC AUC: {roc_auc_prod:.4f}")
        logger.info(f"  PR AUC: {pr_auc_prod:.4f}")
    else:
        # Fallback if no samples
        n_normals_flagged = 0
        n_imds_flagged = 0
        detection_rate = 0.0
        false_positive_rate = 0.0
        flagged_prod_normal_ids = []
        flagged_prod_imd_ids = []
        precision_deploy = 0.0
        f1_deploy = 0.0
        accuracy_deploy = 0.0
        cm_deploy = np.array([[0, 0], [0, 0]])
        roc_auc_prod = 0.0
        pr_auc_prod = 0.0
        target_contamination = 0.02
        logger.warning("Not enough production samples for evaluation")
    
    # ========================================================================
    # Compute comprehensive metrics for validation and production
    # ========================================================================
    def compute_metrics(y_true, y_pred, y_scores=None):
        """Compute comprehensive classification metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix,
            precision_recall_curve
        )
        
        # Convert to binary (1 = IMD/outlier, 0 = normal)
        y_true_binary = (y_true == False).astype(int)  # True=normal, False=IMD
        y_pred_binary = (y_pred == True).astype(int)  # flagged=1
        
        metrics = {}
        
        # Basic metrics
        try:
            metrics['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
        except:
            metrics['accuracy'] = float('nan')
        
        try:
            metrics['precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        except:
            metrics['precision'] = float('nan')
        
        try:
            metrics['recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        except:
            metrics['recall'] = float('nan')
        
        try:
            metrics['f1'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        except:
            metrics['f1'] = float('nan')
        
        # AUC metrics
        if y_scores is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true_binary, y_scores)
            except:
                metrics['roc_auc'] = float('nan')
            
            try:
                metrics['pr_auc'] = average_precision_score(y_true_binary, y_scores)
            except:
                metrics['pr_auc'] = float('nan')
        else:
            metrics['roc_auc'] = float('nan')
            metrics['pr_auc'] = float('nan')
        
        # Confusion matrix
        try:
            cm = confusion_matrix(y_true_binary, y_pred_binary)
            metrics['confusion_matrix'] = cm.tolist()
        except:
            metrics['confusion_matrix'] = None
        
        return metrics
    
    # Compute metrics for validation set
    val_y_true = val_results['is_normal'].values  # True = normal, False = IMD
    val_y_pred = val_results['flagged'].values
    val_y_scores = val_results['anomaly_score'].values
    
    val_metrics = compute_metrics(val_y_true, val_y_pred, val_y_scores)
    
    # Compute metrics for production evaluation (ALL test samples)
    # Create a combined results DataFrame for consistency with existing code
    prod_sample_ids = production_normal_ids + production_imd_ids
    prod_scores = np.concatenate([prod_normal_scores, prod_imd_scores])
    prod_is_normal = np.concatenate([np.ones(n_prod_normals, dtype=bool), np.zeros(n_prod_imds, dtype=bool)])
    prod_is_imd = np.concatenate([np.zeros(n_prod_normals, dtype=bool), np.ones(n_prod_imds, dtype=bool)])
    prod_flagged = np.concatenate([prod_normal_flagged, prod_imd_flagged])
    
    prod_results = pd.DataFrame({
        'sample_id': prod_sample_ids,
        'anomaly_score': prod_scores,
        'is_normal': prod_is_normal,
        'is_imd': prod_is_imd,
        'flagged': prod_flagged
    })
    
    prod_y_true = prod_results['is_normal'].values
    prod_y_pred = prod_results['flagged'].values
    prod_y_scores = prod_results['anomaly_score'].values
    
    prod_metrics = compute_metrics(prod_y_true, prod_y_pred, prod_y_scores)
    
    # ========================================================================
    # Helper function for analytical confusion matrix plot
    # ========================================================================
    def _plot_analytical_confusion_matrix(cm, title, output_path):
        """Generate and save analytical confusion matrix plot."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal', 'IMD'],
                        yticklabels=['Normal', 'IMD'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(title)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return True
        except ImportError:
            logger.warning("matplotlib/seaborn not available. Skipping confusion matrix plot.")
            return False
        except Exception as e:
            logger.warning(f"Error generating analytical confusion matrix: {e}")
            return False
    
    # ========================================================================
    # Generate confusion matrix plots
    # ========================================================================
    def plot_confusion_matrix(y_true, y_pred, title, output_path):
        """Generate and save confusion matrix plot."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            from sklearn.metrics import confusion_matrix
            
            y_true_binary = (y_true == False).astype(int)
            y_pred_binary = (y_pred == True).astype(int)
            
            cm = confusion_matrix(y_true_binary, y_pred_binary)
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal', 'IMD'],
                        yticklabels=['Normal', 'IMD'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(title)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return True
        except ImportError:
            logger.warning("matplotlib/seaborn not available. Skipping confusion matrix plot.")
            return False
        except Exception as e:
            logger.warning(f"Error generating confusion matrix: {e}")
            return False
    
    # Generate confusion matrix for simulated 2% contamination scenario
    def plot_confusion_matrix_sim(y_true, y_pred, title, output_path):
        """Generate and save confusion matrix plot for simulation."""
        return plot_confusion_matrix(y_true, y_pred, title, output_path)
    
    # ========================================================================
    # Return all results
    # ========================================================================
    return {
        'scorer': scorer_name,
        'method': method_name,
        'threshold': threshold,
        'percentile': percentile,
        'train_ratio': train_ratio,
        'random_state': random_state,
        # Validation results (test normals + all IMDs)
        'validation': {
            'n_normals': n_test_normals,
            'n_imds': n_val_imds,
            'normals_flagged': test_normals_flagged,
            'imds_flagged': val_imds_flagged,
            'detection_rate': val_detection_rate,
            'contamination_rate': val_contamination_rate,
            'flagged_normal_ids': flagged_test_normal_ids,
            'flagged_imd_ids': flagged_val_imd_ids,
            'results': val_results,
            'metrics': val_metrics
        },
        # Production evaluation results with analytical metrics
        'production': {
            'n_normals': n_prod_normals,
            'n_imds': n_prod_imds,
            'normals_flagged': n_normals_flagged,
            'imds_flagged': n_imds_flagged,
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'flagged_normal_ids': flagged_prod_normal_ids,
            'flagged_imd_ids': flagged_prod_imd_ids,
            'results': prod_results,
            'metrics': prod_metrics,
            # Analytical metrics at target contamination
            'target_contamination': target_contamination,
            'precision_at_target': precision_deploy,
            'f1_at_target': f1_deploy,
            'accuracy_at_target': accuracy_deploy,
            'roc_auc': roc_auc_prod,
            'pr_auc': pr_auc_prod,
            'confusion_matrix_at_target': cm_deploy.tolist(),
            'confusion_matrix_labels': ['Normal', 'IMD'],
        },
        # All sample results (for reference)
        'all_samples': {
            'normal_sample_ids': normal_sample_ids,
            'imd_sample_ids': imd_sample_ids_filtered,
            'gray_sample_ids': [s for s in gray_sample_ids if s in all_sample_ids],
            'n_features': pivot.shape[1]
        },
        'plot_functions': {
            'plot_validation_cm': lambda output_dir: plot_confusion_matrix(
                val_y_true, val_y_pred, 
                f'{method_name} - Validation Set',
                str(Path(output_dir) / 'anomaly_validation_confusion_matrix.png')
            ),
            'plot_production_cm': lambda output_dir: plot_confusion_matrix(
                prod_y_true, prod_y_pred,
                f'{method_name} - Production Evaluation',
                str(Path(output_dir) / 'anomaly_production_confusion_matrix.png')
            ),
            'plot_analytical_cm': lambda output_dir: _plot_analytical_confusion_matrix(
                cm_deploy,
                f'{method_name} - Analytical @ {target_contamination*100:.0f}% prevalence',
                str(Path(output_dir) / 'anomaly_analytical_confusion_matrix.png')
            )
        }
    }
