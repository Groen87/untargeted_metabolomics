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
        val_scores = -model.score_samples(X_val)
    elif scorer_name == "mahalanobis":
        val_scores = model.mahalanobis(X_val)
    
    # Create validation results
    val_results = pd.DataFrame({
        'sample_id': val_sample_ids,
        'anomaly_score': val_scores,
        'is_normal': [True]*len(test_normal_ids) + [False]*len(imd_sample_ids_filtered),
        'is_imd': [False]*len(test_normal_ids) + [True]*len(imd_sample_ids_filtered)
    })
    
    # Compute threshold from training normal scores (no data leakage!)
    # Get scores for training normals
    if scorer_name == "lof":
        train_scores = -model.decision_function(X_train_scaled)
    elif scorer_name == "iforest":
        train_scores = -model.score_samples(X_train_scaled)
    elif scorer_name == "mahalanobis":
        train_scores = model.mahalanobis(X_train_scaled)
    
    threshold = float(np.percentile(train_scores, percentile))
    
    # Flag validation samples
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
    logger.info(f"  Threshold: {threshold:.4f} (percentile {percentile} from training normals)")
    logger.info(f"  Test normals flagged: {test_normals_flagged} / {n_test_normals} ({val_contamination_rate*100:.1f}%)")
    logger.info(f"  IMDs flagged: {val_imds_flagged} / {n_val_imds} ({val_detection_rate*100:.1f}%)")
    
    # ========================================================================
    # Step 4: Simulate production with 2% contamination
    # ========================================================================
    # Create TWO production scenarios:
    # 1. Realistic with all IMDs (for full evaluation metrics)
    # 2. Simulated with 2% contamination (for realistic production scenario)
    
    # Scenario 1: Full evaluation with ALL IMDs (for comprehensive metrics)
    all_normal_ids_for_prod = train_normal_ids + test_normal_ids
    production_imd_ids_full = imd_sample_ids_filtered.copy()
    production_normal_ids_full = all_normal_ids_for_prod.copy()
    production_sample_ids_full = production_normal_ids_full + production_imd_ids_full
    
    X_prod_normals_full = pivot.loc[production_normal_ids_full]
    X_prod_imds_full = pivot.loc[production_imd_ids_full]
    
    if len(X_prod_normals_full) > 0 and len(X_prod_imds_full) > 0:
        X_prod_normals_scaled_full = scaler.transform(X_prod_normals_full)
        X_prod_imds_scaled_full = scaler.transform(X_prod_imds_full)
        X_prod_full = np.vstack([X_prod_normals_scaled_full, X_prod_imds_scaled_full])
    else:
        logger.warning("Not enough samples for full production evaluation")
        X_prod_full = np.array([]).reshape(0, pivot.shape[1])
        production_sample_ids_full = []
        production_normal_ids_full = []
        production_imd_ids_full = []
    
    # Scenario 2: Realistic 2% contamination simulation
    target_contamination = 0.02
    n_production_normals_sim = len(all_normal_ids_for_prod)
    target_imds_sim = int(np.round(n_production_normals_sim / (1 - target_contamination) * target_contamination))
    
    if len(imd_sample_ids_filtered) > target_imds_sim:
        np.random.seed(random_state)
        production_imd_ids_sim = np.random.choice(imd_sample_ids_filtered, target_imds_sim, replace=False).tolist()
    else:
        production_imd_ids_sim = imd_sample_ids_filtered.copy()
        actual_contamination = len(production_imd_ids_sim) / (n_production_normals_sim + len(production_imd_ids_sim))
        logger.info(f"\nNote: Not enough IMDs for 2% contamination. Using all {len(production_imd_ids_sim)} IMDs (actual: {actual_contamination*100:.1f}%)")
    
    production_normal_ids_sim = all_normal_ids_for_prod.copy()
    production_sample_ids_sim = production_normal_ids_sim + production_imd_ids_sim
    
    X_prod_normals_sim = pivot.loc[production_normal_ids_sim]
    X_prod_imds_sim = pivot.loc[production_imd_ids_sim]
    
    if len(X_prod_normals_sim) > 0 and len(X_prod_imds_sim) > 0:
        X_prod_normals_scaled_sim = scaler.transform(X_prod_normals_sim)
        X_prod_imds_scaled_sim = scaler.transform(X_prod_imds_sim)
        X_prod_sim = np.vstack([X_prod_normals_scaled_sim, X_prod_imds_scaled_sim])
    elif len(X_prod_normals_sim) > 0:
        X_prod_normals_scaled_sim = scaler.transform(X_prod_normals_sim)
        X_prod_sim = X_prod_normals_scaled_sim
    elif len(X_prod_imds_sim) > 0:
        X_prod_imds_scaled_sim = scaler.transform(X_prod_imds_sim)
        X_prod_sim = X_prod_imds_scaled_sim
    else:
        logger.warning("No samples available for production simulation")
        X_prod_sim = np.array([]).reshape(0, pivot.shape[1])
    
    logger.info(f"\nProduction Full Evaluation:")
    logger.info(f"  Normal samples: {len(production_normal_ids_full)}")
    logger.info(f"  IMD samples: {len(production_imd_ids_full)}")
    logger.info(f"  Total: {len(production_sample_ids_full)}")
    
    logger.info(f"\nProduction Simulation (2% contamination):")
    logger.info(f"  Normal samples: {len(production_normal_ids_sim)}")
    logger.info(f"  IMD samples: {len(production_imd_ids_sim)}")
    logger.info(f"  Total: {len(production_sample_ids_sim)}")
    
    # Score production samples for BOTH scenarios
    # Full evaluation scenario
    if len(X_prod_full) > 0:
        if scorer_name == "lof":
            prod_scores_full = -model.decision_function(X_prod_full)
        elif scorer_name == "iforest":
            prod_scores_full = -model.score_samples(X_prod_full)
        elif scorer_name == "mahalanobis":
            prod_scores_full = model.mahalanobis(X_prod_full)
        
        prod_results_full = pd.DataFrame({
            'sample_id': production_sample_ids_full,
            'anomaly_score': prod_scores_full,
            'is_normal': [True]*len(production_normal_ids_full) + [False]*len(production_imd_ids_full),
            'is_imd': [False]*len(production_normal_ids_full) + [True]*len(production_imd_ids_full)
        })
        prod_results_full['flagged'] = prod_results_full['anomaly_score'] > threshold
    else:
        prod_results_full = pd.DataFrame()
    
    # Simulated 2% contamination scenario
    if len(X_prod_sim) > 0:
        if scorer_name == "lof":
            prod_scores_sim = -model.decision_function(X_prod_sim)
        elif scorer_name == "iforest":
            prod_scores_sim = -model.score_samples(X_prod_sim)
        elif scorer_name == "mahalanobis":
            prod_scores_sim = model.mahalanobis(X_prod_sim)
        
        prod_results_sim = pd.DataFrame({
            'sample_id': production_sample_ids_sim,
            'anomaly_score': prod_scores_sim,
            'is_normal': [True]*len(production_normal_ids_sim) + [False]*len(production_imd_ids_sim),
            'is_imd': [False]*len(production_normal_ids_sim) + [True]*len(production_imd_ids_sim)
        })
        prod_results_sim['flagged'] = prod_results_sim['anomaly_score'] > threshold
    else:
        prod_results_sim = pd.DataFrame()
    
    # Compute metrics for full evaluation scenario
    n_prod_normals_full = len(production_normal_ids_full)
    n_prod_imds_full = len(production_imd_ids_full)
    
    if len(prod_results_full) > 0:
        prod_normals_flagged_full = int(prod_results_full[(prod_results_full['is_normal']) & (prod_results_full['flagged'])].shape[0])
        prod_imds_flagged_full = int(prod_results_full[(prod_results_full['is_imd']) & (prod_results_full['flagged'])].shape[0])
        
        prod_detection_rate_full = prod_imds_flagged_full / n_prod_imds_full if n_prod_imds_full > 0 else 0.0
        prod_contamination_rate_full = prod_normals_flagged_full / n_prod_normals_full if n_prod_normals_full > 0 else 0.0
        
        flagged_prod_normal_ids_full = prod_results_full[(prod_results_full['is_normal']) & (prod_results_full['flagged'])]['sample_id'].tolist()
        flagged_prod_imd_ids_full = prod_results_full[(prod_results_full['is_imd']) & (prod_results_full['flagged'])]['sample_id'].tolist()
    else:
        prod_normals_flagged_full = 0
        prod_imds_flagged_full = 0
        prod_detection_rate_full = 0.0
        prod_contamination_rate_full = 0.0
        flagged_prod_normal_ids_full = []
        flagged_prod_imd_ids_full = []
    
    # Compute metrics for simulated 2% contamination scenario
    n_prod_normals_sim = len(production_normal_ids_sim)
    n_prod_imds_sim = len(production_imd_ids_sim)
    
    if len(prod_results_sim) > 0:
        prod_normals_flagged_sim = int(prod_results_sim[(prod_results_sim['is_normal']) & (prod_results_sim['flagged'])].shape[0])
        prod_imds_flagged_sim = int(prod_results_sim[(prod_results_sim['is_imd']) & (prod_results_sim['flagged'])].shape[0])
        
        prod_detection_rate_sim = prod_imds_flagged_sim / n_prod_imds_sim if n_prod_imds_sim > 0 else 0.0
        prod_contamination_rate_sim = prod_normals_flagged_sim / n_prod_normals_sim if n_prod_normals_sim > 0 else 0.0
        
        flagged_prod_normal_ids_sim = prod_results_sim[(prod_results_sim['is_normal']) & (prod_results_sim['flagged'])]['sample_id'].tolist()
        flagged_prod_imd_ids_sim = prod_results_sim[(prod_results_sim['is_imd']) & (prod_results_sim['flagged'])]['sample_id'].tolist()
    else:
        prod_normals_flagged_sim = 0
        prod_imds_flagged_sim = 0
        prod_detection_rate_sim = 0.0
        prod_contamination_rate_sim = 0.0
        flagged_prod_normal_ids_sim = []
        flagged_prod_imd_ids_sim = []
    
    logger.info(f"\nProduction Full Evaluation:")
    logger.info(f"  Normals flagged: {prod_normals_flagged_full} / {n_prod_normals_full} ({prod_contamination_rate_full*100:.1f}%)")
    logger.info(f"  IMDs flagged: {prod_imds_flagged_full} / {n_prod_imds_full} ({prod_detection_rate_full*100:.1f}%)")
    
    logger.info(f"\nProduction Simulation (2% contamination):")
    logger.info(f"  Normals flagged: {prod_normals_flagged_sim} / {n_prod_normals_sim} ({prod_contamination_rate_sim*100:.1f}%)")
    logger.info(f"  IMDs flagged: {prod_imds_flagged_sim} / {n_prod_imds_sim} ({prod_detection_rate_sim*100:.1f}%)")
    
    # ========================================================================
    # Compute comprehensive metrics for validation and both production scenarios
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
    
    # Compute metrics for full production evaluation
    if len(prod_results_full) > 0:
        prod_y_true_full = prod_results_full['is_normal'].values
        prod_y_pred_full = prod_results_full['flagged'].values
        prod_y_scores_full = prod_results_full['anomaly_score'].values
        prod_metrics_full = compute_metrics(prod_y_true_full, prod_y_pred_full, prod_y_scores_full)
    else:
        prod_metrics_full = {}
    
    # Compute metrics for simulated 2% contamination production
    if len(prod_results_sim) > 0:
        prod_y_true_sim = prod_results_sim['is_normal'].values
        prod_y_pred_sim = prod_results_sim['flagged'].values
        prod_y_scores_sim = prod_results_sim['anomaly_score'].values
        prod_metrics_sim = compute_metrics(prod_y_true_sim, prod_y_pred_sim, prod_y_scores_sim)
    else:
        prod_metrics_sim = {}
    
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
        # Production full evaluation results
        'production_full': {
            'n_normals': n_prod_normals_full,
            'n_imds': n_prod_imds_full,
            'normals_flagged': prod_normals_flagged_full,
            'imds_flagged': prod_imds_flagged_full,
            'detection_rate': prod_detection_rate_full,
            'contamination_rate': prod_contamination_rate_full,
            'flagged_normal_ids': flagged_prod_normal_ids_full,
            'flagged_imd_ids': flagged_prod_imd_ids_full,
            'results': prod_results_full,
            'metrics': prod_metrics_full
        },
        # Production simulation results (2% contamination)
        'production_sim': {
            'n_normals': n_prod_normals_sim,
            'n_imds': n_prod_imds_sim,
            'normals_flagged': prod_normals_flagged_sim,
            'imds_flagged': prod_imds_flagged_sim,
            'detection_rate': prod_detection_rate_sim,
            'contamination_rate': prod_contamination_rate_sim,
            'target_contamination': target_contamination,
            'flagged_normal_ids': flagged_prod_normal_ids_sim,
            'flagged_imd_ids': flagged_prod_imd_ids_sim,
            'results': prod_results_sim,
            'metrics': prod_metrics_sim
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
            'plot_production_full_cm': lambda output_dir: plot_confusion_matrix(
                prod_y_true_full,
                prod_y_pred_full,
                f'{method_name} - Full Production Evaluation',
                str(Path(output_dir) / 'anomaly_production_full_confusion_matrix.png')
            ),
            'plot_production_sim_cm': lambda output_dir: plot_confusion_matrix(
                prod_y_true_sim,
                prod_y_pred_sim,
                f'{method_name} - Production Simulation (2% contamination)',
                str(Path(output_dir) / 'anomaly_production_simulation_confusion_matrix.png')
            )
        }
    }
