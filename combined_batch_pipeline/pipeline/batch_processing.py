"""
Batch processing module for combined batch pipeline.

This module handles:
1. Processing individual batches (median normalization, LOESS drift correction)
2. Merging processed batches
3. Preparing data for ComBat correction

Uses injection order from metadata for LOESS drift correction.
"""

from typing import Tuple, Dict, List, Optional
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from .data_loader import extract_batch_from_filename, extract_sample_id_from_column, extract_sample_type
from .feature_filtering import filter_features, FeatureFilter

logger = logging.getLogger(__name__)


def identify_qc_samples(
    sample_cols: List[str],
    sample_info: Optional[Dict[str, Dict]] = None,
    qc_pattern: str = "expQC",
    fallback_pattern: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Identify QC and biological samples from column names.
    
    Args:
        sample_cols: List of sample column names
        sample_info: Optional dictionary with sample metadata
        qc_pattern: Pattern to identify QC samples (default: "expQC")
        fallback_pattern: Fallback pattern if primary yields no samples
        
    Returns:
        Tuple of (qc_samples, bio_samples) column names
    """
    qc_samples = []
    bio_samples = []
    
    for col in sample_cols:
        if sample_info and col in sample_info:
            sample_type = sample_info[col].get('sample_type', 'Sample')
            if sample_type == "QC":
                qc_samples.append(col)
                continue
        
        # Fallback to pattern matching
        sample_id = extract_sample_id_from_column(col)
        sample_type = extract_sample_type(sample_id)
        
        if qc_pattern in sample_id:
            qc_samples.append(col)
        elif fallback_pattern and fallback_pattern in sample_id:
            qc_samples.append(col)
        elif sample_type == "QC":
            qc_samples.append(col)
        else:
            bio_samples.append(col)
    
    return qc_samples, bio_samples


def pqn_normalize_batch(
    df: pd.DataFrame,
    sample_cols: List[str],
    sample_info: Optional[Dict[str, Dict]] = None,
    qc_pattern: str = "expQC",
    fallback_qc_pattern: Optional[str] = "QC3",
) -> pd.DataFrame:
    """
    Apply PQN (Probabilistic Quotient Normalization) to a batch.
    
    PQN normalizes each sample to the median of QC samples.
    Formula: normalized_value = raw_value * (qc_median / sample_median)
    
    This is more robust than simple median normalization as it uses
    the QC sample median as the reference point.
    
    Args:
        df: DataFrame with features as rows, samples as columns
        sample_cols: List of sample column names to normalize
        sample_info: Optional dictionary with sample metadata
        qc_pattern: Pattern to identify QC samples
        fallback_qc_pattern: Fallback QC pattern
        
    Returns:
        DataFrame with PQN-normalized values
    """
    # Identify QC samples
    qc_samples, _ = identify_qc_samples(sample_cols, sample_info, qc_pattern, fallback_qc_pattern)
    
    if not qc_samples:
        logger.warning(f"No QC samples found. Skipping PQN normalization.")
        return df.copy()
    
    logger.info(f"Using {len(qc_samples)} QC samples for PQN normalization: {qc_samples[:3]}...")
    
    # Calculate median for each sample across all features
    sample_medians = df[sample_cols].median(axis=0)
    
    # Calculate QC median (median of all QC sample values across all features)
    qc_values = df[qc_samples].values.flatten()
    qc_median = np.median(qc_values[~np.isnan(qc_values)]) if len(qc_values) > 0 else 1.0
    
    logger.info(f"QC median: {qc_median:.2f}")
    
    # Apply PQN normalization
    # Each sample is scaled so its median equals the QC median
    df_normalized = df.copy()
    for col in sample_cols:
        sample_median = sample_medians[col]
        if sample_median > 0 and not np.isnan(sample_median):
            df_normalized[col] = df[col] * (qc_median / sample_median)
        else:
            # If sample median is 0 or NaN, keep original values
            df_normalized[col] = df[col]
    
    return df_normalized


def loess_drift_correction(
    df: pd.DataFrame,
    sample_cols: List[str],
    sample_info: Optional[Dict[str, Dict]] = None,
    injection_order: Optional[Dict[str, int]] = None,
    qc_pattern: str = "expQC",
    fallback_qc_pattern: Optional[str] = "QC3",
    frac: float = 0.5,
) -> pd.DataFrame:

    """
    Apply LOESS drift correction to a batch.
    
    Uses injection order from metadata to properly order samples.
    Only applies LOESS to features with significant time-dependent trend in QC samples.
    Reports QC CV before/after for features that were corrected.
    
    Args:
        df: DataFrame with features as rows, samples as columns
        sample_cols: List of sample column names
        sample_info: Optional dictionary with sample metadata
        injection_order: Optional dictionary mapping columns to injection order index
        qc_pattern: Pattern to identify QC samples
        fallback_qc_pattern: Fallback QC pattern
        frac: LOESS fraction parameter
        
    Returns:
        DataFrame with drift-corrected values
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess
    from scipy.stats import linregress
    import warnings
    
    # Identify QC samples
    qc_samples, bio_samples = identify_qc_samples(sample_cols, sample_info, qc_pattern, fallback_qc_pattern)
    
    if len(qc_samples) < 2:
        logger.warning(f"Need at least 2 QC samples for LOESS. Found {len(qc_samples)}. Skipping drift correction.")
        return df.copy()
    
    logger.info(f"Applying LOESS drift correction with {len(qc_samples)} QC samples")
    
    # Sort samples by injection order
    if injection_order:
        sorted_samples = sorted(sample_cols, key=lambda col: injection_order.get(col, float('inf')))
    else:
        sorted_samples = sample_cols
        logger.warning("No injection order provided, using column order")
    
    # Filter QC and bio samples
    qc_samples = [col for col in qc_samples if col in sorted_samples]
    bio_samples = [col for col in bio_samples if col in sorted_samples]
    
    if len(qc_samples) < 2:
        logger.warning(f"After filtering, only {len(qc_samples)} QC samples remain. Skipping.")
        return df.copy()
    
    # Create position mapping
    position_idx = {col: i for i, col in enumerate(sorted_samples)}
    qc_positions = [position_idx[col] for col in qc_samples]
    qc_positions_array = np.array(qc_positions)
    
    # Pre-compute nearest QC for each bio sample
    bio_to_qc = {}
    for bio_col in bio_samples:
        bio_pos = position_idx[bio_col]
        nearest_qc_pos = min(qc_positions, key=lambda x: abs(x - bio_pos))
        nearest_qc_idx = qc_positions.index(nearest_qc_pos)
        bio_to_qc[bio_col] = qc_samples[nearest_qc_idx]
    
    # Get QC data for all features at once (vectorized)
    qc_df = df.loc[:, qc_samples]
    
    # Select top 50% highest intensity features (by mean across QC samples)
    qc_means = qc_df.mean(axis=1)
    threshold_mean = np.percentile(qc_means, 50)
    high_intensity_features = qc_means[qc_means >= threshold_mean].index
    
    logger.info(f"  Selecting top 50% features by intensity: {len(high_intensity_features)}/{len(qc_means)} features")
    
    # Check for significant time trend using linear regression (only on high-intensity features)
    slopes = []
    p_values = []
    feature_list = []
    
    for feature in qc_df.index.unique():
        if feature not in high_intensity_features:
            # Skip low-intensity features
            slopes.append(0)
            p_values.append(1.0)
            feature_list.append(feature)
            continue
            
        try:
            feature_qc = qc_df.loc[feature, qc_samples].values
            slope, intercept, r_value, p_value, std_err = linregress(qc_positions_array, feature_qc)
            slopes.append(slope)
            p_values.append(p_value)
            feature_list.append(feature)
        except:
            slopes.append(0)
            p_values.append(1.0)
            feature_list.append(feature)
    
    # Features with significant drift: p < 0.05 and |slope| > 1e-6
    has_drift = np.array([p < 0.05 and abs(s) > 1e-6 for s, p in zip(slopes, p_values)])
    features_with_drift = np.sum(has_drift)
    total_features = len(qc_df.index.unique())
    
    logger.info(f"  Features with significant drift (from high-intensity): {features_with_drift}/{total_features}")
    
    # Calculate QC CV before correction for features with drift
    qc_means_before = qc_df.mean(axis=1)
    qc_stds_before = qc_df.std(axis=1)
    qc_cvs_before = (qc_stds_before / qc_means_before).fillna(0)
    
    df_corrected = df.copy()
    features_corrected = 0
    
    # Apply LOESS only to features with significant drift
    drift_features = qc_df.index.unique()[has_drift]
    
    for feature in drift_features:
        feature_values = df.loc[feature, sorted_samples]
        if isinstance(feature_values, pd.DataFrame):
            feature_data = feature_values.mean(axis=0).values
        else:
            feature_data = feature_values.values
        
        qc_intensities = feature_data[qc_positions_array]
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                smoothed = lowess(qc_intensities, qc_positions_array, frac=frac)
            
            qc_mean = np.mean(qc_intensities)
            smoothed_values = smoothed[:, 1]
            smoothed_safe = np.where(smoothed_values > 0, smoothed_values, qc_mean)
            
            # Apply LOESS correction to ALL samples (QC and biological) based on injection position
            # This is the standard approach: fit to QC, correct all samples
            
            # Get positions for ALL samples (QC + biological)
            all_samples_sorted = sorted_samples
            all_positions = np.array([position_idx[col] for col in all_samples_sorted])
            
            # Interpolate LOESS at all sample positions
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    all_smoothed = lowess(qc_intensities, qc_positions_array, frac=frac, xvals=all_positions)
                
                # Apply correction to all samples
                for k, col in enumerate(all_samples_sorted):
                    smoothed_value = all_smoothed[k, 1]
                    smoothed_safe = smoothed_value if smoothed_value > 0 else qc_mean
                    correction = qc_mean / smoothed_safe
                    df_corrected.loc[feature, col] = feature_data[position_idx[col]] * correction
                    
            except:
                # Fallback: apply correction to QC samples only, bio samples use nearest QC
                for i, col in enumerate(qc_samples):
                    correction = qc_mean / smoothed_safe[i]
                    df_corrected.loc[feature, col] *= correction
                for bio_col in bio_samples:
                    df_corrected.loc[feature, bio_col] = df_corrected.loc[feature, bio_to_qc[bio_col]]
            
            features_corrected += 1
            
        except Exception as e:
            logger.debug(f"Failed LOESS for {feature}: {e}")
            continue
    
    # Calculate QC CV after correction for corrected features only
    qc_df_after = df_corrected[qc_samples]
    qc_means_after = qc_df_after.mean(axis=1)
    qc_stds_after = qc_df_after.std(axis=1)
    qc_cvs_after = (qc_stds_after / qc_means_after).fillna(0)
    
    # CV improvement for features that were corrected
    cv_before_drift = qc_cvs_before[has_drift]
    cv_after_drift = qc_cvs_after.reindex(qc_cvs_before.index[has_drift])
    cv_improvement = cv_before_drift - cv_after_drift
    mean_cv_improvement = cv_improvement.mean()
    features_improved = (cv_improvement > 0).sum()
    
    logger.info(f"  Features corrected with LOESS: {features_corrected}/{features_with_drift}")
    logger.info(f"  Mean QC CV improvement (corrected features): {mean_cv_improvement:.4f}")
    logger.info(f"  Mean QC RSD% before: {cv_before_drift.mean()*100:.2f}%")
    logger.info(f"  Mean QC RSD% after: {cv_after_drift.mean()*100:.2f}%")
    logger.info(f"  Features with improved CV: {features_improved}/{features_corrected}")
    
    return df_corrected


def process_batch(
    df: pd.DataFrame,
    batch: str,
    batch_samples: List[str],
    sample_info: Optional[Dict[str, Dict]] = None,
    injection_order: Optional[Dict[str, int]] = None,
    qc_pattern: str = "expQC",
    fallback_qc_pattern: Optional[str] = "QC3",
    frac: float = 0.5,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a single batch: feature filtering + median normalization + LOESS drift correction.
    
    Args:
        df: Full DataFrame with all features and samples
        batch: Batch name
        batch_samples: List of sample column names belonging to this batch
        sample_info: Optional dictionary with sample metadata
        injection_order: Optional dictionary mapping columns to injection order index
        qc_pattern: Pattern to identify QC samples
        fallback_qc_pattern: Fallback QC pattern
        frac: LOESS fraction parameter
        output_dir: Optional directory to save intermediate files
        filter_config: Optional configuration dictionary for feature filtering
        
    Returns:
        Tuple of:
        - df_processed: DataFrame with processed batch data
        - batch_metadata: DataFrame with sample metadata
    """
    logger.info(f"\nProcessing batch: {batch}")
    logger.info(f"  Samples: {len(batch_samples)}")
    logger.debug(f"  Batch samples: {batch_samples[:5]}...")
    
    # Extract batch data
    batch_df = df[batch_samples].copy()
    
    # Step 1: LOESS drift correction (BEFORE normalization)
    logger.info(f"  Applying LOESS drift correction...")
    logger.debug(f"  Injection order keys (first 5): {list(injection_order.keys())[:5] if injection_order else 'None'}...")
    batch_df = loess_drift_correction(
        batch_df,
        batch_samples,
        sample_info=sample_info,
        injection_order=injection_order,
        qc_pattern=qc_pattern,
        fallback_qc_pattern=fallback_qc_pattern,
        frac=frac,
    )
    
    # Step 2: Median normalization (AFTER drift correction)
    logger.info(f"  Applying median normalization...")
    batch_df = pqn_normalize_batch(
        batch_df,
        batch_samples,
        sample_info=sample_info,
        qc_pattern=qc_pattern,
        fallback_qc_pattern=fallback_qc_pattern,
    )
    
    # Create batch metadata
    batch_metadata = pd.DataFrame([
        {
            'sample_id': sample_info[col]['sample_id'] if sample_info and col in sample_info else extract_sample_id_from_column(col),
            'batch': batch,
            'sample_type': sample_info[col]['sample_type'] if sample_info and col in sample_info else extract_sample_type(extract_sample_id_from_column(col)),
            'original_col': col,
            'injection_order': sample_info[col]['injection_order'] if sample_info and col in sample_info else -1,
        }
        for col in batch_samples
    ])
    
    logger.info(f"  \u2713 Batch {batch} processed successfully")
    
    return batch_df, batch_metadata


def merge_batch_results(
    batch_results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge processed batch results into a single DataFrame.
    
    Args:
        batch_results: Dictionary mapping batch names to (data, metadata) tuples
        output_dir: Optional directory to save merged files
        
    Returns:
        Tuple of:
        - merged_data: Combined DataFrame with all features and samples
        - merged_metadata: Combined metadata DataFrame
    """
    logger.info("\nMerging batch results...")
    
    # Concatenate all batch data
    all_dfs = []
    all_metadata = []
    
    for batch, (batch_df, batch_meta) in batch_results.items():
        all_dfs.append(batch_df)
        all_metadata.append(batch_meta)
    
    merged_data = pd.concat(all_dfs, axis=1)
    merged_metadata = pd.concat(all_metadata, ignore_index=True)
    
    logger.info(f"  Merged data shape: {merged_data.shape}")
    logger.info(f"  Merged metadata shape: {merged_metadata.shape}")
    
    # Save if output_dir provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        merged_data.to_csv(output_dir / "merged_data.csv")
        merged_metadata.to_csv(output_dir / "merged_metadata.csv", index=False)
        logger.info(f"  Saved to {output_dir}")
    
    return merged_data, merged_metadata
