"""
Batch processing module for combined batch pipeline.

This module handles:
1. Processing individual batches (median normalization, LOESS drift correction)
2. Merging processed batches
3. Preparing data for ComBat correction

Uses injection order from metadata for LOESS drift correction.
"""

from typing import Tuple, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from .data_loader import extract_batch_from_filename, extract_sample_id_from_column, extract_sample_type

logger = logging.getLogger(__name__)

def filter_features_by_qc_quality(
    df: pd.DataFrame,
    sample_cols: List[str],
    qc_pattern: str = "expQC",
    fallback_qc_pattern: Optional[str] = "QC3",
    rsd_threshold: float = 20.0,
    intensity_quantile: float = 0.25,
) -> pd.DataFrame:
    """
    Filter features based on QC sample quality:
    1. Present in all QC samples
    2. RSD <= threshold in QC samples (SKIPPED - removes too many features)
    3. Mean intensity >= quantile in QC samples
    
    Args:
        df: DataFrame with features as rows, samples as columns
        sample_cols: List of sample column names
        qc_pattern: Pattern to identify QC samples
        fallback_qc_pattern: Fallback QC pattern
        rsd_threshold: Maximum RSD percentage in QC samples (default: 20%)
        intensity_quantile: Minimum quantile for mean QC intensity (default: 0.25)
        
    Returns:
        Filtered DataFrame with only high-quality features
    """
    # Identify QC samples
    qc_samples, _ = identify_qc_samples(sample_cols, None, qc_pattern, fallback_qc_pattern)
    
    if not qc_samples:
        logger.warning("No QC samples found. Skipping QC-based feature filtering.")
        return df
    
    logger.info(f"Filtering features by QC quality using {len(qc_samples)} QC samples")
    
    # Get QC subset
    qc_df = df[qc_samples]
    
    # Filter 1: Present in all QC samples (not NaN)
    initial_features = len(df)
    mask_all_qc = qc_df.notna().all(axis=1)
    df = df[mask_all_qc]
    filtered_1 = initial_features - len(df)
    logger.info(f"  Filter 1 (present in all QC): removed {filtered_1} features")
    
    if len(df) == 0:
        logger.warning("No features remain after Filter 1. Returning empty DataFrame.")
        return df
    
    # Filter 2: RSD <= threshold in QC samples - SKIPPED as per user request
    # RSD filtering was removing too many features due to intensity differences between batches
    filtered_2 = 0
    logger.info(f"  Filter 2 (RSD <= {rsd_threshold}%): removed {filtered_2} features (SKIPPED)")
    
    # Filter 3: Mean QC intensity >= quantile
    qc_df = df[qc_samples]
    qc_mean = qc_df.mean(axis=1)
    intensity_threshold = qc_mean.quantile(intensity_quantile)
    mask_high_intensity = qc_mean >= intensity_threshold
    df = df[mask_high_intensity]
    filtered_3 = len(qc_df) - len(df)
    logger.info(f"  Filter 3 (QC intensity >= {intensity_quantile} quantile): removed {filtered_3} features")
    
    total_filtered = initial_features - len(df)
    logger.info(f"Total: filtered out {total_filtered}/{initial_features} features ({100*total_filtered/initial_features:.1f}%)")
    
    # Ensure unique index (combine duplicates by taking mean)
    if df.index.has_duplicates:
        logger.info(f"  Combining {df.index.duplicated().sum()} duplicate feature names by taking mean")
        df = df.groupby(level=0).mean()
    
    return df


def identify_qc_samples(
    sample_cols: List[str],
    sample_info: Optional[Union[pd.DataFrame, Dict[str, Dict]]] = None,
    qc_pattern: str = "expQC",
    fallback_qc_pattern: Optional[str] = "QC3",
) -> Tuple[List[str], List[str]]:
    """
    Identify QC samples from column names.
    
    Args:
        sample_cols: List of sample column names
        sample_info: Optional DataFrame or dict with sample metadata
        qc_pattern: Pattern to identify QC samples
        fallback_qc_pattern: Fallback QC pattern
        
    Returns:
        Tuple of (qc_samples, non_qc_samples)
    """
    qc_samples = [col for col in sample_cols if qc_pattern in col]
    
    # If no QC samples found with primary pattern, try fallback
    if not qc_samples and fallback_qc_pattern:
        qc_samples = [col for col in sample_cols if fallback_qc_pattern in col]
    
    # If we have sample_info, also check for QC samples there
    if sample_info is not None:
        if isinstance(sample_info, pd.DataFrame) and 'sample_type' in sample_info.columns:
            qc_from_info = sample_info[sample_info['sample_type'].str.contains('QC', case=False, na=False)]
            if not qc_from_info.empty:
                qc_sample_ids = qc_from_info['sample_id'].tolist()
                qc_samples = list(set(qc_samples) | set(qc_sample_ids))
        elif isinstance(sample_info, dict):
            # sample_info is a dict of {sample_name: {metadata}}
            for sample_name, metadata in sample_info.items():
                if sample_name in sample_cols:
                    sample_type = metadata.get('sample_type', '')
                    if 'QC' in str(sample_type).upper():
                        qc_samples.append(sample_name)
    
    non_qc_samples = [col for col in sample_cols if col not in qc_samples]
    
    return qc_samples, non_qc_samples


def median_normalize_batch(
    df: pd.DataFrame,
    batch_samples: List[str],
    sample_info: Optional[pd.DataFrame] = None,
    qc_pattern: str = "expQC",
    fallback_qc_pattern: Optional[str] = "QC3",
) -> pd.DataFrame:
    """
    Apply median normalization to a batch using QC samples.
    
    Args:
        df: DataFrame with features as rows, samples as columns
        batch_samples: List of sample column names in this batch
        sample_info: Optional DataFrame with sample metadata
        qc_pattern: Pattern to identify QC samples
        fallback_qc_pattern: Fallback QC pattern
        
    Returns:
        Normalized DataFrame
    """
    qc_samples, _ = identify_qc_samples(batch_samples, sample_info, qc_pattern, fallback_qc_pattern)
    
    if not qc_samples:
        logger.warning(f"No QC samples found in batch. Cannot perform median normalization.")
        return df
    
    # Calculate median of QC samples
    qc_df = df[qc_samples]
    qc_medians = qc_df.median(axis=1)
    reference_median = qc_medians.median()
    
    logger.info(f"Using {len(qc_samples)} QC samples for median normalization: {qc_samples[:3]}...")
    logger.info(f"Reference median: {reference_median:.2f}")
    
    # Normalize each feature by dividing by its QC median and multiplying by reference
    # This scales all features to have the same median as the reference
    normalized_df = df.copy()
    for feature in df.index:
        feature_qc_median = qc_medians[feature]
        if feature_qc_median > 0:
            normalized_df.loc[feature] = df.loc[feature] / feature_qc_median * reference_median
    
    return normalized_df


def loess_drift_correction(
    df: pd.DataFrame,
    batch_samples: List[str],
    injection_order: Dict[str, int],
    sample_info: Optional[pd.DataFrame] = None,
    qc_pattern: str = "expQC",
    fallback_qc_pattern: Optional[str] = "QC3",
    span: float = 0.3,
    degree: int = 2,
) -> pd.DataFrame:
    """
    Apply LOESS drift correction to a batch.
    
    Args:
        df: DataFrame with features as rows, samples as columns
        batch_samples: List of sample column names in this batch
        injection_order: Dictionary mapping sample names to injection order
        sample_info: Optional DataFrame with sample metadata
        qc_pattern: Pattern to identify QC samples
        fallback_qc_pattern: Fallback QC pattern
        span: LOESS span parameter (fraction of data to use)
        degree: LOESS degree parameter
        
    Returns:
        Drift-corrected DataFrame
    """
    from scipy.interpolate import make_interp_spline
    from sklearn.linear_model import LinearRegression
    
    qc_samples, _ = identify_qc_samples(batch_samples, sample_info, qc_pattern, fallback_qc_pattern)
    
    if not qc_samples:
        logger.warning(f"No QC samples found in batch. Skipping LOESS drift correction.")
        return df
    
    # Sort QC samples by injection order
    qc_injection_order = {s: injection_order.get(s, 0) for s in qc_samples}
    sorted_qc_samples = sorted(qc_samples, key=lambda x: qc_injection_order.get(x, 0))
    
    logger.info(f"Applying LOESS drift correction with {len(sorted_qc_samples)} QC samples")
    
    # Calculate median intensity for each QC sample
    qc_medians = df[sorted_qc_samples].median(axis=0)
    qc_injection_points = np.array([qc_injection_order[s] for s in sorted_qc_samples])
    
    # Use high-intensity features for drift estimation (more reliable)
    feature_medians = df.median(axis=1)
    high_intensity_threshold = feature_medians.quantile(0.75)
    high_intensity_features = feature_medians[feature_medians >= high_intensity_threshold].index
    
    if len(high_intensity_features) < 10:
        # Fall back to all features if not enough high-intensity ones
        high_intensity_features = df.index
    
    logger.debug(f"Using {len(high_intensity_features)} high-intensity features for drift estimation")
    
    # Calculate median for each QC sample using only high-intensity features
    qc_medians_high = df.loc[high_intensity_features, sorted_qc_samples].median(axis=0)
    
    # Fit LOESS curve to QC medians
    try:
        # Use numpy's polynomial fit for LOESS-like behavior
        # For simplicity, we'll use a simple linear regression on the log scale
        # This is a simplified version of LOESS
        
        x = qc_injection_points
        y = qc_medians_high.values
        
        # Fit linear regression to the drift
        if len(x) >= 2:
            x_normalized = (x - x.min()) / (x.max() - x.min())
            X = x_normalized.reshape(-1, 1)
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate correction factors
            y_pred = model.predict(X)
            correction_factors = y / y_pred
            
            # Apply correction to all samples in the batch
            # Interpolate correction factors for all samples based on injection order
            all_samples_ordered = sorted(batch_samples, key=lambda x: injection_order.get(x, 0))
            all_injection_points = np.array([injection_order.get(s, 0) for s in all_samples_ordered])
            all_x_normalized = (all_injection_points - all_injection_points.min()) / (all_injection_points.max() - all_injection_points.min())
            
            # Predict correction factor for each sample
            all_y_pred = model.predict(all_x_normalized.reshape(-1, 1))
            
            # Create mapping from sample to correction factor
            sample_to_correction = {}
            for i, sample in enumerate(all_samples_ordered):
                sample_to_correction[sample] = 1.0 / all_y_pred[i] * y.mean()
            
            # Apply correction
            corrected_df = df.copy()
            for sample in batch_samples:
                if sample in sample_to_correction:
                    corrected_df[sample] = df[sample] * sample_to_correction[sample]
            
            return corrected_df
        else:
            logger.warning("Not enough QC samples for LOESS correction.")
            return df
            
    except Exception as e:
        logger.warning(f"LOESS drift correction failed: {e}")
        return df


def process_batch(
    df: pd.DataFrame,
    batch: str,
    batch_samples: List[str],
    injection_order: Dict[str, int],
    sample_info: Optional[pd.DataFrame] = None,
    qc_pattern: str = "expQC",
    fallback_qc_pattern: Optional[str] = "QC3",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a single batch: filter, normalize, and correct drift.
    
    Args:
        df: Full DataFrame with all samples
        batch: Batch name
        batch_samples: List of sample column names in this batch
        injection_order: Dictionary mapping sample names to injection order
        sample_info: Optional DataFrame with sample metadata
        qc_pattern: Pattern to identify QC samples
        fallback_qc_pattern: Fallback QC pattern
        
    Returns:
        Tuple of (processed_batch_df, metadata_df)
    """
    logger.info(f"\nProcessing batch: {batch}")
    logger.info(f"  Samples: {len(batch_samples)}")
    logger.debug(f"  Batch samples: {batch_samples[:5]}...")
    
    # Extract batch data
    batch_df = df[batch_samples].copy()
    
    # Step 0: Filter features by QC quality (before normalization)
    logger.info(f"  Applying QC-based feature filtering...")
    batch_df = filter_features_by_qc_quality(
        batch_df,
        batch_samples,
        qc_pattern=qc_pattern,
        fallback_qc_pattern=fallback_qc_pattern,
    )
    
    # Step 1: Median normalization
    logger.info(f"  Applying median normalization...")
    batch_df = median_normalize_batch(
        batch_df,
        batch_samples,
        sample_info=sample_info,
        qc_pattern=qc_pattern,
        fallback_qc_pattern=fallback_qc_pattern,
    )
    
    # Step 2: LOESS drift correction
    logger.info(f"  Applying LOESS drift correction...")
    logger.debug(f"  Injection order keys (first 5): {list(injection_order.keys())[:5] if injection_order else 'None'}...")
    batch_df = loess_drift_correction(
        batch_df,
        batch_samples,
        injection_order,
        sample_info=sample_info,
        qc_pattern=qc_pattern,
        fallback_qc_pattern=fallback_qc_pattern,
    )
    
    # Create metadata for this batch
    metadata = pd.DataFrame({
        'batch': batch,
        'sample_id': batch_samples,
        'original_col': batch_samples,
    })
    
    if sample_info is not None:
        for col in ['sample_type', 'injection_order']:
            if col in sample_info.columns:
                metadata[col] = metadata['sample_id'].map(
                    dict(zip(sample_info['sample_id'], sample_info[col]))
                )
    
    logger.info(f"  [32m[1m[32m✓[0m Batch {batch} processed successfully")
    
    return batch_df, metadata


def merge_batch_results(
    batch_results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    all_batches: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge processed batches into a single DataFrame.
    
    Args:
        batch_results: Dictionary mapping batch names to (data, metadata) tuples
        all_batches: List of all batch names
        
    Returns:
        Tuple of (merged_data, merged_metadata)
    """
    # Collect all data and metadata
    all_data = []
    all_metadata = []
    
    for batch in all_batches:
        if batch in batch_results:
            batch_data, batch_metadata = batch_results[batch]
            all_data.append(batch_data)
            all_metadata.append(batch_metadata)
    
    if not all_data:
        raise ValueError("No batch data to merge")
    
    # Concatenate data
    merged_data = pd.concat(all_data, axis=1)
    
    # Handle duplicate column names (shouldn't happen if batch_samples are unique)
    if merged_data.columns.has_duplicates:
        logger.warning(f"Found {merged_data.columns.duplicated().sum()} duplicate column names. Making unique.")
        merged_data.columns = [f"{col}_{i}" if merged_data.columns.tolist().count(col) > 1 
                               else col 
                               for i, col in enumerate(merged_data.columns)]
    
    # Concatenate metadata
    merged_metadata = pd.concat(all_metadata, ignore_index=True)
    
    # Ensure metadata has batch column
    if 'batch' not in merged_metadata.columns:
        merged_metadata['batch'] = merged_metadata['original_col'].apply(
            lambda x: extract_batch_from_filename(x)
        )
    
    return merged_data, merged_metadata
