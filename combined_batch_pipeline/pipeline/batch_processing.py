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

from .data_loader import extract_batch_from_filename, extract_sample_id_from_filename, extract_sample_type

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
        sample_id = extract_sample_id_from_filename(col)
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


def median_normalize_batch(
    df: pd.DataFrame,
    sample_cols: List[str],
    sample_info: Optional[Dict[str, Dict]] = None,
    qc_pattern: str = "expQC",
    fallback_qc_pattern: Optional[str] = "QC3",
) -> pd.DataFrame:
    """
    Apply median normalization to a batch.
    
    Uses QC samples as reference for normalization.
    Formula: normalized_value = raw_value * (reference_median / sample_median)
    
    Args:
        df: DataFrame with features as rows, samples as columns
        sample_cols: List of sample column names to normalize
        sample_info: Optional dictionary with sample metadata
        qc_pattern: Pattern to identify QC samples
        fallback_qc_pattern: Fallback QC pattern
        
    Returns:
        DataFrame with median-normalized values
    """
    # Identify QC samples
    qc_samples, _ = identify_qc_samples(sample_cols, sample_info, qc_pattern, fallback_qc_pattern)
    
    if not qc_samples:
        logger.warning(f"No QC samples found. Skipping median normalization.")
        return df.copy()
    
    logger.info(f"Using {len(qc_samples)} QC samples for median normalization: {qc_samples[:3]}...")
    
    # Calculate median for each sample
    sample_medians = df[sample_cols].median(axis=0)
    
    # Calculate reference median (median of QC sample medians)
    qc_medians = df[qc_samples].median(axis=0)
    reference_median = qc_medians.median()
    
    logger.info(f"Reference median: {reference_median:.2f}")
    
    # Apply normalization
    df_normalized = df.copy()
    for col in sample_cols:
        df_normalized[col] = df[col] * (reference_median / sample_medians[col])
    
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
    
    Uses injection order from metadata (creation dates) to properly order samples.
    If no injection order is provided, uses column order as fallback.
    
    Args:
        df: DataFrame with features as rows, samples as columns
        sample_cols: List of sample column names (will be reordered by injection order)
        sample_info: Optional dictionary with sample metadata
        injection_order: Optional dictionary mapping columns to injection order index
        qc_pattern: Pattern to identify QC samples
        fallback_qc_pattern: Fallback QC pattern
        frac: LOESS fraction parameter
        
    Returns:
        DataFrame with drift-corrected values
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    # Identify QC samples
    qc_samples, bio_samples = identify_qc_samples(sample_cols, sample_info, qc_pattern, fallback_qc_pattern)
    
    if len(qc_samples) < 2:
        logger.warning(f"Need at least 2 QC samples for LOESS. Found {len(qc_samples)}. Skipping drift correction.")
        return df.copy()
    
    logger.info(f"Applying LOESS drift correction with {len(qc_samples)} QC samples")
    
    # Sort samples by injection order if available, otherwise use column order
    if injection_order:
        # Sort all samples by injection order
        sorted_samples = sorted(sample_cols, key=lambda col: injection_order.get(col, float('inf')))
    else:
        sorted_samples = sample_cols
        logger.warning("No injection order provided, using column order")
    
    # Create injection order mapping based on sorted order
    injection_idx = {col: i for i, col in enumerate(sorted_samples)}
    
    # For each feature, fit LOESS to QC samples and apply correction
    df_corrected = df.copy()
    
    # Use high-intensity features for drift estimation
    feature_means = df[sorted_samples].mean(axis=1)
    high_intensity_mask = feature_means > feature_means.quantile(0.9)
    high_intensity_features = df.index[high_intensity_mask]
    
    logger.info(f"Using {len(high_intensity_features)} high-intensity features for drift estimation")
    
    # Fit LOESS for each high-intensity feature
    for feature in high_intensity_features:
        feature_data = df.loc[feature, sorted_samples].values
        
        # Get QC sample indices and intensities
        qc_indices = [injection_idx[col] for col in qc_samples]
        qc_intensities = [feature_data[injection_idx[col]] for col in qc_samples]
        
        # Fit LOESS
        try:
            smoothed = lowess(qc_intensities, qc_indices, frac=frac)
            
            # Calculate correction factors
            qc_mean = np.mean(qc_intensities)
            smoothed_values = smoothed[:, 1]
            
            # Avoid division by zero
            smoothed_safe = np.where(smoothed_values > 0, smoothed_values, qc_mean)
            
            # Correction factor: target / smoothed
            for i, col in enumerate(qc_samples):
                correction = qc_mean / smoothed_safe[i]
                df_corrected.loc[feature, col] *= correction
            
            # For biological samples, interpolate from nearest QC
            for bio_col in bio_samples:
                bio_idx = injection_idx[bio_col]
                nearest_qc_idx = min(qc_indices, key=lambda x: abs(x - bio_idx))
                nearest_qc_col = qc_samples[qc_indices.index(nearest_qc_idx)]
                df_corrected.loc[feature, bio_col] = df_corrected.loc[feature, nearest_qc_col]
                
        except Exception as e:
            logger.debug(f"Failed to fit LOESS for feature {feature}: {e}")
            continue
    
    # For features not in high_intensity_features, use median correction
    if len(high_intensity_features) < len(df):
        median_correction = df_corrected[sorted_samples].median(axis=0) / df[sorted_samples].median(axis=0)
        for feature in df.index:
            if feature not in high_intensity_features:
                df_corrected.loc[feature] = df.loc[feature] * median_correction
    
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
    Process a single batch: median normalization + LOESS drift correction.
    
    Args:
        df: Full DataFrame with all features and samples
        batch: Batch name
        batch_samples: List of sample column names belonging to this batch
        sample_info: Optional dictionary with sample metadata
        injection_order: Optional dictionary mapping columns to injection order
        qc_pattern: Pattern to identify QC samples
        fallback_qc_pattern: Fallback QC pattern
        frac: LOESS fraction parameter
        output_dir: Optional directory to save intermediate files
        
    Returns:
        Tuple of:
        - df_processed: DataFrame with processed batch data
        - batch_metadata: DataFrame with sample metadata
    """
    logger.info(f"\nProcessing batch: {batch}")
    logger.info(f"  Samples: {len(batch_samples)}")
    
    # Extract batch data
    batch_df = df[batch_samples].copy()
    
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
    batch_df = loess_drift_correction(
        batch_df,
        batch_samples,
        sample_info=sample_info,
        injection_order=injection_order,
        qc_pattern=qc_pattern,
        fallback_qc_pattern=fallback_qc_pattern,
        frac=frac,
    )
    
    # Create batch metadata
    batch_metadata = pd.DataFrame([
        {
            'sample_id': sample_info[col]['sample_id'] if sample_info and col in sample_info else extract_sample_id_from_filename(col),
            'batch': batch,
            'sample_type': sample_info[col]['sample_type'] if sample_info and col in sample_info else extract_sample_type(extract_sample_id_from_filename(col)),
            'original_col': col,
            'injection_order': sample_info[col]['injection_order'] if sample_info and col in sample_info else -1,
        }
        for col in batch_samples
    ])
    
    logger.info(f"  ✓ Batch {batch} processed successfully")
    
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
