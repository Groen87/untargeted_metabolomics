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
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Apply LOESS drift correction to a batch.
    
    Uses the clean implementation from loess_drift_correction_clean.py
    
    Args:
        df: DataFrame with features as rows, samples as columns
        sample_cols: List of sample column names
        sample_info: Optional dictionary with sample metadata
        injection_order: Optional dictionary mapping columns to injection order index
        qc_pattern: Pattern to identify QC samples
        fallback_qc_pattern: Fallback QC pattern
        frac: LOESS fraction parameter
        output_dir: Optional directory to save plots
        
    Returns:
        DataFrame with drift-corrected values
    """
    from .loess_drift_correction_clean import loess_drift_correction as clean_loess
    return clean_loess(
        df=df,
        sample_cols=sample_cols,
        sample_info=sample_info,
        injection_order=injection_order,
        qc_pattern=qc_pattern,
        fallback_qc_pattern=fallback_qc_pattern,
        frac=frac,
        qc_intensity_threshold=0.1,
        output_dir=output_dir,
    )


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
    
    # Remove experiment QC samples (expQC, QC3) after drift correction
    # These are used for within-batch LOESS correction and should NOT be used for inter-batch correction
    # Identify which samples are experiment QC
    exp_qc_samples = []
    for col in batch_samples:
        if col in batch_df.columns:
            if qc_pattern in col or (fallback_qc_pattern and fallback_qc_pattern in col):
                exp_qc_samples.append(col)
    
    if exp_qc_samples:
        logger.info(f"  Removing {len(exp_qc_samples)} experiment QC samples (used for LOESS)")
        logger.debug(f"  Removed: {exp_qc_samples[:5]}...")
        batch_df = batch_df.drop(columns=exp_qc_samples)
        # Also remove from batch_samples for subsequent processing
        batch_samples = [s for s in batch_samples if s not in exp_qc_samples]
    
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
