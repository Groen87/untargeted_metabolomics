"""
Clean LOESS drift correction based on the working metabolomics_pipeline implementation.

This module provides a properly validated LOESS drift correction that:
1. Removes outliers from QC samples before fitting
2. Only corrects features with sufficient QC signal
3. Uses proper interpolation for correction factors
4. Clips correction factors to prevent extreme values
5. Reports RSD% (not CV ratio) for familiar interpretation
"""

import logging
from typing import Tuple, List, Optional, Dict
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess

logger = logging.getLogger(__name__)


def identify_qc_samples(
    sample_cols: List[str],
    sample_info: Optional[Dict[str, Dict]] = None,
    qc_pattern: str = "expQC",
    fallback_pattern: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Identify QC and biological samples from column names."""
    qc_samples = []
    bio_samples = []
    
    for col in sample_cols:
        if sample_info and col in sample_info:
            sample_type = sample_info[col].get('sample_type', 'Sample')
            if sample_type == "QC":
                qc_samples.append(col)
                continue
        
        if qc_pattern.lower() in col.lower():
            qc_samples.append(col)
            continue
        
        if fallback_pattern and fallback_pattern.lower() in col.lower():
            qc_samples.append(col)
            continue
        
        bio_samples.append(col)
    
    return qc_samples, bio_samples


def loess_drift_correction(
    df: pd.DataFrame,
    sample_cols: List[str],
    sample_info: Optional[Dict[str, Dict]] = None,
    injection_order: Optional[Dict[str, int]] = None,
    qc_pattern: str = "expQC",
    fallback_qc_pattern: Optional[str] = "QC3",
    frac: float = 0.5,
    qc_intensity_threshold: float = 0.1,
    min_qc_present: float = 0.5,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Apply LOESS drift correction to a batch.
    
    Based on the working implementation from metabolomics_pipeline.
    
    Args:
        df: DataFrame with features as rows, samples as columns
        sample_cols: List of sample column names
        sample_info: Optional dictionary with sample metadata
        injection_order: Optional dictionary mapping columns to injection order index
        qc_pattern: Pattern to identify QC samples (default: "expQC")
        fallback_qc_pattern: Fallback QC pattern (default: "QC3")
        frac: LOESS fraction parameter (default: 0.5)
        qc_intensity_threshold: Minimum intensity threshold for QC features (default: 0.1)
        min_qc_present: Minimum fraction of QC samples where feature must be present (default: 0.5)
        output_dir: Optional directory to save plots
        
    Returns:
        DataFrame with drift-corrected values
    """
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
    
    # Filter to only samples that are in sorted_samples
    qc_samples = [col for col in qc_samples if col in sorted_samples]
    bio_samples = [col for col in bio_samples if col in sorted_samples]
    
    if len(qc_samples) < 2:
        logger.warning(f"After filtering, only {len(qc_samples)} QC samples remain. Skipping.")
        return df.copy()
    
    # Create position mapping
    position_idx = {col: i for i, col in enumerate(sorted_samples)}
    qc_positions = np.array([position_idx[col] for col in qc_samples])
    
    # Get QC data
    qc_df = df.loc[:, qc_samples]
    
    # --- Pre-process: Replace outliers in QC samples ---
    # This is CRITICAL - prevents outliers from skewing LOESS fit
    for feature in df.index:
        qc_values = qc_df.loc[feature].to_numpy()
        Q1 = np.percentile(qc_values, 25)
        Q3 = np.percentile(qc_values, 75)
        IQR = Q3 - Q1
        outlier_mask = (qc_values < Q1 - 3.0 * IQR) | (qc_values > Q3 + 3.0 * IQR)
        if np.sum(~outlier_mask) > 0 and np.sum(outlier_mask) > 0:
            median_non_outlier = np.median(qc_values[~outlier_mask])
            df.loc[feature, qc_samples] = np.where(outlier_mask, median_non_outlier, qc_values)
            qc_df.loc[feature] = df.loc[feature, qc_samples].values
    
    # --- Separate features by QC signal ---
    # Only correct features that are present in QC samples
    qc_present_mask = (qc_df > 0).any(axis=1)
    qc_intensity_mask = (qc_df.mean(axis=1) > qc_intensity_threshold)
    high_qc_features = qc_present_mask & qc_intensity_mask
    high_qc_features = high_qc_features[high_qc_features].index
    
    logger.info(f"  Features with sufficient QC signal: {len(high_qc_features)}/{len(df.index)}")
    
    # Initialize corrected DataFrame
    df_corrected = df.copy()
    features_corrected = 0
    
    if len(high_qc_features) == 0:
        logger.warning("No features with sufficient QC signal for LOESS correction.")
        return df_corrected
    
    # --- LOESS Correction for HIGH-QC FEATURES ONLY ---
    # Get all sample positions
    all_positions = np.arange(len(sorted_samples))
    
    for feature in high_qc_features:
        qc_values = qc_df.loc[feature].to_numpy().flatten()
        
        try:
            # Fit LOESS to QC samples
            smoothed = lowess(qc_values, qc_positions, frac=frac, it=3)
            median_qc = np.median(qc_values)
            
            # Interpolate correction factors at ALL sample positions
            correction_factors = np.interp(
                all_positions,
                smoothed[:, 0],  # x-coordinates from LOESS (QC positions)
                median_qc / smoothed[:, 1]  # y-coordinates (correction = median / smoothed_value)
            )
            
            # Clip correction factors to prevent extreme values
            correction_factors = np.clip(correction_factors, 0.5, 2.0)
            
            # Apply correction to ALL samples in this batch
            row_values = df.loc[feature, sorted_samples].to_numpy()
            df_corrected.loc[feature, sorted_samples] = row_values * correction_factors
            
            features_corrected += 1
            
        except Exception as e:
            logger.debug(f"Failed LOESS for {feature}: {e}")
            continue
    
    # --- Calculate RSD improvement on QC samples ---
    if features_corrected > 0:
        # Calculate RSD before and after for high-QC features
        qc_rsd_before = (
            df.loc[high_qc_features, qc_samples].std(axis=1) /
            df.loc[high_qc_features, qc_samples].mean(axis=1) * 100
        )
        qc_rsd_after = (
            df_corrected.loc[high_qc_features, qc_samples].std(axis=1) /
            df_corrected.loc[high_qc_features, qc_samples].mean(axis=1) * 100
        )
        
        mean_rsd_before = qc_rsd_before.mean()
        mean_rsd_after = qc_rsd_after.mean()
        rsd_improvement = mean_rsd_before - mean_rsd_after
        features_improved = (qc_rsd_after < qc_rsd_before).sum()
        
        logger.info(f"  Features corrected with LOESS: {features_corrected}/{len(high_qc_features)}")
        logger.info(f"  Mean QC RSD% before: {mean_rsd_before:.2f}%")
        logger.info(f"  Mean QC RSD% after: {mean_rsd_after:.2f}%")
        logger.info(f"  Mean QC RSD% improvement: {rsd_improvement:.2f}%")
        logger.info(f"  Features with improved RSD: {features_improved}/{features_corrected}")
        
        # Log worst features
        worst_features = qc_rsd_after.nlargest(5)
        if len(worst_features) > 0:
            logger.debug(f"  Features with highest RSD after LOESS: {worst_features.to_dict()}")
    else:
        logger.info("  No features corrected with LOESS")
    
    return df_corrected
