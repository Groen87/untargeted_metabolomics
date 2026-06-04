"""
Drift correction module for metabolomics data using LOESS smoothing.

This module provides functionality to correct for signal drift across injection order
using Locally Estimated Scatterplot Smoothing (LOESS).

Key Features:
- Identifies QC samples by pattern matching
- Selects high-quality features (high intensity in QC samples)
- Fits LOESS curves to QC sample intensities
- Applies drift correction to all samples
- Generates diagnostic plots

The drift correction is performed separately for each feature, using the QC samples
to estimate the drift pattern, then applying the inverse of this pattern to all samples.
"""

import os
import logging
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

logger = logging.getLogger(__name__)


def identify_qc_samples(
    sample_cols: List[str],
    qc_pattern: str = "expQC",
) -> Tuple[List[str], List[str]]:
    """
    Identify QC and non-QC (biological) samples from column names.
    
    Args:
        sample_cols: List of sample column names
        qc_pattern: Pattern to identify QC samples (default: "expQC")
        
    Returns:
        Tuple of (qc_samples, bio_samples) where each is a list of column names
    """
    qc_samples = [col for col in sample_cols if qc_pattern in col]
    bio_samples = [col for col in sample_cols if qc_pattern not in col]
    return qc_samples, bio_samples


def select_high_qc_features(
    intensity_df: pd.DataFrame,
    qc_samples: List[str],
    threshold_quantile: float = 0.9,
) -> pd.DataFrame:
    """
    Select features with high intensity in QC samples.
    
    These features are used to estimate the drift pattern, as they should have
    consistent signal across QC injections.
    
    Args:
        intensity_df: DataFrame with features as rows and samples as columns
        qc_samples: List of QC sample column names
        threshold_quantile: Quantile threshold for selecting high-QC features (default: 0.9)
        
    Returns:
        DataFrame with only high-QC features (subset of input)
    """
    if not qc_samples:
        return intensity_df
    
    qc_data = intensity_df[qc_samples]
    
    # Calculate mean intensity across QC samples for each feature
    qc_means = qc_data.mean(axis=1)
    
    # Select features above the threshold quantile
    threshold = qc_means.quantile(threshold_quantile)
    high_qc_mask = qc_means >= threshold
    
    return intensity_df.loc[high_qc_mask]


def fit_loess_curve(
    x: np.ndarray,
    y: np.ndarray,
    frac: float = 0.5,
) -> np.ndarray:
    """
    Fit a LOESS curve to the data.
    
    Args:
        x: Independent variable (e.g., injection order indices)
        y: Dependent variable (e.g., intensities)
        frac: Fraction of data to use for each LOESS fit (0.0-1.0)
        
    Returns:
        Array of smoothed y values from the LOESS fit
    """
    # LOESS requires x values to be sorted
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Fit LOESS curve
    smoothed = lowess(y_sorted, x_sorted, frac=frac)
    
    # Return smoothed values in original order
    result = np.zeros_like(y)
    for i, idx in enumerate(sorted_indices):
        result[idx] = smoothed[i, 1]
    
    return result


def correct_drift_with_loess(
    intensity_df: pd.DataFrame,
    qc_pattern: str = "expQC",
    frac: float = 0.5,
    qc_intensity_threshold: float = 0.1,
    output_dir: str = "output",
) -> pd.DataFrame:
    """
    Correct drift in metabolomics data using LOESS smoothing for high-QC features.
    
    This function performs the following steps:
    1. Identifies QC and biological samples
    2. Selects features with high intensity in QC samples
    3. For each feature, fits a LOESS curve to QC sample intensities across injection order
    4. Calculates correction factors as the inverse of the LOESS curve
    5. Applies correction to all samples (QC and biological)
    6. Generates diagnostic plots
    
    The drift correction assumes that QC samples should have consistent intensities
    across the run, and any deviation is due to instrument drift.
    
    Args:
        intensity_df: Input DataFrame with features as rows and samples as columns.
            Should have a 'Feature' column or all columns are samples.
        qc_pattern: Pattern to identify QC samples in column names (default: "expQC")
        frac: Fraction of data for LOESS smoothing (0.0-1.0, default: 0.5).
            Higher values = smoother curve, lower values = more local fitting
        qc_intensity_threshold: Quantile threshold for selecting high-QC features
            (default: 0.1, meaning top 10% of features by QC intensity)
        output_dir: Directory to save output files (default: "output")
        
    Returns:
        Drift-corrected DataFrame with all original features retained.
        The 'Feature' column is preserved if present in input.
        
    Raises:
        ValueError: If input DataFrame is invalid or QC samples are missing
    
    Note:
        The correction is applied to ALL features, not just the high-QC ones used
        for fitting. This assumes that the drift pattern estimated from high-QC
        features is representative of the overall drift.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Input Setup ---
    if 'Feature' in intensity_df.columns:
        feature_col = intensity_df['Feature']
        sample_cols = [col for col in intensity_df.columns if col != 'Feature']
        intensity_df = intensity_df.drop(columns=['Feature'])
    else:
        feature_col = None
        sample_cols = list(intensity_df.columns)
    
    if not sample_cols:
        raise ValueError("No sample columns found in DataFrame")
    
    # Convert to numeric
    for col in sample_cols:
        intensity_df[col] = pd.to_numeric(intensity_df[col], errors='coerce')
    
    # --- Identify QC and biological samples ---
    qc_samples, bio_samples = identify_qc_samples(sample_cols, qc_pattern)
    
    if not qc_samples:
        logger.warning(f"No QC samples found matching pattern '{qc_pattern}'. Skipping drift correction.")
        # Return original data with Feature column restored
        if feature_col is not None:
            intensity_df['Feature'] = feature_col
        return intensity_df
    
    logger.info(f"Found {len(qc_samples)} QC samples: {qc_samples}")
    logger.info(f"Found {len(bio_samples)} biological samples")
    
    # --- Select high-QC features for drift estimation ---
    # Use a higher threshold to get the most reliable features
    high_qc_df = select_high_qc_features(
        intensity_df, qc_samples, 
        threshold_quantile=1.0 - qc_intensity_threshold
    )
    
    logger.info(f"Using {len(high_qc_df)} high-QC features for drift estimation")
    
    # --- Create injection order mapping ---
    # Assume samples are in injection order (column order = injection order)
    injection_order = {col: i for i, col in enumerate(sample_cols)}
    
    # --- Fit LOESS and calculate correction factors ---
    # For each high-QC feature, fit LOESS to QC samples and calculate correction
    
    # Initialize correction factors (will be multiplied by original intensities)
    correction_factors = pd.DataFrame(
        index=intensity_df.index,
        columns=sample_cols,
        data=1.0  # Default: no correction
    )
    
    for feature_idx, feature_row in high_qc_df.iterrows():
        # Get intensities for this feature across all samples
        feature_intensities = intensity_df.loc[feature_idx, sample_cols].values
        
        # Get QC sample indices and their intensities
        qc_indices = [injection_order[col] for col in qc_samples]
        qc_intensities = [feature_intensities[injection_order[col]] for col in qc_samples]
        
        # Fit LOESS curve to QC intensities
        try:
            smoothed = fit_loess_curve(
                np.array(qc_indices),
                np.array(qc_intensities),
                frac=frac
            )
            
            # Calculate correction factor: target / smoothed
            # Target is the mean of QC intensities (what they should all be)
            qc_mean = np.mean(qc_intensities)
            
            # Avoid division by zero
            smoothed_safe = np.where(smoothed > 0, smoothed, qc_mean)
            
            # Correction factor for QC samples
            for i, col in enumerate(qc_samples):
                correction_factors.loc[feature_idx, col] = qc_mean / smoothed_safe[i]
            
            # For biological samples, interpolate correction factors
            # Use the nearest QC sample's correction factor
            for bio_col in bio_samples:
                bio_idx = injection_order[bio_col]
                
                # Find nearest QC sample
                nearest_qc_idx = min(qc_indices, key=lambda x: abs(x - bio_idx))
                nearest_qc_col = qc_samples[qc_indices.index(nearest_qc_idx)]
                
                correction_factors.loc[feature_idx, bio_col] = correction_factors.loc[feature_idx, nearest_qc_col]
                
        except Exception as e:
            logger.warning(f"Failed to fit LOESS for feature {feature_idx}: {e}")
            # Skip this feature, keep correction factor as 1.0
            continue
    
    # For features not in high_qc_df, use the median correction factor across all samples
    # This applies the overall drift pattern to all features
    if len(high_qc_df) < len(intensity_df):
        median_correction = correction_factors.median(axis=0)
        for feature_idx in intensity_df.index:
            if feature_idx not in high_qc_df.index:
                correction_factors.loc[feature_idx] = median_correction
    
    # --- Apply correction ---
    corrected_df = intensity_df.copy()
    for col in sample_cols:
        corrected_df[col] = intensity_df[col] * correction_factors[col]
    
    # --- Restore Feature column if it existed ---
    if feature_col is not None:
        corrected_df['Feature'] = feature_col
    
    # --- Save diagnostic plot for first feature ---
    if len(high_qc_df) > 0 and len(qc_samples) >= 3:
        try:
            first_feature = high_qc_df.index[0]
            first_feature_data = intensity_df.loc[first_feature, sample_cols].values
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(sample_cols, first_feature_data, 'o-', label='Original', alpha=0.5)
            
            # Plot LOESS curve
            qc_indices = [injection_order[col] for col in qc_samples]
            qc_intensities = [first_feature_data[injection_order[col]] for col in qc_samples]
            smoothed = fit_loess_curve(np.array(qc_indices), np.array(qc_intensities), frac=frac)
            
            # Create x values for LOESS curve (use QC sample positions)
            loess_x = [injection_order[col] for col in qc_samples]
            ax.plot(loess_x, smoothed, 'r-', label='LOESS fit', linewidth=2)
            
            ax.set_title(f"LOESS Drift Correction: {first_feature}")
            ax.set_xlabel('Injection Order')
            ax.set_ylabel('Intensity')
            ax.legend()
            ax.grid(True)
            
            plot_path = os.path.join(output_dir, "loess_drift_correction_example.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved LOESS diagnostic plot to {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save diagnostic plot: {e}")
    
    return corrected_df
