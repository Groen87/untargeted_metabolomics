"""Correct drift in metabolomics data using LOESS smoothing."""

import os
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def correct_drift_with_loess(
    batch: str,
    mode: str,
    intensity_df: pd.DataFrame,
    qc_pattern: str = "expQC",
    frac: float = 0.5,
    qc_intensity_threshold: float = 0.1,
    output_dir: str = "output",
) -> pd.DataFrame:
    """
    Correct drift in metabolomics data using LOESS for high-QC features.

    Args:
        intensity_df: Input DataFrame with features as rows and samples as columns.
            Can include metadata columns like 'Feature' and 'RT [min]'.
        qc_pattern: Pattern to identify QC samples in column names
        frac: Fraction of data for LOESS smoothing (0.0-1.0)
        qc_intensity_threshold: Threshold to classify high-QC features
        output_dir: Directory to save output files

    Returns:
        Drift-corrected DataFrame with all original features and metadata columns
        (including RT [min]) retained.

    Raises:
        ValueError: If input DataFrame is invalid or QC samples are missing
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Input Setup ---
    # Identify metadata columns (Feature, RT [min], etc.) vs sample columns
    metadata_cols = []
    if 'Feature' in intensity_df.columns:
        metadata_cols.append('Feature')
    if 'RT [min]' in intensity_df.columns:
        metadata_cols.append('RT [min]')
    
    sample_cols = [col for col in intensity_df.columns if col not in metadata_cols]
    
    # Store metadata
    metadata_df = intensity_df[metadata_cols].copy()
    
    # Work with sample data only
    intensity_df = intensity_df[sample_cols].copy()
    
    feature_col = metadata_df['Feature'] if 'Feature' in metadata_cols else None

    if not sample_cols:
        raise ValueError("No sample columns found in DataFrame")

    # Convert to numeric
    for col in sample_cols:
        intensity_df[col] = pd.to_numeric(intensity_df[col], errors='coerce')

    # Set Feature column as index for processing
    if feature_col is not None:
        intensity_df = intensity_df.set_index(feature_col)

    qc_samples = [
        col for col in intensity_df.columns
        if qc_pattern.lower() in col.lower() and
           not any(col.startswith(prefix) for prefix in ['Area:', 'PQF:', 'Gap', 'Peak', 'Number', 'Status'])
    ]

    if not qc_samples:
        logger.error(
            f"No QC samples found with pattern '{qc_pattern}'. Available columns: {list(intensity_df.columns)}")
        raise ValueError(f"No QC samples found with pattern '{qc_pattern}'")
    else:
        logger.info(f"Found QC samples: {qc_samples}")

    # Initialize corrected DataFrame (with same index as intensity_df)
    corrected_df = intensity_df.copy()

    # --- Pre-process: Replace outliers in QC samples ---
    for feature in intensity_df.index:
        qc_values = intensity_df.loc[feature, qc_samples].to_numpy()
        Q1 = np.percentile(qc_values, 25)
        Q3 = np.percentile(qc_values, 75)
        IQR = Q3 - Q1
        outlier_mask = (qc_values < Q1 - 3.0 * IQR) | (qc_values > Q3 + 3.0 * IQR)
        if np.sum(~outlier_mask) > 0:
            median_non_outlier = np.median(qc_values[~outlier_mask])
            qc_values[outlier_mask] = median_non_outlier
            intensity_df.loc[feature, qc_samples] = qc_values

    # --- Separate features by QC signal ---
    qc_intensity_mask = (intensity_df[qc_samples] > qc_intensity_threshold).any(axis=1)
    high_qc_features = qc_intensity_mask[qc_intensity_mask].index
    low_qc_features = qc_intensity_mask[~qc_intensity_mask].index

    # --- LOESS Correction for HIGH-QC FEATURES ONLY ---
    injection_order = np.arange(len(sample_cols))
    qc_positions = np.array([i for i, col in enumerate(sample_cols) if col in qc_samples])

    for feature in high_qc_features:
        qc_values = intensity_df.loc[feature, qc_samples].to_numpy().flatten()
        smoothed = lowess(qc_values, qc_positions, frac=frac, it=3)
        median_qc = np.median(qc_values)
        correction_factors = np.interp(
            injection_order,
            smoothed[:, 0],
            median_qc / smoothed[:, 1]
        )
        correction_factors = np.clip(correction_factors, 0.5, 2.0)
        row_values = intensity_df.loc[feature, sample_cols].to_numpy()
        corrected_df.loc[feature, sample_cols] = row_values * correction_factors

    # --- Quality Checks (high-QC features only) ---
    if len(high_qc_features) > 0:
        qc_rsd_before = (
            intensity_df.loc[high_qc_features, qc_samples].std(axis=1) /
            intensity_df.loc[high_qc_features, qc_samples].mean(axis=1) * 100
        )
        qc_rsd_after = (
            corrected_df.loc[high_qc_features, qc_samples].std(axis=1) /
            corrected_df.loc[high_qc_features, qc_samples].mean(axis=1) * 100
        )

        print("\n=== RSD BEFORE LOESS (High-QC Features) ===")
        print(qc_rsd_before.describe())
        print("\n=== RSD AFTER LOESS (High-QC Features) ===")
        print(qc_rsd_after.describe())

        worst_features = qc_rsd_after.nlargest(10)
        print("\nFeatures with highest RSD after LOESS (High-QC Features):")
        print(worst_features)

        # --- Plotting ---
        top_features = qc_rsd_before.nlargest(3).index
        intensity_df_rel = intensity_df.loc[high_qc_features, qc_samples].div(
            intensity_df.loc[high_qc_features, qc_samples].mean(axis=1), axis=0
        )
        corrected_df_rel = corrected_df.loc[high_qc_features, qc_samples].div(
            corrected_df.loc[high_qc_features, qc_samples].mean(axis=1), axis=0
        )

        if len(top_features) > 0:
            plt.figure(figsize=(12, 6))
            for feature in top_features:
                plt.plot(
                    intensity_df_rel.loc[feature], 'o--',
                    label=f"{feature} (Before)", alpha=0.7, linewidth=2
                )
                plt.plot(
                    corrected_df_rel.loc[feature], 'o-',
                    label=f"{feature} (After)", linewidth=2
                )
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title("Top 3 Corrected Features (Relative Intensities) Before/After LOESS")
            plt.xlabel("QC Sample")
            plt.ylabel("Relative Intensity")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/top_3_corrected_relative_features.png",
                dpi=200, bbox_inches='tight'
            )
            plt.close()

        # --- Save Results ---
        with open(f"{output_dir}/qc_rsd_after_loess.txt", "w") as f:
            f.write("Feature\tRSD (%)\n")
            f.write(qc_rsd_after.describe().to_string())
    else:
        print("⚠️ Warning: No high-QC features found for LOESS correction")

    # Reattach metadata columns to corrected data
    # Reset index to get Feature as a column
    corrected_df_reset = corrected_df.reset_index()
    
    # Merge with metadata
    if feature_col is not None:
        # Use the original metadata_df which has the correct Feature order
        result_df = pd.concat([metadata_df.reset_index(drop=True), corrected_df_reset], axis=1)
    else:
        result_df = corrected_df_reset
    
    # Save corrected data (ALL features, including low-QC ones, with metadata)
    result_df.to_csv(f"{output_dir}/{batch}_{mode}_drift_corrected.csv", index=False)

    return result_df