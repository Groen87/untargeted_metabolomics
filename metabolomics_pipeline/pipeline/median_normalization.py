import os
import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def median_normalize(
    batch: str,
    mode: str,
    corrected_df: pd.DataFrame,
    output_dir: str = "output",
) -> pd.DataFrame:
    """
    Apply median normalization to metabolomics data.
    - Uses 'expQC' samples (case-insensitive) as reference for scaling.
    - Normalizes ONLY sample columns (excludes metadata like 'Area:', 'PQF:', etc.).
    - Each sample is divided by the median of QC samples.
    - Preserves relative scale and handles missing values.
    - Keeps 'Feature' and 'RT [min]' as the first columns in the output.
    """
    os.makedirs(output_dir, exist_ok=True)

    if corrected_df.empty:
        raise ValueError("Input DataFrame is empty")

    logger.info(f"Median normalization input columns: {list(corrected_df.columns)}")

    # Identify metadata columns to preserve (Compounds ID, Feature, RT [min])
    metadata_cols = []
    if 'Compounds ID' in corrected_df.columns:
        metadata_cols.append('Compounds ID')
    if 'Feature' in corrected_df.columns:
        metadata_cols.append('Feature')
    if 'RT [min]' in corrected_df.columns:
        metadata_cols.append('RT [min]')
    
    logger.info(f"Metadata columns: {metadata_cols}")

    # Exclude metadata columns and other metadata prefixes
    METADATA_PREFIXES = ['Area:', 'PQF:', 'Gap', 'Peak', 'Number', 'Status']
    sample_cols = [
        col for col in corrected_df.columns
        if col not in metadata_cols and not any(col.startswith(prefix) for prefix in METADATA_PREFIXES)
    ]

    if not sample_cols:
        raise ValueError(f"No sample columns found in DataFrame. Columns: {corrected_df.columns.tolist()}")

    # Store metadata for later reattachment
    metadata_df = corrected_df[metadata_cols].copy()
    
    # Set Compounds ID as index if it exists (unique identifier), otherwise Feature
    if 'Compounds ID' in corrected_df.columns:
        corrected_df = corrected_df.set_index('Compounds ID')
    elif 'Feature' in corrected_df.columns:
        corrected_df = corrected_df.set_index('Feature')

    # Identify QC samples (case-insensitive) - try multiple patterns
    qc_patterns = ['expqc', 'qc3', 'qc4', 'qc']
    qc_cols = []
    for pattern in qc_patterns:
        qc_cols = [col for col in sample_cols if pattern in col.lower()]
        if qc_cols:
            logger.info(f"Found QC samples with pattern '{pattern}': {qc_cols}")
            break

    if not qc_cols:
        raise ValueError(
            f"No QC samples found with any pattern. Tried: {qc_patterns}. "
            f"Available columns: {sample_cols}"
        )

    # Calculate median of QC samples (reference)
    qc_data = corrected_df[qc_cols].replace('', np.nan).astype(float)
    reference_median = qc_data.median().median()  # Median of QC medians
    logger.info(f"Reference median (from expQC samples): {reference_median}")

    # Calculate median for each sample
    sample_data = corrected_df[sample_cols].replace('', np.nan).astype(float)
    sample_medians = sample_data.median(axis=0)

    # Calculate scaling factors: reference_median / sample_median
    # This normalizes each sample to the reference median
    scaling_factors = reference_median / sample_medians

    # Apply median normalization: multiply each sample by its scaling factor
    normalized_df = sample_data.multiply(scaling_factors, axis=1)

    # Reset index to get Compounds ID or Feature as a column
    normalized_df = normalized_df.reset_index()
    
    # Reattach all metadata columns
    # normalized_df has the index column from reset_index, metadata_df has all metadata
    # Drop the index column from normalized_df to avoid duplication
    index_col_name = 'Compounds ID' if 'Compounds ID' in metadata_df.columns else 'Feature'
    if index_col_name in normalized_df.columns:
        normalized_df = normalized_df.drop(columns=[index_col_name])
    
    result_df = pd.concat([metadata_df.reset_index(drop=True), normalized_df], axis=1)

    # Save normalized data
    result_df.to_csv(f"{output_dir}/{batch}_{mode}_median_normalized.csv", index=False)
    result_df.to_csv(f"{output_dir}/median_normalized.csv", index=False)

    central_dir = "/Users/j.groen/PycharmProjects/untargeted_pipeline/metabolomics_pipeline/data/median_normalized_batches"
    os.makedirs(central_dir, exist_ok=True)
    result_df.to_csv(f"{central_dir}/{batch}_{mode}_median_normalized.csv", index=False)

    return result_df
