"""
PQN (Probabilistic Quotient Normalization) module for metabolomics data.

This module provides functionality to apply PQN normalization, which is a
widely used method in metabolomics for correcting systematic biases between
samples while preserving the relative scale of features.

Key Features:
- Uses expQC samples as reference for normalization
- Preserves relative scale and rare metabolites (ideal for IMD workflows)
- Handles metadata columns appropriately
- Maintains Feature column as the first column

PQN Normalization:
    For each sample i:
        scaling_factor_i = median(sample_i) / reference_median
        normalized_sample_i = sample_i / scaling_factor_i
    
    where reference_median = median of (median(expQC_1), median(expQC_2), ...)
"""

import os
import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Metadata column prefixes that should be excluded from normalization
METADATA_PREFIXES = ['Area:', 'PQF:', 'Gap', 'Peak', 'Number', 'Status']


def identify_sample_columns(
    columns: List[str],
    exclude_prefixes: Optional[List[str]] = None,
) -> List[str]:
    """
    Identify sample columns from DataFrame columns.
    
    Excludes metadata columns and the 'Feature' column.
    
    Args:
        columns: List of DataFrame column names
        exclude_prefixes: List of prefixes to exclude (default: METADATA_PREFIXES)
        
    Returns:
        List of column names that are sample columns
    """
    if exclude_prefixes is None:
        exclude_prefixes = METADATA_PREFIXES
    
    sample_cols = [
        col for col in columns
        if col != 'Feature' and not any(col.startswith(prefix) for prefix in exclude_prefixes)
    ]
    return sample_cols


def identify_qc_samples(
    sample_cols: List[str],
    qc_pattern: str = "expQC",
) -> List[str]:
    """
    Identify QC samples from sample column names.
    
    Args:
        sample_cols: List of sample column names
        qc_pattern: Pattern to identify QC samples (case-insensitive, default: "expQC")
        
    Returns:
        List of QC sample column names
    """
    return [col for col in sample_cols if qc_pattern.lower() in col.lower()]


def calculate_pqn_scaling_factors(
    data: pd.DataFrame,
    qc_cols: List[str],
    sample_cols: List[str],
) -> pd.Series:
    """
    Calculate PQN scaling factors for each sample.
    
    The scaling factor for each sample is:
        sample_median / reference_median
    
    where reference_median is the median of all QC sample medians.
    
    Args:
        data: DataFrame with feature data (features x samples)
        qc_cols: List of QC sample column names
        sample_cols: List of all sample column names
        
    Returns:
        Series with scaling factors indexed by sample column names
    """
    # Calculate median for each QC sample
    qc_medians = data[qc_cols].median(axis=0)
    
    # Calculate reference median: median of QC sample medians
    reference_median = qc_medians.median()
    logger.info(f"Reference median (from {len(qc_cols)} QC samples): {reference_median:.2f}")
    
    # Calculate median for each sample
    sample_medians = data[sample_cols].median(axis=0)
    
    # Calculate scaling factors: sample_median / reference_median
    scaling_factors = sample_medians / reference_median
    
    return scaling_factors


def pqn_normalize(
    corrected_df: pd.DataFrame,
    output_dir: str = "output",
    qc_pattern: str = "expQC",
) -> pd.DataFrame:
    """
    Apply PQN (Probabilistic Quotient Normalization) to metabolomics data.
    
    PQN normalization corrects for systematic biases between samples by scaling
    each sample so that the median intensity of that sample matches a reference median
    calculated from QC samples. This preserves the relative scale of features while
    correcting for overall sample-to-sample differences.
    
    Steps:
    1. Identify sample columns (excluding metadata and Feature)
    2. Identify QC samples from sample columns
    3. Calculate reference median from QC sample medians
    4. Calculate scaling factor for each sample: sample_median / reference_median
    5. Divide each sample by its scaling factor
    6. Save normalized data to pqn_normalized.csv
    
    Args:
        corrected_df: Input DataFrame with drift-corrected data.
            Should have features as rows and samples as columns.
            May have a 'Feature' column and/or metadata columns.
        output_dir: Directory to save output files (default: "output")
        qc_pattern: Pattern to identify QC samples (case-insensitive, default: "expQC")
        
    Returns:
        DataFrame with PQN-normalized data.
        The 'Feature' column is the first column, followed by normalized sample columns.
        
    Raises:
        ValueError: If input DataFrame is empty
        ValueError: If no sample columns are found
        ValueError: If no QC samples are found
    
    Note:
        This normalization method is particularly suitable for metabolomics data
        because it preserves the relative scale of features, which is important
        for detecting biologically meaningful differences.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if corrected_df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Identify sample columns
    sample_cols = identify_sample_columns(corrected_df.columns)
    
    if not sample_cols:
        raise ValueError(
            f"No sample columns found in DataFrame. "
            f"Columns: {corrected_df.columns.tolist()}"
        )
    
    # Set Feature as index if it exists
    if 'Feature' in corrected_df.columns:
        corrected_df = corrected_df.set_index('Feature')
    
    # Identify QC samples
    qc_cols = identify_qc_samples(sample_cols, qc_pattern)
    
    if not qc_cols:
        raise ValueError(
            f"No QC samples matching pattern '{qc_pattern}' found in DataFrame columns. "
            f"Available columns: {sample_cols}"
        )
    
    logger.info(f"Found {len(qc_cols)} QC samples: {qc_cols}")
    
    # Calculate PQN scaling factors
    scaling_factors = calculate_pqn_scaling_factors(
        corrected_df, qc_cols, sample_cols
    )
    
    # Apply PQN: divide each sample by its scaling factor
    normalized_df = corrected_df[sample_cols].div(scaling_factors, axis=1)
    
    # Reattach Feature index as the FIRST column
    normalized_df.insert(0, 'Feature', corrected_df.index)
    
    # Save normalized data
    output_path = os.path.join(output_dir, "pqn_normalized.csv")
    normalized_df.to_csv(output_path, index=False)
    logger.info(f"Saved PQN-normalized data to {output_path}")
    
    return normalized_df
