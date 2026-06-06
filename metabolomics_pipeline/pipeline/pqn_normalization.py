import os
import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def pqn_normalize(
    batch: str,
    mode: str,
    corrected_df: pd.DataFrame,
    output_dir: str = "output",

) -> pd.DataFrame:
    """
    Apply PQN (Probabilistic Quotient Normalization) to metabolomics data.
    - Uses 'expQC' samples (case-insensitive) as reference.
    - Normalizes ONLY sample columns (excludes metadata like 'Area:', 'PQF:', etc.).
    - Preserves relative scale and rare metabolites (ideal for IMD workflows).
    - Keeps 'Feature' and 'RT [min]' as the first columns in the output.
    """
    os.makedirs(output_dir, exist_ok=True)

    if corrected_df.empty:
        raise ValueError("Input DataFrame is empty")

    # Identify metadata columns to preserve (Feature, RT [min])
    metadata_cols = []
    if 'Feature' in corrected_df.columns:
        metadata_cols.append('Feature')
    if 'RT [min]' in corrected_df.columns:
        metadata_cols.append('RT [min]')
    
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
    
    # Set Feature as index if it exists
    if 'Feature' in corrected_df.columns:
        corrected_df = corrected_df.set_index('Feature')

    # Identify expQC samples (case-insensitive)
    qc_cols = [col for col in sample_cols if 'expqc' in col.lower()]

    if not qc_cols:
        raise ValueError(
            "No 'expQC' samples found in DataFrame columns. "
            f"Available columns: {sample_cols}"
        )
    else:
        logger.info(f"Found expQC samples: {qc_cols}")

    # Calculate median for each QC sample
    qc_medians = corrected_df[qc_cols].median(axis=0)

    # Calculate reference median: median of QC sample medians
    reference_median = qc_medians.median()
    logger.info(f"Reference median (from expQC samples): {reference_median}")

    # Calculate median for each sample
    sample_medians = corrected_df[sample_cols].median(axis=0)

    # Calculate scaling factors: sample_median / reference_median
    scaling_factors = sample_medians / reference_median

    # Apply PQN: divide each sample by its scaling factor
    normalized_df = corrected_df[sample_cols].div(scaling_factors, axis=1)

    # Reset index to get Feature as a column
    normalized_df = normalized_df.reset_index()
    
    # Reattach all metadata columns (Feature, RT [min])
    # metadata_df already has Feature and RT [min], normalized_df has Feature from reset_index
    # So we need to drop the Feature column from normalized_df to avoid duplication
    if 'Feature' in normalized_df.columns:
        normalized_df = normalized_df.drop(columns=['Feature'])
    
    result_df = pd.concat([metadata_df.reset_index(drop=True), normalized_df], axis=1)

    # Save normalized data
    result_df.to_csv(f"{output_dir}/{batch}_{mode}_pqn_normalized.csv", index=False)
    result_df.to_csv(f"{output_dir}/pqn_normalized.csv", index=False)

    central_dir = "/Users/j.groen/PycharmProjects/untargeted_pipeline/metabolomics_pipeline/data/pqn_normalized_batches"
    os.makedirs(central_dir, exist_ok=True)  # Ensure the directory exists
    result_df.to_csv(f"{central_dir}/{batch}_{mode}_pqn_normalized.csv", index=False)

    return result_df