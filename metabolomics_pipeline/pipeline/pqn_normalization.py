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
    - Keeps 'Feature' as the first column in the output.
    """
    os.makedirs(output_dir, exist_ok=True)

    if corrected_df.empty:
        raise ValueError("Input DataFrame is empty")

    # Exclude metadata columns (e.g., 'Area:', 'PQF:', 'Gap', 'Peak', 'Number', 'Status')
    METADATA_PREFIXES = ['Area:', 'PQF:', 'Gap', 'Peak', 'Number', 'Status']
    sample_cols = [
        col for col in corrected_df.columns
        if col != 'Feature' and not any(col.startswith(prefix) for prefix in METADATA_PREFIXES)
    ]

    if not sample_cols:
        raise ValueError(f"No sample columns found in DataFrame. Columns: {corrected_df.columns.tolist()}")

    # Set Feature as index if it exists
    if 'Feature' in corrected_df.columns:
        corrected_df = corrected_df.set_index('Feature')

    # Identify expQC samples (case-insensitive)
    qc_cols = [col for col in sample_cols if 'expqc' in col.lower()]

    # Fallback to QC3 if no expQC samples are found
    if not qc_cols:
        logger.warning("No 'expQC' samples found. Trying 'QC3' as fallback.")
        qc_cols = [col for col in sample_cols if 'qc3' in col.lower()]

        if not qc_cols:
            raise ValueError(
                "No 'expQC', 'QC3', 'QC4', or 'blauw' samples found in DataFrame columns. "
                f"Available columns: {sample_cols}"
            )
        else:
            logger.info(f"Fallback QC samples found: {qc_cols}")
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

    # Reattach Feature index as the FIRST column
    normalized_df.insert(0, 'Feature', corrected_df.index)

    # Save normalized data
    normalized_df.to_csv(f"{output_dir}/{batch}_{mode}_pqn_normalized.csv")
    normalized_df.to_csv(f"{output_dir}/pqn_normalized.csv")

    central_dir = "/Users/j.groen/PycharmProjects/untargeted_pipeline/metabolomics_pipeline/data/pqn_normalized_batches"
    os.makedirs(central_dir, exist_ok=True)  # Ensure the directory exists
    normalized_df.to_csv(f"{central_dir}/{batch}_{mode}_pqn_normalized.csv")

    return normalized_df