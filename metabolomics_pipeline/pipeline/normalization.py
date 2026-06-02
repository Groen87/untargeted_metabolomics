import os
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def pqn_normalize(
    corrected_df: pd.DataFrame,
    output_dir: str = "output",
) -> pd.DataFrame:
    """
    Apply PQN (Probabilistic Quotient Normalization) to metabolomics data.
    - Uses 'expQC' samples (case-insensitive) as reference.
    - Normalizes ONLY sample columns (excludes metadata like 'Area:', 'PQF:', etc.).
    - Preserves relative scale and rare metabolites (ideal for IMD workflows).
    """
    os.makedirs(output_dir, exist_ok=True)

    if corrected_df.empty:
        raise ValueError("Input DataFrame is empty")

    # Exclude metadata columns (e.g., 'Area:', 'PQF:', 'Gap Status:', etc.)
    METADATA_PREFIXES = ['Area:', 'PQF:', 'Gap', 'Peak', 'Number', 'Status', 'Feature']
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

    if not qc_cols:
        raise ValueError("No 'expQC' samples found in DataFrame columns.")

    logger.info(f"Found {len(qc_cols)} expQC samples: {qc_cols}")

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

    # Reattach Feature index as column
    normalized_df['Feature'] = corrected_df.index

    # Save normalized data
    normalized_df.to_csv(f"{output_dir}/pqn_normalized_data.csv", index=False)

    return normalized_df