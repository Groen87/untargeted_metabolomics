"""
ComBat batch correction for metabolomics data.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from combat.pycombat import combat  # Assuming ComBat is used

def run_combat_correction(
    merged_data_path: str,
    merged_batch_path: str,
    output_dir: str,
    current_batch_label: int = 1,
    reference_batch_label: int = 2,
) -> pd.DataFrame:
    """
    Run ComBat batch correction on merged data and extract the current batch.

    Args:
        merged_data_path: Path to merged_data_for_combat.csv
        merged_batch_path: Path to merged_batch_for_combat.csv
        output_dir: Directory to save corrected current batch
        current_batch_label: Label for current batch (default: 1)
        reference_batch_label: Label for reference batch (default: 2)

    Returns:
        DataFrame of batch-corrected current batch samples
    """
    # Load merged files
    merged_data = pd.read_csv(merged_data_path, index_col="Feature")
    merged_batch = pd.read_csv(merged_batch_path)

    # Get sample IDs (columns) from merged_data
    sample_ids = [col for col in merged_data.columns if col != "Feature"]

    # Create batch vector: map sample_id to batch value
    batch_map = dict(zip(merged_batch["sample_id"], merged_batch["batch"]))
    batch_vector = np.array([batch_map[sid] for sid in sample_ids])

    # Run ComBat
    corrected_data = combat(
        data=merged_data.values,
        batch=batch_vector,
        mod=None,  # No moderator variable
        par_prior=True,
        prior_plots=False,
    )

    # Convert back to DataFrame
    corrected_df = pd.DataFrame(
        corrected_data,
        index=merged_data.index,
        columns=sample_ids,
    )

    # Save corrected data
    output_path = Path(output_dir) / "combat_corrected.csv"
    corrected_df.to_csv(output_path)
    return corrected_df