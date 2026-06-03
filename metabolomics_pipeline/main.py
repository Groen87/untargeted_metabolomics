#!/usr/bin/env python3
"""
Main script to run the metabolomics data processing pipeline for NEG and POS ion modes.
Performs drift correction, normalization, and ComBat batch correction.
"""

import logging
import sys
from pathlib import Path
import pandas as pd

# Import pipeline functions
from metabolomics_pipeline import Config
from metabolomics_pipeline.pipeline import (
    process_metabolomics_data,
    correct_drift_with_loess,
    pqn_normalize,
    merge_batches_for_combat,
)
from metabolomics_pipeline.pipeline.combat_utils import run_combat_and_visualize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    """Run the full pipeline for NEG and POS ion modes."""
    try:
        # Load config
        config = Config()
        batch_folder = config["batch_folder"]
        reference_batch = config.get("reference_batch")
        logger.info(f"Starting pipeline for batch: {batch_folder}")

        # Process each ion mode
        for mode in ["NEG", "POS"]:
            logger.info(f"\n=== Processing {mode} mode ===")

            # Construct paths
            input_file = Path(f"data/{batch_folder}/{batch_folder}_{mode}.csv")
            metadata_file = Path(f"data/{batch_folder}/{batch_folder}_meta.xlsx")
            output_dir = Path(f"data/{batch_folder}/output/{mode}/")

            # Debug: Check paths
            logger.info(f"Input file: {input_file} (exists: {input_file.exists()})")
            logger.info(f"Metadata file: {metadata_file} (exists: {metadata_file.exists()})")

            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            if not metadata_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {output_dir}")

            # Step 1: Data Processing
            logger.info(f"[{mode}] Processing raw data...")
            transformed_df, batchdata_df = process_metabolomics_data(
                input_file=str(input_file),
                metadata_file=str(metadata_file),
                output_dir=str(output_dir),
            )

            # Step 2: Drift Correction
            logger.info(f"[{mode}] Correcting drift with LOESS...")
            corrected_df = correct_drift_with_loess(
                transformed_df,
                qc_pattern=config["qc_pattern"],
                qc_intensity_threshold=config["qc_intensity_threshold"],
                frac=config["frac"],
                output_dir=str(output_dir),
            )

            # Step 3: PQN Normalization
            logger.info(f"[{mode}] Applying PQN normalization...")
            normalized_df = pqn_normalize(
                corrected_df,
                output_dir=str(output_dir),
            )

            # Step 4: Merge with Reference Batch for ComBat
            if reference_batch:
                ref_data_file = Path(f"data/{reference_batch}/output/{mode}/pqn_normalized.csv")
                ref_batch_file = Path(f"data/{reference_batch}/output/{mode}/batch_data.csv")

                if ref_data_file.exists() and ref_batch_file.exists():
                    logger.info(f"[{mode}] Merging with reference batch {reference_batch} for ComBat...")

                    # Define directories
                    combat_input_dir = output_dir / "combat_input"
                    combat_output_dir = output_dir / "combat_corrected"
                    combat_input_dir.mkdir(parents=True, exist_ok=True)
                    combat_output_dir.mkdir(parents=True, exist_ok=True)

                    # Merge batches
                    merged_data, merged_batch = merge_batches_for_combat(
                        drift_corrected_file_batch1=str(output_dir / "pqn_normalized.csv"),
                        drift_corrected_file_batch2=str(ref_data_file),
                        batch_file_batch1=str(output_dir / "batch_data.csv"),
                        batch_file_batch2=str(ref_batch_file),
                        combat_input_dir=combat_input_dir,
                        combat_output_dir=combat_output_dir,
                        batch1_label="current",
                        batch2_label="reference"
                    )

                    logger.info(f"[{mode}] Merged data for ComBat saved to {combat_input_dir}/")

                    # Debug: Check the merged files
                    merged_data = pd.read_csv(str(combat_input_dir / "merged_data_for_combat.csv"), index_col=0)
                    merged_batch = pd.read_csv(str(combat_input_dir / "merged_batch_for_combat.csv"))
                    logger.info(f"Merged data shape: {merged_data.shape} (features x samples)")
                    logger.info(f"Merged batch shape: {merged_batch.shape} (samples x metadata)")

                    # --- STEP 5: RUN COMBAT (INDENTED TO STAY INSIDE THE IF BLOCK) ---
                    logger.info(f"[{mode}] Running ComBat batch correction...")
                    combat_output_dir.mkdir(parents=True, exist_ok=True)

                    combat_corrected_df, combat_metrics = run_combat_and_visualize(
                        merged_data_path=str(combat_input_dir / "merged_data_for_combat.csv"),
                        merged_batch_path=str(combat_input_dir / "merged_batch_for_combat.csv"),
                        output_dir=str(combat_output_dir),
                        show_plots=False,
                        save_plots=True,
                    )
                    logger.info(f"[{mode}] ComBat correction saved to {combat_output_dir}/")
                    logger.info(f"[{mode}] ComBat metrics: {combat_metrics}")
                else:
                    logger.warning(f"[{mode}] Reference files not found. Skipping ComBat.")
            else:
                logger.warning(f"[{mode}] No reference_batch in config. Skipping ComBat.")

            logger.info(f"[{mode}] Pipeline completed! Output saved to {output_dir}/")

        logger.info("\nAll ion modes processed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()