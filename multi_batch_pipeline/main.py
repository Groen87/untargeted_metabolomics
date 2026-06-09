#!/usr/bin/env python3
"""
Main entry point for the multi-batch metabolomics data processing pipeline.

This script provides a complete workflow for processing multiple batch folders
(e.g., "MZ25_36", "MZ26_10") through the metabolomics pipeline and performing
multi-batch ComBat correction.

Workflow:
1. Process each batch folder independently for specified ion modes (NEG/POS)
   - Data loading and filtering
   - Drift correction with LOESS
   - Median normalization
2. Merge all median-normalized files using RT-based feature matching
3. Perform ComBat batch correction across all batches simultaneously
4. Generate QC reports (optional)

Usage:
    # Process all batches in data directory
    python multi_batch_pipeline/main.py
    
    # Process specific batches
    python multi_batch_pipeline/main.py --batches MZ25_36 MZ26_10 MZ27_15
    
    # Process only NEG mode
    python multi_batch_pipeline/main.py --batches MZ25_36 MZ26_10 --mode NEG
    
    # Disable QC reports
    python multi_batch_pipeline/main.py --batches MZ25_36 MZ26_10 --no-qc
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import traceback

# Import pipeline functions
from multi_batch_pipeline.config.config import Config
from multi_batch_pipeline.pipeline import (
    process_metabolomics_data,
    correct_drift_with_loess,
    median_normalize,
)
from multi_batch_pipeline.pipeline.multi_batch_combat import (
    find_median_files,
    merge_multiple_batches,
    run_multi_batch_combat,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Batch Discovery Functions
# =============================================================================

def find_batch_folders(data_dir: str = "data") -> List[str]:
    """
    Auto-discover batch folders in the data directory.
    
    A batch folder is identified by the presence of:
    - {batch_name}_NEG.csv and/or {batch_name}_POS.csv files
    - output/NEG/ and/or output/POS/ directories
    
    Args:
        data_dir: Base data directory to search (default: "data")
        
    Returns:
        Sorted list of batch folder names
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Data directory {data_dir} does not exist")
        return []
    
    batch_folders = []
    for item in data_path.iterdir():
        if item.is_dir():
            # Check if it contains NEG or POS data files
            neg_file = item / f"{item.name}_NEG.csv"
            pos_file = item / f"{item.name}_POS.csv"
            neg_dir = item / "output" / "NEG"
            pos_dir = item / "output" / "POS"
            
            if neg_file.exists() or pos_file.exists() or neg_dir.exists() or pos_dir.exists():
                batch_folders.append(item.name)
    
    return sorted(batch_folders)


# =============================================================================
# Single Batch Processing Functions
# =============================================================================

def process_single_batch(
    batch_folder: str,
    mode: str,
    config: Config,
    data_dir: str = "data",
) -> Tuple[Path, Path, bool]:
    """
    Process a single batch through the full pipeline.
    
    Performs the following steps:
    1. Load and transform raw data
    2. Apply LOESS drift correction
    3. Apply PQN normalization
    
    Args:
        batch_folder: Name of the batch folder to process
        mode: Ion mode, either 'NEG' or 'POS'
        config: Configuration object with pipeline parameters
        data_dir: Base data directory (default: "data")
        
    Returns:
        Tuple of:
        - median_normalized_path: Path to the median-normalized output file
        - batch_data_path: Path to the batch metadata output file
        - success: Boolean indicating if processing succeeded
    """
    try:
        # Construct paths
        input_file = Path(f"{data_dir}/{batch_folder}/{batch_folder}_{mode}.csv")
        metadata_file = Path(f"{data_dir}/{batch_folder}/{batch_folder}_meta.xlsx")
        output_dir = Path(f"{data_dir}/{batch_folder}/output/{mode}/")
        
        logger.info(f"  Processing {batch_folder}/{mode}...")
        
        # Check if already processed
        median_path = output_dir / "median_normalized.csv"
        batch_data_path = output_dir / "batch_data.csv"
        
        if median_path.exists() and batch_data_path.exists():
            logger.info(f"  ✓ {batch_folder}/{mode} already processed. Skipping.")
            return median_path, batch_data_path, True
        
        # Validate input files
        if not input_file.exists():
            logger.warning(f"  ✗ Input file not found: {input_file}")
            return None, None, False
        if not metadata_file.exists():
            logger.warning(f"  ✗ Metadata file not found: {metadata_file}")
            return None, None, False
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # --- Step 1: Data Processing ---
        logger.info(f"  [{mode}] Processing raw data...")
        transformed_df, batchdata_df = process_metabolomics_data(
            input_file=str(input_file),
            metadata_file=str(metadata_file),
            output_dir=str(output_dir),
        )
        
        # --- Step 2: Drift Correction ---
        logger.info(f"  [{mode}] Correcting drift with LOESS...")
        corrected_df = correct_drift_with_loess(
            transformed_df,
            qc_pattern=config["qc_pattern"],
            qc_intensity_threshold=config["qc_intensity_threshold"],
            frac=config["frac"],
            output_dir=str(output_dir),
            fallback_qc_pattern=config.get("fallback_qc_pattern", "QC3"),
        )
        
        # --- Step 3: Median Normalization ---
        logger.info(f"  [{mode}] Applying median normalization...")
        normalized_df = median_normalize(
            corrected_df,
            output_dir=str(output_dir),
            qc_pattern=config["qc_pattern"],
            fallback_qc_pattern=config.get("fallback_qc_pattern", "QC3"),
        )
        
        logger.info(f"  ✓ {batch_folder}/{mode} processed successfully!")
        return median_path, batch_data_path, True
        
    except Exception as e:
        logger.error(f"  ✗ Failed to process {batch_folder}/{mode}: {e}")
        logger.error(traceback.format_exc())
        return None, None, False


def process_all_batches(
    batch_folders: List[str],
    modes: List[str],
    config: Config,
    data_dir: str = "data",
) -> Dict[str, List[Tuple[str, Path, Path]]]:
    """
    Process all batches for all specified ion modes.
    
    Args:
        batch_folders: List of batch folder names to process
        modes: List of ion modes to process (e.g., ["NEG", "POS"])
        config: Configuration object with pipeline parameters
        data_dir: Base data directory (default: "data")
        
    Returns:
        Dictionary mapping mode to list of (batch_folder, median_path, batch_data_path) tuples
    """
    results = {mode: [] for mode in modes}
    
    for mode in modes:
        logger.info(f"\nProcessing {mode} mode for all batches...")
        for batch_folder in batch_folders:
            median_path, batch_data_path, success = process_single_batch(
                batch_folder=batch_folder,
                mode=mode,
                config=config,
                data_dir=data_dir,
            )
            if success:
                results[mode].append((batch_folder, median_path, batch_data_path))
        
        logger.info(f"  ✓ Processed {len(results[mode])}/{len(batch_folders)} batches for {mode}")
    
    return results


# =============================================================================
# Multi-Batch Pipeline Functions
# =============================================================================

def run_full_multi_batch_pipeline(
    batch_folders: List[str],
    modes: List[str] = ["NEG", "POS"],
    data_dir: str = "data",
    output_dir: Optional[Path] = None,
    rt_threshold: float = 0.02,
    show_plots: bool = False,
    save_plots: bool = True,
    run_qc: bool = True,
    config_path: Optional[str] = None,
) -> Dict[str, Tuple[pd.DataFrame, dict]]:
    """
    Run the complete multi-batch processing pipeline.
    
    This function orchestrates the entire workflow:
    1. Process each batch independently (data processing → drift correction → PQN)
    2. For each ion mode, merge all batches using RT-based feature matching
    3. Run ComBat batch correction on merged data
    4. Generate QC reports (optional)
    
    Args:
        batch_folders: List of batch folder names to process
        modes: List of ion modes to process (default: ["NEG", "POS"])
        data_dir: Base data directory (default: "data")
        output_dir: Base output directory for multi-batch results
            (default: data/multi_batch_output/)
        rt_threshold: RT threshold for feature matching in minutes (default: 0.02)
        show_plots: Whether to display plots interactively (default: False)
        save_plots: Whether to save plots to disk (default: True)
        run_qc: Whether to run QC analysis after ComBat (default: True)
        config_path: Path to config file (default: uses multi_batch_pipeline/config/config.yaml)
        
    Returns:
        Dictionary mapping mode to (combat_corrected_data, metrics) tuple
        
    Raises:
        ValueError: If no batches are specified or found
    """
    # Load configuration
    if config_path:
        config = Config(config_path)
    else:
        config = Config()
    
    if output_dir is None:
        output_dir = Path(f"{data_dir}/multi_batch_output/")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # --- Step 1: Process all batches for all modes ---
    logger.info(f"\n{'='*70}")
    logger.info(f"=== Step 1: Processing individual batches ===")
    logger.info(f"{'='*70}\n")
    
    batch_results = process_all_batches(
        batch_folders=batch_folders,
        modes=modes,
        config=config,
        data_dir=data_dir,
    )
    
    # --- Step 2: For each mode, run multi-batch ComBat ---
    for mode in modes:
        logger.info(f"\n{'='*70}")
        logger.info(f"=== Step 2: Multi-batch ComBat for {mode} ===")
        logger.info(f"{'='*70}\n")
        
        batch_files = batch_results[mode]
        
        if len(batch_files) < 2:
            logger.warning(
                f"Only {len(batch_files)} batches processed for {mode}. "
                f"Need at least 2 batches for ComBat. Skipping."
            )
            continue
        
        # Use the existing multi_batch_combat logic
        mode_output_dir = output_dir / mode
        
        try:
            combat_corrected_df, combat_metrics = run_multi_batch_combat(
                batch_folders=[bf[0] for bf in batch_files],
                mode=mode,
                output_dir=mode_output_dir,
                data_dir=data_dir,
                rt_threshold=rt_threshold,
                show_plots=show_plots,
                save_plots=save_plots,
                run_qc=run_qc,
                ref_batch=config.get("ref_batch"),
            )
            
            results[mode] = (combat_corrected_df, combat_metrics)
            logger.info(f"  ✓ ComBat correction completed for {mode}!")
            logger.info(f"  Output saved to {mode_output_dir}/\n")
            
        except Exception as e:
            logger.error(f"Failed to run multi-batch ComBat for {mode}: {e}")
            logger.error(traceback.format_exc())
    
    return results


# =============================================================================
# Command-line Interface
# =============================================================================

def main():
    """Command-line interface for multi-batch processing."""
    parser = argparse.ArgumentParser(
        description="Process multiple batches through the metabolomics pipeline "
                    "and perform multi-batch ComBat correction"
    )
    
    # Batch specification
    parser.add_argument(
        "--batches",
        nargs="+",
        default=None,
        help="List of batch folder names to process (default: auto-detect from data directory)",
    )
    
    # Mode specification
    parser.add_argument(
        "--mode",
        choices=["NEG", "POS"],
        default=None,
        help="Ion mode to process (default: both NEG and POS)",
    )
    
    # Directory options
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base data directory (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for multi-batch results "
             "(default: data/multi_batch_output/)",
    )
    
    # Algorithm parameters
    parser.add_argument(
        "--rt-threshold",
        type=float,
        default=0.02,
        help="RT threshold for feature matching in minutes (default: 0.02)",
    )
    
    # Visualization options
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots interactively",
    )
    
    # QC options
    parser.add_argument(
        "--no-qc",
        action="store_true",
        help="Disable QC report generation",
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: multi_batch_pipeline/config/config.yaml)",
    )
    
    args = parser.parse_args()
    
    try:
        # --- Determine batches to process ---
        if args.batches:
            batch_folders = args.batches
        else:
            batch_folders = find_batch_folders(args.data_dir)
            if not batch_folders:
                raise ValueError(
                    f"No batch folders found in {args.data_dir}. "
                    "Please specify --batches or ensure data directory contains batch folders."
                )
            logger.info(f"Auto-detected batch folders: {batch_folders}")
        
        # --- Determine modes to process ---
        if args.mode:
            modes = [args.mode]
        else:
            modes = ["NEG", "POS"]
        
        # --- Log configuration ---
        logger.info(f"\nStarting multi-batch pipeline...")
        logger.info(f"  Batches: {batch_folders}")
        logger.info(f"  Modes: {modes}")
        logger.info(f"  Data directory: {args.data_dir}")
        logger.info(f"  Output directory: {args.output_dir or f'{args.data_dir}/multi_batch_output/'}")
        logger.info(f"  RT threshold: {args.rt_threshold} minutes")
        logger.info(f"  Save plots: {not args.no_plots}")
        logger.info(f"  Run QC: {not args.no_qc}")
        
        # --- Run pipeline ---
        run_full_multi_batch_pipeline(
            batch_folders=batch_folders,
            modes=modes,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            rt_threshold=args.rt_threshold,
            show_plots=args.show_plots,
            save_plots=not args.no_plots,
            run_qc=not args.no_qc,
            config_path=args.config,
        )
        
        logger.info("\n" + "="*70)
        logger.info("Multi-batch pipeline completed successfully!")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Multi-batch pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
