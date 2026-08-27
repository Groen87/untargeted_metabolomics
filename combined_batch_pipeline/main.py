#!/usr/bin/env python3
"""
Main entry point for the combined batch metabolomics pipeline.

This pipeline processes a single CSV file containing all batches combined,
where:
- All batches are in one file
- Columns are named: "Area: {filename} ({F#})"
- Batch name is embedded in the filename (e.g., posneg_MZ25_36_...)
- Duplicates have _1.raw and _2.raw suffixes
- Features are already aligned by Compound Discoverer
- Injection order comes from metadata file creation dates (deduplicated)

Workflow:
1. Load combined CSV
2. Load metadata file for injection order (based on creation dates, deduplicated)
3. Extract batch information from column names
4. For each batch:
   a. Average duplicate samples (_1 + _2)
   b. Apply PQN normalization
   c. Apply LOESS drift correction (using injection order from metadata)
5. Merge all batches
6. Run ComBat batch correction
7. Run RALPS batch correction (alternative method)
8. Generate QC reports (optional)
9. Generate comparison plots (ComBat vs RALPS)

Usage:
    python combined_batch_pipeline/main.py --input data/combined_all_batches.csv --metadata metadata.csv
    python combined_batch_pipeline/main.py --input data/combined.csv --metadata metadata.csv --output output/my_run
    python combined_batch_pipeline/main.py --input data/combined.csv --metadata metadata.csv --no-qc --no-plots
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
import re

from combined_batch_pipeline.config.config import Config
from combined_batch_pipeline.pipeline.data_loader import (
    load_combined_data,
    extract_batch_from_filename,
    average_duplicates,
    get_all_batches,
    get_batch_samples,
)
from combined_batch_pipeline.pipeline.batch_processing import (
    process_batch,
    merge_batch_results,
    identify_qc_samples,
)
from combined_batch_pipeline.pipeline.feature_filtering import filter_features
from combined_batch_pipeline.pipeline.combat_correction import run_combat_on_merged_data
from combined_batch_pipeline.pipeline.quality_control import run_qc_analysis, log_qc_rsd_simple
from combined_batch_pipeline.pipeline.ralps_correction import run_ralps_correction
from combined_batch_pipeline.pipeline.batch_effect_analysis import analyze_batch_effects
from combined_batch_pipeline.pipeline.injection_order import clean_sample_name

# Configure logging - add DEBUG level for troubleshooting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def extract_base_name_from_column(col: str) -> str:
    """Extract base name from column without _1/_2 suffix."""
    clean_name = col.split('Area: ')[1].split('.raw')[0].split(' (')[0].strip()
    if clean_name.endswith('_1'):
        return clean_name[:-2]
    elif clean_name.endswith('_2'):
        return clean_name[:-2]
    return clean_name


def run_full_pipeline(
    input_file: str,
    output_dir: str = "output/combined_batch_pipeline",
    metadata_file: Optional[str] = None,
    config_path: Optional[str] = None,
    qc_pattern: str = "expQC",
    fallback_qc_pattern: str = "QC3",
    frac: float = 0.5,
    ref_batch: Optional[int] = None,
    run_qc: bool = True,
    save_plots: bool = True,
    show_plots: bool = False,
) -> Dict:
    """
    Run the complete combined batch pipeline.
    
    Args:
        input_file: Path to combined CSV file
        output_dir: Output directory
        metadata_file: Path to metadata CSV file (for injection order)
        config_path: Path to config file (optional)
        qc_pattern: Pattern to identify QC samples
        fallback_qc_pattern: Fallback QC pattern
        frac: LOESS fraction parameter
        ref_batch: Reference batch for ComBat
        run_qc: Whether to run QC analysis
        save_plots: Whether to save plots
        show_plots: Whether to show plots
        
    Returns:
        Dictionary with results and metrics
    """
    # Load configuration
    if config_path:
        config = Config(config_path)
    else:
        config = Config()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"COMBINED BATCH PIPELINE")
    logger.info(f"{'='*70}")
    logger.info(f"Input file: {input_file}")
    if metadata_file:
        logger.info(f"Metadata file: {metadata_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Step 1: Load data
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP 1: Loading data")
    logger.info(f"{'='*70}")
    
    df, batch_groups, sample_info, injection_order = load_combined_data(
        input_file=input_file,
        intensity_threshold=config.get("intensity_threshold", 10000),
        metadata_file=metadata_file,
    )
    
    all_batches = sorted(batch_groups.keys())
    logger.info(f"Identified {len(all_batches)} batches: {all_batches}")
    
    # Debug: Print injection order info
    if injection_order:
        logger.debug(f"Injection order loaded for {len(injection_order)} columns")
        logger.debug(f"Sample injection order entries: {list(injection_order.items())[:5]}")
    else:
        logger.warning("No injection order loaded!")
    
    # Ensure injection_order is a dict (not None)
    if injection_order is None:
        injection_order = {}
        logger.warning("No injection order loaded from metadata. Using column order.")
    

    # Step 1.5: Global feature filtering (before batch processing)
    # This ensures all batches have the same features for ComBat correction
    logger.info(f"\n{'='*70}")
    logger.info("STEP 1.5: Global feature filtering")
    logger.info(f"{'='*70}")
    
    # Build filter config from main config
    filter_config = {
        'filter_low_variance': config.get('filter_low_variance', True),
        'variance_threshold': config.get('variance_threshold', 0.01),
        'variance_quantile': config.get('variance_quantile', None),
        'filter_single_batch': config.get('filter_single_batch', True),
        'min_batches': config.get('min_batches', 2),
        'filter_low_intensity': config.get('filter_low_intensity', True),
        'intensity_threshold': config.get('intensity_threshold', 10000.0),
        'intensity_quantile': config.get('intensity_quantile', None),
        'filter_qc_present': config.get('filter_qc_present', True),
        'filter_qc_intensity': config.get('filter_qc_intensity', True),
        'qc_intensity_quantile': config.get('qc_intensity_quantile', 0.25),
        'filter_blank_contaminants': config.get('filter_blank_contaminants', True),
        'blank_pattern': config.get('blank_pattern', 'blanco'),
        'blank_ratio_threshold': config.get('blank_ratio_threshold', 2.0),
        'filter_high_qc3_rsd': config.get('filter_high_qc3_rsd', True),
        'qc3_rsd_threshold': config.get('qc3_rsd_threshold', 30.0),
        'qc_pattern': qc_pattern,
        'fallback_qc_pattern': fallback_qc_pattern,
    }
    
    # Build batch_info for global filtering
    batch_info = {}
    for batch in all_batches:
        for col in batch_groups[batch]:
            batch_info[col] = batch
    
    # Get all sample columns (Area: columns)
    area_cols = [col for col in df.columns if col.startswith('Area:')]
    
    if area_cols:
        logger.info(f"Applying global feature filtering to {len(df)} features...")
        df = filter_features(
            df=df,
            sample_cols=area_cols,
            batch_info=batch_info,
            sample_info=sample_info,
            config=filter_config,
        )
        logger.info(f"Global feature filtering complete. Data shape: {df.shape}")
    else:
        logger.warning("No Area: columns found. Skipping global feature filtering.")
    
    # Remove blank samples from data AFTER filtering to prevent contamination
    # (Filtering needs blank samples for calculations, e.g., blank contaminant filter)
    blank_pattern = config.get('blank_pattern', 'blanco')
    blank_samples = []
    for col in df.columns:
        if blank_pattern.lower() in col.lower():
            blank_samples.append(col)
    
    if blank_samples:
        logger.info(f"Removing {len(blank_samples)} blank samples from data (after filtering)")
        logger.debug(f"Blank samples removed: {blank_samples[:5]}...")
        df = df.drop(columns=blank_samples)
        
        # Also remove from batch_groups
        for batch, samples in list(batch_groups.items()):
            batch_groups[batch] = [s for s in samples if s not in blank_samples]
        
        # Remove from injection_order
        for blank_col in blank_samples:
            injection_order.pop(blank_col, None)
        
        logger.info(f"Data shape after removing blanks: {df.shape}")

    # Step 2: Process each batch
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP 2: Processing batches")
    logger.info(f"{'='*70}")
    
    batch_results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    
    for batch in all_batches:
        batch_samples = batch_groups[batch]
        logger.debug(f"\nProcessing batch {batch} with {len(batch_samples)} samples")
        logger.debug(f"Sample columns: {batch_samples[:5]}...")
        
        # First, average duplicates within this batch
        batch_df = df[batch_samples].copy()
        batch_df, col_mapping = average_duplicates(batch_df, batch_samples)
        
        # Update sample names after averaging
        averaged_samples = list(batch_df.columns)
        logger.debug(f"After averaging: {len(averaged_samples)} columns")
        logger.debug(f"Averaged columns: {averaged_samples[:5]}...")
        logger.debug(f"Column mapping: {list(col_mapping.items())[:5]}")
        
        # Build updated sample_info and injection_order for averaged columns
        updated_sample_info: Dict[str, Dict] = {}
        updated_injection_order: Dict[str, int] = {}
        
        for new_col in averaged_samples:
            if new_col in col_mapping:
                original_cols = col_mapping[new_col]
                # col_mapping can be a string (single col) or "average of [col1, col2]"
                if isinstance(original_cols, str):
                    if original_cols.startswith('average of '):
                        # Parse the list from the string
                        orig_list_str = original_cols.replace('average of ', '')
                        orig_list_str = orig_list_str.strip('[]')
                        original_cols_list = [c.strip().strip("'\"") for c in orig_list_str.split(',')]
                    else:
                        original_cols_list = [original_cols]
                else:
                    original_cols_list = [original_cols]
                
                # Use info from first original column
                first_orig = original_cols_list[0]
                if first_orig in sample_info:
                    updated_sample_info[new_col] = sample_info[first_orig].copy()
                    updated_sample_info[new_col]['original_col'] = new_col
                else:
                    updated_sample_info[new_col] = {
                        'sample_id': new_col,
                        'batch': batch,
                        'sample_type': 'Sample',
                        'original_col': new_col,
                        'injection_order': -1,
                    }
                
                # Use injection order from first original column
                # The injection_order maps original column names to indices
                if first_orig in injection_order:
                    updated_injection_order[new_col] = injection_order[first_orig]
                    logger.debug(f"  Mapped {new_col} -> injection order {injection_order[first_orig]} (from {first_orig})")
                else:
                    # Try to find a column with matching base name
                    base_name = extract_base_name_from_column(first_orig)
                    matched = False
                    for orig_col, idx in injection_order.items():
                        if extract_base_name_from_column(orig_col) == base_name:
                            updated_injection_order[new_col] = idx
                            logger.debug(f"  Mapped {new_col} -> injection order {idx} (from {orig_col}, matched by base name)")
                            matched = True
                            break
                    if not matched:
                        updated_injection_order[new_col] = -1
                        logger.debug(f"  WARNING: Could not find injection order for {new_col} (from {first_orig})")
            else:
                updated_sample_info[new_col] = {
                    'sample_id': new_col,
                    'batch': batch,
                    'sample_type': 'Sample',
                    'original_col': new_col,
                    'injection_order': -1,
                }
                updated_injection_order[new_col] = -1
                logger.debug(f"  WARNING: No mapping for {new_col}")
        
        logger.debug(f"Updated injection order for batch: {updated_injection_order}")
        
        # Process the batch
        processed_df, batch_metadata = process_batch(
            df=batch_df,
            batch=batch,
            batch_samples=averaged_samples,
            sample_info=updated_sample_info,
            injection_order=updated_injection_order if injection_order else None,
            qc_pattern=qc_pattern,
            fallback_qc_pattern=fallback_qc_pattern,
            frac=frac,
            output_dir=output_dir / "batch_outputs" / batch,
        )
        
        batch_results[batch] = (processed_df, batch_metadata)
    
    # Step 3: Merge all batches
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP 3: Merging batches")
    logger.info(f"{'='*70}")
    
    merged_data, merged_metadata = merge_batch_results(
        batch_results=batch_results,
        output_dir=output_dir / "merged",
    )
    
    logger.info(f"Merged data shape: {merged_data.shape}")
    logger.info(f"Merged metadata shape: {merged_metadata.shape}")
    
    # Clean batch_groups to only include samples that are in merged_data
    # (some samples may have been removed during processing)
    # Keep ALL samples including QC for RALPS
    cleaned_batch_groups = {}
    for batch, samples in batch_groups.items():
        cleaned_samples = [s for s in samples if s in merged_data.columns]
        if cleaned_samples:
            cleaned_batch_groups[batch] = cleaned_samples
    
    # Create a copy for ComBat (without QC samples)
    qc_pattern = config.get('qc_pattern', 'expQC')
    fallback_qc_pattern = config.get('fallback_qc_pattern', 'QC3')
    
    combat_batch_groups = {}
    for batch, samples in cleaned_batch_groups.items():
        filtered_samples = []
        for sample in samples:
            if qc_pattern not in sample and (fallback_qc_pattern is None or fallback_qc_pattern not in sample):
                filtered_samples.append(sample)
        combat_batch_groups[batch] = filtered_samples
    
    # Use cleaned_batch_groups for RALPS (has QC samples, only samples in merged_data)
    batch_groups = cleaned_batch_groups
    
    logger.info(f"Cleaned batch_groups to match merged data ({len(merged_data.columns)} columns)")
    logger.info(f"Created combat_batch_groups (QC samples removed for ComBat)")
    
    # Step 4: ComBat correction
    # Step 3.5: Batch effect analysis (before correction)
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP 3.5: Batch effect analysis (before ComBat)")
    logger.info(f"{'='*70}")
    
    # Get batch labels from metadata
    batch_dict = dict(zip(merged_metadata['original_col'], merged_metadata['batch']))
    batch_vector = np.array([batch_dict.get(col, 'unknown') for col in merged_data.columns])
    
    # Analyze batch effects
    pre_combat_batch_metrics = analyze_batch_effects(
        data=merged_data,
        batch_labels=batch_vector,
        output_dir=output_dir / "batch_effect_analysis",
        prefix="pre_combat"
    )
    
    # Step 4: ComBat batch correction
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP 4: ComBat batch correction")
    logger.info(f"{'='*70}")
    
    corrected_data, combat_metrics = run_combat_on_merged_data(
        merged_data=merged_data,
        merged_metadata=merged_metadata,
        output_dir=output_dir / "combat",
        ref_batch=ref_batch,
        save_plots=save_plots,
        show_plots=show_plots,
    )
    
    # Check QC4 RSD before/after ComBat
    qc4_pattern = config.get('qc4_pattern', 'QC4')
    log_qc_rsd_simple(merged_data, corrected_data, [qc4_pattern], "ComBat")
    
    # Step 4.5: Post-ComBat batch effect analysis
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP 4.5: Post-ComBat batch effect analysis")
    logger.info(f"{'='*70}")
    
    # Get batch labels for corrected data (same as before ComBat)
    batch_dict_corrected = dict(zip(merged_metadata['original_col'], merged_metadata['batch']))
    batch_vector_corrected = np.array([batch_dict_corrected.get(col, 'unknown') for col in corrected_data.columns])
    
    post_combat_batch_metrics = analyze_batch_effects(
        data=corrected_data,
        batch_labels=batch_vector_corrected,
        output_dir=output_dir / "batch_effect_analysis",
        prefix="post_combat"
    )
    
    # Step 5: RALPS correction (alternative to ComBat)
    run_ralps = config.get('run_ralps', True)
    if run_ralps:
        logger.info(f"\n{'='*70}")
        logger.info(f"STEP 5: RALPS batch correction (alternative method)")
        logger.info(f"{'='*70}")
        
        try:
            # Convert batch_groups to the format RALPS expects
            # batch_groups is already a dict of {batch_name: [sample_cols]}
            ralps_data, ralps_output_dir = run_ralps_correction(
                df=corrected_data,  # Use ComBat-corrected data as input for RALPS
                batch_groups=batch_groups,
                sample_info=sample_info,
                output_dir=output_dir / "ralps",
                qc3_pattern=config.get('qc3_pattern', 'QC3'),
                qc4_pattern=config.get('qc4_pattern', 'QC4'),
                blanco_pattern=config.get('blank_pattern', 'blanco'),
                blaauw_pattern=config.get('blaauw_pattern', 'blaauw'),
            )
            
            # Save RALPS results
            ralps_final = output_dir / "ralps_final"
            ralps_final.mkdir(parents=True, exist_ok=True)
            ralps_data.to_csv(ralps_final / "ralps_corrected_data.csv")
            logger.info(f"RALPS corrected data saved to {ralps_final / 'ralps_corrected_data.csv'}")
            
        except Exception as e:
            logger.error(f"RALPS correction failed: {e}", exc_info=True)
            ralps_data = corrected_data  # Fall back to ComBat data
    else:
        ralps_data = corrected_data
    
    # Step 6: QC analysis
    if run_qc:
        logger.info(f"\n{'='*70}")
        logger.info(f"STEP 6: QC analysis")
        logger.info(f"{'='*70}")
        
        try:
            run_qc_analysis(
                clinical_data=merged_metadata,
                metabolites_after=corrected_data,
                metabolites_before=merged_data,
                batch_column="batch",
                output_path=str(output_dir / "qc_reports"),
            )
        except Exception as e:
            logger.error(f"QC analysis failed: {e}")
    
    # Save final results
    final_output = output_dir / "final"
    final_output.mkdir(parents=True, exist_ok=True)
    corrected_data.to_csv(final_output / "final_corrected_data.csv")
    
    merged_metadata.to_csv(final_output / "final_metadata.csv", index=False)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"PIPELINE COMPLETED")
    logger.info(f"{'='*70}")
    logger.info(f"Final corrected data: {final_output / 'final_corrected_data.csv'}")
    logger.info(f"Final metadata: {final_output / 'final_metadata.csv'}")
    
    return {
        'corrected_data': corrected_data,
        'metadata': merged_metadata,
        'combat_metrics': combat_metrics,
        'batches': all_batches,
    }


def main():
    """Command-line interface for combined batch pipeline."""
    parser = argparse.ArgumentParser(
        description="Process combined batch metabolomics data with ComBat correction"
    )
    
    parser.add_argument(
        "--input",
        default="data/combined_all_batches.csv",
        help="Path to combined CSV file (default: data/combined_all_batches.csv)",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to metadata CSV file for injection order (required for LOESS)",
    )
    parser.add_argument(
        "--output",
        default="output/combined_batch_pipeline",
        help="Output directory (default: output/combined_batch_pipeline)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: combined_batch_pipeline/config/config.yaml)",
    )
    parser.add_argument(
        "--qc-pattern",
        default="expQC",
        help="Pattern to identify QC samples (default: expQC)",
    )
    parser.add_argument(
        "--fallback-qc",
        default="QC3",
        help="Fallback QC pattern (default: QC3)",
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.5,
        help="LOESS fraction parameter (default: 0.5)",
    )
    parser.add_argument(
        "--ref-batch",
        type=int,
        default=None,
        help="Reference batch for ComBat (default: None = automatic)",
    )
    parser.add_argument(
        "--no-qc",
        action="store_true",
        help="Skip QC analysis",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving plots",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots interactively",
    )
    
    args = parser.parse_args()
    
    # Validate metadata file is provided
    if not args.metadata:
        print("ERROR: --metadata argument is required for injection order")
        print("Usage: python combined_batch_pipeline/main.py --input data.csv --metadata metadata.csv")
        sys.exit(1)
    
    try:
        run_full_pipeline(
            input_file=args.input,
            output_dir=args.output,
            metadata_file=args.metadata,
            config_path=args.config,
            qc_pattern=args.qc_pattern,
            fallback_qc_pattern=args.fallback_qc,
            frac=args.frac,
            ref_batch=args.ref_batch,
            run_qc=not args.no_qc,
            save_plots=not args.no_plots,
            show_plots=args.show_plots,
        )
        
        logger.info("\nCombined batch pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
