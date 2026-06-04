#!/usr/bin/env python3
"""
Script to perform multi-batch ComBat correction on multiple PQN-normalized files,
followed by quality control (QC) using inmoose.

This script:
1. Gathers all pqn_normalized.csv files from specified batches
2. Merges them into a single dataset
3. Performs ComBat batch correction with QC checks and visualizations
4. Runs QC on the corrected data (optional)

Usage:
    python multi_batch_combat.py --batches batch1 batch2 batch3 --mode NEG
    python multi_batch_combat.py --batches batch1 batch2 --mode POS
    python multi_batch_combat.py --batches batch1 batch2 --mode NEG --data-dir /path/to/your/data
    python multi_batch_combat.py --batches batch1 batch2 --mode NEG --no-qc
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import traceback

# Import pipeline functions
from pipeline.merge_batches_for_combat import parse_feature
from pipeline.combat_utils import run_combat_and_visualize

# QC-specific imports (handle gracefully if not installed)
try:
    from inmoose.cohort_qc.cohort_metric import CohortMetric
    from inmoose.cohort_qc.qc_report import QCReport
    INMOOSE_AVAILABLE = True
except ImportError:
    INMOOSE_AVAILABLE = False
    logging.warning("inmoose.cohort_qc not available. QC report generation will be skipped.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Helpers for QC
# ------------------------------------------------------------------------------

def _safe_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    NaN-safe imputation for metabolomics PCA compatibility.
    """
    df = df.copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    min_pos = df[df > 0].min().min()
    fill_value = (min_pos / 2) if pd.notna(min_pos) else 1e-10
    df = df.fillna(fill_value)
    df = df.replace([np.inf, -np.inf], fill_value)
    return df

def run_final_qc(
    clinical_data: pd.DataFrame,
    metabolites_after: pd.DataFrame,
    metabolites_before: Optional[pd.DataFrame] = None,
    batch_column: str = "batch",
    output_path: str = "reports",
) -> None:
    """
    Run QC on the final merged and corrected data.
    """
    clinical_data = clinical_data.copy()
    metabolites_after = metabolites_after.copy()

    # Skip QC if metabolites_after is empty or None
    if metabolites_after.empty or metabolites_after is None:
        logger.warning("Skipping QC: metabolites_after is empty or None.")
        return

    # 1. Fix sample IDs
    if "sample_id" not in clinical_data.columns:
        raise ValueError("clinical_data must contain 'sample_id' column.")

    clinical_data["sample_id"] = clinical_data["sample_id"].astype(str).str.strip()
    clinical_data = clinical_data.set_index("sample_id")

    metabolites_after.columns = metabolites_after.columns.astype(str).str.strip()

    if metabolites_before is not None:
        metabolites_before = metabolites_before.copy()
        metabolites_before.columns = metabolites_before.columns.astype(str).str.strip()

    # 2. Batch column cleanup
    if batch_column not in clinical_data.columns:
        raise ValueError(f"Batch column '{batch_column}' not found in clinical_data.")

    clinical_data[batch_column] = pd.to_numeric(
        clinical_data[batch_column],
        errors="raise"
    )

    # 3. Align samples
    common_samples = metabolites_after.columns.intersection(clinical_data.index)

    logger.info(f"Found {len(common_samples)} common samples for QC")

    if len(common_samples) == 0:
        raise ValueError("No overlapping samples between clinical and metabolomics data.")

    clinical_data = clinical_data.loc[common_samples]
    metabolites_after = metabolites_after.loc[:, common_samples]

    if metabolites_before is not None:
        metabolites_before = metabolites_before.loc[:, common_samples]

    # 4. NaN + Inf cleanup
    metabolites_after = _safe_impute(metabolites_after)

    if metabolites_before is not None:
        metabolites_before = _safe_impute(metabolites_before)

    # 5. Diagnostics
    logger.info(f"Sample count for QC: {clinical_data.shape[0]}")
    logger.info(f"NaNs after imputation: {metabolites_after.isna().sum().sum()}")

    # 6. Run inmoose QC (if available)
    if not INMOOSE_AVAILABLE:
        logger.warning("Skipping QC report generation (inmoose not available).")
        return

    try:
        # Initialize CohortMetric with only the required arguments
        cohort_qc = CohortMetric(
            clinical_df=clinical_data,
            batch_column=batch_column,
            data_expression_df=metabolites_after,
            data_expression_df_before=metabolites_before if metabolites_before is not None else None,
        )
        cohort_qc.process()

        qc_report = QCReport(cohort_qc)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        qc_report.save_report(output_path=str(Path(output_path) / "qc_report.html"))
        logger.info(f"✓ QC report saved to {output_path}")
    except Exception as e:
        logger.error(f"QC failed: {e}")
        logger.error(traceback.format_exc())
        raise

# ------------------------------------------------------------------------------
# Main Script Functions
# ------------------------------------------------------------------------------

def find_pqn_files(batch_folders: List[str], mode: str, data_dir: str = "data") -> List[Tuple[str, Path, Path]]:
    """
    Find all pqn_normalized.csv and batch_data.csv files for the given batches and mode.

    Args:
        batch_folders: List of batch folder names
        mode: Ion mode (NEG or POS)
        data_dir: Base data directory

    Returns:
        List of tuples: (batch_label, pqn_normalized_path, batch_data_path)
    """
    results = []
    for batch_folder in batch_folders:
        pqn_path = Path(f"{data_dir}/{batch_folder}/output/{mode}/pqn_normalized.csv")
        batch_data_path = Path(f"{data_dir}/{batch_folder}/output/{mode}/batch_data.csv")

        if pqn_path.exists() and batch_data_path.exists():
            results.append((batch_folder, pqn_path, batch_data_path))
            logger.info(f"Found PQN files for batch {batch_folder}: {pqn_path}")
        else:
            logger.warning(f"PQN files not found for batch {batch_folder} mode {mode}")
            logger.warning(f"  Expected: {pqn_path}")
            logger.warning(f"  Expected: {batch_data_path}")

    return results

def merge_multiple_batches(
        batch_files: List[Tuple[str, Path, Path]],
        output_dir: Path,
        mode: str,
        rt_threshold: float = 0.02,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge multiple PQN-normalized batches into a single dataset for ComBat.

    Args:
        batch_files: List of tuples (batch_label, pqn_normalized_path, batch_data_path)
        output_dir: Directory to save merged files
        mode: Ion mode (for naming)
        rt_threshold: RT threshold for feature matching

    Returns:
        Tuple of (merged_data, merged_batch) DataFrames
    """
    if len(batch_files) < 2:
        raise ValueError(f"Need at least 2 batches to merge. Got {len(batch_files)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all data
    data_frames = []
    batch_dfs = []
    batch_labels = []

    for batch_label, pqn_path, batch_data_path in batch_files:
        df = pd.read_csv(pqn_path, index_col='Feature')
        df = df.apply(pd.to_numeric, errors='coerce')
        batch_df = pd.read_csv(batch_data_path)

        data_frames.append(df)
        batch_dfs.append(batch_df)
        batch_labels.append(batch_label)

    # Print sample medians for each batch
    for i, (batch_label, df) in enumerate(zip(batch_labels, data_frames)):
        batch_medians = df.median(axis=0)
        logger.info(f"\n=== Batch {batch_label} ({i + 1}) Sample Medians ===")
        logger.info(f"Median of medians: {batch_medians.median():.2f}")
        logger.info(f"CV of medians: {batch_medians.std() / batch_medians.mean() * 100:.2f}%")

    # Clean and label batch metadata
    all_batch_dfs = []
    for i, (batch_label, batch_df) in enumerate(zip(batch_labels, batch_dfs)):
        batch_df_clean = batch_df.copy()
        batch_df_clean['batch'] = i + 1  # 1-indexed
        batch_df_clean['batch_label'] = batch_label
        # Remove unnecessary columns
        for col in ['batch_type']:
            batch_df_clean = batch_df_clean.drop(columns=[col], errors='ignore')
        all_batch_dfs.append(batch_df_clean)

    # Merge all batch metadata
    merged_batch = pd.concat(all_batch_dfs, ignore_index=True)

    # Group features by base name + RT across all batches
    from collections import defaultdict
    feature_groups = defaultdict(list)

    for batch_idx, df in enumerate(data_frames):
        batch_label = batch_labels[batch_idx]
        for feature in df.index:
            base, rt, _ = parse_feature(feature)
            feature_groups[base].append((batch_label, feature, rt, batch_idx))

    # Assign match keys (use first batch's feature names as reference)
    feature_to_match_key = {}
    for base, features in feature_groups.items():
        # Sort by RT
        features_sorted = sorted(features, key=lambda x: x[2])
        current_group = [features_sorted[0]]

        for i in range(1, len(features_sorted)):
            prev_batch, prev_feature, prev_rt, prev_idx = current_group[-1]
            curr_batch, curr_feature, curr_rt, curr_idx = features_sorted[i]
            if abs(curr_rt - prev_rt) <= rt_threshold:
                current_group.append((curr_batch, curr_feature, curr_rt, curr_idx))
            else:
                # Assign match key to the current group (use first batch's feature if available)
                first_batch_feature = next(
                    (f for b, f, _, idx in current_group if idx == 0),
                    current_group[0][1]
                )
                for _, f, _, _ in current_group:
                    feature_to_match_key[f] = first_batch_feature
                current_group = [(curr_batch, curr_feature, curr_rt, curr_idx)]

        # Assign match key to the last group
        first_batch_feature = next(
            (f for b, f, _, idx in current_group if idx == 0),
            current_group[0][1]
        )
        for _, f, _, _ in current_group:
            feature_to_match_key[f] = first_batch_feature

    # Rename features in each batch
    renamed_dfs = []
    for df in data_frames:
        df_renamed = df.rename(index=lambda x: feature_to_match_key.get(x, x))
        renamed_dfs.append(df_renamed)

    # Merge duplicates within each batch
    merged_dfs = []
    for df in renamed_dfs:
        df_merged = df.groupby(level=0).mean()
        merged_dfs.append(df_merged)

    # Concatenate all batches (keep ALL features)
    merged_data = pd.concat(merged_dfs, axis=1, join='outer')

    # Remove expQC samples from data and batch info
    expqc_cols = [col for col in merged_data.columns if 'expqc' in col.lower()]
    if expqc_cols:
        logger.info(f"Removing {len(expqc_cols)} expQC columns from data")
        merged_data = merged_data.drop(columns=expqc_cols)
        merged_batch = merged_batch[~merged_batch['sample_id'].str.lower().str.contains('expqc')]
        logger.info(f"Removed expQC samples from batch metadata")

    # Identify batch-specific features
    batch_sample_sets = []
    for i in range(len(batch_labels)):
        batch_samples = set(merged_batch[merged_batch['batch'] == i + 1]['sample_id'])
        batch_sample_sets.append(batch_samples)

    # Find features present in only one batch
    batch_specific_mask = pd.Series(False, index=merged_data.index)

    for batch_idx in range(len(batch_labels)):
        batch_samples = batch_sample_sets[batch_idx]
        other_samples = set()
        for j in range(len(batch_labels)):
            if j != batch_idx:
                other_samples.update(batch_sample_sets[j])

        batch_cols = [col for col in merged_data.columns if col in batch_samples]
        other_cols = [col for col in merged_data.columns if col in other_samples]

        # Features present only in this batch
        if other_cols:
            batch_only_mask = merged_data[batch_cols].notna().any(axis=1) & merged_data[other_cols].isna().all(axis=1)
            batch_specific_mask = batch_specific_mask | batch_only_mask

    # Save batch-specific features
    if batch_specific_mask.any():
        for batch_idx in range(len(batch_labels)):
            batch_samples = batch_sample_sets[batch_idx]
            other_samples = set()
            for j in range(len(batch_labels)):
                if j != batch_idx:
                    other_samples.update(batch_sample_sets[j])

            batch_cols = [col for col in merged_data.columns if col in batch_samples]
            other_cols = [col for col in merged_data.columns if col in other_samples]

            batch_only_mask = merged_data[batch_cols].notna().any(axis=1) & merged_data[other_cols].isna().all(axis=1)
            if batch_only_mask.any():
                batch_only_features = merged_data.loc[batch_only_mask]
                batch_label = batch_labels[batch_idx]
                batch_only_path = output_dir / f"{batch_label}_only_features.csv"
                batch_only_features.to_csv(batch_only_path)
                logger.info(f"Saved {batch_label}-only features (n={batch_only_mask.sum()}) to {batch_only_path}")

    # Remove ALL features present in only one batch
    if batch_specific_mask.any():
        merged_data = merged_data[~batch_specific_mask]
        logger.info(f"Removed {batch_specific_mask.sum()} features present in only one batch")

    # Ensure sample names match between data and batch info
    data_samples = set(merged_data.columns)
    batch_samples = set(merged_batch['sample_id'])
    common_samples = list(data_samples & batch_samples)

    if not common_samples:
        raise ValueError(f"No common samples between data ({len(data_samples)}) and batch info ({len(batch_samples)})")

    merged_data = merged_data[common_samples]
    merged_batch = merged_batch[merged_batch['sample_id'].isin(common_samples)]

    # Save outputs for ComBat
    merged_data_path = output_dir / "merged_data_for_combat.csv"
    merged_batch_path = output_dir / "merged_batch_for_combat.csv"
    merged_data.to_csv(merged_data_path)
    merged_batch.to_csv(merged_batch_path, index=False)

    logger.info(f"Merged data saved to {merged_data_path}")
    logger.info(f"Merged batch metadata saved to {merged_batch_path}")
    logger.info(f"Final merged data shape: {merged_data.shape} (features x samples)")
    logger.info(f"Final merged batch shape: {merged_batch.shape} (samples x metadata)")

    return merged_data, merged_batch

def run_multi_batch_combat(
        batch_folders: List[str],
        mode: str,
        output_dir: Path = None,
        data_dir: str = "data",
        rt_threshold: float = 0.02,
        show_plots: bool = False,
        save_plots: bool = True,
        run_qc: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Run multi-batch ComBat correction on PQN-normalized files.

    Args:
        batch_folders: List of batch folder names to process
        mode: Ion mode (NEG or POS)
        output_dir: Output directory (default: data/multi_batch_combat/{mode}/)
        data_dir: Base data directory
        rt_threshold: RT threshold for feature matching
        show_plots: Whether to show plots interactively
        save_plots: Whether to save plots to disk
        run_qc: Whether to run QC after ComBat (default: True)

    Returns:
        Tuple of (combat_corrected_data, metrics)
    """
    if output_dir is None:
        output_dir = Path(f"{data_dir}/multi_batch_combat/{mode}/")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Find all PQN files
    logger.info(f"Searching for PQN-normalized files in batches: {batch_folders}")
    batch_files = find_pqn_files(batch_folders, mode, data_dir)

    if len(batch_files) < 2:
        raise ValueError(f"Need at least 2 batches with PQN files. Found {len(batch_files)}")

    logger.info(f"Found {len(batch_files)} batches with PQN files")

    # Step 2: Merge all batches
    logger.info(f"Merging {len(batch_files)} batches...")
    combat_input_dir = output_dir / "combat_input"
    combat_input_dir.mkdir(parents=True, exist_ok=True)

    merged_data, merged_batch = merge_multiple_batches(
        batch_files=batch_files,
        output_dir=combat_input_dir,
        mode=mode,
        rt_threshold=rt_threshold,
    )

    # Step 3: Run ComBat
    logger.info(f"Running ComBat batch correction...")
    combat_output_dir = output_dir / "combat_corrected"
    combat_output_dir.mkdir(parents=True, exist_ok=True)

    combat_corrected_df, combat_metrics = run_combat_and_visualize(
        merged_data_path=str(combat_input_dir / "merged_data_for_combat.csv"),
        merged_batch_path=str(combat_input_dir / "merged_batch_for_combat.csv"),
        output_dir=str(combat_output_dir),
        show_plots=show_plots,
        save_plots=save_plots,
    )

    logger.info(f"ComBat correction completed!")
    logger.info(f"ComBat metrics: {combat_metrics}")
    logger.info(f"Output saved to {combat_output_dir}/")

    # Step 4: Run QC if enabled
    if run_qc:
        logger.info("Running QC on ComBat-corrected data...")
        qc_output_dir = output_dir / "qc"
        try:
            # Pass merged_data as metabolites_before
            run_final_qc(
                clinical_data=merged_batch,
                metabolites_after=combat_corrected_df,
                metabolites_before=merged_data,  # <-- Pass merged_data as "before" data
                batch_column="batch",
                output_path=str(qc_output_dir),
            )
        except Exception as e:
            logger.error(f"QC failed: {e}")

    return combat_corrected_df, combat_metrics

def main():
    """Command-line interface for multi-batch ComBat correction."""
    parser = argparse.ArgumentParser(
        description="Run multi-batch ComBat correction on PQN-normalized files"
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        required=True,
        help="List of batch folder names to process",
    )
    parser.add_argument(
        "--mode",
        choices=["NEG", "POS"],
        required=True,
        help="Ion mode (NEG or POS)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base data directory (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: data/multi_batch_combat/{mode}/)",
    )
    parser.add_argument(
        "--rt-threshold",
        type=float,
        default=0.02,
        help="RT threshold for feature matching (default: 0.02)",
    )
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
    parser.add_argument(
        "--no-qc",
        action="store_true",
        help="Disable QC report generation",
    )

    args = parser.parse_args()

    try:
        run_multi_batch_combat(
            batch_folders=args.batches,
            mode=args.mode,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            rt_threshold=args.rt_threshold,
            show_plots=args.show_plots,
            save_plots=not args.no_plots,
            run_qc=not args.no_qc,
        )
        logger.info("Multi-batch ComBat correction completed successfully!")
    except Exception as e:
        logger.error(f"Multi-batch ComBat failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()