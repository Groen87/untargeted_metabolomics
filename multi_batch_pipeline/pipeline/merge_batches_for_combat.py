"""
Feature merging module for ComBat batch correction.

This module provides functionality to merge two PQN-normalized batches for
ComBat batch correction, with RT-based feature matching.

Key Features:
- Parses feature names to extract base name, RT, and digit
- Groups features by base name + RT across batches
- Matches features using RT threshold
- Handles batch-specific features
- Removes expQC samples
- Ensures sample alignment between data and metadata
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import pandas as pd
from collections import defaultdict


def parse_feature(feature_name: str) -> Tuple[str, float, str]:
    """
    Parse a feature name into its components.
    
    The feature name format is typically: "base_name_RT digit"
    For example: "C6H12O6_5.172 1" -> ("C6H12O6", 5.172, "1")
    
    Args:
        feature_name: The feature name string to parse
        
    Returns:
        Tuple of (base_name, RT, digit) where:
        - base_name: The base chemical name (e.g., "C6H12O6")
        - RT: Retention time as float (e.g., 5.172)
        - digit: The feature digit suffix (e.g., "1")
        
    Note:
        If parsing fails, RT defaults to 0.0 and digit defaults to "1".
    """
    # Split off the digit suffix (last space-separated part)
    parts = feature_name.rsplit(' ', 1)
    name_part = parts[0] if len(parts) > 1 else feature_name
    digit = parts[1] if len(parts) > 1 else '1'
    
    # Split off the RT (last underscore-separated part of name_part)
    if '_' in name_part:
        base, rt_str = name_part.rsplit('_', 1)
        try:
            rt = float(rt_str)
        except ValueError:
            rt = 0.0
    else:
        base, rt = name_part, 0.0
    
    return base, rt, digit


def group_features_by_base(
    features: List[str],
    batch_label: str,
) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Group features by their base name.
    
    Args:
        features: List of feature name strings
        batch_label: Label for the batch these features belong to
        
    Returns:
        Dictionary mapping base names to lists of (batch_label, feature_name, RT) tuples
    """
    feature_groups = defaultdict(list)
    
    for feature in features:
        base, rt, _ = parse_feature(feature)
        feature_groups[base].append((batch_label, feature, rt))
    
    return feature_groups


def match_features_across_batches(
    feature_groups: Dict[str, List[Tuple[str, str, float]]],
    rt_threshold: float = 0.02,
    prefer_batch: str = 'batch1',
) -> Dict[str, str]:
    """
    Match features across batches using RT-based grouping.
    
    Features with the same base name and RT within the threshold are considered
    the same feature and are assigned a common match key.
    
    Args:
        feature_groups: Dictionary from group_features_by_base()
        rt_threshold: RT difference threshold for matching (in minutes, default: 0.02)
        prefer_batch: Which batch's feature names to prefer as match keys (default: 'batch1')
        
    Returns:
        Dictionary mapping original feature names to match keys
    """
    feature_to_match_key = {}
    
    for base, features in feature_groups.items():
        # Sort features by RT
        features_sorted = sorted(features, key=lambda x: x[2])
        current_group = [features_sorted[0]]
        
        for i in range(1, len(features_sorted)):
            prev_batch, prev_feature, prev_rt = current_group[-1]
            curr_batch, curr_feature, curr_rt = features_sorted[i]
            
            # Check if features are close enough in RT to be considered the same
            if abs(curr_rt - prev_rt) <= rt_threshold:
                current_group.append((curr_batch, curr_feature, curr_rt))
            else:
                # Assign match key to the current group
                # Prefer the specified batch's feature name
                preferred_feature = next(
                    (f for b, f, _ in current_group if b == prefer_batch),
                    current_group[0][1]
                )
                for _, f, _ in current_group:
                    feature_to_match_key[f] = preferred_feature
                current_group = [(curr_batch, curr_feature, curr_rt)]
        
        # Assign match key to the last group
        preferred_feature = next(
            (f for b, f, _ in current_group if b == prefer_batch),
            current_group[0][1]
        )
        for _, f, _ in current_group:
            feature_to_match_key[f] = preferred_feature
    
    return feature_to_match_key


def remove_expqc_samples(
    merged_data: pd.DataFrame,
    merged_batch: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove expQC samples from data and batch metadata.
    
    Args:
        merged_data: DataFrame with feature data (features x samples)
        merged_batch: DataFrame with sample metadata
        
    Returns:
        Tuple of (merged_data, merged_batch) with expQC samples removed
    """
    # Find columns containing 'expqc' (case-insensitive)
    expqc_cols = [col for col in merged_data.columns if 'expqc' in col.lower()]
    
    if not expqc_cols:
        return merged_data, merged_batch
    
    # Remove from data
    merged_data = merged_data.drop(columns=expqc_cols)
    
    # Remove from batch metadata
    merged_batch = merged_batch[~merged_batch['sample_id'].str.lower().str.contains('expqc')]
    
    return merged_data, merged_batch


def identify_batch_specific_features(
    merged_data: pd.DataFrame,
    merged_batch: pd.DataFrame,
    batch1_label: int = 1,
    batch2_label: int = 2,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Identify features that are present in only one batch.
    
    Args:
        merged_data: DataFrame with feature data (features x samples)
        merged_batch: DataFrame with sample metadata
        batch1_label: Batch label for first batch (default: 1)
        batch2_label: Batch label for second batch (default: 2)
        
    Returns:
        Tuple of:
        - batch1_only_features: DataFrame with features only in batch 1
        - batch_specific_mask: Boolean Series indicating batch-specific features
    """
    # Get sample sets for each batch
    batch1_samples = set(merged_batch[merged_batch['batch'] == batch1_label]['sample_id'])
    batch2_samples = set(merged_batch[merged_batch['batch'] == batch2_label]['sample_id'])
    
    batch1_cols = [col for col in merged_data.columns if col in batch1_samples]
    batch2_cols = [col for col in merged_data.columns if col in batch2_samples]
    
    # Batch1-only features: present in batch1 cols, absent from batch2 cols
    batch1_only_mask = merged_data[batch1_cols].notna().any(axis=1) & merged_data[batch2_cols].isna().all(axis=1)
    
    # Batch2-only features: present in batch2 cols, absent from batch1 cols
    batch2_only_mask = merged_data[batch2_cols].notna().any(axis=1) & merged_data[batch1_cols].isna().all(axis=1)
    
    # Combined mask for all batch-specific features
    batch_specific_mask = batch1_only_mask | batch2_only_mask
    
    # Extract batch1-only features
    batch1_only_features = None
    if batch1_only_mask.any():
        batch1_only_features = merged_data.loc[batch1_only_mask]
    
    return batch1_only_features, batch_specific_mask


def align_samples(
    merged_data: pd.DataFrame,
    merged_batch: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure sample names match between data and batch info.
    
    Args:
        merged_data: DataFrame with feature data (features x samples)
        merged_batch: DataFrame with sample metadata
        
    Returns:
        Tuple of aligned (merged_data, merged_batch)
        
    Raises:
        ValueError: If no common samples exist
    """
    data_samples = set(merged_data.columns)
    batch_samples = set(merged_batch['sample_id'])
    common_samples = list(data_samples & batch_samples)
    
    if not common_samples:
        raise ValueError(
            f"No common samples between data ({len(data_samples)}) "
            f"and batch info ({len(batch_samples)})"
        )
    
    merged_data = merged_data[common_samples]
    merged_batch = merged_batch[merged_batch['sample_id'].isin(common_samples)]
    
    return merged_data, merged_batch


def merge_batches_for_combat(
    drift_corrected_file_batch1: str,
    drift_corrected_file_batch2: str,
    batch_file_batch1: str,
    batch_file_batch2: str,
    combat_input_dir: str = "output",
    combat_output_dir: Optional[str] = None,
    batch1_label: str = "current",
    batch2_label: str = "reference",
    rt_threshold: float = 0.02,
    combat_script_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge two PQ-normalized batches for ComBat batch correction.
    
    This function performs the following steps:
    1. Loads data and metadata from both batches
    2. Prints diagnostic information about sample medians
    3. Groups features by base name + RT across batches
    4. Matches features using RT threshold
    5. Renames features to create consistent identifiers
    6. Merges duplicates within each batch
    7. Concatenates batches
    8. Removes expQC samples
    9. Identifies and saves batch-specific features
    10. Removes features present in only one batch
    11. Aligns samples between data and metadata
    12. Saves merged data for ComBat
    
    Args:
        drift_corrected_file_batch1: Path to PQN-normalized CSV for batch 1
        drift_corrected_file_batch2: Path to PQN-normalized CSV for batch 2
        batch_file_batch1: Path to batch metadata CSV for batch 1
        batch_file_batch2: Path to batch metadata CSV for batch 2
        combat_input_dir: Directory to save merged files (default: "output")
        combat_output_dir: Directory for ComBat output (optional)
        batch1_label: Label for batch 1 (default: "current")
        batch2_label: Label for batch 2 (default: "reference")
        rt_threshold: RT threshold for feature matching in minutes (default: 0.02)
        combat_script_path: Path to ComBat script (optional, not used)
        
    Returns:
        Tuple of (merged_data, merged_batch) DataFrames where:
        - merged_data: DataFrame with all features (rows) and samples (columns)
        - merged_batch: DataFrame with sample metadata
        
    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If no common samples exist after filtering
    """
    # --- Setup directories ---
    combat_input_dir = Path(combat_input_dir).resolve()
    combat_input_dir.mkdir(parents=True, exist_ok=True)
    
    if combat_output_dir is None:
        combat_output_dir = combat_input_dir / "combat_corrected"
    else:
        combat_output_dir = Path(combat_output_dir).resolve()
    combat_output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Load data ---
    df1 = pd.read_csv(drift_corrected_file_batch1, index_col='Feature')
    df2 = pd.read_csv(drift_corrected_file_batch2, index_col='Feature')
    batch1 = pd.read_csv(batch_file_batch1)
    batch2 = pd.read_csv(batch_file_batch2)
    
    # --- Print diagnostic information ---
    print("\n=== Batch 1 Sample Medians ===")
    batch1_medians = df1.median(axis=0)
    print(batch1_medians.describe())
    print(f"CV of medians: {batch1_medians.std() / batch1_medians.mean() * 100:.2f}%")
    
    print("\n=== Batch 2 Sample Medians ===")
    batch2_medians = df2.median(axis=0)
    print(batch2_medians.describe())
    print(f"CV of medians: {batch2_medians.std() / batch2_medians.mean() * 100:.2f}%")
    
    # --- Clean batch metadata ---
    batch1['batch'] = 1
    batch2['batch'] = 2
    for col in ['batch_type']:
        batch1 = batch1.drop(columns=[col], errors='ignore')
        batch2 = batch2.drop(columns=[col], errors='ignore')
    
    # --- Merge features ---
    # Group features by base name from both batches
    feature_groups = defaultdict(list)
    
    for feature in df1.index:
        base, rt, _ = parse_feature(feature)
        feature_groups[base].append(('batch1', feature, rt))
    for feature in df2.index:
        base, rt, _ = parse_feature(feature)
        feature_groups[base].append(('batch2', feature, rt))
    
    # Match features using RT threshold
    feature_to_match_key = match_features_across_batches(
        feature_groups, rt_threshold, prefer_batch='batch1'
    )
    
    # Rename features
    df1_renamed = df1.rename(index=lambda x: feature_to_match_key.get(x, x))
    df2_renamed = df2.rename(index=lambda x: feature_to_match_key.get(x, x))
    
    # Merge duplicates within each batch (by taking mean)
    df1_merged = df1_renamed.groupby(level=0).mean()
    df2_merged = df2_renamed.groupby(level=0).mean()
    
    # Concatenate (keep ALL features)
    merged_data = pd.concat([df1_merged, df2_merged], axis=1, join='outer')
    merged_batch = pd.concat([batch1, batch2], ignore_index=True)
    
    # --- Remove expQC samples ---
    merged_data, merged_batch = remove_expqc_samples(merged_data, merged_batch)
    print(f"✓ Removed expQC samples from data and metadata")
    
    # --- Identify and handle batch-specific features ---
    batch1_only_features, batch_specific_mask = identify_batch_specific_features(
        merged_data, merged_batch, batch1_label=1, batch2_label=2
    )
    
    # Save batch1-only features
    if batch1_only_features is not None and len(batch1_only_features) > 0:
        batch1_only_path = combat_input_dir / "current_batch_only_features.csv"
        batch1_only_features.to_csv(batch1_only_path)
        print(f"✓ Batch1-only features (n={len(batch1_only_features)}) saved to {batch1_only_path}")
    
    # Remove ALL features present in only one batch
    if batch_specific_mask.any():
        merged_data = merged_data[~batch_specific_mask]
        print(f"✓ Removed {batch_specific_mask.sum()} features present in only one batch")
    
    # --- Align samples ---
    merged_data, merged_batch = align_samples(merged_data, merged_batch)
    
    # --- Save outputs for ComBat ---
    merged_data_path = combat_input_dir / "merged_data_for_combat.csv"
    merged_batch_path = combat_input_dir / "merged_batch_for_combat.csv"
    merged_data.to_csv(merged_data_path)
    merged_batch.to_csv(merged_batch_path, index=False)
    
    print(f"✓ Merged data saved to {merged_data_path}")
    print(f"✓ Batch metadata saved to {merged_batch_path}")
    
    return merged_data, merged_batch
