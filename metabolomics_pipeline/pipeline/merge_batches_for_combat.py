import os
import subprocess
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from collections import defaultdict

def parse_feature(feature_name: str) -> Tuple[str, float, str]:
    """
    Parse a feature name into (base_name, RT, digit).
    Example: "C6H12O6_5.172 1" -> ("C6H12O6", 5.172, "1")
    """
    parts = feature_name.rsplit(' ', 1)
    name_part = parts[0] if len(parts) > 1 else feature_name
    digit = parts[1] if len(parts) > 1 else '1'
    if '_' in name_part:
        base, rt_str = name_part.rsplit('_', 1)
        try:
            rt = float(rt_str)
        except ValueError:
            rt = 0.0
    else:
        base, rt = name_part, 0.0
    return base, rt, digit

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
    Merge two PQ-normalized batches for Combat batch correction.
    Steps:
    1. Loads and merges features with matching base names and RTs.
    2. Removes all sample columns containing 'expQC' (case-insensitive).
    3. Saves batch1-only features to a separate CSV.
    4. Removes ALL features present in only one batch (batch1-only or batch2-only).
    5. Ensures sample names match between data and batch info.
    Returns:
        Tuple of (merged_data, merged_batch) DataFrames.
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

    # Check if median normalization is needed (intrabatch scale differences)
    print("\n=== Batch 1 Sample Medians ===")
    batch1_medians = df1.median(axis=0)
    print(batch1_medians.describe())
    print(f"CV of medians: {batch1_medians.std() / batch1_medians.mean() * 100:.2f}%")

    print("\n=== Batch 2 Sample Medians ===")
    batch2_medians = df2.median(axis=0)
    print(batch2_medians.describe())
    print(f"CV of medians: {batch2_medians.std() / batch2_medians.mean() * 100:.2f}%")

    # Clean batch metadata
    batch1['batch'] = 1
    batch2['batch'] = 2
    for col in ['batch_type']:
        batch1 = batch1.drop(columns=[col], errors='ignore')
        batch2 = batch2.drop(columns=[col], errors='ignore')

    # --- Merge features ---
    feature_groups = defaultdict(list)

    # Group features by base name + RT
    for feature in df1.index:
        base, rt, _ = parse_feature(feature)
        feature_groups[base].append(('batch1', feature, rt))
    for feature in df2.index:
        base, rt, _ = parse_feature(feature)
        feature_groups[base].append(('batch2', feature, rt))

    # Assign match keys (use batch1 feature names)
    feature_to_match_key = {}
    for base, features in feature_groups.items():
        features_sorted = sorted(features, key=lambda x: x[2])  # Sort by RT
        current_group = [features_sorted[0]]

        for i in range(1, len(features_sorted)):
            prev_batch, prev_feature, prev_rt = current_group[-1]
            curr_batch, curr_feature, curr_rt = features_sorted[i]
            if abs(curr_rt - prev_rt) <= rt_threshold:
                current_group.append((curr_batch, curr_feature, curr_rt))
            else:
                # Assign match key to the current group
                batch1_feature = next((f for b, f, _ in current_group if b == 'batch1'), current_group[0][1])
                for _, f, _ in current_group:
                    feature_to_match_key[f] = batch1_feature
                current_group = [(curr_batch, curr_feature, curr_rt)]

        # Assign match key to the last group
        batch1_feature = next((f for b, f, _ in current_group if b == 'batch1'), current_group[0][1])
        for _, f, _ in current_group:
            feature_to_match_key[f] = batch1_feature

    # Rename features
    df1_renamed = df1.rename(index=lambda x: feature_to_match_key.get(x, x))
    df2_renamed = df2.rename(index=lambda x: feature_to_match_key.get(x, x))

    # Merge duplicates within each batch
    df1_merged = df1_renamed.groupby(level=0).mean()
    df2_merged = df2_renamed.groupby(level=0).mean()

    # Concatenate (keep ALL features)
    merged_data = pd.concat([df1_merged, df2_merged], axis=1, join='outer')
    merged_batch = pd.concat([batch1, batch2], ignore_index=True)

    # --- Remove expQC samples from data and batch info ---
    # Find all columns containing 'expqc' (case-insensitive)
    expqc_cols = [col for col in merged_data.columns if 'expqc' in col.lower()]
    if expqc_cols:
        print(f"✓ Removing {len(expqc_cols)} expQC columns from data: {expqc_cols}")
        merged_data = merged_data.drop(columns=expqc_cols)
        # Remove corresponding rows from merged_batch
        merged_batch = merged_batch[~merged_batch['sample_id'].str.lower().str.contains('expqc')]
        print(f"✓ Removed expQC samples from batch metadata")

    # --- Identify and handle batch-specific features ---
    # Use filtered batch metadata to get sample IDs
    batch1_samples = set(merged_batch[merged_batch['batch'] == 1]['sample_id'])
    batch2_samples = set(merged_batch[merged_batch['batch'] == 2]['sample_id'])
    batch1_cols = [col for col in merged_data.columns if col in batch1_samples]
    batch2_cols = [col for col in merged_data.columns if col in batch2_samples]

    # Batch1-only features: rows with values in batch1 cols AND all NaN in batch2 cols
    batch1_only_mask = merged_data[batch1_cols].notna().any(axis=1) & merged_data[batch2_cols].isna().all(axis=1)
    if batch1_only_mask.any():
        # Save ONLY the batch1-only features (rows) with ALL columns (samples)
        batch1_only_features = merged_data.loc[batch1_only_mask]
        batch1_only_path = combat_input_dir / "current_batch_only_features.csv"
        batch1_only_features.to_csv(batch1_only_path)
        print(f"✓ Batch1-only features (n={batch1_only_mask.sum()}) saved to {batch1_only_path}")

    # Batch2-only features: rows with values in batch2 cols AND all NaN in batch1 cols
    batch2_only_mask = merged_data[batch2_cols].notna().any(axis=1) & merged_data[batch1_cols].isna().all(axis=1)

    # Remove ALL features present in only one batch (both batch1-only and batch2-only)
    batch_specific_mask = batch1_only_mask | batch2_only_mask
    if batch_specific_mask.any():
        merged_data = merged_data[~batch_specific_mask]
        print(f"✓ Removed {batch_specific_mask.sum()} features present in only one batch")

    # --- Ensure sample names match between data and batch info ---
    data_samples = set(merged_data.columns)
    batch_samples = set(merged_batch['sample_id'])
    common_samples = list(data_samples & batch_samples)

    if not common_samples:
        raise ValueError(f"No common samples between data ({data_samples}) and batch info ({batch_samples})")

    merged_data = merged_data[common_samples]
    merged_batch = merged_batch[merged_batch['sample_id'].isin(common_samples)]

    # --- Save outputs for COMBAT ---
    merged_data_path = combat_input_dir / "merged_data_for_combat.csv"
    merged_batch_path = combat_input_dir / "merged_batch_for_combat.csv"
    merged_data.to_csv(merged_data_path)
    merged_batch.to_csv(merged_batch_path, index=False)

    print(f"✓ Merged data saved to {merged_data_path}")
    print(f"✓ Batch metadata saved to {merged_batch_path}")
    return merged_data, merged_batch