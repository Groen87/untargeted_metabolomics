import os
import subprocess
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.interpolate import UnivariateSpline, interp1d

def parse_feature(feature_name: str) -> str:
    """
    Parse a feature name to get the base name (without RT and digit).
    Example: "C6H12O6 1" -> "C6H12O6"
    Now just returns the base name since RT is in a separate column.
    """
    parts = feature_name.rsplit(' ', 1)
    name_part = parts[0] if len(parts) > 1 else feature_name
    return name_part

def merge_batches_for_combat(
    drift_corrected_file_batch1: str,
    drift_corrected_file_batch2: str,
    batch_file_batch1: str,
    batch_file_batch2: str,
    combat_input_dir: str = "output",
    combat_output_dir: Optional[str] = None,
    batch1_label: str = "current",
    batch2_label: str = "reference",
    rt_threshold: Optional[float] = None,
    combat_script_path: Optional[Path] = None,
    reference_batch_label: Optional[str] = None,
    use_spline: Optional[bool] = None,
    config_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge two PQ-normalized batches for Combat batch correction.
    
    Uses RT warping (spline or LOESS) to align RTs to a reference batch (MZ26_10_QC3),
    then matches features by name + warped RT (within threshold).
    
    Steps:
    1. Load data and RT columns from both batches.
    2. Identify reference batch (MZ26_10_QC3) and warp other batch RTs to match.
    3. Match features by name + warped RT (within rt_threshold).
    4. For features not in reference, try to match across other batches.
    5. Keep unique features that have no matches.
    6. Remove expQC samples and ensure sample names match.
    
    Returns:
        Tuple of (merged_data, merged_batch) DataFrames.
    
    Args:
        drift_corrected_file_batch1: Path to first batch's drift-corrected CSV
        drift_corrected_file_batch2: Path to second batch's drift-corrected CSV
        batch_file_batch1: Path to first batch's metadata CSV
        batch_file_batch2: Path to second batch's metadata CSV
        combat_input_dir: Directory for input files
        combat_output_dir: Directory for output files (default: combat_input_dir/combat_corrected)
        batch1_label: Label for first batch
        batch2_label: Label for second batch
        rt_threshold: RT difference threshold in minutes for feature matching.
            If None, uses config value (default: 0.05 min)
        reference_batch_label: Label of reference batch for RT alignment.
            If None, uses config value (default: "MZ26_10")
        use_spline: Use spline interpolation (True) or cubic interpolation (False) for RT warping.
            If None, uses config value (default: True)
        config_path: Path to config YAML file. If None, uses default config.
    """
    # Load config if parameters not provided
    if config_path is not None or rt_threshold is None or reference_batch_label is None or use_spline is None:
        try:
            from ..config.config import Config
            config = Config(config_path) if config_path else Config()
            if rt_threshold is None:
                rt_threshold = config.get('rt_threshold', 0.05)
            if reference_batch_label is None:
                reference_batch_label = config.get('reference_batch_label', 'MZ26_10')
            if use_spline is None:
                use_spline = config.get('use_spline', True)
        except ImportError:
            # Fallback if config not available
            if rt_threshold is None:
                rt_threshold = 0.05
            if reference_batch_label is None:
                reference_batch_label = 'MZ26_10'
            if use_spline is None:
                use_spline = True
    """
    Merge two PQ-normalized batches for Combat batch correction.
    
    Uses RT warping (spline or LOESS) to align RTs to a reference batch (MZ26_10_QC3),
    then matches features by name + warped RT (within threshold).
    
    Steps:
    1. Load data and RT columns from both batches.
    2. Identify reference batch (MZ26_10_QC3) and warp other batch RTs to match.
    3. Match features by name + warped RT (within rt_threshold).
    4. For features not in reference, try to match across other batches.
    5. Keep unique features that have no matches.
    6. Remove expQC samples and ensure sample names match.
    
    Returns:
        Tuple of (merged_data, merged_batch) DataFrames.
    
    Args:
        drift_corrected_file_batch1: Path to first batch's drift-corrected CSV
        drift_corrected_file_batch2: Path to second batch's drift-corrected CSV
        batch_file_batch1: Path to first batch's metadata CSV
        batch_file_batch2: Path to second batch's metadata CSV
        combat_input_dir: Directory for input files
        combat_output_dir: Directory for output files (default: combat_input_dir/combat_corrected)
        batch1_label: Label for first batch
        batch2_label: Label for second batch
        rt_threshold: RT difference threshold for feature matching (default: 0.05 min)
        reference_batch_label: Label of reference batch for RT alignment (default: "MZ26_10")
        use_spline: Use spline interpolation (True) or LOESS (False) for RT warping (default: True)
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
    # Use Compounds ID as index if available (handles isomers with duplicate Feature names)
    # Otherwise fall back to Feature
    df1_raw = pd.read_csv(drift_corrected_file_batch1)
    df2_raw = pd.read_csv(drift_corrected_file_batch2)
    
    # Determine which column to use as index
    index_col = 'Compounds ID' if 'Compounds ID' in df1_raw.columns else 'Feature'
    
    df1 = df1_raw.set_index(index_col)
    df2 = df2_raw.set_index(index_col)
    
    # Load RT columns from the input files
    # RT is stored as a column, not in the index
    # Include index_col (Compounds ID or Feature) in the selection so we can set it as index
    rt_cols = [index_col, 'Feature', 'RT [min]']
    rt1 = df1_raw[rt_cols].set_index(index_col)
    rt2 = df2_raw[rt_cols].set_index(index_col)
    
    batch1 = pd.read_csv(batch_file_batch1)
    batch2 = pd.read_csv(batch_file_batch2)

    # Check if median normalization is needed (intrabatch scale differences)
    # Only calculate median on numeric sample columns (exclude Feature, RT [min])
    sample_cols_df1 = [col for col in df1.columns if col not in ['Feature', 'RT [min]']]
    sample_cols_df2 = [col for col in df2.columns if col not in ['Feature', 'RT [min]']]
    
    print("\n=== Batch 1 Sample Medians ===")
    batch1_medians = df1[sample_cols_df1].median(axis=0)
    print(batch1_medians.describe())
    print(f"CV of medians: {batch1_medians.std() / batch1_medians.mean() * 100:.2f}%")

    print("\n=== Batch 2 Sample Medians ===")
    batch2_medians = df2[sample_cols_df2].median(axis=0)
    print(batch2_medians.describe())
    print(f"CV of medians: {batch2_medians.std() / batch2_medians.mean() * 100:.2f}%")

    # Clean batch metadata
    batch1['batch'] = 1
    batch2['batch'] = 2
    for col in ['batch_type']:
        batch1 = batch1.drop(columns=[col], errors='ignore')
        batch2 = batch2.drop(columns=[col], errors='ignore')

    # --- RT Warping: Align RTs to reference batch ---
    # Determine which batch is the reference batch (MZ26_10_QC3)
    # We need to check which batch label contains the reference_batch_label
    
    is_batch1_reference = reference_batch_label in batch1_label
    is_batch2_reference = reference_batch_label in batch2_label
    
    if not is_batch1_reference and not is_batch2_reference:
        print(f"⚠️ Warning: Reference batch '{reference_batch_label}' not found in batch labels. Using batch1 as reference.")
        is_batch1_reference = True
    
    # Get reference batch RT data
    if is_batch1_reference:
        ref_rt = rt1
        ref_label = batch1_label
        other_rt = rt2
        other_label = batch2_label
        ref_df = df1
        other_df = df2
    else:
        ref_rt = rt2
        ref_label = batch2_label
        other_rt = rt1
        other_label = batch1_label
        ref_df = df2
        other_df = df1
    
    # For RT warping, we need common features between reference and other batch
    # to build the warping function
    common_features_ref = set(ref_rt.index) & set(other_rt.index)
    
    if len(common_features_ref) >= 3:
        # We have enough common features to build a warping function
        ref_rt_values = np.array([ref_rt.loc[feat, 'RT [min]'] for feat in common_features_ref])
        other_rt_values = np.array([other_rt.loc[feat, 'RT [min]'] for feat in common_features_ref])
        
        # Sort by reference RT
        sort_idx = np.argsort(ref_rt_values)
        ref_rt_sorted = ref_rt_values[sort_idx]
        other_rt_sorted = other_rt_values[sort_idx]
        
        # Create warping function: other_RT -> reference_RT
        # FIX: Ensure other_rt_sorted is strictly increasing for UnivariateSpline with s > 0
        if use_spline:
            # Use spline interpolation
            # Check if other_rt_sorted is strictly increasing
            if len(other_rt_sorted) > 1 and not np.all(np.diff(other_rt_sorted) > 0):
                # Remove duplicates while maintaining correspondence
                _, unique_indices = np.unique(other_rt_sorted, return_index=True)
                other_rt_sorted = other_rt_sorted[np.sort(unique_indices)]
                ref_rt_sorted = ref_rt_sorted[np.sort(unique_indices)]
                
                # Check again after deduplication
                if len(other_rt_sorted) > 1 and not np.all(np.diff(other_rt_sorted) > 0):
                    print("⚠️  RT values not strictly increasing after deduplication. Using s=0 for spline.")
                    try:
                        warp_func = UnivariateSpline(other_rt_sorted, ref_rt_sorted, s=0)
                    except Exception as e:
                        print(f"⚠️  Spline failed: {e}. Falling back to cubic interpolation.")
                        warp_func = interp1d(other_rt_sorted, ref_rt_sorted, kind='cubic', fill_value='extrapolate')
                        print("✓ Using cubic interpolation for RT warping (fallback)")
                else:
                    try:
                        warp_func = UnivariateSpline(other_rt_sorted, ref_rt_sorted, s=0.1)
                        print("✓ Using spline interpolation for RT warping")
                    except Exception as e:
                        print(f"⚠️  Spline failed: {e}. Falling back to cubic interpolation.")
                        warp_func = interp1d(other_rt_sorted, ref_rt_sorted, kind='cubic', fill_value='extrapolate')
                        print("✓ Using cubic interpolation for RT warping (fallback)")
            else:
                try:
                    warp_func = UnivariateSpline(other_rt_sorted, ref_rt_sorted, s=0.1)
                    print("✓ Using spline interpolation for RT warping")
                except ValueError as e:
                    if "x must be increasing" in str(e):
                        print("⚠️  RT values not strictly increasing. Using s=0 for spline.")
                        warp_func = UnivariateSpline(other_rt_sorted, ref_rt_sorted, s=0)
                        print("✓ Using spline interpolation for RT warping (s=0)")
                    else:
                        print(f"⚠️  Spline failed: {e}. Falling back to cubic interpolation.")
                        warp_func = interp1d(other_rt_sorted, ref_rt_sorted, kind='cubic', fill_value='extrapolate')
                        print("✓ Using cubic interpolation for RT warping (fallback)")
        else:
            # Use LOESS-like interpolation (linear for simplicity, or cubic)
            warp_func = interp1d(other_rt_sorted, ref_rt_sorted, kind='cubic', fill_value='extrapolate')
            print("✓ Using cubic interpolation for RT warping")
        
        # Warp all RTs in the other batch
        warped_rt_values = {}
        for feat in other_rt.index:
            original_rt = other_rt.loc[feat, 'RT [min]']
            warped_rt = warp_func(original_rt)
            warped_rt_values[feat] = float(warped_rt)
        
        # Create warped RT series for other batch
        other_rt_warped = other_rt.copy()
        other_rt_warped['RT [min]'] = other_rt_warped.index.map(warped_rt_values)
        
        print(f"✓ Warped {len(other_rt)} RTs from {other_label} to align with {ref_label}")
    else:
        # Not enough common features for warping, use original RTs
        print(f"⚠️ Only {len(common_features_ref)} common features, skipping RT warping")
        other_rt_warped = other_rt.copy()
    
    # --- Feature Matching with Warped RTs ---
    # Now match features by Feature name + RT (within threshold)
    # We need to group by Feature name (not by Compounds ID/index)
    
    # Build dictionaries mapping (Feature_name, index) -> RT for both batches
    # Reference batch: index -> (Feature, RT)
    ref_feature_data = {}
    for idx in ref_df.index:
        if idx in ref_rt.index:
            feature_name = ref_rt.loc[idx, 'Feature']
            rt_val = ref_rt.loc[idx, 'RT [min]']
            ref_feature_data[idx] = (feature_name, rt_val)
    
    # Other batch: index -> (Feature, warped_RT)
    other_feature_data = {}
    for idx in other_df.index:
        if idx in other_rt_warped.index:
            feature_name = other_rt_warped.loc[idx, 'Feature']
            rt_val = other_rt_warped.loc[idx, 'RT [min]']
            other_feature_data[idx] = (feature_name, rt_val)
    
    # Group features by Feature name across both batches
    feature_groups = defaultdict(list)
    
    for idx, (feat_name, rt) in ref_feature_data.items():
        feature_groups[feat_name].append(('ref', idx, rt))
    
    for idx, (feat_name, rt) in other_feature_data.items():
        feature_groups[feat_name].append(('other', idx, rt))
    
    # For each feature name, find matches based on RT similarity
    # We match by Feature name + RT, so isomers with same name but similar RT will be merged
    feature_to_match_key = {}
    matched_indices = set()
    
    for feat_name, entries in feature_groups.items():
        # Sort by RT
        entries_sorted = sorted(entries, key=lambda x: x[2])
        
        # Group entries by RT similarity (within threshold)
        groups = []
        current_group = [entries_sorted[0]]
        
        for i in range(1, len(entries_sorted)):
            prev_rt = current_group[-1][2]
            curr_rt = entries_sorted[i][2]
            
            if abs(curr_rt - prev_rt) <= rt_threshold:
                current_group.append(entries_sorted[i])
            else:
                groups.append(current_group)
                current_group = [entries_sorted[i]]
        groups.append(current_group)
        
        # For each RT group, create a match key
        # The match key will be the Feature name (since we're matching isomers with same name)
        for group in groups:
            # Use the Feature name as the match key (all entries in group have same feat_name)
            match_key = feat_name
            
            for source, idx, _ in group:
                feature_to_match_key[idx] = match_key
                matched_indices.add(idx)
    
    # Features that weren't matched (no RT within threshold of any other)
    # These will be kept as unique features with their original Feature name
    all_indices = set(ref_feature_data.keys()) | set(other_feature_data.keys())
    unmatched_indices = all_indices - matched_indices
    
    for idx in unmatched_indices:
        # For unmatched features, use their Feature name as the match key
        if idx in ref_feature_data:
            feature_to_match_key[idx] = ref_feature_data[idx][0]  # Feature name
        elif idx in other_feature_data:
            feature_to_match_key[idx] = other_feature_data[idx][0]  # Feature name
    
    # --- Apply matching and merge ---
    # Rename features in both batches according to match keys (Feature names)
    df1_renamed = df1.rename(index=lambda x: feature_to_match_key.get(x, x))
    df2_renamed = df2.rename(index=lambda x: feature_to_match_key.get(x, x))
    
    # Merge duplicates within each batch (isomers with same Feature name and similar RT)
    df1_merged = df1_renamed.groupby(level=0).mean()
    df2_merged = df2_renamed.groupby(level=0).mean()
    
    # Concatenate (keep ALL features, including unmatched ones)
    merged_data = pd.concat([df1_merged, df2_merged], axis=1, join='outer')
    merged_batch = pd.concat([batch1, batch2], ignore_index=True)
    
    # --- Add RT column to merged data ---
    # Use the RT from the match key (which is from reference batch when available)
    # Build a mapping from Feature name to RT using the first occurrence
    rt_values = {}
    for feat_name in merged_data.index:
        # Find the first index that maps to this Feature name
        # and get its RT
        for idx, (fn, rt) in ref_feature_data.items():
            if feature_to_match_key.get(idx, None) == feat_name:
                rt_values[feat_name] = rt
                break
        else:
            for idx, (fn, rt) in other_feature_data.items():
                if feature_to_match_key.get(idx, None) == feat_name:
                    rt_values[feat_name] = rt
                    break
            else:
                rt_values[feat_name] = None
    
    merged_data['RT [min]'] = pd.Series(rt_values)
    
    print(f"✓ Matched {len(matched_indices)} features by name + warped RT")
    print(f"✓ Kept {len(unmatched_indices)} unique features")

    # --- Remove expQC samples from data and batch info ---
    # Find all columns containing 'expqc' (case-insensitive), excluding RT column
    expqc_cols = [col for col in merged_data.columns if col != 'RT [min]' and 'expqc' in col.lower()]
    if expqc_cols:
        print(f"✓ Removing {len(expqc_cols)} expQC columns from data: {expqc_cols}")
        merged_data = merged_data.drop(columns=expqc_cols)
        # Remove corresponding rows from merged_batch
        merged_batch = merged_batch[~merged_batch['sample_id'].str.lower().str.contains('expqc')]
        print(f"✓ Removed expQC samples from batch metadata")

    # --- Identify batch-specific features (keep them as unique) ---
    # Use filtered batch metadata to get sample IDs
    batch1_samples = set(merged_batch[merged_batch['batch'] == 1]['sample_id'])
    batch2_samples = set(merged_batch[merged_batch['batch'] == 2]['sample_id'])
    batch1_cols = [col for col in merged_data.columns if col != 'RT [min]' and col in batch1_samples]
    batch2_cols = [col for col in merged_data.columns if col != 'RT [min]' and col in batch2_samples]

    # Batch1-only features: rows with values in batch1 cols AND all NaN in batch2 cols
    batch1_only_mask = merged_data[batch1_cols].notna().any(axis=1) & merged_data[batch2_cols].isna().all(axis=1)
    
    # Batch2-only features: rows with values in batch2 cols AND all NaN in batch1 cols
    batch2_only_mask = merged_data[batch2_cols].notna().any(axis=1) & merged_data[batch1_cols].isna().all(axis=1)
    
    # Count unique features (already handled in matching, but log for info)
    n_batch1_only = batch1_only_mask.sum()
    n_batch2_only = batch2_only_mask.sum()
    if n_batch1_only > 0 or n_batch2_only > 0:
        print(f"✓ Found {n_batch1_only} batch1-only and {n_batch2_only} batch2-only unique features (kept)")

    # --- Ensure sample names match between data and batch info ---
    # Separate RT column from sample columns
    sample_cols = [col for col in merged_data.columns if col != 'RT [min]']
    data_samples = set(sample_cols)
    batch_samples = set(merged_batch['sample_id'])
    common_samples = list(data_samples & batch_samples)

    if not common_samples:
        raise ValueError(f"No common samples between data ({data_samples}) and batch info ({batch_samples})")

    # Keep RT column and common sample columns
    merged_data = merged_data[['RT [min]'] + common_samples]
    merged_batch = merged_batch[merged_batch['sample_id'].isin(common_samples)]

    # --- Save outputs for COMBAT ---
    merged_data_path = combat_input_dir / "merged_data_for_combat.csv"
    merged_batch_path = combat_input_dir / "merged_batch_for_combat.csv"
    
    # Reset index to include Feature as a column, with RT [min] as second column
    merged_data_reset = merged_data.reset_index()
    # Reorder columns: Feature, RT [min], then samples
    cols = ['Feature', 'RT [min]'] + [c for c in merged_data_reset.columns if c not in ['Feature', 'RT [min]']]
    merged_data_reset = merged_data_reset[cols]
    merged_data_reset.to_csv(merged_data_path, index=False)
    merged_batch.to_csv(merged_batch_path, index=False)

    print(f"✓ Merged data saved to {merged_data_path}")
    print(f"✓ Batch metadata saved to {merged_batch_path}")
    return merged_data, merged_batch