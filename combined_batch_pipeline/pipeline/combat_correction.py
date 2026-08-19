"""
ComBat batch correction module for combined batch pipeline.

This module handles:
1. Running ComBat batch correction on merged data
2. Generating diagnostic visualizations
3. Calculating batch effect removal metrics
"""

from typing import Tuple, Dict, Optional, List
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from inmoose.pycombat import pycombat_norm
    COMBAT_AVAILABLE = True
except ImportError:
    COMBAT_AVAILABLE = False

logger = logging.getLogger(__name__)


def run_combat_on_merged_data(
    merged_data: pd.DataFrame,
    merged_metadata: pd.DataFrame,
    output_dir: Path,
    ref_batch: Optional[int] = None,
    show_plots: bool = False,
    save_plots: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run ComBat batch correction on merged data.
    
    Args:
        merged_data: DataFrame with all features (rows) and samples (columns)
        merged_metadata: DataFrame with sample metadata (must have 'batch' column)
        output_dir: Directory to save results
        ref_batch: Reference batch for ComBat (optional)
        show_plots: Whether to display plots
        save_plots: Whether to save plots
        
    Returns:
        Tuple of:
        - corrected_data: ComBat-corrected DataFrame
        - metrics: Dictionary with correction metrics
    """
    if not COMBAT_AVAILABLE:
        raise ImportError("pycombat_norm from inmoose is required but not installed")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\nRunning ComBat batch correction...")
    
    # Align data and metadata
    common_samples = list(set(merged_data.columns) & set(merged_metadata['original_col']))
    merged_data = merged_data[common_samples]
    merged_metadata = merged_metadata[merged_metadata['original_col'].isin(common_samples)]
    
    # Create batch vector
    batch_dict = dict(zip(merged_metadata['original_col'], merged_metadata['batch']))
    batch_vector = np.array([batch_dict[col] for col in merged_data.columns])
    
    # Get unique batches and create numeric batch labels
    unique_batches = sorted(set(batch_vector))
    batch_to_num = {b: i+1 for i, b in enumerate(unique_batches)}
    numeric_batch_vector = np.array([batch_to_num[b] for b in batch_vector])
    
    logger.info(f"Batches: {unique_batches}")
    logger.info(f"Batch sizes: {[(b, np.sum(numeric_batch_vector == batch_to_num[b])) for b in unique_batches]}")
    
    # Handle NaN values (replace with small value)
    data_for_combat = merged_data.copy()
    min_positive = data_for_combat[data_for_combat > 0].min().min()
    small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
    data_for_combat = data_for_combat.fillna(small_value)
    
    # Filter out zero-variance features
    feature_variances = data_for_combat.var(axis=1)
    nonzero_var_mask = feature_variances > 0
    data_for_combat = data_for_combat.loc[nonzero_var_mask]
    
    logger.info(f"Features with non-zero variance: {nonzero_var_mask.sum()}/{len(nonzero_var_mask)}")
    
    # Add small epsilon to prevent numerical issues
    epsilon = 1e-10
    data_for_combat = data_for_combat + epsilon
    
    # Run ComBat
    logger.info("Running pycombat_norm...")
    combat_result = pycombat_norm(
        data_for_combat,
        numeric_batch_vector,
        covar_mod=np.zeros((len(numeric_batch_vector), 0)),
        ref_batch=ref_batch,
        na_cov_action="na.omit"
    )
    
    # Create corrected DataFrame
    corrected_data = pd.DataFrame(
        combat_result,
        index=data_for_combat.index,
        columns=data_for_combat.columns
    )
    corrected_data = corrected_data.clip(lower=0)
    
    # Restore zero-variance features
    zero_var_features = feature_variances[~nonzero_var_mask].index
    for feat in zero_var_features:
        corrected_data.loc[feat] = merged_data.loc[feat]
    
    # Save corrected data
    corrected_data.to_csv(output_dir / "combat_corrected_data.csv")
    logger.info(f"Saved ComBat-corrected data to {output_dir / 'combat_corrected_data.csv'}")
    
    # Calculate metrics
    def calculate_batch_metrics(df: pd.DataFrame, batch_vec: np.ndarray) -> Dict:
        unique_batches = np.unique(batch_vec)
        if len(unique_batches) < 2:
            return {}
        
        metrics = {}
        for i, batch_label in enumerate(unique_batches):
            batch_data = df.loc[:, batch_vec == batch_label]
            metrics[f'batch_{batch_label}_median'] = float(batch_data.median().median())
        
        # Calculate median CV between batches
        batch_medians = [metrics[f'batch_{b}_median'] for b in unique_batches]
        metrics['median_cv'] = float(np.std(batch_medians) / np.mean(batch_medians) * 100)
        
        return metrics
    
    metrics = {
        'pre_correction': calculate_batch_metrics(merged_data, numeric_batch_vector),
        'post_correction': calculate_batch_metrics(corrected_data, numeric_batch_vector),
    }
    
    # Generate plots if requested
    if save_plots:
        generate_combat_plots(
            merged_data,
            corrected_data,
            numeric_batch_vector,
            batch_to_num,
            output_dir,
        )
    
    return corrected_data, metrics


def generate_combat_plots(
    data_before: pd.DataFrame,
    data_after: pd.DataFrame,
    batch_vector: np.ndarray,
    batch_to_num: Dict,
    output_dir: Path,
) -> None:
    """Generate diagnostic plots for ComBat correction."""
    unique_batches = sorted(batch_to_num.keys())
    num_batches = len(unique_batches)
    
    # Create color palette
    palette = sns.color_palette('viridis', n_colors=num_batches)
    batch_colors = {b: palette[i] for i, b in enumerate(unique_batches)}
    
    # Boxplot before correction
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x=[batch_to_num[b] for b in batch_vector],
        y=data_before.T.mean(axis=1),
        palette=palette,
        showfliers=False
    )
    plt.title("Before ComBat - Mean Intensity by Batch")
    plt.xlabel("Batch")
    plt.ylabel("Mean Intensity")
    plt.xticks(range(1, num_batches + 1), unique_batches)
    plt.savefig(output_dir / "before_combat_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Boxplot after correction
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x=[batch_to_num[b] for b in batch_vector],
        y=data_after.T.mean(axis=1),
        palette=palette,
        showfliers=False
    )
    plt.title("After ComBat - Mean Intensity by Batch")
    plt.xlabel("Batch")
    plt.ylabel("Mean Intensity")
    plt.xticks(range(1, num_batches + 1), unique_batches)
    plt.savefig(output_dir / "after_combat_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved plots to {output_dir}")
