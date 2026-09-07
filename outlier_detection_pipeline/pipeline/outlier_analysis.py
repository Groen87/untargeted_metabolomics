"""
Outlier analysis module for identifying features that make samples outliers.

Uses log IQR (Interquartile Range) to rank features by how different they are
for each outlier compared to the rest of the dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import logging

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


def compute_feature_iqr(features: pd.DataFrame) -> pd.Series:
    """Compute IQR (Q3 - Q1) for each feature across all samples."""
    q1 = features.quantile(0.25, axis=0)
    q3 = features.quantile(0.75, axis=0)
    return q3 - q1


def compute_log_iqr(features: pd.DataFrame) -> pd.Series:
    """Compute log(IQR) for each feature."""
    iqr = compute_feature_iqr(features)
    return np.log(iqr + 1e-10)


def compute_deviation_scores(
    sample: pd.Series,
    reference_features: pd.DataFrame,
) -> pd.Series:
    """
    Compute log IQR-weighted deviation scores for a sample.
    
    For each feature:
    1. Compute IQR across all samples
    2. Compute log(IQR) to normalize scale
    3. Compute |sample_value - median| / IQR (scale-invariant deviation)
    4. Weight by log(IQR) so features with more natural variation get emphasis
    """
    median = reference_features.median(axis=0)
    iqr = compute_feature_iqr(reference_features)
    log_iqr = np.log(iqr + 1e-10)
    
    abs_dev = (sample - median).abs() / (iqr + 1e-10)
    weighted_dev = abs_dev * (log_iqr + 1e-10)
    
    return weighted_dev


def analyze_outliers_log_iqr(
    all_features: pd.DataFrame,
    outlier_indices: List[str],
    n_top: int = 20,
) -> Dict[str, Dict[str, Any]]:
    """
    For each outlier, find the top N features with highest log IQR-weighted deviation.
    
    Args:
        all_features: DataFrame with ALL samples (samples x features)
        outlier_indices: List of sample indices that are outliers
        n_top: Number of top features to return per outlier
        
    Returns:
        Dictionary mapping outlier index -> {
            'top_features': [(feature_name, deviation_score), ...],
            'all_deviations': {feature_name: score, ...}
        }
    """
    results = {}
    
    for outlier_idx in outlier_indices:
        if outlier_idx not in all_features.index:
            logger.warning(f"Outlier index {outlier_idx} not found in features")
            continue
        
        sample = all_features.loc[outlier_idx]
        deviations = compute_deviation_scores(sample, all_features)
        
        sorted_features = sorted(
            zip(deviations.index, deviations.values),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_features = sorted_features[:n_top]
        
        results[outlier_idx] = {
            'top_features': top_features,
            'all_deviations': dict(zip(deviations.index, deviations.values)),
        }
    
    return results


def plot_outlier_log_iqr(
    outlier_analysis: Dict[str, Dict[str, Any]],
    all_features: pd.DataFrame,
    output_dir: Path,
    n_top: int = 20,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    For each outlier, create a horizontal bar plot showing the top N features
    ranked by their log IQR-weighted deviation score.
    
    Args:
        outlier_analysis: Dictionary from analyze_outliers_log_iqr()
        all_features: Original feature DataFrame
        output_dir: Output directory
        n_top: Number of top features to show
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available. Skipping log IQR plots.")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    iqr = compute_feature_iqr(all_features)
    
    for outlier_idx, analysis in outlier_analysis.items():
        top_features = analysis.get('top_features', [])
        if not top_features:
            continue
        
        features = [f[0] for f in top_features[:n_top]]
        deviations = [f[1] for f in top_features[:n_top]]
        feature_iqrs = [iqr.get(f, 0) for f in features]
        
        sorted_indices = np.argsort(deviations)[::-1]
        features = [features[i] for i in sorted_indices]
        deviations = [deviations[i] for i in sorted_indices]
        feature_iqrs = [feature_iqrs[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        bars = ax.barh(features, deviations, color=colors, alpha=0.7)
        
        for i, (bar, f, iqr_val, dev) in enumerate(zip(bars, features, feature_iqrs, deviations)):
            ax.text(dev, i, f'  IQR={iqr_val:.2f}', va='center', fontsize=8)
        
        ax.set_xlabel('Log IQR-Weighted Deviation')
        ax.set_title(f'Outlier {outlier_idx}: Top {n_top} Features by Log IQR Deviation')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        output_path = output_dir / f"log_iqr_outlier_{outlier_idx}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Log IQR plots saved to {output_dir}/")


def save_outlier_log_iqr_results(
    outlier_analysis: Dict[str, Dict[str, Any]],
    output_dir: Path,
    n_top: int = 20,
) -> Path:
    """Save outlier log IQR analysis to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for outlier_idx, analysis in outlier_analysis.items():
        for rank, (feature_name, deviation) in enumerate(analysis.get('top_features', [])[:n_top], 1):
            rows.append({
                'outlier': outlier_idx,
                'rank': rank,
                'feature': feature_name,
                'log_iqr_weighted_deviation': deviation,
            })
    
    df = pd.DataFrame(rows)
    output_path = output_dir / "outlier_log_iqr_analysis.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Outlier log IQR analysis saved to {output_path}")
    return output_path


def print_outlier_log_iqr_summary(
    outlier_analysis: Dict[str, Dict[str, Any]],
    n_top: int = 20,
) -> None:
    """Print formatted summary of log IQR analysis."""
    logger.info(f"\n{'='*70}")
    logger.info("OUTLIER LOG IQR ANALYSIS")
    logger.info(f"{'='*70}")
    
    for outlier_idx, analysis in outlier_analysis.items():
        top_features = analysis.get('top_features', [])
        logger.info(f"\n--- Outlier: {outlier_idx} ---")
        logger.info(f"  Top {min(n_top, len(top_features))} features by log IQR-weighted deviation:")
        for rank, (feature_name, deviation) in enumerate(top_features[:n_top], 1):
            logger.info(f"    {rank}. {feature_name}: {deviation:.4f}")
    
    logger.info(f"\n{'='*70}")
