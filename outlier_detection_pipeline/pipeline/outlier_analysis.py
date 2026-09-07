"""
Outlier analysis module for identifying features that make samples outliers.
Uses log IQR (Interquartile Range) to rank features.
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
    q1 = features.quantile(0.25, axis=0)
    q3 = features.quantile(0.75, axis=0)
    return q3 - q1


def compute_deviation_scores(sample: pd.Series, reference_features: pd.DataFrame) -> pd.Series:
    """
    Compute absolute deviation from median in IQR units.
    Returns |sample_value - median| / IQR for each feature.
    """
    median = reference_features.median(axis=0)
    iqr = compute_feature_iqr(reference_features)
    # Absolute deviation in IQR units (how many IQRs away from median)
    return (sample - median).abs() / (iqr + 1e-10)


def filter_features_by_name(features: pd.DataFrame, feature_filter: Optional[str] = None) -> pd.DataFrame:
    """Filter features by name substring."""
    if feature_filter is None:
        return features
    return features.loc[:, features.columns.str.contains(feature_filter, case=False, regex=False)]


def analyze_outliers_log_iqr(all_features: pd.DataFrame, outlier_indices: List[str], n_top: int = 20, feature_filter: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    For each outlier, find the top N features with highest absolute deviation in IQR units.
    
    Args:
        all_features: DataFrame with ALL samples (samples x features)
        outlier_indices: List of sample indices that are outliers
        n_top: Number of top features to return per outlier
        feature_filter: Optional substring to filter features (e.g., 'hmdb')
        
    Returns:
        Dictionary mapping outlier index -> {
            'top_features': [(feature_name, iqr_deviation), ...],
            'all_deviations': {feature_name: iqr_deviation, ...}
        }
    """
    # Apply feature filter if specified
    if feature_filter:
        all_features = filter_features_by_name(all_features, feature_filter)
        logger.info(f"Filtered to {len(all_features.columns)} features containing '{feature_filter}'")
    
    results = {}
    
    for outlier_idx in outlier_indices:
        if outlier_idx not in all_features.index:
            continue
        
        sample = all_features.loc[outlier_idx]
        deviations = compute_deviation_scores(sample, all_features)
        
        # Sort by absolute deviation in IQR units (descending)
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


def plot_outlier_log_iqr(outlier_analysis: Dict[str, Dict[str, Any]], all_features: pd.DataFrame, output_dir: Path, n_top: int = 20, figsize: Tuple[int, int] = (12, 8)) -> None:
    if not HAS_MATPLOTLIB:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get IQR for each feature for reference
    iqr = compute_feature_iqr(all_features)
    median = all_features.median(axis=0)
    
    for outlier_idx, analysis in outlier_analysis.items():
        top_features = analysis.get('top_features', [])
        if not top_features:
            continue
        features = [f[0] for f in top_features[:n_top]]
        deviations = [f[1] for f in top_features[:n_top]]
        
        # Sort by deviation (descending)
        sorted_indices = np.argsort(deviations)[::-1]
        features = [features[i] for i in sorted_indices]
        deviations = [deviations[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        ax.barh(features, deviations, color=colors, alpha=0.7)
        
        # Label shows how many IQR units away from median
        for i, (f, dev) in enumerate(zip(features, deviations)):
            ax.text(dev, i, f'  {dev:.2f}x IQR', va='center', fontsize=8)
        
        ax.set_xlabel('Deviation in IQR units (absolute)')
        ax.set_title(f'Outlier {outlier_idx}: Top {n_top} Features by IQR Deviation')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(output_dir / f"log_iqr_outlier_{outlier_idx}.png", dpi=300, bbox_inches='tight')
        plt.close()


def save_outlier_log_iqr_results(outlier_analysis: Dict[str, Dict[str, Any]], output_dir: Path, n_top: int = 20) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for outlier_idx, analysis in outlier_analysis.items():
        for rank, (feature_name, deviation) in enumerate(analysis.get('top_features', [])[:n_top], 1):
            rows.append({'outlier': outlier_idx, 'rank': rank, 'feature': feature_name, 'iqr_deviation': deviation})
    df = pd.DataFrame(rows)
    output_path = output_dir / "outlier_log_iqr_analysis.csv"
    df.to_csv(output_path, index=False)
    return output_path
