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
    Compute log(IQR)-weighted absolute deviation from median.
    For RANKING: uses |sample - median| / IQR * log(IQR)
    """
    median = reference_features.median(axis=0)
    iqr = compute_feature_iqr(reference_features)
    log_iqr = np.log(iqr + 1e-10)
    abs_dev = (sample - median).abs() / (iqr + 1e-10)
    return abs_dev * (log_iqr + 1e-10)


def compute_absolute_iqr_deviation(sample: pd.Series, reference_features: pd.DataFrame) -> pd.Series:
    """Compute |sample - median| / IQR - how many IQR units away."""
    median = reference_features.median(axis=0)
    iqr = compute_feature_iqr(reference_features)
    return (sample - median).abs() / (iqr + 1e-10)


def filter_features_by_name(features: pd.DataFrame, feature_filter: Optional[str] = None) -> pd.DataFrame:
    if feature_filter is None:
        return features
    return features.loc[:, features.columns.str.contains(feature_filter, case=False, regex=False)]


def analyze_outliers_log_iqr(all_features: pd.DataFrame, outlier_indices: List[str], n_top: int = 20, feature_filter: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    if feature_filter:
        all_features = filter_features_by_name(all_features, feature_filter)
        logger.info(f"Filtered to {len(all_features.columns)} features containing '{feature_filter}'")
    
    results = {}
    for outlier_idx in outlier_indices:
        if outlier_idx not in all_features.index:
            continue
        sample = all_features.loc[outlier_idx]
        weighted_deviations = compute_deviation_scores(sample, all_features)
        abs_deviations = compute_absolute_iqr_deviation(sample, all_features)
        sorted_indices = np.argsort(weighted_deviations.values)[::-1]
        
        top_features = []
        for idx in sorted_indices[:n_top]:
            feature = weighted_deviations.index[idx]
            top_features.append((feature, weighted_deviations.values[idx], abs_deviations.values[idx]))
        
        results[outlier_idx] = {
            'top_features': top_features,
            'all_deviations': {f: (weighted_deviations[f], abs_deviations[f]) for f in weighted_deviations.index},
        }
    return results


def plot_outlier_log_iqr(outlier_analysis: Dict[str, Dict[str, Any]], all_features: pd.DataFrame, output_dir: Path, n_top: int = 20, figsize: Tuple[int, int] = (12, 8)) -> None:
    if not HAS_MATPLOTLIB:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for outlier_idx, analysis in outlier_analysis.items():
        top_features = analysis.get('top_features', [])
        if not top_features:
            continue
        features = [f[0] for f in top_features[:n_top]]
        weighted_deviations = [f[1] for f in top_features[:n_top]]
        abs_deviations = [f[2] for f in top_features[:n_top]]
        
        sorted_indices = np.argsort(weighted_deviations)[::-1]
        features = [features[i] for i in sorted_indices]
        weighted_deviations = [weighted_deviations[i] for i in sorted_indices]
        abs_deviations = [abs_deviations[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        ax.barh(features, weighted_deviations, color=colors, alpha=0.7)
        
        for i, (f, abs_dev) in enumerate(zip(features, abs_deviations)):
            ax.text(weighted_deviations[i], i, f'  {abs_dev:.2f}x IQR', va='center', fontsize=8)
        
        ax.set_xlabel('Log(IQR)-Weighted Deviation (ranked)')
        ax.set_title(f'Outlier {outlier_idx}: Top {n_top} Features by Log(IQR) Deviation')
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
        for rank, (feature_name, weighted_dev, abs_dev) in enumerate(analysis.get('top_features', [])[:n_top], 1):
            rows.append({
                'outlier': outlier_idx,
                'rank': rank,
                'feature': feature_name,
                'log_iqr_weighted_deviation': weighted_dev,
                'abs_iqr_deviation': abs_dev
            })
    df = pd.DataFrame(rows)
    output_path = output_dir / "outlier_log_iqr_analysis.csv"
    df.to_csv(output_path, index=False)
    return output_path
