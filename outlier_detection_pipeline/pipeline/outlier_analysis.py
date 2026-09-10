"""
Outlier analysis module for identifying features that make samples outliers.
Uses Z-scores (standard deviations from mean) or log IQR (Interquartile Range) to rank features.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import logging

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


def compute_feature_iqr(features: pd.DataFrame) -> pd.Series:
    """Compute interquartile range for each feature."""
    q1 = features.quantile(0.25, axis=0)
    q3 = features.quantile(0.75, axis=0)
    return q3 - q1


def compute_feature_std(features: pd.DataFrame) -> pd.Series:
    """Compute standard deviation for each feature."""
    return features.std(axis=0)


def compute_zscore_deviation(sample: pd.Series, reference_features: pd.DataFrame) -> pd.Series:
    """
    Compute Z-score: absolute deviation from mean in units of standard deviation.
    For RANKING: uses |sample - mean| / std
    """
    mean = reference_features.mean(axis=0)
    std = compute_feature_std(reference_features)
    return (sample - mean).abs() / (std + 1e-10)


def compute_deviation_scores(
    sample: pd.Series,
    reference_features: pd.DataFrame,
    use_zscore: bool = True,
) -> pd.Series:
    """
    Compute deviation scores for ranking features.
    
    If use_zscore=True: uses Z-scores (|sample - mean| / std)
    If use_zscore=False: uses log(IQR)-weighted deviation (|sample - median| / IQR * log(IQR))
    """
    if use_zscore:
        return compute_zscore_deviation(sample, reference_features)
    else:
        median = reference_features.median(axis=0)
        iqr = compute_feature_iqr(reference_features)
        log_iqr = np.log(iqr + 1e-10)
        abs_dev = (sample - median).abs() / (iqr + 1e-10)
        return abs_dev * (log_iqr + 1e-10)


def compute_absolute_deviation(
    sample: pd.Series,
    reference_features: pd.DataFrame,
    use_zscore: bool = True,
) -> pd.Series:
    """Compute absolute deviation (Z-score or IQR-based)."""
    if use_zscore:
        return compute_zscore_deviation(sample, reference_features)
    else:
        median = reference_features.median(axis=0)
        iqr = compute_feature_iqr(reference_features)
        return (sample - median).abs() / (iqr + 1e-10)


def filter_features_by_name(features: pd.DataFrame, feature_filter: Optional[str] = None) -> pd.DataFrame:
    """Filter features by name containing the specified substring."""
    if feature_filter is None:
        return features
    return features.loc[:, features.columns.str.contains(feature_filter, case=False, regex=False)]


def analyze_outliers(
    all_features: pd.DataFrame,
    outlier_indices: List[str],
    n_top: int = 20,
    feature_filter: Optional[str] = None,
    use_zscore: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze outliers using Z-scores or log IQR method.

    Args:
        all_features: All features DataFrame
        outlier_indices: List of outlier sample indices
        n_top: Number of top features to return per outlier
        feature_filter: Optional substring to filter features
        use_zscore: If True, use Z-scores (mean/std). If False, use log IQR method.

    Returns:
        Dictionary mapping outlier index to its feature analysis
    """
    if feature_filter:
        all_features = filter_features_by_name(all_features, feature_filter)
        logger.info(f"Filtered to {len(all_features.columns)} features containing '{feature_filter}'")

    results = {}
    method_name = "Z-score" if use_zscore else "log IQR"

    for outlier_idx in outlier_indices:
        if outlier_idx not in all_features.index:
            continue

        sample = all_features.loc[outlier_idx]
        weighted_deviations = compute_deviation_scores(sample, all_features, use_zscore=use_zscore)
        abs_deviations = compute_absolute_deviation(sample, all_features, use_zscore=use_zscore)
        sorted_indices = np.argsort(weighted_deviations.values)[::-1]

        top_features = []
        for idx in sorted_indices[:n_top]:
            feature = weighted_deviations.index[idx]
            top_features.append((feature, weighted_deviations.values[idx], abs_deviations.values[idx]))

        results[outlier_idx] = {
            'top_features': top_features,
            'all_deviations': {f: (weighted_deviations[f], abs_deviations[f]) for f in weighted_deviations.index},
        }

    logger.info(f"Analyzed {len(results)} outliers using {method_name} method")
    return results


# Backwards compatibility alias
def analyze_outliers_log_iqr(
    all_features: pd.DataFrame,
    outlier_indices: List[str],
    n_top: int = 20,
    feature_filter: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Backwards compatibility wrapper for analyze_outliers with use_zscore=False."""
    return analyze_outliers(
        all_features=all_features,
        outlier_indices=outlier_indices,
        n_top=n_top,
        feature_filter=feature_filter,
        use_zscore=False,
    )


def plot_outlier_analysis(
    outlier_analysis: Dict[str, Dict[str, Any]],
    all_features: pd.DataFrame,
    output_dir: Path,
    n_top: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    use_zscore: bool = True,
) -> None:
    """Plot outlier analysis for each outlier using Z-scores or log IQR."""
    if not HAS_MATPLOTLIB:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_name = "Z-score" if use_zscore else "log IQR"
    xlabel = 'Z-score Deviation (ranked)' if use_zscore else 'Log(IQR)-Weighted Deviation (ranked)'
    title_prefix = 'Z-score' if use_zscore else 'Log(IQR)'

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
            unit = "std" if use_zscore else "log(IQR)"
            ax.text(weighted_deviations[i], i, f'  {weighted_deviations[i]:.2f}x {unit}', va='center', fontsize=8)

        ax.set_xlabel(xlabel)
        ax.set_title(f'Outlier {outlier_idx}: Top {n_top} Features by {title_prefix} Deviation')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        filename = f"zscore_outlier_{outlier_idx}.png" if use_zscore else f"log_iqr_outlier_{outlier_idx}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()


# Backwards compatibility alias
def plot_outlier_log_iqr(
    outlier_analysis: Dict[str, Dict[str, Any]],
    all_features: pd.DataFrame,
    output_dir: Path,
    n_top: int = 20,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """Backwards compatibility wrapper for plot_outlier_analysis with use_zscore=False."""
    plot_outlier_analysis(
        outlier_analysis=outlier_analysis,
        all_features=all_features,
        output_dir=output_dir,
        n_top=n_top,
        figsize=figsize,
        use_zscore=False,
    )


def save_outlier_analysis_results(
    outlier_analysis: Dict[str, Dict[str, Any]],
    output_dir: Path,
    n_top: int = 20,
    use_zscore: bool = True,
) -> Path:
    """Save outlier analysis results to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_name = "zscore" if use_zscore else "log_iqr"
    weighted_col = 'zscore_deviation' if use_zscore else 'log_iqr_weighted_deviation'
    abs_col = 'abs_zscore' if use_zscore else 'abs_iqr_deviation'

    rows = []
    for outlier_idx, analysis in outlier_analysis.items():
        for rank, (feature_name, weighted_dev, abs_dev) in enumerate(analysis.get('top_features', [])[:n_top], 1):
            rows.append({
                'outlier': outlier_idx,
                'rank': rank,
                'feature': feature_name,
                weighted_col: weighted_dev,
                abs_col: abs_dev,
            })

    df = pd.DataFrame(rows)
    output_path = output_dir / f"outlier_{method_name}_analysis.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"Outlier {method_name} analysis saved to {output_path}")
    return output_path


# Backwards compatibility alias
def save_outlier_log_iqr_results(
    outlier_analysis: Dict[str, Dict[str, Any]],
    output_dir: Path,
    n_top: int = 20,
) -> Path:
    """Backwards compatibility wrapper for save_outlier_analysis_results with use_zscore=False."""
    return save_outlier_analysis_results(
        outlier_analysis=outlier_analysis,
        output_dir=output_dir,
        n_top=n_top,
        use_zscore=False,
    )
