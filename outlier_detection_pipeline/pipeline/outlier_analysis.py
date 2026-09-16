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
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend; avoids Tk 'main thread is not in main loop' errors on Windows
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


def compute_signed_zscore(sample: pd.Series, reference_features: pd.DataFrame) -> pd.Series:
    """
    Compute signed Z-score: (sample - mean) / std.
    Positive = increased relative to the normal mean, negative = decreased.
    Used for DIRECTIONAL plotting (ranking still uses the absolute value).
    """
    mean = reference_features.mean(axis=0)
    std = compute_feature_std(reference_features)
    return (sample - mean) / (std + 1e-10)


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
        # Signed deviation for directional plotting (Z-score mode only).
        signed_deviations = (compute_signed_zscore(sample, all_features)
                             if use_zscore else weighted_deviations)
        sorted_indices = np.argsort(weighted_deviations.values)[::-1]

        top_features = []
        for idx in sorted_indices[:n_top]:
            feature = weighted_deviations.index[idx]
            top_features.append((feature, weighted_deviations.values[idx],
                                 abs_deviations.values[idx], signed_deviations.values[idx]))

        results[outlier_idx] = {
            'top_features': top_features,
            'all_deviations': {f: (weighted_deviations[f], abs_deviations[f],
                                   signed_deviations[f]) for f in weighted_deviations.index},
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
        # Signed deviation (4th element) for directional bars; falls back to
        # the absolute value for the legacy log-IQR path that has no sign.
        signed_deviations = [f[3] if len(f) > 3 else f[1] for f in top_features[:n_top]]

        sorted_indices = np.argsort(weighted_deviations)[::-1]
        features = [features[i] for i in sorted_indices]
        weighted_deviations = [weighted_deviations[i] for i in sorted_indices]
        abs_deviations = [abs_deviations[i] for i in sorted_indices]
        signed_deviations = [signed_deviations[i] for i in sorted_indices]

        fig, ax = plt.subplots(figsize=figsize)
        # Colour by direction: increased (red) vs decreased (blue). In log-IQR
        # mode signed==absolute (all positive), so everything is 'increased'.
        colors = ['#d62728' if s >= 0 else '#1f77b4' for s in signed_deviations]
        ax.barh(features, signed_deviations, color=colors, alpha=0.7)
        ax.axvline(0, color='black', linewidth=0.8)

        # Label each bar just inside its tip (between the tip and zero) so the
        # text never extends into the y-axis compound-name labels on the left.
        # A white semi-transparent box keeps the text legible over the bar.
        max_abs = max((abs(s) for s in signed_deviations), default=1.0)
        offset = 0.02 * max_abs
        for i, (f, abs_dev, s) in enumerate(zip(features, abs_deviations, signed_deviations)):
            unit = "std" if use_zscore else "log(IQR)"
            arrow = '+' if s >= 0 else '-'
            if s >= 0:
                x = s - offset
                ha = 'right'
            else:
                x = s + offset
                ha = 'left'
            ax.text(x, i, f'{s:+.2f}x {unit} ({arrow})', va='center', ha=ha,
                    fontsize=8, bbox=dict(facecolor='white', alpha=0.7,
                                         edgecolor='none', pad=1))

        ax.set_xlabel('Signed Z-score deviation (right = increased vs normal mean, left = decreased)'
                      if use_zscore else xlabel)
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
        for rank, feat_tuple in enumerate(analysis.get('top_features', [])[:n_top], 1):
            feature_name = feat_tuple[0]
            weighted_dev = feat_tuple[1]
            abs_dev = feat_tuple[2]
            signed_dev = feat_tuple[3] if len(feat_tuple) > 3 else weighted_dev
            rows.append({
                'outlier': outlier_idx,
                'rank': rank,
                'feature': feature_name,
                weighted_col: weighted_dev,
                abs_col: abs_dev,
                'signed_zscore': signed_dev if use_zscore else '',
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
