"""Pipeline modules for outlier detection."""

from .data_loader import load_data, split_data
from .model import ExtendedIsolationForestModel
from .evaluation import evaluate_model, print_metrics
from .pca import SparsePCAWrapper
from .realistic_evaluation import run_realistic_evaluation, save_realistic_results, plot_realistic_results
from .outlier_analysis import (
    analyze_outliers,
    plot_outlier_analysis,
    save_outlier_analysis_results,
    analyze_outliers_log_iqr,
    plot_outlier_log_iqr,
    save_outlier_log_iqr_results,
)

__all__ = [
    "load_data",
    "split_data", 
    "ExtendedIsolationForestModel",
    "evaluate_model",
    "print_metrics",
    "SparsePCAWrapper",
    "run_realistic_evaluation",
    "save_realistic_results",
    "plot_realistic_results",
    "analyze_outliers",
    "plot_outlier_analysis",
    "save_outlier_analysis_results",
    "analyze_outliers_log_iqr",
    "plot_outlier_log_iqr",
    "save_outlier_log_iqr_results",
]
