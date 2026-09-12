"""
Evaluation metrics module for outlier detection pipeline.

Computes classification metrics for outlier detection results.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
import logging

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend; avoids Tk 'main thread is not in main loop' errors on Windows
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    metrics: Optional[List[str]] = None,
    pos_label: int = -1,
    outlier_classes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Evaluate model predictions against ground truth.

    For outlier detection:
    - Inliers (normal, Classification=0) = 1
    - Outliers (Classification=1,2,3) = -1

    Args:
        y_true: True labels (Classification values: 0=normal, 1,2,3=outliers)
        y_pred: Predicted labels (-1 for outliers, 1 for inliers)
        y_scores: Anomaly scores (lower = more anomalous)
        metrics: List of metrics to compute
        pos_label: Positive class label (default: -1 for outliers)
        outlier_classes: List of classification values that are outliers (default: [1,2,3])

    Returns:
        Dictionary with computed metrics
    """
    if metrics is None:
        metrics = ['accuracy', 'f1', 'f1_weighted', 'precision', 'recall',
                   'roc_auc', 'confusion_matrix', 'classification_report']

    if outlier_classes is None:
        outlier_classes = [1, 2, 3]

    results = {}

    # Convert ground truth to binary: 0=inlier (classification 0), 1=outlier (classification 1,2,3)
    y_true_binary = (y_true.isin(outlier_classes)).astype(int)

    # Convert predictions to binary: 0=inlier (pred=1), 1=outlier (pred=-1)
    y_pred_binary = (y_pred == pos_label).astype(int)

    # Compute each requested metric
    if 'accuracy' in metrics:
        results['accuracy'] = float(accuracy_score(y_true_binary, y_pred_binary))

    if 'f1' in metrics:
        try:
            results['f1'] = float(f1_score(y_true_binary, y_pred_binary, pos_label=1))
        except Exception:
            results['f1'] = float('nan')

    if 'f1_weighted' in metrics:
        try:
            results['f1_weighted'] = float(f1_score(y_true_binary, y_pred_binary, pos_label=1, average='weighted'))
        except Exception:
            results['f1_weighted'] = float('nan')

    if 'precision' in metrics:
        try:
            results['precision'] = float(precision_score(y_true_binary, y_pred_binary, pos_label=1))
        except Exception:
            results['precision'] = float('nan')

    if 'recall' in metrics:
        try:
            results['recall'] = float(recall_score(y_true_binary, y_pred_binary, pos_label=1))
        except Exception:
            results['recall'] = float('nan')

    if 'roc_auc' in metrics and y_scores is not None:
        try:
            # For ROC AUC: higher values should indicate more likely positive
            # IsolationForest gives lower scores for outliers, so we negate
            score_for_auc = -y_scores
            results['roc_auc'] = float(roc_auc_score(y_true_binary, score_for_auc))
        except Exception:
            results['roc_auc'] = float('nan')

    if 'confusion_matrix' in metrics:
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        results['confusion_matrix'] = cm.tolist()
        results['confusion_matrix_labels'] = ['Inlier', 'Outlier']

    if 'classification_report' in metrics:
        report = classification_report(y_true_binary, y_pred_binary, target_names=['Inlier', 'Outlier'])
        results['classification_report'] = report

    # Add additional info
    results['n_samples'] = len(y_true)
    results['n_outliers_true'] = int(y_true_binary.sum())
    results['n_outliers_predicted'] = int((y_pred == pos_label).sum())

    return results


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print formatted metrics to console."""
    logger.info("\n" + "="*60)
    logger.info("EVALUATION METRICS")
    logger.info("="*60)

    for key, value in metrics.items():
        if key == 'confusion_matrix':
            logger.info(f"\n{key}:")
            cm = np.array(value)
            logger.info(f"  [[{cm[0,0]}, {cm[0,1]}],")
            logger.info(f"   [{cm[1,0]}, {cm[1,1]}]]")
            logger.info(f"  Labels: {metrics.get('confusion_matrix_labels', ['Inlier', 'Outlier'])}")
        elif key == 'classification_report':
            logger.info(f"\n{key}:\n{value}")
        elif isinstance(value, float):
            logger.info(f"{key:25s}: {value:.4f}")
        else:
            logger.info(f"{key:25s}: {value}")

    logger.info("="*60)


def _make_serializable(value: Any) -> Any:
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.integer, np.floating)):
        return int(value) if isinstance(value, np.integer) else float(value)
    elif isinstance(value, pd.DataFrame):
        return value.to_dict()
    elif isinstance(value, dict):
        return {k: _make_serializable(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_make_serializable(item) for item in value]
    else:
        return value


def save_metrics(metrics: Dict[str, Any], output_dir: Path) -> None:
    """Save metrics to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    serializable_metrics = _make_serializable(metrics)

    output_path = output_dir / "metrics.json"
    with open(output_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)

    logger.info(f"Metrics saved to {output_path}")


def save_predictions(
    predictions: np.ndarray,
    scores: np.ndarray,
    patient_ids: pd.Index,
    true_labels: pd.Series,
    output_dir: Path,
    split_name: str = "test",
) -> None:
    """Save predictions to CSV file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        'patient_id': patient_ids,
        'true_label': true_labels.values,
        'predicted_label': predictions,
        'anomaly_score': scores,
    })

    output_path = output_dir / f"{split_name}_predictions.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"Predictions saved to {output_path}")


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_dir: Path,
    outlier_classes: Optional[List[int]] = None,
    pos_label: int = -1,
) -> None:
    """Plot a seaborn confusion matrix."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib/seaborn not available. Skipping confusion matrix plot.")
        return

    if outlier_classes is None:
        outlier_classes = [1, 2, 3]

    # Convert to binary
    y_true_binary = (y_true.isin(outlier_classes)).astype(int)
    y_pred_binary = (y_pred == pos_label).astype(int)

    cm = confusion_matrix(y_true_binary, y_pred_binary)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Inlier', 'Outlier'],
                yticklabels=['Inlier', 'Outlier'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Confusion matrix plot saved to {output_path}")


def plot_precision_recall_curve(
    y_true: pd.Series,
    y_scores: np.ndarray,
    output_dir: Path,
    outlier_classes: Optional[List[int]] = None,
    pos_label: int = -1,
) -> None:
    """Plot a seaborn precision-recall curve."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib/seaborn not available. Skipping PR curve plot.")
        return

    if outlier_classes is None:
        outlier_classes = [1, 2, 3]

    # Convert to binary
    y_true_binary = (y_true.isin(outlier_classes)).astype(int)

    # IsolationForest: lower scores = more anomalous, so negate for PR curve
    y_scores_for_pr = -y_scores

    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_scores_for_pr)
    avg_precision = average_precision_score(y_true_binary, y_scores_for_pr)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.2f})')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='blue')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "precision_recall_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Precision-Recall curve saved to {output_path}")
