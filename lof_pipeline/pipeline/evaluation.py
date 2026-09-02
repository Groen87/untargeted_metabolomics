"""
Evaluation module for LOF outlier detection pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
import logging
import json
from pathlib import Path

try:
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
    """Evaluate model predictions."""
    if metrics is None:
        metrics = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'confusion_matrix']
    
    if outlier_classes is None:
        outlier_classes = [1, 2, 3]
    
    results = {}
    
    # Convert to binary if needed
    y_true_binary = (y_true.isin(outlier_classes)).astype(int)
    y_pred_binary = (y_pred == pos_label).astype(int)
    
    if 'accuracy' in metrics:
        results['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
    
    if 'f1' in metrics:
        try:
            results['f1'] = f1_score(y_true_binary, y_pred_binary, pos_label=1)
        except:
            results['f1'] = float('nan')
    
    if 'precision' in metrics:
        try:
            results['precision'] = precision_score(y_true_binary, y_pred_binary, pos_label=1)
        except:
            results['precision'] = float('nan')
    
    if 'recall' in metrics:
        try:
            results['recall'] = recall_score(y_true_binary, y_pred_binary, pos_label=1)
        except:
            results['recall'] = float('nan')
    
    if 'roc_auc' in metrics and y_scores is not None:
        try:
            results['roc_auc'] = roc_auc_score(y_true_binary, -y_scores)
        except:
            results['roc_auc'] = float('nan')
    
    if 'confusion_matrix' in metrics:
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        results['confusion_matrix'] = cm.tolist()
        results['confusion_matrix_labels'] = ['Inlier', 'Outlier']
    
    return results


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print metrics."""
    logger.info(f"\n{'='*50}")
    logger.info("EVALUATION METRICS")
    logger.info(f"{'='*50}")
    for name, value in metrics.items():
        if name == 'confusion_matrix':
            logger.info(f"{name}:\n{np.array(value)}")
        elif name == 'confusion_matrix_labels':
            logger.info(f"{name}  : {value}")
        else:
            logger.info(f"{name:25s}: {value:.4f}")
    logger.info(f"{'='*50}")


def save_metrics(metrics: Dict[str, Any], output_dir: Path) -> None:
    """Save metrics to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


def save_predictions(
    predictions: np.ndarray,
    scores: np.ndarray,
    patient_ids,
    true_labels: pd.Series,
    output_dir: Path,
    split_name: str = "test",
) -> None:
    """Save predictions to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame({
        'patient_id': patient_ids,
        'true_label': true_labels.values,
        'prediction': predictions,
        'score': scores,
    })
    
    predictions_path = output_dir / f"{split_name}_predictions.csv"
    df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_dir: Path,
    outlier_classes: Optional[List[int]] = None,
    pos_label: int = -1,
) -> None:
    """Plot confusion matrix."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib/seaborn not available. Skipping confusion matrix plot.")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_true_binary = (y_true.isin(outlier_classes or [1, 2, 3])).astype(int)
    y_pred_binary = (y_pred == pos_label).astype(int)
    
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Inlier', 'Outlier'],
                yticklabels=['Inlier', 'Outlier'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix plot saved to {output_dir / 'confusion_matrix.png'}")


def plot_precision_recall_curve(
    y_true: pd.Series,
    y_scores: np.ndarray,
    output_dir: Path,
    outlier_classes: Optional[List[int]] = None,
    pos_label: int = -1,
) -> None:
    """Plot precision-recall curve."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib/seaborn not available. Skipping precision-recall plot.")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_true_binary = (y_true.isin(outlier_classes or [1, 2, 3])).astype(int)
    
    precision, recall, _ = precision_recall_curve(y_true_binary, -y_scores)
    avg_precision = average_precision_score(y_true_binary, -y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP={avg_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Precision-Recall curve saved to {output_dir / 'precision_recall_curve.png'}")
