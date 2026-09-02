"""
Evaluation metrics module for outlier detection pipeline.

Computes classification metrics for outlier detection results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    f1_score as sklearn_f1_score,
)
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    metrics: Optional[list] = None,
    pos_label: int = -1,
    outlier_classes: Optional[list] = None,
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
    # IsolationForest returns: -1 for outliers, 1 for inliers
    y_pred_binary = (y_pred == pos_label).astype(int)
    
    if 'accuracy' in metrics:
        results['accuracy'] = float(accuracy_score(y_true_binary, y_pred_binary))
    
    if 'f1' in metrics:
        try:
            results['f1'] = float(f1_score(y_true_binary, y_pred_binary, pos_label=1))
        except:
            results['f1'] = float('nan')
    
    if 'f1_weighted' in metrics:
        try:
            results['f1_weighted'] = float(sklearn_f1_score(y_true_binary, y_pred_binary, pos_label=1, average='weighted'))
        except:
            results['f1_weighted'] = float('nan')
    
    if 'precision' in metrics:
        try:
            results['precision'] = float(precision_score(y_true_binary, y_pred_binary, pos_label=1))
        except:
            results['precision'] = float('nan')
    
    if 'recall' in metrics:
        try:
            results['recall'] = float(recall_score(y_true_binary, y_pred_binary, pos_label=1))
        except:
            results['recall'] = float('nan')
    
    if 'roc_auc' in metrics and y_scores is not None:
        try:
            # For ROC AUC, we need scores where higher = more likely positive
            # IsolationForest gives lower scores for outliers, so we negate
            if y_scores is not None:
                score_for_auc = -y_scores  # Negate so higher = more anomalous
                results['roc_auc'] = float(roc_auc_score(y_true_binary, score_for_auc))
        except:
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
    results['n_outliers_true'] = int((y_true == pos_label).sum())
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


def save_metrics(metrics: Dict[str, Any], output_dir: Path) -> None:
    """Save metrics to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, np.integer):
            serializable_metrics[key] = int(value)
        elif isinstance(value, np.floating):
            serializable_metrics[key] = float(value)
        elif isinstance(value, pd.DataFrame):
            serializable_metrics[key] = value.to_dict()
        else:
            serializable_metrics[key] = value
    
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
