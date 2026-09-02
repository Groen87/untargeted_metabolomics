"""
Realistic evaluation module for outlier detection pipeline.

Implements the recommended strategy:
1. Train on full training set with contamination matching training data
2. Test with realistic contamination (e.g., 2%) by LOO abnormal samples
3. Repeat N iterations for stable metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
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


def run_realistic_evaluation(
    model: any,
    X_normal_test: pd.DataFrame,
    X_abnormal_test: pd.DataFrame,
    y_normal_test: pd.Series,
    y_abnormal_test: pd.Series,
    target_contamination: float = 0.02,
    n_iterations: int = 50,
    random_seed: int = 42,
    outlier_classes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run realistic LOO evaluation with target contamination rate.
    
    For each iteration:
    1. Select 1 abnormal sample (randomly from abnormal pool)
    2. Combine with ALL normal samples from test set
    3. This creates a test set with ~2% contamination (1 abnormal / ~50 total)
    4. Get outlier scores from model
    5. Check if abnormal sample is in top N% (where N = target_contamination * 100)
    
    Args:
        model: Trained ExtendedIsolationForestModel
        X_normal_test: Normal test samples features
        X_abnormal_test: Abnormal test samples features
        y_normal_test: Normal test samples labels
        y_abnormal_test: Abnormal test samples labels
        target_contamination: Target contamination rate (e.g., 0.02 for 2%)
        n_iterations: Number of LOO iterations
        random_seed: Random seed for reproducibility
        outlier_classes: List of classification values that are outliers
        
    Returns:
        Dictionary with:
        - detection_rate: % of iterations where abnormal was detected
        - false_positive_rate: % of normal samples flagged as outliers
        - per_iteration_results: List of dicts with detailed results
        - confusion_matrix_aggregated: Aggregated confusion matrix
        - metrics: Aggregated metrics across all iterations
    """
    if outlier_classes is None:
        outlier_classes = [1, 2, 3]
    
    np.random.seed(random_seed)
    
    n_normal = len(X_normal_test)
    n_abnormal = len(X_abnormal_test)
    n_top = int(target_contamination * n_normal)  # Number of top outliers to flag
    
    if n_top < 1:
        n_top = 1  # Always flag at least 1
    
    logger.info(f"\n{'='*70}")
    logger.info("REALISTIC EVALUATION (LOO Abnormal)")
    logger.info(f"{'='*70}")
    logger.info(f"Normal test samples: {n_normal}")
    logger.info(f"Abnormal test samples: {n_abnormal}")
    logger.info(f"Target contamination: {target_contamination:.2%}")
    logger.info(f"Top N outliers to flag: {n_top}")
    logger.info(f"Iterations: {n_iterations}")
    
    # Combine all normal samples (they stay the same in each iteration)
    X_normal_combined = X_normal_test
    y_normal_combined = y_normal_test
    
    # Store results for each iteration
    per_iteration_results = []
    all_true_labels = []
    all_pred_labels = []
    all_scores = []
    
    # For tracking
    detected_count = 0
    fp_count = 0
    total_normal_samples = 0
    
    for iteration in range(n_iterations):
        # Select one random abnormal sample
        abnormal_idx = np.random.randint(0, n_abnormal)
        X_abnormal_selected = X_abnormal_test.iloc[[abnormal_idx]]
        y_abnormal_selected = y_abnormal_test.iloc[[abnormal_idx]]
        
        # Create test set: all normals + 1 abnormal
        X_test_iter = pd.concat([X_normal_combined, X_abnormal_selected])
        y_test_iter = pd.concat([y_normal_combined, y_abnormal_selected])
        
        # Get scores (lower = more anomalous for IsolationForest)
        scores = model.decision_function(X_test_iter)
        
        # Get predictions based on threshold (top N by score)
        # Since lower scores = more anomalous, we sort ascending
        sorted_indices = np.argsort(scores)
        top_outlier_indices = sorted_indices[:n_top]
        
        # Create binary predictions: 1 for top outliers, 0 otherwise
        y_pred_iter = np.zeros(len(scores), dtype=int)
        y_pred_iter[top_outlier_indices] = 1
        
        # Convert ground truth to binary (0=normal, 1=outlier)
        y_true_binary = (y_test_iter.isin(outlier_classes)).astype(int)
        
        # Check if abnormal sample was detected
        # The abnormal sample is at position n_normal (last position)
        abnormal_position = n_normal
        abnormal_detected = y_pred_iter[abnormal_position] == 1
        
        if abnormal_detected:
            detected_count += 1
        
        # Count false positives (normal samples flagged as outliers)
        fp_iter = np.sum(y_pred_iter[:n_normal] == 1)
        fp_count += fp_iter
        total_normal_samples += n_normal
        
        # Compute metrics for this iteration
        iter_results = {
            'iteration': iteration,
            'abnormal_sample_id': X_abnormal_test.index[abnormal_idx],
            'abnormal_class': y_abnormal_selected.iloc[0],
            'abnormal_detected': bool(abnormal_detected),
            'abnormal_score': float(scores[abnormal_position]),
            'false_positives': int(fp_iter),
            'n_top': n_top,
        }
        
        per_iteration_results.append(iter_results)
        
        # Store for aggregated metrics
        all_true_labels.extend(y_true_binary.tolist())
        all_pred_labels.extend(y_pred_iter.tolist())
        all_scores.extend(scores.tolist())
    
    # Compute aggregated metrics
    detection_rate = detected_count / n_iterations
    false_positive_rate = fp_count / total_normal_samples if total_normal_samples > 0 else 0
    
    # Aggregated confusion matrix
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    
    # Compute various metrics
    try:
        accuracy = accuracy_score(all_true_labels, all_pred_labels)
    except:
        accuracy = float('nan')
    
    try:
        precision = precision_score(all_true_labels, all_pred_labels)
    except:
        precision = float('nan')
    
    try:
        recall = recall_score(all_true_labels, all_pred_labels)
    except:
        recall = float('nan')
    
    try:
        f1 = f1_score(all_true_labels, all_pred_labels)
    except:
        f1 = float('nan')
    
    try:
        roc_auc = roc_auc_score(all_true_labels, -np.array(all_scores))
    except:
        roc_auc = float('nan')
    
    results = {
        'evaluation_strategy': 'realistic',
        'n_iterations': n_iterations,
        'n_normal_test': n_normal,
        'n_abnormal_test': n_abnormal,
        'target_contamination': target_contamination,
        'n_top': n_top,
        'detection_rate': float(detection_rate),
        'false_positive_rate': float(false_positive_rate),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': ['Normal', 'Outlier'],
        'per_iteration_results': per_iteration_results,
    }
    
    logger.info(f"\n{'='*70}")
    logger.info("REALISTIC EVALUATION RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Detection rate: {detection_rate:.2%}")
    logger.info(f"False positive rate: {false_positive_rate:.2%}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1: {f1:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"{'='*70}")
    
    return results


def save_realistic_results(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Save realistic evaluation results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save aggregated metrics
    metrics_path = output_dir / "realistic_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Realistic metrics saved to {metrics_path}")
    
    # Save per-iteration results as CSV
    if 'per_iteration_results' in results:
        iter_df = pd.DataFrame(results['per_iteration_results'])
        iter_path = output_dir / "realistic_iteration_results.csv"
        iter_df.to_csv(iter_path, index=False)
        logger.info(f"Per-iteration results saved to {iter_path}")
    
    # Save summary
    summary = {
        'detection_rate': results['detection_rate'],
        'false_positive_rate': results['false_positive_rate'],
        'n_iterations': results['n_iterations'],
        'n_normal_test': results['n_normal_test'],
        'n_abnormal_test': results['n_abnormal_test'],
        'target_contamination': results['target_contamination'],
    }
    summary_path = output_dir / "realistic_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


def plot_realistic_results(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot realistic evaluation results."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib/seaborn not available. Skipping plots.")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Detection by iteration
    plt.figure(figsize=(12, 6))
    
    # Extract iteration data
    iterations = [r['iteration'] for r in results['per_iteration_results']]
    detected = [1 if r['abnormal_detected'] else 0 for r in results['per_iteration_results']]
    fps = [r['false_positives'] for r in results['per_iteration_results']]
    
    # Detection rate over iterations
    plt.subplot(1, 2, 1)
    plt.plot(iterations, detected, 'o-', color='blue', alpha=0.5)
    plt.axhline(y=results['detection_rate'], color='red', linestyle='--',
                label=f'Detection Rate: {results["detection_rate"]:.2%}')
    plt.xlabel('Iteration')
    plt.ylabel('Detected (1=Yes, 0=No)')
    plt.title('Abnormal Sample Detection by Iteration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # False positives over iterations
    plt.subplot(1, 2, 2)
    plt.plot(iterations, fps, 'o-', color='green', alpha=0.5)
    plt.axhline(y=np.mean(fps), color='red', linestyle='--',
                label=f'Avg FP: {np.mean(fps):.2f}')
    plt.xlabel('Iteration')
    plt.ylabel('False Positives')
    plt.title('False Positives by Iteration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "realistic_detection_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Realistic evaluation plot saved to {output_dir / 'realistic_detection_plot.png'}")
