"""
Realistic evaluation module for outlier detection pipeline.

Implements the recommended strategy:
1. Train on full training set with contamination matching training data
2. Test with realistic contamination (e.g., 2%) by LOO abnormal samples
3. Repeat N iterations for stable metrics
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
import logging

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
    X_normal_train: Optional[pd.DataFrame] = None,
    y_normal_train: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Run realistic evaluation with an absolute anomaly threshold (single pass).

    An absolute anomaly threshold is derived from the normal training score
    distribution: the (100 * target_contamination)-th percentile of the
    model's raw score_samples() on the normal training samples. Every test
    sample is then scored ONCE and flagged if its score falls below this
    cutoff, so a clean batch of normals can legitimately flag 0 outliers.

    IsolationForest scores each sample independently of the other samples in
    the batch, so a single deterministic scoring pass over all test samples
    is fully representative of per-sample deployment behaviour; no
    Monte-Carlo batch resampling is needed to estimate detection or
    false-positive rate. Those per-sample metrics are prevalence-independent
    and directly deployment-representative.

    Aggregate metrics that mix the two classes (precision/F1/accuracy) depend
    on the batch abnormal-to-normal ratio, which differs between the
    high-prevalence labelled test set and the low-prevalence deployment
    scenario. They are therefore computed ANALYTICALLY at the assumed
    deployment prevalence (target_contamination) from the
    prevalence-independent recall and FPR, rather than from the raw test-set
    prevalence.

    Args:
        model: Trained ExtendedIsolationForestModel (trained on normals only)
        X_normal_test: Normal test samples features
        X_abnormal_test: Abnormal test samples features
        y_normal_test: Normal test samples labels
        y_abnormal_test: Abnormal test samples labels
        target_contamination: Deployment contamination rate (e.g. 0.02 for
            2%). Used (a) as the percentile of the normal-training score
            distribution defining the absolute flag threshold, and (b) as
            the assumed prevalence for analytic precision/F1/accuracy.
        n_iterations: Deprecated/ignored (kept for call-site compatibility).
            Scoring is a single deterministic pass.
        random_seed: Kept for call-site compatibility (no longer used).
        outlier_classes: List of classification values that are outliers
        X_normal_train: Normal training samples, used to calibrate the
            absolute anomaly threshold (score reference distribution).
            Strongly recommended: should be the normals the model trained
            on, NOT the test normals, to avoid threshold leakage.
        y_normal_train: Normal training labels (optional)

    Returns:
        Dictionary with:
        - detection_rate (recall): fraction of abnormals flagged (prevalence-independent)
        - false_positive_rate: fraction of normals flagged (prevalence-independent)
        - precision/f1/accuracy: computed analytically at deployment prevalence
        - roc_auc: ranking quality over the scored test set
        - anomaly_threshold: the calibrated absolute score cutoff
        - per_iteration_results: per-sample results (one entry per test sample)
    """
    if outlier_classes is None:
        outlier_classes = [1, 2, 3]

    n_normal = len(X_normal_test)
    n_abnormal = len(X_abnormal_test)
    n_test_total = n_normal + n_abnormal

    # Derive an absolute anomaly threshold from the normal training score
    # distribution. The threshold is the (100 * target_contamination)-th
    # percentile of the raw score_samples() on normal training data, i.e.
    # the score below which only ~target_contamination of clean normals fall.
    # Test samples are flagged ONLY if their score drops below this cutoff,
    # so a clean batch can legitimately flag 0 outliers. This decouples the
    # realistic (low, e.g. 2%) prevalence from the higher contamination used
    # during training/tuning.
    # Calibrate the absolute anomaly threshold from an OUT-OF-SAMPLE (not
    # in-sample) normal score distribution. IsolationForest scores its own
    # training points optimistically (shorter path lengths), so a percentile
    # of in-sample training-normal scores lands at a too-lenient cutoff and
    # inflates the false-positive rate on unseen normals. The model stores
    # out-of-fold raw score_samples() of normal training samples
    # (model.oof_normal_scores_): each normal was scored by a fold model that
    # did NOT see it, so this is an honest reference distribution.
    oof_scores = getattr(model, 'oof_normal_scores_', None)
    if oof_scores is not None and len(oof_scores) > 0:
        reference_scores = np.asarray(oof_scores)
        reference_source = f'{len(reference_scores)} out-of-fold normal scores (honest calibration)'
    elif X_normal_train is not None and len(X_normal_train) > 0:
        reference_scores = np.asarray(model.score_samples(X_normal_train))
        reference_source = f'{len(X_normal_train)} in-sample training-normal scores (optimistic; no OOF available)'
        logger.warning(
            "No out-of-fold normal scores available for threshold calibration; "
            "falling back to in-sample training-normal scores, which are "
            "optimistic and tend to inflate the false-positive rate."
        )
    else:
        reference_scores = np.asarray(model.score_samples(X_normal_test))
        reference_source = f'{len(X_normal_test)} test-normal scores (leaky fallback)'
        logger.warning(
            "No normal training samples or OOF scores for threshold calibration; "
            "falling back to test normals as the score reference (leaky)."
        )

    # score_samples: lower = more anomalous. Use raw (unshifted) scores so the
    # threshold is independent of any contamination set during model.fit().
    # Percentile: e.g. 2nd percentile => ~2% of reference normals fall below.
    anomaly_threshold = float(np.percentile(reference_scores, 100.0 * target_contamination))

    # IsolationForest scores each sample independently of the other samples
    # in the batch (the score is a function of the single row vs. the fitted
    # forest, not of the batch composition). Therefore a single deterministic
    # scoring pass over every test sample is fully representative of the
    # per-sample deployment behaviour, and no Monte-Carlo batch resampling is
    # needed to estimate detection or false-positive rates. Aggregate metrics
    # that mix the two classes (precision/F1/accuracy) depend on the batch's
    # abnormal-to-normal ratio, which differs between the (high-prevalence)
    # labelled test set and the (low-prevalence, ~target_contamination)
    # deployment scenario; those are computed analytically at the assumed
    # deployment prevalence below.

    # Score every test sample once.
    normal_scores = model.score_samples(X_normal_test)
    abnormal_scores = model.score_samples(X_abnormal_test) if n_abnormal > 0 else np.array([])

    # Per-sample flag decisions at the calibrated absolute threshold.
    normal_pred = (normal_scores <= anomaly_threshold).astype(int)
    abnormal_pred = (abnormal_scores <= anomaly_threshold).astype(int) if n_abnormal > 0 else np.array([], dtype=int)

    # Per-sample metrics (prevalence-independent, deployment-representative).
    n_detected = int(abnormal_pred.sum()) if n_abnormal > 0 else 0
    detection_rate = (n_detected / n_abnormal) if n_abnormal > 0 else float('nan')
    n_fp = int(normal_pred.sum())
    false_positive_rate = (n_fp / n_normal) if n_normal > 0 else float('nan')

    # ROC-AUC over the scored test set (ranking quality, prevalence-independent).
    try:
        if n_abnormal > 0:
            all_scores = np.concatenate([normal_scores, abnormal_scores])
            all_true = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_abnormal, dtype=int)])
            roc_auc = float(roc_auc_score(all_true, -all_scores))
        else:
            roc_auc = float('nan')
    except Exception:
        roc_auc = float('nan')

    # Aggregate metrics at the assumed deployment prevalence
    # (target_contamination). Given prevalence-independent recall (detection)
    # and FPR, the deployment precision/F1/accuracy follow analytically:
    #   precision = (p * recall) / (p * recall + (1-p) * fpr)
    # This avoids resampling a 2%-contaminated batch and is exact.
    p = target_contamination
    if n_abnormal > 0 and n_normal > 0:
        denom = (p * detection_rate) + ((1.0 - p) * false_positive_rate)
        precision_deploy = float((p * detection_rate) / denom) if denom > 0 else float('nan')
        recall_deploy = float(detection_rate)
        if (precision_deploy + recall_deploy) > 0:
            f1_deploy = float(2.0 * precision_deploy * recall_deploy / (precision_deploy + recall_deploy))
        else:
            f1_deploy = 0.0
        # Accuracy at deployment prevalence:
        #   P(correct) = (1-p)*(1-fpr) + p*recall
        accuracy_deploy = float((1.0 - p) * (1.0 - false_positive_rate) + p * detection_rate)
        # Expected confusion counts for a notional batch of n_test_total at p.
        n_outliers_batch = max(1, int(round(p * n_test_total)))
        n_normals_batch = n_test_total - n_outliers_batch
        cm = np.array([
            [int(round(n_normals_batch * (1.0 - false_positive_rate))), int(round(n_normals_batch * false_positive_rate))],
            [int(round(n_outliers_batch * (1.0 - detection_rate))), int(round(n_outliers_batch * detection_rate))],
        ])
    else:
        precision_deploy = float('nan')
        recall_deploy = float('nan')
        f1_deploy = float('nan')
        accuracy_deploy = float('nan')
        cm = np.array([[n_normal, 0], [0, n_abnormal]])

    # Per-sample results (one row per test sample) for downstream CSV/plots.
    per_sample_results = []
    for i in range(n_normal):
        per_sample_results.append({
            'sample_id': X_normal_test.index[i],
            'true_label': 0,
            'score': float(normal_scores[i]),
            'flagged': int(normal_pred[i]),
        })
    for i in range(n_abnormal):
        per_sample_results.append({
            'sample_id': X_abnormal_test.index[i],
            'true_label': 1,
            'score': float(abnormal_scores[i]),
            'flagged': int(abnormal_pred[i]),
        })

    logger.info(f"\n{'='*70}")
    logger.info("REALISTIC EVALUATION (absolute threshold, single scoring pass)")
    logger.info(f"{'='*70}")
    logger.info(f"Normal test samples: {n_normal}")
    logger.info(f"Abnormal test samples: {n_abnormal}")
    logger.info(f"Target (deployment) contamination: {target_contamination:.2%}")
    logger.info(f"Threshold calibration source: {reference_source}")
    logger.info(f"Anomaly threshold ({100.0*target_contamination:.4g}-th pct of normal scores): {anomaly_threshold:.6f}")
    logger.info(f"Detection rate (recall): {detection_rate:.2%}  ({n_detected}/{n_abnormal})")
    logger.info(f"False positive rate: {false_positive_rate:.2%}  ({n_fp}/{n_normal})")
    logger.info(f"ROC AUC (test ranking): {roc_auc:.4f}")
    logger.info(f"Precision @ {target_contamination:.2%} prevalence: {precision_deploy:.4f}")
    logger.info(f"F1 @ {target_contamination:.2%} prevalence: {f1_deploy:.4f}")
    logger.info(f"Accuracy @ {target_contamination:.2%} prevalence: {accuracy_deploy:.4f}")
    logger.info(f"Confusion matrix (notional batch of {n_test_total} at {target_contamination:.2%}):\n{cm}")
    logger.info(f"{'='*70}")

    results = {
        'evaluation_strategy': 'realistic',
        'scoring_mode': 'single_pass_absolute_threshold',
        'n_normal_test': n_normal,
        'n_abnormal_test': n_abnormal,
        'n_reference_normals': int(len(reference_scores)),
        'target_contamination': target_contamination,
        'anomaly_threshold': anomaly_threshold,
        'threshold_percentile': float(100.0 * target_contamination),
        'detection_rate': float(detection_rate),
        'false_positive_rate': float(false_positive_rate),
        'n_detected': n_detected,
        'n_false_positives': n_fp,
        'recall': float(recall_deploy),
        'precision': float(precision_deploy),
        'f1': float(f1_deploy),
        'accuracy': float(accuracy_deploy),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': ['Normal', 'Outlier'],
        'per_iteration_results': per_sample_results,
    }

    return results


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):
        return obj.item() if hasattr(obj, 'item') else float(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return int(obj) if isinstance(obj, np.integer) else float(obj)
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, pd.Index):
        return obj.tolist()
    else:
        return obj


def save_realistic_results(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Save realistic evaluation results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    serializable_results = _make_serializable(results)

    # Save aggregated metrics
    metrics_path = output_dir / "realistic_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Realistic metrics saved to {metrics_path}")

    # Save per-sample results as CSV
    if 'per_iteration_results' in results:
        iter_df = pd.DataFrame(results['per_iteration_results'])
        iter_path = output_dir / "realistic_per_sample_results.csv"
        iter_df.to_csv(iter_path, index=False)
        logger.info(f"Per-sample results saved to {iter_path}")

    # Save summary
    summary = {
        'detection_rate': results['detection_rate'],
        'false_positive_rate': results['false_positive_rate'],
        'precision': results.get('precision'),
        'f1': results.get('f1'),
        'roc_auc': results.get('roc_auc'),
        'anomaly_threshold': results.get('anomaly_threshold'),
        'n_normal_test': results['n_normal_test'],
        'n_abnormal_test': results['n_abnormal_test'],
        'n_reference_normals': results.get('n_reference_normals'),
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

    # Plot 1: Score distribution (normals vs abnormals) with the calibrated threshold
    plt.figure(figsize=(12, 6))

    rows = results.get('per_iteration_results', [])
    normal_scores = np.array([r['score'] for r in rows if r['true_label'] == 0])
    abnormal_scores = np.array([r['score'] for r in rows if r['true_label'] == 1])
    threshold = results.get('anomaly_threshold', None)

    plt.subplot(1, 2, 1)
    if len(normal_scores) > 0:
        plt.hist(normal_scores, bins=30, alpha=0.6, color='blue', label=f'Normal (n={len(normal_scores)})')
    if len(abnormal_scores) > 0:
        plt.hist(abnormal_scores, bins=30, alpha=0.6, color='red', label=f'Abnormal (n={len(abnormal_scores)})')
    if threshold is not None:
        plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                    label=f"Threshold: {threshold:.4f}")
    plt.xlabel('score_samples (lower = more anomalous)')
    plt.ylabel('Count')
    plt.title('Test score distribution & anomaly threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Flagged fractions
    plt.subplot(1, 2, 2)
    labels = ['Detection rate\n(abnormal)', 'False pos. rate\n(normal)']
    vals = [results.get('detection_rate', float('nan')), results.get('false_positive_rate', float('nan'))]
    plt.bar(labels, vals, color=['red', 'blue'], alpha=0.6)
    plt.ylim(0, 1)
    plt.ylabel('Rate')
    plt.title(f"Per-sample rates at {results.get('target_contamination', 0):.2%} threshold")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f'{v:.2%}', ha='center')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "realistic_detection_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Realistic evaluation plot saved to {output_dir / 'realistic_detection_plot.png'}")

    # Plot 2: Confusion Matrix for aggregated results
    if 'confusion_matrix' in results:
        cm = np.array(results['confusion_matrix'])
        labels = results.get('confusion_matrix_labels', ['Normal', 'Outlier'])

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix (Realistic Evaluation)\n'
                  f'Detection Rate: {results["detection_rate"]:.2%}, '
                  f'FPR: {results["false_positive_rate"]:.2%}')
        plt.tight_layout()
        plt.savefig(output_dir / "realistic_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Realistic confusion matrix saved to {output_dir / 'realistic_confusion_matrix.png'}")
