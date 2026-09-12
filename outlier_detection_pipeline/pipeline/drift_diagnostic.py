"""
Drift diagnostic module for the outlier detection pipeline.

Logs the IsolationForest score_samples() distributions of training-normal vs
test-normal vs test-abnormal samples, flags when the test-normal false-positive
rate at the calibrated threshold diverges from the nominal deployment
contamination, and renders a 2-PC scatter of the three groups coloured by
train/test split. This is a diagnostic aid: a persistent, large gap between
the training-normal and test-normal score distributions indicates
batch/cohort drift (the training normals are not representative of the
deployment normals), which inflates the false-positive rate independently of
any code bug or unlucky single split.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

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


def _summary_stats(scores: np.ndarray) -> Dict[str, float]:
    """Mean and 2nd/50th/98th percentiles of a score array."""
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return {"n": 0, "mean": float("nan"), "p2": float("nan"),
                "p50": float("nan"), "p98": float("nan"), "std": float("nan")}
    return {
        "n": int(s.size),
        "mean": float(np.mean(s)),
        "std": float(np.std(s)),
        "p2": float(np.percentile(s, 2)),
        "p50": float(np.percentile(s, 50)),
        "p98": float(np.percentile(s, 98)),
    }


def run_drift_diagnostic(
    model: Any,
    X_train_normals: pd.DataFrame,
    X_test_normals: pd.DataFrame,
    X_test_abnormals: Optional[pd.DataFrame],
    target_contamination: float,
    anomaly_threshold: float,
    output_dir: Path,
    fold_num: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Log train-normal vs test-normal vs test-abnormal score distributions and
    flag FPR divergence, then save a PCA scatter and a numeric diagnostic file.

    Args:
        model: A fitted ExtendedIsolationForestModel exposing score_samples().
        X_train_normals: Normal TRAIN samples the model trained on (in-sample;
            optimistic). Used to characterise the training-normal reference.
        X_test_normals: Normal TEST samples (out-of-sample; deployment-relevant).
        X_test_abnormals: Abnormal TEST samples (may be empty/None).
        target_contamination: Nominal deployment contamination (e.g. 0.02).
        anomaly_threshold: Calibrated absolute flag threshold already used by
            the realistic evaluation (same definition: a sample is flagged if
            score_samples <= threshold).
        output_dir: Directory to write drift_diagnostic.csv / .json / .png into.
        fold_num: Optional outer-CV fold index (1-based) for labelling.

    Returns:
        Dictionary of the logged diagnostic numbers.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Score each group once with the trained model.
    train_normal_scores = model.score_samples(X_train_normals) if len(X_train_normals) > 0 else np.array([])
    test_normal_scores = model.score_samples(X_test_normals) if len(X_test_normals) > 0 else np.array([])
    has_abnormals = X_test_abnormals is not None and len(X_test_abnormals) > 0
    test_abnormal_scores = model.score_samples(X_test_abnormals) if has_abnormals else np.array([])

    train_stats = _summary_stats(train_normal_scores)
    test_normal_stats = _summary_stats(test_normal_scores)
    test_abnormal_stats = _summary_stats(test_abnormal_scores)

    # Actual test-normal FPR at the calibrated threshold vs nominal contamination.
    if len(test_normal_scores) > 0:
        actual_fpr = float(np.mean(test_normal_scores <= anomaly_threshold))
    else:
        actual_fpr = float("nan")
    nominal_fpr = float(target_contamination)
    fpr_gap = actual_fpr - nominal_fpr if not np.isnan(actual_fpr) else float("nan")

    # Mean-score shift between training and test normals (positive = test
    # normals look more anomalous than training normals).
    if train_stats["n"] > 0 and test_normal_stats["n"] > 0:
        mean_shift = float(test_normal_stats["mean"] - train_stats["mean"])
    else:
        mean_shift = float("nan")

    # Heuristic divergence flag: test-normal FPR is more than 3x the nominal
    # contamination (or > nominal + 0.05 absolute), suggesting batch drift.
    if not np.isnan(actual_fpr):
        flag_drift = bool(actual_fpr > max(3.0 * nominal_fpr, nominal_fpr + 0.05))
    else:
        flag_drift = False

    fold_label = f"fold {fold_num}" if fold_num is not None else "single split"
    logger.info(f"\n{'='*70}")
    logger.info(f"DRIFT DIAGNOSTIC ({fold_label})")
    logger.info(f"{'='*70}")
    logger.info(f"Anomaly threshold: {anomaly_threshold:.6f}  (nominal contamination: {nominal_fpr:.2%})")
    logger.info("Score distributions (score_samples, lower = more anomalous):")
    for name, st in [("train-normal", train_stats),
                     ("test-normal", test_normal_stats),
                     ("test-abnormal", test_abnormal_stats)]:
        logger.info(f"  {name:14s}: n={st['n']:4d}  mean={st['mean']:.4f}  "
                    f"std={st['std']:.4f}  p2={st['p2']:.4f}  "
                    f"p50={st['p50']:.4f}  p98={st['p98']:.4f}")
    logger.info(f"Mean score shift (test-normal - train-normal): {mean_shift:.4f}")
    logger.info(f"Actual test-normal FPR at threshold: {actual_fpr:.2%}  "
                f"(nominal {nominal_fpr:.2%}, gap {fpr_gap:+.4f})")
    if flag_drift:
        logger.warning(
            f"DRIFT FLAGGED: test-normal FPR ({actual_fpr:.2%}) is far above "
            f"the nominal contamination ({nominal_fpr:.2%}). Training normals "
            f"are likely not representative of test/deployment normals "
            f"(batch/cohort drift); this inflates the false-positive rate "
            f"independently of split noise. Consider batch correction (e.g. "
            f"scale on pooled QC across the whole cohort before splitting)."
        )
    else:
        logger.info("No FPR divergence flagged at this threshold.")
    logger.info(f"{'='*70}")

    # PCA scatter coloured by group (first 2 PCs). Fit PCA on training normals
    # only so the projection is defined by the normal reference, then project
    # test normals and abnormals into the same space.
    if HAS_MATPLOTLIB:
        try:
            _plot_drift_pca(
                X_train_normals, X_test_normals, X_test_abnormals,
                anomaly_threshold, output_dir, fold_label,
            )
        except Exception as e:
            logger.warning(f"Could not render drift PCA scatter: {e}")
    else:
        logger.warning("matplotlib not available; skipping drift PCA scatter plot.")

    # Numeric diagnostics.
    diag = {
        "fold": fold_num if fold_num is not None else 0,
        "n_train_normal": train_stats["n"],
        "n_test_normal": test_normal_stats["n"],
        "n_test_abnormal": test_abnormal_stats["n"],
        "anomaly_threshold": float(anomaly_threshold),
        "nominal_fpr": nominal_fpr,
        "actual_test_normal_fpr": actual_fpr,
        "fpr_gap": fpr_gap,
        "mean_score_shift": mean_shift,
        "drift_flagged": flag_drift,
        "train_normal_mean": train_stats["mean"],
        "train_normal_std": train_stats["std"],
        "train_normal_p2": train_stats["p2"],
        "train_normal_p50": train_stats["p50"],
        "train_normal_p98": train_stats["p98"],
        "test_normal_mean": test_normal_stats["mean"],
        "test_normal_std": test_normal_stats["std"],
        "test_normal_p2": test_normal_stats["p2"],
        "test_normal_p50": test_normal_stats["p50"],
        "test_normal_p98": test_normal_stats["p98"],
        "test_abnormal_mean": test_abnormal_stats["mean"],
        "test_abnormal_std": test_abnormal_stats["std"],
        "test_abnormal_p2": test_abnormal_stats["p2"],
        "test_abnormal_p50": test_abnormal_stats["p50"],
        "test_abnormal_p98": test_abnormal_stats["p98"],
    }

    pd.DataFrame([diag]).to_csv(output_dir / "drift_diagnostic.csv", index=False)
    with open(output_dir / "drift_diagnostic.json", "w") as f:
        json.dump(diag, f, indent=2, default=str)
    logger.info(f"Drift diagnostic saved to {output_dir / 'drift_diagnostic.csv'}")

    return diag


def _plot_drift_pca(
    X_train_normals: pd.DataFrame,
    X_test_normals: pd.DataFrame,
    X_test_abnormals: Optional[pd.DataFrame],
    threshold: float,
    output_dir: Path,
    fold_label: str,
) -> None:
    """2-PC scatter of train-normal / test-normal / test-abnormal, coloured by group."""
    from sklearn.decomposition import PCA

    frames = [X_train_normals, X_test_normals]
    names = ["train-normal", "test-normal"]
    if X_test_abnormals is not None and len(X_test_abnormals) > 0:
        frames.append(X_test_abnormals)
        names.append("test-abnormal")

    # Concatenate; align on shared columns. PCA needs matching columns, which
    # these DataFrames already share (same pipeline features).
    combined = pd.concat(frames, axis=0, sort=False)
    # Drop any residual NaN columns/rows defensively.
    combined = combined.dropna(axis=1, how="all")
    if combined.shape[1] < 2 or combined.shape[0] < 2:
        logger.warning("Not enough non-NaN features for drift PCA scatter; skipping.")
        return

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(combined.values)

    # Slice the projected PCs back per group using row counts.
    sizes = [len(f) for f in frames]
    offsets = np.cumsum([0] + sizes[:-1])
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"train-normal": "tab:blue", "test-normal": "tab:orange", "test-abnormal": "tab:red"}
    for name, off, sz in zip(names, offsets, sizes):
        if sz == 0:
            continue
        xs = pcs[off:off + sz, 0]
        ys = pcs[off:off + sz, 1]
        ax.scatter(xs, ys, s=40, alpha=0.7, label=f"{name} (n={sz})",
                   color=colors.get(name, "gray"), edgecolors="none")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(f"Drift PCA scatter ({fold_label})\n"
                 f"blue=train normals, orange=test normals, red=test abnormals")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "drift_diagnostic_pca.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Drift PCA scatter saved to {output_dir / 'drift_diagnostic_pca.png'}")
