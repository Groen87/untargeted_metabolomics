"""
Fused anomaly detection for the pathway pipeline.

Two building blocks:

1. build_pathway_zsummary_features: per-pathway z-score summaries (signed mean
   z, mean |z|, max |z|, top-k mean |z|). Unlike -log10(p) these preserve
   magnitude and direction, giving the anomaly detector dynamic range.

2. run_fused_anomaly_detection: trains one detector per feature view (e.g.
   pathway z-summaries + metabolite-level z-scores), calibrates each view's
   scores against the training-normal distribution (z-score), and combines
   views with max() before threshold optimization. Calibration on controls
   keeps the false-positive rate in check while the second view adds
   detections among samples the first view misses.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_pathway_zsummary_features(
    zscores: pd.DataFrame,
    feature_to_pathway: pd.DataFrame,
    min_pathway_size: int = 3,
    topk: int = 3,
) -> pd.DataFrame:
    """
    Build per-pathway z-score summary features.

    For each pathway with at least min_pathway_size matched features, compute
    per sample:
      mean_z     : signed mean z (keeps direction of the disturbance)
      mean_abs_z : mean |z| (global block deviation)
      max_abs_z  : max |z| (single worst metabolite)
      topk_abs_z : mean of the top-k |z| (localized block deviation)

    Returns a DataFrame indexed by sample_id with one column per
    (pathway, statistic).
    """
    feature_cols = [c for c in zscores.columns]
    col_set = set(feature_cols)

    parts: List[pd.DataFrame] = []
    for pw_name, grp in feature_to_pathway.groupby("pathway_name"):
        feats = [f for f in grp["feature"].tolist() if f in col_set]
        if len(feats) < min_pathway_size:
            continue
        Z = zscores[feats].to_numpy(dtype=float)
        absZ = np.abs(Z)
        with np.errstate(invalid="ignore"):
            mean_z = np.nanmean(Z, axis=1)
            mean_abs = np.nanmean(absZ, axis=1)
            max_abs = np.nanmax(absZ, axis=1)
        # For top-k, treat NaN as 0 (no deviation) so sorting is well-defined
        absZ0 = np.where(np.isnan(absZ), 0.0, absZ)
        k_eff = max(1, min(topk, absZ0.shape[1]))
        topk_mean = np.sort(absZ0, axis=1)[:, -k_eff:].mean(axis=1)

        pw = pd.DataFrame(
            {
                f"{pw_name}::mean_z": mean_z,
                f"{pw_name}::mean_abs_z": mean_abs,
                f"{pw_name}::max_abs_z": max_abs,
                f"{pw_name}::topk_abs_z": topk_mean,
            },
            index=zscores.index,
        )
        parts.append(pw)

    if not parts:
        logger.warning("No pathways met min_pathway_size; z-summary view is empty")
        return pd.DataFrame(index=zscores.index)

    out = pd.concat(parts, axis=1)
    return out.fillna(0.0)


def _fit_view(
    X_train: np.ndarray,
    scorer_name: str,
    contamination: float,
    n_neighbors: int,
    n_estimators: int,
    random_state: int,
    pca_components=None,
):
    """Fit scaler (+ optional PCA) and detector on training normals for one view."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)

    pca = None
    if pca_components is not None:
        from sklearn.decomposition import PCA

        n_comp = pca_components
        if isinstance(n_comp, float) and n_comp <= 0:
            n_comp = None
        pca = PCA(n_components=n_comp, random_state=random_state)
        Xt = pca.fit_transform(Xt)
        logger.info(
            f"    PCA reduced view to {pca.n_components_} components "
            f"(explained variance: {float(pca.explained_variance_ratio_.sum()):.4f})"
        )

    if scorer_name == "lof":
        from sklearn.neighbors import LocalOutlierFactor

        model = LocalOutlierFactor(
            n_neighbors=n_neighbors, novelty=True, contamination="auto", n_jobs=-1
        )
        method_name = "Local Outlier Factor"
    elif scorer_name == "iforest":
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        method_name = "Isolation Forest"
    elif scorer_name == "mahalanobis":
        from sklearn.covariance import MinCovDet

        model = MinCovDet(random_state=random_state)
        method_name = "Mahalanobis Distance"
    else:
        raise ValueError(
            f"Unknown scorer: {scorer_name}. Use 'lof', 'iforest', or 'mahalanobis'"
        )

    model.fit(Xt)
    return scaler, pca, model, method_name


def _score_view(model, scorer_name: str, X: np.ndarray) -> np.ndarray:
    """Raw anomaly scores for one view (higher = more anomalous)."""
    if scorer_name == "lof":
        return -model.decision_function(X)
    if scorer_name == "iforest":
        return -model.score_samples(X)
    return model.mahalanobis(X)


def _transform_view(scaler, pca, X: np.ndarray) -> np.ndarray:
    Xt = scaler.transform(X)
    if pca is not None:
        Xt = pca.transform(Xt)
    return Xt


def _compute_metrics(y_true_normal: np.ndarray, y_pred_flagged: np.ndarray, y_scores=None) -> Dict:
    """Classification metrics. True = normal, flagged = predicted anomalous."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true_binary = (~y_true_normal).astype(int)
    y_pred_binary = y_pred_flagged.astype(int)

    metrics: Dict = {}
    try:
        metrics["accuracy"] = accuracy_score(y_true_binary, y_pred_binary)
    except Exception:
        metrics["accuracy"] = float("nan")
    try:
        metrics["precision"] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    except Exception:
        metrics["precision"] = float("nan")
    try:
        metrics["recall"] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    except Exception:
        metrics["recall"] = float("nan")
    try:
        metrics["f1"] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    except Exception:
        metrics["f1"] = float("nan")
    if y_scores is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true_binary, y_scores)
        except Exception:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = average_precision_score(y_true_binary, y_scores)
        except Exception:
            metrics["pr_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    try:
        metrics["confusion_matrix"] = confusion_matrix(y_true_binary, y_pred_binary).tolist()
    except Exception:
        metrics["confusion_matrix"] = None
    return metrics


def run_fused_anomaly_detection(
    feature_views: Dict[str, pd.DataFrame],
    normal_sample_ids: List,
    imd_sample_ids: List,
    gray_sample_ids: Optional[List] = None,
    scorer_name: str = "lof",
    contamination: float = 0.02,
    n_neighbors: int = 20,
    n_estimators: int = 100,
    random_state: int = 42,
    percentile: float = 95.0,
    train_ratio: float = 0.8,
    optimization_metric: str = "f1",
    max_contamination: float = 0.05,
    min_detection: float = 0.80,
    view_pca: Optional[Dict[str, object]] = None,
    target_contamination: float = 0.02,
) -> Dict:
    """
    Multi-view anomaly detection with control-calibrated score fusion.

    Per view: fit scaler/PCA/detector on TRAINING NORMALS only. Scores are
    calibrated to z-scores using the training-normal score distribution, then
    combined across views with max(). The decision threshold is optimized on
    the validation set (held-out normals + all IMDs), same methodology as
    run_anomaly_detection in pathway_analysis_clean.

    Args:
        feature_views: dict view_name -> DataFrame (samples x features),
            all sharing the same sample index.
        normal_sample_ids / imd_sample_ids / gray_sample_ids: sample labels.
        view_pca: dict view_name -> PCA n_components (None = no PCA for that
            view). Missing entries default to no PCA.

    Returns a dict shaped like run_anomaly_detection's return value so the
    pipeline's saving/logging code works unchanged, plus per-view diagnostics
    under 'views'.
    """
    from sklearn.model_selection import train_test_split

    from pathway_pipeline.pipeline.pathway_analysis_clean import (
        find_optimal_anomaly_threshold,
    )

    gray_sample_ids = list(gray_sample_ids or [])
    view_pca = view_pca or {}

    # Samples present in ALL views
    common_ids = None
    for view in feature_views.values():
        idx = set(view.index)
        common_ids = idx if common_ids is None else (common_ids & idx)
    common_ids = sorted(common_ids)

    normal_ids = [s for s in normal_sample_ids if s in common_ids]
    imd_ids = [s for s in imd_sample_ids if s in common_ids]
    gray_ids = [s for s in gray_sample_ids if s in common_ids]

    logger.info(
        f"\nFused anomaly detection: {len(feature_views)} views, "
        f"{len(normal_ids)} normals, {len(imd_ids)} IMDs, {len(gray_ids)} gray"
    )
    for name, view in feature_views.items():
        logger.info(f"  View '{name}': {view.shape[1]} features")

    if len(normal_ids) < 4:
        raise ValueError(
            f"Need at least 4 normal samples for the train/test split, got {len(normal_ids)}"
        )

    # ------------------------------------------------------------------
    # Split normals into train/test (shared across views)
    # ------------------------------------------------------------------
    train_ids, test_ids = train_test_split(
        normal_ids, train_size=train_ratio, random_state=random_state
    )
    logger.info(f"\nSplit normals into train/test: {len(train_ids)} train, {len(test_ids)} test")

    # ------------------------------------------------------------------
    # Fit each view on training normals, calibrate on train-normal scores
    # ------------------------------------------------------------------
    views_out: Dict[str, Dict] = {}
    method_name = None
    for name, view in feature_views.items():
        X_train = view.loc[train_ids].to_numpy(dtype=float)
        scaler, pca, model, method_name = _fit_view(
            X_train,
            scorer_name,
            contamination,
            n_neighbors,
            n_estimators,
            random_state,
            pca_components=view_pca.get(name, None),
        )
        train_scores = _score_view(model, scorer_name, _transform_view(scaler, pca, X_train))
        mu = float(np.mean(train_scores))
        sigma = float(np.std(train_scores))
        if sigma <= 0:
            sigma = 1.0
            logger.warning(f"  View '{name}': zero score variance on training normals")
        views_out[name] = {
            "scaler": scaler,
            "pca": pca,
            "model": model,
            "mu": mu,
            "sigma": sigma,
            "n_features": int(view.shape[1]),
        }
        logger.info(
            f"  View '{name}' trained; train-normal scores mu={mu:.4f}, sigma={sigma:.4f}"
        )

    def fused_scores(sample_ids: List) -> tuple:
        """(fused score, per-view calibrated z DataFrame) for given samples."""
        per_view = {}
        for name, vw in views_out.items():
            X = feature_views[name].loc[sample_ids].to_numpy(dtype=float)
            raw = _score_view(vw["model"], scorer_name, _transform_view(vw["scaler"], vw["pca"], X))
            per_view[name] = (raw - vw["mu"]) / vw["sigma"]
        zdf = pd.DataFrame(per_view, index=sample_ids)
        fused = zdf.max(axis=1).to_numpy(dtype=float)
        return fused, zdf

    # ------------------------------------------------------------------
    # Validation set: held-out test normals + all IMDs
    # ------------------------------------------------------------------
    val_ids = test_ids + imd_ids
    fused_val, zdf_val = fused_scores(val_ids)
    val_is_normal = np.array([True] * len(test_ids) + [False] * len(imd_ids))

    logger.info(
        f"\nValidation fused score statistics: "
        f"Min={fused_val.min():.4f}, Max={fused_val.max():.4f}, Mean={fused_val.mean():.4f}"
    )

    threshold_info = find_optimal_anomaly_threshold(
        scores=fused_val,
        is_normal=val_is_normal,
        max_contamination=max_contamination,
        min_detection=min_detection,
        optimization_metric=optimization_metric,
        n_thresholds=100,
    )
    threshold = threshold_info["optimal_threshold"]

    val_flagged = fused_val > threshold
    n_test_normals = len(test_ids)
    n_val_imds = len(imd_ids)
    test_normals_flagged = int(np.sum(val_flagged[:n_test_normals]))
    val_imds_flagged = int(np.sum(val_flagged[n_test_normals:]))
    val_detection_rate = val_imds_flagged / n_val_imds if n_val_imds else 0.0
    val_contamination_rate = test_normals_flagged / n_test_normals if n_test_normals else 0.0

    val_results = pd.DataFrame(
        {
            "sample_id": val_ids,
            "anomaly_score": fused_val,
            "is_normal": val_is_normal,
            "is_imd": ~val_is_normal,
            "flagged": val_flagged,
        }
    )
    for name in views_out:
        val_results[f"score_z::{name}"] = zdf_val[name].to_numpy()

    logger.info(f"\nOptimized threshold on validation set: {threshold:.4f}")
    logger.info(
        f"  Test normals flagged: {test_normals_flagged} / {n_test_normals} "
        f"({val_contamination_rate*100:.1f}%)"
    )
    logger.info(
        f"  IMDs flagged: {val_imds_flagged} / {n_val_imds} ({val_detection_rate*100:.1f}%)"
    )

    val_metrics = _compute_metrics(val_is_normal, val_flagged, fused_val)

    # ------------------------------------------------------------------
    # Production evaluation: same never-seen samples as the classic runner
    # (held-out test normals + all IMDs), scored independently
    # ------------------------------------------------------------------
    prod_normal_ids = list(test_ids)
    prod_imd_ids = list(imd_ids)
    prod_ids = prod_normal_ids + prod_imd_ids
    fused_prod, zdf_prod = fused_scores(prod_ids)
    prod_flagged = fused_prod > threshold

    n_prod_normals = len(prod_normal_ids)
    n_prod_imds = len(prod_imd_ids)
    n_normals_flagged = int(np.sum(prod_flagged[:n_prod_normals]))
    n_imds_flagged = int(np.sum(prod_flagged[n_prod_normals:]))
    detection_rate = n_imds_flagged / n_prod_imds if n_prod_imds else 0.0
    false_positive_rate = n_normals_flagged / n_prod_normals if n_prod_normals else 0.0
    flagged_prod_normal_ids = [
        prod_normal_ids[i] for i in range(n_prod_normals) if prod_flagged[i]
    ]
    flagged_prod_imd_ids = [
        prod_imd_ids[i] for i in range(n_prod_imds) if prod_flagged[n_prod_normals + i]
    ]

    logger.info(f"\nProduction Evaluation (held-out normals + all IMDs):")
    logger.info(
        f"  Normals flagged: {n_normals_flagged} / {n_prod_normals} "
        f"({false_positive_rate*100:.1f}%)"
    )
    logger.info(
        f"  IMDs flagged: {n_imds_flagged} / {n_prod_imds} ({detection_rate*100:.1f}%)"
    )

    # Analytical metrics at target contamination
    p = target_contamination
    recall = detection_rate
    fpr = false_positive_rate
    denom = (p * recall) + ((1.0 - p) * fpr)
    precision_deploy = (p * recall) / denom if denom > 0 else 0.0
    f1_deploy = (
        2.0 * precision_deploy * recall / (precision_deploy + recall)
        if (precision_deploy + recall) > 0
        else 0.0
    )
    accuracy_deploy = (1.0 - p) * (1.0 - fpr) + p * recall

    n_notional = max(100, n_prod_normals + n_prod_imds)
    n_outliers_notional = max(1, int(round(p * n_notional)))
    n_normals_notional = n_notional - n_outliers_notional
    cm_deploy = np.array(
        [
            [
                int(round(n_normals_notional * (1.0 - fpr))),
                int(round(n_normals_notional * fpr)),
            ],
            [
                int(round(n_outliers_notional * (1.0 - recall))),
                int(round(n_outliers_notional * recall)),
            ],
        ]
    )

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        all_prod_true = np.concatenate(
            [np.zeros(n_prod_normals, dtype=int), np.ones(n_prod_imds, dtype=int)]
        )
        roc_auc_prod = float(roc_auc_score(all_prod_true, fused_prod))
        pr_auc_prod = float(average_precision_score(all_prod_true, fused_prod))
    except Exception:
        roc_auc_prod = float("nan")
        pr_auc_prod = float("nan")

    logger.info(f"\nAnalytical Metrics @ {target_contamination*100:.0f}% prevalence:")
    logger.info(f"  Precision: {precision_deploy:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1: {f1_deploy:.4f}")
    logger.info(f"  Accuracy: {accuracy_deploy:.4f}")
    logger.info(f"  ROC AUC: {roc_auc_prod:.4f}")
    logger.info(f"  PR AUC: {pr_auc_prod:.4f}")

    prod_is_normal = np.array(
        [True] * n_prod_normals + [False] * n_prod_imds
    )
    prod_results = pd.DataFrame(
        {
            "sample_id": prod_ids,
            "anomaly_score": fused_prod,
            "is_normal": prod_is_normal,
            "is_imd": ~prod_is_normal,
            "flagged": prod_flagged,
        }
    )
    for name in views_out:
        prod_results[f"score_z::{name}"] = zdf_prod[name].to_numpy()

    prod_metrics = _compute_metrics(prod_is_normal, prod_flagged, fused_prod)
    prod_metrics["detection_rate"] = detection_rate
    prod_metrics["false_positive_rate"] = false_positive_rate

    # ------------------------------------------------------------------
    # Plots (same interface as the classic runner)
    # ------------------------------------------------------------------
    def _plot_cm(y_true_normal_arr, y_pred_flagged_arr, title, output_path):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix

            y_true_binary = (~y_true_normal_arr).astype(int)
            y_pred_binary = y_pred_flagged_arr.astype(int)
            cm = confusion_matrix(y_true_binary, y_pred_binary)

            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Normal", "IMD"],
                yticklabels=["Normal", "IMD"],
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            return True
        except Exception as e:
            logger.warning(f"Could not generate confusion matrix plot: {e}")
            return False

    view_list = " + ".join(feature_views.keys())
    if len(feature_views) > 1:
        method_out = f"Fused {method_name} (max of control-calibrated z over: {view_list})"
    else:
        method_out = f"{method_name} on {view_list} (control-calibrated z)"

    return {
        "scorer": scorer_name,
        "method": method_out,
        "threshold": threshold,
        "percentile": percentile,
        "train_ratio": train_ratio,
        "random_state": random_state,
        "validation": {
            "n_normals": n_test_normals,
            "n_imds": n_val_imds,
            "normals_flagged": test_normals_flagged,
            "imds_flagged": val_imds_flagged,
            "detection_rate": val_detection_rate,
            "contamination_rate": val_contamination_rate,
            "flagged_normal_ids": [val_ids[i] for i in range(n_test_normals) if val_flagged[i]],
            "flagged_imd_ids": [
                val_ids[n_test_normals + i] for i in range(n_val_imds)
                if val_flagged[n_test_normals + i]
            ],
            "results": val_results,
            "metrics": val_metrics,
        },
        "production": {
            "n_normals": n_prod_normals,
            "n_imds": n_prod_imds,
            "normals_flagged": n_normals_flagged,
            "imds_flagged": n_imds_flagged,
            "detection_rate": detection_rate,
            "false_positive_rate": false_positive_rate,
            "flagged_normal_ids": flagged_prod_normal_ids,
            "flagged_imd_ids": flagged_prod_imd_ids,
            "results": prod_results,
            "metrics": prod_metrics,
            "target_contamination": target_contamination,
            "precision_at_target": precision_deploy,
            "f1_at_target": f1_deploy,
            "accuracy_at_target": accuracy_deploy,
            "roc_auc": roc_auc_prod,
            "pr_auc": pr_auc_prod,
            "confusion_matrix_at_target": cm_deploy.tolist(),
            "confusion_matrix_labels": ["Normal", "IMD"],
        },
        "all_samples": {
            "normal_sample_ids": normal_ids,
            "imd_sample_ids": imd_ids,
            "gray_sample_ids": gray_ids,
            "n_features": {name: vw["n_features"] for name, vw in views_out.items()},
        },
        "views": {
            name: {
                "n_features": vw["n_features"],
                "train_score_mu": vw["mu"],
                "train_score_sigma": vw["sigma"],
            }
            for name, vw in views_out.items()
        },
        "plot_functions": {
            "plot_validation_cm": lambda output_dir: _plot_cm(
                val_is_normal,
                val_flagged,
                f"{method_out} - Validation Set",
                str(Path(output_dir) / "anomaly_validation_confusion_matrix.png"),
            ),
            "plot_production_cm": lambda output_dir: _plot_cm(
                prod_is_normal,
                prod_flagged,
                f"{method_out} - Production Evaluation",
                str(Path(output_dir) / "anomaly_production_confusion_matrix.png"),
            ),
            "plot_analytical_cm": lambda output_dir: _plot_precomputed_cm(
                cm_deploy,
                f"{method_out} - Analytical @ {target_contamination*100:.0f}% prevalence",
                str(Path(output_dir) / "anomaly_analytical_confusion_matrix.png"),
            ),
        },
    }


def _plot_precomputed_cm(cm, title, output_path):
    """Plot a precomputed confusion matrix; returns True on success."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "IMD"],
            yticklabels=["Normal", "IMD"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        logger.warning(f"Could not generate analytical confusion matrix plot: {e}")
        return False
