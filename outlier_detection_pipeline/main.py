#!/usr/bin/env python3
"""
Main entry point for the outlier detection pipeline.

This pipeline performs Extended Isolation Forest outlier detection on
merged_data_with_classification.csv with cross-validation.

Data structure:
- Input CSV has patient IDs as rows and features as columns
- Two non-feature columns: 'Oordeel targeted' and 'Classification'
- Classification 0 = normal (used for training)
- Classification 1, 2, 3 = outliers (split between validation and test)

Workflow:
1. Load data from merged_data_with_classification.csv
2. Split data into train and test sets
3. Optional PCA for dimensionality reduction
4. Train Extended Isolation Forest on training set with CV
5. Evaluate on test set
6. Save predictions, metrics, and model

Usage:
    python outlier_detection_pipeline/main.py
    python outlier_detection_pipeline/main.py --config config/custom.yaml
    python outlier_detection_pipeline/main.py --input data/my_data.csv --output outputs/my_run
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from outlier_detection_pipeline.config.config import Config
from outlier_detection_pipeline.pipeline.data_loader import load_data, split_data, get_class_distribution
from outlier_detection_pipeline.pipeline.model import ExtendedIsolationForestModel
from outlier_detection_pipeline.pipeline.pca import SparsePCAWrapper
from outlier_detection_pipeline.pipeline.evaluation import (
    evaluate_model,
    print_metrics,
    save_metrics,
    save_predictions,
    plot_confusion_matrix,
    plot_precision_recall_curve,
)
from outlier_detection_pipeline.pipeline.outlier_analysis import (
    analyze_outliers,
    plot_outlier_analysis,
)
from outlier_detection_pipeline.pipeline.realistic_evaluation import (
    run_realistic_evaluation,
    save_realistic_results,
    plot_realistic_results,
)
from outlier_detection_pipeline.pipeline.drift_diagnostic import run_drift_diagnostic
from outlier_detection_pipeline.pipeline.hyperparameter_tuning import (
    tune_and_train,
    tune_ae,
    tune_hyperparameters,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def _setup_logging(output_dir: Path) -> None:
    """Setup file logging in addition to stream logging."""
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(logs_dir / "outlier_detection.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)


def _log_section_header(title: str) -> None:
    """Log a section header."""
    logger.info(f"\n{'='*70}")
    logger.info(title)
    logger.info(f"{'='*70}")


def _apply_feature_filter(
    features: pd.DataFrame,
    feature_filter: Optional[str],
) -> pd.DataFrame:
    """Apply feature filtering based on configuration."""
    if not feature_filter:
        return features

    original_n_features = len(features.columns)

    if feature_filter == 'hmdb':
        features = features[[col for col in features.columns if 'HMDB' in col]]
    elif isinstance(feature_filter, str):
        features = features[[col for col in features.columns if feature_filter in col]]
    elif isinstance(feature_filter, list):
        features = features[[col for col in features.columns if any(s in col for s in feature_filter)]]

    n_filtered = original_n_features - len(features.columns)
    logger.info(f"Filtered features: {n_filtered} removed, {len(features.columns)} remaining (filter: {feature_filter})")

    return features


def _handle_nan_values(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    nan_strategy: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Handle NaN values in data based on configured strategy."""
    nan_count_train = X_train.isna().sum().sum()
    nan_count_test = X_test.isna().sum().sum()
    total_nan = nan_count_train + nan_count_test

    if total_nan == 0:
        return X_train, X_test, y_train, y_test

    logger.warning(f"Found {total_nan} NaN values in data. Strategy: {nan_strategy}")

    if nan_strategy == 'drop_columns':
        cols_with_nan_train = set(X_train.columns[X_train.isna().any()])
        cols_with_nan_test = set(X_test.columns[X_test.isna().any()])
        cols_with_nan = list(cols_with_nan_train | cols_with_nan_test)
        n_dropped = len(cols_with_nan)
        X_train = X_train.drop(columns=cols_with_nan)
        X_test = X_test.drop(columns=cols_with_nan)
        logger.warning(f"Dropped {n_dropped} columns with NaN values: {cols_with_nan[:5]}{'...' if len(cols_with_nan) > 5 else ''}")

    elif nan_strategy == 'drop_rows':
        rows_with_nan_train = X_train.index[X_train.isna().any(axis=1)].tolist()
        rows_with_nan_test = X_test.index[X_test.isna().any(axis=1)].tolist()
        n_dropped = len(rows_with_nan_train) + len(rows_with_nan_test)
        X_train = X_train.dropna(axis=0)
        y_train = y_train[X_train.index]
        X_test = X_test.dropna(axis=0)
        y_test = y_test[X_test.index]
        logger.warning(f"Dropped {n_dropped} rows with NaN values")

    elif nan_strategy == 'impute_mean':
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_test.mean())
        logger.warning("Imputed NaN values with column means (separately for train/test)")

    else:
        raise ValueError(f"Unknown nan_strategy: {nan_strategy}. Use 'drop_columns', 'drop_rows', or 'impute_mean'.")

    logger.info(f"Train shape after NaN handling: {X_train.shape}")
    logger.info(f"Test shape after NaN handling: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def _apply_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: Config,
    output_dir: Path,
    normal_class: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, SparsePCAWrapper]:
    """Apply PCA dimensionality reduction if configured."""
    use_sparse_pca = config.get('use_sparse_pca', False)

    if not use_sparse_pca:
        return X_train, X_test, y_train, y_test, None

    _log_section_header("PCA Dimensionality Reduction")

    n_components = config.get('n_components', 100)
    alpha = config.get('alpha', 1.0)
    max_iter = config.get('max_iter', 1000)
    pca_random_state = config.get('pca_random_state', 42)
    save_pca_model = config.get('save_pca_model', True)
    nan_strategy = config.get('pca_nan_strategy', 'drop_columns')
    pca_method = config.get('pca_method', 'sparse')
    batch_size = config.get('pca_batch_size', 1000)
    intermediate_components = config.get('pca_intermediate_components', None)

    # Handle NaN values before PCA
    X_train, X_test, y_train, y_test = _handle_nan_values(
        X_train, X_test, y_train, y_test, nan_strategy
    )

    pca = SparsePCAWrapper(
        n_components=n_components,
        alpha=alpha,
        max_iter=max_iter,
        random_state=pca_random_state,
        method=pca_method,
        batch_size=batch_size,
        intermediate_components=intermediate_components,
    )

    # Fit PCA on NORMAL training data only (no data leakage from abnormalities)
    X_train_normals = X_train[y_train == normal_class]
    pca.fit(X_train_normals)

    # Transform normals using the fitted PCA
    X_train_transformed = pca.transform(X_train_normals)

    # Transform abnormalities if they exist
    X_train_abnormals_mask = (y_train != normal_class)
    if X_train_abnormals_mask.any():
        X_train_abnormals = pca.transform(X_train[X_train_abnormals_mask])
        # Combine back for training (normals first, then abnormalities)
        X_train = pd.concat([X_train_transformed, X_train_abnormals])
        y_train = pd.concat([y_train[~X_train_abnormals_mask], y_train[X_train_abnormals_mask]])
    else:
        X_train = X_train_transformed
        y_train = y_train[y_train == normal_class]

    # Transform test set
    X_test = pca.transform(X_test)

    logger.info(f"Features reduced from original to {X_train.shape[1]} components")

    if save_pca_model:
        pca.save(output_dir / "pca_model.joblib")

    return X_train, X_test, y_train, y_test, pca


def _train_with_hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Config,
    output_dir: Path,
    normal_class: int,
) -> Tuple[ExtendedIsolationForestModel, Dict[str, Any]]:
    """Train model with hyperparameter tuning."""
    _log_section_header("Hyperparameter Tuning with CV")

    param_grid = config.get('param_grid', None)
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_samples': ['auto', 0.5, 0.8],
            'max_features': [0.5, 0.8, 1.0],
            'contamination': ['auto'],
            'bootstrap': [False, True],
        }

    n_jobs = config.get('n_jobs', -1)
    random_state = config.get('random_state', 42)
    n_splits_tuning = config.get('n_splits_tuning', 5)
    tuning_scoring = config.get('tuning_scoring', 'pr_auc')

    # Optuna configuration
    use_optuna = config.get('use_optuna', False)
    n_trials = config.get('n_trials', 100)
    optuna_sampler = config.get('optuna_sampler', 'tpe')
    optuna_pruner = config.get('optuna_pruner', 'median')
    optuna_storage_url = config.get('optuna_storage_url', None)
    optuna_study_name = config.get('optuna_study_name', None)

    # Run hyperparameter tuning
    best_model, scaler, best_params, tuning_results = tune_and_train(
        X_train=X_train,
        y_train=y_train,
        normal_classification=normal_class,
        param_grid=param_grid,
        n_splits=n_splits_tuning,
        random_state=random_state,
        n_jobs=n_jobs,
        scoring=tuning_scoring,
        output_dir=output_dir,
        use_optuna=use_optuna,
        n_trials=n_trials,
        optuna_sampler=optuna_sampler,
        optuna_pruner=optuna_pruner,
        study_name=optuna_study_name,
        storage_url=optuna_storage_url,
    )

    # Create ExtendedIsolationForestModel wrapper with best parameters
    model = ExtendedIsolationForestModel(
        n_estimators=best_params['n_estimators'],
        max_samples=best_params['max_samples'],
        max_features=best_params['max_features'],
        bootstrap=best_params['bootstrap'],
        n_jobs=n_jobs,
        random_state=random_state,
        contamination=best_params.get('contamination', 'auto'),
    )
    model.model = best_model
    model.scaler = scaler
    model.is_fitted_ = True

    logger.info("\nBest hyperparameters found:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")

    # Save tuning results
    tuning_results.to_csv(output_dir / "tuning_results.csv", index=False)
    logger.info(f"Tuning results saved to {output_dir / 'tuning_results.csv'}")

    return model, best_params


def _train_ae_with_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Config,
    output_dir: Path,
    normal_class: int,
) -> ExtendedIsolationForestModel:
    """Tune the AE scorer's hyperparameters and wrap the best model.

    Mirrors _train_with_hyperparameter_tuning but uses the AE-specific grid
    search (tune_ae), which sweeps latent_dim / hidden_dim / epochs / lr via
    make_scorer('ae', ...). The fitted AE scorer is dropped into the
    ExtendedIsolationForestModel wrapper so the rest of the pipeline
    (threshold calibration, realistic eval) runs unchanged.
    """
    _log_section_header("AE Hyperparameter Tuning with CV")

    param_grid = config.get('ae_param_grid', None)
    random_state = config.get('random_state', 42)
    n_splits_tuning = config.get('n_splits_tuning', 5)
    tuning_scoring = config.get('tuning_scoring', 'pr_auc')
    n_jobs = config.get('n_jobs', -1)
    contamination = config.get('contamination', 'auto')

    best_model, scaler, best_params, tuning_results = tune_ae(
        X=X_train,
        y=y_train,
        normal_classification=normal_class,
        param_grid=param_grid,
        n_splits=n_splits_tuning,
        random_state=random_state,
        scoring=tuning_scoring,
        refit=True,
    )

    scorer_kwargs = dict(best_params)
    model = ExtendedIsolationForestModel(
        n_jobs=n_jobs,
        random_state=random_state,
        contamination=contamination,
        scorer_name='ae',
        scorer_kwargs=scorer_kwargs,
    )
    model.model = best_model
    model.scaler = scaler
    model.is_fitted_ = True

    logger.info("\nBest AE hyperparameters found:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")

    tuning_results.to_csv(output_dir / "tuning_results.csv", index=False)
    logger.info(f"Tuning results saved to {output_dir / 'tuning_results.csv'}")

    return model


def _train_without_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Config,
    normal_class: int,
) -> ExtendedIsolationForestModel:
    """Train model without hyperparameter tuning using config defaults."""
    _log_section_header("Training with CV (train on normals, validate on full)")

    n_estimators = config.get('n_estimators', 100)
    max_samples = config.get('max_samples', 'auto')
    max_features = config.get('max_features', 1.0)
    bootstrap = config.get('bootstrap', False)
    n_jobs = config.get('n_jobs', -1)
    random_state = config.get('random_state', 42)
    contamination = config.get('contamination', 'auto')
    n_splits = config.get('n_splits', 5)
    scorer_name = config.get('scorer', 'iforest')
    # Extra per-scorer kwargs (e.g. ocsvm nu/gamma, pca_recon n_components).
    scorer_kwargs = config.get('scorer_kwargs', None)

    model = ExtendedIsolationForestModel(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=random_state,
        contamination=contamination,
        scorer_name=scorer_name,
        scorer_kwargs=scorer_kwargs,
    )

    # Train with cross-validation
    cv_preds_train, train_scores, fold_scores = model.cross_val_predict(
        X=X_train,
        y=y_train,
        normal_classification=normal_class,
        n_splits=n_splits,
    )

    logger.info("Training with cross-validation complete.")
    logger.info(f"Final model trained on all {len(y_train[y_train == normal_class])} normal samples from train set")

    return model


def _train_without_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Config,
    normal_class: int,
    reference_normals: Optional[pd.DataFrame] = None,
) -> ExtendedIsolationForestModel:
    """Train the model once (no inner k-fold CV).

    Fits the scaler and the scorer on the training normals only, in a single
    pass. When `reference_normals` is provided (an out-of-sample set of
    normals the model never trained on, e.g. the held-out confident-normal
    test slice), their raw score_samples() are stored as the honest
    `oof_normal_scores_` reference distribution for absolute-threshold
    calibration. When omitted, `oof_normal_scores_` is left None and the
    realistic eval falls back to in-sample training-normal scores (with a
    warning that this is optimistic).
    """
    _log_section_header("Training (single fit, no inner CV)")
    n_estimators = config.get('n_estimators', 100)
    max_samples = config.get('max_samples', 'auto')
    max_features = config.get('max_features', 1.0)
    bootstrap = config.get('bootstrap', False)
    n_jobs = config.get('n_jobs', -1)
    random_state = config.get('random_state', 42)
    contamination = config.get('contamination', 'auto')
    scorer_name = config.get('scorer', 'iforest')
    scorer_kwargs = config.get('scorer_kwargs', None)

    model = ExtendedIsolationForestModel(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=random_state,
        contamination=contamination,
        scorer_name=scorer_name,
        scorer_kwargs=scorer_kwargs,
    )

    model.fit(X_train, y_train, normal_classification=normal_class)

    n_train_normal = int((y_train == normal_class).sum())
    logger.info(f"Model trained on {n_train_normal} normal samples (single fit, "
                f"no inner k-fold CV).")

    if reference_normals is not None and len(reference_normals) > 0:
        model.oof_normal_scores_ = np.asarray(model.score_samples(reference_normals))
        logger.info(f"Calibrating absolute threshold from {len(model.oof_normal_scores_)} "
                    f"out-of-sample (held-out) normal scores.")
    else:
        model.oof_normal_scores_ = None
        logger.warning("No out-of-sample reference normals provided; threshold "
                       "calibration will fall back to in-sample training-normal "
                       "scores, which are optimistic and tend to inflate FPR.")

    return model


def _evaluate_realistic(
    model: ExtendedIsolationForestModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Config,
    output_dir: Path,
    normal_class: int,
    outlier_classes: List[int],
    group_map: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run realistic evaluation."""
    _log_section_header("Realistic Evaluation (LOO Abnormal)")

    realistic_contamination = config.get('realistic_test_contamination', 0.02)
    realistic_n_iterations = config.get('realistic_n_iterations', 50)
    random_state = config.get('random_state', 42)
    save_realistic_results_flag = config.get('save_realistic_results', True)
    save_plots = config.get('save_plots', True)

    # Separate test set into normal and abnormal
    X_test_normal = X_test[y_test == normal_class]
    y_test_normal = y_test[y_test == normal_class]
    X_test_abnormal = X_test[y_test.isin(outlier_classes)]
    y_test_abnormal = y_test[y_test.isin(outlier_classes)]

    # Use the CV-trained model (already trained on normals only)
    model_final = model

    # Run realistic evaluation
    realistic_results = run_realistic_evaluation(
        model=model_final,
        X_normal_test=X_test_normal,
        X_abnormal_test=X_test_abnormal,
        y_normal_test=y_test_normal,
        y_abnormal_test=y_test_abnormal,
        target_contamination=realistic_contamination,
        n_iterations=realistic_n_iterations,
        random_seed=random_state,
        outlier_classes=outlier_classes,
        X_normal_train=X_train[y_train == normal_class],
        y_normal_train=y_train[y_train == normal_class],
        group_map=group_map,
    )

    # Save realistic results
    if save_realistic_results_flag:
        save_realistic_results(realistic_results, output_dir)
    
    # Save realistic plots (always save if plots are enabled)
    if save_plots:
        plot_realistic_results(realistic_results, output_dir)

    # Also run standard evaluation for comparison
    test_preds = model_final.predict(X_test)
    test_scores = model_final.decision_function(X_test)

    # Drift diagnostic: compare train-normal vs test-normal vs test-abnormal
    # score distributions and flag FPR divergence from the nominal
    # contamination. Uses the SAME threshold the realistic eval flagged with
    # so the reported FPR is directly comparable. Defaults to on; a batch
    # where train and test normals are drawn from the same distribution shows
    # no drift.
    if config.get('run_drift_diagnostic', True):
        run_drift_diagnostic(
            model=model_final,
            X_train_normals=X_train[y_train == normal_class],
            X_test_normals=X_test_normal,
            X_test_abnormals=X_test_abnormal if len(X_test_abnormal) > 0 else None,
            target_contamination=realistic_contamination,
            anomaly_threshold=realistic_results.get('anomaly_threshold', float('nan')),
            output_dir=output_dir,
            fold_num=None,
        )

    # Per-group breakdown by (raw Classification, Oordeel) at deployment prevalence.
    if group_map is not None and len(group_map) > 0:
        _per_group_breakdown(
            per_sample_results=realistic_results.get('per_iteration_results', []),
            anomaly_threshold=float(realistic_results.get('anomaly_threshold', float('nan'))),
            group_map=group_map,
            target_contamination=realistic_contamination,
            output_dir=output_dir,
            label="",
        )

    return test_preds, test_scores, realistic_results


def _evaluate_standard(
    model: ExtendedIsolationForestModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Config,
    normal_class: int,
    outlier_classes: List[int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run standard evaluation."""
    _log_section_header("Evaluating on test set")

    test_scores = model.decision_function(X_test)

    # Calculate test set contamination for proper threshold
    test_contamination = (y_test != normal_class).mean()
    logger.info(f"Test set contamination: {test_contamination:.2%}")

    # Use score-based threshold matching test contamination
    n_outliers_expected = int(np.round(test_contamination * len(X_test)))
    if n_outliers_expected > 0:
        sorted_scores = np.sort(test_scores)
        threshold_idx = min(n_outliers_expected - 1, len(sorted_scores) - 1)
        threshold = sorted_scores[threshold_idx]
        test_preds = np.where(test_scores <= threshold, -1, 1)
    else:
        test_preds = model.predict(X_test)

    logger.info(f"Flagging {np.sum(test_preds == -1)} outliers (expected ~{n_outliers_expected})")

    # Compute metrics for standard evaluation
    metrics_list = config.get_list('metrics', ['accuracy', 'f1', 'f1_weighted', 'precision', 'recall', 'roc_auc', 'confusion_matrix'])
    test_metrics = evaluate_model(
        y_true=y_test,
        y_pred=test_preds,
        y_scores=test_scores,
        metrics=metrics_list,
        pos_label=-1,
        outlier_classes=outlier_classes,
    )
    print_metrics(test_metrics)

    return test_preds, test_scores, test_metrics


def _save_false_negatives_csv(
    realistic_results: Optional[Dict[str, Any]],
    output_dir: Path,
    fold: Optional[int] = None,
    group_map: Optional[pd.DataFrame] = None,
) -> None:
    """Save CSVs of the realistic-eval false negatives and false positives.

    Roles follow the lab protocol (matching the per-group breakdown and the
    Z-score analysis): a sample is a false negative only if it is a TRUE
    OUTLIER (raw Class 1 AND Oordeel 1) that was NOT flagged, and a false
    positive only if it is a TRUE INLIER (raw Class 0 AND Oordeel 0) that WAS
    flagged. Samples with NaN Oordeel and every other (Class, Oordeel)
    combination are gray and excluded.

    When `group_map` is not available (e.g. non-confident-normals schemes
    with no Oordeel/Classification metadata), falls back to the binary label
    (`true_label == 1` for false negatives) so the file is still produced.

    Each row records the sample_id, role, raw Classification, Oordeel,
    flagged, score, and anomaly_threshold so they can be inspected manually.
    The files are written next to the other fold outputs; pooled across
    outer-CV folds by the caller.
    """
    if realistic_results is None:
        return
    per_sample = realistic_results.get('per_iteration_results', [])
    if not per_sample:
        return

    threshold = realistic_results.get('anomaly_threshold')

    # Determine the true_outlier / true_inlier id sets from group_map. Ids are
    # normalised to strings so a dtype mismatch between the per-sample rows
    # (which round-trip through dict keys) and the group_map index does not
    # silently drop every match.
    if group_map is not None and len(group_map) > 0:
        gm = group_map.copy()
        gm['raw_classification'] = pd.to_numeric(gm.get('raw_classification'), errors='coerce')
        gm['oordeel'] = pd.to_numeric(gm.get('oordeel'), errors='coerce')
        gm = gm.dropna(subset=['oordeel'])
        true_outlier_ids = {str(i) for i in gm[((gm['raw_classification'] == 1) & (gm['oordeel'] == 1))].index.tolist()}
        true_inlier_ids = {str(i) for i in gm[((gm['raw_classification'] == 0) & (gm['oordeel'] == 0))].index.tolist()}
        rc_by_id = {str(i): v for i, v in gm['raw_classification'].to_dict().items()}
        oo_by_id = {str(i): v for i, v in gm['oordeel'].to_dict().items()}
        use_roles = True
    else:
        true_outlier_ids = set()
        true_inlier_ids = set()
        rc_by_id = {}
        oo_by_id = {}
        use_roles = False

    fn_rows = []
    fp_rows = []
    gray_rows = []
    for r in per_sample:
        sid = r.get('sample_id')
        sid_s = str(sid)
        flagged = int(r.get('flagged', 0))
        row = {
            'sample_id': sid,
            'flagged': flagged,
            'score': r.get('score'),
            'anomaly_threshold': threshold,
        }
        if use_roles:
            row['raw_classification'] = rc_by_id.get(sid_s)
            row['oordeel'] = oo_by_id.get(sid_s)
            if sid_s in true_outlier_ids and flagged == 0:
                row['role'] = 'false_negative'
                fn_rows.append(row)
            elif sid_s in true_inlier_ids and flagged == 1:
                row['role'] = 'false_positive'
                fp_rows.append(row)
            elif flagged == 1:
                # Neither a true outlier (1,1) nor a true inlier (0,0) but
                # flagged by the model: a gray_investigation sample the model
                # scored as an outlier. (True inliers that were flagged are
                # already handled above, so this is genuinely the gray pool.)
                row['role'] = 'gray_flagged'
                gray_rows.append(row)
        else:
            # Binary-label fallback (no group metadata): true_label == 1 is
            # the abnormal pool; an unflagged one is a false negative.
            row['true_label'] = r.get('true_label')
            if r.get('true_label') == 1 and flagged == 0:
                row['role'] = 'false_negative'
                fn_rows.append(row)

    suffix = "" if fold is None else f"_fold{fold}"
    if fn_rows:
        fn_df = pd.DataFrame(fn_rows)
        if 'sample_id' in fn_df.columns:
            fn_df = fn_df.sort_values('sample_id')
        fn_df.to_csv(output_dir / f"false_negative_imds{suffix}.csv", index=False)
        logger.info(f"Saved {len(fn_df)} false-negative IMD sample(s) to "
                    f"{output_dir / f'false_negative_imds{suffix}.csv'}")
    else:
        logger.info("No false-negative IMD samples (all true outliers flagged).")

    if fp_rows:
        fp_df = pd.DataFrame(fp_rows)
        if 'sample_id' in fp_df.columns:
            fp_df = fp_df.sort_values('sample_id')
        fp_df.to_csv(output_dir / f"false_positive_imds{suffix}.csv", index=False)
        logger.info(f"Saved {len(fp_df)} false-positive sample(s) to "
                    f"{output_dir / f'false_positive_imds{suffix}.csv'}")

    if gray_rows:
        gray_df = pd.DataFrame(gray_rows)
        if 'sample_id' in gray_df.columns:
            gray_df = gray_df.sort_values('sample_id')
        gray_df.to_csv(output_dir / f"gray_flagged_imds{suffix}.csv", index=False)
        logger.info(f"Saved {len(gray_df)} gray-investigation sample(s) flagged as "
                    f"outliers to {output_dir / f'gray_flagged_imds{suffix}.csv'}")
    elif use_roles:
        logger.info("No gray-investigation samples flagged as outliers.")


def _apply_univariate_guardrail(
    original_features: Optional[pd.DataFrame],
    reference_normal_ids: Optional[pd.Index],
    test_sample_ids: Optional[pd.Index],
    output_dir: Path,
    config: Config,
) -> Dict[Any, List[str]]:
    """Deterministic univariate guardrail for clinically important biomarkers.

    A deployment-time safety net (NOT a training filter): for each biomarker
    listed in the ``univariate_guardrail.biomarkers`` config (each entry a dict
    with ``name`` and ``z_threshold``), compute the biomarker's mean/std over
    the TRAINING confident normals (``reference_normal_ids``), then flag every
    TEST sample (``test_sample_ids``) whose signed Z-score on that biomarker
    exceeds the threshold in magnitude. A sample is flagged by the guardrail if
    ANY listed biomarker exceeds its threshold.

    This is intentionally OR-combined with the model flag by the caller: a
    sample is reviewed if the guardrail fires OR the model flags it. The
    guardrail guarantees that clinically must-not-miss metabolites (e.g.
    glutarylcarnitine) are caught even when the multivariate model scores them
    as normal, without biasing the training set.

    Writes ``guardrail_flags.csv`` (one row per fired biomarker-sample pair:
    sample_id, biomarker, zscore, z_threshold, direction) and returns a dict
    mapping each flagged sample_id to its list of human-readable reasons.

    When ``univariate_guardrail.biomarkers`` is absent or empty (the default),
    this is a no-op and returns ``{}``.
    """
    if original_features is None or original_features.empty:
        return {}
    if reference_normal_ids is None or len(reference_normal_ids) == 0:
        logger.info("Univariate guardrail skipped: no reference normal ids.")
        return {}
    if test_sample_ids is None or len(test_sample_ids) == 0:
        return {}

    biomarkers = config.get('univariate_guardrail.biomarkers', None)
    if not biomarkers:
        return {}
    if not isinstance(biomarkers, list):
        biomarkers = [biomarkers]

    # Resolve each configured biomarker to an actual feature column (exact,
    # case-insensitive match after Unicode normalization), skipping any that
    # are not present so a typo never aborts the run.
    available = {str(c).strip().casefold(): c for c in original_features.columns}
    resolved = []
    for bm in biomarkers:
        if not isinstance(bm, dict):
            continue
        name = bm.get('name')
        thr = bm.get('z_threshold')
        if not name or thr is None:
            logger.warning(f"Univariate guardrail: skipping malformed entry {bm}; "
                           "expected {{name, z_threshold}}.")
            continue
        col = available.get(str(name).strip().casefold())
        if col is None:
            logger.warning(f"Univariate guardrail: biomarker '{name}' not found "
                           f"in original_features columns; skipped.")
            continue
        resolved.append((name, col, float(thr)))
    if not resolved:
        logger.info("Univariate guardrail: no resolvable biomarkers configured; "
                    "no-op.")
        return {}

    cols = [c for _, c, _ in resolved]
    # Select by membership (not reindex) so duplicate sample IDs in the index
    # do not raise 'cannot reindex on an axis with duplicate labels'. A sample
    # may appear more than once; both rows are scored.
    ref = original_features.loc[original_features.index.isin(reference_normal_ids)].dropna(how='all')
    if ref.empty:
        logger.warning("Univariate guardrail: reference normal features empty "
                       "after alignment; skipped.")
        return {}
    means = ref[cols].mean(axis=0)
    stds = ref[cols].std(axis=0)

    test_feats = original_features.loc[original_features.index.isin(test_sample_ids)].dropna(how='all')
    if test_feats.empty:
        return {}

    rows = []
    flags: Dict[Any, List[str]] = {}
    for sid in test_feats.index:
        sample = test_feats.loc[sid]
        if isinstance(sample, pd.DataFrame):
            sample = sample.iloc[0]
        for name, col, thr in resolved:
            m = means[col]
            s = stds[col]
            if pd.isna(s) or s < 1e-10 or pd.isna(sample[col]):
                continue
            z = float((sample[col] - m) / (s + 1e-10))
            if abs(z) > thr:
                direction = 'increased' if z >= 0 else 'decreased'
                reason = (f"{name} {z:+.2f}x std ({direction}, threshold {thr:g})")
                flags.setdefault(sid, []).append(reason)
                rows.append({
                    'sample_id': sid, 'biomarker': name, 'zscore': z,
                    'z_threshold': thr, 'direction': direction,
                })

    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "guardrail_flags.csv", index=False)
        logger.info(f"Univariate guardrail: flagged {len(flags)} sample(s) on "
                    f"{len(rows)} biomarker-sample pair(s); wrote "
                    f"{output_dir / 'guardrail_flags.csv'}")
    else:
        logger.info("Univariate guardrail: no samples exceeded any configured "
                    "biomarker threshold.")
    return flags


def _compute_combined_guardrail_metrics(
    per_sample_results: List[Dict[str, Any]],
    guardrail_flags: Dict[Any, List[str]],
    group_map: Optional[pd.DataFrame],
    target_contamination: float,
) -> Dict[str, Any]:
    """Recompute detection/FPR/precision/F1/accuracy/ROC-AUC for the COMBINED
    decision (model flag OR univariate guardrail), mirroring the prevalence-
    aware computation in realistic_evaluation.run_realistic_evaluation.

    The combined flag for a test sample is 1 if the model flagged it
    (``flagged`` in per_sample_results) OR the guardrail flagged it (its id is
    a key in ``guardrail_flags``). Headline totals are restricted to the clean
    lab-protocol roles (true outlier = Class 1 & Oordeel 1; true inlier =
    Class 0 & Oordeel 0) when a group_map is supplied, exactly as the model-
    only metrics are. Returns a dict with the same keys the sweep consumes.
    """
    if not per_sample_results:
        return {}
    gm_norm_ids = {str(i) for i in guardrail_flags.keys()} if guardrail_flags else set()

    rows = []
    for r in per_sample_results:
        sid = r.get('sample_id')
        model_flagged = int(r.get('flagged', 0))
        gr_flagged = 1 if str(sid) in gm_norm_ids else 0
        rows.append({
            'sample_id': sid,
            'true_label': int(r.get('true_label', 0)),
            'score': float(r.get('score', float('nan'))),
            'flagged': 1 if (model_flagged or gr_flagged) else 0,
            'model_flagged': model_flagged,
            'guardrail_flagged': gr_flagged,
        })
    df = pd.DataFrame(rows)

    # Clean roles: true outlier = (1,1), true inlier = (0,0); gray excluded.
    if group_map is not None and len(group_map) > 0:
        gm = group_map.copy()
        gm['raw_classification'] = pd.to_numeric(gm.get('raw_classification'), errors='coerce')
        gm['oordeel'] = pd.to_numeric(gm.get('oordeel'), errors='coerce')
        gm = gm.dropna(subset=['oordeel'])
        to_ids = {str(i) for i in gm[((gm['raw_classification'] == 1) & (gm['oordeel'] == 1))].index.tolist()}
        tio_ids = {str(i) for i in gm[((gm['raw_classification'] == 0) & (gm['oordeel'] == 0))].index.tolist()}
        df['sid_s'] = df['sample_id'].astype(str)
        is_true_outlier = df['sid_s'].isin(to_ids)
        is_true_inlier = df['sid_s'].isin(tio_ids)
    else:
        is_true_outlier = df['true_label'] == 1
        is_true_inlier = df['true_label'] == 0

    n_abheadline = int(is_true_outlier.sum())
    n_norheadline = int(is_true_inlier.sum())
    ab_flagged = df.loc[is_true_outlier, 'flagged'].to_numpy(dtype=int) if n_abheadline > 0 else np.array([], dtype=int)
    no_flagged = df.loc[is_true_inlier, 'flagged'].to_numpy(dtype=int)
    n_detected = int(ab_flagged.sum())
    detection_rate = (n_detected / n_abheadline) if n_abheadline > 0 else float('nan')
    n_fp = int(no_flagged.sum())
    false_positive_rate = (n_fp / n_norheadline) if n_norheadline > 0 else float('nan')

    # ROC-AUC is ranking quality of the MODEL score; the guardrail has no
    # ranking, so the combined ROC-AUC equals the model-only ROC-AUC. We keep
    # it for a complete combined row but note it is unchanged.
    try:
        ab = df.loc[df['true_label'] == 1, 'score'].to_numpy(dtype=float)
        no = df.loc[df['true_label'] == 0, 'score'].to_numpy(dtype=float)
        if len(ab) > 0 and len(no) > 0:
            from sklearn.metrics import roc_auc_score
            roc_auc = float(roc_auc_score(np.concatenate([np.zeros(len(no)), np.ones(len(ab))]),
                                         -np.concatenate([no, ab])))
        else:
            roc_auc = float('nan')
    except Exception:
        roc_auc = float('nan')

    p = target_contamination
    if n_abheadline > 0 and n_norheadline > 0 and not np.isnan(detection_rate) and not np.isnan(false_positive_rate):
        denom = (p * detection_rate) + ((1.0 - p) * false_positive_rate)
        precision_deploy = float((p * detection_rate) / denom) if denom > 0 else float('nan')
        recall_deploy = float(detection_rate)
        f1_deploy = float(2.0 * precision_deploy * recall_deploy / (precision_deploy + recall_deploy)) \
            if (precision_deploy + recall_deploy) > 0 else 0.0
        accuracy_deploy = float((1.0 - p) * (1.0 - false_positive_rate) + p * detection_rate)
    else:
        precision_deploy = float('nan')
        f1_deploy = float('nan')
        accuracy_deploy = float('nan')

    return {
        'detection_rate': float(detection_rate) if not np.isnan(detection_rate) else float('nan'),
        'false_positive_rate': float(false_positive_rate) if not np.isnan(false_positive_rate) else float('nan'),
        'roc_auc': roc_auc,
        'precision': precision_deploy,
        'f1': f1_deploy,
        'accuracy': accuracy_deploy,
        'n_detected': n_detected,
        'n_false_positives': n_fp,
        'n_true_outlier': n_abheadline,
        'n_true_inlier': n_norheadline,
        'per_sample': df.to_dict(orient='records'),
    }


def _plot_fn_fp_zscore_analysis(
    original_features: Optional[pd.DataFrame],
    output_dir: Path,
    config: Config,
    n_top: int = 20,
) -> None:
    """Plot a Z-score deviation bar chart for each false-negative,
    false-positive, and gray-investigation-flagged sample, reading the sample
    ids from the already-generated ``false_negative_imds.csv``,
    ``false_positive_imds.csv``, and ``gray_flagged_imds.csv`` files in
    ``output_dir``.

    The role of each sample (false_negative / false_positive / gray) is
    determined solely by which CSV it was listed in; the CSVs themselves are
    produced by ``_save_false_negatives_csv`` using the lab-protocol role
    definitions (true_outlier = Class 1 & Oordeel 1 unflagged; true_inlier =
    Class 0 & Oordeel 0 flagged; gray = every other combination flagged as an
    outlier). This avoids re-deriving the roles here and guarantees the plots
    match the CSVs exactly.

    Exactly one plot is produced per unique sample_id: if a sample appears in
    multiple CSVs (should not happen with the role logic, but handled
    defensively) it is plotted once and tagged with the first role
    encountered (precedence: false negatives, then false positives, then
    gray). Each plot is a horizontal bar chart of the top-`n_top` most-
    deviating features, saved as ``zscore_fn_<sample_id>.png`` /
    ``zscore_fp_<sample_id>.png`` / ``zscore_gray_<sample_id>.png`` inside a
    ``zscore_fn_fp`` subfolder, and a combined ranked-feature CSV is written.
    """
    if original_features is None or original_features.empty:
        logger.info("FN/FP Z-score analysis skipped: original_features is empty.")
        return

    use_zscore = config.get('use_zscore_analysis', True)
    if not use_zscore:
        logger.info("FN/FP Z-score analysis skipped: use_zscore_analysis is false.")
        return

    # Read the generated FN/FP CSVs. The sample_id column is normalised to a
    # string so a dtype mismatch with the original_features index (which may
    # be int) does not silently drop every match.
    fn_csv = output_dir / "false_negative_imds.csv"
    fp_csv = output_dir / "false_positive_imds.csv"
    gray_csv = output_dir / "gray_flagged_imds.csv"
    fn_ids: List[Any] = []
    fp_ids: List[Any] = []
    gray_ids: List[Any] = []
    if fn_csv.exists():
        try:
            fn_df = pd.read_csv(fn_csv)
            if 'sample_id' in fn_df.columns:
                fn_ids = fn_df['sample_id'].astype(str).tolist()
        except Exception as e:
            logger.warning(f"Could not read {fn_csv}: {e}")
    if fp_csv.exists():
        try:
            fp_df = pd.read_csv(fp_csv)
            if 'sample_id' in fp_df.columns:
                fp_ids = fp_df['sample_id'].astype(str).tolist()
        except Exception as e:
            logger.warning(f"Could not read {fp_csv}: {e}")
    if gray_csv.exists():
        try:
            gray_df = pd.read_csv(gray_csv)
            if 'sample_id' in gray_df.columns:
                gray_ids = gray_df['sample_id'].astype(str).tolist()
        except Exception as e:
            logger.warning(f"Could not read {gray_csv}: {e}")

    logger.info(f"FN/FP Z-score analysis: read {len(fn_ids)} false-negative, "
                f"{len(fp_ids)} false-positive, and {len(gray_ids)} gray-flagged "
                f"sample id(s) from CSVs in {output_dir}.")

    # Deduplicate while preserving role precedence (fn, then fp, then gray).
    # Each sample_id is plotted exactly once.
    role_by_id: Dict[str, str] = {}
    ordered_ids: List[str] = []
    for sid in fn_ids:
        if sid not in role_by_id:
            role_by_id[sid] = 'fn'
            ordered_ids.append(sid)
    for sid in fp_ids:
        if sid not in role_by_id:
            role_by_id[sid] = 'fp'
            ordered_ids.append(sid)
    for sid in gray_ids:
        if sid not in role_by_id:
            role_by_id[sid] = 'gray'
            ordered_ids.append(sid)

    if not ordered_ids:
        logger.info("No false-negative, false-positive, or gray-flagged "
                    "samples in the CSVs to plot Z-scores for.")
        return

    # Map the (string-normalised) CSV sample ids back to the actual
    # original_features index values so analyze_outliers' `idx in index`
    # lookup succeeds regardless of index dtype.
    of_index_by_str = {str(i): i for i in original_features.index}
    resolved_ids = [of_index_by_str[sid] for sid in ordered_ids if sid in of_index_by_str]
    missing = [sid for sid in ordered_ids if sid not in of_index_by_str]
    if missing:
        logger.warning(f"{len(missing)} sample id(s) from the FN/FP CSVs not "
                       f"found in original_features index (e.g. {missing[:3]}); "
                       f"they will be skipped in the Z-score analysis.")
    if not resolved_ids:
        logger.warning("No FN/FP sample ids matched the original_features "
                       "index; skipping Z-score plots.")
        return

    zscore_dir = output_dir / "zscore_fn_fp"
    zscore_dir.mkdir(parents=True, exist_ok=True)

    try:
        from outlier_detection_pipeline.pipeline.outlier_analysis import HAS_MATPLOTLIB
    except Exception:
        HAS_MATPLOTLIB = False
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available; FN/FP Z-score PNG plots will "
                       "be skipped, but the ranked-feature CSV is still written.")

    analysis = analyze_outliers(original_features, resolved_ids, n_top=n_top, use_zscore=True)
    if analysis:
        plot_outlier_analysis(analysis, original_features, zscore_dir, n_top=n_top, use_zscore=True)
        # plot_outlier_analysis writes 'zscore_outlier_<id>.png'; rename to the
        # fn/fp role so the two kinds are distinguishable.
        for sid in analysis.keys():
            role = role_by_id.get(str(sid))
            if role is None:
                continue
            src = zscore_dir / f"zscore_outlier_{sid}.png"
            if src.exists():
                dst = zscore_dir / f"zscore_{role}_{sid}.png"
                src.replace(dst)

    all_rows = []
    for sid, a in analysis.items():
        role = role_by_id.get(str(sid))
        if role is None:
            continue
        for rank, feat_tuple in enumerate(a.get('top_features', [])[:n_top], 1):
            feat = feat_tuple[0]
            wdev = feat_tuple[1]
            adev = feat_tuple[2]
            sdev = feat_tuple[3] if len(feat_tuple) > 3 else wdev
            all_rows.append({
                'sample_id': sid, 'role': role, 'rank': rank,
                'feature': feat, 'weighted_zscore': wdev, 'abs_zscore': adev,
                'signed_zscore': sdev,
            })

    if all_rows:
        pd.DataFrame(all_rows).to_csv(zscore_dir / "zscore_fn_fp_analysis.csv", index=False)
    logger.info(f"FN/FP Z-score analysis: plotted {len(analysis)} unique sample(s) "
                f"({len(fn_ids)} false-negative, {len(fp_ids)} false-positive, "
                f"{len(gray_ids)} gray-flagged in CSVs); wrote {len(all_rows)} "
                f"ranked-feature rows and any available plots to {zscore_dir}")


def _per_group_breakdown(
    per_sample_results: List[Dict[str, Any]],
    anomaly_threshold: float,
    group_map: pd.DataFrame,
    target_contamination: float,
    output_dir: Path,
    label: str = "",
) -> Optional[pd.DataFrame]:
    """Score the model per (raw Classification, Oordeel targeted) group.

    For each (Class, Oordeel) combination present in the test set, reports the
    flag rate and a per-group confusion count at the deployment prevalence
    (target_contamination), using the absolute anomaly threshold calibrated
    on confident normals.

    Ground-truth definition (lab protocol):
      - True outlier  = raw Classification 1 (IMD) AND Oordeel targeted 1.
        A flagged sample in this group is a TP.
      - True inlier   = raw Class 0 AND Oordeel 0. A flagged sample here is
        a FP.
      - Every other (Class, Oordeel) combination is a gray area: reported
        as its own group but NOT counted in the TP/FN totals. Samples with
        a NaN Oordeel targeted are dropped entirely before the breakdown.

    Args:
        per_sample_results: rows with sample_id, true_label (binary), score, flagged
        anomaly_threshold: the calibrated absolute score cutoff used
        group_map: DataFrame indexed by sample_id with columns
            'raw_classification' and 'oordeel'
        target_contamination: assumed deployment prevalence (e.g. 0.02)
        output_dir: where to save the per-group CSV/log
        label: optional prefix for the saved filename

    Returns:
        DataFrame with one row per (Class, Oordeel) group, or None.
    """
    if not per_sample_results or group_map is None or len(group_map) == 0:
        return None

    rows = pd.DataFrame(per_sample_results)
    if 'sample_id' not in rows.columns:
        return None
    rows = rows.set_index('sample_id')
    merged = rows.join(group_map[['raw_classification', 'oordeel']], how='left')
    merged['raw_classification'] = pd.to_numeric(merged['raw_classification'], errors='coerce')
    merged['oordeel'] = pd.to_numeric(merged['oordeel'], errors='coerce')

    # Drop samples with a NaN Oordeel targeted: their role is undefined, so
    # they cannot be scored against ground truth.
    nan_oordeel_mask = merged['oordeel'].isna()
    n_nan_oordeel = int(nan_oordeel_mask.sum())
    if n_nan_oordeel > 0:
        logger.info(f"Per-group breakdown: dropping {n_nan_oordeel} samples "
                    f"with NaN in Oordeel targeted.")
        merged = merged[~nan_oordeel_mask]

    # Ground-truth roles (lab protocol):
    #   true_outlier  = raw Class 1 AND Oordeel 1
    #   true_inlier   = raw Class 0 AND Oordeel 0
    #   gray_investigation = every other combination
    rc = merged['raw_classification']
    oo = merged['oordeel']
    is_true_outlier = ((rc == 1) & (oo == 1)).astype(int)
    is_true_inlier = ((rc == 0) & (oo == 0)).astype(int)
    merged['is_true_outlier'] = is_true_outlier
    merged['is_true_inlier'] = is_true_inlier
    merged['role'] = np.select(
        [is_true_outlier.astype(bool), is_true_inlier.astype(bool)],
        ['true_outlier', 'true_inlier'],
        default='gray_investigation',
    )

    p = float(target_contamination)
    records = []
    for (rcv, oov), grp in merged.groupby(['raw_classification', 'oordeel'], dropna=False):
        n = len(grp)
        n_flagged = int(grp['flagged'].sum())
        flag_rate = (n_flagged / n) if n else float('nan')
        role = ('true_outlier' if (rcv == 1 and oov == 1)
                else 'true_inlier' if (rcv == 0 and oov == 0)
                else 'gray_investigation')
        records.append({
            'raw_classification': rcv,
            'oordeel': oov,
            'role': role,
            'n_samples': n,
            'n_flagged': n_flagged,
            'flag_rate': flag_rate,
        })

    breakdown = pd.DataFrame(records).sort_values(['raw_classification', 'oordeel']).reset_index(drop=True)

    # Headline metrics at deployment prevalence, excluding the gray groups
    # (everything that is not a true outlier or a true inlier) per the protocol.
    non_gray = merged[merged['role'] != 'gray_investigation']
    tp = int(((non_gray['flagged'] == 1) & (non_gray['is_true_outlier'] == 1)).sum())
    fn = int(((non_gray['flagged'] == 0) & (non_gray['is_true_outlier'] == 1)).sum())
    fp = int(((non_gray['flagged'] == 1) & (non_gray['is_true_inlier'] == 1)).sum())
    tn = int(((non_gray['flagged'] == 0) & (non_gray['is_true_inlier'] == 1)).sum())
    n_true_outlier = int(non_gray['is_true_outlier'].sum())
    n_true_inlier = int(non_gray['is_true_inlier'].sum())
    detection = (tp / n_true_outlier) if n_true_outlier else float('nan')
    fpr = (fp / n_true_inlier) if n_true_inlier else float('nan')
    valid = not (np.isnan(detection) or np.isnan(fpr))
    denom = (p * detection) + ((1.0 - p) * fpr) if valid else 0.0
    precision_deploy = float((p * detection) / denom) if denom else float('nan')
    f1_deploy = (2 * precision_deploy * detection / (precision_deploy + detection)) if (precision_deploy + detection) else 0.0
    accuracy_deploy = float((1 - p) * (1 - fpr) + p * detection) if valid else float('nan')

    n_total = int(len(non_gray))
    n_out_batch = max(1, int(round(p * n_total)))
    n_in_batch = n_total - n_out_batch
    cm = (np.array([
        [int(round(n_in_batch * (1 - fpr))), int(round(n_in_batch * fpr))],
        [int(round(n_out_batch * (1 - detection))), int(round(n_out_batch * detection))],
    ]) if valid else np.array([[tn, fp], [fn, tp]]))

    hdr = f"PER-GROUP BREAKDOWN ({label})" if label else "PER-GROUP BREAKDOWN"
    _log_section_header(hdr)
    logger.info(f"Anomaly threshold: {anomaly_threshold:.6f}  | deployment prevalence: {p:.2%}")
    logger.info(f"{'Class':>6} {'Oordeel':>8} {'role':>20} {'n':>5} {'flagged':>8} {'flag_rate':>10}")
    for _, r in breakdown.iterrows():
        logger.info(f"{r['raw_classification']:>6} {r['oordeel']:>8} {r['role']:>20} "
                    f"{r['n_samples']:>5} {r['n_flagged']:>8} {r['flag_rate']:>10.2%}")
    logger.info(f"Headline (excl. gray groups): TP={tp} FN={fn} FP={fp} TN={tn} | "
                f"detection={detection:.2%} FPR={fpr:.2%} "
                f"precision@{p:.0%}={precision_deploy:.4f} f1={f1_deploy:.4f} acc={accuracy_deploy:.4f}")
    logger.info(f"Deployment-batch confusion (n={n_total} @ {p:.2%}):\n{cm}")

    fname = (f"per_group_breakdown_{label}.csv" if label else "per_group_breakdown.csv")
    breakdown.to_csv(output_dir / fname, index=False)
    logger.info(f"Per-group breakdown saved to {output_dir / fname}")
    return breakdown


def _save_outputs(
    model: ExtendedIsolationForestModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_preds: np.ndarray,
    test_scores: np.ndarray,
    config: Config,
    output_dir: Path,
    normal_class: int,
    outlier_classes: List[int],
    test_metrics: Optional[Dict[str, Any]] = None,
    realistic_results: Optional[Dict[str, Any]] = None,
    pca: Optional[SparsePCAWrapper] = None,
    original_features: Optional[pd.DataFrame] = None,
    group_map: Optional[pd.DataFrame] = None,
    reference_normal_ids: Optional[pd.Index] = None,
) -> Dict[str, Any]:
    """Save all pipeline outputs."""
    _log_section_header("Saving outputs")

    save_plots = config.get('save_plots', True)
    save_model = config.get('save_model', True)
    save_preds = config.get('save_predictions', True)

    if save_model:
        model.save(output_dir / "model.joblib")

    if save_preds:
        save_predictions(
            predictions=test_preds,
            scores=test_scores,
            patient_ids=X_test.index,
            true_labels=y_test,
            output_dir=output_dir,
            split_name="test",
        )

    # Save metrics
    if test_metrics is None:
        metrics_list = config.get_list('metrics', ['accuracy', 'f1', 'f1_weighted', 'precision', 'recall', 'roc_auc', 'confusion_matrix'])
        test_metrics = evaluate_model(
            y_true=y_test,
            y_pred=test_preds,
            y_scores=test_scores,
            metrics=metrics_list,
            pos_label=-1,
            outlier_classes=outlier_classes,
        )
        print_metrics(test_metrics)

    save_metrics(test_metrics, output_dir / "test")

    # Generate and save plots
    if save_plots:
        plot_confusion_matrix(
            y_true=y_test,
            y_pred=test_preds,
            output_dir=output_dir,
            outlier_classes=outlier_classes,
            pos_label=-1,
        )
        plot_precision_recall_curve(
            y_true=y_test,
            y_scores=test_scores,
            output_dir=output_dir,
            outlier_classes=outlier_classes,
            pos_label=-1,
        )
        
        # For realistic evaluation, also save the realistic confusion matrix
        if realistic_results is not None:
            plot_realistic_results(realistic_results, output_dir)

    # Save the false-negative and false-positive IMD CSVs first (these use the
    # lab-protocol role definitions), then generate the per-sample Z-score
    # plots by reading those CSVs so the plots always match the saved files.
    _save_false_negatives_csv(realistic_results, output_dir, fold=None, group_map=group_map)

    if original_features is not None:
        try:
            _plot_fn_fp_zscore_analysis(
                original_features=original_features,
                output_dir=output_dir,
                config=config,
            )
        except Exception as e:
            logger.exception(f"FN/FP Z-score analysis failed: {e}")

    # Deterministic univariate guardrail for clinically important biomarkers
    # (deployment-time safety net, OR-combined with the model flag). Reads the
    # biomarker list from univariate_guardrail.biomarkers; no-op when unset.
    try:
        guardrail_flags = _apply_univariate_guardrail(
            original_features=original_features,
            reference_normal_ids=reference_normal_ids,
            test_sample_ids=X_test.index,
            output_dir=output_dir,
            config=config,
        )
    except Exception as e:
        logger.exception(f"Univariate guardrail failed: {e}")
        guardrail_flags = {}
    # Log which test samples the guardrail catches that the model missed
    # (model flagged = test_preds == -1): these are the must-not-miss rescues.
    if guardrail_flags:
        model_flagged = set(X_test.index[test_preds == -1])
        rescued = [str(s) for s in guardrail_flags if s not in model_flagged]
        if rescued:
            logger.info(f"Univariate guardrail rescued {len(rescued)} sample(s) "
                        f"the model did NOT flag: {rescued}")

    return {
        'test_metrics': test_metrics,
        'model': model,
        'splits': {
            'train': len(X_test),
            'test': len(X_test),
        },
        'realistic_results': realistic_results,
    }


def _run_fold_drift_diagnostic(
    model: ExtendedIsolationForestModel,
    X_train_p: pd.DataFrame,
    y_train_p: pd.Series,
    X_test_p: pd.DataFrame,
    y_test_p: pd.Series,
    normal_class: int,
    realistic_results: Optional[Dict[str, Any]],
    config: Config,
    output_dir: Path,
    fold_num: int,
) -> None:
    """Run the drift diagnostic for one outer-CV fold.

    Splits the fold's held-out test set into normal/abnormal and compares the
    score distributions against the fold's training normals. The threshold is
    taken from the realistic evaluation when available (so the reported FPR is
    exactly the one the realistic eval used); otherwise it falls back to the
    configured deployment contamination percentile of the OOF normal scores.
    """
    outlier_classes = config.get_list('outlier_classifications', [1, 2, 3])
    X_test_normal = X_test_p[y_test_p == normal_class]
    X_test_abnormal = X_test_p[y_test_p.isin(outlier_classes)]
    X_train_normal = X_train_p[y_train_p == normal_class]

    target_contamination = config.get('realistic_test_contamination', 0.02)
    if realistic_results is not None and realistic_results.get('anomaly_threshold') is not None:
        threshold = float(realistic_results['anomaly_threshold'])
    else:
        oof = getattr(model, 'oof_normal_scores_', None)
        if oof is not None and len(oof) > 0:
            threshold = float(np.percentile(oof, 100.0 * target_contamination))
        else:
            threshold = float('nan')

    run_drift_diagnostic(
        model=model,
        X_train_normals=X_train_normal,
        X_test_normals=X_test_normal,
        X_test_abnormals=X_test_abnormal if len(X_test_abnormal) > 0 else None,
        target_contamination=target_contamination,
        anomaly_threshold=threshold,
        output_dir=output_dir,
        fold_num=fold_num,
    )


def _run_outer_cv(
    features: pd.DataFrame,
    classification: pd.Series,
    normal_class: int,
    outlier_classes: List[int],
    config: Config,
    output_dir: Path,
    n_folds: int,
    random_seed: int,
    original_features: Optional[pd.DataFrame] = None,
    group_map: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Run k-fold outer cross-validation of the full pipeline.

    Each fold is used once as the held-out test set; the remaining folds form
    the train set. The full train -> (optional PCA) -> train model ->
    realistic-evaluate process runs per fold. Per-fold results are aggregated
    (mean/std) to average out an unlucky single split on small datasets and
    give honest, out-of-sample estimates of detection rate / FPR / ROC AUC.

    Splits are stratified by the binary normal/abnormal label so each test
    fold has a representative class ratio. Abnormal samples are only ever
    scored in the test fold of their own fold (never trained on).
    """
    _log_section_header(f"OUTER CROSS-VALIDATION ({n_folds} folds)")

    # Confident-normals split: the CV folds are built over the CONFIDENT
    # NORMALS only (binary label 0). Each fold holds out 1/k of normals as the
    # FPR-measuring test normals, while ALL abnormals (binary label 1) go into
    # the test set of EVERY fold. The model only ever fits on normals (y==0),
    # so abnormals are never trained on, and no abnormal sample is wasted in a
    # train split. This matches the lab protocol: train on confident normals,
    # test on all abnormals + a held-out normal slice. It also corrects for the
    # artificially enriched test set by scoring every abnormal in every fold
    # while the realistic eval reports mixed metrics analytically at the
    # (low) deployment prevalence, decoupled from the enriched test-set ratio.
    scheme = config.get('classification_scheme', 'default')
    confident_normals_mode = (scheme == 'confident_normals')
    y_binary = (classification != normal_class).astype(int).values

    if confident_normals_mode:
        normal_idx = np.where(y_binary == 0)[0]
        abnormal_idx = np.where(y_binary == 1)[0]
        if n_folds <= 1:
            train_ratio = float(config.get('train_ratio', 0.8))
            n_normal = len(normal_idx)
            n_train = int(round(train_ratio * n_normal))
            rng = np.random.RandomState(random_seed)
            perm = rng.permutation(n_normal)
            normal_train_pos = perm[:n_train]
            normal_test_pos = perm[n_train:]
            folds = [(normal_train_pos, normal_test_pos)]
            logger.info(f"confident_normals single split: {n_train}/{n_normal} "
                        f"normals train, {n_normal - n_train} normals held out for FPR; "
                        f"{len(abnormal_idx)} abnormals scored once (never trained).")
        else:
            logger.info(f"confident_normals split: {len(normal_idx)} confident normals "
                        f"split into {n_folds} folds (held-out per fold for FPR); "
                        f"{len(abnormal_idx)} abnormals scored in EVERY fold (never trained).")
            folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                        random_state=random_seed).split(normal_idx,
                                                                        y_binary[normal_idx]))
    else:
        folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                     random_state=random_seed).split(features, y_binary))

    fold_results = []
    all_per_sample = []

    metrics_keys = ['detection_rate', 'false_positive_rate', 'roc_auc',
                   'precision', 'f1', 'accuracy', 'anomaly_threshold']

    for fold_num, fold_split in enumerate(folds):
        if confident_normals_mode:
            normal_train_pos, normal_test_pos = fold_split
            train_idx = normal_idx[normal_train_pos]
            # Test = held-out normals + ALL abnormals (every fold).
            test_idx = np.concatenate([normal_idx[normal_test_pos], abnormal_idx])
        else:
            train_idx, test_idx = fold_split
        _log_section_header(f"OUTER FOLD {fold_num + 1}/{n_folds}")

        X_train = features.iloc[train_idx].copy()
        y_train = classification.iloc[train_idx].copy()
        X_test = features.iloc[test_idx].copy()
        y_test = classification.iloc[test_idx].copy()

        logger.info(f"Fold {fold_num + 1}: train={len(X_train)} (normal={int((y_train==normal_class).sum())}, "
                    f"abnormal={int((y_train!=normal_class).sum())}), test={len(X_test)} "
                    f"(normal={int((y_test==normal_class).sum())}, abnormal={int((y_test!=normal_class).sum())})")

        fold_out_dir = output_dir / f"fold_{fold_num + 1}"
        fold_out_dir.mkdir(parents=True, exist_ok=True)

        # PCA on this fold (fit on this fold's train normals only)
        X_train_p, X_test_p, y_train_p, y_test_p, pca = _apply_pca(
            X_train, X_test, y_train, y_test, config, fold_out_dir, normal_class
        )

        # Train model on this fold. Hyperparameter tuning is IF-specific;
        # non-IF scorers ignore it and use the no-tuning path.
        use_hp_tuning = config.get('use_hyperparameter_tuning', False)
        fold_scorer = config.get('scorer', 'iforest')
        if use_hp_tuning and fold_scorer == 'iforest':
            model, _ = _train_with_hyperparameter_tuning(
                X_train_p, y_train_p, config, fold_out_dir, normal_class
            )
            model.cross_val_predict(
                X=X_train_p, y=y_train_p, normal_classification=normal_class,
                n_splits=config.get('n_splits_tuning', 5),
            )
        elif use_hp_tuning and fold_scorer == 'ae':
            model = _train_ae_with_tuning(
                X_train_p, y_train_p, config, fold_out_dir, normal_class
            )
            model.cross_val_predict(
                X=X_train_p, y=y_train_p, normal_classification=normal_class,
                n_splits=config.get('n_splits_tuning', 5),
            )
        else:
            model = _train_without_tuning(X_train_p, y_train_p, config, normal_class)

        # Evaluate on this fold's held-out test set
        if config.get('evaluation_strategy', 'standard') == 'realistic':
            test_preds, test_scores, realistic_results = _evaluate_realistic(
                model, X_train_p, y_train_p, X_test_p, y_test_p, config,
                fold_out_dir, normal_class, outlier_classes,
                group_map=group_map,
            )
            fold_metrics = {k: realistic_results.get(k) for k in metrics_keys}
            fold_metrics['n_normal_test'] = realistic_results.get('n_normal_test')
            fold_metrics['n_abnormal_test'] = realistic_results.get('n_abnormal_test')
            # Per-fold list of IMD samples flagged false negative
            _save_false_negatives_csv(realistic_results, fold_out_dir, fold=fold_num + 1, group_map=group_map)
            # Pool per-sample scores for an aggregated score-distribution view
            for r in realistic_results.get('per_iteration_results', []):
                r = dict(r)
                r['fold'] = fold_num + 1
                all_per_sample.append(r)
        else:
            test_preds, test_scores, test_metrics = _evaluate_standard(
                model, X_test_p, y_test_p, config, normal_class, outlier_classes
            )
            fold_metrics = {k: test_metrics.get(k) for k in metrics_keys if k in test_metrics}
            fold_metrics['anomaly_threshold'] = float('nan')
            realistic_results = None

        fold_metrics['fold'] = fold_num + 1

        # Drift diagnostic for this fold. Compares this fold's train-normal
        # vs test-normal vs test-abnormal score distributions and flags FPR
        # divergence from the nominal deployment contamination, plus a 2-PC
        # scatter coloured by train/test split. Uses the same threshold the
        # realistic eval flagged with when available.
        if config.get('run_drift_diagnostic', True):
            _run_fold_drift_diagnostic(
                model=model,
                X_train_p=X_train_p,
                y_train_p=y_train_p,
                X_test_p=X_test_p,
                y_test_p=y_test_p,
                normal_class=normal_class,
                realistic_results=realistic_results,
                config=config,
                output_dir=fold_out_dir,
                fold_num=fold_num + 1,
            )

        fold_results.append(fold_metrics)

        logger.info(f"Fold {fold_num + 1} results: "
                    f"detection_rate={fold_metrics.get('detection_rate')}, "
                    f"FPR={fold_metrics.get('false_positive_rate')}, "
                    f"roc_auc={fold_metrics.get('roc_auc')}")

    # Aggregate across folds (mean +/- std), skipping NaNs.
    agg = {}
    for k in metrics_keys:
        vals = [fr.get(k) for fr in fold_results if fr.get(k) is not None and not (isinstance(fr.get(k), float) and np.isnan(fr.get(k)))]
        if vals:
            agg[f'{k}_mean'] = float(np.mean(vals))
            agg[f'{k}_std'] = float(np.std(vals))
            agg[f'{k}_folds'] = [float(v) for v in vals]

    _log_section_header("OUTER CV AGGREGATED RESULTS")
    logger.info(f"Folds: {n_folds}")
    for k in ['detection_rate', 'false_positive_rate', 'roc_auc', 'precision', 'f1', 'accuracy']:
        if f'{k}_mean' in agg:
            logger.info(f"  {k}: {agg[f'{k}_mean']:.4f} +/- {agg[f'{k}_std']:.4f}  "
                        f"per-fold: {[round(v,4) for v in agg[f'{k}_folds']]}")
    if 'anomaly_threshold_mean' in agg:
        logger.info(f"  anomaly_threshold: {agg['anomaly_threshold_mean']:.6f} +/- {agg['anomaly_threshold_std']:.6f}")

    # Save per-fold and aggregated results
    import json
    pd.DataFrame(fold_results).to_csv(output_dir / "outer_cv_per_fold.csv", index=False)
    with open(output_dir / "outer_cv_aggregated.json", 'w') as f:
        json.dump({k: v for k, v in agg.items() if not k.endswith('_folds')} | {'per_fold': fold_results}, f, indent=2, default=str)
    if all_per_sample:
        pd.DataFrame(all_per_sample).to_csv(output_dir / "outer_cv_per_sample.csv", index=False)

    # Pooled false-negative / false-positive IMD sample ids across all
    # outer-CV folds, using the lab-protocol roles (true outlier = Class 1 &
    # Oordeel 1; true inlier = Class 0 & Oordeel 0). Gray groups and NaN
    # Oordeel are excluded. Ids normalised to strings for dtype-safe matching.
    if group_map is not None and len(group_map) > 0:
        gm = group_map.copy()
        gm['raw_classification'] = pd.to_numeric(gm.get('raw_classification'), errors='coerce')
        gm['oordeel'] = pd.to_numeric(gm.get('oordeel'), errors='coerce')
        gm = gm.dropna(subset=['oordeel'])
        pooled_true_outlier_ids = {str(i) for i in gm[((gm['raw_classification'] == 1) & (gm['oordeel'] == 1))].index.tolist()}
        pooled_true_inlier_ids = {str(i) for i in gm[((gm['raw_classification'] == 0) & (gm['oordeel'] == 0))].index.tolist()}
    else:
        pooled_true_outlier_ids = set()
        pooled_true_inlier_ids = set()
    fn_rows = []
    fp_rows = []
    gray_rows = []
    for r in all_per_sample:
        sid_s = str(r.get('sample_id'))
        flagged = int(r.get('flagged', 0))
        if group_map is not None and len(group_map) > 0:
            if sid_s in pooled_true_outlier_ids and flagged == 0:
                fn_rows.append({'sample_id': r.get('sample_id'), 'fold': r.get('fold'),
                                'score': r.get('score'), 'flagged': flagged, 'role': 'false_negative'})
            elif sid_s in pooled_true_inlier_ids and flagged == 1:
                fp_rows.append({'sample_id': r.get('sample_id'), 'fold': r.get('fold'),
                                'score': r.get('score'), 'flagged': flagged, 'role': 'false_positive'})
            elif flagged == 1:
                gray_rows.append({'sample_id': r.get('sample_id'), 'fold': r.get('fold'),
                                  'score': r.get('score'), 'flagged': flagged, 'role': 'gray_flagged'})
        else:
            if r.get('true_label') == 1 and flagged == 0:
                fn_rows.append({'sample_id': r.get('sample_id'), 'fold': r.get('fold'),
                                'score': r.get('score'), 'flagged': flagged, 'role': 'false_negative'})
    if fn_rows:
        fn_df = pd.DataFrame(fn_rows).sort_values(['sample_id', 'fold'])
        fn_df.to_csv(output_dir / "false_negative_imds.csv", index=False)
        logger.info(f"Saved {len(fn_df)} pooled false-negative IMD sample(s) "
                    f"to {output_dir / 'false_negative_imds.csv'}")
    if fp_rows:
        fp_df = pd.DataFrame(fp_rows).sort_values(['sample_id', 'fold'])
        fp_df.to_csv(output_dir / "false_positive_imds.csv", index=False)
        logger.info(f"Saved {len(fp_df)} pooled false-positive sample(s) "
                    f"to {output_dir / 'false_positive_imds.csv'}")
    if gray_rows:
        gray_df = pd.DataFrame(gray_rows).sort_values(['sample_id', 'fold'])
        gray_df.to_csv(output_dir / "gray_flagged_imds.csv", index=False)
        logger.info(f"Saved {len(gray_df)} pooled gray-investigation sample(s) "
                    f"flagged as outliers to "
                    f"{output_dir / 'gray_flagged_imds.csv'}")

    # Generate the per-sample Z-score plots from the pooled FN/FP CSVs above
    # (one plot per unique sample_id). Read from the CSVs so the plots always
    # match the saved files; same helper used by the single-run path.
    if original_features is not None:
        try:
            _plot_fn_fp_zscore_analysis(
                original_features=original_features,
                output_dir=output_dir,
                config=config,
            )
        except Exception as e:
            logger.exception(f"FN/FP Z-score analysis failed: {e}")

    # Pooled deterministic univariate guardrail (deployment-time safety net,
    # OR-combined with the model flag). In confident-normals mode the model
    # only ever trains on confident normals, so use the full confident-normal
    # set (binary label 0) as the reference; every test sample (abnormals +
    # held-out normals across folds) is scored against it. Reference is
    # undefined in non-confident-normals schemes, so skip there.
    guardrail_flags: Dict[Any, List[str]] = {}
    try:
        if confident_normals_mode and original_features is not None:
            y_bin_pooled = (classification != normal_class).astype(int)
            pooled_normal_ids = classification.index[y_bin_pooled.values == 0]
            # Test = every sample that appears in any fold's test set.
            seen_test = set()
            for r in all_per_sample:
                seen_test.add(r.get('sample_id'))
            test_ids = pd.Index([s for s in original_features.index if s in seen_test])
            guardrail_flags = _apply_univariate_guardrail(
                original_features=original_features,
                reference_normal_ids=pooled_normal_ids,
                test_sample_ids=test_ids,
                output_dir=output_dir,
                config=config,
            )
            if guardrail_flags:
                logger.info(f"Univariate guardrail flagged {len(guardrail_flags)} "
                            f"pooled test sample(s) on configured biomarkers.")
    except Exception as e:
        logger.exception(f"Univariate guardrail failed: {e}")

    # Combined (model flag OR univariate guardrail) metrics, pooled across
    # all outer-CV folds. A true outlier is detected if EITHER the model or
    # the guardrail flags it (e.g. an IMD patient the model misses but the
    # guardrail catches is counted as detected). ROC-AUC is ranking quality of
    # the model score and is unchanged by the OR (the guardrail has no rank).
    combined: Dict[str, Any] = {}
    if all_per_sample:
        # Dedupe so each sample counts once: in confident_normals mode an
        # abnormal is scored in every fold, so keep one record per sample_id.
        deduped = {r.get('sample_id'): r for r in all_per_sample}
        try:
            combined = _compute_combined_guardrail_metrics(
                per_sample_results=list(deduped.values()),
                guardrail_flags=guardrail_flags,
                group_map=group_map,
                target_contamination=float(config.get('realistic_test_contamination', 0.02)),
            )
            if combined:
                _log_section_header("COMBINED (MODEL OR GUARDRAIL) POOLED RESULTS")
                logger.info(
                    f"Combined detection rate: {combined.get('detection_rate')}  "
                    f"({combined.get('n_detected')}/{combined.get('n_true_outlier')} true-outlier; "
                    f"model-only + guardrail rescues counted)")
                logger.info(f"Combined false positive rate: {combined.get('false_positive_rate')}  "
                            f"({combined.get('n_false_positives')}/{combined.get('n_true_inlier')} true-inlier)")
                logger.info(f"Combined precision: {combined.get('precision')}")
                logger.info(f"Combined F1: {combined.get('f1')}")
                logger.info(f"Combined accuracy: {combined.get('accuracy')}")
                logger.info(f"Combined ROC-AUC (= model-only; guardrail has no rank): {combined.get('roc_auc')}")
                # Audit CSV: per-sample combined flags.
                try:
                    pd.DataFrame(combined.get('per_sample', [])).to_csv(
                        output_dir / "combined_model_or_guardrail.csv", index=False)
                except Exception as e2:
                    logger.warning(f"Could not write combined audit CSV: {e2}")

                # True outliers missed by BOTH the model and the guardrail
                # (the combined system's residual false negatives). These are
                # the must-not-miss patients that slip through entirely.
                missed = [r for r in combined.get('per_sample', [])
                          if int(r.get('true_label', 0)) == 1
                          and int(r.get('model_flagged', 0)) == 0
                          and int(r.get('guardrail_flagged', 0)) == 0]
                # When a group_map is supplied, restrict the headline count to
                # the clean true-outlier role (Class 1 & Oordeel 1) so it is
                # consistent with the combined detection_rate denominator.
                if group_map is not None and len(group_map) > 0 and missed:
                    gm = group_map.copy()
                    gm['raw_classification'] = pd.to_numeric(gm.get('raw_classification'), errors='coerce')
                    gm['oordeel'] = pd.to_numeric(gm.get('oordeel'), errors='coerce')
                    gm = gm.dropna(subset=['oordeel'])
                    to_ids = {str(i) for i in gm[((gm['raw_classification'] == 1) & (gm['oordeel'] == 1))].index.tolist()}
                    missed = [r for r in missed if str(r.get('sample_id')) in to_ids]
                missed_ids = [str(r.get('sample_id')) for r in missed]
                logger.info(f"Missed by BOTH model and guardrail: {len(missed)} "
                            f"true-outlier sample(s){(' -> ' + str(missed_ids)) if missed_ids else ''}")
                if missed:
                    try:
                        pd.DataFrame(missed)[
                            ['sample_id', 'true_label', 'score',
                             'model_flagged', 'guardrail_flagged', 'flagged']
                        ].to_csv(output_dir / "missed_by_both.csv", index=False)
                        logger.info(f"Missed-by-both list saved to "
                                    f"{output_dir / 'missed_by_both.csv'}")
                    except Exception as e4:
                        logger.warning(f"Could not write missed-by-both CSV: {e4}")
        except Exception as e3:
            logger.exception(f"Combined guardrail metrics failed: {e3}")

    # Pooled per-group breakdown across all outer-CV folds.
    if group_map is not None and len(group_map) > 0 and all_per_sample:
        # Pooled threshold: mean of the per-fold calibrated thresholds.
        per_fold_thresholds = [
            fr.get('anomaly_threshold') for fr in fold_results
            if fr.get('anomaly_threshold') is not None
            and not (isinstance(fr.get('anomaly_threshold'), float)
                     and np.isnan(fr.get('anomaly_threshold')))
        ]
        pooled_threshold = float(np.mean(per_fold_thresholds)) if per_fold_thresholds else float('nan')
        # Dedupe so each sample counts once: in confident_normals mode an
        # abnormal is scored in every fold, so keep one record per sample_id
        # (the latest fold) to avoid double-counting in the pooled breakdown.
        deduped = {r.get('sample_id'): r for r in all_per_sample}
        _per_group_breakdown(
            per_sample_results=list(deduped.values()),
            anomaly_threshold=pooled_threshold,
            group_map=group_map,
            target_contamination=config.get('realistic_test_contamination', 0.02),
            output_dir=output_dir,
            label="pooled_outer_cv",
        )

    return {
        'evaluation_strategy': 'outer_cv',
        'n_folds': n_folds,
        'per_fold_results': fold_results,
        'aggregated': agg,
        'per_sample_results': all_per_sample,
        'combined': combined,
    }


def _run_components_sweep(
    features: pd.DataFrame,
    classification: pd.Series,
    normal_class: int,
    outlier_classes: List[int],
    config: Config,
    output_dir: Path,
    random_seed: int,
    original_features: Optional[pd.DataFrame] = None,
    group_map: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Sweep n_components to map the bias-variance tradeoff between detection
    rate and FPR for IsolationForest anomaly detection.

    Runs the full pipeline once per n_components value in
    `components_sweep_values`, using outer k-fold CV
    (`cv_outer_folds`) per value so each point is an honest out-of-sample
    estimate that averages out split noise. A detection-rate-vs-FPR curve
    over n_components is logged and saved as CSV + a plot, so you can pick
    the n_components that keeps FPR acceptable while maximizing detection.

    The sweep overrides the `n_components`, `use_sparse_pca`, and
    `evaluation_strategy` keys in memory for each iteration (PCA is forced
    on, and realistic evaluation is forced so detection rate / FPR are
    computed) and restores them afterwards; no YAML edit is needed.
    """
    sweep_values = config.get_list('components_sweep_values', [])
    if not sweep_values:
        sweep_values = [10, 20, 30, 40, 50, 60, 80, 100]
    sweep_values = [int(v) for v in sweep_values]
    # Each sweep point uses outer CV for an honest out-of-sample estimate;
    # default to 5 folds when the config leaves cv_outer_folds at 1, unless
    # confident_normals is active, in which case cv_outer_folds=1 means a
    # single confident-normal split (train on most normals, test on the rest
    # plus all abnormals).
    n_folds = int(config.get('cv_outer_folds', 1))
    if n_folds < 2 and config.get('classification_scheme', 'default') != 'confident_normals':
        n_folds = 5

    # Save the original config values so we can restore them after the sweep.
    orig_n_components = config.get('n_components', 100)
    orig_use_pca = config.get('use_sparse_pca', False)
    orig_cv_folds = config.get('cv_outer_folds', 1)
    # The sweep measures detection rate / FPR, which are only produced by the
    # realistic evaluation path. Force evaluation_strategy to 'realistic' for
    # the sweep iterations regardless of the user's setting, then restore it.
    orig_eval_strategy = config.get('evaluation_strategy', 'realistic')
    config.set('evaluation_strategy', 'realistic')

    _log_section_header(f"COMPONENTS SWEEP (n_components in {sweep_values})")
    logger.info(f"Each value uses outer CV with {n_folds} folds. "
                f"Outputs per value under sweep_components_<N>/.")

    sweep_rows = []
    for nc in sweep_values:
        _log_section_header(f"SWEEP POINT: n_components = {nc}")
        config.set('n_components', nc)
        config.set('use_sparse_pca', True)
        config.set('cv_outer_folds', n_folds)

        sweep_dir = output_dir / f"sweep_components_{nc}"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        result = _run_outer_cv(
            features=features,
            classification=classification,
            normal_class=normal_class,
            outlier_classes=outlier_classes,
            config=config,
            output_dir=sweep_dir,
            n_folds=n_folds,
            random_seed=random_seed,
            original_features=original_features,
            group_map=group_map,
        )

        agg = result.get('aggregated', {})
        comb = result.get('combined', {}) or {}
        row = {
            'n_components': nc,
            'detection_rate': agg.get('detection_rate_mean', float('nan')),
            'detection_rate_std': agg.get('detection_rate_std', float('nan')),
            'false_positive_rate': agg.get('false_positive_rate_mean', float('nan')),
            'false_positive_rate_std': agg.get('false_positive_rate_std', float('nan')),
            'roc_auc': agg.get('roc_auc_mean', float('nan')),
            'precision': agg.get('precision_mean', float('nan')),
            'f1': agg.get('f1_mean', float('nan')),
            'accuracy': agg.get('accuracy_mean', float('nan')),
            'anomaly_threshold': agg.get('anomaly_threshold_mean', float('nan')),
            # Combined (model flag OR univariate guardrail) metrics, pooled
            # across folds. A true outlier is detected if EITHER fires; an IMD
            # patient the model misses but the guardrail catches is counted as
            # detected. ROC-AUC is ranking quality of the model score and is
            # unchanged by the OR (the guardrail has no rank).
            'combined_detection_rate': comb.get('detection_rate', float('nan')),
            'combined_false_positive_rate': comb.get('false_positive_rate', float('nan')),
            'combined_precision': comb.get('precision', float('nan')),
            'combined_f1': comb.get('f1', float('nan')),
            'combined_accuracy': comb.get('accuracy', float('nan')),
            'combined_roc_auc': comb.get('roc_auc', float('nan')),
        }
        sweep_rows.append(row)
        logger.info(f"SWEEP POINT n_components={nc}: "
                    f"detection={row['detection_rate']:.3f}, "
                    f"FPR={row['false_positive_rate']:.3f}, "
                    f"roc_auc={row['roc_auc']:.3f}; "
                    f"combined det={row['combined_detection_rate']:.3f}, "
                    f"combined FPR={row['combined_false_positive_rate']:.3f}, "
                    f"combined F1={row['combined_f1']:.3f}")

    # Restore the original config values.
    config.set('n_components', orig_n_components)
    config.set('use_sparse_pca', orig_use_pca)
    config.set('cv_outer_folds', orig_cv_folds)
    config.set('evaluation_strategy', orig_eval_strategy)

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(output_dir / "components_sweep.csv", index=False)
    logger.info(f"Components sweep table saved to {output_dir / 'components_sweep.csv'}")

    _log_section_header("COMPONENTS SWEEP SUMMARY (model vs combined model+guardrail)")
    logger.info(f"{'n_comp':>7} {'detect':>8} {'FPR':>8} {'roc':>8} {'f1':>8} "
                f"{'comb_det':>9} {'comb_FPR':>9} {'comb_f1':>8} {'comb_roc':>8}")
    for _, r in sweep_df.iterrows():
        logger.info(f"{int(r['n_components']):>7d} "
                    f"{r['detection_rate']:>8.3f} "
                    f"{r['false_positive_rate']:>8.3f} "
                    f"{r['roc_auc']:>8.3f} "
                    f"{r['f1']:>8.3f} "
                    f"{r['combined_detection_rate']:>9.3f} "
                    f"{r['combined_false_positive_rate']:>9.3f} "
                    f"{r['combined_f1']:>8.3f} "
                    f"{r['combined_roc_auc']:>8.3f}")

    # Detection-vs-FPR tradeoff plot over n_components.
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend; avoids Tk 'main thread is not in main loop' errors on Windows
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(10, 6))
        x = sweep_df['n_components'].values
        ax1.plot(x, sweep_df['detection_rate'].values, 'o-', color='tab:red',
                 label='Detection rate (model, recall)')
        ax1.plot(x, sweep_df['false_positive_rate'].values, 's--', color='tab:blue',
                 label='False positive rate (model)')
        # Combined (model OR guardrail): detection can only rise, FPR can only
        # rise, vs the model-only curves.
        ax1.plot(x, sweep_df['combined_detection_rate'].values, 'D-', color='tab:orange',
                 label='Detection rate (model OR guardrail)')
        ax1.plot(x, sweep_df['combined_false_positive_rate'].values, 'x--', color='tab:purple',
                 label='False positive rate (model OR guardrail)')
        ax1.set_xlabel('n_components')
        ax1.set_ylabel('Rate')
        ax1.set_ylim(0, 1)
        ax1.axhline(config.get('realistic_test_contamination', 0.02), color='tab:blue',
                    linestyle=':', alpha=0.5, label='Nominal contamination (2%)')
        ax1.set_title('n_components sweep: detection rate vs FPR (model vs model+guardrail)')
        ax1.legend(loc='center right')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(x, sweep_df['roc_auc'].values, '^-.', color='tab:green',
                 label='ROC AUC')
        ax2.set_ylabel('ROC AUC')
        ax2.set_ylim(0, 1)
        ax2.legend(loc='lower right')

        fig.tight_layout()
        fig.savefig(output_dir / "components_sweep.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Components sweep plot saved to {output_dir / 'components_sweep.png'}")
    except Exception as e:
        logger.warning(f"Could not render components sweep plot: {e}")

    return {
        'evaluation_strategy': 'components_sweep',
        'sweep_values': sweep_values,
        'sweep_results': sweep_rows,
        'sweep_df': sweep_df,
    }


def run_pipeline(
    input_file: str,
    output_dir: str = "outputs/outlier_detection",
    config_path: str = None,
) -> Dict[str, Any]:
    """
    Run the complete outlier detection pipeline.

    Args:
        input_file: Path to merged_data_with_classification.csv
        output_dir: Output directory
        config_path: Path to config YAML file

    Returns:
        Dictionary with results and metrics
    """
    # Load configuration
    config = Config(config_path) if config_path else Config()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    _setup_logging(output_dir)

    logger.info(f"\n{'='*70}")
    logger.info("OUTLIER DETECTION PIPELINE")
    logger.info(f"{'='*70}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")

    # Step 1: Load data
    _log_section_header("STEP 1: Loading data")

    non_feature_cols = config.get_list('non_feature_columns', ['Oordeel targeted', 'Classification'])
    patient_id_col = config.get('patient_id_column', None)
    filter_to_endogenous = config.get('filter_to_endogenous', False)
    use_hmdb_cache = config.get('use_hmdb_cache', True)
    endogenous_metabolites_file = config.get('endogenous_metabolites_file', None)
    exclude_metabolites = config.get_list('exclude_metabolites', [])
    exclude_substrings = config.get_list('exclude_substrings', [])
    filter_to_smpdb_hmdb = config.get('filter_to_smpdb_hmdb', False)
    smpdb_pathways_file = config.get('smpdb_pathways_file', None)
    hmdb_xml_file = config.get('hmdb_xml_file', None)
    log_hmdb_tagged_features = config.get('log_hmdb_tagged_features', False)

    features, classification, oordeel, raw_classification = load_data(
        input_file=input_file,
        non_feature_columns=non_feature_cols,
        patient_id_column=patient_id_col,
        filter_to_endogenous=filter_to_endogenous,
        use_hmdb_cache=use_hmdb_cache,
        endogenous_metabolites_file=endogenous_metabolites_file,
        exclude_metabolites=exclude_metabolites,
        classification_scheme=config.get('classification_scheme', 'default'),
        exclude_substrings=exclude_substrings,
        smpdb_pathways_file=smpdb_pathways_file,
        filter_to_smpdb_hmdb=filter_to_smpdb_hmdb,
        hmdb_xml_file=hmdb_xml_file,
        log_hmdb_tagged_features=log_hmdb_tagged_features,
    )

    # Store original features for IQR analysis (before any filtering or PCA)
    original_features = features.copy()

    logger.info(f"Loaded {len(features)} samples with {len(features.columns)} features")
    logger.info(f"Classification distribution: {classification.value_counts().to_dict()}")

    # Step 1.2: Optional feature filtering
    feature_filter = config.get('feature_filter', None)
    features = _apply_feature_filter(features, feature_filter)

    # Build a sample_id -> (raw Classification, Oordeel) group map aligned to
    # the features index, for the per-group evaluation breakdown. Feature
    # filtering only drops columns, so the row index is unchanged.
    group_map = pd.DataFrame({
        'raw_classification': raw_classification.reindex(features.index),
        'oordeel': oordeel.reindex(features.index),
    }, index=features.index)

    # Step 2: Split data (stratified train-test split)
    _log_section_header("STEP 2: Splitting data (stratified train-test)")

    normal_class = config.get('normal_classification', 0)
    outlier_classes = config.get_list('outlier_classifications', [1, 2, 3])
    train_ratio = config.get('train_ratio', 0.8)
    test_ratio = config.get('test_ratio', 0.2)
    random_seed = config.get('random_seed', 42)

    cv_outer_folds = int(config.get('cv_outer_folds', 1))
    evaluation_strategy = config.get('evaluation_strategy', 'standard')

    if evaluation_strategy == 'components_sweep':
        # Sweep n_components to map the detection-rate vs FPR tradeoff. Runs
        # the full outer-CV pipeline once per value in
        # components_sweep_values and saves a detection-vs-FPR curve, so you
        # can pick the n_components that keeps FPR acceptable while
        # maximizing detection. Set `evaluation_strategy: components_sweep`
        # in the YAML to enable.
        return _run_components_sweep(
            features=features,
            classification=classification,
            normal_class=normal_class,
            outlier_classes=outlier_classes,
            config=config,
            output_dir=output_dir,
            random_seed=random_seed,
            original_features=original_features,
            group_map=group_map,
        )

    if cv_outer_folds > 1:
        # K-fold outer CV: each fold is a held-out test set, the rest is the
        # train set. Runs the full train->PCA->evaluate process per fold and
        # aggregates the per-fold results. This averages out an unlucky
        # single train/test split on small datasets and gives honest,
        # out-of-sample estimates of detection rate / FPR / ROC AUC across k
        # folds. Abnormal samples only ever appear in the test fold of their
        # own fold (never trained on), and are pooled across folds for metrics.
        return _run_outer_cv(
            features=features,
            classification=classification,
            normal_class=normal_class,
            outlier_classes=outlier_classes,
            config=config,
            output_dir=output_dir,
            n_folds=cv_outer_folds,
            random_seed=random_seed,
            original_features=original_features,
            group_map=group_map,
        )

    if config.get('classification_scheme', 'default') == 'confident_normals':
        # Confident-normals single-run split: train on (1-test_ratio) of the
        # confident normals; test on the held-out normal slice + ALL abnormals
        # (so every abnormal is scored, none wasted in a train split). The
        # model only fits on normals; the artificially enriched test set is
        # handled by the realistic eval reporting at the deployment prevalence.
        y_bin = (classification != normal_class).astype(int)
        normal_ids = classification.index[y_bin.values == 0]
        abnormal_ids = classification.index[y_bin.values == 1]
        from sklearn.model_selection import train_test_split
        # Split the confident normals by POSITION (not label): build a Series
        # of positional indices 0..len-1 carrying the normal_ids as its index,
        # split it, then select train/test normal labels via .loc on the Series'
        # own index. Using the positional values (not the .index labels) avoids
        # an IndexError when the sample ids are not 0..N-1 or are strings.
        pos_series = pd.Series(np.arange(len(normal_ids)), index=normal_ids)
        train_pos, test_pos = train_test_split(
            pos_series,
            train_size=train_ratio, test_size=test_ratio,
            random_state=random_seed,
        )
        train_idx = train_pos.index
        test_idx = test_pos.index.append(abnormal_ids)
        X_train = features.loc[train_idx].copy()
        y_train = classification.loc[train_idx].copy()
        X_test = features.loc[test_idx].copy()
        y_test = classification.loc[test_idx].copy()
        logger.info("confident_normals single-run split: train on "
                    f"{len(train_idx)} confident normals; test on "
                    f"{len(test_pos)} held-out normals + "
                    f"{len(abnormal_ids)} abnormals.")
    else:
        splits = split_data(
            features=features,
            classification=classification,
            normal_classification=normal_class,
            outlier_classifications=outlier_classes,
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )
        X_train, y_train = splits['train']
        X_test, y_test = splits['test']

    # Step 1.5: Optional PCA for dimensionality reduction
    X_train, X_test, y_train, y_test, pca = _apply_pca(
        X_train, X_test, y_train, y_test, config, output_dir, normal_class
    )

    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Test: {len(X_test)} samples")
    logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")

    # Step 3: Train the model. In the confident-normals single-run path the
    # model is fit once on the training normals (no inner k-fold CV), and the
    # held-out confident normals provide the out-of-sample score reference
    # for absolute-threshold calibration. With hyperparameter tuning, the
    # tuner's own CV still runs and produces the OOF reference.
    if config.get('classification_scheme', 'default') == 'confident_normals':
        heldout_normal_mask = y_test == normal_class
        reference_normals = (X_test[heldout_normal_mask]
                             if heldout_normal_mask.any() else None)
        if reference_normals is not None:
            logger.info(f"Using {len(reference_normals)} held-out normal test "
                        f"samples as the out-of-sample threshold reference.")
    else:
        reference_normals = None

    _log_section_header("STEP 3: Training Extended Isolation Forest")

    use_hyperparameter_tuning = config.get('use_hyperparameter_tuning', False)
    scorer_name = config.get('scorer', 'iforest')
    # Hyperparameter tuning is scorer-specific: iforest and ae have tuners;
    # other scorers ignore it and use the no-tuning path (fixed defaults /
    # scorer_kwargs from config).
    if use_hyperparameter_tuning and scorer_name == 'iforest':
        model, best_params = _train_with_hyperparameter_tuning(
            X_train, y_train, config, output_dir, normal_class
        )
        # Run CV with best model for evaluation
        cv_preds_train, train_scores, fold_scores = model.cross_val_predict(
            X=X_train,
            y=y_train,
            normal_classification=normal_class,
            n_splits=config.get('n_splits_tuning', 5),
        )
    elif use_hyperparameter_tuning and scorer_name == 'ae':
        model = _train_ae_with_tuning(
            X_train, y_train, config, output_dir, normal_class
        )
        cv_preds_train, train_scores, fold_scores = model.cross_val_predict(
            X=X_train,
            y=y_train,
            normal_classification=normal_class,
            n_splits=config.get('n_splits_tuning', 5),
        )
    else:
        model = _train_without_cv(X_train, y_train, config, normal_class,
                                  reference_normals=reference_normals)

    # Step 4: Evaluation (Standard or Realistic)
    evaluation_strategy = config.get('evaluation_strategy', 'standard')

    if evaluation_strategy == 'realistic':
        test_preds, test_scores, realistic_results = _evaluate_realistic(
            model, X_train, y_train, X_test, y_test, config, output_dir, normal_class, outlier_classes, group_map=group_map
        )
    else:
        test_preds, test_scores, test_metrics = _evaluate_standard(
            model, X_test, y_test, config, normal_class, outlier_classes
        )
        realistic_results = None

    # Step 5: Save outputs
    results = _save_outputs(
        model=model,
        X_test=X_test,
        y_test=y_test,
        test_preds=test_preds,
        test_scores=test_scores,
        config=config,
        output_dir=output_dir,
        normal_class=normal_class,
        outlier_classes=outlier_classes,
        test_metrics=test_metrics if evaluation_strategy != 'realistic' else None,
        realistic_results=realistic_results,
        pca=pca,
        original_features=original_features,
        group_map=group_map,
        reference_normal_ids=y_train[y_train == normal_class].index,
    )

    return results


def main():
    """Command-line interface for outlier detection pipeline."""
    parser = argparse.ArgumentParser(
        description="Run outlier detection pipeline with Extended Isolation Forest"
    )

    parser.add_argument(
        "--input",
        default=None,
        help="Path to merged_data_with_classification.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/outlier_detection",
        help="Output directory (default: outputs/outlier_detection)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML file (default: outlier_detection_pipeline/config/config.yaml)",
    )

    args = parser.parse_args()

    # Use default input from config if not provided
    if args.input is None:
        config = Config(args.config)
        args.input = config.get('input_file', 'data/merged_data_with_classification.csv')

    try:
        run_pipeline(
            input_file=args.input,
            output_dir=args.output,
            config_path=args.config,
        )
        logger.info("\nOutlier detection pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
