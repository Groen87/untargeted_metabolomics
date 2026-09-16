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
    save_outlier_analysis_results,
    analyze_outliers_log_iqr,
    plot_outlier_log_iqr,
    save_outlier_log_iqr_results,
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
) -> None:
    """Save a CSV of IMD samples flagged FALSE NEGATIVE (true_label=1, not flagged).

    Uses the per-sample rows produced by the realistic evaluation. Each row
    records the sample_id, score, and anomaly_threshold so you can inspect
    which IMDs slipped below the cutoff. The file is written next to the
    other fold outputs; pooled across outer-CV folds by the caller.
    """
    if realistic_results is None:
        return
    per_sample = realistic_results.get('per_iteration_results', [])
    if not per_sample:
        return

    rows = [
        {
            'sample_id': r.get('sample_id'),
            'true_label': r.get('true_label'),
            'flagged': r.get('flagged'),
            'score': r.get('score'),
            'anomaly_threshold': realistic_results.get('anomaly_threshold'),
        }
        for r in per_sample
        if r.get('true_label') == 1 and int(r.get('flagged', 0)) == 0
    ]
    if not rows:
        logger.info("No false-negative IMD samples (all abnormals flagged).")
        return

    fn_df = pd.DataFrame(rows)
    if not fn_df.empty and 'sample_id' in fn_df.columns:
        fn_df = fn_df.sort_values('sample_id')
    name = "false_negative_imds.csv" if fold is None else f"false_negative_imds_fold{fold}.csv"
    fn_df.to_csv(output_dir / name, index=False)
    logger.info(f"Saved {len(fn_df)} false-negative IMD sample(s) to {output_dir / name}")


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
) -> Dict[str, Any]:
    """Save all pipeline outputs."""
    _log_section_header("Saving outputs")

    save_plots = config.get('save_plots', True)
    save_model = config.get('save_model', True)
    save_preds = config.get('save_predictions', True)

    # Perform outlier analysis using original features (before PCA/filtering)
    outlier_mask = (test_preds == -1)
    outlier_indices = list(X_test.index[outlier_mask])
    log_iqr_feature_filter = config.get('log_iqr_feature_filter', None)
    use_zscore = config.get('use_zscore_analysis', True)

    if len(outlier_indices) > 0 and original_features is not None:
        if use_zscore:
            outlier_analysis = analyze_outliers(
                original_features, outlier_indices, n_top=20, 
                feature_filter=log_iqr_feature_filter, use_zscore=True
            )
            save_outlier_analysis_results(outlier_analysis, output_dir, n_top=20, use_zscore=True)
            plot_outlier_analysis(outlier_analysis, original_features, output_dir, n_top=20, use_zscore=True)
        else:
            outlier_analysis = analyze_outliers_log_iqr(
                original_features, outlier_indices, n_top=20, feature_filter=log_iqr_feature_filter
            )
            save_outlier_log_iqr_results(outlier_analysis, output_dir, n_top=20)
            plot_outlier_log_iqr(outlier_analysis, original_features, output_dir, n_top=20)

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

    # Save the sample ids of IMD samples flagged false negative (true_label=1, not flagged)
    # so they can be inspected manually.
    _save_false_negatives_csv(realistic_results, output_dir, fold=None)

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
            _save_false_negatives_csv(realistic_results, fold_out_dir, fold=fold_num + 1)
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

    # Pooled false-negative IMD sample ids across all outer-CV folds
    # (true_label=1, flagged=0) so they can be inspected manually.
    fn_rows = [
        {'sample_id': r.get('sample_id'), 'fold': r.get('fold'),
         'score': r.get('score'), 'flagged': r.get('flagged')}
        for r in all_per_sample
        if r.get('true_label') == 1 and int(r.get('flagged', 0)) == 0
    ]
    if fn_rows:
        fn_df = pd.DataFrame(fn_rows).sort_values(['sample_id', 'fold'])
        fn_df.to_csv(output_dir / "false_negative_imds.csv", index=False)
        logger.info(f"Saved {len(fn_df)} pooled false-negative IMD sample(s) "
                    f"to {output_dir / 'false_negative_imds.csv'}")

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
        }
        sweep_rows.append(row)
        logger.info(f"SWEEP POINT n_components={nc}: "
                    f"detection={row['detection_rate']:.3f}, "
                    f"FPR={row['false_positive_rate']:.3f}, "
                    f"roc_auc={row['roc_auc']:.3f}")

    # Restore the original config values.
    config.set('n_components', orig_n_components)
    config.set('use_sparse_pca', orig_use_pca)
    config.set('cv_outer_folds', orig_cv_folds)
    config.set('evaluation_strategy', orig_eval_strategy)

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(output_dir / "components_sweep.csv", index=False)
    logger.info(f"Components sweep table saved to {output_dir / 'components_sweep.csv'}")

    _log_section_header("COMPONENTS SWEEP SUMMARY (detection vs FPR vs n_components)")
    logger.info(f"{'n_comp':>7} {'detect':>8} {'FPR':>8} {'roc_auc':>8} {'prec':>8} {'f1':>8}")
    for _, r in sweep_df.iterrows():
        logger.info(f"{int(r['n_components']):>7d} "
                    f"{r['detection_rate']:>8.3f} "
                    f"{r['false_positive_rate']:>8.3f} "
                    f"{r['roc_auc']:>8.3f} "
                    f"{r['precision']:>8.3f} "
                    f"{r['f1']:>8.3f}")

    # Detection-vs-FPR tradeoff plot over n_components.
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend; avoids Tk 'main thread is not in main loop' errors on Windows
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(10, 6))
        x = sweep_df['n_components'].values
        ax1.plot(x, sweep_df['detection_rate'].values, 'o-', color='tab:red',
                 label='Detection rate (recall)')
        ax1.plot(x, sweep_df['false_positive_rate'].values, 's--', color='tab:blue',
                 label='False positive rate')
        ax1.set_xlabel('n_components')
        ax1.set_ylabel('Rate')
        ax1.set_ylim(0, 1)
        ax1.axhline(config.get('realistic_test_contamination', 0.02), color='tab:blue',
                    linestyle=':', alpha=0.5, label='Nominal contamination (2%)')
        ax1.set_title('n_components sweep: detection rate vs FPR')
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
        train_normal_ids, test_normal_ids = train_test_split(
            pd.Series(range(len(normal_ids)), index=normal_ids),
            train_size=train_ratio, test_size=test_ratio,
            random_state=random_seed,
        )
        train_idx = normal_ids[train_normal_ids.index]
        test_idx = normal_ids[test_normal_ids.index].append(abnormal_ids)
        X_train = features.loc[train_idx].copy()
        y_train = classification.loc[train_idx].copy()
        X_test = features.loc[test_idx].copy()
        y_test = classification.loc[test_idx].copy()
        logger.info("confident_normals single-run split: train on "
                    f"{len(train_idx)} confident normals; test on "
                    f"{len(test_normal_ids)} held-out normals + "
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
