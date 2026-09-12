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

    model = ExtendedIsolationForestModel(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=random_state,
        contamination=contamination,
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

    y_binary = (classification != normal_class).astype(int).values
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    fold_results = []
    all_per_sample = []

    metrics_keys = ['detection_rate', 'false_positive_rate', 'roc_auc',
                   'precision', 'f1', 'accuracy', 'anomaly_threshold']

    for fold_num, (train_idx, test_idx) in enumerate(skf.split(features, y_binary)):
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

        # Train model on this fold
        use_hp_tuning = config.get('use_hyperparameter_tuning', False)
        if use_hp_tuning:
            model, _ = _train_with_hyperparameter_tuning(
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
            )
            fold_metrics = {k: realistic_results.get(k) for k in metrics_keys}
            fold_metrics['n_normal_test'] = realistic_results.get('n_normal_test')
            fold_metrics['n_abnormal_test'] = realistic_results.get('n_abnormal_test')
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

    return {
        'evaluation_strategy': 'outer_cv',
        'n_folds': n_folds,
        'per_fold_results': fold_results,
        'aggregated': agg,
        'per_sample_results': all_per_sample,
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

    features, classification, oordeel = load_data(
        input_file=input_file,
        non_feature_columns=non_feature_cols,
        patient_id_column=patient_id_col,
        filter_to_endogenous=filter_to_endogenous,
        use_hmdb_cache=use_hmdb_cache,
        endogenous_metabolites_file=endogenous_metabolites_file,
        exclude_metabolites=exclude_metabolites,
    )

    # Store original features for IQR analysis (before any filtering or PCA)
    original_features = features.copy()

    logger.info(f"Loaded {len(features)} samples with {len(features.columns)} features")
    logger.info(f"Classification distribution: {classification.value_counts().to_dict()}")

    # Step 1.2: Optional feature filtering
    feature_filter = config.get('feature_filter', None)
    features = _apply_feature_filter(features, feature_filter)

    # Step 2: Split data (stratified train-test split)
    _log_section_header("STEP 2: Splitting data (stratified train-test)")

    normal_class = config.get('normal_classification', 0)
    outlier_classes = config.get_list('outlier_classifications', [1, 2, 3])
    train_ratio = config.get('train_ratio', 0.8)
    test_ratio = config.get('test_ratio', 0.2)
    random_seed = config.get('random_seed', 42)

    cv_outer_folds = int(config.get('cv_outer_folds', 1))
    evaluation_strategy = config.get('evaluation_strategy', 'standard')

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
        )

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

    # Step 3: Train model with cross-validation
    _log_section_header("STEP 3: Training Extended Isolation Forest with CV")

    use_hyperparameter_tuning = config.get('use_hyperparameter_tuning', False)

    if use_hyperparameter_tuning:
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
    else:
        model = _train_without_tuning(X_train, y_train, config, normal_class)

    # Step 4: Evaluation (Standard or Realistic)
    evaluation_strategy = config.get('evaluation_strategy', 'standard')

    if evaluation_strategy == 'realistic':
        test_preds, test_scores, realistic_results = _evaluate_realistic(
            model, X_train, y_train, X_test, y_test, config, output_dir, normal_class, outlier_classes
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
