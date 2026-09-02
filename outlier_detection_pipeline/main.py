#!/usr/bin/env python3
"""
Main entry point for the outlier detection pipeline.

This pipeline performs Extended Isolation Forest outlier detection on
merged_data_with_classification.csv with cross-validation.

Data structure:
- Input CSV has patient IDs as rows and features as columns
- Two non-feature columns: 'Oordeel trageted' and 'Classification'
- Classification 0 = normal (used for training)
- Classification 1, 2, 3 = outliers (split between validation and test)

Workflow:
1. Load data from merged_data_with_classification.csv
2. Split data:
   - 80% of Classification 0 -> train
   - 10% of Classification 0 -> validation
   - 10% of Classification 0 -> test
   - Classification 1,2,3 -> split between validation and test
3. Train Extended Isolation Forest on training set with CV
4. Evaluate on validation and test sets
5. Save predictions, metrics, and model

Usage:
    python outlier_detection_pipeline/main.py
    python outlier_detection_pipeline/main.py --config config/custom.yaml
    python outlier_detection_pipeline/main.py --input data/my_data.csv --output outputs/my_run
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

from outlier_detection_pipeline.config.config import Config
from outlier_detection_pipeline.pipeline.data_loader import load_data, split_data
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
from outlier_detection_pipeline.pipeline.realistic_evaluation import (
    run_realistic_evaluation,
    save_realistic_results,
    plot_realistic_results,
)

# Configure logging (stream handler only, file handler added later)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


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
    if config_path:
        config = Config(config_path)
    else:
        config = Config()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Add file handler for logging
    file_handler = logging.FileHandler(logs_dir / "outlier_detection.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    logger.info(f"\n{'='*70}")
    logger.info("OUTLIER DETECTION PIPELINE")
    logger.info(f"{'='*70}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Step 1: Load data
    logger.info(f"\n{'='*70}")
    logger.info("STEP 1: Loading data")
    logger.info(f"{'='*70}")
    
    non_feature_cols = config.get_list('non_feature_columns', ['Oordeel targeted', 'Classification'])
    patient_id_col = config.get('patient_id_column', None)
    
    features, classification, oordeel = load_data(
        input_file=input_file,
        non_feature_columns=non_feature_cols,
        patient_id_column=patient_id_col,
    )
    
    logger.info(f"Loaded {len(features)} samples with {len(features.columns)} features")
    logger.info(f"Classification distribution: {classification.value_counts().to_dict()}")

    # Step 1.2: Optional feature filtering
    feature_filter = config.get('feature_filter', None)
    if feature_filter:
        original_n_features = len(features.columns)
        if feature_filter == 'hmdb':
            features = features[[col for col in features.columns if 'HMDB' in col]]
        elif isinstance(feature_filter, str):
            # Custom substring filter
            features = features[[col for col in features.columns if feature_filter in col]]
        elif isinstance(feature_filter, list):
            # List of substrings - keep features containing any of them
            features = features[[col for col in features.columns if any(s in col for s in feature_filter)]]
        
        n_filtered = original_n_features - len(features.columns)
        logger.info(f"Filtered features: {n_filtered} removed, {len(features.columns)} remaining (filter: {feature_filter})")
    
    # Step 2: Split data (stratified train-test split)
    logger.info(f"\n{'='*70}")
    logger.info("STEP 2: Splitting data (stratified train-test)")
    logger.info(f"{'='*70}")
    
    normal_class = config.get('normal_classification', 0)
    outlier_classes = config.get_list('outlier_classifications', [1, 2, 3])
    train_ratio = config.get('train_ratio', 0.8)
    test_ratio = config.get('test_ratio', 0.2)
    random_seed = config.get('random_seed', 42)
    
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
    use_sparse_pca = config.get('use_sparse_pca', False)
    if use_sparse_pca:
        n_components = config.get('n_components', 100)
        alpha = config.get('alpha', 1.0)
        max_iter = config.get('max_iter', 1000)
        pca_random_state = config.get('pca_random_state', 42)
        save_pca_model = config.get('save_pca_model', True)
        nan_strategy = config.get('pca_nan_strategy', 'drop_columns')
        pca_method = config.get('pca_method', 'sparse')
        batch_size = config.get('pca_batch_size', 1000)
        intermediate_components = config.get('pca_intermediate_components', None)
        
        logger.info(f"\n{'='*70}")
        logger.info("STEP 1.5: PCA Dimensionality Reduction")
        logger.info(f"{'='*70}")
        
        # Handle NaN values before PCA (SparsePCA doesn't support NaN)
        # Apply to X_train and X_test separately to maintain consistency
        nan_count_train = X_train.isna().sum().sum()
        nan_count_test = X_test.isna().sum().sum()
        total_nan = nan_count_train + nan_count_test
        
        if total_nan > 0:
            logger.warning(f"Found {total_nan} NaN values in data. Strategy: {nan_strategy}")
            
            if nan_strategy == 'drop_columns':
                # Drop columns with NaN from both train and test
                # Use X_train to find columns with NaN (since X_train/X_test have same columns after split)
                cols_with_nan = X_train.columns[X_train.isna().any()].tolist()
                n_dropped = len(cols_with_nan)
                X_train = X_train.dropna(axis=1)
                X_test = X_test.dropna(axis=1)
                logger.warning(f"Dropped {n_dropped} columns with NaN values: {cols_with_nan[:5]}{'...' if len(cols_with_nan) > 5 else ''}")
            elif nan_strategy == 'drop_rows':
                # Drop rows with NaN from both train and test
                rows_with_nan_train = X_train.index[X_train.isna().any(axis=1)].tolist()
                rows_with_nan_test = X_test.index[X_test.isna().any(axis=1)].tolist()
                n_dropped = len(rows_with_nan_train) + len(rows_with_nan_test)
                X_train = X_train.dropna(axis=0)
                y_train = y_train[X_train.index]
                X_test = X_test.dropna(axis=0)
                y_test = y_test[X_test.index]
                logger.warning(f"Dropped {n_dropped} rows with NaN values")
            elif nan_strategy == 'impute_mean':
                # Impute separately for train and test to avoid leakage
                X_train = X_train.fillna(X_train.mean())
                X_test = X_test.fillna(X_test.mean())
                logger.warning("Imputed NaN values with column means (separately for train/test)")
            else:
                raise ValueError(f"Unknown nan_strategy: {nan_strategy}. Use 'drop_columns', 'drop_rows', or 'impute_mean'.")
        
        logger.info(f"Train shape after NaN handling: {X_train.shape}")
        logger.info(f"Test shape after NaN handling: {X_test.shape}")
        
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
        
        # Transform both train and test using the fitted PCA
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        
        logger.info(f"Features reduced from original to {X_train.shape[1]} components")
        
        if save_pca_model:
            pca.save(output_dir / "pca_model.joblib")
    
    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Test: {len(X_test)} samples")
    logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    # Step 3: Train model with cross-validation
    logger.info(f"\n{'='*70}")
    logger.info("STEP 3: Training Extended Isolation Forest with CV")
    logger.info(f"{'='*70}")
    
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
    
    # Train with cross-validation (unsupervised: train on normals only)
    logger.info(f"\n{'='*70}")
    logger.info("STEP 3: Training with CV (train on normals, validate on full)")
    logger.info(f"{'='*70}")
    
    n_splits = config.get('n_splits', 5)
    
    # Note: For Extended Isolation Forest (unsupervised):
    # - We do CV on the train set
    # - Each fold: train on normal samples only, validate on full fold (normals + abnormalities)
    # - Final model: trained on ALL normal samples from train set
    
    cv_preds_train, train_scores, fold_scores = model.cross_val_predict(
        X=X_train,
        y=y_train,
        normal_classification=normal_class,
        n_splits=n_splits,
    )
    
    logger.info("Training with cross-validation complete.")
    logger.info(f"Final model trained on all {len(y_train[y_train == normal_class])} normal samples from train set")
    

    # Step 4: Evaluation (Standard or Realistic)
    evaluation_strategy = config.get('evaluation_strategy', 'standard')
    save_realistic_results_flag = config.get('save_realistic_results', True)

    if evaluation_strategy == 'realistic':
        # Realistic evaluation: LOO abnormal with target contamination
        realistic_contamination = config.get('realistic_test_contamination', 0.02)
        realistic_n_iterations = config.get('realistic_n_iterations', 50)
        
        # Separate test set into normal and abnormal
        X_test_normal = X_test[y_test == normal_class]
        y_test_normal = y_test[y_test == normal_class]
        X_test_abnormal = X_test[y_test.isin(outlier_classes)]
        y_test_abnormal = y_test[y_test.isin(outlier_classes)]
        
        logger.info(f"\n{'='*70}")
        logger.info("STEP 4: Realistic Evaluation (LOO Abnormal)")
        logger.info(f"{'='*70}")
        
        # Use the CV-trained model (already trained on normals only)
        model_final = model  # model was trained on normals in cross_val_predict
        
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
            plot_realistic_results(realistic_results, output_dir)
        
        # Also run standard evaluation for comparison
        test_preds = model_final.predict(X_test)
        test_scores = model_final.decision_function(X_test)
        
    else:
        # Standard evaluation (original behavior)
        logger.info(f"\n{'='*70}")
        logger.info("STEP 4: Evaluating on test set")
        logger.info(f"{'='*70}")
        
        test_preds = model.predict(X_test)
        test_scores = model.decision_function(X_test)

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
    
    # Step 5: Save outputs
    logger.info(f"\n{'='*70}")
    logger.info("STEP 5: Saving outputs")
    logger.info(f"{'='*70}")
    
    save_plots = config.get('save_plots', True)
    save_model = config.get('save_model', True)
    save_preds = config.get('save_predictions', True)
    
    if save_model:
        if evaluation_strategy == 'realistic':
            model_final.save(output_dir / "model.joblib")
        else:
            model.save(output_dir / "model.joblib")
    
    if save_preds:
        # Save test predictions
        save_predictions(
            predictions=test_preds,
            scores=test_scores,
            patient_ids=X_test.index,
            true_labels=y_test,
            output_dir=output_dir,
            split_name="test",
        )
    
    
    # Compute test_metrics if not already done (for realistic evaluation path)
    if evaluation_strategy == 'realistic':
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
    
    # Save metrics
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
    
    return {
        'test_metrics': test_metrics,
        'model': model_final if evaluation_strategy == 'realistic' else model,
        'splits': {
            'train': len(X_train),
            'test': len(X_test),
        },
        'realistic_results': realistic_results if evaluation_strategy == 'realistic' else None,
    }


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
