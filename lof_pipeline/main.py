#!/usr/bin/env python3
"""
Main entry point for the LOF outlier detection pipeline.

This pipeline performs Local Outlier Factor outlier detection on
merged_data_with_classification.csv with cross-validation.

Data structure:
- Input CSV has patient IDs as rows and features as columns
- Two non-feature columns: 'Oordeel targeted' and 'Classification'
- Classification 0 = normal (used for training)
- Classification 1, 2, 3 = outliers (split between validation and test)

Workflow:
1. Load data from merged_data_with_classification.csv
2. Split data into train/test sets
3. Train LOF on training normals with CV
4. Evaluate on test set
5. Save predictions, metrics, and model

Usage:
    python lof_pipeline/main.py
    python lof_pipeline/main.py --config config/custom.yaml
    python lof_pipeline/main.py --input data/my_data.csv --output outputs/my_run
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

from lof_pipeline.config.config import Config
from lof_pipeline.pipeline.data_loader import load_data, split_data
from lof_pipeline.pipeline.model import LOFModel
from lof_pipeline.pipeline.pca import SparsePCAWrapper
from lof_pipeline.pipeline.evaluation import (
    evaluate_model,
    print_metrics,
    save_metrics,
    save_predictions,
    plot_confusion_matrix,
    plot_precision_recall_curve,
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


def run_pipeline(
    input_file: str,
    output_dir: str = "outputs/lof_outlier_detection",
    config_path: str = None,
) -> Dict[str, Any]:
    """
    Run the complete LOF outlier detection pipeline.
    
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
    file_handler = logging.FileHandler(logs_dir / "lof_detection.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    logger.info(f"\n{'='*70}")
    logger.info("LOF OUTLIER DETECTION PIPELINE")
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
            features = features[[col for col in features.columns if feature_filter in col]]
        elif isinstance(feature_filter, list):
            features = features[[col for col in features.columns if any(s in col for s in feature_filter)]]
        
        n_filtered = original_n_features - len(features.columns)
        logger.info(f"Filtered features: {n_filtered} removed, {len(features.columns)} remaining (filter: {feature_filter})")
    
    # Step 2: Split data
    logger.info(f"\n{'='*70}")
    logger.info("STEP 2: Splitting data")
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
    
    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Test: {len(X_test)} samples")
    logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    # Step 1.5: Optional PCA for dimensionality reduction
    use_pca = config.get('use_pca', False)
    if use_pca:
        n_components = config.get('n_components', 100)
        pca_method = config.get('pca_method', 'regular')
        pca_random_state = config.get('pca_random_state', 42)
        save_pca_model = config.get('save_pca_model', True)
        nan_strategy = config.get('pca_nan_strategy', 'drop_columns')
        pca_batch_size = config.get('pca_batch_size', 1000)
        pca_alpha = config.get('pca_alpha', 1.0)
        pca_max_iter = config.get('pca_max_iter', 1000)
        pca_intermediate_components = config.get('pca_intermediate_components', None)
        
        logger.info(f"\n{'='*70}")
        logger.info("STEP 1.5: PCA Dimensionality Reduction")
        logger.info(f"{'='*70}")
        
        # Handle NaN values before PCA
        nan_count_train = X_train.isna().sum().sum()
        nan_count_test = X_test.isna().sum().sum()
        total_nan = nan_count_train + nan_count_test
        
        if total_nan > 0:
            logger.warning(f"Found {total_nan} NaN values in data. Strategy: {nan_strategy}")
            
            if nan_strategy == 'drop_columns':
                cols_with_nan = X_train.columns[X_train.isna().any()].tolist()
                n_dropped = len(cols_with_nan)
                X_train = X_train.dropna(axis=1)
                X_test = X_test.dropna(axis=1)
                logger.warning(f"Dropped {n_dropped} columns with NaN values: {cols_with_nan[:5]}{'...' if len(cols_with_nan) > 5 else ''}")
            elif nan_strategy == 'drop_rows':
                X_train = X_train.dropna(axis=0)
                y_train = y_train[X_train.index]
                X_test = X_test.dropna(axis=0)
                y_test = y_test[X_test.index]
                logger.warning("Dropped rows with NaN values")
            elif nan_strategy == 'impute_mean':
                X_train = X_train.fillna(X_train.mean())
                X_test = X_test.fillna(X_test.mean())
                logger.warning("Imputed NaN values with column means (separately for train/test)")
            else:
                raise ValueError(f"Unknown nan_strategy: {nan_strategy}")
        
        logger.info(f"Train shape after NaN handling: {X_train.shape}")
        logger.info(f"Test shape after NaN handling: {X_test.shape}")
        
        pca = SparsePCAWrapper(
            n_components=n_components,
            alpha=pca_alpha,
            max_iter=pca_max_iter,
            random_state=pca_random_state,
            method=pca_method,
            batch_size=pca_batch_size,
            intermediate_components=pca_intermediate_components,
        )
        
        # Fit PCA on NORMAL training data only (no data leakage)
        X_train_normals = X_train[y_train == normal_class]
        pca.fit(X_train_normals)
        
        # Transform normals and abnormalities separately
        X_train_transformed = pca.transform(X_train_normals)
        
        X_train_abnormals_mask = (y_train != normal_class)
        if X_train_abnormals_mask.any():
            X_train_abnormals = pca.transform(X_train[X_train_abnormals_mask])
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
    
    # Step 3: Train model with cross-validation
    logger.info(f"\n{'='*70}")
    logger.info("STEP 3: Training LOF with CV")
    logger.info(f"{'='*70}")
    
    n_neighbors = config.get('n_neighbors', 20)
    algorithm = config.get('algorithm', 'auto')
    leaf_size = config.get('leaf_size', 30)
    metric = config.get('metric', 'minkowski')
    p = config.get('p', 2)
    contamination = config.get('contamination', 'auto')
    novelty = config.get('novelty', False)
    n_jobs = config.get('n_jobs', -1)
    random_state = config.get('random_state', 42)
    n_splits = config.get('n_splits', 5)
    
    model = LOFModel(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=metric,
        p=p,
        contamination=contamination,
        novelty=novelty,
        n_jobs=n_jobs,
        random_state=random_state,
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
    
    # Step 4: Evaluate on test set
    logger.info(f"\n{'='*70}")
    logger.info("STEP 4: Evaluating on test set")
    logger.info(f"{'='*70}")
    
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
        test_preds = model.fit_predict(X_test)
    
    logger.info(f"Flagging {np.sum(test_preds == -1)} outliers (expected ~{n_outliers_expected})")
    
    # Compute metrics
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
    
    save_metrics(test_metrics, output_dir / "test")
    
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
        'model': model,
        'splits': {
            'train': len(X_train),
            'test': len(X_test),
        },
    }


def main():
    """Command-line interface for LOF outlier detection pipeline."""
    parser = argparse.ArgumentParser(
        description="Run LOF outlier detection pipeline"
    )
    
    parser.add_argument(
        "--input",
        default=None,
        help="Path to merged_data_with_classification.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/lof_outlier_detection",
        help="Output directory (default: outputs/lof_outlier_detection)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML file",
    )
    
    args = parser.parse_args()
    
    if args.input is None:
        config = Config(args.config)
        args.input = config.get('input_file', 'data/merged_data_with_classification.csv')
    
    try:
        run_pipeline(
            input_file=args.input,
            output_dir=args.output,
            config_path=args.config,
        )
        logger.info("\nLOF outlier detection pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
