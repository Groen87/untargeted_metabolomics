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
from outlier_detection_pipeline.pipeline.evaluation import (
    evaluate_model,
    print_metrics,
    save_metrics,
    save_predictions,
    plot_confusion_matrix,
    plot_precision_recall_curve,
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
    
    # Step 4: Evaluate on test set
    logger.info(f"\n{'='*70}")
    logger.info("STEP 4: Evaluating on test set")
    logger.info(f"{'='*70}")
    
    test_preds = model.predict(X_test)
    test_scores = model.decision_function(X_test)
    
    metrics_list = config.get_list('metrics', ['accuracy', 'f1', 'f1_weighted', 'precision', 'recall', 'roc_auc', 'confusion_matrix'])
    
    test_metrics = evaluate_model(
        y_true=y_test,
        y_pred=test_preds,
        y_scores=test_scores,
        metrics=metrics_list,
        pos_label=-1,  # Outliers are -1
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
        # Save test predictions
        save_predictions(
            predictions=test_preds,
            scores=test_scores,
            patient_ids=X_test.index,
            true_labels=y_test,
            output_dir=output_dir,
            split_name="test",
        )
        
        # Save training CV predictions
        save_predictions(
            predictions=cv_preds_train,
            scores=train_scores,
            patient_ids=X_train.index,
            true_labels=y_train,
            output_dir=output_dir,
            split_name="train_cv",
        )
    
    # Save metrics
    save_metrics(test_metrics, output_dir / "test")
    
    # Generate and save plots
    if save_plots:
        plot_confusion_matrix(
            y_true=y_test,
            y_pred=test_preds,
            output_dir=output_dir,
            outlier_classes=[1, 2, 3],
            pos_label=-1,
        )
        plot_precision_recall_curve(
            y_true=y_test,
            y_scores=test_scores,
            output_dir=output_dir,
            outlier_classes=[1, 2, 3],
            pos_label=-1,
        )
    
    # Save CV fold scores for analysis
    if save_preds:
        cv_results = pd.DataFrame({
            'patient_id': X_train.index,
            'true_label': y_train.values,
            'final_prediction': cv_preds_train,
            'final_score': train_scores,
        })
        # Add fold scores
        for i in range(n_splits):
            cv_results[f'fold_{i}_score'] = fold_scores[i] if i < len(fold_scores) else np.nan
        
        cv_results.to_csv(output_dir / "train_cv_detailed.csv", index=False)
        logger.info(f"Detailed CV results saved to {output_dir / 'train_cv_detailed.csv'}")
    
    logger.info(f"\n{'='*70}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'='*70}")
    
    return {
        'test_metrics': test_metrics,
        'model': model,
        'splits': {
            'train': len(X_train),
            'test': len(X_test),
        },
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
