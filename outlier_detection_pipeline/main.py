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
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/outlier_detection.log")
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
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*70}")
    logger.info("OUTLIER DETECTION PIPELINE")
    logger.info(f"{'='*70}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Step 1: Load data
    logger.info(f"\n{'='*70}")
    logger.info("STEP 1: Loading data")
    logger.info(f"{'='*70}")
    
    non_feature_cols = config.get_list('non_feature_columns', ['Oordeel trageted', 'Classification'])
    patient_id_col = config.get('patient_id_column', None)
    
    features, classification, oordeel = load_data(
        input_file=input_file,
        non_feature_columns=non_feature_cols,
        patient_id_column=patient_id_col,
    )
    
    logger.info(f"Loaded {len(features)} samples with {len(features.columns)} features")
    logger.info(f"Classification distribution: {classification.value_counts().to_dict()}")
    
    # Step 2: Split data
    logger.info(f"\n{'='*70}")
    logger.info("STEP 2: Splitting data")
    logger.info(f"{'='*70}")
    
    normal_class = config.get('normal_classification', 0)
    outlier_classes = config.get_list('outlier_classifications', [1, 2, 3])
    train_ratio = config.get('train_ratio', 0.8)
    test_ratio = config.get('test_ratio', 0.1)
    val_ratio = config.get('val_ratio', 0.1)
    random_seed = config.get('random_seed', 42)
    
    splits = split_data(
        features=features,
        classification=classification,
        normal_classification=normal_class,
        outlier_classifications=outlier_classes,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed,
    )
    
    X_train, y_train = splits['train']
    X_val, y_val = splits['validation']
    X_test, y_test = splits['test']
    
    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Validation: {len(X_val)} samples")
    logger.info(f"Test: {len(X_test)} samples")
    
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
    
    # Train with cross-validation
    cv_preds_train, train_scores = model.cross_val_predict(
        X=X_train,
        y=y_train,
        n_splits=n_splits,
    )
    
    logger.info("Training with cross-validation complete.")
    
    # Step 4: Evaluate on validation set
    logger.info(f"\n{'='*70}")
    logger.info("STEP 4: Evaluating on validation set")
    logger.info(f"{'='*70}")
    
    val_preds = model.predict(X_val)
    val_scores = model.decision_function(X_val)
    
    metrics_list = config.get_list('metrics', ['accuracy', 'f1', 'f1_weighted', 'precision', 'recall', 'roc_auc', 'confusion_matrix'])
    
    val_metrics = evaluate_model(
        y_true=y_val,
        y_pred=val_preds,
        y_scores=val_scores,
        metrics=metrics_list,
        pos_label=-1,  # Outliers are -1
    )
    
    print_metrics(val_metrics)
    
    # Step 5: Evaluate on test set
    logger.info(f"\n{'='*70}")
    logger.info("STEP 5: Evaluating on test set")
    logger.info(f"{'='*70}")
    
    test_preds = model.predict(X_test)
    test_scores = model.decision_function(X_test)
    
    test_metrics = evaluate_model(
        y_true=y_test,
        y_pred=test_preds,
        y_scores=test_scores,
        metrics=metrics_list,
        pos_label=-1,  # Outliers are -1
    )
    
    print_metrics(test_metrics)
    
    # Step 6: Save outputs
    logger.info(f"\n{'='*70}")
    logger.info("STEP 6: Saving outputs")
    logger.info(f"{'='*70}")
    
    save_plots = config.get('save_plots', True)
    save_model = config.get('save_model', True)
    save_preds = config.get('save_predictions', True)
    
    if save_model:
        model.save(output_dir / "model.joblib")
    
    if save_preds:
        # Save validation predictions
        save_predictions(
            predictions=val_preds,
            scores=val_scores,
            patient_ids=X_val.index,
            true_labels=y_val,
            output_dir=output_dir,
            split_name="validation",
        )
        
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
    save_metrics(val_metrics, output_dir / "validation")
    save_metrics(test_metrics, output_dir / "test")
    
    logger.info(f"\n{'='*70}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'='*70}")
    
    return {
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'model': model,
        'splits': {
            'train': len(X_train),
            'validation': len(X_val),
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
