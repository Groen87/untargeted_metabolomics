"""
Hyperparameter tuning module for Isolation Forest outlier detection pipeline.

Implements:
- Grid search over hyperparameter combinations
- Cross-validation with training on normals only
- Evaluation on validation set
- Best model selection based on metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    normal_classification: int = 0,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    scoring: str = 'f1',
    refit: bool = True,
) -> Tuple[IsolationForest, Dict[str, Any], pd.DataFrame]:
    """
    Perform hyperparameter tuning using grid search with cross-validation.
    
    For unsupervised outlier detection:
    - Train on normal samples only
    - Validate on full folds (normals + abnormalities)
    - Use metrics appropriate for outlier detection
    
    Args:
        X: Training features (with both normal and abnormal samples)
        y: Training labels/classification
        normal_classification: Value indicating normal samples
        param_grid: Dictionary of hyperparameters to search
        n_splits: Number of CV folds
        random_state: Random seed
        n_jobs: Number of jobs for parallel processing
        scoring: Metric to optimize ('f1', 'precision', 'recall', 'roc_auc')
        refit: Whether to refit best model on all normal data
        
    Returns:
        Tuple of:
        - best_model: Best sklearn IsolationForest model
        - best_params: Best hyperparameters
        - results: DataFrame with all results
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_samples': ['auto', 0.5, 0.8],
            'max_features': [0.5, 0.8, 1.0],
            'contamination': ['auto'],
            'bootstrap': [False, True],
        }
    
    logger.info(f"\n{'='*70}")
    logger.info("HYPERPARAMETER TUNING")
    logger.info(f"{'='*70}")
    logger.info(f"Parameter grid: {param_grid}")
    logger.info(f"Scoring metric: {scoring}")
    logger.info(f"CV folds: {n_splits}")
    
    # Identify normal and abnormal samples
    normal_mask = (y == normal_classification).values
    normal_indices = X.index[normal_mask]
    abnormal_indices = X.index[~normal_mask]
    
    logger.info(f"Training on {len(normal_indices)} normal samples")
    logger.info(f"Validating on {len(X)} samples ({len(normal_indices)} normal, {len(abnormal_indices)} abnormal)")
    
    # Scale all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_normal_scaled = X_scaled[normal_mask]
    
    # Create labels for stratified splitting
    y_binary = (y != normal_classification).astype(int).values
    
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Generate all parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    logger.info(f"Testing {len(param_combinations)} parameter combinations")
    
    results = []
    best_score = -np.inf
    best_params = None
    best_model = None
    
    for combo_idx, params in enumerate(param_combinations):
        logger.info(f"\nTesting combination {combo_idx + 1}/{len(param_combinations)}: {params}")
        
        fold_scores = []
        
        for fold_num, (train_fold_idx, val_fold_idx) in enumerate(skf.split(X_scaled, y_binary)):
            # Train on normal samples from training fold only
            train_y_binary = y_binary[train_fold_idx]
            train_normal_positions = train_fold_idx[train_y_binary == 0]
            X_train_fold = X_scaled[train_normal_positions]
            
            # Validate on full validation fold
            X_val_fold = X_scaled[val_fold_idx]
            y_val_fold = y.iloc[val_fold_idx]
            
            # Train model
            model = IsolationForest(
                n_estimators=params['n_estimators'],
                max_samples=params['max_samples'],
                max_features=params['max_features'],
                bootstrap=params['bootstrap'],
                n_jobs=n_jobs,
                random_state=random_state + combo_idx * 100 + fold_num,
                contamination=params.get('contamination', 'auto'),
            )
            model.fit(X_train_fold)
            
            # Get predictions and scores for validation fold
            val_preds = model.predict(X_val_fold)
            val_scores = model.decision_function(X_val_fold)
            
            # Convert to binary
            y_true_binary = (y_val_fold != normal_classification).astype(int)
            y_pred_binary = (val_preds == -1).astype(int)
            
            # Compute metric
            if scoring == 'f1':
                from sklearn.metrics import f1_score
                try:
                    score = f1_score(y_true_binary, y_pred_binary, pos_label=1)
                except:
                    score = 0.0
            elif scoring == 'precision':
                from sklearn.metrics import precision_score
                try:
                    score = precision_score(y_true_binary, y_pred_binary, pos_label=1)
                except:
                    score = 0.0
            elif scoring == 'recall':
                from sklearn.metrics import recall_score
                try:
                    score = recall_score(y_true_binary, y_pred_binary, pos_label=1)
                except:
                    score = 0.0
            elif scoring == 'roc_auc':
                from sklearn.metrics import roc_auc_score
                try:
                    score = roc_auc_score(y_true_binary, -val_scores)
                except:
                    score = 0.0
            else:
                from sklearn.metrics import f1_score
                try:
                    score = f1_score(y_true_binary, y_pred_binary, pos_label=1)
                except:
                    score = 0.0
            
            fold_scores.append(score)
        
        # Average score across folds
        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        results.append({
            'combination': combo_idx,
            **params,
            'mean_score': avg_score,
            'std_score': std_score,
            'fold_scores': fold_scores,
        })
        
        logger.info(f"  Score: {avg_score:.4f} +/- {std_score:.4f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = params.copy()
            # Refit best model on all normal data
            if refit:
                best_model = IsolationForest(
                    n_estimators=params['n_estimators'],
                    max_samples=params['max_samples'],
                    max_features=params['max_features'],
                    bootstrap=params['bootstrap'],
                    n_jobs=n_jobs,
                    random_state=random_state,
                    contamination=params.get('contamination', 'auto'),
                )
                best_model.fit(X_normal_scaled)
        
        logger.info(f"  Current best: {best_score:.4f} with params: {best_params}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    logger.info(f"\n{'='*70}")
    logger.info("TUNING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best score ({scoring}): {best_score:.4f}")
    
    return best_model, best_params, results_df


def tune_and_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    normal_classification: int = 0,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    scoring: str = 'f1',
    output_dir: Optional[Path] = None,
) -> Tuple[IsolationForest, StandardScaler, Dict[str, Any], pd.DataFrame]:
    """
    Convenience function: tune hyperparameters and return best model with scaler.
    
    Returns:
        Tuple of:
        - best_model: Best sklearn IsolationForest
        - scaler: Fitted StandardScaler
        - best_params: Best hyperparameters
        - results_df: All tuning results
    """
    logger.info("Starting hyperparameter tuning...")
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Tune
    best_model, best_params, results_df = tune_hyperparameters(
        X=X_train,
        y=y_train,
        normal_classification=normal_classification,
        param_grid=param_grid,
        n_splits=n_splits,
        random_state=random_state,
        n_jobs=n_jobs,
        scoring=scoring,
        refit=True,
    )
    
    # Save results if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_dir / "tuning_results.csv", index=False)
        logger.info(f"Tuning results saved to {output_dir / 'tuning_results.csv'}")
        
        with open(output_dir / "best_params.json", 'w') as f:
            import json
            json.dump(best_params, f, indent=2)
        logger.info(f"Best parameters saved to {output_dir / 'best_params.json'}")
    
    return best_model, scaler, best_params, results_df
