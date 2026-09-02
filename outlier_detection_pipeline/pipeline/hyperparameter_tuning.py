"""
Hyperparameter tuning module for Isolation Forest outlier detection pipeline.

Implements:
- Grid search over hyperparameter combinations
- Optuna-based hyperparameter optimization
- Cross-validation with training on normals only
- Evaluation on validation set
- Best model selection based on metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import logging
import joblib
from pathlib import Path

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Grid search will be used as fallback.")

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
    use_optuna: bool = False,
    n_trials: int = 100,
    optuna_sampler: str = 'tpe',
    optuna_pruner: Optional[str] = 'median',
    study_name: Optional[str] = None,
    storage_url: Optional[str] = None,
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
    logger.info(f"Method: {'Optuna' if use_optuna and OPTUNA_AVAILABLE else 'Grid Search'}")
    logger.info(f"Parameter grid: {param_grid}")
    logger.info(f"Scoring metric: {scoring}")
    logger.info(f"CV folds: {n_splits}")
    
    # Use Optuna if requested and available
    if use_optuna and OPTUNA_AVAILABLE:
        return _tune_with_optuna(
            X=X,
            y=y,
            normal_classification=normal_classification,
            param_grid=param_grid,
            n_splits=n_splits,
            random_state=random_state,
            n_jobs=n_jobs,
            scoring=scoring,
            refit=refit,
            n_trials=n_trials,
            optuna_sampler=optuna_sampler,
            optuna_pruner=optuna_pruner,
            study_name=study_name,
            storage_url=storage_url,
        )
    
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
            score = _compute_metric(y_true_binary, y_pred_binary, val_scores, scoring)
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


def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray, scoring: str) -> float:
    """Compute the specified metric."""
    try:
        if scoring == 'f1':
            from sklearn.metrics import f1_score
            return f1_score(y_true, y_pred, pos_label=1)
        elif scoring == 'precision':
            from sklearn.metrics import precision_score
            return precision_score(y_true, y_pred, pos_label=1)
        elif scoring == 'recall':
            from sklearn.metrics import recall_score
            return recall_score(y_true, y_pred, pos_label=1)
        elif scoring == 'roc_auc':
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, -scores)
        else:
            from sklearn.metrics import f1_score
            return f1_score(y_true, y_pred, pos_label=1)
    except Exception as e:
        logger.warning(f"Error computing metric {scoring}: {e}")
        return 0.0


def _tune_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    normal_classification: int = 0,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    scoring: str = 'f1',
    refit: bool = True,
    n_trials: int = 100,
    optuna_sampler: str = 'tpe',
    optuna_pruner: Optional[str] = 'median',
    study_name: Optional[str] = None,
    storage_url: Optional[str] = None,
) -> Tuple[IsolationForest, Dict[str, Any], pd.DataFrame]:
    """
    Perform hyperparameter tuning using Optuna.
    
    Args:
        X: Training features
        y: Training labels
        normal_classification: Value indicating normal samples
        param_grid: Parameter grid for search space definition
        n_splits: Number of CV folds
        random_state: Random seed
        n_jobs: Number of jobs for parallel processing
        scoring: Metric to optimize
        refit: Whether to refit best model on all normal data
        n_trials: Number of Optuna trials
        optuna_sampler: Sampler type ('tpe', 'random', 'cmaes')
        optuna_pruner: Pruner type ('median', 'halving', None)
        study_name: Name for Optuna study
        storage_url: URL for Optuna storage (e.g., 'sqlite:///optuna.db')
        
    Returns:
        Tuple of (best_model, best_params, results_df)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not available. Please install with: pip install optuna")
    
    logger.info(f"\n{'='*70}")
    logger.info("OPTUNA HYPERPARAMETER TUNING")
    logger.info(f"{'='*70}")
    logger.info(f"Trials: {n_trials}")
    logger.info(f"Sampler: {optuna_sampler}")
    logger.info(f"Pruner: {optuna_pruner}")
    logger.info(f"Scoring: {scoring}")
    logger.info(f"CV folds: {n_splits}")
    
    # Identify normal and abnormal samples
    normal_mask = (y == normal_classification).values
    X_normal = X[normal_mask]
    
    # Scale all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_normal_scaled = X_scaled[normal_mask]
    
    # Create labels for stratified splitting
    y_binary = (y != normal_classification).astype(int).values
    
    # Define search space from param_grid
    search_space = _create_optuna_search_space(param_grid, random_state)
    
    # Create Optuna study
    direction = 'maximize'
    
    sampler_map = {
        'tpe': TPESampler(seed=random_state),
        'random': RandomSampler(seed=random_state),
        'cmaes': None,  # Will use default
    }
    sampler = sampler_map.get(optuna_sampler.lower(), TPESampler(seed=random_state))
    
    pruner_map = {
        'median': MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        'halving': SuccessiveHalvingPruner(min_resource=1, reduction_factor=4),
        None: None,
    }
    pruner = pruner_map.get(optuna_pruner.lower() if optuna_pruner else None, None)
    
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        study_name=study_name,
    )
    
    # Create objective function
    objective = _create_optuna_objective(
        X_scaled=X_scaled,
        y=y,
        y_binary=y_binary,
        normal_classification=normal_classification,
        n_splits=n_splits,
        random_state=random_state,
        n_jobs=n_jobs,
        scoring=scoring,
        search_space=search_space,
    )
    
    # Run optimization
    logger.info(f"Starting Optuna optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    
    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value
    
    logger.info(f"\n{'='*70}")
    logger.info("OPTUNA TUNING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best score ({scoring}): {best_score:.4f}")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    
    # Convert best_params to standard format
    best_params_standard = {
        'n_estimators': int(best_params.get('n_estimators', 100)),
        'max_samples': best_params.get('max_samples', 'auto'),
        'max_features': float(best_params.get('max_features', 1.0)),
        'bootstrap': bool(best_params.get('bootstrap', False)),
        'contamination': best_params.get('contamination', 'auto'),
    }
    
    # Refit best model on all normal data
    best_model = None
    if refit:
        best_model = IsolationForest(
            n_estimators=best_params_standard['n_estimators'],
            max_samples=best_params_standard['max_samples'],
            max_features=best_params_standard['max_features'],
            bootstrap=best_params_standard['bootstrap'],
            n_jobs=n_jobs,
            random_state=random_state,
            contamination=best_params_standard['contamination'],
        )
        best_model.fit(X_normal_scaled)
    
    # Create results DataFrame
    results = []
    for trial in study.trials:
        results.append({
            'trial_number': trial.number,
            'value': trial.value,
            **trial.params,
            'state': trial.state,
            'duration': trial.duration.total_seconds() if trial.duration else None,
        })
    results_df = pd.DataFrame(results)
    
    return best_model, best_params_standard, results_df


def _create_optuna_search_space(param_grid: Dict[str, List[Any]], random_state: int) -> Dict[str, Any]:
    """Create Optuna search space from parameter grid."""
    search_space = {}
    
    for param, values in param_grid.items():
        if param == 'n_estimators':
            # Integer parameter
            if isinstance(values, list):
                search_space[param] = optuna.suggest_int(param, min(values), max(values))
            else:
                search_space[param] = optuna.suggest_int(param, 50, 500)
        elif param == 'max_samples':
            # Can be 'auto' or float
            if isinstance(values, list):
                float_values = [v for v in values if isinstance(v, (int, float))]
                if float_values:
                    search_space[param] = optuna.suggest_float(param, min(float_values), max(float_values))
                else:
                    search_space[param] = optuna.suggest_categorical(param, values)
            else:
                search_space[param] = optuna.suggest_categorical(param, ['auto', 0.5, 0.8, 1.0])
        elif param == 'max_features':
            # Float parameter
            if isinstance(values, list):
                search_space[param] = optuna.suggest_float(param, min(values), max(values))
            else:
                search_space[param] = optuna.suggest_float(param, 0.1, 1.0)
        elif param == 'contamination':
            # Can be 'auto' or float
            if isinstance(values, list):
                search_space[param] = optuna.suggest_categorical(param, values)
            else:
                search_space[param] = optuna.suggest_categorical(param, ['auto'])
        elif param == 'bootstrap':
            # Boolean parameter
            search_space[param] = optuna.suggest_categorical(param, [True, False])
        else:
            # Default: categorical
            search_space[param] = optuna.suggest_categorical(param, values)
    
    return search_space


def _create_optuna_objective(
    X_scaled: np.ndarray,
    y: pd.Series,
    y_binary: np.ndarray,
    normal_classification: int,
    n_splits: int,
    random_state: int,
    n_jobs: int,
    scoring: str,
    search_space: Dict[str, Any],
) -> Callable[[optuna.Trial], float]:
    """Create the Optuna objective function."""
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna."""
        # Sample parameters
        params = {}
        for param, suggest_func in search_space.items():
            params[param] = suggest_func(trial)
        
        # Ensure parameter types are correct
        params['n_estimators'] = int(params.get('n_estimators', 100))
        params['max_features'] = float(params.get('max_features', 1.0))
        params['bootstrap'] = bool(params.get('bootstrap', False))
        
        # If max_samples is categorical and got a float, handle it
        if 'max_samples' in params and isinstance(params['max_samples'], (int, float)):
            # Clamp to valid range
            params['max_samples'] = float(np.clip(params['max_samples'], 0.1, 1.0))
        
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
                max_samples=params.get('max_samples', 'auto'),
                max_features=params['max_features'],
                bootstrap=params['bootstrap'],
                n_jobs=n_jobs,
                random_state=random_state + fold_num,
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
            score = _compute_metric(y_true_binary, y_pred_binary, val_scores, scoring)
            fold_scores.append(score)
        
        # Return average score across folds
        return float(np.mean(fold_scores))
    
    return objective


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
    use_optuna: bool = False,
    n_trials: int = 100,
    optuna_sampler: str = 'tpe',
    optuna_pruner: Optional[str] = 'median',
    study_name: Optional[str] = None,
    storage_url: Optional[str] = None,
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
        use_optuna=use_optuna,
        n_trials=n_trials,
        optuna_sampler=optuna_sampler,
        optuna_pruner=optuna_pruner,
        study_name=study_name,
        storage_url=storage_url,
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
