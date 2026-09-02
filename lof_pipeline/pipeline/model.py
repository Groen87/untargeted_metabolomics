"""
Local Outlier Factor model module for outlier detection.

Handles model training, prediction, and cross-validation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class LOFModel:
    """
    Local Outlier Factor wrapper for outlier detection.
    
    This class wraps sklearn's LocalOutlierFactor with additional functionality:
    - Cross-validation support
    - Score normalization
    - Consistent interface with IsolationForest pipeline
    """
    
    def __init__(
        self,
        n_neighbors: int = 20,
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: int = 2,
        metric_params: Optional[Dict] = None,
        contamination: float = 0.1,
        novelty: bool = True,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        """
        Initialize the model.
        
        Args:
            n_neighbors: Number of neighbors to use
            algorithm: Algorithm for nearest neighbor search ('auto', 'ball_tree', 'kd_tree', 'brute')
            leaf_size: Leaf size for tree-based algorithms
            metric: Distance metric
            p: Power parameter for Minkowski metric
            metric_params: Additional metric parameters
            contamination: Expected proportion of outliers ('auto' or float)
            novelty: Whether to use novelty detection (requires contamination)
            n_jobs: Number of jobs for parallel processing
            random_state: Random seed
        """
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params or {}
        self.contamination = contamination
        self.novelty = novelty
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # If novelty=True, contamination must be a float, not 'auto'
        if self.novelty and self.contamination == 'auto':
            self.contamination = 0.1  # Default contamination for novelty detection
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted_ = False
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "LOFModel":
        """
        Fit the model on training data.
        
        Args:
            X: Training features
            y: Training labels (optional, not used for LOF)
            
        Returns:
            self
        """
        logger.info("Fitting Local Outlier Factor...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit model
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            contamination=self.contamination,
            novelty=self.novelty,
            n_jobs=self.n_jobs,
        )
        
        self.model.fit(X_scaled)
        self.is_fitted_ = True
        
        logger.info("Model fitting complete.")
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict outliers.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions (-1 for outliers, 1 for inliers)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly scores (negative LOF scores).
        
        Args:
            X: Features to score
            
        Returns:
            Anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        # LOF returns negative outlier scores, so we negate to match IF convention
        return -self.model.negative_outlier_factor_
    
    def cross_val_predict(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        normal_classification: int = 0,
        n_splits: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform cross-validated predictions for unsupervised outlier detection.
        
        For LOF:
        - Train on normal samples only
        - Validate on full folds
        
        Args:
            X: Features (train set with both normal and abnormal samples)
            y: Labels/classification (for identifying normal samples)
            normal_classification: Value indicating normal samples
            n_splits: Number of CV folds
            
        Returns:
            Tuple of (predictions, scores, fold_scores)
        """
        logger.info(f"Performing {n_splits}-fold cross-validation (unsupervised)...")
        logger.info(f"Training on normal samples only, validating on full folds")
        
        # Identify normal and abnormal samples in train set
        normal_mask = (y == normal_classification).values
        normal_indices = X.index[normal_mask]
        abnormal_indices = X.index[~normal_mask]
        
        logger.info(f"Train set: {len(X)} samples ({len(normal_indices)} normal, {len(abnormal_indices)} abnormal)")
        
        # Scale all data first
        X_scaled = self.scaler.fit_transform(X)
        
        from sklearn.model_selection import StratifiedKFold
        
        # Create labels for stratified splitting (0=normal, 1=abnormal)
        y_binary = (y != normal_classification).astype(int).values
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        fold_scores = []
        fold_predictions = []
        
        for fold_num, (train_fold_idx, val_fold_idx) in enumerate(skf.split(X_scaled, y_binary)):
            logger.info(f"Fold {fold_num + 1}/{n_splits}")
            
            # Get indices for this fold
            X_val_fold = X_scaled[val_fold_idx]
            
            # From training fold, extract only normal samples for training
            train_y_binary = y_binary[train_fold_idx]
            train_normal_positions = train_fold_idx[train_y_binary == 0]
            X_train_fold = X_scaled[train_normal_positions]
            
            # Train model on this fold
            fold_model = LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                metric=self.metric,
                p=self.p,
                metric_params=self.metric_params,
                contamination=self.contamination,
                novelty=self.novelty,
                n_jobs=self.n_jobs,
            )
            fold_model.fit(X_train_fold)
            
            # Get scores for validation fold
            val_scores = -fold_model.negative_outlier_factor_
            fold_scores.append(val_scores)
            
            # Get predictions for validation fold
            val_preds = fold_model.predict(X_val_fold)
            fold_predictions.append(val_preds)
            
            logger.debug(f"  Fold {fold_num + 1}: {len(X_train_fold)} train, {len(X_val_fold)} val samples")
        
        # After all folds, train final model on ALL normal samples from training set
        X_normal_all = X_scaled[normal_mask]
        
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            contamination=self.contamination,
            novelty=self.novelty,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X_normal_all)
        self.is_fitted_ = True
        
        # Get final scores and predictions on full X (training set)
        scores = -self.model.negative_outlier_factor_
        predictions = self.model.predict(X_scaled)
        
        logger.info("Cross-validation complete.")
        logger.info(f"Final model trained on {len(X_normal_all)} normal samples from training set")
        return predictions, scores, np.concatenate(fold_scores)
    
    def save(self, path: str) -> None:
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'n_neighbors': self.n_neighbors,
            'algorithm': self.algorithm,
            'leaf_size': self.leaf_size,
            'metric': self.metric,
            'p': self.p,
            'metric_params': self.metric_params,
            'contamination': self.contamination,
            'novelty': self.novelty,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'is_fitted_': self.is_fitted_,
        }
        
        joblib.dump(data, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "LOFModel":
        """Load model from file."""
        data = joblib.load(path)
        
        model = cls(
            n_neighbors=data['n_neighbors'],
            algorithm=data['algorithm'],
            leaf_size=data['leaf_size'],
            metric=data['metric'],
            p=data['p'],
            metric_params=data['metric_params'],
            contamination=data['contamination'],
            novelty=data['novelty'],
            n_jobs=data['n_jobs'],
            random_state=data['random_state'],
        )
        
        model.model = data['model']
        model.scaler = data['scaler']
        model.is_fitted_ = data['is_fitted_']
        
        logger.info(f"Model loaded from {path}")
        return model
