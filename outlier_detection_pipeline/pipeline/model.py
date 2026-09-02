"""
Extended Isolation Forest model module for outlier detection.

Handles model training, prediction, and cross-validation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class ExtendedIsolationForestModel:
    """
    Extended Isolation Forest wrapper for outlier detection.
    
    This class wraps sklearn's IsolationForest with additional functionality:
    - Cross-validation support
    - Threshold tuning
    - Score normalization
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: str = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        n_jobs: int = -1,
        random_state: int = 42,
        contamination: str = "auto",
    ):
        """
        Initialize the model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_samples: Number of samples to draw for each tree
            max_features: Number of features to draw for each tree
            bootstrap: Whether to use bootstrap sampling
            n_jobs: Number of jobs for parallel processing
            random_state: Random seed
            contamination: Expected proportion of outliers
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.contamination = contamination
        
        self.model = None
        self.scaler = StandardScaler()
        self.threshold_ = None
        self.is_fitted_ = False
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "ExtendedIsolationForestModel":
        """
        Fit the model on training data.
        
        Args:
            X: Training features
            y: Training labels (optional, not used for IsolationForest)
            
        Returns:
            self
        """
        logger.info("Fitting Extended Isolation Forest...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit model
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            contamination=self.contamination,
        )
        
        self.model.fit(X_scaled)
        self.is_fitted_ = True
        
        logger.info("Model fitting complete.")
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Predict outliers.
        
        Args:
            X: Features to predict on
            threshold: Decision threshold. If None, uses model's default.
            
        Returns:
            Predictions (-1 for outliers, 1 for inliers)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if threshold is not None:
            # Use custom threshold
            scores = self.decision_function(X_scaled)
            predictions = np.where(scores < threshold, -1, 1)
        else:
            predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly scores.
        
        Args:
            X: Features to score
            
        Returns:
            Anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)
    
    def cross_val_predict(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        normal_classification: int = 0,
        n_splits: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform cross-validated predictions for unsupervised outlier detection.
        
        Special handling for Extended Isolation Forest:
        - Train folds: Only use normal samples (Classification == normal_classification)
        - Validation folds: Use full fold (including abnormalities)
        
        Args:
            X: Features (train set with both normal and abnormal samples)
            y: Labels/classification (for identifying normal samples)
            normal_classification: Value indicating normal samples
            n_splits: Number of CV folds
            
        Returns:
            Tuple of (predictions, scores, fold_scores)
            - predictions: Final predictions from model trained on all normal samples
            - scores: Final anomaly scores
            - fold_scores: Scores from each CV fold (for analysis)
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
        
        # For unsupervised CV: we need custom logic
        # Split ALL samples (normals + abnormalities) into K folds
        # Each fold: train on normal samples from K-1 folds, validate on held-out fold (normals + abnormalities)
        
        from sklearn.model_selection import StratifiedKFold
        
        # Create labels for stratified splitting (0=normal, 1=abnormal)
        y_binary = (y != normal_classification).astype(int)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        fold_scores = []
        fold_predictions = []
        
        for fold_num, (train_fold_idx, val_fold_idx) in enumerate(skf.split(X_scaled, y_binary)):
            logger.info(f"Fold {fold_num + 1}/{n_splits}")
            
            # Get indices for this fold
            X_train_fold = X_scaled[train_fold_idx]
            X_val_fold = X_scaled[val_fold_idx]
            
            # From training fold, extract only normal samples for training
            train_y_binary = y_binary[train_fold_idx]
            train_normal_positions = train_fold_idx[train_y_binary == 0]  # Only normals
            X_train_fold = X_scaled[train_normal_positions]
            
            # Train model on this fold
            fold_model = IsolationForest(
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                n_jobs=self.n_jobs,
                random_state=self.random_state + fold_num,  # Different seed per fold
                contamination=self.contamination,
            )
            fold_model.fit(X_train_fold)
            
            # Get scores for validation fold
            val_scores = fold_model.decision_function(X_val_fold)
            fold_scores.append(val_scores)
            
            # Get predictions for validation fold
            val_preds = fold_model.predict(X_val_fold)
            fold_predictions.append(val_preds)
            
            logger.debug(f"  Fold {fold_num + 1}: {len(X_train_fold)} train, {len(X_val_fold)} val samples")
        
        # After all folds, train final model on ALL normal samples from training set
        X_normal_all = X_scaled[normal_mask]
        
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            contamination=self.contamination,
        )
        self.model.fit(X_normal_all)
        self.is_fitted_ = True
        
        # Get final scores and predictions on full X (training set)
        scores = self.model.decision_function(X_scaled)
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
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'contamination': self.contamination,
            'is_fitted_': self.is_fitted_,
        }
        
        joblib.dump(data, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "ExtendedIsolationForestModel":
        """Load model from file."""
        data = joblib.load(path)
        
        model = cls(
            n_estimators=data['n_estimators'],
            max_samples=data['max_samples'],
            max_features=data['max_features'],
            bootstrap=data['bootstrap'],
            n_jobs=data['n_jobs'],
            random_state=data['random_state'],
            contamination=data['contamination'],
        )
        
        model.model = data['model']
        model.scaler = data['scaler']
        model.is_fitted_ = data['is_fitted_']
        
        logger.info(f"Model loaded from {path}")
        return model
