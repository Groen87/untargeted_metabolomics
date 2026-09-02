"""
Sparse PCA module for dimensionality reduction.

Provides sparse PCA functionality for feature reduction before outlier detection.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from sklearn.decomposition import SparsePCA, PCA
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class SparsePCAWrapper:
    """
    Wrapper for Sparse PCA dimensionality reduction.
    
    Falls back to regular PCA if SparsePCA is not available or if
    the data doesn't support sparse PCA well.
    """
    
    def __init__(
        self,
        n_components: int = 100,
        alpha: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        """
        Initialize Sparse PCA.
        
        Args:
            n_components: Number of principal components to keep
            alpha: Sparsity controlling parameter (higher = more sparse)
            max_iter: Maximum iterations
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.pca = None
        self.is_fitted = False
        self.components_ = None
        self.explained_variance_ratio_ = None
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit PCA and transform data.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Transformed features (n_samples, n_components)
        """
        logger.info(f"Fitting Sparse PCA with {self.n_components} components...")
        
        # Initialize PCA
        self.pca = SparsePCA(
            n_components=self.n_components,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        
        # Fit and transform
        X_transformed = self.pca.fit_transform(X)
        
        # Get components and explained variance
        self.components_ = self.pca.components_
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.is_fitted = True
        
        # Convert to DataFrame
        column_names = [f"PC_{i+1}" for i in range(self.n_components)]
        result = pd.DataFrame(X_transformed, index=X.index, columns=column_names)
        
        logger.info(f"PCA complete. Explained variance ratio: {self.explained_variance_ratio_.sum():.4f}")
        logger.info(f"Original features: {X.shape[1]}, Reduced to: {self.n_components}")
        
        return result
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted PCA.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Transformed features (n_samples, n_components)
        """
        if not self.is_fitted:
            raise RuntimeError("PCA not fitted. Call fit_transform() first.")
        
        X_transformed = self.pca.transform(X)
        column_names = [f"PC_{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(X_transformed, index=X.index, columns=column_names)
    
    def save(self, path: Path) -> None:
        """Save PCA model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"PCA model saved to {path}")
    
    @staticmethod
    def load(path: Path) -> "SparsePCAWrapper":
        """Load PCA model from file."""
        return joblib.load(path)
    
    def get_explained_variance(self) -> np.ndarray:
        """Get explained variance ratio for each component."""
        if self.explained_variance_ratio_ is None:
            return np.array([])
        return self.explained_variance_ratio_
