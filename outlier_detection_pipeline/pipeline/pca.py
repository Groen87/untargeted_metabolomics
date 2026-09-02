"""
Sparse PCA module for dimensionality reduction.

Provides multiple PCA methods with different speed/sparsity trade-offs.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union
from sklearn.decomposition import SparsePCA, PCA, MiniBatchSparsePCA
import logging
import joblib
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class SparsePCAWrapper:
    """
    Wrapper for PCA dimensionality reduction with multiple method options.
    
    Supports:
    - regular: Standard PCA (fastest, no sparsity)
    - sparse: SparsePCA (slow, most sparse)
    - mini_batch_sparse: MiniBatchSparsePCA (faster, approximate sparse)
    - two_phase: Regular PCA first, then SparsePCA (balanced)
    """
    
    def __init__(
        self,
        n_components: int = 100,
        alpha: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        method: str = "sparse",
        batch_size: Optional[int] = None,
        intermediate_components: Optional[int] = None,
    ):
        """
        Initialize PCA.
        
        Args:
            n_components: Number of principal components to keep
            alpha: Sparsity controlling parameter (higher = more sparse) - for sparse methods
            max_iter: Maximum iterations - for sparse methods
            random_state: Random seed for reproducibility
            method: PCA method to use ('regular', 'sparse', 'mini_batch_sparse', 'two_phase')
            batch_size: Batch size for mini_batch_sparse (default: 1000)
            intermediate_components: Number of components for first phase of two_phase method
        """
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.method = method
        self.batch_size = batch_size or 1000
        self.intermediate_components = intermediate_components or min(500, n_components * 5)
        
        self.pca = None
        self.pca_phase1 = None  # For two-phase method
        self.is_fitted = False
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.method_used = None
    
    def _estimate_compute_time(self, n_samples: int, n_features: int) -> float:
        """Estimate compute time in seconds based on method and data size."""
        # Rough estimates based on empirical testing
        base_time = n_samples * n_features / 1e6  # Base time in seconds
        
        method_multipliers = {
            'regular': 0.1,
            'sparse': 10.0,
            'mini_batch_sparse': 1.0,
            'two_phase': 2.0,
        }
        
        return base_time * method_multipliers.get(self.method, 1.0) * self.n_components / 100
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit PCA and transform data.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Transformed features (n_samples, n_components)
        """
        n_samples, n_features = X.shape
        estimated_time = self._estimate_compute_time(n_samples, n_features)
        
        logger.info(f"Fitting {self.method} PCA with {self.n_components} components...")
        logger.info(f"Data shape: {n_samples} samples x {n_features} features")
        logger.info(f"Estimated compute time: {estimated_time:.1f} seconds")
        
        start_time = time.time()
        
        if self.method == "regular":
            X_transformed, explained_variance = self._fit_regular_pca(X)
        elif self.method == "sparse":
            X_transformed, explained_variance = self._fit_sparse_pca(X)
        elif self.method == "mini_batch_sparse":
            X_transformed, explained_variance = self._fit_mini_batch_sparse_pca(X)
        elif self.method == "two_phase":
            X_transformed, explained_variance = self._fit_two_phase_pca(X)
        else:
            raise ValueError(f"Unknown PCA method: {self.method}. Use 'regular', 'sparse', 'mini_batch_sparse', or 'two_phase'.")
        
        self.is_fitted = True
        self.explained_variance_ratio_ = explained_variance
        
        # Convert to DataFrame
        column_names = [f"PC_{i+1}" for i in range(self.n_components)]
        result = pd.DataFrame(X_transformed, index=X.index, columns=column_names)
        
        elapsed = time.time() - start_time
        logger.info(f"PCA complete in {elapsed:.1f} seconds.")
        logger.info(f"Explained variance ratio: {explained_variance.sum():.4f}")
        logger.info(f"Original features: {n_features}, Reduced to: {self.n_components}")
        
        if elapsed > 60:
            logger.warning(f"PCA took {elapsed:.1f} seconds. Consider using a faster method:")
            logger.warning(f"  - 'regular': ~{estimated_time*0.1:.1f}s (no sparsity)")
            logger.warning(f"  - 'mini_batch_sparse': ~{estimated_time*0.1:.1f}s (approximate sparse)")
            logger.warning(f"  - 'two_phase': ~{estimated_time*0.2:.1f}s (balanced)")
        
        return result
    
    def _fit_regular_pca(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit regular PCA."""
        self.pca = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
        )
        X_transformed = self.pca.fit_transform(X)
        self.components_ = self.pca.components_
        return X_transformed, self.pca.explained_variance_ratio_
    
    def _fit_sparse_pca(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit SparsePCA."""
        self.pca = SparsePCA(
            n_components=self.n_components,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        X_transformed = self.pca.fit_transform(X)
        self.components_ = self.pca.components_
        # SparsePCA doesn't have explained_variance_ratio_, compute it
        explained_var = np.var(X_transformed, axis=0)
        explained_var_ratio = explained_var / explained_var.sum()
        return X_transformed, explained_var_ratio
    
    def _fit_mini_batch_sparse_pca(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit MiniBatchSparsePCA."""
        self.pca = MiniBatchSparsePCA(
            n_components=self.n_components,
            alpha=self.alpha,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )
        X_transformed = self.pca.fit_transform(X)
        self.components_ = self.pca.components_
        # Compute explained variance ratio
        explained_var = np.var(X_transformed, axis=0)
        explained_var_ratio = explained_var / explained_var.sum()
        return X_transformed, explained_var_ratio
    
    def _fit_two_phase_pca(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Two-phase PCA: Regular PCA first, then SparsePCA."""
        logger.info(f"  Phase 1: Regular PCA to {self.intermediate_components} components")
        
        # Phase 1: Regular PCA to reduce dimensionality
        self.pca_phase1 = PCA(
            n_components=self.intermediate_components,
            random_state=self.random_state,
        )
        X_intermediate = self.pca_phase1.fit_transform(X)
        
        # Phase 2: Sparse PCA on intermediate results
        logger.info(f"  Phase 2: Sparse PCA to {self.n_components} components")
        self.pca = SparsePCA(
            n_components=self.n_components,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        X_transformed = self.pca.fit_transform(X_intermediate)
        self.components_ = self.pca.components_
        
        # Compute explained variance ratio
        explained_var = np.var(X_transformed, axis=0)
        explained_var_ratio = explained_var / explained_var.sum()
        return X_transformed, explained_var_ratio
    

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "SparsePCAWrapper":
        """
        Fit PCA without transforming data.
        
        Args:
            X: Input features (n_samples, n_features) as DataFrame or numpy array
            
        Returns:
            self
        """
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X_index = X.index
            X_values = X.values
        else:
            X_index = None
            X_values = X
        
        n_samples, n_features = X_values.shape
        estimated_time = self._estimate_compute_time(n_samples, n_features)
        
        logger.info(f"Fitting {self.method} PCA with {self.n_components} components...")
        logger.info(f"Data shape: {n_samples} samples x {n_features} features")
        logger.info(f"Estimated compute time: {estimated_time:.1f} seconds")
        
        start_time = time.time()
        
        if self.method == "regular":
            X_transformed, explained_variance = self._fit_regular_pca(X_values)
        elif self.method == "sparse":
            X_transformed, explained_variance = self._fit_sparse_pca(X_values)
        elif self.method == "mini_batch_sparse":
            X_transformed, explained_variance = self._fit_mini_batch_sparse_pca(X_values)
        elif self.method == "two_phase":
            X_transformed, explained_variance = self._fit_two_phase_pca(X_values)
        else:
            raise ValueError(f"Unknown PCA method: {self.method}. Use 'regular', 'sparse', 'mini_batch_sparse', or 'two_phase'.")
        
        self.is_fitted = True
        self.explained_variance_ratio_ = explained_variance
        
        elapsed = time.time() - start_time
        logger.info(f"PCA fitting complete in {elapsed:.1f} seconds.")
        
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Transform new data using fitted PCA.
        
        Args:
            X: Input features (n_samples, n_features) as DataFrame or numpy array
            
        Returns:
            Transformed features (n_samples, n_components)
        """
        if not self.is_fitted:
            raise RuntimeError("PCA not fitted. Call fit() or fit_transform() first.")
        
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X_index = X.index
            X_values = X.values
        else:
            X_index = None
            X_values = X
        
        if self.method == "two_phase":
            # Apply both phases
            X_intermediate = self.pca_phase1.transform(X_values)
            X_transformed = self.pca.transform(X_intermediate)
        else:
            X_transformed = self.pca.transform(X_values)
        
        column_names = [f"PC_{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(X_transformed, index=X_index, columns=column_names)
    
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
