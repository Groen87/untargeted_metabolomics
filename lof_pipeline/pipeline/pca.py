"""
PCA module for LOF pipeline.
Reuses the SparsePCAWrapper from the outlier_detection_pipeline.
"""

import sys
import os

# Add the repo root to path
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import SparsePCAWrapper from the outlier_detection_pipeline
try:
    from outlier_detection_pipeline.pipeline.pca import SparsePCAWrapper
except ImportError:
    # Fallback: copy the class definition here
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
        """Wrapper for PCA dimensionality reduction with multiple method options."""
        
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
            self.n_components = n_components
            self.alpha = alpha
            self.max_iter = max_iter
            self.random_state = random_state
            self.method = method
            self.batch_size = batch_size or 1000
            self.intermediate_components = intermediate_components or min(500, n_components * 5)
            self.pca = None
            self.pca_phase1 = None
            self.is_fitted = False
            self.components_ = None
            self.explained_variance_ratio_ = None
            self.method_used = None
        
        def _estimate_compute_time(self, n_samples: int, n_features: int) -> float:
            base_time = n_samples * n_features / 1e6
            method_multipliers = {'regular': 0.1, 'sparse': 10.0, 'mini_batch_sparse': 1.0, 'two_phase': 2.0}
            return base_time * method_multipliers.get(self.method, 1.0) * self.n_components / 100
        
        def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
            n_samples, n_features = X.shape
            estimated_time = self._estimate_compute_time(n_samples, n_features)
            logger.info(f"Fitting {self.method} PCA with {self.n_components} components...")
            logger.info(f"Data shape: {n_samples} samples x {n_features} features")
            logger.info(f"Estimated compute time: {estimated_time:.1f} seconds")
            start_time = time.time()
            
            if self.method == "regular":
                X_transformed, explained_variance = self._fit_regular_pca(X.values)
            elif self.method == "sparse":
                X_transformed, explained_variance = self._fit_sparse_pca(X.values)
            elif self.method == "mini_batch_sparse":
                X_transformed, explained_variance = self._fit_mini_batch_sparse_pca(X.values)
            elif self.method == "two_phase":
                X_transformed, explained_variance = self._fit_two_phase_pca(X.values)
            else:
                raise ValueError(f"Unknown PCA method: {self.method}")
            
            self.is_fitted = True
            self.explained_variance_ratio_ = explained_variance
            column_names = [f"PC_{i+1}" for i in range(self.n_components)]
            result = pd.DataFrame(X_transformed, index=X.index, columns=column_names)
            elapsed = time.time() - start_time
            logger.info(f"PCA complete in {elapsed:.1f} seconds.")
            return result
        
        def _fit_regular_pca(self, X_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
            X_transformed = self.pca.fit_transform(X_array)
            self.components_ = self.pca.components_
            return X_transformed, self.pca.explained_variance_ratio_
        
        def _fit_sparse_pca(self, X_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            self.pca = SparsePCA(n_components=self.n_components, alpha=self.alpha,
                               max_iter=self.max_iter, random_state=self.random_state)
            X_transformed = self.pca.fit_transform(X_array)
            self.components_ = self.pca.components_
            explained_var = np.var(X_transformed, axis=0)
            explained_var_ratio = explained_var / explained_var.sum()
            return X_transformed, explained_var_ratio
        
        def _fit_mini_batch_sparse_pca(self, X_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            self.pca = MiniBatchSparsePCA(n_components=self.n_components, alpha=self.alpha,
                                          max_iter=self.max_iter, batch_size=self.batch_size,
                                          random_state=self.random_state)
            X_transformed = self.pca.fit_transform(X_array)
            self.components_ = self.pca.components_
            explained_var = np.var(X_transformed, axis=0)
            explained_var_ratio = explained_var / explained_var.sum()
            return X_transformed, explained_var_ratio
        
        def _fit_two_phase_pca(self, X_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            self.pca_phase1 = PCA(n_components=self.intermediate_components, random_state=self.random_state)
            X_intermediate = self.pca_phase1.fit_transform(X_array)
            self.pca = SparsePCA(n_components=self.n_components, alpha=self.alpha,
                               max_iter=self.max_iter, random_state=self.random_state)
            X_transformed = self.pca.fit_transform(X_intermediate)
            self.components_ = self.pca.components_
            explained_var = np.var(X_transformed, axis=0)
            explained_var_ratio = explained_var / explained_var.sum()
            return X_transformed, explained_var_ratio
        
        def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "SparsePCAWrapper":
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
            n_samples, n_features = X_values.shape
            estimated_time = self._estimate_compute_time(n_samples, n_features)
            logger.info(f"Fitting {self.method} PCA with {self.n_components} components...")
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
                raise ValueError(f"Unknown PCA method: {self.method}")
            
            self.is_fitted = True
            self.explained_variance_ratio_ = explained_variance
            elapsed = time.time() - start_time
            logger.info(f"PCA fitting complete in {elapsed:.1f} seconds.")
            return self
        
        def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
            if not self.is_fitted:
                raise RuntimeError("PCA not fitted. Call fit() or fit_transform() first.")
            if isinstance(X, pd.DataFrame):
                X_index = X.index
                X_values = X.values
            else:
                X_index = None
                X_values = X
            
            if self.method == "two_phase":
                X_intermediate = self.pca_phase1.transform(X_values)
                X_transformed = self.pca.transform(X_intermediate)
            else:
                X_transformed = self.pca.transform(X_values)
            
            column_names = [f"PC_{i+1}" for i in range(self.n_components)]
            return pd.DataFrame(X_transformed, index=X_index, columns=column_names)
        
        def save(self, path: Path) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, path)
            logger.info(f"PCA model saved to {path}")
        
        @staticmethod
        def load(path: Path) -> "SparsePCAWrapper":
            return joblib.load(path)
        
        def get_explained_variance(self) -> np.ndarray:
            if self.explained_variance_ratio_ is None:
                return np.array([])
            return self.explained_variance_ratio_

__all__ = ['SparsePCAWrapper']
