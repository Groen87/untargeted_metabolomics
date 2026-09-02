"""
Configuration module for LOF outlier detection pipeline.
"""

import yaml
from typing import Any, List, Optional
from pathlib import Path


class Config:
    """Configuration loader for LOF pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML config file. If None, uses defaults.
        """
        self.config_path = config_path
        self._config = {}
        
        if config_path:
            self._load_config(config_path)
        else:
            self._load_defaults()
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def _load_defaults(self) -> None:
        """Load default configuration."""
        self._config = {
            # Data
            'input_file': 'data/merged_data_with_classification.csv',
            'non_feature_columns': ['Oordeel targeted', 'Classification'],
            'patient_id_column': None,
            'normal_classification': 0,
            'outlier_classifications': [1, 2, 3],
            
            # Splitting
            'train_ratio': 0.8,
            'test_ratio': 0.2,
            'random_seed': 42,
            
            # PCA
            'use_pca': False,
            'pca_method': 'regular',
            'n_components': 100,
            'pca_random_state': 42,
            'save_pca_model': True,
            'pca_nan_strategy': 'drop_columns',
            'pca_batch_size': 1000,
            
            # LOF Model
            'n_neighbors': 20,
            'algorithm': 'auto',
            'leaf_size': 30,
            'metric': 'minkowski',
            'p': 2,
            'contamination': 'auto',
            'novelty': False,
            'n_jobs': -1,
            
            # Cross-validation
            'n_splits': 5,
            
            # Output
            'output_dir': 'outputs/lof_outlier_detection',
            'save_model': True,
            'save_predictions': True,
            'save_plots': True,
            'save_metrics': True,
            
            # Metrics
            'metrics': ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'confusion_matrix'],
            
            # Feature filtering
            'feature_filter': None,
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def get_list(self, key: str, default: List[Any] = None) -> List[Any]:
        """Get configuration value as list."""
        value = self._config.get(key, default)
        if value is None:
            return default or []
        if isinstance(value, list):
            return value
        return [value]
    
    def get_dict(self, key: str, default: dict = None) -> dict:
        """Get configuration value as dict."""
        value = self._config.get(key, default)
        if value is None:
            return default or {}
        return value
