"""
Configuration management for outlier detection pipeline.

Loads parameters from YAML config file and provides defaults.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List


class Config:
    """Configuration class for outlier detection pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file. If None, uses default.
        """
        self._config = self._load_config(config_path)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            # Try default path
            default_path = Path(__file__).parent / "config.yaml"
            if default_path.exists():
                config_path = str(default_path)
            else:
                return {}

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config if config else {}
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}, using defaults")
            return {}
        except yaml.YAMLError as e:
            print(f"Warning: Error parsing config file: {e}, using defaults")
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Supports nested keys using dot notation (e.g., 'model.n_estimators').

        Args:
            key: Configuration key (supports dot notation for nested values)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_list(self, key: str, default: Optional[List] = None) -> List:
        """
        Get a configuration value as a list.

        If the value is not already a list, wraps it in a list.

        Args:
            key: Configuration key
            default: Default list if key not found

        Returns:
            List of values
        """
        value = self.get(key, default)
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value in memory.

        Supports nested keys using dot notation (e.g., 'model.n_estimators').
        Used by the n_components sweep to override one parameter per iteration
        without reloading the YAML.

        Args:
            key: Configuration key (supports dot notation for nested values)
            value: Value to set
        """
        keys = key.split('.')
        d = self._config
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return the full configuration as a dictionary."""
        return self._config
