"""Configuration management for the combined batch pipeline."""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Load and manage pipeline configuration from YAML file."""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._setup_logging()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}. "
                "Please provide a valid config file path."
            )
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        log_level = getattr(logging, self._config.get("log_level", "INFO").upper())
        log_file = self._config.get("log_file", "logs/combined_batch_pipeline.log")

        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in config."""
        return key in self._config
