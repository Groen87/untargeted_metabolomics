"""RALPS batch correction wrapper for metabolomics data."""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_ralps(
    merged_data_path: str,
    merged_batch_path: str,
    output_dir: str,
    mode: str = "NEG",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run RALPS batch correction on merged metabolomics data.
    
    RALPS (Robust Adjustment for Batch Effects using Linear Models and Prior Selection)
    is an alternative to ComBat for batch effect correction.
    
    Args:
        merged_data_path: Path to merged data CSV file (features x samples)
        merged_batch_path: Path to batch metadata CSV file
        output_dir: Directory to save corrected data
        mode: Ion mode (NEG or POS) for naming output files
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (corrected_data, batch_info) DataFrames
        
    Raises:
        ImportError: If RALPS is not installed
        ValueError: If input files are invalid
    """
    try:
        import ralps
    except ImportError:
        raise ImportError(
            "RALPS is not installed. Please install it with: "
            "pip install git+https://github.com/zamboni-lab/RALPS.git"
        )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = pd.read_csv(merged_data_path, index_col=0)  # Features x Samples
    batch_info = pd.read_csv(merged_batch_path)
    
    # Ensure sample names match
    data.columns = data.columns.astype(str)
    batch_info['sample_id'] = batch_info['sample_id'].astype(str)
    
    # Get common samples
    common_samples = list(set(data.columns) & set(batch_info['sample_id']))
    if not common_samples:
        raise ValueError(
            f"No common samples between data and batch info! "
            f"Data: {set(data.columns)}, Batch: {set(batch_info['sample_id'])}"
        )
    
    # Filter to common samples
    data = data[common_samples]
    batch_info = batch_info[batch_info['sample_id'].isin(common_samples)]
    
    # Create batch vector
    batch_vector = pd.Series(
        [batch_info[batch_info['sample_id'] == col]['batch'].iloc[0] 
         for col in data.columns],
        index=data.columns,
        dtype="category"
    )
    
    logger.info(f"Running RALPS batch correction...")
    logger.info(f"Data shape: {data.shape} (features x samples)")
    logger.info(f"Batch vector: {batch_vector.value_counts().to_dict()}")
    
    # RALPS expects data as samples x features (transposed)
    data_t = data.T
    
    # Run RALPS
    # RALPS automatically detects batch from the data
    # We need to pass batch as a categorical variable
    try:
        corrected_data_t = ralps.correct(
            data=data_t,
            batch=batch_vector.values,
            random_state=random_state,
        )
    except Exception as e:
        logger.error(f"RALPS correction failed: {e}")
        raise
    
    # Transpose back to features x samples
    corrected_data = corrected_data_t.T
    
    # Save results
    corrected_data_path = output_dir / f"ralps_corrected_data.csv"
    batch_path = output_dir / f"ralps_batch_metadata.csv"
    
    corrected_data.to_csv(corrected_data_path)
    batch_info.to_csv(batch_path, index=False)
    
    logger.info(f"✓ RALPS correction saved to {corrected_data_path}")
    logger.info(f"✓ Batch metadata saved to {batch_path}")
    
    return corrected_data, batch_info
