"""
Data loading module for combined batch pipeline.

This module handles loading and parsing the combined CSV file where:
- All batches are in a single file
- Columns are named: "Area: {filename} ({F#})"
- Batch name is embedded in the filename (e.g., posneg_MZ25_36_...)
- Duplicates have _1.raw and _2.raw suffixes
- Injection order comes from metadata file creation dates
"""

import re
from typing import Tuple, Dict, List, Optional
import pandas as pd
import numpy as np
import logging

from .injection_order import (
    clean_sample_name,
    get_injection_order_from_metadata,
    get_injection_order_mapping,
    get_sample_info_from_metadata,
)

logger = logging.getLogger(__name__)


def extract_batch_from_filename(filename: str) -> Optional[str]:
    """
    Extract batch name from a filename.
    
    Examples:
        posneg_MZ25_36_25230101131_1.raw -> MZ25_36
        posneg_MZ26_10_26050123531_1.raw -> MZ26_10
        C:\\Untargeted project Joost\\Data Exploris 120\\MZ25_36\\posneg_MZ25_36_25230101131_1.raw -> MZ25_36
    
    Args:
        filename: The filename to parse
        
    Returns:
        The batch name (e.g., "MZ25_36") or None if not found
    """
    # Handle Windows paths
    clean_name = filename.replace('\\', '/').split('/')[-1]
    
    # Pattern: posneg_{BATCH}_{REST} or Posneg_{BATCH}_{REST}
    # Batch name is typically MZ##_## (e.g., MZ25_36)
    
    # Try posneg_ prefix first
    posneg_match = re.search(r'posneg_([A-Z]+\d+)_(\d+)', clean_name)
    if posneg_match:
        return f"{posneg_match.group(1)}_{posneg_match.group(2)}"
    
    # Try Posneg_ prefix
    posneg_match = re.search(r'Posneg_([A-Z]+\d+)_(\d+)', clean_name)
    if posneg_match:
        return f"{posneg_match.group(1)}_{posneg_match.group(2)}"
    
    # Try without posneg prefix: MZ##_##_...
    batch_match = re.search(r'([A-Z]+\d+)_(\d+)', clean_name)
    if batch_match:
        return f"{batch_match.group(1)}_{batch_match.group(2)}"
    
    # Fallback: try to find any MZ##_## pattern
    fallback_match = re.search(r'(MZ\d+_\d+)', clean_name)
    if fallback_match:
        return fallback_match.group(1)
    
    return None


def extract_sample_id_from_filename(filename: str) -> str:
    """
    Extract a clean sample ID from a filename.
    
    Removes:
    - "Area: " prefix
    - " (F#)" suffix
    - ".raw" extension
    - _1, _2 suffixes (for duplicates, except expQC)
    
    Args:
        filename: The filename to parse
        
    Returns:
        Clean sample ID
    """
    # Use the same cleaning logic as the original pipeline
    return clean_sample_name(filename)


def extract_sample_type(sample_id: str) -> str:
    """
    Extract sample type from sample ID.
    
    Args:
        sample_id: Clean sample ID
        
    Returns:
        Sample type: "QC", "blanco", "blauw", "Mix", or "Sample"
    """
    sample_id_lower = sample_id.lower()
    
    if any(kw in sample_id_lower for kw in ['expqc']):
        return "QC"
    elif 'blanco' in sample_id_lower:
        return "blanco"
    elif 'blauw' in sample_id_lower:
        return "blauw"
    elif 'mix' in sample_id_lower:
        return "Mix"
    else:
        return "Sample"


def load_combined_data(
    input_file: str,
    intensity_threshold: float = 10000.0,
    metadata_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, Dict], Optional[Dict[str, int]]]:
    """
    Load combined batch data from a single CSV file.
    
    This function:
    1. Reads the CSV file
    2. Identifies sample columns (those starting with "Area:")
    3. Extracts batch names from filenames
    4. Groups samples by batch
    5. Filters out low-intensity features
    6. Optionally loads injection order from metadata file
    
    Args:
        input_file: Path to the combined CSV file
        intensity_threshold: Minimum intensity threshold for filtering (default: 10000)
        metadata_file: Optional path to metadata CSV file for injection order
        
    Returns:
        Tuple of:
        - df: DataFrame with features as rows, samples as columns
        - batch_groups: Dictionary mapping batch names to lists of sample column names
        - sample_info: Dictionary with metadata for each sample column
        - injection_order: Dictionary mapping column names to injection order (if metadata provided)
    """
    logger.info(f"Loading data from {input_file}")
    
    # Read CSV
    df = pd.read_csv(input_file)
    
    # Identify columns
    name_col = 'Name'
    area_cols = [col for col in df.columns if col.startswith('Area:')]
    
    if not area_cols:
        raise ValueError(f"No Area: columns found in {input_file}")
    
    logger.info(f"Found {len(area_cols)} sample columns")
    
    # Set Name as index
    df = df.set_index(name_col)
    
    # Convert area columns to numeric
    for col in area_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter out low-intensity features
    initial_features = len(df)
    intensity_mask = (df[area_cols] > intensity_threshold).any(axis=1)
    df = df[intensity_mask]
    filtered_features = initial_features - len(df)
    logger.info(f"Filtered out {filtered_features} features with all intensities <= {intensity_threshold}")
    
    # Load injection order from metadata if provided
    injection_order: Optional[Dict[str, int]] = None
    metadata_sample_info: Dict[str, Dict] = {}
    
    if metadata_file:
        logger.info(f"Loading injection order from {metadata_file}")
        try:
            # Get injection order mapping from metadata
            inj_order = get_injection_order_from_metadata(metadata_file)
            inj_mapping = {sample: idx for idx, sample in enumerate(inj_order)}
            
            # Get sample info from metadata
            metadata_sample_info = get_sample_info_from_metadata(metadata_file)
            
            # Build injection order for columns
            injection_order = {}
            for col in area_cols:
                sample_id = extract_sample_id_from_filename(col)
                if sample_id in inj_mapping:
                    injection_order[col] = inj_mapping[sample_id]
                else:
                    logger.warning(f"Could not find injection order for: {sample_id}")
                    injection_order[col] = -1
            
            logger.info(f"Loaded injection order for {len(injection_order)} columns")
        except Exception as e:
            logger.warning(f"Could not load injection order from metadata: {e}")
    
    # Group samples by batch
    batch_groups: Dict[str, List[str]] = {}
    sample_info: Dict[str, Dict] = {}
    
    for col in area_cols:
        # Extract batch and sample info
        filename = col.split('Area: ')[1].split(' (')[0].strip()
        batch = extract_batch_from_filename(filename)
        sample_id = extract_sample_id_from_filename(col)
        
        # Use sample type from metadata if available, otherwise infer
        if metadata_file and sample_id in metadata_sample_info:
            sample_type = metadata_sample_info[sample_id]['sample_type']
        else:
            sample_type = extract_sample_type(sample_id)
        
        if batch is None:
            logger.warning(f"Could not extract batch from column: {col}")
            batch = "UNKNOWN"
        
        if batch not in batch_groups:
            batch_groups[batch] = []
        
        batch_groups[batch].append(col)
        
        sample_info[col] = {
            'batch': batch,
            'sample_id': sample_id,
            'sample_type': sample_type,
            'original_col': col,
            'injection_order': injection_order.get(col, -1) if injection_order else -1,
        }
    
    logger.info(f"Identified {len(batch_groups)} batches: {sorted(batch_groups.keys())}")
    for batch, cols in sorted(batch_groups.items()):
        logger.info(f"  {batch}: {len(cols)} samples")
    
    return df, batch_groups, sample_info, injection_order


def average_duplicates(
    df: pd.DataFrame,
    sample_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Average duplicate samples (e.g., _1 and _2).
    
    Args:
        df: DataFrame with sample data
        sample_cols: List of sample column names
        
    Returns:
        Tuple of:
        - df_averaged: DataFrame with duplicates averaged
        - col_mapping: Dictionary mapping new column names to original columns
    """
    # Group columns by base name (without _1/_2 suffix)
    col_groups: Dict[str, List[str]] = {}
    
    for col in sample_cols:
        # Extract base name without _1/_2
        base = col.split('Area: ')[1].split('.raw')[0].split(' (')[0].strip()
        
        # Remove _1 or _2 suffix (except for expQC)
        if 'expqc' in base.lower():
            # Keep _1/_2 for expQC
            pass
        elif base.endswith('_1'):
            base = base[:-2]
        elif base.endswith('_2'):
            base = base[:-2]
        
        if base not in col_groups:
            col_groups[base] = []
        col_groups[base].append(col)
    
    # Create averaged DataFrame
    df_averaged = pd.DataFrame(index=df.index)
    col_mapping: Dict[str, str] = {}
    
    for base, cols in col_groups.items():
        if len(cols) == 1:
            # No duplicates, just copy
            new_col = f"Area: {base}.raw"
            df_averaged[new_col] = df[cols[0]]
            col_mapping[new_col] = cols[0]
        else:
            # Average duplicates
            new_col = f"Area: {base}.raw"
            df_averaged[new_col] = df[cols].mean(axis=1)
            col_mapping[new_col] = f"average of {cols}"
    
    return df_averaged, col_mapping


# =============================================================================
# Batch Identification Functions
# =============================================================================

def get_all_batches(df: pd.DataFrame) -> List[str]:
    """
    Get list of all unique batches from DataFrame columns.
    
    Args:
        df: DataFrame with sample columns
        
    Returns:
        Sorted list of unique batch names
    """
    area_cols = [col for col in df.columns if col.startswith('Area:')]
    batches = set()
    
    for col in area_cols:
        batch = extract_batch_from_filename(col)
        if batch:
            batches.add(batch)
    
    return sorted(batches)


def get_batch_samples(
    df: pd.DataFrame,
    batch: str,
) -> List[str]:
    """
    Get all sample columns belonging to a specific batch.
    
    Args:
        df: DataFrame with sample columns
        batch: Batch name to filter by
        
    Returns:
        List of column names belonging to the batch
    """
    area_cols = [col for col in df.columns if col.startswith('Area:')]
    batch_samples = []
    
    for col in area_cols:
        col_batch = extract_batch_from_filename(col)
        if col_batch == batch:
            batch_samples.append(col)
    
    return batch_samples
