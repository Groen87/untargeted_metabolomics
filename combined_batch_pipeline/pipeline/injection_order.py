"""
Injection order extraction module for combined batch pipeline.

This module handles:
- Loading metadata CSV files (tab-separated, UTF-16 encoding)
- Removing duplicate samples (keeping first occurrence) to match data processing
- Extracting injection order from creation dates
- Mapping cleaned sample names to injection order
"""

import pandas as pd
from typing import List, Callable, Optional, Dict
import re


def clean_sample_name(x: str) -> str:
    """
    Clean and standardize sample names from file paths.
    
    This function:
    - Extracts the base filename from paths (Windows or Unix)
    - Removes .raw extension
    - Removes parenthetical annotations (e.g., "(Fxx)")
    - Removes _1/_2 suffixes for all samples (duplicates will be handled separately)
    - Strips whitespace
    
    Args:
        x: Input string (file path, column name, or sample identifier)
        
    Returns:
        Cleaned sample name string
        
    Example:
        >>> clean_sample_name("path/to/Sample1_1.raw")
        'Sample1'
        >>> clean_sample_name("expQC_1.raw")
        'expQC'
        >>> clean_sample_name("QC3_1 (F01).raw")
        'QC3'
    """
    if not isinstance(x, str):
        return str(x).split('.raw')[0].strip()
    
    # Extract filename from path (Windows or Unix)
    filename = x.replace('\\', '/').split('/')[-1]
    
    # Remove .raw extension and parenthetical annotations
    base_name = filename.split('.raw')[0].split(' (')[0].strip()
    
    # Remove _1/_2 suffixes for all samples
    return base_name.replace('_1', '').replace('_2', '').strip()


def get_injection_order_from_metadata(
    metadata_file: str,
    file_name_col: str = "File Name",
    date_col: str = "Creation Date",
    date_format: str = "%d-%m-%Y %H:%M:%S",
    sample_name_cleaner: Optional[Callable[[str], str]] = None,
) -> List[str]:
    """
    Extract chronological injection order from a metadata file.
    
    This function:
    1. Reads the metadata CSV file
    2. Cleans sample names (removes _1/_2 suffixes)
    3. Deduplicates by keeping first occurrence (earliest injection)
    4. Sorts remaining samples by creation date
    
    Args:
        metadata_file: Path to CSV file with sample metadata
        file_name_col: Column name containing file names (default: "File Name")
        date_col: Column name containing creation dates (default: "Creation Date")
        date_format: Format string for parsing dates (default: "%d-%m-%Y %H:%M:%S")
        sample_name_cleaner: Optional function to clean sample names
            
    Returns:
        List of unique sample names in chronological injection order
        
    Raises:
        FileNotFoundError: If metadata file cannot be read
        ValueError: If required columns are missing from metadata file
    """
    # Use provided cleaner or default
    if sample_name_cleaner is None:
        sample_name_cleaner = clean_sample_name
    
    # Read metadata file - tab-separated, UTF-16 encoding
    meta = pd.read_csv(metadata_file, sep='\t', encoding='utf-16')
    
    # Validate required columns
    if file_name_col not in meta.columns:
        raise ValueError(f"Metadata file must contain '{file_name_col}' column")
    if date_col not in meta.columns:
        raise ValueError(f"Metadata file must contain '{date_col}' column")
    
    # Convert Creation Date to datetime
    meta[date_col] = pd.to_datetime(meta[date_col], format=date_format, errors='coerce')
    
    # Extract and clean sample names
    meta['Sample'] = meta[file_name_col].apply(sample_name_cleaner)
    
    # Sort by Creation Date first
    meta = meta.sort_values(date_col)
    
    # Deduplicate: keep first occurrence (earliest injection time for each sample)
    # This matches what we do with the data (averaging duplicates)
    injection_order = []
    seen_samples = set()
    for _, row in meta.iterrows():
        sample = row['Sample']
        if sample not in seen_samples:
            injection_order.append(sample)
            seen_samples.add(sample)
    
    print(f"\u2713 Extracted {len(injection_order)} unique samples in chronological order")
    return injection_order


def get_injection_order_mapping(
    metadata_file: str,
    file_name_col: str = "File Name",
    date_col: str = "Creation Date",
    date_format: str = "%d-%m-%Y %H:%M:%S",
) -> Dict[str, int]:
    """
    Get a mapping of sample names to their injection order index.
    
    Args:
        metadata_file: Path to CSV metadata file
        file_name_col: Column name containing file names
        date_col: Column name containing creation dates
        date_format: Format string for parsing dates
        
    Returns:
        Dictionary mapping cleaned sample name to injection order index (0-based)
    """
    injection_order = get_injection_order_from_metadata(
        metadata_file, file_name_col, date_col, date_format
    )
    
    return {sample: idx for idx, sample in enumerate(injection_order)}


def get_sample_info_from_metadata(
    metadata_file: str,
    file_name_col: str = "File Name",
    sample_type_col: str = "Sample Type",
    date_col: str = "Creation Date",
    date_format: str = "%d-%m-%Y %H:%M:%S",
) -> Dict[str, Dict]:
    """
    Get complete sample information from metadata file.
    
    This deduplicates by keeping the first occurrence (earliest injection).
    
    Args:
        metadata_file: Path to CSV metadata file
        file_name_col: Column name containing file names
        sample_type_col: Column name containing sample types
        date_col: Column name containing creation dates
        date_format: Format string for parsing dates
        
    Returns:
        Dictionary mapping cleaned sample name to info dict with:
        - sample_type (from first occurrence)
        - creation_date (from first occurrence)
        - original_file_name (from first occurrence)
    """
    # Read metadata
    meta = pd.read_csv(metadata_file, sep='\t', encoding='utf-16')
    
    # Clean sample names
    meta['Cleaned_Sample'] = meta[file_name_col].apply(clean_sample_name)
    
    # Convert date
    meta[date_col] = pd.to_datetime(meta[date_col], format=date_format, errors='coerce')
    
    # Sort by date and deduplicate (keep first)
    meta = meta.sort_values(date_col)
    
    # Build info dictionary - keep first occurrence of each sample
    sample_info = {}
    seen_samples = set()
    for _, row in meta.iterrows():
        cleaned = row['Cleaned_Sample']
        if cleaned not in seen_samples:
            sample_info[cleaned] = {
                'sample_type': str(row[sample_type_col]).strip(),
                'creation_date': row[date_col],
                'original_file_name': str(row[file_name_col]).strip(),
            }
            seen_samples.add(cleaned)
    
    return sample_info
