"""
Injection order extraction module for metabolomics pipeline.

This module provides functionality to extract the chronological injection order
from metadata files, which is essential for drift correction.

Key Features:
- Reads Excel metadata files
- Extracts sample names and creation dates
- Cleans and standardizes sample names
- Returns samples in chronological injection order
"""

import pandas as pd
from typing import List, Callable, Optional


def clean_sample_name(x: str) -> str:
    """
    Clean and standardize sample names from file paths.
    
    This function:
    - Extracts the base filename from paths (Windows or Unix)
    - Removes .raw extension
    - Removes parenthetical annotations (e.g., "(Fxx)")
    - Preserves _1/_2 suffixes ONLY for expQC samples (case-insensitive)
    - Removes _1/_2 suffixes for all other samples (including QC3, QC4, etc.)
    - Strips whitespace
    
    Args:
        x: Input string (file path, column name, or sample identifier)
        
    Returns:
        Cleaned sample name string
        
    Example:
        >>> clean_sample_name("path/to/Sample1_1.raw")
        'Sample1'
        >>> clean_sample_name("expQC_1.raw")
        'expQC_1'
        >>> clean_sample_name("QC3_1 (F01).raw")
        'QC3'
    """
    if not isinstance(x, str):
        return str(x).split('.raw')[0].strip()
    
    # Extract filename from path (Windows or Unix)
    filename = x.replace('\\', '/').split('/')[-1]
    
    # Remove .raw extension and parenthetical annotations
    base_name = filename.split('.raw')[0].split(' (')[0].strip()
    
    # ONLY for expQC samples: preserve _1/_2 (case-insensitive)
    if 'expqc' in base_name.lower():
        return base_name
    
    # For ALL other samples: remove _1/_2
    return base_name.replace('_1', '').replace('_2', '').strip()


def get_injection_order(
    metadata_file: str,
    file_name_col: str = "File Name",
    date_col: str = "Creation Date",
    date_format: str = "%d-%m-%Y %H:%M:%S",
    sample_name_cleaner: Optional[Callable[[str], str]] = None,
) -> List[str]:
    """
    Extract chronological injection order from a metadata file.
    
    This function reads an Excel metadata file, extracts sample names and
    creation dates, and returns the samples in chronological order based on
    their creation dates.
    
    Args:
        metadata_file: Path to Excel file with sample metadata
        file_name_col: Column name containing file names (default: "File Name")
        date_col: Column name containing creation dates (default: "Creation Date")
        date_format: Format string for parsing dates (default: "%d-%m-%Y %H:%M:%S")
            Matches format like "17-10-2025 14:25:53"
        sample_name_cleaner: Optional function to clean sample names
            (default: uses clean_sample_name from this module)
            
    Returns:
        List of sample names in chronological injection order
        (duplicates removed, order preserved)
        
    Raises:
        FileNotFoundError: If metadata file cannot be read
        ValueError: If required columns are missing from metadata file
    
    Example:
        >>> order = get_injection_order("metadata.xlsx")
        >>> print(order[:5])
        ['QC3', 'Sample1', 'Sample2', 'QC4', 'Sample3']
    """
    # Use provided cleaner or default
    if sample_name_cleaner is None:
        sample_name_cleaner = clean_sample_name
    
    # Read metadata file
    meta = pd.read_excel(metadata_file)
    
    # Validate required columns
    if file_name_col not in meta.columns:
        raise ValueError(f"Metadata file must contain '{file_name_col}' column")
    if date_col not in meta.columns:
        raise ValueError(f"Metadata file must contain '{date_col}' column")
    
    # Convert Creation Date to datetime
    meta[date_col] = pd.to_datetime(meta[date_col], format=date_format)
    
    # Extract and clean sample names
    meta['Sample'] = meta[file_name_col].apply(sample_name_cleaner)
    
    # Deduplicate and sort by Creation Date
    injection_order = meta.sort_values(date_col)['Sample'].tolist()
    injection_order = list(dict.fromkeys(injection_order))  # Preserve order, remove duplicates
    
    print(f"✓ Extracted {len(injection_order)} samples in chronological order")
    return injection_order
