"""Extract injection order from metadata file."""

import pandas as pd
from typing import List

def get_injection_order(metadata_file: str) -> List[str]:
    """
    Extract chronological injection order from a metadata file.

    Args:
        metadata_file: Path to Excel file with 'File Name' and 'Creation Date' columns

    Returns:
        List of sample names in chronological order

    Raises:
        ValueError: If required columns are missing
        FileNotFoundError: If metadata file cannot be read
    """
    meta = pd.read_excel(metadata_file)

    # Validate required columns
    if 'File Name' not in meta.columns:
        raise ValueError("Metadata file must contain 'File Name' column")
    if 'Creation Date' not in meta.columns:
        raise ValueError("Metadata file must contain 'Creation Date' column")

    # Convert Creation Date to datetime
    meta['Creation Date'] = pd.to_datetime(
        meta['Creation Date'],
        format='%d-%m-%Y %H:%M:%S'  # Matches "17-10-2025 14:25:53"
    )

    # Extract and clean sample names
    def clean_sample_name(x: str) -> str:
        """
        Clean sample name from a string (path or column name).
        Preserves _1/_2 ONLY for expQC samples (case-insensitive).
        Merges all other samples (including QC3, QC4, etc.).
        """
        if not isinstance(x, str):
            return str(x).split('.raw')[0].strip()

        # Extract filename from path (Windows or Unix)
        filename = x.replace('\\', '/').split('/')[-1]

        # Remove .raw extension and (Fxx) suffix
        base_name = filename.split('.raw')[0].split(' (')[0].strip()

        # ONLY for expQC samples: preserve _1/_2 (case-insensitive)
        if 'expqc' in base_name.lower():
            return base_name
        # For ALL other samples (including QC3, QC4, etc.): remove _1/_2
        else:
            return base_name.replace('_1', '').replace('_2', '').strip()

    meta['Sample'] = meta['File Name'].apply(clean_sample_name)

    # Deduplicate and sort by Creation Date
    injection_order = meta.sort_values('Creation Date')['Sample'].tolist()
    injection_order = list(dict.fromkeys(injection_order))  # Preserve order, remove duplicates

    print(f"✓ Extracted {len(injection_order)} samples in chronological order")
    return injection_order