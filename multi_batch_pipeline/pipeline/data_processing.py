"""
Data processing module for metabolomics pipeline.

This module handles the loading, transformation, and quality control filtering
of raw metabolomics data from Compound Discoverer output files.

Key Features:
- Automatic delimiter detection for CSV/TSV files
- Sample name cleaning and standardization
- Median-based normalization
- Quality control filtering (QC RSD, intensity thresholds)
- Biological sample filtering
- IMD metabolite preservation
"""

import os
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import csv

from .injection_order import get_injection_order


# =============================================================================
# Constants
# =============================================================================

INTENSITY_THRESHOLD_DEFAULT = 5000
"""Default intensity threshold for filtering low-abundance features."""

QC_RSD_THRESHOLD = 10.0
"""Maximum RSD (Relative Standard Deviation) allowed for QC samples."""

QC_INTENSITY_QUANTILE = 0.30
"""Minimum intensity quantile for features to pass QC filtering."""

BIO_MISSING_THRESHOLD = 0.1
"""Maximum fraction of missing values allowed in biological samples."""

BIO_INTENSITY_QUANTILE = 0.20
"""Minimum intensity quantile for features to pass biological filtering."""

# Known Inborn Metabolic Disorder (IMD) metabolites that should always be preserved
IMD_METABOLITES = [
    'Phenylalanine', 'Tyrosine', 'Leucine', 'Isoleucine', 'Valine', 'Methionine',
    '3-Hydroxyglutaric', 'Methylmalonic', 'Homocysteine', 'Carnitine'
]


# =============================================================================
# Utility Functions
# =============================================================================

def detect_delimiter(file_path: str, sample_size: int = 1024) -> str:
    """
    Detect the delimiter (comma or tab) of a CSV/TSV file.
    
    Uses Python's csv.Sniffer to automatically detect the delimiter.
    Falls back to tab if detection fails and tabs are present, otherwise comma.
    
    Args:
        file_path: Path to the CSV/TSV file
        sample_size: Number of bytes to read for delimiter detection (default: 1024)
        
    Returns:
        Detected delimiter character (',' or '\t')
    """
    with open(file_path, 'r') as f:
        sample = f.read(sample_size)
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            return dialect.delimiter
        except csv.Error:
            return '\t' if '\t' in sample else ','


def clean_sample_name(x: str) -> str:
    """
    Clean and standardize sample names from file paths or column names.
    
    This function:
    - Extracts the base filename from paths
    - Removes file extensions (.raw)
    - Removes parenthetical annotations
    - Preserves _1/_2 suffixes ONLY for expQC samples (case-insensitive)
    - Strips whitespace
    
    Args:
        x: Input string (file path, column name, or sample identifier)
        
    Returns:
        Cleaned sample name string
        
    Example:
        >>> clean_sample_name("path/to/Sample1_1.raw")
        'Sample1_1'
        >>> clean_sample_name("expQC_1.raw")
        'expQC_1'
        >>> clean_sample_name("Sample2 (replicate).raw")
        'Sample2'
    """
    if not isinstance(x, str):
        return str(x).split('.raw')[0].strip()
    
    # Normalize path separators and get filename
    filename = x.replace('\\', '/').split('/')[-1]
    
    # Remove .raw extension and parenthetical annotations
    base_name = filename.split('.raw')[0].split(' (')[0].strip()
    
    # Preserve _1/_2 suffixes for expQC samples
    if 'expqc' in base_name.lower():
        return base_name
    
    # Remove _1/_2 suffixes for non-QC samples
    return base_name.replace('_1', '').replace('_2', '').strip()


def normalize_by_median(df: pd.DataFrame, sample_columns: List[str]) -> pd.DataFrame:
    """
    Normalize each sample by its median intensity, scaled to the median of all sample medians.
    
    This performs median normalization which helps correct for systematic differences
    in sample loading or instrument sensitivity across runs.
    
    Formula for each sample column:
        normalized_value = raw_value * (global_median / sample_median)
    
    Args:
        df: Input DataFrame with features as rows and samples as columns
        sample_columns: List of column names corresponding to sample data
        
    Returns:
        DataFrame with median-normalized values
        
    Note:
        The 'Feature' column (if present) is preserved unchanged.
    """
    # Compute the median intensity for each sample
    sample_medians = df[sample_columns].median(axis=0)
    
    # Compute the median of all sample medians (reference median)
    reference_median = np.median(sample_medians)
    
    # Normalize each sample by its median, scaled to the reference median
    normalized_df = df.copy()
    for col in sample_columns:
        normalized_df[col] = df[col] * (reference_median / sample_medians[col])
    
    return normalized_df


# =============================================================================
# Quality Control Filtering Functions
# =============================================================================

def filter_by_qc_quality(
    df: pd.DataFrame,
    qc_samples: List[str],
    rsd_threshold: float = QC_RSD_THRESHOLD,
    intensity_quantile: float = QC_INTENSITY_QUANTILE,
    missing_threshold: float = 0.1,  # New: Max 10% missing in QC samples
    min_snr: float = 2.0,  # New: Minimum SNR
    imd_features: List[str] = IMD_METABOLITES,  # New: IMD features to exempt
) -> pd.DataFrame:
    """
    Filter features based on QC sample quality metrics.
    """
    if not qc_samples:
        return df

    initial_features = len(df)
    qc_df = df[qc_samples]

    # Identify IMD features (case-insensitive)
    is_imd = df['Feature'].str.contains('|'.join(imd_features), case=False, regex=True)

    # Filter 1: Present in all QC samples
    mask_all_qc = qc_df.notna().all(axis=1)
    mask_all_qc[is_imd] = True  # Exempt IMD features
    df = df[mask_all_qc]
    print(f"✓ QC Filter 1: Removed {initial_features - len(df)} features (not present in all QC samples)")

    # Filter 2: Missing values ≤ threshold in QC samples (NEW)
    missing_rate = qc_df.isna().mean(axis=1)
    mask_low_missing = missing_rate <= missing_threshold
    mask_low_missing[is_imd] = True  # Exempt IMD features
    df = df[mask_low_missing]
    print(f"✓ QC Filter 2: Removed {initial_features - len(df)} features (missing in >{missing_threshold*100}% QC samples)")

    # Filter 3: RSD ≤ threshold in QC samples
    qc_df = df[qc_samples]
    qc_rsd = (qc_df.std(axis=1) / qc_df.mean(axis=1) * 100).fillna(100)
    mask_low_rsd = qc_rsd <= rsd_threshold
    mask_low_rsd[is_imd] = True  # Exempt IMD features
    df = df[mask_low_rsd]
    print(f"✓ QC Filter 3: Removed {initial_features - len(df)} features (RSD > {rsd_threshold}%)")

    # Filter 4: SNR ≥ min_snr (NEW)
    qc_df = df[qc_samples]
    qc_mean = qc_df.mean(axis=1)
    qc_std = qc_df.std(axis=1)
    snr = qc_mean / qc_std
    mask_high_snr = snr >= min_snr
    mask_high_snr[is_imd] = True  # Exempt IMD features
    df = df[mask_high_snr]
    print(f"✓ QC Filter 4: Removed {initial_features - len(df)} features (SNR < {min_snr})")

    # Filter 5: Mean QC intensity ≥ quantile
    qc_mean = qc_df.mean(axis=1)
    intensity_threshold = qc_mean.quantile(intensity_quantile)
    mask_high_intensity = qc_mean >= intensity_threshold
    mask_high_intensity[is_imd] = True  # Exempt IMD features
    df = df[mask_high_intensity]
    print(f"✓ QC Filter 5: Removed {initial_features - len(df)} features (intensity < {intensity_quantile*100}th percentile)")

    # Log IMD features that passed due to exemption
    imd_passed = is_imd[mask_all_qc & mask_low_missing & mask_low_rsd & mask_high_snr & mask_high_intensity]
    if imd_passed.any():
        print(f"⚠️ Preserved {imd_passed.sum()} IMD features that would have been filtered")

    return df

def filter_by_biological_quality(
    df: pd.DataFrame,
    bio_samples: List[str],
    missing_threshold: float = BIO_MISSING_THRESHOLD,
    intensity_quantile: float = BIO_INTENSITY_QUANTILE,
) -> pd.DataFrame:
    """
    Filter features based on biological sample quality metrics.
    
    Applies two sequential filters:
    1. Features must have ≤ threshold fraction of missing values across biological samples
    2. Features must have mean intensity ≥ specified quantile across biological samples
    
    Args:
        df: Input DataFrame with features as rows and samples as columns
        bio_samples: List of column names corresponding to biological samples
        missing_threshold: Maximum allowed fraction of missing values (default: 0.2)
        intensity_quantile: Minimum intensity quantile (default: 0.10)
        
    Returns:
        DataFrame with only features passing all biological quality filters
    """
    if not bio_samples:
        return df
    
    # Filter 1: Missing values ≤ threshold
    missing_rate = df[bio_samples].isna().mean(axis=1)
    low_missing_features = missing_rate[missing_rate <= missing_threshold].index
    df = df.loc[low_missing_features]
    
    # Filter 2: Mean intensity ≥ quantile
    bio_mean = df[bio_samples].mean(axis=1)
    intensity_threshold = bio_mean.quantile(intensity_quantile)
    high_intensity_features = bio_mean[bio_mean >= intensity_threshold].index
    df = df.loc[high_intensity_features]
    
    return df


def readd_imd_features(df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-add known IMD (Inborn Metabolic Disorder) features that were filtered out.
    
    Some IMD metabolites may have poor quality metrics (high RSD, low intensity)
    but are clinically important and should be preserved in the analysis.
    
    Args:
        df: Filtered DataFrame (may have IMD features removed)
        original_df: Original DataFrame before filtering
        
    Returns:
        DataFrame with IMD features re-added from the original data
    """
    # Find the feature column in original_df (case-insensitive)
    feature_col = next((col for col in original_df.columns if 'feature' in col.lower()), None)
    if feature_col is None:
        return df
    
    # Check if any IMD features are present in the filtered data
    imd_feature_mask = df['Feature'].str.contains('|'.join(IMD_METABOLITES), case=False, regex=True)
    if not imd_feature_mask.any():
        return df
    
    # Get the indices of IMD features from the original data
    imd_features = original_df.loc[original_df[feature_col].isin(df[imd_feature_mask]['Feature'])].index
    
    # Re-add IMD features
    df = pd.concat([df, original_df.loc[imd_features]])
    df = df[~df.index.duplicated(keep='first')]
    
    return df


# =============================================================================
# Main Data Processing Function
# =============================================================================

def process_metabolomics_data(
    input_file: str,
    metadata_file: str,
    output_dir: str = "output",
    intensity_threshold: int = INTENSITY_THRESHOLD_DEFAULT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process raw metabolomics data from Compound Discoverer output.
    
    This is the main data processing function that performs:
    1. Loading and initial filtering (removes rows without Formula)
    2. Sample mapping and injection order extraction
    3. Vectorized data transformation and median normalization
    4. Quality control filtering (QC and biological samples)
    5. IMD feature preservation
    6. Global intensity filtering
    7. Batch metadata creation
    
    Args:
        input_file: Path to CSV file with peak area data from Compound Discoverer
        metadata_file: Path to Excel file with sample metadata (injection order)
        output_dir: Directory to save output files (default: "output")
        intensity_threshold: Threshold to filter low-intensity features (default: 10000)
        
    Returns:
        Tuple of (transformed_data, batch_metadata) DataFrames where:
        - transformed_data: Processed feature data with samples as columns
        - batch_metadata: DataFrame with sample metadata (sample_id, batch, group, etc.)
        
    Raises:
        FileNotFoundError: If input_file or metadata_file doesn't exist
        ValueError: If required columns are missing from the input data
    """
    start_time = datetime.now()
    print(f"\n[{start_time}] Starting metabolomics data processing")
    print(f"Input file: {input_file}")
    print(f"Metadata file: {metadata_file}\n")
    
    # --- STEP 0: GET INJECTION ORDER ---
    print("[STEP 0] Getting injection order from metadata...")
    injection_order = get_injection_order(metadata_file)
    
    # --- STEP 1: LOAD AND FILTER ---
    print("[STEP 1] Loading and filtering data...")
    delimiter = detect_delimiter(input_file)
    df = pd.read_csv(input_file, sep=delimiter, low_memory=False)
    
    if 'Formula' in df.columns:
        initial_rows = len(df)
        df = df.dropna(subset=['Formula'])
        print(f"✓ Removed {initial_rows - len(df)} rows without Formula\n")
    else:
        print("⚠️ Warning: No 'Formula' column found. Skipping formula-based filtering.\n")
    
    # --- STEP 2: PRECOMPUTE MAPPINGS ---
    print("[STEP 2] Precomputing sample mappings...")
    area_cols = [col for col in df.columns if 'Area:' in col]
    
    col_to_base = {}
    base_to_cols = {}
    for col in area_cols:
        sample_part = col.split('Area: ')[1].split('.raw')[0].split(' (')[0].strip()
        base_id = clean_sample_name(sample_part)
        col_to_base[col] = base_id
        base_to_cols.setdefault(base_id, []).append(col)
    
    # Clean injection order
    cleaned_injection_order = [clean_sample_name(s) for s in injection_order]
    ordered_base_ids = list(dict.fromkeys(cleaned_injection_order))
    ordered_base_ids = [base_id for base_id in ordered_base_ids if base_id in base_to_cols]
    
    missing_in_metadata = set(base_to_cols.keys()) - set(ordered_base_ids)
    if missing_in_metadata:
        print(f"⚠️ Warning: {len(missing_in_metadata)} samples in data not found in metadata: {sorted(missing_in_metadata)}")
    
    bio_samples = [base_id for base_id in ordered_base_ids
                  if not any(kw in base_id.lower() for kw in ['qc', 'expqc', 'blank', 'mb'])]
    qc_samples = [base_id for base_id in ordered_base_ids
                  if any(kw in base_id.lower() for kw in ['qc', 'expqc', 'blank', 'mb'])]
    print(f"✓ Mapped {len(area_cols)} area columns to {len(base_to_cols)} base samples\n")
    
    # --- STEP 3: VECTORIZED DATA TRANSFORMATION AND MEDIAN NORMALIZATION ---
    print("[STEP 3] Transforming data and applying median normalization (vectorized)...")
    df['base_feature'] = df.apply(
        lambda row: f"{row['Name']}_{row['Formula']}_{row['RT [min]']}"
        if pd.notna(row['Name']) else f"{row['Formula']}_{row['RT [min]']}",
        axis=1
    )
    df['feature'] = df['base_feature'] + ' ' + (df.groupby('base_feature').cumcount() + 1).astype(str)
    
    # Initialize output DataFrame
    transformed_df = pd.DataFrame(index=df.index, columns=['Feature'] + ordered_base_ids)
    transformed_df['Feature'] = df['feature']
    
    # Apply median normalization to each sample column before merging duplicates
    for base_id, cols in base_to_cols.items():
        if base_id not in ordered_base_ids:
            continue
        sample_data = df[cols].copy().replace('', np.nan).astype(float)

        sample_data = sample_data.mask(sample_data <= intensity_threshold, np.nan)

        if len(cols) > 1:
            mask_all_valid = sample_data.notna().all(axis=1)
            merged = sample_data.mean(axis=1)
            merged[~mask_all_valid] = np.nan
            transformed_df[base_id] = merged
        else:
            transformed_df[base_id] = sample_data.iloc[:, 0]
    
    print(f"✓ Transformed {len(transformed_df)} features\n")
    
    # --- STEP 4: QUALITY CONTROL FILTERING ---
    print("[STEP 4] Applying quality control filters...")
    
    # Identify QC samples
    qc_samples = [base_id for base_id in ordered_base_ids
                  if any(kw in base_id.lower() for kw in ['qc', 'expqc', 'blank', 'mb'])]
    
    # Filter by QC quality
    initial_qc_features = len(transformed_df)
    transformed_df = filter_by_qc_quality(transformed_df, qc_samples)
    qc_filtered = initial_qc_features - len(transformed_df)
    print(f"✓ Removed {qc_filtered} features by QC quality filters")
    
    # Filter by biological quality
    initial_bio_features = len(transformed_df)
    transformed_df = filter_by_qc_quality(
        transformed_df,
        qc_samples,
        rsd_threshold=QC_RSD_THRESHOLD,
        intensity_quantile=QC_INTENSITY_QUANTILE,
        missing_threshold=0.1,  # New
        min_snr=2.0,  # New
        imd_features=IMD_METABOLITES,  # New
    )
    bio_filtered = initial_bio_features - len(transformed_df)
    print(f"✓ Removed {bio_filtered} features by biological quality filters")
    
    # Re-add IMD features
    initial_imd_features = len(transformed_df)
    transformed_df = readd_imd_features(transformed_df, df)
    imd_added = len(transformed_df) - initial_imd_features
    if imd_added > 0:
        print(f"✓ Re-added {imd_added} known IMD features")
    
    print(f"✓ Final feature count: {len(transformed_df)} (started with {len(df)})\n")
    
    # --- STEP 5: GLOBAL FILTERING ---
    print("[STEP 5] Applying global intensity filter...")
    if bio_samples:
        features_before = len(transformed_df)
        # Remove features where >50% of biological samples are ≤ threshold OR missing
        below_or_nan = (transformed_df[bio_samples] <= intensity_threshold) | transformed_df[bio_samples].isna()
        missing_rate = below_or_nan.mean(axis=1)
        keep_features = missing_rate <= 0.5  # Max 50% of biological samples ≤ threshold or missing
        transformed_df = transformed_df.loc[keep_features]
        features_removed = features_before - len(transformed_df)
        print(f"✓ Removed {features_removed} features (>50% of biological samples ≤ threshold or missing)\n")
    else:
        print("⚠️ Warning: No biological samples found for global filtering\n")
    
    # --- NEW STEP: FILTER COLUMNS TO KEEP ONLY 'Feature' AND 'posneg*' ---
    print("[NEW STEP] Filtering columns to keep only 'Feature' and 'posneg*'...")
    cols_to_keep = ['Feature'] + [col for col in transformed_df.columns if col.startswith('posneg')]
    if not cols_to_keep:
        raise ValueError("No columns starting with 'posneg' found. Check your sample names.")
    transformed_df = transformed_df[cols_to_keep]
    print(f"✓ Kept {len(cols_to_keep)} columns: {cols_to_keep}\n")
    
    # --- STEP 6: CREATE BATCH DATA ---
    print("[STEP 6] Creating batch metadata...")
    batchdata_df = pd.DataFrame([
        {
            'sample_id': base_id,
            'batch': 1,
            'group': 1 if ('qc3' in base_id.lower() and 'expqc_3' not in base_id.lower()) else
                     2 if ('qc4' in base_id.lower() and 'expqc_4' not in base_id.lower()) else 0,
            'benchmark': 1 if 'blauw' in base_id.lower() else 0
        }
        for base_id in [col for col in cols_to_keep if col != 'Feature']  # Only use filtered columns
    ])
    print(f"✓ Created batch data for {len(batchdata_df)} samples\n")
    
    # --- STEP 7: SAVE OUTPUT ---
    print("[STEP 7] Saving outputs...")
    os.makedirs(output_dir, exist_ok=True)
    transformed_df.to_csv(f'{output_dir}/transformed_data.csv', index=False)
    batchdata_df.to_csv(f'{output_dir}/batch_data.csv', index=False)
    print(f"✓ Saved to {output_dir}\n")
    print(f"[SUMMARY] Done in {(datetime.now() - start_time).total_seconds():.1f} seconds")
    
    return transformed_df, batchdata_df
