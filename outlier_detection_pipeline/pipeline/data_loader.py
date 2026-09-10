"""
Data loading and preprocessing module for outlier detection pipeline.

Handles:
- Loading merged_data_with_classification.csv
- Identifying feature vs non-feature columns
- Filtering out drug/drug metabolite features from ChEMBL SDF
- Splitting data into train/validation/test sets based on Classification
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
import logging
import gzip
from pathlib import Path
import pickle
import hashlib
import re

logger = logging.getLogger(__name__)


def _get_chembl_cache_path(chembl_file: str) -> Path:
    """
    Get the cache file path for a given ChEMBL file.
    
    Args:
        chembl_file: Path to ChEMBL SDF file
        
    Returns:
        Path to the cache pickle file
    """
    cache_dir = Path.home() / ".cache" / "chembl_metabolomics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_hash = hashlib.md5(chembl_file.encode()).hexdigest()[:16]
    return cache_dir / f"chembl_names_{file_hash}.pkl"


def _load_chembl_compound_names(chembl_file: str, use_cache: bool = True) -> set:
    """
    Load compound names and IDs from ChEMBL SDF file.
    
    Extracts:
    - Compound names from header lines
    - ChEMBL IDs (e.g., CHEMBL123456) from property fields
    - Synonyms if available
    
    Handles both .sdf and .sdf.gz files.
    
    Args:
        chembl_file: Path to ChEMBL SDF file (can be .sdf or .sdf.gz)
        use_cache: Whether to use cached results if available
        
    Returns:
        Set of compound names and IDs (normalized to uppercase)
    """
    try:
        cache_path = _get_chembl_cache_path(chembl_file)
        
        # Try to load from cache first
        if use_cache and cache_path.exists():
            with open(cache_path, 'rb') as f:
                compound_names = pickle.load(f)
            logger.info(f"Loaded {len(compound_names)} ChEMBL compound names from cache: {cache_path}")
            return compound_names
        
        compound_names = set()
        
        # Determine if file is gzipped
        open_func = gzip.open if chembl_file.endswith('.gz') else open
        mode = 'rt' if chembl_file.endswith('.gz') else 'r'
        encoding = 'utf-8' if not chembl_file.endswith('.gz') else None
        
        # First, scan to find all available property tags in this SDF file
        # This helps us understand the structure
        all_prop_tags = set()
        sample_header_lines = []
        
        with open_func(chembl_file, mode, encoding=encoding) as f:
            header_count = 0
            for line in f:
                line = line.strip()
                if line.startswith('>') and line != '>':
                    tag = line[1:].strip()
                    all_prop_tags.add(tag.upper())
                elif line == '$$$$':
                    break  # Just scan first record for tags
                else:
                    # This might be a header line
                    if header_count < 5:
                        sample_header_lines.append(line)
                        header_count += 1
        
        logger.info(f"Found property tags in ChEMBL SDF: {sorted(list(all_prop_tags))}")
        logger.info(f"Sample header/first lines: {sample_header_lines}")
        
        # Now parse the file properly
        with open_func(chembl_file, mode, encoding=encoding) as f:
            current_name = None
            current_chembl_id = None
            
            for line in f:
                line = line.strip()
                
                # End of molecule record
                if line == '$$$$':
                    current_name = None
                    current_chembl_id = None
                    continue
                
                # New molecule record starts with a header line (compound name)
                # This is the FIRST line of a record
                if line and not line.startswith('>') and not line.startswith('$'):
                    # Check if this looks like an atom line (starts with number or coordinate pattern)
                    if line and (line[0].isdigit() or re.match(r'^[\d\s.-]+$', line[:20])):
                        # This is an atom/bond line, skip
                        continue
                    
                    # This is the molecule header line (compound name)
                    current_name = line.strip()
                    current_chembl_id = None
                    
                    # Add the compound name if reasonable length and looks like a real name
                    # (not just an ID like CHEMBL123456)
                    if current_name and len(current_name) >= 3:
                        # Only add if it contains at least one letter (not just numbers/dashes)
                        if re.search(r'[a-zA-Z]', current_name):
                            compound_names.add(current_name)
                    
                    continue
                
                # Property section starts with >
                if line.startswith('>'):
                    prop_name = line[1:].strip()
                    prop_name_upper = prop_name.upper()
                    
                    # Known name fields in ChEMBL
                    name_fields = {
                        '<PREF_NAME>', 'PREF_NAME',
                        '<PREFERRED_NAME>', 'PREFERRED_NAME', 
                        '<GENERIC_NAME>', 'GENERIC_NAME',
                        '<MOLECULE_TYPE>', 'MOLECULE_TYPE',
                        '<COMPOUND_NAME>', 'COMPOUND_NAME',
                        '<NAME>', 'NAME',
                        '<TITLE>', 'TITLE',
                        '<COMMON_NAME>', 'COMMON_NAME',
                        '<TRADITIONAL_NAME>', 'TRADITIONAL_NAME',
                        '<INCHI_KEY>', 'INCHI_KEY',
                        '<SMILES>', 'SMILES',
                    }
                    
                    # Also check for any tag that contains NAME
                    if any(name_keyword in prop_name_upper for name_keyword in ['NAME', 'TITLE', 'PREF', 'GENERIC', 'COMMON', 'TRADITIONAL']):
                        # Next line contains the name
                        try:
                            name_line = next(f, '').strip()
                            if name_line and len(name_line) >= 3:
                                # Skip if it's a CHEMBL ID
                                if not re.match(r'^CHEMBL\d+$', name_line):
                                    compound_names.add(name_line)
                        except StopIteration:
                            pass
                    
                    elif prop_name_upper == '<CHEMBL_ID>' or prop_name_upper == 'CHEMBL_ID':
                        # Next line contains the ChEMBL ID - we still want this for matching
                        try:
                            chembl_id_line = next(f, '').strip()
                            if chembl_id_line and len(chembl_id_line) >= 3:
                                current_chembl_id = chembl_id_line
                                # Add CHEMBL ID to match against features that use CHEMBL IDs
                                compound_names.add(chembl_id_line)
                        except StopIteration:
                            pass
                    
                    elif prop_name_upper == 'SYNONYMS':
                        # Read synonyms - next line contains them
                        try:
                            synonyms_line = next(f, '').strip()
                            if synonyms_line:
                                # Synonyms might be comma or semicolon separated
                                synonyms = re.split(r'[;,]', synonyms_line)
                                for syn in synonyms:
                                    syn = syn.strip()
                                    if syn and len(syn) >= 3 and re.search(r'[a-zA-Z]', syn):
                                        compound_names.add(syn)
                        except StopIteration:
                            pass
                    
                    elif prop_name_upper == '<CHEMBL_COMPOUND>':
                        # Some SDF files have this
                        pass
                    
                    # Try ALL property values that look like names
                    elif prop_name_upper not in ['<CHEMBL_ID>', 'CHEMBL_ID', 'SYNONYMS', '<CHEMBL_COMPOUND>', 'M  END']:
                        # Read the value line
                        try:
                            value_line = next(f, '').strip()
                            if value_line and len(value_line) >= 3 and len(value_line) < 100:
                                # Check if it looks like a chemical name (has letters and reasonable length)
                                if re.search(r'[a-zA-Z]', value_line) and not re.match(r'^CHEMBL\d+$', value_line):
                                    compound_names.add(value_line)
                        except StopIteration:
                            pass
        
        # Normalize all names to uppercase for case-insensitive matching
        compound_names = {name.upper() for name in compound_names if name and len(name) >= 3}
        
        # Save to cache for future runs
        if use_cache:
            # Remove old cache if it exists
            if cache_path.exists():
                cache_path.unlink()
            with open(cache_path, 'wb') as f:
                pickle.dump(compound_names, f)
            logger.info(f"Saved ChEMBL compound names cache to {cache_path}")
        
        logger.info(f"Loaded {len(compound_names)} ChEMBL compound names from {chembl_file}")
        if len(compound_names) == 0:
            logger.warning(f"No ChEMBL compound names found in {chembl_file}. Check SDF structure.")
        
        # Log some sample names for debugging
        if len(compound_names) > 0:
            sample_names = list(compound_names)[:10]
            logger.info(f"Sample ChEMBL names: {sample_names}{'...' if len(compound_names) > 10 else ''}")
        
        return compound_names
        
    except Exception as e:
        logger.error(f"Failed to load ChEMBL file {chembl_file}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return set()


def _filter_out_chembl_features(
    features: pd.DataFrame,
    chembl_names: set,
) -> pd.DataFrame:
    """
    Filter out feature columns that match ChEMBL compound names.
    
    Removes any features whose column names EXACTLY match (case-insensitive)
    ChEMBL compound names or IDs.
    
    HMDB features are ALWAYS kept, even if they match ChEMBL names,
    to preserve endogenous metabolites that may also appear in ChEMBL.
    
    Args:
        features: DataFrame with feature columns
        chembl_names: Set of ChEMBL compound names (uppercase)
        
    Returns:
        Filtered DataFrame with ChEMBL features removed
    """
    if not chembl_names:
        logger.warning("No ChEMBL names provided. Returning all features.")
        return features
    
    original_cols = set(features.columns)
    
    # Find columns that DO NOT exactly match any ChEMBL name
    # EXCEPT: always keep columns containing 'HMDB' (endogenous metabolites)
    non_chembl_columns = []
    removed_cols_with_matches = []
    
    for col in features.columns:
        col_upper = str(col).upper()
        
        # Always keep HMDB features regardless of ChEMBL match
        if 'HMDB' in col_upper:
            non_chembl_columns.append(col)
            continue
        
        # Check if this column EXACTLY matches any ChEMBL name
        is_chembl = False
        matching_name = None
        
        for name in chembl_names:
            # Skip very short names that cause false positives (3 chars or less)
            if len(name) <= 3:
                continue
            
            # Exact match only (case-insensitive)
            if col_upper == name:
                is_chembl = True
                matching_name = name
                break
        
        if is_chembl:
            removed_cols_with_matches.append((col, matching_name))
        else:
            non_chembl_columns.append(col)
    
    filtered_features = features[non_chembl_columns]
    
    n_removed = len(original_cols) - len(non_chembl_columns)
    logger.info(f"Filtered out ChEMBL features: {n_removed} ChEMBL features removed, {len(non_chembl_columns)} non-ChEMBL features retained")
    
    if n_removed > 0:
        removed_cols = list(original_cols - set(non_chembl_columns))[:10]
        logger.info(f"Example removed ChEMBL features: {removed_cols}{'...' if n_removed > 10 else ''}")
        
        # Log matching details for debugging
        if n_removed <= 50:
            for col, match in removed_cols_with_matches[:10]:
                logger.info(f"  Removed '{col}' -> matched ChEMBL name: '{match}'")
        else:
            # Sample and show
            import random
            sample = random.sample(removed_cols_with_matches, min(10, len(removed_cols_with_matches)))
            for col, match in sample:
                logger.info(f"  Removed '{col}' -> matched ChEMBL name: '{match}'")
    
    return filtered_features


def load_data(
    input_file: str,
    non_feature_columns: List[str],
    patient_id_column: Optional[str] = None,
    chembl_file: Optional[str] = None,
    filter_chembl: bool = False,
    use_chembl_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data from CSV file and optionally filter out ChEMBL features.

    Args:
        input_file: Path to CSV file
        non_feature_columns: List of column names that are NOT features
        patient_id_column: Column name for patient IDs (if not index)
        chembl_file: Path to ChEMBL SDF file for ChEMBL feature filtering
        filter_chembl: Whether to filter out features matching ChEMBL compounds
        use_chembl_cache: Whether to use cached ChEMBL data if available

    Returns:
        Tuple of:
        - features: DataFrame of features (rows = samples, columns = features)
        - classification: Series with Classification values
        - oordeel: Series with Oordeel targeted values
    """
    logger.info(f"Loading data from {input_file}")

    # Load CSV
    df = pd.read_csv(input_file, index_col=0 if patient_id_column is None else None)

    if patient_id_column is not None:
        df = df.set_index(patient_id_column)

    logger.info(f"Loaded data with shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Data cleaning: Filter samples based on Classification and Oordeel targeted
    classification_col = df['Classification']
    oordeel_col = df['Oordeel targeted']

    # Remove ambiguous: Classification 2 or 3 with Oordeel targeted = 0
    ambiguous_mask = ((classification_col.isin([2, 3])) & (oordeel_col == 0))
    n_ambiguous = ambiguous_mask.sum()

    if n_ambiguous > 0:
        ambiguous_indices = df.index[ambiguous_mask]
        logger.warning(f"Found {n_ambiguous} ambiguous samples (Classification 2/3 with Oordeel targeted=0). Removing these.")
        df = df[~ambiguous_mask]
        logger.warning(f"Removed samples: {list(ambiguous_indices[:5])}{'...' if n_ambiguous > 5 else ''}")

    # After removing ambiguous, update classification to be consistent
    inconsistent_mask = (df['Classification'] == 0) & (df['Oordeel targeted'] != 0)
    n_inconsistent = inconsistent_mask.sum()

    if n_inconsistent > 0:
        inconsistent_indices = df.index[inconsistent_mask]
        logger.warning(f"Found {n_inconsistent} samples with Classification=0 but Oordeel targeted!=0. "
                      f"Updating Classification to 1 (outlier) for consistency.")
        df.loc[inconsistent_mask, 'Classification'] = 1
        logger.warning(f"Updated samples: {list(inconsistent_indices[:5])}{'...' if n_inconsistent > 5 else ''}")

    # Extract non-feature columns
    classification = df['Classification']
    oordeel = df['Oordeel targeted']

    # Get feature columns (all columns except non-feature columns)
    feature_cols = [col for col in df.columns if col not in non_feature_columns]
    features = df[feature_cols]

    # Filter out ChEMBL features if requested
    if filter_chembl and chembl_file:
        chembl_path = Path(chembl_file)
        if chembl_path.exists():
            chembl_names = _load_chembl_compound_names(str(chembl_path), use_cache=use_chembl_cache)
            if chembl_names:
                features = _filter_out_chembl_features(features, chembl_names)
            else:
                logger.warning(f"Could not load ChEMBL names from {chembl_file}. Using all features.")
        else:
            logger.warning(f"ChEMBL file not found at {chembl_file}. Using all features.")

    logger.info(f"Feature columns: {len(features.columns)}")
    logger.info(f"Non-feature columns: {non_feature_columns}")

    return features, classification, oordeel


def split_data(
    features: pd.DataFrame,
    classification: pd.Series,
    normal_classification: int,
    outlier_classifications: List[int],
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Split data into train and test sets using stratified split.

    For Extended Isolation Forest (unsupervised):
    - Stratified train-test split (80-20) to maintain class distribution
    - Train set contains both normal and abnormal samples
    - Test set contains both normal and abnormal samples
    - During CV: train only on normal samples from training folds
    - Validate on full validation folds (including abnormalities)

    Args:
        features: DataFrame of features
        classification: Series with Classification values
        normal_classification: Classification value for normal samples
        outlier_classifications: List of outlier classification values
        train_ratio: Ratio for training set (default: 0.8)
        test_ratio: Ratio for test set (default: 0.2)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with keys: 'train', 'test'
        Each value is a tuple of (features, classification)
    """
    # Check for NaN in classification and drop if present
    df_combined = pd.concat([features, classification.rename('Classification')], axis=1)
    df_combined = df_combined.dropna(subset=['Classification'])

    if classification.isna().any():
        n_dropped = classification.isna().sum()
        logger.warning(f"Found {n_dropped} NaN values in Classification. Dropping these samples.")

    features = df_combined[features.columns]
    classification = df_combined['Classification']

    # Stratified train-test split (maintains class distribution)
    X_for_split = pd.DataFrame(index=features.index)
    X_for_split['classification'] = classification.values

    train_df, test_df = train_test_split(
        X_for_split,
        train_size=train_ratio,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=classification,
    )

    train_indices = train_df.index
    test_indices = test_df.index

    logger.info(f"Train set: {len(train_indices)} samples")
    logger.info(f"Test set: {len(test_indices)} samples")
    logger.info(f"Train class distribution: {classification[train_indices].value_counts().to_dict()}")
    logger.info(f"Test class distribution: {classification[test_indices].value_counts().to_dict()}")

    # Create splits
    splits = {}
    for name, indices in [('train', train_indices), ('test', test_indices)]:
        splits[name] = (
            features.loc[indices].copy(),
            classification.loc[indices].copy(),
        )

    return splits


def get_class_distribution(classification: pd.Series) -> Dict[int, int]:
    """Get distribution of classification values."""
    return classification.value_counts().to_dict()
