"""
Data loading and preprocessing module for outlier detection pipeline.

Handles:
- Loading merged_data_with_classification.csv
- Identifying feature vs non-feature columns
- Filtering out drug/drug metabolite features from ChEMBL SQLite database
- Caching ChEMBL compound names to TXT file for faster subsequent runs
- Splitting data into train/validation/test sets based on Classification
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import shutil
import hashlib

logger = logging.getLogger(__name__)


def _get_chembl_cache_path() -> Path:
    """
    Get the cache file path for ChEMBL compound names.
    
    Returns:
        Path to the cache TXT file
    """
    cache_dir = Path.home() / ".cache" / "chembl_metabolomics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "chembl_names.txt"


def _load_chembl_compound_names_from_file(file_path: str) -> set:
    """
    Load compound names from a local TXT file.
    
    Expected format: one compound name per line.
    Empty lines and lines starting with # are skipped.
    Names are normalized to uppercase.
    
    Args:
        file_path: Path to the TXT file containing compound names
        
    Returns:
        Set of compound names (uppercase)
    """
    try:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"ChEMBL names file not found: {file_path}")
            return set()
        
        compound_names = set()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                # Normalize to uppercase
                name = line.upper()
                if len(name) >= 3:
                    compound_names.add(name)
        
        logger.info(f"Loaded {len(compound_names)} ChEMBL compound names from file: {file_path}")
        return compound_names
        
    except Exception as e:
        logger.error(f"Failed to load ChEMBL names from file {file_path}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return set()


def _load_chembl_compound_names_from_sqlite(sqlite_path: str, use_cache: bool = True) -> set:
    """
    Load compound names directly from ChEMBL SQLite database.
    
    Extracts ChEMBL IDs, preferred names, and synonyms from the database.
    Handles both .db files and .tar.gz archives containing .db files.
    
    Args:
        sqlite_path: Path to SQLite database file or .tar.gz archive
        
    Returns:
        Set of compound names (uppercase)
    """
    try:
        # Try to load from cache first
        cache_path = _get_chembl_cache_path()
        if use_cache and cache_path.exists():
            logger.info(f"Loading ChEMBL names from cache: {cache_path}")
            return _load_chembl_compound_names_from_file(str(cache_path))
        
        path = Path(sqlite_path)
        if not path.exists():
            logger.error(f"ChEMBL SQLite file not found: {sqlite_path}")
            return set()
        
        db_path = None
        temp_dir = None
        
        try:
            # Check if it's a tarball
            if str(path).endswith('.tar.gz') or str(path).endswith('.tgz') or str(path).endswith('.tar'):
                logger.info(f"Extracting database from tarball: {path}")
                temp_dir = tempfile.mkdtemp()
                
                # Use appropriate mode based on extension
                mode = 'r:gz' if str(path).endswith('.tar.gz') or str(path).endswith('.tgz') else 'r'
                with tarfile.open(path, mode) as tar:
                    # Find the database file inside
                    for member in tar.getmembers():
                        if member.name.endswith('.db') or member.name.endswith('.sqlite'):
                            db_path = Path(temp_dir) / member.name
                            db_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(db_path, 'wb') as f:
                                f.write(tar.extractfile(member).read())
                            logger.info(f"  Extracted database: {db_path}")
                            break
                
                if not db_path:
                    logger.error(f"No .db or .sqlite file found in {path}")
                    return set()
            
            # Check if it's already a database file
            elif str(path).endswith('.db') or str(path).endswith('.sqlite'):
                db_path = path
                logger.info(f"Using SQLite database directly: {db_path}")
            
            else:
                logger.error(f"Unsupported file type: {path}. Expected .db, .sqlite, or .tar.gz")
                return set()
            
            # Connect to database and extract names
            logger.info(f"Connecting to SQLite database: {db_path}")
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Get table list
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"Found {len(tables)} tables")
            
            names = set()
            
            # Primary source: molecule_dictionary
            if 'molecule_dictionary' in tables:
                logger.info("Extracting from molecule_dictionary table...")
                cursor.execute("SELECT chembl_id, pref_name, molecule_type FROM molecule_dictionary")
                count = 0
                for row in cursor.fetchall():
                    for val in row:
                        if val and isinstance(val, str):
                            name = val.strip().upper()
                            if len(name) >= 3:
                                names.add(name)
                                count += 1
                logger.info(f"  Found {count} names from molecule_dictionary")
            
            # Secondary source: compound_synonyms
            if 'compound_synonyms' in tables:
                logger.info("Extracting from compound_synonyms table...")
                cursor.execute("SELECT chembl_id, synonyms FROM compound_synonyms")
                syn_count = 0
                for row in cursor.fetchall():
                    # chembl_id
                    if row[0] and isinstance(row[0], str):
                        name = row[0].strip().upper()
                        if len(name) >= 3:
                            names.add(name)
                            syn_count += 1
                    
                    # synonyms
                    if row[1] and isinstance(row[1], str):
                        for syn in row[1].split('|'):
                            syn = syn.strip().upper()
                            if len(syn) >= 3:
                                names.add(syn)
                                syn_count += 1
                logger.info(f"  Added {syn_count} names from synonyms")
            
            # Additional: compound_structures (try different column names)
            if 'compound_structures' in tables:
                logger.info("Extracting from compound_structures table...")
                # Try to find the ID column - different ChEMBL versions use different names
                cursor.execute("PRAGMA table_info(compound_structures);")
                columns = [col[1] for col in cursor.fetchall()]
                id_col = None
                for col in columns:
                    if 'chembl' in col.lower() or 'id' in col.lower():
                        id_col = col
                        break
                
                if id_col:
                    cursor.execute(f"SELECT {id_col} FROM compound_structures")
                    struct_count = 0
                    for row in cursor.fetchall():
                        if row[0] and isinstance(row[0], str):
                            name = row[0].strip().upper()
                            if len(name) >= 3 and name not in names:
                                names.add(name)
                                struct_count += 1
                    logger.info(f"  Added {struct_count} names from compound_structures")
                else:
                    logger.info("  No ID column found in compound_structures, skipping")
            
            logger.info(f"Total unique ChEMBL compound names: {len(names)}")
            
            # Save to cache for faster subsequent runs
            if use_cache:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    for name in sorted(names):
                        f.write(name + '\n')
                logger.info(f"Saved ChEMBL compound names cache to {cache_path}")
            
            return names
            
        finally:
            if 'conn' in locals():
                conn.close()
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        logger.error(f"Failed to load ChEMBL names from SQLite {sqlite_path}: {e}")
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
    filter_chembl: bool = False,
    chembl_sqlite_file: Optional[str] = None,
    use_chembl_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data from CSV file and optionally filter out ChEMBL features.

    Args:
        input_file: Path to CSV file
        non_feature_columns: List of column names that are NOT features
        patient_id_column: Column name for patient IDs (if not index)
        filter_chembl: Whether to filter out features matching ChEMBL compounds
        chembl_sqlite_file: Path to ChEMBL SQLite database file or .tar.gz archive

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
    if filter_chembl and chembl_sqlite_file:
        logger.info(f"Using ChEMBL SQLite database: {chembl_sqlite_file}")
        chembl_names = _load_chembl_compound_names_from_sqlite(chembl_sqlite_file, use_cache=use_chembl_cache)
        if chembl_names:
            features = _filter_out_chembl_features(features, chembl_names)
    elif filter_chembl:
        logger.warning("ChEMBL filtering requested but no chembl_sqlite_file provided. Set chembl_sqlite_file in config.")

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
