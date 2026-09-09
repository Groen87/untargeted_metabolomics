"""
Data loading and preprocessing module for outlier detection pipeline.

Handles:
- Loading merged_data_with_classification.csv
- Identifying feature vs non-feature columns
- Filtering out drug/drug metabolite features from DrugBank XML
- Splitting data into train/validation/test sets based on Classification
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
import pickle
import hashlib

logger = logging.getLogger(__name__)


def _get_drugbank_cache_path(drugbank_file: str) -> Path:
    """
    Get the cache file path for a given DrugBank file.
    
    Args:
        drugbank_file: Path to DrugBank XML file
        
    Returns:
        Path to the cache pickle file
    """
    cache_dir = Path.home() / ".cache" / "drugbank_metabolomics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_hash = hashlib.md5(drugbank_file.encode()).hexdigest()[:16]
    return cache_dir / f"drugbank_names_{file_hash}.pkl"


def _load_drugbank_compound_names(drugbank_file: str, use_cache: bool = True) -> set:
    """
    Load all compound names and synonyms from DrugBank XML file.
    
    Extracts drug names, synonyms, and metabolite names to create a comprehensive
    set that can be matched against feature column names for filtering.
    
    Uses iterparse to handle large files efficiently.
    
    Args:
        drugbank_file: Path to DrugBank XML file
        use_cache: Whether to use cached results if available
        
    Returns:
        Set of all drug and drug metabolite names (normalized to uppercase)
    """
    try:
        cache_path = _get_drugbank_cache_path(drugbank_file)
        
        # Try to load from cache first
        if use_cache and cache_path.exists():
            with open(cache_path, 'rb') as f:
                compound_names = pickle.load(f)
            logger.info(f"Loaded {len(compound_names)} DrugBank compound names from cache: {cache_path}")
            return compound_names
        
        compound_names = set()
        context = ET.iterparse(drugbank_file, events=('start', 'end'))
        
        current_names = set()
        in_synonyms = False
        in_calculated_properties = False
        
        for event, elem in context:
            if event == 'start':
                tag_lower = elem.tag.lower().split('}')[-1]
                
                # Check for drug/drugbank entry
                if 'drug' in tag_lower or tag_lower == 'drugbank':
                    current_names = set()
                
                # Check for name element
                elif tag_lower == 'name':
                    if elem.text and elem.text.strip():
                        current_names.add(elem.text.strip())
                
                # Check for generic_name
                elif tag_lower == 'generic_name':
                    if elem.text and elem.text.strip():
                        current_names.add(elem.text.strip())
                
                # Check for synonyms container
                elif tag_lower in ('synonyms', 'synonym'):
                    in_synonyms = True
                    if elem.text and elem.text.strip():
                        current_names.add(elem.text.strip())
                
                # Check for calculated properties (contains metabolites)
                elif tag_lower == 'calculated_properties':
                    in_calculated_properties = True
                
                # Check for metabolites
                elif tag_lower == 'metabolites' or tag_lower == 'metabolite':
                    if elem.text and elem.text.strip():
                        current_names.add(elem.text.strip())
            
            elif event == 'end':
                tag_lower = elem.tag.lower().split('}')[-1]
                
                # When drug entry ends, add collected names to master set
                if 'drug' in tag_lower and current_names:
                    compound_names.update(current_names)
                    current_names = set()
                
                # End of synonyms
                if tag_lower in ('synonyms', 'synonym'):
                    in_synonyms = False
                
                if tag_lower == 'calculated_properties':
                    in_calculated_properties = False
                
                # Clear processed elements to free memory
                elem.clear()
        
        # Normalize all names to uppercase for case-insensitive matching
        compound_names = {name.upper() for name in compound_names if name}
        
        # Save to cache for future runs
        if use_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(compound_names, f)
            logger.info(f"Saved DrugBank compound names cache to {cache_path}")
        
        logger.info(f"Loaded {len(compound_names)} DrugBank compound names and synonyms from {drugbank_file}")
        if len(compound_names) == 0:
            logger.warning(f"No DrugBank compound names found in {drugbank_file}. Check XML structure.")
        return compound_names
        
    except Exception as e:
        logger.error(f"Failed to load DrugBank file {drugbank_file}: {e}")
        return set()


def _filter_out_drug_features(
    features: pd.DataFrame,
    drugbank_names: set,
) -> pd.DataFrame:
    """
    Filter out feature columns that match DrugBank compound names.
    
    Removes any features whose column names contain DrugBank drug or
    drug metabolite names (case-insensitive substring matching).
    
    Args:
        features: DataFrame with feature columns
        drugbank_names: Set of DrugBank compound names (uppercase)
        
    Returns:
        Filtered DataFrame with drug features removed
    """
    if not drugbank_names:
        logger.warning("No DrugBank names provided. Returning all features.")
        return features
    
    original_cols = set(features.columns)
    
    # Find columns that DO NOT contain any DrugBank name
    non_drug_columns = []
    for col in features.columns:
        col_upper = str(col).upper()
        is_drug = False
        for name in drugbank_names:
            if name in col_upper:
                is_drug = True
                break
        if not is_drug:
            non_drug_columns.append(col)
    
    filtered_features = features[non_drug_columns]
    
    n_removed = len(original_cols) - len(non_drug_columns)
    logger.info(f"Filtered out drug features: {n_removed} drug features removed, {len(non_drug_columns)} non-drug features retained")
    
    if n_removed > 0:
        removed_cols = list(original_cols - set(non_drug_columns))[:10]
        logger.info(f"Example removed drug features: {removed_cols}{'...' if n_removed > 10 else ''}")
    
    return filtered_features


def load_data(
    input_file: str,
    non_feature_columns: List[str],
    patient_id_column: Optional[str] = None,
    drugbank_file: Optional[str] = None,
    filter_drugs: bool = False,
    use_drugbank_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data from CSV file and optionally filter out drug features.

    Args:
        input_file: Path to CSV file
        non_feature_columns: List of column names that are NOT features
        patient_id_column: Column name for patient IDs (if not index)
        drugbank_file: Path to DrugBank XML file for drug feature filtering
        filter_drugs: Whether to filter out features matching DrugBank compounds
        use_drugbank_cache: Whether to use cached DrugBank data if available

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

    # Filter out drug features if requested
    if filter_drugs and drugbank_file:
        drugbank_path = Path(drugbank_file)
        if drugbank_path.exists():
            drugbank_names = _load_drugbank_compound_names(str(drugbank_path), use_cache=use_drugbank_cache)
            if drugbank_names:
                features = _filter_out_drug_features(features, drugbank_names)
            else:
                logger.warning(f"Could not load DrugBank names from {drugbank_file}. Using all features.")
        else:
            logger.warning(f"DrugBank file not found at {drugbank_file}. Using all features.")

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
