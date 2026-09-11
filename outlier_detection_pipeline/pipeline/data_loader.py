"""
Data loading and preprocessing module for outlier detection pipeline.

Handles:
- Loading merged_data_with_classification.csv
- Identifying feature vs non-feature columns
- Filtering out drug/drug metabolite features from HMDB XML
  - Removes features matching HMDB compounds with 'Drug action pathway' in Process
  - Keeps features matching HMDB compounds with 'Metabolic pathway' in Process
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


def _get_hmdb_cache_path(hmdb_file: str) -> Path:
    """
    Get the cache file path for a given HMDB file.
    
    Args:
        hmdb_file: Path to HMDB XML file
        
    Returns:
        Path to the cache pickle file
    """
    cache_dir = Path.home() / ".cache" / "hmdb_metabolomics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_hash = hashlib.md5(hmdb_file.encode()).hexdigest()[:16]
    return cache_dir / f"hmdb_names_{file_hash}.pkl"


def _load_hmdb_compound_names(hmdb_file: str, use_cache: bool = True) -> Tuple[set, set]:
    """
    Load compound names from HMDB XML file, categorizing by pathway type.
    
    Extracts compound names and accession numbers from HMDB XML.
    Categorizes compounds based on their Process field:
    - Drug metabolites: compounds with 'Drug action pathway' but NO 'Metabolic pathway'
    - Endogenous metabolites: compounds with 'Metabolic pathway'
    
    Uses iterparse to handle large files efficiently.
    
    Args:
        hmdb_file: Path to HMDB XML file
        use_cache: Whether to use cached results if available
        
    Returns:
        Tuple of (endogenous_metabolite_names, drug_metabolite_names)
        Both sets are normalized to uppercase
    """
    try:
        cache_path = _get_hmdb_cache_path(hmdb_file)
        
        # Try to load from cache first
        if use_cache and cache_path.exists():
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            logger.info(f"Loaded HMDB compound names from cache: {cache_path}")
            return cached_data['endogenous'], cached_data['drug']
        
        endogenous_names = set()
        drug_names = set()
        
        context = ET.iterparse(hmdb_file, events=('start', 'end'))
        
        current_compound = {
            'names': set(),
            'accession': None,
            'has_drug_pathway': False,
            'has_metabolic_pathway': False
        }
        in_metabolite = False
        depth = 0
        skip_depth = -1
        
        for event, elem in context:
            tag_lower = elem.tag.lower().split('}')[-1]
            
            if event == 'start':
                # Track when we enter a metabolite element
                if tag_lower == 'metabolite':
                    in_metabolite = True
                    depth = 0
                    current_compound = {
                        'names': set(),
                        'accession': None,
                        'has_drug_pathway': False,
                        'has_metabolic_pathway': False
                    }
                    skip_depth = -1
                
                if in_metabolite and skip_depth == -1:
                    depth += 1
                    
                    # Collect accession (HMDB ID)
                    if tag_lower == 'accession':
                        if elem.text and elem.text.strip():
                            current_compound['accession'] = elem.text.strip()
                    
                    # Collect names
                    elif tag_lower == 'name':
                        if elem.text and elem.text.strip():
                            current_compound['names'].add(elem.text.strip())
                    elif tag_lower == 'synonym':
                        if elem.text and elem.text.strip():
                            current_compound['names'].add(elem.text.strip())
                    
                    # Check for pathway information
                    elif tag_lower == 'process':
                        if elem.text and elem.text.strip():
                            process_text = elem.text.strip().lower()
                            if 'drug action pathway' in process_text:
                                current_compound['has_drug_pathway'] = True
                            if 'metabolic pathway' in process_text:
                                current_compound['has_metabolic_pathway'] = True
            
            elif event == 'end':
                if in_metabolite:
                    tag_lower = elem.tag.lower().split('}')[-1]
                    
                    # When metabolite entry ends, categorize based on pathways
                    if tag_lower == 'metabolite':
                        # Add accession to names
                        if current_compound['accession']:
                            current_compound['names'].add(current_compound['accession'])
                        
                        # Categorize: Drug metabolite if has drug pathway but NO metabolic pathway
                        if current_compound['has_drug_pathway'] and not current_compound['has_metabolic_pathway']:
                            for name in current_compound['names']:
                                if name and len(name) >= 3:
                                    drug_names.add(name.upper())
                        else:
                            # Endogenous or both - keep as endogenous
                            for name in current_compound['names']:
                                if name and len(name) >= 3:
                                    endogenous_names.add(name.upper())
                        
                        current_compound = {
                            'names': set(),
                            'accession': None,
                            'has_drug_pathway': False,
                            'has_metabolic_pathway': False
                        }
                    
                    if in_metabolite:
                        depth -= 1
                
                # Clear processed elements to free memory
                elem.clear()
        
        # Save to cache
        if use_cache:
            cache_data = {
                'endogenous': endogenous_names,
                'drug': drug_names
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Saved HMDB compound names cache to {cache_path}")
        
        logger.info(f"Loaded {len(endogenous_names)} endogenous metabolite names and {len(drug_names)} drug metabolite names from {hmdb_file}")
        
        # Log sample names for debugging
        if len(drug_names) > 0:
            sample_drug = list(drug_names)[:10]
            logger.info(f"Sample drug metabolite names: {sample_drug}{'...' if len(drug_names) > 10 else ''}")
        if len(endogenous_names) > 0:
            sample_endogenous = list(endogenous_names)[:10]
            logger.info(f"Sample endogenous metabolite names: {sample_endogenous}{'...' if len(endogenous_names) > 10 else ''}")
        
        return endogenous_names, drug_names
        
    except Exception as e:
        logger.error(f"Failed to load HMDB file {hmdb_file}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return set(), set()


def _filter_out_drug_features(
    features: pd.DataFrame,
    drug_names: set,
) -> pd.DataFrame:
    """
    Filter out feature columns that match HMDB drug metabolite names.
    
    Removes any features whose column names EXACTLY match HMDB drug metabolite
    names (case-insensitive).
    HMDB features are ALWAYS kept, even if they match drug names,
    to preserve endogenous metabolites that may also appear in the drug list.
    
    Args:
        features: DataFrame with feature columns
        drug_names: Set of HMDB drug metabolite names (uppercase)
        
    Returns:
        Filtered DataFrame with drug features removed
    """
    if not drug_names:
        logger.warning("No drug metabolite names provided. Returning all features.")
        return features
    
    original_cols = set(features.columns)
    
    # Find columns that DO NOT match any drug metabolite name
    # EXCEPT: always keep columns containing 'HMDB' (endogenous metabolites)
    non_drug_columns = []
    removed_cols_with_matches = []
    
    for col in features.columns:
        col_upper = str(col).upper()
        
        # Always keep HMDB features regardless of drug match
        if 'HMDB' in col_upper:
            non_drug_columns.append(col)
            continue
        
        # Check if this column EXACTLY matches any drug metabolite name
        is_drug = False
        matching_name = None
        
        for name in drug_names:
            # Skip very short names that cause false positives (3 chars or less)
            if len(name) <= 3:
                continue
            
            # Exact match only (case-insensitive)
            if col_upper == name:
                is_drug = True
                matching_name = name
                break
        
        if is_drug:
            removed_cols_with_matches.append((col, matching_name))
        else:
            non_drug_columns.append(col)
    
    filtered_features = features[non_drug_columns]
    n_removed = len(original_cols) - len(non_drug_columns)
    
    logger.info(f"Filtered out drug metabolite features: {n_removed} drug features removed, {len(non_drug_columns)} non-drug features retained")
    
    if n_removed > 0:
        removed_cols = list(original_cols - set(non_drug_columns))[:10]
        logger.info(f"Example removed drug features: {removed_cols}{'...' if n_removed > 10 else ''}")
    
    return filtered_features


def load_data(
    input_file: str,
    non_feature_columns: List[str],
    patient_id_column: Optional[str] = None,
    hmdb_file: Optional[str] = None,
    filter_drug_metabolites: bool = False,
    use_hmdb_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data from CSV file and optionally filter out drug metabolite features.
    
    Args:
        input_file: Path to CSV file
        non_feature_columns: List of column names that are NOT features
        patient_id_column: Column name for patient IDs (if not index)
        hmdb_file: Path to HMDB XML file for drug metabolite filtering
        filter_drug_metabolites: Whether to filter out features matching HMDB drug metabolites
        use_hmdb_cache: Whether to use cached HMDB data if available
        
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
    
    # Filter out drug metabolite features if requested
    if filter_drug_metabolites and hmdb_file:
        hmdb_path = Path(hmdb_file)
        if hmdb_path.exists():
            endogenous_names, drug_names = _load_hmdb_compound_names(str(hmdb_path), use_cache=use_hmdb_cache)
            if drug_names:
                features = _filter_out_drug_features(features, drug_names)
            else:
                logger.warning(f"Could not load HMDB drug metabolite names from {hmdb_file}. Using all features.")
        else:
            logger.warning(f"HMDB file not found at {hmdb_file}. Using all features.")
    
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
