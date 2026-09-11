"""
Data loading and preprocessing module for outlier detection pipeline.

Handles:
- Loading merged_data_with_classification.csv
- Identifying feature vs non-feature columns
- Filtering features to keep only ChEBI human metabolites (role: CHEBI:77746) from SDF
- Splitting data into train/validation/test sets based on Classification
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)


def _load_chebi_human_metabolite_names(sdf_path: str) -> set:
    """
    Load compound names from ChEBI SDF file, filtering for human metabolites.
    
    Extracts compound names and synonyms for compounds with role 'CHEBI:77746' (human metabolites).
    
    ChEBI SDF format:
    - Compounds separated by $$$$
    - First line of each compound: compound name
    - Property tags: > <TAG>
    - Property values: line immediately following the tag
    - Role CHEBI:77746 may be in any field - we search all property values
    
    Args:
        sdf_path: Path to ChEBI SDF file
        
    Returns:
        Set of compound names and synonyms (uppercase) for human metabolites
    """
    try:
        path = Path(sdf_path)
        if not path.exists():
            logger.error(f"ChEBI SDF file not found: {sdf_path}")
            return set()
        
        names = set()
        current_compound = {'names': set(), 'chebi_id': None, 'all_values': []}
        expect_value = False
        current_tag = None
        all_tags = set()
        n_human_metabolites = 0
        n_total = 0
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.rstrip('\n\r')
                
                # End of compound record
                if line == '$$$$':
                    n_total += 1
                    # Check if this compound has role CHEBI:77746 in ANY field
                    is_human = False
                    
                    # Search all collected values for role indicators
                    for val in current_compound['all_values']:
                        val_upper = str(val).upper()
                        if 'CHEBI:77746' in val_upper or 'HUMAN METABOLITE' in val_upper:
                            is_human = True
                            break
                    
                    # If human metabolite, add all names
                    if is_human:
                        n_human_metabolites += 1
                        # Add ChEBI ID
                        if current_compound.get('chebi_id'):
                            chebi_id = current_compound['chebi_id'].strip().upper()
                            if len(chebi_id) >= 3:
                                names.add(chebi_id)
                        # Add all names
                        for name in current_compound['names']:
                            if len(name) >= 3:
                                names.add(name)
                    
                    # Reset for next compound
                    current_compound = {'names': set(), 'chebi_id': None, 'all_values': []}
                    expect_value = False
                    current_tag = None
                    continue
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Property tag line (e.g., ">  <ChEBI ID>")
                if line.startswith('>') and '<' in line:
                    # Extract tag name
                    tag_match = re.search(r'\>\s*<([^>]+)>', line)
                    if tag_match:
                        current_tag = tag_match.group(1).strip()
                        all_tags.add(current_tag)
                        expect_value = True
                    continue
                
                # Property value line (immediately after tag)
                if expect_value and current_tag:
                    value = line.strip()
                    
                    # Normalize tag name for comparison
                    tag_upper = current_tag.upper()
                    
                    # Store ChEBI ID
                    if any(t in tag_upper for t in ['CHEBIID', 'CHEBIID', 'CHEBIID', 'ID', 'CHEBICOMPOUNDID', 'CHEBICOMPOUND_ID']):
                        current_compound['chebi_id'] = value
                    
                    # Store name/synonym
                    elif any(n in tag_upper for n in ['NAME', 'SYNONYM', 'IUPAC', 'CHEBINAME', 'CHEBIIUPACNAME', 'CHEBIIUPAC']):
                        # Handle multi-line values (split by semicolons, pipes, commas)
                        for v in re.split(r'[;|,]', value):
                            v = v.strip().upper()
                            if len(v) >= 3:
                                current_compound['names'].add(v)
                    
                    # Also add the value itself as a name (for generic tags)
                    if len(value) >= 3:
                        current_compound['names'].add(value.upper())
                    
                    # Store ALL values for role checking
                    current_compound['all_values'].append(value)
                    
                    expect_value = False
                    current_tag = None
                    continue
                
                # If not a tag and not a value, it might be the compound name (first line)
                if current_tag is None and not current_compound['names']:
                    name_val = line.strip().upper()
                    if len(name_val) >= 3:
                        current_compound['names'].add(name_val)
                    current_compound['all_values'].append(line.strip())
        
        # Process the last compound
        n_total += 1
        is_human = False
        for val in current_compound['all_values']:
            val_upper = str(val).upper()
            if 'CHEBI:77746' in val_upper or 'HUMAN METABOLITE' in val_upper:
                is_human = True
                break
        
        if is_human:
            n_human_metabolites += 1
            if current_compound.get('chebi_id'):
                chebi_id = current_compound['chebi_id'].strip().upper()
                if len(chebi_id) >= 3:
                    names.add(chebi_id)
            for name in current_compound['names']:
                if len(name) >= 3:
                    names.add(name)
        
        logger.info(f"Loaded {len(names)} ChEBI human metabolite names from SDF: {sdf_path}")
        logger.info(f"Scanned {n_total} compounds, {n_human_metabolites} human metabolites found")
        logger.info(f"Found SDF tags: {sorted(all_tags)}")
        
        # Log sample names for debugging
        if len(names) > 0:
            sample_names = list(names)[:10]
            logger.info(f"Sample ChEBI human metabolite names: {sample_names}{'...' if len(names) > 10 else ''}")
        else:
            logger.warning("No human metabolite names found. Searched ALL property values for 'CHEBI:77746' or 'HUMAN METABOLITE'. Your SDF file may not contain ontology/role information.")
        
        return names
        
    except Exception as e:
        logger.error(f"Failed to load ChEBI names from SDF {sdf_path}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return set()


def _filter_features_by_chebi_names(
    features: pd.DataFrame,
    chebi_names: set,
) -> pd.DataFrame:
    """
    Filter feature columns to keep ONLY those that match ChEBI human metabolite names.
    
    This is the INVERSE of the ChEMBL filtering - we KEEP matches, not remove them.
    
    Args:
        features: DataFrame with feature columns
        chebi_names: Set of ChEBI human metabolite names (uppercase)
        
    Returns:
        Filtered DataFrame with only ChEBI human metabolite features
    """
    if not chebi_names:
        logger.warning("No ChEBI names provided. Returning all features.")
        return features
    
    # Find columns that match any ChEBI human metabolite name (case-insensitive)
    kept_columns = []
    removed_count = 0
    
    for col in features.columns:
        col_upper = str(col).upper()
        
        # Check for exact match
        matched = False
        for name in chebi_names:
            if len(name) > 3 and col_upper == name:
                matched = True
                break
        
        if matched:
            kept_columns.append(col)
        else:
            removed_count += 1
    
    filtered_features = features[kept_columns]
    
    n_kept = len(kept_columns)
    logger.info(f"Filtered features by ChEBI human metabolites: {removed_count} removed, {n_kept} ChEBI human metabolite features kept")
    
    if n_kept > 0:
        kept_sample = kept_columns[:10]
        logger.info(f"Example kept features: {kept_sample}{'...' if n_kept > 10 else ''}")
    
    return filtered_features


def load_data(
    input_file: str,
    non_feature_columns: List[str],
    patient_id_column: Optional[str] = None,
    filter_chebi_human_metabolites: bool = False,
    chebi_sdf_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data from CSV file and optionally filter features to ChEBI human metabolites.

    Args:
        input_file: Path to CSV file
        non_feature_columns: List of column names that are NOT features
        patient_id_column: Column name for patient IDs (if not index)
        filter_chebi_human_metabolites: Whether to filter features to keep only ChEBI human metabolites
        chebi_sdf_file: Path to ChEBI SDF file with role information

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

    # Filter features to keep only ChEBI human metabolites if requested
    if filter_chebi_human_metabolites and chebi_sdf_file:
        logger.info(f"Filtering features to keep only ChEBI human metabolites from: {chebi_sdf_file}")
        chebi_names = _load_chebi_human_metabolite_names(chebi_sdf_file)
        if chebi_names:
            features = _filter_features_by_chebi_names(features, chebi_names)
    elif filter_chebi_human_metabolites:
        logger.warning("ChEBI human metabolite filtering requested but no chebi_sdf_file provided. Set chebi_sdf_file in config.")

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
