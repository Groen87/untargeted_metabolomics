"""
Data loading and preprocessing module for outlier detection pipeline.

Handles:
- Loading merged_data_with_classification.csv
- Identifying feature vs non-feature columns
- Splitting data into train/validation/test sets based on Classification
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def load_data(
    input_file: str,
    non_feature_columns: List[str],
    patient_id_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data from CSV file.
    
    Args:
        input_file: Path to CSV file
        non_feature_columns: List of column names that are NOT features
        patient_id_column: Column name for patient IDs (if not index)
        
    Returns:
        Tuple of:
        - features: DataFrame of features (rows = samples, columns = features)
        - classification: Series with Classification values
        - oordeel: Series with Oordeel trageted values
    """
    logger.info(f"Loading data from {input_file}")
    
    # Load CSV
    df = pd.read_csv(input_file, index_col=0 if patient_id_column is None else None)
    
    if patient_id_column is not None:
        # Set patient ID as index
        df = df.set_index(patient_id_column)
    
    logger.info(f"Loaded data with shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Data cleaning: Filter samples based on Classification and Oordeel targeted
    # Rule: Keep samples where Classification indicates outlier status clearly
    # - Normal: Classification=0 AND Oordeel targeted=0
    # - Outlier: Classification=1,2,3 (regardless of Oordeel targeted)
    # - Remove: Classification=2 or 3 with Oordeel targeted=0 (ambiguous)
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
    # Classification=0 with Oordeel targeted!=0: treat as outlier (Classification=1)
    # This ensures Classification=0 means Normal, Classification!=0 means Outlier
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
    
    logger.info(f"Feature columns: {len(feature_cols)}")
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
    # Combine features and classification into a DataFrame for consistent filtering
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
