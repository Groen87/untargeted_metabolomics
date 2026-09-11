"""
Data loading and preprocessing module for outlier detection pipeline.

Handles:
- Loading merged_data_with_classification.csv
- Identifying feature vs non-feature columns
- Filtering features against an HMDB endogenous metabolites keep-list
  - Reads a precomputed TSV produced by hmdb_drug_filter.py
    (columns: HMDB_ID, Name, Synonyms) listing metabolites with
    Metabolic or Disease pathways
  - Keeps only feature columns that match a name in the keep-list
  - HMDB-prefixed feature columns are always kept
- Splitting data into train/validation/test sets based on Classification
"""

import pandas as pd
import numpy as np
import unicodedata
from typing import Tuple, Dict, List, Optional, Set
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import pickle
import hashlib


def _normalize_name(name: str) -> str:
    """
    Normalize a metabolite or feature-column name for exact matching.

    - Strips surrounding whitespace
    - Applies Unicode NFKC normalization (e.g. folds full-width digits, unifies
      Greek alpha variants such as U+0391 'Α' vs U+0041 'A')
    - Uppercases
    - Strips a leading UTF-8 BOM if present

    Returns the normalized name. Exact (not partial) matching is preserved.
    """
    if name is None:
        return ''
    s = str(name)
    if s.startswith('\ufeff'):
        s = s[1:]
    s = unicodedata.normalize('NFKC', s).strip().upper()
    return s

logger = logging.getLogger(__name__)


def _get_hmdb_cache_path(endogenous_file: str) -> Path:
    """
    Get the cache file path for a given endogenous metabolites file.

    Args:
        endogenous_file: Path to endogenous_metabolites.tsv

    Returns:
        Path to the cache pickle file
    """
    cache_dir = Path.home() / ".cache" / "hmdb_metabolomics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_hash = hashlib.md5(endogenous_file.encode()).hexdigest()[:16]
    return cache_dir / f"endogenous_names_{file_hash}.pkl"


def _load_endogenous_metabolite_names(endogenous_file: str, use_cache: bool = True) -> Set[str]:
    """
    Load the set of endogenous metabolite names from a precomputed TSV file.

    The TSV is produced by hmdb_drug_filter.py and contains metabolites that
    have at least one Metabolic or Disease pathway. It has the columns:
        HMDB_ID <TAB> Name <TAB> Synonyms
    where Synonyms is a '; '-separated list of alternative names.

    The returned set contains the HMDB ID, the primary Name, and each synonym,
    all normalized to uppercase. Very short names (shorter than 3 characters)
    are skipped to avoid false-positive matches.

    Args:
        endogenous_file: Path to endogenous_metabolites.tsv
        use_cache: Whether to use cached results if available

    Returns:
        Set of endogenous metabolite names (uppercase). Empty on failure.
    """
    try:
        cache_path = _get_hmdb_cache_path(endogenous_file)

        # Try to load from cache first
        if use_cache and cache_path.exists():
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            logger.info(f"Loaded endogenous metabolite names from cache: {cache_path}")
            return cached_data['endogenous']

        endogenous_names: Set[str] = set()

        endogenous_path = Path(endogenous_file)
        with open(endogenous_path, 'r', encoding='utf-8') as f:
            header = f.readline().rstrip('\n')
            expected_cols = ['HMDB_ID', 'Name', 'Synonyms']
            cols = [c.strip() for c in header.split('\t')]
            if cols != expected_cols:
                logger.warning(
                    f"Unexpected header in {endogenous_file}: {cols}. "
                    f"Expected {expected_cols}. Proceeding by column position."
                )

            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                fields = line.split('\t')
                # Pad in case Synonyms is missing
                while len(fields) < 3:
                    fields.append('')
                hmdb_id = fields[0]
                name = fields[1]
                synonyms_str = fields[2]

                hmdb_id = _normalize_name(hmdb_id)
                name = _normalize_name(name)
                if hmdb_id:
                    endogenous_names.add(hmdb_id)
                if name:
                    endogenous_names.add(name)
                if synonyms_str:
                    for syn in synonyms_str.split(';'):
                        syn = _normalize_name(syn)
                        if syn:
                            endogenous_names.add(syn)

        # Drop very short names that cause false positives
        endogenous_names = {n for n in endogenous_names if len(n) >= 3}

        # Save to cache
        if use_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump({'endogenous': endogenous_names}, f)
            logger.info(f"Saved endogenous metabolite names cache to {cache_path}")

        logger.info(f"Loaded {len(endogenous_names)} endogenous metabolite names from {endogenous_file}")

        if len(endogenous_names) > 0:
            sample_endogenous = list(endogenous_names)[:10]
            logger.info(f"Sample endogenous metabolite names: {sample_endogenous}{'...' if len(endogenous_names) > 10 else ''}")

        return endogenous_names

    except Exception as e:
        logger.error(f"Failed to load endogenous metabolites file {endogenous_file}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return set()


def _filter_to_endogenous_features(
    features: pd.DataFrame,
    endogenous_names: set,
) -> pd.DataFrame:
    """
    Keep only feature columns that match the HMDB endogenous metabolite keep-list.

    A feature column is retained if EITHER:
      - its name contains 'HMDB' (endogenous metabolites are always kept), OR
      - its name EXACTLY matches (case-insensitive) a name in the keep-list.

    Feature columns that do not match are removed. This is a positive
    keep-list: features not present in the HMDB keep-list are dropped.

    Args:
        features: DataFrame with feature columns
        endogenous_names: Set of HMDB endogenous metabolite names (uppercase)

    Returns:
        Filtered DataFrame containing only kept feature columns
    """
    if not endogenous_names:
        logger.warning("No endogenous metabolite names provided. Returning all features.")
        return features

    original_cols = set(features.columns)

    kept_columns = []
    removed_cols = []
    n_matched_hmdb = 0
    n_matched_name = 0

    for col in features.columns:
        col_upper = _normalize_name(col)

        # Always keep HMDB features regardless of keep-list match
        if 'HMDB' in col_upper:
            kept_columns.append(col)
            n_matched_hmdb += 1
            continue

        # Exact match (case-insensitive, Unicode-normalized) against the keep-list
        if col_upper in endogenous_names:
            kept_columns.append(col)
            n_matched_name += 1
        else:
            removed_cols.append(col)

    filtered_features = features[kept_columns]
    n_removed = len(original_cols) - len(kept_columns)

    logger.info(
        f"Filtered to endogenous metabolite features: {n_removed} features removed, "
        f"{len(kept_columns)} endogenous features retained "
        f"({n_matched_hmdb} matched by HMDB prefix, {n_matched_name} matched by name)"
    )

    if n_removed > 0:
        logger.info(f"Example removed features: {removed_cols[:10]}{'...' if n_removed > 10 else ''}")
        if len(endogenous_names) > 0:
            sample_keep = list(endogenous_names)[:5]
            logger.info(f"Example keep-list names (normalized): {sample_keep}")

    return filtered_features


def _exclude_metabolites(
    features: pd.DataFrame,
    exclude_names: List[str],
) -> pd.DataFrame:
    """
    Drop feature columns whose names match a user-supplied exclude list.

    Matching is exact and case-insensitive after Unicode normalization (the
    same normalization used for the HMDB keep-list), so a user can enter
    metabolite names with any casing or surrounding whitespace.

    Args:
        features: DataFrame with feature columns
        exclude_names: List of metabolite/feature names to exclude

    Returns:
        DataFrame with the matched feature columns removed
    """
    if not exclude_names:
        return features

    exclude_set = {
        _normalize_name(name)
        for name in exclude_names
        if name is not None and str(name).strip() != ''
    }
    exclude_set = {n for n in exclude_set if n}
    if not exclude_set:
        return features

    original_cols = list(features.columns)
    kept_columns = [
        col for col in original_cols
        if _normalize_name(col) not in exclude_set
    ]
    removed_cols = [col for col in original_cols if col not in kept_columns]

    filtered_features = features[kept_columns]

    logger.info(
        f"Excluded {len(removed_cols)} metabolite features, "
        f"{len(kept_columns)} features remaining"
    )
    if removed_cols:
        logger.info(
            f"Excluded features: {removed_cols[:10]}"
            f"{'...' if len(removed_cols) > 10 else ''}"
        )

    return filtered_features


def load_data(
    input_file: str,
    non_feature_columns: List[str],
    patient_id_column: Optional[str] = None,
    endogenous_metabolites_file: Optional[str] = None,
    filter_to_endogenous: bool = False,
    use_hmdb_cache: bool = True,
    exclude_metabolites: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data from CSV file and optionally filter to endogenous metabolite features.

    Args:
        input_file: Path to CSV file
        non_feature_columns: List of column names that are NOT features
        patient_id_column: Column name for patient IDs (if not index)
        endogenous_metabolites_file: Path to endogenous_metabolites.tsv (the
            output of hmdb_drug_filter.py) used as a positive keep-list
        filter_to_endogenous: Whether to filter features to the endogenous
            metabolite keep-list
        use_hmdb_cache: Whether to use cached HMDB data if available
        exclude_metabolites: Optional list of metabolite/feature names to drop
            from the analysis (exact, case-insensitive match)

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
    
    # Exclude user-specified metabolite features (exact, case-insensitive match)
    if exclude_metabolites:
        features = _exclude_metabolites(features, exclude_metabolites)

    # Filter to endogenous metabolite features if requested
    if filter_to_endogenous and endogenous_metabolites_file:
        endogenous_path = Path(endogenous_metabolites_file)
        if endogenous_path.exists():
            endogenous_names = _load_endogenous_metabolite_names(str(endogenous_path), use_cache=use_hmdb_cache)
            if endogenous_names:
                features = _filter_to_endogenous_features(features, endogenous_names)
            else:
                logger.warning(f"Could not load endogenous metabolite names from {endogenous_metabolites_file}. Using all features.")
        else:
            logger.warning(f"Endogenous metabolites file not found at {endogenous_metabolites_file}. Using all features.")
    
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
