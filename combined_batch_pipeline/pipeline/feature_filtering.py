"""
Feature filtering module for combined batch pipeline.

This module provides configurable filtering steps to remove:
1. Features with almost no variation (low variance / near-constant)
2. Features only present in a single batch (gap-filled in others)
3. Features with low intensity
4. Features high in blank samples (contaminants)

All filters are configurable via YAML config.
"""

from typing import Tuple, List, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureFilter:
    """
    Configurable feature filtering for metabolomics data.
    
    All filters are optional and can be enabled/disabled via configuration.
    """
    
    def __init__(
        self,
        # Filter 1: Low variance (near-constant features)
        filter_low_variance: bool = True,
        variance_threshold: float = 0.01,  # Features with variance < this threshold are removed
        variance_quantile: Optional[float] = None,  # Alternative: remove bottom N% by variance
        
        # Filter 2: Single batch features (gap-filled in others)
        filter_single_batch: bool = True,
        min_batches: int = 2,  # Features must be present in at least this many batches
        
        # Filter 3: Low intensity
        filter_low_intensity: bool = True,
        intensity_threshold: float = 10000.0,  # Minimum mean intensity across all samples
        intensity_quantile: Optional[float] = None,  # Alternative: remove bottom N% by intensity
        
        # Filter 4: QC-based filtering
        filter_qc_present: bool = True,
        filter_qc_intensity: bool = True,
        qc_intensity_quantile: float = 0.25,
        
        # Filter 5: Blank sample filter (contaminants)
        filter_blank_contaminants: bool = True,
        blank_pattern: str = "blanco",  # Pattern to identify blank samples
        blank_ratio_threshold: float = 2.0,  # Features with blank/sample ratio > this are removed
        
        # General
        qc_pattern: str = "expQC",
        fallback_qc_pattern: Optional[str] = "QC3",
    ):
        self.filter_low_variance = filter_low_variance
        self.variance_threshold = variance_threshold
        self.variance_quantile = variance_quantile
        
        self.filter_single_batch = filter_single_batch
        self.min_batches = min_batches
        
        self.filter_low_intensity = filter_low_intensity
        self.intensity_threshold = intensity_threshold
        self.intensity_quantile = intensity_quantile
        
        self.filter_qc_present = filter_qc_present
        self.filter_qc_intensity = filter_qc_intensity
        self.qc_intensity_quantile = qc_intensity_quantile
        
        self.filter_blank_contaminants = filter_blank_contaminants
        self.blank_pattern = blank_pattern
        self.blank_ratio_threshold = blank_ratio_threshold
        
        self.qc_pattern = qc_pattern
        self.fallback_qc_pattern = fallback_qc_pattern
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'FeatureFilter':
        """Create FeatureFilter from configuration dictionary."""
        return cls(
            filter_low_variance=config.get('filter_low_variance', True),
            variance_threshold=config.get('variance_threshold', 0.01),
            variance_quantile=config.get('variance_quantile', None),
            
            filter_single_batch=config.get('filter_single_batch', True),
            min_batches=config.get('min_batches', 2),
            
            filter_low_intensity=config.get('filter_low_intensity', True),
            intensity_threshold=config.get('intensity_threshold', 10000.0),
            intensity_quantile=config.get('intensity_quantile', None),
            
            filter_qc_present=config.get('filter_qc_present', True),
            filter_qc_intensity=config.get('filter_qc_intensity', True),
            qc_intensity_quantile=config.get('qc_intensity_quantile', 0.25),
            
            filter_blank_contaminants=config.get('filter_blank_contaminants', True),
            blank_pattern=config.get('blank_pattern', 'blanco'),
            blank_ratio_threshold=config.get('blank_ratio_threshold', 2.0),
            
            qc_pattern=config.get('qc_pattern', 'expQC'),
            fallback_qc_pattern=config.get('fallback_qc_pattern', 'QC3'),
        )
    
    def _identify_qc_samples(
        self,
        sample_cols: List[str],
        sample_info: Optional[Dict[str, Dict]] = None,
    ) -> List[str]:
        """Identify QC samples from column names."""
        from .data_loader import extract_sample_id_from_column, extract_sample_type
        
        qc_samples = []
        
        for col in sample_cols:
            if sample_info and col in sample_info:
                sample_type = sample_info[col].get('sample_type', 'Sample')
                if sample_type == "QC":
                    qc_samples.append(col)
                    continue
            
            # Fallback to pattern matching
            sample_id = extract_sample_id_from_column(col)
            sample_type = extract_sample_type(sample_id)
            
            if self.qc_pattern in sample_id:
                qc_samples.append(col)
            elif self.fallback_qc_pattern and self.fallback_qc_pattern in sample_id:
                qc_samples.append(col)
            elif sample_type == "QC":
                qc_samples.append(col)
        
        return qc_samples
    
    def _identify_blank_samples(
        self,
        sample_cols: List[str],
        sample_info: Optional[Dict[str, Dict]] = None,
    ) -> List[str]:
        """Identify blank samples from column names (contain 'blanco')."""
        from .data_loader import extract_sample_id_from_column
        
        blank_samples = []
        
        for col in sample_cols:
            if sample_info and col in sample_info:
                sample_type = sample_info[col].get('sample_type', 'Sample')
                sample_id = sample_info[col].get('sample_id', col)
                if sample_type == "Blank" or self.blank_pattern.lower() in sample_id.lower():
                    blank_samples.append(col)
                    continue
            
            # Fallback to pattern matching on column name
            sample_id = extract_sample_id_from_column(col)
            if self.blank_pattern.lower() in sample_id.lower():
                blank_samples.append(col)
        
        return blank_samples
    
    def _filter_low_variance(
        self,
        df: pd.DataFrame,
        sample_cols: List[str],
        blank_samples: List[str],
    ) -> Tuple[pd.DataFrame, int]:
        """
        Remove features with almost no variation.
        
        Features with very low variance are either:
        - Uninformative (constant or near-constant)
        - Noise
        
        Uses only non-blank samples for variance calculation.
        
        Returns:
            Filtered DataFrame and number of features removed.
        """
        if not self.filter_low_variance:
            return df, 0
        
        # Calculate variance for each feature across non-blank samples only
        non_blank_cols = [col for col in sample_cols if col not in blank_samples]
        
        if len(non_blank_cols) == 0:
            logger.warning("No non-blank samples for variance calculation. Skipping low variance filter.")
            return df, 0
        
        variances = df[non_blank_cols].var(axis=1)
        
        if self.variance_quantile is not None:
            # Remove bottom N% by variance
            threshold = variances.quantile(self.variance_quantile)
            mask = variances >= threshold
            removed = (variances < threshold).sum()
            logger.info(f"  Low variance filter (quantile {self.variance_quantile}): removed {removed} features")
        else:
            # Remove features with variance below absolute threshold
            mask = variances >= self.variance_threshold
            removed = (variances < self.variance_threshold).sum()
            logger.info(f"  Low variance filter (threshold={self.variance_threshold}): removed {removed} features")
        
        return df[mask], removed
    
    def _filter_single_batch(
        self,
        df: pd.DataFrame,
        sample_cols: List[str],
        batch_info: Dict[str, str],  # column -> batch mapping
        blank_samples: List[str],
    ) -> Tuple[pd.DataFrame, int]:
        """
        Remove features only present in a single batch.
        
        These features are likely gap-filled (imputed) in all other batches,
        which means they're not reliable measurements.
        
        Uses only non-blank samples for batch presence calculation.
        A feature is considered "present" in a batch if its median intensity
        is above a small threshold (to avoid noise).
        
        Returns:
            Filtered DataFrame and number of features removed.
        """
        if not self.filter_single_batch:
            return df, 0
        
        # Group samples by batch, excluding blanks
        batch_samples = {}
        for col, batch in batch_info.items():
            if col in sample_cols and col not in blank_samples:
                if batch not in batch_samples:
                    batch_samples[batch] = []
                batch_samples[batch].append(col)
        
        if len(batch_samples) < 2:
            logger.debug("  Single batch filter: skipped (need at least 2 batches)")
            return df, 0
        
        # Vectorized approach: for each batch, compute median per feature
        # This is much faster than looping through each feature
        batch_medians = {}
        for batch, cols in batch_samples.items():
            if len(cols) == 0:
                continue
            batch_df = df[cols]
            # Compute median per feature for this batch
            medians = batch_df.median(axis=1)
            # Handle case where median returns a Series with duplicate index
            if isinstance(medians, pd.Series):
                medians = medians.groupby(level=0).first()
            batch_medians[batch] = medians
        
        # Combine into a DataFrame: features x batches
        median_df = pd.DataFrame(batch_medians)
        
        # Count how many batches each feature has median > small threshold
        # Use a small threshold to avoid noise (e.g., 1% of global median)
        global_median = df[sample_cols].median().median()
        noise_threshold = global_median * 0.01 if global_median > 0 else 1e-6
        
        present_mask = median_df > noise_threshold
        feature_batch_counts = present_mask.sum(axis=1)
        
        # Keep features present in at least min_batches
        mask = feature_batch_counts >= self.min_batches
        removed = (~mask).sum()
        
        logger.info(f"  Single batch filter (min_batches={self.min_batches}, noise_threshold={noise_threshold:.2f}): removed {removed} features")
        
        return df[mask], removed
    
    def _filter_low_intensity(
        self,
        df: pd.DataFrame,
        sample_cols: List[str],
        blank_samples: List[str],
    ) -> Tuple[pd.DataFrame, int]:
        """
        Remove features with low intensity.
        
        Uses only non-blank samples for intensity calculation.
        
        Returns:
            Filtered DataFrame and number of features removed.
        """
        if not self.filter_low_intensity:
            return df, 0
        
        # Calculate mean intensity for each feature across non-blank samples
        non_blank_cols = [col for col in sample_cols if col not in blank_samples]
        
        if len(non_blank_cols) == 0:
            logger.warning("No non-blank samples for intensity calculation. Skipping low intensity filter.")
            return df, 0
        
        mean_intensities = df[non_blank_cols].mean(axis=1)
        
        if self.intensity_quantile is not None:
            # Remove bottom N% by intensity
            threshold = mean_intensities.quantile(self.intensity_quantile)
            mask = mean_intensities >= threshold
            removed = (mean_intensities < threshold).sum()
            logger.info(f"  Low intensity filter (quantile {self.intensity_quantile}): removed {removed} features")
        else:
            # Remove features with mean intensity below threshold
            mask = mean_intensities >= self.intensity_threshold
            removed = (mean_intensities < self.intensity_threshold).sum()
            logger.info(f"  Low intensity filter (threshold={self.intensity_threshold}): removed {removed} features")
        
        return df[mask], removed
    
    def _filter_qc_present(
        self,
        df: pd.DataFrame,
        sample_cols: List[str],
        qc_samples: List[str],
    ) -> Tuple[pd.DataFrame, int]:
        """
        Remove features not present in all QC samples.
        
        Returns:
            Filtered DataFrame and number of features removed.
        """
        if not self.filter_qc_present or not qc_samples:
            return df, 0
        
        qc_df = df[qc_samples]
        mask = qc_df.notna().all(axis=1)
        removed = (~mask).sum()
        
        logger.info(f"  QC present filter: removed {removed} features")
        
        return df[mask], removed
    
    def _filter_qc_intensity(
        self,
        df: pd.DataFrame,
        sample_cols: List[str],
        qc_samples: List[str],
    ) -> Tuple[pd.DataFrame, int]:
        """
        Remove features with low intensity in QC samples.
        
        Returns:
            Filtered DataFrame and number of features removed.
        """
        if not self.filter_qc_intensity or not qc_samples:
            return df, 0
        
        qc_df = df[qc_samples]
        qc_mean = qc_df.mean(axis=1)
        threshold = qc_mean.quantile(self.qc_intensity_quantile)
        mask = qc_mean >= threshold
        removed = (~mask).sum()
        
        logger.info(f"  QC intensity filter (quantile {self.qc_intensity_quantile}): removed {removed} features")
        
        return df[mask], removed
    
    def _filter_blank_contaminants(
        self,
        df: pd.DataFrame,
        sample_cols: List[str],
        blank_samples: List[str],
    ) -> Tuple[pd.DataFrame, int]:
        """
        Remove features that are high in blank samples (contaminants).
        
        A feature is considered a contaminant if its mean intensity in blanks
        is significantly higher than in biological samples.
        
        Returns:
            Filtered DataFrame and number of features removed.
        """
        if not self.filter_blank_contaminants or not blank_samples:
            return df, 0
        
        # Calculate mean intensity in blanks
        blank_df = df[blank_samples]
        blank_mean = blank_df.mean(axis=1)
        
        # Calculate mean intensity in non-blank samples
        non_blank_cols = [col for col in sample_cols if col not in blank_samples]
        
        if len(non_blank_cols) == 0:
            logger.warning("No non-blank samples for contaminant filter. Skipping.")
            return df, 0
        
        bio_df = df[non_blank_cols]
        bio_mean = bio_df.mean(axis=1)
        
        # Avoid division by zero
        bio_mean_safe = bio_mean.replace(0, np.nan)
        
        # Calculate ratio: blank_mean / bio_mean
        # Features with ratio > threshold are contaminants
        ratio = blank_mean / bio_mean_safe
        
        # Only consider features where bio_mean > 0
        valid_mask = bio_mean_safe > 0
        
        # Features are contaminants if bio_mean is not significantly above blank_mean
        # i.e., bio_mean <= blank_mean * threshold
        contaminant_mask = (ratio <= self.blank_ratio_threshold) & valid_mask
        
        # Keep features that are NOT contaminants
        keep_mask = ~contaminant_mask
        removed = contaminant_mask.sum()
        
        logger.info(f"  Blank contaminant filter (ratio > {self.blank_ratio_threshold}): removed {removed} features")
        
        return df[keep_mask], removed
    
    def filter(
        self,
        df: pd.DataFrame,
        sample_cols: List[str],
        batch_info: Optional[Dict[str, str]] = None,
        sample_info: Optional[Dict[str, Dict]] = None,
    ) -> pd.DataFrame:
        """
        Apply all configured filters to the DataFrame.
        
        Args:
            df: DataFrame with features as rows, samples as columns
            sample_cols: List of sample column names
            batch_info: Dictionary mapping sample columns to batch names
            sample_info: Optional dictionary with sample metadata
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        logger.info(f"\nApplying feature filters...")
        logger.info(f"Initial feature count: {len(df)}")
        
        initial_count = len(df)
        total_removed = 0
        
        # Identify QC and blank samples
        qc_samples = self._identify_qc_samples(sample_cols, sample_info)
        blank_samples = self._identify_blank_samples(sample_cols, sample_info)
        
        if qc_samples:
            logger.info(f"Found {len(qc_samples)} QC samples")
        if blank_samples:
            logger.info(f"Found {len(blank_samples)} blank samples")
        
        # Apply filters in order
        
        # Filter 1: Blank contaminants (do this first to remove contaminants before other filters)
        df, removed = self._filter_blank_contaminants(df, sample_cols, blank_samples)
        total_removed += removed
        
        # Filter 2: Low variance
        df, removed = self._filter_low_variance(df, sample_cols, blank_samples)
        total_removed += removed
        
        # Filter 3: Single batch
        if batch_info:
            df, removed = self._filter_single_batch(df, sample_cols, batch_info, blank_samples)
            total_removed += removed
        else:
            logger.debug("  Single batch filter: skipped (no batch_info provided)")
        
        # Filter 4: Low intensity
        df, removed = self._filter_low_intensity(df, sample_cols, blank_samples)
        total_removed += removed
        

        

        
        # Summary
        if initial_count > 0:
            pct_removed = 100 * total_removed / initial_count
            logger.info(f"\nTotal: filtered out {total_removed}/{initial_count} features ({pct_removed:.1f}%)")
        
        # Ensure unique index
        if df.index.has_duplicates:
            logger.info(f"  Combining {df.index.duplicated().sum()} duplicate feature names by taking mean")
            df = df.groupby(level=0).mean()
        
        return df


def filter_features(
    df: pd.DataFrame,
    sample_cols: List[str],
    batch_info: Optional[Dict[str, str]] = None,
    sample_info: Optional[Dict[str, Dict]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Convenience function to filter features using configuration.
    
    Args:
        df: DataFrame with features as rows, samples as columns
        sample_cols: List of sample column names
        batch_info: Dictionary mapping sample columns to batch names
        sample_info: Optional dictionary with sample metadata
        config: Optional configuration dictionary
        
    Returns:
        Filtered DataFrame
    """
    if config is None:
        config = {}
    
    feature_filter = FeatureFilter.from_config(config)
    return feature_filter.filter(df, sample_cols, batch_info, sample_info)
