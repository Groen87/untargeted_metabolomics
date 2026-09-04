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
        variance_use_nonzero: bool = False,  # Calculate variance only on non-zero/gap-filled values
        variance_nonzero_threshold: float = 1000.0,  # Threshold to consider a value as non-gap-filled
        
        # Filter 2: Single batch features (gap-filled in others)
        filter_single_batch: bool = True,
        min_batches: int = 2,  # Features must be present in at least this many batches
        single_batch_noise_quantile: float = 0.10,  # Noise threshold = this quantile * global median
        single_batch_preserve_high_signal: bool = False,  # Preserve features with high signal in any batch
        single_batch_high_signal_threshold: float = 100000.0,  # Keep if max in any batch >= this
        
        # Filter 3: Low intensity
        filter_low_intensity: bool = True,
        intensity_threshold: float = 10000.0,  # Minimum mean intensity across all samples
        intensity_quantile: Optional[float] = None,  # Alternative: remove bottom N% by intensity
        intensity_use_max: bool = False,  # Use max intensity instead of mean (preserves rare high-signal features)
        intensity_max_threshold: float = 100000.0,  # Minimum max intensity across all samples
        
        # Filter 4: QC-based filtering
        filter_qc_present: bool = True,
        filter_qc_intensity: bool = True,
        qc_intensity_quantile: float = 0.25,
        
        # Filter 5: Blank sample filter (contaminants)
        filter_blank_contaminants: bool = True,
        blank_pattern: str = "blanco",  # Pattern to identify blank samples
        blank_ratio_threshold: float = 2.0,  # Features with blank/sample ratio > this are removed
        blank_high_signal_threshold: float = 10.0,  # Keep features if any bio sample >= this * blank_mean
        
        # QC3 RSD filter
        filter_high_qc3_rsd: bool = True,
        qc3_rsd_threshold: float = 30.0,  # RSD threshold in QC3 samples
        qc3_rsd_intensity_threshold: float = 100000.0,  # Intensity threshold for tiered RSD filtering
        qc3_rsd_low_intensity: float = 50.0,  # RSD threshold for features below intensity threshold
        qc3_rsd_high_intensity: float = 25.0,  # RSD threshold for features at or above intensity threshold
        use_intensity_based_rsd: bool = False,  # Use tiered RSD thresholds based on intensity
        
        # General
        qc_pattern: str = "expQC",
        fallback_qc_pattern: Optional[str] = "QC3",
    ):
        self.filter_low_variance = filter_low_variance
        self.variance_threshold = variance_threshold
        self.variance_quantile = variance_quantile
        self.variance_use_nonzero = variance_use_nonzero
        self.variance_nonzero_threshold = variance_nonzero_threshold
        
        self.filter_single_batch = filter_single_batch
        self.min_batches = min_batches
        self.single_batch_noise_quantile = single_batch_noise_quantile
        self.single_batch_preserve_high_signal = single_batch_preserve_high_signal
        self.single_batch_high_signal_threshold = single_batch_high_signal_threshold
        
        self.filter_low_intensity = filter_low_intensity
        self.intensity_threshold = intensity_threshold
        self.intensity_quantile = intensity_quantile
        self.intensity_use_max = intensity_use_max
        self.intensity_max_threshold = intensity_max_threshold
        
        self.filter_qc_present = filter_qc_present
        self.filter_qc_intensity = filter_qc_intensity
        self.qc_intensity_quantile = qc_intensity_quantile
        
        self.filter_blank_contaminants = filter_blank_contaminants
        self.blank_pattern = blank_pattern
        self.blank_ratio_threshold = blank_ratio_threshold
        self.blank_high_signal_threshold = blank_high_signal_threshold
        
        self.filter_high_qc3_rsd = filter_high_qc3_rsd
        self.qc3_rsd_threshold = qc3_rsd_threshold
        self.qc3_rsd_intensity_threshold = qc3_rsd_intensity_threshold
        self.qc3_rsd_low_intensity = qc3_rsd_low_intensity
        self.qc3_rsd_high_intensity = qc3_rsd_high_intensity
        self.use_intensity_based_rsd = use_intensity_based_rsd
        
        self.qc_pattern = qc_pattern
        self.fallback_qc_pattern = fallback_qc_pattern
    
    def _is_hmdb_feature(self, feature_name: str) -> bool:
        """Check if a feature name contains 'HMDB' (should be preserved)."""
        return 'HMDB' in str(feature_name)

    def _get_hmdb_features(self, df: pd.DataFrame) -> set:
        """Get set of feature names that contain 'HMDB'."""
        return {idx for idx in df.index if self._is_hmdb_feature(idx)}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'FeatureFilter':
        """Create FeatureFilter from configuration dictionary."""
        return cls(
            filter_low_variance=config.get('filter_low_variance', True),
            variance_threshold=config.get('variance_threshold', 0.01),
            variance_quantile=config.get('variance_quantile', None),
            variance_use_nonzero=config.get('variance_use_nonzero', False),
            variance_nonzero_threshold=config.get('variance_nonzero_threshold', 1000.0),
            
            filter_single_batch=config.get('filter_single_batch', True),
            min_batches=config.get('min_batches', 2),
            single_batch_noise_quantile=config.get('single_batch_noise_quantile', 0.10),
            single_batch_preserve_high_signal=config.get('single_batch_preserve_high_signal', False),
            single_batch_high_signal_threshold=config.get('single_batch_high_signal_threshold', 100000.0),
            
            filter_low_intensity=config.get('filter_low_intensity', True),
            intensity_threshold=config.get('intensity_threshold', 10000.0),
            intensity_quantile=config.get('intensity_quantile', None),
            intensity_use_max=config.get('intensity_use_max', False),
            intensity_max_threshold=config.get('intensity_max_threshold', 100000.0),
            
            filter_qc_present=config.get('filter_qc_present', True),
            filter_qc_intensity=config.get('filter_qc_intensity', True),
            qc_intensity_quantile=config.get('qc_intensity_quantile', 0.25),
            
            filter_blank_contaminants=config.get('filter_blank_contaminants', True),
            blank_pattern=config.get('blank_pattern', 'blanco'),
            blank_ratio_threshold=config.get('blank_ratio_threshold', 2.0),
            blank_high_signal_threshold=config.get('blank_high_signal_threshold', 10.0),
            
            filter_high_qc3_rsd=config.get('filter_high_qc3_rsd', True),
            qc3_rsd_threshold=config.get('qc3_rsd_threshold', 30.0),
            qc3_rsd_intensity_threshold=config.get('qc3_rsd_intensity_threshold', 100000.0),
            qc3_rsd_low_intensity=config.get('qc3_rsd_low_intensity', 50.0),
            qc3_rsd_high_intensity=config.get('qc3_rsd_high_intensity', 25.0),
            use_intensity_based_rsd=config.get('use_intensity_based_rsd', False),
            
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
        
        # Preserve HMDB features
        hmdb_features = self._get_hmdb_features(df)
        
        # Calculate variance for each feature across non-blank samples only
        non_blank_cols = [col for col in sample_cols if col not in blank_samples]
        
        if len(non_blank_cols) == 0:
            logger.warning("No non-blank samples for variance calculation. Skipping low variance filter.")
            return df, 0
        
        if self.variance_use_nonzero:
            # For gap-filled data: only calculate variance on values above threshold
            # This preserves features that have real variation in non-gap-filled samples
            df_nonblank = df[non_blank_cols]
            # Create mask of values above the gap-fill threshold
            nonzero_mask = df_nonblank > self.variance_nonzero_threshold
            # Calculate variance only on non-gap-filled values for each feature
            # Also track max intensity for preserving rare high-signal features
            variances = pd.Series(0.0, index=df.index, dtype=float)
            max_intensities = df_nonblank.max(axis=1)
            
            for feature in df.index:
                # Always preserve HMDB features
                if self._is_hmdb_feature(feature):
                    variances[feature] = float('inf')
                    continue
                    
                feature_values = df_nonblank.loc[feature]
                nonzero_values = feature_values[feature_values > self.variance_nonzero_threshold]
                if len(nonzero_values) >= 2:
                    variances[feature] = float(nonzero_values.var())
                else:
                    # If feature has at least one high-signal value, preserve it
                    # (this handles rare IMD features that appear in only 1-2 samples)
                    if max_intensities[feature] >= self.variance_nonzero_threshold * 10:
                        variances[feature] = float('inf')  # Force to pass any variance threshold
                    else:
                        variances[feature] = 0.0
        else:
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
        
        # Preserve HMDB features by forcing them to pass the filter
        mask_with_hmdb = mask.copy()
        for feature in hmdb_features:
            if feature in mask_with_hmdb.index:
                mask_with_hmdb[feature] = True
        
        return df[mask_with_hmdb], removed
    
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
        Features with 'HMDB' in their name are preserved.
        
        Uses only non-blank samples for batch presence calculation.
        A feature is considered "present" in a batch if its median intensity
        is above a small threshold (to avoid noise).
        
        Returns:
            Filtered DataFrame and number of features removed.
        """
        if not self.filter_single_batch:
            return df, 0
        
        # Preserve HMDB features
        hmdb_features = self._get_hmdb_features(df)
        
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
        # Use configurable quantile of global median as threshold to be above gap-filled noise
        global_median = df[sample_cols].median().median()
        noise_threshold = global_median * self.single_batch_noise_quantile if global_median > 0 else 1e-5
        
        present_mask = median_df > noise_threshold
        feature_batch_counts = present_mask.sum(axis=1)
        
        # Keep features present in at least min_batches
        mask = feature_batch_counts >= self.min_batches
        removed = (~mask).sum()
        
        # Ensure mask aligns with df.index
        mask = mask.reindex(df.index, fill_value=True)
        
        # If enabled, also preserve features with high signal in any batch (for rare IMD features)
        if self.single_batch_preserve_high_signal:
            # Check if any batch has max value >= high signal threshold
            batch_maxes = {}
            for batch, cols in batch_samples.items():
                if len(cols) > 0:
                    batch_df = df[cols]
                    batch_maxes[batch] = batch_df.max(axis=1)
            
            if batch_maxes:
                max_df = pd.DataFrame(batch_maxes)
                feature_max = max_df.max(axis=1)
                high_signal_mask = feature_max >= self.single_batch_high_signal_threshold
                # Keep features that either pass batch count OR have high signal
                mask = mask | high_signal_mask.reindex(df.index, fill_value=False)
                # Recalculate removed count
                original_removed = removed
                removed = (~mask).sum()
                logger.info(f"  Single batch filter (min_batches={self.min_batches}, noise_threshold={noise_threshold:.2f}, high_signal_preserved): removed {removed} features ({original_removed} before high-signal preservation)")
            else:
                logger.info(f"  Single batch filter (min_batches={self.min_batches}, noise_threshold={noise_threshold:.2f}): removed {removed} features")
        else:
            logger.info(f"  Single batch filter (min_batches={self.min_batches}, noise_threshold={noise_threshold:.2f}): removed {removed} features")
        
        # Preserve HMDB features by forcing them to pass the filter
        mask_with_hmdb = mask.copy()
        for feature in hmdb_features:
            if feature in mask_with_hmdb.index:
                mask_with_hmdb[feature] = True
        
        return df[mask_with_hmdb], removed
    
    def _filter_low_intensity(
        self,
        df: pd.DataFrame,
        sample_cols: List[str],
        blank_samples: List[str],
    ) -> Tuple[pd.DataFrame, int]:
        """
        Remove features with low intensity.
        
        Uses only non-blank samples for intensity calculation.
        Features with 'HMDB' in their name are preserved.
        When intensity_use_max=True, uses maximum intensity instead of mean to preserve rare high-signal features.
        
        Returns:
            Filtered DataFrame and number of features removed.
        """
        if not self.filter_low_intensity:
            return df, 0
        
        # Get HMDB features to preserve
        hmdb_features = self._get_hmdb_features(df)
        
        # Calculate intensity for each feature across non-blank samples
        non_blank_cols = [col for col in sample_cols if col not in blank_samples]
        
        if len(non_blank_cols) == 0:
            logger.warning("No non-blank samples for intensity calculation. Skipping low intensity filter.")
            return df, 0
        
        if self.intensity_use_max:
            # Use maximum intensity - preserves rare high-signal features (IMD)
            intensities = df[non_blank_cols].max(axis=1)
            mask = intensities >= self.intensity_max_threshold
            removed = (intensities < self.intensity_max_threshold).sum()
            logger.info(f"  Low intensity filter (max > {self.intensity_max_threshold}): removed {removed} features")
        elif self.intensity_quantile is not None:
            # Remove bottom N% by mean intensity
            mean_intensities = df[non_blank_cols].mean(axis=1)
            threshold = mean_intensities.quantile(self.intensity_quantile)
            mask = mean_intensities >= threshold
            removed = (mean_intensities < threshold).sum()
            logger.info(f"  Low intensity filter (quantile {self.intensity_quantile}): removed {removed} features")
        else:
            # Remove features with mean intensity below threshold
            mean_intensities = df[non_blank_cols].mean(axis=1)
            mask = mean_intensities >= self.intensity_threshold
            removed = (mean_intensities < self.intensity_threshold).sum()
            logger.info(f"  Low intensity filter (threshold={self.intensity_threshold}): removed {removed} features")
        
        # Preserve HMDB features by forcing them to pass the filter
        mask_with_hmdb = mask.copy()
        for feature in hmdb_features:
            if feature in mask_with_hmdb.index:
                mask_with_hmdb[feature] = True
        
        return df[mask_with_hmdb], removed
    

    def _filter_high_qc3_rsd(
        self,
        df: pd.DataFrame,
        sample_cols: List[str],
        qc_samples: List[str],
        blank_samples: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Filter 5: Remove features with high RSD in QC3 samples.
        
        Uses tiered RSD thresholds based on feature intensity:
        - Features with mean intensity < 100000: RSD threshold = 50%
        - Features with mean intensity >= 100000: RSD threshold = 25%
        
        This accounts for the fact that low-intensity features naturally have
        higher CV/RSD. Features with 'HMDB' in their name are preserved.
        
        Args:
            df: DataFrame with features as rows, samples as columns
            sample_cols: List of all sample column names
            qc_samples: List of QC sample column names
            blank_samples: Optional list of blank sample column names
            
        Returns:
            Tuple of (filtered DataFrame, number of features removed)
        """
        if not self.filter_high_qc3_rsd or not qc_samples:
            return df, 0
        
        # Preserve HMDB features
        hmdb_features = self._get_hmdb_features(df)
        
        # Identify QC3 samples specifically
        qc3_samples = [col for col in qc_samples if 'QC3' in col]
        
        if len(qc3_samples) < 2:
            logger.debug("  High QC3 RSD filter: skipped (need at least 2 QC3 samples)")
            return df, 0
        
        # Calculate RSD and intensity from QC3 samples for tiered filtering
        qc3_df = df[qc3_samples]
        feature_means = qc3_df.mean(axis=1)
        feature_stds = qc3_df.std(axis=1)
        feature_rsds = (feature_stds / feature_means * 100).fillna(0)
        
        # Use QC3 mean intensities for the tiered classification
        # This is consistent with RSD calculation and avoids outliers in other sample types
        feature_mean_intensities = feature_means
        
        # Apply tiered RSD thresholds based on intensity
        if self.use_intensity_based_rsd:
            # Use configurable tiered thresholds
            low_intensity_mask = feature_mean_intensities < self.qc3_rsd_intensity_threshold
            high_rsd_mask = pd.Series(False, index=df.index)
            high_rsd_mask[low_intensity_mask] = feature_rsds[low_intensity_mask] > self.qc3_rsd_low_intensity
            high_rsd_mask[~low_intensity_mask] = feature_rsds[~low_intensity_mask] > self.qc3_rsd_high_intensity
            threshold_str = f"RSD > {self.qc3_rsd_low_intensity}% (intensity < {self.qc3_rsd_intensity_threshold}) or > {self.qc3_rsd_high_intensity}% (intensity >= {self.qc3_rsd_intensity_threshold})"
        else:
            # Default tiered behavior: 50% for intensity < 100000, 25% for intensity >= 100000
            low_intensity_mask = feature_mean_intensities < 100000.0
            high_rsd_mask = pd.Series(False, index=df.index)
            high_rsd_mask[low_intensity_mask] = feature_rsds[low_intensity_mask] > 50.0
            high_rsd_mask[~low_intensity_mask] = feature_rsds[~low_intensity_mask] > 25.0
            threshold_str = "RSD > 50% (intensity < 100000) or > 25% (intensity >= 100000)"
        
        high_rsd_features = high_rsd_mask[high_rsd_mask].index
        
        if len(high_rsd_features) == 0:
            logger.debug("  High QC3 RSD filter: no features removed")
            return df, 0
        
        # Remove high RSD features, but preserve HMDB features
        high_rsd_features_to_remove = [f for f in high_rsd_features if f not in hmdb_features]
        df_filtered = df.drop(index=high_rsd_features_to_remove)
        removed = len(high_rsd_features_to_remove)
        
        logger.info(f"  High QC3 RSD filter ({threshold_str}): removed {removed} features")
        
        return df_filtered, removed

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
        
        A feature is considered a contaminant if the average of biological samples
        is NOT at least blank_ratio_threshold times higher than the average in blank samples.
        Features with 'HMDB' in their name are preserved.
        
        Formula: contaminant if bio_mean < (blank_mean * blank_ratio_threshold)
        Default threshold=2.0: keeps features where bio_mean >= 2 * blank_mean
        
        Returns:
            Filtered DataFrame and number of features removed.
        """
        if not self.filter_blank_contaminants or not blank_samples:
            return df, 0
        
        # Preserve HMDB features
        hmdb_features = self._get_hmdb_features(df)
        
        # Calculate mean intensity in blanks
        blank_df = df[blank_samples]
        blank_mean = blank_df.mean(axis=1)
        
        # Calculate mean intensity in non-blank (biological) samples
        non_blank_cols = [col for col in sample_cols if col not in blank_samples]
        
        if len(non_blank_cols) == 0:
            logger.warning("No non-blank samples for contaminant filter. Skipping.")
            return df, 0
        
        bio_df = df[non_blank_cols]
        bio_mean = bio_df.mean(axis=1)
        
        # A feature is a contaminant if bio_mean is NOT at least threshold times higher than blank_mean
        # BUT: keep features where ANY biological sample has >= 10x blank_mean (rare high-signal features)
        # i.e., contaminant if (bio_mean < blank_mean * threshold) AND (no bio sample >= 10 * blank_mean)
        
        # Check if any biological sample has >= 10x the blank mean for this feature
        blank_mean_series = pd.Series(blank_mean, index=df.index)
        bio_df = df[non_blank_cols]
        
        # For each feature, check if any bio sample >= high_signal_threshold * blank_mean
        max_bio = bio_df.max(axis=1)
        has_high_signal = (max_bio >= self.blank_high_signal_threshold * blank_mean_series)
        
        # Contaminant: bio_mean < threshold * blank_mean AND no high-signal bio sample
        contaminant_mask = (bio_mean < blank_mean * self.blank_ratio_threshold) & (blank_mean > 0) & (~has_high_signal)
        
        # Keep features that are NOT contaminants
        keep_mask = ~contaminant_mask
        removed = contaminant_mask.sum()
        
        logger.info(f"  Blank contaminant filter (bio < blank*{self.blank_ratio_threshold}, with {self.blank_high_signal_threshold}x loophole): removed {removed} features")
        
        # Preserve HMDB features by forcing them to pass the filter
        keep_mask_with_hmdb = keep_mask.copy()
        for feature in hmdb_features:
            if feature in keep_mask_with_hmdb.index:
                keep_mask_with_hmdb[feature] = True
        
        return df[keep_mask_with_hmdb], removed


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
        
        # Count HMDB features (these will be preserved from filtering)
        hmdb_count = sum(1 for idx in df.index if self._is_hmdb_feature(idx))
        if hmdb_count > 0:
            logger.info(f"Found {hmdb_count} HMDB features (will be preserved from filtering)")
        
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
        
        # Filter 1: Low intensity (do this first to remove low-signal features)
        df, removed = self._filter_low_intensity(df, sample_cols, blank_samples)
        total_removed += removed
        
        # Filter 2: Blank contaminants (remove contaminants before other filters)
        df, removed = self._filter_blank_contaminants(df, sample_cols, blank_samples)
        total_removed += removed
        
        # Filter 3: Low variance
        df, removed = self._filter_low_variance(df, sample_cols, blank_samples)
        total_removed += removed
        
        # Filter 4: Single batch
        if batch_info:
            df, removed = self._filter_single_batch(df, sample_cols, batch_info, blank_samples)
            total_removed += removed
        else:
            logger.debug("  Single batch filter: skipped (no batch_info provided)")
        
        # Filter 5: High RSD in QC3 samples (remove features with high RSD in QC3)
        df, removed = self._filter_high_qc3_rsd(df, sample_cols, qc_samples, blank_samples)
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
