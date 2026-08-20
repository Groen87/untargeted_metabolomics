"""
Quality control module for combined batch pipeline.

This module handles:
1. Running QC analysis on corrected data
2. Generating QC reports using inmoose (if available)
3. Calculating QC metrics

Based on metabolomics_pipeline's run_final_qc which works correctly.
"""

from typing import Optional, List
import pandas as pd
from pathlib import Path
import logging
import numpy as np
import traceback
import re
import warnings

try:
    from inmoose.cohort_qc.cohort_metric import CohortMetric
    from inmoose.cohort_qc.qc_report import QCReport
    INMOOSE_AVAILABLE = True
except ImportError:
    INMOOSE_AVAILABLE = False

logger = logging.getLogger(__name__)


def _safe_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace NaN and inf values with half the minimum positive value.
    
    This is critical for inmoose which uses linear regression internally.
    Divide-by-zero warnings from statsmodels occur when features have
    zero variance or all zeros after imputation.
    """
    df = df.copy()
    
    # Convert to numeric, coercing errors to NaN
    df = df.apply(pd.to_numeric, errors="coerce")
    
    # Calculate minimum positive value across the entire dataframe
    positive_values = df[df > 0]
    if positive_values.empty:
        # If no positive values, use a small default
        min_pos = 1e-10
    else:
        min_pos = positive_values.min().min()
    
    fill_value = min_pos / 2 if pd.notna(min_pos) and min_pos > 0 else 1e-10
    
    # Replace NaN with fill_value
    df = df.fillna(fill_value)
    
    # Replace inf/-inf with fill_value
    df = df.replace([np.inf, -np.inf], fill_value)
    
    # Ensure no zeros remain (can cause divide by zero in statsmodels)
    # Replace zeros with fill_value as well
    df = df.replace(0, fill_value)
    
    return df


def _normalize_column_name(col: str) -> str:
    """
    Normalize column name by removing common prefixes and suffixes.
    This ensures clinical sample IDs match metabolite column names.
    """
    # Remove Area: prefix (case insensitive)
    col = re.sub(r'^Area:\s*', '', col, flags=re.IGNORECASE)
    # Remove .raw suffix (case insensitive)
    col = re.sub(r'\.raw$', '', col, flags=re.IGNORECASE)
    # Strip any remaining whitespace
    return col.strip()


def run_qc_analysis(
    clinical_data: pd.DataFrame,
    metabolites_after: pd.DataFrame,
    metabolites_before: Optional[pd.DataFrame] = None,
    batch_column: str = "batch",
    output_path: str = "reports",
) -> None:
    """
    Run quality control analysis on corrected data.
    
    Args:
        clinical_data: DataFrame with sample metadata (must have 'sample_id' or 'original_col' column)
        metabolites_after: DataFrame with corrected metabolomics data
        metabolites_before: DataFrame with data before correction (optional)
        batch_column: Name of batch column in clinical_data
        output_path: Directory to save QC report
    """
    if metabolites_after.empty or metabolites_after is None:
        logger.warning("Skipping QC: metabolites_after is empty or None.")
        return
    
    clinical_data = clinical_data.copy()
    metabolites_after = metabolites_after.copy()
    
    if metabolites_before is not None:
        metabolites_before = metabolites_before.copy()
    
    # ------------------------------------------------------------------
    # 1. Fix sample IDs - ensure we have sample_id column
    # ------------------------------------------------------------------
    if "sample_id" not in clinical_data.columns:
        if 'original_col' in clinical_data.columns:
            # Use original_col as sample_id
            clinical_data['sample_id'] = clinical_data['original_col']
        else:
            raise ValueError("clinical_data must contain 'sample_id' or 'original_col' column")
    
    # Normalize clinical sample IDs (remove prefixes/suffixes) - do this BEFORE setting index
    clinical_data['sample_id'] = clinical_data['sample_id'].astype(str).map(_normalize_column_name)
    
    # Convert batch column to numeric BEFORE setting index
    if batch_column not in clinical_data.columns:
        raise ValueError(f"Batch column '{batch_column}' not found in clinical_data.")
    
    # Map string batch names to numeric IDs
    batch_values = clinical_data[batch_column].unique()
    try:
        # Try direct numeric conversion first
        clinical_data[batch_column] = pd.to_numeric(clinical_data[batch_column], errors="raise")
    except (ValueError, TypeError):
        # If that fails, create a mapping from string batch names to numeric IDs
        batch_to_num = {b: i+1 for i, b in enumerate(sorted(batch_values))}
        clinical_data[batch_column] = clinical_data[batch_column].map(batch_to_num)
    
    # Ensure it's integer type
    clinical_data[batch_column] = clinical_data[batch_column].astype(int)
    
    # Now set the index
    clinical_data = clinical_data.set_index("sample_id")
    
    # Normalize metabolite column names
    metabolites_after.columns = metabolites_after.columns.astype(str).map(_normalize_column_name)
    
    if metabolites_before is not None:
        metabolites_before.columns = metabolites_before.columns.astype(str).map(_normalize_column_name)
    
    # ------------------------------------------------------------------
    # 2. Align samples by direct intersection
    # ------------------------------------------------------------------
    common_samples = metabolites_after.columns.intersection(clinical_data.index)
    
    logger.info(f"Found {len(common_samples)} common samples")
    
    if len(common_samples) == 0:
        # Debug output
        logger.error(f"No overlapping samples!")
        logger.error(f"Clinical index (first 5): {list(clinical_data.index[:5])}")
        logger.error(f"Metabolite columns (first 5): {list(metabolites_after.columns[:5])}")
        logger.error(f"Batch values in clinical_data: {clinical_data[batch_column].unique()}")
        raise ValueError("No overlapping samples between clinical and metabolomics data.")
    
    clinical_data = clinical_data.loc[common_samples]
    metabolites_after = metabolites_after.loc[:, common_samples]
    
    if metabolites_before is not None:
        metabolites_before = metabolites_before.loc[:, common_samples]
    
    # ------------------------------------------------------------------
    # 3. NaN + Inf + Zero cleanup (CRITICAL for PCA in inmoose)
    # ------------------------------------------------------------------
    # Suppress statsmodels warnings - we handle the data properly
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        metabolites_after = _safe_impute(metabolites_after)
        
        if metabolites_before is not None:
            metabolites_before = _safe_impute(metabolites_before)
    
    # ------------------------------------------------------------------
    # 4. Diagnostics
    # ------------------------------------------------------------------
    logger.info(f"Sample count for QC: {clinical_data.shape[0]}")
    logger.info(f"Batch values: {sorted(clinical_data[batch_column].unique())}")
    logger.info(f"Batch dtype: {clinical_data[batch_column].dtype}")
    
    # ------------------------------------------------------------------
    # 5. Run inmoose QC
    # ------------------------------------------------------------------
    if not INMOOSE_AVAILABLE:
        logger.warning("Skipping QC report generation (inmoose not available).")
        return
    
    try:
        # Suppress warnings during QC processing
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            
            cohort_qc = CohortMetric(
                clinical_df=clinical_data,
                batch_column=batch_column,
                data_expression_df=metabolites_after,
                data_expression_df_before=metabolites_before,
            )
            
            cohort_qc.process()
            
            qc_report = QCReport(cohort_qc)
            
            Path(output_path).mkdir(parents=True, exist_ok=True)
            
            report_path = Path(output_path) / "qc_report.html"
            qc_report.save_report(output_path=str(report_path))
            
            logger.info(f"QC report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"QC failed: {e}")
        logger.error(traceback.format_exc())


def log_qc_rsd_simple(
    data_before: pd.DataFrame,
    data_after: pd.DataFrame,
    qc_patterns: List[str],
    correction_name: str = "Correction",
) -> None:
    """
    Log RSD (Relative Standard Deviation) for QC samples before and after correction.
    
    Args:
        data_before: DataFrame with data before correction
        data_after: DataFrame with data after correction
        qc_patterns: List of patterns to identify QC samples (e.g., ['QC4', 'blaauw'])
        correction_name: Name of the correction method for logging
    """
    import re
    
    # Find QC samples matching any of the patterns
    qc_samples = []
    for col in data_before.columns:
        for pattern in qc_patterns:
            if re.search(pattern, col, re.IGNORECASE):
                qc_samples.append(col)
                break
    
    if len(qc_samples) < 2:
        logger.warning(f"Need at least 2 QC samples for RSD calculation. Found {len(qc_samples)}.")
        return
    
    logger.info(f"\n--- QC RSD Analysis ({correction_name}) ---")
    
    # Calculate RSD for each feature across QC samples
    # Before correction
    qc_before = data_before[qc_samples]
    means_before = qc_before.mean(axis=1)
    stds_before = qc_before.std(axis=1)
    rsds_before = (stds_before / means_before * 100).fillna(0)
    
    # After correction
    qc_after = data_after[qc_samples]
    means_after = qc_after.mean(axis=1)
    stds_after = qc_after.std(axis=1)
    rsds_after = (stds_after / means_after * 100).fillna(0)
    
    # Summary statistics
    mean_rsd_before = rsds_before.mean()
    median_rsd_before = rsds_before.median()
    mean_rsd_after = rsds_after.mean()
    median_rsd_after = rsds_after.median()
    
    logger.info(f"  QC samples used: {len(qc_samples)}")
    logger.info(f"  Mean RSD before {correction_name}: {mean_rsd_before:.2f}%")
    logger.info(f"  Median RSD before {correction_name}: {median_rsd_before:.2f}%")
    logger.info(f"  Mean RSD after {correction_name}: {mean_rsd_after:.2f}%")
    logger.info(f"  Median RSD after {correction_name}: {median_rsd_after:.2f}%")
    
    # Improvement
    improvement = mean_rsd_before - mean_rsd_after
    logger.info(f"  Mean RSD improvement: {improvement:.2f}%")
    
    # Features with improved RSD
    features_improved = (rsds_after < rsds_before).sum()
    total_features = len(rsds_before)
    logger.info(f"  Features with improved RSD: {features_improved}/{total_features}")
    
    # Worst features after correction
    worst_after = rsds_after.nlargest(5)
    if len(worst_after) > 0:
        logger.debug(f"  Top 5 features with highest RSD after {correction_name}:")
        for feature, rsd in worst_after.items():
            logger.debug(f"    {feature}: {rsd:.2f}%")
