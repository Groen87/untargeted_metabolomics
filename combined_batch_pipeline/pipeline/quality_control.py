"""
Quality control module for combined batch pipeline.

This module handles:
1. Running QC analysis on corrected data
2. Generating QC reports using inmoose (if available)
3. Calculating QC metrics
"""

from typing import Optional, List
import pandas as pd
from pathlib import Path
import logging
import numpy as np
import re

try:
    from inmoose.cohort_qc.cohort_metric import CohortMetric
    from inmoose.cohort_qc.qc_report import QCReport
    INMOOSE_AVAILABLE = True
except ImportError:
    INMOOSE_AVAILABLE = False

logger = logging.getLogger(__name__)


def _safe_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN and inf values with half the minimum positive value."""
    df = df.copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    
    min_pos = df[df > 0].min().min()
    fill_value = (min_pos / 2) if pd.notna(min_pos) else 1e-10
    
    df = df.fillna(fill_value)
    df = df.replace([np.inf, -np.inf], fill_value)
    
    return df


def _normalize_sample_id(sample_id: str) -> str:
    """Normalize sample ID by removing common prefixes and suffixes."""
    # Remove common prefixes
    for prefix in ['Area: ', 'area: ', 'Area:', 'area:']:
        if sample_id.startswith(prefix):
            sample_id = sample_id[len(prefix):]
    # Remove common suffixes
    for suffix in ['.raw', '.RAW', '_raw', '_RAW']:
        if sample_id.endswith(suffix):
            sample_id = sample_id[:-len(suffix)]
    return sample_id.strip()


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
        clinical_data: DataFrame with sample metadata
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
    
    # Validate clinical data
    if 'sample_id' not in clinical_data.columns:
        if 'original_col' in clinical_data.columns:
            clinical_data['sample_id'] = clinical_data['original_col']
        else:
            raise ValueError("clinical_data must contain 'sample_id' or 'original_col' column")
    
    # Clean sample IDs - convert to string and strip
    clinical_data['sample_id'] = clinical_data['sample_id'].astype(str).str.strip()
    
    # Normalize clinical sample IDs (remove prefixes/suffixes for matching)
    clinical_data['sample_id_normalized'] = clinical_data['sample_id'].apply(_normalize_sample_id)
    clinical_data = clinical_data.set_index('sample_id')
    
    # Standardize metabolite column names - convert to string and strip
    metabolites_after.columns = metabolites_after.columns.astype(str).str.strip()
    
    # Normalize metabolite column names for matching
    metabolites_after_normalized = pd.DataFrame(
        metabolites_after.values,
        index=metabolites_after.index,
        columns=metabolites_after.columns.map(_normalize_sample_id)
    )
    
    if metabolites_before is not None:
        metabolites_before.columns = metabolites_before.columns.astype(str).str.strip()
        metabolites_before_normalized = pd.DataFrame(
            metabolites_before.values,
            index=metabolites_before.index,
            columns=metabolites_before.columns.map(_normalize_sample_id)
        )
    else:
        metabolites_before_normalized = None
    
    # Validate batch column
    if batch_column not in clinical_data.columns:
        raise ValueError(f"Batch column '{batch_column}' not found in clinical_data.")
    
    # Convert batch column to numeric if possible, otherwise keep as string
    # inmoose expects numeric batch labels, so we need to map string batch names to numbers
    batch_values = clinical_data[batch_column].unique()
    
    # Check if batch values are already numeric
    try:
        clinical_data[batch_column] = pd.to_numeric(
            clinical_data[batch_column],
            errors="raise"
        )
    except (ValueError, TypeError):
        # Batch values are strings (like 'MZ25_36'), map them to numeric IDs
        batch_to_num = {b: i+1 for i, b in enumerate(sorted(batch_values))}
        clinical_data[batch_column] = clinical_data[batch_column].map(batch_to_num)
    
    # Ensure batch column is integer type (not float or string)
    clinical_data[batch_column] = clinical_data[batch_column].astype(int)
    
    # Align samples using normalized IDs
    clinical_samples = set(clinical_data.index.map(_normalize_sample_id))
    metabolite_samples = set(metabolites_after_normalized.columns)
    common_samples = list(clinical_samples & metabolite_samples)
    
    if not common_samples:
        # Debug: show first few samples from each
        logger.error(f"No overlapping samples after normalization.")
        logger.error(f"Clinical samples (first 5): {list(clinical_data.index.map(_normalize_sample_id))[:5]}")
        logger.error(f"Metabolite samples (first 5): {list(metabolites_after_normalized.columns)[:5]}")
        raise ValueError(
            f"No overlapping samples between clinical ({len(clinical_data.index)}) "
            f"and metabolomics data ({len(metabolites_after.columns)})"
        )
    
    # Get original column names for the common samples
    clinical_original_cols = clinical_data.index.tolist()
    metabolite_original_cols = metabolites_after.columns.tolist()
    
    # Find which original columns correspond to common normalized samples
    common_original_cols = []
    for norm_sample in common_samples:
        # Find clinical column that normalizes to this
        clinical_match = [col for col in clinical_original_cols 
                        if _normalize_sample_id(col) == norm_sample]
        # Find metabolite column that normalizes to this
        metabolite_match = [col for col in metabolite_original_cols 
                          if _normalize_sample_id(col) == norm_sample]
        if clinical_match and metabolite_match:
            common_original_cols.append((clinical_match[0], metabolite_match[0]))
    
    # Extract the matching columns
    clinical_data_aligned = clinical_data.loc[[c[0] for c in common_original_cols]]
    metabolites_after_aligned = metabolites_after[[c[1] for c in common_original_cols]]
    
    if metabolites_before is not None:
        metabolites_before_aligned = metabolites_before[[c[1] for c in common_original_cols]]
    else:
        metabolites_before_aligned = None
    
    # Impute NaN/inf values
    metabolites_after_aligned = _safe_impute(metabolites_after_aligned)
    
    if metabolites_before_aligned is not None:
        metabolites_before_aligned = _safe_impute(metabolites_before_aligned)
    
    logger.info(f"Sample count for QC: {clinical_data_aligned.shape[0]}")
    logger.info(f"Batch values: {sorted(clinical_data_aligned[batch_column].unique())}")
    
    # Verify we have more than one batch
    unique_batches = clinical_data_aligned[batch_column].nunique()
    if unique_batches < 2:
        logger.error(f"QC analysis requires at least 2 batches, but found only {unique_batches}")
        raise ValueError(f"QC analysis requires at least 2 batches, but found only {unique_batches}")
    
    # Run inmoose QC if available
    if not INMOOSE_AVAILABLE:
        logger.warning("Skipping QC report generation (inmoose not available).")
        return
    
    try:
        cohort_qc = CohortMetric(
            clinical_df=clinical_data_aligned,
            batch_column=batch_column,
            data_expression_df=metabolites_after_aligned,
            data_expression_df_before=metabolites_before_aligned,
        )
        cohort_qc.process()
        
        qc_report = QCReport(cohort_qc)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        report_path = Path(output_path) / "qc_report.html"
        qc_report.save_report(output_path=str(report_path))
        
        logger.info(f"QC report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"QC failed: {e}")
        raise
