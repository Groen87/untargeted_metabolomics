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
    
    # Clean sample IDs
    clinical_data['sample_id'] = clinical_data['sample_id'].astype(str).str.strip()
    clinical_data = clinical_data.set_index('sample_id')
    
    # Standardize metabolite column names
    metabolites_after.columns = metabolites_after.columns.astype(str).str.strip()
    
    if metabolites_before is not None:
        metabolites_before.columns = metabolites_before.columns.astype(str).str.strip()
    
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
    
    # Align samples
    common_samples = list(set(clinical_data.index) & set(metabolites_after.columns))
    
    if not common_samples:
        raise ValueError(
            f"No overlapping samples between clinical ({len(clinical_data.index)}) "
            f"and metabolomics data ({len(metabolites_after.columns)})"
        )
    
    clinical_data = clinical_data.loc[common_samples]
    metabolites_after = metabolites_after[common_samples]
    
    if metabolites_before is not None:
        metabolites_before = metabolites_before[common_samples]
    
    # Impute NaN/inf values
    metabolites_after = _safe_impute(metabolites_after)
    
    if metabolites_before is not None:
        metabolites_before = _safe_impute(metabolites_before)
    
    logger.info(f"Sample count for QC: {clinical_data.shape[0]}")
    logger.info(f"Batch values: {sorted(clinical_data[batch_column].unique())}")
    
    # Run inmoose QC if available
    if not INMOOSE_AVAILABLE:
        logger.warning("Skipping QC report generation (inmoose not available).")
        return
    
    try:
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
        raise
