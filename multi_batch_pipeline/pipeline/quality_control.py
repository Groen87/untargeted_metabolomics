"""
Quality Control module for multi-batch metabolomics pipeline.

This module provides functionality to run comprehensive QC analysis on
merged and ComBat-corrected metabolomics data using the inmoose library.

Key Features:
- Sample ID alignment between clinical and metabolomics data
- Batch column validation and cleanup
- NaN and infinite value imputation (compatible with PCA)
- CohortMetric processing for batch effect assessment
- HTML QC report generation

Note:
    This module requires inmoose to be installed for full functionality.
    If inmoose is not available, QC will be skipped with a warning.
"""

from typing import Optional, List, Tuple
import pandas as pd
from pathlib import Path
import logging
import numpy as np
import traceback

# -----------------------------------------------------------------------------
# Optional QC imports (handle gracefully if not installed)
# -----------------------------------------------------------------------------

try:
    from inmoose.cohort_qc.cohort_metric import CohortMetric
    from inmoose.cohort_qc.qc_report import QCReport
    INMOOSE_AVAILABLE = True
except ImportError:
    INMOOSE_AVAILABLE = False

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def _safe_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform NaN-safe imputation for metabolomics data.
    
    Replaces NaN and infinite values with half the minimum positive value,
    which is a best practice for metabolomics data to avoid introducing bias.
    This is particularly important for PCA and other dimensionality reduction
    techniques that cannot handle NaN values.
    
    Args:
        df: Input DataFrame with potential NaN/inf values
        
    Returns:
        DataFrame with all NaN/inf values imputed
        
    Note:
        This function is also available in combat_utils.py but is duplicated here
        to allow quality_control.py to be used independently.
    """
    df = df.copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    
    # Find minimum positive value across all data
    min_pos = df[df > 0].min().min()
    
    # Use half the minimum positive value, or a small default if none found
    fill_value = (min_pos / 2) if pd.notna(min_pos) else 1e-10
    
    # Replace NaN and infinite values
    df = df.fillna(fill_value)
    df = df.replace([np.inf, -np.inf], fill_value)
    
    return df


def validate_clinical_data(
    clinical_data: pd.DataFrame,
    required_columns: List[str],
) -> pd.DataFrame:
    """
    Validate that clinical data contains required columns.
    
    Args:
        clinical_data: DataFrame to validate
        required_columns: List of column names that must be present
        
    Returns:
        Validated clinical_data DataFrame
        
    Raises:
        ValueError: If any required columns are missing
    """
    for col in required_columns:
        if col not in clinical_data.columns:
            raise ValueError(f"clinical_data must contain '{col}' column.")
    return clinical_data


def align_samples(
    clinical_data: pd.DataFrame,
    metabolites_data: pd.DataFrame,
    clinical_index_col: str = "sample_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align samples between clinical and metabolomics data.
    
    Args:
        clinical_data: DataFrame with clinical/sample metadata
        metabolites_data: DataFrame with metabolomics data (features x samples)
        clinical_index_col: Column name in clinical_data to use as index (default: "sample_id")
        
    Returns:
        Tuple of aligned (clinical_data, metabolites_data)
        
    Raises:
        ValueError: If no overlapping samples exist
    """
    # Get common samples
    clinical_samples = set(clinical_data[clinical_index_col].astype(str).str.strip())
    metabolite_samples = set(metabolites_data.columns.astype(str).str.strip())
    common_samples = list(clinical_samples & metabolite_samples)
    
    if not common_samples:
        raise ValueError(
            f"No overlapping samples between clinical ({len(clinical_samples)}) "
            f"and metabolomics data ({len(metabolite_samples)})"
        )
    
    logger.info(f"Found {len(common_samples)} common samples for QC")
    
    # Filter to common samples
    clinical_data = clinical_data[clinical_data[clinical_index_col].isin(common_samples)]
    metabolites_data = metabolites_data.loc[:, metabolites_data.columns.isin(common_samples)]
    
    return clinical_data, metabolites_data


# =============================================================================
# Main QC Function
# =============================================================================

def run_final_qc(
    clinical_data: pd.DataFrame,
    metabolites_after: pd.DataFrame,
    metabolites_before: Optional[pd.DataFrame] = None,
    batch_column: str = "batch",
    output_path: str = "reports",
) -> None:
    """
    Run comprehensive quality control analysis on merged and corrected metabolomics data.
    
    This function performs the following steps:
    1. Validates input data (required columns, non-empty)
    2. Cleans and standardizes sample IDs
    3. Validates and cleans batch column
    4. Aligns samples between clinical and metabolomics data
    5. Imputes NaN/inf values for PCA compatibility
    6. Runs inmoose CohortMetric processing
    7. Generates HTML QC report
    
    Args:
        clinical_data: DataFrame containing clinical/sample metadata
            Must contain 'sample_id' and batch_column columns
        metabolites_after: DataFrame with metabolomics data AFTER ComBat correction
            Features as rows, samples as columns
        metabolites_before: DataFrame with metabolomics data BEFORE correction (optional)
            Used for comparative QC analysis
        batch_column: Name of the column in clinical_data containing batch info (default: "batch")
        output_path: Directory to save QC report (default: "reports")
        
    Returns:
        None (QC report is saved to output_path)
        
    Raises:
        ValueError: If required columns are missing from clinical_data
        ValueError: If no overlapping samples exist between clinical and metabolomics data
        ImportError: If inmoose is not installed (INMOOSE_AVAILABLE = False)
    
    Note:
        If inmoose is not installed, this function will log a warning and return early.
        The QC report includes:
        - Batch effect assessment
        - PCA plots (from inmoose)
        - Quality metrics
        - Sample-level diagnostics
    """
    # --- Skip if metabolites_after is empty or None ---
    if metabolites_after.empty or metabolites_after is None:
        logger.warning("Skipping QC: metabolites_after is empty or None.")
        return
    
    # Make copies to avoid modifying inputs
    clinical_data = clinical_data.copy()
    metabolites_after = metabolites_after.copy()
    
    if metabolites_before is not None:
        metabolites_before = metabolites_before.copy()
    
    # --- Step 1: Validate clinical data ---
    required_cols = ["sample_id", batch_column]
    clinical_data = validate_clinical_data(clinical_data, required_cols)
    
    # --- Step 2: Fix sample IDs ---
    clinical_data["sample_id"] = clinical_data["sample_id"].astype(str).str.strip()
    clinical_data = clinical_data.set_index("sample_id")
    
    # Standardize metabolite column names
    metabolites_after.columns = metabolites_after.columns.astype(str).str.strip()
    
    if metabolites_before is not None:
        metabolites_before.columns = metabolites_before.columns.astype(str).str.strip()
    
    # --- Step 3: Batch column cleanup ---
    if batch_column not in clinical_data.columns:
        raise ValueError(f"Batch column '{batch_column}' not found in clinical_data.")
    
    clinical_data[batch_column] = pd.to_numeric(
        clinical_data[batch_column],
        errors="raise"  # Fail explicitly if batch values are not numeric
    )
    
    # --- Step 4: Align samples ---
    clinical_data, metabolites_after = align_samples(
        clinical_data, metabolites_after, "sample_id"
    )
    
    if metabolites_before is not None:
        _, metabolites_before = align_samples(
            clinical_data.reset_index(), metabolites_before, "sample_id"
        )
        metabolites_before = metabolites_before.loc[:, clinical_data.index]
    
    # --- Step 5: NaN + Inf cleanup (CRITICAL for PCA in inmoose) ---
    metabolites_after = _safe_impute(metabolites_after)
    
    if metabolites_before is not None:
        metabolites_before = _safe_impute(metabolites_before)
    
    # --- Step 6: Diagnostics ---
    logger.info(f"Sample count for QC: {clinical_data.shape[0]}")
    logger.info(f"NaNs after imputation: {metabolites_after.isna().sum().sum()}")
    logger.info(f"Batch values: {sorted(clinical_data[batch_column].unique())}")
    
    # --- Step 7: Run inmoose QC (if available) ---
    if not INMOOSE_AVAILABLE:
        logger.warning("Skipping QC report generation (inmoose not available).")
        return
    
    try:
        # Initialize CohortMetric
        # Note: inmoose handles PCA plotting internally
        cohort_qc = CohortMetric(
            clinical_df=clinical_data,
            batch_column=batch_column,
            data_expression_df=metabolites_after,
            data_expression_df_before=metabolites_before,
        )
        cohort_qc.process()
        
        # Generate and save QC report
        qc_report = QCReport(cohort_qc)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        report_path = Path(output_path) / "qc_report.html"
        qc_report.save_report(output_path=str(report_path))
        
        logger.info(f"✓ QC report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"QC failed: {e}")
        logger.error(traceback.format_exc())
        raise
