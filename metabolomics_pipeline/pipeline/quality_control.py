from inmoose.cohort_qc.cohort_metric import CohortMetric
from inmoose.cohort_qc.qc_report import QCReport

from typing import Optional
import pandas as pd
from pathlib import Path
import logging
import numpy as np
import traceback

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _safe_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    NaN-safe imputation for metabolomics PCA compatibility.
    """
    df = df.copy()

    df = df.apply(pd.to_numeric, errors="coerce")

    min_pos = df[df > 0].min().min()
    fill_value = (min_pos / 2) if pd.notna(min_pos) else 1e-10

    df = df.fillna(fill_value)
    df = df.replace([np.inf, -np.inf], fill_value)

    return df


# ------------------------------------------------------------------------------
# Main QC function
# ------------------------------------------------------------------------------

def run_final_qc(
    clinical_data: pd.DataFrame,
    metabolites_after: pd.DataFrame,
    metabolites_before: Optional[pd.DataFrame] = None,
    batch_column: str = "batch",
    output_path: str = "reports",
) -> None:

    # ------------------------------------------------------------------
    # Copy inputs
    # ------------------------------------------------------------------
    clinical_data = clinical_data.copy()
    metabolites_after = metabolites_after.copy()

    if metabolites_before is not None:
        metabolites_before = metabolites_before.copy()

    # ------------------------------------------------------------------
    # 1. Fix sample IDs FIRST
    # ------------------------------------------------------------------
    if "sample_id" not in clinical_data.columns:
        raise ValueError("clinical_data must contain 'sample_id' column.")

    clinical_data["sample_id"] = clinical_data["sample_id"].astype(str).str.strip()
    clinical_data = clinical_data.set_index("sample_id")

    metabolites_after.columns = metabolites_after.columns.astype(str).str.strip()

    if metabolites_before is not None:
        metabolites_before.columns = metabolites_before.columns.astype(str).str.strip()

    # ------------------------------------------------------------------
    # 2. Batch column cleanup
    # ------------------------------------------------------------------
    clinical_data[batch_column] = pd.to_numeric(
        clinical_data[batch_column],
        errors="raise"
    )

    # ------------------------------------------------------------------
    # 3. Align samples
    # ------------------------------------------------------------------
    common_samples = metabolites_after.columns.intersection(clinical_data.index)

    logger.info(f"Found {len(common_samples)} common samples")

    if len(common_samples) == 0:
        raise ValueError("No overlapping samples between clinical and metabolomics data.")

    clinical_data = clinical_data.loc[common_samples]
    metabolites_after = metabolites_after.loc[:, common_samples]

    if metabolites_before is not None:
        metabolites_before = metabolites_before.loc[:, common_samples]

    # ------------------------------------------------------------------
    # 4. NaN + Inf cleanup (CRITICAL for PCA in inmoose)
    # ------------------------------------------------------------------
    metabolites_after = _safe_impute(metabolites_after)

    if metabolites_before is not None:
        metabolites_before = _safe_impute(metabolites_before)

    # ------------------------------------------------------------------
    # 5. Diagnostics
    # ------------------------------------------------------------------
    logger.info(f"sample count: {clinical_data.shape[0]}")
    logger.info(f"NaNs after: {metabolites_after.isna().sum().sum()}")

    logger.info("clinical samples (first 5):")
    logger.info(clinical_data.index[:5].tolist())

    logger.info("metabolite samples (first 5):")
    logger.info(metabolites_after.columns[:5].tolist())

    logger.info(f"batch values: {clinical_data[batch_column].unique()}")

    # ------------------------------------------------------------------
    # 6. Run inmoose QC
    # ------------------------------------------------------------------
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

        qc_report.save_report(output_path=str(Path(output_path) / "qc_report.html"))

        logger.info(f"✓ QC report saved to {output_path}")

    except Exception as e:
        logger.error(f"QC failed: {e}")
        logger.error(traceback.format_exc())
        raise