"""Lab-protocol ground-truth role masks for the outlier-detection pipeline.

Roles (lab protocol):

  - true outlier  = Classification 1 AND Oordeel targeted 1 AND Non-treated 1
    (a non-treated IMD sample).
  - Classification 1 AND Oordeel targeted 1 AND Non-treated 0 (or NaN) is a
    TREATED IMD sample: gray. It is neither a definite outlier nor a definite
    inlier, so it never enters the headline TP/FN totals.
  - true inlier   = Classification 0 AND Oordeel targeted 0, regardless of
    Non-treated.
  - Every other sample with non-NaN Classification AND Oordeel targeted is
    gray. Samples with a NaN in either column have an undefined role.

The Non-treated column is only relevant for the (Classification 1,
Oordeel 1) group. When the input data has no Non-treated column, the legacy
definition (Classification 1 AND Oordeel 1 = true outlier) is used so older
inputs keep working.
"""

import pandas as pd


def lab_protocol_role_masks(gm: pd.DataFrame) -> "tuple[pd.Series, pd.Series]":
    """Boolean (is_true_outlier, is_true_inlier) masks for a group map.

    Args:
        gm: DataFrame indexed by sample_id with columns 'raw_classification'
            and 'oordeel'; optionally a 'non_treated' column. Values may be
            non-numeric (they are coerced, invalid -> NaN).

    Returns:
        (is_true_outlier, is_true_inlier) boolean Series aligned to gm.
        Samples with a NaN Classification or Oordeel are False in both.
    """
    rc = pd.to_numeric(gm['raw_classification'], errors='coerce')
    oo = pd.to_numeric(gm['oordeel'], errors='coerce')
    labelled = rc.notna() & oo.notna()
    imd = labelled & (rc == 1) & (oo == 1)
    if 'non_treated' in gm.columns:
        nt = pd.to_numeric(gm['non_treated'], errors='coerce')
        is_true_outlier = imd & (nt == 1)
    else:
        is_true_outlier = imd
    is_true_inlier = labelled & (rc == 0) & (oo == 0)
    return is_true_outlier, is_true_inlier
