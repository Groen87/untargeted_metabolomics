"""Configurable feature scaling for the outlier detection pipeline.

The scaler is selected via the `scaler` config key:
  - robust   : RobustScaler (median/IQR, robust to outliers) -- default
  - standard : StandardScaler (mean/std)
  - none     : no scaling; the data is passed through unchanged

In all cases the scaler is fit on the training NORMALS only and every
other sample is transformed with that fitted scaler (pure one-class
design). With 'none' the model runs directly on the input scale, e.g.
log10-transformed but unscaled data.
"""

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.preprocessing import FunctionTransformer


def make_scaler(name: str = "robust"):
    """Build the scaler object requested by the `scaler` config key.

    Args:
        name: 'robust' (default), 'standard', or 'none'. Unknown names
            fall back to 'robust' with a logged note.

    Returns:
        An unfitted sklearn-style transformer with fit/transform. 'none'
        returns an identity passthrough (FunctionTransformer) so all
        fit/transform call sites work unchanged.
    """
    import logging
    logger = logging.getLogger(__name__)

    if name == "none":
        logger.info("Scaler 'none': features are used unscaled (identity passthrough).")
        return FunctionTransformer()
    if name == "standard":
        logger.info("Using StandardScaler (mean/std) for feature scaling.")
        return StandardScaler()
    if name != "robust":
        logger.warning(f"Unknown scaler '{name}'; falling back to RobustScaler.")
    return RobustScaler()
