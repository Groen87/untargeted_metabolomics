"""
Pluggable one-class anomaly scorers.

Each scorer implements the same minimal interface as sklearn's
IsolationForest for the methods this pipeline relies on:

    fit(X)             -> self          (train on NORMAL samples only)
    score_samples(X)   -> ndarray       (raw anomaly score, LOWER = more anomalous)
    decision_function(X) -> ndarray    (shifted anomaly score, LOWER = more anomalous)

The "lower = more anomalous" convention matches sklearn's IsolationForest, so
the existing threshold calibration (percentile of score_samples), drift
diagnostic, and realistic evaluation all work unchanged across scorers.

Available scorers (selected via the `scorer` config key):
  - iforest     : sklearn IsolationForest (the original pipeline default)
  - mahalanobis : Mahalanobis distance on a robust covariance estimate of the
                  PCA-reduced features. Parametric, sample-efficient, with a
                  chi-square-derived decision_function. More stable threshold
                  calibration at small n than the tree ensemble.
  - pca_recon   : PCA reconstruction error. Fits a second PCA on the (already
                  PCA-reduced) normal features and scores by reconstruction
                  error. Lower = more anomalous.
  - ocsvm       : One-Class SVM (RBF). Nonparametric boundary; decision_function
                  is the signed distance to the boundary (lower = more anomalous).

A factory `make_scorer(name, **kwargs)` returns a configured scorer instance.
"""

from typing import Any, Optional

import numpy as np
import logging

from sklearn.ensemble import IsolationForest
from sklearn.covariance import MinCovDet
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def make_scorer(name: str, **kwargs: Any) -> Any:
    """Build a scorer instance by name.

    Args:
        name: One of 'iforest', 'mahalanobis', 'pca_recon', 'ocsvm'.
        **kwargs: Passed through to the scorer (e.g. n_estimators, nu, gamma).

    Returns:
        A scorer object with fit / score_samples / decision_function.
    """
    name = (name or 'iforest').lower()
    if name == 'iforest':
        return IForestScorer(**kwargs)
    if name == 'mahalanobis':
        return MahalanobisScorer(**kwargs)
    if name == 'pca_recon':
        return PCAReconScorer(**kwargs)
    if name == 'ocsvm':
        return OCSVMScorer(**kwargs)
    raise ValueError(
        f"Unknown scorer '{name}'. Use one of: iforest, mahalanobis, pca_recon, ocsvm."
    )


class IForestScorer:
    """IsolationForest scorer (wraps sklearn IsolationForest)."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: Any = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        n_jobs: int = -1,
        random_state: int = 42,
        contamination: Any = "auto",
        **_ignored: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.contamination = contamination
        self.model: Optional[IsolationForest] = None

    def fit(self, X: np.ndarray) -> "IForestScorer":
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            contamination=self.contamination,
        )
        self.model.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self.model.score_samples(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.model.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class MahalanobisScorer:
    """Mahalanobis distance scorer using a robust covariance estimate.

    Score convention (lower = more anomalous, matching IsolationForest):
        score_samples    = -(Mahalanobis distance)
        decision_function = score_samples (no separate offset needed; the
        threshold is a percentile of the normal-training score distribution,
        which is what the pipeline already does).

    The robust covariance (MinCovDet) is used so a few borderline normals do
    not dominate the covariance estimate. This is the sample-efficient
    parametric alternative: a single mean + covariance fit instead of an
    ensemble of trees, with a chi-square-derived decision_function.
    """

    def __init__(
        self,
        contamination: Any = "auto",
        random_state: int = 42,
        support_fraction: Optional[float] = None,
        **_ignored: Any,
    ) -> None:
        self.random_state = random_state
        self.support_fraction = support_fraction
        # 'auto' -> use a 2nd-percentile threshold via the pipeline; we still
        # expose a contamination-derived offset for the standard-eval path.
        self.contamination = contamination
        self.mcd: Optional[MinCovDet] = None
        self._offset_: float = 0.0

    def fit(self, X: np.ndarray) -> "MahalanobisScorer":
        # MinCovDet needs more samples than features for a stable estimate.
        n, d = X.shape
        if n <= d:
            # Fall back to a shrunk empirical covariance when too few samples.
            from sklearn.covariance import EmpiricalCovariance
            logger.warning(
                f"MahalanobisScorer: n_samples ({n}) <= n_features ({d}); "
                f"falling back to empirical covariance (less robust)."
            )
            self.mcd = EmpiricalCovariance()
            self.mcd.fit(X)
        else:
            self.mcd = MinCovDet(
                support_fraction=self.support_fraction,
                random_state=self.random_state,
            )
            self.mcd.fit(X)
        # Offset for decision_function: chi-square percentile matching the
        # configured contamination (only meaningful when contamination is a float).
        from scipy.stats import chi2
        try:
            contam = float(self.contamination)
        except (TypeError, ValueError):
            contam = 0.02
        self._offset_ = float(chi2.ppf(1.0 - contam, df=d))
        return self

    def _mahalanobis(self, X: np.ndarray) -> np.ndarray:
        # MinCovDet/EmpiricalCovariance expose mahalanobis() -> squared distance.
        return self.mcd.mahalanobis(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        # Squared Mahalanobis distance; negate so LOWER = more anomalous.
        return -self._mahalanobis(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        # Shift by the chi-square offset so the 0 boundary ~ contamination.
        return self.score_samples(X) + self._offset_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) < 0, -1, 1)


class PCAReconScorer:
    """PCA reconstruction-error scorer.

    Fits a PCA on the (already PCA-reduced) normal features and scores by the
    reconstruction error (squared L2 distance to the PCA subspace). Higher
    error = more anomalous; negated to match the lower-more-anomalous
    convention.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        random_state: int = 42,
        contamination: Any = "auto",
        **_ignored: Any,
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.contamination = contamination
        self.pca: Optional[PCA] = None
        self._offset_: float = 0.0

    def fit(self, X: np.ndarray) -> "PCAReconScorer":
        n, d = X.shape
        nc = self.n_components if self.n_components is not None else max(1, min(d - 1, n - 1))
        nc = max(1, min(int(nc), d - 1, n - 1))
        self.pca = PCA(n_components=nc, random_state=self.random_state)
        self.pca.fit(X)
        # Offset: a high percentile of the training-normal reconstruction error,
        # used only to give decision_function a 0-boundary for the standard path.
        train_err = self._recon_error(X)
        try:
            contam = float(self.contamination)
        except (TypeError, ValueError):
            contam = 0.02
        self._offset_ = float(np.quantile(train_err, 1.0 - contam))
        return self

    def _recon_error(self, X: np.ndarray) -> np.ndarray:
        Xp = self.pca.transform(X)
        Xr = self.pca.inverse_transform(Xp)
        return np.sum((X - Xr) ** 2, axis=1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -self._recon_error(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.score_samples(X) + self._offset_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) < 0, -1, 1)


class OCSVMScorer:
    """One-Class SVM scorer (RBF kernel).

    Wraps sklearn OneClassSVM. decision_function is the signed distance to the
    boundary (lower = more anomalous); score_samples mirrors it (sklearn's OCSVM
    score_samples == decision_function). Trained on normals only.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.05,
        gamma: Any = "scale",
        contamination: Any = "auto",
        **_ignored: Any,
    ) -> None:
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.contamination = contamination
        self.model: Optional[OneClassSVM] = None

    def fit(self, X: np.ndarray) -> "OCSVMScorer":
        self.model = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma,
        )
        self.model.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self.model.score_samples(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.model.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
