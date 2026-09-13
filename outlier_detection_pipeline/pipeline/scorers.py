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
  - ae          : Autoencoder reconstruction error (requires torch). Trains an
                  MLP autoencoder on normals; scores by per-sample squared
                  reconstruction error (lower = more anomalous).
  - deep_svdd   : Deep Support Vector Data Description (requires torch). Trains a
                  deep one-class model that maps normals into a minimal-radius
                  hypersphere; scores by squared distance to the hypersphere
                  center (lower = more anomalous).
  - lof         : Local Outlier Factor. Local-density scorer: a point is anomalous
                  if its local density is low relative to its neighbors'. Catches
                  locally-sparse abnormals that global density/boundary methods
                  miss. Lower = more anomalous.
  - knn         : k-th nearest neighbor distance to the training normals. The
                  nonparametric global-distance alternative to Mahalanobis: no
                  Gaussian assumption, no covariance inversion, just raw distance
                  in feature space. Lower = more anomalous.
  - ensemble    : Averages z-normalized scores from a set of sub-scorers (default
                  iforest + ocsvm). Combines methods with different failure
                  modes; often beats either alone. Sub-scorers configured via
                  scorer_kwargs['sub_scorers'] (list of names).

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
    if name == 'ae':
        return AutoencoderScorer(**kwargs)
    if name == 'deep_svdd':
        return DeepSVDDScorer(**kwargs)
    if name == 'lof':
        return LOFScorer(**kwargs)
    if name == 'knn':
        return KNNHostScorer(**kwargs)
    if name == 'ensemble':
        return EnsembleScorer(**kwargs)
    raise ValueError(
        f"Unknown scorer '{name}'. Use one of: iforest, mahalanobis, pca_recon, "
        f"ocsvm, ae, deep_svdd, lof, knn, ensemble."
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
        # Choose a covariance estimator appropriate to the n/d regime.
        #   - MinCovDet (robust MCD) is best when the data is comfortably
        #     overdetermined (n comfortably > d), because it fits on a robust
        #     subset (~half the samples by default) which must itself exceed d
        #     to stay full-rank.
        #   - OAS shrinkage is the principled choice in the n~d regime: it adds
        #     a structured ridge that keeps the covariance well-conditioned and
        #     invertible when samples barely exceed dimensions. This is what
        #     lets Mahalanobis run reliably at, e.g., 181 normals / 100 comps.
        #   - Below n <= d the sample covariance is singular; OAS shrinkage still
        #     works there (it is defined for n < d), so we use it in that case
        #     too rather than falling back to plain empirical covariance.
        n, d = X.shape
        from sklearn.covariance import OAS
        robust_threshold = max(2 * d + 10, int(3 * d))
        if n > robust_threshold:
            # Comfortably overdetermined: robust MCD resists borderline normals.
            self.mcd = MinCovDet(
                support_fraction=self.support_fraction,
                random_state=self.random_state,
            )
            est_name = "MinCovDet (robust)"
        else:
            # n~d or n<d: OAS shrinkage keeps the estimate full-rank and stable.
            self.mcd = OAS()
            est_name = f"OAS shrinkage (n={n}, d={d})"
        logger.info(f"MahalanobisScorer: covariance estimator = {est_name}")
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


# ---------------------------------------------------------------------------
# Deep one-class scorers (optional; require torch).
# torch is imported lazily inside each class so the pipeline stays importable
# without it; selecting one of these scorers without torch installed raises a
# clear ImportError telling the user to `pip install torch`.
# ---------------------------------------------------------------------------


class AutoencoderScorer:
    """MLP autoencoder reconstruction-error scorer (requires torch).

    Trains a symmetric MLP autoencoder on the normal samples; scores each
    sample by its squared reconstruction error (negated so LOWER = more
    anomalous, matching the other scorers). The network is intentionally small
    (one hidden layer with ReLU) because the dataset is small (~180 normals)
    and a larger net would overfit the training normals and inflate the FPR on
    unseen normals -- the same failure mode as high n_components for IF.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        latent_dim: int = 8,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
        contamination: Any = "auto",
        random_state: int = 42,
        **_ignored: Any,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.contamination = contamination
        self.random_state = random_state
        self._net = None
        self._offset_: float = 0.0

    @staticmethod
    def _require_torch():
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "AutoencoderScorer requires PyTorch. Install it with "
                "`pip install torch`."
            ) from e

    def fit(self, X: np.ndarray) -> "AutoencoderScorer":
        self._require_torch()
        import torch
        from torch import nn, optim

        torch.manual_seed(self.random_state)
        Xf = np.asarray(X, dtype=np.float32)
        n, d = Xf.shape
        hd = max(1, min(self.hidden_dim, d))
        ld = max(1, min(self.latent_dim, hd))

        class _AE(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Sequential(nn.Linear(d, hd), nn.ReLU(), nn.Linear(hd, ld))
                self.dec = nn.Sequential(nn.Linear(ld, hd), nn.ReLU(), nn.Linear(hd, d))

            def forward(self, x):
                return self.dec(self.enc(x))

        self._net = _AE()
        opt = optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        Xt = torch.from_numpy(Xf)
        bs = max(1, min(self.batch_size, n))
        self._net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb = Xt[idx]
                opt.zero_grad()
                recon = self._net(xb)
                loss = loss_fn(recon, xb)
                loss.backward()
                opt.step()
        self._net.eval()
        # Offset for decision_function: high percentile of training recon error.
        train_err = self._recon_error(Xf)
        try:
            contam = float(self.contamination)
        except (TypeError, ValueError):
            contam = 0.02
        self._offset_ = float(np.quantile(train_err, 1.0 - contam))
        return self

    def _recon_error(self, X: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
            recon = self._net(Xt)
            err = torch.sum((Xt - recon) ** 2, dim=1).numpy()
        return err

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -self._recon_error(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.score_samples(X) + self._offset_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) < 0, -1, 1)


class DeepSVDDScorer:
    """Deep Support Vector Data Description (requires torch).

    Trains a deep one-class model that maps normal samples into a
    minimal-radius hypersphere (Ruff et al., Deep SVDD, ICML 2018). The model
    is a small MLP; the objective minimizes the squared distance of the
    training normals to a single center c, whose radius defines the normal
    region. Scores are the squared distance to c (negated so LOWER = more
    anomalous).

    No explicit hypersphere radius is stored; the pipeline's OOF percentile
    threshold calibration supplies the operating point, exactly as for the
    other scorers.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        latent_dim: int = 8,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
        contamination: Any = "auto",
        random_state: int = 42,
        **_ignored: Any,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.contamination = contamination
        self.random_state = random_state
        self._net = None
        self._c = None
        self._offset_: float = 0.0

    @staticmethod
    def _require_torch():
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "DeepSVDDScorer requires PyTorch. Install it with "
                "`pip install torch`."
            ) from e

    def fit(self, X: np.ndarray) -> "DeepSVDDScorer":
        self._require_torch()
        import torch
        from torch import nn, optim

        torch.manual_seed(self.random_state)
        Xf = np.asarray(X, dtype=np.float32)
        n, d = Xf.shape
        hd = max(1, min(self.hidden_dim, d))
        ld = max(1, min(self.latent_dim, hd))

        class _SVDD(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d, hd), nn.ReLU(), nn.Linear(hd, ld), nn.ReLU(),
                )
                # Zero-bias output: avoids the trivial collapse to a constant
                # map (standard Deep SVDD practice).
                for m in self.net:
                    if isinstance(m, nn.Linear):
                        nn.init.zeros_(m.bias)

            def forward(self, x):
                return self.net(x)

        self._net = _SVDD()
        # Initialize c as the mean latent of the normals, then freeze it.
        self._net.eval()
        with torch.no_grad():
            self._c = self._net(torch.from_numpy(Xf)).mean(dim=0)
        opt = optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=1e-4)
        Xt = torch.from_numpy(Xf)
        bs = max(1, min(self.batch_size, n))
        self._net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb = Xt[idx]
                opt.zero_grad()
                z = self._net(xb)
                loss = torch.mean(torch.sum((z - self._c) ** 2, dim=1))
                loss.backward()
                opt.step()
        self._net.eval()
        # Offset for decision_function.
        train_dist = self._squared_dist(Xf)
        try:
            contam = float(self.contamination)
        except (TypeError, ValueError):
            contam = 0.02
        self._offset_ = float(np.quantile(train_dist, 1.0 - contam))
        return self

    def _squared_dist(self, X: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
            z = self._net(Xt)
            dist = torch.sum((z - self._c) ** 2, dim=1).numpy()
        return dist

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -self._squared_dist(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.score_samples(X) + self._offset_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) < 0, -1, 1)


class LOFScorer:
    """Local Outlier Factor scorer.

    Scores each point by how locally sparse it is relative to its neighbors:
    a point whose neighborhood is sparser than its neighbors' neighborhoods is
    anomalous. This is the canonical *local-density* mechanism, which global
    density/boundary methods (IF, Mahalanobis, OCSVM) do not capture: it can
    flag an abnormal that sits in a sparse region even when it is not globally
    far from the normal centroid. Catches heterogeneous abnormals (different
    IMDs shifting different metabolite subsets).

    Trained on normals only. Uses novelty=True so unseen samples can be scored
    (sklearn's LOF otherwise only scores training points). The negative
    LOF score is returned so LOWER = more anomalous (high LOF = anomalous).

    k (n_neighbors) is the key lever: small k = sensitive to local noise,
    large k = smoother, more global. Defaults to max(20, 2*sqrt(n)).
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: Any = "auto",
        **_ignored: Any,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = None
        self._offset_: float = 0.0

    def fit(self, X: np.ndarray) -> "LOFScorer":
        from sklearn.neighbors import LocalOutlierFactor
        n = X.shape[0]
        k = max(2, min(self.n_neighbors, n - 1))
        self.model = LocalOutlierFactor(
            n_neighbors=k,
            novelty=True,
            contamination="auto",
        )
        self.model.fit(X)
        # Offset: high percentile of training-normal negative-LOF scores.
        train_scores = self.model.score_samples(X)
        try:
            contam = float(self.contamination)
        except (TypeError, ValueError):
            contam = 0.02
        self._offset_ = float(np.quantile(train_scores, contam))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        # sklearn score_samples returns negative LOF: lower = more anomalous.
        return self.model.score_samples(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.score_samples(X) - self._offset_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) < 0, -1, 1)


class KNNHostScorer:
    """k-th nearest neighbor distance scorer.

    Scores each point by its distance to the k-th nearest training normal -- the
    nonparametric global-distance alternative to Mahalanobis. No Gaussian
    assumption, no covariance inversion (so no shrinkage-instability), just raw
    distance in feature space. More stable than Mahalanobis at small n (nothing
    to estimate beyond the reference set itself) while still measuring global
    isolation from the normal cloud.

    The score is negated so LOWER = more anomalous. The distance is computed to
    the training normals only (the reference set); abnormals are never in the
    reference, so this is a clean one-class distance.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        contamination: Any = "auto",
        **_ignored: Any,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self._ref = None
        self._offset_: float = 0.0

    def fit(self, X: np.ndarray) -> "KNNHostScorer":
        from sklearn.neighbors import NearestNeighbors
        n = X.shape[0]
        k = max(1, min(self.n_neighbors, n - 1))
        self._nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        self._nn.fit(X)
        self._ref = X
        # Offset: high percentile of training-normal k-th NN distances.
        train_dists, _ = self._nn.kneighbors(X)
        kth = train_dists[:, -1]
        try:
            contam = float(self.contamination)
        except (TypeError, ValueError):
            contam = 0.02
        self._offset_ = float(np.quantile(kth, 1.0 - contam))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        dists, _ = self._nn.kneighbors(X)
        kth = dists[:, -1]
        return -kth

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.score_samples(X) + self._offset_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) < 0, -1, 1)


class EnsembleScorer:
    """Ensemble of one-class scorers via averaged z-normalized scores.

    Combines scorers with different failure modes (e.g. IF global density +
    OCSVM boundary) by averaging their z-normalized score_samples. Each
    sub-scorer's raw scores are standardized to zero mean / unit std on the
    training normals, so the ensemble is not dominated by whichever sub-scorer
    has the largest score scale. The averaged z-score is returned so
    LOWER = more anomalous, matching the convention.

    This often beats any single member when the members err in different
    directions (e.g. one over-flags normals, another misses abnormals): the
    average cancels uncorrelated errors.
    """

    def __init__(
        self,
        sub_scorers: Optional[list] = None,
        contamination: Any = "auto",
        random_state: int = 42,
        **scorer_kwargs: Any,
    ) -> None:
        self.sub_scorer_names = sub_scorers or ["iforest", "ocsvm"]
        self.contamination = contamination
        self.random_state = random_state
        self._scorers = []
        self._means = []
        self._stds = []
        self._offset_: float = 0.0
        # Stash extra kwargs to forward to each sub-scorer.
        self._scorer_kwargs = scorer_kwargs

    def fit(self, X: np.ndarray) -> "EnsembleScorer":
        self._scorers = []
        self._means = []
        self._stds = []
        for i, name in enumerate(self.sub_scorer_names):
            sub = make_scorer(
                name,
                random_state=self.random_state + i,
                contamination=self.contamination,
                **self._scorer_kwargs,
            )
            sub.fit(X)
            self._scorers.append(sub)
            raw = sub.score_samples(X)
            self._means.append(float(np.mean(raw)))
            self._stds.append(float(np.std(raw)) + 1e-12)
        # Offset: low percentile of training-normal ensemble scores.
        ens = self.score_samples(X)
        try:
            contam = float(self.contamination)
        except (TypeError, ValueError):
            contam = 0.02
        self._offset_ = float(np.quantile(ens, contam))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        z = np.zeros(X.shape[0], dtype=float)
        for sub, m, s in zip(self._scorers, self._means, self._stds):
            raw = sub.score_samples(X)
            z += (raw - m) / s
        return z / len(self._scorers)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.score_samples(X) - self._offset_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) < 0, -1, 1)
