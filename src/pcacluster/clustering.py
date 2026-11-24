"""Clustering algorithms with a common interface and estimation of K."""

from collections.abc import Sequence
from typing import Protocol, Self

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


class ClusterAlgo(Protocol):
    name: str

    def fit(self, Z: np.ndarray, K: int) -> Self: ...

    def predict(self, Z: np.ndarray) -> np.ndarray: ...


class KMeansAlgo:
    """K-means with several initializations (n_init >= 5 recommended)."""

    name = "kmeans"

    def __init__(self, n_init: int = 5, seed: int = 0):
        self.n_init = n_init
        self.seed = seed
        self.model_: KMeans | None = None

    def fit(self, Z: np.ndarray, K: int) -> Self:
        self.model_ = KMeans(n_clusters=K, n_init=self.n_init, random_state=self.seed)
        self.model_.fit(Z)
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before predict.")
        return self.model_.predict(Z)

    def bic(self, Z: np.ndarray) -> float:
        """BIC assuming spherical Gaussian clusters with one shared variance
        (Pelleg and Moore, 2000, X-means). Lower is better.

        BIC = -2 log L + m ln(n), with m = (K - 1) + K d + 1 free parameters:
        mixing weights, centers and the shared variance sigma^2, estimated as
        sigma^2 = inertia / ((n - K) d).
        """
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before bic.")
        n, d = Z.shape
        k = self.model_.n_clusters
        if n <= k:
            return float("inf")

        labels = self.model_.predict(Z)
        inertia = -self.model_.score(Z)
        sigma2 = max(inertia / ((n - k) * d), 1e-10)

        cluster_sizes = np.bincount(labels, minlength=k)
        weights = cluster_sizes / n
        ent = float(np.sum(cluster_sizes * np.log(np.maximum(weights, 1e-12))))
        ll = -0.5 * n * d * np.log(2.0 * np.pi * sigma2) - 0.5 * (n - k) * d + ent

        m = (k - 1) + k * d + 1
        return float(-2.0 * ll + m * np.log(n))


class GMMAlgo:
    """Gaussian mixture with diagonal covariance."""

    name = "gmm"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.model_: GaussianMixture | None = None

    def fit(self, Z: np.ndarray, K: int) -> Self:
        self.model_ = GaussianMixture(
            n_components=K, covariance_type="diag", random_state=self.seed
        )
        self.model_.fit(Z)
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before predict.")
        return self.model_.predict(Z)

    def bic(self, Z: np.ndarray) -> float:
        """BIC of the fitted mixture (scikit-learn). Lower is better."""
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before bic.")
        return float(self.model_.bic(Z))


ALGOS: dict[str, type[KMeansAlgo] | type[GMMAlgo]] = {
    "kmeans": KMeansAlgo,
    "gmm": GMMAlgo,
}


def estimate_K(
    Z: np.ndarray,
    algo_name: str,
    K_grid: Sequence[int],
    criterion: str = "bic",
    seed: int = 0,
) -> int:
    """Fit algo_name for each K in K_grid and pick K by criterion.

    algo_name: 'kmeans' or 'gmm'.
    criterion: 'bic' (minimized) or 'silhouette' (maximized, needs K >= 2).
    """
    if not K_grid:
        raise ValueError("K_grid must not be empty.")
    if algo_name not in ALGOS:
        raise ValueError(f"Unknown algo_name: {algo_name!r}")
    if criterion not in ("bic", "silhouette"):
        raise ValueError(f"Unknown criterion: {criterion!r}")
    if criterion == "silhouette" and min(K_grid) < 2:
        raise ValueError(
            f"criterion='silhouette' requires K >= 2 for all K in K_grid, got min K={min(K_grid)}."
        )

    scores: dict[int, float] = {}
    for k in K_grid:
        model = ALGOS[algo_name](seed=seed).fit(Z, k)
        if criterion == "bic":
            # Negated so that the best K is always the maximum.
            scores[k] = -model.bic(Z)
        else:
            labels = model.predict(Z)
            # A model can leave components empty; silhouette needs 2 labels.
            if len(np.unique(labels)) < 2:
                scores[k] = -1.0
            else:
                scores[k] = float(silhouette_score(Z, labels))

    return int(max(scores, key=lambda k: scores[k]))


def fit_original_clustering(
    X_train: np.ndarray, algo_name: str, K: int, seed: int = 0
) -> ClusterAlgo:
    """Fit clustering on the full (original) dimensional space."""
    if algo_name not in ALGOS:
        raise ValueError(f"Unknown algo_name: {algo_name!r}")
    return ALGOS[algo_name](seed=seed).fit(X_train, K)
