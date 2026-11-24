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


def _kmeans_bic(kmeans: KMeans, Z: np.ndarray) -> float:
    """Estimate BIC for a fitted KMeans model assuming spherical Gaussian clusters
    (Pelleg & Moore, 2000, X-means).

    BIC = -2 * log_likelihood + m * ln(n)
    where m is the number of free parameters:
    m = (K - 1) + K * d + 1
    (cluster probabilities, cluster centers, and shared variance sigma^2).
    """
    n, d = Z.shape
    k = kmeans.n_clusters
    if n <= k:
        return float("inf")

    inertia = kmeans.inertia_
    sigma2 = max(inertia / ((n - k) * d), 1e-10)

    cluster_sizes = np.bincount(kmeans.labels_, minlength=k)
    weights = cluster_sizes / n

    ent = float(np.sum(cluster_sizes * np.log(np.maximum(weights, 1e-12))))
    ll = -0.5 * n * d * np.log(2.0 * np.pi * sigma2) - 0.5 * (n - k) * d + ent

    m = (k - 1) + k * d + 1
    return float(-2.0 * ll + m * np.log(n))


def estimate_K(
    Z: np.ndarray,
    algo_name: str,
    K_grid: Sequence[int],
    criterion: str = "bic",
    seed: int = 0,
) -> int:
    """Fit algo_name for each K in K_grid and pick K by criterion.

    algo_name: 'kmeans' or 'gmm'.
    criterion: 'bic' or 'silhouette'.
    """
    if not K_grid:
        raise ValueError("K_grid must not be empty.")
    if algo_name not in ("kmeans", "gmm"):
        raise ValueError(f"Unknown algo_name: {algo_name!r}")
    if criterion not in ("bic", "silhouette"):
        raise ValueError(f"Unknown criterion: {criterion!r}")

    if criterion == "silhouette":
        min_k = min(K_grid)
        if min_k < 2:
            raise ValueError(
                f"criterion='silhouette' requires K >= 2 for all K in K_grid, got min K={min_k}."
            )

    scores: dict[int, float] = {}

    for k in K_grid:
        if algo_name == "kmeans":
            model = KMeansAlgo(n_init=5, seed=seed)
            model.fit(Z, k)
            if criterion == "silhouette":
                labels = model.predict(Z)
                scores[k] = float(silhouette_score(Z, labels))
            elif criterion == "bic":
                assert model.model_ is not None
                scores[k] = _kmeans_bic(model.model_, Z)
            else:
                raise ValueError(f"Unknown criterion: {criterion!r}")
        elif algo_name == "gmm":
            model = GMMAlgo(seed=seed)
            model.fit(Z, k)
            if criterion == "silhouette":
                labels = model.predict(Z)
                if len(np.unique(labels)) < 2:
                    scores[k] = -1.0
                else:
                    scores[k] = float(silhouette_score(Z, labels))
            elif criterion == "bic":
                assert model.model_ is not None
                scores[k] = float(model.model_.bic(Z))
            else:
                raise ValueError(f"Unknown criterion: {criterion!r}")
        else:
            raise ValueError(f"Unknown algo_name: {algo_name!r}")

    if criterion == "silhouette":
        best_k = max(scores, key=lambda k: scores[k])
    else:
        best_k = min(scores, key=lambda k: scores[k])

    return int(best_k)


def fit_original_clustering(
    X_train: np.ndarray, algo_name: str, K: int, seed: int = 0
) -> ClusterAlgo:
    """Fit clustering on the full (original) dimensional space."""
    if algo_name == "kmeans":
        model: ClusterAlgo = KMeansAlgo(seed=seed)
    elif algo_name == "gmm":
        model = GMMAlgo(seed=seed)
    else:
        raise ValueError(f"Unknown algo_name: {algo_name!r}")

    model.fit(X_train, K)
    return model
