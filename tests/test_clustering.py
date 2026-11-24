"""Tests for clustering models and K estimation."""

import numpy as np
import pytest
from sklearn.decomposition import PCA

from pcacluster.clustering import (
    GMMAlgo,
    KMeansAlgo,
    estimate_K,
    fit_original_clustering,
    load_clustering,
    save_clustering,
)
from pcacluster.data import load_dataset
from pcacluster.metrics import ari


def test_kmeans_cluster_fit_predict():
    X, _ = load_dataset("synthetic", n=150, p=10, n_clusters=3, seed=42)
    model = KMeansAlgo(n_init=5, seed=42)
    fitted = model.fit(X, K=3)
    assert fitted is model

    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (150,)
    assert set(np.unique(preds)).issubset({0, 1, 2})


def test_kmeans_cluster_reproducibility():
    X, _ = load_dataset("synthetic", n=100, p=10, n_clusters=3, seed=42)
    model1 = KMeansAlgo(n_init=5, seed=123).fit(X, K=3)
    model2 = KMeansAlgo(n_init=5, seed=123).fit(X, K=3)
    np.testing.assert_array_equal(model1.predict(X), model2.predict(X))


def test_gmm_cluster_fit_predict():
    X, _ = load_dataset("synthetic", n=150, p=10, n_clusters=3, seed=42)
    model = GMMAlgo(seed=42)
    fitted = model.fit(X, K=3)
    assert fitted is model

    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (150,)
    assert set(np.unique(preds)).issubset({0, 1, 2})


def test_gmm_cluster_reproducibility():
    X, _ = load_dataset("synthetic", n=100, p=10, n_clusters=3, seed=42)
    model1 = GMMAlgo(seed=123).fit(X, K=3)
    model2 = GMMAlgo(seed=123).fit(X, K=3)
    np.testing.assert_array_equal(model1.predict(X), model2.predict(X))


def test_estimate_k_silhouette():
    # Well-separated 3 clusters in 10 dimensions
    X, y = load_dataset("synthetic", n=300, p=10, n_clusters=3, separation=10.0, seed=42)
    best_k_km = estimate_K(X, "kmeans", range(2, 6), criterion="silhouette", seed=42)
    assert best_k_km == 3

    best_k_gmm = estimate_K(X, "gmm", range(2, 6), criterion="silhouette", seed=42)
    assert best_k_gmm == 3


def test_estimate_k_silhouette_subsample():
    X, _ = load_dataset("synthetic", n=600, p=10, n_clusters=3, separation=10.0, seed=42)
    best_k = estimate_K(X, "kmeans", range(2, 6), criterion="silhouette", sample_size=200)
    assert best_k == 3


def test_estimate_k_bic():
    # Well-separated 3 clusters in 10 dimensions
    X, y = load_dataset("synthetic", n=300, p=10, n_clusters=3, separation=10.0, seed=42)
    best_k_gmm = estimate_K(X, "gmm", range(2, 6), criterion="bic", seed=42)
    assert best_k_gmm == 3

    best_k_km = estimate_K(X, "kmeans", range(2, 6), criterion="bic", seed=42)
    assert best_k_km == 3


# Synthetic data with K = 5 (signal in K - 1 = 4 dimensions), n = 500, p = 30.
# Expected: K is underestimated with d < K - 1 and recovered with d >= K - 1.
# GMM (one EM run) misses K at d = p for some seeds, so it is checked only up
# to d = 10. Full validation (K = 10) in docs/DECISOES.md, issue #3.
@pytest.mark.parametrize("criterion", ["bic", "silhouette"])
@pytest.mark.parametrize(
    "algo_name, d_grid",
    [("kmeans", (1, 2, 4, 10, 30)), ("gmm", (1, 2, 4, 10))],
)
def test_estimate_k_on_synthetic_by_d(algo_name, d_grid, criterion):
    K = 5
    for seed in (0, 1, 2):
        X, _ = load_dataset("synthetic", n=500, p=30, n_clusters=K, seed=seed)
        X = X - X.mean(axis=0)
        for d in d_grid:
            Z = PCA(n_components=d, random_state=seed).fit_transform(X)
            est = estimate_K(Z, algo_name, range(2, 9), criterion=criterion, seed=seed)
            if d < K - 1:
                assert est < K, f"seed={seed}, d={d}: K_est={est}"
            else:
                assert est == K, f"seed={seed}, d={d}: K_est={est}"


def test_estimate_k_invalid_criterion_or_k():
    X = np.random.default_rng(0).standard_normal((50, 5))
    with pytest.raises(ValueError, match="criterion='silhouette' requires K >= 2"):
        estimate_K(X, "kmeans", range(1, 4), criterion="silhouette")

    with pytest.raises(ValueError, match="Unknown criterion"):
        estimate_K(X, "kmeans", range(2, 4), criterion="invalid_crit")

    with pytest.raises(ValueError, match="Unknown algo_name"):
        estimate_K(X, "invalid_algo", range(2, 4), criterion="bic")


def test_estimate_k_validates_before_fitting(monkeypatch):
    def fail_fit(self, Z, K):
        raise AssertionError("fit must not run with invalid arguments")

    monkeypatch.setattr(KMeansAlgo, "fit", fail_fit)
    X = np.random.default_rng(0).standard_normal((50, 5))
    with pytest.raises(ValueError, match="Unknown criterion"):
        estimate_K(X, "kmeans", range(2, 4), criterion="invalid_crit")


def test_fit_original_clustering_agreement():
    X, y = load_dataset("synthetic", n=200, p=20, n_clusters=4, separation=8.0, seed=42)
    orig_model = fit_original_clustering(X, "kmeans", K=4, seed=42)
    pred_orig = orig_model.predict(X)
    assert ari(y, pred_orig) > 0.8


def test_save_and_load_original_clustering(tmp_path):
    X, _ = load_dataset("synthetic", n=200, p=20, n_clusters=4, separation=8.0, seed=0)
    X_train, X_test = X[:150], X[150:]
    model = fit_original_clustering(X_train, "gmm", K=4, seed=0)
    labels = {"train": model.predict(X_train), "test": model.predict(X_test)}

    path = save_clustering(
        tmp_path / "sub" / "original.npz", labels, algo_name="gmm", K=4, seed=0
    )
    loaded = load_clustering(path)

    assert (loaded["algo_name"], loaded["K"], loaded["seed"]) == ("gmm", 4, 0)
    assert loaded["labels"].keys() == {"train", "test"}
    for split in labels:
        np.testing.assert_array_equal(loaded["labels"][split], labels[split])


def test_ding_he_pca_subspace_property():
    """One case consistent with Ding and He (2004), Theorem 3.1, which says
    the continuous cluster indicators lie in the span of the first K - 1
    principal components. It does not test the theorem: it only checks that,
    for one synthetic dataset with K = 4, projecting onto d = K - 1 = 3
    components keeps K recoverable and the clustering close to the labels.
    """
    K = 4
    X, y = load_dataset(
        "synthetic", n=400, p=30, n_clusters=K, separation=8.0, noise_std=1.0, seed=42
    )

    d = K - 1
    pca = PCA(n_components=d, random_state=42)
    Z = pca.fit_transform(X)

    # 1. K estimation in reduced space recovers K=4
    est_k = estimate_K(Z, "kmeans", range(2, 7), criterion="silhouette", seed=42)
    assert est_k == K

    # 2. Clustering quality in reduced space matches or is close to original
    km = KMeansAlgo(n_init=5, seed=42).fit(Z, K=K)
    reduced_ari = ari(y, km.predict(Z))
    assert reduced_ari > 0.9
