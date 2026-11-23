"""Visualização dos clusterings (K-means e GMM) e das curvas de estimativa de K (Silhouette e BIC).

Uso:
    uv run python scripts/visualize_clustering.py --dataset synthetic
    uv run python scripts/visualize_clustering.py --dataset digits --n 500
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from pcacluster.clustering import GMMCluster, KMeansCluster, _kmeans_bic
from pcacluster.data import load_dataset
from pcacluster.metrics import acc, ari


def plot_clustering_and_k_estimation(
    dataset_name: str,
    n_samples: int | None = 600,
    k_min: int = 2,
    k_max: int = 8,
    seed: int = 42,
    output_dir: Path = Path("results/figures"),
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar dados
    if dataset_name == "synthetic":
        X, y = load_dataset("synthetic", n=n_samples or 600, p=30, n_clusters=4, separation=7.0, seed=seed)
        k_true = 4
    elif dataset_name == "digits":
        X, y = load_dataset("digits", n=n_samples, seed=seed)
        k_true = 10
    else:
        X, y = load_dataset(dataset_name, n=n_samples, seed=seed)
        k_true = len(np.unique(y))

    # 2. Projeção PCA em 2D para visualização espacial
    pca_2d = PCA(n_components=2, random_state=seed)
    Z_2d = pca_2d.fit_transform(X)

    # 3. Calcular curvas de K para Silhouette e BIC
    k_range = range(k_min, k_max + 1)
    sil_kmeans, sil_gmm = [], []
    bic_kmeans, bic_gmm = [], []

    for k in k_range:
        # KMeans
        km = KMeansCluster(n_init=5, seed=seed).fit(X, k)
        labels_km = km.predict(X)
        sil_kmeans.append(silhouette_score(X, labels_km))
        assert km.model_ is not None
        bic_kmeans.append(_kmeans_bic(km.model_, X))

        # GMM
        gmm = GMMCluster(seed=seed).fit(X, k)
        labels_gmm = gmm.predict(X)
        sil_gmm.append(silhouette_score(X, labels_gmm))
        assert gmm.model_ is not None
        bic_gmm.append(float(gmm.model_.bic(X)))

    best_k_sil_km = k_range[int(np.argmax(sil_kmeans))]
    best_k_bic_gmm = k_range[int(np.argmin(bic_gmm))]

    # 4. Ajustar modelos com seus melhores K (ou K verdadeiro)
    km_best = KMeansCluster(n_init=5, seed=seed).fit(X, best_k_sil_km)
    pred_km = km_best.predict(X)

    gmm_best = GMMCluster(seed=seed).fit(X, best_k_bic_gmm)
    pred_gmm = gmm_best.predict(X)

    # 5. Montar figura comparativa (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f"Análise de Clustering e Estimação de K ({dataset_name.capitalize()} | N={len(X)}, P={X.shape[1]})",
        fontsize=16,
        fontweight="bold",
    )

    # Gráfico 1: Curva Silhouette
    ax1 = axes[0, 0]
    ax1.plot(list(k_range), sil_kmeans, "o-", label="K-means", color="#1f77b4", linewidth=2)
    ax1.plot(list(k_range), sil_gmm, "s--", label="GMM (diag)", color="#ff7f0e", linewidth=2)
    ax1.axvline(k_true, color="red", linestyle=":", label=f"K verdadeiro ({k_true})")
    ax1.set_title("Coeficiente de Silhueta Médio (Maior = Melhor)", fontweight="bold")
    ax1.set_xlabel("Número de clusters (K)")
    ax1.set_ylabel("Score de Silhueta")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Gráfico 2: Curva BIC
    ax2 = axes[0, 1]
    ax2.plot(list(k_range), bic_gmm, "s--", label="GMM BIC", color="#ff7f0e", linewidth=2)
    # Plotamos KMeans BIC em eixo secundário se a escala for muito distante
    ax2_km = ax2.twinx()
    ax2_km.plot(list(k_range), bic_kmeans, "o-", label="K-means BIC", color="#1f77b4", linewidth=2)
    ax2.axvline(k_true, color="red", linestyle=":", label=f"K verdadeiro ({k_true})")
    ax2.set_title("Critério de Informação Bayesiano - BIC (Menor = Melhor)", fontweight="bold")
    ax2.set_xlabel("Número de clusters (K)")
    ax2.set_ylabel("GMM BIC", color="#ff7f0e")
    ax2_km.set_ylabel("K-means BIC", color="#1f77b4")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Gráfico 3: Dispersão 2D com K-means
    ari_km = ari(y, pred_km)
    acc_km = acc(y, pred_km) if len(np.unique(pred_km)) == len(np.unique(y)) else float("nan")
    ax3 = axes[1, 0]
    scatter_km = ax3.scatter(
        Z_2d[:, 0], Z_2d[:, 1], c=pred_km, cmap="tab10", s=25, alpha=0.8, edgecolors="none"
    )
    # Projetar centroides se K-means
    if km_best.model_ is not None:
        centroids_2d = pca_2d.transform(km_best.model_.cluster_centers_)
        ax3.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            c="black",
            marker="X",
            s=120,
            linewidths=1.5,
            edgecolors="white",
            label="Centroides",
        )
    ax3.set_title(
        f"K-means (K={best_k_sil_km}) nos primeiros 2 PCs\nARI={ari_km:.3f}"
        + (f" | ACC={acc_km:.3f}" if not np.isnan(acc_km) else ""),
        fontweight="bold",
    )
    ax3.set_xlabel("PC 1")
    ax3.set_ylabel("PC 2")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper right")

    # Gráfico 4: Dispersão 2D com GMM
    ari_gmm = ari(y, pred_gmm)
    acc_gmm = acc(y, pred_gmm) if len(np.unique(pred_gmm)) == len(np.unique(y)) else float("nan")
    ax4 = axes[1, 1]
    ax4.scatter(
        Z_2d[:, 0], Z_2d[:, 1], c=pred_gmm, cmap="tab10", s=25, alpha=0.8, edgecolors="none"
    )
    if gmm_best.model_ is not None:
        means_2d = pca_2d.transform(gmm_best.model_.means_)
        ax4.scatter(
            means_2d[:, 0],
            means_2d[:, 1],
            c="black",
            marker="P",
            s=120,
            linewidths=1.5,
            edgecolors="white",
            label="Médias GMM",
        )
    ax4.set_title(
        f"GMM diag (K={best_k_bic_gmm}) nos primeiros 2 PCs\nARI={ari_gmm:.3f}"
        + (f" | ACC={acc_gmm:.3f}" if not np.isnan(acc_gmm) else ""),
        fontweight="bold",
    )
    ax4.set_xlabel("PC 1")
    ax4.set_ylabel("PC 2")
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(loc="upper right")

    plt.tight_layout()
    fig_path = output_dir / f"clustering_k_estimation_{dataset_name}.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figura salva em: {fig_path.resolve()}")
    return fig_path


def main():
    parser = argparse.ArgumentParser(description="Visualizar clustering e estimação de K.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "digits", "mnist"],
        help="Nome do dataset.",
    )
    parser.add_argument("--n", type=int, default=600, help="Número de amostras para subamostragem.")
    parser.add_argument("--k-min", type=int, default=2, help="K mínimo a testar.")
    parser.add_argument("--k-max", type=int, default=8, help="K máximo a testar.")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória.")
    args = parser.parse_args()

    plot_clustering_and_k_estimation(
        dataset_name=args.dataset,
        n_samples=args.n,
        k_min=args.k_min,
        k_max=args.k_max,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
