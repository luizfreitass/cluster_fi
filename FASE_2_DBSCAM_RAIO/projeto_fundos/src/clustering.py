"""
Módulo de Clustering
Algoritmos:
  1. K-Means
  2. Agrupamento Hierárquico (Ward)
  3. DBSCAN

Inclui seleção automática do número de clusters via Elbow + Silhouette.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

RESULTS_DIR = "resultados"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Colunas apenas com componentes PCA (exclui metadados)
def _get_pc_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith('PC')]


# ─────────────────────────────────────────────
# K-Means
# ─────────────────────────────────────────────
def kmeans_otimo(X: np.ndarray, k_min=2, k_max=12) -> tuple[int, KMeans]:
    """
    Determina o K ótimo usando o método do cotovelo (inércia)
    e o Silhouette Score. Retorna (k_otimo, modelo_ajustado).
    """
    inercias    = []
    silhouettes = []
    ks = range(k_min, k_max + 1)

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inercias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    # K ótimo = maior Silhouette
    k_otimo = ks[int(np.argmax(silhouettes))]

    # Plot cotovelo
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ks, inercias, marker='o', color='steelblue')
    axes[0].set_title('Método do Cotovelo - K-Means')
    axes[0].set_xlabel('Número de Clusters (K)')
    axes[0].set_ylabel('Inércia')

    axes[1].plot(ks, silhouettes, marker='s', color='tomato')
    axes[1].axvline(k_otimo, linestyle='--', color='green',
                    label=f'K ótimo = {k_otimo}')
    axes[1].set_title('Silhouette Score - K-Means')
    axes[1].set_xlabel('Número de Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/kmeans_selecao_k.png", dpi=150)
    plt.close()

    print(f"     K-Means → K ótimo = {k_otimo} "
          f"(Silhouette = {silhouettes[int(np.argmax(silhouettes))]:.4f})")

    km_final = KMeans(n_clusters=k_otimo, random_state=42, n_init=20)
    km_final.fit(X)
    return k_otimo, km_final


# ─────────────────────────────────────────────
# Agrupamento Hierárquico
# ─────────────────────────────────────────────
def hierarquico(X: np.ndarray, n_clusters: int) -> AgglomerativeClustering:
    """Ajusta Agglomerative Clustering com ligação de Ward."""
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward',
        metric='euclidean'
    )
    model.fit(X)
    print(f"     Hierárquico (Ward) → {n_clusters} clusters")
    return model


def plot_dendrograma(X: np.ndarray, n_amostras=500):
    """Plota dendrograma truncado para os primeiros n_amostras."""
    from scipy.cluster.hierarchy import dendrogram, linkage

    if len(X) > n_amostras:
        idx = np.random.choice(len(X), n_amostras, replace=False)
        Xs = X[idx]
    else:
        Xs = X

    Z = linkage(Xs, method='ward')

    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
               leaf_rotation=45, leaf_font_size=9, show_contracted=True)
    ax.set_title('Dendrograma (Ward) — amostra de fundos')
    ax.set_xlabel('Fundo / Nó')
    ax.set_ylabel('Distância')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/dendrograma.png", dpi=150)
    plt.close()
    print("     Dendrograma salvo.")


# ─────────────────────────────────────────────
# DBSCAN
# ─────────────────────────────────────────────
def dbscan_auto(X: np.ndarray, eps_manual=1.5, min_samples=20) -> DBSCAN:
    """
    Ajusta o DBSCAN permitindo parametrização manual para melhor controle.
    Gera o gráfico k-distance para auxiliar no refino do eps.
    """
    k = min_samples
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nbrs.kneighbors(X)
    k_dists  = np.sort(dists[:, k - 1])[::-1]

    # Plot k-distance para ajudar você a achar o "joelho" real olhando a imagem
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_dists, color='steelblue')
    ax.axhline(eps_manual, color='tomato', linestyle='--',
               label=f'eps utilizado = {eps_manual}')
    ax.set_title(f'K-Distance Graph (k={k}) — DBSCAN')
    ax.set_xlabel('Pontos ordenados')
    ax.set_ylabel(f'{k}-NN distância')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/dbscan_kdistance.png", dpi=150)
    plt.close()

    # Ajusta o modelo com os novos parâmetros
    model = DBSCAN(eps=eps_manual, min_samples=min_samples)
    model.fit(X)

    n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
    n_noise    = int(np.sum(model.labels_ == -1))
    
    print(f"     DBSCAN → eps={eps_manual}, min_samples={min_samples}, "
          f"{n_clusters} clusters, {n_noise} ruídos "
          f"({n_noise/len(X)*100:.1f}%)")
    
    return model


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────
def executar_clustering(df_pca: pd.DataFrame) -> dict:
    """
    Executa os três algoritmos de clustering e retorna
    um dicionário com os resultados.
    """
    pc_cols = _get_pc_cols(df_pca)
    X = df_pca[pc_cols].values

    resultados = {}

    # K-Means
    print("  → K-Means:")
    k_otimo, km = kmeans_otimo(X)
    resultados['kmeans'] = {
        'modelo'  : km,
        'labels'  : km.labels_,
        'k'       : k_otimo,
        'nome'    : 'K-Means',
    }

    # Hierárquico (usa mesmo K do K-Means)
    print("  → Agrupamento Hierárquico:")
    plot_dendrograma(X)
    hc = hierarquico(X, n_clusters=k_otimo)
    resultados['hierarquico'] = {
        'modelo' : hc,
        'labels' : hc.labels_,
        'k'      : k_otimo,
        'nome'   : 'Hierárquico (Ward)',
    }

    # DBSCAN
    print("  → DBSCAN:")
    db = dbscan_auto(X)
    n_db = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    resultados['dbscan'] = {
        'modelo' : db,
        'labels' : db.labels_,
        'k'      : n_db,
        'nome'   : 'DBSCAN',
    }

    return resultados

