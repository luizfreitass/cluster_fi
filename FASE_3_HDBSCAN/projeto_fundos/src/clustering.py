"""
Módulo de Clustering
Algoritmos:
  1. K-Means  (K=4, resultado principal)
  2. Agrupamento Hierárquico (Ward)
  3. HDBSCAN  (substitui DBSCAN — mais robusto em alta dimensão)
  4. Análise dos clusters fora da renda fixa (os 12% de interesse)
  5. Análise secundária do cluster dominante (os 88%)

Referência HDBSCAN: Campello et al. (2013) via artigo Carneiro & Freitas (BRACIS)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors

try:
    import hdbscan
    HDBSCAN_DISPONIVEL = True
except ImportError:
    HDBSCAN_DISPONIVEL = False
    print("     [Aviso] hdbscan não instalado. Execute: pip install hdbscan")

RESULTS_DIR = "resultados"
os.makedirs(RESULTS_DIR, exist_ok=True)

K_PRINCIPAL  = 4
K_SECUNDARIO = 4


def _get_pc_cols(df):
    return [c for c in df.columns if c.startswith('PC')]


# ─────────────────────────────────────────────
# Painel análise de K
# ─────────────────────────────────────────────
def analisar_k(X, k_min=2, k_max=12):
    ks          = range(k_min, k_max + 1)
    inercias    = []
    silhouettes = []
    bal_scores  = []

    print(f"     Testando K de {k_min} a {k_max}...")
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        inercias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
        counts = np.bincount(labels)
        bal_scores.append(counts.max() / len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(ks, inercias, marker='o', color='steelblue', linewidth=2)
    axes[0].axvline(K_PRINCIPAL, linestyle='--', color='green',
                    label=f'K={K_PRINCIPAL} selecionado')
    axes[0].set_title('Inércia (Método do Cotovelo)', fontsize=13)
    axes[0].set_xlabel('K'); axes[0].set_ylabel('Inércia')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ks, silhouettes, marker='s', color='tomato', linewidth=2)
    axes[1].axvline(K_PRINCIPAL, linestyle='--', color='green',
                    label=f'K={K_PRINCIPAL} (Sil={silhouettes[K_PRINCIPAL-k_min]:.3f})')
    axes[1].set_title('Silhouette Score Médio', fontsize=13)
    axes[1].set_xlabel('K'); axes[1].set_ylabel('Silhouette')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(ks, [b*100 for b in bal_scores],
                 marker='^', color='darkorange', linewidth=2)
    axes[2].axvline(K_PRINCIPAL, linestyle='--', color='green',
                    label=f'K={K_PRINCIPAL} selecionado')
    axes[2].axhline(40, linestyle=':', color='gray', label='Limiar 40%')
    axes[2].set_title('Balanceamento\n(% do maior cluster)', fontsize=13)
    axes[2].set_xlabel('K'); axes[2].set_ylabel('% maior cluster')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle('Análise de K — K-Means', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/kmeans_analise_k.png", dpi=150, bbox_inches='tight')
    plt.close()

    return silhouettes[K_PRINCIPAL - k_min], bal_scores[K_PRINCIPAL - k_min]


# ─────────────────────────────────────────────
# K-Means principal
# ─────────────────────────────────────────────
def kmeans_final(X, k=K_PRINCIPAL):
    km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    km.fit(X)
    counts = np.bincount(km.labels_)
    print(f"     K-Means final → K={k}")
    for i, c in enumerate(counts):
        print(f"       Cluster {i}: {c:,} fundos ({c/len(X)*100:.1f}%)")
    return km


# ─────────────────────────────────────────────
# Análise dos clusters FORA da renda fixa
# ─────────────────────────────────────────────
def analisar_clusters_minoritarios(X, labels_principais, df_raw=None):
    """
    Foco nos clusters que NÃO são o dominante.
    Esses são os 12% de interesse — renda variável, multimercado, estruturados.
    Gera heatmap comparativo e perfil detalhado de cada grupo.
    """
    cluster_dom = int(np.bincount(labels_principais).argmax())
    clusters_min = [c for c in np.unique(labels_principais) if c != cluster_dom]

    print(f"\n     ── Análise dos Clusters Minoritários (fora do Cluster {cluster_dom}) ──")

    feature_cols = [c for c in [
        'retorno_medio_diario', 'retorno_acumulado', 'volatilidade',
        'max_drawdown', 'pl_medio', 'captacao_media', 'resgate_medio',
        'cotistas_medio', 'fluxo_liquido_medio', 'sharpe'
    ] if df_raw is not None and c in df_raw.columns]

    if df_raw is None or not feature_cols:
        print("     [Aviso] df_raw não disponível para análise minoritária.")
        return

    df_analise = df_raw.copy().reset_index(drop=True)
    df_analise['cluster'] = labels_principais

    # Perfil médio de TODOS os clusters lado a lado para comparação
    medias = df_analise.groupby('cluster')[feature_cols].mean()
    medias_norm = (medias - medias.min()) / (medias.max() - medias.min() + 1e-9)

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(medias_norm, annot=medias.round(4), fmt='.4f',
                cmap='RdYlGn', ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Valor normalizado'})

    # Destaca os clusters minoritários com borda
    for c in clusters_min:
        ax.add_patch(plt.Rectangle(
            (0, c), len(feature_cols), 1,
            fill=False, edgecolor='blue', lw=3, label='Cluster minoritário' if c == clusters_min[0] else ''
        ))

    ax.set_title('Perfil Comparativo — Todos os Clusters\n'
                 '(borda azul = clusters fora da renda fixa)', fontsize=13)
    ax.set_xlabel('Feature'); ax.set_ylabel('Cluster')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/heatmap_comparativo_clusters.png", dpi=150)
    plt.close()

    # Análise aprofundada de cada cluster minoritário
    print(f"\n     Perfil dos clusters minoritários:")
    print(f"     {'Cluster':>8} | {'Fundos':>7} | {'Retorno Acum':>13} | "
          f"{'Volatilidade':>12} | {'PL Médio':>12} | {'Sharpe':>8}")
    print(f"     {'-'*75}")

    for c in clusters_min:
        mask = df_analise['cluster'] == c
        g = df_analise[mask]
        print(f"     {c:>8} | {len(g):>7,} | "
              f"{g['retorno_acumulado'].mean():>13.4f} | "
              f"{g['volatilidade'].mean():>12.6f} | "
              f"{g['pl_medio'].mean():>12,.0f} | "
              f"{g['sharpe'].mean():>8.4f}")

    # Scatter comparativo PC1 x PC2 destacando minoritários
    if 'PC1' in X.shape or True:
        # Usa os dois primeiros componentes diretamente de X
        fig, ax = plt.subplots(figsize=(10, 7))
        cores = cm.tab10(np.linspace(0, 1, K_PRINCIPAL))

        mask_dom = labels_principais == cluster_dom
        ax.scatter(X[mask_dom, 0], X[mask_dom, 1],
                   c='lightgray', alpha=0.3, s=5, label=f'Cluster {cluster_dom} (Renda Fixa)')

        nomes_clusters = {
            0: 'Renda Variável / Multimercado',
            2: 'Multimercado Intermediário',
            3: 'Fundos Estruturados',
        }

        for i, c in enumerate(clusters_min):
            mask = labels_principais == c
            nome = nomes_clusters.get(c, f'Cluster {c}')
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=[cores[i+1]], alpha=0.8, s=20,
                       label=f'Cluster {c}: {nome} (n={mask.sum():,})')

        ax.set_xlabel('Componente Principal 1', fontsize=11)
        ax.set_ylabel('Componente Principal 2', fontsize=11)
        ax.set_title('Mapa PCA — Clusters Minoritários em Destaque\n'
                     '(Foco nos fundos fora da renda fixa)', fontsize=13)
        ax.legend(markerscale=2, loc='upper right')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/mapa_clusters_minoritarios.png", dpi=150)
        plt.close()
        print(f"     Mapa dos clusters minoritários salvo.")

    # Salva CSV detalhado dos fundos minoritários
    df_min = df_analise[df_analise['cluster'].isin(clusters_min)].copy()
    cols_salvar = ['CNPJ_FUNDO', 'DENOM_SOCIAL', 'TP_FUNDO', 'CLASSE',
                   'cluster'] + feature_cols
    cols_salvar = [c for c in cols_salvar if c in df_min.columns]
    df_min[cols_salvar].to_csv(f"{RESULTS_DIR}/fundos_minoritarios.csv", index=False)
    print(f"     CSV com {len(df_min):,} fundos minoritários salvo.")


# ─────────────────────────────────────────────
# Análise secundária do cluster dominante
# ─────────────────────────────────────────────
def analise_secundaria(X, labels_principais, df_raw=None):
    cluster_dom = int(np.bincount(labels_principais).argmax())
    mask = labels_principais == cluster_dom
    X_dom = X[mask]

    print(f"\n     ── Análise Secundária: Cluster {cluster_dom} "
          f"({mask.sum():,} fundos) ──")

    km_sec = KMeans(n_clusters=K_SECUNDARIO, random_state=42,
                    n_init=20, max_iter=500)
    labels_sec = km_sec.fit_predict(X_dom)
    sil_sec = silhouette_score(X_dom, labels_sec)

    counts = np.bincount(labels_sec)
    print(f"     Sub-clusters (K={K_SECUNDARIO}, Silhouette={sil_sec:.4f}):")
    for i, c in enumerate(counts):
        print(f"       Sub-cluster {i}: {c:,} fundos ({c/len(X_dom)*100:.1f}%)")

    if df_raw is not None:
        feature_cols = [c for c in [
            'retorno_medio_diario', 'retorno_acumulado', 'volatilidade',
            'max_drawdown', 'pl_medio', 'captacao_media', 'resgate_medio',
            'cotistas_medio', 'fluxo_liquido_medio', 'sharpe'
        ] if c in df_raw.columns]

        import seaborn as sns
        df_dom = df_raw[mask].copy().reset_index(drop=True)
        df_dom['sub_cluster'] = labels_sec
        medias = df_dom.groupby('sub_cluster')[feature_cols].mean()
        medias_norm = (medias - medias.min()) / (medias.max() - medias.min() + 1e-9)

        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(medias_norm, annot=medias.round(4), fmt='.4f',
                    cmap='RdYlGn', ax=ax, linewidths=0.5,
                    cbar_kws={'label': 'Valor normalizado'})
        ax.set_title(f'Sub-perfis dentro do Cluster Dominante '
                     f'(Cluster {cluster_dom}) — K={K_SECUNDARIO}')
        ax.set_xlabel('Feature'); ax.set_ylabel('Sub-cluster')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/heatmap_secundario.png", dpi=150)
        plt.close()
        print(f"     Heatmap secundário salvo.")

        df_dom[['CNPJ_FUNDO', 'DENOM_SOCIAL', 'TP_FUNDO', 'sub_cluster']].to_csv(
            f"{RESULTS_DIR}/labels_secundario.csv", index=False)

    return labels_sec, mask, sil_sec


# ─────────────────────────────────────────────
# Agrupamento Hierárquico
# ─────────────────────────────────────────────
def hierarquico(X, n_clusters=K_PRINCIPAL):
    model = AgglomerativeClustering(n_clusters=n_clusters,
                                    linkage='ward', metric='euclidean')
    model.fit(X)
    counts = np.bincount(model.labels_)
    print(f"     Hierárquico (Ward) → {n_clusters} clusters")
    for i, c in enumerate(counts):
        print(f"       Cluster {i}: {c:,} fundos ({c/len(X)*100:.1f}%)")
    return model


def plot_dendrograma(X, n_amostras=500):
    from scipy.cluster.hierarchy import dendrogram, linkage
    if len(X) > n_amostras:
        idx = np.random.choice(len(X), n_amostras, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    Z = linkage(Xs, method='ward')
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
               leaf_rotation=45, leaf_font_size=9, show_contracted=True,
               color_threshold=0.7*max(Z[:,2]))
    ax.set_title('Dendrograma (Ward) — amostra de fundos', fontsize=13)
    ax.set_xlabel('Fundo / Nó'); ax.set_ylabel('Distância de Ward')
    ax.axhline(y=0.7*max(Z[:,2]), color='red', linestyle='--',
               alpha=0.5, label='Corte sugerido')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/dendrograma.png", dpi=150)
    plt.close()
    print("     Dendrograma salvo.")


# ─────────────────────────────────────────────
# HDBSCAN
# ─────────────────────────────────────────────
def hdbscan_clustering(X):
    """
    HDBSCAN — Hierarchical Density-Based Spatial Clustering of Applications with Noise.

    Vantagens sobre DBSCAN neste problema:
      - Não exige densidade uniforme (clusters de densidades diferentes)
      - Mais robusto em alta dimensão (7D PCA)
      - Detecta outliers com score de probabilidade por ponto
      - Parâmetro min_cluster_size mais intuitivo que eps

    Referência: Campello et al. (2013) — citado em Carneiro & Freitas (BRACIS 2019)

    Parâmetros testados:
      min_cluster_size : tamanho mínimo para considerar um grupo
      min_samples      : controla conservadorismo na detecção de ruído
    """
    if not HDBSCAN_DISPONIVEL:
        print("     [Erro] hdbscan não instalado. Execute: pip install hdbscan")
        return None

    # Testa diferentes configurações de min_cluster_size
    resultados_teste = []
    print("     Testando configurações HDBSCAN...")
    print(f"     {'min_cls':>8} | {'Clusters':>9} | {'Ruídos':>7} | {'% Ruído':>8} | Silhouette")
    print(f"     {'-'*55}")

    for mcs in [50, 100, 200, 500, 1000]:
        modelo = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=1,
            metric='euclidean',
            cluster_selection_method='eom',   # Excess of Mass — mais estável
            prediction_data=True
        )
        labels = modelo.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = int(np.sum(labels == -1))

        sil = float('nan')
        if n_clusters > 1:
            mask_validos = labels != -1
            if mask_validos.sum() > n_clusters:
                sil = silhouette_score(X[mask_validos], labels[mask_validos])

        resultados_teste.append({
            'min_cluster_size': mcs,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'pct_noise': n_noise / len(X) * 100,
            'silhouette': sil,
            'labels': labels,
            'modelo': modelo,
        })

        sil_str = f"{sil:.4f}" if not np.isnan(sil) else "  —  "
        print(f"     {mcs:>8} | {n_clusters:>9} | {n_noise:>7,} | "
              f"{n_noise/len(X)*100:>7.1f}% | {sil_str}")

    # Escolhe a config com mais clusters úteis e Silhouette válido
    validos = [r for r in resultados_teste
               if r['n_clusters'] > 1 and not np.isnan(r['silhouette'])]

    if validos:
        melhor = max(validos, key=lambda r: r['silhouette'])
    else:
        # Se nenhum produziu clusters úteis, pega o com mais clusters
        melhor = max(resultados_teste, key=lambda r: r['n_clusters'])

    print(f"\n     HDBSCAN escolhido: min_cluster_size={melhor['min_cluster_size']}")
    print(f"       Clusters: {melhor['n_clusters']}")
    print(f"       Ruídos (outliers): {melhor['n_noise']:,} ({melhor['pct_noise']:.1f}%)")
    if not np.isnan(melhor['silhouette']):
        print(f"       Silhouette: {melhor['silhouette']:.4f}")

    # Plot de probabilidade de pertencimento (força do cluster por ponto)
    modelo_final = melhor['modelo']
    if hasattr(modelo_final, 'probabilities_'):
        fig, ax = plt.subplots(figsize=(10, 5))
        probs = modelo_final.probabilities_
        labels_finais = melhor['labels']

        # Histograma de probabilidades por cluster
        cores = cm.tab10(np.linspace(0, 1, max(melhor['n_clusters'], 1)))
        for c in sorted(set(labels_finais)):
            if c == -1:
                continue
            p = probs[labels_finais == c]
            ax.hist(p, bins=30, alpha=0.6,
                    label=f'Cluster {c} (n={int((labels_finais==c).sum()):,})',
                    color=cores[c % len(cores)])

        ax.set_xlabel('Probabilidade de pertencimento ao cluster')
        ax.set_ylabel('Número de fundos')
        ax.set_title(f'HDBSCAN — Confiança por Cluster\n'
                     f'(min_cluster_size={melhor["min_cluster_size"]})')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/hdbscan_probabilidades.png", dpi=150)
        plt.close()

    # K-Distance plot para comparação com DBSCAN anterior
    k = 5
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nbrs.kneighbors(X)
    k_dists  = np.sort(dists[:, k-1])[::-1]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(k_dists, color='steelblue', linewidth=1.5)
    ax.set_title(f'K-Distance Graph (k={k})\n'
                 f'HDBSCAN não depende de eps fixo — usa hierarquia de densidades')
    ax.set_xlabel('Pontos ordenados')
    ax.set_ylabel(f'{k}-NN distância')
    ax.annotate('DBSCAN precisava definir\num eps fixo aqui — problema\nresolvido pelo HDBSCAN',
                xy=(len(k_dists)*0.3, k_dists[int(len(k_dists)*0.3)]),
                xytext=(len(k_dists)*0.5, k_dists[int(len(k_dists)*0.3)]*1.3),
                arrowprops=dict(facecolor='tomato', shrink=0.05),
                fontsize=9, color='tomato')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/hdbscan_kdistance.png", dpi=150)
    plt.close()

    return melhor['modelo'], melhor['labels'], melhor['n_clusters'], melhor['n_noise']


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────
def executar_clustering(df_pca, df_raw=None):
    pc_cols = _get_pc_cols(df_pca)
    X = df_pca[pc_cols].values

    resultados = {}

    # ── K-Means ───────────────────────────────
    print("  → K-Means:")
    sil_k4, bal_k4 = analisar_k(X)
    print(f"     K={K_PRINCIPAL}: Silhouette={sil_k4:.4f}, "
          f"maior cluster={bal_k4*100:.1f}%")
    km = kmeans_final(X)
    resultados['kmeans'] = {
        'modelo': km, 'labels': km.labels_,
        'k': K_PRINCIPAL, 'nome': 'K-Means',
    }

    # ── Foco nos clusters minoritários (os 12%) ──
    print("  → Análise dos Clusters Minoritários (foco do estudo):")
    analisar_clusters_minoritarios(X, km.labels_, df_raw=df_raw)

    # ── Análise secundária do dominante ──────
    print("  → Análise Secundária do Cluster Dominante (renda fixa):")
    labels_sec, mask_dom, sil_sec = analise_secundaria(
        X, km.labels_, df_raw=df_raw)
    resultados['secundario'] = {
        'labels_sec': labels_sec,
        'mask_dom': mask_dom,
        'sil': sil_sec,
    }

    # ── Hierárquico ───────────────────────────
    print("  → Agrupamento Hierárquico:")
    plot_dendrograma(X)
    hc = hierarquico(X)
    resultados['hierarquico'] = {
        'modelo': hc, 'labels': hc.labels_,
        'k': K_PRINCIPAL, 'nome': 'Hierárquico (Ward)',
    }

    # ── HDBSCAN ───────────────────────────────
    print("  → HDBSCAN:")
    resultado_hdb = hdbscan_clustering(X)
    if resultado_hdb is not None:
        modelo_hdb, labels_hdb, n_hdb, n_noise_hdb = resultado_hdb
        resultados['hdbscan'] = {
            'modelo': modelo_hdb,
            'labels': labels_hdb,
            'k': n_hdb,
            'nome': 'HDBSCAN',
        }
    else:
        resultados['hdbscan'] = {
            'modelo': None, 'labels': np.full(len(X), -1),
            'k': 0, 'nome': 'HDBSCAN',
        }

    return resultados