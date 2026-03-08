"""
Módulo de Visualização e Pós-processamento
Gera todos os gráficos e tabelas de interpretação dos clusters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os

from sklearn.manifold import TSNE
from src.preprocessamento import FEATURE_COLS
from src.avaliacao import perfil_clusters

RESULTS_DIR = "resultados"
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="tab10")


def _get_pc_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith('PC')]


# ─────────────────────────────────────────────
# t-SNE
# ─────────────────────────────────────────────
def _tsne_coords(X: np.ndarray, perp=30) -> np.ndarray:
    """Reduz para 2D com t-SNE (usa cache em arquivo)."""
    cache = os.path.join(RESULTS_DIR, "tsne_coords.npy")
    if os.path.exists(cache):
        return np.load(cache)
    tsne = TSNE(n_components=2, perplexity=perp,
                random_state=42, max_iter=1000)
    coords = tsne.fit_transform(X)
    np.save(cache, coords)
    return coords


# ─────────────────────────────────────────────
# Scatter plots
# ─────────────────────────────────────────────
def _scatter_clusters(coords_2d: np.ndarray,
                      labels: np.ndarray,
                      titulo: str,
                      nome_arquivo: str,
                      classe_true: np.ndarray | None = None):
    """
    Plota scatter 2D colorido por cluster.
    Se classe_true fornecido, faz subplot com ground truth (TP_FUNDO).
    """
    n_plots = 2 if classe_true is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    # Clusters
    unique_labels = sorted(set(labels))
    colors = cm.tab10(np.linspace(0, 1, max(len(unique_labels), 2)))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        cor  = 'lightgray' if lbl == -1 else colors[i % len(colors)]
        nome = 'Ruído' if lbl == -1 else f'Cluster {lbl}'
        axes[0].scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                        c=[cor], label=nome, s=8, alpha=0.7)

    axes[0].set_title(f'{titulo}\n(t-SNE)')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    axes[0].legend(markerscale=2, fontsize=8,
                   loc='upper right', framealpha=0.5)

    # Ground truth (se disponível)
    if classe_true is not None:
        cats = pd.Categorical(classe_true)
        for i, cat in enumerate(cats.categories):
            mask = cats == cat
            axes[1].scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                            c=[colors[i % len(colors)]], label=str(cat),
                            s=8, alpha=0.7)
        axes[1].set_title('Tipo de Fundo (TP_FUNDO)\n(t-SNE)')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].legend(markerscale=2, fontsize=7,
                       loc='upper right', framealpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, nome_arquivo), dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Heatmap de perfil
# ─────────────────────────────────────────────
def _heatmap_perfil(df_raw: pd.DataFrame,
                    labels: np.ndarray,
                    nome_alg: str):
    """Heatmap normalizado das features médias por cluster."""
    cols = [c for c in FEATURE_COLS if c in df_raw.columns]
    df   = df_raw[cols].copy()
    df['cluster'] = labels

    medias = (df[df['cluster'] != -1]
                .groupby('cluster')[cols].mean())

    # Normaliza entre 0 e 1 por feature para visualização
    medias_norm = (medias - medias.min()) / (medias.max() - medias.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(max(10, len(cols)), max(4, len(medias) * 0.7)))
    sns.heatmap(medias_norm, annot=medias.round(4),
                fmt='.4f', cmap='RdYlGn', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Valor normalizado'})
    ax.set_title(f'Perfil dos Clusters — {nome_alg}')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Cluster')
    plt.tight_layout()
    fname = f"heatmap_{nome_alg.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Barras com distribuição de TP_FUNDO
# ─────────────────────────────────────────────
def _barras_composicao(df_raw: pd.DataFrame,
                       labels: np.ndarray,
                       col: str,
                       nome_alg: str):
    """Gráfico de barras empilhadas mostrando composição de col por cluster."""
    if col not in df_raw.columns:
        return

    col_data = df_raw[col]
    if isinstance(col_data, pd.DataFrame):
        col_data = col_data.iloc[:, 0]
    df = pd.DataFrame({col: col_data.values})
    df['cluster'] = labels
    df = df[df['cluster'] != -1]

    tabela = (df.groupby(['cluster', col])
                .size()
                .unstack(fill_value=0))
    tabela_pct = tabela.div(tabela.sum(axis=1), axis=0) * 100

    ax = tabela_pct.plot(kind='bar', stacked=True, figsize=(10, 5),
                         colormap='tab20')
    ax.set_title(f'Composição por {col} — {nome_alg}')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Proporção (%)')
    ax.legend(loc='upper right', fontsize=7,
              bbox_to_anchor=(1.15, 1), framealpha=0.5)
    plt.xticks(rotation=0)
    plt.tight_layout()
    fname = f"composicao_{col}_{nome_alg.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Comparação de métricas
# ─────────────────────────────────────────────
def _plot_comparacao_metricas(df_metricas: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metricas_info = [
        ('silhouette',        'Silhouette Score',        'higher is better', 'steelblue'),
        ('davies_bouldin',    'Davies-Bouldin Index',    'lower is better',  'tomato'),
        ('calinski_harabasz', 'Calinski-Harabasz Index', 'higher is better', 'seagreen'),
    ]
    for ax, (col, titulo, nota, cor) in zip(axes, metricas_info):
        vals = df_metricas[col].fillna(0)
        bars = ax.bar(df_metricas['algoritmo'], vals, color=cor, alpha=0.8)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
        ax.set_title(f'{titulo}\n({nota})')
        ax.set_ylabel(col)
        ax.tick_params(axis='x', rotation=10)

    plt.suptitle('Comparação de Algoritmos de Clustering', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparacao_metricas.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────
def gerar_visualizacoes(df_raw: pd.DataFrame,
                        df_pca: pd.DataFrame,
                        resultados: dict,
                        df_metricas: pd.DataFrame):
    pc_cols  = _get_pc_cols(df_pca)
    X        = df_pca[pc_cols].values
    tp_fundo = df_raw.get('TP_FUNDO', pd.Series([''] * len(df_raw))).values

    print("     Calculando t-SNE (pode demorar ~1 min)...")
    coords_2d = _tsne_coords(X)

    # Comparação de métricas
    _plot_comparacao_metricas(df_metricas)
    print("     Gráfico comparativo de métricas salvo.")

    for alg, res in resultados.items():
        # Pula entradas auxiliares sem 'labels' ou 'nome'
        if 'labels' not in res or 'nome' not in res:
            continue
        nome     = res['nome']
        labels   = res['labels']
        print(f"     Gerando gráficos para {nome}...")

        # Scatter t-SNE
        _scatter_clusters(
            coords_2d, labels,
            titulo       = nome,
            nome_arquivo = f"tsne_{alg}.png",
            classe_true  = tp_fundo if len(set(tp_fundo)) > 1 else None,
        )

        # Heatmap de perfil
        _heatmap_perfil(df_raw, labels, nome)

        # Barras de composição
        _barras_composicao(df_raw, labels, 'TP_FUNDO', nome)
        _barras_composicao(df_raw, labels, 'CLASSE',   nome)

        # Salva CSV com labels
        df_out = df_raw[['CNPJ_FUNDO', 'DENOM_SOCIAL',
                         'TP_FUNDO', 'CLASSE']].copy()
        df_out['cluster'] = labels
        path_out = os.path.join(RESULTS_DIR,
                                f"labels_{alg}.csv")
        df_out.to_csv(path_out, index=False)

        # Perfil textual
        cols_feat = [c for c in FEATURE_COLS if c in df_raw.columns]
        perfil_clusters(df_raw, labels, cols_feat, nome)

    print(f"\n     Todos os resultados salvos em: {RESULTS_DIR}/")