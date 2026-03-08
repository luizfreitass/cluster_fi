"""
Módulo de Avaliação dos Clusters
Métricas:
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Index
  - Pureza por Classe ANBIMA / TP_FUNDO (pós-processamento interpretativo)
"""

import numpy as np
import pandas as pd
import os

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

RESULTS_DIR = "resultados"
os.makedirs(RESULTS_DIR, exist_ok=True)


def _get_pc_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith('PC')]


def calcular_metricas(X: np.ndarray, labels: np.ndarray,
                      nome: str) -> dict:
    """
    Calcula as métricas de avaliação interna para um conjunto de labels.
    Retorna dicionário com os valores.
    """
    # Filtra ruído do DBSCAN (label = -1)
    mask = labels != -1
    X_v  = X[mask]
    l_v  = labels[mask]

    if len(set(l_v)) < 2:
        return {
            'algoritmo'          : nome,
            'n_clusters'         : len(set(l_v)),
            'silhouette'         : np.nan,
            'davies_bouldin'     : np.nan,
            'calinski_harabasz'  : np.nan,
            'n_ruido'            : int(np.sum(labels == -1)),
        }

    return {
        'algoritmo'          : nome,
        'n_clusters'         : len(set(l_v)),
        'silhouette'         : round(silhouette_score(X_v, l_v), 4),
        'davies_bouldin'     : round(davies_bouldin_score(X_v, l_v), 4),
        'calinski_harabasz'  : round(calinski_harabasz_score(X_v, l_v), 2),
        'n_ruido'            : int(np.sum(labels == -1)),
    }


def avaliar_clusters(df_pca: pd.DataFrame,
                     resultados: dict) -> pd.DataFrame:
    """
    Avalia todos os algoritmos e salva tabela de métricas em CSV.
    """
    pc_cols = _get_pc_cols(df_pca)
    X = df_pca[pc_cols].values

    rows = []
    for alg, res in resultados.items():
        # Pula entradas auxiliares que não têm 'labels' (ex: análise secundária)
        if 'labels' not in res or 'nome' not in res:
            continue
        m = calcular_metricas(X, res['labels'], res['nome'])
        rows.append(m)

    df_met = pd.DataFrame(rows)

    # Exibe no terminal
    print("\n     ┌─────────────────────────────────────────────────────────────┐")
    print(      "     │                  MÉTRICAS DE AVALIAÇÃO                      │")
    print(      "     ├──────────────────────┬──────────┬──────────┬───────────────┤")
    print(      "     │ Algoritmo            │ Silhouet │ D-Bouldin│ Calinski-Har. │")
    print(      "     ├──────────────────────┼──────────┼──────────┼───────────────┤")
    for _, row in df_met.iterrows():
        print(f"     │ {row['algoritmo']:<20} │ "
              f"{row['silhouette']:>8.4f} │ "
              f"{row['davies_bouldin']:>8.4f} │ "
              f"{row['calinski_harabasz']:>13.2f} │")
    print(      "     └──────────────────────┴──────────┴──────────┴───────────────┘")
    print()
    print("     Interpretação:")
    print("       Silhouette    → quanto maior melhor  (intervalo -1 a 1)")
    print("       Davies-Bouldin → quanto menor melhor (≥ 0)")
    print("       Calinski-Har. → quanto maior melhor  (≥ 0)")

    # Salva CSV
    path_csv = os.path.join(RESULTS_DIR, "metricas_clustering.csv")
    df_met.to_csv(path_csv, index=False)
    print(f"\n     Métricas salvas em: {path_csv}")

    return df_met


def perfil_clusters(df_raw: pd.DataFrame,
                    labels: np.ndarray,
                    feature_cols: list[str],
                    nome_alg: str) -> pd.DataFrame:
    """
    Gera tabela de perfil médio de cada cluster.
    Útil para o pós-processamento interpretativo.
    """
    df = df_raw.copy()
    df['cluster'] = labels

    perfil = (df[df['cluster'] != -1]
                .groupby('cluster')[feature_cols]
                .agg(['mean', 'std', 'count']))

    path_csv = os.path.join(RESULTS_DIR,
                            f"perfil_{nome_alg.lower().replace(' ', '_')}.csv")
    perfil.to_csv(path_csv)
    print(f"     Perfil dos clusters salvo em: {path_csv}")
    return perfil