"""
Módulo de Pré-processamento
Etapas:
  1. Seleção e limpeza de features
  2. Tratamento de outliers (IQR / Winsorização)
  3. Normalização (StandardScaler)
  4. Redução de dimensionalidade (PCA)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

RESULTS_DIR = "resultados"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Features numéricas a utilizar no clustering
FEATURE_COLS = [
    'retorno_medio_diario',
    'retorno_acumulado',
    'volatilidade',
    'max_drawdown',
    'pl_medio',
    'captacao_media',
    'resgate_medio',
    'cotistas_medio',
    'fluxo_liquido_medio',
    'sharpe',
]

# Variância explicada mínima para definir nº de componentes PCA
PCA_VARIANCE = 0.95


def tratar_outliers(df: pd.DataFrame, cols: list[str],
                    lower=0.01, upper=0.99) -> pd.DataFrame:
    """
    Winsorizção: limita valores abaixo do percentil `lower` e
    acima do percentil `upper` para cada feature.
    """
    df = df.copy()
    for c in cols:
        lo = df[c].quantile(lower)
        hi = df[c].quantile(upper)
        df[c] = df[c].clip(lo, hi)
    return df


def preprocessar(df_raw: pd.DataFrame):
    """
    Executa o pipeline completo de pré-processamento.

    Retorna
    -------
    df_pca   : DataFrame com componentes PCA (features para clustering)
    scaler   : StandardScaler ajustado
    pca      : PCA ajustado
    """
    df = df_raw.copy()

    # ── 1. Seleciona features ─────────────────────────────────────────
    cols_disponiveis = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols_disponiveis].copy()

    print(f"     Features selecionadas ({len(cols_disponiveis)}): {cols_disponiveis}")

    # ── 2. Imputa missing values (mediana) ────────────────────────────
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    X = pd.DataFrame(X_imp, columns=cols_disponiveis)

    print(f"     Registros após imputação: {len(X):,}")

    # ── 3. Trata outliers ─────────────────────────────────────────────
    X = tratar_outliers(X, cols_disponiveis)

    # ── 4. Normalização ───────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 5. PCA ────────────────────────────────────────────────────────
    pca_full = PCA().fit(X_scaled)
    var_acum = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(var_acum, PCA_VARIANCE) + 1)
    n_comp = max(2, n_comp)  # mínimo 2 para visualização

    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print(f"     PCA: {n_comp} componentes → "
          f"{var_acum[n_comp-1]*100:.1f}% da variância explicada")

    # Salva gráfico de variância
    _plot_variancia_pca(pca_full.explained_variance_ratio_, n_comp)

    # DataFrame com nomes de componentes
    cols_pca = [f"PC{i+1}" for i in range(n_comp)]
    df_pca = pd.DataFrame(X_pca, columns=cols_pca, index=df_raw.index)

    # Repassa colunas de identificação
    # Usa reset_index para garantir alinhamento e iloc[:,0] para pegar só 1 coluna
    # caso o merge tenha criado colunas duplicadas (ex: CLASSE_x, CLASSE_y)
    df_r = df_raw.reset_index(drop=True)
    def _col1d(df, c, n):
        if c not in df.columns:
            return [''] * n
        col = df[c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return col.values
    n = len(df_pca)
    df_pca['CNPJ_FUNDO']   = _col1d(df_r, 'CNPJ_FUNDO',   n)
    df_pca['DENOM_SOCIAL'] = _col1d(df_r, 'DENOM_SOCIAL', n)
    df_pca['CLASSE']       = _col1d(df_r, 'CLASSE',       n)
    df_pca['TP_FUNDO']     = _col1d(df_r, 'TP_FUNDO',     n)

    return df_pca, scaler, pca


def _plot_variancia_pca(var_ratio, n_comp):
    fig, ax = plt.subplots(figsize=(8, 4))
    var_acum = np.cumsum(var_ratio)
    ax.bar(range(1, len(var_ratio) + 1), var_ratio * 100,
           alpha=0.6, color='steelblue', label='Individual')
    ax.plot(range(1, len(var_ratio) + 1), var_acum * 100,
            marker='o', color='tomato', label='Acumulada')
    ax.axvline(x=n_comp, linestyle='--', color='green',
               label=f'{n_comp} componentes selecionados')
    ax.axhline(y=95, linestyle=':', color='gray', label='95% variância')
    ax.set_xlabel('Componente Principal')
    ax.set_ylabel('Variância Explicada (%)')
    ax.set_title('Variância Explicada pelo PCA')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/pca_variancia.png", dpi=150)
    plt.close()