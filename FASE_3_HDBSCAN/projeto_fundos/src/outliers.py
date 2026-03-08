"""
Módulo de Detecção e Visualização de Outliers
Método: Z-Score (threshold padrão = 3)
Gera:
  - CSV com fundos anômalos (resultados/auditoria/fundos_anomalos.csv)
  - Boxplots por feature
  - Boxplots anotados com nomes dos fundos mais extremos
  - Mapa PCA destacando onde os outliers se localizam
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "resultados/auditoria"
os.makedirs(RESULTS_DIR, exist_ok=True)


def identificar_outliers_zscore(df: pd.DataFrame, cols: list, threshold: float = 3):
    """
    Identifica outliers via Z-Score antes do tratamento (winsorização).
    Gera um relatório CSV e um boxplot simples por feature.

    Parâmetros
    ----------
    df        : DataFrame com as features brutas (saída do coletar_dados)
    cols      : lista de colunas a analisar (FEATURE_COLS do preprocessamento)
    threshold : número de desvios padrão para considerar outlier (padrão=3)

    Retorna
    -------
    df_report : DataFrame com todas as anomalias encontradas
    """
    df_temp = df.copy()
    df_temp['DENOM_SOCIAL'] = df_temp['DENOM_SOCIAL'].fillna('NOME NAO ENCONTRADO') \
        if 'DENOM_SOCIAL' in df_temp.columns else 'NOME NAO ENCONTRADO'

    # Filtra apenas colunas que existem no DataFrame
    cols_validas = [c for c in cols if c in df_temp.columns]

    outliers_report = []

    for col in cols_validas:
        serie = pd.to_numeric(df_temp[col], errors='coerce')
        media = serie.mean()
        desvio = serie.std()

        if desvio == 0 or pd.isna(desvio):
            continue

        z_scores = (serie - media) / desvio
        mask_out  = z_scores.abs() > threshold
        df_out    = df_temp[mask_out]

        for idx, row in df_out.iterrows():
            outliers_report.append({
                'CNPJ'     : row.get('CNPJ_FUNDO', ''),
                'Fundo'    : row.get('DENOM_SOCIAL', 'NOME NAO ENCONTRADO'),
                'Variavel' : col,
                'Valor'    : row[col],
                'Z-Score'  : round(float(z_scores[idx]), 4),
            })

        # Boxplot simples por feature
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=serie.dropna(), color='lightcoral', ax=ax)
        ax.set_title(f"Evidência de Outliers: {col}\n"
                     f"(Z-Score > {threshold} → {mask_out.sum()} fundos)")
        ax.set_xlabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"boxplot_{col}.png"), dpi=150)
        plt.close()

    df_report = pd.DataFrame(outliers_report)
    path_csv  = os.path.join(RESULTS_DIR, "fundos_anomalos.csv")
    df_report.to_csv(path_csv, index=False)

    print(f"     [Auditoria] {len(df_report)} anomalias detectadas em "
          f"{df_report['Variavel'].nunique() if not df_report.empty else 0} features.")
    print(f"     Relatório salvo em: {path_csv}")

    return df_report


def plotar_evidencia_outliers(df: pd.DataFrame, col: str, threshold: float = 3):
    """
    Boxplot anotado com os nomes dos 5 fundos mais extremos acima do threshold.

    Parâmetros
    ----------
    df        : DataFrame com features brutas
    col       : feature a analisar
    threshold : Z-Score mínimo para considerar outlier
    """
    if col not in df.columns:
        print(f"     [Aviso] Coluna '{col}' não encontrada, pulando.")
        return

    serie    = pd.to_numeric(df[col], errors='coerce')
    z_scores = (serie - serie.mean()) / serie.std()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=serie.dropna(), color='lightgray', fliersize=5, ax=ax)

    # Top 5 outliers positivos
    top_outliers = (
        df[z_scores > threshold]
        .assign(_zscore=z_scores)
        .nlargest(5, col)
    )

    for _, row in top_outliers.iterrows():
        nome = str(row['DENOM_SOCIAL']) \
            if 'DENOM_SOCIAL' in row and pd.notnull(row['DENOM_SOCIAL']) \
            else 'Sem Nome'
        nome = nome[:25]   # trunca para caber no gráfico

        ax.annotate(
            nome,
            xy=(row[col], 0),
            xytext=(row[col], 0.15),
            arrowprops=dict(facecolor='crimson', shrink=0.05,
                            width=1, headwidth=5),
            fontsize=8,
            rotation=40,
            ha='center',
        )

    ax.set_title(f"Detecção de Outliers: {col}\n(Z-Score > {threshold})")
    ax.set_xlabel(col)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"boxplot_anotado_{col}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"     Boxplot anotado salvo: {path}")


def plotar_mapa_outliers(df_pca: pd.DataFrame, df_raw: pd.DataFrame,
                          col_referencia: str, threshold: float = 3):
    """
    Scatter PC1 × PC2 destacando onde os outliers de uma feature se
    localizam no espaço PCA.

    Parâmetros
    ----------
    df_pca         : DataFrame com componentes PCA (saída do preprocessar)
    df_raw         : DataFrame com features brutas (saída do coletar_dados)
    col_referencia : feature cujos outliers serão destacados
    threshold      : Z-Score mínimo para considerar outlier
    """
    if col_referencia not in df_raw.columns:
        print(f"     [Aviso] Coluna '{col_referencia}' não encontrada, pulando.")
        return

    if 'PC1' not in df_pca.columns or 'PC2' not in df_pca.columns:
        print("     [Aviso] PC1/PC2 não encontrados no df_pca, pulando mapa.")
        return

    serie    = pd.to_numeric(df_raw[col_referencia], errors='coerce')
    z_scores = (serie - serie.mean()) / serie.std()
    is_out   = z_scores.abs() > threshold

    # Alinha índices — df_pca pode ter sido resetado
    idx_pca = df_pca.index
    is_out  = is_out.reindex(idx_pca, fill_value=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        df_pca.loc[~is_out, 'PC1'],
        df_pca.loc[~is_out, 'PC2'],
        c='lightgray', alpha=0.4, s=8, label='Fundo Normal'
    )
    ax.scatter(
        df_pca.loc[is_out, 'PC1'],
        df_pca.loc[is_out, 'PC2'],
        c='crimson', alpha=0.85, s=25,
        label=f'Outlier de {col_referencia} (n={is_out.sum()})'
    )
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    ax.set_title(f'Localização dos Outliers de {col_referencia}\nno Espaço PCA')
    ax.legend(markerscale=2)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"mapa_outliers_{col_referencia}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"     Mapa de outliers salvo: {path}")