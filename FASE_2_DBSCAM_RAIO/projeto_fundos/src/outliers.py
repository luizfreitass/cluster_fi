import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "resultados/auditoria"
os.makedirs(RESULTS_DIR, exist_ok=True)

def identificar_outliers_zscore(df: pd.DataFrame, cols: list, threshold=3):
    """Identifica outliers antes do tratamento e gera relatório CSV."""
    outliers_report = []
    
    # Preenche nomes vazios para evitar erros de processamento
    df_temp = df.copy()
    df_temp['DENOM_SOCIAL'] = df_temp['DENOM_SOCIAL'].fillna('NOME NAO ENCONTRADO')
    
    for col in cols:
        z_scores = np.abs((df_temp[col] - df_temp[col].mean()) / df_temp[col].std())
        df_outliers = df_temp[z_scores > threshold]
        
        if not df_outliers.empty:
            for _, row in df_outliers.iterrows():
                outliers_report.append({
                    'CNPJ': row['CNPJ_FUNDO'],
                    'Fundo': row['DENOM_SOCIAL'],
                    'Variavel': col,
                    'Valor': row[col],
                    'Z-Score': z_scores[row.name]
                })
        
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df_temp[col], color='lightcoral')
        plt.title(f"Evidência de Outliers: {col}")
        plt.savefig(f"{RESULTS_DIR}/boxplot_{col}.png")
        plt.close()

    df_final_report = pd.DataFrame(outliers_report)
    path_csv = os.path.join(RESULTS_DIR, "fundos_anomalos.csv")
    df_final_report.to_csv(path_csv, index=False)
    
    print(f"     [Auditoria] {len(df_final_report)} anomalias encontradas.")
    return df_final_report

def plotar_evidencia_outliers(df: pd.DataFrame, col: str, threshold=3):
    """Gera um Boxplot anotando os nomes dos fundos mais extremos."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[col], color='lightgray', fliersize=5)
    
    # Identificar os Top 5 Outliers
    z_scores = (df[col] - df[col].mean()) / df[col].std()
    top_outliers = df[z_scores > threshold].nlargest(5, col)
    
    for i, row in top_outliers.iterrows():
        # Tratamento: Converte para string e lida com valores nulos (NaN)
        nome_fundo = str(row['DENOM_SOCIAL']) if pd.notnull(row['DENOM_SOCIAL']) else "Sem Nome"
        
        plt.annotate(nome_fundo[:20], # Agora o slice [:20] funciona sempre
                     xy=(row[col], 0), 
                     xytext=(row[col], 0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                     fontsize=8, rotation=45)

    plt.title(f"Detecção de Outliers: {col}\n(Z-Score > {threshold})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"boxplot_anotado_{col}.png"), dpi=150)
    plt.close()

def plotar_mapa_outliers(df_pca: pd.DataFrame, df_raw: pd.DataFrame, col_referencia: str, threshold=3):
    """Mostra onde os outliers de uma feature estão no mapa do PCA."""
    z_scores = np.abs((df_raw[col_referencia] - df_raw[col_referencia].mean()) / df_raw[col_referencia].std())
    is_outlier = z_scores > threshold

    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca.loc[~is_outlier, 'PC1'], df_pca.loc[~is_outlier, 'PC2'], 
                c='lightgray', alpha=0.5, label='Fundo Normal', s=10)
    
    plt.scatter(df_pca.loc[is_outlier, 'PC1'], df_pca.loc[is_outlier, 'PC2'], 
                c='red', alpha=0.8, label=f'Outlier de {col_referencia}', s=30)

    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title(f'Localização dos Outliers de {col_referencia} no Espaço PCA')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"mapa_outliers_{col_referencia}.png"), dpi=150)
    plt.close()