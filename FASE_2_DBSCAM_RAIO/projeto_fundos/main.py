"""
Projeto Final - Mineração de Dados Aplicada a Finanças
FASE 2 - DBSCAN com raio ajustado pelo colega
"""
import os
import warnings
warnings.filterwarnings('ignore')

# CORREÇÃO: garante que resultados vão para a pasta do próprio main.py
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.coleta import coletar_dados
from src.preprocessamento import preprocessar, FEATURE_COLS
from src.clustering import executar_clustering
from src.avaliacao import avaliar_clusters
from src.visualizacao import gerar_visualizacoes
from src.outliers import identificar_outliers_zscore, plotar_evidencia_outliers, plotar_mapa_outliers

def main():
    print("=" * 60)
    print(" SEGMENTAÇÃO DE FUNDOS DE INVESTIMENTO - CVM")
    print("=" * 60)

    print("\n[1/5] Coletando dados da CVM...")
    df_raw = coletar_dados()
    print(f"     Dados coletados: {df_raw.shape[0]} fundos, {df_raw.shape[1]} colunas")

    print("\n[1.5/5] Identificando outliers antes do tratamento...")
    identificar_outliers_zscore(df_raw, FEATURE_COLS)

    print("\n[2/5] Pré-processando dados...")
    df_features, scaler, pca = preprocessar(df_raw)
    print(f"     Features finais: {df_features.shape[1]} dimensões após PCA")

    print("     Gerando visualizações detalhadas de outliers...")
    for col in ['volatilidade', 'retorno_acumulado']:
        if col in df_raw.columns:
            plotar_evidencia_outliers(df_raw, col)
            plotar_mapa_outliers(df_features, df_raw, col)

    print("\n[3/5] Executando algoritmos de clustering...")
    resultados = executar_clustering(df_features)

    print("\n[4/5] Avaliando resultados...")
    metricas = avaliar_clusters(df_features, resultados)

    print("\n[5/5] Gerando visualizações...")
    gerar_visualizacoes(df_raw, df_features, resultados, metricas)

    print("\n" + "=" * 60)
    print(" CONCLUÍDO! Resultados salvos em: resultados/")
    print("=" * 60)

if __name__ == "__main__":
    main()