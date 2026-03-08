"""
Projeto Final - Mineração de Dados Aplicada a Finanças
FASE 3 - Avatar: HDBSCAN + análise clusters minoritários
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
from src.outliers import (identificar_outliers_zscore,
                           plotar_evidencia_outliers,
                           plotar_mapa_outliers)

def main():
    print("=" * 60)
    print(" SEGMENTAÇÃO DE FUNDOS DE INVESTIMENTO - CVM")
    print("=" * 60)

    # ── 1. Coleta ──────────────────────────────────────────────────
    print("\n[1/6] Coletando dados da CVM...")
    df_raw = coletar_dados()
    print(f"     Dados coletados: {df_raw.shape[0]} fundos, {df_raw.shape[1]} colunas")

    # ── 1.5. Auditoria de Outliers (pré-tratamento) ────────────────
    print("\n[1.5/6] Auditoria de outliers (dados brutos, antes da winsorização)...")
    identificar_outliers_zscore(df_raw, FEATURE_COLS)

    # ── 2. Pré-processamento ───────────────────────────────────────
    print("\n[2/6] Pré-processando dados...")
    df_features, scaler, pca = preprocessar(df_raw)
    print(f"     Features finais: {df_features.shape[1]} dimensões após PCA")

    print("     Gerando visualizações de outliers...")
    for col in ['volatilidade', 'retorno_acumulado', 'pl_medio', 'sharpe']:
        plotar_evidencia_outliers(df_raw, col)
        plotar_mapa_outliers(df_features, df_raw, col)

    # ── 3. Clustering ──────────────────────────────────────────────
    print("\n[3/6] Executando algoritmos de clustering...")
    resultados = executar_clustering(df_features, df_raw=df_raw)

    # ── 4. Avaliação ───────────────────────────────────────────────
    print("\n[4/6] Avaliando resultados...")
    metricas = avaliar_clusters(df_features, resultados)

    # ── 5. Visualização ────────────────────────────────────────────
    print("\n[5/6] Gerando visualizações...")
    gerar_visualizacoes(df_raw, df_features, resultados, metricas)

    print("\n" + "=" * 60)
    print(" CONCLUÍDO! Resultados salvos em: resultados/")
    print("  → Clustering:  resultados/")
    print("  → Auditoria:   resultados/auditoria/")
    print("=" * 60)

if __name__ == "__main__":
    main()