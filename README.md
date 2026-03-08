# ClusterFI — Segmentação de Fundos de Investimento Brasileiros via Clustering

**Instituição:** Universidade Federal de Uberlândia (UFU)  
**Autores:** Artur Batalini Coelho Alvarim, Artur Gabriel Pereira Pedrosa, Lucas Duarte Soares, Luiz Alexandre Anchieta Freitas  
**Contato:** `{artur.alvarim, artur15, duartelucas03, luiz.freitass}@ufu.br`

---

## Sobre o Projeto

O **ClusterFI** aplica Mineração de Dados para segmentar os **27.867 fundos de investimento**
cadastrados na CVM (Comissão de Valores Mobiliários), utilizando uma janela de 12 meses de
informes diários (2025–2026).

O problema central investigado é a **ineficiência das classificações oficiais da CVM**: fundos
rotulados como "Multimercado" ou "Ações" que, comportamentalmente, operam de forma idêntica
à Renda Fixa — os chamados *closet indexers* (falsos ativos) — cobram taxas de gestão ativa sem entregá-la.
O projeto expõe esse fenômeno com evidências computacionais e constrói um **radar preditivo
de anomalias institucionais** como produto aplicado.

A metodologia avança em **três fases evolutivas**:

- **Fase 1** — Baselines particionais: K-Means, Agrupamento Hierárquico de Ward e DBSCAN colapsado
- **Fase 2** — DBSCAN: Ajuste manual de parâmetros e refatoração do DBSCAN
- **Fase 3** — HDBSCAN: modelo final com densidade variável, superior em todas as métricas

Adicionalmente, um **experimento de controle** executa o pipeline sem PCA para quantificar
o impacto da redução de dimensionalidade nos resultados.

---

## Resultados

| Fase | Algoritmo | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabász ↑ | Clusters |
|---|---|---|---|---|---|
| Fase 1 | K-Means (K=4) | 0,7453 | 0,8041 | 13.527 | 4 |
| Fase 1 | Ward (K=4) | 0,6836 | 1,2591 | 11.882 | 4 |
| Fase 1 | DBSCAN (auto) | — | — | — | 1 (colapso) |
| Fase 2 | DBSCAN (eps=1,5) | 0,8205 | 0,2049 | 853 | 3 + ruído |
| **Fase 3** | **HDBSCAN** | **0,8455** | **0,3966** | **51.492** | **2 + ruído** |
| Controle | HDBSCAN sem PCA | 0,8303 | 0,3184 | 42.025 | 2 + ruído |

**Perfil dos clusters — HDBSCAN Fase 3:**

| Cluster | Fundos | Volatilidade | Sharpe | PL Médio | Diagnóstico |
|---|---|---|---|---|---|
| 0 – Massa Conservadora | 21.617 (77,6%) | 0,0051 | 1,17 | R$ 89,1 M | *Closet indexers* confirmados |
| 1 – Risco Extremo | 1.544 (5,5%) | 506,02 | −0,12 | R$ 16,2 M | Fundos em colapso |
| −1 – Ruído (Anomalias) | 4.706 (16,9%) | N/D | N/D | N/D | Radar preditivo B2B |

---

## Estrutura do Projeto
```
CLUSTER_FI/
├── FASE_1_KMEANS_HIER/
│   └── projeto_fundos/
│       ├── data/          # Dados de entrada da Fase 1
│       ├── resultados/    # Outputs: métricas, gráficos, CSVs
│       ├── src/           # Módulos auxiliares da Fase 1
│       └── main.py        # Ponto de entrada da Fase 1
│
├── FASE_2_DBSCAN_RAIO/
│   └── projeto_fundos/
│       ├── data/          # Dados de entrada da Fase 2
│       ├── resultados/    # Outputs: métricas, gráficos, CSVs
│       ├── src/           # Módulos auxiliares da Fase 2
│       └── main.py        # Ponto de entrada da Fase 2
│
├── FASE_3_HDBSCAN/
│   └── projeto_fundos/
│       ├── data/          # Dados de entrada da Fase 3
│       ├── resultados/    # Outputs: métricas, gráficos, CSVs
│       ├── src/           # Módulos auxiliares da Fase 3
│       └── main.py        # Ponto de entrada da Fase 3
│
├── SEM_PCA/
│   └── projeto_fundos/
│       ├── data/          # Dados de entrada do experimento de controle
│       ├── resultados/    # Outputs: métricas, gráficos, CSVs
│       ├── src/           # Módulos auxiliares
│       └── Main3sempca.py # Ponto de entrada do experimento de controle
│
└── requirements.txt       # Dependências do projeto
```

---

## Pré-requisitos

- Python **3.10** ou superior
- pip **23** ou superior
- Git

---

## Instalação

### 1. Clone o repositório
```bash
git clone https://github.com/luizfreitass/cluster_fi.git
cd ClusterFI
```

### 2. Crie e ative um ambiente virtual
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> O ambiente virtual isola as dependências do projeto do Python do sistema.
> Sempre ative-o antes de executar qualquer script: `source .venv/bin/activate`

### 3. Instale as dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Dependências instaladas:**

| Pacote | Versão mínima | Finalidade |
|---|---|---|
| numpy | 1.26 | Operações vetoriais |
| pandas | 2.2 | Manipulação de dados |
| scikit-learn | 1.4 | K-Means, Ward, PCA, métricas |
| hdbscan | 0.8.33 | Modelo final Fase 3 |
| matplotlib | 3.8 | Visualizações |
| seaborn | 0.13 | Visualizações estatísticas |
| scipy | 1.12 | Cálculos científicos e dendrograma |
| requests | 2.31 | Download dos dados da CVM |
| tqdm | 4.66 | Barras de progresso |
| joblib | 1.3 | Paralelismo (HDBSCAN e sklearn) |
| pyarrow | 15.0 | Leitura/escrita Parquet |
| openpyxl | 3.1 | Leitura/escrita Excel |

> **Fedora 42:** caso ocorra erro de compilação do `hdbscan`, instale as dependências de sistema:
> ```bash
> sudo dnf install gcc gcc-c++ python3-devel
> ```
> Em seguida, repita o `pip install -r requirements.txt`.

> **Ubuntu 24.04:** caso necessário:
> ```bash
> sudo apt install build-essential python3-dev
> ```
> Em seguida, repita o `pip install -r requirements.txt`.

---

## Execução — Passo a Passo

> **Atenção:** execute as fases sempre na ordem apresentada.
> Cada fase gera os outputs em sua respectiva pasta `resultados/`,
> que serão utilizados para comparação nas fases seguintes.

---

### Fase 1 — K-Means, Agrupamento Hierárquico de Ward e DBSCAN colapsado

Esta fase aplica os dois algoritmos particionais para estabelecer a estrutura
macro do mercado e servir de baseline comparativo para as fases seguintes.

**O que será executado:**
- Pré-processamento completo: Z-Score, Winsorização, SimpleImputer, StandardScaler e PCA
- K-Means com seleção automática de K (testando K=2 a 12 via cotovelo + Silhouette)
- Agrupamento Hierárquico de Ward com geração de dendrograma
- DBSCAN com seleção automática de eps via K-Distance Graph
- Cálculo das métricas: Silhouette, Davies-Bouldin e Calinski-Harabász
- Geração das projeções t-SNE para visualização dos agrupamentos
```bash
cd FASE_1_KMEANS_HIER/projeto_fundos
python main.py
```

**Analisando os resultados:**
```bash
ls resultados/
```

> **O que observar:**
> O K-Means deve concentrar ~88,9% dos fundos em um único cluster,
> evidenciando que algoritmos particionais revelam a estrutura macro
> mas não conseguem isolar anomalias nem *closet indexers*.
> O eps automático colapsa todo o mercado em 1 único cluster.
> Anote os valores de Silhouette e CH — serão o baseline de comparação.

---

### Fase 2 — DBSCAN e Diagnóstico da Falha Paramétrica

Essa é a fase de refatoração do DBSCAN, em que, ao ajustar os parâmetros,
foram gerados 3 cluster, bem definidos, com métricas razoáveis de avaliação.

**O que será executado:**

- DBSCAN com ajuste manual (eps=1,5; min_samples=20)
- Cálculo das métricas e comparação com a Fase 1
- Geração do K-Distance Graph e projeções t-SNE
```bash
cd ../../FASE_2_DBSCAN_RAIO/projeto_fundos
python main.py
```

**Analisando os resultados:**
```bash
ls resultados/
```

> **O que observar:**

> O ajuste manual produz Silhouette=0,8205 (aparente melhora),
> mas o CH cai para 853 — contradição que expõe a limitação estrutural:

---

### Fase 3 — HDBSCAN (Modelo Final)

Esta é a fase principal. O HDBSCAN opera com densidade variável,
eliminando a necessidade de raio fixo e superando todos os algoritmos
anteriores simultaneamente.

**O que será executado:**
- HDBSCAN com min_cluster_size=500
- Cálculo das métricas de validação interna
- Geração das projeções t-SNE e mapa PCA
- Análise de perfil financeiro dos clusters
- Cruzamento dos agrupamentos com classificações CVM
- Geração dos arquivos `labels_hdbscan.csv` e `perfil_hdbscan.csv`
```bash
cd ../../FASE_3_HDBSCAN/projeto_fundos
python main.py
```

**Analisando os resultados:**
```bash
ls resultados/
```

> **O que observar:**
> Silhouette=0,8455 | Davies-Bouldin=0,3966 | CH=51.492 (+280,7% vs Fase 1).
> O Cluster 0 reúne 21.617 fundos (77,6%) com volatilidade 0,0051 —
> comportamento idêntico à Renda Fixa independente do rótulo CVM.
> O Label −1 isola 4.706 fundos (16,9%) como radar preditivo de anomalias.

---

### Experimento de Controle — Sem PCA

Executa o pipeline completo sem a etapa de redução de dimensionalidade (PCA),
mantendo todas as outras etapas idênticas.
Objetivo: quantificar o impacto real do PCA e validá-lo como pré-condição necessária.

**O que será executado:**
- Pipeline idêntico à Fase 3, removendo apenas o PCA
- Cálculo das métricas e comparação direta com a Fase 3
- Verificação de estabilidade de rótulos entre as execuções
```bash
cd ../../SEM_PCA/projeto_fundos
python Main3sempca.py
```

**Analisando os resultados:**
```bash
ls resultados/
```

> **O que observar:**
> CH cai de 51.492 para 42.025 (−18,4%) e o Davies-Bouldin piora.
> A remoção do PCA também causa inversão de rótulos entre clusters —
> evidência direta de instabilidade estrutural.
> Conclusão: o PCA não é otimização opcional, é pré-condição necessária.

---

## Comparativo Final

Após executar todas as fases, os resultados ficam disponíveis em:
```
FASE_1_KMEANS_HIER/projeto_fundos/resultados/
FASE_2_DBSCAN_RAIO/projeto_fundos/resultados/
FASE_3_HDBSCAN/projeto_fundos/resultados/
SEM_PCA/projeto_fundos/resultados/
```

Resumo esperado das métricas:

| Fase | Algoritmo | Silhouette ↑ | D-Bouldin ↓ | CH ↑ |
|---|---|---|---|---|
| Fase 1 | K-Means (K=4) | 0,7453 | 0,8041 | 13.527 |
| Fase 1 | Ward (K=4) | 0,6836 | 1,2591 | 11.882 |
| Fase 2 | DBSCAN (eps=1,5) | 0,8205 | 0,2049 | 853 |
| **Fase 3** | **HDBSCAN** | **0,8455** | **0,3966** | **51.492** |
| Controle | HDBSCAN sem PCA | 0,8303 | 0,3184 | 42.025 |

---

## Referências Bibliográficas

* **Bellman, R.** (1961) *Adaptive Control Processes: A Guided Tour*. Princeton University Press.
* **Berk, J. B. and Green, R. C.** (2004) "Mutual Fund Flows and Performance in Rational Markets", *Journal of Political Economy*, v. 112, n. 6, p. 1269–1295.
* **Caliński, T. and Harabász, J.** (1974) "A dendrite method for cluster analysis", *Communications in Statistics*, v. 3, n. 1, p. 1–27.
* **Campello, R. J. G. B., Moulavi, D. and Sander, J.** (2013) "Density-Based Clustering Based on Hierarchical Density Estimates", In: *Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)*, Lecture Notes in Computer Science, v. 7819, p. 160–172. Springer.
* **Castro, L.** (2023) *Segmentação de empresas do Ibovespa utilizando o algoritmo K-Means*. Trabalho de Conclusão de Curso (Graduação em Engenharia de Produção) – Universidade de São Paulo (USP), São Carlos.
* **Cremers, K. J. M. and Petajisto, A.** (2009) "How Active Is Your Fund Manager? A New Measure That Predicts Performance", *The Review of Financial Studies*, v. 22, n. 9, p. 3329–3365.
* **Davies, D. L. and Bouldin, D. W.** (1979) "A Cluster Separation Measure", *IEEE Transactions on Pattern Analysis and Machine Intelligence*, v. 1, n. 2, p. 224–227.
* **Ester, M., Kriegel, H.-P., Sander, J. and Xu, X.** (1996) "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise", In: *Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining (KDD)*, p. 226–231. AAAI Press.
* **Freitas, L. M. and Carneiro, M. G.** (2019) "Community Detection to Invariant Pattern Clustering in Images", In: *Brazilian Conference on Intelligent Systems (BRACIS)*.
* **Grubbs, F. E.** (1969) "Procedures for Detecting Outlying Observations in Samples", *Technometrics*, v. 11, n. 1, p. 1–21.
* **Jain, A. K.** (2010) "Data clustering: 50 years beyond K-means", *Pattern Recognition Letters*, v. 31, n. 8, p. 651–666.
* **Lloyd, S. P.** (1982) "Least squares quantization in PCM", *IEEE Transactions on Information Theory*, v. 28, n. 2, p. 129–137.
* **McInnes, L., Healy, J. and Astels, S.** (2017) "hdbscan: Hierarchical density based clustering", *Journal of Open Source Software*, v. 2, n. 11, p. 205.
* **Oliveira, R. E.** (2020) *Análise do mercado acionário brasileiro através de técnica de clusterização*. Trabalho de Conclusão de Curso (Graduação em Estatística) – Universidade Federal de Uberlândia, Uberlândia.
* **Rousseeuw, P. J.** (1987) "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis", *Journal of Computational and Applied Mathematics*, v. 20, p. 53–65.
* **Van der Maaten, L. and Hinton, G.** (2008) "Visualizing Data using t-SNE", *Journal of Machine Learning Research*, v. 9, p. 2579–2605.
* **Ward, J. H.** (1963) "Hierarchical Grouping to Optimize an Objective Function", *Journal of the American Statistical Association*, v. 58, n. 301, p. 236–244.
* **Wilcox, R. R.** (2005) *Introduction to Robust Estimation and Hypothesis Testing*. 2. ed. Academic Press.
* **Wold, S., Esbensen, K. and Geladi, P.** (1987) "Principal component analysis", *Chemometrics and Intelligent Laboratory Systems*, v. 2, n. 1–3, p. 37–52.

---

