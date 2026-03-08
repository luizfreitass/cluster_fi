"""
Módulo de coleta de dados da CVM
"""

import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_URL_DIARIO = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS"
BASE_URL_CAD    = "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv"
DATA_DIR        = "data/raw"
MESES_COLETA    = 12


def _cnpj_limpo(serie):
    return serie.astype(str).str.replace(r'\D', '', regex=True).str.strip()


def _baixar_csv_zip(url):
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            nome_csv = [n for n in z.namelist() if n.endswith('.csv')][0]
            with z.open(nome_csv) as f:
                return pd.read_csv(f, sep=';', encoding='latin1', low_memory=False)
    except Exception as e:
        print(f"     Aviso: falha ao baixar {url} -> {e}")
        return None


def coletar_informes_diarios():
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = []
    hoje = datetime.today()

    for i in range(1, MESES_COLETA + 1):
        alvo = hoje - timedelta(days=30 * i)
        aamm = alvo.strftime("%Y%m")
        cache_path = os.path.join(DATA_DIR, f"inf_diario_{aamm}.csv.gz")

        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, sep=';', encoding='latin1',
                             low_memory=False, compression='gzip')
        else:
            url = f"{BASE_URL_DIARIO}/inf_diario_fi_{aamm}.zip"
            print(f"     Baixando {aamm}...", end=" ")
            df = _baixar_csv_zip(url)
            if df is None:
                print("nao encontrado, pulando.")
                continue
            df.to_csv(cache_path, sep=';', index=False,
                      encoding='latin1', compression='gzip')
            print(f"ok ({len(df):,} registros)")

        frames.append(df)

    if not frames:
        raise RuntimeError("Nenhum arquivo de informe diario foi baixado.")

    return pd.concat(frames, ignore_index=True)


def coletar_cadastro():
    cache_path = os.path.join(DATA_DIR, "cad_fi.csv.gz")
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, sep=';', encoding='latin1',
                         low_memory=False, compression='gzip')
    else:
        print("     Baixando cadastro de fundos...", end=" ")
        resp = requests.get(BASE_URL_CAD, timeout=120)
        df = pd.read_csv(
            io.StringIO(resp.content.decode('latin1')),
            sep=';', low_memory=False
        )
        df.to_csv(cache_path, sep=';', index=False,
                  encoding='latin1', compression='gzip')
        print(f"ok ({len(df):,} fundos)")

    return df


def coletar_dados():
    inf = coletar_informes_diarios()

    print(f"     Colunas brutas: {list(inf.columns)}")

    # Detecta coluna CNPJ
    cnpj_col = next((c for c in inf.columns if 'CNPJ' in c.upper()), None)
    if cnpj_col is None:
        raise KeyError(f"Coluna CNPJ nao encontrada. Colunas: {list(inf.columns)}")
    inf = inf.rename(columns={cnpj_col: 'CNPJ_FUNDO'})

    # Renomeia colunas que mudaram de nome na CVM
    renomear = {'CAPTC_DIA':'VL_CAPTC_DIA', 'RESG_DIA':'VL_RESG_DIA',
                'TP_FUNDO_CLASSE':'TP_FUNDO'}
    inf = inf.rename(columns={k:v for k,v in renomear.items() if k in inf.columns})

    # Normaliza CNPJ
    inf['CNPJ_FUNDO'] = _cnpj_limpo(inf['CNPJ_FUNDO'])

    for c in ['VL_TOTAL','VL_PATRIM_LIQ','VL_QUOTA','VL_CAPTC_DIA','VL_RESG_DIA','NR_COTST']:
        if c in inf.columns:
            inf[c] = pd.to_numeric(inf[c], errors='coerce')

    inf['DT_COMPTC'] = pd.to_datetime(inf['DT_COMPTC'], errors='coerce')

    print(f"     Total registros: {len(inf):,} | CNPJs unicos: {inf['CNPJ_FUNDO'].nunique():,}")

    # Features por fundo
    inf = inf.sort_values(['CNPJ_FUNDO', 'DT_COMPTC'])
    inf['retorno_diario'] = inf.groupby('CNPJ_FUNDO')['VL_QUOTA'].pct_change()

    grp = inf.groupby('CNPJ_FUNDO')

    agg = {
        'retorno_medio_diario' : grp['retorno_diario'].mean(),
        'volatilidade'         : grp['retorno_diario'].std(),
        'pl_medio'             : grp['VL_PATRIM_LIQ'].mean(),
        'dias_reportados'      : grp['DT_COMPTC'].count(),
    }
    if 'VL_CAPTC_DIA' in inf.columns:
        agg['captacao_media'] = grp['VL_CAPTC_DIA'].mean()
        agg['resgate_medio']  = grp['VL_RESG_DIA'].mean()
    if 'NR_COTST' in inf.columns:
        agg['cotistas_medio'] = grp['NR_COTST'].mean()

    features = pd.DataFrame(agg).reset_index()

    # Retorno acumulado e max drawdown separados (evita problema de tipo)
    features['retorno_acumulado'] = grp['retorno_diario'].apply(
        lambda x: float((1 + x.fillna(0)).prod() - 1)
    ).values

    features['max_drawdown'] = grp['retorno_diario'].apply(_max_drawdown).values

    # Preenche colunas opcionais se nao existirem
    for col in ['captacao_media','resgate_medio','cotistas_medio']:
        if col not in features.columns:
            features[col] = 0.0

    features['fluxo_liquido_medio'] = features['captacao_media'] - features['resgate_medio']
    features['sharpe'] = features['retorno_medio_diario'] / features['volatilidade'].replace(0, np.nan)

    print(f"     Fundos com features: {len(features):,}")

    # Cadastro
    cad = coletar_cadastro()
    print(f"     Colunas cadastro: {list(cad.columns)}")

    cnpj_col_cad = next((c for c in cad.columns if 'CNPJ' in c.upper()), None)
    if cnpj_col_cad:
        cad = cad.rename(columns={cnpj_col_cad: 'CNPJ_FUNDO'})
        cad['CNPJ_FUNDO'] = _cnpj_limpo(cad['CNPJ_FUNDO'])

    renomear_cad = {'TP_FUNDO_CLASSE':'TP_FUNDO', 'CLASSE_ANBIMA':'CLASSE',
                    'NM_FUNDO':'DENOM_SOCIAL', 'SITUACAO':'SIT'}
    cad = cad.rename(columns={k:v for k,v in renomear_cad.items() if k in cad.columns})

    for col in ['CNPJ_FUNDO','DENOM_SOCIAL','TP_FUNDO','CLASSE','SIT']:
        if col not in cad.columns:
            cad[col] = ''

    cad_sel = cad[['CNPJ_FUNDO','DENOM_SOCIAL','TP_FUNDO','CLASSE','SIT']].drop_duplicates('CNPJ_FUNDO')
    print(f"     CNPJs no cadastro: {cad_sel['CNPJ_FUNDO'].nunique():,}")

    df = features.merge(cad_sel, on='CNPJ_FUNDO', how='left')
    print(f"     Apos merge: {len(df):,} | com SIT preenchido: {df['SIT'].fillna('').ne('').sum():,}")


    # Nao filtra por SIT — fundos encerrados param de reportar naturalmente
    # Filtrar por SIT causaria perda de dados por falha no merge de CNPJ

    df = df[df['dias_reportados'] >= 20]

    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    print(f"     Fundos finais: {len(df):,}")
    return df.reset_index(drop=True)


def _max_drawdown(retornos):
    retornos = retornos.fillna(0)
    curva = (1 + retornos).cumprod()
    pico  = curva.cummax()
    dd    = (curva - pico) / pico
    return float(dd.min())