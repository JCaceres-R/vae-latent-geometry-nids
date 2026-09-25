"""Carga de datos para OE1-IM.

Conjunto principal = el MISMO sobre el que se calculó la matriz de Pearson
del EDA (notebooks/eda_cicids2017.ipynb, celda 28: `df_clean[feature_cols].corr()`):
  - los 5 CSV crudos leídos con `pd.read_csv` por defecto (int64/float64,
    celda 4 -- no el float32 de pipeline.load_and_merge);
  - `pipeline.sanitize` (inf -> NaN y dropna): 2,100,814 -> 2,100,021 filas;
  - SIN el filtro de tasas negativas de la Sección 7.2 (se aplicó después de
    la celda 28, y tampoco está en pipeline.py);
  - benigno + malicioso mezclados, sin escalar.
Se reimplementa la lectura (sin modificar pipeline.py) solo para leer las 74
columnas + Label y no las 85, lo que no cambia las filas descartadas: las
únicas columnas con NaN/inf (celda 9) están entre las 74.

Conjunto secundario = train_benign.parquet (lo que ve el VAE). Las 51 columnas
de entrada están escaladas con MinMax (afín creciente -> Pearson, Spearman y la
IM por cuantiles no cambian); las otras 23 están en crudo. Todo en float32.
"""
import numpy as np
import pandas as pd

from vae_nids import config as base_cfg
from vae_nids.evaluation.oe1_mi import config as cfg

EDA_CACHE = cfg.CACHE_DIR / "eda_df_clean_74.parquet"
EXPECTED_EDA_ROWS = 2_100_021


def build_eda_dataset(force: bool = False) -> pd.DataFrame:
    if EDA_CACHE.exists() and not force:
        return pd.read_parquet(EDA_CACHE)
    wanted = set(cfg.FEATURES_74) | {base_cfg.LABEL_COL}
    frames = []
    for fname in base_cfg.CSV_FILES:
        df_day = pd.read_csv(base_cfg.DATA_DIR / fname, low_memory=False,
                             usecols=lambda c: c.strip() in wanted)
        df_day.columns = df_day.columns.str.strip()
        print(f"[data] {fname}: {len(df_day):,} filas")
        frames.append(df_day)
    df = pd.concat(frames, ignore_index=True)
    del frames
    n_raw = len(df)
    num = df[cfg.FEATURES_74]
    bad = ~np.isfinite(num.to_numpy(dtype=np.float64)).all(axis=1)
    df = df.loc[~bad, cfg.FEATURES_74 + [base_cfg.LABEL_COL]].reset_index(drop=True)
    print(f"[data] sanitize: {n_raw:,} -> {len(df):,} filas (-{bad.sum():,})")
    if len(df) != EXPECTED_EDA_ROWS:
        raise RuntimeError(f"Se esperaban {EXPECTED_EDA_ROWS:,} filas (df_clean del EDA), "
                           f"hay {len(df):,}")
    df.to_parquet(EDA_CACHE, index=False)
    return df


def load_eda_matrix() -> tuple[np.ndarray, pd.Series]:
    df = build_eda_dataset()
    x = df[cfg.FEATURES_74].to_numpy(dtype=np.float64)
    return x, df[base_cfg.LABEL_COL]


def load_train_benign_matrix() -> np.ndarray:
    df = pd.read_parquet(base_cfg.OUTPUT_DIR / "train_benign.parquet", columns=cfg.FEATURES_74)
    return df[cfg.FEATURES_74].to_numpy(dtype=np.float64)
