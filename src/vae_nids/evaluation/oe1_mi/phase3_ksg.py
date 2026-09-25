"""Fase 3 (parte KSG) -- IM KSG (k=3) de los 2,701 pares sobre una
submuestra de 50,000 filas del dataset EDA (semilla 42). Es la parte cara,
por eso va en su propio script y se cachea en metrics/oe1_ksg_pares.csv.

IM KSG es simétrica en teoría pero el estimador no lo es exactamente
(sklearn escala y agrega ruido a X e y por separado); se calcula una sola
vez por par, con la feature de menor índice como objetivo.

Uso: python -m vae_nids.evaluation.oe1_mi.phase3_ksg
"""
import time

import numpy as np
import pandas as pd

from vae_nids.evaluation.oe1_mi import config as cfg
from vae_nids.evaluation.oe1_mi import data
from vae_nids.evaluation.oe1_mi import estimators as est

OUT_PATH = cfg.METRICS_DIR / "oe1_ksg_pares.csv"


def main(n_sub: int = cfg.KSG_SUBSAMPLE) -> pd.DataFrame:
    x, _ = data.load_eda_matrix()
    rng = np.random.default_rng(cfg.SEED)
    sub = np.sort(rng.choice(x.shape[0], size=n_sub, replace=False))
    xs = x[sub]
    feats = cfg.FEATURES_74
    rows = []
    t0 = time.perf_counter()
    for i in range(len(feats) - 1):
        mi = est.mi_ksg_pairwise(xs[:, i + 1:], xs[:, i], seed=cfg.SEED,
                                 n_neighbors=cfg.KSG_N_NEIGHBORS, n_jobs=-1)
        for off, v in enumerate(mi):
            j = i + 1 + off
            rows.append({"feature_a": feats[i], "feature_b": feats[j],
                         "ksg_mi_nats": float(v), "ksg_r_info": float(est.linfoot(v))})
        print(f"[ksg] {i + 1}/{len(feats) - 1} {feats[i]} ({time.perf_counter() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.attrs["n_sub"] = n_sub
    df.to_csv(OUT_PATH, index=False)
    print(f"[ksg] n_sub={n_sub}, {len(df)} pares, {time.perf_counter() - t0:.0f}s -> {OUT_PATH}")
    return df


if __name__ == "__main__":
    main()
