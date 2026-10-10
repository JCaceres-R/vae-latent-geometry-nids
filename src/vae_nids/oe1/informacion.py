"""Fase 1 (3.7) del OE1 saneado -- entropía e información de lo que queda. Descriptivo: no corta nada.

Sobre train_benign (tras D4) y el conjunto congelado:
  - por columna: valores únicos, fracción de la moda, H en bits (cuantiles con átomos aislados, B = 20,
    el mismo estimador de OE1-IM: vae_nids.evaluation.oe1_mi.estimators.discretize);
  - casi constantes (moda > 99,9 % o H < 0,01 bits): se listan, no se quitan (D2);
  - redundancia residual: IM con corrección de sesgo de Sharmin et al. (2019) y u(X|Y) = I/H(X) en los
    dos sentidos; los 20 pares con mayor u. El histograma da una COTA INFERIOR de la IM verdadera
    (desigualdad del procesamiento de datos: discretizar es una función de X e Y).
  - comparación de listas: conjunto nuevo frente al 51 actual y B' (v2) y Robusto_TB (v2).

Uso:
    python -m vae_nids.oe1.informacion --config configs/oe1_saneado.yaml
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd
import yaml

from vae_nids import config as base_cfg
from vae_nids.evaluation.oe1_mi.estimators import discretize, entropy_bits, mi_hist
from vae_nids.oe1.config import load_config, load_feature_set, repo_path, resolve
from vae_nids.oe1.datos import load_parts

B = 20
MODA_CASI = 0.999
H_CASI = 0.01


def entropy_table(df, cols):
    rows, disc = [], {}
    for c in cols:
        x = df[c].to_numpy()
        vals, cnt = np.unique(x, return_counts=True)
        d = discretize(x, B, "atomos")
        disc[c] = d
        rows.append(dict(columna=c, valores_unicos=int(len(vals)), frac_moda=float(cnt.max() / len(x)),
                         H_bits=entropy_bits(d), intervalos=int(d.n_bins)))
    t = pd.DataFrame(rows)
    t["casi_constante"] = (t.frac_moda > MODA_CASI) | (t.H_bits < H_CASI)
    return t, disc


def redundancy(disc, cols):
    rows = []
    hs = {c: entropy_bits(disc[c]) for c in cols}
    for a, b in itertools.combinations(cols, 2):
        m = mi_hist(disc[a], disc[b], hs[a], hs[b])
        rows.append(dict(X=a, Y=b, mi_corr_bits=m["mi_corr_bits"], u_X_dado_Y=m["u_x_given_y"],
                         u_Y_dado_X=m["u_y_given_x"], H_X=m["H_x"], H_Y=m["H_y"]))
    t = pd.DataFrame(rows)
    t["u_max"] = t[["u_X_dado_Y", "u_Y_dado_X"]].max(axis=1)
    return t.sort_values("u_max", ascending=False)


def alternative_sets():
    """B' y Robusto_TB de la v2 (tablas_fuente), con su procedencia."""
    v2 = repo_path("configs/features/alternativos")
    out = {}
    for f in v2.glob("*.yaml"):
        d = yaml.safe_load(open(f, encoding="utf-8"))
        out[d["nombre"]] = d["columnas"]
    return out


def main(config=None):
    cfg = load_config(config)
    cols = load_feature_set(cfg)
    out = resolve(cfg, "salidas") / "seleccion"
    df = load_parts(cfg, names=("train_benign",), columns=cols)["train_benign"]
    et, disc = entropy_table(df, cols)
    et.to_csv(out / "entropia_por_columna.csv", index=False)
    red = redundancy(disc, cols)
    red.to_csv(out / "redundancia_residual_pares.csv", index=False)
    red.head(20).to_csv(out / "redundancia_top20.csv", index=False)
    cur51 = json.load(open(base_cfg.OUTPUT_DIR / "feature_columns.json"))
    sets = {"nuevo": cols, "actual_51": cur51, **alternative_sets()}
    comp = {}
    for a, b in itertools.combinations(sets, 2):
        A, Bs = set(sets[a]), set(sets[b])
        comp[f"{a}__vs__{b}"] = dict(en_comun=len(A & Bs), solo_en_primero=sorted(A - Bs), solo_en_segundo=sorted(Bs - A))
    pert = pd.DataFrame({k: [c in set(v) for c in sorted(set().union(*sets.values()))] for k, v in sets.items()},
                        index=sorted(set().union(*sets.values())))
    pert.to_csv(out / "pertenencia_conjuntos.csv")
    res = {"n_columnas": len(cols), "casi_constantes": et[et.casi_constante].columna.tolist(),
           "total_length_bwd_vuelve": "Total Length of Bwd Packet" in cols and "Total Length of Bwd Packet" not in cur51,
           "comparaciones": comp}
    json.dump(res, open(out / "informacion_resumen.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    r = main(a.config)
    print(json.dumps({k: v for k, v in r.items() if k != "comparaciones"}, indent=2, ensure_ascii=False))
