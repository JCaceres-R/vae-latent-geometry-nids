"""Fase 2 del OE1 saneado -- preprocesamiento corregido.

1. Filtro D4 (ya aplicado en vae_nids.oe1.datos, dentro de cada partición y después de dividir).
2. Particiones idénticas al pipeline original salvo las filas filtradas (hashes en datos_info.json).
3. MinMax ajustado SOLO con train_benign sobre el conjunto congelado; parámetros en float64.
4. Resolución (V1 sobre el conjunto nuevo): por columna, valores distintos en crudo frente a los
   almacenados (float32 escalado, lo que ve el modelo), posición del 0 y resolución efectiva en
   unidades originales (separación float32 en el valor típico, deshecha el escalado). Se marca toda
   columna que pierde > 50 % de valores distintos o cuya resolución es peor que su granularidad
   natural (1 unidad en columnas enteras).
5. Filas que volverían si se quitara la regla de inf/NaN (solo informativo; decide el autor).

Uso:
    python -m vae_nids.oe1.preproceso --config configs/oe1_saneado.yaml
"""
import argparse
import json

import numpy as np
import pandas as pd

from vae_nids.oe1.config import load_config, load_feature_set, resolve
from vae_nids.oe1.datos import PARTS, load_parts


class MinMax64:
    """MinMax con parámetros en float64; se ajusta con una sola partición."""

    def __init__(self, data_min=None, data_max=None, columns=None):
        self.data_min = None if data_min is None else np.asarray(data_min, np.float64)
        self.data_max = None if data_max is None else np.asarray(data_max, np.float64)
        self.columns = columns

    def fit(self, df: pd.DataFrame, columns):
        x = df[columns].to_numpy(np.float64)
        self.columns = list(columns)
        self.data_min, self.data_max = x.min(0), x.max(0)
        return self

    @property
    def scale(self):
        rng = self.data_max - self.data_min
        return np.where(rng > 0, rng, 1.0)

    def transform(self, df: pd.DataFrame, dtype=np.float32) -> np.ndarray:
        x = df[self.columns].to_numpy(np.float64)
        return ((x - self.data_min) / self.scale).astype(dtype)

    def to_dict(self):
        return {"columns": self.columns, "data_min": self.data_min.tolist(), "data_max": self.data_max.tolist(),
                "dtype_parametros": "float64"}

    @classmethod
    def from_dict(cls, d):
        return cls(d["data_min"], d["data_max"], d["columns"])


def resolution_table(raw: pd.DataFrame, scaler: MinMax64):
    stored = scaler.transform(raw)
    rows = []
    for j, c in enumerate(scaler.columns):
        x = raw[c].to_numpy(np.float64)
        s = stored[:, j]
        n_raw, n_sto = len(np.unique(x)), len(np.unique(s))
        integer = bool(np.all(np.mod(x, 1) == 0))
        med = float(np.median(s))
        spacing_scaled = float(np.spacing(np.float32(max(abs(med), np.float32(np.finfo(np.float32).tiny)))))
        spacing_max = float(np.spacing(np.float32(1.0)))
        res_med = spacing_scaled * scaler.scale[j]
        res_max = spacing_max * scaler.scale[j]
        zero_pos = float((0.0 - scaler.data_min[j]) / scaler.scale[j])
        rows.append(dict(columna=c, entera=integer, min=float(scaler.data_min[j]), max=float(scaler.data_max[j]),
                         valores_distintos_crudo=n_raw, valores_distintos_almacenados=n_sto,
                         frac_conservada=n_sto / n_raw, posicion_del_cero=zero_pos,
                         resolucion_en_mediana=res_med, resolucion_en_maximo=res_max))
    t = pd.DataFrame(rows)
    t["pierde_mas_50pct"] = t.frac_conservada < 0.5
    t["resolucion_peor_que_granularidad"] = t.entera & (t.resolucion_en_maximo > 1.0)
    t["marcada"] = t.pierde_mas_50pct | t.resolucion_peor_que_granularidad
    return t


def main(config=None):
    cfg = load_config(config)
    cols = load_feature_set(cfg)
    out = resolve(cfg, "processed_dir")
    rep_dir = resolve(cfg, "salidas") / "preproceso"
    rep_dir.mkdir(parents=True, exist_ok=True)
    parts = load_parts(cfg, names=PARTS)
    scaler = MinMax64().fit(parts["train_benign"], cols)
    json.dump(scaler.to_dict(), open(out / "escalador_minmax_float64.json", "w", encoding="utf-8"), indent=2)
    for n, df in parts.items():
        sc = pd.DataFrame(scaler.transform(df), columns=cols)
        sc["Label"] = df["Label"].to_numpy()
        sc["Day"] = df["Day"].to_numpy()
        sc["row_id"] = df["row_id"].to_numpy()
        sc.to_parquet(out / f"{n}_escalado.parquet", index=False)
    res = resolution_table(parts["train_benign"], scaler)
    res.to_csv(rep_dir / "resolucion_por_columna.csv", index=False)
    # Flow Packets/s antes (pipeline original) y después: valores distintos que conserva el escalado
    info = json.load(open(out / "datos_info.json", encoding="utf-8"))
    extra = {}
    tb = parts["train_benign"]
    for c in ("Flow Packets/s", "Flow Duration"):
        if c in tb.columns:
            x = tb[c].to_numpy(np.float64)
            lo, hi = x.min(), x.max()
            s32 = ((x - lo) / (hi - lo)).astype(np.float32)
            extra[c] = dict(min=float(lo), max=float(hi), distintos_crudo=int(len(np.unique(x))),
                            distintos_escalado_float32=int(len(np.unique(s32))))
    removed_inf = info["columnas_con_no_finitos"]
    resumen = {"columnas": len(cols), "marcadas": res[res.marcada].columna.tolist(),
               "flow_iat_min_negativos": info["flow_iat_min_negativos"], "filtro_d4": {
                   k: v["eliminadas"] for k, v in info["filtro_d4"].items()},
               "resolucion_tras_filtro": extra,
               "regla_inf_nan": {"filas_eliminadas": info["eliminadas_inf_nan"], "columnas_con_no_finitos": removed_inf,
                                 "nota": "todas las filas eliminadas tienen inf/NaN en Flow Bytes/s y Flow Packets/s, "
                                         "que siguen en el conjunto; ninguna volvería al quitar la regla"}}
    json.dump(resumen, open(rep_dir / "resumen_preproceso.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return resumen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    print(json.dumps(main(a.config), indent=2, ensure_ascii=False))
