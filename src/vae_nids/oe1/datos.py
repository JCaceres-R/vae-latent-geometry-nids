"""Datos del OE1 saneado: lectura en crudo (float64), particiones idénticas al pipeline
original y filtro de duraciones negativas dentro de cada partición (D4).

Reproduce `vae_nids.data.pipeline` paso a paso, con dos diferencias deliberadas:
  1. Los valores se conservan en float64 (el pipeline original los guardaba en float32).
  2. Después de dividir, se quitan los flujos con Flow Duration < 0 de cada partición.
Para que las filas eliminadas por inf/NaN sean exactamente las mismas que en el
pipeline original, la regla de saneamiento se evalúa sobre los valores convertidos a
float32, como hacía el original (un valor finito en float64 que desborda float32 se
volvía inf allí). Se reporta cuántas filas afecta esa sutileza.

Uso:
    python -m vae_nids.oe1.datos --config configs/oe1_saneado.yaml
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from vae_nids.oe1.config import load_config, resolve

PARTS = ["train_benign", "val_benign", "test_benign", "test_attacks"]

# Columnas no enteras que intervienen en identidades: para ellas se guarda, por fila, el cuanto de
# almacenamiento q = 1/2 · 10^(e - max(s, 15) + 1) (e: exponente decimal del valor; s: cifras
# significativas escritas en el CSV). Supuesto: el escritor nunca usa menos de 15 cifras.
QUANTUM_COLS = ["Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s", "Flow Bytes/s", "Flow IAT Mean",
                "Fwd IAT Mean", "Bwd IAT Mean", "Fwd Packet Length Mean", "Bwd Packet Length Mean",
                "Packet Length Mean", "Packet Length Variance", "Packet Length Std", "Down/Up Ratio",
                "Fwd Segment Size Avg", "Bwd Segment Size Avg", "Average Packet Size"]
QPREFIX = "q::"
MIN_SIG_DIGITS = 15


def storage_quantum(text: pd.Series) -> np.ndarray:
    s = text.astype(str).str.strip()
    parts = s.str.split("[eE]", n=1, regex=True)
    mant = parts.str[0].str.lstrip("+-")
    exp_e = pd.to_numeric(parts.str[1], errors="coerce").fillna(0).to_numpy()
    digits = mant.str.replace(".", "", regex=False).str.lstrip("0").str.rstrip("0").str.len().to_numpy()
    x = pd.to_numeric(s, errors="coerce").to_numpy(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.floor(np.log10(np.abs(x)))
        q = 0.5 * 10.0 ** (e - np.maximum(digits, MIN_SIG_DIGITS) + 1)
    q[~np.isfinite(x) | (x == 0)] = 0.0
    del exp_e
    return q


def load_raw(cfg) -> tuple[pd.DataFrame, list[str]]:
    d = cfg["datos"]
    raw_dir = resolve(cfg, "raw_dir")
    drop = set(d["columnas_no_flujo"])
    frames = []
    for fname in d["csv"]:
        df = pd.read_csv(raw_dir / fname, low_memory=False)
        df.columns = df.columns.str.strip()
        df = df.drop(columns=[c for c in drop if c in df.columns])
        txt = pd.read_csv(raw_dir / fname, dtype=str, low_memory=False)
        txt.columns = txt.columns.str.strip()
        for c in QUANTUM_COLS:
            df[QPREFIX + c] = storage_quantum(txt[c])
        del txt
        df["Day"] = fname.split("-")[0]
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    flow_cols = [c for c in full.columns if c not in (d["etiqueta"], "Day") and not c.startswith(QPREFIX)]
    full[flow_cols] = full[flow_cols].astype(np.float64)
    return full, flow_cols


def sanitize(df: pd.DataFrame, flow_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    v64 = df[flow_cols].to_numpy(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        v32 = v64.astype(np.float32)
    bad32 = ~np.isfinite(v32).all(1)
    bad64 = ~np.isfinite(v64).all(1)
    info = {"filas_crudas": int(len(df)), "eliminadas_inf_nan": int(bad32.sum()),
            "eliminadas_solo_por_desborde_float32": int((bad32 & ~bad64).sum())}
    # columnas que causan la eliminación (para la decisión "¿qué filas volverían?")
    nonfinite = ~np.isfinite(v32)
    info["columnas_con_no_finitos"] = {c: int(nonfinite[:, j].sum()) for j, c in enumerate(flow_cols)
                                       if nonfinite[:, j].any()}
    out = df.loc[~bad32].reset_index(drop=True)
    out["row_id"] = np.arange(len(out), dtype=np.int64)        # id estable tras el saneamiento
    return out, info


def split(df: pd.DataFrame, cfg) -> dict[str, pd.DataFrame]:
    d = cfg["datos"]
    lab = d["etiqueta"]
    benign = df[df[lab] == "BENIGN"].reset_index(drop=True)
    strat = benign["Day"] if d["estratificar_por_dia"] else None
    train, temp = train_test_split(benign, train_size=d["fraccion_train"], random_state=d["semilla_split"],
                                   stratify=strat)
    rel = d["fraccion_val"] / (d["fraccion_val"] + d["fraccion_test"])
    strat_t = temp["Day"] if d["estratificar_por_dia"] else None
    val, test = train_test_split(temp, train_size=rel, random_state=d["semilla_split"], stratify=strat_t)
    attacks = df[df[lab] != "BENIGN"]
    return {"train_benign": train.reset_index(drop=True), "val_benign": val.reset_index(drop=True),
            "test_benign": test.reset_index(drop=True), "test_attacks": attacks.reset_index(drop=True)}


def ids_hash(ids) -> str:
    return hashlib.sha256(np.sort(np.asarray(ids, dtype=np.int64)).tobytes()).hexdigest()


def verify_against_original(parts: dict[str, pd.DataFrame], cfg) -> dict:
    """Las particiones (antes del filtro D4) deben tener las mismas filas, en el mismo orden, que los
    parquets del pipeline original. Se comparan columnas que el original NO escaló (están en crudo
    float32) y la etiqueta."""
    from vae_nids import config as base_cfg
    orig_dir = base_cfg.OUTPUT_DIR
    if not (orig_dir / "feature_columns.json").exists():
        return {name: "sin pipeline original para comparar" for name in parts}
    feats51 = json.load(open(orig_dir / "feature_columns.json"))
    out = {}
    for name, df in parts.items():
        p = orig_dir / f"{name}.parquet"
        if not p.exists():
            out[name] = "sin parquet original para comparar"
            continue
        o = pd.read_parquet(p)
        unscaled = [c for c in o.columns if c in df.columns and c not in feats51 and c not in ("Label", "Day")]
        same_n = len(o) == len(df)
        eq = same_n and all(np.array_equal(o[c].to_numpy(np.float32), df[c].to_numpy(np.float32), equal_nan=True)
                            for c in unscaled)
        lab_eq = same_n and (o["Label"].to_numpy() == df["Label"].to_numpy()).all()
        out[name] = {"filas_original": int(len(o)), "filas_nuevas": int(len(df)), "columnas_comparadas": len(unscaled),
                     "identicas": bool(eq and lab_eq)}
    return out


def filter_negative_duration(parts: dict[str, pd.DataFrame], cfg) -> tuple[dict, dict]:
    report = {}
    out = {}
    for name, df in parts.items():
        neg = df["Flow Duration"] < 0
        r = {"filas_antes": int(len(df)), "eliminadas": int(neg.sum()), "filas_despues": int((~neg).sum()),
             "hash_ids_antes": ids_hash(df.row_id), "hash_ids_despues": ids_hash(df.row_id[~neg]),
             "hash_ids_eliminadas": ids_hash(df.row_id[neg])}
        if name == "test_attacks":
            r["por_etiqueta"] = {k: int(v) for k, v in df.loc[neg, cfg["datos"]["etiqueta"]].value_counts().items()}
        report[name] = r
        out[name] = df.loc[~neg].reset_index(drop=True) if cfg["datos"]["filtrar_duracion_negativa"] else df
    return out, report


def build(cfg, save=True) -> tuple[dict, dict]:
    raw, flow_cols = load_raw(cfg)
    clean, info = sanitize(raw, flow_cols)
    del raw
    parts = split(clean, cfg)
    info["universo_columnas"] = flow_cols
    info["verificacion_vs_original"] = verify_against_original(parts, cfg)
    parts, info["filtro_d4"] = filter_negative_duration(parts, cfg)
    info["flow_iat_min_negativos"] = {n: int((df["Flow IAT Min"] < 0).sum()) for n, df in parts.items()}
    if save:
        out = resolve(cfg, "processed_dir")
        out.mkdir(parents=True, exist_ok=True)
        for n, df in parts.items():
            df.to_parquet(out / f"{n}_crudo.parquet", index=False)
        json.dump(info, open(out / "datos_info.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return parts, info


def load_parts(cfg, names=("train_benign",), columns=None) -> dict[str, pd.DataFrame]:
    """Carga particiones crudas ya construidas (float64)."""
    out = resolve(cfg, "processed_dir")
    return {n: pd.read_parquet(out / f"{n}_crudo.parquet", columns=columns) for n in names}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    c = load_config(a.config)
    _, inf = build(c)
    print(json.dumps({k: v for k, v in inf.items() if k != "universo_columnas"}, indent=2, ensure_ascii=False))
