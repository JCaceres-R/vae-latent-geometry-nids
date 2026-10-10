"""Fase 1 del OE1 saneado -- conjunto de entrada por información nula (D1, D2).

Solo se eliminan columnas con información nula, sin umbral y solo con train_benign:
  R0  copia exacta de otra columna (se queda la que ya conservaba el pipeline 77 -> 74);
  R1  entropía nula: un único valor en train_benign;
  R2  función exacta X = f(S) de columnas que se quedan, verificada fila a fila en el
      100 % de train_benign con una fórmula explícita (y tramos para los casos borde).
Regla para elegir qué columna sale en cada familia (fijada antes de mirar resultados):
se quedan las medidas en bruto (conteos, bytes totales, duraciones, sumas, mínimos y
máximos) y salen los cocientes (tasas, medias = total/conteo, razones); entre una
magnitud y su transformación biyectiva (Std frente a Variance) se queda la que está en
las unidades de la medida. Para la información ambas elecciones son equivalentes; la
regla solo atiende a la estabilidad numérica.

Tolerancia (ε por identidad, derivada del almacenamiento y del código fuente).
Almacenamiento: CICFlowMeter escribe cada double con StringBuilder.append(double), es decir
Double.toString, que al releerse devuelve exactamente el mismo double: el error de
almacenamiento es 0 (los valores tienen 16-17 cifras significativas). La única diferencia
legítima entre X y f(S) es el redondeo de las operaciones en punto flotante que hace
CICFlowMeter (BasicFlow.java del fork de Engelen, commit 4dd5319) y las que hace f(S):
  - medias de SummaryStatistics (Apache Commons Math): actualización incremental
    m <- m + (x - m)/k, 3 operaciones redondeadas por valor => |error| <= 3·k·u·|X|;
  - una división o una raíz: <= 1 u cada una; f(S) añade <= 4 u.
Con u = 2^-53, cada identidad declara su ε_rel por fila (función de n_f, n_b o n).
Almacenamiento: algunas celdas del CSV corregido traen solo 15 cifras significativas (no el
double completo). Por eso se suma el cuanto de almacenamiento por fila
q = 1/2 · 10^(e - max(s, 15) + 1) de X y de cada entrada no entera (propagado con la
derivada de f), con s las cifras escritas en esa celda (vae_nids.oe1.datos.storage_quantum).
Cota final: |f(S) - X| <= q_X + sum_i |df/ds_i| q_i + ε_rel · max(|X|, |f(S)|).
Columnas enteras: round(f(S)) == X. Si X = 0, la cota exige f(S) = 0 exacto.
Si una identidad excede su ε en alguna fila, la columna se queda ("casi función").

Ningún paso de selección lee datos de ataque. Solo la comprobación posterior
(`comprobacion_posterior`) lee las demás particiones, después de congelar.

Uso:
    python -m vae_nids.oe1.seleccion --config configs/oe1_saneado.yaml
"""
import argparse
import json
from datetime import date

import numpy as np
import pandas as pd
import yaml

from vae_nids.oe1.config import columns_hash, load_config, repo_path, resolve
from vae_nids.oe1.datos import QPREFIX, load_parts

U = 2.0 ** -53
D, NF, NB = "Flow Duration", "Total Fwd Packet", "Total Bwd packets"
BF, BB = "Total Length of Fwd Packet", "Total Length of Bwd Packet"

# Las 3 columnas que el pipeline 77 -> 74 ya trataba como copias, y la que conservaba.
COPIAS_PIPELINE = {"Fwd Segment Size Avg": "Fwd Packet Length Mean",
                   "Bwd Segment Size Avg": "Bwd Packet Length Mean",
                   "Average Packet Size": "Packet Length Mean"}


def _div0(num, den):
    """num/den con la convención de CICFlowMeter: 0 si den == 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den != 0, num / np.where(den != 0, den, 1.0), 0.0)


def _rate(num, d):
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / (d * 1e-6)


# Cada identidad: columna que sale, entradas (todas se quedan), fórmula, tramos y función.
# Orden de aplicación R2: las fórmulas solo usan columnas en bruto que la regla conserva.
IDENTIDADES = [
    dict(col="Flow Packets/s", entradas=[D, NF, NB], tipo="continua",
         formula=r"(n_f + n_b) / (D \cdot 10^{-6})", texto="(n_f + n_b) / (D·1e-6)",
         tramos="D > 0 en todas las filas tras el saneamiento (con D = 0 el CSV trae inf y la fila se elimina)",
         f=lambda x: _rate(x[NF] + x[NB], x[D])),
    dict(col="Fwd Packets/s", entradas=[D, NF], tipo="continua",
         formula=r"n_f / (D \cdot 10^{-6})", texto="n_f / (D·1e-6)", tramos="igual que Flow Packets/s",
         f=lambda x: _rate(x[NF], x[D])),
    dict(col="Bwd Packets/s", entradas=[D, NB], tipo="continua",
         formula=r"n_b / (D \cdot 10^{-6})", texto="n_b / (D·1e-6)", tramos="igual que Flow Packets/s",
         f=lambda x: _rate(x[NB], x[D])),
    dict(col="Flow Bytes/s", entradas=[D, BF, BB], tipo="continua",
         formula=r"(B_f + B_b) / (D \cdot 10^{-6})", texto="(B_f + B_b) / (D·1e-6)", tramos="igual que Flow Packets/s",
         f=lambda x: _rate(x[BF] + x[BB], x[D])),
    dict(col="Flow IAT Mean", entradas=[D, NF, NB], tipo="continua",
         formula=r"D / (n_f + n_b - 1)\ \text{si } n \ge 2;\ 0 \text{ si no}", texto="D / (n - 1) si n >= 2; 0 si no",
         tramos="n < 2: 0 (convención; con un solo paquete no hay intervalos)",
         f=lambda x: _div0(x[D], x[NF] + x[NB] - 1.0) * ((x[NF] + x[NB]) >= 2)),
    dict(col="Fwd IAT Mean", entradas=["Fwd IAT Total", NF], tipo="continua",
         formula=r"\text{Fwd IAT Total} / (n_f - 1)\ \text{si } n_f \ge 2;\ 0 \text{ si no}",
         texto="Fwd IAT Total / (n_f - 1) si n_f >= 2; 0 si no", tramos="n_f < 2: 0 (y Fwd IAT Total = 0)",
         f=lambda x: _div0(x["Fwd IAT Total"], x[NF] - 1.0) * (x[NF] >= 2)),
    dict(col="Bwd IAT Mean", entradas=["Bwd IAT Total", NB], tipo="continua",
         formula=r"\text{Bwd IAT Total} / (n_b - 1)\ \text{si } n_b \ge 2;\ 0 \text{ si no}",
         texto="Bwd IAT Total / (n_b - 1) si n_b >= 2; 0 si no", tramos="n_b < 2: 0 (y Bwd IAT Total = 0)",
         f=lambda x: _div0(x["Bwd IAT Total"], x[NB] - 1.0) * (x[NB] >= 2)),
    dict(col="Fwd Packet Length Mean", entradas=[BF, NF], tipo="continua",
         formula=r"B_f / n_f\ \text{si } n_f > 0;\ 0 \text{ si no}", texto="B_f / n_f si n_f > 0; 0 si no",
         tramos="n_f = 0: 0", f=lambda x: _div0(x[BF], x[NF])),
    dict(col="Bwd Packet Length Mean", entradas=[BB, NB], tipo="continua",
         formula=r"B_b / n_b\ \text{si } n_b > 0;\ 0 \text{ si no}", texto="B_b / n_b si n_b > 0; 0 si no",
         tramos="n_b = 0 (flujos sin paquetes de vuelta): 0", f=lambda x: _div0(x[BB], x[NB])),
    dict(col="Packet Length Mean", entradas=[BF, BB, NF, NB], tipo="continua",
         formula=r"(B_f + B_b) / (n_f + n_b)", texto="(B_f + B_b) / (n_f + n_b)", tramos="n > 0 siempre",
         f=lambda x: _div0(x[BF] + x[BB], x[NF] + x[NB])),
    dict(col="Packet Length Variance", entradas=["Packet Length Std"], tipo="continua",
         formula=r"(\text{Packet Length Std})^2", texto="Packet Length Std^2", tramos="ninguno",
         f=lambda x: x["Packet Length Std"] ** 2),
    dict(col="Packet Length Max", entradas=["Fwd Packet Length Max", "Bwd Packet Length Max"], tipo="entera",
         formula=r"\max(\text{Fwd PL Max}, \text{Bwd PL Max})", texto="max(Fwd PL Max, Bwd PL Max)",
         tramos="ninguno (el máximo de una dirección vacía es 0)",
         f=lambda x: np.maximum(x["Fwd Packet Length Max"], x["Bwd Packet Length Max"])),
    dict(col="Packet Length Min", entradas=["Fwd Packet Length Min", "Bwd Packet Length Min", NF, NB], tipo="entera",
         formula=r"\text{Bwd PL Min si } n_f = 0;\ \text{Fwd PL Min si } n_b = 0;\ \min(\cdot,\cdot) \text{ si no}",
         texto="Bwd PL Min si n_f = 0; Fwd PL Min si n_b = 0; min(Fwd PL Min, Bwd PL Min) si no",
         tramos="dirección sin paquetes: su mínimo vale 0 en el CSV y no participa (regla corregida de la V3)",
         f=lambda x: np.where(x[NF] == 0, x["Bwd Packet Length Min"],
                              np.where(x[NB] == 0, x["Fwd Packet Length Min"],
                                       np.minimum(x["Fwd Packet Length Min"], x["Bwd Packet Length Min"])))),
    dict(col="Down/Up Ratio", entradas=[NF, NB], tipo="continua",
         formula=r"n_b / n_f\ \text{si } n_f > 0;\ 0 \text{ si no}", texto="n_b / n_f si n_f > 0; 0 si no",
         tramos="n_f = 0: 0. No es entera ni truncada: es el cociente real",
         f=lambda x: _div0(x[NB], x[NF])),
]
def _n(x):
    return x[NF] + x[NB]


# ε_rel por fila (fracción de max(|X|,|f|)) y línea de BasicFlow.java donde CICFlowMeter calcula X.
EPS = {
    "Flow Packets/s": (lambda x: 4 * U + 0 * x[D], "4u (una conversión y dos divisiones)", "BasicFlow.java:1280 (dump //22), 1031"),
    "Fwd Packets/s": (lambda x: 4 * U + 0 * x[D], "4u", "BasicFlow.java:361-366 (getfPktsPerSecond)"),
    "Bwd Packets/s": (lambda x: 4 * U + 0 * x[D], "4u", "BasicFlow.java:369-374 (getbPktsPerSecond)"),
    "Flow Bytes/s": (lambda x: 4 * U + 0 * x[D], "4u", "BasicFlow.java:1279 (dump //21), 1027"),
    "Flow IAT Mean": (lambda x: (3 * (_n(x) - 1) + 4) * U, "(3(n-1)+4)u: media incremental de n-1 IAT", "BasicFlow.java:1287 (flowIAT.getMean)"),
    "Fwd IAT Mean": (lambda x: (3 * np.maximum(x[NF] - 1, 0) + 4) * U, "(3(n_f-1)+4)u", "BasicFlow.java:1294 (forwardIAT.getMean)"),
    "Bwd IAT Mean": (lambda x: (3 * np.maximum(x[NB] - 1, 0) + 4) * U, "(3(n_b-1)+4)u", "BasicFlow.java:1308 (backwardIAT.getMean)"),
    "Fwd Packet Length Mean": (lambda x: (3 * x[NF] + 4) * U, "(3n_f+4)u", "BasicFlow.java:1258 (fwdPktStats.getMean)"),
    "Bwd Packet Length Mean": (lambda x: (3 * x[NB] + 4) * U, "(3n_b+4)u", "BasicFlow.java:1270 (bwdPktStats.getMean)"),
    "Packet Length Mean": (lambda x: (3 * _n(x) + 4) * U, "(3n+4)u", "BasicFlow.java:1336 (flowLengthStats.getMean)"),
    "Packet Length Variance": (lambda x: 4 * U + 0 * x[D], "4u (raíz y cuadrado)", "BasicFlow.java:1337-1338 (getStandardDeviation, getVariance)"),
    "Packet Length Max": (None, "exacta (entera)", "BasicFlow.java:1335 (flowLengthStats.getMax)"),
    "Packet Length Min": (None, "exacta (entera)", "BasicFlow.java:1334 (flowLengthStats.getMin)"),
    "Down/Up Ratio": (lambda x: 2 * U + 0 * x[D], "2u (una división)", "BasicFlow.java:377-382 (getDownUpRatio)"),
}
EPS_COPIA = {  # copia frente a la columna que conserva el pipeline
    "Fwd Segment Size Avg": (lambda x: (3 * x[NF] + 4) * U, "(3n_f+4)u: getSum()/n frente a la media incremental", "BasicFlow.java:391-395 (fAvgSegmentSize) frente a :1258"),
    "Bwd Segment Size Avg": (lambda x: (3 * x[NB] + 4) * U, "(3n_b+4)u", "BasicFlow.java:397-401 (bAvgSegmentSize) frente a :1270"),
    "Average Packet Size": (lambda x: (3 * _n(x) + 4) * U, "(3n+4)u", "BasicFlow.java:384-389 (getAvgPacketSize) frente a :1336"),
}

# Reconstrucción de las copias R0 desde el conjunto final (sin pasar por otra eliminada).
RECONSTRUCCION_COPIAS = {
    "Fwd Segment Size Avg": ("Fwd Packet Length Mean", [BF, NF], "B_f / n_f si n_f > 0; 0 si no"),
    "Bwd Segment Size Avg": ("Bwd Packet Length Mean", [BB, NB], "B_b / n_b si n_b > 0; 0 si no"),
    "Average Packet Size": ("Packet Length Mean", [BF, BB, NF, NB], "(B_f + B_b) / (n_f + n_b)"),
}


def tolerance_check(x_true, pred, eps_rel, tipo, q_abs=0.0):
    x_true = np.asarray(x_true, np.float64)
    pred = np.asarray(pred, np.float64)
    if tipo == "entera":
        ok = np.round(pred) == x_true
        err = np.abs(np.round(pred) - x_true)
        return ok, err, np.zeros_like(err)
    err = np.abs(pred - x_true)
    scale = np.maximum(np.abs(x_true), np.abs(pred))
    with np.errstate(invalid="ignore"):
        ok = (err <= q_abs + eps_rel * scale) & np.isfinite(pred)
        rel = np.where(scale > 0, err / scale, 0.0)
    return ok, err, rel


def _q(df, col):
    return df[QPREFIX + col].to_numpy(np.float64) if QPREFIX + col in df.columns else np.zeros(len(df))


# Propagación del cuanto de entradas no enteras: |df/ds| q_s (solo la varianza tiene una).
Q_ENTRADAS = {"Packet Length Variance": lambda df: 2.0 * np.abs(df["Packet Length Std"].to_numpy()) * _q(df, "Packet Length Std")}


def _ident_by_col():
    return {d["col"]: d for d in IDENTIDADES}


def _arrays(df):
    return {c: df[c].to_numpy(np.float64) for c in df.columns
            if c not in ("Label", "Day", "row_id") and not c.startswith(QPREFIX)}


def verify_identity(df, col, fn, tipo, eps_fn):
    x = _arrays(df)
    pred = fn(x)
    eps = eps_fn(x) if eps_fn is not None else 0.0
    q_abs = _q(df, col) + (Q_ENTRADAS[col](df) if col in Q_ENTRADAS else 0.0)
    ok, err, rel = tolerance_check(df[col].to_numpy(), pred, eps, tipo, q_abs)
    scale = np.maximum(np.abs(df[col].to_numpy()), np.abs(pred))
    with np.errstate(invalid="ignore", divide="ignore"):
        tol = q_abs + np.asarray(eps) * scale
        ratio = np.where(tol > 0, err / tol, 0.0)
    return dict(filas=int(len(df)), frac_verificada=float(ok.mean()) if len(ok) else 1.0, fallos=int((~ok).sum()),
                error_abs_max=float(np.nanmax(err)) if len(err) else 0.0,
                error_rel_max=float(np.nanmax(rel)) if len(rel) else 0.0,
                error_rel_max_ulp=float(np.nanmax(rel) / U) if len(rel) else 0.0,
                error_sobre_eps_max=float(np.nanmax(ratio)) if len(err) and tipo == "continua" else 0.0), ok


def storage_precision(cfg, universe, nrows=300_000):
    """Precisión con la que el CSV de origen guarda cada columna (muestra de las primeras filas del lunes)."""
    raw = resolve(cfg, "raw_dir") / cfg["datos"]["csv"][0]
    s = pd.read_csv(raw, nrows=nrows, dtype=str, low_memory=False)
    s.columns = s.columns.str.strip()
    out = {}
    for c in universe:
        v = s[c].dropna().astype(str)
        mant = v.str.replace(r"[eE].*$", "", regex=True).str.replace("-", "", regex=False)
        digits = mant.str.replace(".", "", regex=False).str.lstrip("0").str.len()
        x = pd.to_numeric(v, errors="coerce").to_numpy()
        out[c] = dict(max_cifras_significativas=int(digits.max()) if len(digits) else 0,
                      usa_notacion_E=bool(v.str.contains("E").any()),
                      entera=bool(np.all(np.isfinite(x[np.isfinite(x)]) & (np.mod(x[np.isfinite(x)], 1) == 0))))
    return out


def systematic_search(df, universe):
    """H(X|Y) = 0 en la muestra (agrupación exacta) para cada par ordenado, con su soporte."""
    codes = {c: np.unique(df[c].to_numpy(), return_inverse=True)[1].ravel() for c in universe}
    rows = []
    for y in universe:
        cy = codes[y]
        order = np.argsort(cy, kind="stable")
        ys = cy[order]
        same = ys[1:] == ys[:-1]
        counts = np.bincount(cy)
        support = float(counts[counts >= 2].sum() / len(cy))
        n_groups = int(len(counts))
        for x in universe:
            if x == y:
                continue
            xs = codes[x][order]
            dep = not np.any(same & (xs[1:] != xs[:-1]))
            if dep:
                rows.append(dict(X=x, Y=y, soporte_Y=support, valores_distintos_Y=n_groups,
                                 valores_distintos_X=int(codes[x].max() + 1)))
    return pd.DataFrame(rows)


def select(cfg):
    parts = load_parts(cfg, names=("train_benign",))
    df = parts["train_benign"]
    universe = [c for c in df.columns if c not in ("Label", "Day", "row_id") and not c.startswith(QPREFIX)]
    assert len(universe) == 77, len(universe)
    estado = {c: dict(estado="queda", regla=None, formula=None, tramos=None, entradas=None, eps=None,
                      frac_verificada=None, error_abs_max=None, error_rel_max=None) for c in universe}

    # R0: copias (igualdad dentro de la tolerancia de almacenamiento)
    for copia, keep in COPIAS_PIPELINE.items():
        eps_fn, eps_txt, fuente = EPS_COPIA[copia]
        ok, err, rel = tolerance_check(df[copia].to_numpy(), df[keep].to_numpy(), eps_fn(_arrays(df)), "continua",
                                       _q(df, copia) + _q(df, keep))
        bit = float(np.mean(df[copia].to_numpy() == df[keep].to_numpy()))
        if not ok.all():
            raise RuntimeError(f"R0: {copia} no es copia de {keep} dentro de la tolerancia")
        estado[copia].update(estado="sale", regla="R0", formula=f"= {keep}", entradas=[keep],
                             frac_verificada=float(ok.mean()), error_abs_max=float(err.max()),
                             error_rel_max=float(rel.max()), error_rel_max_ulp=float(rel.max() / U),
                             frac_bit_a_bit=bit, eps=eps_txt, fuente=fuente)
    # R1: entropía nula
    for c in universe:
        if estado[c]["estado"] == "queda" and df[c].nunique(dropna=False) == 1:
            estado[c].update(estado="sale", regla="R1", formula=f"constante = {df[c].iloc[0]!r}", frac_verificada=1.0)
    # R2: funciones exactas
    casi = []
    for idt in IDENTIDADES:
        c = idt["col"]
        if estado[c]["estado"] != "queda":
            continue
        if any(estado[e]["estado"] != "queda" for e in idt["entradas"]):
            raise RuntimeError(f"{c}: la fórmula usa una columna eliminada")
        eps_fn, eps_txt, fuente = EPS[c]
        r, _ = verify_identity(df, c, idt["f"], idt["tipo"], eps_fn)
        if r["fallos"] == 0:
            estado[c].update(estado="sale", regla="R2", formula=idt["texto"], formula_tex=idt["formula"],
                             tramos=idt["tramos"], entradas=idt["entradas"], tipo=idt["tipo"],
                             eps="round(f) == X" if idt["tipo"] == "entera" else eps_txt, fuente=fuente, **r)
        else:
            casi.append(dict(columna=c, formula=idt["texto"], eps=eps_txt, fuente=fuente, **r))
            estado[c].update(casi_funcion=idt["texto"], eps_casi=eps_txt, frac_verificada=r["frac_verificada"],
                             error_rel_max=r["error_rel_max"], fallos_casi=r["fallos"])
    final = [c for c in universe if estado[c]["estado"] == "queda"]

    # Sin circularidad: reconstruir TODAS las eliminadas solo desde el conjunto final.
    recon = {}
    by = _ident_by_col()
    for c in universe:
        if estado[c]["estado"] != "sale" or estado[c]["regla"] == "R1":
            continue
        if estado[c]["regla"] == "R0":
            keep, ins, txt = RECONSTRUCCION_COPIAS[c]
            idt = by[keep]
            assert set(ins) == set(idt["entradas"])
            fn, tipo, eps_fn = idt["f"], idt["tipo"], EPS_COPIA[c][0]
            estado[c]["reconstruccion_desde_final"] = txt
        else:
            fn, tipo, ins, eps_fn = by[c]["f"], by[c]["tipo"], by[c]["entradas"], EPS[c][0]
        assert all(e in final for e in ins), (c, ins)
        r, _ = verify_identity(df, c, fn, tipo, eps_fn)
        recon[c] = r
        estado[c]["reconstruccion_frac"] = r["frac_verificada"]
    return df, universe, estado, final, casi, recon


def freeze(cfg, universe, estado, final, prec):
    h = columns_hash(final)
    doc = {"descripcion": "Conjunto de entrada del OE1 saneado por información nula (R0/R1/R2), "
                          "derivado solo de train_benign tras el filtro D4.",
           "fecha": str(date.today()), "n_universo": len(universe), "n_quedan": len(final),
           "sha256": h, "columnas_que_quedan": final, "columnas": {}}
    for c in universe:
        e = {k: v for k, v in estado[c].items() if v is not None}
        e["precision_csv"] = prec.get(c)
        doc["columnas"][c] = e
    path = repo_path(cfg["columnas"]["conjunto"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, width=120)
    return h


def comprobacion_posterior(cfg, estado):
    """Solo después de congelar: ¿las identidades aceptadas valen en val, test y en cada familia?"""
    parts = load_parts(cfg, names=("val_benign", "test_benign", "test_attacks"))
    by = _ident_by_col()
    rows = []
    for name, df in parts.items():
        groups = [("todas", df)] if name != "test_attacks" else list(df.groupby("Label"))
        for gname, g in groups:
            for c, e in estado.items():
                if e["estado"] != "sale" or e["regla"] == "R1":
                    continue
                if e["regla"] == "R0":
                    keep = COPIAS_PIPELINE[c]
                    ok, err, rel = tolerance_check(g[c].to_numpy(), g[keep].to_numpy(),
                                                   EPS_COPIA[c][0](_arrays(g)), "continua", _q(g, c) + _q(g, keep))
                    r = dict(filas=len(g), frac_verificada=float(ok.mean()), fallos=int((~ok).sum()),
                             error_rel_max=float(rel.max()) if len(rel) else 0.0)
                else:
                    r, _ = verify_identity(g, c, by[c]["f"], by[c]["tipo"], EPS[c][0])
                rows.append(dict(particion=name, grupo=gname, columna=c, regla=e["regla"], **r))
    return pd.DataFrame(rows)


def main(config=None):
    cfg = load_config(config)
    out = resolve(cfg, "salidas") / "seleccion"
    out.mkdir(parents=True, exist_ok=True)
    df, universe, estado, final, casi, recon = select(cfg)
    prec = storage_precision(cfg, universe)
    h = freeze(cfg, universe, estado, final, prec)
    dep = systematic_search(df, universe)
    dep.to_csv(out / "busqueda_sistematica_dependencias.csv", index=False)
    pd.DataFrame(casi).to_csv(out / "casi_funciones.csv", index=False)
    pd.DataFrame([dict(columna=c, **r) for c, r in recon.items()]).to_csv(out / "reconstruccion_desde_final.csv", index=False)
    tabla = pd.DataFrame([dict(columna=c, **{k: v for k, v in e.items() if k != "entradas" and not isinstance(v, list)},
                               entradas="; ".join(e["entradas"]) if e.get("entradas") else "")
                          for c, e in estado.items()])
    tabla["precision_csv_cifras"] = [prec[c]["max_cifras_significativas"] for c in tabla.columna]
    tabla["precision_csv_entera"] = [prec[c]["entera"] for c in tabla.columna]
    tabla.to_csv(out / "estado_columnas.csv", index=False)
    post = comprobacion_posterior(cfg, estado)
    post.to_csv(out / "comprobacion_posterior.csv", index=False)
    resumen = {"n_universo": len(universe), "n_quedan": len(final), "sha256": h,
               "eliminadas_por_regla": tabla[tabla.estado == "sale"].regla.value_counts().to_dict(),
               "casi_funciones": [c["columna"] for c in casi],
               "fallos_posteriores": post[post.fallos > 0][["particion", "grupo", "columna", "fallos"]].to_dict("records")}
    json.dump(resumen, open(out / "resumen_seleccion.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return resumen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    print(json.dumps(main(a.config), indent=2, ensure_ascii=False))
