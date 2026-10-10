"""Figuras y tablas fuente del informe LaTeX del OE1 saneado (las de selección y preprocesamiento;
las de entrenamiento las genera vae_nids.oe1.diagnosticos).

Uso:
    python -m vae_nids.oe1.figuras_informe --config configs/oe1_saneado.yaml
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from vae_nids import config as base_cfg  # noqa: E402
from vae_nids.oe1 import seleccion  # noqa: E402
from vae_nids.oe1.config import load_config, load_feature_set, resolve  # noqa: E402
from vae_nids.oe1.datos import load_parts  # noqa: E402

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"
CRIT, INK, MUTED, GRID = "#d03b3b", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f7f7f5", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRAY, "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})


def save(fig, d, name):
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_entropia(sel, d):
    e = pd.read_csv(sel / "entropia_por_columna.csv").sort_values("H_bits")
    fig, ax = plt.subplots(figsize=(6.3, 8.6))
    y = np.arange(len(e))
    ax.barh(y, e.H_bits, color=[CRIT if c else BLUE for c in e.casi_constante], height=0.7)
    ax.set_yticks(y, e.columna, fontsize=6)
    ax.set_xlabel("H en bits (cuantiles con átomos aislados, B = 20), train_benign")
    for yi, (h, c) in enumerate(zip(e.H_bits, e.casi_constante)):
        if c:
            ax.text(max(h, 0) + 0.05, yi, f"casi constante (H = {h:.1e})".replace(".", ","), fontsize=5.5, va="center", color=CRIT)
    ax.grid(axis="x", color=GRID)
    save(fig, d, "entropia_por_columna")


def fig_error_identidades(sel, d, cfg):
    df = load_parts(cfg, names=("train_benign",))["train_benign"]
    rows = []
    for idt in seleccion.IDENTIDADES:
        c = idt["col"]
        if idt["tipo"] == "entera":
            continue
        r, _ = seleccion.verify_identity(df, c, idt["f"], idt["tipo"], seleccion.EPS[c][0])
        rows.append(dict(columna=c, regla="R2" if r["fallos"] == 0 else "casi función", ratio=r["error_sobre_eps_max"],
                         ulp=r["error_rel_max_ulp"], fallos=r["fallos"]))
    for copia, keep in seleccion.COPIAS_PIPELINE.items():
        x = seleccion._arrays(df)
        eps = seleccion.EPS_COPIA[copia][0](x)
        q = seleccion._q(df, copia) + seleccion._q(df, keep)
        a, b = df[copia].to_numpy(), df[keep].to_numpy()
        tol = q + eps * np.maximum(np.abs(a), np.abs(b))
        ratio = np.where(tol > 0, np.abs(a - b) / tol, 0.0)
        rows.append(dict(columna=copia, regla="R0", ratio=float(ratio.max()),
                         ulp=float((np.abs(a - b) / np.maximum(np.abs(a), 1e-300)).max() / seleccion.U), fallos=int((ratio > 1).sum())))
    t = pd.DataFrame(rows).sort_values("ratio")
    t.to_csv(sel / "error_sobre_tolerancia.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    y = np.arange(len(t))
    ax.barh(y, t.ratio, color=[ORANGE if f > 0 else BLUE for f in t.fallos], height=0.65)
    ax.axvline(1.0, color=CRIT, lw=1.2, ls="--")
    ax.text(1.05, len(t) - 0.6, "tolerancia ε", color=CRIT, fontsize=7)
    ax.set_xscale("log")
    ax.set_yticks(y, [f"{c} ({r})" for c, r in zip(t.columna, t.regla)], fontsize=6.5)
    ax.set_xlabel("máx. sobre filas de |f(S) − X| / ε (escala log); > 1 = la identidad falla en alguna fila")
    ax.grid(axis="x", color=GRID)
    save(fig, d, "error_sobre_tolerancia")


def fig_conjuntos(sel, d):
    p = pd.read_csv(sel / "pertenencia_conjuntos.csv", index_col=0)
    order = ["nuevo", "actual_51", "B_prima_51", "Robusto_TB_56", "info_nula_eps_ampliado_60"]
    p = p[[c for c in order if c in p.columns]]
    p = p.sort_values(list(p.columns), ascending=False)
    fig, ax = plt.subplots(figsize=(5.2, 0.105 * len(p) + 0.8))
    for j, c in enumerate(p.columns):
        yy = np.where(p[c].to_numpy())[0]
        ax.scatter(np.full(len(yy), j), yy, s=14, color=[BLUE, ORANGE, AQUA, GRAY, "#4a3aa7"][j], marker="s")
    ax.set_xticks(range(len(p.columns)), ["Nuevo (66)", "Actual (51)", "B' (51)", "Robusto TB (56)", "ε ampliado (60)"],
                  rotation=30, ha="right")
    ax.set_yticks(range(len(p)), p.index, fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, len(p.columns) - 0.5)
    ax.grid(axis="y", color=GRID, lw=0.4)
    save(fig, d, "pertenencia_conjuntos")


def fig_heatmap_u(sel, d, cols):
    r = pd.read_csv(sel / "redundancia_residual_pares.csv")
    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for row in r.itertuples():
        m.loc[row.X, row.Y] = row.u_X_dado_Y
        m.loc[row.Y, row.X] = row.u_Y_dado_X
    # Tamaño pensado para una página apaisada (≈ 9 × 6,3 pulgadas útiles) sin reducir: etiquetas de 6,5 pt.
    fig, ax = plt.subplots(figsize=(9.0, 6.3))
    im = ax.imshow(m.to_numpy(), cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(cols)), cols, fontsize=6.5)
    ax.set_title("u(X|Y) = I(X;Y)/H(X): fila X, columna Y (IM corregida, B = 20; cota inferior)", fontsize=8, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.outline.set_visible(False)
    save(fig, d, "heatmap_u")


def fig_resolucion(cfg, d, sel):
    new = json.load(open(resolve(cfg, "salidas") / "preproceso" / "resumen_preproceso.json", encoding="utf-8"))
    old = pd.read_parquet(base_cfg.OUTPUT_DIR / "train_benign.parquet", columns=["Flow Packets/s"])
    raw = load_parts(cfg, names=("train_benign",), columns=["Flow Packets/s", "Flow Duration"])["train_benign"]
    old_d = int(old["Flow Packets/s"].nunique())
    rows = [dict(caso="Flow Packets/s, pipeline original (float32, con duraciones negativas)",
                 distintos=old_d, crudo=int(raw["Flow Packets/s"].nunique()) + 0),
            dict(caso="Flow Packets/s, saneado (filtro D4)", distintos=new["resolucion_tras_filtro"]["Flow Packets/s"]["distintos_escalado_float32"],
                 crudo=new["resolucion_tras_filtro"]["Flow Packets/s"]["distintos_crudo"]),
            dict(caso="Flow Duration, saneado (filtro D4)", distintos=new["resolucion_tras_filtro"]["Flow Duration"]["distintos_escalado_float32"],
                 crudo=new["resolucion_tras_filtro"]["Flow Duration"]["distintos_crudo"])]
    t = pd.DataFrame(rows)
    t["frac"] = t.distintos / t.crudo
    t.to_csv(sel.parent / "preproceso" / "resolucion_antes_despues.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.3, 1.9))
    y = np.arange(len(t))
    ax.barh(y, 100 * t.frac, color=[CRIT, BLUE, BLUE], height=0.55)
    for yi, f in enumerate(t.frac):
        ax.text(100 * f + 1, yi, f"{100 * f:.1f} %".replace(".", ","), va="center", fontsize=7, color=INK)
    ax.set_yticks(y, t.caso, fontsize=7)
    ax.set_xlim(0, 115)
    ax.set_xlabel("% de valores distintos (de train_benign en crudo) que conserva la columna almacenada")
    ax.invert_yaxis()
    save(fig, d, "resolucion_antes_despues")
    return t


def main(config=None):
    cfg = load_config(config)
    out = resolve(cfg, "salidas")
    sel, d = out / "seleccion", out / "figures_informe"
    d.mkdir(parents=True, exist_ok=True)
    cols = load_feature_set(cfg)
    fig_entropia(sel, d)
    fig_conjuntos(sel, d)
    fig_heatmap_u(sel, d, cols)
    print(fig_resolucion(cfg, d, sel).to_string())
    fig_error_identidades(sel, d, cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    main(ap.parse_args().config)
