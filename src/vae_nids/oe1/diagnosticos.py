"""Fase 5 del OE1 saneado -- diagnósticos SOLO con datos benignos (7.3, 7.4) y figuras.

Por semilla: época de parada y motivo, mejor pérdida de validación, KL por dimensión, unidades
activas con el criterio de Burda et al. (2016): A_u = Var_x[mu_u(x)] > 0,01 (el que usa el repo); y,
porque el prompt lo pide así, también KL_u > 0,01. tau = p95 del score sobre val_benign, con
s(x) = NLL gaussiana heterocedástica evaluada en mu (An y Cho, 2015).
Decodificador: log sigma^2 por columna (media y percentiles en val_benign) y columnas en el piso -10.
NLL media por columna (la pérdida total y tau no son comparables entre conjuntos de distinto tamaño).

Uso:
    python -m vae_nids.oe1.diagnosticos --config configs/oe1_saneado.yaml
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from vae_nids.oe1.config import load_config, resolve  # noqa: E402
from vae_nids.oe1.datos import load_parts  # noqa: E402
from vae_nids.oe1.entrenamiento import load_checkpoint  # noqa: E402

COLOR_TRAIN, COLOR_VAL, COLOR_BAR = "#2a78d6", "#eb6834", "#2a78d6"
COLOR_ACTIVE, COLOR_INACTIVE = "#0ca30c", "#d03b3b"
COLOR_GRID, COLOR_AXIS, COLOR_INK, COLOR_MUTED = "#e1e0d9", "#c3c2b7", "#0b0b0b", "#898781"
LOG2PI = float(np.log(2 * np.pi))


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(COLOR_AXIS)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)


@torch.no_grad()
def seed_diagnostics(ckpt_path, val_df, cfg):
    model, scaler, ck = load_checkpoint(ckpt_path)
    x = torch.from_numpy(scaler.transform(val_df))
    mus, lvs, nll_cols, lvx = [], [], [], []
    for s in range(0, len(x), 8192):
        xb = x[s:s + 8192]
        mu, lv = model.encoder(xb)
        mx, lx = model.decoder(mu)
        nll = 0.5 * (LOG2PI + lx + (xb - mx) ** 2 / lx.exp())
        mus.append(mu); lvs.append(lv); nll_cols.append(nll); lvx.append(lx)
    mu, lv, nll, lx = (torch.cat(a).numpy() for a in (mus, lvs, nll_cols, lvx))
    kl = 0.5 * (np.exp(lv) + mu ** 2 - 1.0 - lv)
    var_mu = mu.var(0)
    thr = cfg["evaluacion"]["umbral_unidades_activas"]
    score = nll.sum(1)
    tau = float(np.percentile(score, cfg["evaluacion"]["tau_percentil"]))
    piso = cfg["modelo"]["logvar_min"]
    lx_med = np.median(lx, 0)
    row = dict(seed=ck["semilla"], n_columnas=len(ck["columnas"]), epoca_parada=ck["epoca_parada"],
               mejor_epoca=ck["mejor_epoca"], motivo_parada=ck["motivo_parada"], mejor_val_total=ck["mejor_val_total"],
               activas_burda=int((var_mu > thr).sum()), activas_kl=int((kl.mean(0) > thr).sum()),
               kl_total=float(kl.mean(0).sum()), tau_p95=tau, nll_media=float(score.mean()),
               nll_media_por_columna=float(score.mean() / len(ck["columnas"])),
               columnas_en_piso=int((lx_med <= piso + 1e-6).sum()), frac_logvar_dec_en_piso=float((lx <= piso + 1e-6).mean()),
               segundos=ck.get("segundos_totales"), commit=ck["git"]["commit"], arbol_limpio=ck["git"]["arbol_limpio"])
    for j in range(mu.shape[1]):
        row[f"var_mu_{j}"] = float(var_mu[j])
        row[f"kl_{j}"] = float(kl.mean(0)[j])
    cols_tab = pd.DataFrame({"columna": ck["columnas"], "logvar_media": lx.mean(0), "logvar_p5": np.percentile(lx, 5, 0),
                             "logvar_p50": lx_med, "logvar_p95": np.percentile(lx, 95, 0),
                             "nll_media": nll.mean(0), "en_piso": lx_med <= piso + 1e-6})
    return row, cols_tab


def fig_curves(log, title, path):
    fig, axes = plt.subplots(3, 1, figsize=(7, 7.5), sharex=True)
    for ax, (k, t) in zip(axes, [("total", "Pérdida total (−ELBO)"), ("recon", "Reconstrucción (NLL)"), ("kl", "KL")]):
        ax.plot(log.epoch, log[f"train_{k}"], color=COLOR_TRAIN, lw=2, label="train")
        ax.plot(log.epoch, log[f"val_{k}"], color=COLOR_VAL, lw=2, label="val")
        ax.set_title(t, fontsize=9, color=COLOR_INK, loc="left")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("época", color=COLOR_MUTED)
    fig.suptitle(title, fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_latent(row, path, k=8):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    dims = np.arange(k)
    a.bar(dims, [row[f"kl_{j}"] for j in dims], color=COLOR_BAR, width=0.6, zorder=3)
    a.set_title("KL por dimensión latente (nats)", fontsize=9, color=COLOR_INK, loc="left")
    v = np.array([row[f"var_mu_{j}"] for j in dims])
    b.bar(dims, v, color=[COLOR_ACTIVE if x > 0.01 else COLOR_INACTIVE for x in v], width=0.6, zorder=3)
    b.axhline(0.01, color=COLOR_MUTED, lw=1, ls="--")
    b.set_yscale("log")
    b.set_title(f"Unidades activas ({row['activas_burda']}/{k}): Var(μ) > 0,01 — Burda et al. (2016)",
                fontsize=9, color=COLOR_INK, loc="left")
    for ax in (a, b):
        ax.set_xticks(dims, [f"z{j}" for j in dims]); _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_active(tab, path):
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.bar(tab.seed.astype(str), tab.activas_burda, color=COLOR_BAR, width=0.6, zorder=3)
    for i, (s, v) in enumerate(zip(tab.seed, tab.activas_burda)):
        ax.text(i, v + 0.1, str(v), ha="center", fontsize=8, color=COLOR_INK)
    ax.set_ylim(0, 8.8); ax.set_xlabel("semilla", color=COLOR_MUTED); ax.set_ylabel("unidades activas (de 8)", color=COLOR_MUTED)
    ax.set_title(f"Unidades activas por semilla: media {tab.activas_burda.mean():.1f} ± {tab.activas_burda.std():.1f}".replace(".", ","),
                 fontsize=9, loc="left", color=COLOR_INK)
    _style(ax); fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_logvar(cols_tab, piso, path):
    t = cols_tab.sort_values("logvar_p50")
    fig, ax = plt.subplots(figsize=(7, max(4, 0.13 * len(t))))
    y = np.arange(len(t))
    ax.hlines(y, t.logvar_p5, t.logvar_p95, color="#9ec5f4", lw=2)
    ax.plot(t.logvar_p50, y, "o", color=COLOR_BAR, ms=3.5)
    ax.axvline(piso, color=COLOR_INACTIVE, lw=1, ls="--")
    ax.text(piso + 0.1, len(t) - 0.5, "piso −10", color=COLOR_INACTIVE, fontsize=7, va="top")
    ax.set_yticks(y, t.columna, fontsize=6)
    ax.set_xlabel("log σ² del decodificador en val_benign (mediana y p5–p95)", color=COLOR_MUTED)
    _style(ax); fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def main(config=None, extra_configs=()):
    cfg = load_config(config)
    out = resolve(cfg, "salidas")
    fig_dir, tab_dir = out / "figures", out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True); tab_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in [cfg, *[load_config(e) for e in extra_configs]]:
        from vae_nids.oe1.config import load_feature_set
        cols = load_feature_set(c)
        val = load_parts(c, names=("val_benign",), columns=cols)["val_benign"]
        seeds = c["semillas"]["principal"] if c["entrenamiento"]["nombre_corrida"] == "oe1_info_nula" else c["semillas"]["referencia_74"]
        for s in seeds:
            run = f"{c['entrenamiento']['nombre_corrida']}_seed{s}"
            p = resolve(c, "salidas") / "checkpoints" / f"{run}.pt"
            if not p.exists():
                continue
            row, ctab = seed_diagnostics(p, val, c)
            row["corrida"] = c["entrenamiento"]["nombre_corrida"]
            rows.append(row)
            ctab.to_csv(tab_dir / f"logvar_decodificador_{run}.csv", index=False)
            log = pd.read_csv(resolve(c, "salidas") / "logs" / f"{run}_train_log.csv")
            if s == 42:
                fig_curves(log, f"Curvas de entrenamiento — {c['entrenamiento']['nombre_corrida']} ({len(cols)} columnas), semilla 42",
                           fig_dir / f"curvas_{run}.png")
                fig_latent(row, fig_dir / f"diagnostico_latente_{run}.png")
                fig_logvar(ctab, c["modelo"]["logvar_min"], fig_dir / f"logvar_decodificador_{run}.png")
    tab = pd.DataFrame(rows)
    tab.to_csv(tab_dir / "diagnosticos_por_semilla.csv", index=False)
    main_tab = tab[tab.corrida == "oe1_info_nula"]
    if len(main_tab):
        fig_active(main_tab, fig_dir / "unidades_activas_por_semilla.png")
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    print(main(a.config, a.extra).drop(columns=[c for c in ("commit",) if c]).to_string())
