"""Fase 3 -- matrices de dependencia 74 x 74 (2,701 pares).

Por par: |r| de Pearson, |rho| de Spearman, IM por histograma (B=20,
cuantiles con átomos: I, sesgo, I_corr, normalizaciones, r_info, G), la
misma IM con B=10, B=50 y con qcut literal (B=20) para sensibilidad, y la
IM KSG (de phase3_ksg, submuestra de 50k).

Conjunto principal: dataset EDA (ver data.py). Conjunto secundario:
train_benign.parquet (lo que ve el VAE) -- solo Pearson, Spearman e IM B=20.

Salidas:
  metrics/oe1_pares_dependencia.csv
  metrics/oe1_pares_dependencia_train_benign.csv
  metrics/oe1_sensibilidad_B.json   (correlación de rangos entre variantes)
  figures/oe1_heatmap_pearson_vs_nmi.png
  figures/oe1_scatter_r_vs_rinfo.png

Uso: python -m vae_nids.evaluation.oe1_mi.phase3_matrices
"""
import json
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from vae_nids.evaluation.oe1_mi import config as cfg
from vae_nids.evaluation.oe1_mi import data
from vae_nids.evaluation.oe1_mi import estimators as est
from vae_nids.evaluation.oe1_mi import plotting as pl
from vae_nids.evaluation.oe1_mi.phase3_ksg import OUT_PATH as KSG_PATH

FEATS = cfg.FEATURES_74
PAIRS = list(combinations(range(len(FEATS)), 2))
SENS_COLS = ("nmi_sqrt", "nmi_max", "r_info", "mi_corr_bits", "mi_ref_gauss_bits", "gauss_ratio")


def _gauss_pair(n: int, rho: float = 0.95, seed: int = cfg.SEED) -> tuple[np.ndarray, np.ndarray]:
    z = np.random.default_rng(seed).standard_normal((n, 2))
    return z[:, 0], rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]


def _hist_block(x: np.ndarray, b: int, scheme: str) -> list[dict]:
    """IM por histograma de los 2,701 pares + criterio G: IM de una cópula
    gaussiana rho=0.95 con las mismas marginales discretizadas y el mismo N
    (misma corrección de sesgo). gauss_ratio = I_corr / I_ref; >= 1 significa
    'al menos tan dependiente como un par gaussiano con |r| = 0.95'."""
    t0 = time.perf_counter()
    disc = [est.discretize(x[:, j], b, scheme=scheme) for j in range(x.shape[1])]
    ent = [est.entropy_bits(d) for d in disc]
    z1, z2 = _gauss_pair(x.shape[0])
    ref1 = [est.gaussian_reference_codes(d, z1) for d in disc]
    ref2 = [est.gaussian_reference_codes(d, z2) for d in disc]
    ent1 = [est.entropy_bits(d) for d in ref1]
    ent2 = [est.entropy_bits(d) for d in ref2]
    rows = []
    for i, j in PAIRS:
        r = est.mi_hist(disc[i], disc[j], ent[i], ent[j])
        ref = est.mi_hist(ref1[i], ref2[j], ent1[i], ent2[j])["mi_corr_bits"]
        r["mi_ref_gauss_bits"] = ref
        r["gauss_ratio"] = r["mi_corr_bits"] / ref if ref > 0 else np.nan
        rows.append(r)
    print(f"[hist] B={b} {scheme}: {time.perf_counter() - t0:.0f}s", flush=True)
    return rows


def _corr_abs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pearson = np.corrcoef(x, rowvar=False)
    ranks = np.empty_like(x)
    for j in range(x.shape[1]):
        ranks[:, j] = stats.rankdata(x[:, j], method="average")
    spearman = np.corrcoef(ranks, rowvar=False)
    del ranks
    return pearson, spearman


def _base_frame(pearson: np.ndarray, spearman: np.ndarray) -> pd.DataFrame:
    f2c = cfg.feature_to_eda_cluster()
    grp = {m: rep for rep, ms in cfg.DOCUMENTED_GROUPS.items() for m in ms}
    kept = set(cfg.load_features_51())
    rows = []
    for i, j in PAIRS:
        a, b = FEATS[i], FEATS[j]
        ca, cb = f2c.get(a), f2c.get(b)
        rows.append({
            "feature_a": a, "feature_b": b,
            "cluster_eda_a": ca, "cluster_eda_b": cb,
            "mismo_cluster_eda": ca is not None and ca == cb,
            "fusionado_en_P": a in grp and grp.get(a) == grp.get(b),
            "ambas_en_51": a in kept and b in kept,
            "pearson_r": pearson[i, j], "pearson_abs": abs(pearson[i, j]),
            "spearman_abs": abs(spearman[i, j]),
        })
    return pd.DataFrame(rows)


def compute_pairs(x: np.ndarray, full: bool) -> tuple[pd.DataFrame, dict]:
    pearson, spearman = _corr_abs(x)
    df = _base_frame(pearson, spearman)
    main = pd.DataFrame(_hist_block(x, cfg.B_DEFAULT, "atomos"))
    df = pd.concat([df, main], axis=1)
    variants = {}
    if full:
        for b in (10, 50):
            variants[f"B{b}"] = pd.DataFrame(_hist_block(x, b, "atomos"))
        variants["qcut20"] = pd.DataFrame(_hist_block(x, cfg.B_DEFAULT, "qcut"))
        for name, v in variants.items():
            for c in SENS_COLS + ("bins_x", "bins_y"):
                df[f"{c}_{name}"] = v[c].to_numpy()
    return df, {"pearson": pearson, "spearman": spearman}


def sensitivity_ranks(df: pd.DataFrame) -> dict:
    out = {}
    cols = {"B20": "nmi_sqrt", "B10": "nmi_sqrt_B10", "B50": "nmi_sqrt_B50", "qcut20": "nmi_sqrt_qcut20"}
    names = list(cols)
    for a, b in combinations(names, 2):
        # nmi es NaN si alguna feature tiene H = 0 (pasa con qcut literal en
        # las 3 Fwd Bulk): esos pares se excluyen y se reporta cuántos quedan.
        ok = df[cols[a]].notna() & df[cols[b]].notna()
        rho = stats.spearmanr(df.loc[ok, cols[a]], df.loc[ok, cols[b]]).statistic
        tau = stats.kendalltau(df.loc[ok, cols[a]], df.loc[ok, cols[b]]).statistic
        out[f"{a}_vs_{b}"] = {"spearman": float(rho), "kendall_tau": float(tau), "n_pares": int(ok.sum())}
    return out


def cluster_order() -> list[str]:
    order = [f for cid in sorted(cfg.EDA_CLUSTERS) for f in cfg.EDA_CLUSTERS[cid]]
    return order + [f for f in FEATS if f not in set(order)]


def to_matrix(df: pd.DataFrame, col: str, order: list[str]) -> np.ndarray:
    pos = {f: k for k, f in enumerate(order)}
    m = np.eye(len(order))
    for a, b, v in df[["feature_a", "feature_b", col]].itertuples(index=False):
        m[pos[a], pos[b]] = m[pos[b], pos[a]] = v
    return m


def plot_heatmaps(df: pd.DataFrame) -> None:
    pl.setup()
    order = cluster_order()
    bounds = np.cumsum([len(cfg.EDA_CLUSTERS[c]) for c in sorted(cfg.EDA_CLUSTERS)])
    fig, axes = pl.plt.subplots(1, 2, figsize=(17, 8.6), constrained_layout=True)
    for ax, col, title in ((axes[0], "pearson_abs", "|r| de Pearson"),
                           (axes[1], "nmi_sqrt", "nmi_sqrt (IM corregida, B = 20)")):
        m = to_matrix(df, col, order)
        im = ax.imshow(m, cmap=pl.SEQ_BLUE, vmin=0, vmax=1, interpolation="nearest")
        for bd in bounds:
            ax.axhline(bd - 0.5, color=pl.SERIES[1], lw=0.6)
            ax.axvline(bd - 0.5, color=pl.SERIES[1], lw=0.6)
        ax.set_xticks(range(len(order)), order, rotation=90, fontsize=4.6)
        ax.set_yticks(range(len(order)), order, fontsize=4.6)
        ax.grid(False)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.6)
    fig.suptitle("Dependencia entre las 74 features — mismo orden en ambos paneles "
                 "(bloques = 13 clusters del EDA, después el resto)", fontsize=10)
    fig.savefig(cfg.FIGURES_DIR / "oe1_heatmap_pearson_vs_nmi.png")
    pl.plt.close(fig)


def plot_scatter(df: pd.DataFrame, rinfo_cal: float) -> None:
    pl.setup()
    fig, ax = pl.plt.subplots(figsize=(6.4, 5.6))
    other = ~df["mismo_cluster_eda"]
    same_nf = df["mismo_cluster_eda"] & ~df["fusionado_en_P"]
    fused = df["fusionado_en_P"]
    ax.scatter(df.loc[other, "pearson_abs"], df.loc[other, "r_info"], s=7, color=pl.MUTED,
               alpha=0.45, linewidths=0, label=f"resto de pares ({other.sum()})")
    ax.scatter(df.loc[same_nf, "pearson_abs"], df.loc[same_nf, "r_info"], s=22, color=pl.SERIES[1],
               edgecolors="white", linewidths=0.8, label=f"mismo cluster EDA, no fusionado ({same_nf.sum()})")
    ax.scatter(df.loc[fused, "pearson_abs"], df.loc[fused, "r_info"], s=22, color=pl.SERIES[0],
               edgecolors="white", linewidths=0.8, label=f"fusionado por P ({fused.sum()})")
    ax.axvline(0.95, color=pl.TEXT_2, lw=0.8)
    ax.axhline(0.95, color=pl.TEXT_2, lw=0.8)
    ax.axhline(rinfo_cal, color=pl.TEXT_2, lw=0.8, ls=":")
    ax.text(0.01, rinfo_cal - 0.012, f"L_cal = {rinfo_cal:.4f}", fontsize=7, color=pl.TEXT_2, va="top")
    ax.plot([0, 1], [0, 1], color=pl.GRID, lw=0.8, zorder=0)
    kw = dict(fontsize=7.5, color=pl.TEXT_2)
    ax.text(0.975, 1.005, "redundancia\nconfirmada", ha="center", va="bottom", **kw)
    ax.text(0.975, 0.02, "Pearson\nsobreestimó", ha="center", va="bottom", **kw)
    ax.text(0.47, 1.005, "redundancia no lineal no detectada", ha="center", va="bottom", **kw)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.09)
    ax.set_xlabel("|r| de Pearson")
    ax.set_ylabel("r_info (Linfoot, IM histograma B = 20)")
    ax.set_title("Pearson vs. IM por par (2.701 pares)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=1, fontsize=7.5)
    fig.savefig(cfg.FIGURES_DIR / "oe1_scatter_r_vs_rinfo.png")
    pl.plt.close(fig)


def main() -> pd.DataFrame:
    x, _ = data.load_eda_matrix()
    df, _ = compute_pairs(x, full=True)
    del x
    if KSG_PATH.exists():
        ksg = pd.read_csv(KSG_PATH)
        df = df.merge(ksg, on=["feature_a", "feature_b"], how="left", validate="one_to_one")
    else:
        print("[aviso] falta oe1_ksg_pares.csv -- correr phase3_ksg primero")
    df.to_csv(cfg.METRICS_DIR / "oe1_pares_dependencia.csv", index=False)

    sens = sensitivity_ranks(df)
    with open(cfg.METRICS_DIR / "oe1_sensibilidad_B.json", "w", encoding="utf-8") as f:
        json.dump(sens, f, indent=2)
    print(json.dumps(sens, indent=1))

    calib = json.load(open(cfg.METRICS_DIR / "oe1_validacion_estimador.json", encoding="utf-8"))
    plot_heatmaps(df)
    plot_scatter(df, calib["calibracion_umbral_095"]["B20"]["r_info_gauss_095"])

    xb = data.load_train_benign_matrix()
    dfb, _ = compute_pairs(xb, full=False)
    dfb.to_csv(cfg.METRICS_DIR / "oe1_pares_dependencia_train_benign.csv", index=False)
    print(f"[done] {len(df)} pares (EDA), {len(dfb)} pares (train_benign, N={xb.shape[0]:,})")
    return df


if __name__ == "__main__":
    main()
