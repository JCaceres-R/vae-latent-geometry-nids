"""Fase 5 -- figuras del informe de descriptores (PDF vectorial).

Paleta: instancia de referencia de la guía de visualización (modo claro, impresión).
Uso:
    python -m vae_nids.evaluation.oe2_descriptores.fase5_figuras
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from vae_nids.evaluation.oe2_descriptores import config as C  # noqa: E402
from vae_nids.evaluation.oe2_descriptores.fase3_evaluacion import ATTEMPTED_ONLY, CATALOGO  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
LEVEL_COLOR = {"P": BLUE, "S": ORANGE, "E": AQUA, "R": MUTED}
LEVEL_NAME = {"P": "Primario", "S": "Secundario", "E": "Exploratorio", "R": "Ya en el repo (phase3)"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a62c2b"])
SHORT = {"PortScan": "PortScan", "DoS_Hulk": "Hulk", "DDoS": "DDoS", "DoS_GoldenEye": "GoldenEye",
         "DoS_slowloris": "slowloris", "FTP_Patator": "FTP-Pat.", "SSH_Patator": "SSH-Pat.",
         "DoS_Slowhttptest": "Slowhttp.", "Bot": "Bot", "Web_Attack": "Web", "Infiltration": "Infiltr."}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "legend.frameon": False,
    "pdf.fonttype": 42,
})


def _save(fig, name):
    fig.savefig(C.FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(C.FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_forest(val):
    v = val[val.nivel.isin(["P", "S", "E", "R"])].copy()
    v = v.sort_values("rho")
    fig, ax = plt.subplots(figsize=(6.3, 0.19 * len(v) + 0.9))
    y = np.arange(len(v))
    ax.axvline(0, color=MUTED, lw=0.8)
    for yi, r in zip(y, v.itertuples()):
        c = LEVEL_COLOR[r.nivel]
        ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=c, lw=1.6, solid_capstyle="round")
        sig = (r.nivel == "S" and r.p_holm < 0.05) or (r.nivel != "S" and r.p_perm < 0.05)
        ax.plot(r.rho, yi, "o", ms=5.5, color=c, mfc=c if sig else "white", mew=1.4, zorder=3)
        if np.isfinite(r.rho_attempted):
            ax.plot(r.rho_attempted, yi, marker="x", ms=4, color=INK2, mew=1, zorder=2)
    ax.set_yticks(y, [f"{r.id}  {r.descriptor}" for r in v.itertuples()], fontsize=7)
    ax.set_xlim(-1, 1)
    ax.set_xlabel(f"ρ de Spearman combinado ({len(C.SEEDS)} semillas) entre el descriptor y el AUC de detección")
    for lv in ["P", "S", "E", "R"]:
        ax.plot([], [], "o", color=LEVEL_COLOR[lv], label=LEVEL_NAME[lv])
    ax.plot([], [], "o", color=INK2, mfc="white", label="relleno = p < 0,05 (Holm en secundarios)")
    ax.plot([], [], "x", color=INK2, label="ρ en los 6 grupos attempted (validación)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.6 / len(v) * 10), ncol=3, fontsize=6.5)
    _save(fig, "fig_forest_descriptores")


def fig_heatmap(fam, val):
    f = fam[fam.group.isin(C.MAIN_FAMILIES)].sort_values("auc_mean")
    cols = [("auc_mean", "AUC detección")] + [(c, did) for did, c, _, lv, _ in CATALOGO if lv in ("P", "S")]
    m = np.column_stack([f[c].rank().to_numpy() for c, _ in cols])
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    ax.imshow(m, cmap=SEQ, aspect="auto", vmin=1, vmax=11)
    ax.set_xticks(range(len(cols)), [n for _, n in cols], rotation=45, ha="right")
    ax.set_yticks(range(len(f)), [SHORT[g] for g in f.group])
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=6.5,
                    color="white" if m[i, j] >= 7 else INK)
    ax.axvline(0.5, color="white", lw=3)
    ax.set_title(f"Rango de cada familia (1 = menor, 11 = mayor), media de {len(C.SEEDS)} semillas", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    _save(fig, "fig_rangos_familias")


def fig_scatter(fam, val):
    top = val[val.nivel == "S"].reindex(val[val.nivel == "S"].rho.abs().sort_values(ascending=False).index)
    sel = [("maha__median", "1.1 Mahalanobis (mediana)")] + [(r.columna, f"{r.id} {r.descriptor[:30]}")
                                                                for r in top.head(5).itertuples()]
    f = fam[fam.group.isin(C.MAIN_FAMILIES)]
    fig, axs = plt.subplots(2, 3, figsize=(6.6, 4.4), sharey=True)
    for ax, (col, name) in zip(axs.ravel(), sel):
        r = val[val.columna == col].iloc[0]
        ax.errorbar(f[col], f.auc_mean, xerr=f[col + "_sd"], yerr=f.auc_sd, fmt="o", ms=4, color=BLUE,
                    ecolor="#9ec5f4", elinewidth=1, capsize=0, mec="white", mew=0.6)
        for g, x, yv in zip(f.group, f[col], f.auc_mean):
            ax.annotate(SHORT[g], (x, yv), xytext=(3, 2), textcoords="offset points", fontsize=5.5, color=INK2)
        ax.set_title(f"{name}\nρ = {r.rho:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]", fontsize=7, loc="left")
        ax.tick_params(labelsize=6.5)
    for ax in axs[:, 0]:
        ax.set_ylabel("AUC de detección (nll_mc)")
    fig.tight_layout()
    _save(fig, "fig_dispersion_top")


def fig_redundancy(red):
    labels = {c: did for did, c, _, _, _ in CATALOGO}
    m = red.to_numpy(float)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(m, cmap=DIV, norm=TwoSlopeNorm(0, -1, 1))
    ticks = [labels[c] for c in red.index]
    ax.set_xticks(range(len(ticks)), ticks, rotation=90)
    ax.set_yticks(range(len(ticks)), ticks)
    for i in range(len(m)):
        for j in range(len(m)):
            ax.text(j, i, f"{m[i, j]:.1f}".replace(".", ","), ha="center", va="center", fontsize=5.5,
                    color="white" if abs(m[i, j]) > 0.6 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"ρ entre descriptores (11 familias, media de Fisher sobre {len(C.SEEDS)} semillas)", loc="left", fontsize=8)
    _save(fig, "fig_redundancia")


def fig_mechanisms(fam):
    f = fam[fam.group.isin(C.MAIN_FAMILIES)]
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    for ax, (col, lab) in zip(axs, [("knn__auc", "Separación en z: AUC del kNN blanqueado (5.1)"),
                                    ("dec_knn__auc", "Separación vista por el decodificador: AUC (6.2)")]):
        ax.plot([0.4, 1], [0.4, 1], color=GRID, lw=1, zorder=0)
        ax.errorbar(f[col], f.auc_mean, xerr=f[col + "_sd"], yerr=f.auc_sd, fmt="o", ms=4.5, color=BLUE,
                    ecolor="#9ec5f4", elinewidth=1, mec="white", mew=0.6)
        for g, x, yv in zip(f.group, f[col], f.auc_mean):
            ax.annotate(SHORT[g], (x, yv), xytext=(3, 2), textcoords="offset points", fontsize=6, color=INK2)
        ax.set_xlabel(lab, fontsize=7)
        ax.set_xlim(0.4, 1.01)
    axs[0].set_ylabel("AUC de detección (nll_mc)")
    axs[0].text(0.42, 0.98, "diagonal: el detector\nve lo mismo que la geometría", fontsize=6, color=MUTED, va="top")
    fig.tight_layout()
    _save(fig, "fig_dos_mecanismos")


def fig_duplicates(dup):
    d = dup[dup.group != "BENIGN_test"].sort_values("n")
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    y = np.arange(len(d))
    ax.barh(y + 0.19, d.n, height=0.36, color=BLUE, label="flujos")
    ax.barh(y - 0.19, d.mu_unicos, height=0.36, color=ORANGE, label="vectores μ distintos")
    ax.set_xscale("log")
    ax.set_yticks(y, d.group.str.replace("_", " "), fontsize=6.5)
    for yi, (n, u) in enumerate(zip(d.n, d.mu_unicos)):
        if u / n < 0.95:
            ax.text(n * 1.15, yi + 0.19, f"{100 * u / n:.0f} % distintos".replace(".", ","), fontsize=6,
                    va="center", color=INK2)
    ax.set_xlabel("número (escala log), semilla 42")
    ax.legend(loc="lower right", fontsize=7)
    _save(fig, "fig_duplicados")


def fig_diagnostics(diag):
    kl = diag[[f"kl_dim{j}" for j in range(8)]].to_numpy()
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.7), gridspec_kw=dict(width_ratios=[1.4, 1]))
    im = axs[0].imshow(kl, cmap=SEQ, aspect="auto")
    axs[0].set_yticks(range(len(diag)), [f"semilla {s}" for s in diag.seed], fontsize=6.5)
    axs[0].set_xticks(range(8), [f"z{j}" for j in range(8)])
    axs[0].set_title("KL media por dimensión (nats, benigno)", loc="left")
    cb = fig.colorbar(im, ax=axs[0], fraction=0.05)
    cb.outline.set_visible(False)
    for s in axs[0].spines.values():
        s.set_visible(False)
    ax = axs[1]
    lo, hi = diag[["sum_sigma2_obs", "sum_sigma2_pred"]].min().min() * 0.9, diag[["sum_sigma2_obs", "sum_sigma2_pred"]].max().max() * 1.05
    ax.plot([lo, hi], [lo, hi], color=GRID, lw=1)
    ax.plot(diag.sum_sigma2_pred, diag.sum_sigma2_obs, "o", color=BLUE, ms=5, mec="white", mew=0.6)
    ax.set_xlabel("Σ σ² predicha: 1/(1+G_jj)")
    ax.set_ylabel("Σ σ² observada (codificador)")
    ax.set_title("Kumar y Poole por semilla", loc="left")
    fig.tight_layout()
    _save(fig, "fig_diagnosticos_modelo")


def fig_umap(um):
    fams = C.MAIN_FAMILIES
    fig, axs = plt.subplots(3, 4, figsize=(6.6, 5.0), sharex=True, sharey=True)
    b = um[um.group == "BENIGN"]
    for ax, g in zip(axs.ravel(), ["BENIGN"] + fams):
        ax.scatter(b.u1, b.u2, s=1.5, color="#c3c2b7", linewidths=0, rasterized=True)
        if g != "BENIGN":
            s = um[um.group == g]
            ax.scatter(s.u1, s.u2, s=3 if len(s) > 200 else 8, color=ORANGE, linewidths=0, rasterized=True)
        ax.set_title(SHORT.get(g, "Benigno"), fontsize=7, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    _save(fig, "fig_umap_sin_duplicados")


def fig_flow(flow):
    cols = [c for c in flow.columns if c.endswith("__rho_flow_nll")]
    lab = {c: next((did for did, cc, _, _, _ in CATALOGO if cc.startswith(c.split("__")[0] + "__")), c) for c in cols}
    f = flow.loc[[g for g in C.MAIN_FAMILIES if g in flow.index]]
    m = f[cols].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    ax.imshow(m, cmap=DIV, norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
    ax.set_xticks(range(len(cols)), [f"{lab[c]} {c.split('__')[0]}" for c in cols], rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(f)), [SHORT[g] for g in f.index])
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=5.5,
                        color="white" if abs(m[i, j]) > 0.6 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"ρ de Spearman descriptor–NLL dentro de cada familia (nivel de flujo, media de {len(C.SEEDS)} semillas)", loc="left", fontsize=8)
    _save(fig, "fig_nivel_flujo")


def main():
    t = C.TABLES_DIR
    val = pd.read_csv(t / "valoracion_descriptores.csv")
    fam = pd.read_csv(t / "resumen_por_familia.csv")
    fig_forest(val)
    fig_heatmap(fam, val)
    fig_scatter(fam, val)
    fig_redundancy(pd.read_csv(t / "redundancia_descriptores.csv", index_col=0))
    fig_mechanisms(fam)
    fig_duplicates(pd.read_csv(t / "duplicados_seed42.csv"))
    fig_diagnostics(pd.read_csv(t / "diagnosticos_por_semilla.csv"))
    fig_umap(pd.read_csv(t / "umap_sin_duplicados_seed42.csv"))
    fig_flow(pd.read_csv(t / "nivel_flujo_por_familia.csv", index_col=0))
    print("[fase5] figuras en", C.FIGURES_DIR)


if __name__ == "__main__":
    main()
