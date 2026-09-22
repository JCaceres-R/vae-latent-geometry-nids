"""Fase 6 -- visualización (apoyo, no evidencia).

Genera las 7 figuras pedidas (PNG 300dpi + PDF) en outputs/oe2_robust/figures/,
más la proyección UMAP para 2-3 semillas representativas con paleta de
colores distinguibles (BENIGN/PortScan quedaron indistinguibles en el OE2
original, tab20 con ambos en tonos de azul).

Uso:
    python -m vae_nids.evaluation.oe2_robust.phase6_figures
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vae_nids.evaluation.oe2_robust import config as rcfg
from vae_nids.evaluation.oe2_robust.phase3_geometry import group_names, load_group_mu_logvar, mahalanobis_batch, subsample

DISTINCT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#000000", "#aaffc3",
]


def savefig(fig, name: str):
    fig.tight_layout()
    fig.savefig(rcfg.FIGURES_DIR / f"{name}.png", dpi=300)
    fig.savefig(rcfg.FIGURES_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"[fase6] {name}.png / .pdf")


# ------------------------------------------------------------------
# 1. Distribuciones de Mahalanobis por familia
# ------------------------------------------------------------------

def fig1_mahalanobis_distributions(representative_seed: int, geo: pd.DataFrame):
    with open(rcfg.METRICS_DIR / "phase2_active_dims_per_seed.json", encoding="utf-8") as f:
        active_dims_map = json.load(f)
    active_dims = active_dims_map[str(representative_seed)]

    mu_benign, _ = load_group_mu_logvar(representative_seed, "latent_benign_test")
    mean_b = mu_benign[:, active_dims].mean(axis=0)
    cov_b = np.cov(mu_benign[:, active_dims], rowvar=False)
    inv_cov_b = np.linalg.inv(cov_b)

    data, labels = [], []
    for name in rcfg.MAIN_FAMILIES:
        mu_g, _ = load_group_mu_logvar(representative_seed, name)
        dist = mahalanobis_batch(mu_g[:, active_dims], mean_b, inv_cov_b)
        data.append(dist)
        labels.append(name)
    order = np.argsort([np.median(x) for x in data])
    data = [data[i] for i in order]
    labels = [labels[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].violinplot(data, vert=False, showmedians=True)
    axes[0].set_yticks(range(1, len(labels) + 1))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("Distancia de Mahalanobis")
    axes[0].set_title(f"Distribución completa, seed={representative_seed}")

    med = geo[(geo["metric"] == "mahalanobis_median") & (geo["group"].isin(rcfg.MAIN_FAMILIES))]
    pivot = med.pivot(index="group", columns="seed", values="value")
    pivot = pivot.loc[[l for l in labels if l in pivot.index]]
    means = pivot.mean(axis=1)
    stds = pivot.std(axis=1)
    y = np.arange(len(pivot))
    axes[1].errorbar(means, y, xerr=stds, fmt="o", capsize=3, color="#4C72B0")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(pivot.index, fontsize=8)
    axes[1].set_xlabel("Mediana de Mahalanobis (media ± DE entre semillas)")
    axes[1].set_title(f"Estabilidad entre {pivot.shape[1]} semillas")
    savefig(fig, "fig1_mahalanobis_distributions")


# ------------------------------------------------------------------
# 2. Forest plot AUC-ROC (nll_mc)
# ------------------------------------------------------------------

def fig2_forest_auc(det: pd.DataFrame):
    d = det[(det["score"] == "nll_mc") & (det["group"].isin(rcfg.MAIN_FAMILIES))]
    agg = d.groupby("group")["auc_roc"].agg(["mean", "std", "count"]).reset_index()
    agg = agg.sort_values("mean")
    y = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.errorbar(agg["mean"], y, xerr=agg["std"], fmt="o", capsize=3, color="#4C72B0")
    ax.set_yticks(y)
    ax.set_yticklabels(agg["group"], fontsize=9)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Azar (AUC=0.5)")
    ax.set_xlabel("AUC-ROC (nll_mc), media ± DE entre semillas")
    ax.set_title(f"Forest plot de detectabilidad, {int(agg['count'].iloc[0])} semillas")
    ax.legend(fontsize=8)
    savefig(fig, "fig2_forest_auc")


# ------------------------------------------------------------------
# 3. Scatter geometría vs. detectabilidad con barras de error en ambos ejes
# ------------------------------------------------------------------

def fig3_scatter_geometry_detectability(geo: pd.DataFrame, det: pd.DataFrame, h1: dict):
    g = geo[(geo["metric"] == "mahalanobis_median") & (geo["group"].isin(rcfg.MAIN_FAMILIES))]
    g_agg = g.groupby("group")["value"].agg(["mean", "std"])
    d = det[(det["score"] == "nll_mc") & (det["group"].isin(rcfg.MAIN_FAMILIES))]
    d_agg = d.groupby("group")["auc_roc"].agg(["mean", "std"])
    both = g_agg.join(d_agg, lsuffix="_geo", rsuffix="_det")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(both["mean_geo"], both["mean_det"], xerr=both["std_geo"], yerr=both["std_det"],
                fmt="o", capsize=2, color="#4C72B0", ecolor="gray", elinewidth=0.8)
    for name, row in both.iterrows():
        ax.annotate(name, (row["mean_geo"], row["mean_det"]), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    rho = h1["combined"]["rho_combined"]
    ci_lo, ci_hi = h1["combined"]["ci_low"], h1["combined"]["ci_high"]
    ax.set_xlabel("Distancia de Mahalanobis (mediana, media entre semillas)")
    ax.set_ylabel("AUC-ROC (nll_mc, media entre semillas)")
    ax.set_title(f"Geometría vs. detectabilidad, barras = DE entre semillas\n"
                 f"rho combinado={rho:.3f} IC95=[{ci_lo:.3f},{ci_hi:.3f}]")
    savefig(fig, "fig3_scatter_geometry_detectability")


# ------------------------------------------------------------------
# 4. Distribución de rho entre semillas (par principal)
# ------------------------------------------------------------------

def fig4_rho_distribution(h1: dict):
    rhos = [r["rho"] for r in h1["per_seed"]]
    combined = h1["combined"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(rhos, bins=min(10, len(rhos)), color="#4C72B0", alpha=0.7, edgecolor="black")
    ax.axvline(combined["rho_combined"], color="red", linewidth=2,
               label=f"combinado={combined['rho_combined']:.3f}")
    ax.axvspan(combined["ci_low"], combined["ci_high"], color="red", alpha=0.15, label="IC95 combinado")
    ax.axvline(h1["previous_oe2_single_seed"]["rho"], color="black", linestyle="--",
               label=f"OE2 original (seed=42 solo)={h1['previous_oe2_single_seed']['rho']:.3f}")
    ax.set_xlabel("Spearman rho (mahalanobis_median vs. AUC-ROC nll_mc)")
    ax.set_ylabel(f"Frecuencia ({len(rhos)} semillas)")
    ax.set_title("Distribución de rho entre semillas -- par principal")
    ax.legend(fontsize=8)
    savefig(fig, "fig4_rho_distribution")


# ------------------------------------------------------------------
# 5. Heatmap de rankings por semilla
# ------------------------------------------------------------------

def fig5_ranking_heatmap(det: pd.DataFrame):
    d = det[(det["score"] == "nll_mc") & (det["group"].isin(rcfg.MAIN_FAMILIES))]
    pivot = d.pivot(index="group", columns="seed", values="auc_roc")
    ranks = pivot.rank(axis=0, ascending=False)
    order = ranks.mean(axis=1).sort_values().index
    ranks = ranks.loc[order]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(ranks.values, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(range(ranks.shape[1]))
    ax.set_xticklabels(ranks.columns, fontsize=8)
    ax.set_yticks(range(ranks.shape[0]))
    ax.set_yticklabels(ranks.index, fontsize=8)
    ax.set_xlabel("Semilla")
    ax.set_title("Ranking de detectabilidad (AUC-ROC nll_mc) por familia y semilla\n(1=más detectable)")
    for i in range(ranks.shape[0]):
        for j in range(ranks.shape[1]):
            ax.text(j, i, f"{ranks.values[i, j]:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Ranking")
    savefig(fig, "fig5_ranking_heatmap")


# ------------------------------------------------------------------
# 6. Coeficientes del modelo mixto
# ------------------------------------------------------------------

def fig6_mixed_model_coefficients(mixed: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    if not mixed.get("converged"):
        ax.text(0.5, 0.5, "Modelo mixto no convergió\n(ver informe, sección 7.3)",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        savefig(fig, "fig6_mixed_model_coefficients")
        return

    fe = mixed["fixed_effect_maha_std"]
    lo, hi = mixed["fixed_effect_ci_low"], mixed["fixed_effect_ci_high"]
    ax.errorbar([fe], [1], xerr=[[fe - lo], [hi - fe]], fmt="o", capsize=4, color="#4C72B0", markersize=10)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks([1])
    ax.set_yticklabels(["Efecto fijo (maha_std -> score_std)"])
    slope_var = mixed.get("random_slope_variance_by_family")
    ax.set_title(f"Modelo mixto a nivel de conexión (n={mixed.get('n_obs'):,})\n"
                 f"R2 marginal={mixed.get('r2_marginal', float('nan')):.3f}, "
                 f"R2 condicional={mixed.get('r2_conditional', float('nan')):.3f}, "
                 f"var. pendiente por familia={slope_var if slope_var is not None else 'N/D'}")
    ax.set_xlabel("Coeficiente estandarizado (IC95)")
    savefig(fig, "fig6_mixed_model_coefficients")


# ------------------------------------------------------------------
# 7. Comparación de scores
# ------------------------------------------------------------------

def fig7_score_comparison(det: pd.DataFrame):
    scores = ["nll_mc", "nll_mu", "neg_elbo", "mse", "latent_maha"]
    d = det[det["group"].isin(rcfg.MAIN_FAMILIES) & det["score"].isin(scores)]
    agg = d.groupby(["group", "score"])["auc_roc"].mean().reset_index()
    pivot = agg.pivot(index="group", columns="score", values="auc_roc")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(pivot))
    width = 0.15
    for i, score in enumerate(scores):
        if score not in pivot.columns:
            continue
        ax.bar(x + i * width, pivot[score], width=width, label=score)
    ax.set_xticks(x + width * (len(scores) - 1) / 2)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("AUC-ROC (media entre semillas)")
    ax.set_title("Comparación de scores de anomalía por familia")
    ax.legend(fontsize=8)
    savefig(fig, "fig7_score_comparison")


# ------------------------------------------------------------------
# UMAP para semillas representativas
# ------------------------------------------------------------------

def fig_umap(seed: int):
    import umap

    with open(rcfg.METRICS_DIR / "phase2_active_dims_per_seed.json", encoding="utf-8") as f:
        active_dims_map = json.load(f)
    active_dims = active_dims_map[str(seed)]

    manifest_path = rcfg.LATENT_DIR / f"seed{seed}" / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    rng = np.random.default_rng(rcfg.UMAP_RANDOM_STATE)
    mu_benign, _ = load_group_mu_logvar(seed, "latent_benign_test")
    benign_sub = subsample(mu_benign[:, active_dims], rcfg.UMAP_BENIGN_SUBSAMPLE, rng)

    all_points, all_labels = [benign_sub], ["BENIGN"] * len(benign_sub)
    for name in rcfg.MAIN_FAMILIES:
        mu_g, _ = load_group_mu_logvar(seed, name)
        all_points.append(mu_g[:, active_dims])
        all_labels.extend([name] * mu_g.shape[0])
    x = np.vstack(all_points)
    labels = np.array(all_labels)

    reducer = umap.UMAP(random_state=rcfg.UMAP_RANDOM_STATE, n_neighbors=rcfg.UMAP_N_NEIGHBORS,
                         min_dist=rcfg.UMAP_MIN_DIST, n_components=2)
    proj = reducer.fit_transform(x)

    fig, ax = plt.subplots(figsize=(9, 8))
    unique_labels = ["BENIGN"] + rcfg.MAIN_FAMILIES
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        color = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
        marker = "x" if lab == "BENIGN" else "o"
        ax.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.5, color=color, marker=marker, label=lab)
    ax.set_title(f"Proyección UMAP del espacio latente, seed={seed} "
                 f"(dims activas: {active_dims})")
    ax.legend(fontsize=7, markerscale=2, loc="upper right", bbox_to_anchor=(1.25, 1.0))
    savefig(fig, f"fig_umap_seed{seed}")


# ------------------------------------------------------------------
def main(seeds: list[int] | None = None, include_umap: bool = True):
    seeds = seeds if seeds is not None else rcfg.ALL_SEEDS
    frames = []
    for name in ["mahalanobis_summary", "divergences", "bhattacharyya",
                 "posterior_mahalanobis", "posterior_uncertainty", "neighborhood", "dispersion"]:
        path = rcfg.TABLES_DIR / f"phase3_{name}.csv"
        if path.exists():
            frames.append(pd.read_csv(path)[["seed", "group", "metric", "value"]])
    geo = pd.concat(frames, ignore_index=True)
    det = pd.read_csv(rcfg.TABLES_DIR / "phase4_detectability_metrics.csv")

    with open(rcfg.METRICS_DIR / "phase5_hypothesis_principal.json", encoding="utf-8") as f:
        h1 = json.load(f)
    with open(rcfg.METRICS_DIR / "phase5_mixed_model.json", encoding="utf-8") as f:
        mixed = json.load(f)

    representative_seed = rcfg.OFFICIAL_SEED
    fig1_mahalanobis_distributions(representative_seed, geo)
    fig2_forest_auc(det)
    fig3_scatter_geometry_detectability(geo, det, h1)
    fig4_rho_distribution(h1)
    fig5_ranking_heatmap(det)
    fig6_mixed_model_coefficients(mixed)
    fig7_score_comparison(det)

    if include_umap:
        for seed in rcfg.UMAP_SEEDS:
            if seed in seeds:
                fig_umap(seed)
    else:
        print("[fase6] UMAP omitido en esta corrida (include_umap=False) -- "
              "pendiente, ver informe sección 4.7 / 10.")


if __name__ == "__main__":
    main()
