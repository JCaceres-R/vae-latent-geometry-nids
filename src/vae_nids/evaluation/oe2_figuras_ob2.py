"""
Genera figuras adicionales para OBJETIVO_2_DETALLADO (curva ROC, AUC con IC,
boxplot de Mahalanobis) que faltaban en el documento técnico -- reusa los
artefactos ya generados por las Fases 2-5 de OE2, no recalcula nada de esas
fases (solo redibuja / deriva un boxplot con las distancias crudas).

Uso:
    python -m vae_nids.evaluation.oe2_figuras_ob2
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vae_nids import config as cfg

LATENT_DIR = cfg.PROJECT_ROOT / "outputs" / "latent_vectors"
METRICS_DIR = cfg.PROJECT_ROOT / "outputs" / "metrics"
ROC_DIR = METRICS_DIR / "roc_curves"
OUT_DIR = cfg.PROJECT_ROOT.parent / "textos" / "figuras"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIVE_DIMS = [0, 1, 3, 4, 5, 6]

MAIN_FAMILIES = [
    "PortScan", "DoS_Hulk", "DDoS", "DoS_GoldenEye", "DoS_slowloris",
    "FTP_Patator", "SSH_Patator", "DoS_Slowhttptest", "Bot", "Web_Attack",
    "Infiltration",
]
ATTEMPTED_GROUPS = [
    "DoS_Hulk_attempted", "DoS_GoldenEye_attempted", "DoS_slowloris_attempted",
    "DoS_Slowhttptest_attempted", "Bot_attempted", "WebAttack_BruteForce",
    "WebAttack_BruteForce_attempted",
]


def fig_roc_curves(det: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, groups, title in [
        (axes[0], MAIN_FAMILIES, "Familias principales (11)"),
        (axes[1], ATTEMPTED_GROUPS, "Completado / Attempted (7)"),
    ]:
        cmap = plt.get_cmap("tab20")
        for i, name in enumerate(groups):
            npz = np.load(ROC_DIR / f"oe2_roc_{name}.npz")
            auc = det.loc[det["group"] == name, "auc_roc"].values[0]
            ax.plot(npz["fpr"], npz["tpr"], color=cmap(i / max(len(groups) - 1, 1)),
                    label=f"{name} (AUC={auc:.3f})", linewidth=1.3)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Azar (AUC=0.5)")
        ax.set_xlabel("FPR")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("TPR")
    fig.suptitle("Curvas ROC por familia (Fase 4 de OE2, $\\tau$=percentil 95)")
    fig.tight_layout()
    path = OUT_DIR / "oe2_roc_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[ok] {path}")


def fig_auc_bar(det: pd.DataFrame):
    d = det.sort_values("auc_roc", ascending=True).reset_index(drop=True)
    err_low = d["auc_roc"] - d["auc_roc_ci_low"]
    err_high = d["auc_roc_ci_high"] - d["auc_roc"]

    fig, ax = plt.subplots(figsize=(8, 8))
    y = np.arange(len(d))
    ax.barh(y, d["auc_roc"], xerr=[err_low, err_high], color="#4C72B0",
            ecolor="black", capsize=2, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(d["group"], fontsize=8)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Azar (AUC=0.5)")
    ax.set_xlabel("AUC-ROC (con IC 95% por bootstrap estratificado)")
    ax.set_title("AUC-ROC por familia, 18 grupos (Fase 4 de OE2)")
    ax.set_xlim(0.5, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = OUT_DIR / "oe2_auc_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[ok] {path}")


def fig_mahalanobis_boxplot():
    mu_benign = np.load(LATENT_DIR / "latent_benign_test.npy")[:, ACTIVE_DIMS]
    mean_b = mu_benign.mean(axis=0)
    cov_b = np.cov(mu_benign, rowvar=False)
    inv_cov = np.linalg.inv(cov_b)

    data, labels = [], []
    for name in MAIN_FAMILIES:
        mu = np.load(LATENT_DIR / f"latent_attack_{name}.npy")[:, ACTIVE_DIMS]
        diff = mu - mean_b
        d2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        dist = np.sqrt(np.maximum(d2, 0.0))
        data.append(dist)
        labels.append(name)

    # ordenar por mediana para lectura más fácil
    order = np.argsort([np.median(x) for x in data])
    data = [data[i] for i in order]
    labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot(data, vert=False, tick_labels=labels, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.6))
    ax.set_xlabel("Distancia de Mahalanobis (6 dims activas)")
    ax.set_title("Distribución de Mahalanobis por familia, 11 familias principales\n"
                  "(caja = IQR, línea = mediana; outliers omitidos por escala)")
    fig.tight_layout()
    path = OUT_DIR / "oe2_mahalanobis_boxplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[ok] {path}")


def fig_tau_comparison():
    df = pd.read_csv(METRICS_DIR / "oe2_tau_comparison.csv")
    df = df.sort_values("tpr_at_tau_95", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 8))
    h = 0.35
    ax.barh(y - h / 2, df["tpr_at_tau_95"], height=h, color="#4C72B0", label="TPR@$\\tau$ (percentil 95)")
    ax.barh(y + h / 2, df["tpr_at_tau_99"], height=h, color="#DD8452", label="TPR@$\\tau$ (percentil 99)")
    ax.set_yticks(y)
    ax.set_yticklabels(df["group"], fontsize=8)
    ax.set_xlabel("TPR")
    ax.set_title("Impacto del percentil de $\\tau$ sobre TPR, 18 grupos")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = OUT_DIR / "oe2_tau_comparison_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[ok] {path}")


def main():
    det = pd.read_csv(METRICS_DIR / "oe2_detectability.csv")
    fig_roc_curves(det)
    fig_auc_bar(det)
    fig_mahalanobis_boxplot()
    fig_tau_comparison()


if __name__ == "__main__":
    main()
