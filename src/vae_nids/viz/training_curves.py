"""
Curvas de entrenamiento y diagnóstico del espacio latente para OE1.

Lee outputs/logs/{run_name}_train_log.csv (loss total/reconstrucción/KL por
época, train y val -- ver training.train) y
outputs/metrics/{run_name}_oe1_metrics.json (KL por dimensión, unidades
activas -- ver evaluation.metrics_oe1), y guarda en outputs/figures/:
    {run_name}_training_curves.png     -> total/reconstrucción/KL vs. época
    {run_name}_latent_diagnostics.png  -> KL por dimensión + unidades activas

Paleta y specs de marca siguen la skill dataviz del proyecto (validada con
scripts/validate_palette.js: slots 1-2 azul/naranja, ΔE 24.7 CVD / 33.6
normal-vision -- pasan todos los checks). Train/val se distinguen además
por trazo (sólido/discontinuo) como canal secundario para impresión en
escala de grises.

Uso:
    python -m vae_nids.viz.training_curves --run-name vae_k8_beta1
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vae_nids import config as cfg

LOGS_DIR = cfg.PROJECT_ROOT / "outputs" / "logs"
METRICS_DIR = cfg.PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = cfg.PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLOR_TRAIN = "#2a78d6"      # categórico slot 1
COLOR_VAL = "#eb6834"        # categórico slot 2
COLOR_KL_BAR = "#2a78d6"     # una sola serie -> un solo color (no rampa)
COLOR_ACTIVE = "#0ca30c"     # status "good"
COLOR_INACTIVE = "#d03b3b"   # status "critical"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_INK = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_TEXT_MUTED = "#898781"


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_AXIS)
    ax.spines["bottom"].set_color(COLOR_AXIS)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=1, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLOR_TEXT_MUTED, labelsize=9)


def plot_training_curves(run_name: str) -> Path:
    log = pd.read_csv(LOGS_DIR / f"{run_name}_train_log.csv")

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    panels = [
        ("total", "ELBO total (reconstrucción + β·KL)"),
        ("recon", "Reconstrucción (NLL Gaussiana)"),
        ("kl", "Divergencia KL"),
    ]
    for ax, (key, title) in zip(axes, panels):
        ax.plot(log["epoch"], log[f"train_{key}"], color=COLOR_TRAIN,
                 linewidth=2, linestyle="-", solid_capstyle="round", label="Train")
        ax.plot(log["epoch"], log[f"val_{key}"], color=COLOR_VAL,
                 linewidth=2, linestyle="--", dash_capstyle="round", label="Val")
        ax.set_title(title, fontsize=10, color=COLOR_INK, loc="left")
        ax.set_ylabel("nats", fontsize=9, color=COLOR_TEXT_SECONDARY)
        _style_axes(ax)

    axes[-1].set_xlabel("Época", fontsize=9, color=COLOR_TEXT_SECONDARY)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(f"Curvas de entrenamiento -- {run_name}", fontsize=12, y=0.995)
    fig.tight_layout()

    out_path = FIGURES_DIR / f"{run_name}_training_curves.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_latent_diagnostics(run_name: str) -> Path:
    with open(METRICS_DIR / f"{run_name}_oe1_metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)

    k = metrics["active_units"]["n_total"]
    dims = list(range(k))
    kl_per_dim = metrics["kl_per_dim"]
    variance_per_dim = metrics["active_units"]["variance_per_dim"]
    active_flags = metrics["active_units"]["active_per_dim"]
    threshold = metrics["active_units"]["threshold"]
    n_active = metrics["active_units"]["n_active"]

    fig, (ax_kl, ax_active) = plt.subplots(1, 2, figsize=(11, 4.5))

    bars_kl = ax_kl.bar(dims, kl_per_dim, color=COLOR_KL_BAR, width=0.6, zorder=3)
    ax_kl.set_title("KL por dimensión latente", fontsize=10, color=COLOR_INK, loc="left")
    ax_kl.set_xlabel("Dimensión k", fontsize=9, color=COLOR_TEXT_SECONDARY)
    ax_kl.set_ylabel("KL (nats)", fontsize=9, color=COLOR_TEXT_SECONDARY)
    ax_kl.set_xticks(dims)
    for rect, val in zip(bars_kl, kl_per_dim):
        ax_kl.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=COLOR_TEXT_SECONDARY)
    _style_axes(ax_kl)

    bar_colors = [COLOR_ACTIVE if a else COLOR_INACTIVE for a in active_flags]
    bars_active = ax_active.bar(dims, variance_per_dim, color=bar_colors, width=0.6, zorder=3)
    ax_active.axhline(threshold, color=COLOR_TEXT_MUTED, linewidth=1, linestyle="--", zorder=2)
    ax_active.text(k - 0.5, threshold, f" umbral = {threshold:g}", fontsize=8,
                    color=COLOR_TEXT_MUTED, va="bottom", ha="right")
    ax_active.set_title(f"Unidades activas ({n_active}/{k}) -- Burda et al. (2016)",
                         fontsize=10, color=COLOR_INK, loc="left")
    ax_active.set_xlabel("Dimensión k", fontsize=9, color=COLOR_TEXT_SECONDARY)
    ax_active.set_ylabel("Var_x[μ_i(x)]", fontsize=9, color=COLOR_TEXT_SECONDARY)
    ax_active.set_xticks(dims)
    for rect, val in zip(bars_active, variance_per_dim):
        ax_active.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=COLOR_TEXT_SECONDARY)
    _style_axes(ax_active)

    fig.suptitle(f"Diagnóstico del espacio latente -- {run_name}", fontsize=12, y=1.02)
    fig.tight_layout()

    out_path = FIGURES_DIR / f"{run_name}_latent_diagnostics.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()
    p1 = plot_training_curves(args.run_name)
    p2 = plot_latent_diagnostics(args.run_name)
    print(f"[viz] guardado {p1}")
    print(f"[viz] guardado {p2}")


if __name__ == "__main__":
    main()
