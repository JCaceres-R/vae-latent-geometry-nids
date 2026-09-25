"""Fase 2 -- entropía por feature (74) sobre el dataset EDA.

Salidas: metrics/oe1_entropia_features.csv, figures/oe1_entropia_features.png

Uso: python -m vae_nids.evaluation.oe1_mi.phase2_entropy
"""
import numpy as np
import pandas as pd

from vae_nids.evaluation.oe1_mi import config as cfg
from vae_nids.evaluation.oe1_mi import data
from vae_nids.evaluation.oe1_mi import estimators as est
from vae_nids.evaluation.oe1_mi import plotting as pl


def feature_entropy_table(x: np.ndarray, b: int = cfg.B_DEFAULT) -> pd.DataFrame:
    kept = set(cfg.load_features_51())
    rows = []
    for j, f in enumerate(cfg.FEATURES_74):
        col = x[:, j]
        vals, counts = np.unique(col, return_counts=True)
        d = est.discretize(col, b)
        h = est.entropy_bits(d)
        dq = est.discretize(col, b, scheme="qcut")
        rows.append({
            "feature": f,
            "estado": "conservada" if f in kept else "descartada",
            "n_unicos": int(len(vals)),
            "valor_moda": float(vals[counts.argmax()]),
            "frac_moda": float(counts.max() / len(col)),
            "intervalos_efectivos": d.n_bins,
            "categorica": d.categorical,
            "H_bits": h,
            "H_norm": h / np.log2(d.n_bins) if d.n_bins > 1 else 0.0,
            "intervalos_efectivos_qcut": dq.n_bins,
            "H_bits_qcut": est.entropy_bits(dq),
        })
    df = pd.DataFrame(rows)
    df["baja_informacion"] = (df["H_norm"] < cfg.LOW_INFO_HNORM) | (df["frac_moda"] > cfg.LOW_INFO_MODE)
    return df


def plot(df: pd.DataFrame) -> None:
    pl.setup()
    d = df.sort_values("H_bits", ascending=False).reset_index(drop=True)
    fig, ax = pl.plt.subplots(figsize=(7.5, 12))
    colors = [pl.SERIES[0] if e == "conservada" else pl.SERIES[1] for e in d["estado"]]
    y = np.arange(len(d))
    ax.barh(y, d["H_bits"], color=colors, height=0.72)
    ax.set_yticks(y, [f + ("  ⚑" if lo else "") for f, lo in zip(d["feature"], d["baja_informacion"])],
                  fontsize=7)
    ax.invert_yaxis()
    ax.axvline(np.log2(cfg.B_DEFAULT), color=pl.MUTED, lw=0.8, ls="--")
    ax.text(np.log2(cfg.B_DEFAULT), -1.2, f"log2({cfg.B_DEFAULT})", ha="center", fontsize=7, color=pl.TEXT_2)
    ax.set_xlabel("H (bits), cuantiles con átomos aislados, B = 20")
    ax.set_title("Entropía por feature (74) — dataset EDA, N = 2.100.021")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=pl.SERIES[0], label="conservada (51)"),
                       Patch(color=pl.SERIES[1], label="descartada (23)")],
              loc="lower right", title="⚑ = baja información", title_fontsize=7.5)
    fig.savefig(cfg.FIGURES_DIR / "oe1_entropia_features.png")
    pl.plt.close(fig)


def main() -> pd.DataFrame:
    x, _ = data.load_eda_matrix()
    df = feature_entropy_table(x)
    df.to_csv(cfg.METRICS_DIR / "oe1_entropia_features.csv", index=False)
    plot(df)
    print(df.sort_values("H_bits").to_string(index=False))
    return df


if __name__ == "__main__":
    main()
