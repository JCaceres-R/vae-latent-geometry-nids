"""Fase 1 -- validación del estimador con los números que van al informe
(los asserts están en tests/oe1_mi/) + calibración gaussiana de los
umbrales L_cal y S.

Calibración: un par gaussiano con |rho| = 0.95, discretizado con el MISMO
estimador (cuantiles, B intervalos, N = filas del dataset EDA), da el valor
de r_info y de nmi_max que "equivale" al umbral |r| >= 0.95 de Pearson en esa
escala. Con B finito la discretización pierde información (desigualdad de
procesamiento de datos), así que ese r_info queda por debajo de 0.95.

Salidas: metrics/oe1_validacion_estimador.json, figures/demo_no_lineal.png

Uso: python -m vae_nids.evaluation.oe1_mi.phase1_validation
"""
import json

import numpy as np

from vae_nids.evaluation.oe1_mi import config as cfg
from vae_nids.evaluation.oe1_mi import estimators as est
from vae_nids.evaluation.oe1_mi import plotting as pl
from vae_nids.evaluation.oe1_mi.data import EXPECTED_EDA_ROWS

N_HIST = 1_000_000
N_KSG = cfg.KSG_SUBSAMPLE


def _gauss(rho, n, rng):
    z = rng.standard_normal((n, 2))
    return z[:, 0], rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]


def calibrate(rho: float, b: int, n: int, seed: int = cfg.SEED) -> dict:
    x, y = _gauss(rho, n, np.random.default_rng(seed))
    r = est.mi_hist(est.discretize(x, b), est.discretize(y, b))
    return {k: r[k] for k in ("r_info", "nmi_max", "nmi_sqrt", "mi_corr_bits", "H_x")}


def main() -> dict:
    rng = np.random.default_rng(cfg.SEED)
    out: dict = {"seed": cfg.SEED}

    # Test 1: independencia
    n = N_HIST
    x, y = rng.standard_normal(n), rng.exponential(size=n)
    r = est.mi_hist(est.discretize(x, 20), est.discretize(y, 20))
    out["test1_independencia"] = {"N": n, "B": 20, **{k: r[k] for k in
                                  ("mi_bits", "bias_bits", "mi_corr_bits", "bins_x", "bins_y")}}

    # Test 2: I(X;X) = H(X)
    t2 = {}
    for kind in ("continua_lognormal", "masa_60pct_en_cero", "categorica_7"):
        if kind.startswith("continua"):
            v = rng.lognormal(size=n)
        elif kind.startswith("masa"):
            v = np.where(rng.random(n) < 0.6, 0.0, rng.exponential(size=n))
        else:
            v = rng.integers(0, 7, n).astype(float)
        d = est.discretize(v, 20)
        rr = est.mi_hist(d, d)
        t2[kind] = {"H_bits": rr["H_x"], "I_bits": rr["mi_bits"], "abs_diff": abs(rr["H_x"] - rr["mi_bits"]),
                    "bins_efectivos": d.n_bins, "nmi_sqrt": rr["nmi_sqrt"]}
    out["test2_autoinformacion"] = t2

    # Test 3: Gaussiana bivariada
    t3 = []
    for rho in (0.3, 0.7, 0.95):
        x, y = _gauss(rho, n, rng)
        row = {"rho": rho, "N_hist": n, "N_ksg": N_KSG}
        for b in (10, 20, 50, 100, 200):
            row[f"r_info_hist_B{b}"] = est.mi_hist(est.discretize(x, b), est.discretize(y, b))["r_info"]
        row["r_info_ksg"] = float(est.linfoot(est.mi_ksg_pairwise(x[:N_KSG, None], y[:N_KSG], seed=cfg.SEED)[0]))
        t3.append(row)
    out["test3_gaussiana"] = t3
    out["test3_tolerancias"] = {"ksg_n20k": 0.03, "hist_B100_N500k": 0.01,
                                "hist_B20_sesgo_abajo_max": 0.015}

    # Test 4: no lineal, no monótona
    x = rng.uniform(-1, 1, n)
    y = x ** 2 + 0.05 * rng.standard_normal(n)
    r4 = est.mi_hist(est.discretize(x, 20), est.discretize(y, 20))
    ksg4 = float(est.mi_ksg_pairwise(x[:N_KSG, None], y[:N_KSG], seed=cfg.SEED)[0])
    pearson4 = float(np.corrcoef(x, y)[0, 1])
    out["test4_no_lineal"] = {"modelo": "Y = X^2 + 0.05 N(0,1), X ~ U(-1,1)", "N": n,
                              "pearson_r": pearson4, "mi_corr_bits_B20": r4["mi_corr_bits"],
                              "nmi_sqrt_B20": r4["nmi_sqrt"], "r_info_hist_B20": r4["r_info"],
                              "mi_ksg_nats": ksg4, "r_info_ksg": float(est.linfoot(ksg4))}

    # Calibración de umbrales con N = filas del dataset EDA
    calib = {}
    for b in cfg.B_SENSITIVITY:
        c = calibrate(0.95, b, EXPECTED_EDA_ROWS)
        calib[f"B{b}"] = {
            "r_info_gauss_095": c["r_info"],        # umbral L_cal
            "nmi_max_gauss_095": c["nmi_max"],      # umbral S
            "nmi_sqrt_gauss_095": c["nmi_sqrt"],
            # derivación alternativa (analítica, sin pérdida por discretizar):
            "nmi_max_analitico": (est.linfoot_inverse_nats(0.95) / np.log(2)) / np.log2(b),
        }
    out["calibracion_umbral_095"] = calib
    out["I_nats_para_rinfo_095"] = est.linfoot_inverse_nats(0.95)
    out["H_min_bits_para_rinfo_095"] = est.linfoot_inverse_nats(0.95) / np.log(2)
    out["r_info_max_binaria"] = float(est.linfoot(np.log(2)))  # dos variables binarias idénticas equiprobables

    with open(cfg.METRICS_DIR / "oe1_validacion_estimador.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Figura demo no lineal
    pl.setup()
    fig, ax = pl.plt.subplots(figsize=(5.2, 3.8))
    sub = np.random.default_rng(cfg.SEED).choice(n, 4000, replace=False)
    ax.scatter(x[sub], y[sub], s=6, color=pl.SERIES[0], alpha=0.5, linewidths=0)
    ax.set_xlabel("X ~ U(−1, 1)")
    ax.set_ylabel("Y = X² + ruido")
    ax.set_title("Dependencia no lineal que Pearson no ve")
    txt = (f"|r| de Pearson = {abs(pearson4):.4f}\n"
           f"IM (histograma, B=20) = {r4['mi_corr_bits']:.3f} bits\n"
           f"nmi_sqrt = {r4['nmi_sqrt']:.3f}\n"
           f"r_info KSG = {out['test4_no_lineal']['r_info_ksg']:.3f}")
    ax.text(0.5, 0.97, txt, transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            color=pl.TEXT, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=pl.GRID))
    fig.savefig(cfg.FIGURES_DIR / "demo_no_lineal.png")
    pl.plt.close(fig)

    print(json.dumps(out, indent=1, ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()
