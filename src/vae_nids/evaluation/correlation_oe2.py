"""
OE2 Fase 5 -- correlación geometría-detectabilidad (resultado central).

Une las métricas geométricas (Fase 3, oe2_geometric_metrics.csv) y de
detectabilidad (Fase 4, oe2_detectability.csv) por familia, calcula
Spearman rho entre geometría (Mahalanobis, silhouette) y detectabilidad
(AUC-ROC, TPR@tau) sobre las 11 familias principales, y compara
completado vs. attempted como chequeo secundario (6 pares).

Uso:
    python -m vae_nids.evaluation.correlation_oe2
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from vae_nids import config as cfg

METRICS_DIR = cfg.PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = cfg.PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Las 11 familias principales de la Fase 2/3 (excluye los 7 grupos
# _attempted y el WebAttack_BruteForce standalone -- esos son la
# comparación secundaria del paso 4).
MAIN_FAMILIES = [
    "PortScan", "DoS_Hulk", "DDoS", "DoS_GoldenEye", "DoS_slowloris",
    "FTP_Patator", "SSH_Patator", "DoS_Slowhttptest", "Bot", "Web_Attack",
    "Infiltration",
]

ATTEMPTED_PAIRS = [
    ("DoS_Hulk", "DoS_Hulk_attempted"),
    ("DoS_GoldenEye", "DoS_GoldenEye_attempted"),
    ("DoS_slowloris", "DoS_slowloris_attempted"),
    ("DoS_Slowhttptest", "DoS_Slowhttptest_attempted"),
    ("Bot", "Bot_attempted"),
    ("WebAttack_BruteForce", "WebAttack_BruteForce_attempted"),
]


def main() -> dict:
    geo = pd.read_csv(METRICS_DIR / "oe2_geometric_metrics.csv")
    det = pd.read_csv(METRICS_DIR / "oe2_detectability.csv")

    # --- Paso 1: merge + verificación 1 a 1 ---
    only_in_geo = set(geo["group"]) - set(det["group"])
    only_in_det = set(det["group"]) - set(geo["group"])
    assert not only_in_geo and not only_in_det, (
        f"grupos no coinciden entre archivos: solo en geo={only_in_geo}, solo en det={only_in_det}"
    )
    assert len(geo) == len(det) == 18, f"esperaba 18 filas en cada archivo, geo={len(geo)} det={len(det)}"

    df = geo.merge(det, on="group", validate="one_to_one")
    assert len(df) == 18
    print(f"[paso 1] merge OK: {len(df)} grupos, 1 a 1 confirmado entre "
          f"oe2_geometric_metrics.csv y oe2_detectability.csv.")

    # --- Paso 2: Spearman sobre las 11 familias principales ---
    main_df = df[df["group"].isin(MAIN_FAMILIES)].copy()
    assert len(main_df) == len(MAIN_FAMILIES) == 11, (
        f"esperaba 11 familias principales, encontré {len(main_df)}: {sorted(main_df['group'])}"
    )

    pairs = [
        ("mahalanobis_median", "auc_roc"),
        ("mahalanobis_median", "tpr_at_tau"),
        ("silhouette_score", "auc_roc"),
        ("silhouette_score", "tpr_at_tau"),
        ("mahalanobis_mean", "auc_roc"),
        ("mahalanobis_mean", "tpr_at_tau"),
    ]
    correlations = {}
    print(f"[paso 2] Spearman sobre {len(main_df)} familias principales:")
    for x_col, y_col in pairs:
        rho, pval = spearmanr(main_df[x_col], main_df[y_col])
        key = f"{x_col}_vs_{y_col}"
        correlations[key] = {"rho": float(rho), "p_value": float(pval), "n": int(len(main_df))}
        sig = "significativo (p<0.05)" if pval < 0.05 else "NO significativo"
        print(f"  {key}: rho={rho:.4f} p={pval:.4f} ({sig})")

    # --- Paso 3: scatter mahalanobis_median vs auc_roc, 11 familias etiquetadas ---
    rho_main = correlations["mahalanobis_median_vs_auc_roc"]["rho"]
    p_main = correlations["mahalanobis_median_vs_auc_roc"]["p_value"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(main_df["mahalanobis_median"], main_df["auc_roc"])
    for _, row in main_df.iterrows():
        ax.annotate(row["group"], (row["mahalanobis_median"], row["auc_roc"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    title = f"Geometría vs. detectabilidad (n=11 familias)\nSpearman rho={rho_main:.3f}, p={p_main:.3f}"
    if p_main >= 0.05:
        title += " -- NO significativo, sin línea de tendencia"
    ax.set_xlabel("Distancia de Mahalanobis (mediana)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title(title)
    fig.tight_layout()
    fig_path = FIGURES_DIR / "oe2_geometry_vs_detectability.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[paso 3] figura guardada: {fig_path}")

    # --- Paso 4: completado vs attempted (6 pares) ---
    df_idx = df.set_index("group")
    delta_rows = []
    for completo, attempted in ATTEMPTED_PAIRS:
        row_c, row_a = df_idx.loc[completo], df_idx.loc[attempted]
        delta_maha = float(row_c["mahalanobis_median"] - row_a["mahalanobis_median"])
        delta_auc = float(row_c["auc_roc"] - row_a["auc_roc"])
        same_sign = (delta_maha > 0) == (delta_auc > 0)
        both_positive = delta_maha > 0 and delta_auc > 0
        delta_rows.append({
            "pair": f"{completo} vs {attempted}",
            "delta_mahalanobis_median": delta_maha,
            "delta_auc_roc": delta_auc,
            "same_sign": bool(same_sign),
            "consistent_with_hypothesis": bool(both_positive),
        })

    n_same_sign = sum(r["same_sign"] for r in delta_rows)
    n_hypothesis = sum(r["consistent_with_hypothesis"] for r in delta_rows)
    print(f"[paso 4] {n_same_sign}/6 pares con el mismo signo en ambos deltas; "
          f"{n_hypothesis}/6 consistentes con la hipótesis (completo > attempted en Mahalanobis Y en AUC)")
    for r in delta_rows:
        print(f"  {r['pair']}: delta_maha={r['delta_mahalanobis_median']:+.4f} "
              f"delta_auc={r['delta_auc_roc']:+.4f} same_sign={r['same_sign']}")

    # --- Paso 5: guardar resultado ---
    result = {
        "n_main_families": int(len(main_df)),
        "correlations": correlations,
        "attempted_pairs": {
            "n_same_sign": int(n_same_sign),
            "n_consistent_with_hypothesis": int(n_hypothesis),
            "deltas": delta_rows,
        },
    }
    out_path = METRICS_DIR / "oe2_correlation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[done] {out_path}")
    return result


if __name__ == "__main__":
    main()
