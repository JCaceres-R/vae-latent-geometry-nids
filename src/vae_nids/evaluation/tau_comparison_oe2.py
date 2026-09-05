"""
OE2 -- análisis adicional: impacto del percentil de tau (95 vs. 99) sobre
TPR@tau.

NO modifica nada de la Fase 4 (detectability_oe2.py y su CSV quedan
intactos) -- reusa sus funciones (mismo modelo, mismo score de NLL de
reconstrucción, mismos .npy de mu de la Fase 2) para recalcular tau con
ambos percentiles y comparar TPR. AUC-ROC no depende de tau, no se toca.

tau=95 es lo que ya calibra la Fase 4 (hardcodeado en
detectability_oe2.TAU_PERCENTILE). tau=99 es el percentil especificado en
la versión vigente de la monografía -- este script cuantifica la
diferencia para decidir cuál usar.

Uso:
    python -m vae_nids.evaluation.tau_comparison_oe2
"""
import json

import numpy as np
import pandas as pd
import torch

from vae_nids import config as cfg
from vae_nids.evaluation.detectability_oe2 import encode_mu, load_model, load_x, recon_nll_scores

LATENT_DIR = cfg.PROJECT_ROOT / "outputs" / "latent_vectors"
METRICS_DIR = cfg.PROJECT_ROOT / "outputs" / "metrics"

PERCENTILES = (95, 99)


def main() -> pd.DataFrame:
    model, feature_cols = load_model()
    with open(LATENT_DIR / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    # --- tau con ambos percentiles, mismo val_benign fresco que la Fase 4 ---
    x_val = load_x("val_benign", feature_cols, labels=None)
    mu_val = encode_mu(model, x_val)
    scores_val = recon_nll_scores(model, x_val, mu_val)

    taus = {p: float(np.percentile(scores_val, p)) for p in PERCENTILES}
    print(f"[tau] percentil 95 = {taus[95]:.4f}  (vigente en detectability_oe2.py)")
    print(f"[tau] percentil 99 = {taus[99]:.4f}  (especificado en la monografía)")

    group_files = [name for name in manifest
                   if name != "latent_benign_test.npy" and not manifest[name].get("skipped")]

    rows = []
    for fname in group_files:
        info = manifest[fname]
        labels = info["labels"]
        mu_group = np.load(LATENT_DIR / fname)
        x_group = load_x("test_attacks", feature_cols, labels=labels)
        assert mu_group.shape[0] == x_group.shape[0] == info["n"], (
            f"{fname}: desalineación mu/x/manifest"
        )
        pos_scores = recon_nll_scores(model, x_group, torch.from_numpy(mu_group))

        tpr_95 = float(np.mean(pos_scores > taus[95]))
        tpr_99 = float(np.mean(pos_scores > taus[99]))

        group_name = fname[len("latent_attack_"):-len(".npy")]
        rows.append({
            "group": group_name,
            "n_total": int(len(pos_scores)),
            "tau_95": taus[95],
            "tpr_at_tau_95": tpr_95,
            "tau_99": taus[99],
            "tpr_at_tau_99": tpr_99,
            "delta_tpr": tpr_95 - tpr_99,  # positivo = tau_95 detecta más que tau_99
        })
        print(f"[ok] {group_name}: n={len(pos_scores):,} "
              f"tpr@95={tpr_95:.4f} tpr@99={tpr_99:.4f} delta={tpr_95 - tpr_99:+.4f}")

    df_out = pd.DataFrame(rows)
    out_path = METRICS_DIR / "oe2_tau_comparison.csv"
    df_out.to_csv(out_path, index=False)
    print(f"[done] {out_path}")
    return df_out


if __name__ == "__main__":
    main()
