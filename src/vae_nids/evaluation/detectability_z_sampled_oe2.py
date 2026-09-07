"""
OE2 -- variante de la Fase 4 con z muestreado en vez de mu (Tarea 2 del
análisis adicional post-cierre).

Corre el mismo pipeline que detectability_oe2.py pero alimenta el decoder
con z ~ q_phi(z|x) muestreado vía reparameterization trick
(VAE.reparameterize, vae.py:89-94), consistente con la ecuación 5 de la
monografía -- en vez de mu determinístico. NO reemplaza
detectability_oe2.py ni oe2_detectability.csv: genera un archivo en
paralelo (oe2_detectability_z_sampled.csv) para comparar ambas variantes.

No depende de los .npy de la Fase 2 (que deliberadamente solo guardan mu,
sin logvar) -- el encoder se corre fresco sobre la x recargada de cada
grupo, igual que ya hacía detectability_oe2.py para el residual de NLL.

Reproducibilidad: el muestreo de z es estocástico (eps ~ N(0,I)) -- se
fija torch.manual_seed(42) una sola vez al inicio, que gobierna todo el
stream de RNG en el orden en que se consume (val_benign, test_benign,
luego cada uno de los 18 grupos, en ese orden), mismo patrón que usa
training/train.py para una corrida reproducible.

Mismo TAU_PERCENTILE que la Fase 4 original (95 -- sin resolver todavía,
ver Tarea 1 de tau) para que la única diferencia entre este archivo y
oe2_detectability.csv sea mu vs. z, no el percentil de tau.

Uso:
    python -m vae_nids.evaluation.detectability_z_sampled_oe2
"""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from vae_nids import config as cfg
from vae_nids.evaluation.detectability_oe2 import (
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    TAU_PERCENTILE,
    bootstrap_auc_ci,
    load_model,
    load_x,
    recon_nll_scores,
)

LATENT_DIR = cfg.PROJECT_ROOT / "outputs" / "latent_vectors"
METRICS_DIR = cfg.PROJECT_ROOT / "outputs" / "metrics"

Z_SAMPLE_SEED = 42


@torch.no_grad()
def encode_z(model, x: torch.Tensor) -> torch.Tensor:
    """z ~ q_phi(z|x) muestreado con el reparameterization trick -- NO mu."""
    mu, logvar = model.encoder(x)
    return model.reparameterize(mu, logvar)


def main() -> pd.DataFrame:
    torch.manual_seed(Z_SAMPLE_SEED)
    model, feature_cols = load_model()
    with open(LATENT_DIR / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    x_val = load_x("val_benign", feature_cols, labels=None)
    z_val = encode_z(model, x_val)
    scores_val = recon_nll_scores(model, x_val, z_val)
    tau = float(np.percentile(scores_val, TAU_PERCENTILE))
    print(f"[paso 1] tau (z muestreado, percentil {TAU_PERCENTILE}) = {tau:.4f}")

    x_test_benign = load_x("test_benign", feature_cols, labels=None)
    z_test_benign = encode_z(model, x_test_benign)
    neg_scores = recon_nll_scores(model, x_test_benign, z_test_benign)
    print(f"[ref] test_benign (z) n={len(neg_scores):,} score_mean={neg_scores.mean():.4f}")

    group_files = [name for name in manifest
                   if name != "latent_benign_test.npy" and not manifest[name].get("skipped")]

    rows = []
    for fname in group_files:
        info = manifest[fname]
        labels = info["labels"]
        x_group = load_x("test_attacks", feature_cols, labels=labels)
        assert x_group.shape[0] == info["n"], f"{fname}: desalineación x/manifest"
        z_group = encode_z(model, x_group)
        pos_scores = recon_nll_scores(model, x_group, z_group)

        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_score = np.concatenate([pos_scores, neg_scores])
        auc = float(roc_auc_score(y_true, y_score))
        ci_lo, ci_hi = bootstrap_auc_ci(pos_scores, neg_scores, N_BOOTSTRAP, BOOTSTRAP_SEED)
        tpr_tau = float(np.mean(pos_scores > tau))

        group_name = fname[len("latent_attack_"):-len(".npy")]
        rows.append({
            "group": group_name,
            "n_total": int(len(pos_scores)),
            "auc_roc": auc,
            "auc_roc_ci_low": ci_lo,
            "auc_roc_ci_high": ci_hi,
            "tpr_at_tau": tpr_tau,
            "tau_value": tau,
        })
        print(f"[ok] {group_name}: n={len(pos_scores):,} auc={auc:.4f} "
              f"CI95=[{ci_lo:.4f},{ci_hi:.4f}] tpr@tau={tpr_tau:.4f}")

    df_out = pd.DataFrame(rows)
    out_path = METRICS_DIR / "oe2_detectability_z_sampled.csv"
    df_out.to_csv(out_path, index=False)
    print(f"[done] {out_path}")
    return df_out


if __name__ == "__main__":
    main()
