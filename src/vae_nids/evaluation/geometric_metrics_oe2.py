"""
OE2 Fase 3 -- métricas geométricas (Mahalanobis + silhouette) sobre las
dimensiones activas del espacio latente.

Sin AUC-ROC todavía -- eso es la fase siguiente. Lee los .npy y
manifest.json que dejó encode_latent_oe2.py (Fase 2).

Unidades activas: mismo criterio y misma función que OE1
(evaluation.metrics_oe1.active_units, Burda et al. 2016, umbral 0.01),
pero recalculado aquí sobre latent_benign_test.npy -- no se reutiliza el
resultado del documento de cierre de OE1, que fue de una corrida distinta
del mismo barrido de semillas.

Uso:
    python -m vae_nids.evaluation.geometric_metrics_oe2
"""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score

from vae_nids import config as cfg
from vae_nids.evaluation.metrics_oe1 import active_units

LATENT_DIR = cfg.PROJECT_ROOT / "outputs" / "latent_vectors"
METRICS_DIR = cfg.PROJECT_ROOT / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

ACTIVE_THRESHOLD = 0.01
SILHOUETTE_MAX_N = 5000
SILHOUETTE_SEED = 42


def load_manifest() -> dict:
    with open(LATENT_DIR / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


def mahalanobis_batch(x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    """d_i = sqrt((x_i - mean)^T Sigma^-1 (x_i - mean)), vectorizado."""
    diff = x - mean
    sq = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    return np.sqrt(np.maximum(sq, 0.0))


def subsample(x: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if x.shape[0] <= max_n:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_n, replace=False)
    return x[idx]


def main() -> pd.DataFrame:
    manifest = load_manifest()

    # --- Paso 0: varianza por dimensión sobre el benigno completo ---
    mu_benign_full = np.load(LATENT_DIR / "latent_benign_test.npy")
    au = active_units(torch.from_numpy(mu_benign_full), threshold=ACTIVE_THRESHOLD)
    active_dims = [i for i, a in enumerate(au["active_per_dim"]) if a]
    collapsed_dims = [i for i, a in enumerate(au["active_per_dim"]) if not a]

    print(f"[paso 0] Var_x[mu_j] sobre latent_benign_test.npy (n={mu_benign_full.shape[0]:,}):")
    for j, v in enumerate(au["variance_per_dim"]):
        estado = "activa" if j in active_dims else "COLAPSADA (excluida)"
        print(f"  dim {j}: Var_x[mu_{j}] = {v:.6f}  -> {estado}")
    print(f"[paso 0] activas ({len(active_dims)}/8): {active_dims}  "
          f"excluidas ({len(collapsed_dims)}/8): {collapsed_dims}")

    # --- Paso 1: centroide + covarianza benigno, SOLO dims activas ---
    mu_benign_active = mu_benign_full[:, active_dims]
    mean_benign = mu_benign_active.mean(axis=0)
    cov_benign = np.cov(mu_benign_active, rowvar=False)
    inv_cov_benign = np.linalg.inv(cov_benign)

    # --- Paso 2-3: por cada grupo de ataque (18 = 11 principales + 7 attempted) ---
    group_files = [name for name in manifest
                   if name != "latent_benign_test.npy" and not manifest[name].get("skipped")]

    rows = []
    for fname in group_files:
        mu_attack_full = np.load(LATENT_DIR / fname)
        mu_attack_active = mu_attack_full[:, active_dims]
        n_total = mu_attack_active.shape[0]

        dists = mahalanobis_batch(mu_attack_active, mean_benign, inv_cov_benign)

        benign_sub = subsample(mu_benign_active, SILHOUETTE_MAX_N, SILHOUETTE_SEED)
        attack_sub = subsample(mu_attack_active, SILHOUETTE_MAX_N, SILHOUETTE_SEED)
        n_used = benign_sub.shape[0] + attack_sub.shape[0]
        X = np.vstack([benign_sub, attack_sub])
        labels = np.array([0] * benign_sub.shape[0] + [1] * attack_sub.shape[0])
        sil = float(silhouette_score(X, labels))

        group_name = fname[len("latent_attack_"):-len(".npy")]
        rows.append({
            "group": group_name,
            "n_total": int(n_total),
            "n_used_silhouette": int(n_used),
            "mahalanobis_mean": float(np.mean(dists)),
            "mahalanobis_median": float(np.median(dists)),
            "mahalanobis_p25": float(np.percentile(dists, 25)),
            "mahalanobis_p75": float(np.percentile(dists, 75)),
            "silhouette_score": sil,
            "active_dims_used": str(active_dims),
        })
        print(f"[ok] {group_name}: n={n_total:,} "
              f"maha(mean/median/p25/p75)={np.mean(dists):.3f}/{np.median(dists):.3f}/"
              f"{np.percentile(dists, 25):.3f}/{np.percentile(dists, 75):.3f} "
              f"silhouette={sil:.4f} (n_silhouette={n_used:,})")

    df_out = pd.DataFrame(rows)
    out_path = METRICS_DIR / "oe2_geometric_metrics.csv"
    df_out.to_csv(out_path, index=False)
    print(f"[done] {out_path}")
    return df_out


if __name__ == "__main__":
    main()
