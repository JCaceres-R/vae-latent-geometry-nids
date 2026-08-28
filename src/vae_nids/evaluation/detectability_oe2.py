"""
OE2 Fase 4 -- detectabilidad por familia (AUC-ROC / TPR).

Sin correlación con la geometría todavía -- eso es la Fase 5. Score de
anomalía = NLL de reconstrucción Gaussiana heteroscedástica por muestra
(misma forma funcional que vae.gaussian_nll, ecuación 5 del documento de
fundamentos, pero SIN promediar sobre el batch -- un score por fila), NUNCA
el ELBO completo (sin beta*KL) ni MSE plano. Score más alto = más anómalo.

Usa las mu de 8 dimensiones completas guardadas por encode_latent_oe2.py
(Fase 2) -- NO las 6 dims activas de geometric_metrics_oe2.py (Fase 3); el
decoder espera las 8 dimensiones tal como las produce el encoder, las 2
colapsadas incluidas (casi no varían, pero siguen siendo input válido).

Para el score de reconstrucción hace falta x original además de mu (el
residual (x - mu_x_hat) depende de x) -- se recarga x desde los parquet
filtrando por los mismos labels que dejó manifest.json, lo que reproduce
exactamente el mismo subconjunto/orden que generó cada mu.npy (mismo
archivo sin tocar, mismo filtro, sin aleatoriedad de por medio).

Uso:
    python -m vae_nids.evaluation.detectability_oe2
"""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from vae_nids import config as cfg
from vae_nids.models.vae import VAE, VAEConfig

CHECKPOINT_PATH = cfg.PROJECT_ROOT / "outputs" / "checkpoints" / "vae_k8_beta1_input51_best.pt"
LATENT_DIR = cfg.PROJECT_ROOT / "outputs" / "latent_vectors"
METRICS_DIR = cfg.PROJECT_ROOT / "outputs" / "metrics"
ROC_DIR = METRICS_DIR / "roc_curves"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
ROC_DIR.mkdir(parents=True, exist_ok=True)

TAU_PERCENTILE = 95
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


def load_model() -> tuple[VAE, list[str]]:
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    assert ckpt["seed"] == 42 and ckpt["epoch"] == 52, (
        f"Checkpoint inesperado: seed={ckpt['seed']} epoch={ckpt['epoch']}"
    )
    model = VAE(VAEConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["feature_columns"]


@torch.no_grad()
def encode_mu(model: VAE, x: torch.Tensor) -> torch.Tensor:
    mu, _logvar = model.encoder(x)
    return mu


@torch.no_grad()
def recon_nll_scores(model: VAE, x: torch.Tensor, mu: torch.Tensor) -> np.ndarray:
    """NLL de reconstrucción por muestra: decoder(mu) -> (mu_x_hat,
    logvar_x_hat) -> NLL Gaussiana heteroscedástica, sin sumar/promediar
    sobre el batch (un score por fila, no un escalar). Sin término KL."""
    mu_x_hat, logvar_x_hat = model.decoder(mu)
    var_hat = torch.exp(logvar_x_hat)
    log_2pi = torch.log(torch.tensor(2.0 * torch.pi))
    nll_per_dim = 0.5 * (log_2pi + logvar_x_hat + (x - mu_x_hat) ** 2 / var_hat)
    return nll_per_dim.sum(dim=1).numpy()


def load_x(parquet_name: str, feature_cols: list[str], labels: list[str] | None) -> torch.Tensor:
    df = pd.read_parquet(cfg.OUTPUT_DIR / f"{parquet_name}.parquet")
    if labels is not None:
        df = df[df[cfg.LABEL_COL].isin(labels)]
    return torch.from_numpy(df[feature_cols].to_numpy(dtype=np.float32, copy=True))


def bootstrap_auc_ci(pos_scores: np.ndarray, neg_scores: np.ndarray,
                      n_boot: int, seed: int) -> tuple[float, float]:
    """Bootstrap estratificado (positivos y negativos remuestreados por
    separado, tamaños fijos) -- evita perder una clase entera en grupos
    chicos (ej. Infiltration, n=32)."""
    rng = np.random.default_rng(seed)
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        sp = pos_scores[rng.integers(0, n_pos, n_pos)]
        sn = neg_scores[rng.integers(0, n_neg, n_neg)]
        aucs[b] = roc_auc_score(y_true, np.concatenate([sp, sn]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> pd.DataFrame:
    model, feature_cols = load_model()
    with open(LATENT_DIR / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    # --- Paso 1: calibrar tau sobre val_benign (fresco, sin reusar ningún .npy) ---
    x_val = load_x("val_benign", feature_cols, labels=None)
    mu_val = encode_mu(model, x_val)
    scores_val = recon_nll_scores(model, x_val, mu_val)
    tau = float(np.percentile(scores_val, TAU_PERCENTILE))
    print(f"[paso 1] tau = percentil {TAU_PERCENTILE} de val_benign "
          f"(n={len(scores_val):,}) = {tau:.4f}")

    # --- Referencia negativa: test_benign, mu ya guardada + x recargada ---
    x_test_benign = load_x("test_benign", feature_cols, labels=None)
    mu_test_benign = np.load(LATENT_DIR / "latent_benign_test.npy")
    assert mu_test_benign.shape[0] == x_test_benign.shape[0]
    neg_scores = recon_nll_scores(model, x_test_benign, torch.from_numpy(mu_test_benign))
    print(f"[ref] test_benign n={len(neg_scores):,} score_mean={neg_scores.mean():.4f}")

    group_files = [name for name in manifest
                   if name != "latent_benign_test.npy" and not manifest[name].get("skipped")]

    rows = []
    for fname in group_files:
        info = manifest[fname]
        labels = info["labels"]
        mu_group = np.load(LATENT_DIR / fname)
        x_group = load_x("test_attacks", feature_cols, labels=labels)
        assert mu_group.shape[0] == x_group.shape[0] == info["n"], (
            f"{fname}: desalineación mu/x/manifest "
            f"({mu_group.shape[0]}/{x_group.shape[0]}/{info['n']})"
        )
        pos_scores = recon_nll_scores(model, x_group, torch.from_numpy(mu_group))

        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_score = np.concatenate([pos_scores, neg_scores])
        auc = float(roc_auc_score(y_true, y_score))
        ci_lo, ci_hi = bootstrap_auc_ci(pos_scores, neg_scores, N_BOOTSTRAP, BOOTSTRAP_SEED)
        tpr_tau = float(np.mean(pos_scores > tau))

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        group_name = fname[len("latent_attack_"):-len(".npy")]
        np.savez(ROC_DIR / f"oe2_roc_{group_name}.npz", fpr=fpr, tpr=tpr, thresholds=thresholds)

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
              f"CI95=[{ci_lo:.4f},{ci_hi:.4f}] (ancho={ci_hi - ci_lo:.4f}) tpr@tau={tpr_tau:.4f}")

    df_out = pd.DataFrame(rows)
    out_path = METRICS_DIR / "oe2_detectability.csv"
    df_out.to_csv(out_path, index=False)
    print(f"[done] {out_path}")
    return df_out


if __name__ == "__main__":
    main()
