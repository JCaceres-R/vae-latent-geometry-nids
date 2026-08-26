"""
Métricas de OE1 -- estabilidad del espacio latente del VAE.

- Unidades activas por dimensión (Burda, Grosse & Salakhutdinov, 2016,
  "Importance Weighted Autoencoders", ICLR, Apéndice E): A_u =
  Var_x[E_{q(z|x)}[z_i]] = Var_x[mu_i(x)]; una dimensión se considera activa
  si A_u > umbral (default 0.01, el mismo valor usado en el paper original;
  no depende del tamaño del dataset, ver discusión previa a estas 4 tareas).
- KL por dimensión: la misma fórmula cerrada de vae.kl_divergence, pero SIN
  sumar sobre las k dimensiones -- para detectar colapso parcial del
  posterior dimensión por dimensión, no solo en el agregado.

Ambas se evalúan sobre val_benign.parquet con el mejor checkpoint de una
corrida de entrenamiento.

Uso:
    python -m vae_nids.evaluation.metrics_oe1 --run-name vae_k8_beta1
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from vae_nids import config as cfg
from vae_nids.models.vae import VAE, VAEConfig

CHECKPOINT_DIR = cfg.PROJECT_ROOT / "outputs" / "checkpoints"
METRICS_DIR = cfg.PROJECT_ROOT / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def load_checkpoint(run_name: str) -> tuple[VAE, dict]:
    ckpt_path = CHECKPOINT_DIR / f"{run_name}_best.pt"
    # weights_only=False: el checkpoint es nuestro propio artefacto local
    # (config/seed/feature_columns además de los tensores), no un archivo
    # de terceros -- no aplica la superficie de ataque que weights_only
    # mitiga.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = VAE(VAEConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def encode_all(model: VAE, x: torch.Tensor, batch_size: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """mu/logvar de q(z|x) para TODO x, en batches (evita un único forward
    gigante); devuelve tensores (n, k) concatenados."""
    mus, logvars = [], []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start + batch_size]
        mu, logvar = model.encoder(xb)
        mus.append(mu)
        logvars.append(logvar)
    return torch.cat(mus, dim=0), torch.cat(logvars, dim=0)


def active_units(mu: torch.Tensor, threshold: float = 0.01) -> dict:
    """Criterio de Burda et al. (2016): A_u = Var_x[mu_i(x)]; activa si
    A_u > threshold. mu tiene shape (n_samples, k)."""
    variance_per_dim = mu.var(dim=0)
    active = variance_per_dim > threshold
    return {
        "threshold": threshold,
        "variance_per_dim": variance_per_dim.tolist(),
        "active_per_dim": active.tolist(),
        "n_active": int(active.sum()),
        "n_total": int(mu.shape[1]),
    }


def kl_per_dimension(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu_i, sigma_i^2) || N(0,1)) por dimensión, promediado sobre las
    muestras -- misma fórmula cerrada que vae.kl_divergence, sin sumar
    sobre dimensiones."""
    kl_per_dim_per_sample = 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar)
    return kl_per_dim_per_sample.mean(dim=0)


def evaluate(run_name: str, threshold: float = 0.01, batch_size: int = 4096) -> dict:
    model, ckpt = load_checkpoint(run_name)
    feature_cols = ckpt["feature_columns"]

    df = pd.read_parquet(cfg.OUTPUT_DIR / "val_benign.parquet", columns=feature_cols)
    x_val = torch.from_numpy(df[feature_cols].to_numpy(dtype=np.float32, copy=True))

    mu, logvar = encode_all(model, x_val, batch_size=batch_size)

    au = active_units(mu, threshold=threshold)
    kl_dim = kl_per_dimension(mu, logvar)

    report = {
        "run_name": run_name,
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_loss": ckpt["val_loss"],
        "seed": ckpt["seed"],
        "latent_dim": ckpt["config"]["latent_dim"],
        "beta": ckpt["config"]["beta"],
        "n_val_samples": int(x_val.shape[0]),
        "active_units": au,
        "kl_per_dim": kl_dim.tolist(),
        "kl_total_mean": float(kl_dim.sum()),
    }

    out_path = METRICS_DIR / f"{run_name}_oe1_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[oe1] {au['n_active']}/{au['n_total']} unidades activas (umbral={threshold})")
    print(f"[oe1] Var_x[mu_i] por dimensión: {[round(v, 5) for v in au['variance_per_dim']]}")
    print(f"[oe1] KL por dimensión:          {[round(v, 5) for v in kl_dim.tolist()]}")
    print(f"[oe1] reporte guardado en {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()
    evaluate(args.run_name, threshold=args.threshold)


if __name__ == "__main__":
    main()
