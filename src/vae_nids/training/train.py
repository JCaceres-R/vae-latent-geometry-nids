"""
Entrenamiento del VAE de OE1 sobre CICIDS2017 depurado.

Carga train_benign.parquet / val_benign.parquet ya escalados por
`vae_nids.data.pipeline` (no se reescala aquí), entrena con early stopping
sobre el ELBO total de validación, logea loss total/reconstrucción/KL por
época en CSV, y guarda el mejor checkpoint.

Uso:
    python -m vae_nids.training.train --latent-dim 8 --beta 1.0
"""
import argparse
import csv
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vae_nids import config as cfg
from vae_nids.models.vae import VAE, VAEConfig, elbo_loss, load_feature_columns

# outputs/ ya tiene checkpoints/ y figures/; logs/ es nueva -- separada de
# los .pt para no mezclar artefactos de entrenamiento con el modelo.
CHECKPOINT_DIR = cfg.PROJECT_ROOT / "outputs" / "checkpoints"
LOGS_DIR = cfg.PROJECT_ROOT / "outputs" / "logs"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_split(name: str, feature_cols: list[str]) -> torch.Tensor:
    """Carga un split ya escalado por el pipeline y selecciona feature_cols
    en su orden exacto (el mismo con el que se ajustó scaler.joblib) --
    pd.read_parquet(columns=...) no garantiza ese orden, así que se
    reindexa explícitamente."""
    df = pd.read_parquet(cfg.OUTPUT_DIR / f"{name}.parquet", columns=feature_cols)
    return torch.from_numpy(df[feature_cols].to_numpy(dtype=np.float32, copy=True))


def run_epoch_train(
    model: VAE, x: torch.Tensor, optimizer: torch.optim.Optimizer,
    beta: float, batch_size: int, device: torch.device,
) -> tuple[float, float, float]:
    model.train()
    n = x.shape[0]
    perm = torch.randperm(n)  # misma corriente de RNG que torch.manual_seed(seed) en train()
    total_sum = recon_sum = kl_sum = 0.0
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        xb = x[idx].to(device)
        outputs = model(xb)
        total, recon, kl = elbo_loss(xb, outputs, beta)

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        bs = xb.shape[0]
        total_sum += total.item() * bs
        recon_sum += recon.item() * bs
        kl_sum += kl.item() * bs
    return total_sum / n, recon_sum / n, kl_sum / n


@torch.no_grad()
def run_epoch_eval(
    model: VAE, x: torch.Tensor, beta: float, batch_size: int, device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    n = x.shape[0]
    total_sum = recon_sum = kl_sum = 0.0
    for start in range(0, n, batch_size):
        xb = x[start:start + batch_size].to(device)
        outputs = model(xb)
        total, recon, kl = elbo_loss(xb, outputs, beta)
        bs = xb.shape[0]
        total_sum += total.item() * bs
        recon_sum += recon.item() * bs
        kl_sum += kl.item() * bs
    return total_sum / n, recon_sum / n, kl_sum / n


def train(
    latent_dim: int = 8,
    beta: float = 1.0,
    hidden_dim: int = 32,
    batch_size: int = 1024,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 10,
    min_delta: float = 0.0,
    seed: int = cfg.RANDOM_SEED,
    run_name: str | None = None,
) -> Path:
    # Una sola semilla gobierna TODO el stream de RNG determinista de esta
    # corrida: la inicialización de pesos de VAE(...) más abajo, y cada
    # torch.randperm() de run_epoch_train en las épocas siguientes (ambos
    # consumen el generador global de torch en el orden en que se llaman).
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_cols = load_feature_columns()
    x_train = load_split("train_benign", feature_cols)
    x_val = load_split("val_benign", feature_cols)

    run_name = run_name or f"vae_k{latent_dim}_beta{beta:g}"

    model_cfg = VAEConfig(
        input_dim=len(feature_cols), hidden_dim=hidden_dim, latent_dim=latent_dim, beta=beta,
    )
    model = VAE(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log_path = LOGS_DIR / f"{run_name}_train_log.csv"
    checkpoint_path = CHECKPOINT_DIR / f"{run_name}_best.pt"
    fieldnames = ["epoch", "train_total", "train_recon", "train_kl",
                  "val_total", "val_recon", "val_kl", "seconds"]

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print(f"[train] run_name={run_name} device={device} "
          f"n_train={x_train.shape[0]:,} n_val={x_val.shape[0]:,} "
          f"input_dim={model_cfg.input_dim} latent_dim={latent_dim} beta={beta} seed={seed}")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, max_epochs + 1):
            t0 = time.perf_counter()
            train_total, train_recon, train_kl = run_epoch_train(
                model, x_train, optimizer, beta, batch_size, device
            )
            val_total, val_recon, val_kl = run_epoch_eval(
                model, x_val, beta, batch_size, device
            )
            elapsed = time.perf_counter() - t0

            writer.writerow({
                "epoch": epoch, "train_total": train_total, "train_recon": train_recon,
                "train_kl": train_kl, "val_total": val_total, "val_recon": val_recon,
                "val_kl": val_kl, "seconds": elapsed,
            })
            f.flush()

            print(f"[epoch {epoch:03d}] train_total={train_total:.4f} "
                  f"val_total={val_total:.4f} val_recon={val_recon:.4f} val_kl={val_kl:.4f} "
                  f"({elapsed:.1f}s)")

            if val_total < best_val_loss - min_delta:
                best_val_loss = val_total
                epochs_without_improvement = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": asdict(model_cfg),
                    "seed": seed,
                    "epoch": epoch,
                    "val_loss": val_total,
                    "val_recon": val_recon,
                    "val_kl": val_kl,
                    "feature_columns": feature_cols,
                }, checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"[early stop] sin mejora en {patience} épocas "
                          f"(mejor val_total={best_val_loss:.4f} en época {epoch - patience})")
                    break

    print(f"[done] checkpoint: {checkpoint_path}")
    print(f"[done] log: {log_path}")
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    train(
        latent_dim=args.latent_dim, beta=args.beta, hidden_dim=args.hidden_dim,
        batch_size=args.batch_size, lr=args.lr, max_epochs=args.max_epochs,
        patience=args.patience, min_delta=args.min_delta, seed=args.seed,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
