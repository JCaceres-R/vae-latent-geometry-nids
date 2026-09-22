"""Entrenamiento de una semilla del VAE para OE2 robusto -- misma lógica de
`vae_nids.training.train` (reutiliza `load_split`, `run_epoch_train`,
`run_epoch_eval` de ese módulo sin modificarlo), pero escribiendo
checkpoints/logs bajo `outputs/oe2_robust/` en vez de `outputs/checkpoints`
y `outputs/logs`, para no tocar ni sobrescribir nada del entrenamiento
oficial de OE1/OE2 (regla no negociable de la tarea).

Uso:
    python -m vae_nids.evaluation.oe2_robust.train_seed --seed 0
"""
import argparse
import csv
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from vae_nids.evaluation.oe2_robust import config as rcfg
from vae_nids.models.vae import VAE, VAEConfig, elbo_loss, load_feature_columns
from vae_nids.training.train import load_split, run_epoch_eval, run_epoch_train


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[4],
        ).decode().strip()
    except Exception:
        return "unknown"


def library_versions() -> dict:
    import pandas
    import sklearn
    import scipy
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
    }


def train_one_seed(
    seed: int,
    run_name: str,
    max_epochs: int = rcfg.MAX_EPOCHS,
    patience: int = rcfg.PATIENCE,
    min_delta: float = rcfg.MIN_DELTA,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_cols = load_feature_columns()
    x_train = load_split("train_benign", feature_cols)
    x_val = load_split("val_benign", feature_cols)

    model_cfg = VAEConfig(
        input_dim=len(feature_cols), hidden_dim=rcfg.HIDDEN_DIM,
        latent_dim=rcfg.LATENT_DIM, beta=rcfg.BETA,
    )
    model = VAE(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=rcfg.LR)

    log_path = rcfg.LOGS_DIR / f"{run_name}_train_log.csv"
    checkpoint_path = rcfg.CHECKPOINTS_DIR / f"{run_name}_best.pt"
    fieldnames = ["epoch", "train_total", "train_recon", "train_kl",
                  "val_total", "val_recon", "val_kl", "seconds"]

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    t_start = time.perf_counter()

    print(f"[oe2_robust/train] run_name={run_name} seed={seed} device={device} "
          f"n_train={x_train.shape[0]:,} n_val={x_val.shape[0]:,}")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, max_epochs + 1):
            t0 = time.perf_counter()
            train_total, train_recon, train_kl = run_epoch_train(
                model, x_train, optimizer, rcfg.BETA, rcfg.BATCH_SIZE, device
            )
            val_total, val_recon, val_kl = run_epoch_eval(
                model, x_val, rcfg.BETA, rcfg.BATCH_SIZE, device
            )
            elapsed = time.perf_counter() - t0

            writer.writerow({
                "epoch": epoch, "train_total": train_total, "train_recon": train_recon,
                "train_kl": train_kl, "val_total": val_total, "val_recon": val_recon,
                "val_kl": val_kl, "seconds": elapsed,
            })
            f.flush()

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
                    print(f"[oe2_robust/train] early stop en epoch {epoch} "
                          f"(mejor val_total={best_val_loss:.4f} en epoch {epoch - patience})")
                    break

    total_seconds = time.perf_counter() - t_start
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    metadata = {
        "run_name": run_name,
        "seed": seed,
        "checkpoint_epoch": ckpt["epoch"],
        "val_loss": ckpt["val_loss"],
        "val_recon": ckpt["val_recon"],
        "val_kl": ckpt["val_kl"],
        "total_train_seconds": total_seconds,
        "device": str(device),
        "git_commit": git_commit_hash(),
        "library_versions": library_versions(),
        "config": asdict(model_cfg),
    }
    meta_path = rcfg.CHECKPOINTS_DIR / f"{run_name}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[oe2_robust/train] done: epoch={ckpt['epoch']} val_loss={ckpt['val_loss']:.4f} "
          f"seconds={total_seconds:.1f} checkpoint={checkpoint_path}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=rcfg.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=rcfg.PATIENCE)
    args = parser.parse_args()
    run_name = args.run_name or f"vae_k8_beta1_input51_seed{args.seed}"
    train_one_seed(args.seed, run_name, max_epochs=args.max_epochs, patience=args.patience)


if __name__ == "__main__":
    main()
