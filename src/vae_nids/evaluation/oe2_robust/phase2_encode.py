"""Fase 2 -- codificación a (mu, log sigma^2) por semilla.

A diferencia de encode_latent_oe2.py (Fase 2 del OE2 original, que solo
guarda mu), aquí se guardan AMBOS parámetros del posterior -- log sigma^2
hace falta para: Mahalanobis esperada bajo el posterior (Fase 3.3),
NLL Monte Carlo (Fase 4.1), y el -ELBO completo con L muestras.

Mismos 19 grupos, mismo manifest, mismas exclusiones que
encode_latent_oe2.ATTACK_GROUPS / ATTEMPTED_GROUPS / EXCLUDED_LABELS
(importados desde allí, no redefinidos) -- por semilla, bajo
outputs/oe2_robust/latent_vectors/seed{S}/.

Uso:
    python -m vae_nids.evaluation.oe2_robust.phase2_encode
    python -m vae_nids.evaluation.oe2_robust.phase2_encode --seed 0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vae_nids import config as base_cfg
from vae_nids.evaluation.metrics_oe1 import active_units
from vae_nids.evaluation.oe2_robust import config as rcfg
from vae_nids.models.vae import VAE, VAEConfig

ACTIVE_THRESHOLD = rcfg.ACTIVE_THRESHOLD


def checkpoint_path_for(seed: int) -> Path:
    if seed == rcfg.OFFICIAL_SEED:
        return rcfg.OFFICIAL_CHECKPOINT_PATH
    return rcfg.CHECKPOINTS_DIR / f"vae_k8_beta1_input51_seed{seed}_best.pt"


def load_model(seed: int) -> tuple[VAE, list[str]]:
    ckpt = torch.load(checkpoint_path_for(seed), map_location="cpu", weights_only=False)
    assert ckpt["seed"] == seed, f"checkpoint de seed={ckpt['seed']}, esperaba seed={seed}"
    model = VAE(VAEConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["feature_columns"]


@torch.no_grad()
def encode_mu_logvar(model: VAE, df: pd.DataFrame, feature_cols: list[str],
                      batch_size: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(df[feature_cols].to_numpy(dtype=np.float32, copy=True))
    mus, logvars = [], []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start + batch_size]
        mu, logvar = model.encoder(xb)
        mus.append(mu)
        logvars.append(logvar)
    return torch.cat(mus).numpy(), torch.cat(logvars).numpy()


def encode_seed(seed: int) -> dict:
    out_dir = rcfg.LATENT_DIR / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, feature_cols = load_model(seed)
    manifest = {}

    benign_df = pd.read_parquet(base_cfg.OUTPUT_DIR / "test_benign.parquet")
    mu_b, logvar_b = encode_mu_logvar(model, benign_df, feature_cols)
    np.save(out_dir / "latent_benign_test_mu.npy", mu_b)
    np.save(out_dir / "latent_benign_test_logvar.npy", logvar_b)
    manifest["latent_benign_test"] = {"n": int(mu_b.shape[0]), "labels": ["BENIGN"]}
    print(f"[fase2 seed={seed}] BENIGN test: n={mu_b.shape[0]:,}")

    attacks_df = pd.read_parquet(base_cfg.OUTPUT_DIR / "test_attacks.parquet")
    all_groups = {**rcfg.ATTACK_GROUPS, **rcfg.ATTEMPTED_GROUPS}
    for name, labels in all_groups.items():
        subset = attacks_df[attacks_df[base_cfg.LABEL_COL].isin(labels)]
        n = len(subset)
        if n == 0:
            manifest[name] = {"n": 0, "labels": labels, "skipped": True}
            continue
        mu, logvar = encode_mu_logvar(model, subset, feature_cols)
        np.save(out_dir / f"latent_attack_{name}_mu.npy", mu)
        np.save(out_dir / f"latent_attack_{name}_logvar.npy", logvar)
        manifest[name] = {"n": int(n), "labels": labels}

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # --- Verificación de conteos contra el manifest del OE2 original ---
    original_manifest_path = base_cfg.PROJECT_ROOT / "outputs" / "latent_vectors" / "manifest.json"
    with open(original_manifest_path, encoding="utf-8") as f:
        original_manifest = json.load(f)
    mismatches = []
    for fname, info in original_manifest.items():
        if info.get("skipped"):
            continue
        name = "latent_benign_test" if fname == "latent_benign_test.npy" else fname[len("latent_attack_"):-len(".npy")]
        if name not in manifest:
            mismatches.append(f"{name}: ausente en manifest nuevo")
        elif manifest[name].get("n") != info["n"]:
            mismatches.append(f"{name}: n_original={info['n']} vs n_nuevo={manifest[name].get('n')}")
    if mismatches:
        raise AssertionError(f"[fase2 seed={seed}] conteos no coinciden con el manifest original: {mismatches}")
    print(f"[fase2 seed={seed}] conteos verificados contra manifest original de OE2: OK ({len(manifest)} grupos)")

    # --- Unidades activas (Burda et al. 2016) sobre benigno de esta semilla ---
    au = active_units(torch.from_numpy(mu_b), threshold=ACTIVE_THRESHOLD)
    active_dims = [i for i, a in enumerate(au["active_per_dim"]) if a]
    print(f"[fase2 seed={seed}] dims activas ({len(active_dims)}/8): {active_dims} "
          f"varianza={[round(v, 5) for v in au['variance_per_dim']]}")

    return {
        "seed": seed,
        "n_groups": len(manifest),
        "active_units": au,
        "active_dims": active_dims,
    }


def main(seeds: list[int] | None = None) -> list[dict]:
    seeds = seeds if seeds is not None else rcfg.ALL_SEEDS
    rows = []
    for seed in seeds:
        rows.append(encode_seed(seed))

    # --- Tabla semilla x dimensión ---
    table_rows = []
    for r in rows:
        row = {"seed": r["seed"], "n_active": r["active_units"]["n_active"]}
        for j, v in enumerate(r["active_units"]["variance_per_dim"]):
            row[f"var_dim{j}"] = v
            row[f"active_dim{j}"] = r["active_units"]["active_per_dim"][j]
        table_rows.append(row)
    import csv
    out_csv = rcfg.TABLES_DIR / "phase2_active_dims_per_seed.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"[fase2] tabla semilla x dimensión guardada en {out_csv}")

    active_dims_map = {r["seed"]: r["active_dims"] for r in rows}
    out_json = rcfg.METRICS_DIR / "phase2_active_dims_per_seed.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(active_dims_map, f, indent=2)

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(seeds=[args.seed] if args.seed is not None else None)
