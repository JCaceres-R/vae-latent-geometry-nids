"""
OE2 Fase 2 -- codificación de familias de ataque a mu (espacio latente).

Solo codifica y guarda: mu = E_phi[z|x] (media de q_phi(z|x)), NUNCA z
muestreado con el reparameterization trick. No calcula métricas
(Mahalanobis, silhouette, AUC-ROC) -- eso es la fase siguiente, sobre los
.npy que deja este script.

Usa el checkpoint oficial fijado en el cierre de OE1: seed=42, input_dim=51,
época 52 (outputs/checkpoints/vae_k8_beta1_input51_best.pt). No reentrena
nada -- solo carga en modo eval.

Uso:
    python -m vae_nids.evaluation.encode_latent_oe2
"""
import json

import numpy as np
import pandas as pd
import torch

from vae_nids import config as cfg
from vae_nids.models.vae import VAE, VAEConfig

CHECKPOINT_PATH = cfg.PROJECT_ROOT / "outputs" / "checkpoints" / "vae_k8_beta1_input51_best.pt"
OUTPUT_DIR = cfg.PROJECT_ROOT / "outputs" / "latent_vectors"

# Grupos por familia de ataque. Heartbleed queda fuera (n=11, insuficiente).
ATTACK_GROUPS = {
    "PortScan": ["PortScan"],
    "DoS_Hulk": ["DoS Hulk"],
    "DDoS": ["DDoS"],
    "DoS_GoldenEye": ["DoS GoldenEye"],
    "DoS_slowloris": ["DoS slowloris"],
    "FTP_Patator": ["FTP-Patator"],
    "SSH_Patator": ["SSH-Patator"],
    "DoS_Slowhttptest": ["DoS Slowhttptest"],
    "Bot": ["Bot"],
    "Web_Attack": [
        "Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection",
    ],
    "Infiltration": ["Infiltration"],
}

# Variantes "- Attempted" para la comparación completado/attempted de la
# fase siguiente. Las primeras 5 comparten label base con ATTACK_GROUPS
# (mismo label individual, ya codificado arriba) -- aquí solo se agrega la
# mitad "Attempted" que falta. El par de Web Attack usa el label base de
# Brute Force en solitario, no la fusión de 3 subtipos de Web_Attack.
ATTEMPTED_GROUPS = {
    "DoS_Hulk_attempted": ["DoS Hulk - Attempted"],
    "DoS_GoldenEye_attempted": ["DoS GoldenEye - Attempted"],
    "DoS_slowloris_attempted": ["DoS slowloris - Attempted"],
    "DoS_Slowhttptest_attempted": ["DoS Slowhttptest - Attempted"],
    "Bot_attempted": ["Bot - Attempted"],
    "WebAttack_BruteForce": ["Web Attack - Brute Force"],
    "WebAttack_BruteForce_attempted": ["Web Attack - Brute Force - Attempted"],
}

EXCLUDED_LABELS = ["Heartbleed"]


def load_model() -> tuple[VAE, list[str]]:
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    assert ckpt["seed"] == 42 and ckpt["epoch"] == 52, (
        f"Checkpoint inesperado: seed={ckpt['seed']} epoch={ckpt['epoch']} "
        f"(se esperaba el oficial seed=42 epoch=52)"
    )
    model_cfg = VAEConfig(**ckpt["config"])
    model = VAE(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["feature_columns"]


@torch.no_grad()
def encode_mu(model: VAE, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """mu = encoder(x)[0] -- nunca reparameterize/muestreo de z."""
    x = torch.from_numpy(df[feature_cols].to_numpy(dtype=np.float32, copy=True))
    mu, _logvar = model.encoder(x)
    return mu.numpy()


def main() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, feature_cols = load_model()

    manifest = {}

    benign_df = pd.read_parquet(cfg.OUTPUT_DIR / "test_benign.parquet")
    mu_benign = encode_mu(model, benign_df, feature_cols)
    fname = "latent_benign_test.npy"
    np.save(OUTPUT_DIR / fname, mu_benign)
    manifest[fname] = {"n": int(mu_benign.shape[0]), "shape": list(mu_benign.shape), "labels": ["BENIGN"]}
    print(f"[ok] {fname}: n={mu_benign.shape[0]:,} shape={mu_benign.shape}")

    attacks_df = pd.read_parquet(cfg.OUTPUT_DIR / "test_attacks.parquet")
    label_counts = attacks_df[cfg.LABEL_COL].value_counts()

    all_groups = {**ATTACK_GROUPS, **ATTEMPTED_GROUPS}
    covered_labels = set(EXCLUDED_LABELS)
    for labels in all_groups.values():
        covered_labels.update(labels)
    missing = set(label_counts.index) - covered_labels
    if missing:
        print(f"[info] labels en test_attacks.parquet no cubiertos por ningún "
              f"grupo de esta fase (no pedidos, se dejan fuera): {sorted(missing)}")

    for name, labels in all_groups.items():
        subset = attacks_df[attacks_df[cfg.LABEL_COL].isin(labels)]
        n = len(subset)
        fname = f"latent_attack_{name}.npy"
        if n == 0:
            print(f"[skip] {fname}: n=0 para labels={labels} -- no se guarda archivo")
            manifest[fname] = {"n": 0, "shape": None, "labels": labels, "skipped": True}
            continue
        mu = encode_mu(model, subset, feature_cols)
        np.save(OUTPUT_DIR / fname, mu)
        manifest[fname] = {"n": int(n), "shape": list(mu.shape), "labels": labels}
        print(f"[ok] {fname}: n={n:,} shape={mu.shape} labels={labels}")

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[done] manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    main()
