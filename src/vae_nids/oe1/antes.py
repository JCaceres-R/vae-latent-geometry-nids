"""Diagnósticos del OE1 ACTUAL (51 columnas, pipeline original) con las mismas medidas que el saneado,
para la tabla antes/después (7.4). Solo benigno (val_benign del pipeline original, ya escalado).

Uso:
    python -m vae_nids.oe1.antes --config configs/oe1_saneado.yaml
"""
import argparse

import numpy as np
import pandas as pd
import torch

from vae_nids import config as base_cfg
from vae_nids.models.vae import VAE, VAEConfig
from vae_nids.oe1.config import load_config, resolve

LOG2PI = float(np.log(2 * np.pi))
CKPTS = {42: "vae_k8_beta1_input51_best.pt", 43: "vae_k8_beta1_input51_seed43_best.pt",
         44: "vae_k8_beta1_input51_seed44_best.pt", 45: "vae_k8_beta1_input51_seed45_best.pt",
         46: "vae_k8_beta1_input51_seed46_best.pt"}
LOGS = {42: "vae_k8_beta1_input51_train_log.csv", 43: "vae_k8_beta1_input51_seed43_train_log.csv",
        44: "vae_k8_beta1_input51_seed44_train_log.csv", 45: "vae_k8_beta1_input51_seed45_train_log.csv",
        46: "vae_k8_beta1_input51_seed46_train_log.csv"}


@torch.no_grad()
def main(config=None):
    cfg = load_config(config)
    root = base_cfg.PROJECT_ROOT / "outputs"
    rows = []
    val_cache = {}
    for seed, f in CKPTS.items():
        ck = torch.load(root / "checkpoints" / f, map_location="cpu", weights_only=False)
        cols = ck["feature_columns"]
        if tuple(cols) not in val_cache:
            val_cache[tuple(cols)] = pd.read_parquet(base_cfg.OUTPUT_DIR / "val_benign.parquet", columns=cols)
        x = torch.from_numpy(val_cache[tuple(cols)][cols].to_numpy(np.float32))
        m = VAE(VAEConfig(**ck["config"]))
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        mus, lvs, nlls, lxs = [], [], [], []
        for s in range(0, len(x), 8192):
            xb = x[s:s + 8192]
            mu, lv = m.encoder(xb)
            mx, lx = m.decoder(mu)
            nlls.append((0.5 * (LOG2PI + lx + (xb - mx) ** 2 / lx.exp())).sum(1))
            mus.append(mu); lvs.append(lv); lxs.append(lx)
        mu, lv, nll, lx = (torch.cat(a).numpy() for a in (mus, lvs, nlls, lxs))
        kl = 0.5 * (np.exp(lv) + mu ** 2 - 1 - lv).mean(0)
        log = pd.read_csv(root / "logs" / LOGS[seed])
        piso = cfg["modelo"]["logvar_min"]
        rows.append(dict(seed=seed, n_columnas=len(cols), epoca_parada=int(log.epoch.max()), mejor_epoca=int(ck["epoch"]),
                         mejor_val_total=float(ck["val_loss"]), activas_burda=int((mu.var(0) > 0.01).sum()),
                         activas_kl=int((kl > 0.01).sum()), kl_total=float(kl.sum()),
                         tau_p95=float(np.percentile(nll, cfg["evaluacion"]["tau_percentil"])),
                         nll_media=float(nll.mean()), nll_media_por_columna=float(nll.mean() / len(cols)),
                         columnas_en_piso=int((np.median(lx, 0) <= piso + 1e-6).sum()),
                         frac_logvar_dec_en_piso=float((lx <= piso + 1e-6).mean()), corrida="actual_51"))
    t = pd.DataFrame(rows)
    out = resolve(cfg, "salidas") / "tables"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "diagnosticos_oe1_actual_51.csv", index=False)
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    print(main(ap.parse_args().config).to_string())
