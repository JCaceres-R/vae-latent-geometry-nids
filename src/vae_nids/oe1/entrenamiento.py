"""Fase 3 del OE1 saneado -- entrenamiento con configuración única y checkpoints autocontenidos.

La línea de comandos solo acepta la ruta de la configuración y la semilla. Todo lo demás
(hiperparámetros, conjunto de columnas y su hash, rutas, hilos) sale del archivo de configuración;
si falta una clave, falla.

Cada checkpoint guarda: la configuración completa, la lista de columnas y su hash, los parámetros del
escalador (float64), el commit de git y si el árbol estaba limpio, el hash de train_benign, las
versiones de Python, torch y numpy, la semilla, los hilos, la época de parada y el motivo.

Uso:
    python -m vae_nids.oe1.entrenamiento --config configs/oe1_saneado.yaml --seed 42
"""
import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from vae_nids.models.vae import VAE, VAEConfig, elbo_loss
from vae_nids.oe1.config import REPO_ROOT, columns_hash, load_config, load_feature_set, resolve
from vae_nids.oe1.datos import load_parts
from vae_nids.oe1.preproceso import MinMax64


def set_determinism(seed: int, threads: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    return {"commit": run("rev-parse", "HEAD"), "arbol_limpio": run("status", "--porcelain") == ""}


def data_hash(df: pd.DataFrame, cols) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(df["row_id"].to_numpy(np.int64)).tobytes())
    h.update(np.ascontiguousarray(df[cols].to_numpy(np.float64)).tobytes())
    return h.hexdigest()


def load_xy(cfg, cols, seed):
    parts = load_parts(cfg, names=("train_benign", "val_benign"), columns=list(cols) + ["row_id"])
    tr, va = parts["train_benign"], parts["val_benign"]
    frac = cfg["entrenamiento"]["fraccion_muestra"]
    if frac < 1.0:
        rng = np.random.default_rng(cfg["semillas"]["general"])
        tr = tr.iloc[np.sort(rng.choice(len(tr), int(len(tr) * frac), replace=False))].reset_index(drop=True)
        va = va.iloc[np.sort(rng.choice(len(va), int(len(va) * frac), replace=False))].reset_index(drop=True)
    scaler = MinMax64().fit(tr, cols)
    return tr, va, scaler


def run_epoch(model, x, beta, batch_size, optimizer=None):
    train = optimizer is not None
    model.train(train)
    n = x.shape[0]
    idx_all = torch.randperm(n) if train else torch.arange(n)
    tot = rec = kl = 0.0
    with torch.set_grad_enabled(train):
        for s in range(0, n, batch_size):
            xb = x[idx_all[s:s + batch_size]]
            out = model(xb)
            loss, r, k = elbo_loss(xb, out, beta)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            b = xb.shape[0]
            tot += loss.item() * b
            rec += r.item() * b
            kl += k.item() * b
    return tot / n, rec / n, kl / n


def train(cfg, seed: int):
    cols = load_feature_set(cfg)
    te, me, cp = cfg["entrenamiento"], cfg["modelo"], cfg["cpu"]
    set_determinism(seed, cp["hilos"], cp["deterministico"])
    tr, va, scaler = load_xy(cfg, cols, seed)
    x_tr = torch.from_numpy(scaler.transform(tr))
    x_va = torch.from_numpy(scaler.transform(va))
    run = f"{te['nombre_corrida']}_seed{seed}"
    out = resolve(cfg, "salidas")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "checkpoints" / f"{run}.pt"
    log_path = out / "logs" / f"{run}_train_log.csv"
    torch.manual_seed(seed)                       # el stream de RNG del modelo empieza aquí
    mcfg = VAEConfig(input_dim=len(cols), latent_dim=me["latent_dim"], beta=me["beta"], hidden_dim=me["hidden_dim"],
                     logvar_min=me["logvar_min"], logvar_max=me["logvar_max"])
    model = VAE(mcfg)
    if te["optimizador"] != "adam":
        raise ValueError(f"optimizador no soportado: {te['optimizador']}")
    opt = torch.optim.Adam(model.parameters(), lr=te["lr"])
    meta = {
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")}, "config_path": cfg["_path"],
        "columnas": cols, "columnas_sha256": columns_hash(cols), "escalador": scaler.to_dict(),
        "git": git_state(), "train_benign_sha256": data_hash(tr, cols), "n_train": len(tr), "n_val": len(va),
        "versiones": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "semilla": seed, "hilos": torch.get_num_threads(), "deterministico": cp["deterministico"],
        "modelo_config": asdict(mcfg),
    }
    best, best_state, best_epoch, wait, reason = float("inf"), None, 0, 0, "max_epochs"
    rows = []
    t_start = time.perf_counter()
    for epoch in range(1, te["max_epochs"] + 1):
        t0 = time.perf_counter()
        trl = run_epoch(model, x_tr, me["beta"], te["batch_size"], opt)
        val = run_epoch(model, x_va, me["beta"], te["batch_size"])
        rows.append(dict(epoch=epoch, train_total=trl[0], train_recon=trl[1], train_kl=trl[2], val_total=val[0],
                         val_recon=val[1], val_kl=val[2], seconds=time.perf_counter() - t0))
        if val[0] < best - te["min_delta"]:
            best, best_epoch, wait = val[0], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= te["patience"]:
                reason = f"paciencia ({te['patience']} épocas sin mejorar más de min_delta={te['min_delta']})"
                break
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta.update(epoca_parada=epoch, motivo_parada=reason, mejor_epoca=best_epoch, mejor_val_total=best,
                segundos_totales=time.perf_counter() - t_start)
    torch.save({"model_state_dict": best_state, **meta}, ckpt_path)
    return ckpt_path, meta


def load_checkpoint(path):
    """Reconstruye modelo y escalador solo con lo que guarda el checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("config", "columnas", "columnas_sha256", "escalador", "modelo_config", "model_state_dict"):
        if k not in ck:
            raise KeyError(f"El checkpoint {path} no es autocontenido: falta '{k}'")
    if columns_hash(ck["columnas"]) != ck["columnas_sha256"]:
        raise ValueError("hash de columnas inconsistente en el checkpoint")
    model = VAE(VAEConfig(**ck["modelo_config"]))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, MinMax64.from_dict(ck["escalador"]), ck


def main():
    ap = argparse.ArgumentParser(description="Entrena un VAE del OE1 saneado (solo --config y --seed).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    path, meta = train(load_config(a.config), a.seed)
    print(json.dumps({k: meta[k] for k in ("epoca_parada", "motivo_parada", "mejor_epoca", "mejor_val_total",
                                           "segundos_totales")}, indent=2, ensure_ascii=False), "\n", path)


if __name__ == "__main__":
    main()
