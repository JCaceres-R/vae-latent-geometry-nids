"""Fase 0 -- diagnóstico y presupuesto de cómputo.

1. Confirma splits, script de entrenamiento, config oficial y checkpoint
   oficial (seed=42, epoch=52, val_loss=-185.2791).
2. Reentrena UNA vez seed=42 con la misma config (escribiendo a
   outputs/oe2_robust/, nunca sobre el checkpoint oficial) y compara el
   resultado contra el oficial: misma epoch de parada y val_loss dentro de
   tolerancia -> reproduce; distinta epoch/val_loss -> no determinismo de
   por medio (batching en CPU, orden de reducción en operaciones, etc.),
   se documenta cuál.
3. Mide el tiempo de esa corrida y estima el costo total del plan (10
   semillas).
4. Reporta hardware disponible.

Uso:
    python -m vae_nids.evaluation.oe2_robust.phase0_diagnostics
"""
import json
import platform

import psutil
import torch

from vae_nids import config as base_cfg
from vae_nids.evaluation.oe2_robust import config as rcfg
from vae_nids.evaluation.oe2_robust.train_seed import train_one_seed

REPRO_TOLERANCE_VAL_LOSS = 1.0  # unidades de ELBO -- CPU no garantiza bit-exactitud
REPRO_RUN_NAME = "vae_k8_beta1_input51_seed42_repro_check"


def check_official_checkpoint() -> dict:
    assert rcfg.OFFICIAL_CHECKPOINT_PATH.exists(), (
        f"Checkpoint oficial no encontrado en {rcfg.OFFICIAL_CHECKPOINT_PATH}"
    )
    ckpt = torch.load(rcfg.OFFICIAL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    info = {
        "path": str(rcfg.OFFICIAL_CHECKPOINT_PATH),
        "seed": ckpt["seed"],
        "epoch": ckpt["epoch"],
        "val_loss": ckpt["val_loss"],
        "config": ckpt["config"],
    }
    print(f"[fase0] checkpoint oficial: seed={info['seed']} epoch={info['epoch']} "
          f"val_loss={info['val_loss']:.4f} config={info['config']}")
    return info


def check_splits() -> dict:
    import pandas as pd
    sizes = {}
    for name in ["train_benign", "val_benign", "test_benign", "test_attacks"]:
        df = pd.read_parquet(base_cfg.OUTPUT_DIR / f"{name}.parquet")
        sizes[name] = df.shape
        print(f"[fase0] split {name}: {df.shape}")
    return sizes


def hardware_report() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(f"[fase0] hardware: {info}")
    return info


def reproducibility_check() -> dict:
    official = check_official_checkpoint()
    result = train_one_seed(seed=rcfg.OFFICIAL_SEED, run_name=REPRO_RUN_NAME)

    epoch_matches = result["checkpoint_epoch"] == official["epoch"]
    val_loss_diff = abs(result["val_loss"] - official["val_loss"])
    val_loss_close = val_loss_diff <= REPRO_TOLERANCE_VAL_LOSS

    if epoch_matches and val_loss_diff < 1e-6:
        verdict = "exacta"
    elif val_loss_close:
        verdict = "aproximada"
    else:
        verdict = "no reproduce"

    report = {
        "official_epoch": official["epoch"],
        "official_val_loss": official["val_loss"],
        "repro_epoch": result["checkpoint_epoch"],
        "repro_val_loss": result["val_loss"],
        "epoch_matches": epoch_matches,
        "val_loss_abs_diff": val_loss_diff,
        "tolerance_used": REPRO_TOLERANCE_VAL_LOSS,
        "verdict": verdict,
        "seconds_for_this_run": result["total_train_seconds"],
        "note": (
            "PyTorch en CPU no garantiza determinismo bit-exacto entre "
            "corridas incluso con la misma semilla (orden de reducción en "
            "operaciones paralelizadas, threads de BLAS); se evalúa "
            "aproximación, no igualdad exacta."
        ),
    }
    print(f"[fase0] reproducibilidad: {verdict} "
          f"(epoch oficial={official['epoch']} vs. repro={result['checkpoint_epoch']}, "
          f"|delta val_loss|={val_loss_diff:.4f}, tolerancia={REPRO_TOLERANCE_VAL_LOSS})")
    return report


def main() -> dict:
    hw = hardware_report()
    splits = check_splits()
    repro = reproducibility_check()

    seconds_per_seed = repro["seconds_for_this_run"]
    n_new_seeds = len(rcfg.NEW_SEEDS)
    estimated_total_minutes = seconds_per_seed * n_new_seeds / 60.0

    budget = {
        "seconds_per_seed_observed": seconds_per_seed,
        "n_new_seeds_to_train": n_new_seeds,
        "estimated_total_training_minutes": estimated_total_minutes,
        "decision": f"Entrenar las {len(rcfg.ALL_SEEDS)} semillas completas "
                    f"({rcfg.ALL_SEEDS}); presupuesto estimado de "
                    f"~{estimated_total_minutes:.0f} min en CPU para las "
                    f"{n_new_seeds} nuevas (42 reutiliza el checkpoint oficial).",
    }
    print(f"[fase0] presupuesto: ~{estimated_total_minutes:.1f} min para "
          f"{n_new_seeds} semillas nuevas ({seconds_per_seed:.1f}s/semilla observado)")

    report = {"hardware": hw, "splits": splits, "reproducibility": repro, "budget": budget}
    out_path = rcfg.METRICS_DIR / "phase0_diagnostics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[fase0] reporte guardado en {out_path}")
    return report


if __name__ == "__main__":
    main()
