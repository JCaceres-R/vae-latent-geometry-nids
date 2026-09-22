"""Fase 1 -- entrenamiento multi-semilla.

Entrena las semillas nuevas de config.NEW_SEEDS (42 no se reentrena: usa el
checkpoint oficial). Mismos splits, mismo scaler, misma arquitectura, mismo
criterio de early stopping que OE1/OE2 -- solo cambia la semilla del
modelo (inicialización, orden de minibatches, epsilon del reparam trick).

Uso:
    python -m vae_nids.evaluation.oe2_robust.phase1_train_multiseed
"""
import csv
import json

import torch

from vae_nids.evaluation.oe2_robust import config as rcfg
from vae_nids.evaluation.oe2_robust.train_seed import train_one_seed

# Umbral para señalar una semilla "anómala" en el resumen (no se excluye
# automáticamente -- se decide manualmente en el informe, con criterio
# definido ANTES de ver los resultados de geometría/detectabilidad de OE2):
# val_loss final a más de 3 desviaciones estándar del resto del barrido, o
# colapso total del posterior (0 dimensiones activas).
ANOMALY_ZSCORE_THRESHOLD = 3.0


def run_name_for(seed: int) -> str:
    return f"vae_k8_beta1_input51_seed{seed}"


def main() -> list[dict]:
    results = []

    # Semilla oficial: no se reentrena, se registra tal cual para el resumen.
    official_ckpt = torch.load(rcfg.OFFICIAL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    results.append({
        "seed": rcfg.OFFICIAL_SEED,
        "run_name": "vae_k8_beta1_input51_best (oficial, no reentrenado)",
        "checkpoint_epoch": official_ckpt["epoch"],
        "val_loss": official_ckpt["val_loss"],
        "val_recon": official_ckpt["val_recon"],
        "val_kl": official_ckpt["val_kl"],
        "total_train_seconds": None,
        "reused_official": True,
    })
    print(f"[fase1] seed={rcfg.OFFICIAL_SEED}: checkpoint oficial reutilizado "
          f"(epoch={official_ckpt['epoch']}, val_loss={official_ckpt['val_loss']:.4f})")

    for seed in rcfg.NEW_SEEDS:
        run_name = run_name_for(seed)
        meta = train_one_seed(seed=seed, run_name=run_name)
        results.append({
            "seed": seed,
            "run_name": run_name,
            "checkpoint_epoch": meta["checkpoint_epoch"],
            "val_loss": meta["val_loss"],
            "val_recon": meta["val_recon"],
            "val_kl": meta["val_kl"],
            "total_train_seconds": meta["total_train_seconds"],
            "reused_official": False,
        })

    # --- Resumen + detección de anomalías (val_loss z-score) ---
    val_losses = [r["val_loss"] for r in results]
    mean_vl = sum(val_losses) / len(val_losses)
    var_vl = sum((v - mean_vl) ** 2 for v in val_losses) / max(len(val_losses) - 1, 1)
    sd_vl = var_vl ** 0.5
    for r in results:
        z = (r["val_loss"] - mean_vl) / sd_vl if sd_vl > 0 else 0.0
        r["val_loss_zscore"] = z
        r["flagged_anomalous"] = abs(z) > ANOMALY_ZSCORE_THRESHOLD

    n_flagged = sum(r["flagged_anomalous"] for r in results)
    print(f"[fase1] val_loss: media={mean_vl:.4f} sd={sd_vl:.4f}; "
          f"{n_flagged} semilla(s) marcadas como anómalas (|z|>{ANOMALY_ZSCORE_THRESHOLD})")
    for r in results:
        flag = " <-- ANOMALA" if r["flagged_anomalous"] else ""
        print(f"  seed={r['seed']:>3} epoch={r['checkpoint_epoch']:>3} "
              f"val_loss={r['val_loss']:.4f} z={r['val_loss_zscore']:+.2f}{flag}")

    out_csv = rcfg.TABLES_DIR / "phase1_seed_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[fase1] resumen guardado en {out_csv}")

    out_json = rcfg.METRICS_DIR / "phase1_seed_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    main()
