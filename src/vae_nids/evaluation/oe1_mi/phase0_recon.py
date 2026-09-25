"""Fase 0 -- reconocimiento: verifica contra los artefactos del repo (no
contra lo que dice el prompt) la lista de 74 / 51 features, la composición
de los 13 clusters, los hiperparámetros reales del modelo y las rutas de
las figuras de OB1. Salida: metrics/oe1_fase0_reconocimiento.json.

Uso: python -m vae_nids.evaluation.oe1_mi.phase0_recon
"""
import json

import numpy as np
import pandas as pd

from vae_nids import config as base_cfg
from vae_nids.evaluation.oe1_mi import config as cfg
from vae_nids.evaluation.oe1_mi import data

# Tabla del documento OB1 (sección 2 del prompt): cluster -> (reps, orig, desc)
OB1_TABLE = {
    0: (["Packet Length Std", "Packet Length Mean"], 8, 6),
    1: (["Total Fwd Packet"], 6, 5),
    2: (["Flow IAT Max", "Bwd IAT Max", "Idle Min"], 6, 3),
    3: (["Bwd IAT Mean", "Fwd IAT Mean", "Flow IAT Mean", "Flow IAT Std"], 6, 2),
    4: (["Fwd Packet Length Mean"], 2, 1),
    5: (["Flow Duration"], 3, 2),
    6: (["URG Flag Count"], 2, 1),
    7: (["Bwd Bytes/Bulk Avg"], 2, 1),
    8: (["Flow Packets/s"], 2, 1),
    9: (["PSH Flag Count"], 2, 1),
    10: (["Fwd Packet Length Max", "Fwd Packet Length Std"], 2, 0),
    11: (["Bwd IAT Std", "Fwd IAT Std"], 2, 0),
    12: (["Active Mean", "Active Min"], 2, 0),
}


def _checkpoint_info(path) -> dict:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v for k, v in ck.items() if k != "model_state_dict"}


def main() -> dict:
    out: dict = {}
    f51 = cfg.load_features_51()
    drops = cfg.load_documented_drops()
    with open(cfg.EXCLUSION_LOG_PATH, encoding="utf-8") as f:
        excl = json.load(f)

    # --- 74 / 51 ---
    ck74 = _checkpoint_info(base_cfg.PROJECT_ROOT / "outputs/checkpoints/vae_k8_beta1_input74_best.pt")
    ck51 = _checkpoint_info(base_cfg.PROJECT_ROOT / "outputs/checkpoints/vae_k8_beta1_input51_best.pt")
    out["features_74_match_checkpoint_input74"] = ck74["feature_columns"] == cfg.FEATURES_74
    out["features_51_match_checkpoint_input51"] = ck51["feature_columns"] == f51
    out["features_51_match_exclusion_log"] = excl["final_feature_columns"] == f51
    out["n_74"], out["n_51"], out["n_drops_config"] = len(cfg.FEATURES_74), len(f51), len(drops)
    out["74_minus_drops_equals_51"] = [c for c in cfg.FEATURES_74 if c not in set(drops)] == f51
    log_drops = [d["column"] for d in excl["dropped"]["cluster_collapse"]]
    out["config_drops_equal_log_drops"] = sorted(log_drops) == sorted(drops)

    # --- grupos documentados vs. config ---
    grp_drops = sorted(m for rep, ms in cfg.DOCUMENTED_GROUPS.items() for m in ms if m != rep)
    grp_drops += list(cfg.A_PRIORI_DROPS)
    out["documented_groups_drops_equal_config"] = sorted(grp_drops) == sorted(drops)

    # --- tabla OB1 vs. código ---
    ob1_rows = []
    kept = set(f51)
    for cid, members in cfg.EDA_CLUSTERS.items():
        reps_code = sorted(m for m in members if m in kept)
        n_desc_code = sum(m not in kept for m in members)
        reps_ob1, orig_ob1, desc_ob1 = OB1_TABLE[cid]
        ob1_rows.append({
            "cluster": cid, "orig_code": len(members), "desc_code": n_desc_code,
            "kept_code": reps_code, "orig_ob1": orig_ob1, "desc_ob1": desc_ob1,
            "kept_ob1": sorted(reps_ob1),
            "match": (len(members) == orig_ob1 and n_desc_code == desc_ob1
                      and reps_code == sorted(reps_ob1)),
        })
    out["ob1_table_vs_code"] = ob1_rows
    out["ob1_table_all_match"] = all(r["match"] for r in ob1_rows)

    # --- Pearson recalculado vs. r documentado en el log ---
    x, _ = data.load_eda_matrix()
    idx = {f: i for i, f in enumerate(cfg.FEATURES_74)}
    corr = np.corrcoef(x, rowvar=False)
    checks = []
    for d in excl["dropped"]["cluster_collapse"]:
        if d["r_vs_kept"] is None:
            continue
        r = abs(corr[idx[d["column"]], idx[d["kept_as"]]])
        checks.append({"a": d["column"], "b": d["kept_as"], "r_log": d["r_vs_kept"],
                       "r_recalc": float(r), "abs_diff": float(abs(r - d["r_vs_kept"]))})
    out["pearson_recalc_vs_log"] = checks
    out["pearson_recalc_max_abs_diff"] = max(c["abs_diff"] for c in checks)
    out["n_rows_eda"] = int(x.shape[0])
    v, s = idx["Packet Length Variance"], idx["Packet Length Std"]
    out["pearson_variance_vs_std"] = float(corr[v, s])

    # --- hiperparámetros ---
    logs = {}
    for name in ("vae_k8_beta1_input74", "vae_k8_beta1_input51"):
        lg = pd.read_csv(base_cfg.PROJECT_ROOT / f"outputs/logs/{name}_train_log.csv")
        logs[name] = {"epochs_run": int(len(lg)),
                      "best_epoch": int(lg["val_total"].idxmin() + 1)}
    out["checkpoints"] = {
        "input74": {k: ck74[k] for k in ("config", "seed", "epoch", "val_loss")},
        "input51": {k: ck51[k] for k in ("config", "seed", "epoch", "val_loss")},
    }
    out["train_logs"] = logs

    # --- figuras OB1 ---
    figs = [f"outputs/figures/vae_k8_beta1_input{d}_{kind}.png"
            for d in (74, 51) for kind in ("training_curves", "latent_diagnostics")]
    out["ob1_figures"] = {f: (base_cfg.PROJECT_ROOT / f).exists() for f in figs}

    with open(cfg.METRICS_DIR / "oe1_fase0_reconocimiento.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    for k, v in out.items():
        if not isinstance(v, (list, dict)):
            print(f"{k}: {v}")
    return out


if __name__ == "__main__":
    main()
