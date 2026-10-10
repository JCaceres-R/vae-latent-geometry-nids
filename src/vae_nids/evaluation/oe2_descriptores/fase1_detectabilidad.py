"""Fase 1 -- desenlace: AUC-ROC por familia y semilla con el puntaje principal.

Reutiliza `phase4_detectability.compute_scores` (mismo muestreo MC: semilla
MC_SEED_BASE + seed por grupo), de modo que la semilla 42 reproduce el AUC de
outputs/oe2_robust/tables/phase4_detectability_metrics.csv. Además:
- codifica val_benign (mu, logvar) y lo deja en caché como referencia benigna;
- guarda por flujo nll_mc (distorsión D) y la KL al prior (tasa R).

Uso:
    python -m vae_nids.evaluation.oe2_descriptores.fase1_detectabilidad [--seed 42]
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from vae_nids.evaluation.oe2_descriptores import config as C
from vae_nids.evaluation.oe2_robust import config as rcfg
from vae_nids.evaluation.oe2_robust.phase2_encode import feature_columns_for
from vae_nids.evaluation.oe2_robust.phase3_geometry import load_group_mu_logvar
from vae_nids.evaluation.oe2_robust.phase4_detectability import XCache, compute_scores, load_model


def active_dims_for(seed: int) -> list[int]:
    t = pd.read_csv(rcfg.TABLES_DIR / "phase2_active_dims_per_seed.csv").set_index("seed")
    return [j for j in range(rcfg.LATENT_DIM) if bool(t.loc[seed, f"active_dim{j}"])]


def scores_for_seed(seed: int, x_cache: XCache) -> pd.DataFrame:
    out = C.CACHE_DIR / f"scores_seed{seed}.npz"
    if out.exists():
        print(f"[fase1 seed={seed}] ya calculado")
        return pd.read_csv(C.CACHE_DIR / f"auc_seed{seed}.csv")
    model = load_model(seed)
    act = active_dims_for(seed)
    mu_b, lv_b = load_group_mu_logvar(seed, "latent_benign_test")
    mean_b = mu_b[:, act].mean(0)
    inv_cov_b = np.linalg.inv(np.cov(mu_b[:, act], rowvar=False))
    mc = C.MC_SEED_BASE + seed

    def score(x, mu, lv):
        s = compute_scores(model, x, mu, lv, act, mean_b, inv_cov_b, C.L_MC, mc, C.BETA)
        kl = 0.5 * (np.exp(lv) + mu ** 2 - 1.0 - lv).sum(1)
        return s["nll_mc"].astype(np.float32), kl.astype(np.float32)

    # referencia benigna: val_benign codificado (no entra en el AUC)
    x_val = x_cache.val_benign()
    with torch.no_grad():
        mu_v, lv_v = model.encoder(x_val)
    mu_v, lv_v = mu_v.numpy(), lv_v.numpy()
    np.savez_compressed(C.CACHE_DIR / f"val_benign_seed{seed}.npz", mu=mu_v, logvar=lv_v)

    # comprobación de alineación x <-> mu guardado (primeras 2000 filas)
    x_b = x_cache.benign_test()
    with torch.no_grad():
        mu_chk, _ = model.encoder(x_b[:2000])
    assert np.allclose(mu_chk.numpy(), mu_b[:2000], atol=1e-4), "desalineación x/mu en benign_test"

    arrays, rows = {}, []
    s_neg, r_neg = score(x_b, mu_b, lv_b)
    arrays["BENIGN_nll"], arrays["BENIGN_kl"] = s_neg, r_neg
    for g in C.ALL_GROUPS:
        mu_g, lv_g = load_group_mu_logvar(seed, g)
        x_g = x_cache.attack_group(C.GROUP_LABELS[g])
        assert x_g.shape[0] == mu_g.shape[0], f"{g}: desalineación"
        s_pos, r_pos = score(x_g, mu_g, lv_g)
        arrays[f"{g}_nll"], arrays[f"{g}_kl"] = s_pos, r_pos
        y = np.r_[np.ones(len(s_pos)), np.zeros(len(s_neg))]
        rows.append(dict(
            seed=seed, group=g, n=len(s_pos),
            auc_nll_mc=float(roc_auc_score(y, np.r_[s_pos, s_neg])),
            auc_rate=float(roc_auc_score(y, np.r_[r_pos, r_neg])),
            median_D=float(np.median(s_pos)), median_R=float(np.median(r_pos)),
        ))
        print(f"[fase1 seed={seed}] {g:32s} n={len(s_pos):7d} AUC={rows[-1]['auc_nll_mc']:.4f}")
    np.savez_compressed(out, **arrays)
    df = pd.DataFrame(rows)
    df["benign_median_D"] = float(np.median(s_neg))
    df["benign_median_R"] = float(np.median(r_neg))
    df.to_csv(C.CACHE_DIR / f"auc_seed{seed}.csv", index=False)
    return df


def main(seeds=None):
    torch.set_num_threads(8)
    x_cache = XCache(feature_columns_for(C.SEEDS[0]))
    dfs = [scores_for_seed(s, x_cache) for s in (seeds or C.SEEDS)]
    df = pd.concat(dfs, ignore_index=True)
    if seeds is None:
        df.to_csv(C.TABLES_DIR / "detectabilidad_por_semilla.csv", index=False)
        # verificación contra el pipeline robusto (seed 42); si su fase 4 no está, se registra como NaN
        # y se sigue: esta fase calcula su propio AUC y no depende de la tabla de la fase 4.
        ref_path = rcfg.TABLES_DIR / "phase4_detectability_metrics.csv"
        diff = np.nan
        if ref_path.exists():
            ref = pd.read_csv(ref_path)
            ref = ref[(ref.score == "nll_mc") & (ref.seed == 42)].set_index("group")["auc_roc"]
            mine = df[df.seed == 42].set_index("group")["auc_nll_mc"]
            if len(ref):
                diff = (mine - ref.reindex(mine.index)).abs().max()
        json.dump({"max_abs_diff_auc_seed42_vs_phase4": float(diff)},
                  open(C.TABLES_DIR / "verificacion_auc_seed42.json", "w"), indent=2)
        print(f"[fase1] diferencia máxima de AUC vs phase4 (seed 42): {diff:.2e}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    main([a.seed] if a.seed is not None else None)
