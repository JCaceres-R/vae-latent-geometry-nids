"""Fase 4 -- detectabilidad confirmatoria, por semilla.

Responde a la crítica 2 del director: no un solo score y una sola métrica,
sino 5 scores de anomalía (incluida la probabilidad de reconstrucción MC de
An & Cho 2015 con L=20 muestras del posterior, no una sola muestra ni solo
mu), AUC-ROC/AUC-PR/pAUC/TPR@FPR fijo con IC bootstrap, y pruebas formales
de DeLong (entre familias y entre scores) con corrección de Holm.

Uso:
    python -m vae_nids.evaluation.oe2_robust.phase4_detectability
    python -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed 42
"""
import argparse
import csv
import itertools
import json

import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from vae_nids import config as base_cfg
from vae_nids.evaluation.oe2_robust import config as rcfg
from vae_nids.evaluation.oe2_robust import stats_utils as su
from vae_nids.evaluation.oe2_robust.phase2_encode import checkpoint_path_for
from vae_nids.evaluation.oe2_robust.phase3_geometry import group_names, load_group_mu_logvar, mahalanobis_batch
from vae_nids.models.vae import VAE, VAEConfig

RNG_SEED = 42


# ============================================================
# carga de x, cacheada en memoria una sola vez (no depende de la semilla)
# ============================================================

class XCache:
    def __init__(self, feature_cols: list[str]):
        self.feature_cols = feature_cols
        self._benign_test = pd.read_parquet(base_cfg.OUTPUT_DIR / "test_benign.parquet")
        self._val_benign = pd.read_parquet(base_cfg.OUTPUT_DIR / "val_benign.parquet")
        self._attacks = pd.read_parquet(base_cfg.OUTPUT_DIR / "test_attacks.parquet")

    def x_tensor(self, df: pd.DataFrame) -> torch.Tensor:
        return torch.from_numpy(df[self.feature_cols].to_numpy(dtype=np.float32, copy=True))

    def benign_test(self) -> torch.Tensor:
        return self.x_tensor(self._benign_test)

    def val_benign(self) -> torch.Tensor:
        return self.x_tensor(self._val_benign)

    def attack_group(self, labels: list[str]) -> torch.Tensor:
        subset = self._attacks[self._attacks[base_cfg.LABEL_COL].isin(labels)]
        return self.x_tensor(subset)


def load_model(seed: int) -> VAE:
    ckpt = torch.load(checkpoint_path_for(seed), map_location="cpu", weights_only=False)
    model = VAE(VAEConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ============================================================
# 6.1 -- scores de anomalía
# ============================================================

@torch.no_grad()
def _recon_nll_per_sample(model: VAE, x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu_x_hat, logvar_x_hat = model.decoder(z)
    var_hat = torch.exp(logvar_x_hat)
    log_2pi = torch.log(torch.tensor(2.0 * torch.pi))
    nll_per_dim = 0.5 * (log_2pi + logvar_x_hat + (x - mu_x_hat) ** 2 / var_hat)
    return nll_per_dim.sum(dim=1), mu_x_hat


@torch.no_grad()
def compute_scores(model: VAE, x: torch.Tensor, mu: np.ndarray, logvar: np.ndarray,
                    active_dims: list[int], mean_b: np.ndarray, inv_cov_b: np.ndarray,
                    l_mc: int, mc_seed: int, beta: float) -> dict[str, np.ndarray]:
    torch.manual_seed(mc_seed)
    mu_t = torch.from_numpy(mu.astype(np.float32))
    logvar_t = torch.from_numpy(logvar.astype(np.float32))

    nll_mu, mu_x_hat = _recon_nll_per_sample(model, x, mu_t)
    mse = ((x - mu_x_hat) ** 2).sum(dim=1)

    std = torch.exp(0.5 * logvar_t)
    n = x.shape[0]
    nll_acc = torch.zeros(n)
    for _ in range(l_mc):
        eps = torch.randn_like(std)
        z = mu_t + std * eps
        nll_l, _ = _recon_nll_per_sample(model, x, z)
        nll_acc += nll_l
    nll_mc = nll_acc / l_mc

    kl = 0.5 * (logvar_t.exp() + mu_t.pow(2) - 1.0 - logvar_t).sum(dim=1)
    neg_elbo = nll_mc + beta * kl

    latent_maha = mahalanobis_batch(mu[:, active_dims], mean_b, inv_cov_b)

    return {
        "nll_mu": nll_mu.numpy(),
        "nll_mc": nll_mc.numpy(),
        "neg_elbo": neg_elbo.numpy(),
        "mse": mse.numpy(),
        "latent_maha": latent_maha,
    }


SCORE_NAMES = ["nll_mu", "nll_mc", "neg_elbo", "mse", "latent_maha"]
PRINCIPAL_SCORE = "nll_mc"


# ============================================================
# 6.2 -- métricas por familia y score
# ============================================================

def auc_roc_ci_delong(pos: np.ndarray, neg: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """IC95 de AUC-ROC vía la varianza analítica de DeLong (O(n log n), una
    sola pasada) en vez de 1000 remuestras de bootstrap -- con 5 scores x 18
    familias x 10 semillas, el bootstrap ingenuo sobre grupos de hasta
    ~400k observaciones (PortScan+benigno) es computacionalmente inviable
    (~horas). DeLong da la misma cantidad (varianza asintótica del
    U-estadístico de Mann-Whitney subyacente a la AUC) sin remuestrear."""
    _, var = su.delong_auc_variance(pos, neg)
    se = float(np.sqrt(max(var, 0.0)))
    z = float(sp_stats.norm.ppf(1 - alpha / 2))
    auc = float(roc_auc_score(np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]),
                               np.concatenate([pos, neg])))
    return auc - z * se, auc + z * se


AUC_PR_BOOTSTRAP_MAX_N_PER_SIDE = rcfg.SILHOUETTE_MAX_N  # mismo tope de tratabilidad usado en la Fase 3


def auc_pr_ci_bootstrap(pos: np.ndarray, neg: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """IC95 de AUC-PR por bootstrap -- sin fórmula analítica cerrada
    disponible (a diferencia de AUC-ROC/DeLong), así que se submuestrea a
    lo sumo AUC_PR_BOOTSTRAP_MAX_N_PER_SIDE por lado antes de remuestrear,
    por la misma razón de tratabilidad computacional documentada en
    `auc_roc_ci_delong`."""
    rng = np.random.default_rng(seed)
    pos_s = pos if len(pos) <= AUC_PR_BOOTSTRAP_MAX_N_PER_SIDE else pos[
        rng.choice(len(pos), AUC_PR_BOOTSTRAP_MAX_N_PER_SIDE, replace=False)]
    neg_s = neg if len(neg) <= AUC_PR_BOOTSTRAP_MAX_N_PER_SIDE else neg[
        rng.choice(len(neg), AUC_PR_BOOTSTRAP_MAX_N_PER_SIDE, replace=False)]
    n_pos, n_neg = len(pos_s), len(neg_s)
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    aps = np.empty(n_boot)
    for b in range(n_boot):
        sp = pos_s[rng.integers(0, n_pos, n_pos)]
        sn = neg_s[rng.integers(0, n_neg, n_neg)]
        aps[b] = average_precision_score(y_true, np.concatenate([sp, sn]))
    lo, hi = np.percentile(aps, [2.5, 97.5])
    return float(lo), float(hi)


def tpr_at_fpr(pos: np.ndarray, val_benign_scores: np.ndarray, test_benign_scores: np.ndarray,
               fpr_target: float) -> dict:
    threshold = float(np.percentile(val_benign_scores, 100 * (1 - fpr_target)))
    tpr = float(np.mean(pos > threshold))
    fpr_realized_test = float(np.mean(test_benign_scores > threshold))
    return {"threshold": threshold, "tpr": tpr, "fpr_realized_on_test_benign": fpr_realized_test}


def detectability_for_seed(seed: int, active_dims: list[int], manifest: dict, x_cache: XCache) -> dict:
    model = load_model(seed)

    mu_benign, logvar_benign = load_group_mu_logvar(seed, "latent_benign_test")
    mean_b = mu_benign[:, active_dims].mean(axis=0)
    cov_b = np.cov(mu_benign[:, active_dims], rowvar=False)
    inv_cov_b = np.linalg.inv(cov_b)

    # val_benign no se codificó en Fase 2 (solo hace falta aquí, para
    # calibrar tau y los umbrales de FPR fijo) -- se codifica una sola vez.
    x_val_benign = x_cache.val_benign()
    with torch.no_grad():
        mu_val_t, logvar_val_t = model.encoder(x_val_benign)
    scores_val = compute_scores(model, x_val_benign, mu_val_t.numpy(), logvar_val_t.numpy(),
                                 active_dims, mean_b, inv_cov_b,
                                 rcfg.L_MC_SAMPLES, rcfg.MC_SEED_BASE + seed, rcfg.BETA)

    x_test_benign = x_cache.benign_test()
    scores_test_benign = compute_scores(model, x_test_benign, mu_benign, logvar_benign,
                                         active_dims, mean_b, inv_cov_b,
                                         rcfg.L_MC_SAMPLES, rcfg.MC_SEED_BASE + seed, rcfg.BETA)

    tau_p95 = float(np.percentile(scores_val[PRINCIPAL_SCORE], rcfg.TAU_PERCENTILE))

    metric_rows, roc_curves, raw_scores = [], {}, {}
    for name in group_names(manifest):
        labels = manifest[name]["labels"]
        mu_g, logvar_g = load_group_mu_logvar(seed, name)
        x_g = x_cache.attack_group(labels)
        assert x_g.shape[0] == mu_g.shape[0] == manifest[name]["n"], (
            f"seed={seed} {name}: desalineación x/mu/manifest"
        )
        scores_g = compute_scores(model, x_g, mu_g, logvar_g, active_dims, mean_b, inv_cov_b,
                                   rcfg.L_MC_SAMPLES, rcfg.MC_SEED_BASE + seed, rcfg.BETA)
        raw_scores[name] = scores_g

        for score_name in SCORE_NAMES:
            pos, neg = scores_g[score_name], scores_test_benign[score_name]
            y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            y_score = np.concatenate([pos, neg])
            auc = float(roc_auc_score(y_true, y_score))
            ap = float(average_precision_score(y_true, y_score))
            pauc = float(roc_auc_score(y_true, y_score, max_fpr=rcfg.PAUC_FPR_MAX))
            auc_ci_lo, auc_ci_hi = auc_roc_ci_delong(pos, neg)
            ap_ci_lo, ap_ci_hi = auc_pr_ci_bootstrap(pos, neg, rcfg.N_BOOTSTRAP_AUC, rcfg.N_BOOTSTRAP_SEED)

            row = {
                "seed": seed, "group": name, "score": score_name, "n_pos": len(pos), "n_neg": len(neg),
                "auc_roc": auc, "auc_roc_ci_low": auc_ci_lo, "auc_roc_ci_high": auc_ci_hi,
                "auc_roc_ci_method": "delong_analytic",
                "auc_pr": ap, "auc_pr_ci_low": ap_ci_lo, "auc_pr_ci_high": ap_ci_hi,
                "auc_pr_ci_method": f"bootstrap_subsampled_max{AUC_PR_BOOTSTRAP_MAX_N_PER_SIDE}_per_side",
                "prevalence": len(pos) / (len(pos) + len(neg)),
                "pauc_mcclish_fpr0.10": pauc,
                "tpr_at_tau_p95": float(np.mean(pos > tau_p95)) if score_name == PRINCIPAL_SCORE else None,
            }
            for fpr_target in rcfg.FPR_TARGETS:
                r = tpr_at_fpr(pos, scores_val[score_name], scores_test_benign[score_name], fpr_target)
                row[f"tpr_at_fpr{int(fpr_target*100)}"] = r["tpr"]
                row[f"fpr_realized_on_test_benign_at_fpr{int(fpr_target*100)}"] = r["fpr_realized_on_test_benign"]
            metric_rows.append(row)

            if score_name == PRINCIPAL_SCORE:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_curves[name] = {"fpr": fpr, "tpr": tpr}

    return {
        "metric_rows": metric_rows,
        "roc_curves": roc_curves,
        "raw_scores": raw_scores,
        "scores_test_benign": scores_test_benign,
        "tau_p95": tau_p95,
        "fpr_test_benign_at_tau_p95": float(np.mean(scores_test_benign[PRINCIPAL_SCORE] > tau_p95)),
    }


# ============================================================
# 6.3 -- pruebas formales (DeLong) + control de tamaño de muestra
# ============================================================

def delong_between_families(raw_scores: dict, scores_test_benign: dict, seed: int) -> list[dict]:
    rows = []
    families = [g for g in rcfg.MAIN_FAMILIES if g in raw_scores]
    for score_name in SCORE_NAMES:
        pvals, pairs = [], []
        for a, b in itertools.combinations(families, 2):
            res = su.delong_test_shared_negatives(
                raw_scores[a][score_name], raw_scores[b][score_name], scores_test_benign[score_name]
            )
            pairs.append((a, b, res))
            pvals.append(res["p_value"])
        p_adj = su.holm_correction(np.array(pvals)) if pvals else []
        for (a, b, res), pa in zip(pairs, p_adj):
            rows.append({
                "seed": seed, "score": score_name, "family_a": a, "family_b": b,
                "auc_a": res["auc_a"], "auc_b": res["auc_b"], "diff": res["diff"],
                "z": res["z"], "p_value": res["p_value"], "p_holm": float(pa),
            })
    return rows


def delong_between_scores(raw_scores: dict, scores_test_benign: dict, seed: int) -> list[dict]:
    rows = []
    comparisons = [("nll_mc", "latent_maha"), ("nll_mc", "nll_mu"), ("nll_mc", "mse")]
    families = [g for g in raw_scores]
    for name in families:
        pos_by_score = raw_scores[name]
        neg_by_score = scores_test_benign
        for score_a, score_b in comparisons:
            n_pos, n_neg = len(pos_by_score[score_a]), len(neg_by_score[score_a])
            y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
            score_a_full = np.concatenate([pos_by_score[score_a], neg_by_score[score_a]])
            score_b_full = np.concatenate([pos_by_score[score_b], neg_by_score[score_b]])
            res = su.delong_test_paired(y_true, score_a_full, score_b_full)
            rows.append({
                "seed": seed, "group": name, "score_a": score_a, "score_b": score_b,
                "auc_a": res["auc_a"], "auc_b": res["auc_b"], "diff": res["diff"],
                "z": res["z"], "p_value": res["p_value"],
            })
    return rows


def sample_size_control(raw_scores: dict, scores_test_benign: dict, seed: int) -> list[dict]:
    rng = np.random.default_rng(RNG_SEED)
    neg_full = scores_test_benign[PRINCIPAL_SCORE]
    rows = []
    for name in [g for g in rcfg.MAIN_FAMILIES if g in raw_scores]:
        pos = raw_scores[name][PRINCIPAL_SCORE]
        n_target = len(pos)
        auc_full = float(roc_auc_score(
            np.concatenate([np.ones(len(pos)), np.zeros(len(neg_full))]),
            np.concatenate([pos, neg_full]),
        ))
        aucs_sub = np.empty(rcfg.SAMPLE_SIZE_CONTROL_REPEATS)
        for r in range(rcfg.SAMPLE_SIZE_CONTROL_REPEATS):
            neg_sub = neg_full[rng.choice(len(neg_full), size=min(n_target, len(neg_full)), replace=False)]
            aucs_sub[r] = roc_auc_score(
                np.concatenate([np.ones(len(pos)), np.zeros(len(neg_sub))]),
                np.concatenate([pos, neg_sub]),
            )
        rows.append({
            "seed": seed, "group": name, "n_target": n_target,
            "auc_full_benign": auc_full, "auc_matched_n_mean": float(aucs_sub.mean()),
            "auc_matched_n_ci_low": float(np.percentile(aucs_sub, 2.5)),
            "auc_matched_n_ci_high": float(np.percentile(aucs_sub, 97.5)),
        })
    return rows


# ============================================================
# orquestación
# ============================================================

def write_csv(rows: list[dict], path):
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[fase4] {path} ({len(rows)} filas)")


CACHE_DIR = rcfg.TABLES_DIR / "_cache"


def main(seeds: list[int] | None = None):
    """Igual estrategia de checkpointing por semilla que `phase3_geometry.main`
    (ver su docstring): cada semilla se cachea apenas termina, para poder
    relanzar tras una interrupción sin recomputar semillas ya hechas. Los
    scores crudos (.npz) y curvas ROC ya se guardaban por semilla desde
    antes; lo nuevo es cachear también las filas de métricas agregadas."""
    seeds = seeds if seeds is not None else rcfg.ALL_SEEDS
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(rcfg.OFFICIAL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    feature_cols = ckpt["feature_columns"]
    x_cache = XCache(feature_cols)

    with open(rcfg.METRICS_DIR / "phase2_active_dims_per_seed.json", encoding="utf-8") as f:
        active_dims_map = json.load(f)

    for seed in seeds:
        cache_path = CACHE_DIR / f"phase4_seed{seed}.json"
        if cache_path.exists():
            print(f"[fase4 seed={seed}] ya cacheado en {cache_path.name}, se salta el cómputo")
            continue

        manifest_path = rcfg.LATENT_DIR / f"seed{seed}" / "manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        active_dims = active_dims_map[str(seed)]

        result = detectability_for_seed(seed, active_dims, manifest, x_cache)
        tau_row = {"seed": seed, "tau_p95": result["tau_p95"],
                   "fpr_test_benign_at_tau_p95": result["fpr_test_benign_at_tau_p95"]}
        family_delong = delong_between_families(result["raw_scores"], result["scores_test_benign"], seed)
        score_delong = delong_between_scores(result["raw_scores"], result["scores_test_benign"], seed)
        sample_control = sample_size_control(result["raw_scores"], result["scores_test_benign"], seed)

        for name, curves in result["roc_curves"].items():
            np.savez(rcfg.TABLES_DIR / f"phase4_roc_seed{seed}_{name}.npz", **curves)

        # scores crudos por conexión, persistidos para la Fase 5 (modelo de
        # efectos mixtos a nivel de conexión, 7.3) -- evita recodificar/
        # rescorear con L=20 muestras MC otra vez en esa fase.
        raw_dir = rcfg.LATENT_DIR / f"seed{seed}" / "scores"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name, scores in result["raw_scores"].items():
            np.savez(raw_dir / f"{name}.npz", **scores)
        np.savez(raw_dir / "BENIGN.npz", **result["scores_test_benign"])

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "metric_rows": result["metric_rows"], "tau_row": tau_row,
                "family_delong": family_delong, "score_delong": score_delong,
                "sample_control": sample_control,
            }, f)

        auc_main = {r["group"]: r["auc_roc"] for r in result["metric_rows"]
                    if r["score"] == PRINCIPAL_SCORE and r["group"] in rcfg.MAIN_FAMILIES}
        print(f"[fase4 seed={seed}] tau_p95={result['tau_p95']:.4f} "
              f"fpr_test_benign={result['fpr_test_benign_at_tau_p95']:.4f} "
              f"AUC({PRINCIPAL_SCORE}) principales: {auc_main}")

    all_metric_rows, all_family_delong, all_score_delong, all_sample_control, tau_rows = [], [], [], [], []
    for seed in seeds:
        cache_path = CACHE_DIR / f"phase4_seed{seed}.json"
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        all_metric_rows.extend(cached["metric_rows"])
        tau_rows.append(cached["tau_row"])
        all_family_delong.extend(cached["family_delong"])
        all_score_delong.extend(cached["score_delong"])
        all_sample_control.extend(cached["sample_control"])

    write_csv(all_metric_rows, rcfg.TABLES_DIR / "phase4_detectability_metrics.csv")
    write_csv(all_family_delong, rcfg.TABLES_DIR / "phase4_delong_between_families.csv")
    write_csv(all_score_delong, rcfg.TABLES_DIR / "phase4_delong_between_scores.csv")
    write_csv(all_sample_control, rcfg.TABLES_DIR / "phase4_sample_size_control.csv")
    write_csv(tau_rows, rcfg.TABLES_DIR / "phase4_tau_per_seed.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(seeds=[args.seed] if args.seed is not None else None)
