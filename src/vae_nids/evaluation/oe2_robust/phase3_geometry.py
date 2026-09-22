"""Fase 3 -- geometría distribucional del espacio latente, por semilla.

Responde a la crítica 1 del director: no un solo número por familia sobre
mu, sino (a) la distribución completa de Mahalanobis con pruebas formales y
tamaño de efecto, (b) divergencias entre nubes completas (MMD, Wasserstein
deslizada, distancia de energía), (c) métricas que usan sigma^2 del
posterior (Bhattacharyya, Mahalanobis esperada bajo el posterior,
incertidumbre posterior), (d) solapamiento/densidad local, (e) dispersión
intra-clase, todo con intervalos de confianza.

Uso:
    python -m vae_nids.evaluation.oe2_robust.phase3_geometry
    python -m vae_nids.evaluation.oe2_robust.phase3_geometry --seed 42
"""
import argparse
import csv
import json

import numpy as np
from scipy import stats as sp_stats
from scipy.special import gamma as gamma_function
from sklearn.covariance import MinCovDet
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from vae_nids.evaluation.metrics_oe1 import active_units
from vae_nids.evaluation.oe2_robust import config as rcfg
from vae_nids.evaluation.oe2_robust import stats_utils as su

RNG_SEED = 42
MCD_MAX_N = 20_000  # submuestra del benigno para MinCovDet (FastMCD es O(n) por trial pero con muchos trials -- 20k es tratable en segundos)


def load_group_mu_logvar(seed: int, group: str) -> tuple[np.ndarray, np.ndarray]:
    d = rcfg.LATENT_DIR / f"seed{seed}"
    if group == "latent_benign_test" or group == "BENIGN":
        mu = np.load(d / "latent_benign_test_mu.npy")
        logvar = np.load(d / "latent_benign_test_logvar.npy")
    else:
        mu = np.load(d / f"latent_attack_{group}_mu.npy")
        logvar = np.load(d / f"latent_attack_{group}_logvar.npy")
    return mu, logvar


def mahalanobis_batch(x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    diff = x - mean
    sq = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    return np.sqrt(np.maximum(sq, 0.0))


def subsample(x: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if x.shape[0] <= max_n:
        return x
    idx = rng.choice(x.shape[0], size=max_n, replace=False)
    return x[idx]


def group_names(manifest: dict) -> list[str]:
    return [name for name, info in manifest.items() if not info.get("skipped") and name != "latent_benign_test"]


# ============================================================
# 5.1 -- distribución de Mahalanobis + pruebas + Cliff's delta + robustez MCD
# ============================================================

def phase_5_1(seed: int, active_dims: list[int], manifest: dict) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    mu_benign_full, _ = load_group_mu_logvar(seed, "latent_benign_test")
    mu_benign = mu_benign_full[:, active_dims]
    mean_b = mu_benign.mean(axis=0)
    cov_b = np.cov(mu_benign, rowvar=False)
    inv_cov_b = np.linalg.inv(cov_b)
    dist_benign = mahalanobis_batch(mu_benign, mean_b, inv_cov_b)

    # covarianza robusta (MCD) sobre una submuestra tratable del benigno
    mcd_sub = subsample(mu_benign, MCD_MAX_N, rng)
    mcd = MinCovDet(random_state=RNG_SEED).fit(mcd_sub)
    mean_b_robust = mcd.location_
    inv_cov_b_robust = np.linalg.inv(mcd.covariance_)

    names = group_names(manifest)
    summary_rows, test_rows, cliff_rows, ranking_std, ranking_robust = [], [], [], [], []

    for name in names:
        mu_g_full, _ = load_group_mu_logvar(seed, name)
        mu_g = mu_g_full[:, active_dims]
        dist_g = mahalanobis_batch(mu_g, mean_b, inv_cov_b)
        dist_g_robust = mahalanobis_batch(mu_g, mean_b_robust, inv_cov_b_robust)

        for metric, val in [
            ("mean", np.mean(dist_g)), ("median", np.median(dist_g)),
            ("p5", np.percentile(dist_g, 5)), ("p25", np.percentile(dist_g, 25)),
            ("p75", np.percentile(dist_g, 75)), ("p95", np.percentile(dist_g, 95)),
            ("iqr", np.percentile(dist_g, 75) - np.percentile(dist_g, 25)),
            ("mean_robust_mcd", np.mean(dist_g_robust)),
            ("median_robust_mcd", np.median(dist_g_robust)),
        ]:
            summary_rows.append({"seed": seed, "group": name, "metric": f"mahalanobis_{metric}", "value": float(val)})

        u_stat, mw_p = sp_stats.mannwhitneyu(dist_g, dist_benign, alternative="two-sided")
        ks_stat, ks_p = sp_stats.ks_2samp(dist_g, dist_benign)
        test_rows.append({"seed": seed, "group": name, "test": "mannwhitney_u",
                           "statistic": float(u_stat), "p_value": float(mw_p)})
        test_rows.append({"seed": seed, "group": name, "test": "ks_2samp",
                           "statistic": float(ks_stat), "p_value": float(ks_p)})

        # el delta puntual se calcula sobre la distribución COMPLETA (barato,
        # una sola pasada O(n log n)); el bootstrap de su IC (500 remuestras)
        # sobre esas mismas 250k+ observaciones por familia sería
        # computacionalmente inviable (18 familias x 10 semillas) -- se
        # submuestrea a SILHOUETTE_MAX_N por lado para el IC, mismo criterio
        # de tratabilidad ya usado para silhouette/divergencias en esta fase.
        delta_full = su.cliffs_delta(dist_g, dist_benign)
        dist_g_sub = subsample(dist_g, rcfg.SILHOUETTE_MAX_N, rng)
        dist_benign_sub = subsample(dist_benign, rcfg.SILHOUETTE_MAX_N, rng)
        _, lo, hi = su.cliffs_delta_bootstrap_ci(dist_g_sub, dist_benign_sub,
                                                  n_boot=rcfg.N_BOOTSTRAP_GEOMETRY, seed=RNG_SEED)
        cliff_rows.append({"seed": seed, "group": name, "metric": "cliffs_delta",
                            "value": delta_full, "ci_low": lo, "ci_high": hi,
                            "ci_n_subsample_per_side": min(rcfg.SILHOUETTE_MAX_N, len(dist_g), len(dist_benign))})

        ranking_std.append((name, np.median(dist_g)))
        ranking_robust.append((name, np.median(dist_g_robust)))

    # --- Holm sobre las 18 familias, por tipo de test ---
    for test_name in ["mannwhitney_u", "ks_2samp"]:
        rows_t = [r for r in test_rows if r["test"] == test_name]
        p_adj = su.holm_correction(np.array([r["p_value"] for r in rows_t]))
        for r, pa in zip(rows_t, p_adj):
            r["p_holm"] = float(pa)

    # --- estabilidad del ranking bajo covarianza robusta ---
    main_std = [(n, v) for n, v in ranking_std if n in rcfg.MAIN_FAMILIES]
    main_robust = [(n, v) for n, v in ranking_robust if n in rcfg.MAIN_FAMILIES]
    order = [n for n, _ in main_std]
    vals_std = [v for _, v in main_std]
    vals_robust = [dict(main_robust)[n] for n in order]
    rho_robust, p_robust = sp_stats.spearmanr(vals_std, vals_robust)

    return {
        "summary_rows": summary_rows,
        "test_rows": test_rows,
        "cliff_rows": cliff_rows,
        "robust_ranking_rho": float(rho_robust),
        "robust_ranking_p": float(p_robust),
        "mcd_n_used": int(mcd_sub.shape[0]),
    }


# ============================================================
# 5.2 -- divergencias entre nubes completas (MMD, SW, energy)
# ============================================================

def phase_5_2(seed: int, active_dims: list[int], manifest: dict) -> list[dict]:
    rng_master = np.random.default_rng(RNG_SEED)
    mu_benign_full, _ = load_group_mu_logvar(seed, "latent_benign_test")
    mu_benign = mu_benign_full[:, active_dims].astype(np.float32)

    rows = []
    for name in group_names(manifest):
        mu_g_full, _ = load_group_mu_logvar(seed, name)
        mu_g = mu_g_full[:, active_dims].astype(np.float32)

        mmd_vals, sw_vals, en_vals = [], [], []
        for rep in range(rcfg.N_DIVERGENCE_REPEATS):
            rep_rng = np.random.default_rng(RNG_SEED * 10_000 + rep)
            xb = subsample(mu_benign, rcfg.DIVERGENCE_MAX_N, rep_rng)
            xg = subsample(mu_g, rcfg.DIVERGENCE_MAX_N, rep_rng)
            mmd2, _ = su.mmd2_rbf(xb, xg)
            sw = su.sliced_wasserstein(xb, xg, n_projections=200, seed=int(rep_rng.integers(0, 1_000_000)))
            en = su.energy_distance(xb, xg)
            mmd_vals.append(mmd2)
            sw_vals.append(sw)
            en_vals.append(en)

        for metric, vals in [("mmd2_rbf", mmd_vals), ("sliced_wasserstein", sw_vals), ("energy_distance", en_vals)]:
            arr = np.array(vals)
            lo, hi = np.percentile(arr, [2.5, 97.5])
            rows.append({
                "seed": seed, "group": name, "metric": metric,
                "value": float(arr.mean()), "ci_low": float(lo), "ci_high": float(hi),
                "n_repeats": rcfg.N_DIVERGENCE_REPEATS,
                "n_subsample_per_side": min(rcfg.DIVERGENCE_MAX_N, mu_benign.shape[0], mu_g.shape[0]),
            })
        print(f"[fase3.2 seed={seed}] {name}: MMD2={np.mean(mmd_vals):.4f} "
              f"SW={np.mean(sw_vals):.4f} energy={np.mean(en_vals):.4f}")
    return rows


# ============================================================
# 5.3 -- Bhattacharyya, Mahalanobis esperada bajo posterior, incertidumbre
# ============================================================

def phase_5_3(seed: int, active_dims: list[int], manifest: dict, l_mc: int = rcfg.L_MC_SAMPLES) -> dict:
    mu_benign_full, logvar_benign_full = load_group_mu_logvar(seed, "latent_benign_test")
    mu_benign = mu_benign_full[:, active_dims]
    logvar_benign = logvar_benign_full[:, active_dims]
    mean_b = mu_benign.mean(axis=0)
    cov_b = np.cov(mu_benign, rowvar=False)
    inv_cov_b = np.linalg.inv(cov_b)

    bhatt_rows, posterior_maha_rows, uncertainty_rows = [], [], []
    mc_rng = np.random.default_rng(rcfg.MC_SEED_BASE + seed)

    for name in group_names(manifest):
        mu_g_full, logvar_g_full = load_group_mu_logvar(seed, name)
        mu_g = mu_g_full[:, active_dims]
        logvar_g = logvar_g_full[:, active_dims]

        mean_g = mu_g.mean(axis=0)
        cov_g = np.cov(mu_g, rowvar=False)
        total, term_mean, term_cov = su.bhattacharyya_gaussian(mean_g, cov_g, mean_b, cov_b)
        bhatt_rows.append({"seed": seed, "group": name, "metric": "bhattacharyya_total", "value": total})
        bhatt_rows.append({"seed": seed, "group": name, "metric": "bhattacharyya_mean_term", "value": term_mean})
        bhatt_rows.append({"seed": seed, "group": name, "metric": "bhattacharyya_cov_term", "value": term_cov})

        # Mahalanobis esperada bajo el posterior: L muestras z ~ q(z|x) por conexión
        std_g = np.exp(0.5 * logvar_g)
        n = mu_g.shape[0]
        acc = np.zeros(n)
        for _ in range(l_mc):
            eps = mc_rng.standard_normal(size=mu_g.shape)
            z = mu_g + std_g * eps
            acc += mahalanobis_batch(z, mean_b, inv_cov_b)
        expected_maha = acc / l_mc

        for metric, val in [
            ("mean", np.mean(expected_maha)), ("median", np.median(expected_maha)),
            ("p25", np.percentile(expected_maha, 25)), ("p75", np.percentile(expected_maha, 75)),
        ]:
            posterior_maha_rows.append({"seed": seed, "group": name, "metric": f"posterior_mahalanobis_{metric}",
                                         "value": float(val)})

        mean_sigma2_sum = float(np.mean(np.exp(logvar_g).sum(axis=1)))
        uncertainty_rows.append({"seed": seed, "group": name, "metric": "mean_posterior_variance_sum",
                                  "value": mean_sigma2_sum})

    # incertidumbre posterior del benigno como referencia
    mean_sigma2_sum_benign = float(np.mean(np.exp(logvar_benign).sum(axis=1)))
    uncertainty_rows.append({"seed": seed, "group": "BENIGN", "metric": "mean_posterior_variance_sum",
                              "value": mean_sigma2_sum_benign})

    return {"bhatt_rows": bhatt_rows, "posterior_maha_rows": posterior_maha_rows,
            "uncertainty_rows": uncertainty_rows}


# ============================================================
# 5.4 -- pureza de vecindad + densidad local
# ============================================================

def phase_5_4(seed: int, active_dims: list[int], manifest: dict) -> list[dict]:
    rng = np.random.default_rng(RNG_SEED)
    mu_benign_full, _ = load_group_mu_logvar(seed, "latent_benign_test")
    mu_benign = mu_benign_full[:, active_dims]
    d = len(active_dims)

    # árbol de vecinos benigno completo (submuestreado si es enorme) para densidad kNN
    benign_density_ref = subsample(mu_benign, 50_000, rng)
    nn_density = NearestNeighbors(n_neighbors=rcfg.KNN_K).fit(benign_density_ref)
    n_ref = benign_density_ref.shape[0]
    vol_unit_ball = np.pi ** (d / 2) / gamma_function(d / 2 + 1)

    rows = []
    for name in group_names(manifest):
        mu_g_full, _ = load_group_mu_logvar(seed, name)
        mu_g = mu_g_full[:, active_dims]

        # --- densidad local benigna evaluada en los puntos de ataque (kNN-density) ---
        dists, _ = nn_density.kneighbors(mu_g)
        r_k = np.maximum(dists[:, -1], 1e-12)
        log_density = (np.log(rcfg.KNN_K) - np.log(n_ref) - np.log(vol_unit_ball) - d * np.log(r_k))
        rows.append({"seed": seed, "group": name, "metric": "log_density_benign_mean",
                     "value": float(np.mean(log_density))})
        rows.append({"seed": seed, "group": name, "metric": "log_density_benign_median",
                     "value": float(np.median(log_density))})

        # --- pureza de vecindad: referencia balanceada benigno+ataque submuestreada ---
        half = rcfg.KNN_REFERENCE_MAX_N // 2
        benign_ref = subsample(mu_benign, half, rng)
        attack_ref = subsample(mu_g, half, rng)
        ref = np.vstack([benign_ref, attack_ref])
        is_benign_ref = np.concatenate([np.ones(len(benign_ref)), np.zeros(len(attack_ref))]).astype(bool)

        # k+1 vecinos porque la query puede coincidir físicamente con un punto
        # de `ref` (mismo dataset de ataque submuestreado dos veces) -- se
        # descarta como vecino-0 solo cuando la distancia es exactamente 0,
        # y se usan los siguientes K en cualquier caso.
        query = subsample(mu_g, min(5000, mu_g.shape[0]), rng)
        nn_purity = NearestNeighbors(n_neighbors=rcfg.KNN_K + 1).fit(ref)
        dists_q, idx = nn_purity.kneighbors(query)
        purity_vals = []
        for row_dist, row_idx in zip(dists_q, idx):
            neighbor_idx = row_idx[1:] if row_dist[0] == 0.0 else row_idx[:-1]
            purity_vals.append(is_benign_ref[neighbor_idx].mean())
        purity_vals = np.array(purity_vals)

        rows.append({"seed": seed, "group": name, "metric": "neighborhood_purity_mean",
                     "value": float(purity_vals.mean())})
        rows.append({"seed": seed, "group": name, "metric": "neighborhood_purity_median",
                     "value": float(np.median(purity_vals))})
    return rows


# ============================================================
# 5.5 -- dispersión intra-clase + silhouette repetido
# ============================================================

def phase_5_5(seed: int, active_dims: list[int], manifest: dict) -> list[dict]:
    rng = np.random.default_rng(RNG_SEED)
    mu_benign_full, _ = load_group_mu_logvar(seed, "latent_benign_test")
    mu_benign = mu_benign_full[:, active_dims]

    rows = []
    for name in group_names(manifest):
        mu_g_full, _ = load_group_mu_logvar(seed, name)
        mu_g = mu_g_full[:, active_dims]

        cov_g = np.cov(mu_g, rowvar=False) if mu_g.shape[0] > 1 else np.zeros((len(active_dims),) * 2)
        trace = float(np.trace(cov_g))
        sign, logdet = np.linalg.slogdet(cov_g) if mu_g.shape[0] > len(active_dims) else (0.0, float("nan"))
        rows.append({"seed": seed, "group": name, "metric": "cov_trace", "value": trace})
        rows.append({"seed": seed, "group": name, "metric": "cov_logdet", "value": float(logdet) if sign > 0 else float("nan")})

        sils = []
        for rep in range(rcfg.N_SILHOUETTE_REPEATS):
            rep_rng = np.random.default_rng(RNG_SEED * 10_000 + rep)
            b_sub = subsample(mu_benign, rcfg.SILHOUETTE_MAX_N, rep_rng)
            g_sub = subsample(mu_g, rcfg.SILHOUETTE_MAX_N, rep_rng)
            x = np.vstack([b_sub, g_sub])
            labels = np.array([0] * len(b_sub) + [1] * len(g_sub))
            sils.append(silhouette_score(x, labels))
        sils = np.array(sils)
        lo, hi = np.percentile(sils, [2.5, 97.5])
        rows.append({"seed": seed, "group": name, "metric": "silhouette_mean",
                     "value": float(sils.mean()), "ci_low": float(lo), "ci_high": float(hi)})
    return rows


# ============================================================
# orquestación por semilla
# ============================================================

def run_seed(seed: int, active_dims_map: dict) -> dict:
    manifest_path = rcfg.LATENT_DIR / f"seed{seed}" / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    active_dims = active_dims_map[str(seed)] if str(seed) in active_dims_map else active_dims_map[seed]
    print(f"[fase3 seed={seed}] dims activas usadas: {active_dims}")

    r51 = phase_5_1(seed, active_dims, manifest)
    r52 = phase_5_2(seed, active_dims, manifest)
    r53 = phase_5_3(seed, active_dims, manifest)
    r54 = phase_5_4(seed, active_dims, manifest)
    r55 = phase_5_5(seed, active_dims, manifest)

    return {
        "mahalanobis_summary": r51["summary_rows"],
        "mahalanobis_tests": r51["test_rows"],
        "cliffs_delta": r51["cliff_rows"],
        "robust_ranking": [{"seed": seed, "metric": "spearman_rho_standard_vs_robust_mcd",
                            "value": r51["robust_ranking_rho"], "p_value": r51["robust_ranking_p"],
                            "mcd_n_used": r51["mcd_n_used"]}],
        "divergences": r52,
        "bhattacharyya": r53["bhatt_rows"],
        "posterior_mahalanobis": r53["posterior_maha_rows"],
        "posterior_uncertainty": r53["uncertainty_rows"],
        "neighborhood": r54,
        "dispersion": r55,
    }


def write_csv(rows: list[dict], path):
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[fase3] {path} ({len(rows)} filas)")


def main(seeds: list[int] | None = None):
    seeds = seeds if seeds is not None else rcfg.ALL_SEEDS
    with open(rcfg.METRICS_DIR / "phase2_active_dims_per_seed.json", encoding="utf-8") as f:
        active_dims_map = json.load(f)

    all_results = {k: [] for k in [
        "mahalanobis_summary", "mahalanobis_tests", "cliffs_delta", "robust_ranking",
        "divergences", "bhattacharyya", "posterior_mahalanobis", "posterior_uncertainty",
        "neighborhood", "dispersion",
    ]}
    for seed in seeds:
        result = run_seed(seed, active_dims_map)
        for k in all_results:
            all_results[k].extend(result[k])

    for key, rows in all_results.items():
        write_csv(rows, rcfg.TABLES_DIR / f"phase3_{key}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(seeds=[args.seed] if args.seed is not None else None)
