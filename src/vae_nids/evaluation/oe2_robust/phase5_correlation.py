"""Fase 5 -- correlación geometría-detectabilidad, robustecida.

7.1: hipótesis principal (FIJADA antes de ver resultados, ver docstring de
     `PRINCIPAL_PAIR` -- no se cambia con los datos ya generados).
7.2: análisis secundarios (todas las métricas geométricas x 3 métricas de
     detectabilidad), con corrección Holm/BH.
7.3: modelo de efectos mixtos a nivel de conexión (el que tiene potencia).
7.4: estabilidad de rankings entre semillas (W de Kendall).
7.5: comparación completado/attempted, por semilla.

Uso:
    python -m vae_nids.evaluation.oe2_robust.phase5_correlation
"""
import csv
import json

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from vae_nids.evaluation.oe2_robust import config as rcfg
from vae_nids.evaluation.oe2_robust import stats_utils as su

# --------------------------------------------------------------------
# 7.1 -- PAR PRINCIPAL, FIJADO ANTES DE VER LOS RESULTADOS DE ESTE PIPELINE
# (mediana de Mahalanobis con mu / dims activas, vs. AUC-ROC de nll_mc,
# sobre las 11 familias principales, agregado entre semillas). No se toca.
# --------------------------------------------------------------------
PRINCIPAL_GEOMETRY_METRIC = "mahalanobis_median"
PRINCIPAL_DETECT_SCORE = "nll_mc"
PRINCIPAL_DETECT_METRIC = "auc_roc"

SECONDARY_DETECT_METRICS = ["auc_roc", "auc_pr", "tpr_at_fpr5"]


def write_csv(rows: list[dict], path):
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[fase5] {path} ({len(rows)} filas)")


# ============================================================
# carga y ensamblado de tablas largas -> tabla ancha (seed, group, metric) -> valor
# ============================================================

def load_geometry_wide() -> pd.DataFrame:
    """Concatena todas las tablas de la Fase 3 (formato largo seed,group,metric,value)
    en una sola tabla larga homogénea."""
    frames = []
    for name in ["mahalanobis_summary", "divergences", "bhattacharyya",
                 "posterior_mahalanobis", "posterior_uncertainty", "neighborhood", "dispersion"]:
        path = rcfg.TABLES_DIR / f"phase3_{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        keep = [c for c in ["seed", "group", "metric", "value"] if c in df.columns]
        frames.append(df[keep])
    return pd.concat(frames, ignore_index=True)


# ============================================================
# 7.1 -- hipótesis principal
# ============================================================

def hypothesis_principal(geo: pd.DataFrame, det: pd.DataFrame, seeds: list[int]) -> dict:
    geo_p = geo[(geo["metric"] == PRINCIPAL_GEOMETRY_METRIC) & (geo["group"].isin(rcfg.MAIN_FAMILIES))]
    per_seed_rows = []
    for seed in seeds:
        g = geo_p[geo_p["seed"] == seed].set_index("group")["value"]
        d = det[(det["seed"] == seed) & (det["group"].isin(rcfg.MAIN_FAMILIES))].set_index("group")[PRINCIPAL_DETECT_METRIC]
        common = sorted(set(g.index) & set(d.index))
        assert len(common) == len(rcfg.MAIN_FAMILIES), (
            f"seed={seed}: esperaba {len(rcfg.MAIN_FAMILIES)} familias, encontré {len(common)}"
        )
        x, y = g.loc[common].values, d.loc[common].values
        rho, p_asymptotic = sp_stats.spearmanr(x, y)
        perm = su.permutation_test_spearman(x, y, n_perm=rcfg.N_PERMUTATION, seed=rcfg.PERMUTATION_SEED)
        per_seed_rows.append({
            "seed": seed, "rho": float(rho), "p_asymptotic": float(p_asymptotic),
            "p_permutation": perm["p_permutation"], "n_families": len(common),
        })

    rhos = np.array([r["rho"] for r in per_seed_rows])
    combined = su.fisher_combine(rhos)

    power = {
        "n_families": len(rcfg.MAIN_FAMILIES),
        "min_detectable_rho_alpha05": su.spearman_min_detectable_rho(len(rcfg.MAIN_FAMILIES)),
        "power_rho_0.3": su.spearman_power(len(rcfg.MAIN_FAMILIES), 0.3),
        "power_rho_0.5": su.spearman_power(len(rcfg.MAIN_FAMILIES), 0.5),
        "power_rho_0.7": su.spearman_power(len(rcfg.MAIN_FAMILIES), 0.7),
    }

    result = {
        "pair": f"{PRINCIPAL_GEOMETRY_METRIC} vs. {PRINCIPAL_DETECT_METRIC}({PRINCIPAL_DETECT_SCORE})",
        "per_seed": per_seed_rows,
        "combined": combined,
        "power": power,
        "previous_oe2_single_seed": {"rho": 0.3818, "p_value": 0.2466, "n": 11},
    }
    print(f"[fase5 7.1] rho combinado={combined['rho_combined']:.4f} "
          f"IC95=[{combined['ci_low']:.4f},{combined['ci_high']:.4f}] "
          f"(media por semilla={combined['rho_per_seed_mean']:.4f}, sd={combined['rho_per_seed_sd']:.4f})")
    return result


# ============================================================
# 7.2 -- análisis secundarios
# ============================================================

def secondary_analysis(geo: pd.DataFrame, det: pd.DataFrame, seeds: list[int]) -> dict:
    geo_metrics = sorted(geo["metric"].unique())
    per_seed_rows, pair_summary_rows = [], []

    for geo_metric in geo_metrics:
        geo_m = geo[(geo["metric"] == geo_metric) & (geo["group"].isin(rcfg.MAIN_FAMILIES))]
        if geo_m["seed"].nunique() < len(seeds):
            continue
        for detect_metric in SECONDARY_DETECT_METRICS:
            rhos_seed, taus_seed = [], []
            avg_geo, avg_det = {}, {}
            for seed in seeds:
                g = geo_m[geo_m["seed"] == seed].set_index("group")["value"]
                d = det[(det["seed"] == seed) & (det["group"].isin(rcfg.MAIN_FAMILIES))].set_index("group")[detect_metric]
                common = sorted(set(g.index) & set(d.index))
                if len(common) < 4:
                    continue
                x, y = g.loc[common].values.astype(float), d.loc[common].values.astype(float)
                rho, _ = sp_stats.spearmanr(x, y)
                tau, _ = sp_stats.kendalltau(x, y)
                rhos_seed.append(rho)
                taus_seed.append(tau)
                per_seed_rows.append({
                    "seed": seed, "geometry_metric": geo_metric, "detect_metric": detect_metric,
                    "spearman_rho": float(rho), "kendall_tau": float(tau), "n_families": len(common),
                })
                for fam in common:
                    avg_geo.setdefault(fam, []).append(g.loc[fam])
                    avg_det.setdefault(fam, []).append(d.loc[fam])

            if len(rhos_seed) < 2:
                continue
            combined = su.fisher_combine(np.array(rhos_seed))

            # p-valor representativo para la corrección múltiple: sobre los
            # valores promediados entre semillas (una relación por par,
            # documentado en el informe -- evita el problema de "cuál de las
            # 10 p permutación por semilla se corrige").
            common_fams = sorted(avg_geo.keys())
            x_avg = np.array([np.mean(avg_geo[f]) for f in common_fams])
            y_avg = np.array([np.mean(avg_det[f]) for f in common_fams])
            perm = su.permutation_test_spearman(x_avg, y_avg, n_perm=rcfg.N_PERMUTATION, seed=rcfg.PERMUTATION_SEED)
            # bootstrap sobre familias (resamplea el ÍNDICE 0..n-1 con
            # reemplazo, no los valores directamente, para poder indexar
            # x_avg/y_avg de forma pareada en cada remuestra)
            _, boot_lo, boot_hi = su.bootstrap_ci_1sample(
                np.arange(len(common_fams)),
                lambda idx: sp_stats.spearmanr(x_avg[idx], y_avg[idx])[0],
                n_boot=rcfg.N_BOOTSTRAP_GEOMETRY, seed=rcfg.PERMUTATION_SEED,
            )

            pair_summary_rows.append({
                "geometry_metric": geo_metric, "detect_metric": detect_metric,
                "rho_combined": combined["rho_combined"], "rho_ci_low": combined["ci_low"],
                "rho_ci_high": combined["ci_high"], "rho_per_seed_mean": combined["rho_per_seed_mean"],
                "rho_per_seed_sd": combined["rho_per_seed_sd"], "n_seeds": combined["n_seeds"],
                "rho_seed_averaged_data": float(perm["rho_obs"]),
                "p_permutation_seed_averaged": perm["p_permutation"],
                "rho_bootstrap_over_families_ci_low": boot_lo, "rho_bootstrap_over_families_ci_high": boot_hi,
            })

    pvals = np.array([r["p_permutation_seed_averaged"] for r in pair_summary_rows])
    if len(pvals) > 0:
        p_holm = su.holm_correction(pvals)
        p_bh = su.bh_correction(pvals)
        for r, ph, pb in zip(pair_summary_rows, p_holm, p_bh):
            r["p_holm"] = float(ph)
            r["p_bh"] = float(pb)

    n_sig_raw = int(np.sum(pvals < 0.05)) if len(pvals) else 0
    n_sig_holm = int(np.sum([r.get("p_holm", 1.0) < 0.05 for r in pair_summary_rows]))
    print(f"[fase5 7.2] {len(pair_summary_rows)} pares geometría x detectabilidad evaluados; "
          f"{n_sig_raw} con p crudo <0.05, {n_sig_holm} tras corrección de Holm")

    return {"per_seed_rows": per_seed_rows, "pair_summary_rows": pair_summary_rows}


def ranking_correlation_latent_maha_vs_nll_mc(det_full: pd.DataFrame, seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        d_nll = det_full[(det_full["seed"] == seed) & (det_full["score"] == "nll_mc")
                          & (det_full["group"].isin(rcfg.MAIN_FAMILIES))].set_index("group")["auc_roc"]
        d_maha = det_full[(det_full["seed"] == seed) & (det_full["score"] == "latent_maha")
                           & (det_full["group"].isin(rcfg.MAIN_FAMILIES))].set_index("group")["auc_roc"]
        common = sorted(set(d_nll.index) & set(d_maha.index))
        rho, p = sp_stats.spearmanr(d_nll.loc[common].values, d_maha.loc[common].values)
        rows.append({"seed": seed, "metric": "ranking_corr_nll_mc_vs_latent_maha", "rho": float(rho), "p_value": float(p)})
    return rows


# ============================================================
# 7.3 -- modelo de efectos mixtos a nivel de conexión
# ============================================================

def build_connection_level_data(seeds: list[int]) -> pd.DataFrame:
    rng = np.random.default_rng(rcfg.MIXED_MODEL_SEED)
    rows_x, rows_y, rows_family, rows_seed = [], [], [], []
    groups = rcfg.MAIN_FAMILIES + ["BENIGN"]
    for seed in seeds:
        raw_dir = rcfg.LATENT_DIR / f"seed{seed}" / "scores"
        for group in groups:
            path = raw_dir / f"{group}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            maha = data["latent_maha"]
            score = data["nll_mc"]
            n = len(maha)
            take = min(n, rcfg.MIXED_MODEL_MAX_N_PER_GROUP_PER_SEED)
            idx = rng.choice(n, size=take, replace=False) if n > take else np.arange(n)
            rows_x.append(maha[idx])
            rows_y.append(score[idx])
            rows_family.extend([group] * take)
            rows_seed.extend([seed] * take)
    df = pd.DataFrame({
        "maha": np.concatenate(rows_x),
        "score": np.concatenate(rows_y),
        "family": rows_family,
        "seed": rows_seed,
    })
    df["maha_std"] = (df["maha"] - df["maha"].mean()) / df["maha"].std()
    df["score_std"] = (df["score"] - df["score"].mean()) / df["score"].std()
    return df


def fit_mixed_model(df: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf

    attempts = []

    def try_fit(desc, **kwargs):
        try:
            model = smf.mixedlm("score_std ~ maha_std", data=df, groups=df["family"], **kwargs)
            result = model.fit(reml=True, method="lbfgs", maxiter=200)
            attempts.append({"description": desc, "converged": bool(result.converged), "error": None})
            return result
        except Exception as e:  # noqa: BLE001 -- se documenta cualquier fallo, no se oculta
            attempts.append({"description": desc, "converged": False, "error": str(e)})
            return None

    result = try_fit("intercepto+pendiente aleatoria por familia + componente de varianza por semilla",
                      re_formula="~maha_std", vc_formula={"seed": "0 + C(seed)"})
    if result is None or not result.converged:
        result = try_fit("intercepto+pendiente aleatoria por familia (sin componente de semilla)",
                          re_formula="~maha_std")
    if result is None or not result.converged:
        result = try_fit("solo intercepto aleatorio por familia", re_formula=None)

    if result is None:
        return {"attempts": attempts, "converged": False}

    fixed_effect = float(result.fe_params["maha_std"])
    fixed_ci = result.conf_int().loc["maha_std"].tolist()

    # R^2 marginal/condicional (Nakagawa & Schielzeth, 2013)
    fitted_fixed = result.model.exog @ result.fe_params.values
    var_fixed = float(np.var(fitted_fixed))
    var_random = 0.0
    try:
        var_random += float(np.trace(result.cov_re.values)) if hasattr(result, "cov_re") else 0.0
    except Exception:
        pass
    if hasattr(result, "vcomp") and result.vcomp is not None and len(result.vcomp) > 0:
        var_random += float(np.sum(result.vcomp))
    var_resid = float(result.scale)
    r2_marginal = var_fixed / (var_fixed + var_random + var_resid)
    r2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_resid)

    slope_variance = None
    try:
        slope_variance = float(result.cov_re.loc["maha_std", "maha_std"])
    except Exception:
        pass

    return {
        "attempts": attempts, "converged": bool(result.converged),
        "n_obs": int(result.nobs), "fixed_effect_maha_std": fixed_effect,
        "fixed_effect_ci_low": fixed_ci[0], "fixed_effect_ci_high": fixed_ci[1],
        "fixed_effect_p_value": float(result.pvalues["maha_std"]),
        "random_slope_variance_by_family": slope_variance,
        "var_fixed": var_fixed, "var_random": var_random, "var_residual": var_resid,
        "r2_marginal": r2_marginal, "r2_conditional": r2_conditional,
        "summary_text": str(result.summary()),
    }


def connection_level_spearman(seeds: list[int]) -> list[dict]:
    rows = []
    groups = rcfg.MAIN_FAMILIES
    for seed in seeds:
        raw_dir = rcfg.LATENT_DIR / f"seed{seed}" / "scores"
        for group in groups:
            path = raw_dir / f"{group}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            if len(data["latent_maha"]) < 5:
                continue
            rho, p = sp_stats.spearmanr(data["latent_maha"], data["nll_mc"])
            rows.append({"seed": seed, "group": group, "spearman_rho_connection_level": float(rho), "p_value": float(p),
                         "n": int(len(data["latent_maha"]))})
    return rows


# ============================================================
# 7.4 -- estabilidad de rankings entre semillas
# ============================================================

def seed_stability(geo: pd.DataFrame, det: pd.DataFrame, seeds: list[int]) -> list[dict]:
    rows = []
    families = rcfg.MAIN_FAMILIES

    # AUC-ROC (nll_mc)
    mat = np.full((len(seeds), len(families)), np.nan)
    for i, seed in enumerate(seeds):
        d = det[(det["seed"] == seed) & (det["group"].isin(families))].set_index("group")["auc_roc"]
        for j, fam in enumerate(families):
            mat[i, j] = d.get(fam, np.nan)
    rows.append({"metric": "auc_roc_nll_mc", "kendalls_w": su.kendalls_w(mat), "n_seeds": len(seeds), "n_families": len(families)})

    for geo_metric in [PRINCIPAL_GEOMETRY_METRIC, "mmd2_rbf", "sliced_wasserstein", "energy_distance", "bhattacharyya_total"]:
        gm = geo[(geo["metric"] == geo_metric) & (geo["group"].isin(families))]
        if gm.empty:
            continue
        mat = np.full((len(seeds), len(families)), np.nan)
        for i, seed in enumerate(seeds):
            g = gm[gm["seed"] == seed].set_index("group")["value"]
            for j, fam in enumerate(families):
                mat[i, j] = g.get(fam, np.nan)
        if np.isnan(mat).any():
            continue
        rows.append({"metric": geo_metric, "kendalls_w": su.kendalls_w(mat), "n_seeds": len(seeds), "n_families": len(families)})

    for r in rows:
        print(f"[fase5 7.4] W de Kendall ({r['metric']}): {r['kendalls_w']:.4f}")
    return rows


# ============================================================
# 7.5 -- completado vs. attempted, por semilla
# ============================================================

def attempted_comparison(geo: pd.DataFrame, det: pd.DataFrame, seeds: list[int]) -> list[dict]:
    rows = []
    geo_maha = geo[geo["metric"] == PRINCIPAL_GEOMETRY_METRIC]
    geo_mmd = geo[geo["metric"] == "mmd2_rbf"]
    for completo, attempted in rcfg.ATTEMPTED_PAIRS:
        n_confirm = 0
        for seed in seeds:
            def get(df_, group, seed_):
                v = df_[(df_["seed"] == seed_) & (df_["group"] == group)]["value"]
                return float(v.iloc[0]) if len(v) else np.nan

            def get_auc(group, seed_):
                v = det[(det["seed"] == seed_) & (det["group"] == group)]["auc_roc"]
                return float(v.iloc[0]) if len(v) else np.nan

            d_maha = get(geo_maha, completo, seed) - get(geo_maha, attempted, seed)
            d_mmd = get(geo_mmd, completo, seed) - get(geo_mmd, attempted, seed)
            d_auc = get_auc(completo, seed) - get_auc(attempted, seed)
            confirms = (d_maha > 0) and (d_auc > 0)
            n_confirm += int(confirms)
            rows.append({
                "pair": f"{completo} vs {attempted}", "seed": seed,
                "delta_mahalanobis_median": d_maha, "delta_mmd2": d_mmd, "delta_auc_roc": d_auc,
                "confirms_hypothesis": confirms,
            })
    return rows


# ============================================================
# orquestación
# ============================================================

def main(seeds: list[int] | None = None):
    seeds = seeds if seeds is not None else rcfg.ALL_SEEDS
    geo = load_geometry_wide()
    det_full = pd.read_csv(rcfg.TABLES_DIR / "phase4_detectability_metrics.csv")
    det = det_full[det_full["score"] == PRINCIPAL_DETECT_SCORE].copy()

    h1 = hypothesis_principal(geo, det, seeds)
    with open(rcfg.METRICS_DIR / "phase5_hypothesis_principal.json", "w", encoding="utf-8") as f:
        json.dump(h1, f, indent=2)

    sec = secondary_analysis(geo, det, seeds)
    write_csv(sec["per_seed_rows"], rcfg.TABLES_DIR / "phase5_secondary_per_seed.csv")
    write_csv(sec["pair_summary_rows"], rcfg.TABLES_DIR / "phase5_secondary_summary.csv")

    ranking_rows = ranking_correlation_latent_maha_vs_nll_mc(det_full, seeds)
    write_csv(ranking_rows, rcfg.TABLES_DIR / "phase5_ranking_correlation.csv")

    conn_df = build_connection_level_data(seeds)
    print(f"[fase5 7.3] datos a nivel de conexión: {len(conn_df):,} filas, "
          f"{conn_df['family'].nunique()} grupos, {conn_df['seed'].nunique()} semillas")
    mixed = fit_mixed_model(conn_df)
    with open(rcfg.METRICS_DIR / "phase5_mixed_model.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in mixed.items() if k != "summary_text"}, f, indent=2)
    if "summary_text" in mixed:
        with open(rcfg.METRICS_DIR / "phase5_mixed_model_summary.txt", "w", encoding="utf-8") as f:
            f.write(mixed["summary_text"])
    print(f"[fase5 7.3] modelo mixto: convergió={mixed.get('converged')} "
          f"efecto_fijo(maha_std)={mixed.get('fixed_effect_maha_std')}")

    conn_spearman_rows = connection_level_spearman(seeds)
    write_csv(conn_spearman_rows, rcfg.TABLES_DIR / "phase5_connection_level_spearman.csv")

    stability_rows = seed_stability(geo, det, seeds)
    write_csv(stability_rows, rcfg.TABLES_DIR / "phase5_seed_stability_kendalls_w.csv")

    attempted_rows = attempted_comparison(geo, det, seeds)
    write_csv(attempted_rows, rcfg.TABLES_DIR / "phase5_attempted_comparison.csv")


if __name__ == "__main__":
    main()
