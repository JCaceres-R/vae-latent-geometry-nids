"""Fase 8 -- reconciliación del nivel de conexión: modelo mixto (fase 5 robusta) frente a fase 7.

En la corrida del 11-10-2026 (perfil saneado), el modelo mixto de la fase 5 dio un efecto fijo de
Mahalanobis sobre la NLL de 0,10 [-0,10; 0,31] (p = 0,32; R² marginal 0,002), mientras la fase 7 dio
rho = 0,72 dentro de familia. Esta fase usa la MISMA muestra de conexiones que la fase 5
(build_connection_level_data: hasta 5000 por grupo y semilla, con su semilla de muestreo) y separa
qué decisión de análisis produce la diferencia:

  1. Correlaciones por semilla, de la más parecida a la fase 5 a la más parecida a la fase 7:
     Pearson con el benigno, sin el benigno, dentro de familia (centrado), y Spearman en las mismas
     variantes, más la media de los Spearman por familia y el Spearman entre medianas de familia.
  2. Por familia: Spearman, Pearson, pendiente en escala estandarizada, asimetría de cada variable.
  3. Modelos: el mixto de la fase 5 (leído de su JSON), el mismo sin el benigno, el mismo sobre
     puntuaciones normales de rango (van der Waerden), y MCO con efectos fijos de familia y semilla
     sobre esas puntuaciones (estimador dentro de familia), con errores agrupados por familia.

Uso (perfil saneado):
    VAE_NIDS_OE2_PERFIL=saneado python -m vae_nids.evaluation.oe2_descriptores.fase8_reconciliacion
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, rankdata, skew, spearmanr

from vae_nids.evaluation.oe2_descriptores import config as C
from vae_nids.evaluation.oe2_descriptores.fase3_evaluacion import fisher_mean
from vae_nids.evaluation.oe2_descriptores.fase7_confirmatoria import md_table
from vae_nids.evaluation.oe2_robust import config as rcfg
from vae_nids.evaluation.oe2_robust.phase5_correlation import build_connection_level_data, fit_mixed_model


def normal_scores(x):
    """Puntuaciones normales de van der Waerden: Phi^-1(rango / (n + 1))."""
    return norm.ppf(rankdata(x) / (len(x) + 1))


def centered(v, g):
    codes, inv = np.unique(g, return_inverse=True)
    means = np.bincount(inv, v) / np.bincount(inv)
    return v - means[inv]


def correlations_for_seed(d):
    fam = d[d.family != "BENIGN"]
    x, y, g = fam.maha.to_numpy(), fam.score.to_numpy(), fam.family.to_numpy()
    rx, ry = rankdata(x), rankdata(y)
    per_fam = [spearmanr(s.maha, s.score)[0] for _, s in fam.groupby("family") if len(s) >= 10]
    med = fam.groupby("family")[["maha", "score"]].median()
    return {
        "pearson_con_benigno": pearsonr(d.maha, d.score)[0],
        "pearson_sin_benigno": pearsonr(x, y)[0],
        "pearson_dentro_familia": pearsonr(centered(x, g), centered(y, g))[0],
        "spearman_con_benigno": spearmanr(d.maha, d.score)[0],
        "spearman_sin_benigno": spearmanr(x, y)[0],
        "spearman_dentro_familia": pearsonr(centered(rx, g), centered(ry, g))[0],
        "spearman_medio_por_familia": fisher_mean(per_fam),
        "spearman_entre_medianas_de_familia": spearmanr(med.maha, med.score)[0],
    }


def per_family(df):
    rows = []
    for (seed, g), s in df.groupby(["seed", "family"]):
        if len(s) < 10:
            continue
        x, y = s.maha.to_numpy(), s.score.to_numpy()
        slope = np.polyfit(s.maha_std, s.score_std, 1)[0] if np.ptp(x) > 0 else np.nan
        rows.append(dict(seed=seed, family=g, n=len(s), spearman=spearmanr(x, y)[0], pearson=pearsonr(x, y)[0],
                         pendiente_estandarizada=slope, asimetria_maha=skew(x), asimetria_nll=skew(y),
                         nll_mediana=float(np.median(y)), nll_p99=float(np.percentile(y, 99))))
    t = pd.DataFrame(rows)
    agg = t.groupby("family").agg(n=("n", "mean"), spearman=("spearman", fisher_mean),
                                  pearson=("pearson", fisher_mean),
                                  pendiente_estandarizada=("pendiente_estandarizada", "mean"),
                                  asimetria_maha=("asimetria_maha", "mean"), asimetria_nll=("asimetria_nll", "mean"),
                                  nll_mediana=("nll_mediana", "mean"), nll_p99=("nll_p99", "mean")).reset_index()
    agg["pearson_menos_spearman"] = agg.pearson - agg.spearman
    return agg


def models(df):
    import statsmodels.api as sm
    out = []
    p5 = rcfg.METRICS_DIR / "phase5_mixed_model.json"
    if p5.exists():
        m = json.load(open(p5, encoding="utf-8"))
        out.append(dict(modelo="M1 fase 5: mixto, valores estandarizados, con benigno", efecto=m.get("fixed_effect_maha_std"),
                        ic_lo=m.get("fixed_effect_ci_low"), ic_hi=m.get("fixed_effect_ci_high"),
                        p=m.get("fixed_effect_p_value"), var_pendiente_familia=m.get("random_slope_variance_by_family"),
                        r2_marginal=m.get("r2_marginal"), n=m.get("n_obs"), convergio=m.get("converged")))
    fam = df[df.family != "BENIGN"].copy()
    variants = {}
    f2 = fam.copy()
    f2["maha_std"] = (f2.maha - f2.maha.mean()) / f2.maha.std()
    f2["score_std"] = (f2.score - f2.score.mean()) / f2.score.std()
    variants["M2: mixto, valores estandarizados, sin benigno"] = f2
    f3 = fam.copy()
    f3["maha_std"], f3["score_std"] = normal_scores(f3.maha), normal_scores(f3.score)
    variants["M3: mixto, puntuaciones normales de rango, sin benigno"] = f3
    for name, d in variants.items():
        m = fit_mixed_model(d.reset_index(drop=True))
        out.append(dict(modelo=name, efecto=m.get("fixed_effect_maha_std"), ic_lo=m.get("fixed_effect_ci_low"),
                        ic_hi=m.get("fixed_effect_ci_high"), p=m.get("fixed_effect_p_value"),
                        var_pendiente_familia=m.get("random_slope_variance_by_family"),
                        r2_marginal=m.get("r2_marginal"), n=m.get("n_obs"), convergio=m.get("converged")))
        print(f"[fase8] {name}: efecto={m.get('fixed_effect_maha_std')} p={m.get('fixed_effect_p_value')}")
    # efectos fijos con indicadoras explícitas (en una fórmula, C(...) chocaría con el alias de config)
    x = pd.get_dummies(f3[["family", "seed"]].astype(str), drop_first=True).astype(float)
    x.insert(0, "maha_std", f3.maha_std.to_numpy())
    ols = sm.OLS(f3.score_std.to_numpy(), sm.add_constant(x)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(f3.family)[0]})
    ci = ols.conf_int().loc["maha_std"].tolist()
    out.append(dict(modelo="M4: MCO con efectos fijos de familia y semilla (dentro de familia), rangos normales",
                    efecto=float(ols.params["maha_std"]), ic_lo=ci[0], ic_hi=ci[1], p=float(ols.pvalues["maha_std"]),
                    var_pendiente_familia=np.nan, r2_marginal=float(ols.rsquared), n=int(ols.nobs), convergio=True))
    return pd.DataFrame(out)


def main():
    seeds = C.SEEDS
    df = build_connection_level_data(seeds)
    print(f"[fase8] {len(df):,} conexiones, {df.family.nunique()} grupos, {df.seed.nunique()} semillas")

    rows = [dict(seed=s, **correlations_for_seed(df[df.seed == s])) for s in seeds]
    cor = pd.DataFrame(rows)
    cor.to_csv(C.TABLES_DIR / "reconciliacion_correlaciones_por_semilla.csv", index=False)
    cols = [c for c in cor.columns if c != "seed"]
    resumen = pd.DataFrame([dict(medida=c, media_fisher=fisher_mean(cor[c]), minimo=cor[c].min(), maximo=cor[c].max())
                            for c in cols])
    resumen.to_csv(C.TABLES_DIR / "reconciliacion_correlaciones.csv", index=False)

    fam = per_family(df)
    fam.to_csv(C.TABLES_DIR / "reconciliacion_por_familia.csv", index=False)

    mod = models(df)
    mod.to_csv(C.TABLES_DIR / "reconciliacion_modelos.csv", index=False)

    lines = ["# Reconciliación del nivel de conexión (fase 8)", "",
             "Misma muestra de conexiones que el modelo mixto de la fase 5 (hasta "
             f"{rcfg.MIXED_MODEL_MAX_N_PER_GROUP_PER_SEED} por grupo y semilla; {len(df):,} filas, "
             f"{len(seeds)} semillas). x = Mahalanobis latente (phase4, `latent_maha`); y = NLL Monte Carlo (`nll_mc`).", "",
             "## Correlaciones (media de Fisher entre semillas)", "",
             "De arriba abajo, del análisis de la fase 5 (Pearson sobre valores, con el benigno) al de la fase 7 "
             "(rangos centrados por familia).", "", md_table(resumen), "",
             "## Modelos", "", md_table(mod), "",
             "## Por familia (media entre semillas)", "",
             "`pearson_menos_spearman` grande = la relación lineal la dominan unos pocos valores extremos. "
             "`pendiente_estandarizada` = pendiente de MCO dentro de la familia en la escala que usa la fase 5.", "",
             md_table(fam.sort_values("spearman")), ""]
    (C.ROOT / "RESULTADOS_RECONCILIACION.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[fase8] resumen en {C.ROOT / 'RESULTADOS_RECONCILIACION.md'}")


if __name__ == "__main__":
    main()
