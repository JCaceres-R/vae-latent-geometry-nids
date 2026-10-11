"""Fase 3 -- valoración de descriptores frente a la detectabilidad.

Diseño en dos tiempos (declarado antes de ver resultados):
1. CRIBADO en las 11 familias principales (MAIN_FAMILIES), 10 semillas.
   Por descriptor d y semilla s: rho_s = Spearman(d_s, AUC_s) sobre 11 familias.
   - rho combinado: tanh(media de arctanh(rho_s)).
   - IC 95 %: bootstrap de familias (las mismas familias en todas las semillas).
   - p: permutación de las etiquetas de familia del AUC (la misma en todas las
     semillas), estadístico |rho combinado|. Holm sobre los 10 secundarios
     declarados; Benjamini-Hochberg sobre el resto.
   - estabilidad: W de Kendall del ranking de familias entre semillas, y
     fracción de semillas con el signo del rho combinado.
   - redundancia: rho con la mediana de Mahalanobis (primario) y rho parcial
     con el AUC controlando por ella.
   - robustez: rho combinado quitando una familia cada vez (mínimo y máximo).
   - nivel de flujo: Spearman descriptor-NLL dentro de cada familia (media).
2. VALIDACIÓN en los 6 grupos "attempted" (no participan en el cribado):
   rho combinado sobre esos 6 grupos y concordancia de signo en los 6 pares
   completado/attempted. Con n=6 la potencia es baja: es una comprobación de
   dirección, no una prueba.

Uso:
    python -m vae_nids.evaluation.oe2_descriptores.fase3_evaluacion
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from vae_nids.evaluation.oe2_descriptores import config as C
from vae_nids.evaluation.oe2_robust import config as rcfg

ATTEMPTED_ONLY = [b for _, b in C.ATTEMPTED_PAIRS if b.endswith("_attempted")]

# id, columna, nombre, nivel (P/S/E/R), orientación ("dir" = mayor es más atípico; "sep" = sin dirección)
CATALOGO = [
    ("1.1", "maha__median", "Mahalanobis (mediana)", "P", "dir"),
    ("5.1", "knn__auc", "kNN al benigno (z blanqueado)", "S", "dir"),
    ("2.4", "fr__auc", "Fisher-Rao al k-ésimo posterior benigno", "S", "dir"),
    ("7.1", "rate__sep", "Tasa KL(q||p) por flujo", "S", "sep"),
    ("3.2", "mmd_agg__value", "MMD² analítico entre posteriores agregados", "S", "dir"),
    ("6.2", "dec_knn__auc", "kNN en el espacio decodificado", "S", "dir"),
    ("6.1", "logmag__sep", "Log-magnificación del decodificador", "S", "sep"),
    ("4.2", "airm__value", "Distancia afín-invariante entre covarianzas", "S", "dir"),
    ("8.1", "lid__sep", "LID relativa al benigno", "S", "sep"),
    ("10.1a", "sil_w__value", "Silhouette del grupo (z blanqueado)", "S", "dir"),
    ("10.3", "probe__value", "Sonda lineal sobre (μ, log σ²)", "S", "dir"),
    ("1.1b", "maha__auc", "Mahalanobis (AUC frente al control)", "E", "dir"),
    ("10.1b", "sil_fr__value", "Silhouette del grupo (Fisher-Rao)", "E", "dir"),
    ("1.2", "gmm__auc", "Log-densidad de mezcla benigna (K=8)", "E", "dir"),
    ("1.3", "rmd__auc", "Mahalanobis relativa al prior", "E", "dir"),
    ("1.4", "typ__auc", "Tipicidad radial bilateral", "E", "dir"),
    ("4.1", "erank__value", "Rango efectivo de la familia", "E", "sep"),
    ("7.2", "logvol__sep", "Log-volumen del posterior", "E", "sep"),
    ("11.1", "kp__sep", "Inconsistencia Kumar-Poole", "E", "sep"),
    ("5.1m", "knn__frac_p95", "kNN: fracción > p95 del control", "E", "dir"),
    ("2.4m", "fr__median", "Fisher-Rao (mediana)", "E", "dir"),
    ("7.1m", "rate__median", "Tasa (mediana, nats)", "E", "sep"),
    ("6.1m", "logmag__median", "Log-magnificación (mediana)", "E", "sep"),
]

REPO_METRICS = {
    "phase3_mahalanobis_summary": ["mahalanobis_median_robust_mcd", "mahalanobis_p95"],
    "phase3_divergences": ["mmd2_rbf", "sliced_wasserstein", "energy_distance"],
    "phase3_bhattacharyya": ["bhattacharyya_total", "bhattacharyya_mean_term", "bhattacharyya_cov_term"],
    "phase3_neighborhood": ["neighborhood_purity_median", "log_density_benign_median"],
    "phase3_dispersion": ["silhouette_mean", "cov_logdet"],
    "phase3_posterior_mahalanobis": ["posterior_mahalanobis_median"],
    "phase3_posterior_uncertainty": ["mean_posterior_variance_sum"],
    "phase3_cliffs_delta": ["cliffs_delta"],
}


def fisher_mean(r):
    r = np.clip(np.asarray(r, float), -0.999, 0.999)
    return float(np.tanh(np.nanmean(np.arctanh(r))))


def kendall_w(m):
    """W de Kendall para una matriz (jueces=semillas x objetos=familias)."""
    m = m[~np.isnan(m).any(1)]
    k, n = m.shape
    if k < 2:
        return np.nan
    r = np.apply_along_axis(rankdata, 1, m)
    s = ((r.sum(0) - k * (n + 1) / 2) ** 2).sum()
    return float(12 * s / (k ** 2 * (n ** 3 - n)))


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy, rxz, ryz = (np.corrcoef(a, b)[0, 1] for a, b in ((rx, ry), (rx, rz), (ry, rz)))
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return float((rxy - rxz * ryz) / den) if den > 0 else np.nan


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(run, 1.0)
    return adj


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)[::-1]
    adj = np.empty_like(p)
    run = 1.0
    for i, j in enumerate(o):
        run = min(run, p[j] * n / (n - i))
        adj[j] = run
    return adj


def load_tables():
    det = pd.read_csv(C.TABLES_DIR / "detectabilidad_por_semilla.csv")
    des = pd.read_csv(C.TABLES_DIR / "descriptores_por_semilla.csv")
    for col in [c for c in des.columns if c.endswith("__auc")]:
        des[col.replace("__auc", "__sep")] = 2 * (des[col] - 0.5).abs()
    # descriptores ya calculados por el pipeline robusto (phase3), para compararlos con la misma vara
    # (el OE4 no corre la fase 3 robusta: si sus tablas no están, se valoran solo los descriptores propios)
    rep = []
    for tab, metrics in REPO_METRICS.items():
        if not (rcfg.TABLES_DIR / f"{tab}.csv").exists():
            continue
        t = pd.read_csv(rcfg.TABLES_DIR / f"{tab}.csv")
        t = t[t.metric.isin(metrics)][["seed", "group", "metric", "value"]]
        rep.append(t)
    if not rep:
        return det, des
    rep = pd.concat(rep).pivot_table(index=["seed", "group"], columns="metric", values="value").reset_index()
    rep.columns = ["seed", "group"] + [f"repo:{c}" for c in rep.columns[2:]]
    des = des.merge(rep, on=["seed", "group"], how="left")
    return det, des


def matrix(des, det, col, groups):
    """(semillas x grupos) del descriptor y del AUC."""
    d = des[des.group.isin(groups)].pivot(index="seed", columns="group", values=col)[groups]
    a = det[det.group.isin(groups)].pivot(index="seed", columns="group", values="auc_nll_mc")[groups]
    return d.loc[C.SEEDS].to_numpy(float), a.loc[C.SEEDS].to_numpy(float)


def _row_pearson(x, y):
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    den = np.sqrt((x ** 2).sum(-1) * (y ** 2).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (x * y).sum(-1) / den, np.nan)


def combined_rho(dm, am):
    """Spearman por semilla (fila) y su media de Fisher. Las semillas con algún NaN
    se calculan quitando esas familias."""
    rs = np.full(len(dm), np.nan)
    ok = np.isfinite(dm).all(1)
    if ok.any():
        rs[ok] = _row_pearson(rankdata(dm[ok], axis=1), rankdata(am[ok], axis=1))
    for i in np.where(~ok)[0]:
        m = np.isfinite(dm[i])
        if m.sum() > 3:
            rs[i] = spearmanr(dm[i, m], am[i, m])[0]
    return fisher_mean(rs), rs


def permutation_rhos(dm, am, perms):
    """|rho combinado| para cada permutación de familias (vectorizado)."""
    ok = np.isfinite(dm).all(1)
    rd = rankdata(dm[ok], axis=1)[:, None, :]
    ra = rankdata(am[ok], axis=1)[:, perms]                    # (semillas, nperm, familias)
    r = np.clip(_row_pearson(np.broadcast_to(rd, ra.shape), ra), -0.999, 0.999)
    return np.abs(np.tanh(np.nanmean(np.arctanh(r), axis=0)))


def evaluate(des, det, col, maha_col="maha__median", rng=None):
    fam = C.MAIN_FAMILIES
    dm, am = matrix(des, det, col, fam)
    mm, _ = matrix(des, det, maha_col, fam)
    rho, rs = combined_rho(dm, am)
    nf = len(fam)
    boot = []
    for _ in range(C.N_BOOT):
        idx = rng.integers(0, nf, nf)
        if len(np.unique(idx)) < 4:
            continue
        boot.append(combined_rho(dm[:, idx], am[:, idx])[0])
    perms = np.array([rng.permutation(nf) for _ in range(C.N_PERM)])
    perm = permutation_rhos(dm, am, perms)
    p = (1 + (perm >= abs(rho) - 1e-12).sum()) / (1 + C.N_PERM)
    lofo = [combined_rho(np.delete(dm, i, 1), np.delete(am, i, 1))[0] for i in range(nf)]
    red = fisher_mean([spearmanr(d, m)[0] for d, m in zip(dm, mm)])
    par = fisher_mean([partial_spearman(d, a, m) for d, a, m in zip(dm, am, mm)])
    sign = float(np.nanmean(np.sign(rs) == np.sign(rho)))
    # ¿el descriptor ordena también a las 9 familias de AUC alto? (sin PortScan ni Bot)
    keep = [i for i, g in enumerate(fam) if g not in ("PortScan", "Bot")]
    rho9, _ = combined_rho(dm[:, keep], am[:, keep])
    perm9 = permutation_rhos(dm[:, keep], am[:, keep], np.array([rng.permutation(len(keep)) for _ in range(C.N_PERM)]))
    p9 = (1 + (perm9 >= abs(rho9) - 1e-12).sum()) / (1 + C.N_PERM)
    # validación en grupos attempted
    dv, av = matrix(des, det, col, ATTEMPTED_ONLY)
    rho_val, _ = combined_rho(dv, av)
    conc = []
    for a, b in C.ATTEMPTED_PAIRS:
        if not b.endswith("_attempted"):
            continue
        da, aa = matrix(des, det, col, [a, b])
        conc.extend(np.sign(da[:, 0] - da[:, 1]) == np.sign(aa[:, 0] - aa[:, 1]))
    flow_col = col.split("__")[0] + "__rho_flow_nll"
    flow = (des[des.group.isin(fam)][flow_col].mean() if flow_col in des else np.nan)
    return dict(rho=rho, ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)),
                p_perm=float(p), rho_seed_min=float(np.nanmin(rs)), rho_seed_max=float(np.nanmax(rs)),
                sign_consistency=sign, kendall_w=kendall_w(dm), rho_with_maha=red, partial_rho=par,
                lofo_min=float(np.min(lofo)), lofo_max=float(np.max(lofo)),
                lofo_drop_at_min=fam[int(np.argmin(lofo))], lofo_drop_at_max=fam[int(np.argmax(lofo))],
                rho_sin_portscan_bot=rho9, p_sin_portscan_bot=float(p9),
                rho_attempted=rho_val, pair_concordance=float(np.mean(conc)) if conc else np.nan,
                flow_rho_nll_mean=float(flow) if flow == flow else np.nan)


def tier(r, level):
    if r["kendall_w"] < 0.5:
        return "D: inestable entre semillas"
    if r.get("p_holm", 1) < 0.05 and r["sign_consistency"] >= 0.9:
        return "A: asociación robusta"
    if (r["p_perm"] < 0.05 or abs(r["rho"]) >= 0.5) and r["sign_consistency"] >= 0.8:
        return "B: asociación prometedora"
    return "C: estable, sin asociación clara"


def main():
    rng = np.random.default_rng(C.RNG_SEED)
    det, des = load_tables()
    rows = []
    cat = list(CATALOGO) + [("R", c, c.replace("repo:", "").replace("_", " "), "R", "dir")
                            for c in des.columns if c.startswith("repo:")]
    for did, col, name, level, orient in cat:
        r = evaluate(des, det, col, rng=rng)
        r.update(id=did, columna=col, descriptor=name, nivel=level, orientacion=orient)
        rows.append(r)
        print(f"{did:8s} {name:48s} rho={r['rho']:+.2f} [{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] "
              f"p={r['p_perm']:.3f} W={r['kendall_w']:.2f} rhoM={r['rho_with_maha']:+.2f} "
              f"val={r['rho_attempted']:+.2f}")
    df = pd.DataFrame(rows)
    sec = df.nivel == "S"
    df.loc[sec, "p_holm"] = holm(df.loc[sec, "p_perm"])
    oth = ~sec & (df.nivel != "P")
    df.loc[oth, "p_bh"] = bh(df.loc[oth, "p_perm"])
    df["tier"] = [tier(r, r["nivel"]) for r in df.to_dict("records")]
    df.to_csv(C.TABLES_DIR / "valoracion_descriptores.csv", index=False)

    # matriz de redundancia entre descriptores (media de Fisher entre semillas, 11 familias)
    cols = [c for _, c, _, lv, _ in CATALOGO if lv in ("P", "S")]
    red = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            da, _ = matrix(des, det, a, C.MAIN_FAMILIES)
            db, _ = matrix(des, det, b, C.MAIN_FAMILIES)
            red.loc[a, b] = fisher_mean([spearmanr(x, y)[0] for x, y in zip(da, db)])
    red.to_csv(C.TABLES_DIR / "redundancia_descriptores.csv")

    # resumen por familia: media entre semillas (para figuras y tablas del informe)
    fam_rows = []
    for g in C.MAIN_FAMILIES + ATTEMPTED_ONLY:
        sub = des[des.group == g]
        a = det[det.group == g]
        row = dict(group=g, auc_mean=a.auc_nll_mc.mean(), auc_sd=a.auc_nll_mc.std(),
                   n=int(sub.n.iloc[0]), n_unique_mean=sub.n_unique.mean(), n_eff_mean=sub.n_eff.mean(),
                   median_D=a.median_D.mean(), median_R=a.median_R.mean())
        for _, col, _, _, _ in CATALOGO:
            row[col] = sub[col].mean()
            row[col + "_sd"] = sub[col].std()
        fam_rows.append(row)
    pd.DataFrame(fam_rows).to_csv(C.TABLES_DIR / "resumen_por_familia.csv", index=False)

    # nivel de flujo por familia (media entre semillas)
    fl = [c for c in des.columns if c.endswith("__rho_flow_nll")]
    des[des.group.isin(C.MAIN_FAMILIES)].groupby("group")[fl].mean().to_csv(C.TABLES_DIR / "nivel_flujo_por_familia.csv")
    json.dump({"n_familias": len(C.MAIN_FAMILIES), "n_semillas": len(C.SEEDS), "n_boot": C.N_BOOT,
               "n_perm": C.N_PERM, "grupos_validacion": ATTEMPTED_ONLY},
              open(C.TABLES_DIR / "valoracion_meta.json", "w"), indent=2)
    return df


if __name__ == "__main__":
    main()
