"""Fase 4 -- umbrales y agrupación por clique bajo cada criterio.

Criterios pedidos (sobre el dataset EDA, IM histograma B=20 con átomos):
  P      |r| >= 0.95, con Packet Length Variance descartada a priori
         (var = std^2, como en el EDA) -- DEBE reproducir 74 -> 51.
  L      r_info >= 0.95
  A_k    nmi_sqrt >= mu + k sigma (2,701 pares), k in {0.5, 1, 1.5, 2}
  S      nmi_max >= nmi_max de un par gaussiano |rho|=0.95 con el mismo
         estimador (fase 1, calibración)
Criterios adicionales (diagnóstico, rotulados como tales en el informe):
  P_sin_apriori  P sin la regla var = std^2 (qué haría Pearson solo)
  L_cal  r_info >= r_info de un par gaussiano |rho|=0.95 con el mismo
         estimador (corrige la pérdida por discretizar; 0.9414 con B=20)
  R      |rho| de Spearman >= 0.95 (dependencia monótona)
  L_ksg  r_info KSG >= 0.95 (submuestra 50k)
  G      gauss_ratio >= 1: I_corr(X;Y) >= IM de una cópula gaussiana rho=0.95
         con las MISMAS marginales discretizadas, mismo N y misma corrección
         de sesgo. Es el equivalente par a par de |r| >= 0.95: no tiene el
         techo de entropía de Linfoot (r_info no puede llegar a 0.95 si
         min(H) < 1.68 bits) ni el sesgo por discretizar de L.
  G_consenso  G con B=10, B=20 y B=50 a la vez (criterio para la
         recomendación: no depende de B).
Sensibilidad: L, L_cal, S y A_k con B=10, B=50 y qcut literal (B=20); P, L,
L_cal y S sobre train_benign.

Salidas:
  metrics/oe1_criterios.json
  tables/oe1_criterios_resumen.csv
  tables/oe1_grupos_por_criterio.csv
  tables/oe1_sensibilidad_umbrales.csv

Uso: python -m vae_nids.evaluation.oe1_mi.phase4_criteria
"""
import json

import numpy as np
import pandas as pd

from vae_nids.evaluation.oe1_mi import cliques as cq
from vae_nids.evaluation.oe1_mi import config as cfg

FEATS = cfg.FEATURES_74


def _edges(df: pd.DataFrame, col: str, thr: float, exclude: set[str] = frozenset()) -> set[frozenset[str]]:
    m = (df[col] >= thr) & ~df["feature_a"].isin(exclude) & ~df["feature_b"].isin(exclude)
    return {frozenset(p) for p in df.loc[m, ["feature_a", "feature_b"]].itertuples(index=False, name=None)}


ROBUST_PARTS = {
    # medida -> (columna, umbral); una arista "robusta" pasa TODAS
    "L_B10": ("r_info_B10", 0.95), "L_B20": ("r_info", 0.95), "L_B50": ("r_info_B50", 0.95),
    "L_ksg": ("ksg_r_info", 0.95), "G_B10": ("gauss_ratio_B10", 1.0), "G_B20": ("gauss_ratio", 1.0),
    "G_B50": ("gauss_ratio_B50", 1.0),
}


def add_robust_scores(df: pd.DataFrame) -> pd.DataFrame:
    """robusto_min >= 1  <=> el par pasa las 7 medidas IM (L con B=10/20/50,
    L con KSG, G con B=10/20/50). robusto_max < 1 <=> no pasa NINGUNA."""
    norm = pd.concat([df[c] / t for c, t in ROBUST_PARTS.values()], axis=1)
    df["robusto_min"] = norm.min(axis=1, skipna=False)
    df["robusto_max"] = norm.max(axis=1, skipna=False)
    df["gauss_ratio_min_B"] = df[["gauss_ratio", "gauss_ratio_B10", "gauss_ratio_B50"]].min(axis=1, skipna=False)
    return df


def u_lookup(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """u[(j, k)] = I(j; k) / H(j): fracción de la entropía de j explicada por k."""
    out = {}
    for a, b, uab, uba in df[["feature_a", "feature_b", "u_x_given_y", "u_y_given_x"]].itertuples(index=False):
        out[(a, b)] = uab
        out[(b, a)] = uba
    return out


def run_criterion(name: str, df: pd.DataFrame, col: str, thr: float, entropy: dict[str, float],
                  apriori: dict[str, str] | None = None, p_edges: set | None = None,
                  variant: str = "B20", dataset: str = "EDA",
                  u_given: dict | None = None) -> dict:
    apriori = apriori or {}
    nodes = [f for f in FEATS if f not in apriori]
    edges = _edges(df, col, thr, exclude=set(apriori))
    res = cq.resolve_groups(nodes, edges)
    kept51 = set(cfg.load_features_51())
    doc_rep = {frozenset(ms): rep for rep, ms in cfg.DOCUMENTED_GROUPS.items()}
    groups = []
    dropped = list(apriori)
    for g in res["groups"]:
        rep = cq.choose_representative(g, kept51, entropy, doc_rep, u_given)
        drops = [f for f in g if f != rep]
        dropped += drops
        groups.append({"members": g, "rep": rep, "dropped": drops})
    final = [f for f in FEATS if f not in set(dropped)]
    out = {
        "criterion": name, "variant": variant, "dataset": dataset, "column": col,
        "threshold": float(thr), "n_edges": len(edges), "n_groups": len(groups),
        "n_dropped": len(dropped), "final_size": len(final),
        "groups": groups, "dropped": sorted(dropped), "final_features": final,
        "ambiguous": res["ambiguous"], "log": res["log"],
        "apriori_drops": apriori,
        "added_vs_51": sorted(set(final) - kept51),
        "removed_vs_51": sorted(kept51 - set(final)),
    }
    if p_edges is not None:
        out["edges_shared_with_P"] = len(edges & p_edges)
        out["edges_only_here"] = sorted(sorted(e) for e in edges - p_edges)
        out["edges_only_P"] = sorted(sorted(e) for e in p_edges - edges)
    return out


def check_p_reproduces(p: dict) -> dict:
    got = {frozenset(g["members"]) for g in p["groups"]}
    exp = {frozenset(ms) for ms in cfg.DOCUMENTED_GROUPS.values()}
    reps_ok = all(g["rep"] == dict((frozenset(v), k) for k, v in cfg.DOCUMENTED_GROUPS.items())
                  .get(frozenset(g["members"])) for g in p["groups"])
    final_ok = p["final_features"] == cfg.load_features_51()
    return {"groups_equal": got == exp, "reps_equal": reps_ok, "final_equals_51": final_ok,
            "missing": [sorted(s) for s in exp - got], "extra": [sorted(s) for s in got - exp]}


def main() -> dict:
    df = pd.read_csv(cfg.METRICS_DIR / "oe1_pares_dependencia.csv")
    dfb = pd.read_csv(cfg.METRICS_DIR / "oe1_pares_dependencia_train_benign.csv")
    ent_df = pd.read_csv(cfg.METRICS_DIR / "oe1_entropia_features.csv")
    entropy = dict(zip(ent_df["feature"], ent_df["H_bits"]))
    calib = json.load(open(cfg.METRICS_DIR / "oe1_validacion_estimador.json",
                           encoding="utf-8"))["calibracion_umbral_095"]
    calib["qcut20"] = calib["B20"]  # sin átomos en la gaussiana: misma calibración

    df = add_robust_scores(df)
    U = u_lookup(df)
    _run = globals()["run_criterion"]

    def run_criterion(*a, **k):  # noqa: F811 -- mismo u_given en todos los criterios del EDA
        k.setdefault("u_given", U)
        return _run(*a, **k)

    results: list[dict] = []
    p = run_criterion("P", df, "pearson_abs", cfg.PEARSON_THRESHOLD, entropy, apriori=cfg.A_PRIORI_DROPS)
    repro = check_p_reproduces(p)
    print("[P] reproducción:", repro)
    if not (repro["groups_equal"] and repro["final_equals_51"]):
        with open(cfg.METRICS_DIR / "oe1_criterios.json", "w", encoding="utf-8") as f:
            json.dump({"P": p, "P_reproduction": repro}, f, indent=2, ensure_ascii=False)
        raise SystemExit("P NO reproduce 74 -> 51: se detiene la fase 4 (ver oe1_criterios.json)")
    p_edges = _edges(df, "pearson_abs", cfg.PEARSON_THRESHOLD, exclude=set(cfg.A_PRIORI_DROPS))
    results.append(p)
    results.append(run_criterion("P_sin_apriori", df, "pearson_abs", cfg.PEARSON_THRESHOLD, entropy,
                                 p_edges=p_edges))

    def suffix(v):
        return "" if v == "B20" else f"_{v}"

    adaptive = {}
    for v in ("B20", "B10", "B50", "qcut20"):
        s = suffix(v)
        results.append(run_criterion("L", df, f"r_info{s}", cfg.RINFO_THRESHOLD, entropy,
                                     p_edges=p_edges, variant=v))
        results.append(run_criterion("L_cal", df, f"r_info{s}", calib[v]["r_info_gauss_095"], entropy,
                                     p_edges=p_edges, variant=v))
        results.append(run_criterion("S", df, f"nmi_max{s}", calib[v]["nmi_max_gauss_095"], entropy,
                                     p_edges=p_edges, variant=v))
        results.append(run_criterion("G", df, f"gauss_ratio{s}", 1.0, entropy, p_edges=p_edges, variant=v))
        mu, sd = float(df[f"nmi_sqrt{s}"].mean()), float(df[f"nmi_sqrt{s}"].std(ddof=0))
        adaptive[v] = {"mu": mu, "sigma": sd}
        for k in cfg.ADAPTIVE_K:
            results.append(run_criterion(f"A_{k:g}", df, f"nmi_sqrt{s}", mu + k * sd, entropy,
                                         p_edges=p_edges, variant=v))
    df["gauss_ratio_min_B"] = df[["gauss_ratio", "gauss_ratio_B10", "gauss_ratio_B50"]].min(axis=1, skipna=False)
    g_cons = run_criterion("G_consenso", df, "gauss_ratio_min_B", 1.0, entropy, p_edges=p_edges,
                           variant="B10&B20&B50")
    results.append(g_cons)
    results.append(run_criterion("R", df, "spearman_abs", 0.95, entropy, p_edges=p_edges))
    results.append(run_criterion("Robusto", df, "robusto_min", 1.0, entropy, p_edges=p_edges,
                                 variant="7 medidas IM"))

    # Recomendado: parte de P (las 23 descartadas quedan fuera, con su regla a
    # priori), restaura toda descartada cuya fusión con su representante NO
    # pasa ninguna de las 7 medidas IM (robusto_max < 1) y aplica la
    # resolución por clique a las aristas robustas entre las features restantes.
    grp = {m: rep for rep, ms in cfg.DOCUMENTED_GROUPS.items() for m in ms if m != rep}
    grp.update(cfg.A_PRIORI_DROPS)
    rob_max = {frozenset((a, b)): v for a, b, v in df[["feature_a", "feature_b", "robusto_max"]]
               .itertuples(index=False)}
    restored = sorted(d for d, rep in grp.items() if rob_max[frozenset((d, rep))] < 1)
    rec = run_criterion("Recomendado", df, "robusto_min", 1.0, entropy, p_edges=p_edges,
                        apriori={d: r for d, r in grp.items() if d not in restored},
                        variant="P + restauradas + robustas")
    rec["restored"] = restored
    results.append(rec)
    if "ksg_r_info" in df:
        results.append(run_criterion("L_ksg", df, "ksg_r_info", cfg.RINFO_THRESHOLD, entropy,
                                     p_edges=p_edges, variant="KSG50k"))

    # train_benign
    pb = run_criterion("P", dfb, "pearson_abs", cfg.PEARSON_THRESHOLD, entropy,
                       apriori=cfg.A_PRIORI_DROPS, p_edges=p_edges, dataset="train_benign")
    results.append(pb)
    results.append(run_criterion("L", dfb, "r_info", cfg.RINFO_THRESHOLD, entropy, p_edges=p_edges,
                                 dataset="train_benign"))
    results.append(run_criterion("L_cal", dfb, "r_info", calib["B20"]["r_info_gauss_095"], entropy,
                                 p_edges=p_edges, dataset="train_benign"))
    results.append(run_criterion("S", dfb, "nmi_max", calib["B20"]["nmi_max_gauss_095"], entropy,
                                 p_edges=p_edges, dataset="train_benign"))
    results.append(run_criterion("G", dfb, "gauss_ratio", 1.0, entropy, p_edges=p_edges,
                                 dataset="train_benign"))

    # Sensibilidad: pares que cambian de lado respecto de B20
    sens_rows = []
    for crit, colbase, thr_of in (
        ("L", "r_info", lambda v: cfg.RINFO_THRESHOLD),
        ("L_cal", "r_info", lambda v: calib[v]["r_info_gauss_095"]),
        ("S", "nmi_max", lambda v: calib[v]["nmi_max_gauss_095"]),
        ("G", "gauss_ratio", lambda v: 1.0),
        *[(f"A_{k:g}", "nmi_sqrt", (lambda kk: lambda v: adaptive[v]["mu"] + kk * adaptive[v]["sigma"])(k))
          for k in cfg.ADAPTIVE_K],
    ):
        base = df[colbase] >= thr_of("B20")
        for v in ("B10", "B50", "qcut20"):
            other = df[f"{colbase}_{v}"] >= thr_of(v)
            sens_rows.append({"criterio": crit, "variante": v, "umbral_B20": thr_of("B20"),
                              "umbral_variante": thr_of(v), "aristas_B20": int(base.sum()),
                              "aristas_variante": int(other.sum()),
                              "entran": int((other & ~base).sum()), "salen": int((base & ~other).sum())})
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(cfg.TABLES_DIR / "oe1_sensibilidad_umbrales.csv", index=False)

    summary = pd.DataFrame([{k: r[k] for k in ("criterion", "variant", "dataset", "column", "threshold",
                                                "n_edges", "n_groups", "n_dropped", "final_size")}
                            | {"edges_shared_with_P": r.get("edges_shared_with_P"),
                               "n_ambiguous": len(r["ambiguous"]),
                               "anadidas_vs_51": "; ".join(r["added_vs_51"]),
                               "quitadas_vs_51": "; ".join(r["removed_vs_51"])}
                            for r in results])
    summary.to_csv(cfg.TABLES_DIR / "oe1_criterios_resumen.csv", index=False)
    grows = [{"criterion": r["criterion"], "variant": r["variant"], "dataset": r["dataset"],
              "grupo": gi, "representante": g["rep"], "n": len(g["members"]),
              "miembros": "; ".join(g["members"]), "descartadas": "; ".join(g["dropped"])}
             for r in results for gi, g in enumerate(r["groups"])]
    pd.DataFrame(grows).to_csv(cfg.TABLES_DIR / "oe1_grupos_por_criterio.csv", index=False)

    out = {"P_reproduction": repro, "adaptive_mu_sigma": adaptive, "calibration": calib,
           "results": results}
    with open(cfg.METRICS_DIR / "oe1_criterios.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(summary.drop(columns=["anadidas_vs_51", "quitadas_vs_51"]).to_string(index=False))
        print(sens.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
