"""Fase 5 -- contraste Pearson vs. IM y tablas para las preguntas 1-5.

Salidas:
  tables/oe1_cuadrantes.csv            pares de los cuadrantes fuera de la diagonal (|r| vs r_info B=20)
  tables/oe1_cuadrantes_conteo.csv     conteos de los 4 cuadrantes (r_info B=20; también KSG y G como extra)
  tables/oe1_q1_clusters.csv           los 13 clusters bajo L y bajo G_consenso
  tables/oe1_q2_cluster1.csv           los 15 pares del cluster 1
  tables/oe1_q3_pares_51.csv           pares entre las 51 conservadas marcados por alguna medida IM
  tables/oe1_q4_fusiones_P.csv         pares fusionados por P (+ Variance-Std) con todas las medidas
  tables/oe1_informacion_retenida.csv  por feature descartada: cuánto de su entropía explica la mejor conservada
  metrics/oe1_contraste.json           resumen + acuerdo KSG vs. histograma (pregunta 5)

Uso: python -m vae_nids.evaluation.oe1_mi.phase5_contrast
"""
import json

import numpy as np
import pandas as pd

from vae_nids.evaluation.oe1_mi import config as cfg
from vae_nids.evaluation.oe1_mi.phase4_criteria import add_robust_scores

MEASURES = ["pearson_abs", "spearman_abs", "r_info", "r_info_B10", "r_info_B50", "ksg_r_info",
            "gauss_ratio", "gauss_ratio_B10", "gauss_ratio_B50", "nmi_sqrt", "nmi_max",
            "u_x_given_y", "u_y_given_x", "H_x", "H_y", "mi_corr_bits", "mi_ref_gauss_bits",
            "g_stat", "g_dof", "g_pvalue"]


def FEATS_OUT(crit: dict) -> list[str]:
    keep = set(crit["final_features"])
    return [f for f in cfg.FEATURES_74 if f not in keep]


def quadrant(r_abs: pd.Series, dep: pd.Series, thr_x: float, thr_y: float) -> pd.Series:
    hi_r, hi_i = r_abs >= thr_x, dep >= thr_y
    return pd.Series(np.select(
        [hi_r & hi_i, hi_r & ~hi_i, ~hi_r & hi_i],
        ["redundancia_confirmada", "pearson_sobreestimo", "redundancia_no_lineal_no_detectada"],
        default="independientes_o_debiles"), index=r_abs.index)


def pair_lookup(df: pd.DataFrame) -> dict[frozenset, pd.Series]:
    return {frozenset((a, b)): row for a, b, row in
            zip(df["feature_a"], df["feature_b"], (r for _, r in df.iterrows()))}


def u_given(row: pd.Series, target: str) -> float:
    """Fracción de H(target) explicada por la otra feature del par."""
    return row["u_x_given_y"] if row["feature_a"] == target else row["u_y_given_x"]


def cluster_table(df: pd.DataFrame, crit: dict[str, dict]) -> pd.DataFrame:
    lk = pair_lookup(df)
    rows = []
    for cid, members in cfg.EDA_CLUSTERS.items():
        mset = set(members)
        p_groups = [ms for ms in cfg.DOCUMENTED_GROUPS.values() if set(ms) <= mset]
        row = {"cluster": cid, "miembros": "; ".join(members),
               "grupos_P": " | ".join("{" + ", ".join(g) + "}" for g in p_groups) or "(ninguno)"}
        for name, c in crit.items():
            col, thr = c["column"], c["threshold"]
            descr = []
            for g in p_groups:
                vals = [lk[frozenset(p)][col] for p in
                        ((a, b) for i, a in enumerate(g) for b in g[i + 1:])]
                n_ok = sum(v >= thr for v in vals if v == v)
                estado = "sigue siendo clique" if n_ok == len(vals) else f"se parte ({n_ok}/{len(vals)} pares pasan)"
                descr.append(f"{{{', '.join(g)}}}: {estado}, min={np.nanmin(vals):.4f}")
            touching = [gr for gr in c["groups"] if set(gr["members"]) & mset]
            ext = sorted({f for gr in touching for f in gr["members"] if f not in mset})
            row[f"{name}_grupos_P"] = " | ".join(descr) or "(P no fusiona)"
            row[f"{name}_grupos_resueltos"] = " | ".join(
                "{" + ", ".join(gr["members"]) + "} -> " + gr["rep"] for gr in touching) or "(ninguno)"
            row[f"{name}_se_amplia_con"] = "; ".join(ext)
            row[f"{name}_n_conservadas_del_cluster"] = len([f for f in members if f in set(c["final_features"])])
        row["P_n_conservadas_del_cluster"] = len([f for f in members if f in set(cfg.load_features_51())])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> dict:
    df = pd.read_csv(cfg.METRICS_DIR / "oe1_pares_dependencia.csv")
    df = add_robust_scores(df)
    crit_all = json.load(open(cfg.METRICS_DIR / "oe1_criterios.json", encoding="utf-8"))
    by = {(r["criterion"], r["variant"], r["dataset"]): r for r in crit_all["results"]}
    L, G = by[("L", "B20", "EDA")], by[("G_consenso", "B10&B20&B50", "EDA")]
    Lcal = by[("L_cal", "B20", "EDA")]
    kept = set(cfg.load_features_51())
    out: dict = {}

    # --- Cuadrantes ---
    df["cuadrante"] = quadrant(df["pearson_abs"], df["r_info"], 0.95, 0.95)
    counts = []
    for name, dep, thr in (("r_info_hist_B20", df["r_info"], 0.95),
                           ("r_info_hist_B20_Lcal", df["r_info"], Lcal["threshold"]),
                           ("r_info_KSG", df["ksg_r_info"], 0.95),
                           ("G_B20_gauss_ratio", df["gauss_ratio"], 1.0),
                           ("G_consenso", df["gauss_ratio_min_B"], 1.0)):
        q = quadrant(df["pearson_abs"], dep.fillna(-1), 0.95, thr).value_counts()
        counts.append({"medida_eje_y": name, "umbral_y": thr, **{k: int(q.get(k, 0)) for k in (
            "redundancia_confirmada", "pearson_sobreestimo", "redundancia_no_lineal_no_detectada",
            "independientes_o_debiles")}})
    counts = pd.DataFrame(counts)
    counts.to_csv(cfg.TABLES_DIR / "oe1_cuadrantes_conteo.csv", index=False)
    off = df[df["cuadrante"].isin(["pearson_sobreestimo", "redundancia_no_lineal_no_detectada"])]
    off = off.sort_values(["cuadrante", "r_info"], ascending=[True, False])
    off[["cuadrante", "feature_a", "feature_b", "mismo_cluster_eda", "fusionado_en_P", "ambas_en_51"]
        + MEASURES].to_csv(cfg.TABLES_DIR / "oe1_cuadrantes.csv", index=False)
    out["cuadrantes"] = counts.to_dict(orient="records")

    # --- Q1 ---
    q1 = cluster_table(df, {"L": L, "Gcons": G})
    q1.to_csv(cfg.TABLES_DIR / "oe1_q1_clusters.csv", index=False)

    # --- Q2: cluster 1 ---
    c1 = cfg.EDA_CLUSTERS[1]
    q2 = df[df["feature_a"].isin(c1) & df["feature_b"].isin(c1)]
    q2[["feature_a", "feature_b"] + MEASURES].to_csv(cfg.TABLES_DIR / "oe1_q2_cluster1.csv", index=False)
    out["q2_cluster1"] = {
        "n_pares": len(q2),
        "pearson_min": float(q2["pearson_abs"].min()),
        "r_info_min": float(q2["r_info"].min()), "r_info_max": float(q2["r_info"].max()),
        "n_pares_L": int((q2["r_info"] >= 0.95).sum()),
        "n_pares_G_consenso": int((q2[["gauss_ratio", "gauss_ratio_B10", "gauss_ratio_B50"]].min(axis=1) >= 1).sum()),
        "ksg_r_info_min": float(q2["ksg_r_info"].min()),
        "u_min": float(q2[["u_x_given_y", "u_y_given_x"]].min(axis=1).min()),
        "u_max": float(q2[["u_x_given_y", "u_y_given_x"]].max(axis=1).max()),
        "nmi_max_min": float(q2["nmi_max"].min()), "nmi_max_max": float(q2["nmi_max"].max()),
    }

    # --- Q3: redundancias entre las 51 ---
    gmin = df["gauss_ratio_min_B"]
    both = df["ambas_en_51"]
    flags = pd.DataFrame({
        "L_B10": df["r_info_B10"] >= 0.95, "L_B20": df["r_info"] >= 0.95, "L_B50": df["r_info_B50"] >= 0.95,
        "L_ksg": df["ksg_r_info"] >= 0.95, "G_B10": df["gauss_ratio_B10"] >= 1, "G_B20": df["gauss_ratio"] >= 1,
        "G_B50": df["gauss_ratio_B50"] >= 1,
    })
    q3 = pd.concat([df, flags], axis=1)[both & flags.any(axis=1)].copy()
    q3["n_medidas_de_7"] = flags[both & flags.any(axis=1)].sum(axis=1)
    q3 = q3.sort_values(["n_medidas_de_7", "r_info"], ascending=False)
    dfb_ = pd.read_csv(cfg.METRICS_DIR / "oe1_pares_dependencia_train_benign.csv")
    q3 = q3.merge(dfb_[["feature_a", "feature_b", "pearson_abs", "r_info", "gauss_ratio"]].rename(
        columns={"pearson_abs": "benign_pearson", "r_info": "benign_r_info", "gauss_ratio": "benign_gauss_ratio"}),
        on=["feature_a", "feature_b"], how="left")
    q3[["feature_a", "feature_b", "n_medidas_de_7", *flags.columns] + MEASURES
       + ["benign_pearson", "benign_r_info", "benign_gauss_ratio"]].to_csv(
        cfg.TABLES_DIR / "oe1_q3_pares_51.csv", index=False)
    out["q3"] = {
        "n_pares_51": int(both.sum()),
        "n_L_B20": int((both & flags["L_B20"]).sum()),
        "n_L_todas_B_y_KSG": int((both & flags[["L_B10", "L_B20", "L_B50", "L_ksg"]].all(axis=1)).sum()),
        "n_G_consenso": int((both & (gmin >= 1)).sum()),
        "n_G_consenso_y_KSG": int((both & (gmin >= 1) & flags["L_ksg"]).sum()),
        "n_alguna_medida": int(len(q3)),
        "robustos_7_de_7_benign_L_y_G": int(((q3["n_medidas_de_7"] == 7) & (q3["benign_r_info"] >= 0.95)
                                              & (q3["benign_gauss_ratio"] >= 1)).sum()),
    }

    # --- Q4: fusiones de P ---
    grp = {m: rep for rep, ms in cfg.DOCUMENTED_GROUPS.items() for m in ms}
    q4 = pd.concat([df, flags], axis=1)[df["fusionado_en_P"] | (
        (df["feature_a"] == "Packet Length Std") & (df["feature_b"] == "Packet Length Variance"))].copy()
    q4["grupo_P"] = [grp.get(a, "a priori var=std^2") for a in q4["feature_a"]]
    q4["pearson_menos_rinfo"] = q4["pearson_abs"] - q4["r_info"]
    q4["min_H_bits"] = q4[["H_x", "H_y"]].min(axis=1)
    q4["linfoot_aplicable"] = q4["min_H_bits"] >= 1.679226985456238
    q4["G_consenso"] = gmin[q4.index] >= 1
    q4 = q4.sort_values("r_info")
    q4[["grupo_P", "feature_a", "feature_b", "pearson_menos_rinfo", "min_H_bits", "linfoot_aplicable",
        "G_consenso", *flags.columns] + MEASURES].to_csv(cfg.TABLES_DIR / "oe1_q4_fusiones_P.csv", index=False)
    out["q4"] = {
        "n_pares": len(q4),
        "n_rinfo_menor_095": int((q4["r_info"] < 0.95).sum()),
        "n_rinfo_menor_095_linfoot_aplicable": int(((q4["r_info"] < 0.95) & q4["linfoot_aplicable"]).sum()),
        "n_G_consenso": int(q4["G_consenso"].sum()),
        "n_no_G_consenso": int((~q4["G_consenso"]).sum()),
        "pares_no_G_consenso": [f"{a} ~ {b}" for a, b in
                                q4.loc[~q4["G_consenso"], ["feature_a", "feature_b"]].itertuples(index=False)],
    }

    # --- Información retenida por cada descartada ---
    lk = pair_lookup(df)
    rows = []
    for f in cfg.load_documented_drops():
        rep = grp.get(f, cfg.A_PRIORI_DROPS.get(f))
        rr = lk[frozenset((f, rep))]
        best = max(((k, u_given(lk[frozenset((f, k))], f)) for k in kept), key=lambda t: t[1])
        rows.append({"descartada": f, "representante": rep, "u_descartada_dado_rep": u_given(rr, f),
                     "r_info_con_rep": rr["r_info"], "gauss_ratio_con_rep": rr["gauss_ratio"],
                     "gauss_ratio_min_B_con_rep": gmin[rr.name], "ksg_r_info_con_rep": rr["ksg_r_info"],
                     "pearson_con_rep": rr["pearson_abs"],
                     "mejor_conservada": best[0], "u_descartada_dado_mejor": best[1]})
    ret = pd.DataFrame(rows).sort_values("u_descartada_dado_rep")
    ret.to_csv(cfg.TABLES_DIR / "oe1_informacion_retenida.csv", index=False)

    # --- Conjunto recomendado: cambios, evidencia y chequeo de transitividad ---
    rec = by[("Recomendado", "P + restauradas + robustas", "EDA")]
    final_rec = set(rec["final_features"])
    dfb = pd.read_csv(cfg.METRICS_DIR / "oe1_pares_dependencia_train_benign.csv")
    lkb = pair_lookup(dfb)
    changes = []
    for f in rec["restored"]:
        rep = grp.get(f)
        r = lk[frozenset((f, rep))]
        changes.append({"feature": f, "accion": "restaurar (vuelve a la entrada)", "con": rep,
                        "pearson": r["pearson_abs"], "r_info_B20": r["r_info"], "ksg_r_info": r["ksg_r_info"],
                        "gauss_ratio_min_B": gmin[r.name], "robusto_max": np.nan,
                        "u_f_dado_con": u_given(r, f), "benign_r_info": lkb[frozenset((f, rep))]["r_info"],
                        "benign_gauss_ratio": lkb[frozenset((f, rep))]["gauss_ratio"]})
    for g in rec["groups"]:
        for f in g["dropped"]:
            r = lk[frozenset((f, g["rep"]))]
            rb = lkb[frozenset((f, g["rep"]))]
            changes.append({"feature": f, "accion": "descartar (nueva fusión)", "con": g["rep"],
                            "pearson": r["pearson_abs"], "r_info_B20": r["r_info"], "ksg_r_info": r["ksg_r_info"],
                            "gauss_ratio_min_B": gmin[r.name], "u_f_dado_con": u_given(r, f),
                            "benign_r_info": rb["r_info"], "benign_gauss_ratio": rb["gauss_ratio"]})
    pd.DataFrame(changes).to_csv(cfg.TABLES_DIR / "oe1_recomendado_cambios.csv", index=False)

    # Transitividad: cada feature fuera del conjunto recomendado, ¿qué fracción
    # de su entropía explica la MEJOR feature que sí queda?
    trans = []
    for f in FEATS_OUT(rec):
        best = max(((k, u_given(lk[frozenset((f, k))], f), lk[frozenset((f, k))]) for k in final_rec),
                   key=lambda t: t[1])
        trans.append({"fuera": f, "mejor_conservada": best[0], "u_fuera_dado_mejor": best[1],
                      "r_info": best[2]["r_info"], "gauss_ratio_min_B": gmin[best[2].name],
                      "robusto_con_mejor": bool(best[2]["robusto_min"] >= 1)
                      if "robusto_min" in best[2] else None})
    trans = pd.DataFrame(trans).sort_values("u_fuera_dado_mejor")
    trans.to_csv(cfg.TABLES_DIR / "oe1_recomendado_transitividad.csv", index=False)
    out["recomendado"] = {"final_size": rec["final_size"], "restored": rec["restored"],
                          "groups": rec["groups"], "ambiguous": rec["ambiguous"],
                          "u_min_fuera": float(trans["u_fuera_dado_mejor"].min()),
                          "n_fuera": len(trans)}

    # --- Q5: acuerdo KSG vs. histograma en los pares de Q3 y Q4 ---
    def agreement(sub: pd.DataFrame) -> dict:
        h = sub["r_info"] >= 0.95
        k = sub["ksg_r_info"] >= 0.95
        return {"n": len(sub), "ambos_si": int((h & k).sum()), "ambos_no": int((~h & ~k).sum()),
                "solo_hist": int((h & ~k).sum()), "solo_ksg": int((~h & k).sum()),
                "corr_spearman_rinfo_hist_vs_ksg": float(sub[["r_info", "ksg_r_info"]].corr("spearman").iloc[0, 1]),
                "discrepancias": [f"{a} ~ {b} (hist {h_:.4f}, ksg {k_:.4f})" for a, b, h_, k_ in
                                  sub.loc[h != k, ["feature_a", "feature_b", "r_info", "ksg_r_info"]]
                                  .itertuples(index=False)]}
    out["q5_q3"] = agreement(q3)
    out["q5_q4"] = agreement(q4)
    out["q5_global_spearman_rinfo_hist_vs_ksg"] = float(df[["r_info", "ksg_r_info"]].corr("spearman").iloc[0, 1])

    with open(cfg.METRICS_DIR / "oe1_contraste.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(counts.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
