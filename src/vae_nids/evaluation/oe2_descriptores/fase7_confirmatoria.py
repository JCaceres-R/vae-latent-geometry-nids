"""Fase 7 -- prueba confirmatoria de las hipótesis del OE2 (punto 8 de los pendientes).

Hipótesis registradas el 10-10-2026, ANTES de ver ningún resultado de los modelos del OE1 saneado.
Salen de la valoración sobre el modelo de 51 variables (outputs/oe2_descriptores/, 10 semillas):
un primario y seis secundarios, cada uno con el signo esperado y el rho que dio entonces.

- Primario: mediana de Mahalanobis frente al AUC (nll_mc). Una sola prueba, alfa = 0,05, sin corrección.
- Secundarios: Holm entre los 6. Se confirma un secundario si p_Holm < 0,05 y el signo es el esperado.
- p: permutación de las etiquetas de familia del AUC (la misma en todas las semillas), bilateral sobre
  |rho combinado|; también se reporta la unilateral en la dirección registrada.
- IC 95 %: bootstrap jerárquico, remuestreando familias y semillas a la vez.
- Nivel de flujo (muestra de fase2, flujo_seed{S}.npz): rho total con los flujos de las 11 familias
  juntos, rho dentro de familia (rangos centrados por familia) y media de los rho por familia. El
  efecto entre familias es el rho combinado de arriba.
- Comparación con el modelo de 51 variables, si sus tablas están en el repo: rho de cada hipótesis,
  AUC por familia y concordancia del ranking de familias.

Alcance de la confirmación: los modelos son nuevos (otra entrada, 66 variables, y otro entrenamiento),
pero las familias y los flujos son los mismos que se usaron para elegir los descriptores. Es una
confirmación sobre modelos reentrenados de forma independiente, no sobre datos nuevos.

Uso (perfil saneado):
    VAE_NIDS_OE2_PERFIL=saneado python -m vae_nids.evaluation.oe2_descriptores.fase7_confirmatoria
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from vae_nids.evaluation.oe2_descriptores import config as C
from vae_nids.evaluation.oe2_descriptores.fase3_evaluacion import (
    ATTEMPTED_ONLY,
    _row_pearson,
    combined_rho,
    fisher_mean,
    holm,
    kendall_w,
)

FECHA_REGISTRO = "2026-10-10"
ALFA = 0.05

# (id, columna, nombre, rol, signo esperado, rho en el modelo de 51 variables, descriptor por flujo)
HIPOTESIS = [
    ("1.1", "maha__median", "Mahalanobis (mediana)", "P", +1, 0.604, "maha"),
    ("2.4", "fr__auc", "Fisher-Rao al 10.º posterior benigno", "S", +1, 0.724, "fr"),
    ("6.2", "dec_knn__auc", "kNN en el espacio decodificado", "S", +1, 0.691, "dec_knn"),
    ("7.1", "rate__sep", "Tasa KL(q||p) por flujo", "S", +1, 0.681, "rate"),
    ("5.1", "knn__auc", "kNN al benigno (z blanqueado)", "S", +1, 0.694, "knn"),
    ("6.1", "logmag__sep", "Log-magnificación del decodificador", "S", +1, 0.523, "logmag"),
    ("10.1b", "sil_fr__value", "Silhouette del grupo (Fisher-Rao)", "S", -1, -0.629, None),
]


def load_tables(root):
    det = pd.read_csv(root / "tables" / "detectabilidad_por_semilla.csv")
    des = pd.read_csv(root / "tables" / "descriptores_por_semilla.csv")
    for col in [c for c in des.columns if c.endswith("__auc")]:
        des[col.replace("__auc", "__sep")] = 2 * (des[col] - 0.5).abs()
    return det, des


def matrix(des, det, col, groups, seeds):
    d = des[des.group.isin(groups)].pivot(index="seed", columns="group", values=col)[groups]
    a = det[det.group.isin(groups)].pivot(index="seed", columns="group", values="auc_nll_mc")[groups]
    return d.loc[seeds].to_numpy(float), a.loc[seeds].to_numpy(float)


def signed_permutation_rhos(dm, am, perms):
    """rho combinado (con signo) para cada permutación de las familias del AUC."""
    ok = np.isfinite(dm).all(1)
    rd = rankdata(dm[ok], axis=1)[:, None, :]
    ra = rankdata(am[ok], axis=1)[:, perms]
    r = np.clip(_row_pearson(np.broadcast_to(rd, ra.shape), ra), -0.999, 0.999)
    return np.tanh(np.nanmean(np.arctanh(r), axis=0))


def hierarchical_bootstrap(dm, am, rng, n_boot):
    ns, nf = dm.shape
    out = []
    for _ in range(n_boot):
        i_s = rng.integers(0, ns, ns)
        i_f = rng.integers(0, nf, nf)
        if len(np.unique(i_f)) < 4:
            continue
        out.append(combined_rho(dm[i_s][:, i_f], am[i_s][:, i_f])[0])
    return np.asarray(out)


def test_hypothesis(des, det, col, sign, rng):
    fam, seeds = C.MAIN_FAMILIES, C.SEEDS
    dm, am = matrix(des, det, col, fam, seeds)
    rho, rs = combined_rho(dm, am)
    perms = np.array([rng.permutation(len(fam)) for _ in range(C.N_PERM)])
    perm = signed_permutation_rhos(dm, am, perms)
    p_two = (1 + (np.abs(perm) >= abs(rho) - 1e-12).sum()) / (1 + C.N_PERM)
    p_one = (1 + (sign * perm >= sign * rho - 1e-12).sum()) / (1 + C.N_PERM)
    boot = hierarchical_bootstrap(dm, am, rng, C.N_BOOT)
    keep = [i for i, g in enumerate(fam) if g not in ("PortScan", "Bot")]
    rho9, _ = combined_rho(dm[:, keep], am[:, keep])
    lofo = [combined_rho(np.delete(dm, i, 1), np.delete(am, i, 1))[0] for i in range(len(fam))]
    dv, av = matrix(des, det, col, ATTEMPTED_ONLY, seeds)
    rho_val, _ = combined_rho(dv, av)
    return dict(rho=rho, ic_lo=float(np.percentile(boot, 2.5)), ic_hi=float(np.percentile(boot, 97.5)),
                p_bilateral=float(p_two), p_unilateral=float(p_one),
                signo_observado=int(np.sign(rho)), signo_coincide=bool(np.sign(rho) == sign),
                frac_semillas_con_signo=float(np.nanmean(np.sign(rs) == sign)),
                rho_semilla_min=float(np.nanmin(rs)), rho_semilla_max=float(np.nanmax(rs)),
                kendall_w_descriptor=kendall_w(dm), rho_sin_portscan_bot=rho9,
                lofo_min=float(np.min(lofo)), lofo_max=float(np.max(lofo)),
                lofo_familia_en_min=fam[int(np.argmin(lofo))], rho_attempted=rho_val)


def flow_level(seeds):
    """Total, dentro de familia (rangos centrados) y media por familia, con los flujos muestreados."""
    rows = []
    for _, _, name, _, _, _, k in HIPOTESIS:
        if k is None:
            continue
        per_seed = []
        for s in seeds:
            path = C.CACHE_DIR / f"flujo_seed{s}.npz"
            if not path.exists():
                continue
            z = np.load(path)
            xs, ys, gs, fam_rhos = [], [], [], []
            for gi, g in enumerate(C.MAIN_FAMILIES):
                if f"{g}__{k}" not in z:
                    continue
                x, y = z[f"{g}__{k}"].astype(float), z[f"{g}__nll"].astype(float)
                ok = np.isfinite(x) & np.isfinite(y)
                x, y = x[ok], y[ok]
                xs.append(x); ys.append(y); gs.append(np.full(len(x), gi))
                if len(x) >= 10 and np.ptp(x) > 0 and np.ptp(y) > 0:
                    fam_rhos.append(spearmanr(x, y)[0])
            x, y, g = np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)
            rx, ry = rankdata(x), rankdata(y)
            total = np.corrcoef(rx, ry)[0, 1]
            cx = rx - (np.bincount(g, rx) / np.bincount(g))[g]
            cy = ry - (np.bincount(g, ry) / np.bincount(g))[g]
            within = np.corrcoef(cx, cy)[0, 1]
            per_seed.append((total, within, fisher_mean(fam_rhos), len(x)))
        if not per_seed:
            continue
        t, w, f, n = map(np.asarray, zip(*per_seed))
        rows.append(dict(descriptor=name, descriptor_flujo=k, n_semillas=len(per_seed), n_flujos_por_semilla=int(n.mean()),
                         rho_total=fisher_mean(t), rho_dentro_familia=fisher_mean(w),
                         rho_medio_por_familia=fisher_mean(f),
                         rho_total_min=float(t.min()), rho_total_max=float(t.max()),
                         rho_dentro_min=float(w.min()), rho_dentro_max=float(w.max())))
    return pd.DataFrame(rows)


def comparison_51(det_new):
    """AUC por familia y ranking de familias: modelo de 51 variables frente al perfil actual."""
    p = C.ROOT_ORIGINAL / "tables" / "detectabilidad_por_semilla.csv"
    if C.PERFIL == "original" or not p.exists():
        return None, None
    det_old = pd.read_csv(p)
    groups = C.MAIN_FAMILIES + ATTEMPTED_ONLY
    a_old = det_old.groupby("group").auc_nll_mc.agg(["mean", "std"])
    a_new = det_new.groupby("group").auc_nll_mc.agg(["mean", "std"])
    t = pd.DataFrame({"auc_51_media": a_old["mean"], "auc_51_sd": a_old["std"],
                      "auc_66_media": a_new["mean"], "auc_66_sd": a_new["std"]}).loc[groups]
    t["diferencia_66_menos_51"] = t.auc_66_media - t.auc_51_media
    t = t.reset_index().rename(columns={"index": "group"})
    main = t[t.group.isin(C.MAIN_FAMILIES)]
    rho_rank = float(spearmanr(main.auc_51_media, main.auc_66_media)[0])
    return t, rho_rank


def md_table(df, floatfmt=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("—" if not np.isfinite(v) else f"{v:.{floatfmt}f}")
            else:
                cells.append(str(v).replace("|", r"\|"))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    rng = np.random.default_rng(C.RNG_SEED + 7)
    det, des = load_tables(C.ROOT)
    rows = []
    for hid, col, name, rol, sign, rho51, _ in HIPOTESIS:
        r = test_hypothesis(des, det, col, sign, rng)
        r.update(id=hid, columna=col, descriptor=name, rol=rol, signo_esperado=sign, rho_modelo_51=rho51)
        rows.append(r)
        print(f"[fase7] {hid:6s} {name:42s} rho={r['rho']:+.3f} [{r['ic_lo']:+.3f},{r['ic_hi']:+.3f}] "
              f"p={r['p_bilateral']:.4f} (51 variables: {rho51:+.3f})")
    df = pd.DataFrame(rows)
    sec = df.rol == "S"
    df.loc[sec, "p_holm"] = holm(df.loc[sec, "p_bilateral"])
    df.loc[~sec, "p_holm"] = df.loc[~sec, "p_bilateral"]          # el primario no se corrige
    df["confirmada"] = (df.p_holm < ALFA) & df.signo_coincide
    first = ["id", "descriptor", "rol", "signo_esperado", "rho_modelo_51", "rho", "ic_lo", "ic_hi",
             "p_bilateral", "p_unilateral", "p_holm", "confirmada"]
    df = df[first + [c for c in df.columns if c not in first]]
    df.to_csv(C.TABLES_DIR / "confirmatoria_hipotesis.csv", index=False)

    flow = flow_level(C.SEEDS)
    flow.to_csv(C.TABLES_DIR / "confirmatoria_nivel_flujo.csv", index=False)

    cmp_auc, rho_rank = comparison_51(det)
    if cmp_auc is not None:
        cmp_auc.to_csv(C.TABLES_DIR / "comparacion_auc_51_vs_66.csv", index=False)

    meta = dict(fecha_registro=FECHA_REGISTRO, perfil=C.PERFIL, semillas=C.SEEDS, n_familias=len(C.MAIN_FAMILIES),
                n_perm=C.N_PERM, n_boot=C.N_BOOT, alfa=ALFA,
                primario_confirmado=bool(df.loc[df.rol == "P", "confirmada"].iloc[0]),
                secundarios_confirmados=df.loc[sec & df.confirmada, "id"].tolist(),
                rho_ranking_auc_51_vs_66=rho_rank,
                hipotesis=[dict(id=h[0], columna=h[1], rol=h[3], signo=h[4], rho_51=h[5]) for h in HIPOTESIS])
    json.dump(meta, open(C.TABLES_DIR / "confirmatoria.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    lines = [f"# OE2 — prueba confirmatoria (perfil {C.PERFIL})", "",
             f"Hipótesis registradas el {FECHA_REGISTRO}, antes de ver los resultados de este perfil "
             f"(`fase7_confirmatoria.py`). {len(C.SEEDS)} semillas ({', '.join(map(str, C.SEEDS))}), "
             f"{len(C.MAIN_FAMILIES)} familias, {C.N_PERM:,} permutaciones, {C.N_BOOT:,} remuestreos jerárquicos.", "",
             "## Hipótesis", "",
             md_table(df[["id", "descriptor", "rol", "signo_esperado", "rho_modelo_51", "rho", "ic_lo", "ic_hi",
                          "p_bilateral", "p_holm", "confirmada", "rho_sin_portscan_bot", "rho_attempted"]]), "",
             f"- Primario confirmado: **{'sí' if meta['primario_confirmado'] else 'no'}**.",
             f"- Secundarios confirmados (Holm entre 6): **{', '.join(meta['secundarios_confirmados']) or 'ninguno'}**.", ""]
    if len(flow):
        lines += ["## Nivel de flujo (entre familias y dentro de cada familia)", "",
                  "Entre familias = el rho combinado de la tabla de arriba. Total = flujos de las 11 familias juntos. "
                  f"Dentro = rangos centrados en cada familia (hasta {C.FLOW_SAMPLE_MAX} flujos por familia y semilla).", "",
                  md_table(flow[["descriptor", "rho_total", "rho_dentro_familia", "rho_medio_por_familia",
                                 "rho_dentro_min", "rho_dentro_max"]]), ""]
    if cmp_auc is not None:
        lines += ["## AUC por familia: 51 frente a 66 variables", "",
                  f"Spearman del ranking de las 11 familias por AUC medio entre los dos modelos: **{rho_rank:+.3f}**.", "",
                  md_table(cmp_auc), ""]
    (C.ROOT / "RESULTADOS_CONFIRMATORIA.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[fase7] resumen en {C.ROOT / 'RESULTADOS_CONFIRMATORIA.md'}")
    return df


if __name__ == "__main__":
    main()
