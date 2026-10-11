"""OE4 -- reproducibilidad del ranking de familias de ataque frente a beta y k.

Enunciado (propuesta, 22-07-2026): reentrenar con 2-3 configuraciones de beta/k, comparar el ranking de
familias (Mahalanobis, AUC) entre configuraciones y medir su concordancia (Spearman / Kendall).

Configuraciones (fijadas el 11-10-2026): base beta = 1, k = 8 (OE1 saneado) frente a beta = 0,5,
beta = 2 y k = 16; 5 semillas cada una (42-46). La base se restringe a las mismas 5 semillas.

Lee lo que dejaron, por configuración, vae_nids.oe1.run (diagnósticos) y las fases 1, 2, 3 y 7 de
oe2_descriptores. Escribe en outputs/oe4/:
  tabla_configuraciones.csv     unidades activas, KL, épocas, piso del decodificador
  auc_por_familia.csv           AUC (nll_mc) por familia y configuración
  concordancia_rankings.csv     Spearman y tau de Kendall de cada configuración frente a la base, por métrica,
                                y W de Kendall entre las 4 configuraciones
  hallazgo_por_configuracion.csv  primario y 6 secundarios de la fase 7 en cada configuración
  RESULTADOS_OE4.md, figures/

Uso:
    python -m vae_nids.evaluation.oe4.analisis
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402

from vae_nids import config as base_cfg  # noqa: E402
from vae_nids.evaluation.encode_latent_oe2 import ATTACK_GROUPS  # noqa: E402
from vae_nids.evaluation.oe2_descriptores.fase3_evaluacion import kendall_w  # noqa: E402

P = base_cfg.PROJECT_ROOT
ROOT = P / "outputs" / "oe4"
CONFIGS = [  # (etiqueta, archivo de configuración, carpeta de descriptores)
    ("base β=1, k=8", "configs/oe1_saneado.yaml", P / "outputs" / "oe2_descriptores_saneado"),
    ("β=0,5", "configs/oe4_beta0p5.yaml", P / "outputs" / "oe4" / "beta0p5" / "oe2_descriptores"),
    ("β=2", "configs/oe4_beta2.yaml", P / "outputs" / "oe4" / "beta2" / "oe2_descriptores"),
    ("k=16", "configs/oe4_k16.yaml", P / "outputs" / "oe4" / "k16" / "oe2_descriptores"),
]
SEEDS = [42, 43, 44, 45, 46]
FAMILIES = list(ATTACK_GROUPS)
METRICAS = [  # (columna, nombre) -- la AUC sale de detectabilidad_por_semilla
    ("auc_nll_mc", "AUC (nll_mc)"),
    ("maha__median", "Mahalanobis (mediana)"),
    ("rate__sep", "Tasa KL"),
    ("fr__auc", "Fisher-Rao"),
    ("knn__auc", "kNN blanqueado"),
    ("dec_knn__auc", "kNN decodificado"),
    ("logmag__sep", "Log-magnificación"),
    ("sil_fr__value", "Silhouette Fisher-Rao"),
]
INK, MUTED = "#0b0b0b", "#52514e"
COLORS = ["#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a"]


def load_config(path):
    with open(P / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def family_table(droot):
    det = pd.read_csv(droot / "tables" / "detectabilidad_por_semilla.csv")
    des = pd.read_csv(droot / "tables" / "descriptores_por_semilla.csv")
    for col in [c for c in des.columns if c.endswith("__auc")]:
        des[col.replace("__auc", "__sep")] = 2 * (des[col] - 0.5).abs()
    t = des.merge(det[["seed", "group", "auc_nll_mc"]], on=["seed", "group"])
    return t[t.seed.isin(SEEDS)]


def diagnostics(cfg, label):
    d = pd.read_csv(P / cfg["salidas"]["dir"] / "tables" / "diagnosticos_por_semilla.csv")
    d = d[(d.corrida == cfg["entrenamiento"]["nombre_corrida"]) & d.seed.isin(SEEDS)]
    return dict(configuracion=label, beta=cfg["modelo"]["beta"], k=cfg["modelo"]["latent_dim"], semillas=len(d),
                activas_media=d.activas_burda.mean(), activas_sd=d.activas_burda.std(), kl_total=d.kl_total.mean(),
                mejor_epoca=d.mejor_epoca.mean(), nll_por_columna=d.nll_media_por_columna.mean(),
                frac_piso_decodificador=d.frac_logvar_dec_en_piso.mean())


def md_table(df, d=3):
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(("—" if not np.isfinite(v) else f"{v:.{d}f}") if isinstance(v, (float, np.floating))
                                     else str(v).replace("|", r"\|") for v in r) + " |")
    return "\n".join(out)


def main():
    global CONFIGS
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    faltan = [lab for lab, path, droot in CONFIGS
              if not (droot / "tables" / "descriptores_por_semilla.csv").exists()
              or not (P / load_config(path)["salidas"]["dir"] / "tables" / "diagnosticos_por_semilla.csv").exists()]
    if CONFIGS[0][0] in faltan:
        raise FileNotFoundError("faltan las tablas de la configuración base (perfil saneado)")
    CONFIGS = [c for c in CONFIGS if c[0] not in faltan]
    cfgs = {lab: load_config(path) for lab, path, _ in CONFIGS}
    tabs = {lab: family_table(droot) for lab, _, droot in CONFIGS}
    base = CONFIGS[0][0]

    conf = pd.DataFrame([diagnostics(cfgs[lab], lab) for lab, _, _ in CONFIGS])
    conf.to_csv(ROOT / "tabla_configuraciones.csv", index=False)

    auc = pd.DataFrame({lab: t.groupby("group").auc_nll_mc.mean() for lab, t in tabs.items()}).reindex(
        FAMILIES + [g for g in tabs[base].group.unique() if g not in FAMILIES and g != "BENIGN_control"])
    auc.index.name = "familia"
    auc.to_csv(ROOT / "auc_por_familia.csv")

    rows = []
    for col, name in METRICAS:
        means = {lab: t[t.group.isin(FAMILIES)].groupby("group")[col].mean().reindex(FAMILIES) for lab, t in tabs.items()}
        for lab, m in means.items():
            if lab == base:
                continue
            rows.append(dict(metrica=name, configuracion=lab, spearman_vs_base=spearmanr(means[base], m)[0],
                             kendall_tau_vs_base=kendalltau(means[base], m)[0],
                             w_entre_semillas=kendall_w(tabs[lab][tabs[lab].group.isin(FAMILIES)]
                                                        .pivot(index="seed", columns="group", values=col)[FAMILIES].to_numpy(float))))
        rows.append(dict(metrica=name, configuracion=f"W entre las {len(means)} configuraciones", spearman_vs_base=np.nan,
                         kendall_tau_vs_base=np.nan, w_entre_semillas=kendall_w(np.vstack([m.to_numpy(float) for m in means.values()]))))
    con = pd.DataFrame(rows)
    con.to_csv(ROOT / "concordancia_rankings.csv", index=False)

    hall = []
    for lab, _, droot in CONFIGS:
        p = droot / "tables" / "confirmatoria_hipotesis.csv"
        if not p.exists():
            continue
        h = pd.read_csv(p)
        for r in h.itertuples():
            hall.append(dict(configuracion=lab, id=r.id, descriptor=r.descriptor, rol=r.rol, rho=r.rho, ic_lo=r.ic_lo,
                             ic_hi=r.ic_hi, p_holm=r.p_holm, confirmada=bool(r.confirmada)))
    hall = pd.DataFrame(hall)
    hall.to_csv(ROOT / "hallazgo_por_configuracion.csv", index=False)
    # nota: la base usa aquí sus 8 semillas (su fase 7); las configuraciones del OE4, 5.

    # figuras
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    order = auc.loc[FAMILIES, base].sort_values().index
    for i, lab in enumerate(auc.columns):
        ax.plot(auc.loc[order, lab], np.arange(len(order)) + (i - 1.5) * 0.12, "o", ms=4, color=COLORS[i], label=lab)
    ax.set_yticks(np.arange(len(order)), order, fontsize=7)
    ax.set_xlabel("AUC-ROC (nll_mc), media de 5 semillas", color=MUTED)
    ax.set_title("Detectabilidad por familia en cada configuración", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "auc_por_familia_y_configuracion.png", dpi=200); plt.close(fig)
    if len(hall):
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ids = list(dict.fromkeys(hall.descriptor))
        for i, lab in enumerate(hall.configuracion.unique()):
            h = hall[hall.configuracion == lab].set_index("descriptor").reindex(ids)
            y = np.arange(len(ids)) + (i - 1.5) * 0.15
            ax.errorbar(h.rho, y, xerr=[h.rho - h.ic_lo, h.ic_hi - h.rho], fmt="o", ms=4, lw=1, capsize=2,
                        color=COLORS[i], label=lab)
        ax.axvline(0, color=MUTED, lw=1, ls="--")
        ax.set_yticks(np.arange(len(ids)), ids, fontsize=7)
        ax.set_xlabel("ρ combinado descriptor–AUC (IC 95 %)", color=MUTED)
        ax.set_title("Hallazgo central en cada configuración", loc="left", fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="lower left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout(); fig.savefig(ROOT / "figures" / "hallazgo_por_configuracion.png", dpi=200); plt.close(fig)

    order = list(dict.fromkeys(hall.descriptor)) if len(hall) else []   # orden de la fase 7: primario primero
    piv = hall.pivot(index="descriptor", columns="configuracion", values="rho").reindex(index=order, columns=[c[0] for c in CONFIGS]) if len(hall) else None
    okp = hall.pivot(index="descriptor", columns="configuracion", values="confirmada").reindex(index=order, columns=[c[0] for c in CONFIGS]) if len(hall) else None
    lines = ["# OE4 — reproducibilidad del ranking frente a β y k", "",
             f"Configuraciones con 66 variables y semillas {SEEDS}; la base (β = 1, k = 8) se restringe a esas semillas "
             "en todo salvo la fila del hallazgo, que usa su fase 7 completa (8 semillas).", "",
             *([f"**Configuraciones sin resultados (alguna etapa falló): {', '.join(faltan)}.**", ""] if faltan else []),
             "## Configuraciones", "", md_table(conf, 3), "",
             "## Concordancia del ranking de las 11 familias frente a la base", "",
             "`w_entre_semillas` = W de Kendall entre las 5 semillas de esa configuración (estabilidad interna). "
             "La fila «W entre las N configuraciones» trata cada configuración como un juez.", "",
             md_table(con, 3), "", "## AUC por familia", "", md_table(auc.reset_index(), 4), ""]
    if piv is not None:
        lines += ["## Hallazgo central en cada configuración (ρ combinado; * = confirmado con Holm)", "",
                  md_table(pd.DataFrame({c: [f"{piv.loc[i, c]:+.3f}{'*' if okp.loc[i, c] else ''}" if pd.notna(piv.loc[i, c]) else "—"
                                             for i in piv.index] for c in piv.columns}, index=piv.index).reset_index()), ""]
    lines.append("Figuras: `outputs/oe4/figures/`.")
    (ROOT / "RESULTADOS_OE4.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[oe4] resumen en {ROOT / 'RESULTADOS_OE4.md'}")


if __name__ == "__main__":
    main()
