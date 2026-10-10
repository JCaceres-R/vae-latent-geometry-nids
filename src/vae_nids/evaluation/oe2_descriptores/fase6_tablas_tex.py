"""Fase 6 -- tablas LaTeX y macros de cifras del informe, generadas desde los CSV.

Ninguna cifra del informe se escribe a mano: el .tex usa \\input{tablas/...} y las
macros de tablas/cifras.tex.
Uso:
    python -m vae_nids.evaluation.oe2_descriptores.fase6_tablas_tex <carpeta_destino>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vae_nids.evaluation.oe2_descriptores import config as C
from vae_nids.evaluation.oe2_descriptores.fase3_evaluacion import ATTEMPTED_ONLY

GROUP_TEX = {g: g.replace("_", r"\_") for g in C.ALL_GROUPS + ["BENIGN_test"]}


def num(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace(".", "{,}").replace("-", "$-$").replace("+", "$+$")


def pval(p):
    if not np.isfinite(p):
        return "--"
    return r"$<$0{,}001" if p < 0.001 else num(p, 3)


def tex_escape(s):
    return (str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("μ", r"$\mu$").replace("σ²", r"$\sigma^2$").replace("||", r"$\|$").replace("²", r"$^2$"))


def tabla_detectabilidad(out):
    fam = pd.read_csv(C.TABLES_DIR / "resumen_por_familia.csv").set_index("group")
    dup = pd.read_csv(C.TABLES_DIR / "duplicados_seed42.csv").set_index("group")
    det = pd.read_csv(C.TABLES_DIR / "detectabilidad_por_semilla.csv")
    rows = []
    for g in C.MAIN_FAMILIES + ATTEMPTED_ONLY:
        a = det[det.group == g].auc_nll_mc
        rows.append(f"{GROUP_TEX[g]} & {int(dup.loc[g, 'n']):,} & {int(dup.loc[g, 'mu_unicos']):,} & "
                    f"{int(round(dup.loc[g, 'n_efectivo'])):,} & {num(a.mean(), 3)} & {num(a.std(), 3)} & "
                    f"{num(a.min(), 3)}--{num(a.max(), 3)} \\\\".replace(",", r"\,").replace(r"{\,}", "{,}"))
        if g == C.MAIN_FAMILIES[-1]:
            rows.append(r"\midrule")
    body = "\n".join(rows)
    (out / "tabla_detectabilidad.tex").write_text(
        "\\begin{tabular}{@{}lrrrccc@{}}\n\\toprule\n"
        "Grupo & $n$ & $\\mu$ distintos & $n$ efectivo & AUC media & DE & mín.--máx. \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n", encoding="utf-8")


def tabla_diagnosticos(out):
    d = pd.read_csv(C.TABLES_DIR / "diagnosticos_por_semilla.csv")
    rows = [f"{int(r.seed)} & {int(r.n_activas)} & {num(r.kl_total, 2)} & {num(r.erank_G_mediana, 2)} & "
            f"{num(100 * r.frac_logvar_dec_en_piso, 1)} & {num(r.sum_sigma2_obs, 2)} & {num(r.sum_sigma2_pred, 2)} & "
            f"{num(r.kp_spearman, 2)} \\\\" for r in d.itertuples()]
    (out / "tabla_diagnosticos.tex").write_text(
        "\\begin{tabular}{@{}rrrrrrrr@{}}\n\\toprule\n"
        "Semilla & Dims. activas & KL total & Rango ef. $G$ & \\% $\\log\\sigma_x^2$ en piso & "
        "$\\sum\\sigma^2$ obs. & $\\sum\\sigma^2$ pred. & $\\rho$ K--P \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def _val_rows(v):
    rows = []
    for r in v.itertuples():
        ph = pval(r.p_holm) if r.nivel == "S" else (pval(r.p_bh) if r.nivel in ("E", "R") else "--")
        rows.append(
            f"{tex_escape(r.id)} & {tex_escape(r.descriptor)} & {num(r.rho, 2, True)} & "
            f"[{num(r.ci_lo, 2, True)}, {num(r.ci_hi, 2, True)}] & {pval(r.p_perm)} & {ph} & "
            f"{num(r.kendall_w, 2)} & {num(r.sign_consistency, 1)} & {num(r.rho_with_maha, 2, True)} & "
            f"{num(r.partial_rho, 2, True)} & {num(r.rho_sin_portscan_bot, 2, True)} & "
            f"{num(r.rho_attempted, 2, True)} & {tex_escape(r.tier[0])} \\\\")
    return rows


HEAD_VAL = ("Id & Descriptor & $\\bar\\rho$ & IC 95\\,\\% & $p$ & $p$ aj. & $W$ & Signo & "
            "$\\rho_{M}$ & $\\rho_{\\cdot|M}$ & $\\rho_{9}$ & $\\rho_{att}$ & Nivel \\\\")


def tabla_valoracion(out):
    v = pd.read_csv(C.TABLES_DIR / "valoracion_descriptores.csv")
    main = v[v.nivel.isin(["P", "S"])].sort_values("rho", key=lambda s: -s.abs())
    (out / "tabla_valoracion.tex").write_text(
        "\\begin{tabular}{@{}l>{\\raggedright\\arraybackslash}p{3.3cm}rcrrrrrrrrc@{}}\n\\toprule\n"
        + HEAD_VAL + "\n\\midrule\n" + "\n".join(_val_rows(main)) + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8")
    allv = v.sort_values(["nivel", "rho"], key=lambda s: s if s.name == "nivel" else -s.abs())
    order = {"P": 0, "S": 1, "E": 2, "R": 3}
    allv = allv.assign(o=allv.nivel.map(order)).sort_values(["o", "rho"], ascending=[True, False])
    (out / "tabla_valoracion_completa.tex").write_text(
        "\\begin{longtable}{@{}l>{\\raggedright\\arraybackslash}p{3.6cm}rcrrrrrrrrc@{}}\n"
        "\\caption{Valoración completa (todos los descriptores calculados). Columnas como en el cuadro~\\ref{tab:valoracion}; "
        "$p$ aj.: Holm en secundarios, Benjamini--Hochberg en exploratorios y del repositorio.}\\label{tab:valoracion-completa}\\\\\n"
        "\\toprule\n" + HEAD_VAL + "\n\\midrule\n\\endfirsthead\n\\toprule\n" + HEAD_VAL + "\n\\midrule\n\\endhead\n"
        + "\n".join(_val_rows(allv)) + "\n\\bottomrule\n\\end{longtable}\n", encoding="utf-8")


def tabla_nivel_flujo(out):
    v = pd.read_csv(C.TABLES_DIR / "valoracion_descriptores.csv")
    fl = pd.read_csv(C.TABLES_DIR / "nivel_flujo_por_familia.csv", index_col=0)
    keep = [c for c in fl.columns if c.split("__")[0] in ("maha", "knn", "fr", "rate", "dec_knn", "logmag", "lid", "gmm")]
    names = {"maha": "1.1", "knn": "5.1", "fr": "2.4", "rate": "7.1", "dec_knn": "6.2", "logmag": "6.1", "lid": "8.1", "gmm": "1.2"}
    head = "Familia & " + " & ".join(names[c.split("__")[0]] for c in keep) + " \\\\"
    rows = [f"{GROUP_TEX[g]} & " + " & ".join(num(fl.loc[g, c], 2, True) for c in keep) + " \\\\"
            for g in C.MAIN_FAMILIES if g in fl.index]
    rows.append("\\midrule")
    rows.append("Media & " + " & ".join(num(fl.loc[C.MAIN_FAMILIES, c].mean(), 2, True) for c in keep) + " \\\\")
    (out / "tabla_nivel_flujo.tex").write_text(
        "\\begin{tabular}{@{}l" + "r" * len(keep) + "@{}}\n\\toprule\n" + head + "\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def cifras(out):
    v = pd.read_csv(C.TABLES_DIR / "valoracion_descriptores.csv").set_index("columna")
    det = pd.read_csv(C.TABLES_DIR / "detectabilidad_por_semilla.csv")
    dup = pd.read_csv(C.TABLES_DIR / "duplicados_seed42.csv").set_index("group")
    dg = pd.read_csv(C.TABLES_DIR / "diagnosticos_por_semilla.csv")
    meta = json.load(open(C.TABLES_DIR / "verificacion_auc_seed42.json"))
    m = {}
    letters = {"maha__median": "Maha", "knn__auc": "Knn", "fr__auc": "Fr", "rate__sep": "Rate", "mmd_agg__value": "Mmd",
               "dec_knn__auc": "Deck", "logmag__sep": "Logmag", "airm__value": "Airm", "lid__sep": "Lid",
               "sil_w__value": "Silw", "probe__value": "Probe", "sil_fr__value": "Silfr", "gmm__auc": "Gmm",
               "logvol__sep": "Logvol", "maha__auc": "Mahaauc", "fr__median": "Frmed", "rate__median": "Ratemed",
               "typ__auc": "Typ", "rmd__auc": "Rmd"}
    for col, L in letters.items():
        r = v.loc[col]
        m[f"rho{L}"] = num(r.rho, 2, True)
        m[f"ci{L}"] = f"[{num(r.ci_lo, 2, True)}, {num(r.ci_hi, 2, True)}]"
        m[f"p{L}"] = pval(r.p_perm)
        m[f"pholm{L}"] = pval(r.p_holm) if np.isfinite(r.get("p_holm", np.nan)) else "--"
        m[f"w{L}"] = num(r.kendall_w, 2)
        m[f"rhoM{L}"] = num(r.rho_with_maha, 2, True)
        m[f"rhoNueve{L}"] = num(r.rho_sin_portscan_bot, 2, True)
        m[f"rhoAtt{L}"] = num(r.rho_attempted, 2, True)
        m[f"par{L}"] = num(r.partial_rho, 2, True)
        m[f"flujo{L}"] = num(r.flow_rho_nll_mean, 2, True)
        m[f"lofoMin{L}"] = num(r.lofo_min, 2, True)
        m[f"lofoMax{L}"] = num(r.lofo_max, 2, True)
    fam_auc = det[det.group.isin(C.MAIN_FAMILIES)].groupby("group").auc_nll_mc.mean()
    m["aucMin"] = num(fam_auc.min(), 3); m["aucMinFam"] = GROUP_TEX[fam_auc.idxmin()]
    m["aucMax"] = num(fam_auc.max(), 3); m["aucMaxFam"] = GROUP_TEX[fam_auc.idxmax()]
    m["aucBot"] = num(fam_auc["Bot"], 3); m["aucPortScan"] = num(fam_auc["PortScan"], 3)
    m["difAucFase"] = f"{meta['max_abs_diff_auc_seed42_vs_phase4']:.1e}".replace(".", "{,}").replace("e-0", r"\times10^{-") + "}"
    m["portscanUnicos"] = f"{int(dup.loc['PortScan', 'mu_unicos']):,}".replace(",", r"\,")
    m["portscanFrac"] = num(100 * dup.loc["PortScan", "frac_unicos"], 1)
    m["portscanNeff"] = num(dup.loc["PortScan", "n_efectivo"], 0)
    m["portscanMaxRep"] = f"{int(dup.loc['PortScan', 'max_repeticiones']):,}".replace(",", r"\,")
    m["benignNeff"] = f"{int(round(dup.loc['BENIGN_test', 'n_efectivo'])):,}".replace(",", r"\,")
    m["benignFrac"] = num(100 * dup.loc["BENIGN_test", "frac_unicos"], 1)
    m["botattFrac"] = num(100 * dup.loc["Bot_attempted", "frac_unicos"], 1)
    m["erankMin"] = num(dg.erank_G_mediana.min(), 2); m["erankMax"] = num(dg.erank_G_mediana.max(), 2)
    m["pisoMin"] = num(100 * dg.frac_logvar_dec_en_piso.min(), 0); m["pisoMax"] = num(100 * dg.frac_logvar_dec_en_piso.max(), 0)
    m["kpMin"] = num(dg.kp_spearman.min(), 2); m["kpMax"] = num(dg.kp_spearman.max(), 2)
    m["nSemillas"] = str(len(C.SEEDS)); m["nFamilias"] = str(len(C.MAIN_FAMILIES))
    m["nPerm"] = f"{C.N_PERM:,}".replace(",", r"\,"); m["nBoot"] = f"{C.N_BOOT:,}".replace(",", r"\,")
    m["nRef"] = f"{C.REF_MAX_N:,}".replace(",", r"\,")
    lines = ["% Generado por fase6_tablas_tex.py: no editar a mano."]
    for k, val in m.items():
        lines.append(f"\\newcommand{{\\cifra{k}}}{{{val}}}")
    (out / "cifras.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(dest):
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    tabla_detectabilidad(out)
    tabla_diagnosticos(out)
    tabla_valoracion(out)
    tabla_nivel_flujo(out)
    cifras(out)
    print("[fase6] tablas en", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else C.ROOT / "tablas_tex")
