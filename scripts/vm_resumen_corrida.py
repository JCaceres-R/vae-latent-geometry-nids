"""Resumen de la corrida de pendientes en la VM (scripts/run_pendientes_vm.sh).

Lee lo que dejó cada etapa y escribe, en outputs/vm_pendientes/:
  RESUMEN_CORRIDA.md        estado de cada etapa, entorno y cifras clave del OE1 y del OE2
  entorno.json, pip_freeze.txt
  ARCHIVOS_PARA_COMMIT.txt  rutas que conviene versionar (las usa scripts/vm_preparar_commit.sh)
  MENSAJE_COMMIT.txt        mensaje sugerido para el commit

Cada sección es independiente: si una etapa falló, su sección dice "no disponible" y el resto sigue.

Uso:
    python scripts/vm_resumen_corrida.py
"""
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "vm_pendientes"
OE1 = REPO / "outputs" / "oe1_saneado"
ROB = REPO / "outputs" / "oe2_robust_saneado"
ROB51 = REPO / "outputs" / "oe2_robust"
DES = REPO / "outputs" / "oe2_descriptores_saneado"
DES51 = REPO / "outputs" / "oe2_descriptores"
PROC = REPO / "data" / "processed" / "oe1_saneado"

# Lo que se versiona: tablas, métricas, figuras, logs y checkpoints pequeños (≈40 KB). No se versionan
# los vectores latentes, los scores por conexión ni las cachés (cientos de MB, regenerables).
COMMIT_PATHS = [
    "outputs/vm_pendientes",
    "outputs/oe1_saneado/checkpoints", "outputs/oe1_saneado/logs", "outputs/oe1_saneado/tables",
    "outputs/oe1_saneado/figures", "outputs/oe1_saneado/figures_informe", "outputs/oe1_saneado/preproceso",
    "outputs/oe1_saneado/informe_tablas", "outputs/oe1_saneado/registro_run.log",
    "outputs/oe1_saneado/datos_info.json",
    "outputs/oe2_robust_saneado/tables", "outputs/oe2_robust_saneado/metrics", "outputs/oe2_robust_saneado/figures",
    "outputs/oe2_robust_saneado/verificacion_conteos.json",
    "outputs/oe2_descriptores_saneado/tables", "outputs/oe2_descriptores_saneado/figures",
    "outputs/oe2_descriptores_saneado/tablas_tex", "outputs/oe2_descriptores_saneado/RESULTADOS_CONFIRMATORIA.md",
]


def sh(*args):
    try:
        return subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(error: {e})"


def f(x, d=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "—" if not np.isfinite(x) else f"{x:.{d}f}"


def md_table(df, d=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f(v, d) if isinstance(v, (float, np.floating)) else str(v).replace("|", r"\|")
                                     for v in r) + " |")
    return "\n".join(out)


def section(title, fn):
    lines = [f"## {title}", ""]
    try:
        lines += fn()
    except Exception:  # noqa: BLE001
        lines += ["No disponible (falta un archivo o la etapa falló):", "", "```", traceback.format_exc(limit=2), "```"]
    return lines + [""]


# ---------------------------------------------------------------------------------------------
def etapas():
    p = OUT / "estado_etapas.jsonl"
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    last = {}
    for r in rows:                       # si una etapa se relanzó, vale el último registro
        last[r["id"]] = r
    t = pd.DataFrame(list(last.values()))
    t["min"] = (t["segundos"] / 60).round(1)
    return [md_table(t[["id", "etapa", "estado", "inicio", "min", "log"]], 1), "",
            f"Fallidas: **{(t.estado == 'FALLO').sum()}**. Saltadas por dependencia: **{(t.estado == 'SALTADA').sum()}**.",
            "Si alguna falló, la cola de su log está en `RUN_LOG.md` y el log completo en `outputs/vm_pendientes/logs/`."]


def entorno():
    import importlib.metadata as md
    info = {"fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "host": platform.node(),
            "sistema": platform.platform(), "python": sys.version.split()[0], "cpus": os.cpu_count(),
            "git_commit": sh("git", "rev-parse", "HEAD"), "git_rama": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "git_arbol_limpio_al_resumir": sh("git", "status", "--porcelain", "--untracked-files=no") == ""}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:  # noqa: BLE001
        pass
    for pkg in ("torch", "numpy", "pandas", "scikit-learn", "scipy", "umap-learn", "statsmodels"):
        try:
            info[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            info[pkg] = None
    json.dump(info, open(OUT / "entorno.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    (OUT / "pip_freeze.txt").write_text(sh(sys.executable, "-m", "pip", "freeze") + "\n", encoding="utf-8")
    return [f"- `{k}`: {v}" for k, v in info.items()]


def oe1():
    tab = pd.read_csv(OE1 / "tables" / "diagnosticos_por_semilla.csv")
    m = tab[tab.corrida == "oe1_info_nula"]
    r74 = tab[tab.corrida == "referencia_74"]
    cols = ["seed", "n_columnas", "epoca_parada", "mejor_epoca", "mejor_val_total", "activas_burda", "activas_kl",
            "kl_total", "tau_p95", "nll_media_por_columna", "columnas_en_piso", "frac_logvar_dec_en_piso"]
    lines = ["Modelos del OE1 saneado (66 variables) y referencia de 74. Diagnósticos solo con `val_benign`.", "",
             md_table(tab[["corrida"] + cols], 3), "",
             f"- Unidades activas (Burda, Var(μ) > 0,01), 66 variables: media **{m.activas_burda.mean():.2f}** "
             f"(sd {m.activas_burda.std():.2f}; {', '.join(map(str, m.activas_burda))}).",
             f"- KL total media: {m.kl_total.mean():.2f} nats. Fracción de log σ² del decodificador en el piso −10: "
             f"{m.frac_logvar_dec_en_piso.mean():.3f}."]
    if len(r74):
        lines.append(f"- Referencia 74 (semilla {int(r74.seed.iloc[0])}): {int(r74.activas_burda.iloc[0])} unidades activas, "
                     f"KL {r74.kl_total.iloc[0]:.2f}.")
    old = OE1 / "tables" / "diagnosticos_oe1_actual_51.csv"
    if old.exists():
        o = pd.read_csv(old)
        lines += ["", "**Antes y después** (medias por semilla):", "",
                  md_table(pd.DataFrame([
                      dict(modelo="OE1 actual, 51 variables", semillas=len(o), activas=o.activas_burda.mean(),
                           kl_total=o.kl_total.mean(), mejor_epoca=o.mejor_epoca.mean(),
                           nll_por_columna=o.nll_media_por_columna.mean(), frac_piso=o.frac_logvar_dec_en_piso.mean()),
                      dict(modelo="OE1 saneado, 66 variables", semillas=len(m), activas=m.activas_burda.mean(),
                           kl_total=m.kl_total.mean(), mejor_epoca=m.mejor_epoca.mean(),
                           nll_por_columna=m.nll_media_por_columna.mean(), frac_piso=m.frac_logvar_dec_en_piso.mean())]), 3)]
    info = PROC / "datos_info.json"
    if info.exists():
        d = json.load(open(info, encoding="utf-8"))
        (OE1 / "datos_info.json").write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        lines += ["", f"- Filas eliminadas por inf/NaN: {d['eliminadas_inf_nan']}. Filtro D4 por partición: "
                  + ", ".join(f"{k} {v['eliminadas']}" for k, v in d["filtro_d4"].items()) + ".",
                  "- Particiones frente al pipeline original: "
                  + ", ".join(f"{k}: {v['identicas'] if isinstance(v, dict) else v}" for k, v in d["verificacion_vs_original"].items())
                  + ". (Copia en `outputs/oe1_saneado/datos_info.json`.)"]
    return lines


def oe2_robusto():
    det = pd.read_csv(ROB / "tables" / "phase4_detectability_metrics.csv")
    d = det[det.score == "nll_mc"]
    a = d.groupby("group").auc_roc.agg(["mean", "std", "min", "max"]).reset_index().sort_values("mean")
    act = pd.read_csv(ROB / "tables" / "phase2_active_dims_per_seed.csv")
    lines = [f"Semillas: {', '.join(map(str, sorted(act.seed)))}. Dimensiones activas por semilla (benigno de test): "
             + ", ".join(f"{int(s)}→{int(n)}" for s, n in zip(act.seed, act.n_active)) + ".", "",
             "**AUC-ROC (nll_mc, L = 20) por grupo**, media entre semillas:", "", md_table(a, 4), ""]
    old = ROB51 / "tables" / "phase4_detectability_metrics.csv"
    if old.exists():
        o = pd.read_csv(old)
        o = o[o.score == "nll_mc"]
        n_old = o.seed.nunique()
        o = o.groupby("group").auc_roc.mean()
        n = d.groupby("group").auc_roc.mean()
        c = pd.DataFrame({"auc_51": o, "auc_66": n}).dropna()
        c["dif"] = c.auc_66 - c.auc_51
        lines += [f"**Frente al modelo de 51 variables** (tabla de la fase 4 del OE2 robusto: {n_old} semilla(s); "
                  "la comparación con 10 semillas está en la sección 6):", "",
                  md_table(c.reset_index().rename(columns={"index": "group"}), 4), ""]
    tau = ROB / "tables" / "phase4_tau_per_seed.csv"
    if tau.exists():
        t = pd.read_csv(tau)
        lines.append(f"- τ (p95 de val_benign): FPR realizada en test_benign media {t.fpr_test_benign_at_tau_p95.mean():.4f}.")
    h1p = ROB / "metrics" / "phase5_hypothesis_principal.json"
    if h1p.exists():
        h1 = json.load(open(h1p, encoding="utf-8"))
        c = h1["combined"]
        lines.append(f"- Hipótesis principal ({h1['pair']}): ρ combinado **{f(c['rho_combined'])}** "
                     f"IC95 [{f(c['ci_low'])}, {f(c['ci_high'])}], {c['n_seeds']} semillas.")
    mm = ROB / "metrics" / "phase5_mixed_model.json"
    if mm.exists():
        m = json.load(open(mm, encoding="utf-8"))
        lines.append(f"- Modelo mixto a nivel de conexión: convergió={m.get('converged')}, efecto fijo "
                     f"{f(m.get('fixed_effect_maha_std'))} [{f(m.get('fixed_effect_ci_low'))}, {f(m.get('fixed_effect_ci_high'))}], "
                     f"p={f(m.get('fixed_effect_p_value'), 4)}, R² marginal {f(m.get('r2_marginal'))}.")
    kw = ROB / "tables" / "phase5_seed_stability_kendalls_w.csv"
    if kw.exists():
        k = pd.read_csv(kw)
        lines.append("- W de Kendall entre semillas: " + ", ".join(f"{r.metric} {r.kendalls_w:.3f}" for r in k.itertuples()) + ".")
    ver = sorted((ROB / "latent_vectors").glob("seed*/verificacion_conteos_vs_oe2_original.json"))
    if ver:
        res = {p.parent.name: json.load(open(p, encoding="utf-8")) for p in ver}
        json.dump(res, open(ROB / "verificacion_conteos.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        ok = all(v["identicos"] for v in res.values())
        lines.append(f"- Conteos por grupo frente al OE2 original: {'idénticos en todas las semillas' if ok else 'HAY DIFERENCIAS'} "
                     "(`outputs/oe2_robust_saneado/verificacion_conteos.json`).")
    return lines


def oe2_descriptores():
    v = pd.read_csv(DES / "tables" / "valoracion_descriptores.csv")
    keep = v[v.nivel.isin(["P", "S"])][["id", "descriptor", "nivel", "rho", "ci_lo", "ci_hi", "p_perm", "p_holm",
                                        "kendall_w", "rho_sin_portscan_bot", "rho_attempted", "tier"]]
    lines = ["Cribado (fase 3) en los modelos saneados; mismo diseño que con 51 variables:", "", md_table(keep, 3), ""]
    if (DES51 / "tables" / "valoracion_descriptores.csv").exists():
        o = pd.read_csv(DES51 / "tables" / "valoracion_descriptores.csv").set_index("columna")
        v2 = v.set_index("columna")
        common = [c for c in v2.index if c in o.index]
        c = pd.DataFrame({"id": v2.loc[common, "id"], "rho_51": o.loc[common, "rho"], "rho_66": v2.loc[common, "rho"]})
        c["dif"] = c.rho_66 - c.rho_51
        lines += ["**ρ de cada descriptor, 51 frente a 66 variables** (todos los del catálogo):", "",
                  md_table(c.reset_index(), 3), ""]
    ver = DES / "tables" / "verificacion_auc_seed42.json"
    if ver.exists():
        lines.append(f"- Fase 1 frente a la fase 4 robusta (semilla 42): diferencia máxima de AUC "
                     f"{json.load(open(ver))['max_abs_diff_auc_seed42_vs_phase4']:.2e}.")
    dg = DES / "tables" / "diagnosticos_por_semilla.csv"
    if dg.exists():
        g = pd.read_csv(dg)
        lines.append(f"- Rango efectivo de la métrica del decodificador G (mediana por semilla): "
                     f"{g.erank_G_mediana.min():.2f} a {g.erank_G_mediana.max():.2f}. Log σ² del decodificador en el piso: "
                     f"{100 * g.frac_logvar_dec_en_piso.min():.0f} a {100 * g.frac_logvar_dec_en_piso.max():.0f} %.")
    return lines


def confirmatoria():
    p = DES / "RESULTADOS_CONFIRMATORIA.md"
    text = p.read_text(encoding="utf-8").splitlines()
    return [ln.replace("## ", "### ", 1) if ln.startswith("## ") else ln for ln in text if not ln.startswith("# ")]


def archivos():
    paths = [p for p in COMMIT_PATHS if (REPO / p).exists()]
    (OUT / "ARCHIVOS_PARA_COMMIT.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    size = {}
    for p in paths:
        q = REPO / p
        size[p] = sum(x.stat().st_size for x in q.rglob("*") if x.is_file()) if q.is_dir() else q.stat().st_size
    return ["Rutas a versionar (`ARCHIVOS_PARA_COMMIT.txt`, las usa `scripts/vm_preparar_commit.sh`):", "",
            md_table(pd.DataFrame({"ruta": list(size), "MB": [s / 2 ** 20 for s in size.values()]}), 2), "",
            "No se versionan (pesados y regenerables): `outputs/oe2_robust_saneado/latent_vectors/`, "
            "`outputs/oe2_robust_saneado/tables/*.npz` y `_cache/`, `outputs/oe2_descriptores_saneado/_cache/`, "
            "`data/`."]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Corrida de pendientes en la VM: OE1 saneado y OE2 recalculado", "",
             f"Generado por `scripts/vm_resumen_corrida.py` el {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. "
             "Log cronológico: `RUN_LOG.md`. Todas las cifras salen de los CSV/JSON de `outputs/`; ninguna se escribe a mano.", ""]
    lines += section("1. Estado de las etapas", etapas)
    lines += section("2. Entorno", entorno)
    lines += section("3. OE1 saneado: reentrenamiento y diagnósticos", oe1)
    lines += section("4. OE2 robusto sobre los modelos saneados", oe2_robusto)
    lines += section("5. OE2: valoración de descriptores sobre los modelos saneados", oe2_descriptores)
    lines += section("6. OE2: prueba confirmatoria (hipótesis registradas el 10-10-2026)", confirmatoria)
    lines += section("7. Archivos para el commit", archivos)
    msg = ["VM: OE1 saneado (8 semillas + referencia 74) y OE2 recalculado sobre 66 variables", "",
           "Corrida de scripts/run_pendientes_vm.sh. Resumen en outputs/vm_pendientes/RESUMEN_CORRIDA.md,",
           "log en outputs/vm_pendientes/RUN_LOG.md.", "",
           "- OE1: reentrenamiento, diagnósticos, figuras y tablas del informe",
           "- OE2 robusto (perfil saneado): fases 2-6 en outputs/oe2_robust_saneado/",
           "- OE2 descriptores (perfil saneado): fases 1-7 en outputs/oe2_descriptores_saneado/",
           "- Fase 7: prueba confirmatoria (primario + 6 secundarios con Holm)"]
    (OUT / "MENSAJE_COMMIT.txt").write_text("\n".join(msg) + "\n", encoding="utf-8")
    (OUT / "RESUMEN_CORRIDA.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[resumen] {OUT / 'RESUMEN_CORRIDA.md'}")


if __name__ == "__main__":
    main()
