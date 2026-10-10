"""Tablas LaTeX y macros de cifras del informe del OE1 saneado, generadas desde los CSV/JSON de
outputs/oe1_saneado/ (ninguna cifra del informe se escribe a mano). Copia además las tablas fuente.

Uso:
    python -m vae_nids.oe1.tablas_informe --config configs/oe1_saneado.yaml --dest <carpeta del informe>
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vae_nids.oe1.config import load_config, repo_path, resolve


def n(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{d}f}".replace(".", "{,}")


def ent(x):
    return f"{int(x):,}".replace(",", r"\,")


def esc(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("^", r"\^{}")


def tab(head, cols, rows, long=False, caption=None, label=None):
    if long:
        return ("\\begin{longtable}{@{}" + cols + "@{}}\n" + (f"\\caption{{{caption}}}\\label{{{label}}}\\\\\n" if caption else "")
                + "\\toprule\n" + head + " \\\\\n\\midrule\n\\endfirsthead\n\\toprule\n" + head + " \\\\\n\\midrule\n\\endhead\n"
                + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}\n")
    return "\\begin{tabular}{@{}" + cols + "@{}}\n\\toprule\n" + head + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def main(config, dest):
    cfg = load_config(config)
    out = resolve(cfg, "salidas")
    sel, pre, tabd = out / "seleccion", out / "preproceso", out / "tables"
    dest = Path(dest)
    T = dest / "tablas"
    T.mkdir(parents=True, exist_ok=True)
    m = {}
    est = pd.read_csv(sel / "estado_columnas.csv")
    fs = yaml.safe_load(open(repo_path(cfg["columnas"]["conjunto"]), encoding="utf-8"))
    res = json.load(open(sel / "resumen_seleccion.json", encoding="utf-8"))
    err = pd.read_csv(sel / "error_sobre_tolerancia.csv").set_index("columna")
    m["nUniverso"], m["nQuedan"] = str(res["n_universo"]), str(res["n_quedan"])
    m["nRcero"] = str(res["eliminadas_por_regla"].get("R0", 0)); m["nRuno"] = str(res["eliminadas_por_regla"].get("R1", 0))
    m["nRdos"] = str(res["eliminadas_por_regla"].get("R2", 0)); m["nCasi"] = str(len(res["casi_funciones"]))
    m["hashCorto"] = res["sha256"][:12]; m["hashLargo"] = res["sha256"]
    m["hashPartido"] = r"\allowbreak{}".join(res["sha256"][i:i + 16] for i in range(0, 64, 16))
    m["nAlternativo"] = str(res["n_quedan"] - len(res["casi_funciones"]))

    # eliminadas
    rows = []
    for r in est[est.estado == "sale"].sort_values(["regla", "columna"]).itertuples():
        e = err.loc[r.columna] if r.columna in err.index else None
        ratio = n(e.ratio, 2) if e is not None else "0"
        ulp = n(e.ulp, 0) if e is not None else "0"
        formula = esc(r.formula if r.regla == "R2" else r.formula.replace("= ", "copia de "))
        rows.append(f"{esc(r.columna)} & {r.regla} & {formula} & {n(100 * r.frac_verificada, 1)} & {ulp} & {ratio} \\\\")
    (T / "tabla_eliminadas.tex").write_text(tab(
        r"Columna & Regla & Fórmula (solo con columnas que quedan) & \% verif. & Error máx. (ulp) & máx. error/$\varepsilon$",
        r">{\raggedright\arraybackslash}p{3.1cm}c>{\raggedright\arraybackslash}p{5.4cm}rrr", rows), encoding="utf-8")
    # casi funciones
    casi = pd.read_csv(sel / "casi_funciones.csv")
    rows = [f"{esc(r.columna)} & {esc(r.formula)} & {ent(r.fallos)} ({n(100 * (1 - r.frac_verificada), 2)}\\,\\%) & "
            f"{n(r.error_rel_max_ulp, 0)} & {n(err.loc[r.columna].ratio, 1)} \\\\" for r in casi.itertuples()]
    (T / "tabla_casi.tex").write_text(tab(r"Columna & Fórmula & Filas que fallan & Error máx. (ulp) & máx. error/$\varepsilon$",
                                          r">{\raggedright\arraybackslash}p{3cm}>{\raggedright\arraybackslash}p{4.6cm}rrr", rows), encoding="utf-8")
    m["casiFallosMin"] = n(100 * (1 - casi.frac_verificada.max()), 3); m["casiFallosMax"] = n(100 * (1 - casi.frac_verificada.min()), 1)
    m["casiUlpMin"] = n(casi.error_rel_max_ulp.min(), 0); m["casiUlpMax"] = n(casi.error_rel_max_ulp.max(), 0)
    # casi constantes
    e = pd.read_csv(sel / "entropia_por_columna.csv")
    cc = e[e.casi_constante]
    rows = [f"{esc(r.columna)} & {r.valores_unicos} & {n(100 * r.frac_moda, 3)} & {r.H_bits:.1e} \\\\".replace("e-0", "e$-$").replace(".", "{,}", 1)
            for r in cc.itertuples()]
    (T / "tabla_casi_constantes.tex").write_text(tab(r"Columna & Valores únicos & Moda (\%) & $H$ (bits)", "lrrr", rows), encoding="utf-8")
    m["nCasiConst"] = str(len(cc))
    # comprobación posterior
    post = pd.read_csv(sel / "comprobacion_posterior.csv")
    g = post.groupby("particion").agg(filas=("filas", "sum"), fallos=("fallos", "sum"), grupos=("grupo", "nunique"))
    rows = [f"{esc(p)} & {r.grupos} & {ent(r.filas)} & {ent(r.fallos)} \\\\" for p, r in g.iterrows()]
    (T / "tabla_post.tex").write_text(tab(r"Partición & Grupos & Comprobaciones (filas $\times$ identidades) & Fallos", "lrrr", rows), encoding="utf-8")
    m["postComprobaciones"] = ent(g.filas.sum()); m["postFallos"] = ent(g.fallos.sum())
    m["postEtiquetas"] = str(int(g.loc["test_attacks", "grupos"]))
    # redundancia top 20
    top = pd.read_csv(sel / "redundancia_top20.csv")
    rows = [f"{esc(r.X)} & {esc(r.Y)} & {n(r.u_X_dado_Y, 3)} & {n(r.u_Y_dado_X, 3)} & {n(r.mi_corr_bits, 3)} \\\\" for r in top.itertuples()]
    (T / "tabla_top20.tex").write_text(tab(r"$X$ & $Y$ & $u(X|Y)$ & $u(Y|X)$ & $I$ corregida (bits)", "llrrr", rows), encoding="utf-8")
    # comparación de listas
    info = json.load(open(sel / "informacion_resumen.json", encoding="utf-8"))
    names = {"nuevo": "Nuevo (66)", "actual_51": "Actual (51)", "B_prima_51": "B$'$ (51)", "Robusto_TB_56": "Robusto TB (56)",
             "info_nula_eps_ampliado_60": r"$\varepsilon$ ampliado (60)"}
    rows = []
    for k, v in info["comparaciones"].items():
        a, b = k.split("__vs__")
        if a != "nuevo" and b != "nuevo":
            continue
        rows.append(f"{names[a]} frente a {names[b]} & {v['en_comun']} & {len(v['solo_en_primero'])} & {len(v['solo_en_segundo'])} \\\\")
    (T / "tabla_comparacion.tex").write_text(tab(r"Comparación & En común & Solo en el 1.º & Solo en el 2.º", "lrrr", rows), encoding="utf-8")
    c51 = info["comparaciones"]["nuevo__vs__actual_51"]
    (T / "lista_vuelven.tex").write_text(", ".join(esc(c) for c in c51["solo_en_primero"]) + ".", encoding="utf-8")
    (T / "lista_salen.tex").write_text(", ".join(esc(c) for c in c51["solo_en_segundo"]) + ".", encoding="utf-8")
    m["nComunActual"] = str(c51["en_comun"]); m["nVuelven"] = str(len(c51["solo_en_primero"])); m["nSalenActual"] = str(len(c51["solo_en_segundo"]))
    # preprocesamiento
    dinfo = json.load(open(resolve(cfg, "processed_dir") / "datos_info.json", encoding="utf-8"))
    rp = json.load(open(pre / "resumen_preproceso.json", encoding="utf-8"))
    rows = []
    for p in ("train_benign", "val_benign", "test_benign", "test_attacks"):
        f = dinfo["filtro_d4"][p]
        v = dinfo["verificacion_vs_original"][p]
        rows.append(f"{esc(p)} & {ent(f['filas_antes'])} & {ent(f['eliminadas'])} & {ent(f['filas_despues'])} & "
                    f"{'sí' if v['identicas'] else 'NO'} & {ent(dinfo['flow_iat_min_negativos'][p])} & \\texttt{{{f['hash_ids_despues'][:10]}}} \\\\")
    (T / "tabla_filtro.tex").write_text(tab(r"Partición & Filas antes & Filtradas (D4) & Filas después & ¿Igual al original? & Flow IAT Min $<0$ & Hash ids",
                                            "lrrrcrl", rows), encoding="utf-8")
    m["filasCrudas"] = ent(dinfo["filas_crudas"]); m["filasInf"] = ent(dinfo["eliminadas_inf_nan"])
    m["dcuatroTrain"] = str(dinfo["filtro_d4"]["train_benign"]["eliminadas"]); m["dcuatroVal"] = str(dinfo["filtro_d4"]["val_benign"]["eliminadas"])
    m["dcuatroTotal"] = str(sum(v["eliminadas"] for v in dinfo["filtro_d4"].values()))
    m["iatNegTrain"] = ent(dinfo["flow_iat_min_negativos"]["train_benign"])
    resol = pd.read_csv(pre / "resolucion_por_columna.csv")
    mk = resol[resol.marcada]
    rows = [f"{esc(r.columna)} & {ent(r.valores_distintos_crudo)} & {ent(r.valores_distintos_almacenados)} & {n(100 * r.frac_conservada, 1)} & "
            f"{n(r.resolucion_en_maximo, 1)} & {n(r.resolucion_en_mediana, 4)} \\\\" for r in mk.itertuples()]
    (T / "tabla_resolucion.tex").write_text(tab(r"Columna & \makecell{Distintos\\(crudo)} & \makecell{Distintos\\(float32)} & \makecell{\%\\conservado} & \makecell{Resol. en\\el máx.} & \makecell{Resol. en\\la mediana}",
                                                "lrrrrr", rows), encoding="utf-8")
    m["nMarcadas"] = str(len(mk)); m["minConservado"] = n(100 * resol.frac_conservada.min(), 1)
    rad = pd.read_csv(pre / "resolucion_antes_despues.csv")
    m["fpsAntes"] = n(100 * rad.frac.iloc[0], 1); m["fpsDespues"] = n(100 * rad.frac.iloc[1], 1)
    # anexo: 77 columnas
    prec = {c: d.get("precision_csv", {}) for c, d in fs["columnas"].items()}
    rows = []
    for r in est.itertuples():
        st = "queda" if r.estado == "queda" else "sale"
        if r.estado == "queda" and isinstance(getattr(r, "casi_funcion", None), str):
            st = "queda (casi función)"
        eps = r.eps if isinstance(r.eps, str) else (r.eps_casi if isinstance(getattr(r, "eps_casi", None), str) else "--")
        fv = n(100 * r.frac_verificada, 2) if np.isfinite(r.frac_verificada) else "--"
        p = prec.get(r.columna) or {}
        rows.append(f"{esc(r.columna)} & {st} & {r.regla if isinstance(r.regla, str) else '--'} & {esc(eps)} & {fv} & "
                    f"{p.get('max_cifras_significativas', '--')} & {'sí' if p.get('entera') else 'no'} \\\\")
    (T / "tabla_77.tex").write_text(tab(r"Columna & Estado & Regla & $\varepsilon$ (por fila) & \% verif. & Cifras CSV & Entera",
                                        r">{\raggedright\arraybackslash}p{3.6cm}>{\raggedright\arraybackslash}p{2.3cm}c>{\raggedright\arraybackslash}p{3.6cm}rrc",
                                        rows, long=True, caption="Las 77 columnas de flujo: estado, regla, tolerancia y precisión en el CSV (muestra de 300\\,000 filas del lunes).",
                                        label="tab:77"), encoding="utf-8")
    dep = pd.read_csv(sel / "busqueda_sistematica_dependencias.csv").sort_values("soporte_Y", ascending=False)
    rows = [f"{esc(r.X)} & {esc(r.Y)} & {n(r.soporte_Y, 3)} & {ent(r.valores_distintos_Y)} & {ent(r.valores_distintos_X)} \\\\" for r in dep.itertuples()]
    (T / "tabla_dependencias.tex").write_text(tab(r"$X$ & $Y$ & Soporte de $Y$ & Valores de $Y$ & Valores de $X$", "llrrr", rows), encoding="utf-8")
    m["nDependencias"] = str(len(dep))
    # entrenamiento (si existe)
    dpath = tabd / "diagnosticos_por_semilla.csv"
    if dpath.exists():
        d = pd.read_csv(dpath)
        ant = pd.read_csv(tabd / "diagnosticos_oe1_actual_51.csv")
        rows = []
        for r in pd.concat([d, ant]).itertuples():
            rows.append(f"{esc(r.corrida)} & {r.seed} & {r.n_columnas} & {r.mejor_epoca} & {r.epoca_parada} & {n(r.mejor_val_total, 2)} & "
                        f"{r.activas_burda} & {r.activas_kl} & {n(r.kl_total, 2)} & {n(r.tau_p95, 1)} & {n(r.nll_media_por_columna, 3)} & {r.columnas_en_piso} \\\\")
        (T / "tabla_diag.tex").write_text(tab(r"Corrida & Semilla & Col. & Mejor ép. & Parada & Mejor val & Activas (Var) & Activas (KL) & KL & $\tau$ & NLL/col. & En piso",
                                              "lrrrrrrrrrrr", rows), encoding="utf-8")
        new = d[d.corrida == "oe1_info_nula"]
        ref = d[d.corrida == "referencia_74"]

        def ms(s, k=1):
            return f"{n(s.mean(), k)} $\\pm$ {n(s.std(), k)}" if len(s) > 1 else n(s.mean(), k)
        rows = []
        for lab, t in [("Actual (51, pipeline original)", ant), ("Saneado (66)", new), ("Referencia 74 (preproc. corregido)", ref)]:
            if len(t) == 0:
                continue
            rows.append(f"{lab} & {int(t.n_columnas.iloc[0])} & {len(t)} & {ms(t.activas_burda)} & {ms(t.mejor_epoca, 0)} & "
                        f"{ms(t.columnas_en_piso, 1)} & {ms(t.tau_p95, 1)} & {ms(t.nll_media_por_columna, 3)} \\\\")
        (T / "tabla_antes_despues.tex").write_text(tab(r"Conjunto & Col. & Semillas & Unidades activas & Mejor época & Col. en el piso & $\tau$ (p95) & NLL media por columna",
                                                       "lrrrrrrr", rows), encoding="utf-8")
        m["activasMedia"] = n(new.activas_burda.mean(), 1); m["activasDE"] = n(new.activas_burda.std(), 1)
        m["activasAntes"] = n(ant.activas_burda.mean(), 1)
        m["epocaMin"] = str(int(new.mejor_epoca.min())); m["epocaMax"] = str(int(new.mejor_epoca.max()))
        m["pisoFrac"] = n(100 * new.frac_logvar_dec_en_piso.mean(), 1)
        m["nSemillasNuevo"] = str(len(new))
        m["reentrenamiento"] = "completo"
    else:
        m["reentrenamiento"] = "pendiente"
    (T / "cifras.tex").write_text("% Generado por vae_nids.oe1.tablas_informe: no editar a mano.\n"
                                  + "\n".join(f"\\newcommand{{\\cifra{k}}}{{{v}}}" for k, v in m.items()) + "\n", encoding="utf-8")
    # tablas fuente
    tf = dest / "tablas_fuente"
    tf.mkdir(exist_ok=True)
    for f in list(sel.glob("*.csv")) + list(sel.glob("*.json")) + list(pre.glob("*")) + list(tabd.glob("*.csv")):
        if f.name == "busqueda_sistematica_dependencias.csv" or f.stat().st_size < 20_000_000:
            shutil.copy(f, tf / f.name)
    shutil.copy(resolve(cfg, "processed_dir") / "datos_info.json", tf / "datos_info.json")
    shutil.copy(repo_path(cfg["columnas"]["conjunto"]), tf / "oe1_info_nula.yaml")
    for f in repo_path("configs/features/alternativos").glob("*.yaml"):
        shutil.copy(f, tf / f.name)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    print(main(a.config, a.dest))
