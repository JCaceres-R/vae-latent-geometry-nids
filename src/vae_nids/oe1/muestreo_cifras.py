"""Control de calidad (9.5.3): muestrea cifras del PDF del informe y las compara con las tablas fuente.

Cada cifra esperada se calcula aquí directamente desde los CSV/JSON de outputs/ (no desde las macros
del informe) y se busca en el texto extraído del PDF.

Uso:
    python -m vae_nids.oe1.muestreo_cifras --config configs/oe1_saneado.yaml --pdf <informe.pdf>
"""
import argparse
import json
import re

import pandas as pd
import pypdfium2 as pdfium

from vae_nids.oe1.config import load_config, resolve


def norm(s):
    return re.sub(r"[\s   ]", "", s).replace("−", "-").replace("–", "-")


def fmt(x, d):
    return f"{x:.{d}f}".replace(".", ",")


def checks(cfg):
    out = resolve(cfg, "salidas")
    sel, pre, tab = out / "seleccion", out / "preproceso", out / "tables"
    res = json.load(open(sel / "resumen_seleccion.json", encoding="utf-8"))
    info = json.load(open(resolve(cfg, "processed_dir") / "datos_info.json", encoding="utf-8"))
    casi = pd.read_csv(sel / "casi_funciones.csv").set_index("columna")
    top = pd.read_csv(sel / "redundancia_top20.csv")
    resol = pd.read_csv(pre / "resolucion_por_columna.csv").set_index("columna")
    post = pd.read_csv(sel / "comprobacion_posterior.csv")
    rad = pd.read_csv(pre / "resolucion_antes_despues.csv")
    c = [
        ("columnas que quedan", f"{res['n_quedan']}columnas"),
        ("hash del conjunto (prefijo)", res["sha256"][:12]),
        ("filas crudas", str(info["filas_crudas"])),
        ("filas eliminadas por inf/NaN", str(info["eliminadas_inf_nan"])),
        ("train antes del filtro", str(info["filtro_d4"]["train_benign"]["filas_antes"])),
        ("train después del filtro", str(info["filtro_d4"]["train_benign"]["filas_despues"])),
        ("val después del filtro", str(info["filtro_d4"]["val_benign"]["filas_despues"])),
        ("ataques", str(info["filtro_d4"]["test_attacks"]["filas_despues"])),
        ("Flow IAT Min < 0 en train", str(info["flow_iat_min_negativos"]["train_benign"])),
        ("comprobaciones posteriores", str(int(post.filas.sum()))),
        ("fallos Fwd Packets/s", str(int(casi.loc["Fwd Packets/s", "fallos"]))),
        ("fallos Bwd Packets/s", str(int(casi.loc["Bwd Packets/s", "fallos"]))),
        ("fallos Down/Up Ratio", str(int(casi.loc["Down/Up Ratio", "fallos"]))),
        ("u(Protocol|Fwd Seg Size Min)", fmt(top.iloc[3].u_X_dado_Y, 3)),
        ("distintos crudos Flow Duration", str(int(resol.loc["Flow Duration", "valores_distintos_crudo"]))),
        ("distintos float32 Flow Duration", str(int(resol.loc["Flow Duration", "valores_distintos_almacenados"]))),
        ("Flow Packets/s antes (%)", fmt(100 * rad.frac.iloc[0], 1)),
        ("Flow Packets/s después (%)", fmt(100 * rad.frac.iloc[1], 1)),
        ("% conservado mínimo (Idle Mean)", fmt(100 * resol.frac_conservada.min(), 1)),
    ]
    dpath = tab / "diagnosticos_por_semilla.csv"
    if dpath.exists():
        d = pd.read_csv(dpath)
        s42 = d[(d.corrida == "oe1_info_nula") & (d.seed == 42)].iloc[0]
        main = d[d.corrida == "oe1_info_nula"]
        ant = pd.read_csv(tab / "diagnosticos_oe1_actual_51.csv")
        c += [("mejor época semilla 42", f"oe1_info_nula42{s42.n_columnas}{s42.mejor_epoca}{s42.epoca_parada}"),
              ("tau semilla 42", fmt(s42.tau_p95, 1)),
              ("NLL/col semilla 42", fmt(s42.nll_media_por_columna, 3)),
              ("unidades activas media", fmt(main.activas_burda.mean(), 1)),
              ("unidades activas antes", fmt(ant.activas_burda.mean(), 1))]
    return c


def main(config, pdf):
    cfg = load_config(config)
    doc = pdfium.PdfDocument(pdf)
    text = norm("".join(doc[i].get_textpage().get_text_range() for i in range(len(doc))))
    rows = []
    for name, val in checks(cfg):
        ok = norm(val) in text
        rows.append(dict(cifra=name, esperado=val, encontrada_en_pdf=ok))
    t = pd.DataFrame(rows)
    t.to_csv(resolve(cfg, "salidas") / "muestreo_cifras_informe.csv", index=False)
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--pdf", required=True)
    a = ap.parse_args()
    t = main(a.config, a.pdf)
    print(t.to_string())
    print(f"{int(t.encontrada_en_pdf.sum())}/{len(t)} cifras encontradas")
