"""Registro de conjuntos alternativos (3.6.3). Solo se registran: no se entrenan en esta tarea.

  - B_prima_51 y Robusto_TB_56: tomados de tablas_fuente/ del informe OE1-IM v2
    (v6_opciones_listas.json -> listas["B'"]; v8_conjuntos.csv -> columna Robusto_TB == "sí").
  - info_nula_eps_ampliado_60: el conjunto congelado menos las 6 "casi funciones" (tasas, varianza y
    Down/Up Ratio), por si el autor decide aceptarlas con un ε ampliado.

Uso:
    python -m vae_nids.oe1.alternativos --v2 <carpeta informe_oe1_im_v2>
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from vae_nids.oe1.config import columns_hash, load_config, repo_path, resolve


def write(name, cols, procedencia):
    path = repo_path(f"configs/features/alternativos/{name}.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"nombre": name, "n": len(cols), "sha256": columns_hash(cols), "procedencia": procedencia,
           "nota": "Registrado para comparación; no se entrena en el saneamiento del OE1.", "columnas": cols}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, width=120)
    return path


def main(v2_dir, config=None):
    cfg = load_config(config)
    v2 = Path(v2_dir) / "tablas_fuente"
    listas = json.load(open(v2 / "v6_opciones_listas.json", encoding="utf-8"))["listas"]
    write("B_prima_51", listas["B'"], f"{v2 / 'v6_opciones_listas.json'} -> listas[\"B'\"] (informe OE1-IM v2, V6)")
    conj = pd.read_csv(v2 / "v8_conjuntos.csv", encoding="utf-8")
    rob = conj.loc[conj["Robusto_TB"].astype(str).str.startswith("s"), "columna"].tolist()
    write(f"Robusto_TB_{len(rob)}", rob, f"{v2 / 'v8_conjuntos.csv'} -> Robusto_TB == 'sí' (informe OE1-IM v2, V8)")
    fs = yaml.safe_load(open(repo_path(cfg["columnas"]["conjunto"]), encoding="utf-8"))
    casi = json.load(open(resolve(cfg, "salidas") / "seleccion" / "resumen_seleccion.json", encoding="utf-8"))["casi_funciones"]
    alt = [c for c in fs["columnas_que_quedan"] if c not in casi]
    write(f"info_nula_eps_ampliado_{len(alt)}", alt,
          "configs/features/oe1_info_nula.yaml menos las casi funciones de outputs/oe1_saneado/seleccion/resumen_seleccion.json")
    return len(listas["B'"]), len(rob), len(alt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    print(main(a.v2, a.config))
