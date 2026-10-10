"""Punto de entrada único del OE1 saneado. Ejecuta en orden y es reanudable:

  1. datos (si no existen las particiones crudas)      -> vae_nids.oe1.datos
  2. selección por información nula (si no está congelada) -> vae_nids.oe1.seleccion
  3. preprocesamiento y resolución                      -> vae_nids.oe1.preproceso
  4. entrenamiento: semillas principales con el conjunto congelado y la referencia de 74
     (si existe el checkpoint final de una semilla, se salta)
  5. diagnósticos y figuras (solo benigno)

Uso:
    python -m vae_nids.oe1.run --config configs/oe1_saneado.yaml [--ref74 configs/oe1_saneado_ref74.yaml]
"""
import argparse
import datetime as dt
import json

from vae_nids.oe1 import datos, diagnosticos, entrenamiento, preproceso, seleccion
from vae_nids.oe1.config import load_config, repo_path, resolve


def log(path, msg):
    line = f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main(config, ref74=None):
    cfg = load_config(config)
    out = resolve(cfg, "salidas")
    out.mkdir(parents=True, exist_ok=True)
    lp = out / "registro_run.log"
    if not (resolve(cfg, "processed_dir") / "train_benign_crudo.parquet").exists():
        log(lp, "datos: construyendo particiones crudas"); datos.build(cfg)
    if not repo_path(cfg["columnas"]["conjunto"]).exists():
        log(lp, "selección: congelando conjunto"); seleccion.main(config)
    log(lp, "preproceso"); preproceso.main(config)
    runs = [(cfg, cfg["semillas"]["principal"])]
    if ref74:
        c74 = load_config(ref74)
        runs.append((c74, c74["semillas"]["referencia_74"]))
    for c, seeds in runs:
        for s in seeds:
            name = f"{c['entrenamiento']['nombre_corrida']}_seed{s}"
            if (resolve(c, "salidas") / "checkpoints" / f"{name}.pt").exists():
                log(lp, f"{name}: ya existe, se salta"); continue
            log(lp, f"{name}: entrenando")
            _, meta = entrenamiento.train(c, s)
            log(lp, f"{name}: parada en época {meta['epoca_parada']} ({meta['motivo_parada']}), "
                    f"mejor época {meta['mejor_epoca']}, val {meta['mejor_val_total']:.4f}, {meta['segundos_totales']:.0f} s")
    log(lp, "diagnósticos")
    tab = diagnosticos.main(config, [ref74] if ref74 else [])
    log(lp, "listo: " + json.dumps({"corridas": int(len(tab))}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ref74", default=None)
    a = ap.parse_args()
    main(a.config, a.ref74)
