"""Resumen de la corrida OE3 + OE4 + reconciliación en la VM (scripts/run_oe3_oe4_vm.sh).

Escribe en outputs/vm_oe3_oe4/: RESUMEN_CORRIDA.md, entorno.json, pip_freeze.txt,
ARCHIVOS_PARA_COMMIT.txt y MENSAJE_COMMIT.txt. Reutiliza las funciones de vm_resumen_corrida.py.

Uso:
    python scripts/vm_resumen_oe3_oe4.py
"""
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

_spec = importlib.util.spec_from_file_location("vm_resumen_corrida", Path(__file__).with_name("vm_resumen_corrida.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

REPO = R.REPO
R.OUT = REPO / "outputs" / "vm_oe3_oe4"
# Se versionan tablas, figuras, logs y checkpoints (≈ 40–75 KB); el .gitignore excluye latentes y cachés.
R.COMMIT_PATHS = [
    "outputs/vm_oe3_oe4",
    "outputs/oe2_descriptores_saneado/tables", "outputs/oe2_descriptores_saneado/RESULTADOS_RECONCILIACION.md",
    "outputs/oe3",
    "outputs/oe4",
]


def embed(path):
    """Incrusta un RESULTADOS_*.md bajando un nivel sus títulos."""
    def fn():
        text = (REPO / path).read_text(encoding="utf-8").splitlines()
        return ([f"Fuente: `{path}`.", ""]
                + [("#" + ln) if ln.startswith("## ") else ln for ln in text if not ln.startswith("# ")])
    return fn


def main():
    R.OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Corrida en la VM: reconciliación (OE2), OE3 y OE4", "",
             f"Generado por `scripts/vm_resumen_oe3_oe4.py` el {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. "
             "Log cronológico: `RUN_LOG.md`. Todas las cifras salen de los CSV de `outputs/`.", ""]
    lines += R.section("1. Estado de las etapas", R.etapas)
    lines += R.section("2. Entorno", R.entorno)
    lines += R.section("3. OE2: reconciliación del nivel de conexión",
                       embed("outputs/oe2_descriptores_saneado/RESULTADOS_RECONCILIACION.md"))
    lines += R.section("4. OE3: calidad generativa", embed("outputs/oe3/RESULTADOS_OE3.md"))
    lines += R.section("5. OE4: reproducibilidad frente a β y k", embed("outputs/oe4/RESULTADOS_OE4.md"))
    lines += R.section("6. Archivos para el commit", R.archivos)
    msg = ["VM: reconciliación del nivel de conexión, OE3 (calidad generativa) y OE4 (β=0,5, β=2, k=16)", "",
           "Corrida de scripts/run_oe3_oe4_vm.sh. Resumen en outputs/vm_oe3_oe4/RESUMEN_CORRIDA.md,",
           "log en outputs/vm_oe3_oe4/RUN_LOG.md.", "",
           "- OE2: fase 8 (modelo mixto frente a nivel de flujo)",
           "- OE3: KS, Wasserstein, clasificador real/sintético y memorización en outputs/oe3/",
           "- OE4: 15 modelos, descriptores, confirmatoria y concordancia de rankings en outputs/oe4/"]
    (R.OUT / "MENSAJE_COMMIT.txt").write_text("\n".join(msg) + "\n", encoding="utf-8")
    (R.OUT / "RESUMEN_CORRIDA.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[resumen] {R.OUT / 'RESUMEN_CORRIDA.md'}")


if __name__ == "__main__":
    main()
