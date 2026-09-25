"""Corre OE1-IM completo, fase por fase (solo análisis; no reentrena nada).

Uso: python -m vae_nids.evaluation.oe1_mi.run_all [--skip-ksg]

Tiempos aproximados (CPU 8 núcleos): fase 0 ~1 min (incluye construir la
caché del dataset EDA la primera vez, ~3 min), fase 1 ~1 min, fase 2 ~1 min,
KSG ~6 min, fase 3 ~12 min, fases 4-5 < 1 min.
"""
import argparse

from vae_nids.evaluation.oe1_mi import (phase0_recon, phase1_validation, phase2_entropy,
                                        phase3_ksg, phase3_matrices, phase4_criteria,
                                        phase5_contrast)
from vae_nids.evaluation.oe1_mi.phase3_ksg import OUT_PATH as KSG_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ksg", action="store_true",
                        help="reutilizar metrics/oe1_ksg_pares.csv si existe")
    args = parser.parse_args()
    phase0_recon.main()
    phase1_validation.main()
    phase2_entropy.main()
    if not (args.skip_ksg and KSG_PATH.exists()):
        phase3_ksg.main()
    phase3_matrices.main()
    phase4_criteria.main()
    phase5_contrast.main()


if __name__ == "__main__":
    main()
