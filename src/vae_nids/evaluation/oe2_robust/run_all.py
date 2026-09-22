"""Orquestador de OE2 robusto -- corre las Fases 0-6 en orden, imprime el
resumen final pedido en la Sección 10 del prompt (número de semillas,
resultado de la hipótesis principal, ruta del informe).

Pensado para correr en segundo plano (entrenar 9 semillas nuevas toma del
orden de una hora en CPU, según el presupuesto de la Fase 0). Cada fase es
también invocable por separado como módulo (`python -m
vae_nids.evaluation.oe2_robust.phaseN_...`).

Uso:
    python -m vae_nids.evaluation.oe2_robust.run_all
"""
import json
import time

from vae_nids.evaluation.oe2_robust import config as rcfg
from vae_nids.evaluation.oe2_robust import (
    phase0_diagnostics,
    phase1_train_multiseed,
    phase2_encode,
    phase3_geometry,
    phase4_detectability,
    phase5_correlation,
    phase6_figures,
)


def main():
    t0 = time.perf_counter()

    print("=" * 70)
    print("FASE 0 -- diagnóstico y presupuesto")
    print("=" * 70)
    phase0_diagnostics.main()

    print("=" * 70)
    print("FASE 1 -- entrenamiento multi-semilla")
    print("=" * 70)
    phase1_train_multiseed.main()

    print("=" * 70)
    print("FASE 2 -- codificación (mu, log sigma^2)")
    print("=" * 70)
    phase2_encode.main()

    print("=" * 70)
    print("FASE 3 -- geometría distribucional")
    print("=" * 70)
    phase3_geometry.main()

    print("=" * 70)
    print("FASE 4 -- detectabilidad confirmatoria")
    print("=" * 70)
    phase4_detectability.main()

    print("=" * 70)
    print("FASE 5 -- correlación geometría-detectabilidad")
    print("=" * 70)
    phase5_correlation.main()

    print("=" * 70)
    print("FASE 6 -- visualización (UMAP pendiente, se corre aparte)")
    print("=" * 70)
    phase6_figures.main(include_umap=False)

    total_minutes = (time.perf_counter() - t0) / 60.0

    with open(rcfg.METRICS_DIR / "phase5_hypothesis_principal.json", encoding="utf-8") as f:
        h1 = json.load(f)

    print("=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"Semillas usadas: {rcfg.ALL_SEEDS} (n={len(rcfg.ALL_SEEDS)})")
    print(f"Hipótesis principal ({h1['pair']}):")
    print(f"  rho combinado = {h1['combined']['rho_combined']:.4f} "
          f"IC95=[{h1['combined']['ci_low']:.4f},{h1['combined']['ci_high']:.4f}]")
    print(f"  OE2 original (seed=42 solo): rho={h1['previous_oe2_single_seed']['rho']:.4f} "
          f"p={h1['previous_oe2_single_seed']['p_value']:.4f}")
    print(f"Tiempo total del pipeline: {total_minutes:.1f} min")
    print(f"Informe: {rcfg.REPORTS_DIR / 'INFORME_OE2_ROBUSTO.md'}")


if __name__ == "__main__":
    main()
