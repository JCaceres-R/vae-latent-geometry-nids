# OE2 robusto -- cómo reproducir

Pipeline nuevo (`src/vae_nids/evaluation/oe2_robust/`), aislado del OE2
original de una sola semilla. No modifica ni sobrescribe nada de
`outputs/checkpoints`, `outputs/latent_vectors`, `outputs/metrics` ni
`outputs/figures` (los del OE2 original) -- todo lo nuevo vive bajo
`outputs/oe2_robust/`.

## Requisitos

Entorno del proyecto (`.venv`) más `statsmodels` y `psutil`:

```
python -m pip install statsmodels psutil
```

`umap-learn` ya es dependencia del proyecto (Fase 6).

## Orden de ejecución

Todo de una vez (Fases 0-6, ~1-2 horas en CPU, ver Fase 0 del informe para
el presupuesto observado):

```
python -m vae_nids.evaluation.oe2_robust.run_all
```

O fase por fase (cada una es re-ejecutable de forma independiente una vez
que las anteriores dejaron sus artefactos):

```
python -m vae_nids.evaluation.oe2_robust.phase0_diagnostics
python -m vae_nids.evaluation.oe2_robust.phase1_train_multiseed
python -m vae_nids.evaluation.oe2_robust.phase2_encode
python -m vae_nids.evaluation.oe2_robust.phase3_geometry
python -m vae_nids.evaluation.oe2_robust.phase4_detectability
python -m vae_nids.evaluation.oe2_robust.phase5_correlation
python -m vae_nids.evaluation.oe2_robust.phase6_figures
```

Cada fase acepta `--seed N` para correr una sola semilla (útil para
depurar o para el primer chequeo de consistencia contra el OE2 original
sobre seed=42 antes de lanzar el barrido completo):

```
python -m vae_nids.evaluation.oe2_robust.phase2_encode --seed 42
python -m vae_nids.evaluation.oe2_robust.phase3_geometry --seed 42
```

## Tests

```
python -m pytest tests/oe2_robust/ -q
```

Incluye las primitivas estadísticas (DeLong contra un valor conocido,
Bhattacharyya=0 entre gaussianas idénticas, MMD≈0 entre la misma muestra) y
la verificación de conteos del manifest de la Fase 2 contra el manifest del
OE2 original (se salta si la Fase 2 no se ha corrido todavía).

## Estructura de salidas

```
outputs/oe2_robust/
  checkpoints/        -- un .pt + _metadata.json por semilla nueva (42 reutiliza el oficial)
  logs/                -- curva de entrenamiento por semilla (CSV)
  latent_vectors/
    seed{S}/
      latent_*_mu.npy, latent_*_logvar.npy, manifest.json
      scores/*.npz      -- scores crudos por conexión (Fase 4, reusados por la Fase 5)
  metrics/             -- JSON de cada fase (diagnóstico, hipótesis principal, modelo mixto, ...)
  tables/              -- CSV en formato largo (seed, group, metric, value, ci_low, ci_high) por fase
  figures/              -- 7 figuras obligatorias + UMAP por semilla representativa (PNG 300dpi + PDF)
  INFORME_OE2_ROBUSTO.md
  README.md            -- este archivo
```

## Reproducibilidad

Cada checkpoint nuevo guarda un `_metadata.json` con: semilla, época de
parada, val_loss, tiempo de entrenamiento, hash del commit de git, y
versiones de python/torch/numpy/pandas/sklearn/scipy. Las semillas de
numpy/torch/random/sklearn/UMAP están fijadas explícitamente en cada
función que las usa (ver `config.py`: `MC_SEED_BASE`,
`N_BOOTSTRAP_*_SEED`, `PERMUTATION_SEED`, `UMAP_RANDOM_STATE`, etc.), no
dejadas al azar global del proceso.
