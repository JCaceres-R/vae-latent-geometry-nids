# Commits sugeridos (8 de octubre de 2026)

No se hizo ningún commit ni push: todos los cambios son locales, en `main`. Agrupación sugerida (los datos,
los checkpoints y `outputs/**/_cache/` no se suben; ya cubre `.gitignore` lo de `data/processed/`).

Antes de commitear, agrega a `.gitignore`:

```
# OE2 descriptores y OE1 saneado: intermedios pesados y checkpoints
outputs/oe2_descriptores/_cache/
outputs/oe1_saneado/checkpoints/
outputs/oe1_saneado/humo/
```

## 1. OE2: evaluación de descriptores de la geometría latente

```
git add src/vae_nids/evaluation/oe2_descriptores/ outputs/oe2_descriptores/tables/ outputs/oe2_descriptores/figures/ outputs/oe2_descriptores/tablas_tex/
git commit -m "OE2 descriptores: evaluación de 33 descriptores de geometría latente en 10 semillas (cribado + validación attempted)"
```

## 2. OE1 saneado, fase 1: conjunto por información nula

```
git add src/vae_nids/oe1/__init__.py src/vae_nids/oe1/config.py src/vae_nids/oe1/datos.py src/vae_nids/oe1/seleccion.py src/vae_nids/oe1/informacion.py src/vae_nids/oe1/alternativos.py configs/features/ outputs/oe1_saneado/seleccion/ tests/oe1_saneado/
git commit -m "OE1-saneamiento fase 1: conjunto de entrada por información nula (77 -> 66), congelado con SHA-256"
```

## 3. OE1 saneado, fase 2: preprocesamiento corregido

```
git add src/vae_nids/oe1/preproceso.py outputs/oe1_saneado/preproceso/
git commit -m "OE1-saneamiento fase 2: filtro de duraciones negativas por partición, MinMax float64 y resolución"
```

## 4. OE1 saneado, fases 3 y 4: configuración única, checkpoints autocontenidos, hiperparámetros

```
git add configs/oe1_saneado.yaml configs/oe1_saneado_ref74.yaml configs/oe1_saneado_humo.yaml src/vae_nids/oe1/entrenamiento.py docs/oe1_saneado/hiperparametros.md
git commit -m "OE1-saneamiento fases 3-4: configuración única, checkpoints autocontenidos y tabla de hiperparámetros"
```

## 5. OE1 saneado, fase 5: reentrenamiento y diagnósticos (solo benigno)

```
git add src/vae_nids/oe1/diagnosticos.py src/vae_nids/oe1/antes.py src/vae_nids/oe1/run.py outputs/oe1_saneado/tables/ outputs/oe1_saneado/figures/ outputs/oe1_saneado/logs/ outputs/oe1_saneado/registro_run.log
git commit -m "OE1-saneamiento fase 5: reentrenamiento 8 semillas + referencia 74, diagnósticos benignos"
```

## 6. OE1 saneado, fase 6: informe

```
git add docs/oe1_saneado/INFORME_SANEAMIENTO_OE1.md docs/COMMITS_SUGERIDOS_2026-10-08.md requirements.txt
git commit -m "OE1-saneamiento fase 6: informe del saneamiento"
git tag oe1-saneado-v1
```

Después: `git push` y `git push origin oe1-saneado-v1`. Los informes LaTeX y sus zip están fuera del repo, en `Monografia\`.
