# Log de sesión — EDA, pipeline y OE1 (vae-latent-geometry-nids)

Material de referencia crudo para redactar la documentación final de la
monografía. No es narrativa pulida — es el registro de qué se hizo, con
qué parámetros, qué se decidió y por qué, y dónde está cada dato.

## 1. Alcance de esta sesión

Cubre, en orden:
1. Creación del notebook de EDA (`eda_cicids2017.ipynb`).
2. Sección 7 del EDA: dataset final limpio y anotado (duplicados exactos, near-constant, clusters candidatos).
3. Sincronización de `pipeline.py`/`config.py` con los 3 duplicados exactos (77→74 features) y corrida completa del pipeline.
4. Construcción de `training/`, `evaluation/`, `viz/` para OE1.
5. Revisión de los 13 clusters de features correlacionadas pendientes (umbral |r|≥0.95 por clique) → 74→51 features.
6. Cierre de OE1: entrenamiento, unidades activas/KL por dimensión, barrido de 5 semillas.

## 2. Archivos creados/modificados

### Notebooks (commiteados salvo lo indicado)
- `notebooks/00_raw_data_eda.ipynb` — ya existía; solo se re-ejecutó (outputs actualizados por el usuario), no se tocó su código.
- `notebooks/eda_cicids2017.ipynb` — creado esta sesión. 7 secciones: carga, perfilado básico (nulos/infinitos, distribución de clases, desbalance), fuga de datos (near-constant, IDs/puertos), distribuciones + correlación (4.1-4.3: histogramas, matriz de correlación, clustering de pares por union-find), preparación de partición (import de `split_benign`/`scale_no_leakage`, sin ejecutar), resumen, y **Sección 7 "Dataset final limpio y anotado"** (7.1 near-constant vs. familia, 7.2 bug de `sanitize` con tasas negativas, 7.3 duplicados exactos, 7.4 clusters candidatos, 7.5 `df_final`, 7.6 embudo, 7.7 export a `eda_exclusion_log.json`, 7.8 placeholder de dimensión).
- `notebooks/cluster_review_oe1.md` — **sin commitear**. Matrices de correlación par-a-par reales (clusters 0,1,2,3,5) + resolución por clique de los 13 clusters.
- `notebooks/oe1_report.md` — **sin commitear**. Reporte de cierre de OE1: comparación 74 vs. 51 features, barrido de 5 semillas, apéndice con datos sin redondear.

### Código de producción (`src/vae_nids/`)
- `config.py` — **modificado y commiteado** (commit `9a3a1dc`): se agregó `DUPLICATE_COLS` (3 columnas duplicadas exactas).
- `data/pipeline.py` — **modificado y commiteado**: `get_feature_columns()` ahora también excluye `cfg.DUPLICATE_COLS`.
- `models/vae_example.py` — **sin tocar en toda la sesión** (referencia educativa, explícitamente protegida).
- `models/vae.py` — **creado, sin commitear**. Réplica exacta de la arquitectura de `vae_example.py` (Encoder/Decoder Dense-32, reparametrización, ELBO heterocedástico), con `input_dim` derivado de `feature_columns.json` (`load_feature_columns()`) y `beta`/`latent_dim` como parámetros de `VAEConfig`, no hardcodeados.
- `training/train.py` — **creado, sin commitear**. Loop de entrenamiento con minibatching manual (sin `DataLoader`), early stopping sobre `val_total`, log CSV por época, checkpoint con `state_dict` + config + seed + `feature_columns`. CLI vía `argparse`.
- `evaluation/metrics_oe1.py` — **creado, sin commitear**. `active_units()` (criterio Burda et al. 2016) y `kl_per_dimension()`, evaluadas sobre `val_benign.parquet` con el checkpoint indicado.
- `viz/training_curves.py` — **creado, sin commitear**. Curvas de entrenamiento (3 paneles: total/recon/KL, train vs. val) y diagnóstico latente (KL por dimensión + unidades activas, barras). Paleta y specs de marca siguiendo la skill `dataviz` del proyecto (validada con `scripts/validate_palette.js`: slots 1-2 azul `#2a78d6`/naranja `#eb6834`, ΔE CVD 24.7 / normal-vision 33.6).
- `README.md` — **modificado y commiteado**: nota de duplicados exactos y "74 features" (antes decía 77). **Sigue desincronizado** en la sección de objetivos (documenta 5 OE, la monografía ya usa 4) y en el checklist "Estado actual" (checkboxes vacíos pese al trabajo ya hecho).

### Datos (`data/processed/`, gitignorado — no se commitea nada de esto)
- `feature_columns.json` — 74 → **51** features (ver Sección 5).
- `eda_exclusion_log.json` — trazabilidad completa: `dropped.near_constant_no_pattern`, `dropped.exact_duplicates`, `dropped.cluster_collapse` (23 entradas, cada una con `r_vs_kept` real y razón), `resolved_clusters` (13 clusters con miembros, representante, r reales), `invalidated_checkpoints`.
- `train_benign.parquet` (1,159,948 filas), `val_benign.parquet` (248,560), `test_benign.parquet` (248,561), `test_attacks.parquet` (442,952), `scaler.joblib`, `label_report.csv` — generados por `python -m vae_nids.data.pipeline` sobre el dataset completo. **`scaler.joblib` sigue fit sobre 74 columnas** (no se re-fit al bajar a 51 — no hacía falta, `MinMaxScaler` es por-columna, seleccionar un subconjunto de columnas ya escaladas es válido).

### Outputs de entrenamiento (`outputs/`, gitignorado en `checkpoints/`; `logs/`/`metrics/`/`figures/` no están en `.gitignore` pero tampoco se commitearon)
Ver Sección 6 para la lista completa de corridas y sus artefactos.

## 3. Parámetros y configuración usados

### Pipeline de datos (`config.py`)
- `RANDOM_SEED = 42`
- `TRAIN_FRAC=0.70`, `VAL_FRAC=0.15`, `TEST_FRAC=0.15`, estratificado por día
- `IDENTIFIER_COLS = [Flow ID, Src IP, Dst IP, Timestamp]`
- `PORT_COLS = [Src Port, Dst Port]`
- `DUPLICATE_COLS = [Bwd Segment Size Avg, Average Packet Size, Fwd Segment Size Avg]` (r=1.0 con Bwd Packet Length Mean, Packet Length Mean, Fwd Packet Length Mean respectivamente)

### Arquitectura del VAE
- `input_dim`: 74 (baseline inicial) → **51** (final, post-revisión de clusters)
- `hidden_dim = 32`, `latent_dim (k) = 8`, `beta (β) = 1.0`
- clamps de `logvar`: `[-10, 10]`
- ELBO = NLL Gaussiana heterocedástica (suma sobre features, media sobre batch) + β·KL cerrado (Kingma & Welling 2013, Apéndice B)

### Entrenamiento
- `batch_size = 1024`, optimizer `Adam(lr=1e-3)`
- `max_epochs`: 100 (1er intento, no convergió) → **250** (definitivo)
- `patience = 10` épocas
- `min_delta`: 0.0 (1er intento, nunca disparó) → **0.10** (derivado empíricamente del ruido real de `val_total`, ver Sección 4)
- Semilla única (`torch.manual_seed(seed)`) gobierna inicialización de pesos + todos los `torch.randperm()` de shuffling por época
- Semillas usadas: `42` (baseline OE1.1/OE1.2) + `43, 44, 45, 46` (barrido OE1.3, 5 semillas en total)

### Métrica de unidades activas (OE1.2)
- Criterio: Burda, Grosse & Salakhutdinov (2016), *Importance Weighted Autoencoders*, ICLR, Apéndice E — `A_u = Var_x[E_{q(z|x)}[z_i]] = Var_x[μ_i(x)]`, activa si `A_u > 0.01`.
- Umbral `0.01`: valor original del paper, no depende del tamaño del dataset (ver justificación completa en el turno donde se preguntó por defaults antes de escribir código).

### Revisión de clusters correlacionados (pre-OE1, features 74→51)
- Umbral: `|r| ≥ 0.95` (Pearson)
- Verificación **por clique** (todos los pares del subconjunto deben cruzar el umbral entre sí), explícitamente **no** single-linkage/encadenamiento — para no colapsar por transitividad asimetrías forward/backward que Engelen et al. (2021) señalan como relevantes para ciertas familias de ataque (ej. IAT en DoS Slowhttptest).
- Matrices de correlación calculadas sobre el dataset completo sanitizado (benigno+malicioso mezclado), mismo procedimiento que la Sección 4.2 del EDA.

## 4. Decisiones metodológicas clave (con justificación)

- **Dataset**: CICIDS2017 versión corregida (Engelen, Rimmer & Joosen, 2021), 5 CSV, 2,100,814 filas crudas, 84 columnas. Sanitización elimina 793 filas (0.0377%) por nulos/infinitos en `Flow Bytes/s`, `Flow Packets/s`, etc.
- **`min_delta=0.10` (no arbitrario)**: se analizaron los deltas de `val_total` época-a-época de la corrida de 100 épocas que no convergió. Ruido robusto (MAD) medido en ventanas "planas" ≈ 0.10-0.12 nats — no "sexta cifra decimal" como se sospechaba inicialmente. Se validó empíricamente contra el log real qué `min_delta` produce una racha de 10 épocas sin mejora: 0.10 es el techo defendible sin empezar a filtrar mejora genuina (ventana de aceleración real en épocas 82-100, coincidente con el colapso de una dimensión latente).
- **74→51 features**: 13 clusters de features con `|r|≥0.95` resueltos por clique real (no agregado min/max). Detalle completo, con cada `r` real y razón, en `cluster_review_oe1.md` y `eda_exclusion_log.json`. Hallazgo citable: en el cluster de conteos (ACK Flag Count, Header Length, Total Packets fwd/bwd) la simetría fwd/bwd es total (r>0.99, colapsan juntos); en tamaño de payload y temporización (IAT) la asimetría fwd/bwd SÍ se sostiene empíricamente y esas features no colapsan entre direcciones.
- **Checkpoints preservados, no sobrescritos**: al pasar de 74→51 features se renombraron los artefactos previos a `*_input74_*` en vez de sobrescribirlos, específicamente para poder documentar la comparación antes/después.
- **`val_total` no es comparable entre input_dim distintos**: la NLL de reconstrucción se suma sobre `input_dim` dimensiones, así que un `input_dim` menor arroja mecánicamente una pérdida total menor en magnitud — no implica "mejor reconstrucción".
- **La identidad de dimensión no es comparable entre semillas**: no hay alineación canónica del espacio latente entre entrenamientos independientes; solo el *conteo* de unidades activas es comparable entre semillas, no *cuál* índice colapsa.

## 5. Cronología resumida de esta sesión

1. Notebook EDA creado y ejecutado sobre el dataset completo por el usuario.
2. Sección 7 del EDA agregada (dataset final limpio y anotado) → `eda_exclusion_log.json` v1 (77→74 features, solo duplicados exactos).
3. `pipeline.py`/`config.py` sincronizados con esos 3 duplicados; pipeline completo corrido (74 features).
4. `training/`, `evaluation/`, `viz/` construidos para OE1; primera corrida (74 features, 100 épocas) no convergió por criterio (min_delta=0.0 demasiado laxo).
5. `min_delta=0.10` derivado del log real; segunda corrida (74 features, 250 épocas) convergió en época 137 (mejor: 127) → **7/8 unidades activas**.
6. Revisión de los 13 clusters de features correlacionadas (`|r|≥0.95`, por clique) → decisión final 74→**51** features, documentada en `cluster_review_oe1.md`.
7. `feature_columns.json` y `eda_exclusion_log.json` actualizados; checkpoint de 74 preservado como referencia.
8. Reentrenamiento con 51 features (misma seed/β/k/min_delta/patience): converge en época 62 (mejor: 52) → **6/8 unidades activas** (dimensión 2 colapsa en ambas configuraciones — robusto; dimensión 7 colapsa solo con 51 features — cambio real, documentado).
9. Barrido de 5 semillas (42,43,44,45,46) sobre la config final (51 features): **5-6/8 unidades activas en las 5** (media 5.4, std≈0.49) → OE1 cerrado.

## 6. Inventario de corridas de entrenamiento (todas en `outputs/`)

| run_name | seed | input_dim | época conv. (mejor) | val_total | activas |
|---|---|---|---|---|---|
| `vae_k8_beta1_input74` | 42 | 74 | 137 (127) | −277.2749 | 7/8 |
| `vae_k8_beta1_input51` | 42 | 51 | 62 (52) | −185.2791 | 6/8 |
| `vae_k8_beta1_input51_seed43` | 43 | 51 | 96 (86) | −190.0494 | 5/8 |
| `vae_k8_beta1_input51_seed44` | 44 | 51 | 90 (80) | −186.0789 | 6/8 |
| `vae_k8_beta1_input51_seed45` | 45 | 51 | 94 (84) | −187.8910 | 5/8 |
| `vae_k8_beta1_input51_seed46` | 46 | 51 | 92 (82) | −186.3978 | 5/8 |

Datos completos sin redondear (Var_x[μᵢ] y KL por dimensión, las 6 corridas): apéndice de `oe1_report.md`.

Cada corrida generó: `outputs/checkpoints/{run_name}_best.pt`, `outputs/logs/{run_name}_train_log.csv`, `outputs/metrics/{run_name}_oe1_metrics.json`; las corridas `input74` e `input51` (seed 42) además tienen `outputs/figures/{run_name}_training_curves.png` y `..._latent_diagnostics.png`.

## 7. Referencias bibliográficas citadas en el trabajo de esta sesión

- Engelen, G., Rimmer, V., & Joosen, W. (2021). *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study*. IEEE SPW. — dataset depurado; asimetría fwd/bwd relevante para ataques sigilosos.
- Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes*. arXiv:1312.6114. — formulación del VAE, KL cerrado (Apéndice B).
- An, J., & Cho, S. (2015). *Variational autoencoder based anomaly detection using reconstruction probability*. — NLL heterocedástica en vez de MSE.
- Burda, Y., Grosse, R., & Salakhutdinov, R. (2016). *Importance Weighted Autoencoders*. ICLR. — criterio de unidades activas (Apéndice E, umbral 0.01).
- Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018). *Kitsune*. NDSS. — contexto/motivación del proyecto (mencionado en README/monografía, no usado directamente en el código de esta sesión).

## 8. Pendientes / notas abiertas (flageadas durante la sesión, no resueltas)

- **README desincronizado**: documenta 5 objetivos específicos (OE1-OE5); la monografía ya consolidó a 4 OE. Checklist "Estado actual" no refleja el trabajo hecho.
- **`config.py`/`pipeline.py` no reflejan la revisión de clusters**: `get_feature_columns()` todavía solo excluye `DUPLICATE_COLS` (3 columnas) → si se re-corre `python -m vae_nids.data.pipeline` desde cero, regenera 74 features, no 51. Falta trasladar la decisión de `eda_exclusion_log.json` (`dropped.cluster_collapse`, 23 columnas) a una lista en `config.py` si se quiere que el pipeline de producción sea consistente con `feature_columns.json`.
- **`scaler.joblib` sigue fit sobre 74 columnas** — válido para seleccionar subconjuntos (MinMaxScaler es por-columna), pero no se regeneró un scaler "nativo" de 51 columnas.
- **`requirements.txt` no incluye `nbformat`/`nbclient`** — se usaron solo transitoriamente para validar notebooks ejecutándolos vía CLI; no son necesarios para producción pero sí para reproducir esa validación en otra máquina.
- **Nada del trabajo de esta sesión (Sección 4 en adelante) está commiteado** salvo `config.py`/`pipeline.py`/`README.md` (commit `9a3a1dc`, que también incluye `vae_example.py` y las dos primeras versiones del notebook EDA).
- **Clusters resueltos con criterio mecánico, no 100% validado semánticamente por un experto de dominio** más allá de la revisión que ya se hizo — en particular, cluster 1 (colapso de 6 métricas de conteo fwd+bwd) es el caso más agresivo y el que más vale la pena re-mirar si OE2 encuentra algo raro en las métricas de conteo por familia de ataque.

## 9. Mapa de archivos de referencia

| archivo | contenido |
|---|---|
| `notebooks/eda_cicids2017.ipynb` | EDA completo, Sección 7 = primera pasada de limpieza (77→74) |
| `notebooks/cluster_review_oe1.md` | Matrices de correlación reales + resolución por clique (74→51) |
| `notebooks/oe1_report.md` | Reporte de cierre de OE1 (74 vs 51, barrido de semillas, apéndice de datos) |
| `notebooks/session_log.md` | Este archivo |
| `data/processed/eda_exclusion_log.json` | Trazabilidad machine-readable de toda exclusión de features |
| `outputs/metrics/oe1_seed_sweep.json` | Resultados crudos del barrido de semillas 43-46 |
