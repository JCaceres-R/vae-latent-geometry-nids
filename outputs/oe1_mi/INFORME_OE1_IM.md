# OE1-IM — Validación de la reducción de features con entropía e información mutua

**Fecha:** 2026-09-25 · **Alcance:** solo análisis. No se reentrenó el VAE ni se tocó el pipeline, `feature_columns.json`, `config.py` ni ningún artefacto existente.
**Código:** `src/vae_nids/evaluation/oe1_mi/` (se sigue el patrón de `evaluation/oe2_robust/` en lugar de crear `analysis/`) · **Tests:** `tests/oe1_mi/` (29 tests, todos pasan) · **Reproducir:** `python -m vae_nids.evaluation.oe1_mi.run_all`

## Resumen

- **Rama recomendada: B.** La IM confirma la mayor parte de la reducción 74 → 51, pero encuentra dos cosas que Pearson no ve:
  - **22 pares redundantes entre las 51 features conservadas.** Son redundancias no lineales. Las detectan las 7 medidas IM a la vez y se repiten en `train_benign`. Pearson les da entre 0,04 y 0,948.
  - **1 fusión que la IM contradice con todas las medidas.** Es Total Length of Bwd Packet frente a los conteos del cluster 1.
- **Conjunto propuesto: 46 features** (sección 7).
- **La cifra de "cuántas features sobran" depende del criterio IM.** Según el criterio, quedan entre 21 y 54 features (sección 5). Por eso la recomendación usa solo los cambios que resisten todas las variantes.
- **La crítica del director se confirma con datos.**
  - Pares con \|r\| < 0,95 pero r_info ≥ 0,95: 89 con el histograma (B = 20) y 305 con KSG.
  - Casos extremos: Flow Packets/s ↔ Flow IAT Mean (\|r\| = 0,0404, r_info = 0,9947) y Flow Duration ↔ Flow Packets/s (\|r\| = 0,0720, r_info = 0,9850).

---

## 1. Reconocimiento (fase 0)

Verificación automática en `metrics/oe1_fase0_reconocimiento.json`: todas las comprobaciones dan `True`.

### 1.1 Datos sobre los que se calculó la matriz de Pearson del EDA

| Pregunta | Respuesta | Evidencia |
|---|---|---|
| ¿Qué split? | **Ninguno**: el dataset completo, benigno y malicioso **mezclados**. No es el split de entrenamiento benigno. | `notebooks/eda_cicids2017.ipynb`, celda 28: `corr = df_clean[feature_cols].corr()`. Coincide con `notebooks/cluster_review_oe1.md:5-9` ("dataset completo sanitizado, benigno+malicioso mezclado") y `notebooks/session_log.md:73`. |
| ¿Antes o después del escalado? | **Antes**, en crudo. Lectura por defecto (int64 / float64) de los 5 CSV. | Celda 4 (`load_dataset`). |
| ¿Con o sin inf/NaN? | **Sin**. Se aplica `pipeline.sanitize`, que pasa de 2.100.814 a **2.100.021 filas** (−793). | Celda 11; `src/vae_nids/data/pipeline.py:70`. |
| ¿Filtro de tasas negativas? | **No**. Ese filtro (−31 filas) se aplicó después, en la sección 7.2 (celda 45), y tampoco está en `pipeline.py`. | Celdas 45 y 53. |

**Conjunto usado aquí:** exactamente ese (`data.py`), con 2.100.021 filas × 74 features. El recuento de filas se verifica. El Pearson recalculado difiere del registrado en `eda_exclusion_log.json` como máximo en **5,4·10⁻⁵**, lo que corresponde al redondeo a 4 decimales del log.

**Análisis secundario (no pedido; añadido porque cambia la lectura):** `train_benign.parquet` (1.159.948 filas), que es lo que ve el VAE (secciones 5 y 8).

### 1.2 Cómo se formaron los clusters y las cliques

- **Clusters.** En la celda 38 se agrupan los pares con \|r\| ≥ 0,9 mediante **union-find** sobre componentes conexas. Es single-linkage y no usa `networkx`, que no es dependencia del proyecto. Salen 13 clusters.
- **Cliques.** **No hay código de clique en el repositorio.** La resolución a \|r\| ≥ 0,95 por clique se hizo **a mano** en `notebooks/cluster_review_oe1.md` (desde la línea 10). Las decisiones quedaron en `data/processed/eda_exclusion_log.json → resolved_clusters`.
- **Representantes.** Se eligieron a mano, desempatando por "métrica agregada sobre direccional" o por "orden de esquema CICFlowMeter" (`eda_exclusion_log.json → dropped.cluster_collapse[*].reason`).
- **Decisiones manuales que no son "clique pura":**
  - **Cluster 2.** Dos 5-cliques comparten un núcleo de 4. Se fusiona el núcleo y Bwd IAT Max e Idle Min quedan sueltos.
  - **Cluster 3.** Bwd IAT Min ↔ Fwd IAT Min = 0,9502 cruza el umbral pero **no se usó** "para preservar la asimetría fwd/bwd".
  - **Packet Length Variance.** Se descartó **a priori** por var = std², "antes de calcular las matrices". Su \|r\| con Packet Length Std es **0,9394**: **Pearson solo no la habría fusionado**.

Para comparar sin sesgo, esa lógica manual se formalizó como un algoritmo explícito (`cliques.py`, docstring del módulo): Bron–Kerbosch, cliques de tamaño máximo, empaquetamiento único y, si no es único, intersección y partes privadas. **El criterio P reproduce exactamente los 12 grupos, sus representantes y las 51 features** (`P_reproduction`: `groups_equal`, `reps_equal` y `final_equals_51` son `True`).

### 1.3 Listas de 74 / 51 features y los 13 clusters

- **74 features.** Coinciden exactamente con `feature_columns` del checkpoint `vae_k8_beta1_input74_best.pt` (lista en `config.py → FEATURES_74`).
- **51 features.** Coinciden con `data/processed/feature_columns.json`, con el checkpoint `input51` y con `eda_exclusion_log.json`.
- **23 descartadas.** Salen de `src/vae_nids/config.py:60-83` (`CORRELATED_CLUSTER_DROP_COLS`) y coinciden con el log.
- **Tabla del documento OB1 frente al código:** **coincide en los 13 clusters** (representantes, originales y descartadas). No hay discrepancias.

### 1.4 Hiperparámetros reales del modelo

| Hiperparámetro | Valor real (OE1) | Dónde se fija | Justificación escrita en el repo |
|---|---|---|---|
| J (latent_dim) | 8 | `src/vae_nids/models/vae.py:38` | Comentario: "k -- variable en OE4". No hay justificación del valor 8. |
| Ancho oculto | 32 | `vae.py:40` | "ancho de encoder/decoder (arquitectura fija)". No hay justificación del valor. |
| β | 1,0 | `vae.py:39` | "peso del término KL en el ELBO -- variable en OE4". |
| Clamps de log σ² | [−10, 10] | `vae.py:44-45`, aplicados en 61 y 78 | `vae.py:42-43`, literal: *"Clamps de estabilidad numérica: sin esto, log_sigma2 puede irse a valores extremos durante el entrenamiento (explota o colapsa a -inf)."* |
| Optimizador y lr | Adam, 1e-3 | `src/vae_nids/training/train.py:90, 115` | No encontrada. |
| Batch | 1024 | `train.py:89` | No encontrada. |
| Paciencia | 10 | `train.py:92` | No encontrada. |
| min_delta | **0,10** (por CLI; el valor por defecto del código es 0,0) | `train.py:93, 188`; valor usado en `notebooks/session_log.md:62` y `notebooks/oe1_report.md:118-120` | `session_log.md:78`, literal: *"se analizaron los deltas de `val_total` época-a-época de la corrida de 100 épocas que no convergió. Ruido robusto (MAD) medido en ventanas 'planas' ≈ 0.10-0.12 nats […] 0.10 es el techo defendible sin empezar a filtrar mejora genuina"*. También en `notebooks/explicacion_validacion_detalle.md:11`. |
| max_epochs | **250** (por CLI; el valor por defecto es 100) | `train.py:91, 186`; `session_log.md:60` | `session_log.md:60`: "100 (1er intento, no convergió) → 250 (definitivo)". |

**Coherencia con los artefactos.**
- Los logs tienen 137 épocas (input74) y 62 (input51). Ambas cifras son compatibles con max_epochs = 250 y no con 100.
- Los checkpoints guardan las épocas 127 y 52, pero el mínimo crudo de `val_total` está en las épocas 137 y 53. Es lo esperado con min_delta = 0,10: mejoras menores que 0,10 no guardan checkpoint.
- `config` de los dos checkpoints: `{latent_dim: 8, beta: 1.0, hidden_dim: 32, logvar_min: -10, logvar_max: 10}`.

**"Fundamentos Matemáticos del VAE".** **No existe en el repositorio** ni en `../Monografia/`, donde se revisaron los 33 archivos. Solo hay referencias indirectas a un "documento de fundamentos" (por ejemplo, "ecuación 5 del documento de fundamentos" en `notebooks/oe2_cierre_borrador.md:168`, `notebooks/oe2_contexto_completo.md:165` y `src/vae_nids/evaluation/detectability_oe2.py:7`). No se extrajo ni se inventó ninguna justificación.

**Discrepancia encontrada (fuera del alcance de OE1-IM, pero real):** `src/vae_nids/evaluation/oe2_robust/config.py:40` dice "idéntico a OE1/OE2 original", pero fija `MAX_EPOCHS = 100` (línea 46) y `MIN_DELTA = 0.0` (línea 48), distinto de 250 / 0,10 en OE1. El checkpoint de la semilla 0 de OE2-robusto se guardó en la época 99, lo que es compatible con haber tocado el tope de 100.

### 1.5 Figuras 1-4 del OB1

| Figura | Ruta |
|---|---|
| Curvas de entrenamiento, 74 features, semilla 42 | `outputs/figures/vae_k8_beta1_input74_training_curves.png` |
| Diagnóstico latente, 74 features, semilla 42 | `outputs/figures/vae_k8_beta1_input74_latent_diagnostics.png` |
| Curvas de entrenamiento, 51 features, semilla 42 | `outputs/figures/vae_k8_beta1_input51_training_curves.png` |
| Diagnóstico latente, 51 features, semilla 42 | `outputs/figures/vae_k8_beta1_input51_latent_diagnostics.png` |

---

## 2. Estimador y validación (fase 1)

### 2.1 Decisiones de diseño (y un cambio respecto del prompt)

- **Unidades.** Entropías e IM por histograma en bits. IM KSG en nats. r_info = √(1 − e^(−2·I_nats)).
- **Corrección de sesgo** (Sharmin 2019). Sesgo = (I_x − 1)(I_y − 1) / (2N ln 2), con los intervalos **efectivos**; si I_corr sale negativa se recorta a 0.
  - Con N = 2,1 M el sesgo es mínimo: máximo 1,24·10⁻⁴ bits y mediana 2,06·10⁻⁵ bits.
  - En 78 de los 2.701 pares I_corr quedó recortada a 0.
- **Discretización: "cuantiles con átomos aislados".** Es la variante equivalente que permite el prompt, y la introduje después de ver el siguiente artefacto del `qcut(duplicates='drop')` literal.
  - Fwd Bytes/Bulk Avg, Fwd Packet/Bulk Avg y Fwd Bulk Rate Avg tienen 95,86 % de ceros. Con qcut literal todos sus bordes se fusionan en **un solo intervalo**, de modo que **H = 0**, aunque el 4,1 % distinto de cero es información real.
  - En general, qcut mezcla el átomo (por ejemplo, el 0) con los valores vecinos del mismo intervalo.
  - **Regla:** todo valor con frecuencia ≥ N/B forma su propia categoría. El resto se reparte por igual frecuencia en max(1, round(B·n_resto/N)) intervalos.
  - Sin átomos, el resultado es **idéntico** a qcut (lo verifica un test).
  - Con átomos, las tres features Fwd Bulk pasan a H = 0,2488 bits.
  - qcut literal se reporta como variante de sensibilidad (`qcut20`).
- **Si la feature tiene ≤ B valores únicos**, se usan sus valores como categorías, como pide el prompt.

### 2.2 Resultados de los 4 tests obligatorios

Números con semilla 42 en `metrics/oe1_validacion_estimador.json`. Asserts en `tests/oe1_mi/test_estimators.py`.

**Test 1 — independencia** (X normal, Y exponencial, N = 10⁶, B = 20):
- I = 2,6276·10⁻⁴ bits; sesgo teórico = 2,6041·10⁻⁴; **I_corr = 2,35·10⁻⁶ bits**.
- La corrección elimina el 99,1 % del valor crudo.
- Tolerancia del test: I_corr < 5·10⁻⁴ bits con N = 200k.

**Test 2 — I(X;X) = H(X)**: diferencia **exactamente 0,0** en los tres casos.

| Caso | H (bits) | Intervalos efectivos |
|---|---|---|
| Lognormal continua | 4,3219 | 20 |
| 60 % de masa en 0 | 2,1710 | 9 |
| Categórica de 7 valores | 2,8074 | 7 |

**Test 3 — gaussiana bivariada** (N = 10⁶ para el histograma, 50k para KSG):

| ρ | Histograma B=10 | B=20 | B=50 | B=100 | B=200 | KSG (k=3) |
|---|---|---|---|---|---|---|
| 0,30 | 0,2873 | 0,2950 | 0,2981 | 0,2994 | 0,3005 | 0,3064 |
| 0,70 | 0,6748 | 0,6891 | 0,6961 | 0,6980 | 0,6987 | 0,7019 |
| 0,95 | 0,9286 | **0,9415** | 0,9472 | 0,9486 | 0,9487 | 0,9507 |

- Tolerancias de los tests: KSG ±0,03 (n = 20k); histograma B = 100 ±0,01; histograma B = 20 con sesgo hacia abajo ≤ 0,015.
- Errores observados: KSG como máximo 0,0064; B = 100 como máximo 0,0021.
- **Consecuencia directa para el criterio L.** Con B = 20, un par gaussiano con \|r\| = 0,95 exacto da r_info = 0,9415. **L con r_info ≥ 0,95 es más estricto que P incluso con datos perfectamente gaussianos.** Por eso se añade L_cal (sección 5).

**Test 4 — no lineal y no monótono** (Y = X² + 0,05·ε, X ~ U(−1, 1), N = 10⁶; figura `figures/demo_no_lineal.png`):
- **Pearson r = 0,0011.**
- **IM = 2,0002 bits** (nmi_sqrt 0,4628; r_info 0,9683).
- KSG = 1,5861 nats (r_info 0,9788).
- Es el argumento del director en miniatura.

### 2.3 Calibración de umbrales (gaussiana ρ = 0,95, mismo estimador, N = 2.100.021)

| B | r_info (umbral L_cal) | nmi_max (umbral S) | nmi_max analítico = I(0,95)/log₂B |
|---|---|---|---|
| 10 | 0,92861 | 0,43057 | 0,50550 |
| 20 | **0,94140** | **0,36278** | 0,38854 |
| 50 | 0,94720 | 0,29075 | 0,29753 |

- **Derivación del umbral de S** (lo que pide la tabla de criterios): es el nmi_max que obtiene, **con el mismo estimador**, un par gaussiano cuya \|r\| es exactamente 0,95. La alternativa analítica (IM de Linfoot en 0,95, que es 1,68 bits, dividida por log₂B) no descuenta la pérdida por discretizar; se reporta pero no se usa.
- **Techo estructural de Linfoot.** r_info ≥ 0,95 exige I ≥ 1,164 nats = 1,679 bits. Una feature con H < 1,679 bits **no puede pasar L ni siquiera consigo misma**; dos binarias idénticas y equiprobables tienen r_info = 0,866.
  - En el dataset afecta a URG, PSH, Bwd Bulk, Idle, Active y otras (sección 6).

### 2.4 Criterio adicional G: gaussiano-equivalente por par

Añadido por mí; no lo pedía el prompt. Resuelve a la vez los dos problemas de L: el sesgo por discretizar y el techo de entropía.

- **Definición.** Para cada par se toma una cópula gaussiana con ρ = 0,95 y se discretiza con **exactamente las mismas marginales** (probabilidades de intervalo) que las dos features, con el mismo N y la misma corrección de sesgo. Se define `gauss_ratio` = I_corr(X;Y) / I_ref. La arista existe si gauss_ratio ≥ 1, es decir, "al menos tan dependiente como un par gaussiano con \|r\| = 0,95 que tuviera estas marginales".
- **Tests.** Separa ρ = 0,97 de ρ = 0,93 y marca dos binarias idénticas que Linfoot no puede marcar.
- **Limitación.** Con distribuciones muy concentradas en átomos resulta **mucho más permisivo** que L (sección 5); la equivalencia con \|r\| solo es exacta para cópulas gaussianas.

---

## 3. Entropía por feature (fase 2)

Tabla completa: `metrics/oe1_entropia_features.csv`. Figura: `figures/oe1_entropia_features.png`.

- **Rango.** H va de 1,1·10⁻⁵ bits (Bwd URG Flags) a 4,3219 bits (Flow IAT Mean, cerca de log₂20 = 4,3219).
- **Las 6 features con máxima entropía** (≥ 4,3216 bits) son tasas y tiempos del flujo. Una de ellas, Fwd Packets/s, ya estaba descartada.
- **Features de baja información** (H normalizada < 0,1 o moda > 99 %):

| Feature | Estado | Únicos | Fracción de la moda | Intervalos efectivos | H (bits) | H normalizada |
|---|---|---|---|---|---|---|
| Bwd URG Flags | **conservada** | 2 | 0,9999995 (N−1 filas) | 2 | 0,000011 | 0,000011 |
| Fwd URG Flags | descartada | 5 | 0,999956 | 5 | 0,000772 | 0,000333 |
| URG Flag Count | **conservada** | 5 | 0,999955 | 5 | 0,000780 | 0,000336 |
| Subflow Bwd Packets | **conservada** | 2 | 0,999939 | 2 | 0,000941 | 0,000941 |
| ECE Flag Count | **conservada** | 5 | 0,999673 | 5 | 0,004518 | 0,001946 |
| CWR Flag Count | **conservada** | 9 | 0,999636 | 9 | 0,004954 | 0,001563 |

**5 de las 51 conservadas son de baja información.** No se eliminan, como pide el prompt, pero a efectos prácticos son casi constantes: Bwd URG Flags es distinta de 0 en **1** de 2,1 M filas, y URG Flag Count en 94. Si su presencia está justificada por su valor ante familias de ataque concretas, conviene revisarlo en OE2; aquí no se decide.

---

## 4. Matrices y sensibilidad a B (fase 3)

**Salida principal:** `metrics/oe1_pares_dependencia.csv`, 2.701 filas × 54 columnas. Incluye:
- \|r\|, \|ρ\| de Spearman;
- I, sesgo, I_corr, las 4 normalizaciones, r_info, G, p-valor;
- las mismas medidas con B = 10, B = 50 y qcut20;
- G (gauss_ratio), KSG e indicadores de cluster y de fusión.

El mismo cálculo sobre `train_benign` está en `metrics/oe1_pares_dependencia_train_benign.csv`.

**Figuras:**
- `figures/oe1_heatmap_pearson_vs_nmi.png`: \|r\| y nmi_sqrt con el mismo orden de features (bloques = 13 clusters).
- `figures/oe1_scatter_r_vs_rinfo.png`: los cuatro cuadrantes, con la línea de L_cal.

**Correlación de rangos de nmi_sqrt entre variantes** (`metrics/oe1_sensibilidad_B.json`):

| Variantes | Spearman | Kendall τ | Pares |
|---|---|---|---|
| B20 frente a B10 | 0,9948 | 0,9425 | 2.701 |
| B20 frente a B50 | 0,9959 | 0,9498 | 2.701 |
| B10 frente a B50 | 0,9860 | 0,9051 | 2.701 |
| B20 (átomos) frente a qcut20 | 0,9655 | 0,8690 | 2.485* |

\*Se excluyen los 216 pares que involucran las 3 features Fwd Bulk, cuya H vale 0 con qcut.

**Pares que cambian de lado de cada umbral respecto de B20** (`tables/oe1_sensibilidad_umbrales.csv`):

| Criterio | Aristas B20 | B10: entran / salen | B50: entran / salen | qcut20: entran / salen |
|---|---|---|---|---|
| L (r_info ≥ 0,95) | 113 | 0 / **47** | **60** / 0 | 0 / 23 |
| L_cal | 128 | 2 / 22 | 57 / 0 | 0 / 26 |
| S | 271 | 0 / 66 | 103 / 0 | 1 / 82 |
| G | 500 | 20 / 85 | 120 / 8 | 4 / 183 |
| A_1 | 396 | 39 / 33 | 9 / 26 | 25 / 80 |
| A_2 | 112 | 3 / 15 | 18 / 17 | 30 / 18 |

**Sin rodeos:**
- **El orden de los pares es estable frente a B** (Spearman ≥ 0,986).
- **Las decisiones por umbral fijo no lo son.** L pasa de 66 aristas (B = 10) a 173 (B = 50), porque r_info por histograma sube de forma sistemática con B (test 3).
- Cualquier conclusión que dependa de un solo B **no es firme**. Por eso la recomendación (sección 7) exige que se cumpla con B = 10, 20 y 50 a la vez.

**χ² (solo informativo).** El estadístico 2N·ln 2·I_corr es enorme (mediana 1,26·10⁶). **2.548 de 2.701 pares tienen p < 10⁻¹⁰** y 2.604 tienen p < 0,05. Con N = 2,1 M casi cualquier desviación de la independencia es "significativa". El test responde "¿hay alguna dependencia?", no "¿son redundantes?", así que **no se usó para decidir nada**.

---

## 5. Criterios de agrupación (fase 4)

Todos usan la misma resolución por clique. Detalle en `metrics/oe1_criterios.json`, `tables/oe1_criterios_resumen.csv` y `tables/oe1_grupos_por_criterio.csv`. Para A_k: μ = 0,2039 y σ = 0,1729, calculados sobre la nmi_sqrt de los 2.701 pares con B = 20.

| Criterio | Arista si... | Umbral | Aristas | Grupos | Descartadas | **Tamaño final** | Aristas compartidas con P |
|---|---|---|---|---|---|---|---|
| **P** (reproduce el original) | \|r\| ≥ 0,95, más var = std² a priori | 0,95 | 51 | 12 | 23 | **51** ✔ | — |
| P sin la regla a priori | \|r\| ≥ 0,95 | 0,95 | 52 | 12 | 22 | 52 | 51 |
| **L** | r_info (B = 20) ≥ 0,95 | 0,95 | 113 | 5 | 26 | **48** | 24 |
| L_cal (extra) | r_info (B = 20) ≥ r_info gaussiano con ρ = 0,95 | 0,9414 | 128 | 5 | 27 | 47 | 28 |
| **A_0,5** | nmi_sqrt ≥ μ + 0,5σ | 0,2904 | 746 | 10 | 53 | **21** | 50 |
| **A_1** | nmi_sqrt ≥ μ + σ | 0,3768 | 396 | 17 | 45 | **29** | 48 |
| **A_1,5** | nmi_sqrt ≥ μ + 1,5σ | 0,4633 | 206 | 13 | 43 | **31** | 37 |
| **A_2** | nmi_sqrt ≥ μ + 2σ | 0,5498 | 112 | 17 | 35 | **39** | 28 |
| **S** | nmi_max ≥ nmi_max gaussiano con ρ = 0,95 | 0,3628 | 271 | 15 | 51 | **23** | 37 |
| G (extra) | gauss_ratio (B = 20) ≥ 1 | 1 | 500 | 12 | 50 | 24 | 48 |
| G_consenso (extra) | G con B = 10, 20 y 50 | 1 | 408 | 14 | 41 | 33 | 45 |
| R (extra) | \|ρ\| de Spearman ≥ 0,95 | 0,95 | 58 | 13 | 26 | 48 | 15 |
| L_ksg (extra) | r_info KSG ≥ 0,95 | 0,95 | 351 | 7 | 32 | 42 | 45 |
| **Robusto** (extra) | las 7 medidas IM a la vez* | — | 66 | 9 | 20 | 54 | 15 |
| P sobre train_benign | \|r\| ≥ 0,95 | 0,95 | 47 | 11 | 20 | **54** | 44 |
| L sobre train_benign | r_info ≥ 0,95 | 0,95 | 109 | 6 | 24 | 50 | 23 |

\*Las 7 medidas: L con B = 10, 20 y 50; L_ksg; G con B = 10, 20 y 50.

Las variantes con B = 10, 50 y qcut20 de L, L_cal, S, G y A_k están en `tables/oe1_criterios_resumen.csv`.

**Lectura:**

1. **P reproduce exactamente el 74 → 51**, así que la comparación es válida.
2. **A_k fusiona de más con cualquier k.** Con k = 0,5 quedan 21 features y con k = 2 quedan 39. Como advertía el prompt, con 2.701 pares mayormente débiles, μ + kσ cae en nmi_sqrt entre 0,29 y 0,55, es decir, "comparten entre el 29 % y el 55 % de la entropía". Eso no es redundancia en el sentido del umbral original. **No se elige k**; ninguno es comparable con P.
3. **S no es "estricto" en la práctica.** Su umbral calibrado (0,3628) está pensado para features de ~20 intervalos. Con features de 2 a 5 intervalos, compartir el 36 % de la entropía es fácil, y S colapsa grupos heterogéneos (Protocol, FWD Init Win Bytes, Fwd Seg Size Min, Packet Length Min…). Queda con **23 features**.
4. **L y P coinciden en solo 24 de sus aristas** (P tiene 51 y L 113). L no es una versión más estricta de P: **mide otra cosa**. Deja fuera fusiones de P entre features de baja entropía (techo de Linfoot) e incluye 89 pares no lineales.
5. **El propio P depende del conjunto de datos.** Calculado solo sobre `train_benign`, Pearson da **54 features**, no 51:
   - la clique A del cluster 0 se reduce a {Packet Length Max, Packet Length Std}, así que Bwd Packet Length Max y Std no se fusionan;
   - el cluster 3 queda ambiguo, y Bwd IAT Min y Fwd IAT Min no se fusionan;
   - fusiona Bwd IAT Std ↔ Fwd IAT Std (cluster 11), que con el dataset mezclado no pasaba.
   
   La reducción original se decidió sobre benigno y malicioso juntos, mientras que el VAE solo ve benigno.

---

## 6. Contraste Pearson frente a IM (fase 5)

### Cuadrantes (umbral 0,95 en \|r\| y en r_info con B = 20)

`tables/oe1_cuadrantes.csv` lista los pares fuera de la diagonal; `tables/oe1_cuadrantes_conteo.csv` tiene los conteos.

| | r_info ≥ 0,95 | r_info < 0,95 |
|---|---|---|
| **\|r\| ≥ 0,95** | **24** redundancia confirmada | **28** Pearson sobreestimó |
| **\|r\| < 0,95** | **89** redundancia no lineal no detectada | **2.560** independientes o débiles |

Con otras medidas en el eje y:

| Medida | Confirmada | Pearson sobreestimó | No lineal no detectada | Débiles |
|---|---|---|---|---|
| L_cal (0,9414) | 28 | 24 | 100 | 2.549 |
| KSG | 46 | 6 | 305 | 2.344 |
| G (B = 20) | 49 | 3 | 451 | 2.198 |
| G_consenso | 46 | 6 | 362 | 2.287 |

**Todas las medidas IM coinciden** en que el cuadrante "redundancia no lineal no detectada" **no está vacío**. Lo que varía es su tamaño.

### Pregunta 1 — Los 13 clusters bajo L (y bajo G_consenso)

Tabla completa: `tables/oe1_q1_clusters.csv`.

| Cluster | Grupos de P | ¿Sigue siendo clique bajo L? (mínimo r_info) | Qué hace L | Bajo G_consenso |
|---|---|---|---|---|
| 0 | A = {Bwd Max, Bwd Std, Pkt Max, Pkt Std}; B = {Bwd Mean, Pkt Mean, Subflow Bwd Bytes}; Variance a priori | A **se parte** (3/6 pares; mínimo 0,9296: los pares de Bwd Packet Length Std); B se mantiene (0,9851) | Une A sin Bwd Std, B, **Variance** (r_info con Std = 0,9980) y **Total Length of Bwd Packet** (del cluster 1) en un grupo de 8 | A y B se mantienen (mínimo gauss_ratio 1,317 y 1,510) |
| 1 | Clique de 6 | **Se parte** (8/15; mínimo 0,8867) | {Total Fwd, Total Bwd, Fwd Header, Bwd Header}. ACK Flag Count queda fuera; Total Length of Bwd Packet pasa al cluster 0 | Se parte (12/15; mínimo 0,766) |
| 2 | Núcleo {Flow IAT Max, Fwd IAT Max, Idle Max, Idle Mean} | **Se parte** (1/6; mínimo 0,8631). *Techo de Linfoot: H(Idle) = 1,21 bits < 1,68* | Flow IAT Max se une al bloque de tasas y tiempos; Idle queda sin fusionar | Se mantiene (1,101) |
| 3 | {Bwd IAT Mean, Bwd IAT Min}; {Fwd IAT Mean, Fwd IAT Min} | Bwd se mantiene (0,9546); Fwd **se parte** (0,9462 < 0,95, aunque ≥ L_cal) | Fwd IAT Mean, Bwd IAT Mean, Flow IAT Mean y Flow IAT Std pasan a bloques de tiempo más grandes | Ambos se mantienen |
| 4 | {Fwd Pkt Len Mean, Subflow Fwd Bytes} | Se mantiene (0,9885) | La resolución de L mete Fwd Pkt Len Mean en una 5-clique más grande (con Fwd Pkt Len Max y Min, Pkt Len Min y Total Length of Fwd Packet) y deja Subflow Fwd Bytes suelto | Se mantiene |
| 5 | {Flow Duration, Fwd IAT Total, Bwd IAT Total} | Se mantiene (0,9574) | Se reparte entre dos bloques más grandes: Flow Duration va al de tasas y Fwd/Bwd IAT Total al de IAT direccionales | Se parte (2/3: Flow Duration ↔ Bwd IAT Total da gauss_ratio 0,981 con B = 10) |
| 6 | {Fwd URG Flags, URG Flag Count} | **Se parte** (0,0325). *Artefacto: H = 0,0008 bits* | — | Se mantiene (2,778; nmi_max 0,979) |
| 7 | {Bwd Bytes/Bulk Avg, Bwd Packet/Bulk Avg} | **Se parte** (0,7634). *Techo: H ≈ 0,72–0,80 bits* | — | Se mantiene (1,720) |
| 8 | {Flow Packets/s, Fwd Packets/s} | Se mantiene (0,9968) | **Se amplía** con Bwd Packets/s, Flow Duration, Flow IAT Max/Mean y Flow Bytes/s | Se mantiene |
| 9 | {Bwd PSH Flags, PSH Flag Count} | **Se parte** (0,9295) | — | Se mantiene (1,699) |
| 10 | P no fusiona (\|r\| = 0,948) | — | Fwd Packet Length Max entra en el grupo de longitudes fwd | G fusiona Fwd Pkt Len Max con Mean |
| 11 | P no fusiona (0,934) | — | No fusiona | G fusiona Bwd IAT Std en el grupo de Bwd IAT |
| 12 | P no fusiona (0,907) | — | No fusiona | G fusiona Active Min → Active Mean |

**Coincidencia con una decisión manual del EDA:** Bwd IAT Min ↔ Fwd IAT Min (\|r\| = 0,9502), que se dejó sin fusionar a mano, tiene r_info 0,8231, KSG 0,8789 y gauss_ratio 0,6175. **La IM respalda no fusionarlos.**

### Pregunta 2 — Cluster 1 (la fusión más agresiva)

Tabla: `tables/oe1_q2_cluster1.csv`. Los 15 pares tienen \|r\| ≥ 0,9939, pero:

- **La IM no confirma que los 6 sean mutuamente redundantes.** Solo 8 de 15 pares pasan L; 12 de 15 pasan G_consenso; 14 de 15 pasan KSG.
- **El problema es Total Length of Bwd Packet** (bytes, no conteo).
  - Con Total Fwd Packet: r_info 0,8867, KSG 0,9458 y gauss_ratio 0,766 (mínimo sobre las tres B). **Es el único par de toda la reducción P que no pasa ninguna de las 7 medidas IM.**
  - Direcciones: u(TLBwd \| Total Fwd) = **0,2870** y u(Total Fwd \| TLBwd) = **0,3931**. **Ninguna de las dos determina a la otra.**
  - Su mejor explicadora entre las 51 es Packet Length Std (u = 0,5716), no un conteo.
- **ACK Flag Count** queda en el borde: r_info 0,9437–0,9614 frente a los conteos. Pasa L_cal y G_consenso (≥ 1,33) y KSG (≥ 0,9685).
- **El argumento de la simetría de los conteos sí se confirma:**
  - Total Fwd Packet ↔ Total Bwd packets: r_info 0,9611; u 0,6556 / 0,6666.
  - Fwd Header Length ↔ Bwd Header Length: r_info 0,9790; u 0,6963 / 0,7238.
  - En los dos pares la redundancia es simétrica.
- **Conclusión:** el núcleo de conteos y headers de 5 features está respaldado por la IM. **La inclusión de Total Length of Bwd Packet en la clique es una contradicción**: Pearson la fusiona por la alta correlación en la cola (flujos grandes), pero en la mayoría de los flujos el volumen de bytes bwd aporta información que no está en los conteos.

### Pregunta 3 — ¿Pares entre las 51 con r_info ≥ 0,95?

**Sí. Son 39 pares** con r_info ≥ 0,95 (B = 20). Todos con \|r\| < 0,95.
- **22 de ellos pasan las 7 medidas IM** (L con B = 10, 20 y 50; KSG; G con B = 10, 20 y 50) **y además L y G en `train_benign`** (22 de 22).
- Lista completa con todas las medidas: `tables/oe1_q3_pares_51.csv` (ordenada por número de medidas).
- Los 22 robustos forman cuatro bloques:

| Bloque | Pares (\|r\| → r_info con B = 20 / KSG) |
|---|---|
| **Tasas y tiempos del flujo** (15 pares entre Flow Duration, Flow Packets/s, Flow IAT Mean, Std y Max, Bwd Packets/s) | Flow Packets/s ↔ Flow IAT Mean: **0,0404 → 0,9947 / 0,9999**; Flow Packets/s ↔ Bwd Packets/s: 0,4815 → 0,9939 / 1,0000; Flow IAT Mean ↔ Bwd Packets/s: 0,0438 → 0,9918; Flow Duration ↔ Flow IAT Max: 0,6604 → 0,9913; Flow Duration ↔ Flow Packets/s: 0,0720 → 0,9850 / 0,9999; … (Flow IAT Std ↔ Bwd Packets/s: 0,0549 → 0,9745) |
| IAT fwd/bwd | Bwd IAT Mean ↔ Bwd IAT Max: 0,7841 → 0,9905; Fwd IAT Mean ↔ Bwd IAT Mean: 0,9483 → 0,9667; Flow IAT Std ↔ Fwd IAT Mean: 0,9222 → 0,9685 |
| Longitudes fwd | Total Length of Fwd Packet ↔ Fwd Packet Length Max: 0,2142 → 0,9879; Fwd Packet Length Max ↔ Mean: 0,7919 → 0,9842 |
| Mínimos y momentos | **Fwd Packet Length Min ↔ Packet Length Min: 0,8394 → 0,9855** (nmi_sqrt 0,9930; u = 0,992 y 0,994: casi idénticas en información); Packet Length Mean ↔ Std: 0,9351 → 0,9738 |

- **Explicación física del bloque de tasas y tiempos:** Flow Packets/s = (paquetes fwd + bwd) / Flow Duration. La relación es hiperbólica, por eso Pearson ≈ 0; es monótona (Spearman 0,9586 para Flow Duration ↔ Flow Packets/s y 0,9959 para Flow Packets/s ↔ Flow IAT Mean) y casi determinista a conteo fijo. **Es exactamente el caso de redundancia no lineal que señala el director.**

### Pregunta 4 — ¿Fusiones de Pearson con r_info bastante menor que \|r\|?

Tabla: `tables/oe1_q4_fusiones_P.csv`. Son 40 pares dentro de grupos de P más el par Variance–Std.

- **19 de 41 tienen r_info < 0,95.** En **7** de ellos Linfoot no es aplicable porque la entropía mínima del par es < 1,68 bits:

| Par | \|r\| | r_info | Diferencia | Mínima H (bits) | Lectura |
|---|---|---|---|---|---|
| Fwd URG Flags ↔ URG Flag Count | 0,9996 | 0,0325 | 0,967 | 0,0008 | Artefacto de Linfoot; nmi_max 0,979 y u 0,989: **redundancia confirmada** |
| Bwd Bytes/Bulk ↔ Bwd Packet/Bulk | 0,9790 | 0,7634 | 0,216 | 0,72 | Techo de Linfoot (≤ 0,795 con esa H); nmi_max 0,791 y G 1,72: se sostiene |
| Idle Max / Idle Mean ↔ Flow / Fwd IAT Max (4 pares) | 0,984–0,998 | 0,863–0,876 | 0,12–0,13 | 1,21 | Techo (≤ 0,902). **Asimétrico:** u(Idle Max \| Flow IAT Max) = 0,868, pero u(Flow IAT Max \| Idle Max) = 0,243. Descartar Idle conservando Flow IAT Max, que es lo que hizo P, deja explicado el 83–87 % de la entropía de Idle |
| Idle Mean ↔ Idle Max | 0,9871 | 0,8816 | 0,106 | 1,21 | Techo; nmi_max 0,895 |

- **Los 12 pares con r_info < 0,95 donde Linfoot sí es aplicable:**
  - Los 5 pares de Total Length of Bwd Packet con el cluster 1 (0,8867–0,9350).
  - Los 3 pares de Bwd Packet Length Std en la clique A (0,9296–0,9353).
  - ACK Flag Count con Total Fwd y Total Bwd (0,9437 y 0,9445).
  - Fwd IAT Mean ↔ Fwd IAT Min (0,9462).
  - PSH (0,9295; H mínima 1,71, justo por encima del techo).
- **Bajo G_consenso** solo fallan 4 de 41: tres de Total Length of Bwd Packet y Flow Duration ↔ Bwd IAT Total (0,981, al límite).
- **Contradicciones firmes** (ninguna de las 7 medidas IM llega al umbral): **solo Total Fwd Packet ↔ Total Length of Bwd Packet.**
- **Contradicciones que dependen de la medida:** las demás. Todas las rechaza el histograma con Linfoot y las acepta al menos otra medida, sea G o KSG.

### Pregunta 5 — ¿Coinciden KSG y el histograma?

`metrics/oe1_contraste.json`.

- **Globalmente el orden coincide:** Spearman entre r_info por histograma y r_info KSG = 0,9725.
- **Pero KSG da valores sistemáticamente más altos.** Nunca ocurre que el histograma diga "sí" y KSG diga "no".

| Conjunto | Ambos sí | Ambos no | Solo histograma | Solo KSG |
|---|---|---|---|---|
| Pares de la pregunta 3 (272, marcados por alguna medida) | 39 | 162 | 0 | 71 |
| Fusiones de P (pregunta 4, 41) | 22 | 4 | 0 | 15 |

- **Pregunta 3:** los 39 pares con r_info ≥ 0,95 **también los marca KSG** (39 de 39). La redundancia no lineal entre las 51 es **firme** frente al estimador.
- **Pregunta 4:** en 15 de los 19 "Pearson sobreestimó", KSG dice lo contrario que el histograma (por ejemplo, Flow IAT Max ↔ Idle Max: histograma 0,8756 frente a KSG 0,9954). **Esas contradicciones no son firmes.**
  - Las 4 en que ambos dicen "no" son Total Fwd Packet ↔ Total Length of Bwd Packet (KSG 0,9458), Bwd Bulk, URG y PSH.
  - Las tres últimas son casos de techo de entropía, que afecta a los dos estimadores por igual (G las acepta).
  - **Queda una sola contradicción firme:** Total Length of Bwd Packet.
- **Advertencia sobre KSG con empates:** sklearn solo añade ruido de ~10⁻¹⁰ para romper empates, y muchas features tienen >50 % de la masa en un átomo. Un KSG más alto en esos pares puede ser, en parte, un artefacto (sección 8).

---

## 7. Recomendación

### **Rama B.** Hay redundancias adicionales robustas y una fusión que no se sostiene.

**Cómo se construyó el conjunto propuesto** (criterio `Recomendado` en `metrics/oe1_criterios.json`):

1. Se parte de P: las 23 descartadas siguen fuera, **excepto** las fusiones que **ninguna** de las 7 medidas IM sostiene. Eso restaura 1 feature.
2. Sobre las features resultantes se aplica la misma resolución por clique con las **aristas robustas**, es decir, las que pasan las 7 medidas. Las ambigüedades se dejan sin fusionar (postura conservadora).
3. El representante de cada grupo nuevo es el miembro que más información explica de los demás (máxima media de u(otro \| rep)).

**Conjunto de entrada propuesto: 46 features.**

> Protocol, Total Fwd Packet, Total Length of Fwd Packet, **Total Length of Bwd Packet**, Fwd Packet Length Max, Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std, Bwd Packet Length Min, Flow Bytes/s, Flow Packets/s, Flow IAT Min, Fwd IAT Mean, Fwd IAT Std, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Fwd PSH Flags, Bwd URG Flags, Packet Length Mean, Packet Length Std, FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, URG Flag Count, CWR Flag Count, ECE Flag Count, Down/Up Ratio, Fwd Bytes/Bulk Avg, Fwd Packet/Bulk Avg, Fwd Bulk Rate Avg, Bwd Bytes/Bulk Avg, Bwd Bulk Rate Avg, Subflow Fwd Packets, Subflow Bwd Packets, FWD Init Win Bytes, Bwd Init Win Bytes, Fwd Act Data Pkts, Fwd Seg Size Min, Active Mean, Active Std, Active Max, Active Min, Idle Std, Idle Min

**Cambios frente a las 51, con evidencia** (`tables/oe1_recomendado_cambios.csv`):

| Cambio | Con | \|r\| | r_info B=20 | KSG | gauss_ratio mínimo en B | u(f \| con) | train_benign r_info / G |
|---|---|---|---|---|---|---|---|
| **+ Total Length of Bwd Packet** (restaurar) | Total Fwd Packet | 0,9970 | 0,8867 | 0,9458 | 0,766 | 0,287 | 0,8573 / 0,753 |
| − Flow IAT Mean | Flow Packets/s | 0,0404 | 0,9947 | 0,9999 | 1,851 | 0,760 | 0,9950 / 2,135 |
| − Bwd Packets/s | Flow Packets/s | 0,4815 | 0,9939 | 1,0000 | 1,742 | 0,737 | 0,9941 / 2,059 |
| − Flow Duration | Flow Packets/s | 0,0720 | 0,9850 | 0,9999 | 1,436 | 0,587 | 0,9849 / 1,628 |
| − Flow IAT Max | Flow Packets/s | 0,0612 | 0,9844 | 0,9995 | 1,424 | 0,580 | 0,9829 / 1,568 |
| − Flow IAT Std | Flow Packets/s | 0,0507 | 0,9792 | 0,9967 | 1,409 | 0,609 | 0,9809 / 1,612 |
| − Packet Length Min | Fwd Packet Length Min | 0,8394 | 0,9855 | 0,9962 | 2,183 | 0,994 | 0,9932 / 2,596 |

**Comprobación de transitividad** (`tables/oe1_recomendado_transitividad.csv`). Flow Duration y Flow IAT Max eran representantes de P, así que se revisó qué queda explicando a sus descartadas:

| Descartada | Mejor conservada | u |
|---|---|---|
| Fwd IAT Total | Fwd IAT Mean | 0,779 |
| Bwd IAT Total | Bwd IAT Max | 0,851 |
| Fwd IAT Max | Fwd IAT Mean | 0,791 |
| Idle Max | Idle Min | 0,772 |
| Idle Mean | Idle Min | 0,818 |

El mínimo de u sobre las 28 features que quedan fuera es **0,5385** (Fwd IAT Min → Fwd IAT Mean), que **ya estaba en la reducción P**. Ninguna de las fusiones nuevas deja una feature peor explicada que las que P ya aceptó (el rango de u de P es 0,54–0,99).

**Qué tan robusta es la recomendación:**

- **Robusta frente a B, al estimador y al conjunto de datos.** Cada cambio pasa, o falla en el caso de la restauración, con histograma B = 10, 20 y 50, con KSG y con G, y se repite en `train_benign`.
- **No depende de k:** A_k no interviene.
- **No es robusto el número total.** Con cualquier criterio IM único el conjunto final va de 21 a 54 features (sección 5). El 46 es el resultado de aceptar solo lo que todas las medidas comparten; **no es "el" tamaño correcto**.
- **Tres ambigüedades quedaron sin fusionar** (cadenas de pares robustos sin clique):
  - {Bwd IAT Max, Bwd IAT Mean, Fwd IAT Mean};
  - {Fwd Packet Length Max, Fwd Packet Length Mean, Total Length of Fwd Packet};
  - {Packet Length Mean, Packet Length Std, Total Length of Bwd Packet}.
  
  Colapsarlas requiere una decisión de dominio, como la que tomó el EDA en el cluster 3.
- **Elección de representante.** Para {Fwd Packet Length Min, Packet Length Min} el algoritmo eligió la dirigida (u 0,9939 frente a 0,9921; diferencia 0,0018). Por la convención del EDA ("agregada sobre direccional") el autor puede preferir Packet Length Min: en información son equivalentes.
- **Coherencia con el argumento fwd/bwd original.** El bloque que se colapsa es de métricas del **flujo agregado** más Bwd Packets/s. Las IAT direccionales (Fwd IAT Mean y Std; Bwd IAT Mean, Std y Max) se conservan, así que el argumento del EDA sobre la asimetría fwd/bwd en temporización sigue en pie.
- **Alternativa mínima (Rama B conservadora, 52 features):** restaurar solo Total Length of Bwd Packet y no fusionar nada nuevo. Es el único cambio que corrige un error; los demás reducen redundancia. Si el autor prefiere no tocar las features temporales por razones de dominio (por ejemplo, DoS lentos), esta es la opción defendible. Aun así, conviene documentar en la monografía los 22 pares robustos como redundancia conocida que se conserva a propósito.

**No se reentrenó nada.** Adoptar cualquiera de las dos variantes implica rehacer OE1.1–OE1.3 y afecta a OE2. Esa decisión es del autor.

---

## 8. Limitaciones

1. **"Redundante" no significa "determinista".** Incluso las fusiones que Pearson aceptó con \|r\| ≥ 0,99 tienen u entre 0,5 y 0,7 con B = 20: la IM ve estructura fina que ninguna feature explica de la otra. Todo el análisis compara contra el **equivalente** de \|r\| = 0,95 (que deja ~10 % de varianza sin explicar), no contra la determinación exacta.
2. **Traducir 0,95 a la escala de la IM no es único.**
   - L (Linfoot) tiene sesgo hacia abajo por discretizar y un techo que la hace inaplicable con H < 1,68 bits: 25 de las 74 features tienen H < 1,68.
   - G no tiene techo, pero con marginales dominadas por átomos es muy permisivo (500 aristas con B = 20).
   - S y A_k carecen de un anclaje comparable.
   - La recomendación evita elegir exigiendo todas las medidas a la vez, pero ese consenso también es una decisión metodológica.
3. **La discretización con átomos es una decisión mía.** Es una variante "equivalente" del qcut que pide el prompt. Con qcut literal cambian las decisiones: por ejemplo, G pierde 183 aristas y la correlación de rangos con B20 baja a 0,9655.
4. **KSG en datos con empates masivos.** El estimador de Kraskov supone densidad continua. sklearn solo añade ruido de ~10⁻¹⁰, y muchas features tienen >50 % de la masa en un valor. KSG da valores sistemáticamente más altos que el histograma (71 pares "solo KSG" en la pregunta 3 y 15 en la pregunta 4) y parte de eso puede ser sesgo. Además se calculó sobre 50.000 filas, no sobre 2,1 M.
5. **Solo dependencia por pares.** La IM por pares no ve redundancia multivariada. Por ejemplo, Flow Duration es casi función de (conteo total, Flow Packets/s), que es una relación de tres variables. Un análisis con IM condicional o redundancia de grupo podría justificar reducciones distintas.
6. **La resolución por clique es mi formalización de reglas manuales.** Reproduce P exactamente, pero en grafos más densos (L, G, S) las reglas c y d deciden casos que el EDA nunca enfrentó.
7. **El conjunto de datos importa**, y no es el que ve el VAE. La reducción original y este análisis usan benigno y malicioso juntos, como pide el prompt. Sobre `train_benign`, el propio Pearson da 54 features en lugar de 51. Los 22 pares robustos y la restauración sí se repiten en benigno (secciones 6 y 7).
8. **Monte Carlo sin cuantificar.** La referencia gaussiana de G y la calibración de L_cal y S se simularon con una semilla (42) y N = 2,1 M. No se estimó el error Monte Carlo. Es pequeño salvo en marginales extremas (URG), donde el umbral es ruidoso.
9. **No se evaluó el efecto en detección.** Que una feature sea redundante en información no garantiza que su eliminación no afecte al AUC por familia de ataque. Verificarlo exige reentrenar, lo cual está fuera del alcance (punto de parada).
10. **χ² no sirve para decidir con este N**: 2.548 de 2.701 pares tienen p < 10⁻¹⁰.

---

## 9. Archivos generados

**Código** (todo nuevo; no se modificó ningún archivo existente)

| Archivo | Contenido |
|---|---|
| `src/vae_nids/evaluation/oe1_mi/__init__.py` | Paquete |
| `src/vae_nids/evaluation/oe1_mi/config.py` | Rutas, constantes, 74 features, 13 clusters, grupos documentados |
| `src/vae_nids/evaluation/oe1_mi/data.py` | Reconstrucción de `df_clean` del EDA (con caché) y carga de `train_benign` |
| `src/vae_nids/evaluation/oe1_mi/estimators.py` | Discretización (átomos / qcut), H, IM con corrección de sesgo, normalizaciones, Linfoot, G, KSG |
| `src/vae_nids/evaluation/oe1_mi/cliques.py` | Bron–Kerbosch, resolución por clique, elección de representante |
| `src/vae_nids/evaluation/oe1_mi/plotting.py` | Estilo de figuras |
| `src/vae_nids/evaluation/oe1_mi/phase0_recon.py` … `phase5_contrast.py`, `phase3_ksg.py`, `run_all.py` | Una fase por script |
| `tests/oe1_mi/__init__.py`, `tests/oe1_mi/test_estimators.py` | 29 tests (4 obligatorios más discretización, cliques y G) |

**Salidas** (`outputs/oe1_mi/`)

| Archivo | Contenido |
|---|---|
| `INFORME_OE1_IM.md` | Este informe |
| `metrics/oe1_fase0_reconocimiento.json` | Verificaciones de la fase 0 |
| `metrics/oe1_validacion_estimador.json` | Tests con N grande y calibración de umbrales |
| `metrics/oe1_entropia_features.csv` | Entropía por feature (átomos y qcut) |
| `metrics/oe1_ksg_pares.csv` | IM KSG de los 2.701 pares (submuestra de 50k) |
| `metrics/oe1_pares_dependencia.csv` | Todas las medidas por par (dataset EDA) |
| `metrics/oe1_pares_dependencia_train_benign.csv` | Pearson, Spearman, IM B20 y G sobre `train_benign` |
| `metrics/oe1_sensibilidad_B.json` | Correlación de rangos entre B y esquemas |
| `metrics/oe1_criterios.json` | Grupos, descartes y conjunto final de cada criterio y variante |
| `metrics/oe1_contraste.json` | Cuadrantes, resúmenes de las preguntas 2–5 y conjunto recomendado |
| `tables/oe1_criterios_resumen.csv` | Tabla comparativa de criterios |
| `tables/oe1_grupos_por_criterio.csv` | Grupos por criterio |
| `tables/oe1_sensibilidad_umbrales.csv` | Pares que cambian de lado por B |
| `tables/oe1_cuadrantes.csv`, `tables/oe1_cuadrantes_conteo.csv` | Cuadrantes (lista y conteos) |
| `tables/oe1_q1_clusters.csv` … `tables/oe1_q4_fusiones_P.csv` | Preguntas 1–4 |
| `tables/oe1_informacion_retenida.csv` | u de cada una de las 23 descartadas respecto de su representante y de la mejor conservada |
| `tables/oe1_recomendado_cambios.csv`, `tables/oe1_recomendado_transitividad.csv` | Evidencia del conjunto recomendado |
| `figures/demo_no_lineal.png` | Test 4 |
| `figures/oe1_entropia_features.png` | Entropía por feature |
| `figures/oe1_heatmap_pearson_vs_nmi.png` | \|r\| frente a nmi_sqrt |
| `figures/oe1_scatter_r_vs_rinfo.png` | Cuadrantes |

**Caché** (no versionada; `data/processed/` está en `.gitignore`): `data/processed/oe1_mi_cache/eda_df_clean_74.parquet`.
