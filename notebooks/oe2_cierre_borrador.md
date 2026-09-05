# Cierre de OE2 — Geometría Comparativa y Correlación con Detectabilidad

**BORRADOR de compilación.** Estructura calcada de "Cierre de OE1 —
Estabilidad del Espacio Latente" para mantener consistencia entre
capítulos de la monografía. Este documento **compila** los resultados ya
generados en `notebooks/oe2_diagnostico_inicial.md`,
`oe2_fase2_encoding.md`, `oe2_fase3_geometric_metrics.md`,
`oe2_fase4_detectability.md`, `oe2_fase5_correlation.md`, el script/figura
de la Fase 6, y los dos análisis adicionales post-cierre
(`oe2_tarea1_tau_comparison.md`, `oe2_tarea2_mu_vs_z_factibilidad.md`) —
**no reinterpreta ni recalcula nada**. Los puntos marcados
**[PENDIENTE DE DECISIÓN]** requieren una decisión del autor antes de
poder cerrar el capítulo.

---

## 1. Resumen

OE2 (semanas 3-6 del cronograma) responde a la pregunta central de la
monografía: proyectando cada familia de ataque de CICIDS2017 sobre el
espacio latente de un VAE entrenado exclusivamente con tráfico benigno
(línea base de OE1), ¿la distancia geométrica de una familia respecto al
benigno se correlaciona con qué tan detectable es esa familia? Se
evaluaron 11 familias de ataque principales más 7 grupos adicionales
(variantes `- Attempted` para una comparación secundaria), sobre el
checkpoint oficial fijado en el cierre de OE1 (seed=42, input_dim=51,
época 52). El resultado central (Fase 5) es que **la correlación de
Spearman entre geometría (distancia de Mahalanobis, silhouette) y
detectabilidad (AUC-ROC, TPR) no es estadísticamente significativa**
(rho=0.3818, p=0.2466 para el par principal, n=11) — un hallazgo que se
reporta tal cual, no se suaviza ni se interpreta como éxito parcial.

## 2. Línea base heredada de OE1

Checkpoint oficial usado en todo OE2, verificado (no reentrenado) al
inicio de la Fase 2:

| Campo | Valor |
|---|---|
| Ruta | `outputs/checkpoints/vae_k8_beta1_input51_best.pt` |
| seed | 42 |
| época | 52 |
| val_loss | -185.27911716408389 |
| input_dim | 51 |
| latent_dim (J) | 8 |
| β | 1.0 |
| hidden_dim | 32 |
| logvar_min / logvar_max | -10.0 / 10.0 |

Dimensionalidad efectiva heredada de OE1: 5-6/8 dimensiones activas según
semilla (barrido de 5 semillas, media 5.4/8) — ver `oe1_report.md`. OE2
**recalculó** las dims activas de forma independiente en su propia Fase
3 en vez de asumir el resultado de OE1 (ver Sección 5).

## 3. Fase 1 — Diagnóstico inicial

Diagnóstico de solo lectura antes de tocar el pipeline. Detalle completo:
`oe2_diagnostico_inicial.md`.

**Taxonomía real de labels** en `test_attacks.parquet` (n=442,952, columna
`Label`, 23 valores distintos) — se verificó la taxonomía real en vez de
asumir nombres genéricos; incluye subtipos DoS por separado (Hulk,
GoldenEye, slowloris, Slowhttptest) y las variantes `- Attempted` como
clases propias, consistente con la decisión de preprocesamiento de OE1 de
no fusionarlas con la clase base.

Confirmación de estado de git (checkpoints y `.py` de modelo/training/
evaluation ya trackeados) y del checkpoint oficial (existe, seed/época
correctos) antes de proceder.

## 4. Fase 2 — Codificación a mu

Script: `src/vae_nids/evaluation/encode_latent_oe2.py`. Detalle:
`oe2_fase2_encoding.md`.

Cada muestra se codifica a **mu = E_φ[z|x]** (media de q_φ(z|x)),
**nunca** z muestreado con el reparameterization trick — el objetivo de
esta fase es un punto fijo por muestra para las métricas geométricas de
la Fase 3, no una variable aleatoria. Modelo cargado en modo eval, con
verificación de que el checkpoint es el oficial (seed=42, época=52)
antes de codificar.

19 archivos `.npy` (todos shape `(n, 8)`) en `outputs/latent_vectors/` +
`manifest.json`: 11 familias principales (PortScan, DoS_Hulk, DDoS,
DoS_GoldenEye, DoS_slowloris, FTP_Patator, SSH_Patator, DoS_Slowhttptest,
Bot, Web_Attack [fusión de 3 subtipos de Web Attack], Infiltration) + 7
grupos para la comparación secundaria completado/attempted (5 variantes
`_attempted` de familias ya cubiertas, más el par `WebAttack_BruteForce`
/`WebAttack_BruteForce_attempted` aislado de la fusión Web_Attack) +
referencia benigna (`latent_benign_test.npy`, n=248,561).

Excluidos deliberadamente por tamaño de muestra insuficiente:
`Heartbleed` (n=11), `FTP-Patator - Attempted` (11),
`SSH-Patator - Attempted` (8), `Infiltration - Attempted` (16),
`Web Attack - XSS - Attempted` (652 — no se pidió para esta fase).

Verificación de consistencia: suma de filas codificadas (443,103,
incluyendo el solape intencional de 151 filas de Brute Force contadas
tanto en `Web_Attack` como en `WebAttack_BruteForce`) menos ese solape =
442,952 = total exacto de `test_attacks.parquet`.

## 5. Fase 3 — Métricas geométricas (Mahalanobis + silhouette)

Script: `src/vae_nids/evaluation/geometric_metrics_oe2.py`. Detalle:
`oe2_fase3_geometric_metrics.md`.

### 5.1. Verificación de dimensiones activas (recalculada, no asumida)

Mismo criterio de Burda et al. (2016, umbral 0.01) y misma función que
OE1, pero recalculado sobre `latent_benign_test.npy` en vez de reusar el
resultado del cierre de OE1 (que era de otra corrida, sobre `val_benign`):

| dim | Var_x[mu_j] | Estado |
|---:|---:|---|
| 0 | 0.572597 | activa |
| 1 | 0.255990 | activa |
| **2** | 0.000009 | **colapsada — excluida** |
| 3 | 0.953659 | activa |
| 4 | 0.965305 | activa |
| 5 | 0.255043 | activa |
| 6 | 0.773592 | activa |
| **7** | 0.000028 | **colapsada — excluida** |

Dimensiones excluidas: 2 y 7 (mismas que en el cierre de OE1 para la
corrida seed=42/input51, confirmado de forma independiente). **Dims
activas usadas en las Fases 3, 5 y 6: `[0, 1, 3, 4, 5, 6]` (6/8).**

### 5.2. Tabla completa (18 grupos)

Centroide y covarianza del benigno calculados solo sobre las 6 dims
activas, con `latent_benign_test.npy` completo (n=248,561). Mahalanobis
por muestra contra ese benigno; silhouette (familia vs. benigno) con
submuestreo a 5,000 por grupo si supera ese tamaño (seed=42).

| group | n_total | n_silhouette | maha_mean | maha_median | maha_p25 | maha_p75 | silhouette |
|---|---:|---:|---:|---:|---:|---:|---:|
| PortScan | 159,023 | 10,000 | 2.875 | 2.871 | 2.728 | 2.883 | 0.4989 |
| DoS_Hulk | 158,469 | 10,000 | 3.571 | 3.086 | 2.944 | 3.996 | 0.4561 |
| DDoS | 95,123 | 10,000 | 5.821 | 5.726 | 4.848 | 6.782 | 0.4877 |
| DoS_GoldenEye | 7,567 | 10,000 | 4.424 | 4.060 | 3.249 | 5.395 | 0.3134 |
| DoS_slowloris | 4,001 | 9,001 | 6.492 | 6.265 | 3.264 | 8.818 | 0.3557 |
| FTP_Patator | 3,973 | 8,973 | 2.004 | 2.004 | 2.000 | 2.008 | 0.4521 |
| SSH_Patator | 2,980 | 7,980 | 1.388 | 1.375 | 1.369 | 1.383 | 0.3175 |
| DoS_Slowhttptest | 1,742 | 6,742 | 5.053 | 3.562 | 3.404 | 5.985 | 0.3443 |
| Bot | 738 | 5,738 | 2.272 | 1.794 | 1.791 | 1.796 | 0.1749 |
| Web_Attack | 190 | 5,190 | 3.850 | 3.578 | 2.730 | 5.194 | 0.2059 |
| Infiltration | 32 | 5,032 | 35.606 | 4.351 | 3.399 | 27.184 | 0.8820 |
| DoS_Hulk_attempted | 579 | 5,579 | 5.119 | 5.594 | 2.861 | 6.305 | 0.3678 |
| DoS_GoldenEye_attempted | 80 | 5,080 | 2.923 | 2.860 | 2.860 | 2.861 | 0.1520 |
| DoS_slowloris_attempted | 1,706 | 6,706 | 3.641 | 3.705 | 3.705 | 3.705 | 0.4527 |
| DoS_Slowhttptest_attempted | 3,367 | 8,367 | 4.919 | 5.353 | 5.352 | 5.354 | 0.4916 |
| Bot_attempted | 1,470 | 6,470 | 2.864 | 2.864 | 2.863 | 2.865 | 0.3499 |
| WebAttack_BruteForce | 151 | 5,151 | 3.904 | 3.581 | 2.736 | 5.205 | 0.2086 |
| WebAttack_BruteForce_attempted | 1,214 | 6,214 | 2.169 | 2.160 | 2.159 | 2.163 | 0.2566 |

Nota metodológica: `Infiltration` (n=32) muestra media=35.6 vs.
mediana=4.35 — un outlier extremo domina el promedio en un grupo chico,
razón por la que la Fase 5 usa la mediana como métrica primaria de
Mahalanobis (y reporta también la media, sin ocultar la diferencia).

## 6. Fase 4 — Detectabilidad por familia (AUC-ROC / TPR)

Script: `src/vae_nids/evaluation/detectability_oe2.py`. Detalle:
`oe2_fase4_detectability.md`.

**Score de anomalía**: NLL de reconstrucción Gaussiana heteroscedástica
por muestra (An & Cho, 2015; misma forma funcional que la ecuación del
documento de fundamentos), **sin** el término β·KL, **sin** MSE plano.
Score más alto = más anómalo. Usa las **8 dimensiones completas** de mu
(no las 6 activas de la Fase 3 — el decoder espera las 8 tal como las
produce el encoder).

**τ = -156.9726** (percentil 95 de los scores de reconstrucción sobre
`val_benign.parquet` completo, n=248,560 — split fresco, nunca tocado
antes de esta fase, para no contaminar el umbral con datos de
evaluación). **[PENDIENTE DE DECISIÓN — ver Sección 9.1]**: la versión
vigente de la monografía especifica percentil 99, no 95.

AUC-ROC con IC 95% por bootstrap **estratificado** (positivos y
negativos remuestreados por separado con tamaño fijo, 1000 iteraciones,
seed=42 — evita perder una clase entera en grupos chicos como
Infiltration).

| group | n | AUC-ROC | CI95 low | CI95 high | ancho CI | TPR@τ |
|---|---:|---:|---:|---:|---:|---:|
| PortScan | 159,023 | 0.6759 | 0.6743 | 0.6776 | 0.0034 | 0.0002 |
| DoS_Hulk | 158,469 | 0.9176 | 0.9166 | 0.9186 | 0.0021 | 0.0673 |
| DDoS | 95,123 | 0.9696 | 0.9691 | 0.9702 | 0.0011 | 0.7937 |
| DoS_GoldenEye | 7,567 | 0.9512 | 0.9501 | 0.9523 | 0.0022 | 0.4761 |
| DoS_slowloris | 4,001 | 0.9410 | 0.9375 | 0.9449 | 0.0074 | 0.8065 |
| FTP_Patator | 3,973 | 0.9327 | 0.9317 | 0.9337 | 0.0020 | 0.0035 |
| SSH_Patator | 2,980 | 0.9824 | 0.9811 | 0.9835 | 0.0024 | 0.9842 |
| DoS_Slowhttptest | 1,742 | 0.9908 | 0.9897 | 0.9916 | 0.0018 | 0.9943 |
| Bot | 738 | 0.6448 | 0.6377 | 0.6520 | 0.0143 | 0.0596 |
| Web_Attack | 190 | 0.9444 | 0.9322 | 0.9556 | 0.0234 | 0.6000 |
| Infiltration | 32 | 0.9928 | 0.9885 | 0.9959 | 0.0074 | 0.9688 |
| DoS_Hulk_attempted | 579 | 0.9358 | 0.9284 | 0.9427 | 0.0144 | 0.7323 |
| DoS_GoldenEye_attempted | 80 | 0.7945 | 0.7885 | 0.8021 | 0.0136 | 0.0250 |
| DoS_slowloris_attempted | 1,706 | 0.9728 | 0.9708 | 0.9748 | 0.0041 | 0.9578 |
| DoS_Slowhttptest_attempted | 3,367 | 0.9858 | 0.9851 | 0.9864 | 0.0014 | 0.8931 |
| Bot_attempted | 1,470 | 0.6128 | 0.6109 | 0.6148 | 0.0039 | 0.0000 |
| WebAttack_BruteForce | 151 | 0.9627 | 0.9540 | 0.9708 | 0.0168 | 0.6358 |
| WebAttack_BruteForce_attempted | 1,214 | 0.7475 | 0.7458 | 0.7496 | 0.0037 | 0.0000 |

El ancho de CI más grande no corresponde al n más chico: `Infiltration`
(n=32) tiene CI angosto (0.0074) por su AUC casi perfecto (0.9928), 
mientras que `Web_Attack` (n=190, AUC más ambiguo) tiene el CI más ancho
de los 18 (0.0234) — la varianza del bootstrap depende tanto de n como
de qué tan lejos esté el AUC de 0.5.

## 7. Fase 5 — Correlación geometría-detectabilidad (RESULTADO CENTRAL)

Script: `src/vae_nids/evaluation/correlation_oe2.py`. Detalle:
`oe2_fase5_correlation.md`.

Une Fase 3 + Fase 4 por `group` (18 filas, verificado 1 a 1). Spearman
sobre las **11 familias principales** (excluye los 7 grupos
`_attempted`/`WebAttack_BruteForce` standalone):

| Par | rho | p-valor | Significativo (p<0.05) |
|---|---:|---:|---|
| **mahalanobis_median vs auc_roc** | **0.3818** | **0.2466** | **No** |
| mahalanobis_median vs tpr_at_tau | 0.3364 | 0.3118 | No |
| silhouette_score vs auc_roc | 0.1727 | 0.6115 | No |
| silhouette_score vs tpr_at_tau | -0.0455 | 0.8944 | No |
| mahalanobis_mean vs auc_roc | 0.5000 | 0.1173 | No |
| mahalanobis_mean vs tpr_at_tau | 0.4727 | 0.1420 | No |

**Ninguna de las 6 correlaciones es estadísticamente significativa con
n=11.** Esto se reporta tal cual — no es un resultado nulo por error
metodológico (dims activas verificadas de forma independiente, score de
anomalía consistente con la formulación teórica, splits sin fuga de
datos, bootstrap estratificado para evitar sesgo en grupos chicos): es
el hallazgo en sí.

Figura: `outputs/figures/oe2_geometry_vs_detectability.png` (Mahalanobis
mediana vs. AUC-ROC, 11 familias etiquetadas, **sin línea de tendencia
forzada** porque rho no es significativo).

**Caso que contradice la hipótesis — `Infiltration`**: es la familia con
el **AUC-ROC más alto de las 18** (0.9928), pese a **no ser la
geométricamente más cercana** al benigno (SSH_Patator, mediana=1.375, y
Bot, mediana=1.794, están más cerca) — su distancia Mahalanobis mediana
(4.351) está por debajo de la mediana del conjunto de 11 familias, una
distancia moderada-baja, no la mínima. Es decir: una familia con
distancia geométrica solo moderada resulta ser la más detectable de
todas — exactamente el patrón que la hipótesis de OE2 (más lejos → más
detectable) no predice.

**Comparación secundaria completado/attempted** (6 pares, delta = completo
- attempted):

| Par | Δ mahalanobis_median | Δ auc_roc | Mismo signo | Consistente con hipótesis |
|---|---:|---:|---|---|
| DoS_Hulk vs. attempted | -2.5082 | -0.0182 | Sí | **No** (ambos negativos) |
| DoS_GoldenEye vs. attempted | +1.2001 | +0.1567 | Sí | Sí |
| DoS_slowloris vs. attempted | +2.5599 | -0.0317 | No | No |
| DoS_Slowhttptest vs. attempted | -1.7907 | +0.0050 | No | No |
| Bot vs. attempted | -1.0694 | +0.0320 | No | No |
| WebAttack_BruteForce vs. attempted | +1.4207 | +0.2152 | Sí | Sí |

**Solo 2/6 pares confirman la hipótesis** (completo más lejos Y más
detectable que attempted: `DoS_GoldenEye` y `WebAttack_BruteForce`). El
caso `DoS_Hulk` es notable: mismo signo en ambos deltas, pero **ambos
negativos** — el ataque completo está *más cerca* del benigno y es
*menos* detectable que su variante Attempted, lo opuesto de lo esperado.
La comparación secundaria **no respalda** una relación sistemática en la
dirección esperada.

### Conclusión de la Fase 5

Con n=11 familias, **no hay evidencia estadística de que la distancia
geométrica en el espacio latente (Mahalanobis o silhouette) prediga la
detectabilidad de una familia de ataque**. La comparación
completado/attempted tampoco confirma la hipótesis de forma sistemática
(2/6 pares).

## 8. Fase 6 — Proyección UMAP del espacio latente (apoyo visual)

Script: `src/vae_nids/evaluation/umap_projection_oe2.py`. Figura:
`outputs/figures/oe2_latent_umap_projection.png`. **Esta fase es
únicamente apoyo visual — no valida ni reemplaza el resultado
cuantitativo de la Fase 5.**

Proyección: `UMAP(random_state=42, n_neighbors=15, min_dist=0.1,
n_components=2)`, ajustada una sola vez sobre benigno submuestreado
(n=3,000, seed=42) + las 11 familias principales completas, recortadas a
las mismas 6 dims activas de la Fase 3 (`[0, 1, 3, 4, 5, 6]`) — una única
proyección conjunta, nunca una por familia por separado (quedarían en
sistemas de coordenadas no comparables).

**Lectura visual (cualitativa, no reemplaza los números)**: la mayoría de
las familias de ataque —incluyendo `DoS_Hulk` (naranja) y `DDoS` (verde),
las de mayor volumen— ocupan una región densa y muy solapada en el centro
de la proyección, con los centroides de la mayoría de las familias
agrupados relativamente cerca entre sí. `FTP_Patator` es la excepción más
visible: su centroide queda claramente aislado del resto, en el borde
inferior-izquierdo de la nube central. El benigno submuestreado aparece
como una nube más dispersa que rodea el núcleo denso de ataques.
**Limitación de la figura a corregir**: `BENIGN` y `PortScan` quedaron
con tonos de azul/lavanda muy parecidos en la paleta usada
(`tab20`), lo que dificulta distinguirlos a simple vista en la nube
externa — no se puede leer con confianza si esa región dispersa es
benigno, PortScan, o ambos mezclados. No se detectó ningún caso de una
familia con Mahalanobis mediana muy baja (ej. `SSH_Patator`, 1.375)
apareciendo visualmente lejísimos del resto — su centroide se ubica
dentro del núcleo denso central junto con las demás familias, consistente
con (no contradice) su distancia geométrica baja. Dicho esto, UMAP
preserva estructura de vecindad **local**, no distancias globales — la
posición relativa en este plot no es una prueba cuantitativa y debe
leerse solo como apoyo exploratorio.

## 9. Análisis adicional post-cierre

Dos análisis realizados después de cerrar y commitear las Fases 1-6,
para informar decisiones pendientes sobre la Fase 4. Ninguno modifica los
resultados ya reportados en las secciones anteriores.

### 9.1. Impacto del percentil de τ (95 vs. 99)

Detalle: `oe2_tarea1_tau_comparison.md`, tabla en
`outputs/metrics/oe2_tau_comparison.csv`.

τ con percentil 99 (especificado en la versión vigente de la monografía)
= -132.5056, frente a τ=-156.9726 con percentil 95 (el que usa
actualmente `detectability_oe2.py`). El cambio de percentil **no afecta
AUC-ROC** (no depende de τ), solo TPR@τ.

Cambios más dramáticos en TPR@τ al pasar de percentil 95 a 99:

| group | TPR@τ95 | TPR@τ99 | Δ |
|---|---:|---:|---:|
| DoS_slowloris_attempted | 0.9578 | 0.0023 | **-0.9555** |
| SSH_Patator | 0.9842 | 0.2369 | -0.7473 |
| DDoS | 0.7937 | 0.1446 | -0.6491 |

`DoS_slowloris_attempted` colapsa de detección casi total a
prácticamente nula. `SSH_Patator` y `DDoS` (esta última con volumen
considerable, n=95,123) también caen con fuerza. **[PENDIENTE DE
DECISIÓN]**: qué percentil usar como τ oficial de la monografía — 95
(código actual) o 99 (spec vigente) — sopesando este costo en TPR contra
la reducción de falsos positivos esperada de un umbral más conservador
(no cuantificada en este análisis, que solo midió TPR).

### 9.2. Factibilidad de usar z muestreado en vez de mu (Fase 4)

Detalle: `oe2_tarea2_mu_vs_z_factibilidad.md`.

El reparameterization trick ya existe (`VAE.reparameterize`,
`vae.py:89-94`) y ya está encadenado en `VAE.forward`. Cambiar
`detectability_oe2.py` para usar z en vez de mu es un cambio de código
pequeño, pero no cosmético: introduce no-determinismo (requiere fijar
semilla de torch explícitamente) y una decisión de diseño real —¿una
sola muestra de z por fila, o promedio sobre K muestras para un
estimado Monte Carlo más estable?— que cambia qué significa el score
resultante. No requiere tocar la Fase 2 (el encoder puede correrse fresco
sobre la x ya recargada en `detectability_oe2.py`, sin depender de los
`.npy` de mu guardados, que deliberadamente no incluyen `logvar`).
**[PENDIENTE DE DECISIÓN]**: si vale la pena rehacer la Fase 4 con z en
vez de mu, y en tal caso, con qué K.

## 10. Conclusión de OE2

OE2 se completó con las 6 fases planificadas más dos análisis de
sensibilidad adicionales. El **resultado central es negativo respecto a
la hipótesis de partida**: no se encontró correlación estadísticamente
significativa entre la geometría del espacio latente (distancia de
Mahalanobis, silhouette score) y la detectabilidad de una familia de
ataque (AUC-ROC, TPR), con n=11 familias principales (mejor caso:
rho=0.50, p=0.117, usando la media de Mahalanobis en vez de la mediana).
La comparación secundaria completado/attempted tampoco confirma la
hipótesis de forma sistemática (2/6 pares). El caso de `Infiltration`
—máxima detectabilidad con distancia geométrica solo moderada— es el
ejemplo más claro de esta desconexión. Este resultado queda documentado
tal cual, sin suavizarlo: la evidencia no respalda que la geometría del
espacio latente por sí sola sea un predictor de detectabilidad en este
VAE y este dataset.

## 11. Pendientes de decisión y alcance

- **[PENDIENTE DE DECISIÓN]** Percentil de τ para la Fase 4 oficial: 95
  (código actual, ya reportado en la Sección 6) vs. 99 (spec vigente de
  la monografía) — ver Sección 9.1 para el costo cuantificado en TPR.
- **[PENDIENTE DE DECISIÓN]** Uso de mu (determinístico, actual) vs. z
  muestreado (estocástico, requiere fijar semilla y decidir K) como
  entrada al decoder para el score de anomalía de la Fase 4 — ver
  Sección 9.2.
- **[PENDIENTE DE DECISIÓN]** Si el Experimento 4 (resistencia a la
  evasión) entra en el alcance de esta monografía o queda fuera — no
  se abordó en ninguna fase de OE2 hasta ahora y no hay trabajo
  preliminar al respecto en este repo.

## 12. Referencias

- Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational
  Bayes*. arXiv:1312.6114.
- Burda, Y., Grosse, R., & Salakhutdinov, R. (2016). *Importance
  Weighted Autoencoders*. ICLR 2016.
- An, J., & Cho, S. (2015). *Variational Autoencoder based Anomaly
  Detection using Reconstruction Probability*. SNU Data Mining Center
  Technical Report.
- Engelen, G., Rimmer, V., & Joosen, W. (2021). *Troubleshooting an
  Intrusion Detection Dataset: the CICIDS2017 Case Study*. IEEE SPW
  (WTMC).
- Mahalanobis, P. C. (1936). *On the Generalised Distance in
  Statistics*. Proceedings of the National Institute of Sciences of
  India.
- Rousseeuw, P. J. (1987). *Silhouettes: A Graphical Aid to the
  Interpretation and Validation of Cluster Analysis*. Journal of
  Computational and Applied Mathematics.
- Spearman, C. (1904). *The Proof and Measurement of Association
  between Two Things*. American Journal of Psychology.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the
  Bootstrap*. Chapman & Hall/CRC.
- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform
  Manifold Approximation and Projection for Dimension Reduction*.
  arXiv:1802.03426.
