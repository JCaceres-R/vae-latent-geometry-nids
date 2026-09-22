# Informe técnico — OE2 robusto (multi-semilla, geometría distribucional, detectabilidad confirmatoria)

> **Estado: BORRADOR EN CONSTRUCCIÓN.** Este archivo se está llenando a medida
> que cada fase del pipeline (`src/vae_nids/evaluation/oe2_robust/`) termina
> de correr. Las secciones marcadas `[PENDIENTE]` se completan al final,
> cuando las 10 semillas hayan pasado por las Fases 1-6. Cada número citado
> en este informe proviene de un archivo en `outputs/oe2_robust/tables/` o
> `outputs/oe2_robust/metrics/` -- ninguno se escribe a mano.

## 1. Resumen ejecutivo

[PENDIENTE -- se completa al final con: qué se hizo, resultado de la
hipótesis principal (rho combinado, IC95, comparación con las 10 semillas
individuales), y si el resultado del OE2 anterior (rho=0.38, p=0.25, una
sola semilla) se mantiene, cambia o se invierte.]

## 2. Respuesta a cada crítica del director

### 2.1 Crítica 1 — Fase 3 de geometría demasiado determinista (un número por familia sobre mu)

Implementado en `phase3_geometry.py`:
- Distribución completa de Mahalanobis (no solo mediana): media, mediana,
  p5/p25/p75/p95, IQR, por familia y semilla.
- Pruebas formales familia vs. benigno sobre la distribución completa
  (Mann-Whitney U, Kolmogorov-Smirnov de 2 muestras), con corrección de
  Holm sobre las 18 familias.
- Tamaño de efecto (delta de Cliff) con IC bootstrap.
- Sensibilidad con covarianza robusta (MinCovDet) y comparación de
  rankings.
- Divergencias entre nubes completas: MMD (RBF), Wasserstein deslizada,
  distancia de energía, con repeticiones de submuestreo e IC.
- Métricas que usan sigma^2 del posterior: Bhattacharyya (medias + forma
  por separado), Mahalanobis esperada bajo el posterior (L=20 muestras),
  incertidumbre posterior media.
- Solapamiento/densidad local: pureza de vecindad (kNN), densidad benigna
  local (kNN-density) evaluada en los puntos de ataque.
- Dispersión intra-clase (traza/log-det de covarianza) y silhouette con 20
  submuestreos repetidos.

[PENDIENTE -- lectura de qué mostró.]

### 2.2 Crítica 2 — Fase 4 de detectabilidad incompleta (un solo score, una sola métrica)

Implementado en `phase4_detectability.py`: 5 scores de anomalía
(`nll_mu`, `nll_mc` con L=20 muestras del posterior -- score principal --,
`neg_elbo`, `mse`, `latent_maha`), AUC-ROC/AUC-PR/pAUC parcial (McClish,
FPR<=0.10)/TPR a FPR fijo (1%, 5%, 10%) con umbrales calibrados en
val_benign y verificados en test_benign, todo con IC95 bootstrap. Pruebas
de DeLong entre familias (con negativos compartidos) y entre scores
(pareado), con corrección de Holm. Control de tamaño de muestra
(submuestreo del benigno al mismo n de cada familia, 100 repeticiones).

[PENDIENTE -- lectura de qué mostró.]

### 2.3 Crítica 3 — una sola semilla no permite afirmar nada sobre geometría-detectabilidad

10 semillas entrenadas de forma independiente (`phase1_train_multiseed.py`,
mismos splits/scaler/arquitectura, solo cambia la semilla del modelo). Toda
métrica de las Fases 3-5 se reporta por semilla y combinada (media entre
semillas, IC, y para el par principal, combinación por transformada de
Fisher). Estabilidad de rankings entre semillas cuantificada con W de
Kendall (Fase 7.4).

[PENDIENTE -- lectura de qué mostró.]

## 3. Configuración y reproducibilidad

- **Hardware:** [PENDIENTE -- de `phase0_diagnostics.json`]
- **Versiones de librerías:** ver `*_metadata.json` de cada checkpoint
  (`outputs/oe2_robust/checkpoints/`).
- **Commit de git:** ver `*_metadata.json` de cada checkpoint.
- **Semillas usadas:** [PENDIENTE]
- **Tiempos de cómputo por fase:** [PENDIENTE]
- **Verificación de reproducción del checkpoint seed=42:** [PENDIENTE -- de
  `phase0_diagnostics.json`, campo `reproducibility`]

## 4. Resultados por fase

### 4.1 Fase 0 — Diagnóstico y presupuesto

[PENDIENTE]

### 4.2 Fase 1 — Entrenamiento multi-semilla

[PENDIENTE -- tabla resumen de `phase1_seed_summary.csv`]

### 4.3 Fase 2 — Codificación y dimensiones activas por semilla

[PENDIENTE -- tabla de `phase2_active_dims_per_seed.csv`]

### 4.4 Fase 3 — Geometría distribucional

[PENDIENTE -- resúmenes de `phase3_*.csv`]

### 4.5 Fase 4 — Detectabilidad confirmatoria

[PENDIENTE -- resúmenes de `phase4_*.csv`]

### 4.6 Fase 5 — Correlación geometría-detectabilidad

[PENDIENTE -- resúmenes de `phase5_*.csv` y `.json`]

### 4.7 Fase 6 — Visualización

Las 7 figuras obligatorias (`fig1`-`fig7`) se generan y validan sobre
seed=42 sin errores (ver `outputs/oe2_robust/figures/`). **La proyección
UMAP queda pendiente por decisión explícita durante la ejecución** (ajustar
UMAP sobre ~330k puntos de una sola semilla resultó computacionalmente
costoso bajo la carga concurrente del entrenamiento multi-semilla; se
pospuso para correrla aparte, sin bloquear el resto del pipeline, una vez
liberado el cómputo del entrenamiento). Se retoma en la sección 10.

[PENDIENTE -- completar inventario final de figuras, incluida la UMAP.]

## 5. Hipótesis principal

[PENDIENTE -- de `phase5_hypothesis_principal.json`]

## 6. Análisis secundarios

[PENDIENTE -- tabla completa de `phase5_secondary_summary.csv`, p crudo y
ajustado (Holm, BH)]

## 7. Análisis a nivel de conexión y estabilidad entre semillas

### 7.1 Modelo de efectos mixtos

[PENDIENTE -- de `phase5_mixed_model.json`]

### 7.2 Estabilidad entre semillas (W de Kendall)

[PENDIENTE -- de `phase5_seed_stability_kendalls_w.csv`]

## 8. Revisión de afirmaciones del OB2 anterior

| # | Afirmación del OB2 anterior (seed=42, una corrida) | Veredicto | Evidencia nueva |
|---|---|---|---|
| 1 | Correlación Mahalanobis (mediana) vs. AUC-ROC no significativa (rho=0.3818, p=0.2466) | [PENDIENTE] | |
| 2 | Ninguna de las 6 combinaciones geometría-detectabilidad es significativa | [PENDIENTE] | |
| 3 | Infiltration es la familia más detectable (con el error de posición de mediana ya identificado: es la 9.ª de 11, no está "por debajo de la mediana") | [PENDIENTE] | |
| 4 | SSH_Patator es la familia más cercana y aun así de las más detectables | [PENDIENTE] | |
| 5 | Solo 2 de 6 pares completado/attempted confirman la hipótesis; DoS_Hulk es una inversión | [PENDIENTE] | |
| 6 | AUC-ROC con mu vs. z muestreado prácticamente no cambia (dif. media ~0.003) | [PENDIENTE] | |
| 7 | Dos dimensiones colapsadas (2 y 7) de forma estable | [PENDIENTE] | |
| 8 | tau = percentil 95 generaliza (FPR test = 5.0776%) | [PENDIENTE] | |

## 9. Limitaciones

[PENDIENTE -- incluir explícitamente: potencia estadística con n=11
familias (rho mínimo detectable, potencia para rho=0.3/0.5/0.7); supuestos
del DeLong con negativos compartidos; alcance de la submuestra en
divergencias/silhouette/bootstrap; qué NO se hizo (Experimento 4, excluido
también aquí, igual que en la monografía v9).]

## 10. Decisiones tomadas por el agente y desviaciones del plan

- Reentrenamiento de verificación de seed=42 (Fase 0): usado también como
  corrida de referencia de tiempo/presupuesto.
- [PENDIENTE -- registrar aquí cualquier reducción de repeticiones/tamaños
  de submuestra frente a lo pedido en el prompt, con la razón computacional
  y el momento en que se decidió (antes de ver resultados de geometría/
  detectabilidad, para no sesgar).]
- Proyección UMAP (Fase 6) pospuesta durante la ejecución del pipeline: se
  corre por separado después de las Fases 1-5, para no competir por CPU
  con el entrenamiento de las 9 semillas nuevas ni con el resto de las
  fases (todas más baratas computacionalmente). No afecta ningún resultado
  cuantitativo -- es apoyo visual, explícitamente no-evidencia según el
  propio prompt (Sección 8).
- **Cómputo movido de la máquina local a una VM Linux, por decisión del
  autor** (no del agente): las Fases 0-2 (diagnóstico, entrenamiento de las
  10 semillas, codificación) y una validación completa de las Fases 3-5
  sobre seed=42 (confirmando que reproducen los números del OE2 original:
  mediana de Mahalanobis exacta, rho de Spearman exacto = 0.3818) se
  corrieron localmente en Windows/CPU. El resto (Fases 3-5 sobre las 9
  semillas nuevas, Fase 6 completa con UMAP) se corre en una VM Linux vía
  `scripts/run_oe2_robust_vm.sh`, que reutiliza los checkpoints ya
  entrenados (versionados en git, así que la VM no reentrena) y es
  resumible por semilla si la VM se interrumpe. Log de esa corrida en
  `outputs/oe2_robust/VM_RUN_LOG.md`.
- Reducción de `SILHOUETTE_MAX_N`/`DIVERGENCE_MAX_N` de 5000 a 1500 por
  lado (ver `config.py`), y de la estrategia de IC de AUC-ROC/AUC-PR (de
  bootstrap ingenuo de 1000 remuestras sobre grupos de hasta ~400k filas a
  varianza analítica de DeLong para AUC-ROC, y bootstrap submuestreado para
  AUC-PR) -- ambas decisiones se tomaron en la Fase 0/inicio de la Fase 3-4,
  antes de ver ningún resultado de geometría o detectabilidad, por
  inviabilidad computacional medida empíricamente (ver Sección 9).
- Comparación entre familias de AUC-ROC (DeLong) implementada como una
  generalización del DeLong pareado clásico para el caso de negativos
  compartidos y positivos independientes (documentado en
  `stats_utils.delong_test_shared_negatives`), porque las familias no
  comparten las mismas observaciones de ataque (solo comparten el benigno
  de test) -- no es el caso de uso estándar de DeLong (mismos casos, dos
  clasificadores).
- Modelo de efectos mixtos: semilla incorporada como componente de
  varianza (`vc_formula`) evaluado dentro de cada grupo-familia, no como
  efecto cruzado global independiente -- limitación de la implementación
  de efectos cruzados en `statsmodels.MixedLM`, documentada explícitamente
  en 7.1.

## 11. Inventario de archivos generados

[PENDIENTE -- tabla generada de forma semi-automática al final]

## 12. Recomendaciones para la redacción de la monografía

[PENDIENTE]
