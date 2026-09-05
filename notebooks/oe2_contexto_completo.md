# OE2 — Contexto completo y estado actual

Documento de traspaso: todo lo hecho en OE2 hasta ahora (Fases 1-5
cerradas, Fase 6 pendiente de correr en otra máquina por costo
computacional). Pensado para que alguien sin contexto previo pueda
retomar el trabajo — incluye parámetros exactos, código, resultados y
estado de git.

## 0. Qué es OE2

Segundo Objetivo Específico del proyecto: **"Geometría comparativa y
correlación con detectabilidad"** (semanas 3-6 del cronograma de 13
semanas). Es el **hallazgo central** de la monografía — OE1 valida que el
fenómeno se puede medir, OE2 es el hallazgo central, OE3/OE4 validan que
ese hallazgo es legítimo y robusto (`README.md`).

Pregunta que responde OE2: proyectando cada familia de ataque de
CICIDS2017 sobre el espacio latente de un VAE entrenado *solo* con
tráfico benigno, ¿la distancia geométrica de una familia al benigno
(Mahalanobis, silhouette) se correlaciona con qué tan detectable es esa
familia por el VAE (AUC-ROC, TPR)?

## 1. Línea base heredada de OE1

OE1 cerró con un espacio latente J=8 (arquitectura fija, ver
`notebooks/oe1_report.md` y el PDF "Cierre de OE1"), dimensionalidad
efectiva 5-6/8 dimensiones activas dependiendo de semilla (barrido de 5
semillas, media 5.4/8).

**Checkpoint oficial fijado para todo OE2** (verificado, no reentrenado):

```
outputs/checkpoints/vae_k8_beta1_input51_best.pt
```

| Campo | Valor |
|---|---|
| seed | 42 |
| epoch | 52 |
| val_loss | -185.27911716408389 |
| input_dim | 51 |
| latent_dim (J) | 8 |
| beta | 1.0 |
| hidden_dim | 32 |
| logvar_min / logvar_max | -10.0 / 10.0 |

Los 6 checkpoints (`input51_best` + 4 semillas del barrido + `input74_best`)
están trackeados en git desde el commit `7e3b8c7` ("track OE1 checkpoints
(seed sweep + base run)").

## 2. Pipeline de datos y dependencias

- Dataset: CICIDS2017 corregido (Engelen et al., 2021), 74→51 features
  tras deduplicar (3 cols) y colapsar 13 clusters correlacionados por
  clique estricto (`|r|≥0.95`) — ver `src/vae_nids/config.py` y
  `notebooks/cluster_review_oe1.md`.
- Parquet ya escalados (`MinMaxScaler` ajustado solo sobre train benigno,
  sin fuga): `data/processed/{train,val,test}_benign.parquet` y
  `data/processed/test_attacks.parquet`.
- `requirements.txt` — se agregó `umap-learn>=0.5` para la Fase 6 (única
  dependencia nueva de todo OE2; el resto ya estaba: pandas, numpy,
  scikit-learn, torch, matplotlib). **Instalar en la VM**:
  ```
  pip install -r requirements.txt
  ```
  (o al menos `pip install umap-learn` si el resto del entorno ya existe
  — trae `numba`, `llvmlite`, `pynndescent`, `tqdm` como dependencias).

## 3. Fase 1 — Diagnóstico inicial (solo lectura)

Detalle completo: `notebooks/oe2_diagnostico_inicial.md`.

- **Taxonomía real de labels** en `test_attacks.parquet` (n=442,952, 23
  valores distintos, columna `Label`) — incluye subtipos DoS separados y
  variantes `- Attempted` como clases propias. Tabla completa en el doc.
- Confirmación de que los `.py` de modelo/training/evaluation/viz ya
  estaban trackeados; `.pt` y `.parquet` no (antes de `7e3b8c7`).
- Tamaños: `data/processed/` 239 MB, `outputs/checkpoints/` 204 KB.

## 4. Fase 2 — Codificación a mu (espacio latente)

Script: `src/vae_nids/evaluation/encode_latent_oe2.py`
(**⚠️ no commiteado todavía** — ver sección 9).

Codifica cada muestra a `mu = E_phi[z|x]` (la media de q_phi(z|x)),
**nunca** z muestreado con reparameterization trick. Modelo en modo eval,
con un assert que verifica `seed==42` y `epoch==52` del checkpoint antes
de codificar nada.

Genera 19 archivos `.npy` (todos shape `(n, 8)`) en
`outputs/latent_vectors/` + `manifest.json`:

| Archivo | n | Labels |
|---|---:|---|
| `latent_benign_test.npy` | 248,561 | BENIGN |
| `latent_attack_PortScan.npy` | 159,023 | PortScan |
| `latent_attack_DoS_Hulk.npy` | 158,469 | DoS Hulk |
| `latent_attack_DDoS.npy` | 95,123 | DDoS |
| `latent_attack_DoS_GoldenEye.npy` | 7,567 | DoS GoldenEye |
| `latent_attack_DoS_slowloris.npy` | 4,001 | DoS slowloris |
| `latent_attack_FTP_Patator.npy` | 3,973 | FTP-Patator |
| `latent_attack_SSH_Patator.npy` | 2,980 | SSH-Patator |
| `latent_attack_DoS_Slowhttptest.npy` | 1,742 | DoS Slowhttptest |
| `latent_attack_Bot.npy` | 738 | Bot |
| `latent_attack_Web_Attack.npy` | 190 | Brute Force + XSS + Sql Injection (fusión) |
| `latent_attack_Infiltration.npy` | 32 | Infiltration |
| `latent_attack_DoS_Hulk_attempted.npy` | 579 | DoS Hulk - Attempted |
| `latent_attack_DoS_GoldenEye_attempted.npy` | 80 | DoS GoldenEye - Attempted |
| `latent_attack_DoS_slowloris_attempted.npy` | 1,706 | DoS slowloris - Attempted |
| `latent_attack_DoS_Slowhttptest_attempted.npy` | 3,367 | DoS Slowhttptest - Attempted |
| `latent_attack_Bot_attempted.npy` | 1,470 | Bot - Attempted |
| `latent_attack_WebAttack_BruteForce.npy` | 151 | Web Attack - Brute Force (solo) |
| `latent_attack_WebAttack_BruteForce_attempted.npy` | 1,214 | Web Attack - Brute Force - Attempted |

Excluidos deliberadamente: `Heartbleed` (n=11, insuficiente),
`FTP-Patator - Attempted` (11), `SSH-Patator - Attempted` (8),
`Infiltration - Attempted` (16), `Web Attack - XSS - Attempted` (652).

**⚠️ `outputs/latent_vectors/` no está en git** (no ignorado tampoco, ver
sección 9) — son 19 `.npy` pequeños (n×8 floats), fáciles de regenerar
corriendo `python -m vae_nids.evaluation.encode_latent_oe2`.

## 5. Fase 3 — Métricas geométricas (Mahalanobis + silhouette)

Script: `src/vae_nids/evaluation/geometric_metrics_oe2.py`. Commit
`8fbf0e0`.

**Dimensiones activas recalculadas sobre `latent_benign_test.npy`**
(mismo criterio Burda et al. 2016, umbral 0.01, misma función que OE1
`evaluation.metrics_oe1.active_units`; recalculado, no asumido del cierre
de OE1):

| dim | Var_x[mu_j] | Estado |
|---:|---:|---|
| 0 | 0.572597 | activa |
| 1 | 0.255990 | activa |
| **2** | 0.000009 | **colapsada** |
| 3 | 0.953659 | activa |
| 4 | 0.965305 | activa |
| 5 | 0.255043 | activa |
| 6 | 0.773592 | activa |
| **7** | 0.000028 | **colapsada** |

**Dims activas usadas en Fase 3, 5 y 6: `[0, 1, 3, 4, 5, 6]` (6/8).**

Centroide + covarianza del benigno (n=248,561 completo, solo dims
activas) → Mahalanobis por muestra de cada uno de los 18 grupos; para
silhouette (familia vs. benigno), submuestreo a 5,000 por grupo si supera
ese tamaño (seed=42).

Tabla completa (18 filas) en `outputs/metrics/oe2_geometric_metrics.csv`
y `notebooks/oe2_fase3_geometric_metrics.md`. Columnas: `group, n_total,
n_used_silhouette, mahalanobis_mean, mahalanobis_median, mahalanobis_p25,
mahalanobis_p75, silhouette_score, active_dims_used`.

Nota clave: `Infiltration` (n=32) tiene media Mahalanobis=35.6 pero
mediana=4.35 — un outlier domina el promedio en un grupo chico (por eso
Fase 5 usa mediana como métrica primaria).

## 6. Fase 4 — Detectabilidad por familia (AUC-ROC / TPR)

Script: `src/vae_nids/evaluation/detectability_oe2.py`. Commit `9a33157`.

**Score de anomalía**: NLL de reconstrucción Gaussiana heteroscedástica
por muestra (ecuación 5 del documento de fundamentos), **sin** el término
β·KL, **sin** MSE plano. Score más alto = más anómalo. Usa las **8
dimensiones completas** de `mu` (no las 6 activas — el decoder espera las
8).

**tau = -156.9726** (percentil 95 de los scores sobre `val_benign.parquet`
completo, n=248,560 — split fresco, nunca tocado hasta esta fase, para no
contaminar el umbral con datos de evaluación).

AUC-ROC con IC 95% por bootstrap **estratificado** (positivos y negativos
remuestreados por separado con tamaño fijo, 1000 iteraciones, seed=42 —
evita perder una clase entera en grupos chicos como Infiltration).

Tabla completa (18 filas) en `outputs/metrics/oe2_detectability.csv` y
`notebooks/oe2_fase4_detectability.md`. Columnas: `group, n_total,
auc_roc, auc_roc_ci_low, auc_roc_ci_high, tpr_at_tau, tau_value`. Curvas
ROC completas por grupo en `outputs/metrics/roc_curves/oe2_roc_{group}.npz`
(fpr, tpr, thresholds).

Hallazgo de forma: el ancho de CI más grande no es el de menor n
(`Infiltration`, n=32, ancho=0.0074) sino el de AUC más ambiguo
(`Web_Attack`, n=190, ancho=0.0234) — la varianza del bootstrap depende
tanto de n como de qué tan lejos esté el AUC de 0.5/1.0.

## 7. Fase 5 — Correlación geometría-detectabilidad (RESULTADO CENTRAL)

Script: `src/vae_nids/evaluation/correlation_oe2.py`. Commit `4a0e159`.

Une Fase 3 + Fase 4 por `group` (18 filas, 1 a 1 verificado). Spearman
sobre las **11 familias principales** (excluye los 7 grupos
`_attempted`/`WebAttack_BruteForce` standalone):

| Par | rho | p-valor | Significativo (p<0.05) |
|---|---:|---:|---|
| mahalanobis_median vs auc_roc | **0.3818** | **0.2466** | **No** |
| mahalanobis_median vs tpr_at_tau | 0.3364 | 0.3118 | No |
| silhouette_score vs auc_roc | 0.1727 | 0.6115 | No |
| silhouette_score vs tpr_at_tau | -0.0455 | 0.8944 | No |
| mahalanobis_mean vs auc_roc | 0.5000 | 0.1173 | No |
| mahalanobis_mean vs tpr_at_tau | 0.4727 | 0.1420 | No |

**Ninguna correlación es significativa con n=11.** Con `mahalanobis_mean`
sube el rho pero sigue sin ser significativo (más influenciado por el
outlier de Infiltration).

Scatter plot (Mahalanobis mediana vs. AUC-ROC, 11 familias etiquetadas,
sin línea de tendencia forzada porque rho no es significativo):
`outputs/figures/oe2_geometry_vs_detectability.png`.

**Comparación secundaria completado/attempted** (6 pares: delta =
completo - attempted, en `mahalanobis_median` y `auc_roc`):

| Par | delta_mahalanobis_median | delta_auc_roc | Mismo signo | Consistente con hipótesis |
|---|---:|---:|---|---|
| DoS_Hulk vs. attempted | -2.5082 | -0.0182 | Sí | **No** (ambos negativos) |
| DoS_GoldenEye vs. attempted | +1.2001 | +0.1567 | Sí | Sí |
| DoS_slowloris vs. attempted | +2.5599 | -0.0317 | No | No |
| DoS_Slowhttptest vs. attempted | -1.7907 | +0.0050 | No | No |
| Bot vs. attempted | -1.0694 | +0.0320 | No | No |
| WebAttack_BruteForce vs. attempted | +1.4207 | +0.2152 | Sí | Sí |

**Solo 2/6 pares confirman la hipótesis** (completo más lejos Y más
detectable que attempted). La comparación secundaria **no respalda** una
relación sistemática en la dirección esperada.

Resultado completo (rhos, p-valores, los 6 deltas): `outputs/metrics/oe2_correlation.json`.
Detalle narrado: `notebooks/oe2_fase5_correlation.md`.

### Conclusión cuantitativa de OE2 hasta ahora

Con n=11 familias, **no hay evidencia estadística de que la distancia
geométrica en el espacio latente (Mahalanobis o silhouette) prediga la
detectabilidad de una familia de ataque** (todas las correlaciones no
significativas, p>0.11 en el mejor caso). La comparación
completado/attempted tampoco confirma la hipótesis de forma sistemática
(2/6). Este es el resultado central a reportar — no es un resultado nulo
por error metodológico (dims activas verificadas, score correcto, splits
sin fuga, bootstrap estratificado), es el hallazgo en sí.

## 8. Fase 6 — Proyección UMAP (PENDIENTE, correr en la VM)

Script: `src/vae_nids/evaluation/umap_projection_oe2.py` (**no
commiteado**). Es **solo apoyo visual** — no reemplaza ni valida el
resultado cuantitativo de la Fase 5.

**Por qué está pendiente**: no es una cuestión de si el resultado importa
computacionalmente sino de tamaño — el fallback fue t-SNE de sklearn (no
había `umap-learn` instalado), pero el conjunto combinado (benigno
submuestreado a 3,000 + las 11 familias completas, incluyendo
PortScan≈159k y DoS_Hulk≈158k sin submuestrear) da **~437,000 puntos**, y
t-SNE de sklearn no escala bien más allá de ~50-100k (puede tardar de 30
min a varias horas). Se decidió instalar `umap-learn` (que sí escala bien
a este tamaño) en vez de t-SNE o forzar el subsampleo de los ataques
grandes. Aun con UMAP, correrlo en la laptop del usuario resultó
demasiado lento — de ahí la VM.

**Parámetros exactos ya fijados en el script** (no cambiar sin motivo):

- Dims activas: `[0, 1, 3, 4, 5, 6]` (mismas 6 de la Fase 3 — proyección
  visual consistente con la geometría ya reportada).
- Benigno submuestreado a 3,000 filas, `seed=42` (si no, la nube benigna
  tapa visualmente a familias chicas como Infiltration n=32).
- Las 11 familias principales **completas**, sin submuestrear (mismo
  set que la Fase 3/5, excluye los 7 `_attempted`/`WebAttack_BruteForce`).
- Una sola proyección UMAP ajustada sobre benigno+11 familias
  concatenados — **nunca** una proyección por familia separada (quedarían
  en sistemas de coordenadas distintos, no comparables).
- `UMAP(random_state=42, n_neighbors=15, min_dist=0.1, n_components=2, verbose=True)`.
- Plot: un color por grupo (12 = benigno + 11), leyenda con nombres,
  centroide de cada grupo marcado con `X` más grande y borde negro.
- Salida: `outputs/figures/oe2_latent_umap_projection.png`.

**Cómo correrlo en la VM** (desde la raíz del repo, con el venv activado
y `pip install -r requirements.txt` ya corrido):

```
python -u -m vae_nids.evaluation.umap_projection_oe2
```

El `-u` fuerza salida sin buffer para ver el progreso en vivo (el script
ya tiene `flush=True` en todos sus `print`, más timestamps de import y de
ajuste). `verbose=True` en UMAP imprime su propio progreso interno
(vecinos, épocas).

**Después de correrlo**: falta el commit —
`"OE2 fase 6: proyección UMAP del espacio latente por familia"` — sobre
el script + la figura resultante. Reportar también: qué método se usó
(debería ser UMAP, ya instalado — solo caer a t-SNE si por algún motivo
UMAP falla en la VM, y decirlo explícitamente), y si la separación visual
contradice a simple vista los números de Mahalanobis de la Fase 3 (ej. si
`SSH_Patator`, que tiene la mediana Mahalanobis más baja de las 11
familias, 1.38, se ve visualmente lejísimos del benigno en el plot, eso
amerita investigar por qué).

## 9. Estado de git — qué falta commitear

Commits ya hechos (en orden):

| Commit | Contenido |
|---|---|
| `7e3b8c7` | Checkpoints de OE1 trackeados (prerequisito de OE2) |
| `8fbf0e0` | Fase 3 (script + CSV geométrico) |
| `9a33157` | Fase 4 (script + CSV detectabilidad + 18 `.npz` de curvas ROC) |
| `4a0e159` | Fase 5 (script + JSON correlación + figura scatter) |

**Sin commitear todavía** (working tree actual):

- `requirements.txt` (modificado: se agregó `umap-learn>=0.5`)
- `src/vae_nids/evaluation/encode_latent_oe2.py` (script de la Fase 2 —
  nunca se pidió commitearlo explícitamente, pero sus outputs sí se usan
  en todas las fases siguientes)
- `src/vae_nids/evaluation/umap_projection_oe2.py` (script de la Fase 6,
  agregado con timestamps/flush para la corrida en la VM)
- `outputs/latent_vectors/` (los 19 `.npy` + `manifest.json` de la Fase 2
  — no está en `.gitignore`, así que en algún momento hay que decidir si
  se trackea o se ignora; son archivos chicos)
- `notebooks/oe2_diagnostico_inicial.md`, `oe2_fase2_encoding.md`,
  `oe2_fase3_geometric_metrics.md`, `oe2_fase4_detectability.md`,
  `oe2_fase5_correlation.md` (los reportes narrados de cada fase, en
  formato `.md` para poder copiarlos fuera del chat de VS Code)

**No se hizo ningún `git push`** — el usuario pushea manualmente, nunca
Claude.

## 10. Checklist para retomar en la VM

1. Clonar/traer el repo con el working tree actual (o al menos: los
   commits hasta `4a0e159`, más los archivos sin commitear listados
   arriba — especialmente `outputs/latent_vectors/*.npy` y
   `encode_latent_oe2.py`, que son insumo directo de la Fase 6).
2. `pip install -r requirements.txt` (trae `umap-learn` + sus
   dependencias: numba, llvmlite, pynndescent, tqdm).
3. Confirmar que `outputs/checkpoints/vae_k8_beta1_input51_best.pt`
   existe (seed=42, epoch=52 — el script de la Fase 6 no lo valida
   directamente porque ya parte de los `.npy`, pero si hay que
   regenerar la Fase 2 sí hace falta).
4. Si `outputs/latent_vectors/` no vino en el traspaso: correr
   `python -m vae_nids.evaluation.encode_latent_oe2` para regenerarlo
   (tarda segundos, no minutos).
5. Correr `python -u -m vae_nids.evaluation.umap_projection_oe2`.
6. Revisar la figura, commitear Fase 6, y decidir qué hacer con el resto
   de lo pendiente de la sección 9 (script de Fase 2, `.npy`, los `.md`
   de reportes).
