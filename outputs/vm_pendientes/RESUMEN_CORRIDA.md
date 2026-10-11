# Corrida de pendientes en la VM: OE1 saneado y OE2 recalculado

Generado por `scripts/vm_resumen_corrida.py` el 2026-10-11 16:24 UTC. Log cronológico: `RUN_LOG.md`. Todas las cifras salen de los CSV/JSON de `outputs/`; ninguna se escribe a mano.

## 1. Estado de las etapas

| id | etapa | estado | inicio | min | log |
|---|---|---|---|---|---|
| E01 | Tests (OE1 saneado y estadística del OE2) | OK | 2026-10-11T14:38:21Z | 0.2 | outputs/vm_pendientes/logs/E01.log |
| E10 | OE1: datos, preproceso, 8 semillas + referencia 74, diagnósticos | OK | 2026-10-11T14:38:35Z | 0.9 | outputs/vm_pendientes/logs/E10.log |
| E11 | OE1: verificación de checkpoints y diagnósticos | OK | 2026-10-11T14:39:29Z | 0.0 | outputs/vm_pendientes/logs/E11.log |
| E12 | OE1: figuras del informe (selección y preproceso) | OK | 2026-10-11T14:39:30Z | 0.1 | outputs/vm_pendientes/logs/E12.log |
| E13 | OE1: tablas LaTeX y cifras del informe | OK | 2026-10-11T14:39:37Z | 0.0 | outputs/vm_pendientes/logs/E13.log |
| E20 | OE2 robusto: codificación (mu, log sigma^2) por semilla | OK | 2026-10-11T14:39:37Z | 0.0 | outputs/vm_pendientes/logs/E20.log |
| E21 | OE2 robusto: geometría distribucional | OK | 2026-10-11T14:39:40Z | 0.0 | outputs/vm_pendientes/logs/E21.log |
| E22_s42 | OE2 robusto: detectabilidad, semilla 42 | OK | 2026-10-11T14:39:46Z | 0.1 | outputs/vm_pendientes/logs/E22_s42.log |
| E22_s43 | OE2 robusto: detectabilidad, semilla 43 | OK | 2026-10-11T14:39:50Z | 10.6 | outputs/vm_pendientes/logs/E22_s43.log |
| E22_s44 | OE2 robusto: detectabilidad, semilla 44 | OK | 2026-10-11T14:50:28Z | 0.1 | outputs/vm_pendientes/logs/E22_s44.log |
| E22_s45 | OE2 robusto: detectabilidad, semilla 45 | OK | 2026-10-11T14:50:32Z | 0.0 | outputs/vm_pendientes/logs/E22_s45.log |
| E22_s46 | OE2 robusto: detectabilidad, semilla 46 | OK | 2026-10-11T14:50:35Z | 0.1 | outputs/vm_pendientes/logs/E22_s46.log |
| E22_s47 | OE2 robusto: detectabilidad, semilla 47 | OK | 2026-10-11T14:50:39Z | 10.6 | outputs/vm_pendientes/logs/E22_s47.log |
| E22_s48 | OE2 robusto: detectabilidad, semilla 48 | OK | 2026-10-11T15:01:17Z | 10.8 | outputs/vm_pendientes/logs/E22_s48.log |
| E22_s49 | OE2 robusto: detectabilidad, semilla 49 | OK | 2026-10-11T15:12:08Z | 0.0 | outputs/vm_pendientes/logs/E22_s49.log |
| E22 | OE2 robusto: detectabilidad, tablas de todas las semillas | OK | 2026-10-11T15:12:11Z | 0.1 | outputs/vm_pendientes/logs/E22.log |
| E23 | OE2 robusto: correlación geometría-detectabilidad | OK | 2026-10-11T15:12:15Z | 8.2 | outputs/vm_pendientes/logs/E23.log |
| E24 | OE2 robusto: figuras | OK | 2026-10-11T15:20:28Z | 0.2 | outputs/vm_pendientes/logs/E24.log |
| E25 | OE2 robusto: UMAP (apoyo visual) | OK | 2026-10-11T15:20:38Z | 0.0 | outputs/vm_pendientes/logs/E25.log |
| E30 | Descriptores: detectabilidad y referencia benigna | OK | 2026-10-11T15:20:41Z | 3.8 | outputs/vm_pendientes/logs/E30.log |
| E31 | Descriptores: 17 descriptores por familia y semilla | OK | 2026-10-11T15:24:31Z | 57.1 | outputs/vm_pendientes/logs/E31.log |
| E32 | Descriptores: cribado y validación (attempted) | OK | 2026-10-11T16:21:35Z | 0.8 | outputs/vm_pendientes/logs/E32.log |
| E33 | Descriptores: duplicados, métrica del decodificador, UMAP sin duplicados | OK | 2026-10-11T16:22:24Z | 1.7 | outputs/vm_pendientes/logs/E33.log |
| E34 | Descriptores: figuras | OK | 2026-10-11T16:24:07Z | 0.2 | outputs/vm_pendientes/logs/E34.log |
| E35 | Descriptores: tablas LaTeX y cifras | OK | 2026-10-11T16:24:20Z | 0.0 | outputs/vm_pendientes/logs/E35.log |
| E36 | Prueba confirmatoria (primario + 6 secundarios, Holm) | OK | 2026-10-11T16:24:23Z | 0.2 | outputs/vm_pendientes/logs/E36.log |
| E99 | Resumen de la corrida | OK | 2026-10-11T03:37:45Z | 0.0 | outputs/vm_pendientes/logs/E99.log |

Fallidas: **0**. Saltadas por dependencia: **0**.
Si alguna falló, la cola de su log está en `RUN_LOG.md` y el log completo en `outputs/vm_pendientes/logs/`.

## 2. Entorno

- `fecha_utc`: 2026-10-11T16:24:37+00:00
- `host`: tesis-juan
- `sistema`: Linux-5.15.0-194-generic-x86_64-with-glibc2.35
- `python`: 3.10.12
- `cpus`: 8
- `git_commit`: c818eead44d7b628d603588314db7fde60af0f80
- `git_rama`: main
- `git_arbol_limpio_al_resumir`: False
- `ram_gb`: 23.5
- `torch`: 2.13.0+cpu
- `numpy`: 2.2.6
- `pandas`: 2.3.3
- `scikit-learn`: 1.7.2
- `scipy`: 1.15.3
- `umap-learn`: 0.5.12
- `statsmodels`: 0.15.0

## 3. OE1 saneado: reentrenamiento y diagnósticos

Modelos del OE1 saneado (66 variables) y referencia de 74. Diagnósticos solo con `val_benign`.

| corrida | seed | n_columnas | epoca_parada | mejor_epoca | mejor_val_total | activas_burda | activas_kl | kl_total | tau_p95 | nll_media_por_columna | columnas_en_piso | frac_logvar_dec_en_piso |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oe1_info_nula | 42 | 66 | 87 | 77 | -242.748 | 7 | 7 | 9.784 | -219.926 | -3.857 | 66 | 0.910 |
| oe1_info_nula | 43 | 66 | 57 | 47 | -241.935 | 7 | 7 | 7.883 | -196.207 | -3.815 | 65 | 0.888 |
| oe1_info_nula | 44 | 66 | 74 | 64 | -245.045 | 7 | 7 | 8.532 | -223.585 | -3.870 | 66 | 0.915 |
| oe1_info_nula | 45 | 66 | 85 | 75 | -245.152 | 7 | 7 | 9.008 | -222.822 | -3.880 | 66 | 0.921 |
| oe1_info_nula | 46 | 66 | 48 | 38 | -236.982 | 7 | 7 | 8.288 | -193.439 | -3.751 | 64 | 0.875 |
| oe1_info_nula | 47 | 66 | 109 | 99 | -246.341 | 7 | 7 | 7.467 | -221.164 | -3.871 | 66 | 0.915 |
| oe1_info_nula | 48 | 66 | 122 | 112 | -243.526 | 7 | 7 | 8.793 | -202.471 | -3.850 | 66 | 0.910 |
| oe1_info_nula | 49 | 66 | 95 | 85 | -244.236 | 6 | 6 | 8.482 | -219.092 | -3.857 | 66 | 0.902 |
| referencia_74 | 42 | 74 | 91 | 81 | -270.471 | 7 | 7 | 9.652 | -221.478 | -3.814 | 73 | 0.888 |

- Unidades activas (Burda, Var(μ) > 0,01), 66 variables: media **6.88** (sd 0.35; 7, 7, 7, 7, 7, 7, 7, 6).
- KL total media: 8.53 nats. Fracción de log σ² del decodificador en el piso −10: 0.905.
- Referencia 74 (semilla 42): 7 unidades activas, KL 9.65.

**Antes y después** (medias por semilla):

| modelo | semillas | activas | kl_total | mejor_epoca | nll_por_columna | frac_piso |
|---|---|---|---|---|---|---|
| OE1 actual, 51 variables | 5 | 5.400 | 6.897 | 76.800 | -3.836 | 0.899 |
| OE1 saneado, 66 variables | 8 | 6.875 | 8.530 | 74.625 | -3.844 | 0.905 |

- Filas eliminadas por inf/NaN: 793. Filtro D4 por partición: train_benign 24, val_benign 7, test_benign 0, test_attacks 0.
- Particiones frente al pipeline original: train_benign: True, val_benign: True, test_benign: True, test_attacks: True. (Copia en `outputs/oe1_saneado/datos_info.json`.)

## 4. OE2 robusto sobre los modelos saneados

Semillas: 42, 43, 44, 45, 46, 47, 48, 49. Dimensiones activas por semilla (benigno de test): 42→7, 43→7, 44→7, 45→7, 46→7, 47→7, 48→7, 49→6.

**AUC-ROC (nll_mc, L = 20) por grupo**, media entre semillas:

| group | mean | std | min | max |
|---|---|---|---|---|
| Bot_attempted | 0.6025 | 0.0189 | 0.5679 | 0.6259 |
| Bot | 0.6095 | 0.0127 | 0.5972 | 0.6298 |
| PortScan | 0.6717 | 0.0330 | 0.5996 | 0.7035 |
| WebAttack_BruteForce_attempted | 0.6846 | 0.0279 | 0.6582 | 0.7427 |
| DoS_GoldenEye_attempted | 0.7063 | 0.0604 | 0.5861 | 0.7511 |
| DoS_Hulk_attempted | 0.9085 | 0.0166 | 0.8737 | 0.9214 |
| FTP_Patator | 0.9130 | 0.0343 | 0.8540 | 0.9516 |
| DoS_Hulk | 0.9248 | 0.0306 | 0.8538 | 0.9503 |
| Web_Attack | 0.9289 | 0.0126 | 0.9098 | 0.9434 |
| DoS_slowloris | 0.9391 | 0.0084 | 0.9259 | 0.9554 |
| WebAttack_BruteForce | 0.9467 | 0.0143 | 0.9271 | 0.9653 |
| SSH_Patator | 0.9518 | 0.0255 | 0.9126 | 0.9851 |
| DoS_slowloris_attempted | 0.9526 | 0.0292 | 0.8887 | 0.9781 |
| DoS_GoldenEye | 0.9654 | 0.0131 | 0.9414 | 0.9810 |
| DoS_Slowhttptest | 0.9802 | 0.0115 | 0.9662 | 0.9928 |
| DoS_Slowhttptest_attempted | 0.9822 | 0.0062 | 0.9726 | 0.9905 |
| DDoS | 0.9838 | 0.0141 | 0.9517 | 0.9957 |
| Infiltration | 0.9874 | 0.0033 | 0.9835 | 0.9929 |

**Frente al modelo de 51 variables** (OE2 robusto, 10 semillas):

| group | auc_51 | auc_66 | dif |
|---|---|---|---|
| Bot | 0.6239 | 0.6095 | -0.0144 |
| Bot_attempted | 0.6134 | 0.6025 | -0.0109 |
| DDoS | 0.9756 | 0.9838 | 0.0082 |
| DoS_GoldenEye | 0.9603 | 0.9654 | 0.0051 |
| DoS_GoldenEye_attempted | 0.8032 | 0.7063 | -0.0969 |
| DoS_Hulk | 0.9161 | 0.9248 | 0.0087 |
| DoS_Hulk_attempted | 0.9384 | 0.9085 | -0.0299 |
| DoS_Slowhttptest | 0.9942 | 0.9802 | -0.0140 |
| DoS_Slowhttptest_attempted | 0.9876 | 0.9822 | -0.0054 |
| DoS_slowloris | 0.9453 | 0.9391 | -0.0062 |
| DoS_slowloris_attempted | 0.9488 | 0.9526 | 0.0038 |
| FTP_Patator | 0.9267 | 0.9130 | -0.0136 |
| Infiltration | 0.9920 | 0.9874 | -0.0046 |
| PortScan | 0.6667 | 0.6717 | 0.0051 |
| SSH_Patator | 0.9578 | 0.9518 | -0.0059 |
| WebAttack_BruteForce | 0.9481 | 0.9467 | -0.0014 |
| WebAttack_BruteForce_attempted | 0.7174 | 0.6846 | -0.0328 |
| Web_Attack | 0.9318 | 0.9289 | -0.0029 |

- τ (p95 de val_benign): FPR realizada en test_benign media 0.0506.
- Hipótesis principal (mahalanobis_median vs. auc_roc(nll_mc)): ρ combinado **0.759** IC95 [0.619, 0.853], 8 semillas.
- Modelo mixto a nivel de conexión: convergió=True, efecto fijo 0.104 [-0.099, 0.306], p=0.3155, R² marginal 0.002.
- W de Kendall entre semillas: auc_roc_nll_mc 0.887, mahalanobis_median 0.798, mmd2_rbf 0.854, sliced_wasserstein 0.861, energy_distance 0.805, bhattacharyya_total 0.917.
- Conteos por grupo frente al OE2 original: idénticos en todas las semillas (`outputs/oe2_robust_saneado/verificacion_conteos.json`).

## 5. OE2: valoración de descriptores sobre los modelos saneados

Cribado (fase 3) en los modelos saneados; mismo diseño que con 51 variables:

| id | descriptor | nivel | rho | ci_lo | ci_hi | p_perm | p_holm | kendall_w | rho_sin_portscan_bot | rho_attempted | tier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.1 | Mahalanobis (mediana) | P | 0.772 | 0.443 | 0.970 | 0.000 | — | 0.808 | 0.598 | 0.856 | B: asociación prometedora |
| 5.1 | kNN al benigno (z blanqueado) | S | 0.741 | 0.325 | 0.928 | 0.001 | 0.010 | 0.769 | 0.526 | 0.674 | A: asociación robusta |
| 2.4 | Fisher-Rao al k-ésimo posterior benigno | S | 0.775 | 0.426 | 0.910 | 0.000 | 0.001 | 0.788 | 0.599 | 0.661 | A: asociación robusta |
| 7.1 | Tasa KL(q\|\|p) por flujo | S | 0.830 | 0.515 | 0.973 | 0.000 | 0.004 | 0.904 | 0.704 | 0.952 | A: asociación robusta |
| 3.2 | MMD² analítico entre posteriores agregados | S | 0.435 | -0.272 | 0.887 | 0.115 | 0.460 | 0.771 | 0.027 | 0.849 | C: estable, sin asociación clara |
| 6.2 | kNN en el espacio decodificado | S | 0.680 | 0.249 | 0.880 | 0.003 | 0.019 | 0.681 | 0.416 | 0.638 | A: asociación robusta |
| 6.1 | Log-magnificación del decodificador | S | 0.592 | 0.110 | 0.888 | 0.013 | 0.077 | 0.683 | 0.265 | 0.938 | B: asociación prometedora |
| 4.2 | Distancia afín-invariante entre covarianzas | S | -0.144 | -0.671 | 0.396 | 0.634 | 1.000 | 0.892 | -0.265 | -0.456 | C: estable, sin asociación clara |
| 8.1 | LID relativa al benigno | S | 0.444 | -0.055 | 0.823 | 0.059 | 0.295 | 0.561 | 0.189 | 0.366 | C: estable, sin asociación clara |
| 10.1a | Silhouette del grupo (z blanqueado) | S | -0.411 | -0.831 | 0.099 | 0.177 | 0.530 | 0.931 | -0.450 | -0.754 | C: estable, sin asociación clara |
| 10.3 | Sonda lineal sobre (μ, log σ²) | S | 0.035 | -0.611 | 0.643 | 0.909 | 1.000 | 0.915 | -0.225 | -0.158 | C: estable, sin asociación clara |

**ρ de cada descriptor, 51 frente a 66 variables** (todos los del catálogo):

| columna | id | rho_51 | rho_66 | dif |
|---|---|---|---|---|
| maha__median | 1.1 | 0.604 | 0.772 | 0.168 |
| knn__auc | 5.1 | 0.694 | 0.741 | 0.047 |
| fr__auc | 2.4 | 0.724 | 0.775 | 0.052 |
| rate__sep | 7.1 | 0.681 | 0.830 | 0.149 |
| mmd_agg__value | 3.2 | 0.318 | 0.435 | 0.117 |
| dec_knn__auc | 6.2 | 0.691 | 0.680 | -0.011 |
| logmag__sep | 6.1 | 0.523 | 0.592 | 0.069 |
| airm__value | 4.2 | 0.006 | -0.144 | -0.150 |
| lid__sep | 8.1 | 0.383 | 0.444 | 0.062 |
| sil_w__value | 10.1a | -0.466 | -0.411 | 0.054 |
| probe__value | 10.3 | -0.014 | 0.035 | 0.050 |
| maha__auc | 1.1b | 0.653 | 0.813 | 0.160 |
| sil_fr__value | 10.1b | -0.629 | -0.627 | 0.002 |
| gmm__auc | 1.2 | 0.659 | 0.737 | 0.078 |
| rmd__auc | 1.3 | 0.579 | 0.713 | 0.134 |
| typ__auc | 1.4 | 0.648 | 0.762 | 0.114 |
| erank__value | 4.1 | 0.280 | 0.141 | -0.139 |
| logvol__sep | 7.2 | 0.499 | 0.559 | 0.060 |
| kp__sep | 11.1 | 0.239 | 0.381 | 0.142 |
| knn__frac_p95 | 5.1m | 0.455 | 0.627 | 0.171 |
| fr__median | 2.4m | 0.629 | 0.724 | 0.095 |
| rate__median | 7.1m | 0.695 | 0.786 | 0.091 |
| logmag__median | 6.1m | 0.519 | 0.627 | 0.108 |
| repo:bhattacharyya_cov_term | R | -0.583 | -0.557 | 0.025 |
| repo:bhattacharyya_mean_term | R | 0.294 | 0.316 | 0.022 |
| repo:bhattacharyya_total | R | -0.341 | -0.283 | 0.058 |
| repo:cliffs_delta | R | 0.663 | 0.798 | 0.136 |
| repo:cov_logdet | R | 0.669 | 0.656 | -0.013 |
| repo:energy_distance | R | 0.466 | 0.673 | 0.207 |
| repo:log_density_benign_median | R | -0.681 | -0.741 | -0.059 |
| repo:mahalanobis_median_robust_mcd | R | 0.652 | 0.655 | 0.003 |
| repo:mahalanobis_p95 | R | 0.414 | 0.615 | 0.201 |
| repo:mean_posterior_variance_sum | R | -0.179 | -0.135 | 0.044 |
| repo:mmd2_rbf | R | 0.174 | 0.362 | 0.188 |
| repo:neighborhood_purity_median | R | 0.391 | 0.413 | 0.022 |
| repo:posterior_mahalanobis_median | R | 0.549 | 0.759 | 0.210 |
| repo:silhouette_mean | R | 0.230 | 0.464 | 0.234 |
| repo:sliced_wasserstein | R | 0.578 | 0.757 | 0.179 |

- Fase 1 frente a la fase 4 robusta (semilla 42): diferencia máxima de AUC 1.11e-16.
- Rango efectivo de la métrica del decodificador G (mediana por semilla): 1.10 a 1.97. Log σ² del decodificador en el piso: 87 a 92 %.

## 6. OE2: prueba confirmatoria (hipótesis registradas el 10-10-2026)


Hipótesis registradas el 2026-10-10, antes de ver los resultados de este perfil (`fase7_confirmatoria.py`). 8 semillas (42, 43, 44, 45, 46, 47, 48, 49), 11 familias, 10,000 permutaciones, 2,000 remuestreos jerárquicos.

### Hipótesis

| id | descriptor | rol | signo_esperado | rho_modelo_51 | rho | ic_lo | ic_hi | p_bilateral | p_holm | confirmada | rho_sin_portscan_bot | rho_attempted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.1 | Mahalanobis (mediana) | P | 1 | 0.604 | 0.772 | 0.394 | 0.976 | 0.001 | 0.001 | True | 0.598 | 0.856 |
| 2.4 | Fisher-Rao al 10.º posterior benigno | S | 1 | 0.724 | 0.775 | 0.392 | 0.918 | 0.000 | 0.002 | True | 0.599 | 0.661 |
| 6.2 | kNN en el espacio decodificado | S | 1 | 0.691 | 0.680 | 0.133 | 0.900 | 0.002 | 0.006 | True | 0.416 | 0.638 |
| 7.1 | Tasa KL(q\|\|p) por flujo | S | 1 | 0.681 | 0.830 | 0.482 | 0.981 | 0.000 | 0.002 | True | 0.704 | 0.952 |
| 5.1 | kNN al benigno (z blanqueado) | S | 1 | 0.694 | 0.741 | 0.290 | 0.943 | 0.001 | 0.004 | True | 0.526 | 0.674 |
| 6.1 | Log-magnificación del decodificador | S | 1 | 0.523 | 0.592 | 0.050 | 0.908 | 0.015 | 0.031 | True | 0.265 | 0.938 |
| 10.1b | Silhouette del grupo (Fisher-Rao) | S | -1 | -0.629 | -0.627 | -0.897 | -0.178 | 0.026 | 0.031 | True | -0.585 | -0.689 |

- Primario confirmado: **sí**.
- Secundarios confirmados (Holm entre 6): **2.4, 6.2, 7.1, 5.1, 6.1, 10.1b**.

### Nivel de flujo (entre familias y dentro de cada familia)

Entre familias = el rho combinado de la tabla de arriba. Total = flujos de las 11 familias juntos. Dentro = rangos centrados en cada familia (hasta 5000 flujos por familia y semilla).

| descriptor | rho_total | rho_dentro_familia | rho_medio_por_familia | rho_dentro_min | rho_dentro_max |
|---|---|---|---|---|---|
| Mahalanobis (mediana) | 0.779 | 0.716 | 0.601 | 0.540 | 0.822 |
| Fisher-Rao al 10.º posterior benigno | 0.841 | 0.706 | 0.598 | 0.646 | 0.734 |
| kNN en el espacio decodificado | 0.737 | 0.596 | 0.443 | 0.204 | 0.724 |
| Tasa KL(q\|\|p) por flujo | 0.803 | 0.747 | 0.654 | 0.655 | 0.853 |
| kNN al benigno (z blanqueado) | 0.795 | 0.706 | 0.621 | 0.554 | 0.813 |
| Log-magnificación del decodificador | 0.708 | 0.646 | 0.403 | 0.380 | 0.782 |

### AUC por familia: 51 frente a 66 variables

Spearman del ranking de las 11 familias por AUC medio entre los dos modelos: **+0.964**.

| group | auc_51_media | auc_51_sd | auc_66_media | auc_66_sd | diferencia_66_menos_51 |
|---|---|---|---|---|---|
| PortScan | 0.667 | 0.031 | 0.672 | 0.033 | 0.005 |
| DoS_Hulk | 0.916 | 0.015 | 0.925 | 0.031 | 0.009 |
| DDoS | 0.976 | 0.012 | 0.984 | 0.014 | 0.008 |
| DoS_GoldenEye | 0.960 | 0.009 | 0.965 | 0.013 | 0.005 |
| DoS_slowloris | 0.945 | 0.006 | 0.939 | 0.008 | -0.006 |
| FTP_Patator | 0.927 | 0.014 | 0.913 | 0.034 | -0.014 |
| SSH_Patator | 0.958 | 0.033 | 0.952 | 0.026 | -0.006 |
| DoS_Slowhttptest | 0.994 | 0.002 | 0.980 | 0.012 | -0.014 |
| Bot | 0.624 | 0.017 | 0.609 | 0.013 | -0.014 |
| Web_Attack | 0.932 | 0.011 | 0.929 | 0.013 | -0.003 |
| Infiltration | 0.992 | 0.001 | 0.987 | 0.003 | -0.005 |
| DoS_Hulk_attempted | 0.938 | 0.012 | 0.908 | 0.017 | -0.030 |
| DoS_GoldenEye_attempted | 0.803 | 0.044 | 0.706 | 0.060 | -0.097 |
| DoS_slowloris_attempted | 0.949 | 0.025 | 0.953 | 0.029 | 0.004 |
| DoS_Slowhttptest_attempted | 0.988 | 0.003 | 0.982 | 0.006 | -0.005 |
| Bot_attempted | 0.613 | 0.016 | 0.602 | 0.019 | -0.011 |
| WebAttack_BruteForce_attempted | 0.717 | 0.060 | 0.685 | 0.028 | -0.033 |

## 7. Archivos para el commit

Rutas a versionar (`ARCHIVOS_PARA_COMMIT.txt`, las usa `scripts/vm_preparar_commit.sh`):

| ruta | MB |
|---|---|
| outputs/vm_pendientes | 0.23 |
| outputs/oe1_saneado/checkpoints | 0.34 |
| outputs/oe1_saneado/logs | 0.10 |
| outputs/oe1_saneado/tables | 0.04 |
| outputs/oe1_saneado/figures | 1.51 |
| outputs/oe1_saneado/figures_informe | 0.10 |
| outputs/oe1_saneado/preproceso | 0.01 |
| outputs/oe1_saneado/informe_tablas | 0.45 |
| outputs/oe1_saneado/registro_run.log | 0.00 |
| outputs/oe1_saneado/datos_info.json | 0.00 |
| outputs/oe2_robust_saneado/tables | 128.25 |
| outputs/oe2_robust_saneado/metrics | 0.00 |
| outputs/oe2_robust_saneado/figures | 24.43 |
| outputs/oe2_robust_saneado/verificacion_conteos.json | 0.00 |
| outputs/oe2_descriptores_saneado/tables | 0.64 |
| outputs/oe2_descriptores_saneado/figures | 1.21 |
| outputs/oe2_descriptores_saneado/tablas_tex | 0.02 |
| outputs/oe2_descriptores_saneado/RESULTADOS_CONFIRMATORIA.md | 0.00 |

No se versionan (pesados y regenerables): `outputs/oe2_robust_saneado/latent_vectors/`, `outputs/oe2_robust_saneado/tables/*.npz` y `_cache/`, `outputs/oe2_descriptores_saneado/_cache/`, `data/`.
