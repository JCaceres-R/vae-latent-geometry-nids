# Corrida en la VM: reconciliación (OE2), OE3 y OE4

Generado por `scripts/vm_resumen_oe3_oe4.py` el 2026-10-12 00:52 UTC. Log cronológico: `RUN_LOG.md`. Todas las cifras salen de los CSV de `outputs/`.

## 1. Estado de las etapas

| id | etapa | estado | inicio | min | log |
|---|---|---|---|---|---|
| E01 | Tests (OE1 saneado y estadística del OE2) | OK | 2026-10-11T17:52:06Z | 0.2 | outputs/vm_oe3_oe4/logs/E01.log |
| E10 | Reconciliación: modelo mixto (fase 5) frente a nivel de flujo (fase 7) | OK | 2026-10-11T17:52:20Z | 0.3 | outputs/vm_oe3_oe4/logs/E10.log |
| E20 | OE3: calidad generativa del decodificador | OK | 2026-10-11T17:52:36Z | 4.2 | outputs/vm_oe3_oe4/logs/E20.log |
| E3_beta0p5_1 | OE4 beta0p5: entrenamiento (5 semillas) y diagnósticos | OK | 2026-10-11T17:56:47Z | 74.3 | outputs/vm_oe3_oe4/logs/E3_beta0p5_1.log |
| E3_beta0p5_2 | OE4 beta0p5: codificación (mu, log sigma^2) | OK | 2026-10-11T19:11:04Z | 0.1 | outputs/vm_oe3_oe4/logs/E3_beta0p5_2.log |
| E3_beta0p5_3 | OE4 beta0p5: detectabilidad y referencia benigna | OK | 2026-10-11T19:11:12Z | 2.4 | outputs/vm_oe3_oe4/logs/E3_beta0p5_3.log |
| E3_beta0p5_4 | OE4 beta0p5: descriptores por familia y semilla | OK | 2026-10-11T19:13:36Z | 37.7 | outputs/vm_oe3_oe4/logs/E3_beta0p5_4.log |
| E3_beta0p5_5 | OE4 beta0p5: cribado de descriptores | OK | 2026-10-11T19:51:16Z | 0.6 | outputs/vm_oe3_oe4/logs/E3_beta0p5_5.log |
| E3_beta0p5_6 | OE4 beta0p5: prueba confirmatoria | OK | 2026-10-11T19:51:49Z | 0.2 | outputs/vm_oe3_oe4/logs/E3_beta0p5_6.log |
| E3_beta2_1 | OE4 beta2: entrenamiento (5 semillas) y diagnósticos | OK | 2026-10-11T19:52:00Z | 76.2 | outputs/vm_oe3_oe4/logs/E3_beta2_1.log |
| E3_beta2_2 | OE4 beta2: codificación (mu, log sigma^2) | OK | 2026-10-11T21:08:12Z | 0.1 | outputs/vm_oe3_oe4/logs/E3_beta2_2.log |
| E3_beta2_3 | OE4 beta2: detectabilidad y referencia benigna | OK | 2026-10-11T21:08:20Z | 2.4 | outputs/vm_oe3_oe4/logs/E3_beta2_3.log |
| E3_beta2_4 | OE4 beta2: descriptores por familia y semilla | OK | 2026-10-11T21:10:44Z | 34.6 | outputs/vm_oe3_oe4/logs/E3_beta2_4.log |
| E3_beta2_5 | OE4 beta2: cribado de descriptores | OK | 2026-10-11T21:45:17Z | 0.5 | outputs/vm_oe3_oe4/logs/E3_beta2_5.log |
| E3_beta2_6 | OE4 beta2: prueba confirmatoria | OK | 2026-10-11T21:45:47Z | 0.2 | outputs/vm_oe3_oe4/logs/E3_beta2_6.log |
| E3_k16_1 | OE4 k16: entrenamiento (5 semillas) y diagnósticos | OK | 2026-10-11T21:45:59Z | 122.6 | outputs/vm_oe3_oe4/logs/E3_k16_1.log |
| E3_k16_2 | OE4 k16: codificación (mu, log sigma^2) | OK | 2026-10-11T23:48:36Z | 0.1 | outputs/vm_oe3_oe4/logs/E3_k16_2.log |
| E3_k16_3 | OE4 k16: detectabilidad y referencia benigna | OK | 2026-10-11T23:48:43Z | 3.5 | outputs/vm_oe3_oe4/logs/E3_k16_3.log |
| E3_k16_4 | OE4 k16: descriptores por familia y semilla | OK | 2026-10-11T23:52:12Z | 59.8 | outputs/vm_oe3_oe4/logs/E3_k16_4.log |
| E3_k16_5 | OE4 k16: cribado de descriptores | OK | 2026-10-12T00:51:58Z | 0.5 | outputs/vm_oe3_oe4/logs/E3_k16_5.log |
| E3_k16_6 | OE4 k16: prueba confirmatoria | OK | 2026-10-12T00:52:28Z | 0.2 | outputs/vm_oe3_oe4/logs/E3_k16_6.log |
| E40 | OE4: concordancia de rankings y hallazgo por configuración | OK | 2026-10-12T00:52:38Z | 0.1 | outputs/vm_oe3_oe4/logs/E40.log |

Fallidas: **0**. Saltadas por dependencia: **0**.
Si alguna falló, la cola de su log está en `RUN_LOG.md` y el log completo en `outputs/vm_pendientes/logs/`.

## 2. Entorno

- `fecha_utc`: 2026-10-12T00:52:44+00:00
- `host`: tesis-juan
- `sistema`: Linux-5.15.0-194-generic-x86_64-with-glibc2.35
- `python`: 3.10.12
- `cpus`: 8
- `git_commit`: 75a5abd97ce744da02f703ea8a376f1c3d6693ea
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

## 3. OE2: reconciliación del nivel de conexión

Fuente: `outputs/oe2_descriptores_saneado/RESULTADOS_RECONCILIACION.md`.


Misma muestra de conexiones que el modelo mixto de la fase 5 (hasta 5000 por grupo y semilla; 309,248 filas, 8 semillas). x = Mahalanobis latente (phase4, `latent_maha`); y = NLL Monte Carlo (`nll_mc`).

### Correlaciones (media de Fisher entre semillas)

De arriba abajo, del análisis de la fase 5 (Pearson sobre valores, con el benigno) al de la fase 7 (rangos centrados por familia).

| medida | media_fisher | minimo | maximo |
|---|---|---|---|
| pearson_con_benigno | 0.452 | 0.155 | 0.655 |
| pearson_sin_benigno | 0.487 | 0.164 | 0.703 |
| pearson_dentro_familia | 0.576 | 0.234 | 0.796 |
| spearman_con_benigno | 0.827 | 0.486 | 0.906 |
| spearman_sin_benigno | 0.783 | 0.350 | 0.889 |
| spearman_dentro_familia | 0.718 | 0.534 | 0.822 |
| spearman_medio_por_familia | 0.605 | 0.452 | 0.763 |
| spearman_entre_medianas_de_familia | 0.799 | 0.500 | 0.955 |

### Modelos

| modelo | efecto | ic_lo | ic_hi | p | var_pendiente_familia | r2_marginal | n | convergio |
|---|---|---|---|---|---|---|---|---|
| M1 fase 5: mixto, valores estandarizados, con benigno | 0.104 | -0.099 | 0.306 | 0.315 | 0.126 | 0.002 | 309248 | True |
| M2: mixto, valores estandarizados, sin benigno | 0.106 | -0.091 | 0.303 | 0.292 | 0.110 | 0.002 | 269248 | True |
| M3: mixto, puntuaciones normales de rango, sin benigno | 0.718 | 0.581 | 0.854 | 0.000 | 0.053 | 0.501 | 269248 | True |
| M4: MCO con efectos fijos de familia y semilla (dentro de familia), rangos normales | 0.604 | 0.451 | 0.757 | 0.000 | — | 0.709 | 269248 | True |

### Por familia (media entre semillas)

`pearson_menos_spearman` grande = la relación lineal la dominan unos pocos valores extremos. `pendiente_estandarizada` = pendiente de MCO dentro de la familia en la escala que usa la fase 5.

| family | n | spearman | pearson | pendiente_estandarizada | asimetria_maha | asimetria_nll | nll_mediana | nll_p99 | pearson_menos_spearman |
|---|---|---|---|---|---|---|---|---|---|
| Bot | 738.000 | 0.289 | 0.927 | 0.001 | 3.567 | 7.243 | -254.799 | -74.359 | 0.638 |
| SSH_Patator | 2980.000 | 0.346 | 0.751 | 0.000 | -3.145 | -4.350 | -205.207 | -197.626 | 0.404 |
| DoS_Slowhttptest | 1742.000 | 0.351 | 0.212 | 0.002 | 0.549 | 3.193 | -184.294 | 375.546 | -0.139 |
| FTP_Patator | 3973.000 | 0.353 | 0.477 | -0.001 | -6.260 | -1.976 | -218.184 | -212.184 | 0.124 |
| DoS_slowloris | 4001.000 | 0.535 | 0.744 | 0.002 | -0.681 | -0.241 | -146.353 | -47.130 | 0.209 |
| PortScan | 5000.000 | 0.551 | 0.542 | 0.003 | 4.634 | 9.517 | -246.955 | -142.795 | -0.010 |
| Web_Attack | 190.000 | 0.744 | 0.834 | 0.001 | -0.468 | -0.244 | -214.025 | -160.315 | 0.091 |
| Infiltration | 32.000 | 0.754 | 0.917 | 1.495 | 2.138 | 3.169 | -174.783 | 3944313.906 | 0.164 |
| DoS_Hulk | 5000.000 | 0.764 | 0.652 | 0.002 | 1.764 | 12.455 | -219.599 | 59.411 | -0.111 |
| DoS_GoldenEye | 5000.000 | 0.786 | 0.587 | 0.003 | 0.736 | 6.575 | -185.484 | 1294.277 | -0.199 |
| DDoS | 5000.000 | 0.798 | 0.748 | 0.003 | 0.757 | 7.166 | -161.773 | 333.197 | -0.050 |
| BENIGN | 5000.000 | 0.858 | 0.776 | 0.001 | 1.325 | 3.164 | -264.094 | -183.587 | -0.082 |

## 4. OE3: calidad generativa

Fuente: `outputs/oe3/RESULTADOS_OE3.md`.


Modelos del OE1 saneado (8 semillas). Comparación frente a `test_benign` en la escala del modelo. `real_val` = val_benign frente a test_benign: lo que difieren dos muestras reales.

### Resumen (media entre semillas)

| comparacion | ks_mediana | ks_max | frac_variables_ks_menor_0_1 | w1_media | frac_fuera_0_1 | sw2 | c2st_auc | dcr_mediana | dcr_mediana_real_test | dcr_razon | frac_copias_exactas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real_val | 0.0035 | 0.0075 | 1.0000 | 0.0004 | 0.0000 | 0.0071 | 0.4987 | 0.0002 | 0.0002 | 0.9852 | 0.2213 |
| prior_muestra | 0.4555 | 0.5501 | 0.0076 | 0.0324 | 0.3949 | 0.7164 | 1.0000 | 0.2185 | 0.0002 | 1399.8233 | 0.0000 |
| prior_media | 0.5154 | 0.9871 | 0.0189 | 0.0152 | 0.3281 | 0.0602 | 1.0000 | nan | nan | nan | nan |
| posterior_agregado | 0.4493 | 0.5389 | 0.0133 | 0.0142 | 0.4029 | 0.0622 | 1.0000 | 0.0643 | 0.0002 | 414.6713 | 0.0000 |

- `c2st_auc`: 0,5 = el clasificador no distingue; cuanto más cerca de 1, más fácil separar sintético de real.
- `dcr_razon` < 1: los sintéticos quedan más cerca de train que los reales de test (señal de memorización); `frac_copias_exactas` > 0: copias literales.

### Las 10 variables peor reproducidas (z ~ prior, KS medio entre semillas)

| variable | ks_media |
|---|---|
| CWR Flag Count | 0.530 |
| URG Flag Count | 0.525 |
| Fwd URG Flags | 0.522 |
| Fwd Bulk Rate Avg | 0.520 |
| Subflow Bwd Packets | 0.516 |
| Bwd URG Flags | 0.513 |
| Bwd Bytes/Bulk Avg | 0.513 |
| Bwd Packet/Bulk Avg | 0.513 |
| Total Bwd packets | 0.512 |
| ECE Flag Count | 0.511 |

Figuras: `outputs/oe3/figures/` (KS por variable e histogramas de la semilla 42; C2ST).

## 5. OE4: reproducibilidad frente a β y k

Fuente: `outputs/oe4/RESULTADOS_OE4.md`.


Configuraciones con 66 variables y semillas [42, 43, 44, 45, 46]; la base (β = 1, k = 8) se restringe a esas semillas en todo salvo la fila del hallazgo, que usa su fase 7 completa (8 semillas).

### Configuraciones

| configuracion | beta | k | semillas | activas_media | activas_sd | kl_total | mejor_epoca | nll_por_columna | frac_piso_decodificador |
|---|---|---|---|---|---|---|---|---|---|
| base β=1, k=8 | 1.000 | 8 | 5 | 7.000 | 0.000 | 8.699 | 60.200 | -3.835 | 0.902 |
| β=0,5 | 0.500 | 8 | 5 | 8.000 | 0.000 | 13.293 | 94.800 | -3.895 | 0.927 |
| β=2 | 2.000 | 8 | 5 | 5.200 | 0.837 | 4.334 | 96.600 | -3.821 | 0.890 |
| k=16 | 1.000 | 16 | 5 | 6.800 | 0.447 | 7.693 | 130.200 | -3.869 | 0.916 |

### Concordancia del ranking de las 11 familias frente a la base

`w_entre_semillas` = W de Kendall entre las 5 semillas de esa configuración (estabilidad interna). La fila «W entre las N configuraciones» trata cada configuración como un juez.

| metrica | configuracion | spearman_vs_base | kendall_tau_vs_base | w_entre_semillas |
|---|---|---|---|---|
| AUC (nll_mc) | β=0,5 | 0.836 | 0.636 | 0.969 |
| AUC (nll_mc) | β=2 | 0.982 | 0.927 | 0.979 |
| AUC (nll_mc) | k=16 | 0.991 | 0.964 | 0.796 |
| AUC (nll_mc) | W entre las 4 configuraciones | — | — | 0.927 |
| Mahalanobis (mediana) | β=0,5 | 0.927 | 0.818 | 0.948 |
| Mahalanobis (mediana) | β=2 | 0.873 | 0.745 | 0.809 |
| Mahalanobis (mediana) | k=16 | 0.909 | 0.782 | 0.690 |
| Mahalanobis (mediana) | W entre las 4 configuraciones | — | — | 0.919 |
| Tasa KL | β=0,5 | 0.918 | 0.818 | 0.983 |
| Tasa KL | β=2 | 0.955 | 0.855 | 0.957 |
| Tasa KL | k=16 | 0.982 | 0.927 | 0.883 |
| Tasa KL | W entre las 4 configuraciones | — | — | 0.950 |
| Fisher-Rao | β=0,5 | 0.945 | 0.891 | 0.929 |
| Fisher-Rao | β=2 | 0.973 | 0.891 | 0.843 |
| Fisher-Rao | k=16 | 0.982 | 0.927 | 0.738 |
| Fisher-Rao | W entre las 4 configuraciones | — | — | 0.974 |
| kNN blanqueado | β=0,5 | 0.900 | 0.782 | 0.939 |
| kNN blanqueado | β=2 | 0.900 | 0.745 | 0.822 |
| kNN blanqueado | k=16 | 0.900 | 0.745 | 0.638 |
| kNN blanqueado | W entre las 4 configuraciones | — | — | 0.944 |
| kNN decodificado | β=0,5 | 0.627 | 0.418 | 0.908 |
| kNN decodificado | β=2 | 0.909 | 0.782 | 0.839 |
| kNN decodificado | k=16 | 0.845 | 0.709 | 0.662 |
| kNN decodificado | W entre las 4 configuraciones | — | — | 0.786 |
| Log-magnificación | β=0,5 | 0.609 | 0.455 | 0.719 |
| Log-magnificación | β=2 | 0.682 | 0.491 | 0.876 |
| Log-magnificación | k=16 | 0.873 | 0.745 | 0.673 |
| Log-magnificación | W entre las 4 configuraciones | — | — | 0.841 |
| Silhouette Fisher-Rao | β=0,5 | 0.991 | 0.964 | 0.994 |
| Silhouette Fisher-Rao | β=2 | 0.973 | 0.927 | 0.976 |
| Silhouette Fisher-Rao | k=16 | 1.000 | 1.000 | 0.963 |
| Silhouette Fisher-Rao | W entre las 4 configuraciones | — | — | 0.986 |

### AUC por familia

| familia | base β=1, k=8 | β=0,5 | β=2 | k=16 |
|---|---|---|---|---|
| PortScan | 0.6604 | 0.6963 | 0.6561 | 0.6652 |
| DoS_Hulk | 0.9174 | 0.9653 | 0.8942 | 0.9298 |
| DDoS | 0.9800 | 0.9951 | 0.9716 | 0.9828 |
| DoS_GoldenEye | 0.9629 | 0.9830 | 0.9624 | 0.9709 |
| DoS_slowloris | 0.9392 | 0.9417 | 0.9372 | 0.9559 |
| FTP_Patator | 0.9097 | 0.8847 | 0.9166 | 0.9238 |
| SSH_Patator | 0.9453 | 0.9053 | 0.9623 | 0.9452 |
| DoS_Slowhttptest | 0.9806 | 0.9683 | 0.9913 | 0.9853 |
| Bot | 0.6033 | 0.5931 | 0.5975 | 0.6277 |
| Web_Attack | 0.9257 | 0.9419 | 0.9235 | 0.9334 |
| Infiltration | 0.9878 | 0.9817 | 0.9893 | 0.9879 |
| DoS_Hulk_attempted | 0.9045 | 0.9333 | 0.9072 | 0.9186 |
| DoS_GoldenEye_attempted | 0.6901 | 0.7862 | 0.6880 | 0.7384 |
| DoS_slowloris_attempted | 0.9500 | 0.9668 | 0.9068 | 0.9286 |
| DoS_Slowhttptest_attempted | 0.9819 | 0.9902 | 0.9789 | 0.9827 |
| Bot_attempted | 0.5976 | 0.6295 | 0.5805 | 0.6216 |
| WebAttack_BruteForce | 0.9440 | 0.9555 | 0.9391 | 0.9471 |
| WebAttack_BruteForce_attempted | 0.6811 | 0.7181 | 0.6735 | 0.6477 |

### Hallazgo central en cada configuración (ρ combinado; * = confirmado con Holm)

| descriptor | base β=1, k=8 | β=0,5 | β=2 | k=16 |
|---|---|---|---|---|
| Mahalanobis (mediana) | +0.772* | +0.913* | +0.603* | +0.647* |
| Fisher-Rao al 10.º posterior benigno | +0.775* | +0.859* | +0.716* | +0.711* |
| kNN en el espacio decodificado | +0.680* | +0.864* | +0.639 | +0.651* |
| Tasa KL(q\|\|p) por flujo | +0.830* | +0.941* | +0.658 | +0.807* |
| kNN al benigno (z blanqueado) | +0.741* | +0.948* | +0.707* | +0.537* |
| Log-magnificación del decodificador | +0.592* | +0.251 | +0.551 | +0.542* |
| Silhouette del grupo (Fisher-Rao) | -0.627* | -0.753* | -0.624 | -0.642* |

Figuras: `outputs/oe4/figures/`.

## 6. Archivos para el commit

Rutas a versionar (`ARCHIVOS_PARA_COMMIT.txt`, las usa `scripts/vm_preparar_commit.sh`):

| ruta | MB |
|---|---|
| outputs/vm_oe3_oe4 | 0.15 |
| outputs/oe2_descriptores_saneado/tables | 0.65 |
| outputs/oe2_descriptores_saneado/RESULTADOS_RECONCILIACION.md | 0.00 |
| outputs/oe3 | 0.49 |
| outputs/oe4 | 1195.74 |

No se versionan (pesados y regenerables, ver `.gitignore`): vectores latentes, scores por conexión, curvas ROC en `.npz`, cachés `_cache/` y `data/`.
