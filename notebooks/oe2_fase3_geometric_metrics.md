# OE2 — Fase 3: métricas geométricas (Mahalanobis + silhouette)

Sin AUC-ROC todavía — eso es la fase siguiente. Lee los `.npy` y
`manifest.json` de la Fase 2 (`oe2_fase2_encoding.md`). Commit
`8fbf0e0` — "OE2 fase 3: métricas geométricas (Mahalanobis + silhouette,
dims activas)".

## 0. Verificación de dimensiones activas (no asumida, recalculada)

Recalculado sobre `latent_benign_test.npy` completo (n=248,561), mismo
criterio de Burda et al. (2016, umbral 0.01) y misma función que OE1
(`evaluation.metrics_oe1.active_units`) — no se reutilizó el resultado del
documento de cierre de OE1, que era de otra corrida (`val_benign`) dentro
del mismo barrido.

| dim | Var_x[mu_j] | Estado |
|---:|---:|---|
| 0 | 0.572597 | activa |
| 1 | 0.255990 | activa |
| **2** | **0.000009** | **colapsada — excluida** |
| 3 | 0.953659 | activa |
| 4 | 0.965305 | activa |
| 5 | 0.255043 | activa |
| 6 | 0.773592 | activa |
| **7** | **0.000028** | **colapsada — excluida** |

**Dimensiones excluidas: 2 y 7**, con varianzas 0.000009 y 0.000028
respectivamente (umbral 0.01). Coincide cualitativamente con lo reportado
en el cierre de OE1 para la corrida seed=42/input51 (dims 2 y 7
colapsadas), pero el valor exacto es distinto porque aquí se mide sobre
`test_benign` y allá sobre `val_benign` — confirmado de forma
independiente, no asumido. **Dims activas usadas en todo lo siguiente:
[0, 1, 3, 4, 5, 6] (6/8).**

## 1-3. Tabla completa (18 grupos)

Centroide/covarianza benigno calculados solo sobre las 6 dims activas, con
`latent_benign_test.npy` completo (n=248,561). Mahalanobis por muestra
contra ese benigno; silhouette con submuestreo a 5,000 por grupo si supera
ese tamaño (seed=42).

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

`active_dims_used` = `[0, 1, 3, 4, 5, 6]` para las 18 filas (columna
completa en el CSV, omitida aquí por ser constante).

## Notas de lectura (no interpretación de OE2 todavía, solo observaciones de forma)

- **Infiltration** (n=32): media Mahalanobis muy alta (35.6) pero mediana
  baja (4.35) y silhouette la más alta de la tabla (0.882) — exactamente
  el caso que motivó pedir mediana/percentiles en vez de solo la media: un
  outlier extremo (p75=27.2) domina el promedio en un grupo chico.
- **FTP_Patator, SSH_Patator y las variantes `_attempted` con IQR casi nulo**
  (p25≈p75, ej. `DoS_slowloris_attempted`, `Bot_attempted`,
  `DoS_Slowhttptest_attempted`): sugiere grupos muy homogéneos/concentrados
  en un punto del espacio latente — a confirmar en la fase de
  interpretación, no aquí.
- Archivo fuente: `outputs/metrics/oe2_geometric_metrics.csv` (columnas:
  `group, n_total, n_used_silhouette, mahalanobis_mean, mahalanobis_median,
  mahalanobis_p25, mahalanobis_p75, silhouette_score, active_dims_used`).

Script: `src/vae_nids/evaluation/geometric_metrics_oe2.py`.
