# OE2 — Fase 5: correlación geometría-detectabilidad (resultado central)

Une `oe2_geometric_metrics.csv` (Fase 3) y `oe2_detectability.csv` (Fase 4)
por `group`. Commit `4a0e159` — "OE2 fase 5: correlación
geometría-detectabilidad (resultado central)".

## 1. Merge

18 grupos, 1 a 1 confirmado entre ambos archivos (assert en el script,
sin discrepancias).

## 2. Spearman sobre las 11 familias principales

| Par | rho | p-valor | Significativo (p<0.05) |
|---|---:|---:|---|
| mahalanobis_median vs auc_roc | **0.3818** | **0.2466** | No |
| mahalanobis_median vs tpr_at_tau | 0.3364 | 0.3118 | No |
| silhouette_score vs auc_roc | 0.1727 | 0.6115 | No |
| silhouette_score vs tpr_at_tau | -0.0455 | 0.8944 | No |
| mahalanobis_mean vs auc_roc | 0.5000 | 0.1173 | No |
| mahalanobis_mean vs tpr_at_tau | 0.4727 | 0.1420 | No |

**Coincide exactamente con el preview**: rho≈0.38 para Mahalanobis
(mediana) vs. AUC, no significativo con n=11. Ningún número aquí lo
contradice.

Usando `mahalanobis_mean` en vez de mediana el rho sube a 0.50 (vs. AUC) y
0.47 (vs. TPR@tau) — más alto pero **tampoco significativo** (p=0.117 y
p=0.142). Es coherente con que la media está más influenciada por
outliers como `Infiltration` (mean=35.6 vs. mediana=4.35, Fase 3) — sube
la correlación aparente pero no cambia la conclusión: con n=11, ninguna de
las 6 combinaciones geometría↔detectabilidad alcanza significancia
estadística. Silhouette score, en particular, no muestra ninguna relación
(rho cerca de 0, incluso signo negativo con TPR@tau).

## 3. Scatter plot

`outputs/figures/oe2_geometry_vs_detectability.png` — Mahalanobis
(mediana) en x, AUC-ROC en y, 11 familias etiquetadas. Sin línea de
tendencia (rho no significativo, no se maquilla).

## 4. Comparación completado vs. attempted (6 pares)

| Par | delta_mahalanobis_median | delta_auc_roc | Mismo signo | Consistente con hipótesis (ambos +) |
|---|---:|---:|---|---|
| DoS_Hulk vs. attempted | -2.5082 | -0.0182 | Sí | **No** (ambos negativos) |
| DoS_GoldenEye vs. attempted | +1.2001 | +0.1567 | Sí | Sí |
| DoS_slowloris vs. attempted | +2.5599 | -0.0317 | No | No |
| DoS_Slowhttptest vs. attempted | -1.7907 | +0.0050 | No | No |
| Bot vs. attempted | -1.0694 | +0.0320 | No | No |
| WebAttack_BruteForce vs. attempted | +1.4207 | +0.2152 | Sí | Sí |

**3/6 pares con el mismo signo en ambos deltas; solo 2/6 consistentes con
la hipótesis** (completo más lejos Y más detectable que attempted:
`DoS_GoldenEye` y `WebAttack_BruteForce`).

El caso a marcar: **`DoS_Hulk`** tiene el mismo signo en ambos deltas
(same_sign=True) pero **ambos negativos** — es decir, `DoS_Hulk`
*completo* está más cerca del benigno (menor Mahalanobis) Y es *menos*
detectable que su variante `- Attempted`, exactamente al revés de la
hipótesis, aunque "consistente" en el sentido de que geometría y
detectabilidad se mueven juntas. Los otros 3 pares (`DoS_slowloris`,
`DoS_Slowhttptest`, `Bot`) tienen signos opuestos entre mahalanobis y AUC
— para esos, geometría y detectabilidad ni siquiera se mueven juntas.

Con solo 2/6 pares confirmando la hipótesis completo>attempted en ambas
métricas simultáneamente, esta comparación secundaria **no respalda** una
relación sistemática completo/attempted en la dirección esperada.

## Archivos

- `outputs/metrics/oe2_correlation.json` (rhos, p-valores, los 6 deltas
  con signo y flag de consistencia).
- `outputs/figures/oe2_geometry_vs_detectability.png`.
- Script: `src/vae_nids/evaluation/correlation_oe2.py`.
