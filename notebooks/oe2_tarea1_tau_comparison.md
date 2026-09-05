# OE2 — Tarea 1: impacto del percentil de τ (95 vs. 99)

Análisis adicional, **no modifica nada de la Fase 4** — `detectability_oe2.py`
y `oe2_detectability.csv` quedan intactos. Reusa sus mismas funciones
(mismo modelo, mismo score de NLL de reconstrucción, mismos `.npy` de la
Fase 2) desde un script nuevo. AUC-ROC no depende de τ, no se recalculó.

Script: `src/vae_nids/evaluation/tau_comparison_oe2.py`.
Tabla: `outputs/metrics/oe2_tau_comparison.csv`.

## τ calibrado con ambos percentiles

Sobre el mismo `val_benign.parquet` completo (fresco, igual que la Fase 4):

| Percentil | τ |
|---|---:|
| 95 (vigente en el código) | -156.9726 |
| 99 (especificado en la monografía) | -132.5056 |

## Tabla completa (18 grupos)

| group | n | TPR@τ95 | TPR@τ99 | delta_tpr |
|---|---:|---:|---:|---:|
| PortScan | 159,023 | 0.0002 | 0.0000 | +0.0002 |
| DoS_Hulk | 158,469 | 0.0673 | 0.0141 | +0.0532 |
| DDoS | 95,123 | 0.7937 | 0.1446 | **+0.6491** |
| DoS_GoldenEye | 7,567 | 0.4761 | 0.2825 | +0.1936 |
| DoS_slowloris | 4,001 | 0.8065 | 0.7253 | +0.0812 |
| FTP_Patator | 3,973 | 0.0035 | 0.0000 | +0.0035 |
| SSH_Patator | 2,980 | 0.9842 | 0.2369 | **+0.7473** |
| DoS_Slowhttptest | 1,742 | 0.9943 | 0.6487 | +0.3456 |
| Bot | 738 | 0.0596 | 0.0379 | +0.0217 |
| Web_Attack | 190 | 0.6000 | 0.4684 | +0.1316 |
| Infiltration | 32 | 0.9688 | 0.6250 | +0.3438 |
| DoS_Hulk_attempted | 579 | 0.7323 | 0.3230 | +0.4093 |
| DoS_GoldenEye_attempted | 80 | 0.0250 | 0.0250 | +0.0000 |
| DoS_slowloris_attempted | 1,706 | 0.9578 | 0.0023 | **+0.9555** |
| DoS_Slowhttptest_attempted | 3,367 | 0.8931 | 0.8307 | +0.0624 |
| Bot_attempted | 1,470 | 0.0000 | 0.0000 | +0.0000 |
| WebAttack_BruteForce | 151 | 0.6358 | 0.4702 | +0.1656 |
| WebAttack_BruteForce_attempted | 1,214 | 0.0000 | 0.0000 | +0.0000 |

`delta_tpr = TPR@τ95 - TPR@τ99`, siempre positivo aquí (τ99 es más
exigente — más alto — así que nunca detecta más que τ95).

## ¿Algún grupo cambia de forma dramática?

**Sí, uno de forma extrema y dos de forma fuerte:**

1. **`DoS_slowloris_attempted`** — el caso más dramático de los 18: pasa
   de **TPR=0.9578 (casi perfecto) a TPR=0.0023 (esencialmente cero)**
   con τ99. Con τ95 casi todas las muestras se detectan; con τ99 casi
   ninguna. Este grupo por sí solo debería pesar fuerte en la decisión
   del percentil.
2. **`SSH_Patator`**: 0.9842 → 0.2369 (delta +0.7473) — de detección
   casi total a menos de una cuarta parte.
3. **`DDoS`**: 0.7937 → 0.1446 (delta +0.6491) — de mayoría detectada a
   minoría detectada, y es la familia con más volumen (n=95,123) después
   de PortScan/DoS_Hulk, así que el impacto práctico es grande.

Con cambios más moderados pero no triviales: `DoS_Hulk_attempted`
(+0.4093), `DoS_Slowhttptest` (+0.3456, aunque se mantiene alto en ambos:
0.99→0.65), `Infiltration` (+0.3438, también se mantiene alto en ambos:
0.97→0.63).

Grupos que ya eran indetectables con τ95 (`PortScan`, `FTP_Patator`,
`Bot`, `Bot_attempted`, `WebAttack_BruteForce_attempted`,
`DoS_GoldenEye_attempted`) no cambian de forma relevante — ya estaban
cerca de 0 y siguen igual.

**Lectura para la decisión**: τ99 es sistemáticamente más conservador
(nunca detecta más que τ95, por construcción), pero el costo no es
uniforme — colapsa casi por completo la detección de
`DoS_slowloris_attempted` y golpea fuerte a `SSH_Patator` y `DDoS`,
mientras deja prácticamente intactos a los grupos que ya eran difíciles.
Esto es lo que hay que sopesar contra la ventaja de τ99 (menos falsos
positivos esperados sobre benigno, por construcción — no medido aquí,
solo se pidió TPR).
