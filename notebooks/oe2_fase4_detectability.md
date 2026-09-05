# OE2 — Fase 4: detectabilidad por familia (AUC-ROC / TPR)

Sin correlación con la geometría todavía — eso es la Fase 5. Score de
anomalía = NLL de reconstrucción Gaussiana heteroscedástica por muestra
(sin el término β·KL, sin MSE plano). Decoder alimentado con las 8
dimensiones completas de `mu` (no las 6 activas de la Fase 3). Commit
`9a33157` — "OE2 fase 4: detectabilidad por familia (AUC-ROC/TPR)".

## Tau calibrado

**tau = -156.9726** (percentil 95 de los scores de reconstrucción sobre
`val_benign.parquet` completo, n=248,560 — split fresco, no reusado de
ningún `.npy` de la Fase 2/3).

Referencia negativa para todas las comparaciones: `test_benign` (n=248,561,
`latent_benign_test.npy` + x recargada), score medio = -195.16.

## Tabla completa (18 grupos)

| group | n | AUC-ROC | CI95 low | CI95 high | ancho CI | TPR@tau |
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
| Web_Attack | 190 | 0.9444 | 0.9322 | 0.9556 | **0.0234** | 0.6000 |
| Infiltration | 32 | 0.9928 | 0.9885 | 0.9959 | 0.0074 | 0.9688 |
| DoS_Hulk_attempted | 579 | 0.9358 | 0.9284 | 0.9427 | 0.0144 | 0.7323 |
| DoS_GoldenEye_attempted | 80 | 0.7945 | 0.7885 | 0.8021 | 0.0136 | 0.0250 |
| DoS_slowloris_attempted | 1,706 | 0.9728 | 0.9708 | 0.9748 | 0.0041 | 0.9578 |
| DoS_Slowhttptest_attempted | 3,367 | 0.9858 | 0.9851 | 0.9864 | 0.0014 | 0.8931 |
| Bot_attempted | 1,470 | 0.6128 | 0.6109 | 0.6148 | 0.0039 | 0.0000 |
| WebAttack_BruteForce | 151 | 0.9627 | 0.9540 | 0.9708 | **0.0168** | 0.6358 |
| WebAttack_BruteForce_attempted | 1,214 | 0.7475 | 0.7458 | 0.7496 | 0.0037 | 0.0000 |

CI95 = bootstrap estratificado, 1000 iteraciones, seed=42 (positivos y
negativos remuestreados por separado con tamaño fijo, para no perder una
clase entera en grupos chicos).

## Anchos de CI sospechosos (n insuficiente)

Ordenando por ancho de CI, de mayor a menor:

1. **Web_Attack** (n=190): ancho=0.0234 — el más ancho de los 18.
2. **WebAttack_BruteForce** (n=151): ancho=0.0168.
3. **DoS_Hulk_attempted** (n=579): ancho=0.0144.
4. **Bot** (n=738): ancho=0.0143.
5. **DoS_GoldenEye_attempted** (n=80): ancho=0.0136.

**Contraintuitivo**: **Infiltration (n=32, la muestra más chica de las
18) NO es la más ancha** (0.0074, empatado con `DoS_slowloris` que tiene
125× más muestras). La razón: el ancho del bootstrap depende tanto de n
como de qué tan lejos esté el AUC de 0.5 — Infiltration separa casi
perfecto (AUC=0.9928), y cuando la separación es casi perfecta casi todos
los remuestreos también lo son, así que la varianza cae aunque n sea
mínimo. El patrón real de "n insuficiente" aparece en grupos con AUC más
ambiguo y n moderado-bajo (Web_Attack, Bot, los `_attempted` chicos), no
necesariamente en el grupo de n absoluto más bajo.

## Nota de forma (no interpretación de OE2, solo observación)

Varios grupos con AUC decente muestran TPR@tau ≈ 0 (`Bot_attempted`,
`WebAttack_BruteForce_attempted`: 0.0000; `PortScan`: 0.0002;
`FTP_Patator`: 0.0035; `DoS_GoldenEye_attempted`: 0.0250) — el AUC mide
separación en todos los umbrales posibles, pero tau es un único punto de
corte (percentil 95 de benigno en validación); un grupo puede rankear
consistentemente más anómalo que el benigno típico sin llegar a cruzar
ese punto de corte específico. Queda para la Fase 5, no se interpreta aquí.

## Archivos

- Tabla: `outputs/metrics/oe2_detectability.csv` (columnas: `group,
  n_total, auc_roc, auc_roc_ci_low, auc_roc_ci_high, tpr_at_tau,
  tau_value`).
- Curvas ROC completas (fpr, tpr, thresholds) por grupo:
  `outputs/metrics/roc_curves/oe2_roc_{group}.npz` (18 archivos).
- Script: `src/vae_nids/evaluation/detectability_oe2.py`.
