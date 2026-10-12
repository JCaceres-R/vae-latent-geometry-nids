# OE2 — prueba confirmatoria (perfil saneado, `oe4_beta0p5.yaml`)

Hipótesis registradas el 2026-10-10, antes de ver los resultados de este perfil (`fase7_confirmatoria.py`). 5 semillas (42, 43, 44, 45, 46), 11 familias, 10,000 permutaciones, 2,000 remuestreos jerárquicos.

## Hipótesis

| id | descriptor | rol | signo_esperado | rho_modelo_51 | rho | ic_lo | ic_hi | p_bilateral | p_holm | confirmada | rho_sin_portscan_bot | rho_attempted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.1 | Mahalanobis (mediana) | P | 1 | 0.604 | 0.913 | 0.667 | 0.997 | 0.000 | 0.000 | True | 0.884 | 0.882 |
| 2.4 | Fisher-Rao al 10.º posterior benigno | S | 1 | 0.724 | 0.859 | 0.556 | 0.982 | 0.001 | 0.002 | True | 0.764 | 0.777 |
| 6.2 | kNN en el espacio decodificado | S | 1 | 0.691 | 0.864 | 0.561 | 0.989 | 0.001 | 0.002 | True | 0.784 | 0.744 |
| 7.1 | Tasa KL(q\|\|p) por flujo | S | 1 | 0.681 | 0.941 | 0.743 | 0.999 | 0.000 | 0.001 | True | 0.892 | 0.913 |
| 5.1 | kNN al benigno (z blanqueado) | S | 1 | 0.694 | 0.948 | 0.794 | 0.996 | 0.000 | 0.001 | True | 0.914 | 0.618 |
| 6.1 | Log-magnificación del decodificador | S | 1 | 0.523 | 0.251 | -0.571 | 0.847 | 0.382 | 0.382 | False | -0.333 | 0.805 |
| 10.1b | Silhouette del grupo (Fisher-Rao) | S | -1 | -0.629 | -0.753 | -0.930 | -0.365 | 0.009 | 0.017 | True | -0.698 | -0.600 |

- Primario confirmado: **sí**.
- Secundarios confirmados (Holm entre 6): **2.4, 6.2, 7.1, 5.1, 10.1b**.

## Nivel de flujo (entre familias y dentro de cada familia)

Entre familias = el rho combinado de la tabla de arriba. Total = flujos de las 11 familias juntos. Dentro = rangos centrados en cada familia (hasta 5000 flujos por familia y semilla).

| descriptor | rho_total | rho_dentro_familia | rho_medio_por_familia | rho_dentro_min | rho_dentro_max |
|---|---|---|---|---|---|
| Mahalanobis (mediana) | 0.769 | 0.567 | 0.533 | 0.402 | 0.677 |
| Fisher-Rao al 10.º posterior benigno | 0.872 | 0.694 | 0.582 | 0.648 | 0.796 |
| kNN en el espacio decodificado | 0.838 | 0.640 | 0.510 | 0.579 | 0.716 |
| Tasa KL(q\|\|p) por flujo | 0.841 | 0.658 | 0.620 | 0.556 | 0.774 |
| kNN al benigno (z blanqueado) | 0.843 | 0.619 | 0.540 | 0.499 | 0.763 |
| Log-magnificación del decodificador | 0.493 | 0.586 | 0.266 | 0.373 | 0.759 |

## AUC por familia: 51 frente a 66 variables

Spearman del ranking de las 11 familias por AUC medio entre los dos modelos: **+0.791**.

| group | auc_51_media | auc_51_sd | auc_66_media | auc_66_sd | diferencia_66_menos_51 |
|---|---|---|---|---|---|
| PortScan | 0.667 | 0.031 | 0.696 | 0.047 | 0.030 |
| DoS_Hulk | 0.916 | 0.015 | 0.965 | 0.009 | 0.049 |
| DDoS | 0.976 | 0.012 | 0.995 | 0.003 | 0.020 |
| DoS_GoldenEye | 0.960 | 0.009 | 0.983 | 0.005 | 0.023 |
| DoS_slowloris | 0.945 | 0.006 | 0.942 | 0.016 | -0.004 |
| FTP_Patator | 0.927 | 0.014 | 0.885 | 0.010 | -0.042 |
| SSH_Patator | 0.958 | 0.033 | 0.905 | 0.020 | -0.052 |
| DoS_Slowhttptest | 0.994 | 0.002 | 0.968 | 0.012 | -0.026 |
| Bot | 0.624 | 0.017 | 0.593 | 0.019 | -0.031 |
| Web_Attack | 0.932 | 0.011 | 0.942 | 0.012 | 0.010 |
| Infiltration | 0.992 | 0.001 | 0.982 | 0.002 | -0.010 |
| DoS_Hulk_attempted | 0.938 | 0.012 | 0.933 | 0.009 | -0.005 |
| DoS_GoldenEye_attempted | 0.803 | 0.044 | 0.786 | 0.025 | -0.017 |
| DoS_slowloris_attempted | 0.949 | 0.025 | 0.967 | 0.020 | 0.018 |
| DoS_Slowhttptest_attempted | 0.988 | 0.003 | 0.990 | 0.002 | 0.003 |
| Bot_attempted | 0.613 | 0.016 | 0.630 | 0.039 | 0.016 |
| WebAttack_BruteForce_attempted | 0.717 | 0.060 | 0.718 | 0.025 | 0.001 |
