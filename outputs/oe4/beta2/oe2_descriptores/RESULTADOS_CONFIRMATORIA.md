# OE2 — prueba confirmatoria (perfil saneado, `oe4_beta2.yaml`)

Hipótesis registradas el 2026-10-10, antes de ver los resultados de este perfil (`fase7_confirmatoria.py`). 5 semillas (42, 43, 44, 45, 46), 11 familias, 10,000 permutaciones, 2,000 remuestreos jerárquicos.

## Hipótesis

| id | descriptor | rol | signo_esperado | rho_modelo_51 | rho | ic_lo | ic_hi | p_bilateral | p_holm | confirmada | rho_sin_portscan_bot | rho_attempted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.1 | Mahalanobis (mediana) | P | 1 | 0.604 | 0.603 | -0.037 | 0.956 | 0.031 | 0.031 | True | 0.257 | 0.854 |
| 2.4 | Fisher-Rao al 10.º posterior benigno | S | 1 | 0.724 | 0.716 | 0.275 | 0.902 | 0.008 | 0.044 | True | 0.512 | 0.607 |
| 6.2 | kNN en el espacio decodificado | S | 1 | 0.691 | 0.639 | 0.096 | 0.948 | 0.021 | 0.085 | False | 0.439 | 0.564 |
| 7.1 | Tasa KL(q\|\|p) por flujo | S | 1 | 0.681 | 0.658 | 0.074 | 0.935 | 0.028 | 0.085 | False | 0.380 | 0.886 |
| 5.1 | kNN al benigno (z blanqueado) | S | 1 | 0.694 | 0.707 | 0.065 | 0.944 | 0.007 | 0.044 | True | 0.452 | 0.567 |
| 6.1 | Log-magnificación del decodificador | S | 1 | 0.523 | 0.551 | -0.176 | 0.936 | 0.063 | 0.085 | False | 0.163 | 0.954 |
| 10.1b | Silhouette del grupo (Fisher-Rao) | S | -1 | -0.629 | -0.624 | -0.913 | -0.106 | 0.038 | 0.085 | False | -0.626 | -0.765 |

- Primario confirmado: **sí**.
- Secundarios confirmados (Holm entre 6): **2.4, 5.1**.

## Nivel de flujo (entre familias y dentro de cada familia)

Entre familias = el rho combinado de la tabla de arriba. Total = flujos de las 11 familias juntos. Dentro = rangos centrados en cada familia (hasta 5000 flujos por familia y semilla).

| descriptor | rho_total | rho_dentro_familia | rho_medio_por_familia | rho_dentro_min | rho_dentro_max |
|---|---|---|---|---|---|
| Mahalanobis (mediana) | 0.683 | 0.685 | 0.480 | 0.598 | 0.743 |
| Fisher-Rao al 10.º posterior benigno | 0.724 | 0.629 | 0.513 | 0.561 | 0.661 |
| kNN en el espacio decodificado | 0.670 | 0.486 | 0.397 | 0.410 | 0.625 |
| Tasa KL(q\|\|p) por flujo | 0.717 | 0.738 | 0.656 | 0.694 | 0.783 |
| kNN al benigno (z blanqueado) | 0.684 | 0.629 | 0.564 | 0.538 | 0.737 |
| Log-magnificación del decodificador | 0.728 | 0.672 | 0.455 | 0.518 | 0.824 |

## AUC por familia: 51 frente a 66 variables

Spearman del ranking de las 11 familias por AUC medio entre los dos modelos: **+1.000**.

| group | auc_51_media | auc_51_sd | auc_66_media | auc_66_sd | diferencia_66_menos_51 |
|---|---|---|---|---|---|
| PortScan | 0.667 | 0.031 | 0.656 | 0.015 | -0.011 |
| DoS_Hulk | 0.916 | 0.015 | 0.894 | 0.009 | -0.022 |
| DDoS | 0.976 | 0.012 | 0.972 | 0.009 | -0.004 |
| DoS_GoldenEye | 0.960 | 0.009 | 0.962 | 0.008 | 0.002 |
| DoS_slowloris | 0.945 | 0.006 | 0.937 | 0.003 | -0.008 |
| FTP_Patator | 0.927 | 0.014 | 0.917 | 0.016 | -0.010 |
| SSH_Patator | 0.958 | 0.033 | 0.962 | 0.024 | 0.005 |
| DoS_Slowhttptest | 0.994 | 0.002 | 0.991 | 0.001 | -0.003 |
| Bot | 0.624 | 0.017 | 0.598 | 0.008 | -0.026 |
| Web_Attack | 0.932 | 0.011 | 0.924 | 0.012 | -0.008 |
| Infiltration | 0.992 | 0.001 | 0.989 | 0.001 | -0.003 |
| DoS_Hulk_attempted | 0.938 | 0.012 | 0.907 | 0.007 | -0.031 |
| DoS_GoldenEye_attempted | 0.803 | 0.044 | 0.688 | 0.035 | -0.115 |
| DoS_slowloris_attempted | 0.949 | 0.025 | 0.907 | 0.019 | -0.042 |
| DoS_Slowhttptest_attempted | 0.988 | 0.003 | 0.979 | 0.002 | -0.009 |
| Bot_attempted | 0.613 | 0.016 | 0.580 | 0.012 | -0.033 |
| WebAttack_BruteForce_attempted | 0.717 | 0.060 | 0.674 | 0.011 | -0.044 |
