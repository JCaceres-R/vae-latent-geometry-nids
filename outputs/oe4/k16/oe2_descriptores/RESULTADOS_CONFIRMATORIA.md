# OE2 — prueba confirmatoria (perfil saneado, `oe4_k16.yaml`)

Hipótesis registradas el 2026-10-10, antes de ver los resultados de este perfil (`fase7_confirmatoria.py`). 5 semillas (42, 43, 44, 45, 46), 11 familias, 10,000 permutaciones, 2,000 remuestreos jerárquicos.

## Hipótesis

| id | descriptor | rol | signo_esperado | rho_modelo_51 | rho | ic_lo | ic_hi | p_bilateral | p_holm | confirmada | rho_sin_portscan_bot | rho_attempted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.1 | Mahalanobis (mediana) | P | 1 | 0.604 | 0.647 | 0.087 | 0.967 | 0.003 | 0.003 | True | 0.450 | 0.732 |
| 2.4 | Fisher-Rao al 10.º posterior benigno | S | 1 | 0.724 | 0.711 | 0.295 | 0.913 | 0.002 | 0.010 | True | 0.486 | 0.597 |
| 6.2 | kNN en el espacio decodificado | S | 1 | 0.691 | 0.651 | 0.201 | 0.897 | 0.004 | 0.015 | True | 0.375 | 0.710 |
| 7.1 | Tasa KL(q\|\|p) por flujo | S | 1 | 0.681 | 0.807 | 0.327 | 0.987 | 0.000 | 0.002 | True | 0.627 | 0.856 |
| 5.1 | kNN al benigno (z blanqueado) | S | 1 | 0.694 | 0.537 | -0.062 | 0.954 | 0.016 | 0.049 | True | 0.170 | 0.559 |
| 6.1 | Log-magnificación del decodificador | S | 1 | 0.523 | 0.542 | -0.164 | 0.875 | 0.020 | 0.049 | True | 0.163 | 0.691 |
| 10.1b | Silhouette del grupo (Fisher-Rao) | S | -1 | -0.629 | -0.642 | -0.929 | -0.143 | 0.016 | 0.049 | True | -0.632 | -0.612 |

- Primario confirmado: **sí**.
- Secundarios confirmados (Holm entre 6): **2.4, 6.2, 7.1, 5.1, 6.1, 10.1b**.

## Nivel de flujo (entre familias y dentro de cada familia)

Entre familias = el rho combinado de la tabla de arriba. Total = flujos de las 11 familias juntos. Dentro = rangos centrados en cada familia (hasta 5000 flujos por familia y semilla).

| descriptor | rho_total | rho_dentro_familia | rho_medio_por_familia | rho_dentro_min | rho_dentro_max |
|---|---|---|---|---|---|
| Mahalanobis (mediana) | 0.681 | 0.686 | 0.579 | 0.290 | 0.767 |
| Fisher-Rao al 10.º posterior benigno | 0.800 | 0.689 | 0.584 | 0.483 | 0.767 |
| kNN en el espacio decodificado | 0.703 | 0.538 | 0.306 | 0.289 | 0.674 |
| Tasa KL(q\|\|p) por flujo | 0.821 | 0.785 | 0.638 | 0.597 | 0.868 |
| kNN al benigno (z blanqueado) | 0.684 | 0.658 | 0.555 | 0.452 | 0.764 |
| Log-magnificación del decodificador | 0.618 | 0.482 | 0.245 | 0.395 | 0.577 |

## AUC por familia: 51 frente a 66 variables

Spearman del ranking de las 11 familias por AUC medio entre los dos modelos: **+0.973**.

| group | auc_51_media | auc_51_sd | auc_66_media | auc_66_sd | diferencia_66_menos_51 |
|---|---|---|---|---|---|
| PortScan | 0.667 | 0.031 | 0.665 | 0.035 | -0.002 |
| DoS_Hulk | 0.916 | 0.015 | 0.930 | 0.043 | 0.014 |
| DDoS | 0.976 | 0.012 | 0.983 | 0.010 | 0.007 |
| DoS_GoldenEye | 0.960 | 0.009 | 0.971 | 0.019 | 0.011 |
| DoS_slowloris | 0.945 | 0.006 | 0.956 | 0.008 | 0.011 |
| FTP_Patator | 0.927 | 0.014 | 0.924 | 0.042 | -0.003 |
| SSH_Patator | 0.958 | 0.033 | 0.945 | 0.039 | -0.013 |
| DoS_Slowhttptest | 0.994 | 0.002 | 0.985 | 0.010 | -0.009 |
| Bot | 0.624 | 0.017 | 0.628 | 0.014 | 0.004 |
| Web_Attack | 0.932 | 0.011 | 0.933 | 0.009 | 0.002 |
| Infiltration | 0.992 | 0.001 | 0.988 | 0.002 | -0.004 |
| DoS_Hulk_attempted | 0.938 | 0.012 | 0.919 | 0.018 | -0.020 |
| DoS_GoldenEye_attempted | 0.803 | 0.044 | 0.738 | 0.057 | -0.065 |
| DoS_slowloris_attempted | 0.949 | 0.025 | 0.929 | 0.048 | -0.020 |
| DoS_Slowhttptest_attempted | 0.988 | 0.003 | 0.983 | 0.004 | -0.005 |
| Bot_attempted | 0.613 | 0.016 | 0.622 | 0.031 | 0.008 |
| WebAttack_BruteForce_attempted | 0.717 | 0.060 | 0.648 | 0.035 | -0.070 |
