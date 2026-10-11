# OE2 — prueba confirmatoria (perfil saneado)

Hipótesis registradas el 2026-10-10, antes de ver los resultados de este perfil (`fase7_confirmatoria.py`). 8 semillas (42, 43, 44, 45, 46, 47, 48, 49), 11 familias, 10,000 permutaciones, 2,000 remuestreos jerárquicos.

## Hipótesis

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

## Nivel de flujo (entre familias y dentro de cada familia)

Entre familias = el rho combinado de la tabla de arriba. Total = flujos de las 11 familias juntos. Dentro = rangos centrados en cada familia (hasta 5000 flujos por familia y semilla).

| descriptor | rho_total | rho_dentro_familia | rho_medio_por_familia | rho_dentro_min | rho_dentro_max |
|---|---|---|---|---|---|
| Mahalanobis (mediana) | 0.779 | 0.716 | 0.601 | 0.540 | 0.822 |
| Fisher-Rao al 10.º posterior benigno | 0.841 | 0.706 | 0.598 | 0.646 | 0.734 |
| kNN en el espacio decodificado | 0.737 | 0.596 | 0.443 | 0.204 | 0.724 |
| Tasa KL(q\|\|p) por flujo | 0.803 | 0.747 | 0.654 | 0.655 | 0.853 |
| kNN al benigno (z blanqueado) | 0.795 | 0.706 | 0.621 | 0.554 | 0.813 |
| Log-magnificación del decodificador | 0.708 | 0.646 | 0.403 | 0.380 | 0.782 |

## AUC por familia: 51 frente a 66 variables

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
