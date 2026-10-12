# Reconciliación del nivel de conexión (fase 8)

Misma muestra de conexiones que el modelo mixto de la fase 5 (hasta 5000 por grupo y semilla; 309,248 filas, 8 semillas). x = Mahalanobis latente (phase4, `latent_maha`); y = NLL Monte Carlo (`nll_mc`).

## Correlaciones (media de Fisher entre semillas)

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

## Modelos

| modelo | efecto | ic_lo | ic_hi | p | var_pendiente_familia | r2_marginal | n | convergio |
|---|---|---|---|---|---|---|---|---|
| M1 fase 5: mixto, valores estandarizados, con benigno | 0.104 | -0.099 | 0.306 | 0.315 | 0.126 | 0.002 | 309248 | True |
| M2: mixto, valores estandarizados, sin benigno | 0.106 | -0.091 | 0.303 | 0.292 | 0.110 | 0.002 | 269248 | True |
| M3: mixto, puntuaciones normales de rango, sin benigno | 0.718 | 0.581 | 0.854 | 0.000 | 0.053 | 0.501 | 269248 | True |
| M4: MCO con efectos fijos de familia y semilla (dentro de familia), rangos normales | 0.604 | 0.451 | 0.757 | 0.000 | — | 0.709 | 269248 | True |

## Por familia (media entre semillas)

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
