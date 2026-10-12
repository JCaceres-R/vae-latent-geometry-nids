# OE4 — reproducibilidad del ranking frente a β y k

Configuraciones con 66 variables y semillas [42, 43, 44, 45, 46]; la base (β = 1, k = 8) se restringe a esas semillas en todo salvo la fila del hallazgo, que usa su fase 7 completa (8 semillas).

## Configuraciones

| configuracion | beta | k | semillas | activas_media | activas_sd | kl_total | mejor_epoca | nll_por_columna | frac_piso_decodificador |
|---|---|---|---|---|---|---|---|---|---|
| base β=1, k=8 | 1.000 | 8 | 5 | 7.000 | 0.000 | 8.699 | 60.200 | -3.835 | 0.902 |
| β=0,5 | 0.500 | 8 | 5 | 8.000 | 0.000 | 13.293 | 94.800 | -3.895 | 0.927 |
| β=2 | 2.000 | 8 | 5 | 5.200 | 0.837 | 4.334 | 96.600 | -3.821 | 0.890 |
| k=16 | 1.000 | 16 | 5 | 6.800 | 0.447 | 7.693 | 130.200 | -3.869 | 0.916 |

## Concordancia del ranking de las 11 familias frente a la base

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

## AUC por familia

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

## Hallazgo central en cada configuración (ρ combinado; * = confirmado con Holm)

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