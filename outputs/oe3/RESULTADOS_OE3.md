# OE3 — calidad generativa del decodificador

Modelos del OE1 saneado (8 semillas). Comparación frente a `test_benign` en la escala del modelo. `real_val` = val_benign frente a test_benign: lo que difieren dos muestras reales.

## Resumen (media entre semillas)

| comparacion | ks_mediana | ks_max | frac_variables_ks_menor_0_1 | w1_media | frac_fuera_0_1 | sw2 | c2st_auc | dcr_mediana | dcr_mediana_real_test | dcr_razon | frac_copias_exactas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real_val | 0.0035 | 0.0075 | 1.0000 | 0.0004 | 0.0000 | 0.0071 | 0.4987 | 0.0002 | 0.0002 | 0.9852 | 0.2213 |
| prior_muestra | 0.4555 | 0.5501 | 0.0076 | 0.0324 | 0.3949 | 0.7164 | 1.0000 | 0.2185 | 0.0002 | 1399.8233 | 0.0000 |
| prior_media | 0.5154 | 0.9871 | 0.0189 | 0.0152 | 0.3281 | 0.0602 | 1.0000 | nan | nan | nan | nan |
| posterior_agregado | 0.4493 | 0.5389 | 0.0133 | 0.0142 | 0.4029 | 0.0622 | 1.0000 | 0.0643 | 0.0002 | 414.6713 | 0.0000 |

- `c2st_auc`: 0,5 = el clasificador no distingue; cuanto más cerca de 1, más fácil separar sintético de real.
- `dcr_razon` < 1: los sintéticos quedan más cerca de train que los reales de test (señal de memorización); `frac_copias_exactas` > 0: copias literales.

## Las 10 variables peor reproducidas (z ~ prior, KS medio entre semillas)

| variable | ks_media |
|---|---|
| CWR Flag Count | 0.530 |
| URG Flag Count | 0.525 |
| Fwd URG Flags | 0.522 |
| Fwd Bulk Rate Avg | 0.520 |
| Subflow Bwd Packets | 0.516 |
| Bwd URG Flags | 0.513 |
| Bwd Bytes/Bulk Avg | 0.513 |
| Bwd Packet/Bulk Avg | 0.513 |
| Total Bwd packets | 0.512 |
| ECE Flag Count | 0.511 |

Figuras: `outputs/oe3/figures/` (KS por variable e histogramas de la semilla 42; C2ST).
