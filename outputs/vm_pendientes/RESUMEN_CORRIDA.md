# Corrida de pendientes en la VM: OE1 saneado y OE2 recalculado

Generado por `scripts/vm_resumen_corrida.py` el 2026-10-11 03:37 UTC. Log cronológico: `RUN_LOG.md`. Todas las cifras salen de los CSV/JSON de `outputs/`; ninguna se escribe a mano.

## 1. Estado de las etapas

| id | etapa | estado | inicio | min | log |
|---|---|---|---|---|---|
| E01 | Tests (OE1 saneado y estadística del OE2) | OK | 2026-10-10T19:53:58Z | 0.6 | outputs/vm_pendientes/logs/E01.log |
| E10 | OE1: datos, preproceso, 8 semillas + referencia 74, diagnósticos | OK | 2026-10-10T19:54:35Z | 115.7 | outputs/vm_pendientes/logs/E10.log |
| E11 | OE1: verificación de checkpoints y diagnósticos | OK | 2026-10-10T21:50:18Z | 0.0 | outputs/vm_pendientes/logs/E11.log |
| E12 | OE1: figuras del informe (selección y preproceso) | OK | 2026-10-10T21:50:19Z | 0.2 | outputs/vm_pendientes/logs/E12.log |
| E13 | OE1: tablas LaTeX y cifras del informe | OK | 2026-10-10T21:50:28Z | 0.0 | outputs/vm_pendientes/logs/E13.log |
| E20 | OE2 robusto: codificación (mu, log sigma^2) por semilla | OK | 2026-10-10T21:50:28Z | 0.2 | outputs/vm_pendientes/logs/E20.log |
| E21 | OE2 robusto: geometría distribucional | OK | 2026-10-10T21:50:38Z | 28.4 | outputs/vm_pendientes/logs/E21.log |
| E22_s42 | OE2 robusto: detectabilidad, semilla 42 | OK | 2026-10-10T22:19:05Z | 11.4 | outputs/vm_pendientes/logs/E22_s42.log |
| E22_s43 | OE2 robusto: detectabilidad, semilla 43 | FALLO | 2026-10-10T22:30:27Z | 0.9 | outputs/vm_pendientes/logs/E22_s43.log |
| E22_s44 | OE2 robusto: detectabilidad, semilla 44 | OK | 2026-10-10T22:31:21Z | 11.0 | outputs/vm_pendientes/logs/E22_s44.log |
| E22_s45 | OE2 robusto: detectabilidad, semilla 45 | OK | 2026-10-10T22:42:24Z | 10.7 | outputs/vm_pendientes/logs/E22_s45.log |
| E22_s46 | OE2 robusto: detectabilidad, semilla 46 | OK | 2026-10-10T22:53:05Z | 10.9 | outputs/vm_pendientes/logs/E22_s46.log |
| E22_s47 | OE2 robusto: detectabilidad, semilla 47 | FALLO | 2026-10-10T23:03:58Z | 0.1 | outputs/vm_pendientes/logs/E22_s47.log |
| E22_s48 | OE2 robusto: detectabilidad, semilla 48 | FALLO | 2026-10-10T23:04:02Z | 0.1 | outputs/vm_pendientes/logs/E22_s48.log |
| E22_s49 | OE2 robusto: detectabilidad, semilla 49 | OK | 2026-10-10T23:04:07Z | 10.8 | outputs/vm_pendientes/logs/E22_s49.log |
| E22 | OE2 robusto: detectabilidad, tablas de todas las semillas | FALLO | 2026-10-10T23:14:52Z | 0.1 | outputs/vm_pendientes/logs/E22.log |
| E23 | OE2 robusto: correlación geometría-detectabilidad | SALTADA | 2026-10-10T23:14:56Z | 0.0 | outputs/vm_pendientes/logs/E23.log |
| E24 | OE2 robusto: figuras | SALTADA | 2026-10-10T23:14:56Z | 0.0 | outputs/vm_pendientes/logs/E24.log |
| E25 | OE2 robusto: UMAP (apoyo visual) | OK | 2026-10-10T23:14:56Z | 262.7 | outputs/vm_pendientes/logs/E25.log |
| E30 | Descriptores: detectabilidad y referencia benigna | FALLO | 2026-10-11T03:37:40Z | 0.1 | outputs/vm_pendientes/logs/E30.log |
| E31 | Descriptores: 17 descriptores por familia y semilla | SALTADA | 2026-10-11T03:37:45Z | 0.0 | outputs/vm_pendientes/logs/E31.log |
| E32 | Descriptores: cribado y validación (attempted) | SALTADA | 2026-10-11T03:37:45Z | 0.0 | outputs/vm_pendientes/logs/E32.log |
| E33 | Descriptores: duplicados, métrica del decodificador, UMAP sin duplicados | SALTADA | 2026-10-11T03:37:45Z | 0.0 | outputs/vm_pendientes/logs/E33.log |
| E34 | Descriptores: figuras | SALTADA | 2026-10-11T03:37:45Z | 0.0 | outputs/vm_pendientes/logs/E34.log |
| E35 | Descriptores: tablas LaTeX y cifras | SALTADA | 2026-10-11T03:37:45Z | 0.0 | outputs/vm_pendientes/logs/E35.log |
| E36 | Prueba confirmatoria (primario + 6 secundarios, Holm) | SALTADA | 2026-10-11T03:37:45Z | 0.0 | outputs/vm_pendientes/logs/E36.log |

Fallidas: **5**. Saltadas por dependencia: **8**.
Si alguna falló, la cola de su log está en `RUN_LOG.md` y el log completo en `outputs/vm_pendientes/logs/`.

## 2. Entorno

- `fecha_utc`: 2026-10-11T03:37:45+00:00
- `host`: tesis-juan
- `sistema`: Linux-5.15.0-194-generic-x86_64-with-glibc2.35
- `python`: 3.10.12
- `cpus`: 8
- `git_commit`: d5d14493617551cf625ab39ed0bf742aa527b1f1
- `git_rama`: main
- `git_arbol_limpio_al_resumir`: False
- `ram_gb`: 23.5
- `torch`: 2.13.0+cpu
- `numpy`: 2.2.6
- `pandas`: 2.3.3
- `scikit-learn`: 1.7.2
- `scipy`: 1.15.3
- `umap-learn`: 0.5.12
- `statsmodels`: 0.15.0

## 3. OE1 saneado: reentrenamiento y diagnósticos

Modelos del OE1 saneado (66 variables) y referencia de 74. Diagnósticos solo con `val_benign`.

| corrida | seed | n_columnas | epoca_parada | mejor_epoca | mejor_val_total | activas_burda | activas_kl | kl_total | tau_p95 | nll_media_por_columna | columnas_en_piso | frac_logvar_dec_en_piso |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oe1_info_nula | 42 | 66 | 87 | 77 | -242.748 | 7 | 7 | 9.784 | -219.926 | -3.857 | 66 | 0.910 |
| oe1_info_nula | 43 | 66 | 57 | 47 | -241.935 | 7 | 7 | 7.883 | -196.207 | -3.815 | 65 | 0.888 |
| oe1_info_nula | 44 | 66 | 74 | 64 | -245.045 | 7 | 7 | 8.532 | -223.585 | -3.870 | 66 | 0.915 |
| oe1_info_nula | 45 | 66 | 85 | 75 | -245.152 | 7 | 7 | 9.008 | -222.822 | -3.880 | 66 | 0.921 |
| oe1_info_nula | 46 | 66 | 48 | 38 | -236.982 | 7 | 7 | 8.288 | -193.439 | -3.751 | 64 | 0.875 |
| oe1_info_nula | 47 | 66 | 109 | 99 | -246.341 | 7 | 7 | 7.467 | -221.164 | -3.871 | 66 | 0.915 |
| oe1_info_nula | 48 | 66 | 122 | 112 | -243.526 | 7 | 7 | 8.793 | -202.471 | -3.850 | 66 | 0.910 |
| oe1_info_nula | 49 | 66 | 95 | 85 | -244.236 | 6 | 6 | 8.482 | -219.092 | -3.857 | 66 | 0.902 |
| referencia_74 | 42 | 74 | 91 | 81 | -270.471 | 7 | 7 | 9.652 | -221.478 | -3.814 | 73 | 0.888 |

- Unidades activas (Burda, Var(μ) > 0,01), 66 variables: media **6.88** (sd 0.35; 7, 7, 7, 7, 7, 7, 7, 6).
- KL total media: 8.53 nats. Fracción de log σ² del decodificador en el piso −10: 0.905.
- Referencia 74 (semilla 42): 7 unidades activas, KL 9.65.

**Antes y después** (medias por semilla):

| modelo | semillas | activas | kl_total | mejor_epoca | nll_por_columna | frac_piso |
|---|---|---|---|---|---|---|
| OE1 actual, 51 variables | 5 | 5.400 | 6.897 | 76.800 | -3.836 | 0.899 |
| OE1 saneado, 66 variables | 8 | 6.875 | 8.530 | 74.625 | -3.844 | 0.905 |

- Filas eliminadas por inf/NaN: 793. Filtro D4 por partición: train_benign 24, val_benign 7, test_benign 0, test_attacks 0.
- Particiones frente al pipeline original: train_benign: True, val_benign: True, test_benign: True, test_attacks: True. (Copia en `outputs/oe1_saneado/datos_info.json`.)

## 4. OE2 robusto sobre los modelos saneados

Semillas: 42, 43, 44, 45, 46, 47, 48, 49. Dimensiones activas por semilla (benigno de test): 42→7, 43→7, 44→7, 45→7, 46→7, 47→7, 48→7, 49→6.

**AUC-ROC (nll_mc, L = 20) por grupo**, media entre semillas:

| group | mean | std | min | max |
|---|---|---|---|---|
| Bot_attempted | 0.5930 | — | 0.5930 | 0.5930 |
| Bot | 0.6298 | — | 0.6298 | 0.6298 |
| WebAttack_BruteForce_attempted | 0.6603 | — | 0.6603 | 0.6603 |
| PortScan | 0.6856 | — | 0.6856 | 0.6856 |
| DoS_GoldenEye_attempted | 0.7339 | — | 0.7339 | 0.7339 |
| FTP_Patator | 0.8540 | — | 0.8540 | 0.8540 |
| DoS_Hulk_attempted | 0.9163 | — | 0.9163 | 0.9163 |
| SSH_Patator | 0.9331 | — | 0.9331 | 0.9331 |
| DoS_slowloris | 0.9352 | — | 0.9352 | 0.9352 |
| Web_Attack | 0.9434 | — | 0.9434 | 0.9434 |
| DoS_Hulk | 0.9503 | — | 0.9503 | 0.9503 |
| WebAttack_BruteForce | 0.9653 | — | 0.9653 | 0.9653 |
| DoS_Slowhttptest | 0.9686 | — | 0.9686 | 0.9686 |
| DoS_GoldenEye | 0.9700 | — | 0.9700 | 0.9700 |
| DoS_slowloris_attempted | 0.9780 | — | 0.9780 | 0.9780 |
| Infiltration | 0.9835 | — | 0.9835 | 0.9835 |
| DoS_Slowhttptest_attempted | 0.9855 | — | 0.9855 | 0.9855 |
| DDoS | 0.9957 | — | 0.9957 | 0.9957 |

**Frente al modelo de 51 variables** (OE2 robusto, 10 semillas):

| group | auc_51 | auc_66 | dif |
|---|---|---|---|
| Bot | 0.6239 | 0.6298 | 0.0059 |
| Bot_attempted | 0.6134 | 0.5930 | -0.0204 |
| DDoS | 0.9756 | 0.9957 | 0.0200 |
| DoS_GoldenEye | 0.9603 | 0.9700 | 0.0096 |
| DoS_GoldenEye_attempted | 0.8032 | 0.7339 | -0.0692 |
| DoS_Hulk | 0.9161 | 0.9503 | 0.0342 |
| DoS_Hulk_attempted | 0.9384 | 0.9163 | -0.0221 |
| DoS_Slowhttptest | 0.9942 | 0.9686 | -0.0256 |
| DoS_Slowhttptest_attempted | 0.9876 | 0.9855 | -0.0021 |
| DoS_slowloris | 0.9453 | 0.9352 | -0.0101 |
| DoS_slowloris_attempted | 0.9488 | 0.9780 | 0.0292 |
| FTP_Patator | 0.9267 | 0.8540 | -0.0727 |
| Infiltration | 0.9920 | 0.9835 | -0.0085 |
| PortScan | 0.6667 | 0.6856 | 0.0189 |
| SSH_Patator | 0.9578 | 0.9331 | -0.0247 |
| WebAttack_BruteForce | 0.9481 | 0.9653 | 0.0171 |
| WebAttack_BruteForce_attempted | 0.7174 | 0.6603 | -0.0572 |
| Web_Attack | 0.9318 | 0.9434 | 0.0116 |

- τ (p95 de val_benign): FPR realizada en test_benign media 0.0509.
- Conteos por grupo frente al OE2 original: idénticos en todas las semillas (`outputs/oe2_robust_saneado/verificacion_conteos.json`).

## 5. OE2: valoración de descriptores sobre los modelos saneados

No disponible (falta un archivo o la etapa falló):

```
Traceback (most recent call last):
  File "/home/estudiante/vae-latent-geometry-nids/scripts/vm_resumen_corrida.py", line 77, in section
    lines += fn()
  File "/home/estudiante/vae-latent-geometry-nids/scripts/vm_resumen_corrida.py", line 204, in oe2_descriptores
    v = pd.read_csv(DES / "tables" / "valoracion_descriptores.csv")
FileNotFoundError: [Errno 2] No such file or directory: '/home/estudiante/vae-latent-geometry-nids/outputs/oe2_descriptores_saneado/tables/valoracion_descriptores.csv'

```

## 6. OE2: prueba confirmatoria (hipótesis registradas el 10-10-2026)

No disponible (falta un archivo o la etapa falló):

```
Traceback (most recent call last):
  File "/home/estudiante/vae-latent-geometry-nids/scripts/vm_resumen_corrida.py", line 77, in section
    lines += fn()
  File "/home/estudiante/vae-latent-geometry-nids/scripts/vm_resumen_corrida.py", line 231, in confirmatoria
    text = p.read_text(encoding="utf-8").splitlines()
FileNotFoundError: [Errno 2] No such file or directory: '/home/estudiante/vae-latent-geometry-nids/outputs/oe2_descriptores_saneado/RESULTADOS_CONFIRMATORIA.md'

```

## 7. Archivos para el commit

Rutas a versionar (`ARCHIVOS_PARA_COMMIT.txt`, las usa `scripts/vm_preparar_commit.sh`):

| ruta | MB |
|---|---|
| outputs/vm_pendientes | 0.08 |
| outputs/oe1_saneado/checkpoints | 0.34 |
| outputs/oe1_saneado/logs | 0.10 |
| outputs/oe1_saneado/tables | 0.04 |
| outputs/oe1_saneado/figures | 1.51 |
| outputs/oe1_saneado/figures_informe | 0.10 |
| outputs/oe1_saneado/preproceso | 0.01 |
| outputs/oe1_saneado/informe_tablas | 0.45 |
| outputs/oe1_saneado/registro_run.log | 0.00 |
| outputs/oe1_saneado/datos_info.json | 0.00 |
| outputs/oe2_robust_saneado/tables | 80.81 |
| outputs/oe2_robust_saneado/metrics | 0.00 |
| outputs/oe2_robust_saneado/figures | 23.15 |
| outputs/oe2_robust_saneado/verificacion_conteos.json | 0.00 |
| outputs/oe2_descriptores_saneado/tables | 0.00 |
| outputs/oe2_descriptores_saneado/figures | 0.00 |

No se versionan (pesados y regenerables): `outputs/oe2_robust_saneado/latent_vectors/`, `outputs/oe2_robust_saneado/tables/*.npz` y `_cache/`, `outputs/oe2_descriptores_saneado/_cache/`, `data/`.
