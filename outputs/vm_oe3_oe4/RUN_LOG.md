
# Corrida OE3 + OE4 + reconciliación — 2026-10-11T17:49:57Z

- Host: tesis-juan; `uname -a`: Linux tesis-juan 5.15.0-194-generic #204-Ubuntu SMP Wed Sep 2 13:09:23 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux
- CPUs: 8; memoria: 23 GB total, 22 GB disponibles
- Disco libre en el repo: 58G
- Commit: 75a5abd97ce744da02f703ea8a376f1c3d6693ea (rama main)
- Cambios sin commit en archivos versionados: 22

## Entorno

- Python: Python 3.10.12
- Dependencias: versiones exactas de requirements-vm.txt
- torch: 2.13.0+cpu | hilos: 8

## E01 — Tests (OE1 saneado y estadística del OE2)

- Inicio: 2026-10-11T17:52:06Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m pytest tests/oe1_saneado tests/oe2_robust/test_stats_utils.py -q -p no:cacheprovider`
- Fin: 2026-10-11T17:52:20Z (14 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E01.log`. Últimas líneas:

```
===== [2026-10-11T17:52:06Z] E01: Tests (OE1 saneado y estadística del OE2) (intento 1) =====
........................                                                 [100%]
24 passed in 12.64s
```

## E10 — Reconciliación: modelo mixto (fase 5) frente a nivel de flujo (fase 7)

- Inicio: 2026-10-11T17:52:20Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase8_reconciliacion`
- Fin: 2026-10-11T17:52:36Z (16 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E10.log`. Últimas líneas:

```
===== [2026-10-11T17:52:20Z] E10: Reconciliación: modelo mixto (fase 5) frente a nivel de flujo (fase 7) (intento 1) =====
[fase8] 309,248 conexiones, 12 grupos, 8 semillas
/home/estudiante/vae-latent-geometry-nids/src/vae_nids/evaluation/oe2_robust/phase5_correlation.py:249: ConvergenceWarning: The Hessian matrix at the estimated parameter values is not positive definite.
  result = model.fit(reml=True, method="lbfgs", maxiter=200)
[fase8] M2: mixto, valores estandarizados, sin benigno: efecto=0.10592863446845337 p=0.29228387310562576
[fase8] M3: mixto, puntuaciones normales de rango, sin benigno: efecto=0.71762201766437 p=7.972253044286313e-25
[fase8] resumen en /home/estudiante/vae-latent-geometry-nids/outputs/oe2_descriptores_saneado/RESULTADOS_RECONCILIACION.md
```

## E20 — OE3: calidad generativa del decodificador

- Inicio: 2026-10-11T17:52:36Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe3.calidad_generativa`
- Fin: 2026-10-11T17:56:47Z (251 s = 4 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E20.log`. Últimas líneas:

```
===== [2026-10-11T17:52:36Z] E20: OE3: calidad generativa del decodificador (intento 1) =====
[oe3 seed=42] real_val             KS med=0.004 C2ST=0.501 SW2=0.0053
[oe3 seed=42] prior_media          KS med=0.553 C2ST=1.000 SW2=0.0568
[oe3 seed=42] prior_muestra        KS med=0.459 C2ST=1.000 SW2=0.7183
[oe3 seed=42] posterior_agregado   KS med=0.440 C2ST=1.000 SW2=0.0510
[oe3 seed=43] real_val             KS med=0.004 C2ST=0.497 SW2=0.0096
[oe3 seed=43] prior_media          KS med=0.501 C2ST=1.000 SW2=0.0624
[oe3 seed=43] prior_muestra        KS med=0.455 C2ST=1.000 SW2=2.0429
[oe3 seed=43] posterior_agregado   KS med=0.456 C2ST=1.000 SW2=0.0699
[oe3 seed=44] real_val             KS med=0.004 C2ST=0.497 SW2=0.0061
[oe3 seed=44] prior_media          KS med=0.527 C2ST=1.000 SW2=0.0537
[oe3 seed=44] prior_muestra        KS med=0.454 C2ST=1.000 SW2=0.2250
[oe3 seed=44] posterior_agregado   KS med=0.448 C2ST=1.000 SW2=0.0684
[oe3 seed=45] real_val             KS med=0.004 C2ST=0.497 SW2=0.0053
[oe3 seed=45] prior_media          KS med=0.480 C2ST=1.000 SW2=0.0567
[oe3 seed=45] prior_muestra        KS med=0.450 C2ST=1.000 SW2=0.0664
[oe3 seed=45] posterior_agregado   KS med=0.445 C2ST=1.000 SW2=0.0716
[oe3 seed=46] real_val             KS med=0.003 C2ST=0.497 SW2=0.0108
[oe3 seed=46] prior_media          KS med=0.490 C2ST=1.000 SW2=0.0797
[oe3 seed=46] prior_muestra        KS med=0.469 C2ST=1.000 SW2=0.7885
[oe3 seed=46] posterior_agregado   KS med=0.463 C2ST=1.000 SW2=0.0686
[oe3 seed=47] real_val             KS med=0.003 C2ST=0.498 SW2=0.0067
[oe3 seed=47] prior_media          KS med=0.568 C2ST=1.000 SW2=0.0504
[oe3 seed=47] prior_muestra        KS med=0.457 C2ST=1.000 SW2=0.8418
[oe3 seed=47] posterior_agregado   KS med=0.457 C2ST=1.000 SW2=0.0448
[oe3 seed=48] real_val             KS med=0.003 C2ST=0.503 SW2=0.0082
[oe3 seed=48] prior_media          KS med=0.500 C2ST=1.000 SW2=0.0690
[oe3 seed=48] prior_muestra        KS med=0.449 C2ST=1.000 SW2=0.9219
[oe3 seed=48] posterior_agregado   KS med=0.439 C2ST=1.000 SW2=0.0622
[oe3 seed=49] real_val             KS med=0.003 C2ST=0.500 SW2=0.0051
[oe3 seed=49] prior_media          KS med=0.505 C2ST=1.000 SW2=0.0530
[oe3 seed=49] prior_muestra        KS med=0.451 C2ST=1.000 SW2=0.1266
[oe3 seed=49] posterior_agregado   KS med=0.447 C2ST=1.000 SW2=0.0610
[oe3] resumen en /home/estudiante/vae-latent-geometry-nids/outputs/oe3/RESULTADOS_OE3.md
```

## E3_beta0p5_1 — OE4 beta0p5: entrenamiento (5 semillas) y diagnósticos

- Inicio: 2026-10-11T17:56:47Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.oe1.run --config configs/oe4_beta0p5.yaml`
- Fin: 2026-10-11T19:11:04Z (4457 s = 74 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E3_beta0p5_1.log`. Últimas líneas:

```
===== [2026-10-11T17:56:47Z] E3_beta0p5_1: OE4 beta0p5: entrenamiento (5 semillas) y diagnósticos (intento 1) =====
[2026-10-11T17:56:51] preproceso
[2026-10-11T17:57:25] oe4_beta0p5_seed42: entrenando
[2026-10-11T18:18:21] oe4_beta0p5_seed42: parada en época 150 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 140, val -249.6861, 1250 s
[2026-10-11T18:18:21] oe4_beta0p5_seed43: entrenando
[2026-10-11T18:29:12] oe4_beta0p5_seed43: parada en época 78 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 68, val -247.8179, 645 s
[2026-10-11T18:29:12] oe4_beta0p5_seed44: entrenando
[2026-10-11T18:42:42] oe4_beta0p5_seed44: parada en época 96 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 86, val -248.4324, 804 s
[2026-10-11T18:42:42] oe4_beta0p5_seed45: entrenando
[2026-10-11T18:59:22] oe4_beta0p5_seed45: parada en época 118 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 108, val -252.0603, 995 s
[2026-10-11T18:59:22] oe4_beta0p5_seed46: entrenando
[2026-10-11T19:10:54] oe4_beta0p5_seed46: parada en época 82 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 72, val -247.6524, 687 s
[2026-10-11T19:10:54] diagnósticos
[2026-10-11T19:11:03] listo: {"corridas": 5}
```

## E3_beta0p5_2 — OE4 beta0p5: codificación (mu, log sigma^2)

- Inicio: 2026-10-11T19:11:04Z
- Comando: `env VAE_NIDS_OE2_CONFIG=configs/oe4_beta0p5.yaml /home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase2_encode`
- Fin: 2026-10-11T19:11:12Z (8 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E3_beta0p5_2.log`. Últimas líneas:

```
===== [2026-10-11T19:11:04Z] E3_beta0p5_2: OE4 beta0p5: codificación (mu, log sigma^2) (intento 1) =====
[fase2 seed=42] BENIGN test: n=248,561
[fase2 seed=42] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=42] dims activas (8/8): [0, 1, 2, 3, 4, 5, 6, 7] varianza=[0.7187, 0.95775, 0.57448, 0.3673, 0.46057, 0.77048, 0.57562, 0.93585]
[fase2 seed=43] BENIGN test: n=248,561
[fase2 seed=43] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=43] dims activas (8/8): [0, 1, 2, 3, 4, 5, 6, 7] varianza=[0.44532, 0.5729, 0.79456, 0.59998, 0.97088, 0.56979, 0.86572, 0.63339]
[fase2 seed=44] BENIGN test: n=248,561
[fase2 seed=44] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=44] dims activas (8/8): [0, 1, 2, 3, 4, 5, 6, 7] varianza=[0.62187, 0.32831, 0.83233, 0.40036, 0.80228, 0.67525, 1.00808, 0.95117]
[fase2 seed=45] BENIGN test: n=248,561
[fase2 seed=45] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=45] dims activas (8/8): [0, 1, 2, 3, 4, 5, 6, 7] varianza=[0.93613, 0.27264, 0.93741, 0.67649, 0.35059, 0.67525, 1.03043, 0.30019]
[fase2 seed=46] BENIGN test: n=248,561
[fase2 seed=46] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=46] dims activas (8/8): [0, 1, 2, 3, 4, 5, 6, 7] varianza=[0.92424, 0.44623, 0.46973, 0.648, 0.91633, 0.60112, 0.99119, 0.60644]
[fase2] tabla semilla x dimensión guardada en /home/estudiante/vae-latent-geometry-nids/outputs/oe4/beta0p5/oe2_robust/tables/phase2_active_dims_per_seed.csv
```

## E3_beta0p5_3 — OE4 beta0p5: detectabilidad y referencia benigna

- Inicio: 2026-10-11T19:11:12Z
- Comando: `env VAE_NIDS_OE2_CONFIG=configs/oe4_beta0p5.yaml /home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase1_detectabilidad`
- Fin: 2026-10-11T19:13:36Z (144 s = 2 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E3_beta0p5_3.log`. Últimas líneas:

```
[fase1 seed=44] Bot_attempted                    n=   1470 AUC=0.6201
[fase1 seed=44] WebAttack_BruteForce             n=    151 AUC=0.9651
[fase1 seed=44] WebAttack_BruteForce_attempted   n=   1214 AUC=0.7113
[fase1 seed=45] PortScan                         n= 159023 AUC=0.7523
[fase1 seed=45] DoS_Hulk                         n= 158469 AUC=0.9745
[fase1 seed=45] DDoS                             n=  95123 AUC=0.9963
[fase1 seed=45] DoS_GoldenEye                    n=   7567 AUC=0.9903
[fase1 seed=45] DoS_slowloris                    n=   4001 AUC=0.9559
[fase1 seed=45] FTP_Patator                      n=   3973 AUC=0.8821
[fase1 seed=45] SSH_Patator                      n=   2980 AUC=0.9303
[fase1 seed=45] DoS_Slowhttptest                 n=   1742 AUC=0.9754
[fase1 seed=45] Bot                              n=    738 AUC=0.6046
[fase1 seed=45] Web_Attack                       n=    190 AUC=0.9549
[fase1 seed=45] Infiltration                     n=     32 AUC=0.9819
[fase1 seed=45] DoS_Hulk_attempted               n=    579 AUC=0.9456
[fase1 seed=45] DoS_GoldenEye_attempted          n=     80 AUC=0.8193
[fase1 seed=45] DoS_slowloris_attempted          n=   1706 AUC=0.9839
[fase1 seed=45] DoS_Slowhttptest_attempted       n=   3367 AUC=0.9927
[fase1 seed=45] Bot_attempted                    n=   1470 AUC=0.6765
[fase1 seed=45] WebAttack_BruteForce             n=    151 AUC=0.9669
[fase1 seed=45] WebAttack_BruteForce_attempted   n=   1214 AUC=0.7592
[fase1 seed=46] PortScan                         n= 159023 AUC=0.6648
[fase1 seed=46] DoS_Hulk                         n= 158469 AUC=0.9701
[fase1 seed=46] DDoS                             n=  95123 AUC=0.9966
[fase1 seed=46] DoS_GoldenEye                    n=   7567 AUC=0.9796
[fase1 seed=46] DoS_slowloris                    n=   4001 AUC=0.9322
[fase1 seed=46] FTP_Patator                      n=   3973 AUC=0.8964
[fase1 seed=46] SSH_Patator                      n=   2980 AUC=0.9229
[fase1 seed=46] DoS_Slowhttptest                 n=   1742 AUC=0.9632
[fase1 seed=46] Bot                              n=    738 AUC=0.5990
[fase1 seed=46] Web_Attack                       n=    190 AUC=0.9328
[fase1 seed=46] Infiltration                     n=     32 AUC=0.9800
[fase1 seed=46] DoS_Hulk_attempted               n=    579 AUC=0.9206
[fase1 seed=46] DoS_GoldenEye_attempted          n=     80 AUC=0.7664
[fase1 seed=46] DoS_slowloris_attempted          n=   1706 AUC=0.9639
[fase1 seed=46] DoS_Slowhttptest_attempted       n=   3367 AUC=0.9882
[fase1 seed=46] Bot_attempted                    n=   1470 AUC=0.5823
[fase1 seed=46] WebAttack_BruteForce             n=    151 AUC=0.9462
[fase1 seed=46] WebAttack_BruteForce_attempted   n=   1214 AUC=0.7047
[fase1] diferencia máxima de AUC vs phase4 (seed 42): nan
```

## E3_beta0p5_4 — OE4 beta0p5: descriptores por familia y semilla

- Inicio: 2026-10-11T19:13:36Z
- Comando: `env VAE_NIDS_OE2_CONFIG=configs/oe4_beta0p5.yaml /home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase2_descriptores`
- Fin: 2026-10-11T19:51:16Z (2260 s = 37 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E3_beta0p5_4.log`. Últimas líneas:

```
[fase2 seed=44] WebAttack_BruteForce             n=    151 únicos=    151 AUC_knn=0.952 sil_w=0.392  (425s)
[fase2 seed=44] WebAttack_BruteForce_attempted   n=   1214 únicos=   1214 AUC_knn=0.480 sil_w=0.971  (440s)
[fase2 seed=45] BENIGN_control                   n=  50000 únicos=  47670 AUC_knn=0.500 sil_w=0.016  (57s)
[fase2 seed=45] PortScan                         n= 159023 únicos=   1670 AUC_knn=0.813 sil_w=0.196  (77s)
[fase2 seed=45] DoS_Hulk                         n= 158469 únicos= 158469 AUC_knn=0.972 sil_w=0.624  (175s)
[fase2 seed=45] DDoS                             n=  95123 únicos=  95123 AUC_knn=0.997 sil_w=0.506  (246s)
[fase2 seed=45] DoS_GoldenEye                    n=   7567 únicos=   7567 AUC_knn=0.985 sil_w=0.130  (266s)
[fase2 seed=45] DoS_slowloris                    n=   4001 únicos=   3595 AUC_knn=0.958 sil_w=0.247  (284s)
[fase2 seed=45] FTP_Patator                      n=   3973 únicos=   3973 AUC_knn=0.920 sil_w=0.965  (302s)
[fase2 seed=45] SSH_Patator                      n=   2980 únicos=   2980 AUC_knn=0.893 sil_w=0.939  (321s)
[fase2 seed=45] DoS_Slowhttptest                 n=   1742 únicos=   1731 AUC_knn=0.977 sil_w=0.118  (339s)
[fase2 seed=45] Bot                              n=    738 únicos=    738 AUC_knn=0.806 sil_w=0.659  (351s)
[fase2 seed=45] Web_Attack                       n=    190 únicos=    190 AUC_knn=0.933 sil_w=0.329  (359s)
[fase2 seed=45] Infiltration                     n=     32 únicos=     32 AUC_knn=0.982 sil_w=-0.565  (365s)
[fase2 seed=45] DoS_Hulk_attempted               n=    579 únicos=    573 AUC_knn=0.961 sil_w=0.250  (375s)
[fase2 seed=45] DoS_GoldenEye_attempted          n=     80 únicos=     77 AUC_knn=0.893 sil_w=0.926  (381s)
[fase2 seed=45] DoS_slowloris_attempted          n=   1706 únicos=   1703 AUC_knn=0.967 sil_w=0.913  (398s)
[fase2 seed=45] DoS_Slowhttptest_attempted       n=   3367 únicos=   3361 AUC_knn=0.664 sil_w=0.616  (417s)
[fase2 seed=45] Bot_attempted                    n=   1470 únicos=    713 AUC_knn=0.184 sil_w=0.961  (429s)
[fase2 seed=45] WebAttack_BruteForce             n=    151 únicos=    151 AUC_knn=0.935 sil_w=0.501  (436s)
[fase2 seed=45] WebAttack_BruteForce_attempted   n=   1214 únicos=   1214 AUC_knn=0.452 sil_w=0.964  (451s)
[fase2 seed=46] BENIGN_control                   n=  50000 únicos=  47725 AUC_knn=0.500 sil_w=-0.004  (49s)
[fase2 seed=46] PortScan                         n= 159023 únicos=   1670 AUC_knn=0.720 sil_w=0.329  (70s)
[fase2 seed=46] DoS_Hulk                         n= 158469 únicos= 158469 AUC_knn=0.995 sil_w=0.603  (169s)
[fase2 seed=46] DDoS                             n=  95123 únicos=  95123 AUC_knn=0.997 sil_w=0.473  (239s)
[fase2 seed=46] DoS_GoldenEye                    n=   7567 únicos=   7567 AUC_knn=0.992 sil_w=0.236  (260s)
[fase2 seed=46] DoS_slowloris                    n=   4001 únicos=   3595 AUC_knn=0.973 sil_w=0.182  (279s)
[fase2 seed=46] FTP_Patator                      n=   3973 únicos=   3973 AUC_knn=0.959 sil_w=0.953  (299s)
[fase2 seed=46] SSH_Patator                      n=   2980 únicos=   2980 AUC_knn=0.939 sil_w=0.944  (317s)
[fase2 seed=46] DoS_Slowhttptest                 n=   1742 únicos=   1731 AUC_knn=0.982 sil_w=0.098  (336s)
[fase2 seed=46] Bot                              n=    738 únicos=    738 AUC_knn=0.779 sil_w=0.760  (349s)
[fase2 seed=46] Web_Attack                       n=    190 únicos=    190 AUC_knn=0.942 sil_w=0.414  (356s)
[fase2 seed=46] Infiltration                     n=     32 únicos=     32 AUC_knn=0.988 sil_w=-0.678  (363s)
[fase2 seed=46] DoS_Hulk_attempted               n=    579 únicos=    573 AUC_knn=0.900 sil_w=0.235  (374s)
[fase2 seed=46] DoS_GoldenEye_attempted          n=     80 únicos=     77 AUC_knn=0.687 sil_w=0.945  (380s)
[fase2 seed=46] DoS_slowloris_attempted          n=   1706 únicos=   1703 AUC_knn=0.959 sil_w=0.922  (398s)
[fase2 seed=46] DoS_Slowhttptest_attempted       n=   3367 únicos=   3361 AUC_knn=0.650 sil_w=0.656  (417s)
[fase2 seed=46] Bot_attempted                    n=   1470 únicos=    713 AUC_knn=0.112 sil_w=0.987  (428s)
[fase2 seed=46] WebAttack_BruteForce             n=    151 únicos=    151 AUC_knn=0.945 sil_w=0.582  (435s)
[fase2 seed=46] WebAttack_BruteForce_attempted   n=   1214 únicos=   1214 AUC_knn=0.443 sil_w=0.940  (451s)
```

## E3_beta0p5_5 — OE4 beta0p5: cribado de descriptores

- Inicio: 2026-10-11T19:51:16Z
- Comando: `env VAE_NIDS_OE2_CONFIG=configs/oe4_beta0p5.yaml /home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase3_evaluacion`
- Fin: 2026-10-11T19:51:49Z (33 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E3_beta0p5_5.log`. Últimas líneas:

```
===== [2026-10-11T19:51:16Z] E3_beta0p5_5: OE4 beta0p5: cribado de descriptores (intento 1) =====
/home/estudiante/vae-latent-geometry-nids/src/vae_nids/evaluation/oe2_descriptores/fase3_evaluacion.py:77: RuntimeWarning: Mean of empty slice
  return float(np.tanh(np.nanmean(np.arctanh(r))))
1.1      Mahalanobis (mediana)                            rho=+0.91 [+0.68,+1.00] p=0.000 W=0.95 rhoM=+1.00 val=+0.88
5.1      kNN al benigno (z blanqueado)                    rho=+0.95 [+0.86,+0.99] p=0.000 W=0.94 rhoM=+0.88 val=+0.62
2.4      Fisher-Rao al k-ésimo posterior benigno          rho=+0.86 [+0.59,+0.98] p=0.000 W=0.93 rhoM=+0.73 val=+0.78
7.1      Tasa KL(q||p) por flujo                          rho=+0.94 [+0.76,+1.00] p=0.000 W=0.98 rhoM=+0.93 val=+0.91
3.2      MMD² analítico entre posteriores agregados       rho=+0.66 [+0.13,+0.93] p=0.023 W=0.92 rhoM=+0.75 val=+0.88
6.2      kNN en el espacio decodificado                   rho=+0.86 [+0.59,+0.98] p=0.000 W=0.91 rhoM=+0.77 val=+0.74
6.1      Log-magnificación del decodificador              rho=+0.25 [-0.53,+0.79] p=0.384 W=0.72 rhoM=-0.01 val=+0.81
4.2      Distancia afín-invariante entre covarianzas      rho=-0.08 [-0.68,+0.50] p=0.800 W=0.95 rhoM=-0.27 val=-0.49
8.1      LID relativa al benigno                          rho=+0.29 [-0.63,+0.84] p=0.267 W=0.60 rhoM=+0.21 val=+0.42
10.1a    Silhouette del grupo (z blanqueado)              rho=-0.47 [-0.86,+0.16] p=0.137 W=0.97 rhoM=-0.47 val=-0.80
10.3     Sonda lineal sobre (μ, log σ²)                   rho=+0.19 [-0.55,+0.76] p=0.555 W=0.95 rhoM=+0.28 val=-0.43
1.1b     Mahalanobis (AUC frente al control)              rho=+0.90 [+0.68,+0.99] p=0.000 W=0.98 rhoM=+0.96 val=+0.96
10.1b    Silhouette del grupo (Fisher-Rao)                rho=-0.75 [-0.94,-0.37] p=0.008 W=0.99 rhoM=-0.70 val=-0.60
1.2      Log-densidad de mezcla benigna (K=8)             rho=+0.90 [+0.68,+0.99] p=0.000 W=0.93 rhoM=+0.87 val=+0.96
1.3      Mahalanobis relativa al prior                    rho=+0.74 [+0.47,+0.91] p=0.002 W=0.80 rhoM=+0.81 val=+0.95
1.4      Tipicidad radial bilateral                       rho=+0.90 [+0.66,+0.99] p=0.000 W=0.98 rhoM=+0.96 val=+0.96
4.1      Rango efectivo de la familia                     rho=-0.19 [-0.76,+0.51] p=0.547 W=0.85 rhoM=-0.16 val=+0.17
7.2      Log-volumen del posterior                        rho=+0.34 [-0.42,+0.80] p=0.217 W=0.70 rhoM=+0.12 val=+0.86
11.1     Inconsistencia Kumar-Poole                       rho=+0.42 [+0.03,+0.72] p=0.084 W=0.53 rhoM=+0.54 val=+0.73
5.1m     kNN: fracción > p95 del control                  rho=+0.79 [+0.58,+0.93] p=0.000 W=0.63 rhoM=+0.68 val=+0.76
2.4m     Fisher-Rao (mediana)                             rho=+0.83 [+0.52,+0.97] p=0.000 W=0.90 rhoM=+0.76 val=+0.76
7.1m     Tasa (mediana, nats)                             rho=+0.92 [+0.67,+1.00] p=0.000 W=0.95 rhoM=+0.97 val=+0.87
6.1m     Log-magnificación (mediana)                      rho=+0.30 [-0.45,+0.80] p=0.288 W=0.75 rhoM=+0.04 val=+0.80
```

## E3_beta0p5_6 — OE4 beta0p5: prueba confirmatoria

- Inicio: 2026-10-11T19:51:49Z
- Comando: `env VAE_NIDS_OE2_CONFIG=configs/oe4_beta0p5.yaml /home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase7_confirmatoria`
- Fin: 2026-10-11T19:52:00Z (11 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E3_beta0p5_6.log`. Últimas líneas:

```
===== [2026-10-11T19:51:49Z] E3_beta0p5_6: OE4 beta0p5: prueba confirmatoria (intento 1) =====
[fase7] 1.1    Mahalanobis (mediana)                      rho=+0.913 [+0.667,+0.997] p=0.0002 (51 variables: +0.604)
[fase7] 2.4    Fisher-Rao al 10.º posterior benigno       rho=+0.859 [+0.556,+0.982] p=0.0006 (51 variables: +0.724)
[fase7] 6.2    kNN en el espacio decodificado             rho=+0.864 [+0.561,+0.989] p=0.0006 (51 variables: +0.691)
[fase7] 7.1    Tasa KL(q||p) por flujo                    rho=+0.941 [+0.743,+0.999] p=0.0001 (51 variables: +0.681)
[fase7] 5.1    kNN al benigno (z blanqueado)              rho=+0.948 [+0.794,+0.996] p=0.0002 (51 variables: +0.694)
[fase7] 6.1    Log-magnificación del decodificador        rho=+0.251 [-0.571,+0.847] p=0.3823 (51 variables: +0.523)
[fase7] 10.1b  Silhouette del grupo (Fisher-Rao)          rho=-0.753 [-0.930,-0.365] p=0.0087 (51 variables: -0.629)
[fase7] resumen en /home/estudiante/vae-latent-geometry-nids/outputs/oe4/beta0p5/oe2_descriptores/RESULTADOS_CONFIRMATORIA.md
```

## E3_beta2_1 — OE4 beta2: entrenamiento (5 semillas) y diagnósticos

- Inicio: 2026-10-11T19:52:00Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.oe1.run --config configs/oe4_beta2.yaml`
- Fin: 2026-10-11T21:08:12Z (4572 s = 76 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E3_beta2_1.log`. Últimas líneas:

```
===== [2026-10-11T19:52:00Z] E3_beta2_1: OE4 beta2: entrenamiento (5 semillas) y diagnósticos (intento 1) =====
[2026-10-11T19:52:04] preproceso
[2026-10-11T19:52:37] oe4_beta2_seed42: entrenando
[2026-10-11T20:07:36] oe4_beta2_seed42: parada en época 105 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 95, val -240.8496, 892 s
[2026-10-11T20:07:36] oe4_beta2_seed43: entrenando
[2026-10-11T20:24:19] oe4_beta2_seed43: parada en época 118 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 108, val -241.7369, 998 s
[2026-10-11T20:24:19] oe4_beta2_seed44: entrenando
[2026-10-11T20:35:05] oe4_beta2_seed44: parada en época 77 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 67, val -240.8967, 641 s
[2026-10-11T20:35:05] oe4_beta2_seed45: entrenando
[2026-10-11T20:48:22] oe4_beta2_seed45: parada en época 94 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 84, val -241.3959, 792 s
[2026-10-11T20:48:22] oe4_beta2_seed46: entrenando
[2026-10-11T21:08:01] oe4_beta2_seed46: parada en época 139 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 129, val -241.7758, 1174 s
[2026-10-11T21:08:01] diagnósticos
[2026-10-11T21:08:11] listo: {"corridas": 5}
```

## E3_beta2_2 — OE4 beta2: codificación (mu, log sigma^2)

- Inicio: 2026-10-11T21:08:12Z
- Comando: `env VAE_NIDS_OE2_CONFIG=configs/oe4_beta2.yaml /home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase2_encode`
- Fin: 2026-10-11T21:08:20Z (8 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E3_beta2_2.log`. Últimas líneas:

```
===== [2026-10-11T21:08:12Z] E3_beta2_2: OE4 beta2: codificación (mu, log sigma^2) (intento 1) =====
[fase2 seed=42] BENIGN test: n=248,561
[fase2 seed=42] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=42] dims activas (5/8): [0, 1, 2, 5, 6] varianza=[0.66916, 0.18312, 0.42444, 1e-05, 3e-05, 1.01327, 0.4504, 3e-05]
[fase2 seed=43] BENIGN test: n=248,561
[fase2 seed=43] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=43] dims activas (5/8): [0, 2, 3, 5, 7] varianza=[1.05661, 0.0, 0.32174, 0.31566, 1e-05, 0.75915, 0.0, 0.39142]
[fase2 seed=44] BENIGN test: n=248,561
[fase2 seed=44] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=44] dims activas (6/8): [0, 2, 3, 4, 5, 7] varianza=[0.19046, 1e-05, 0.62817, 0.98937, 0.64448, 0.1533, 1e-05, 0.3563]
[fase2 seed=45] BENIGN test: n=248,561
[fase2 seed=45] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=45] dims activas (4/8): [2, 3, 5, 6] varianza=[1e-05, 2e-05, 0.33422, 0.9835, 1e-05, 0.63425, 0.735, 3e-05]
[fase2 seed=46] BENIGN test: n=248,561
[fase2 seed=46] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=46] dims activas (6/8): [0, 2, 3, 4, 5, 6] varianza=[0.18185, 2e-05, 0.30572, 1.04326, 0.11161, 0.49481, 0.71933, 2e-05]
[fase2] tabla semilla x dimensión guardada en /home/estudiante/vae-latent-geometry-nids/outputs/oe4/beta2/oe2_robust/tables/phase2_active_dims_per_seed.csv
```

## E3_beta2_3 — OE4 beta2: detectabilidad y referencia benigna

- Inicio: 2026-10-11T21:08:20Z
- Comando: `env VAE_NIDS_OE2_CONFIG=configs/oe4_beta2.yaml /home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase1_detectabilidad`
- Fin: 2026-10-11T21:10:44Z (144 s = 2 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E3_beta2_3.log`. Últimas líneas:

```
[fase1 seed=44] Bot_attempted                    n=   1470 AUC=0.5754
[fase1 seed=44] WebAttack_BruteForce             n=    151 AUC=0.9185
[fase1 seed=44] WebAttack_BruteForce_attempted   n=   1214 AUC=0.6883
[fase1 seed=45] PortScan                         n= 159023 AUC=0.6682
[fase1 seed=45] DoS_Hulk                         n= 158469 AUC=0.9004
[fase1 seed=45] DDoS                             n=  95123 AUC=0.9874
[fase1 seed=45] DoS_GoldenEye                    n=   7567 AUC=0.9738
[fase1 seed=45] DoS_slowloris                    n=   4001 AUC=0.9344
[fase1 seed=45] FTP_Patator                      n=   3973 AUC=0.9145
[fase1 seed=45] SSH_Patator                      n=   2980 AUC=0.9742
[fase1 seed=45] DoS_Slowhttptest                 n=   1742 AUC=0.9903
[fase1 seed=45] Bot                              n=    738 AUC=0.5937
[fase1 seed=45] Web_Attack                       n=    190 AUC=0.9253
[fase1 seed=45] Infiltration                     n=     32 AUC=0.9891
[fase1 seed=45] DoS_Hulk_attempted               n=    579 AUC=0.9025
[fase1 seed=45] DoS_GoldenEye_attempted          n=     80 AUC=0.6708
[fase1 seed=45] DoS_slowloris_attempted          n=   1706 AUC=0.9048
[fase1 seed=45] DoS_Slowhttptest_attempted       n=   3367 AUC=0.9802
[fase1 seed=45] Bot_attempted                    n=   1470 AUC=0.5687
[fase1 seed=45] WebAttack_BruteForce             n=    151 AUC=0.9417
[fase1 seed=45] WebAttack_BruteForce_attempted   n=   1214 AUC=0.6686
[fase1 seed=46] PortScan                         n= 159023 AUC=0.6425
[fase1 seed=46] DoS_Hulk                         n= 158469 AUC=0.8839
[fase1 seed=46] DDoS                             n=  95123 AUC=0.9648
[fase1 seed=46] DoS_GoldenEye                    n=   7567 AUC=0.9576
[fase1 seed=46] DoS_slowloris                    n=   4001 AUC=0.9362
[fase1 seed=46] FTP_Patator                      n=   3973 AUC=0.8907
[fase1 seed=46] SSH_Patator                      n=   2980 AUC=0.9207
[fase1 seed=46] DoS_Slowhttptest                 n=   1742 AUC=0.9909
[fase1 seed=46] Bot                              n=    738 AUC=0.5927
[fase1 seed=46] Web_Attack                       n=    190 AUC=0.9183
[fase1 seed=46] Infiltration                     n=     32 AUC=0.9905
[fase1 seed=46] DoS_Hulk_attempted               n=    579 AUC=0.9050
[fase1 seed=46] DoS_GoldenEye_attempted          n=     80 AUC=0.6761
[fase1 seed=46] DoS_slowloris_attempted          n=   1706 AUC=0.9078
[fase1 seed=46] DoS_Slowhttptest_attempted       n=   3367 AUC=0.9808
[fase1 seed=46] Bot_attempted                    n=   1470 AUC=0.5726
[fase1 seed=46] WebAttack_BruteForce             n=    151 AUC=0.9316
[fase1 seed=46] WebAttack_BruteForce_attempted   n=   1214 AUC=0.6829
[fase1] diferencia máxima de AUC vs phase4 (seed 42): nan
```

## E3_beta2_4 — OE4 beta2: descriptores por familia y semilla

- Inicio: 2026-10-11T21:10:44Z
- Comando: `env VAE_NIDS_OE2_CONFIG=configs/oe4_beta2.yaml /home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase2_descriptores`
- Fin: 2026-10-11T21:45:17Z (2073 s = 34 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E3_beta2_4.log`. Últimas líneas:

```
[fase2 seed=44] WebAttack_BruteForce             n=    151 únicos=    151 AUC_knn=0.954 sil_w=0.475  (402s)
[fase2 seed=44] WebAttack_BruteForce_attempted   n=   1214 únicos=   1214 AUC_knn=0.612 sil_w=0.950  (418s)
[fase2 seed=45] BENIGN_control                   n=  50000 únicos=  47725 AUC_knn=0.500 sil_w=0.001  (41s)
[fase2 seed=45] PortScan                         n= 159023 únicos=   1670 AUC_knn=0.788 sil_w=0.204  (60s)
[fase2 seed=45] DoS_Hulk                         n= 158469 únicos= 158469 AUC_knn=0.989 sil_w=0.707  (144s)
[fase2 seed=45] DDoS                             n=  95123 únicos=  95123 AUC_knn=0.989 sil_w=0.537  (201s)
[fase2 seed=45] DoS_GoldenEye                    n=   7567 únicos=   7567 AUC_knn=0.983 sil_w=0.102  (221s)
[fase2 seed=45] DoS_slowloris                    n=   4001 únicos=   3596 AUC_knn=0.956 sil_w=0.522  (240s)
[fase2 seed=45] FTP_Patator                      n=   3973 únicos=   3973 AUC_knn=0.977 sil_w=0.985  (258s)
[fase2 seed=45] SSH_Patator                      n=   2980 únicos=   2980 AUC_knn=0.959 sil_w=0.953  (275s)
[fase2 seed=45] DoS_Slowhttptest                 n=   1742 únicos=   1731 AUC_knn=0.969 sil_w=0.357  (294s)
[fase2 seed=45] Bot                              n=    738 únicos=    738 AUC_knn=0.846 sil_w=0.687  (305s)
[fase2 seed=45] Web_Attack                       n=    190 únicos=    190 AUC_knn=0.975 sil_w=0.375  (313s)
[fase2 seed=45] Infiltration                     n=     32 únicos=     32 AUC_knn=0.982 sil_w=-0.647  (319s)
[fase2 seed=45] DoS_Hulk_attempted               n=    579 únicos=    573 AUC_knn=0.921 sil_w=0.250  (329s)
[fase2 seed=45] DoS_GoldenEye_attempted          n=     80 únicos=     77 AUC_knn=0.762 sil_w=0.907  (335s)
[fase2 seed=45] DoS_slowloris_attempted          n=   1706 únicos=   1703 AUC_knn=0.944 sil_w=0.888  (352s)
[fase2 seed=45] DoS_Slowhttptest_attempted       n=   3367 únicos=   3361 AUC_knn=0.564 sil_w=0.541  (369s)
[fase2 seed=45] Bot_attempted                    n=   1470 únicos=    713 AUC_knn=0.111 sil_w=0.986  (379s)
[fase2 seed=45] WebAttack_BruteForce             n=    151 únicos=    151 AUC_knn=0.977 sil_w=0.492  (386s)
[fase2 seed=45] WebAttack_BruteForce_attempted   n=   1214 únicos=   1214 AUC_knn=0.552 sil_w=0.960  (402s)
[fase2 seed=46] BENIGN_control                   n=  50000 únicos=  47725 AUC_knn=0.500 sil_w=-0.004  (47s)
[fase2 seed=46] PortScan                         n= 159023 únicos=   1670 AUC_knn=0.736 sil_w=0.186  (68s)
[fase2 seed=46] DoS_Hulk                         n= 158469 únicos= 158469 AUC_knn=0.959 sil_w=0.662  (157s)
[fase2 seed=46] DDoS                             n=  95123 únicos=  95123 AUC_knn=0.975 sil_w=0.588  (215s)
[fase2 seed=46] DoS_GoldenEye                    n=   7567 únicos=   7567 AUC_knn=0.975 sil_w=0.131  (238s)
[fase2 seed=46] DoS_slowloris                    n=   4001 únicos=   3595 AUC_knn=0.954 sil_w=0.099  (258s)
[fase2 seed=46] FTP_Patator                      n=   3973 únicos=   3973 AUC_knn=0.955 sil_w=0.965  (277s)
[fase2 seed=46] SSH_Patator                      n=   2980 únicos=   2980 AUC_knn=0.962 sil_w=0.933  (295s)
[fase2 seed=46] DoS_Slowhttptest                 n=   1742 únicos=   1731 AUC_knn=0.981 sil_w=0.081  (315s)
[fase2 seed=46] Bot                              n=    738 únicos=    738 AUC_knn=0.730 sil_w=0.651  (327s)
[fase2 seed=46] Web_Attack                       n=    190 únicos=    190 AUC_knn=0.969 sil_w=0.321  (334s)
[fase2 seed=46] Infiltration                     n=     32 únicos=     32 AUC_knn=0.981 sil_w=-0.664  (340s)
[fase2 seed=46] DoS_Hulk_attempted               n=    579 únicos=    573 AUC_knn=0.926 sil_w=0.172  (350s)
[fase2 seed=46] DoS_GoldenEye_attempted          n=     80 únicos=     77 AUC_knn=0.783 sil_w=0.893  (357s)
[fase2 seed=46] DoS_slowloris_attempted          n=   1706 únicos=   1703 AUC_knn=0.968 sil_w=0.907  (376s)
[fase2 seed=46] DoS_Slowhttptest_attempted       n=   3367 únicos=   3361 AUC_knn=0.881 sil_w=0.623  (395s)
[fase2 seed=46] Bot_attempted                    n=   1470 únicos=    713 AUC_knn=0.159 sil_w=0.983  (407s)
[fase2 seed=46] WebAttack_BruteForce             n=    151 únicos=    151 AUC_knn=0.978 sil_w=0.454  (415s)
[fase2 seed=46] WebAttack_BruteForce_attempted   n=   1214 únicos=   1214 AUC_knn=0.591 sil_w=0.931  (431s)
```

## E3_beta2_5 — OE4 beta2: cribado de descriptores

- Inicio: 2026-10-11T21:45:17Z
- Comando: `env VAE_NIDS_OE2_CONFIG=configs/oe4_beta2.yaml /home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase3_evaluacion`
- Fin: 2026-10-11T21:45:47Z (30 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E3_beta2_5.log`. Últimas líneas:

```
===== [2026-10-11T21:45:17Z] E3_beta2_5: OE4 beta2: cribado de descriptores (intento 1) =====
/home/estudiante/vae-latent-geometry-nids/src/vae_nids/evaluation/oe2_descriptores/fase3_evaluacion.py:77: RuntimeWarning: Mean of empty slice
  return float(np.tanh(np.nanmean(np.arctanh(r))))
1.1      Mahalanobis (mediana)                            rho=+0.60 [+0.06,+0.97] p=0.030 W=0.81 rhoM=+1.00 val=+0.85
5.1      kNN al benigno (z blanqueado)                    rho=+0.71 [+0.23,+0.91] p=0.007 W=0.82 rhoM=+0.80 val=+0.57
2.4      Fisher-Rao al k-ésimo posterior benigno          rho=+0.72 [+0.33,+0.88] p=0.005 W=0.84 rhoM=+0.74 val=+0.61
7.1      Tasa KL(q||p) por flujo                          rho=+0.66 [+0.05,+0.93] p=0.021 W=0.96 rhoM=+0.86 val=+0.89
3.2      MMD² analítico entre posteriores agregados       rho=+0.46 [-0.33,+0.93] p=0.138 W=0.96 rhoM=+0.80 val=+0.92
6.2      kNN en el espacio decodificado                   rho=+0.64 [+0.20,+0.93] p=0.022 W=0.84 rhoM=+0.40 val=+0.56
6.1      Log-magnificación del decodificador              rho=+0.55 [-0.13,+0.91] p=0.059 W=0.88 rhoM=+0.26 val=+0.95
4.2      Distancia afín-invariante entre covarianzas      rho=-0.19 [-0.73,+0.45] p=0.543 W=0.91 rhoM=-0.21 val=-0.49
8.1      LID relativa al benigno                          rho=+0.06 [-0.53,+0.70] p=0.834 W=0.66 rhoM=+0.11 val=+0.31
10.1a    Silhouette del grupo (z blanqueado)              rho=-0.40 [-0.80,+0.22] p=0.201 W=0.94 rhoM=-0.38 val=-0.83
10.3     Sonda lineal sobre (μ, log σ²)                   rho=-0.12 [-0.68,+0.54] p=0.705 W=0.92 rhoM=-0.00 val=-0.45
1.1b     Mahalanobis (AUC frente al control)              rho=+0.66 [+0.07,+0.98] p=0.022 W=0.91 rhoM=+0.92 val=+0.94
10.1b    Silhouette del grupo (Fisher-Rao)                rho=-0.62 [-0.92,-0.09] p=0.037 W=0.98 rhoM=-0.70 val=-0.76
1.2      Log-densidad de mezcla benigna (K=8)             rho=+0.55 [-0.05,+0.92] p=0.062 W=0.89 rhoM=+0.82 val=+0.84
1.3      Mahalanobis relativa al prior                    rho=+0.65 [+0.11,+0.94] p=0.019 W=0.83 rhoM=+0.96 val=+0.96
1.4      Tipicidad radial bilateral                       rho=+0.63 [+0.02,+0.96] p=0.030 W=0.89 rhoM=+0.92 val=+0.93
4.1      Rango efectivo de la familia                     rho=+0.16 [-0.45,+0.78] p=0.537 W=0.54 rhoM=+0.02 val=+0.05
7.2      Log-volumen del posterior                        rho=+0.51 [-0.26,+0.87] p=0.087 W=0.88 rhoM=+0.26 val=+0.82
11.1     Inconsistencia Kumar-Poole                       rho=-0.11 [-0.60,+0.37] p=0.658 W=0.47 rhoM=+0.33 val=+0.50
5.1m     kNN: fracción > p95 del control                  rho=+0.60 [+0.27,+0.83] p=0.005 W=0.53 rhoM=+0.61 val=+0.74
2.4m     Fisher-Rao (mediana)                             rho=+0.60 [+0.10,+0.88] p=0.028 W=0.81 rhoM=+0.74 val=+0.59
7.1m     Tasa (mediana, nats)                             rho=+0.70 [+0.23,+0.93] p=0.015 W=0.95 rhoM=+0.89 val=+0.89
6.1m     Log-magnificación (mediana)                      rho=+0.52 [-0.19,+0.87] p=0.060 W=0.76 rhoM=+0.36 val=+0.95
```

## E3_beta2_6 — OE4 beta2: prueba confirmatoria

- Inicio: 2026-10-11T21:45:47Z
- Comando: `env VAE_NIDS_OE2_CONFIG=configs/oe4_beta2.yaml /home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase7_confirmatoria`
- Fin: 2026-10-11T21:45:59Z (12 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E3_beta2_6.log`. Últimas líneas:

```
===== [2026-10-11T21:45:47Z] E3_beta2_6: OE4 beta2: prueba confirmatoria (intento 1) =====
[fase7] 1.1    Mahalanobis (mediana)                      rho=+0.603 [-0.037,+0.956] p=0.0313 (51 variables: +0.604)
[fase7] 2.4    Fisher-Rao al 10.º posterior benigno       rho=+0.716 [+0.275,+0.902] p=0.0077 (51 variables: +0.724)
[fase7] 6.2    kNN en el espacio decodificado             rho=+0.639 [+0.096,+0.948] p=0.0213 (51 variables: +0.691)
[fase7] 7.1    Tasa KL(q||p) por flujo                    rho=+0.658 [+0.074,+0.935] p=0.0276 (51 variables: +0.681)
[fase7] 5.1    kNN al benigno (z blanqueado)              rho=+0.707 [+0.065,+0.944] p=0.0073 (51 variables: +0.694)
[fase7] 6.1    Log-magnificación del decodificador        rho=+0.551 [-0.176,+0.936] p=0.0632 (51 variables: +0.523)
[fase7] 10.1b  Silhouette del grupo (Fisher-Rao)          rho=-0.624 [-0.913,-0.106] p=0.0385 (51 variables: -0.629)
[fase7] resumen en /home/estudiante/vae-latent-geometry-nids/outputs/oe4/beta2/oe2_descriptores/RESULTADOS_CONFIRMATORIA.md
```

## E3_k16_1 — OE4 k16: entrenamiento (5 semillas) y diagnósticos

- Inicio: 2026-10-11T21:45:59Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.oe1.run --config configs/oe4_k16.yaml`
- Fin: 2026-10-11T23:48:36Z (7357 s = 122 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E3_k16_1.log`. Últimas líneas:

```
===== [2026-10-11T21:45:59Z] E3_k16_1: OE4 k16: entrenamiento (5 semillas) y diagnósticos (intento 1) =====
[2026-10-11T21:46:02] preproceso
[2026-10-11T21:46:38] oe4_k16_seed42: entrenando
[2026-10-11T22:08:44] oe4_k16_seed42: parada en época 127 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 117, val -247.4276, 1319 s
[2026-10-11T22:08:44] oe4_k16_seed43: entrenando
[2026-10-11T22:17:19] oe4_k16_seed43: parada en época 49 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 39, val -238.9352, 510 s
[2026-10-11T22:17:19] oe4_k16_seed44: entrenando
[2026-10-11T22:53:18] oe4_k16_seed44: parada en época 207 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 197, val -248.7593, 2154 s
[2026-10-11T22:53:18] oe4_k16_seed45: entrenando
[2026-10-11T23:22:53] oe4_k16_seed45: parada en época 170 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 160, val -248.3573, 1769 s
[2026-10-11T23:22:53] oe4_k16_seed46: entrenando
[2026-10-11T23:48:25] oe4_k16_seed46: parada en época 148 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 138, val -246.5424, 1527 s
[2026-10-11T23:48:25] diagnósticos
[2026-10-11T23:48:35] listo: {"corridas": 5}
```

## E3_k16_2 — OE4 k16: codificación (mu, log sigma^2)

- Inicio: 2026-10-11T23:48:36Z
- Comando: `env VAE_NIDS_OE2_CONFIG=configs/oe4_k16.yaml /home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase2_encode`
- Fin: 2026-10-11T23:48:43Z (7 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E3_k16_2.log`. Últimas líneas:

```
===== [2026-10-11T23:48:36Z] E3_k16_2: OE4 k16: codificación (mu, log sigma^2) (intento 1) =====
[fase2 seed=42] BENIGN test: n=248,561
[fase2 seed=42] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=42] dims activas (7/16): [0, 3, 6, 9, 10, 13, 14] varianza=[0.32769, 3e-05, 1e-05, 0.98028, 1e-05, 4e-05, 0.98303, 1e-05, 1e-05, 0.48529, 0.41453, 2e-05, 4e-05, 0.90961, 0.39661, 2e-05]
[fase2 seed=43] BENIGN test: n=248,561
[fase2 seed=43] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=43] dims activas (7/16): [0, 5, 6, 7, 9, 13, 15] varianza=[0.42723, 5e-05, 0.00014, 4e-05, 3e-05, 0.9278, 0.42996, 0.15086, 3e-05, 0.54544, 3e-05, 3e-05, 2e-05, 0.78466, 5e-05, 1.00263]
[fase2 seed=44] BENIGN test: n=248,561
[fase2 seed=44] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=44] dims activas (7/16): [2, 4, 6, 7, 8, 11, 14] varianza=[2e-05, 0.0, 1.03532, 1e-05, 0.20761, 6e-05, 0.51722, 0.42914, 0.93891, 1e-05, 3e-05, 0.55272, 1e-05, 1e-05, 0.38396, 1e-05]
[fase2 seed=45] BENIGN test: n=248,561
[fase2 seed=45] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=45] dims activas (7/16): [1, 2, 3, 4, 5, 12, 14] varianza=[9e-05, 0.49319, 0.99897, 0.39688, 0.6482, 0.24529, 3e-05, 5e-05, 2e-05, 5e-05, 7e-05, 3e-05, 0.94043, 8e-05, 0.11732, 0.00011]
[fase2 seed=46] BENIGN test: n=248,561
[fase2 seed=46] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=46] dims activas (6/16): [1, 2, 4, 9, 10, 15] varianza=[1e-05, 0.96692, 0.31077, 2e-05, 0.52417, 2e-05, 1e-05, 0.0, 2e-05, 1.08893, 0.42932, 1e-05, 1e-05, 1e-05, 1e-05, 0.56974]
[fase2] tabla semilla x dimensión guardada en /home/estudiante/vae-latent-geometry-nids/outputs/oe4/k16/oe2_robust/tables/phase2_active_dims_per_seed.csv
```

## E3_k16_3 — OE4 k16: detectabilidad y referencia benigna

- Inicio: 2026-10-11T23:48:43Z
- Comando: `env VAE_NIDS_OE2_CONFIG=configs/oe4_k16.yaml /home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase1_detectabilidad`
- Fin: 2026-10-11T23:52:12Z (209 s = 3 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E3_k16_3.log`. Últimas líneas:

```
[fase1 seed=44] Bot_attempted                    n=   1470 AUC=0.6592
[fase1 seed=44] WebAttack_BruteForce             n=    151 AUC=0.9531
[fase1 seed=44] WebAttack_BruteForce_attempted   n=   1214 AUC=0.6113
[fase1 seed=45] PortScan                         n= 159023 AUC=0.7000
[fase1 seed=45] DoS_Hulk                         n= 158469 AUC=0.9465
[fase1 seed=45] DDoS                             n=  95123 AUC=0.9877
[fase1 seed=45] DoS_GoldenEye                    n=   7567 AUC=0.9792
[fase1 seed=45] DoS_slowloris                    n=   4001 AUC=0.9685
[fase1 seed=45] FTP_Patator                      n=   3973 AUC=0.9423
[fase1 seed=45] SSH_Patator                      n=   2980 AUC=0.9336
[fase1 seed=45] DoS_Slowhttptest                 n=   1742 AUC=0.9922
[fase1 seed=45] Bot                              n=    738 AUC=0.6265
[fase1 seed=45] Web_Attack                       n=    190 AUC=0.9405
[fase1 seed=45] Infiltration                     n=     32 AUC=0.9902
[fase1 seed=45] DoS_Hulk_attempted               n=    579 AUC=0.9278
[fase1 seed=45] DoS_GoldenEye_attempted          n=     80 AUC=0.7425
[fase1 seed=45] DoS_slowloris_attempted          n=   1706 AUC=0.9614
[fase1 seed=45] DoS_Slowhttptest_attempted       n=   3367 AUC=0.9822
[fase1 seed=45] Bot_attempted                    n=   1470 AUC=0.6470
[fase1 seed=45] WebAttack_BruteForce             n=    151 AUC=0.9519
[fase1 seed=45] WebAttack_BruteForce_attempted   n=   1214 AUC=0.6865
[fase1 seed=46] PortScan                         n= 159023 AUC=0.6586
[fase1 seed=46] DoS_Hulk                         n= 158469 AUC=0.9583
[fase1 seed=46] DDoS                             n=  95123 AUC=0.9856
[fase1 seed=46] DoS_GoldenEye                    n=   7567 AUC=0.9664
[fase1 seed=46] DoS_slowloris                    n=   4001 AUC=0.9477
[fase1 seed=46] FTP_Patator                      n=   3973 AUC=0.8793
[fase1 seed=46] SSH_Patator                      n=   2980 AUC=0.9104
[fase1 seed=46] DoS_Slowhttptest                 n=   1742 AUC=0.9688
[fase1 seed=46] Bot                              n=    738 AUC=0.6416
[fase1 seed=46] Web_Attack                       n=    190 AUC=0.9367
[fase1 seed=46] Infiltration                     n=     32 AUC=0.9860
[fase1 seed=46] DoS_Hulk_attempted               n=    579 AUC=0.9137
[fase1 seed=46] DoS_GoldenEye_attempted          n=     80 AUC=0.7302
[fase1 seed=46] DoS_slowloris_attempted          n=   1706 AUC=0.9592
[fase1 seed=46] DoS_Slowhttptest_attempted       n=   3367 AUC=0.9859
[fase1 seed=46] Bot_attempted                    n=   1470 AUC=0.6126
[fase1 seed=46] WebAttack_BruteForce             n=    151 AUC=0.9547
[fase1 seed=46] WebAttack_BruteForce_attempted   n=   1214 AUC=0.6104
[fase1] diferencia máxima de AUC vs phase4 (seed 42): nan
```

## E3_k16_4 — OE4 k16: descriptores por familia y semilla

- Inicio: 2026-10-11T23:52:12Z
- Comando: `env VAE_NIDS_OE2_CONFIG=configs/oe4_k16.yaml /home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase2_descriptores`
- Fin: 2026-10-12T00:51:58Z (3586 s = 59 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E3_k16_4.log`. Últimas líneas:

```
[fase2 seed=44] WebAttack_BruteForce             n=    151 únicos=    151 AUC_knn=0.966 sil_w=0.260  (689s)
[fase2 seed=44] WebAttack_BruteForce_attempted   n=   1214 únicos=   1214 AUC_knn=0.555 sil_w=0.921  (716s)
[fase2 seed=45] BENIGN_control                   n=  50000 únicos=  47782 AUC_knn=0.500 sil_w=-0.001  (77s)
[fase2 seed=45] PortScan                         n= 159023 únicos=   1670 AUC_knn=0.890 sil_w=0.297  (112s)
[fase2 seed=45] DoS_Hulk                         n= 158469 únicos= 158469 AUC_knn=0.981 sil_w=0.362  (264s)
[fase2 seed=45] DDoS                             n=  95123 únicos=  95123 AUC_knn=0.979 sil_w=0.424  (366s)
[fase2 seed=45] DoS_GoldenEye                    n=   7567 únicos=   7567 AUC_knn=0.974 sil_w=0.084  (404s)
[fase2 seed=45] DoS_slowloris                    n=   4001 únicos=   3596 AUC_knn=0.959 sil_w=0.038  (437s)
[fase2 seed=45] FTP_Patator                      n=   3973 únicos=   3973 AUC_knn=0.949 sil_w=0.958  (471s)
[fase2 seed=45] SSH_Patator                      n=   2980 únicos=   2980 AUC_knn=0.958 sil_w=0.938  (504s)
[fase2 seed=45] DoS_Slowhttptest                 n=   1742 únicos=   1731 AUC_knn=0.952 sil_w=0.054  (537s)
[fase2 seed=45] Bot                              n=    738 únicos=    738 AUC_knn=0.843 sil_w=0.651  (557s)
[fase2 seed=45] Web_Attack                       n=    190 únicos=    190 AUC_knn=0.965 sil_w=0.149  (569s)
[fase2 seed=45] Infiltration                     n=     32 únicos=     32 AUC_knn=0.975 sil_w=-0.627  (580s)
[fase2 seed=45] DoS_Hulk_attempted               n=    579 únicos=    573 AUC_knn=0.908 sil_w=0.070  (598s)
[fase2 seed=45] DoS_GoldenEye_attempted          n=     80 únicos=     77 AUC_knn=0.691 sil_w=0.925  (609s)
[fase2 seed=45] DoS_slowloris_attempted          n=   1706 únicos=   1703 AUC_knn=0.977 sil_w=0.922  (640s)
[fase2 seed=45] DoS_Slowhttptest_attempted       n=   3367 únicos=   3361 AUC_knn=0.217 sil_w=0.602  (674s)
[fase2 seed=45] Bot_attempted                    n=   1470 únicos=    713 AUC_knn=0.160 sil_w=0.965  (693s)
[fase2 seed=45] WebAttack_BruteForce             n=    151 únicos=    151 AUC_knn=0.974 sil_w=0.224  (705s)
[fase2 seed=45] WebAttack_BruteForce_attempted   n=   1214 únicos=   1214 AUC_knn=0.531 sil_w=0.942  (733s)
[fase2 seed=46] BENIGN_control                   n=  50000 únicos=  47712 AUC_knn=0.500 sil_w=0.010  (71s)
[fase2 seed=46] PortScan                         n= 159023 únicos=   1670 AUC_knn=0.835 sil_w=0.369  (107s)
[fase2 seed=46] DoS_Hulk                         n= 158469 únicos= 158469 AUC_knn=0.957 sil_w=0.675  (245s)
[fase2 seed=46] DDoS                             n=  95123 únicos=  95123 AUC_knn=0.990 sil_w=0.649  (338s)
[fase2 seed=46] DoS_GoldenEye                    n=   7567 únicos=   7567 AUC_knn=0.975 sil_w=0.390  (376s)
[fase2 seed=46] DoS_slowloris                    n=   4001 únicos=   3596 AUC_knn=0.973 sil_w=0.225  (410s)
[fase2 seed=46] FTP_Patator                      n=   3973 únicos=   3973 AUC_knn=0.988 sil_w=0.962  (445s)
[fase2 seed=46] SSH_Patator                      n=   2980 únicos=   2980 AUC_knn=0.995 sil_w=0.924  (478s)
[fase2 seed=46] DoS_Slowhttptest                 n=   1742 únicos=   1731 AUC_knn=0.986 sil_w=0.188  (509s)
[fase2 seed=46] Bot                              n=    738 únicos=    738 AUC_knn=0.811 sil_w=0.693  (529s)
[fase2 seed=46] Web_Attack                       n=    190 únicos=    190 AUC_knn=0.956 sil_w=0.107  (541s)
[fase2 seed=46] Infiltration                     n=     32 únicos=     32 AUC_knn=0.984 sil_w=-0.628  (553s)
[fase2 seed=46] DoS_Hulk_attempted               n=    579 únicos=    573 AUC_knn=0.930 sil_w=0.108  (571s)
[fase2 seed=46] DoS_GoldenEye_attempted          n=     80 únicos=     77 AUC_knn=0.750 sil_w=0.948  (582s)
[fase2 seed=46] DoS_slowloris_attempted          n=   1706 únicos=   1703 AUC_knn=0.978 sil_w=0.935  (614s)
[fase2 seed=46] DoS_Slowhttptest_attempted       n=   3367 únicos=   3361 AUC_knn=0.212 sil_w=0.623  (648s)
[fase2 seed=46] Bot_attempted                    n=   1470 únicos=    713 AUC_knn=0.151 sil_w=0.953  (669s)
[fase2 seed=46] WebAttack_BruteForce             n=    151 únicos=    151 AUC_knn=0.959 sil_w=0.197  (680s)
[fase2 seed=46] WebAttack_BruteForce_attempted   n=   1214 únicos=   1214 AUC_knn=0.503 sil_w=0.949  (709s)
```

## E3_k16_5 — OE4 k16: cribado de descriptores

- Inicio: 2026-10-12T00:51:58Z
- Comando: `env VAE_NIDS_OE2_CONFIG=configs/oe4_k16.yaml /home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase3_evaluacion`
- Fin: 2026-10-12T00:52:28Z (30 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E3_k16_5.log`. Últimas líneas:

```
===== [2026-10-12T00:51:58Z] E3_k16_5: OE4 k16: cribado de descriptores (intento 1) =====
/home/estudiante/vae-latent-geometry-nids/src/vae_nids/evaluation/oe2_descriptores/fase3_evaluacion.py:77: RuntimeWarning: Mean of empty slice
  return float(np.tanh(np.nanmean(np.arctanh(r))))
1.1      Mahalanobis (mediana)                            rho=+0.65 [+0.18,+0.95] p=0.003 W=0.69 rhoM=+1.00 val=+0.73
5.1      kNN al benigno (z blanqueado)                    rho=+0.54 [+0.05,+0.94] p=0.015 W=0.64 rhoM=+0.78 val=+0.56
2.4      Fisher-Rao al k-ésimo posterior benigno          rho=+0.71 [+0.36,+0.89] p=0.001 W=0.74 rhoM=+0.60 val=+0.60
7.1      Tasa KL(q||p) por flujo                          rho=+0.81 [+0.47,+0.97] p=0.000 W=0.88 rhoM=+0.85 val=+0.86
3.2      MMD² analítico entre posteriores agregados       rho=+0.36 [-0.43,+0.87] p=0.193 W=0.81 rhoM=+0.58 val=+0.56
6.2      kNN en el espacio decodificado                   rho=+0.65 [+0.29,+0.87] p=0.004 W=0.66 rhoM=+0.55 val=+0.71
6.1      Log-magnificación del decodificador              rho=+0.54 [-0.00,+0.85] p=0.020 W=0.67 rhoM=+0.45 val=+0.69
4.2      Distancia afín-invariante entre covarianzas      rho=+0.07 [-0.44,+0.55] p=0.800 W=0.87 rhoM=-0.14 val=-0.48
8.1      LID relativa al benigno                          rho=+0.43 [-0.06,+0.81] p=0.036 W=0.46 rhoM=+0.01 val=+0.18
10.1a    Silhouette del grupo (z blanqueado)              rho=-0.42 [-0.81,+0.13] p=0.148 W=0.92 rhoM=-0.56 val=-0.78
10.3     Sonda lineal sobre (μ, log σ²)                   rho=-0.00 [-0.58,+0.60] p=0.996 W=0.92 rhoM=-0.08 val=+0.12
1.1b     Mahalanobis (AUC frente al control)              rho=+0.72 [+0.32,+0.97] p=0.001 W=0.77 rhoM=+0.92 val=+0.74
10.1b    Silhouette del grupo (Fisher-Rao)                rho=-0.64 [-0.95,-0.25] p=0.015 W=0.96 rhoM=-0.73 val=-0.61
1.2      Log-densidad de mezcla benigna (K=8)             rho=+0.73 [+0.34,+0.96] p=0.001 W=0.81 rhoM=+0.81 val=+0.95
1.3      Mahalanobis relativa al prior                    rho=+0.59 [+0.01,+0.95] p=0.005 W=0.59 rhoM=+0.92 val=+0.72
1.4      Tipicidad radial bilateral                       rho=+0.72 [+0.31,+0.98] p=0.001 W=0.77 rhoM=+0.92 val=+0.74
4.1      Rango efectivo de la familia                     rho=+0.17 [-0.48,+0.79] p=0.551 W=0.82 rhoM=+0.02 val=+0.33
7.2      Log-volumen del posterior                        rho=+0.56 [-0.05,+0.88] p=0.021 W=0.74 rhoM=+0.60 val=+0.75
11.1     Inconsistencia Kumar-Poole                       rho=+0.12 [-0.36,+0.56] p=0.632 W=0.57 rhoM=-0.06 val=+0.24
5.1m     kNN: fracción > p95 del control                  rho=+0.42 [-0.11,+0.89] p=0.057 W=0.53 rhoM=+0.63 val=+0.78
2.4m     Fisher-Rao (mediana)                             rho=+0.68 [+0.29,+0.86] p=0.003 W=0.69 rhoM=+0.60 val=+0.53
7.1m     Tasa (mediana, nats)                             rho=+0.75 [+0.41,+0.96] p=0.001 W=0.92 rhoM=+0.87 val=+0.85
6.1m     Log-magnificación (mediana)                      rho=+0.49 [-0.09,+0.83] p=0.045 W=0.71 rhoM=+0.43 val=+0.72
```

## E3_k16_6 — OE4 k16: prueba confirmatoria

- Inicio: 2026-10-12T00:52:28Z
- Comando: `env VAE_NIDS_OE2_CONFIG=configs/oe4_k16.yaml /home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase7_confirmatoria`
- Fin: 2026-10-12T00:52:38Z (10 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E3_k16_6.log`. Últimas líneas:

```
===== [2026-10-12T00:52:28Z] E3_k16_6: OE4 k16: prueba confirmatoria (intento 1) =====
[fase7] 1.1    Mahalanobis (mediana)                      rho=+0.647 [+0.087,+0.967] p=0.0035 (51 variables: +0.604)
[fase7] 2.4    Fisher-Rao al 10.º posterior benigno       rho=+0.711 [+0.295,+0.913] p=0.0020 (51 variables: +0.724)
[fase7] 6.2    kNN en el espacio decodificado             rho=+0.651 [+0.201,+0.897] p=0.0037 (51 variables: +0.691)
[fase7] 7.1    Tasa KL(q||p) por flujo                    rho=+0.807 [+0.327,+0.987] p=0.0003 (51 variables: +0.681)
[fase7] 5.1    kNN al benigno (z blanqueado)              rho=+0.537 [-0.062,+0.954] p=0.0165 (51 variables: +0.694)
[fase7] 6.1    Log-magnificación del decodificador        rho=+0.542 [-0.164,+0.875] p=0.0205 (51 variables: +0.523)
[fase7] 10.1b  Silhouette del grupo (Fisher-Rao)          rho=-0.642 [-0.929,-0.143] p=0.0165 (51 variables: -0.629)
[fase7] resumen en /home/estudiante/vae-latent-geometry-nids/outputs/oe4/k16/oe2_descriptores/RESULTADOS_CONFIRMATORIA.md
```

## E40 — OE4: concordancia de rankings y hallazgo por configuración

- Inicio: 2026-10-12T00:52:38Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe4.analisis`
- Fin: 2026-10-12T00:52:43Z (5 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E40.log`. Últimas líneas:

```
===== [2026-10-12T00:52:38Z] E40: OE4: concordancia de rankings y hallazgo por configuración (intento 1) =====
[oe4] resumen en /home/estudiante/vae-latent-geometry-nids/outputs/oe4/RESULTADOS_OE4.md
```

## Fin de la corrida — 2026-10-12T00:52:43Z

- E01: OK
- E10: OK
- E20: OK
- E3_beta0p5_1: OK
- E3_beta0p5_2: OK
- E3_beta0p5_3: OK
- E3_beta0p5_4: OK
- E3_beta0p5_5: OK
- E3_beta0p5_6: OK
- E3_beta2_1: OK
- E3_beta2_2: OK
- E3_beta2_3: OK
- E3_beta2_4: OK
- E3_beta2_5: OK
- E3_beta2_6: OK
- E3_k16_1: OK
- E3_k16_2: OK
- E3_k16_3: OK
- E3_k16_4: OK
- E3_k16_5: OK
- E3_k16_6: OK
- E40: OK

## E99 — Resumen de la corrida

- Inicio: 2026-10-12T00:52:43Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python scripts/vm_resumen_oe3_oe4.py`
- Fin: 2026-10-12T00:52:44Z (1 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_oe3_oe4/logs/E99.log`. Últimas líneas:

```
===== [2026-10-12T00:52:43Z] E99: Resumen de la corrida (intento 1) =====
[resumen] /home/estudiante/vae-latent-geometry-nids/outputs/vm_oe3_oe4/RESUMEN_CORRIDA.md
```
