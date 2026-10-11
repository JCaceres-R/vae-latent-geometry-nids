
# Corrida de pendientes (OE1 saneado + OE2 recalculado) — 2026-10-10T19:43:31Z

- Host: tesis-juan; `uname -a`: Linux tesis-juan 5.15.0-194-generic #204-Ubuntu SMP Wed Sep 2 13:09:23 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux
- CPUs: 8; memoria: 23 GB total, 22 GB disponibles
- Disco libre en el repo: 59G
- Commit: d5d14493617551cf625ab39ed0bf742aa527b1f1 (rama main)
- Cambios sin commit en archivos versionados: 37

## Entorno

- Python: Python 3.10.12
- Dependencias: SIN versiones exactas (requirements.txt falló; ver E00_entorno.log y pip_freeze.txt)
- torch: 2.13.0+cpu | hilos: 8

## E01 — Tests (OE1 saneado y estadística del OE2)

- Inicio: 2026-10-10T19:53:58Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m pytest tests/oe1_saneado tests/oe2_robust/test_stats_utils.py -q -p no:cacheprovider`
- Fin: 2026-10-10T19:54:35Z (37 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E01.log`. Últimas líneas:

```
===== [2026-10-10T19:53:58Z] E01: Tests (OE1 saneado y estadística del OE2) =====
........................                                                 [100%]
24 passed in 33.85s
```

## E10 — OE1: datos, preproceso, 8 semillas + referencia 74, diagnósticos

- Inicio: 2026-10-10T19:54:35Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.oe1.run --config configs/oe1_saneado.yaml --ref74 configs/oe1_saneado_ref74.yaml`
- Fin: 2026-10-10T21:50:18Z (6943 s = 115 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E10.log`. Últimas líneas:

```
===== [2026-10-10T19:54:35Z] E10: OE1: datos, preproceso, 8 semillas + referencia 74, diagnósticos =====
[2026-10-10T19:54:40] datos: construyendo particiones crudas
[2026-10-10T19:59:09] preproceso
[2026-10-10T19:59:40] oe1_info_nula_seed42: entrenando
[2026-10-10T20:12:19] oe1_info_nula_seed42: parada en época 87 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 77, val -242.7477, 740 s
[2026-10-10T20:12:19] oe1_info_nula_seed43: entrenando
[2026-10-10T20:20:25] oe1_info_nula_seed43: parada en época 57 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 47, val -241.9347, 481 s
[2026-10-10T20:20:25] oe1_info_nula_seed44: entrenando
[2026-10-10T20:30:56] oe1_info_nula_seed44: parada en época 74 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 64, val -245.0449, 627 s
[2026-10-10T20:30:56] oe1_info_nula_seed45: entrenando
[2026-10-10T20:43:31] oe1_info_nula_seed45: parada en época 85 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 75, val -245.1515, 751 s
[2026-10-10T20:43:31] oe1_info_nula_seed46: entrenando
[2026-10-10T20:50:22] oe1_info_nula_seed46: parada en época 48 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 38, val -236.9820, 406 s
[2026-10-10T20:50:22] oe1_info_nula_seed47: entrenando
[2026-10-10T21:06:04] oe1_info_nula_seed47: parada en época 109 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 99, val -246.3414, 938 s
[2026-10-10T21:06:04] oe1_info_nula_seed48: entrenando
[2026-10-10T21:23:15] oe1_info_nula_seed48: parada en época 122 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 112, val -243.5257, 1027 s
[2026-10-10T21:23:15] oe1_info_nula_seed49: entrenando
[2026-10-10T21:36:45] oe1_info_nula_seed49: parada en época 95 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 85, val -244.2359, 805 s
[2026-10-10T21:36:45] referencia_74_seed42: entrenando
[2026-10-10T21:50:01] referencia_74_seed42: parada en época 91 (paciencia (10 épocas sin mejorar más de min_delta=0.1)), mejor época 81, val -270.4709, 791 s
[2026-10-10T21:50:01] diagnósticos
[2026-10-10T21:50:17] listo: {"corridas": 9}
```

## E11 — OE1: verificación de checkpoints y diagnósticos

- Inicio: 2026-10-10T21:50:18Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -`
- Fin: 2026-10-10T21:50:19Z (1 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E11.log`. Últimas líneas:

```
===== [2026-10-10T21:50:18Z] E11: OE1: verificación de checkpoints y diagnósticos =====
         corrida  seed  n_columnas  mejor_epoca  activas_burda  kl_total  frac_logvar_dec_en_piso
0  oe1_info_nula    42          66           77              7  9.784456                 0.910270
1  oe1_info_nula    43          66           47              7  7.883008                 0.887778
2  oe1_info_nula    44          66           64              7  8.531523                 0.915067
3  oe1_info_nula    45          66           75              7  9.008251                 0.921152
4  oe1_info_nula    46          66           38              7  8.288469                 0.875387
5  oe1_info_nula    47          66           99              7  7.467426                 0.915344
6  oe1_info_nula    48          66          112              7  8.792538                 0.909653
7  oe1_info_nula    49          66           85              6  8.482490                 0.901580
8  referencia_74    42          74           81              7  9.651970                 0.887711
OK: 9 checkpoints y diagnósticos completos
```

## E12 — OE1: figuras del informe (selección y preproceso)

- Inicio: 2026-10-10T21:50:19Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.oe1.figuras_informe --config configs/oe1_saneado.yaml`
- Fin: 2026-10-10T21:50:28Z (9 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E12.log`. Últimas líneas:

```
===== [2026-10-10T21:50:19Z] E12: OE1: figuras del informe (selección y preproceso) =====
                                                                    caso  distintos   crudo      frac
0  Flow Packets/s, pipeline original (float32, con duraciones negativas)      10084  577678  0.017456
1                                    Flow Packets/s, saneado (filtro D4)     573049  577678  0.991987
2                                     Flow Duration, saneado (filtro D4)     517280  524335  0.986545
/home/estudiante/vae-latent-geometry-nids/src/vae_nids/oe1/figuras_informe.py:65: RuntimeWarning: invalid value encountered in divide
  ratio = np.where(tol > 0, np.abs(a - b) / tol, 0.0)
```

## E13 — OE1: tablas LaTeX y cifras del informe

- Inicio: 2026-10-10T21:50:28Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.oe1.tablas_informe --config configs/oe1_saneado.yaml --dest outputs/oe1_saneado/informe_tablas`
- Fin: 2026-10-10T21:50:28Z (0 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E13.log`. Últimas líneas:

```
===== [2026-10-10T21:50:28Z] E13: OE1: tablas LaTeX y cifras del informe =====
{'nUniverso': '77', 'nQuedan': '66', 'nRcero': '3', 'nRuno': '0', 'nRdos': '8', 'nCasi': '6', 'hashCorto': '49435ba7a103', 'hashLargo': '49435ba7a10312a12aef49ee6c0f75b7916cf92813ba8cf3cf12291739405d97', 'hashPartido': '49435ba7a10312a1\\allowbreak{}2aef49ee6c0f75b7\\allowbreak{}916cf92813ba8cf3\\allowbreak{}cf12291739405d97', 'nAlternativo': '60', 'casiFallosMin': '0{,}006', 'casiFallosMax': '3{,}6', 'casiUlpMin': '8', 'casiUlpMax': '100', 'nCasiConst': '6', 'postComprobaciones': '10\\,340\\,726', 'postFallos': '0', 'postEtiquetas': '24', 'nComunActual': '45', 'nVuelven': '21', 'nSalenActual': '6', 'filasCrudas': '2\\,100\\,814', 'filasInf': '793', 'dcuatroTrain': '24', 'dcuatroVal': '7', 'dcuatroTotal': '31', 'iatNegTrain': '1\\,873', 'nMarcadas': '18', 'minConservado': '90{,}6', 'fpsAntes': '1{,}7', 'fpsDespues': '99{,}2', 'nDependencias': '12', 'activasMedia': '6{,}9', 'activasDE': '0{,}4', 'activasAntes': '5{,}4', 'epocaMin': '38', 'epocaMax': '112', 'pisoFrac': '90{,}5', 'nSemillasNuevo': '8', 'reentrenamiento': 'completo'}
```

## E20 — OE2 robusto: codificación (mu, log sigma^2) por semilla

- Inicio: 2026-10-10T21:50:28Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase2_encode`
- Fin: 2026-10-10T21:50:38Z (10 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E20.log`. Últimas líneas:

```
===== [2026-10-10T21:50:28Z] E20: OE2 robusto: codificación (mu, log sigma^2) por semilla =====
[fase2 seed=42] BENIGN test: n=248,561
[fase2 seed=42] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=42] dims activas (7/8): [0, 1, 2, 3, 5, 6, 7] varianza=[0.45247, 0.50358, 0.67387, 0.2291, 1e-05, 0.9603, 0.95085, 0.81844]
[fase2 seed=43] BENIGN test: n=248,561
[fase2 seed=43] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=43] dims activas (7/8): [0, 2, 3, 4, 5, 6, 7] varianza=[0.518, 1e-05, 0.24181, 0.47448, 0.40548, 0.9034, 0.9277, 0.73609]
[fase2 seed=44] BENIGN test: n=248,561
[fase2 seed=44] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=44] dims activas (7/8): [0, 1, 2, 3, 4, 5, 7] varianza=[0.34886, 0.29869, 0.21955, 0.88417, 0.39338, 0.66482, 2e-05, 0.98913]
[fase2 seed=45] BENIGN test: n=248,561
[fase2 seed=45] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=45] dims activas (7/8): [0, 1, 2, 3, 5, 6, 7] varianza=[0.28543, 0.58517, 0.2835, 1.02701, 2e-05, 0.94555, 0.88361, 0.28052]
[fase2 seed=46] BENIGN test: n=248,561
[fase2 seed=46] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=46] dims activas (7/8): [0, 2, 3, 4, 5, 6, 7] varianza=[0.96318, 2e-05, 0.45795, 0.18572, 0.95186, 0.54947, 0.92148, 0.44471]
[fase2 seed=47] BENIGN test: n=248,561
[fase2 seed=47] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=47] dims activas (7/8): [1, 2, 3, 4, 5, 6, 7] varianza=[1e-05, 0.3072, 0.3399, 0.83378, 0.9283, 0.21438, 0.46446, 0.9617]
[fase2 seed=48] BENIGN test: n=248,561
[fase2 seed=48] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=48] dims activas (7/8): [1, 2, 3, 4, 5, 6, 7] varianza=[1e-05, 1.00232, 0.14376, 0.74554, 0.22264, 0.69455, 0.9413, 0.40605]
[fase2 seed=49] BENIGN test: n=248,561
[fase2 seed=49] conteos frente al manifest del OE2 original: idénticos
[fase2 seed=49] dims activas (6/8): [0, 1, 3, 4, 5, 7] varianza=[0.54036, 0.45019, 1e-05, 0.60727, 0.946, 0.31784, 4e-05, 0.9782]
[fase2] tabla semilla x dimensión guardada en /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase2_active_dims_per_seed.csv
```

## E21 — OE2 robusto: geometría distribucional

- Inicio: 2026-10-10T21:50:38Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase3_geometry`
- Fin: 2026-10-10T22:19:03Z (1705 s = 28 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E21.log`. Últimas líneas:

```
[fase3.2 seed=48] Web_Attack: MMD2=0.4617 SW=0.8955 energy=2.5983
[fase3.2 seed=48] Infiltration: MMD2=0.4600 SW=5.7837 energy=6.0869
[fase3.2 seed=48] DoS_Hulk_attempted: MMD2=0.3408 SW=0.8794 energy=2.2055
[fase3.2 seed=48] DoS_GoldenEye_attempted: MMD2=0.2401 SW=0.6822 energy=1.9023
[fase3.2 seed=48] DoS_slowloris_attempted: MMD2=0.7546 SW=1.3857 energy=6.1959
[fase3.2 seed=48] DoS_Slowhttptest_attempted: MMD2=0.5741 SW=1.1575 energy=4.5300
[fase3.2 seed=48] Bot_attempted: MMD2=0.4396 SW=0.6876 energy=1.8398
[fase3.2 seed=48] WebAttack_BruteForce: MMD2=0.5478 SW=0.9813 energy=3.0674
[fase3.2 seed=48] WebAttack_BruteForce_attempted: MMD2=0.4943 SW=0.8123 energy=2.6477
[fase3 seed=48] cacheado en phase3_seed48.json
[fase3 seed=49] dims activas usadas: [0, 1, 3, 4, 5, 7]
[fase3.2 seed=49] PortScan: MMD2=0.3802 SW=0.6168 energy=1.3583
[fase3.2 seed=49] DoS_Hulk: MMD2=0.9844 SW=2.1140 energy=8.4316
[fase3.2 seed=49] DDoS: MMD2=0.9833 SW=2.4597 energy=9.6150
[fase3.2 seed=49] DoS_GoldenEye: MMD2=0.6254 SW=1.8749 energy=5.6679
[fase3.2 seed=49] DoS_slowloris: MMD2=0.3263 SW=0.9740 energy=2.3943
[fase3.2 seed=49] FTP_Patator: MMD2=0.5696 SW=0.9950 energy=3.4298
[fase3.2 seed=49] SSH_Patator: MMD2=0.5291 SW=0.9246 energy=3.0637
[fase3.2 seed=49] DoS_Slowhttptest: MMD2=0.3786 SW=1.1983 energy=3.2409
[fase3.2 seed=49] Bot: MMD2=0.2122 SW=0.5949 energy=1.2129
[fase3.2 seed=49] Web_Attack: MMD2=0.3395 SW=0.7960 energy=1.8730
[fase3.2 seed=49] Infiltration: MMD2=0.5169 SW=2.5020 energy=4.2726
[fase3.2 seed=49] DoS_Hulk_attempted: MMD2=0.4796 SW=1.0098 energy=2.7792
[fase3.2 seed=49] DoS_GoldenEye_attempted: MMD2=0.3562 SW=0.7998 energy=2.3513
[fase3.2 seed=49] DoS_slowloris_attempted: MMD2=0.6197 SW=1.2196 energy=4.6164
[fase3.2 seed=49] DoS_Slowhttptest_attempted: MMD2=0.4978 SW=1.1465 energy=4.0357
[fase3.2 seed=49] Bot_attempted: MMD2=0.4533 SW=0.7074 energy=1.7342
[fase3.2 seed=49] WebAttack_BruteForce: MMD2=0.4132 SW=0.8789 energy=2.2816
[fase3.2 seed=49] WebAttack_BruteForce_attempted: MMD2=0.6014 SW=1.0732 energy=3.8039
[fase3 seed=49] cacheado en phase3_seed49.json
[fase3] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase3_mahalanobis_summary.csv (1296 filas)
[fase3] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase3_mahalanobis_tests.csv (288 filas)
[fase3] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase3_cliffs_delta.csv (144 filas)
[fase3] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase3_robust_ranking.csv (8 filas)
[fase3] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase3_divergences.csv (432 filas)
[fase3] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase3_bhattacharyya.csv (432 filas)
[fase3] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase3_posterior_mahalanobis.csv (576 filas)
[fase3] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase3_posterior_uncertainty.csv (152 filas)
[fase3] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase3_neighborhood.csv (576 filas)
[fase3] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase3_dispersion.csv (432 filas)
```

- Semillas del OE2 (perfil saneado): 42 43 44 45 46 47 48 49

## E22_s42 — OE2 robusto: detectabilidad, semilla 42

- Inicio: 2026-10-10T22:19:05Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed 42`
- Fin: 2026-10-10T22:30:27Z (682 s = 11 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E22_s42.log`. Últimas líneas:

```
===== [2026-10-10T22:19:05Z] E22_s42: OE2 robusto: detectabilidad, semilla 42 =====
[fase4 seed=42] tau_p95=-216.6112 fpr_test_benign=0.0504 AUC(nll_mc) principales: {'PortScan': 0.6727937868783134, 'DoS_Hulk': 0.9364883046767849, 'DDoS': 0.9880926504933847, 'DoS_GoldenEye': 0.970326622265858, 'DoS_slowloris': 0.9354712161592529, 'FTP_Patator': 0.8834263278935187, 'SSH_Patator': 0.9403647239146108, 'DoS_Slowhttptest': 0.9661804240732041, 'Bot': 0.5992895540334501, 'Web_Attack': 0.9424951070996233, 'Infiltration': 0.9851551782057524}
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_detectability_metrics.csv (90 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_families.csv (275 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_scores.csv (54 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_sample_size_control.csv (11 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_tau_per_seed.csv (1 filas)
```

## E22_s43 — OE2 robusto: detectabilidad, semilla 43

- Inicio: 2026-10-10T22:30:27Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed 43`
- Fin: 2026-10-10T22:31:21Z (54 s = 0 min)
- Estado: **FALLO** (exit 132)

Log completo: `outputs/vm_pendientes/logs/E22_s43.log`. Últimas líneas:

```
===== [2026-10-10T22:30:27Z] E22_s43: OE2 robusto: detectabilidad, semilla 43 =====
```

## E22_s44 — OE2 robusto: detectabilidad, semilla 44

- Inicio: 2026-10-10T22:31:21Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed 44`
- Fin: 2026-10-10T22:42:24Z (663 s = 11 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E22_s44.log`. Últimas líneas:

```
===== [2026-10-10T22:31:21Z] E22_s44: OE2 robusto: detectabilidad, semilla 44 =====
[fase4 seed=44] tau_p95=-220.6024 fpr_test_benign=0.0505 AUC(nll_mc) principales: {'PortScan': 0.7034949734438936, 'DoS_Hulk': 0.9339770147399621, 'DDoS': 0.9909033794947294, 'DoS_GoldenEye': 0.9755805759301139, 'DoS_slowloris': 0.9367101188401952, 'FTP_Patator': 0.8948803908805251, 'SSH_Patator': 0.9466809634106264, 'DoS_Slowhttptest': 0.9829699994269195, 'Bot': 0.5987046643733361, 'Web_Attack': 0.917509119332986, 'Infiltration': 0.9861680709161935}
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_detectability_metrics.csv (90 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_families.csv (275 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_scores.csv (54 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_sample_size_control.csv (11 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_tau_per_seed.csv (1 filas)
```

## E22_s45 — OE2 robusto: detectabilidad, semilla 45

- Inicio: 2026-10-10T22:42:24Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed 45`
- Fin: 2026-10-10T22:53:05Z (641 s = 10 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E22_s45.log`. Últimas líneas:

```
===== [2026-10-10T22:42:24Z] E22_s45: OE2 robusto: detectabilidad, semilla 45 =====
[fase4 seed=45] tau_p95=-219.7184 fpr_test_benign=0.0502 AUC(nll_mc) principales: {'PortScan': 0.6654609816776426, 'DoS_Hulk': 0.941644531711277, 'DDoS': 0.9892944824650568, 'DoS_GoldenEye': 0.9730427271580849, 'DoS_slowloris': 0.9554209053555706, 'FTP_Patator': 0.9276939078197939, 'SSH_Patator': 0.9419632309614409, 'DoS_Slowhttptest': 0.9689632803108146, 'Bot': 0.611930674043807, 'Web_Attack': 0.936622345589635, 'Infiltration': 0.9867484742175965}
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_detectability_metrics.csv (90 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_families.csv (275 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_scores.csv (54 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_sample_size_control.csv (11 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_tau_per_seed.csv (1 filas)
```

## E22_s46 — OE2 robusto: detectabilidad, semilla 46

- Inicio: 2026-10-10T22:53:05Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed 46`
- Fin: 2026-10-10T23:03:58Z (653 s = 10 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E22_s46.log`. Últimas líneas:

```
===== [2026-10-10T22:53:05Z] E22_s46: OE2 robusto: detectabilidad, semilla 46 =====
[fase4 seed=46] tau_p95=-190.4567 fpr_test_benign=0.0508 AUC(nll_mc) principales: {'PortScan': 0.5995772324903615, 'DoS_Hulk': 0.8538023097830248, 'DDoS': 0.9517279047000607, 'DoS_GoldenEye': 0.9413692102185536, 'DoS_slowloris': 0.925861181982235, 'FTP_Patator': 0.9331724992241853, 'SSH_Patator': 0.9851412832937528, 'DoS_Slowhttptest': 0.9927837052577506, 'Bot': 0.5972130242924888, 'Web_Attack': 0.9222267159242282, 'Infiltration': 0.9879820748226794}
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_detectability_metrics.csv (90 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_families.csv (275 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_scores.csv (54 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_sample_size_control.csv (11 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_tau_per_seed.csv (1 filas)
```

## E22_s47 — OE2 robusto: detectabilidad, semilla 47

- Inicio: 2026-10-10T23:03:58Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed 47`
- Fin: 2026-10-10T23:04:02Z (4 s = 0 min)
- Estado: **FALLO** (exit 132)

Log completo: `outputs/vm_pendientes/logs/E22_s47.log`. Últimas líneas:

```
===== [2026-10-10T23:03:58Z] E22_s47: OE2 robusto: detectabilidad, semilla 47 =====
```

## E22_s48 — OE2 robusto: detectabilidad, semilla 48

- Inicio: 2026-10-10T23:04:02Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed 48`
- Fin: 2026-10-10T23:04:07Z (5 s = 0 min)
- Estado: **FALLO** (exit 132)

Log completo: `outputs/vm_pendientes/logs/E22_s48.log`. Últimas líneas:

```
===== [2026-10-10T23:04:02Z] E22_s48: OE2 robusto: detectabilidad, semilla 48 =====
```

## E22_s49 — OE2 robusto: detectabilidad, semilla 49

- Inicio: 2026-10-10T23:04:07Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed 49`
- Fin: 2026-10-10T23:14:52Z (645 s = 10 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E22_s49.log`. Últimas líneas:

```
===== [2026-10-10T23:04:07Z] E22_s49: OE2 robusto: detectabilidad, semilla 49 =====
[fase4 seed=49] tau_p95=-216.4896 fpr_test_benign=0.0509 AUC(nll_mc) principales: {'PortScan': 0.6855836603215291, 'DoS_Hulk': 0.9503017085265734, 'DDoS': 0.9956678055389666, 'DoS_GoldenEye': 0.969966442822154, 'DoS_slowloris': 0.9351730510330082, 'FTP_Patator': 0.8539810178851843, 'SSH_Patator': 0.9330937007644188, 'DoS_Slowhttptest': 0.9685625708420378, 'Bot': 0.6298016341410754, 'Web_Attack': 0.9434199568505793, 'Infiltration': 0.9835011576635112}
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_detectability_metrics.csv (90 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_families.csv (275 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_scores.csv (54 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_sample_size_control.csv (11 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_tau_per_seed.csv (1 filas)
```

## E22 — OE2 robusto: detectabilidad, tablas de todas las semillas

- Inicio: 2026-10-10T23:14:52Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase4_detectability`
- Fin: 2026-10-10T23:14:56Z (4 s = 0 min)
- Estado: **FALLO** (exit 132)

Log completo: `outputs/vm_pendientes/logs/E22.log`. Últimas líneas:

```
===== [2026-10-10T23:14:52Z] E22: OE2 robusto: detectabilidad, tablas de todas las semillas =====
[fase4 seed=42] ya cacheado en phase4_seed42.json, se salta el cómputo
```

## E23 — OE2 robusto: correlación geometría-detectabilidad

- **SALTADA**: la etapa E22 no terminó bien (FALLO).

## E24 — OE2 robusto: figuras

- **SALTADA**: la etapa E23 no terminó bien (SALTADA).

## E25 — OE2 robusto: UMAP (apoyo visual)

- Inicio: 2026-10-10T23:14:56Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -c from vae_nids.evaluation.oe2_robust import config as rcfg, phase6_figures as p6; [p6.fig_umap(s) for s in rcfg.UMAP_SEEDS]`
- Fin: 2026-10-11T03:37:40Z (15764 s = 262 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E25.log`. Últimas líneas:

```
===== [2026-10-10T23:14:56Z] E25: OE2 robusto: UMAP (apoyo visual) =====
/home/estudiante/vae-latent-geometry-nids/.venv-linux/lib/python3.10/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
[fase6] fig_umap_seed42.png / .pdf
/home/estudiante/vae-latent-geometry-nids/.venv-linux/lib/python3.10/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
[fase6] fig_umap_seed43.png / .pdf
/home/estudiante/vae-latent-geometry-nids/.venv-linux/lib/python3.10/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
[fase6] fig_umap_seed44.png / .pdf
```

## E30 — Descriptores: detectabilidad y referencia benigna

- Inicio: 2026-10-11T03:37:40Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase1_detectabilidad`
- Fin: 2026-10-11T03:37:45Z (5 s = 0 min)
- Estado: **FALLO** (exit 132)

Log completo: `outputs/vm_pendientes/logs/E30.log`. Últimas líneas:

```
===== [2026-10-11T03:37:40Z] E30: Descriptores: detectabilidad y referencia benigna =====
```

## E31 — Descriptores: 17 descriptores por familia y semilla

- **SALTADA**: la etapa E30 no terminó bien (FALLO).

## E32 — Descriptores: cribado y validación (attempted)

- **SALTADA**: la etapa E31 no terminó bien (SALTADA).

## E33 — Descriptores: duplicados, métrica del decodificador, UMAP sin duplicados

- **SALTADA**: la etapa E30 no terminó bien (FALLO).

## E34 — Descriptores: figuras

- **SALTADA**: la etapa E32 no terminó bien (SALTADA).

## E35 — Descriptores: tablas LaTeX y cifras

- **SALTADA**: la etapa E32 no terminó bien (SALTADA).

## E36 — Prueba confirmatoria (primario + 6 secundarios, Holm)

- **SALTADA**: la etapa E31 no terminó bien (SALTADA).

## Fin de la corrida — 2026-10-11T03:37:45Z

- E01: OK
- E10: OK
- E11: OK
- E12: OK
- E13: OK
- E20: OK
- E21: OK
- E22_s42: OK
- E22_s43: FALLO
- E22_s44: OK
- E22_s45: OK
- E22_s46: OK
- E22_s47: FALLO
- E22_s48: FALLO
- E22_s49: OK
- E22: FALLO
- E23: SALTADA
- E24: SALTADA
- E25: OK
- E30: FALLO
- E31: SALTADA
- E32: SALTADA
- E33: SALTADA
- E34: SALTADA
- E35: SALTADA
- E36: SALTADA

## E99 — Resumen de la corrida

- Inicio: 2026-10-11T03:37:45Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python scripts/vm_resumen_corrida.py`
- Fin: 2026-10-11T03:37:46Z (1 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E99.log`. Últimas líneas:

```
===== [2026-10-11T03:37:45Z] E99: Resumen de la corrida =====
[resumen] /home/estudiante/vae-latent-geometry-nids/outputs/vm_pendientes/RESUMEN_CORRIDA.md
```
