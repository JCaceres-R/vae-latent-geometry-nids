
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

# Corrida de pendientes (OE1 saneado + OE2 recalculado) — 2026-10-11T14:35:11Z

- Host: tesis-juan; `uname -a`: Linux tesis-juan 5.15.0-194-generic #204-Ubuntu SMP Wed Sep 2 13:09:23 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux
- CPUs: 8; memoria: 23 GB total, 22 GB disponibles
- Disco libre en el repo: 58G
- Commit: c818eead44d7b628d603588314db7fde60af0f80 (rama main)
- Cambios sin commit en archivos versionados: 23

## Entorno

- Python: Python 3.10.12
- Dependencias: SIN versiones exactas (requirements.txt falló; ver E00_entorno.log y pip_freeze.txt)
- torch: 2.13.0+cpu | hilos: 8

## E01 — Tests (OE1 saneado y estadística del OE2)

- Inicio: 2026-10-11T14:38:21Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m pytest tests/oe1_saneado tests/oe2_robust/test_stats_utils.py -q -p no:cacheprovider`
- Fin: 2026-10-11T14:38:35Z (14 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E01.log`. Últimas líneas:

```
===== [2026-10-10T19:53:58Z] E01: Tests (OE1 saneado y estadística del OE2) =====
........................                                                 [100%]
24 passed in 33.85s
===== [2026-10-11T14:38:21Z] E01: Tests (OE1 saneado y estadística del OE2) (intento 1) =====
........................                                                 [100%]
24 passed in 13.01s
```

## E10 — OE1: datos, preproceso, 8 semillas + referencia 74, diagnósticos

- Inicio: 2026-10-11T14:38:35Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.oe1.run --config configs/oe1_saneado.yaml --ref74 configs/oe1_saneado_ref74.yaml`
- Fin: 2026-10-11T14:39:29Z (54 s = 0 min)
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
===== [2026-10-11T14:38:35Z] E10: OE1: datos, preproceso, 8 semillas + referencia 74, diagnósticos (intento 1) =====
[2026-10-11T14:38:38] preproceso
[2026-10-11T14:39:11] oe1_info_nula_seed42: ya existe, se salta
[2026-10-11T14:39:11] oe1_info_nula_seed43: ya existe, se salta
[2026-10-11T14:39:11] oe1_info_nula_seed44: ya existe, se salta
[2026-10-11T14:39:11] oe1_info_nula_seed45: ya existe, se salta
[2026-10-11T14:39:11] oe1_info_nula_seed46: ya existe, se salta
[2026-10-11T14:39:11] oe1_info_nula_seed47: ya existe, se salta
[2026-10-11T14:39:11] oe1_info_nula_seed48: ya existe, se salta
[2026-10-11T14:39:11] oe1_info_nula_seed49: ya existe, se salta
[2026-10-11T14:39:11] referencia_74_seed42: ya existe, se salta
[2026-10-11T14:39:11] diagnósticos
[2026-10-11T14:39:28] listo: {"corridas": 9}
```

## E11 — OE1: verificación de checkpoints y diagnósticos

- Inicio: 2026-10-11T14:39:29Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -`
- Fin: 2026-10-11T14:39:30Z (1 s = 0 min)
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
===== [2026-10-11T14:39:29Z] E11: OE1: verificación de checkpoints y diagnósticos (intento 1) =====
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

- Inicio: 2026-10-11T14:39:30Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.oe1.figuras_informe --config configs/oe1_saneado.yaml`
- Fin: 2026-10-11T14:39:37Z (7 s = 0 min)
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
===== [2026-10-11T14:39:30Z] E12: OE1: figuras del informe (selección y preproceso) (intento 1) =====
                                                                    caso  distintos   crudo      frac
0  Flow Packets/s, pipeline original (float32, con duraciones negativas)      10084  577678  0.017456
1                                    Flow Packets/s, saneado (filtro D4)     573049  577678  0.991987
2                                     Flow Duration, saneado (filtro D4)     517280  524335  0.986545
/home/estudiante/vae-latent-geometry-nids/src/vae_nids/oe1/figuras_informe.py:65: RuntimeWarning: invalid value encountered in divide
  ratio = np.where(tol > 0, np.abs(a - b) / tol, 0.0)
```

## E13 — OE1: tablas LaTeX y cifras del informe

- Inicio: 2026-10-11T14:39:37Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.oe1.tablas_informe --config configs/oe1_saneado.yaml --dest outputs/oe1_saneado/informe_tablas`
- Fin: 2026-10-11T14:39:37Z (0 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E13.log`. Últimas líneas:

```
===== [2026-10-10T21:50:28Z] E13: OE1: tablas LaTeX y cifras del informe =====
{'nUniverso': '77', 'nQuedan': '66', 'nRcero': '3', 'nRuno': '0', 'nRdos': '8', 'nCasi': '6', 'hashCorto': '49435ba7a103', 'hashLargo': '49435ba7a10312a12aef49ee6c0f75b7916cf92813ba8cf3cf12291739405d97', 'hashPartido': '49435ba7a10312a1\\allowbreak{}2aef49ee6c0f75b7\\allowbreak{}916cf92813ba8cf3\\allowbreak{}cf12291739405d97', 'nAlternativo': '60', 'casiFallosMin': '0{,}006', 'casiFallosMax': '3{,}6', 'casiUlpMin': '8', 'casiUlpMax': '100', 'nCasiConst': '6', 'postComprobaciones': '10\\,340\\,726', 'postFallos': '0', 'postEtiquetas': '24', 'nComunActual': '45', 'nVuelven': '21', 'nSalenActual': '6', 'filasCrudas': '2\\,100\\,814', 'filasInf': '793', 'dcuatroTrain': '24', 'dcuatroVal': '7', 'dcuatroTotal': '31', 'iatNegTrain': '1\\,873', 'nMarcadas': '18', 'minConservado': '90{,}6', 'fpsAntes': '1{,}7', 'fpsDespues': '99{,}2', 'nDependencias': '12', 'activasMedia': '6{,}9', 'activasDE': '0{,}4', 'activasAntes': '5{,}4', 'epocaMin': '38', 'epocaMax': '112', 'pisoFrac': '90{,}5', 'nSemillasNuevo': '8', 'reentrenamiento': 'completo'}
===== [2026-10-11T14:39:37Z] E13: OE1: tablas LaTeX y cifras del informe (intento 1) =====
{'nUniverso': '77', 'nQuedan': '66', 'nRcero': '3', 'nRuno': '0', 'nRdos': '8', 'nCasi': '6', 'hashCorto': '49435ba7a103', 'hashLargo': '49435ba7a10312a12aef49ee6c0f75b7916cf92813ba8cf3cf12291739405d97', 'hashPartido': '49435ba7a10312a1\\allowbreak{}2aef49ee6c0f75b7\\allowbreak{}916cf92813ba8cf3\\allowbreak{}cf12291739405d97', 'nAlternativo': '60', 'casiFallosMin': '0{,}006', 'casiFallosMax': '3{,}6', 'casiUlpMin': '8', 'casiUlpMax': '100', 'nCasiConst': '6', 'postComprobaciones': '10\\,340\\,726', 'postFallos': '0', 'postEtiquetas': '24', 'nComunActual': '45', 'nVuelven': '21', 'nSalenActual': '6', 'filasCrudas': '2\\,100\\,814', 'filasInf': '793', 'dcuatroTrain': '24', 'dcuatroVal': '7', 'dcuatroTotal': '31', 'iatNegTrain': '1\\,873', 'nMarcadas': '18', 'minConservado': '90{,}6', 'fpsAntes': '1{,}7', 'fpsDespues': '99{,}2', 'nDependencias': '12', 'activasMedia': '6{,}9', 'activasDE': '0{,}4', 'activasAntes': '5{,}4', 'epocaMin': '38', 'epocaMax': '112', 'pisoFrac': '90{,}5', 'nSemillasNuevo': '8', 'reentrenamiento': 'completo'}
```

## E20 — OE2 robusto: codificación (mu, log sigma^2) por semilla

- Inicio: 2026-10-11T14:39:37Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase2_encode`
- Fin: 2026-10-11T14:39:40Z (3 s = 0 min)
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
===== [2026-10-11T14:39:37Z] E20: OE2 robusto: codificación (mu, log sigma^2) por semilla (intento 1) =====
[fase2 seed=42] ya codificado (/home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/latent_vectors/seed42/manifest.json), se reutiliza
[fase2 seed=43] ya codificado (/home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/latent_vectors/seed43/manifest.json), se reutiliza
[fase2 seed=44] ya codificado (/home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/latent_vectors/seed44/manifest.json), se reutiliza
[fase2 seed=45] ya codificado (/home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/latent_vectors/seed45/manifest.json), se reutiliza
[fase2 seed=46] ya codificado (/home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/latent_vectors/seed46/manifest.json), se reutiliza
[fase2 seed=47] ya codificado (/home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/latent_vectors/seed47/manifest.json), se reutiliza
[fase2 seed=48] ya codificado (/home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/latent_vectors/seed48/manifest.json), se reutiliza
[fase2 seed=49] ya codificado (/home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/latent_vectors/seed49/manifest.json), se reutiliza
[fase2] tabla semilla x dimensión guardada en /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase2_active_dims_per_seed.csv
```

## E21 — OE2 robusto: geometría distribucional

- Inicio: 2026-10-11T14:39:40Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase3_geometry`
- Fin: 2026-10-11T14:39:43Z (3 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E21.log`. Últimas líneas:

```
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
===== [2026-10-11T14:39:40Z] E21: OE2 robusto: geometría distribucional (intento 1) =====
[fase3 seed=42] ya cacheado en phase3_seed42.json, se salta el cómputo
[fase3 seed=43] ya cacheado en phase3_seed43.json, se salta el cómputo
[fase3 seed=44] ya cacheado en phase3_seed44.json, se salta el cómputo
[fase3 seed=45] ya cacheado en phase3_seed45.json, se salta el cómputo
[fase3 seed=46] ya cacheado en phase3_seed46.json, se salta el cómputo
[fase3 seed=47] ya cacheado en phase3_seed47.json, se salta el cómputo
[fase3 seed=48] ya cacheado en phase3_seed48.json, se salta el cómputo
[fase3 seed=49] ya cacheado en phase3_seed49.json, se salta el cómputo
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

- Inicio: 2026-10-11T14:39:46Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed 42`
- Fin: 2026-10-11T14:39:50Z (4 s = 0 min)
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
===== [2026-10-11T14:39:46Z] E22_s42: OE2 robusto: detectabilidad, semilla 42 (intento 1) =====
[fase4 seed=42] ya cacheado en phase4_seed42.json, se salta el cómputo
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_detectability_metrics.csv (90 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_families.csv (275 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_scores.csv (54 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_sample_size_control.csv (11 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_tau_per_seed.csv (1 filas)
```

## E22_s43 — OE2 robusto: detectabilidad, semilla 43

- Inicio: 2026-10-11T14:39:50Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed 43`
- Fin: 2026-10-11T14:50:28Z (638 s = 10 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E22_s43.log`. Últimas líneas:

```
===== [2026-10-10T22:30:27Z] E22_s43: OE2 robusto: detectabilidad, semilla 43 =====
===== [2026-10-11T14:39:50Z] E22_s43: OE2 robusto: detectabilidad, semilla 43 (intento 1) =====
[fase4 seed=43] tau_p95=-193.8532 fpr_test_benign=0.0513 AUC(nll_mc) principales: {'PortScan': 0.6605693569433858, 'DoS_Hulk': 0.9210831208560054, 'DDoS': 0.9798902125726776, 'DoS_GoldenEye': 0.954163904184169, 'DoS_slowloris': 0.9425399382147817, 'FTP_Patator': 0.9091158747505487, 'SSH_Patator': 0.9125628115432428, 'DoS_Slowhttptest': 0.9922691533707977, 'Bot': 0.6093326275472514, 'Web_Attack': 0.9098446447223905, 'Infiltration': 0.9929365930294777}
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_detectability_metrics.csv (90 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_families.csv (275 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_scores.csv (54 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_sample_size_control.csv (11 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_tau_per_seed.csv (1 filas)
```

## E22_s44 — OE2 robusto: detectabilidad, semilla 44

- Inicio: 2026-10-11T14:50:28Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed 44`
- Fin: 2026-10-11T14:50:32Z (4 s = 0 min)
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
===== [2026-10-11T14:50:28Z] E22_s44: OE2 robusto: detectabilidad, semilla 44 (intento 1) =====
[fase4 seed=44] ya cacheado en phase4_seed44.json, se salta el cómputo
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_detectability_metrics.csv (90 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_families.csv (275 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_scores.csv (54 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_sample_size_control.csv (11 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_tau_per_seed.csv (1 filas)
```

## E22_s45 — OE2 robusto: detectabilidad, semilla 45

- Inicio: 2026-10-11T14:50:32Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed 45`
- Fin: 2026-10-11T14:50:35Z (3 s = 0 min)
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
===== [2026-10-11T14:50:32Z] E22_s45: OE2 robusto: detectabilidad, semilla 45 (intento 1) =====
[fase4 seed=45] ya cacheado en phase4_seed45.json, se salta el cómputo
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_detectability_metrics.csv (90 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_families.csv (275 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_scores.csv (54 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_sample_size_control.csv (11 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_tau_per_seed.csv (1 filas)
```

## E22_s46 — OE2 robusto: detectabilidad, semilla 46

- Inicio: 2026-10-11T14:50:35Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed 46`
- Fin: 2026-10-11T14:50:39Z (4 s = 0 min)
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
===== [2026-10-11T14:50:35Z] E22_s46: OE2 robusto: detectabilidad, semilla 46 (intento 1) =====
[fase4 seed=46] ya cacheado en phase4_seed46.json, se salta el cómputo
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_detectability_metrics.csv (90 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_families.csv (275 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_scores.csv (54 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_sample_size_control.csv (11 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_tau_per_seed.csv (1 filas)
```

## E22_s47 — OE2 robusto: detectabilidad, semilla 47

- Inicio: 2026-10-11T14:50:39Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed 47`
- Fin: 2026-10-11T15:01:17Z (638 s = 10 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E22_s47.log`. Últimas líneas:

```
===== [2026-10-10T23:03:58Z] E22_s47: OE2 robusto: detectabilidad, semilla 47 =====
===== [2026-10-11T14:50:39Z] E22_s47: OE2 robusto: detectabilidad, semilla 47 (intento 1) =====
[fase4 seed=47] tau_p95=-218.0228 fpr_test_benign=0.0499 AUC(nll_mc) principales: {'PortScan': 0.7006591069605311, 'DoS_Hulk': 0.9421651002726052, 'DDoS': 0.9937259323228679, 'DoS_GoldenEye': 0.9810019731138178, 'DoS_slowloris': 0.9420576616057764, 'FTP_Patator': 0.95045395973272, 'SSH_Patator': 0.9761145556507823, 'DoS_Slowhttptest': 0.977284738902011, 'Bot': 0.6265687056213178, 'Web_Attack': 0.9372718419856272, 'Infiltration': 0.9853061723279195}
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_detectability_metrics.csv (90 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_families.csv (275 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_scores.csv (54 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_sample_size_control.csv (11 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_tau_per_seed.csv (1 filas)
```

## E22_s48 — OE2 robusto: detectabilidad, semilla 48

- Inicio: 2026-10-11T15:01:17Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed 48`
- Fin: 2026-10-11T15:12:08Z (651 s = 10 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E22_s48.log`. Últimas líneas:

```
===== [2026-10-10T23:04:02Z] E22_s48: OE2 robusto: detectabilidad, semilla 48 =====
===== [2026-10-11T15:01:17Z] E22_s48: OE2 robusto: detectabilidad, semilla 48 (intento 1) =====
[fase4 seed=48] tau_p95=-199.8864 fpr_test_benign=0.0506 AUC(nll_mc) principales: {'PortScan': 0.6857133686325084, 'DoS_Hulk': 0.9191364150716627, 'DDoS': 0.9810459675023081, 'DoS_GoldenEye': 0.9579251030036329, 'DoS_slowloris': 0.9391678893614167, 'FTP_Patator': 0.9515980492651013, 'SSH_Patator': 0.9786976156906806, 'DoS_Slowhttptest': 0.9925576497770074, 'Bot': 0.6028777224359239, 'Web_Attack': 0.9218206840680218, 'Infiltration': 0.9916588634178329}
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_detectability_metrics.csv (90 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_families.csv (275 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_scores.csv (54 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_sample_size_control.csv (11 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_tau_per_seed.csv (1 filas)
```

## E22_s49 — OE2 robusto: detectabilidad, semilla 49

- Inicio: 2026-10-11T15:12:08Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed 49`
- Fin: 2026-10-11T15:12:11Z (3 s = 0 min)
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
===== [2026-10-11T15:12:08Z] E22_s49: OE2 robusto: detectabilidad, semilla 49 (intento 1) =====
[fase4 seed=49] ya cacheado en phase4_seed49.json, se salta el cómputo
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_detectability_metrics.csv (90 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_families.csv (275 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_scores.csv (54 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_sample_size_control.csv (11 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_tau_per_seed.csv (1 filas)
```

## E22 — OE2 robusto: detectabilidad, tablas de todas las semillas

- Inicio: 2026-10-11T15:12:11Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase4_detectability`
- Fin: 2026-10-11T15:12:15Z (4 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E22.log`. Últimas líneas:

```
===== [2026-10-10T23:14:52Z] E22: OE2 robusto: detectabilidad, tablas de todas las semillas =====
[fase4 seed=42] ya cacheado en phase4_seed42.json, se salta el cómputo
===== [2026-10-11T15:12:11Z] E22: OE2 robusto: detectabilidad, tablas de todas las semillas (intento 1) =====
[fase4 seed=42] ya cacheado en phase4_seed42.json, se salta el cómputo
[fase4 seed=43] ya cacheado en phase4_seed43.json, se salta el cómputo
[fase4 seed=44] ya cacheado en phase4_seed44.json, se salta el cómputo
[fase4 seed=45] ya cacheado en phase4_seed45.json, se salta el cómputo
[fase4 seed=46] ya cacheado en phase4_seed46.json, se salta el cómputo
[fase4 seed=47] ya cacheado en phase4_seed47.json, se salta el cómputo
[fase4 seed=48] ya cacheado en phase4_seed48.json, se salta el cómputo
[fase4 seed=49] ya cacheado en phase4_seed49.json, se salta el cómputo
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_detectability_metrics.csv (720 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_families.csv (2200 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_delong_between_scores.csv (432 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_sample_size_control.csv (88 filas)
[fase4] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase4_tau_per_seed.csv (8 filas)
```

## E23 — OE2 robusto: correlación geometría-detectabilidad

- Inicio: 2026-10-11T15:12:15Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_robust.phase5_correlation`
- Fin: 2026-10-11T15:20:28Z (493 s = 8 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E23.log`. Últimas líneas:

```
===== [2026-10-11T15:12:15Z] E23: OE2 robusto: correlación geometría-detectabilidad (intento 1) =====
[fase5 7.1] rho combinado=0.7593 IC95=[0.6191,0.8526] (media por semilla=0.7182, sd=0.1789)
/home/estudiante/vae-latent-geometry-nids/src/vae_nids/evaluation/oe2_robust/phase5_correlation.py:163: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  lambda idx: sp_stats.spearmanr(x_avg[idx], y_avg[idx])[0],
/home/estudiante/vae-latent-geometry-nids/src/vae_nids/evaluation/oe2_robust/phase5_correlation.py:163: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  lambda idx: sp_stats.spearmanr(x_avg[idx], y_avg[idx])[0],
/home/estudiante/vae-latent-geometry-nids/src/vae_nids/evaluation/oe2_robust/phase5_correlation.py:163: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  lambda idx: sp_stats.spearmanr(x_avg[idx], y_avg[idx])[0],
[fase5 7.2] 81 pares geometría x detectabilidad evaluados; 47 con p crudo <0.05, 6 tras corrección de Holm
[fase5] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase5_secondary_per_seed.csv (648 filas)
[fase5] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase5_secondary_summary.csv (81 filas)
[fase5] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase5_ranking_correlation.csv (8 filas)
[fase5 7.3] datos a nivel de conexión: 309,248 filas, 12 grupos, 8 semillas
/home/estudiante/vae-latent-geometry-nids/src/vae_nids/evaluation/oe2_robust/phase5_correlation.py:249: ConvergenceWarning: The Hessian matrix at the estimated parameter values is not positive definite.
  result = model.fit(reml=True, method="lbfgs", maxiter=200)
[fase5 7.3] modelo mixto: convergió=True efecto_fijo(maha_std)=0.103590690890833
[fase5] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase5_connection_level_spearman.csv (88 filas)
[fase5 7.4] W de Kendall (auc_roc_nll_mc): 0.8866
[fase5 7.4] W de Kendall (mahalanobis_median): 0.7983
[fase5 7.4] W de Kendall (mmd2_rbf): 0.8537
[fase5 7.4] W de Kendall (sliced_wasserstein): 0.8608
[fase5 7.4] W de Kendall (energy_distance): 0.8051
[fase5 7.4] W de Kendall (bhattacharyya_total): 0.9173
[fase5] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase5_seed_stability_kendalls_w.csv (6 filas)
[fase5] /home/estudiante/vae-latent-geometry-nids/outputs/oe2_robust_saneado/tables/phase5_attempted_comparison.csv (48 filas)
```

## E24 — OE2 robusto: figuras

- Inicio: 2026-10-11T15:20:28Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -c from vae_nids.evaluation.oe2_robust import phase6_figures as p6; p6.main(include_umap=False)`
- Fin: 2026-10-11T15:20:38Z (10 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E24.log`. Últimas líneas:

```
===== [2026-10-11T15:20:28Z] E24: OE2 robusto: figuras (intento 1) =====
[fase6] fig1_mahalanobis_distributions.png / .pdf
[fase6] fig2_forest_auc.png / .pdf
[fase6] fig3_scatter_geometry_detectability.png / .pdf
[fase6] fig4_rho_distribution.png / .pdf
[fase6] fig5_ranking_heatmap.png / .pdf
[fase6] fig6_mixed_model_coefficients.png / .pdf
[fase6] fig7_score_comparison.png / .pdf
[fase6] UMAP omitido en esta corrida (include_umap=False) -- pendiente, ver informe sección 4.7 / 10.
```

## E25 — OE2 robusto: UMAP (apoyo visual)

- Inicio: 2026-10-11T15:20:38Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -c from vae_nids.evaluation.oe2_robust import config as rcfg, phase6_figures as p6; [p6.fig_umap(s) for s in rcfg.UMAP_SEEDS if not (rcfg.FIGURES_DIR / f'fig_umap_seed{s}.png').exists()]`
- Fin: 2026-10-11T15:20:41Z (3 s = 0 min)
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
===== [2026-10-11T15:20:38Z] E25: OE2 robusto: UMAP (apoyo visual) (intento 1) =====
```

## E30 — Descriptores: detectabilidad y referencia benigna

- Inicio: 2026-10-11T15:20:41Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase1_detectabilidad`
- Fin: 2026-10-11T15:24:31Z (230 s = 3 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E30.log`. Últimas líneas:

```
[fase1 seed=47] Bot_attempted                    n=   1470 AUC=0.6259
[fase1 seed=47] WebAttack_BruteForce             n=    151 AUC=0.9516
[fase1 seed=47] WebAttack_BruteForce_attempted   n=   1214 AUC=0.6686
[fase1 seed=48] PortScan                         n= 159023 AUC=0.6857
[fase1 seed=48] DoS_Hulk                         n= 158469 AUC=0.9191
[fase1 seed=48] DDoS                             n=  95123 AUC=0.9810
[fase1 seed=48] DoS_GoldenEye                    n=   7567 AUC=0.9579
[fase1 seed=48] DoS_slowloris                    n=   4001 AUC=0.9392
[fase1 seed=48] FTP_Patator                      n=   3973 AUC=0.9516
[fase1 seed=48] SSH_Patator                      n=   2980 AUC=0.9787
[fase1 seed=48] DoS_Slowhttptest                 n=   1742 AUC=0.9926
[fase1 seed=48] Bot                              n=    738 AUC=0.6029
[fase1 seed=48] Web_Attack                       n=    190 AUC=0.9218
[fase1 seed=48] Infiltration                     n=     32 AUC=0.9917
[fase1 seed=48] DoS_Hulk_attempted               n=    579 AUC=0.9159
[fase1 seed=48] DoS_GoldenEye_attempted          n=     80 AUC=0.7307
[fase1 seed=48] DoS_slowloris_attempted          n=   1706 AUC=0.9370
[fase1 seed=48] DoS_Slowhttptest_attempted       n=   3367 AUC=0.9768
[fase1 seed=48] Bot_attempted                    n=   1470 AUC=0.6126
[fase1 seed=48] WebAttack_BruteForce             n=    151 AUC=0.9369
[fase1 seed=48] WebAttack_BruteForce_attempted   n=   1214 AUC=0.7427
[fase1 seed=49] PortScan                         n= 159023 AUC=0.6856
[fase1 seed=49] DoS_Hulk                         n= 158469 AUC=0.9503
[fase1 seed=49] DDoS                             n=  95123 AUC=0.9957
[fase1 seed=49] DoS_GoldenEye                    n=   7567 AUC=0.9700
[fase1 seed=49] DoS_slowloris                    n=   4001 AUC=0.9352
[fase1 seed=49] FTP_Patator                      n=   3973 AUC=0.8540
[fase1 seed=49] SSH_Patator                      n=   2980 AUC=0.9331
[fase1 seed=49] DoS_Slowhttptest                 n=   1742 AUC=0.9686
[fase1 seed=49] Bot                              n=    738 AUC=0.6298
[fase1 seed=49] Web_Attack                       n=    190 AUC=0.9434
[fase1 seed=49] Infiltration                     n=     32 AUC=0.9835
[fase1 seed=49] DoS_Hulk_attempted               n=    579 AUC=0.9163
[fase1 seed=49] DoS_GoldenEye_attempted          n=     80 AUC=0.7339
[fase1 seed=49] DoS_slowloris_attempted          n=   1706 AUC=0.9780
[fase1 seed=49] DoS_Slowhttptest_attempted       n=   3367 AUC=0.9855
[fase1 seed=49] Bot_attempted                    n=   1470 AUC=0.5930
[fase1 seed=49] WebAttack_BruteForce             n=    151 AUC=0.9653
[fase1 seed=49] WebAttack_BruteForce_attempted   n=   1214 AUC=0.6603
[fase1] diferencia máxima de AUC vs phase4 (seed 42): 1.11e-16
```

## E31 — Descriptores: 17 descriptores por familia y semilla

- Inicio: 2026-10-11T15:24:31Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase2_descriptores`
- Fin: 2026-10-11T16:21:35Z (3424 s = 57 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E31.log`. Últimas líneas:

```
[fase2 seed=47] WebAttack_BruteForce             n=    151 únicos=    151 AUC_knn=0.960 sil_w=0.195  (404s)
[fase2 seed=47] WebAttack_BruteForce_attempted   n=   1214 únicos=   1214 AUC_knn=0.534 sil_w=0.903  (420s)
[fase2 seed=48] BENIGN_control                   n=  50000 únicos=  47780 AUC_knn=0.500 sil_w=0.002  (44s)
[fase2 seed=48] PortScan                         n= 159023 únicos=   1670 AUC_knn=0.776 sil_w=0.201  (65s)
[fase2 seed=48] DoS_Hulk                         n= 158469 únicos= 158469 AUC_knn=0.973 sil_w=0.486  (160s)
[fase2 seed=48] DDoS                             n=  95123 únicos=  95123 AUC_knn=0.992 sil_w=0.416  (225s)
[fase2 seed=48] DoS_GoldenEye                    n=   7567 únicos=   7567 AUC_knn=0.980 sil_w=0.029  (246s)
[fase2 seed=48] DoS_slowloris                    n=   4001 únicos=   3595 AUC_knn=0.953 sil_w=0.235  (264s)
[fase2 seed=48] FTP_Patator                      n=   3973 únicos=   3973 AUC_knn=0.954 sil_w=0.954  (283s)
[fase2 seed=48] SSH_Patator                      n=   2980 únicos=   2980 AUC_knn=0.964 sil_w=0.905  (302s)
[fase2 seed=48] DoS_Slowhttptest                 n=   1742 únicos=   1731 AUC_knn=0.985 sil_w=0.160  (320s)
[fase2 seed=48] Bot                              n=    738 únicos=    738 AUC_knn=0.851 sil_w=0.744  (332s)
[fase2 seed=48] Web_Attack                       n=    190 únicos=    190 AUC_knn=0.964 sil_w=0.223  (339s)
[fase2 seed=48] Infiltration                     n=     32 únicos=     32 AUC_knn=0.981 sil_w=-0.623  (345s)
[fase2 seed=48] DoS_Hulk_attempted               n=    579 únicos=    573 AUC_knn=0.920 sil_w=0.054  (356s)
[fase2 seed=48] DoS_GoldenEye_attempted          n=     80 únicos=     77 AUC_knn=0.698 sil_w=0.935  (363s)
[fase2 seed=48] DoS_slowloris_attempted          n=   1706 únicos=   1703 AUC_knn=0.961 sil_w=0.889  (380s)
[fase2 seed=48] DoS_Slowhttptest_attempted       n=   3367 únicos=   3361 AUC_knn=0.256 sil_w=0.655  (398s)
[fase2 seed=48] Bot_attempted                    n=   1470 únicos=    713 AUC_knn=0.133 sil_w=0.976  (409s)
[fase2 seed=48] WebAttack_BruteForce             n=    151 únicos=    151 AUC_knn=0.971 sil_w=0.336  (416s)
[fase2 seed=48] WebAttack_BruteForce_attempted   n=   1214 únicos=   1214 AUC_knn=0.567 sil_w=0.925  (432s)
[fase2 seed=49] BENIGN_control                   n=  50000 únicos=  47727 AUC_knn=0.500 sil_w=0.014  (44s)
[fase2 seed=49] PortScan                         n= 159023 únicos=   1670 AUC_knn=0.783 sil_w=0.239  (64s)
[fase2 seed=49] DoS_Hulk                         n= 158469 únicos= 158469 AUC_knn=0.994 sil_w=0.687  (153s)
[fase2 seed=49] DDoS                             n=  95123 únicos=  95123 AUC_knn=0.997 sil_w=0.555  (214s)
[fase2 seed=49] DoS_GoldenEye                    n=   7567 únicos=   7567 AUC_knn=0.985 sil_w=0.176  (234s)
[fase2 seed=49] DoS_slowloris                    n=   4001 únicos=   3596 AUC_knn=0.958 sil_w=0.140  (253s)
[fase2 seed=49] FTP_Patator                      n=   3973 únicos=   3973 AUC_knn=0.982 sil_w=0.942  (270s)
[fase2 seed=49] SSH_Patator                      n=   2980 únicos=   2980 AUC_knn=0.987 sil_w=0.920  (289s)
[fase2 seed=49] DoS_Slowhttptest                 n=   1742 únicos=   1731 AUC_knn=0.991 sil_w=0.179  (307s)
[fase2 seed=49] Bot                              n=    738 únicos=    738 AUC_knn=0.787 sil_w=0.682  (319s)
[fase2 seed=49] Web_Attack                       n=    190 únicos=    190 AUC_knn=0.965 sil_w=0.233  (326s)
[fase2 seed=49] Infiltration                     n=     32 únicos=     32 AUC_knn=0.992 sil_w=-0.368  (333s)
[fase2 seed=49] DoS_Hulk_attempted               n=    579 únicos=    573 AUC_knn=0.941 sil_w=0.312  (344s)
[fase2 seed=49] DoS_GoldenEye_attempted          n=     80 únicos=     77 AUC_knn=0.820 sil_w=0.946  (351s)
[fase2 seed=49] DoS_slowloris_attempted          n=   1706 únicos=   1703 AUC_knn=0.978 sil_w=0.905  (368s)
[fase2 seed=49] DoS_Slowhttptest_attempted       n=   3367 únicos=   3361 AUC_knn=0.906 sil_w=0.621  (387s)
[fase2 seed=49] Bot_attempted                    n=   1470 únicos=    713 AUC_knn=0.121 sil_w=0.986  (398s)
[fase2 seed=49] WebAttack_BruteForce             n=    151 únicos=    151 AUC_knn=0.970 sil_w=0.354  (405s)
[fase2 seed=49] WebAttack_BruteForce_attempted   n=   1214 únicos=   1214 AUC_knn=0.511 sil_w=0.928  (422s)
```

## E32 — Descriptores: cribado y validación (attempted)

- Inicio: 2026-10-11T16:21:35Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase3_evaluacion`
- Fin: 2026-10-11T16:22:24Z (49 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E32.log`. Últimas líneas:

```
2.4      Fisher-Rao al k-ésimo posterior benigno          rho=+0.78 [+0.43,+0.91] p=0.000 W=0.79 rhoM=+0.78 val=+0.66
7.1      Tasa KL(q||p) por flujo                          rho=+0.83 [+0.52,+0.97] p=0.000 W=0.90 rhoM=+0.89 val=+0.95
3.2      MMD² analítico entre posteriores agregados       rho=+0.44 [-0.27,+0.89] p=0.115 W=0.77 rhoM=+0.62 val=+0.85
6.2      kNN en el espacio decodificado                   rho=+0.68 [+0.25,+0.88] p=0.003 W=0.68 rhoM=+0.66 val=+0.64
6.1      Log-magnificación del decodificador              rho=+0.59 [+0.11,+0.89] p=0.013 W=0.68 rhoM=+0.48 val=+0.94
4.2      Distancia afín-invariante entre covarianzas      rho=-0.14 [-0.67,+0.40] p=0.634 W=0.89 rhoM=-0.31 val=-0.46
8.1      LID relativa al benigno                          rho=+0.44 [-0.05,+0.82] p=0.059 W=0.56 rhoM=+0.46 val=+0.37
10.1a    Silhouette del grupo (z blanqueado)              rho=-0.41 [-0.83,+0.10] p=0.177 W=0.93 rhoM=-0.46 val=-0.75
10.3     Sonda lineal sobre (μ, log σ²)                   rho=+0.04 [-0.61,+0.64] p=0.909 W=0.92 rhoM=+0.08 val=-0.16
1.1b     Mahalanobis (AUC frente al control)              rho=+0.81 [+0.56,+0.98] p=0.000 W=0.81 rhoM=+0.88 val=+0.91
10.1b    Silhouette del grupo (Fisher-Rao)                rho=-0.63 [-0.92,-0.22] p=0.026 W=0.96 rhoM=-0.69 val=-0.69
1.2      Log-densidad de mezcla benigna (K=8)             rho=+0.74 [+0.39,+0.97] p=0.002 W=0.82 rhoM=+0.85 val=+0.86
1.3      Mahalanobis relativa al prior                    rho=+0.71 [+0.39,+0.94] p=0.000 W=0.62 rhoM=+0.74 val=+0.86
1.4      Tipicidad radial bilateral                       rho=+0.76 [+0.50,+0.94] p=0.000 W=0.72 rhoM=+0.82 val=+0.94
4.1      Rango efectivo de la familia                     rho=+0.14 [-0.47,+0.81] p=0.618 W=0.73 rhoM=+0.27 val=+0.13
7.2      Log-volumen del posterior                        rho=+0.56 [+0.05,+0.87] p=0.019 W=0.63 rhoM=+0.45 val=+0.89
11.1     Inconsistencia Kumar-Poole                       rho=+0.38 [+0.02,+0.69] p=0.069 W=0.44 rhoM=+0.41 val=+0.01
5.1m     kNN: fracción > p95 del control                  rho=+0.63 [+0.07,+0.90] p=0.010 W=0.73 rhoM=+0.47 val=+0.75
2.4m     Fisher-Rao (mediana)                             rho=+0.72 [+0.32,+0.93] p=0.002 W=0.77 rhoM=+0.83 val=+0.57
7.1m     Tasa (mediana, nats)                             rho=+0.79 [+0.45,+0.97] p=0.001 W=0.87 rhoM=+0.97 val=+0.88
6.1m     Log-magnificación (mediana)                      rho=+0.63 [+0.25,+0.87] p=0.007 W=0.69 rhoM=+0.53 val=+0.85
R        bhattacharyya cov term                           rho=-0.56 [-0.83,-0.16] p=0.048 W=0.92 rhoM=-0.71 val=-0.81
R        bhattacharyya mean term                          rho=+0.32 [-0.23,+0.81] p=0.252 W=0.74 rhoM=+0.36 val=+0.42
R        bhattacharyya total                              rho=-0.28 [-0.65,+0.22] p=0.346 W=0.92 rhoM=-0.40 val=-0.79
R        cliffs delta                                     rho=+0.80 [+0.51,+0.98] p=0.000 W=0.82 rhoM=+0.86 val=+0.91
R        cov logdet                                       rho=+0.66 [+0.24,+0.93] p=0.022 W=0.98 rhoM=+0.70 val=+0.79
R        energy distance                                  rho=+0.67 [+0.24,+0.93] p=0.009 W=0.81 rhoM=+0.66 val=+0.71
R        log density benign median                        rho=-0.74 [-0.94,-0.37] p=0.002 W=0.79 rhoM=-0.89 val=-0.45
R        mahalanobis median robust mcd                    rho=+0.65 [+0.35,+0.86] p=0.002 W=0.55 rhoM=+0.62 val=+0.28
R        mahalanobis p95                                  rho=+0.61 [+0.06,+0.90] p=0.025 W=0.89 rhoM=+0.70 val=+0.77
R        mean posterior variance sum                      rho=-0.14 [-0.81,+0.46] p=0.626 W=0.73 rhoM=-0.28 val=-0.89
R        mmd2 rbf                                         rho=+0.36 [-0.20,+0.84] p=0.212 W=0.85 rhoM=+0.41 val=+0.25
/home/estudiante/vae-latent-geometry-nids/src/vae_nids/evaluation/oe2_descriptores/fase3_evaluacion.py:77: RuntimeWarning: Mean of empty slice
  return float(np.tanh(np.nanmean(np.arctanh(r))))
/home/estudiante/vae-latent-geometry-nids/src/vae_nids/evaluation/oe2_descriptores/fase3_evaluacion.py:77: RuntimeWarning: Mean of empty slice
  return float(np.tanh(np.nanmean(np.arctanh(r))))
R        neighborhood purity median                       rho=+0.41 [+nan,+nan] p=0.279 W=0.25 rhoM=+0.18 val=+nan
R        posterior mahalanobis median                     rho=+0.76 [+0.46,+0.97] p=0.000 W=0.80 rhoM=+0.99 val=+0.77
R        silhouette mean                                  rho=+0.46 [-0.03,+0.86] p=0.102 W=0.84 rhoM=+0.31 val=+0.37
R        sliced wasserstein                               rho=+0.76 [+0.35,+0.95] p=0.002 W=0.86 rhoM=+0.76 val=+0.81
```

## E33 — Descriptores: duplicados, métrica del decodificador, UMAP sin duplicados

- Inicio: 2026-10-11T16:22:24Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase4_diagnosticos`
- Fin: 2026-10-11T16:24:07Z (103 s = 1 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E33.log`. Últimas líneas:

```
===== [2026-10-11T16:22:24Z] E33: Descriptores: duplicados, métrica del decodificador, UMAP sin duplicados (intento 1) =====
/home/estudiante/vae-latent-geometry-nids/.venv-linux/lib/python3.10/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
[fase4] listo
```

## E34 — Descriptores: figuras

- Inicio: 2026-10-11T16:24:07Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase5_figuras`
- Fin: 2026-10-11T16:24:20Z (13 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E34.log`. Últimas líneas:

```
===== [2026-10-11T16:24:07Z] E34: Descriptores: figuras (intento 1) =====
[fase5] figuras en /home/estudiante/vae-latent-geometry-nids/outputs/oe2_descriptores_saneado/figures
```

## E35 — Descriptores: tablas LaTeX y cifras

- Inicio: 2026-10-11T16:24:20Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase6_tablas_tex`
- Fin: 2026-10-11T16:24:23Z (3 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E35.log`. Últimas líneas:

```
===== [2026-10-11T16:24:20Z] E35: Descriptores: tablas LaTeX y cifras (intento 1) =====
[fase6] tablas en /home/estudiante/vae-latent-geometry-nids/outputs/oe2_descriptores_saneado/tablas_tex
```

## E36 — Prueba confirmatoria (primario + 6 secundarios, Holm)

- Inicio: 2026-10-11T16:24:23Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python -m vae_nids.evaluation.oe2_descriptores.fase7_confirmatoria`
- Fin: 2026-10-11T16:24:36Z (13 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E36.log`. Últimas líneas:

```
===== [2026-10-11T16:24:23Z] E36: Prueba confirmatoria (primario + 6 secundarios, Holm) (intento 1) =====
[fase7] 1.1    Mahalanobis (mediana)                      rho=+0.772 [+0.394,+0.976] p=0.0007 (51 variables: +0.604)
[fase7] 2.4    Fisher-Rao al 10.º posterior benigno       rho=+0.775 [+0.392,+0.918] p=0.0004 (51 variables: +0.724)
[fase7] 6.2    kNN en el espacio decodificado             rho=+0.680 [+0.133,+0.900] p=0.0021 (51 variables: +0.691)
[fase7] 7.1    Tasa KL(q||p) por flujo                    rho=+0.830 [+0.482,+0.981] p=0.0004 (51 variables: +0.681)
[fase7] 5.1    kNN al benigno (z blanqueado)              rho=+0.741 [+0.290,+0.943] p=0.0011 (51 variables: +0.694)
[fase7] 6.1    Log-magnificación del decodificador        rho=+0.592 [+0.050,+0.908] p=0.0154 (51 variables: +0.523)
[fase7] 10.1b  Silhouette del grupo (Fisher-Rao)          rho=-0.627 [-0.897,-0.178] p=0.0259 (51 variables: -0.629)
[fase7] resumen en /home/estudiante/vae-latent-geometry-nids/outputs/oe2_descriptores_saneado/RESULTADOS_CONFIRMATORIA.md
```

## Fin de la corrida — 2026-10-11T16:24:36Z

- E01: OK
- E10: OK
- E11: OK
- E12: OK
- E13: OK
- E20: OK
- E21: OK
- E22_s42: OK
- E22_s43: OK
- E22_s44: OK
- E22_s45: OK
- E22_s46: OK
- E22_s47: OK
- E22_s48: OK
- E22_s49: OK
- E22: OK
- E23: OK
- E24: OK
- E25: OK
- E30: OK
- E31: OK
- E32: OK
- E33: OK
- E34: OK
- E35: OK
- E36: OK

## E99 — Resumen de la corrida

- Inicio: 2026-10-11T16:24:36Z
- Comando: `/home/estudiante/vae-latent-geometry-nids/.venv-linux/bin/python scripts/vm_resumen_corrida.py`
- Fin: 2026-10-11T16:24:37Z (1 s = 0 min)
- Estado: **OK** (exit 0)

Log completo: `outputs/vm_pendientes/logs/E99.log`. Últimas líneas:

```
===== [2026-10-11T03:37:45Z] E99: Resumen de la corrida =====
[resumen] /home/estudiante/vae-latent-geometry-nids/outputs/vm_pendientes/RESUMEN_CORRIDA.md
===== [2026-10-11T16:24:36Z] E99: Resumen de la corrida (intento 1) =====
[resumen] /home/estudiante/vae-latent-geometry-nids/outputs/vm_pendientes/RESUMEN_CORRIDA.md
```
