# vae-latent-geometry-nids

Caracterización de la geometría del espacio latente de un VAE para detección de intrusiones.
Monografía de grado, Ingeniería Electrónica, Universidad Distrital Francisco José de Caldas.

## Contexto

La mayoría de los trabajos que usan un VAE (Variational Autoencoder) para detectar intrusiones lo hacen
de forma puramente discriminativa. Entrenan con tráfico benigno, usan el error o la probabilidad de
reconstrucción como puntaje de anomalía y no explican **por qué** el modelo responde tan distinto ante
familias de ataque diferentes.

Este proyecto no busca otro detector de intrusiones con VAE; ese problema ya está resuelto y saturado en
la literatura. El foco es diagnóstico:

> **Pregunta de investigación:** ¿qué propiedades geométricas del espacio latente probabilístico de un
> VAE entrenado con tráfico benigno (posición respecto del núcleo benigno, dispersión, forma de la nube
> posterior) explican por qué unas familias de ataque son más representables y reconstruibles que otras?

**Objetivo general:** analizar cuantitativamente la geometría del espacio latente de un VAE entrenado con
tráfico benigno para determinar por qué unas familias de ataque resultan más reconstruibles que otras.

**Hipótesis:** la reconstruibilidad diferencial frente a distintas familias de ataque se explica por la
geometría que cada familia induce en el espacio latente respecto de la región de alta densidad del tráfico
benigno.

La métrica de éxito no es el AUC o el F1 global, sino evidencia geométrica cuantitativa y verificable.

## Objetivos específicos y estado

| OE | Enunciado | Estado |
|---|---|---|
| **OE1** | Estabilidad del espacio latente: unidades activas y KL por dimensión | **Completo.** Entrada saneada (66 variables) y 8 semillas entrenadas en la VM |
| **OE2** | Geometría comparativa y correlación con la detectabilidad (hallazgo central) | **Completo sobre los modelos del OE1.** Falta la reconciliación del nivel de conexión (fase 8, ya implementada) |
| **OE3** | Capacidad generativa: KS, Wasserstein y clasificador real frente a sintético | **Implementado**; pendiente de correr en la VM |
| **OE4** | Reproducibilidad del ranking frente a β y k | **Implementado** (β = 0,5, β = 2, k = 16; 5 semillas cada una); pendiente de correr en la VM |

El OE1 valida que el fenómeno se puede medir, el OE2 es el hallazgo central, y el OE3 y el OE4 validan que
ese hallazgo es legítimo y robusto. **Todos los objetivos usan los mismos modelos**: los del OE1 saneado,
o variantes suyas que solo cambian β o k en el OE4.

## Resultados hasta ahora

Cifras de la corrida en la VM del 11-10-2026. El detalle está en `outputs/vm_pendientes/RESUMEN_CORRIDA.md`
y la lectura en `docs/LECTURA_RESULTADOS_VM_2026-10-11.md`.

**OE1 (66 variables, 8 semillas, β = 1, k = 8):**

| | 51 variables (antes, 5 semillas) | 66 variables (saneado, 8 semillas) |
|---|---|---|
| Unidades activas (Var(μ) > 0,01) | 5,4 | **6,9** |
| KL total (nats) | 6,9 | 8,5 |
| Log σ² del decodificador en el piso −10 | 89,9 % | 90,5 % |

**OE2 (11 familias, 8 semillas; desenlace: AUC-ROC de la NLL Monte Carlo):**

- El ranking de familias por detectabilidad es estable entre semillas (W de Kendall 0,89) y frente al modelo
  de 51 variables (Spearman 0,96). Las familias más difíciles son Bot (AUC 0,61) y PortScan (0,67).
- **Prueba confirmatoria**, con hipótesis registradas antes de ver los resultados del modelo saneado
  (`fase7_confirmatoria.py`):

| Descriptor | ρ con el AUC | IC 95 % | p (Holm) |
|---|---|---|---|
| **Primario:** Mahalanobis (mediana) | **0,77** | [0,39; 0,98] | 0,001 |
| Tasa KL(q ‖ p) por flujo | 0,83 | [0,48; 0,98] | 0,002 |
| Fisher–Rao al 10.º posterior benigno | 0,78 | [0,39; 0,92] | 0,002 |
| kNN al benigno (z blanqueado) | 0,74 | [0,29; 0,94] | 0,004 |
| kNN en el espacio decodificado | 0,68 | [0,13; 0,90] | 0,006 |
| Log-magnificación del decodificador | 0,59 | [0,05; 0,91] | 0,031 |
| Silhouette del grupo (Fisher–Rao) | −0,63 | [−0,90; −0,18] | 0,031 |

  Se confirman el primario y los 6 secundarios. La relación también aparece dentro de cada familia, a
  nivel de flujo (ρ entre 0,60 y 0,75).
- Salvedades: la confirmación se hizo con modelos reentrenados de forma independiente, pero sobre las
  mismas familias con las que se eligieron los descriptores. El modelo mixto a nivel de conexión de la
  fase 5 no encuentra efecto (p = 0,32), lo que parece contradecir el nivel de flujo; la fase 8 lo
  reconcilia.

## Dataset y preprocesamiento

Versión corregida de CICIDS2017 ([Engelen, Rimmer y Joosen, 2021](https://intrusion-detection.distrinet-research.be/WTMC2021/),
KU Leuven), que corrige errores de simulación, construcción de flujos, extracción de características y
etiquetado del CICIDS2017 original. Se usa esta versión para que los hallazgos geométricos no sean
artefactos de un dataset con errores conocidos.

Preprocesamiento vigente (OE1 saneado, `src/vae_nids/oe1/`, `configs/oe1_saneado.yaml`; informe en
`docs/oe1_saneado/INFORME_SANEAMIENTO_OE1.md`):

- **Fuga de datos.** Se excluyen `Flow ID`, `Src IP`, `Dst IP`, `Timestamp`, `Src Port` y `Dst Port`. Las
  IP de atacante y víctima son fijas en la simulación; dejarlas permitiría memorizar direcciones.
- **Selección por información nula (77 → 66 variables), sin umbral y solo con `train_benign`.** Solo salen
  las columnas con información nula: 3 copias exactas y 8 funciones exactas de columnas que se quedan, cada
  una con su fórmula verificada en el 100 % de las filas y su línea en el código de CICFlowMeter. Cualquier
  otra redundancia se conserva y la comprime el codificador. El conjunto queda congelado con SHA-256
  (`configs/features/oe1_info_nula.yaml`).
- **Saneamiento.** Se eliminan las filas con inf/NaN (793) y, dentro de cada partición y después de
  dividir, los flujos con duración negativa (31).
- **Split 70/15/15** del benigno, estratificado por día. El 100 % del tráfico malicioso se reserva para
  evaluación.
- **Escalado sin fuga.** MinMax ajustado solo con `train_benign`, con parámetros en float64.
- **Etiquetas `X - Attempted`.** Se mantienen como grupos propios y sirven de validación externa en el OE2.

Historia: la primera versión redujo 74 → 51 variables con clusters de Pearson |r| ≥ 0,95 calculados sobre
datos mezclados (`notebooks/cluster_review_oe1.md`). El saneamiento la reemplazó; los resultados con 51
variables se conservan solo como comparación.

## Modelo

VAE gaussiano (`src/vae_nids/models/vae.py`): codificador y decodificador de una capa oculta (32, ReLU),
k = 8, β = 1, decodificador heterocedástico (μ_x, log σ_x²) y log σ² acotado a [−10, 10]. Puntaje de
anomalía: NLL Monte Carlo con L = 20 muestras del posterior; umbral τ = p95 en `val_benign`.
Entrenamiento: Adam (lr 1e-3, lote 1024), parada por paciencia (10 épocas, Δ mínimo 0,1 nats), determinista
y con checkpoints autocontenidos (configuración, columnas, escalador, commit y versiones).

## Estructura del repositorio

```
vae-latent-geometry-nids/
├── configs/
│   ├── oe1_saneado.yaml          # configuración única del modelo base (66 variables, semillas 42-49)
│   ├── oe1_saneado_ref74.yaml    # referencia de 74 variables
│   ├── oe4_beta0p5.yaml, oe4_beta2.yaml, oe4_k16.yaml   # variantes del OE4
│   └── features/                 # conjuntos de columnas congelados (SHA-256)
├── data/                         # no versionado: raw/ (5 CSV, 1,1 GB) y processed/ (parquets)
├── docs/                         # informes, guías de la VM, estado y lectura de resultados (ver abajo)
├── notebooks/                    # EDA y bitácoras del OE1/OE2 inicial (51 variables)
├── outputs/
│   ├── oe1_saneado/              # checkpoints, curvas, diagnósticos, tablas del informe
│   ├── oe2_robust_saneado/       # OE2 robusto: geometría, detectabilidad, correlación
│   ├── oe2_descriptores_saneado/ # 33 descriptores, cribado, confirmatoria (fase 7), reconciliación (fase 8)
│   ├── oe3/, oe4/                # se generan en la segunda corrida de la VM
│   ├── vm_pendientes/            # documentación de la 1.ª corrida en la VM (resumen, log, logs por etapa)
│   ├── oe2_robust/, oe2_descriptores/, checkpoints/, latent_vectors/   # modelo de 51 variables (comparación)
│   └── oe1_mi/                   # validación con entropía e información mutua (OE1-IM)
├── scripts/                      # corridas en la VM: run_pendientes_vm.sh, run_oe3_oe4_vm.sh y auxiliares
├── src/vae_nids/
│   ├── data/pipeline.py          # pipeline original (51 variables)
│   ├── oe1/                      # OE1 saneado: datos, selección, preproceso, entrenamiento, diagnósticos
│   ├── models/vae.py             # arquitectura
│   ├── training/                 # entrenamiento original
│   └── evaluation/
│       ├── oe2_robust/           # OE2 multi-semilla (fases 0-6), perfiles original/saneado
│       ├── oe2_descriptores/     # descriptores (fases 1-6), confirmatoria (7), reconciliación (8)
│       ├── oe3/                  # calidad generativa
│       ├── oe4/                  # concordancia de rankings frente a β y k
│       └── oe1_mi/, *_oe2.py     # OE1-IM y OE2 inicial (una semilla)
└── tests/                        # oe1_saneado, oe2_robust, oe1_mi
```

## Cómo reproducir

### Entorno local

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # versiones de Windows/Python 3.14
pip install -e .
pytest tests/oe1_saneado tests/oe2_robust -q
```

Los 5 CSV corregidos van en `data/raw/`. El cómputo pesado (entrenamiento, OE2 completo, OE3, OE4) se
corre en la VM Linux, no en local.

### En la VM (Ubuntu 22.04, Python 3.10, solo CPU)

Dos corridas sin supervisión. Cada una deja su documentación (`RESUMEN_CORRIDA.md`, `RUN_LOG.md`, el estado
de cada etapa y sus logs) en una carpeta que se versiona, para revisarla desde otra máquina.

| Corrida | Qué hace | Documentación | Guía |
|---|---|---|---|
| `scripts/run_pendientes_vm.sh` | OE1 saneado completo y OE2 sobre esos modelos | `outputs/vm_pendientes/` | `docs/EJECUTAR_PENDIENTES_EN_VM.md` |
| `scripts/run_oe3_oe4_vm.sh` | Reconciliación (fase 8), OE3 y OE4 | `outputs/vm_oe3_oe4/` | `docs/EJECUTAR_OE3_OE4_EN_VM.md` |

```bash
git pull
tmux new -A -s p
bash scripts/run_oe3_oe4_vm.sh
# al terminar:
bash scripts/vm_preparar_commit.sh outputs/vm_oe3_oe4
git commit -F outputs/vm_oe3_oe4/MENSAJE_COMMIT.txt && git push
```

Detalles de la VM:
- Las versiones exactas están en `requirements-vm.txt`.
- Las corridas limitan las instrucciones SIMD de MKL, Arrow, torch y numba (`scripts/vm_comun.sh`). Sin
  eso, la VM daba SIGILL intermitentes.
- El OE2 elige su perfil con `VAE_NIDS_OE2_PERFIL=saneado` y, para el OE4, la configuración con
  `VAE_NIDS_OE2_CONFIG=configs/oe4_<variante>.yaml`.

## Documentación

| Documento | Contenido |
|---|---|
| `docs/ESTADO_OE1_OE2_2026-10-10.md` | Estado del OE1 y del OE2 y decisión de alinear el OE2 con el OE1 saneado |
| `docs/oe1_saneado/INFORME_SANEAMIENTO_OE1.md` | Saneamiento del OE1: selección por información nula, preprocesamiento, hiperparámetros |
| `docs/PENDIENTES_OE2_GEOMETRIA_LATENTE.md` | Investigación de descriptores (221 trabajos), catálogo de 33, hallazgos sobre el modelo |
| `docs/LECTURA_RESULTADOS_VM_2026-10-11.md` | Lectura de los resultados de la primera corrida en la VM |
| `docs/EJECUTAR_PENDIENTES_EN_VM.md`, `docs/EJECUTAR_OE3_OE4_EN_VM.md` | Guías de las dos corridas en la VM |
| `docs/COMMITS_SUGERIDOS_*.md` | Agrupación de commits sugerida en cada etapa |

Los informes en LaTeX (OE1 saneado, geometría latente) están fuera del repo, en la carpeta de la
monografía.

## Referencias clave

- Engelen, G., Rimmer, V., & Joosen, W. (2021). *Troubleshooting an Intrusion Detection Dataset: the
  CICIDS2017 Case Study*. IEEE Security and Privacy Workshops (SPW), 7–12.
- Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018). *Kitsune: An ensemble of autoencoders for
  online network intrusion detection*. IEEE Symposium on Security and Privacy (SP), 1131–1146.
- Arvanitidis, G., et al. (2022). Métrica pull-back de Fisher–Rao del decodificador (arXiv:2106.05367).
- Kumar, A., & Poole, B. (2020). Covarianza del codificador y jacobiano del decodificador en VAEs
  (arXiv:2002.00041).
- Lee, K., et al. (2018). Mahalanobis como puntaje fuera de distribución (arXiv:1807.03888).
- Miyamoto, H. K., et al. (2023). Distancia de Fisher–Rao en forma cerrada, caso diagonal (arXiv:2304.14885).

La bibliografía completa del OE2 (93 artículos) está en la carpeta de la monografía y en
`docs/PENDIENTES_OE2_GEOMETRIA_LATENTE.md`, sección 10.
