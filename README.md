# vae-latent-geometry-nids

Caracterización de la Geometría del Espacio Latente de un VAE para Detección de Intrusiones.
Monografía de grado — Ingeniería Electrónica, Universidad Distrital Francisco José de Caldas.

## Contexto

La mayoría de trabajos que usan un VAE (Variational Autoencoder) para detección de
intrusiones lo hacen de forma puramente discriminativa: entrenan sobre tráfico benigno
y usan el error o la probabilidad de reconstrucción como score de anomalía, sin explicar
**por qué** el modelo responde de forma tan distinta ante familias de ataque diferentes.

Este proyecto no busca "otro detector de intrusiones con VAE" — ese problema ya está
resuelto y saturado en la literatura. El foco es diagnóstico:

> **Pregunta de investigación:** ¿Qué propiedades geométricas del espacio latente
> probabilístico de un VAE entrenado sobre tráfico benigno —posición respecto al núcleo
> benigno, dispersión, forma de la nube posterior— explican por qué unas familias de
> ataque son más representables y reconstruibles por el modelo que otras?

**Objetivo general:** Analizar cuantitativamente la geometría del espacio latente de un
VAE entrenado sobre tráfico benigno, para determinar por qué unas familias de ataque
resultan más reconstruibles que otras.

**Hipótesis:** La reconstruibilidad diferencial de un VAE semisupervisado frente a
distintas familias de ataque es directamente explicable por la geometría que cada
familia induce en el espacio latente respecto a la región de alta densidad del tráfico
benigno.

La métrica de éxito no es AUC/F1 global, sino evidencia geométrica cuantitativa y
verificable.

## Objetivos Específicos

- **OE1** — Estabilidad del espacio latente (semanas 1-2)
- **OE2** — Geometría comparativa y correlación con detectabilidad (semanas 3-6)
- **OE3** — Capacidad generativa (semanas 7-9)
- **OE4** — Reproducibilidad frente a β, k (semanas 10-11)

OE1 valida que el fenómeno se puede medir; OE2 es el hallazgo central; OE3 y OE4
validan que ese hallazgo es legítimo y robusto.

## Cronograma (13 semanas)

| Fase | Semanas |
|------|---------|
| OE1 — Estabilidad del espacio latente | 1–2 |
| OE2 — Geometría comparativa y correlación con detectabilidad | 3–6 |
| OE3 — Capacidad generativa | 7–9 |
| OE4 — Reproducibilidad frente a β, k | 10–11 |
| Cierre — Consolidación y sustentación | 12–13 |

## Dataset

Versión corregida de CICIDS2017 ([Engelen, Rimmer & Joosen, 2021](https://intrusion-detection.distrinet-research.be/WTMC2021/) — KU Leuven), que corrige errores de simulación,
construcción de flujos, extracción de características y labelling presentes en el
CICIDS2017 original. Se usa esta versión específicamente para evitar que hallazgos
geométricos terminen siendo artefactos espurios de un dataset con errores conocidos de
generación.

Decisiones de preprocesamiento ya tomadas (ver `src/vae_nids/data/pipeline.py`):

- **Fuga de datos**: se excluyen `Flow ID`, `Src IP`, `Dst IP`, `Timestamp`, `Src Port`,
  `Dst Port` como features (las IPs de atacante/víctima son fijas durante toda la
  simulación en CICIDS2017 — dejarlas permitiría al modelo memorizar direcciones en vez
  de aprender patrones de tráfico).
- **Duplicados exactos**: el EDA (`notebooks/eda_cicids2017.ipynb`, Sección 7.3)
  identificó 3 columnas con correlación r = 1.0 frente a otra ya presente
  (`Bwd Segment Size Avg`, `Average Packet Size`, `Fwd Segment Size Avg`); se excluyen
  también como features.
- **Features redundantes por correlación**: revisión adicional de 13 clusters con
  `|r| ≥ 0.95` (verificado por clique real, no single-linkage — ver
  `notebooks/cluster_review_oe1.md` y `data/processed/eda_exclusion_log.json`) excluyó
  23 features más, manteniendo un representante por grupo. Preserva asimetrías
  forward/backward donde la evidencia empírica las sostiene (tamaño de payload,
  temporización IAT); las colapsa donde no (conteos de paquetes/flags, simétricos
  entre fwd/bwd en este dataset). Quedan **51 features** numéricas.
- **Etiquetas `X - Attempted`** (flujos capturados durante la ventana de un ataque pero
  sin payload malicioso real): se mantienen como clases propias, no se fusionan con
  BENIGN ni con el ataque completo, para no distorsionar la evaluación aislada por
  familia (OE2).
- **Sanitización**: se eliminan filas con nulos/infinitos (~0.04% del total, división
  por cero en `Flow Bytes/s`, `Flow Packets/s`, `Flow IAT *`).
- **Split 70/15/15**, estratificado por día, aplicado únicamente sobre tráfico BENIGN.
  El 100% del tráfico malicioso se reserva para evaluación.
- **Normalización sin fuga**: `MinMaxScaler` se ajusta solo con el split de
  entrenamiento benigno, y se aplica de forma estática a val/test/ataques.

## Estructura del repositorio

```
vae-latent-geometry-nids/
├── data/
│   ├── raw/                # 5 CSV de Engelen et al. (no versionado, ~1GB)
│   └── processed/          # .parquet generados por el pipeline (no versionado, reproducible)
├── notebooks/               # EDA exploratorio y prototipos rápidos
├── outputs/
│   ├── figures/              # gráficas para la monografía (ROC, proyecciones latentes, histogramas)
│   └── checkpoints/          # pesos del modelo entrenado (no versionado)
├── src/vae_nids/
│   ├── config.py               # rutas, semilla, columnas, proporciones de split
│   ├── data/
│   │   └── pipeline.py           # carga → sanitización → taxonomía de labels → split → escalado
│   ├── models/                 # arquitectura del VAE (encoder/decoder probabilístico)
│   ├── training/                # loop de entrenamiento, calibración de β, early stopping (OE1)
│   ├── evaluation/              # umbral τ, ROC/AUC por familia, métricas geométricas (OE2/OE4)
│   └── viz/                     # visualizaciones del espacio latente (t-SNE/UMAP, proyecciones)
└── tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # ajustar rutas si es necesario
```

Coloca los 5 CSV corregidos en `data/raw/`, luego:

```bash
python -m vae_nids.data.pipeline
```

Esto genera en `data/processed/`: `train_benign.parquet`, `val_benign.parquet`,
`test_benign.parquet`, `test_attacks.parquet`, `scaler.joblib` y
`feature_columns.json`.

## Estado actual

- [x] Pipeline de datos (74 → 51 features tras EDA y revisión de
      clusters correlacionados, ver `data/processed/eda_exclusion_log.json`)
- [x] Arquitectura VAE de producción (`src/vae_nids/models/vae.py`)
- [x] OE1 — Estabilidad del espacio latente: entrenamiento convergente,
      unidades activas 5-6/8 (barrido de 5 semillas), documentado en
      `notebooks/oe1_report.md`
- [ ] OE2 — Geometría comparativa y correlación con detectabilidad
- [ ] OE3 — Capacidad generativa
- [ ] OE4 — Reproducibilidad frente a β, k

## Referencias clave

- Engelen, G., Rimmer, V., & Joosen, W. (2021). *Troubleshooting an Intrusion Detection
  Dataset: the CICIDS2017 Case Study*. IEEE Security and Privacy Workshops (SPW), 7–12.
- Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018). *Kitsune: An ensemble of
  autoencoders for online network intrusion detection*. IEEE Symposium on Security and
  Privacy (SP), 1131–1146.