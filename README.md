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

## Objetivos específicos

| # | Objetivo | Evidencia esperada |
|---|----------|---------------------|
| **OE1** | Establecer una línea base geométricamente estable del espacio latente (ELBO converge sin colapso del posterior) | Curvas de ELBO, histograma de σ² por dimensión latente, checkpoints reproducibles |
| **OE2** | Cuantificar posición y dispersión de cada familia de ataque en el espacio latente | Distancia de Mahalanobis al centroide benigno, silhouette score, proyecciones t-SNE/UMAP anotadas |
| **OE3** | Correlacionar la geometría latente con la detectabilidad observada (AUC-ROC, TPR/FPR) | Matriz de correlación Pearson/Spearman geometría-vs-detección |
| **OE4** | Validar que la geometría corresponde a un modelo genuinamente generativo | Muestreo z ~ N(0,I), test Kolmogorov-Smirnov entre tráfico sintético y benigno real |
| **OE5** | Evaluar robustez de los hallazgos frente a β y la dimensión latente k | Comparación de resultados (Mahalanobis, AUC) en ≥2-3 configuraciones distintas |

OE1 valida que el fenómeno se puede medir; OE2 es el hallazgo central; OE3, OE4 y OE5
validan que ese hallazgo es legítimo y robusto.

## Cronograma (13 semanas)

| Fase | Semanas |
|------|---------|
| OE1 — Estabilidad del espacio latente | 1–2 |
| OE2 — Geometría comparativa y correlación | 3–6 |
| OE3 — Capacidad generativa | 7–9 |
| OE4/OE5 — Reproducibilidad frente a β, k | 10–11 |
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
  también como features. Quedan **74 features** numéricas.
- **Etiquetas `X - Attempted`** (flujos capturados durante la ventana de un ataque pero
  sin payload malicioso real): se mantienen como clases propias, no se fusionan con
  BENIGN ni con el ataque completo, para no distorsionar la evaluación aislada por
  familia (OE2/OE3).
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
│   ├── evaluation/              # umbral τ, ROC/AUC por familia, métricas geométricas (OE2/OE3/OE5)
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

- [x] Pipeline de datos (carga, sanitización, taxonomía de labels, split, escalado)
- [ ] Arquitectura VAE (encoder/decoder, reparametrización, ELBO) — OE1
- [ ] Calibración de β y verificación de no colapso del posterior — OE1
- [ ] Métricas geométricas por familia (Mahalanobis, silhouette, t-SNE/UMAP) — OE2
- [ ] Correlación geometría–detectabilidad — OE3
- [ ] Validación generativa (muestreo + test KS) — OE4
- [ ] Análisis de robustez frente a β/k — OE5

## Referencias clave

- Engelen, G., Rimmer, V., & Joosen, W. (2021). *Troubleshooting an Intrusion Detection
  Dataset: the CICIDS2017 Case Study*. IEEE Security and Privacy Workshops (SPW), 7–12.
- Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018). *Kitsune: An ensemble of
  autoencoders for online network intrusion detection*. IEEE Symposium on Security and
  Privacy (SP), 1131–1146.