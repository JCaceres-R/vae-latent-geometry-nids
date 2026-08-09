# Pipeline de datos — CICIDS2017 corregido (Engelen et al., 2021)

Template inicial para la monografía (Autoencoder/VAE benign-only sobre
tráfico de red). Cubre desde el CSV crudo hasta los tensores listos para
entrenar, siguiendo exactamente la sección de metodología de la propuesta.

## Estructura

```
project_template/
├── config.py          # rutas, semilla, columnas, proporciones de split
├── data_pipeline.py   # carga -> sanitización -> taxonomía de labels -> split -> escalado
├── requirements.txt
└── data/
    ├── raw/            # <- coloca aquí los 5 CSV (Monday...Friday-WorkingHours.csv)
    └── processed/      # <- se genera al correr el pipeline
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # o .venv\Scripts\activate en Windows
pip install -r requirements.txt
```

Copia los 5 CSV a `data/raw/`, luego:

```bash
python data_pipeline.py
```

## Decisiones ya tomadas (documentar en la monografía)

- **Fuga de datos**: se excluyen `Flow ID`, `Src IP`, `Dst IP`, `Timestamp`
  como features (las IPs de atacante/víctima son fijas durante toda la
  simulación en CICIDS2017 — dejarlas permitiría al modelo memorizar
  direcciones en vez de aprender patrones de tráfico). Los puertos
  (`Src Port`, `Dst Port`) también se excluyen por defecto; revisar si tu
  arquitectura los necesita.
- **Etiquetas `X - Attempted`**: se mantienen como clases propias (no se
  fusionan con BENIGN ni con el ataque completo), para evaluación aislada
  por familia en el Experimento 2. Ojo: al ser flujos casi sin contenido
  malicioso real, es esperable que el AE/VAE los reconstruya como
  benignos — repórtalos por separado del ataque "completo" en tus curvas
  ROC/AUC para no distorsionar la métrica de cada familia.
- **Sanitización**: se eliminan filas con nulos/infinitos (division por
  cero en `Flow Bytes/s`, `Flow Packets/s`, `Flow IAT *`). En el dataset
  de referencia esto afecta <650 filas de 2.1M — despreciable.
- **Split 70/15/15**: aplicado únicamente sobre BENIGN, estratificado por
  día para conservar diversidad temporal. El 100% del tráfico malicioso
  se reserva para evaluación (Experimento 2 y 3 de la propuesta).
- **Normalización sin fuga**: `MinMaxScaler` se ajusta solo con
  `train_benign`, y se aplica de forma estática a val/test/attacks.

## Siguientes pasos sugeridos (fuera de este template)

1. `eda.py` — distribuciones univariadas, matriz de correlación entre las
   features numéricas, verificación de que `len(feature_columns.json) == 78`
   (o el valor que corresponda a tu arquitectura final).
2. `model.py` — Autoencoder/VAE (`x ∈ R^d → bottleneck → x̂`), con
   `train_benign.parquet` como único input de entrenamiento.
3. Cálculo del umbral `τ` (percentil 99 del error de reconstrucción sobre
   `val_benign.parquet`).
4. Evaluación por familia usando `test_attacks.parquet` (agrupar por
   `Label` o por `attack_family`, según si quieres separar `Attempted`).
