# El split 70/15/15 y qué es "Validación" en este proyecto

## 1. El split, con números reales

El dataset **solo se divide sobre tráfico BENIGN** (los ataques nunca se
reparten en train/val/test — el 100% de `test_attacks.parquet` se
reserva aparte para evaluación). División estratificada por día
(`config.py`: `TRAIN_FRAC=0.70`, `VAL_FRAC=0.15`, `TEST_FRAC=0.15`):

| Split | Archivo | n filas | % real |
|---|---|---:|---:|
| Train | `train_benign.parquet` | 1,159,948 | 70.0% |
| **Validación** | `val_benign.parquet` | 248,560 | 15.0% |
| Test | `test_benign.parquet` | 248,561 | 15.0% |

Total benigno: 1,657,069 filas.

## 2. Qué es "Validación", conceptualmente

Son tres roles distintos y **no intercambiables**:

- **Train**: el modelo *ve* estos datos directamente — los pesos de la
  red (encoder/decoder del VAE) se ajustan con gradiente descendente
  sobre este split. Es "material de estudio".
- **Validación**: el modelo **no** ajusta sus pesos con estos datos
  (no hay backpropagation sobre ellos), pero sí se usan para tomar
  **decisiones sobre el proceso de entrenamiento y evaluación** —
  cuándo parar de entrenar, qué umbral usar. Es como un "examen de
  práctica": no estudias directamente de él, pero sí lo usas para decidir
  cuándo dejar de estudiar o para calibrar qué tan estricto vas a ser.
- **Test**: el modelo nunca lo toca para nada hasta el final. Es el único
  split que da una medida honesta de qué tan bien funciona el modelo con
  datos que nunca influyeron ninguna decisión.

**La razón de separar validación de test** (y no simplemente usar "todo
lo que no es train" como una sola cosa): si calibras una decisión
(cuándo parar de entrenar, qué umbral usar) *y luego* mides el
desempeño *sobre los mismos datos que usaste para calibrar esa
decisión*, el número que obtienes está sesgado optimistamente — elegiste
justo el punto que mejor le queda a esos datos específicos. Necesitas un
tercer conjunto (test) que la decisión nunca haya visto, para saber si
esa decisión generaliza.

## 3. Qué hacemos específicamente con `val_benign.parquet` en este proyecto

Tiene **dos usos concretos**, en dos fases distintas — nunca se usa para
calcular las métricas finales de detectabilidad o geometría:

### Uso 1 — Early stopping durante el entrenamiento del VAE (OE1)

En `training/train.py`, cada época se calcula el ELBO total
(reconstrucción + β·KL) tanto en train como en `val_benign`. El
entrenamiento se detiene cuando `val_total` deja de mejorar durante
`patience=10` épocas seguidas (con `min_delta=0.10`, calibrado
empíricamente sobre el ruido real del log — ver `oe1_report.md`). El
checkpoint que se guarda como "el mejor" es el de la época con menor
`val_total`, no la última época entrenada.

**Por qué no se usa train para esto**: el ELBO en train baja
monotónicamente mientras el modelo sigue aprendiendo (incluyendo
sobre-ajustarse); usar ese número para decidir cuándo parar nunca
detectaría el sobre-ajuste. Validación sí lo detecta, porque el modelo
no la memoriza.

### Uso 2 — Calibración de τ, el umbral de anomalía (OE2, Fase 4)

En `evaluation/detectability_oe2.py`, τ se calcula como un percentil (95
actualmente, 99 en discusión — ver `oe2_resumen_pendientes.md`) de la
distribución de scores de reconstrucción **sobre `val_benign.parquet`
completo**, no sobre train ni sobre test.

**Por qué val y no test**: τ es una decisión (¿a partir de qué score de
anomalía se dispara una alerta?). Si esa decisión se calibrara usando
`test_benign`/`test_attacks` —los mismos datos con los que después se
mide AUC-ROC y TPR por familia—, el resultado estaría contaminado: el
umbral estaría "hecho a la medida" de los datos que se usan para medir
qué tan bien funciona. Al calibrar τ sobre validación (un split que
`detectability_oe2.py` describe explícitamente como "split fresco,
nunca tocado antes de esta fase"), el umbral queda fijo *antes* de mirar
los datos de evaluación, y el AUC-ROC/TPR medidos después sobre
test_benign + test_attacks son una medida honesta.

**Por qué no se usa train para esto tampoco**: train es lo que el VAE ya
memorizó directamente — usar train para calibrar τ mediría qué tan bien
reconstruye el modelo los datos que literalmente usó para aprender a
reconstruir, lo cual sería aún más optimista que usar test.

## 4. Resumen — qué split se usa para qué, en todo el proyecto

| Qué se decide/mide | Split usado | Dónde |
|---|---|---|
| Ajustar pesos del VAE (gradiente) | `train_benign` | `training/train.py` |
| Ajustar `MinMaxScaler` | `train_benign` (solo) | `data/pipeline.py` |
| Cuándo parar de entrenar (early stopping) | **`val_benign`** | `training/train.py` |
| Calibrar τ (umbral de anomalía) | **`val_benign`** | `evaluation/detectability_oe2.py` |
| Medir AUC-ROC / TPR por familia (resultado final) | `test_benign` + `test_attacks` | `evaluation/detectability_oe2.py` |
| Geometría (Mahalanobis, silhouette) | `test_benign` + `test_attacks` (vía `latent_benign_test.npy`) | `evaluation/geometric_metrics_oe2.py` |

En ningún punto del pipeline se usa `val_benign` para las métricas que
terminan reportándose como resultado de OE2 (Fase 3, 4 o 5) — solo para
las dos decisiones de calibración de arriba. Esa disciplina es
justamente lo que permite decir, sin reservas, que los números de AUC-ROC
y Mahalanobis de este proyecto no tienen fuga de datos.
