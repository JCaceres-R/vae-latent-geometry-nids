# Saneamiento del OE1: conjunto por información nula, preprocesamiento corregido y reentrenamiento

Fecha: 8 de octubre de 2026. Rama: `main` (sin commits: el autor hace los commits y el push; ver `COMMITS_SUGERIDOS.md`).

## 1. Resumen en 10 líneas

1. El conjunto de entrada se derivó de nuevo **solo con `train_benign`**, sin umbral y sin Pearson: de las 77 columnas de flujo salen solo las que tienen información nula. Quedan **66** (antes 51).
2. Salen **3 copias** (R0), **0 constantes** (R1) y **8 funciones exactas** (R2), cada una con su fórmula verificada en el 100 % de las filas y su línea en el código fuente de CICFlowMeter.
3. Seis cocientes (las 4 tasas, `Packet Length Variance` y `Down/Up Ratio`) son funciones de lo que queda salvo errores de 8 a 100 ulp en el 0,01–3,6 % de las filas. Por la regla estricta **se quedan** como «casi funciones». Si el autor acepta un ε ampliado, el conjunto sería de 60 (registrado aparte).
4. Ninguna identidad aceptada falla en `val_benign`, `test_benign` ni en ninguna familia de ataque (10,3 millones de comprobaciones, 0 fallos).
5. `Total Length of Bwd Packet` vuelve al conjunto.
6. Se filtraron los 31 flujos con duración negativa (24 en train, 7 en val), después de dividir. Las particiones son idénticas al pipeline original salvo esas filas (verificado columna a columna y con hashes).
7. `Flow Packets/s` pasa de conservar el 1,8 % de sus valores distintos a conservar el 99,2 %.
8. Los hiperparámetros viven en un solo archivo (`configs/oe1_saneado.yaml`); los checkpoints son autocontenidos y el entrenamiento es determinista (dos corridas idénticas).
9. Reentrenamiento: 8 semillas (42–49) más la referencia de 74 columnas. Resultados en la sección 7.
10. Limitación principal: conservar la información no garantiza el mismo rendimiento del modelo; eso se mide en otra tarea.

## 2. Reconocimiento (fase 0)

**Pipeline de datos** (`src/vae_nids/data/pipeline.py`, `src/vae_nids/config.py`):

- Lectura de los 5 CSV en `float32` (`pipeline.py:52-53`), descartando `Flow ID`, `Src IP`, `Dst IP`, `Timestamp` (`config.py:36`) y los puertos (`config.py:42`) al leer.
- Saneamiento: inf → NaN y `dropna` sobre todas las columnas numéricas (`pipeline.py:70-84`). Elimina 793 filas (2 100 814 → 2 100 021), todas por inf/NaN en `Flow Bytes/s` y `Flow Packets/s` (252 también en `Flow IAT *`). No hay deduplicación de filas ni filtro de duraciones negativas.
- Particiones: solo el benigno, 70/15/15, `train_test_split` estratificado por día con `random_state=42` (`pipeline.py:121-141`, `config.py:26,88-95`). Ataques: todo lo no benigno (442 952 filas). Resultado: train 1 159 948, val 248 560, test 248 561.
- MinMax ajustado solo con `train_benign` (`pipeline.py:144-160`); las particiones se guardan en `float32`.
- Reducción 74 → 51: lista fija de 23 columnas en `config.py:60-83` (clusters de Pearson |r| ≥ 0,95 sobre datos **mezclados**, `notebooks/cluster_review_oe1.md`). Las 3 copias del paso 77 → 74 están en `config.py:49-53`.

**Universo:** 84 columnas en el CSV menos 7 de metadatos (Flow ID, IP, puertos, Timestamp, Label) = **77 columnas de flujo** (lista completa en `configs/features/oe1_info_nula.yaml`).

**Entrenamiento** (`src/vae_nids/training/train.py`): valores por defecto `max_epochs=100`, `min_delta=0.0`, `patience=10`, `batch_size=1024`, `lr=1e-3` (líneas 85-95 y 181-190). Los valores reales (250 y 0,10) se pasaron por línea de comandos (`notebooks/oe1_report.md:8,63,119-120`). El checkpoint guarda pesos, `config` del modelo, semilla, época, pérdidas y columnas, pero **no** los hiperparámetros de entrenamiento ni el escalador. τ = p95 del score en `val_benign` (`oe2_robust/phase4_detectability.py:193`).

**OE1 actual:** figuras `outputs/figures/vae_k8_beta1_input{51,74}_{training_curves,latent_diagnostics}.png` (`src/vae_nids/viz/training_curves.py`). El barrido de semillas documentado tiene **5 semillas (42–46)**, no 8: 6, 5, 6, 5 y 5 unidades activas, media 5,4 (`outputs/metrics/oe1_seed_sweep.json` y los JSON por semilla). **Discrepancia con el prompt**, que dice 8 semillas; la media 5,4 sí coincide.

**Código reutilizable:** estimadores de OE1-IM (`src/vae_nids/evaluation/oe1_mi/estimators.py`: `discretize` con átomos aislados y `mi_hist` con la corrección de Sharmin). Las verificaciones V2–V4 de la v2 (`02_Objetivos_Especificos/OE1_Seleccion_de_variables/3_informe_IM_v2/informe_oe1_im_v2/scripts/`) se reimplementaron como identidades R2 (V2: tasas e IAT; V3: regla corregida del mínimo; V4: los Subflow no son copias). No hizo falta traer nada de otra rama.

## 3. Conjunto de entrada

**Fundamento.** Solo hay dos casos en que una columna X se puede quitar sin umbral, y los dos son teoremas (Cover y Thomas, 2006, cap. 2). (a) Si H(X) = 0, entonces I(X;Y) = 0 para todo Y. (b) Si X = f(S) con S columnas que se quedan, entonces H(X|S) = 0 y, por tanto, I(X;Y|S) = 0 para cualquier Y, incluida cualquier clase de ataque, vista o no. Es el criterio de redundancia de Macedo et al. (2022, §3.2: «Xi is redundant given S if H(Xi|S) = 0», verificado en el PDF). Cualquier otra redundancia solo se puede quitar con un umbral: se conserva, y la comprime el codificador (J = 8). H(X|S) = 0 **no se estima con histogramas**: con varias columnas en S casi cada fila cae en su propia celda y el estimador da 0 sin que haya relación. Por eso cada eliminación exige una fórmula explícita verificada fila a fila.

**Tolerancia ε (derivada, no elegida).** El código fuente del CICFlowMeter corregido (`GintsEngelen/CICFlowMeter`, `BasicFlow.java`, commit 4dd5319) escribe cada valor con `StringBuilder.append(double)`. La tolerancia de cada identidad suma tres términos:
- el cuanto de almacenamiento de cada celda, q = ½·10^(e − máx(s, 15) + 1), con s las cifras escritas (algunas celdas solo traen 15);
- su propagación a través de f;
- el redondeo de la operación con que CICFlowMeter calcula X: 3 operaciones por valor en la media incremental de Apache Commons Math, una por división y raíz.

El ε de cada columna está en `configs/features/oe1_info_nula.yaml`.

**Columnas eliminadas** (fuente: `outputs/oe1_saneado/seleccion/estado_columnas.csv`):

| Columna | Regla | Fórmula (solo con columnas que quedan) | % verificado | Error máx. | Línea en BasicFlow.java |
|---|---|---|---|---|---|
| Fwd Segment Size Avg | R0 | = Fwd Packet Length Mean = B_f/n_f | 100 | 262 ulp | 391-395 frente a 1258 |
| Bwd Segment Size Avg | R0 | = Bwd Packet Length Mean = B_b/n_b | 100 | 398 ulp | 397-401 frente a 1270 |
| Average Packet Size | R0 | = Packet Length Mean = (B_f+B_b)/n | 100 | 511 ulp | 384-389 frente a 1336 |
| Fwd Packet Length Mean | R2 | B_f/n_f si n_f > 0; 0 si no | 100 | 286 ulp | 1258 |
| Bwd Packet Length Mean | R2 | B_b/n_b si n_b > 0; 0 si no | 100 | 398 ulp | 1270 |
| Packet Length Mean | R2 | (B_f+B_b)/(n_f+n_b) | 100 | 511 ulp | 1336 |
| Flow IAT Mean | R2 | D/(n−1) si n ≥ 2; 0 si no | 100 | 327 ulp | 1287 |
| Fwd IAT Mean | R2 | Fwd IAT Total/(n_f−1) si n_f ≥ 2; 0 si no | 100 | 234 ulp | 1294 |
| Bwd IAT Mean | R2 | Bwd IAT Total/(n_b−1) si n_b ≥ 2; 0 si no | 100 | 502 ulp | 1308 |
| Packet Length Max | R2 | máx(Fwd PL Max, Bwd PL Max) | 100 | 0 | 1335 |
| Packet Length Min | R2 | Bwd PL Min si n_f = 0; Fwd PL Min si n_b = 0; mín si no | 100 | 0 | 1334 |

Todas se reconstruyen desde el conjunto final, sin pasar por otra eliminada, en el 100 % de las filas (`reconstruccion_desde_final.csv`).

**Casi funciones que se quedan** (`casi_funciones.csv`): `Flow Packets/s`, `Fwd Packets/s`, `Bwd Packets/s` y `Flow Bytes/s` (n/(D/10⁶) en `BasicFlow.java:361-374, 1279-1280`), `Packet Length Variance` (Std²) y `Down/Up Ratio` (n_b/n_f). El código fuente calcula exactamente esas fórmulas, pero el CSV corregido trae valores que difieren entre 8 y 100 ulp en el 0,006–3,6 % de las filas. El dataset tuvo algún posprocesado que no reproduce esa cuenta. **Decisión del autor:** mantenerlas (66 columnas) o aceptarlas con ε ampliado (`configs/features/alternativos/info_nula_eps_ampliado_60.yaml`).

**Casi constantes (D2: se quedan):** `Fwd URG Flags`, `Bwd URG Flags` (H = 1,9·10⁻⁵ bits; el prompt daba ≈1,1·10⁻⁵), `URG Flag Count`, `CWR Flag Count`, `ECE Flag Count` y `Subflow Bwd Packets` (`entropia_por_columna.csv`). Descripción: Kuhn y Johnson (2013, §3.5), **por verificar** (libro no accesible).

**Búsqueda sistemática (3.5):** 12 pares con H(X|Y) = 0 en la muestra (`busqueda_sistematica_dependencias.csv`). Tres son las copias R0, ya tratadas. Los demás no tienen fórmula con sentido físico: `Subflow Fwd/Bwd Packets` (2 valores) frente a los conteos; `Idle Max` frente a `Flow IAT Max` (soporte 0,73); flags y `Idle/Active Std` frente a `Flow IAT Std` (soporte 0,50, Y casi único). No eliminan nada.

**Comprobación posterior (3.6, después de congelar):** 0 fallos en `val_benign`, `test_benign` y en todas las etiquetas de ataque (`comprobacion_posterior.csv`). Cumplir el teorema I(X;Y|S) = 0 exige que las identidades valgan también en el ataque, y valen.

**Redundancia residual (descriptiva):** los 20 pares con mayor u(X|Y) están en `redundancia_top20.csv` (IM con la corrección de Sharmin, B = 20). Destacan los tres «Bulk» de forward (u = 1,00), `Protocol` frente a `Fwd Seg Size Min` (0,996) y `Packet Length Std` frente a `Variance` (0,98). El histograma es una **cota inferior** de la IM verdadera (procesamiento de datos). Estas redundancias se conservan a propósito: no son nulas.

**Comparación de listas** (`informacion_resumen.json`, `pertenencia_conjuntos.csv`):
- Nuevo (66) frente al actual (51): 45 en común. Salen del actual 6 cocientes: `Bwd IAT Mean`, `Flow IAT Mean`, `Fwd IAT Mean`, `Fwd Packet Length Mean`, `Packet Length Mean` y `Packet Length Min`.
- Vuelven 21, entre ellas `Total Length of Bwd Packet`, `Total Bwd packets`, `Bwd Header Length` y los totales de IAT.
- Frente a `B'` (51): 46 en común.

## 4. Preprocesamiento

- Filtro D4 (`Flow Duration` < 0) dentro de cada partición, después de dividir: train 24, val 7, test 0, ataques 0 (31 en total, como dice el prompt).
- Particiones idénticas al original antes del filtro: comparadas columna a columna (26 columnas sin escalar) en las 4 particiones; hashes SHA-256 de los `row_id` antes, después y de las eliminadas en `data/processed/oe1_saneado/datos_info.json`.
- Regla de inf/NaN: idéntica. Las 793 filas eliminadas tienen inf/NaN en `Flow Bytes/s` y `Flow Packets/s`, que siguen en el conjunto: quitar la regla no devolvería ninguna.
- Escalado MinMax solo con `train_benign`, parámetros en `float64` (`escalador_minmax_float64.json`). El modelo recibe `float32`.
- Resolución (V1 repetida, `outputs/oe1_saneado/preproceso/resolucion_por_columna.csv`): ninguna columna pierde más del 50 % de sus valores distintos. La que más pierde es `Idle Mean`, que conserva el 90,6 %; las demás conservan al menos el 97,2 %. `Flow Packets/s` conserva el 99,2 % (antes 1,75 %). **Problema reportado:** 18 columnas enteras de rango grande (duraciones, IAT, bytes, Idle/Active) tienen, en el extremo del rango, una separación `float32` mayor que 1 unidad (hasta 163 en `Bwd Bulk Rate Avg`). Es la precisión de entrada del modelo (`float32`), no del preprocesamiento. El prompt pedía detenerse aquí; por instrucción del autor («pasa derecho») se siguió y se documenta.
- `Flow IAT Min` negativos: se conservan (D4) y se documentan. Tras el filtro: train 1873, val 375, test 414, ataques 229. El prompt daba 1897 en train; la diferencia (24) coincide con los flujos de duración negativa filtrados.

## 5. Configuración y reproducibilidad

- `configs/oe1_saneado.yaml`: único archivo de configuración. `vae_nids.oe1.config.load_config` falla si falta cualquier clave; la línea de comandos solo acepta `--config` y `--seed`.
- `configs/features/oe1_info_nula.yaml`: conjunto congelado, estado de cada una de las 77 columnas y SHA-256 de la lista (`49435ba7…5d97`), comprobado al cargar.
- Checkpoints (`outputs/oe1_saneado/checkpoints/*.pt`, fuera de git): configuración completa, columnas y hash, escalador `float64`, commit y árbol limpio, hash de `train_benign`, versiones de Python, torch y numpy, semilla, hilos, época y motivo de parada.
- Determinismo: semillas de `random`, `numpy` y `torch`, `torch.use_deterministic_algorithms(True)` y 8 hilos. Dos corridas de humo con la semilla 42 dieron pérdidas idénticas.
- Tests: `tests/oe1_saneado/` (8 tests, todos en verde). Cubren identidades sintéticas, perturbación detectada, determinismo y no circularidad, que la selección no lea ataques, el filtro D4, el escalador solo con train y en `float64`, la configuración obligatoria y el checkpoint autocontenido.
- Portabilidad: rutas relativas a la raíz del repo, con variables de entorno `VAE_NIDS_RAW_DIR`, `VAE_NIDS_PROCESSED_DIR` y `VAE_NIDS_OUTPUTS_DIR`.

## 6. Hiperparámetros

Ver [`hiperparametros.md`](hiperparametros.md).

## 7. Reentrenamiento y antes/después

**Pendiente: se corre en la VM** (guía: [`EJECUTAR_EN_VM.md`](EJECUTAR_EN_VM.md)).

- En la máquina local se alcanzaron a entrenar las semillas 42 a 48. Mejores épocas: 108, 49, 96, 73, 126, 82 y 97; todas pararon por paciencia, ninguna por el tope de 250 épocas.
- Falta la semilla 49, la referencia de 74 y los diagnósticos.
- Comando: `python -m vae_nids.oe1.run --config configs/oe1_saneado.yaml --ref74 configs/oe1_saneado_ref74.yaml`.
- Ya está calculado el «antes» (OE1 actual, 51 columnas, semillas 42–46): 5,4 unidades activas de media (6, 5, 6, 5, 5), mejor época entre 52 y 86, y la mediana de log σ² del decodificador en el piso en 50–51 de las 51 columnas (`outputs/oe1_saneado/tables/diagnosticos_oe1_actual_51.csv`).

## 8. Cambios que afectan a la monografía y al documento OB1 (no se editaron)

- **Nota al pie de §6.2** (13 clusters, Pearson |r| ≥ 0,95): reemplazarla por el criterio de información nula. Reducción 77 → 66 por copias y funciones exactas, sin umbral y solo con `train_benign`.
- **§6.3 (Control de fuga de datos):** declarar que la reducción anterior se calculó con datos mezclados y que el saneamiento la rehízo solo con `train_benign`; mencionar el filtro D4.
- **§3 (An y Cho):** matizar que, con la varianza del decodificador en el piso, la NLL se comporta como un error cuadrático de pesos casi uniformes (cifra de la sección 7).
- **Tabla de clusters del OB1:** sustituirla por la tabla de columnas eliminadas (sección 3) y la lista de casi funciones.

## 9. Limitaciones

- Conservar la información no garantiza el mismo rendimiento de detección; eso se mide después, con ataques, en otra tarea.
- Las 6 casi funciones dependen de una decisión del autor sobre ε.
- La entrada al modelo sigue en `float32`: 18 columnas tienen resolución peor que 1 unidad en el extremo de su rango.
- Referencias marcadas «por verificar»: An y Cho (2015), Higgins et al. (2017), Prechelt (1998), Kuhn y Johnson (2013) y Cover y Thomas (2006), por no poder abrir los PDF desde esta máquina.
- El barrido del OE1 actual tiene 5 semillas; la comparación antes/después usa esas 5 frente a las 8 nuevas.

## 10. Archivos generados

| Archivo | Contenido |
|---|---|
| `configs/oe1_saneado.yaml`, `configs/oe1_saneado_ref74.yaml`, `configs/oe1_saneado_humo.yaml` | Configuración única, referencia de 74 y prueba de humo |
| `configs/features/oe1_info_nula.yaml` | Conjunto congelado (66) con el estado de cada columna |
| `configs/features/referencia_74.yaml` | 77 menos las 3 copias |
| `configs/features/alternativos/*.yaml` | `B_prima_51`, `Robusto_TB_56`, `info_nula_eps_ampliado_60` (solo registrados) |
| `src/vae_nids/oe1/` | `config`, `datos`, `seleccion`, `informacion`, `alternativos`, `preproceso`, `entrenamiento`, `diagnosticos`, `antes`, `run` |
| `tests/oe1_saneado/test_oe1_saneado.py` | 8 tests |
| `outputs/oe1_saneado/seleccion/` | Estado por columna, casi funciones, reconstrucción, búsqueda sistemática, comprobación posterior, entropía, redundancia, pertenencia |
| `outputs/oe1_saneado/preproceso/` | Resolución por columna y resumen |
| `outputs/oe1_saneado/tables/`, `figures/`, `logs/` | Diagnósticos por semilla, log σ² del decodificador, curvas y figuras |
| `docs/oe1_saneado/hiperparametros.md` | Tabla de hiperparámetros |
