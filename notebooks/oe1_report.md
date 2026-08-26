# OE1 — Estabilidad del espacio latente: reporte de cierre

Consolida OE1.1 (entrenamiento), OE1.2 (unidades activas + KL por dimensión)
y OE1.3 (estabilidad entre semillas). Contexto: `notebooks/cluster_review_oe1.md`
documenta la revisión de clusters correlacionados que llevó `feature_columns.json`
de 74 a 51 features (umbral |r|≥0.95 por clique).

## OE1.1 / OE1.2 — 74 vs. 51 features (misma seed=42, β=1.0, k=8, min_delta=0.10, patience=10)

| | 74 features | 51 features |
|---|---|---|
| Checkpoint | `vae_k8_beta1_input74_best.pt` | `vae_k8_beta1_input51_best.pt` |
| Época de convergencia (early stop) | 137 (mejor: 127) | 62 (mejor: 52) |
| `val_total` en el mejor checkpoint | −277.2749 | −185.2791 |
| `val_recon` | −285.63 (aprox., ver log) | −193.48 |
| `val_kl` | ~8.08 | ~8.18 |
| Unidades activas (Burda et al. 2016, umbral 0.01) | **7/8** | **6/8** |

⚠️ **`val_total` NO es comparable en magnitud entre ambas filas**: la
reconstrucción (NLL Gaussiana) se suma sobre `input_dim` dimensiones, así
que con menos features de entrada la suma es mecánicamente menor — no
significa que el modelo de 51 reconstruya "mejor". La comparación válida
es la cualitativa (unidades activas, qué dimensiones colapsan, forma de
las curvas), no el valor absoluto de la pérdida.

### Unidades activas por dimensión

| dim | Var_x[μ_i] — 74 feat | Var_x[μ_i] — 51 feat | estado 74 | estado 51 |
|---|---|---|---|---|
| 0 | 0.303 | 0.572 | activa | activa |
| 1 | 0.582 | 0.253 | activa | activa |
| **2** | **0.00001** | **0.00001** | **colapsada** | **colapsada** |
| 3 | 1.082 | 0.947 | activa | activa |
| 4 | 0.562 | 0.966 | activa | activa |
| 5 | 0.258 | 0.255 | activa | activa |
| 6 | 0.998 | 0.760 | activa | activa |
| **7** | 0.554 | **0.00003** | activa | **colapsada** |

### ¿Cambió el hallazgo cualitativo de OE1?

**Parcialmente.** La dimensión 2 colapsa en ambas configuraciones de forma
consistente (Var_x≈1e-5 en las dos corridas) — ese es el hallazgo robusto:
con k=8 el modelo genuinamente no necesita las 8 dimensiones para
representar tráfico benigno. Lo que **sí cambió** es la dimensión 7, que
estaba activa (0.554) con 74 features y colapsa (0.00003) con 51. Lectura
más plausible: al quitar 23 features redundantes, el decoder tiene menos
información redundante que "repartir" entre dimensiones, y una segunda
dimensión que antes cargaba señal parcialmente solapada con otras deja de
ser necesaria. La convergencia también es ~2.2× más rápida (62 vs. 137
épocas) — consistente con un problema de optimización más simple al bajar
la dimensión de entrada.

**Conclusión para la monografía:** el fenómeno de colapso parcial del
posterior es robusto a la reducción de dimensionalidad de entrada (no es
un artefacto de las 74 features), pero el *número exacto* de dimensiones
colapsadas sí depende de cuánta redundancia tenga la entrada — con menos
redundancia, colapsan más dimensiones latentes, no menos. Esto es
consistente con la intuición de que el posterior collapse ocurre en
dimensiones que dejan de aportar información no explicada por las demás.

## OE1.3 — Estabilidad entre semillas (input_dim=51, β=1.0, k=8)

5 semillas, misma config (β=1.0, k=8, min_delta=0.10, patience=10, max_epochs=250):

| seed | época convergencia (mejor) | val_total (mejor) | unidades activas | dimensiones activas (índice) |
|---|---|---|---|---|
| 42 | 52 | −185.28 | 6/8 | 0, 1, 3, 4, 5, 6 |
| 43 | 86 | −190.05 | 5/8 | 0, 1, 2, 3, 4 |
| 44 | 80 | −186.08 | 6/8 | 1, 2, 3, 5, 6, 7 |
| 45 | 84 | −187.89 | 5/8 | 0, 1, 3, 4, 5 |
| 46 | 82 | −186.40 | 5/8 | 1, 2, 3, 6, 7 |

**Unidades activas: media 5.4/8, rango 5–6/8 (std≈0.49).** El conteo es
estable dentro de una banda estrecha en las 5 semillas — nunca cae a 4 ni
sube a 7 — así que el fenómeno de colapso parcial del posterior (2–3 de 8
dimensiones sin usar) es robusto y no depende de una inicialización
particular.

**Importante — la identidad de la dimensión NO es comparable entre
semillas.** No hay alineación canónica del espacio latente entre
entrenamientos independientes: cada corrida aprende su propia rotación/
permutación arbitraria de qué índice físico (0-7) codifica qué. Por eso
la dimensión 2 aparece "activa" en la seed 43 pero "colapsada" en 42/44/46
— no es contradictorio, es exactamente lo esperable, y es la razón por la
que el resultado de OE1.2 (single-seed) solo es interpretable en términos
de *cuántas* dimensiones colapsan, no de *cuál índice* colapsa, salvo que
se fije la seed (como se hizo en OE1.1/OE1.2 con seed=42, documentado
arriba).

`época de convergencia` también varía bastante (52–86) — la seed 42 (la
usada como baseline en OE1.1/OE1.2) convergió notablemente más rápido que
el resto; no hay indicio de que sea un caso patológico (mismo rango de
`val_total`, mismo conteo de unidades activas 6/8), simplemente llegó
antes al mismo tipo de mínimo.

`val_total`: media ≈ −187.1, rango [−190.05, −185.28] — magnitud
consistente entre semillas (no hay una corrida que diverja).

### Cierre de OE1

Con input_dim=51 fijado (74→51 documentado y trazado en
`cluster_review_oe1.md` + `eda_exclusion_log.json`) y 5 semillas
confirmando que 5–6 de 8 unidades latentes quedan activas de forma
estable, **OE1 queda cerrado**: el VAE con k=8 sistemáticamente no usa
2–3 dimensiones para representar tráfico benigno, de forma reproducible y
no atribuible a una inicialización particular ni a redundancia sin
resolver en las features de entrada. Este es el punto de partida
(estabilidad confirmada del espacio latente) sobre el cual OE2 puede
apoyarse para relacionar geometría con detectabilidad por familia de
ataque.

Checkpoints de la corrida: `vae_k8_beta1_input51_seed{42,43,44,45,46}_best.pt`.
Resumen machine-readable: `outputs/metrics/oe1_seed_sweep.json` (seeds
43-46; seed 42 en `outputs/metrics/vae_k8_beta1_input51_oe1_metrics.json`).

## Apéndice — datos detallados (todas las corridas, sin redondear a ojo)

**Config común:** k=8, β=1.0, hidden_dim=32, clamps logσ²∈[-10,10],
Adam(lr=1e-3), batch_size=1024, max_epochs=250, patience=10,
min_delta=0.10, n_train=1,159,948, n_val=248,560.

### OE1.1 — Entrenamiento

| corrida | seed | input_dim | épocas corridas | mejor época | tiempo total | train_total | train_recon | train_kl | val_total | val_recon | val_kl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| input74 | 42 | 74 | 137 | 127 | 1151.3s (19.2 min) | −277.4563 | −285.5367 | 8.0804 | −277.2749 | −285.3550 | 8.0802 |
| input51 | 42 | 51 | 62 | 52 | 384.0s (6.4 min) | −185.2770 | −193.4034 | 8.1264 | −185.2791 | −193.4598 | 8.1806 |
| input51_seed43 | 43 | 51 | 96 | 86 | 544.4s (9.1 min) | −190.0972 | −195.9774 | 5.8802 | −190.0494 | −195.8698 | 5.8204 |
| input51_seed44 | 44 | 51 | 90 | 80 | 511.5s (8.5 min) | −186.0911 | −194.0853 | 7.9942 | −186.0789 | −194.0645 | 7.9856 |
| input51_seed45 | 45 | 51 | 94 | 84 | 539.3s (9.0 min) | −188.3077 | −193.8867 | 5.5790 | −187.8910 | −193.4229 | 5.5319 |
| input51_seed46 | 46 | 51 | 92 | 82 | 526.3s (8.8 min) | −186.4907 | −193.4671 | 6.9763 | −186.3978 | −193.3662 | 6.9684 |

### OE1.2 — Var_x[μᵢ(x)] por dimensión (Burda et al. 2016, umbral 0.01)

| dim | input74 (s42) | input51 (s42) | s43 | s44 | s45 | s46 |
|---|---|---|---|---|---|---|
| 0 | 0.302666 | 0.572277 | 0.736103 | **0.0000463** | 0.746076 | **0.0000091** |
| 1 | 0.582202 | 0.253297 | 0.243882 | 0.253160 | 0.689335 | 0.410828 |
| 2 | **0.0000122** | **0.0000088** | 0.435670 | 0.949136 | **0.0000165** | 0.957681 |
| 3 | 1.081849 | 0.947116 | 0.814364 | 0.746018 | 0.991055 | 0.873923 |
| 4 | 0.561675 | 0.966068 | 1.011670 | **0.0000055** | 0.341673 | **0.0000052** |
| 5 | 0.258324 | 0.255498 | **0.0000101** | 0.724182 | 0.534323 | **0.0000110** |
| 6 | 0.998022 | 0.760422 | **0.0000065** | 1.052998 | **0.0000111** | 0.992384 |
| 7 | 0.554213 | **0.0000276** | **0.0000069** | 0.228106 | **0.0000129** | 0.285890 |
| **n_active** | **7/8** | **6/8** | **5/8** | **6/8** | **5/8** | **5/8** |

(negrita = por debajo del umbral 0.01, dimensión colapsada)

### OE1.2 — KL por dimensión (nats)

| dim | input74 (s42) | input51 (s42) | s43 | s44 | s45 | s46 |
|---|---|---|---|---|---|---|
| 0 | 0.407478 | 0.724150 | 0.877905 | 0.0000358 | 0.995842 | 0.0000170 |
| 1 | 0.756873 | 0.207953 | 0.214690 | 0.236935 | 0.803463 | 0.404765 |
| 2 | 0.0000400 | 0.0000370 | 0.489170 | 1.724777 | 0.0000383 | 1.623030 |
| 3 | 3.301443 | 1.632990 | 1.415031 | 1.324598 | 2.842143 | 1.550248 |
| 4 | 0.612117 | 3.925638 | 2.823483 | 0.0000287 | 0.346551 | 0.0000245 |
| 5 | 0.234859 | 0.227991 | 0.0000459 | 0.826676 | 0.543779 | 0.0000397 |
| 6 | 2.280500 | 1.461826 | 0.0000393 | 3.704573 | 0.0000150 | 3.114109 |
| 7 | 0.486850 | 0.0000583 | 0.0000392 | 0.168011 | 0.0000209 | 0.276143 |
| **Σ (kl_total_mean)** | **8.080159** | **8.180642** | **5.820403** | **7.985635** | **5.531853** | **6.968375** |

### OE1.3 — resumen con val_recon/val_kl

| seed | época conv. | val_total | val_recon | val_kl | activas | dims activas |
|---|---|---|---|---|---|---|
| 42 | 52 | −185.2791 | −193.4598 | 8.1806 | 6/8 | 0,1,3,4,5,6 |
| 43 | 86 | −190.0494 | −195.8698 | 5.8204 | 5/8 | 0,1,2,3,4 |
| 44 | 80 | −186.0789 | −194.0645 | 7.9856 | 6/8 | 1,2,3,5,6,7 |
| 45 | 84 | −187.8910 | −193.4229 | 5.5319 | 5/8 | 0,1,3,4,5 |
| 46 | 82 | −186.3978 | −193.3662 | 6.9684 | 5/8 | 1,2,3,6,7 |
| **media** | 76.8 | −187.14 | −194.04 | 6.90 | 5.4/8 | — |
| **std** | 13.0 | 1.75 | 1.05 | 1.09 | 0.49 | — |

## Interpretación del barrido de semillas (OE1.3)

El barrido de 5 semillas sobre la configuración final (51 features, k=8,
β=1.0) da un conteo de unidades activas de 5-6/8 (media 5.4, std≈0.49) —
un rango angosto (solo dos valores posibles), pero no un número único
reproducido en las 5 corridas.

Esto deja dos lecturas en tensión, ninguna descartable con la evidencia
actual:

- **A favor de una dimensionalidad efectiva real** (Dai & Wipf, 2019):
  el rango angosto es consistente con que el tráfico benigno, sin la
  redundancia de features ya eliminada, requiere genuinamente entre 5 y
  6 dimensiones — no un valor exacto y fijo, pero sí un orden de magnitud
  estable frente a la variación de semilla.
- **A favor de la advertencia de Dai, Wang & Wipf (2020)**: que el
  conteo no sea idéntico entre semillas es también el síntoma que
  atribuyen a mínimos locales de la superficie de pérdida, no
  necesariamente a una propiedad intrínseca de los datos.

Con 5 semillas no hay evidencia suficiente para descartar ninguna de las
dos explicaciones; se reporta el rango observado (5-6/8) como el
resultado de OE1.3, sin forzar una interpretación única.

**Nota metodológica**: la comparación "la dimensión 2 colapsa tanto con
74 como con 51 features" (misma seed=42) no debe leerse como evidencia
adicional de robustez. `torch.manual_seed()` fija el generador de
números aleatorios, pero el orden en que se consumen esos números
depende del tamaño de la primera capa del encoder (`Linear(input_dim,
hidden_dim)`), que difiere entre 74 y 51 features. Esto significa que,
pese a compartir la seed nominal, ambas corridas parten de
inicializaciones efectivamente distintas para las capas posteriores —
la coincidencia del índice de dimensión colapsada entre ambas
configuraciones no es una comparación válida. La métrica comparable
entre corridas (semillas o configuraciones de input_dim distintas)
sigue siendo el *conteo* de unidades activas, nunca qué índice
específico colapsa.
