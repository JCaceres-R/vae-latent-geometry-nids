# Revisión de clusters correlacionados — pre-OE1

Complementa `data/processed/eda_exclusion_log.json` (`pending_review_clusters`).
Ese log solo guarda el rango agregado (`corr_min_abs`/`corr_max_abs`) por
cluster; este documento trae las matrices de correlación par-a-par reales
(recalculadas sobre el dataset completo sanitizado, benigno+malicioso
mezclado, mismo procedimiento que la Sección 4.2 del EDA) y aplica el
criterio de decisión acordado:

**Umbral: |r| ≥ 0.95 (Pearson), verificado por clique (no single-linkage).**
Una clique es un subconjunto donde TODOS los pares cruzan el umbral entre
sí — encadenar A-B y B-C sin que A-C cruce el umbral no cuenta. El motivo
de ser estrictos: evitar colapsar por transitividad asimetrías
forward/backward que Engelen et al. (2021) señalan como relevantes para
detectar ciertas familias de ataque (p. ej. IAT en DoS Slowhttptest).

Nada de esto está aplicado todavía a `feature_columns.json` ni se
reentrenó OE1 — es la base para decidir antes de tocar esos archivos.

## Cluster 2 (pedido explícito — el rango agregado no alcanzaba para decidir)

| | Bwd IAT Max | Flow IAT Max | Fwd IAT Max | Idle Max | Idle Mean | Idle Min |
|---|---|---|---|---|---|---|
| **Bwd IAT Max** | 1.000 | 0.960 | 0.958 | 0.958 | 0.952 | **0.927** |
| **Flow IAT Max** | | 1.000 | 0.999 | 0.998 | 0.985 | 0.954 |
| **Fwd IAT Max** | | | 1.000 | 0.997 | 0.984 | 0.953 |
| **Idle Max** | | | | 1.000 | 0.987 | 0.956 |
| **Idle Mean** | | | | | 1.000 | 0.990 |

De los 15 pares posibles, solo uno falla el umbral: `Bwd IAT Max` ↔
`Idle Min` = 0.927. Eso rompe la clique de 6 en dos cliques máximas de 5
que se solapan en 4 miembros, y `Bwd IAT Max` / `Idle Min` nunca
coinciden en la misma clique.

**Resultado (clique cerrada, sin ambigüedad):**
- Colapsa: `Flow IAT Max`, `Fwd IAT Max`, `Idle Max`, `Idle Mean` → 1 representante, 3 fuera.
- **No colapsan**: `Bwd IAT Max` e `Idle Min` quedan sueltos (no forman clique entre sí ni cada uno con el núcleo a la vez). `Bwd IAT Max` es justo la variable con asimetría fwd/bwd a preservar.

## Cluster 0 (7 features, tras excluir `Packet Length Variance`)

| | Bwd Max | Bwd Mean | Bwd Std | Pkt Max | Pkt Mean | Pkt Std | Subflow Bwd Bytes |
|---|---|---|---|---|---|---|---|
| **Bwd Pkt Len Max** | 1.000 | 0.951 | 0.984 | 0.989 | 0.908 | 0.983 | 0.918 |
| **Bwd Pkt Len Mean** | | 1.000 | 0.938 | 0.938 | 0.968 | 0.963 | 0.981 |
| **Bwd Pkt Len Std** | | | 1.000 | 0.973 | 0.870 | 0.982 | 0.881 |
| **Packet Len Max** | | | | 1.000 | 0.913 | 0.988 | 0.906 |
| **Packet Len Mean** | | | | | 1.000 | 0.935 | 0.988 |
| **Packet Len Std** | | | | | | 1.000 | 0.930 |

**Resultado: dos cliques limpias, cubren las 7 features sin solape:**
- Clique A (4): `Bwd Packet Length Max`, `Bwd Packet Length Std`, `Packet Length Max`, `Packet Length Std`
- Clique B (3): `Bwd Packet Length Mean`, `Packet Length Mean`, `Subflow Bwd Bytes`

⚠️ Los miembros de cada clique NO son alias del mismo nombre (a diferencia
de los 3 duplicados exactos ya resueltos) — son features bwd-only vs.
overall conceptualmente distintas que solo correlacionan alto en este
dataset. **Representante pendiente de elegir por clique.**

## Cluster 3 (6 features)

| | Bwd IAT Mean | Bwd IAT Min | Flow IAT Mean | Flow IAT Std | Fwd IAT Mean | Fwd IAT Min |
|---|---|---|---|---|---|---|
| **Bwd IAT Mean** | 1.000 | 0.955 | 0.783 | 0.949 | 0.948 | 0.902 |
| **Bwd IAT Min** | | 1.000 | 0.741 | 0.861 | 0.927 | 0.950 |
| **Flow IAT Mean** | | | 1.000 | 0.783 | 0.917 | 0.868 |
| **Flow IAT Std** | | | | 1.000 | 0.922 | 0.830 |
| **Fwd IAT Mean** | | | | | 1.000 | 0.967 |

**Resultado: NO hay clique de 3+.** Solo dos pares aislados cruzan 0.95:
- `Bwd IAT Mean` ↔ `Bwd IAT Min` = 0.955 → colapsa (por separado)
- `Fwd IAT Mean` ↔ `Fwd IAT Min` = 0.967 → colapsa (por separado)
- `Bwd IAT Min` ↔ `Fwd IAT Min` = 0.950 (cruza el umbral pero ambos miembros ya están tomados por su propio par — no se usa para fusionar, por diseño: preserva la asimetría fwd/bwd)
- `Flow IAT Mean` y `Flow IAT Std` quedan sueltos (ninguno alcanza 0.95 con nada).

## Cluster 1 (6 features)

| | ACK Flag Count | Bwd Header Length | Fwd Header Length | Total Bwd packets | Total Fwd Packet | Total Length of Bwd Packet |
|---|---|---|---|---|---|---|
| **ACK Flag Count** | 1.000 | 0.999 | 0.999 | 1.000 | 1.000 | 0.996 |
| **Bwd Header Length** | | 1.000 | 0.999 | 1.000 | 0.998 | 0.994 |
| **Fwd Header Length** | | | 1.000 | 0.999 | 1.000 | 0.996 |
| **Total Bwd packets** | | | | 1.000 | 0.999 | 0.994 |
| **Total Fwd Packet** | | | | | 1.000 | 0.997 |

**Resultado: clique completa de 6** (mínimo par 0.994) → 1 representante, 5 fuera.

⚠️ Colapsa 6 métricas de naturaleza distinta (conteo de flags, bytes de
header, conteo de paquetes) en una sola, **incluyendo fwd y bwd juntos**
(`Fwd Header Length` ↔ `Bwd Header Length` = 0.999). Acá la asimetría
fwd/bwd **no se sostiene empíricamente** en este dataset — confirmar
explícitamente antes de colapsar, no dejar pasar en automático solo
porque la regla mecánica lo permite.

## Cluster 4 (post-duplicado exacto — quedan 2 de 3)

`Fwd Packet Length Mean` ↔ `Subflow Fwd Bytes` = **0.975** → colapsa.
(El tercer miembro original, `Fwd Segment Size Avg`, ya se excluyó en la
Sección 7.3 como duplicado exacto de `Fwd Packet Length Mean`.)

## Cluster 5 (3 features)

| | Bwd IAT Total | Flow Duration | Fwd IAT Total |
|---|---|---|---|
| **Bwd IAT Total** | 1.000 | 0.983 | 0.982 |
| **Flow Duration** | | 1.000 | 0.9995 |

**Resultado: clique completa de 3** → 1 representante, 2 fuera. También
mezcla fwd+bwd+flow, mismo tipo de nota que el cluster 1 (aquí es más
esperable: IAT Total ≈ Flow Duration por definición).

## Clusters de 2 features (resueltos directamente por el umbral)

| cluster | par | r | resultado |
|---|---|---|---|
| 6 | Fwd URG Flags ↔ URG Flag Count | 0.9996 | colapsa |
| 7 | Bwd Bytes/Bulk Avg ↔ Bwd Packet/Bulk Avg | 0.979 | colapsa |
| 8 | Flow Packets/s ↔ Fwd Packets/s | 0.967 | colapsa |
| 9 | Bwd PSH Flags ↔ PSH Flag Count | 0.956 | colapsa (al límite) |
| **10** | Fwd Packet Length Max ↔ Fwd Packet Length Std | **0.948** | **NO colapsa** |
| **11** | Bwd IAT Std ↔ Fwd IAT Std | **0.934** | **NO colapsa** |
| **12** | Active Mean ↔ Active Min | **0.907** | **NO colapsa** |

## Pendiente de decisión humana antes de tocar `feature_columns.json`

- [ ] **Cluster 0** — representante de Clique A (Bwd Max/Std vs. Packet Max/Std) y de Clique B (Bwd Mean vs. Packet Mean vs. Subflow Bwd Bytes).
- [ ] **Cluster 1** — confirmar que sí se colapsa pese a mezclar fwd/bwd, y elegir representante entre los 6.
- [ ] **Cluster 2** — confirmar la resolución (colapsar el núcleo de 4, dejar sueltos Bwd IAT Max e Idle Min) y elegir representante del núcleo.
- [ ] **Cluster 3** — elegir representante de cada par ({Bwd Mean, Bwd Min} y {Fwd Mean, Fwd Min}).
- [ ] **Cluster 5** — elegir representante entre Bwd IAT Total / Flow Duration / Fwd IAT Total.
- [ ] Representantes triviales de clusters 4, 6, 7, 8, 9 (2 features cada uno, sin ambigüedad semántica aparente) — proponer default o confirmar.

Una vez cerrado esto: regenerar `feature_columns.json`, actualizar
`eda_exclusion_log.json` con el detalle de cada decisión (corr real +
razón), marcar el checkpoint `vae_k8_beta1` como inválido (input_dim
cambia), y repetir OE1.1/OE1.2 con el nuevo `input_dim`, documentando 74
features vs. el nuevo conteo en el mismo reporte.
