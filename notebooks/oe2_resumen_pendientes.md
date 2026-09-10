# OE2 — Resumen de estado final y pendientes

Commit `1eabe45` — "OE2: cierre parcial — comparación mu/z, propuesta de
alcance y reconciliación de hipótesis (tau pendiente de confirmación)".
Documento completo: `oe2_cierre_borrador.md`. Sin push.

**Nota**: el mensaje de commit se cambió respecto al pedido original
("OE2: cierre completo — tau=99, ...") porque τ **no quedó resuelto** —
ver punto 1.

## 1. τ (95 vs. 99) — BLOQUEADO, no resuelto

Sí hay cambios dramáticos de TPR entre ambos percentiles, así que no se
decidió por cuenta propia ni se tocó nada de la Fase 4/5 (siguen con
τ=percentil 95, como estaban):

| group | TPR@τ95 | TPR@τ99 | Δ |
|---|---:|---:|---:|
| **DoS_slowloris_attempted** | 0.9578 | **0.0023** | -0.9555 — casi total → casi nulo, calza exacto con el criterio de "~0 a alto" |
| SSH_Patator | 0.9842 | 0.2369 | -0.7473 — caída fuerte, no llega a ~0 |
| DDoS | 0.7937 | 0.1446 | -0.6491 — caída fuerte, n=95,123 |

**Pendiente**: confirmar qué percentil usar. Hasta entonces, Fase 4 y
Fase 5 siguen con τ=95.

## 2. Mu vs. z muestreado — IMPLEMENTADO

Variante paralela, sin tocar la original:
- Script: `src/vae_nids/evaluation/detectability_z_sampled_oe2.py`
- Resultado: `outputs/metrics/oe2_detectability_z_sampled.csv`
- Original intacto: `outputs/metrics/oe2_detectability.csv` (con mu)

**Hallazgo clave**: AUC-ROC casi no cambia entre mu y z (diferencia
media ≈0.003, máxima 0.0124 en `SSH_Patator`) — confirma que la elección
mu/z **no** explica el resultado no significativo de la Fase 5.

TPR@τ sí es más sensible al ruido de una sola muestra de z:
`SSH_Patator` cae 19 puntos porcentuales (0.9842 mu → 0.7923 z),
consistente con lo anticipado en el reporte de factibilidad original.

## 3. Propuesta de exclusión del Experimento 4 — PENDIENTE DE APROBACIÓN

Texto exacto incorporado en `oe2_cierre_borrador.md` (Sección 11):

> Se propone excluirlo del alcance de esta monografía por restricción de
> tiempo/alcance: los Experimentos 2 y 3, ya consolidados en las Fases
> 1-6, cubren el núcleo verificable de OE2. El Experimento 4 introduce
> una pregunta distinta y no trivial —caracterizar los límites
> operacionales de τ frente a tráfico benigno complejo que se solapa con
> ataques sigilosos—, que requeriría identificar o construir tráfico
> benigno de alto volumen/ráfaga, definir cuantificablemente qué cuenta
> como "solapamiento", y probablemente esperar a que se resuelva primero
> la calibración de τ. Dado el cronograma de 13 semanas y que OE3/OE4
> siguen pendientes, absorber esa carga arriesga el cierre del resto de
> objetivos. Si se aprueba, debe quedar documentada como limitación de
> alcance explícita, no como vacío no reconocido.

## 4. Reconciliación de la hipótesis — incorporada, sin suavizar

Párrafo agregado en `oe2_cierre_borrador.md` (Sección 10.1): reporta que
la hipótesis de la Sección 4 de la monografía no se sostiene (rho=0.38,
p=0.25, n=11), con `Infiltration` como contradicción directa y nombrada
(más detectable de las 18 pese a no ser la más cercana geométricamente).

## Qué falta de tu parte

1. **τ**: 95 (código actual) o 99 (spec vigente) — con la tabla del
   punto 1 en mano.
2. **Experimento 4**: aprobar o rechazar la propuesta de exclusión del
   punto 3.
3. (Opcional) Decidir si `oe2_detectability.csv` (mu) o
   `oe2_detectability_z_sampled.csv` (z) queda como oficial — el impacto
   en la conclusión central es bajo, así que no es urgente.
