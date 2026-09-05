# OE2 — Tarea 2: factibilidad de usar z muestreado en vez de mu

Reporte de factibilidad únicamente — **no se implementó nada**, la Fase 4
sigue usando `mu` tal como está.

## 1. ¿Existe ya el reparameterization trick?

Sí. `VAE.reparameterize` en `src/vae_nids/models/vae.py:89-94`:

```python
@staticmethod
def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """z = mu + sigma * eps, eps ~ N(0, I) -- truco de reparametrización."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps
```

Y ya está encadenado end-to-end en `VAE.forward` (`vae.py:96-108`) — un
solo `model(x)` ya devuelve `mu`, `logvar`, `z` (muestreado) y
`mu_x_hat`/`logvar_x_hat` **calculados a partir de ese z**, no de `mu`.
Esto es exactamente lo que se usa en `training/train.py` para entrenar.

## 2. Qué tan fácil es reusarlo en `detectability_oe2.py`

**Es un cambio pequeño y localizado, no estructural.** El código actual
hace, en dos funciones separadas:

```python
# encode_mu(): mu, _logvar = model.encoder(x); return mu
# recon_nll_scores(model, x, mu): mu_x_hat, logvar_x_hat = model.decoder(mu); ...
```

Para usar z en vez de mu, el cambio es literalmente reemplazar esas dos
llamadas por una: `outputs = model(x)` ya trae `outputs["mu_x_hat"]` y
`outputs["logvar_x_hat"]` calculados desde `outputs["z"]`. No hay que
tocar shapes, batching, ni el resto del pipeline (la fórmula de NLL por
muestra es idéntica, solo cambia qué vector latente alimenta al decoder).

**Lo que sí complica, y no es cosmético:**

1. **Deja de ser determinista.** Con `mu`, correr el script dos veces da
   exactamente el mismo score por fila. Con `z` muestreado, cada corrida
   saca un `eps ~ N(0,I)` distinto → los scores (y por lo tanto AUC-ROC,
   TPR@tau, y el bootstrap de CI de la Fase 4) cambian de una corrida a
   otra si no se fija `torch.manual_seed(...)` explícitamente antes de
   cada `model(x)`. Ahora mismo el script no fija semilla de torch en
   absoluto porque no la necesita (todo es determinista).
2. **Una sola muestra de z es una estimación ruidosa de novedad.** El
   score con `mu` es la reconstrucción "más probable"; el score con un
   solo `z` muestreado es una realización de una variable aleatoria (el
   ELBO reconstruction term es un estimador Monte Carlo de 1 muestra).
   Para que el score sea estable/comparable entre grupos, lo más
   correcto no es un solo `z`, sino promediar el NLL sobre K muestras de
   z por fila (K=10, 50...) — más fiel a "log-verosimilitud de
   reconstrucción" en expectativa, pero también más caro (K forward
   passes del decoder por muestra en vez de 1).
3. **No se puede reusar `outputs/latent_vectors/*.npy` tal cual.** Esos
   archivos solo guardan `mu` (así se pidió explícitamente en la Fase 2:
   "guarda ÚNICAMENTE mu... nunca muestrees z"), no `logvar` — sin
   `logvar` no se puede reparametrizar. La solución no requiere tocar la
   Fase 2: como `detectability_oe2.py` ya recarga `x` desde los parquet
   para cada grupo (lo necesita para el residual de NLL), un encoder
   fresco `model(x)` ahí mismo resuelve esto sin depender de los `.npy`
   guardados — pero significa que esta variante ya no usa el artefacto
   de la Fase 2, corre el encoder de nuevo.

## 3. Estimado de esfuerzo

- **Código**: chico. Un script nuevo (o una variante con flag
  `--use-z`/`K muestras`) de ~30-40 líneas, reusando casi todo lo que ya
  existe en `detectability_oe2.py`.
- **Decisión de diseño que sí hay que tomar antes de programar**: ¿1
  muestra de z (más simple, pero con varianza extra sin controlar) o
  promedio sobre K muestras (más robusto, más caro, y hay que fijar K y
  la semilla de torch)? Esto no es solo una preferencia de estilo — con
  K=1 el bootstrap de CI de AUC-ROC (Fase 4) mezclaría dos fuentes de
  varianza (la del muestreo de z y la del bootstrap sobre grupos), y
  habría que decidir si eso es aceptable o si conviene fijar z una vez
  (con semilla) y tratarlo como determinista de ahí en adelante para esa
  corrida.
- **No hay que rehacer nada de la Fase 2 ni de las Fases 3/5** — Fase 3
  (geometría) y Fase 5 (correlación) ya usan `mu` para las métricas
  geométricas por una razón distinta (Mahalanobis/silhouette necesitan un
  punto fijo por muestra, no una variable aleatoria), esa parte no
  debería cambiar aunque se decida usar z para detectabilidad.

**Conclusión corta**: técnicamente fácil (el trick ya existe y ya está
enchufado en `forward()`), pero no es un cambio de una línea sin
consecuencias — activa una decisión metodológica (1 muestra vs. K
promediadas, y cómo fijar la semilla) que cambia qué significa el número
resultante. Vale la pena decidir esa parte antes de programarlo.
