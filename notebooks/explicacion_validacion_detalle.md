# Validación en detalle: qué parámetros fija, qué representan, y por qué no 70/30

## 1. Los parámetros concretos que se fijan con validación (no son abstractos)

En este proyecto hay exactamente **3 números** cuyo valor se decide
mirando `val_benign.parquet` — nada más, y nada menos:

| Parámetro | Valor actual | Dónde se fija | Qué representa |
|---|---|---|---|
| **Época/checkpoint** | 52 (para seed=42, input51) | `training/train.py`, monitoreando `val_total` cada época | El punto en el que el modelo terminó de aprender la estructura *general* del tráfico benigno, antes de empezar a memorizar detalles específicos de las 1,159,948 filas de train que no generalizan |
| **min_delta** | 0.10 | Calibrado empíricamente en una corrida preliminar, midiendo el ruido (MAD) de `val_total` en ventanas planas del log | El umbral de "mejora real" vs. "ruido aleatorio" del propio proceso de medición — sin esto, el early stopping se dispararía por fluctuaciones sin sentido, o nunca se dispararía |
| **τ (tau)** | percentil 95 de los scores de reconstrucción (en discusión: 95 vs. 99) | `evaluation/detectability_oe2.py`, sobre `val_benign` completo | El punto de corte que separa "esto se parece a benigno" de "esto es raro" — calibrado mirando *solo* cómo se comporta el modelo frente a tráfico benigno que nunca vio directamente |

Estos tres son los **únicos** lugares del pipeline donde `val_benign` entra
en juego. En ningún otro cálculo (Mahalanobis, silhouette, AUC-ROC, TPR
finales) se usa validación — esos usan `test_benign` + `test_attacks`.

## 2. Qué tienen en común los tres

Los tres son casos del mismo patrón: **"necesito probar algo contra
datos que el modelo no memorizó directamente, para decidir un parámetro
del proceso — sin gastar todavía los datos con los que voy a reportar el
resultado final."**

- Época/checkpoint: "¿en qué punto el modelo generaliza mejor?" — hay que
  *probarlo* en algo no memorizado (si lo pruebas en train, el número
  baja indefinidamente porque literalmente estás optimizando eso mismo).
- min_delta: "¿cuánto ruido normal tiene mi métrica de validación?" — hay
  que *medirlo* en la práctica, no adivinarlo.
- τ: "¿qué tan raro es 'raro' para tráfico benigno normal?" — hay que
  *calibrarlo* contra benigno real, no contra un número arbitrario.

## 3. El experimento que responde "por qué no simplemente 70/30"

Corrí esto ahora mismo con el modelo y los datos reales del proyecto:

```
tau (calibrado en val_benign, percentil 95) = -156.9726

FPR sobre val_benign (mismo dato usado para calibrar tau): 5.0000%
   <- por construcción, casi exactamente 5%

FPR sobre test_benign (dato independiente, nunca tocado antes): 5.0776%
   <- esto es lo que realmente informa si tau generaliza
```

**Esto es la prueba concreta del problema.** τ se definió *como* "el
percentil 95 de los scores en val_benign" — así que, matemáticamente, el
5.00% de val_benign *tiene* que superar τ. Eso no es un resultado, es una
tautología: pusiste la vara ahí a propósito para que quedara en 5%. No te
dice nada sobre si el modelo funciona bien.

Lo que sí te dice algo real es el 5.0776% sobre `test_benign` — un
conjunto de datos que τ nunca vio. Si ese número hubiera salido, por
ejemplo, 18% o 2%, sabrías que el umbral calibrado en validación **no
generaliza** bien a tráfico benigno nuevo (señal de que el modelo no es
tan estable como parece, o de que hay algo distinto entre esos días de
tráfico). Que salga 5.08% —muy cerca de 5.00%— es la evidencia real de
que τ generaliza razonablemente. Pero esa evidencia **solo existe porque
τ y la medición final vienen de conjuntos distintos**.

## 4. Qué pasaría con un 70/30 simple (el escenario contrafactual)

Si hicieras 70% train / 30% "test" (sin validación separada), tendrías
que elegir una de estas dos rutas, y ambas tienen el mismo problema:

**Ruta A — usar el 30% para todo (early stopping, calibrar τ, Y reportar
el resultado final):**
El número que reportarías como "FPR al 95%" sería el 5.0000% tautológico
de arriba, no el 5.0776% real. Estarías reportando una propiedad de tu
propia definición de τ, disfrazada de resultado experimental. Cualquiera
que audite la metodología (tu director, un evaluador de la monografía)
puede señalar esto como fuga de datos: calibraste y evaluaste con el
mismo conjunto.

**Ruta B — usar train para early stopping y calibrar τ (evitar tocar el
30% hasta el final):**
No funciona porque train es exactamente lo que el modelo está
memorizando en ese momento. `train_total` (la pérdida sobre train) baja
de forma casi monótona mientras el modelo sigue entrenando —seguirá
bajando incluso cuando el modelo ya empezó a sobre-ajustarse a
peculiaridades específicas de esas 1,159,948 filas—, así que nunca vas a
ver la señal de "ya deja de generalizar, para aquí". Es estructuralmente
incapaz de detectar sobre-ajuste, porque es lo mismo que se está
optimizando directamente.

**Por eso la solución no es "usar train" ni "usar el mismo test para
todo" — es un tercer conjunto**, independiente tanto del entrenamiento
directo (train) como de la medición final (test), donde puedas *probar*
tus decisiones de proceso antes de gastar el conjunto con el que vas a
reportar.

## 5. ¿Por qué no simplemente aceptar el sesgo, ya que hay tantos datos?

Con 1.65M filas de benigno, alguien podría argumentar "el sesgo de usar
el mismo split dos veces sería mínimo en la práctica, ¿para qué
complicarse?". Es cierto que con muestras grandes el sesgo *numérico*
suele ser chico (por la ley de los grandes números) — pero acá no hay
ninguna razón para aceptar ese sesgo, chico o no: hay benigno de sobra
(1.65M filas) para dar un 15% completo a validación sin sacrificar nada
de train ni de test. La única razón real para aceptar esa fuga sería
escasez de datos, que no es el caso. Usar 3 splits acá no cuesta nada y
evita tener que poner una nota metodológica explicando por qué se aceptó
un sesgo conocido.

## 6. Resumen en una frase

**Train** es lo que el modelo memoriza. **Test** es lo único con lo que
se reporta el resultado final, y no debe influir ninguna decisión previa.
**Validación** es el único lugar donde el proyecto se permite "probar y
ajustar" (cuándo parar de entrenar, dónde poner el umbral) sin gastar el
conjunto que va a dar la cifra que termina en la monografía.
