# Lectura de la corrida de pendientes en la VM (11 de octubre de 2026)

Fuente: commit `37b3d53`. Las cifras salen de `outputs/vm_pendientes/RESUMEN_CORRIDA.md` y de las tablas que cita.

## 1. La corrida

- Las 27 etapas terminaron bien.
- La primera corrida (10-10) murió por SIGILL (exit 132) en 3 semillas de la fase 4 y en los descriptores.
  En la segunda se limitaron las instrucciones SIMD de Arrow, torch y numba y no hubo ningún SIGILL, ni
  siquiera un reintento.
- Muy probablemente eso explica también la corrida del 23-09: la fase 4 del OE2 robusto (51 variables) solo
  quedó con la semilla 42.
- Entorno de la VM: 8 núcleos, 23 GB, Python 3.10.12, torch 2.13.0+cpu, numpy 2.2.6, pandas 2.3.3. Las
  versiones exactas de `requirements.txt` (Windows, Python 3.14) no existen para Python 3.10, así que se
  instaló sin fijar versiones. Lo usado está en `outputs/vm_pendientes/pip_freeze.txt`.

## 2. OE1 saneado (66 variables, 8 semillas)

| | 51 variables (5 semillas) | 66 variables (8 semillas) |
|---|---|---|
| Unidades activas (Burda, Var(μ) > 0,01) | 5,4 | **6,9** (7 en siete semillas, 6 en una) |
| KL total (nats) | 6,9 | 8,5 |
| Mejor época (media) | 76,8 | 74,6 |
| NLL media por columna | −3,84 | −3,84 |
| Log σ² del decodificador en el piso −10 | 89,9 % | 90,5 % |

- Referencia de 74 variables (semilla 42): 7 unidades activas, KL 9,65.
- Las particiones son idénticas al pipeline original. El filtro D4 quitó 24 filas de train y 7 de val. Los
  conteos por familia coinciden con el OE2 original en todas las semillas.
- Lectura: con la entrada saneada, el espacio latente usa más dimensiones y es más estable entre semillas.
  El piso de la varianza del decodificador no cambia: sigue en torno al 90 %.

## 3. OE2: detectabilidad (nll_mc, L = 20)

- El ranking de las 11 familias por AUC casi no cambia respecto de 51 variables: Spearman **0,964**
  (10 semillas frente a 8). W de Kendall del AUC entre semillas: 0,887.
- Las familias más difíciles siguen siendo Bot (0,61) y PortScan (0,67). Las más fáciles, Infiltration,
  DDoS y DoS Slowhttptest (≈ 0,98).
- El cambio más grande está en DoS GoldenEye attempted: baja de 0,80 a 0,71.
- τ = p95 de val_benign da un FPR de 0,051 en test_benign.

## 4. OE2: prueba confirmatoria (hipótesis registradas el 10-10, antes de ver estos resultados)

| Hipótesis | ρ 51 var. | ρ 66 var. | IC 95 % | p_Holm | ¿Confirmada? |
|---|---|---|---|---|---|
| **Primario:** Mahalanobis (mediana) | 0,604 | **0,772** | [0,39; 0,98] | 0,001 | **Sí** |
| Fisher–Rao al 10.º posterior benigno | 0,724 | 0,775 | [0,39; 0,92] | 0,002 | Sí |
| kNN en el espacio decodificado | 0,691 | 0,680 | [0,13; 0,90] | 0,006 | Sí |
| Tasa KL por flujo | 0,681 | **0,830** | [0,48; 0,98] | 0,002 | Sí |
| kNN al benigno (z blanqueado) | 0,694 | 0,741 | [0,29; 0,94] | 0,004 | Sí |
| Log-magnificación del decodificador | 0,523 | 0,592 | [0,05; 0,91] | 0,031 | Sí |
| Silhouette del grupo (Fisher–Rao), signo − | −0,629 | −0,627 | [−0,90; −0,18] | 0,031 | Sí |

- Se confirman el primario y los 6 secundarios. Con las tablas de 51 variables se habrían confirmado 4 de
  los 6.
- La salvedad de Mahalanobis se atenúa: sin PortScan ni Bot, ρ = 0,60 (con 51 variables era 0,41).
- Nivel de flujo: la relación también se da dentro de cada familia. El ρ dentro de familia va de 0,60
  (kNN decodificado) a 0,75 (tasa KL), y el de Mahalanobis es 0,72.
- Validación en los grupos attempted (no usados en el cribado): |ρ| entre 0,64 y 0,95, y los 7 con el
  signo esperado.

## 5. Salvedades que hay que declarar o revisar

1. **Alcance de la confirmación.** Los modelos son nuevos (otra entrada y otro entrenamiento), pero las
   familias y los flujos son los mismos con los que se eligieron los descriptores. Es una confirmación
   sobre modelos reentrenados de forma independiente, no sobre datos nuevos.
2. **Modelo mixto a nivel de conexión (fase 5 del OE2 robusto).** Su efecto fijo de Mahalanobis sobre la
   NLL es 0,10 [−0,10; 0,31], con p = 0,32 y R² marginal 0,002. Parece contradecir el ρ dentro de familia
   de la fase 7 (0,72). Probablemente miden cosas distintas: el modelo mixto usa valores estandarizados
   (no rangos), incluye el benigno y tiene pendientes aleatorias por familia, que pueden tener signos
   opuestos. **Hay que reconciliarlos antes de redactar.** Los scores por conexión solo están en la VM
   (no se versionan), así que esa revisión se corre allá.
3. **Versiones de las bibliotecas.** La VM usó versiones distintas de las de Windows. Conviene fijar las de
   la VM (`pip_freeze.txt`) en un `requirements-vm.txt` para que la corrida sea reproducible.
4. **El rótulo de la sección 4 de `RESUMEN_CORRIDA.md` estaba mal.** Decía que la comparación usaba 10
   semillas, pero la tabla de la fase 4 de 51 variables tiene solo la semilla 42. Ya está corregido (y el
   generador también). La comparación válida, 10 contra 8 semillas, es la de la sección 6.

## 6. Qué sigue

1. Reconciliar el modelo mixto con la descomposición a nivel de flujo (salvedad 2). Es corto y se corre
   en la VM.
2. OE1: cerrar la sección 7 del informe saneado. Las tablas LaTeX ya están en
   `outputs/oe1_saneado/informe_tablas/`.
3. OE2: actualizar el informe de geometría latente con las cifras de 66 variables. Las tablas LaTeX están
   en `outputs/oe2_descriptores_saneado/tablas_tex/`.
4. Redactar el OE1 y el OE2 en la monografía.
5. Decidir sobre el piso −10 de la log-varianza del decodificador (sigue en ≈ 90 %).
6. Diseñar el OE3 y el OE4 sobre estos mismos modelos.
