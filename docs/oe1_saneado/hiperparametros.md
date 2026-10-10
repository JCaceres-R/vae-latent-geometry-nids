# Hiperparámetros del OE1 saneado

Valores fijados en `configs/oe1_saneado.yaml` (D5). El código de entrenamiento (`vae_nids.oe1.entrenamiento`)
solo lee de ahí: si falta una clave, falla en vez de usar un valor por defecto silencioso.

Tipos de justificación:
- **Estándar con referencia**: valor habitual respaldado por una fuente.
- **Diagnóstico propio**: respaldado por una medida de este proyecto.
- **Decisión de diseño**: elección del autor sin respaldo externo, declarada como tal.

Cada cita se comprobó contra el PDF cuando estaba disponible. Las fuentes que no se pudieron abrir quedan marcadas **por verificar**.

| Hiperparámetro | Valor | Qué controla (en simple) | Tipo | Referencia (verificación) | Evidencia del proyecto |
|---|---|---|---|---|---|
| β | 1 | Cuánto pesa el término KL frente a la reconstrucción en la pérdida. | Estándar | Kingma y Welling (2014), Ec. (3): ELBO = −KL(q‖p) + E[log p(x\|z)], sin factor (verificado en el PDF de arXiv:1312.6114). Higgins et al. (2017) introducen β ≠ 1, que aquí no se usa (**por verificar**: OpenReview no accesible desde esta máquina). | — |
| J (dimensión latente) | 8 | Cuántas coordenadas tiene el espacio latente. | Diagnóstico propio + referencia | Burda et al. (2016), Sec. 5.2 y Ap. C: unidad activa si A_u = Cov_x(E_q[u]) > 10⁻²; la distribución de log A_u es bimodal (verificado en el PDF). | Unidades activas por semilla: ver `outputs/oe1_saneado/tables/diagnosticos_por_semilla.csv` y la sección 7 del informe. Si todas las semillas dejan dimensiones inactivas, J = 8 no es una restricción. |
| Capa oculta | 32 | Ancho de la única capa oculta del codificador y del decodificador. | Decisión de diseño | Sin referencia. Criterio: red pequeña, de una capa, que comprime 66 columnas en 8 dimensiones y entrena en CPU en minutos. | Tiempo por época medido en la fase 5. |
| Clamp de log σ² | [−10, 10] | Límite a la varianza que predicen el codificador y el decodificador. | Estándar con matiz | Mattei y Frellsen (2018), Teo. 1: la verosimilitud de un decodificador gaussiano sin restricción no está acotada (σ → 0). Prop. 2: si los autovalores de la covarianza son ≥ ξ > 0, queda acotada (verificado en arXiv:1802.04826). El clamp impone ξ = e⁻¹⁰. Contexto: Rybkin et al. (2021), Introducción: aprender la varianza «ingenuamente» puede llevarla a cero (verificado en el PDF). | Fracción de log σ² del decodificador en el piso −10 y columnas en el piso: fase 5 (D3 lo mantiene). |
| Optimizador y tasa de aprendizaje | Adam, lr = 10⁻³ | Cómo se actualizan los pesos y con qué paso. | Estándar | Kingma y Ba (2015), Algoritmo 1: «Good default settings … are α = 0.001, β1 = 0.9, β2 = 0.999 and ε = 10⁻⁸» (verificado en arXiv:1412.6980). | — |
| Tamaño de lote | 1024 | Cuántos flujos se usan en cada paso de gradiente. | Decisión de diseño | No existe `notas_tecnicas_vae.pdf` ni en el repositorio ni en `Monografia\`. Criterio: costo por iteración en CPU con ~1,16 millones de flujos de entrenamiento. | Segundos por época: fase 5. |
| Parada temprana: paciencia | 10 épocas | Cuántas épocas sin mejora se toleran antes de parar. | Estándar + decisión | Prechelt (1998), parada temprana con paciencia (**por verificar**: capítulo de Springer no accesible). | Época de parada y motivo por semilla: fase 5. |
| Parada temprana: `min_delta` | 0,10 nats | Mejora mínima de la pérdida de validación para contar como mejora. | Decisión de diseño con diagnóstico propio | `notebooks/session_log.md` (líneas 62 y 78): se midió el ruido época a época de `val_total` en la corrida de 100 épocas que no convergía (MAD ≈ 0,10–0,12 nats en tramos planos) y se eligió 0,10. Unidades: nats de la pérdida de validación (−ELBO medio por flujo). **Advertencia:** en valor absoluto su efecto cambia con el número de columnas (la NLL suma sobre columnas: 66 frente a 51). | Épocas de parada del saneado frente al OE1 actual (tabla antes/después). |
| Máximo de épocas | 250 | Tope de épocas. | Decisión de diseño | `notebooks/session_log.md` (línea 60): 100 épocas no bastaron en la primera corrida; se subió a 250. | Ninguna corrida del saneado debería pararse por el tope: fase 5. |
| Score s(x) | NLL gaussiana heterocedástica evaluada en μ | Puntaje de anomalía: qué tan mal reconstruye el modelo un flujo. | Estándar | An y Cho (2015): probabilidad de reconstrucción con decodificador gaussiano (**por verificar**: el informe técnico de SNU no es accesible). **Observación:** con la varianza del decodificador en el piso, s(x) se comporta como un error cuadrático con pesos casi uniformes (1/σ² ≈ e¹⁰ en casi todas las columnas). La cifra está en la fase 5. | τ = p95 de s(x) en `val_benign`. |

Referencias permitidas en esta tarea: las de la sección 6.3 del prompt de saneamiento. No se añadió ninguna otra.
