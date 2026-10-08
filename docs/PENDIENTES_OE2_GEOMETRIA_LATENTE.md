# OE2 — Descriptores de la geometría del espacio latente: hallazgos y pendientes

Documento de traspaso entre máquinas. Resume la búsqueda bibliográfica de descriptores (fase previa del
OE2), lo que se encontró al inspeccionar el modelo real y lo que falta implementar y escribir.

- **Corte:** 6 de octubre de 2026.
- **Estado:** investigación terminada; informe en LaTeX sin escribir; nada implementado aún en el repo.
- **Qué NO viaja con este commit:** los PDFs descargados, los scripts de prueba y las figuras. Están en la
  carpeta de la monografía de la máquina `johan` (ver sección 9).

---

## 1. Por qué se hizo esto

El OE2 anterior describió la geometría con dos medidas sobre μ (mediana de Mahalanobis al centroide benigno
y silhouette euclidiano) y obtuvo ρ = 0,38 con el AUC por familia (n = 11, no significativo).

Crítica del director: el latente de un VAE es probabilístico. Cada flujo es una distribución
q(z|x) = N(μ(x), diag σ²(x)) moldeada por el prior N(0, I). Una mediana de distancias desde μ ignora σ,
ignora el prior y supone un benigno elíptico con un solo centro.

Lo que se buscó: formas de describir y comparar grupos de puntos y de distribuciones en el latente, con su
matemática, su comportamiento con n = 32 y su costo, para recomendar un conjunto concreto.

## 2. Hallazgos sobre el modelo real

Salen de leer el repo en solo lectura: checkpoints de las 10 semillas, tablas de `outputs/oe2_robust/tables/`
y los μ de la semilla 42 en `outputs/latent_vectors/`. No se cruzó ningún descriptor nuevo con el AUC.

### 2.1 El "anillo" de la UMAP es PortScan, no el benigno
- Los 159 023 flujos de PortScan son solo **905 vectores μ distintos** (0,6 %); n efectivo ≈ 145. Un mismo
  vector aparece 6648 veces. El 99 % de sus flujos tiene el 15.º vecino propio a distancia 0.
- `umap_projection_oe2.py` usa `n_neighbors=15`. Con duplicados exactos el grafo de vecinos degenera y UMAP
  reparte los puntos en un disco uniforme.
- En la figura del repo el anillo tiene el color de PortScan (`tab20`, índice 1), no el de BENIGN (índice 0,
  3000 puntos con alfa 0,3, casi invisibles).
- La homología persistente del benigno no muestra un lazo dominante.
- **Pendiente:** repetir la UMAP sin duplicados y con grupos balanceados para confirmarlo.

Otros grupos con muchos duplicados: DoS_slowloris (51 % únicos, n efectivo ≈ 731), DoS_slowloris_attempted
(6 %), DoS_Slowhttptest_attempted (23 %), Bot_attempted (32 %). El benigno tiene 75,5 % de únicos.

### 2.2 El benigno no es una nube elíptica (semilla 42)
- El 20 % de los flujos benignos tiene Mahalanobis < 1; una gaussiana en 6 dimensiones daría 1,4 %.
- Curtosis en exceso por dimensión activa de hasta 17,9.
- La dimensión latente 4 es **bimodal**: picos estrechos en ≈ −0,8 y ≈ +1,2, y solo el 2 % de los flujos entre
  −0,75 y +0,75. El centroide benigno cae en ese hueco.
- El BIC de una mezcla gaussiana ajustada al benigno sigue bajando hasta K = 8 (máximo probado).

### 2.3 El decodificador ve casi una sola dirección latente
- Se calculó la métrica pull-back de Fisher–Rao G(z) = J_μᵀ diag(1/σ_x²) J_μ + ½ J_ℓᵀ J_ℓ en 20 000 puntos
  z ~ N(0, I), con el jacobiano exacto (una capa oculta ReLU).
- En las 10 semillas el **rango efectivo de G es 1,05 a 1,2** (1,6 a 1,9 en la semilla 5). En la semilla 42
  los autovalores medianos van de ≈ 2770 a ≈ 0,006.
- El 88 a 91 % de las log-varianzas del decodificador está en el piso −10.
- Consecuencia: Mahalanobis blanquea y da el mismo peso a las 6 dimensiones activas, pero el puntaje de
  detección (NLL) solo "ve" 1 o 2. Ren et al. (2021) describen este mismo mecanismo de dilución.

### 2.4 σ no es despreciable
- Suma media de σ² del codificador en benigno: ≈ 2,40 sobre 6 dimensiones activas (≈ 0,40 por dimensión),
  comparable a la varianza de μ (0,25 a 0,97).
- La relación de Kumar y Poole (2020), σ_j² ≈ 1 / (1 + G_jj), evaluada con el decodificador real predice una
  suma de 2,46. La teoría y el modelo coinciden.
- KL por dimensión estimada (semilla 42): ≈ [0,40; 0,13; 0; 1,71; 3,93; 0,13; 0,84; 0] nats (suma 7,15;
  la KL de validación es 8,18). "Activa o colapsada" es un continuo, no una dicotomía.
- La mayoría de las familias de ataque tiene posteriores más concentrados que el benigno (suma de σ² entre
  0,67 y 1,8). Infiltration es la excepción: σ² en el techo del clamp (media 2682).

### 2.5 Ejemplo real del fallo del centroide: Bot
| | Benigno (control) | Bot |
|---|---|---|
| Mediana de Mahalanobis | 1,81 | 1,79 |
| Flujos más lejos de su 10.º vecino benigno que el p95 benigno | 5 % | 88 % |
| Flujos "más cerca del centroide que el benigno mediano y en zona sin benigno" | 0,1 % | 82 % |
| Lo mismo que la segunda fila, pero en el espacio decodificado | 5 % | 5 % |

Bot está separado en z pero el decodificador lo reconstruye como benigno. Hay dos mecanismos de detección
(Denouden et al., 2018): fuera de la variedad (error de reconstrucción alto) y sobre la variedad pero en
zona no poblada (solo se ve en el latente).

Otros casos: SSH_Patator tiene mediana de Mahalanobis 1,38, menor que el benigno, y está realmente dentro
del soporte benigno. Todas las familias caen en zonas de mayor magnificación que el benigno (7,4 a 11,3
frente a 5,9).

### 2.6 Redundancia entre descriptores (Spearman, 11 familias, semilla 42)
- Mediana de Mahalanobis frente a tipicidad radial: 0,97. Frente a log-densidad de mezcla: 0,87.
- Frente a kNN al benigno en z: 0,51. Frente a kNN en espacio decodificado: 0,74.
- Frente a LID relativa y a log-magnificación: ≤ 0,30.

## 3. Lo que el repo ya calcula

`phase3_geometry.py` ya obtiene, sobre μ y dimensiones activas: distribución de Mahalanobis con pruebas y
delta de Cliff, covarianza robusta MCD, MMD² RBF, Wasserstein rebanado, energía, Bhattacharyya entre nubes
gaussianas, Mahalanobis esperada bajo el posterior, suma de σ², densidad kNN, pureza de vecindad, traza y
log-det, silhouette micro. `phase5` ya tiene Spearman a nivel de flujo y un modelo mixto.

Lo nuevo es lo que usa σ por flujo, el decodificador o la referencia benigna por vecinos.

## 4. Catálogo de descriptores

Notación: flujo x con posterior N(μ, diag σ²); benigno con media m_b y covarianza Σ_b; J dimensiones.
Nivel: **P** primario, **S** secundario, **E** exploratorio, **X** no recomendado.

| Id | Descriptor | Nivel | Usa | ¿En el repo? |
|---|---|---|---|---|
| 1.1 | Mahalanobis al centroide benigno | P | μ | sí |
| 1.2 | Mahalanobis multi-centro / log-densidad de mezcla benigna | E | μ | no |
| 1.3 | Norma respecto del prior y Mahalanobis relativa | E | μ | no |
| 1.4 | Tipicidad radial bilateral | E | μ | no |
| 2.1 | KL y Jeffreys entre dos posteriores | E | μ, σ | no |
| 2.2 | Wasserstein-2 entre dos posteriores | E | μ, σ | no |
| 2.3 | Bhattacharyya y Hellinger entre dos posteriores | E | μ, σ | parcial |
| 2.4 | Fisher–Rao al k-ésimo posterior benigno | S | μ, σ | no |
| 2.5 | Mutual Likelihood Score | E | μ, σ | no |
| 3.1 | MMD² sobre μ, insesgado | E | μ | sí |
| 3.2 | MMD² analítico entre posteriores agregados | S | μ, σ | no |
| 3.3 | Distancia de energía | E | μ | sí |
| 3.4 | Wasserstein rebanado sobre μ y sobre (μ, σ) | E | μ, σ | solo μ |
| 4.1 | Espectro de la covarianza y rango efectivo | E | μ | parcial |
| 4.2 | Distancia afín-invariante entre covarianzas | S | μ | no |
| 4.3 | Ángulos principales entre subespacios | E | μ | no |
| 5.1 | Distancia al k-ésimo vecino benigno | S | μ | parcial |
| 5.2 | LOF y razón de densidades kNN | E | μ | parcial |
| 5.3 | Desajuste entre posterior agregado y prior | E | μ, σ | no |
| 6.1 | Métrica pull-back de Fisher–Rao y log-magnificación | S | decodificador | no |
| 6.2 | Distancia "vista por el decodificador" al benigno | S | decodificador | no |
| 6.3 | Geodésicas y curvatura | X | decodificador | no |
| 7.1 | Tasa: KL al prior por flujo y por dimensión | S | μ, σ | parcial (OE1) |
| 7.2 | Log-volumen del posterior, Σ log σ² | E | σ | parcial |
| 7.3 | Razón ancho / separación | E | μ, σ | no |
| 8.1 | LID relativa al benigno | S | μ | no |
| 8.2 | Dimensión intrínseca del grupo (TwoNN, MLE) | E | μ | no |
| 9.1 | Homología persistente (H0, H1) | E | μ | no |
| 10.1 | Silhouette del grupo de ataque (blanqueado y Fisher–Rao) | S | μ, σ | sí (micro, sin blanquear) |
| 10.2 | Davies–Bouldin, Calinski–Harabasz, razón de Fisher | E | μ | no |
| 10.3 | Sonda lineal: AUC de regresión logística sobre (μ, log σ²) | S | μ, σ | no |
| 11.1 | Reconstrucción estocástica y desajuste codificador–decodificador | E | todo | no |
| 11.2 | Atribución del error por variable (gradiente) | E | modelo | no |

### 4.1 Fórmulas de los recomendados

**1.1 Mahalanobis (primario).** $D_M(x)=\sqrt{(\mu-m_b)^\top\Sigma_b^{-1}(\mu-m_b)}$, mediana por familia.

**2.4 Fisher–Rao entre gaussianas diagonales** (Miyamoto et al. 2023, Ec. 5.3):

$$d_{FR}(q_1,q_2)=2\sqrt2\Big[\sum_j \operatorname{artanh}^2\sqrt{\tfrac{(\mu_{1j}-\mu_{2j})^2+2(\sigma_{1j}-\sigma_{2j})^2}{(\mu_{1j}-\mu_{2j})^2+2(\sigma_{1j}+\sigma_{2j})^2}}\Big]^{1/2}$$

Se toma la distancia al k-ésimo posterior benigno más cercano (k = 10). Los candidatos se preseleccionan
con W2, que es euclidiana en (μ, σ): $W_2^2=\lVert\mu_1-\mu_2\rVert^2+\lVert\sigma_1-\sigma_2\rVert^2$.

**3.2 MMD² analítico entre posteriores agregados.** Con núcleo RBF de ancho h, para dos posteriores:

$$\mathbb E\,k(z,z')=\prod_j\sqrt{\tfrac{h^2}{h^2+\sigma_{1j}^2+\sigma_{2j}^2}}\exp\Big(-\tfrac12\sum_j\tfrac{(\mu_{1j}-\mu_{2j})^2}{h^2+\sigma_{1j}^2+\sigma_{2j}^2}\Big)$$

y MMD² = media dentro de A + media dentro de B − 2 · media cruzada, excluyendo i = j (estimador insesgado).
No requiere muestrear z.

**4.2 Distancia afín-invariante entre covarianzas** (Pennec et al. 2006):
$d_{AI}=\big[\sum_i\log^2\lambda_i(\Sigma_b^{-1}\Sigma_f)\big]^{1/2}$, con contracción de Ledoit–Wolf.

**5.1 kNN al benigno.** $r_k(x)=\lVert\tilde\mu(x)-\tilde\mu^{\,b}_{(k)}\rVert$ con $\tilde\mu=\Sigma_b^{-1/2}(\mu-m_b)$.
Resumen por familia: fracción de flujos por encima del percentil 95 del propio benigno (benigno de control
no usado en la referencia).

**6.1 Métrica y log-magnificación** (Arvanitidis et al. 2022, Prop. 3.2), con ℓ = log σ_x²:

$$G(z)=J_\mu^\top\mathrm{diag}(e^{-\ell})J_\mu+\tfrac12J_\ell^\top J_\ell,\qquad \mathrm{MF}(z)=\tfrac12\log\det(I+G(z))$$

Con una capa oculta ReLU el jacobiano es exacto: $J_\mu=W_m\,\mathrm{diag}(\mathbb 1[h>0])\,W_1$. $J_\ell=0$ donde
actúa el clamp.

**6.2 Distancia vista por el decodificador.** $\phi(x)=\mu_x(\mu(x))/\sigma_x(\mu(x))$ y kNN al benigno en ese
espacio. No depende de cómo se parametrice z, así que es comparable entre semillas sin alinear nada.

**7.1 Tasa.** $R_j(x)=\tfrac12(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2)$, $R(x)=\sum_jR_j(x)$. Usar mediana por familia
(Infiltration tiene atípicos extremos). Reportar D y R por separado: −ELBO = D + R.

**8.1 LID relativa** (Ma et al. 2018, Ec. 4): $\widehat{\mathrm{LID}}(x)=-\big[\tfrac1k\sum_{i=1}^k\log\tfrac{r_i(x)}{r_k(x)}\big]^{-1}$
con $r_i$ las distancias a los vecinos **benignos** (k = 20).

**10.1 Silhouette del grupo.** $s(i)=(b(i)-a(i))/\max\{a(i),b(i)\}$, promediado **solo sobre la familia**, con
muestreo por grupo. Dos versiones: euclidiana sobre μ blanqueado y con Fisher–Rao (matriz precomputada).

**10.3 Sonda lineal.** AUC en validación cruzada estratificada de una regresión logística con pesos
balanceados sobre (μ, log σ²) de las dimensiones activas.

## 5. Recomendación para el OE2

- **Primario:** mediana de Mahalanobis frente a AUC (decisión del autor). Límites a declarar: supuesto
  elíptico falso, unilateral, mismo peso a direcciones que el decodificador ignora.
- **Secundarios (10):** 5.1, 2.4, 7.1, 3.2, 6.2, 6.1, 4.2, 8.1, 10.1, 10.3. Cubren posición por vecinos,
  posteriores con σ, incertidumbre, solapamiento agregado, métrica del decodificador, forma, dimensión local
  y separabilidad.
- **Por qué no otros:** tipicidad radial es redundante con Mahalanobis (ρ = 0,97); energía es un MMD
  (Sejdinovic et al. 2013, Teo. 22); Wasserstein rebanado y rango efectivo están sesgados con n = 32; TwoNN
  es inservible con n = 32; las geodésicas exigen una métrica suave que aquí no existe.

### 5.1 Comparaciones múltiples
- Con 11 familias el |ρ| mínimo detectable es 0,60 a α = 0,05, y 0,78 con Holm sobre 10 secundarios
  (potencia 0,36 para ρ = 0,7).
- Propuesta: una sola prueba confirmatoria (el primario). Secundarios con tamaño de efecto e intervalos
  (bootstrap sobre familias y semillas) y Holm entre ellos. Exploratorios sin p-valores o con FDR.
- `stats_utils.holm_correction` ya existe en el repo.

### 5.2 Análisis a nivel de flujo
- Sirven los descriptores por flujo: 5.1, 2.4, 7.1, 6.1, 6.2, 8.1.
- Es una pregunta distinta: "dentro de una familia, ¿los flujos más alejados se detectan mejor?". El repo ya
  ve signos opuestos (−0,53 en PortScan, +0,88 en slowloris). Separar el efecto entre familias (media de la
  familia) del efecto dentro de la familia (desviación respecto de esa media).
- El n efectivo es el de flujos distintos, no el nominal: PortScan vale ≈ 145, no 159 023.

### 5.3 Tres puntos transversales
- **Prior frente a benigno.** Medir contra el prior (norma, KL) no exige estimar nada, pero solo equivale a
  medir contra el benigno si el posterior agregado benigno se parece a N(0, I). Aquí no se parece.
- **Dimensiones colapsadas.** Las manejan solas KL, Fisher–Rao, W2, Hellinger, log-volumen y todo lo basado
  en el decodificador (aportan ≈ 0). Exigen excluirlas a mano Mahalanobis, kNN blanqueado, covarianzas,
  silhouette euclidiano y cualquier cosa sobre z muestreado. Conviene reportar la KL media por dimensión en
  lugar del criterio binario de Burda.
- **Semillas.** Con posterior diagonal la ambigüedad es permutación y signo de ejes, no una rotación
  cualquiera (Kumar y Poole 2020, Sec. 3.1; Rolinek et al. 2019). Son comparables sin alinear lo
  afín-invariante (Mahalanobis, kNN blanqueado, 4.2, sonda lineal) y lo definido en el espacio de
  reconstrucción (6.2). El silhouette euclidiano sin blanquear no lo es.

## 6. Resultados de las pruebas sintéticas (8-D, 6 activas, 2 colapsadas)

- **Solo cambia σ:** los descriptores sobre μ no responden (AUC por flujo 0,50); KL y Fisher–Rao sí (0,99).
- **Ataque en el hueco de un benigno bimodal:** Mahalanobis AUC 0,40 y KL 0,29 (peor que el azar); kNN en z
  0,96; kNN decodificado y Fisher–Rao 1,00.
- **Misma Mahalanobis, distinta dirección:** desplazar 3 desviaciones en una dirección que el decodificador
  ve o en una que ignora no cambia Mahalanobis ni kNN en z; kNN decodificado y log-magnificación sí distinguen.
- **Sesgo con n = 32** (en desviaciones estándar del estimador): Wasserstein rebanado +8,1; rango efectivo
  −3,6; persistencia máxima de H1 −2,8; ángulo principal +1,5; distancia afín-invariante +1,3. Sin sesgo
  apreciable: MMD insesgado, energía, kNN, Fisher–Rao, KL, silhouette por grupo.
- **Silhouette micro frente a por grupo:** con un grupo pequeño el promedio global lo domina el benigno y
  puede cambiar de signo respecto del silhouette de la familia.
- **Tiempos:** ningún descriptor pasó de ≈ 20 s por familia con n = 250 000 en un portátil de 4 núcleos;
  los O(n²) usan submuestra de 2000 (500 para silhouette).

## 7. Pendientes de implementación en el repo

Nada de esto está hecho. Orden sugerido:

1. **Quitar o ponderar duplicados** antes de cualquier descriptor basado en vecinos, y reportar por familia
   n, n de únicos y n efectivo. Afecta a silhouette, kNN, LID, TwoNN, homología y UMAP.
2. **Repetir la UMAP** sin duplicados y con grupos balanceados (o con jitter documentado).
3. **Módulo de descriptores por flujo** que lea `mu` y `logvar` por semilla (`phase2_encode.py` ya los guarda):
   5.1, 2.4, 7.1 (total y por dimensión), 8.1.
4. **Módulo del decodificador:** jacobiano exacto, G(z), log-magnificación (6.1) y embebido decodificado con
   kNN (6.2). Requiere cargar el checkpoint de cada semilla.
5. **Descriptores por familia nuevos:** 3.2, 4.2, 10.1 (por grupo, blanqueado y Fisher–Rao), 10.3.
6. **Referencia benigna:** separar el benigno en referencia y control para fijar umbrales (percentil 95) sin
   fuga.
7. **Comprobación de Kumar–Poole por semilla** con datos reales: comparar σ² observado con 1 / (1 + G_jj) en
   benigno y por familia (descriptor 11.1).
8. **Fase 5:** añadir los secundarios con Holm entre ellos, intervalos por bootstrap y la descomposición
   entre familias y dentro de familia a nivel de flujo.
9. Decidir si el piso −10 de la log-varianza del decodificador se mantiene (Dai y Wipf 2019 predicen que la
   varianza tiende a 0; Seitzer et al. 2022 explican el efecto sobre el ajuste de la media).

Avisos de implementación:
- Fijar el ancho de banda del MMD una sola vez sobre el benigno, no por par.
- Recortar log σ² al rango del clamp antes de KL, Bhattacharyya o MLS; el término de volumen domina si σ → 0.
- Los ángulos principales no están definidos en coordenadas blanqueadas (el benigno queda isótropo).
- Con ReLU la métrica es constante a trozos: sirve punto a punto, no para integrar geodésicas.

## 8. Pendientes del informe

1. Escribir `informe_geometria_latente.tex` (español, `babel` con `spanish,es-tabla,es-noshorthands`,
   `lmodern`, cajas `tcolorbox` de Idea, Decisión, Hipótesis y Problema, "En pocas palabras" por sección).
2. Diagrama TikZ: un flujo como elipse, una familia como mezcla de elipses, el prior.
3. Mapa visual del catálogo (familias × propiedad que capturan).
4. Sección de antecedentes en NIDS. Ningún trabajo cribado relaciona geometría latente por familia con
   detectabilidad. Los más cercanos: Pitsiorlas et al. 2024 (Mahalanobis latente al vecino de entrenamiento,
   r = 0,45 a nivel de muestra, NSL-KDD, sin σ ni familias), Nguyen et al. 2019 (AUC por ataque y huellas de
   gradiente), Denouden et al. 2018 (dos mecanismos de detección).
5. Compilar dos veces, revisar cada página, comprimir en zip y dar el resumen final.

## 9. Archivos fuera del repo

En la máquina `johan`: `C:\Users\johan\OneDrive\Desktop\Monografia\`
(en la otra máquina la carpeta equivalente es `C:\Users\user\Desktop\Monografia\`).

| Carpeta | Contenido |
|---|---|
| `informe_geometria_latente\` | `ESTADO_Y_CONTINUACION.md`, `notas_lectura.md` (una nota por trabajo leído, con ecuación o sección), `cribado.csv` (209 trabajos), `catalogo_descriptores.csv`, `fichas.tex`, `referencias_geometria.bib`, `pendientes_de_descarga.md` |
| `informe_geometria_latente\scripts\` | `geo_descriptores.py` (todos los descriptores), `pruebas_juguete.py`, `inspeccion_decodificador.py`, `piloto_modelo_real.py`, `diagnostico_duplicados.py`, `leer_checkpoint_sin_torch.py`, `catalogo.py`, `maestro_referencias.py` |
| `informe_geometria_latente\figuras\` | Tablas y figuras de las pruebas y del piloto real |
| `05_Bibliografia\geometria_latente\` | 83 PDFs de acceso abierto |

Para continuar en otra máquina hay que copiar al menos `informe_geometria_latente\` (≈ 1 MB sin los PDFs).

Restricción de la máquina `johan`: una directiva de Control de aplicaciones de Windows bloquea las DLL de
PyTorch, POT y ripser. Por eso los checkpoints se leyeron con numpy y el piloto real solo usó μ (no hay
log σ² en local). En la VM Linux se puede usar todo.

## 10. Referencias clave (verificadas)

| Tema | Referencia |
|---|---|
| Covarianza del codificador como métrica | Chadebec y Allassonnière 2022, arXiv:2209.07370 |
| Pull-back euclidiano | Arvanitidis et al. 2018, arXiv:1710.11379 |
| Pull-back de Fisher–Rao | Arvanitidis et al. 2022, arXiv:2106.05367 |
| Curvatura casi nula en VAEs | Shao et al. 2018, arXiv:1711.08014 |
| Covarianza del codificador y jacobiano; identificabilidad | Kumar y Poole 2020, arXiv:2002.00041 |
| Régimen polarizado | Rolinek et al. 2019, arXiv:1812.06775 |
| Varianza del decodificador y dimensiones superfluas | Dai y Wipf 2019, arXiv:1903.05789 |
| Tasa y distorsión | Alemi et al. 2018, arXiv:1711.00464 |
| Descomposición de la KL media | Chen et al. 2018, arXiv:1802.04942 |
| μ frente a z muestreado | Locatello et al. 2019, arXiv:1811.12359 |
| Mahalanobis como puntaje OOD | Lee et al. 2018, arXiv:1807.03888 |
| Mahalanobis relativa; dilución por dimensiones | Ren et al. 2021, arXiv:2106.09022 |
| Reconstrucción más distancia latente | Denouden et al. 2018, arXiv:1812.02765 |
| Conjunto típico | Nalisnick et al. 2019, arXiv:1906.02994 |
| Fisher–Rao cerrado, caso diagonal | Miyamoto et al. 2023, arXiv:2304.14885 |
| W2 entre gaussianas y entre mezclas | Delon y Desolneux 2020, arXiv:1907.05254 |
| Núcleos entre distribuciones | Jebara et al. 2004, JMLR 5:819–844 |
| Embeddings gaussianos | Vilnis y McCallum 2015, arXiv:1412.6623; Shi y Jain 2019, arXiv:1904.09658 |
| MMD | Gretton et al. 2012, JMLR 13:723–773 |
| Energía es un MMD | Sejdinovic et al. 2013, arXiv:1207.6076 |
| Divergencias rebanadas | Nadjahi et al. 2020, arXiv:2003.05783 |
| kNN como puntaje OOD | Sun et al. 2022, arXiv:2204.06507 |
| Dimensión intrínseca | Levina y Bickel 2004 (NeurIPS 17); Facco et al. 2017, arXiv:1803.06992; Ma et al. 2018, arXiv:1801.02613 |
| Homología persistente | Fay et al. 2026, arXiv:2505.20435; Chazal y Michel 2021, arXiv:1710.04019 |
| Silhouette y desbalance | Pavlopoulos et al. 2024, arXiv:2401.05831; Batool y Hennig 2021, arXiv:1910.11339 |
| Silhouette para objetos inciertos | Tavakkol et al. 2021, DOI 10.1007/s10479-018-3043-4; Ko et al. 2023, DOI 10.3390/s23073708 |
| Covarianzas, métrica afín-invariante | Pennec et al. 2006, DOI 10.1007/s11263-005-3222-z |
| Invariancias entre representaciones | Kornblith et al. 2019, arXiv:1905.00414 |
| NIDS | Pitsiorlas et al. 2024, arXiv:2409.13774; Nguyen et al. 2019, arXiv:1903.06661; Engelen et al. 2021, DOI 10.1109/SPW53761.2021.00009 |

No se pudo verificar por API la referencia original de Holm (1979); no se cita.
