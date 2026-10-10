# Pendientes del OE1 y del OE2 en la VM: guía de ejecución

Una sola corrida, `scripts/run_pendientes_vm.sh`, hace todo lo que falta de los dos objetivos y lo deja
documentado en `outputs/vm_pendientes/`. Después de hacer commit y push desde la VM, se revisa desde la
otra máquina con `git pull`.

## Decisión de fondo

El OE2 se recalcula entero sobre los modelos del OE1 saneado: 66 variables, semillas 42 a 49. Los modelos
del OE2 son los mismos del OE1 (no se entrena ninguno aparte), así que todo queda alineado. Los resultados
de 51 variables (`outputs/oe2_robust/` y `outputs/oe2_descriptores/`) no se tocan y sirven de comparación.

## Qué hace la corrida

| Etapa | Qué hace | Salida | Tiempo aprox. |
|---|---|---|---|
| E00 | venv `.venv-linux`, torch CPU, `requirements.txt` | `logs/E00_entorno.log` | 5–10 min |
| E01 | Tests del OE1 saneado y de la estadística del OE2. Si fallan, la corrida se detiene | | < 1 min |
| E10 | OE1: datos desde los CSV crudos, preproceso, 8 semillas + referencia de 74, diagnósticos | `outputs/oe1_saneado/` | 2,5–3,5 h |
| E11 | Verifica que existan los 9 checkpoints y la tabla de diagnósticos | | segundos |
| E12–E13 | OE1: figuras del informe y tablas LaTeX y cifras (`outputs/oe1_saneado/informe_tablas/`) | | minutos |
| E20 | OE2 robusto: codificación μ y log σ² por semilla, unidades activas | `outputs/oe2_robust_saneado/` | minutos |
| E21 | OE2 robusto: geometría distribucional (fase 3) | | ≈ 1 h |
| E22_sNN, E22 | OE2 robusto: detectabilidad, semilla por semilla, y luego la tabla conjunta | | 1–3 h |
| E23–E25 | OE2 robusto: correlación (fase 5), figuras, UMAP | | minutos |
| E30–E35 | Descriptores (fases 1–6): AUC, 17 descriptores, cribado, diagnósticos, figuras, tablas | `outputs/oe2_descriptores_saneado/` | ≈ 1,5 h |
| E36 | **Prueba confirmatoria (fase 7)** | `RESULTADOS_CONFIRMATORIA.md` | segundos |
| E99 | Resumen de toda la corrida | `outputs/vm_pendientes/RESUMEN_CORRIDA.md` | segundos |

En total, entre 6 y 10 horas, según los núcleos de la VM.

Si una etapa falla, solo se saltan las que dependen de ella; el resto sigue. Si fallan los tests (E01),
no se corre nada.

### La prueba confirmatoria (fase 7)

Las hipótesis quedaron registradas en el código (`fase7_confirmatoria.py`) el 10-10-2026, antes de ver los
resultados de los modelos saneados:

- **Primario:** mediana de Mahalanobis frente al AUC (nll_mc), con signo positivo. Una sola prueba con
  α = 0,05.
- **Secundarios (Holm entre 6):** Fisher–Rao (+), kNN decodificado (+), tasa KL (+), kNN blanqueado (+),
  log-magnificación (+) y silhouette con Fisher–Rao (−).
- Se reportan IC por bootstrap jerárquico (familias y semillas), la descomposición a nivel de flujo
  (total, dentro de cada familia) y la comparación con el modelo de 51 variables.

Alcance: son modelos nuevos (otra entrada y otro entrenamiento), pero las familias y los flujos son los
mismos con los que se eligieron los descriptores. Es una confirmación sobre modelos reentrenados de forma
independiente, no sobre datos nuevos. Así hay que declararlo en la monografía.

## Antes, en esta máquina (Windows)

1. Seguir `docs/COMMITS_SUGERIDOS_2026-10-10.md`: mover las corridas de control de Windows, hacer los
   commits y el push.

## En la VM

```bash
cd ~/vae-latent-geometry-nids          # o donde esté el repo
git pull
ls data/raw/                           # deben estar los 5 CSV: Monday ... Friday-WorkingHours.csv
tmux new -s pendientes
bash scripts/run_pendientes_vm.sh
# Ctrl+B D para salir sin detenerlo; tmux attach -t pendientes para volver
```

Progreso:

```bash
tail -f outputs/vm_pendientes/RUN_LOG.md
cat outputs/vm_pendientes/estado_etapas.jsonl
tail -f outputs/oe1_saneado/registro_run.log      # durante el entrenamiento del OE1
```

**Si los CSV crudos no están en la VM**, cópialos desde esta máquina (1,1 GB). En PowerShell, desde la raíz
del repo (cambia usuario, VM y ruta):

```powershell
scp data/raw/*.csv usuario@vm:~/vae-latent-geometry-nids/data/raw/
```

**Si se corta** (SSH caído, VM reiniciada), relanza el mismo comando. El OE1 salta las semillas que ya
tienen checkpoint y las fases del OE2 reutilizan su caché por semilla.

**Si quieres reentrenar desde cero** (por ejemplo, después de cambiar la configuración), borra en la VM
`outputs/oe1_saneado/checkpoints/`, `outputs/oe2_robust_saneado/` y `outputs/oe2_descriptores_saneado/`
antes de relanzar.

### Si una etapa falla

- Lee la sección de esa etapa en `RUN_LOG.md`, que tiene la cola del log, o el log completo en
  `outputs/vm_pendientes/logs/<etapa>.log`.
- `exit 137`: la VM se quedó sin memoria. `exit 132`: instrucción de CPU no soportada; el script ya limita
  MKL a SSE4.2, así que avísame.
- Para relanzar una sola fase a mano:
  ```bash
  source .venv-linux/bin/activate
  export VAE_NIDS_OE2_PERFIL=saneado
  python -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed 45
  python -m vae_nids.evaluation.oe2_descriptores.fase7_confirmatoria
  python scripts/vm_resumen_corrida.py            # regenera el resumen
  ```
- Aunque algo falle, haz igual el commit y el push de lo que haya: el resumen y los logs bastan para
  diagnosticarlo desde aquí.

## Al terminar, en la VM

```bash
bash scripts/vm_preparar_commit.sh     # git add solo de lo que se versiona; avisa si algo pesa > 20 MB
git commit -F outputs/vm_pendientes/MENSAJE_COMMIT.txt
git push
```

Se versionan tablas, métricas, figuras, logs, los checkpoints del OE1 (≈ 40 KB cada uno) y toda la
carpeta `outputs/vm_pendientes/`. No se versionan los vectores latentes, los scores por conexión ni las
cachés (cientos de MB, regenerables; ya están en `.gitignore`).

## Después, en esta máquina

`git pull` y revisar, en este orden:

1. `outputs/vm_pendientes/RESUMEN_CORRIDA.md`: estado de cada etapa y todas las cifras clave.
2. `outputs/vm_pendientes/RUN_LOG.md`: log cronológico, con la cola de cada etapa.
3. `outputs/oe2_descriptores_saneado/RESULTADOS_CONFIRMATORIA.md`.

Con eso se cierran el informe del OE1 saneado (sección 7), se actualiza el informe de geometría latente
y se redactan el OE1 y el OE2 en la monografía.
