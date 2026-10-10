# OE1 saneado: reentrenamiento en la VM Linux

> **Reemplazada el 10-10-2026** por `scripts/run_pendientes_vm.sh` (guía en
> [`docs/EJECUTAR_PENDIENTES_EN_VM.md`](../EJECUTAR_PENDIENTES_EN_VM.md)). Ese script hace este
> reentrenamiento, recalcula todo el OE2 sobre los modelos resultantes y deja la documentación de la corrida
> en `outputs/vm_pendientes/`. Esta página queda como referencia de los pasos manuales.

Todo lo liviano ya está hecho en la máquina local: selección, preprocesamiento, tests, prueba de humo e informe en borrador. Lo único pesado que falta es el reentrenamiento (fase 5) y sus diagnósticos. Esta guía es para correrlo en la VM.

## Estado al 10-10-2026

- En la máquina local se alcanzaron a entrenar las semillas 42 a 48 del conjunto saneado. Sus checkpoints están en `outputs/oe1_saneado/checkpoints/` y sus registros en `outputs/oe1_saneado/logs/`.
- Falta la semilla 49, la referencia de 74 columnas (semilla 42) y los diagnósticos.
- **Recomendación:** reentrenar las 8 semillas completas en la VM, para que todas salgan del mismo entorno (Linux, mismas versiones). Las de Windows sirven de control: con `deterministic=True` y los mismos hilos deberían dar pérdidas muy parecidas, aunque no necesariamente idénticas entre sistemas operativos.

## Pasos

1. **Código.** Haz el commit y el push de los cambios locales (ver `docs/COMMITS_SUGERIDOS_2026-10-08.md`) y en la VM:
   ```bash
   git pull
   ```
2. **Datos.** La VM necesita los 5 CSV crudos en `data/raw/`: Monday a Friday-WorkingHours.csv, 1,1 GB. Si ya están, no hagas nada; si no, cópialos desde esta máquina. `data/processed/` no hace falta: el pipeline lo construye.
3. **Entorno** (venv propio, no toca `.venv`):
   ```bash
   python3 -m venv .venv-linux && source .venv-linux/bin/activate
   pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   pip install -e .
   ```
4. **Tests y humo** (segundos):
   ```bash
   pytest tests/oe1_saneado -q
   python -m vae_nids.oe1.datos --config configs/oe1_saneado.yaml
   python -m vae_nids.oe1.entrenamiento --config configs/oe1_saneado_humo.yaml --seed 42
   ```
   `datos` reconstruye las particiones crudas (≈2 min) y verifica el filtro D4: deben salir 24 filas en train y 7 en val.
5. **Reentrenamiento completo** (reanudable; dentro de `tmux`):
   ```bash
   tmux new -s oe1
   python -m vae_nids.oe1.run --config configs/oe1_saneado.yaml --ref74 configs/oe1_saneado_ref74.yaml
   # Ctrl+B D para salir; tmux attach -t oe1 para volver
   tail -f outputs/oe1_saneado/registro_run.log
   ```
   Si existe el checkpoint final de una semilla, se salta. Para reentrenar todo en la VM, no copies los checkpoints de Windows.
   **Tiempo estimado:** en la máquina local, con 8 hilos y compitiendo por CPU, cada semilla tardó entre 10 y 25 minutos (≈13 s por época, de 59 a 136 épocas). Las 8 semillas más la referencia de 74 suman unas 2,5–3 horas, menos en la VM si tiene más núcleos libres.
6. **Traer los resultados** a esta máquina (son pequeños):
   - `outputs/oe1_saneado/tables/`
   - `outputs/oe1_saneado/figures/`
   - `outputs/oe1_saneado/logs/`
   - `outputs/oe1_saneado/registro_run.log`

   Los checkpoints (`outputs/oe1_saneado/checkpoints/`) no hace falta traerlos.

## Después, en local (liviano)

Cuando los resultados estén aquí, se cierra el informe: tablas finales, sección de reentrenamiento, compilación, revisión visual, muestreo de cifras y zip. Ninguno de esos pasos entrena nada:

```bash
python -m vae_nids.oe1.tablas_informe --config configs/oe1_saneado.yaml --dest "<Monografia>/02_Objetivos_Especificos/OE1_Seleccion_de_variables/4_informe_oe1_saneado/informe_oe1_saneado"
```

La tabla del OE1 actual (51 columnas) ya está calculada: `outputs/oe1_saneado/tables/diagnosticos_oe1_actual_51.csv`.
