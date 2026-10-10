# Commits sugeridos (10 de octubre de 2026)

Reemplaza a `COMMITS_SUGERIDOS_2026-10-08.md`. No se hizo ningún commit: todo está en local, en `main`.
Hay que hacer esto **antes** de lanzar la corrida de la VM (`docs/EJECUTAR_PENDIENTES_EN_VM.md`), porque
la VM necesita el código nuevo.

Comandos para PowerShell, desde la raíz del repo.

## 0. Apartar las corridas de control de Windows

Las semillas 42 a 48 entrenadas aquí no se versionan: la VM reentrena las 8 en Linux y esas son las que se
suben. Si se dejan donde están, el `git pull` posterior choca con los archivos que traiga la VM. La carpeta
de destino ya está en `.gitignore`.

```powershell
New-Item -ItemType Directory -Force outputs/oe1_saneado_control_windows
Move-Item outputs/oe1_saneado/checkpoints, outputs/oe1_saneado/logs, outputs/oe1_saneado/registro_run.log, outputs/oe1_saneado/run_stdout.log, outputs/oe1_saneado/datos.log outputs/oe1_saneado_control_windows/
```

## 1. OE2 descriptores: evaluación sobre 51 variables

Incluye ya el perfil saneado del paquete y la fase 7 (prueba confirmatoria, hipótesis registradas el
10-10-2026).

```powershell
git add src/vae_nids/evaluation/oe2_descriptores/ outputs/oe2_descriptores/tables/ outputs/oe2_descriptores/figures/ outputs/oe2_descriptores/tablas_tex/ outputs/oe2_descriptores/fase1.log outputs/oe2_descriptores/fase2.log
git commit -m "OE2 descriptores: 33 descriptores de geometría latente en 10 semillas (51 variables) + fase 7 confirmatoria"
```

## 2. OE1 saneado, fase 1: conjunto por información nula

```powershell
git add src/vae_nids/oe1/__init__.py src/vae_nids/oe1/config.py src/vae_nids/oe1/datos.py src/vae_nids/oe1/seleccion.py src/vae_nids/oe1/informacion.py src/vae_nids/oe1/alternativos.py configs/features/ outputs/oe1_saneado/seleccion/ tests/oe1_saneado/
git commit -m "OE1-saneamiento fase 1: conjunto de entrada por información nula (77 -> 66), congelado con SHA-256"
```

## 3. OE1 saneado, fase 2: preprocesamiento corregido

```powershell
git add src/vae_nids/oe1/preproceso.py outputs/oe1_saneado/preproceso/
git commit -m "OE1-saneamiento fase 2: filtro de duraciones negativas por partición, MinMax float64 y resolución"
```

## 4. OE1 saneado, fases 3 y 4: configuración única, checkpoints autocontenidos, hiperparámetros

```powershell
git add configs/oe1_saneado.yaml configs/oe1_saneado_ref74.yaml configs/oe1_saneado_humo.yaml src/vae_nids/oe1/entrenamiento.py docs/oe1_saneado/hiperparametros.md requirements.txt
git commit -m "OE1-saneamiento fases 3-4: configuración única, checkpoints autocontenidos y tabla de hiperparámetros"
```

## 5. OE1 saneado, fases 5 y 6: diagnósticos, figuras, tablas e informe en borrador

```powershell
git add src/vae_nids/oe1/diagnosticos.py src/vae_nids/oe1/antes.py src/vae_nids/oe1/run.py src/vae_nids/oe1/figuras_informe.py src/vae_nids/oe1/tablas_informe.py src/vae_nids/oe1/muestreo_cifras.py outputs/oe1_saneado/tables/ outputs/oe1_saneado/figures_informe/ outputs/oe1_saneado/muestreo_cifras_informe.csv docs/oe1_saneado/
git commit -m "OE1-saneamiento fases 5-6: diagnósticos (solo benigno), figuras, tablas e informe en borrador"
```

## 6. OE2 sobre los modelos del OE1 saneado y corrida de pendientes en la VM

```powershell
git add .gitignore .gitattributes src/vae_nids/evaluation/oe2_robust/config.py src/vae_nids/evaluation/oe2_robust/phase0_diagnostics.py src/vae_nids/evaluation/oe2_robust/phase1_train_multiseed.py src/vae_nids/evaluation/oe2_robust/phase2_encode.py src/vae_nids/evaluation/oe2_robust/phase4_detectability.py scripts/run_pendientes_vm.sh scripts/vm_preparar_commit.sh scripts/vm_resumen_corrida.py docs/
git commit -m "OE2: perfil saneado (66 variables, modelos del OE1) y script único de pendientes para la VM"
```

## 7. Push

```powershell
git status --short      # no debería quedar nada relevante sin commit
git push
```

Después, en la VM: `docs/EJECUTAR_PENDIENTES_EN_VM.md`. Cuando la VM haga su commit y su push, aquí basta con
`git pull`.
