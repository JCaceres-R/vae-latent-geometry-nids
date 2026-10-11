# Segunda corrida en la VM: reconciliación, OE3 y OE4

Script: `scripts/run_oe3_oe4_vm.sh`. La documentación de la corrida queda en `outputs/vm_oe3_oe4/`, igual que
la de la primera (`outputs/vm_pendientes/`).

Requiere que la primera corrida (`scripts/run_pendientes_vm.sh`) se haya hecho **en esta misma VM**: usa sus
checkpoints, sus datos procesados y los scores por conexión del OE2, que no se versionan.

## Qué hace

| Etapa | Qué hace | Salida | Tiempo aprox. |
|---|---|---|---|
| E00–E01 | Entorno (`requirements-vm.txt`, las versiones exactas de la VM) y tests | | 2–5 min |
| E10 | **Reconciliación (OE2):** modelo mixto de la fase 5 frente al nivel de flujo de la fase 7, con la misma muestra de conexiones | `outputs/oe2_descriptores_saneado/RESULTADOS_RECONCILIACION.md` | 15–30 min |
| E20 | **OE3:** 50 000 flujos sintéticos por modelo (8 modelos), en 3 variantes; KS y Wasserstein por variable, Wasserstein rebanado, clasificador real frente a sintético, memorización | `outputs/oe3/` | 30–45 min |
| E3_*_1…6 | **OE4**, para β = 0,5, β = 2 y k = 16: entrenar 5 semillas (42–46), codificar, AUC, 17 descriptores, cribado y prueba confirmatoria | `outputs/oe4/<config>/` | 1,5–2,5 h por configuración |
| E40 | **OE4:** concordancia del ranking de las 11 familias con la base (Spearman, Kendall, W) y el hallazgo en cada configuración | `outputs/oe4/RESULTADOS_OE4.md` | segundos |
| E99 | Resumen | `outputs/vm_oe3_oe4/RESUMEN_CORRIDA.md` | segundos |

En total, entre 7 y 10 horas.

Si falla una configuración del OE4, las otras siguen y E40 compara las que terminaron. Un proceso que muere
por SIGILL se reintenta solo, hasta 3 veces.

## Decisiones de diseño (fijadas antes de ver resultados)

- **OE3.** La variante principal es z ~ N(0, I) y x ~ p(x | z): la generación genuina del VAE.
  - z ~ N(0, I) con x = μ_x(z) separa el efecto del ruido del decodificador.
  - z ~ q(z | x) sobre val_benign separa el efecto del desajuste entre el prior y el posterior agregado.
  - Cada indicador se compara con lo que difieren dos muestras reales (val_benign frente a test_benign).
- **OE4.** Las configuraciones son β = 0,5, β = 2 y k = 16, cada una con 5 semillas. Son idénticas a la base
  en todo lo demás (`configs/oe4_*.yaml`). Se usan las 66 variables del OE1 saneado y las mismas hipótesis de
  la fase 7.
- **Reconciliación.** Se reutiliza la misma muestra de conexiones que el modelo mixto de la fase 5 y se
  separa el efecto de cada decisión: incluir el benigno, usar valores o rangos, y comparar entre familias o
  dentro de cada familia.

## Pasos

Antes, en esta máquina: commit y push del código nuevo (lo que liste `git status`).

En la VM:

```bash
cd ~/vae-latent-geometry-nids
git pull
tmux new -A -s p
bash scripts/run_oe3_oe4_vm.sh
```

Para salir sin detenerla: `Ctrl+B` y luego `D`. Para ver el avance:

```bash
cat outputs/vm_oe3_oe4/estado_etapas.jsonl
```

Al terminar:

```bash
bash scripts/vm_preparar_commit.sh outputs/vm_oe3_oe4
git commit -F outputs/vm_oe3_oe4/MENSAJE_COMMIT.txt
git push
```

Aquí: `git pull` y revisar `outputs/vm_oe3_oe4/RESUMEN_CORRIDA.md`.
