#!/usr/bin/env bash
# ============================================================================
# Segunda corrida en la VM Linux: todo lo que queda por calcular, sin supervisión.
#
#   1. Reconciliación del nivel de conexión (OE2): modelo mixto de la fase 5 frente a la fase 7.
#   2. OE3: calidad generativa del decodificador (8 modelos del OE1 saneado).
#   3. OE4: reentrenar con beta = 0,5, beta = 2 y k = 16 (5 semillas cada una: 15 modelos),
#      codificarlos, calcular AUC y descriptores, la prueba confirmatoria en cada configuración,
#      y comparar el ranking de familias con la base (beta = 1, k = 8).
#   4. Resumen para revisar desde la otra máquina: outputs/vm_oe3_oe4/
#
# Requiere la corrida anterior (scripts/run_pendientes_vm.sh) hecha EN ESTA VM: usa sus checkpoints,
# sus datos procesados y los scores por conexión del OE2 robusto (no se versionan).
#
# Uso (guía completa en docs/EJECUTAR_OE3_OE4_EN_VM.md):
#   git pull
#   tmux new -A -s p
#   bash scripts/run_oe3_oe4_vm.sh
#   (Ctrl+B D para salir; tmux attach -t p para volver)
#
# Reanudable: el entrenamiento salta las semillas con checkpoint y las fases del OE2 reutilizan su caché.
#
# Al terminar:
#   bash scripts/vm_preparar_commit.sh outputs/vm_oe3_oe4
#   git commit -F outputs/vm_oe3_oe4/MENSAJE_COMMIT.txt && git push
# ============================================================================
set -uo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/outputs/vm_oe3_oe4"
RESUMEN_SCRIPT="scripts/vm_resumen_oe3_oe4.py"
# shellcheck source=vm_comun.sh
source "$(dirname "${BASH_SOURCE[0]}")/vm_comun.sh"

cabecera "Corrida OE3 + OE4 + reconciliación"

# ----------------------------------------------------------------------------
# 0. Prerrequisitos: lo que dejó la corrida anterior en esta VM
# ----------------------------------------------------------------------------
MISSING=""
for f in data/processed/oe1_saneado/train_benign_crudo.parquet data/processed/oe1_saneado/test_benign_escalado.parquet \
         outputs/oe1_saneado/checkpoints/oe1_info_nula_seed42.pt outputs/oe1_saneado/checkpoints/oe1_info_nula_seed49.pt \
         outputs/oe2_robust_saneado/latent_vectors/seed42/scores/BENIGN.npz \
         outputs/oe2_robust_saneado/metrics/phase5_mixed_model.json \
         outputs/oe2_descriptores_saneado/tables/confirmatoria_hipotesis.csv \
         configs/oe4_beta0p5.yaml configs/oe4_beta2.yaml configs/oe4_k16.yaml requirements-vm.txt; do
  [ -e "$f" ] || MISSING="$MISSING $f"
done
if [ -n "$MISSING" ]; then
  md "" "## ERROR FATAL" "" "Faltan archivos:${MISSING}" "" \
     "Esta corrida necesita lo que dejó scripts/run_pendientes_vm.sh en esta misma VM, y el código nuevo (git pull)."
  echo "FATAL: faltan archivos:${MISSING}" >&2
  exit 1
fi
echo "[ok] prerrequisitos presentes"

preparar_entorno

etapa E01 "Tests (OE1 saneado y estadística del OE2)" "" \
  "$PY" -m pytest tests/oe1_saneado tests/oe2_robust/test_stats_utils.py -q -p no:cacheprovider \
  || fatal "los tests fallaron; revisar ${OUT_DIR#"$REPO_ROOT"/}/logs/E01.log."

export VAE_NIDS_OE2_PERFIL=saneado

# ----------------------------------------------------------------------------
# 1. Reconciliación (minutos) y 2. OE3 (≈ 30–45 min): primero lo corto
# ----------------------------------------------------------------------------
etapa E10 "Reconciliación: modelo mixto (fase 5) frente a nivel de flujo (fase 7)" "E01" \
  "$PY" -m vae_nids.evaluation.oe2_descriptores.fase8_reconciliacion

etapa E20 "OE3: calidad generativa del decodificador" "E01" \
  "$PY" -m vae_nids.evaluation.oe3.calidad_generativa

# ----------------------------------------------------------------------------
# 3. OE4: por configuración, entrenar (≈ 1 h), codificar, AUC, descriptores, cribado y confirmatoria
# ----------------------------------------------------------------------------
OE4_IDS=""
for tag in beta0p5 beta2 k16; do
  cfg="configs/oe4_${tag}.yaml"
  i="E3_${tag}"
  etapa "${i}_1" "OE4 ${tag}: entrenamiento (5 semillas) y diagnósticos" "E01" \
    "$PY" -m vae_nids.oe1.run --config "$cfg"
  etapa "${i}_2" "OE4 ${tag}: codificación (mu, log sigma^2)" "${i}_1" \
    env VAE_NIDS_OE2_CONFIG="$cfg" "$PY" -m vae_nids.evaluation.oe2_robust.phase2_encode
  etapa "${i}_3" "OE4 ${tag}: detectabilidad y referencia benigna" "${i}_2" \
    env VAE_NIDS_OE2_CONFIG="$cfg" "$PY" -m vae_nids.evaluation.oe2_descriptores.fase1_detectabilidad
  etapa "${i}_4" "OE4 ${tag}: descriptores por familia y semilla" "${i}_3" \
    env VAE_NIDS_OE2_CONFIG="$cfg" "$PY" -m vae_nids.evaluation.oe2_descriptores.fase2_descriptores
  etapa "${i}_5" "OE4 ${tag}: cribado de descriptores" "${i}_4" \
    env VAE_NIDS_OE2_CONFIG="$cfg" "$PY" -m vae_nids.evaluation.oe2_descriptores.fase3_evaluacion
  etapa "${i}_6" "OE4 ${tag}: prueba confirmatoria" "${i}_4" \
    env VAE_NIDS_OE2_CONFIG="$cfg" "$PY" -m vae_nids.evaluation.oe2_descriptores.fase7_confirmatoria
  OE4_IDS="$OE4_IDS ${i}_1 ${i}_2 ${i}_3 ${i}_4 ${i}_5 ${i}_6"
done

# Corre aunque falle una configuración: compara las que hayan terminado y anota las que faltan.
etapa E40 "OE4: concordancia de rankings y hallazgo por configuración" "E01" \
  "$PY" -m vae_nids.evaluation.oe4.analisis

# ----------------------------------------------------------------------------
# 4. Resumen y archivos para el commit
# ----------------------------------------------------------------------------
md "" "## Fin de la corrida — $(ts)" ""
for k in E01 E10 E20 $OE4_IDS E40; do
  md "- ${k}: ${STATUS[$k]:-no corrió}"
done
etapa E99 "Resumen de la corrida" "" "$PY" "$RESUMEN_SCRIPT"

echo ""
echo "===== CORRIDA TERMINADA ====="
echo "Resumen: outputs/vm_oe3_oe4/RESUMEN_CORRIDA.md"
echo "Log:     outputs/vm_oe3_oe4/RUN_LOG.md"
echo ""
echo "Para versionar los resultados:"
echo "  bash scripts/vm_preparar_commit.sh outputs/vm_oe3_oe4"
echo "  git commit -F outputs/vm_oe3_oe4/MENSAJE_COMMIT.txt && git push"
