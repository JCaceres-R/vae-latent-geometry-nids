#!/usr/bin/env bash
# ============================================================================
# OE2 robusto -- script único para correr TODO el pipeline (Fases 0-6 + UMAP)
# en una VM Linux, sin supervisión, con logs en Markdown para revisar después.
#
# Uso:
#   1. Copiar/clonar este repo en la VM (rama oe2-robust).
#   2. Copiar a mano la carpeta data/processed/ (parquets + scaler.joblib +
#      feature_columns.json) desde la máquina donde se generó -- está
#      excluida de git por .gitignore. Sin esto, el script se detiene en el
#      chequeo de prerrequisitos con instrucciones claras.
#   3. Ejecutar, idealmente dentro de tmux/screen o con nohup, porque puede
#      tardar varias horas:
#        tmux new -s oe2robust
#        bash scripts/run_oe2_robust_vm.sh
#      (Ctrl+B D para salir de tmux sin matar el proceso; `tmux attach -t
#      oe2robust` para volver a entrar.)
#      Alternativa sin tmux:
#        nohup bash scripts/run_oe2_robust_vm.sh > /dev/null 2>&1 &
#        disown
#   4. Revisar progreso en vivo:
#        tail -f outputs/oe2_robust/VM_RUN_LOG.md
#
# Reanudable: cada fase pesada (entrenamiento, geometría, detectabilidad)
# cachea su trabajo por semilla en outputs/oe2_robust/**/_cache o checkpoints
# ya guardados -- si el script se interrumpe (SSH caído, VM reiniciada) y se
# vuelve a lanzar TAL CUAL, retoma donde iba en vez de recalcular todo desde
# cero. No hace falta pasar flags para reanudar.
# ============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="$REPO_ROOT/.venv-linux"
LOG_DIR="$REPO_ROOT/outputs/oe2_robust/logs"
MD_LOG="$REPO_ROOT/outputs/oe2_robust/VM_RUN_LOG.md"
PYTHON_BIN="$VENV_DIR/bin/python"

mkdir -p "$LOG_DIR"

# El wheel de pip de PyTorch trae Intel MKL/oneDNN, cuyo dispatcher interno
# de instrucciones SIMD es menos cuidadoso detectando el CPU real que
# OpenBLAS (el backend de numpy) -- en VMs con CPU sin AVX/AVX2/AVX512
# (frecuente en VMs académicas virtualizadas con exposición conservadora
# de CPU flags al hipervisor), MKL puede intentar ejecutar una instrucción
# AVX2 inexistente y el proceso muere con SIGILL (exit code 132), sin
# traceback de Python. Forzar techo de instrucciones a SSE4.2 evita esto;
# no cambia el resultado numérico, solo el código máquina usado. No hace
# nada si el CPU sí soporta AVX2 -- inofensivo dejarlo siempre activo.
export MKL_ENABLE_INSTRUCTIONS=SSE4_2
export DNNL_MAX_CPU_ISA=SSE41
export MKL_DEBUG_CPU_TYPE=5

timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

md_init() {
  {
    echo "# OE2 robusto -- log de ejecución en VM"
    echo ""
    echo "Generado por \`scripts/run_oe2_robust_vm.sh\`. Cada sección corresponde"
    echo "a una fase del pipeline, en el orden en que se ejecutó."
    echo ""
    echo "- Inicio de la corrida: $(timestamp)"
    echo "- Host: $(hostname)"
    echo "- \`uname -a\`: $(uname -a)"
  } >> "$MD_LOG"
}

# run_phase <nombre-legible> <nombre-archivo-log> <comando...>
# Corre el comando, mide tiempo, escribe un bloque en el .md con estado
# (OK/FALLÓ) y la cola del log crudo. Si falla, el script se detiene (las
# fases siguientes dependen de los artefactos de esta).
run_phase() {
  local label="$1"; local slug="$2"; shift 2
  local raw_log="$LOG_DIR/vm_${slug}.log"
  local start_ts start_s end_s dur rc

  start_ts="$(timestamp)"
  start_s=$(date +%s)
  echo "" >> "$MD_LOG"
  echo "## $label" >> "$MD_LOG"
  echo "" >> "$MD_LOG"
  echo "- Inicio: $start_ts" >> "$MD_LOG"
  echo ""
  echo "===== [$start_ts] iniciando: $label ====="

  echo "===== [$start_ts] $label =====" >> "$raw_log"
  "$@" >> "$raw_log" 2>&1
  rc=$?

  end_s=$(date +%s)
  dur=$((end_s - start_s))

  if [ $rc -eq 0 ]; then
    echo "- Fin: $(timestamp) (duración: ${dur}s = $((dur/60)) min)" >> "$MD_LOG"
    echo "- Estado: **OK**" >> "$MD_LOG"
    echo "===== [$( timestamp )] OK: $label (${dur}s) ====="
  else
    echo "- Fin: $(timestamp) (duración: ${dur}s = $((dur/60)) min)" >> "$MD_LOG"
    echo "- Estado: **FALLO** (exit code $rc)" >> "$MD_LOG"
    echo "===== [$( timestamp )] FALLO: $label (exit $rc) ====="
  fi

  echo "" >> "$MD_LOG"
  echo "Log completo: \`outputs/oe2_robust/logs/vm_${slug}.log\`. Últimas líneas:" >> "$MD_LOG"
  echo "" >> "$MD_LOG"
  echo '```' >> "$MD_LOG"
  tail -n 60 "$raw_log" >> "$MD_LOG"
  echo '```' >> "$MD_LOG"

  return $rc
}

fatal() {
  echo "" >> "$MD_LOG"
  echo "## ERROR FATAL" >> "$MD_LOG"
  echo "" >> "$MD_LOG"
  printf '%b\n' "$1" >> "$MD_LOG"
  printf 'FATAL: %b\n' "$1" >&2
  exit 1
}

# ----------------------------------------------------------------------
# 0. Prerrequisitos
# ----------------------------------------------------------------------
md_init

echo "===== chequeo de prerrequisitos ====="
MISSING=""
for f in \
  "data/processed/train_benign.parquet" \
  "data/processed/val_benign.parquet" \
  "data/processed/test_benign.parquet" \
  "data/processed/test_attacks.parquet" \
  "data/processed/feature_columns.json" \
  "outputs/checkpoints/vae_k8_beta1_input51_best.pt" \
; do
  if [ ! -f "$REPO_ROOT/$f" ]; then
    MISSING="$MISSING\n  - $f"
  fi
done

if [ -n "$MISSING" ]; then
  fatal "Faltan archivos requeridos, no incluidos en git (ver .gitignore):$MISSING

Estos dos casos cubren la mayoría:
  (a) data/processed/*.parquet + scaler.joblib + feature_columns.json:
      copiar la carpeta completa data/processed/ desde la máquina donde
      se generó (ej. scp -r usuario@origen:.../data/processed ./data/).
  (b) outputs/checkpoints/vae_k8_beta1_input51_best.pt: este archivo SÍ
      debería venir en el repo (está trackeado en git) -- si falta, hacer
      'git lfs pull' si el repo usa Git LFS, o revisar que el clone/pull
      haya sido completo."
fi

command -v python3 >/dev/null 2>&1 || fatal "python3 no está instalado en esta VM."
command -v git >/dev/null 2>&1 || fatal "git no está instalado en esta VM."

echo "[ok] prerrequisitos de datos presentes."

# ----------------------------------------------------------------------
# 1. Entorno virtual + dependencias
# ----------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "===== creando venv en $VENV_DIR ====="
  python3 -m venv "$VENV_DIR" || fatal "no se pudo crear el venv con 'python3 -m venv'. \
En Debian/Ubuntu suele faltar el paquete del módulo venv -- probar:\n  sudo apt-get install -y python3-venv\ny volver a correr este script."
fi

echo "===== instalando dependencias ====="
"$PYTHON_BIN" -m pip install --upgrade pip >> "$LOG_DIR/vm_setup.log" 2>&1
"$PYTHON_BIN" -m pip install -r requirements.txt >> "$LOG_DIR/vm_setup.log" 2>&1 \
  || fatal "falló pip install -r requirements.txt (ver $LOG_DIR/vm_setup.log)."
"$PYTHON_BIN" -m pip install -r scripts/requirements-oe2-robust.txt >> "$LOG_DIR/vm_setup.log" 2>&1 \
  || fatal "falló pip install -r scripts/requirements-oe2-robust.txt (ver $LOG_DIR/vm_setup.log)."
"$PYTHON_BIN" -m pip install -e . >> "$LOG_DIR/vm_setup.log" 2>&1 \
  || fatal "falló pip install -e . (ver $LOG_DIR/vm_setup.log)."

{
  echo ""
  echo "## Entorno"
  echo ""
  echo "- Python: $("$PYTHON_BIN" --version 2>&1)"
  echo "- CUDA disponible: $("$PYTHON_BIN" -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
  echo "- Paquetes clave:"
  echo '```'
  "$PYTHON_BIN" -m pip freeze 2>/dev/null | grep -iE "^(torch|numpy|pandas|scikit-learn|scipy|umap-learn|statsmodels|psutil)="
  echo '```'
} >> "$MD_LOG"

# ----------------------------------------------------------------------
# 2. Tests (fail-fast si algo del entorno está mal antes de correr horas de cómputo)
# ----------------------------------------------------------------------
run_phase "Tests unitarios (oe2_robust)" "tests" \
  "$PYTHON_BIN" -m pytest tests/oe2_robust/ -q \
  || fatal "los tests unitarios fallaron -- revisar antes de continuar (no se corre el pipeline sobre un entorno no verificado)."

# ----------------------------------------------------------------------
# 3. Pipeline, fase por fase (cada una resumible internamente)
# ----------------------------------------------------------------------
run_phase "Fase 0 -- diagnóstico y presupuesto" "phase0" \
  "$PYTHON_BIN" -m vae_nids.evaluation.oe2_robust.phase0_diagnostics \
  || fatal "Fase 0 falló."

run_phase "Fase 1 -- entrenamiento multi-semilla" "phase1" \
  "$PYTHON_BIN" -m vae_nids.evaluation.oe2_robust.phase1_train_multiseed \
  || fatal "Fase 1 falló."

run_phase "Fase 2 -- codificación (mu, log sigma^2)" "phase2" \
  "$PYTHON_BIN" -m vae_nids.evaluation.oe2_robust.phase2_encode \
  || fatal "Fase 2 falló."

run_phase "Fase 3 -- geometría distribucional" "phase3" \
  "$PYTHON_BIN" -m vae_nids.evaluation.oe2_robust.phase3_geometry \
  || fatal "Fase 3 falló."

run_phase "Fase 4 -- detectabilidad confirmatoria" "phase4" \
  "$PYTHON_BIN" -m vae_nids.evaluation.oe2_robust.phase4_detectability \
  || fatal "Fase 4 falló."

run_phase "Fase 5 -- correlación geometría-detectabilidad" "phase5" \
  "$PYTHON_BIN" -m vae_nids.evaluation.oe2_robust.phase5_correlation \
  || fatal "Fase 5 falló."

run_phase "Fase 6 -- figuras (sin UMAP)" "phase6" \
  "$PYTHON_BIN" -c "from vae_nids.evaluation.oe2_robust import phase6_figures as p6; p6.main(include_umap=False)" \
  || fatal "Fase 6 (figuras) falló."

run_phase "Fase 6b -- proyección UMAP" "phase6_umap" \
  "$PYTHON_BIN" -c "
from vae_nids.evaluation.oe2_robust import config as rcfg, phase6_figures as p6
for seed in rcfg.UMAP_SEEDS:
    p6.fig_umap(seed)
"
# nota: la UMAP NO usa fatal -- si falla o tarda demasiado, el resto del
# informe ya está completo y utilizable; se documenta el fallo en el .md
# y el pipeline sigue hasta el resumen final.

# ----------------------------------------------------------------------
# 4. Resumen final
# ----------------------------------------------------------------------
{
  echo ""
  echo "## Resumen final"
  echo ""
  echo "- Fin de la corrida: $(timestamp)"
  echo ""
  echo '```'
} >> "$MD_LOG"
"$PYTHON_BIN" -c "
import json
from vae_nids.evaluation.oe2_robust import config as rcfg
with open(rcfg.METRICS_DIR / 'phase5_hypothesis_principal.json', encoding='utf-8') as f:
    h1 = json.load(f)
print(f\"Semillas: {rcfg.ALL_SEEDS} (n={len(rcfg.ALL_SEEDS)})\")
print(f\"Par principal: {h1['pair']}\")
c = h1['combined']
print(f\"rho combinado = {c['rho_combined']:.4f}  IC95=[{c['ci_low']:.4f},{c['ci_high']:.4f}]\")
print(f\"media por semilla = {c['rho_per_seed_mean']:.4f}  sd={c['rho_per_seed_sd']:.4f}\")
prev = h1['previous_oe2_single_seed']
print(f\"OE2 original (una sola semilla): rho={prev['rho']:.4f} p={prev['p_value']:.4f}\")
" >> "$MD_LOG" 2>&1
echo '```' >> "$MD_LOG"

echo "" >> "$MD_LOG"
echo "Pipeline completo. Informe a completar a mano/por Claude en:" >> "$MD_LOG"
echo '`outputs/oe2_robust/INFORME_OE2_ROBUSTO.md`, con todos los CSV en `outputs/oe2_robust/tables/` y JSON en `outputs/oe2_robust/metrics/`.' >> "$MD_LOG"

echo ""
echo "===== PIPELINE COMPLETO ====="
echo "Log: $MD_LOG"
