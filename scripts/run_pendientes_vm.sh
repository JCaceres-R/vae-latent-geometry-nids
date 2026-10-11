#!/usr/bin/env bash
# ============================================================================
# Pendientes del OE1 y del OE2 en la VM Linux, en una sola corrida y sin supervisión.
#
#   1. OE1 saneado: datos, preprocesamiento, 8 semillas (42-49) + referencia de 74,
#      diagnósticos, figuras y tablas del informe.
#   2. OE2 robusto recalculado sobre ESOS modelos (perfil saneado, 66 variables):
#      fases 2-6 + UMAP -> outputs/oe2_robust_saneado/
#   3. OE2 descriptores sobre esos modelos: fases 1-6 + fase 7 (prueba
#      confirmatoria, hipótesis registradas el 10-10-2026)
#      -> outputs/oe2_descriptores_saneado/
#   4. Resumen para revisar desde la otra máquina: outputs/vm_pendientes/
#
# Los resultados de 51 variables (outputs/oe2_robust/, outputs/oe2_descriptores/) no se tocan:
# sirven de comparación.
#
# Uso (guía completa en docs/EJECUTAR_PENDIENTES_EN_VM.md):
#   git pull
#   tmux new -s pendientes
#   bash scripts/run_pendientes_vm.sh
#   (Ctrl+B D para salir; tmux attach -t pendientes para volver)
#   tail -f outputs/vm_pendientes/RUN_LOG.md          # progreso
#
# Reanudable: si se corta, se relanza igual. El OE1 salta las semillas con checkpoint y
# las fases del OE2 reutilizan su caché por semilla.
#
# Al terminar:
#   bash scripts/vm_preparar_commit.sh                # solo hace git add de lo que se versiona
#   git commit -F outputs/vm_pendientes/MENSAJE_COMMIT.txt && git push
# ============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="$REPO_ROOT/outputs/vm_pendientes"
LOG_DIR="$OUT_DIR/logs"
MD_LOG="$OUT_DIR/RUN_LOG.md"
ESTADO="$OUT_DIR/estado_etapas.jsonl"
VENV_DIR="$REPO_ROOT/.venv-linux"
PY="$VENV_DIR/bin/python"
mkdir -p "$LOG_DIR"

# Igual que scripts/run_oe2_robust_vm.sh: en VMs que no exponen AVX2, MKL/oneDNN de torch
# puede morir con SIGILL (exit 132). Limitar el conjunto de instrucciones no cambia los resultados.
export MKL_ENABLE_INSTRUCTIONS=SSE4_2
export DNNL_MAX_CPU_ISA=SSE41
export MKL_DEBUG_CPU_TYPE=5
# Corrida del 10-10-2026: SIGILL (exit 132) intermitente, a veces a los 4 s de arrancar (al leer los
# parquets o al importar). Se limitan también los demás despachadores de SIMD que deciden en tiempo de
# ejecución: Arrow (lectura de parquet), los kernels propios de torch (ATen) y numba (UMAP, pynndescent).
# No cambian los resultados, solo el código máquina. PYTHONFAULTHANDLER deja en el log la pila de Python
# del proceso que muere, para saber qué biblioteca fue.
export ARROW_USER_SIMD_LEVEL=SSE4_2
export ATEN_CPU_CAPABILITY=default
export NUMBA_CPU_NAME=generic
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
MAX_INTENTOS=3   # una etapa que muere por SIGILL (132) se reintenta hasta este número de veces

declare -A STATUS

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

md() { printf '%s\n' "$@" >> "$MD_LOG"; }

registrar() {   # id etapa estado inicio segundos rc log
  printf '{"id": "%s", "etapa": "%s", "estado": "%s", "inicio": "%s", "segundos": %s, "rc": %s, "log": "%s"}\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" >> "$ESTADO"
}

# etapa <id> <descripción> "<dependencias separadas por espacio>" <comando...>
# Si alguna dependencia no terminó OK, la etapa se salta (SALTADA) y se sigue con las demás.
etapa() {
  local id="$1" label="$2" deps="$3"; shift 3
  local raw="$LOG_DIR/${id}.log" rel="outputs/vm_pendientes/logs/${id}.log"
  local start_ts start_s dur rc d
  start_ts="$(ts)"
  for d in $deps; do
    if [ "${STATUS[$d]:-}" != "OK" ]; then
      STATUS[$id]="SALTADA"
      md "" "## ${id} — ${label}" "" "- **SALTADA**: la etapa ${d} no terminó bien (${STATUS[$d]:-no corrió})."
      registrar "$id" "$label" "SALTADA" "$start_ts" 0 -1 "$rel"
      echo "===== [$start_ts] SALTADA ${id}: ${label} (depende de ${d}) ====="
      return 1
    fi
  done
  md "" "## ${id} — ${label}" "" "- Inicio: ${start_ts}" "- Comando: \`$*\`"
  echo "===== [$start_ts] ${id}: ${label} ====="
  start_s=$(date +%s)
  local intento=1 stdin_copy=""
  if [ ! -t 0 ]; then              # etapas con heredoc: se guarda la entrada para poder reintentar
    stdin_copy="$(mktemp)"; cat > "$stdin_copy"
  fi
  while :; do
    echo "===== [$(ts)] ${id}: ${label} (intento ${intento}) =====" >> "$raw"
    if [ -n "$stdin_copy" ]; then "$@" < "$stdin_copy" >> "$raw" 2>&1; else "$@" >> "$raw" 2>&1; fi
    rc=$?
    [ $rc -eq 132 ] && [ $intento -lt $MAX_INTENTOS ] || break
    md "- Intento ${intento} murió por SIGILL (exit 132); se reintenta."
    echo "      SIGILL en ${id}, intento ${intento}; reintentando"
    intento=$((intento + 1))
  done
  [ -n "$stdin_copy" ] && rm -f "$stdin_copy"
  dur=$(( $(date +%s) - start_s ))
  if [ $rc -eq 0 ]; then STATUS[$id]="OK"; else STATUS[$id]="FALLO"; fi
  md "- Fin: $(ts) (${dur} s = $((dur / 60)) min)" "- Estado: **${STATUS[$id]}** (exit ${rc})" "" \
     "Log completo: \`${rel}\`. Últimas líneas:" "" '```'
  tail -n 40 "$raw" >> "$MD_LOG"
  md '```'
  registrar "$id" "$label" "${STATUS[$id]}" "$start_ts" "$dur" "$rc" "$rel"
  echo "===== [$(ts)] ${STATUS[$id]} ${id} (${dur} s) ====="
  return $rc
}

fatal() {
  md "" "## ERROR FATAL" "" "$1"
  echo "FATAL: $1" >&2
  "$PY" scripts/vm_resumen_corrida.py >> "$LOG_DIR/E99.log" 2>&1 || true
  exit 1
}

# ----------------------------------------------------------------------------
# 0. Cabecera y prerrequisitos
# ----------------------------------------------------------------------------
{
  echo ""
  echo "# Corrida de pendientes (OE1 saneado + OE2 recalculado) — $(ts)"
  echo ""
  echo "- Host: $(hostname); \`uname -a\`: $(uname -a)"
  echo "- CPUs: $(nproc 2>/dev/null || echo ?); memoria: $(free -g 2>/dev/null | awk '/Mem:/{print $2" GB total, "$7" GB disponibles"}')"
  echo "- Disco libre en el repo: $(df -h . | awk 'NR==2{print $4}')"
  echo "- Commit: $(git rev-parse HEAD 2>/dev/null) (rama $(git rev-parse --abbrev-ref HEAD 2>/dev/null))"
  echo "- Cambios sin commit en archivos versionados: $(git status --porcelain --untracked-files=no | wc -l)"
} >> "$MD_LOG"

command -v python3 >/dev/null 2>&1 || { echo "FATAL: falta python3" >&2; exit 1; }

MISSING=""
if ! ls data/processed/oe1_saneado/train_benign_crudo.parquet >/dev/null 2>&1; then
  for f in Monday-WorkingHours.csv Tuesday-WorkingHours.csv Wednesday-WorkingHours.csv \
           Thursday-WorkingHours.csv Friday-WorkingHours.csv; do
    [ -f "data/raw/$f" ] || MISSING="$MISSING data/raw/$f"
  done
fi
for f in configs/oe1_saneado.yaml configs/oe1_saneado_ref74.yaml configs/features/oe1_info_nula.yaml \
         configs/features/referencia_74.yaml outputs/latent_vectors/manifest.json \
         src/vae_nids/evaluation/oe2_descriptores/fase7_confirmatoria.py; do
  [ -f "$f" ] || MISSING="$MISSING $f"
done
if [ -n "$MISSING" ]; then
  md "" "## ERROR FATAL" "" "Faltan archivos:${MISSING}" "" \
     "Los CSV crudos de CICIDS2017 (versión corregida, Engelen et al.) van en data/raw/ (no están en git)." \
     "Los demás vienen del repo: revisa que hiciste git pull después del commit local."
  echo "FATAL: faltan archivos:${MISSING}" >&2
  exit 1
fi
echo "[ok] prerrequisitos presentes"

# ----------------------------------------------------------------------------
# 1. Entorno
# ----------------------------------------------------------------------------
SETUP_LOG="$LOG_DIR/E00_entorno.log"
if [ ! -x "$PY" ]; then
  python3 -m venv "$VENV_DIR" >> "$SETUP_LOG" 2>&1 \
    || { echo "FATAL: no se pudo crear el venv (sudo apt-get install -y python3-venv)" >&2; exit 1; }
fi
DEPS_MODO="versiones exactas de requirements.txt"
{
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cpu \
    || "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
} >> "$SETUP_LOG" 2>&1
if ! "$PY" -m pip install -r requirements.txt >> "$SETUP_LOG" 2>&1; then
  # Las versiones exactas son las de Windows/Python 3.14; si alguna no existe para el Python de la VM,
  # se instalan sin fijar versión y queda registrado (pip_freeze.txt guarda lo que se usó).
  DEPS_MODO="SIN versiones exactas (requirements.txt falló; ver E00_entorno.log y pip_freeze.txt)"
  sed -E 's/[=<>~!].*$//' requirements.txt | grep -vE '^\s*(#|$|torch)' > "$OUT_DIR/requirements_sin_versiones.txt"
  "$PY" -m pip install -r "$OUT_DIR/requirements_sin_versiones.txt" >> "$SETUP_LOG" 2>&1 \
    || fatal "pip install falló también sin versiones (ver outputs/vm_pendientes/logs/E00_entorno.log)."
fi
"$PY" -m pip install -e . >> "$SETUP_LOG" 2>&1 || fatal "pip install -e . falló (ver E00_entorno.log)."
md "" "## Entorno" "" "- Python: $("$PY" --version 2>&1)" "- Dependencias: ${DEPS_MODO}" \
   "- torch: $("$PY" -c 'import torch; print(torch.__version__, "| hilos:", torch.get_num_threads())' 2>&1)"

# ----------------------------------------------------------------------------
# 2. Tests (si fallan, el entorno no es confiable y no se corre nada más)
# ----------------------------------------------------------------------------
etapa E01 "Tests (OE1 saneado y estadística del OE2)" "" \
  "$PY" -m pytest tests/oe1_saneado tests/oe2_robust/test_stats_utils.py -q -p no:cacheprovider \
  || fatal "los tests fallaron; revisar outputs/vm_pendientes/logs/E01.log antes de correr horas de cómputo."

# ----------------------------------------------------------------------------
# 3. OE1 saneado
# ----------------------------------------------------------------------------
etapa E10 "OE1: datos, preproceso, 8 semillas + referencia 74, diagnósticos" "E01" \
  "$PY" -m vae_nids.oe1.run --config configs/oe1_saneado.yaml --ref74 configs/oe1_saneado_ref74.yaml

etapa E11 "OE1: verificación de checkpoints y diagnósticos" "E10" \
  "$PY" - <<'EOF'
import yaml, pandas as pd
from pathlib import Path
cfg = yaml.safe_load(open("configs/oe1_saneado.yaml", encoding="utf-8"))
ck = Path(cfg["salidas"]["dir"]) / "checkpoints"
need = [ck / f"oe1_info_nula_seed{s}.pt" for s in cfg["semillas"]["principal"]] + [ck / "referencia_74_seed42.pt"]
miss = [str(p) for p in need if not p.exists()]
assert not miss, f"faltan checkpoints: {miss}"
t = pd.read_csv(Path(cfg["salidas"]["dir"]) / "tables" / "diagnosticos_por_semilla.csv")
assert len(t) == len(need), f"diagnósticos con {len(t)} filas, se esperaban {len(need)}"
print(t[["corrida", "seed", "n_columnas", "mejor_epoca", "activas_burda", "kl_total", "frac_logvar_dec_en_piso"]].to_string())
print("OK:", len(need), "checkpoints y diagnósticos completos")
EOF

etapa E12 "OE1: figuras del informe (selección y preproceso)" "E10" \
  "$PY" -m vae_nids.oe1.figuras_informe --config configs/oe1_saneado.yaml

etapa E13 "OE1: tablas LaTeX y cifras del informe" "E11" \
  "$PY" -m vae_nids.oe1.tablas_informe --config configs/oe1_saneado.yaml --dest outputs/oe1_saneado/informe_tablas

# ----------------------------------------------------------------------------
# 4. OE2 robusto sobre los modelos del OE1 saneado
# ----------------------------------------------------------------------------
export VAE_NIDS_OE2_PERFIL=saneado

etapa E20 "OE2 robusto: codificación (mu, log sigma^2) por semilla" "E11" \
  "$PY" -m vae_nids.evaluation.oe2_robust.phase2_encode

etapa E21 "OE2 robusto: geometría distribucional" "E20" \
  "$PY" -m vae_nids.evaluation.oe2_robust.phase3_geometry

# La fase 4 se corre semilla por semilla (cada una queda en caché) y después se agregan las tablas.
# En la corrida anterior del OE2 robusto la tabla final solo quedó con la semilla 42; así, si una
# semilla falla, las demás quedan hechas y el log dice cuál fue.
SEEDS_OE2="$("$PY" -c 'from vae_nids.evaluation.oe2_robust import config as r; print(" ".join(map(str, r.ALL_SEEDS)))')"
md "" "- Semillas del OE2 (perfil saneado): ${SEEDS_OE2}"
for s in $SEEDS_OE2; do
  etapa "E22_s${s}" "OE2 robusto: detectabilidad, semilla ${s}" "E20" \
    "$PY" -m vae_nids.evaluation.oe2_robust.phase4_detectability --seed "$s"
done
etapa E22 "OE2 robusto: detectabilidad, tablas de todas las semillas" "E20" \
  "$PY" -m vae_nids.evaluation.oe2_robust.phase4_detectability

etapa E23 "OE2 robusto: correlación geometría-detectabilidad" "E21 E22" \
  "$PY" -m vae_nids.evaluation.oe2_robust.phase5_correlation

etapa E24 "OE2 robusto: figuras" "E23" \
  "$PY" -c "from vae_nids.evaluation.oe2_robust import phase6_figures as p6; p6.main(include_umap=False)"

# Proyecta todos los flujos de ataque (≈ 440 000 puntos por semilla): tardó 4,4 h la primera vez. Si la
# figura de una semilla ya existe, no se recalcula.
etapa E25 "OE2 robusto: UMAP (apoyo visual)" "E20" \
  "$PY" -c "from vae_nids.evaluation.oe2_robust import config as rcfg, phase6_figures as p6; [p6.fig_umap(s) for s in rcfg.UMAP_SEEDS if not (rcfg.FIGURES_DIR / f'fig_umap_seed{s}.png').exists()]"

# ----------------------------------------------------------------------------
# 5. OE2 descriptores sobre los modelos del OE1 saneado
# ----------------------------------------------------------------------------
etapa E30 "Descriptores: detectabilidad y referencia benigna" "E20" \
  "$PY" -m vae_nids.evaluation.oe2_descriptores.fase1_detectabilidad

etapa E31 "Descriptores: 17 descriptores por familia y semilla" "E30" \
  "$PY" -m vae_nids.evaluation.oe2_descriptores.fase2_descriptores

etapa E32 "Descriptores: cribado y validación (attempted)" "E31 E21" \
  "$PY" -m vae_nids.evaluation.oe2_descriptores.fase3_evaluacion

etapa E33 "Descriptores: duplicados, métrica del decodificador, UMAP sin duplicados" "E30" \
  "$PY" -m vae_nids.evaluation.oe2_descriptores.fase4_diagnosticos

etapa E34 "Descriptores: figuras" "E32 E33" \
  "$PY" -m vae_nids.evaluation.oe2_descriptores.fase5_figuras

etapa E35 "Descriptores: tablas LaTeX y cifras" "E32 E33" \
  "$PY" -m vae_nids.evaluation.oe2_descriptores.fase6_tablas_tex

etapa E36 "Prueba confirmatoria (primario + 6 secundarios, Holm)" "E31" \
  "$PY" -m vae_nids.evaluation.oe2_descriptores.fase7_confirmatoria

# ----------------------------------------------------------------------------
# 6. Resumen y archivos para el commit
# ----------------------------------------------------------------------------
md "" "## Fin de la corrida — $(ts)" ""
for k in E01 E10 E11 E12 E13 E20 E21 $(for s in $SEEDS_OE2; do printf 'E22_s%s ' "$s"; done) E22 E23 E24 E25 \
         E30 E31 E32 E33 E34 E35 E36; do
  md "- ${k}: ${STATUS[$k]:-no corrió}"
done
etapa E99 "Resumen de la corrida" "" "$PY" scripts/vm_resumen_corrida.py

echo ""
echo "===== CORRIDA TERMINADA ====="
echo "Resumen: outputs/vm_pendientes/RESUMEN_CORRIDA.md"
echo "Log:     outputs/vm_pendientes/RUN_LOG.md"
echo ""
echo "Para versionar los resultados:"
echo "  bash scripts/vm_preparar_commit.sh"
echo "  git commit -F outputs/vm_pendientes/MENSAJE_COMMIT.txt && git push"
