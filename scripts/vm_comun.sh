# Funciones comunes de las corridas en la VM Linux. Se usa con `source` desde un script que antes define:
#   OUT_DIR         carpeta de la documentación de la corrida (RUN_LOG.md, estado_etapas.jsonl, logs/)
#   RESUMEN_SCRIPT  script de Python que escribe el resumen final (se llama también si hay error fatal)
# Es la misma lógica de scripts/run_pendientes_vm.sh (que la tiene en línea porque se escribió antes).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$OUT_DIR/logs"
MD_LOG="$OUT_DIR/RUN_LOG.md"
ESTADO="$OUT_DIR/estado_etapas.jsonl"
VENV_DIR="$REPO_ROOT/.venv-linux"
PY="$VENV_DIR/bin/python"
mkdir -p "$LOG_DIR"

# Conjunto de instrucciones limitado: en la corrida del 10-10-2026 hubo SIGILL (exit 132) intermitente;
# con estos límites, la del 11-10-2026 terminó sin ninguno. No cambian los resultados.
export MKL_ENABLE_INSTRUCTIONS=SSE4_2
export DNNL_MAX_CPU_ISA=SSE41
export MKL_DEBUG_CPU_TYPE=5
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
  local raw="$LOG_DIR/${id}.log" rel="${OUT_DIR#"$REPO_ROOT"/}/logs/${id}.log"
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
  "$PY" "$RESUMEN_SCRIPT" >> "$LOG_DIR/E99.log" 2>&1 || true
  exit 1
}

cabecera() {   # cabecera <título>
  {
    echo ""
    echo "# $1 — $(ts)"
    echo ""
    echo "- Host: $(hostname); \`uname -a\`: $(uname -a)"
    echo "- CPUs: $(nproc 2>/dev/null || echo ?); memoria: $(free -g 2>/dev/null | awk '/Mem:/{print $2" GB total, "$7" GB disponibles"}')"
    echo "- Disco libre en el repo: $(df -h . | awk 'NR==2{print $4}')"
    echo "- Commit: $(git rev-parse HEAD 2>/dev/null) (rama $(git rev-parse --abbrev-ref HEAD 2>/dev/null))"
    echo "- Cambios sin commit en archivos versionados: $(git status --porcelain --untracked-files=no | wc -l)"
  } >> "$MD_LOG"
}

preparar_entorno() {
  local setup_log="$LOG_DIR/E00_entorno.log" modo="versiones exactas de requirements-vm.txt"
  command -v python3 >/dev/null 2>&1 || { echo "FATAL: falta python3" >&2; exit 1; }
  if [ ! -x "$PY" ]; then
    python3 -m venv "$VENV_DIR" >> "$setup_log" 2>&1 \
      || { echo "FATAL: no se pudo crear el venv (sudo apt-get install -y python3-venv)" >&2; exit 1; }
  fi
  {
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cpu \
      || "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  } >> "$setup_log" 2>&1
  if ! "$PY" -m pip install -r requirements-vm.txt >> "$setup_log" 2>&1; then
    modo="SIN versiones exactas (requirements-vm.txt falló; ver E00_entorno.log y pip_freeze.txt)"
    sed -E 's/[=<>~!].*$//' requirements-vm.txt | grep -vE '^\s*(#|$|torch)' > "$OUT_DIR/requirements_sin_versiones.txt"
    "$PY" -m pip install -r "$OUT_DIR/requirements_sin_versiones.txt" >> "$setup_log" 2>&1 \
      || fatal "pip install falló también sin versiones (ver ${OUT_DIR#"$REPO_ROOT"/}/logs/E00_entorno.log)."
  fi
  "$PY" -m pip install -e . >> "$setup_log" 2>&1 || fatal "pip install -e . falló (ver E00_entorno.log)."
  md "" "## Entorno" "" "- Python: $("$PY" --version 2>&1)" "- Dependencias: ${modo}" \
     "- torch: $("$PY" -c 'import torch; print(torch.__version__, "| hilos:", torch.get_num_threads())' 2>&1)"
}
