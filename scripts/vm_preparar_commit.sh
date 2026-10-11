#!/usr/bin/env bash
# Deja en el área de preparación (git add) solo lo que se versiona de una corrida en la VM. El argumento
# es la carpeta de documentación de la corrida (por defecto outputs/vm_pendientes, la de
# scripts/run_pendientes_vm.sh; outputs/vm_oe3_oe4 para scripts/run_oe3_oe4_vm.sh).
# No hace commit ni push: eso lo haces tú después de revisar.
#
#   bash scripts/vm_preparar_commit.sh outputs/vm_oe3_oe4
#   git commit -F outputs/vm_oe3_oe4/MENSAJE_COMMIT.txt && git push
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DIR="${1:-outputs/vm_pendientes}"
DIR="${DIR%/}"
LIST="$DIR/ARCHIVOS_PARA_COMMIT.txt"
if [ ! -f "$LIST" ]; then
  echo "No existe $LIST: la corrida no llegó a escribir su resumen (regenéralo con el script vm_resumen_* que le corresponde)." >&2
  exit 1
fi

while IFS= read -r p; do
  [ -z "$p" ] && continue
  [ -e "$p" ] && git add -- "$p"
done < "$LIST"

# Ningún archivo grande debería colarse (el .gitignore excluye latentes, scores y cachés).
BIG=$(git diff --cached --name-only -z | xargs -0 -r du -k 2>/dev/null | awk '$1 > 20480 {print $2" ("int($1/1024)" MB)"}')
if [ -n "$BIG" ]; then
  echo "AVISO: hay archivos de más de 20 MB preparados para el commit:"
  echo "$BIG"
  echo "Revísalos antes de hacer commit (git restore --staged <ruta> para sacarlos)."
fi

echo ""
echo "Preparados: $(git diff --cached --name-only | wc -l) archivos, $(git diff --cached --name-only -z | xargs -0 -r du -ck 2>/dev/null | tail -1 | awk '{printf "%.1f MB", $1/1024}')"
git diff --cached --stat | tail -n 25
echo ""
echo "Si todo se ve bien:"
echo "  git commit -F $DIR/MENSAJE_COMMIT.txt && git push"
