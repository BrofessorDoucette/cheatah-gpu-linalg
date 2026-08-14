#!/usr/bin/env bash
# Javadoc / Doxygen DOCUMENTATION-COVERAGE gate for cheatah-gpu-linalg's public API.
#
# Runs Doxygen in a strict mode — EXTRACT_ALL=NO so nothing is auto-extracted,
# WARN_IF_UNDOCUMENTED + WARN_NO_PARAMDOC so every type, function, member, parameter and return
# value must be documented — and FAILS (exit 1) if any public entity lacks documentation. This is
# the "100% Javadoc" hard gate the QA gate enforces before every push.
#
#   scripts/doc_coverage.sh          # check; exit 1 if anything is undocumented
#   scripts/doc_coverage.sh report   # same, but always print the full doxygen log
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

DOXYGEN="${DOXYGEN:-doxygen}"
command -v "$DOXYGEN" >/dev/null 2>&1 || DOXYGEN="$HOME/Tools/doxygen-1.16.1/bin/doxygen"
command -v "$DOXYGEN" >/dev/null 2>&1 || { echo "doc-coverage: doxygen not found (install it, or set \$DOXYGEN)"; exit 1; }

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

( cat Doxyfile
  echo "EXTRACT_ALL=NO"
  echo "WARN_IF_UNDOCUMENTED=YES"
  echo "WARN_NO_PARAMDOC=YES"
  echo "GENERATE_HTML=NO"
  echo "GENERATE_XML=NO"
  echo "GENERATE_LATEX=NO"
  echo "HAVE_DOT=NO"
  echo "QUIET=YES"
) | "$DOXYGEN" - 2>"$log" >/dev/null

[ "${1:-}" = "report" ] && { echo "--- doxygen strict log ---"; cat "$log"; echo "---"; }

n="$(grep -cE 'not documented' "$log" || true)"
if [ "$n" -ne 0 ]; then
    echo "doc-coverage: FAIL — $n undocumented public entit$([ "$n" -eq 1 ] && echo y || echo ies):"
    grep -E 'not documented' "$log"
    echo
    echo "doc-coverage: document the above (brief + @param/@return) and re-run; the gate requires 100%."
    exit 1
fi
echo "doc-coverage: 100% — every public gpu-linalg entity (types, functions, params, returns) is documented."
exit 0
