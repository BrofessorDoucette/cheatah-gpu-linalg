#!/usr/bin/env bash
# bench_table_lint.sh — check this repo's published benchmark tables.
#
# Deliberately runs the CHEATAH repo's scripts/bench_table.purr rather than reimplementing the
# checks here. A second implementation is a second thing to keep in step, and the failure mode
# is silent: the two would drift until one of them accepted a table the other rejected, and
# nobody would notice which was right. CHEATAH_BENCH_DOCS points it at this repo's documents.
#
#   bash scripts/bench_table_lint.sh              # warns on staleness
#   bash scripts/bench_table_lint.sh check --strict   # staleness is fatal (release path)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

CHEATAH_ROOT="${CHEATAH_DIR:-$(cd .. && pwd)/cheatah}"
PURRC="$CHEATAH_ROOT/build/release/bin/purrc"
CHEATAH="$CHEATAH_ROOT/build/release/bin/cheatah"
LINT_SRC="$CHEATAH_ROOT/scripts/bench_table.purr"

# Skip rather than fail when the sibling is absent or unbuilt: this repo must stay buildable on
# its own, and a missing sibling is a checkout state, not a documentation defect.
if [ ! -x "$PURRC" ] || [ ! -x "$CHEATAH" ] || [ ! -f "$LINT_SRC" ]; then
    echo "[bench-table] SKIP — needs a built cheatah at $CHEATAH_ROOT (set CHEATAH_DIR)"
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

"$PURRC" "$LINT_SRC" -o "$TMP/bench_table.so" >/dev/null \
    || { echo "[bench-table] FAILED: purrc $LINT_SRC"; exit 1; }

CHEATAH_BENCH_DOCS="PERFORMANCE.md,README.md" "$CHEATAH" "$TMP/bench_table.so" "$@"
