#!/usr/bin/env bash
# setup-hooks.sh — point git at the version-controlled hooks in .githooks/.
# Run once per clone (CMake configure also does this automatically). Idempotent.
set -eu
REPO_ROOT="$(git rev-parse --show-toplevel)"
git -C "$REPO_ROOT" config core.hooksPath .githooks
chmod +x "$REPO_ROOT/.githooks/"* "$REPO_ROOT/scripts/"*.sh 2>/dev/null || true
echo "hooks: core.hooksPath -> .githooks (pre-push now runs scripts/qa.sh)"
