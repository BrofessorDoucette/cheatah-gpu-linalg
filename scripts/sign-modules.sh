#!/usr/bin/env bash
# sign-modules.sh — write the .sha512 sidecars that mark cheatah-gpu-linalg's module headers as
# VERIFIED cheatah modules. This is what makes the extension biome-installable: with the sidecar
# present, purrc resolves `import gpulinalg` on the extension path (CHEATAH_MODULE_PATH, which
# `biome add cheatah-gpu-linalg` sets) — so a user with a standard cheatah install never touches
# git or --import-root. The sidecar is the SHA-512 hex of the header; purrc verifies the match.
#
# Only the top-level package header needs signing: `import gpulinalg` resolves the `gpulinalg`
# module (purr/gpulinalg/gpulinalg.hpp), which re-exports the whole surface. Re-run whenever
# that header changes; scripts/test-biome-install.sh fails when it is out of sync.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

for h in purr/gpulinalg/gpulinalg.hpp; do
    sha512sum "$h" | cut -d' ' -f1 > "$h.sha512"
    echo "signed: $h.sha512"
done
