#!/usr/bin/env bash
# run_purr_examples.sh — compile + run the PURE-CHEATAH GPU examples (examples/purr/*.purr)
# against the Vulkan backend. Mirrors cheatah-gpu's metal_gate purr flow: purrc from the sibling
# cheatah checkout, the Vulkan SDK's headers, this repo's include roots + shader dir, and the
# `import gpulinalg` module under purr/. Prints each program's output; exits non-zero if any
# fails to print PASS:. Skips cleanly (exit 0) when the toolchain isn't present.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
bold() { printf '\n\033[1m[purr-examples] %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31m[purr-examples] FAILED: %s\033[0m\n' "$*"; exit 1; }

CHEATAH_DIR="${CHEATAH_DIR:-$PWD/../cheatah}"
CHEATAH_GPU_DIR="${CHEATAH_GPU_DIR:-$PWD/../cheatah-gpu}"
PURRC=""; CHEATAH=""
for c in release debug asan; do
    [ -z "$PURRC" ]   && [ -x "$CHEATAH_DIR/build/$c/bin/purrc" ]   && PURRC="$CHEATAH_DIR/build/$c/bin/purrc"
    [ -z "$CHEATAH" ] && [ -x "$CHEATAH_DIR/build/$c/bin/cheatah" ] && CHEATAH="$CHEATAH_DIR/build/$c/bin/cheatah"
done
SDK_INC="$(ls -d "$HOME"/Tools/vulkan-sdk/*/x86_64/include "$HOME"/VulkanSDK/*/x86_64/include 2>/dev/null | sort -V | tail -1)"
SDK_LIB="$(ls -d "$HOME"/Tools/vulkan-sdk/*/x86_64/lib "$HOME"/VulkanSDK/*/x86_64/lib 2>/dev/null | sort -V | tail -1)"
SPV_DIR="$PWD/build/shaders"
if [ ! -x "$PURRC" ] || [ ! -x "$CHEATAH" ] || [ -z "$SDK_INC" ] || [ ! -d "$SPV_DIR" ]; then
    bold "Skipping (need cheatah toolchain, Vulkan SDK, and a built build/shaders — cmake --build build)."
    exit 0
fi

W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
for t in examples/purr/*.purr; do
    nm="$(basename "$t" .purr)"
    bold "── $nm ──"
    CPATH="$SDK_INC" "$PURRC" --import-root "$PWD/purr" "$t" -o "$W/$nm.so" \
        --cxxflag "-I$PWD/include" \
        --cxxflag "-I$CHEATAH_DIR/stdlib/linalg" \
        --cxxflag "-I$CHEATAH_DIR/stdlib/ndarray" \
        --cxxflag "-I$CHEATAH_GPU_DIR" \
        --cxxflag "-DCHEATAH_GPU_LINALG_VULKAN=1" \
        --cxxflag "-DCHEATAH_GPU_LINALG_SPV_DIR=\"$SPV_DIR\"" \
        --link "$SDK_LIB/libvulkan.so" \
        >"$W/$nm.log" 2>&1 || { sed 's/^/    /' "$W/$nm.log"; fail "compile $t"; }
    out="$("$CHEATAH" "$W/$nm.so" 2>&1)"; echo "$out" | sed 's/^/    /'
    echo "$out" | grep -q "PASS:" || fail "$t did not pass"
done
bold "All purr examples PASSED."
exit 0
