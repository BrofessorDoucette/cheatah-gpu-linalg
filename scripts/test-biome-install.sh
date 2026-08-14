#!/usr/bin/env bash
# test-biome-install.sh — sandbox the EXACT experience of someone with a standard cheatah install
# who runs `biome add cheatah-gpu-linalg`. Nothing from this working tree (build/, .git, dev env
# vars) is allowed to make it falsely pass: we copy ONLY the installable purr package to a
# throwaway location (as biome fetches it), compile a fresh user project against it with cheatah
# env vars CLEARED, and the only module wiring is the extension on CHEATAH_MODULE_PATH —
# precisely what biome's EXTENSIONS support sets. The native side (this repo's headers, the
# sibling cheatah-gpu surface, the built SPIR-V kernels, the Vulkan SDK) rides on compiler flags
# exactly as the extension's CMake integration passes them to a consumer build.
#
# If the gpulinalg.hpp.sha512 sidecar, module layout, or namespaces are wrong, this fails
# exactly as a real user's install would.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
REPO="$PWD"
CHEATAH_DIR="${CHEATAH_DIR:-$PWD/../cheatah}"
CHEATAH_GPU_DIR="${CHEATAH_GPU_DIR:-$PWD/../cheatah-gpu}"

# The user's installed toolchain (purrc/cheatah). Its stdlib root is BAKED into the binary, so
# io/ndarray/linalg resolve without any CHEATAH_ROOT env — just like a real install.
find_tool() {
    local n="$1"
    for c in release debug asan; do
        [ -x "$CHEATAH_DIR/build/$c/bin/$n" ] && { echo "$CHEATAH_DIR/build/$c/bin/$n"; return 0; }
    done
    command -v "$n" 2>/dev/null
}
PURRC="$(find_tool purrc)"; CHEATAH="$(find_tool cheatah)"
[ -x "$PURRC" ] && [ -x "$CHEATAH" ] || { echo "biome-install: no cheatah toolchain (set CHEATAH_DIR)"; exit 2; }
SDK_INC="$(ls -d "$HOME"/Tools/vulkan-sdk/*/x86_64/include "$HOME"/VulkanSDK/*/x86_64/include 2>/dev/null | sort -V | tail -1)"
SDK_LIB="$(ls -d "$HOME"/Tools/vulkan-sdk/*/x86_64/lib "$HOME"/VulkanSDK/*/x86_64/lib 2>/dev/null | sort -V | tail -1)"
SPV_DIR="$PWD/build/shaders"
[ -n "$SDK_INC" ] && [ -d "$SPV_DIR" ] \
    || { echo "biome-install: need the Vulkan SDK and a built build/shaders (cmake --build build)"; exit 2; }

# 1. Simulate biome fetching cheatah-gpu-linalg to a fresh dir OUTSIDE this tree. A consumer
#    gets the `gpulinalg/` package (header + the .sha512 sidecar) — copy exactly that.
INSTALL="$(mktemp -d)"; PROJ="$(mktemp -d)"
trap 'rm -rf "$INSTALL" "$PROJ"' EXIT
cp -r purr/gpulinalg "$INSTALL/gpulinalg"
[ -f "$INSTALL/gpulinalg/gpulinalg.hpp.sha512" ] \
    || { echo "biome-install: the fetched copy has no gpulinalg.hpp.sha512 — run scripts/sign-modules.sh"; exit 1; }

# 2. A brand-new user project that just imports the package and does array math on the device.
mkdir -p "$PROJ/src"
cat > "$PROJ/src/main.purr" <<'PURR'
# Seconds after `biome add cheatah-gpu-linalg`. This user never saw a git repo.
import io
import ndarray as nd
import gpulinalg as gpu

let xs: list<float> = []
for i in range(1000) {
    xs.append(0.001 * i)
}
let dx = gpu.to_device(nd.array(xs))
let s = gpu.sum(dx)
let d = s - 499.5
if d * d < 0.0001 {
    io.print("RESULT: PASS")
} else {
    io.print("RESULT: FAIL", s)
}
PURR

# 3. Compile from the user's project with a CLEAN environment: every cheatah env var cleared;
#    module resolution comes ONLY from CHEATAH_MODULE_PATH -> the fetched package, and the
#    native flags are the ones the extension's CMake integration hands a consumer target.
clean_env=(env -u CHEATAH_ROOT -u CHEATAH_LIB_DIR -u CHEATAH_TRUST -u CHEATAH_DIR
           CHEATAH_MODULE_PATH="$INSTALL" CPATH="$SDK_INC")
if ! out="$(cd "$PROJ" && "${clean_env[@]}" "$PURRC" src/main.purr -o app.so \
        --cxxflag "-I$REPO/include" \
        --cxxflag "-I$CHEATAH_DIR/stdlib/linalg" \
        --cxxflag "-I$CHEATAH_DIR/stdlib/ndarray" \
        --cxxflag "-I$CHEATAH_GPU_DIR" \
        --cxxflag "-DCHEATAH_GPU_LINALG_VULKAN=1" \
        --cxxflag "-DCHEATAH_GPU_LINALG_SPV_DIR=\"$SPV_DIR\"" \
        --link "$SDK_LIB/libvulkan.so" 2>&1)"; then
    echo "biome-install: FAILED — a fresh user could not compile 'import gpulinalg':"
    echo "$out" | sed 's/^/    /'
    exit 1
fi
run="$("$CHEATAH" "$PROJ/app.so" 2>&1)"
echo "$run" | sed 's/^/    /'
echo "$run" | grep -q "RESULT: PASS" \
    || { echo "biome-install: FAILED — the user program did not pass"; exit 1; }
echo "biome-install: PASS — fresh project, package fetched to a throwaway dir on CHEATAH_MODULE_PATH, clean env."
