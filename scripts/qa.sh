#!/usr/bin/env bash
# qa.sh — the cheatah-gpu-linalg QA gate, matching the household standard (cheatah / cheatah-gpu):
#
#   1. Plain build + ctest: every op on BOTH backends (Vulkan on the best real device + the
#      emulated Metal flow), plus a forced-llvmpipe Vulkan pass so the software rasterizer stays
#      covered no matter what GPU the machine has.
#   2. ASan + UBSan build + ctest (leak detection scoped to our code: the vk tests disable it via
#      their ctest ENVIRONMENT — vendor driver blobs hold process-lifetime caches).
#   3. Valgrind memcheck over the emulated-Metal test binaries (pure our-code + emulator — the
#      meaningful leak surface; the Vulkan binaries would drown in driver noise).
#   4. The 100%-Javadoc documentation-coverage gate (scripts/doc_coverage.sh): every public
#      entity, parameter and return value must be documented, or the gate fails.
#   5. Unit coverage (hard gate): clang source-based coverage over the backend-neutral headers
#      must be 100% lines AND functions; the README coverage table must be committed in sync.
#   6. cppcheck (hard gate): performance + security static analysis.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
bold() { printf '\n\033[1m[gpu-linalg-qa] %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31m[gpu-linalg-qa] FAILED: %s\033[0m\n' "$*"; exit 1; }

# 1. Plain build + both-backend tests ---------------------------------------------------------
bold "Configuring + building (both backends)…"
cmake -S . -B build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >/tmp/gpu_linalg_cfg.log 2>&1 || { tail -20 /tmp/gpu_linalg_cfg.log; fail "configure"; }
cmake --build build -j >/tmp/gpu_linalg_build.log 2>&1 || { tail -30 /tmp/gpu_linalg_build.log; fail "build"; }
bold "Running every op on both backends…"
ctest --test-dir build --output-on-failure || fail "tests"

bold "Re-running the Vulkan tests on llvmpipe (software rasterizer)…"
CHEATAH_GPU_LINALG_VK_DEVICE=llvmpipe ctest --test-dir build -R ':vk(:|$)' --output-on-failure \
    || fail "llvmpipe tests"

# 2. ASan + UBSan ----------------------------------------------------------------------------
bold "Building + running under ASan + UBSan…"
cmake -S . -B build-asan -G Ninja -DCHEATAH_GPU_LINALG_SANITIZE=ON \
    >/tmp/gpu_linalg_asan_cfg.log 2>&1 || { tail -20 /tmp/gpu_linalg_asan_cfg.log; fail "ASan configure"; }
cmake --build build-asan -j >/tmp/gpu_linalg_asan_build.log 2>&1 \
    || { tail -30 /tmp/gpu_linalg_asan_build.log; fail "ASan build"; }
ASAN_OPTIONS=detect_leaks=1 ctest --test-dir build-asan --output-on-failure || fail "ASan tests"

# 3. Valgrind over the emulated-Metal binaries ------------------------------------------------
if command -v valgrind >/dev/null 2>&1; then
    bold "Valgrind memcheck over the emulated-Metal test binaries…"
    for exe in build/gpu_linalg_*_mtl; do
        valgrind --error-exitcode=1 --leak-check=full --errors-for-leak-kinds=definite,indirect \
            "$exe" >/tmp/gpu_linalg_vg.log 2>&1 \
            || { sed 's/^/    /' /tmp/gpu_linalg_vg.log; fail "valgrind $exe"; }
    done
else
    bold "valgrind not found — skipping memcheck (ASan still ran)."
fi

# 4. Documentation coverage: the 100%-Javadoc hard gate --------------------------------------
bold "Checking documentation coverage (100% Javadoc)…"
bash scripts/doc_coverage.sh || fail "doc coverage"

# Published benchmark tables must be GENERATED, not typed, and must say when they went stale.
# Pure text work — it measures nothing — so it belongs beside the other doc gates.
bold "Checking benchmark-table provenance…"
bash scripts/bench_table_lint.sh || fail "benchmark tables"
# Private-reference scan — this repo is private TODAY, but it must be publishable on
# demand (it joins a Biome Standard when it goes public), so the guard runs here too.
bash scripts/check_no_private_refs.sh || fail "a private-project reference is in the tree"

# 5. Unit coverage: refresh the README table, fail if it drifted, hard-fail unless 100% -------
bold "Measuring unit-test coverage (clang source-based) + refreshing the README table…"
bash scripts/coverage.sh update-readme >/tmp/gpu_linalg_coverage.log 2>&1 \
    || { tail -30 /tmp/gpu_linalg_coverage.log; fail "coverage report"; }
if ! git diff --quiet -- README.md; then
    printf '\n[gpu-linalg-qa] The README coverage table is out of date. Updated it to:\n\n'
    git --no-pager diff -- README.md | sed -n '/coverage:start/,/coverage:end/p'
    fail "README coverage table changed — 'git add README.md && git commit', then run again"
fi
cat /tmp/gpu_linalg_coverage.log
covnums=$(sed -n 's/.*lines [0-9.]*% (\([0-9]*\)\/\([0-9]*\)), functions [0-9.]*% (\([0-9]*\)\/\([0-9]*\)).*/\1 \2 \3 \4/p' /tmp/gpu_linalg_coverage.log)
[ -n "$covnums" ] || fail "could not parse the coverage summary (coverage.sh output changed?)"
read -r lcov_n lcov_d fcov_n fcov_d <<<"$covnums"
if [ "$lcov_n" != "$lcov_d" ] || [ "$fcov_n" != "$fcov_d" ]; then
    fail "unit-test coverage below 100% — lines $lcov_n/$lcov_d, functions $fcov_n/$fcov_d (find gaps with: scripts/coverage.sh show <file>)"
fi
bold "Unit-test coverage: 100% lines ($lcov_n/$lcov_d) + functions ($fcov_n/$fcov_d)."

# 6. cppcheck: performance + security static analysis ----------------------------------------
bold "Running cppcheck (performance + security)…"
bash scripts/cppcheck.sh || fail "cppcheck"

# 6b. clang-tidy: the broad check set from the committed .clang-tidy; cert-* always fatal -----
#     Self-contained ON PURPOSE: this repo must stay publishable, so no sibling-checkout
#     dependency. The MECHANICS below mirror cheatah/scripts/clang_tidy.sh (the canonical
#     driver — the one sanctioned duplicate); the POLICY file .clang-tidy is a verbatim copy
#     of cheatah's, kept honest by deploy/scripts/tidy_fleet.sh. TIDY_WERROR is the per-repo
#     burn-down ratchet: categories fatal beyond the unconditional cert-* floor.
#     The allowlist keeps this repo's OWN code: the build also compiles sibling sources
#     (cheatah stdlib, cheatah-gpu), which are linted in their own repos' gates.
bold "Running clang-tidy (cert fatal; ratchet: ${TIDY_WERROR:-cert-* only})…"
command -v clang-tidy >/dev/null 2>&1 || fail "clang-tidy not installed (apt install clang-tidy)"
command -v run-clang-tidy >/dev/null 2>&1 || fail "run-clang-tidy not installed (ships with clang-tidy)"
[ -f build/compile_commands.json ] || fail "no compile_commands.json in build/ (the configure above exports it)"
_tidy_werror="cert-*${TIDY_WERROR:+,$TIDY_WERROR}"
# Canary first: prove this config CAN fail before trusting a green run — a stage that
# cannot fail certifies nothing.
_tidy_tmp="$(mktemp -d)"
printf '#include <cstdlib>\nint main() { return std::system("ls"); }\n' > "$_tidy_tmp/canary.cpp"
if clang-tidy --quiet --config-file=.clang-tidy --warnings-as-errors="$_tidy_werror" \
        "$_tidy_tmp/canary.cpp" -- -std=c++20 >/dev/null 2>&1; then
    rm -rf "$_tidy_tmp"
    fail "clang-tidy CANARY passed clean — the config or tool is broken; nothing can be certified"
fi
rm -rf "$_tidy_tmp"
_tidy_src="^$PWD/(tests|examples)/"
_tidy_hdr="^$PWD/(include|kernels|purr|tests|examples)/"
run-clang-tidy -p build -quiet -j "$(nproc)" -header-filter "$_tidy_hdr" \
    -warnings-as-errors "$_tidy_werror" "$_tidy_src" >/tmp/gpu_linalg_tidy.log 2>&1 \
    || { grep ' error: ' /tmp/gpu_linalg_tidy.log | sort -u | head -30; fail "clang-tidy (full log: /tmp/gpu_linalg_tidy.log)"; }
bold "clang-tidy: WErrors=$_tidy_werror errors=0 warnings-outstanding=$(grep ' warning: ' /tmp/gpu_linalg_tidy.log | sort -u | grep -c . || true)"

# 7. Performance ratchet (report mode — informative; `bench/gpu_bench_gate.sh gate` is the
#    enforcing form, run manually since numbers are same-machine) -----------------------------
if [ -x build-bench/bench/gpu_linalg_bench_vk ]; then
    bash bench/gpu_bench_gate.sh report || true
else
    bold "bench binaries not built — skipping perf report (cmake -B build-bench -DCHEATAH_GPU_LINALG_BENCH=ON)."
fi

bold "cheatah-gpu-linalg QA gate PASSED."
exit 0
