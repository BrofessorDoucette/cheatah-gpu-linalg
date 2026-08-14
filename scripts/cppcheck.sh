#!/usr/bin/env bash
# Static analysis: run cppcheck across cheatah-gpu-linalg's C++ for PERFORMANCE and SECURITY problems.
# Part of the QA gate; also runnable on its own:
#
#     bash scripts/cppcheck.sh
#
# A real finding exits non-zero (so the gate fails); `// cppcheck-suppress <id>` inline can annotate
# an intentional exception.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

command -v cppcheck >/dev/null 2>&1 || {
    echo "cppcheck not found — install it (e.g. 'sudo apt install cppcheck')."
    exit 1
}

cppcheck \
    --enable=warning,performance,portability \
    --std=c++20 --language=c++ \
    --library=googletest \
    --inline-suppr \
    --suppress=missingInclude \
    --suppress=missingIncludeSystem \
    --suppress=unusedFunction \
    --suppress=checkersReport \
    --suppress=unmatchedSuppression \
    --error-exitcode=1 \
    -q \
    -j "$(nproc)" \
    -i build \
    -i build -i build-asan -i build-bench \
     \
     \
     \
     \
    include tests bench examples \
    || { echo "[cppcheck] performance/security findings above — fix them or annotate with // cppcheck-suppress"; exit 1; }

echo "[cppcheck] clean — no performance/security findings."
