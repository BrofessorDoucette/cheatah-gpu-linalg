#!/usr/bin/env bash
# Measure cheatah-gpu-linalg test coverage with clang source-based coverage, over the
# backend-neutral surface (container, routines, elementwise, factories, bridge, conv,
# kernel stand-ins, context switch). The two backend bring-up headers
# (vulkan_context.hpp / metal_context.hpp) are excluded exactly the way ../cheatah-gpu
# excludes gpu/vulkan + gpu/metal from ITS host gate: they need real driver matrices and
# graduate into the denominator as they gain dedicated tests. The emulated-Metal test lane
# runs every kernel and every neutral line headlessly, so this gate is honest and hard.
#
#   scripts/coverage.sh               # per-file summary report
#   scripts/coverage.sh show <file>   # uncovered lines of one file
#   scripts/coverage.sh funcs <file>  # per-function coverage of one file
#   scripts/coverage.sh update-readme # rewrite the coverage table in README.md
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

B=build/cov
cmake -S . -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping" \
  -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate -fcoverage-mapping" >/tmp/cgl_cov_cfg.log 2>&1 \
  || { tail -15 /tmp/cgl_cov_cfg.log; exit 1; }
cmake --build "$B" >/tmp/cgl_cov_build.log 2>&1 \
  || { tail -25 /tmp/cgl_cov_build.log; exit 1; }

# ONE combined binary (Metal-emulated lane) = true llvm-cov union semantics; the Vulkan-lane
# guards binary adds the probe's failure path, which the always-available emulated lane
# cannot reach. Both run every test they contain, so no zero-count records pollute the union.
( cd "$B"
  rm -f cov_*.profraw
  LLVM_PROFILE_FILE="cov_0.profraw" ./gpu_linalg_all_mtl >/dev/null 2>&1
  LLVM_PROFILE_FILE="cov_1.profraw" ./gpu_linalg_guards_vk >/dev/null 2>&1
  llvm-profdata merge -sparse cov_*.profraw -o merged.profdata )

OBJS=("$B"/gpu_linalg_all_mtl -object "$B"/gpu_linalg_guards_vk)
PROF="-instr-profile=$B/merged.profdata"
SRCS=$(git ls-files 'include/cheatah_gpu_linalg/*.hpp' 'purr/gpulinalg/*.hpp' \
       | grep -vE 'vulkan_context\.hpp|metal_context\.hpp')

case "${1:-report}" in
    show)  llvm-cov show   "${OBJS[@]}" $PROF "${2:?usage: coverage.sh show <file>}" 2>/dev/null \
             | grep -nE '\|[[:space:]]*0\|' || echo "all lines covered in ${2}" ;;
    funcs) llvm-cov report "${OBJS[@]}" $PROF -show-functions "${2:?usage: coverage.sh funcs <file>}" 2>/dev/null ;;
    update-readme)
        read -r reg mreg rcov fun mfun fexec lines mlin lcov br mbr bcov < <(
            llvm-cov report "${OBJS[@]}" $PROF $SRCS 2>/dev/null \
              | awk '$1=="TOTAL"{print $2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13}')
        : "${lines:=0}" "${mlin:=0}" "${fun:=0}" "${mfun:=0}"
        lcn=$((lines - mlin)); fcn=$((fun - mfun))
        tbl="$(mktemp)"
        {
            echo "<!-- coverage:start -->"
            echo "| Metric | neutral surface |"
            echo "|--------|-----------------|"
            echo "| **Lines** | $lcov ($lcn/$lines) |"
            echo "| **Functions** | $fexec ($fcn/$fun) |"
            echo "| Regions | $rcov |"
            echo "| Branches | $bcov |"
            echo "<!-- coverage:end -->"
        } > "$tbl"
        awk -v tf="$tbl" '
            /<!-- coverage:start -->/ { while ((getline l < tf) > 0) print l; close(tf); skip=1; next }
            /<!-- coverage:end -->/   { skip=0; next }
            !skip { print }
        ' README.md > README.md.new && mv README.md.new README.md
        rm -f "$tbl"
        echo "README coverage table: lines $lcov ($lcn/$lines), functions $fexec ($fcn/$fun)"
        ;;
    *)     llvm-cov report "${OBJS[@]}" $PROF $SRCS 2>/dev/null ;;
esac
