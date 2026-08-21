#!/usr/bin/env bash
# gpu_bench_gate.sh — the performance RATCHET (cheatah's compiler_bench_gate.sh pattern): a
# committed same-machine baseline of the key device benchmarks; `gate` fails only on a CONFIRMED
# regression (threshold × floor × re-measure), `update` moves the baseline — a deliberate,
# reviewed step after a named optimization lands, so the ratchet only ever moves faster.
#
#   bench/gpu_bench_gate.sh gate     # compare against bench/gpu_bench_baseline.csv (SKIP if none)
#   bench/gpu_bench_gate.sh report   # print current-vs-baseline, never fail
#   bench/gpu_bench_gate.sh update   # rewrite the baseline from a fresh run
#
# Same-machine dev tool, NOT a push gate (numbers are hardware-specific). Runs the Vulkan binary
# on the default (best) device.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
MODE="${1:-gate}"
BASE="bench/gpu_bench_baseline.csv"
BIN="build-bench/bench/gpu_linalg_bench_vk"
THRESHOLD="${THRESHOLD:-1.10}"   # fail only above 1.10× baseline …
MIN_GAP_US="${MIN_GAP_US:-15}"   # … AND slower by ≥ 15 µs (absolute floor vs timer noise)
FILTER='BM_matmul_f32_dev/(256|1024|4096)/|BM_matmul_f64_dev/1024/|BM_dot_f32_dev/16777216/|BM_sum_f32_dev/16777216/|BM_add_f32_dev/16777216/|BM_axpy_fused_f32_dev/16777216/|BM_dispatch_overhead'

bold() { printf '\n\033[1m[gpu-bench-gate] %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31m[gpu-bench-gate] FAILED: %s\033[0m\n' "$*"; exit 1; }

[ -x "$BIN" ] || fail "no $BIN — configure with -DCHEATAH_GPU_LINALG_BENCH=ON (build-bench/)"

# -> "name,us" lines on stdout, where us really IS a median.
#
# This used to run ONE repetition and read `real_time`, while writing a baseline file whose
# header called the column `median_us`. It was not a median of anything — it was a single
# sample. Now it takes repetitions and reads the median aggregate Google Benchmark computes,
# with --benchmark_enable_random_interleaving so the repetitions of the different sizes are
# scattered through the run rather than measured as consecutive blocks (GPU clocks ramp and
# throttle over a run, so a block ordering biases whichever size runs last).
#
# Aggregate rows arrive named "<case>_median"; the suffix is stripped so the baseline keys
# stay exactly what they were and an existing baseline file still matches.
measure() {
    "$BIN" --benchmark_filter="$FILTER" --benchmark_min_time=0.15s \
        --benchmark_repetitions=5 --benchmark_enable_random_interleaving=true \
        --benchmark_report_aggregates_only=true --benchmark_format=json 2>/dev/null |
        python3 -c 'import json, sys
for b in json.load(sys.stdin)["benchmarks"]:
    if b.get("aggregate_name") != "median":
        continue
    name = b["name"]
    if name.endswith("_median"):
        name = name[: -len("_median")]
    print(name + "," + format(b["real_time"], ".2f"))'
}

if [ "$MODE" = "update" ]; then
    bold "Measuring a fresh baseline…"
    { echo "# name,median_us — median over 5 interleaved repetitions (--benchmark_repetitions=5";
      echo "# --benchmark_enable_random_interleaving=true). Regenerate with:";
      echo "#   bench/gpu_bench_gate.sh update      (same machine only — these are machine-specific)";
      measure; } > "$BASE"
    bold "Baseline updated: $BASE"
    exit 0
fi

[ -f "$BASE" ] || { bold "No baseline ($BASE) — SKIP (run 'update' to create one)."; exit 0; }

bold "Measuring current numbers…"
CUR="$(measure)"
STATUS=0
while IFS=, read -r name base_us; do
    case "$name" in \#*|"") continue ;; esac
    cur_us="$(printf '%s\n' "$CUR" | awk -F, -v n="$name" '$1==n{print $2}')"
    [ -n "$cur_us" ] || { printf '  %-45s MISSING from current run\n' "$name"; STATUS=1; continue; }
    slower="$(python3 -c "print(1 if $cur_us > $base_us*$THRESHOLD and $cur_us-$base_us > $MIN_GAP_US else 0)")"
    if [ "$slower" = "1" ]; then
        # CONFIRMATION: re-measure the suspect alone before declaring a regression.
        confirm="$("$BIN" --benchmark_filter="^${name%%/*}" --benchmark_min_time=0.4s \
                   --benchmark_repetitions=9 --benchmark_enable_random_interleaving=true \
                   --benchmark_report_aggregates_only=true --benchmark_format=json 2>/dev/null |
                   python3 -c "
import json, sys
for b in json.load(sys.stdin)['benchmarks']:
    if b.get('aggregate_name') != 'median':
        continue
    n = b['name']
    if n.endswith('_median'):
        n = n[: -len('_median')]
    if n == '$name':
        print(f\"{b['real_time']:.2f}\")" )"
        [ -n "$confirm" ] && cur_us="$confirm"
        slower="$(python3 -c "print(1 if $cur_us > $base_us*$THRESHOLD and $cur_us-$base_us > $MIN_GAP_US else 0)")"
    fi
    if [ "$slower" = "1" ]; then
        printf '  \033[31m%-45s %10s µs vs baseline %10s µs — REGRESSION\033[0m\n' "$name" "$cur_us" "$base_us"
        STATUS=1
    else
        printf '  %-45s %10s µs vs baseline %10s µs\n' "$name" "$cur_us" "$base_us"
    fi
done < "$BASE"

if [ "$MODE" = "report" ]; then exit 0; fi
[ "$STATUS" -eq 0 ] || fail "confirmed regression(s) above — investigate, or 'update' after a deliberate trade-off"
bold "No regressions — the ratchet holds."
exit 0
