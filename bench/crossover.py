#!/usr/bin/env python3
"""Host-vs-device crossover table — THE artifact the consuming application's CPU/GPU
dispatcher feeds on.

Runs gpu_linalg_bench_vk over paired host/device sweeps (same op, same element type: f64),
joins on size, and prints a Markdown table with the smallest size where the device path's
wall time beats the host path. Usage:

    bench/crossover.py [path-to-gpu_linalg_bench_vk]
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp

# --out is pulled out FIRST: this script takes the benchmark binary as a bare positional, so
# leaving the flag in argv made it the binary path.
OUT = None
_args = sys.argv[1:]
if "--out" in _args:
    _i = _args.index("--out")
    if _i + 1 >= len(_args):
        sys.exit("crossover: --out needs a path")
    OUT = _args[_i + 1]
    del _args[_i:_i + 2]
BENCH = _args[0] if _args else "build-bench/bench/gpu_linalg_bench_vk"

# op -> (host bench, device bench, size label)
PAIRS = {
    "matmul (double)": ("BM_matmul_host", "BM_matmul_f64_dev", "n (matrix is n x n)"),
    "dot (double)": ("BM_dot_host", "BM_dot_f64_dev", "n (elements)"),
    "sum (double)": ("BM_sum_host", "BM_sum_f64_dev", "n (elements)"),
    "add (double)": ("BM_add_host", "BM_add_f64_dev", "n (elements)"),
    "axpy (double)": ("BM_axpy_host", "BM_axpy_fused_f64_dev", "n (elements)"),
}

names = "|".join({n for pair in PAIRS.values() for n in pair[:2]})
out = subprocess.run(
    [BENCH, f"--benchmark_filter=^({names})/", "--benchmark_format=json",
     "--benchmark_min_time=0.2s"],
    check=True, capture_output=True, text=True).stdout
runs = {}
for b in json.loads(out)["benchmarks"]:
    if b.get("run_type") != "iteration":
        continue
    name = b["run_name"]                      # e.g. BM_dot_host/16384/real_time
    parts = name.split("/")
    runs.setdefault(parts[0], {})[int(parts[1])] = b["real_time"]  # microseconds

# Collect as well as print, so the published artifact and the console output cannot disagree
# about what was measured.
LINES = ["| op | size | crossover (device wins at) | host @ crossover | device @ crossover |",
         "|---|---|---|---|---|"]


def emit(line):
    LINES.append(line)
    print(line)


print(LINES[0])
print(LINES[1])
for op, (h, d, label) in PAIRS.items():
    host, dev = runs.get(h, {}), runs.get(d, {})
    common = sorted(set(host) & set(dev))
    if not common:
        emit(f"| {op} | {label} | (no overlapping sizes) | — | — |")
        continue
    cross = next((n for n in common if dev[n] < host[n]), None)
    if cross is None:
        emit(f"| {op} | {label} | host wins through {common[-1]:,} | — | — |")
    else:
        emit(f"| {op} | {label} | **n ≥ {cross:,}** | {host[cross]:.0f} µs | {dev[cross]:.0f} µs |")
# Sanity: monotone verdict — once the device wins it should keep winning.
for op, (h, d, _) in PAIRS.items():
    host, dev = runs.get(h, {}), runs.get(d, {})
    common = sorted(set(host) & set(dev))
    won = False
    for n in common:
        if dev[n] < host[n]:
            won = True
        elif won:
            emit(f"<!-- WARNING: {op} verdict not monotone at n={n:,} -->")
            break

# Write the stamped region body. An artifact or it is not publishable — see
# ../cheatah/scripts/bench_table.purr for what reads these fields.
if OUT:
    st = stamp.build(
        suite="gpu-host-device-crossover",
        watch="include/, kernels/, bench/crossover.py",
        competitors="none — cheatah host arrays against cheatah device arrays",
        harness="paired host/device sweeps, same element type; medians over interleaved reps",
        statistic="median real_time; crossover is the first size where the device wins",
        produced_by=f"python3 bench/crossover.py --out {OUT}")
    stamp.write_region(OUT, st, "\n".join(LINES))
