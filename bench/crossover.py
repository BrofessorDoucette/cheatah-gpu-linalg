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

BENCH = sys.argv[1] if len(sys.argv) > 1 else "build-bench/bench/gpu_linalg_bench_vk"

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

print("| op | size | crossover (device wins at) | host @ crossover | device @ crossover |")
print("|---|---|---|---|---|")
for op, (h, d, label) in PAIRS.items():
    host, dev = runs.get(h, {}), runs.get(d, {})
    common = sorted(set(host) & set(dev))
    if not common:
        print(f"| {op} | {label} | (no overlapping sizes) | — | — |")
        continue
    cross = next((n for n in common if dev[n] < host[n]), None)
    if cross is None:
        print(f"| {op} | {label} | host wins through {common[-1]:,} | — | — |")
    else:
        print(f"| {op} | {label} | **n ≥ {cross:,}** | {host[cross]:.0f} µs | {dev[cross]:.0f} µs |")
# Sanity: monotone verdict — once the device wins it should keep winning.
for op, (h, d, _) in PAIRS.items():
    host, dev = runs.get(h, {}), runs.get(d, {})
    common = sorted(set(host) & set(dev))
    won = False
    for n in common:
        if dev[n] < host[n]:
            won = True
        elif won:
            print(f"<!-- WARNING: {op} verdict not monotone at n={n:,} -->")
            break
