#!/usr/bin/env python3
"""Generate the device-resident throughput table PERFORMANCE.md publishes.

The binary has no bench_main.cpp (that lives in the cheatah repo), so the layout work happens
here, reading the Google Benchmark JSON. Everything it publishes is a median over interleaved
repetitions — see bench/stamp.py's gb_medians for why interleaving matters more on a GPU than
on a CPU.

    python3 bench/emit_tables.py --out docs/bench/gpu-device-throughput.md
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stamp  # noqa: E402

BIN = "build/bench/gpu_linalg_bench_vk"

# case -> (display, how to turn microseconds into a rate). A row exists here or it does not
# get published; adding a benchmark without a row shows up as a missing row, not a wrong one.
#
# FLOP counts are the standard 2*M*N*K for a GEMM; byte counts are what the kernel must move
# (read + write) for the elementwise and reduction ops, which is what makes the GB/s figure
# comparable against the measured ~375 GB/s stream ceiling rather than a made-up one.
ROWS = [
    ("BM_matmul_f32_dev/1024/real_time",     "matmul",                          "1024³",  lambda us: f"{2 * 1024**3 / (us * 1e-6) / 1e12:.2f} TFLOP/s"),
    ("BM_matmul_f32_dev/2048/real_time",     "matmul",                          "2048³",  lambda us: f"{2 * 2048**3 / (us * 1e-6) / 1e12:.2f} TFLOP/s"),
    ("BM_matmul_f32_dev/4096/real_time",     "matmul",                          "4096³",  lambda us: f"**{2 * 4096**3 / (us * 1e-6) / 1e12:.2f} TFLOP/s**"),
    ("BM_matmul_f16coop_dev/4096/real_time", "matmul f16 (tensor cores, opt-in)", "4096³", lambda us: f"**{2 * 4096**3 / (us * 1e-6) / 1e12:.2f} TFLOP/s**"),
    ("BM_matmul_f64_dev/1024/real_time",     "matmul f64",                      "1024³",  lambda us: f"{2 * 1024**3 / (us * 1e-6) / 1e9:.0f} GFLOP/s"),
    ("BM_dot_f32_dev/16777216/real_time",    "dot",                             "16M",    lambda us: f"{16777216 * 4 * 2 / (us * 1e-6) / 1e9:.0f} GB/s"),
    ("BM_sum_f32_dev/16777216/real_time",    "sum",                             "16M",    lambda us: f"{16777216 * 4 / (us * 1e-6) / 1e9:.0f} GB/s"),
    ("BM_add_f32_dev/16777216/real_time",    "add",                             "16M",    lambda us: f"{16777216 * 4 * 3 / (us * 1e-6) / 1e9:.0f} GB/s"),
    ("BM_axpy_fused_f32_dev/16777216/real_time", "axpy (fused)",                "16M",    lambda us: f"{16777216 * 4 * 3 / (us * 1e-6) / 1e9:.0f} GB/s"),
]


def fmt_time(us):
    if us >= 1000.0:
        return f"{us / 1000.0:.2f} ms"
    return f"{us:.0f} µs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    filt = "^BM_(matmul_f32_dev|matmul_f16coop_dev|matmul_f64_dev|dot_f32_dev|sum_f32_dev|add_f32_dev|axpy_fused_f32_dev)/"
    med = stamp.gb_medians(BIN, filt, reps=args.reps, min_time="0.2s")

    lines = ["| op | size | wall | rate |", "|---|---|---|---|"]
    missing = []
    for case, display, size, rate in ROWS:
        us = med.get(case)
        if us is None:
            missing.append(case)
            lines.append(f"| {display} | {size} | — | — (not measured) |")
            continue
        lines.append(f"| {display} | {size} | {fmt_time(us)} | {rate(us)} |")
    if missing:
        # Loudly, and by forcing publishable:false — a table with silent gaps reads as
        # complete, which is the failure mode this whole system exists to prevent.
        print("MISSING cases (table marked not publishable):", ", ".join(missing),
              file=sys.stderr)

    st = stamp.build(
        suite="gpu-device-throughput",
        watch="include/, kernels/, bench/gpu_linalg_bench.cpp",
        competitors="none — device-resident cheatah kernels against the hardware's own ceilings",
        harness=f"reps={args.reps}, min_time=0.2s, random-interleaving=on",
        statistic="median real_time per case; rates derived from the standard FLOP/byte counts",
        produced_by=f"python3 bench/emit_tables.py --out {args.out}",
        publishable="false" if missing else "true")
    stamp.write_region(args.out, st, "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
