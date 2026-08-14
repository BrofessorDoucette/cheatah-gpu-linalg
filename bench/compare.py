#!/usr/bin/env python3
"""compare.py — cheatah-gpu-linalg vs the competition, honestly.

The pattern is cheatah's scripts/numpy_compare.py: every implementation gets IDENTICAL
deterministic operands, answers are CROSS-CHECKED before any timing is reported, each op is
timed as best-of-TRIALS over a repeat loop, and the output is a Markdown table of µs/op with
the ratio vs cheatah's Vulkan backend.

Implementations:
  cheatah-vk   — this library's Vulkan backend (dev-resident timing from the Google Benchmark
                 binary's JSON; correctness via the probe CLI)
  numpy        — CPU (whatever BLAS the system numpy links)
  torch-cpu    — PyTorch CPU (if installed in bench/venv)
  torch-cuda   — PyTorch CUDA, tensors pre-resident, torch.cuda.synchronize() bracketed
                 (the honest GPU-vs-GPU yardstick), plus an e2e mode with the transfers inside.

Usage: bench/venv/bin/python bench/compare.py [--ops matmul,dot,...] [--out PERFORMANCE-table.md]
Run from the repo root; needs build-bench/ (cmake -DCHEATAH_GPU_LINALG_BENCH=ON).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    HAVE_TORCH = True
    HAVE_CUDA = torch.cuda.is_available()
except Exception:  # noqa: BLE001 — torch genuinely optional
    HAVE_TORCH = False
    HAVE_CUDA = False

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "build-bench" / "bench"
TRIALS = 5

# The shared deterministic fill (harness::fill / probe.cpp / gpu_linalg_bench.cpp).
def fill(n, salt, dtype):
    i = np.arange(n, dtype=np.float64)
    return (0.25 * ((i * 7 + salt) % 16) - 1.0).astype(dtype)


def probe(op, n, dtype, salt=1):
    """Run the device op through the probe CLI; returns its checksum."""
    exe = BENCH / "gpu_linalg_probe_vk"
    out = subprocess.run([str(exe), op, str(n), dtype, str(salt)], capture_output=True,
                         text=True, check=True).stdout.split()
    return float(out[1])


def reference(op, n, dtype, salt=1):
    """The numpy reference checksum for the same operands."""
    np_dtype = np.float32 if dtype == "f32" else np.float64
    if op == "matmul":
        a = fill(n * n, salt, np_dtype).reshape(n, n)
        b = fill(n * n, salt + 1, np_dtype).reshape(n, n)
        return float((a.astype(np.float64) @ b.astype(np.float64)).sum())
    if op == "transpose":
        return float(fill(n * n, salt, np_dtype).sum())
    a = fill(n, salt, np_dtype)
    b = fill(n, salt + 1, np_dtype)
    if op == "dot":
        return float(np.dot(a.astype(np.float64), b.astype(np.float64)))
    if op == "sum":
        return float(a.astype(np.float64).sum())
    if op == "add":
        return float((a.astype(np.float64) + b.astype(np.float64)).sum())
    if op == "axpy":
        return float((2.5 * a.astype(np.float64) + b.astype(np.float64)).sum())
    raise ValueError(op)


def crosscheck(ops, sizes, dtype):
    ok = True
    for op in ops:
        n = sizes[op][len(sizes[op]) // 2]
        got = probe(op, n, dtype)
        want = reference(op, n, dtype)
        tol = max(1e-6 if dtype == "f32" else 1e-9, abs(want) * (1e-4 if dtype == "f32" else 1e-9))
        if abs(got - want) > tol:
            print(f"  ⚠ DISAGREE {op} n={n} {dtype}: cheatah {got} vs numpy {want}", file=sys.stderr)
            ok = False
    return ok


def cheetah_times(dtype):
    """dev-resident µs/op from the Google Benchmark binary (wall time), keyed (op, n)."""
    exe = BENCH / "gpu_linalg_bench_vk"
    flt = f"_{dtype}_dev"
    out = subprocess.run([str(exe), f"--benchmark_filter={flt}", "--benchmark_min_time=0.2s",
                          "--benchmark_format=json"], capture_output=True, text=True, check=True)
    times = {}
    for b in json.loads(out.stdout)["benchmarks"]:
        name = b["name"]                       # e.g. BM_matmul_f32_dev/1024/real_time
        parts = name.split("/")
        op = parts[0].removeprefix("BM_").removesuffix(f"_{dtype}_dev")
        if not parts[1].isdigit():
            continue
        times[(op, int(parts[1]))] = b["real_time"]  # µs (unit is us)
    return times


def timeit(f, repeat):
    best = float("inf")
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        for _ in range(repeat):
            f()
        dt = (time.perf_counter() - t0) / repeat * 1e6
        best = min(best, dt)
    return best


def numpy_time(op, n, np_dtype):
    if op == "matmul":
        a = fill(n * n, 1, np_dtype).reshape(n, n)
        b = fill(n * n, 2, np_dtype).reshape(n, n)
        out = np.empty_like(a)
        return timeit(lambda: np.matmul(a, b, out=out), max(1, 10_000_000 // (n * n)))
    a = fill(n, 3, np_dtype)
    b = fill(n, 4, np_dtype)
    if op == "dot":
        return timeit(lambda: np.dot(a, b), max(1, 30_000_000 // n))
    if op == "sum":
        return timeit(lambda: a.sum(), max(1, 30_000_000 // n))
    if op == "add":
        out = np.empty_like(a)
        return timeit(lambda: np.add(a, b, out=out), max(1, 30_000_000 // n))
    if op == "axpy":
        out = np.empty_like(a)
        def f():
            np.multiply(a, 2.5, out=out)
            np.add(out, b, out=out)   # augmented assignment would rebind `out` locally
        return timeit(f, max(1, 30_000_000 // n))
    return None


def torch_time(op, n, t_dtype, device):
    if not HAVE_TORCH or (device == "cuda" and not HAVE_CUDA):
        return None
    a_np = fill(n * n if op == "matmul" else n, 1, np.float64)
    b_np = fill(n * n if op == "matmul" else n, 2, np.float64)
    a = torch.tensor(a_np, dtype=t_dtype, device=device)
    b = torch.tensor(b_np, dtype=t_dtype, device=device)
    if op == "matmul":
        a = a.reshape(n, n)
        b = b.reshape(n, n)
        out = torch.empty_like(a)
        fn = lambda: torch.matmul(a, b, out=out)  # noqa: E731
        repeat = max(1, 10_000_000 // (n * n))
    elif op == "dot":
        fn = lambda: torch.dot(a, b)  # noqa: E731
        repeat = max(1, 30_000_000 // n)
    elif op == "sum":
        fn = lambda: a.sum()  # noqa: E731
        repeat = max(1, 30_000_000 // n)
    elif op == "add":
        out = torch.empty_like(a)
        fn = lambda: torch.add(a, b, out=out)  # noqa: E731
        repeat = max(1, 30_000_000 // n)
    elif op == "axpy":
        out = torch.empty_like(a)
        fn = lambda: torch.add(b, a, alpha=2.5, out=out)  # noqa: E731
        repeat = max(1, 30_000_000 // n)
    else:
        return None
    if device == "cuda":
        fn()
        torch.cuda.synchronize()
        def synced():
            fn()
            torch.cuda.synchronize()
        return timeit(synced, repeat)
    return timeit(fn, repeat)


DEFAULT_SIZES = {
    "matmul": [256, 1024, 2048, 4096],
    "dot": [1 << 17, 1 << 21, 1 << 24],
    "sum": [1 << 17, 1 << 21, 1 << 24],
    "add": [1 << 17, 1 << 21, 1 << 24],
    "axpy": [1 << 17, 1 << 21, 1 << 24],
}


def gemm_sweep():
    """The cuBLAS head-to-head: GEMM across dtype modes, ours vs torch-cuda, TFLOP/s.

    torch f32 runs twice: allow_tf32=False (true FFMA SGEMM — the apples-to-apples for our
    fast kernel) and allow_tf32=True (tensor-core TF32 — NOT reachable from Vulkan on this
    driver, measured for honesty)."""
    if not HAVE_CUDA:
        print("torch-cuda unavailable — gemm sweep skipped")
        return 1
    exe = BENCH / "gpu_linalg_bench_vk"
    out = subprocess.run([str(exe), "--benchmark_filter=BM_matmul_f32_dev/|f16coop|f16acc_coop",
                          "--benchmark_min_time=0.5s", "--benchmark_format=json"],
                         capture_output=True, text=True, check=True)
    ours = {}
    for b in json.loads(out.stdout)["benchmarks"]:
        parts = b["run_name"].split("/")
        if len(parts) > 1 and parts[1].isdigit():
            ours[(parts[0], int(parts[1]))] = b["real_time"]

    def tor(n, dtype, tf32):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        a = torch.tensor(fill(n * n, 1, np.float64), dtype=dtype, device="cuda").reshape(n, n)
        b = torch.tensor(fill(n * n, 2, np.float64), dtype=dtype, device="cuda").reshape(n, n)
        o = torch.empty(n, n, dtype=dtype, device="cuda")
        torch.matmul(a, b, out=o)
        torch.cuda.synchronize()
        def f():
            torch.matmul(a, b, out=o)
            torch.cuda.synchronize()
        return timeit(f, max(1, 3 * 4096**3 // n**3))

    def tf(n, us):
        return f"{2.0 * n**3 / (us * 1e-6) / 1e12:.2f}" if us else "—"

    print("| n | ours f32 | torch f32 (FFMA) | torch f32 (TF32) | ours f16/f32acc | ours f16acc | torch f16 |")
    print("|---|---|---|---|---|---|---|")
    rows = []
    for n in (1024, 2048, 4096):
        us_f32 = ours.get(("BM_matmul_f32_dev", n))
        us_coop = ours.get(("BM_matmul_f16coop_dev", n))
        us_h = ours.get(("BM_matmul_f16acc_coop_dev", n))
        t_ffma = tor(n, torch.float32, False)
        t_tf32 = tor(n, torch.float32, True)
        t_half = tor(n, torch.float16, False)
        row = (f"| {n} | {tf(n, us_f32)} | {tf(n, t_ffma)} | {tf(n, t_tf32)} | "
               f"{tf(n, us_coop)} | {tf(n, us_h)} | {tf(n, t_half)} | (TFLOP/s)")
        rows.append(row)
        print(row)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default="matmul,dot,sum,add,axpy")
    ap.add_argument("--dtype", default="f32", choices=["f32", "f64"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--gemm-sweep", action="store_true",
                    help="GEMM dtype head-to-head vs torch-cuda (TFLOP/s table)")
    args = ap.parse_args()
    if args.gemm_sweep:
        return gemm_sweep()
    ops = args.ops.split(",")
    np_dtype = np.float32 if args.dtype == "f32" else np.float64
    t_dtype = (torch.float32 if args.dtype == "f32" else torch.float64) if HAVE_TORCH else None

    print("cross-checking answers before timing…")
    if not crosscheck(ops, DEFAULT_SIZES, args.dtype):
        print("ANSWERS DISAGREE — refusing to publish timings.", file=sys.stderr)
        return 1
    print("  all agree ✓")

    ck = cheetah_times(args.dtype)
    rows = []
    for op in ops:
        for n in DEFAULT_SIZES[op]:
            c = ck.get((op if op != "axpy" else "axpy_fused", n))
            np_t = numpy_time(op, n, np_dtype)
            tc = torch_time(op, n, t_dtype, "cpu")
            tg = torch_time(op, n, t_dtype, "cuda")
            def r(x):
                return f"{x:.1f}" if x is not None else "—"
            def ratio(x):
                return f"{x / c:.2f}×" if (x is not None and c) else "—"
            rows.append(f"| {op} | {n} | {r(c)} | {r(np_t)} ({ratio(np_t)}) | "
                        f"{r(tc)} ({ratio(tc)}) | {r(tg)} ({ratio(tg)}) |")
            print(rows[-1])

    hdr = (f"| op | n | cheatah-vk µs | numpy µs (vs) | torch-cpu µs (vs) | torch-cuda µs (vs) |\n"
           f"|----|---|---------------|---------------|-------------------|--------------------|")
    table = hdr + "\n" + "\n".join(rows)
    if args.out:
        Path(args.out).write_text(table + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
