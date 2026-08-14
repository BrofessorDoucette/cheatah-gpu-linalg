# cheatah-gpu-linalg — performance

**Provenance** (all numbers measured on this exact stack — they are machine-specific):
NVIDIA GeForce RTX 3070 Ti Laptop 8 GB (driver 580.159.03), i7-12700H, 62 GB RAM,
Vulkan SDK 1.4.341.1 / slangc 2026.1, Linux 7.0.11-76070011. Wall-clock (`real_time`) timings
from `bench/gpu_linalg_bench_vk` (Google Benchmark, Release); operands device-resident unless
marked e2e. Reproduce: `cmake -B build-bench -DCHEATAH_GPU_LINALG_BENCH=ON && cmake --build
build-bench -j && build-bench/bench/gpu_linalg_bench_vk`. Cross-implementation tables come from
`bench/compare.py` (answers cross-checked before timing). Baseline ratchet:
`bench/gpu_bench_gate.sh` + `bench/gpu_bench_baseline.csv`.

## The optimization story (measure → change → re-measure)

| change | dispatch overhead | GEMM f32 @4096 (wall) | dot f32 @16M |
|---|---|---|---|
| starting point | 1,461 µs | ~0.1 TFLOP/s | 5.6 GB/s |
| pooled buffer allocator (scratch + temporaries recycle) | **23 µs** | — | — |
| device-local data buffers (operands in GDDR6, staged transfers — they were in PCIe-remote host memory) | — | 1.46 TFLOP/s | — |
| register-tiled GEMM (64×64 block, 4×4 microtile/thread, padded groupshared) | — | 5.74 TFLOP/s | — |
| two-stage reductions (G tree-reducing workgroups ×256, still deterministic) | — | — | 305 GB/s |
| FAST GEMM path (128×128 block, 8×8 microtile, 16-wide double-buffered K slabs, vec4 loads, zero bounds checks; exact tiles only, edge kernel otherwise) | — | **8.6–8.8 TFLOP/s** | — |
| **TENSOR CORES** — KHR cooperative-matrix f16-in/f32-acc GEMM (opt-in, `coop_ok()`-gated) | — | **11.7 TFLOP/s** | — |
| on-device `finalize_partials` (reductions read back ONE element, or none — the resident forms) | — | — | 305 GB/s, 8-byte readback |
| **host-CACHED readback staging** (CPU reads from write-combined memory crawl; downloads staged through a HOST_CACHED heap) | — | — | download 0.22 → **7.6 GB/s** (34×) |
| allocation/copy sweep: persistent descriptor set (no per-dispatch pool reset + set alloc), content-cached dims buffers (repeat shapes = zero memcpys), device-side `fill` kernel behind zeros/ones/full (no host staging vector, no PCIe upload) | 49 µs (pre-change ≈ 51) | — | `zeros(4096²)` 59 ms → **386 µs (153×)** |
| fast GEMM round 4: warp-coherent B quads (consecutive lanes/banks, was 4-way conflicted) then A-store bank skew (row dim 128→129) | — | 8.4 → 8.9 → **9.39 TFLOP/s** | — |
| coop GEMM round 4: 2 A-tiles × 8 B-tiles per warp (16 accumulators, 10 loads per 16 MulAdds; v1 was 5-per-4) | — | f16 11.0 → 16.9 → **19.35 TFLOP/s** | — |
| REJECTED by measurement (the lore): BK=8 occupancy probe (regs bind, not shared), vec4-A K-blocked loop (register spill), 8-warp 256×128 coop block, B-only groupshared coop staging (2nd confirmation: driver coop loads + L2 win), 2× k-unroll | — | — | — |
| round 5 REJECTED: vectorized double-buffered coop staging (3rd staging loss), vec4-A transpose-gather f32 (strided scalar global loads outweigh the 10→4 LDS cut), BK=32 (48 KB shared ceiling), f16-acc GEMM as a speed play (load-bound; kept as instrument + numerics finding) | — | — | — |
| **fast-64 GEMM** (64-row fast sibling; M%64-not-%128 training-shaped rectangles no longer fall to the edge kernel; routing fast > fast64 > edge) | — | (64,4096,4096): 3.7 → **5.8 TFLOP/s** | — |
| round 6 ATTRIBUTION (bench-only surgical variants): the f32 gap = global-traffic ~21 %, barriers ~8 %, epilogue ~8 %, LDS/issue ~24 % | — | — | — |
| **vec4 C epilogue** (C rebound as a float4 view shader-side; 16 stores, was 64) | — | 1024³: 5.2 → **6.24 TFLOP/s** (+20 %) | — |
| **submission fusing** (partial+finalize reductions record into ONE submit; BM_submit_only proved the floor is fence-wait 23.5 µs, record ~2 µs — cmdbuf cache + push descriptors evidence-REJECTED) | **50 → 30 µs** | — | sum crossover **2M → 256k** |
| **coop fragment prefetch** (next k's A + first two B tiles load during MulAdds; 243 regs 0 spills — b2/b3 crossed the register cliff, REJECTED) | — | f16 19.35 → **20.3 TFLOP/s** | — |
| round 6 REJECTED: 512-thread 4×8-microtile occupancy play (compiler allocates 104 regs → still 1 block/SM) | — | — | — |

## The efficiency ledger — why we can't beat cuBLAS, measured (round 5)

Peak probes (`BM_ffma_peak` / `BM_coop_peak_*` / `BM_lds_peak`, pure-compute kernels) measure
what this driver+silicon can ACTUALLY sustain, replacing paper numbers; `bench/vkinspect` dumps
the driver's own per-kernel stats and the cooperative-matrix property table.

| path | measured ceiling | our GEMM | efficiency | binding constraint (driver-reported) |
|---|---|---|---|---|
| f32 FFMA | **14.2 TFLOP/s** (not paper 17.5 — laptop clocks) | 9.39 | **66 %** | 128 regs, 0 spills, 44.7 KB shared → 2 blocks/SM; 10 LDS per 64 FFMA; every load-restructure measured and lost |
| f16 tensor, f32 accumulate | **34.7 TFLOP/s** | 19.35 | **56 %** | fragment-load latency at 227 regs → 2 blocks/SM; staging lost 3× (driver global coopMatLoad + L2 win) |
| f16 tensor, f16 accumulate | **66.6 TFLOP/s** (GeForce half-rate f32-acc CONFIRMED, 1.92×) | 18.3 @2048 | 27 % | same loads bind → the 2× MulAdd rate buys nothing until loads do; f16-acc also drifts >5 % at K=4096 (why cuBLAS HGEMM accumulates in f32) |
| groupshared | 7.1 TB/s | — | — | not the GEMM limiter |
| DRAM | 375 GB/s | — | — | elementwise at 360 = 96 % — done as a category |

What the property table says (vkinspect): f16-acc shapes exist (16×16×16/16×8×16/16×8×8),
**bf16×bf16→f32 exists** (training-grade range at tensor rate — future lever once slangc emits
bf16 coop types), **NO TF32-class f32×f32 combo** — cuBLAS's Ampere SGEMM trick is not reachable
from Vulkan on this driver, so f32 stays FFMA and 66 % of 14.2 is the honest position against
cuBLAS's ~90 % of the same ceiling.

## Device-resident throughput (f32, RTX 3070 Ti)

| op | size | wall | rate |
|---|---|---|---|
| matmul | 1024³ | 417 µs | 5.2 TFLOP/s |
| matmul | 2048³ | 2.2 ms | 7.7 TFLOP/s |
| matmul | 4096³ | 14.6 ms | **9.39 TFLOP/s** |
| matmul f16 (tensor cores, opt-in) | 4096³ | 7.1 ms | **19.35 TFLOP/s** |
| matmul f64 | 1024³ | 8.4 ms | 277 GFLOP/s (GA104 f64 ≈ hardware ceiling — 1:32 rate silicon) |
| dot | 16M | 422 µs | 305 GB/s |
| sum | 16M | 215 µs | 311 GB/s |
| add | 16M | 663 µs | 330 GB/s |
| axpy (fused) | 16M | 655 µs | 300 GB/s — vs 2.7 ms chained `a*α + y` (fusion ≈ 4×) |
| batched matmul | 64×64³ | 53 µs | one z-dispatch — **38×** faster than 64 looped dispatches |
| dispatch overhead | 1-elem op | **30 µs** (was 50 pre-fusing; floor = fence-wait 23.5) | the fixed floor every device op pays |

## Conv-support kernels (new — device numbers pending the next GPU bench run)

Four 1-D kernels (f32 + f64, both backends, `conv.hpp`) remove the neural-net consumer's
host-side conv staging — measured there at ~40 % of training wall-clock counting the PCIe
ping-pong: `im2col2d` / `col2im2d` (the zero-padded receptive-field gather and its adjoint in
GATHER form — each dx cell sums its ≤ K² contributors in the host reference's association
order, so no atomics and the same bits every run) and the fused epilogues `conv_bias_act` /
`conv_act_grad` (bias + activation {identity, relu, tanh, sigmoid} + the filter-major ↔
sample-major layout transpose, one dispatch each; derivatives evaluated from the output). All
four mirror the consumer's host loops element-for-element (`gpu:conv` pins parity, exact for
the data-movement ops). Bench entries at the training shape B=16, C=F=64, 32², K=3:
`BM_im2col2d_*_dev`, `BM_col2im2d_*_dev`, `BM_conv_bias_act_*_dev`, `BM_conv_act_grad_*_dev`.

## vs cheatah's own host linalg (same process, `-O3 -march=native`, single-threaded by design)

| op | host (double) | device (f32) | device/host |
|---|---|---|---|
| matmul 1024³ | 186 ms (11.6 GFLOP/s) | 0.45 ms | **~415×** |
| dot 16M | 10.2 ms (26 GB/s) | 0.42 ms | ~24× |

Measured ceilings (round 4): sustainable stream bandwidth is **~375 GB/s** (vec4 copy/triad
probes at 64–128M — not the paper 448), so add @ 360 GB/s and reductions @ 305–330 GB/s already
run at 85–96 % of what the silicon actually sustains. PCIe: upload ~7 GB/s, download ~7.6 GB/s.

Crossover guidance: below n≈128 (matmul) / n≈256k (reductions) the ~50 µs dispatch floor keeps the
host path competitive — use host arrays for tiny operands, device arrays for everything else, and
avoid ping-ponging (e2e matmul @4096 is 322 ms — the PCIe transfers, not the compute, dominate a
one-shot offload; staged-transfer streaming is a named next step).

## The host↔device crossover table (the consuming application's CPU/GPU dispatcher feeds on this)

Generated by `bench/crossover.py` (paired host/device sweeps, SAME element type — f64; wall
time). Below the crossover the ~50 µs dispatch floor keeps the host path ahead; above it the
device path wins and keeps winning (the script asserts the verdict is monotone):

| op | size | crossover (device wins at) | host @ crossover | device @ crossover |
|---|---|---|---|---|
| matmul (double) | n (matrix is n x n) | **n ≥ 128** | 264 µs | 211 µs |
| dot (double) | n (elements) | **n ≥ 2,097,152** | 1302 µs | 165 µs |
| sum (double) | n (elements) | **n ≥ 2,097,152** | 540 µs | 107 µs |
| add (double) | n (elements) | **n ≥ 262,144** | 112 µs | 46 µs |
| axpy (double) | n (elements) | **n ≥ 262,144** | 116 µs | 40 µs |

Rule of thumb: matmul goes to the GPU almost immediately (n ≥ 128); streaming ops and every
reduction win from ~256k elements (round 6's submission fusing halved the reduction floor).

## vs NumPy / PyTorch

Run `bench/venv/bin/python bench/compare.py` for the full cross-checked table (NumPy CPU,
torch-CPU, torch-CUDA/cuBLAS). Honest expectations, updated for round 4: cuBLAS-class on this GPU is ≈12–14 TFLOP/s f32 and
40+ f16 — the FFMA kernel at 9.39 is ~70 % of that ceiling, the coop kernel at 19.35 is ~50 %
of the f16 one; both moved this round by measure→keep-or-revert (the rejected cells are listed
in the story table on purpose).

## Host-bridged ops (EXECUTE ON THE HOST — bridge.hpp, listed separately on purpose)

| op | size | e2e |
|---|---|---|
| svd | 256² | 3.0 ms |
| solve | 1024² | 127 ms |

These download → run cheatah's host algorithm → upload; they exist so the full linalg surface
works on device-resident data, not for speed.

## The unified API + examples (round 3)

`step<host_array>(…)` vs `step<device_array>(…)` — one compile-time template argument flips
CPU↔GPU; `factories.hpp` mirrors ndarray's construction/inspection surface, `to_device/to_host`
are the explicit converters, and the `stats()` transfer ledger makes DEVICE RESIDENCY a tested
property (see `examples/` — the 300-epoch training loop asserts zero mid-loop downloads;
reductions have device-resident `sum(out, a)`-style overloads that never touch the bus).

## Apple (honest scope)

The Metal backend off-Apple is a correctness rig (emulated stand-ins) — NO Apple performance
number exists or is claimed. `kernels/metal/gemm_simdgroup.metal` is the native simdgroup_matrix
GEMM skeleton (slangc cannot lower CoopMat to MSL), unwired and untested until the first Mac:
compile, gate behind a `simdgroup_ok()` mirror of `coop_ok()`, port the coop bench, tune by
measurement. The NVIDIA lesson to carry over: start from global fragment loads — load latency,
not MulAdd rate, bound first here.

## Named next steps

- Deeper tensor-core work: the coop-matrix kernel is a v1 (global-fragment loads — measured
  FASTER than groupshared staging on this driver); cuBLAS f16 reaches far higher still. An
  f16 `device_array<half>` API (not just the bench kernel) comes with it.
- Streaming/pinned transfer path for e2e workloads (the one-shot offload is transfer-bound).
- Push descriptors + persistent dims ring (below-23 µs dispatch floor; torch's launch ≈ 10 µs).
- True device factorizations (Cholesky first), replacing the host bridge where profitable.
- vec4 elementwise landed ~bandwidth-neutral on NVIDIA (kept: axpy improved; the ~330 GB/s
  plateau appears to be this laptop GPU's sustainable DRAM bandwidth under compute).
