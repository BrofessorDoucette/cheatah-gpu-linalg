# cheatah-gpu-linalg

GPU-accelerated linear algebra for [cheatah](../cheatah) — a **device backend** for its
`stdlib/linalg`, built on the [cheatah-gpu](../cheatah-gpu) Metal/Vulkan surface.

## How it plugs in

cheatah's `stdlib/linalg` is one two-layer template per operation —
`template <ndarray::Field T, template <typename> class Array> … Array<T> …` — and chooses host vs
device **at compile time** with zero runtime cost and without ever naming a device type:

- an array-like models a structural `ArrayLike` surface (host-resident shape/strides/size/offset), and
- it advertises *where its elements live* through the `location_of` trait. cheatah defines only the
  host tag; a device extension defines its own tag and specializes `location_of`.

This library is that extension. It provides:

1. **`cheatah::gpu::linalg::device_array<T>`** — a dense, row-major, GPU-resident array whose
   elements live in a cheatah-gpu buffer while its shape/strides stay on the host.
2. **one `location_of` specialization** — the entire opt-in; it makes `device_array<T>` satisfy
   `cheatah::linalg::DeviceArray` (and *not* `HostArray`).
3. **device routine overloads** (`matmul`, …) constrained `requires DeviceArray<Array<T>>`, in
   `device_array`'s namespace, so cheatah's allocating fronts dispatch to them **by ADL** — no CPO,
   no tag, no ambiguity.

The result: `cheatah::linalg::matmul(dev_a, dev_b)` runs on the GPU with no extra plumbing.

```cpp
#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"
namespace gl = cheatah::gpu::linalg;

auto a = gl::device_array<double>::from_host({M, K}, host_a);
auto b = gl::device_array<double>::from_host({K, N}, host_b);
gl::device_array<double> c = cheatah::linalg::matmul(a, b);   // runs on the GPU
c.to_host(out);
```

## Kernels: Slang, once — both backends

Every kernel is written ONCE in [Slang](https://shader-slang.org/) (`kernels/linalg.slang`) and
compiled per backend at build time by `slangc` (ships in the Vulkan SDK):

| backend | kernel artifact | runs on |
|---------|-----------------|---------|
| **Vulkan** (`cheatah_gpu_linalg_vulkan`) | SPIR-V, one `.spv` per entry × element | any Vulkan device — discrete/integrated GPUs, llvmpipe |
| **Metal** (`cheatah_gpu_linalg_metal`), Apple | slang-generated MSL, runtime-compiled | the Apple GPU |
| **Metal**, off Apple | *(source ignored)* | cheatah-gpu's software-emulated device runs the C++ stand-ins in `kernels.hpp` |

The routine/container code never names a backend — `context.hpp` picks the implementation
(`CHEATAH_GPU_LINALG_VULKAN` ⇒ `vulkan_context.hpp`, else `metal_context.hpp`), both exposing the
same `new_buffer` / `contents` / `dispatch_1d` / `dispatch_2d` surface built on cheatah-gpu's
`vk.*` / `mtl.*` forwarders and its `gpu.dispatch` workgroup math.

Vulkan device selection prefers discrete > integrated > CPU; force one with
`CHEATAH_GPU_LINALG_VK_DEVICE=<name substring | index>` (e.g. `llvmpipe`). `*_f64` kernels need the
device's `shaderFloat64` feature (NVIDIA/llvmpipe: yes; Intel Xe: f32 only — a clean throw says so).

## Status

The full `cheatah::linalg` surface works with `device_array` operands. Three tiers:

- **Device kernels** (real f32/f64 + complex c64/c128, Slang-single-source):
  `matmul` (register-tiled GEMM, **9.55 TFLOP/s f32 on an RTX 3070 Ti Laptop** — 67% of that
  part's measured 14.2 TFLOP/s FFMA ceiling — plus an opt-in KHR cooperative-matrix
  (tensor-core) f16-in/f32-accumulate path at **21.5 TFLOP/s**; both medians over 5
  interleaved repetitions, see [PERFORMANCE.md](PERFORMANCE.md); batched
  `[B,M,K]@[B,K,N]` in one z-dispatch), `outer`, `conj_transpose` (true Hermitian adjoint for
  complex), `kron`, `dot`/`vdot` (conjugating for complex)/`inner`, `trace` — reductions are
  deterministic partial sums with a two-stage tree path above 64k elements (305+ GB/s).
- **Elementwise device ops** (f32/f64) with OPERATORS — `a + b`, `2.0 * a`, `a += b` resolve by
  ADL exactly as purr's codegen emits them: add/sub/mul/divide (array⊗array strict-shape,
  array⊗scalar both orders), sqrt/exp/log/abs (f64 exp/log host-evaluated — SPIR-V has no f64
  transcendentals), `sum`/`mean`, and the fused **`axpy`** (α·x + y in one dispatch — ≈4× the
  chained form). No general broadcasting yet (scalar broadcast only).
- **Conv-support device kernels** (f32/f64, `conv.hpp`): `im2col2d`/`col2im2d` (the zero-padded
  receptive-field gather and its deterministic gather-form adjoint — no atomics) and the fused
  `conv_bias_act`/`conv_act_grad` epilogues (bias + activation {identity, relu, tanh, sigmoid} +
  the filter-major ↔ sample-major layout transpose in one dispatch each, derivatives evaluated
  from the output) — a conv layer's staging and epilogues stay device-resident around the GEMM.
- **Host-bridged factorizations** (`bridge.hpp` — EXECUTE ON THE HOST, honestly labelled):
  cholesky, qr, svd, svdvals, eig, eigvals, eigh, eigvalsh, solve, lstsq, inv, pinv,
  matrix_power, det, slogdet, cond, matrix_rank, norm — download → cheatah's host algorithm →
  upload, so device-resident code can call the whole surface; complex comes free.

Memory model: `device_array` data lives in DEVICE-LOCAL memory (VRAM) on Vulkan with staged
transfers; scratch (dims/scalars/partials) and every released buffer recycles through a pooled
allocator (a dispatch costs ~30 µs end to end, down from ~1.4 ms unpooled; 23.5 µs of that
is the fence wait, which is the floor).

Every op is tested on BOTH backends (`gpu:<op>:vk` / `:mtl`), `scripts/qa.sh` is the full gate
(both backends + forced-llvmpipe + ASan/UBSan + Valgrind + 100% doc coverage + 100% unit
coverage + cppcheck + a perf-ratchet report), and
[PERFORMANCE.md](PERFORMANCE.md) holds the honest numbers + `bench/` the suite
(`compare.py` cross-checks answers against NumPy/PyTorch before timing anything).

Not yet: general broadcasting, true device factorizations, streaming e2e transfers,
real-Apple validation — see PERFORMANCE.md's named next steps. (Cooperative-matrix
tensor-core GEMM was on this list long after it shipped; it is listed above, where it
belongs. No Apple performance number exists or is claimed.)

## Build

```sh
cmake -S . -B build -G Ninja  # finds ../cheatah, ../cheatah-gpu and the Vulkan SDK by default
cmake --build build -j
ctest --test-dir build --output-on-failure   # every op, both backends
bash scripts/qa.sh                           # the full QA gate
```

Override sibling locations with `-DCHEATAH_DIR=…`, `-DCHEATAH_GPU_DIR=…`, metal-cpp with
`-DCHEATAH_GPU_METAL_CPP=…` (vendored in cheatah-gpu by default), and the SDK with
`-DCHEATAH_GPU_LINALG_VK_ROOT=…` (defaults to `$VULKAN_SDK`, then the newest
`~/Tools/vulkan-sdk/*/x86_64`).

## Coverage

Clang source-based coverage of the unit suite over the backend-neutral headers (the two
backend bring-up headers are excluded the way cheatah-gpu excludes its driver layers — they
need real driver matrices). Enforced at 100% lines + functions by `scripts/qa.sh`.

<!-- coverage:start -->
| Metric | neutral surface |
|--------|-----------------|
| **Lines** | 100.00% (1315/1315) |
| **Functions** | 100.00% (139/139) |
| Regions | 92.88% |
| Branches | 79.79% |
<!-- coverage:end -->

## License

MIT — Copyright (c) 2026 BigBrain LLC. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Security

Single-trust threat model (the cheatah household standard): the embedding process is trusted,
its inputs may still be wrong — robustness bugs at the API boundary are treated as security
bugs. See [SECURITY.md](SECURITY.md) for the policy and
[SECURITY-AUDIT-v0.3.0.md](SECURITY-AUDIT-v0.3.0.md) for the latest full audit (element-count
overflow caps, transfer bounds, dispatch limits, the vec4 tail invariant, loader surfaces,
fusing discipline).
