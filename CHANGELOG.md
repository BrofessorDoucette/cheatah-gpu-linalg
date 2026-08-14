# Changelog

All notable changes to cheatah-gpu-linalg. This project is **alpha** — expect breaking changes
between releases. The repo is public at github.com/BrofessorDoucette/cheatah-gpu-linalg
(published as a fresh history at v0.4.0-alpha); it joins the Biome Standard alongside the
other cheatah extensions.

## v0.4.3-alpha (2026-08-14) — the sibling default learns about CPM

Round three of the acceptance test reached this repo's cross-repo include and caught the
last stale assumption: `CHEATAH_GPU_DIR` defaulted to a SIBLING checkout (`../cheatah-gpu`),
which does not exist inside a consumer's `_deps` tree (CPM names it `cheatah-gpu-src`), so
the exported include for `gpu/dispatch/dispatch.hpp` pointed at nothing. The default now
prefers `cheatah-gpu_SOURCE_DIR` when a consumer's CPM defined it (biome fetches cheatah-gpu
first); sibling checkouts behave exactly as before.

## v0.4.2-alpha (2026-08-14) — consumer.cmake lands where a consumer looks

One more consumer-context defect from the standard's acceptance test: `consumer.cmake` (and
the volk archive path it exports) was written to `CMAKE_BINARY_DIR` — the ROOT build tree,
which is this repo's own build when standalone but the CONSUMER'S root when fetched as a
subproject, so the per-extension path a consuming toolchain checks never saw it (and two
extensions would have collided at the root). Both now use `CMAKE_CURRENT_BINARY_DIR`;
sibling standalone flows are byte-identical (the two directories coincide there).

## v0.4.1-alpha (2026-08-14) — a well-behaved dependency

The Biome Standard's fresh-directory acceptance test consumed this repo through CPM for the
first time and caught two consumer-context defects, both fixed:

- **The build ends at the consumer surface when fetched as a subproject**: libraries, shader
  compilation, and `consumer.cmake` build; the test battery, examples, and benchmark tooling
  (which assume sibling checkouts and a dev toolchain) no longer configure inside a
  consumer's `_deps` tree — previously `examples/` exploded the consumer's generate step.
- **`consumer.cmake` exports `CHEATAH_GPU_LINALG_CONSUMER_CXXFLAGS`** — the lane's baked
  shader-directory define — completing the flat-variable protocol (`_INCLUDES`/`_LIBS`/
  `_SHADERS`/`_CXXFLAGS`) that lets a non-CMake compile line (purrc under biome) consume
  this build without knowing any backend detail.

## v0.4.0-alpha (2026-08-13) — public, packaged, and covered to 100%

The publication release: the repo goes public at
github.com/BrofessorDoucette/cheatah-gpu-linalg as a fresh single-commit history, packaged as
a biome-installable extension, with unit coverage driven to 100% and enforced.

### Publication

- Fresh public history; per-file license headers on all sources; ACKNOWLEDGMENTS.md for the
  open-source foundations (volk, Slang, Vulkan SDK, metal-cpp, GoogleTest, Google Benchmark,
  and the NumPy/PyTorch numerics conventions).
- SECURITY.md (single-trust policy + reporting) and SECURITY-AUDIT-v0.3.0.md (the full
  round-6 audit, extracted from the README, which now links instead of inlining).
- The private-reference scanner stays a hard gate stage and guards every commit message via
  the pre-push hook (scripts/setup-hooks.sh wires it).

### 100% unit coverage, enforced

- Two new suites close the gaps: `factories_test.cpp` (transfers + contiguity, fill kernel vs
  staged complex fill, zeros/ones/full family, array/scalar/arange/reshape/get, printing,
  transfer-stats ledger, single-stage reductions) and `guards_test.cpp` (the `available()`
  probe on BOTH answers via a poisoned Vulkan lane, every audit-hardened boundary, dot
  length-mismatch throws across element types, the two-stage reduction regime, the
  device-resident dot front, matmul/conj_transpose/kron validators, empty-operand short
  circuits, and the benchmark kernel stand-ins).
- `scripts/coverage.sh`: clang source-based coverage over the backend-neutral headers via ONE
  combined test binary (true llvm-cov union semantics — per-TU binaries pollute the union
  with zero-count template records), with `show`/`funcs`/`update-readme` modes; the README
  carries the committed coverage table. **100.00% lines (1316/1316) + 100.00% functions
  (139/139).**
- `scripts/qa.sh` grows two hard stages: the 100% lines+functions coverage gate (with README
  drift check) and cppcheck (performance + security). The gate is now: both-backend tests +
  llvmpipe, ASan/UBSan, Valgrind, 100% Javadoc, 100% unit coverage, cppcheck, perf report.
- `device_array` internals: the `DeviceBuffer` wrapper struct is replaced by a
  `shared_ptr<detail::Buffer>` with a releasing deleter — identical ownership semantics,
  less machinery.

### Fixed

- Leak-on-throw in the dot family, caught by the new guard tests under LeakSanitizer: the
  scalar front allocated its result buffer before `dot_partials` validated operand lengths,
  so a length-mismatch throw leaked the allocation (and the device-resident front could have
  thrown from inside a fused submission, leaving it half-recorded). Length validation now
  happens at the fronts — before any allocation and before `begin_fuse` — and
  `dot_partials` documents that contract.

### Biome packaging

- `scripts/sign-modules.sh` writes the `gpulinalg.hpp.sha512` sidecar that makes
  `import gpulinalg` resolve as a VERIFIED module on the extension path.
- `scripts/test-biome-install.sh` sandboxes the real consumer flow: the purr package copied
  to a throwaway dir, a fresh project compiled with cheatah env vars cleared and only
  `CHEATAH_MODULE_PATH` wired — must print `RESULT: PASS` on the device.
- `cheatah.toml` manifest (targets the v1.10.0-alpha toolchain; requires cheatah-gpu).

## v0.3.0-alpha (2026-07-25) — the full doc-tag contract, a fully tested bridge, and release hygiene

The first tagged release, so this entry carries the whole arc: the dual-backend device surface
and its performance campaign (rounds 4–6), the 2026-07-17 security audit, and this release's
own work — a 100%-Javadoc documentation contract with a hard gate, test coverage for the last
six untested bridge routines, and version/CHANGELOG hygiene.

### This release — documentation contract + doc gate
- **Every public entity now carries the full cheatah doc-tag contract**: brief, `@param` /
  `@return`, `@complexity` (real Big-O), `@alloc` (host) and `@gpualloc` (device) told apart
  honestly, and `@test` naming the actual ctest that exercises it. The bridge routines document
  **transitively** — cheatah's host algorithm's complexity and scratch (LU / Householder QR /
  Golub–Reinsch SVD / tridiagonal QL) plus the transfer cost and the f32 conversion buffers —
  and the device paths document the pooled/content-cached scratch story (`@alloc none` because
  dims/scalar/partial buffers recycle; allocating fronts carry `@gpualloc` one pooled device
  data buffer). Known host round-trips (f64 exp/log, arange, astype, to_string) say so.
- **A 100%-Javadoc hard gate** (`Doxyfile` + `scripts/doc_coverage.sh`, the cheatah-gpu ALIASES
  block including `@gpualloc`/`@destroy`): a strict EXTRACT_ALL=NO Doxygen pass that fails on
  any undocumented public entity, wired into `scripts/qa.sh` as its own stage. Green at 100%.

### This release — the last six untested bridge routines
- **matrix_power, eigvals, eigh, lstsq, pinv and svdvals** now have golden-math test groups in
  the `gpu:bridge:{vk,mtl}` suites: each compares against the host path AND checks a
  closed-form property that can fail on its own (A⁰ = I and A³ vs a matmul chain; Σλ = tr(A);
  V·diag(λ)·Vᵀ = A reconstruction; normal-equation residual orthogonality; the Moore–Penrose
  B·B⁺·B = B; svdvals ≡ svd's singular values, descending). Failure verified by perturbation.
  All 18 host-bridged routines are now covered.

### This release — GoogleTest migration (the ecosystem standard)
- **Every C++ unit-test suite is now GoogleTest** (pinned at the same googletest commit as
  cheatah / cheatah-gpu), replacing the custom `tests/harness.hpp` Checker + `RESULT: PASS`
  regex machinery — exit codes are the truth now. Coverage is byte-for-byte preserved: the
  deterministic sign-varied fills, every in-test reference loop (golden math, not golden
  files), and every tolerance (`Checker::near`'s absolute-plus-relative bound lives on as
  `EXPECT_NEAR_REL`). Each TU still builds twice (vk + mtl); `gtest_discover_tests` registers
  every TEST as its own ctest entry under the stable `gpu:<op>:<bk>:` prefix, so the header
  `@test` tags and `-R` filters keep working — 22 monolithic ctest entries became 144
  per-TEST ones. Failability re-verified by perturbation. The benchmarks were already Google
  Benchmark (nothing to migrate there); the examples stay standalone programs by design.

### This release — hygiene
- Version 0.2.0 → 0.3.0; this CHANGELOG; the fully merged `perf-round4/5/6` branches deleted.

### The surface (built up over v0.1–v0.2, untagged)
- **7-op dual-backend device surface**: matmul (incl. one-dispatch batched), outer,
  conj_transpose, kron, dot/vdot/inner, trace — ONE Slang source compiled per backend (SPIR-V
  for Vulkan; slang-generated MSL for Apple, C++ stand-ins on the emulated Metal device), every
  op tested on BOTH backends plus a forced-llvmpipe pass.
- **The host bridge** (`bridge.hpp`): all 18 of cheatah's factorization/solver fronts work on
  device-resident data — download → cheatah's own host kernel → upload, honestly labelled as
  host-executing; complex comes free; f32 widens to f64 in the middle.
- **Elementwise + operators + factories**: add/sub/mul/divide (array⊗array strict-shape,
  array⊗scalar broadcast both orders), sqrt/exp/log/abs, fused `axpy` (≈4× the chained form),
  deterministic sum/mean with device-resident overloads, the ndarray-parity construction/
  inspection surface, and the `stats()` transfer ledger that makes device residency a tested
  property. `import gpulinalg` drives all of it from pure cheatah.
- **Memory model**: VRAM (device-local) data buffers with staged transfers, a pooled size-class
  allocator (dispatch ~1.4 ms → 23 µs), content-cached dims buffers, host-cached readback
  staging (0.22 → 7.6 GB/s downloads).

### Performance (rounds 4–6 — see PERFORMANCE.md for the full measured story)
- **f32 GEMM 9.39 TFLOP/s** @4096³ (66% of the measured 14.2 FFMA ceiling): 128×128
  double-buffered register tiling, warp-coherent B quads, A-store bank skew, the vec4-C
  epilogue (+20% @1024³), and the fast-64 sibling for M%64 rectangles.
- **Cooperative-matrix f16 GEMM 20.3 TFLOP/s** (opt-in, `coop_ok()`-gated): 2×8 tiles per warp
  + fragment prefetch; the efficiency ledger documents why cuBLAS's remaining gap is
  driver-bound (no TF32-class combo exposed via Vulkan on this driver).
- **Two-stage deterministic reductions** at 305–330 GB/s (85–96% of the measured ~375 GB/s
  sustainable bandwidth) with an on-device finalize (8-byte readback) — and **submission
  fusing** dropped the dispatch floor 50 → **30 µs** (fence wait is the measured floor),
  halving the reduction crossover to n ≥ 256k.
- The crossover table (matmul wins from n ≥ 128; streaming/reductions from ~256k) feeds the
  consuming application's CPU/GPU dispatcher.

### Security (audit 2026-07-17 — six findings fixed)
- Overflow-checked shape products + a 2³²−1 element cap (makes 32-bit kernel indexing provably
  non-wrapping); upload/download byte/offset validation on both backends; dispatch workgroup
  counts checked against device limits; the pow2-≥256 B pool invariant that keeps vec4 tails in
  bounds; SPIR-V blob size caps + kernel-name path-separator rejection; `begin_fuse` nesting
  rejected and allocation ordered before fusing. All under the single-trust threat model.

### Not yet (honest scope)
- **No f16 `device_array<half>` API** — the tensor-core kernels exist and are benchmarked, but
  the public element types are f32/f64 (+ c64/c128 on the seam ops); deeper coop-matrix work is
  a named next step.
- **No general broadcasting** (scalar broadcast only) and **no true device factorizations**
  (the bridge executes on the host; device Cholesky first when profitable).
- **No streaming/pinned e2e transfer path** (one-shot offload is transfer-bound) and the
  dispatch floor stands at 30 µs (push descriptors + a persistent dims ring are the next lever).
- **No coverage tooling yet** — the QA gate runs both backends, llvmpipe, ASan/UBSan, Valgrind
  and the doc gate, but line/function coverage is not measured in this repo.
- **Real-Apple unvalidated** — the Metal backend off-Apple is a correctness rig (emulated
  stand-ins); no Apple performance numbers exist or are claimed, and the native
  simdgroup_matrix GEMM skeleton stays unwired until the first Mac.
