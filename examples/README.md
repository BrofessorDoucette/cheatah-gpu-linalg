# Examples — the same code on CPU and GPU

Every example is ONE templated function; `step<host_array>(…)` runs cheatah's single-threaded
SIMD CPU path, `step<device_array>(…)` runs the GPU kernels. The template argument is the entire
difference. Each example cross-checks the two results, prints per-path timings, asserts DEVICE
RESIDENCY through the transfer ledger (`stats()` — the training loop must download zero bytes),
and is CTest-gated (`ctest -R example:`) so it can never drift from the API.

| Example | Shows |
|---------|-------|
| `mlp_forward` | 3-layer MLP inference — matmul + operators + ReLU as `(x+abs(x))/2`; upload once, download only logits |
| `linear_regression_gd` | a 200-epoch gradient-descent TRAINING LOOP with zero mid-loop downloads; fused `axpy` updates; device-resident loss via `sum(out, a)` read back once |
| `batched_inference` | 32 models applied in ONE `matmul([B,M,K],[B,K,N])` call = one device dispatch |
| `vector_pipeline` | numpy-style operator chains, ufuncs and reductions, location-blind |
| `purr/gpu_regression.purr` | **the training loop in PURE CHEATAH** — `import gpulinalg`, `to_device` once, then ordinary `linalg.matmul`/operators/`axpy` run on the GPU; zero mid-loop downloads asserted |
| `purr/gpu_pipeline.purr` | pure-cheatah array math (operators + reductions), GPU-executed, host-verified — 16 bytes total cross the bus |

Build + run (built with the normal repo build; also run by `ctest`):

```sh
cmake -B build -G Ninja && cmake --build build -j
./build/examples/example_mlp_forward_vk        # Vulkan (real GPU)
./build/examples/example_mlp_forward_mtl       # Metal (emulated off-Apple)
examples/run_purr_examples.sh                  # the pure-cheatah examples (needs purrc built)
ctest --test-dir build -R example:
```

The `.purr` examples import the `gpulinalg` module (purr/gpulinalg/gpulinalg.hpp): a value made
by `gpulinalg.to_device(x)` is a device array, and every `linalg.*` call and operator the
cheatah program writes after that resolves to the GPU overloads in the generated C++ — the
language surface doesn't change at all.
