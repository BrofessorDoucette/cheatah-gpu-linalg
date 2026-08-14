# Acknowledgments

cheatah-gpu-linalg is original work by Joshua Doucette, acting on behalf of BigBrain LLC —
but it stands on open-source giants. The library itself vendors nothing at runtime: the
projects below are build/test-time tooling or fetched, pinned dependencies of the test and
benchmark suites.

## Standing on the shoulders of

| Project | Link | How it informed cheatah-gpu-linalg |
|---|---|---|
| **cheatah** | <https://github.com/BrofessorDoucette/cheatah> | The host language and toolchain; this library plugs into `cheatah::linalg`'s device seam and follows its QA conventions. |
| **cheatah-gpu** | <https://github.com/BrofessorDoucette/cheatah-gpu> | The backend surfaces underneath: volk-loaded Vulkan forwarders, Metal (and the software-emulated Metal device the CI lanes run on). |
| **volk** | <https://github.com/zeux/volk> | The Vulkan meta-loader (vendored via cheatah-gpu, pinned) — why nothing here links `libvulkan` directly. |
| **Slang** | <https://github.com/shader-slang/slang> | Every kernel is written once in Slang and compiled per backend (`slangc` from the Vulkan SDK) to SPIR-V and MSL. |
| **Vulkan SDK / Khronos** | <https://vulkan.lunarg.com/> | The Vulkan specification, validation layers, and toolchain. |
| **metal-cpp** | <https://developer.apple.com/metal/cpp/> | Apple's C++ Metal bindings (via cheatah-gpu) for the native Apple path. |
| **GoogleTest** | <https://github.com/google/googletest> | The test framework (pinned to the same commit as the cheatah toolchain). |
| **Google Benchmark** | <https://github.com/google/benchmark> | The benchmark harness behind `bench/` and the performance ratchet (same pin as cheatah). |
| **NumPy / PyTorch** | <https://numpy.org/> · <https://pytorch.org/> | The answer-verification references the benchmark suite cross-checks against before timing anything. |

> No source code from the projects above has been copied into cheatah-gpu-linalg. Where a
> project is used, it is fetched at a pinned version or invoked as a tool; where an idea
> informed the design, it is credited here.
