// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
#pragma once

/**
 * @file gpulinalg.hpp
 * @brief The PURR-facing module of cheatah-gpu-linalg: `import gpulinalg` from cheatah code.
 *
 * A cheatah program drives the GPU with the SAME linalg calls it uses on the CPU: a value made
 * by `gpulinalg.to_device(x)` flows through `let` bindings as a `device_array`, and every
 * `linalg.matmul(...)`, every `a + b`, every `linalg.dot(...)` the program writes resolves to
 * the GPU overloads by ordinary C++ overload resolution in the generated code — no new
 * language surface at all. This header only re-exports the handful of entry points purr calls
 * BY MODULE NAME (`gpulinalg.<fn>`), aliasing them into the module's namespace.
 *
 * Compile a .purr importing this with the Vulkan backend flags — see
 * examples/run_purr_examples.sh for the exact purrc invocation.
 */

#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

namespace cheatah::gpulinalg {

// Container movement + inspection (the sync points, documented in factories.hpp).
using cheatah::gpu::linalg::to_device;
using cheatah::gpu::linalg::to_host;
using cheatah::gpu::linalg::get;
using cheatah::gpu::linalg::to_string;
using cheatah::gpu::linalg::size_of;

// Factories (device-resident results).
using cheatah::gpu::linalg::zeros;
using cheatah::gpu::linalg::ones;
using cheatah::gpu::linalg::full;
using cheatah::gpu::linalg::arange;
using cheatah::gpu::linalg::reshape;

// The fused training primitive + reductions (host-returning AND device-resident overloads).
using cheatah::gpu::linalg::axpy;
using cheatah::gpu::linalg::sum;
using cheatah::gpu::linalg::mean;

// The transfer ledger — device residency as something a .purr program can assert on.
using cheatah::gpu::linalg::stats;
using cheatah::gpu::linalg::reset_stats;

/**
 * A 1-element device array — the resident slot a loss accumulator lives in (purr cannot spell
 * `device_array<double>::uninitialized({1})`, so the module provides it).
 * @return A fresh uninitialized 1-element device array.
 * @complexity O(1).
 * @alloc metadata only.
 * @gpualloc one pooled device data buffer (one 256 B size class).
 * @test example:purr
 */
[[nodiscard]] inline cheatah::gpu::linalg::device_array<double> scalar_slot() {
    return cheatah::gpu::linalg::device_array<double>::uninitialized({1});
}

}  // namespace cheatah::gpulinalg
