// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
#pragma once

/**
 * @file context.hpp
 * @brief The backend switch: ONE device-context surface, two implementations.
 *
 * Every routine and the `device_array` container drive the GPU through the same small interface —
 * `detail::ctx()` returning the process-wide `detail::Context`, with `new_buffer` / `contents` /
 * `release_buffer` and the blocking `dispatch_1d` / `dispatch_2d` — and never name a backend. This
 * header picks the implementation at compile time.
 *
 * A consumer normally defines NOTHING and gets the platform default — Metal on Apple, Vulkan
 * everywhere else (the same rule as cheatah-gpu's backend switch). A LANE PIN overrides it, for
 * the dual-backend test matrix and special builds: `CHEATAH_GPU_LINALG_VULKAN` /
 * `CHEATAH_GPU_LINALG_METAL` (cheatah-gpu's `CHEATAH_GPU_BACKEND_*` spellings are honored too).
 *
 *   - Vulkan lane → vulkan_context.hpp (SPIR-V from slangc, real GPUs and llvmpipe on any OS
 *     with a Vulkan loader — reached through volk, so nothing links libvulkan);
 *   - Metal lane  → metal_context.hpp (slang-generated MSL on Apple; cheatah-gpu's
 *     software-emulated Metal device + C++ stand-ins elsewhere).
 *
 * The kernels themselves are the SAME Slang source either way (kernels/linalg.slang).
 */

#if defined(CHEATAH_GPU_LINALG_VULKAN) || defined(CHEATAH_GPU_BACKEND_VULKAN)
#  include "cheatah_gpu_linalg/vulkan_context.hpp"
#elif defined(CHEATAH_GPU_LINALG_METAL) || defined(CHEATAH_GPU_BACKEND_METAL)
#  include "cheatah_gpu_linalg/metal_context.hpp"
#elif defined(__APPLE__)
#  include "cheatah_gpu_linalg/metal_context.hpp"
#else
#  include "cheatah_gpu_linalg/vulkan_context.hpp"
#endif

namespace cheatah::gpu::linalg {

/// Whether the device context can come up on this machine — a cached one-shot probe that NEVER
/// throws. This is the runtime "is there a GPU?" question (the nullopt-probe pattern at the
/// linalg layer): gate work on it instead of hand-rolling a try/catch around a first dispatch.
/// `detail::ctx()` itself still throws on first touch when the device is genuinely required.
/// @return true when the backend context is (or can be) live; false when bring-up failed.
/// @complexity O(1) after the first call (the probe result is cached).
/// @alloc none after the first call. @gpualloc none.
/// @test gpu:guards
inline bool available() noexcept {
    static const bool ok = [] {
        try {
            (void)detail::ctx();
            return true;
        } catch (...) {
            return false;
        }
    }();
    return ok;
}

/// Why `available()` is false — the probe's exception text, "" while available. The consumer's
/// skip note ("gpu lane skipped: …") should carry this instead of inventing its own reason.
/// @return The bring-up failure message, or the empty string while the device is available.
/// @complexity O(1) after the first call (the reason is cached).
/// @alloc one cached string on the first call. @gpualloc none.
/// @test gpu:guards
inline const std::string& unavailable_reason() noexcept {
    static const std::string reason = [] {
        try {
            (void)detail::ctx();
            return std::string();
        } catch (const std::exception& e) {
            // Exhaustive: context bring-up throws std::runtime_error only, so no bare
            // catch(...) — an untestable branch would just decay the coverage contract.
            return std::string(e.what());
        }
    }();
    return reason;
}

}  // namespace cheatah::gpu::linalg
