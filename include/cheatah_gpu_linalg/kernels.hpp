// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
#pragma once

/**
 * @file kernels.hpp
 * @brief The kernel name tables + dispatch ABI, and the emulated-Metal C++ stand-ins.
 *
 * The real kernels are written ONCE in Slang (kernels/linalg.slang) and compiled per backend at
 * build time: SPIR-V modules for Vulkan (loaded by vulkan_context.hpp), Metal Shading Language for
 * Apple (runtime-compiled by metal_context.hpp). This header carries what the C++ side must agree
 * on with that source:
 *   - the per-element kernel NAMES a routine dispatches (`gemm_name<T>` …),
 *   - the workgroup sizes the contexts dimension against (`kLocal1d` / `kLocal2d`), and
 *   - the reduction width cap (`kMaxReduce`) of the deterministic partial-sum contract.
 *
 * Off Apple, the METAL backend runs on cheatah-gpu's software-emulated device, which cannot
 * compile MSL; the C++ stand-ins at the bottom are registered by kernel name and reproduce each
 * Slang kernel's semantics exactly (same buffers, same dims layout, same output).
 */

#include <cmath>
#include <complex>
#include <cstdint>

#if !defined(__APPLE__) && !defined(CHEATAH_GPU_LINALG_VULKAN)
#  include "gpu/metal/emulated/emulated.hpp"
#endif

namespace cheatah::gpu::linalg::kernels {

/// Threads per workgroup of every 1-D kernel — matches `[numthreads(256,1,1)]` in linalg.slang.
inline constexpr std::uint32_t kLocal1d = 256;
/// Threads per workgroup axis of every 2-D kernel — matches `[numthreads(16,16,1)]`.
inline constexpr std::uint32_t kLocal2d = 16;
/// The partial-sum width cap: a reduction dispatches P = min(n, kMaxReduce) threads, each writing
/// one deterministic partial the routine sums on the host.
inline constexpr std::uint32_t kMaxReduce = 256;
/// The register-tiled GEMM's C-block edge: one 16x16-thread workgroup computes a 64x64 tile
/// (4x4 register microtile per thread) — matches `gemm` in linalg.slang.
inline constexpr std::uint32_t kGemmBlock = 64;
/// The fast GEMM path's C-block edge (one 16×16-thread workgroup per 128×128 tile, 8×8
/// microtile per thread) and its K-slab width.
inline constexpr std::uint32_t kGemmFastBlock = 128;
/// The fast GEMM path's K-slab width (the double-buffered groupshared depth).
inline constexpr std::uint32_t kGemmFastK = 16;

/// The kernel name a routine dispatches for element type `T`. Specialized per supported element;
/// the primary is left null so an unsupported element fails loudly at the GpuElement constraint,
/// never as a null-named dispatch.
/// GEMM has TWO paths: `gemm_fast` (128×128 blocks, double-buffered, vec4 loads, ZERO bounds
/// checks — real elements with M%128==0 && N%128==0 && K%8==0 only) and `gemm_edge` (64×64,
/// guarded, any shape, all elements). The routine gates on shape.
template <class T> inline constexpr const char* gemm_fast_name = nullptr;
template <> inline constexpr const char* gemm_fast_name<float>  = "gemm_fast_f32";  ///< The f32 entry.
template <> inline constexpr const char* gemm_fast_name<double> = "gemm_fast_f64";  ///< The f64 entry.

/// The 64-row fast sibling (f32 only): M%64, N%128, K%16 — training-shaped batch-64 rectangles.
template <class T> inline constexpr const char* gemm_fast64_name = nullptr;
template <> inline constexpr const char* gemm_fast64_name<float> = "gemm_fast64_f32";  ///< The f32 entry.
/// The guarded 64x64 edge-GEMM kernel name (any shape, every element type).
template <class T> inline constexpr const char* gemm_edge_name = nullptr;
template <> inline constexpr const char* gemm_edge_name<float>  = "gemm_edge_f32";  ///< The f32 entry.
template <> inline constexpr const char* gemm_edge_name<double> = "gemm_edge_f64";  ///< The f64 entry.
template <> inline constexpr const char* gemm_edge_name<std::complex<float>>  = "gemm_edge_c64";  ///< The c64 entry.
template <> inline constexpr const char* gemm_edge_name<std::complex<double>> = "gemm_edge_c128";  ///< The c128 entry.

/// Batched GEMM kernel name per element type (grid z = batch; dims = {M, N, K, B}).
template <class T> inline constexpr const char* gemm_batched_name = nullptr;
template <> inline constexpr const char* gemm_batched_name<float>  = "gemm_batched_f32";  ///< The f32 entry.
template <> inline constexpr const char* gemm_batched_name<double> = "gemm_batched_f64";  ///< The f64 entry.
template <> inline constexpr const char* gemm_batched_name<std::complex<float>>  = "gemm_batched_c64";  ///< The c64 entry.
template <> inline constexpr const char* gemm_batched_name<std::complex<double>> = "gemm_batched_c128";  ///< The c128 entry.

/// Outer product kernel name per element type (see linalg.slang `outer`).
template <class T> inline constexpr const char* outer_name = nullptr;
template <> inline constexpr const char* outer_name<float>  = "outer_f32";  ///< The f32 entry.
template <> inline constexpr const char* outer_name<double> = "outer_f64";  ///< The f64 entry.
template <> inline constexpr const char* outer_name<std::complex<float>>  = "outer_c64";  ///< The c64 entry.
template <> inline constexpr const char* outer_name<std::complex<double>> = "outer_c128";  ///< The c128 entry.

/// Transpose kernel name per element type (the real-element conj_transpose).
template <class T> inline constexpr const char* transpose_name = nullptr;
template <> inline constexpr const char* transpose_name<float>  = "transpose_f32";  ///< The f32 entry.
template <> inline constexpr const char* transpose_name<double> = "transpose_f64";  ///< The f64 entry.
template <> inline constexpr const char* transpose_name<std::complex<float>>  = "transpose_c64";  ///< The c64 entry.
template <> inline constexpr const char* transpose_name<std::complex<double>> = "transpose_c128";  ///< The c128 entry.

/// Kronecker product kernel name per element type.
template <class T> inline constexpr const char* kron_name = nullptr;
template <> inline constexpr const char* kron_name<float>  = "kron_f32";  ///< The f32 entry.
template <> inline constexpr const char* kron_name<double> = "kron_f64";  ///< The f64 entry.
template <> inline constexpr const char* kron_name<std::complex<float>>  = "kron_c64";  ///< The c64 entry.
template <> inline constexpr const char* kron_name<std::complex<double>> = "kron_c128";  ///< The c128 entry.

/// Dot-product partial-sum kernel name per element type (serves dot, vdot and inner — identical
/// for the real elements this library supports).
template <class T> inline constexpr const char* dot_name = nullptr;
template <> inline constexpr const char* dot_name<float>  = "dot_partial_f32";  ///< The f32 entry.
template <> inline constexpr const char* dot_name<double> = "dot_partial_f64";  ///< The f64 entry.
template <> inline constexpr const char* dot_name<std::complex<float>>  = "dot_partial_c64";  ///< The c64 entry.
template <> inline constexpr const char* dot_name<std::complex<double>> = "dot_partial_c128";  ///< The c128 entry.

/// Conjugating dot (vdot) kernel name — COMPLEX elements only (real vdot is bilinear and shares
/// the dot kernel).
template <class T> inline constexpr const char* vdot_name = nullptr;
template <> inline constexpr const char* vdot_name<std::complex<float>>  = "vdot_partial_c64";  ///< The c64 entry.
template <> inline constexpr const char* vdot_name<std::complex<double>> = "vdot_partial_c128";  ///< The c128 entry.

/// Trace partial-sum kernel name per element type.
template <class T> inline constexpr const char* trace_name = nullptr;
template <> inline constexpr const char* trace_name<float>  = "trace_partial_f32";  ///< The f32 entry.
template <> inline constexpr const char* trace_name<double> = "trace_partial_f64";  ///< The f64 entry.
template <> inline constexpr const char* trace_name<std::complex<float>>  = "trace_partial_c64";  ///< The c64 entry.
template <> inline constexpr const char* trace_name<std::complex<double>> = "trace_partial_c128";  ///< The c128 entry.

/// Two-stage dot partial kernel names (G workgroups x 256, groupshared tree-reduce — the
/// large-n path; the routine picks stage-1 vs stage-2 by size).
template <class T> inline constexpr const char* dot2_name = nullptr;
template <> inline constexpr const char* dot2_name<float>  = "dot_partial2_f32";  ///< The f32 entry.
template <> inline constexpr const char* dot2_name<double> = "dot_partial2_f64";  ///< The f64 entry.
template <> inline constexpr const char* dot2_name<std::complex<float>>  = "dot_partial2_c64";  ///< The c64 entry.
template <> inline constexpr const char* dot2_name<std::complex<double>> = "dot_partial2_c128";  ///< The c128 entry.
/// Two-stage conjugating vdot partial kernel name — complex elements only.
template <class T> inline constexpr const char* vdot2_name = nullptr;
template <> inline constexpr const char* vdot2_name<std::complex<float>>  = "vdot_partial2_c64";  ///< The c64 entry.
template <> inline constexpr const char* vdot2_name<std::complex<double>> = "vdot_partial2_c128";  ///< The c128 entry.
/// Two-stage sum partial kernel name (the large-n reduction path).
template <class T> inline constexpr const char* sum2_name = nullptr;
template <> inline constexpr const char* sum2_name<float>  = "sum_partial2_f32";  ///< The f32 entry.
template <> inline constexpr const char* sum2_name<double> = "sum_partial2_f64";  ///< The f64 entry.

/// On-device partial finalizer (one 256-thread group tree-sums G partials into out[0]).
template <class T> inline constexpr const char* finalize_name = nullptr;
template <> inline constexpr const char* finalize_name<float>  = "finalize_partials_f32";  ///< The f32 entry.
template <> inline constexpr const char* finalize_name<double> = "finalize_partials_f64";  ///< The f64 entry.
template <> inline constexpr const char* finalize_name<std::complex<float>>  = "finalize_partials_c64";  ///< The c64 entry.
template <> inline constexpr const char* finalize_name<std::complex<double>> = "finalize_partials_c128";  ///< The c128 entry.

/// The two-stage reduction geometry: n above this uses the G-group tree kernels; G is capped.
inline constexpr std::size_t kTwoStageMin = 1u << 16;
/// The cap on tree-reducing workgroups G in the two-stage reduction geometry.
inline constexpr std::uint32_t kMaxGroups = 1024;

/// Sum partial-sum kernel name per element type (the deterministic reduction contract).
template <class T> inline constexpr const char* sum_name = nullptr;
template <> inline constexpr const char* sum_name<float>  = "sum_partial_f32";  ///< The f32 entry.
template <> inline constexpr const char* sum_name<double> = "sum_partial_f64";  ///< The f64 entry.

/// Fused axpy (out = α·x + y) kernel name per element type.
template <class T> inline constexpr const char* axpy_name = nullptr;
template <> inline constexpr const char* axpy_name<float>  = "axpy_f32";  ///< The f32 entry.
template <> inline constexpr const char* axpy_name<double> = "axpy_f64";  ///< The f64 entry.

/// Batched 2-D im2col gather kernel name per element type (real elements — the conv-support
/// tier; dims = {B, C, H, W, KH, KW, OH, OW, stride, pad}).
template <class T> inline constexpr const char* im2col2d_name = nullptr;
template <> inline constexpr const char* im2col2d_name<float>  = "im2col2d_f32";  ///< The f32 entry.
template <> inline constexpr const char* im2col2d_name<double> = "im2col2d_f64";  ///< The f64 entry.

/// Batched 2-D col2im kernel name per element type (the gather-form adjoint; same dims).
template <class T> inline constexpr const char* col2im2d_name = nullptr;
template <> inline constexpr const char* col2im2d_name<float>  = "col2im2d_f32";  ///< The f32 entry.
template <> inline constexpr const char* col2im2d_name<double> = "col2im2d_f64";  ///< The f64 entry.

/// Fused conv forward epilogue (bias + activation + layout transpose) kernel name per element
/// type; dims = {B, F, OO, act} with act 0 = identity, 1 = relu, 2 = tanh, 3 = sigmoid.
template <class T> inline constexpr const char* conv_bias_act_name = nullptr;
template <> inline constexpr const char* conv_bias_act_name<float>  = "conv_bias_act_f32";  ///< The f32 entry.
template <> inline constexpr const char* conv_bias_act_name<double> = "conv_bias_act_f64";  ///< The f64 entry.

/// Fused conv backward epilogue (chain rule from the output + layout transpose) kernel name per
/// element type; same dims as conv_bias_act.
template <class T> inline constexpr const char* conv_act_grad_name = nullptr;
template <> inline constexpr const char* conv_act_grad_name<float>  = "conv_act_grad_f32";  ///< The f32 entry.
template <> inline constexpr const char* conv_act_grad_name<double> = "conv_act_grad_f64";  ///< The f64 entry.

/// Elementwise binary kernel names per element type, indexed by the shared operator enum: the
/// array⊗array family (ew_*) and the array⊗scalar family (ews_*, with the swap flag for the
/// reversed s−a / s÷a forms). One table per family keeps every dispatch a straight lookup.
enum class EwOp : std::uint32_t { add = 0, sub = 1, mul = 2, div = 3 };
/// The array⊗array binary kernel-name table for element `T`, indexed by EwOp.
template <class T> inline constexpr const char* const* ew_names = nullptr;
/// The f32 array⊗array table (EwOp order).
inline constexpr const char* ew_names_f32[4] = {"ew_add_f32", "ew_sub_f32", "ew_mul_f32",
                                                "ew_div_f32"};
/// The f64 array⊗array table (EwOp order).
inline constexpr const char* ew_names_f64[4] = {"ew_add_f64", "ew_sub_f64", "ew_mul_f64",
                                                "ew_div_f64"};
template <> inline constexpr const char* const* ew_names<float>  = ew_names_f32;  ///< The f32 entry.
template <> inline constexpr const char* const* ew_names<double> = ew_names_f64;  ///< The f64 entry.
/// The f32 array⊗scalar table (EwOp order).
inline constexpr const char* ews_names_f32[4] = {"ews_add_f32", "ews_sub_f32", "ews_mul_f32",
                                                 "ews_div_f32"};
/// The f64 array⊗scalar table (EwOp order).
inline constexpr const char* ews_names_f64[4] = {"ews_add_f64", "ews_sub_f64", "ews_mul_f64",
                                                 "ews_div_f64"};
/// The array⊗scalar binary kernel-name table for element `T`, indexed by EwOp.
template <class T> inline constexpr const char* const* ews_names = nullptr;
template <> inline constexpr const char* const* ews_names<float>  = ews_names_f32;  ///< The f32 entry.
template <> inline constexpr const char* const* ews_names<double> = ews_names_f64;  ///< The f64 entry.

/// Constant-fill (out[i] = s) — the factories' device-side zeros/ones/full (no host staging
/// vector, no PCIe upload). Real elements only, mirroring the ews family.
template <class T> inline constexpr const char* fill_name = nullptr;
template <> inline constexpr const char* fill_name<float>  = "fill_f32";  ///< The f32 entry.
template <> inline constexpr const char* fill_name<double> = "fill_f64";  ///< The f64 entry.

/// Elementwise unary kernel names per element type, indexed by the shared enum.
enum class EwFn : std::uint32_t { neg = 0, abs = 1, sqrt = 2, exp = 3, log = 4 };
/// The f32 unary table (EwFn order).
inline constexpr const char* ewu_names_f32[5] = {"ew_neg_f32", "ew_abs_f32", "ew_sqrt_f32",
                                                 "ew_exp_f32", "ew_log_f32"};
/// The f64 unary table (EwFn order; exp/log dispatch is host-evaluated — see elementwise.hpp).
inline constexpr const char* ewu_names_f64[5] = {"ew_neg_f64", "ew_abs_f64", "ew_sqrt_f64",
                                                 "ew_exp_f64", "ew_log_f64"};
/// The unary kernel-name table for element `T`, indexed by EwFn.
template <class T> inline constexpr const char* const* ewu_names = nullptr;
template <> inline constexpr const char* const* ewu_names<float>  = ewu_names_f32;  ///< The f32 entry.
template <> inline constexpr const char* const* ewu_names<double> = ewu_names_f64;  ///< The f64 entry.

#if !defined(__APPLE__) && !defined(CHEATAH_GPU_LINALG_VULKAN)
namespace emu = cheatah::gpu::metal::emulated;

/// CPU GEMM stand-in mirroring linalg.slang's register-tiled `gemm` result (the tiling is a
/// device optimization; the product is the same): row-major C[M×N] = A[M×K]·B[K×N],
/// dims = {M, N, K}. The kernel's workgroup grid always covers all of C, so the stand-in iterates
/// the dims directly (the DispatchShape is block-granular for this kernel).
/// @param b  The bound buffer pointers, in the kernel's binding order (dims last unless noted).
/// @param n The binding count (the stand-in is a no-op when fewer than expected).
/// @complexity O(M·N·K).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:matmul
template <class T>
inline void gemm_emulated(void** b, unsigned n, const emu::DispatchShape& /*shape*/) {
    if (n < 4) return;
    const T* A = static_cast<const T*>(b[0]);
    const T* B = static_cast<const T*>(b[1]);
    T* C       = static_cast<T*>(b[2]);
    const std::uint32_t* dims = static_cast<const std::uint32_t*>(b[3]);
    const std::uint32_t M = dims[0], N = dims[1], K = dims[2];
    for (std::uint32_t row = 0; row < M; ++row)
        for (std::uint32_t col = 0; col < N; ++col) {
            T acc = T{};
            for (std::uint32_t k = 0; k < K; ++k)
                acc += A[static_cast<std::size_t>(row) * K + k] *
                       B[static_cast<std::size_t>(k) * N + col];
            C[static_cast<std::size_t>(row) * N + col] = acc;
        }
}

/// CPU stand-in for `gemm_batched`: C[z] = A[z]·B[z] per batch layer z of the 3-D grid,
/// dims = {M, N, K, B}.
/// @param b  The bound buffer pointers, in the kernel's binding order (dims last unless noted).
/// @param n The binding count (the stand-in is a no-op when fewer than expected).
/// @param shape The dispatch thread grid (its height/width/depth bound the loops).
/// @complexity O(B·M·N·K).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:matmul_batched
template <class T>
inline void gemm_batched_emulated(void** b, unsigned n, const emu::DispatchShape& shape) {
    if (n < 4) return;
    const T* A = static_cast<const T*>(b[0]);
    const T* B = static_cast<const T*>(b[1]);
    T* C       = static_cast<T*>(b[2]);
    const std::uint32_t* dims = static_cast<const std::uint32_t*>(b[3]);
    const std::uint32_t M = dims[0], N = dims[1], K = dims[2], Bn = dims[3];
    for (unsigned long z = 0; z < shape.threads.depth && z < Bn; ++z) {
        const std::size_t ba = static_cast<std::size_t>(z) * M * K;
        const std::size_t bb = static_cast<std::size_t>(z) * K * N;
        const std::size_t bc = static_cast<std::size_t>(z) * M * N;
        for (unsigned long row = 0; row < shape.threads.height && row < M; ++row)
            for (unsigned long col = 0; col < shape.threads.width; ++col) {
                if (col >= N) continue;
                T acc = T{};
                for (std::uint32_t k = 0; k < K; ++k)
                    acc += A[ba + static_cast<std::size_t>(row) * K + k] *
                           B[bb + static_cast<std::size_t>(k) * N + col];
                C[bc + static_cast<std::size_t>(row) * N + col] = acc;
            }
    }
}

/// CPU stand-in for `outer`: out[n×m] = a_i·b_j, dims = {n, m}.
/// @param b  The bound buffer pointers, in the kernel's binding order (dims last unless noted).
/// @param n The binding count (the stand-in is a no-op when fewer than expected).
/// @param shape The dispatch thread grid (its height/width/depth bound the loops).
/// @complexity O(n·m).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:outer
template <class T>
inline void outer_emulated(void** b, unsigned n, const emu::DispatchShape& shape) {
    if (n < 4) return;
    const T* a = static_cast<const T*>(b[0]);
    const T* v = static_cast<const T*>(b[1]);
    T* out     = static_cast<T*>(b[2]);
    const std::uint32_t* dims = static_cast<const std::uint32_t*>(b[3]);
    const std::uint32_t rows = dims[0], cols = dims[1];
    for (unsigned long i = 0; i < shape.threads.height && i < rows; ++i)
        for (unsigned long j = 0; j < shape.threads.width; ++j)
            if (j < cols) out[i * cols + j] = a[i] * v[j];
}

/// is_cplx_v<T>: local complex detection for the stand-ins (mirrors the kernels' CONJ seam).
template <class T> inline constexpr bool is_cplx_v = false;
template <class U> inline constexpr bool is_cplx_v<std::complex<U>> = true;  ///< The complex case.

/// CPU stand-in for `transpose`: out[c×r] = in[r×c]ᵀ, dims = {r, c} — CONJUGATED for a complex
/// element (the Hermitian adjoint), exactly like the device kernel's CONJ seam.
/// @param b  The bound buffer pointers, in the kernel's binding order (dims last unless noted).
/// @param n The binding count (the stand-in is a no-op when fewer than expected).
/// @param shape The dispatch thread grid (its height/width/depth bound the loops).
/// @complexity O(r·c).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:transpose
template <class T>
inline void transpose_emulated(void** b, unsigned n, const emu::DispatchShape& shape) {
    if (n < 3) return;
    const T* in = static_cast<const T*>(b[0]);
    T* out      = static_cast<T*>(b[1]);
    const std::uint32_t* dims = static_cast<const std::uint32_t*>(b[2]);
    const std::uint32_t r = dims[0], c = dims[1];
    for (unsigned long i = 0; i < shape.threads.height && i < r; ++i)
        for (unsigned long j = 0; j < shape.threads.width; ++j)
            if (j < c) {
                if constexpr (is_cplx_v<T>) out[j * r + i] = std::conj(in[i * c + j]);
                else out[j * r + i] = in[i * c + j];
            }
}

/// CPU stand-in for the conjugating `vdot_partial` (complex elements): partial[g] = Σ conj(a[k])·b[k].
/// @param b  The bound buffer pointers, in the kernel's binding order (dims last unless noted).
/// @param nb The binding count (the stand-in is a no-op when fewer than expected).
/// @param width The 1-D dispatch width (how many partial threads are emulated).
/// @complexity O(n).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:complex
template <class T>
inline void vdot_partial_emulated(void** b, unsigned nb, unsigned long width) {
    if (nb < 4) return;
    const T* a = static_cast<const T*>(b[0]);
    const T* v = static_cast<const T*>(b[1]);
    T* partial = static_cast<T*>(b[2]);
    const std::uint32_t* dims = static_cast<const std::uint32_t*>(b[3]);
    const std::uint32_t n = dims[0], P = dims[1];
    for (unsigned long g = 0; g < width && g < P; ++g) {
        T acc = T{};
        for (std::uint32_t k = static_cast<std::uint32_t>(g); k < n; k += P)
            acc += std::conj(a[k]) * v[k];
        partial[g] = acc;
    }
}

/// CPU stand-in for `kron`: K[(ar·br)×(ac·bc)] = A⊗B, dims = {ar, ac, br, bc}.
/// @param b  The bound buffer pointers, in the kernel's binding order (dims last unless noted).
/// @param n The binding count (the stand-in is a no-op when fewer than expected).
/// @param shape The dispatch thread grid (its height/width/depth bound the loops).
/// @complexity O(n⁴) in the output area.
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:kron
template <class T>
inline void kron_emulated(void** b, unsigned n, const emu::DispatchShape& shape) {
    if (n < 4) return;
    const T* A = static_cast<const T*>(b[0]);
    const T* B = static_cast<const T*>(b[1]);
    T* K       = static_cast<T*>(b[2]);
    const std::uint32_t* dims = static_cast<const std::uint32_t*>(b[3]);
    const std::uint32_t ar = dims[0], ac = dims[1], br = dims[2], bc = dims[3];
    const std::uint32_t rows = ar * br, cols = ac * bc;
    for (unsigned long i = 0; i < shape.threads.height && i < rows; ++i)
        for (unsigned long j = 0; j < shape.threads.width; ++j) {
            if (j >= cols) continue;
            const std::uint32_t ai = static_cast<std::uint32_t>(i) / br;
            const std::uint32_t bi = static_cast<std::uint32_t>(i) % br;
            const std::uint32_t aj = static_cast<std::uint32_t>(j) / bc;
            const std::uint32_t bj = static_cast<std::uint32_t>(j) % bc;
            K[i * cols + j] = A[ai * ac + aj] * B[bi * bc + bj];
        }
}

/// CPU stand-in for `dot_partial`: thread g writes partial[g] = Σ a[k]·b[k], k = g, g+P, …;
/// dims = {n, P}. Bit-identical to the device kernel — the same strided association order.
/// @param b  The bound buffer pointers, in the kernel's binding order (dims last unless noted).
/// @param nb The binding count (the stand-in is a no-op when fewer than expected).
/// @param width The 1-D dispatch width (how many partial threads are emulated).
/// @complexity O(n).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:dot
template <class T>
inline void dot_partial_emulated(void** b, unsigned nb, unsigned long width) {
    if (nb < 4) return;
    const T* a = static_cast<const T*>(b[0]);
    const T* v = static_cast<const T*>(b[1]);
    T* partial = static_cast<T*>(b[2]);
    const std::uint32_t* dims = static_cast<const std::uint32_t*>(b[3]);
    const std::uint32_t n = dims[0], P = dims[1];
    for (unsigned long g = 0; g < width && g < P; ++g) {
        T acc = T{};
        for (std::uint32_t k = static_cast<std::uint32_t>(g); k < n; k += P) acc += a[k] * v[k];
        partial[g] = acc;
    }
}

/// CPU stand-in for `trace_partial`: partial[g] = Σ a[k·(c+1)], k = g, g+P, …; dims = {m, c, P}.
/// @param b  The bound buffer pointers, in the kernel's binding order (dims last unless noted).
/// @param nb The binding count (the stand-in is a no-op when fewer than expected).
/// @param width The 1-D dispatch width (how many partial threads are emulated).
/// @complexity O(m).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:trace
template <class T>
inline void trace_partial_emulated(void** b, unsigned nb, unsigned long width) {
    if (nb < 3) return;
    const T* a = static_cast<const T*>(b[0]);
    T* partial = static_cast<T*>(b[1]);
    const std::uint32_t* dims = static_cast<const std::uint32_t*>(b[2]);
    const std::uint32_t m = dims[0], c = dims[1], P = dims[2];
    for (unsigned long g = 0; g < width && g < P; ++g) {
        T acc = T{};
        for (std::uint32_t k = static_cast<std::uint32_t>(g); k < m; k += P)
            acc += a[static_cast<std::size_t>(k) * (c + 1)];
        partial[g] = acc;
    }
}

/// CPU stand-in for the ew_* binary family: out[i] = a[i] OP b[i], dims = {n}; the operator is a
/// template parameter mirroring the per-operator kernel compilation.
/// @param b  The bound buffer pointers, in the kernel's binding order (dims last unless noted).
/// @param nb The binding count (the stand-in is a no-op when fewer than expected).
/// @complexity O(n).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:elementwise
template <class T, EwOp Op>
inline void ew_binary_emulated(void** b, unsigned nb, unsigned long /*width*/) {
    if (nb < 4) return;
    const T* a = static_cast<const T*>(b[0]);
    const T* v = static_cast<const T*>(b[1]);
    T* out     = static_cast<T*>(b[2]);
    const std::uint32_t n = static_cast<const std::uint32_t*>(b[3])[0];
    for (unsigned long i = 0; i < n; ++i) {
        if constexpr (Op == EwOp::add) out[i] = a[i] + v[i];
        else if constexpr (Op == EwOp::sub) out[i] = a[i] - v[i];
        else if constexpr (Op == EwOp::mul) out[i] = a[i] * v[i];
        else out[i] = a[i] / v[i];
    }
}

/// CPU stand-ins for the stream kernels: fill (out[i] = s), copy (bandwidth probe), triad
/// (a = b + s*c, bandwidth probe). dims = {n} in the last binding, like their GPU forms.
/// @param b  The bound buffer pointers, in the kernel's binding order (dims last unless noted).
/// @param nb The binding count (the stand-in is a no-op when fewer than expected).
/// @complexity O(n).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:elementwise
template <class T>
inline void fill_emulated(void** b, unsigned nb, unsigned long /*width*/) {
    if (nb < 3) return;
    T* out = static_cast<T*>(b[0]);
    const T s = static_cast<const T*>(b[1])[0];
    const std::uint32_t n = static_cast<const std::uint32_t*>(b[2])[0];
    for (std::uint32_t i = 0; i < n; ++i) out[i] = s;
}
/// CPU stand-in for `copy` (out[i] = in[i], the bandwidth probe); dims = {n} last.
/// @param b  The bound buffer pointers {in, out, dims}.
/// @param nb The binding count (no-op when fewer than expected).
/// @complexity O(n).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
inline void copy_emulated(void** b, unsigned nb, unsigned long /*width*/) {
    if (nb < 3) return;
    const float* in = static_cast<const float*>(b[0]);
    float* out = static_cast<float*>(b[1]);
    const std::uint32_t n = static_cast<const std::uint32_t*>(b[2])[0];
    for (std::uint32_t i = 0; i < n; ++i) out[i] = in[i];
}
/// CPU stand-in for `triad` (a = x + s·y, the bandwidth probe); dims = {n} last.
/// @param b  The bound buffer pointers {a, x, y, s, dims}.
/// @param nb The binding count (no-op when fewer than expected).
/// @complexity O(n).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
inline void triad_emulated(void** b, unsigned nb, unsigned long /*width*/) {
    if (nb < 5) return;
    float* a = static_cast<float*>(b[0]);
    const float* x = static_cast<const float*>(b[1]);
    const float* y = static_cast<const float*>(b[2]);
    const float s = static_cast<const float*>(b[3])[0];
    const std::uint32_t n = static_cast<const std::uint32_t*>(b[4])[0];
    for (std::uint32_t i = 0; i < n; ++i) a[i] = x[i] + s * y[i];
}

/// CPU stand-in for the ews_* array⊗scalar family: dims = {n, swap}; swap = 1 computes s OP a.
/// @param b  The bound buffer pointers, in the kernel's binding order (dims last unless noted).
/// @param nb The binding count (the stand-in is a no-op when fewer than expected).
/// @complexity O(n).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:operators
template <class T, EwOp Op>
inline void ews_binary_emulated(void** b, unsigned nb, unsigned long /*width*/) {
    if (nb < 4) return;
    const T* a = static_cast<const T*>(b[0]);
    const T s  = static_cast<const T*>(b[1])[0];
    T* out     = static_cast<T*>(b[2]);
    const std::uint32_t* dims = static_cast<const std::uint32_t*>(b[3]);
    const std::uint32_t n = dims[0], swap = dims[1];
    for (unsigned long i = 0; i < n; ++i) {
        const T x = a[i];
        const T l = swap ? s : x, r = swap ? x : s;
        if constexpr (Op == EwOp::add) out[i] = l + r;
        else if constexpr (Op == EwOp::sub) out[i] = l - r;
        else if constexpr (Op == EwOp::mul) out[i] = l * r;
        else out[i] = l / r;
    }
}

/// CPU stand-in for the ew_* unary family: out[i] = F(a[i]), dims = {n}.
/// @param b  The bound buffer pointers, in the kernel's binding order (dims last unless noted).
/// @param nb The binding count (the stand-in is a no-op when fewer than expected).
/// @complexity O(n).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:elementwise
template <class T, EwFn Fn>
inline void ew_unary_emulated(void** b, unsigned nb, unsigned long /*width*/) {
    if (nb < 3) return;
    const T* a = static_cast<const T*>(b[0]);
    T* out     = static_cast<T*>(b[1]);
    const std::uint32_t n = static_cast<const std::uint32_t*>(b[2])[0];
    for (unsigned long i = 0; i < n; ++i) {
        if constexpr (Fn == EwFn::neg) out[i] = -a[i];
        else if constexpr (Fn == EwFn::abs) out[i] = std::abs(a[i]);
        else if constexpr (Fn == EwFn::sqrt) out[i] = std::sqrt(a[i]);
        else if constexpr (Fn == EwFn::exp) out[i] = std::exp(a[i]);
        else out[i] = std::log(a[i]);
    }
}

/// CPU stand-in for `sum_partial`: partial[g] = Σ a[k], k = g, g+P, …; dims = {n, P} — the same
/// strided association order as the device kernel (bit-identical).
/// @param b  The bound buffer pointers, in the kernel's binding order (dims last unless noted).
/// @param nb The binding count (the stand-in is a no-op when fewer than expected).
/// @param width The 1-D dispatch width (how many partial threads are emulated).
/// @complexity O(n).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:elementwise
template <class T>
inline void sum_partial_emulated(void** b, unsigned nb, unsigned long width) {
    if (nb < 3) return;
    const T* a = static_cast<const T*>(b[0]);
    T* partial = static_cast<T*>(b[1]);
    const std::uint32_t* dims = static_cast<const std::uint32_t*>(b[2]);
    const std::uint32_t n = dims[0], P = dims[1];
    for (unsigned long g = 0; g < width && g < P; ++g) {
        T acc = T{};
        for (std::uint32_t k = static_cast<std::uint32_t>(g); k < n; k += P) acc += a[k];
        partial[g] = acc;
    }
}

/// CPU stand-in for the TWO-STAGE `dot_partial2`/`vdot_partial2`: reproduces the kernel's exact
/// association order — per-thread strided accumulation, then the fixed groupshared tree.
/// @param b  The bound buffer pointers, in the kernel's binding order (dims last unless noted).
/// @param nb The binding count (the stand-in is a no-op when fewer than expected).
/// @complexity O(n) accumulation + O(G·256) tree folds.
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:dot
template <class T, bool Conj>
inline void dot_partial2_emulated(void** b, unsigned nb, unsigned long /*width*/) {
    if (nb < 4) return;
    const T* a = static_cast<const T*>(b[0]);
    const T* v = static_cast<const T*>(b[1]);
    T* partial = static_cast<T*>(b[2]);
    const std::uint32_t* dims = static_cast<const std::uint32_t*>(b[3]);
    const std::uint32_t n = dims[0], G = dims[1];
    const std::uint32_t stride = G * 256;
    const std::uint32_t n4 = n >> 2, tail = n & 3;
    auto term = [&](std::uint32_t k) {
        if constexpr (Conj) return std::conj(a[k]) * v[k];
        else return a[k] * v[k];
    };
    for (std::uint32_t g = 0; g < G; ++g) {
        T red[256];
        for (std::uint32_t t = 0; t < 256; ++t) {
            T acc{};
            if constexpr (is_cplx_v<T>) {
                // Complex device kernel keeps the scalar strided loop — mirror it exactly.
                for (std::uint32_t k = g * 256 + t; k < n; k += stride) acc += term(k);
            } else {
                // vec4 lane accumulation folded (x+y)+(z+w) — the device kernel's exact order.
                T l0{}, l1{}, l2{}, l3{};
                for (std::uint32_t k4 = g * 256 + t; k4 < n4; k4 += stride) {
                    l0 += term(k4 * 4 + 0);
                    l1 += term(k4 * 4 + 1);
                    l2 += term(k4 * 4 + 2);
                    l3 += term(k4 * 4 + 3);
                }
                acc = (l0 + l1) + (l2 + l3);
                if (g == 0 && t < tail) acc += term(n4 * 4 + t);
            }
            red[t] = acc;
        }
        for (std::uint32_t s2 = 128; s2 > 0; s2 >>= 1)
            for (std::uint32_t t = 0; t < s2; ++t) red[t] += red[t + s2];
        partial[g] = red[0];
    }
}

/// CPU stand-in for the TWO-STAGE `sum_partial2` (same contract over one input).
/// @param b  The bound buffer pointers, in the kernel's binding order (dims last unless noted).
/// @param nb The binding count (the stand-in is a no-op when fewer than expected).
/// @complexity O(n) accumulation + O(G·256) tree folds.
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:elementwise
template <class T>
inline void sum_partial2_emulated(void** b, unsigned nb, unsigned long /*width*/) {
    if (nb < 3) return;
    const T* a = static_cast<const T*>(b[0]);
    T* partial = static_cast<T*>(b[1]);
    const std::uint32_t* dims = static_cast<const std::uint32_t*>(b[2]);
    const std::uint32_t n = dims[0], G = dims[1];
    const std::uint32_t stride = G * 256;
    const std::uint32_t n4 = n >> 2, tail = n & 3;
    for (std::uint32_t g = 0; g < G; ++g) {
        T red[256];
        for (std::uint32_t t = 0; t < 256; ++t) {
            T l0{}, l1{}, l2{}, l3{};
            for (std::uint32_t k4 = g * 256 + t; k4 < n4; k4 += stride) {
                l0 += a[k4 * 4 + 0];
                l1 += a[k4 * 4 + 1];
                l2 += a[k4 * 4 + 2];
                l3 += a[k4 * 4 + 3];
            }
            T acc = (l0 + l1) + (l2 + l3);
            if (g == 0 && t < tail) acc += a[n4 * 4 + t];
            red[t] = acc;
        }
        for (std::uint32_t s2 = 128; s2 > 0; s2 >>= 1)
            for (std::uint32_t t = 0; t < s2; ++t) red[t] += red[t + s2];
        partial[g] = red[0];
    }
}

/// CPU stand-in for `finalize_partials`: the same single-group strided + tree fold into out[0].
/// @param b  The bound buffer pointers, in the kernel's binding order (dims last unless noted).
/// @param nb The binding count (the stand-in is a no-op when fewer than expected).
/// @complexity O(G) + a fixed 256-wide tree.
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:dot
template <class T>
inline void finalize_partials_emulated(void** b, unsigned nb, unsigned long /*width*/) {
    if (nb < 3) return;
    const T* partials = static_cast<const T*>(b[0]);
    T* out = static_cast<T*>(b[1]);
    const std::uint32_t G = static_cast<const std::uint32_t*>(b[2])[0];
    T red[256];
    for (std::uint32_t t = 0; t < 256; ++t) {
        T acc = T{};
        for (std::uint32_t k = t; k < G; k += 256) acc += partials[k];
        red[t] = acc;
    }
    for (std::uint32_t s2 = 128; s2 > 0; s2 >>= 1)
        for (std::uint32_t t = 0; t < s2; ++t) red[t] += red[t + s2];
    out[0] = red[0];
}

/// CPU stand-in for the fused `axpy`: out[i] = α·x[i] + y[i], dims = {n}.
/// @param b  The bound buffer pointers, in the kernel's binding order (dims last unless noted).
/// @param nb The binding count (the stand-in is a no-op when fewer than expected).
/// @complexity O(n).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:elementwise
template <class T>
inline void axpy_emulated(void** b, unsigned nb, unsigned long /*width*/) {
    if (nb < 5) return;
    const T* x = static_cast<const T*>(b[0]);
    const T* y = static_cast<const T*>(b[1]);
    const T alpha = static_cast<const T*>(b[2])[0];
    T* out = static_cast<T*>(b[3]);
    const std::uint32_t n = static_cast<const std::uint32_t*>(b[4])[0];
    for (unsigned long i = 0; i < n; ++i) out[i] = alpha * x[i] + y[i];
}

/// The conv epilogues' scalar activation (matches the Slang cba_act table exactly): 0 = identity,
/// 1 = relu, 2 = tanh, 3 = sigmoid.
/// @param z   The pre-activation value.
/// @param act The activation code (0–3; validated by the dispatching routine).
/// @return f(z) for the selected activation.
/// @complexity O(1).
/// @alloc none. @gpualloc none.
/// @test gpu:conv
template <class T>
inline T conv_act_f(T z, std::uint32_t act) {
    if (act == 1) return z > T(0) ? z : T(0);
    if (act == 2) return std::tanh(z);
    if (act == 3) return T(1) / (T(1) + std::exp(-z));
    return z;
}
/// The conv epilogues' scalar derivative FROM THE OUTPUT a = f(z) (matches cag_dact):
/// identity' = 1, relu' = [a > 0], tanh' = 1 − a², sigmoid' = a(1 − a).
/// @param a   The activation OUTPUT f(z) (not the pre-activation).
/// @param act The activation code (0–3; validated by the dispatching routine).
/// @return f′(z) expressed in terms of a.
/// @complexity O(1).
/// @alloc none. @gpualloc none.
/// @test gpu:conv
template <class T>
inline T conv_act_df(T a, std::uint32_t act) {
    if (act == 1) return a > T(0) ? T(1) : T(0);
    if (act == 2) return T(1) - a * a;
    if (act == 3) return a * (T(1) - a);
    return T(1);
}

/// CPU stand-in for `im2col2d`: col[((c·KH+kh)·KW+kw)·B·OO + b·OO + oh·OW + ow] = x in-bounds,
/// 0 for padding (FULL overwrite); dims = {B, C, H, W, KH, KW, OH, OW, stride, pad}. The same
/// per-element gather as the device kernel, iterated over the col linear index.
/// @param b  The bound buffer pointers {x, col, dims}.
/// @param nb The binding count (the stand-in is a no-op when fewer than expected).
/// @complexity O(B·C·KH·KW·OH·OW).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:conv
template <class T>
inline void im2col2d_emulated(void** b, unsigned nb, unsigned long /*width*/) {
    if (nb < 3) return;
    const T* x = static_cast<const T*>(b[0]);
    T* col     = static_cast<T*>(b[1]);
    const std::uint32_t* dims = static_cast<const std::uint32_t*>(b[2]);
    const std::uint32_t B = dims[0], C = dims[1], H = dims[2], W = dims[3];
    const std::uint32_t KH = dims[4], KW = dims[5], OH = dims[6], OW = dims[7];
    const std::uint32_t stride = dims[8], pad = dims[9];
    const std::size_t OO = static_cast<std::size_t>(OH) * OW, BOO = B * OO;
    const std::size_t n = static_cast<std::size_t>(C) * KH * KW * BOO;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t ow = i % OW, oh = (i / OW) % OH, bb = (i / OO) % B;
        const std::uint32_t kw = (i / BOO) % KW, kh = (i / (BOO * KW)) % KH;
        const std::uint32_t c = static_cast<std::uint32_t>(i / (BOO * KW * KH));
        const long long ih = static_cast<long long>(oh) * stride + kh - pad;
        const long long iw = static_cast<long long>(ow) * stride + kw - pad;
        const bool inside = ih >= 0 && ih < H && iw >= 0 && iw < W;
        col[i] = inside ? x[(static_cast<std::size_t>(bb) * C + c) * H * W + ih * W + iw] : T(0);
    }
}

/// CPU stand-in for `col2im2d`: dx[i] = Σ over its ≤ KH·KW contributing dcol cells, kh-major —
/// the gather-form adjoint, bit-identical to the reference scatter-add's association order.
/// Same dims as im2col2d.
/// @param b  The bound buffer pointers {dcol, dx, dims}.
/// @param nb The binding count (the stand-in is a no-op when fewer than expected).
/// @complexity O(B·C·H·W·KH·KW) bound checks (≤ KH·KW adds per input cell).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:conv
template <class T>
inline void col2im2d_emulated(void** b, unsigned nb, unsigned long /*width*/) {
    if (nb < 3) return;
    const T* dcol = static_cast<const T*>(b[0]);
    T* dx         = static_cast<T*>(b[1]);
    const std::uint32_t* dims = static_cast<const std::uint32_t*>(b[2]);
    const std::uint32_t B = dims[0], C = dims[1], H = dims[2], W = dims[3];
    const std::uint32_t KH = dims[4], KW = dims[5], OH = dims[6], OW = dims[7];
    const std::uint32_t stride = dims[8], pad = dims[9];
    const std::size_t OO = static_cast<std::size_t>(OH) * OW, BOO = B * OO;
    const std::size_t n = static_cast<std::size_t>(B) * C * H * W;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t iw = i % W, ih = (i / W) % H, c = (i / (static_cast<std::size_t>(H) * W)) % C;
        const std::uint32_t bb = static_cast<std::uint32_t>(i / (static_cast<std::size_t>(C) * H * W));
        T acc{};
        for (std::uint32_t kh = 0; kh < KH; ++kh) {
            const long long ohs = static_cast<long long>(ih) + pad - kh;
            if (ohs < 0 || ohs % stride != 0) continue;
            const std::uint32_t oh = static_cast<std::uint32_t>(ohs / stride);
            if (oh >= OH) continue;
            for (std::uint32_t kw = 0; kw < KW; ++kw) {
                const long long ows = static_cast<long long>(iw) + pad - kw;
                if (ows < 0 || ows % stride != 0) continue;
                const std::uint32_t ow = static_cast<std::uint32_t>(ows / stride);
                if (ow >= OW) continue;
                acc += dcol[(static_cast<std::size_t>(c) * KH * KW + kh * KW + kw) * BOO +
                            bb * OO + oh * OW + ow];
            }
        }
        dx[i] = acc;
    }
}

/// CPU stand-in for `conv_bias_act`: a[(b·F+f)·OO + o] = act(yc[f·B·OO + b·OO + o] + bias[f]);
/// dims = {B, F, OO, act}.
/// @param b  The bound buffer pointers {yc, bias, a, dims}.
/// @param nb The binding count (the stand-in is a no-op when fewer than expected).
/// @complexity O(B·F·OO).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:conv
template <class T>
inline void conv_bias_act_emulated(void** b, unsigned nb, unsigned long /*width*/) {
    if (nb < 4) return;
    const T* yc   = static_cast<const T*>(b[0]);
    const T* bias = static_cast<const T*>(b[1]);
    T* a          = static_cast<T*>(b[2]);
    const std::uint32_t* dims = static_cast<const std::uint32_t*>(b[3]);
    const std::uint32_t B = dims[0], F = dims[1], OO = dims[2], act = dims[3];
    const std::size_t n = static_cast<std::size_t>(B) * F * OO;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t o = i % OO, f = (i / OO) % F;
        const std::uint32_t bb = static_cast<std::uint32_t>(i / (static_cast<std::size_t>(F) * OO));
        a[i] = conv_act_f(yc[(static_cast<std::size_t>(f) * B + bb) * OO + o] + bias[f], act);
    }
}

/// CPU stand-in for `conv_act_grad`: dyc[f·B·OO + b·OO + o] = d[(b·F+f)·OO + o] ·
/// act'(a[(b·F+f)·OO + o]) with the derivative FROM THE OUTPUT; dims = {B, F, OO, act}.
/// @param b  The bound buffer pointers {d, a, dyc, dims}.
/// @param nb The binding count (the stand-in is a no-op when fewer than expected).
/// @complexity O(B·F·OO).
/// @alloc none — writes in place through the bound buffers. @gpualloc none.
/// @test gpu:conv
template <class T>
inline void conv_act_grad_emulated(void** b, unsigned nb, unsigned long /*width*/) {
    if (nb < 4) return;
    const T* d  = static_cast<const T*>(b[0]);
    const T* a  = static_cast<const T*>(b[1]);
    T* dyc      = static_cast<T*>(b[2]);
    const std::uint32_t* dims = static_cast<const std::uint32_t*>(b[3]);
    const std::uint32_t B = dims[0], F = dims[1], OO = dims[2], act = dims[3];
    const std::size_t BOO = static_cast<std::size_t>(B) * OO;
    const std::size_t n = static_cast<std::size_t>(F) * BOO;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t o = i % OO, bb = (i / OO) % B;
        const std::uint32_t f = static_cast<std::uint32_t>(i / BOO);
        const std::size_t j = (static_cast<std::size_t>(bb) * F + f) * OO + o;
        dyc[i] = d[j] * conv_act_df(a[j], act);
    }
}

/// Register every emulated stand-in with the software Metal device, both element types (the
/// device kernels lack a Metal f64 — Metal has no double — but the emulator provides it, which is
/// how cheatah's default double element is exercised on this backend). Idempotent: registering a
/// name again just overwrites it with the same pointer.
inline void register_emulated_kernels() {
    emu::register_kernel(gemm_fast_name<float>,  &gemm_emulated<float>);
    emu::register_kernel(gemm_fast64_name<float>, &gemm_emulated<float>);
    emu::register_kernel(gemm_fast_name<double>, &gemm_emulated<double>);
    emu::register_kernel(gemm_edge_name<float>,  &gemm_emulated<float>);
    emu::register_kernel(gemm_edge_name<double>, &gemm_emulated<double>);
    emu::register_kernel(gemm_batched_name<float>,  &gemm_batched_emulated<float>);
    emu::register_kernel(gemm_batched_name<double>, &gemm_batched_emulated<double>);
    emu::register_kernel(outer_name<float>,      &outer_emulated<float>);
    emu::register_kernel(outer_name<double>,     &outer_emulated<double>);
    emu::register_kernel(transpose_name<float>,  &transpose_emulated<float>);
    emu::register_kernel(transpose_name<double>, &transpose_emulated<double>);
    emu::register_kernel(kron_name<float>,       &kron_emulated<float>);
    emu::register_kernel(kron_name<double>,      &kron_emulated<double>);
    emu::register_kernel(dot_name<float>,        &dot_partial_emulated<float>);
    emu::register_kernel(dot_name<double>,       &dot_partial_emulated<double>);
    emu::register_kernel(trace_name<float>,      &trace_partial_emulated<float>);
    emu::register_kernel(trace_name<double>,     &trace_partial_emulated<double>);
    using c64 = std::complex<float>;
    using c128 = std::complex<double>;
    emu::register_kernel(gemm_edge_name<c64>,     &gemm_emulated<c64>);
    emu::register_kernel(gemm_edge_name<c128>,    &gemm_emulated<c128>);
    emu::register_kernel(gemm_batched_name<c64>,  &gemm_batched_emulated<c64>);
    emu::register_kernel(gemm_batched_name<c128>, &gemm_batched_emulated<c128>);
    emu::register_kernel(outer_name<c64>,         &outer_emulated<c64>);
    emu::register_kernel(outer_name<c128>,        &outer_emulated<c128>);
    emu::register_kernel(transpose_name<c64>,     &transpose_emulated<c64>);
    emu::register_kernel(transpose_name<c128>,    &transpose_emulated<c128>);
    emu::register_kernel(kron_name<c64>,          &kron_emulated<c64>);
    emu::register_kernel(kron_name<c128>,         &kron_emulated<c128>);
    emu::register_kernel(dot_name<c64>,           &dot_partial_emulated<c64>);
    emu::register_kernel(dot_name<c128>,          &dot_partial_emulated<c128>);
    emu::register_kernel(vdot_name<c64>,          &vdot_partial_emulated<c64>);
    emu::register_kernel(vdot_name<c128>,         &vdot_partial_emulated<c128>);
    emu::register_kernel(trace_name<c64>,         &trace_partial_emulated<c64>);
    emu::register_kernel(trace_name<c128>,        &trace_partial_emulated<c128>);
    emu::register_kernel(dot2_name<float>,       &dot_partial2_emulated<float, false>);
    emu::register_kernel(dot2_name<double>,      &dot_partial2_emulated<double, false>);
    emu::register_kernel(dot2_name<c64>,         &dot_partial2_emulated<c64, false>);
    emu::register_kernel(dot2_name<c128>,        &dot_partial2_emulated<c128, false>);
    emu::register_kernel(vdot2_name<c64>,        &dot_partial2_emulated<c64, true>);
    emu::register_kernel(vdot2_name<c128>,       &dot_partial2_emulated<c128, true>);
    emu::register_kernel(sum2_name<float>,       &sum_partial2_emulated<float>);
    emu::register_kernel(sum2_name<double>,      &sum_partial2_emulated<double>);
    emu::register_kernel(finalize_name<float>,   &finalize_partials_emulated<float>);
    emu::register_kernel(finalize_name<double>,  &finalize_partials_emulated<double>);
    emu::register_kernel(finalize_name<c64>,     &finalize_partials_emulated<c64>);
    emu::register_kernel(finalize_name<c128>,    &finalize_partials_emulated<c128>);
    emu::register_kernel(sum_name<float>,        &sum_partial_emulated<float>);
    emu::register_kernel(sum_name<double>,       &sum_partial_emulated<double>);
    emu::register_kernel(axpy_name<float>,       &axpy_emulated<float>);
    emu::register_kernel(axpy_name<double>,      &axpy_emulated<double>);
    emu::register_kernel(im2col2d_name<float>,       &im2col2d_emulated<float>);
    emu::register_kernel(im2col2d_name<double>,      &im2col2d_emulated<double>);
    emu::register_kernel(col2im2d_name<float>,       &col2im2d_emulated<float>);
    emu::register_kernel(col2im2d_name<double>,      &col2im2d_emulated<double>);
    emu::register_kernel(conv_bias_act_name<float>,  &conv_bias_act_emulated<float>);
    emu::register_kernel(conv_bias_act_name<double>, &conv_bias_act_emulated<double>);
    emu::register_kernel(conv_act_grad_name<float>,  &conv_act_grad_emulated<float>);
    emu::register_kernel(conv_act_grad_name<double>, &conv_act_grad_emulated<double>);
    emu::register_kernel("ew_add_f32",  &ew_binary_emulated<float, EwOp::add>);
    emu::register_kernel("ew_sub_f32",  &ew_binary_emulated<float, EwOp::sub>);
    emu::register_kernel("ew_mul_f32",  &ew_binary_emulated<float, EwOp::mul>);
    emu::register_kernel("ew_div_f32",  &ew_binary_emulated<float, EwOp::div>);
    emu::register_kernel("ew_add_f64",  &ew_binary_emulated<double, EwOp::add>);
    emu::register_kernel("ew_sub_f64",  &ew_binary_emulated<double, EwOp::sub>);
    emu::register_kernel("ew_mul_f64",  &ew_binary_emulated<double, EwOp::mul>);
    emu::register_kernel("ew_div_f64",  &ew_binary_emulated<double, EwOp::div>);
    emu::register_kernel("ews_add_f32", &ews_binary_emulated<float, EwOp::add>);
    emu::register_kernel("ews_sub_f32", &ews_binary_emulated<float, EwOp::sub>);
    emu::register_kernel("ews_mul_f32", &ews_binary_emulated<float, EwOp::mul>);
    emu::register_kernel("ews_div_f32", &ews_binary_emulated<float, EwOp::div>);
    emu::register_kernel("ews_add_f64", &ews_binary_emulated<double, EwOp::add>);
    emu::register_kernel("ews_sub_f64", &ews_binary_emulated<double, EwOp::sub>);
    emu::register_kernel("ews_mul_f64", &ews_binary_emulated<double, EwOp::mul>);
    emu::register_kernel("ews_div_f64", &ews_binary_emulated<double, EwOp::div>);
    emu::register_kernel("fill_f32", &fill_emulated<float>);
    emu::register_kernel("fill_f64", &fill_emulated<double>);
    emu::register_kernel("copy_f32", &copy_emulated);
    emu::register_kernel("triad_f32", &triad_emulated);
    emu::register_kernel("ew_neg_f32",  &ew_unary_emulated<float, EwFn::neg>);
    emu::register_kernel("ew_abs_f32",  &ew_unary_emulated<float, EwFn::abs>);
    emu::register_kernel("ew_sqrt_f32", &ew_unary_emulated<float, EwFn::sqrt>);
    emu::register_kernel("ew_exp_f32",  &ew_unary_emulated<float, EwFn::exp>);
    emu::register_kernel("ew_log_f32",  &ew_unary_emulated<float, EwFn::log>);
    emu::register_kernel("ew_neg_f64",  &ew_unary_emulated<double, EwFn::neg>);
    emu::register_kernel("ew_abs_f64",  &ew_unary_emulated<double, EwFn::abs>);
    emu::register_kernel("ew_sqrt_f64", &ew_unary_emulated<double, EwFn::sqrt>);
    emu::register_kernel("ew_exp_f64",  &ew_unary_emulated<double, EwFn::exp>);
    emu::register_kernel("ew_log_f64",  &ew_unary_emulated<double, EwFn::log>);
}
#endif  // metal backend, off Apple

}  // namespace cheatah::gpu::linalg::kernels
