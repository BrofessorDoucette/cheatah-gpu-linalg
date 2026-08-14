// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
#pragma once

/**
 * @file routines.hpp
 * @brief The device linalg routines — same names + signatures as cheatah's, `requires DeviceArray`.
 *
 * cheatah's allocating fronts (in stdlib/linalg/backend.hpp) do the shape-checking and allocate the
 * result, then call the same-named out-param kernel unqualified. Because these overloads live in
 * `device_array`'s namespace and are constrained to `DeviceArray`, ADL selects them for a device
 * operand while the host overloads (constrained to `HostArray`) are non-viable — no CPO, no tag,
 * no ambiguity. So `cheatah::linalg::matmul(dev_a, dev_b)` runs on the GPU with no extra plumbing.
 *
 * The surface this library implements today (real elements, float/double):
 *   - array results: `matmul` (tiled GEMM), `outer`, `conj_transpose`, `kron`;
 *   - scalar-out reductions: `dot`, `vdot`, `inner` (identical for real elements) and `trace`,
 *     computed as DETERMINISTIC partial sums on the device and summed on the host.
 */

#include "cheatah_gpu_linalg/device_array.hpp"
#include "cheatah_gpu_linalg/kernels.hpp"

#include "backend.hpp"   // cheatah stdlib/linalg: the allocating fronts + host kernel declarations

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

namespace cheatah::gpu::linalg {

/// GpuElement<T>: the REAL element types this library has GPU kernels for; constraining here means
/// an unsupported element is a clean "no device overload" rather than a null-named dispatch.
template <class T>
concept GpuElement = std::is_same_v<T, float> || std::is_same_v<T, double>;

/// GpuComplexElement<T>: the complex elements the seam-wired product/reduction kernels support
/// (c64/c128 — float2/double2 on the device, layout-identical to std::complex).
template <class T>
concept GpuComplexElement =
    std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;

/// GpuField<T>: any element with device kernels — the constraint of the seam-wired routines
/// (matmul/outer/conj_transpose/kron/dot/vdot/inner/trace). Elementwise ops stay GpuElement.
template <class T>
concept GpuField = GpuElement<T> || GpuComplexElement<T>;

namespace detail {

/// A scratch device buffer holding `n` uint32 dims — the last binding of every kernel.
/// CONTENT-CACHED: identical dims recur every iteration of any loop (training epochs re-dispatch
/// the same shapes), so repeat calls return the SAME persistent buffer — no pool round-trip, no
/// memcpy. Callers still pass the handle to `ctx().release_buffer`, which ignores cached buffers,
/// keeping every call site uniform. Falls back to a plain pooled buffer past the cache bounds.
inline Buffer* dims_buffer(const std::uint32_t* dims, std::size_t n) {
    struct Key {
        std::array<std::uint32_t, 8> v;
        std::size_t n;
        bool operator==(const Key&) const = default;
    };
    struct Hash {
        std::size_t operator()(const Key& k) const {
            std::size_t h = k.n;
            for (std::size_t i = 0; i < k.n; ++i) h = h * 1000003u + k.v[i];
            return h;
        }
    };
    static std::unordered_map<Key, Buffer*, Hash> cache;
    if (n <= 8 && cache.size() < 4096) {
        Key k{};
        k.n = n;
        std::copy(dims, dims + n, k.v.begin());
        if (auto it = cache.find(k); it != cache.end()) return it->second;
        Buffer* b = ctx().new_cached_buffer(n * sizeof(std::uint32_t));
        std::memcpy(Context::contents(b), dims, n * sizeof(std::uint32_t));
        cache.emplace(k, b);
        return b;
    }
    Buffer* b = ctx().new_buffer(n * sizeof(std::uint32_t));
    std::memcpy(Context::contents(b), dims, n * sizeof(std::uint32_t));
    return b;
}

/// Tree-sum G partials into a single element ON DEVICE (finalize_partials kernel), then download
/// just that one element — the reduction result crosses the bus as sizeof(T), not G·sizeof(T).
/// Tree-sum G partials into result[0] ON DEVICE — the shared tail of every reduction; the
/// device-resident overloads stop here (zero downloads), the host-returning forms download the
/// single element after.
template <class T>
inline void finalize_into(Buffer* partials, std::uint32_t G, Buffer* result) {
    Context& c = ctx();
    const std::uint32_t fdims[1] = {G};
    Buffer* fdimbuf = dims_buffer(fdims, 1);
    Buffer* bind[3] = {partials, result, fdimbuf};
    c.dispatch_blocks_1d(kernels::finalize_name<T>, bind, 3, 1);
    c.release_buffer(fdimbuf);
}

template <class T>
inline T finalize_partials(Buffer* partials, std::uint32_t G) {
    Context& c = ctx();
    Buffer* result = c.new_buffer(sizeof(T));
    finalize_into<T>(partials, G, result);
    T out;
    c.download(result, &out, sizeof(T));
    c.release_buffer(result);
    return out;
}

/// Run the dot-family partial stage (size-picked one- or two-stage kernel) and return the
/// partials buffer + count — shared by the host-returning and device-resident fronts. Lengths
/// are PRE-VALIDATED by the fronts: validating here would be too late — both callers hold
/// allocations/fuse state a throw from this depth would leak (the leak ASan caught).
template <GpuField T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>>
Buffer* dot_partials(const Array<T>& a, const Array<T>& b, const char* kernel,
                     const char* kernel2, std::uint32_t& count) {
    const std::size_t n = cheatah::linalg::vector_len(a);
    Context& c = ctx();
    if (n >= kernels::kTwoStageMin) {
        const std::uint32_t G = static_cast<std::uint32_t>(std::min<std::size_t>(
            (n + kernels::kLocal1d * 8 - 1) / (kernels::kLocal1d * 8), kernels::kMaxGroups));
        const std::uint32_t dims[2] = {static_cast<std::uint32_t>(n), G};
        Buffer* partial = c.new_buffer(G * sizeof(T));
        Buffer* dimbuf = dims_buffer(dims, 2);
        Buffer* bind[6] = {a.buffer(), b.buffer(), partial, dimbuf, a.buffer(), b.buffer()};
        c.dispatch_blocks_1d(kernel2, bind, 6, G);
        c.release_buffer(dimbuf);
        count = G;
        return partial;
    }
    const std::uint32_t P =
        static_cast<std::uint32_t>(std::min<std::size_t>(std::max<std::size_t>(n, 1), kernels::kMaxReduce));
    const std::uint32_t dims[2] = {static_cast<std::uint32_t>(n), P};
    Buffer* partial = c.new_buffer(P * sizeof(T));
    Buffer* dimbuf = dims_buffer(dims, 2);
    Buffer* bind[4] = {a.buffer(), b.buffer(), partial, dimbuf};
    c.dispatch_1d(kernel, bind, 4, P);
    c.release_buffer(dimbuf);
    count = P;
    return partial;
}

/// The shared body of the dot-family reductions (dot/vdot/inner are the same bilinear sum for the
/// real elements this library supports): dispatch P = min(n, kMaxReduce) partial-sum threads, then
/// sum the P partials on the host — deterministic, no atomics.
template <GpuField T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>>
T device_dot(const Array<T>& a, const Array<T>& b, const char* kernel, const char* kernel2) {
    // Validate BEFORE any allocation: a later throw would leak the result buffer.
    if (cheatah::linalg::vector_len(a) != cheatah::linalg::vector_len(b))
        throw std::runtime_error("cheatah-gpu-linalg dot: length mismatch");
    if (cheatah::linalg::vector_len(a) == 0) return T{};
    Context& c = ctx();
    std::uint32_t count = 0;
    Buffer* result = c.new_buffer(sizeof(T));   // allocate BEFORE the fuse (throw-safety)
    c.begin_fuse();                      // partial + finalize: ONE submission (one fence wait)
    Buffer* partial = dot_partials(a, b, kernel, kernel2, count);
    finalize_into<T>(partial, count, result);
    c.end_fuse();
    T sum;
    c.download(result, &sum, sizeof(T));
    c.release_buffer(result);
    c.release_buffer(partial);
    return sum;
}

}  // namespace detail

/**
 * Device GEMM: `out = a @ b`, row-major, `a` is M×K, `b` is K×N, `out` is M×N. Invoked by cheatah's
 * allocating `matmul(a, b)` front (which has already allocated `out`); routes to the fast 128×128
 * (or 64-row) register-tiled kernel when the shape tiles exactly, the guarded 64×64 edge kernel
 * otherwise, and the one-dispatch batched kernel for 3-D [B,M,K]@[B,K,N] operands.
 *
 * @param out The pre-allocated M×N (or B×M×N) device product.
 * @param a   The M×K (or B×M×K) device left operand.
 * @param b   The K×N (or B×K×N) device right operand (throws on an inner-dimension mismatch).
 * @complexity O(M·N·K) device work (O(B·M·N·K) batched) in one blocking dispatch.
 * @alloc none — the dims scratch is a content-cached pooled buffer (repeat shapes reuse the same
 *        handle).
 * @gpualloc none — all three buffers are the caller's.
 * @test gpu:matmul
 * @test gpu:matmul_batched
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && GpuField<T>
void matmul(Array<T>& out, const Array<T>& a, const Array<T>& b) {
    if (a.ndim() == 3) {
        // Batched [B,M,K] @ [B,K,N] (validated by cheatah's front): ONE dispatch, grid z = batch —
        // the whole batch shares a single submit instead of B of them.
        const std::size_t B = a.shape()[0], M = a.shape()[1], K = a.shape()[2];
        const std::size_t N = b.shape()[2];
        if (B == 0 || M == 0 || N == 0) return;
        const std::uint32_t dims[4] = {static_cast<std::uint32_t>(M),
                                       static_cast<std::uint32_t>(N),
                                       static_cast<std::uint32_t>(K),
                                       static_cast<std::uint32_t>(B)};
        detail::Context& c = detail::ctx();
        detail::Buffer* dimbuf = detail::dims_buffer(dims, 4);
        detail::Buffer* bind[4] = {a.buffer(), b.buffer(), out.buffer(), dimbuf};
        c.dispatch_3d(kernels::gemm_batched_name<T>, bind, 4, N, M, B);
        c.release_buffer(dimbuf);
        return;
    }
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::runtime_error("cheatah-gpu-linalg matmul: operands must be 2-D");
    const std::size_t M = a.shape()[0], K = a.shape()[1];
    const std::size_t Kb = b.shape()[0], N = b.shape()[1];
    if (K != Kb)
        throw std::runtime_error("cheatah-gpu-linalg matmul: inner dimensions do not match");
    if (M == 0 || N == 0) return;

    const std::uint32_t dims[3] = {static_cast<std::uint32_t>(M), static_cast<std::uint32_t>(N),
                                   static_cast<std::uint32_t>(K)};
    detail::Context& c = detail::ctx();
    detail::Buffer* dimbuf = detail::dims_buffer(dims, 3);
    detail::Buffer* bind[4] = {a.buffer(), b.buffer(), out.buffer(), dimbuf};
    // Both kernels are BLOCK-indexed. The FAST path (128x128 blocks, double-buffered, vec4
    // loads, no bounds checks) requires real elements and exact tiling; anything else takes the
    // guarded 64x64 edge kernel.
    bool fast = false;
    if constexpr (std::is_same_v<T, float>)   // f64's 8x8 microtile spills registers — edge wins
        fast = (M % kernels::kGemmFastBlock == 0) && (N % kernels::kGemmFastBlock == 0) &&
               (K % kernels::kGemmFastK == 0);
    bool fast64 = false;
    if constexpr (std::is_same_v<T, float>)   // 64-row sibling: M%64 (not %128) rectangles
        fast64 = !fast && (M % 64 == 0) && (N % kernels::kGemmFastBlock == 0) &&
                 (K % kernels::kGemmFastK == 0);
    if (fast) {
        const std::uint32_t gx = static_cast<std::uint32_t>(N / kernels::kGemmFastBlock);
        const std::uint32_t gy = static_cast<std::uint32_t>(M / kernels::kGemmFastBlock);
        c.dispatch_blocks_2d(kernels::gemm_fast_name<T>, bind, 4, gx, gy);
    } else if (fast64) {
        const std::uint32_t gx = static_cast<std::uint32_t>(N / kernels::kGemmFastBlock);
        const std::uint32_t gy = static_cast<std::uint32_t>(M / 64);
        c.dispatch_blocks_2d(kernels::gemm_fast64_name<T>, bind, 4, gx, gy);
    } else {
        const std::uint32_t gx = static_cast<std::uint32_t>((N + kernels::kGemmBlock - 1) / kernels::kGemmBlock);
        const std::uint32_t gy = static_cast<std::uint32_t>((M + kernels::kGemmBlock - 1) / kernels::kGemmBlock);
        c.dispatch_blocks_2d(kernels::gemm_edge_name<T>, bind, 4, gx, gy);
    }
    c.release_buffer(dimbuf);
}

/**
 * Device outer product: `out[n×m] = aᵢ·bⱼ`. Invoked by cheatah's allocating `outer(a, b)` front;
 * one thread per output element over an m×n 2-D grid, blocking on completion.
 *
 * @param out The pre-allocated n×m device product.
 * @param a   The length-n device vector.
 * @param b   The length-m device vector.
 * @complexity O(n·m) device work in one blocking dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc none — all buffers are the caller's.
 * @test gpu:outer
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && GpuField<T>
void outer(Array<T>& out, const Array<T>& a, const Array<T>& b) {
    const std::size_t n = cheatah::linalg::vector_len(a);
    const std::size_t m = cheatah::linalg::vector_len(b);
    if (n == 0 || m == 0) return;
    const std::uint32_t dims[2] = {static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(m)};
    detail::Context& c = detail::ctx();
    detail::Buffer* dimbuf = detail::dims_buffer(dims, 2);
    detail::Buffer* bind[4] = {a.buffer(), b.buffer(), out.buffer(), dimbuf};
    c.dispatch_2d(kernels::outer_name<T>, bind, 4, m, n);
    c.release_buffer(dimbuf);
}

/**
 * Device conjugate transpose: `out[c×r] = aᴴ` for an r×c input (the kernel conjugates complex
 * elements; conjugation is the identity for real ones). Invoked by cheatah's allocating front;
 * one thread per input element, blocking on completion.
 *
 * @param out The pre-allocated c×r device transpose.
 * @param a   The r×c device operand (throws unless 2-D).
 * @complexity O(r·c) device work in one blocking dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc none — both buffers are the caller's.
 * @test gpu:transpose
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && GpuField<T>
void conj_transpose(Array<T>& out, const Array<T>& a) {
    if (a.ndim() != 2)
        throw std::runtime_error("cheatah-gpu-linalg conj_transpose: operand must be 2-D");
    const std::size_t r = a.shape()[0], cdim = a.shape()[1];
    if (r == 0 || cdim == 0) return;
    const std::uint32_t dims[2] = {static_cast<std::uint32_t>(r),
                                   static_cast<std::uint32_t>(cdim)};
    detail::Context& c = detail::ctx();
    detail::Buffer* dimbuf = detail::dims_buffer(dims, 2);
    detail::Buffer* bind[3] = {a.buffer(), out.buffer(), dimbuf};
    c.dispatch_2d(kernels::transpose_name<T>, bind, 3, cdim, r);
    c.release_buffer(dimbuf);
}

/**
 * Device Kronecker product: `out[(ar·br)×(ac·bc)] = a ⊗ b`. Invoked by cheatah's allocating
 * `kron(a, b)` front; one thread per output element, blocking on completion.
 *
 * @param out The pre-allocated (ar·br)×(ac·bc) device product.
 * @param a   The ar×ac device left operand (throws unless 2-D).
 * @param b   The br×bc device right operand (throws unless 2-D).
 * @complexity O(n⁴) in the output area (one multiply per output element).
 * @alloc none — content-cached dims scratch only.
 * @gpualloc none — all buffers are the caller's.
 * @test gpu:kron
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && GpuField<T>
void kron(Array<T>& out, const Array<T>& a, const Array<T>& b) {
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::runtime_error("cheatah-gpu-linalg kron: operands must be 2-D");
    const std::size_t ar = a.shape()[0], ac = a.shape()[1];
    const std::size_t br = b.shape()[0], bc = b.shape()[1];
    const std::size_t rows = ar * br, cols = ac * bc;
    if (rows == 0 || cols == 0) return;
    const std::uint32_t dims[4] = {static_cast<std::uint32_t>(ar), static_cast<std::uint32_t>(ac),
                                   static_cast<std::uint32_t>(br), static_cast<std::uint32_t>(bc)};
    detail::Context& c = detail::ctx();
    detail::Buffer* dimbuf = detail::dims_buffer(dims, 4);
    detail::Buffer* bind[4] = {a.buffer(), b.buffer(), out.buffer(), dimbuf};
    c.dispatch_2d(kernels::kron_name<T>, bind, 4, cols, rows);
    c.release_buffer(dimbuf);
}

/**
 * Device dot product (bilinear Σ aᵢbᵢ) into the caller's scalar — the DeviceArray overload of
 * cheatah's scalar-out reduction kernel, dispatched as deterministic partial sums (the same bits
 * every run), finalized on device and downloaded as ONE element.
 *
 * @param out The caller's scalar result.
 * @param a   The first device vector.
 * @param b   The second device vector (throws on a length mismatch).
 * @complexity O(n) device work; one fused submission (partial + finalize).
 * @alloc none — the partial/result scratch recycles through the pooled allocator; the readback
 *        is sizeof(T).
 * @gpualloc none — pooled scratch only.
 * @test gpu:dot
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && GpuField<T>
void dot(T& out, const Array<T>& a, const Array<T>& b) {
    out = detail::device_dot(a, b, kernels::dot_name<T>, kernels::dot2_name<T>);
}

/**
 * Device vdot: the Hermitian Σ conj(aᵢ)·bᵢ for complex elements (its own conjugating kernel);
 * for real elements the conjugation is the identity and it shares @ref dot's kernel.
 *
 * @param out The caller's scalar result.
 * @param a   The first (conjugated) device vector.
 * @param b   The second device vector (throws on a length mismatch).
 * @complexity O(n) device work; one fused submission (partial + finalize).
 * @alloc none — pooled partial/result scratch; sizeof(T) readback.
 * @gpualloc none — pooled scratch only.
 * @test gpu:dot
 * @test gpu:complex
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && GpuField<T>
void vdot(T& out, const Array<T>& a, const Array<T>& b) {
    if constexpr (GpuComplexElement<T>)
        out = detail::device_dot(a, b, kernels::vdot_name<T>,
                                 kernels::vdot2_name<T>);        // conj(a)·b — the Hermitian form
    else
        out = detail::device_dot(a, b, kernels::dot_name<T>,
                                 kernels::dot2_name<T>);         // bilinear ≡ dot for real T
}

/**
 * Device inner product — the bilinear Σ aᵢbᵢ, identical to @ref dot for flattened vectors
 * (dispatched through the same kernel).
 *
 * @param out The caller's scalar result.
 * @param a   The first device vector.
 * @param b   The second device vector (throws on a length mismatch).
 * @complexity O(n) device work; one fused submission (partial + finalize).
 * @alloc none — pooled partial/result scratch; sizeof(T) readback.
 * @gpualloc none — pooled scratch only.
 * @test gpu:dot
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && GpuField<T>
void inner(T& out, const Array<T>& a, const Array<T>& b) {
    out = detail::device_dot(a, b, kernels::dot_name<T>, kernels::dot2_name<T>);
}

/**
 * DEVICE-RESIDENT dot: the scalar result lands in a 1-element `device_array` with ZERO
 * downloads — the training-loop form (chain it, or read it back once per epoch).
 *
 * @param out The 1-element device array the result lands in (throws unless size() == 1).
 * @param a   The first device vector.
 * @param b   The second device vector (throws on a length mismatch).
 * @complexity O(n) device work; one fused submission, zero downloads.
 * @alloc none — pooled partial scratch only; nothing crosses the bus.
 * @gpualloc none — pooled scratch only.
 * @test gpu:dot
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && GpuField<T>
void dot(Array<T>& out, const Array<T>& a, const Array<T>& b) {
    if (out.size() != 1)
        throw std::runtime_error("cheatah-gpu-linalg dot: resident out must have exactly 1 element");
    // Validate BEFORE begin_fuse: a throw from inside the fuse would leave it half-recorded.
    if (cheatah::linalg::vector_len(a) != cheatah::linalg::vector_len(b))
        throw std::runtime_error("cheatah-gpu-linalg dot: length mismatch");
    std::uint32_t count = 0;
    detail::Context& c = detail::ctx();
    c.begin_fuse();                      // partial + finalize: ONE submission
    detail::Buffer* partial = detail::dot_partials(a, b, kernels::dot_name<T>,
                                                   kernels::dot2_name<T>, count);
    detail::finalize_into<T>(partial, count, out.buffer());
    c.end_fuse();
    c.release_buffer(partial);
}

/**
 * Device trace (diagonal sum of a 2-D matrix) into the caller's scalar — deterministic partial
 * sums over the strided diagonal, summed on the host in a fixed order.
 *
 * @param out The caller's scalar result.
 * @param a   The r×c device matrix (min(r,c) diagonal elements are summed).
 * @complexity O(min(r,c)) device work + O(P) host partial summation (P ≤ 256).
 * @alloc none — the P-element partial buffer and dims scratch recycle through the pool.
 * @gpualloc none — pooled scratch only.
 * @test gpu:trace
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && GpuField<T>
void trace(T& out, const Array<T>& a) {
    const std::size_t r = a.shape()[0], cdim = a.shape()[1];
    const std::size_t m = std::min(r, cdim);
    if (m == 0) {
        out = T{};
        return;
    }
    const std::uint32_t P =
        static_cast<std::uint32_t>(std::min<std::size_t>(m, kernels::kMaxReduce));
    const std::uint32_t dims[3] = {static_cast<std::uint32_t>(m),
                                   static_cast<std::uint32_t>(cdim), P};
    detail::Context& c = detail::ctx();
    detail::Buffer* partial = c.new_buffer(P * sizeof(T));
    detail::Buffer* dimbuf = detail::dims_buffer(dims, 3);
    detail::Buffer* bind[3] = {a.buffer(), partial, dimbuf};
    c.dispatch_1d(kernels::trace_name<T>, bind, 3, P);
    const T* p = static_cast<const T*>(detail::Context::contents(partial));
    T sum = T{};
    for (std::uint32_t g = 0; g < P; ++g) sum += p[g];
    c.release_buffer(partial);
    c.release_buffer(dimbuf);
    out = sum;
}

}  // namespace cheatah::gpu::linalg
