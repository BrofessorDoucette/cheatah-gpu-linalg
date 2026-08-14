// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
#pragma once

/**
 * @file elementwise.hpp
 * @brief Elementwise device ops + operators on `device_array` — the training-loop arithmetic.
 *
 * cheatah's ndarray reaches arithmetic through ordinary C++ operator resolution (purr's codegen
 * emits `a + b` and ADL finds the overload), so device elementwise needs NO cheatah changes:
 * everything here lives in `device_array`'s namespace and resolves the same way, from C++ and
 * from purr. The named fronts match cheatah's ndarray spellings (add/sub/mul/divide, sqrt/exp/
 * log/abs, sum/mean) so device code reads like host code.
 *
 * Semantics this iteration: array⊗array is STRICT same-shape (no general broadcasting — throw on
 * mismatch); array⊗scalar broadcasts the scalar (both orders). Shapes are preserved (elementwise
 * over the flat contiguous buffer, any rank). Every kernel writes out[i] from lane i only, so
 * `out` may alias an operand — which is what makes the compound assigns free.
 *
 * `axpy(out, α, x, y)` is the deliberate extra: `α·x + y` fused into ONE dispatch (operator
 * chaining costs two dispatches and a temporary — at ~tens of µs per dispatch that fusion is the
 * single biggest win available to a training loop).
 */

#include "cheatah_gpu_linalg/device_array.hpp"
#include "cheatah_gpu_linalg/kernels.hpp"
#include "cheatah_gpu_linalg/routines.hpp"   // GpuElement + detail::dims_buffer

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace cheatah::gpu::linalg {

namespace detail {

/// Throw unless a and b have identical shapes (the strict array⊗array contract).
template <GpuElement T>
inline void require_same_shape(const device_array<T>& a, const device_array<T>& b) {
    if (a.shape() != b.shape())
        throw std::runtime_error("cheatah-gpu-linalg: elementwise shape mismatch");
}

/// A 1-element device buffer holding scalar `s` (the scalar-operand binding).
template <GpuElement T>
inline Buffer* scalar_buffer(T s) {
    Buffer* b = ctx().new_buffer(sizeof(T));
    std::memcpy(Context::contents(b), &s, sizeof(T));
    return b;
}

/// Dispatch one ew_* array⊗array kernel over n elements: bindings {a, b, out, dims{n}}.
template <GpuElement T>
inline void ew_dispatch(kernels::EwOp op, device_array<T>& out, const device_array<T>& a,
                        const device_array<T>& b) {
    const std::uint32_t n = static_cast<std::uint32_t>(a.size());
    if (n == 0) return;
    Context& c = ctx();
    const std::uint32_t dims[1] = {n};
    Buffer* dimbuf = dims_buffer(dims, 1);
    Buffer* bind[4] = {a.buffer(), b.buffer(), out.buffer(), dimbuf};
    c.dispatch_1d(kernels::ew_names<T>[static_cast<std::uint32_t>(op)], bind, 4, (n + 3) / 4);
    c.release_buffer(dimbuf);
}

/// Dispatch one ews_* array⊗scalar kernel: bindings {a, s, out, dims{n, swap}}; swap = 1 computes
/// s OP a (the reversed subtraction/division forms).
template <GpuElement T>
inline void ews_dispatch(kernels::EwOp op, device_array<T>& out, const device_array<T>& a, T s,
                         bool swap) {
    const std::uint32_t n = static_cast<std::uint32_t>(a.size());
    if (n == 0) return;
    Context& c = ctx();
    const std::uint32_t dims[2] = {n, swap ? 1u : 0u};
    Buffer* dimbuf = dims_buffer(dims, 2);
    Buffer* sbuf = scalar_buffer(s);
    Buffer* bind[4] = {a.buffer(), sbuf, out.buffer(), dimbuf};
    c.dispatch_1d(kernels::ews_names<T>[static_cast<std::uint32_t>(op)], bind, 4, (n + 3) / 4);
    c.release_buffer(sbuf);
    c.release_buffer(dimbuf);
}

/// Dispatch one ew_* unary kernel: bindings {a, out, dims{n}}.
template <GpuElement T>
inline void ewu_dispatch(kernels::EwFn fn, device_array<T>& out, const device_array<T>& a) {
    const std::uint32_t n = static_cast<std::uint32_t>(a.size());
    if (n == 0) return;
    Context& c = ctx();
    const std::uint32_t dims[1] = {n};
    Buffer* dimbuf = dims_buffer(dims, 1);
    Buffer* bind[3] = {a.buffer(), out.buffer(), dimbuf};
    c.dispatch_1d(kernels::ewu_names<T>[static_cast<std::uint32_t>(fn)], bind, 3, (n + 3) / 4);
    c.release_buffer(dimbuf);
}

}  // namespace detail

// ---- named out-param fronts (cheatah ndarray spellings; out may alias an operand) --------------

/**
 * out = a + b (same shape).
 * @param out The destination (may alias @p a or @p b).
 * @param a   The left device operand.
 * @param b   The right device operand (throws on a shape mismatch).
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc none — all buffers are the caller's.
 * @test gpu:elementwise
 */
template <GpuElement T>
void add(device_array<T>& out, const device_array<T>& a, const device_array<T>& b) {
    detail::require_same_shape(a, b);
    detail::ew_dispatch(kernels::EwOp::add, out, a, b);
}
/**
 * out = a − b (same shape).
 * @param out The destination (may alias an operand).
 * @param a   The left device operand.
 * @param b   The right device operand (throws on a shape mismatch).
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc none.
 * @test gpu:elementwise
 */
template <GpuElement T>
void sub(device_array<T>& out, const device_array<T>& a, const device_array<T>& b) {
    detail::require_same_shape(a, b);
    detail::ew_dispatch(kernels::EwOp::sub, out, a, b);
}
/**
 * out = a ⊙ b, the Hadamard product (same shape).
 * @param out The destination (may alias an operand).
 * @param a   The left device operand.
 * @param b   The right device operand (throws on a shape mismatch).
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc none.
 * @test gpu:elementwise
 */
template <GpuElement T>
void mul(device_array<T>& out, const device_array<T>& a, const device_array<T>& b) {
    detail::require_same_shape(a, b);
    detail::ew_dispatch(kernels::EwOp::mul, out, a, b);
}
/**
 * out = a ⊘ b, elementwise division (same shape).
 * @param out The destination (may alias an operand).
 * @param a   The dividend device operand.
 * @param b   The divisor device operand (throws on a shape mismatch).
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc none.
 * @test gpu:elementwise
 */
template <GpuElement T>
void divide(device_array<T>& out, const device_array<T>& a, const device_array<T>& b) {
    detail::require_same_shape(a, b);
    detail::ew_dispatch(kernels::EwOp::div, out, a, b);
}

/**
 * out = α·x + y in ONE dispatch — the fused training-loop primitive (same shapes; operator
 * chaining costs two dispatches and a temporary).
 * @param out   The destination (may alias @p x or @p y).
 * @param alpha The scalar coefficient.
 * @param x     The scaled device operand.
 * @param y     The added device operand (throws on a shape mismatch).
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — the 1-element α buffer and dims scratch recycle through the pool.
 * @gpualloc none.
 * @test gpu:elementwise
 */
template <GpuElement T>
void axpy(device_array<T>& out, T alpha, const device_array<T>& x, const device_array<T>& y) {
    detail::require_same_shape(x, y);
    const std::uint32_t n = static_cast<std::uint32_t>(x.size());
    if (n == 0) return;
    detail::Context& c = detail::ctx();
    const std::uint32_t dims[1] = {n};
    detail::Buffer* dimbuf = detail::dims_buffer(dims, 1);
    detail::Buffer* abuf = detail::scalar_buffer(alpha);
    detail::Buffer* bind[5] = {x.buffer(), y.buffer(), abuf, out.buffer(), dimbuf};
    c.dispatch_1d(kernels::axpy_name<T>, bind, 5, (n + 3) / 4);
    c.release_buffer(abuf);
    c.release_buffer(dimbuf);
}

/**
 * α·x + y, allocating.
 * @param alpha The scalar coefficient.
 * @param x     The scaled device operand.
 * @param y     The added device operand (throws on a shape mismatch).
 * @return A fresh device array holding α·x + y.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — pooled α/dims scratch only.
 * @gpualloc one pooled device data buffer for the result (owned by the returned array).
 * @test gpu:elementwise
 */
template <GpuElement T>
[[nodiscard]] device_array<T> axpy(T alpha, const device_array<T>& x, const device_array<T>& y) {
    device_array<T> out = device_array<T>::uninitialized(x.shape());
    axpy(out, alpha, x, y);
    return out;
}

// ---- named allocating fronts + unary math ------------------------------------------------------

/**
 * a + b, allocating.
 * @param a The left device operand.
 * @param b The right device operand (throws on a shape mismatch).
 * @return A fresh device array holding the sum.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:elementwise
 */
template <GpuElement T>
[[nodiscard]] device_array<T> add(const device_array<T>& a, const device_array<T>& b) {
    device_array<T> out = device_array<T>::uninitialized(a.shape());
    add(out, a, b);
    return out;
}
/**
 * a − b, allocating.
 * @param a The left device operand.
 * @param b The right device operand (throws on a shape mismatch).
 * @return A fresh device array holding the difference.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:elementwise
 */
template <GpuElement T>
[[nodiscard]] device_array<T> sub(const device_array<T>& a, const device_array<T>& b) {
    device_array<T> out = device_array<T>::uninitialized(a.shape());
    sub(out, a, b);
    return out;
}
/**
 * a ⊙ b, allocating.
 * @param a The left device operand.
 * @param b The right device operand (throws on a shape mismatch).
 * @return A fresh device array holding the Hadamard product.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:elementwise
 */
template <GpuElement T>
[[nodiscard]] device_array<T> mul(const device_array<T>& a, const device_array<T>& b) {
    device_array<T> out = device_array<T>::uninitialized(a.shape());
    mul(out, a, b);
    return out;
}
/**
 * a ⊘ b, allocating.
 * @param a The dividend device operand.
 * @param b The divisor device operand (throws on a shape mismatch).
 * @return A fresh device array holding the quotient.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:elementwise
 */
template <GpuElement T>
[[nodiscard]] device_array<T> divide(const device_array<T>& a, const device_array<T>& b) {
    device_array<T> out = device_array<T>::uninitialized(a.shape());
    divide(out, a, b);
    return out;
}

/**
 * √a elementwise, allocating (cheatah's `sqrt` ufunc spelling).
 * @param a The device operand.
 * @return A fresh device array holding the elementwise square root.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:elementwise
 */
template <GpuElement T>
[[nodiscard]] device_array<T> sqrt(const device_array<T>& a) {
    device_array<T> out = device_array<T>::uninitialized(a.shape());
    detail::ewu_dispatch(kernels::EwFn::sqrt, out, a);
    return out;
}
/**
 * eᵃ elementwise, allocating. f64 note: SPIR-V (and GPU hardware generally) has NO double
 * transcendentals — GLSL.std.450 Exp/Log exist only for 16/32-bit floats — so the double form
 * is evaluated on the HOST over the downloaded elements, full f64 precision; float runs the
 * device kernel.
 * @param a The device operand.
 * @return A fresh device array holding eᵃ.
 * @complexity O(n): one device dispatch for float; a host loop + round-trip for double.
 * @alloc none for float (dims scratch only); one O(n) host vector for the double host path.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:elementwise
 */
template <GpuElement T>
[[nodiscard]] device_array<T> exp(const device_array<T>& a) {
    device_array<T> out = device_array<T>::uninitialized(a.shape());
    if constexpr (std::is_same_v<T, double>) {
        std::vector<T> tmp(a.size());
        a.to_host(tmp.data());
        for (std::size_t i = 0; i < a.size(); ++i) tmp[i] = std::exp(tmp[i]);
        out.write_from(tmp.data());
    } else {
        detail::ewu_dispatch(kernels::EwFn::exp, out, a);
    }
    return out;
}
/**
 * ln a elementwise, allocating (same f64 host-evaluation note as @ref exp).
 * @param a The device operand.
 * @return A fresh device array holding ln a.
 * @complexity O(n): one device dispatch for float; a host loop + round-trip for double.
 * @alloc none for float (dims scratch only); one O(n) host vector for the double host path.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:elementwise
 */
template <GpuElement T>
[[nodiscard]] device_array<T> log(const device_array<T>& a) {
    device_array<T> out = device_array<T>::uninitialized(a.shape());
    if constexpr (std::is_same_v<T, double>) {
        std::vector<T> tmp(a.size());
        a.to_host(tmp.data());
        for (std::size_t i = 0; i < a.size(); ++i) tmp[i] = std::log(tmp[i]);
        out.write_from(tmp.data());
    } else {
        detail::ewu_dispatch(kernels::EwFn::log, out, a);
    }
    return out;
}
/**
 * |a| elementwise, allocating.
 * @param a The device operand.
 * @return A fresh device array holding the elementwise absolute value.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:elementwise
 */
template <GpuElement T>
[[nodiscard]] device_array<T> abs(const device_array<T>& a) {
    device_array<T> out = device_array<T>::uninitialized(a.shape());
    detail::ewu_dispatch(kernels::EwFn::abs, out, a);
    return out;
}

// ---- reductions — the deterministic partial-sum contract (same bits every run) -----------------

namespace detail {
/// The sum partial stage (size-picked kernel) -> partials buffer + count, shared by the
/// host-returning and device-resident fronts.
template <GpuElement T>
inline Buffer* sum_partials(const device_array<T>& a, std::uint32_t& count) {
    const std::size_t n = a.size();
    Context& cc = ctx();
    if (n >= kernels::kTwoStageMin) {
        const std::uint32_t G = static_cast<std::uint32_t>(
            std::min<std::size_t>((n + kernels::kLocal1d * 8 - 1) / (kernels::kLocal1d * 8),
                                  kernels::kMaxGroups));
        const std::uint32_t dims2[2] = {static_cast<std::uint32_t>(n), G};
        Buffer* partial = cc.new_buffer(G * sizeof(T));
        Buffer* dimbuf = dims_buffer(dims2, 2);
        Buffer* bind[4] = {a.buffer(), partial, dimbuf, a.buffer()};  // + vec4 view
        cc.dispatch_blocks_1d(kernels::sum2_name<T>, bind, 4, G);
        cc.release_buffer(dimbuf);
        count = G;
        return partial;
    }
    const std::uint32_t P = static_cast<std::uint32_t>(
        std::min<std::size_t>(std::max<std::size_t>(n, 1), kernels::kMaxReduce));
    const std::uint32_t dims[2] = {static_cast<std::uint32_t>(n), P};
    Buffer* partial = cc.new_buffer(P * sizeof(T));
    Buffer* dimbuf = dims_buffer(dims, 2);
    Buffer* bind[3] = {a.buffer(), partial, dimbuf};
    cc.dispatch_1d(kernels::sum_name<T>, bind, 3, P);
    cc.release_buffer(dimbuf);
    count = P;
    return partial;
}
}  // namespace detail

/**
 * Σ aᵢ over the whole array — the DETERMINISTIC partial-sum contract (a size-picked one- or
 * two-stage partial kernel + an on-device finalize, fused into one submission; fixed association
 * order, so the same bits come back every run).
 * @param a The device operand (any rank; 0 elements yield T{}).
 * @return The scalar sum, downloaded as ONE element.
 * @complexity O(n) device work; one fused submission (partial + finalize).
 * @alloc none — pooled partial/result scratch; sizeof(T) readback.
 * @gpualloc none — pooled scratch only.
 * @test gpu:elementwise
 */
template <GpuElement T>
[[nodiscard]] T sum(const device_array<T>& a) {
    if (a.size() == 0) return T{};
    std::uint32_t count = 0;
    detail::Context& c = detail::ctx();
    detail::Buffer* result = c.new_buffer(sizeof(T));   // allocate BEFORE the fuse (throw-safety)
    c.begin_fuse();                      // partial + finalize: ONE submission (one fence wait)
    detail::Buffer* partial = detail::sum_partials(a, count);
    detail::finalize_into<T>(partial, count, result);
    c.end_fuse();
    T total;
    c.download(result, &total, sizeof(T));
    c.release_buffer(result);
    c.release_buffer(partial);
    return total;
}

/**
 * DEVICE-RESIDENT sum: the result lands in a 1-element `device_array` with ZERO downloads —
 * keep a loss accumulator on the GPU and read it back once per epoch.
 * @param out The 1-element device array the sum lands in (throws unless size() == 1).
 * @param a   The device operand.
 * @complexity O(n) device work; one fused submission, zero downloads.
 * @alloc none — pooled partial scratch only; nothing crosses the bus.
 * @gpualloc none — pooled scratch only.
 * @test gpu:elementwise
 */
template <GpuElement T>
void sum(device_array<T>& out, const device_array<T>& a) {
    if (out.size() != 1)
        throw std::runtime_error("cheatah-gpu-linalg sum: resident out must have exactly 1 element");
    std::uint32_t count = 0;
    detail::Context& c = detail::ctx();
    c.begin_fuse();                      // partial + finalize: ONE submission
    detail::Buffer* partial = detail::sum_partials(a, count);
    detail::finalize_into<T>(partial, count, out.buffer());
    c.end_fuse();
    c.release_buffer(partial);
}

/**
 * The arithmetic mean Σ aᵢ / n (n = 0 yields 0, matching an empty sum).
 * @param a The device operand.
 * @return The scalar mean.
 * @complexity O(n) device work (one fused @ref sum) + one host division.
 * @alloc none — pooled partial/result scratch; sizeof(T) readback.
 * @gpualloc none — pooled scratch only.
 * @test gpu:elementwise
 */
template <GpuElement T>
[[nodiscard]] T mean(const device_array<T>& a) {
    const std::size_t n = a.size();
    return n == 0 ? T{} : sum(a) / static_cast<T>(n);
}

/**
 * DEVICE-RESIDENT mean: resident sum then one scalar-scale dispatch — zero downloads.
 * @param out The 1-element device array the mean lands in (throws unless size() == 1).
 * @param a   The device operand.
 * @complexity O(n) device work; two submissions (fused sum + scale), zero downloads.
 * @alloc none — pooled scratch only; nothing crosses the bus.
 * @gpualloc none — pooled scratch only.
 * @test gpu:elementwise
 */
template <GpuElement T>
void mean(device_array<T>& out, const device_array<T>& a) {
    sum(out, a);
    if (a.size() != 0) out *= static_cast<T>(1) / static_cast<T>(a.size());
}

// ---- operators — the purr path (codegen emits `a + b`; ADL resolves here) ----------------------

/**
 * a + b elementwise (the spelling purr's codegen emits; resolves here by ADL).
 * @param a The left device operand.
 * @param b The right device operand (throws on a shape mismatch).
 * @return A fresh device array holding the sum.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:operators
 */
template <GpuElement T>
[[nodiscard]] device_array<T> operator+(const device_array<T>& a, const device_array<T>& b) {
    return add(a, b);
}
/**
 * a − b elementwise.
 * @param a The left device operand.
 * @param b The right device operand (throws on a shape mismatch).
 * @return A fresh device array holding the difference.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:operators
 */
template <GpuElement T>
[[nodiscard]] device_array<T> operator-(const device_array<T>& a, const device_array<T>& b) {
    return sub(a, b);
}
/**
 * a ⊙ b elementwise.
 * @param a The left device operand.
 * @param b The right device operand (throws on a shape mismatch).
 * @return A fresh device array holding the Hadamard product.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:operators
 */
template <GpuElement T>
[[nodiscard]] device_array<T> operator*(const device_array<T>& a, const device_array<T>& b) {
    return mul(a, b);
}
/**
 * a ⊘ b elementwise.
 * @param a The dividend device operand.
 * @param b The divisor device operand (throws on a shape mismatch).
 * @return A fresh device array holding the quotient.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:operators
 */
template <GpuElement T>
[[nodiscard]] device_array<T> operator/(const device_array<T>& a, const device_array<T>& b) {
    return divide(a, b);
}

// The scalar-broadcast family, both orders. Every form is one blocking vec4 dispatch of the
// ews_* kernel; @alloc none (pooled 1-element scalar + dims scratch); @gpualloc one pooled
// device data buffer for the result. @test gpu:operators.

/**
 * a + s, scalar broadcast.
 * @param a The device operand. @param s The broadcast scalar.
 * @return A fresh device array holding a + s.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — pooled scalar/dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:operators
 */
template <GpuElement T>
[[nodiscard]] device_array<T> operator+(const device_array<T>& a, T s) {
    device_array<T> out = device_array<T>::uninitialized(a.shape());
    detail::ews_dispatch(kernels::EwOp::add, out, a, s, false);
    return out;
}
/**
 * s + a, scalar broadcast (commuted @ref operator+(const device_array<T>&, T)).
 * @param s The broadcast scalar. @param a The device operand.
 * @return A fresh device array holding s + a.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — pooled scalar/dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:operators
 */
template <GpuElement T>
[[nodiscard]] device_array<T> operator+(T s, const device_array<T>& a) { return a + s; }
/**
 * a − s, scalar broadcast.
 * @param a The device operand. @param s The broadcast scalar.
 * @return A fresh device array holding a − s.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — pooled scalar/dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:operators
 */
template <GpuElement T>
[[nodiscard]] device_array<T> operator-(const device_array<T>& a, T s) {
    device_array<T> out = device_array<T>::uninitialized(a.shape());
    detail::ews_dispatch(kernels::EwOp::sub, out, a, s, false);
    return out;
}
/**
 * s − a, scalar broadcast (the kernel's swap flag reverses the operands).
 * @param s The broadcast scalar. @param a The device operand.
 * @return A fresh device array holding s − a.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — pooled scalar/dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:operators
 */
template <GpuElement T>
[[nodiscard]] device_array<T> operator-(T s, const device_array<T>& a) {
    device_array<T> out = device_array<T>::uninitialized(a.shape());
    detail::ews_dispatch(kernels::EwOp::sub, out, a, s, true);
    return out;
}
/**
 * a·s, scalar broadcast.
 * @param a The device operand. @param s The broadcast scalar.
 * @return A fresh device array holding a·s.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — pooled scalar/dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:operators
 */
template <GpuElement T>
[[nodiscard]] device_array<T> operator*(const device_array<T>& a, T s) {
    device_array<T> out = device_array<T>::uninitialized(a.shape());
    detail::ews_dispatch(kernels::EwOp::mul, out, a, s, false);
    return out;
}
/**
 * s·a, scalar broadcast (commuted @ref operator*(const device_array<T>&, T)).
 * @param s The broadcast scalar. @param a The device operand.
 * @return A fresh device array holding s·a.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — pooled scalar/dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:operators
 */
template <GpuElement T>
[[nodiscard]] device_array<T> operator*(T s, const device_array<T>& a) { return a * s; }
/**
 * a÷s, scalar broadcast.
 * @param a The dividend device operand. @param s The broadcast divisor.
 * @return A fresh device array holding a÷s.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — pooled scalar/dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:operators
 */
template <GpuElement T>
[[nodiscard]] device_array<T> operator/(const device_array<T>& a, T s) {
    device_array<T> out = device_array<T>::uninitialized(a.shape());
    detail::ews_dispatch(kernels::EwOp::div, out, a, s, false);
    return out;
}
/**
 * s÷a, scalar broadcast (the kernel's swap flag reverses the operands).
 * @param s The broadcast dividend. @param a The divisor device operand.
 * @return A fresh device array holding s÷a.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — pooled scalar/dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:operators
 */
template <GpuElement T>
[[nodiscard]] device_array<T> operator/(T s, const device_array<T>& a) {
    device_array<T> out = device_array<T>::uninitialized(a.shape());
    detail::ews_dispatch(kernels::EwOp::div, out, a, s, true);
    return out;
}

/**
 * Unary −a.
 * @param a The device operand.
 * @return A fresh device array holding the negation.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:operators
 */
template <GpuElement T>
[[nodiscard]] device_array<T> operator-(const device_array<T>& a) {
    device_array<T> out = device_array<T>::uninitialized(a.shape());
    detail::ewu_dispatch(kernels::EwFn::neg, out, a);
    return out;
}

// Compound assigns — in place (the kernels are alias-safe, so `a` is both operand and
// destination). One blocking vec4 dispatch each; @alloc none (pooled scratch); @gpualloc none.

/**
 * a += b in place.
 * @param a The accumulating device operand. @param b The added device operand.
 * @return @p a.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — pooled scratch only. @gpualloc none — in place.
 * @test gpu:operators
 */
template <GpuElement T>
device_array<T>& operator+=(device_array<T>& a, const device_array<T>& b) {
    add(a, a, b);
    return a;
}
/**
 * a −= b in place.
 * @param a The accumulating device operand. @param b The subtracted device operand.
 * @return @p a.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — pooled scratch only. @gpualloc none — in place.
 * @test gpu:operators
 */
template <GpuElement T>
device_array<T>& operator-=(device_array<T>& a, const device_array<T>& b) {
    sub(a, a, b);
    return a;
}
/**
 * a ⊙= b in place.
 * @param a The accumulating device operand. @param b The multiplying device operand.
 * @return @p a.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — pooled scratch only. @gpualloc none — in place.
 * @test gpu:operators
 */
template <GpuElement T>
device_array<T>& operator*=(device_array<T>& a, const device_array<T>& b) {
    mul(a, a, b);
    return a;
}
/**
 * a ⊘= b in place.
 * @param a The accumulating device operand. @param b The dividing device operand.
 * @return @p a.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — pooled scratch only. @gpualloc none — in place.
 * @test gpu:operators
 */
template <GpuElement T>
device_array<T>& operator/=(device_array<T>& a, const device_array<T>& b) {
    divide(a, a, b);
    return a;
}
/**
 * a += s in place (scalar broadcast).
 * @param a The accumulating device operand. @param s The broadcast scalar.
 * @return @p a.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — pooled scalar/dims scratch only. @gpualloc none — in place.
 * @test gpu:operators
 */
template <GpuElement T>
device_array<T>& operator+=(device_array<T>& a, T s) {
    detail::ews_dispatch(kernels::EwOp::add, a, a, s, false);
    return a;
}
/**
 * a −= s in place (scalar broadcast).
 * @param a The accumulating device operand. @param s The broadcast scalar.
 * @return @p a.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — pooled scalar/dims scratch only. @gpualloc none — in place.
 * @test gpu:operators
 */
template <GpuElement T>
device_array<T>& operator-=(device_array<T>& a, T s) {
    detail::ews_dispatch(kernels::EwOp::sub, a, a, s, false);
    return a;
}
/**
 * a ⊙= s in place (scalar broadcast — the resident @ref mean's scale step).
 * @param a The accumulating device operand. @param s The broadcast scalar.
 * @return @p a.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — pooled scalar/dims scratch only. @gpualloc none — in place.
 * @test gpu:operators
 */
template <GpuElement T>
device_array<T>& operator*=(device_array<T>& a, T s) {
    detail::ews_dispatch(kernels::EwOp::mul, a, a, s, false);
    return a;
}
/**
 * a ⊘= s in place (scalar broadcast).
 * @param a The accumulating device operand. @param s The broadcast divisor.
 * @return @p a.
 * @complexity O(n) device work in one blocking vec4 dispatch.
 * @alloc none — pooled scalar/dims scratch only. @gpualloc none — in place.
 * @test gpu:operators
 */
template <GpuElement T>
device_array<T>& operator/=(device_array<T>& a, T s) {
    detail::ews_dispatch(kernels::EwOp::div, a, a, s, false);
    return a;
}

}  // namespace cheatah::gpu::linalg
