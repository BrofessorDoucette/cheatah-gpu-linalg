// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
#pragma once

/**
 * @file factories.hpp
 * @brief Ergonomic parity with `cheatah::ndarray` — the SAME construction/inspection API for
 *        `device_array`, so generic code differs by ONE compile-time template argument.
 *
 * Every free function here mirrors its `cheatah::ndarray` namesake (zeros/ones/full/arange/
 * array/scalar, the `*_like` family, reshape, get, to_string, size_of, is_contiguous) — the
 * argument-bearing ones resolve by ADL, which is what lets a
 * `template <template <class> class Array>` function body stay location-blind.
 *
 * DEVICE-RESIDENCY notes:
 *   - `reshape` is ZERO-COPY (device arrays are always contiguous: new shape/strides share the
 *     buffer — the same view semantics ndarray gives contiguous reshapes).
 *   - `get` / `to_string` / `to_host` are documented SYNC POINTS (they download) — inspection
 *     and verification tools, not hot-loop citizens. `get` moves exactly one element.
 *   - `to_device` / `to_host` are the explicit first-class converters between the containers.
 *   - `stats()` / `reset_stats()` expose the context's transfer ledger, so residency is a
 *     testable property (the examples assert zero mid-loop downloads).
 */

#include "cheatah_gpu_linalg/device_array.hpp"
#include "cheatah_gpu_linalg/routines.hpp"   // detail::dims_buffer + kernels::fill_name

#include "ndarray.hpp"   // cheatah stdlib/ndarray: basic_ndarray + to_string

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cheatah::gpu::linalg {

/// The host container under its device-parity name: `step<host_array>(…)` vs
/// `step<device_array>(…)` is the entire CPU↔GPU switch.
template <class T>
using host_array = cheatah::ndarray::basic_ndarray<T>;

/**
 * The transfer/dispatch ledger of the process-wide context (uploads, downloads, dispatches).
 * @return The live counters (a reference into the context — reads reflect later ops).
 * @complexity O(1). @alloc none. @gpualloc none.
 * @test example:vector_pipeline
 */
[[nodiscard]] inline const detail::Context::TransferStats& stats() {
    return detail::ctx().stats();
}
/**
 * Zero the ledger (examples call this right before a hot loop, then assert on @ref stats).
 * @complexity O(1). @alloc none. @gpualloc none.
 * @test example:vector_pipeline
 */
inline void reset_stats() { detail::ctx().reset_stats(); }

/**
 * Upload a host array into a fresh device array (same shape + element type; the operand must be
 * contiguous — pack first via ndarray's reshape/copy if it is a strided view, else this throws).
 * @param a The contiguous host array.
 * @return A device array holding a copy of @p a's elements.
 * @complexity O(n) transfer.
 * @alloc metadata only — the Vulkan upload stages through a pooled host-visible buffer.
 * @gpualloc one pooled device data buffer of size()·sizeof(T).
 * @test example:vector_pipeline
 */
template <ndarray::Field T>
[[nodiscard]] device_array<T> to_device(const host_array<T>& a) {
    if (!cheatah::ndarray::is_contiguous(a))
        throw std::runtime_error("cheatah-gpu-linalg to_device: operand must be contiguous");
    return device_array<T>::from_host(a.shape(), a.buffer()->data() + a.offset());
}

/**
 * Download a device array into a fresh host array (same shape + element type). A SYNC POINT.
 * @param a The device array.
 * @return A host ndarray holding a copy of @p a's elements.
 * @complexity O(n) transfer.
 * @alloc the O(n) host result; the Vulkan readback stages through a pooled host-cached buffer.
 * @gpualloc none.
 * @test example:mlp_forward
 */
template <ndarray::Field T>
[[nodiscard]] host_array<T> to_host(const device_array<T>& a) {
    host_array<T> h = host_array<T>::uninitialized(a.shape());
    a.to_host(h.buffer()->data());
    return h;
}

namespace detail {

/// Constant-fill a device array IN PLACE. Real elements dispatch the `fill` kernel — no host
/// staging vector, no PCIe transfer; complex elements (no fill kernel, mirroring the ews family)
/// take the staged-upload fallback.
template <ndarray::Field T>
inline void fill_device(device_array<T>& a, T value) {
    if (a.size() == 0) return;
    if constexpr (GpuElement<T>) {
        Context& c = ctx();
        Buffer* s = c.new_buffer(sizeof(T));
        std::memcpy(Context::contents(s), &value, sizeof(T));
        const std::uint32_t dims[1] = {static_cast<std::uint32_t>(a.size())};
        Buffer* d = dims_buffer(dims, 1);
        Buffer* bind[3] = {a.buffer(), s, d};
        c.dispatch_1d(kernels::fill_name<T>, bind, 3, (a.size() + 3) / 4);
        c.release_buffer(s);
        c.release_buffer(d);
    } else {
        std::vector<T> fill(a.size(), value);
        a.write_from(fill.data());
    }
}

}  // namespace detail

// ---- factories (the ndarray namesakes, device-resident results) --------------------------------

/**
 * A device array of the given shape, every element `value` (ndarray's `full`). Real elements
 * fill ON DEVICE (no host staging vector, no PCIe upload); complex elements take the
 * staged-upload fallback.
 * @param shape The dimensions (cheatah's signed integers; negatives throw).
 * @param value The fill value.
 * @return A fresh device array of @p value s.
 * @complexity O(n) device work (real) or O(n) host fill + upload (complex).
 * @alloc none for real elements (pooled scalar/dims scratch); one O(n) host vector for complex.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:elementwise
 */
template <ndarray::Field T>
[[nodiscard]] device_array<T> full(const std::vector<long long>& shape, T value) {
    std::vector<std::size_t> ushape;
    ushape.reserve(shape.size());
    for (long long d : shape) {
        if (d < 0) throw std::runtime_error("cheatah-gpu-linalg: negative dimension");
        ushape.push_back(static_cast<std::size_t>(d));
    }
    device_array<T> a = device_array<T>::uninitialized(std::move(ushape));
    detail::fill_device(a, value);
    return a;
}

/**
 * A double device array of zeros (ndarray's `zeros`) — the device-side fill kernel, no upload.
 * @param shape The dimensions (negatives throw).
 * @return A fresh zeroed device array.
 * @complexity O(n) device work.
 * @alloc none — pooled scalar/dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test gpu:elementwise
 */
[[nodiscard]] inline device_array<double> zeros(const std::vector<long long>& shape) {
    return full<double>(shape, 0.0);
}
/**
 * A double device array of ones (ndarray's `ones`) — the device-side fill kernel, no upload.
 * @param shape The dimensions (negatives throw).
 * @return A fresh device array of ones.
 * @complexity O(n) device work.
 * @alloc none — pooled scalar/dims scratch only.
 * @gpualloc one pooled device data buffer for the result.
 * @test example:batched_inference
 */
[[nodiscard]] inline device_array<double> ones(const std::vector<long long>& shape) {
    return full<double>(shape, 1.0);
}

/**
 * Same shape/element as @p a, every element `value` — ADL-reachable, so generic code can call
 * it unqualified on either container.
 * @param a     The array whose shape/element is copied.
 * @param value The fill value.
 * @return A fresh device array of @p value s.
 * @complexity O(n) device work (real elements fill on device).
 * @alloc none for real elements; one O(n) host vector for complex.
 * @gpualloc one pooled device data buffer for the result.
 */
template <ndarray::Field T>
[[nodiscard]] device_array<T> full_like(const device_array<T>& a, T value) {
    device_array<T> out = device_array<T>::uninitialized(a.shape());
    detail::fill_device(out, value);
    return out;
}
/**
 * Zeros with @p a's shape/element (ADL-reachable).
 * @param a The array whose shape/element is copied.
 * @return A fresh zeroed device array.
 * @complexity O(n) device work.
 * @alloc none for real elements. @gpualloc one pooled device data buffer for the result.
 */
template <ndarray::Field T>
[[nodiscard]] device_array<T> zeros_like(const device_array<T>& a) {
    return full_like(a, T{});
}
/**
 * Ones with @p a's shape/element (ADL-reachable).
 * @param a The array whose shape/element is copied.
 * @return A fresh device array of ones.
 * @complexity O(n) device work.
 * @alloc none for real elements. @gpualloc one pooled device data buffer for the result.
 */
template <ndarray::Field T>
[[nodiscard]] device_array<T> ones_like(const device_array<T>& a) {
    return full_like(a, T{1});
}

/**
 * A 1-D device array from a host vector (ndarray's `array`).
 * @param values The host elements.
 * @return A fresh 1-D device array holding a copy of @p values.
 * @complexity O(n) transfer.
 * @alloc metadata only — pooled staging on Vulkan.
 * @gpualloc one pooled device data buffer for the result.
 * @test example:vector_pipeline
 */
template <ndarray::Field T>
[[nodiscard]] device_array<T> array(const std::vector<T>& values) {
    return device_array<T>::from_host({values.size()}, values.data());
}
/**
 * A 1-D device array from a braced list (`array({1.0, 2.0, 3.0})`).
 * @param values The literal elements.
 * @return A fresh 1-D device array holding a copy of @p values.
 * @complexity O(n) transfer.
 * @alloc metadata only — pooled staging on Vulkan.
 * @gpualloc one pooled device data buffer for the result.
 * @test example:vector_pipeline
 */
template <ndarray::Field T>
[[nodiscard]] device_array<T> array(std::initializer_list<T> values) {
    return device_array<T>::from_host({values.size()}, values.begin());
}
/**
 * A 0-d scalar device array.
 * @param value The element.
 * @return A fresh 0-d device array holding @p value.
 * @complexity O(1) transfer.
 * @alloc metadata only. @gpualloc one pooled device data buffer (one size class, 256 B minimum).
 */
template <ndarray::Field T>
[[nodiscard]] device_array<T> scalar(T value) {
    return device_array<T>::from_host({}, &value);
}
/**
 * Evenly spaced values in [start, stop) with the given step (ndarray's `arange`; a zero step
 * throws).
 * @param start The first value.
 * @param stop  The exclusive bound.
 * @param step  The (non-zero) increment.
 * @return A fresh 1-D device array of the sequence.
 * @complexity O(n) host generation + O(n) transfer.
 * @alloc one O(n) host vector (the sequence is generated host-side, then uploaded).
 * @gpualloc one pooled device data buffer for the result.
 */
template <ndarray::Field T>
[[nodiscard]] device_array<T> arange(T start, T stop, T step) {
    if (step == T{}) throw std::runtime_error("cheatah-gpu-linalg: arange step must be non-zero");
    std::vector<T> v;
    if (stop != start && ((stop > start) == (step > T{}))) {
        const auto est = (stop - start) / step;    // reserve is a hint — clamp before the cast
        v.reserve(est < T(1 << 24) ? static_cast<std::size_t>(est) + 1 : std::size_t(1) << 24);
    }
    if constexpr (std::floating_point<T>) {
        // Integer induction with x = start + i*step — numpy computes arange the same way.
        // A floating-point loop counter (cert-flp30-c) accumulates rounding error every
        // pass; the multiply form keeps each element one rounding away from exact. The
        // bound check per element preserves the exclusive-stop semantics at the boundary.
        const double span = (static_cast<double>(stop) - static_cast<double>(start)) /
                            static_cast<double>(step);
        const std::size_t count = span > 0.0 ? static_cast<std::size_t>(std::ceil(span)) : 0;
        for (std::size_t i = 0; i < count; ++i) {
            const T x = start + static_cast<T>(i) * step;
            if ((step > T{}) ? (x < stop) : (x > stop)) v.push_back(x);
        }
    } else {
        for (T x = start; (step > T{}) ? (x < stop) : (x > stop); x += step) v.push_back(x);
    }
    return array(v);
}

// ---- shape + inspection ------------------------------------------------------------------------

/**
 * Reshape — ZERO-COPY: device arrays are always contiguous, so the new view shares the buffer
 * (throws on a negative dimension, an element-count overflow, or a count mismatch).
 * @param a     The device array to re-view.
 * @param shape The new dimensions (same element count).
 * @return A sibling array sharing @p a's device buffer under the new shape.
 * @complexity O(ndim).
 * @alloc the new shape/strides vectors.
 * @gpualloc none — zero-copy by design.
 * @test example:mlp_forward
 */
template <ndarray::Field T>
[[nodiscard]] device_array<T> reshape(const device_array<T>& a,
                                      const std::vector<long long>& shape) {
    std::vector<std::size_t> ushape;
    std::size_t total = 1;
    for (long long d : shape) {
        if (d < 0) throw std::runtime_error("cheatah-gpu-linalg: negative dimension");
        const std::size_t ud = static_cast<std::size_t>(d);
        if (ud != 0 && total > (std::numeric_limits<std::size_t>::max)() / ud)
            throw std::runtime_error("cheatah-gpu-linalg: reshape element-count overflow");
        ushape.push_back(ud);
        total *= ud;
    }
    if (total != a.size())
        throw std::runtime_error("cheatah-gpu-linalg: reshape element-count mismatch");
    return a.with_shape(std::move(ushape));
}

/**
 * Read ONE element by signed multi-index (ndarray's `get`) — a single-element SYNC POINT
 * (exactly sizeof(T) crosses the bus; throws on a rank mismatch or an out-of-range index).
 * @param a     The device array.
 * @param index One signed index per dimension.
 * @return The element's value.
 * @complexity O(ndim) index math + an O(1) offset download.
 * @alloc none fresh — the Vulkan readback stages through a pooled host-cached buffer.
 * @gpualloc none.
 * @test example:linear_regression_gd
 */
template <ndarray::Field T>
[[nodiscard]] T get(const device_array<T>& a, const std::vector<long long>& index) {
    if (index.size() != a.ndim())
        throw std::runtime_error("cheatah-gpu-linalg get: rank mismatch");
    std::size_t flat = 0;
    for (std::size_t d = 0; d < index.size(); ++d) {
        if (index[d] < 0 || static_cast<std::size_t>(index[d]) >= a.shape()[d])
            throw std::runtime_error("cheatah-gpu-linalg get: index out of range");
        flat += static_cast<std::size_t>(index[d]) *
                static_cast<std::size_t>(a.strides()[d]);
    }
    T out;
    detail::ctx().download_at(a.buffer(), &out, sizeof(T), flat * sizeof(T));
    return out;
}

/**
 * Render like ndarray renders (downloads, then reuses ndarray's own to_string — identical
 * output by construction). A SYNC POINT.
 * @param a The device array.
 * @return The rendered text.
 * @complexity O(n) download + O(n) formatting.
 * @alloc a full host copy of @p a plus the string.
 * @gpualloc none.
 */
template <ndarray::Field T>
[[nodiscard]] std::string to_string(const device_array<T>& a) {
    return cheatah::ndarray::to_string(to_host(a));
}
/**
 * Stream form — same rendering as ndarray's operator<< (a SYNC POINT, via @ref to_string).
 * @param os The destination stream.
 * @param a  The device array.
 * @return @p os.
 * @complexity O(n) download + O(n) formatting.
 * @alloc a full host copy of @p a plus the string. @gpualloc none.
 */
template <ndarray::Field T>
inline std::ostream& operator<<(std::ostream& os, const device_array<T>& a) {
    return os << to_string(a);
}

/**
 * Element count as cheatah's signed integer (ndarray's `size_of`).
 * @param a The device array.
 * @return The element count.
 * @complexity O(1). @alloc none. @gpualloc none.
 */
template <ndarray::Field T>
[[nodiscard]] long long size_of(const device_array<T>& a) {
    return static_cast<long long>(a.size());
}
/**
 * Device arrays are always contiguous (parity with ndarray's `is_contiguous`).
 * @param a The device array (unused — contiguity is structural).
 * @return true, always.
 * @complexity O(1). @alloc none. @gpualloc none.
 */
template <ndarray::Field T>
[[nodiscard]] constexpr bool is_contiguous(const device_array<T>& a) {
    (void)a;
    return true;
}

/**
 * Element conversion (ndarray's `astype`) — HOST round-trip today (documented; a device convert
 * kernel is a listed next step).
 * @param a The device array to convert.
 * @return A fresh device array with each element cast to `U`.
 * @complexity O(n): download, convert on the host, upload.
 * @alloc two O(n) host vectors (the download and the converted copy).
 * @gpualloc one pooled device data buffer for the result.
 */
template <class U, ndarray::Field T>
    requires std::convertible_to<T, U>
[[nodiscard]] device_array<U> astype(const device_array<T>& a) {
    std::vector<T> tmp(a.size());
    a.to_host(tmp.data());
    std::vector<U> conv(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) conv[i] = static_cast<U>(tmp[i]);
    return device_array<U>::from_host(a.shape(), conv.data());
}

/**
 * Host mirror of the fused device `axpy` (out = α·x + y, out may alias x or y) so a generic
 * `axpy(...)` call is location-blind — on host arrays it is a single contiguous loop.
 * @param out   The host destination (may alias @p x or @p y; throws on a shape mismatch).
 * @param alpha The scalar coefficient.
 * @param x     The scaled host operand.
 * @param y     The added host operand.
 * @complexity O(n) on the CPU.
 * @alloc none — writes straight into @p out.
 * @gpualloc none — never touches the device.
 * @test example:linear_regression_gd
 */
template <ndarray::Field T>
void axpy(host_array<T>& out, T alpha, const host_array<T>& x, const host_array<T>& y) {
    if (x.shape() != y.shape() || out.shape() != x.shape())
        throw std::runtime_error("cheatah-gpu-linalg axpy: shape mismatch");
    const T* xp = x.buffer()->data() + x.offset();
    const T* yp = y.buffer()->data() + y.offset();
    T* op = out.buffer()->data() + out.offset();
    const std::size_t n = out.size();
    for (std::size_t i = 0; i < n; ++i) op[i] = alpha * xp[i] + yp[i];
}

}  // namespace cheatah::gpu::linalg
