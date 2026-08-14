// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
#pragma once

/**
 * @file device_array.hpp
 * @brief `device_array<T>` — the GPU-resident container that plugs into cheatah's linalg seam.
 *
 * cheatah's `stdlib/linalg` is written as one two-layer template per operation,
 * `template <ndarray::Field T, template <typename> class Array> ... Array<T> ...`, and dispatches
 * host vs device purely at compile time: an array-like models a structural `ArrayLike` surface and
 * advertises WHERE its elements live through the `location_of` trait. cheatah defines only the host
 * tag; a device extension defines its own tag and specializes `location_of` for its container. That
 * alone makes the container satisfy `cheatah::linalg::DeviceArray` (and NOT `HostArray`), so the
 * allocating fronts (e.g. `linalg::matmul(a, b)`) resolve — by ADL, no CPO — to the device kernels
 * this library supplies (see routines.hpp).
 *
 * `device_array` keeps its shape/strides on the host (that metadata is tiny and every linalg body
 * reads it on the CPU) while the elements live in a cheatah-gpu device buffer. Element storage is
 * row-major (C-order) and contiguous — the layout the GEMM kernel and future kernels assume.
 */

#include "cheatah_gpu_linalg/context.hpp"

#include "concepts.hpp"   // cheatah stdlib/linalg: location_of, host_location, ArrayLike, DeviceArray
#include "ndarray.hpp"    // cheatah stdlib/ndarray: Field / Element

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <vector>

namespace cheatah::gpu::linalg {

/// The location tag for `device_array`. cheatah names only `host_location`; this is the device
/// counterpart the seam is designed to accept.
struct device_location {};

/**
 * A dense, row-major, GPU-resident array. Models cheatah's `ArrayLike` (host-resident shape /
 * strides / ndim / size / offset + a `uninitialized` factory) so it flows through the linalg
 * templates; its elements live in a shared-storage cheatah-gpu buffer.
 */
template <class T>
class device_array {
public:
    /// The element type (the standard-container spelling of `T`).
    using value_type = T;

    /// An empty array: no shape, no buffer (assign a factory result to make it real).
    device_array() = default;

    /**
     * Allocate an uninitialized array of the given shape (the factory the allocating linalg
     * fronts call as `Array<T>::uninitialized({rows, cols})`). Throws on an element-count
     * overflow and past the 2³²−1 element cap (the 32-bit kernel-indexing invariant).
     * @param shape The dimensions, row-major.
     * @return A fresh array whose elements are uninitialized device memory.
     * @complexity O(ndim) host metadata work; the buffer is a pooled acquire (hash lookup when
     *             recycled, one driver allocation otherwise).
     * @alloc the shape/strides vectors (O(ndim)).
     * @gpualloc one pooled device data buffer of size()·sizeof(T) — VRAM (device-local) on
     *           Vulkan, shared storage on Metal; released to the pool by the last owner's
     *           destructor.
     * @test gpu:matmul
     */
    static device_array uninitialized(std::vector<std::size_t> shape) {
        device_array a;
        a.shape_ = std::move(shape);
        a.size_ = 1;
        for (std::size_t d : a.shape_) {
            // Overflow-checked product: an overflowed size_ would under-allocate and turn the
            // first upload into a heap overflow (audit 2026-07-17).
            if (d != 0 && a.size_ > (std::numeric_limits<std::size_t>::max)() / d)
                throw std::runtime_error("cheatah-gpu-linalg: shape element-count overflow");
            a.size_ *= d;
        }
        // Kernel index math is 32-bit (row * N + col etc.): capping ELEMENTS at 2^32-1 makes
        // every in-kernel index/product provably non-wrapping.
        if (a.size_ > 0xFFFFFFFFull)
            throw std::runtime_error(
                "cheatah-gpu-linalg: arrays are limited to 2^32-1 elements (32-bit kernel indexing)");
        a.strides_ = c_order_strides(a.shape_);
        a.buf_ = std::shared_ptr<detail::Buffer>(
            detail::ctx().new_data_buffer(a.size_ * sizeof(T)),
            [](detail::Buffer* b) { if (b) detail::ctx().release_buffer(b); });
        return a;
    }

    /**
     * Allocate and upload `size()` contiguous row-major elements from host memory.
     * @param shape The dimensions, row-major.
     * @param data  The host elements (size() of them, contiguous).
     * @return A fresh device array holding a copy of @p data.
     * @complexity O(n) transfer.
     * @alloc the O(ndim) metadata; the Vulkan upload stages through a pooled host-visible buffer.
     * @gpualloc one pooled device data buffer of size()·sizeof(T).
     * @test gpu:matmul
     */
    static device_array from_host(std::vector<std::size_t> shape, const T* data) {
        device_array a = uninitialized(std::move(shape));
        a.write_from(data);
        return a;
    }

    /**
     * Upload `size()` contiguous elements into this (already-allocated) array — device-local
     * memory on Vulkan goes through a staged transfer; unified memory is a memcpy.
     * @param data The host elements (size() of them, contiguous; bounds-checked against the
     *             allocation).
     * @complexity O(n) transfer.
     * @alloc none fresh — the Vulkan staging buffer recycles through the pool.
     * @gpualloc none — writes into this array's existing buffer.
     * @test gpu:bridge
     */
    void write_from(const T* data) {
        detail::ctx().upload(buf_.get(), data, size_ * sizeof(T));
    }

    /**
     * Copy `size()` elements back to host memory (the download mirror of @ref write_from —
     * a SYNC POINT).
     * @param out The host destination (size() elements of room).
     * @complexity O(n) transfer.
     * @alloc none fresh — the Vulkan readback stages through a pooled host-cached buffer.
     * @gpualloc none.
     * @test gpu:matmul
     */
    void to_host(T* out) const { detail::ctx().download(buf_.get(), out, size_ * sizeof(T)); }

    // --- ArrayLike structural surface (all host-resident metadata) --------------------------------
    /// @return The dimensions, row-major. @complexity O(1). @alloc none.
    [[nodiscard]] const std::vector<std::size_t>& shape() const { return shape_; }
    /// @return The element strides (C-order, in elements). @complexity O(1). @alloc none.
    [[nodiscard]] const std::vector<std::ptrdiff_t>& strides() const { return strides_; }
    /// @return The rank. @complexity O(1). @alloc none.
    [[nodiscard]] std::size_t ndim() const { return shape_.size(); }
    /// @return The element count. @complexity O(1). @alloc none.
    [[nodiscard]] std::size_t size() const { return size_; }
    /// @return The view offset — always 0 (device arrays are whole-buffer views).
    /// @complexity O(1). @alloc none.
    [[nodiscard]] std::size_t offset() const { return 0; }

    /**
     * A new view of the SAME buffer under a different (equal-size) contiguous shape — the
     * zero-copy backend of `reshape` (shares ownership through the same shared_ptr).
     * @param shape The new dimensions (caller guarantees the same element count — `reshape`
     *              validates).
     * @return A sibling array sharing this one's device buffer.
     * @complexity O(ndim).
     * @alloc the new shape/strides vectors.
     * @gpualloc none — zero-copy by design.
     * @test gpu:elementwise
     */
    [[nodiscard]] device_array with_shape(std::vector<std::size_t> shape) const {
        device_array v;
        v.buf_ = buf_;
        v.shape_ = std::move(shape);
        v.size_ = size_;
        v.strides_ = c_order_strides(v.shape_);
        return v;
    }

    // --- device access used by the kernels (NOT part of the linalg concept surface) ---------------
    /// @return The backend buffer handle (nullptr for an empty array) — NON-owning: the array
    ///         keeps ownership, never pass it to release_buffer. @complexity O(1). @alloc none.
    [[nodiscard]] detail::Buffer* buffer() const { return buf_.get(); }

private:
    static std::vector<std::ptrdiff_t> c_order_strides(const std::vector<std::size_t>& shape) {
        std::vector<std::ptrdiff_t> s(shape.size());
        std::ptrdiff_t acc = 1;
        for (std::size_t i = shape.size(); i-- > 0;) {
            s[i] = acc;
            acc *= static_cast<std::ptrdiff_t>(shape[i]);
        }
        return s;
    }

    /// Shared owner of the backend buffer: the deleter releases it through the active backend's
    /// context when the last `device_array` sharing it goes away.
    std::shared_ptr<detail::Buffer> buf_;
    std::vector<std::size_t> shape_;
    std::vector<std::ptrdiff_t> strides_;
    std::size_t size_ = 0;
};

}  // namespace cheatah::gpu::linalg

namespace cheatah::linalg {
/// Teach cheatah's location trait that `device_array` lives on the device. This one specialization
/// is the entire opt-in: it makes `device_array<T>` satisfy `DeviceArray` and fail `HostArray`, so
/// the linalg fronts route to this library's kernels without cheatah ever naming it.
template <class T>
struct location_of<cheatah::gpu::linalg::device_array<T>> {
    /// The device tag — what flips the DeviceArray/HostArray concept split.
    using type = cheatah::gpu::linalg::device_location;
};
}  // namespace cheatah::linalg
