// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
#pragma once

/**
 * @file bridge.hpp
 * @brief The factorization/solver surface for device arrays — an HONEST host bridge.
 *
 * Every routine in this file EXECUTES ON THE HOST: the device operands are downloaded, cheatah's
 * own host algorithm (LU / Householder QR / Golub–Reinsch SVD / tridiagonal-QL / Hessenberg-QR)
 * runs on them, and the results are uploaded back into the device out-params. The bridge exists so
 * the FULL `cheatah::linalg` surface is *available* for device-resident data — a training loop can
 * keep everything in `device_array` and still call `solve`/`svd`/`eig` — not so those calls are
 * fast. True device factorization kernels are future work; benchmarks list bridge ops separately.
 *
 * Mechanics: cheatah's allocating fronts (routines.hpp) validate metadata and allocate the device
 * outputs via `device_array<T>::uninitialized`, then call the out-param kernel unqualified — ADL
 * lands here because these overloads live in `device_array`'s namespace and require `DeviceArray`.
 * The bridge always computes through cheatah's SHIPPED host instantiations (`double` /
 * `std::complex<double>`), converting f32 elements at the download/upload boundary — so cheatah's
 * explicit-instantiation list is untouched and f32 inputs only GAIN precision in the middle.
 *
 * Include note: this header includes cheatah's `"linalg.hpp"` UMBRELLA — never a bare
 * `"routines.hpp"`, which quote-lookup would resolve to THIS repo's own routines.hpp.
 */

#include "cheatah_gpu_linalg/device_array.hpp"

#include "linalg.hpp"   // cheatah stdlib/linalg umbrella: fronts + host kernels + result structs

#include <complex>
#include <cstddef>
#include <vector>
#include <type_traits>
#include <vector>

namespace cheatah::gpu::linalg {

/// BridgeElement<T>: the element types the host bridge accepts — every real/complex float width,
/// because no device kernel is involved (contrast GpuElement, which gates real device kernels).
template <class T>
concept BridgeElement =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;

namespace detail {

/// The host element the bridge computes in for a device element `T`: cheatah's shipped `double`
/// base (complex stays complex). f32 widens on download and narrows on upload.
template <class T>
using host_elem_t =
    std::conditional_t<cheatah::ndarray::is_complex_v<T>, std::complex<double>, double>;

/// Download a device array into a host `basic_ndarray<host_elem_t<T>>` (same shape, converted
/// element-wise; a straight memcpy-shaped loop — device_array is contiguous row-major).
template <BridgeElement T>
[[nodiscard]] inline cheatah::ndarray::basic_ndarray<host_elem_t<T>> download(
    const device_array<T>& a) {
    using H = host_elem_t<T>;
    auto h = cheatah::ndarray::basic_ndarray<H>::uninitialized(a.shape());
    if constexpr (std::is_same_v<T, H>) {
        a.to_host(h.buffer()->data());
    } else {
        std::vector<T> tmp(a.size());
        a.to_host(tmp.data());
        H* dst = h.buffer()->data();
        for (std::size_t i = 0; i < a.size(); ++i) dst[i] = static_cast<H>(tmp[i]);
    }
    return h;
}

/// Upload a host result into the pre-allocated device out-param (element-converted; shapes match
/// by construction — cheatah's front allocated `dst` from the same metadata the host op used).
template <BridgeElement T>
inline void upload(device_array<T>& dst, const cheatah::ndarray::basic_ndarray<host_elem_t<T>>& src) {
    using H = host_elem_t<T>;
    const H* s = src.buffer()->data() + src.offset();
    if constexpr (std::is_same_v<T, H>) {
        dst.write_from(s);
    } else {
        std::vector<T> tmp(src.size());
        for (std::size_t i = 0; i < src.size(); ++i) tmp[i] = static_cast<T>(s[i]);
        dst.write_from(tmp.data());
    }
}

/// Upload for outputs whose element differs from the input's (eig's complex outs from a real
/// input; eigh's real values from a complex input): converts host element `H` to device `U`.
template <class U, class H>
inline void upload_as(device_array<U>& dst, const cheatah::ndarray::basic_ndarray<H>& src) {
    const H* s = src.buffer()->data() + src.offset();
    if constexpr (std::is_same_v<U, H>) {
        dst.write_from(s);
    } else {
        std::vector<U> tmp(src.size());
        for (std::size_t i = 0; i < src.size(); ++i) tmp[i] = static_cast<U>(s[i]);
        dst.write_from(tmp.data());
    }
}

}  // namespace detail

// ---- single-output factorizations / solvers (HOST-BRIDGED — see the file comment) -------------

/**
 * Device cholesky — host bridge (download → cheatah host kernel → upload).
 *
 * @param out The pre-allocated n×n device factor L (filled by upload).
 * @param a   The n×n symmetric positive-definite device operand (throws otherwise, from the host
 *            kernel).
 * @complexity host O(n³) (cheatah's column-by-column Cholesky) + O(n²) elements over the bus.
 * @alloc host f64 copies of @p a and the factor (O(n²) each; an f32 element adds one O(n²)
 *        conversion vector per transfer) + the host kernel's own O(n²) private factor scratch.
 * @gpualloc none — @p out was allocated by cheatah's front; transfers recycle the context's
 *           pooled staging buffers.
 * @test gpu:bridge
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && BridgeElement<T>
void cholesky(Array<T>& out, const Array<T>& a) {
    auto ha = detail::download(a);
    auto h = decltype(ha)::uninitialized(ha.shape());
    cheatah::linalg::cholesky(h, ha);
    detail::upload(out, h);
}

/**
 * Device inv — host bridge (LU with partial pivoting + whole-identity back-solve).
 *
 * @param out The pre-allocated n×n device inverse.
 * @param a   The square device operand (a singular matrix throws, from the host kernel).
 * @complexity host O(n³) + O(n²) elements over the bus.
 * @alloc host f64 copies of @p a and the result (O(n²) each) + the host kernel's LU and
 *        identity back-solve scratch (O(n²)); f32 adds conversion vectors.
 * @gpualloc none — @p out pre-allocated by cheatah's front; pooled staging only.
 * @test gpu:bridge
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && BridgeElement<T>
void inv(Array<T>& out, const Array<T>& a) {
    auto ha = detail::download(a);
    auto h = decltype(ha)::uninitialized(ha.shape());
    cheatah::linalg::inv(h, ha);
    detail::upload(out, h);
}

/**
 * Device pinv — host bridge (Moore–Penrose pseudo-inverse via Golub–Reinsch SVD; the result is
 * the transposed shape n×m for an m×n input).
 *
 * @param out The pre-allocated n×m device pseudo-inverse.
 * @param a   The m×n device operand (any shape).
 * @complexity host iterative O(n³) via SVD + O(m·n) elements over the bus.
 * @alloc host f64 copies of @p a and the result + the host kernel's SVD/assembly scratch
 *        (O(m·n)); f32 adds conversion vectors.
 * @gpualloc none — @p out pre-allocated by cheatah's front; pooled staging only.
 * @test gpu:bridge
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && BridgeElement<T>
void pinv(Array<T>& out, const Array<T>& a) {
    auto ha = detail::download(a);
    auto h = decltype(ha)::uninitialized({a.shape()[1], a.shape()[0]});
    cheatah::linalg::pinv(h, ha);
    detail::upload(out, h);
}

/**
 * Device matrix_power — host bridge (binary exponentiation; negative @p n via the host inverse).
 *
 * @param out The pre-allocated square device result Aⁿ.
 * @param a   The square device operand.
 * @param n   The integer exponent (0 yields the identity).
 * @complexity host O(n³·log|n|) by binary exponentiation + O(n²) elements over the bus.
 * @alloc host f64 copies of @p a and the result + the binary-exponentiation products' own
 *        intermediates on the host; f32 adds conversion vectors.
 * @gpualloc none — @p out pre-allocated by cheatah's front; pooled staging only.
 * @test gpu:bridge
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && BridgeElement<T>
void matrix_power(Array<T>& out, const Array<T>& a, long long n) {
    auto ha = detail::download(a);
    auto h = decltype(ha)::uninitialized(ha.shape());
    cheatah::linalg::matrix_power(h, ha, n);
    detail::upload(out, h);
}

/**
 * Device solve — host bridge (x from A·x = b via LU with partial pivoting).
 *
 * @param out The pre-allocated length-n device solution vector.
 * @param a   The n×n device coefficient matrix.
 * @param b   The length-n device right-hand side.
 * @complexity host O(n³) + O(n²) elements over the bus.
 * @alloc host f64 copies of @p a, @p b and x + the host kernel's O(n²) LU scratch; f32 adds
 *        conversion vectors.
 * @gpualloc none — @p out pre-allocated by cheatah's front; pooled staging only.
 * @test gpu:bridge
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && BridgeElement<T>
void solve(Array<T>& out, const Array<T>& a, const Array<T>& b) {
    auto ha = detail::download(a);
    auto hb = detail::download(b);
    auto h = decltype(ha)::uninitialized({a.shape()[0]});
    cheatah::linalg::solve(h, ha, hb);
    detail::upload(out, h);
}

/**
 * Device lstsq — host bridge (min ‖Ax−b‖, computed as pinv(A)·b on the host).
 *
 * @param out The pre-allocated device solution (a.cols × b.cols).
 * @param a   The m×n device coefficient matrix.
 * @param b   The m×k device right-hand side.
 * @complexity host iterative O(n³) via SVD + O(m·n) elements over the bus.
 * @alloc host f64 copies of @p a, @p b and x + the intermediate n×m pseudo-inverse and its SVD
 *        scratch on the host; f32 adds conversion vectors.
 * @gpualloc none — @p out pre-allocated by cheatah's front; pooled staging only.
 * @test gpu:bridge
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && BridgeElement<T>
void lstsq(Array<T>& out, const Array<T>& a, const Array<T>& b) {
    auto ha = detail::download(a);
    auto hb = detail::download(b);
    auto h = decltype(ha)::uninitialized({a.shape()[1], b.shape()[1]});
    cheatah::linalg::lstsq(h, ha, hb);
    detail::upload(out, h);
}

/**
 * Device svdvals — host bridge (descending singular values, length min(m,n) — the values-only
 * fast path of the Golub–Reinsch reduction, a large constant factor below the full svd).
 *
 * @param out The pre-allocated length-min(m,n) device singular-value vector.
 * @param a   The m×n device operand.
 * @complexity host iterative O(n³) + O(m·n) elements over the bus.
 * @alloc host f64 copies of @p a and the values + the reduction's own O(m·n) workspace on the
 *        host; f32 adds conversion vectors.
 * @gpualloc none — @p out pre-allocated by cheatah's front; pooled staging only.
 * @test gpu:bridge
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && BridgeElement<T>
void svdvals(Array<T>& out, const Array<T>& a) {
    auto ha = detail::download(a);
    const std::size_t m = a.shape()[0], n = a.shape()[1];
    auto h = decltype(ha)::uninitialized({m < n ? m : n});
    cheatah::linalg::svdvals(h, ha);
    detail::upload(out, h);
}

// ---- multi-output decompositions (HOST-BRIDGED) ------------------------------------------------

/**
 * Device qr — host bridge (reduced Householder QR; requires rows ≥ cols, from the host kernel).
 *
 * @param q The pre-allocated m×n device orthonormal factor.
 * @param r The pre-allocated n×n device upper-triangular factor.
 * @param a The m×n device operand.
 * @complexity host O(n³) + O(m·n) elements over the bus.
 * @alloc host f64 copies of @p a and both factors + the host kernel's O(m²) private
 *        factorization scratch; f32 adds conversion vectors.
 * @gpualloc none — @p q and @p r pre-allocated by cheatah's front; pooled staging only.
 * @test gpu:bridge
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && BridgeElement<T>
void qr(Array<T>& q, Array<T>& r, const Array<T>& a) {
    auto ha = detail::download(a);
    const std::size_t m = a.shape()[0], n = a.shape()[1];
    auto hq = decltype(ha)::uninitialized({m, n});
    auto hr = decltype(ha)::uninitialized({n, n});
    cheatah::linalg::qr(hq, hr, ha);
    detail::upload(q, hq);
    detail::upload(r, hr);
}

/**
 * Device svd — host bridge (Golub–Reinsch; requires rows ≥ cols, from the host kernel).
 *
 * @param u  The pre-allocated m×n device left-singular-vector factor.
 * @param s  The pre-allocated length-n device singular-value vector (descending).
 * @param vh The pre-allocated n×n device Vᴴ factor.
 * @param a  The m×n device operand.
 * @complexity host iterative O(n³) + O(m·n) elements over the bus.
 * @alloc host f64 copies of @p a and all three factors + the reduction's own O(m·n) workspace on
 *        the host; f32 adds conversion vectors.
 * @gpualloc none — outputs pre-allocated by cheatah's front; pooled staging only.
 * @test gpu:bridge
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && BridgeElement<T>
void svd(Array<T>& u, Array<T>& s, Array<T>& vh, const Array<T>& a) {
    auto ha = detail::download(a);
    const std::size_t m = a.shape()[0], n = a.shape()[1];
    auto hu = decltype(ha)::uninitialized({m, n});
    auto hs = decltype(ha)::uninitialized({n});
    auto hv = decltype(ha)::uninitialized({n, n});
    cheatah::linalg::svd(hu, hs, hv, ha);
    detail::upload(u, hu);
    detail::upload(s, hs);
    detail::upload(vh, hv);
}

/**
 * Device eig — host bridge (Hessenberg + shifted QR; COMPLEX outputs from a real or complex
 * square input, since a real matrix can have complex eigenpairs).
 *
 * @param values  The pre-allocated length-n complex device spectrum.
 * @param vectors The pre-allocated n×n complex device eigenvector matrix (columns).
 * @param a       The n×n device operand.
 * @complexity host iterative O(n³) + O(n²) elements over the bus.
 * @alloc host c128 copies of @p a's widening plus both outputs + the host kernel's O(n²)
 *        factorization/eigenvector scratch; narrower elements add conversion vectors.
 * @gpualloc none — outputs pre-allocated by cheatah's front; pooled staging only.
 * @test gpu:bridge
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && BridgeElement<T>
void eig(Array<ndarray::complex_of_t<T>>& values, Array<ndarray::complex_of_t<T>>& vectors,
         const Array<T>& a) {
    auto ha = detail::download(a);
    const std::size_t n = a.shape()[0];
    auto hvals = cheatah::ndarray::basic_ndarray<std::complex<double>>::uninitialized({n});
    auto hvecs = cheatah::ndarray::basic_ndarray<std::complex<double>>::uninitialized({n, n});
    cheatah::linalg::eig(hvals, hvecs, ha);
    detail::upload_as(values, hvals);
    detail::upload_as(vectors, hvecs);
}

/**
 * Device eigvals — host bridge (the COMPLEX spectrum of a real or complex square matrix,
 * descending; symmetric input routes through tridiagonal QL on the host, the rest through
 * Hessenberg + shifted QR).
 *
 * @param out The pre-allocated length-n complex device spectrum.
 * @param a   The n×n device operand.
 * @complexity host iterative O(n³) + O(n²) elements over the bus.
 * @alloc host c128 copies of the operand and spectrum + the host iteration's own O(n²) scratch;
 *        narrower elements add conversion vectors.
 * @gpualloc none — @p out pre-allocated by cheatah's front; pooled staging only.
 * @test gpu:bridge
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && BridgeElement<T>
void eigvals(Array<ndarray::complex_of_t<T>>& out, const Array<T>& a) {
    auto ha = detail::download(a);
    auto h = cheatah::ndarray::basic_ndarray<std::complex<double>>::uninitialized({a.shape()[0]});
    cheatah::linalg::eigvals(h, ha);
    detail::upload_as(out, h);
}

/**
 * Device eigh — host bridge (Householder tridiagonalization + QL for a symmetric/Hermitian
 * input: a REAL spectrum plus eigenvectors matching the input's element type).
 *
 * @param values  The pre-allocated length-n REAL device spectrum (descending).
 * @param vectors The pre-allocated n×n device eigenvector matrix (input's element type).
 * @param a       The n×n symmetric/Hermitian device operand.
 * @complexity host iterative O(n³) + O(n²) elements over the bus.
 * @alloc host f64/c128 copies of the operand and both outputs + the host solver's own O(n²)
 *        scratch; narrower elements add conversion vectors.
 * @gpualloc none — outputs pre-allocated by cheatah's front; pooled staging only.
 * @test gpu:bridge
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && BridgeElement<T>
void eigh(Array<ndarray::real_base_t<T>>& values, Array<T>& vectors, const Array<T>& a) {
    auto ha = detail::download(a);
    const std::size_t n = a.shape()[0];
    auto hvals = cheatah::ndarray::basic_ndarray<double>::uninitialized({n});
    auto hvecs = decltype(ha)::uninitialized({n, n});
    cheatah::linalg::eigh(hvals, hvecs, ha);
    detail::upload_as(values, hvals);
    detail::upload(vectors, hvecs);
}

/**
 * Device eigvalsh — host bridge (the REAL spectrum of a symmetric/Hermitian matrix, descending —
 * tridiagonal QL skipping the eigenvector accumulation).
 *
 * @param out The pre-allocated length-n REAL device spectrum.
 * @param a   The n×n symmetric/Hermitian device operand.
 * @complexity host iterative O(n³) + O(n²) elements over the bus.
 * @alloc host copies of the operand and spectrum + the host solver's own O(n²) scratch;
 *        narrower elements add conversion vectors.
 * @gpualloc none — @p out pre-allocated by cheatah's front; pooled staging only.
 * @test gpu:bridge
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && BridgeElement<T>
void eigvalsh(Array<ndarray::real_base_t<T>>& out, const Array<T>& a) {
    auto ha = detail::download(a);
    auto h = cheatah::ndarray::basic_ndarray<double>::uninitialized({a.shape()[0]});
    cheatah::linalg::eigvalsh(h, ha);
    detail::upload_as(out, h);
}

// ---- scalar-out routines (HOST-BRIDGED) --------------------------------------------------------

/**
 * Device det — host bridge (LU with partial pivoting; pivot product × permutation sign).
 *
 * @param out The caller's scalar determinant (narrowed back to @p a's element type).
 * @param a   The square device operand.
 * @complexity host O(n³) + O(n²) elements downloaded.
 * @alloc one host f64 copy of @p a + the host kernel's O(n²) LU scratch; f32 adds a conversion
 *        vector.
 * @gpualloc none — download-only; pooled staging.
 * @test gpu:bridge
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && BridgeElement<T>
void det(T& out, const Array<T>& a) {
    detail::host_elem_t<T> h;
    cheatah::linalg::det(h, detail::download(a));
    out = static_cast<T>(h);
}

/**
 * Device slogdet — host bridge (sign and log|det| via LU — the overflow-safe determinant).
 *
 * @param out The caller's SLogDet result (plain doubles).
 * @param a   The square device operand.
 * @complexity host O(n³) + O(n²) elements downloaded.
 * @alloc one host f64 copy of @p a + the host kernel's O(n²) LU scratch; f32 adds a conversion
 *        vector.
 * @gpualloc none — download-only; pooled staging.
 * @test gpu:bridge
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && BridgeElement<T>
void slogdet(cheatah::linalg::SLogDet& out, const Array<T>& a) {
    cheatah::linalg::slogdet(out, detail::download(a));
}

/**
 * Device cond — host bridge (largest/smallest singular-value ratio from a Golub–Reinsch SVD).
 *
 * @param out The caller's scalar condition number.
 * @param a   The device operand.
 * @complexity host iterative O(n³) via SVD + O(n²) elements downloaded.
 * @alloc one host f64 copy of @p a + the host kernel's O(n²) factorization scratch; f32 adds a
 *        conversion vector.
 * @gpualloc none — download-only; pooled staging.
 * @test gpu:bridge
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && BridgeElement<T>
void cond(T& out, const Array<T>& a) {
    detail::host_elem_t<T> h;
    cheatah::linalg::cond(h, detail::download(a));
    out = static_cast<T>(h);
}

/**
 * Device matrix_rank — host bridge (SVD singular-value thresholding).
 *
 * @param out The caller's rank (cheatah's signed integer).
 * @param a   The device operand.
 * @complexity host iterative O(n³) via SVD + O(n²) elements downloaded.
 * @alloc one host f64 copy of @p a + the host kernel's O(n²) factorization scratch; f32 adds a
 *        conversion vector.
 * @gpualloc none — download-only; pooled staging.
 * @test gpu:bridge
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && BridgeElement<T>
void matrix_rank(long long& out, const Array<T>& a) {
    cheatah::linalg::matrix_rank(out, detail::download(a));
}

/**
 * Device norm — host bridge (L2 for vectors, Frobenius for matrices — dispatched on rank by the
 * host kernel).
 *
 * @param out The caller's scalar norm.
 * @param a   The device operand (any rank).
 * @complexity host O(size) over the elements + the same O(size) downloaded.
 * @alloc one host f64 copy of @p a (the host kernel itself sums in place); f32 adds a
 *        conversion vector.
 * @gpualloc none — download-only; pooled staging.
 * @test gpu:bridge
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && BridgeElement<T>
void norm(T& out, const Array<T>& a) {
    detail::host_elem_t<T> h;
    cheatah::linalg::norm(h, detail::download(a));
    out = static_cast<T>(h);
}

}  // namespace cheatah::gpu::linalg
