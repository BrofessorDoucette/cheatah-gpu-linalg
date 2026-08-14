// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
#pragma once

/**
 * @file conv.hpp
 * @brief Device conv-support routines — im2col2d/col2im2d + the fused conv epilogues.
 *
 * The four kernels a conv layer needs AROUND the GEMM, device-resident: the im2col staging
 * gather and its adjoint (the data movement a host-side lowering pays PCIe ping-pong for), and
 * the fused forward/backward epilogues (bias + activation + the filter-major ↔ sample-major
 * layout transpose in one pass each). All four mirror the neural-net consumer's host reference
 * loops element-for-element — same layouts, same zero-padding, same association order — so a
 * training step can swap its host staging for these with bitwise-matching data movement.
 *
 * Layout contract (row-major, contiguous — device_array's layout):
 *   - x / dx   [B, C·H·W]        laid out [B][C][H][W];
 *   - col/dcol [C·KH·KW, B·OH·OW] — column b·OH·OW + oh·OW + ow is output position (oh, ow) of
 *     sample b, row c·KH·KW + kh·KW + kw its (channel, kernel-offset) coordinate;
 *   - yc / dyc [F, B·OO]         (filter-major, the GEMM's output side);
 *   - a / d    [B, F·OO]         (sample-major, the stack's block side);  OO = OH·OW.
 *
 * Every kernel writes each output element from exactly one thread — deterministic by
 * construction, no atomics (col2im2d is the GATHER form of the scatter-add adjoint).
 * The activation selector: 0 = identity, 1 = relu, 2 = tanh, 3 = sigmoid; derivatives are
 * evaluated FROM THE OUTPUT a = f(z), so no pre-activations are retained.
 */

#include "cheatah_gpu_linalg/device_array.hpp"
#include "cheatah_gpu_linalg/kernels.hpp"
#include "cheatah_gpu_linalg/routines.hpp"   // GpuElement + detail::dims_buffer

#include <cstdint>
#include <stdexcept>
#include <string>

namespace cheatah::gpu::linalg {

namespace detail {

/// Throw unless `got` elements are exactly `want` (the conv routines' size contract — geometry
/// is passed explicitly, so a mismatched operand is caller error, reported with the op's name).
inline void require_conv_size(const char* op, const char* operand, std::size_t got,
                              std::size_t want) {
    if (got != want)
        throw std::runtime_error(std::string("cheatah-gpu-linalg ") + op + ": " + operand +
                                 " has wrong element count");
}

/// Throw unless the activation selector is one of the four codes the kernels implement.
inline void require_act_code(const char* op, long long act) {
    if (act < 0 || act > 3)
        throw std::runtime_error(std::string("cheatah-gpu-linalg ") + op +
                                 ": act code must be 0..3");
}

}  // namespace detail

/**
 * Device batched 2-D im2col GATHER into the caller's column block:
 * `col[(c·KH+kh)·KW+kw, b·OO + oh·OW + ow] = x[b, c, oh·stride − pad + kh, ow·stride − pad + kw]`
 * with out-of-plane (padding) cells written as ZEROS — a FULL overwrite, so a reused workspace
 * needs no zeroing pass. One thread per col element, writes contiguous. Mirrors the host
 * reference's element order and zero-padding exactly (pure data movement — bit-identical).
 *
 * @param col    The [C·KH·KW, B·OH·OW]-element device destination (workspace, overwritten).
 * @param x      The [B, C·H·W]-element device input block, laid out [B][C][H][W].
 * @param B      The batch size.
 * @param C      The input channel count.
 * @param H      The input plane height.
 * @param W      The input plane width.
 * @param KH     The kernel height.
 * @param KW     The kernel width.
 * @param OH     The output plane height ((H + 2·pad − KH)/stride + 1).
 * @param OW     The output plane width.
 * @param stride The spatial step (≥ 1).
 * @param pad    The symmetric zero padding (≥ 0). Throws on an operand size mismatch.
 * @complexity O(B·C·KH·KW·OH·OW) device work in one blocking dispatch.
 * @alloc none — the 10-uint dims scratch recycles through the pool.
 * @gpualloc none — both data buffers are the caller's.
 * @test gpu:conv
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && GpuElement<T>
void im2col2d(Array<T>& col, const Array<T>& x, long long B, long long C, long long H,
              long long W, long long KH, long long KW, long long OH, long long OW,
              long long stride, long long pad) {
    const std::size_t n = static_cast<std::size_t>(C * KH * KW) * static_cast<std::size_t>(B * OH * OW);
    detail::require_conv_size("im2col2d", "col", col.size(), n);
    detail::require_conv_size("im2col2d", "x", x.size(), static_cast<std::size_t>(B * C * H * W));
    if (n == 0) return;
    const std::uint32_t dims[10] = {
        static_cast<std::uint32_t>(B),  static_cast<std::uint32_t>(C),
        static_cast<std::uint32_t>(H),  static_cast<std::uint32_t>(W),
        static_cast<std::uint32_t>(KH), static_cast<std::uint32_t>(KW),
        static_cast<std::uint32_t>(OH), static_cast<std::uint32_t>(OW),
        static_cast<std::uint32_t>(stride), static_cast<std::uint32_t>(pad)};
    detail::Context& c = detail::ctx();
    detail::Buffer* dimbuf = detail::dims_buffer(dims, 10);
    detail::Buffer* bind[3] = {x.buffer(), col.buffer(), dimbuf};
    c.dispatch_1d(kernels::im2col2d_name<T>, bind, 3, n);
    c.release_buffer(dimbuf);
}

/**
 * Device batched 2-D col2im — the EXACT adjoint of @ref im2col2d, in GATHER form: each dx cell
 * sums its ≤ KH·KW contributing dcol cells (skipping stride-incompatible and out-of-range
 * kernel offsets), one thread per dx element, FULL overwrite (the scatter-add's zeroing folded
 * in). The kh-major accumulation order matches the host reference's scatter-add per cell, so
 * the result is bit-identical — and deterministic by construction, no atomics.
 *
 * @param dx     The [B, C·H·W]-element device input-gradient block, overwritten.
 * @param dcol   The [C·KH·KW, B·OH·OW]-element device column gradients.
 * @param B      The batch size.
 * @param C      The input channel count.
 * @param H      The input plane height.
 * @param W      The input plane width.
 * @param KH     The kernel height.
 * @param KW     The kernel width.
 * @param OH     The output plane height.
 * @param OW     The output plane width.
 * @param stride The spatial step (≥ 1).
 * @param pad    The symmetric zero padding (≥ 0). Throws on an operand size mismatch.
 * @complexity O(B·C·H·W·KH·KW) device work in one blocking dispatch (≤ KH·KW adds per cell).
 * @alloc none — the 10-uint dims scratch recycles through the pool.
 * @gpualloc none — both data buffers are the caller's.
 * @test gpu:conv
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && GpuElement<T>
void col2im2d(Array<T>& dx, const Array<T>& dcol, long long B, long long C, long long H,
              long long W, long long KH, long long KW, long long OH, long long OW,
              long long stride, long long pad) {
    const std::size_t n = static_cast<std::size_t>(B * C * H * W);
    detail::require_conv_size("col2im2d", "dx", dx.size(), n);
    detail::require_conv_size("col2im2d", "dcol", dcol.size(),
                              static_cast<std::size_t>(C * KH * KW) *
                                  static_cast<std::size_t>(B * OH * OW));
    if (n == 0) return;
    const std::uint32_t dims[10] = {
        static_cast<std::uint32_t>(B),  static_cast<std::uint32_t>(C),
        static_cast<std::uint32_t>(H),  static_cast<std::uint32_t>(W),
        static_cast<std::uint32_t>(KH), static_cast<std::uint32_t>(KW),
        static_cast<std::uint32_t>(OH), static_cast<std::uint32_t>(OW),
        static_cast<std::uint32_t>(stride), static_cast<std::uint32_t>(pad)};
    detail::Context& c = detail::ctx();
    detail::Buffer* dimbuf = detail::dims_buffer(dims, 10);
    detail::Buffer* bind[3] = {dcol.buffer(), dx.buffer(), dimbuf};
    c.dispatch_1d(kernels::col2im2d_name<T>, bind, 3, n);
    c.release_buffer(dimbuf);
}

/**
 * Device conv forward epilogue, fused THREE-in-one — bias add + activation + layout transpose:
 * `a[(b·F+f)·OO + o] = act(yc[f·B·OO + b·OO + o] + bias[f])`, taking the GEMM's filter-major
 * [F, B·OO] output to the per-sample [B, F·OO] activation block in ONE dispatch. One thread per
 * output element. f64 note: tanh/sigmoid evaluate their transcendental in f32 on the device
 * (SPIR-V has no f64 exp/tanh); identity/relu stay full-width exact.
 *
 * @param a        The [B, F·OO]-element device activation block, overwritten.
 * @param yc       The [F, B·OO]-element device GEMM output.
 * @param bias     The F-element device bias row.
 * @param B        The batch size.
 * @param F        The filter (output-channel) count.
 * @param OO       The output plane element count OH·OW.
 * @param act_code The activation selector: 0 = identity, 1 = relu, 2 = tanh, 3 = sigmoid
 *                 (throws on any other code, and on an operand size mismatch).
 * @complexity O(B·F·OO) device work in one blocking dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc none — all three data buffers are the caller's.
 * @test gpu:conv
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && GpuElement<T>
void conv_bias_act(Array<T>& a, const Array<T>& yc, const Array<T>& bias, long long B,
                   long long F, long long OO, long long act_code) {
    detail::require_act_code("conv_bias_act", act_code);
    const std::size_t n = static_cast<std::size_t>(B * F * OO);
    detail::require_conv_size("conv_bias_act", "a", a.size(), n);
    detail::require_conv_size("conv_bias_act", "yc", yc.size(), n);
    detail::require_conv_size("conv_bias_act", "bias", bias.size(), static_cast<std::size_t>(F));
    if (n == 0) return;
    const std::uint32_t dims[4] = {static_cast<std::uint32_t>(B), static_cast<std::uint32_t>(F),
                                   static_cast<std::uint32_t>(OO),
                                   static_cast<std::uint32_t>(act_code)};
    detail::Context& c = detail::ctx();
    detail::Buffer* dimbuf = detail::dims_buffer(dims, 4);
    detail::Buffer* bind[4] = {yc.buffer(), bias.buffer(), a.buffer(), dimbuf};
    c.dispatch_1d(kernels::conv_bias_act_name<T>, bind, 4, n);
    c.release_buffer(dimbuf);
}

/**
 * Device conv backward epilogue (the adjoint of @ref conv_bias_act's act + layout, fused):
 * `dyc[f·B·OO + b·OO + o] = d[(b·F+f)·OO + o] · act'(a[(b·F+f)·OO + o])` — the chain-rule step
 * and the transpose back to filter-major in ONE dispatch. The derivative is evaluated FROM THE
 * OUTPUT a = f(z) (identity' = 1, relu' = [a > 0], tanh' = 1 − a², sigmoid' = a(1 − a)) — pure
 * arithmetic, exact in both element widths, no retained pre-activations.
 *
 * @param dyc      The [F, B·OO]-element device delta block, overwritten.
 * @param d        The [B, F·OO]-element device upstream gradient dL/dA.
 * @param a        The [B, F·OO]-element device activations (the forward output).
 * @param B        The batch size.
 * @param F        The filter (output-channel) count.
 * @param OO       The output plane element count OH·OW.
 * @param act_code The activation selector: 0 = identity, 1 = relu, 2 = tanh, 3 = sigmoid
 *                 (throws on any other code, and on an operand size mismatch).
 * @complexity O(B·F·OO) device work in one blocking dispatch.
 * @alloc none — content-cached dims scratch only.
 * @gpualloc none — all three data buffers are the caller's.
 * @test gpu:conv
 */
template <ndarray::Field T, template <typename> class Array>
    requires cheatah::linalg::DeviceArray<Array<T>> && GpuElement<T>
void conv_act_grad(Array<T>& dyc, const Array<T>& d, const Array<T>& a, long long B, long long F,
                   long long OO, long long act_code) {
    detail::require_act_code("conv_act_grad", act_code);
    const std::size_t n = static_cast<std::size_t>(B * F * OO);
    detail::require_conv_size("conv_act_grad", "dyc", dyc.size(), n);
    detail::require_conv_size("conv_act_grad", "d", d.size(), n);
    detail::require_conv_size("conv_act_grad", "a", a.size(), n);
    if (n == 0) return;
    const std::uint32_t dims[4] = {static_cast<std::uint32_t>(B), static_cast<std::uint32_t>(F),
                                   static_cast<std::uint32_t>(OO),
                                   static_cast<std::uint32_t>(act_code)};
    detail::Context& c = detail::ctx();
    detail::Buffer* dimbuf = detail::dims_buffer(dims, 4);
    detail::Buffer* bind[4] = {d.buffer(), a.buffer(), dyc.buffer(), dimbuf};
    c.dispatch_1d(kernels::conv_act_grad_name<T>, bind, 4, n);
    c.release_buffer(dimbuf);
}

}  // namespace cheatah::gpu::linalg
