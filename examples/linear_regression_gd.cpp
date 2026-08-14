// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// linear_regression_gd — the miniature training-framework demo: a TRAINING LOOP that never
// leaves the GPU.
//
// Gradient descent on w for ||Xw - y||^2: each epoch is matmul (predictions), operator subtraction
// (residual), conj_transpose+matmul (gradient), and a fused axpy weight update — one templated
// function, run on the CPU and the GPU by flipping the template argument. On the device path the
// data goes up ONCE; the loop is asserted to download ZERO bytes (the loss is tracked with the
// DEVICE-RESIDENT sum overload and read back only after training). The ledger prints the proof.
//
//   cmake -B build -G Ninja && cmake --build build -j
//   ./build/examples/example_linear_regression_gd_vk
#include "examples/example_common.hpp"

namespace gl = cheatah::gpu::linalg;

template <template <class> class Array>
Array<double> train(const Array<double>& X, const Array<double>& y, Array<double> w, int epochs,
                    double lr, std::size_t n_samples) {
    using namespace cheatah::linalg;        // matmul/conj_transpose for both containers
    using namespace cheatah::gpu::linalg;   // axpy — fused kernel on device, plain loop on host
    for (int e = 0; e < epochs; ++e) {
        Array<double> r = matmul(X, w) - y;                 // residual  [n, 1]
        Array<double> g = matmul(conj_transpose(X), r);     // gradient  [d, 1]
        axpy(w, -2.0 * lr / static_cast<double>(n_samples), g, w);   // w -= lr' * g, fused
    }
    return w;
}

int main() {
    example::Verdict v;
    const std::size_t n = 512, d = 32;
    const int epochs = 300;
    const double lr = 0.01;

    // Synthetic problem: y = X * w_true. The deterministic fill alone makes near-collinear
    // columns (period-16 pattern), so boost a shifted diagonal — well-conditioned X'X, honest GD.
    auto Xs = example::operands(n * d, 7);
    for (std::size_t i = 0; i < n; ++i) Xs[i * d + (i % d)] += 8.0;
    std::vector<double> wtrue(d);
    for (std::size_t i = 0; i < d; ++i) wtrue[i] = example::fill(i, 8);
    std::vector<double> ys(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < d; ++j) ys[i] += Xs[i * d + j] * wtrue[j];

    using H = gl::host_array<double>;
    const H hX = cheatah::ndarray::reshape(cheatah::ndarray::array(Xs), {(long long)n, (long long)d});
    const H hy = cheatah::ndarray::reshape(cheatah::ndarray::array(ys), {(long long)n, 1});
    // Fresh zero weights PER PATH (both containers share buffers on copy — view semantics —
    // so the in-place training update must not cross paths through a shared w0).
    const H hw0 = cheatah::ndarray::basic_ndarray<double>({d, 1}, 0.0);

    const auto [hw, host_ms] =
        example::timed([&] { return train<gl::host_array>(hX, hy, hw0, epochs, lr, n); });

    using D = gl::device_array<double>;
    const D dX = gl::to_device(hX), dy = gl::to_device(hy);
    D dw0 = gl::to_device(cheatah::ndarray::basic_ndarray<double>({d, 1}, 0.0));
    gl::reset_stats();
    const auto [dw, dev_ms] =
        example::timed([&] { return train<gl::device_array>(dX, dy, dw0, epochs, lr, n); });
    const auto mid = gl::stats();
    v.require("ZERO downloads across the whole training loop", mid.bytes_downloaded == 0);
    example::print_ledger("300-epoch device training loop");

    // The DEVICE-RESIDENT loss: sum((Xw - y)^2) computed and kept on the GPU, read back once.
    {
        using namespace cheatah::linalg;
        D r = matmul(dX, dw) - dy;
        D r2 = r * r;
        D loss = gl::device_array<double>::uninitialized({1});
        gl::sum(loss, r2);                       // resident reduction — no download
        const double final_loss = gl::get(loss, {0});   // ONE 8-byte readback
        v.near("final training loss is small", final_loss, 0.0, 1e-2);
    }

    // Agreement with the host-trained weights, and with the ground truth.
    std::vector<double> got(d);
    dw.to_host(got.data());
    for (std::size_t i = 0; i < d; ++i) {
        v.near("w (host vs device)", got[i], hw.buffer()->data()[i], 1e-9);
        v.near("w vs w_true", got[i], wtrue[i], 1e-2);
    }

    std::printf("  host %.2f ms   device %.2f ms   %d epochs, n=%zu d=%zu\n", host_ms, dev_ms,
                epochs, n, d);
    return v.finish("linear_regression_gd — a training loop that never leaves the GPU");
}
