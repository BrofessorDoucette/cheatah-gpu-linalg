// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// mlp_forward — the flagship "same code, one template" demo.
//
// A 3-layer MLP inference pass (matmul → bias → ReLU, twice, then a linear head) written ONCE as
// a location-blind template. `forward<gl::host_array>` runs cheatah's single-threaded SIMD CPU
// path; `forward<gl::device_array>` runs the GPU kernels — the ONLY difference is the template
// argument. ReLU is `(x + abs(x)) / 2`, all elementwise on-device. The device run uploads
// weights + inputs ONCE and downloads ONLY the final logits: the transfer ledger printed at the
// end proves it (and the example FAILS if anything mid-loop downloaded).
//
// Build + run (from the repo root):
//   cmake -B build -G Ninja && cmake --build build -j
//   ./build/examples/example_mlp_forward_vk
#include "examples/example_common.hpp"

namespace gl = cheatah::gpu::linalg;

// The network: x[batch, 256] -> h1[?,128] -> h2[?,64] -> logits[?,10]. Weights row-major.
template <template <class> class Array>
Array<double> forward(const Array<double>& x, const Array<double>& w1, const Array<double>& b1,
                      const Array<double>& w2, const Array<double>& b2, const Array<double>& w3) {
    using namespace cheatah::linalg;   // matmul for BOTH containers; operators/abs come by ADL
    auto layer = [](const Array<double>& in, const Array<double>& w, const Array<double>& bias) {
        Array<double> z = matmul(in, w) + bias;       // bias pre-broadcast to [batch, out]
        return (z + abs(z)) / 2.0;                    // ReLU, elementwise
    };
    return matmul(layer(layer(x, w1, b1), w2, b2), w3);
}

int main() {
    example::Verdict v;
    const std::size_t batch = 64, in = 256, h1 = 128, h2 = 64, out = 10;

    const auto xs = example::operands(batch * in, 1);
    const auto w1s = example::operands(in * h1, 2);
    const auto b1s = example::operands(batch * h1, 3);   // pre-broadcast bias
    const auto w2s = example::operands(h1 * h2, 4);
    const auto b2s = example::operands(batch * h2, 5);
    const auto w3s = example::operands(h2 * out, 6);

    // HOST: build basic_ndarrays…
    using H = gl::host_array<double>;
    const H hx = cheatah::ndarray::reshape(cheatah::ndarray::array(xs), {(long long)batch, (long long)in});
    const H hw1 = cheatah::ndarray::reshape(cheatah::ndarray::array(w1s), {(long long)in, (long long)h1});
    const H hb1 = cheatah::ndarray::reshape(cheatah::ndarray::array(b1s), {(long long)batch, (long long)h1});
    const H hw2 = cheatah::ndarray::reshape(cheatah::ndarray::array(w2s), {(long long)h1, (long long)h2});
    const H hb2 = cheatah::ndarray::reshape(cheatah::ndarray::array(b2s), {(long long)batch, (long long)h2});
    const H hw3 = cheatah::ndarray::reshape(cheatah::ndarray::array(w3s), {(long long)h2, (long long)out});
    const auto [hy, host_ms] = example::timed([&] { return forward<gl::host_array>(hx, hw1, hb1, hw2, hb2, hw3); });

    // DEVICE: the SAME function; operands go up once (to_device), the ledger watches the loop.
    using D = gl::device_array<double>;
    const D dx = gl::to_device(hx), dw1 = gl::to_device(hw1), db1 = gl::to_device(hb1);
    const D dw2 = gl::to_device(hw2), db2 = gl::to_device(hb2), dw3 = gl::to_device(hw3);
    gl::reset_stats();
    const auto [dy, dev_ms] = example::timed([&] { return forward<gl::device_array>(dx, dw1, db1, dw2, db2, dw3); });
    const auto mid = gl::stats();
    v.require("zero downloads during the device forward pass", mid.bytes_downloaded == 0);
    example::print_ledger("device forward pass");

    // Agreement: download the logits ONCE, compare against the host result.
    std::vector<double> got(batch * out);
    dy.to_host(got.data());
    for (std::size_t i = 0; i < batch * out; ++i)
        v.near("logits", got[i], hy.buffer()->data()[i], 1e-9);

    std::printf("  host %.2f ms   device %.2f ms   (same templated function)\n", host_ms, dev_ms);
    return v.finish("mlp_forward — one template argument flips CPU<->GPU");
}
