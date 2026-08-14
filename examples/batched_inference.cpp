// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// batched_inference — B independent weight matrices applied to B inputs in ONE dispatch.
//
// `matmul` on 3-D operands is the batched product [B,M,K] @ [B,K,N] — the same call, host or
// device (one template argument), but on the device the whole batch is a single z-layered
// kernel launch instead of B separate ones: the ledger shows exactly ONE dispatch for the
// entire ensemble.
//
//   cmake -B build -G Ninja && cmake --build build -j
//   ./build/examples/example_batched_inference_vk
#include "examples/example_common.hpp"

namespace gl = cheatah::gpu::linalg;

template <template <class> class Array>
Array<double> ensemble(const Array<double>& inputs, const Array<double>& weights) {
    using namespace cheatah::linalg;
    return matmul(inputs, weights);   // [B,M,K] @ [B,K,N] — one call, one dispatch on device
}

int main() {
    example::Verdict v;
    const std::size_t B = 32, M = 16, K = 64, N = 24;

    const auto xs = example::operands(B * M * K, 9);
    const auto ws = example::operands(B * K * N, 10);

    using H = gl::host_array<double>;
    const H hx = cheatah::ndarray::reshape(cheatah::ndarray::array(xs),
                                           {(long long)B, (long long)M, (long long)K});
    const H hw = cheatah::ndarray::reshape(cheatah::ndarray::array(ws),
                                           {(long long)B, (long long)K, (long long)N});
    const auto [hy, host_ms] = example::timed([&] { return ensemble<gl::host_array>(hx, hw); });

    using D = gl::device_array<double>;
    const D dx = gl::to_device(hx), dw = gl::to_device(hw);
    gl::reset_stats();
    const auto [dy, dev_ms] = example::timed([&] { return ensemble<gl::device_array>(dx, dw); });
    const auto mid = gl::stats();
    v.require("the whole batch was ONE dispatch", mid.dispatches == 1);
    v.require("zero downloads during inference", mid.bytes_downloaded == 0);
    example::print_ledger("batched ensemble (32 models)");

    std::vector<double> got(B * M * N);
    dy.to_host(got.data());
    for (std::size_t i = 0; i < B * M * N; ++i)
        v.near("batched product", got[i], hy.buffer()->data()[i], 1e-9);

    std::printf("  host %.2f ms   device %.2f ms   B=%zu of [%zux%zu]@[%zux%zu]\n", host_ms,
                dev_ms, B, M, K, K, N);
    return v.finish("batched_inference — B matmuls, one call, one device dispatch");
}
