// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// vector_pipeline — the elementwise/reduction story: numpy-style operator chains, ufuncs and
// reductions, location-blind. `normalize<host_array>` and `normalize<device_array>` are the same
// code; every intermediate of the device run stays resident (asserted), and the two paths agree
// to double precision.
//
//   cmake -B build -G Ninja && cmake --build build -j
//   ./build/examples/example_vector_pipeline_vk
#include "examples/example_common.hpp"

#include <cmath>

namespace gl = cheatah::gpu::linalg;

// Standardize, squash, and score a signal: z = (x - mean) / std; s = sqrt(|z|); return <s, z>.
template <template <class> class Array>
double score(const Array<double>& x, std::size_t n) {
    using namespace cheatah::linalg;
    const double mu = mean(x);                       // reduction (sync: returns a host scalar)
    const double var = mean((x - mu) * (x - mu));    // operator chain + reduction
    Array<double> z = (x - mu) / std::sqrt(var + 1e-12);
    Array<double> s = sqrt(abs(z));                  // ufuncs, ADL-resolved per container
    return dot(s, z);
}

int main() {
    example::Verdict v;
    const std::size_t n = 100000;   // crosses the two-stage reduction threshold on device

    const auto xs = example::operands(n, 11);
    using H = gl::host_array<double>;
    const H hx = cheatah::ndarray::array(xs);
    const auto [hscore, host_ms] = example::timed([&] { return score<gl::host_array>(hx, n); });

    using D = gl::device_array<double>;
    const D dx = gl::to_device(hx);
    gl::reset_stats();
    const auto [dscore, dev_ms] = example::timed([&] { return score<gl::device_array>(dx, n); });
    const auto mid = gl::stats();
    // The reductions each read back ONE finalized element (8 bytes) — nothing else may move.
    v.require("only single-element reduction readbacks",
              mid.bytes_downloaded <= 4 * sizeof(double));
    example::print_ledger("device pipeline");

    v.near("score (host vs device)", dscore, hscore, 1e-9);
    std::printf("  host %.2f ms   device %.2f ms   n=%zu\n", host_ms, dev_ms, n);
    return v.finish("vector_pipeline — operators, ufuncs and reductions, location-blind");
}
