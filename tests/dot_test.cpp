// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// dot_test.cpp — the device dot family (dot, vdot, inner — identical bilinear sums for real
// elements) through cheatah's scalar-returning fronts, against a reference accumulation. Lengths
// beyond kMaxReduce exercise the strided partial-sum path (each of the 256 partials covers
// several elements); the reference below reproduces the SAME association order, so double cases
// compare essentially exactly.
#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

#include "tests/harness.hpp"

#include <algorithm>
#include <vector>

namespace {

namespace gl = cheatah::gpu::linalg;
namespace kn = cheatah::gpu::linalg::kernels;

// The partial-sum reference: the same P-strided association order the kernel commits to.
template <class T>
double partial_sum_reference(const std::vector<T>& a, const std::vector<T>& b) {
    const std::size_t n = a.size();
    const std::size_t P = std::min<std::size_t>(n, kn::kMaxReduce);
    double total = 0.0;
    for (std::size_t g = 0; g < P; ++g) {
        double acc = 0.0;
        for (std::size_t k = g; k < n; k += P)
            acc += static_cast<double>(a[k]) * static_cast<double>(b[k]);
        total += acc;
    }
    return total;
}

template <class T>
void run_case(std::size_t n, double tol) {
    const std::vector<T> a = harness::sequence<T>(n, 8);
    const std::vector<T> b = harness::sequence<T>(n, 9);
    gl::device_array<T> da = gl::device_array<T>::from_host({n}, a.data());
    gl::device_array<T> db = gl::device_array<T>::from_host({n}, b.data());

    const double want = partial_sum_reference(a, b);
    EXPECT_NEAR_REL(cheatah::linalg::dot(da, db), want, tol);
    EXPECT_NEAR_REL(cheatah::linalg::vdot(da, db), want, tol);
    EXPECT_NEAR_REL(cheatah::linalg::inner(da, db), want, tol);
}

// Below the partial cap: one element per partial.
TEST(Dot, BelowPartialCapF64) { run_case<double>(7, 1e-12); }
// Strided partials (256 threads x ~4 elements).
TEST(Dot, StridedPartialsF64) { run_case<double>(1000, 1e-12); }
TEST(Dot, StridedPartialsF32) { run_case<float>(1000, 1e-4); }

}  // namespace
