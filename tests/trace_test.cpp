// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// trace_test.cpp — the device trace through cheatah's scalar-returning front: diagonal partial
// sums on rectangular matrices (min(r, c) diagonal entries), including a diagonal longer than
// kMaxReduce so the strided partial path runs, in double and float.
#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

#include "tests/harness.hpp"

#include <algorithm>
#include <vector>

namespace {

namespace gl = cheatah::gpu::linalg;
namespace kn = cheatah::gpu::linalg::kernels;

// The same P-strided association order the kernel commits to.
template <class T>
double partial_trace_reference(const std::vector<T>& A, std::size_t r, std::size_t c) {
    const std::size_t m = std::min(r, c);
    const std::size_t P = std::min<std::size_t>(m, kn::kMaxReduce);
    double total = 0.0;
    for (std::size_t g = 0; g < P; ++g) {
        double acc = 0.0;
        for (std::size_t k = g; k < m; k += P) acc += static_cast<double>(A[k * c + k]);
        total += acc;
    }
    return total;
}

template <class T>
void run_case(std::size_t r, std::size_t c, double tol) {
    const std::vector<T> A = harness::sequence<T>(r * c, 10);
    gl::device_array<T> da = gl::device_array<T>::from_host({r, c}, A.data());
    EXPECT_NEAR_REL(cheatah::linalg::trace(da), partial_trace_reference(A, r, c), tol);
}

TEST(Trace, SquareF64) { run_case<double>(4, 4, 1e-12); }
// Rectangular: min(r, c) diagonal entries.
TEST(Trace, RectangularF64) { run_case<double>(5, 3, 1e-12); }
// Diagonal longer than the partial cap.
TEST(Trace, StridedPartialsF64) { run_case<double>(300, 300, 1e-12); }
TEST(Trace, StridedPartialsF32) { run_case<float>(300, 300, 1e-4); }

}  // namespace
