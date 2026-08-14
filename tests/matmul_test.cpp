// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// matmul_test.cpp — the seam, end to end. Builds GPU-resident arrays, calls cheatah's own
// allocating front `cheatah::linalg::matmul(a, b)`, and checks that it (a) resolved to THIS
// library's device kernel by ADL — not the host path — and (b) computed the right product, in
// double (the emulated/f64 path) AND float, including non-tile-aligned shapes that exercise the
// 16x16 tiled kernel's ragged edges. Also pins the deduction firewall: a host⊗device or f32⊗f64
// operand mix must not deduce ANY overload — a compile-time property asserted right here.
#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

#include "tests/harness.hpp"

#include <vector>

namespace {

namespace gl = cheatah::gpu::linalg;

// The deduction firewall, as compile-time facts: mixing locations (host ndarray with
// device_array) or elements (f32 with f64) has no viable matmul. Phrased as a concept so the
// invalid calls are substitution failures (a requires-expression outside a template would be a
// hard error), then asserted false.
template <class A, class B>
concept Multipliable = requires(const A& a, const B& b) { cheatah::linalg::matmul(a, b); };
using HostD = cheatah::ndarray::basic_ndarray<double>;

TEST(Matmul, DeductionFirewall) {
    static_assert(Multipliable<gl::device_array<double>, gl::device_array<double>>);
    static_assert(!Multipliable<HostD, gl::device_array<double>>);
    static_assert(!Multipliable<gl::device_array<double>, HostD>);
    static_assert(!Multipliable<gl::device_array<float>, gl::device_array<double>>);
}

// One matmul round-trip: fill MxK and KxN host operands deterministically, multiply on the
// device through cheatah's front, and compare every element against an in-test reference loop.
template <class T>
void run_case(std::size_t M, std::size_t K, std::size_t N, double tol) {
    const std::vector<T> A = harness::sequence<T>(M * K, 1);
    const std::vector<T> B = harness::sequence<T>(K * N, 2);
    gl::device_array<T> da = gl::device_array<T>::from_host({M, K}, A.data());
    gl::device_array<T> db = gl::device_array<T>::from_host({K, N}, B.data());

    gl::device_array<T> dc = cheatah::linalg::matmul(da, db);

    std::vector<T> C(M * N);
    dc.to_host(C.data());
    ASSERT_TRUE(dc.ndim() == 2 && dc.shape()[0] == M && dc.shape()[1] == N) << "result shape";
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            double want = 0.0;
            for (std::size_t k = 0; k < K; ++k)
                want += static_cast<double>(A[i * K + k]) * static_cast<double>(B[k * N + j]);
            EXPECT_NEAR_REL(C[i * N + j], want, tol) << "C[" << i * N + j << "]";
        }
}

// The original seam milestone: 2x3 @ 3x2.
TEST(Matmul, SeamMilestoneF64) { run_case<double>(2, 3, 2, 1e-9); }
// Ragged tiles on every axis.
TEST(Matmul, RaggedTilesF64) { run_case<double>(33, 17, 9, 1e-9); }
// The same shape through the f32 kernel.
TEST(Matmul, RaggedTilesF32) { run_case<float>(33, 17, 9, 1e-4); }
// 64-aligned -> still the edge kernel.
TEST(Matmul, Aligned64EdgeKernelF32) { run_case<float>(64, 64, 64, 1e-3); }
// Exactly-tiled -> the FAST kernel (min shape).
TEST(Matmul, FastKernelMinShapeF32) { run_case<float>(128, 16, 128, 1e-3); }
// Fast kernel, K a slab (16) but not block multiple.
TEST(Matmul, FastKernelSlabKF32) { run_case<float>(256, 272, 128, 1e-3); }
// f64 stays on the edge kernel by policy.
TEST(Matmul, EdgeKernelPolicyF64) { run_case<double>(128, 32, 128, 1e-9); }
// M%64 (not %128) -> the FAST-64 kernel (min shape).
TEST(Matmul, Fast64KernelMinShapeF32) { run_case<float>(64, 32, 128, 1e-3); }
// Fast-64 again, multi-block both axes.
TEST(Matmul, Fast64MultiBlockF32) { run_case<float>(192, 48, 256, 1e-3); }

}  // namespace
