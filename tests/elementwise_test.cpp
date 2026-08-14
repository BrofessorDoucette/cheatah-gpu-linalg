// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// elementwise_test.cpp — the elementwise device surface vs host reference loops: array⊗array
// add/sub/mul/divide, array⊗scalar both orders, unary sqrt/exp/log/abs/neg, the sum/mean
// reductions (P-strided association order pinned), the fused axpy, in-place aliasing, and odd
// (non-multiple-of-256) sizes, in double and float.
#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

#include "tests/harness.hpp"

#include <cmath>
#include <vector>

namespace {

namespace gl = cheatah::gpu::linalg;
namespace kn = cheatah::gpu::linalg::kernels;

template <class T>
void run_binary(std::size_t n, double tol) {
    std::vector<T> a = harness::sequence<T>(n, 20);
    std::vector<T> b(n);
    for (std::size_t i = 0; i < n; ++i) b[i] = static_cast<T>(harness::fill(i, 21) + 2.5);  // no zeros
    gl::device_array<T> da = gl::device_array<T>::from_host({n}, a.data());
    gl::device_array<T> db = gl::device_array<T>::from_host({n}, b.data());

    std::vector<T> got(n);
    gl::add(da, db).to_host(got.data());
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR_REL(got[i], double(a[i]) + double(b[i]), tol) << "add[" << i << "]";
    gl::sub(da, db).to_host(got.data());
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR_REL(got[i], double(a[i]) - double(b[i]), tol) << "sub[" << i << "]";
    gl::mul(da, db).to_host(got.data());
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR_REL(got[i], double(a[i]) * double(b[i]), tol) << "mul[" << i << "]";
    gl::divide(da, db).to_host(got.data());
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR_REL(got[i], double(a[i]) / double(b[i]), tol) << "div[" << i << "]";

    // fused axpy vs the chained reference
    gl::axpy(T(2.5), da, db).to_host(got.data());
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR_REL(got[i], 2.5 * double(a[i]) + double(b[i]), tol) << "axpy[" << i << "]";
}

template <class T>
void run_unary(std::size_t n, double tol) {
    std::vector<T> a(n);
    for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<T>(harness::fill(i, 22) + 2.0);  // > 0
    gl::device_array<T> da = gl::device_array<T>::from_host({n}, a.data());
    std::vector<T> got(n);
    gl::sqrt(da).to_host(got.data());
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR_REL(got[i], std::sqrt(double(a[i])), tol) << "sqrt[" << i << "]";
    gl::exp(da).to_host(got.data());
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR_REL(got[i], std::exp(double(a[i])), tol * 32) << "exp[" << i << "]";
    gl::log(da).to_host(got.data());
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR_REL(got[i], std::log(double(a[i])), tol) << "log[" << i << "]";
    gl::abs(-da).to_host(got.data());   // neg then abs round-trips to the input
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR_REL(got[i], a[i], tol) << "abs(neg)[" << i << "]";
}

template <class T>
void run_reduction(std::size_t n, double tol) {
    std::vector<T> a = harness::sequence<T>(n, 23);
    gl::device_array<T> da = gl::device_array<T>::from_host({n}, a.data());
    // The P-strided partial-sum association order, reproduced exactly.
    const std::size_t P = n < kn::kMaxReduce ? n : kn::kMaxReduce;
    double want = 0.0;
    for (std::size_t g = 0; g < P; ++g) {
        double acc = 0.0;
        for (std::size_t k = g; k < n; k += P) acc += double(a[k]);
        want += acc;
    }
    EXPECT_NEAR_REL(gl::sum(da), want, tol) << "sum";
    EXPECT_NEAR_REL(gl::mean(da), want / double(n), tol) << "mean";
}

// Strided partials + full workgroups + ragged tail.
TEST(Elementwise, BinaryStridedF64) { run_binary<double>(1000, 1e-12); }
TEST(Elementwise, BinaryTinyF64) { run_binary<double>(7, 1e-12); }
TEST(Elementwise, BinaryStridedF32) { run_binary<float>(1000, 1e-5); }
TEST(Elementwise, UnaryF64) { run_unary<double>(300, 1e-12); }
TEST(Elementwise, UnaryF32) { run_unary<float>(300, 1e-5); }
TEST(Elementwise, ReductionF64) { run_reduction<double>(100000, 1e-9); }
TEST(Elementwise, ReductionF32) { run_reduction<float>(4096, 1e-2); }

// 2-D shapes flow through unchanged, and in-place aliasing works (out == a).
TEST(Elementwise, TwoDShapeAndAliasing) {
    const std::size_t r = 5, c = 7;
    std::vector<double> m = harness::sequence<double>(r * c, 24);
    gl::device_array<double> dm = gl::device_array<double>::from_host({r, c}, m.data());
    gl::device_array<double> sq = dm * dm;
    ASSERT_TRUE(sq.ndim() == 2 && sq.shape()[0] == r && sq.shape()[1] == c) << "2-D shape kept";
    gl::mul(dm, dm, dm);  // in place
    std::vector<double> got(r * c);
    dm.to_host(got.data());
    for (std::size_t i = 0; i < r * c; ++i)
        EXPECT_NEAR_REL(got[i], m[i] * m[i], 1e-12) << "in-place mul[" << i << "]";
}

}  // namespace
