// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// complex_test.cpp — the c64/c128 device slice vs cheatah's own host complex results: matmul
// (complex product), conj_transpose (the Hermitian adjoint — conjugation must actually happen),
// dot vs vdot (bilinear vs conjugate-linear — they must DIFFER on complex data), trace, outer,
// kron, and a batched complex matmul. c128 runs where shaderFloat64 exists; c64 everywhere.
#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

#include "tests/harness.hpp"

#include <complex>
#include <vector>

namespace {

namespace gl = cheatah::gpu::linalg;
using C64 = std::complex<float>;
using C128 = std::complex<double>;

template <class T>
std::vector<T> cseq(std::size_t n, std::size_t salt) {
    std::vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = T(static_cast<typename T::value_type>(harness::fill(i, salt)),
                 static_cast<typename T::value_type>(harness::fill(i, salt + 1)));
    return v;
}

#define EXPECT_NEAR_C(got, want, tol)                                \
    do {                                                             \
        EXPECT_NEAR_REL(double((got).real()), (want).real(), (tol)); \
        EXPECT_NEAR_REL(double((got).imag()), (want).imag(), (tol)); \
    } while (0)

// Complex matmul vs reference.
template <class T>
void matmul_case(double tol) {
    const std::size_t M = 5, K = 4, N = 6;
    const std::vector<T> A = cseq<T>(M * K, 50);
    const std::vector<T> B = cseq<T>(K * N, 52);
    gl::device_array<T> da = gl::device_array<T>::from_host({M, K}, A.data());
    gl::device_array<T> db = gl::device_array<T>::from_host({K, N}, B.data());
    std::vector<T> got(M * N);
    cheatah::linalg::matmul(da, db).to_host(got.data());
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            C128 want{};
            for (std::size_t k = 0; k < K; ++k)
                want += C128(A[i * K + k]) * C128(B[k * N + j]);
            EXPECT_NEAR_C(got[i * N + j], want, tol);
        }
}

// conj_transpose really conjugates (Hermitian adjoint).
template <class T>
void adjoint_case(double tol) {
    const std::size_t M = 5, K = 4;
    const std::vector<T> A = cseq<T>(M * K, 50);
    gl::device_array<T> da = gl::device_array<T>::from_host({M, K}, A.data());
    std::vector<T> got(K * M);
    cheatah::linalg::conj_transpose(da).to_host(got.data());
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < K; ++j)
            EXPECT_NEAR_C(got[j * M + i], std::conj(C128(A[i * K + j])), tol);
}

// dot (bilinear) vs vdot (conjugate-linear): both right, and different from each other.
template <class T>
void dot_vdot_case(double tol) {
    const std::size_t n = 300;  // strided partials
    const std::vector<T> x = cseq<T>(n, 54);
    const std::vector<T> y = cseq<T>(n, 56);
    gl::device_array<T> dx = gl::device_array<T>::from_host({n}, x.data());
    gl::device_array<T> dy = gl::device_array<T>::from_host({n}, y.data());
    const std::size_t P = n < gl::kernels::kMaxReduce ? n : gl::kernels::kMaxReduce;
    C128 want_dot{}, want_vdot{};
    for (std::size_t g = 0; g < P; ++g) {
        C128 acc_d{}, acc_v{};
        for (std::size_t k = g; k < n; k += P) {
            acc_d += C128(x[k]) * C128(y[k]);
            acc_v += std::conj(C128(x[k])) * C128(y[k]);
        }
        want_dot += acc_d;
        want_vdot += acc_v;
    }
    EXPECT_NEAR_C(cheatah::linalg::dot(dx, dy), want_dot, tol);
    EXPECT_NEAR_C(cheatah::linalg::vdot(dx, dy), want_vdot, tol);
    EXPECT_NEAR_C(cheatah::linalg::inner(dx, dy), want_dot, tol);
    EXPECT_GT(std::abs(want_dot - want_vdot), 1e-3)
        << "dot != vdot on complex data (the conjugation is observable)";
}

// Trace of a square complex matrix.
template <class T>
void trace_case(double tol) {
    const std::size_t n = 7;
    const std::vector<T> S = cseq<T>(n * n, 58);
    gl::device_array<T> ds = gl::device_array<T>::from_host({n, n}, S.data());
    C128 want{};
    for (std::size_t i = 0; i < n; ++i) want += C128(S[i * n + i]);
    EXPECT_NEAR_C(cheatah::linalg::trace(ds), want, tol);
}

// outer + kron, complex products.
template <class T>
void outer_kron_case(double tol) {
    const std::vector<T> u = cseq<T>(3, 60), v = cseq<T>(4, 62);
    gl::device_array<T> du = gl::device_array<T>::from_host({3}, u.data());
    gl::device_array<T> dv = gl::device_array<T>::from_host({4}, v.data());
    std::vector<T> got(12);
    cheatah::linalg::outer(du, dv).to_host(got.data());
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_NEAR_C(got[i * 4 + j], C128(u[i]) * C128(v[j]), tol);

    const std::vector<T> A2 = cseq<T>(4, 64), B2 = cseq<T>(4, 66);
    gl::device_array<T> dA2 = gl::device_array<T>::from_host({2, 2}, A2.data());
    gl::device_array<T> dB2 = gl::device_array<T>::from_host({2, 2}, B2.data());
    std::vector<T> kg(16);
    cheatah::linalg::kron(dA2, dB2).to_host(kg.data());
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_NEAR_C(kg[i * 4 + j],
                          C128(A2[(i / 2) * 2 + (j / 2)]) * C128(B2[(i % 2) * 2 + (j % 2)]),
                          tol);
}

// Batched complex matmul, B = 2 small slices.
template <class T>
void batched_case(double tol) {
    const std::size_t Bn = 2, m = 3, k = 2, n2 = 3;
    const std::vector<T> Ab = cseq<T>(Bn * m * k, 70);
    const std::vector<T> Bb = cseq<T>(Bn * k * n2, 72);
    gl::device_array<T> dab = gl::device_array<T>::from_host({Bn, m, k}, Ab.data());
    gl::device_array<T> dbb = gl::device_array<T>::from_host({Bn, k, n2}, Bb.data());
    std::vector<T> got(Bn * m * n2);
    cheatah::linalg::matmul(dab, dbb).to_host(got.data());
    for (std::size_t z = 0; z < Bn; ++z)
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n2; ++j) {
                C128 want{};
                for (std::size_t kk = 0; kk < k; ++kk)
                    want += C128(Ab[z * m * k + i * k + kk]) *
                            C128(Bb[z * k * n2 + kk * n2 + j]);
                EXPECT_NEAR_C(got[z * m * n2 + i * n2 + j], want, tol);
            }
}

TEST(Complex, MatmulC64) { matmul_case<C64>(1e-4); }
TEST(Complex, MatmulC128) { matmul_case<C128>(1e-12); }
TEST(Complex, AdjointC64) { adjoint_case<C64>(1e-4); }
TEST(Complex, AdjointC128) { adjoint_case<C128>(1e-12); }
TEST(Complex, DotVdotInnerC64) { dot_vdot_case<C64>(1e-4); }
TEST(Complex, DotVdotInnerC128) { dot_vdot_case<C128>(1e-12); }
TEST(Complex, TraceC64) { trace_case<C64>(1e-4); }
TEST(Complex, TraceC128) { trace_case<C128>(1e-12); }
TEST(Complex, OuterKronC64) { outer_kron_case<C64>(1e-4); }
TEST(Complex, OuterKronC128) { outer_kron_case<C128>(1e-12); }
TEST(Complex, BatchedMatmulC64) { batched_case<C64>(1e-4); }
TEST(Complex, BatchedMatmulC128) { batched_case<C128>(1e-12); }

}  // namespace
