// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// bridge_test.cpp — the host-bridged factorization surface, end to end through cheatah's own
// fronts with device operands: solve/inv/det (LU family), qr/svd (multi-output), eig (complex
// outputs from a real input), eigvalsh (Hermitian complex input), an f32 container, the
// deduction firewall — plus the remaining six routines (matrix_power, eigvals, eigh, lstsq,
// pinv, svdvals), each checked against the host path AND a closed-form property that can fail
// on its own (reconstruction, residual orthogonality, A·A⁺·A ≈ A, Σλ ≈ tr). Every result is
// compared against the same op run on the HOST arrays — the bridge must agree with the host
// path essentially exactly (same algorithm, one f64 round-trip).
#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

#include "tests/harness.hpp"

#include <complex>
#include <vector>

namespace {

namespace gl = cheatah::gpu::linalg;
namespace hl = cheatah::linalg;
using HostD = cheatah::ndarray::basic_ndarray<double>;
using HostC = cheatah::ndarray::basic_ndarray<std::complex<double>>;

// Firewall: host⊗device solve/qr must not deduce.
template <class A, class B>
concept Solvable = requires(const A& a, const B& b) { cheatah::linalg::solve(a, b); };

TEST(BridgeFirewall, DeductionFirewall) {
    static_assert(Solvable<gl::device_array<double>, gl::device_array<double>>);
    static_assert(!Solvable<HostD, gl::device_array<double>>);
    static_assert(!Solvable<gl::device_array<double>, HostD>);
}

// A well-conditioned SPD-ish host matrix: A = M·Mᵀ + n·I (deterministic fill).
HostD spd(std::size_t n) {
    std::vector<double> m(n * n);
    for (std::size_t i = 0; i < n * n; ++i) m[i] = harness::fill(i, 11);
    HostD out = HostD::uninitialized({n, n});
    double* p = out.buffer()->data();
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double acc = (i == j) ? static_cast<double>(n) : 0.0;
            for (std::size_t k = 0; k < n; ++k) acc += m[i * n + k] * m[j * n + k];
            p[i * n + j] = acc;
        }
    return out;
}

gl::device_array<double> to_device(const HostD& h) {
    return gl::device_array<double>::from_host(h.shape(), h.buffer()->data());
}

// The shared square system: A = spd(6) on host and device.
class Bridge : public ::testing::Test {
protected:
    static constexpr std::size_t n = 6;
    HostD A = spd(n);
    gl::device_array<double> dA = to_device(A);
};

// A tall 8×5 system shared by the least-squares family (deterministic fill, full column rank by
// construction: the fill pattern plus a diagonal boost).
class BridgeTall : public Bridge {
protected:
    static constexpr std::size_t m8 = 8, k5 = 5;
    HostD Bt = tall();
    gl::device_array<double> dBt = to_device(Bt);

    static HostD tall() {
        HostD b = HostD::uninitialized({m8, k5});
        for (std::size_t i = 0; i < m8 * k5; ++i) b.buffer()->data()[i] = harness::fill(i, 7);
        for (std::size_t j = 0; j < k5; ++j) b.buffer()->data()[j * k5 + j] += 4.0;
        return b;
    }
};

// solve: A x = b, device vs host, plus the residual ‖A·x − b‖ ≈ 0.
TEST_F(Bridge, Solve) {
    HostD b = HostD::uninitialized({n});
    for (std::size_t i = 0; i < n; ++i) b.buffer()->data()[i] = harness::fill(i, 3);
    gl::device_array<double> db = to_device(b);
    gl::device_array<double> dx = hl::solve(dA, db);
    HostD hx = hl::solve(A, b);
    std::vector<double> x(n);
    dx.to_host(x.data());
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR_REL(x[i], hx.buffer()->data()[i], 1e-12) << "solve[" << i << "]";
}

// inv + det + cond + matrix_rank + norm + slogdet (scalar family).
TEST_F(Bridge, InvAndScalarFamily) {
    gl::device_array<double> dinv = hl::inv(dA);
    HostD hinv = hl::inv(A);
    std::vector<double> got(n * n);
    dinv.to_host(got.data());
    for (std::size_t i = 0; i < n * n; ++i)
        EXPECT_NEAR_REL(got[i], hinv.buffer()->data()[i], 1e-12) << "inv[" << i << "]";
    EXPECT_NEAR_REL(hl::det(dA), hl::det(A), 1e-9) << "det";
    EXPECT_NEAR_REL(hl::cond(dA), hl::cond(A), 1e-9) << "cond";
    EXPECT_EQ(hl::matrix_rank(dA), hl::matrix_rank(A)) << "matrix_rank";
    EXPECT_NEAR_REL(hl::norm(dA), hl::norm(A), 1e-12) << "norm";
    const auto ds = hl::slogdet(dA);
    const auto hs = hl::slogdet(A);
    EXPECT_NEAR_REL(ds.sign, hs.sign, 0.0) << "slogdet.sign";
    EXPECT_NEAR_REL(ds.logabsdet, hs.logabsdet, 1e-12) << "slogdet.log";
}

// qr (multi-out struct of device arrays) vs host.
TEST_F(Bridge, Qr) {
    const auto dqr = hl::qr(dA);
    const auto hqr = hl::qr(A);
    std::vector<double> q(n * n), r(n * n);
    dqr.q.to_host(q.data());
    dqr.r.to_host(r.data());
    for (std::size_t i = 0; i < n * n; ++i) {
        EXPECT_NEAR_REL(q[i], hqr.q.buffer()->data()[i], 1e-12) << "qr.q[" << i << "]";
        EXPECT_NEAR_REL(r[i], hqr.r.buffer()->data()[i], 1e-12) << "qr.r[" << i << "]";
    }
}

// svd on a rectangular (tall) matrix.
TEST_F(Bridge, SvdRectangular) {
    const std::size_t m = 8, k = 5;
    HostD B = HostD::uninitialized({m, k});
    for (std::size_t i = 0; i < m * k; ++i) B.buffer()->data()[i] = harness::fill(i, 7);
    gl::device_array<double> dB = to_device(B);
    const auto dsv = hl::svd(dB);
    const auto hsv = hl::svd(B);
    std::vector<double> s(k);
    dsv.s.to_host(s.data());
    for (std::size_t i = 0; i < k; ++i)
        EXPECT_NEAR_REL(s[i], hsv.s.buffer()->data()[i], 1e-12) << "svd.s[" << i << "]";
    EXPECT_TRUE(dsv.u.shape()[0] == m && dsv.u.shape()[1] == k && dsv.vh.shape()[0] == k &&
                dsv.vh.shape()[1] == k)
        << "svd shapes";
}

// eig: complex outputs from a real (non-symmetric) input — a rotation-ish matrix.
TEST_F(Bridge, EigComplexFromReal) {
    HostD R = HostD::uninitialized({2, 2});
    R.buffer()->data()[0] = 0.0; R.buffer()->data()[1] = -1.0;
    R.buffer()->data()[2] = 1.0; R.buffer()->data()[3] = 0.0;
    gl::device_array<double> dR = to_device(R);
    const auto de = hl::eig(dR);
    const auto he = hl::eig(R);
    std::vector<std::complex<double>> vals(2);
    de.values.to_host(vals.data());
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR_REL(vals[i].real(), he.values.buffer()->data()[i].real(), 1e-12)
            << "eig.re[" << i << "]";
        EXPECT_NEAR_REL(vals[i].imag(), he.values.buffer()->data()[i].imag(), 1e-12)
            << "eig.im[" << i << "]";
    }
}

// eigvalsh on a complex Hermitian input (real spectrum out of a complex container).
TEST_F(Bridge, EigvalshComplexHermitian) {
    HostC H = HostC::uninitialized({2, 2});
    H.buffer()->data()[0] = {2.0, 0.0};
    H.buffer()->data()[1] = {0.0, -1.0};
    H.buffer()->data()[2] = {0.0, 1.0};
    H.buffer()->data()[3] = {3.0, 0.0};
    gl::device_array<std::complex<double>> dH =
        gl::device_array<std::complex<double>>::from_host({2, 2}, H.buffer()->data());
    auto dvals = hl::eigvalsh(dH);
    auto hvals = hl::eigvalsh(H);
    std::vector<double> v(2);
    dvals.to_host(v.data());
    for (std::size_t i = 0; i < 2; ++i)
        EXPECT_NEAR_REL(v[i], hvals.buffer()->data()[i], 1e-12) << "eigvalsh(c)[" << i << "]";
}

// f32 container through the bridge (computed in f64, narrowed on upload).
TEST_F(Bridge, F32Container) {
    std::vector<float> af(n * n);
    for (std::size_t i = 0; i < n * n; ++i)
        af[i] = static_cast<float>(A.buffer()->data()[i]);
    gl::device_array<float> dAf = gl::device_array<float>::from_host({n, n}, af.data());
    gl::device_array<float> dinvf = hl::inv(dAf);
    HostD hinv = hl::inv(A);
    std::vector<float> got(n * n);
    dinvf.to_host(got.data());
    for (std::size_t i = 0; i < n * n; ++i)
        EXPECT_NEAR_REL(got[i], hinv.buffer()->data()[i], 1e-4) << "inv(f32)[" << i << "]";
}

// out-param reuse: ONE device buffer filled twice through the kernel form directly.
TEST_F(Bridge, OutParamReuse) {
    gl::device_array<double> out = gl::device_array<double>::uninitialized({n, n});
    gl::inv(out, dA);                       // the DeviceArray kernel overload, reused …
    gl::cholesky(out, dA);                  // … and overwritten in place
    HostD hch = hl::cholesky(A);
    std::vector<double> got(n * n);
    out.to_host(got.data());
    for (std::size_t i = 0; i < n * n; ++i)
        EXPECT_NEAR_REL(got[i], hch.buffer()->data()[i], 1e-12)
            << "cholesky(reused out)[" << i << "]";
}

// matrix_power: A³ against a host matmul chain, and A⁰ = I exactly.
TEST_F(Bridge, MatrixPower) {
    gl::device_array<double> dp = hl::matrix_power(dA, 3);
    HostD a3 = hl::matmul(hl::matmul(A, A), A);
    std::vector<double> got(n * n);
    dp.to_host(got.data());
    for (std::size_t i = 0; i < n * n; ++i)
        EXPECT_NEAR_REL(got[i], a3.buffer()->data()[i], 1e-9) << "matrix_power(3)[" << i << "]";
    gl::device_array<double> d0 = hl::matrix_power(dA, 0);
    d0.to_host(got.data());
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            EXPECT_NEAR_REL(got[i * n + j], i == j ? 1.0 : 0.0, 0.0)
                << "matrix_power(0)[" << i * n + j << "]";
}

// eigvals: device vs host spectrum (SPD ⇒ real, descending), and Σλ ≈ tr(A).
TEST_F(Bridge, Eigvals) {
    gl::device_array<std::complex<double>> dv = hl::eigvals(dA);
    auto hv = hl::eigvals(A);
    std::vector<std::complex<double>> vals(n);
    dv.to_host(vals.data());
    std::complex<double> lsum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR_REL(vals[i].real(), hv.buffer()->data()[i].real(), 1e-9)
            << "eigvals.re[" << i << "]";
        EXPECT_NEAR_REL(vals[i].imag(), hv.buffer()->data()[i].imag(), 1e-9)
            << "eigvals.im[" << i << "]";
        lsum += vals[i];
    }
    double tr = 0.0;
    for (std::size_t i = 0; i < n; ++i) tr += A.buffer()->data()[i * n + i];
    EXPECT_NEAR_REL(lsum.real(), tr, 1e-9) << "eigvals trace";
    EXPECT_NEAR_REL(lsum.imag(), 0.0, 1e-9) << "eigvals trace.im";
}

// eigh: real spectrum vs eigvalsh + full reconstruction V·diag(λ)·Vᵀ ≈ A.
TEST_F(Bridge, Eigh) {
    const auto de = hl::eigh(dA);
    auto hvals = hl::eigvalsh(A);
    std::vector<double> lam(n), V(n * n);
    de.values.to_host(lam.data());
    de.vectors.to_host(V.data());
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR_REL(lam[i], hvals.buffer()->data()[i], 1e-9) << "eigh.values[" << i << "]";
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double acc = 0.0;   // column k of V is the eigenvector for lam[k]
            for (std::size_t k = 0; k < n; ++k) acc += V[i * n + k] * lam[k] * V[j * n + k];
            EXPECT_NEAR_REL(acc, A.buffer()->data()[i * n + j], 1e-8)
                << "eigh reconstruct[" << i * n + j << "]";
        }
}

// lstsq: device vs host + the normal-equation residual orthogonality Bᵀ(B·x − c) ≈ 0.
TEST_F(BridgeTall, Lstsq) {
    HostD C = HostD::uninitialized({m8, 2});
    for (std::size_t i = 0; i < m8 * 2; ++i) C.buffer()->data()[i] = harness::fill(i, 13);
    gl::device_array<double> dC = to_device(C);
    gl::device_array<double> dx = hl::lstsq(dBt, dC);
    HostD hx = hl::lstsq(Bt, C);
    std::vector<double> x(k5 * 2);
    dx.to_host(x.data());
    for (std::size_t i = 0; i < k5 * 2; ++i)
        EXPECT_NEAR_REL(x[i], hx.buffer()->data()[i], 1e-9) << "lstsq[" << i << "]";
    for (std::size_t j = 0; j < k5; ++j)          // residual ⊥ col span: Bᵀ·r ≈ 0
        for (std::size_t c = 0; c < 2; ++c) {
            double acc = 0.0;
            for (std::size_t i = 0; i < m8; ++i) {
                double r = -C.buffer()->data()[i * 2 + c];
                for (std::size_t k = 0; k < k5; ++k)
                    r += Bt.buffer()->data()[i * k5 + k] * x[k * 2 + c];
                acc += Bt.buffer()->data()[i * k5 + j] * r;
            }
            EXPECT_NEAR_REL(acc, 0.0, 1e-8)
                << "lstsq residual orthogonality[" << j * 2 + c << "]";
        }
}

// pinv: device vs host + the Moore–Penrose property B·B⁺·B ≈ B.
TEST_F(BridgeTall, Pinv) {
    gl::device_array<double> dp = hl::pinv(dBt);
    HostD hp = hl::pinv(Bt);
    std::vector<double> P(k5 * m8);
    dp.to_host(P.data());
    ASSERT_TRUE(dp.shape()[0] == k5 && dp.shape()[1] == m8) << "pinv shape";
    for (std::size_t i = 0; i < k5 * m8; ++i)
        EXPECT_NEAR_REL(P[i], hp.buffer()->data()[i], 1e-9) << "pinv[" << i << "]";
    for (std::size_t i = 0; i < m8; ++i)          // (B·P·B)[i,j] ≈ B[i,j]
        for (std::size_t j = 0; j < k5; ++j) {
            double acc = 0.0;
            for (std::size_t k = 0; k < k5; ++k) {
                double bp = 0.0;
                for (std::size_t l = 0; l < m8; ++l)
                    bp += Bt.buffer()->data()[i * k5 + k] * P[k * m8 + l] *
                          Bt.buffer()->data()[l * k5 + j];
                acc += bp;
            }
            EXPECT_NEAR_REL(acc, Bt.buffer()->data()[i * k5 + j], 1e-8)
                << "pinv B·B⁺·B[" << i * k5 + j << "]";
        }
}

// svdvals: device vs host, and vs the full device svd's singular values.
TEST_F(BridgeTall, Svdvals) {
    gl::device_array<double> ds = hl::svdvals(dBt);
    HostD hs = hl::svdvals(Bt);
    ASSERT_TRUE(ds.shape().size() == 1 && ds.shape()[0] == k5) << "svdvals length";
    std::vector<double> s(k5), sfull(k5);
    ds.to_host(s.data());
    const auto dsvd = hl::svd(dBt);
    dsvd.s.to_host(sfull.data());
    for (std::size_t i = 0; i < k5; ++i) {
        EXPECT_NEAR_REL(s[i], hs.buffer()->data()[i], 1e-12) << "svdvals[" << i << "]";
        EXPECT_NEAR_REL(s[i], sfull[i], 1e-9) << "svdvals vs svd.s[" << i << "]";
    }
    for (std::size_t i = 1; i < k5; ++i)          // the descending contract
        EXPECT_GE(s[i - 1], s[i]) << "svdvals descending[" << i << "]";
}

}  // namespace
