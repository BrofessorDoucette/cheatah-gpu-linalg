// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// conv_test.cpp — the four conv-support device kernels (im2col2d / col2im2d / conv_bias_act /
// conv_act_grad) against in-test transliterations of the neural-net consumer's host reference
// loops (golden math, not golden files): seeded non-uniform operands go up, the kernel runs on
// the device, and the download is compared ELEMENTWISE — exactly (im2col/col2im are pure data
// movement with a fixed per-cell association order; the arithmetic act codes are single-op
// exact) or at 1e-6 for the transcendental act codes (device tanh/exp vs libm, and the f64
// device path evaluates its transcendental in f32 — SPIR-V has no f64 exp/tanh). Coverage:
// pad>0 borders, stride 2, ragged sizes, act codes 0-3, minimal shapes, and a perturbation
// guard (a deliberately-wrong pad must change the result — proving the harness can fail).
#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

#include "tests/harness.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

namespace gl = cheatah::gpu::linalg;

// The conv geometry of one case (mirrors the consumer's Conv2d fields; OH/OW derived the same
// way: (dim + 2·pad − k) / stride + 1).
struct Geom {
    long long B, C, H, W, KH, KW, stride, pad;
    long long oh() const { return (H + 2 * pad - KH) / stride + 1; }
    long long ow() const { return (W + 2 * pad - KW) / stride + 1; }
};

// ---- the host reference loops: transliterated from the consumer's net.hpp (the oracle) --------

template <class T>
void host_im2col2d(std::vector<T>& col, const std::vector<T>& x, const Geom& g) {
    const long long OH = g.oh(), OW = g.ow(), OO = OH * OW, BOO = g.B * OO;
    for (long long c = 0; c < g.C; ++c)
        for (long long kh = 0; kh < g.KH; ++kh)
            for (long long kw = 0; kw < g.KW; ++kw) {
                T* row = col.data() + ((c * g.KH + kh) * g.KW + kw) * BOO;
                for (long long b = 0; b < g.B; ++b) {
                    const T* plane = x.data() + (b * g.C + c) * g.H * g.W;
                    T* dst = row + b * OO;
                    for (long long oh = 0; oh < OH; ++oh) {
                        const long long ih = oh * g.stride - g.pad + kh;
                        if (ih < 0 || ih >= g.H) {
                            for (long long ow = 0; ow < OW; ++ow) dst[oh * OW + ow] = T(0);
                            continue;
                        }
                        const T* src = plane + ih * g.W;
                        for (long long ow = 0; ow < OW; ++ow) {
                            const long long iw = ow * g.stride - g.pad + kw;
                            dst[oh * OW + ow] = (iw < 0 || iw >= g.W) ? T(0) : src[iw];
                        }
                    }
                }
            }
}

template <class T>
void host_col2im2d(std::vector<T>& dx, const std::vector<T>& dcol, const Geom& g) {
    const long long OH = g.oh(), OW = g.ow(), OO = OH * OW, BOO = g.B * OO;
    for (auto& v : dx) v = T(0);
    for (long long c = 0; c < g.C; ++c)
        for (long long kh = 0; kh < g.KH; ++kh)
            for (long long kw = 0; kw < g.KW; ++kw) {
                const T* row = dcol.data() + ((c * g.KH + kh) * g.KW + kw) * BOO;
                for (long long b = 0; b < g.B; ++b) {
                    T* plane = dx.data() + (b * g.C + c) * g.H * g.W;
                    const T* src = row + b * OO;
                    for (long long oh = 0; oh < OH; ++oh) {
                        const long long ih = oh * g.stride - g.pad + kh;
                        if (ih < 0 || ih >= g.H) continue;
                        T* dstrow = plane + ih * g.W;
                        for (long long ow = 0; ow < OW; ++ow) {
                            const long long iw = ow * g.stride - g.pad + kw;
                            if (iw >= 0 && iw < g.W) dstrow[iw] += src[oh * OW + ow];
                        }
                    }
                }
            }
}

template <class T>
T host_act_f(T z, long long act) {
    if (act == 1) return z > T(0) ? z : T(0);
    if (act == 2) return std::tanh(z);
    if (act == 3) return T(1) / (T(1) + std::exp(-z));
    return z;
}

template <class T>
T host_act_df(T a, long long act) {
    if (act == 1) return a > T(0) ? T(1) : T(0);
    if (act == 2) return T(1) - a * a;
    if (act == 3) return a * (T(1) - a);
    return T(1);
}

template <class T>
void host_conv_bias_act(std::vector<T>& a, const std::vector<T>& yc, const std::vector<T>& bias,
                        long long B, long long F, long long OO, long long act) {
    const long long BOO = B * OO;
    for (long long b = 0; b < B; ++b)
        for (long long f = 0; f < F; ++f) {
            const T bv = bias[f];
            const T* src = yc.data() + f * BOO + b * OO;
            T* dst = a.data() + (b * F + f) * OO;
            for (long long o = 0; o < OO; ++o) dst[o] = host_act_f(src[o] + bv, act);
        }
}

template <class T>
void host_conv_act_grad(std::vector<T>& dyc, const std::vector<T>& d, const std::vector<T>& a,
                        long long B, long long F, long long OO, long long act) {
    const long long BOO = B * OO;
    for (long long b = 0; b < B; ++b)
        for (long long f = 0; f < F; ++f) {
            const T* dsrc = d.data() + (b * F + f) * OO;
            const T* asrc = a.data() + (b * F + f) * OO;
            T* dst = dyc.data() + f * BOO + b * OO;
            for (long long o = 0; o < OO; ++o) dst[o] = dsrc[o] * host_act_df(asrc[o], act);
        }
}

// ---- im2col2d / col2im2d: pure data movement + fixed-order sums — EXACT equality --------------

template <class T>
void run_im2col(const Geom& g) {
    const std::size_t xn = static_cast<std::size_t>(g.B * g.C * g.H * g.W);
    const std::size_t cn =
        static_cast<std::size_t>(g.C * g.KH * g.KW) * static_cast<std::size_t>(g.B * g.oh() * g.ow());
    const std::vector<T> x = harness::sequence<T>(xn, 5);
    std::vector<T> want(cn);
    host_im2col2d(want, x, g);

    gl::device_array<T> dx = gl::device_array<T>::from_host({xn}, x.data());
    // Poison the destination so the FULL-overwrite contract (padding written as zeros) is real.
    const std::vector<T> poison(cn, T(7));
    gl::device_array<T> dcol = gl::device_array<T>::from_host({cn}, poison.data());
    gl::im2col2d(dcol, dx, g.B, g.C, g.H, g.W, g.KH, g.KW, g.oh(), g.ow(), g.stride, g.pad);

    std::vector<T> got(cn);
    dcol.to_host(got.data());
    for (std::size_t i = 0; i < cn; ++i)
        EXPECT_EQ(got[i], want[i]) << "col[" << i << "]";
}

template <class T>
void run_col2im(const Geom& g) {
    const std::size_t xn = static_cast<std::size_t>(g.B * g.C * g.H * g.W);
    const std::size_t cn =
        static_cast<std::size_t>(g.C * g.KH * g.KW) * static_cast<std::size_t>(g.B * g.oh() * g.ow());
    const std::vector<T> dcol = harness::sequence<T>(cn, 9);
    std::vector<T> want(xn, T(3));   // pre-fill: the reference zeroes first, like the kernel
    host_col2im2d(want, dcol, g);

    gl::device_array<T> dd = gl::device_array<T>::from_host({cn}, dcol.data());
    const std::vector<T> poison(xn, T(7));
    gl::device_array<T> ddx = gl::device_array<T>::from_host({xn}, poison.data());
    gl::col2im2d(ddx, dd, g.B, g.C, g.H, g.W, g.KH, g.KW, g.oh(), g.ow(), g.stride, g.pad);

    std::vector<T> got(xn);
    ddx.to_host(got.data());
    for (std::size_t i = 0; i < xn; ++i)
        EXPECT_EQ(got[i], want[i]) << "dx[" << i << "]";
}

// Padded borders, stride 1 (every border cell exercises the zero-fill path).
const Geom kPadded{2, 3, 5, 7, 3, 3, 1, 1};
// Stride 2 + pad (the stride-divisibility skip in the adjoint), ragged planes.
const Geom kStrided{2, 3, 5, 7, 3, 3, 2, 1};
// No padding, ragged non-square kernel.
const Geom kRagged{3, 2, 6, 5, 2, 3, 1, 0};
// Minimal 1×1 everything.
const Geom kMinimal{1, 1, 1, 1, 1, 1, 1, 0};

TEST(Conv, Im2col2dPaddedF64) { run_im2col<double>(kPadded); }
TEST(Conv, Im2col2dPaddedF32) { run_im2col<float>(kPadded); }
TEST(Conv, Im2col2dStride2F64) { run_im2col<double>(kStrided); }
TEST(Conv, Im2col2dStride2F32) { run_im2col<float>(kStrided); }
TEST(Conv, Im2col2dRaggedF64) { run_im2col<double>(kRagged); }
TEST(Conv, Im2col2dMinimalF64) { run_im2col<double>(kMinimal); }

TEST(Conv, Col2im2dPaddedF64) { run_col2im<double>(kPadded); }
TEST(Conv, Col2im2dPaddedF32) { run_col2im<float>(kPadded); }
TEST(Conv, Col2im2dStride2F64) { run_col2im<double>(kStrided); }
TEST(Conv, Col2im2dStride2F32) { run_col2im<float>(kStrided); }
TEST(Conv, Col2im2dRaggedF64) { run_col2im<double>(kRagged); }
TEST(Conv, Col2im2dMinimalF64) { run_col2im<double>(kMinimal); }

// ---- the fused epilogues, all four act codes --------------------------------------------------

template <class T>
void run_bias_act(long long B, long long F, long long OO, long long act, double tol) {
    const std::size_t n = static_cast<std::size_t>(B * F * OO);
    const std::vector<T> yc = harness::sequence<T>(n, 11);
    const std::vector<T> bias = harness::sequence<T>(static_cast<std::size_t>(F), 13);
    std::vector<T> want(n);
    host_conv_bias_act(want, yc, bias, B, F, OO, act);

    gl::device_array<T> dyc = gl::device_array<T>::from_host({n}, yc.data());
    gl::device_array<T> db = gl::device_array<T>::from_host({static_cast<std::size_t>(F)}, bias.data());
    gl::device_array<T> da = gl::device_array<T>::uninitialized({n});
    gl::conv_bias_act(da, dyc, db, B, F, OO, act);

    std::vector<T> got(n);
    da.to_host(got.data());
    for (std::size_t i = 0; i < n; ++i) {
        if (tol == 0.0)
            EXPECT_EQ(got[i], want[i]) << "a[" << i << "] act=" << act;
        else
            EXPECT_NEAR_REL(got[i], want[i], tol) << "a[" << i << "] act=" << act;
    }
}

template <class T>
void run_act_grad(long long B, long long F, long long OO, long long act, double tol) {
    const std::size_t n = static_cast<std::size_t>(B * F * OO);
    const std::vector<T> d = harness::sequence<T>(n, 17);
    const std::vector<T> a = harness::sequence<T>(n, 19);
    std::vector<T> want(n);
    host_conv_act_grad(want, d, a, B, F, OO, act);

    gl::device_array<T> dd = gl::device_array<T>::from_host({n}, d.data());
    gl::device_array<T> da = gl::device_array<T>::from_host({n}, a.data());
    gl::device_array<T> ddyc = gl::device_array<T>::uninitialized({n});
    gl::conv_act_grad(ddyc, dd, da, B, F, OO, act);

    std::vector<T> got(n);
    ddyc.to_host(got.data());
    for (std::size_t i = 0; i < n; ++i) {
        if (tol == 0.0)
            EXPECT_EQ(got[i], want[i]) << "dyc[" << i << "] act=" << act;
        else
            EXPECT_NEAR_REL(got[i], want[i], tol) << "dyc[" << i << "] act=" << act;
    }
}

// identity/relu are single-op arithmetic — exact in both widths; tanh/sigmoid tolerate 1e-6
// (device transcendentals vs libm; the f64 forward evaluates its transcendental in f32).
TEST(Conv, BiasActIdentityF64) { run_bias_act<double>(3, 4, 15, 0, 0.0); }
TEST(Conv, BiasActReluF64) { run_bias_act<double>(3, 4, 15, 1, 0.0); }
TEST(Conv, BiasActTanhF64) { run_bias_act<double>(3, 4, 15, 2, 1e-6); }
TEST(Conv, BiasActSigmoidF64) { run_bias_act<double>(3, 4, 15, 3, 1e-6); }
TEST(Conv, BiasActIdentityF32) { run_bias_act<float>(3, 4, 15, 0, 0.0); }
TEST(Conv, BiasActReluF32) { run_bias_act<float>(3, 4, 15, 1, 0.0); }
TEST(Conv, BiasActTanhF32) { run_bias_act<float>(3, 4, 15, 2, 1e-6); }
TEST(Conv, BiasActSigmoidF32) { run_bias_act<float>(3, 4, 15, 3, 1e-6); }
TEST(Conv, BiasActMinimal) { run_bias_act<double>(1, 1, 1, 1, 0.0); }

// The backward derivative is pure arithmetic in every code; tanh/sigmoid still get 1e-6
// headroom for FMA contraction of 1−a² / a(1−a) on the device.
TEST(Conv, ActGradIdentityF64) { run_act_grad<double>(3, 4, 15, 0, 0.0); }
TEST(Conv, ActGradReluF64) { run_act_grad<double>(3, 4, 15, 1, 0.0); }
TEST(Conv, ActGradTanhF64) { run_act_grad<double>(3, 4, 15, 2, 1e-6); }
TEST(Conv, ActGradSigmoidF64) { run_act_grad<double>(3, 4, 15, 3, 1e-6); }
TEST(Conv, ActGradIdentityF32) { run_act_grad<float>(3, 4, 15, 0, 0.0); }
TEST(Conv, ActGradReluF32) { run_act_grad<float>(3, 4, 15, 1, 0.0); }
TEST(Conv, ActGradTanhF32) { run_act_grad<float>(3, 4, 15, 2, 1e-6); }
TEST(Conv, ActGradSigmoidF32) { run_act_grad<float>(3, 4, 15, 3, 1e-6); }
TEST(Conv, ActGradMinimal) { run_act_grad<double>(1, 1, 1, 3, 1e-6); }

// ---- perturbation guard: the harness must be ABLE to fail -------------------------------------

// A deliberately-wrong pad must change the device result vs the correctly-padded oracle — if a
// mismatched geometry still "passed", the elementwise comparison above would prove nothing.
TEST(Conv, PerturbationGuardWrongPadDiffers) {
    const Geom g = kPadded;
    const std::size_t xn = static_cast<std::size_t>(g.B * g.C * g.H * g.W);
    const std::size_t cn =
        static_cast<std::size_t>(g.C * g.KH * g.KW) * static_cast<std::size_t>(g.B * g.oh() * g.ow());
    const std::vector<double> x = harness::sequence<double>(xn, 5);
    std::vector<double> want(cn);
    host_im2col2d(want, x, g);

    Geom wrong = g;
    wrong.pad = 0;   // same OH/OW passed below, only the pad lies
    gl::device_array<double> dx = gl::device_array<double>::from_host({xn}, x.data());
    gl::device_array<double> dcol = gl::device_array<double>::uninitialized({cn});
    gl::im2col2d(dcol, dx, g.B, g.C, g.H, g.W, g.KH, g.KW, g.oh(), g.ow(), g.stride, wrong.pad);

    std::vector<double> got(cn);
    dcol.to_host(got.data());
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < cn; ++i)
        if (got[i] != want[i]) ++mismatches;
    EXPECT_GT(mismatches, 0u) << "a wrong pad produced the padded oracle's exact bits — "
                                 "the parity checks above could not fail";
}

}  // namespace
