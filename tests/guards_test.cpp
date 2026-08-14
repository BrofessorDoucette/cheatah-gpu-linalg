// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// guards_test.cpp — every boundary the security audit hardened, actually exercised: the
// shape-overflow and 2^32-1 element-cap throws, elementwise shape-mismatch, the conv
// operand-size and activation-code validators, and the `available()` probe on both its
// answers (the Vulkan lane poisons device selection via the environment BEFORE first touch,
// so the false path and its recorded reason are real; the Metal-emulated lane always comes
// up, proving the true path).
#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

#include "tests/harness.hpp"

#include <climits>
#include <cstdlib>
#include <stdexcept>

namespace {

namespace gl = cheatah::gpu::linalg;

// Declared FIRST in this TU: gtest runs tests in declaration order, and the process-wide
// context is lazy — this must be the first ctx() touch in the binary for the poisoning to
// bite on the Vulkan lane.
TEST(Guards, AvailableProbeAnswersHonestly) {
#if defined(CHEATAH_GPU_LINALG_VULKAN)
    ::setenv("CHEATAH_GPU_LINALG_VK_DEVICE", "no-such-device-xyzzy", 1);
    EXPECT_FALSE(gl::available());
    EXPECT_FALSE(gl::unavailable_reason().empty());
    ::unsetenv("CHEATAH_GPU_LINALG_VK_DEVICE");
#else
    EXPECT_TRUE(gl::available());
    EXPECT_TRUE(gl::unavailable_reason().empty());
#endif
}

#if !defined(CHEATAH_GPU_LINALG_VULKAN)
// The device-touching guards run on the Metal-emulated lane (always available); the Vulkan
// lane's context is deliberately poisoned above, so device work there would only re-test
// the probe.

TEST(Guards, ShapeElementCountOverflowThrows) {
    const long long big = 1LL << 31;
    EXPECT_THROW((void)gl::device_array<float>::uninitialized({big, big, big}),
                 std::runtime_error);
    float dummy = 0.0f;
    EXPECT_THROW((void)gl::device_array<float>::from_host({big, big, big}, &dummy),
                 std::runtime_error);
}

TEST(Guards, ElementCapAt32BitIndexing) {
    // The product fits in size_t but exceeds 2^32-1 elements — the cap that keeps kernel
    // index arithmetic non-wrapping must reject it before any allocation.
    EXPECT_THROW((void)gl::device_array<float>::uninitialized({1LL << 20, 1LL << 13}),
                 std::runtime_error);
}

TEST(Guards, WithShapeIsZeroCopyAndReshapeGuards) {
    gl::device_array<float> a = gl::device_array<float>::uninitialized({4});
    // with_shape is the DOCUMENTED unvalidated zero-copy sibling (reshape validates):
    // same device buffer, new dims, no allocation.
    gl::device_array<float> v = a.with_shape({2, 2});
    EXPECT_EQ(v.buffer(), a.buffer());
    EXPECT_EQ(v.ndim(), 2u);
    EXPECT_THROW((void)gl::reshape(a, {1LL << 20, 1LL << 13}), std::runtime_error);
}

TEST(Guards, ElementwiseShapeMismatchThrows) {
    gl::device_array<float> a = gl::device_array<float>::uninitialized({2, 2});
    gl::device_array<float> b = gl::device_array<float>::uninitialized({4, 2});
    EXPECT_THROW((void)gl::add(a, b), std::runtime_error);
}

TEST(Guards, DotGuardsAndTwoStageBranch) {
    gl::device_array<float> a3 = gl::device_array<float>::uninitialized({3});
    gl::device_array<float> b4 = gl::device_array<float>::uninitialized({4});
    float sink = 0.0f;
    EXPECT_THROW(gl::dot(sink, a3, b4), std::runtime_error);
    gl::device_array<double> a3d = gl::device_array<double>::uninitialized({3});
    gl::device_array<double> b4d = gl::device_array<double>::uninitialized({4});
    double sinkd = 0.0;
    EXPECT_THROW(gl::dot(sinkd, a3d, b4d), std::runtime_error);
    gl::device_array<std::complex<float>> a3c =
        gl::device_array<std::complex<float>>::uninitialized({3});
    gl::device_array<std::complex<float>> b4c =
        gl::device_array<std::complex<float>>::uninitialized({4});
    std::complex<float> sinkc{};
    EXPECT_THROW(gl::dot(sinkc, a3c, b4c), std::runtime_error);
    EXPECT_THROW(gl::vdot(sinkc, a3c, b4c), std::runtime_error);

    // n >= kTwoStageMin drives the two-stage partial kernels (f32 / f64 / complex, dot + vdot).
    const std::size_t n = 70000;
    std::vector<float> xf(n), yf(n);
    std::vector<double> xd(n), yd(n);
    std::vector<std::complex<float>> xc(n), yc(n);
    double want_f = 0.0, want_d = 0.0;
    std::complex<double> want_c{0.0, 0.0}, want_vc{0.0, 0.0};
    for (std::size_t i = 0; i < n; ++i) {
        const float v = static_cast<float>(static_cast<long long>(i % 13) - 6) * 0.25f;
        const float w = static_cast<float>(static_cast<long long>(i % 7) - 3) * 0.5f;
        xf[i] = v; yf[i] = w;
        xd[i] = v; yd[i] = w;
        xc[i] = {v, w}; yc[i] = {w, v};
        want_f += double(v) * double(w);
        want_d += double(v) * double(w);
        want_c += std::complex<double>(v, w) * std::complex<double>(w, v);
        want_vc += std::conj(std::complex<double>(v, w)) * std::complex<double>(w, v);
    }
    auto df = gl::device_array<float>::from_host({n}, xf.data());
    auto ef = gl::device_array<float>::from_host({n}, yf.data());
    float got_f = 0.0f;
    gl::dot(got_f, df, ef);
    EXPECT_NEAR(got_f, want_f, std::abs(want_f) * 1e-4 + 1e-3);
    auto dd = gl::device_array<double>::from_host({n}, xd.data());
    auto ed = gl::device_array<double>::from_host({n}, yd.data());
    double got_d = 0.0;
    gl::dot(got_d, dd, ed);
    EXPECT_NEAR(got_d, want_d, std::abs(want_d) * 1e-10 + 1e-9);
    auto dc = gl::device_array<std::complex<float>>::from_host({n}, xc.data());
    auto ec = gl::device_array<std::complex<float>>::from_host({n}, yc.data());
    std::complex<float> gc{};
    gl::dot(gc, dc, ec);
    EXPECT_NEAR(gc.real(), want_c.real(), std::abs(want_c.real()) * 1e-3 + 1e-1);
    EXPECT_NEAR(gc.imag(), want_c.imag(), std::abs(want_c.imag()) * 1e-3 + 1e-1);
    std::complex<float> gv{};
    gl::vdot(gv, dc, ec);
    EXPECT_NEAR(gv.real(), want_vc.real(), std::abs(want_vc.real()) * 1e-3 + 1e-1);
    EXPECT_NEAR(gv.imag(), want_vc.imag(), std::abs(want_vc.imag()) * 1e-3 + 1e-1);
}

TEST(Guards, ResidentDotStaysOnDevice) {
    // The device-resident dot front: the result lands in a 1-element device array with
    // no host round-trip, in both reduction regimes and instantiations, and the out-size
    // guard holds.
    const std::size_t big = 70000, small = 33;
    std::vector<float> xs(big), ys(big);
    double want_big = 0.0, want_small = 0.0;
    for (std::size_t i = 0; i < big; ++i) {
        xs[i] = static_cast<float>(static_cast<long long>(i % 11) - 5) * 0.25f;
        ys[i] = static_cast<float>(static_cast<long long>(i % 5) - 2) * 0.5f;
        want_big += double(xs[i]) * double(ys[i]);
        if (i < small) want_small += double(xs[i]) * double(ys[i]);
    }
    auto a = gl::device_array<float>::from_host({big}, xs.data());
    auto b = gl::device_array<float>::from_host({big}, ys.data());
    auto as = gl::device_array<float>::from_host({small}, xs.data());
    auto bs = gl::device_array<float>::from_host({small}, ys.data());
    auto out = gl::device_array<float>::uninitialized({1});
    gl::dot(out, a, b);                      // two-stage regime
    EXPECT_NEAR(gl::get(out, {0}), want_big, std::abs(want_big) * 1e-4 + 1e-3);
    gl::dot(out, as, bs);                    // single-stage regime
    EXPECT_NEAR(gl::get(out, {0}), want_small, std::abs(want_small) * 1e-4 + 1e-3);

    std::vector<std::complex<float>> xc(big), yc(big);
    for (std::size_t i = 0; i < big; ++i) {
        xc[i] = {xs[i], ys[i]};
        yc[i] = {ys[i], xs[i]};
    }
    auto ac = gl::device_array<std::complex<float>>::from_host({big}, xc.data());
    auto bc = gl::device_array<std::complex<float>>::from_host({big}, yc.data());
    auto outc = gl::device_array<std::complex<float>>::uninitialized({1});
    gl::dot(outc, ac, bc);                   // the complex instantiation of the resident front
    std::complex<double> want_cc{0.0, 0.0};
    for (std::size_t i = 0; i < big; ++i)
        want_cc += std::complex<double>(xc[i]) * std::complex<double>(yc[i]);
    const std::complex<float> gv = gl::get(outc, {0});
    EXPECT_NEAR(gv.real(), want_cc.real(), std::abs(want_cc.real()) * 1e-3 + 1e-1);
    EXPECT_NEAR(gv.imag(), want_cc.imag(), std::abs(want_cc.imag()) * 1e-3 + 1e-1);

    auto out2 = gl::device_array<float>::uninitialized({2});
    EXPECT_THROW(gl::dot(out2, a, b), std::runtime_error);   // resident out-size guard
    EXPECT_THROW(gl::dot(out, a, as), std::runtime_error);   // resident length mismatch
}

TEST(Guards, EmptyOperandsShortCircuit) {
    // Zero-length operands take the documented early returns — no dispatch, identity results.
    auto e = gl::device_array<float>::uninitialized({0});
    float s = 1.0f;
    gl::dot(s, e, e);
    EXPECT_EQ(s, 0.0f);
    auto m0 = gl::device_array<float>::uninitialized({0, 3});
    float t = 1.0f;
    gl::trace(t, m0);
    EXPECT_EQ(t, 0.0f);
}

TEST(Guards, MatmulFamilyValidatorsThrow) {
    auto v   = gl::device_array<float>::uninitialized({4});
    auto m22 = gl::device_array<float>::uninitialized({2, 2});
    auto m32 = gl::device_array<float>::uninitialized({3, 2});
    auto out = gl::device_array<float>::uninitialized({2, 2});
    EXPECT_THROW(gl::matmul(out, v, m22), std::runtime_error);         // operand not 2-D
    EXPECT_THROW(gl::matmul(out, m22, m32), std::runtime_error);       // inner dims differ
    EXPECT_THROW(gl::conj_transpose(out, v), std::runtime_error);      // operand not 2-D
    EXPECT_THROW(gl::kron(out, v, m22), std::runtime_error);           // operand not 2-D
}

TEST(Guards, FactoriesAccessGuards) {
    gl::device_array<float> a = gl::device_array<float>::uninitialized({4});
    EXPECT_THROW((void)gl::reshape(a, {1LL << 40, 1LL << 40}), std::runtime_error);
    EXPECT_THROW((void)gl::get(a, {0, 0}), std::runtime_error);  // rank mismatch
    EXPECT_THROW((void)gl::get(a, {9}), std::runtime_error);     // index out of range
}

TEST(Guards, BenchKernelStandInsStayExact) {
    // copy_f32 and triad_f32 exist for the benchmark suite; dispatching them here keeps the
    // emulated stand-ins in the tested set (and the bench honest about what it times).
    namespace det = gl::detail;
    const std::size_t n = 300;
    std::vector<float> xs(n), ys(n);
    for (std::size_t i = 0; i < n; ++i) { xs[i] = float(i) * 0.5f; ys[i] = float(i % 9) - 4.0f; }
    auto x = gl::device_array<float>::from_host({n}, xs.data());
    auto y = gl::device_array<float>::from_host({n}, ys.data());
    auto out = gl::device_array<float>::uninitialized({n});
    const std::uint32_t n32 = static_cast<std::uint32_t>(n);
    det::Context& c = det::ctx();
    det::Buffer* dimbuf = det::dims_buffer(&n32, 1);

    det::Buffer* copy_bind[3] = {x.buffer(), out.buffer(), dimbuf};
    c.dispatch_1d("copy_f32", copy_bind, 3, n);
    std::vector<float> got(n);
    out.to_host(got.data());
    for (std::size_t i = 0; i < n; ++i) ASSERT_EQ(got[i], xs[i]) << i;

    auto s = gl::scalar(2.0f);
    det::Buffer* triad_bind[5] = {out.buffer(), x.buffer(), y.buffer(), s.buffer(), dimbuf};
    c.dispatch_1d("triad_f32", triad_bind, 5, n);
    out.to_host(got.data());
    for (std::size_t i = 0; i < n; ++i) ASSERT_EQ(got[i], xs[i] + 2.0f * ys[i]) << i;
    c.release_buffer(dimbuf);
}

TEST(Guards, ConvValidatorsThrow) {
    gl::device_array<float> a = gl::device_array<float>::uninitialized({2 * 3 * 4});
    gl::device_array<float> yc = gl::device_array<float>::uninitialized({2 * 3 * 4});
    gl::device_array<float> bias = gl::device_array<float>::uninitialized({3});
    gl::device_array<float> bad_bias = gl::device_array<float>::uninitialized({5});
    // act code out of range
    EXPECT_THROW(gl::conv_bias_act(a, yc, bias, 2, 3, 4, 7), std::runtime_error);
    // wrong operand element count
    EXPECT_THROW(gl::conv_bias_act(a, yc, bad_bias, 2, 3, 4, 0), std::runtime_error);
}
#endif  // !CHEATAH_GPU_LINALG_VULKAN

}  // namespace
