// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// factories_test.cpp — the convenience surface end to end: host<->device transfers (and the
// contiguity requirement), fill_device on real AND complex elements, the zeros/ones/full
// family, array/scalar/arange/reshape/get, printing (to_string + operator<<), size_of /
// is_contiguous, the transfer-stats ledger, and the SMALL-array single-stage branch of the
// sum/mean reduction (everything below kTwoStageMin).
#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

#include "tests/harness.hpp"

#include <complex>
#include <sstream>
#include <vector>

namespace {

namespace gl = cheatah::gpu::linalg;
namespace nd = cheatah::ndarray;

TEST(Factories, ToDeviceToHostRoundTrip) {
    nd::basic_ndarray<float> h = nd::basic_ndarray<float>::uninitialized({2, 3});
    for (long long i = 0; i < 6; ++i) h.buffer()->data()[i] = static_cast<float>(i) * 1.5f;
    gl::device_array<float> d = gl::to_device(h);
    EXPECT_EQ(d.size(), 6u);
    nd::basic_ndarray<float> back = gl::to_host(d);
    ASSERT_EQ(back.shape(), h.shape());
    for (long long i = 0; i < 6; ++i)
        EXPECT_EQ(back.buffer()->data()[i], h.buffer()->data()[i]) << i;
}

TEST(Factories, ToDeviceRejectsNonContiguous) {
    nd::basic_ndarray<float> h = nd::basic_ndarray<float>::uninitialized({3, 4});
    for (long long i = 0; i < 12; ++i) h.buffer()->data()[i] = static_cast<float>(i);
    // A transposed VIEW over the same buffer: swapped strides, not contiguous.
    nd::basic_ndarray<float> t(h.buffer(), {4, 3}, {1, 4}, 0);
    ASSERT_FALSE(nd::is_contiguous(t));
    EXPECT_THROW((void)gl::to_device(t), std::runtime_error);
}

TEST(Factories, FillKernelRealAndStagedComplex) {
    // full<float> drives the on-device fill kernel; full<complex> takes the staged-upload
    // fallback (no complex fill kernel, mirroring the elementwise family).
    gl::device_array<float> f = gl::full<float>({300}, 2.5f);
    std::vector<float> got(300);
    f.to_host(got.data());
    for (float v : got) ASSERT_EQ(v, 2.5f);

    gl::device_array<std::complex<float>> c =
        gl::full<std::complex<float>>({17}, std::complex<float>(1.0f, -2.0f));
    std::vector<std::complex<float>> cg(17);
    c.to_host(cg.data());
    for (auto v : cg) ASSERT_EQ(v, std::complex<float>(1.0f, -2.0f));
}

TEST(Factories, ZerosOnesFullFamily) {
    gl::device_array<double> z = gl::zeros({2, 2});
    std::vector<double> zg(4);
    z.to_host(zg.data());
    for (double v : zg) EXPECT_EQ(v, 0.0);

    gl::device_array<double> o = gl::ones({3});
    std::vector<double> og(3);
    o.to_host(og.data());
    for (double v : og) EXPECT_EQ(v, 1.0);

    gl::device_array<float> f = gl::full<float>({2, 3}, 7.0f);
    gl::device_array<float> fl = gl::full_like(f, -1.0f);
    gl::device_array<float> zl = gl::zeros_like(f);
    gl::device_array<float> ol = gl::ones_like(f);
    std::vector<float> g(6);
    f.to_host(g.data());
    for (float v : g) EXPECT_EQ(v, 7.0f);
    fl.to_host(g.data());
    for (float v : g) EXPECT_EQ(v, -1.0f);
    zl.to_host(g.data());
    for (float v : g) EXPECT_EQ(v, 0.0f);
    ol.to_host(g.data());
    for (float v : g) EXPECT_EQ(v, 1.0f);
}

TEST(Factories, ArrayScalarArange) {
    gl::device_array<float> a = gl::array<float>({1.0f, 2.0f, 3.0f});
    EXPECT_EQ(a.size(), 3u);
    EXPECT_EQ(gl::get(a, {2}), 3.0f);

    gl::device_array<double> v = gl::array(std::vector<double>{4.0, 5.0});
    EXPECT_EQ(gl::get(v, {0}), 4.0);

    gl::device_array<float> s = gl::scalar(9.0f);
    EXPECT_EQ(s.size(), 1u);
    EXPECT_EQ(gl::get(s, {}), 9.0f);  // rank-0: the empty index reads the one element

    gl::device_array<float> r = gl::arange(0.0f, 5.0f, 1.0f);
    EXPECT_EQ(r.size(), 5u);
    EXPECT_EQ(gl::get(r, {4}), 4.0f);
}

TEST(Factories, ReshapeAndAccessors) {
    gl::device_array<float> a = gl::arange(0.0f, 6.0f, 1.0f);
    gl::device_array<float> m = gl::reshape(a, {2, 3});
    EXPECT_EQ(m.ndim(), 2u);
    EXPECT_EQ(gl::get(m, {1, 2}), 5.0f);
    EXPECT_EQ(gl::size_of(m), 6);
    EXPECT_TRUE(gl::is_contiguous(m));
}

TEST(Factories, PrintingMatchesContents) {
    gl::device_array<float> a = gl::array<float>({1.5f, -2.0f});
    const std::string s = gl::to_string(a);
    EXPECT_NE(s.find("1.5"), std::string::npos) << s;
    EXPECT_NE(s.find("-2"), std::string::npos) << s;
    std::ostringstream os;
    os << a;
    EXPECT_EQ(os.str(), s);
}

TEST(Factories, TransferStatsLedgerMoves) {
    gl::reset_stats();
    const auto& st = gl::stats();
    gl::device_array<float> a = gl::array<float>({1.0f, 2.0f, 3.0f, 4.0f});
    std::vector<float> out(4);
    a.to_host(out.data());
    EXPECT_GE(st.bytes_uploaded, 4u * sizeof(float));
    EXPECT_GE(st.bytes_downloaded, 4u * sizeof(float));
    gl::reset_stats();
    EXPECT_EQ(st.bytes_uploaded, 0u);
    EXPECT_EQ(st.bytes_downloaded, 0u);
}

TEST(Factories, SmallArraySumMeanSingleStage) {
    // n far below kTwoStageMin exercises the one-stage partial branch of the reduction.
    gl::device_array<float> a = gl::array<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    EXPECT_NEAR(gl::sum(a), 15.0f, 1e-6);
    EXPECT_NEAR(gl::mean(a), 3.0f, 1e-6);
    gl::device_array<double> d = gl::array(std::vector<double>{2.0, 4.0});
    EXPECT_NEAR(gl::sum(d), 6.0, 1e-12);
    EXPECT_NEAR(gl::mean(d), 3.0, 1e-12);
}

}  // namespace
