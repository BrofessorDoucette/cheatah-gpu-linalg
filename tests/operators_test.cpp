// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// operators_test.cpp — the purr-usability proof: every operator is called UNQUALIFIED from
// outside cheatah::gpu::linalg (exactly what purr's codegen emits — `a + b`, `2.0 * a`,
// `a += b`), resolved purely by ADL on device_array. Plus the firewall: device⊗host and
// mixed-element operator expressions must not compile.
#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

#include "tests/harness.hpp"

#include <vector>

namespace {

namespace gl = cheatah::gpu::linalg;
using HostD = cheatah::ndarray::basic_ndarray<double>;

// Firewall: no operator+ between a device array and a host ndarray, nor across element widths.
template <class A, class B>
concept Addable = requires(const A& a, const B& b) { a + b; };

TEST(Operators, Firewall) {
    static_assert(Addable<gl::device_array<double>, gl::device_array<double>>);
    static_assert(!Addable<gl::device_array<double>, HostD>);
    static_assert(!Addable<HostD, gl::device_array<double>>);
    static_assert(!Addable<gl::device_array<float>, gl::device_array<double>>);
}

constexpr std::size_t kN = 100;

struct Operands {
    std::vector<double> av, bv;
    gl::device_array<double> a, b;
    Operands()
        : av(harness::sequence<double>(kN, 30)),
          bv(kN),
          a(gl::device_array<double>::from_host({kN}, av.data())),
          b([this] {
              for (std::size_t i = 0; i < kN; ++i) bv[i] = harness::fill(i, 31) + 3.0;
              return gl::device_array<double>::from_host({kN}, bv.data());
          }()) {}
};

// Unqualified operator chains, exactly as purr emits them.
TEST(Operators, UnqualifiedChain) {
    Operands o;
    gl::device_array<double> c = o.a + o.b;
    gl::device_array<double> d = c * 2.0;
    gl::device_array<double> e = 1.0 - d;
    gl::device_array<double> f = e / 4.0;
    gl::device_array<double> g = -f;
    std::vector<double> got(kN);
    g.to_host(got.data());
    for (std::size_t i = 0; i < kN; ++i) {
        const double want = -((1.0 - (o.av[i] + o.bv[i]) * 2.0) / 4.0);
        EXPECT_NEAR_REL(got[i], want, 1e-12) << "chain[" << i << "]";
    }
}

// Compound assigns mutate in place.
TEST(Operators, CompoundAssigns) {
    Operands o;
    o.a += o.b;
    o.a *= 2.0;
    o.a -= o.b;
    std::vector<double> got(kN);
    o.a.to_host(got.data());
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_NEAR_REL(got[i], (o.av[i] + o.bv[i]) * 2.0 - o.bv[i], 1e-12)
            << "compound[" << i << "]";
}

// Scalar right-division (s / a).
TEST(Operators, ScalarRightDivision) {
    Operands o;
    gl::device_array<double> h = 10.0 / o.b;
    std::vector<double> got(kN);
    h.to_host(got.data());
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_NEAR_REL(got[i], 10.0 / o.bv[i], 1e-12) << "s/a[" << i << "]";
}

}  // namespace
