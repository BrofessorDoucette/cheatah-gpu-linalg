// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// gpu_linalg_bench.cpp — the per-function benchmark suite, paired host/dev/e2e variants (the
// eigen_compare_bench.cpp pattern): BM_<op>_<variant>/<size>.
//
//   host — cheatah's own single-threaded SIMD linalg on basic_ndarray (routines.cpp is compiled
//          into this binary at -O3 -march=native; the honest same-process CPU baseline);
//   dev  — device-resident: operands uploaded ONCE outside the timed loop, result left resident
//          (the steady-state training-loop shape); the blocking dispatch is the measured unit;
//   e2e  — upload + compute + download inside the loop (the "one-shot offload" shape).
//
// GFLOP/s / GB/s are attached as counters where meaningful. BM_dispatch_overhead is first-class:
// it prices the fixed cost every dev-variant op pays. The backend axis (NVIDIA / llvmpipe /
// emulated Metal) is runtime: the same env forcing scripts/qa.sh uses.
#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

#include <benchmark/benchmark.h>

#include <cstring>
#include <vector>

namespace gl = cheatah::gpu::linalg;
namespace hl = cheatah::linalg;
using HostD = cheatah::ndarray::basic_ndarray<double>;

namespace {

// Deterministic operand fill (harness::fill's cycle, inlined so bench/ has no test include).
double fill(std::size_t i, std::size_t salt) {
    return 0.25 * static_cast<double>((i * 7 + salt) % 16) - 1.0;
}

template <class T>
std::vector<T> seq(std::size_t n, std::size_t salt) {
    std::vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<T>(fill(i, salt));
    return v;
}

HostD host_mat(std::size_t r, std::size_t c, std::size_t salt) {
    HostD m = HostD::uninitialized({r, c});
    for (std::size_t i = 0; i < r * c; ++i) m.buffer()->data()[i] = fill(i, salt);
    return m;
}

// ---- matmul ------------------------------------------------------------------------------------

void BM_matmul_host(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const HostD a = host_mat(n, n, 1), b = host_mat(n, n, 2);
    HostD c = HostD::uninitialized({n, n});
    for (auto _ : state) {
        hl::matmul(c, a, b);
        benchmark::DoNotOptimize(c.buffer()->data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        2.0 * n * n * n * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_matmul_host)->RangeMultiplier(2)->Range(64, 1024)->UseRealTime()->Unit(benchmark::kMicrosecond);

template <class T>
void matmul_dev_impl(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::vector<T> av = seq<T>(n * n, 1), bv = seq<T>(n * n, 2);
    gl::device_array<T> a = gl::device_array<T>::from_host({n, n}, av.data());
    gl::device_array<T> b = gl::device_array<T>::from_host({n, n}, bv.data());
    gl::device_array<T> c = gl::device_array<T>::uninitialized({n, n});
    for (auto _ : state) {
        gl::matmul(c, a, b);   // the DeviceArray kernel overload — blocking dispatch
        benchmark::DoNotOptimize(c.buffer());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        2.0 * n * n * n * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
void BM_matmul_f32_dev(benchmark::State& s) { matmul_dev_impl<float>(s); }
void BM_matmul_f64_dev(benchmark::State& s) { matmul_dev_impl<double>(s); }
BENCHMARK(BM_matmul_f32_dev)->RangeMultiplier(2)->Range(64, 4096)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_matmul_f64_dev)->RangeMultiplier(2)->Range(64, 2048)->UseRealTime()->Unit(benchmark::kMicrosecond);

template <class T>
void matmul_e2e_impl(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::vector<T> av = seq<T>(n * n, 1), bv = seq<T>(n * n, 2);
    std::vector<T> out(n * n);
    for (auto _ : state) {
        gl::device_array<T> a = gl::device_array<T>::from_host({n, n}, av.data());
        gl::device_array<T> b = gl::device_array<T>::from_host({n, n}, bv.data());
        hl::matmul(a, b).to_host(out.data());
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        2.0 * n * n * n * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
void BM_matmul_f32_e2e(benchmark::State& s) { matmul_e2e_impl<float>(s); }
BENCHMARK(BM_matmul_f32_e2e)->RangeMultiplier(2)->Range(64, 4096)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- batched matmul: one z-dispatch vs B separate 2-D dispatches -------------------------------

template <class T>
void matmul_batched_impl(benchmark::State& state, bool z_batched) {
    const std::size_t B = static_cast<std::size_t>(state.range(0));
    const std::size_t n = static_cast<std::size_t>(state.range(1));
    const std::vector<T> av = seq<T>(B * n * n, 1), bv = seq<T>(B * n * n, 2);
    gl::device_array<T> a3 = gl::device_array<T>::from_host({B, n, n}, av.data());
    gl::device_array<T> b3 = gl::device_array<T>::from_host({B, n, n}, bv.data());
    gl::device_array<T> c3 = gl::device_array<T>::uninitialized({B, n, n});
    // the looped-2-D comparison reuses per-slice views by uploading each slice once
    std::vector<gl::device_array<T>> a2, b2, c2;
    if (!z_batched)
        for (std::size_t z = 0; z < B; ++z) {
            a2.push_back(gl::device_array<T>::from_host({n, n}, av.data() + z * n * n));
            b2.push_back(gl::device_array<T>::from_host({n, n}, bv.data() + z * n * n));
            c2.push_back(gl::device_array<T>::uninitialized({n, n}));
        }
    for (auto _ : state) {
        if (z_batched) {
            gl::matmul(c3, a3, b3);
        } else {
            for (std::size_t z = 0; z < B; ++z) gl::matmul(c2[z], a2[z], b2[z]);
        }
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        2.0 * B * n * n * n * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
void BM_matmul_batched_f32_dev(benchmark::State& s) { matmul_batched_impl<float>(s, true); }
void BM_matmul_looped_f32_dev(benchmark::State& s) { matmul_batched_impl<float>(s, false); }
BENCHMARK(BM_matmul_batched_f32_dev)->Args({64, 64})->Args({64, 256})->Args({8, 256})
    ->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_matmul_looped_f32_dev)->Args({64, 64})->Args({64, 256})->Args({8, 256})
    ->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- reductions + elementwise ------------------------------------------------------------------

void BM_dot_host(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const HostD a = host_mat(1, n, 3), b = host_mat(1, n, 4);
    for (auto _ : state) {
        double d = hl::dot(a, b);
        benchmark::DoNotOptimize(d);
    }
    state.counters["GB/s"] = benchmark::Counter(
        2.0 * n * sizeof(double) * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_dot_host)->Range(1 << 14, 1 << 26)->RangeMultiplier(8)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_dot_host)->Arg(1 << 10)->Arg(1 << 12)->UseRealTime()->Unit(benchmark::kMicrosecond);

template <class T>
void dot_dev_impl(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::vector<T> av = seq<T>(n, 3), bv = seq<T>(n, 4);
    gl::device_array<T> a = gl::device_array<T>::from_host({n}, av.data());
    gl::device_array<T> b = gl::device_array<T>::from_host({n}, bv.data());
    for (auto _ : state) {
        T d;
        gl::dot(d, a, b);
        benchmark::DoNotOptimize(d);
    }
    state.counters["GB/s"] = benchmark::Counter(
        2.0 * n * sizeof(T) * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
void BM_dot_f32_dev(benchmark::State& s) { dot_dev_impl<float>(s); }
void BM_dot_f64_dev(benchmark::State& s) { dot_dev_impl<double>(s); }
BENCHMARK(BM_dot_f32_dev)->Range(1 << 14, 1 << 26)->RangeMultiplier(8)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_dot_f64_dev)->Range(1 << 14, 1 << 26)->RangeMultiplier(8)->UseRealTime()->Unit(benchmark::kMicrosecond);

template <class T>
void sum_dev_impl(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::vector<T> av = seq<T>(n, 5);
    gl::device_array<T> a = gl::device_array<T>::from_host({n}, av.data());
    for (auto _ : state) {
        T s = gl::sum(a);
        benchmark::DoNotOptimize(s);
    }
    state.counters["GB/s"] = benchmark::Counter(
        1.0 * n * sizeof(T) * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
void BM_sum_f32_dev(benchmark::State& s) { sum_dev_impl<float>(s); }
void BM_sum_f64_dev(benchmark::State& s) { sum_dev_impl<double>(s); }
BENCHMARK(BM_sum_f32_dev)->Range(1 << 14, 1 << 26)->RangeMultiplier(8)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_sum_f64_dev)->Range(1 << 10, 1 << 24)->RangeMultiplier(4)->UseRealTime()->Unit(benchmark::kMicrosecond);

template <class T>
void add_dev_impl(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::vector<T> av = seq<T>(n, 6), bv = seq<T>(n, 7);
    gl::device_array<T> a = gl::device_array<T>::from_host({n}, av.data());
    gl::device_array<T> b = gl::device_array<T>::from_host({n}, bv.data());
    gl::device_array<T> c = gl::device_array<T>::uninitialized({n});
    for (auto _ : state) {
        gl::add(c, a, b);
        benchmark::ClobberMemory();
    }
    state.counters["GB/s"] = benchmark::Counter(
        3.0 * n * sizeof(T) * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
void BM_add_f32_dev(benchmark::State& s) { add_dev_impl<float>(s); }
void BM_add_f64_dev(benchmark::State& s) { add_dev_impl<double>(s); }
BENCHMARK(BM_add_f32_dev)->Range(1 << 14, 1 << 26)->RangeMultiplier(8)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_add_f64_dev)->Range(1 << 10, 1 << 24)->RangeMultiplier(4)->UseRealTime()->Unit(benchmark::kMicrosecond);

template <class T>
void axpy_dev_impl(benchmark::State& state, bool fused) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::vector<T> xv = seq<T>(n, 8), yv = seq<T>(n, 9);
    gl::device_array<T> x = gl::device_array<T>::from_host({n}, xv.data());
    gl::device_array<T> y = gl::device_array<T>::from_host({n}, yv.data());
    gl::device_array<T> out = gl::device_array<T>::uninitialized({n});
    for (auto _ : state) {
        if (fused) {
            gl::axpy(out, T(2.5), x, y);
        } else {
            out = x * T(2.5) + y;   // two dispatches + a temporary — what fusion saves
        }
        benchmark::ClobberMemory();
    }
    state.counters["GB/s"] = benchmark::Counter(
        3.0 * n * sizeof(T) * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
void BM_axpy_fused_f32_dev(benchmark::State& s) { axpy_dev_impl<float>(s, true); }
void BM_axpy_fused_f64_dev(benchmark::State& s) { axpy_dev_impl<double>(s, true); }
void BM_axpy_chained_f32_dev(benchmark::State& s) { axpy_dev_impl<float>(s, false); }
BENCHMARK(BM_axpy_fused_f32_dev)->Range(1 << 14, 1 << 26)->RangeMultiplier(8)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_axpy_chained_f32_dev)->Range(1 << 14, 1 << 26)->RangeMultiplier(8)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_axpy_fused_f64_dev)->Range(1 << 10, 1 << 24)->RangeMultiplier(4)->UseRealTime()->Unit(benchmark::kMicrosecond);

// Host mirrors for the crossover table (double — the elements cheatah's host linalg ships).
// add uses the allocating operator form on purpose: that IS the host expression the device
// out-param call replaces in generic code.
void BM_sum_host(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const HostD a = host_mat(1, n, 5);
    for (auto _ : state) {
        double v = cheatah::ndarray::sum(a);
        benchmark::DoNotOptimize(v);
    }
    state.counters["GB/s"] = benchmark::Counter(
        1.0 * n * sizeof(double) * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
void BM_add_host(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const HostD a = host_mat(1, n, 6), b = host_mat(1, n, 7);
    for (auto _ : state) {
        HostD c = a + b;
        benchmark::DoNotOptimize(c.buffer());
    }
    state.counters["GB/s"] = benchmark::Counter(
        3.0 * n * sizeof(double) * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
void BM_axpy_host(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const HostD x = host_mat(1, n, 8), y = host_mat(1, n, 9);
    HostD out = HostD::uninitialized({1, n});
    for (auto _ : state) {
        gl::axpy(out, 2.5, x, y);
        benchmark::ClobberMemory();
    }
    state.counters["GB/s"] = benchmark::Counter(
        3.0 * n * sizeof(double) * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_sum_host)->Range(1 << 10, 1 << 24)->RangeMultiplier(4)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_add_host)->Range(1 << 10, 1 << 24)->RangeMultiplier(4)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_axpy_host)->Range(1 << 10, 1 << 24)->RangeMultiplier(4)->UseRealTime()->Unit(benchmark::kMicrosecond);

// Small-size points for the crossover table (the wide ranges above start at 16k).
BENCHMARK(BM_dot_f64_dev)->Arg(1 << 10)->Arg(1 << 12)->UseRealTime()->Unit(benchmark::kMicrosecond);

void BM_transpose_f32_dev(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::vector<float> av = seq<float>(n * n, 10);
    gl::device_array<float> a = gl::device_array<float>::from_host({n, n}, av.data());
    gl::device_array<float> t = gl::device_array<float>::uninitialized({n, n});
    for (auto _ : state) {
        gl::conj_transpose(t, a);
        benchmark::ClobberMemory();
    }
    state.counters["GB/s"] = benchmark::Counter(
        2.0 * n * n * sizeof(float) * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_transpose_f32_dev)->RangeMultiplier(2)->Range(256, 4096)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- the fixed cost every op pays --------------------------------------------------------------

void BM_dispatch_overhead(benchmark::State& state) {
    const double one = 1.0;
    gl::device_array<double> a = gl::device_array<double>::from_host({1}, &one);
    for (auto _ : state) {
        double d;
        gl::dot(d, a, a);   // 1-element dot: ~pure dispatch + scratch cost
        benchmark::DoNotOptimize(d);
    }
}
BENCHMARK(BM_dispatch_overhead)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- host-bridged factorizations (labelled: these EXECUTE ON HOST + transfer) ------------------

void BM_bridge_svd_e2e(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::vector<double> av = seq<double>(n * n, 11);
    gl::device_array<double> a = gl::device_array<double>::from_host({n, n}, av.data());
    for (auto _ : state) {
        auto s = hl::svd(a);   // host-bridged
        benchmark::DoNotOptimize(s.s.buffer());
    }
}
BENCHMARK(BM_bridge_svd_e2e)->Arg(64)->Arg(256)->UseRealTime()->Unit(benchmark::kMillisecond);

void BM_bridge_solve_e2e(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    std::vector<double> av = seq<double>(n * n, 12);
    for (std::size_t i = 0; i < n; ++i) av[i * n + i] += static_cast<double>(n);  // well-conditioned
    const std::vector<double> bv = seq<double>(n, 13);
    gl::device_array<double> a = gl::device_array<double>::from_host({n, n}, av.data());
    gl::device_array<double> b = gl::device_array<double>::from_host({n}, bv.data());
    for (auto _ : state) {
        auto x = hl::solve(a, b);   // host-bridged
        benchmark::DoNotOptimize(x.buffer());
    }
}
BENCHMARK(BM_bridge_solve_e2e)->Arg(64)->Arg(256)->Arg(1024)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- conv-support kernels (device-resident, the training-loop shape B=16 C=64 32² K=3) ---------

// The shared realistic conv geometry: B=16, C=F=64, H=W=32, 3×3, stride 1, pad 1 → OH=OW=32.
constexpr long long cB = 16, cC = 64, cH = 32, cW = 32, cK = 3, cF = 64, cOO = 32 * 32;
constexpr std::size_t cXN = cB * cC * cH * cW;                 // 1 Mi elements
constexpr std::size_t cColN = cC * cK * cK * cB * cOO;         // 9 Mi elements
constexpr std::size_t cAN = cB * cF * cOO;                     // 1 Mi elements

template <class T>
void im2col2d_dev_impl(benchmark::State& state) {
    const std::vector<T> xv = seq<T>(cXN, 1);
    gl::device_array<T> x = gl::device_array<T>::from_host({cXN}, xv.data());
    gl::device_array<T> col = gl::device_array<T>::uninitialized({cColN});
    for (auto _ : state) {
        gl::im2col2d(col, x, cB, cC, cH, cW, cK, cK, 32, 32, 1, 1);
        benchmark::DoNotOptimize(col.buffer());
        benchmark::ClobberMemory();
    }
    state.counters["GB/s"] = benchmark::Counter(   // one gathered read + one write per col cell
        2.0 * cColN * sizeof(T) * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
void BM_im2col2d_f32_dev(benchmark::State& s) { im2col2d_dev_impl<float>(s); }
void BM_im2col2d_f64_dev(benchmark::State& s) { im2col2d_dev_impl<double>(s); }
BENCHMARK(BM_im2col2d_f32_dev)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_im2col2d_f64_dev)->UseRealTime()->Unit(benchmark::kMicrosecond);

template <class T>
void col2im2d_dev_impl(benchmark::State& state) {
    const std::vector<T> cv = seq<T>(cColN, 2);
    gl::device_array<T> dcol = gl::device_array<T>::from_host({cColN}, cv.data());
    gl::device_array<T> dx = gl::device_array<T>::uninitialized({cXN});
    for (auto _ : state) {
        gl::col2im2d(dx, dcol, cB, cC, cH, cW, cK, cK, 32, 32, 1, 1);
        benchmark::DoNotOptimize(dx.buffer());
        benchmark::ClobberMemory();
    }
    state.counters["GB/s"] = benchmark::Counter(   // ≤ K² gathered reads per dx cell + one write
        (1.0 * cColN + cXN) * sizeof(T) * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
void BM_col2im2d_f32_dev(benchmark::State& s) { col2im2d_dev_impl<float>(s); }
void BM_col2im2d_f64_dev(benchmark::State& s) { col2im2d_dev_impl<double>(s); }
BENCHMARK(BM_col2im2d_f32_dev)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_col2im2d_f64_dev)->UseRealTime()->Unit(benchmark::kMicrosecond);

template <class T>
void conv_bias_act_dev_impl(benchmark::State& state) {
    const std::vector<T> ycv = seq<T>(cAN, 3), bv = seq<T>(cF, 4);
    gl::device_array<T> yc = gl::device_array<T>::from_host({cAN}, ycv.data());
    gl::device_array<T> bias = gl::device_array<T>::from_host({static_cast<std::size_t>(cF)}, bv.data());
    gl::device_array<T> a = gl::device_array<T>::uninitialized({cAN});
    for (auto _ : state) {
        gl::conv_bias_act(a, yc, bias, cB, cF, cOO, 1);   // relu — the common training act
        benchmark::DoNotOptimize(a.buffer());
        benchmark::ClobberMemory();
    }
    state.counters["GB/s"] = benchmark::Counter(
        2.0 * cAN * sizeof(T) * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
void BM_conv_bias_act_f32_dev(benchmark::State& s) { conv_bias_act_dev_impl<float>(s); }
void BM_conv_bias_act_f64_dev(benchmark::State& s) { conv_bias_act_dev_impl<double>(s); }
BENCHMARK(BM_conv_bias_act_f32_dev)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_conv_bias_act_f64_dev)->UseRealTime()->Unit(benchmark::kMicrosecond);

template <class T>
void conv_act_grad_dev_impl(benchmark::State& state) {
    const std::vector<T> dv = seq<T>(cAN, 5), av = seq<T>(cAN, 6);
    gl::device_array<T> d = gl::device_array<T>::from_host({cAN}, dv.data());
    gl::device_array<T> a = gl::device_array<T>::from_host({cAN}, av.data());
    gl::device_array<T> dyc = gl::device_array<T>::uninitialized({cAN});
    for (auto _ : state) {
        gl::conv_act_grad(dyc, d, a, cB, cF, cOO, 1);
        benchmark::DoNotOptimize(dyc.buffer());
        benchmark::ClobberMemory();
    }
    state.counters["GB/s"] = benchmark::Counter(
        3.0 * cAN * sizeof(T) * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
void BM_conv_act_grad_f32_dev(benchmark::State& s) { conv_act_grad_dev_impl<float>(s); }
void BM_conv_act_grad_f64_dev(benchmark::State& s) { conv_act_grad_dev_impl<double>(s); }
BENCHMARK(BM_conv_act_grad_f32_dev)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_conv_act_grad_f64_dev)->UseRealTime()->Unit(benchmark::kMicrosecond);

}  // namespace

// ---- tensor-core GEMM (opt-in KHR cooperative matrix; f16 inputs, f32 accumulate) --------------
#if defined(CHEATAH_GPU_LINALG_VULKAN)
namespace {

// float -> IEEE half bits (round-to-nearest-even is overkill for bench operands; truncation of
// the deterministic quarter-step fill values is exact anyway — they are all half-representable).
std::uint16_t to_half(float f) {
    union { float f; std::uint32_t u; } v{f};
    const std::uint32_t sign = (v.u >> 16) & 0x8000u;
    const std::int32_t exp = static_cast<std::int32_t>((v.u >> 23) & 0xFF) - 127 + 15;
    const std::uint32_t mant = (v.u >> 13) & 0x3FFu;
    if (exp <= 0) return static_cast<std::uint16_t>(sign);
    if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | mant);
}

void BM_matmul_f16coop_dev(benchmark::State& state) {
    gl::detail::Context& c = gl::detail::ctx();
    if (!c.coop_ok()) {
        state.SkipWithError("no VK_KHR_cooperative_matrix on this device");
        return;
    }
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    std::vector<std::uint16_t> ah(n * n), bh(n * n);
    for (std::size_t i = 0; i < n * n; ++i) {
        ah[i] = to_half(static_cast<float>(fill(i, 1)));
        bh[i] = to_half(static_cast<float>(fill(i, 2)));
    }
    gl::detail::Buffer* A = c.new_data_buffer(n * n * sizeof(std::uint16_t));
    gl::detail::Buffer* B = c.new_data_buffer(n * n * sizeof(std::uint16_t));
    gl::detail::Buffer* C = c.new_data_buffer(n * n * sizeof(float));
    c.upload(A, ah.data(), n * n * sizeof(std::uint16_t));
    c.upload(B, bh.data(), n * n * sizeof(std::uint16_t));
    const std::uint32_t dims[3] = {static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(n),
                                   static_cast<std::uint32_t>(n)};
    gl::detail::Buffer* D = gl::detail::dims_buffer(dims, 3);
    gl::detail::Buffer* bind[4] = {A, B, C, D};
    const std::uint32_t gx = static_cast<std::uint32_t>(n / 128);
    const std::uint32_t gy = static_cast<std::uint32_t>(n / 128);

    // Correctness spot-check before timing: f16-quantized reference on a few elements.
    c.dispatch_blocks_2d("gemm_coop_f32", bind, 4, gx, gy);
    std::vector<float> out(n);
    c.download(C, out.data(), n * sizeof(float));   // first row
    {
        std::vector<float> af(n), bfcol(n);
        for (std::size_t k = 0; k < n; ++k) af[k] = static_cast<float>(fill(k, 1));
        double want0 = 0;
        for (std::size_t k = 0; k < n; ++k)
            want0 += static_cast<double>(af[k]) * static_cast<double>(fill(k * n, 2));
        if (std::abs(out[0] - want0) > 1e-2 * std::max(1.0, std::abs(want0))) {
            state.SkipWithError("coop GEMM mismatch vs reference");
            return;
        }
    }

    for (auto _ : state) {
        c.dispatch_blocks_2d("gemm_coop_f32", bind, 4, gx, gy);
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        2.0 * n * n * n * state.iterations() / 1e9, benchmark::Counter::kIsRate);
    c.release_buffer(A);
    c.release_buffer(B);
    c.release_buffer(C);
    c.release_buffer(D);
}
BENCHMARK(BM_matmul_f16coop_dev)->RangeMultiplier(2)->Range(1024, 4096)->UseRealTime()->Unit(benchmark::kMicrosecond);

}  // namespace
#endif  // CHEATAH_GPU_LINALG_VULKAN

// ---- round 4: bandwidth ceiling, shape sweeps, dispatch breakdown, transfers -------------------
namespace {

// The machine's REAL sustainable bandwidth: pure vec4 stream copy (1R+1W) and triad (2R+1W).
void bw_impl(benchmark::State& state, bool triad) {
    gl::detail::Context& c = gl::detail::ctx();
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    gl::detail::Buffer* A = c.new_data_buffer(n * sizeof(float));
    gl::detail::Buffer* B = c.new_data_buffer(n * sizeof(float));
    gl::detail::Buffer* C = c.new_data_buffer(n * sizeof(float));
    gl::detail::Buffer* S = c.new_buffer(sizeof(float));
    const float s = 2.5f;
    std::memcpy(gl::detail::Context::contents(S), &s, sizeof(float));
    const std::uint32_t dims[1] = {static_cast<std::uint32_t>(n)};
    gl::detail::Buffer* D = gl::detail::dims_buffer(dims, 1);
    const std::uint64_t width = (n + 3) / 4;
    for (auto _ : state) {
        if (triad) {
            gl::detail::Buffer* bind[5] = {A, B, C, S, D};
            c.dispatch_1d("triad_f32", bind, 5, width);
        } else {
            gl::detail::Buffer* bind[3] = {B, A, D};
            c.dispatch_1d("copy_f32", bind, 3, width);
        }
        benchmark::ClobberMemory();
    }
    const double bytes = (triad ? 3.0 : 2.0) * n * sizeof(float);
    state.counters["GB/s"] =
        benchmark::Counter(bytes * state.iterations() / 1e9, benchmark::Counter::kIsRate);
    c.release_buffer(A);
    c.release_buffer(B);
    c.release_buffer(C);
    c.release_buffer(S);
    c.release_buffer(D);
}
void BM_ceiling_copy(benchmark::State& s) { bw_impl(s, false); }
void BM_ceiling_triad(benchmark::State& s) { bw_impl(s, true); }
BENCHMARK(BM_ceiling_copy)->Arg(1 << 26)->Arg(1 << 27)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ceiling_triad)->Arg(1 << 26)->Arg(1 << 27)->UseRealTime()->Unit(benchmark::kMicrosecond);

// Training-realistic GEMM rectangles (fast path where exactly tiled, edge otherwise — the name
// carries the shape so regressions attribute cleanly).
template <class T>
void matmul_shape_impl(benchmark::State& state) {
    const std::size_t M = static_cast<std::size_t>(state.range(0));
    const std::size_t K = static_cast<std::size_t>(state.range(1));
    const std::size_t N = static_cast<std::size_t>(state.range(2));
    const std::vector<T> av = seq<T>(M * K, 1), bv = seq<T>(K * N, 2);
    gl::device_array<T> a = gl::device_array<T>::from_host({M, K}, av.data());
    gl::device_array<T> b = gl::device_array<T>::from_host({K, N}, bv.data());
    gl::device_array<T> c = gl::device_array<T>::uninitialized({M, N});
    for (auto _ : state) {
        gl::matmul(c, a, b);
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        2.0 * M * K * N * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
void BM_matmul_shapes_f32_dev(benchmark::State& s) { matmul_shape_impl<float>(s); }
BENCHMARK(BM_matmul_shapes_f32_dev)
    ->Args({512, 1024, 1024})   // MLP layer
    ->Args({64, 4096, 4096})    // skinny-M, K-heavy
    ->Args({4096, 64, 4096})    // thin-K (bandwidth-shaped)
    ->UseRealTime()->Unit(benchmark::kMicrosecond);

// Upload/download GB/s vs size — the PCIe reality that the crossover table folds in.
void BM_upload(benchmark::State& state) {
    gl::detail::Context& c = gl::detail::ctx();
    const std::size_t bytes = static_cast<std::size_t>(state.range(0));
    std::vector<char> host(bytes, 1);
    gl::detail::Buffer* B = c.new_data_buffer(bytes);
    for (auto _ : state) c.upload(B, host.data(), bytes);
    state.counters["GB/s"] = benchmark::Counter(
        double(bytes) * state.iterations() / 1e9, benchmark::Counter::kIsRate);
    c.release_buffer(B);
}
void BM_download(benchmark::State& state) {
    gl::detail::Context& c = gl::detail::ctx();
    const std::size_t bytes = static_cast<std::size_t>(state.range(0));
    std::vector<char> host(bytes);
    gl::detail::Buffer* B = c.new_data_buffer(bytes);
    for (auto _ : state) c.download(B, host.data(), bytes);
    state.counters["GB/s"] = benchmark::Counter(
        double(bytes) * state.iterations() / 1e9, benchmark::Counter::kIsRate);
    c.release_buffer(B);
}
BENCHMARK(BM_upload)->Arg(1 << 20)->Arg(1 << 24)->Arg(1 << 28)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_download)->Arg(1 << 20)->Arg(1 << 24)->Arg(1 << 28)->UseRealTime()->Unit(benchmark::kMicrosecond);

}  // namespace

// Factory cost: zeros() was host-vector fill + full-size PCIe upload; now a single fill dispatch.
void BM_zeros_f64_dev(benchmark::State& state) {
    const long long n = state.range(0);
    for (auto _ : state) {
        gl::device_array<double> a = gl::zeros({n, n});
        benchmark::DoNotOptimize(a.buffer());
    }
    state.counters["GB/s"] = benchmark::Counter(
        double(n) * n * sizeof(double) * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_zeros_f64_dev)->Arg(1024)->Arg(4096)->UseRealTime()->Unit(benchmark::kMicrosecond);

// ---- round 5: peak probes + f16-accumulate GEMM (Vulkan-only diagnosis instruments) ------------
#if defined(CHEATAH_GPU_LINALG_VULKAN)
namespace {

// The REAL achievable f32 FLOP rate at sustained clocks: pure register FMA chains, no memory.
void BM_ffma_peak(benchmark::State& state) {
    gl::detail::Context& c = gl::detail::ctx();
    const std::uint32_t groups = 2048, iters = 2048;
    gl::detail::Buffer* out = c.new_data_buffer(groups * 256 * sizeof(float));
    const std::uint32_t dims[1] = {iters};
    gl::detail::Buffer* D = gl::detail::dims_buffer(dims, 1);
    gl::detail::Buffer* bind[2] = {out, D};
    for (auto _ : state) {
        c.dispatch_blocks_1d("ffma_peak_f32", bind, 2, groups);
        benchmark::ClobberMemory();
    }
    const double flop = double(groups) * 256 * iters * 16;
    state.counters["GFLOP/s"] =
        benchmark::Counter(flop * state.iterations() / 1e9, benchmark::Counter::kIsRate);
    c.release_buffer(out);
    c.release_buffer(D);
}
BENCHMARK(BM_ffma_peak)->UseRealTime()->Unit(benchmark::kMicrosecond);

// Groupshared vec4 load throughput — what the GEMM inner loop spends.
void BM_lds_peak(benchmark::State& state) {
    gl::detail::Context& c = gl::detail::ctx();
    const std::uint32_t groups = 512, iters = 512;
    gl::detail::Buffer* out = c.new_data_buffer(groups * 256 * 4 * sizeof(float));
    const std::uint32_t dims[1] = {iters};
    gl::detail::Buffer* D = gl::detail::dims_buffer(dims, 1);
    gl::detail::Buffer* bind[2] = {out, D};
    for (auto _ : state) {
        c.dispatch_blocks_1d("lds_peak_f32", bind, 2, groups);
        benchmark::ClobberMemory();
    }
    const double bytes = double(groups) * 256 * iters * 4 * 16;
    state.counters["GB/s"] =
        benchmark::Counter(bytes * state.iterations() / 1e9, benchmark::Counter::kIsRate);
    c.release_buffer(out);
    c.release_buffer(D);
}
BENCHMARK(BM_lds_peak)->UseRealTime()->Unit(benchmark::kMicrosecond);

// Tensor-core MulAdd ceiling per accumulator type — measures the GeForce f16-acc/f32-acc rate
// ratio directly (fragments loaded once; the loop is pure coopMatMulAdd).
void coop_peak_impl(benchmark::State& state, const char* kernel, std::size_t acc_bytes) {
    gl::detail::Context& c = gl::detail::ctx();
    if (!c.coop_ok()) {
        state.SkipWithError("no VK_KHR_cooperative_matrix on this device");
        return;
    }
    const std::uint32_t groups = 512, iters = 1024;
    std::vector<std::uint16_t> frag(256);
    for (std::size_t i = 0; i < 256; ++i) frag[i] = to_half(static_cast<float>(fill(i, 3)));
    gl::detail::Buffer* in = c.new_data_buffer(256 * sizeof(std::uint16_t));
    c.upload(in, frag.data(), 256 * sizeof(std::uint16_t));
    gl::detail::Buffer* out = c.new_data_buffer(groups * 4 * 256 * acc_bytes);
    const std::uint32_t dims[1] = {iters};
    gl::detail::Buffer* D = gl::detail::dims_buffer(dims, 1);
    gl::detail::Buffer* bind[3] = {in, out, D};
    for (auto _ : state) {
        c.dispatch_blocks_1d(kernel, bind, 3, groups);
        benchmark::ClobberMemory();
    }
    const double flop = double(groups) * 4 * iters * 4 * 2 * 16 * 16 * 16;
    state.counters["GFLOP/s"] =
        benchmark::Counter(flop * state.iterations() / 1e9, benchmark::Counter::kIsRate);
    c.release_buffer(in);
    c.release_buffer(out);
    c.release_buffer(D);
}
void BM_coop_peak_f32acc(benchmark::State& s) { coop_peak_impl(s, "coop_peak_f32acc_f32", 4); }
void BM_coop_peak_f16acc(benchmark::State& s) { coop_peak_impl(s, "coop_peak_f16acc_f32", 2); }
BENCHMARK(BM_coop_peak_f32acc)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_coop_peak_f16acc)->UseRealTime()->Unit(benchmark::kMicrosecond);

// f16-ACCUMULATE tensor GEMM (inference-class numerics — the f32-acc kernel stays the default).
void BM_matmul_f16acc_coop_dev(benchmark::State& state) {
    gl::detail::Context& c = gl::detail::ctx();
    if (!c.coop_ok()) {
        state.SkipWithError("no VK_KHR_cooperative_matrix on this device");
        return;
    }
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    std::vector<std::uint16_t> ah(n * n), bh(n * n);
    for (std::size_t i = 0; i < n * n; ++i) {
        ah[i] = to_half(static_cast<float>(fill(i, 1)));
        bh[i] = to_half(static_cast<float>(fill(i, 2)));
    }
    gl::detail::Buffer* A = c.new_data_buffer(n * n * sizeof(std::uint16_t));
    gl::detail::Buffer* B = c.new_data_buffer(n * n * sizeof(std::uint16_t));
    gl::detail::Buffer* C = c.new_data_buffer(n * n * sizeof(std::uint16_t));
    c.upload(A, ah.data(), n * n * sizeof(std::uint16_t));
    c.upload(B, bh.data(), n * n * sizeof(std::uint16_t));
    const std::uint32_t dims[3] = {static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(n),
                                   static_cast<std::uint32_t>(n)};
    gl::detail::Buffer* D = gl::detail::dims_buffer(dims, 3);
    gl::detail::Buffer* bind[4] = {A, B, C, D};
    const std::uint32_t gx = static_cast<std::uint32_t>(n / 128);
    const std::uint32_t gy = static_cast<std::uint32_t>(n / 128);

    // Reference spot-check with a WIDENED tolerance (5% — f16 accumulation over K=n): the
    // speedup must not be a silent wrong answer, but bit-parity with f64 is not the contract.
    c.dispatch_blocks_2d("gemm_coop_f16acc_f32", bind, 4, gx, gy);
    std::vector<std::uint16_t> outh(n);
    c.download(C, outh.data(), n * sizeof(std::uint16_t));
    {
        double want0 = 0;
        for (std::size_t k = 0; k < n; ++k)
            want0 += fill(k, 1) * fill(k * n, 2);
        const std::uint16_t h = outh[0];
        const int exp = (h >> 10) & 0x1F;
        const double mant = 1.0 + double(h & 0x3FF) / 1024.0;
        double got = (exp == 0) ? 0.0 : std::ldexp(mant, exp - 15);
        if (h & 0x8000) got = -got;
        if (std::abs(got - want0) > 5e-2 * std::max(1.0, std::abs(want0))) {
            state.SkipWithError("f16acc GEMM mismatch vs reference");
            return;
        }
    }
    for (auto _ : state) {
        c.dispatch_blocks_2d("gemm_coop_f16acc_f32", bind, 4, gx, gy);
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        2.0 * n * n * n * state.iterations() / 1e9, benchmark::Counter::kIsRate);
    c.release_buffer(A);
    c.release_buffer(B);
    c.release_buffer(C);
    c.release_buffer(D);
}
// Capped at 2048: f16 accumulation over K=4096 drifts past the 5% tolerance with these
// operands — measured, and THE reason cuBLAS HGEMM accumulates in f32 (ledger finding).
BENCHMARK(BM_matmul_f16acc_coop_dev)->RangeMultiplier(2)->Range(1024, 2048)->UseRealTime()->Unit(benchmark::kMicrosecond);

}  // namespace
#endif  // CHEATAH_GPU_LINALG_VULKAN

#if defined(CHEATAH_GPU_LINALG_VULKAN)
namespace {

// Attribution probes: the fast GEMM minus one cost each; deltas vs BM_matmul_f32_dev name the
// shares of the ceiling gap. Results are intentionally wrong in some modes — timing only.
void attr_impl(benchmark::State& state, const char* kernel) {
    gl::detail::Context& c = gl::detail::ctx();
    const std::size_t n = 4096;
    const std::vector<float> av = seq<float>(n * n, 1), bv = seq<float>(n * n, 2);
    gl::detail::Buffer* A = c.new_data_buffer(n * n * sizeof(float));
    gl::detail::Buffer* B = c.new_data_buffer(n * n * sizeof(float));
    gl::detail::Buffer* C = c.new_data_buffer(n * n * sizeof(float));
    c.upload(A, av.data(), n * n * sizeof(float));
    c.upload(B, bv.data(), n * n * sizeof(float));
    const std::uint32_t dims[3] = {4096, 4096, 4096};
    gl::detail::Buffer* D = gl::detail::dims_buffer(dims, 3);
    gl::detail::Buffer* bind[4] = {A, B, C, D};
    for (auto _ : state) {
        c.dispatch_blocks_2d(kernel, bind, 4, 32, 32);
        benchmark::ClobberMemory();
    }
    state.counters["GFLOP/s"] = benchmark::Counter(
        2.0 * n * n * n * state.iterations() / 1e9, benchmark::Counter::kIsRate);
    c.release_buffer(A);
    c.release_buffer(B);
    c.release_buffer(C);
    c.release_buffer(D);
}
void BM_gemm_attr_nobar(benchmark::State& s) { attr_impl(s, "gemm_attr_nobar_f32"); }
void BM_gemm_attr_noglobal(benchmark::State& s) { attr_impl(s, "gemm_attr_noglobal_f32"); }
void BM_gemm_attr_noepi(benchmark::State& s) { attr_impl(s, "gemm_attr_noepi_f32"); }
BENCHMARK(BM_gemm_attr_nobar)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_gemm_attr_noglobal)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_gemm_attr_noepi)->UseRealTime()->Unit(benchmark::kMicrosecond);

}  // namespace
#endif  // CHEATAH_GPU_LINALG_VULKAN

#if defined(CHEATAH_GPU_LINALG_VULKAN)
namespace {

// The dispatch floor, split: full path (record + submit + fence) vs replaying a pre-recorded
// command buffer (submit + fence ONLY). The delta is what a command-buffer cache could save.
void BM_submit_only(benchmark::State& state) {
    gl::detail::Context& c = gl::detail::ctx();
    gl::detail::Buffer* out = c.new_data_buffer(256 * sizeof(float));
    const std::uint32_t dims[1] = {1};
    gl::detail::Buffer* D = gl::detail::dims_buffer(dims, 1);
    gl::detail::Buffer* bind[2] = {out, D};
    c.record_dispatch("ffma_peak_f32", bind, 2, {1, 1, 1});
    for (auto _ : state) {
        c.submit_recorded();
        benchmark::ClobberMemory();
    }
    c.release_buffer(out);
    c.release_buffer(D);
}
BENCHMARK(BM_submit_only)->UseRealTime()->Unit(benchmark::kMicrosecond);

// The same single-workgroup kernel through the FULL per-dispatch path, for the exact delta.
void BM_record_and_submit(benchmark::State& state) {
    gl::detail::Context& c = gl::detail::ctx();
    gl::detail::Buffer* out = c.new_data_buffer(256 * sizeof(float));
    const std::uint32_t dims[1] = {1};
    gl::detail::Buffer* D = gl::detail::dims_buffer(dims, 1);
    gl::detail::Buffer* bind[2] = {out, D};
    for (auto _ : state) {
        c.dispatch_blocks_1d("ffma_peak_f32", bind, 2, 1);
        benchmark::ClobberMemory();
    }
    c.release_buffer(out);
    c.release_buffer(D);
}
BENCHMARK(BM_record_and_submit)->UseRealTime()->Unit(benchmark::kMicrosecond);

}  // namespace
#endif  // CHEATAH_GPU_LINALG_VULKAN
