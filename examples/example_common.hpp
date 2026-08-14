// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
#pragma once

// example_common.hpp — the tiny shared skeleton of the examples: deterministic operand fills,
// wall-clock timing, host↔device agreement checks, the transfer-ledger printout, and the
// PASS:/FAIL: verdict the ctest registration keys off.
//
// The examples' whole point: ONE templated function, `step<host_array>(…)` vs
// `step<device_array>(…)` — the CPU and GPU versions are the same code, differing by a single
// compile-time template argument. Inside a templated body, write
// `using namespace cheatah::linalg;` once (the linalg fronts live there for BOTH containers);
// operators, factories (`zeros_like`, …) and reductions resolve per-container by ADL.

#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace example {

inline double fill(std::size_t i, std::size_t salt) {
    return 0.25 * static_cast<double>((i * 7 + salt) % 16) - 1.0;
}

inline std::vector<double> operands(std::size_t n, std::size_t salt) {
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = fill(i, salt);
    return v;
}

/// Wall-clock an invocation, returning (result, milliseconds).
template <class F>
auto timed(F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    auto r = f();
    const auto t1 = std::chrono::steady_clock::now();
    return std::pair{std::move(r), std::chrono::duration<double, std::milli>(t1 - t0).count()};
}

class Verdict {
public:
    void require(const std::string& what, bool ok) {
        if (!ok) {
            std::printf("  FAIL %s\n", what.c_str());
            ++fails_;
        }
    }
    void near(const std::string& what, double got, double want, double tol) {
        if (std::abs(got - want) > tol + tol * std::abs(want)) {
            std::printf("  FAIL %s: got %.9g want %.9g\n", what.c_str(), got, want);
            ++fails_;
        }
    }
    int finish(const char* title) {
        std::printf("%s: %s\n", fails_ == 0 ? "PASS" : "FAIL", title);
        return fails_ == 0 ? 0 : 1;
    }

private:
    int fails_ = 0;
};

/// Print the device transfer ledger — residency made visible.
inline void print_ledger(const char* label) {
    const auto& s = cheatah::gpu::linalg::stats();
    std::printf("  [ledger] %s: %.2f MB up, %.2f MB down, %llu dispatches\n", label,
                s.bytes_uploaded / 1e6, s.bytes_downloaded / 1e6,
                static_cast<unsigned long long>(s.dispatches));
}

}  // namespace example
