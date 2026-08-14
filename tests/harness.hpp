// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
#pragma once

// harness.hpp — the shared skeleton of the per-op GoogleTest suites: deterministic operand fills
// plus the tolerance bound every expectation uses (absolute + relative term, exactly the check
// the pre-gtest Checker enforced). Every test computes its expected values with an in-test
// reference loop over the same host data it uploaded — golden math, not golden files.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace harness {

/// A deterministic, sign-varied fill value: cycles through [-1, +3) in steps of 0.25, offset by
/// `salt` so two operands never coincide. Exactly representable in float, so float and double
/// references agree on the inputs.
inline double fill(std::size_t i, std::size_t salt) {
    return 0.25 * static_cast<double>((i * 7 + salt) % 16) - 1.0;
}

/// `n` deterministic elements of `T` (see @ref fill).
template <class T>
std::vector<T> sequence(std::size_t n, std::size_t salt) {
    std::vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<T>(fill(i, salt));
    return v;
}

/// The tolerance bound: |got - want| must be <= tol plus a relative term for large magnitudes.
inline double bound(double want, double tol) { return tol + tol * std::abs(want); }

}  // namespace harness

/// EXPECT_NEAR with harness::bound's absolute-plus-relative tolerance (the Checker::near check).
#define EXPECT_NEAR_REL(got, want, tol)                                     \
    EXPECT_NEAR(static_cast<double>(got), static_cast<double>(want),        \
                ::harness::bound(static_cast<double>(want), (tol)))
