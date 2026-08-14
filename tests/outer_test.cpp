// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// outer_test.cpp — device outer product through cheatah's allocating front: any pair of vector
// lengths (including 2-D row/column vector shapes), checked element-wise against the reference
// rank-1 loop, in double and float.
#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

#include "tests/harness.hpp"

#include <vector>

namespace {

namespace gl = cheatah::gpu::linalg;

template <class T>
void run_case(std::size_t n, std::size_t m, double tol) {
    const std::vector<T> a = harness::sequence<T>(n, 3);
    const std::vector<T> b = harness::sequence<T>(m, 4);
    gl::device_array<T> da = gl::device_array<T>::from_host({n}, a.data());
    gl::device_array<T> db = gl::device_array<T>::from_host({m}, b.data());

    gl::device_array<T> dc = cheatah::linalg::outer(da, db);

    std::vector<T> C(n * m);
    dc.to_host(C.data());
    ASSERT_TRUE(dc.ndim() == 2 && dc.shape()[0] == n && dc.shape()[1] == m) << "result shape";
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j)
            EXPECT_NEAR_REL(C[i * m + j],
                            static_cast<double>(a[i]) * static_cast<double>(b[j]), tol)
                << "outer[" << i * m + j << "]";
}

TEST(Outer, SmallF64) { run_case<double>(3, 5, 1e-12); }
// Ragged 16x16 tiles.
TEST(Outer, RaggedTilesF64) { run_case<double>(100, 33, 1e-12); }
TEST(Outer, RaggedTilesF32) { run_case<float>(100, 33, 1e-5); }

// A 2-D column vector (n x 1) flattens like numpy's outer.
TEST(Outer, ColumnTimesRow2D) {
    const std::vector<double> col = {1.0, 2.0, 3.0};
    const std::vector<double> row = {4.0, 5.0};
    gl::device_array<double> dcol = gl::device_array<double>::from_host({3, 1}, col.data());
    gl::device_array<double> drow = gl::device_array<double>::from_host({2}, row.data());
    gl::device_array<double> dc = cheatah::linalg::outer(dcol, drow);
    std::vector<double> C(6);
    dc.to_host(C.data());
    const double want[6] = {4, 5, 8, 10, 12, 15};
    for (std::size_t i = 0; i < 6; ++i)
        EXPECT_NEAR_REL(C[i], want[i], 1e-12) << "outer(col,row)[" << i << "]";
}

}  // namespace
