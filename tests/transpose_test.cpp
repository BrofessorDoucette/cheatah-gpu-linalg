// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// transpose_test.cpp — device conj_transpose (a plain transpose for real elements) through
// cheatah's allocating front, checked element-wise, in double and float, on square and ragged
// rectangular shapes.
#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

#include "tests/harness.hpp"

#include <vector>

namespace {

namespace gl = cheatah::gpu::linalg;

template <class T>
void run_case(std::size_t r, std::size_t c, double tol) {
    const std::vector<T> A = harness::sequence<T>(r * c, 5);
    gl::device_array<T> da = gl::device_array<T>::from_host({r, c}, A.data());

    gl::device_array<T> dt = cheatah::linalg::conj_transpose(da);

    std::vector<T> Tr(c * r);
    dt.to_host(Tr.data());
    ASSERT_TRUE(dt.ndim() == 2 && dt.shape()[0] == c && dt.shape()[1] == r) << "result shape";
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j)
            EXPECT_NEAR_REL(Tr[j * r + i], A[i * c + j], tol) << "transpose[" << j * r + i << "]";
}

TEST(Transpose, SquareF64) { run_case<double>(4, 4, 1e-12); }
// Ragged 16x16 tiles.
TEST(Transpose, RaggedTilesF64) { run_case<double>(33, 17, 1e-12); }
TEST(Transpose, RaggedTilesF32) { run_case<float>(33, 17, 1e-6); }
// Degenerate row.
TEST(Transpose, DegenerateRowF32) { run_case<float>(1, 7, 1e-6); }

}  // namespace
