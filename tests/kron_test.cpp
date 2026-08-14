// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// kron_test.cpp — device Kronecker product through cheatah's allocating front, checked
// element-wise against the reference block loop, in double and float.
#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

#include "tests/harness.hpp"

#include <vector>

namespace {

namespace gl = cheatah::gpu::linalg;

template <class T>
void run_case(std::size_t ar, std::size_t ac, std::size_t br, std::size_t bc, double tol) {
    const std::vector<T> A = harness::sequence<T>(ar * ac, 6);
    const std::vector<T> B = harness::sequence<T>(br * bc, 7);
    gl::device_array<T> da = gl::device_array<T>::from_host({ar, ac}, A.data());
    gl::device_array<T> db = gl::device_array<T>::from_host({br, bc}, B.data());

    gl::device_array<T> dk = cheatah::linalg::kron(da, db);

    const std::size_t rows = ar * br, cols = ac * bc;
    std::vector<T> K(rows * cols);
    dk.to_host(K.data());
    ASSERT_TRUE(dk.ndim() == 2 && dk.shape()[0] == rows && dk.shape()[1] == cols)
        << "result shape";
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j) {
            const double want = static_cast<double>(A[(i / br) * ac + (j / bc)]) *
                                static_cast<double>(B[(i % br) * bc + (j % bc)]);
            EXPECT_NEAR_REL(K[i * cols + j], want, tol) << "kron[" << i * cols + j << "]";
        }
}

// The classic 4x4.
TEST(Kron, Classic4x4F64) { run_case<double>(2, 2, 2, 2, 1e-12); }
// Ragged output tiles (21x10).
TEST(Kron, RaggedBlocksF64) { run_case<double>(3, 5, 7, 2, 1e-12); }
TEST(Kron, RaggedBlocksF32) { run_case<float>(3, 5, 7, 2, 1e-5); }

}  // namespace
