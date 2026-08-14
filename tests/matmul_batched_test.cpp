// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// matmul_batched_test.cpp — batched device GEMM through cheatah's front: [B,M,K] @ [B,K,N] in
// ONE z-batched dispatch, every slice checked against an in-test reference loop; B = 1 must agree
// with the 2-D path exactly; ragged tile shapes; both element types.
#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

#include "tests/harness.hpp"

#include <vector>

namespace {

namespace gl = cheatah::gpu::linalg;

template <class T>
void run_case(std::size_t B, std::size_t M, std::size_t K, std::size_t N, double tol) {
    const std::vector<T> A = harness::sequence<T>(B * M * K, 40);
    const std::vector<T> Bv = harness::sequence<T>(B * K * N, 41);
    gl::device_array<T> da = gl::device_array<T>::from_host({B, M, K}, A.data());
    gl::device_array<T> db = gl::device_array<T>::from_host({B, K, N}, Bv.data());

    gl::device_array<T> dc = cheatah::linalg::matmul(da, db);

    std::vector<T> C(B * M * N);
    dc.to_host(C.data());
    ASSERT_TRUE(dc.ndim() == 3 && dc.shape()[0] == B && dc.shape()[1] == M &&
                dc.shape()[2] == N)
        << "batched shape";
    for (std::size_t z = 0; z < B; ++z)
        for (std::size_t i = 0; i < M; ++i)
            for (std::size_t j = 0; j < N; ++j) {
                double want = 0.0;
                for (std::size_t k = 0; k < K; ++k)
                    want += double(A[z * M * K + i * K + k]) * double(Bv[z * K * N + k * N + j]);
                EXPECT_NEAR_REL(C[z * M * N + i * N + j], want, tol)
                    << "C[" << z * M * N + i * N + j << "]";
            }
}

// Small ragged slices.
TEST(MatmulBatched, SmallRaggedF64) { run_case<double>(4, 5, 3, 6, 1e-12); }
// B = 1 ≡ the 2-D kernel's pinned shape.
TEST(MatmulBatched, BatchOfOneRaggedF64) { run_case<double>(1, 33, 17, 9, 1e-12); }
// Tile-aligned batch.
TEST(MatmulBatched, TileAlignedF32) { run_case<float>(8, 16, 16, 16, 1e-4); }
// Ragged everything.
TEST(MatmulBatched, RaggedEverythingF32) { run_case<float>(3, 20, 7, 33, 1e-4); }

// B = 1 batched must agree with the 2-D path bit-for-bit (same kernel math).
TEST(MatmulBatched, BatchOfOneMatchesTwoD) {
    const std::size_t M = 12, K = 5, N = 9;
    const std::vector<double> A = harness::sequence<double>(M * K, 42);
    const std::vector<double> Bv = harness::sequence<double>(K * N, 43);
    gl::device_array<double> a3 = gl::device_array<double>::from_host({1, M, K}, A.data());
    gl::device_array<double> b3 = gl::device_array<double>::from_host({1, K, N}, Bv.data());
    gl::device_array<double> a2 = gl::device_array<double>::from_host({M, K}, A.data());
    gl::device_array<double> b2 = gl::device_array<double>::from_host({K, N}, Bv.data());
    std::vector<double> c3(M * N), c2(M * N);
    cheatah::linalg::matmul(a3, b3).to_host(c3.data());
    cheatah::linalg::matmul(a2, b2).to_host(c2.data());
    for (std::size_t i = 0; i < M * N; ++i)
        EXPECT_NEAR_REL(c3[i], c2[i], 0.0) << "B1==2D[" << i << "]";
}

}  // namespace
