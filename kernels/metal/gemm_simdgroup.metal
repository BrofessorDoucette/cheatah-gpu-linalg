// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// gemm_simdgroup.metal — the APPLE tensor path: simdgroup_matrix (Metal's cooperative-matrix
// analogue) GEMM, mirroring gemm_coop's 2x8-tiles-per-simdgroup tiling.
//
// STATUS: STRUCTURAL SKELETON, PENDING FIRST APPLE MACHINE. This file is NOT wired into the
// build: slangc cannot lower CoopMat to Metal (the coop kernels are spvonly), so the Apple
// tensor path is native MSL. There is no Metal compiler in the Linux dev environment — this
// kernel has never been compiled or measured, and NO performance claim exists for it. On the
// first Mac: compile with `xcrun -sdk macosx metal`, register as "gemm_simdgroup_f32" in the
// Metal context behind a simdgroup_ok() gate (mirror coop_ok()), port the BM_matmul_f16coop
// bench, and tune tile counts by measurement (the NVIDIA lesson: fragment-load latency, not
// MulAdd rate, will likely bind first — start from global loads, not threadgroup staging).
//
// simdgroup_matrix is 8x8 (vs KHR's 16x16): the 2x8 tiling here covers a 16x64 strip per
// simdgroup, 4 simdgroups per 64x64 block. dims = {M, N, K}; exact tiles only.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void gemm_simdgroup_f32(
    device const half*  A     [[buffer(0)]],
    device const half*  B     [[buffer(1)]],
    device float*       C     [[buffer(2)]],
    device const uint*  dims  [[buffer(3)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const uint N = dims[1], K = dims[2];
    const uint row0 = gid.y * 64 + simd * 16;   // this simdgroup's 16-row strip (2 A tiles)
    const uint col0 = gid.x * 64;               // 8 B tiles of 8 columns

    simdgroup_float8x8 acc[2][8];
    for (uint r = 0; r < 2; ++r)
        for (uint j = 0; j < 8; ++j) acc[r][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k = 0; k < K; k += 8) {
        simdgroup_half8x8 a0, a1;
        simdgroup_load(a0, A + row0 * K + k, K);
        simdgroup_load(a1, A + (row0 + 8) * K + k, K);
        for (uint j = 0; j < 8; ++j) {
            simdgroup_half8x8 b;
            simdgroup_load(b, B + k * N + col0 + j * 8, N);
            simdgroup_multiply_accumulate(acc[0][j], a0, b, acc[0][j]);
            simdgroup_multiply_accumulate(acc[1][j], a1, b, acc[1][j]);
        }
    }
    for (uint r = 0; r < 2; ++r)
        for (uint j = 0; j < 8; ++j)
            simdgroup_store(acc[r][j], C + (row0 + r * 8) * N + col0 + j * 8, N);
}
