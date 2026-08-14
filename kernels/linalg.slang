// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// linalg.slang — THE kernel source of cheatah-gpu-linalg: every device routine's shader, written
// once in Slang and compiled per backend at build time (CMakeLists.txt):
//   - Vulkan: slangc -target spirv, one .spv per entry point (entry renamed "main" per module);
//   - Metal:  slangc -target metal, one .metal carrying all f32 entries (Apple compiles it at
//             runtime; the off-Apple software-emulated Metal device ignores shader source and runs
//             the registered C++ stand-ins in kernels.hpp instead).
//
// Compiled twice per target: -DELEM=float -DSUFFIX=_f32 and -DELEM=double -DSUFFIX=_f64 (Vulkan
// only — Metal has no double), so every entry exists per element type: gemm_f32, gemm_f64, …
//
// ABI (shared with kernels.hpp + the contexts — keep in sync):
//   - Data buffers are row-major, contiguous, offset 0 — exactly device_array<T>'s layout.
//   - The LAST buffer of every kernel is a uint dims buffer, spelled out per kernel below.
//   - 1-D entries use [numthreads(256,1,1)] (kernels.hpp: kLocal1d); 2-D entries
//     [numthreads(16,16,1)] (kLocal2d). Every entry bounds-checks against dims, so a rounded-up
//     dispatch is safe.
//   - Reductions are DETERMINISTIC partial sums: P (= dims-declared, at most 256) threads each
//     stride the input and write partial[g]; the routine sums the P partials on the host. No
//     atomics, no execution-order dependence — the same bits every run.

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
#define K(name) CAT(name, SUFFIX)

// Elementwise operator selection: the ew_binary / ews_binary / ew_unary kernel skeletons below are
// compiled once per operator with -DEW_KIND_<OP> (+ -DEW_NAME=<entry base>), mapped here to the
// actual expression. Keeps the CMake command to plain object-like -D defines.
// COMPLEX mode (-DCOMPLEX): ELEM is float2/double2 (re, im); products go through cmul and the
// conjugating kernels through cconj. The additions in every accumulation are componentwise —
// exactly complex addition — so only multiplication needs the macro seam: MUL(a,b) is cmul under
// COMPLEX and plain a*b otherwise. std::complex<T> is layout-identical to T2 (array-of-two), so
// the C++ side binds the same buffers unchanged.
#ifdef COMPLEX
ELEM cmul(ELEM a, ELEM b) { return ELEM(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
ELEM cconj(ELEM a) { return ELEM(a.x, -a.y); }
#  define MUL(a, b) cmul(a, b)
#  define CONJ(a) cconj(a)
#  define ZERO ELEM(0, 0)
#else
#  define MUL(a, b) ((a) * (b))
#  define CONJ(a) (a)
#  define ZERO ELEM(0)
#endif

#if defined(EW_KIND_ADD)
#  define EW_OP(x, y) ((x) + (y))
#elif defined(EW_KIND_SUB)
#  define EW_OP(x, y) ((x) - (y))
#elif defined(EW_KIND_MUL)
#  define EW_OP(x, y) ((x) * (y))
#elif defined(EW_KIND_DIV)
#  define EW_OP(x, y) ((x) / (y))
#endif
#if defined(EW_KIND_NEG)
#  define EW_FN(x) (-(x))
#elif defined(EW_KIND_ABS)
#  define EW_FN(x) (abs(x))
#elif defined(EW_KIND_SQRT)
#  define EW_FN(x) (sqrt(x))
#elif defined(EW_KIND_EXP)
#  define EW_FN(x) (exp(x))
#elif defined(EW_KIND_LOG)
#  define EW_FN(x) (log(x))
#endif

#ifdef KERNEL_gemm_edge
// GEMM (EDGE path): C[M x N] = A[M x K] * B[K x N] for ANY shape, register-tiled: each workgroup
// computes a 64x64 block of
// C through groupshared 64x16 A / 16x64 B tiles, and each of its 16x16 threads accumulates a 4x4
// register microtile — 16 FMAs per 8 shared-memory loads instead of the naive 1:2, which is what
// lifts a scalar-core Vulkan kernel into the TFLOPS range. Shared arrays are padded (+1) against
// bank conflicts. Edge blocks are guarded, so any M/N/K works — including the ragged shapes the
// tests pin. dims = {M, N, K}.
[[vk::binding(0, 0)]] StructuredBuffer<ELEM>   gemm_A;
[[vk::binding(1, 0)]] StructuredBuffer<ELEM>   gemm_B;
[[vk::binding(2, 0)]] RWStructuredBuffer<ELEM> gemm_C;
[[vk::binding(3, 0)]] StructuredBuffer<uint>   gemm_dims;

groupshared ELEM gemm_As[64][17];   // [row-in-block][k] (+1 pad: stride 17 kills bank conflicts)
groupshared ELEM gemm_Bs[16][65];   // [k][col-in-block]

[shader("compute")]
[numthreads(16, 16, 1)]
void K(gemm_edge)(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    const uint M = gemm_dims[0], N = gemm_dims[1], Kd = gemm_dims[2];
    const uint block_row = gid.y * 64, block_col = gid.x * 64;
    const uint tx = ltid.x, ty = ltid.y;             // 16x16 threads
    const uint lin = ty * 16 + tx;                   // 0..255, for cooperative tile loads
    ELEM acc[4][4];
    [unroll] for (uint i = 0; i < 4; ++i)
        [unroll] for (uint j = 0; j < 4; ++j) acc[i][j] = ZERO;

    const uint tiles = (Kd + 15) / 16;
    for (uint t = 0; t < tiles; ++t) {
        // Cooperative loads: 256 threads fill A(64x16) and B(16x64) — 4 elements each.
        [unroll]
        for (uint p = 0; p < 4; ++p) {
            const uint ai = lin + p * 256;           // 0..1023 over the 64x16 A tile
            const uint ar = ai / 16, ak = ai % 16;
            const uint garow = block_row + ar, gak = t * 16 + ak;
            gemm_As[ar][ak] = (garow < M && gak < Kd) ? gemm_A[garow * Kd + gak] : ZERO;
            const uint bi = lin + p * 256;           // 0..1023 over the 16x64 B tile
            const uint bk = bi / 64, bc = bi % 64;
            const uint gbk = t * 16 + bk, gbcol = block_col + bc;
            gemm_Bs[bk][bc] = (gbk < Kd && gbcol < N) ? gemm_B[gbk * N + gbcol] : ZERO;
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint k = 0; k < 16; ++k) {
            ELEM a_frag[4];
            ELEM b_frag[4];
            [unroll] for (uint i = 0; i < 4; ++i) a_frag[i] = gemm_As[ty * 4 + i][k];
            [unroll] for (uint j = 0; j < 4; ++j) b_frag[j] = gemm_Bs[k][tx * 4 + j];
            [unroll] for (uint i = 0; i < 4; ++i)
                [unroll] for (uint j = 0; j < 4; ++j) acc[i][j] += MUL(a_frag[i], b_frag[j]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        const uint row = block_row + ty * 4 + i;
        if (row >= M) continue;
        [unroll]
        for (uint j = 0; j < 4; ++j) {
            const uint col = block_col + tx * 4 + j;
            if (col < N) gemm_C[row * N + col] = acc[i][j];
        }
    }
}
#endif  // KERNEL_gemm_edge

#ifdef KERNEL_outer
// Outer product: out[n x m] = a_i * b_j, one thread per output element. dims = {n, m}.
[[vk::binding(0, 0)]] StructuredBuffer<ELEM>   outer_a;
[[vk::binding(1, 0)]] StructuredBuffer<ELEM>   outer_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<ELEM> outer_out;
[[vk::binding(3, 0)]] StructuredBuffer<uint>   outer_dims;

[shader("compute")]
[numthreads(16, 16, 1)]
void K(outer)(uint3 tid : SV_DispatchThreadID) {
    const uint n = outer_dims[0], m = outer_dims[1];
    const uint i = tid.y, j = tid.x;
    if (i < n && j < m) outer_out[i * m + j] = MUL(outer_a[i], outer_b[j]);
}
#endif  // KERNEL_outer

#ifdef KERNEL_transpose
// Transpose: out[c x r] = in[r x c] transposed (the real-element conj_transpose — conjugation is
// the identity for float/double). dims = {r, c}. One thread per INPUT element.
[[vk::binding(0, 0)]] StructuredBuffer<ELEM>   transpose_in;
[[vk::binding(1, 0)]] RWStructuredBuffer<ELEM> transpose_out;
[[vk::binding(2, 0)]] StructuredBuffer<uint>   transpose_dims;

[shader("compute")]
[numthreads(16, 16, 1)]
void K(transpose)(uint3 tid : SV_DispatchThreadID) {
    const uint r = transpose_dims[0], c = transpose_dims[1];
    const uint i = tid.y, j = tid.x;
    if (i < r && j < c) transpose_out[j * r + i] = CONJ(transpose_in[i * c + j]);
}
#endif  // KERNEL_transpose

#ifdef KERNEL_kron
// Kronecker product: K[(ar*br) x (ac*bc)] = A ⊗ B, one thread per output element.
// dims = {ar, ac, br, bc}.
[[vk::binding(0, 0)]] StructuredBuffer<ELEM>   kron_A;
[[vk::binding(1, 0)]] StructuredBuffer<ELEM>   kron_B;
[[vk::binding(2, 0)]] RWStructuredBuffer<ELEM> kron_K;
[[vk::binding(3, 0)]] StructuredBuffer<uint>   kron_dims;

[shader("compute")]
[numthreads(16, 16, 1)]
void K(kron)(uint3 tid : SV_DispatchThreadID) {
    const uint ar = kron_dims[0], ac = kron_dims[1], br = kron_dims[2], bc = kron_dims[3];
    const uint rows = ar * br, cols = ac * bc;
    const uint i = tid.y, j = tid.x;
    if (i >= rows || j >= cols) return;
    const uint ai = i / br, bi = i % br;
    const uint aj = j / bc, bj = j % bc;
    kron_K[i * cols + j] = MUL(kron_A[ai * ac + aj], kron_B[bi * bc + bj]);
}
#endif  // KERNEL_kron

#ifdef CONJ_FIRST
#  define DOT_LHS(a) CONJ(a)
#else
#  define DOT_LHS(a) (a)
#endif
// The conjugating variant (complex vdot) is the same skeleton compiled with -DCONJ_FIRST and its
// own entry base via -DDOT_NAME=vdot_partial.
#ifndef DOT_NAME
#  define DOT_NAME dot_partial
#endif

#ifdef KERNEL_dot_partial
// Dot-product partial sums: thread g sums a[k]*b[k] for k = g, g+P, g+2P, … and writes
// partial[g]; the routine adds the P partials on the host (deterministic — no atomics).
// dims = {n, P}. Serves dot, vdot and inner (identical for real elements).
[[vk::binding(0, 0)]] StructuredBuffer<ELEM>   dot_a;
[[vk::binding(1, 0)]] StructuredBuffer<ELEM>   dot_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<ELEM> dot_partial_out;
[[vk::binding(3, 0)]] StructuredBuffer<uint>   dot_dims;

[shader("compute")]
[numthreads(256, 1, 1)]
void K(DOT_NAME)(uint3 tid : SV_DispatchThreadID) {
    const uint n = dot_dims[0], P = dot_dims[1];
    const uint g = tid.x;
    if (g >= P) return;
    ELEM acc = ZERO;
    for (uint k = g; k < n; k += P) acc += MUL(DOT_LHS(dot_a[k]), dot_b[k]);
    dot_partial_out[g] = acc;
}
#endif  // KERNEL_dot_partial

#ifdef KERNEL_trace_partial
// Trace partial sums over the diagonal of a contiguous r x c matrix: element (i, i) sits at
// i * (c + 1). dims = {m, c, P} with m = min(r, c). Same deterministic partial-sum contract.
[[vk::binding(0, 0)]] StructuredBuffer<ELEM>   trace_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<ELEM> trace_partial_out;
[[vk::binding(2, 0)]] StructuredBuffer<uint>   trace_dims;

[shader("compute")]
[numthreads(256, 1, 1)]
void K(trace_partial)(uint3 tid : SV_DispatchThreadID) {
    const uint m = trace_dims[0], c = trace_dims[1], P = trace_dims[2];
    const uint g = tid.x;
    if (g >= P) return;
    ELEM acc = ZERO;
    for (uint k = g; k < m; k += P) acc += trace_a[k * (c + 1)];
    trace_partial_out[g] = acc;
}
#endif  // KERNEL_trace_partial

#ifdef KERNEL_ew_binary
// Elementwise binary ops over contiguous same-shape arrays: out[i] = a[i] OP b[i], one thread per
// element. ONE kernel per operator, selected at compile time (-DEW_OP=...): compiled as
// ew_add / ew_sub / ew_mul / ew_div. dims = {n}. out may alias a or b (no cross-lane reads).
[[vk::binding(0, 0)]] StructuredBuffer<vector<ELEM, 4>>   ew_a;
[[vk::binding(1, 0)]] StructuredBuffer<vector<ELEM, 4>>   ew_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<vector<ELEM, 4>> ew_out;
[[vk::binding(3, 0)]] StructuredBuffer<uint>              ew_dims;

[shader("compute")]
[numthreads(256, 1, 1)]
void K(EW_NAME)(uint3 tid : SV_DispatchThreadID) {
    // vec4 lanes; the tail quad may touch [n, n+3) — safe: the pool's 256-byte size classes
    // guarantee capacity, and content past n is never read.
    const uint n4 = (ew_dims[0] + 3) >> 2;
    const uint i = tid.x;
    if (i < n4) ew_out[i] = EW_OP(ew_a[i], ew_b[i]);
}
#endif  // KERNEL_ew_binary

#ifdef KERNEL_ews_binary
// Elementwise array-scalar ops: out[i] = a[i] OP s (swap = 0) or s OP a[i] (swap = 1 — the
// reversed forms s - a and s / a). The scalar rides in a 1-element ELEM buffer (same cost class
// as the dims buffer; no push-constant machinery). dims = {n, swap}. Compiled as
// ews_add / ews_sub / ews_mul / ews_div.
[[vk::binding(0, 0)]] StructuredBuffer<vector<ELEM, 4>>   ews_a;
[[vk::binding(1, 0)]] StructuredBuffer<ELEM>              ews_s;
[[vk::binding(2, 0)]] RWStructuredBuffer<vector<ELEM, 4>> ews_out;
[[vk::binding(3, 0)]] StructuredBuffer<uint>              ews_dims;

[shader("compute")]
[numthreads(256, 1, 1)]
void K(EW_NAME)(uint3 tid : SV_DispatchThreadID) {
    const uint n4 = (ews_dims[0] + 3) >> 2, swap = ews_dims[1];
    const uint i = tid.x;
    if (i >= n4) return;
    const vector<ELEM, 4> x = ews_a[i];
    const vector<ELEM, 4> s = vector<ELEM, 4>(ews_s[0]);
    ews_out[i] = (swap != 0) ? EW_OP(s, x) : EW_OP(x, s);
}
#endif  // KERNEL_ews_binary

#ifdef KERNEL_ew_unary
// Elementwise unary ops: out[i] = F(a[i]). Compiled as ew_neg / ew_abs / ew_sqrt / ew_exp /
// ew_log (-DEW_FN=...). dims = {n}. out may alias a.
[[vk::binding(0, 0)]] StructuredBuffer<vector<ELEM, 4>>   ewu_in;
[[vk::binding(1, 0)]] RWStructuredBuffer<vector<ELEM, 4>> ewu_out;
[[vk::binding(2, 0)]] StructuredBuffer<uint>              ewu_dims;

[shader("compute")]
[numthreads(256, 1, 1)]
void K(EW_NAME)(uint3 tid : SV_DispatchThreadID) {
    const uint n4 = (ewu_dims[0] + 3) >> 2;
    const uint i = tid.x;
    if (i < n4) ewu_out[i] = EW_FN(ewu_in[i]);
}
#endif  // KERNEL_ew_unary

#ifdef KERNEL_sum_partial
// Sum partial sums — the deterministic reduction contract (see dot_partial): thread g writes
// partial[g] = sum of a[k], k = g, g+P, ...; the routine adds the P partials on the host.
// dims = {n, P}.
[[vk::binding(0, 0)]] StructuredBuffer<ELEM>   sum_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<ELEM> sum_partial_out;
[[vk::binding(2, 0)]] StructuredBuffer<uint>   sum_dims;

[shader("compute")]
[numthreads(256, 1, 1)]
void K(sum_partial)(uint3 tid : SV_DispatchThreadID) {
    const uint n = sum_dims[0], P = sum_dims[1];
    const uint g = tid.x;
    if (g >= P) return;
    ELEM acc = ZERO;
    for (uint k = g; k < n; k += P) acc += sum_a[k];
    sum_partial_out[g] = acc;
}
#endif  // KERNEL_sum_partial

#ifdef KERNEL_axpy
// Fused out = alpha * x + y — THE training-loop primitive, one dispatch instead of the two (and
// the temporary) that operator chaining would cost. alpha rides in a 1-element buffer.
// dims = {n}. out may alias x or y.
[[vk::binding(0, 0)]] StructuredBuffer<vector<ELEM, 4>>   axpy_x;
[[vk::binding(1, 0)]] StructuredBuffer<vector<ELEM, 4>>   axpy_y;
[[vk::binding(2, 0)]] StructuredBuffer<ELEM>              axpy_alpha;
[[vk::binding(3, 0)]] RWStructuredBuffer<vector<ELEM, 4>> axpy_out;
[[vk::binding(4, 0)]] StructuredBuffer<uint>              axpy_dims;

[shader("compute")]
[numthreads(256, 1, 1)]
void K(axpy)(uint3 tid : SV_DispatchThreadID) {
    const uint n4 = (axpy_dims[0] + 3) >> 2;
    const uint i = tid.x;
    if (i < n4) axpy_out[i] = vector<ELEM, 4>(axpy_alpha[0]) * axpy_x[i] + axpy_y[i];
}
#endif  // KERNEL_axpy

#ifdef KERNEL_gemm_batched
// Batched GEMM: C[z] = A[z] * B[z] for z in [0, B), each slice row-major M x N = (M x K)(K x N),
// the same 16x16 groupshared tiling as `gemm` with the THIRD grid dimension selecting the batch
// (one workgroup layer per batch — launch overhead amortized across the whole batch).
// dims = {M, N, K, B}.
[[vk::binding(0, 0)]] StructuredBuffer<ELEM>   gemmb_A;
[[vk::binding(1, 0)]] StructuredBuffer<ELEM>   gemmb_B;
[[vk::binding(2, 0)]] RWStructuredBuffer<ELEM> gemmb_C;
[[vk::binding(3, 0)]] StructuredBuffer<uint>   gemmb_dims;

groupshared ELEM gemmb_tile_a[16][16];
groupshared ELEM gemmb_tile_b[16][16];

[shader("compute")]
[numthreads(16, 16, 1)]
void K(gemm_batched)(uint3 tid : SV_DispatchThreadID, uint3 ltid : SV_GroupThreadID) {
    const uint M = gemmb_dims[0], N = gemmb_dims[1], Kd = gemmb_dims[2];
    const uint z = tid.z;
    const uint baseA = z * M * Kd, baseB = z * Kd * N, baseC = z * M * N;
    const uint row = tid.y, col = tid.x;
    const uint lr = ltid.y, lc = ltid.x;
    ELEM acc = ZERO;
    const uint tiles = (Kd + 15) / 16;
    for (uint t = 0; t < tiles; ++t) {
        const uint ka = t * 16 + lc;
        const uint kb = t * 16 + lr;
        gemmb_tile_a[lr][lc] = (row < M && ka < Kd) ? gemmb_A[baseA + row * Kd + ka] : ZERO;
        gemmb_tile_b[lr][lc] = (kb < Kd && col < N) ? gemmb_B[baseB + kb * N + col] : ZERO;
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint k = 0; k < 16; ++k) acc += MUL(gemmb_tile_a[lr][k], gemmb_tile_b[k][lc]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < M && col < N) gemmb_C[baseC + row * N + col] = acc;
}
#endif  // KERNEL_gemm_batched

#ifdef KERNEL_dot_partial2
// TWO-STAGE dot partials for large n: G workgroups x 256 threads; thread t of group g strides
// the input from g*256+t by G*256, the group tree-reduces its 256 accumulators in groupshared
// memory (fixed order — deterministic), and writes partial[g]; the routine host-sums the G
// partials. dims = {n, G}. Compiled with -DCONJ_FIRST -DDOT_NAME2=vdot_partial2 for complex vdot.
#ifndef DOT_NAME2
#  define DOT_NAME2 dot_partial2
#endif
[[vk::binding(0, 0)]] StructuredBuffer<ELEM>   dot2_a;
[[vk::binding(1, 0)]] StructuredBuffer<ELEM>   dot2_b;
[[vk::binding(2, 0)]] RWStructuredBuffer<ELEM> dot2_partial_out;
[[vk::binding(3, 0)]] StructuredBuffer<uint>   dot2_dims;

groupshared ELEM dot2_red[256];

#ifndef COMPLEX
[[vk::binding(4, 0)]] StructuredBuffer<vector<ELEM, 4>> dot2_a4;
[[vk::binding(5, 0)]] StructuredBuffer<vector<ELEM, 4>> dot2_b4;
#endif

[shader("compute")]
[numthreads(256, 1, 1)]
void K(DOT_NAME2)(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    const uint n = dot2_dims[0], G = dot2_dims[1];
    const uint t = ltid.x, g = gid.x;
    const uint stride = G * 256;
#ifndef COMPLEX
    // vec4 accumulation over the exact-quad region, folded once in the FIXED order
    // (x+y)+(z+w); the <=3 tail elements are owned by fixed threads of group 0 — deterministic.
    const uint n4 = n >> 2;
    vector<ELEM, 4> acc4 = vector<ELEM, 4>(ZERO, ZERO, ZERO, ZERO);
    for (uint k = g * 256 + t; k < n4; k += stride)
        acc4 += dot2_a4[k] * dot2_b4[k];
    ELEM acc = (acc4.x + acc4.y) + (acc4.z + acc4.w);
    const uint tail = n & 3;
    if (g == 0 && t < tail) acc += MUL(DOT_LHS(dot2_a[n4 * 4 + t]), dot2_b[n4 * 4 + t]);
#else
    // Complex elements ARE 2-vectors already — the scalar strided loop stays.
    ELEM acc = ZERO;
    for (uint k = g * 256 + t; k < n; k += stride) acc += MUL(DOT_LHS(dot2_a[k]), dot2_b[k]);
#endif
    dot2_red[t] = acc;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s) dot2_red[t] += dot2_red[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0) dot2_partial_out[g] = dot2_red[0];
}
#endif  // KERNEL_dot_partial2

#ifdef KERNEL_sum_partial2
// TWO-STAGE sum partials — the dot_partial2 contract over one input. dims = {n, G}.
[[vk::binding(0, 0)]] StructuredBuffer<ELEM>   sum2_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<ELEM> sum2_partial_out;
[[vk::binding(2, 0)]] StructuredBuffer<uint>   sum2_dims;

groupshared ELEM sum2_red[256];

[[vk::binding(3, 0)]] StructuredBuffer<vector<ELEM, 4>> sum2_a4;

[shader("compute")]
[numthreads(256, 1, 1)]
void K(sum_partial2)(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    const uint n = sum2_dims[0], G = sum2_dims[1];
    const uint t = ltid.x, g = gid.x;
    const uint stride = G * 256;
    const uint n4 = n >> 2;
    vector<ELEM, 4> acc4 = vector<ELEM, 4>(ZERO, ZERO, ZERO, ZERO);
    for (uint k = g * 256 + t; k < n4; k += stride) acc4 += sum2_a4[k];
    ELEM acc = (acc4.x + acc4.y) + (acc4.z + acc4.w);
    const uint tail = n & 3;
    if (g == 0 && t < tail) acc += sum2_a[n4 * 4 + t];
    sum2_red[t] = acc;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s) sum2_red[t] += sum2_red[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0) sum2_partial_out[g] = sum2_red[0];
}
#endif  // KERNEL_sum_partial2

#ifdef KERNEL_gemm_fast
// GEMM (FAST path): C = A*B for EXACTLY-TILED real matrices (M%128==0, N%128==0, K%8==0 — the
// routine gates on this, so the hot loop carries ZERO bounds checks). One 256-thread (16x16)
// workgroup computes a 128x128 block of C; each thread an 8x8 register microtile; K advances in
// 16-wide slabs staged through DOUBLE-BUFFERED groupshared tiles (compute slab t while loading
// slab t+1), with float4-vectorized global loads (one vec4 per thread per matrix per slab).
// 128 FMAs per 16 shared loads per k-step. dims = {M, N, K}. Real elements only (complex runs
// the edge kernel).
[[vk::binding(0, 0)]] StructuredBuffer<vector<ELEM, 4>> gemmf_A4;
[[vk::binding(1, 0)]] StructuredBuffer<vector<ELEM, 4>> gemmf_B4;
[[vk::binding(2, 0)]] RWStructuredBuffer<vector<ELEM, 4>> gemmf_C4;  // vec4 view of C (N%4==0)
[[vk::binding(3, 0)]] StructuredBuffer<uint>            gemmf_dims;

groupshared ELEM gemmf_As[2][16][129];           // [buf][k][row-in-block+PAD] (A transposed on
                                                 // store; 129 skews the 4 k-group store banks)
groupshared vector<ELEM, 4> gemmf_Bs4[2][16][32];  // [buf][k][col-quad] — vec4 stores AND reads

[shader("compute")]
[numthreads(16, 16, 1)]
void K(gemm_fast)(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    const uint N = gemmf_dims[1], Kd = gemmf_dims[2];
    const uint block_row = gid.y * 128, block_col = gid.x * 128;
    const uint tx = ltid.x, ty = ltid.y;
    const uint lin = ty * 16 + tx;                  // 0..255
    // A-slab load geometry: 128 rows x 8 k = 256 float4 (2 per row); this thread owns
    // row ar, k-quad ak (0 or 4). B-slab: 8 k x 128 n = 256 float4 (32 per k-row).
    // 16-wide K slabs: each thread stages TWO vec4 per matrix per slab (512 vec4 / 256 threads).
    const uint ar0 = lin >> 2, ak0 = (lin & 3) * 4;      // A vec4 #1: rows 0..63
    const uint ar1 = 64 + ar0;                            // A vec4 #2: rows 64..127
    const uint bk0 = lin >> 5, bn0 = (lin & 31) * 4;      // B vec4 #1: k 0..7
    const uint bk1 = 8 + bk0;                             // B vec4 #2: k 8..15
    const uint K4 = Kd >> 2, N4 = N >> 2;           // row strides in float4 units

    ELEM acc[8][8];
    [unroll] for (uint i = 0; i < 8; ++i)
        [unroll] for (uint j = 0; j < 8; ++j) acc[i][j] = ELEM(0);

    // Preload slab 0 into buffer 0.
    vector<ELEM, 4> a_reg0 = gemmf_A4[(block_row + ar0) * K4 + (ak0 >> 2)];
    vector<ELEM, 4> a_reg1 = gemmf_A4[(block_row + ar1) * K4 + (ak0 >> 2)];
    vector<ELEM, 4> b_reg0 = gemmf_B4[bk0 * N4 + ((block_col + bn0) >> 2)];
    vector<ELEM, 4> b_reg1 = gemmf_B4[bk1 * N4 + ((block_col + bn0) >> 2)];
    gemmf_As[0][ak0 + 0][ar0] = a_reg0.x;
    gemmf_As[0][ak0 + 1][ar0] = a_reg0.y;
    gemmf_As[0][ak0 + 2][ar0] = a_reg0.z;
    gemmf_As[0][ak0 + 3][ar0] = a_reg0.w;
    gemmf_As[0][ak0 + 0][ar1] = a_reg1.x;
    gemmf_As[0][ak0 + 1][ar1] = a_reg1.y;
    gemmf_As[0][ak0 + 2][ar1] = a_reg1.z;
    gemmf_As[0][ak0 + 3][ar1] = a_reg1.w;
    gemmf_Bs4[0][bk0][bn0 >> 2] = b_reg0;
    gemmf_Bs4[0][bk1][bn0 >> 2] = b_reg1;
    GroupMemoryBarrierWithGroupSync();

    const uint slabs = Kd >> 4;
    for (uint t = 0; t < slabs; ++t) {
        const uint cur = t & 1;
        // Kick off the NEXT slab's global loads into registers while computing this one.
        const bool more = (t + 1) < slabs;
        if (more) {
            const uint k0 = (t + 1) * 16;
            a_reg0 = gemmf_A4[(block_row + ar0) * K4 + ((k0 + ak0) >> 2)];
            a_reg1 = gemmf_A4[(block_row + ar1) * K4 + ((k0 + ak0) >> 2)];
            b_reg0 = gemmf_B4[(k0 + bk0) * N4 + ((block_col + bn0) >> 2)];
            b_reg1 = gemmf_B4[(k0 + bk1) * N4 + ((block_col + bn0) >> 2)];
        }
        [unroll]
        for (uint k = 0; k < 16; ++k) {
            ELEM a_frag[8];
            [unroll] for (uint i = 0; i < 8; ++i) a_frag[i] = gemmf_As[cur][k][ty * 8 + i];
            // Warp-coherent quads: threads 0..15 read quads 0..15 (then 16..31) — consecutive
            // lanes hit consecutive banks instead of the 2-stride pattern's 4-way conflicts.
            const vector<ELEM, 4> b0 = gemmf_Bs4[cur][k][tx];
            const vector<ELEM, 4> b1 = gemmf_Bs4[cur][k][16 + tx];
            [unroll] for (uint i = 0; i < 8; ++i) {
                acc[i][0] += a_frag[i] * b0.x;
                acc[i][1] += a_frag[i] * b0.y;
                acc[i][2] += a_frag[i] * b0.z;
                acc[i][3] += a_frag[i] * b0.w;
                acc[i][4] += a_frag[i] * b1.x;
                acc[i][5] += a_frag[i] * b1.y;
                acc[i][6] += a_frag[i] * b1.z;
                acc[i][7] += a_frag[i] * b1.w;
            }
        }
        if (more) {
            const uint nxt = cur ^ 1;
            gemmf_As[nxt][ak0 + 0][ar0] = a_reg0.x;
            gemmf_As[nxt][ak0 + 1][ar0] = a_reg0.y;
            gemmf_As[nxt][ak0 + 2][ar0] = a_reg0.z;
            gemmf_As[nxt][ak0 + 3][ar0] = a_reg0.w;
            gemmf_As[nxt][ak0 + 0][ar1] = a_reg1.x;
            gemmf_As[nxt][ak0 + 1][ar1] = a_reg1.y;
            gemmf_As[nxt][ak0 + 2][ar1] = a_reg1.z;
            gemmf_As[nxt][ak0 + 3][ar1] = a_reg1.w;
            gemmf_Bs4[nxt][bk0][bn0 >> 2] = b_reg0;
            gemmf_Bs4[nxt][bk1][bn0 >> 2] = b_reg1;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [unroll]
    for (uint i = 0; i < 8; ++i) {
        const uint row = block_row + ty * 8 + i;
        const uint q = (row * N + block_col) >> 2;   // C as vec4: 16 stores, not 64
        gemmf_C4[q + tx] = vector<ELEM, 4>(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        gemmf_C4[q + 16 + tx] = vector<ELEM, 4>(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
    }
}
#endif  // KERNEL_gemm_fast

#ifdef KERNEL_gemm_attr
// ATTRIBUTION PROBE (bench-only, results intentionally WRONG in some modes): the fast GEMM
// with one cost surgically removed per compile — ATTR_NOBAR (no barriers), ATTR_NOGLOBAL (no
// in-loop global traffic), ATTR_NOEPI (one-element epilogue). Timing deltas attribute the gap
// to the measured ceiling; never routed by the library.
[[vk::binding(0, 0)]] StructuredBuffer<vector<ELEM, 4>> gemma_A4;
[[vk::binding(1, 0)]] StructuredBuffer<vector<ELEM, 4>> gemma_B4;
[[vk::binding(2, 0)]] RWStructuredBuffer<vector<ELEM, 4>> gemma_C4;  // vec4 view of C (N%4==0)
[[vk::binding(3, 0)]] StructuredBuffer<uint>            gemma_dims;

groupshared ELEM gemma_As[2][16][129];           // [buf][k][row-in-block+PAD] (A transposed on
                                                 // store; 129 skews the 4 k-group store banks)
groupshared vector<ELEM, 4> gemma_Bs4[2][16][32];  // [buf][k][col-quad] — vec4 stores AND reads

[shader("compute")]
[numthreads(16, 16, 1)]
void K(ATTR_NAME)(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    const uint N = gemma_dims[1], Kd = gemma_dims[2];
    const uint block_row = gid.y * 128, block_col = gid.x * 128;
    const uint tx = ltid.x, ty = ltid.y;
    const uint lin = ty * 16 + tx;                  // 0..255
    // A-slab load geometry: 128 rows x 8 k = 256 float4 (2 per row); this thread owns
    // row ar, k-quad ak (0 or 4). B-slab: 8 k x 128 n = 256 float4 (32 per k-row).
    // 16-wide K slabs: each thread stages TWO vec4 per matrix per slab (512 vec4 / 256 threads).
    const uint ar0 = lin >> 2, ak0 = (lin & 3) * 4;      // A vec4 #1: rows 0..63
    const uint ar1 = 64 + ar0;                            // A vec4 #2: rows 64..127
    const uint bk0 = lin >> 5, bn0 = (lin & 31) * 4;      // B vec4 #1: k 0..7
    const uint bk1 = 8 + bk0;                             // B vec4 #2: k 8..15
    const uint K4 = Kd >> 2, N4 = N >> 2;           // row strides in float4 units

    ELEM acc[8][8];
    [unroll] for (uint i = 0; i < 8; ++i)
        [unroll] for (uint j = 0; j < 8; ++j) acc[i][j] = ELEM(0);

    // Preload slab 0 into buffer 0.
    vector<ELEM, 4> a_reg0 = gemma_A4[(block_row + ar0) * K4 + (ak0 >> 2)];
    vector<ELEM, 4> a_reg1 = gemma_A4[(block_row + ar1) * K4 + (ak0 >> 2)];
    vector<ELEM, 4> b_reg0 = gemma_B4[bk0 * N4 + ((block_col + bn0) >> 2)];
    vector<ELEM, 4> b_reg1 = gemma_B4[bk1 * N4 + ((block_col + bn0) >> 2)];
    gemma_As[0][ak0 + 0][ar0] = a_reg0.x;
    gemma_As[0][ak0 + 1][ar0] = a_reg0.y;
    gemma_As[0][ak0 + 2][ar0] = a_reg0.z;
    gemma_As[0][ak0 + 3][ar0] = a_reg0.w;
    gemma_As[0][ak0 + 0][ar1] = a_reg1.x;
    gemma_As[0][ak0 + 1][ar1] = a_reg1.y;
    gemma_As[0][ak0 + 2][ar1] = a_reg1.z;
    gemma_As[0][ak0 + 3][ar1] = a_reg1.w;
    gemma_Bs4[0][bk0][bn0 >> 2] = b_reg0;
    gemma_Bs4[0][bk1][bn0 >> 2] = b_reg1;
#ifndef ATTR_NOBAR
    GroupMemoryBarrierWithGroupSync();
#endif

    const uint slabs = Kd >> 4;
    for (uint t = 0; t < slabs; ++t) {
        const uint cur = t & 1;
        // Kick off the NEXT slab's global loads into registers while computing this one.
        const bool more = (t + 1) < slabs;
#ifndef ATTR_NOGLOBAL
        if (more) {
            const uint k0 = (t + 1) * 16;
            a_reg0 = gemma_A4[(block_row + ar0) * K4 + ((k0 + ak0) >> 2)];
            a_reg1 = gemma_A4[(block_row + ar1) * K4 + ((k0 + ak0) >> 2)];
            b_reg0 = gemma_B4[(k0 + bk0) * N4 + ((block_col + bn0) >> 2)];
            b_reg1 = gemma_B4[(k0 + bk1) * N4 + ((block_col + bn0) >> 2)];
        }
#endif
        [unroll]
        for (uint k = 0; k < 16; ++k) {
            ELEM a_frag[8];
            [unroll] for (uint i = 0; i < 8; ++i) a_frag[i] = gemma_As[cur][k][ty * 8 + i];
            // Warp-coherent quads: threads 0..15 read quads 0..15 (then 16..31) — consecutive
            // lanes hit consecutive banks instead of the 2-stride pattern's 4-way conflicts.
            const vector<ELEM, 4> b0 = gemma_Bs4[cur][k][tx];
            const vector<ELEM, 4> b1 = gemma_Bs4[cur][k][16 + tx];
            [unroll] for (uint i = 0; i < 8; ++i) {
                acc[i][0] += a_frag[i] * b0.x;
                acc[i][1] += a_frag[i] * b0.y;
                acc[i][2] += a_frag[i] * b0.z;
                acc[i][3] += a_frag[i] * b0.w;
                acc[i][4] += a_frag[i] * b1.x;
                acc[i][5] += a_frag[i] * b1.y;
                acc[i][6] += a_frag[i] * b1.z;
                acc[i][7] += a_frag[i] * b1.w;
            }
        }
#ifndef ATTR_NOGLOBAL
        if (more) {
            const uint nxt = cur ^ 1;
            gemma_As[nxt][ak0 + 0][ar0] = a_reg0.x;
            gemma_As[nxt][ak0 + 1][ar0] = a_reg0.y;
            gemma_As[nxt][ak0 + 2][ar0] = a_reg0.z;
            gemma_As[nxt][ak0 + 3][ar0] = a_reg0.w;
            gemma_As[nxt][ak0 + 0][ar1] = a_reg1.x;
            gemma_As[nxt][ak0 + 1][ar1] = a_reg1.y;
            gemma_As[nxt][ak0 + 2][ar1] = a_reg1.z;
            gemma_As[nxt][ak0 + 3][ar1] = a_reg1.w;
            gemma_Bs4[nxt][bk0][bn0 >> 2] = b_reg0;
            gemma_Bs4[nxt][bk1][bn0 >> 2] = b_reg1;
        }
#endif
#ifndef ATTR_NOBAR
        GroupMemoryBarrierWithGroupSync();
#endif
    }

#ifdef ATTR_NOEPI
    ELEM r = ELEM(0);
    [unroll] for (uint i = 0; i < 8; ++i)
        [unroll] for (uint j = 0; j < 8; ++j) r += acc[i][j];
    gemma_C4[((block_row + ty) * N + block_col) >> 2] = vector<ELEM, 4>(r);
#else
    [unroll]
    for (uint i = 0; i < 8; ++i) {
        const uint row = block_row + ty * 8 + i;
        const uint q = (row * N + block_col) >> 2;   // C as vec4: 16 stores, not 64
        gemma_C4[q + tx] = vector<ELEM, 4>(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        gemma_C4[q + 16 + tx] = vector<ELEM, 4>(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
    }
#endif
}
#endif  // KERNEL_gemm_attr

#ifdef KERNEL_gemm_fast64
// GEMM (FAST-64 path): the 64-ROW sibling of gemm_fast for training-shaped rectangles
// (M%64==0 but not %128 — e.g. batch-64 MLP layers routed to the slow edge kernel before).
// One 128-thread (16x8) workgroup computes a 64x128 block of C; same 8x8 microtile, BK=16
// double-buffered slabs, vec4 loads, zero bounds checks. dims = {M, N, K}. f32 only.
[[vk::binding(0, 0)]] StructuredBuffer<vector<ELEM, 4>> gemms_A4;
[[vk::binding(1, 0)]] StructuredBuffer<vector<ELEM, 4>> gemms_B4;
[[vk::binding(2, 0)]] RWStructuredBuffer<vector<ELEM, 4>> gemms_C4;  // vec4 view of C (N%4==0)
[[vk::binding(3, 0)]] StructuredBuffer<uint>            gemms_dims;

groupshared ELEM gemms_As[2][16][65];              // [buf][k][row+PAD] (65 skews store banks)
groupshared vector<ELEM, 4> gemms_Bs4[2][16][32];  // [buf][k][col-quad]

[shader("compute")]
[numthreads(16, 8, 1)]
void K(gemm_fast64)(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    const uint N = gemms_dims[1], Kd = gemms_dims[2];
    const uint block_row = gid.y * 64, block_col = gid.x * 128;
    const uint tx = ltid.x, ty = ltid.y;
    const uint lin = ty * 16 + tx;                  // 0..127
    // A slab: 64 rows x 16 k = 256 vec4 -> 2 per thread (row ar, k-quads ak0, ak0+4).
    // B slab: 16 k x 128 n = 512 vec4 -> 4 per thread (k rows bk0+{0,4,8,12}).
    const uint ar = lin >> 1, ak0 = (lin & 1) * 8;
    const uint bk0 = lin >> 5, bn0 = (lin & 31) * 4;
    const uint K4 = Kd >> 2, N4 = N >> 2;

    ELEM acc[8][8];
    [unroll] for (uint i = 0; i < 8; ++i)
        [unroll] for (uint j = 0; j < 8; ++j) acc[i][j] = ELEM(0);

    vector<ELEM, 4> a_reg0 = gemms_A4[(block_row + ar) * K4 + (ak0 >> 2)];
    vector<ELEM, 4> a_reg1 = gemms_A4[(block_row + ar) * K4 + (ak0 >> 2) + 1];
    vector<ELEM, 4> b_reg0 = gemms_B4[(bk0 + 0) * N4 + ((block_col + bn0) >> 2)];
    vector<ELEM, 4> b_reg1 = gemms_B4[(bk0 + 4) * N4 + ((block_col + bn0) >> 2)];
    vector<ELEM, 4> b_reg2 = gemms_B4[(bk0 + 8) * N4 + ((block_col + bn0) >> 2)];
    vector<ELEM, 4> b_reg3 = gemms_B4[(bk0 + 12) * N4 + ((block_col + bn0) >> 2)];
    [unroll] for (uint j = 0; j < 4; ++j) {
        gemms_As[0][ak0 + j][ar] = a_reg0[j];
        gemms_As[0][ak0 + 4 + j][ar] = a_reg1[j];
    }
    gemms_Bs4[0][bk0 + 0][bn0 >> 2] = b_reg0;
    gemms_Bs4[0][bk0 + 4][bn0 >> 2] = b_reg1;
    gemms_Bs4[0][bk0 + 8][bn0 >> 2] = b_reg2;
    gemms_Bs4[0][bk0 + 12][bn0 >> 2] = b_reg3;
    GroupMemoryBarrierWithGroupSync();

    const uint slabs = Kd >> 4;
    for (uint t = 0; t < slabs; ++t) {
        const uint cur = t & 1;
        const bool more = (t + 1) < slabs;
        if (more) {
            const uint k0 = (t + 1) * 16;
            a_reg0 = gemms_A4[(block_row + ar) * K4 + ((k0 + ak0) >> 2)];
            a_reg1 = gemms_A4[(block_row + ar) * K4 + ((k0 + ak0) >> 2) + 1];
            b_reg0 = gemms_B4[(k0 + bk0 + 0) * N4 + ((block_col + bn0) >> 2)];
            b_reg1 = gemms_B4[(k0 + bk0 + 4) * N4 + ((block_col + bn0) >> 2)];
            b_reg2 = gemms_B4[(k0 + bk0 + 8) * N4 + ((block_col + bn0) >> 2)];
            b_reg3 = gemms_B4[(k0 + bk0 + 12) * N4 + ((block_col + bn0) >> 2)];
        }
        [unroll]
        for (uint k = 0; k < 16; ++k) {
            ELEM a_frag[8];
            [unroll] for (uint i = 0; i < 8; ++i) a_frag[i] = gemms_As[cur][k][ty * 8 + i];
            const vector<ELEM, 4> b0 = gemms_Bs4[cur][k][tx];
            const vector<ELEM, 4> b1 = gemms_Bs4[cur][k][16 + tx];
            [unroll] for (uint i = 0; i < 8; ++i) {
                acc[i][0] += a_frag[i] * b0.x;
                acc[i][1] += a_frag[i] * b0.y;
                acc[i][2] += a_frag[i] * b0.z;
                acc[i][3] += a_frag[i] * b0.w;
                acc[i][4] += a_frag[i] * b1.x;
                acc[i][5] += a_frag[i] * b1.y;
                acc[i][6] += a_frag[i] * b1.z;
                acc[i][7] += a_frag[i] * b1.w;
            }
        }
        if (more) {
            const uint nxt = cur ^ 1;
            [unroll] for (uint j = 0; j < 4; ++j) {
                gemms_As[nxt][ak0 + j][ar] = a_reg0[j];
                gemms_As[nxt][ak0 + 4 + j][ar] = a_reg1[j];
            }
            gemms_Bs4[nxt][bk0 + 0][bn0 >> 2] = b_reg0;
            gemms_Bs4[nxt][bk0 + 4][bn0 >> 2] = b_reg1;
            gemms_Bs4[nxt][bk0 + 8][bn0 >> 2] = b_reg2;
            gemms_Bs4[nxt][bk0 + 12][bn0 >> 2] = b_reg3;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [unroll]
    for (uint i = 0; i < 8; ++i) {
        const uint row = block_row + ty * 8 + i;
        const uint q = (row * N + block_col) >> 2;   // C as vec4: 16 stores, not 64
        gemms_C4[q + tx] = vector<ELEM, 4>(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        gemms_C4[q + 16 + tx] = vector<ELEM, 4>(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
    }
}
#endif  // KERNEL_gemm_fast64

#ifdef KERNEL_ffma_peak
// PEAK PROBE: pure register FMA chains, zero memory traffic in the loop — measures the REAL
// achievable f32 FLOP rate through this driver at the machine's sustained clocks. 8 independent
// chains per thread (ILP), dims = {iters}. FLOP = threads * iters * 8 * 2.
[[vk::binding(0, 0)]] RWStructuredBuffer<ELEM> fp_out;
[[vk::binding(1, 0)]] StructuredBuffer<uint>   fp_dims;

[shader("compute")]
[numthreads(256, 1, 1)]
void K(ffma_peak)(uint3 tid : SV_DispatchThreadID) {
    const uint iters = fp_dims[0];
    ELEM a0 = ELEM(tid.x & 7) * ELEM(1e-8), a1 = a0 + ELEM(0.1), a2 = a0 + ELEM(0.2);
    ELEM a3 = a0 + ELEM(0.3), a4 = a0 + ELEM(0.4), a5 = a0 + ELEM(0.5);
    ELEM a6 = a0 + ELEM(0.6), a7 = a0 + ELEM(0.7);
    const ELEM m = ELEM(0.999999), b = ELEM(1e-7);
    for (uint t = 0; t < iters; ++t) {
        a0 = a0 * m + b; a1 = a1 * m + b; a2 = a2 * m + b; a3 = a3 * m + b;
        a4 = a4 * m + b; a5 = a5 * m + b; a6 = a6 * m + b; a7 = a7 * m + b;
    }
    fp_out[tid.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
#endif  // KERNEL_ffma_peak

#ifdef KERNEL_lds_peak
// PEAK PROBE: groupshared vec4 load throughput (the currency the GEMM inner loop spends).
// dims = {iters}. Bytes = threads * iters * 4 * 16.
[[vk::binding(0, 0)]] RWStructuredBuffer<vector<ELEM, 4>> lp_out;
[[vk::binding(1, 0)]] StructuredBuffer<uint>              lp_dims;

groupshared vector<ELEM, 4> lp_s[512];

[shader("compute")]
[numthreads(256, 1, 1)]
void K(lds_peak)(uint3 tid : SV_DispatchThreadID, uint3 ltid : SV_GroupThreadID) {
    const uint iters = lp_dims[0];
    lp_s[ltid.x] = vector<ELEM, 4>(ELEM(ltid.x));
    lp_s[256 + ltid.x] = vector<ELEM, 4>(ELEM(ltid.x) + ELEM(0.5));
    GroupMemoryBarrierWithGroupSync();
    vector<ELEM, 4> acc = vector<ELEM, 4>(ELEM(0));
    uint i = ltid.x;
    for (uint t = 0; t < iters; ++t) {
        acc += lp_s[i & 511];
        acc += lp_s[(i + 128) & 511];
        acc += lp_s[(i + 256) & 511];
        acc += lp_s[(i + 384) & 511];
        i += 33;
    }
    lp_out[tid.x] = acc;
}
#endif  // KERNEL_lds_peak

#ifdef KERNEL_gemm_coop
// GEMM (TENSOR-CORE path, opt-in): C(f32) = A(f16) * B(f16) via KHR cooperative matrices —
// each 32-thread subgroup drives the matrix units on 16x16x16 fragments. One 128-thread
// workgroup computes a 128x128 C block: subgroup w owns the 32-row strip w*32 as TWO A tiles
// x EIGHT B tiles = SIXTEEN accumulators, so a k-step is 10 fragment loads for 16 MulAdds (v1
// was 5-for-4, v2 6-for-8 — raising compute density per load is what moves the number).
// Fragments load straight from global: measured FASTER than groupshared staging on this driver.
// Requires VK_KHR_cooperative_matrix + shaderFloat16 + vulkanMemoryModel (context-gated).
// Exactly-tiled shapes only: M%128==0, N%128==0, K%16==0. dims = {M, N, K}.
using namespace linalg;

[[vk::binding(0, 0)]] StructuredBuffer<float16_t> gemmc_A;
[[vk::binding(1, 0)]] StructuredBuffer<float16_t> gemmc_B;
[[vk::binding(2, 0)]] RWStructuredBuffer<float>   gemmc_C;
[[vk::binding(3, 0)]] StructuredBuffer<uint>      gemmc_dims;

[shader("compute")]
[numthreads(128, 1, 1)]
void K(gemm_coop)(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    const uint N = gemmc_dims[1], Kd = gemmc_dims[2];
    const uint warp = ltid.x / 32;                    // subgroup index 0..3
    const uint row0 = gid.y * 128 + warp * 32;        // this subgroup's 32-row strip
    const uint col0 = gid.x * 128;

    typealias MatA = CoopMat<float16_t, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
    typealias MatB = CoopMat<float16_t, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
    typealias MatC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

    MatC c00 = MatC(0.0);
    MatC c01 = MatC(0.0);
    MatC c02 = MatC(0.0);
    MatC c03 = MatC(0.0);
    MatC c04 = MatC(0.0);
    MatC c05 = MatC(0.0);
    MatC c06 = MatC(0.0);
    MatC c07 = MatC(0.0);
    MatC c10 = MatC(0.0);
    MatC c11 = MatC(0.0);
    MatC c12 = MatC(0.0);
    MatC c13 = MatC(0.0);
    MatC c14 = MatC(0.0);
    MatC c15 = MatC(0.0);
    MatC c16 = MatC(0.0);
    MatC c17 = MatC(0.0);

    // A-fragment software pipeline: next k-step's A tiles load while this step's MulAdds run.
    MatA a0 = MatA.Load<CoopMatMatrixLayout.RowMajor>(gemmc_A, row0 * Kd, Kd);
    MatA a1 = MatA.Load<CoopMatMatrixLayout.RowMajor>(gemmc_A, (row0 + 16) * Kd, Kd);
    MatB b0 = MatB.Load<CoopMatMatrixLayout.RowMajor>(gemmc_B, col0 + 0, N);
    MatB b1 = MatB.Load<CoopMatMatrixLayout.RowMajor>(gemmc_B, col0 + 16, N);
    for (uint k = 0; k < Kd; k += 16) {
        MatA a0n = a0;
        MatA a1n = a1;
        MatB b0n = b0;
        MatB b1n = b1;
        if (k + 16 < Kd) {
            a0n = MatA.Load<CoopMatMatrixLayout.RowMajor>(gemmc_A, row0 * Kd + k + 16, Kd);
            a1n = MatA.Load<CoopMatMatrixLayout.RowMajor>(gemmc_A, (row0 + 16) * Kd + k + 16, Kd);
            b0n = MatB.Load<CoopMatMatrixLayout.RowMajor>(gemmc_B, (k + 16) * N + col0 + 0, N);
            b1n = MatB.Load<CoopMatMatrixLayout.RowMajor>(gemmc_B, (k + 16) * N + col0 + 16, N);
        }
        let b4 = MatB.Load<CoopMatMatrixLayout.RowMajor>(gemmc_B, k * N + col0 + 64, N);
        let b5 = MatB.Load<CoopMatMatrixLayout.RowMajor>(gemmc_B, k * N + col0 + 80, N);
        let b6 = MatB.Load<CoopMatMatrixLayout.RowMajor>(gemmc_B, k * N + col0 + 96, N);
        let b7 = MatB.Load<CoopMatMatrixLayout.RowMajor>(gemmc_B, k * N + col0 + 112, N);
        let b2 = MatB.Load<CoopMatMatrixLayout.RowMajor>(gemmc_B, k * N + col0 + 32, N);
        let b3 = MatB.Load<CoopMatMatrixLayout.RowMajor>(gemmc_B, k * N + col0 + 48, N);
        c00 = coopMatMulAdd<float, false>(a0, b0, c00);
        c01 = coopMatMulAdd<float, false>(a0, b1, c01);
        c02 = coopMatMulAdd<float, false>(a0, b2, c02);
        c03 = coopMatMulAdd<float, false>(a0, b3, c03);
        c04 = coopMatMulAdd<float, false>(a0, b4, c04);
        c05 = coopMatMulAdd<float, false>(a0, b5, c05);
        c06 = coopMatMulAdd<float, false>(a0, b6, c06);
        c07 = coopMatMulAdd<float, false>(a0, b7, c07);
        c10 = coopMatMulAdd<float, false>(a1, b0, c10);
        c11 = coopMatMulAdd<float, false>(a1, b1, c11);
        c12 = coopMatMulAdd<float, false>(a1, b2, c12);
        c13 = coopMatMulAdd<float, false>(a1, b3, c13);
        c14 = coopMatMulAdd<float, false>(a1, b4, c14);
        c15 = coopMatMulAdd<float, false>(a1, b5, c15);
        c16 = coopMatMulAdd<float, false>(a1, b6, c16);
        c17 = coopMatMulAdd<float, false>(a1, b7, c17);
        a0 = a0n;
        a1 = a1n;
        b0 = b0n;
        b1 = b1n;
    }
    c00.Store<CoopMatMatrixLayout.RowMajor>(gemmc_C, row0 * N + col0, N);
    c01.Store<CoopMatMatrixLayout.RowMajor>(gemmc_C, row0 * N + col0 + 16, N);
    c02.Store<CoopMatMatrixLayout.RowMajor>(gemmc_C, row0 * N + col0 + 32, N);
    c03.Store<CoopMatMatrixLayout.RowMajor>(gemmc_C, row0 * N + col0 + 48, N);
    c04.Store<CoopMatMatrixLayout.RowMajor>(gemmc_C, row0 * N + col0 + 64, N);
    c05.Store<CoopMatMatrixLayout.RowMajor>(gemmc_C, row0 * N + col0 + 80, N);
    c06.Store<CoopMatMatrixLayout.RowMajor>(gemmc_C, row0 * N + col0 + 96, N);
    c07.Store<CoopMatMatrixLayout.RowMajor>(gemmc_C, row0 * N + col0 + 112, N);
    c10.Store<CoopMatMatrixLayout.RowMajor>(gemmc_C, (row0 + 16) * N + col0, N);
    c11.Store<CoopMatMatrixLayout.RowMajor>(gemmc_C, (row0 + 16) * N + col0 + 16, N);
    c12.Store<CoopMatMatrixLayout.RowMajor>(gemmc_C, (row0 + 16) * N + col0 + 32, N);
    c13.Store<CoopMatMatrixLayout.RowMajor>(gemmc_C, (row0 + 16) * N + col0 + 48, N);
    c14.Store<CoopMatMatrixLayout.RowMajor>(gemmc_C, (row0 + 16) * N + col0 + 64, N);
    c15.Store<CoopMatMatrixLayout.RowMajor>(gemmc_C, (row0 + 16) * N + col0 + 80, N);
    c16.Store<CoopMatMatrixLayout.RowMajor>(gemmc_C, (row0 + 16) * N + col0 + 96, N);
    c17.Store<CoopMatMatrixLayout.RowMajor>(gemmc_C, (row0 + 16) * N + col0 + 112, N);
}
#endif  // KERNEL_gemm_coop

#ifdef KERNEL_gemm_coop_f16acc
// GEMM (TENSOR-CORE, f16 ACCUMULATE, opt-in): 2x the f32-acc MulAdd rate on GeForce Ampere;
// inference-class numerics (f16 accumulation drifts with K — measured >5% at K=4096). Same
// tiling contract as gemm_coop. C is f16. —
// each 32-thread subgroup drives the matrix units on 16x16x16 fragments. One 128-thread
// workgroup computes a 128x128 C block: subgroup w owns the 32-row strip w*32 as TWO A tiles
// x EIGHT B tiles = SIXTEEN accumulators, so a k-step is 10 fragment loads for 16 MulAdds (v1
// was 5-for-4, v2 6-for-8 — raising compute density per load is what moves the number).
// Fragments load straight from global: measured FASTER than groupshared staging on this driver.
// Requires VK_KHR_cooperative_matrix + shaderFloat16 + vulkanMemoryModel (context-gated).
// Exactly-tiled shapes only: M%128==0, N%128==0, K%16==0. dims = {M, N, K}.
using namespace linalg;

[[vk::binding(0, 0)]] StructuredBuffer<float16_t> gemmh_A;
[[vk::binding(1, 0)]] StructuredBuffer<float16_t> gemmh_B;
[[vk::binding(2, 0)]] RWStructuredBuffer<float16_t> gemmh_C;
[[vk::binding(3, 0)]] StructuredBuffer<uint>      gemmh_dims;

[shader("compute")]
[numthreads(128, 1, 1)]
void K(gemm_coop_f16acc)(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    const uint N = gemmh_dims[1], Kd = gemmh_dims[2];
    const uint warp = ltid.x / 32;                    // subgroup index 0..3
    const uint row0 = gid.y * 128 + warp * 32;        // this subgroup's 32-row strip
    const uint col0 = gid.x * 128;

    typealias MatA = CoopMat<float16_t, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
    typealias MatB = CoopMat<float16_t, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
    typealias MatC = CoopMat<float16_t, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

    MatC c00 = MatC(float16_t(0));
    MatC c01 = MatC(float16_t(0));
    MatC c02 = MatC(float16_t(0));
    MatC c03 = MatC(float16_t(0));
    MatC c04 = MatC(float16_t(0));
    MatC c05 = MatC(float16_t(0));
    MatC c06 = MatC(float16_t(0));
    MatC c07 = MatC(float16_t(0));
    MatC c10 = MatC(float16_t(0));
    MatC c11 = MatC(float16_t(0));
    MatC c12 = MatC(float16_t(0));
    MatC c13 = MatC(float16_t(0));
    MatC c14 = MatC(float16_t(0));
    MatC c15 = MatC(float16_t(0));
    MatC c16 = MatC(float16_t(0));
    MatC c17 = MatC(float16_t(0));

    for (uint k = 0; k < Kd; k += 16) {
        let a0 = MatA.Load<CoopMatMatrixLayout.RowMajor>(gemmh_A, row0 * Kd + k, Kd);
        let a1 = MatA.Load<CoopMatMatrixLayout.RowMajor>(gemmh_A, (row0 + 16) * Kd + k, Kd);
        let b0 = MatB.Load<CoopMatMatrixLayout.RowMajor>(gemmh_B, k * N + col0 + 0, N);
        let b1 = MatB.Load<CoopMatMatrixLayout.RowMajor>(gemmh_B, k * N + col0 + 16, N);
        let b2 = MatB.Load<CoopMatMatrixLayout.RowMajor>(gemmh_B, k * N + col0 + 32, N);
        let b3 = MatB.Load<CoopMatMatrixLayout.RowMajor>(gemmh_B, k * N + col0 + 48, N);
        let b4 = MatB.Load<CoopMatMatrixLayout.RowMajor>(gemmh_B, k * N + col0 + 64, N);
        let b5 = MatB.Load<CoopMatMatrixLayout.RowMajor>(gemmh_B, k * N + col0 + 80, N);
        let b6 = MatB.Load<CoopMatMatrixLayout.RowMajor>(gemmh_B, k * N + col0 + 96, N);
        let b7 = MatB.Load<CoopMatMatrixLayout.RowMajor>(gemmh_B, k * N + col0 + 112, N);
        c00 = coopMatMulAdd<float16_t, false>(a0, b0, c00);
        c01 = coopMatMulAdd<float16_t, false>(a0, b1, c01);
        c02 = coopMatMulAdd<float16_t, false>(a0, b2, c02);
        c03 = coopMatMulAdd<float16_t, false>(a0, b3, c03);
        c04 = coopMatMulAdd<float16_t, false>(a0, b4, c04);
        c05 = coopMatMulAdd<float16_t, false>(a0, b5, c05);
        c06 = coopMatMulAdd<float16_t, false>(a0, b6, c06);
        c07 = coopMatMulAdd<float16_t, false>(a0, b7, c07);
        c10 = coopMatMulAdd<float16_t, false>(a1, b0, c10);
        c11 = coopMatMulAdd<float16_t, false>(a1, b1, c11);
        c12 = coopMatMulAdd<float16_t, false>(a1, b2, c12);
        c13 = coopMatMulAdd<float16_t, false>(a1, b3, c13);
        c14 = coopMatMulAdd<float16_t, false>(a1, b4, c14);
        c15 = coopMatMulAdd<float16_t, false>(a1, b5, c15);
        c16 = coopMatMulAdd<float16_t, false>(a1, b6, c16);
        c17 = coopMatMulAdd<float16_t, false>(a1, b7, c17);
    }
    c00.Store<CoopMatMatrixLayout.RowMajor>(gemmh_C, row0 * N + col0, N);
    c01.Store<CoopMatMatrixLayout.RowMajor>(gemmh_C, row0 * N + col0 + 16, N);
    c02.Store<CoopMatMatrixLayout.RowMajor>(gemmh_C, row0 * N + col0 + 32, N);
    c03.Store<CoopMatMatrixLayout.RowMajor>(gemmh_C, row0 * N + col0 + 48, N);
    c04.Store<CoopMatMatrixLayout.RowMajor>(gemmh_C, row0 * N + col0 + 64, N);
    c05.Store<CoopMatMatrixLayout.RowMajor>(gemmh_C, row0 * N + col0 + 80, N);
    c06.Store<CoopMatMatrixLayout.RowMajor>(gemmh_C, row0 * N + col0 + 96, N);
    c07.Store<CoopMatMatrixLayout.RowMajor>(gemmh_C, row0 * N + col0 + 112, N);
    c10.Store<CoopMatMatrixLayout.RowMajor>(gemmh_C, (row0 + 16) * N + col0, N);
    c11.Store<CoopMatMatrixLayout.RowMajor>(gemmh_C, (row0 + 16) * N + col0 + 16, N);
    c12.Store<CoopMatMatrixLayout.RowMajor>(gemmh_C, (row0 + 16) * N + col0 + 32, N);
    c13.Store<CoopMatMatrixLayout.RowMajor>(gemmh_C, (row0 + 16) * N + col0 + 48, N);
    c14.Store<CoopMatMatrixLayout.RowMajor>(gemmh_C, (row0 + 16) * N + col0 + 64, N);
    c15.Store<CoopMatMatrixLayout.RowMajor>(gemmh_C, (row0 + 16) * N + col0 + 80, N);
    c16.Store<CoopMatMatrixLayout.RowMajor>(gemmh_C, (row0 + 16) * N + col0 + 96, N);
    c17.Store<CoopMatMatrixLayout.RowMajor>(gemmh_C, (row0 + 16) * N + col0 + 112, N);
}
#endif  // KERNEL_gemm_coop_f16acc

#ifdef KERNEL_coop_peak
// PEAK PROBE: pure tensor-core MulAdd loop, fragments loaded ONCE — the ceiling of the coop
// path per accumulator type. ACC_T selects f32 or f16 accumulate (GeForce Ampere runs f16-acc
// at 2x the f32-acc rate — this probe measures that directly). dims = {iters}.
// FLOP = subgroups * iters * 4 * 2*16^3.
using namespace linalg;

[[vk::binding(0, 0)]] StructuredBuffer<float16_t>   cp_in;
[[vk::binding(1, 0)]] RWStructuredBuffer<ACC_T>     cp_out;
[[vk::binding(2, 0)]] StructuredBuffer<uint>        cp_dims;

[shader("compute")]
[numthreads(128, 1, 1)]
void K(CP_NAME)(uint3 gid : SV_GroupID) {
    const uint iters = cp_dims[0];
    typealias MatA = CoopMat<float16_t, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
    typealias MatB = CoopMat<float16_t, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
    typealias MatC = CoopMat<ACC_T, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;
    let a = MatA.Load<CoopMatMatrixLayout.RowMajor>(cp_in, 0, 16);
    let b = MatB.Load<CoopMatMatrixLayout.RowMajor>(cp_in, 0, 16);
    MatC c0 = MatC(ACC_T(0));
    MatC c1 = MatC(ACC_T(0));
    MatC c2 = MatC(ACC_T(0));
    MatC c3 = MatC(ACC_T(0));
    for (uint t = 0; t < iters; ++t) {
        c0 = coopMatMulAdd<ACC_T, false>(a, b, c0);
        c1 = coopMatMulAdd<ACC_T, false>(a, b, c1);
        c2 = coopMatMulAdd<ACC_T, false>(a, b, c2);
        c3 = coopMatMulAdd<ACC_T, false>(a, b, c3);
    }
    c0.Store<CoopMatMatrixLayout.RowMajor>(cp_out, (gid.x * 4 + 0) * 256, 16);
    c1.Store<CoopMatMatrixLayout.RowMajor>(cp_out, (gid.x * 4 + 1) * 256, 16);
    c2.Store<CoopMatMatrixLayout.RowMajor>(cp_out, (gid.x * 4 + 2) * 256, 16);
    c3.Store<CoopMatMatrixLayout.RowMajor>(cp_out, (gid.x * 4 + 3) * 256, 16);
}
#endif  // KERNEL_coop_peak


#ifdef KERNEL_finalize_partials
// ON-DEVICE partial finalization: ONE 256-thread group tree-sums the G partials into out[0] —
// no host round-trip, so reductions can stay device-resident (a 1-element result array) or read
// back a single element instead of G. Same fixed tree order as the stage-2 kernels.
// dims = {G}.
[[vk::binding(0, 0)]] StructuredBuffer<ELEM>   finp_partials;
[[vk::binding(1, 0)]] RWStructuredBuffer<ELEM> finp_out;
[[vk::binding(2, 0)]] StructuredBuffer<uint>   finp_dims;

groupshared ELEM finp_red[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void K(finalize_partials)(uint3 ltid : SV_GroupThreadID) {
    const uint G = finp_dims[0];
    const uint t = ltid.x;
    ELEM acc = ZERO;
    for (uint k = t; k < G; k += 256) acc += finp_partials[k];
    finp_red[t] = acc;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s) finp_red[t] += finp_red[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0) finp_out[0] = finp_red[0];
}
#endif  // KERNEL_finalize_partials

#ifdef KERNEL_fill
// Constant fill: out[i] = s. The device-side backend of the factories (zeros/ones/full/*_like) —
// construction without a host staging vector or a PCIe upload. dims = {n}.
[[vk::binding(0, 0)]] RWStructuredBuffer<vector<ELEM, 4>> fill_out;
[[vk::binding(1, 0)]] StructuredBuffer<ELEM>              fill_s;
[[vk::binding(2, 0)]] StructuredBuffer<uint>              fill_dims;

[shader("compute")]
[numthreads(256, 1, 1)]
void K(fill)(uint3 tid : SV_DispatchThreadID) {
    const uint n4 = (fill_dims[0] + 3) >> 2;
    if (tid.x < n4) fill_out[tid.x] = vector<ELEM, 4>(fill_s[0]);
}
#endif  // KERNEL_fill

#ifdef KERNEL_copy
// Bandwidth-ceiling probe: pure vec4 stream copy (one read + one write per element). What this
// sustains at large n IS this machine's real DRAM ceiling under compute — every elementwise/
// reduction number is judged against it, not the paper spec. dims = {n}.
[[vk::binding(0, 0)]] StructuredBuffer<vector<ELEM, 4>>   copy_in;
[[vk::binding(1, 0)]] RWStructuredBuffer<vector<ELEM, 4>> copy_out;
[[vk::binding(2, 0)]] StructuredBuffer<uint>              copy_dims;

[shader("compute")]
[numthreads(256, 1, 1)]
void K(copy)(uint3 tid : SV_DispatchThreadID) {
    const uint n4 = (copy_dims[0] + 3) >> 2;
    const uint i = tid.x;
    if (i < n4) copy_out[i] = copy_in[i];
}
#endif  // KERNEL_copy

#ifdef KERNEL_im2col2d
// Batched 2-D im2col GATHER (the conv lowering's device-resident staging step): one thread per
// col element over col [C·KH·KW, B·OH·OW], where column b·OH·OW + oh·OW + ow holds output
// position (oh, ow) of sample b's receptive field and row c·KH·KW + kh·KW + kw its (channel,
// kernel-offset) coordinate; x is [B, C·H·W] laid out [B][C][H][W]. FULL OVERWRITE — padding
// cells are written as ZEROS, so the reused workspace needs no zeroing pass. Writes are
// contiguous (thread i writes col[i]). dims = {B, C, H, W, KH, KW, OH, OW, stride, pad}.
[[vk::binding(0, 0)]] StructuredBuffer<ELEM>   i2c_x;
[[vk::binding(1, 0)]] RWStructuredBuffer<ELEM> i2c_col;
[[vk::binding(2, 0)]] StructuredBuffer<uint>   i2c_dims;

[shader("compute")]
[numthreads(256, 1, 1)]
void K(im2col2d)(uint3 tid : SV_DispatchThreadID) {
    const uint B = i2c_dims[0], C = i2c_dims[1], H = i2c_dims[2], W = i2c_dims[3];
    const uint KH = i2c_dims[4], KW = i2c_dims[5], OH = i2c_dims[6], OW = i2c_dims[7];
    const uint stride = i2c_dims[8], pad = i2c_dims[9];
    const uint OO = OH * OW, BOO = B * OO;
    const uint i = tid.x;
    if (i >= C * KH * KW * BOO) return;
    // i = ((c·KH + kh)·KW + kw)·BOO + b·OO + oh·OW + ow — the col layout's linear index.
    const uint ow = i % OW, oh = (i / OW) % OH, b = (i / OO) % B;
    const uint kw = (i / BOO) % KW, kh = (i / (BOO * KW)) % KH, c = i / (BOO * KW * KH);
    const int ih = int(oh * stride + kh) - int(pad);
    const int iw = int(ow * stride + kw) - int(pad);
    const bool inside = ih >= 0 && ih < int(H) && iw >= 0 && iw < int(W);
    i2c_col[i] = inside ? i2c_x[(b * C + c) * H * W + uint(ih) * W + uint(iw)] : ZERO;
}
#endif  // KERNEL_im2col2d

#ifdef KERNEL_col2im2d
// Batched 2-D col2im — the EXACT adjoint of im2col2d, in GATHER form: one thread per dx element
// over dx [B, C·H·W]; each input cell SUMS its ≤ KH·KW contributing dcol cells (kh-major,
// kw-minor — the same association order as the host reference's scatter-add, so the result is
// bit-identical AND deterministic by construction: no atomics, no cross-thread accumulation).
// Full overwrite (the scatter-add's zeroing folded in). Same dims as im2col2d:
// dims = {B, C, H, W, KH, KW, OH, OW, stride, pad}.
[[vk::binding(0, 0)]] StructuredBuffer<ELEM>   c2i_dcol;
[[vk::binding(1, 0)]] RWStructuredBuffer<ELEM> c2i_dx;
[[vk::binding(2, 0)]] StructuredBuffer<uint>   c2i_dims;

[shader("compute")]
[numthreads(256, 1, 1)]
void K(col2im2d)(uint3 tid : SV_DispatchThreadID) {
    const uint B = c2i_dims[0], C = c2i_dims[1], H = c2i_dims[2], W = c2i_dims[3];
    const uint KH = c2i_dims[4], KW = c2i_dims[5], OH = c2i_dims[6], OW = c2i_dims[7];
    const uint stride = c2i_dims[8], pad = c2i_dims[9];
    const uint OO = OH * OW, BOO = B * OO;
    const uint i = tid.x;
    if (i >= B * C * H * W) return;
    // i = (b·C + c)·H·W + ih·W + iw — the [B][C][H][W] layout's linear index.
    const uint iw = i % W, ih = (i / W) % H, c = (i / (H * W)) % C, b = i / (C * H * W);
    ELEM acc = ZERO;
    for (uint kh = 0; kh < KH; ++kh) {
        // Invert ih = oh·stride − pad + kh: contributes iff (ih + pad − kh) is a non-negative
        // exact multiple of stride landing inside the output plane.
        const int ohs = int(ih + pad) - int(kh);
        if (ohs < 0 || uint(ohs) % stride != 0) continue;
        const uint oh = uint(ohs) / stride;
        if (oh >= OH) continue;
        for (uint kw = 0; kw < KW; ++kw) {
            const int ows = int(iw + pad) - int(kw);
            if (ows < 0 || uint(ows) % stride != 0) continue;
            const uint ow = uint(ows) / stride;
            if (ow >= OW) continue;
            acc += c2i_dcol[((c * KH + kh) * KW + kw) * BOO + b * OO + oh * OW + ow];
        }
    }
    c2i_dx[i] = acc;
}
#endif  // KERNEL_col2im2d

#ifdef KERNEL_conv_bias_act
// The conv forward epilogue, fused THREE-in-one — bias add + activation + layout transpose:
// a[(b·F + f)·OO + o] = act(yc[f·B·OO + b·OO + o] + bias[f]), taking the GEMM's filter-major
// [F, B·OO] output to the per-sample [B, F·OO] block in one pass. One thread per output element
// (contiguous writes). The act selector is a UNIFORM branch: 0 = identity, 1 = relu, 2 = tanh,
// 3 = sigmoid. f64 note: tanh/sigmoid evaluate their transcendental in f32 (GLSL.std.450 has no
// f64 exp/tanh — see the ew_exp f32only note); the arithmetic acts stay full-width exact.
// dims = {B, F, OO, act}.
[[vk::binding(0, 0)]] StructuredBuffer<ELEM>   cba_yc;
[[vk::binding(1, 0)]] StructuredBuffer<ELEM>   cba_bias;
[[vk::binding(2, 0)]] RWStructuredBuffer<ELEM> cba_a;
[[vk::binding(3, 0)]] StructuredBuffer<uint>   cba_dims;

ELEM cba_act(ELEM z, uint act) {
    if (act == 1) return z > ZERO ? z : ZERO;
    if (act == 2) return ELEM(tanh(float(z)));
    if (act == 3) return ELEM(1) / (ELEM(1) + ELEM(exp(-float(z))));
    return z;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void K(conv_bias_act)(uint3 tid : SV_DispatchThreadID) {
    const uint B = cba_dims[0], F = cba_dims[1], OO = cba_dims[2], act = cba_dims[3];
    const uint i = tid.x;
    if (i >= B * F * OO) return;
    // i = (b·F + f)·OO + o — the per-sample activation layout's linear index.
    const uint o = i % OO, f = (i / OO) % F, b = i / (F * OO);
    cba_a[i] = cba_act(cba_yc[f * B * OO + b * OO + o] + cba_bias[f], act);
}
#endif  // KERNEL_conv_bias_act

#ifdef KERNEL_conv_act_grad
// The conv backward epilogue (adjoint of conv_bias_act's act + layout, fused):
// dyc[f·B·OO + b·OO + o] = d[(b·F + f)·OO + o] · act'(a[(b·F + f)·OO + o]) — the chain-rule
// step and the layout transpose back to filter-major in one pass, one thread per dyc element
// (contiguous writes). The derivative is evaluated FROM THE OUTPUT a = f(z): identity' = 1,
// relu' = [a > 0], tanh' = 1 − a², sigmoid' = a(1 − a) — pure arithmetic, no transcendentals,
// exact in both element widths. dims = {B, F, OO, act}.
[[vk::binding(0, 0)]] StructuredBuffer<ELEM>   cag_d;
[[vk::binding(1, 0)]] StructuredBuffer<ELEM>   cag_a;
[[vk::binding(2, 0)]] RWStructuredBuffer<ELEM> cag_dyc;
[[vk::binding(3, 0)]] StructuredBuffer<uint>   cag_dims;

ELEM cag_dact(ELEM a, uint act) {
    if (act == 1) return a > ZERO ? ELEM(1) : ZERO;
    if (act == 2) return ELEM(1) - a * a;
    if (act == 3) return a * (ELEM(1) - a);
    return ELEM(1);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void K(conv_act_grad)(uint3 tid : SV_DispatchThreadID) {
    const uint B = cag_dims[0], F = cag_dims[1], OO = cag_dims[2], act = cag_dims[3];
    const uint BOO = B * OO;
    const uint i = tid.x;
    if (i >= F * BOO) return;
    // i = f·B·OO + b·OO + o — the filter-major delta layout's linear index.
    const uint o = i % OO, b = (i / OO) % B, f = i / BOO;
    const uint j = (b * F + f) * OO + o;    // the per-sample [B, F·OO] source index
    cag_dyc[i] = cag_d[j] * cag_dact(cag_a[j], act);
}
#endif  // KERNEL_conv_act_grad

#ifdef KERNEL_triad
// STREAM-triad probe: a = b + s*c (two reads + one write per element). dims = {n}.
[[vk::binding(0, 0)]] RWStructuredBuffer<vector<ELEM, 4>> triad_a;
[[vk::binding(1, 0)]] StructuredBuffer<vector<ELEM, 4>>   triad_b;
[[vk::binding(2, 0)]] StructuredBuffer<vector<ELEM, 4>>   triad_c;
[[vk::binding(3, 0)]] StructuredBuffer<ELEM>              triad_s;
[[vk::binding(4, 0)]] StructuredBuffer<uint>              triad_dims;

[shader("compute")]
[numthreads(256, 1, 1)]
void K(triad)(uint3 tid : SV_DispatchThreadID) {
    const uint n4 = (triad_dims[0] + 3) >> 2;
    const uint i = tid.x;
    if (i < n4) triad_a[i] = triad_b[i] + vector<ELEM, 4>(triad_s[0]) * triad_c[i];
}
#endif  // KERNEL_triad
