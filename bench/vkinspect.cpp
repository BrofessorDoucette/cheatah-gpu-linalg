// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// vkinspect — the diagnosis CLI behind the "why can't we beat cuBLAS" ledger. Prints
//   1. every cooperative-matrix M×N×K/type combination the driver advertises (tile shapes and
//      accumulator types MUST come from this table, never assumption), and
//   2. driver-reported pipeline statistics (registers, spills, …) for the GEMM kernels — the
//      arbiter for every register-pressure tuning decision.
// Vulkan-only; run from the repo root so the SPIR-V blobs resolve.
#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

#include <cstdio>
#include <cstdlib>

namespace {

const char* comp_type(VkComponentTypeKHR t) {
    switch (t) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
        case VK_COMPONENT_TYPE_FLOAT64_KHR: return "f64";
        case VK_COMPONENT_TYPE_SINT8_KHR: return "s8";
        case VK_COMPONENT_TYPE_SINT16_KHR: return "s16";
        case VK_COMPONENT_TYPE_SINT32_KHR: return "s32";
        case VK_COMPONENT_TYPE_SINT64_KHR: return "s64";
        case VK_COMPONENT_TYPE_UINT8_KHR: return "u8";
        case VK_COMPONENT_TYPE_UINT16_KHR: return "u16";
        case VK_COMPONENT_TYPE_UINT32_KHR: return "u32";
        case VK_COMPONENT_TYPE_UINT64_KHR: return "u64";
        default: {
            static char buf[16];
            std::snprintf(buf, sizeof buf, "e%d", static_cast<int>(t));
            return buf;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    // The capture flag is a pipeline-CREATION property — set the env before the context builds
    // anything (harmless if the extension is missing).
    setenv("CHEATAH_GPU_PIPELINE_STATS", "1", 1);
    auto& c = cheatah::gpu::linalg::detail::ctx();

    std::printf("== cooperative-matrix property table ==\n");
    const auto shapes = c.coop_shapes();
    if (shapes.empty()) std::printf("  (none — no VK_KHR_cooperative_matrix)\n");
    for (const auto& p : shapes)
        std::printf("  %2ux%2ux%2u  A=%-3s B=%-3s C=%-3s result=%-3s scope=%s%s\n",
                    p.MSize, p.NSize, p.KSize, comp_type(p.AType), comp_type(p.BType),
                    comp_type(p.CType), comp_type(p.ResultType),
                    p.scope == VK_SCOPE_SUBGROUP_KHR ? "subgroup" : "other",
                    p.saturatingAccumulation ? " sat" : "");

    std::printf("\n== pipeline statistics ==\n");
    const char* kernels[] = {"gemm_fast_f32", "gemm_edge_f32", "gemm_batched_f32",
                             "dot_partial_f32", "ew_add_f32"};
    for (const char* k : kernels) {
        std::printf("%s\n", k);
        std::fputs(c.pipeline_stats(k).c_str(), stdout);
    }
    if (c.coop_ok()) {
        std::printf("gemm_coop_f32\n");
        std::fputs(c.pipeline_stats("gemm_coop_f32").c_str(), stdout);
    }
    // Extra kernel names on the command line inspect anything else (e.g. experiment variants).
    for (int i = 1; i < argc; ++i) {
        std::printf("%s\n", argv[i]);
        std::fputs(c.pipeline_stats(argv[i]).c_str(), stdout);
    }
    return 0;
}
