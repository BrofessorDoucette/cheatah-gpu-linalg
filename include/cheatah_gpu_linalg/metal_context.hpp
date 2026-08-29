// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
#pragma once

/**
 * @file metal_context.hpp
 * @brief The METAL device context — one of the two backends behind context.hpp.
 *
 * cheatah-gpu deliberately stops at a faithful, native GPU surface (the `mtl.*` forwarders) and
 * leaves the ergonomic layer to its consumers; this is that layer for linear algebra, on Metal: a
 * process-wide device + command queue, per-kernel libraries compiled from the slang-generated MSL
 * (kernels/linalg.slang → `slangc -target metal` at build time), a cached compute-pipeline lookup
 * and 1-D/2-D dispatch. On Apple the real Metal runtime compiles + executes those kernels; off
 * Apple cheatah-gpu's software-emulated Metal device runs the registered C++ stand-ins
 * (kernels.hpp) — the SAME call sequence either way, so a kernel dispatched here is exercised end
 * to end on a Linux host too.
 *
 * On Apple the MSL for kernel `name` is read from `<msl dir>/<name>.metal`, where the directory is
 * the `CHEATAH_GPU_LINALG_MSL_DIR` compile definition (set by CMake to the build's shader
 * directory) or the same-named environment variable, which takes precedence.
 */

#include "gpu/metal/types.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#if !defined(__APPLE__)
#  include "gpu/metal/emulated/emulated.hpp"
#endif

#include "cheatah_gpu_linalg/kernels.hpp"

namespace cheatah::gpu::linalg {

/// Short alias for the cheatah-gpu Metal surface, shared by the container + routines + context.
namespace mtl = cheatah::gpu::metal;

namespace detail {

/// The backend-native buffer object routines pass around (opaque outside the context).
using Buffer = mtl::Buffer;

/// Process-wide Metal context: device + queue + per-kernel libraries + a pipeline cache.
/// Constructed lazily on first use via `ctx()`; lives for the process and releases everything it
/// owns at exit (the emulator's own registries are function-local statics that complete
/// construction inside this constructor, so they are destroyed AFTER this — the teardown order is
/// safe by construction).
class Context {
public:
    Context() {
#if !defined(__APPLE__)
        // Off Apple there is no MSL compiler; register the CPU stand-ins the emulator dispatches
        // to by function name BEFORE any pipeline is built for them.
        kernels::register_emulated_kernels();
#endif
        pool_  = mtl::AutoreleasePool::alloc()->init();
        dev_   = mtl::CreateSystemDefaultDevice();
        queue_ = dev_ ? dev_->newCommandQueue() : nullptr;
    }

    ~Context() {
        for (auto& [cls, list] : free_lists_)
            for (Buffer* b : list) b->release();
        for (Buffer* b : cached_) b->release();
        for (auto& [name, pso] : pipelines_) pso->release();
        if (queue_) queue_->release();
        if (dev_) dev_->release();
        if (pool_) pool_->release();
    }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    [[nodiscard]] bool ok() const { return dev_ && queue_; }

    /// Tensor-core cooperative matrices are a Vulkan-path feature; the Metal backend reports
    /// false (Apple simdgroup_matrix is a possible future analogue).
    [[nodiscard]] static bool coop_ok() { return false; }

    /// The transfer/dispatch ledger (see the Vulkan context) — unified memory still counts its
    /// host<->buffer copies, so residency assertions hold on both backends.
    struct TransferStats {
        std::uint64_t bytes_uploaded = 0;
        std::uint64_t bytes_downloaded = 0;
        std::uint64_t dispatches = 0;
    };
    [[nodiscard]] const TransferStats& stats() const { return stats_; }
    void reset_stats() { stats_ = {}; }

    /// A shared-storage device buffer of ≥ `bytes` bytes. Shared storage is host-addressable via
    /// `contents()`, which is how host<->device transfer works for the unified-memory model.
    /// POOLED like the Vulkan context: released buffers park in size-class free lists so per-op
    /// scratch (dims/scalars/partials/temporaries) recycles instead of re-allocating.
    [[nodiscard]] Buffer* new_buffer(std::size_t bytes) {
        const std::size_t cls = size_class(bytes);
        if (auto it = free_lists_.find(cls); it != free_lists_.end() && !it->second.empty()) {
            Buffer* b = it->second.back();
            it->second.pop_back();
            pooled_bytes_ -= cls;
            return b;
        }
        return dev_->newBuffer(cls, mtl::ResourceStorageModeShared);
    }

    /// Data buffers on Metal are the same shared-storage buffers (unified memory — shared IS
    /// device-resident on Apple silicon and on the emulator), so the kind split is a no-op here.
    [[nodiscard]] Buffer* new_data_buffer(std::size_t bytes) { return new_buffer(bytes); }

    /// The host-visible pointer to a buffer's elements.
    [[nodiscard]] static void* contents(Buffer* b) { return b->contents(); }

    /// Host->buffer copy (unified memory: a plain memcpy).
    void upload(Buffer* dst, const void* src, std::size_t bytes) {
        if (bytes > static_cast<std::size_t>(dst->length()))
            throw std::runtime_error("cheatah-gpu-linalg metal: upload exceeds buffer size");
        stats_.bytes_uploaded += bytes;
        std::memcpy(dst->contents(), src, bytes);
    }

    /// Buffer->host copy (unified memory: a plain memcpy).
    void download(Buffer* src, void* dst, std::size_t bytes) { download_at(src, dst, bytes, 0); }

    /// Download `bytes` from byte `offset` of a buffer.
    void download_at(Buffer* src, void* dst, std::size_t bytes, std::size_t offset) {
        const std::size_t len = static_cast<std::size_t>(src->length());
        if (offset > len || bytes > len - offset)
            throw std::runtime_error("cheatah-gpu-linalg metal: download exceeds buffer size");
        stats_.bytes_downloaded += bytes;
        std::memcpy(dst, static_cast<const char*>(src->contents()) + offset, bytes);
    }

    /// A buffer PERMANENTLY owned by a content cache (the dims cache) — release_buffer ignores
    /// it, mirroring the Vulkan context, so cached handles flow through uniform call sites.
    [[nodiscard]] Buffer* new_cached_buffer(std::size_t bytes) {
        Buffer* b = new_buffer(bytes);
        cached_.insert(b);
        return b;
    }

    /// Submission fusing (Vulkan-context parity): dispatches here are synchronous host calls
    /// (emulated) or per-encoder commits, so fusing is a NO-OP — the calls exist so shared
    /// call sites compile identically against both contexts.
    void begin_fuse() {}
    void end_fuse() {}

    /// Return a buffer to the pool; beyond the cap it is genuinely released.
    void release_buffer(Buffer* b) {
        if (cached_.count(b) != 0) return;   // content-cache property: the cache keeps it alive
        const std::size_t cls = static_cast<std::size_t>(b->length());
        if (pooled_bytes_ + cls <= kPoolCapBytes) {
            pooled_bytes_ += cls;
            free_lists_[cls].push_back(b);
            return;
        }
        b->release();
    }

    /// Bind `count` buffers at indices 0..count-1 and run `name` over a 1-D grid of `width`
    /// threads (threadgroups of kLocal1d, the kernel's `numthreads`), blocking until complete.
    void dispatch_1d(const char* name, Buffer** buffers, unsigned count, std::uint64_t width) {
        dispatch(name, buffers, count, mtl::Size(width, 1, 1),
                 mtl::Size(width < kernels::kLocal1d ? width : kernels::kLocal1d, 1, 1));
    }

    /// Bind `count` buffers at indices 0..count-1 and run `name` over a `w`×`h` 2-D grid
    /// (threadgroups of kLocal2d×kLocal2d), blocking until complete.
    void dispatch_2d(const char* name, Buffer** buffers, unsigned count, std::uint64_t w,
                     std::uint64_t h) {
        dispatch(name, buffers, count, mtl::Size(w, h, 1),
                 mtl::Size(w < kernels::kLocal2d ? w : kernels::kLocal2d,
                           h < kernels::kLocal2d ? h : kernels::kLocal2d, 1));
    }

    /// Bind `count` buffers and launch EXACTLY `groups` threadgroups of kLocal1d threads (for
    /// group-indexed 1-D kernels — the two-stage reductions).
    void dispatch_blocks_1d(const char* name, Buffer** buffers, unsigned count,
                            std::uint32_t groups) {
        mtl::ComputePipelineState* pso = pipeline(name);
        ++stats_.dispatches;
        mtl::AutoreleasePool* pool = mtl::AutoreleasePool::alloc()->init();
        mtl::CommandBuffer* cb = queue_->commandBuffer();
        mtl::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        for (unsigned i = 0; i < count; ++i) enc->setBuffer(buffers[i], 0, i);
        enc->dispatchThreadgroups(mtl::Size(groups, 1, 1), mtl::Size(kernels::kLocal1d, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        pool->release();
    }

    /// Bind `count` buffers and launch EXACTLY `gx`×`gy` threadgroups of kLocal2d×kLocal2d (for
    /// block-indexed kernels like the register-tiled GEMM).
    void dispatch_blocks_2d(const char* name, Buffer** buffers, unsigned count, std::uint32_t gx,
                            std::uint32_t gy) {
        mtl::ComputePipelineState* pso = pipeline(name);
        ++stats_.dispatches;
        mtl::AutoreleasePool* pool = mtl::AutoreleasePool::alloc()->init();
        mtl::CommandBuffer* cb = queue_->commandBuffer();
        mtl::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        for (unsigned i = 0; i < count; ++i) enc->setBuffer(buffers[i], 0, i);
        enc->dispatchThreadgroups(mtl::Size(gx, gy, 1),
                                  mtl::Size(kernels::kLocal2d, kernels::kLocal2d, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        pool->release();
    }

    /// Bind `count` buffers and run `name` over a `w`×`h`×`depth` 3-D grid (threadgroups of
    /// kLocal2d×kLocal2d×1 — z maps one layer per slice, e.g. a GEMM batch).
    void dispatch_3d(const char* name, Buffer** buffers, unsigned count, std::uint64_t w,
                     std::uint64_t h, std::uint64_t depth) {
        dispatch(name, buffers, count, mtl::Size(w, h, depth),
                 mtl::Size(w < kernels::kLocal2d ? w : kernels::kLocal2d,
                           h < kernels::kLocal2d ? h : kernels::kLocal2d, 1));
    }

private:
    /// Encode + commit one dispatch of `name` over `grid` threads and block until it completes.
    /// `dispatchThreads` takes the exact thread grid (Metal handles the ragged edge groups). The
    /// command buffer + encoder are autoreleased objects — a per-dispatch pool drains them, so a
    /// long linalg run doesn't accumulate one pair per dispatch.
    void dispatch(const char* name, Buffer** buffers, unsigned count, mtl::Size grid,
                  mtl::Size tptg) {
        mtl::ComputePipelineState* pso = pipeline(name);
        ++stats_.dispatches;
        mtl::AutoreleasePool* pool = mtl::AutoreleasePool::alloc()->init();
        mtl::CommandBuffer* cb = queue_->commandBuffer();
        mtl::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        for (unsigned i = 0; i < count; ++i) enc->setBuffer(buffers[i], 0, i);
        enc->dispatchThreads(grid, tptg);
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        pool->release();
    }

    /// The MSL source for kernel `name`. On Apple: the slang-generated `<name>.metal` from the
    /// shader directory (env `CHEATAH_GPU_LINALG_MSL_DIR` beats the baked-in build path). Off
    /// Apple the emulator never reads source — a placeholder is enough to build the library.
    /// A kernel name is BARE or PATH-QUALIFIED, exactly as on the Vulkan lane (see its
    /// `spv_bytes`): a bare name resolves against the shader directory, a qualified name is used as
    /// given. The MSL function inside the library is always the BASENAME.
    [[nodiscard]] static std::string basename(const std::string& name) {
        const std::size_t s = name.rfind('/');
        return s == std::string::npos ? name : name.substr(s + 1);
    }
    [[nodiscard]] static std::string resolve(const std::string& name) {
        if (name.find('/') != std::string::npos) return name;
#if defined(__APPLE__)
        const char* env = std::getenv("CHEATAH_GPU_LINALG_MSL_DIR");
#  if defined(CHEATAH_GPU_LINALG_MSL_DIR)
        const std::string dir = env ? env : CHEATAH_GPU_LINALG_MSL_DIR;
#  else
        const std::string dir =
            env ? std::string(env)
                : (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
                   "build" / "shaders")
                      .string();
#  endif
        return dir + "/" + name;
#else
        return name;   // the emulator never reads source; the name is only a table key
#endif
    }
    [[nodiscard]] static std::string msl_source(const std::string& name) {
#if defined(__APPLE__)
        std::ifstream in(resolve(name) + ".metal");
        if (!in)
            throw std::runtime_error("cheatah-gpu-linalg: missing MSL for kernel '" + name + "'");
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
#else
        return "// emulated Metal device: source unused, kernel '" + name +
               "' runs the registered C++ stand-in\n";
#endif
    }

    /// The compute pipeline for `name`, built once and cached (pipeline creation is expensive and
    /// a linalg program reruns the same handful of kernels many times). Each kernel gets its own
    /// small library, compiled from its own slang-generated MSL file.
    mtl::ComputePipelineState* pipeline(const std::string& name) {
        const std::string key = resolve(name);
        if (auto it = pipelines_.find(key); it != pipelines_.end()) return it->second;
        const std::string source = msl_source(name);
        // A local pool drains the autoreleased NS::Strings; the pipeline itself is owned and
        // cached for the process lifetime.
        mtl::AutoreleasePool* pool = mtl::AutoreleasePool::alloc()->init();
        mtl::Error* err = nullptr;
        mtl::String* src = mtl::String::string(source.c_str(), mtl::UTF8StringEncoding);
        mtl::Library* lib =
            dev_->newLibrary(src, static_cast<const mtl::CompileOptions*>(nullptr), &err);
        if (!lib) {
            pool->release();
            throw std::runtime_error("cheatah-gpu-linalg: MSL for kernel '" + name +
                                     "' did not compile");
        }
        const std::string fn_name = basename(name);
        mtl::String* fname = mtl::String::string(fn_name.c_str(), mtl::UTF8StringEncoding);
        mtl::Function* fn = lib->newFunction(fname);
        mtl::ComputePipelineState* pso = dev_->newComputePipelineState(fn, &err);
        fn->release();
        lib->release();
        pool->release();
        if (!pso)
            throw std::runtime_error("cheatah-gpu-linalg: no compute pipeline for kernel '" +
                                     name + "'");
        pipelines_.emplace(key, pso);
        return pso;
    }

    /// Free-list size classes: next power of two, floor 256 bytes (mirrors the Vulkan context).
    [[nodiscard]] static std::size_t size_class(std::size_t bytes) {
        if (bytes > (std::size_t(1) << 62))
            throw std::runtime_error("cheatah-gpu-linalg metal: allocation size overflow");
        std::size_t cls = 256;
        while (cls < bytes) cls <<= 1;
        return cls;
    }
    static constexpr std::size_t kPoolCapBytes = std::size_t(1) << 30;  // 1 GiB

    mtl::AutoreleasePool* pool_ = nullptr;
    mtl::Device* dev_ = nullptr;
    mtl::CommandQueue* queue_ = nullptr;
    std::unordered_map<std::string, mtl::ComputePipelineState*> pipelines_;
    std::unordered_map<std::size_t, std::vector<Buffer*>> free_lists_;  // pooled-buffer classes
    std::unordered_set<Buffer*> cached_;  // content-cache-owned scratch (dims), freed at teardown
    std::size_t pooled_bytes_ = 0;
    TransferStats stats_{};
};

/// The one shared context. First call constructs the device; later calls reuse it.
inline Context& ctx() {
    static Context c;
    return c;
}

}  // namespace detail
}  // namespace cheatah::gpu::linalg
