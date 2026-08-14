// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
#pragma once

/**
 * @file vulkan_context.hpp
 * @brief The VULKAN device context — one of the two backends behind context.hpp.
 *
 * The Vulkan mirror of metal_context.hpp, built on cheatah-gpu's generated `vk.*` forwarders
 * (gpu/vulkan/commands.hpp) and its `gpu.dispatch` workgroup math: a process-wide instance +
 * device + compute queue, per-kernel compute pipelines loaded from the SPIR-V that `slangc`
 * compiled out of kernels/linalg.slang at build time, one shared descriptor-set layout (every
 * kernel binds storage buffers 0..count-1), and blocking 1-D/2-D dispatch. Unlike Metal there is
 * no runtime shader compiler in the loop — the `.spv` for kernel `name` is read from
 * `<spv dir>/<name>.spv`, where the directory is the `CHEATAH_GPU_LINALG_SPV_DIR` compile
 * definition (set by CMake to the build's shader directory) or the same-named environment
 * variable, which takes precedence.
 *
 * Device selection: the highest-scoring physical device with a compute queue (discrete >
 * integrated > virtual > CPU). Set `CHEATAH_GPU_LINALG_VK_DEVICE` to a device-name substring
 * (case-sensitive, e.g. "llvmpipe") or a zero-based index to force one — that is how the QA gate
 * exercises the software rasterizer alongside the real GPU.
 *
 * f64: the `*_f64` kernels need the `shaderFloat64` device feature; it is enabled when the device
 * has it, and dispatching an `_f64` kernel without it throws (a clean "unsupported here" rather
 * than a validation error).
 */

#include "gpu/dispatch/dispatch.hpp"
#include "gpu/vulkan/commands.hpp"

#include "cheatah_gpu_linalg/kernels.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cheatah::gpu::linalg {

/// Short alias for the cheatah-gpu Vulkan surface, shared by the container + routines + context.
namespace vkc = cheatah::gpu::vulkan;

namespace detail {

/// The backend-native buffer object routines pass around (opaque outside the context): a storage
/// buffer bound to host-visible, host-coherent memory, persistently mapped.
struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* map = nullptr;         // host pointer for host-visible buffers; nullptr when device-local
    VkDeviceSize bytes = 0;
    bool device_local = false;   // data buffers live in VRAM; scratch stays host-visible
    unsigned kind = 0;           // 0 scratch/WC, 1 device-local, 2 cached readback staging
    bool cached = false;         // permanently owned by a content cache; release_buffer ignores it
};

/// Process-wide Vulkan compute context. Constructed lazily on first use via `ctx()`; lives for
/// the process. Throws std::runtime_error on any setup failure — a linalg program cannot proceed
/// without its device.
class Context {
public:
    static constexpr std::uint32_t kMaxBindings = 8;  ///< storage-buffer slots in the shared layout

    Context() {
#if defined(VOLK_H_)
        // The volk lane (no libvulkan link anywhere): bring the loader up in-process, then aim
        // every vk* pointer at the selected driver. Failing HERE means the machine has no
        // Vulkan loader/driver at all — the same "no device" outcome as an empty enumeration.
        static const VkResult volk_ok = volkInitialize();
        if (volk_ok != VK_SUCCESS)
            throw std::runtime_error("cheatah-gpu-linalg: no Vulkan loader (volkInitialize failed)");
#endif
        create_instance();
#if defined(VOLK_H_)
        volkLoadInstance(instance_);
#endif
        pick_physical_device();
        create_device();
        create_descriptor_machinery();
        create_command_machinery();
    }

    ~Context() {
        if (device_ != VK_NULL_HANDLE) {
            vkc::DeviceWaitIdle(device_);
            for (auto& [cls, list] : free_lists_)
                for (Buffer* b : list) destroy_buffer(b);
            for (Buffer* b : cached_bufs_) destroy_buffer(b);
            for (auto& [name, pipe] : pipelines_) vkc::DestroyPipeline(device_, pipe, nullptr);
            vkc::DestroyFence(device_, fence_, nullptr);
            vkc::DestroyCommandPool(device_, cmd_pool_, nullptr);
            vkc::DestroyDescriptorPool(device_, pool_, nullptr);
            vkc::DestroyPipelineLayout(device_, pipe_layout_, nullptr);
            vkc::DestroyDescriptorSetLayout(device_, set_layout_, nullptr);
            vkc::DestroyDevice(device_, nullptr);
        }
        if (instance_ != VK_NULL_HANDLE) vkc::DestroyInstance(instance_, nullptr);
    }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    [[nodiscard]] bool ok() const { return device_ != VK_NULL_HANDLE; }

    /// Whether the KHR cooperative-matrix (tensor-core) kernels can run on this device.
    [[nodiscard]] bool coop_ok() const { return coop_ok_; }

    /// Every cooperative-matrix M×N×K/type combination the driver advertises — the DATA the
    /// tensor-path tile shapes must come from (never assumption).
    [[nodiscard]] std::vector<VkCooperativeMatrixPropertiesKHR> coop_shapes() const {
        std::vector<VkCooperativeMatrixPropertiesKHR> v;
        if (!coop_ok_) return v;
        auto fp = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            vkc::GetInstanceProcAddr(instance_,
                                     "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
        if (fp == nullptr) return v;
        std::uint32_t n = 0;
        fp(physical_, &n, nullptr);
        v.resize(n);
        for (VkCooperativeMatrixPropertiesKHR& p : v)
            p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        fp(physical_, &n, v.data());
        return v;
    }

    /// Driver-reported statistics (register count, spills, occupancy…) for a kernel's pipeline.
    /// Requires VK_KHR_pipeline_executable_properties AND the CHEATAH_GPU_PIPELINE_STATS env var
    /// set BEFORE the pipeline is first built (the capture flag is a creation-time property).
    [[nodiscard]] std::string pipeline_stats(const char* name) {
        if (!exec_stats_ok_) return "  (VK_KHR_pipeline_executable_properties unavailable)\n";
        VkPipeline pipe = pipeline(name);
        auto get_props = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
            vkc::GetInstanceProcAddr(instance_, "vkGetPipelineExecutablePropertiesKHR"));
        auto get_stats = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
            vkc::GetInstanceProcAddr(instance_, "vkGetPipelineExecutableStatisticsKHR"));
        if (get_props == nullptr || get_stats == nullptr) return "  (loader has no entry points)\n";
        VkPipelineInfoKHR pinfo{};
        pinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR;
        pinfo.pipeline = pipe;
        std::uint32_t nexec = 0;
        get_props(device_, &pinfo, &nexec, nullptr);
        std::vector<VkPipelineExecutablePropertiesKHR> props(nexec);
        for (auto& p : props) p.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
        get_props(device_, &pinfo, &nexec, props.data());
        std::string out;
        for (std::uint32_t e = 0; e < nexec; ++e) {
            out += "  exec '" + std::string(props[e].name) + "' subgroup " +
                   std::to_string(props[e].subgroupSize) + "\n";
            VkPipelineExecutableInfoKHR einfo{};
            einfo.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR;
            einfo.pipeline = pipe;
            einfo.executableIndex = e;
            std::uint32_t nstat = 0;
            get_stats(device_, &einfo, &nstat, nullptr);
            std::vector<VkPipelineExecutableStatisticKHR> stats(nstat);
            for (auto& st : stats) st.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
            get_stats(device_, &einfo, &nstat, stats.data());
            for (const auto& st : stats) {
                out += "    " + std::string(st.name) + ": ";
                switch (st.format) {
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                        out += st.value.b32 ? "true" : "false"; break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                        out += std::to_string(st.value.i64); break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                        out += std::to_string(st.value.u64); break;
                    default:
                        out += std::to_string(st.value.f64); break;
                }
                out += "\n";
            }
        }
        return out;
    }

    /// BENCH PROBE (additive, not used by the library): record one dispatch into a persistent
    /// re-submittable command buffer, then replay it with `submit_recorded` — splits the
    /// dispatch floor into record-side vs submit+fence-side shares.
    void record_dispatch(const char* name, Buffer** buffers, unsigned count,
                         cheatah::gpu::dispatch::Dim3 groups) {
        if (cmd2_ == VK_NULL_HANDLE) {
            VkCommandBufferAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool = cmd_pool_;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            check(vkc::AllocateCommandBuffers(device_, &ai, &cmd2_), "AllocateCommandBuffers");
        }
        VkPipeline pipe = pipeline(name);
        VkDescriptorBufferInfo infos[kMaxBindings]{};
        VkWriteDescriptorSet writes[kMaxBindings]{};
        for (unsigned i = 0; i < count; ++i) {
            infos[i].buffer = buffers[i]->buf;
            infos[i].range = VK_WHOLE_SIZE;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set_;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkc::UpdateDescriptorSets(device_, count, writes, 0, nullptr);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;   // no ONE_TIME flag: replayable
        check(vkc::BeginCommandBuffer(cmd2_, &bi), "BeginCommandBuffer");
        vkc::CmdBindPipeline(cmd2_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkc::CmdBindDescriptorSets(cmd2_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 1,
                                   &set_, 0, nullptr);
        vkc::CmdDispatch(cmd2_, groups.x, groups.y, groups.z);
        check(vkc::EndCommandBuffer(cmd2_), "EndCommandBuffer");
    }
    /// Replay the command buffer prepared by @ref record_dispatch (submit + fence only).
    void submit_recorded() {
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd2_;
        check(vkc::QueueSubmit(queue_, 1, &si, fence_), "QueueSubmit");
        check(vkc::WaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "WaitForFences");
        check(vkc::ResetFences(device_, 1, &fence_), "ResetFences");
    }


    /// SUBMISSION FUSING: between begin_fuse/end_fuse, dispatches RECORD into one command
    /// buffer (a compute→compute barrier between each) and end_fuse pays the ~23 µs
    /// submit+fence floor ONCE for the whole chain. Built for the two-stage reductions'
    /// partial+finalize pair; ≤ kMaxFuse dispatches per fuse; results are visible only after
    /// end_fuse (do NOT read buffers between the calls). Internal-use contract.
    void begin_fuse() {
        if (fuse_active_)
            throw std::runtime_error("cheatah-gpu-linalg vulkan: nested begin_fuse");
        fuse_active_ = true;
        fuse_count_ = 0;
    }
    void end_fuse() {
        fuse_active_ = false;
        if (fuse_count_ == 0) return;
        fuse_count_ = 0;
        check(vkc::EndCommandBuffer(cmd_), "EndCommandBuffer");
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        check(vkc::QueueSubmit(queue_, 1, &si, fence_), "QueueSubmit");
        check(vkc::WaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "WaitForFences");
        check(vkc::ResetFences(device_, 1, &fence_), "ResetFences");
        check(vkc::ResetCommandBuffer(cmd_, 0), "ResetCommandBuffer");
    }

    /// The transfer/dispatch ledger — device residency as a MEASURABLE property (examples assert
    /// "zero bytes downloaded inside the loop" with this).
    struct TransferStats {
        std::uint64_t bytes_uploaded = 0;
        std::uint64_t bytes_downloaded = 0;
        std::uint64_t dispatches = 0;
    };
    [[nodiscard]] const TransferStats& stats() const { return stats_; }
    void reset_stats() { stats_ = {}; }

    /// A storage buffer of ≥ `bytes` bytes in host-visible, host-coherent memory, persistently
    /// mapped — the unified-memory analogue of Metal's shared storage. POOLED: released buffers
    /// park in a size-class free list, because `vkAllocateMemory` + map costs ~250 µs on a real
    /// driver — recycling turns the per-op scratch (dims/scalars/partials/temporaries) from the
    /// dominant cost of every small dispatch into a hash lookup.
    [[nodiscard]] Buffer* new_buffer(std::size_t bytes) { return acquire(bytes, 0); }

    /// A DEVICE-LOCAL storage buffer for array elements — VRAM, the memory the GPU can actually
    /// stream at full bandwidth (host-visible memory on a discrete card is PCIe-remote: reading
    /// GEMM operands from it caps the kernel at bus speed, not the 448 GB/s the silicon has).
    /// Filled/read through @ref upload / @ref download (staged copies). Pooled like scratch.
    [[nodiscard]] Buffer* new_data_buffer(std::size_t bytes) { return acquire(bytes, 1); }

    /// A scratch buffer PERMANENTLY owned by a content cache (the dims cache): `release_buffer`
    /// ignores it, so cached handles can flow through the uniform acquire/release call sites
    /// without ever re-entering the pool. Destroyed with the context.
    [[nodiscard]] Buffer* new_cached_buffer(std::size_t bytes) {
        Buffer* b = acquire(bytes, 0);
        b->cached = true;
        cached_bufs_.push_back(b);
        return b;
    }

    /// The host-visible pointer to a SCRATCH buffer's elements (nullptr for device-local data
    /// buffers — those move bytes through upload/download).
    [[nodiscard]] static void* contents(Buffer* b) { return b->map; }

    /// Copy `bytes` of host memory into a buffer. Host-visible: plain memcpy. Device-local:
    /// staged through a pooled host-visible buffer + one blocking transfer submit.
    void upload(Buffer* dst, const void* src, std::size_t bytes) {
        if (bytes > dst->bytes)
            throw std::runtime_error("cheatah-gpu-linalg vulkan: upload exceeds buffer size");
        stats_.bytes_uploaded += bytes;
        if (!dst->device_local) {
            std::memcpy(dst->map, src, bytes);
            return;
        }
        Buffer* staging = new_buffer(bytes);
        std::memcpy(staging->map, src, bytes);
        run_copy(staging, dst, bytes);
        release_buffer(staging);
    }

    /// Copy `bytes` of a buffer into host memory (the download mirror of @ref upload).
    void download(Buffer* src, void* dst, std::size_t bytes) { download_at(src, dst, bytes, 0); }

    /// Download `bytes` starting at byte `offset` of a buffer (single-element `get` reads one
    /// element instead of the whole array).
    void download_at(Buffer* src, void* dst, std::size_t bytes, std::size_t offset) {
        if (offset > src->bytes || bytes > src->bytes - offset)
            throw std::runtime_error("cheatah-gpu-linalg vulkan: download exceeds buffer size");
        stats_.bytes_downloaded += bytes;
        if (!src->device_local) {
            std::memcpy(dst, static_cast<const char*>(src->map) + offset, bytes);
            return;
        }
        Buffer* staging = acquire(bytes, 2);   // host-CACHED readback staging
        run_copy_at(src, staging, bytes, offset, 0);
        std::memcpy(dst, staging->map, bytes);
        release_buffer(staging);
    }

    /// Return a buffer to its pool (dispatches are blocking, so nothing is in flight). The pool
    /// is capped; beyond the cap the buffer is genuinely destroyed.
    void release_buffer(Buffer* b) {
        if (b->cached) return;   // content-cache property: the cache keeps it alive
        if (pooled_bytes_ + b->bytes <= kPoolCapBytes) {
            pooled_bytes_ += b->bytes;
            free_lists_[pool_key(b->bytes, b->kind)].push_back(b);
            return;
        }
        destroy_buffer(b);
    }

    /// Bind `count` buffers at bindings 0..count-1 and run `name` over a 1-D grid of `width`
    /// threads (workgroups of kLocal1d, the kernel's `numthreads`), blocking until complete.
    void dispatch_1d(const char* name, Buffer** buffers, unsigned count, std::uint64_t width) {
        namespace d = cheatah::gpu::dispatch;
        submit(name, buffers, count,
               d::Dim3{d::group_count_1d(static_cast<std::uint32_t>(width), kernels::kLocal1d)});
    }

    /// Bind `count` buffers at bindings 0..count-1 and run `name` over a `w`×`h` 2-D grid
    /// (workgroups of kLocal2d×kLocal2d), blocking until complete.
    void dispatch_2d(const char* name, Buffer** buffers, unsigned count, std::uint64_t w,
                     std::uint64_t h) {
        namespace d = cheatah::gpu::dispatch;
        submit(name, buffers, count,
               d::group_count_3d(d::Dim3{static_cast<std::uint32_t>(w),
                                         static_cast<std::uint32_t>(h)},
                                 d::Dim3{kernels::kLocal2d, kernels::kLocal2d}));
    }

    /// Bind `count` buffers and launch EXACTLY `groups` workgroups of kLocal1d threads (for
    /// group-indexed 1-D kernels — the two-stage reductions).
    void dispatch_blocks_1d(const char* name, Buffer** buffers, unsigned count,
                            std::uint32_t groups) {
        submit(name, buffers, count, cheatah::gpu::dispatch::Dim3{groups, 1u, 1u});
    }

    /// Bind `count` buffers and launch EXACTLY `gx`×`gy` workgroups (for block-indexed kernels
    /// like the register-tiled GEMM, whose workgroup ID — not thread ID — addresses the tile).
    void dispatch_blocks_2d(const char* name, Buffer** buffers, unsigned count, std::uint32_t gx,
                            std::uint32_t gy) {
        submit(name, buffers, count, cheatah::gpu::dispatch::Dim3{gx, gy, 1u});
    }

    /// Bind `count` buffers and run `name` over a `w`×`h`×`depth` 3-D grid (workgroups of
    /// kLocal2d×kLocal2d×1 — the z axis maps one workgroup layer per slice, e.g. a GEMM batch).
    void dispatch_3d(const char* name, Buffer** buffers, unsigned count, std::uint64_t w,
                     std::uint64_t h, std::uint64_t depth) {
        namespace d = cheatah::gpu::dispatch;
        submit(name, buffers, count,
               d::group_count_3d(d::Dim3{static_cast<std::uint32_t>(w),
                                         static_cast<std::uint32_t>(h),
                                         static_cast<std::uint32_t>(depth)},
                                 d::Dim3{kernels::kLocal2d, kernels::kLocal2d, 1u}));
    }

private:
    /// Free-list size classes: next power of two, floor 256 bytes — few classes, high reuse.
    /// Power-of-two size classes from 256 B. INVARIANT the vec4 kernels rely on: every class is
    /// 16-byte aligned and ≥ the ceil-to-vec4 size of any request it serves — writing
    /// ceil(n/4) vec4s into a class sized for n elements never leaves the allocation (a
    /// pow2 ≥ 256 divided by any element size ≤ 16 is a multiple of 4 elements).
    [[nodiscard]] static std::size_t size_class(std::size_t bytes) {
        if (bytes > (std::size_t(1) << 62))
            throw std::runtime_error("cheatah-gpu-linalg vulkan: allocation size overflow");
        std::size_t cls = 256;
        while (cls < bytes) cls <<= 1;
        return cls;
    }

    /// Pool key: the size class tagged with the memory kind (0 scratch/WC, 1 device-local,
    /// 2 host-CACHED readback staging — CPU reads from write-combined memory run ~0.2 GB/s,
    /// so downloads stage through a cached heap instead).
    [[nodiscard]] static std::size_t pool_key(std::size_t cls, unsigned kind) {
        return (cls << 2) | kind;
    }

    /// Acquire a pooled or fresh buffer of the given kind (0 WC scratch, 1 device-local,
    /// 2 cached readback staging).
    [[nodiscard]] Buffer* acquire(std::size_t bytes, unsigned kind) {
        const std::size_t cls = size_class(bytes);
        const std::size_t key = pool_key(cls, kind);
        if (auto it = free_lists_.find(key); it != free_lists_.end() && !it->second.empty()) {
            Buffer* b = it->second.back();
            it->second.pop_back();
            pooled_bytes_ -= b->bytes;
            return b;
        }
        Buffer* b = new Buffer;
        b->bytes = cls;
        b->kind = kind;
        b->device_local = (kind == 1);
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = cls;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkc::CreateBuffer(device_, &bi, nullptr, &b->buf), "CreateBuffer");
        VkMemoryRequirements req{};
        vkc::GetBufferMemoryRequirements(device_, b->buf, &req);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = (kind == 1) ? device_memory_type(req.memoryTypeBits)
                                         : host_memory_type(req.memoryTypeBits, kind == 2);
        check(vkc::AllocateMemory(device_, &ai, nullptr, &b->mem), "AllocateMemory");
        check(vkc::BindBufferMemory(device_, b->buf, b->mem, 0), "BindBufferMemory");
        if (kind != 1)
            check(vkc::MapMemory(device_, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map), "MapMemory");
        return b;
    }

    /// One blocking transfer submit: src[0..bytes) -> dst[0..bytes) via the shared command buffer.
    void run_copy(Buffer* src, Buffer* dst, std::size_t bytes) { run_copy_at(src, dst, bytes, 0, 0); }

    void run_copy_at(Buffer* src, Buffer* dst, std::size_t bytes, std::size_t src_off,
                     std::size_t dst_off) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkc::BeginCommandBuffer(cmd_, &bi), "BeginCommandBuffer(copy)");
        VkBufferCopy region{};
        region.srcOffset = src_off;
        region.dstOffset = dst_off;
        region.size = bytes;
        vkc::CmdCopyBuffer(cmd_, src->buf, dst->buf, 1, &region);
        check(vkc::EndCommandBuffer(cmd_), "EndCommandBuffer(copy)");
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        check(vkc::QueueSubmit(queue_, 1, &si, fence_), "QueueSubmit(copy)");
        check(vkc::WaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "WaitForFences(copy)");
        check(vkc::ResetFences(device_, 1, &fence_), "ResetFences(copy)");
        check(vkc::ResetCommandBuffer(cmd_, 0), "ResetCommandBuffer(copy)");
    }

    /// The memory type for device-local data (falls back to any type when no DEVICE_LOCAL-only
    /// exists — llvmpipe's memory is all host-visible AND device-local, which is exactly right).
    [[nodiscard]] std::uint32_t device_memory_type(std::uint32_t type_bits) const {
        for (std::uint32_t i = 0; i < memprops_.memoryTypeCount; ++i)
            if ((type_bits & (1u << i)) &&
                (memprops_.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
                return i;
        return host_memory_type(type_bits);
    }

    /// The pool cap: past this, released buffers are destroyed instead of parked. Big enough for
    /// a training loop's working set of temporaries, small enough to never squeeze the 8 GB card.
    static constexpr std::size_t kPoolCapBytes = std::size_t(1) << 30;  // 1 GiB

    void destroy_buffer(Buffer* b) {
        if (b->map != nullptr) vkc::UnmapMemory(device_, b->mem);
        vkc::DestroyBuffer(device_, b->buf, nullptr);
        vkc::FreeMemory(device_, b->mem, nullptr);
        delete b;
    }

    static void check(VkResult r, const char* what) {
        if (r != VK_SUCCESS)
            throw std::runtime_error(std::string("cheatah-gpu-linalg vulkan: ") + what +
                                     " failed (" + std::to_string(static_cast<int>(r)) + ")");
    }

    void create_instance() {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "cheatah-gpu-linalg";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &app;
        check(vkc::CreateInstance(&ci, nullptr, &instance_), "CreateInstance");
    }

    /// Highest-scoring device with a compute queue, unless CHEATAH_GPU_LINALG_VK_DEVICE forces
    /// one by name substring or zero-based index.
    void pick_physical_device() {
        std::uint32_t n = 0;
        check(vkc::EnumeratePhysicalDevices(instance_, &n, nullptr), "EnumeratePhysicalDevices");
        if (n == 0) throw std::runtime_error("cheatah-gpu-linalg vulkan: no devices");
        std::vector<VkPhysicalDevice> devs(n);
        check(vkc::EnumeratePhysicalDevices(instance_, &n, devs.data()),
              "EnumeratePhysicalDevices");

        const char* want = std::getenv("CHEATAH_GPU_LINALG_VK_DEVICE");
        int best_score = -1;
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::uint32_t family = compute_family(devs[i]);
            if (family == UINT32_MAX) continue;
            VkPhysicalDeviceProperties props{};
            vkc::GetPhysicalDeviceProperties(devs[i], &props);
            if (want && *want) {
                // Forced selection: a zero-based index, or a deviceName substring.
                char* end = nullptr;
                const long idx = std::strtol(want, &end, 10);
                const bool by_index = end && *end == '\0';
                if (by_index ? (static_cast<long>(i) == idx)
                             : (std::string(props.deviceName).find(want) != std::string::npos)) {
                    select(devs[i], family, props);
                    return;
                }
                continue;
            }
            int score = 0;
            switch (props.deviceType) {
                case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   score = 4; break;
                case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score = 3; break;
                case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    score = 2; break;
                case VK_PHYSICAL_DEVICE_TYPE_CPU:            score = 1; break;
                default:                                     score = 0; break;
            }
            if (score > best_score) {
                best_score = score;
                select(devs[i], family, props);
            }
        }
        if (physical_ == VK_NULL_HANDLE)
            throw std::runtime_error(want ? std::string("cheatah-gpu-linalg vulkan: no device "
                                                        "matches CHEATAH_GPU_LINALG_VK_DEVICE='") +
                                                want + "'"
                                          : std::string("cheatah-gpu-linalg vulkan: no device "
                                                        "with a compute queue"));
    }

    void select(VkPhysicalDevice d, std::uint32_t family, const VkPhysicalDeviceProperties& p) {
        physical_ = d;
        family_ = family;
        device_name_ = p.deviceName;
        limits_ = p.limits;
    }

    [[nodiscard]] static std::uint32_t compute_family(VkPhysicalDevice d) {
        std::uint32_t n = 0;
        vkc::GetPhysicalDeviceQueueFamilyProperties(d, &n, nullptr);
        std::vector<VkQueueFamilyProperties> fams(n);
        vkc::GetPhysicalDeviceQueueFamilyProperties(d, &n, fams.data());
        for (std::uint32_t i = 0; i < n; ++i)
            if (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return i;
        return UINT32_MAX;
    }

    void create_device() {
        VkPhysicalDeviceFeatures have{};
        vkc::GetPhysicalDeviceFeatures(physical_, &have);
        f64_ok_ = have.shaderFloat64 == VK_TRUE;

        // Tensor-core (KHR cooperative matrix) support needs three features + one extension;
        // probe them all, enable what exists, and record coop_ok_ for kernel gating.
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_have{};
        coop_have.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        VkPhysicalDeviceVulkan12Features v12_have{};
        v12_have.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        v12_have.pNext = &coop_have;
        VkPhysicalDeviceFeatures2 have2{};
        have2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        have2.pNext = &v12_have;
        vkc::GetPhysicalDeviceFeatures2(physical_, &have2);
        std::uint32_t next = 0;
        vkc::EnumerateDeviceExtensionProperties(physical_, nullptr, &next, nullptr);
        std::vector<VkExtensionProperties> exts(next);
        vkc::EnumerateDeviceExtensionProperties(physical_, nullptr, &next, exts.data());
        bool has_coop_ext = false;
        for (const auto& e : exts)
            if (std::string(e.extensionName) == "VK_KHR_cooperative_matrix") has_coop_ext = true;
        coop_ok_ = has_coop_ext && coop_have.cooperativeMatrix == VK_TRUE &&
                   v12_have.shaderFloat16 == VK_TRUE && v12_have.vulkanMemoryModel == VK_TRUE;

        VkPhysicalDeviceFeatures want{};
        want.shaderFloat64 = f64_ok_ ? VK_TRUE : VK_FALSE;
        VkPhysicalDeviceVulkan12Features v12_want{};
        v12_want.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_want{};
        coop_want.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        std::vector<const char*> dev_exts;
        for (const VkExtensionProperties& e : exts)
            if (std::strcmp(e.extensionName, "VK_KHR_pipeline_executable_properties") == 0)
                exec_stats_ok_ = true;
        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR exec_want{};
        exec_want.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
        exec_want.pipelineExecutableInfo = VK_TRUE;
        if (exec_stats_ok_) dev_exts.push_back("VK_KHR_pipeline_executable_properties");
        if (coop_ok_) {
            v12_want.shaderFloat16 = VK_TRUE;
            v12_want.vulkanMemoryModel = VK_TRUE;
            v12_want.vulkanMemoryModelDeviceScope = v12_have.vulkanMemoryModelDeviceScope;
            coop_want.cooperativeMatrix = VK_TRUE;
            v12_want.pNext = &coop_want;
            dev_exts.push_back("VK_KHR_cooperative_matrix");
        }

        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = family_;
        qi.queueCount = 1;
        qi.pQueuePriorities = &prio;
        VkDeviceCreateInfo di{};
        di.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        if (exec_stats_ok_) {
            // Chain the exec-properties feature ahead of whatever else is chained.
            exec_want.pNext = coop_ok_ ? static_cast<void*>(&v12_want) : nullptr;
            di.pNext = &exec_want;
        } else {
            di.pNext = coop_ok_ ? static_cast<const void*>(&v12_want) : nullptr;
        }
        di.queueCreateInfoCount = 1;
        di.pQueueCreateInfos = &qi;
        di.enabledExtensionCount = static_cast<std::uint32_t>(dev_exts.size());
        di.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();
        di.pEnabledFeatures = &want;
        check(vkc::CreateDevice(physical_, &di, nullptr, &device_), "CreateDevice");
        vkc::GetDeviceQueue(device_, family_, 0, &queue_);
        vkc::GetPhysicalDeviceMemoryProperties(physical_, &memprops_);
    }

    void create_descriptor_machinery() {
        // One layout serves every kernel: kMaxBindings storage-buffer slots, compute stage. A
        // pipeline whose shader uses fewer bindings is valid against the larger layout.
        VkDescriptorSetLayoutBinding bindings[kMaxBindings]{};
        for (std::uint32_t i = 0; i < kMaxBindings; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = kMaxBindings;
        li.pBindings = bindings;
        check(vkc::CreateDescriptorSetLayout(device_, &li, nullptr, &set_layout_),
              "CreateDescriptorSetLayout");

        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &set_layout_;
        check(vkc::CreatePipelineLayout(device_, &pli, nullptr, &pipe_layout_),
              "CreatePipelineLayout");

        VkDescriptorPoolSize size{};
        size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        size.descriptorCount = kMaxBindings * (1 + kMaxFuse);
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.maxSets = 1 + kMaxFuse;
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &size;
        check(vkc::CreateDescriptorPool(device_, &pi, nullptr, &pool_), "CreateDescriptorPool");

        // ONE descriptor set for the context's lifetime: dispatches are serial and blocking, so
        // the set is never in flight when rewritten — per-dispatch pool reset + set allocation
        // was pure churn.
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = pool_;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &set_layout_;
        check(vkc::AllocateDescriptorSets(device_, &dsai, &set_), "AllocateDescriptorSets");
        for (unsigned i = 0; i < kMaxFuse; ++i)
            check(vkc::AllocateDescriptorSets(device_, &dsai, &fuse_sets_[i]),
                  "AllocateDescriptorSets");
    }

    void create_command_machinery() {
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = family_;
        check(vkc::CreateCommandPool(device_, &ci, nullptr, &cmd_pool_), "CreateCommandPool");
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = cmd_pool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        check(vkc::AllocateCommandBuffers(device_, &ai, &cmd_), "AllocateCommandBuffers");
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        check(vkc::CreateFence(device_, &fi, nullptr, &fence_), "CreateFence");
    }

    [[nodiscard]] std::uint32_t host_memory_type(std::uint32_t type_bits,
                                                 bool cached_preferred = false) const {
        const VkMemoryPropertyFlags base =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if (cached_preferred) {
            // Readback staging: CPU READS from write-combined memory crawl (~0.2 GB/s), so
            // prefer a HOST_CACHED type when the device offers one.
            const VkMemoryPropertyFlags cached = base | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            for (std::uint32_t i = 0; i < memprops_.memoryTypeCount; ++i)
                if ((type_bits & (1u << i)) &&
                    (memprops_.memoryTypes[i].propertyFlags & cached) == cached)
                    return i;
        }
        for (std::uint32_t i = 0; i < memprops_.memoryTypeCount; ++i)
            if ((type_bits & (1u << i)) &&
                (memprops_.memoryTypes[i].propertyFlags & base) == base)
                return i;
        throw std::runtime_error("cheatah-gpu-linalg vulkan: no host-visible coherent memory");
    }

    /// The SPIR-V bytes for kernel `name`: `<spv dir>/<name>.spv`. Resolution order: the
    /// `CHEATAH_GPU_LINALG_SPV_DIR` environment variable, the same-named compile definition,
    /// then this repo's own `build/shaders` (derived from __FILE__ — so a consumer that passes
    /// NO definitions still finds the kernels of the checkout it compiled against).
    [[nodiscard]] static std::vector<char> spv_bytes(const std::string& name) {
        const char* env = std::getenv("CHEATAH_GPU_LINALG_SPV_DIR");
#if defined(CHEATAH_GPU_LINALG_SPV_DIR)
        const std::string dir = env ? env : CHEATAH_GPU_LINALG_SPV_DIR;
#else
        const std::string dir =
            env ? std::string(env)
                : (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
                   "build" / "shaders")
                      .string();
#endif
        std::ifstream in(dir + "/" + name + ".spv", std::ios::binary | std::ios::ate);
        if (!in)
            throw std::runtime_error("cheatah-gpu-linalg vulkan: missing SPIR-V for kernel '" +
                                     name + "'");
        if (in.tellg() > std::streampos(256u << 20))
            throw std::runtime_error("cheatah-gpu-linalg vulkan: implausibly large SPIR-V blob: " +
                                     name);
        std::vector<char> bytes(static_cast<std::size_t>(in.tellg()));
        in.seekg(0);
        in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        return bytes;
    }

    /// The compute pipeline for `name`, built from its .spv once and cached. slangc renames each
    /// module's single entry point to "main".
    VkPipeline pipeline(const std::string& name) {
        if (auto it = pipelines_.find(name); it != pipelines_.end()) return it->second;
        if (!coop_ok_ && name.find("coop") != std::string::npos)
            throw std::runtime_error("cheatah-gpu-linalg vulkan: device '" + device_name_ +
                                     "' has no VK_KHR_cooperative_matrix — kernel '" + name +
                                     "' unavailable");
        const bool needs_f64 =
            (name.size() > 4 && name.compare(name.size() - 4, 4, "_f64") == 0) ||
            (name.size() > 5 && name.compare(name.size() - 5, 5, "_c128") == 0);
        if (!f64_ok_ && needs_f64)
            throw std::runtime_error("cheatah-gpu-linalg vulkan: device '" + device_name_ +
                                     "' has no shaderFloat64 — kernel '" + name +
                                     "' unavailable");
        if (std::string_view(name).find('/') != std::string_view::npos)
            throw std::runtime_error("cheatah-gpu-linalg vulkan: kernel names cannot contain '/'");
        const std::vector<char> code = spv_bytes(name);
        VkShaderModuleCreateInfo mi{};
        mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        mi.codeSize = code.size();
        mi.pCode = reinterpret_cast<const std::uint32_t*>(code.data());
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkc::CreateShaderModule(device_, &mi, nullptr, &mod), "CreateShaderModule");

        VkComputePipelineCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        if (exec_stats_ok_ && std::getenv("CHEATAH_GPU_PIPELINE_STATS") != nullptr)
            pi.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
        pi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pi.stage.module = mod;
        pi.stage.pName = "main";
        pi.layout = pipe_layout_;
        VkPipeline pipe = VK_NULL_HANDLE;
        check(vkc::CreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipe),
              "CreateComputePipelines");
        vkc::DestroyShaderModule(device_, mod, nullptr);
        pipelines_.emplace(name, pipe);
        return pipe;
    }

    /// Record + submit one dispatch of `groups` workgroups and block on its fence. The context's
    /// single persistent descriptor set is rewritten in place (dispatches are serial and blocking
    /// by design — cheatah's numerics are single-threaded on purpose — so it is never in flight).
    void submit(const char* name, Buffer** buffers, unsigned count,
                cheatah::gpu::dispatch::Dim3 groups) {
        if (count > kMaxBindings)
            throw std::runtime_error("cheatah-gpu-linalg vulkan: too many buffer bindings");
        // Vulkan guarantees ≥ 65535 per axis; exceeding the device limit is undefined — check.
        if (groups.x > limits_.maxComputeWorkGroupCount[0] ||
            groups.y > limits_.maxComputeWorkGroupCount[1] ||
            groups.z > limits_.maxComputeWorkGroupCount[2])
            throw std::runtime_error(std::string("cheatah-gpu-linalg vulkan: dispatch of '") +
                                     name + "' exceeds maxComputeWorkGroupCount");
        VkPipeline pipe = pipeline(name);
        ++stats_.dispatches;

        if (fuse_active_) {
            if (fuse_count_ >= kMaxFuse)
                throw std::runtime_error("cheatah-gpu-linalg vulkan: fuse chain too long");
            VkDescriptorSet fset = fuse_sets_[fuse_count_];
            VkDescriptorBufferInfo finfos[kMaxBindings]{};
            VkWriteDescriptorSet fwrites[kMaxBindings]{};
            for (unsigned i = 0; i < count; ++i) {
                finfos[i].buffer = buffers[i]->buf;
                finfos[i].range = VK_WHOLE_SIZE;
                fwrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                fwrites[i].dstSet = fset;
                fwrites[i].dstBinding = i;
                fwrites[i].descriptorCount = 1;
                fwrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                fwrites[i].pBufferInfo = &finfos[i];
            }
            vkc::UpdateDescriptorSets(device_, count, fwrites, 0, nullptr);
            if (fuse_count_ == 0) {
                VkCommandBufferBeginInfo fbi{};
                fbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                fbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                check(vkc::BeginCommandBuffer(cmd_, &fbi), "BeginCommandBuffer");
            } else {
                VkMemoryBarrier mb{};
                mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkc::CmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                                        nullptr, 0, nullptr);
            }
            vkc::CmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkc::CmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 1,
                                       &fset, 0, nullptr);
            vkc::CmdDispatch(cmd_, groups.x, groups.y, groups.z);
            ++fuse_count_;
            return;
        }

        VkDescriptorSet set = set_;   // the persistent set — rewritten, never reallocated

        VkDescriptorBufferInfo infos[kMaxBindings]{};
        VkWriteDescriptorSet writes[kMaxBindings]{};
        for (unsigned i = 0; i < count; ++i) {
            infos[i].buffer = buffers[i]->buf;
            infos[i].range = VK_WHOLE_SIZE;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkc::UpdateDescriptorSets(device_, count, writes, 0, nullptr);

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkc::BeginCommandBuffer(cmd_, &bi), "BeginCommandBuffer");
        vkc::CmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkc::CmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 1, &set,
                                   0, nullptr);
        vkc::CmdDispatch(cmd_, groups.x, groups.y, groups.z);
        check(vkc::EndCommandBuffer(cmd_), "EndCommandBuffer");

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        check(vkc::QueueSubmit(queue_, 1, &si, fence_), "QueueSubmit");
        check(vkc::WaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "WaitForFences");
        check(vkc::ResetFences(device_, 1, &fence_), "ResetFences");
        check(vkc::ResetCommandBuffer(cmd_, 0), "ResetCommandBuffer");
    }

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memprops_{};
    std::uint32_t family_ = UINT32_MAX;
    std::string device_name_;
    VkPhysicalDeviceLimits limits_{};
    bool f64_ok_ = false;
    bool coop_ok_ = false;
    bool exec_stats_ok_ = false;
    TransferStats stats_{};
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;   // the one persistent set (see submit)
    static constexpr unsigned kMaxFuse = 4;
    VkDescriptorSet fuse_sets_[kMaxFuse] = {};   // per-dispatch sets of an in-flight fuse chain
    bool fuse_active_ = false;
    unsigned fuse_count_ = 0;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd2_ = VK_NULL_HANDLE;   // the replayable probe command buffer
    VkFence fence_ = VK_NULL_HANDLE;
    std::unordered_map<std::string, VkPipeline> pipelines_;
    std::unordered_map<std::size_t, std::vector<Buffer*>> free_lists_;  // pooled-buffer size classes
    std::vector<Buffer*> cached_bufs_;  // content-cache-owned scratch (dims), freed at teardown
    std::size_t pooled_bytes_ = 0;
};

/// The one shared context. First call constructs the device; later calls reuse it.
inline Context& ctx() {
    static Context c;
    return c;
}

}  // namespace detail
}  // namespace cheatah::gpu::linalg
