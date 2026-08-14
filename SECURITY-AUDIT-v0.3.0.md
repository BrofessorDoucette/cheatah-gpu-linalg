# cheatah-gpu-linalg — Security Audit, v0.3.0-alpha

Audited 2026-07-17 (the round-6 audit), full surface: the device-array container, both
backend contexts, every transfer path, the dispatch layer, and the loader surfaces.

## Threat model

Single-trust (the cheatah household standard): the embedding process is trusted, but its
inputs may still be wrong — robustness bugs at the API boundary are treated as security
bugs. The GPU adds two twists: 32-bit index arithmetic inside kernels, and driver behavior
that is undefined (not merely wrong) when limits are exceeded.

## Findings and the fixes shipped by v0.3.0-alpha

| # | Severity | Area | Finding | Fix |
|---|---|---|---|---|
| 1 | **High** | `device_array` shapes | element-count product could overflow, under-allocating before 32-bit kernel indexing (`row*N+col`) wrapped | shape products overflow-checked; arrays capped at 2³²−1 elements, making kernel index arithmetic provably non-wrapping; `reshape` products get the same check |
| 2 | **High** | transfers | an overflowed byte count meant a silent heap overflow on first `upload` | `upload`/`download`/`download_at` validate byte counts and offsets against the underlying allocation, both backends |
| 3 | **Medium** | dispatch | workgroup counts beyond `maxComputeWorkGroupCount` are undefined behavior in Vulkan | every dispatch validates counts against the device limit |
| 4 | **Medium** | allocator | kernels write `ceil(n/4)` vec4s — a tight allocation could be exited | pooled size classes are powers of two ≥ 256 B (the vec4-tail invariant), documented and enforced at the allocator |
| 5 | Low | loader | unbounded SPIR-V blobs; kernel names usable for path traversal | blobs size-capped (256 MB); kernel names reject path separators; `size_class` rejects overflow-inducing sizes |
| 6 | Low | fusing | an allocation throw mid-fuse could leave a half-recorded submission | `begin_fuse` rejects nesting; call sites allocate before fusing |

## Evidence

QA gate (`scripts/qa.sh`): both backends + a forced-llvmpipe lane, ASan+UBSan test pass,
Valgrind memcheck over the emulated-Metal binaries — all green as of the audit and required
green ever since.

## Residual and non-goals

Driver and GPU firmware are trusted (there is no defense here against a malicious ICD).
Kernel outputs are not constant-time and carry no secrecy claims — this is numerics, not
cryptography.
