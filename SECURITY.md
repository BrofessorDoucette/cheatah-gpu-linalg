# Security

## Threat model

Single-trust, the cheatah household standard: the process that links this library is
trusted; its *inputs* may still be hostile or wrong. Robustness failures at the API
boundary — overflowing shapes, out-of-range transfers, limit-exceeding dispatches — are
treated as security bugs, because on a GPU the failure mode is undefined behavior in the
driver, not a clean crash.

## Standing security review

| Area | Finding | Status |
|------|---------|--------|
| `device_array` shape math | element-count overflow → under-allocation → out-of-bounds kernel writes | **Fixed (v0.3.0)** — overflow-checked products, 2³²−1 element cap makes 32-bit kernel indexing non-wrapping |
| transfer bounds | byte count/offset unchecked against the allocation → heap overflow on upload | **Fixed (v0.3.0)** — validated on both backends, `download_at` included |
| dispatch limits | workgroup counts past `maxComputeWorkGroupCount` = undefined behavior | **Fixed (v0.3.0)** — validated per dispatch |
| allocator vec4 tail | `ceil(n/4)`-vec4 kernel writes could exit a tight allocation | **Fixed (v0.3.0)** — power-of-two size classes ≥ 256 B, invariant documented and enforced |
| loader surfaces | unbounded SPIR-V blobs; kernel-name path traversal | **Fixed (v0.3.0)** — 256 MB cap; names reject path separators |
| submission fusing | allocation throw mid-fuse → half-recorded submission | **Fixed (v0.3.0)** — nesting rejected; allocate-before-fuse discipline |
| driver / ICD | a malicious Vulkan ICD or Metal driver owns the process | **By design** — the driver is inside the trust boundary |
| timing | kernels are not constant-time | **By design** — numerics, no secrecy claims |

The full v0.3.0-alpha audit is in [SECURITY-AUDIT-v0.3.0.md](SECURITY-AUDIT-v0.3.0.md).

## Reporting

Open a GitHub issue for anything without exploitation impact. For something sensitive,
email the maintainer (see NOTICE) before filing publicly.
