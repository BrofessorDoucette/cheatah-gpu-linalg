<!-- cheatah-bench-stamp v1
     suite:        gpu-device-throughput
     generated:    2026-08-20
     commit:       70dbb00 (dirty)
     gpu:          NVIDIA GeForce RTX 3070 Ti Laptop GPU (driver 580.159.03)
     host:         12th Gen Intel(R) Core(TM) i7-12700H, 20 CPUs, Linux 7.0.11-76070011-generic
     competitors:  none — device-resident cheatah kernels against the hardware's own ceilings
     harness:      reps=5, min_time=0.2s, random-interleaving=on
     statistic:    median real_time per case; rates derived from the standard FLOP/byte counts
     watch:        include/, kernels/, bench/gpu_linalg_bench.cpp
     publishable:  true

     PRODUCED BY:
       python3 bench/emit_tables.py --out docs/bench/gpu-device-throughput.md
-->

| op | size | wall | rate |
|---|---|---|---|
| matmul | 1024³ | 337 µs | 6.38 TFLOP/s |
| matmul | 2048³ | 1.91 ms | 9.01 TFLOP/s |
| matmul | 4096³ | 14.57 ms | **9.43 TFLOP/s** |
| matmul f16 (tensor cores, opt-in) | 4096³ | 6.38 ms | **21.54 TFLOP/s** |
| matmul f64 | 1024³ | 8.05 ms | 267 GFLOP/s |
| dot | 16M | 404 µs | 332 GB/s |
| sum | 16M | 222 µs | 302 GB/s |
| add | 16M | 640 µs | 315 GB/s |
| axpy (fused) | 16M | 616 µs | 327 GB/s |
