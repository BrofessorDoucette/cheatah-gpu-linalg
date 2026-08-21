<!-- cheatah-bench-stamp v1
     suite:        gpu-device-throughput
     generated:    2026-08-20
     commit:       360b18c
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
| matmul | 1024³ | 337 µs | 6.37 TFLOP/s |
| matmul | 2048³ | 1.90 ms | 9.05 TFLOP/s |
| matmul | 4096³ | 14.44 ms | **9.52 TFLOP/s** |
| matmul f16 (tensor cores, opt-in) | 4096³ | 6.33 ms | **21.70 TFLOP/s** |
| matmul f64 | 1024³ | 8.02 ms | 268 GFLOP/s |
| dot | 16M | 436 µs | 308 GB/s |
| sum | 16M | 217 µs | 310 GB/s |
| add | 16M | 567 µs | 355 GB/s |
| axpy (fused) | 16M | 581 µs | 347 GB/s |
