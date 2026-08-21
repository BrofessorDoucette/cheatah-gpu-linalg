<!-- cheatah-bench-stamp v1
     suite:        gpu-host-device-crossover
     generated:    2026-08-20
     commit:       70dbb00 (dirty)
     gpu:          NVIDIA GeForce RTX 3070 Ti Laptop GPU (driver 580.159.03)
     host:         12th Gen Intel(R) Core(TM) i7-12700H, 20 CPUs, Linux 7.0.11-76070011-generic
     competitors:  none — cheatah host arrays against cheatah device arrays
     harness:      paired host/device sweeps, same element type; medians over interleaved reps
     statistic:    median real_time; crossover is the first size where the device wins
     watch:        include/, kernels/, bench/crossover.py
     publishable:  true

     PRODUCED BY:
       python3 bench/crossover.py --out docs/bench/gpu-host-device-crossover.md
-->

| op | size | crossover (device wins at) | host @ crossover | device @ crossover |
|---|---|---|---|---|
| matmul (double) | n (matrix is n x n) | **n ≥ 64** | 203 µs | 115 µs |
| dot (double) | n (elements) | **n ≥ 262,144** | 444 µs | 41 µs |
| sum (double) | n (elements) | **n ≥ 32,768** | 51 µs | 49 µs |
| add (double) | n (elements) | **n ≥ 4,096** | 35 µs | 24 µs |
| axpy (double) | n (elements) | **n ≥ 32,768** | 41 µs | 31 µs |
