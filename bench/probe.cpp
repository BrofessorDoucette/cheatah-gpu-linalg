// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
// probe.cpp — the cross-implementation correctness CLI compare.py drives BEFORE timing anything:
//   gpu_linalg_probe_<backend> <op> <n> <dtype> <salt>
// fills operands with the shared deterministic sequence, runs the device op through cheatah's
// front, and prints "checksum <value>" (a plain sum of the result elements) plus the first three
// elements — numpy/torch regenerate identical operands and must agree before a single timing row
// is published. Ops: matmul (n x n), dot, sum, add, axpy, transpose.
#include "cheatah_gpu_linalg/cheatah_gpu_linalg.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace gl = cheatah::gpu::linalg;
namespace hl = cheatah::linalg;

namespace {

double fill(std::size_t i, std::size_t salt) {
    return 0.25 * static_cast<double>((i * 7 + salt) % 16) - 1.0;
}

template <class T>
std::vector<T> seq(std::size_t n, std::size_t salt) {
    std::vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<T>(fill(i, salt));
    return v;
}

template <class T>
int run(const std::string& op, std::size_t n, std::size_t salt) {
    if (op == "matmul") {
        const std::vector<T> av = seq<T>(n * n, salt), bv = seq<T>(n * n, salt + 1);
        gl::device_array<T> a = gl::device_array<T>::from_host({n, n}, av.data());
        gl::device_array<T> b = gl::device_array<T>::from_host({n, n}, bv.data());
        std::vector<T> out(n * n);
        hl::matmul(a, b).to_host(out.data());
        double sum = 0;
        for (T x : out) sum += static_cast<double>(x);
        std::printf("checksum %.9e first %.9e %.9e %.9e\n", sum, double(out[0]), double(out[1]),
                    double(out[2]));
        return 0;
    }
    if (op == "transpose") {
        const std::vector<T> av = seq<T>(n * n, salt);
        gl::device_array<T> a = gl::device_array<T>::from_host({n, n}, av.data());
        std::vector<T> out(n * n);
        hl::conj_transpose(a).to_host(out.data());
        double sum = 0;
        for (T x : out) sum += static_cast<double>(x);
        std::printf("checksum %.9e first %.9e %.9e %.9e\n", sum, double(out[0]), double(out[1]),
                    double(out[2]));
        return 0;
    }
    const std::vector<T> av = seq<T>(n, salt), bv = seq<T>(n, salt + 1);
    gl::device_array<T> a = gl::device_array<T>::from_host({n}, av.data());
    gl::device_array<T> b = gl::device_array<T>::from_host({n}, bv.data());
    if (op == "dot") {
        T d;
        gl::dot(d, a, b);
        std::printf("checksum %.9e first %.9e 0 0\n", double(d), double(d));
        return 0;
    }
    if (op == "sum") {
        std::printf("checksum %.9e first 0 0 0\n", double(gl::sum(a)));
        return 0;
    }
    if (op == "add" || op == "axpy") {
        std::vector<T> out(n);
        if (op == "add") gl::add(a, b).to_host(out.data());
        else gl::axpy(T(2.5), a, b).to_host(out.data());
        double sum = 0;
        for (T x : out) sum += static_cast<double>(x);
        std::printf("checksum %.9e first %.9e %.9e %.9e\n", sum, double(out[0]), double(out[1]),
                    double(out[2]));
        return 0;
    }
    std::fprintf(stderr, "unknown op '%s'\n", op.c_str());
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <op> <n> <f32|f64> <salt>\n", argv[0]);
        return 2;
    }
    const std::string op = argv[1];
    const std::size_t n = std::strtoull(argv[2], nullptr, 10);
    const std::string dtype = argv[3];
    const std::size_t salt = std::strtoull(argv[4], nullptr, 10);
    if (dtype == "f32") return run<float>(op, n, salt);
    if (dtype == "f64") return run<double>(op, n, salt);
    std::fprintf(stderr, "unknown dtype '%s'\n", dtype.c_str());
    return 2;
}
