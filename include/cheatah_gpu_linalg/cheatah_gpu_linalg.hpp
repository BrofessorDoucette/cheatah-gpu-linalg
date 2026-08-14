// Copyright (c) 2026 BigBrain LLC. MIT-licensed (see LICENSE).
// Original work; see ACKNOWLEDGMENTS.md for the open-source ideas we build upon.
#pragma once

/**
 * @file cheatah_gpu_linalg.hpp
 * @brief Umbrella header for cheatah-gpu-linalg — GPU-accelerated linear algebra for cheatah.
 *
 * A device backend for cheatah's `stdlib/linalg`: it supplies a GPU-resident `device_array<T>` and
 * device kernels that the linalg fronts dispatch to automatically. Build cheatah's linalg over
 * `device_array` instead of `basic_ndarray` and the operation runs on the GPU (Metal today; real on
 * Apple, software-emulated elsewhere via cheatah-gpu). Include this and use `cheatah::linalg::*`.
 */

#include "cheatah_gpu_linalg/bridge.hpp"
#include "cheatah_gpu_linalg/conv.hpp"
#include "cheatah_gpu_linalg/device_array.hpp"
#include "cheatah_gpu_linalg/elementwise.hpp"
#include "cheatah_gpu_linalg/factories.hpp"
#include "cheatah_gpu_linalg/routines.hpp"
