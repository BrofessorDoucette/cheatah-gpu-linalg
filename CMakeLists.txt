cmake_minimum_required(VERSION 3.24)

# cheatah-gpu-linalg — GPU-accelerated linear algebra for cheatah, on BOTH backends.
#
# A device backend for cheatah's stdlib/linalg: a GPU-resident device_array<T> plus device kernels
# that cheatah's linalg fronts dispatch to by ADL. The kernels are written ONCE in Slang
# (kernels/linalg.slang) and compiled here at build time:
#   - SPIR-V per entry point for the VULKAN backend (runs on real GPUs and llvmpipe), and
#   - Metal Shading Language for the METAL backend on Apple (runtime-compiled); off Apple the
#     Metal backend runs on cheatah-gpu's software-emulated device via C++ stand-ins.
# Header-only C++; every test builds twice — once per backend — so the same seam is proven on
# real Vulkan hardware AND on the emulated Metal flow, on this one machine.
project(cheatah_gpu_linalg
    VERSION 0.4.1
    DESCRIPTION "GPU-accelerated linear algebra for cheatah (device backend for stdlib/linalg)"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Sibling repos this extension consumes ---------------------------------
set(CHEATAH_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/../cheatah"
    CACHE PATH "cheatah repo (provides stdlib/linalg + stdlib/ndarray headers)")
set(CHEATAH_GPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cheatah-gpu"
    CACHE PATH "cheatah-gpu repo (vk.*/mtl.* surfaces, gpu.dispatch math, emulated Metal device)")

# metal-cpp: Apple's MIT-licensed header-only C++ interface to Metal. Vendored inside cheatah-gpu by
# default; override with -DCHEATAH_GPU_METAL_CPP=/path or the same-named env var. No downloads.
if(DEFINED ENV{CHEATAH_GPU_METAL_CPP})
    set(CHEATAH_GPU_METAL_CPP "$ENV{CHEATAH_GPU_METAL_CPP}" CACHE PATH "metal-cpp headers")
else()
    set(CHEATAH_GPU_METAL_CPP "${CHEATAH_GPU_DIR}/third_party/metal-cpp" CACHE PATH "metal-cpp headers")
endif()

# ---- The Vulkan SDK (headers new enough for cheatah-gpu's generated surface, slangc, loader) ----
# $VULKAN_SDK (the SDK's setup-env.sh) wins; otherwise the newest ~/Tools/vulkan-sdk install — the
# same discovery cheatah-gpu's gates use. The distro's vulkan headers are often too old.
if(DEFINED ENV{VULKAN_SDK})
    set(_vk_root "$ENV{VULKAN_SDK}")
else()
    file(GLOB _vk_candidates "$ENV{HOME}/Tools/vulkan-sdk/*/x86_64" "$ENV{HOME}/VulkanSDK/*/x86_64")
    list(SORT _vk_candidates)
    list(POP_BACK _vk_candidates _vk_root)
endif()
set(CHEATAH_GPU_LINALG_VK_ROOT "${_vk_root}" CACHE PATH "Vulkan SDK root (include/, lib/, bin/slangc)")

# ---- volk (the Vulkan meta-loader) -----------------------------------------
# volk dlopens the loader at runtime, so a consumer binary carries NO DT_NEEDED on
# libvulkan.so.1 — the property that lets a purrc-dlopened module reach the GPU without the
# LD_PRELOAD workaround (the loader's ELF ctor crashes inside a nested dlopen). The source
# comes from cheatah-gpu's vendored fetch (its Vulkan backend pins volk 1.4.350); no network
# from this repo. Absent → the legacy -lvulkan link with a warning, so a fresh machine still
# builds before cheatah-gpu has fetched anything.
file(GLOB _volk_candidates "${CHEATAH_GPU_DIR}/build/*/_deps/volk-src")
list(SORT _volk_candidates)
list(POP_BACK _volk_candidates _volk_default)
set(CHEATAH_GPU_LINALG_VOLK_SRC "${_volk_default}" CACHE PATH "volk checkout (volk.h + volk.c)")

find_program(CHEATAH_GPU_LINALG_SLANGC slangc
    HINTS "${CHEATAH_GPU_LINALG_VK_ROOT}/bin"
    DOC "Slang compiler (ships in the Vulkan SDK)")
if(NOT CHEATAH_GPU_LINALG_SLANGC)
    message(FATAL_ERROR "cheatah-gpu-linalg: slangc not found — the kernels are Slang; install the "
                        "Vulkan SDK or set CHEATAH_GPU_LINALG_SLANGC.")
endif()

# ---- Compile the Slang kernels: one .spv per entry x element, one .metal per f32 entry ---------
set(_shader_dir "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${_shader_dir}")
set(_slang_src "${CMAKE_CURRENT_SOURCE_DIR}/kernels/linalg.slang")
# Every kernel entry as "name|guard|extra-defines" (extra defines comma-separated). The elementwise
# skeletons (ew_binary/ews_binary/ew_unary) are compiled once per operator via EW_KIND_*/EW_NAME.
set(_shader_specs
    "gemm_fast|KERNEL_gemm_fast|"
    "gemm_fast64|KERNEL_gemm_fast64||f32only"
    "gemm_attr_nobar|KERNEL_gemm_attr|ATTR_NAME=gemm_attr_nobar,ATTR_NOBAR|f32only"
    "gemm_attr_noglobal|KERNEL_gemm_attr|ATTR_NAME=gemm_attr_noglobal,ATTR_NOGLOBAL|f32only"
    "gemm_attr_noepi|KERNEL_gemm_attr|ATTR_NAME=gemm_attr_noepi,ATTR_NOEPI|f32only"
    "gemm_coop|KERNEL_gemm_coop||f32only|spvonly"
    "gemm_coop_f16acc|KERNEL_gemm_coop_f16acc||f32only|spvonly"
    "coop_peak_f32acc|KERNEL_coop_peak|ACC_T=float,CP_NAME=coop_peak_f32acc|f32only|spvonly"
    "coop_peak_f16acc|KERNEL_coop_peak|ACC_T=float16_t,CP_NAME=coop_peak_f16acc|f32only|spvonly"
    "ffma_peak|KERNEL_ffma_peak||f32only"
    "lds_peak|KERNEL_lds_peak||f32only"
    "gemm_edge|KERNEL_gemm_edge||complex"
    "gemm_batched|KERNEL_gemm_batched||complex"
    "outer|KERNEL_outer||complex"
    "transpose|KERNEL_transpose||complex"
    "kron|KERNEL_kron||complex"
    "dot_partial|KERNEL_dot_partial||complex"
    "vdot_partial|KERNEL_dot_partial|CONJ_FIRST,DOT_NAME=vdot_partial|conly"
    "dot_partial2|KERNEL_dot_partial2||complex"
    "vdot_partial2|KERNEL_dot_partial2|CONJ_FIRST,DOT_NAME2=vdot_partial2|conly"
    "sum_partial2|KERNEL_sum_partial2|"
    "finalize_partials|KERNEL_finalize_partials||complex"
    "trace_partial|KERNEL_trace_partial||complex"
    "sum_partial|KERNEL_sum_partial|"
    "axpy|KERNEL_axpy|"
    "im2col2d|KERNEL_im2col2d|"
    "col2im2d|KERNEL_col2im2d|"
    "conv_bias_act|KERNEL_conv_bias_act|"
    "conv_act_grad|KERNEL_conv_act_grad|"
    "fill|KERNEL_fill|"
    "copy|KERNEL_copy||f32only"
    "triad|KERNEL_triad||f32only"
    "ew_add|KERNEL_ew_binary|EW_KIND_ADD,EW_NAME=ew_add"
    "ew_sub|KERNEL_ew_binary|EW_KIND_SUB,EW_NAME=ew_sub"
    "ew_mul|KERNEL_ew_binary|EW_KIND_MUL,EW_NAME=ew_mul"
    "ew_div|KERNEL_ew_binary|EW_KIND_DIV,EW_NAME=ew_div"
    "ews_add|KERNEL_ews_binary|EW_KIND_ADD,EW_NAME=ews_add"
    "ews_sub|KERNEL_ews_binary|EW_KIND_SUB,EW_NAME=ews_sub"
    "ews_mul|KERNEL_ews_binary|EW_KIND_MUL,EW_NAME=ews_mul"
    "ews_div|KERNEL_ews_binary|EW_KIND_DIV,EW_NAME=ews_div"
    "ew_neg|KERNEL_ew_unary|EW_KIND_NEG,EW_NAME=ew_neg"
    "ew_abs|KERNEL_ew_unary|EW_KIND_ABS,EW_NAME=ew_abs"
    "ew_sqrt|KERNEL_ew_unary|EW_KIND_SQRT,EW_NAME=ew_sqrt"
    "ew_exp|KERNEL_ew_unary|EW_KIND_EXP,EW_NAME=ew_exp|f32only"
    "ew_log|KERNEL_ew_unary|EW_KIND_LOG,EW_NAME=ew_log|f32only")
set(_shader_outputs "")
foreach(_spec IN LISTS _shader_specs)
    string(REPLACE "|" ";" _parts "${_spec}")
    list(GET _parts 0 _op)
    list(GET _parts 1 _guard)
    list(LENGTH _parts _np)
    set(_extra "")
    if(_np GREATER 2)
        list(GET _parts 2 _extras_raw)
        if(NOT _extras_raw STREQUAL "")
            string(REPLACE "," ";" _extras "${_extras_raw}")
            foreach(_e IN LISTS _extras)
                list(APPEND _extra "-D${_e}")
            endforeach()
        endif()
    endif()
    # f32only ops: SPIR-V's GLSL.std.450 transcendentals (exp/log) exist only for 16/32-bit
    # floats — an f64 module would be invalid (NVIDIA rejects it; llvmpipe miscompiles it). The
    # C++ front evaluates those on the host for double instead.
    set(_elems "float:_f32" "double:_f64")
    if("f32only" IN_LIST _parts)
        set(_elems "float:_f32")                       # no f64 module exists (see above)
    elseif("complex" IN_LIST _parts)
        list(APPEND _elems "float2:_c64" "double2:_c128")   # + complex variants (-DCOMPLEX)
    elseif("conly" IN_LIST _parts)
        set(_elems "float2:_c64" "double2:_c128")      # complex-ONLY (e.g. the conjugating vdot)
    endif()
    foreach(_elem IN LISTS _elems)
        string(REPLACE ":" ";" _elem "${_elem}")
        list(GET _elem 0 _ctype)
        list(GET _elem 1 _suffix)
        set(_cplx "")
        if(_suffix STREQUAL "_c64" OR _suffix STREQUAL "_c128")
            set(_cplx "-DCOMPLEX")
        endif()
        set(_spv "${_shader_dir}/${_op}${_suffix}.spv")
        add_custom_command(OUTPUT "${_spv}"
            COMMAND "${CHEATAH_GPU_LINALG_SLANGC}" "${_slang_src}"
                    -D${_guard} ${_extra} ${_cplx} -DELEM=${_ctype} -DSUFFIX=${_suffix}
                    -target spirv -entry ${_op}${_suffix} -stage compute -o "${_spv}"
            DEPENDS "${_slang_src}"
            COMMENT "slangc ${_op}${_suffix}.spv"
            VERBATIM)
        list(APPEND _shader_outputs "${_spv}")
    endforeach()
    # Metal source (f32 only — Metal has no double). Generated on every platform so a broken
    # metal target fails the build here, not on the first Apple machine. spvonly ops (KHR
    # cooperative matrix) have no Metal analogue.
    if("spvonly" IN_LIST _parts)
        continue()
    endif()
    set(_metal_out "${_shader_dir}/${_op}_f32.metal")
    add_custom_command(OUTPUT "${_metal_out}"
        COMMAND "${CHEATAH_GPU_LINALG_SLANGC}" "${_slang_src}"
                -D${_guard} ${_extra} -DELEM=float -DSUFFIX=_f32
                -target metal -entry ${_op}_f32 -stage compute -o "${_metal_out}"
        DEPENDS "${_slang_src}"
        COMMENT "slangc ${_op}_f32.metal"
        VERBATIM)
    list(APPEND _shader_outputs "${_metal_out}")
endforeach()
add_custom_target(gpu_linalg_shaders ALL DEPENDS ${_shader_outputs})

# ---- The header-only library, per backend ----------------------------------
# Shared surface: this repo's headers + the cheatah seam headers.
add_library(cheatah_gpu_linalg_common INTERFACE)
target_include_directories(cheatah_gpu_linalg_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}          # tests/harness.hpp as "tests/harness.hpp"
    ${CHEATAH_DIR}/stdlib/ndarray        # cheatah "ndarray.hpp"
    ${CHEATAH_DIR}/stdlib/linalg         # cheatah "concepts.hpp" / "backend.hpp" (the device seam)
    ${CHEATAH_GPU_DIR})                  # cheatah-gpu "gpu/..." surfaces
target_compile_features(cheatah_gpu_linalg_common INTERFACE cxx_std_20)
# The host-bridge (bridge.hpp) executes cheatah's HOST factorization kernels, so their compiled
# TUs come along (the downstream-consumer pattern: compile the sibling checkout's stdlib sources
# straight into the consumer, at full optimization — these run the actual math).
set(_cheatah_host_tus
    ${CHEATAH_DIR}/stdlib/linalg/routines.cpp
    ${CHEATAH_DIR}/stdlib/linalg/simd.cpp
    ${CHEATAH_DIR}/stdlib/ndarray/ndarray.cpp
    ${CHEATAH_DIR}/stdlib/ndarray/ufunc_simd.cpp)
set_source_files_properties(${_cheatah_host_tus} PROPERTIES COMPILE_OPTIONS "-O3;-march=native")
target_sources(cheatah_gpu_linalg_common INTERFACE ${_cheatah_host_tus})

# METAL backend: metal-cpp everywhere; the emulated device TUs + shim off Apple; the
# slang-generated MSL directory on Apple.
add_library(cheatah_gpu_linalg_metal INTERFACE)
target_link_libraries(cheatah_gpu_linalg_metal INTERFACE cheatah_gpu_linalg_common)
target_include_directories(cheatah_gpu_linalg_metal INTERFACE ${CHEATAH_GPU_METAL_CPP})
# The explicit lane pin: context.hpp's platform default is Vulkan off Apple, so the emulated
# Metal test lane must say which backend it is (consumers define nothing and get the default).
target_compile_definitions(cheatah_gpu_linalg_metal INTERFACE CHEATAH_GPU_LINALG_METAL=1)
target_compile_options(cheatah_gpu_linalg_metal INTERFACE -fblocks -include cmath)
if(APPLE)
    target_compile_definitions(cheatah_gpu_linalg_metal INTERFACE
        CHEATAH_GPU_LINALG_MSL_DIR="${_shader_dir}")
    target_link_libraries(cheatah_gpu_linalg_metal INTERFACE
        "-framework Metal" "-framework Foundation" "-framework QuartzCore")
else()
    target_include_directories(cheatah_gpu_linalg_metal INTERFACE ${CHEATAH_GPU_DIR}/gpu/metal/shim)
    target_sources(cheatah_gpu_linalg_metal INTERFACE
        ${CHEATAH_GPU_DIR}/gpu/metal/emulated/emulated_metal.cpp
        ${CHEATAH_GPU_DIR}/gpu/metal/emulated/block_stubs.cpp)
endif()

# VULKAN backend: cheatah-gpu's vk.* forwarders over volk (preferred — the loader is dlopen'd
# at runtime, no libvulkan DT_NEEDED on any consumer) or the plain SDK headers + -lvulkan when
# no volk source is available. Kernels load from the .spv directory baked in here (env
# CHEATAH_GPU_LINALG_SPV_DIR overrides).
add_library(cheatah_gpu_linalg_vulkan INTERFACE)
target_link_libraries(cheatah_gpu_linalg_vulkan INTERFACE cheatah_gpu_linalg_common)
target_compile_definitions(cheatah_gpu_linalg_vulkan INTERFACE
    CHEATAH_GPU_LINALG_VULKAN=1
    CHEATAH_GPU_LINALG_SPV_DIR="${_shader_dir}")
if(CHEATAH_GPU_LINALG_VK_ROOT)
    target_include_directories(cheatah_gpu_linalg_vulkan INTERFACE
        "${CHEATAH_GPU_LINALG_VK_ROOT}/include")
endif()
if(EXISTS "${CHEATAH_GPU_LINALG_VOLK_SRC}/volk.c")
    # The one compiled TU of the Vulkan lane: volk's function-pointer table. A real STATIC
    # archive (not an INTERFACE source) so non-CMake consumers — purrc --link lines — can link
    # the same object: build/libcheatah_gpu_linalg_volk.a.
    enable_language(C)
    add_library(cheatah_gpu_linalg_volk STATIC "${CHEATAH_GPU_LINALG_VOLK_SRC}/volk.c")
    target_include_directories(cheatah_gpu_linalg_volk PUBLIC "${CHEATAH_GPU_LINALG_VOLK_SRC}")
    if(CHEATAH_GPU_LINALG_VK_ROOT)
        target_include_directories(cheatah_gpu_linalg_volk PRIVATE
            "${CHEATAH_GPU_LINALG_VK_ROOT}/include")
    endif()
    set_target_properties(cheatah_gpu_linalg_volk PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(cheatah_gpu_linalg_volk PUBLIC ${CMAKE_DL_LIBS})
    target_link_libraries(cheatah_gpu_linalg_vulkan INTERFACE cheatah_gpu_linalg_volk)
    message(STATUS "cheatah-gpu-linalg: Vulkan via volk (${CHEATAH_GPU_LINALG_VOLK_SRC}) — no libvulkan link")
else()
    message(WARNING "cheatah-gpu-linalg: no volk source found (build cheatah-gpu once to fetch it) "
                    "— falling back to -lvulkan; consumers dlopen'd by a runtime will need LD_PRELOAD")
    if(CHEATAH_GPU_LINALG_VK_ROOT)
        target_link_directories(cheatah_gpu_linalg_vulkan INTERFACE
            "${CHEATAH_GPU_LINALG_VK_ROOT}/lib")
    endif()
    target_link_libraries(cheatah_gpu_linalg_vulkan INTERFACE vulkan)
endif()

# The default alias a consumer links: the backend that drives real GPU silicon on this OS.
if(APPLE)
    add_library(cheatah_gpu_linalg ALIAS cheatah_gpu_linalg_metal)
else()
    add_library(cheatah_gpu_linalg ALIAS cheatah_gpu_linalg_vulkan)
endif()

# ---- consumer.cmake: the ONE description of how a sibling repo consumes this build ---------
# Consuming projects include(<here>/build/consumer.cmake) instead of re-deriving
# SDK paths or naming a backend: it defines the IMPORTED target cheatah_gpu_linalg::consumer
# for CMake targets, plus flat variables for non-CMake link lines (purrc --cxxflag/--link).
# Generated for THIS platform's default lane; env CHEATAH_GPU_LINALG_SPV_DIR/_MSL_DIR still
# override shader locations at runtime.
set(_consumer_includes "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CHEATAH_GPU_DIR}")
set(_consumer_libs "")
if(APPLE)
    list(APPEND _consumer_includes "${CHEATAH_GPU_METAL_CPP}")
    set(_consumer_libs "-framework Metal;-framework Foundation;-framework QuartzCore")
else()
    if(EXISTS "${CHEATAH_GPU_LINALG_VOLK_SRC}/volk.c")
        list(APPEND _consumer_includes "${CHEATAH_GPU_LINALG_VOLK_SRC}")
        set(_consumer_libs "${CMAKE_BINARY_DIR}/libcheatah_gpu_linalg_volk.a;dl")
    else()
        set(_consumer_libs "vulkan;dl")
    endif()
    if(CHEATAH_GPU_LINALG_VK_ROOT)
        list(APPEND _consumer_includes "${CHEATAH_GPU_LINALG_VK_ROOT}/include")
    endif()
endif()
# The compile flags a consumer's TU needs beyond -I: the lane's baked shader directory (the
# same-named environment variable still overrides it at runtime).
if(APPLE)
    set(_consumer_cxxflags "-DCHEATAH_GPU_LINALG_MSL_DIR=\\\"${_shader_dir}\\\"")
else()
    set(_consumer_cxxflags "-DCHEATAH_GPU_LINALG_SPV_DIR=\\\"${_shader_dir}\\\"")
endif()
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/consumer.cmake" CONTENT [=[
# GENERATED by cheatah-gpu-linalg's configure — do not edit; include() from a sibling repo.
set(CHEATAH_GPU_LINALG_CONSUMER_INCLUDES "@_consumer_includes@")
set(CHEATAH_GPU_LINALG_CONSUMER_LIBS "@_consumer_libs@")
set(CHEATAH_GPU_LINALG_CONSUMER_SHADERS "@_shader_dir@")
set(CHEATAH_GPU_LINALG_CONSUMER_CXXFLAGS "@_consumer_cxxflags@")
if(NOT TARGET cheatah_gpu_linalg::consumer)
    add_library(cheatah_gpu_linalg_consumer_iface INTERFACE)
    add_library(cheatah_gpu_linalg::consumer ALIAS cheatah_gpu_linalg_consumer_iface)
    target_include_directories(cheatah_gpu_linalg_consumer_iface
        INTERFACE ${CHEATAH_GPU_LINALG_CONSUMER_INCLUDES})
    target_link_libraries(cheatah_gpu_linalg_consumer_iface
        INTERFACE ${CHEATAH_GPU_LINALG_CONSUMER_LIBS})
endif()
]=] @ONLY)

# ---- Benchmarks (off by default; -DCHEATAH_GPU_LINALG_BENCH=ON) ------------
option(CHEATAH_GPU_LINALG_BENCH "Build the Google Benchmark suite + probe CLI" OFF)
if(CHEATAH_GPU_LINALG_BENCH)
    add_subdirectory(bench)
endif()

# ---- Sanitizers (a dedicated build dir turns this on: scripts/qa.sh) -------
option(CHEATAH_GPU_LINALG_SANITIZE "Build the tests under ASan + UBSan" OFF)
if(CHEATAH_GPU_LINALG_SANITIZE)
    # metal-cpp deliberately calls objc_msgSend through a typed function-pointer cast — its
    # documented mechanism — so the `function` UBSan check is off, matching cheatah-gpu's gate.
    add_compile_options(-fsanitize=address,undefined -fno-sanitize=function -g -O1)
    add_link_options(-fsanitize=address,undefined)
endif()

# ---- Consumer cut-off: a CPM/biome subproject wants the LIBRARIES + SHADERS + consumer.cmake
#      above and NOTHING below — the test battery and examples assume sibling checkouts and a
#      dev toolchain that a fetched dependency does not have (the standard e2e caught exactly
#      this: examples/ exploding inside a consumer's _deps tree).
if(NOT PROJECT_IS_TOP_LEVEL)
    return()
endif()

# ---- Tests: every op, on BOTH backends (GoogleTest — the ecosystem standard) ----------------
# Same mechanism and pin as ../cheatah-gpu (which matches ../cheatah). Fetched after the
# sanitizer block so a sanitized build instruments gtest too.
include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        52eb8108c5bdec04579160ae17225d66034bd723)  # pinned (matches ../cheatah)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)
enable_testing()

# ---- Examples (usage documentation that cannot drift — CTest-gated) --------
add_subdirectory(examples)
set(_tests matmul outer transpose kron dot trace bridge elementwise operators matmul_batched complex conv factories guards)
foreach(_t IN LISTS _tests)
    # Each TU compiled twice — once per backend. Discovery registers every TEST as its own ctest
    # entry, prefixed gpu:<op>:<bk>: so the header @test tags and -R filters keep working; the
    # gtest exit code is the verdict (no output-regex machinery).
    add_executable(gpu_linalg_${_t}_mtl tests/${_t}_test.cpp)
    target_link_libraries(gpu_linalg_${_t}_mtl PRIVATE cheatah_gpu_linalg_metal GTest::gtest_main)
    gtest_discover_tests(gpu_linalg_${_t}_mtl TEST_PREFIX "gpu:${_t}:mtl:")

    add_executable(gpu_linalg_${_t}_vk tests/${_t}_test.cpp)
    target_link_libraries(gpu_linalg_${_t}_vk PRIVATE cheatah_gpu_linalg_vulkan GTest::gtest_main)
    add_dependencies(gpu_linalg_${_t}_vk gpu_linalg_shaders)
    # GPU driver stacks hold caches/threads for the process lifetime; leak detection is meaningful
    # for OUR code under the emulated backend, not for the vendor driver blob.
    gtest_discover_tests(gpu_linalg_${_t}_vk TEST_PREFIX "gpu:${_t}:vk:"
        PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
endforeach()

# One combined binary of every test TU (Metal-emulated lane), for the coverage gate: a single
# object gives llvm-cov true union semantics — per-TU binaries each carry zero-count copies of
# template instantiations their own tests never call, which drags a multi-object report below
# what actually ran. Not registered with ctest (the per-TU binaries above are the test suite).
set(_all_test_sources "")
foreach(_t IN LISTS _tests)
    list(APPEND _all_test_sources tests/${_t}_test.cpp)
endforeach()
add_executable(gpu_linalg_all_mtl ${_all_test_sources})
target_link_libraries(gpu_linalg_all_mtl PRIVATE cheatah_gpu_linalg_metal GTest::gtest_main)
