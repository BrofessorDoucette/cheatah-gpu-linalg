# bench/ — Google Benchmark suite + the probe CLI, per backend (like the tests). Off by default:
# -DCHEATAH_GPU_LINALG_BENCH=ON fetches google/benchmark (same pin as cheatah's tests/benchmarks).
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
include(FetchContent)
FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        192ef10025eb2c4cdd392bc502f0c852196baa48) # pinned — cheatah's same pin
FetchContent_MakeAvailable(benchmark)

foreach(_bk IN ITEMS vk mtl)
    if(_bk STREQUAL "vk")
        set(_lib cheatah_gpu_linalg_vulkan)
    else()
        set(_lib cheatah_gpu_linalg_metal)
    endif()
    add_executable(gpu_linalg_bench_${_bk} ${CMAKE_CURRENT_SOURCE_DIR}/gpu_linalg_bench.cpp)
    target_link_libraries(gpu_linalg_bench_${_bk} PRIVATE ${_lib} benchmark::benchmark_main)
    add_dependencies(gpu_linalg_bench_${_bk} gpu_linalg_shaders)

    add_executable(gpu_linalg_probe_${_bk} ${CMAKE_CURRENT_SOURCE_DIR}/probe.cpp)
    target_link_libraries(gpu_linalg_probe_${_bk} PRIVATE ${_lib})
    add_dependencies(gpu_linalg_probe_${_bk} gpu_linalg_shaders)
endforeach()

# vkinspect — Vulkan-only diagnosis CLI (coop-matrix property table + pipeline register/spill
# statistics via VK_KHR_pipeline_executable_properties).
add_executable(gpu_linalg_vkinspect ${CMAKE_CURRENT_SOURCE_DIR}/vkinspect.cpp)
target_link_libraries(gpu_linalg_vkinspect PRIVATE cheatah_gpu_linalg_vulkan)
add_dependencies(gpu_linalg_vkinspect gpu_linalg_shaders)
