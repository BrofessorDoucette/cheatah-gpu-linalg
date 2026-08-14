# examples/ — runnable usage documentation that cannot drift (the downstream-consumer pattern):
# each example
# is ONE templated function instantiated for host_array and device_array, built per backend like
# the tests, registered in CTest with a PASS:/FAIL: regex, and asserting device residency through
# the transfer ledger.
set(_examples mlp_forward linear_regression_gd batched_inference vector_pipeline)
foreach(_e IN LISTS _examples)
    foreach(_bk IN ITEMS vk mtl)
        if(_bk STREQUAL "vk")
            set(_lib cheatah_gpu_linalg_vulkan)
        else()
            set(_lib cheatah_gpu_linalg_metal)
        endif()
        add_executable(example_${_e}_${_bk} ${CMAKE_CURRENT_SOURCE_DIR}/${_e}.cpp)
        target_link_libraries(example_${_e}_${_bk} PRIVATE ${_lib})
        target_include_directories(example_${_e}_${_bk} PRIVATE ${CMAKE_SOURCE_DIR})
        add_dependencies(example_${_e}_${_bk} gpu_linalg_shaders)
        add_test(NAME example:${_e}:${_bk} COMMAND example_${_e}_${_bk})
        set_tests_properties(example:${_e}:${_bk} PROPERTIES
            PASS_REGULAR_EXPRESSION "PASS:"
            FAIL_REGULAR_EXPRESSION "FAIL:")
    endforeach()
endforeach()

# The PURE-CHEATAH examples (examples/purr/*.purr): compiled by purrc against the Vulkan backend
# through the `import gpulinalg` module (purr/gpulinalg). One ctest entry runs them all via the
# script, which skips cleanly when the cheatah toolchain isn't built.
add_test(NAME example:purr COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_purr_examples.sh)
set_tests_properties(example:purr PROPERTIES FAIL_REGULAR_EXPRESSION "FAILED:")
