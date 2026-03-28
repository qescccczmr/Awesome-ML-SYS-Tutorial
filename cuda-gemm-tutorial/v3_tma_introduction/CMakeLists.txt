# V3: TMA Introduction
add_executable(test_v3
    test.cu
    kernel.cu
    ${CMAKE_SOURCE_DIR}/v1_baseline_tiled_gemm/kernel.cu
    ${CMAKE_SOURCE_DIR}/v2_double_buffering/kernel.cu
)
target_link_libraries(test_v3 common CUDA::cuda_driver)
target_compile_options(test_v3 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)
