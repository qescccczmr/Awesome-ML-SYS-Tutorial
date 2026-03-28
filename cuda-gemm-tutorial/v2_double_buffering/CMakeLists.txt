# V2: Double Buffering
add_executable(test_v2
    test.cu
    kernel.cu
    ${CMAKE_SOURCE_DIR}/v1_baseline_tiled_gemm/kernel.cu  # 用于性能对比
)
target_link_libraries(test_v2 common)
target_compile_options(test_v2 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)
