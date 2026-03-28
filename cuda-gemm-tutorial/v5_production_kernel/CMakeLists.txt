# V5: Production Kernel (TMA + WGMMA + 3-Stage Pipeline)
# WGMMA指令需要sm_90a (architecture-specific features)
add_executable(test_v5
    test.cu
    kernel.cu
    ${CMAKE_SOURCE_DIR}/v1_baseline_tiled_gemm/kernel.cu
    ${CMAKE_SOURCE_DIR}/v2_double_buffering/kernel.cu
    ${CMAKE_SOURCE_DIR}/v3_tma_introduction/kernel.cu
    ${CMAKE_SOURCE_DIR}/v4_wgmma_introduction/kernel.cu
)
set_target_properties(test_v5 PROPERTIES CUDA_ARCHITECTURES "90a")
target_link_libraries(test_v5 common CUDA::cuda_driver)
target_compile_options(test_v5 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)
