# V1: Baseline Tiled GEMM
add_executable(test_v1
    test.cu
    kernel.cu
)
target_link_libraries(test_v1 common)
target_compile_options(test_v1 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)
