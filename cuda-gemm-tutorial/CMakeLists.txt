cmake_minimum_required(VERSION 3.18)
project(H200_GEMM_Tutorial CUDA CXX)

# Require CUDA and set architecture
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 90)  # H200/H100 (sm_90)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# Find cuBLAS
find_package(CUDAToolkit REQUIRED)

# Common library (header-only)
add_library(common INTERFACE)
target_include_directories(common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(common INTERFACE CUDA::cudart CUDA::cublas)

# Add subdirectories for each version
add_subdirectory(v1_baseline_tiled_gemm)
add_subdirectory(v2_double_buffering)
add_subdirectory(v3_tma_introduction)
add_subdirectory(v4_wgmma_introduction)
add_subdirectory(v5_production_kernel)
