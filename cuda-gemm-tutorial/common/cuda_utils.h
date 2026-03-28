#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

// ============================================================================
// CUDA Error Checking Macros
// ============================================================================
// Usage:
//   CUDA_CHECK(cudaMalloc(&ptr, size));
//   CUBLAS_CHECK(cublasCreate(&handle));
//
// On error: prints filename, line number, error string, then exits.
// ============================================================================

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t _status = (call);                                        \
        if (_status != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n",              \
                    __FILE__, __LINE__, (int)_status);                          \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Check for any pending CUDA errors (call after kernel launches)
#define CUDA_CHECK_LAST_ERROR()                                                 \
    do {                                                                        \
        cudaError_t _err = cudaGetLastError();                                  \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Synchronize device then check for errors
#define CUDA_SYNC_CHECK()                                                       \
    do {                                                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                    \
        CUDA_CHECK_LAST_ERROR();                                                \
    } while (0)
