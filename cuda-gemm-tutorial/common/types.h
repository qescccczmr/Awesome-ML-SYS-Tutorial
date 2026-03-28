#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <type_traits>

// ============================================================================
// Type Definitions and Conversions
// ============================================================================
// Unified type aliases and conversion helpers for FP16 and BF16.
//
// Supported types:
//   fp16_t  = __half       (cuda_fp16.h)
//   bf16_t  = __nv_bfloat16 (cuda_bf16.h)
// ============================================================================

using fp16_t = __half;
using bf16_t = __nv_bfloat16;

// ---- cuBLAS type helpers (used by test.cu to call cublasGemmEx) ----

inline cudaDataType_t cuda_data_type(const fp16_t*) { return CUDA_R_16F;  }
inline cudaDataType_t cuda_data_type(const bf16_t*) { return CUDA_R_16BF; }

inline cublasComputeType_t cublas_compute_type(const fp16_t*) { return CUBLAS_COMPUTE_32F; }
inline cublasComputeType_t cublas_compute_type(const bf16_t*) { return CUBLAS_COMPUTE_32F; }

// ---- Host-side scalar conversions ----

__host__ __device__ inline float to_float(fp16_t v) { return __half2float(v); }
__host__ __device__ inline float to_float(bf16_t v) { return __bfloat162float(v); }

template<typename T>
__host__ __device__ inline T from_float(float v);

template<>
__host__ __device__ inline fp16_t from_float<fp16_t>(float v) { return __float2half(v); }

template<>
__host__ __device__ inline bf16_t from_float<bf16_t>(float v) { return __float2bfloat16(v); }
