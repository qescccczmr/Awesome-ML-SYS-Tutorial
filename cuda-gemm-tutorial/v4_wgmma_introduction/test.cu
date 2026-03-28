#include "cuda_utils.h"
#include "types.h"
#include "matrix_utils.h"
#include "benchmark.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>

// 各版本launcher声明
void gemm_v1(half* C, const half* A, const half* B, int M, int N, int K);
void gemm_v2(half* C, const half* A, const half* B, int M, int N, int K);
void gemm_v3(half* C, const half* A, const half* B, int M, int N, int K);
void gemm_v4(half* C, const half* A, const half* B, int M, int N, int K);

// cuBLAS参考实现
void gemm_cublas(cublasHandle_t handle,
                 half* C, const half* A, const half* B,
                 int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16F, N,
        A, CUDA_R_16F, K,
        &beta,
        C, CUDA_R_16F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// ============================================================================
// 正确性测试
// ============================================================================
bool run_correctness_test(cublasHandle_t handle,
                          int M, int N, int K,
                          const char* test_name) {
    size_t bytes_A = (size_t)M * K * sizeof(half);
    size_t bytes_B = (size_t)K * N * sizeof(half);
    size_t bytes_C = (size_t)M * N * sizeof(half);

    half *h_A = new half[M * K];
    half *h_B = new half[K * N];
    half *h_C = new half[M * N];
    half *h_C_ref = new half[M * N];

    init_matrix_random(h_A, M, K, -1.0f, 1.0f, 42);
    init_matrix_random(h_B, K, N, -1.0f, 1.0f, 43);

    half *d_A, *d_B, *d_C, *d_C_ref;
    CUDA_CHECK(cudaMalloc(&d_A,     bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B,     bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C,     bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_ref, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    // V4 kernel
    gemm_v4(d_C, d_A, d_B, M, N, K);
    CUDA_SYNC_CHECK();

    // cuBLAS reference
    gemm_cublas(handle, d_C_ref, d_A, d_B, M, N, K);
    CUDA_SYNC_CHECK();

    CUDA_CHECK(cudaMemcpy(h_C,     d_C,     bytes_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_ref, d_C_ref, bytes_C, cudaMemcpyDeviceToHost));

    bool passed = matrices_are_close(h_C, h_C_ref, M * N, 1e-2f, 1e-2f);
    float err   = max_abs_error(h_C, h_C_ref, M * N);

    printf("  [%s] M=%d N=%d K=%d | MaxErr=%.4f | %s\n",
           test_name, M, N, K, err, passed ? "PASSED" : "FAILED");

    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_C_ref;
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C)); CUDA_CHECK(cudaFree(d_C_ref));
    return passed;
}

// ============================================================================
// 性能对比：V1 vs V2 vs V3 vs V4 vs cuBLAS
// ============================================================================
void run_comparison(cublasHandle_t handle, int M, int N, int K) {
    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)M * N * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_A, 1, M * K * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_B, 1, K * N * sizeof(half)));

    GEMMBenchmark b1("V1 Tiled",  M, N, K);
    b1.run([&]{ gemm_v1(d_C, d_A, d_B, M, N, K); });

    GEMMBenchmark b2("V2 DblBuf", M, N, K);
    b2.run([&]{ gemm_v2(d_C, d_A, d_B, M, N, K); });

    GEMMBenchmark b3("V3 TMA",    M, N, K);
    b3.run([&]{ gemm_v3(d_C, d_A, d_B, M, N, K); });

    GEMMBenchmark b4("V4 WGMMA",  M, N, K);
    b4.run([&]{ gemm_v4(d_C, d_A, d_B, M, N, K); });

    GEMMBenchmark bc("cuBLAS",    M, N, K);
    bc.run([&]{ gemm_cublas(handle, d_C, d_A, d_B, M, N, K); });

    printf("M=%d N=%d K=%d:\n", M, N, K);
    b1.print();
    b2.print();
    b3.print();
    b4.print();
    bc.print();
    printf("  V4/V3 speedup: %.2fx | V4 vs cuBLAS: %.1f%%\n\n",
           b3.time_ms / b4.time_ms,
           b4.tflops / bc.tflops * 100.0f);

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    bool do_benchmark = (argc > 1 && strcmp(argv[1], "--benchmark") == 0);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\n=== V4: WGMMA Introduction GEMM ===\n");
    printf("GPU: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // ---- 正确性测试 ----
    // V4 TILE_M=64, TILE_N=64
    printf("--- Correctness Tests ---\n");
    bool all_passed = true;
    all_passed &= run_correctness_test(handle, 64,  64,  64,  "Square 64");
    all_passed &= run_correctness_test(handle, 128, 128, 128, "Square 128");
    all_passed &= run_correctness_test(handle, 256, 256, 256, "Square 256");
    all_passed &= run_correctness_test(handle, 64,  128, 64,  "Rectangular");
    all_passed &= run_correctness_test(handle, 128, 64,  128, "Rectangular 2");
    printf("\nCorrectness: %s\n", all_passed ? "ALL PASSED" : "SOME FAILED");

    // ---- 性能对比 ----
    if (do_benchmark) {
        printf("\n--- Performance Comparison: V1 vs V2 vs V3 vs V4 vs cuBLAS ---\n");
        for (int s : {512, 1024, 2048, 4096}) {
            run_comparison(handle, s, s, s);
        }
        printf("\n--- MOE Workload ---\n");
        run_comparison(handle, 2048, 1024, 4096);
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return all_passed ? 0 : 1;
}
