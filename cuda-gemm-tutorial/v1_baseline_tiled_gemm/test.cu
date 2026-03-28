#include "cuda_utils.h"
#include "types.h"
#include "matrix_utils.h"
#include "benchmark.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>

// V1 kernel launcher声明（定义在kernel.cu）
void gemm_v1(half* C, const half* A, const half* B, int M, int N, int K);

// ============================================================================
// cuBLAS参考实现
// ============================================================================
// 用cuBLAS计算C = A × B，作为正确性验证的参考
// 注意cuBLAS默认列主序，所以我们用转置技巧：
//   C(行主序) = A(行主序) × B(行主序)
//   等价于：C^T = B^T × A^T（列主序）
void gemm_cublas(cublasHandle_t handle,
                 half* C, const half* A, const half* B,
                 int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    // cuBLAS: C[N×M] = B[N×K] × A[K×M] (列主序，相当于行主序的 C=A×B)
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,              // cuBLAS的(m,n,k) = 我们的(N,M,K)
        &alpha,
        B, CUDA_R_16F, N,     // 列主序B[N×K]
        A, CUDA_R_16F, K,     // 列主序A[K×M]
        &beta,
        C, CUDA_R_16F, N,     // 列主序C[N×M]
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

    // 分配host内存
    half* h_A     = new half[M * K];
    half* h_B     = new half[K * N];
    half* h_C     = new half[M * N];
    half* h_C_ref = new half[M * N];

    // 随机初始化（固定seed保证可重复）
    init_matrix_random(h_A, M, K, -1.0f, 1.0f, 42);
    init_matrix_random(h_B, K, N, -1.0f, 1.0f, 43);

    // 分配device内存
    half *d_A, *d_B, *d_C, *d_C_ref;
    CUDA_CHECK(cudaMalloc(&d_A,     bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B,     bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C,     bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_ref, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    // 运行我们的kernel
    gemm_v1(d_C, d_A, d_B, M, N, K);
    CUDA_SYNC_CHECK();

    // 运行cuBLAS参考
    gemm_cublas(handle, d_C_ref, d_A, d_B, M, N, K);
    CUDA_SYNC_CHECK();

    // 拷贝结果到host
    CUDA_CHECK(cudaMemcpy(h_C,     d_C,     bytes_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_ref, d_C_ref, bytes_C, cudaMemcpyDeviceToHost));

    // 比较
    bool passed = matrices_are_close(h_C, h_C_ref, M * N, 1e-2f, 1e-2f);
    float err   = max_abs_error(h_C, h_C_ref, M * N);

    printf("  [%s] M=%d N=%d K=%d | MaxErr=%.4f | %s\n",
           test_name, M, N, K, err, passed ? "PASSED" : "FAILED");

    // 释放内存
    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_C_ref;
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C)); CUDA_CHECK(cudaFree(d_C_ref));

    return passed;
}

// ============================================================================
// 性能Benchmark
// ============================================================================
void run_benchmark(int M, int N, int K) {
    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)M * N * sizeof(half)));

    // 初始化为随机值（防止编译器优化掉）
    CUDA_CHECK(cudaMemset(d_A, 1, M * K * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_B, 1, K * N * sizeof(half)));

    GEMMBenchmark bench("V1 Tiled", M, N, K);
    bench.run([&]{ gemm_v1(d_C, d_A, d_B, M, N, K); }, 5, 100);
    bench.print();
    printf("    Peak%%: %.1f%% of H200 FP16 theoretical peak\n", bench.peak_pct());

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    bool do_benchmark = (argc > 1 && strcmp(argv[1], "--benchmark") == 0);

    // 打印GPU信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\n=== V1: Baseline Tiled GEMM ===\n");
    printf("GPU: %s (sm_%d%d)\n\n", prop.name,
           prop.major, prop.minor);

    // 创建cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // ---- 正确性测试 ----
    printf("--- Correctness Tests ---\n");
    bool all_passed = true;

    // 测试1: 基础方阵
    all_passed &= run_correctness_test(handle, 64,  64,  64,  "Small square");
    all_passed &= run_correctness_test(handle, 128, 128, 128, "Mid square");
    all_passed &= run_correctness_test(handle, 256, 256, 256, "Large square");
    // 测试2: 非对齐尺寸（边界处理）
    all_passed &= run_correctness_test(handle, 100, 100, 100, "Non-aligned 100");
    all_passed &= run_correctness_test(handle, 200, 300, 150, "Rectangular");
    // 测试3: 单位矩阵验证（C = A×I = A）
    {
        const int S = 128;
        half* h_A = new half[S * S];
        half* h_I = new half[S * S];
        half* h_C = new half[S * S];
        init_matrix_random(h_A, S, S, -1.0f, 1.0f);
        init_matrix_identity(h_I, S);

        half *d_A, *d_I, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, S * S * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_I, S * S * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C, S * S * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, S*S*sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_I, h_I, S*S*sizeof(half), cudaMemcpyHostToDevice));

        gemm_v1(d_C, d_A, d_I, S, S, S);
        CUDA_SYNC_CHECK();
        CUDA_CHECK(cudaMemcpy(h_C, d_C, S*S*sizeof(half), cudaMemcpyDeviceToHost));

        bool passed = matrices_are_close(h_C, h_A, S*S, 1e-2f, 1e-2f);
        printf("  [Identity test] A×I==A | %s\n", passed ? "PASSED" : "FAILED");
        all_passed &= passed;

        delete[] h_A; delete[] h_I; delete[] h_C;
        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_I)); CUDA_CHECK(cudaFree(d_C));
    }

    printf("\nCorrectness: %s\n", all_passed ? "ALL PASSED" : "SOME FAILED");

    // ---- 性能测试 ----
    if (do_benchmark) {
        printf("\n--- Performance Benchmark ---\n");
        for (int s : {512, 1024, 2048, 4096}) {
            run_benchmark(s, s, s);
        }
        // MOE典型形状
        printf("  [MOE workload]\n");
        run_benchmark(2048, 1024, 4096);
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return all_passed ? 0 : 1;
}
