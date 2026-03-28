#pragma once

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <string>
#include <functional>

// ============================================================================
// GEMM Benchmark Framework
// ============================================================================
// Wraps CUDA event timing and TFLOPS calculation.
//
// Usage:
//   GEMMBenchmark bench("V1 Tiled", M, N, K);
//   bench.run([&]{ gemm_launcher(d_C, d_A, d_B, M, N, K); });
//   bench.print();
// ============================================================================

struct GEMMBenchmark {
    std::string name;
    int M, N, K;
    float time_ms  = 0.0f;
    float tflops   = 0.0f;

    GEMMBenchmark(const char* name, int M, int N, int K)
        : name(name), M(M), N(N), K(K) {}

    // Run kernel function with warmup + timed iterations
    void run(std::function<void()> kernel_fn,
             int warmup = 5, int iters = 100) {
        cudaEvent_t t_start, t_stop;
        CUDA_CHECK(cudaEventCreate(&t_start));
        CUDA_CHECK(cudaEventCreate(&t_stop));

        // Warmup
        for (int i = 0; i < warmup; i++) { kernel_fn(); }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK_LAST_ERROR();

        // Timed iterations
        CUDA_CHECK(cudaEventRecord(t_start));
        for (int i = 0; i < iters; i++) { kernel_fn(); }
        CUDA_CHECK(cudaEventRecord(t_stop));
        CUDA_CHECK(cudaEventSynchronize(t_stop));
        CUDA_CHECK_LAST_ERROR();

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, t_start, t_stop));
        time_ms = total_ms / iters;

        // 2*M*N*K FLOPs for GEMM (one multiply + one add per element)
        double flops = 2.0 * (double)M * N * K;
        tflops = (float)(flops / (time_ms * 1e-3) / 1e12);

        CUDA_CHECK(cudaEventDestroy(t_start));
        CUDA_CHECK(cudaEventDestroy(t_stop));
    }

    void print() const {
        printf("  [%s] M=%d N=%d K=%d | %.3f ms | %.2f TFLOPS\n",
               name.c_str(), M, N, K, time_ms, tflops);
    }

    void print_csv() const {
        printf("%s,%d,%d,%d,%.3f,%.2f\n",
               name.c_str(), M, N, K, time_ms, tflops);
    }

    // % of theoretical H200 FP16 peak (1979 TFLOPS)
    float peak_pct(float peak_tflops = 1979.0f) const {
        return tflops / peak_tflops * 100.0f;
    }
};

// Run benchmark on a list of square sizes
inline void run_benchmark_suite(const char* name,
                                std::function<void(int,int,int)> launcher,
                                const std::initializer_list<int>& sizes,
                                int warmup = 5, int iters = 50) {
    printf("\n=== Benchmark: %s ===\n", name);
    printf("%-20s  %6s  %6s  %6s  %10s  %10s  %8s\n",
           "Kernel", "M", "N", "K", "Time(ms)", "TFLOPS", "Peak%");
    for (int s : sizes) {
        GEMMBenchmark b(name, s, s, s);
        b.run([&]{ launcher(s, s, s); }, warmup, iters);
        printf("%-20s  %6d  %6d  %6d  %10.3f  %10.2f  %7.1f%%\n",
               b.name.c_str(), s, s, s, b.time_ms, b.tflops, b.peak_pct());
    }
}
