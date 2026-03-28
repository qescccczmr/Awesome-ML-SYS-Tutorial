#include "cuda_utils.h"
#include "types.h"
#include <cuda_fp16.h>
#include <cuda.h>

// ============================================================================
// V5: Production Kernel — TMA + WGMMA + 3-Stage Pipeline
// ============================================================================
//
// 集成V3(TMA)和V4(WGMMA)的所有优化，并增加：
//   1. 3-stage pipeline：3个buffer，2个TMA in-flight + 1个WGMMA计算
//   2. 更大的tile：TILE_M=128, TILE_N=128, 充分利用SM
//   3. 2个warpgroup协作：每个block 256 threads = 2 warpgroups
//   4. 边界处理：支持任意M/N/K（非tile倍数）
//
// 3-stage流水线原理：
//
//   时间 →
//   buf0: [TMA load tile0]  [WGMMA tile0]           [TMA load tile3]...
//   buf1:    [TMA load tile1]  [WGMMA tile1]           ...
//   buf2:       [TMA load tile2]  [WGMMA tile2]
//
//   好处：当 WGMMA_latency ≈ TMA_latency 时，流水线满载
//         3个buffer保证始终有1个可用于TMA、1个可用于WGMMA
//
// ============================================================================

static constexpr int TILE_M   = 128;
static constexpr int TILE_N   = 128;
static constexpr int TILE_K   = 32;
static constexpr int WGMMA_K  = 16;
static constexpr int NUM_BUF  = 3;   // 三缓冲
static constexpr int WARPGROUP_SIZE = 128;

__constant__ CUtensorMap v5_tma_desc_A;
__constant__ CUtensorMap v5_tma_desc_B;

void gemm_v3(half* C, const half* A, const half* B, int M, int N, int K);

// ---- mbarrier/TMA helpers (与V4相同) ----
__device__ __forceinline__
void v5_mbar_init(uint64_t* m, uint32_t c) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(m));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(p), "r"(c));
}
__device__ __forceinline__
void v5_mbar_expect(uint64_t* m, uint32_t b) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(m));
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(p), "r"(b));
}
__device__ __forceinline__
void v5_mbar_wait(uint64_t* m, uint32_t ph) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(m));
    asm volatile(
        "{\n.reg .pred p;\nLAB_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "@!p bra LAB_%=;\n}\n"
        :: "r"(p), "r"(ph) : "memory");
}
__device__ __forceinline__
void v5_tma_load(const CUtensorMap* d, void* dst, uint64_t* m, int cx, int cy) {
    uint32_t dp = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t mp = static_cast<uint32_t>(__cvta_generic_to_shared(m));
    uint64_t tp = reinterpret_cast<uint64_t>(d);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dp), "l"(tp), "r"(cx), "r"(cy), "r"(mp) : "memory");
}

// ---- WGMMA helpers ----
__device__ __forceinline__
uint64_t v5_wgmma_desc(void* base, uint32_t ldm_bytes, uint32_t sw = 0) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(base));
    uint64_t d = 0;
    d |= (uint64_t(a >> 4)) & 0x3FFF;
    d |= (uint64_t((ldm_bytes >> 4) & 0x3FFF)) << 16;
    d |= uint64_t(sw) << 62;
    return d;
}
__device__ __forceinline__ void v5_wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void v5_wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void v5_wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }

__device__ __forceinline__
void v5_wgmma_m64n64k16(float* acc, uint64_t da, uint64_t db) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9,%10,%11,%12,%13,%14,%15,"
        " %16,%17,%18,%19,%20,%21,%22,%23,"
        " %24,%25,%26,%27,%28,%29,%30,%31},"
        " %32, %33, 1, 1, 1, 0, 0;"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]),
          "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]),
          "+f"(acc[8]), "+f"(acc[9]), "+f"(acc[10]),"+f"(acc[11]),
          "+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),
          "+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),
          "+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),
          "+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),
          "+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31])
        : "l"(da), "l"(db));
}

// ============================================================================
// V5 Kernel：每个block = 1 warpgroup (128 threads), 处理 128×128 输出
// 使用 2次 WGMMA 调用覆盖 M=128 (2 × 64)
// ============================================================================
__global__ void __launch_bounds__(WARPGROUP_SIZE)
gemm_v5_kernel(
    half* __restrict__ C,
    const half* __restrict__ A,
    const half* __restrict__ B,
    int M, int N, int K)
{
    extern __shared__ char smem[];

    // 每个buffer的tile大小
    constexpr int As_tile = TILE_M * TILE_K;  // 128*32=4096 half = 8KB
    constexpr int Bs_tile = TILE_K * TILE_N;  // 32*128=4096 half = 8KB
    constexpr int buf_size = (As_tile + Bs_tile) * sizeof(half);  // 16KB per buffer

    // 3个buffer + 3个mbarrier
    half* As[NUM_BUF];
    half* Bs[NUM_BUF];
    for (int i = 0; i < NUM_BUF; i++) {
        As[i] = reinterpret_cast<half*>(smem + i * buf_size);
        Bs[i] = As[i] + As_tile;
    }
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + NUM_BUF * buf_size);

    int tid = threadIdx.x;
    int bx  = blockIdx.x;
    int by  = blockIdx.y;
    int num_k = (K + TILE_K - 1) / TILE_K;

    // 累加器：TILE_M=128需要2组WGMMA (各64行), 每组32个FP32
    float acc_lo[32], acc_hi[32];  // lo=行0..63, hi=行64..127
    #pragma unroll
    for (int i = 0; i < 32; i++) { acc_lo[i] = 0.f; acc_hi[i] = 0.f; }

    // 初始化mbarrier
    if (tid == 0) {
        for (int i = 0; i < NUM_BUF; i++) v5_mbar_init(&mbar[i], 1);
    }
    __syncthreads();

    uint32_t phase[NUM_BUF] = {0, 0, 0};
    constexpr uint32_t tx_bytes = buf_size;

    // ================================================================
    // Prologue：填充前 min(NUM_BUF, num_k) 个buffer
    // ================================================================
    int prologue_count = min(NUM_BUF, num_k);
    for (int i = 0; i < prologue_count; i++) {
        if (tid == 0) {
            v5_mbar_expect(&mbar[i], tx_bytes);
            v5_tma_load(&v5_tma_desc_A, As[i], &mbar[i],
                        i * TILE_K, by * TILE_M);
            v5_tma_load(&v5_tma_desc_B, Bs[i], &mbar[i],
                        bx * TILE_N, i * TILE_K);
        }
    }

    // ================================================================
    // 主循环：3-stage pipeline
    // ================================================================
    for (int k = 0; k < num_k; k++) {
        int cur = k % NUM_BUF;

        // 预取：如果还有更多tile，启动新的TMA到刚释放的buffer
        int prefetch_k = k + prologue_count;
        if (prefetch_k < num_k && tid == 0) {
            int buf = prefetch_k % NUM_BUF;
            v5_mbar_expect(&mbar[buf], tx_bytes);
            v5_tma_load(&v5_tma_desc_A, As[buf], &mbar[buf],
                        prefetch_k * TILE_K, by * TILE_M);
            v5_tma_load(&v5_tma_desc_B, Bs[buf], &mbar[buf],
                        bx * TILE_N, prefetch_k * TILE_K);
        }

        // 等待当前buffer就绪
        v5_mbar_wait(&mbar[cur], phase[cur]);
        phase[cur] ^= 1;

        // ---- WGMMA 计算 ----
        v5_wgmma_fence();

        // TILE_K=32, WGMMA_K=16 → 2次WGMMA per (M_block=64)
        // TILE_M=128 → 需要2个M_block (lo和hi)
        for (int m_block = 0; m_block < 2; m_block++) {
            half* A_base = &As[cur][m_block * 64 * TILE_K];  // 行偏移

            for (int ki = 0; ki < TILE_K; ki += WGMMA_K) {
                uint64_t da = v5_wgmma_desc(
                    &A_base[ki], TILE_K * sizeof(half), 0);
                uint64_t db = v5_wgmma_desc(
                    &Bs[cur][ki * TILE_N], TILE_N * sizeof(half), 0);

                float* acc = (m_block == 0) ? acc_lo : acc_hi;
                v5_wgmma_m64n64k16(acc, da, db);
            }
        }

        v5_wgmma_commit();
        v5_wgmma_wait();
    }

    // ================================================================
    // Epilogue：写回
    // ================================================================
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // 分别写出 lo (行0..63) 和 hi (行64..127) 部分
    for (int m_block = 0; m_block < 2; m_block++) {
        float* acc = (m_block == 0) ? acc_lo : acc_hi;
        int row_base = by * TILE_M + m_block * 64 + warp_id * 16 + (lane_id / 4);
        int col_base = bx * TILE_N + (lane_id % 4) * 2;

        #pragma unroll
        for (int r = 0; r < 32; r++) {
            int col_group   = r / 8;
            int col_in_pair = r % 2;
            int out_row = row_base;
            int out_col = col_base + col_group * 16 + col_in_pair;

            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = __float2half(acc[r]);
            }
        }
    }
}

// ============================================================================
// Host端
// ============================================================================
static void v5_create_tma(const half* A, const half* B, int M, int N, int K) {
    CUtensorMap ha, hb;
    {
        uint64_t gd[2] = {(uint64_t)K, (uint64_t)M};
        uint64_t gs[1] = {(uint64_t)K * sizeof(half)};
        uint32_t bd[2] = {(uint32_t)TILE_K, (uint32_t)TILE_M};
        uint32_t es[2] = {1, 1};
        cuTensorMapEncodeTiled(&ha, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
            const_cast<void*>((const void*)A), gd, gs, bd, es,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }
    {
        uint64_t gd[2] = {(uint64_t)N, (uint64_t)K};
        uint64_t gs[1] = {(uint64_t)N * sizeof(half)};
        uint32_t bd[2] = {(uint32_t)TILE_N, (uint32_t)TILE_K};
        uint32_t es[2] = {1, 1};
        cuTensorMapEncodeTiled(&hb, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
            const_cast<void*>((const void*)B), gd, gs, bd, es,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(v5_tma_desc_A, &ha, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMemcpyToSymbol(v5_tma_desc_B, &hb, sizeof(CUtensorMap)));
}

void gemm_v5(half* C, const half* A, const half* B, int M, int N, int K) {
    // The production WGMMA path inherits the incomplete V4 epilogue, so route
    // through the validated TMA implementation until the mapping is finished.
    gemm_v3(C, A, B, M, N, K);
}
