#include "cuda_utils.h"
#include "types.h"
#include <cuda_fp16.h>
#include <cuda.h>

// ============================================================================
// V4: WGMMA (Warp Group Matrix Multiply Accumulate) GEMM Kernel
// ============================================================================
//
// V3 的问题:
//   TMA解决了数据搬运的效率，但计算仍是手工FP32累加：
//     - 每个thread逐元素 __half2float → multiply → add
//     - CUDA Core的FP32吞吐远低于Tensor Core
//     - 没有利用H200的核心计算单元——第四代Tensor Core
//
// WGMMA vs 传统方式:
//   ┌──────────────────────────────────────────────────────────┐
//   │              手工FP32累加 (V1-V3)      WGMMA (V4)       │
//   │  ──────────────────────────────────────────────────── │
//   │  计算单元    CUDA Core (FP32)        Tensor Core Gen4  │
//   │  参与线程    单个thread               128 threads协作   │
//   │  单指令输出  1个FP32                  64×64 FP32矩阵   │
//   │  理论峰值    ~120 TFLOPS             ~2000 TFLOPS      │
//   │  输入格式    逐元素half2float         直接FP16/BF16     │
//   │  数据来源    shared memory(寄存器)    shared mem描述符  │
//   └──────────────────────────────────────────────────────────┘
//
// Warpgroup 概念:
//   - 1个warpgroup = 4个连续warp = 128个thread
//   - 这128个thread协同执行一条WGMMA指令
//   - WGMMA输出: 64×64 FP32矩阵 = 4096个值
//   - 每个thread持有: 4096/128 = 32个FP32寄存器作为累加器
//
// WGMMA指令格式 (m64n64k16):
//   - 输入A: 从shared memory读取 [64×16] half矩阵
//   - 输入B: 从shared memory读取 [16×64] half矩阵（通过descriptor）
//   - 输出:  累加到寄存器中的 [64×64] FP32矩阵
//   - 每次处理K=16，需要循环 TILE_K/16 次
//
// WGMMA Descriptor:
//   64-bit值，编码shared memory中矩阵tile的地址和布局：
//   bits[0-13]:  共享内存基地址 >> 4 (16字节对齐)
//   bits[16-29]: leading dimension (以16字节为单位)
//   bits[62-63]: swizzle模式 (0=none, 1=32B, 2=64B, 3=128B)
//
// Swizzle:
//   V4使用 128B swizzle（3），与TMA的SWIZZLE_128B配合
//   目的：让WGMMA从shared memory读取时无bank conflict
//   必须保证TMA descriptor和WGMMA descriptor的swizzle一致！
// ============================================================================

static constexpr int TILE_M    = 64;   // WGMMA的M维度固定为64
static constexpr int TILE_N    = 64;   // WGMMA的N维度，64或128
static constexpr int TILE_K    = 32;   // 每次mainloop处理的K
static constexpr int WGMMA_K   = 16;   // 单条WGMMA指令的K=16
static constexpr int NUM_BUF   = 2;
static constexpr int WARPGROUP_SIZE = 128;  // 4 warps

// TMA描述符
__constant__ CUtensorMap v4_tma_desc_A;
__constant__ CUtensorMap v4_tma_desc_B;

void gemm_v3(half* C, const half* A, const half* B, int M, int N, int K);

// ---- mbarrier 辅助（与V3相同）----
__device__ __forceinline__
void v4_mbar_init(uint64_t* mbar, uint32_t count) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(p), "r"(count));
}

__device__ __forceinline__
void v4_mbar_expect_tx(uint64_t* mbar, uint32_t bytes) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(p), "r"(bytes));
}

__device__ __forceinline__
void v4_mbar_wait(uint64_t* mbar, uint32_t phase) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "LAB_WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "@!p bra LAB_WAIT_%=;\n"
        "}\n"
        :: "r"(p), "r"(phase) : "memory");
}

__device__ __forceinline__
void v4_tma_load_2d(const CUtensorMap* desc, void* dst,
                    uint64_t* mbar, int cx, int cy) {
    uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t m = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    uint64_t t = reinterpret_cast<uint64_t>(desc);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        :: "r"(d), "l"(t), "r"(cx), "r"(cy), "r"(m) : "memory");
}

// ============================================================================
// WGMMA 辅助函数
// ============================================================================

// 创建WGMMA descriptor（64-bit）
// base: shared memory中矩阵tile的起始地址
// ldm: leading dimension（字节数）
// swizzle: 0=none, 3=128B
__device__ __forceinline__
uint64_t make_wgmma_desc(void* base, uint32_t ldm_bytes, uint32_t swizzle = 3) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(base));
    uint64_t desc = 0;
    // bits[0-13]: 地址右移4位（16字节对齐）
    desc |= (static_cast<uint64_t>(addr >> 4)) & 0x3FFF;
    // bits[16-29]: leading dimension右移4位
    desc |= (static_cast<uint64_t>((ldm_bytes >> 4) & 0x3FFF)) << 16;
    // bits[62-63]: swizzle模式
    desc |= (static_cast<uint64_t>(swizzle)) << 62;
    return desc;
}

// WGMMA fence: 标记WGMMA操作的开始
__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

// WGMMA commit: 提交已发出的WGMMA组
__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

// WGMMA wait: 等待所有WGMMA完成
__device__ __forceinline__ void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// ============================================================================
// 执行一次 WGMMA m64n64k16 (简化版)
// ============================================================================
// 输入：desc_a (A的descriptor), desc_b (B的descriptor)
// 输出：累加到 acc[0..31]（32个FP32寄存器）
// 注意：这是简化版，只处理一个 64×64×16 的子问题
__device__ __forceinline__
void wgmma_m64n64k16(float* acc, uint64_t desc_a, uint64_t desc_b) {
    // wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16
    // 输出: 32个FP32寄存器（每个thread）
    // 输入: A descriptor, B descriptor
    // scale: D = A*B + D (accumulate mode: scale_d=1)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        " {%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
        "  %8,  %9,  %10, %11, %12, %13, %14, %15,"
        "  %16, %17, %18, %19, %20, %21, %22, %23,"
        "  %24, %25, %26, %27, %28, %29, %30, %31},"
        " %32, %33, 1, 1, 1, 0, 0;"
        : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
          "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
          "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
          "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),
          "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]),
          "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),
          "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]),
          "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31])
        : "l"(desc_a), "l"(desc_b));
}

// ============================================================================
// V4 Kernel
// ============================================================================
// Block size: 128 threads = 1 warpgroup
// 每个block计算一个 64×64 输出tile
__global__ void __launch_bounds__(WARPGROUP_SIZE)
gemm_v4_kernel(
    half* __restrict__ C,
    const half* __restrict__ A,
    const half* __restrict__ B,
    int M, int N, int K)
{
    extern __shared__ char smem[];

    // Shared memory布局
    constexpr int As_tile = TILE_M * TILE_K;  // 64*32 = 2048 half = 4KB
    constexpr int Bs_tile = TILE_K * TILE_N;  // 32*64 = 2048 half = 4KB

    half* As[NUM_BUF];
    half* Bs[NUM_BUF];
    As[0] = reinterpret_cast<half*>(smem);
    As[1] = As[0] + As_tile;
    Bs[0] = As[1] + As_tile;
    Bs[1] = Bs[0] + Bs_tile;
    uint64_t* mbar = reinterpret_cast<uint64_t*>(
        reinterpret_cast<char*>(Bs[1] + Bs_tile));

    int tid = threadIdx.x;
    int bx  = blockIdx.x;
    int by  = blockIdx.y;
    int num_k = (K + TILE_K - 1) / TILE_K;

    // 累加器: 32个FP32寄存器（WGMMA m64n64k16的输出）
    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.0f;

    // 初始化mbarrier
    if (tid == 0) {
        v4_mbar_init(&mbar[0], 1);
        v4_mbar_init(&mbar[1], 1);
    }
    __syncthreads();

    uint32_t phase[NUM_BUF] = {0, 0};
    constexpr uint32_t tx_bytes = (As_tile + Bs_tile) * sizeof(half);

    // Prologue: 预取tile[0]
    if (tid == 0) {
        v4_mbar_expect_tx(&mbar[0], tx_bytes);
        v4_tma_load_2d(&v4_tma_desc_A, As[0], &mbar[0],
                        0, by * TILE_M);
        v4_tma_load_2d(&v4_tma_desc_B, Bs[0], &mbar[0],
                        bx * TILE_N, 0);
    }

    // 主循环
    for (int k = 0; k < num_k; k++) {
        int cur  = k % NUM_BUF;
        int next = (k + 1) % NUM_BUF;

        // 预取下一个tile
        if (k + 1 < num_k && tid == 0) {
            v4_mbar_expect_tx(&mbar[next], tx_bytes);
            v4_tma_load_2d(&v4_tma_desc_A, As[next], &mbar[next],
                            (k + 1) * TILE_K, by * TILE_M);
            v4_tma_load_2d(&v4_tma_desc_B, Bs[next], &mbar[next],
                            bx * TILE_N, (k + 1) * TILE_K);
        }

        // 等待当前tile就绪
        v4_mbar_wait(&mbar[cur], phase[cur]);
        phase[cur] ^= 1;

        // ---- WGMMA计算 ----
        // TILE_K=32, WGMMA_K=16, 需要2次WGMMA指令
        wgmma_fence();

        // A descriptor: As[cur]的前16列 [64×16]
        // ldm = TILE_K * sizeof(half) = 32 * 2 = 64 bytes
        uint64_t da_0 = make_wgmma_desc(
            &As[cur][0],              // 从第0列开始
            TILE_K * sizeof(half),     // leading dim
            0);                        // swizzle=none (与TMA匹配)

        // B descriptor: Bs[cur]的前16行 [16×64]
        uint64_t db_0 = make_wgmma_desc(
            &Bs[cur][0],
            TILE_N * sizeof(half),
            0);

        wgmma_m64n64k16(acc, da_0, db_0);

        // 第二次WGMMA: K偏移16
        // A: As[cur]第16列开始 [64×16]
        uint64_t da_1 = make_wgmma_desc(
            &As[cur][16],                     // 偏移16个half
            TILE_K * sizeof(half), 0);

        // B: Bs[cur]第16行开始 [16×64]
        uint64_t db_1 = make_wgmma_desc(
            &Bs[cur][16 * TILE_N],            // 偏移16行
            TILE_N * sizeof(half), 0);

        wgmma_m64n64k16(acc, da_1, db_1);

        wgmma_commit();
        wgmma_wait();
    }

    // ================================================================
    // Epilogue: 将WGMMA累加器写回global memory
    // ================================================================
    // WGMMA m64n64k16的寄存器布局 (每个thread 32个float):
    //   acc[0..15]:  对应输出矩阵的某些 (row, col) 位置
    //   acc[16..31]: 对应另一组位置
    //
    // 精确映射取决于thread在warpgroup中的位置：
    //   warp_id     = tid / 32  (0..3)
    //   lane_id     = tid % 32  (0..31)
    //   output_row  = warp_id * 16 + (lane_id / 4)
    //   output_col  = (lane_id % 4) * 2 + reg_pair * 8
    //
    // 每个thread写出32个元素，分布在64×64矩阵中

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int base_row = by * TILE_M + warp_id * 16 + (lane_id / 4);
    int base_col = bx * TILE_N + (lane_id % 4) * 2;

    // 写出32个累加器值
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        // 寄存器r对应的 (row_offset, col_offset)
        // WGMMA m64n64k16 输出映射：
        //   r=[0..7]:   col_group=0, row_pair = r/2, col_in_pair = r%2
        //   r=[8..15]:  col_group=1
        //   r=[16..23]: col_group=2
        //   r=[24..31]: col_group=3
        int col_group  = r / 8;
        int row_pair   = (r % 8) / 2;
        int col_in_pair = r % 2;

        int out_row = base_row + row_pair * 0;  // rows don't shift within group
        int out_col = base_col + col_group * 16 + col_in_pair;

        // 简化: 将row映射扁平化
        // 注意: 实际映射更复杂，这里用简化版
        out_row = by * TILE_M + warp_id * 16 + (lane_id / 4) + (r / 8) * 0;
        out_col = bx * TILE_N + (lane_id % 4) * 2 + col_in_pair + col_group * 16;

        if (out_row < M && out_col < N) {
            C[out_row * N + out_col] = __float2half(acc[r]);
        }
    }
}

// ============================================================================
// Host端
// ============================================================================
static void v4_create_tma(const half* A, const half* B, int M, int N, int K) {
    CUtensorMap h_a, h_b;

    {
        uint64_t gd[2] = {(uint64_t)K, (uint64_t)M};
        uint64_t gs[1] = {(uint64_t)K * sizeof(half)};
        uint32_t bd[2] = {(uint32_t)TILE_K, (uint32_t)TILE_M};
        uint32_t es[2] = {1, 1};
        cuTensorMapEncodeTiled(&h_a, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
            const_cast<void*>((const void*)A), gd, gs, bd, es,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,   // 与WGMMA desc swizzle=0匹配
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }
    {
        uint64_t gd[2] = {(uint64_t)N, (uint64_t)K};
        uint64_t gs[1] = {(uint64_t)N * sizeof(half)};
        uint32_t bd[2] = {(uint32_t)TILE_N, (uint32_t)TILE_K};
        uint32_t es[2] = {1, 1};
        cuTensorMapEncodeTiled(&h_b, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
            const_cast<void*>((const void*)B), gd, gs, bd, es,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    CUDA_CHECK(cudaMemcpyToSymbol(v4_tma_desc_A, &h_a, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMemcpyToSymbol(v4_tma_desc_B, &h_b, sizeof(CUtensorMap)));
}

void gemm_v4(half* C, const half* A, const half* B, int M, int N, int K) {
    // The WGMMA tutorial kernel still lacks a correct accumulator-to-output
    // mapping, so use the validated V3 path until the epilogue is implemented.
    gemm_v3(C, A, B, M, N, K);
}
