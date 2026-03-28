#include "cuda_utils.h"
#include "types.h"
#include <cuda_fp16.h>
#include <stdint.h>

// ============================================================================
// V2: Double Buffering GEMM Kernel
// ============================================================================
// 在V1基础上，引入 Software Double Buffering（软件双缓冲）
//
// V1的问题：
//   阶段1（加载tile到shared memory）和阶段2（计算）是串行的
//   → 加载时SM空转等待，计算时内存接口空闲
//   → 无法掩盖global memory延迟（~400-600 cycles）
//
// Double Buffering解决方案：
//   使用2个shared memory buffer（ping-pong）：
//     - buffer[cur]:  当前正在被计算的tile
//     - buffer[next]: 下一个tile正在被异步加载
//   → 加载和计算重叠（overlap）→ 掩盖内存延迟
//
// 关键新API（sm_80+ cp.async）：
//   cp.async.ca.shared.global：异步拷贝，不阻塞thread
//   cp.async.commit_group：将已发出的cp.async打成一组
//   cp.async.wait_group N：等待直到最多N组未完成（0=全部）
//
// 流水线示意（Prologue + MainLoop + Epilogue）：
//
//   Prologue:  [Async-Load tile0 → buf0]
//
//   k=0: [Async-Load tile1 → buf1]  ←── 与下面的计算重叠
//        [Wait tile0]                ←── 等tile0到位
//        [Compute buf0 (tile0)]
//
//   k=1: [Async-Load tile2 → buf0]  ←── 复用buf0（tile0已用完）
//        [Wait tile1]
//        [Compute buf1 (tile1)]
//
//   ...（交替使用buf0和buf1）
//
//   Epilogue:  [Wait last tile]
//              [Compute last tile]
// ============================================================================

static constexpr int TILE_M    = 128;
static constexpr int TILE_N    = 128;
static constexpr int TILE_K    = 32;
static constexpr int BLOCK_DIM = 16;
static constexpr int TM        = TILE_M / BLOCK_DIM;  // 8
static constexpr int TN        = TILE_N / BLOCK_DIM;  // 8

// cp.async PTX内联汇编宏
// 将src（global memory地址）的size字节异步拷贝到dst（shared memory地址）
// 使用ca（cache all）策略，适合数据会被重用的场景
#define CP_ASYNC_CA(dst, src, bytes) \
    asm volatile( \
        "cp.async.ca.shared.global [%0], [%1], %2;" \
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), \
           "l"(static_cast<const void*>(src)), \
           "n"(bytes) \
        : "memory")

// 将当前所有未提交的cp.async指令打成一个group
// group是cp.async.wait_group的等待粒度
#define CP_ASYNC_COMMIT_GROUP() \
    asm volatile("cp.async.commit_group;" ::: "memory")

// 等待直到inflight的group数量 <= N
// N=0: 等待所有cp.async完成（完全同步）
// N=1: 等待所有group完成，除了最近提交的那1个
#define CP_ASYNC_WAIT(N) \
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory")

// ============================================================================
// 核心Kernel
// ============================================================================
__global__ void gemm_v2_kernel(
    half* __restrict__ C,
    const half* __restrict__ A,
    const half* __restrict__ B,
    int M, int N, int K)
{
    // ---- 双缓冲shared memory ----
    // 相比V1（单个buffer），这里分配2倍：
    //   As[2][TILE_M][TILE_K] = 2 × 128×32×2B = 16KB
    //   Bs[2][TILE_K][TILE_N+8] = 2 × 32×136×2B ≈ 17KB
    //   总计 ≈ 33KB，仍远低于228KB上限
    __shared__ __align__(16) half As[2][TILE_M][TILE_K];
    __shared__ __align__(16) half Bs[2][TILE_K][TILE_N + 8];  // +8 padding避免bank conflict

    int tx = threadIdx.x;  // 0..15
    int ty = threadIdx.y;  // 0..15
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 当前thread负责的输出起始位置
    int c_row_start = by * TILE_M + ty * TM;
    int c_col_start = bx * TILE_N + tx * TN;

    // FP32累加器（8×8 = 64个）
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    int tid = ty * BLOCK_DIM + tx;  // 线性thread ID：0..255

    // ================================================================
    // 辅助函数：异步加载tile k_tile 到 buffer buf_idx
    // ================================================================
    // 每次用cp.async加载16字节（8个half）
    // 256个thread每轮可发出 256 × 1次 = 256次异步请求
    // TILE_M×TILE_K = 128×32 = 4096个half = 512次16字节请求
    //   → 每个thread负责 512/256 = 2次As加载
    // 同理Bs也是2次
    auto async_load = [&](int k_tile, int buf) {
        // ---- 加载 As[buf][TILE_M][TILE_K] ----
        // 每个thread加载16字节（8个half）的步进方式遍历
        for (int base = tid; base < (TILE_M * TILE_K) / 8; base += BLOCK_DIM * BLOCK_DIM) {
            int as_row = (base * 8) / TILE_K;
            int as_col = (base * 8) % TILE_K;
            int a_row  = by * TILE_M + as_row;
            int a_col  = k_tile * TILE_K + as_col;
            bool can_vector_load_a =
                (a_row < M) && (a_col < K) && ((a_col + 7) < K);

            if (can_vector_load_a) {
                const half* a_src = &A[a_row * K + a_col];
                can_vector_load_a =
                    ((reinterpret_cast<uintptr_t>(a_src) & 0xF) == 0);
            }

            if (can_vector_load_a) {
                // 正常路径：16字节对齐加载
                const half* a_src = &A[a_row * K + a_col];
                CP_ASYNC_CA(&As[buf][as_row][as_col], a_src, 16);
            } else {
                // 边界路径：逐元素加载（安全但较慢）
                for (int e = 0; e < 8; e++) {
                    int r = a_row, c = a_col + e;
                    As[buf][as_row][as_col + e] =
                        (r < M && c < K) ? A[r * K + c] : __float2half(0.0f);
                }
            }
        }

        // ---- 加载 Bs[buf][TILE_K][TILE_N] ----
        for (int base = tid; base < (TILE_K * TILE_N) / 8; base += BLOCK_DIM * BLOCK_DIM) {
            int bs_row = (base * 8) / TILE_N;
            int bs_col = (base * 8) % TILE_N;
            int b_row  = k_tile * TILE_K + bs_row;
            int b_col  = bx * TILE_N + bs_col;
            bool can_vector_load_b =
                (b_row < K) && (b_col < N) && ((b_col + 7) < N);

            if (can_vector_load_b) {
                const half* b_src = &B[b_row * N + b_col];
                can_vector_load_b =
                    ((reinterpret_cast<uintptr_t>(b_src) & 0xF) == 0);
            }

            if (can_vector_load_b) {
                const half* b_src = &B[b_row * N + b_col];
                CP_ASYNC_CA(&Bs[buf][bs_row][bs_col], b_src, 16);
            } else {
                for (int e = 0; e < 8; e++) {
                    int r = b_row, c = b_col + e;
                    Bs[buf][bs_row][bs_col + e] =
                        (r < K && c < N) ? B[r * N + c] : __float2half(0.0f);
                }
            }
        }

        // 将本轮所有cp.async打成一个group，以便后续wait
        CP_ASYNC_COMMIT_GROUP();
    };

    // ================================================================
    // Prologue：预取第一个tile（k_tile=0）到buffer[0]
    // ================================================================
    // 在进入主循环前先把tile0的加载发出去
    // 主循环第一次迭代时，tile0的数据可能已经到了
    async_load(0, 0);
    // 此时有1个group in-flight（tile0的加载）

    // ================================================================
    // 主循环：每次迭代处理一个K-tile
    // ================================================================
    for (int k = 0; k < num_k_tiles; k++) {
        int cur_buf  = k & 1;        // 当前tile用的buffer（0或1交替）
        int next_buf = 1 - cur_buf;  // 下一个tile要用的buffer

        // -- Step 1: 发出下一个tile的异步加载 --
        if (k + 1 < num_k_tiles) {
            async_load(k + 1, next_buf);
            // 此时in-flight的group数：
            //   - group[k+1]: 刚发出的tile[k+1]加载（还在路上）
            //   - group[k]:   上一轮或prologue发出的tile[k]加载
            //
            // wait_group<1>：等到in-flight数量 <= 1
            // → 意味着 group[k]（tile[k]）已完成
            // → 同时 group[k+1]（tile[k+1]）仍可异步进行
            CP_ASYNC_WAIT(1);
        } else {
            // 最后一个tile：不需要预取，等待当前tile完成即可
            CP_ASYNC_WAIT(0);
        }

        // 内存栅栏：确保shared memory对所有thread可见
        // （cp.async完成后数据在L1/shared memory中，需要fence）
        __syncthreads();

        // -- Step 2: 计算当前tile（cur_buf）--
        #pragma unroll
        for (int ki = 0; ki < TILE_K; ki++) {
            // 从shared memory预加载到寄存器（减少重复访问smem）
            float a_frag[TM], b_frag[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                a_frag[i] = __half2float(As[cur_buf][ty * TM + i][ki]);
            #pragma unroll
            for (int j = 0; j < TN; j++)
                b_frag[j] = __half2float(Bs[cur_buf][ki][tx * TN + j]);

            // 8×8 外积累加
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }

        // 计算完毕后同步：
        // 下一轮可能会用async_load覆盖cur_buf（因为计算已完成）
        // 必须确保所有thread的计算都完成，再允许覆盖
        __syncthreads();
    }

    // ================================================================
    // Epilogue：将FP32累加器写回global memory（转换为half）
    // ================================================================
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int c_row = c_row_start + i;
            int c_col = c_col_start + j;
            if (c_row < M && c_col < N) {
                C[c_row * N + c_col] = __float2half(acc[i][j]);
            }
        }
    }
}

// ============================================================================
// Launcher
// ============================================================================
void gemm_v2(half* C, const half* A, const half* B, int M, int N, int K) {
    dim3 block(BLOCK_DIM, BLOCK_DIM);  // 16×16 = 256 threads
    dim3 grid(
        (N + TILE_N - 1) / TILE_N,
        (M + TILE_M - 1) / TILE_M
    );
    gemm_v2_kernel<<<grid, block>>>(C, A, B, M, N, K);
}
