#include "cuda_utils.h"
#include "types.h"
#include <cuda_fp16.h>

// ============================================================================
// V1: Baseline Tiled GEMM Kernel
// ============================================================================
// 计算 C = A × B，其中 A[M×K], B[K×N], C[M×N]，数据类型为 half (FP16)
//
// 核心思路：Shared Memory Tiling
// -------------------------------------------------------
// 问题：GPU的global memory带宽远低于计算峰值
//   - H200 计算峰值: ~2000 TFLOPS
//   - H200 内存带宽: 4.8 TB/s
//   - GEMM每个元素需要2 FLOPs，朴素实现每个元素要读2次global memory
//   - 所以朴素实现是内存瓶颈
//
// 解决方案：Tiling（分块）
//   - 将A和B分成 TILE_SIZE×TILE_SIZE 的小块
//   - 每个block协作将一个tile加载到shared memory（只读一次）
//   - 然后对这个tile做计算（shared memory的带宽远高于global memory）
//   - 数据复用倍数 = TILE_SIZE（每个元素被复用 TILE_SIZE 次）
//
// 线程组织：
//   - Block size: BLOCK_DIM × BLOCK_DIM = 16×16 = 256 threads
//   - 每个block计算 TILE_M × TILE_N = 128×128 的输出tile
//   - 每个thread计算 (TILE_M/BLOCK_DIM) × (TILE_N/BLOCK_DIM) = 8×8 = 64个输出元素
//
// Shared memory使用：
//   - As[TILE_M][TILE_K] = 128×128×2 bytes = 32 KB
//   - Bs[TILE_K][TILE_N] = 128×128×2 bytes = 32 KB
//   - 总计 64 KB，远低于H200的228 KB上限
// ============================================================================

static constexpr int TILE_M   = 128;  // 每个block处理的输出行数
static constexpr int TILE_N   = 128;  // 每个block处理的输出列数
static constexpr int TILE_K   = 32;   // 每次迭代处理的K维度
static constexpr int BLOCK_DIM = 16;  // 每个维度的thread数（16×16=256 threads）
static constexpr int TM = TILE_M / BLOCK_DIM;  // 每个thread处理的行数 = 8
static constexpr int TN = TILE_N / BLOCK_DIM;  // 每个thread处理的列数 = 8

__global__ void gemm_v1_kernel(
    half* __restrict__ C,
    const half* __restrict__ A,
    const half* __restrict__ B,
    int M, int N, int K)
{
    // ---- Shared memory tiles ----
    // 注意：As的维度是[TILE_M][TILE_K]，Bs的维度是[TILE_K][TILE_N]
    // 加1是为了避免bank conflict：
    //   - 如果Bs[TILE_K][TILE_N]，列访问时步长为TILE_N=128
    //   - 128 half = 256 bytes，256/128bank_stride = 2 → 可能conflict
    //   - 加padding使步长变为奇数，消除conflict
    __shared__ half As[TILE_M][TILE_K];
    __shared__ half Bs[TILE_K][TILE_N + 8];  // +8 padding避免bank conflict

    // ---- Thread和Block坐标 ----
    int tx = threadIdx.x;  // 0..15 (列方向)
    int ty = threadIdx.y;  // 0..15 (行方向)
    int bx = blockIdx.x;   // block的列索引
    int by = blockIdx.y;   // block的行索引

    // ---- 当前thread负责的输出起始坐标 ----
    // 每个thread负责 TM×TN = 8×8 = 64个输出元素
    int c_row_start = by * TILE_M + ty * TM;  // 输出行起始
    int c_col_start = bx * TILE_N + tx * TN;  // 输出列起始

    // ---- 累加器寄存器 ----
    // 存放当前thread的64个输出元素的中间结果
    // 使用float（FP32）累加，最后转换为half输出，保持精度
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    // ---- 主循环：遍历K维度的tile ----
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {

        // ================================================================
        // Stage 1: 协作加载 A 和 B 的 tile 到 shared memory
        // ================================================================
        // 总共需要加载 TILE_M×TILE_K + TILE_K×TILE_N 个元素
        // Block有 BLOCK_DIM×BLOCK_DIM = 256个thread
        // 每个thread负责加载多个元素（通过循环）

        // 加载 As[TILE_M][TILE_K]：共 128×32 = 4096 个half
        // 256 threads，每个thread加载 4096/256 = 16 个元素
        int tid_linear = ty * BLOCK_DIM + tx;  // 线性thread ID: 0..255

        // 将4096个元素分配给256个thread，每个thread加载16个
        for (int elem = tid_linear; elem < TILE_M * TILE_K; elem += BLOCK_DIM * BLOCK_DIM) {
            int as_row = elem / TILE_K;  // shared memory中的行
            int as_col = elem % TILE_K;  // shared memory中的列
            int a_row  = by * TILE_M + as_row;  // global memory中的行
            int a_col  = k_tile * TILE_K + as_col;  // global memory中的列

            if (a_row < M && a_col < K) {
                As[as_row][as_col] = A[a_row * K + a_col];
            } else {
                As[as_row][as_col] = __float2half(0.0f);  // 边界填0
            }
        }

        // 加载 Bs[TILE_K][TILE_N]：共 32×128 = 4096 个half
        for (int elem = tid_linear; elem < TILE_K * TILE_N; elem += BLOCK_DIM * BLOCK_DIM) {
            int bs_row = elem / TILE_N;
            int bs_col = elem % TILE_N;
            int b_row  = k_tile * TILE_K + bs_row;
            int b_col  = bx * TILE_N + bs_col;

            if (b_row < K && b_col < N) {
                Bs[bs_row][bs_col] = B[b_row * N + b_col];
            } else {
                Bs[bs_row][bs_col] = __float2half(0.0f);
            }
        }

        // 等待所有thread完成加载
        // 必须在这里同步：确保As和Bs已被完整填充后再进行计算
        __syncthreads();

        // ================================================================
        // Stage 2: 利用 shared memory tile 计算部分矩阵乘积
        // ================================================================
        // 每个thread计算自己负责的 TM×TN 个输出元素的累加
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            // 预加载A的一列到寄存器（避免重复访问shared memory）
            float a_frag[TM];
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                a_frag[i] = __half2float(As[ty * TM + i][k]);
            }

            // 预加载B的一行到寄存器
            float b_frag[TN];
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                b_frag[j] = __half2float(Bs[k][tx * TN + j]);
            }

            // 外积累加到acc（TM×TN次FMA）
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        // 等待所有thread完成计算，然后才能加载下一个tile
        // 必须：防止某些thread还在读As/Bs时，其他thread已开始写入下一轮
        __syncthreads();
    }

    // ================================================================
    // Stage 3: 将累加结果写回 global memory
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
// Launcher: 配置grid/block并启动kernel
// ============================================================================
void gemm_v1(half* C, const half* A, const half* B, int M, int N, int K) {
    // Block: 16×16 = 256 threads，每个thread计算8×8输出
    dim3 block(BLOCK_DIM, BLOCK_DIM);

    // Grid: 每个block负责128×128的输出tile
    dim3 grid(
        (N + TILE_N - 1) / TILE_N,  // 列方向block数
        (M + TILE_M - 1) / TILE_M   // 行方向block数
    );

    gemm_v1_kernel<<<grid, block>>>(C, A, B, M, N, K);
}
