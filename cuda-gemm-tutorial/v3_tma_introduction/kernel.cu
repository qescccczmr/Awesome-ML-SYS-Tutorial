#include "cuda_utils.h"
#include "types.h"
#include <cuda_fp16.h>
#include <cuda.h>  // CUDA Driver API: CUtensorMap, cuTensorMapEncodeTiled

// ============================================================================
// V3: TMA (Tensor Memory Accelerator) GEMM Kernel
// ============================================================================
//
// V2 的问题:
//   cp.async 虽然是异步的，但仍需要SM执行地址计算指令：
//     - 每个thread都要算 global_addr, smem_addr
//     - 循环发出多次 cp.async（每次16字节，128x32 tile需要256次）
//     - 占用了大量寄存器和指令slot
//
// TMA 的优势:
//   ┌────────────────────────────────────────────────────────┐
//   │                   cp.async (V2)          TMA (V3)     │
//   │  ─────────────────────────────────────────────────── │
//   │  发起者       所有256个thread          仅thread 0     │
//   │  地址计算     SM自己算(循环)           TMA硬件自动算   │
//   │  指令数       ~256条cp.async           1条PTX指令     │
//   │  寄存器       需要循环计数器+地址      几乎不需要      │
//   │  带宽         受SM指令吞吐限制         专用DMA引擎     │
//   │  多维支持     手工计算stride           descriptor自动  │
//   │  同步         cp.async.wait_group      mbarrier       │
//   └────────────────────────────────────────────────────────┘
//
// TMA 工作流:
//   1. Host端: cuTensorMapEncodeTiled 创建描述符 (描述矩阵形状+tile大小)
//   2. Host端: cudaMemcpyToSymbol 上传描述符到 __constant__ 内存
//   3. Device端: thread 0 用1条PTX指令发起整个tile的异步DMA传输
//   4. Device端: 所有thread用 mbarrier 等待传输完成
//
// mbarrier (Memory Barrier) 同步模型:
//   - 生产者 (thread 0): arrive_expect_tx → 告诉barrier期望接收N字节
//   - 搬运者 (TMA硬件): cp.async.bulk.tensor → 完成传输后自动通知barrier
//   - 消费者 (所有thread): try_wait → 等待barrier完成
//   - Phase机制: 0→1→0→1... 交替，避免race condition
//
//   Phase图示:
//   ┌─────────────────────────────────────────────────────┐
//   │  时间 →                                            │
//   │                                                     │
//   │  mbar[0]:  phase=0        phase=1        phase=0   │
//   │            [arrive+TMA]   [wait完成]     [下一轮]  │
//   │                                                     │
//   │  mbar[1]:       phase=0        phase=1             │
//   │                 [arrive+TMA]   [wait完成]          │
//   │                                                     │
//   │  计算:               [buf0计算]      [buf1计算]    │
//   └─────────────────────────────────────────────────────┘
// ============================================================================

static constexpr int TILE_M    = 128;
static constexpr int TILE_N    = 128;
static constexpr int TILE_K    = 32;
static constexpr int BLOCK_DIM = 16;
static constexpr int TM        = TILE_M / BLOCK_DIM;  // 8
static constexpr int TN        = TILE_N / BLOCK_DIM;  // 8
static constexpr int NUM_BUF   = 2;  // 双缓冲

// TMA描述符存放在 constant memory（device可见，host填充）
__constant__ CUtensorMap tma_desc_A;
__constant__ CUtensorMap tma_desc_B;

// ============================================================================
// mbarrier 辅助内联函数
// ============================================================================

// 初始化mbarrier（thread 0调用一次）
// arrive_count: 有多少个arrive会发生（TMA完成自动arrive 1次）
__device__ __forceinline__
void mbar_init(uint64_t* mbar, uint32_t arrive_count) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                 :: "r"(mbar_ptr), "r"(arrive_count));
}

// 告诉mbarrier期望接收 tx_bytes 字节的TMA数据
// 当TMA传输完成这些字节后，mbarrier自动完成
__device__ __forceinline__
void mbar_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                 :: "r"(mbar_ptr), "r"(tx_bytes));
}

// 等待mbarrier完成（自旋等待直到phase匹配）
// phase: 当前期望的奇偶值，0或1交替
__device__ __forceinline__
void mbar_wait(uint64_t* mbar, uint32_t phase) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    // try_wait.parity: 检查mbarrier的phase奇偶性
    // 当barrier完成时phase翻转，与期望值匹配则返回true
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "LAB_WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "@!p bra LAB_WAIT_%=;\n"
        "}\n"
        :: "r"(mbar_ptr), "r"(phase) : "memory");
}

// 使mbarrier失效（用于prologue中未使用的buffer）
__device__ __forceinline__
void mbar_invalidate(uint64_t* mbar) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(mbar_ptr));
}

// ============================================================================
// TMA 加载辅助函数
// ============================================================================

// 发起TMA 2D tile加载：从global memory异步传输到shared memory
// tma_desc: __constant__中的TMA描述符
// dst: shared memory目标地址
// mbar: 关联的mbarrier
// coord_x: tile在最内层维度(K)的起始坐标
// coord_y: tile在外层维度(M或N)的起始坐标
__device__ __forceinline__
void tma_load_2d(const CUtensorMap* tma_desc, void* dst,
                 uint64_t* mbar, int coord_x, int coord_y) {
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    uint64_t desc_ptr = reinterpret_cast<uint64_t>(tma_desc);

    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst_ptr),
           "l"(desc_ptr),
           "r"(coord_x),
           "r"(coord_y),
           "r"(mbar_ptr)
        : "memory");
}

// ============================================================================
// V3 Kernel
// ============================================================================
__global__ void gemm_v3_kernel(
    half* __restrict__ C,
    const half* __restrict__ A,
    const half* __restrict__ B,
    int M, int N, int K)
{
    // ---- Shared memory: 双缓冲 tile + 2个mbarrier ----
    // 注意：mbarrier必须在shared memory中（硬件要求）
    extern __shared__ char smem_raw[];

    // 布局：[As_buf0][As_buf1][Bs_buf0][Bs_buf1][mbar0][mbar1]
    constexpr int As_size = TILE_M * TILE_K;  // 128*32 = 4096 half = 8KB
    constexpr int Bs_size = TILE_K * TILE_N;  // 32*128 = 4096 half = 8KB

    half* As[NUM_BUF];
    half* Bs[NUM_BUF];
    As[0] = reinterpret_cast<half*>(smem_raw);
    As[1] = As[0] + As_size;
    Bs[0] = As[1] + As_size;
    Bs[1] = Bs[0] + Bs_size;

    // mbarrier需要8字节对齐
    uint64_t* mbar = reinterpret_cast<uint64_t*>(
        reinterpret_cast<char*>(Bs[1] + Bs_size));

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = ty * BLOCK_DIM + tx;

    int c_row_start = by * TILE_M + ty * TM;
    int c_col_start = bx * TILE_N + tx * TN;

    // 累加器
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // ================================================================
    // 初始化 mbarrier（仅thread 0执行）
    // ================================================================
    // arrive_count=1: 只有TMA完成时自动arrive一次
    // （不同于__syncthreads()需要所有thread arrive）
    if (tid == 0) {
        mbar_init(&mbar[0], 1);
        mbar_init(&mbar[1], 1);
    }
    __syncthreads();  // 确保mbar初始化完成

    // Phase数组：每个mbarrier的当前期望phase
    uint32_t phase[NUM_BUF] = {0, 0};

    // ================================================================
    // Prologue：预取 tile[0] 到 buf[0]
    // ================================================================
    // TMA传输字节数：As + Bs
    constexpr uint32_t tma_bytes_A = As_size * sizeof(half);  // 8KB
    constexpr uint32_t tma_bytes_B = Bs_size * sizeof(half);  // 8KB

    if (tid == 0) {
        // 告诉 mbar[0]：期望接收 tma_bytes_A + tma_bytes_B 字节
        mbar_expect_tx(&mbar[0], tma_bytes_A + tma_bytes_B);

        // 发起 As[0] 的 TMA 加载
        // 对于 A[M][K] (row-major):
        //   dim0 (innermost) = K → coord_x = k_tile * TILE_K
        //   dim1 (outermost) = M → coord_y = by * TILE_M
        tma_load_2d(&tma_desc_A, As[0], &mbar[0],
                    0,             // coord_x = k_start = 0
                    by * TILE_M);  // coord_y = m_start

        // 发起 Bs[0] 的 TMA 加载
        // 对于 B[K][N] (row-major):
        //   dim0 (innermost) = N → coord_x = bx * TILE_N
        //   dim1 (outermost) = K → coord_y = k_tile * TILE_K
        tma_load_2d(&tma_desc_B, Bs[0], &mbar[0],
                    bx * TILE_N,   // coord_x = n_start
                    0);            // coord_y = k_start = 0
    }

    // ================================================================
    // 主循环
    // ================================================================
    for (int k = 0; k < num_k_tiles; k++) {
        int cur  = k % NUM_BUF;
        int next = (k + 1) % NUM_BUF;

        // -- 预取下一个tile --
        if (k + 1 < num_k_tiles && tid == 0) {
            mbar_expect_tx(&mbar[next], tma_bytes_A + tma_bytes_B);
            tma_load_2d(&tma_desc_A, As[next], &mbar[next],
                        (k + 1) * TILE_K,  by * TILE_M);
            tma_load_2d(&tma_desc_B, Bs[next], &mbar[next],
                        bx * TILE_N,  (k + 1) * TILE_K);
        }

        // -- 等待当前tile的TMA完成 --
        // 所有thread等待 mbar[cur] 的 phase[cur]
        mbar_wait(&mbar[cur], phase[cur]);
        phase[cur] ^= 1;  // 翻转phase，下次使用此buffer时等新phase

        // -- 计算当前tile（与V1/V2相同的FP32累加）--
        // As[cur]布局: [TILE_M][TILE_K] = [128][32]
        // Bs[cur]布局: [TILE_K][TILE_N] = [32][128]
        #pragma unroll
        for (int ki = 0; ki < TILE_K; ki++) {
            float a_frag[TM], b_frag[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                a_frag[i] = __half2float(As[cur][(ty * TM + i) * TILE_K + ki]);
            #pragma unroll
            for (int j = 0; j < TN; j++)
                b_frag[j] = __half2float(Bs[cur][ki * TILE_N + tx * TN + j]);
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }

        // 复用当前buffer之前，需要确保整个block都已经完成对它的读取。
        // 仅靠mbarrier无法约束消费者之间的相对进度；没有这个同步时，
        // 更快的thread可能会提前发起下一轮TMA并覆盖仍在被读取的shared memory。
        __syncthreads();
    }

    // ================================================================
    // Epilogue：写回global memory
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
// Host端：TMA描述符创建
// ============================================================================
// cuTensorMapEncodeTiled 参数详解：
//   tensorMap:      输出的TMA描述符
//   tensorDataType: 数据类型
//   tensorRank:     维度数（2D矩阵=2）
//   globalAddress:  矩阵在global memory的地址
//   globalDim[]:    每个维度的大小（dim0=最内层=列）
//   globalStrides[]:从dim1开始的stride（字节）
//   boxDim[]:       tile大小
//   elementStrides: 元素步长（通常{1,1}）
//   interleave:     交错模式
//   swizzle:        shared memory swizzle模式
//   l2Promotion:    L2缓存策略
//   oobFill:        越界填充模式
static void create_tma_descriptors(
    const half* A, const half* B, int M, int N, int K)
{
    CUtensorMap h_desc_A, h_desc_B;

    // A[M][K]: row-major → dim0=K(列), dim1=M(行)
    {
        uint64_t globalDim[2]     = {(uint64_t)K, (uint64_t)M};
        uint64_t globalStrides[1] = {(uint64_t)K * sizeof(half)};  // dim1 stride
        uint32_t boxDim[2]        = {TILE_K, TILE_M};
        uint32_t elemStrides[2]   = {1, 1};

        CUresult err = cuTensorMapEncodeTiled(
            &h_desc_A,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            2,                                    // rank
            const_cast<void*>((const void*)A),    // globalAddress
            globalDim,
            globalStrides,
            boxDim,
            elemStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,           // V3用NONE保持简单
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        if (err != CUDA_SUCCESS) {
            const char* str;
            cuGetErrorString(err, &str);
            fprintf(stderr, "cuTensorMapEncodeTiled(A) failed: %s\n", str);
            exit(1);
        }
    }

    // B[K][N]: row-major → dim0=N(列), dim1=K(行)
    {
        uint64_t globalDim[2]     = {(uint64_t)N, (uint64_t)K};
        uint64_t globalStrides[1] = {(uint64_t)N * sizeof(half)};
        uint32_t boxDim[2]        = {TILE_N, TILE_K};
        uint32_t elemStrides[2]   = {1, 1};

        CUresult err = cuTensorMapEncodeTiled(
            &h_desc_B,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            2,
            const_cast<void*>((const void*)B),
            globalDim,
            globalStrides,
            boxDim,
            elemStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        if (err != CUDA_SUCCESS) {
            const char* str;
            cuGetErrorString(err, &str);
            fprintf(stderr, "cuTensorMapEncodeTiled(B) failed: %s\n", str);
            exit(1);
        }
    }

    // 上传到 __constant__ memory
    CUDA_CHECK(cudaMemcpyToSymbol(tma_desc_A, &h_desc_A, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMemcpyToSymbol(tma_desc_B, &h_desc_B, sizeof(CUtensorMap)));
}

// ============================================================================
// Launcher
// ============================================================================
void gemm_v3(half* C, const half* A, const half* B, int M, int N, int K) {
    // 创建TMA描述符（每次调用都重建，因为A/B/M/N/K可能变化）
    create_tma_descriptors(A, B, M, N, K);

    dim3 block(BLOCK_DIM, BLOCK_DIM);  // 16×16 = 256 threads
    dim3 grid(
        (N + TILE_N - 1) / TILE_N,
        (M + TILE_M - 1) / TILE_M
    );

    // 动态shared memory大小：
    // 2 × As_size + 2 × Bs_size + 2 × sizeof(uint64_t)
    constexpr int smem_bytes =
        NUM_BUF * TILE_M * TILE_K * sizeof(half) +  // As[2]
        NUM_BUF * TILE_K * TILE_N * sizeof(half) +  // Bs[2]
        NUM_BUF * sizeof(uint64_t) +                 // mbar[2]
        128;  // padding for alignment

    // sm_90允许更大的shared memory
    CUDA_CHECK(cudaFuncSetAttribute(
        gemm_v3_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));

    gemm_v3_kernel<<<grid, block, smem_bytes>>>(C, A, B, M, N, K);
}
