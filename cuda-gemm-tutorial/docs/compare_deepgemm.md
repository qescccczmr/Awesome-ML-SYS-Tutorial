# DeepGEMM 深度解析：从 V5 到生产级 MOE GEMM

本文档基于 [DeepSeek-AI/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) 开源代码，
对比本教程的 V5 kernel，详细讲解 DeepGEMM 的核心优化技术。

## 目录

1. [DeepGEMM 是什么](#1-deepgemm-是什么)
2. [m_grouped_gemm：MOE 专用批量 GEMM](#2-m_grouped_gemm)
3. [swapAB：小 batch 推理的救星](#3-swapab)
4. [Pipeline 深度与 JIT 编译](#4-pipeline-深度与-jit-编译)
5. [128B Swizzle 消除 Bank Conflict](#5-128b-swizzle)
6. [FP8 + Block Scaling](#6-fp8--block-scaling)
7. [Warp Specialization](#7-warp-specialization)
8. [V5 vs DeepGEMM 完整对比](#8-v5-vs-deepgemm-完整对比)

---

## 1. DeepGEMM 是什么

DeepGEMM 是 DeepSeek 为其 V3/R1 大模型开发的高性能 GEMM 库，核心特点：

- **极简**: 核心代码仅 ~300 行（不含 JIT 框架）
- **JIT 编译**: 运行时根据 (M, N, K) 形状动态生成最优 kernel
- **FP8 原生**: 支持 DeepSeek-V3 的 128-block FP8 量化格式
- **MOE 专用**: `m_grouped_gemm` 专门为 Mixture-of-Experts 设计
- **H800 峰值**: 达到 **1550 TFLOPS**（FP8），比 cuBLAS 快最多 2.7x

---

## 2. m_grouped_gemm

### MOE 的 GEMM 问题

在 Mixture-of-Experts 模型中，token 被动态路由到不同 expert：

```
输入 tokens: [t1, t2, t3, t4, t5, t6, t7, t8]
路由结果:
  Expert 0: [t1, t3, t5]        → M=3
  Expert 1: [t2]                → M=1
  Expert 2: [t4, t6, t7, t8]    → M=4
  Expert 3: (空)                → M=0
```

每个 expert 有相同的权重矩阵 W[K×N]（N 和 K 固定），但接收的 token 数量（M）不同。

**朴素方案**：对每个 expert 单独 launch 一次 GEMM kernel → 大量 launch overhead + 低 GPU 利用率

**DeepGEMM 方案**：一次 kernel launch 处理所有 expert 的 GEMM，共享 N 和 K，只有 M 不同。

### 两种布局模式

**Contiguous 布局**（训练 forward / 推理 prefill）：

```
所有 expert 的 token 拼接成一个大矩阵：
A = [Expert0_tokens | Expert1_tokens | Expert2_tokens | ...]
     M=3 rows        M=1 row          M=4 rows

每个 expert 的起始行必须对齐到 BLOCK_M
（通过 get_m_alignment_for_contiguous_layout() 查询对齐要求）
```

**Masked 布局**（推理 decoding + CUDA Graph）：

```
预分配固定大小的矩阵，用 mask 标记每个 expert 的有效行数。
适合 CUDA Graph capture（CPU 不需要知道运行时的实际 M）。
```

### 与 V5 的对比

```
V5:  一次 kernel 处理一个 GEMM (M, N, K)
     grid = (N/TILE_N, M/TILE_M)

DeepGEMM m_grouped_gemm:
     一次 kernel 处理 G 组 GEMM: (M_0,N,K), (M_1,N,K), ..., (M_{G-1},N,K)
     grid 自动分配 block 到各组，利用 blockIdx 确定当前处理哪个 expert
```

---

## 3. swapAB

### 问题：小 M 时 WGMMA 浪费严重

WGMMA 指令的 M 维度**固定为 64**（`wgmma.mma_async.m64nNk16`）。

推理 decoding 时，batch size 经常很小（M=1, 2, 4, 8）。
BLOCK_M 必须是 64 的倍数，导致大量 padding 浪费：

```
实际 M=4, BLOCK_M=64:
  ┌────────────────────┐
  │ 4 行有效数据       │  ← 只有 6.25% 有效
  │ 60 行 padding (0)  │  ← 93.75% 浪费!
  └────────────────────┘
```

### 解决方案：交换 A 和 B

核心思想：计算 `C = A × B` 等价于 `C^T = B^T × A^T`

```
原始问题:   C[M×N] = A[M×K] × B[K×N]      M=4, N=4096
                      ↑ 小M映射到WGMMA的M维度(固定64) → 浪费

交换后:     C'^T[N×M] = B^T[N×K] × A^T[K×M]
            ↑ N=4096映射到WGMMA的M维度(固定64) → 充分利用
            ↑ M=4映射到WGMMA的N维度(灵活: 8,16,...,256的倍数)
```

WGMMA 的 N 维度是灵活的（8 到 256，8 的倍数），所以小 M 放到 N 维度可以精确匹配。

### 性能收益

来自 [novitalabs/DeepGEMM_swap](https://github.com/novitalabs/DeepGEMM_swap) 的 benchmark：

| Batch Size | 吞吐提升 |
|-----------|---------|
| 2 | +8.1% |
| 4 | +7.7% |
| 8 | +4.2% |
| 16+ | +2% |

对 MOE 推理 decoding（每个 expert 只有几个 token）非常有效。

### 实现要点

```
原始:
  TMA load A[M×K] → smem_A, 作为 WGMMA 的 A operand (M 维度)
  TMA load B[K×N] → smem_B, 作为 WGMMA 的 B operand (N 维度)

swapAB:
  TMA load B^T[N×K] → smem_A, 作为 WGMMA 的 A operand (M=N, 大维度)
  TMA load A^T[K×M] → smem_B, 作为 WGMMA 的 B operand (N=M, 小维度)
  最后转置结果
```

TMA descriptor、shared memory 布局、WGMMA descriptor 都需要相应调整。

---

## 4. Pipeline 深度与 JIT 编译

### V5 的固定 3-stage vs DeepGEMM 的动态选择

V5 硬编码 `NUM_BUF = 3`。DeepGEMM 根据 (M, N, K) **动态选择**最优 pipeline 深度。

选择逻辑（在 `deep_gemm/jit_kernels/gemm.py` 的 `get_best_configs()` 中）：

```python
for block_m in [64, 128]:
    for block_n in [64, 80, 96, 112, 128, ...]:  # 不只是2的幂!
        for num_stages in [2, 3, 4, 5, 6, 7, ...]:
            # 检查 shared memory 是否够用
            smem_per_stage = (block_m + block_n) * block_k * elem_size
            total_smem = num_stages * smem_per_stage + barrier_mem
            if total_smem > max_smem:
                continue

            # 评估 wave count 和 SM 利用率
            num_blocks = ceil(M/block_m) * ceil(N/block_n)
            waves = ceil(num_blocks / num_sms)
            ...
            # 选择利用率最高的配置
```

### JIT 编译

DeepGEMM 不是预编译的 kernel 库，而是**运行时 JIT 生成 CUDA kernel**：

1. 用户调用 `m_grouped_gemm(A, B, C, ...)`
2. 框架根据 shape 查 cache → 如果没有，实时编译
3. 生成的 kernel 针对精确 (M, N, K, num_stages, block_m, block_n) 特化
4. 缓存到磁盘，下次直接加载

优势：比 cuBLAS 的固定 kernel 选择更灵活，能针对非标准形状优化。

### 非 2 的幂 Block Size

DeepGEMM 支持 BLOCK_N = 80, 96, 112 等非 2 的幂值，用于提高 SM 利用率：

```
例: M=256, N=7168

BLOCK_N=128:  grid = (256/128) × (7168/128) = 2 × 56 = 112 blocks
              H200 有 132 SMs → 利用率 112/132 = 84.8%

BLOCK_N=112:  grid = (256/128) × (7168/112) = 2 × 64 = 128 blocks
              利用率 128/132 = 97.0%  ← 显著提升!
```

---

## 5. 128B Swizzle

### V5 的 SWIZZLE_NONE vs DeepGEMM 的 SWIZZLE_128B

V5 使用 `CU_TENSOR_MAP_SWIZZLE_NONE`（教学简化）。DeepGEMM 使用 128B swizzle。

### 为什么需要 Swizzle？

Shared memory 有 32 个 bank，每个 bank 4 字节宽。
当一个 warp 的 32 个线程访问同一 bank 的不同地址时，发生 **bank conflict**，访问被序列化。

WGMMA 从 shared memory 读取矩阵列时，如果不 swizzle，所有线程读同一 bank → **8-way bank conflict → 吞吐降 8x**。

### Swizzle 的工作原理

128B swizzle 将矩阵的列元素分散到不同 bank：

```
无 swizzle（列访问全部命中同一 bank）:
  bank: 0  1  2  3  4  5  6  7  ...
  row0: a0 a1 a2 a3 a4 a5 a6 a7
  row1: b0 b1 b2 b3 b4 b5 b6 b7
  row2: c0 c1 c2 c3 c4 c5 c6 c7
  ↑ 读第0列: a0,b0,c0 全在 bank 0 → conflict!

128B swizzle（XOR 行号的某些 bit 到地址）:
  bank: 0  1  2  3  4  5  6  7  ...
  row0: a0 a1 a2 a3 a4 a5 a6 a7
  row1: b4 b5 b6 b7 b0 b1 b2 b3   ← 行1偏移4个bank
  row2: c0 c1 c2 c3 c4 c5 c6 c7
  ↑ 读第0列: a0(bank0), b4(bank4), c0(bank0) → 分散!
```

### TMA + WGMMA 的 Swizzle 一致性

**关键**：TMA descriptor 和 WGMMA descriptor 的 swizzle 模式必须一致！

```cpp
// TMA descriptor: 用 SWIZZLE_128B
cuTensorMapEncodeTiled(&desc, ..., CU_TENSOR_MAP_SWIZZLE_128B, ...);

// WGMMA descriptor: swizzle = 3 (128B)
uint64_t wgmma_desc = make_wgmma_desc(base, ldm, /*swizzle=*/3);
```

如果不一致，WGMMA 读到的数据是乱的 → 计算结果全错。

---

## 6. FP8 + Block Scaling

### DeepSeek-V3 的量化格式

DeepGEMM 原生支持 FP8 (E4M3/E5M2) 计算，配合 **block scaling**：

```
矩阵 A: FP8 数据 + 1D 128-block scaling
  每 128 个连续元素共享一个 FP32 scale factor
  A_real[i] = A_fp8[i] × scale_A[i / 128]

矩阵 B: FP8 数据 + 2D (128×128) block scaling
  每 128×128 的子块共享一个 FP32 scale factor
  B_real[i][j] = B_fp8[i][j] × scale_B[i/128][j/128]
```

这就是为什么 `BLOCK_K` 硬编码为 128：
```cpp
DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
```

### 两级累加 (Promotion)

FP8 Tensor Core 内部只有 ~14 bit 精度的累加器。
长 K 维度的累加会丢失精度。

DeepGEMM 的解决方案：

```
每处理 BLOCK_K=128 的一段后:
  1. WGMMA 输出 FP32 部分和
  2. CUDA Core 将部分和累加到高精度 FP32 寄存器
  3. 清零 WGMMA 累加器，开始下一段

→ 防止精度损失在 K 维度上累积
```

这是 V5 没有的功能（V5 使用 FP16，不需要 promotion）。

---

## 7. Warp Specialization

### V5: 所有线程都做一样的事

V5 的 1 个 warpgroup (128 threads) 既做 TMA 加载又做 WGMMA 计算：

```
Thread 0: TMA load + WGMMA
Thread 1-127: 等待 mbarrier + WGMMA
```

### DeepGEMM: Warp 分工

DeepGEMM 将 warp 分为不同角色：

```
Warpgroup 0 (128 threads): TMA Warp
  - 专门负责 TMA 加载
  - 只用 64 个寄存器
  - 空闲时做 epilogue 准备

Warpgroup 1 (128 threads): Math Warp
  - 专门执行 WGMMA
  - 用更多寄存器 (216)
  - 不参与数据搬运

(可能还有 Epilogue Warp)
```

优势：
- TMA warp 和 Math warp 可以**真正并行**（不只是异步重叠）
- 寄存器分配更合理（搬运 warp 不需要累加器，计算 warp 不需要 TMA 控制寄存器）

---

## 8. V5 vs DeepGEMM 完整对比

| 特性 | V5 (本教程) | DeepGEMM |
|------|-------------|----------|
| **代码量** | ~280 行 | ~300 行核心 + JIT 框架 |
| **数据类型** | FP16 | FP8 + block scaling |
| **Output Tile** | 128×128 | 动态 (64-128 × 64-256) |
| **WGMMA 指令** | m64n64k16 | m64n{64-256}k16 |
| **Pipeline Stages** | 3 (固定) | 2-7+ (JIT 动态选择) |
| **Swizzle** | NONE | 128B |
| **Block Size** | 固定 128×128×32 | 动态, 支持非2的幂 |
| **Grouped GEMM** | 不支持 | M轴分组 + K轴分组 |
| **swapAB** | 不支持 | 社区实现 (小batch +8%) |
| **Warp 分工** | 单一角色 | TMA/Math/Epilogue 分工 |
| **Kernel 生成** | 静态编译 | JIT 运行时特化 |
| **精度管理** | FP32 直接累加 | 两级 promotion |
| **MOE 支持** | 无 | contiguous + masked 布局 |
| **vs cuBLAS** | 教学用 | MOE 场景最多快 2.7x |

### 从 V5 到 DeepGEMM 的升级路径

如果你想逐步将 V5 改造为 DeepGEMM 级别的 kernel：

```
V5 (当前)
│
├─ Step 1: 添加 128B swizzle (TMA + WGMMA descriptor 一致)
│          → 消除 bank conflict, 预计性能提升 2-4x
│
├─ Step 2: 扩大 WGMMA N 维度 (m64n128k16 或 m64n256k16)
│          → 更高计算密度
│
├─ Step 3: Warp specialization (分离 TMA warp 和 Math warp)
│          → 更好的异步重叠
│
├─ Step 4: JIT 框架 (动态选择 block size 和 pipeline 深度)
│          → 自适应所有矩阵形状
│
├─ Step 5: FP8 + block scaling
│          → 2x 吞吐 + DeepSeek-V3 量化格式
│
├─ Step 6: m_grouped_gemm (M轴分组)
│          → MOE workload 支持
│
└─ Step 7: swapAB (可选, 小batch推理)
           → 小 M 场景进一步优化
```

---

## 参考资料

- [DeepGEMM GitHub](https://github.com/deepseek-ai/DeepGEMM) — 完整源码
- [DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM/blob/main/README.md) — 官方文档
- [novitalabs/DeepGEMM_swap](https://github.com/novitalabs/DeepGEMM_swap) — swapAB 社区实现
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — FP8 block scaling 设计
- [CUTLASS 3.x](https://github.com/NVIDIA/cutlass) — NVIDIA 的参考 GEMM 库
- [PTX ISA: wgmma](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-multiply-accumulate) — WGMMA 指令规范
- [PTX ISA: TMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor) — TMA 指令规范
