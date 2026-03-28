# V3: TMA (Tensor Memory Accelerator) Introduction

## 核心问题：V2 的瓶颈在哪？

V2 使用 `cp.async` 实现了异步数据搬运，但仍然有根本性的效率问题：

```
┌────────────────────────────────────────────────────────────┐
│                   cp.async (V2)              TMA (V3)      │
│  ────────────────────────────────────────────────────────  │
│  发起者       所有256个thread               仅thread 0     │
│  地址计算     SM自己算(循环)                TMA硬件自动算   │
│  指令数       ~256条cp.async                1条PTX指令     │
│  寄存器       需要循环计数器+地址           几乎不需要      │
│  带宽         受SM指令吞吐限制              专用DMA引擎     │
│  同步         cp.async.wait_group           mbarrier       │
└────────────────────────────────────────────────────────────┘
```

V2 中每个 thread 都要执行地址计算和 `cp.async` 指令，256 个线程发出 256 次拷贝命令。SM 的指令发射单元成为瓶颈——时间都花在「安排搬运」而不是「做计算」上了。

## TMA 是什么？

TMA (Tensor Memory Accelerator) 是 Hopper 架构新增的**硬件 DMA 引擎**：

1. **Host 端**创建一个 descriptor（`CUtensorMap`），描述矩阵的形状、数据类型、tile 大小
2. 上传 descriptor 到 `__constant__` 内存
3. **Device 端** thread 0 发一条 PTX 指令，TMA 硬件自动完成整个 tile 的搬运
4. 所有线程通过 `mbarrier` 等待搬运完成

## TMA Descriptor 创建

```cpp
// Host 端：告诉 TMA 引擎矩阵 A[M][K] 的形状
uint64_t globalDim[2]     = {K, M};              // dim0=列(K), dim1=行(M)
uint64_t globalStrides[1] = {K * sizeof(half)};   // 行间距(字节)
uint32_t boxDim[2]        = {TILE_K, TILE_M};     // 每次搬运的 tile 大小

cuTensorMapEncodeTiled(&desc_A, ..., globalDim, globalStrides, boxDim, ...);
```

关键理解：
- `globalDim[0]` 是**最内层维度**（对 row-major 矩阵就是列数）
- `globalStrides` 从 dim1 开始（dim0 的 stride 由数据类型隐含）
- `boxDim` 对应搬运的 tile 尺寸

## mbarrier 同步模型

mbarrier 是 TMA 的同步机制，替代了 `cp.async.wait_group`：

```
生产者 (thread 0):     mbar_expect_tx(bytes)  →  告诉 barrier 期望接收 N 字节
搬运者 (TMA 硬件):     cp.async.bulk.tensor   →  完成后自动 arrive
消费者 (所有 thread):  mbar_wait(phase)        →  等待 barrier 完成
```

### Phase 机制

mbarrier 使用 phase（0/1 交替）避免 race condition：

```
时间 →

mbar[0]:  phase=0           phase=1           phase=0
          [arrive+TMA]      [wait完成]        [下一轮TMA]

mbar[1]:       phase=0           phase=1
               [arrive+TMA]      [wait完成]

计算:              [buf0计算]        [buf1计算]
```

每次使用完一个 buffer 后，phase 翻转 (`phase ^= 1`)。下次等待这个 buffer 时用新的 phase，保证等到的是新一轮 TMA 的完成信号。

## Kernel 结构

```
初始化 mbar[0], mbar[1]  (仅 thread 0)
│
├─ Prologue: TMA 预取 tile[0] → buf[0]
│
└─ Main Loop (k = 0 .. num_k-1):
    │
    ├─ 预取: TMA tile[k+1] → buf[next]     (thread 0)
    │
    ├─ 等待: mbar_wait(cur, phase[cur])     (所有 thread)
    │   phase[cur] ^= 1
    │
    ├─ 计算: FP32 累加（与 V1/V2 相同）
    │
    └─ (无需 __syncthreads，phase 机制保证安全)

Epilogue: 写回 C
```

## 代码关键片段

### TMA 加载（1 条 PTX 搬运整个 tile）

```cpp
// thread 0 发起 TMA：将 A 矩阵的一个 128×32 tile 搬到 shared memory
asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
    :: "r"(smem_ptr), "l"(tma_desc), "r"(coord_x), "r"(coord_y), "r"(mbar_ptr)
    : "memory");
```

对比 V2 需要每个线程循环发出多次 `cp.async`，V3 只需 thread 0 执行 1 条指令。

### mbarrier 等待（自旋等待 phase 匹配）

```cpp
asm volatile(
    "{ .reg .pred p;                                        \n"
    "LAB_WAIT_%=:                                           \n"
    "  mbarrier.try_wait.parity.shared.b64 p, [%0], %1;    \n"
    "  @!p bra LAB_WAIT_%=;                                 \n"
    "}"
    :: "r"(mbar_ptr), "r"(phase) : "memory");
```

## 编译信息

```
V3 kernel: 100 registers, 0 spill, dynamic shared memory
```

相比 V2 的 128 寄存器，V3 降到 100——因为不再需要地址计算的寄存器。

## 运行

```bash
cd build

# 正确性测试
./v3_tma_introduction/test_v3

# 性能对比 V1 vs V2 vs V3 vs cuBLAS
./v3_tma_introduction/test_v3 --benchmark
```

## 局限性

V3 的计算部分仍然使用 CUDA Core 做逐元素 FP32 累加，没有利用 Tensor Core。
这是 V4 要解决的问题。

## 下一步

→ [V4: WGMMA Introduction](../v4_wgmma_introduction/README.md) — 用 Tensor Core 替代 CUDA Core 计算
