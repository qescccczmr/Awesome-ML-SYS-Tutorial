# V4: WGMMA (Warp Group Matrix Multiply Accumulate) Introduction

## 核心问题：V3 的瓶颈在哪？

V3 用 TMA 解决了数据搬运效率，但**计算仍是手工 FP32 累加**：

```cpp
// V3 的计算：每个 thread 逐元素处理
for (int ki = 0; ki < TILE_K; ki++) {
    float a = __half2float(As[cur][row * TILE_K + ki]);
    float b = __half2float(Bs[cur][ki * TILE_N + col]);
    acc += a * b;  // CUDA Core FP32 运算
}
```

这完全没用到 H200 的核心计算单元——**第四代 Tensor Core**。

```
┌──────────────────────────────────────────────────────────┐
│              手工FP32累加 (V1-V3)      WGMMA (V4)       │
│  ──────────────────────────────────────────────────── │
│  计算单元    CUDA Core (FP32)        Tensor Core Gen4  │
│  参与线程    单个thread               128 threads协作   │
│  单指令输出  1个FP32                  64×64 FP32矩阵   │
│  理论峰值    ~120 TFLOPS             ~2000 TFLOPS      │
│  输入格式    逐元素half2float         直接FP16描述符    │
│  数据来源    shared memory→寄存器     shared mem描述符  │
└──────────────────────────────────────────────────────────┘
```

## WGMMA 是什么？

WGMMA (Warp Group Matrix Multiply Accumulate) 是 Hopper 架构的 Tensor Core 指令：

- **Warpgroup** = 4 个连续 warp = 128 个 thread
- 这 128 个 thread **协同执行一条指令**
- 一条 `wgmma.mma_async.m64n64k16` 输出 64×64 = 4096 个 FP32 值
- 每个 thread 持有 4096/128 = **32 个 FP32 寄存器**作为累加器

## WGMMA Descriptor

WGMMA 不直接读寄存器中的数据，而是通过 **64-bit descriptor** 描述 shared memory 中的矩阵块：

```
63  62  61..30  29..16  15..14  13..0
[swizzle]  [reserved]  [ldm>>4]  [reserved]  [addr>>4]

bits[0-13]:  shared memory 基地址右移 4 位（16字节对齐）
bits[16-29]: leading dimension 右移 4 位
bits[62-63]: swizzle 模式（0=none, 3=128B）
```

```cpp
uint64_t make_wgmma_desc(void* base, uint32_t ldm_bytes, uint32_t swizzle) {
    uint32_t addr = __cvta_generic_to_shared(base);
    uint64_t desc = 0;
    desc |= (uint64_t(addr >> 4)) & 0x3FFF;            // 地址
    desc |= (uint64_t((ldm_bytes >> 4) & 0x3FFF)) << 16; // leading dim
    desc |= uint64_t(swizzle) << 62;                     // swizzle
    return desc;
}
```

## WGMMA 生命周期

```
wgmma_fence()      ← 标记 WGMMA 操作开始，隔离前后的内存操作
    │
    ├─ wgmma_m64n64k16(acc, desc_A, desc_B)  ← 发出异步 MMA
    ├─ wgmma_m64n64k16(acc, desc_A, desc_B)  ← 可以连续发多条
    │
wgmma_commit()     ← 提交当前组的所有 WGMMA
wgmma_wait()       ← 等待所有 WGMMA 完成，结果在 acc 寄存器中
```

## 累加器寄存器映射

WGMMA m64n64k16 的 32 个 FP32 输出寄存器按如下方式映射到 64×64 输出矩阵：

```
thread 在 warpgroup 中的位置：
  warp_id  = tid / 32    (0..3)
  lane_id  = tid % 32    (0..31)

输出坐标：
  base_row = warp_id * 16 + (lane_id / 4)     → 0..63
  base_col = (lane_id % 4) * 2                 → 0,2,4,6

寄存器 r (0..31) 的映射：
  col_group   = r / 8       → 0,1,2,3 (每组 16 列间隔)
  col_in_pair = r % 2       → 0,1 (相邻两列)
  out_col = base_col + col_group * 16 + col_in_pair
```

这个映射在 epilogue 阶段用于将累加器写回 global memory。

## Kernel 结构

```
Block: 128 threads = 1 warpgroup
Tile:  64×64 output (一条 WGMMA 指令的输出)

初始化 mbar[0], mbar[1]
│
├─ Prologue: TMA 预取 tile[0]
│
└─ Main Loop:
    ├─ TMA 预取 next tile
    ├─ mbar_wait(cur)
    │
    ├─ wgmma_fence()
    ├─ WGMMA #1: A[64×16] × B[16×64] (ki=0)
    ├─ WGMMA #2: A[64×16] × B[16×64] (ki=16)
    ├─ wgmma_commit()
    └─ wgmma_wait()

Epilogue: acc[32] → C[64×64] 通过寄存器映射
```

V4 的 TILE_K=32, WGMMA_K=16，所以每次 mainloop 执行 2 条 WGMMA 指令。

## 编译信息

```
V4 kernel: 58 registers, 0 spill
```

寄存器从 V1/V2 的 128 降到 58！原因：
- 不需要加载数据到寄存器 fragment（WGMMA 从 descriptor 读 shared memory）
- 只需要 32 个 FP32 累加器 + 少量控制寄存器

## 运行

```bash
cd build

# 正确性测试
./v4_wgmma_introduction/test_v4

# 性能对比 V1 vs V2 vs V3 vs V4 vs cuBLAS
./v4_wgmma_introduction/test_v4 --benchmark
```

## 与 V3 的关键差异

| | V3 (TMA) | V4 (WGMMA) |
|---|---|---|
| Block 大小 | 16×16 = 256 threads | 1×128 = 128 threads |
| Output tile | 128×128 | 64×64 |
| 计算方式 | 逐元素 FP32 累加 | Tensor Core m64n64k16 |
| 累加器 | acc[8][8] per thread | acc[32] per thread |
| 数据加载 | TMA → smem → 寄存器 | TMA → smem → descriptor |

## 局限性

- Output tile 只有 64×64（WGMMA 的 M 维度固定 64）
- 要覆盖更大的 tile（如 128×128），需要多次 WGMMA 调用
- 只有 2 个 buffer，流水线深度有限

## 下一步

→ [V5: Production Kernel](../v5_production_kernel/README.md) — 128×128 tile + 3-stage pipeline + 双 WGMMA 块
