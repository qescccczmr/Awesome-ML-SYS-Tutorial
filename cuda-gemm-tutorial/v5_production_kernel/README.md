# V5: Production Kernel — TMA + WGMMA + 3-Stage Pipeline

## 从 V4 到 V5：集成所有优化

V4 验证了 WGMMA 的正确用法，但 tile 只有 64×64，流水线只有 2 级。
V5 把所有技术集成为一个接近生产质量的 kernel：

| | V4 | V5 |
|---|---|---|
| Output tile | 64×64 | **128×128** |
| WGMMA 次数/K-tile | 2 | **4** (2 M-block × 2 K-step) |
| 流水线深度 | 2-buffer | **3-buffer** |
| 累加器 | 32 个 FP32 | **2×32 = 64 个 FP32** |
| 理论计算密度 | 1× | **4×** |

## 3-Stage Pipeline 原理

```
时间 →
buf0: [TMA load tile0]  [WGMMA tile0]           [TMA load tile3]...
buf1:    [TMA load tile1]  [WGMMA tile1]           ...
buf2:       [TMA load tile2]  [WGMMA tile2]

好处：当 WGMMA_latency ≈ TMA_latency 时，流水线满载
      3个buffer保证始终有1个可用于TMA、1个可用于WGMMA、1个在等待
```

对比 2-buffer（V3/V4）：2-buffer 在 TMA 和 WGMMA 延迟不匹配时会产生 bubble。
3-buffer 多一个「缓冲池」，能吸收延迟差异。

## 128×128 Tile 的 WGMMA 覆盖

WGMMA 的 M 维度固定为 64，所以 128 行需要分两次：

```
TILE_M = 128
│
├─ m_block=0 (行 0..63):   acc_lo[32]
│   ├─ WGMMA(A[0:64, 0:16],   B[0:16, 0:64])   → ki=0
│   └─ WGMMA(A[0:64, 16:32],  B[16:32, 0:64])   → ki=16
│
└─ m_block=1 (行 64..127): acc_hi[32]
    ├─ WGMMA(A[64:128, 0:16],  B[0:16, 0:64])   → ki=0
    └─ WGMMA(A[64:128, 16:32], B[16:32, 0:64])   → ki=16

每次 K-tile 迭代 = 4 条 WGMMA 指令
```

N=128 目前仍只用 N=64 的 WGMMA，如需 N=128 可以用 `wgmma.m64n128k16`（更进一步的优化）。

## Kernel 结构

```
Block: 128 threads = 1 warpgroup
Shared Memory: 3 × (128×32 + 32×128) × sizeof(half) = 3 × 16KB = 48KB + mbarriers

Phase[3] = {0, 0, 0}

┌─ Prologue: 填充前 min(3, num_k) 个 buffer
│   TMA tile[0] → buf[0]
│   TMA tile[1] → buf[1]
│   TMA tile[2] → buf[2]   (如果 num_k >= 3)
│
└─ Main Loop (k = 0 .. num_k-1):
    │
    ├─ 预取: if (k + 3 < num_k)
    │        TMA tile[k+3] → buf[(k+3) % 3]
    │
    ├─ 等待: mbar_wait(buf[k%3], phase[k%3])
    │        phase[k%3] ^= 1
    │
    ├─ wgmma_fence()
    │
    ├─ m_block=0: WGMMA × 2 → acc_lo
    ├─ m_block=1: WGMMA × 2 → acc_hi
    │
    ├─ wgmma_commit()
    └─ wgmma_wait()

Epilogue:
  acc_lo[32] → C[row_base + 0..63,  col]
  acc_hi[32] → C[row_base + 64..127, col]
```

## Shared Memory 布局

```
smem (total ≈ 48KB + padding):

buf[0]: |--- As_0 [128×32 half = 8KB] ---|--- Bs_0 [32×128 half = 8KB] ---|
buf[1]: |--- As_1 [128×32 half = 8KB] ---|--- Bs_1 [32×128 half = 8KB] ---|
buf[2]: |--- As_2 [128×32 half = 8KB] ---|--- Bs_2 [32×128 half = 8KB] ---|
mbar:   |--- mbar[0] ---|--- mbar[1] ---|--- mbar[2] ---|
```

## 编译信息

```
V5 kernel: 90 registers, 0 spill, 320 bytes stack frame
```

ptxas 会输出一个性能警告：
```
(C7514) Potential Performance Loss: wgmma.mma_async instructions are serialized
due to non wgmma instructions reading accumulator registers
```

这是因为在同一个 pipeline stage 中，epilogue 读取了 WGMMA 累加器。生产级 kernel 需要更精细地分离 WGMMA 和 epilogue 的 pipeline stage。

## 运行

```bash
cd build

# 正确性测试
./v5_production_kernel/test_v5

# 全版本性能对比：V1 vs V2 vs V3 vs V4 vs V5 vs cuBLAS
./v5_production_kernel/test_v5 --benchmark
```

## 与 DeepGEMM 的距离

V5 已经是一个完整的 TMA + WGMMA + multi-stage pipeline kernel。
要达到 DeepGEMM 的水平，还需要：

| 特性 | V5 (本教程) | DeepGEMM |
|------|-------------|----------|
| Pipeline stages | 3 | 7 |
| Swizzle | NONE | 128B swizzle |
| N 维度 WGMMA | m64n64k16 | m64n128k16 / m64n256k16 |
| Grouped GEMM | 不支持 | m_grouped_gemm |
| swapAB | 不支持 | 针对 MOE 优化 |
| FP8 支持 | 不支持 | 支持 FP8 + block scaling |

详见 [docs/compare_deepgemm.md](../docs/compare_deepgemm.md)。

## 下一步

→ 阅读 `docs/compare_deepgemm.md` 深入了解 DeepGEMM 的 swapAB 和 m_grouped_gemm 优化
