# V2: Software Double Buffering

## 🎯 本版本目标

理解**Software Pipelining（软件流水线）**如何通过隐藏内存延迟提升性能。
掌握 `cp.async` 异步拷贝指令和双缓冲的经典模式。

---

## 📚 背景知识

### V1的性能瓶颈回顾

```
V1的执行时间线（串行）：

时间 →
[Load tile0 to smem]─────[Compute tile0]─[Load tile1]─────[Compute tile1]─...
│                    │    │              │ │            │   │              │
│  ~400-600 cycles  │    │  ~200 cycles │ │ 400-600c   │   │  200 cycles  │

问题：Load和Compute是串行的，Load期间SM完全空转！
```

### Double Buffering如何解决

```
V2的执行时间线（流水线）：

时间 →
[Prefetch tile0]─────────────────────────────────────────────────
                 [Prefetch tile1]─────────          ← 与Compute重叠！
                                  [Wait tile0]
                                              [Compute tile0]
                                  [Prefetch tile2]────
                                                       [Wait tile1]
                                                                   [Compute tile1]
                                                       ...

关键：在Compute tile[k]的同时，异步Prefetch tile[k+1]
→ 内存延迟被完全隐藏（只要Compute时间 ≥ Load时间）
```

### cp.async vs 普通内存拷贝

```
普通拷贝（V1方式）：
  LDG.E.128 r, [global_addr]  ← 加载到寄存器（同步，阻塞等待）
  STS.128 [shared_addr], r    ← 从寄存器写到shared memory
  → Thread被阻塞，直到数据到达寄存器

cp.async（V2方式）：
  cp.async.ca.shared.global [shared_addr], [global_addr], 16
  → 数据直接从L2/HBM异步传输到shared memory，绕过寄存器
  → Thread立即继续执行（不阻塞！）
  → 优点：节省寄存器，隐藏延迟，更高效
```

---

## 🔧 实现详解

### 关键数据结构

```cpp
// 双缓冲：[0]和[1]交替使用
__shared__ half As[2][TILE_M][TILE_K];    // 2 × 8KB = 16KB
__shared__ half Bs[2][TILE_K][TILE_N+8]; // 2 × 8.5KB = 17KB
// 总计 ≈ 33KB（V1的2倍，仍在228KB限制内）

// Buffer切换逻辑（ping-pong）：
int cur_buf  = k & 1;        // 偶数迭代用buf0，奇数用buf1
int next_buf = 1 - cur_buf;  // 下一个tile写入另一个buf
```

### 流水线同步机制

```cpp
// Prologue：预取tile[0]，打成group[0]
async_load(0, buf0);
CP_ASYNC_COMMIT_GROUP();  // group[0]开始传输

// 主循环 k=0..N-1
for (int k = 0; k < num_tiles; k++) {
    // Step 1: 预取tile[k+1]（发出异步请求）
    if (k+1 < num_tiles) {
        async_load(k+1, next_buf);
        CP_ASYNC_COMMIT_GROUP();  // group[k+1]开始传输
        // 现在in-flight: group[k]（tile[k]）和group[k+1]（tile[k+1]）

        CP_ASYNC_WAIT(1);   // 等到in-flight数量 ≤ 1
                            // → group[k]（tile[k]）完成
                            // → group[k+1]（tile[k+1]）仍在传输（与计算重叠！）
    } else {
        CP_ASYNC_WAIT(0);   // 最后一个tile：等所有group完成
    }

    __syncthreads();  // 确保shared memory对所有thread可见

    // Step 2: 计算tile[k]（此时tile[k+1]正在后台传输）
    compute(cur_buf);

    __syncthreads();  // 计算完毕，允许下轮覆盖cur_buf
}
```

### cp.async实现细节

```cpp
// PTX内联汇编
// dst: shared memory地址（必须16字节对齐）
// src: global memory地址
// 16: 拷贝16字节（= 8个half）
asm volatile(
    "cp.async.ca.shared.global [%0], [%1], %2;"
    :: "r"(smem_addr_as_uint32),   // dst - 用__cvta_generic_to_shared转换
       "l"(global_ptr),             // src - 64位指针
       "n"(16)                      // 字节数（必须是编译时常量！）
    : "memory"
);

// 对齐要求：
// - 16字节（= 8 half）: 最高效，使用128位内存事务
// - 8字节（= 4 half）:  次级效率
// - 4字节（= 2 half）:  最低效率
// 本代码使用16字节，确保TILE_K和TILE_N是8的倍数
```

---

## 📊 性能分析

### 预期结果（H200）

| 矩阵大小 | V1 TFLOPS | V2 TFLOPS | 提升 |
|---------|-----------|-----------|------|
| 512×512 | 100-200 | 200-350 | 1.5-2x |
| 1024×1024 | 200-300 | 350-500 | 1.5-2x |
| 2048×2048 | 250-350 | 400-600 | 1.5-2x |

### 为什么还达不到峰值？

1. **计算效率低**：手工FP32累加，未使用Tensor Core
   - Tensor Core比CUDA core快16-32倍
   - 解决方案→V4: WGMMA

2. **TILE_K偏小（32）**：数据复用倍数不够
   - 理想的TILE_K应该更大（≥64），但受限于shared memory
   - 解决方案→V3/V4: TMA+WGMMA允许更大的tile

3. **没有硬件加速搬运**：cp.async仍占用SM资源计算地址
   - 解决方案→V3: TMA完全offload内存搬运

### Nsight Compute分析

```bash
ncu --set full -o v2_profile ./test_v2 --benchmark

# 对比V1，应该看到：
# sm__warps_active.avg.pct_of_peak_sustained_active 提升
#   → 更少的内存等待stall
# dram__throughput 降低
#   → 不再100%内存瓶颈（被compute掩盖了一些）
```

---

## 🚀 下一步 (V3 - TMA)

V3将用**TMA（Tensor Memory Accelerator）**替换cp.async：
- 专用DMA引擎，完全不占用SM资源
- 单条PTX指令拷贝整个tile（无循环）
- 使用`mbarrier`替代`cp.async.wait_group`
- 顺带引入**Grouped GEMM**（MOE的基础）
- 预期性能提升：再提升1.5-2×

---

## 🔨 运行方法

```bash
# V2需要sm_80+（cp.async在Ampere引入）
# H200是sm_90，完全支持

mkdir -p build && cd build
cmake ..
make test_v2 -j

# 正确性测试
./v2_double_buffering/test_v2

# 性能对比（V1 vs V2 vs cuBLAS）
./v2_double_buffering/test_v2 --benchmark
```
