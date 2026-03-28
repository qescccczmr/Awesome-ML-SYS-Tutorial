# V1: Baseline Tiled GEMM

## 🎯 本版本目标

理解**Shared Memory Tiling**如何通过数据复用提升GEMM性能，建立后续优化的基线。

---

## 📚 背景知识

### 为什么朴素GEMM很慢？

```
朴素实现（triple-loop）：
for i in range(M):
    for j in range(N):
        for k in range(K):
            C[i][j] += A[i][k] * B[k][j]

问题：
- 计算量：2×M×N×K FLOPs
- 内存读取：A每行被读N次，B每列被读M次
- 带宽需求：(M×K×N + K×N×M) × 2字节 = 2×M×N×K×2 bytes
- 算术强度：(2MNK FLOPs) / (4MNK bytes) = 0.5 FLOPs/byte
- H200平衡点：~417 FLOPs/byte → 严重内存瓶颈！
```

### Tiling如何提升性能？

```
原理：让每个数据元素尽量在被丢弃前多用几次

以128×128的tile为例：
- 把A分成若干128×128的块（tile），载入shared memory
- 这个tile被复用 TILE_N=128 次（计算C的128列时都用它）
- 数据复用倍数 = TILE_SIZE
- 新算术强度 ≈ TILE_SIZE/4 = 32 FLOPs/byte（提升64倍！）
```

### Shared Memory Bank Conflict

```
Shared memory结构：32个bank，每个bank宽4字节

冲突情况：同一个warp的不同thread访问同一个bank（但不同地址）
  → 串行化执行 → 性能下降

避免方法：
  1. 确保访问步长不是32的倍数
  2. 添加padding（本代码中Bs的列数+8）
  3. 使用transpose布局

例如：Bs[TILE_K][TILE_N + 8]
  - 没有padding: 访问Bs[0][0], Bs[1][0], ...步长=128 half=256B
  - 256 / 4 = 64 → bank offset = 64 % 32 = 0 → 冲突！
  - 加8 padding: 步长=(128+8)×2=272B → 272/4=68 → 68%32=4 → 无冲突
```

---

## 🔧 实现详解

### 整体结构

```
每个CUDA Block处理 128×128 的输出tile C[by*128:(by+1)*128][bx*128:(bx+1)*128]

主循环（k_tile = 0 .. K/32）：
  ┌─ Stage 1: 协作加载 ─────────────────────────────────┐
  │  所有256个thread一起加载 As[128×32] 和 Bs[32×128]   │
  │  到shared memory（合并访问，高效）                   │
  └─────────────────────────────────────────────────────┘
  ↓ __syncthreads()
  ┌─ Stage 2: 计算 ──────────────────────────────────────┐
  │  每个thread计算自己的 8×8 输出片段                   │
  │  内循环：k=0..31，outer product累加到acc[8][8]       │
  └─────────────────────────────────────────────────────┘
  ↓ __syncthreads()

最后：将acc[8][8]写回global memory C
```

### 关键代码段落

#### 1. Shared Memory声明
```cpp
__shared__ half As[TILE_M][TILE_K];       // 128×32 = 8KB
__shared__ half Bs[TILE_K][TILE_N + 8];  // 32×136 = ~8.5KB（+8 padding避免bank conflict）
// 总计约16.5KB，远低于228KB限制
```

#### 2. 协作加载（线性化分配）
```cpp
int tid_linear = ty * BLOCK_DIM + tx;  // 0..255的线性ID
// 将128×32=4096个元素分给256个thread，每个thread加载16个
for (int elem = tid_linear; elem < TILE_M * TILE_K; elem += BLOCK_DIM * BLOCK_DIM) {
    int as_row = elem / TILE_K;
    int as_col = elem % TILE_K;
    As[as_row][as_col] = A[global_row * K + global_col];  // 简化版
}
```
**为什么用线性化**：确保连续的thread访问连续的内存地址（coalesced access），最大化内存带宽利用率。

#### 3. 累加计算
```cpp
for (int k = 0; k < TILE_K; k++) {
    float a_frag[TM];  // 预加载到寄存器，避免重复访问shared memory
    float b_frag[TN];
    for (int i = 0; i < TM; i++) a_frag[i] = __half2float(As[ty*TM+i][k]);
    for (int j = 0; j < TN; j++) b_frag[j] = __half2float(Bs[k][tx*TN+j]);
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            acc[i][j] += a_frag[i] * b_frag[j];  // FMA指令
}
```
**为什么预加载到寄存器**：寄存器访问比shared memory快得多（0 cycle vs 23 cycle latency）。

---

## 📊 性能分析

### 预期结果（H200）

| 矩阵大小 | 预期TFLOPS | 峰值% | 瓶颈 |
|---------|-----------|-------|------|
| 512×512 | 100-200 | 5-10% | 内存带宽 |
| 1024×1024 | 200-300 | 10-15% | 内存带宽 |
| 2048×2048 | 250-350 | 12-17% | 内存带宽 |
| 4096×4096 | 280-380 | 14-19% | 计算（tiling开始有效） |

### 性能瓶颈分析

1. **还是内存瓶颈**：即使有tiling，V1仍然内存瓶颈
   - 原因：TILE_K=32太小，每个tile的复用不够
   - 解决方案→V2：增加pipeline深度，不等待内存

2. **没有pipeline**：Stage1（加载）和Stage2（计算）串行
   - 加载时SM空转，计算时内存空闲
   - 解决方案→V2：Double buffering让两者重叠

3. **手工float累加**：没有使用Tensor Core
   - 计算效率约为Tensor Core的1/16
   - 解决方案→V4：使用WGMMA指令

### 用Nsight Compute分析

```bash
ncu --set full -o v1_profile ./test_v1 --benchmark

# 关注指标：
# dram__throughput ~ 100% → 内存瓶颈确认
# sm__pipe_tensor_op_hmma_cycles_active ~ 0% → 未使用Tensor Core
```

---

## 🚀 下一步 (V2)

V2将引入**Double Buffering（双缓冲）**：
- 使用2个shared memory buffer（ping-pong）
- 在计算tile[i]时，预取tile[i+1]（重叠IO和计算）
- 使用`cp.async`异步拷贝指令
- 预期性能提升：1.5-2×

---

## 🔨 运行方法

```bash
# 在项目根目录
mkdir -p build && cd build
cmake ..
make test_v1 -j

# 正确性测试
./v1_baseline_tiled_gemm/test_v1

# 性能测试
./v1_baseline_tiled_gemm/test_v1 --benchmark
```
