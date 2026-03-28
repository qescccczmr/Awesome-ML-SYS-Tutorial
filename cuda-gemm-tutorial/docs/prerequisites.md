# 前置知识和学习资源

## V1-V2 所需知识

### CUDA 基础
- Thread/Block/Grid 层次结构
- Kernel 启动语法: `kernel<<<grid, block>>>(args)`
- Shared memory: `__shared__`, 声明和使用
- 同步: `__syncthreads()`
- 内存类型: global, shared, registers, local

### 矩阵乘法优化
- Naive triple-loop 实现
- Tiling 原理：为什么 shared memory 能提升性能
- Coalesced global memory access（连续thread访问连续地址）
- Bank conflict（32个bank，4字节宽）

**学习资源**:
- [CUDA C++ Programming Guide - Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Optimizing Matrix Multiply (Volkov & Demmel)](https://people.eecs.berkeley.edu/~volkov/volkov08-GTC.pdf)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

## V3-V5 所需知识

### PTX Assembly
- 基本语法和数据类型
- CUDA 内联汇编: `asm volatile("ptx_instruction" : outputs : inputs)`
- 指令延迟和流水线

### GPU 架构
- SM 结构和 warp 调度
- Tensor Core 演进（Volta→Ampere→Hopper）
- Occupancy 计算

**学习资源**:
- [Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [PTX ISA Reference 8.0+](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- GTC 讲座：搜索 "Hopper GEMM", "TMA", "WGMMA"

## 推荐学习路径

```
Week 0: 基础回顾
  - 手写 naive GEMM（不看参考）
  - 手写 tiled GEMM（不看参考）
  - 阅读 Volkov & Demmel 论文

Week 1-2: V1+V2
  - 学习本教程 V1 代码
  - 用 Nsight Compute 分析性能瓶颈
  - 理解 double buffering 流水线原理

Week 3: V3 (TMA)
  - 阅读 Hopper whitepaper TMA 章节
  - 理解 cuTensorMapEncodeTiled API
  - 调试 mbarrier 同步问题（预期会遇到）

Week 4: V4 (WGMMA)
  - 阅读 PTX ISA wgmma 章节
  - 理解 warpgroup（128 threads）概念
  - 调试 swizzle mismatch 问题（预期会遇到）

Week 5: V5 + deepgemm 对比
  - 实现 3-stage 流水线
  - 阅读 deepgemm 源码对比
  - （可选）实现 swapAB 优化
```

## 自测清单

开始 V1 前，应该能：
- [ ] 写出 vector add CUDA kernel
- [ ] 解释为什么 `__syncthreads()` 在 shared memory 写入后必须
- [ ] 计算给定访问模式的 bank conflict 次数
- [ ] 用 Nsight Compute 分析简单 kernel

开始 V3 前，应该额外能：
- [ ] 读懂基本的 PTX 汇编
- [ ] 解释 warp 的执行模型
- [ ] 计算理论带宽利用率
