# H200 Architecture Overview

## Key Specifications

| Feature | H200 | H100 (for comparison) |
|---------|------|----------------------|
| Architecture | Hopper (sm_90) | Hopper (sm_90) |
| SM Count | 132 | 132 (SXM) |
| FP16 Tensor Core Peak | ~2000 TFLOPS | ~2000 TFLOPS |
| Memory | HBM3e, 141 GB | HBM3, 80 GB |
| Memory Bandwidth | 4.8 TB/s | 3.35 TB/s |
| L2 Cache | 50 MB | 50 MB |
| Shared Memory / SM | 228 KB | 228 KB |

**H200 vs H100**: 相同的sm_90计算架构，H200的区别是更快更大的HBM3e内存。所有CUDA代码完全兼容。

## Hopper 新特性

### 1. TMA (Tensor Memory Accelerator)
- 专用DMA引擎，异步搬运global↔shared memory
- 支持多维张量stride，自动处理swizzle
- 使用`mbarrier`异步屏障同步
- PTX: `cp.async.bulk.tensor.2d`

### 2. WGMMA (Warp Group Matrix Multiply Accumulate)
- 第四代Tensor Core，128 threads (4 warps) 协同工作
- 单指令计算64×64×16矩阵乘
- 支持FP16/BF16/FP8输入，FP32累加
- PTX: `wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16`

### 3. mbarrier (Asynchronous Barrier)
- 基于phase的同步，避免计数器溢出
- 支持TMA的transaction counting (`arrive_expect_tx`)
- 允许构建3-7+ stage深度流水线

### 4. Shared Memory Swizzle
- 目的：消除shared memory bank conflict
- 类型：32B/64B/128B swizzle
- **关键**：TMA和WGMMA的swizzle必须一致，否则数据错乱！

## Memory Hierarchy

```
HBM3e (4.8 TB/s, 141GB)
    ↓ TMA (async DMA, 不占SM)
 L2 Cache (50 MB)
    ↓
 Shared Memory (228 KB/SM, 32 banks)
    ↓ WGMMA
 Registers (per thread, 32个FP32用于WGMMA累加器)
```

## 性能 Roofline

- FP16峰值算力: ~2000 TFLOPS
- 内存带宽: 4.8 TB/s
- 平衡点: ~417 FLOPs/byte
- 大矩阵(M,N,K>2048): compute-bound → TMA+WGMMA是关键
- 小矩阵(M,N,K<512): memory-bound → 带宽利用率是关键
