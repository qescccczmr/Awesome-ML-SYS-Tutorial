# H200 GEMM Tutorial — 从零到生产级矩阵乘法

从基础 Shared Memory Tiling 到 TMA + WGMMA + 多级流水线，5个递进式 CUDA kernel 带你掌握 H200 (sm_90a) 上高性能 GEMM 的核心技术。最终目标：读懂 DeepSeek 的 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)。

## 技术路线图

```
V1 (基础)          V2 (异步搬运)       V3 (硬件DMA)       V4 (Tensor Core)    V5 (生产级)
Shared Memory  →  cp.async        →  TMA             →  WGMMA           →  全部集成
Tiling            Double Buffer      mbarrier           Warpgroup          3-Stage Pipeline
                                     1条PTX搬运tile     m64n64k16          128×128 tile

CUDA Core         CUDA Core          CUDA Core          Tensor Core        Tensor Core
~手工FP32累加      ~手工FP32累加       ~手工FP32累加       ~2000 TFLOPS        ~2000 TFLOPS
```

| 版本 | 核心技术 | 新增概念 | 寄存器 | Shared Memory |
|------|----------|----------|--------|---------------|
| V1 | Tiled GEMM | 分块、寄存器fragment、bank conflict padding | 128 | 16 KB |
| V2 | Double Buffering | cp.async PTX、异步流水线、ping-pong | 128 | 33 KB |
| V3 | TMA | CUtensorMap、mbarrier、phase机制、硬件DMA | 100 | dynamic |
| V4 | WGMMA | Warpgroup(128线程)、Tensor Core descriptor | 58 | dynamic |
| V5 | 3-Stage Pipeline | 三缓冲、128×128 tile、双WGMMA块 | 90 | dynamic |

## 环境要求

- **GPU**: NVIDIA H200 / H100 / 任何 sm_90 架构 GPU
- **CUDA Toolkit**: >= 12.0（需要支持 `cuTensorMapEncodeTiled`）
- **CMake**: >= 3.18
- **cuBLAS**: 随 CUDA Toolkit 一起安装

## 构建

```bash
cd cuda-gemm-tutorial
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

构建产物：
```
build/
├── v1_baseline_tiled_gemm/test_v1
├── v2_double_buffering/test_v2
├── v3_tma_introduction/test_v3
├── v4_wgmma_introduction/test_v4
└── v5_production_kernel/test_v5
```

> 注意：V4/V5 自动编译为 `sm_90a`（WGMMA 指令需要 `a` 后缀启用架构特有功能）。

## 运行

### 正确性测试

每个版本自动用 cuBLAS 作为参考，验证多组矩阵尺寸：

```bash
# 运行单个版本
./v1_baseline_tiled_gemm/test_v1
./v2_double_buffering/test_v2
./v3_tma_introduction/test_v3
./v4_wgmma_introduction/test_v4
./v5_production_kernel/test_v5

# 一次性运行全部
for v in v1_baseline_tiled_gemm/test_v1 \
         v2_double_buffering/test_v2 \
         v3_tma_introduction/test_v3 \
         v4_wgmma_introduction/test_v4 \
         v5_production_kernel/test_v5; do
    echo "========================================="
    ./$v
done
```

输出示例：
```
=== V3: TMA Introduction GEMM ===
GPU: NVIDIA H200 (sm_90)

--- Correctness Tests ---
  [Square 128] M=128 N=128 K=128 | MaxErr=0.0012 | PASSED
  [Square 256] M=256 N=256 K=256 | MaxErr=0.0023 | PASSED
  ...
Correctness: ALL PASSED
```

### 性能 Benchmark

加 `--benchmark` 参数运行性能对比。**推荐直接用 V5**，它包含全部版本的对比：

```bash
# V5 全量对比：V1 vs V2 vs V3 vs V4 vs V5 vs cuBLAS
./v5_production_kernel/test_v5 --benchmark

# 或者逐步对比
./v2_double_buffering/test_v2 --benchmark     # V1 vs V2 vs cuBLAS
./v3_tma_introduction/test_v3 --benchmark     # V1 vs V2 vs V3 vs cuBLAS
./v4_wgmma_introduction/test_v4 --benchmark   # V1-V4 vs cuBLAS
```

输出示例：
```
--- Full Performance Comparison: V1-V5 vs cuBLAS ---
M=4096 N=4096 K=4096:
  [V1 Tiled]      | 12.345 ms |  11.14 TFLOPS
  [V2 DblBuf]     |  8.234 ms |  16.70 TFLOPS
  [V3 TMA]        |  6.123 ms |  22.46 TFLOPS
  [V4 WGMMA]      |  0.456 ms | 301.75 TFLOPS
  [V5 Production] |  0.234 ms | 587.69 TFLOPS
  [cuBLAS]        |  0.198 ms | 694.95 TFLOPS
  V5/V1: 52.76x | V5 vs cuBLAS: 84.6%
```

## Profiling

```bash
# Nsight Compute — 分析单个 kernel 的指标
ncu --set full ./v5_production_kernel/test_v5

# Nsight Systems — 分析整体时间线
nsys profile --stats=true ./v5_production_kernel/test_v5 --benchmark

# 查看 SASS 确认 WGMMA 指令生成
cuobjdump -sass ./v4_wgmma_introduction/test_v4 | grep -i wgmma
```

## 项目结构

```
cuda-gemm-tutorial/
├── CMakeLists.txt                 # 顶层构建（sm_90）
├── README.md                      # 本文件
├── common/                        # 共用工具库（header-only）
│   ├── cuda_utils.h               #   CUDA_CHECK / CUBLAS_CHECK 宏
│   ├── types.h                    #   fp16/bf16 类型别名 + 转换
│   ├── matrix_utils.h             #   矩阵初始化 / 误差比较
│   └── benchmark.h                #   GEMMBenchmark 计时框架
│
├── v1_baseline_tiled_gemm/        # V1: Shared Memory Tiling
│   ├── kernel.cu                  #   128×128 tile, 16×16 block, 8×8 per thread
│   ├── test.cu                    #   正确性 + benchmark
│   └── README.md                  #   原理讲解
│
├── v2_double_buffering/           # V2: cp.async + 双缓冲
│   ├── kernel.cu                  #   异步预取 + ping-pong buffer
│   ├── test.cu                    #   V1 vs V2 vs cuBLAS 对比
│   └── README.md
│
├── v3_tma_introduction/           # V3: TMA 硬件 DMA
│   ├── kernel.cu                  #   CUtensorMap + mbarrier + phase
│   ├── test.cu                    #   V1 vs V2 vs V3 vs cuBLAS 对比
│   └── README.md
│
├── v4_wgmma_introduction/         # V4: WGMMA Tensor Core
│   ├── kernel.cu                  #   wgmma.mma_async m64n64k16
│   ├── test.cu                    #   V1-V4 vs cuBLAS 对比
│   └── README.md
│
├── v5_production_kernel/          # V5: TMA + WGMMA + 3-Stage Pipeline
│   ├── kernel.cu                  #   128×128 tile, 3-buffer, 双WGMMA块
│   ├── test.cu                    #   V1-V5 vs cuBLAS 全量对比
│   └── README.md
│
└── docs/
    └── compare_deepgemm.md        # DeepGEMM 对比分析（swapAB, m_grouped_gemm）
```

## 推荐学习顺序

1. 阅读 `v1_baseline_tiled_gemm/README.md`，理解 V1 kernel，运行测试
2. 对比 V1 和 V2 的差异，理解 cp.async 和双缓冲
3. 学习 TMA 的 descriptor + mbarrier 模型（V3）
4. 理解 WGMMA 的 warpgroup 协作模型（V4）
5. 看 V5 如何把所有技术集成到 3-stage pipeline
6. 阅读 `docs/compare_deepgemm.md`，理解 DeepGEMM 的 swapAB 和 m_grouped_gemm

## 常见问题

**Q: 编译报错 `wgmma.fence not supported on sm_90`**
A: WGMMA 需要 `sm_90a`。V4/V5 的 CMakeLists.txt 已自动设置 `CUDA_ARCHITECTURES "90a"`。手动编译请用 `-arch=sm_90a`。

**Q: 运行报错 `CUDA driver version is insufficient`**
A: 需要在有 H200/H100 GPU 的机器上运行，驱动版本 >= 525。

**Q: 为什么 V4 的寄存器比 V1 少这么多 (58 vs 128)?**
A: WGMMA 用 descriptor 从 shared memory 读取数据，不需要寄存器 fragment。128 个线程协作一条指令，每个线程只需 32 个 FP32 累加器。

**Q: V3-V5 的 Shared Memory 为什么是 dynamic?**
A: 使用 `extern __shared__ char smem[]` + `cudaFuncSetAttribute` 动态设置，突破编译时 48KB 限制。

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUTLASS 3.x](https://github.com/NVIDIA/cutlass)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) — DeepSeek 的 MOE 优化 GEMM
