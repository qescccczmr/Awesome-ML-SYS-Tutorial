# 调试指南

## 常见错误速查表

| 症状 | 原因 | 解决方案 |
|------|------|----------|
| 输出全是0 | TMA descriptor未拷贝到device | 检查 `cudaMemcpyToSymbol` |
| 输出全是NaN | 累加器未初始化 | 检查寄存器初始化 |
| 随机错误值 | 缺少 `__syncthreads()` | shared mem 写入后加barrier |
| Kernel hang | mbarrier phase错误 | 检查 arrive/wait 逻辑 |
| 性能低20-30% | Bank conflict | Nsight Compute查bank_conflicts |
| 数据错乱（非零但错） | Swizzle模式不匹配 | TMA和WGMMA的swizzle必须一致 |
| 性能骤降50%+ | 寄存器溢出spill | 检查 `--ptxas-options=-v` 寄存器数 |

---

## V1-V2 调试

```bash
# 1. 先检查kernel启动错误
CUDA_LAUNCH_BLOCKING=1 ./test_v1

# 2. 检查寄存器使用量
nvcc --ptxas-options=-v v1_baseline_tiled_gemm/kernel.cu

# 3. 内存错误检测
compute-sanitizer --tool memcheck ./test_v1
```

**常见问题**：
- `__syncthreads()` 忘记加 → 数据竞争 → 随机错误
- Tile size太大 → shared memory超限 → 启动失败
- 边界条件处理错误 → 非64倍数矩阵出错

---

## V3 TMA 调试

```bash
# 验证TMA指令是否生成
nvcc -ptx -arch=sm_90 v3_tma_introduction/kernel.cu -o /tmp/v3.ptx
grep "cp.async.bulk.tensor" /tmp/v3.ptx
# 必须有输出，否则TMA未被使用
```

**常见问题**：
- TMA descriptor没有拷贝到constant/device memory → 输出全0
- `arrive_expect_tx` 字节数和实际传输不匹配 → kernel hang
- tile坐标计算错误 → 数据来自错误位置

---

## V4-V5 WGMMA 调试

```bash
# 验证WGMMA指令生成
cuobjdump --dump-sass ./test_v4 | grep -i wgmma
# 应看到: WGMMA.MMA.F32.F16.F16.M64N64K16

# 检查寄存器数量（超过128会spill）
nvcc --ptxas-options=-v v4_wgmma_introduction/kernel.cu 2>&1 | grep "registers"
```

**Swizzle mismatch（最常见的坑）**：
```
症状: 输出有值但与cuBLAS差距大
原因: TMA descriptor的swizzle != WGMMA descriptor的swizzle
检查:
  1. cuTensorMapEncodeTiled → CU_TENSOR_MAP_SWIZZLE_128B
  2. WGMMA desc bits[62:63] = 3 (128B swizzle)
  必须一致！
```

---

## Nsight Compute 关键指标

```bash
# 完整性能分析
ncu --set full -o profile ./test_v5
ncu-ui profile.ncu-rep  # GUI查看
```

| 指标 | 含义 | 目标 |
|------|------|------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM利用率 | V5 > 80% |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 内存带宽利用率 | V1-V2 ~100% |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | Tensor Core利用率 | V4-V5 > 70% |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | Bank conflicts | 应为0 |
| `lts__t_sectors_op_read.sum` | 寄存器spill读（local mem） | 应为0 |

---

## 常用命令参考

```bash
# 检查CUDA版本和GPU
nvcc --version
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv

# 生成PTX查看指令
nvcc -ptx -arch=sm_90 kernel.cu -o kernel.ptx
cat kernel.ptx | grep -A5 "wgmma\|cp.async.bulk"

# 生成SASS（机器码级别）
cuobjdump --dump-sass ./executable > sass.txt

# 检查shared memory和寄存器使用
nvcc --ptxas-options=-v kernel.cu 2>&1 | grep -E "registers|smem|bytes"

# 内存检查
compute-sanitizer ./test_v3

# 时钟频率检查（确保未降频）
nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv
```
