# DeepEP 通信系统学习计划

## 驱动问题

**DeepEP 如何在 MoE EP 场景下，将跨节点 All-to-All 通信的 SM 占用从 50%+ 压缩到个位数百分比，同时实现通信与计算的近乎完全重叠？**

这个问题在读者掌握以下三层背景后才能完整理解：
1. EP 的 Dispatch/Combine 通信模式是什么（通信语义层）
2. NCCL 实现 All-to-All 的代价是什么（对比基线层）
3. GPU SM 资源是如何被通信 kernel 竞争的（硬件资源层）

驱动问题出现在第三步「DeepEP 全貌」中——在读者已经理解了 EP 通信模式和 NCCL 代价之后，自然引出：**有没有办法用极少的 SM 完成同样的通信，并把剩余 SM 留给 GEMM？**

---

## 动机定位

这个 topic 填补了当前知识图谱中「EP 通信基础设施」这一空白。已有文章[深入浅出 DeepSeek MoE，EP 与 FSDP 经典二次开发](../../rlhf/sys-design/readme-4.md)建立了 EP 的概念框架（什么是 Expert Parallelism、Dispatch/Combine 两次 All-to-All、EP vs TP 的通信量对比），[RDMA 权重传输设计](../../rlhf/sys-design/readme-5.md)建立了 RDMA 通信栈的底层认知（QP、内存注册、GPU-Direct RDMA、NCCL 与原生 RDMA 的关系）。

DeepEP 是连接这两篇文章的那座桥：它把 EP 通信语义（上层）和 RDMA 原语（底层）焊接在一起，并在此之上引入了 low-SM occupation 这一核心创新。这个 topic 的深度定位是**理解复现级（understand-reproduce）**——它是 SGLang/verl 等框架依赖的基础设施。

---

## 前置知识检查

学习本 topic 之前，建议回顾以下内容：

- [深入浅出 DeepSeek MoE，EP 与 FSDP](../../rlhf/sys-design/readme-4.md)：EP 的核心动机、Dispatch/Combine 通信模式、EP vs TP 通信量对比表。**本文默认读者已掌握这些概念，不再重复推导。**
- [RDMA 权重传输设计](../../rlhf/sys-design/readme-5.md)：RDMA 的 QP 模型、内存注册、GPU-Direct RDMA 数据路径、NCCL 如何在底层使用 RDMA。**DeepEP 的 RDMA 直驱能力依赖这些底层认知。**
- [NCCL 与 NVIDIA TOPO](../../torch/nccl/readme.md)：NCCL Ring/Tree 算法、三种通信协议（Simple/LL/LL128）。**理解 NCCL 的 SM 开销需要先知道 NCCL kernel 的工作方式。**

---

## 学习路线图

> 顺序硬约束：概念框架 → 具体模型/场景 → 工程代码。绝对不能反过来。

### 第零步：GPU 进程间通信基础——从 NVLink 到 RDMA

- **深度层级**：理解复现
- **从何推导**：起点。在进入 DeepEP 具体设计之前，先建立 GPU 通信基础设施的概念框架——节点内/节点间的三种通信路径、各自的带宽/延迟特征、RDMA 编程模型的核心约束。
- **目标**：理解 GPU 间数据传输的三种路径（NVLink、PCIe、RDMA），掌握 RDMA/GPU-Direct 的工作机制，理解 NCCL 的通信抽象及其局限
- **方法**：概念框架
- **需要独立展开的概念**：
  - NVLink vs PCIe：节点内高速互联的两种方式，带宽差异（450 GB/s vs 32 GB/s）
  - RDMA 与 GPU-Direct：内核旁路、零拷贝、NIC 直接读写 GPU 显存的数据路径
  - RDMA 编程模型：MR 注册、QP、WR、CQ 的核心概念及开销
  - NCCL 的价值与代价：拓扑感知 + 协议抽象的收益，灵活性受限 + SM 占用高的代价
- **参考资源**：
  - [RDMA 权重传输设计](../../rlhf/sys-design/readme-5.md)（详细的 RDMA 编程模型）
  - RDMA Aware Networks Programming User Manual（Mellanox 官方手册）
  - GPU-Direct RDMA 官方文档

### 第一步：EP 通信的硬件资源视角——SM 是被谁占掉的

- **深度层级**：理解复现
- **从何推导**：第零步建立了通信基础设施的认知。这一步聚焦 GPU 端的资源消耗——前置文章告诉我们 EP 需要两次 All-to-All，这里从「这两次 All-to-All 在 GPU 上实际消耗了什么 SM 资源」切入。
- **目标**：建立 SM 资源竞争的直觉——NCCL All-to-All kernel 在 H800 上占用多少 SM？与 GEMM kernel 并发时会发生什么？
- **方法**：概念框架 + 数据支撑
- **需要独立展开的概念**：
  - SM 占用率的含义：一个 CUDA kernel 启动时申请多少 SM，占用期间其他 kernel 是否能并发
  - NCCL 的 persistent kernel 模式：为什么 NCCL 会「霸占」SM 而不释放
  - GEMM 对 SM 的需求：A100/H800 上 GEMM 为何需要尽量多的 SM 才能跑满
- **参考资源**：NCCL 源码中 `ncclDevKernel`、NVIDIA NSight 分析 NCCL 占用的博客

### 第二步：All-to-All 的语义分层——Dispatch 与 Combine 不对称在哪

- **深度层级**：理解复现
- **从何推导**：第一步确立了「SM 被 NCCL 占用」是问题所在。这一步深入通信语义本身，为后续 DeepEP 的设计决策铺路——Dispatch 和 Combine 在数据形状、时序依赖、与计算的关系上各有不同，这些不对称性直接决定了 DeepEP 的 API 设计。
- **目标**：精确理解 EP forward 的数据流：`[tokens, hidden] → dispatch → expert GEMM → combine → [tokens, hidden]`，每个阶段的 tensor shape 变化、数据分布方式
- **方法**：概念框架（数据流图）
- **需要独立展开的概念**：
  - Token routing 的不均匀性（load imbalance）：为什么接收到的 token 数量每个 rank 不一样
  - Dispatch 与 GEMM 的流水依赖：必须等所有 token 到齐才能开始 GEMM 吗？
  - Combine 与 Dispatch 的对偶关系：为什么 combine 可以在 dispatch 结束之前就开始
- **参考资源**：DeepSeek-V3 技术报告 EP 章节、DeepEP README 的 API 说明

### 第三步：DeepEP 全貌——定位、架构与两种模式

- **深度层级**：理解复现
- **从何推导**：前两步确立了问题（SM 竞争）和语义（Dispatch/Combine 不对称）。这一步给出 DeepEP 的整体答案，自然提出**驱动问题**：DeepEP 如何用极少的 SM 完成 All-to-All，同时把剩余 SM 留给 GEMM？
- **目标**：交代 DeepEP 的来源（DeepSeek 开源，配合 DeepSeek-V3/R1 的 EP 推理训练）、支持的硬件（NVLink intra-node + InfiniBand/RDMA inter-node）、两种模式的定位
- **方法**：概念框架
- **需要独立展开的概念**：
  - Normal mode 的定位：高吞吐 prefill/训练场景，允许较高 SM 占用换取带宽
  - Low SM occupation mode 的定位：decode 场景，通信量小但对延迟敏感，需要把 SM 让给 attention 和 GEMM
  - Buffer 对象的设计哲学：为什么用固定 Buffer 而非临时分配
- **参考资源**：[DeepEP GitHub](https://github.com/deepseek-ai/DeepEP)

### 第四步：Normal Mode 深度解析——高吞吐 All-to-All 的实现

- **深度层级**：理解复现（原理级）
- **从何推导**：第三步给出了两种模式的定位。Normal mode 先讲，因为它是对 NCCL All-to-All 的直接替代，理解它才能对比出 low SM mode 的创新之处。
- **目标**：理解 normal mode dispatch/combine 的 kernel 实现思路：intra-node 用 NVLink P2P，inter-node 用 RDMA write；chunk 化传输；数据布局
- **方法**：概念框架 + 关键代码路径
- **需要独立展开的概念**：
  - **为什么不用 NCCL AlltoAll**：NCCL 的通用性（拓扑感知、协议抽象）在 EP 场景下变成开销（数据量不对称需要额外元数据交换、SM 占用高）。生成 NCCL vs DeepEP 对比表（7 个维度：数据路径、WR 发起方、路由元数据、完成检测、SM 占用、适用场景、协议开销）
  - NVLink P2P write：GPU-A 的 kernel 直接写 GPU-B 显存（通过 CUDA IPC 交换的虚拟地址），绕过 NCCL 协议栈
  - RDMA write 直驱：GPU kernel 通过 doorbell 寄存器发起 `ibv_post_send`，NIC 自主从 GPU 显存 DMA 读取并发送
  - **RDMA 编程模型的三个关键约束**（新增子章节）：
    - 约束 1：MR 注册必须提前完成，开销巨大（10～50ms）→ 预分配 Buffer 设计
    - 约束 2：WR 提交可在 CPU 或 GPU 完成 → Normal Mode（GPU 提交）vs Low SM Mode（CPU 提交）
    - 约束 3：完成检测的 Polling vs Event-driven → 为什么选 Polling
  - Chunk pipeline：如何把大 tensor 切成 chunk 流水传输
  - SM 占用数量：normal mode 下仍需较多 SM（8～16），与 NCCL（16～32）的对比
- **需要替代方案对比的设计决策**：
  - NCCL AlltoAll vs DeepEP 直驱（为什么绕过 NCCL）
  - GPU 提交 WR vs CPU 提交 WR（为什么 Normal Mode 选 GPU 提交）
- **参考资源**：DeepEP `csrc/kernels/internode.cu`、`intranode.cu`

### 第五步：Low SM Occupation Mode——持久化 Warp 组的通信内核

- **深度层级**：理解复现（原理级，本文核心章节）
- **从何推导**：第四步的 normal mode 仍然占用较多 SM。从「GEMM 需要几乎全部 SM」这一约束推导：能否把通信逻辑压缩到少数几个 warp，让 GEMM kernel 和通信 kernel 真正并发在同一个 GPU 上？
- **目标**：理解 low SM mode 的 persistent warp 设计：少量 warp group 常驻，轮询 RDMA completion queue，负责搬运数据，不干扰主计算流
- **方法**：概念框架 + 关键代码路径
- **需要独立展开的概念**：
  - Persistent kernel 模式：kernel 不退出，持续轮询工作队列
  - Warp-level polling：用 `ld.acquire` / `st.release` 实现无锁状态检测
  - SM 占用数量：low SM mode 只用 1-2 个 SM（具体数字看 DeepEP 配置），其余 SM 全部留给 GEMM
  - NIC-initiated DMA vs CPU-initiated：为什么 low SM mode 下通信发起方是 NIC 而不是 SM
  - Hook 机制：DeepEP 的 `get_dispatch_layout` / `dispatch` / `combine` API 如何通过 hook 把 low SM kernel 嵌入 CUDA stream
- **baseline**：用多个 SM 的 persistent kernel 做通信（normal mode）
- **中间方案**：减少 SM 数量但仍用标准 CUDA block——block 调度粒度太粗，最少也得占 1 个 SM per CTA
- **最终方案**：persistent warp group，以 warp 为粒度固定绑定到少数 SM，其余 SM 完全空出给计算 kernel
- **需要替代方案对比的设计决策**：为什么用 persistent warp 而不是 CUDA cooperative groups 或 NCCL proxy thread
- **参考资源**：DeepEP `csrc/kernels/low_latency_dispatch.cu`，NVIDIA CUDA persistent thread 编程模型文档

### 第六步：通信计算 Overlap——流水线的三种形态

- **深度层级**：理解复现
- **从何推导**：第五步确立了 low SM mode 能「空出 SM」。这一步回答：空出来的 SM 如何与通信真正并发？三种 overlap 形态对应 EP forward 的三个阶段。
- **目标**：理解 DeepEP 的 overlap 机制，能画出 dispatch-GEMM-combine 的时序图，区分三种形态的适用场景
- **方法**：概念框架（时序图）+ 关键代码路径
- **统一章节：三种 overlap 形态（合并为一步）**
  - **形态 1：Dispatch overlap with Attention**（decode 场景）：dispatch 通信在 attention kernel 执行期间完成，attention 结束即可立即开始 expert GEMM
  - **形态 2：Expert GEMM overlap with inter-node transfer**（prefill/训练）：第一批 token 到达就启动 GEMM，不等所有 token——Stream-K 风格的 token-level pipeline
  - **形态 3：Combine overlap with next-layer attention**：combine 回传与下一层的 attention 并发，进一步隐藏 combine 延迟
- **需要独立展开的概念**：
  - CUDA multi-stream 并发的条件：两个 kernel 在不同 stream 上、SM 资源不互相抢占
  - Event / `cudaStreamWaitEvent` 的同步语义：如何在 overlap 中保证数据依赖正确
  - `get_next_event` hook：DeepEP API 暴露的 event 接口如何让调用方控制同步点
- **参考资源**：DeepEP README「Overlapping」章节，DeepSeek-V3 技术报告图 4

### 第七步：Buffer 设计与内存管理——为什么固定预分配

- **深度层级**：理解复现
- **从何推导**：前几步从通信语义和 kernel 角度分析 DeepEP。这一步从内存管理角度收尾：DeepEP 的 `Buffer` 对象为什么要预分配固定大小、为什么要 pin memory、RDMA 内存注册的代价如何被摊销。
- **目标**：理解 `Buffer` 的生命周期管理，能解释为什么不能「按需分配通信 buffer」
- **方法**：概念框架
- **需要独立展开的概念**：
  - RDMA 内存注册（MR registration）：必须提前完成、注册后才能用于 RDMA write
  - Pinned memory：为什么通信 buffer 必须锁页（`cudaMallocHost` 或 `cudaMalloc` + `ibv_reg_mr`）
  - Buffer reuse 与 in-flight 计数：如何保证上一次通信完成前不覆盖 buffer
- **参考资源**：DeepEP `deep_ep/buffer.py`，RDMA verbs 文档

### 第八步：代码导读——从 Python API 到 CUDA kernel 的调用链

- **深度层级**：理解复现（代码级）
- **从何推导**：第三到七步建立了足够的概念地图。这一步做代码导读，沿着 `buffer.dispatch()` → C++ binding → CUDA kernel 的调用链走一遍，把前面所有概念在代码中找到对应位置。
- **目标**：能独立阅读 DeepEP 源码，定位 normal mode / low SM mode 的分支点，理解 hook 注册流程
- **方法**：代码分析
- **关键文件路径**（需在阅读时补充 commit hash）：
  - `deep_ep/buffer.py`：Python 层 Buffer API
  - `csrc/deep_ep.cpp`：pybind11 binding
  - `csrc/kernels/internode.cu`：inter-node RDMA dispatch/combine（normal mode）
  - `csrc/kernels/intranode.cu`：intra-node NVLink dispatch/combine
  - `csrc/kernels/low_latency_dispatch.cu`：low SM occupation 核心 kernel
- **参考资源**：DeepEP GitHub repo

---

## 推荐资源

### 官方文档
- DeepEP README：`https://github.com/deepseek-ai/DeepEP`
- DeepSeek-V3 技术报告：`https://arxiv.org/abs/2412.19437`（EP 通信章节）
- NVIDIA CUDA C Programming Guide：Persistent Thread 章节
- **RDMA Aware Networks Programming User Manual**：`https://www.mellanox.com/related-docs/prod_software/RDMA_Aware_Programming_user_manual.pdf`（Mellanox 官方 RDMA 编程手册）
- **GPU-Direct RDMA 官方文档**：`https://docs.nvidia.com/cuda/gpudirect-rdma/`（NVIDIA 对 GDR 的技术说明）

### 代码仓库
- DeepEP：`https://github.com/deepseek-ai/DeepEP`（`csrc/kernels/` 是核心）
- NCCL 源码：`https://github.com/NVIDIA/nccl`（理解 persistent kernel 和 channel 机制）
- SGLang 中的 EP 集成：搜索 `deepep` 关键词

### 社区文章
- [深入浅出 DeepSeek MoE，EP 与 FSDP](../../rlhf/sys-design/readme-4.md)（本 repo 前置文章 — EP 概念、通信量分析）
- [RDMA 权重传输设计](../../rlhf/sys-design/readme-5.md)（本 repo 前置文章 — RDMA 编程模型、GPU-Direct、MR 注册）
- [NCCL 与 NVIDIA TOPO](../nccl/readme.md)（本 repo 前置文章 — NCCL 工作方式、拓扑检测）

### 拓展阅读与对比分析
- **vLLM 的 EP 实现**：vLLM 使用 NCCL All-to-All 而非 DeepEP，对比两者在 SM 占用和延迟上的差距
- **Megascale-Infer**：字节跳动对 MoE 推理的 EP 通信优化，另一个思路的参考
- **Tutel**：微软的 MoE dispatch 实现，早于 DeepEP 的开源方案，可作为演进对比
- **NCCL 源码中的 `ncclDevKernel`**：理解 NCCL persistent kernel 占用多少 SM 的第一手来源
- **Understanding Modern GPU Architectures**：`https://developer.nvidia.com/blog/cuda-refresher-reviewing-the-origins-of-gpu-computing/`（NVIDIA 开发者博客 — SM 资源管理和 warp 调度）

---

## 文章结构建议

- **文章类型**：sys-design + code-walkthrough 混合
- **建议路径**：`torch/deepep/readme.md`
- **系列归属**：可作为「MoE 系统基础设施」系列第一篇，与 `rlhf/sys-design/readme-4.md`（EP 概念）和 `rlhf/sys-design/readme-5.md`（RDMA）并列
- **预计章节**：
  0. **GPU 进程间通信基础**（新增）：NVLink vs PCIe vs RDMA、GPU-Direct 数据路径、RDMA 编程模型（MR/QP/WR/CQ）、NCCL 的通信抽象及其局限
  1. EP 通信在 GPU 上消耗了什么资源：NCCL persistent kernel 的 SM 占用、GEMM 对 SM 的线性敏感性
  2. Dispatch 与 Combine 的不对称性：scatter vs gather、细粒度流水线机会、接收方等待模式
  3. DeepEP 全貌：两种模式定位（Normal vs Low SM）、Buffer 对象设计哲学、驱动问题的提出
  4. Normal Mode：为什么不用 NCCL（对比表）、NVLink P2P write、GPU-Direct RDMA write、RDMA 编程模型的三个关键约束
  5. Low SM Occupation Mode：设计演进（baseline → 中间 → 最终）、NIC 主导的四步数据传输、两级 Buffer、内存序（ld.acquire）、方案对比表
  6. 通信计算 Overlap：三种形态（Dispatch overlap Attention、GEMM overlap Combine、层间 Pipeline）、Overlap 的前提（Low SM Mode 必要性）
  7. Buffer 设计：预分配（规避 MR 注册开销）、Handle 机制、get_next_event（overlap 粘合剂）
  8. 代码导读：Python → C++ → CUDA kernel 调用链、真实代码片段（commit 567632d）
  9. 总结：三层解耦（发起通信、数据传输、完成检测）、DeepEP 对 MoE 推理的系统性价值

---

## 自检

- [x] 每个步骤标注了「从何推导」，推导链完整（第零步起点 → 第一步 → ... → 第八步）
- [x] 步骤顺序严格遵循「概念 → 模型/场景 → 代码」
- [x] **新增第零步**：GPU 进程间通信基础（NVLink/PCIe/RDMA、GPU-Direct、RDMA 编程模型、NCCL 抽象）
- [x] 驱动问题已显式标注，出现在第三步（读者有足够背景后）
- [x] 核心概念（persistent warp、SM occupation、overlap 三形态、RDMA 三约束）已标注需独立展开
- [x] 进行了广泛视野搜索（vLLM、Megascale-Infer、Tutel 对比）
- [x] 涉及设计方案展示了 baseline → 中间方案 → 最终方案演进（第五步）
- [x] 需要替代方案对比的决策已标注：
  - 第四步：NCCL vs DeepEP（7 维度对比表）、GPU 提交 WR vs CPU 提交 WR
  - 第五步：persistent warp vs NCCL/cooperative groups
- [x] 同一决策的工程挑战合并为一步（第六步三种 overlap 形态）
- [x] 没有独立的「概念/约束映射」章节（RDMA 约束融入第四步，直接推导出设计选择）

## 更新日志

**2026-03-22**：根据 readme.md 实际补充内容，同步更新 learn-plan：
1. 新增「第零步：GPU 进程间通信基础」，覆盖 NVLink、PCIe、RDMA、GPU-Direct、NCCL 抽象
2. 第四步增加「RDMA 编程模型的三个关键约束」子章节，展示 MR 注册、WR 提交、完成检测的设计权衡
3. 第四步增加 NCCL vs DeepEP 对比表（7 个维度）
4. 更新「文章结构建议」，补充第 0 章的具体内容
5. 更新「推荐资源」，补充 RDMA 编程手册、GPU-Direct 文档、NCCL 源码、GPU 架构博客
