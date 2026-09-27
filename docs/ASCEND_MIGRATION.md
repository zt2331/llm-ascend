# CUDA → 昇腾 CANN 迁移差异（面试弹药）

> 本文是「国产卡适配经验」的核心素材。把每条都**实测确认**后写进简历/面试回答。

---

## 1. 软件栈对照

| 层次 | NVIDIA | 昇腾 |
|---|---|---|
| 驱动/固件 | NVIDIA Driver | NPU Driver + Firmware |
| 通信库 | NCCL | **HCCL** |
| 计算库 | cuBLAS / cuDNN / cuFFT | **CANN 算子库**（aclNN 等） |
| 编译/图引擎 | CUDA Graph / TensorRT | **GE（Graph Engine）** |
| 运行时 | CUDA Runtime | **AscendCL / Runtime** |
| 框架适配 | torch.cuda | **torch_npu** |
| 推理引擎 | vLLM / TensorRT-LLM / SGLang | **vllm-ascend / MindIE** |
| 量化工具 | llm-compressor / AutoAWQ / GPTQ | **msModelSlim（msmodelslim）** |
| 监控 | `nvidia-smi` / DCGM | **`npu-smi info`** |

**一句话**：算子层基本兼容（都是 PyTorch），**运行层与工具链要换**。

---

## 2. 硬件架构差异

| | NVIDIA GPU | 昇腾 NPU |
|---|---|---|
| 架构 | SM（流多处理器） | **达芬奇（DaVinci）** |
| 计算单元 | CUDA Core + Tensor Core | **Cube（矩阵）/ Vector（向量）/ Scalar（标量）** |
| 内存 | HBM / GDDR | HBM（910 系列）/ LPDDR（推理卡） |
| 低精度支持 | FP16/BF16/INT8/**INT4**/FP8 | FP16/BF16/**INT8** 为主，INT4 支持面窄 |

**关键结论**：昇腾 **Cube 单元对 INT8 有原生支持**，所以 **W8A8 是昇腾上的主流量化方案**；
而 NVIDIA 生态里 **AWQ/GPTQ 的 INT4** 更普遍。**同一个模型迁到昇腾，量化选型要跟着换。**

---

## 3. 代码层差异（本项目如何处理）

| 项目 | 原写法（CUDA） | 昇腾写法 | 本项目做法 |
|---|---|---|---|
| 设备 | `"cuda"` | `"npu"` | `utils/device.py` 自动选择 |
| 注册后端 | 自动 | **必须 `import torch_npu`** | `device.py` 内统一导入 |
| 张量搬移 | `.cuda()` / `.to("cuda")` | `.npu()` / `.to("npu")` | `dev.get_device()` |
| 清缓存 | `torch.cuda.empty_cache()` | `torch.npu.empty_cache()` | `dev.empty_cache()` |
| 同步 | `torch.cuda.synchronize()` | `torch.npu.synchronize()` | `dev.synchronize()` |
| 显存查询 | `torch.cuda.memory_allocated()` | `torch.npu.memory_allocated()` | `dev.memory_allocated_gb()` |
| 设备数 | `torch.cuda.device_count()` | `torch.npu.device_count()` | `dev.device_count()` |
| 可见设备 | `CUDA_VISIBLE_DEVICES` | **`ASCEND_RT_VISIBLE_DEVICES`** | `serve_ascend.sh` 内设置 |

> 这就是"设备抽象层"存在的意义：**业务代码不写死设备，迁移只改一层。**

---

## 4. 迁移中最容易踩的坑（务必实测记录）

### 4.1 算子覆盖度不足 → CPU fallback
某些算子没有 NPU 实现时，框架会**回退到 CPU 执行**，表现为**性能断崖式下跌**（不是报错，而是慢）。
**排查方法**：看日志里是否有 fallback 警告；用 profiler 看算子耗时分布。
**应对**：换等价算子、升级 CANN 版本、或该层不量化。

### 4.2 动态 shape 触发图重编译
昇腾依赖**图编译**。变长输入（不同 prompt 长度）会导致**反复重新编译**，
表现为**首 token 延迟抖动**、前几次请求特别慢。
**应对**：固定/分档 shape、启用分块预填充（chunked prefill）、预热。

### 4.3 图模式 vs 单算子模式
- **图模式**：整体编译成图，性能好但灵活性差（不支持动态控制流）。
- **单算子模式（Eager）**：逐个算子执行，灵活但慢。
**结论**：和图编译相关的配置（类似 CUDA 上 `enforce_eager` 的开关）会显著影响性能，
需要像调 CUDA Graph 一样去调。

### 4.4 量化格式不通用
NVIDIA 的 AWQ/GPTQ（INT4，`qweight/qzeros/scales/g_idx`）与昇腾的量化格式**不互通**。
**在昇腾上量化必须用昇腾的链路（msModelSlim）**，产出昇腾能识别的模型。
直接把 CUDA 侧的 AWQ 产物搬到昇腾 → 加载失败或输出乱码。

### 4.5 版本兼容矩阵严格
`CANN ↔ torch_npu ↔ torch ↔ vLLM ↔ vllm-ascend` 五者版本**强绑定**。
任意一个版本不匹配都可能启动失败。**先查官方兼容矩阵再动手**。

### 4.6 环境变量
使用前通常需要：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
```
否则找不到算子库或设备。

---

## 5. 面试怎么讲（三个必答点）

**Q1：你为什么做昇腾适配？**
> "我在 NVIDIA 上用 vLLM 做过量化和部署，想验证**已有的推理优化经验迁移到国产硬件后哪些成立、哪些失效**。
> 所以我把同一套评测口径（PPL、准确率、decode 吞吐、TTFT）在两套硬件上跑了对比。"

**Q2：迁移最大的困难？**
> "两点：一是**算子覆盖度**，没有 NPU 实现的算子会回退到 CPU，性能断崖；
> 二是**动态 shape 触发图重编译**，长变长输入会引起 TTFT 抖动。
> 这两个在 NVIDIA 上不太会遇到——CUDA 那边是 CUDA Graph，这边是图编译。"

**Q3：昇腾和 NVIDIA 量化有什么不同？**
> "昇腾对 **W8A8(INT8)** 支持最成熟，因为达芬奇 Cube 单元对 INT8 有原生支持；
> NVIDIA 生态里 **AWQ/GPTQ 的 INT4** 更主流。
> 所以迁到昇腾时我没有直接搬 INT4，而是走 **msModelSlim 的 W8A8 链路**——
> 这是**硬件能力决定技术选型**的典型例子。"

---

## 6. 自测清单（跑完项目后逐条打勾）

- [ ] `npu-smi info` 能正确显示 NPU 型号与显存
- [ ] `python -c "import torch, torch_npu; print(torch.npu.is_available())"` 为 True
- [ ] NPU 上矩阵乘成功执行
- [ ] vLLM/vllm-ascend 能起服务并返回正常文本（非乱码）
- [ ] 记录到至少一个「算子 fallback」或「图重编译」的实例
- [ ] 记录昇腾侧量化方案与 NVIDIA 侧的差异（W8A8 vs INT4）
- [ ] 拿到 NVIDIA vs 昇腾 的吞吐/TTFT 对比数据
