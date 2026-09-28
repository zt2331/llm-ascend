# 实验结果记录（实测数据）

> 所有数字均为**本项目脚本实测输出**，实验条件已注明。
> 复现命令见各节末尾。

---

## 1. 实验环境

| 项 | 值 |
|---|---|
| 硬件 | 昇腾 NPU Ascend910_9382，单卡 **61.27 GiB** |
| CANN | 9.1.0 |
| 软件 | torch 2.10.0 / torch_npu 2.10.0.post4 / vLLM 0.23.0 / vllm-ascend 0.23.0 |
| 模型 | **Qwen3.6-27B**（多模态，64 层 = 48 linear_attention + 16 full_attention） |
| 数据 | **WikiText-2**：校准 510 条 / 评测 580 条（列名 `text`） |
| 评测 | PPL，`--seq 512`，vLLM 后端，`--max-num-seqs 8`，dtype bfloat16 |

---

## 2. 量化算法对比（核心结果）

| 模型 | 精度格式 | 权重 | 激活 | PPL ↓ | **Δ vs base** | 昇腾可部署 |
|---|---|---|---|---|---|---|
| **base** | FP16 | FP16 | FP16 | **8.02** | — | ✅ |
| **W8A8（SmoothQuant）** | int-quantized | INT8 | INT8（动态 per-token） | **8.09** | **+0.87%** | ✅ |
| **AWQ** | pack-quantized | INT4 | FP16 | **8.20** | **+2.24%** | ❌ 不支持 |

**结论**
1. **W8A8 最优**（+0.87%），且是**昇腾唯一可部署**的方案 → 昇腾上应主推 W8A8。
2. AWQ 精度也不错（+2.24%），但 `vllm-ascend` 不支持 W4A16，**只能用于评测对比**。
3. 8bit 权重 + 动态激活量化 > 4bit 权重 + fp16 激活 —— 位宽优势抵消了激量化误差。

**复现**
```bash
export MODEL_PATH=/workspace/models/Qwen3.6-27B
QMETHODS=smooth,awq KEEP=54 CALIB=128 SEQ=1024 STAGES=prune,quantize bash scripts/run_all.sh
python eval/eval_ppl.py --model /workspace/models/Qwen3.6-27B \
  --backend vllm --language-model-only --seq 512
python eval/eval_ppl.py --model output_models/quantized/*-mm --backend vllm --seq 512
```

---

## 2.5 ⚠️ 重要：当前对比混淆了「位宽」与「算法」两个变量

上面 8.09 (W8A8) vs 8.20 (AWQ) 的对比**不能直接说"SmoothQuant 比 AWQ 好"**，
因为两者**同时变了两个变量**：

| 维度 | W8A8(SmoothQuant) | AWQ(W4A16) |
|---|---|---|
| **权重位宽** | **8 bit**（256 级） | **4 bit**（**仅 16 级**） |
| 激活位宽 | 8 bit（动态 per-token） | 16 bit（不量化） |
| 权重量化算法 | min/max per-channel | 激活感知缩放 + 网格搜索 |
| 激活量化算法 | **SmoothQuant 平滑** | — |

**结论解读**
- 位宽差 4 bit → 权重量化固有误差相差约 **2^4 = 16 倍**；算法优化难以完全弥补。
- 因此 **W8A8 优于 W4A16 是符合预期的**（也是文献中的普遍规律）。
- AWQ 的"激活感知"优势应体现在**同为 4bit 时**与 RTN-4bit 的对比上。

### 正确的对照实验设计（控制变量）

| # | 配置 | 隔离的变量 | 状态 |
|---|---|---|---|
| 1 | W8A8 + **RTN** | W8A8 的朴素基线 | 待测 |
| 2 | W8A8 + **SmoothQuant** | ← **平滑的增益** | ✅ 8.09 |
| 3 | W4A16 + **RTN** | 4bit 的朴素基线 | 待测 |
| 4 | W4A16 + **AWQ** | ← **AWQ 的增益** | ✅ 8.20 |

计算增益：
```
平滑增益 = PPL(RTN-W8A8)  - 8.09
AWQ 增益 = PPL(RTN-W4A16) - 8.20
```

**复现**
```bash
python quantize/gen_llmcomp.py --method rtn --scheme W8A8 \
  --model output_models/pruned/pruned-Qwen3.6-27B-56layers --calib 128 --seq 1024
python quantize/gen_llmcomp.py --method rtn --scheme W4A16 \
  --model output_models/pruned/pruned-Qwen3.6-27B-56layers --calib 128 --seq 1024
python eval/eval_ppl.py --model output_models/quantized/*-mm --backend vllm --seq 512
```

### 文献参考退化区间（与本次结果一致）

| 方法 | 典型退化 | 本次实测 |
|---|---|---|
| SmoothQuant W8A8 | <1% | +0.87% ✅ |
| AWQ W4A16 | 1~3% | +2.24% ✅ |
| RTN W4A16 | 5~15% | 待测 |

> 📌 注意区分：**SmoothQuant 不是朴素方法**。朴素 W8A8 是 `--method rtn`
> （直接 min/max + round）；SmoothQuant 是在其之上叠加了激活离群迁移
> `Y=(X/s)(W·s)ᵀ`（MIT，2022）。

---

## 3. 关键对照实验：校准数据质量的影响

同一套代码、同一模型，**只换校准/评测语料**：

| 语料 | base | W8A8(SmoothQuant) | Δ | AWQ | Δ |
|---|---|---|---|---|---|
| 合成兜底语料（早期） | 12.6 | **19.40** | **+54%** | 14.32 | +13.6% |
| **WikiText-2（真实）** | **8.02** | **8.09** | **+0.87%** | 8.20 | +2.24% |

> ⚠️ 两行**不可直接比较**：base 从 12.6 → 8.02 说明语料难度不同。
> 有意义的是**同一行内的相对退化**。

**结论（重要）**
- 早期 W8A8 退化 +54%，**根因是校准数据**（当时是脚本内置的二十几段合成文本，
  且只有 16 条 × 512 token ≈ 8K token），不是 W8A8/SmoothQuant 算法本身。
- 换成真实语料（510 条，CALIB=128）后，W8A8 退化降到 **+0.87%**。
- 原理：W8A8 要量化**激活**，且 SmoothQuant 的平滑系数
  `s = max|X|^α / max|W|^(1-α)` **完全依赖校准数据的激活统计**；
  数据差 → 平滑系数错 → 激活分布被"平滑"坏 → 精度崩。
- AWQ 不量化激活，对校准数据不敏感，所以早期退化（+13.6%）远小于 W8A8（+54%）。

---

## 4. 国产卡适配踩坑记录（面试素材）

### 4.1 算子覆盖度不足 → CPU fallback
```
CAUTION: The operator 'aten::bitwise_left_shift.Tensor_out' is not currently
supported on the NPU backend and will fall back to run on the CPU.
Decompressing model: 100%|██| 256/256 [01:08<00:00, 3.74it/s]
```
- 4bit 权重的**位解包**在 NPU 无实现 → 回退 CPU → 反量化耗时 **68 秒**（3.74 it/s）
- **不报错，只是慢** —— 这类问题最容易被忽略

### 4.1b 算子缺失 + fallback 有 bug → 直接崩（比 4.1 更严重）
```
[W] VariableFallbackKernel.cpp:250 Warning: CAUTION: The operator
    'aten::cholesky_inverse' is not currently supported on the NPU backend
    and will fall back to run on the CPU.
...gptq_quantize.py:154:  H = torch.cholesky_inverse(H)
[W] ArgSortKernelNpuOpApi.cpp:26 Warning: kernel [ArgSort] can not support
    dtype int32 or int64 on AiCore, Now this kernel is running on AiCpu.
2026-09-28T11:39:29 | GPTQ | METRIC - time 88.18s
2026-09-28T11:39:29 | GPTQ | METRIC - error 0.28
RuntimeError: device_allocator INTERNAL ASSERT FAILED at
"/pytorch/c10/core/CachingDeviceAllocator.h":116
Allocator for npu is not a DeviceAllocator.
```
- GPTQ 求 Hessian 逆用的 `aten::cholesky_inverse` 在 NPU 无实现 → torch_npu 的
  **CPU fallback 路径本身有 bug** → 抛 `Allocator for npu is not a DeviceAllocator`
- 后果不是「慢」，是**跑到第 1 层就崩**，且已耗掉 **88 秒/层**（64 层 ≈ 90 分钟白跑）
- 与 4.1 的区别：4.1 是「能跑但慢」，这里是「**不能跑**」——同一根因（算子缺失）的两种后果
- **解法**：不修 torch_npu，而是把该算子显式搬到 CPU 计算再搬回（`utils/npu_compat.py`）。
  Hessian 是 `[in,in]` 方阵（5120²≈105MB / 17408²≈1.2GB），CPU 完全放得下

### 4.2 量化格式支持面窄
```
NotImplementedError: No compressed-tensors compatible scheme was found
for quant_type=W4A16, layer_type=linear.
    at vllm_ascend/quantization/compressed_tensors_config.py
```
- `vllm-ascend` 的 compressed-tensors **只支持 W8A8**，不支持 W4A16/INT4
- 与官方文档一致：昇腾侧量化路径是 `msModelSlim`（W8A8 为主）

### 4.3 量化算子对 scale 的 dtype 有要求
```
AclNN_Parameter_Error(EZ1001): Tensor scale not implemented for DT_FLOAT16,
should be in dtype support list [DT_UINT64, DT_BFLOAT16, DT_INT64, DT_FLOAT]
```
- 强制 `--dtype float16` 会让 weight_scale 变 fp16 → 昇腾 `aclnnQuantMatmulWeightNz` 拒绝
- 模型原生是 **bfloat16**，用 `--dtype bfloat16` 即可

### 4.4 Mamba 混合架构的并发限制
```
ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (196).
Each decode sequence requires one Mamba cache block...
```
- Qwen3.6 是 Mamba/线性注意力混合架构，**每个 decode 序列占一个 Mamba cache block**
- vLLM 默认 `max_num_seqs=256` > 可用 196 → 必须限制（评测用 8 足够）

### 4.5 多模态模型量化后 config 被剥离
- llm-compressor 保存时会把 `Qwen3_5ForConditionalGeneration` 写成
  `Qwen3_5ForCausalLM`：丢 `vision_config`/`text_config`、视觉塔权重丢失
- 但权重键名仍是 wrapper 命名空间 → transformers 键名对不上 →
  `weight_scale` 加载不上 → **随机初始化 → PPL = nan**
- vLLM 侧还会因残留 `processor_config.json` 报 config 类型错误
- **解法**：`utils/mm_wrapper.py` 恢复完整 wrapper（原始 config + 视觉塔
  + 量化后的 language_model），已做成量化脚本保存后的自动步骤

### 4.6 gen 长度与 max_model_len 冲突
```
The decoder prompt (length 512) plus the number of requested output tokens
(at least 1) exceeds the maximum model length of 512.
```
- `max_model_len` 必须 **> 输入长度**（要留出至少 1 个输出 token）

---

## 5. 待补充

| 项 | 说明 |
|---|---|
| 剪枝 56 层的 PPL | 剪枝 + 蒸馏后的精度 |
| RTN W8A8 | 对照组，验证 SmoothQuant 增益 |
| GPTQ W4A16 | 二阶补偿路线对比 |
| decode 吞吐 / TTFT | `deploy_metric.py` |
| 并发压测 | `loadtest.py`（需先起服务） |
| 官方 W8A8 版本对比 | `Eco-Tech/Qwen3.6-27B-w8a8`（36.45GB）作为基线 |

---

## 6. 一句话总结

> 在昇腾 NPU 上完成 Qwen3.6-27B（多模态 + 混合线性注意力）的 W8A8/AWQ 量化：
> **W8A8 精度退化仅 +0.87%（PPL 8.02→8.09）且是昇腾唯一可部署方案**；
> 通过对照实验定位出早期 +54% 退化源于**校准数据质量**而非算法，
> 并沉淀了量化算子 dtype 限制、Mamba 并发限制、算子 CPU fallback、
> 多模态 wrapper 剥离等国产卡适配踩坑。
