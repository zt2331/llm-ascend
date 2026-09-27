# 项目架构与技术细节（昇腾 NPU 版）

## 1. 总览

```
data/校准语料
     │
     ├─[剪枝] prune/prune.py          ShortGPT 块重要度删层      → 模型变小
     │
     ├─[蒸馏] distill/distill.py      base 当 teacher, KL 找回   → 精度恢复
     │
     ├─[量化] quantize/                                          → 省显存/提速
     │     ├─ ascend_quant.py     昇腾原生 msModelSlim  W8A8  ★昇腾主力
     │     │                      （校准式 min/max，**不含平滑**）
     │     ├─ w8a8_smooth.py      SmoothQuant + W8A8 ★平滑的唯一实现
     │     ├─ gen_llmcomp.py      llm-compressor (AWQ/GPTQ/RTN/W8A16)
     │     │                      --method smooth 会委托给 w8a8_smooth.py
     │     ├─ manual_awq.py       ★从零手写 AWQ（激活感知+缩放折叠+位序打包）
     │     └─ manual_gptq.py      ★从零手写 GPTQ（Hessian+逐列误差补偿）
     │
     ├─[评测] eval/
     │     ├─ eval_ppl.py         PPL（左移对齐交叉熵）
     │     ├─ eval_bench.py       语言建模准确率 + Mini-Bench + 可选 lm-eval 四基准
     │     └─ deploy_metric.py    vllm-ascend: decode 吞吐 / TTFT / 显存
     │
     └─[服务化] scripts/serve_ascend.sh + loadtest.py + deploy/{k8s,prometheus}.yaml
              └─[报告] report/make_report.py  → 表格(MD/CSV) + 图表(PNG)
```

全流程由 `config.MODEL_PATH` 驱动，产物统一落 `output_models/`。

---

## 2. 模块职责

| 模块 | 职责 | 关键技术 |
|---|---|---|
| `config.py` | 集中路径/模型搜索 | 环境变量 > qwen_model > ModelArts 目录 > ModelScope/HF 缓存；绝对路径 |
| `utils/device.py` | **★ 设备抽象层** | NPU/CUDA/CPU 自动切换；`torch_npu` 注册；显存/缓存/同步 API 统一 |
| `utils/model_utils.py` | 模型工具 | 文本 decoder 定位、多模态识别、量化 ignore 构造 |
| `prune/prune.py` | 结构化剪枝 | **ShortGPT 块重要度** = 层输入/输出余弦相似度，删最高（最冗余）层 |
| `distill/distill.py` | 蒸馏找回 | base = teacher，student 学 teacher 软 logits（KL），温度 T，乘 T² |
| `quantize/ascend_quant.py` | **昇腾原生量化** | msModelSlim 的 W8A8（昇腾主力方案）。**只有 `w_bit/a_bit`+`Calibrator` 校准，无平滑/离群抑制** → 本质是「带校准的 RTN」 |
| `quantize/w8a8_smooth.py` | SmoothQuant W8A8 | 激活离群迁移到权重 `Y=(X/s)(W·s)ᵀ`。**平滑的唯一实现**；`gen_llmcomp.py --method smooth` 委托到此 |
| `quantize/gen_llmcomp.py` | 调库量化 | llm-compressor `oneshot` + QuantizationModifier/GPTQModifier，负责 AWQ/GPTQ/RTN |
| `quantize/manual_awq.py` | **手写 AWQ** | 激活感知 + 逐通道缩放 + 缩放折叠 + AWQ_ORDER 打包 |
| `quantize/manual_gptq.py` | **手写 GPTQ** | Hessian 二阶 + 逐列误差补偿 + GPTQ 布局打包 |
| `eval/eval_ppl.py` | PPL | 左移对齐 cross-entropy，同口径可比 |
| `eval/eval_bench.py` | 基准 | next-token 准确率 + 内置 Mini-Bench + 可选 lm-eval |
| `eval/deploy_metric.py` | 部署指标 | vLLM(Ascend) decode tok/s、TTFT、显存 |
| `report/make_report.py` | **报告生成** | Markdown/CSV 表格 + 7 类 PNG 图表 |
| `scripts/run_all.sh` | 一键编排 | 分阶段、失败不中断、日志落盘 |
| `scripts/loadtest.py` | 并发压测 | 并发扫描 → QPS/延迟 P50/P95/TTFT/错误率 |
| `deploy/k8s.yaml` | 生产部署 | 昇腾 NPU 资源的 Deployment/Service/HPA |
| `deploy/prometheus.yaml` | 可观测 | vLLM 指标 + NPU 硬件指标采集 |

---

## 3. 量化算法细节（手写部分）

### 3.1 手写 AWQ（`manual_awq.py`）

**核心恒等式**：`X·Wᵀ = (X/s)·(W·s)ᵀ` —— 权重大 s 倍、激活小 s 倍，结果不变。

**流程**：
1. 采集**归一化层输出**的激活峰值（q/k/v 共享同一输入，gate/up 共享同一输入）；
2. 按**输入通道**计算缩放：
   `s_j = (max|X_j|)^α / (max|W_j|)^(1−α)`，归一化到均值 1 后限幅；
3. 缩放权重 `W' = W·diag(s)`；
4. 对 `W'` 做 **per-group(128) 非对称量化** → `q / scale_q / zero`；
5. **缩放折叠**：把 `1/s` 折进前置 Norm 的 weight（`γ' = γ/s`），
   该 Norm 输出天然变成 `X/s`，**推理时零额外开销**；
6. 按内核 `AWQ_ORDER = [0,2,4,6,1,3,5,7]` 打包 `qweight/qzeros`；
7. 输出完整 `config.json`（含 `model_type/architectures` + `quantization_config`）。

**⚠️ 关键坑（务必记住）**：
> 缩放**必须按输入通道**。若一组内所有通道同乘常数 s，
> 则 `scale_q` 也同乘 s，`round(W·s / (s·scale_q))` 会把 s **约掉** → 等于没做 AWQ。
> 只有改变**通道间相对分布**才有效。

**为什么位序是 `[0,2,4,6,1,3,5,7]`**：vLLM/AutoAWQ 内核按特定交错顺序从 int32 中取 4bit，
用自然序 0..7 打包会**能加载但输出乱码**。

### 3.2 手写 GPTQ（`manual_gptq.py`）

**流程**：
1. 采集每个 Linear 的输入 `X`，构造 **Hessian** `H = XᵀX`（二阶信息）；
2. **阻尼**：`H += λ·mean(diag(H))·I`，避免病态；死通道（`diag(H)==0`）置零防 NaN；
3. 对 `H⁻¹` 做上三角 Cholesky 分解，得到逐列补偿系数；
4. **逐列贪心量化 + 误差补偿**：
   ```
   量化第 j 列 → ŵⱼ
   err = (wⱼ − ŵⱼ) / H⁻¹[j,j]
   W[:, j:] −= err ⊗ H⁻¹[j, j:]        # 把误差传播给后面未量化的列
   ```
5. 分块（block=128）处理，块间也做补偿；
6. 按 vLLM GPTQ 布局打包：
   `qweight[in/8, out]`、`qzeros[gnum, out/8]`、`scales[gnum, out]`、`g_idx[in]`。

**AWQ vs GPTQ**：
| | AWQ | GPTQ |
|---|---|---|
| 依据 | 激活幅度（一阶统计） | Hessian（二阶） |
| 手段 | 缩放保护重要通道 | 逐列误差补偿 |
| 速度 | 快（只统计激活） | 慢（算 Hessian + 逐列） |
| 精度 | 4bit 很强 | 通常略高，低 bit 更明显 |

---

## 3.5 量化库的两条路线：通用库 vs 厂商原生库（★国产卡适配关键）

本项目**同一模型用两套库各量化一遍**，这是国产卡适配岗最想看的对比。

| | **llm-compressor** | **msModelSlim** |
|---|---|---|
| 出身 | vLLM 社区（通用开源） | **华为昇腾官方** |
| 入口脚本 | `quantize/gen_llmcomp.py`（AWQ/GPTQ/RTN）<br>`quantize/w8a8_smooth.py`（SmoothQuant W8A8） | `quantize/ascend_quant.py` |
| 产物目录 | `<模型>-llmcomp-<method>-<scheme>` | `<模型>-msmodelslim-<scheme>` |
| 产物格式 | compressed-tensors | 昇腾原生量化格式 |
| 后端 | 通用 PyTorch 算子（昇腾上可能 fallback CPU） | 面向达芬奇 Cube 单元（INT8 原生支持） |
| 部署引擎 | vLLM / vllm-ascend | MindIE / vllm-ascend |
| 算法覆盖 | RTN / AWQ / GPTQ / SmoothQuant | 昇腾官方量化流程（W8A8 为主） |

### 为什么要两条都做

1. **技术选型由硬件决定**：昇腾 Cube 对 INT8 有原生支持 → **W8A8 是昇腾主力**；
   NVIDIA 生态里 AWQ/GPTQ 的 INT4 更普遍。同模型迁移时量化方案要跟着换。
2. **通用库的兼容性风险**：llm-compressor 走通用算子，在昇腾上可能某些算子
   没有 NPU 实现而 fallback 到 CPU，性能断崖。
3. **厂商原生库的优势**：msModelSlim 针对昇腾硬件调优，且产出的模型
   能被 MindIE / vllm-ascend 最稳地加载。
4. **面试价值**：能讲清"同一模型在两套工具链上的差异与取舍"，
   比只会用一套库有说服力得多。

### 本项目 7 条量化路径一览

| 路径 | 脚本 | 库 | 说明 |
|---|---|---|---|
| 昇腾原生 W8A8 | `ascend_quant.py` | **msModelSlim** | 华为官方，昇腾主力 |
| SmoothQuant W8A8 | `w8a8_smooth.py` | llm-compressor | 激活离群迁移，通用库 |
| AWQ 4bit | `gen_llmcomp.py --method awq` | llm-compressor | `AWQModifier + QuantizationModifier` |
| GPTQ 4bit | `gen_llmcomp.py --method gptq` | llm-compressor | Hessian 二阶补偿 |
| RTN 基线 | `gen_llmcomp.py --method rtn` | llm-compressor | 朴素量化，对照用 |
| 手写 AWQ | `manual_awq.py` | **纯 PyTorch** | 展示算法实现细节 |
| 手写 GPTQ | `manual_gptq.py` | **纯 PyTorch** | Hessian + 逐列补偿 |

> ⚠️ **ignore 正则必须带 `re:` 前缀**（如 `"re:.*linear_attn.*"`），
> 否则按字面精确匹配、静默失效 —— 本项目踩过此坑，
> 导致 linear_attn 被误量化、vLLM 加载后输出异常。

---

## 4. 昇腾适配要点

### 4.1 设备抽象层（`utils/device.py`）
唯一必须新增的一层：业务代码不写死设备，迁移只改这一处。

| 操作 | NVIDIA | 昇腾 |
|---|---|---|
| 设备字符串 | `cuda` | `npu`（**必须先 import torch_npu**） |
| 清缓存 | `torch.cuda.empty_cache()` | `torch.npu.empty_cache()` |
| 同步 | `torch.cuda.synchronize()` | `torch.npu.synchronize()` |
| 显存查询 | `torch.cuda.memory_allocated()` | `torch.npu.memory_allocated()` |
| 可见设备 | `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES` |
| 监控 | `nvidia-smi` | `npu-smi info` |

### 4.2 量化选型差异（重点）
- **昇腾**：达芬奇 Cube 单元**原生支持 INT8** → **W8A8 是主力**（msModelSlim / MindIE）。
- **NVIDIA**：INT4 生态成熟 → AWQ/GPTQ 更普遍。
- **结论**：同一模型迁到昇腾，量化方案要跟着换；手写 AWQ/GPTQ 主要价值在于
  **理解算法 + 在 NVIDIA 侧验证可正确解码**。

### 4.3 常见坑
- **算子 fallback**：无 NPU 实现的算子回退 CPU → 性能断崖（不报错，只是慢）。
- **动态 shape**：触发图重编译 → TTFT 抖动。
- **版本矩阵**：`CANN ↔ torch_npu ↔ torch ↔ vLLM ↔ vllm-ascend` 五者强绑定。

---

## 5. 指标口径

| 指标 | 口径 |
|---|---|
| PPL | 同 tokenizer、同测试集、同 seq_len，`exp(平均 NLL)`，左移对齐 |
| next-token 准确率 | 模型 argmax 预测是否等于真实下一个 token |
| Mini-Bench | 内置 30 题四选一，按选项对数概率（长度归一）选最大 |
| decode tok/s | vLLM 预热后批量生成的稳态吞吐 |
| TTFT | 流式首 token 时间（prefill 主导） |
| 并发 QPS/延迟/P95 | `loadtest.py` 并发扫描，`并发 = QPS × 平均延迟`（Little's Law） |

---

## 6. 面试可讲（昇腾 + 推理向）

1. **国产卡适配**：CUDA→CANN 软件栈差异、设备抽象层、`npu-smi`、torch_npu。
2. **量化选型由硬件决定**：昇腾 W8A8（Cube 原生 INT8）vs NVIDIA AWQ/GPTQ INT4。
3. **手写 AWQ/GPTQ**：激活感知缩放与折叠、Hessian 二阶补偿、`AWQ_ORDER` 位序——
   以及"**能加载 ≠ 能正确解码**"。
4. **量化作用域**：只量化标准 Linear；视觉塔/混合线性注意力/lm_head 保持高精度。
5. **推理性能**：decode（访存密集）vs prefill（计算密集）、吞吐-延迟权衡、压测与分位数。
6. **工程化**：一键流水线、失败降级、结果自动出表出图。
