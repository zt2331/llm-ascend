# Qwen3.6-27B 模型结构与官方部署要点（权威版）

> 来源：ModelScope 官方 `Qwen/Qwen3.6-27B` 的 `config.json` / `README.md`，
> 以及 SGLang 官方文档《Qwen3.6-27B on Ascend NPUs》。
> **写代码前先看本文，不要凭经验猜结构。**

---

## 1. 基本信息

| 项 | 值 |
|---|---|
| 架构 | **`Qwen3_5ForConditionalGeneration`**（**多模态**） |
| `model_type` | `qwen3_5` |
| **`language_model_only`** | **`false`** ← 官方开关（vLLM 有对应参数） |
| `tie_word_embeddings` | `false`（lm_head 独立） |
| 参数量 / 精度 | 约 27B / BF16 |
| 权重文件 | **15 个分片，共约 55.4 GB** |
| 上下文 | **262144（256K）**，mRoPE（`mrope_interleaved: true`） |
| transformers | `4.57.1` |

---

## 2. 文本部分（`text_config`）

| 项 | 值 | 说明 |
|---|---|---|
| `num_hidden_layers` | **64** | |
| **`layer_types`** | **48 × `linear_attention` + 16 × `full_attention`** | 周期：每 4 层一个 full |
| `full_attention_interval` | `4` | 与 layer_types 呼应 |
| `hidden_size` | 5120 | |
| `intermediate_size` | 17408 | |
| `num_attention_heads` / `num_key_value_heads` | **24 / 4** | GQA |
| `head_dim` | 256 | |
| `attn_output_gate` | `true`（`output_gate_type: swish`） | 注意力输出门控 |
| `linear_num_key_heads` / `linear_num_value_heads` | 16 / 48 | GatedDeltaNet |
| `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 | |
| `linear_conv_kernel_dim` | 4 | 因果卷积核 |
| `mamba_ssm_dtype` | `float32` | Mamba 状态精度 |
| **`mtp_num_hidden_layers`** | **1** | **多 token 预测层（mtp.\* 命名空间）** |
| `vocab_size` | 248320 | |
| `partial_rotary_factor` | 0.25 | |

> **核心特征**：这是**混合线性注意力（GatedDeltaNet + 全注意力）多模态模型**。
> 48 层线性注意力 + 16 层全注意力交替，**不是**标准 Transformer。

## 3. 视觉部分（`vision_config`）

| 项 | 值 |
|---|---|
| `depth` | **27** |
| `hidden_size` | 1152 |
| `intermediate_size` | 4304 |
| `num_heads` | 16 |
| `patch_size` / `spatial_merge_size` / `temporal_patch_size` | 16 / 2 / 2 |
| `out_hidden_size` | 5120（对齐文本 hidden） |
| 特殊 token | `image_token_id=248056`、`video_token_id=248057`、`vision_start/end=248053/248054` |

---

## 4. 官方部署要点

### vLLM（官方 README）
```bash
# 标准（多模态，8 卡 TP）
vllm serve Qwen/Qwen3.6-27B --port 8000 --tensor-parallel-size 8 \
  --max-model-len 262144 --reasoning-parser qwen3

# ★ 纯文本：跳过视觉编码器与多模态 profiling，省显存给 KV cache
vllm serve Qwen/Qwen3.6-27B --port 8000 --tensor-parallel-size 8 \
  --max-model-len 262144 --reasoning-parser qwen3 --language-model-only

# MTP 投机解码
vllm serve ... --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```
> 推荐 `vllm >= 0.19.0`。

### 昇腾 NPU（SGLang 官方文档）
| 功能 | 参数 |
|---|---|
| 张量并行 | `--tp-size 2` |
| **量化** | **`--quantization modelslim`** ← **昇腾官方量化路径就是 msModelSlim** |
| Chunked Prefill | `--chunked-prefill-size 32768` |
| NPU Graph | 默认开启；`--disable-cuda-graph` 关闭 |
| 投机解码 | `--speculative-algorithm NEXTN --speculative-num-steps 3 ...` |

**官方已提供 W8A8 量化版本**：
```
Eco-Tech/Qwen3.6-27B-w8a8   （W8A8 量化版，36.45 GB）
  → 单卡 64GB 可放，--tp-size 1 即可（A2/A3 系列）
```
> 本项目单卡 **61.27 GiB**，直接放这个官方量化版**绰绰有余**。

---

## 5. 本项目涉及该模型时最容易出错的地方（踩坑清单）

### 5.1 剪枝：`layer_types` 必须**按保留索引重排**，不能截断前缀
本项目的剪枝保留「块重要度最高」的层，**索引是任意的**（不是前 N 层）。
- ❌ 错误：`layer_types[:keep]` → 把"前 N 层的类型"套到实际保留的层上
- ✅ 正确：`[original_types[i] for i in keep_idx]`
- 同时：裁剪后层类型不再满足每 4 层周期 → **必须清空 `full_attention_interval`**，
  否则模型会按周期重新推导，与实际 `layer_types` 冲突。

**实测影响**：64→54 层（删 10 层任意索引）时，旧实现有 **19/54 个位置类型错误**。

### 5.2 MTP 层
`mtp_num_hidden_layers: 1`，MTP 权重在 `mtp.*` 命名空间，与文本层数无关。
剪枝/量化时若报缺少 `mtp.*` 权重，把该字段设为 0 关闭。

### 5.3 量化：llm-compressor 会剥离多模态 wrapper
量化后 `config.json` 变成 `Qwen3_5ForCausalLM`（纯文本）、丢失视觉塔，
但 `processor_config.json` 仍留着 → **vLLM 报**
`TypeError: Expected Qwen3_5Config, but found Qwen3_5TextConfig`。

两种处理：
1. **纯文本部署**：`python scripts/09_make_text_only.py <目录> --apply`（移走处理器文件）
2. **完整多模态**：`python scripts/08_fix_mm_wrapper.py <目录> --orig <base>`（恢复 wrapper），
   再配 `--language-model-only` 跳过视觉编码器

### 5.4 ignore 正则必须带 `re:` 前缀
`"re:.*linear_attn.*"` 而不是 `".*linear_attn.*"`，否则按字面精确匹配、**静默失效**。

### 5.5 量化作用域（本模型）
| 模块 | 是否量化 | 原因 |
|---|---|---|
| `mlp.{gate,up,down}_proj` | ✅ | 参数量大头 |
| `self_attn.{q,k,v,o}_proj`（16 层） | ✅ | 标准 Linear |
| **`linear_attn.*`（48 层）** | ❌ | 混合线性注意力，vLLM 量化加载器命名不匹配 → 乱码 |
| **`visual.*`** | ❌ | 视觉塔，结构特殊且基准是文本任务 |
| `lm_head` / 各种 norm | ❌ | 输出分布 / 数值范围敏感 |

### 5.6 显存
- 27B **BF16 ≈ 54 GB** → 61 GiB 卡上做评测极易 OOM
- **W8A8 ≈ 27–36 GB** → 可行
- 官方 W8A8 版 36.45 GB（含未量化的 fp16 部分）
- 本项目 27B 评测请用 `--backend vllm`（保持量化精度，不反量化）

---

## 6. 与本项目的对应关系

| 环节 | 脚本 | 注意点 |
|---|---|---|
| 结构探测 | `scripts/05_model_index_check.py` | 打印层数 / layer_types 分布 / 是否 MoE / 是否多模态 |
| 量化产物诊断 | `scripts/07_inspect_quant.py` | 量化是否落盘、哪些层被量化、config 是否被剥离 |
| 恢复多模态 wrapper | `scripts/08_fix_mm_wrapper.py` | 两步量化法第二步 |
| 纯文本化 | `scripts/09_make_text_only.py` | 移走残留处理器文件使 vLLM 一致加载 |
| 剪枝 | `prune/prune.py` | layer_types 按索引重排 + 清理 full_attention_interval |
| 量化 | `quantize/*.py` | ignore 带 `re:`；跳过 linear_attn / visual |
| 评测 | `eval/eval_ppl.py --backend vllm` | 避免 27B OOM |
| 部署 | `scripts/serve_ascend.sh` | 支持 `LANGUAGE_MODEL_ONLY=1` |
