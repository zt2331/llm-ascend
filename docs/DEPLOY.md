# 部署指南（昇腾 NPU / vllm-ascend）

> 本文只讲**怎么把已经量化好的产物跑起来**。
> 量化流程见 `docs/ARCHITECTURE.md`，实测数据见 `docs/RESULTS.md`。

---

## 0. 先看你手上到底有什么

产物目录名由脚本自动生成，**先列一遍**再照抄命令：

```bash
cd /workspace/git-reops/llm-ascend

# 所有量化产物
ls -1 output_models/quantized/

# 剪枝 / 蒸馏产物
ls -1 output_models/pruned/ output_models/distilled/ 2>/dev/null

# 基础模型
ls -1 /workspace/models/

# 看某个产物是什么格式（决定用哪个 --quantization）
python -m utils.quant_check output_models/quantized/<目录名>
```

`quant_check` 会直接告诉你：**权重是不是整型**（= 真量化）、用的是
`weight_scale+input_scale`（W8A8）还是 `weight_packed`（4bit），
以及 config/ignore 是否自洽。

> ⚠️ **多模态产物注意 `-mm` 后缀**：Qwen3.6-27B 是
> `Qwen3_5ForConditionalGeneration`，llm-compressor 保存时会剥掉多模态
> wrapper。量化脚本保存后会自动跑 `utils/mm_wrapper.py` 恢复，输出到
> **`<原目录>-mm`**。**部署要用 `-mm` 那个目录**，否则 vLLM 加载失败或 PPL=nan。

---

## 1. 可部署清单

| # | 模型 | 产物目录 | `--quantization` | 昇腾可部署 |
|---|---|---|---|---|
| 1 | **SmoothQuant W8A8** ★主力 | `Qwen3.6-27B-llmcomp-smooth-W8A8-mm` | `compressed-tensors` | ✅ |
| 2 | RTN W8A8（无平滑基线） | `Qwen3.6-27B-llmcomp-rtn-W8A8-mm` | `compressed-tensors` | ✅ |
| 3 | msModelSlim W8A8 | `Qwen3.6-27B-msmodelslim-W8A8` | `modelslim` | ✅ |
| 4 | 官方 W8A8 发布版 | `Eco-Tech/Qwen3.6-27B-w8a8` | `compressed-tensors` | ✅ |
| 5 | 基础模型 BF16 | `/workspace/models/Qwen3.6-27B` | *（不传）* | ✅（显存紧） |
| 6 | 剪枝 / 蒸馏产物 | `output_models/pruned/*` | 视其上叠加的量化而定 | ✅ |
| 7 | AWQ W4A16 | `*-llmcomp-awq-W4A16` | — | ❌ 不支持 |
| 8 | GPTQ W4A16 | `*-llmcomp-gptq-W4A16` | — | ❌ 不支持 |
| 9 | 手写 AWQ/GPTQ | `*-manual-awq-4bit` / `*-manual-gptq-4bit` | — | ❌ 自研格式 |

**为什么 7~9 不能部署**：vllm-ascend 0.23.0 的 compressed-tensors 路径
**只实现了 W8A8**。加载 W4A16 会直接抛：

```
NotImplementedError: No compressed-tensors compatible scheme was found
for quant_type=W4A16, layer_type=linear.
```

它们仍可在 NVIDIA 上部署，也可以在昇腾上用 transformers 回退方式**评测**
（不追求速度），但**不能作为"国产卡部署"的成果**。

---

## 2. 部署命令

统一走封装脚本 `scripts/serve_ascend.sh`，它已经内置了我们踩过的所有坑
（`dtype=bfloat16`、`--allowed-origins`、Mamba 并发限制等）。

### ① SmoothQuant W8A8（推荐从这里开始）

```bash
cd /workspace/git-reops/llm-ascend

MAX_NUM_SEQS=8 MAX_MODEL_LEN=4096 LANGUAGE_MODEL_ONLY=1 \
  bash scripts/serve_ascend.sh \
  output_models/quantized/Qwen3.6-27B-llmcomp-smooth-W8A8-mm \
  compressed-tensors 8000
```

### ② RTN W8A8（对照基线）

```bash
MAX_NUM_SEQS=8 MAX_MODEL_LEN=4096 LANGUAGE_MODEL_ONLY=1 \
  bash scripts/serve_ascend.sh \
  output_models/quantized/Qwen3.6-27B-llmcomp-rtn-W8A8-mm \
  compressed-tensors 8000
```

### ③ msModelSlim W8A8（昇腾官方路径）

```bash
MAX_NUM_SEQS=8 MAX_MODEL_LEN=4096 \
  bash scripts/serve_ascend.sh \
  output_models/quantized/Qwen3.6-27B-msmodelslim-W8A8 \
  modelslim 8000
```

### ④ 基础模型 BF16（不量化对照）

```bash
MAX_NUM_SEQS=4 MAX_MODEL_LEN=4096 LANGUAGE_MODEL_ONLY=1 \
  bash scripts/serve_ascend.sh /workspace/models/Qwen3.6-27B "" 8000
```

> 27B BF16 ≈ 55 GB，卡只有 61 GB，KV cache 余量很小，
> 所以 `MAX_NUM_SEQS` 要压到 4 以下，或把 `MAX_MODEL_LEN` 降到 2048。

### ⑤ 官方 W8A8

```bash
# 先下载
modelscope download --model Eco-Tech/Qwen3.6-27B-w8a8 --local_dir /workspace/models/Qwen3.6-27B-w8a8

MAX_NUM_SEQS=8 MAX_MODEL_LEN=4096 LANGUAGE_MODEL_ONLY=1 \
  bash scripts/serve_ascend.sh /workspace/models/Qwen3.6-27B-w8a8 compressed-tensors 8000
```

---

## 3. 起聊天网页（显示 prefill / decode 速度）

**两个终端**：

```bash
# 终端 1：推理服务
bash scripts/serve_ascend.sh <模型目录> compressed-tensors 8000

# 终端 2：网页（静态托管，零依赖）
bash scripts/serve_chat.sh 8080
```

浏览器打开 `http://<服务器IP>:8080/`，页面上把「vLLM 地址」填成
`http://<服务器IP>:8000`，点**连接**，然后就能对话。

**页面会显示每轮的**：

| 指标 | 含义 |
|---|---|
| **TTFT** | 请求发出 → 第一个 token 到达，反映 **prefill** 阶段 |
| **Prefill 速度** | `prompt_tokens ÷ TTFT` |
| **Decode 速度** | `(输出tokens − 1) ÷ (末token − 首token)`，即稳态逐 token 速度 |
| 输入/输出 tokens | 取自服务端 `usage`（精确，不是估算） |
| 总耗时 / 生成阶段 | 端到端 vs 纯生成 |

还有「压测 ×5」按钮：同一问题连发 5 次，输出 avg / p50 / p95，
用来看抖动。

> **为什么 decode 用「末 token 时刻」而不是「流结束时刻」**：
> 流末尾还有 usage 块和 `[DONE]`，算进去会**低估 decode 速度约 20%**。
> 这个坑是 `tests/test_chat_metrics.js` 抓出来的。

---

## 4. 踩坑速查（部署侧）

| 现象 | 原因 / 处理 |
|---|---|
| `aclnnQuantMatmulWeightNz` 报 161002 | `dtype` 必须是 **bfloat16**，不能 float16。脚本已默认 |
| `max_num_seqs exceeds available Mamba cache blocks (N)` | Qwen3.6 是 Mamba 混合架构，每序列占一个 block。按报错里的 N 调小 `MAX_NUM_SEQS` |
| `Expected Qwen3_5Config, but found Qwen3_5TextConfig` | 用了**没恢复 wrapper** 的目录。改用 `-mm` 目录 |
| PPL 全是 nan | 同上：`weight_scale` 没加载上。先 `python -m utils.quant_check <目录>` |
| 浏览器报 CORS | vLLM 缺 `--allowed-origins '*'`。`serve_ascend.sh` 已默认加上 |
| 显存不够 | 加 `LANGUAGE_MODEL_ONLY=1`（跳过视觉塔，省几个 GB） |
| `npu-smi info` 报 `-9005` | **无害**，是容器内 DCMI 管理接口不通，与计算无关。用 `torch.npu.memory_allocated()` 看显存即可 |

---

## 5. 一键验证服务是否正常

```bash
# 模型列表
curl -s http://127.0.0.1:8000/v1/models | head -c 400

# 发一条，看是否流式返回
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<模型名>","messages":[{"role":"user","content":"你好"}],
       "max_tokens":32,"stream":true}' | head -5

# 并发压测（脚本自带，输出 TTFT / P50 / P95 / 吞吐 / 错误率）
python scripts/loadtest.py --port 8000
```
