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

# 终端 2：聊天页 + 流式反代（★不是普通静态服务器，见 §3.2）
bash scripts/serve_chat.sh 8080
```

访问：直连环境 `http://<服务器IP>:8080/`；云 IDE `https://<host>/proxy/8080/`。
页面会**自动探测**接口地址，一般无需手填。

**页面显示每轮**：

| 指标 | 含义 |
|---|---|
| **TTFT** | 请求发出 → 第一个 token 到达，反映 **prefill** 阶段 |
| **Prefill 速度** | `prompt_tokens ÷ TTFT` |
| **Decode 速度** | `(输出tokens − 1) ÷ (末token − 首token)`，即稳态逐 token 速度 |
| 输入/输出 tokens | 取自服务端 `usage`（精确，不是估算） |
| 总耗时 / 生成阶段 | 端到端 vs 纯生成 |

另有两个开关：**思考模式**（默认关，关闭时发送
`chat_template_kwargs: {"enable_thinking": false}`，不再输出一长段思维链）、
**压测 ×5**（同问题连发 5 次，输出 avg / p50 / p95）。

### 3.2 ★ 为什么不能只用 `python -m http.server`

云 IDE 把端口代理到 `/proxy/<port>/` 下，那层反向代理（多为 nginx）
**默认开启 `proxy_buffering`**：它会把 vLLM 的 SSE 流**攒齐再一次性**发给浏览器。

一旦被缓冲，页面上的数字**全是假的**：

| 指标 | 假象 | 真实原因 |
|---|---|---|
| TTFT | 36.11 s | 其实是**整个生成耗时**（首个 chunk 直到最后才到） |
| Decode | **3207.9 tok/s** | 27B 模型不可能——所有 token 在同一瞬间到达 |
| 生成阶段 | 319 ms | 1024 个 token 挤在 319ms 内到达 = 非流式 |

**解法**：`scripts/serve_chat.sh` 起的是 `scripts/chat_relay.py`，
一个**页面 + 流式反向代理**的中继：

```
浏览器 ──HTTPS──> 云IDE代理 ──HTTP──> chat_relay ──HTTP──> vLLM:8000
```

它从 vLLM 拿到 SSE 后**立刻逐块 flush**，并在响应头加
**`X-Accel-Buffering: no`**（nginx 认这个头，客户端设不了），
让前置代理不再缓冲。顺带把静态页一起托管 → **页面与接口同源，不需要 CORS**。

**万一前置代理仍然缓冲**，页面会**明确标出**「⚠️ 检测到响应被缓冲」并说明
TTFT/decode 不可信，而不是显示一个看起来很正常的假数字。判据：输出 ≥10 个
token 但生成阶段占比 <5%（正常应占 60%+）。

> 想直连不走代理测真实数字（在服务器上跑，不经过任何代理）：
> ```bash
> python scripts/loadtest.py --port 8000    # 直接打 127.0.0.1:8000
> ```

> **为什么 decode 用「末 token 时刻」而不是「流结束时刻」**：
> 流末尾还有 usage 块和 `[DONE]`，算进去会**低估 decode 速度约 20%**。
> 这个坑是 `tests/test_chat_metrics.js` 抓出来的。

### 3.1 云 IDE / 远程环境：地址必须走代理

如果你是通过**华为云 online IDE**（或任何 `.../proxy/<port>/` 形式）访问页面，
浏览器地址会长这样：

```
https://online-xxxx.huawei.com/proxy/8080/chat.html
```

此时**不能**把 vLLM 地址填成 `http://127.0.0.1:8000`，原因有二：

1. **`127.0.0.1` 是「你浏览器所在机器」**，不是跑 vLLM 的那台服务器；
2. 页面是 **HTTPS**，请求 HTTP 会被浏览器按**混合内容**直接拦掉。

表现都是 `Failed to fetch`。**正确做法**：把端口换成 8000 的**同源代理地址**

```
https://online-xxxx.huawei.com/proxy/8000
```

> 页面会**自动推断**这个地址并预填（识别 `/proxy/<port>/` 前缀），
> 一般不用手改。若 vLLM 不在 8000，改成 `/proxy/<实际端口>` 即可。
>
> 走代理时是**同源请求，根本不涉及 CORS**，所以 `--allowed-origins` 在
> 这个场景下不是必需的（直连时才需要）。

自测代理是否通：浏览器直接打开 `https://<host>/proxy/8000/v1/models`，
能返回模型列表 JSON 就说明通了。

---

## 4. 踩坑速查（部署侧）

| 现象 | 原因 / 处理 |
|---|---|
| `aclnnQuantMatmulWeightNz` 报 161002 | `dtype` 必须是 **bfloat16**，不能 float16。脚本已默认 |
| `max_num_seqs exceeds available Mamba cache blocks (N)` | Qwen3.6 是 Mamba 混合架构，每序列占一个 block。按报错里的 N 调小 `MAX_NUM_SEQS` |
| `Expected Qwen3_5Config, but found Qwen3_5TextConfig` | 用了**没恢复 wrapper** 的目录。改用 `-mm` 目录 |
| PPL 全是 nan | 同上：`weight_scale` 没加载上。先 `python -m utils.quant_check <目录>` |
| `argument --allowed-origins: invalid loads value: '*'` | ★该参数要的是 **JSON 数组**，必须写 `'["*"]'` 而不是 `'*'`。`serve_ascend.sh` 已修正默认值 |
| `ValueError: Free memory on device (26/61 GiB) ... is less than desired GPU memory utilization (0.9, 55 GiB)` | **显存被别的进程占着**。注意 vLLM 是按 `gmu × 总量` 判定的，不是按空闲量。先跑 `python scripts/npu_mem.py` 看谁占着（不依赖坏掉的 npu-smi），再 `python scripts/npu_mem.py --kill` 清理 |
| 浏览器报 CORS | vLLM 缺 `--allowed-origins`。脚本已默认带上 `["*"]`；若要限定来源：`ALLOWED_ORIGINS='["http://10.0.0.5:8080"]'` |
| 页面报 **`Failed to fetch`** | ★最常见：① 地址填了 `127.0.0.1`（那是浏览器自己的机器）；② HTTPS 页面请求 HTTP 被按混合内容拦。**用 `serve_chat.sh` 起中继即可自动解决**，见 §3.2 |
| **TTFT 异常大、decode 上千 tok/s** | ★响应被前置代理缓冲了（不是流式到达）。用 `scripts/serve_chat.sh` 启动，它会加 `X-Accel-Buffering: no`；页面也会明确标出该轮数字不可信 |
| 模型先输出一大段「思考过程」 | 思考模式。页面取消勾选「思考模式」即可（发送 `enable_thinking: false`）；或给 vLLM 加 `--default-chat-template-kwargs '{"enable_thinking": false}'` |
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
