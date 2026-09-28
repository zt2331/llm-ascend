# 昇腾 NPU 大模型压缩与部署（Qwen3.6-27B）

> 面向 **AI Infra / 大模型推理** 的完整项目：在**国产昇腾 NPU** 上完成
> **剪枝 → 蒸馏 → 量化 → 评测 → vLLM 服务化 → 压测**，并**自动生成含表格与图表的报告**。
>
> 目标环境：华为昇腾在线开发环境（镜像 `vLLM-Ascend 0.23.0`：
> Ubuntu22.04 / vLLM 0.23.0 / torch 2.10.0 / Python 3.12 / torch_npu 2.10.0.post4 /
> vllm-ascend 0.23.0 / triton_ascend 3.2.2）。

---

## 一、30 秒跑通（已经配好环境的情况）

```bash
cd llm-ascend

# ① 先确认环境（NPU 可用？缺哪些依赖？）
python scripts/02_check_env.py
bash scripts/01b_install_deps.sh     # 缺依赖时装（不用 conda）

# 数据已随仓库提供（data/calib/validation.parquet 510 条 + data/test/test.parquet 580 条），
# 只需校验一次：
python scripts/04_check_data.py

# ② 一键全流程（环境自检 → 找模型 → 备数据 → 剪枝 → 蒸馏 → 量化 → 评测 → 出报告）
bash scripts/run_all.sh

# ③ 看结果
cat output_models/results/REPORT.md
ls  output_models/results/fig_*.png
```

> 首次跑建议先用小参数验证流程通不通（几分钟级）：
> ```bash
> KEEP=40 CALIB=8 SEQ=256 SKIP_DISTILL=1 SKIP_MANUAL=1 bash scripts/run_all.sh
> ```

跑完你会得到：

```
output_models/results/
├── REPORT.md                 # 汇总报告（表格 + 图表引用）
├── summary.csv               # 汇总表（Excel 可开）
├── ppl.json / bench.json / deploy_metric.json   # 原始评测数据
├── fig_ppl.png               # PPL 对比
├── fig_bench.png             # 准确率对比
├── fig_mini_categories.png   # Mini-Bench 分类准确率
├── fig_throughput.png        # decode 吞吐对比
├── fig_ttft.png              # TTFT 对比
├── fig_acc_vs_speed.png      # 精度-速度权衡
└── fig_loadtest.png          # 并发压测吞吐-延迟曲线（跑了压测才有）
```

---

## 一·五、部署 + 可视化对话台

量化完想知道**到底快了多少**，两个终端起服务：

```bash
# 终端 1：vLLM 推理服务（昇腾 NPU）
MAX_NUM_SEQS=8 LANGUAGE_MODEL_ONLY=1 bash scripts/serve_ascend.sh \
    output_models/quantized/Qwen3.6-27B-llmcomp-smooth-W8A8-mm \
    compressed-tensors 8000

# 终端 2：聊天网页（单文件、零依赖、无需构建）
bash scripts/serve_chat.sh 8080   # 页面 + 流式反代（同一端口）
```

浏览器打开 `http://<服务器IP>:8080/`（云 IDE 用 `https://<host>/proxy/8080/`）
→ 页面自动探测接口 → 直接对话。

**每轮回答下方实时显示**：

| 指标 | 含义 |
|---|---|
| **TTFT** | 请求发出 → 首个 token 到达，反映 **prefill** 阶段 |
| **Prefill 速度** | `prompt_tokens ÷ TTFT` |
| **Decode 速度** | `(输出tokens − 1) ÷ (末token − 首token)` |
| 输入/输出 tokens | 取自服务端 `usage`（精确，非估算） |

还有「压测 ×5」按钮输出 avg / p50 / p95，以及 decode 速度趋势图。

> 完整部署清单与启动命令见
> **[`docs/DEPLOY.md`](docs/DEPLOY.md)**。
>
> 度量口径有测试保障：`node tests/test_chat_metrics.js`（用虚拟时钟 + 合成
> SSE 流验证 TTFT / prefill / decode 的数学）。

---

## 二、环境准备

### 2.0 最快路径：不用 conda（⭐ 推荐先这样跑通）

**昇腾镜像自带的系统 Python 已经装好了 `torch_npu / vllm / vllm-ascend`，不需要 conda。**
只要补几个项目依赖即可：

```bash
cd llm-ascend

# 自检：确认 NPU 可用 + 看缺哪些包
python scripts/02_check_env.py

# 补依赖（pyarrow 读写 parquet、matplotlib 出图）
bash scripts/01b_install_deps.sh
# 等价于： pip install pyarrow matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

# 再自检，直到显示「环境完全就绪 ✅」
python scripts/02_check_env.py
```

> 📌 提示：系统 Python 常见路径 `/usr/local/python3.12.13/bin/python`。
> `02_check_env.py` 会打印实际使用的解释器与环境类型。

### 2.1 可选：装 Miniconda 建独立环境（练习用）

> 目的是**练习"像在 CUDA 上一样熟练地搭国产环境"**，不是跑通的必要条件。
> 📖 **完整详细步骤见 `docs/CONDA_SETUP.md`**（含检查点、报错速查、保底方案）。
>
> ⚠️ 最大风险在 `torch_npu`：它与系统 CANN 版本**强绑定**，新环境里必须装
> **同一版本**。`01c_conda_setup.sh` 会自动探测镜像自带版本并对齐，
> 失败时自动降级为「继承系统包」的保底方案。

**方式一：一键脚本（推荐）**

```bash
cd llm-ascend
bash scripts/01c_conda_setup.sh     # 装 Miniconda + 建环境 + 装昇腾栈 + 验证
```

**方式二：按文档手动一步步来** → 见 `docs/CONDA_SETUP.md`

粗略流程：

```bash
# ① 装 Miniconda（自动识别 ARM64/x86_64，用清华镜像）
bash scripts/00_install_miniconda.sh
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# ② 建环境（Python 必须与镜像一致：3.12）
conda create -n ascend python=3.12 -y
conda activate ascend

# ③ 装昇腾栈（版本要探测镜像自带值并对齐）
pip install torch==2.10.0 torch-npu==2.10.0.post4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install vllm==0.23.0 vllm-ascend==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# ④ 项目依赖 + 验证
pip install pyarrow matplotlib tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
python scripts/02_check_env.py
```

**若在新环境里 torch_npu 装不上 / NPU 不可用**，两个保底方案（详见 `docs/CONDA_SETUP.md`）：
1. **继承系统包**：用 `.pth` 把镜像自带的 site-packages 挂进 conda 环境（零风险）
2. **直接用镜像自带 Python**：不激活 conda，直接跑（5 分钟跑通）

---

## 三、分步执行（想单独跑某一步时）

```bash
export PYTHONPATH=$PWD:$PYTHONPATH

python scripts/02_check_env.py                    # 环境自检
python scripts/03_find_model.py                   # 找 base 模型
python scripts/04_check_data.py                   # 校验数据（数据已固定，随 git 管理）

python prune/prune.py     --keep 48 --calib 16 --seq 512     # 剪枝
python distill/distill.py --epochs 1 --max-steps 40          # 蒸馏
python quantize/ascend_quant.py --scheme W8A8                # 昇腾原生 W8A8（推荐）
python quantize/gen_llmcomp.py  --scheme W8A8                # llm-compressor 对照

python eval/eval_ppl.py        --seq 1024         # PPL
python eval/eval_bench.py                         # 准确率（内置，离线可跑）
python eval/deploy_metric.py                      # decode 吞吐 / TTFT

python report/make_report.py                      # 出表 + 出图
```

部署与压测：

```bash
bash scripts/serve_ascend.sh output_models/quantized/<tag> compressed-tensors 8000
python scripts/loadtest.py --port 8000 --concurrency 1,2,4,8 --n-req 64
python report/make_report.py       # 重新出报告，纳入压测曲线
```

---

## 四、常用参数

| 变量 | 默认 | 说明 |
|---|---|---|
| `KEEP` | 48 | 剪枝后保留层数 |
| `CALIB` | 32 | 校准样本条数 |
| `SEQ` | 512 | 校准序列长度 |
| `QUANT_SCHEME` | W8A8 | 量化方案（昇腾推荐 W8A8） |
| `EVAL_SEQ` | 1024 | PPL 评测序列长度 |
| `SKIP_DISTILL` | 0 | 设 1 跳过蒸馏（省时间） |
| `SKIP_MANUAL` | 0 | 设 1 跳过手写 AWQ/GPTQ |
| `MANUAL_MAX_LAYERS` | 4 | 手写 AWQ/GPTQ 量化的层数（0=全部，较慢） |
| `ALLOW_NO_NPU` | 0 | 设 1 允许无 NPU 继续（CPU 很慢） |
| `STAGES` | 全部 | 只跑部分阶段，如 `STAGES=eval,report` |

示例：

```bash
KEEP=40 CALIB=64 SEQ=1024 SKIP_DISTILL=1 bash scripts/run_all.sh
STAGES=eval,report bash scripts/run_all.sh          # 只重新评测并出报告
MANUAL_MAX_LAYERS=0 bash scripts/run_all.sh         # 手写 AWQ/GPTQ 全模型量化（慢）
SKIP_MANUAL=1 bash scripts/run_all.sh               # 跳过硬的手写量化
```

---

## 五、目录结构

```
llm_ascend/
├── config.py                  # 路径/模型搜索（NPU 环境常见目录全覆盖）
├── utils/
│   ├── device.py              # ★ 设备抽象层：NPU/CUDA/CPU 自动切换
│   ├── model_utils.py         # 文本 decoder 定位、量化 ignore 构造
│   ├── dataio.py              # 轻量 parquet 读写（不依赖 datasets 库）
│   ├── quant_check.py         # 量化产物自检（权重 dtype 是否真的是整型）
│   ├── mm_wrapper.py          # 多模态 wrapper 恢复（避免评测 nan）
│   └── npu_compat.py          # ★ 绕开 NPU 缺失算子（cholesky_inverse 等）
├── web/chat.html              # ★ 聊天网页（单文件零依赖，显示 prefill/decode 速度）
├── tests/
│   └── test_chat_metrics.js   # 网页度量口径测试（虚拟时钟 + 合成 SSE）
├── prune/prune.py             # 结构化剪枝（ShortGPT 块重要度）
├── distill/distill.py         # 知识蒸馏（KL，base 当 teacher）
├── quantize/
│   ├── ascend_quant.py        # ★ 昇腾原生量化（msModelSlim，W8A8）——昇腾主力
│   ├── w8a8_smooth.py         # SmoothQuant W8A8（激活离群迁移）——平滑的唯一实现
│   ├── gen_llmcomp.py         # llm-compressor 量化（AWQ/GPTQ/RTN）
│   ├── manual_awq.py          # ★ 从零手写 AWQ（激活感知+缩放折叠+AWQ_ORDER 打包）
│   └── manual_gptq.py         # ★ 从零手写 GPTQ（Hessian 二阶+逐列误差补偿）
├── eval/
│   ├── eval_ppl.py            # PPL
│   ├── eval_bench.py          # 语言建模准确率 + Mini-Bench + 可选 lm-eval 四基准
│   └── deploy_metric.py       # vLLM(Ascend) decode 吞吐 / TTFT / 显存
├── report/make_report.py      # ★ 表格(MD/CSV) + 图表(PNG) 生成
├── scripts/
│   ├── 00_install_miniconda.sh    # 可选：装 Miniconda
│   ├── 01_create_env.sh           # 可选：建 conda 环境
│   ├── 01b_install_deps.sh        # ★ 不用 conda，直接补依赖
│   ├── 01c_conda_setup.sh         # ★ 一键 conda 建环境（自动对齐版本+保底）
│   ├── 02_check_env.py        # 环境自检（版本/设备/算子/CANN）
│   ├── 03_find_model.py
│   ├── 04_check_data.py       # 数据校验（数据固定，只检查不生成）
│   ├── run_all.sh             # ★ 一键全流程
│   ├── serve_ascend.sh        # vLLM(Ascend) 启动
│   ├── serve_chat.sh          # ★ 聊天页 + 流式反代启动器
│   ├── chat_relay.py          # ★ 静态托管 + vLLM SSE 反代（防代理缓冲）
│   ├── npu_mem.py             # ★ 不依赖 npu-smi 的显存诊断
│   └── loadtest.py            # 并发压测
├── deploy/
│   ├── k8s.yaml               # 昇腾 NPU 的 Deployment/Service/HPA
│   └── prometheus.yaml        # vLLM 指标 + NPU 硬件指标采集
└── docs/
    ├── ARCHITECTURE.md        # ★ 架构与算法细节（含手写 AWQ/GPTQ 推导）
    ├── DEPLOY.md              # ★ 可部署清单 + 每个模型的 vLLM 启动命令
    ├── RESULTS.md             # ★ 实测数据 + 国产卡适配踩坑记录
    ├── ASCEND_MIGRATION.md    # ★ CUDA→CANN 迁移差异（面试弹药）
    └── CONDA_SETUP.md         # ★ Miniconda + 手动建环境详细步骤
```

---

## 六、和 NVIDIA 版的差异（本项目唯一必须新增的一层）

| 维度 | NVIDIA | 昇腾 NPU |
|---|---|---|
| 设备名 | `cuda` / `.cuda()` | `npu` / `.npu()`，且**必须先 `import torch_npu`** |
| 显存 API | `torch.cuda.*` | `torch.npu.*` |
| 硬件监控 | `nvidia-smi` | `npu-smi info` |
| 软件栈 | CUDA + cuDNN | **CANN**（图引擎 GE / Runtime / 算子库） |
| 推理框架 | vLLM / SGLang | **vllm-ascend** / **MindIE** |
| 量化工具 | llm-compressor / AutoAWQ | **msModelSlim**（W8A8 为主） |
| 主流量化 | AWQ/GPTQ **INT4** | **W8A8 INT8**（硬件支持最成熟） |
| 执行模式 | Eager + CUDA Graph | **图模式 / 单算子模式**，图编译更关键 |

> 详见 `docs/ASCEND_MIGRATION.md`。

---

## 七、常见问题

**Q: `scripts/02_check_env.py` 显示 NPU 不可用？**
A: ① 确认执行过 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`；
② 若在新建 conda 环境里失败，用 `REUSE_BASE=1 bash scripts/01_create_env.sh` 复用镜像环境。

**Q: 找不到模型？**
A: `python scripts/03_find_model.py` 看搜索到了什么，然后
`export MODEL_PATH=/实际路径/Qwen3.6-27B`。

**Q: 量化阶段失败？**
A: 昇腾优先走 `quantize/ascend_quant.py`（msModelSlim）。若未安装会有清晰提示。
可先用 `llm-compressor` 路径对照，或跳过量化只做部署+评测：
`STAGES=check,model,data,eval,report bash scripts/run_all.sh`。

**Q: 显存不够 / OOM？**
A: 调小 `MAX_MODEL_LEN`、`GMU`（`gpu-memory-utilization`），或提高量化压缩比；
评测阶段会逐个加载模型，脚本已自动释放缓存。

**Q: 27B 模型跑得慢？**
A: 先用小参数验证流程：`KEEP=40 CALIB=8 SEQ=256 SKIP_DISTILL=1 bash scripts/run_all.sh`。

---

## 八、这套项目在面试里能讲什么

1. **国产卡适配**：CUDA→CANN 的软件栈差异、torch_npu 设备抽象、npu-smi 监控。
2. **量化选型差异**：昇腾走 W8A8（Cube 单元原生支持 INT8），NVIDIA 走 AWQ/GPTQ INT4——**为什么必须换**。
3. **量化作用域**：只量化标准 Linear；视觉塔 / 混合线性注意力 / lm_head 保持高精度。
4. **推理性能**：decode（访存密集）vs prefill（计算密集）、吞吐-延迟权衡、并发压测。
5. **工程化**：一键流水线、结果自动出表出图、环境自检与降级策略。

---

## 九、代码托管：推送到远程仓库（Git）

仓库已初始化（分支 `main`），`.gitignore` 已排除权重/产物/数据，
`.gitattributes` 强制文本文件用 **LF** 换行（关键：`.sh` 带 CRLF 在 Linux 上会报
`bash: $'\r': command not found`）。

### 9.1 在本地推送到远程

**① 先在 Gitee/GitHub 上新建一个空仓库**（不要勾选"初始化 README"，保持空仓库）
- Gitee：https://gitee.com/projects/new
- GitHub：https://github.com/new

**② 关联远程并推送**

```bash
cd llm_ascend

# Gitee（国内快，推荐）
git remote add origin https://gitee.com/<你的用户名>/llm_ascend.git

# 或 GitHub
# git remote add origin https://github.com/<你的用户名>/llm_ascend.git

git push -u origin main
```

> 若仓库是**私有**的，推送时会要求账号密码 —— Gitee/GitHub 都需要用
> **访问令牌（Token）** 代替密码，不是登录密码。
> Gitee: 设置 → 私人令牌；GitHub: Settings → Developer settings → Personal access tokens。

### 9.2 在昇腾开发环境拉取

```bash
# 联网环境下
cd ~/work            # 或你的工作目录
git clone https://gitee.com/<你的用户名>/llm_ascend.git
cd llm_ascend

# 之后更新
git pull
```

**拉取后先做两件事**：

```bash
# ① 确认脚本换行符正确（应为 LF）
file scripts/*.sh | grep -i crlf && echo "有 CRLF，需修复" || echo "换行符 OK"

# 若真的出现 CRLF（少数情况），一键修复：
#   sed -i 's/\r$//' scripts/*.sh

# ② 环境自检
python scripts/02_check_env.py
```

### 9.3 常见问题

| 问题 | 解决 |
|---|---|
| `bash: $'\r': command not found` | 脚本是 CRLF：`sed -i 's/\r$//' scripts/*.sh` |
| `Permission denied (publickey)` | 用 HTTPS + Token，或配置 SSH key |
| `src refspec main does not match any` | 先 `git add -A && git commit -m init` |
| `remote origin already exists` | `git remote set-url origin <新地址>` |
| 推送很慢/超时 | 改用 Gitee；或 `git config --global http.postBuffer 524288000` |

### 9.4 可选：把真实结果也纳入版本管理

默认 `output_models/` 被忽略（产物不入库）。若想把**实测报告**提交上去作为证据：

```bash
git add -f output_models/results/REPORT.md output_models/results/*.png
git commit -m "docs: 加入实测实验结果报告与图表"
git push
```
