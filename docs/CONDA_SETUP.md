# 昇腾 NPU 环境：Miniconda + 手动建环境（详细步骤）

> 目标：在昇腾开发环境里安装 Miniconda，手动创建一个隔离的 conda 环境，
> 并装好 `torch / torch_npu / vllm / vllm-ascend` 完整推理栈。
>
> ⚠️ **重要前提**：`torch_npu` 与 **CANN 版本强绑定**。镜像里已有可用的
> CANN 和 torch_npu，新建环境时**必须装同一版本**，否则会出现
> `libascendcl.so not found` / 算子找不到 / NPU 不可用等问题。
>
> 每一步都有【检查点】，不通过就别往下走。

---

## 步骤 0：前置检查（2 分钟）

先摸清环境底细，后面所有版本都要对齐它。

```bash
# 0.1 架构与系统
uname -m
cat /etc/os-release | head -2
python -V
which python

# 0.2 CANN 位置与版本
echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
ls -d /usr/local/Ascend/*/ 2>/dev/null
find /usr/local/Ascend -maxdepth 3 -name "set_env.sh" 2>/dev/null | head

# 0.3 镜像自带的昇腾栈版本（★ 记下来，等下 conda 里要装同样的）
python - <<'PY'
from importlib.metadata import version, PackageNotFoundError
for d in ["torch", "torch-npu", "vllm", "vllm-ascend", "triton-ascend", "transformers", "numpy"]:
    try:
        print(f"  {d:<16}{version(d)}")
    except PackageNotFoundError:
        print(f"  {d:<16}(未安装)")
PY

# 0.4 磁盘空间（miniconda + 依赖约需 10GB+）
df -h /workspace /root 2>/dev/null | head -5
```

**【检查点 0】**
- [ ] `uname -m` 记住架构（`x86_64` 还是 `aarch64`，决定下载哪个安装包）
- [ ] 记下镜像自带的 `torch` / `torch-npu` / `vllm` / `vllm-ascend` **确切版本号**
- [ ] 剩余空间 ≥ 15GB

> 📌 本镜像参考值（你要以实际输出为准）：
> `ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0`，
> `torch 2.10.0` / `torch-npu 2.10.0.post4` / `vllm 0.23.0` /
> `vllm-ascend 0.23.0` / `triton-ascend 3.2.2`。

---

## 步骤 1：安装 Miniconda（5 分钟）

```bash
# 1.1 自动识别架构并下载（用清华镜像，快）
cd /tmp
ARCH=$(uname -m)
case "$ARCH" in
  x86_64|amd64)  PKG=Miniconda3-latest-Linux-x86_64.sh ;;
  aarch64|arm64) PKG=Miniconda3-latest-Linux-aarch64.sh ;;
  *) echo "不支持的架构: $ARCH"; exit 1 ;;
esac
echo "架构=$ARCH  安装包=$PKG"

wget -q --show-progress \
  "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/$PKG" -O "$PKG"

# 1.2 静默安装到 /workspace/miniconda3
#     （装 /workspace 下可持久化；装 /root 下可能随容器重启丢失）
bash "$PKG" -b -p /workspace/miniconda3

# 1.3 让 conda 在当前 shell 生效
source /workspace/miniconda3/etc/profile.d/conda.sh

# 1.4 写进 ~/.bashrc，以后新开终端自动可用
/workspace/miniconda3/bin/conda init bash
source ~/.bashrc
```

**【检查点 1】**
```bash
conda --version          # 应输出 conda 25.x 或类似
which conda              # 应是 /workspace/miniconda3/bin/conda
```

**（可选，推荐）配置国内 conda 源**，避免后面下载慢：
```bash
cat > ~/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
conda config --show channels
```

---

## 步骤 2：创建 conda 环境（2 分钟）

**Python 版本必须与镜像一致（3.12）**，否则 torch_npu 的 cp312 二进制不兼容。

```bash
conda create -n ascend python=3.12 -y
conda activate ascend

python -V                # 必须是 3.12.x
which python             # 应是 /workspace/miniconda3/envs/ascend/bin/python
```

**【检查点 2】**
- [ ] `python -V` 显示 **3.12.x**
- [ ] `which python` 指向 conda 环境路径（不是 `/usr/local/...`）

---

## 步骤 3：安装昇腾推理栈（★ 最关键，15–30 分钟）

### 3.1 升级基础工具 + 配置 pip 源

```bash
pip install -U pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple

# 永久设置 pip 源（可选）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.2 安装 torch（昇腾用 CPU 版 torch + torch_npu 提供 NPU 后端）

```bash
pip install torch==2.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
> 说明：`torch 2.10.0+cpu` 是**正常的**，昇腾不依赖 CUDA 版 torch。

### 3.3 安装 torch_npu（★ 版本必须与 torch 严格对应）

```bash
pip install torch-npu==2.10.0.post4 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**【检查点 3.3】** —— 这一步最容易失败，立刻验证：
```bash
python -c "import torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu available:', torch.npu.is_available())"
```
必须是：
```
torch 2.10.0+cpu
torch_npu 2.10.0.post4
npu available: True
```

> ❌ **如果失败**（找不到包 / `libascendcl.so` 报错 / `is_available()` 为 False），
> **立刻跳到文末【保底方案】**，不要继续往下装。

### 3.4 安装 vLLM

```bash
pip install vllm==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

⚠️ vLLM 可能会**改动 torch 版本**，装完必须复核：
```bash
pip show torch | grep -i version          # 应仍是 2.10.0
python -c "import torch; print(torch.__version__)"
```
若被改掉了，重装回来：
```bash
pip install torch==2.10.0 torch-npu==2.10.0.post4 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.5 安装 vllm-ascend 插件

```bash
pip install vllm-ascend==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证导入（vllm-ascend 不暴露 __version__，用包元数据查）
python -c "
from importlib.metadata import version
import vllm_ascend
print('vllm-ascend', version('vllm-ascend'))
print('导入成功')
"
```

### 3.6 安装 triton-ascend（可选但建议）

```bash
pip install triton-ascend==3.2.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**【检查点 3】**
```bash
python - <<'PY'
import torch, torch_npu
print("NPU 可用:", torch.npu.is_available())
print("NPU 数量:", torch.npu.device_count())
print("设备名  :", torch.npu.get_device_name(0))
a = torch.randn(512, 512, dtype=torch.float16).npu()
b = torch.randn(512, 512, dtype=torch.float16).npu()
c = a @ b
torch.npu.synchronize()
print("矩阵乘 OK, shape =", tuple(c.shape))
PY
```
**必须看到「矩阵乘 OK」**，否则不要继续。

---

## 步骤 4：安装项目依赖（5 分钟）

vLLM 已经带上了 `transformers / numpy / safetensors` 等，只需补项目特有的：

```bash
pip install pyarrow matplotlib tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 📌 本项目**不需要 `datasets` 库**（已用 `utils/dataio.py` 基于 pandas+pyarrow 替代），
> 避免它的依赖与 vLLM 要求的 transformers 版本冲突。

---

## 步骤 5：配置昇腾环境变量（3 分钟）

每次新开终端都需要，写进 `~/.bashrc` 一次即可：

```bash
# 先找到 set_env.sh（不同镜像路径不同）
SET_ENV=$(find /usr/local/Ascend -maxdepth 3 -name "set_env.sh" 2>/dev/null | head -1)
echo "找到: $SET_ENV"

cat >> ~/.bashrc <<EOF

# ---- 昇腾 NPU 环境 ----
source $SET_ENV
export ASCEND_RT_VISIBLE_DEVICES=0
EOF

source ~/.bashrc
echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
```

**【检查点 5】**
- [ ] `echo $ASCEND_HOME_PATH` 有值
- [ ] `npu-smi info` 能输出（容器内可能权限受限，报 `-9005` 不影响推理）

---

## 步骤 6：完整验证

```bash
cd /workspace/git-reops/llm-ascend
python scripts/02_check_env.py
```

**期望输出**（关键几行）：
```
[1] Python: 3.12.x
    解释器: /workspace/miniconda3/envs/ascend/bin/python
    环境类型: conda 环境
[2] 关键包版本
    torch                    2.10.0+cpu
    torch-npu              2.10.0.post4
    vllm                        0.23.0
    vllm-ascend                 0.23.0
[3] torch.npu.is_available() = True
      npu:0  Ascend910_9382  显存 61.3 GiB
[5] [ OK ] NPU 矩阵乘成功
结论: 环境完全就绪 ✅
```

---

## 步骤 7：跑项目

```bash
cd /workspace/git-reops/llm-ascend

# 先找模型（确认能搜到 Qwen3.6-27B）
python scripts/03_find_model.py

# 小参数验证流程（几分钟）
KEEP=40 CALIB=8 SEQ=256 SKIP_DISTILL=1 SKIP_MANUAL=1 bash scripts/run_all.sh

# 通过后跑完整版
bash scripts/run_all.sh
```

---

## 步骤 8：把 conda 环境固定下来（可选）

conda 环境不会自动记录"从 pip 装了什么"，导出清单便于重建：

```bash
pip freeze > requirements-conda-lock.txt
# 以后重建：
# pip install -r requirements-conda-lock.txt
```

---

# 🔧 保底方案（步骤 3.3 失败时用）

## 方案 A：conda 环境「继承」镜像自带的昇腾包（推荐）

**原理**：镜像自带 Python 是 3.12，conda 环境也是 3.12 → **ABI 标签都是 `cp312`，
二进制扩展可以复用**。用一个 `.pth` 文件把系统 site-packages 挂进 conda 环境，
就能直接用到镜像里已验证可用的 `torch / torch_npu / vllm / vllm-ascend`。

```bash
conda activate ascend

# 找到 conda 环境的 site-packages
SP=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "conda site-packages: $SP"

# 找到镜像自带 python 的 site-packages
SYSPY=/usr/local/python3.12.13
SYSSP=$($SYSPY/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "system site-packages: $SYSSP"

# 挂进去
echo "$SYSSP" > "$SP/_ascend_system.pth"

# 验证
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
```

- ✅ 优点：**零风险**，直接复用镜像已验证的版本组合
- ⚠️ 缺点：conda 环境不再是"纯净隔离"的（但功能完全正常）

> 若 `SYSPY` 路径不对，用 `which python` 反查系统 python 的真实位置。

## 方案 B：直接用镜像自带 Python（放弃 conda 隔离）

```bash
# 不激活任何 conda 环境
/usr/local/python3.12.13/bin/python scripts/02_check_env.py
```
镜像自带 Python 已经装好全部昇腾栈，只补 `pyarrow matplotlib` 即可跑本项目。

## 方案 C：确认 torch_npu 是否在 PyPI 上有对应版本

```bash
pip index versions torch-npu 2>/dev/null || \
  pip install torch-npu== 2>&1 | head -5
```
若当前 pip 源没有 `2.10.0.post4`，换源或使用华为官方源，或走方案 A。

---

# 常见错误速查

| 报错 | 原因 | 解决 |
|---|---|---|
| `Could not find a version that satisfies torch-npu==2.10.0.post4` | pip 源里没有该版本 | 换源 / 走方案 A / 用镜像自带 |
| `ImportError: libascendcl.so: cannot open shared object file` | 未 source `set_env.sh` | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| `torch.npu.is_available() == False` | torch 与 torch_npu 版本不匹配，或 CANN 路径没配 | 重装配套版本；检查 `ASCEND_HOME_PATH` |
| vLLM 装完 `torch` 版本被改 | vLLM 依赖解析覆盖 | 重装 `torch==2.10.0 torch-npu==2.10.0.post4` |
| `undefined symbol` / `GLIBCXX not found` | conda 的 libstdc++ 比系统旧 | `conda install -c conda-forge libstdcxx-ng` 或走方案 A |
| `npu-smi: command not found` | 只有推理不需要它 | 忽略，不影响 |
| `npu get board type failed. ret is -9005` | 容器内权限受限 | 忽略，不影响推理 |

---

# 时间预估

| 步骤 | 耗时 |
|---|---|
| 0 前置检查 | 2 min |
| 1 装 Miniconda | 5 min |
| 2 建 conda 环境 | 2 min |
| 3 装昇腾栈 | **15–30 min**（最耗时、最易出错） |
| 4 项目依赖 | 5 min |
| 5 环境变量 | 3 min |
| 6-7 验证 + 跑项目 | 10 min + 项目本身时间 |
| **合计** | **约 40–60 分钟** |

> 💡 如果只是想让项目跑起来，**方案 B（直接用镜像自带 Python）5 分钟就能跑**。
> 走完整 conda 路线的价值在于**练习"从零搭国产硬件环境"**，这项能力本身就是面试谈资。
