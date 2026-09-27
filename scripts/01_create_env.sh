#!/usr/bin/env bash
# 01_create_env.sh —— 创建 conda 环境并安装昇腾推理/量化依赖
#
# 设计原则：**不破坏镜像自带的可工作环境**。
#   - 默认新建独立环境 env=ascend（隔离，适合练习/交付）
#   - 如果新环境的 torch_npu/vllm-ascend 装不上（版本矩阵不匹配），
#     脚本会打印「回退方案」，可直接用镜像自带环境跑（BASE_PY=python）
#
# 用法:
#   bash scripts/01_create_env.sh                 # 新建 conda 环境 ascend
#   ENV_NAME=myenv bash scripts/01_create_env.sh  # 自定义名字
#   REUSE_BASE=1 bash scripts/01_create_env.sh    # 不新建，直接用镜像自带环境装缺失依赖
set -e
cd "$(dirname "$0")/.."
PROJ="$PWD"

ENV_NAME="${ENV_NAME:-ascend}"
PY_VER="${PY_VER:-3.12}"
REUSE_BASE="${REUSE_BASE:-0}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "=== 环境准备 (ENV_NAME=$ENV_NAME, PY_VER=$PY_VER, REUSE_BASE=$REUSE_BASE) ==="

if [ "$REUSE_BASE" = "1" ]; then
  PY="python"
  echo "[模式] 复用镜像自带 Python: $($PY -V 2>&1)"
else
  # ---- 激活 conda ----
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "[ERROR] 未找到 conda。请先运行: bash scripts/00_install_miniconda.sh"
    exit 1
  fi
  conda create -y -n "$ENV_NAME" "python=$PY_VER" || echo "[SKIP] 环境已存在"
  conda activate "$ENV_NAME"
  PY="python"
  echo "[模式] conda 环境: $ENV_NAME ($($PY -V 2>&1))"
fi

echo ""
echo "=== [1/3] 基础依赖 ==="
"$PY" -m pip install -U pip -i "$PIP_INDEX"

# 说明：torch / torch_npu / vllm / vllm-ascend 的版本必须严格对应。
# 下图镜像自带版本：torch 2.10.0 + torch_npu 2.10.0.post4 + vllm 0.23.0 + vllm-ascend 0.23.0
echo ""
echo "=== [2/3] 昇腾核心栈（按镜像自带版本对齐） ==="
"$PY" -m pip install "torch==2.10.0" -i "$PIP_INDEX" || \
  echo "[WARN] torch 安装失败，可能镜像已内置或需指定源"

"$PY" -m pip install "torch-npu==2.10.0.post4" -i "$PIP_INDEX" || \
  echo "[WARN] torch_npu 安装失败 —— 若在 conda 新环境中失败，请改用 REUSE_BASE=1 复用镜像环境"

"$PY" -m pip install "vllm==0.23.0" -i "$PIP_INDEX" || \
  echo "[WARN] vllm 安装失败"

"$PY" -m pip install "vllm-ascend==0.23.0" -i "$PIP_INDEX" || \
  echo "[WARN] vllm-ascend 安装失败"

"$PY" -m pip install "triton-ascend==3.2.2" -i "$PIP_INDEX" || \
  echo "[WARN] triton-ascend 安装失败（非致命）"

echo ""
echo "=== [3/3] 项目所需的量化 / 评测 / 绘图依赖 ==="
"$PY" -m pip install \
  transformers datasets accelerate safetensors sentencepiece huggingface_hub \
  pandas pyarrow numpy tqdm matplotlib \
  -i "$PIP_INDEX"

# llm-compressor 为可选（昇腾上主要用 msModelSlim 做 W8A8）
"$PY" -m pip install llmcompressor -i "$PIP_INDEX" || \
  echo "[WARN] llmcompressor 安装失败（可改用 quantize/ascend_quant.py 的 msModelSlim 路径）"

echo ""
echo "=== 自检 ==="
"$PY" scripts/02_check_env.py || true

cat <<EOF

[OK] 环境准备结束。

如果自检显示 torch_npu 不可用，请执行回退方案（用镜像自带环境，保证能跑）：
    REUSE_BASE=1 bash scripts/01_create_env.sh
    python scripts/02_check_env.py

确认可用后，一键跑全流程：
    bash scripts/run_all.sh

EOF
