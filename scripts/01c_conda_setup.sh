#!/usr/bin/env bash
# 01c_conda_setup.sh —— 昇腾 NPU：Miniconda + 手动建 conda 环境（全自动，带检查点）
#
# 与 01_create_env.sh 的区别：
#   本脚本面向「昇腾镜像自带系统 Python」的场景，会：
#     1) 先探测镜像自带的昇腾栈版本，conda 环境里装【同样的版本】；
#     2) 每步做验证，失败自动降级到「继承系统包」的保底方案；
#     3) 最后直接给出可运行的命令。
#
# 用法:
#   bash scripts/01c_conda_setup.sh
#   PREFIX=/workspace/miniconda3 ENV_NAME=ascend bash scripts/01c_conda_setup.sh
#   TORCH_VER=2.10.0 TORCH_NPU_VER=2.10.0.post4 bash scripts/01c_conda_setup.sh
#
# 详细手动步骤见 docs/CONDA_SETUP.md
set -u
cd "$(dirname "$0")/.."
PROJ="$(pwd)"

PREFIX="${PREFIX:-/workspace/miniconda3}"
ENV_NAME="${ENV_NAME:-ascend}"
PY_VER="${PY_VER:-3.12}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
CONDA_MIRROR="${CONDA_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda}"

# 下面两个默认值取自镜像实测；脚本会先探测并覆盖
TORCH_VER="${TORCH_VER:-}"
TORCH_NPU_VER="${TORCH_NPU_VER:-}"
VLLM_VER="${VLLM_VER:-}"
VLLM_ASCEND_VER="${VLLM_ASCEND_VER:-}"

say()  { echo ""; echo "=============================================================="; echo "== $*"; echo "=============================================================="; }
ok()   { echo "  [ OK ] $*"; }
warn() { echo "  [WARN] $*"; }
err()  { echo "  [FAIL] $*"; }

# ---------------------------------------------------------------
# 步骤 0：前置探测
# ---------------------------------------------------------------
say "步骤 0/7  前置检查"

ARCH="$(uname -m)"
echo "  架构           : $ARCH"
echo "  CANN_HOME      : ${ASCEND_HOME_PATH:-（未设置）}"
SYS_PY="$(which python)"
echo "  当前 python    : $SYS_PY"
echo "  磁盘 /workspace:"; df -h /workspace 2>/dev/null | tail -1 | sed 's/^/    /'

# 探测镜像自带的昇腾栈版本
detect() {
  "$SYS_PY" - "$1" <<'PY' 2>/dev/null
import sys
try:
    from importlib.metadata import version
    print(version(sys.argv[1]))
except Exception:
    pass
PY
}
D_TORCH="$(detect torch)"
D_TORCH_NPU="$(detect torch-npu)"
D_VLLM="$(detect vllm)"
D_VLLM_ASCEND="$(detect vllm-ascend)"

echo ""
echo "  镜像自带版本（conda 里将装同样的）:"
echo "    torch        = ${D_TORCH:-?}"
echo "    torch-npu    = ${D_TORCH_NPU:-?}"
echo "    vllm         = ${D_VLLM:-?}"
echo "    vllm-ascend  = ${D_VLLM_ASCEND:-?}"

TORCH_VER="${TORCH_VER:-${D_TORCH:-2.10.0}}"
TORCH_NPU_VER="${TORCH_NPU_VER:-${D_TORCH_NPU:-2.10.0.post4}}"
VLLM_VER="${VLLM_VER:-${D_VLLM:-0.23.0}}"
VLLM_ASCEND_VER="${VLLM_ASCEND_VER:-${D_VLLM_ASCEND:-0.23.0}}"

# 去掉本地版本后缀（2.10.0+cpu → 2.10.0）
TORCH_VER_CLEAN="${TORCH_VER%%+*}"
echo ""
echo "  将安装: torch==$TORCH_VER_CLEAN  torch-npu==$TORCH_NPU_VER  vllm==$VLLM_VER  vllm-ascend==$VLLM_ASCEND_VER"

# ---------------------------------------------------------------
# 步骤 1：安装 Miniconda
# ---------------------------------------------------------------
say "步骤 1/7  安装 Miniconda 到 $PREFIX"

case "$ARCH" in
  x86_64|amd64)  PKG=Miniconda3-latest-Linux-x86_64.sh ;;
  aarch64|arm64) PKG=Miniconda3-latest-Linux-aarch64.sh ;;
  *) err "不支持的架构: $ARCH"; exit 1 ;;
esac

if [ -x "$PREFIX/bin/conda" ]; then
  ok "已存在: $("$PREFIX/bin/conda" --version)"
else
  cd /tmp
  if [ ! -f "$PKG" ]; then
    echo "  下载 $CONDA_MIRROR/$PKG ..."
    wget -q --show-progress "$CONDA_MIRROR/$PKG" -O "$PKG" || {
      err "下载失败，请检查网络"; exit 1; }
  fi
  bash "$PKG" -b -p "$PREFIX" || { err "安装失败"; exit 1; }
  ok "安装完成"
fi

# 让 conda 在本脚本内可用
# shellcheck disable=SC1091
source "$PREFIX/etc/profile.d/conda.sh"
"$PREFIX/bin/conda" init bash >/dev/null 2>&1 || true
ok "conda: $("$PREFIX/bin/conda" --version)"

# ---------------------------------------------------------------
# 步骤 2：创建环境
# ---------------------------------------------------------------
say "步骤 2/7  创建 conda 环境 $ENV_NAME (python=$PY_VER)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  ok "环境已存在，直接复用"
else
  conda create -y -n "$ENV_NAME" "python=$PY_VER" || { err "创建失败"; exit 1; }
fi
conda activate "$ENV_NAME"
PY="$(which python)"
ok "python: $PY ($(python -V 2>&1))"

case "$(python -V 2>&1)" in
  *3.12*) ok "Python 3.12 —— 与镜像一致，torch_npu 的 cp312 二进制兼容" ;;
  *) warn "Python 版本不是 3.12，torch_npu 可能不兼容，建议重来（PY_VER=3.12）" ;;
esac

python -m pip install -U pip setuptools wheel -i "$PIP_INDEX" >/dev/null 2>&1 || true

# ---------------------------------------------------------------
# 步骤 3：安装昇腾推理栈
# ---------------------------------------------------------------
say "步骤 3/7  安装昇腾推理栈（最耗时，请耐心）"

pipq() { python -m pip install "$@" -i "$PIP_INDEX" || python -m pip install "$@"; }

echo "  [3.1] torch==$TORCH_VER_CLEAN"
pipq "torch==$TORCH_VER_CLEAN" || warn "torch 安装失败"

echo "  [3.2] torch-npu==$TORCH_NPU_VER"
NPU_INSTALL_OK=0
if pipq "torch-npu==$TORCH_NPU_VER"; then
  NPU_INSTALL_OK=1
else
  warn "torch-npu 安装失败"
fi

# --- 检查点：torch_npu 是否真的可用 ---
check_npu() {
  python - <<'PY' 2>/dev/null
import sys
try:
    import torch, torch_npu
    ok = bool(torch.npu.is_available())
    print("OK" if ok else "NO")
except Exception:
    print("ERR")
PY
}
NPU_STATE="$(check_npu)"
echo "  torch_npu 自检: $NPU_STATE"

if [ "$NPU_STATE" != "OK" ]; then
  warn "torch_npu 在 conda 环境里不可用，启用【保底方案 A：继承系统包】"
  SP="$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
  # 反查系统 python 的 site-packages
  SYSSP=""
  for cand in /usr/local/python3.12.13 /usr/local/python3 /usr/local/python; do
    if [ -x "$cand/bin/python" ]; then
      SYSSP="$("$cand/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" 2>/dev/null)"
      [ -n "$SYSSP" ] && break
    fi
  done
  if [ -z "$SYSSP" ]; then
    SYSSP="$(/usr/bin/env -i PATH=/usr/local/bin:/usr/bin:/bin python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" 2>/dev/null)"
  fi
  echo "    conda site-packages : $SP"
  echo "    系统 site-packages  : ${SYSSP:-（未找到）}"
  if [ -n "$SYSSP" ] && [ -d "$SYSSP" ]; then
    echo "$SYSSP" > "$SP/_ascend_system.pth"
    ok "已写入 .pth，conda 环境现在可复用镜像自带的昇腾包"
    NPU_STATE="$(check_npu)"
    echo "    重新自检: $NPU_STATE"
  else
    err "未找到系统 site-packages，请改用镜像自带 Python 运行项目（见 docs/CONDA_SETUP.md 方案 B）"
  fi
fi

echo "  [3.3] vllm==$VLLM_VER"
pipq "vllm==$VLLM_VER" || warn "vllm 安装失败（若保底方案已生效则可忽略）"

# vllm 可能改动 torch，复核
if [ "$NPU_STATE" = "OK" ]; then
  CUR_TORCH="$(detect torch)"
  case "$CUR_TORCH" in
    "$TORCH_VER_CLEAN"*) ok "torch 版本未被 vllm 改动: $CUR_TORCH" ;;
    *) warn "torch 被改为 $CUR_TORCH，重装回 $TORCH_VER_CLEAN"
       pipq "torch==$TORCH_VER_CLEAN" "torch-npu==$TORCH_NPU_VER" || true ;;
  esac
fi

echo "  [3.4] vllm-ascend==$VLLM_ASCEND_VER"
pipq "vllm-ascend==$VLLM_ASCEND_VER" || warn "vllm-ascend 安装失败"

echo "  [3.5] triton-ascend（可选）"
pipq "triton-ascend" || warn "triton-ascend 安装失败（非致命）"

# ---------------------------------------------------------------
# 步骤 4：项目依赖
# ---------------------------------------------------------------
say "步骤 4/7  安装项目依赖"
pipq pyarrow matplotlib tqdm || warn "部分依赖安装失败"

# ---------------------------------------------------------------
# 步骤 5：环境变量
# ---------------------------------------------------------------
say "步骤 5/7  配置昇腾环境变量"

SET_ENV="$(find /usr/local/Ascend -maxdepth 3 -name set_env.sh 2>/dev/null | head -1)"
echo "  set_env.sh: ${SET_ENV:-（未找到）}"
if [ -n "$SET_ENV" ]; then
  # shellcheck disable=SC1090
  source "$SET_ENV" || true
  if ! grep -q "01c_conda_setup.sh 添加" "$HOME/.bashrc" 2>/dev/null; then
    {
      echo ""
      echo "# ---- 昇腾 NPU 环境（由 01c_conda_setup.sh 添加）----"
      echo "source $SET_ENV"
      echo "export ASCEND_RT_VISIBLE_DEVICES=0"
    } >> "$HOME/.bashrc"
    ok "已写入 ~/.bashrc"
  else
    ok "~/.bashrc 已配置过"
  fi
else
  warn "未找到 set_env.sh，若 NPU 不可用请手动 source"
fi

# ---------------------------------------------------------------
# 步骤 6：最终验证
# ---------------------------------------------------------------
say "步骤 6/7  最终验证"
python scripts/02_check_env.py || true

# ---------------------------------------------------------------
# 步骤 7：完成
# ---------------------------------------------------------------
say "步骤 7/7  完成"

cat <<EOF
环境: $ENV_NAME   ($(which python))

以后每次使用（新开终端）:
    source $PREFIX/etc/profile.d/conda.sh
    conda activate $ENV_NAME
    cd $PROJ
    python scripts/02_check_env.py

跑项目:
    python scripts/03_find_model.py
    KEEP=40 CALIB=8 SEQ=256 SKIP_DISTILL=1 SKIP_MANUAL=1 bash scripts/run_all.sh   # 小参数验证
    bash scripts/run_all.sh                                                          # 完整版

导出环境快照（便于复现）:
    pip freeze > requirements-conda-lock.txt

如果 NPU 仍不可用，见 docs/CONDA_SETUP.md 的【保底方案】。
EOF
