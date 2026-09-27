#!/usr/bin/env bash
# 01b_install_deps.sh —— 把项目缺的依赖装进【当前 Python】（不需要 conda）
#
# 适用场景：
#   昇腾镜像自带系统 Python（如 /usr/local/python3.12.13/bin/python），
#   里面已有 torch_npu / vllm / vllm-ascend，只缺本项目需要的几个包。
#   此时无需 conda，直接装即可。
#
# 用法:
#   bash scripts/01b_install_deps.sh
#   PIP_INDEX=https://mirrors.aliyun.com/pypi/simple bash scripts/01b_install_deps.sh
set -u
cd "$(dirname "$0")/.."

PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PY="${PY:-python}"

echo "=============================================================="
echo " 安装项目依赖（当前 Python，无需 conda）"
echo " 解释器: $($PY -c 'import sys;print(sys.executable)')"
echo " 版本  : $($PY -V 2>&1)"
echo " 源    : $PIP_INDEX"
echo "=============================================================="

pip_i() {
  # 先试国内镜像，失败则回退官方源
  if ! "$PY" -m pip install "$@" -i "$PIP_INDEX"; then
    echo "[WARN] 镜像源失败，回退默认源重试: $*"
    "$PY" -m pip install "$@" || return 1
  fi
}

echo ""
echo "=== [1/4] 必需：parquet 读写 ==="
pip_i pyarrow && echo "[OK] pyarrow"

echo ""
echo "=== [2/4] 必需：绘图（生成报告图表）==="
pip_i matplotlib && echo "[OK] matplotlib"

echo ""
echo "=== [3/4] 可选：量化库（llm-compressor 路径）==="
pip_i llmcompressor && echo "[OK] llmcompressor" || \
  echo "[SKIP] llmcompressor 未装上（可用昇腾原生 msModelSlim 路径替代）"

echo ""
echo "=== [4/4] 复核环境 ==="
"$PY" scripts/02_check_env.py || true

cat <<EOF

==============================================================
完成。若上面自检显示「环境完全就绪 ✅」，即可运行：
    bash scripts/run_all.sh

先小参数验证流程（推荐）：
    KEEP=40 CALIB=8 SEQ=256 SKIP_DISTILL=1 SKIP_MANUAL=1 bash scripts/run_all.sh
==============================================================
EOF
