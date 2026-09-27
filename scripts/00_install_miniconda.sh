#!/usr/bin/env bash
# 00_install_miniconda.sh —— 安装 Miniconda（自动识别 ARM64 / x86_64）
#
# 昇腾 ModelArts 环境常为 aarch64(Kunpeng)，脚本会自动选对应安装包。
# 用法:  bash scripts/00_install_miniconda.sh
set -e

MIRROR="${MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda}"
PREFIX="${PREFIX:-$HOME/miniconda3}"

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64)  PKG="Miniconda3-latest-Linux-x86_64.sh" ;;
  aarch64|arm64) PKG="Miniconda3-latest-Linux-aarch64.sh" ;;
  *) echo "[ERROR] 不支持的架构: $ARCH"; exit 1 ;;
esac

echo "=== [1/3] 架构=$ARCH  安装包=$PKG  目标=$PREFIX ==="

if [ -x "$PREFIX/bin/conda" ]; then
  echo "[SKIP] 已存在 conda: $("$PREFIX/bin/conda" --version)"
else
  echo "=== [2/3] 下载并安装 Miniconda ==="
  cd /tmp
  if [ ! -f "$PKG" ]; then
    curl -fL --retry 3 -o "$PKG" "$MIRROR/$PKG" || wget -O "$PKG" "$MIRROR/$PKG"
  fi
  bash "$PKG" -b -p "$PREFIX"
fi

echo "=== [3/3] 初始化 conda（写入 ~/.bashrc） ==="
"$PREFIX/bin/conda" init bash >/dev/null 2>&1 || true

cat <<EOF

[OK] Miniconda 安装完成: $PREFIX

下一步（在当前 shell 生效）:
    source "$PREFIX/etc/profile.d/conda.sh"
    conda --version

然后建环境:
    bash scripts/01_create_env.sh

EOF
