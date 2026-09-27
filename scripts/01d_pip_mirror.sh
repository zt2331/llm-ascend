#!/usr/bin/env bash
# 01d_pip_mirror.sh —— 多镜像轮询安装（解决某个源 403 / 超时 / 限流）
#
# 用法:
#   bash scripts/01d_pip_mirror.sh pyarrow matplotlib
#   bash scripts/01d_pip_mirror.sh pyarrow matplotlib tqdm
#   PKGS="pyarrow matplotlib" bash scripts/01d_pip_mirror.sh
set -u

PKGS="${*:-${PKGS:-pyarrow matplotlib tqdm}}"

# 按优先级排列：昇腾/华为环境优先华为云源
MIRRORS="
https://repo.huaweicloud.com/repository/pypi/simple/
https://mirrors.aliyun.com/pypi/simple/
https://pypi.mirrors.ustc.edu.cn/simple/
https://mirrors.cloud.tencent.com/pypi/simple/
https://mirrors.bfsu.edu.cn/pypi/web/simple/
https://pypi.tuna.tsinghua.edu.cn/simple
https://pypi.org/simple/
"

echo "=============================================================="
echo " 多镜像安装: $PKGS"
echo " 架构: $(uname -m)    Python: $(python -V 2>&1)"
echo "=============================================================="

echo ""
echo "=== 当前 pip 源配置 ==="
pip config list 2>/dev/null || echo "(无自定义配置)"
for f in /etc/pip.conf "$HOME/.pip/pip.conf" "$HOME/.config/pip/pip.conf"; do
  [ -f "$f" ] && { echo "--- $f ---"; cat "$f"; }
done

SUCCESS=""
for M in $MIRRORS; do
  echo ""
  echo ">>> 尝试源: $M"
  if pip install $PKGS -i "$M" --retries 5 --timeout 60 --no-cache-dir 2>&1 | tail -6; then
    # 复核是否真的装上了
    if python - "$PKGS" <<'PY'
import sys, importlib.util
missing = [p for p in sys.argv[1].split() if importlib.util.find_spec(p.replace('-','_')) is None]
sys.exit(1 if missing else 0)
PY
    then
      SUCCESS="$M"; echo "  [ OK ] 安装成功: $M"; break
    fi
  fi
  echo "  [FAIL] 该源不可用，换下一个"
done

echo ""
echo "=============================================================="
if [ -n "$SUCCESS" ]; then
  echo "完成！使用的源: $SUCCESS"
  echo "可把它设为默认源:"
  echo "    pip config set global.index-url $SUCCESS"
else
  cat <<'EOF'
所有源都失败。可选方案：
  1) 重试清华源（403 常是临时的，过几分钟再来）
       pip install <包> -i https://pypi.tuna.tsinghua.edu.cn/simple
  2) 用 ModelArts 内网源（若有）
       pip config list     # 查看是否已配置内网源
  3) 手动下载 wheel 再本地安装
       pip download <包> -i <可用源> -d /tmp/whl
       pip install /tmp/whl/*.whl
  4) 检查网络代理
       env | grep -i proxy
EOF
fi
echo "=============================================================="
