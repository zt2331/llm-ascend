#!/usr/bin/env bash
# serve_chat.sh —— 托管聊天网页（web/chat.html）
#
# 用法:
#   bash scripts/serve_chat.sh            # 默认 :8080
#   bash scripts/serve_chat.sh 9000       # 指定端口
#
# 它只是一个静态文件服务器（Python 自带 http.server），
# 聊天页面本身是单文件、零依赖，不需要 node/构建。
#
# 完整使用流程（两个终端）:
#   终端 1:  bash scripts/serve_ascend.sh <模型目录> [quantization] 8000
#   终端 2:  bash scripts/serve_chat.sh 8080
#   浏览器:  http://<服务器IP>:8080/
#
# 注意: vLLM 侧必须允许跨域。serve_ascend.sh 已默认带
#       --allowed-origins '["*"]'（★注意是 JSON 数组，不是裸的 *），
#       若你手工起 vllm，请自行加上该参数
#       否则浏览器控制台会报 CORS 错误。
set -e
cd "$(dirname "$0")/../web"

PORT="${1:-8080}"

if [ ! -f chat.html ]; then
  echo "[FAIL] 未找到 chat.html（当前目录: $PWD）" >&2
  exit 1
fi

echo "=============================================================="
echo " 聊天网页已启动"
echo "   目录: $PWD"
echo "   端口: $PORT"
echo ""
echo "   浏览器打开:  http://<本机IP>:$PORT/"
echo "   本机自测:    curl -sI http://127.0.0.1:$PORT/ | head -1"
echo ""
echo "   别忘了同时启动推理服务:"
echo "     bash scripts/serve_ascend.sh <模型目录> compressed-tensors 8000"
echo "   页面上把「vLLM 地址」填成 http://<本机IP>:8000 再点「连接」"
echo ""
echo "   Ctrl+C 停止"
echo "=============================================================="

exec python3 -m http.server "$PORT" --bind 0.0.0.0
