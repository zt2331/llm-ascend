#!/usr/bin/env bash
# serve_chat.sh —— 启动聊天页 + vLLM 流式反向代理（同一端口）
#
# 用法:
#   bash scripts/serve_chat.sh                # 页面与 API 都在 :8080
#   bash scripts/serve_chat.sh 8080 8000      # 指定 页面端口 与 vLLM端口
#
# 为什么不是简单的静态服务器
# --------------------------
# 云 IDE 把端口代理到 /proxy/<port>/ 下，那层反向代理（多为 nginx）
# 默认开启 proxy_buffering，会把 vLLM 的 SSE 流攒齐再一次性发给浏览器。
# 结果页面上测出来的 TTFT/decode 全是假的 —— 实测出现过
# "TTFT 36s / decode 3207 tok/s / 生成阶段 319ms" 这种值，27B 模型不可能。
#
# 本脚本起的是 scripts/chat_relay.py：它从 vLLM 拿流后立刻逐块转发，
# 并在响应头加 X-Accel-Buffering: no，让前置 nginx 不再缓冲。
# 它还把静态页一起托管 —— 于是【页面与接口同源】，连 CORS 都不需要。
#
# 完整流程:
#   终端 1:  bash scripts/serve_ascend.sh <模型目录> compressed-tensors 8000
#   终端 2:  bash scripts/serve_chat.sh 8080
#   浏览器:  云 IDE → https://<host>/proxy/8080/
#            （页面会自动探测同源接口，一般无需手填地址）
set -e
cd "$(dirname "$0")/.."

PORT="${1:-8080}"
VLLM_PORT="${2:-8000}"

exec python3 scripts/chat_relay.py --port "$PORT" --vllm-port "$VLLM_PORT"
