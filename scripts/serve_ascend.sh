#!/usr/bin/env bash
# serve_ascend.sh —— 在昇腾 NPU 上启动 vLLM OpenAI 兼容服务
#
# 用法:
#   bash scripts/serve_ascend.sh <模型目录> [quantization] [port]
# 示例:
#   bash scripts/serve_ascend.sh qwen_model/Qwen3.6-27B                 # 不量化
#   bash scripts/serve_ascend.sh output_models/quantized/xxx compressed-tensors
#   bash scripts/serve_ascend.sh output_models/quantized/xxx ascend      # msModelSlim 产物
#
# 说明:
#   vllm-ascend 以插件形式注册 NPU 平台，安装后 `vllm serve` 默认走 NPU。
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MODEL="${1:?用法: bash scripts/serve_ascend.sh <模型目录> [quantization] [port]}"
QUANT="${2:-}"
PORT="${3:-8000}"

# ---- 昇腾环境变量 ----
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

ARGS=(serve "$MODEL" --host 0.0.0.0 --port "$PORT"
      --trust-remote-code --dtype float16
      --max-model-len "${MAX_MODEL_LEN:-4096}"
      --gpu-memory-utilization "${GMU:-0.90}")
if [ -n "$QUANT" ]; then
  ARGS+=(--quantization "$QUANT")
fi

echo "=============================================================="
echo " 启动 vLLM (Ascend NPU) 服务"
echo " 模型:   $MODEL"
echo " 量化:   ${QUANT:-（无）}"
echo " 端口:   $PORT"
echo " 设备:   ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
echo "=============================================================="
echo "启动后可用以下命令验证:"
echo "  curl -s http://127.0.0.1:$PORT/v1/models | head"
echo "  python scripts/loadtest.py --port $PORT"
echo ""

exec vllm "${ARGS[@]}"
