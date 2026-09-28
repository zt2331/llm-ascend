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
# 环境变量:
#   LANGUAGE_MODEL_ONLY=1   跳过视觉编码器（官方 --language-model-only）
#   MAX_MODEL_LEN / GMU     覆盖默认长度与显存利用率
#   DTYPE                   加载精度，默认 bfloat16（★昇腾量化算子要求 bf16/fp32，
#                           用 float16 会报 aclnnQuantMatmulWeightNz 161002）
#   MAX_NUM_SEQS            最大并发序列，默认 8（★Qwen3.6 是 Mamba 混合架构，
#                           每序列占一个 Mamba cache block。设太大报
#                           "max_num_seqs exceeds available Mamba cache blocks (N)"，
#                           按报错里的 N 调小即可）
#   ALLOWED_ORIGINS         允许的跨域来源，默认 "*"（聊天网页需要）
#   ENABLE_PREFIX_CACHING   默认 1，多轮对话复用 prefix 可显著降 TTFT
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
      --trust-remote-code --dtype "${DTYPE:-bfloat16}"
      --max-model-len "${MAX_MODEL_LEN:-4096}"
      --gpu-memory-utilization "${GMU:-0.90}"
      --max-num-seqs "${MAX_NUM_SEQS:-8}"
      --allowed-origins "${ALLOWED_ORIGINS:-*}")
if [ -n "$QUANT" ]; then
  ARGS+=(--quantization "$QUANT")
fi
# prefix caching：多轮聊天复用历史 KV，TTFT 明显下降
if [ "${ENABLE_PREFIX_CACHING:-1}" = "1" ]; then
  ARGS+=(--enable-prefix-caching)
fi

# 官方纯文本模式：跳过视觉编码器与多模态 profiling，省显存给 KV cache
# （Qwen3.6-27B 是多模态模型，config 里有 language_model_only 字段，vLLM 有对应参数）
if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then
  ARGS+=(--language-model-only)
  echo "[INFO] 启用 --language-model-only（跳过视觉编码器）"
fi

# 若目录里残留多模态处理器文件但 config 是纯文本，vLLM 会报
#   TypeError: Expected Qwen3_5Config, but found Qwen3_5TextConfig
# 检测并给出提示（不自动改，避免误删）
if [ -f "$MODEL/config.json" ]; then
  if [ -f "$MODEL/processor_config.json" ] && ! grep -q '"vision_config"' "$MODEL/config.json" 2>/dev/null; then
    echo ""
    echo "[WARN] 检测到: 存在 processor_config.json，但 config.json 无 vision_config"
    echo "       vLLM 可能按多模态初始化并报 config 类型错误。两种处理："
    echo "       ① 纯文本部署: python scripts/09_make_text_only.py $MODEL --apply"
    echo "       ② 完整多模态: python scripts/08_fix_mm_wrapper.py $MODEL --orig <base模型>"
    echo ""
  fi
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
echo "★ 聊天网页（显示 prefill / decode 速度）:"
echo "  bash scripts/serve_chat.sh           # 另开一个终端，默认 :8080"
echo "  然后浏览器打开 http://<服务器IP>:8080/"
echo "  （服务已带 --allowed-origins ${ALLOWED_ORIGINS:-*}，跨域已放行）"
echo ""

exec vllm "${ARGS[@]}"
