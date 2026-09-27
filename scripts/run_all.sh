#!/usr/bin/env bash
# =============================================================================
# run_all.sh —— 昇腾 NPU 一键全流程
#   环境自检 → 找模型 → 备数据 → 剪枝 → 蒸馏 → 量化 → 评测 → 生成报告(表+图)
#
# 用法:
#   bash scripts/run_all.sh                       # 跑全部
#   KEEP=48 CALIB=32 SEQ=512 bash scripts/run_all.sh
#   STAGES=eval,report bash scripts/run_all.sh    # 只跑指定阶段（逗号分隔）
#   SKIP_DISTILL=1 bash scripts/run_all.sh        # 跳过蒸馏（省时间）
#   ALLOW_NO_NPU=1 bash scripts/run_all.sh        # 无 NPU 也继续（用 CPU，很慢）
#
# 可选环境变量:
#   KEEP(保留层数,默认48) CALIB(校准条数,32) SEQ(序列长度,512)
#   QUANT_SCHEME(W8A8/W4A16,默认W8A8)  EVAL_SEQ(PPL用,1024)
# =============================================================================
set -u
cd "$(dirname "$0")/.."
PROJ="$(pwd)"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

KEEP="${KEEP:-48}"
CALIB="${CALIB:-32}"
SEQ="${SEQ:-512}"
QUANT_SCHEME="${QUANT_SCHEME:-W8A8}"
EVAL_SEQ="${EVAL_SEQ:-1024}"
SKIP_DISTILL="${SKIP_DISTILL:-0}"
ALLOW_NO_NPU="${ALLOW_NO_NPU:-0}"
STAGES="${STAGES:-check,model,data,prune,distill,quantize,eval,report}"
LOG_DIR="$PROJ/output_models/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/run_$TS.log"

PY="${PY:-python}"
command -v "$PY" >/dev/null 2>&1 || { echo "[ERROR] 找不到 python"; exit 1; }

# ---------- 工具函数 ----------
declare -a FAILED=()
declare -a DONE=()

want() { case ",$STAGES," in *",$1,"*) return 0;; *) return 1;; esac; }

run_stage() {
  local name="$1"; shift
  local log="$LOG_DIR/${name}_$TS.log"
  echo ""
  echo "=============================================================="
  echo "== [$name] $*"
  echo "=============================================================="
  if "$@" 2>&1 | tee "$log"; then
    DONE+=("$name")
    return 0
  else
    local rc=${PIPESTATUS[0]}
    echo "[FAIL] 阶段 $name 失败 (exit=$rc)，日志: $log"
    FAILED+=("$name(exit=$rc)")
    return $rc
  fi
}

echo "=============================================================="
echo " 昇腾 NPU 大模型压缩与部署 —— 一键全流程"
echo " 项目: $PROJ"
echo " 时间: $TS"
echo " 阶段: $STAGES"
echo " 参数: KEEP=$KEEP CALIB=$CALIB SEQ=$SEQ SCHEME=$QUANT_SCHEME"
echo "=============================================================="

# ---------- 0. 环境自检 ----------
if want check; then
  if ! run_stage check "$PY" scripts/02_check_env.py; then
    if [ "$ALLOW_NO_NPU" != "1" ]; then
      echo ""
      echo "[STOP] NPU 不可用。请先修复环境，或加 ALLOW_NO_NPU=1 用 CPU 慢跑。"
      exit 2
    fi
    echo "[WARN] NPU 不可用，按 ALLOW_NO_NPU=1 继续（速度会很慢）"
  fi
fi

# ---------- 1. 找模型 ----------
if want model; then
  run_stage model "$PY" scripts/03_find_model.py || true
  MODEL_PATH="$("$PY" -c "import config; print(config.MODEL_PATH or '')" 2>/dev/null || echo "")"
  if [ -z "$MODEL_PATH" ]; then
    echo "[STOP] 未找到模型。请 export MODEL_PATH=/path/to/Qwen3.6-27B 后重试。"
    exit 3
  fi
  export MODEL_PATH
  echo "[OK] MODEL_PATH=$MODEL_PATH"
fi

# ---------- 2. 备数据 ----------
if want data; then
  run_stage data "$PY" scripts/04_prepare_data.py --calib "$CALIB" --test 32
fi

# ---------- 3. 剪枝 ----------
if want prune; then
  run_stage prune "$PY" prune/prune.py --keep "$KEEP" --calib "$CALIB" --seq "$SEQ" || true
fi

# ---------- 4. 蒸馏 ----------
if want distill && [ "$SKIP_DISTILL" != "1" ]; then
  run_stage distill "$PY" distill/distill.py --epochs 1 --max-steps 40 \
      --calib "$CALIB" --seq "$SEQ" || true
fi

# ---------- 5. 量化 ----------
if want quantize; then
  # ★ 量化对象：优先用蒸馏产物，其次剪枝产物，最后才回退 base。
  #   否则剪枝/蒸馏白做，量化出来的还是原始模型。
  QTARGET="$MODEL_PATH"
  _pick() { [ -n "$1" ] && [ -d "$1" ] && QTARGET="$1" || true; }
  _latest() { ls -dt "$1"/* 2>/dev/null | head -1; }
  if [ "${SKIP_DISTILL:-0}" != "1" ]; then
    _pick "$( _latest "$PROJ/output_models/distilled" )"
  fi
  if [ "$QTARGET" = "$MODEL_PATH" ]; then
    _pick "$( _latest "$PROJ/output_models/pruned" )"
  fi
  echo ""
  echo "[量化对象] $QTARGET"
  if [ "$QTARGET" = "$MODEL_PATH" ]; then
    echo "          （未找到剪枝/蒸馏产物，回退到原始模型）"
  fi
  QARGS=(--model "$QTARGET" --calib "$CALIB" --seq "$SEQ")

  # 要跑哪些 llm-compressor 量化：ascend,smooth,awq,gptq,gptq8,rtn
  QMETHODS="${QMETHODS:-smooth,awq,gptq}"
  echo "[llm-compressor 量化方法] $QMETHODS"

  has_m() { case ",$QMETHODS," in *",$1,"*) return 0;; *) return 1;; esac; }

  # 5.1 昇腾原生 W8A8（msModelSlim，需要 msmodelslim）
  if has_m ascend; then
    run_stage quantize_ascend "$PY" quantize/ascend_quant.py \
        --scheme "$QUANT_SCHEME" "${QARGS[@]}" || true
  fi

  # 5.2 SmoothQuant W8A8（激活离群迁移 + INT8，昇腾主力推荐）
  if has_m smooth; then
    run_stage quantize_smooth "$PY" quantize/w8a8_smooth.py "${QARGS[@]}" || true
  fi

  # 5.3 llm-compressor 调库路径：AWQ（激活感知）
  if has_m awq; then
    run_stage quantize_awq "$PY" quantize/gen_llmcomp.py \
        --method awq --bits 4 --group-size 128 "${QARGS[@]}" || true
  fi

  # 5.4 llm-compressor 调库路径：GPTQ（Hessian 二阶 + 逐列误差补偿）
  if has_m gptq; then
    run_stage quantize_gptq "$PY" quantize/gen_llmcomp.py \
        --method gptq --bits 4 --group-size 128 "${QARGS[@]}" || true
  fi
  if has_m gptq8; then
    run_stage quantize_gptq8 "$PY" quantize/gen_llmcomp.py \
        --method gptq --bits 8 --group-size 128 "${QARGS[@]}" || true
  fi

  # 5.5 RTN 基线（最朴素量化，只作对照）
  if has_m rtn; then
    run_stage quantize_rtn "$PY" quantize/gen_llmcomp.py \
        --method rtn --scheme "$QUANT_SCHEME" "${QARGS[@]}" || true
  fi

  # 5.6 从零手写 AWQ / GPTQ（纯 PyTorch 实现，展示算法细节）
  if [ "${SKIP_MANUAL:-0}" != "1" ]; then
    ML="${MANUAL_MAX_LAYERS:-4}"
    echo ""
    echo "[说明] 手写 AWQ/GPTQ 逐列循环较慢，默认只量化前 $ML 层做演示。"
    echo "       全模型量化: MANUAL_MAX_LAYERS=0 bash scripts/run_all.sh"
    run_stage quantize_manual_awq "$PY" quantize/manual_awq.py \
        --bits 4 "${QARGS[@]}" --max-layers "$ML" || true
    run_stage quantize_manual_gptq "$PY" quantize/manual_gptq.py \
        --bits 4 "${QARGS[@]}" --max-layers "$ML" || true
  fi
fi

# ---------- 6. 评测 ----------
if want eval; then
  run_stage eval_ppl "$PY" eval/eval_ppl.py --seq "$EVAL_SEQ" || true
  run_stage eval_bench "$PY" eval/eval_bench.py --seq "$SEQ" || true
  run_stage eval_deploy "$PY" eval/deploy_metric.py || true
fi

# ---------- 7. 报告 ----------
if want report; then
  run_stage report "$PY" report/make_report.py || true
fi

# ---------- 汇总 ----------
echo ""
echo "=============================================================="
echo " 完成汇总"
echo "=============================================================="
echo "成功阶段: ${DONE[*]:-（无）}"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "失败阶段: ${FAILED[*]}"
fi
echo ""
echo "结果目录: $PROJ/output_models/results"
ls -1 "$PROJ/output_models/results" 2>/dev/null | sed 's/^/   - /' || true
echo ""
echo "日志目录: $LOG_DIR"
echo ""
echo "查看报告:  cat output_models/results/REPORT.md"
echo "查看图表:  ls output_models/results/fig_*.png"
echo "启动服务:  bash scripts/serve_ascend.sh <模型目录> ${QUANT_SCHEME}"
echo "=============================================================="
