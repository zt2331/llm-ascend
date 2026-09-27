#!/usr/bin/env python
"""08_fix_mm_wrapper.py —— 恢复被 llm-compressor 剥离的多模态 wrapper。

核心逻辑在 utils/mm_wrapper.py（量化脚本也会自动调用它）。
本脚本是命令行入口，方便手动对已有产物做修复。

背景
    llm-compressor 保存量化产物时会把多模态模型（Qwen3_5ForConditionalGeneration）
    写成纯文本（Qwen3_5ForCausalLM）：config 丢 vision_config/text_config、
    视觉塔权重丢失，但权重键名仍是 wrapper 命名空间
    → transformers 键名对不上 → weight_scale 加载不上 → PPL = nan
    → vLLM 还会因残留 processor_config.json 报 config 类型错误

用法
    python scripts/08_fix_mm_wrapper.py <量化产物目录> [--orig <原始模型>] [--out <输出>]
    默认 orig = config.MODEL_PATH，out = <量化目录>-mm
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.mm_wrapper import restore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("quant_dir")
    ap.add_argument("--orig", default=None, help="原始多模态模型目录（默认 config.MODEL_PATH）")
    ap.add_argument("--out", default=None, help="输出目录（默认 <量化目录>-mm）")
    ap.add_argument("--scale-dtype", default="float32",
                    choices=["keep", "float32", "bfloat16"],
                    help="把 weight_scale/input_scale 转成该 dtype。昇腾 "
                         "aclnnQuantMatmulWeightNz 只接受 [UINT64,BF16,INT64,FLOAT]，"
                         "fp16 会报 161002，故默认 float32")
    args = ap.parse_args()

    od = os.path.abspath(args.orig) if args.orig else config.require_model()
    out = restore(args.quant_dir, od, args.out, args.scale_dtype)
    print(f"\n使用方式:\n    export MODEL_PATH={out}")
    print(f"    python eval/eval_ppl.py --model {out} --backend vllm --seq 512 --max-num-seqs 8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
