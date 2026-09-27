#!/usr/bin/env python
"""调库量化（llm-compressor / compressed-tensors）—— 昇腾 NPU 版。

产物格式 compressed-tensors，vLLM(vllm-ascend) / vLLM(CUDA) 均可尝试加载。
昇腾侧推荐 **W8A8(INT8)**（硬件支持最成熟）。

用法（项目根）:
    python quantize/gen_llmcomp.py --scheme W8A8                    # INT8 W8A8（推荐）
    python quantize/gen_llmcomp.py --scheme W4A16                   # 4bit 权重
    python quantize/gen_llmcomp.py --method gptq --bits 4           # GPTQ
    python quantize/gen_llmcomp.py --model output_models/pruned/xxx # 对指定模型量化
"""
import argparse
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import device as dev
from utils.model_utils import quant_ignore_patterns


def build_calib_loader(tok, calib_rows, seq):
    from torch.utils.data import DataLoader

    class _DS(torch.utils.data.Dataset):
        def __init__(self):
            self.enc = [tok(r["text"], truncation=True, max_length=seq)["input_ids"]
                        for r in calib_rows]

        def __len__(self):
            return len(self.enc)

        def __getitem__(self, i):
            return {"input_ids": torch.tensor(self.enc[i], dtype=torch.long)}

    return DataLoader(_DS(), batch_size=1, shuffle=False)


def make_recipe(method, scheme, group_size, ignore):
    ign = json.dumps(ignore)
    if method == "gptq":
        return ("quant_stage:\n"
                "  quant_modifiers:\n"
                "    GPTQModifier:\n"
                "      targets: ['Linear']\n"
                f"      ignore: {ign}\n"
                f"      scheme: {scheme}\n"
                f"      group_size: {group_size}\n")
    return ("quant_stage:\n"
            "  quant_modifiers:\n"
            "    QuantizationModifier:\n"
            "      targets: ['Linear']\n"
            f"      ignore: {ign}\n"
            f"      scheme: {scheme}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="待量化模型目录（默认 base）")
    ap.add_argument("--method", default="rtn", choices=["rtn", "auxt", "gptq", "awq"])
    ap.add_argument("--scheme", default="W8A8", choices=["W8A8", "W8A16", "W4A16"])
    ap.add_argument("--bits", type=int, default=8, help="仅 gptq 生效")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config.ensure_dirs()
    model_path = os.path.abspath(args.model) if args.model else config.require_model()

    if args.method == "gptq":
        scheme = "W4A16" if args.bits == 4 else "W8A16"
    else:
        scheme = args.scheme

    stem = os.path.basename(os.path.normpath(model_path))
    out_dir = args.out or os.path.join(config.QUANT_DIR, f"{stem}-{args.method}-{scheme}")

    print("=" * 72)
    print(f"llm-compressor 量化   method={args.method}  scheme={scheme}")
    print(f"模型: {model_path}")
    print(f"设备: {dev.default_device('auto')}（llm-compressor 会自行放置模型）")
    print(f"输出: {out_dir}")
    print("=" * 72)

    try:
        from llmcompressor import oneshot
    except Exception as e:
        print(f"[FAIL] llmcompressor 不可用: {e}")
        print("       → 昇腾上建议改用: python quantize/ascend_quant.py --scheme W8A8")
        return 3

    from datasets import Dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    pq = os.path.join(config.CALIB_DIR, "validation.parquet")
    full = Dataset.from_parquet(pq)
    calib_rows = [r for r in full.select(range(min(args.calib, len(full))))]

    ignore = quant_ignore_patterns(model_path)
    recipe = make_recipe(args.method if args.method in ("gptq",) else "rtn",
                         scheme, args.group_size, ignore)
    print(f"\nignore = {ignore}")
    print(f"recipe:\n{recipe}")

    os.makedirs(out_dir, exist_ok=True)
    loader = build_calib_loader(tok, calib_rows, args.seq)
    try:
        oneshot(model=model_path, output_dir=out_dir, recipe=recipe, dataset=loader,
                precision="auto", trust_remote_code_model=True, save_compressed=True)
    except Exception as e:
        print(f"[FAIL] 量化失败: {type(e).__name__}: {str(e)[:400]}")
        print("       → 可尝试: python quantize/ascend_quant.py --scheme W8A8（昇腾原生路径）")
        return 4

    # 补 tokenizer / processor 文件
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
               "chat_template.jinja", "processor_config.json", "preprocessor_config.json"):
        src = os.path.join(model_path, fn)
        dst = os.path.join(out_dir, fn)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy(src, dst)

    # 产物自检
    cfg_p = os.path.join(out_dir, "config.json")
    if os.path.isfile(cfg_p):
        cfg = json.load(open(cfg_p, encoding="utf-8"))
        qc = cfg.get("quantization_config")
        print(f"\n[OK] 产物 config.quantization_config = {json.dumps(qc, ensure_ascii=False)[:200]}")
        print(f"[OK] architectures = {cfg.get('architectures')}")
    print(f"[OK] 已写出 -> {out_dir}")
    print(f"\n部署: bash scripts/serve_ascend.sh {out_dir} compressed-tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
