#!/usr/bin/env python
"""W8A8 INT8 量化（SmoothQuant + QuantizationModifier）—— 昇腾 NPU 版。

★ 这是昇腾上**最该主推**的量化方案：达芬奇 Cube 单元对 INT8 有原生支持，
  昇腾生态（msModelSlim / MindIE / vllm-ascend）对 W8A8 支持最成熟。

SmoothQuant 原理：
  激活存在离群大值（个别通道比其它大上百倍），直接对激活做 INT8 会溢出/精度崩。
  做法是把量化难度**从激活迁移到权重**：  Y = (X/s)·(W·s)ᵀ
  其中 s = max|X|^α / max|W|^(1−α)，α 为迁移强度（越大越把难度给权重）。

产物：compressed-tensors（W8A8），可尝试用 `vllm serve --quantization compressed-tensors` 加载。

用法（项目根）:
    python quantize/w8a8_smooth.py
    python quantize/w8a8_smooth.py --alpha 0.7 --calib 64 --seq 2048
    python quantize/w8a8_smooth.py --model output_models/pruned/xxx
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import device as dev
from utils.model_utils import quant_ignore_patterns


def build_calib_loader(tok, rows, seq):
    import torch
    from torch.utils.data import DataLoader

    class _DS(torch.utils.data.Dataset):
        def __init__(self):
            self.enc = [tok(r["text"], truncation=True, max_length=seq)["input_ids"]
                        for r in rows]

        def __len__(self):
            return len(self.enc)

        def __getitem__(self, i):
            return {"input_ids": torch.tensor(self.enc[i], dtype=torch.long)}

    return DataLoader(_DS(), batch_size=1, shuffle=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--alpha", type=float, default=0.75, help="SmoothQuant 迁移强度 0~1")
    ap.add_argument("--scheme", default="W8A8", choices=["W8A8", "W8A16"])
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config.ensure_dirs()
    model_path = os.path.abspath(args.model) if args.model else config.require_model()
    stem = os.path.basename(os.path.normpath(model_path))
    out_dir = args.out or os.path.join(config.QUANT_DIR, f"{stem}-smooth-{args.scheme}")

    print("=" * 72)
    print(f"SmoothQuant {args.scheme}   alpha={args.alpha}")
    print(f"模型: {model_path}")
    print(f"设备: {dev.default_device('auto')}")
    print(f"输出: {out_dir}")
    print("=" * 72)

    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization.quantization import QuantizationModifier
        from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
    except Exception as e:
        print(f"[FAIL] llmcompressor 不可用: {e}")
        print("       → 昇腾上可改用: python quantize/ascend_quant.py --scheme W8A8")
        return 3

    from utils.dataio import Dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    full = Dataset.from_parquet(os.path.join(config.CALIB_DIR, "validation.parquet"))
    rows = [r for r in full.select(range(min(args.calib, len(full))))]

    ignore = quant_ignore_patterns(model_path)
    recipe = [
        SmoothQuantModifier(smoothing_strength=args.alpha),
        QuantizationModifier(targets=["Linear"], scheme=args.scheme, ignore=ignore),
    ]
    print(f"\nignore = {ignore}")
    print(f"recipe = SmoothQuant(alpha={args.alpha}) + QuantizationModifier({args.scheme})")

    os.makedirs(out_dir, exist_ok=True)
    loader = build_calib_loader(tok, rows, args.seq)
    try:
        oneshot(model=model_path, dataset=loader, recipe=recipe,
                num_calibration_samples=min(len(rows), 256),
                max_seq_length=args.seq, output_dir=out_dir,
                save_compressed=True)
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {str(e)[:400]}")
        return 4

    import shutil
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
               "chat_template.jinja", "processor_config.json", "preprocessor_config.json"):
        src = os.path.join(model_path, fn)
        if os.path.isfile(src) and not os.path.exists(os.path.join(out_dir, fn)):
            shutil.copy(src, out_dir)

    cfg_p = os.path.join(out_dir, "config.json")
    if os.path.isfile(cfg_p):
        cfg = json.load(open(cfg_p, encoding="utf-8"))
        print(f"\n[OK] quantization_config = "
              f"{json.dumps(cfg.get('quantization_config'), ensure_ascii=False)[:220]}")
    print(f"[OK] 已写出 -> {out_dir}")
    print(f"\n部署: bash scripts/serve_ascend.sh {out_dir} compressed-tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
