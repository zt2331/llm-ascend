#!/usr/bin/env python
"""PPL 困惑度评测 —— 昇腾 NPU 版。

口径：PPL = exp(平均交叉熵)，左移对齐（logits[:, :-1] 对 labels[:, 1:]）。
同 tokenizer、同测试集、同 seq_len，保证压缩前后可比。

用法（项目根）:
    python eval/eval_ppl.py                      # 评测 base + 所有产物
    python eval/eval_ppl.py --model <目录>       # 只测一个
    python eval/eval_ppl.py --seq 1024
"""
import argparse
import glob
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import device as dev
from utils.model_utils import load_model


def list_targets(explicit=None):
    """返回 [(标签, 路径)]：base + output_models 下的全部产物。"""
    items = []
    base = config.MODEL_PATH
    if base:
        items.append(("base", base))
    for d in (config.QUANT_DIR, config.PRUNE_DIR, config.DISTILL_DIR):
        for p in sorted(glob.glob(os.path.join(d, "*"))):
            if not os.path.isdir(p):
                continue
            if os.path.isfile(os.path.join(p, "config.json")):
                items.append((os.path.relpath(p, config.OUT), p))
    if explicit:
        items = [(explicit, explicit)]
    return items


@torch.no_grad()
def compute_ppl(model, tok, texts, seq=1024, stride=1024, prefer_device="auto"):
    ids = []
    for t in texts:
        ids += tok(t, add_special_tokens=False)["input_ids"]
    if len(ids) < seq:
        return None, 0
    chunks = [ids[s:s + seq] for s in range(0, len(ids) - seq + 1, stride)]
    target = dev.get_device(prefer_device)
    total_nll, total_tok = 0.0, 0
    for c in chunks:
        b = torch.tensor([c], dtype=torch.long, device=target)
        logits = model(b).logits[:, :-1].contiguous()
        lab = b[:, 1:].contiguous()
        nll = F.cross_entropy(logits.view(-1, logits.shape[-1]).float(),
                              lab.view(-1), reduction="sum")
        total_nll += float(nll.item())
        total_tok += lab.numel()
    return math.exp(total_nll / max(total_tok, 1)), total_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--max-texts", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    config.ensure_dirs()
    from utils.dataio import Dataset
    from transformers import AutoTokenizer

    pq = os.path.join(config.TEST_DIR, "test.parquet")
    if not os.path.isfile(pq):
        raise SystemExit("缺少测试集，请先运行: python scripts/04_prepare_data.py")
    full = Dataset.from_parquet(pq)
    texts = [r["text"] for r in full.select(range(min(args.max_texts, len(full))))]

    tok_path = config.require_model()
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    print("=" * 72)
    print(f"PPL 评测   设备={dev.default_device(args.device)}   seq={args.seq}")
    print("=" * 72)

    results = []
    for label, path in list_targets(args.model):
        print(f"\n>>> {label}\n    {path}")
        try:
            model = load_model(path, prefer_device=args.device)
            ppl, ntok = compute_ppl(model, tok, texts, args.seq, args.stride, args.device)
            del model
            dev.empty_cache(args.device)
            if ppl is None:
                print("    [SKIP] 测试文本太短")
                results.append({"model": label, "path": path, "ppl": None, "tokens": ntok})
            else:
                print(f"    PPL = {ppl:.4f}   (tokens={ntok})")
                results.append({"model": label, "path": path,
                                "ppl": round(ppl, 4), "tokens": ntok})
        except Exception as e:
            print(f"    [FAIL] {type(e).__name__}: {str(e)[:200]}")
            results.append({"model": label, "path": path, "ppl": None,
                            "error": f"{type(e).__name__}: {str(e)[:200]}"})

    out = os.path.join(config.RESULTS_DIR, "ppl.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印表格
    base_ppl = next((r["ppl"] for r in results if r["model"] == "base" and r["ppl"]), None)
    print("\n" + "-" * 72)
    print(f"{'model':<44}{'PPL':>12}{'Δ vs base':>14}")
    print("-" * 72)
    for r in results:
        p = r["ppl"]
        if p is None:
            print(f"{r['model']:<44}{'N/A':>12}{'-':>14}")
        else:
            delta = f"{(p - base_ppl) / base_ppl * 100:+.2f}%" if base_ppl else "-"
            print(f"{r['model']:<44}{p:>12.4f}{delta:>14}")
    print("-" * 72)
    print(f"已保存 -> {out}")


if __name__ == "__main__":
    main()
