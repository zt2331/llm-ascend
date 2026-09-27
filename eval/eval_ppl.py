#!/usr/bin/env python
"""PPL 困惑度评测 —— 昇腾 NPU 版（双后端：transformers / vLLM）。

口径：PPL = exp(-平均对数概率)，左移对齐（logits[:, :-1] 对 labels[:, 1:]）。

★ 为什么提供 vLLM 后端：
   27B 模型 fp16 权重约 54GB，加上压缩张量反量化的临时开销 > 61GB 显存 →
   transformers 加载会 NPU OOM。
   vLLM 直接以量化精度（如 int8 ≈ 27GB）加载，**不做全量反量化**，
   显存占用大幅下降，并用 prompt_logprobs 直接取每个 token 的对数概率。

用法（项目根）:
    python eval/eval_ppl.py                       # 自动选后端（量化模型走 vLLM）
    python eval/eval_ppl.py --backend vllm        # 强制 vLLM
    python eval/eval_ppl.py --backend hf          # 强制 transformers
    python eval/eval_ppl.py --model <目录> --seq 512
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
from utils.dataio import Dataset
from utils.model_utils import load_model
from utils.quant_check import inspect_quant_dir, format_report


def detect_quant(path):
    """从 config.json 读量化方式。"""
    fp = os.path.join(path, "config.json")
    if os.path.isfile(fp):
        try:
            with open(fp, encoding="utf-8") as f:
                q = json.load(f).get("quantization_config")
            if q:
                return q.get("quant_method") or "compressed-tensors"
        except Exception:
            pass
    return None


def list_targets(explicit=None):
    if explicit:
        p = explicit.rstrip("/")
        return [(os.path.basename(p), p)]
    items = []
    if config.MODEL_PATH:
        items.append(("base", config.MODEL_PATH))
    for d in (config.QUANT_DIR, config.PRUNE_DIR, config.DISTILL_DIR):
        for p in sorted(glob.glob(os.path.join(d, "*"))):
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
                items.append((os.path.relpath(p, config.OUT), p))
    return items


# ----------------------------------------------------------------------
# 后端 A：transformers（fp16 模型可用；量化模型可能 OOM）
# ----------------------------------------------------------------------
@torch.no_grad()
def ppl_hf(model_path, tok, texts, seq, stride, prefer_device):
    model = load_model(model_path, prefer_device=prefer_device, need_logits=True)
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
    del model
    dev.empty_cache(prefer_device)
    return math.exp(total_nll / max(total_tok, 1)), total_tok


# ----------------------------------------------------------------------
# 后端 B：vLLM（量化模型推荐；保持量化精度，显存占用低）
# ----------------------------------------------------------------------
def ppl_vllm(model_path, tok, texts, seq, quant, gpu_util, prefer_device):
    from vllm import LLM, SamplingParams

    kw = {"quantization": quant} if quant else {}
    llm = LLM(model=model_path, dtype="float16", trust_remote_code=True,
              max_model_len=max(seq, 512), gpu_memory_utilization=gpu_util,
              disable_log_stats=True, **kw)
    # prompt_logprobs=0 → 返回每个 prompt 位置真实 token 的对数概率
    sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0)

    total_logp, n_tok = 0.0, 0
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"][:seq]
        if len(ids) < 16:
            continue
        try:
            out = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)
        except Exception as e:
            print(f"      [WARN] 单条样本失败: {str(e)[:80]}")
            continue
        plps = getattr(out[0], "prompt_logprobs", None)
        if not plps:
            continue
        for pos, d in enumerate(plps):
            if not d or pos >= len(ids):
                continue
            tid = ids[pos]
            lp = d[tid].logprob if tid in d else next(iter(d.values())).logprob
            total_logp += float(lp)
            n_tok += 1

    del llm
    dev.empty_cache(prefer_device)
    if n_tok == 0:
        return None, 0
    return math.exp(-total_logp / n_tok), n_tok


def run_one(path, tok, texts, args, prefer_device):
    quant = detect_quant(path)
    # ★ 加载前先校验量化产物；坏了直接报原因，避免拿到 nan 才猜
    if quant:
        info = inspect_quant_dir(path)
        if not info["ok"]:
            print(format_report(info, title="量化产物校验未通过"))
            raise RuntimeError(
                "量化产物无效（详见上方报告）；请勿继续评测，先按提示修复")
        else:
            print(f"    [校验] 量化产物 OK"
                  f"（量化标记张量 {info['n_quant_markers']} 个，"
                  f"ignore={info['ignore']}）")
    backend = args.backend
    if backend == "auto":
        backend = "vllm" if quant else "hf"
    print(f"    后端={backend}  量化={quant or '无'}")

    if backend == "vllm":
        try:
            return ppl_vllm(path, tok, texts, args.seq, quant, args.gpu_util, prefer_device)
        except Exception as e:
            print(f"    [WARN] vLLM 失败({type(e).__name__}: {str(e)[:120]})，回退 transformers")
    return ppl_hf(path, tok, texts, args.seq, args.stride, prefer_device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--backend", default="auto", choices=["auto", "hf", "vllm"])
    ap.add_argument("--seq", type=int, default=512, help="评测序列长度（27B 建议 ≤512）")
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--max-texts", type=int, default=64)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    config.ensure_dirs()
    from transformers import AutoTokenizer

    pq = os.path.join(config.TEST_DIR, "test.parquet")
    if not os.path.isfile(pq):
        raise SystemExit("缺少测试集，请先运行: python scripts/04_prepare_data.py")
    full = Dataset.from_parquet(pq)
    texts = [r["text"] for r in full.select(range(min(args.max_texts, len(full))))]

    tok = AutoTokenizer.from_pretrained(config.require_model(), trust_remote_code=True)

    print("=" * 74)
    print(f"PPL 评测   设备={dev.default_device(args.device)}   seq={args.seq}  "
          f"backend={args.backend}")
    print("=" * 74)

    results = []
    for label, path in list_targets(args.model):
        print(f"\n>>> {label}\n    {path}")
        try:
            ppl, ntok = run_one(path, tok, texts, args, args.device)
            if ppl is None:
                print("    [SKIP] 无法计算（样本过短或后端失败）")
                results.append({"model": label, "path": path, "ppl": None, "tokens": ntok})
            else:
                print(f"    PPL = {ppl:.4f}   (tokens={ntok})")
                results.append({"model": label, "path": path,
                                "ppl": round(ppl, 4), "tokens": ntok})
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"    [FAIL] {msg}")
            if "out of memory" in str(e).lower():
                print("    → 显存不足。建议: --backend vllm 或调小 --seq")
            results.append({"model": label, "path": path, "ppl": None, "error": msg})

    out = os.path.join(config.RESULTS_DIR, "ppl.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    base_ppl = next((r["ppl"] for r in results if r["model"] == "base" and r["ppl"]), None)
    print("\n" + "-" * 74)
    print(f"{'model':<46}{'PPL':>12}{'Δ vs base':>14}")
    print("-" * 74)
    for r in results:
        p = r["ppl"]
        if p is None:
            print(f"{r['model']:<46}{'N/A':>12}{'-':>14}")
        else:
            delta = f"{(p - base_ppl) / base_ppl * 100:+.2f}%" if base_ppl else "-"
            print(f"{r['model']:<46}{p:>12.4f}{delta:>14}")
    print("-" * 74)
    print(f"已保存 -> {out}")


if __name__ == "__main__":
    main()
