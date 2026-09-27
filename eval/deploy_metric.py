#!/usr/bin/env python
"""部署指标评测 —— 昇腾 NPU (vllm-ascend) 版。

测量：decode 吞吐(tok/s)、TTFT(近似)、显存占用。
vLLM 在昇腾上通过 vllm-ascend 插件运行，设备为 npu。

用法（项目根）:
    python eval/deploy_metric.py                    # 测 base + 所有产物
    python eval/deploy_metric.py --model <目录> --quant compressed-tensors
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import device as dev


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


def run_one(path, quant, prefer_device="auto"):
    """用 vLLM 加载并测量 decode 吞吐 / TTFT / 显存。"""
    from vllm import LLM, SamplingParams

    kw = {}
    if quant:
        kw["quantization"] = quant
    common = dict(model=path, trust_remote_code=True, dtype="float16",
                  gpu_memory_utilization=0.90, max_model_len=2048,
                  enforce_eager=False)
    common.update(kw)

    t0 = time.time()
    llm = LLM(**common)
    load_s = time.time() - t0

    sp = SamplingParams(max_tokens=64, temperature=0)
    prompts = ["请用一句话介绍人工智能。", "介绍一下昇腾 NPU 的特点。"]
    llm.generate(["预热"], SamplingParams(max_tokens=4, temperature=0))   # 预热

    ts = []
    for _ in range(3):
        dev.synchronize(prefer_device)
        t = time.time()
        outs = llm.generate(prompts, sp)
        dev.synchronize(prefer_device)
        dt = time.time() - t
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        ts.append(ntok / max(dt, 1e-6))
    decode_tps = sum(ts) / len(ts)

    # TTFT 近似：单请求短输出
    t = time.time()
    llm.generate([prompts[0]], SamplingParams(max_tokens=1, temperature=0))
    dev.synchronize(prefer_device)
    ttft = time.time() - t

    # 采样文本样例（验证输出不是乱码）
    sample = outs[0].outputs[0].text[:60].replace("\n", " ")

    return {"decode_tok_s": round(decode_tps, 1),
            "ttft_s": round(ttft, 3),
            "load_s": round(load_s, 1),
            "mem_allocated_gb": dev.memory_allocated_gb(prefer_device),
            "sample": sample}


def list_targets(explicit=None):
    items = []
    if config.MODEL_PATH:
        items.append(("base", config.MODEL_PATH))
    for d in (config.QUANT_DIR, config.PRUNE_DIR, config.DISTILL_DIR):
        for p in sorted(glob.glob(os.path.join(d, "*"))):
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
                items.append((os.path.relpath(p, config.OUT), p))
    if explicit:
        items = [(explicit, explicit)]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    config.ensure_dirs()
    print("=" * 72)
    print(f"部署指标评测  backend={dev.default_device(args.device)}  "
          f"device={dev.device_name(args.device)}")
    print("=" * 72)

    results = []
    for label, path in list_targets(args.model):
        q = args.quant if args.model else detect_quant(path)
        print(f"\n>>> {label}  (quant={q})")
        try:
            r = run_one(path, q, args.device)
            r.update({"model": label, "path": path, "quant": q})
            results.append(r)
            print(f"    decode = {r['decode_tok_s']} tok/s | TTFT = {r['ttft_s']} s "
                  f"| load = {r['load_s']} s")
            print(f"    样例输出: {r['sample']}")
        except Exception as e:
            print(f"    [FAIL] {type(e).__name__}: {str(e)[:200]}")
            results.append({"model": label, "path": path, "quant": q,
                            "error": f"{type(e).__name__}: {str(e)[:200]}"})

    out = os.path.join(config.RESULTS_DIR, "deploy_metric.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "-" * 72)
    print(f"{'model':<40}{'decode tok/s':>14}{'TTFT(s)':>12}")
    print("-" * 72)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<40}{'FAIL':>14}{'-':>12}")
        else:
            print(f"{r['model']:<40}{r['decode_tok_s']:>14}{r['ttft_s']:>12}")
    print("-" * 72)
    print(f"已保存 -> {out}")


if __name__ == "__main__":
    main()
