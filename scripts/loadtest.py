#!/usr/bin/env python
"""并发压测 —— 昇腾 NPU Serving 版。

扫描不同并发度，输出吞吐 / 平均延迟 / P50 / P95 / 错误率 / TTFT，
并保存 results/loadtest.json（报告脚本会自动画「吞吐-延迟曲线」）。

用法（先起服务）:
    bash scripts/serve_ascend.sh <模型目录> compressed-tensors 8000 &
    python scripts/loadtest.py --port 8000 --concurrency 1,2,4,8,16 --n-req 64

只看单并发:
    python scripts/loadtest.py --port 8000 --concurrency 8
"""
import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def one_req(base, model, prompt, max_tokens, timeout=180):
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(
        f"{base}/completions", data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    t0 = time.time()
    ttft = None
    ntok = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                if ttft is None:
                    ttft = time.time() - t0
                ntok += 1
        return {"ok": True, "total": time.time() - t0, "ttft": ttft, "tok": ntok}
    except Exception as e:
        return {"ok": False, "err": str(e)[:100], "total": time.time() - t0,
                "ttft": None, "tok": 0}


def get_served_model(base):
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=15) as r:
            data = json.loads(r.read().decode())
        return data["data"][0]["id"]
    except Exception as e:
        print(f"[WARN] 无法获取模型名: {e}")
        return None


def pct(arr, p):
    if not arr:
        return float("nan")
    arr = sorted(arr)
    return arr[min(len(arr) - 1, int(len(arr) * p))]


def run_once(base, model, prompt, concurrency, n_req, max_tokens):
    print(f"  并发={concurrency:<3} 请求={n_req:<4}", end="", flush=True)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(one_req, base, model, prompt, max_tokens) for _ in range(n_req)]
        results = [f.result() for f in concurrent.futures.as_completed(futs)]
    wall = time.time() - t0

    ok = [r for r in results if r["ok"]]
    err = [r for r in results if not r["ok"]]
    totals = [r["total"] for r in ok]
    tttfs = [r["ttft"] for r in ok if r.get("ttft") is not None]
    out_tokens = sum(r["tok"] for r in ok)

    row = {
        "concurrency": concurrency,
        "n_req": n_req,
        "success": len(ok),
        "failed": len(err),
        "error_rate": round(len(err) / max(len(results), 1), 4),
        "qps": round(len(ok) / max(wall, 1e-6), 2),
        "output_tok_s": round(out_tokens / max(wall, 1e-6), 1),
        "latency_avg_s": round(statistics.mean(totals), 3) if totals else None,
        "latency_p50_s": round(pct(totals, 0.5), 3) if totals else None,
        "latency_p95_s": round(pct(totals, 0.95), 3) if totals else None,
        "ttft_avg_s": round(statistics.mean(tttfs), 3) if tttfs else None,
        "ttft_p95_s": round(pct(tttfs, 0.95), 3) if tttfs else None,
    }
    print(f"→ QPS={row['qps']:<7} 输出={row['output_tok_s']:<8} "
          f"P50={row['latency_p50_s']}s P95={row['latency_p95_s']}s "
          f"错误率={row['error_rate']:.2%}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=None, help="served-model-name，不填自动探测")
    ap.add_argument("--concurrency", default="1,2,4,8", help="逗号分隔，如 1,2,4,8,16")
    ap.add_argument("--n-req", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt", default="请用三句话介绍昇腾 NPU 的推理优化要点。")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}/v1"
    model = args.model or get_served_model(base)
    if not model:
        print("[FAIL] 无法确定模型名，请用 --model 指定（served-model-name）")
        return 1

    concur = [int(c) for c in str(args.concurrency).split(",") if c.strip()]
    print("=" * 72)
    print(f"并发压测  endpoint={base}  model={model}")
    print(f"并发序列={concur}  每档请求数={args.n_req}  输出上限={args.max_tokens}")
    print("=" * 72)

    rows = []
    for c in concur:
        try:
            rows.append(run_once(base, model, args.prompt, c, args.n_req, args.max_tokens))
        except Exception as e:
            print(f" [FAIL] {type(e).__name__}: {e}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = os.path.join(config.RESULTS_DIR, "loadtest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"endpoint": base, "model": model, "max_tokens": args.max_tokens,
                   "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"\n已保存 -> {out}")

    if rows:
        print("\n" + "-" * 88)
        print(f"{'并发':>6}{'QPS':>10}{'输出tok/s':>12}{'P50(s)':>10}{'P95(s)':>10}"
              f"{'TTFT(s)':>10}{'错误率':>10}")
        print("-" * 88)
        for r in rows:
            print(f"{r['concurrency']:>6}{r['qps']:>10}{r['output_tok_s']:>12}"
                  f"{str(r['latency_p50_s']):>10}{str(r['latency_p95_s']):>10}"
                  f"{str(r['ttft_avg_s']):>10}{r['error_rate']:>10.2%}")
        print("-" * 88)
    print("\n重新生成报告以纳入压测曲线: python report/make_report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
