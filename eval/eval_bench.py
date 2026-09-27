#!/usr/bin/env python
"""基准评测 —— 昇腾 NPU 版（保证离线也能出数）。

三层评测，逐级降级，保证「一键跑通一定有结果」：
  A. 语言建模准确率（next-token top-1 命中率）—— 用 test.parquet，零依赖
  B. 内置小型选择题基准 Mini-Bench —— 内置题目，零下载
  C. 标准四基准 lm-evaluation-harness（MMLU/GSM8K/TruthfulQA/HellaSwag）
     —— 若已安装且能下载数据集则执行（--limit 控制规模）

用法（项目根）:
    python eval/eval_bench.py                       # A + B（+ C 若可用）
    python eval/eval_bench.py --tasks mmlu,gsm8k --limit 50
    python eval/eval_bench.py --skip-lm-eval
"""
import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import device as dev
from utils.model_utils import load_model

# ---------------- B. 内置 Mini-Bench（30 题，四选一） ----------------
MINI_BENCH = [
    ("常识", "太阳从哪个方向升起？", ["东方", "西方", "南方", "北方"], 0),
    ("常识", "一年有多少个月？", ["10", "11", "12", "13"], 2),
    ("常识", "水的化学式是什么？", ["CO2", "H2O", "O2", "NaCl"], 1),
    ("常识", "中国的首都是？", ["上海", "北京", "广州", "深圳"], 1),
    ("常识", "一周有几天？", ["5", "6", "7", "8"], 2),
    ("数学", "3 + 5 = ?", ["6", "7", "8", "9"], 2),
    ("数学", "12 × 4 = ?", ["36", "44", "48", "52"], 2),
    ("数学", "100 - 37 = ?", ["53", "63", "67", "73"], 1),
    ("数学", "一个正方形有几条边？", ["3", "4", "5", "6"], 1),
    ("数学", "9 的平方是多少？", ["18", "72", "81", "99"], 2),
    ("科学", "水的沸点在标准大气压下是多少摄氏度？", ["50", "80", "100", "120"], 2),
    ("科学", "地球绕着什么转？", ["月亮", "太阳", "火星", "木星"], 1),
    ("科学", "人体最大的器官是？", ["心脏", "肝脏", "皮肤", "肺"], 2),
    ("科学", "光在真空中的速度约为？", ["3万公里/秒", "30万公里/秒", "300万公里/秒", "3亿公里/秒"], 1),
    ("科学", "DNA 的全称是？", ["核糖核酸", "脱氧核糖核酸", "氨基酸", "蛋白质"], 1),
    ("语言", "下列哪个词是名词？", ["快速", "奔跑", "书本", "美丽"], 2),
    ("语言", "“large” 的反义词是？", ["big", "huge", "small", "tall"], 2),
    ("语言", "“他昨天去了图书馆”这句话的时态是？", ["现在时", "过去时", "将来时", "进行时"], 1),
    ("语言", "下面哪句是疑问句？", ["今天天晴。", "你去哪儿？", "真美啊！", "请坐下。"], 1),
    ("语言", "“三心二意”的意思最接近？", ["专心致志", "犹豫不决", "一心一意", "坚定不移"], 1),
    ("推理", "所有的猫都喜欢鱼。咪咪是一只猫。那么咪咪？", ["喜欢鱼", "不喜欢鱼", "无法判断", "喜欢狗"], 0),
    ("推理", "如果 A 比 B 高，B 比 C 高，那么？", ["C 比 A 高", "A 比 C 高", "A 和 C 一样高", "无法判断"], 1),
    ("推理", "1, 2, 4, 8, ? 下一个数是？", ["10", "12", "16", "20"], 2),
    ("推理", "2, 3, 5, 7, 11, ? 下一个数是？", ["12", "13", "14", "15"], 1),
    ("推理", "某商品原价 100 元，打八折后是多少？", ["20 元", "80 元", "90 元", "120 元"], 1),
    ("计算机", "CPU 的中文含义是？", ["中央处理器", "内存", "硬盘", "显卡"], 0),
    ("计算机", "HTTP 默认使用的端口是？", ["21", "22", "80", "443"], 2),
    ("计算机", "在 Python 中，len([1,2,3]) 的结果是？", ["2", "3", "4", "报错"], 1),
    ("计算机", "1 GB 等于多少 MB？", ["100", "512", "1000", "1024"], 3),
    ("计算机", "下列哪项是关系型数据库？", ["Redis", "MongoDB", "MySQL", "Elasticsearch"], 2),
]


@torch.no_grad()
def next_token_accuracy(model, tok, texts, seq=512, prefer_device="auto", max_chunks=64):
    """语言建模准确率：模型预测的下一个 token 是否命中真实 token。"""
    ids = []
    for t in texts:
        ids += tok(t, add_special_tokens=False)["input_ids"]
    if len(ids) < seq:
        return None
    chunks = [ids[s:s + seq] for s in range(0, len(ids) - seq + 1, seq)][:max_chunks]
    target = dev.get_device(prefer_device)
    hit, tot = 0, 0
    for c in chunks:
        b = torch.tensor([c], dtype=torch.long, device=target)
        logits = model(b).logits[:, :-1]
        pred = logits.argmax(dim=-1)
        lab = b[:, 1:]
        hit += int((pred == lab).sum().item())
        tot += lab.numel()
    return hit / max(tot, 1), tot


@torch.no_grad()
def _choice_logprob(model, tok, prompt, choice, prefer_device):
    """计算 choice 作为 prompt 续写的平均对数概率（长度归一化）。"""
    target = dev.get_device(prefer_device)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(choice, add_special_tokens=False)["input_ids"]
    if not c_ids:
        return -1e9
    ids = p_ids + c_ids
    b = torch.tensor([ids], dtype=torch.long, device=target)
    logits = model(b).logits[0]
    logp = F.log_softmax(logits.float(), dim=-1)
    start = len(p_ids) - 1
    total = 0.0
    for k, cid in enumerate(c_ids):
        total += float(logp[start + k, cid].item())
    return total / len(c_ids)


@torch.no_grad()
def mini_bench(model, tok, prefer_device="auto"):
    """内置选择题基准，按类别统计准确率。"""
    per_cat = {}
    details = []
    for cat, q, choices, ans in MINI_BENCH:
        scores = [_choice_logprob(model, tok, q + "\n答案：", c, prefer_device) for c in choices]
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        ok = (pred == ans)
        per_cat.setdefault(cat, []).append(1 if ok else 0)
        details.append({"category": cat, "question": q, "pred": choices[pred],
                        "answer": choices[ans], "correct": ok})
    cat_acc = {k: round(sum(v) / len(v), 4) for k, v in per_cat.items()}
    overall = round(sum(sum(v) for v in per_cat.values()) / len(MINI_BENCH), 4)
    return {"overall": overall, "by_category": cat_acc, "details": details}


def try_lm_eval(model_path, tasks, limit, prefer_device):
    """可选：调用 lm-evaluation-harness。失败返回 None。"""
    try:
        import lm_eval  # noqa
    except Exception:
        print("    [SKIP] 未安装 lm-evaluation-harness")
        return None
    try:
        from lm_eval import simple_evaluate
        device = "npu" if dev.has_npu() else ("cuda" if dev.has_cuda() else "cpu")
        r = simple_evaluate(model="hf", model_args=f"pretrained={model_path},dtype=float16,trust_remote_code=True",
                            tasks=tasks, num_fewshot=0, batch_size=1, limit=limit, device=device)
        out = {}
        for t in tasks:
            v = r.get("results", {}).get(t, {})
            out[t] = v.get("acc,none") or v.get("acc_norm,none") or v.get("exact_match,none")
        return out
    except Exception as e:
        print(f"    [FAIL] lm-eval 执行失败: {type(e).__name__}: {str(e)[:200]}")
        return None


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
    ap.add_argument("--tasks", default="mmlu,gsm8k,truthfulqa_mc2,hellaswag")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-lm-eval", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    from datasets import Dataset
    from transformers import AutoTokenizer

    pq = os.path.join(config.TEST_DIR, "test.parquet")
    if not os.path.isfile(pq):
        raise SystemExit("缺少测试集，请先运行: python scripts/04_prepare_data.py")
    full = Dataset.from_parquet(pq)
    texts = [r["text"] for r in full.select(range(min(32, len(full))))]

    tok = AutoTokenizer.from_pretrained(config.require_model(), trust_remote_code=True)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print("=" * 72)
    print(f"基准评测   设备={dev.default_device(args.device)}")
    print("=" * 72)

    results = {}
    for label, path in list_targets(args.model):
        print(f"\n>>> {label}\n    {path}")
        entry = {}
        try:
            model = load_model(path, prefer_device=args.device)
            acc, tot = next_token_accuracy(model, tok, texts, args.seq, args.device)
            entry["next_token_acc"] = round(acc, 4) if acc is not None else None
            entry["next_token_tokens"] = tot
            print(f"    [A] 语言建模准确率 = {entry['next_token_acc']}  ({tot} tokens)")

            mb = mini_bench(model, tok, args.device)
            entry["mini_bench_overall"] = mb["overall"]
            entry["mini_bench_by_category"] = mb["by_category"]
            entry["mini_bench_details"] = mb["details"]
            print(f"    [B] Mini-Bench 准确率 = {mb['overall']:.4f}   分类: {mb['by_category']}")
            del model
            dev.empty_cache(args.device)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"    [FAIL] {entry['error']}")

        if not args.skip_lm_eval and "error" not in entry:
            le = try_lm_eval(path, tasks, args.limit, args.device)
            if le:
                entry["lm_eval"] = le
                print(f"    [C] lm-eval: {le}")

        results[label] = entry

    out = os.path.join(config.RESULTS_DIR, "bench.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 汇总表
    base = results.get("base", {})
    print("\n" + "-" * 78)
    print(f"{'model':<40}{'next-tok acc':>14}{'Mini-Bench':>14}{'Δ Mini-Bench':>14}")
    print("-" * 78)
    for label, e in results.items():
        a = e.get("next_token_acc")
        m = e.get("mini_bench_overall")
        d = "-"
        if m is not None and base.get("mini_bench_overall"):
            d = f"{(m - base['mini_bench_overall']) * 100:+.2f} pt"
        print(f"{label:<40}{(f'{a:.4f}' if a is not None else 'N/A'):>14}"
              f"{(f'{m:.4f}' if m is not None else 'N/A'):>14}{d:>14}")
    print("-" * 78)
    print(f"已保存 -> {out}")


if __name__ == "__main__":
    main()
