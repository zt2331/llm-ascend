#!/usr/bin/env python
"""报告生成：把评测结果汇总成 **表格 (Markdown/CSV) + 图表 (PNG)**。

输入: output_models/results/{ppl.json, bench.json, deploy_metric.json}
输出: output_models/results/
        REPORT.md            汇总报告（表格）
        summary.csv          汇总表（可用 Excel 打开）
        fig_ppl.png          PPL 对比
        fig_bench.png        准确率对比
        fig_mini_categories.png  Mini-Bench 分类准确率
        fig_throughput.png   decode 吞吐对比
        fig_ttft.png         TTFT 对比
        fig_acc_vs_speed.png 精度-速度权衡散点图

用法: python report/make_report.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")          # 无显示环境
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

RES = config.RESULTS_DIR

# 图表统一用英文标签，避免无中文字体时出现方框
plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _load(name):
    p = os.path.join(RES, name)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _short(s, n=26):
    s = str(s)
    return s if len(s) <= n else s[:n - 3] + "..."


def _bar(labels, values, title, ylabel, fname, color="#4C78A8", fmt="{:.3f}"):
    if not values:
        return None
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels) + 2), 4.2))
    bars = ax.bar(range(len(labels)), values, color=color, width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([_short(l) for l in labels], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = os.path.join(RES, fname)
    fig.savefig(out)
    plt.close(fig)
    print(f"  [图] {out}")
    return out


def _grouped(labels, series: dict, title, ylabel, fname):
    """series = {系列名: [值...]}"""
    if not labels or not series:
        return None
    n = len(series)
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels) + 2), 4.2))
    xs = range(len(labels))
    for i, (name, vals) in enumerate(series.items()):
        pos = [x + (i - (n - 1) / 2) * width for x in xs]
        bars = ax.bar(pos, vals, width=width, label=name)
        for b, v in zip(bars, vals):
            if v is not None:
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([_short(l) for l in labels], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(RES, fname)
    fig.savefig(out)
    plt.close(fig)
    print(f"  [图] {out}")
    return out


def build_summary(ppl, bench, deploy):
    """合并三个结果源 -> {model: {...}}"""
    rows = {}

    def _get(model):
        return rows.setdefault(model, {"model": model})

    for r in (ppl or []):
        _get(r["model"]).update({"ppl": r.get("ppl"), "path": r.get("path")})
    for m, e in (bench or {}).items():
        _get(m).update({
            "next_token_acc": e.get("next_token_acc"),
            "mini_bench": e.get("mini_bench_overall"),
            "mini_categories": e.get("mini_bench_by_category"),
            "lm_eval": e.get("lm_eval"),
            "bench_error": e.get("error"),
        })
    for r in (deploy or []):
        _get(r["model"]).update({
            "decode_tok_s": r.get("decode_tok_s"),
            "ttft_s": r.get("ttft_s"),
            "load_s": r.get("load_s"),
            "quant": r.get("quant"),
            "sample": r.get("sample"),
            "deploy_error": r.get("error"),
        })
    return rows


def write_markdown(rows, meta):
    base = rows.get("base", {})
    lines = []
    lines.append("# 昇腾 NPU 大模型压缩与部署 —— 实验结果报告\n")
    lines.append("> 本报告由 `report/make_report.py` 自动生成，数据来源为项目评测脚本的真实输出。\n")

    lines.append("## 0. 运行环境\n")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    for k, v in meta.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # 1. 主汇总表
    lines.append("## 1. 汇总表（精度 + 性能）\n")
    lines.append("| 模型 | 位宽/方法 | PPL | PPL 退化 | 语言建模准确率 | Mini-Bench | "
                 "Mini-Bench 变化 | decode (tok/s) | TTFT (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m, r in rows.items():
        ppl = r.get("ppl")
        d = "-"
        if ppl and base.get("ppl"):
            d = f"{(ppl - base['ppl']) / base['ppl'] * 100:+.2f}%"
        nb = r.get("next_token_acc")
        mb = r.get("mini_bench")
        mbd = "-"
        if mb is not None and base.get("mini_bench") is not None:
            mbd = f"{(mb - base['mini_bench']) * 100:+.2f} pt"
        lines.append(
            f"| {m} | {r.get('quant') or 'fp16'} "
            f"| {f'{ppl:.4f}' if ppl else 'N/A'} | {d} "
            f"| {f'{nb:.4f}' if nb is not None else 'N/A'} "
            f"| {f'{mb:.4f}' if mb is not None else 'N/A'} | {mbd} "
            f"| {r.get('decode_tok_s') if r.get('decode_tok_s') else 'N/A'} "
            f"| {r.get('ttft_s') if r.get('ttft_s') else 'N/A'} |")
    lines.append("")

    # 2. Mini-Bench 分类
    cats = set()
    for r in rows.values():
        if r.get("mini_categories"):
            cats.update(r["mini_categories"].keys())
    if cats:
        cats = sorted(cats)
        lines.append("## 2. Mini-Bench 分类准确率\n")
        lines.append("| 模型 | " + " | ".join(cats) + " |")
        lines.append("|---" * (len(cats) + 1) + "|")
        for m, r in rows.items():
            mc = r.get("mini_categories") or {}
            vals = " | ".join(f"{mc[c]:.3f}" if c in mc else "-" for c in cats)
            lines.append(f"| {m} | {vals} |")
        lines.append("")

    # 3. 标准四基准（若 lm-eval 可用）
    le_models = {m: r["lm_eval"] for m, r in rows.items() if r.get("lm_eval")}
    if le_models:
        tasks = sorted({t for v in le_models.values() for t in v})
        lines.append("## 3. 标准四基准（lm-evaluation-harness）\n")
        lines.append("| 模型 | " + " | ".join(tasks) + " |")
        lines.append("|---" * (len(tasks) + 1) + "|")
        for m, v in le_models.items():
            lines.append(f"| {m} | " + " | ".join(
                f"{v[t]:.4f}" if v.get(t) is not None else "-" for t in tasks) + " |")
        lines.append("")
    else:
        lines.append("## 3. 标准四基准（lm-evaluation-harness）\n")
        lines.append("未执行（未安装或数据集不可用）。安装后可运行：\n")
        lines.append("```bash\npython eval/eval_bench.py --tasks mmlu,gsm8k,truthfulqa_mc2,hellaswag --limit 50\n```\n")

    # 4. 输出样例（验证正确性）
    samples = {m: r["sample"] for m, r in rows.items() if r.get("sample")}
    if samples:
        lines.append("## 4. 部署输出样例（验证非乱码）\n")
        lines.append("| 模型 | 样例输出（截断） |")
        lines.append("|---|---|")
        for m, s in samples.items():
            lines.append(f"| {m} | `{s}` |")
        lines.append("")

    # 5. 图表
    lines.append("## 5. 图表\n")
    figs = [
        ("fig_ppl.png", "PPL 对比（越低越好）"),
        ("fig_bench.png", "准确率对比"),
        ("fig_mini_categories.png", "Mini-Bench 分类准确率"),
        ("fig_throughput.png", "decode 吞吐对比（越高越好）"),
        ("fig_ttft.png", "TTFT 对比（越低越好）"),
        ("fig_acc_vs_speed.png", "精度-速度权衡"),
    ]
    for fn, cap in figs:
        if os.path.isfile(os.path.join(RES, fn)):
            lines.append(f"### {cap}\n")
            lines.append(f"![{cap}]({fn})\n")

    lines.append("---\n")
    lines.append("> 说明：`base` 为原始模型；其余为剪枝 / 蒸馏 / 量化产物。")
    lines.append("> 若某项为 N/A，说明该阶段未执行或执行失败，请查看终端日志。\n")

    out = os.path.join(RES, "REPORT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [表] {out}")


def write_csv(rows):
    import csv
    out = os.path.join(RES, "summary.csv")
    cols = ["model", "quant", "ppl", "next_token_acc", "mini_bench",
            "decode_tok_s", "ttft_s", "load_s"]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows.values():
            w.writerow(r)
    print(f"  [表] {out}")


def main():
    os.makedirs(RES, exist_ok=True)
    print("=" * 72)
    print("生成报告（表格 + 图表）")
    print("=" * 72)

    ppl = _load("ppl.json")
    bench = _load("bench.json")
    deploy = _load("deploy_metric.json")

    if not any([ppl, bench, deploy]):
        print("[FAIL] 未找到任何结果文件，请先运行评测：")
        print("   python eval/eval_ppl.py && python eval/eval_bench.py && python eval/deploy_metric.py")
        return 1

    rows = build_summary(ppl, bench, deploy)
    labels = list(rows.keys())
    print(f"\n共 {len(labels)} 个模型: {labels}")

    # ------- 环境信息 -------
    meta = {}
    try:
        from utils import device as dev
        d = dev.describe()
        meta = {"后端": d["backend"], "设备": d["device_name"],
                "设备数": d["device_count"], "显存(GB)": d["memory_total_gb"],
                "torch": d["torch"], "torch_npu": d["torch_npu"], "CANN": d["cann"]}
    except Exception as e:
        meta = {"说明": f"设备信息读取失败: {e}"}
    try:
        import config as cfg
        meta["模型"] = os.path.basename(cfg.MODEL_PATH or "?")
    except Exception:
        pass

    # ------- 表格 -------
    print("\n[1/3] 生成表格")
    write_markdown(rows, meta)
    write_csv(rows)

    # ------- 图表 -------
    print("\n[2/3] 生成图表")
    # PPL
    ys = [rows[m].get("ppl") for m in labels]
    if any(v for v in ys):
        xs = [m for m, v in zip(labels, ys) if v]
        vs = [v for v in ys if v]
        _bar(xs, vs, "Perplexity (lower is better)", "PPL", "fig_ppl.png",
             color="#4C78A8", fmt="{:.3f}")

    # 准确率（分组）
    nta = [rows[m].get("next_token_acc") for m in labels]
    mb = [rows[m].get("mini_bench") for m in labels]
    if any(v is not None for v in nta + mb):
        idx = [i for i, (a, b) in enumerate(zip(nta, mb)) if a is not None or b is not None]
        lab2 = [labels[i] for i in idx]
        s1 = [nta[i] if nta[i] is not None else 0 for i in idx]
        s2 = [mb[i] if mb[i] is not None else 0 for i in idx]
        _grouped(lab2, {"Next-token acc": s1, "Mini-Bench acc": s2},
                 "Accuracy comparison (higher is better)", "Accuracy", "fig_bench.png")

    # Mini-Bench 分类
    cats = sorted({c for r in rows.values() for c in (r.get("mini_categories") or {})})
    if cats:
        series = {}
        for m, r in rows.items():
            mc = r.get("mini_categories") or {}
            if mc:
                series[m] = [mc.get(c, 0) for c in cats]
        if series:
            _grouped(cats, series, "Mini-Bench accuracy by category", "Accuracy",
                     "fig_mini_categories.png")

    # 吞吐
    tp = [rows[m].get("decode_tok_s") for m in labels]
    if any(v for v in tp):
        xs = [m for m, v in zip(labels, tp) if v]
        vs = [v for v in tp if v]
        _bar(xs, vs, "Decode throughput (higher is better)", "tokens/s",
             "fig_throughput.png", color="#54A24B", fmt="{:.1f}")

    # TTFT
    tt = [rows[m].get("ttft_s") for m in labels]
    if any(v for v in tt):
        xs = [m for m, v in zip(labels, tt) if v]
        vs = [v for v in tt if v]
        _bar(xs, vs, "Time to first token (lower is better)", "seconds",
             "fig_ttft.png", color="#E45756", fmt="{:.3f}")

    # 精度-速度散点
    pts = [(rows[m].get("decode_tok_s"), rows[m].get("mini_bench"), m)
           for m in labels
           if rows[m].get("decode_tok_s") and rows[m].get("mini_bench") is not None]
    if len(pts) >= 2:
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=70, color="#B279A2")
        for x, y, m in pts:
            ax.annotate(_short(m, 18), (x, y), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Decode throughput (tokens/s)")
        ax.set_ylabel("Mini-Bench accuracy")
        ax.set_title("Accuracy vs Speed trade-off")
        fig.tight_layout()
        out = os.path.join(RES, "fig_acc_vs_speed.png")
        fig.savefig(out)
        plt.close(fig)
        print(f"  [图] {out}")

    # ------- 压测曲线（可选） -------
    lt = _load("loadtest.json")
    if lt and lt.get("rows"):
        rows_lt = lt["rows"]
        fig, ax1 = plt.subplots(figsize=(6.8, 4.4))
        xs = [r["concurrency"] for r in rows_lt]
        ax1.plot(xs, [r["output_tok_s"] for r in rows_lt], "o-", color="#4C78A8",
                 label="output tok/s")
        ax1.set_xlabel("Concurrency")
        ax1.set_ylabel("Output tokens/s", color="#4C78A8")
        ax1.tick_params(axis="y", labelcolor="#4C78A8")
        ax2 = ax1.twinx()
        ax2.plot(xs, [r.get("latency_p95_s") or 0 for r in rows_lt], "s--",
                 color="#E45756", label="P95 latency (s)")
        ax2.set_ylabel("P95 latency (s)", color="#E45756")
        ax2.tick_params(axis="y", labelcolor="#E45756")
        ax1.set_title("Throughput-Latency trade-off (load test)")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(RES, "fig_loadtest.png")
        fig.savefig(out)
        plt.close(fig)
        print(f"  [图] {out}")

        # 追加到报告
        rep = os.path.join(RES, "REPORT.md")
        if os.path.isfile(rep):
            with open(rep, "a", encoding="utf-8") as f:
                f.write("\n## 6. 并发压测（Serving 容量）\n\n")
                f.write(f"endpoint: `{lt.get('endpoint')}`　model: `{lt.get('model')}`"
                        f"　输出上限: {lt.get('max_tokens')}\n\n")
                f.write("| 并发 | QPS | 输出 tok/s | 平均延迟(s) | P50(s) | P95(s) "
                        "| TTFT(s) | 错误率 |\n")
                f.write("|---|---|---|---|---|---|---|---|\n")
                for r in rows_lt:
                    f.write(f"| {r['concurrency']} | {r['qps']} | {r['output_tok_s']} "
                            f"| {r['latency_avg_s']} | {r['latency_p50_s']} "
                            f"| {r['latency_p95_s']} | {r['ttft_avg_s']} "
                            f"| {r['error_rate']:.2%} |\n")
                f.write("\n![吞吐-延迟曲线](fig_loadtest.png)\n")

    # ------- 终端汇总 -------
    print("\n[3/3] 终端汇总\n")

    def fmt(v, p="{:.4f}"):
        return p.format(v) if isinstance(v, (int, float)) else "N/A"

    print("-" * 92)
    print(f"{'model':<34}{'PPL':>10}{'nextTok':>10}{'MiniBench':>11}{'tok/s':>10}{'TTFT':>9}")
    print("-" * 92)
    for m in labels:
        r = rows[m]
        print(f"{_short(m, 32):<34}"
              f"{fmt(r.get('ppl')):>10}"
              f"{fmt(r.get('next_token_acc')):>10}"
              f"{fmt(r.get('mini_bench')):>11}"
              f"{fmt(r.get('decode_tok_s'), '{:.1f}'):>10}"
              f"{fmt(r.get('ttft_s'), '{:.3f}'):>9}")
    print("-" * 92)
    print(f"\n报告目录: {RES}")
    print("   - REPORT.md   汇总报告（含图表引用）")
    print("   - summary.csv 汇总表")
    print("   - fig_*.png   各类图表")
    return 0


if __name__ == "__main__":
    sys.exit(main())
