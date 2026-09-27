#!/usr/bin/env python
"""10_prepare_real_data.py —— 把【真实语料】接入项目（支持多种格式）。

项目对数据的唯一约定：parquet 文件，含一列文本（默认列名 text）。
本脚本负责把任意常见格式转成该格式，并按需切分校准/评测集。

支持输入格式
    .parquet  直接读（列名不是 text 也能自动识别第一个字符串列）
    .jsonl / .json   自动找文本字段（text/content/instruction/output/prompt/...）
    .csv / .tsv      自动找字符串列
    .txt / .md       整篇当一个文档；空行分段

用法
    # 单个文件 → 自动切分为校准/评测
    python scripts/10_prepare_real_data.py --input /path/to/corpus.parquet
    python scripts/10_prepare_real_data.py --input /path/to/data.jsonl --calib 256 --test 64

    # 分别指定校准集与评测集（各自可多文件/目录）
    python scripts/10_prepare_real_data.py \
        --calib-input /path/calib_dir --test-input /path/test.parquet

    # 先只看会做什么，不写文件
    python scripts/10_prepare_real_data.py --input x.jsonl --dry-run
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.dataio import save_rows

# 问题/指令类字段（通常较短）
Q_KEYS = ["instruction", "question", "prompt", "query", "input", "user"]
# 回答/正文类字段（通常较长，是我们要的文本）
A_KEYS = ["output", "response", "answer", "completion", "assistant",
          "text", "content", "document", "passage", "body", "article",
          "chunk", "sentence", "code"]


def _expand(paths):
    """把文件/目录/通配符展开成文件列表。"""
    out = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("*.parquet", "*.jsonl", "*.json", "*.csv", "*.tsv",
                        "*.txt", "*.md"):
                out.extend(sorted(glob.glob(os.path.join(p, ext))))
        else:
            out.extend(sorted(glob.glob(p)))
    return [f for f in out if os.path.isfile(f)]


def extract_text(obj):
    """从一条记录里取出最合适的文本。

    ★ 不要按固定顺序取第一个命中的字段：
      instruction-tuning 数据里 `instruction` 只有几个字，
      而真正的内容在 `output` 里。若先取 instruction 会被长度过滤掉 → 0 条。
    策略：
      1) instruction/question/prompt 与 output/response/answer 同时存在 → 拼接（最完整）
      2) 否则取回答/正文类字段
      3) 否则取最长的字符串字段
    """
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return None
    low = {k.lower(): k for k in obj}

    def get(keys):
        for n in keys:
            k = low.get(n)
            if k is not None and isinstance(obj[k], str) and obj[k].strip():
                return obj[k].strip()
        return None

    q = get(Q_KEYS)
    a = get(A_KEYS)
    if q and a:
        return f"{q}\n{a}"
    if a:
        return a
    if q:
        return q
    strs = [v for v in obj.values() if isinstance(v, str) and v.strip()]
    return max(strs, key=len) if strs else None


def read_file(path):
    """读任意支持格式 → [文本, ...]。"""
    ext = os.path.splitext(path)[1].lower()
    texts = []

    if ext == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        col = "text" if "text" in df.columns else None
        if col is None:
            cands = [c for c in df.columns if df[c].dtype == object]
            col = cands[0] if cands else df.columns[0]
        texts = [str(t) for t in df[col].tolist()]

    elif ext in (".jsonl", ".ndjson"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                t = extract_text(obj)
                if t:
                    texts.append(t)

    elif ext == ".json":
        with open(path, encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # 可能是 {"data": [...]} 之类的包装
            for key in ("data", "records", "items", "rows", "examples"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if isinstance(data, list):
            for obj in data:
                t = extract_text(obj)
                if t:
                    texts.append(t)
        elif isinstance(data, str):
            texts = [data]

    elif ext in (".csv", ".tsv"):
        import csv
        sep = "\t" if ext == ".tsv" else ","
        with open(path, encoding="utf-8", errors="ignore", newline="") as f:
            rd = csv.DictReader(f, delimiter=sep)
            for row in rd:
                t = extract_text(row)
                if t:
                    texts.append(t)

    else:  # .txt / .md 等纯文本
        with open(path, encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        paras = [p.strip() for p in raw.split("\n\n") if len(p.strip()) >= 50]
        texts = paras or [l.strip() for l in raw.splitlines() if len(l.strip()) >= 20]

    return [t for t in texts if t and len(t.strip()) >= 8]


def collect(paths, label):
    files = _expand(paths)
    if not files:
        print(f"  [{label}] 未找到任何文件: {paths}")
        return []
    texts = []
    for f in files:
        try:
            t = read_file(f)
        except Exception as e:
            print(f"  [{label}] 读取失败 {os.path.basename(f)}: {str(e)[:80]}")
            continue
        print(f"  [{label}] {os.path.basename(f):<40} {len(t)} 条")
        texts.extend(t)
    return texts


def stats(texts, name):
    if not texts:
        print(f"  {name}: 空")
        return
    import statistics
    lens = [len(t) for t in texts]
    print(f"  {name}: {len(texts)} 条 | 平均 {statistics.mean(lens):.0f} 字符 "
          f"| 最短 {min(lens)} | 最长 {max(lens)}")
    print(f"        估算总量 ≈ {sum(lens) / 2:.0f} tokens（按 2 字符/token 粗估）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="*", default=None,
                    help="单一语料来源（自动切分为校准/评测）")
    ap.add_argument("--calib-input", nargs="*", default=None, help="校准集来源")
    ap.add_argument("--test-input", nargs="*", default=None, help="评测集来源")
    ap.add_argument("--calib", type=int, default=128, help="校准集条数（默认 128）")
    ap.add_argument("--test", type=int, default=64, help="评测集条数（默认 64）")
    ap.add_argument("--dry-run", action="store_true", help="只预览，不写文件")
    args = ap.parse_args()

    if not (args.input or args.calib_input or args.test_input):
        ap.error("需要 --input 或 --calib-input/--test-input")

    print("=" * 74)
    print("接入真实语料")
    print("=" * 74)

    if args.calib_input or args.test_input:
        print("\n[1] 分别读取校准集与评测集")
        calib = collect(args.calib_input or [], "calib")
        test = collect(args.test_input or [], "test")
    else:
        print("\n[1] 读取语料并自动切分")
        all_texts = collect(args.input, "all")
        if not all_texts:
            print("[FAIL] 没读到任何文本")
            return 1
        need = args.calib + args.test
        if len(all_texts) < need:
            print(f"  [WARN] 语料只有 {len(all_texts)} 条，少于需要的 {need} 条；"
                  f"校准集与评测集会有重叠")
        # 固定顺序切分，保证可复现（不 shuffle）
        calib = all_texts[:args.calib]
        test = all_texts[args.calib:args.calib + args.test]
        if len(test) < args.test:
            # 不够就从前面补（允许重叠，但提示）
            test = all_texts[-args.test:] if len(all_texts) >= args.test else all_texts

    print("\n[2] 统计")
    stats(calib, "校准集")
    stats(test, "评测集")

    if args.dry_run:
        print("\n[--dry-run] 未写文件。")
        return 0

    print("\n[3] 写出")
    config.ensure_dirs()
    cp = os.path.join(config.CALIB_DIR, "validation.parquet")
    tp = os.path.join(config.TEST_DIR, "test.parquet")
    save_rows(calib, cp)
    save_rows(test, tp)
    print(f"  校准 -> {cp}  ({len(calib)} 条)")
    print(f"  评测 -> {tp}  ({len(test)} 条)")

    # 校验能被项目读取
    try:
        from utils.dataio import Dataset
        for name, p in (("校准", cp), ("评测", tp)):
            d = Dataset.from_parquet(p)
            print(f"  [校验] {name}: {len(d)} 条, 列={d.column_names}, "
                  f"首条前60字='{d[0]['text'][:60].replace(chr(10),' ')}'")
    except Exception as e:
        print(f"  [WARN] 校验失败: {e}")
        return 1

    print("\n[OK] 完成。建议参数：")
    print(f"  校准条数 CALIB={len(calib)}   序列长度 SEQ=1024~2048")
    print("  重跑量化：")
    print("    QMETHODS=smooth,rtn KEEP=54 CALIB=%d SEQ=1024 "
          "STAGES=prune,quantize bash scripts/run_all.sh" % len(calib))
    return 0


if __name__ == "__main__":
    sys.exit(main())
