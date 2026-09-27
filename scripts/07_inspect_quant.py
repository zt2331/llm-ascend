#!/usr/bin/env python
"""07_inspect_quant.py —— 诊断量化产物：判断量化是否真的落盘、config 是否被剥离。

背景（现场现象）：
  * 评测 PPL = nan
  * transformers 加载报告大量
      xxx.weight       | UNEXPECTED |    ← checkpoint 里是普通权重
      xxx.weight_scale | MISSING    |    ← 模型期望量化权重却没有
  * vLLM 报 TypeError: Expected Qwen3_5Config, but found Qwen3_5TextConfig

本脚本回答三个问题：
  1. 量化权重到底有没有写进 checkpoint？（统计 weight_scale / input_scale 数量）
  2. 哪些层被量化了、哪些没有？
  3. config 是否被剥离成纯文本（导致 vLLM 多模态加载失败）？

用法:
    python scripts/07_inspect_quant.py <量化产物目录>
    python scripts/07_inspect_quant.py output_models/quantized/xxx --orig <原始模型目录>
"""
import argparse
import glob
import json
import os
import struct
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"


def read_header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return {k: v for k, v in hdr.items() if k != "__metadata__"}


def collect_keys(d):
    keys = {}
    dtype = {}
    for p in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        hdr = read_header(p)
        for k, meta in hdr.items():
            keys[k] = os.path.basename(p)
            dtype[k] = meta.get("dtype")
    return keys, dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("quant_dir")
    ap.add_argument("--orig", default=None, help="原始模型目录（用于对比 config）")
    args = ap.parse_args()

    d = os.path.abspath(args.quant_dir)
    if not os.path.isdir(d):
        print(f"[FAIL] 目录不存在: {d}")
        return 1

    print("=" * 78)
    print(f"量化产物诊断: {d}")
    print("=" * 78)

    # ---------- 1. config ----------
    cfg_p = os.path.join(d, "config.json")
    cfg = {}
    if os.path.isfile(cfg_p):
        with open(cfg_p, encoding="utf-8") as f:
            cfg = json.load(f)
    print("\n[1] config.json")
    print(f"    architectures      : {cfg.get('architectures')}")
    print(f"    model_type         : {cfg.get('model_type')}")
    print(f"    有 vision_config   : {bool(cfg.get('vision_config'))}")
    print(f"    有 text_config     : {bool(cfg.get('text_config'))}")
    qc = cfg.get("quantization_config") or {}
    print(f"    quantization_config: {json.dumps(qc, ensure_ascii=False)[:400]}")

    # 多模态剥离判断
    mm_files = [f for f in ("processor_config.json", "preprocessor_config.json",
                            "video_preprocessor_config.json")
                if os.path.isfile(os.path.join(d, f))]
    is_text_only = "ForCausalLM" in " ".join(cfg.get("architectures") or []) \
        and not cfg.get("vision_config")
    print(f"\n    多模态配套文件: {mm_files or '无'}")
    if is_text_only and mm_files:
        print(f"    {BAD} config 是【纯文本】但存在多模态处理器文件 →")
        print("         vLLM 会按多模态初始化，与文本 config 冲突：")
        print("         TypeError: Expected Qwen3_5Config, but found Qwen3_5TextConfig")
        print("         → 需要把多模态 wrapper 的 config/视觉塔权重恢复回来")
    elif cfg.get("vision_config"):
        print(f"    {OK} config 保留多模态 wrapper")
    else:
        print(f"    ℹ️ 纯文本模型，无此问题")

    # ---------- 2. 张量统计 ----------
    keys, dtype = collect_keys(d)
    print(f"\n[2] 权重张量统计（共 {len(keys)} 个）")
    if not keys:
        print(f"    {BAD} 没有任何 .safetensors 张量 —— 目录是空的！")
        return 2

    cats = Counter()
    for k in keys:
        if k.endswith("weight_scale"):
            cats["weight_scale"] += 1
        elif k.endswith("input_scale"):
            cats["input_scale"] += 1
        elif k.endswith("weight_shape"):
            cats["weight_shape"] += 1
        elif k.endswith("weight_packed"):
            cats["weight_packed"] += 1
        elif k.endswith("qweight"):
            cats["qweight(gptq)"] += 1
        elif k.endswith("qzeros"):
            cats["qzeros(gptq)"] += 1
        elif k.endswith("scales"):
            cats["scales(gptq/awq)"] += 1
        elif k.endswith("weight"):
            cats["weight(普通)"] += 1
        else:
            cats["其它"] += 1
    for c, n in cats.most_common():
        print(f"    {c:<20}{n}")

    # ---------- 2.5 键名前缀（判断是否与 config 结构匹配）----------
    pref = Counter(".".join(k.split(".")[:2]) for k in keys)
    print("\n[2.5] 键名前缀分布（Top 8）")
    for p_, n in pref.most_common(8):
        print(f"    {p_:<40}{n}")
    info_prefixes = set(pref)
    has_lm_ns = any("language_model" in p_ for p_ in info_prefixes)
    has_plain_ns = any(p_ == "model.layers" for p_ in info_prefixes)
    print(f"\n    含 'language_model' 命名空间: {has_lm_ns}")
    print(f"    含 'model.layers'（纯文本命名空间）: {has_plain_ns}")
    if cfg.get("architectures") and "CausalLM" in " ".join(cfg.get("architectures") or []):
        if has_lm_ns:
            print(f"    {BAD} config 是纯文本({cfg.get('architectures')})，"
                  f"但权重键带 language_model 前缀")
            print("         → transformers 按文本结构建模型，找不到这些键，")
            print("           量化张量(weight_scale)加载不上 → 随机初始化 → PPL=nan")
            print("         → 修复: python scripts/08_fix_mm_wrapper.py <目录> --orig <base>")
    print("\n    量化张量键示例:")
    for k in [x for x in sorted(keys) if x.endswith("weight_scale")][:3]:
        print(f"      {k}")

    quant_markers = cats["weight_scale"] + cats["weight_packed"] + cats["qweight"]
    print()
    if quant_markers == 0:
        print(f"    {BAD} 没有任何量化标记张量（weight_scale / weight_packed / qweight）")
        print("         → 量化【没有真正落盘】，checkpoint 还是普通权重")
        print("         这就是 PPL=nan 的根因：模型按量化结构建好，权重却是空/随机")
    else:
        print(f"    {OK} 检测到 {quant_markers} 个量化标记张量 → 量化已落盘")

    # ---------- 3. 逐模块看量化覆盖 ----------
    print("\n[3] 逐类模块的量化覆盖")
    groups = {
        "mlp.gate_proj": [], "mlp.up_proj": [], "mlp.down_proj": [],
        "self_attn.q_proj": [], "self_attn.k_proj": [], "self_attn.v_proj": [],
        "self_attn.o_proj": [], "linear_attn": [], "visual": [], "lm_head": [],
    }
    for k in keys:
        for g in groups:
            if g in k:
                groups[g].append(k)
                break
    print(f"    {'模块':<20}{'权重':>6}{'weight_scale':>14}{'input_scale':>13}  结论")
    for g, ks in groups.items():
        if not ks:
            continue
        nw = sum(1 for k in ks if k.endswith("weight"))
        ns = sum(1 for k in ks if k.endswith("weight_scale"))
        ni = sum(1 for k in ks if k.endswith("input_scale"))
        if ns == 0 and nw > 0:
            verdict = "未量化（保持 fp16）"
        elif ns > 0:
            verdict = "已量化"
        else:
            verdict = "无权重"
        print(f"    {g:<20}{nw:>6}{ns:>14}{ni:>13}  {verdict}")

    # ---------- 4. 结论与建议 ----------
    print("\n[4] 结论")
    problems = []
    if quant_markers == 0:
        problems.append("量化未落盘（无 weight_scale）")
    n_la = sum(1 for k in keys if "linear_attn" in k and k.endswith("weight_scale"))
    if n_la > 0:
        problems.append(f"linear_attn 被量化了（{n_la} 个 scale），应跳过")
    n_vis = sum(1 for k in keys if "visual" in k and k.endswith("weight_scale"))
    if n_vis > 0:
        problems.append(f"视觉塔被量化了（{n_vis} 个 scale），应跳过")
    if is_text_only and mm_files:
        problems.append("config 被剥离成纯文本（vLLM 无法加载）")

    if not problems:
        print(f"    {OK} 未发现明显问题")
    else:
        for p in problems:
            print(f"    ❌ {p}")
        print("\n    建议：")
        if quant_markers == 0:
            print("      * 重跑量化，并确认日志里出现 'Compressing model' 与 'Writing model shards'")
            print("      * 检查 ignore 正则是否带 re: 前缀（不带会被当字面匹配而失效）")
        if n_la > 0 or n_vis > 0:
            print("      * 确认 ignore 含 're:.*linear_attn.*' 与 're:visual($|\\.)'")
        if is_text_only and mm_files:
            print("      * 需要恢复多模态 wrapper（脚本 scripts/08_fix_mm_wrapper.py）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
