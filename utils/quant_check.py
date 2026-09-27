"""量化产物校验：判断 checkpoint 是否"真的量化落盘"，并给出可操作结论。

为什么需要它
    现场遇到过：llm-compressor 跑完打印"[OK] 已写出"，但产出的
    checkpoint 里**没有 weight_scale**（量化没落盘），
    config 却声明是量化的 →
      * transformers 加载时把这些层当作量化层，weight_scale 随机初始化
      * 反量化出垃圾权重 → logits 变 NaN → PPL = nan
    这种"静默失败"必须尽早、显式地暴露出来。

用法
    from utils.quant_check import inspect_quant_dir, format_report
    info = inspect_quant_dir(path)
    print(format_report(info))
    if not info["ok"]:
        raise RuntimeError("量化产物无效")
"""
import glob
import json
import os
import struct
from collections import Counter

# 量化标记张量后缀
QUANT_SUFFIXES = {
    "weight_scale": "compressed-tensors 权重缩放",
    "weight_packed": "compressed-tensors 打包权重(INT4)",
    "input_scale": "compressed-tensors 激活缩放",
    "qweight": "GPTQ 打包权重",
    "qzeros": "GPTQ 零点",
    "scales": "GPTQ/AWQ 缩放",
    "g_idx": "GPTQ 组索引",
}


def _read_header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def collect_tensor_info(d):
    keys, dtypes = set(), {}
    for p in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        try:
            hdr = _read_header(p)
        except Exception:
            continue
        for k, meta in hdr.items():
            if k == "__metadata__":
                continue
            keys.add(k)
            dtypes[k] = meta.get("dtype")
    return keys, dtypes


def inspect_quant_dir(d):
    """返回诊断结果 dict。"""
    d = os.path.abspath(d)
    info = {"path": d, "ok": False, "problems": [], "warnings": []}

    cfg_p = os.path.join(d, "config.json")
    cfg = {}
    if os.path.isfile(cfg_p):
        try:
            with open(cfg_p, encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            info["problems"].append(f"config.json 解析失败: {e}")
    info["cfg"] = cfg
    info["architectures"] = cfg.get("architectures")
    info["quant_config"] = cfg.get("quantization_config")
    info["ignore"] = (info["quant_config"] or {}).get("ignore")

    keys, dtypes = collect_tensor_info(d)
    info["n_tensors"] = len(keys)
    if not keys:
        info["problems"].append("目录里没有任何 .safetensors 张量（空产物）")
        return info

    cats = Counter()
    for k in keys:
        hit = None
        for suf in QUANT_SUFFIXES:
            if k.endswith(suf):
                hit = suf
                break
        cats[hit or ("weight" if k.endswith("weight") else "other")] += 1
    info["cats"] = dict(cats)

    # ★ 决定性指标：权重张量的 dtype
    #   int8 / uint8 / int32 → 量化成功；float16 / bfloat16 → 根本没量化
    wdtypes = Counter(dtypes[k] for k in keys if k.endswith("weight"))
    info["weight_dtypes"] = dict(wdtypes)
    info["has_int_weight"] = any(d in ("I8", "U8", "I32", "I16", "U16")
                                 for d in wdtypes)
    info["has_float_weight"] = any(d in ("F16", "BF16", "F32") for d in wdtypes)

    n_scale = cats.get("weight_scale", 0)
    n_packed = cats.get("weight_packed", 0)
    n_qw = cats.get("qweight", 0)
    info["n_quant_markers"] = n_scale + n_packed + n_qw

    # ---- 判定 0：权重 dtype（最直接的证据）----
    if not info["has_int_weight"] and info["n_quant_markers"] == 0:
        info["problems"].append(
            f"权重张量全是浮点（dtype 分布 {info['weight_dtypes']}），"
            "且无任何量化标记张量 → **量化完全没有生效**，产物就是原始 fp16 模型")

    # ---- 判定 1：量化是否落盘 ----
    if info["n_quant_markers"] == 0:
        info["problems"].append(
            "没有任何量化标记张量（weight_scale / weight_packed / qweight）"
            " → 量化【没有真正落盘】")
    else:
        # 量化标记存在，检查它是否覆盖了 config 声称要量化的层
        claimed = _claimed_linear_count(cfg)
        if claimed and n_scale and n_scale < claimed * 0.5:
            info["warnings"].append(
                f"config 声称量化 {claimed} 个 Linear，但只有 {n_scale} 个 weight_scale")

    # ---- 判定 2：ignore 是否写进了 config ----
    if info["n_quant_markers"] > 0 and not info["ignore"]:
        info["warnings"].append(
            "quantization_config 里没有 ignore 列表 → 加载时会尝试量化所有 Linear，"
            "与保存时跳过的层不一致（会出现大量 weight_scale MISSING）")

    # ---- 判定 3：config 是否被剥离成纯文本 ----
    arch = " ".join(info["architectures"] or [])
    has_vision = "vision_config" in cfg
    mm_files = [f for f in ("processor_config.json", "preprocessor_config.json",
                            "video_preprocessor_config.json")
                if os.path.isfile(os.path.join(d, f))]
    info["mm_files"] = mm_files
    if not has_vision and mm_files:
        info["problems"].append(
            "config 是纯文本（无 vision_config）但目录里有多模态处理器文件 "
            f"{mm_files} → vLLM 会报 "
            "TypeError: Expected Qwen3_5Config, but found Qwen3_5TextConfig")

    # ---- 判定 4：linear_attn / visual 是否被误量化 ----
    n_la = sum(1 for k in keys if "linear_attn" in k and k.endswith("weight_scale"))
    n_vis = sum(1 for k in keys if "visual" in k and k.endswith("weight_scale"))
    info["n_linear_attn_quant"] = n_la
    info["n_visual_quant"] = n_vis
    if n_la:
        info["problems"].append(
            f"linear_attn 被量化了（{n_la} 个 weight_scale）—— 混合线性注意力应跳过，"
            "否则 vLLM 加载后输出异常")
    if n_vis:
        info["problems"].append(f"视觉塔被量化了（{n_vis} 个 weight_scale）—— 应跳过")

    info["ok"] = not info["problems"]
    return info


def _claimed_linear_count(cfg):
    """粗略估算 config 声称量化的 Linear 层数（按层数 × 每层线性层数）。"""
    try:
        tc = cfg.get("text_config", cfg) or {}
        n = tc.get("num_hidden_layers") or 0
        # 每层：mlp 3 个 + self_attn 4 个(full) 或 linear_attn 5 个
        lt = tc.get("layer_types") or []
        if lt:
            n_full = sum(1 for t in lt if "full" in str(t).lower())
            n_lin = len(lt) - n_full
            return n_lin * 5 + n_full * 4 + n * 3
        return n * 7
    except Exception:
        return 0


def format_report(info, title="量化产物校验"):
    L = ["=" * 74, title, "=" * 74]
    L.append(f"路径: {info.get('path')}")
    L.append(f"architectures: {info.get('architectures')}")
    qc = info.get("quant_config")
    if qc:
        L.append(f"quantization_config: {json.dumps(qc, ensure_ascii=False)}")
        L.append(f"  └─ ignore: {info.get('ignore')}")
    else:
        L.append("quantization_config: 无")
    L.append(f"张量总数: {info.get('n_tensors')}   分类: {info.get('cats')}")
    L.append(f"权重 dtype 分布: {info.get('weight_dtypes')}")
    L.append(f"  └─ 有整型权重(量化成功标志): {info.get('has_int_weight')}")
    L.append(f"量化标记张量数: {info.get('n_quant_markers')}")

    if info.get("problems"):
        L.append("")
        L.append("❌ 发现的问题:")
        for p in info["problems"]:
            L.append(f"   * {p}")
        L.append("")
        L.append("可能原因与对策:")
        if info.get("n_quant_markers", 0) == 0:
            L.append("   - 量化未落盘：检查量化日志是否有 'Compressing model' / 'Writing shards'")
            L.append("   - 确认 ignore 正则带 re: 前缀（不带会被当字面匹配）")
            L.append("   - 确认该模型结构被 llm-compressor 正确识别")
        if "纯文本" in " ".join(info["problems"]):
            L.append("   - 纯文本部署: python scripts/09_make_text_only.py <目录> --apply")
            L.append("   - 完整多模态: python scripts/08_fix_mm_wrapper.py <目录> --orig <base>")
        if info.get("n_linear_attn_quant"):
            L.append("   - ignore 需含 're:.*linear_attn.*'")
    else:
        L.append("")
        L.append("✅ 未发现问题")

    if info.get("warnings"):
        L.append("")
        L.append("⚠️ 提示:")
        for w in info["warnings"]:
            L.append(f"   * {w}")
    L.append("=" * 74)
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python -m utils.quant_check <量化产物目录>")
        sys.exit(2)
    print(format_report(inspect_quant_dir(sys.argv[1])))
    sys.exit(0 if inspect_quant_dir(sys.argv[1])["ok"] else 1)
