#!/usr/bin/env python
"""08_fix_mm_wrapper.py —— 恢复被 llm-compressor 剥离的多模态 wrapper。

背景（现场现象）
    llm-compressor 的 oneshot 保存量化产物时，会把多模态模型
    （Qwen3_5ForConditionalGeneration）**剥离成纯文本模型**（Qwen3_5ForCausalLM）：
      * config.json 变成纯文本（丢 vision_config / text_config）
      * 视觉塔权重丢失
      * 但 processor_config.json 等仍留着
    → vLLM 按多模态初始化、却读到文本 config，报：
        TypeError: Expected Qwen3_5Config, but found Qwen3_5TextConfig

修复思路（两步量化法的第二步）
    原始 VLM 的 config + 视觉塔权重  ＋  量化后的 language_model 权重
    = 完整的多模态 wrapper，quantization_config 置于顶层 config。

用法:
    python scripts/08_fix_mm_wrapper.py <量化产物目录> [--orig <原始模型目录>] [--out <输出目录>]
    # 默认 orig = config.MODEL_PATH（base 模型），out = <量化目录>-mm
"""
import argparse
import copy
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

WEIGHT_EXTS = (".safetensors", ".bin")


def load_all(d):
    from safetensors.torch import load_file
    t = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        t.update(load_file(f))
    return t


def list_keys(d):
    """只读各分片的头部，返回 {key: shard_path}（不加载数据）。"""
    from safetensors import safe_open
    m = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, framework="pt", device="cpu") as h:
            for k in h.keys():
                m[k] = f
    return m


def load_keys(d, keep_fn=None, skip_fn=None):
    """按需读取：只加载满足 keep_fn / 不满足 skip_fn 的张量。

    大模型（27B fp16 ≈ 55GB）整目录读入内存代价很高，
    这里先只读头部拿到键名，再挑选需要的张量读取。
    """
    from safetensors import safe_open
    keymap = list_keys(d)
    need = {}
    for k, f in keymap.items():
        if keep_fn is not None and not keep_fn(k):
            continue
        if skip_fn is not None and skip_fn(k):
            continue
        need.setdefault(f, []).append(k)
    out = {}
    for f, ks in need.items():
        with safe_open(f, framework="pt", device="cpu") as h:
            for k in ks:
                out[k] = h.get_tensor(k)
    return out, set(keymap.keys())


def detect_lm_prefix(keys):
    """找到 language_model 的真实前缀。"""
    for cand in ("model.language_model.", "language_model.", "model."):
        if any(k.startswith(cand + "embed_tokens.weight") for k in keys):
            return cand
    return None


def map_lm_key(key, lm_prefix):
    """把纯文本量化产物的键映射回 wrapper 命名空间。

    ★ lm_head 属于 wrapper 的【顶层】，不能加 language_model. 前缀，
      否则会变成 model.language_model.lm_head.weight（多余的嵌套）。
    """
    if key.startswith(lm_prefix):
        return key
    if key == "lm_head.weight" or key.startswith("lm_head."):
        return key
    if key.startswith("model."):
        return lm_prefix + key[len("model."):]
    return lm_prefix + key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("quant_dir")
    ap.add_argument("--orig", default=None, help="原始多模态模型目录（默认 config.MODEL_PATH）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    qd = os.path.abspath(args.quant_dir.rstrip("/"))
    od = os.path.abspath(args.orig) if args.orig else config.require_model()
    out = args.out or (qd.rstrip("/") + "-mm")

    print("=" * 78)
    print("恢复多模态 wrapper")
    print(f"  量化产物: {qd}")
    print(f"  原始模型: {od}")
    print(f"  输出    : {out}")
    print("=" * 78)

    # ---------- config ----------
    with open(os.path.join(qd, "config.json"), encoding="utf-8") as f:
        qcfg = json.load(f)
    with open(os.path.join(od, "config.json"), encoding="utf-8") as f:
        ocfg = json.load(f)

    qq = qcfg.get("quantization_config")
    if qq is None:
        qq = (qcfg.get("text_config") or {}).get("quantization_config")
    if qq is None:
        raise SystemExit("[FAIL] 量化产物 config 里找不到 quantization_config")

    is_wrapper = not ocfg.get("vision_config") is None
    if not is_wrapper:
        print("[INFO] 原始模型不是多模态，无需恢复")

    final_cfg = copy.deepcopy(ocfg)
    final_cfg["quantization_config"] = qq
    # 视觉塔是未量化的 fp16 张量，必须让 vLLM 跳过它，否则会按量化建 weight_scale
    ign = qq.setdefault("ignore", [])
    for pat in ("re:.*visual.*", "re:.*linear_attn.*"):
        if pat not in ign:
            ign.append(pat)

    # ---------- 权重 ----------
    print("\n[1/3] 读取原始模型键名（只读头部）...")
    _, orig_keys = load_keys(od)
    print(f"      原始模型共 {len(orig_keys)} 个键")
    lm_prefix = detect_lm_prefix(orig_keys)
    print(f"      language_model 前缀 = {lm_prefix}")

    print("\n[2/3] 按需读取张量 ...")
    qw, q_keys = load_keys(qd)
    print(f"      量化产物: 全 {len(q_keys)} 键，读取 {len(qw)} 个")

    if is_wrapper:
        if lm_prefix is None:
            raise SystemExit("[FAIL] 原始模型里找不到 language_model（embed_tokens）")
        # 原始模型只读【非 language_model】的部分（视觉塔等），省内存
        ow, _ = load_keys(od, keep_fn=lambda k: not k.startswith(lm_prefix))
        print(f"      原始模型非 LM 权重: 读取 {len(ow)} 个（视觉塔等）")

        merged = dict(ow)
        n_lm = 0
        for k, v in qw.items():
            nk = map_lm_key(k, lm_prefix)
            if nk in merged:
                print(f"      [WARN] 键冲突，用原始权重: {nk}")
                continue
            merged[nk] = v
            n_lm += 1
        print(f"      合并完成: 原始非LM {len(ow)} 个 + 量化 language_model {n_lm} 个")
    else:
        merged = qw

    # 校验：量化张量是否都在
    n_scale = sum(1 for k in merged if k.endswith("weight_scale"))
    n_qscale_in = sum(1 for k in qw if k.endswith("weight_scale"))
    print(f"      weight_scale: 量化产物 {n_qscale_in} 个 → 合并后 {n_scale} 个")
    if n_qscale_in and n_scale < n_qscale_in:
        raise SystemExit("[FAIL] 合并后 weight_scale 变少，键映射可能出错")

    # ---------- 写出 ----------
    print("[3/3] 写出 ...")
    os.makedirs(out, exist_ok=True)
    # 权重较大时分片保存
    from safetensors.torch import save_file
    total = sum(v.numel() * v.element_size() for v in merged.values())
    max_shard = 4 * 1024 ** 3
    if total <= max_shard:
        save_file(merged, os.path.join(out, "model.safetensors"))
        print(f"      单文件写出 {total / 1024**3:.1f} GiB")
    else:
        # 简单分片
        shards, cur, cur_sz, idx = [], {}, 0, 1
        for k, v in merged.items():
            sz = v.numel() * v.element_size()
            if cur_sz + sz > max_shard and cur:
                shards.append(cur); cur, cur_sz = {}, 0
            cur[k] = v; cur_sz += sz
        if cur:
            shards.append(cur)
        weight_map = {}
        for i, sh in enumerate(shards, 1):
            fn = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
            save_file(sh, os.path.join(out, fn))
            for k in sh:
                weight_map[k] = fn
        with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
        print(f"      分片写出 {len(shards)} 个文件, 共 {total / 1024**3:.1f} GiB")

    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(final_cfg, f, indent=2, ensure_ascii=False)

    # 拷贝配套文件（tokenizer / processor / chat template）
    for fn in os.listdir(od):
        if fn == "config.json" or fn.endswith(WEIGHT_EXTS) or "index.json" in fn:
            continue
        s, dst = os.path.join(od, fn), os.path.join(out, fn)
        if os.path.isfile(s) and not os.path.exists(dst):
            shutil.copy(s, dst)

    # ---------- 校验 ----------
    with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
        oc = json.load(f)
    print("\n[校验]")
    print(f"    architectures = {oc.get('architectures')}")
    print(f"    有 vision_config = {bool(oc.get('vision_config'))}")
    print(f"    quantization_config.quant_method = "
          f"{(oc.get('quantization_config') or {}).get('quant_method')}")
    assert oc.get("vision_config"), "vision_config 丢失"
    print("\n使用方式:")
    print(f"    export MODEL_PATH={out}")
    print(f"    python eval/eval_ppl.py --model {out} --backend vllm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
