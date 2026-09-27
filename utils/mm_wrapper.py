"""多模态 wrapper 恢复：把 llm-compressor 剥离成纯文本的量化产物复原。

为什么需要
    llm-compressor 的 oneshot 保存后会把多模态模型（如 Qwen3_5ForConditionalGeneration）
    写成纯文本（Qwen3_5ForCausalLM）：
      * config.json 丢 vision_config / text_config
      * 视觉塔权重丢失
      * 但权重键名仍处于 wrapper 命名空间（model.language_model.*）
    → transformers 按文本结构建模型，键名对不上 → 量化张量(weight_scale)加载不上
      → 随机初始化 → PPL = nan
    → vLLM 还会因残留 processor_config.json 报
      TypeError: Expected Qwen3_5Config, but found Qwen3_5TextConfig

做法
    原始 VLM 的 config + 视觉塔权重  ＋  量化后的 language_model 权重
    = 完整多模态 wrapper，quantization_config 置于顶层 config。

引用的键名布局（官方 Qwen3.6-27B，共 1199 键）
    model.language_model.*  850
    model.visual.*          333
    mtp.*                    15
    lm_head.weight            1
"""
import glob
import json
import os
import shutil

WEIGHT_EXTS = (".safetensors", ".bin")


def list_keys(d):
    """只读各分片头部，返回 {key: shard_path}（不加载数据）。"""
    from safetensors import safe_open
    m = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, framework="pt", device="cpu") as h:
            for k in h.keys():
                m[k] = f
    return m


def load_keys(d, keep_fn=None, skip_fn=None):
    """按需读取：先读头部拿键名，再只加载需要的张量（大模型省内存）。"""
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

    ★ lm_head 属于 wrapper 顶层，不能加 language_model. 前缀。
    """
    if key.startswith(lm_prefix):
        return key
    if key == "lm_head.weight" or key.startswith("lm_head."):
        return key
    if key.startswith("model."):
        return lm_prefix + key[len("model."):]
    return lm_prefix + key


def is_multimodal(path):
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return False
    return bool(cfg.get("vision_config")) or bool(cfg.get("text_config")
                                                  and cfg.get("image_token_id"))


def restore(quant_dir, orig_dir, out_dir=None, scale_dtype="float32",
            log=print):
    """核心：恢复多模态 wrapper。返回输出目录，失败抛异常。"""
    qd = os.path.abspath(quant_dir.rstrip("/"))
    od = os.path.abspath(orig_dir.rstrip("/"))
    out = out_dir or (qd + "-mm")

    log("=" * 74)
    log("恢复多模态 wrapper")
    log(f"  量化产物: {qd}")
    log(f"  原始模型: {od}")
    log(f"  输出    : {out}")
    log("=" * 74)

    with open(os.path.join(qd, "config.json"), encoding="utf-8") as f:
        qcfg = json.load(f)
    with open(os.path.join(od, "config.json"), encoding="utf-8") as f:
        ocfg = json.load(f)

    qq = qcfg.get("quantization_config")
    if qq is None:
        qq = (qcfg.get("text_config") or {}).get("quantization_config")
    if qq is None:
        raise RuntimeError("量化产物 config 里找不到 quantization_config")

    final_cfg = json.loads(json.dumps(ocfg))
    final_cfg["quantization_config"] = qq
    # 未量化的部分必须在 ignore 里声明
    ign = qq.setdefault("ignore", [])
    for pat in ("lm_head", "re:.*visual.*", "re:.*linear_attn.*"):
        if pat not in ign:
            ign.append(pat)

    _, orig_keys = load_keys(od)
    lm_prefix = detect_lm_prefix(orig_keys)
    log(f"  原始模型 {len(orig_keys)} 键，language_model 前缀 = {lm_prefix}")

    qw, q_keys = load_keys(qd)
    log(f"  量化产物 {len(q_keys)} 键，读取 {len(qw)} 个")

    if lm_prefix is None:
        log("  [INFO] 原始模型非 wrapper，直接透传量化产物")
        merged = qw
    else:
        ow, _ = load_keys(od, keep_fn=lambda k: not k.startswith(lm_prefix))
        log(f"  原始模型非 LM 权重 {len(ow)} 个（视觉塔等）")
        merged = dict(ow)
        n_lm = 0
        for k, v in qw.items():
            nk = map_lm_key(k, lm_prefix)
            if nk in merged:
                log(f"  [WARN] 键冲突，用原始权重: {nk}")
                continue
            merged[nk] = v
            n_lm += 1
        log(f"  合并: 原始非LM {len(ow)} + 量化 LM {n_lm}")

    if scale_dtype != "keep":
        import torch as _t
        tgt = {"float32": _t.float32, "bfloat16": _t.bfloat16}[scale_dtype]
        n = 0
        for k in list(merged):
            if k.endswith("weight_scale") or k.endswith("input_scale"):
                if merged[k].dtype != tgt:
                    merged[k] = merged[k].to(tgt)
                    n += 1
        log(f"  scale dtype 转换: {n} 个 -> {scale_dtype}")

    n_scale = sum(1 for k in merged if k.endswith("weight_scale"))
    n_in = sum(1 for k in qw if k.endswith("weight_scale"))
    log(f"  weight_scale: {n_in} -> {n_scale}")
    if n_in and n_scale < n_in:
        raise RuntimeError("合并后 weight_scale 变少，键映射可能出错")

    # 写出
    from safetensors.torch import save_file
    os.makedirs(out, exist_ok=True)
    total = sum(v.numel() * v.element_size() for v in merged.values())
    max_shard = 4 * 1024 ** 3
    if total <= max_shard:
        save_file(merged, os.path.join(out, "model.safetensors"))
    else:
        shards, cur, cur_sz = [], {}, 0
        for k, v in merged.items():
            sz = v.numel() * v.element_size()
            if cur_sz + sz > max_shard and cur:
                shards.append(cur)
                cur, cur_sz = {}, 0
            cur[k] = v
            cur_sz += sz
        if cur:
            shards.append(cur)
        wm = {}
        for i, sh in enumerate(shards, 1):
            fn = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
            save_file(sh, os.path.join(out, fn))
            for k in sh:
                wm[k] = fn
        with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total}, "weight_map": wm},
                      f, indent=2)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(final_cfg, f, indent=2, ensure_ascii=False)

    for fn in os.listdir(od):
        if fn == "config.json" or fn.endswith(WEIGHT_EXTS) or "index.json" in fn:
            continue
        s_, d_ = os.path.join(od, fn), os.path.join(out, fn)
        if os.path.isfile(s_) and not os.path.exists(d_):
            shutil.copy(s_, d_)

    log(f"  [OK] -> {out}")
    log(f"       architectures = {final_cfg.get('architectures')}")
    log(f"       有 vision_config = {bool(final_cfg.get('vision_config'))}")
    log(f"       ignore = {ign}")
    return out


def maybe_restore(quant_dir, orig_dir, scale_dtype="float32", log=print):
    """量化脚本保存后调用：源模型是多模态才恢复。返回输出目录或 None。"""
    if not is_multimodal(orig_dir):
        return None
    try:
        return restore(quant_dir, orig_dir, scale_dtype=scale_dtype, log=log)
    except Exception as e:
        log(f"[WARN] 多模态 wrapper 自动恢复失败: {e}")
        log(f"       可手动执行: python scripts/08_fix_mm_wrapper.py {quant_dir} "
            f"--orig {orig_dir}")
        return None
