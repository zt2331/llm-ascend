"""模型工具（昇腾版）：多模态感知的「文本 decoder 定位」+ 设备感知加载。

- 纯文本模型（Qwen3.6-27B 等）：decoder 就是 model.layers。
- 多模态模型：视觉塔保持高精度，压缩只作用于文本 decoder。
- 加载时用 utils.device 选设备，NPU 上走 torch_npu。
"""
import json
import os
import re

import torch
import torch.nn as nn

from utils import device as dev

VISION_ATTRS = ("visual", "vision_model", "vision_tower", "image_encoder", "vision_encoder")

# ★ llm-compressor 的 ignore 列表：**正则必须带 `re:` 前缀**，
#   否则会被当作"字面字符串精确匹配"，导致通配模式全部失效（曾踩此坑）。
_RE = "re:"
# 跳过视觉塔
VISION_IGNORE = [_RE + re.escape(a) + r"($|\.)" for a in VISION_ATTRS]


def _find_lm_head_owner(model):
    """返回 (owner_module, attr_name)，或用 None 表示模型没有 lm_head。"""
    for owner, attr in ((model, "lm_head"),):
        if hasattr(owner, attr):
            return owner, attr
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "lm_head"):
        return lm, "lm_head"
    inner = getattr(model, "model", None)
    if inner is not None:
        if hasattr(inner, "lm_head"):
            return inner, "lm_head"
        lm2 = getattr(inner, "language_model", None)
        if lm2 is not None and hasattr(lm2, "lm_head"):
            return lm2, "lm_head"
    return None


def ensure_lm_head(model, path):
    """确保模型带 lm_head，否则 AutoModel 返回的基座模型取不到 logits。

    多模态模型（如 Qwen3_5ForConditionalGeneration）用 AutoModel 加载后是
    `Qwen3_5Model`，**不含 lm_head**；此时 eval 阶段 `model(x).logits` 会失败。
    这里从 checkpoint 里把 lm_head 权重找出来补上（或与 embed_tokens 共享）。
    """
    if _find_lm_head_owner(model) is not None:
        return model

    # 找 hidden_size / vocab_size
    cfg = getattr(model, "config", None)
    tc = getattr(cfg, "text_config", None) or cfg
    hidden = getattr(tc, "hidden_size", None)
    vocab = getattr(tc, "vocab_size", None)
    if not hidden or not vocab:
        return model

    weight = None
    try:
        import glob
        from safetensors import safe_open
        for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            with safe_open(shard, framework="pt", device="cpu") as f:
                if "lm_head.weight" in f.keys():
                    weight = f.get_tensor("lm_head.weight")
                    break
    except Exception:
        weight = None

    head = nn.Linear(hidden, vocab, bias=False)
    if weight is not None and tuple(weight.shape) == (vocab, hidden):
        with torch.no_grad():
            head.weight.copy_(weight)
        print(f"[INFO] 已从 checkpoint 补上 lm_head: {tuple(weight.shape)}")
    else:
        # 没有独立 lm_head 权重 → 与词嵌入共享（tie_word_embeddings）
        dec = _find_transformer_decoder(model)
        emb = getattr(dec, "embed_tokens", None)
        if emb is not None and tuple(emb.weight.shape) == (vocab, hidden):
            head.weight = emb.weight
            print("[INFO] lm_head 未找到，已与 embed_tokens 共享权重")
        else:
            print("[WARN] 无法补上 lm_head，eval 取 logits 可能失败")
            return model
    model.lm_head = head
    return model


# ----------------------------- 加载 -----------------------------
def load_model(path, prefer_device="auto", dtype="float16", device_map=None,
               eval_mode=True, need_logits=False):
    """设备感知加载。

    need_logits=True 时（评测用），会确保模型带 lm_head；
    多模态模型用 AutoModel 加载后基座不含 lm_head，会自动从 checkpoint 补上。
    """
    from transformers import AutoModel, AutoModelForCausalLM

    tdtype = getattr(torch, dtype, torch.float16)
    kwargs = dict(trust_remote_code=True, torch_dtype=tdtype, low_cpu_mem_usage=True)
    d = dev.default_device(prefer_device)
    if device_map is not None:
        kwargs["device_map"] = device_map

    model = None
    arch = ""
    try:
        from transformers import AutoConfig
        arch = " ".join(AutoConfig.from_pretrained(
            path, trust_remote_code=True).architectures or [])
    except Exception:
        arch = ""

    # ★ 多模态 VL 模型（如 Qwen3_5ForConditionalGeneration）用基座 AutoModel 加载时，
    #   参数命名空间是 `language_model.*`，而 checkpoint 是 `model.language_model.*`，
    #   量化参数(weight_scale)会因前缀不匹配加载不上 → NaN。
    #   所以优先用 ForImageTextToText / ForConditionalGeneration 这类正确类。
    for loader_name in (["AutoModelForImageTextToText"] if ("ConditionalGeneration" in arch
                                                            or "ImageTextToText" in arch)
                        else []) + (
                        ["AutoModelForCausalLM"] if need_logits else []) + ["AutoModel"]:
        try:
            import transformers
            loader = getattr(transformers, loader_name, None)
            if loader is None:
                continue
            model = loader.from_pretrained(path, **kwargs)
            print(f"[INFO] 用 {loader_name} 加载（architectures={arch or '未知'}）")
            break
        except Exception as e:
            print(f"[INFO] {loader_name} 不可用: {str(e)[:80]}")
            model = None
    if model is None:
        raise RuntimeError(f"无法加载模型: {path}")

    if need_logits:
        ensure_lm_head(model, path)

    if device_map is None and d != "cpu":
        try:
            model.to(dev.get_device(prefer_device))
        except Exception as e:
            print(f"[WARN] 移动到 {d} 失败：{e}（继续用 CPU）")
    if eval_mode:
        model.eval()
    return model


# ------------------------- decoder 定位 -------------------------
def _find_transformer_decoder(model):
    for cand in (model, getattr(model, "model", None),
                 getattr(model, "language_model", None),
                 getattr(model, "transformer", None)):
        if cand is not None and hasattr(cand, "layers"):
            return cand
    for _, mod in model.named_modules():
        if hasattr(mod, "layers") and isinstance(getattr(mod, "layers"), nn.ModuleList):
            return mod
    return None


def text_decoder(model):
    return _find_transformer_decoder(model)


def decoder_layers(model):
    d = _find_transformer_decoder(model)
    return list(d.layers) if d is not None else None


def is_multimodal(model) -> bool:
    for attr in VISION_ATTRS:
        if hasattr(model, attr) or hasattr(getattr(model, "model", None), attr):
            return True
    try:
        for name, _ in model.named_modules():
            if any(s in VISION_ATTRS for s in name.split(".")):
                return True
    except Exception:
        pass
    return False


def is_multimodal_config(path) -> bool:
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return False
    keys = " ".join(str(k).lower() for k in cfg.keys())
    arch = " ".join(str(a).lower() for a in cfg.get("architectures", []))
    return any(k in keys for k in ("vision_config", "visual_config",
                                   "image_encoder", "vision_tower")) or "vl" in arch


def iter_decoder_linears(model, suffixes=None):
    """只遍历文本 decoder 内的 nn.Linear（跳过视觉塔）。返回 [(全名, module)]。"""
    suffixes = suffixes or ("q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj")
    dec = _find_transformer_decoder(model)
    if dec is None:
        return []
    prefix = _prefix_of(dec, model)
    out = []
    for name, mod in dec.named_modules():
        if isinstance(mod, nn.Linear) and any(s in name for s in suffixes):
            out.append(((prefix + "." + name) if prefix else name, mod))
    return out


def iter_linears_in(module, suffixes=None):
    """遍历**任意模块内**的 nn.Linear（按后缀过滤）。返回 [(相对名, module)]。

    用于逐层处理（如手写 AWQ/GPTQ 遍历某个 decoder layer 内部的线性层）。
    注意：iter_decoder_linears 需要传入"整个模型"才能定位 .layers；
    传入单层时请用本函数。
    """
    suffixes = suffixes or ("q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj")
    out = []
    for name, mod in module.named_modules():
        if isinstance(mod, nn.Linear) and any(s in name for s in suffixes):
            out.append((name, mod))
    return out


def _prefix_of(sub, root):
    for pname, pmod in root.named_modules():
        if pmod is sub:
            return pname
    return ""


def has_linear_attn_config(path) -> bool:
    """从 config.json 判断是否为「混合线性注意力」模型（如 Qwen3.5/3.6 的 GatedDeltaNet）。

    这类模型 layer_types 里会同时出现 linear_attention / full_attention。
    """
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return False
    tc = cfg.get("text_config", cfg) or {}
    lt = tc.get("layer_types") or cfg.get("layer_types") or []
    return any("linear" in str(t).lower() for t in lt)


def quant_ignore_patterns(path, extra=None):
    """构造 llm-compressor 的 ignore 列表。

    三处默认不量化（保持 bf16）：
      1. lm_head            —— 直接决定输出 logits 分布
      2. 视觉塔(visual...)  —— 多模态模型，结构特殊且基准是文本任务
      3. linear_attn        —— ★ 混合线性注意力(GatedDeltaNet)：
         vLLM 的量化加载器与 llm-compressor 的模块命名不一致
         （vLLM 期望 in_proj_baa/in_proj_qkvzz 等融合名），
         量化后会出现「能加载但输出乱码」。必须跳过。
    """
    ignore = list(extra) if extra else ["lm_head"]
    if is_multimodal_config(path):
        ignore += VISION_IGNORE
    if has_linear_attn_config(path):
        # 注意 re: 前缀，否则不生效
        ignore.append(_RE + r".*linear_attn.*")
    return ignore
