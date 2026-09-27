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
# llm-compressor recipe 里用于跳过视觉塔的正则
VISION_IGNORE = [re.escape(a) + r"($|\.)" for a in VISION_ATTRS]


# ----------------------------- 加载 -----------------------------
def load_model(path, prefer_device="auto", dtype="float16", device_map=None,
               eval_mode=True):
    """设备感知加载：优先 AutoModel（多模态/文本都能出），失败退 AutoModelForCausalLM。"""
    from transformers import AutoModel, AutoModelForCausalLM

    tdtype = getattr(torch, dtype, torch.float16)
    kwargs = dict(trust_remote_code=True, torch_dtype=tdtype, low_cpu_mem_usage=True)
    d = dev.default_device(prefer_device)
    if device_map is not None:
        kwargs["device_map"] = device_map

    try:
        model = AutoModel.from_pretrained(path, **kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)

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


def quant_ignore_patterns(path, extra=("lm_head",)):
    """构造 llm-compressor 的 ignore 列表。多模态自动追加视觉塔。

    注意：昇腾侧只量化标准 Linear；若模型含混合线性注意力（如 linear_attn），
    建议一并 ignore，避免推理引擎量化加载器命名不匹配。
    """
    ignore = list(extra)
    if is_multimodal_config(path):
        ignore += VISION_IGNORE
    return ignore
