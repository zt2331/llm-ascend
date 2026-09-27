#!/usr/bin/env python
"""从零手写 AWQ 量化 → 产出「vLLM 可直接加载」的 checkpoint（主流格式）。

算法（不依赖任何第三方量化库）：
  1. 校准：统计每个「归一化层输出」的激活幅度（q/k/v 共享、gate/up 共享同一输入）；
  2. 逐输入通道缩放：s_j = (max|X_j|)^α / (max|W_j|)^(1-α)，归一化后限幅；
     —— 激活大的通道放大（保护它），权重大的通道缩小（别撑坏量化区间）；
  3. 缩放权重 W' = W · diag(s)，并对 W' 做 per-group(128) 非对称量化 → q / scale_q / zero；
  4. **缩放折叠**：把 1/s 折进前一个归一化层的 weight（γ' = γ / s），
     使该 Norm 输出天然变成 X/s，**推理时零额外开销**；
  5. 按内核 AWQ_ORDER=[0,2,4,6,1,3,5,7] 打包 qweight/qzeros；
  6. 输出完整 config.json（含 model_type/architectures + quantization_config）。

> 为什么缩放必须按「输入通道」而不是「按组常数」：
>   若一组内所有通道同乘常数 s，则 scale_q 也同乘 s，`round(W·s / (s·scale_q))`
>   会把 s 约掉，等于没做 AWQ。只有改变通道间的相对分布才有效。

产出格式：AWQ（zero_point=True），vLLM 用 `--quantization awq_marlin` 加载。

用法（项目根）:
    python quantize/manual_awq.py                       # 全部文本 decoder 层 4bit
    python quantize/manual_awq.py --max-layers 2        # 快速试跑前 2 层
    python quantize/manual_awq.py --model output_models/pruned/xxx
"""
import argparse
import json
import os
import shutil
import sys

import torch
import torch.nn as nn
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import device as dev
from utils.model_utils import load_model, decoder_layers, iter_linears_in

PROJ_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# vLLM / AutoAWQ 内核要求的组内位序（不是自然序 0..7）
AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]

# 需要缩放（并共享同一前置 Norm）的投影组
SCALE_GROUPS = {
    "attn": (("q_proj", "k_proj", "v_proj"), ("input_layernorm", "ln_1", "attention_norm")),
    "mlp": (("gate_proj", "up_proj"), ("post_attention_layernorm", "ln_2", "ffn_norm")),
}


def pack_last_awq(int_mat, bits=4):
    """int_mat:[r,c] 沿最后一维每 8 个打包进 int32，按 AWQ_ORDER 组内排序。"""
    shifts = torch.arange(0, 32, bits, device=int_mat.device)
    m = (int_mat.to(torch.int8) & 0x0F).view(int_mat.shape[0], int_mat.shape[1] // 8, 8)
    m = m[:, :, AWQ_ORDER]
    return (m.to(torch.int64) << shifts[None, None, :]).sum(-1).to(torch.int32)


# ----------------------------------------------------------------------
# 1) 计算缩放
# ----------------------------------------------------------------------
def awq_channel_scale(W, act_max, alpha=0.5, lo=0.1, hi=10.0):
    """按输入通道算缩放。W:[out,in]  act_max:[in]（每输入通道激活峰值）→ s:[in]

    s_j = (max|X_j|)^α / (max|W_j|)^(1-α)，再归一化到均值 1 并限幅。
    """
    w_max = W.abs().amax(dim=0).float().clamp_min(1e-8)      # [in]
    x_max = act_max.float().clamp_min(1e-8)                  # [in]
    s = (x_max.pow(alpha)) / (w_max.pow(1.0 - alpha))
    s = s / s.mean().clamp_min(1e-8)
    return s.clamp(lo, hi)


# ----------------------------------------------------------------------
# 2) 分组非对称量化
# ----------------------------------------------------------------------
@torch.no_grad()
def quant_group_asym(W, bits=4, group_size=128):
    """W:[out,in] → (q int8, scale_q [out,gnum], zero [out,gnum])，组内非对称。"""
    out_f, in_f = W.shape
    qmin, qmax = 0, (1 << bits) - 1
    pad = (-in_f) % group_size
    if pad:
        W = torch.nn.functional.pad(W, (0, pad))
    gnum = W.shape[1] // group_size
    Wg = W.float().view(out_f, gnum, group_size)

    mn = Wg.min(dim=2, keepdim=True).values
    mx = Wg.max(dim=2, keepdim=True).values
    scale_q = ((mx - mn) / (qmax - qmin)).clamp_min(1e-8)
    zero = torch.round(qmin - mn / scale_q).clamp(qmin, qmax)
    q = (torch.round(Wg / scale_q) + zero).clamp(qmin, qmax)

    return (q.view(out_f, -1)[:, :in_f].to(torch.int8),
            scale_q.view(out_f, gnum),
            zero.view(out_f, gnum))


# ----------------------------------------------------------------------
# 3) 采集激活（Norm 输出 = q/k/v 或 gate/up 的共享输入）
# ----------------------------------------------------------------------
@torch.no_grad()
def collect_norm_activations(model, ds, prefer_device, max_batches=32):
    """返回 {layer_idx: {'attn': Tensor[in], 'mlp': Tensor[in]}}（每通道激活峰值）。"""
    layers = decoder_layers(model)
    if not layers:
        raise RuntimeError("未找到文本 decoder 的 .layers")

    targets = {}       # (layer_idx, kind) -> module
    for i, layer in enumerate(layers):
        for kind, (_projs, norm_names) in SCALE_GROUPS.items():
            for nm in norm_names:
                mod = getattr(layer, nm, None)
                if mod is not None:
                    targets[(i, kind)] = mod
                    break

    peaks = {}
    handles = []
    for key, mod in targets.items():
        def mk(k):
            def hook(m, inputs, output):
                x = output if isinstance(output, torch.Tensor) else None
                if x is None or x.dim() < 2:
                    return
                x = x.detach().float().reshape(-1, x.shape[-1])
                pk = x.abs().amax(dim=0).cpu()
                peaks[k] = torch.maximum(peaks[k], pk) if k in peaks else pk
            return hook
        handles.append(mod.register_forward_hook(mk(key)))

    target = dev.get_device(prefer_device)
    model.eval()
    with torch.no_grad():
        for n, ex in enumerate(ds):
            if n >= max_batches:
                break
            model(ex["input_ids"].unsqueeze(0).to(target))
    for h in handles:
        h.remove()

    out = {}
    for (i, kind), v in peaks.items():
        out.setdefault(i, {})[kind] = v
    return out


def find_norm_module(layer, kind):
    _projs, norm_names = SCALE_GROUPS[kind]
    for nm in norm_names:
        mod = getattr(layer, nm, None)
        if mod is not None:
            return nm, mod
    return None, None


# ----------------------------------------------------------------------
# 4) 构建量化 state_dict
# ----------------------------------------------------------------------
@torch.no_grad()
def build_state_dict(model, acts, bits=4, group_size=128, alpha=0.5, max_layers=0):
    src = model.state_dict()
    new = {}
    replaced = set()
    n_quant = 0
    layers = decoder_layers(model)

    for i, layer in enumerate(layers):
        if max_layers and i >= max_layers:
            break
        lin = dict(iter_linears_in(layer))
        if not lin:
            continue
        pfx = f"model.layers.{i}" if any(k.startswith("model.layers.") for k in src) else f"layers.{i}"

        # --- 计算缩放：attn 组（q/k/v 共享）、mlp 组（gate/up 共享）---
        scales = {}
        for kind, (projs, _norms) in SCALE_GROUPS.items():
            present = [p for p in projs if p in lin]
            if not present:
                continue
            nm, norm = find_norm_module(layer, kind)
            a = (acts.get(i) or {}).get(kind)
            if nm is None or norm is None or a is None or not hasattr(norm, "weight"):
                scales[kind] = None          # 无法缩放 → 退化为普通分组量化
                continue
            # 用该组所有投影权重一起估计 w_max（共享同一输入通道缩放）
            Wcat = torch.cat([lin[p].weight.detach().float() for p in present], dim=0)
            s = awq_channel_scale(Wcat.to(a.device), a, alpha=alpha).to(norm.weight.device)
            scales[kind] = s
            # ---- 折叠：把 1/s 折进前置 Norm 的 weight ----
            full_norm = f"{pfx}.{nm}"
            gamma = src[full_norm].detach().float()
            new[full_norm] = (gamma / s.to(gamma.device)).to(src[full_norm].dtype)
            replaced.add(full_norm)

        # --- 逐投影量化 ---
        for name, mod in lin.items():
            base = f"{pfx}.{name}"
            W = mod.weight.detach().float()
            kind = "attn" if any(k in name for k in ("q_proj", "k_proj", "v_proj")) else \
                   "mlp" if any(k in name for k in ("gate_proj", "up_proj")) else None
            s = scales.get(kind) if kind else None
            Wq = W * s.unsqueeze(0).to(W.device) if s is not None else W

            q, scale_q, zero = quant_group_asym(Wq, bits, group_size)
            qweight = pack_last_awq(q.t().contiguous())        # [in, out/8]
            qzeros = pack_last_awq(zero.t().contiguous())      # [gnum, out/8]
            scales_t = scale_q.t().contiguous().half()         # [gnum, out]
            new[f"{base}.qweight"] = qweight
            new[f"{base}.qzeros"] = qzeros
            new[f"{base}.scales"] = scales_t
            if mod.bias is not None:
                new[f"{base}.bias"] = mod.bias.detach().half()
            replaced.update({f"{base}.weight", f"{base}.bias"})
            n_quant += 1

    # --- 其余权重（embed / norm / lm_head / visual...）原样保留 ---
    seen = set()
    for k, v in src.items():
        if k in replaced:
            continue
        p = v.data_ptr()
        if p in seen:
            continue
        seen.add(p)
        new[k] = (v if v.dtype in (torch.float16, torch.bfloat16) else v.half()).clone()

    return new, n_quant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--calib", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--max-layers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    config.ensure_dirs()
    model_path = os.path.abspath(args.model) if args.model else config.require_model()
    stem = os.path.basename(os.path.normpath(model_path))
    out_dir = os.path.join(config.QUANT_DIR, f"{stem}-manual-awq-{args.bits}bit")

    print("=" * 72)
    print(f"手写 AWQ 量化 (bits={args.bits}, group={args.group_size}, alpha={args.alpha})")
    print(f"模型: {model_path}")
    print(f"设备: {dev.default_device(args.device)}")
    print("=" * 72)

    from utils.dataio import Dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    full = Dataset.from_parquet(os.path.join(config.CALIB_DIR, "validation.parquet"))
    rows = full.select(range(min(args.calib, len(full))))
    ds = [{"input_ids": torch.tensor(
        tok(r["text"], truncation=True, max_length=args.seq)["input_ids"], dtype=torch.long)}
        for r in rows]

    model = load_model(model_path, prefer_device=args.device, dtype="float32", eval_mode=True)
    print("\n[1/3] 采集归一化层输出激活 ...")
    acts = collect_norm_activations(model, ds, args.device)
    print(f"      采集到 {len(acts)} 层的激活统计")

    print("[2/3] 缩放 + 分组量化 + 折叠 ...")
    new_sd, n_quant = build_state_dict(model, acts, args.bits, args.group_size,
                                       args.alpha, args.max_layers)
    print(f"      量化 {n_quant} 个线性层，共 {len(new_sd)} 个张量")

    print("[3/3] 写出 checkpoint ...")
    os.makedirs(out_dir, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in new_sd.items()},
              os.path.join(out_dir, "model.safetensors"))

    src_cfg = os.path.join(model_path, "config.json")
    cfg = json.load(open(src_cfg, encoding="utf-8")) if os.path.isfile(src_cfg) else {}
    cfg["quantization_config"] = {
        "quant_method": "awq", "bits": args.bits, "group_size": args.group_size,
        "zero_point": True, "version": "gemm",
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    for fn in ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "merges.txt", "chat_template.jinja",
               "processor_config.json", "preprocessor_config.json"):
        p = os.path.join(model_path, fn)
        if os.path.isfile(p):
            shutil.copy(p, out_dir)
    with open(os.path.join(out_dir, "quantize_config.json"), "w", encoding="utf-8") as f:
        json.dump({"quant_method": "awq", "bits": args.bits, "group_size": args.group_size,
                   "zero_point": True, "n_linear": n_quant, "alpha": args.alpha,
                   "awq_order": AWQ_ORDER, "scale_folded_into_norm": True}, f, indent=2)

    print(f"\n[OK] 已写出 -> {out_dir}")
    print(f"     n_linear={n_quant}  alpha={args.alpha}  AWQ_ORDER={AWQ_ORDER}")
    print("     （1/s 已折叠进前置 Norm，推理零额外开销）")
    print("\n部署(NVIDIA): vllm serve <目录> --quantization awq_marlin --dtype float16")
    print("部署(昇腾)  : 昇腾对 INT4-AWQ 支持有限，建议主力走 W8A8；本模块用于")
    print("              展示算法实现 + 在 NVIDIA 侧验证可正确解码。")


if __name__ == "__main__":
    main()
