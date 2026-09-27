#!/usr/bin/env python
"""结构化剪枝（ShortGPT 块重要度）—— 昇腾 NPU 版。

原理：块重要度 = 层输入与层输出 hidden 的**余弦相似度**。
      相似度越高 → 该层对信息流改变越小 → 越冗余 → 删除。
      （只删「文本 decoder」的层；多模态模型视觉塔保持不动。）

用法（项目根）:
    python prune/prune.py --keep 48                 # 从 N 层保留 48 层
    python prune/prune.py --keep 48 --calib 16 --seq 512
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import device as dev
from utils.model_utils import load_model, decoder_layers, text_decoder


@torch.no_grad()
def block_importance(model, ds, prefer_device):
    """对文本 decoder 每层算 输入/输出 余弦相似度。"""
    layers = decoder_layers(model)
    if not layers:
        raise RuntimeError("未找到文本 decoder 的 .layers，请检查模型结构")

    in_cache, out_cache, handles = {}, {}, []

    def mk(i):
        def inh(m, inputs):                       # forward_pre_hook: (module, input)
            x = inputs[0]
            if isinstance(x, torch.Tensor):
                in_cache[i] = x.detach().float().reshape(-1, x.shape[-1]).cpu()
        def outh(m, inputs, out):                 # forward_hook: (module, input, output)
            if isinstance(out, torch.Tensor):
                out_cache[i] = out.detach().float().reshape(-1, out.shape[-1]).cpu()
        return inh, outh

    for i, layer in enumerate(layers):
        hin, hout = mk(i)
        handles.append(layer.register_forward_pre_hook(hin))
        handles.append(layer.register_forward_hook(hout))

    model.eval()
    target = dev.get_device(prefer_device)
    with torch.no_grad():
        for ex in ds:
            model(ex["input_ids"].unsqueeze(0).to(target))
    for h in handles:
        h.remove()

    sims = {}
    for i in in_cache:
        a, b = in_cache[i], out_cache[i]
        cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-8)
        sims[i] = float(cos.mean())
    return sims


def prune_and_save(model_path, sims, keep, out_dir, tok, prefer_device):
    n = len(sims)
    ordered = sorted(range(n), key=lambda i: sims[i])   # 相似度低=重要
    keep_idx = sorted(ordered[:keep])
    dropped = [i for i in range(n) if i not in set(keep_idx)]
    print(f"[剪枝] 共 {n} 层 → 保留 {len(keep_idx)} 层，删除 {len(dropped)} 层: {dropped}")

    model = load_model(model_path, prefer_device="cpu", eval_mode=True)
    dec = text_decoder(model)
    dec.layers = nn.ModuleList([dec.layers[i] for i in keep_idx])

    cfg = getattr(dec, "config", None) or getattr(model, "config", None)

    # ============ 同步配置（关键，写错会导致模型结构错乱）============
    # Qwen3.6/3.5 是「混合线性注意力」模型：
    #   layer_types = [linear_attention ×3, full_attention ×1] 重复（48 + 16 = 64 层）
    # 而剪枝保留的是【最重要的层，索引任意】，不是前缀！
    # 所以 layer_types 必须【按实际保留的索引重排】，绝不能简单截断前缀。
    new_types = {}
    for field in ("layer_types", "decoder_layer_types", "hidden_layers"):
        val = getattr(cfg, field, None)
        if isinstance(val, (list, tuple)) and len(val) == n:
            new_types[field] = [val[i] for i in keep_idx]      # ← 按保留索引重排
            setattr(cfg, field, new_types[field])

    if new_types:
        f0 = list(new_types)[0]
        from collections import Counter
        print(f"[剪枝] 重排 {f0}: {dict(Counter(new_types[f0]))}")

    cfg.num_hidden_layers = len(keep_idx)
    if hasattr(model, "config"):
        model.config.num_hidden_layers = len(keep_idx)

    # full_attention_interval：若存在且新层类型不再是固定周期，必须去掉，
    # 否则模型会按「每 N 层一个 full attention」重新推导，与实际 layer_types 冲突。
    fai = getattr(cfg, "full_attention_interval", None)
    if fai and new_types.get("layer_types"):
        lt = new_types["layer_types"]
        periodic = all(lt[i] == ("full_attention" if (i + 1) % fai == 0
                                 else "linear_attention") for i in range(len(lt)))
        if not periodic:
            setattr(cfg, "full_attention_interval", None)
            print(f"[剪枝] 层类型不再满足每 {fai} 层的周期 → 已清空 full_attention_interval")
        else:
            print(f"[剪枝] 层类型仍满足每 {fai} 层周期，保留 full_attention_interval")

    # MTP（多 token 预测）层是独立的 mtp.*，与文本层数无关；
    # 若保持开启但其权重未随剪枝保存，加载会缺参 → 这里显式关闭并提示。
    mtp = getattr(cfg, "mtp_num_hidden_layers", None)
    if mtp:
        print(f"[剪枝] 检测到 MTP(mtp_num_hidden_layers={mtp})："
              f"其权重在 mtp.* 命名空间，与文本层剪枝无关。")
        print("       若后续加载报缺少 mtp.* 权重，可设为 0 关闭 MTP。")

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "prune_info.json"), "w", encoding="utf-8") as f:
        json.dump({"method": "shortgpt_block_importance",
                   "original_layers": n, "kept_layers": len(keep_idx),
                   "kept_idx": keep_idx, "dropped_idx": dropped,
                   "block_similarity": {str(k): round(v, 4) for k, v in sims.items()}},
                  f, indent=2, ensure_ascii=False)
    print(f"[剪枝] 已保存 -> {out_dir}")
    return out_dir, sims, keep_idx, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=48, help="保留层数（默认 48）")
    ap.add_argument("--calib", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    config.ensure_dirs()
    model_path = config.require_model()
    from utils.dataio import Dataset
    from transformers import AutoTokenizer

    print(f"[剪枝] 模型: {model_path}  设备: {dev.default_device(args.device)}")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    pq = os.path.join(config.CALIB_DIR, "validation.parquet")
    full = Dataset.from_parquet(pq)
    df = full.select(range(min(args.calib, len(full))))
    ds = [{"input_ids": torch.tensor(
        tok(t["text"], truncation=True, max_length=args.seq)["input_ids"], dtype=torch.long)}
        for t in df]

    model = load_model(model_path, prefer_device=args.device)
    sims = block_importance(model, ds, args.device)
    del model
    dev.empty_cache(args.device)

    out_dir = os.path.join(config.PRUNE_DIR, f"pruned-{config.MODEL_TAG}-{args.keep}layers")
    prune_and_save(model_path, sims, args.keep, out_dir, tok, args.device)
    print(f"\n下一步: 量化 python quantize/gen_llmcomp.py --model {out_dir}")


if __name__ == "__main__":
    main()
