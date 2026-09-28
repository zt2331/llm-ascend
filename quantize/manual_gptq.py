#!/usr/bin/env python
"""从零手写 GPTQ 量化 → 产出「vLLM 可直接加载」的 GPTQ checkpoint。

算法（不依赖任何第三方量化库）：
  1. 校准：收集每个 Linear 的输入激活 X，构造 Hessian  H = XᵀX  （二阶信息）；
  2. 阻尼：H ← H + λ·mean(diag(H))·I，避免病态/不可逆；死通道置零；
  3. 逐列贪心量化 + **误差补偿**：
       量化第 j 列 → 误差 err = (w_j - ŵ_j) / H⁻¹[j,j]
       把 err 按 H⁻¹[j, j:] 传播给后面未量化的列，使其提前"预补偿"；
  4. per-group(128) 非对称量化，逐块（block=128）处理；
  5. 按 vLLM GPTQ 布局打包：qweight[in/8, out] / qzeros[gnum, out/8] /
     scales[gnum, out] / g_idx[in]。

> 与 AWQ 的区别：AWQ 靠"激活显著性 + 缩放"保护重要通道；
> GPTQ 靠"二阶 Hessian + 逐列误差补偿"逼近最优量化误差，通常精度略高但更慢。

产出格式：GPTQ（sym=False），vLLM 用 `--quantization gptq_marlin` 加载。

用法（项目根）:
    python quantize/manual_gptq.py                    # 全部文本 decoder 层 4bit
    python quantize/manual_gptq.py --max-layers 2     # 快速试跑
    python quantize/manual_gptq.py --model output_models/pruned/xxx --bits 4
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
from utils.model_utils import load_model, decoder_layers, iter_decoder_linears, iter_linears_in

PROJ_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ----------------------------------------------------------------------
# 1) 采集每个 Linear 的输入激活（构造 Hessian 用）
# ----------------------------------------------------------------------
@torch.no_grad()
def collect_linear_inputs(model, ds, prefer_device, max_batches=32):
    """返回 {linear_full_name: Tensor[n_samples, in]}（CPU，float32）。"""
    bufs = {}
    handles = []
    for name, mod in iter_decoder_linears(model, PROJ_SUFFIXES):
        def mk(nm):
            def hook(m, inputs, output):
                x = inputs[0]
                if not isinstance(x, torch.Tensor) or x.dim() < 2:
                    return
                x2 = x.detach().float().reshape(-1, x.shape[-1]).cpu()
                bufs[nm] = torch.cat([bufs[nm], x2], dim=0) if nm in bufs else x2
            return hook
        handles.append(mod.register_forward_hook(mk(name)))

    target = dev.get_device(prefer_device)
    model.eval()
    with torch.no_grad():
        for n, ex in enumerate(ds):
            if n >= max_batches:
                break
            model(ex["input_ids"].unsqueeze(0).to(target))
    for h in handles:
        h.remove()
    return bufs


# ----------------------------------------------------------------------
# 2) GPTQ 核心：Hessian + 逐列误差补偿
# ----------------------------------------------------------------------
@torch.no_grad()
def gptq_quantize(W, X, bits=4, group_size=128, percdamp=0.01, blocksize=128):
    """W:[out,in]  X:[n,in] → (Q_int[out,in] int, Scale[out,gnum], Zero[out,gnum])。"""
    out_f, in_f = W.shape
    W = W.clone().float()
    qmin, qmax = 0, (1 << bits) - 1
    gnum = (in_f + group_size - 1) // group_size

    # ---- Hessian ----
    if X is None or X.shape[0] < 2:
        H = torch.eye(in_f)
    else:
        X = X.float()
        if X.shape[0] > 4096:                     # 控制显存
            X = X[torch.randperm(X.shape[0])[:4096]]
        H = X.t() @ X
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0                              # 死通道置 0，避免 NaN

    damp = percdamp * torch.mean(torch.diag(H))
    idx = torch.arange(in_f)
    H[idx, idx] += damp

    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except Exception:
        Hinv = torch.linalg.pinv(H)

    # ---- 逐块 / 逐列 ----
    Q = torch.zeros(out_f, in_f)
    Scale = torch.zeros(out_f, gnum)
    Zero = torch.zeros(out_f, gnum)

    def _group_params(grp):                       # grp:[out, g]
        mn = grp.min(dim=1).values
        mx = grp.max(dim=1).values
        s = ((mx - mn) / (qmax - qmin)).clamp_min(1e-8)
        z = torch.round(qmin - mn / s).clamp(qmin, qmax)
        return s, z

    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        cnt = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(cnt):
            col = i1 + i
            g = col // group_size
            w = W1[:, i]
            d = Hinv1[i, i]
            # 用"当前（已被补偿过的）"该组权重重新估计量化参数
            gs, ge = g * group_size, min((g + 1) * group_size, in_f)
            scale, zero = _group_params(W[:, gs:ge])
            Scale[:, g] = scale
            Zero[:, g] = zero
            q = torch.round(w / scale + zero).clamp(qmin, qmax)
            dq = (q - zero) * scale
            Q1[:, i] = dq
            err = (w - dq) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err

        Q[:, i1:i2] = Q1
        if i2 < in_f:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    # 最终量化整数（按最终 scale/zero 重算，保证打包一致）
    Q_int = torch.zeros(out_f, in_f, dtype=torch.int32)
    for g in range(gnum):
        gs, ge = g * group_size, min((g + 1) * group_size, in_f)
        s = Scale[:, g:g + 1].clamp_min(1e-8)
        z = Zero[:, g:g + 1]
        Q_int[:, gs:ge] = torch.round(Q[:, gs:ge] / s + z).clamp(qmin, qmax).to(torch.int32)
    return Q_int, Scale, Zero


# ----------------------------------------------------------------------
# 3) vLLM GPTQ 打包
# ----------------------------------------------------------------------
def pack_gptq(Q_int, Scale, Zero, bits=4, group_size=128):
    """→ (qweight[in/8,out], qzeros[gnum,out/8], scales[gnum,out], g_idx[in]) int32/fp16。"""
    out_f, in_f = Q_int.shape
    gnum = (in_f + group_size - 1) // group_size
    in_f_pad = ((in_f + 7) // 8) * 8
    if in_f_pad != in_f:
        Q_int = torch.nn.functional.pad(Q_int, (0, in_f_pad - in_f))

    # qweight: 沿输入维每 8 个打包，布局 [in/8, out]
    qw = torch.zeros(in_f_pad // 8, out_f, dtype=torch.int32)
    for k in range(8):
        qw |= (Q_int[:, k::8].t().to(torch.int32) & 0xF) << (4 * k)

    # qzeros: 沿输出维每 8 个打包，布局 [gnum, out/8]
    out_pad = ((out_f + 7) // 8) * 8
    z = Zero.t().to(torch.int32)                       # [gnum, out]
    if out_pad != out_f:
        z = torch.nn.functional.pad(z, (0, out_pad - out_f))
    qz = torch.zeros(gnum, out_pad // 8, dtype=torch.int32)
    for k in range(8):
        qz |= (z[:, k::8] & 0xF) << (4 * k)

    scales = Scale.t().contiguous().half()             # [gnum, out]
    g_idx = torch.arange(in_f_pad, dtype=torch.int32) // group_size
    return qw.contiguous(), qz.contiguous(), scales, g_idx


# ----------------------------------------------------------------------
# 4) 主流程
# ----------------------------------------------------------------------
@torch.no_grad()
def build_state_dict(model, xbuf, bits=4, group_size=128, percdamp=0.01,
                     blocksize=128, max_layers=0):
    src = model.state_dict()
    new = {}
    replaced = set()
    n_quant = 0
    layers = decoder_layers(model)
    pfx = "model.layers" if any(k.startswith("model.layers.") for k in src) else "layers"

    for i, layer in enumerate(layers):
        if max_layers and i >= max_layers:
            break
        for name, mod in iter_linears_in(layer, PROJ_SUFFIXES):
            base = f"{pfx}.{i}.{name}"
            X = xbuf.get(base)
            # ★ 必须把权重也搬到 CPU：
            #  (1) 激活 X 本来就是 CPU（collect_linear_inputs 里 .cpu()，为省显存），
            #      而 gptq_quantize 内部的 Q/Scale/Zero/H/Hinv 都在 CPU 上创建；
            #      若 W 留在 NPU，会与它们 device mismatch（err=(w-dq)/d 等处直接报错）。
            #  (2) 昇腾 NPU 没有实现 aten::cholesky_inverse，torch_npu 的 CPU fallback
            #      在本环境会抛 "Allocator for npu is not a DeviceAllocator"。
            #      全部放 CPU 就完全不触发 fallback，也顺带更快。
            #  Hessian 是 [in,in] 方阵（5120²≈105MB / 17408²≈1.2GB），CPU 完全够用。
            Q_int, Scale, Zero = gptq_quantize(mod.weight.detach().float().cpu(), X,
                                               bits, group_size, percdamp, blocksize)
            qw, qz, sc, gidx = pack_gptq(Q_int, Scale, Zero, bits, group_size)
            new[f"{base}.qweight"] = qw
            new[f"{base}.qzeros"] = qz
            new[f"{base}.scales"] = sc
            new[f"{base}.g_idx"] = gidx
            if mod.bias is not None:
                new[f"{base}.bias"] = mod.bias.detach().half()
            replaced.update({f"{base}.weight", f"{base}.bias"})
            n_quant += 1

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
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--blocksize", type=int, default=128)
    ap.add_argument("--calib", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--max-layers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    config.ensure_dirs()
    model_path = os.path.abspath(args.model) if args.model else config.require_model()
    stem = os.path.basename(os.path.normpath(model_path))
    out_dir = os.path.join(config.QUANT_DIR, f"{stem}-manual-gptq-{args.bits}bit")

    print("=" * 72)
    print(f"手写 GPTQ 量化 (bits={args.bits}, group={args.group_size}, "
          f"damp={args.percdamp})")
    print(f"模型: {model_path}")
    print(f"设备: {dev.default_device(args.device)}")
    print("=" * 72)

    from utils.dataio import Dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    full = Dataset.from_parquet(config.CALIB_PARQUET)
    rows = full.select(range(min(args.calib, len(full))))
    ds = [{"input_ids": torch.tensor(
        tok(r["text"], truncation=True, max_length=args.seq)["input_ids"], dtype=torch.long)}
        for r in rows]

    # ★ 同 manual_awq：不要用 float32 加载整个模型（27B×4B≈108GB 必 OOM）。
    #   Hessian 构造与误差补偿都在内部 .float() 逐层算，模型保持 bf16 即可。
    model = load_model(model_path, prefer_device=args.device, dtype="bfloat16", eval_mode=True)

    # ★ GPTQ 的 Hessian 需要“每个线性层的全部校准激活”同时驻留 CPU，
    #   27B 上很容易吃掉几十 GB 内存。先估算并预警，避免跑到一半被 OOM killer 干掉。
    n_tok = sum(int(e["input_ids"].numel()) for e in ds)
    est_gb = sum(n_tok * m.in_features * 4
                 for _, m in iter_decoder_linears(model, PROJ_SUFFIXES)) / 1024 ** 3
    print(f"[预估] 激活缓冲 {est_gb:.1f} GB（{len(ds)} 条 × 最多 {args.seq} token，fp32）")
    if est_gb > 24:
        sug = max(2, int(args.calib * 16 / max(est_gb, 1e-6)))
        print(f"[WARN] 激活缓冲偏大，可能耗尽内存 → 建议 --calib {sug}，"
              f"或先剪枝再跑 GPTQ")

    print("\n[1/3] 采集 Linear 输入激活（构造 Hessian）...")
    xbuf = collect_linear_inputs(model, ds, args.device)
    print(f"      采集到 {len(xbuf)} 个线性层的输入激活")

    print("[2/3] GPTQ 逐列量化 + 误差补偿 + 打包 ...")
    new_sd, n_quant = build_state_dict(model, xbuf, args.bits, args.group_size,
                                       args.percdamp, args.blocksize, args.max_layers)
    print(f"      量化 {n_quant} 个线性层，共 {len(new_sd)} 个张量")

    print("[3/3] 写出 checkpoint ...")
    os.makedirs(out_dir, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in new_sd.items()},
              os.path.join(out_dir, "model.safetensors"))

    src_cfg = os.path.join(model_path, "config.json")
    cfg = json.load(open(src_cfg, encoding="utf-8")) if os.path.isfile(src_cfg) else {}
    cfg["quantization_config"] = {
        "quant_method": "gptq", "bits": args.bits, "group_size": args.group_size,
        "desc_act": False, "sym": False,
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
        json.dump({"quant_method": "gptq", "bits": args.bits, "group_size": args.group_size,
                   "desc_act": False, "sym": False, "percdamp": args.percdamp,
                   "blocksize": args.blocksize, "n_linear": n_quant}, f, indent=2)

    print(f"\n[OK] 已写出 -> {out_dir}   n_linear={n_quant}")
    print("\n部署(NVIDIA): vllm serve <目录> --quantization gptq_marlin --dtype float16")
    print("部署(昇腾)  : 昇腾对 INT4-GPTQ 支持有限，建议主力走 W8A8；本模块用于")
    print("              展示算法实现 + 在 NVIDIA 侧验证可正确解码。")


if __name__ == "__main__":
    main()
