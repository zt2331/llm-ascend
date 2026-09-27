#!/usr/bin/env python
"""调库量化（llm-compressor）—— 昇腾 NPU 版，支持 SmoothQuant / AWQ / GPTQ / RTN。

四种方法统一入口（都由 llm-compressor 实现）：
  ┌────────┬──────────────────────────────────────────────┬─────────────────────┐
  │ method │ 做法                                          │ 典型 scheme         │
  ├────────┼──────────────────────────────────────────────┼─────────────────────┤
  │ smooth │ **平滑量化 SmoothQuant**：把激活的离群大值      │ W8A8（昇腾主力）     │
  │        │ 迁移到权重  Y=(X/s)(W·s)ᵀ，让 8bit 激活不溢出  │                     │
  │        │ SmoothQuantModifier + QuantizationModifier     │                     │
  │ awq    │ 激活感知：按输入通道缩放保护重要通道            │ W4A16(/W4A16_ASYM)  │
  │ gptq   │ Hessian 二阶 + 逐列误差补偿                    │ W4A16 / W8A16       │
  │ rtn    │ 直接四舍五入（无校准补偿），精度最差，仅作基线   │ W8A8 / W4A16        │
  └────────┴──────────────────────────────────────────────┴─────────────────────┘

★ smooth / awq / gptq 都需要校准数据（本脚本已接 data/calib）。
★ 量化范围自动跳过 lm_head / 视觉塔 / linear_attn（utils.model_utils 生成 ignore）。

用法（项目根）:
    python quantize/gen_llmcomp.py --method smooth --scheme W8A8            # 平滑量化 W8A8
    python quantize/gen_llmcomp.py --method smooth --alpha 0.75            # 调迁移强度
    python quantize/gen_llmcomp.py --method awq  --bits 4                  # AWQ 4bit
    python quantize/gen_llmcomp.py --method gptq --bits 4                  # GPTQ 4bit
    python quantize/gen_llmcomp.py --method gptq --bits 8                  # GPTQ 8bit
    python quantize/gen_llmcomp.py --method rtn  --scheme W8A8             # RTN 基线
    python quantize/gen_llmcomp.py --model output_models/pruned/xxx --method awq --bits 4

说明：`quantize/w8a8_smooth.py` 是 SmoothQuant 的**专用入口**（等价于
      `--method smooth --scheme W8A8`），保留它是为了保持原项目结构清晰。
"""
import argparse
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import device as dev
from utils.dataio import Dataset
from utils.model_utils import quant_ignore_patterns


def build_calib_loader(tok, calib_rows, seq):
    from torch.utils.data import DataLoader

    class _DS(torch.utils.data.Dataset):
        def __init__(self):
            self.enc = [tok(r["text"], truncation=True, max_length=seq)["input_ids"]
                        for r in calib_rows]

        def __len__(self):
            return len(self.enc)

        def __getitem__(self, i):
            return {"input_ids": torch.tensor(self.enc[i], dtype=torch.long)}

    return DataLoader(_DS(), batch_size=1, shuffle=False)


def _import_awq_modifier():
    """AWQModifier 在不同 llm-compressor 版本里路径不同，逐个尝试。

    0.13+ 推荐 `llmcompressor.modifiers.transform.awq`；
    旧的 `llmcompressor.modifiers.awq` 是兼容 shim（会返回 list，且有 DeprecationWarning）。
    因此优先试新路径。
    """
    for path in ("llmcompressor.modifiers.transform.awq",
                 "llmcompressor.modifiers.awq",
                 "llmcompressor.modifiers.quantization.awq"):
        try:
            m = __import__(path, fromlist=["AWQModifier"])
            return getattr(m, "AWQModifier")
        except Exception:
            continue
    return None


def _flatten(x):
    """把可能是 list/嵌套 list 的 modifier 序列拉平。"""
    out = []
    for i in (x if isinstance(x, (list, tuple)) else [x]):
        out.extend(_flatten(i) if isinstance(i, (list, tuple)) else [i])
    return out


def build_recipe(method, scheme, group_size, ignore, alpha=0.75):
    """构造 recipe（modifier 对象列表）。

    注意各版本 API 差异：
      * 0.13+ ：AWQModifier 只接收 duo_scaling 等；量化参数放在 QuantizationModifier
      * 0.13+ ：GPTQModifier 不接受 group_size（由 scheme 决定，W4A16 默认 128）
    """
    from llmcompressor.modifiers.quantization.quantization import QuantizationModifier

    # ---------------- GPTQ ----------------
    if method == "gptq":
        from llmcompressor.modifiers.gptq import GPTQModifier
        last = None
        plans = [
            (dict(targets=["Linear"], scheme=scheme, ignore=ignore, group_size=group_size),
             f"GPTQModifier(scheme={scheme}, group_size={group_size})"),
            (dict(targets=["Linear"], scheme=scheme, ignore=ignore),
             f"GPTQModifier(scheme={scheme})  # group_size 由 scheme 决定"),
            (dict(scheme=scheme, ignore=ignore),
             f"GPTQModifier(scheme={scheme})"),
        ]
        for kwargs, desc in plans:
            try:
                m = GPTQModifier(**kwargs)
                print(f"  [recipe] {desc}")
                return [m]
            except Exception as e:
                last = e
        raise RuntimeError(f"GPTQModifier 构造失败（尝试 3 种签名）: {last}")

    # ---------------- AWQ ----------------
    if method == "awq":
        AWQ = _import_awq_modifier()
        if AWQ is None:
            raise RuntimeError(
                "当前 llm-compressor 未找到 AWQModifier。排查：\n"
                "  python -c \"import llmcompressor,os;print(os.path.dirname(llmcompressor.__file__))\"\n"
                "  find <该目录> -name '*awq*'\n"
                "替代: --method gptq 或 --method rtn")
        awq_scheme = scheme if ("ASYM" in scheme or "SYM" in scheme) else "W4A16_ASYM"
        last = None
        plans = [
            # 新 API：AWQModifier 只负责算缩放
            (dict(duo_scaling="both"), True, f"AWQModifier(duo_scaling='both') + QuantizationModifier({awq_scheme})"),
            (dict(duo_scaling=True), True, f"AWQModifier(duo_scaling=True) + QuantizationModifier({awq_scheme})"),
            (dict(), True, f"AWQModifier() + QuantizationModifier({awq_scheme})"),
            # 旧 API：兼容 shim，自己就会返回 [AWQModifier, QuantizationModifier]
            (dict(scheme=awq_scheme, ignore=ignore, targets=["Linear"]), False,
             f"AWQModifier(旧 shim, scheme={awq_scheme})"),
            (dict(ignore=ignore), False, "AWQModifier(旧 shim)"),
        ]
        for kwargs, add_quant, desc in plans:
            try:
                mods = _flatten(AWQ(**kwargs))
                if add_quant:
                    mods.append(QuantizationModifier(targets=["Linear"],
                                                     scheme=awq_scheme, ignore=ignore))
                print(f"  [recipe] {desc}   （共 {len(mods)} 个 modifier）")
                return mods
            except Exception as e:
                last = e
        raise RuntimeError(f"AWQModifier 构造失败（尝试 {len(plans)} 种签名）: {last}")

    # ---------------- SmoothQuant（平滑量化）----------------
    if method == "smooth":
        from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
        last = None
        for kwargs in (dict(smoothing_strength=alpha), dict()):
            try:
                sq = SmoothQuantModifier(**kwargs)
                mods = [sq, QuantizationModifier(targets=["Linear"],
                                                 scheme=scheme, ignore=ignore)]
                print(f"  [recipe] SmoothQuantModifier(smoothing_strength="
                      f"{kwargs.get('smoothing_strength', '默认')}) + "
                      f"QuantizationModifier({scheme})")
                return mods
            except Exception as e:
                last = e
        raise RuntimeError(f"SmoothQuantModifier 构造失败: {last}")

    # ---------------- RTN 基线 ----------------
    print(f"  [recipe] QuantizationModifier(scheme={scheme})  [RTN 基线]")
    return [QuantizationModifier(targets=["Linear"], scheme=scheme, ignore=ignore)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="待量化模型目录（默认 base）")
    ap.add_argument("--method", default="awq", choices=["smooth", "awq", "gptq", "rtn"],
                    help="smooth(平滑量化 SmoothQuant) / awq(激活感知) / "
                         "gptq(二阶误差补偿) / rtn(朴素基线)")
    ap.add_argument("--alpha", type=float, default=0.75,
                    help="SmoothQuant 迁移强度 0~1（越大越把量化难度给权重）")
    ap.add_argument("--scheme", default=None, help="显式指定 scheme；不填按 method+bits 推断")
    ap.add_argument("--bits", type=int, default=4, help="位宽（awq/gptq 默认 4）")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config.ensure_dirs()
    model_path = os.path.abspath(args.model) if args.model else config.require_model()

    if args.scheme:
        scheme = args.scheme
    elif args.method in ("rtn", "smooth"):
        scheme = "W8A8"
    else:
        scheme = "W4A16" if args.bits == 4 else "W8A16"

    stem = os.path.basename(os.path.normpath(model_path))
    out_dir = args.out or os.path.join(config.QUANT_DIR, f"{stem}-llmcomp-{args.method}-{scheme}")

    print("=" * 74)
    print(f"llm-compressor 量化   method={args.method}   scheme={scheme}")
    print(f"模型: {model_path}")
    print(f"设备: {dev.default_device('auto')}")
    print(f"输出: {out_dir}")
    print("=" * 74)

    try:
        from llmcompressor import oneshot
    except Exception as e:
        print(f"[FAIL] llmcompressor 不可用: {e}")
        return 3

    from transformers import AutoTokenizer

    ignore = quant_ignore_patterns(model_path)
    print(f"\nignore = {ignore}")

    try:
        recipe = build_recipe(args.method, scheme, args.group_size, ignore,
                          alpha=args.alpha)
    except Exception as e:
        print(f"[FAIL] 构造 recipe 失败: {e}")
        return 3

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    full = Dataset.from_parquet(os.path.join(config.CALIB_DIR, "validation.parquet"))
    rows = [r for r in full.select(range(min(args.calib, len(full))))]
    loader = build_calib_loader(tok, rows, args.seq)
    print(f"校准样本 {len(rows)} 条，最大长度 {args.seq}\n")

    os.makedirs(out_dir, exist_ok=True)
    try:
        oneshot(model=model_path, output_dir=out_dir, recipe=recipe, dataset=loader,
                precision="auto", trust_remote_code_model=True, save_compressed=True)
    except Exception as e:
        print(f"[FAIL] 量化失败: {type(e).__name__}: {str(e)[:500]}")
        print("       可尝试: --method rtn（不依赖校准补偿）或 python quantize/ascend_quant.py")
        return 4

    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
               "chat_template.jinja", "processor_config.json", "preprocessor_config.json"):
        src = os.path.join(model_path, fn)
        dst = os.path.join(out_dir, fn)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy(src, dst)

    cfg_p = os.path.join(out_dir, "config.json")
    if os.path.isfile(cfg_p):
        cfg = json.load(open(cfg_p, encoding="utf-8"))
        print(f"\n[OK] quantization_config = "
              f"{json.dumps(cfg.get('quantization_config'), ensure_ascii=False)[:200]}")
        print(f"[OK] architectures = {cfg.get('architectures')}")
    print(f"[OK] 已写出 -> {out_dir}")
    print(f"\n部署: bash scripts/serve_ascend.sh {out_dir} compressed-tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
