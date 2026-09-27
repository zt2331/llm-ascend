#!/usr/bin/env python
"""昇腾原生量化（msModelSlim / msmodelslim）—— 昇腾 NPU 推荐路径。

为什么用 msModelSlim：
  昇腾硬件对 **W8A8(INT8)** 支持最成熟（Cube 单元对 INT8 有原生支持），
  而 NVIDIA 生态更常用 AWQ/GPTQ 的 INT4。国产卡量化应走昇腾自己的工具链，
  产出的模型才能被 vllm-ascend / MindIE 正确加载。

API 随版本略有差异，本脚本做了多形态兼容尝试并给出清晰提示。
若本机未安装 msmodelslim，脚本会打印两种获取方式，并允许回退到
llm-compressor 路径（quantize/gen_llmcomp.py）。

用法（项目根）:
    python quantize/ascend_quant.py --scheme W8A8
    python quantize/ascend_quant.py --scheme W8A8 --model output_models/pruned/xxx
    python quantize/ascend_quant.py --scheme W8A8 --calib 32 --seq 2048
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import device as dev


def _pkg_version(dist):
    """从包元数据读版本（None 表示未安装）。"""
    try:
        from importlib.metadata import version, PackageNotFoundError
        return version(dist)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _scan_msmodelslim():
    """扫描已安装的 msmodelslim 包，找出定义 Calibrator/QuantConfig 的模块路径。

    昇腾 msModelSlim 各版本模块路径不统一（26.x 与早期差异较大），
    用它自动定位正确的导入路径，避免"装了却导不进来"。
    """
    import importlib.util
    import re

    try:
        spec = importlib.util.find_spec("msmodelslim")
    except Exception:
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    root = list(spec.submodule_search_locations)[0]
    found = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                src = open(p, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            hits = [c for c in ("Calibrator", "QuantConfig")
                    if re.search(rf"^\s*class\s+{c}\b", src, re.M)]
            if hits:
                rel = os.path.relpath(p, root)[:-3].replace(os.sep, ".")
                if rel.endswith(".__init__"):
                    rel = rel[: -len(".__init__")]
                found.append((f"msmodelslim.{rel}", hits))
    return found


def _print_api_help(has_module: bool):
    """打印 msModelSlim 的 API 定位帮助。"""
    if not has_module:
        print("\n  未检测到 msmodelslim 包本身。获取方式：")
        print("    1) 昇腾镜像是常见的自带来源")
        print("    2) gitcode.com/Ascend/msmodelslim 或 MindStudio 工具包")
        print("    3) 回退: python quantize/gen_llmcomp.py --scheme W8A8")
        return
    print("\n  已检测到 msmodelslim，但未找到预期的 Calibrator/QuantConfig。")
    print("  正在扫描包内实际 API ...")
    found = _scan_msmodelslim()
    if found:
        print("  在本机 msmodelslim 里发现以下定义：")
        for mod, hits in found[:10]:
            print(f"    {mod}   ->  {', '.join(hits)}")
        print("\n  按上面的路径调整本脚本顶部的导入，或参考其 docstring 用法。")
    else:
        print("  未扫描到 Calibrator/QuantConfig —— 该版本 API 可能已重构。")
        print('  查看位置: python -c "import msmodelslim,os;print(os.path.dirname(msmodelslim.__file__))"')
    print("\n  可直接使用的回退方案（无需 msmodelslim）：")
    print("    python quantize/gen_llmcomp.py --scheme W8A8     # llm-compressor")
    print("    python quantize/manual_awq.py                    # 手写 AWQ（纯 PyTorch）")
    print("    python quantize/manual_gptq.py                   # 手写 GPTQ（纯 PyTorch）")


def _import_msmodelslim():
    """尝试多种导入路径（不同版本包名有差异）。"""
    tries = [
        ("msmodelslim.pytorch.llm_ptq.llm_ptq_tools", "Calibrator", "QuantConfig"),
        ("msmodelslim.pytorch.llm_ptq", "Calibrator", "QuantConfig"),
        ("msmodelslim", "Calibrator", "QuantConfig"),
    ]
    # 动态补充：扫描包内真实路径（应对 26.x 等新版本路径变更）
    for mod, hits in _scan_msmodelslim():
        if "Calibrator" in hits and "QuantConfig" in hits:
            entry = (mod, "Calibrator", "QuantConfig")
            if entry not in tries:
                tries.insert(0, entry)

    for mod, cname, qname in tries:
        try:
            m = __import__(mod, fromlist=[cname, qname])
            return getattr(m, cname), getattr(m, qname), mod
        except Exception:
            continue
    return None, None, None


def build_calib(tok, calib_n, seq):
    from utils.dataio import Dataset
    pq = os.path.join(config.CALIB_DIR, "validation.parquet")
    full = Dataset.from_parquet(pq)
    rows = full.select(range(min(calib_n, len(full))))
    data = []
    for r in rows:
        ids = tok(r["text"], truncation=True, max_length=seq)["input_ids"]
        data.append(ids)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--scheme", default="W8A8", choices=["W8A8", "W8A16", "W4A16"])
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config.ensure_dirs()
    model_path = os.path.abspath(args.model) if args.model else config.require_model()
    stem = os.path.basename(os.path.normpath(model_path))
    out_dir = args.out or os.path.join(config.QUANT_DIR, f"{stem}-msmodelslim-{args.scheme}")

    print("=" * 72)
    print(f"昇腾原生量化 (msModelSlim)  scheme={args.scheme}")
    print(f"模型: {model_path}")
    print(f"设备: {dev.default_device('auto')}")
    print(f"输出: {out_dir}")
    print("=" * 72)

    Calibrator, QuantConfig, modname = _import_msmodelslim()
    if Calibrator is None:
        print("[FAIL] 未能导入 msmodelslim 的 Calibrator/QuantConfig")
        _has_ms = _pkg_version("msmodelslim") is not None
        _print_api_help(_has_ms)
        return 3

    print(f"[OK] 已导入 msmodelslim 于 {modname}")
    import torch
    from transformers import AutoTokenizer
    from utils.model_utils import load_model, quant_ignore_patterns

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = load_model(model_path, prefer_device="auto", eval_mode=True)
    calib = build_calib(tok, args.calib, args.seq)
    print(f"[INFO] 校准样本 {len(calib)} 条，最大长度 {args.seq}")

    # 不量化的层（lm_head / 视觉塔 / 混合线性注意力）
    # ★ msModelSlim 的 disable_names 只接受【具体 Linear 层】的名字，
    #   传入容器模块名（如 visual、visual.blocks）会报
    #   "invalid key `visual`"。所以这里只收集 nn.Linear 的叶子名字。
    import torch.nn as nn

    skip_linear_names = []
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if n.endswith("lm_head") or "visual" in n or "linear_attn" in n:
            skip_linear_names.append(n)
    print(f"[INFO] 需跳过的 Linear 层: {len(skip_linear_names)} 个"
          f"（lm_head / visual / linear_attn）")
    if skip_linear_names:
        print(f"       示例: {skip_linear_names[:3]}")

    # msModelSlim 各版本对名字前缀要求不同，准备多种格式依次尝试
    def _strip_prefix(names, prefix):
        out = []
        for n in names:
            out.append(n[len(prefix):] if prefix and n.startswith(prefix) else n)
        return out

    disable_variants = [
        ("完整 Linear 名", skip_linear_names),
        ("去掉 model. 前缀", _strip_prefix(skip_linear_names, "model.")),
        ("仅 lm_head", [n for n in skip_linear_names if n.endswith("lm_head")]),
        ("不跳过（仅调试）", []),
    ]

    w_bit, a_bit = (8, 8) if args.scheme == "W8A8" else (8, 16)
    if args.scheme == "W4A16":
        w_bit, a_bit = (4, 16)

    os.makedirs(out_dir, exist_ok=True)

    # ---- 多版本 API 兼容尝试 ----
    attempts = []

    def _try(fn, tag):
        try:
            fn()
            print(f"[OK] 量化成功（{tag}）")
            return True
        except TypeError as e:
            attempts.append(f"{tag}: TypeError {str(e)[:180]}")
        except Exception as e:
            attempts.append(f"{tag}: {type(e).__name__} {str(e)[:180]}")
        return False

    def make_api(disable, dev_type, use_calib):
        def api():
            kwargs = dict(w_bit=w_bit, a_bit=a_bit)
            if disable:
                kwargs["disable_names"] = disable
            if dev_type:
                kwargs["dev_type"] = dev_type
            qc = QuantConfig(**kwargs)
            if use_calib:
                cal = Calibrator(model, qc, calib_data=calib)
            else:
                cal = Calibrator(model, qc, calib_data=calib, disable_level="L0")
            cal.run()
            cal.save(out_dir, save_type=["safe_tensor"])
        return api

    dev_type = "npu" if dev.has_npu() else "cpu"
    plan = []
    for label, names in disable_variants:
        plan.append((make_api(names, dev_type, True), f"Calibrator(calib_data) + {label}"))
    # 再来一轮：不带 dev_type（用默认），兼容老版本
    for label, names in disable_variants[:2]:
        plan.append((make_api(names, None, False), f"Calibrator(disable_level) + {label}"))

    for fn, tag in plan:
        if _try(fn, tag):
            # 复制配套文件
            import shutil
            for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                      "merges.txt", "chat_template.jinja", "config.json"):
                src = os.path.join(model_path, f)
                dst = os.path.join(out_dir, f)
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy(src, dst)
            # ★ 保存后立刻自检：量化是否真的落盘（防止"打印 OK 但产物是 fp16"的静默失败）
            try:
                from utils.quant_check import inspect_quant_dir, format_report
                _info = inspect_quant_dir(out_dir)
                print("")
                print(format_report(_info, title="量化产物自检"))
                if not _info["ok"]:
                    print("[FAIL] 量化产物校验未通过 —— 该产物不可用于评测/部署，"
                          "请按上方提示排查后重新量化")
                    return 5
            except Exception as _e:
                print(f"[WARN] 自检执行失败: {_e}")

            print(f"[OK] 已写出 -> {out_dir}")
            print(f"\n部署: bash scripts/serve_ascend.sh {out_dir} ascend")
            return 0

    print("[FAIL] msModelSlim 调用均失败，尝试记录：")
    for a in attempts:
        print("   -", a)
    print(f"\n  已导入的 API 来自: {modname}")
    try:
        import inspect
        print(f"  QuantConfig 签名: {inspect.signature(QuantConfig)}")
        print(f"  Calibrator  签名: {inspect.signature(Calibrator.__init__)}")
    except Exception as e:
        print(f"  （无法打印签名: {e}）")
    print("\n  请对照上面的真实签名调整调用参数；或直接用回退方案：")
    print("    python quantize/gen_llmcomp.py --scheme W8A8     # llm-compressor")
    print("    python quantize/manual_awq.py                    # 手写 AWQ（纯 PyTorch）")
    print("    python quantize/manual_gptq.py                   # 手写 GPTQ（纯 PyTorch）")
    return 4


if __name__ == "__main__":
    sys.exit(main())
