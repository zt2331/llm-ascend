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


def _import_msmodelslim():
    """尝试多种导入路径（不同版本包名有差异）。"""
    tries = [
        ("msmodelslim.pytorch.llm_ptq.llm_ptq_tools", "Calibrator", "QuantConfig"),
        ("msmodelslim.pytorch.llm_ptq", "Calibrator", "QuantConfig"),
        ("msmodelslim", "Calibrator", "QuantConfig"),
    ]
    for mod, cname, qname in tries:
        try:
            m = __import__(mod, fromlist=[cname, qname])
            return getattr(m, cname), getattr(m, qname), mod
        except Exception:
            continue
    return None, None, None


def build_calib(tok, calib_n, seq):
    from datasets import Dataset
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
    out_dir = args.out or os.path.join(config.QUANT_DIR, f"{stem}-ascend-{args.scheme}")

    print("=" * 72)
    print(f"昇腾原生量化 (msModelSlim)  scheme={args.scheme}")
    print(f"模型: {model_path}")
    print(f"设备: {dev.default_device('auto')}")
    print(f"输出: {out_dir}")
    print("=" * 72)

    Calibrator, QuantConfig, modname = _import_msmodelslim()
    if Calibrator is None:
        print("[FAIL] 未找到 msmodelslim。获取方式（任选）：")
        print("  1) 昇腾镜像/开发环境通常自带：检查 /usr/local/Ascend 下的 MindStudio 工具")
        print("  2) 从昇腾社区下载 msModelSlim 工具包后：")
        print("       pip install msmodelslim-*.whl")
        print("  3) 回退到 llm-compressor 路径：")
        print("       python quantize/gen_llmcomp.py --scheme W8A8")
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
    disable_names = []
    try:
        for n, _m in model.named_modules():
            if n.endswith("lm_head") or "visual" in n or "linear_attn" in n:
                disable_names.append(n)
    except Exception:
        pass
    print(f"[INFO] disable_names 数量 = {len(disable_names)}")

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
            attempts.append(f"{tag}: TypeError {str(e)[:160]}")
        except Exception as e:
            attempts.append(f"{tag}: {type(e).__name__} {str(e)[:160]}")
        return False

    def api_v1():
        qc = QuantConfig(w_bit=w_bit, a_bit=a_bit, disable_names=disable_names,
                         dev_type="npu" if dev.has_npu() else "cpu", dev_id=0)
        cal = Calibrator(model, qc, calib_data=calib)
        cal.run()
        cal.save(out_dir, save_type=["safe_tensor"])

    def api_v2():
        qc = QuantConfig(w_bit=w_bit, a_bit=a_bit)
        cal = Calibrator(model, qc, calib_data=calib, disable_names=disable_names)
        cal.run()
        cal.save(out_dir, save_type=["safe_tensor"])

    def api_v3():
        qc = QuantConfig(w_bit=w_bit, a_bit=a_bit, disable_names=disable_names)
        cal = Calibrator(model, qc, calib_data=calib, disable_level="L0")
        cal.run()
        cal.save(out_dir, save_type=["safe_tensor"], part_file_size=None)

    for fn, tag in ((api_v1, "QuantConfig(dev_type)+Calibrator(calib_data)"),
                    (api_v2, "Calibrator(disable_names)"),
                    (api_v3, "Calibrator(disable_level)")):
        if _try(fn, tag):
            # 复制配套文件
            import shutil
            for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                      "merges.txt", "chat_template.jinja", "config.json"):
                src = os.path.join(model_path, f)
                dst = os.path.join(out_dir, f)
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy(src, dst)
            print(f"[OK] 已写出 -> {out_dir}")
            print(f"\n部署: bash scripts/serve_ascend.sh {out_dir} ascend")
            return 0

    print("[FAIL] msModelSlim 调用均失败，尝试记录：")
    for a in attempts:
        print("   -", a)
    print("\n建议：")
    print("  1) 核对本机 msModelSlim 版本与官方示例（API 签名常有差异）")
    print("  2) 回退: python quantize/gen_llmcomp.py --scheme W8A8")
    return 4


if __name__ == "__main__":
    sys.exit(main())
