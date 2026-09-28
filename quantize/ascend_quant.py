#!/usr/bin/env python
"""昇腾原生量化（msModelSlim / msmodelslim）—— 昇腾 NPU 推荐路径。

★ 与 quantize/w8a8_smooth.py 的区别（两者都是 W8A8，但算法不同）：
    本脚本  = msModelSlim 的【校准式 min/max 量化】，
              只传 w_bit/a_bit 给 QuantConfig，**不含平滑/离群抑制**，
              本质是“带激活校准的 RTN”。
    w8a8_smooth.py = llm-compressor 的 SmoothQuant，
              显式做了激活离群迁移 Y=(X/s)(W·s)ᵀ 后才是 INT8。
  想比较“平滑到底带来多少增益”，要跑的是 w8a8_smooth.py vs
  gen_llmcomp.py --method rtn --scheme W8A8（同一库、同一位宽，只差平滑）。

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


def build_calib(tok, calib_n, seq, form="list"):
    """构造 msModelSlim 的 calib_data。

    ★ 格式要求（三轮实测踩坑记录，务必保持）：
      1) msModelSlim 会校验 `calib_data[i]` 的类型，**只接受 list 或 dict**：
           传 tuple → TypeError: calib_data[0] must be list or dict, not tuple.
      2) 校验通过后按 `model(*(calib_data[i]))` 调用（其警告原文即如此），
         所以容器里必须恰好是 forward 所需的位置参数。
      3) ★ 张量必须**带 batch 维**，即 input_ids 形状为 [1, seq] 而不是 [seq]。
         原因：Qwen3.5 的 linear_attn 里有
             batch_size, seq_len, _ = hidden_states.shape
         传 1 维 input_ids 时 embedding 输出只有 2 维 → 
             ValueError: not enough values to unpack (expected 3, got 2)
         （调用栈落在 modeling_qwen3_5.py 的 linear_attn.forward）
      4) 张量还必须与模型**同设备**，否则 embedding 报 indices is on cpu。

      四种写法对比：
        [ [ids_1d], ... ]        → 2 维 hidden → 解包失败            ❌
        [ [ids_1xN], ... ]       → 3 维 hidden → ✓                  ✅
        [ [(ids,), ...] ]        → must be list or dict, not tuple. ❌
        [ {"input_ids": ...} ]   → IndexError: tuple index out of range ❌
    """
    import torch
    from utils.dataio import Dataset

    pq = config.CALIB_PARQUET
    full = Dataset.from_parquet(pq)
    rows = full.select(range(min(calib_n, len(full))))
    data = []
    for r in rows:
        ids = tok(r["text"], truncation=True, max_length=seq)["input_ids"]
        # ★ .unsqueeze(0) 补 batch 维：[seq] → [1, seq]
        t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        if form == "dict":
            data.append({"input_ids": t})
        else:                                   # 默认 list：与 msModelSlim 的
            data.append([t])                    # model(*(calib_data[i])) 语义一致
    return data


def move_calib(data, device):
    """把校准张量搬到指定设备；非张量元素原样保留。

    msModelSlim 明确提示 model 与 calib_data 要在同一设备上
    （"check model and calib_data on the device that QuantConfig indicates"），
    否则 forward 时 embedding 会报 "indices is on cpu"。
    """
    import torch
    from utils import device as dev

    if device in (None, "cpu"):
        return data
    tgt = dev.get_device("auto") if device == "npu" else device
    out = []
    for d in data:
        if isinstance(d, list):
            out.append([x.to(tgt) if isinstance(x, torch.Tensor) else x for x in d])
        elif isinstance(d, dict):
            out.append({k: (v.to(tgt) if isinstance(v, torch.Tensor) else v)
                        for k, v in d.items()})
        else:
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--scheme", default="W8A8", choices=["W8A8", "W8A16", "W4A16"])
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smooth", action="store_true",
                    help="启用 msModelSlim 的平滑（do_smooth=True）。"
                         "默认关闭 —— 此时是【校准式 min/max 量化】（带校准的 RTN），"
                         "与 quantize/w8a8_smooth.py 的 SmoothQuant 不是同一个算法。"
                         "打开后才真正做激活离群抑制。")
    ap.add_argument("--debug", action="store_true",
                    help="只尝试第一种方案，不做 cpu 兜底（避免调试时让大模型在 CPU 上重跑一遍）。"
                         "失败时的完整调用栈默认就会打印，无需此开关。")
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

    # ★ 校准数据必须是「单元素 list 的 list」（msModelSlim 只接受 list/dict，
    #   且按 model(*(calib_data[i])) 调用），见 build_calib 注释。
    # ★ 还必须与模型【同设备】：msModelSlim 明确提示
    #   "check model and calib_data on the device that QuantConfig indicates"；
    #   否则 forward 时 embedding 报 "indices is on cpu"。
    calib_cpu = build_calib(tok, args.calib, args.seq, "list")
    print(f"[INFO] 校准样本 {len(calib_cpu)} 条，最大长度 {args.seq}")

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

    # ★ disable_names 的格式已实测确认：**必须用带 `model.` 前缀的完整 Linear 名**。
    #   - "model.visual.blocks.0.attn.qkv"  → 通过校验 ✓
    #   - "visual.blocks.0.attn.qkv"（去掉前缀）→ ValueError: disable_names has
    #     invalid key `visual.blocks.0.attn.qkv` ✗
    #   所以不再尝试多种前缀变体，直接用 skip_linear_names。

    w_bit, a_bit = (8, 8) if args.scheme == "W8A8" else (8, 16)
    if args.scheme == "W4A16":
        w_bit, a_bit = (4, 16)

    os.makedirs(out_dir, exist_ok=True)

    # ---- 调用 ----
    # ★ 签名已实测确认（msModelSlim 会在失败时自己打印出来）：
    #   QuantConfig(w_bit=8, a_bit=8, act_method=1, w_method='MinMax',
    #               disable_names=None, pr=1.0, mm_tensor=True,
    #               dev_type='cpu', dev_id=None, ..., w_sym=True,
    #               is_lowbit=False, do_smooth=False, use_sigma=False, ...,
    #               disable_last_linear=True, ..., open_outlier=True,
    #               is_dynamic=False, group_size=64, percdamp=0.01, pdmix=False)
    #   Calibrator(self, model, cfg, calib_data=None, disable_level='L0',
    #              all_tensors=None, mix_cfg=None)
    #   注意 dev_type 默认就是 'cpu'（不是 npu！），不显式传 npu 的话
    #   msModelSlim 会把已经加载到 NPU 上的模型再搬回 CPU，白白多花时间。
    import inspect
    _qc_params = set(inspect.signature(QuantConfig).parameters)
    print(f"[INFO] QuantConfig 支持 dev_type={'dev_type' in _qc_params}, "
          f"do_smooth={'do_smooth' in _qc_params}, "
          f"open_outlier={'open_outlier' in _qc_params}")

    if args.smooth and "do_smooth" not in _qc_params:
        print("[FAIL] 该版本 QuantConfig 没有 do_smooth 参数，无法启用平滑；"
              "请改用 python quantize/w8a8_smooth.py（llm-compressor SmoothQuant）")
        return 2
    if args.smooth:
        print("[INFO] do_smooth=True → 启用平滑（激活离群抑制）")

    attempts = []

    def _try(fn, tag):
        try:
            fn()
            print(f"[OK] 量化成功（{tag}）")
            return True
        except Exception as e:
            attempts.append(f"{tag}: {type(e).__name__} {str(e)[:200]}")
            # ★ msModelSlim 用 `raise Exception(...) from e` 包了一层，
            #   真正的调用栈在 __cause__ 里。只打印 str(e) 会丢掉关键信息。
            #   ★ 默认就打印（不靠 --debug）：失败输出本身就是诊断信息，
            #     之前因为要记得加参数，白白多跑了一轮 27B。
            import traceback
            print(f"\n{'='*70}\n[调用栈] {tag}\n{'='*70}")
            cur, depth = e, 0
            seen = set()
            while cur is not None and depth < 6 and id(cur) not in seen:
                seen.add(id(cur))
                print(f"── 异常链第 {depth} 层: {type(cur).__name__}: {str(cur)[:300]}")
                tb = "".join(traceback.format_exception(
                    type(cur), cur, cur.__traceback__))
                for line in tb.strip().splitlines()[-30:]:
                    print("   " + line)
                print()
                cur = cur.__cause__ or cur.__context__
                depth += 1
        return False

    def make_api(disable, dev_type, do_smooth, calib_data):
        def api():
            kwargs = dict(w_bit=w_bit, a_bit=a_bit)
            if "dev_type" in _qc_params:
                kwargs["dev_type"] = dev_type
            if "do_smooth" in _qc_params:
                kwargs["do_smooth"] = do_smooth
            if disable:
                kwargs["disable_names"] = disable
            qc = QuantConfig(**kwargs)
            cal = Calibrator(model, qc, calib_data=calib_data)
            cal.run()
            cal.save(out_dir, save_type=["safe_tensor"])
        return api

    npu_dev = "npu" if dev.has_npu() else "cpu"
    # 校准数据形式固定为 list（实测确认：dict 形式报 IndexError: tuple index
    # out of range；list 形式已经能跑到 forward）。这里只用 list，
    # 但把校准张量放到与 dev_type 一致的设备上（否则 embedding 报 indices is on cpu）。
    plan = [(make_api(skip_linear_names, npu_dev, args.smooth,
                      move_calib(calib_cpu, npu_dev)),
             f"dev_type={npu_dev}, calib_on={npu_dev}, do_smooth={args.smooth}")]
    if npu_dev != "cpu" and not args.debug:
        # 兜底：dev_type=cpu（会把模型搬回 CPU，很慢）。--debug 时跳过，
        # 免得调试期间让 27B 在 CPU 上白跑一遍。
        plan.append((make_api(skip_linear_names, "cpu", args.smooth,
                              move_calib(calib_cpu, "cpu")),
                     f"dev_type=cpu, calib_on=cpu, do_smooth={args.smooth}"))

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
            # ★ 多模态模型：自动恢复被剥离的 wrapper（否则 vLLM 加载失败 / 评测 nan）
            _target_dir = out_dir
            try:
                from utils.mm_wrapper import maybe_restore
                _fixed = maybe_restore(out_dir, model_path)
                if _fixed:
                    _target_dir = _fixed
                    print(f"[OK] 已自动恢复多模态 wrapper -> {_fixed}")
                    print(f"     评测/部署请使用: {_fixed}")
            except Exception as _e:
                print(f"[WARN] wrapper 自动恢复异常: {_e}")

            # ★ 保存后自检：检查【最终产物】量化是否真的落盘
            try:
                from utils.quant_check import inspect_quant_dir, format_report
                _info = inspect_quant_dir(_target_dir)
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
