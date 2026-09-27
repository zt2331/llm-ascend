#!/usr/bin/env python
"""05_model_index_check.py —— 诊断模型目录，并修复失效的 safetensors 索引。

典型问题（本项目现场遇到的）：
    FileNotFoundError: .../model-00002-of-00015.safetensors
  即 `model.safetensors.index.json` 引用的分片名与实际文件对不上
  （模型被转换/重存过，但索引没更新）。
  vLLM 不读索引、直接扫分片所以能加载；而 transformers 读索引 → 直接报错，
  导致 prune/distill/量化全部失败。

本脚本：
  1. 列出实际分片、索引引用的分片、config 关键信息（架构/精度/层数/是否 MoE）
  2. 判断索引是否失效
  3. 用 --fix 在本地生成一个「overlay 目录」：
     用符号链接指向原始分片 + 写入**重建的正确索引**
     （不修改原目录，零拷贝，省磁盘）

用法:
    python scripts/05_model_index_check.py                    # 只诊断
    python scripts/05_model_index_check.py --fix              # 诊断并生成 overlay
    python scripts/05_model_index_check.py --fix --out /workspace/models/Qwen3.6-27B-fixed
"""
import argparse
import glob
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

WEIGHT_EXTS = (".safetensors", ".bin")


def read_safetensors_keys(path):
    """只读 safetensors 头部，取出张量名列表（不加载数据）。"""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        return [k for k in hdr.keys() if k != "__metadata__"]
    except Exception as e:
        print(f"    [WARN] 读取失败 {os.path.basename(path)}: {e}")
        return []


def inspect(model_dir):
    print("=" * 78)
    print("模型目录诊断")
    print("=" * 78)
    print(f"路径: {model_dir}\n")

    # ---------- 1. config ----------
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        print("[FAIL] 没有 config.json，路径可能不对")
        return None
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    tc = cfg.get("text_config", cfg)
    print("[1] config.json 关键信息")
    print(f"    architectures        : {cfg.get('architectures')}")
    print(f"    model_type           : {cfg.get('model_type')}")
    print(f"    torch_dtype/dtype    : {cfg.get('torch_dtype') or cfg.get('dtype')}")
    print(f"    是否多模态           : {bool(cfg.get('vision_config'))}")
    print(f"    num_hidden_layers    : {tc.get('num_hidden_layers')}")
    print(f"    hidden_size          : {tc.get('hidden_size')}")
    ne = tc.get("num_experts") or tc.get("num_local_experts")
    print(f"    MoE(num_experts)     : {ne if ne else '否（稠密模型）'}")
    qc = cfg.get("quantization_config")
    print(f"    quantization_config  : {json.dumps(qc, ensure_ascii=False) if qc else '无（未量化）'}")
    lt = tc.get("layer_types")
    if lt:
        from collections import Counter
        print(f"    layer_types 分布     : {dict(Counter(lt))}")
    print(f"    image_token_id       : {cfg.get('image_token_id', '无')}")

    # ---------- 2. 实际分片 ----------
    actual = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    actual += sorted(glob.glob(os.path.join(model_dir, "*.bin")))
    total = sum(os.path.getsize(p) for p in actual)
    print(f"\n[2] 实际权重文件：{len(actual)} 个，共 {total / 1024**3:.2f} GiB")
    for p in actual[:20]:
        print(f"    {os.path.basename(p):<48}{os.path.getsize(p) / 1024**3:>8.2f} GiB")
    if len(actual) > 20:
        print(f"    ...（还有 {len(actual) - 20} 个）")

    # ---------- 3. 索引 ----------
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    idx_files = None
    if os.path.isfile(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            idx = json.load(f)
        wm = idx.get("weight_map", {})
        idx_files = sorted({v for v in wm.values()})
        print(f"\n[3] model.safetensors.index.json")
        print(f"    引用的分片数 : {len(idx_files)}")
        print(f"    张量总数     : {len(wm)}")
        for f_ in idx_files[:20]:
            exists = os.path.isfile(os.path.join(model_dir, f_))
            print(f"    {'  OK ' if exists else ' 缺失'}  {f_}")
        if len(idx_files) > 20:
            print(f"    ...（还有 {len(idx_files) - 20} 个）")
    else:
        print("\n[3] 没有 model.safetensors.index.json（单文件模型或索引缺失）")

    # ---------- 4. 张量覆盖率（判断权重是否完整）----------
    print("\n[4] 张量覆盖率（判断模型是否完整）")
    cover = None
    idx_map = {}
    if idx_files:
        try:
            with open(idx_path, encoding="utf-8") as f:
                idx_map = json.load(f).get("weight_map", {})
        except Exception:
            idx_map = {}
        want = set(idx_map.keys())
        have = set()
        for p_ in actual:
            have |= set(read_safetensors_keys(p_))
        got = want & have
        cover = (len(got), len(want))
        pct = len(got) / max(len(want), 1) * 100
        print(f"    索引声明张量 : {len(want)}")
        print(f"    实际含有张量 : {len(got)}  ({pct:.1f}%)")
        miss_t = sorted(want - have)
        if miss_t:
            print(f"    ❌ 缺失张量 {len(miss_t)} 个，例如:")
            for t in miss_t[:6]:
                print(f"        {t}")
            print("       → 权重【不完整】，极可能是下载中断")
            print("       → 此时重建索引很危险：能加载，但缺失层随机初始化 → 输出乱码")
        else:
            print("    ✅ 索引声明的张量全部存在（权重完整）")

    # ---------- 5. 判定 ----------
    actual_names = {os.path.basename(p_) for p_ in actual}
    missing = [f_ for f_ in (idx_files or []) if f_ not in actual_names]
    complete = bool(cover) and cover[0] == cover[1]
    print("\n[5] 结论")
    if cover and cover[0] != cover[1]:
        print(f"    ❌❌ 模型权重不完整：{cover[0]}/{cover[1]} 个张量")
        print(f"        缺 {len(missing)} 个分片: {missing[:6]}")
        print("        → 【不要】用 --fix！先补齐权重文件（见末尾提示）")
    elif missing:
        print(f"    ❌ 索引失效：引用了 {len(missing)} 个不存在的分片")
        print(f"       例如: {missing[:3]}")
        print("       → transformers 会直接 FileNotFoundError")
        print("       → 但张量齐全，可用 --fix 重建索引")
    elif idx_files:
        print("    ✅ 索引与实际分片一致，模型完整")
    else:
        print("    ℹ️ 无索引（transformers 会自行扫描所有分片）")

    # ---------- 6. 精度线索 ----------
    print("\n[6] 精度线索")
    n_layer = tc.get("num_hidden_layers") or 0
    print(f"    层数 = {n_layer}, hidden = {tc.get('hidden_size')}")
    print(f"    现有权重总量 ≈ {total / 1024**3:.2f} GiB（{len(actual)} 个分片）")
    if idx_files and len(actual) < len(idx_files):
        est = total * len(idx_files) / max(len(actual), 1)
        print(f"    ⚠️ 索引要求 {len(idx_files)} 个分片 → 完整模型应约 {est / 1024**3:.1f} GiB")
    print(f"    当前字节/参数 ≈ {total * 1024**3 / 27e9:.2f}（2.0=BF16, 1.0=FP8, 0.5=INT4）")
    if cover and cover[0] != cover[1]:
        print("    ⚠️ 权重不完整，该估算不可靠")

    return {"actual": actual, "idx_files": idx_files, "missing": missing,
            "cfg": cfg, "complete": complete, "cover": cover}


def build_overlay(model_dir, out_dir):
    """生成 overlay：符号链接原始文件 + 重建正确的索引（不拷贝权重）。"""
    info = inspect(model_dir)
    if info is None:
        return None
    actual, cfg = info["actual"], info["cfg"]

    if not actual:
        print("\n[FAIL] 目录里没有权重文件，无法修复")
        return None

    # ★ 安全闸：权重不完整时拒绝重建索引（重建出来的模型能加载但输出乱码）
    cover = info.get("cover")
    if cover and cover[0] != cover[1]:
        print("\n" + "!" * 78)
        print("[拒绝执行] 模型权重不完整，重建索引会产出【能加载但输出乱码】的模型。")
        print(f"           索引声明 {cover[1]} 个张量，实际只有 {cover[0]} 个。")
        print("           请先补齐缺失的分片，再重新运行本脚本。")
        print("!" * 78)
        return None

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[FIX] 生成 overlay -> {out_dir}")

    # 1) 软链所有非索引文件（含分片、config、tokenizer、processor 等）
    n_link = 0
    for fn in os.listdir(model_dir):
        if fn == "model.safetensors.index.json":
            continue
        if fn.endswith(".index.json"):
            continue
        src = os.path.join(model_dir, fn)
        dst = os.path.join(out_dir, fn)
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
                n_link += 1
            except OSError:
                import shutil
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    n_link += 1
    print(f"  软链/复制 {n_link} 个文件（权重不拷贝）")

    # 2) 重建索引：扫描每个分片的张量名
    weight_map = {}
    total_size = 0
    for p in actual:
        base = os.path.basename(p)
        keys = read_safetensors_keys(p)
        total_size += os.path.getsize(p)
        for k in keys:
            if k in weight_map:
                print(f"  [WARN] 张量重复: {k} 同时出现在 {weight_map[k]} 和 {base}")
            weight_map[k] = base
    new_idx = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    out_idx = os.path.join(out_dir, "model.safetensors.index.json")
    with open(out_idx, "w", encoding="utf-8") as f:
        json.dump(new_idx, f, indent=2)
    print(f"  重建索引: {len(weight_map)} 个张量 -> {len(set(weight_map.values()))} 个分片")

    # 3) 校验
    print("\n[校验]")
    ref = sorted(set(weight_map.values()))
    bad = [f_ for f_ in ref if not os.path.isfile(os.path.join(out_dir, f_))]
    if bad:
        print(f"  ❌ overlay 里仍缺 {len(bad)} 个分片: {bad[:3]}")
        return None
    print(f"  ✅ 索引与实际一致（{len(ref)} 个分片）")
    print(f"  ✅ 多模态: {bool(cfg.get('vision_config'))}")
    print(f"\n使用方式:")
    print(f"    export MODEL_PATH={out_dir}")
    print(f"\n    python -c \"import sys;sys.path.insert(0,'.');import config;print(config.MODEL_PATH)\"")
    print(f"    python scripts/02_check_env.py")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="模型目录（默认 config.MODEL_PATH）")
    ap.add_argument("--fix", action="store_true", help="生成 overlay 目录并重建索引")
    ap.add_argument("--out", default=None, help="overlay 输出目录")
    args = ap.parse_args()

    model_dir = args.model or config.MODEL_PATH
    if not model_dir or not os.path.isdir(model_dir):
        print(f"[FAIL] 模型目录无效: {model_dir}")
        print("       请先 export MODEL_PATH=... 或用 --model 指定")
        return 1

    if not args.fix:
        info = inspect(model_dir)
        print("\n" + "-" * 78)
        print("后续操作提示")
        print("-" * 78)
        cover = (info or {}).get("cover")
        if cover and cover[0] != cover[1]:
            print("模型权重不完整，请任选一种方式补齐：")
            print("  1) 重新下载该模型（推荐，用 ModelScope / HuggingFace 官方仓库）")
            print("       modelscope download --model <repo_id> --local_dir <目录>")
            print("  2) 从环境里其它位置找一份完整副本（见下方检查命令）")
            print("  3) 换用 shared_assets 里其它【完整】的模型跑本项目")
            print("\n快速检查哪些模型是完整的：")
            print("  python scripts/06_check_all_models.py")
        else:
            print("如需修复索引: python scripts/05_model_index_check.py --fix")
        return 0

    # 默认放到仓库外，避免污染 git 仓库
    base = "/workspace/model_fixed" if os.path.isdir("/workspace") \
        else os.path.join(os.path.dirname(config.PROJECT_ROOT), "model_fixed")
    out_dir = args.out or os.path.join(base, os.path.basename(os.path.normpath(model_dir)))
    out_dir = os.path.abspath(out_dir)
    if os.path.abspath(out_dir) == os.path.abspath(model_dir):
        print("[FAIL] overlay 目录不能与原目录相同")
        return 1
    res = build_overlay(model_dir, out_dir)
    return 0 if res else 2


if __name__ == "__main__":
    sys.exit(main())
