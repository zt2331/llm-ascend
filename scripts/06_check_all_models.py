#!/usr/bin/env python
"""06_check_all_models.py —— 批量检查模型目录是否【完整可用】。

背景：共享模型目录常出现"下载中断"——分片缺失但 index.json 还是旧的，
transformers 加载会 FileNotFoundError，或更糟：能加载但缺失层随机初始化。

本脚本扫描指定根目录下所有模型，按张量覆盖率判定是否完整。

用法:
    python scripts/06_check_all_models.py
    python scripts/06_check_all_models.py --root /workspace/shared_assets/models
    python scripts/06_check_all_models.py --root /workspace/shared_assets/models --detail
"""
import argparse
import glob
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_keys(path):
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        return {k for k in hdr.keys() if k != "__metadata__"}
    except Exception:
        return set()


def check_model(d):
    """返回该模型目录的完整性信息。"""
    cfg_p = os.path.join(d, "config.json")
    info = {"path": d, "name": os.path.basename(os.path.normpath(d))}
    try:
        with open(cfg_p, encoding="utf-8") as f:
            cfg = json.load(f)
        tc = cfg.get("text_config", cfg) or {}
        info["arch"] = ",".join(cfg.get("architectures", []) or []) or cfg.get("model_type", "?")
        info["layers"] = tc.get("num_hidden_layers")
        info["multimodal"] = bool(cfg.get("vision_config"))
        info["quant"] = (cfg.get("quantization_config") or {}).get("quant_method")
    except Exception as e:
        info["error"] = f"config.json 读取失败: {e}"
        return info

    shards = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    info["n_shard"] = len(shards)
    info["size_gb"] = round(sum(os.path.getsize(p) for p in shards) / 1024 ** 3, 2)

    idx_p = os.path.join(d, "model.safetensors.index.json")
    if shards and os.path.isfile(idx_p):
        try:
            with open(idx_p, encoding="utf-8") as f:
                wm = json.load(f).get("weight_map", {})
            want = set(wm.keys())
            files = {os.path.basename(p) for p in shards}
            missing_files = sorted({v for v in wm.values() if v not in files})
            have = set()
            for p in shards:
                have |= read_keys(p)
            got = len(want & have)
            info.update(want=len(want), got=got,
                        pct=round(got / max(len(want), 1) * 100, 1),
                        missing_files=len(missing_files))
            info["complete"] = (got == len(want) and not missing_files)
        except Exception as e:
            info["error"] = f"index 解析失败: {e}"
            info["complete"] = False
    elif shards:
        # 无索引：视为完整（transformers 会扫描全部）
        info.update(want=None, got=None, pct=None, missing_files=0, complete=True)
    else:
        info["complete"] = False
        info["error"] = "没有 .safetensors 权重文件"

    # 附带检查必需文件
    need = ["tokenizer.json", "tokenizer_config.json"]
    info["missing_aux"] = [f for f in need if not os.path.isfile(os.path.join(d, f))]
    return info


def find_models(root, max_depth=4):
    root = os.path.abspath(root)
    base = root.rstrip("/").count("/")
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.rstrip("/").count("/") - base
        dirnames[:] = [x for x in dirnames if not x.startswith(".")]
        if "config.json" in filenames:
            out.append(dirpath)
        if depth >= max_depth:
            dirnames[:] = []
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/shared_assets/models",
                    help="扫描根目录（默认 /workspace/shared_assets/models）")
    ap.add_argument("--detail", action="store_true", help="打印每个模型的详情")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"[FAIL] 目录不存在: {args.root}")
        print("       用 --root 指定其它位置，例如 --root /workspace")
        return 1

    print("=" * 96)
    print(f"模型完整性检查   根目录: {args.root}")
    print("=" * 96)

    dirs = find_models(args.root)
    if not dirs:
        print("未找到任何含 config.json 的模型目录")
        return 1

    results = [check_model(d) for d in dirs]

    hdr = f"{'模型':<34}{'分片':>5}{'大小GB':>9}{'张量覆盖':>12}{'层数':>6}{'多模态':>7}  状态"
    print("\n" + hdr)
    print("-" * 96)
    for r in sorted(results, key=lambda x: (not x.get("complete"), x["name"])):
        cov = "-"
        if r.get("want"):
            cov = f"{r['got']}/{r['want']}"
        elif r.get("pct") is not None:
            cov = f"{r['pct']}%"
        state = "✅ 可用" if r.get("complete") else "❌ 不完整"
        if r.get("error") and not r.get("complete"):
            state = f"❌ {r['error'][:22]}"
        name = r["name"][:32]
        print(f"{name:<34}{r.get('n_shard', '-'):>5}{r.get('size_gb', '-'):>9}"
              f"{cov:>12}{str(r.get('layers', '-')):>6}"
              f"{('是' if r.get('multimodal') else '否'):>7}  {state}")

    ok = [r for r in results if r.get("complete")]
    bad = [r for r in results if not r.get("complete")]
    print("\n" + "=" * 96)
    print(f"完整可用: {len(ok)} 个   不完整: {len(bad)} 个")
    if ok:
        print("\n可用模型（建议从中选一个跑项目）：")
        for r in sorted(ok, key=lambda x: -(x.get("size_gb") or 0)):
            print(f"  {r['path']}")
            print(f"      {r.get('size_gb')}GB  层数={r.get('layers')}  "
                  f"多模态={'是' if r.get('multimodal') else '否'}  "
                  f"量化={r.get('quant') or '未量化'}")
    if bad:
        print("\n不完整模型（需补齐或避开）：")
        for r in bad:
            print(f"  {r['path']}")
            if r.get("want"):
                print(f"      张量 {r.get('got')}/{r.get('want')}，缺 {r.get('missing_files')} 个分片")
            elif r.get("error"):
                print(f"      {r['error']}")

    print("\n用法提示：")
    print("  选一个✅完整模型后：")
    print("    export MODEL_PATH=<上面的路径>")
    print("    python scripts/02_check_env.py")

    if args.detail:
        print("\n" + "=" * 96)
        print("详情")
        for r in results:
            print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
