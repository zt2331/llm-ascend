#!/usr/bin/env python
"""03_find_model.py —— 搜索并报告本机可用的 base 模型目录。

先按预设位置搜；若没找到，自动做一次有界深度扫描并给出候选。

用法:
    python scripts/03_find_model.py
    export MODEL_PATH=/path/to/Qwen3.6-27B     # 指定后优先使用
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _size_of(d):
    try:
        n = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)
                if f.endswith((".safetensors", ".bin")))
        return n / 1024 ** 3
    except Exception:
        return -1.0


def _print_candidate(i, c, mark=""):
    cfg = os.path.join(c, "config.json")
    arch = "?"
    try:
        import json
        with open(cfg, encoding="utf-8") as f:
            d = json.load(f)
        arch = ",".join(d.get("architectures", []) or []) or d.get("model_type", "?")
    except Exception:
        pass
    gb = _size_of(c)
    size = f"{gb:.1f} GB" if gb > 0 else "?"
    print(f"  [{i}] {c}")
    print(f"       权重 {size}   架构 {arch}{mark}")


def deep_scan():
    """有界深度扫描（find），返回含 config.json 的目录列表。"""
    roots = ["/workspace", "/home", "/root", "/data", "/cache", "/opt", "/models"]
    roots = [r for r in roots if os.path.isdir(r)]
    if not roots:
        return []
    cmd = ["find"] + roots + [
        "-maxdepth", "6",
        "-name", "config.json",
        "-not", "-path", "*/site-packages/*",
        "-not", "-path", "*/.cache/huggingface/*",
        "-not", "-path", "*/node_modules/*",
        "-not", "-path", "*/output_models/*",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        dirs = sorted({os.path.dirname(p) for p in out.stdout.splitlines() if p.strip()})
        return dirs
    except Exception as e:
        print(f"  （深度扫描失败: {e}）")
        return []


def main():
    print("=" * 74)
    print("模型搜索")
    print("=" * 74)

    env = os.environ.get("MODEL_PATH")
    print(f"\n环境变量 MODEL_PATH = {env or '(未设置)'}")

    roots = config._search_roots()
    print(f"\n预设搜索位置（{len(roots)} 个）:")
    for r in roots:
        print(f"  - {r}")

    cands = config.find_model_candidates()
    if cands:
        print(f"\n✅ 找到 {len(cands)} 个候选（按匹配度排序）：")
        for i, c in enumerate(cands):
            _print_candidate(i, c, "  ← 将被使用" if i == 0 else "")
        used = config.resolve_model_path()
        print(f"\n最终使用: {used}")
        print(f"MODEL_TAG = {config.MODEL_TAG}")
        print("\n如需固定：export MODEL_PATH=<上面的路径>")
        return 0

    print("\n预设位置没找到，启动深度扫描（最多 120 秒）...")
    found = deep_scan()
    if found:
        print(f"\n🔍 深度扫描发现 {len(found)} 个含 config.json 的目录：")
        for i, c in enumerate(found[:20]):
            _print_candidate(i, c)
        print("\n👉 请从中选出 base 模型目录并设置：")
        print(f"    export MODEL_PATH={found[0]}")
        print("  然后重新运行本脚本确认。")
        return 0

    print("\n❌ 仍未找到任何模型目录。")
    print("请手动确认模型位置后设置环境变量：")
    print("    export MODEL_PATH=/你的/模型/路径")
    print("\n也可以手动找一下（按目录名搜索）：")
    print("    find / -maxdepth 6 -type d -iname '*qwen*' 2>/dev/null | head -20")
    return 1


if __name__ == "__main__":
    sys.exit(main())
