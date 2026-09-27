#!/usr/bin/env python
"""03_find_model.py —— 搜索并报告本机可用的 base 模型目录。

用法:
    python scripts/03_find_model.py
    export MODEL_PATH=/path/to/Qwen3.6-27B     # 指定后优先使用
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def main():
    print("=" * 72)
    print("模型搜索")
    print("=" * 72)

    env = os.environ.get("MODEL_PATH")
    print(f"\n环境变量 MODEL_PATH = {env or '(未设置)'}")

    cands = config.find_model_candidates()
    if not cands:
        print("\n[FAIL] 未找到任何含 config.json 的模型目录。")
        print("请任选一种方式：")
        print("  1) export MODEL_PATH=/path/to/Qwen3.6-27B")
        print("  2) mkdir -p qwen_model && ln -s /真实路径 qwen_model/Qwen3.6-27B")
        return 1

    print(f"\n找到 {len(cands)} 个候选（按匹配度排序）：")
    for i, c in enumerate(cands):
        cfg = os.path.join(c, "config.json")
        size = -1
        try:
            size = sum(os.path.getsize(os.path.join(c, f))
                       for f in os.listdir(c) if f.endswith((".safetensors", ".bin")))
        except Exception:
            pass
        gb = f"{size / 1024**3:.1f} GB" if size > 0 else "?"
        mark = "  ← 将被使用" if i == 0 else ""
        print(f"  [{i}] {c}")
        print(f"       权重 {gb}{mark}")

    used = config.resolve_model_path()
    print(f"\n最终使用: {used}")
    print(f"MODEL_TAG = {config.MODEL_TAG}")
    print("\n如需固定使用某个目录：export MODEL_PATH=<上面的路径>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
