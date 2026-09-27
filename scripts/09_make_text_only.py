#!/usr/bin/env python
"""09_make_text_only.py —— 把量化产物整理成「一致的纯文本模型」，供 vLLM 加载。

背景（现场现象）
    llm-compressor 量化多模态模型后，产物 config.json 被剥离成纯文本
    （architectures: Qwen3_5ForCausalLM，无 vision_config），
    但目录里仍留着 processor_config.json / preprocessor_config.json。
    → vLLM 看到处理器文件就按【多模态】初始化，却读到【文本】config，报：
        TypeError: Expected Qwen3_5Config, but found Qwen3_5TextConfig

两种正确选择（本项目都支持）
  ① 纯文本部署（本脚本）：删掉残留的多模态处理器文件，让 vLLM 一致地按文本模型加载。
     模型失去图像能力，但文本任务（PPL/基准）完全正常，且省显存。
  ② 完整多模态部署：用 scripts/08_fix_mm_wrapper.py 恢复 wrapper，
     再用 vLLM `--language-model-only` 跳过视觉编码器（官方参数）。

用法:
    python scripts/09_make_text_only.py <量化产物目录>            # 预演，不改动
    python scripts/09_make_text_only.py <量化产物目录> --apply     # 实际执行
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 这些文件会让 vLLM 走多模态路径
MM_FILES = ["processor_config.json", "preprocessor_config.json",
            "video_preprocessor_config.json"]
# 多模态相关的 config 键（若存在说明仍是 wrapper）
MM_CFG_KEYS = ["vision_config", "image_token_id", "video_token_id",
               "vision_start_token_id", "vision_end_token_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("quant_dir")
    ap.add_argument("--apply", action="store_true", help="实际执行（默认只预演）")
    args = ap.parse_args()

    d = os.path.abspath(args.quant_dir.rstrip("/"))
    if not os.path.isdir(d):
        print(f"[FAIL] 目录不存在: {d}")
        return 1

    print("=" * 74)
    print(f"整理为纯文本模型: {d}")
    print(f"模式: {'实际执行' if args.apply else '预演（不改动）'}")
    print("=" * 74)

    # ---------- 1. 检查 config ----------
    cfg_p = os.path.join(d, "config.json")
    if not os.path.isfile(cfg_p):
        print("[FAIL] 没有 config.json")
        return 1
    with open(cfg_p, encoding="utf-8") as f:
        cfg = json.load(f)

    arch = cfg.get("architectures") or []
    has_vision = any(k in cfg for k in MM_CFG_KEYS)
    is_text_arch = any("CausalLM" in a for a in arch)

    print("\n[1] 当前 config")
    print(f"    architectures   : {arch}")
    print(f"    model_type      : {cfg.get('model_type')}")
    print(f"    含多模态 config 键: {has_vision}")
    print(f"    language_model_only: {cfg.get('language_model_only', '(未设置)')}")

    # ---------- 2. 找残留处理器文件 ----------
    present = [f for f in MM_FILES if os.path.isfile(os.path.join(d, f))]
    print(f"\n[2] 残留的多模态处理器文件: {present or '无'}")

    if has_vision:
        print("\n    ⚠️ config 里还有多模态键 —— 说明这是完整 wrapper，")
        print("       不需要纯文本化；直接部署即可（可加 --language-model-only 跳视觉塔）。")
        return 0

    if not present and not is_text_arch:
        print("\n    ℹ️ 既无多模态文件、架构也不是 CausalLM，请人工确认。")

    if not present:
        print("\n[OK] 没有需要清理的文件，已是纯文本模型")
        return 0

    # ---------- 3. 执行 ----------
    print("\n[3] 处理")
    if not args.apply:
        for f in present:
            print(f"    [预演] 将删除 {f}")
        print("\n    重新运行并加 --apply 才会真正删除。")
        return 0

    for f in present:
        p = os.path.join(d, f)
        bak = p + ".mm_bak"
        if not os.path.exists(bak):
            shutil.move(p, bak)
            print(f"    已移走 {f} -> {os.path.basename(bak)}（可恢复）")

    # ---------- 4. 校验 ----------
    print("\n[4] 校验")
    left = [f for f in MM_FILES if os.path.isfile(os.path.join(d, f))]
    print(f"    剩余多模态处理器文件: {left or '无'}  "
          f"{'✅' if not left else '❌'}")
    print(f"    architectures: {arch}")

    print("\n[OK] 完成。部署方式：")
    print(f"    vllm serve {d} --quantization compressed-tensors \\")
    print(f"        --dtype float16 --gpu-memory-utilization 0.90 --trust-remote-code")
    print("\n    或用本项目脚本：")
    print(f"    bash scripts/serve_ascend.sh {d} compressed-tensors 8000")
    print("\n    如需恢复：把 *.mm_bak 改回原名即可。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
