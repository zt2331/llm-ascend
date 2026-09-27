#!/usr/bin/env python
"""02_check_env.py —— 昇腾环境自检（跑任何流程前先跑这个）。

检查内容：
  1. Python / 关键包版本（与镜像版本矩阵对照）
  2. torch_npu 是否可用、NPU 设备数量/型号/显存
  3. CANN 版本 + npu-smi 输出
  4. 在 NPU 上做一次真实张量运算（验证算子可用）
  5. 项目依赖是否齐全（matplotlib/llmcompressor 等）
"""
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 镜像自带版本（用于对照，不一致只提示不报错）
EXPECT = {
    "torch": "2.10.0",
    "torch_npu": "2.10.0.post4",
    "vllm": "0.23.0",
    "vllm_ascend": "0.23.0",
    "triton_ascend": "3.2.2",
}

OK = "[ OK ]"
BAD = "[FAIL]"
WARN = "[WARN]"


def _ver(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "?")
    except Exception:
        return None


def main():
    print("=" * 72)
    print("昇腾 NPU 环境自检")
    print("=" * 72)

    # ---------- 1. Python ----------
    print(f"\n[1] Python: {sys.version.split()[0]}  ({sys.executable})")

    # ---------- 2. 版本对照 ----------
    print("\n[2] 版本对照（镜像自带 vs 当前）")
    real_mods = {"torch": "torch", "torch_npu": "torch_npu", "vllm": "vllm",
                 "vllm_ascend": "vllm_ascend", "triton_ascend": "triton_ascend"}
    missing = []
    for k, mod in real_mods.items():
        v = _ver(mod)
        if v is None:
            missing.append(k)
            print(f"    {k:16s} {'(未安装)':>18s}   期望 {EXPECT[k]}")
        else:
            flag = OK if v.startswith(EXPECT[k].split(".post")[0]) else WARN
            print(f"    {k:16s} {v:>18s}   期望 {EXPECT[k]}  {flag}")

    # ---------- 3. torch_npu / NPU ----------
    print("\n[3] torch_npu 与 NPU 设备")
    npu_ok = False
    try:
        import torch  # noqa
        import torch_npu  # noqa: F401
        npu_ok = bool(torch.npu.is_available())
        print(f"    torch.npu.is_available() = {npu_ok}")
        if npu_ok:
            n = torch.npu.device_count()
            print(f"    NPU 数量 = {n}")
            for i in range(n):
                try:
                    props = torch.npu.get_device_properties(i)
                    gb = getattr(props, "total_memory", 0) / 1024 ** 3
                    print(f"      npu:{i}  {torch.npu.get_device_name(i)}  显存 {gb:.1f} GiB")
                except Exception as e:
                    print(f"      npu:{i}  读取属性失败: {e}")
    except Exception as e:
        print(f"    {BAD} 无法使用 NPU: {e}")
        print("    → 若在新建的 conda 环境里，请改用镜像自带环境：")
        print("      REUSE_BASE=1 bash scripts/01_create_env.sh")

    # ---------- 4. CANN / npu-smi ----------
    print("\n[4] CANN 与 npu-smi")
    cann = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME") or "(未设置环境变量)"
    print(f"    ASCEND_HOME_PATH = {cann}")
    if shutil.which("npu-smi"):
        try:
            out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30)
            lines = (out.stdout or "").strip().splitlines()
            for line in lines[:12]:
                print("    | " + line)
        except Exception as e:
            print(f"    npu-smi 执行失败: {e}")
    else:
        print(f"    {WARN} 未找到 npu-smi（不影响训练/推理，仅影响监控）")
        print("    → 可执行: source /usr/local/Ascend/ascend-toolkit/set_env.sh")

    # ---------- 5. NPU 上真实算子验证 ----------
    print("\n[5] NPU 真实算子验证（矩阵乘）")
    if npu_ok:
        try:
            import torch
            a = torch.randn(512, 512, dtype=torch.float16).npu()
            b = torch.randn(512, 512, dtype=torch.float16).npu()
            c = a @ b
            torch.npu.synchronize()
            print(f"    {OK} NPU 矩阵乘成功, 输出 shape={tuple(c.shape)}, "
                  f"均值={c.float().mean().item():.4f}")
        except Exception as e:
            print(f"    {BAD} NPU 算子执行失败: {e}")
    else:
        print(f"    {WARN} 跳过（NPU 不可用）")

    # ---------- 6. 项目依赖 ----------
    print("\n[6] 项目依赖")
    deps = ["pandas", "pyarrow", "numpy", "tqdm", "matplotlib",
            "datasets", "transformers", "safetensors"]
    for d in deps:
        v = _ver(d)
        print(f"    {d:16s} {v if v else '(未安装)'}   {OK if v else BAD}")
    for d in ["llmcompressor", "msmodelslim"]:
        v = _ver(d)
        print(f"    {d:16s} {v if v else '(未安装, 可选)'}")

    # ---------- 汇总 ----------
    print("\n" + "=" * 72)
    if npu_ok:
        print("结论: 环境可用 ✅  可以执行: bash scripts/run_all.sh")
    else:
        print("结论: NPU 不可用 ❌  请先按上面的提示修好 NPU 再跑流程")
    print("=" * 72)
    return 0 if npu_ok else 2


if __name__ == "__main__":
    sys.exit(main())
