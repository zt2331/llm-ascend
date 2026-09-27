#!/usr/bin/env python
"""02_check_env.py —— 昇腾环境自检（跑任何流程前先跑这个）。

检查内容：
  1. Python 与关键包版本（用 importlib.metadata 读包元数据，避免误报）
  2. torch_npu 是否可用、NPU 设备数量/型号/显存
  3. CANN 环境变量 + npu-smi
  4. 在 NPU 上做一次真实张量运算（验证算子可用）
  5. 项目依赖是否齐全
"""
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"

# 包名映射：(发行包名, 导入模块名, 期望版本)
# ⚠️ 注意：很多包没有 __version__ 属性，必须用 importlib.metadata 读包元数据，
#    否则会出现"装了却显示没装"的误报（如 vllm-ascend）。
PKGS = [
    ("torch",        "torch",         "2.10.0"),
    ("torch-npu",    "torch_npu",     "2.10.0.post4"),
    ("vllm",         "vllm",          "0.23.0"),
    ("vllm-ascend",  "vllm_ascend",   "0.23.0"),
    ("triton-ascend", "triton",       "3.2.2"),   # 模块名是 triton，发行包名是 triton-ascend
    ("transformers", "transformers",  None),
    ("numpy",        "numpy",         None),
    ("pandas",       "pandas",        None),
    ("pyarrow",      "pyarrow",       None),
    ("matplotlib",   "matplotlib",    None),
    ("safetensors",  "safetensors",   None),
    ("tqdm",         "tqdm",          None),
]

# 项目跑通所需（缺失会直接失败或出不了图）
REQUIRED = ["numpy", "pandas", "pyarrow"]
CHART_REQUIRED = ["matplotlib"]
OPTIONAL = ["llmcompressor", "msmodelslim"]


def pkg_version(dist_name):
    """从包元数据读版本（最可靠）。返回 None 表示未安装。"""
    try:
        from importlib.metadata import version, PackageNotFoundError
    except Exception:
        return None
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def can_import(module_name):
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def main():
    print("=" * 74)
    print("昇腾 NPU 环境自检")
    print("=" * 74)

    # ---------- 1. Python ----------
    print(f"\n[1] Python: {sys.version.split()[0]}")
    print(f"    解释器: {sys.executable}")
    in_conda = "conda" in sys.executable or os.environ.get("CONDA_PREFIX")
    print(f"    环境类型: {'conda 环境' if in_conda else '镜像自带系统 Python（无需 conda 即可运行）'}")

    # ---------- 2. 版本对照 ----------
    print("\n[2] 关键包版本（读包元数据，非模块属性）")
    print(f"    {'包名':<16}{'版本':>18}   {'期望':<16}{'状态'}")
    missing_pkgs = []
    for dist, mod, expect in PKGS:
        v = pkg_version(dist)
        imp = can_import(mod)
        if v is None:
            if imp:
                v = "(可导入,无元数据)"
            else:
                missing_pkgs.append(dist)
                print(f"    {dist:<16}{'(未安装)':>18}   {str(expect or '-'):<16}{BAD}")
                continue
        status = OK
        if expect and isinstance(v, str) and v[0].isdigit():
            base = expect.split(".post")[0]
            status = OK if v.startswith(base) else WARN
        print(f"    {dist:<16}{v:>18}   {str(expect or '-'):<16}{status}")

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
        print("    → 检查: source /usr/local/Ascend/ascend-toolkit/set_env.sh")

    # ---------- 4. CANN ----------
    print("\n[4] CANN 与 npu-smi")
    print(f"    ASCEND_HOME_PATH = {os.environ.get('ASCEND_HOME_PATH', '(未设置)')}")
    if shutil.which("npu-smi"):
        try:
            out = subprocess.run(["npu-smi", "info"], capture_output=True,
                                 text=True, timeout=30)
            lines = (out.stdout or out.stderr or "").strip().splitlines()
            if not lines:
                print(f"    {WARN} npu-smi 无输出（容器内常见，不影响推理）")
            for line in lines[:8]:
                print("    | " + line)
        except Exception as e:
            print(f"    {WARN} npu-smi 执行异常: {e}")
    else:
        print(f"    {WARN} 未找到 npu-smi（仅影响硬件监控，不影响推理）")

    # ---------- 5. NPU 算子验证 ----------
    print("\n[5] NPU 真实算子验证（矩阵乘）")
    if npu_ok:
        try:
            import torch
            a = torch.randn(512, 512, dtype=torch.float16).npu()
            b = torch.randn(512, 512, dtype=torch.float16).npu()
            c = a @ b
            torch.npu.synchronize()
            print(f"    {OK} NPU 矩阵乘成功, shape={tuple(c.shape)}, "
                  f"均值={c.float().mean().item():.4f}")
        except Exception as e:
            print(f"    {BAD} NPU 算子执行失败: {e}")
    else:
        print(f"    {WARN} 跳过（NPU 不可用）")

    # ---------- 6. 项目依赖 ----------
    print("\n[6] 项目依赖")
    miss_req, miss_chart = [], []
    print("    必需（缺了跑不动）:")
    for d in REQUIRED:
        v = pkg_version(d)
        print(f"      {d:<16}{v if v else '(未安装)':>14}   {OK if v else BAD}")
        if not v:
            miss_req.append(d)
    print("    出图所需（缺了只有表没有图）:")
    for d in CHART_REQUIRED:
        v = pkg_version(d)
        print(f"      {d:<16}{v if v else '(未安装)':>14}   {OK if v else BAD}")
        if not v:
            miss_chart.append(d)
    print("    可选:")
    for d in OPTIONAL:
        v = pkg_version(d)
        # 已装也打 [OK]，保持与上面两栏一致（之前只打印版本号，容易误以为没装）
        print(f"      {d:<16}{v if v else '(未安装)':>14}   {OK if v else '[SKIP]'}")

    # ---------- 7. 可用量化路径 ----------
    print("\n[7] 可用量化路径")
    has_ms = pkg_version("msmodelslim") is not None
    has_lc = pkg_version("llmcompressor") is not None
    print(f"      {'昇腾原生 msModelSlim (W8A8)':<30}"
          f"{'✅ 可用  -> quantize/ascend_quant.py' if has_ms else '❌ 不可用（缺 msmodelslim）'}")
    print(f"      {'llm-compressor (调库量化)':<30}"
          f"{'✅ 可用  -> quantize/gen_llmcomp.py / w8a8_smooth.py' if has_lc else '❌ 不可用（缺 llmcompressor）'}")
    print(f"      {'手写 AWQ / GPTQ (纯 PyTorch)':<30}"
          f"✅ 始终可用  -> quantize/manual_awq.py / manual_gptq.py")

    # ---------- 汇总 ----------
    print("\n" + "=" * 74)
    if not npu_ok:
        print("结论: NPU 不可用 ❌  请先修好 NPU 再跑流程")
        return 2

    if miss_req or miss_chart:
        need = miss_req + miss_chart
        print(f"结论: NPU 可用 ✅  但缺少依赖: {', '.join(need)}")
        print("      一键安装（用国内镜像）:")
        print(f"        pip install {' '.join(need)} -i "
              "https://pypi.tuna.tsinghua.edu.cn/simple")
        print("      装完再跑: bash scripts/run_all.sh")
        return 3

    print("结论: 环境完全就绪 ✅  可以执行: bash scripts/run_all.sh")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
