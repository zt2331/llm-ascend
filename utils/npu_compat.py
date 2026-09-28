#!/usr/bin/env python
"""昇腾 NPU 缺失算子的兼容层。

问题
----
部分 PyTorch 算子在 Ascend NPU 上没有实现，torch_npu 会走 **CPU fallback**
（把张量拷回 CPU 算完再拷回来）。本环境的 fallback 路径有 bug，
实测在 GPTQ 量化时直接崩掉：

    [W] VariableFallbackKernel.cpp:250 Warning: CAUTION: The operator
        'aten::cholesky_inverse' is not currently supported on the NPU backend
        and will fall back to run on the CPU.
    ...
    RuntimeError: device_allocator INTERNAL ASSERT FAILED at
    "/pytorch/c10/core/CachingDeviceAllocator.h":116
    Allocator for npu is not a DeviceAllocator.

触发点：**`aten::cholesky_inverse`**（GPTQ 求 Hessian 逆矩阵时必用）
    - llm-compressor: `modifiers/gptq/gptq_quantize.py` → `H = torch.cholesky_inverse(H)`
    - 副作用还不只是崩：CPU fallback 慢到 **88 秒/层**（实测），
      64 层就是 ~90 分钟，而且最后仍然抛异常，等于白跑。

做法
----
不去修 torch_npu，而是把这类算子**显式搬到 CPU 上算，再搬回原设备**，
从而完全不触发那条有 bug 的 fallback 分支。

开销可接受：Hessian 是 `[in, in]` 方阵
    - hidden 5120  → 5120²  × 4B ≈ 105 MB
    - intermediate 17408 → 17408² × 4B ≈ 1.2 GB
CPU 完全放得下，且求逆本身是 O(n³)，无论放哪都慢，改成 CPU 只是「不再崩」。

用法
----
    from utils.npu_compat import patch_unsupported_linalg
    patch_unsupported_linalg()      # 在调用 oneshot / GPTQ 之前执行一次
"""
import torch

_PATCHED = False
_APPLIED = []

# 已确认在 NPU 上缺失、且 CPU fallback 会崩的算子
# 格式: (torch 模块上的属性名, 是否属于 torch.linalg 子模块)
_BROKEN = [
    ("cholesky_inverse", False),   # aten::cholesky_inverse —— GPTQ 必用，实测崩溃
]


def _wrap_cpu(fn, qualname):
    """包一层：输入在非 CPU 设备上就先搬到 CPU 算，算完搬回去。"""

    def wrapper(*args, **kwargs):
        out = kwargs.pop("out", None)
        dev = None
        for a in args:
            if isinstance(a, torch.Tensor):
                dev = a.device
                break
        if dev is None or dev.type == "cpu":
            return fn(*args, **kwargs)          # CPU 输入不需要绕

        cpu_args = [a.detach().cpu() if isinstance(a, torch.Tensor) else a
                    for a in args]
        r = fn(*cpu_args, **kwargs)

        if out is not None:                     # 保持 out= 语义
            out.copy_(r.to(out.device))
            return out
        return r.to(dev)

    wrapper.__name__ = qualname
    wrapper.__qualname__ = qualname
    wrapper.__doc__ = f"[npu_compat] {qualname}: 强制在 CPU 上计算后搬回原设备。"
    return wrapper


def patch_unsupported_linalg(verbose=True):
    """把 NPU 未实现、且 fallback 会崩的算子改为在 CPU 上执行。

    幂等：重复调用只生效一次。返回本次实际打补丁的算子名列表。
    """
    global _PATCHED
    if _PATCHED:
        return list(_APPLIED)
    _PATCHED = True

    for attr, is_linalg in _BROKEN:
        holder = torch.linalg if is_linalg else torch
        orig = getattr(holder, attr, None)
        if orig is None:
            continue
        qual = f"torch.{'linalg.' if is_linalg else ''}{attr}"
        setattr(holder, attr, _wrap_cpu(orig, qual))
        _APPLIED.append(qual)

    if verbose and _APPLIED:
        print(f"[npu_compat] 已绕开 NPU 缺失算子（改走 CPU）: {', '.join(_APPLIED)}")
    return list(_APPLIED)


def patch_needed(prefer_device="auto"):
    """只有真的在加速器上跑才需要打补丁（CPU 上纯属多余）。"""
    from utils import device as dev
    need = dev.default_device(prefer_device) != "cpu"
    if need:
        patch_unsupported_linalg()
    return need


if __name__ == "__main__":
    print(f"torch {torch.__version__}")
    applied = patch_unsupported_linalg()
    print(f"打补丁的算子: {applied or '（无）'}")

    # 自测：在可用设备上验证语义不变
    _dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.randn(8, 8, dtype=torch.float64, device=_dev)
    S = A @ A.t() + 8 * torch.eye(8, dtype=torch.float64, device=_dev)
    L = torch.linalg.cholesky(S)
    got = torch.cholesky_inverse(L)
    ref = torch.linalg.inv(S)
    err = (got.cpu() - ref.cpu()).abs().max().item()
    print(f"设备={_dev}  输出 device={got.device}  与 inv(S) 最大误差={err:.2e}")
    assert got.device.type == _dev
    assert err < 1e-8
    print("OK")
