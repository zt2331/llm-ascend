"""设备抽象层：自动适配 昇腾 NPU / NVIDIA CUDA / CPU。

★ 这是本项目相对 NVIDIA 版「唯一必须新增的一层」：
  - 昇腾上设备名是 "npu"，且**必须先 import torch_npu** 才能注册该后端；
  - 显存/缓存 API 从 torch.cuda.* 换成 torch.npu.*；
  - 其余 PyTorch 代码基本不用改（这正是国产卡迁移的现实：算子层兼容，运行层要换）。

所有脚本统一调用这里的函数，不要在业务代码里直接写 .cuda() / torch.cuda.*。
"""
import os

import torch

_NPU_OK = None


def has_npu() -> bool:
    """是否可用昇腾 NPU（会自动尝试导入 torch_npu 注册后端）。"""
    global _NPU_OK
    if _NPU_OK is None:
        try:
            import torch_npu  # noqa: F401  导入即注册 npu 后端
            _NPU_OK = bool(torch.npu.is_available())
        except Exception:
            _NPU_OK = False
    return bool(_NPU_OK)


def has_cuda() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def default_device(prefer: str = "auto") -> str:
    """返回设备字符串。prefer: auto | npu | cuda | cpu"""
    if prefer and prefer != "auto":
        return prefer
    if has_npu():
        return "npu"
    if has_cuda():
        return "cuda"
    return "cpu"


def get_device(prefer: str = "auto"):
    d = default_device(prefer)
    if d == "npu":
        import torch_npu  # noqa: F401
    try:
        return torch.device(d if d != "cpu" else "cpu")
    except Exception:
        return torch.device("cpu")


def device_count(prefer: str = "auto") -> int:
    d = default_device(prefer)
    if d == "npu":
        try:
            return int(torch.npu.device_count())
        except Exception:
            return 1
    if d == "cuda":
        try:
            return int(torch.cuda.device_count())
        except Exception:
            return 1
    return 0


def synchronize(prefer: str = "auto") -> None:
    d = default_device(prefer)
    try:
        if d == "npu":
            torch.npu.synchronize()
        elif d == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass


def empty_cache(prefer: str = "auto") -> None:
    d = default_device(prefer)
    try:
        if d == "npu":
            torch.npu.empty_cache()
        elif d == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


def _mem(fn, idx: int = 0) -> float:
    try:
        return round(fn(idx) / 1024 ** 3, 2)  # GiB
    except Exception:
        return -1.0


def memory_allocated_gb(prefer: str = "auto") -> float:
    d = default_device(prefer)
    if d == "npu":
        return _mem(torch.npu.memory_allocated)
    if d == "cuda":
        return _mem(torch.cuda.memory_allocated)
    return -1.0


def memory_reserved_gb(prefer: str = "auto") -> float:
    d = default_device(prefer)
    if d == "npu":
        return _mem(torch.npu.memory_reserved)
    if d == "cuda":
        return _mem(torch.cuda.memory_reserved)
    return -1.0


def memory_total_gb(prefer: str = "auto") -> float:
    """设备总显存（HBM）。"""
    d = default_device(prefer)
    try:
        if d == "npu":
            props = torch.npu.get_device_properties(0)
            return round(getattr(props, "total_memory", 0) / 1024 ** 3, 2)
        if d == "cuda":
            props = torch.cuda.get_device_properties(0)
            return round(props.total_memory / 1024 ** 3, 2)
    except Exception:
        pass
    return -1.0


def device_name(prefer: str = "auto") -> str:
    d = default_device(prefer)
    try:
        if d == "npu":
            return torch.npu.get_device_name(0)
        if d == "cuda":
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return d


def describe(prefer: str = "auto") -> dict:
    """给自检脚本用的设备信息汇总。"""
    d = default_device(prefer)
    return {
        "backend": d,
        "device_name": device_name(prefer),
        "device_count": device_count(prefer),
        "memory_total_gb": memory_total_gb(prefer),
        "torch": getattr(torch, "__version__", "?"),
        "torch_npu": _safe_ver("torch_npu"),
        "cann": _cann_version(),
    }


def _safe_ver(mod: str) -> str:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "?")
    except Exception:
        return "(未安装)"


def _cann_version() -> str:
    """读取 CANN 版本（从环境变量或 version.info）。"""
    for k in ("ASCEND_TOOLKIT_VERSION", "CANN_VERSION"):
        if os.environ.get(k):
            return os.environ[k]
    p = "/usr/local/Ascend/ascend-toolkit/latest/version.cfg"
    try:
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "version" in line.lower():
                    return line.strip()
    except Exception:
        pass
    return "(未知)"
