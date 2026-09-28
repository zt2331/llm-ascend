#!/usr/bin/env python
"""npu_mem.py —— 不看 npu-smi 也能查 NPU 显存占用，并找出是谁占着。

为什么需要它：容器里 `npu-smi info` 常报
    npu get board type failed. ret is -9005
（DCMI 管理接口不通），但**计算链路是好的**。这时用 torch_npu 的
mem_get_info() 一样能拿到准确的空闲/总量，再用 /proc 反查占用进程。

用法:
    python scripts/npu_mem.py                 # 只报告
    python scripts/npu_mem.py --need 55.14    # 需要 55.14 GiB，不够则 exit 1
    python scripts/npu_mem.py --kill          # 交互式确认后杀掉占用进程
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GB = 1024 ** 3


def mem_info():
    """返回 (free_gb, total_gb)；取不到返回 (None, None)。"""
    try:
        import torch
        import torch_npu  # noqa: F401  注册 NPU 后端
    except Exception as e:
        print(f"[WARN] torch_npu 不可用: {e}")
        return None, None
    try:
        free, total = torch.npu.mem_get_info()
        return free / GB, total / GB
    except Exception:
        try:
            props = torch.npu.get_device_properties(0)
            total = getattr(props, "total_memory", 0) / GB
            alloc = torch.npu.memory_allocated() / GB
            return total - alloc, total
        except Exception as e:
            print(f"[WARN] 无法读取 NPU 显存: {e}")
            return None, None


def holders(exclude_self=True):
    """通过 /proc 反查哪些进程打开了 /dev/davinci* 等 NPU 设备节点。"""
    me = os.getpid()
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        p = int(pid)
        if exclude_self and p == me:
            continue
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except Exception:
            continue
        n = 0
        for fd in fds:
            try:
                tgt = os.readlink(f"/proc/{pid}/fd/{fd}")
            except Exception:
                continue
            if any(k in tgt for k in ("/dev/davinci", "devmm_svm", "hisi_hdc")):
                n += 1
        if not n:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        except Exception:
            cmd = "(无法读取)"
        out.append((p, n, cmd[:160] or "(无 cmdline)"))
    return sorted(out, key=lambda x: -x[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--need", type=float, default=None,
                    help="需要多少 GiB 空闲；不足则 exit 1")
    ap.add_argument("--need-gmu", type=float, default=None,
                    help="按 vLLM 的 gpu-memory-utilization 计算需求（= gmu × 总量）；不足则 exit 1")
    ap.add_argument("--kill", action="store_true",
                    help="杀掉占用 NPU 的进程（会逐个确认）")
    args = ap.parse_args()

    print("=" * 66)
    print(" NPU 显存检查（不依赖 npu-smi）")
    print("=" * 66)

    free, total = mem_info()
    if free is None:
        print(" 无法获取显存信息。")
        return 3
    used = total - free
    print(f" 总量: {total:6.2f} GiB")
    print(f" 已用: {used:6.2f} GiB  ({used/total*100:4.1f}%)")
    print(f" 空闲: {free:6.2f} GiB  ({free/total*100:4.1f}%)")

    hs = holders()
    print(f"\n 占用 NPU 设备的进程（{len(hs)} 个）:")
    if not hs:
        print("   （无——若显存仍被占用，可能是内核态泄漏或已在别的 PID namespace）")
    for pid, n, cmd in hs:
        print(f"   PID {pid:>7}  句柄×{n:<3}  {cmd}")

    if used > total * 0.10 and not hs:
        print("\n [提示] 显存被占用，但本容器内找不到占用进程。三种可能：")
        print("        ① 上一条命令还在别的终端跑着 —— `ps -ef | grep -E 'python|vllm'`")
        print("        ② 同一张卡被别的容器/用户共享（容器只能看到自己的 PID 空间，")
        print("           看不到对方的进程，此时只能等或换卡）")
        print("        ③ 内核态/驱动层尚未回收")

    if args.kill and hs:
        import signal
        for pid, _n, cmd in hs:
            try:
                ans = input(f" 杀掉 PID {pid} ({cmd[:60]})? [y/N] ").strip().lower()
            except EOFError:
                ans = ""
            if ans == "y":
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f"   已发送 SIGTERM 到 {pid}")
                except Exception as e:
                    print(f"   失败: {e}")

    need = args.need
    if args.need_gmu is not None:
        # vLLM 的判定是 free >= gmu × total（注意是按【总量】的比例，不是按空闲）
        need = args.need_gmu * total
        print(f"\n vLLM 口径: gpu-memory-utilization={args.need_gmu} "
              f"→ 需要空闲 ≥ {need:.2f} GiB（= {args.need_gmu} × {total:.2f}）")

    if need is not None:
        print(f"\n 需要空闲 {need:.2f} GiB，实际 {free:.2f} GiB")
        if free < need:
            print(" ❌ 不足 —— vLLM 会直接报 ValueError 起不来（那串 60 行 traceback 的根因）")
            print("    处理: 杀掉占用进程 / 调低 GMU / 调低 MAX_MODEL_LEN")
            return 1
        print(" ✅ 充足")

    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
