#!/usr/bin/env python
"""04_check_data.py —— 校验数据文件（数据已固定，只检查，不生成、不兜底）。

约定的数据（随仓库一起版本管理）：
    data/calib/validation.parquet   校准集，列名 text
    data/test/test.parquet          评测集，列名 text

本脚本只做校验与统计，缺文件/缺列直接失败，不做任何自动生成或格式转换。

用法:
    python scripts/04_check_data.py
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.dataio import load_rows, TEXT_COL, Dataset


def check_one(name, path):
    print(f"\n[{name}] {path}")
    if not os.path.isfile(path):
        print(f"   [FAIL] 文件不存在")
        return None
    try:
        rows = load_rows(path)
    except Exception as e:
        print(f"   [FAIL] {e}")
        return None
    if not rows:
        print("   [FAIL] 空文件")
        return None

    texts = [r[TEXT_COL] for r in rows]
    lens = [len(t) for t in texts]
    chars = sum(lens)
    print(f"   [ OK ] {len(texts)} 条 | 列名 {TEXT_COL}")
    print(f"          长度: 平均 {statistics.mean(lens):.0f} / 最短 {min(lens)} / 最长 {max(lens)} 字符")
    print(f"          总量: {chars:,} 字符 ≈ {chars // 2:,} tokens（按 2 字符/token 粗估）")
    print(f"          首条: {texts[0][:70].strip()!r}")
    return {"n": len(texts), "chars": chars}


def main():
    print("=" * 74)
    print("数据校验（数据已固定，随 git 版本管理）")
    print("=" * 74)

    calib = check_one("校准集", config.CALIB_PARQUET)
    test = check_one("评测集", config.TEST_PARQUET)

    print("\n" + "=" * 74)
    if not calib or not test:
        print("结论: ❌ 数据不完整，请补齐上述文件后重试")
        print("=" * 74)
        return 1

    # 复读一次，确认各脚本用的 Dataset 垫片能正常工作
    for name, p in (("校准", config.CALIB_PARQUET), ("评测", config.TEST_PARQUET)):
        d = Dataset.from_parquet(p)
        n = len(d.select(range(min(8, len(d)))))
        print(f"   [校验] {name}集 Dataset 垫片可用（select(8) -> {n} 条）")

    # 推荐参数：校准集越多，W8A8 的激活/平滑系数越准
    rec_calib = min(128, calib["n"])
    rec_seq = 1024
    print("\n结论: ✅ 数据就绪")
    print(f"      校准 {calib['n']} 条 / 评测 {test['n']} 条")
    print("\n推荐参数（可直接用）:")
    print(f"      CALIB={rec_calib}  SEQ={rec_seq}  KEEP=54")
    print("      QMETHODS=smooth,rtn KEEP=54 CALIB=%d SEQ=%d STAGES=prune,quantize \\"
          % (rec_calib, rec_seq))
    print("          bash scripts/run_all.sh")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
