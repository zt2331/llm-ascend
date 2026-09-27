"""数据读写：parquet + `text` 列。

数据已固定，**不做任何格式猜测/兜底**：
    data/calib/validation.parquet   校准集
    data/test/test.parquet          评测集
    列名必须为 `text`
缺文件或缺列直接报错，不静默回退。
"""
import os

TEXT_COL = "text"


class Dataset:
    """最小 Dataset 垫片：仅实现本项目用到的 from_parquet / select / 迭代 / 长度。"""

    def __init__(self, rows):
        self._rows = rows                       # list[dict]

    @classmethod
    def from_parquet(cls, path, **kwargs):
        return cls(load_rows(path))

    def select(self, rng):
        return Dataset([self._rows[i] for i in list(rng)])

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, i):
        return self._rows[i]

    @property
    def column_names(self):
        return list(self._rows[0].keys()) if self._rows else []


def load_rows(path):
    """读 parquet → [{"text": ...}, ...]。文件/列缺失即报错。"""
    import pandas as pd

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"数据文件不存在: {path}\n"
            f"  本项目数据已固定，需要:\n"
            f"    data/calib/validation.parquet  (校准集)\n"
            f"    data/test/test.parquet         (评测集)\n"
            f"  两列名均为 `text`。")
    df = pd.read_parquet(path)
    if TEXT_COL not in df.columns:
        raise ValueError(
            f"{path} 缺少 `{TEXT_COL}` 列，实际列为 {list(df.columns)}。"
            f"请把文本列命名为 `{TEXT_COL}`。")
    return [{TEXT_COL: str(t)} for t in df[TEXT_COL].tolist()]


def save_rows(texts, path):
    """[str] 或 [{"text": ...}] → parquet。"""
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if texts and isinstance(texts[0], dict):
        texts = [r[TEXT_COL] for r in texts]
    pd.DataFrame({TEXT_COL: list(texts)}).to_parquet(path, index=False)
    return path
