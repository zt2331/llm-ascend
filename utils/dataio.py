"""轻量数据读写（替代 datasets 库）—— 只依赖 pandas + pyarrow。

为什么不用 `datasets`：
  1. `datasets` 会拉起一大堆依赖，容易和镜像里已装的 transformers 版本冲突；
  2. 本项目只需要「读 parquet 的 text 列」这一件事，pandas 足够；
  3. 昇腾镜像里 pandas 通常已内置，只需再装 pyarrow。

提供与 `datasets.Dataset` 兼容的最小垫片，业务代码改动最小：
    from utils.dataio import Dataset
    ds = Dataset.from_parquet("xxx.parquet").select(range(16))
    for r in ds: ... r["text"] ...
"""
import os


class Dataset:
    """极简 Dataset 垫片：仅实现本项目用到的 from_parquet / select / 迭代 / 长度。"""

    def __init__(self, rows):
        self._rows = rows                       # list[dict]

    @classmethod
    def from_parquet(cls, path, **kwargs):
        rows = load_rows(path)
        return cls(rows)

    def select(self, rng):
        idx = list(rng)
        return Dataset([self._rows[i] for i in idx])

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, i):
        return self._rows[i]

    @property
    def column_names(self):
        return list(self._rows[0].keys()) if self._rows else []


def load_rows(path, text_col="text"):
    """读 parquet → [{"text": ...}, ...]。若没有 text 列，自动取第一个字符串列。"""
    import pandas as pd
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到数据文件: {path}")
    df = pd.read_parquet(path)
    if text_col not in df.columns:
        cands = [c for c in df.columns if df[c].dtype == object] or list(df.columns)
        df = df.rename(columns={cands[0]: text_col})
    return [{text_col: str(t)} for t in df[text_col].tolist()]


def save_rows(rows, path, text_col="text"):
    """[{"text": ...}] 或 [str] → parquet。"""
    import pandas as pd
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    if rows and isinstance(rows[0], dict):
        texts = [r[text_col] for r in rows]
    else:
        texts = list(rows)
    pd.DataFrame({text_col: texts}).to_parquet(path, index=False)
    return path
