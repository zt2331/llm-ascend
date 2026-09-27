"""集中配置（昇腾 NPU 版）。

- 模型：自动搜索常见位置（环境变量 > qwen_model/ > ModelArts 目录 > ModelScope/HF 缓存），
  也可显式 `export MODEL_PATH=/path/to/Qwen3.6-27B`。
- 路径：全部基于项目根目录的**绝对路径**，任意工作目录运行都不错位。
- 设备：交由 utils.device 自动识别 NPU / CUDA / CPU。
"""
import glob
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME_HINT = os.environ.get("MODEL_NAME", "Qwen3.6-27B")

# 产物目录
OUT = os.path.join(PROJECT_ROOT, "output_models")
QUANT_DIR = os.path.join(OUT, "quantized")
PRUNE_DIR = os.path.join(OUT, "pruned")
DISTILL_DIR = os.path.join(OUT, "distilled")
RESULTS_DIR = os.path.join(OUT, "results")

# 数据目录（校准/评测语料）
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CALIB_DIR = os.path.join(DATA_DIR, "calib")
TEST_DIR = os.path.join(DATA_DIR, "test")


# ------------------------------------------------------------------
# 模型自动搜索
# ------------------------------------------------------------------
def _is_model_dir(p):
    return os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json"))


def _hint_match(p):
    """路径里是否像目标模型（qwen/27b 等，命中越多越优先）。"""
    s = p.lower()
    score = 0
    for kw in ("qwen", "27b"):
        if kw in s:
            score += 1
    if "config.json" in s:
        score += 0
    return score


def _walk_candidates(root, max_depth=3):
    """在 root 下最多 max_depth 层找合法模型目录。"""
    root = os.path.abspath(root)
    results = []
    if not os.path.isdir(root):
        return results
    base_depth = root.rstrip("/").count("/")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.rstrip("/").count("/") - base_depth
        if depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if "config.json" in filenames:
            results.append(dirpath)
    return results


def _search_roots():
    home = os.path.expanduser("~")
    roots = [
        os.path.join(PROJECT_ROOT, "qwen_model"),
        os.path.join(PROJECT_ROOT, "models"),
        os.path.join(home, "work"),
        os.path.join(home, "model"),
        os.path.join(home, "models"),
        "/home/ma-user/work",
        "/home/ma-user/modelarts/inputs",
        "/home/ma-user/modelarts/user-job-dir",
        "/home/ma-user/modelarts",
        "/cache",
        "/data",
    ]
    return [r for r in roots if os.path.isdir(r)]


def _cache_candidates():
    pats = [
        os.path.expanduser("~/.cache/modelscope/hub/models/*/*"),
        os.path.expanduser("~/.cache/modelscope/hub/*/*"),
        os.path.expanduser("~/.cache/huggingface/hub/models--*--*/snapshots/*"),
    ]
    out = []
    for p in pats:
        out.extend(glob.glob(p))
    return out


def find_model_candidates():
    """返回所有候选模型目录（按匹配度排序）。"""
    cands = []
    for r in _search_roots():
        cands.extend(_walk_candidates(r))
        if _is_model_dir(r):
            cands.append(r)
    for c in _cache_candidates():
        if _is_model_dir(c):
            cands.append(c)
    # 去重 + 排序（命中 qwen/27b 的优先，路径短的优先）
    uniq = sorted(set(os.path.abspath(c) for c in cands if _is_model_dir(c)))
    uniq.sort(key=lambda p: (-_hint_match(p), len(p)))
    return uniq


def resolve_model_path(verbose=False):
    env = os.environ.get("MODEL_PATH")
    if env and _is_model_dir(env):
        return os.path.abspath(env)
    if env and not _is_model_dir(env):
        print(f"[WARN] MODEL_PATH={env} 不是合法模型目录（缺 config.json），将自动搜索。")
    cands = find_model_candidates()
    if cands:
        if verbose:
            print("[INFO] 候选模型目录：")
            for c in cands[:10]:
                print("   ", c, "← 已选用" if c == cands[0] else "")
        return cands[0]
    return None


MODEL_PATH = resolve_model_path()
MODEL_TAG = os.environ.get("MODEL_TAG") or (
    re.sub(r"[_ /:]+", "-", os.path.basename(MODEL_PATH)) if MODEL_PATH else MODEL_NAME_HINT
)


def ensure_dirs():
    for d in (OUT, QUANT_DIR, PRUNE_DIR, DISTILL_DIR, RESULTS_DIR, CALIB_DIR, TEST_DIR):
        os.makedirs(d, exist_ok=True)


def require_model():
    """需要模型时调用；找不到就给出清晰指引。"""
    global MODEL_PATH, MODEL_TAG
    if MODEL_PATH is None:
        MODEL_PATH = resolve_model_path(verbose=True)
    if MODEL_PATH is None:
        raise SystemExit(
            "未找到 base 模型目录（需包含 config.json）。请任选一种方式：\n"
            "  1) export MODEL_PATH=/path/to/Qwen3.6-27B\n"
            "  2) mkdir -p qwen_model && 把模型软链/拷贝进去\n"
            "  3) 运行 python scripts/03_find_model.py 查看搜索到了哪些目录"
        )
    return MODEL_PATH


if __name__ == "__main__":
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("MODEL_PATH   =", MODEL_PATH)
    print("MODEL_TAG    =", MODEL_TAG)
    print("OUT          =", OUT)
