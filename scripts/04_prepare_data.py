#!/usr/bin/env python
"""04_prepare_data.py —— 准备校准/评测语料（保证「一键跑通」有数据）。

优先级：
  1. 已有 data/calib/*.parquet + data/test/*.parquet  → 直接使用（推荐，你上传真实数据）
  2. 已有 data/raw/corpus.txt                          → 切分后写成 parquet
  3. 联网时尝试拉取 wikitext-2-raw-v1                  → 切分后写成 parquet
  4. 全失败                                            → 用脚本内置兜底语料（保证能跑）

产物统一列名: text
用法:
    python scripts/04_prepare_data.py --calib 64 --test 32
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

BUILTIN_FALLBACK = """
人工智能正在改变软件工程的实践方式。大语言模型通过海量文本预训练获得语言理解能力，
再通过指令微调与人类反馈对齐，从而具备对话、写作、代码生成与推理等通用能力。
在推理阶段，模型逐个生成 token，每一步都需要读取全部权重，因此显存带宽往往成为瓶颈。

Quantization reduces the numerical precision of model weights and activations.
A typical affine mapping is q = round(x / s) + zp, and the dequantized value is
x_hat = (q - zp) * s, where s is the scale and zp is the zero point.
Symmetric quantization fixes the zero point at zero and is common for weights,
while asymmetric quantization adapts to skewed activation distributions.

模型压缩通常包含三类技术：量化、剪枝与知识蒸馏。量化降低数值位宽，直接减小显存并提升解码速度；
结构化剪枝删除冗余的网络层或通道，使模型真正变小；知识蒸馏则用大模型作为教师，
让小模型学习其软标签分布，从而恢复压缩带来的精度损失。

PagedAttention stores the key and value cache in fixed-size blocks instead of one
contiguous buffer per sequence. This drastically reduces memory fragmentation and
allows the server to batch far more concurrent requests at the same memory budget.
Continuous batching schedules at the iteration level, so finished sequences leave the
batch immediately and new requests join without waiting for the slowest one.

连续批处理与分页注意力共同提升了推理服务的吞吐。吞吐与延迟之间存在权衡关系：
提高并发数通常先带来吞吐上升，当计算或显存接近饱和后，排队时间会快速增长，
导致首 token 延迟与每 token 延迟同时恶化。因此需要按业务目标选择工作点。

Knowledge distillation transfers the dark knowledge encoded in the teacher's soft
probability distribution. Using a temperature T smooths the distribution and reveals
relative similarities between classes that hard labels cannot express.
Multiplying the KL divergence by T squared keeps the gradient magnitude stable.

量化误差主要来自舍入、截断以及离群值对缩放因子的影响。分组量化把权重每 128 个元素分为一组，
每组独立计算缩放因子，从而把离群值的影响限制在局部。激活感知权重量化会先统计每个输入通道的
激活幅度，对重要通道施加缩放保护，再将该缩放折叠进前一层的参数中，推理时无需额外计算。

国产加速卡为大模型推理提供了另一种选择。昇腾 NPU 采用达芬奇架构，包含矩阵计算单元、
向量计算单元与标量计算单元。其软件栈 CANN 提供图引擎、运行时与算子库，
推理框架 MindIE 与社区插件 vllm-ascend 让主流开源模型可以在昇腾硬件上运行。

迁移到国产硬件时常见的差异包括：算子覆盖度不足导致回退到 CPU 执行；
动态形状触发图重编译从而引起延迟抖动；量化格式支持不同，
例如昇腾对八比特整型量化支持较成熟，而四比特权重压缩在生态中的支持面较窄。

向量数据库为检索增强生成提供支撑。文档经过清洗与切片后被嵌入为稠密向量，
查询时通过近似最近邻搜索召回相关片段，再由重排模型精排，最后拼接进提示词交给大模型生成答案。
切片长度、重叠比例与召回数量都会显著影响最终问答质量。

A production inference service usually separates concerns across layers: a load
balancer distributes traffic, a gateway handles authentication, rate limiting,
truncation and streaming, and the inference engine focuses on scheduling and memory
management. Observability across all layers turns a black box into a diagnosable system.

监控体系通常采集三层指标：网关层的请求量、错误率与限流次数；
推理引擎层的排队请求数、缓存使用率与首 token 延迟；硬件层的利用率、显存与温度。
通过对比三层的异常时序，可以快速定位瓶颈究竟出现在网络、调度还是计算资源上。

模型上线前的评测需要覆盖精度与性能两个维度。精度方面常用困惑度与若干标准基准，
例如多学科知识、数学推理、事实性与常识推理；性能方面关注解码吞吐、首 token 延迟、
显存占用以及并发下的分位延迟。只有两个维度同时达标，压缩方案才具备生产价值。

分布式推理通过张量并行把单层权重切分到多张卡上，每层计算后需要一次集合通信来完成同步。
通信开销随并行度上升，因此并行规模并非越大越好。数据并行则在每张卡上放置完整模型，
不同卡处理不同请求，适用于单卡可以容纳整个模型的场景。

投机解码使用一个小模型快速草拟若干 token，再由大模型一次性并行验证，
接受其中正确的部分。当草拟模型的接受率较高时，整体解码速度可以明显提升；
反之则会因为额外的验证开销而变慢，因此需要根据任务分布评估收益。

长上下文场景对显存管理提出了更高要求。除了采用分页注意力降低碎片之外，
还可以压缩缓存的数据类型、复用相同前缀的缓存、以及在预填充阶段进行分块调度，
避免超长输入的预填充阻塞正在解码的请求。

缓存复用是降低首 token 延迟的有效手段。当大量请求共享同一段系统提示词时，
把这段前缀对应的键值缓存保存下来，后续请求可以直接复用，跳过重复的预填充计算。
在客服、审核等固定提示词的场景中，收益尤为显著。

量化后的模型必须经过真实推理验证，而不能仅以能否加载作为判断标准。
打包顺序、缩放因子布局以及配置文件中的量化描述必须与推理内核严格一致，
否则会出现模型可以启动但输出乱码的情况。因此需要用小批量确定性提示词检查输出语义是否正常。

剪枝后的模型需要同步更新配置中的层数字段，否则在构造网络时会出现索引越界。
结构性剪枝删除整个层或整个通道，可以直接减小矩阵规模并被通用推理框架加速；
非结构性剪枝仅将个别权重置零，需要专门的稀疏内核才能获得速度收益。

评测结果应当与公开报告对照，以确认评测口径一致。不同实现的少样本数量、
提示模板与答案抽取方式都会影响分数，因此必须在同一口径下比较压缩前后的模型，
否则差异可能来自评测流程而非压缩本身。
""".strip()


def _have_parquet(d):
    return bool(glob.glob(os.path.join(d, "*.parquet")))


def _write_parquet(texts, out_path):
    import pandas as pd
    df = pd.DataFrame({"text": texts})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    return len(df)


def _split_docs(raw, calib_n, test_n):
    """把长文切成段落，取前 calib_n 段做校准、后 test_n 段做测试。"""
    docs = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 80]
    if not docs:
        docs = [p.strip() for p in raw.split("\n") if len(p.strip()) > 40]
    need = calib_n + test_n
    if len(docs) < need:
        # 不足则循环扩充（demo 用；真实场景请放足量语料）
        reps = (need // max(len(docs), 1)) + 1
        docs = (docs * reps)
    calib = docs[:calib_n]
    test = docs[calib_n:calib_n + test_n]
    return calib, test


def try_download():
    """尝试联网拉 wikitext-2-raw-v1（失败返回 None）。"""
    try:
        from datasets import load_dataset
        print("  [尝试] 下载 wikitext-2-raw-v1 ...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > 100]
        print(f"  [OK] 下载成功，可用段落 {len(texts)}")
        return texts
    except Exception as e:
        print(f"  [跳过] 联网获取失败: {str(e)[:100]}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=int, default=64, help="校准集条数")
    ap.add_argument("--test", type=int, default=32, help="评测集条数")
    ap.add_argument("--force", action="store_true", help="已有 parquet 也重建")
    ap.add_argument("--download", action="store_true",
                    help="允许联网拉取 wikitext（默认不联网，避免卡住）")
    args = ap.parse_args()

    config.ensure_dirs()
    calib_pq = os.path.join(config.CALIB_DIR, "validation.parquet")
    test_pq = os.path.join(config.TEST_DIR, "test.parquet")

    print("=" * 72)
    print("数据准备")
    print("=" * 72)

    if _have_parquet(config.CALIB_DIR) and _have_parquet(config.TEST_DIR) and not args.force:
        print(f"[OK] 已存在真实数据，直接使用：")
        print(f"     校准: {config.CALIB_DIR}")
        print(f"     评测: {config.TEST_DIR}")
        return 0

    corpus_path = os.path.join(config.DATA_DIR, "raw", "corpus.txt")
    texts = None
    source = ""

    if os.path.isfile(corpus_path):
        print(f"[1] 使用本地语料 {corpus_path}")
        with open(corpus_path, encoding="utf-8") as f:
            raw = f.read()
        calib, test = _split_docs(raw, args.calib, args.test)
        texts = (calib, test)
        source = "local corpus.txt"
    elif args.download:
        print("[1] 未找到 data/raw/corpus.txt，尝试联网获取")
        got = try_download()
        if got:
            need = args.calib + args.test
            got = (got * ((need // len(got)) + 1))[:need]
            texts = (got[:args.calib], got[args.calib:need])
            source = "wikitext-2-raw-v1"
        else:
            print("[2] 使用内置兜底语料（可跑通，建议替换为真实数据）")
            calib, test = _split_docs(BUILTIN_FALLBACK, args.calib, args.test)
            texts = (calib, test)
            source = "builtin fallback"
    else:
        print("[1] 未找到 data/raw/corpus.txt（且未指定 --download）")
        print("[2] 使用内置兜底语料（可跑通，建议替换为真实数据）")
        calib, test = _split_docs(BUILTIN_FALLBACK, args.calib, args.test)
        texts = (calib, test)
        source = "builtin fallback"

    calib, test = texts
    n1 = _write_parquet(calib, calib_pq)
    n2 = _write_parquet(test, test_pq)
    print(f"\n[OK] 已写出（来源: {source}）")
    print(f"     校准 {n1} 条 -> {calib_pq}")
    print(f"     评测 {n2} 条 -> {test_pq}")
    if source != "local corpus.txt" and not _have_parquet(config.CALIB_DIR.replace("calib", "raw")):
        print("\n[提示] 想让量化精度更好：把真实语料（列名 text 的 parquet）放到")
        print(f"       {config.CALIB_DIR} 和 {config.TEST_DIR}，再跑 --force 重建。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
