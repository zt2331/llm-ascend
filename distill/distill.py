#!/usr/bin/env python
"""知识蒸馏（KD）—— 昇腾 NPU 版：用 base 当 teacher，找回剪枝损失的精度。

原理：teacher 的软 logits 携带「暗知识」；student 学 teacher 的分布（KL 散度）。
      温度 T 平滑分布，乘 T² 保持梯度量级。teacher 冻结，只训 student。

用法（项目根）:
    python distill/distill.py --student output_models/pruned/xxx --epochs 1 --max-steps 60
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import device as dev
from utils.model_utils import load_model


def build_batches(tok, calib_n, seq):
    from datasets import Dataset
    pq = os.path.join(config.CALIB_DIR, "validation.parquet")
    full = Dataset.from_parquet(pq)
    df = full.select(range(min(calib_n, len(full))))
    out = []
    for t in df:
        ids = tok(t["text"], truncation=True, max_length=seq)["input_ids"]
        if len(ids) >= 8:
            out.append(torch.tensor(ids, dtype=torch.long))
    return out


def kd_loss(student, teacher, batch, temp=2.0, prefer_device="auto"):
    target = dev.get_device(prefer_device)
    b = batch.unsqueeze(0).to(target)
    with torch.no_grad():
        t_logits = teacher(input_ids=b, use_cache=False).logits
    s_logits = student(input_ids=b, use_cache=False).logits
    t_log = F.log_softmax(t_logits / temp, dim=-1)
    s_log = F.log_softmax(s_logits / temp, dim=-1)
    kl = F.kl_div(s_log, t_log, reduction="batchmean", log_target=True) * (temp ** 2)
    return kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=None, help="剪枝后的 student 模型目录")
    ap.add_argument("--teacher", default=None, help="teacher（默认=base 模型）")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config.ensure_dirs()
    base = config.require_model()

    # 自动找最新的剪枝产物
    student_path = args.student
    if not student_path:
        import glob
        cands = sorted(glob.glob(os.path.join(config.PRUNE_DIR, "*")), key=os.path.getmtime)
        if not cands:
            raise SystemExit("未找到剪枝产物，请先运行 prune/prune.py 或指定 --student")
        student_path = cands[-1]
    teacher_path = args.teacher or base

    print(f"[蒸馏] teacher = {teacher_path}")
    print(f"[蒸馏] student = {student_path}")
    print(f"[蒸馏] 设备 = {dev.default_device(args.device)}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(student_path, trust_remote_code=True)

    # teacher 用 CPU 省显存 + student 在 NPU 上训练（27B 同卡放不下时更稳）
    teacher = load_model(teacher_path, prefer_device="cpu", eval_mode=True)
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = load_model(student_path, prefer_device=args.device, eval_mode=False)
    student.train()

    batches = build_batches(tok, args.calib, args.seq)
    if not batches:
        raise SystemExit("校准数据为空，请先运行 scripts/04_prepare_data.py")

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)
    step = 0
    losses = []
    for _ in range(args.epochs):
        for b in batches:
            if step >= args.max_steps:
                break
            opt.zero_grad()
            loss = kd_loss(student, teacher, b, prefer_device=args.device)
            loss.backward()
            opt.step()
            step += 1
            losses.append(float(loss.item()))
            if step % 10 == 0:
                print(f"  step {step:4d}  loss {loss.item():.4f}")
        if step >= args.max_steps:
            break

    out = args.out or os.path.join(config.DISTILL_DIR, "distilled-student")
    student.eval()
    student.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    import json
    with open(os.path.join(out, "distill_info.json"), "w", encoding="utf-8") as f:
        json.dump({"teacher": teacher_path, "student_init": student_path,
                   "steps": step, "lr": args.lr, "temperature": 2.0,
                   "first_loss": losses[0] if losses else None,
                   "last_loss": losses[-1] if losses else None}, f, indent=2, ensure_ascii=False)
    print(f"[蒸馏] 已保存 -> {out}  (loss {losses[0]:.4f} → {losses[-1]:.4f})" if losses
          else f"[蒸馏] 已保存 -> {out}")


if __name__ == "__main__":
    main()
