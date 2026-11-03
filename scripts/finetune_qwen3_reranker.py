#!/usr/bin/env python3
"""
Qwen3-Reranker 微调脚本（官方 yes/no 生成式打分）



数据格式: training_data/reranker/{train,val,test}_pairs.jsonl
  {"query": "...", "doc": "...", "label": 1|0, "query_type": "...", "doc_id": 0}
"""
import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DEFAULT_INSTRUCT = "给定一个矿物相关的查询，判断该文档是否能够回答查询。"
SYSTEM_PROMPT = (
    'Judge whether the Document meets the requirements based on the Query and '
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)


def pad_to_multiple(tokens: List[List[int]], pad_id: int, multiple: int = 8):
    max_len = max(len(t) for t in tokens)
    max_len = int(math.ceil(max_len / multiple) * multiple)
    input_ids = torch.full((len(tokens), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(tokens), max_len), dtype=torch.long)
    for i, tok in enumerate(tokens):
        input_ids[i, : len(tok)] = torch.tensor(tok, dtype=torch.long)
        attention_mask[i, : len(tok)] = 1
    return input_ids, attention_mask


class RerankerPairDataset(Dataset):
    """读取 {query, doc, label} 的 jsonl"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 320,
                 instruct: str = DEFAULT_INSTRUCT, limit: int = 0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not item.get("query") or not item.get("doc"):
                    continue
                self.samples.append(
                    (item["query"].strip(), item["doc"].strip(), int(item["label"]))
                )
                if limit and len(self.samples) >= limit:
                    break
        logger.info("加载 %s: %d 条样本", data_path, len(self.samples))

        self.prefix_ids = tokenizer(
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n<Instruct>: {instruct}\n<Query>: ",
            add_special_tokens=False,
        )["input_ids"]
        self.suffix_ids = tokenizer(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )["input_ids"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query, doc, label = self.samples[idx]
        q_ids = self.tokenizer(query, add_special_tokens=False)["input_ids"]
        d_ids = self.tokenizer(doc, add_special_tokens=False)["input_ids"]

        budget = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        body = q_ids + self.tokenizer("\n<Document>: ", add_special_tokens=False)["input_ids"] + d_ids
        body = body[: max(budget, 16)]
        ids = (self.prefix_ids + body + self.suffix_ids)[: self.max_length]
        return {"input_ids": ids, "label": label}


def collate_batch(batch: List[Dict], pad_id: int):
    input_ids, attention_mask = pad_to_multiple([b["input_ids"] for b in batch], pad_id)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class Qwen3Reranker(torch.nn.Module):
    """只投影 yes/no 两个 token 的 logits，等价于官方判定头的二分类形式"""

    def __init__(self, model_path: str, tokenizer, dtype=torch.bfloat16):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype
        )
        resolve_yes_no_ids(tokenizer, self.model)
        self.yes_id = self.model.config.yes_token_id
        self.no_id = self.model.config.no_token_id

    def forward(self, input_ids, attention_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lengths = attention_mask.sum(dim=1) - 1
        last = hidden[
            torch.arange(hidden.size(0), device=hidden.device), lengths
        ].float()
        weight = self.model.lm_head.weight[[self.no_id, self.yes_id]].float()
        logits = last @ weight.t()  # (B, 2) -> [no, yes]
        scores = torch.softmax(logits, dim=-1)[:, 1]
        return logits, scores


def resolve_yes_no_ids(tokenizer, model) -> None:
    cfg = model.config
    yes_id = getattr(cfg, "yes_token_id", None)
    no_id = getattr(cfg, "no_token_id", None)
    if yes_id is None or no_id is None:
        yes_id = tokenizer.convert_tokens_to_ids("yes")
        no_id = tokenizer.convert_tokens_to_ids("no")
        cfg.yes_token_id, cfg.no_token_id = yes_id, no_id
    if yes_id in (None, tokenizer.unk_token_id) or no_id in (None, tokenizer.unk_token_id):
        raise RuntimeError(f"无法解析 yes/no token id: yes={yes_id}, no={no_id}")
    cfg.yes_token_id, cfg.no_token_id = int(yes_id), int(no_id)


@torch.no_grad()
def evaluate(model: Qwen3Reranker, loader: DataLoader, device) -> Dict[str, float]:
    model.eval()
    total_loss, n = 0.0, 0
    tp = fp = fn = tn = 0
    all_scores, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, scores = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

        pred = (scores > 0.5).long()
        tp += int(((pred == 1) & (labels == 1)).sum())
        fp += int(((pred == 1) & (labels == 0)).sum())
        fn += int(((pred == 0) & (labels == 1)).sum())
        tn += int(((pred == 0) & (labels == 0)).sum())
        all_scores.append(scores.float().cpu())
        all_labels.append(labels.cpu())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    # AUC
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) and len(neg):
        auc = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean().item()
        auc += 0.5 * (pos.unsqueeze(1) == neg.unsqueeze(0)).float().mean().item()
    else:
        auc = float("nan")

    model.train()
    return {"eval_loss": total_loss / max(n, 1), "accuracy": accuracy,
            "precision": precision, "recall": recall, "f1": f1, "auc": auc}


def save_model(model: Qwen3Reranker, tokenizer, output_dir: str, metadata: Dict):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    (out / "training_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("模型已保存: %s", out)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-Reranker 矿物检索重排微调")
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen3-Reranker-0.6B")
    parser.add_argument("--train_data", default="/root/V2_0_Mineral_QA/training_data/reranker/train_pairs.jsonl")
    parser.add_argument("--val_data", default="/root/V2_0_Mineral_QA/training_data/reranker/val_pairs.jsonl")
    parser.add_argument("--test_data", default="/root/V2_0_Mineral_QA/training_data/reranker/test_pairs.jsonl")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/outputs/reranker")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=320)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=0, help=">0 时仅跑少量 step 用于冒烟")
    parser.add_argument("--eval_samples", type=int, default=0, help=">0 时只评估前 N 对")
    parser.add_argument("--train_samples", type=int, default=0, help=">0 时只训练前 N 对（冒烟）")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，拒绝回退 CPU 训练")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("GPU: %s | %.1f GB", torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    train_ds = RerankerPairDataset(args.train_data, tokenizer, args.max_length,
                                   limit=args.train_samples)
    val_ds = RerankerPairDataset(args.val_data, tokenizer, args.max_length,
                                 limit=args.eval_samples)
    collate = lambda b: collate_batch(b, pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2, pin_memory=True,
                              drop_last=True, persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                            collate_fn=collate, num_workers=2, pin_memory=True)

    model = Qwen3Reranker(args.model_path, tokenizer, dtype=torch.bfloat16).to(device)
    logger.info("yes_token_id=%d, no_token_id=%d", model.yes_id, model.no_id)
    if args.gradient_checkpointing:
        model.model.gradient_checkpointing_enable()
        model.model.enable_input_require_grads()
        logger.info("已启用梯度检查点")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    if args.max_steps > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    total_steps = steps_per_epoch * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), max(total_steps, 1)
    )
    logger.info("样本 %d | micro_bs %d x accum %d | 优化步/epoch %d | 总步 %d",
                len(train_ds), args.batch_size, args.grad_accum, steps_per_epoch, total_steps)

    best_f1, best_metrics, global_step = -1.0, {}, 0
    t_start = time.time()
    model.train()

    for epoch in range(args.num_epochs):
        logger.info("=== Epoch %d/%d 开始 ===", epoch + 1, args.num_epochs)
        epoch_loss, epoch_tokens, micro_in_group = 0.0, 0, 0
        t_epoch = time.time()

        for micro_step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, _ = model(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)
                loss_scaled = loss / args.grad_accum

            loss_scaled.backward()
            epoch_loss += loss.item()
            epoch_tokens += int(attention_mask.sum())
            micro_in_group += 1

            if micro_in_group == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                micro_in_group = 0
                global_step += 1

                if global_step % args.log_every == 0:
                    avg_loss = epoch_loss / (args.log_every * args.grad_accum)
                    elapsed = time.time() - t_epoch
                    logger.info(
                        "epoch %d | step %d/%d | loss %.4f | lr %.2e | %.0f tok/s | peak %.1f GB",
                        epoch + 1, global_step, total_steps, avg_loss,
                        scheduler.get_last_lr()[0], epoch_tokens / max(elapsed, 1e-6),
                        torch.cuda.max_memory_allocated() / 1024 ** 3,
                    )
                    epoch_loss, epoch_tokens, t_epoch = 0.0, 0, time.time()

                if args.max_steps > 0 and global_step >= args.max_steps:
                    logger.info("达到 max_steps=%d，冒烟结束", args.max_steps)
                    metrics = evaluate(model, val_loader, device)
                    logger.info("冒烟验证: %s", {k: round(v, 4) for k, v in metrics.items()})
                    save_model(model, tokenizer, args.output_dir,
                               {"mode": "smoke", "steps": global_step, "metrics": metrics})
                    return

        if micro_in_group > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        metrics = evaluate(model, val_loader, device)
        logger.info("Epoch %d 验证: %s", epoch + 1,
                    {k: round(v, 4) for k, v in metrics.items()})
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = metrics
            save_model(model, tokenizer, args.output_dir,
                       {"mode": "best", "epoch": epoch + 1, "step": global_step,
                        "metrics": metrics, "args": vars(args)})
        torch.cuda.empty_cache()

    logger.info("训练完成，用时 %.1f 分钟 | best F1 %.4f | 验证指标 %s",
                (time.time() - t_start) / 60, best_f1,
                {k: round(v, 4) for k, v in best_metrics.items()})

    if args.test_data and Path(args.test_data).exists():
        test_ds = RerankerPairDataset(args.test_data, tokenizer, args.max_length,
                                      limit=args.eval_samples)
        test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                                 collate_fn=collate, num_workers=2, pin_memory=True)
        test_metrics = evaluate(model, test_loader, device)
        logger.info("测试集最终指标(当前权重): %s",
                    {k: round(v, 4) for k, v in test_metrics.items()})
        meta_path = Path(args.output_dir) / "training_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["test_metrics"] = test_metrics
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
