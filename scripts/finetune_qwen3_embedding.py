#!/usr/bin/env python3
"""
Qwen3-Embedding 微调脚本（矿物检索域）

与旧版的关键差异:
  1. last-token pooling（Qwen3-Embedding 官方做法），不再用 [CLS]；
  2. query 侧加官方 Instruct 前缀，doc 侧不加；
  3. 动态 padding（query 64 / doc 256），不再 max_length 512 硬填充；
  4. InfoNCE 同时使用显式难负样本和 batch 内负样本；
  5. 只保存最终/最优模型（不保存 optimizer、不存中间 checkpoint）；
  6. 显式 CUDA 断言，避免静默回退 CPU；
  7. 支持 fp16 + 梯度累积 + 可选梯度检查点，防止 OOM。

数据格式: training_data/embedding/{train,val,test}.json
  [{"query": "...", "positive": "...", "negatives": ["...", ...]}, ...]
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
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Qwen3-Embedding 官方 Instruct 格式
DEFAULT_INSTRUCT = "给定一个矿物相关的查询，检索能够回答该查询的矿物资料。"
QUERY_TEMPLATE = "Instruct: {instruct}\nQuery: {query}"


def pad_to_multiple(tokens: List[List[int]], pad_id: int, multiple: int = 8):
    """把 batch 内 token 序列右侧补齐到 max_len（并对齐到 multiple）"""
    max_len = max(len(t) for t in tokens)
    max_len = int(math.ceil(max_len / multiple) * multiple)
    input_ids = torch.full((len(tokens), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(tokens), max_len), dtype=torch.long)
    for i, tok in enumerate(tokens):
        input_ids[i, : len(tok)] = torch.tensor(tok, dtype=torch.long)
        attention_mask[i, : len(tok)] = 1
    return input_ids, attention_mask


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """last-token pooling：取每条序列最后一个非 padding token 的隐状态"""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_state[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), lengths]


class EmbeddingDataset(Dataset):
    """{query, positive, negatives[]} 数据集"""

    def __init__(self, data_path: str, tokenizer, max_query_len: int = 64,
                 max_doc_len: int = 256, instruct: str = DEFAULT_INSTRUCT,
                 num_negatives: int = 7):
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError(f"{data_path} 应为 JSON 数组")

        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.instruct = instruct
        self.num_negatives = num_negatives

        self.samples = []
        for item in raw:
            negs = list(item.get("negatives") or [])[:num_negatives]
            if not item.get("query") or not item.get("positive"):
                continue
            self.samples.append(
                {
                    "query": QUERY_TEMPLATE.format(instruct=instruct, query=item["query"].strip()),
                    "docs": [item["positive"].strip()] + [n.strip() for n in negs],
                }
            )
        logger.info("加载 %s: %d 条样本", data_path, len(self.samples))

    def __len__(self):
        return len(self.samples)

    def _encode(self, text: str, max_len: int) -> List[int]:
        return self.tokenizer(
            text, truncation=True, max_length=max_len, add_special_tokens=True
        )["input_ids"]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "query_ids": self._encode(sample["query"], self.max_query_len),
            "doc_ids": [self._encode(d, self.max_doc_len) for d in sample["docs"]],
        }


def collate_batch(batch: List[Dict], pad_id: int):
    """动态 padding；负样本数量不足时用全 mask 占位，不计入 loss"""
    q_ids, q_mask = pad_to_multiple([b["query_ids"] for b in batch], pad_id)
    num_docs = max(len(b["doc_ids"]) for b in batch)
    flat_docs, doc_valid = [], []
    for b in batch:
        docs = b["doc_ids"]
        flat_docs.extend(docs)
        doc_valid.extend([True] * len(docs))
        for _ in range(num_docs - len(docs)):
            flat_docs.append([])
            doc_valid.append(False)
    d_ids, d_mask = pad_to_multiple(flat_docs, pad_id)
    # 空文档整行 mask 掉，避免污染 pooling
    for i, ok in enumerate(doc_valid):
        if not ok:
            d_mask[i].zero_()
    return {
        "query_input_ids": q_ids,
        "query_attention_mask": q_mask,
        "doc_input_ids": d_ids.view(len(batch), num_docs, -1),
        "doc_attention_mask": d_mask.view(len(batch), num_docs, -1),
    }


class EmbeddingModel(torch.nn.Module):
    def __init__(self, model_path: str, dtype=torch.bfloat16):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype
        )

    def encode(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        emb = last_token_pool(out.last_hidden_state, attention_mask)
        return F.normalize(emb.float(), p=2, dim=-1)


def contrastive_loss(query_emb: torch.Tensor, doc_emb: torch.Tensor,
                     num_docs_per_query: int, doc_valid: torch.Tensor,
                     temperature: float) -> torch.Tensor:
    """
    query_emb: (B, D)      doc_emb: (B*K, D)（每个 query 的 1 正 + N 负按顺序排布）
    使用 batch 内负样本 + 显式难负样本做 InfoNCE。
    """
    batch_size = query_emb.size(0)
    sim = query_emb @ doc_emb.t() / temperature  # (B, B*K)
    sim = sim.float()

    # 屏蔽无效候选
    valid = doc_valid.view(-1)  # (B*K,)
    sim = sim.masked_fill(~valid.unsqueeze(0), torch.finfo(sim.dtype).min)

    targets = torch.arange(batch_size, device=sim.device) * num_docs_per_query
    return F.cross_entropy(sim, targets)


@torch.no_grad()
def evaluate(model: EmbeddingModel, loader: DataLoader, device, temperature: float) -> Dict[str, float]:
    model.eval()
    total_loss, n = 0.0, 0
    recall1 = recall3 = mrr = 0.0
    for batch in loader:
        q_ids = batch["query_input_ids"].to(device, non_blocking=True)
        q_mask = batch["query_attention_mask"].to(device, non_blocking=True)
        d_ids = batch["doc_input_ids"].to(device, non_blocking=True)
        d_mask = batch["doc_attention_mask"].to(device, non_blocking=True)

        bsz, k, seq = d_ids.shape
        q_emb = model.encode(q_ids, q_mask)
        d_emb = model.encode(d_ids.view(bsz * k, seq), d_mask.view(bsz * k, seq))
        d_emb = d_emb.view(bsz, k, -1)

        sim = torch.einsum("bd,bkd->bk", q_emb, d_emb).float() / temperature
        labels = torch.zeros(bsz, dtype=torch.long, device=sim.device)
        loss = F.cross_entropy(sim, labels)
        total_loss += loss.item() * bsz
        n += bsz

        # 排序指标：正样本排在第 1 / 前 3 / MRR
        ranks = (sim > sim[:, :1]).sum(dim=1) + 1  # 正样本名次
        recall1 += (ranks <= 1).float().sum().item()
        recall3 += (ranks <= 3).float().sum().item()
        mrr += (1.0 / ranks.float()).sum().item()

    model.train()
    return {
        "eval_loss": total_loss / max(n, 1),
        "recall@1": recall1 / max(n, 1),
        "recall@3": recall3 / max(n, 1),
        "mrr": mrr / max(n, 1),
    }


def save_model(model: EmbeddingModel, tokenizer, output_dir: str, metadata: Dict):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    with open(out / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("模型已保存: %s", out)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-Embedding 矿物检索微调")
    parser.add_argument("--model_path", default="/root/V2_0_Mineral_QA/model/Qwen3-Embedding-0.6B")
    parser.add_argument("--train_data", default="/root/V2_0_Mineral_QA/training_data/embedding/train.json")
    parser.add_argument("--val_data", default="/root/V2_0_Mineral_QA/training_data/embedding/val.json")
    parser.add_argument("--test_data", default="/root/V2_0_Mineral_QA/training_data/embedding/test.json")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/outputs/embedding")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2, help="单卡 micro batch")
    parser.add_argument("--grad_accum", type=int, default=8, help="梯度累积（有效 batch=8*2=16）")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_query_len", type=int, default=64)
    parser.add_argument("--max_doc_len", type=int, default=256)
    parser.add_argument("--num_negatives", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=0, help=">0 时仅跑少量 step 用于冒烟")
    parser.add_argument("--eval_samples", type=int, default=0, help=">0 时只评估前 N 条")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- 环境与 CUDA 硬校验 ----
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

    train_ds = EmbeddingDataset(args.train_data, tokenizer, args.max_query_len,
                               args.max_doc_len, num_negatives=args.num_negatives)
    val_ds = EmbeddingDataset(args.val_data, tokenizer, args.max_query_len,
                              args.max_doc_len, num_negatives=args.num_negatives)
    if args.eval_samples > 0:
        val_ds.samples = val_ds.samples[: args.eval_samples]
        logger.info("冒烟模式：验证集截断为 %d 条", len(val_ds.samples))

    collate = lambda b: collate_batch(b, pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2, pin_memory=True,
                              drop_last=True, persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                            collate_fn=collate, num_workers=2, pin_memory=True)

    model = EmbeddingModel(args.model_path, dtype=torch.bfloat16).to(device)
    if args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()
        model.backbone.enable_input_require_grads()
        logger.info("已启用梯度检查点")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    if args.max_steps > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max(total_steps, 1))
    logger.info("样本 %d | micro_bs %d x accum %d | 优化步/epoch %d | 总步 %d",
                len(train_ds), args.batch_size, args.grad_accum, steps_per_epoch, total_steps)

    best_mrr, best_metrics = -1.0, {}
    global_step = 0
    t_start = time.time()
    model.train()

    for epoch in range(args.num_epochs):
        logger.info("=== Epoch %d/%d 开始 ===", epoch + 1, args.num_epochs)
        epoch_loss, epoch_tokens, micro_in_group = 0.0, 0, 0
        t_epoch = time.time()

        for micro_step, batch in enumerate(train_loader):
            q_ids = batch["query_input_ids"].to(device, non_blocking=True)
            q_mask = batch["query_attention_mask"].to(device, non_blocking=True)
            d_ids = batch["doc_input_ids"].to(device, non_blocking=True)
            d_mask = batch["doc_attention_mask"].to(device, non_blocking=True)

            bsz, k, seq = d_ids.shape
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                q_emb = model.encode(q_ids, q_mask)
                d_emb = model.encode(d_ids.view(bsz * k, seq), d_mask.view(bsz * k, seq))
                loss = contrastive_loss(q_emb, d_emb, k, d_mask.view(-1).any(dim=-1),
                                        args.temperature)
                loss_scaled = loss / args.grad_accum

            loss_scaled.backward()
            epoch_loss += loss.item()
            epoch_tokens += int(q_mask.sum()) + int(d_mask.sum())
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
                    tok_per_s = epoch_tokens / max(elapsed, 1e-6)
                    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                    logger.info(
                        "epoch %d | step %d/%d | loss %.4f | lr %.2e | %.0f tok/s | peak %.1f GB",
                        epoch + 1, global_step, total_steps, avg_loss,
                        scheduler.get_last_lr()[0], tok_per_s, peak,
                    )
                    epoch_loss, epoch_tokens, t_epoch = 0.0, 0, time.time()

                if args.max_steps > 0 and global_step >= args.max_steps:
                    logger.info("达到 max_steps=%d，冒烟结束", args.max_steps)
                    metrics = evaluate(model, val_loader, device, args.temperature)
                    logger.info("冒烟验证: %s", metrics)
                    save_model(model, tokenizer, args.output_dir,
                               {"mode": "smoke", "steps": global_step, "metrics": metrics})
                    return

        # 处理最后一个不完整累积组
        if micro_in_group > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        metrics = evaluate(model, val_loader, device, args.temperature)
        logger.info("Epoch %d 验证: %s", epoch + 1,
                    {k: round(v, 4) for k, v in metrics.items()})
        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]
            best_metrics = metrics
            save_model(model, tokenizer, args.output_dir,
                       {"mode": "best", "epoch": epoch + 1, "step": global_step,
                        "metrics": metrics, "args": vars(args)})
        torch.cuda.empty_cache()

    logger.info("训练完成，用时 %.1f 分钟 | best MRR %.4f | 验证指标 %s",
                (time.time() - t_start) / 60, best_mrr,
                {k: round(v, 4) for k, v in best_metrics.items()})

    # 测试集评估（用保存的最优模型权重）
    if args.test_data and Path(args.test_data).exists():
        test_ds = EmbeddingDataset(args.test_data, tokenizer, args.max_query_len,
                                   args.max_doc_len, num_negatives=args.num_negatives)
        if args.eval_samples > 0:
            test_ds.samples = test_ds.samples[: args.eval_samples]
        test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                                 collate_fn=collate, num_workers=2, pin_memory=True)
        test_metrics = evaluate(model, test_loader, device, args.temperature)
        logger.info("测试集最终指标(当前权重): %s",
                    {k: round(v, 4) for k, v in test_metrics.items()})
        meta_path = Path(args.output_dir) / "training_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["test_metrics"] = test_metrics
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
