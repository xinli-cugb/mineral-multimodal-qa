#!/usr/bin/env python3
"""
重排评测

读取 eval_retrieval.py 产出的候选列表，对每条 query 的各通道 Top-N 候选
用微调后的 Qwen3-Reranker-0.6B（yes/no 生成式打分）重排到 top-k，
对比「原通道」与「通道 + 重排」的指标。

默认只重排 dense 与 hybrid；如需 bm25 也重排，传 --methods dense,bm25,hybrid。
全量 4632 条 × 2 通道 × 10 候选约 9 万次打分，建议先用 --max-queries 200 冒烟。

用法：
    python scripts/eval_rerank.py --split test --candidates \
        /root/autodl-tmp/logs/eval/candidates_test.jsonl

产物：
    /root/autodl-tmp/logs/eval/rerank_{split}_{model}.json
    /root/autodl-tmp/logs/eval/rerank_{split}_{model}.jsonl（逐条缓存，可断点续跑）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval_common import (  # noqa: E402
    DEFAULT_INDEX_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_RERANKER,
    doc_id_map,
    evaluate_records,
    metrics_table,
    read_jsonl,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="重排评测")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--candidates", type=Path, default=None,
                        help="默认 {out-dir}/candidates_{split}.jsonl")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR,
                        help="用于按 id 取回文档文本")
    parser.add_argument("--reranker-path", type=Path, default=DEFAULT_RERANKER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", default="dense,hybrid",
                        help="对哪些通道做重排，逗号分隔")
    parser.add_argument("--depth", type=int, default=10, help="参与重排的候选深度")
    parser.add_argument("--topk", type=int, default=10, help="重排后保留条数")
    parser.add_argument("--max-queries", type=int, default=0, help="0 表示全量")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resume", action="store_true", help="从缓存续跑")
    return parser.parse_args()


def main():
    args = parse_args()
    candidates_path = args.candidates or (args.out_dir / f"candidates_{args.split}.jsonl")
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"候选文件不存在: {candidates_path}\n"
            f"请先运行 scripts/eval_retrieval.py 生成各通道候选。"
        )

    rows = read_jsonl(candidates_path)
    if args.max_queries > 0:
        rows = rows[:args.max_queries]
    if not rows:
        raise ValueError("候选文件为空")

    available = [m for m in args.methods.split(",") if m.strip()]
    available = [m for m in available if m in rows[0]]
    if not available:
        raise ValueError(f"候选文件中没有可用通道，现有字段: {list(rows[0].keys())}")
    print(f"📊 split={args.split}, query 数={len(rows)}, 重排通道={available}, "
          f"候选深度={args.depth}, topk={args.topk}")

    docs = doc_id_map(args.index_dir)

    from retrieval.reranker import load_reranker

    print(f"🎯 加载 Reranker: {args.reranker_path}")
    reranker = load_reranker(
        model_type="qwen3",
        model_path=str(args.reranker_path),
        device=args.device,
        max_length=512,
    )

    model_tag = args.reranker_path.name
    cache_path = args.out_dir / f"rerank_{args.split}_{model_tag}.jsonl"
    cached = {}
    if args.resume and cache_path.exists():
        for item in read_jsonl(cache_path):
            cached[(item["method"], item["query"])] = item["ranked_ids"]
        print(f"♻️ 已加载重排缓存 {len(cached)} 条: {cache_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # 非续跑模式覆盖写，避免旧结果重复追加
    cache_fp = open(cache_path, "a" if args.resume else "w", encoding="utf-8")

    records = {method: [] for method in available}
    t0 = time.time()
    total = len(rows) * len(available)
    done = 0

    for method in available:
        for item in rows:
            query = item["query"]
            gold_id = int(item["gold_id"])
            query_type = item.get("query_type", "unknown")

            # 原通道排序（baseline）
            baseline_ids = [int(x) for x in item[method][:args.depth]]
            records[method].append({
                "query": query,
                "gold_id": gold_id,
                "query_type": query_type,
                "ranked_ids": baseline_ids,
            })

            # 重排
            key = (method, query)
            if key in cached:
                reranked_ids = cached[key]
            else:
                cand_list = [
                    {"id": int(doc_id), "text": docs.get(int(doc_id), {}).get("text", "")}
                    for doc_id in baseline_ids
                ]
                cand_list = [c for c in cand_list if c["text"]]
                results = reranker.rerank(
                    query=query,
                    candidates=cand_list,
                    top_k=args.topk,
                    batch_size=args.batch_size,
                )
                reranked_ids = [int(r.id) for r in results]
                cache_fp.write(json.dumps({
                    "method": method,
                    "query": query,
                    "ranked_ids": reranked_ids,
                }, ensure_ascii=False) + "\n")
                cache_fp.flush()

            records[f"{method}+rerank"].append({
                "query": query,
                "gold_id": gold_id,
                "query_type": query_type,
                "ranked_ids": reranked_ids,
            })

            done += 1
            if done % 100 == 0 or done == total:
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0
                print(f"\r   重排进度 {done}/{total} ({speed:.2f} 条/s)", end="", flush=True)

    cache_fp.close()
    print(f"\n⏱️ 重排总耗时 {time.time() - t0:.1f}s")

    payload = {
        "meta": {
            "split": args.split,
            "num_queries": len(rows),
            "candidates_file": str(candidates_path),
            "reranker_path": str(args.reranker_path),
            "methods": available,
            "depth": args.depth,
            "topk": args.topk,
        },
        "results": {},
    }
    for name, recs in records.items():
        payload["results"][name] = evaluate_records(recs)

    print()
    for name, result in payload["results"].items():
        print(metrics_table(result, name))
        print()

    save_json(args.out_dir / f"rerank_{args.split}_{model_tag}.json", payload)


if __name__ == "__main__":
    main()
