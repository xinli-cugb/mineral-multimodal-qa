#!/usr/bin/env python3
"""
全库检索评测

在 6408 篇全库上评测三条检索通道：
    dense   —— 微调后的 Qwen3-Embedding，L2 距离升序
    bm25    —— jieba + Okapi BM25
    hybrid  —— Dense top-10 + BM25 top-20 → RRF(k=60) top-10

指标：Recall@5、Recall@10、NDCG@10、MRR@10；总体 + 按 query_type 分组。

用法：
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    python scripts/eval_retrieval.py --split test --methods dense,bm25,hybrid

产物：
    /root/autodl-tmp/logs/eval/retrieval_{split}.json     指标汇总
    /root/autodl-tmp/logs/eval/candidates_{split}.jsonl   各通道候选（供 eval_rerank.py 复用）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval_common import (  # noqa: E402
    DEFAULT_EMBEDDER,
    DEFAULT_INDEX_DIR,
    DEFAULT_OUT_DIR,
    CorpusIndex,
    dump_jsonl,
    evaluate_records,
    load_split_queries,
    metrics_table,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="全库检索评测")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--embedder-path", type=Path, default=DEFAULT_EMBEDDER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", default="dense,bm25,hybrid",
                        help="逗号分隔，可选 dense/bm25/hybrid")
    parser.add_argument("--max-queries", type=int, default=0, help="0 表示全量")
    parser.add_argument("--depth", type=int, default=10, help="各通道返回深度")
    parser.add_argument("--dense-depth", type=int, default=10,
                        help="hybrid 中 Dense 候选深度（线上为 10）")
    parser.add_argument("--bm25-depth", type=int, default=20,
                        help="hybrid 中 BM25 候选深度（线上为 20）")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF 平滑参数")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force-reencode", action="store_true",
                        help="忽略查询向量缓存，重新编码")
    return parser.parse_args()


def main():
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = set(methods) - {"dense", "bm25", "hybrid"}
    if unknown:
        raise ValueError(f"未知检索通道: {unknown}")

    queries = load_split_queries(args.split)
    if args.max_queries > 0:
        queries = queries[:args.max_queries]
    n = len(queries)
    print(f"📊 split={args.split}, 评测 query 数={n}, 全库文档数 "
          f"(由索引加载)，通道={methods}")

    need_dense = any(m in ("dense", "hybrid") for m in methods)
    need_bm25 = any(m in ("bm25", "hybrid") for m in methods)

    index = CorpusIndex(
        index_dir=args.index_dir,
        embedder_path=args.embedder_path,
        device=args.device,
        need_embedder=need_dense,
        need_bm25=need_bm25,
    )
    print(f"🗄️ 索引就绪: {index.num_docs} 篇 (嵌入维度 {index.index.d})")

    dense_ids = dense_dists = None
    if need_dense:
        cache_file = (
            args.out_dir / "cache"
            / f"query_emb_{args.split}_{args.embedder_path.name}.npz"
        )
        embeddings = index.encode_queries(
            [q["query"] for q in queries],
            batch_size=args.batch_size,
            cache_file=cache_file,
            force=args.force_reencode,
        )
        dense_depth = max(args.depth, args.dense_depth)
        print(f"🔢 稠密检索: 深度 {dense_depth}")
        t0 = time.time()
        dense_ids, dense_dists = index.dense_topk(embeddings, top_k=dense_depth)
        print(f"   稠密检索耗时 {time.time() - t0:.1f}s")

    # BM25 结果在 bm25 / hybrid 中复用，只算一次
    bm25_max_depth = max(args.depth, args.bm25_depth)
    sparse_raw = None
    if need_bm25:
        print(f"🧮 BM25 检索: 深度 {bm25_max_depth}")
        t0 = time.time()
        sparse_raw = []
        for i, item in enumerate(queries, start=1):
            sparse_raw.append(index.bm25_search_raw(item["query"], top_k=bm25_max_depth))
            if i % 500 == 0 or i == n:
                print(f"\r   BM25 进度 {i}/{n}", end="", flush=True)
        print(f"\n   BM25 检索耗时 {time.time() - t0:.1f}s")

    payload = {
        "meta": {
            "split": args.split,
            "num_queries": n,
            "num_docs": index.num_docs,
            "index_dir": str(args.index_dir),
            "embedder_path": str(args.embedder_path),
            "methods": methods,
            "depth": args.depth,
            "dense_depth_for_hybrid": args.dense_depth,
            "bm25_depth_for_hybrid": args.bm25_depth,
            "rrf_k": args.rrf_k,
        },
        "results": {},
    }
    candidate_rows = []

    for method in methods:
        t0 = time.time()
        records = []
        for i, item in enumerate(queries):
            if method == "dense":
                ranked = dense_ids[i][:args.depth]
            elif method == "bm25":
                ranked = [int(r["id"]) for r in sparse_raw[i][:args.depth]]
            else:
                ranked = index.hybrid_topk(
                    item["query"],
                    dense_ids[i][:args.dense_depth],
                    dense_dists[i][:args.dense_depth],
                    top_k=args.depth,
                    rrf_k=args.rrf_k,
                    sparse_results=sparse_raw[i],
                )
            records.append({
                "query": item["query"],
                "gold_id": item["gold_id"],
                "query_type": item["query_type"],
                "ranked_ids": ranked,
            })
        result = evaluate_records(records)
        payload["results"][method] = result
        print()
        print(metrics_table(result, f"{method}（{args.split}，全库 {index.num_docs} 篇）"))
        print(f"⏱️ {method} 排序耗时 {time.time() - t0:.1f}s")

    # 汇总候选，供重排评测复用
    for i, item in enumerate(queries):
        row = {
            "query": item["query"],
            "gold_id": item["gold_id"],
            "query_type": item["query_type"],
        }
        if "dense" in methods:
            row["dense"] = dense_ids[i][:args.depth]
        if "bm25" in methods:
            row["bm25"] = [int(r["id"]) for r in sparse_raw[i][:args.depth]]
        if "hybrid" in methods:
            row["hybrid"] = index.hybrid_topk(
                item["query"],
                dense_ids[i][:args.dense_depth],
                dense_dists[i][:args.dense_depth],
                top_k=args.depth,
                rrf_k=args.rrf_k,
                sparse_results=sparse_raw[i],
            )
        candidate_rows.append(row)

    suffix = f"_n{n}" if args.max_queries > 0 else ""
    save_json(args.out_dir / f"retrieval_{args.split}{suffix}.json", payload)
    dump_jsonl(args.out_dir / f"candidates_{args.split}{suffix}.jsonl", candidate_rows)


if __name__ == "__main__":
    main()
