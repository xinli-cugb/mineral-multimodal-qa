"""
评测公共工具

提供：
- 数据划分读取（query / gold 文档 id / query_type）
- 全库索引加载（FAISS 稠密 + BM25 稀疏）
- 三种检索通道的排序结果（dense / bm25 / hybrid RRF）
- 指标计算（Recall@K、NDCG@K、MRR@K）与按 query_type 分组
- 结果落盘与 markdown 表格输出

注意：必须在导入 torch 前固定线程数，否则本机环境会触发 CUDA Error 304。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "training_data"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "faiss_mineral_6408"
DEFAULT_EMBEDDER = Path("/root/autodl-tmp/outputs/embedding")
DEFAULT_RERANKER = Path("/root/autodl-tmp/outputs/reranker")
DEFAULT_LLM = PROJECT_ROOT / "model/Qwen2.5-7B-Instruct"
DEFAULT_OUT_DIR = Path("/root/autodl-tmp/logs/eval")


# ==================== 数据读取 ====================

def load_split_queries(split: str = "test") -> List[Dict]:
    """
    读取 embedding/{split}.json，返回评测用 query 列表。

    每条记录包含：
        query: 问题文本
        gold_id: 正样本文档 id（int）
        query_type: 问题类型
        positive: 正样本文本
    若数据里缺少 positive_id，则回退为按正样本文本匹配语料 id。
    """
    path = DATA_DIR / "embedding" / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"划分文件不存在: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    items: List[Dict] = []
    missing_id = 0
    for row in raw:
        gold_id = row.get("positive_id")
        if gold_id is None:
            missing_id += 1
        items.append({
            "query": row["query"],
            "gold_id": int(gold_id) if gold_id is not None else None,
            "query_type": row.get("query_type", "unknown"),
            "positive": row.get("positive", ""),
        })

    if missing_id:
        corpus_text_to_id = {}
        for doc in load_corpus_docs():
            corpus_text_to_id.setdefault(doc.get("text", "").strip(), int(doc["id"]))
        for item in items:
            if item["gold_id"] is None:
                item["gold_id"] = corpus_text_to_id.get(item["positive"].strip())
        unresolved = sum(1 for x in items if x["gold_id"] is None)
        print(f"⚠️ {missing_id} 条缺少 positive_id，文本回退后仍未匹配: {unresolved}")

    unresolved = [x for x in items if x["gold_id"] is None]
    if unresolved:
        raise ValueError(f"{len(unresolved)} 条 query 无法确定 gold 文档 id")
    return items


def load_corpus_docs(index_dir: Path = DEFAULT_INDEX_DIR) -> List[Dict]:
    """读取 FAISS 元数据（与索引位置一一对应）"""
    import pickle

    data_file = Path(index_dir) / "data.pkl"
    if not data_file.exists():
        raise FileNotFoundError(f"FAISS 元数据不存在: {data_file}")
    with open(data_file, "rb") as f:
        return pickle.load(f)


def doc_id_map(index_dir: Path = DEFAULT_INDEX_DIR) -> Dict[int, Dict]:
    return {int(d["id"]): d for d in load_corpus_docs(index_dir)}


# ==================== 索引 ====================

class CorpusIndex:
    """
    全库检索索引：FAISS 稠密通道 + BM25 稀疏通道。

    embedder 与 BM25 按需加载，只跑 BM25 时不会占用显存。
    """

    def __init__(
        self,
        index_dir: Path = DEFAULT_INDEX_DIR,
        embedder_path: Optional[Path] = DEFAULT_EMBEDDER,
        device: str = "cuda",
        need_embedder: bool = True,
        need_bm25: bool = True,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        import faiss

        self.index_dir = Path(index_dir)
        self.docs = load_corpus_docs(self.index_dir)
        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))
        self.embedder = None
        self.bm25 = None

        if need_embedder:
            from retrieval.qwen3_embedder import load_embedder

            self.embedder = load_embedder(model_path=str(embedder_path), device=device)

        if need_bm25:
            from retrieval.bm25_search import BM25Index

            self.bm25 = BM25Index(self.docs, k1=bm25_k1, b=bm25_b)

    @property
    def num_docs(self) -> int:
        return len(self.docs)

    # ---------- 查询向量 ----------

    def encode_queries(
        self,
        queries: Sequence[str],
        batch_size: int = 32,
        cache_file: Optional[Path] = None,
        force: bool = False,
    ):
        """
        编码查询并缓存。缓存会校验 query 内容是否与缓存一致，不一致则重算。
        """
        import numpy as np

        if self.embedder is None:
            raise RuntimeError("未加载 embedder，无法编码查询")

        if cache_file and cache_file.exists() and not force:
            cached = np.load(cache_file, allow_pickle=True)
            if list(cached["queries"]) == list(queries):
                print(f"♻️ 复用查询向量缓存: {cache_file}")
                return cached["embeddings"]

        embeddings = self.embedder.encode(
            list(queries), batch_size=batch_size, show_progress=True
        )

        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_file,
                queries=np.array(list(queries), dtype=object),
                embeddings=embeddings,
            )
            print(f"💾 查询向量已缓存: {cache_file}")
        return embeddings

    # ---------- 三路检索 ----------

    def dense_topk(self, embeddings, top_k: int = 10) -> Tuple[List[List[int]], List[List[float]]]:
        """稠密检索：返回每个查询的 doc id 列表与 L2 距离"""
        import numpy as np

        embeddings = np.asarray(embeddings, dtype="float32")
        distances, positions = self.index.search(embeddings, min(top_k, self.index.ntotal))
        all_ids, all_dists = [], []
        for pos_row, dist_row in zip(positions, distances):
            ids, dists = [], []
            for pos, dist in zip(pos_row, dist_row):
                if pos < 0 or pos >= len(self.docs):
                    continue
                ids.append(int(self.docs[int(pos)]["id"]))
                dists.append(float(dist))
            all_ids.append(ids)
            all_dists.append(dists)
        return all_ids, all_dists

    def bm25_search_raw(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 原始结果（字段与稠密结果兼容，供 RRF 使用）"""
        if self.bm25 is None:
            raise RuntimeError("未加载 BM25 索引")
        return self.bm25.search(query, top_k=top_k)

    def bm25_topk(self, query: str, top_k: int = 10) -> List[int]:
        return [int(r["id"]) for r in self.bm25_search_raw(query, top_k=top_k)]

    def hybrid_topk(
        self,
        query: str,
        dense_ids: Sequence[int],
        dense_dists: Sequence[float],
        top_k: int = 10,
        bm25_depth: int = 20,
        rrf_k: int = 60,
        sparse_results: Optional[List[Dict]] = None,
    ) -> List[int]:
        """
        RRF 融合，与线上 pipeline 参数保持一致：
        Dense top-10 + BM25 top-20 → RRF top-k。
        sparse_results 可复用外部已算好的 BM25 结果，避免重复检索。
        """
        from retrieval.hybrid_search import rrf_fusion

        dense_results = [
            {"id": doc_id, "distance": dist}
            for doc_id, dist in zip(dense_ids, dense_dists)
        ]
        if sparse_results is None:
            sparse_results = self.bm25_search_raw(query, top_k=bm25_depth)
        fused = rrf_fusion(dense_results, sparse_results, top_k=top_k, k=rrf_k)
        return [int(r.id) for r in fused]


# ==================== 指标 ====================

def _rank_of(gold_id: int, ranked_ids: Sequence[int], cutoff: int) -> Optional[int]:
    """返回 gold 在 top-cutoff 中的排名（从 1 开始），未命中返回 None"""
    for rank, doc_id in enumerate(ranked_ids[:cutoff], start=1):
        if int(doc_id) == int(gold_id):
            return rank
    return None


def recall_at_k(gold_id: int, ranked_ids: Sequence[int], k: int) -> float:
    return 1.0 if _rank_of(gold_id, ranked_ids, k) else 0.0


def ndcg_at_k(gold_id: int, ranked_ids: Sequence[int], k: int) -> float:
    """单正样本 NDCG：IDCG = 1，DCG = 1 / log2(rank + 1)"""
    import math

    rank = _rank_of(gold_id, ranked_ids, k)
    return 1.0 / math.log2(rank + 1) if rank else 0.0


def mrr_at_k(gold_id: int, ranked_ids: Sequence[int], k: int) -> float:
    rank = _rank_of(gold_id, ranked_ids, k)
    return 1.0 / rank if rank else 0.0


def evaluate_records(
    records: List[Dict],
    k_values: Sequence[int] = (5, 10),
    ndcg_k: int = 10,
    mrr_k: int = 10,
) -> Dict:
    """
    对一组 {"gold_id", "query_type", "ranked_ids"} 记录计算指标。

    返回 {"num_queries", "overall": {...}, "by_type": {...}}
    """
    def _agg(rows: List[Dict]) -> Dict[str, float]:
        if not rows:
            return {}
        metrics = {}
        for k in k_values:
            metrics[f"Recall@{k}"] = sum(
                recall_at_k(r["gold_id"], r["ranked_ids"], k) for r in rows
            ) / len(rows)
        metrics[f"NDCG@{ndcg_k}"] = sum(
            ndcg_at_k(r["gold_id"], r["ranked_ids"], ndcg_k) for r in rows
        ) / len(rows)
        metrics[f"MRR@{mrr_k}"] = sum(
            mrr_at_k(r["gold_id"], r["ranked_ids"], mrr_k) for r in rows
        ) / len(rows)
        return {name: round(value, 4) for name, value in metrics.items()}

    by_type_rows: Dict[str, List[Dict]] = {}
    for row in records:
        by_type_rows.setdefault(row.get("query_type", "unknown"), []).append(row)

    return {
        "num_queries": len(records),
        "overall": _agg(records),
        "by_type": {
            qtype: {"num_queries": len(rows), **_agg(rows)}
            for qtype, rows in sorted(by_type_rows.items())
        },
    }


# ==================== 输出 ====================

def metrics_table(result: Dict, title: str = "") -> str:
    """把 evaluate_records 的结果渲染成 markdown 表"""
    header_metrics = list(result.get("overall", {}).keys())
    if not header_metrics:
        return f"### {title}\n\n（无数据）"

    lines = []
    if title:
        lines.append(f"### {title}")
        lines.append("")
    lines.append("| 分组 | 样本数 | " + " | ".join(header_metrics) + " |")
    lines.append("|---|---|" + "|".join(["---"] * len(header_metrics)) + "|")

    overall = result["overall"]
    lines.append(
        f"| 全部 | {result['num_queries']} | "
        + " | ".join(f"{overall[m]:.4f}" for m in header_metrics)
        + " |"
    )
    for qtype, row in result.get("by_type", {}).items():
        lines.append(
            f"| {qtype} | {row['num_queries']} | "
            + " | ".join(f"{row[m]:.4f}" for m in header_metrics)
            + " |"
        )
    return "\n".join(lines)


def save_json(path: Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 结果已保存: {path}")


def dump_jsonl(path: Path, rows: List[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"💾 明细已保存: {path}")


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_key(*parts) -> str:
    raw = "||".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
