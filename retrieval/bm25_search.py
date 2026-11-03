"""
BM25 稀疏检索模块

用于混合检索中的稀疏通道：对语料做 jieba 分词 + BM25 打分。
文档 id 必须与稠密索引（FAISS/Milvus）保持一致，RRF 才能正确融合。
"""
import json
import logging
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import jieba
import numpy as np

logger = logging.getLogger(__name__)

# 只保留中英文与数字，过滤标点
TOKEN_RE = re.compile(r"[0-9a-zA-Z\u4e00-\u9fff]+")
# 极简停用词：只去掉在检索中几乎无区分度的虚词
STOPWORDS = {
    "的", "是", "了", "和", "与", "及", "在", "有", "为", "对", "把", "被",
    "什么", "多少", "哪些", "如何", "怎么", "怎样", "哪个", "哪里", "介绍",
    "一下", "请问", "吗", "呢", "啊", "吧", "一种", "属于", "可以", "这个",
}


def tokenize(text: str) -> List[str]:
    """中文 jieba + 英文小写分词"""
    tokens = []
    for tok in jieba.lcut(str(text).lower()):
        tok = tok.strip()
        if not TOKEN_RE.fullmatch(tok):
            continue
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


class BM25Index:
    """内存版 BM25（Okapi）索引，文档量级 1w 以内足够快"""

    def __init__(self, documents: List[Dict], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("BM25 语料为空")
        self.k1 = k1
        self.b = b

        self.documents: List[Dict] = []
        self.doc_ids: List = []
        self.doc_tokens: List[Counter] = []
        self.doc_len: List[int] = []

        df = Counter()
        for pos, doc in enumerate(documents):
            doc_id = doc.get("id", pos)
            text = doc.get("text") or doc.get("brief") or ""
            tokens = tokenize(text)
            tf = Counter(tokens)
            self.documents.append(dict(doc))
            self.doc_ids.append(doc_id)
            self.doc_tokens.append(tf)
            self.doc_len.append(len(tokens))
            for token in tf:
                df[token] += 1

        self.n_docs = len(self.documents)
        self.avgdl = float(np.mean(self.doc_len)) if self.doc_len else 0.0
        self.idf = {
            token: float(np.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5)))
            for token, freq in df.items()
        }

        # 倒排表: token -> [(doc_pos, tf)]
        self.postings: Dict[str, List] = {}
        for pos, tf in enumerate(self.doc_tokens):
            for token, count in tf.items():
                self.postings.setdefault(token, []).append((pos, count))

        logger.info(
            "BM25 索引构建完成: %d 篇, 词表 %d, 平均长度 %.1f",
            self.n_docs, len(self.idf), self.avgdl,
        )

    @classmethod
    def from_documents(cls, documents: List[Dict], **kwargs) -> "BM25Index":
        return cls(documents, **kwargs)

    @classmethod
    def from_faiss_data(cls, data_path: str, **kwargs) -> "BM25Index":
        """从 FAISS 的 data.pkl 构建，id 按位置对齐 FAISS 返回的 id"""
        with open(data_path, "rb") as f:
            docs = pickle.load(f)
        return cls(list(docs), **kwargs)

    @classmethod
    def from_jsonl(cls, path: str, text_field: str = "text", **kwargs) -> "BM25Index":
        docs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                docs.append({
                    "id": item.get("id", len(docs)),
                    "name": item.get("name", ""),
                    "english_name": item.get("english_name", ""),
                    "text": item.get(text_field) or item.get("brief", ""),
                    "source": item.get("source", ""),
                })
        return cls(docs, **kwargs)

    def _score_doc(self, pos: int, query_tf: Dict[str, int]) -> float:
        dl = self.doc_len[pos] or 1
        norm = self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
        doc_tf = self.doc_tokens[pos]
        score = 0.0
        for token, qtf in query_tf.items():
            tf = doc_tf.get(token)
            if not tf:
                continue
            idf = self.idf.get(token, 0.0)
            score += idf * (tf * (self.k1 + 1.0)) / (tf + norm) * min(qtf, 3)
        return score

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """返回按 BM25 分数降序的文档，字段与稠密检索结果兼容"""
        query_tf = Counter(tokenize(query))
        if not query_tf:
            return []

        # 只对命中查询词的文档打分
        candidate_pos = set()
        for token in query_tf:
            for pos, _ in self.postings.get(token, []):
                candidate_pos.add(pos)

        scored = [(self._score_doc(pos, query_tf), pos) for pos in candidate_pos]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, pos in scored[:top_k]:
            doc = self.documents[pos]
            results.append({
                "id": self.doc_ids[pos],
                "text": doc.get("text", ""),
                "name": doc.get("name", ""),
                "english_name": doc.get("english_name", ""),
                "source": doc.get("source", "bm25"),
                "distance": float(score),   # 兼容 rrf_fusion 的字段命名
                "score": float(score),
            })
        return results

    def __len__(self):
        return self.n_docs


def build_bm25_from_pipeline_sources(
    faiss_data: Optional[List[Dict]] = None,
    corpus_path: Optional[str] = None,
) -> Optional[BM25Index]:
    """
    优先用稠密索引的同一份元数据构建，保证 id 对齐；
    否则退回到 corpus.jsonl。
    """
    if faiss_data:
        return BM25Index.from_documents(list(faiss_data))
    if corpus_path and Path(corpus_path).exists():
        return BM25Index.from_jsonl(corpus_path)
    return None
