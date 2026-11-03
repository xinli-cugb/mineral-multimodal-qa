"""
双路混合检索模块
实现 BM25 稀疏检索 + 稠密向量检索的混合搜索
包含 RRF (Reciprocal Rank Fusion) 倒数秩融合算法
"""
from typing import List, Dict, Optional, Tuple, Union, Any
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 数据结构 ====================

@dataclass
class RetrievalResult:
    """
    检索结果数据类
    """
    id: Union[int, str]
    text: str
    score: float
    rank: int = 0
    source: str = ""
    url: str = ""
    title: str = ""
    mineral_type: str = ""
    dense_score: float = 0.0
    sparse_score: float = 0.0
    dense_rank: int = 0
    sparse_rank: int = 0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "rank": self.rank,
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "mineral_type": self.mineral_type,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank
        }


@dataclass
class VisualAnchor:
    """
    视觉锚点数据结构
    来自 Swin Transformer 的 Top-3 分类结果
    """
    mineral_name: str
    confidence: float
    rank: int
    
    def to_query_string(self) -> str:
        """转换为查询字符串"""
        if self.rank == 1:
            return self.mineral_name
        else:
            return f"{self.mineral_name} {self.mineral_name}"


# ==================== RRF 融合算法 ====================

def rrf_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    top_k: int = 10,
    k: int = 60
) -> List[RetrievalResult]:
    """
    RRF (Reciprocal Rank Fusion) 倒数秩融合算法
    
    对稠密向量检索和稀疏向量检索的结果进行融合排序
    
    公式: RRF(d) = Σ 1 / (k + rank(d))
    其中:
        - d: 文档
        - rank(d): 文档在某个排序列表中的排名 (从1开始)
        - k: 调整参数，通常取 60
    
    Args:
        dense_results: 稠密向量检索结果
        sparse_results: 稀疏向量检索结果 (BM25)
        top_k: 返回 Top-K 融合结果
        k: RRF 调整参数
    
    Returns:
        融合后的检索结果列表
    """
    if not dense_results and not sparse_results:
        return []
    
    # 构建 RRF 分数字典
    rrf_scores: Dict[Union[int, str], float] = {}
    doc_info: Dict[Union[int, str], Dict] = {}
    
    # 处理 Dense 结果
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc.get('id')
        if doc_id is None:
            continue
        
        # RRF 分数累加
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
        
        # 保存文档信息 (优先使用 Dense 结果)
        if doc_id not in doc_info:
            doc_info[doc_id] = {
                "id": doc_id,
                "text": doc.get('text', ''),
                "source": doc.get('source', ''),
                "url": doc.get('url', ''),
                "title": doc.get('title', ''),
                "mineral_type": doc.get('mineral_type', ''),
                "dense_score": doc.get('distance', 0.0),
                "sparse_score": 0.0,
                "dense_rank": rank,
                "sparse_rank": 0
            }
    
    # 处理 Sparse 结果
    for rank, doc in enumerate(sparse_results, start=1):
        doc_id = doc.get('id')
        if doc_id is None:
            continue
        
        # RRF 分数累加
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
        
        # 更新文档信息
        if doc_id in doc_info:
            doc_info[doc_id]["sparse_score"] = doc.get('distance', 0.0)
            doc_info[doc_id]["sparse_rank"] = rank
        else:
            doc_info[doc_id] = {
                "id": doc_id,
                "text": doc.get('text', ''),
                "source": doc.get('source', ''),
                "url": doc.get('url', ''),
                "title": doc.get('title', ''),
                "mineral_type": doc.get('mineral_type', ''),
                "dense_score": 0.0,
                "sparse_score": doc.get('distance', 0.0),
                "dense_rank": 0,
                "sparse_rank": rank
            }
    
    # 按 RRF 分数排序
    sorted_doc_ids = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # 构建最终结果
    results = []
    for rank, (doc_id, rrf_score) in enumerate(sorted_doc_ids[:top_k], start=1):
        info = doc_info[doc_id]
        result = RetrievalResult(
            id=info["id"],
            text=info["text"],
            score=rrf_score,
            rank=rank,
            source=info["source"],
            url=info["url"],
            title=info["title"],
            mineral_type=info["mineral_type"],
            dense_score=info["dense_score"],
            sparse_score=info["sparse_score"],
            dense_rank=info["dense_rank"],
            sparse_rank=info["sparse_rank"]
        )
        results.append(result)
    
    logger.info(f"🔀 RRF 融合完成: Dense={len(dense_results)}, Sparse={len(sparse_results)} → Top-{len(results)}")
    
    return results


def weighted_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    top_k: int = 10,
    alpha: float = 0.5
) -> List[RetrievalResult]:
    """
    加权融合算法
    
    公式: score = alpha * normalized_dense_score + (1-alpha) * normalized_sparse_score
    
    Args:
        dense_results: 稠密向量检索结果
        sparse_results: 稀疏向量检索结果
        top_k: 返回 Top-K 结果
        alpha: 稠密向量权重 (0-1)
    
    Returns:
        融合后的检索结果列表
    """
    if not dense_results and not sparse_results:
        return []
    
    # 构建分数字典
    doc_scores: Dict[Union[int, str], Dict] = {}
    
    # 处理 Dense 结果
    max_dense = max([d.get('distance', 0) for d in dense_results]) if dense_results else 1.0
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc.get('id')
        if doc_id is None:
            continue
        
        normalized_score = doc.get('distance', 0) / max_dense if max_dense > 0 else 0
        
        doc_scores[doc_id] = {
            "id": doc_id,
            "text": doc.get('text', ''),
            "dense_score": normalized_score,
            "sparse_score": 0.0,
            "source": doc.get('source', ''),
            "url": doc.get('url', ''),
            "title": doc.get('title', ''),
            "mineral_type": doc.get('mineral_type', '')
        }
    
    # 处理 Sparse 结果
    max_sparse = max([s.get('distance', 0) for s in sparse_results]) if sparse_results else 1.0
    for rank, doc in enumerate(sparse_results, start=1):
        doc_id = doc.get('id')
        if doc_id is None:
            continue
        
        normalized_score = doc.get('distance', 0) / max_sparse if max_sparse > 0 else 0
        
        if doc_id in doc_scores:
            doc_scores[doc_id]["sparse_score"] = normalized_score
        else:
            doc_scores[doc_id] = {
                "id": doc_id,
                "text": doc.get('text', ''),
                "dense_score": 0.0,
                "sparse_score": normalized_score,
                "source": doc.get('source', ''),
                "url": doc.get('url', ''),
                "title": doc.get('title', ''),
                "mineral_type": doc.get('mineral_type', '')
            }
    
    # 计算加权分数
    for doc_id, info in doc_scores.items():
        weighted_score = alpha * info["dense_score"] + (1 - alpha) * info["sparse_score"]
        info["score"] = weighted_score
    
    # 排序
    sorted_docs = sorted(
        doc_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )[:top_k]
    
    # 构建结果
    results = []
    for rank, info in enumerate(sorted_docs, start=1):
        result = RetrievalResult(
            id=info["id"],
            text=info["text"],
            score=info["score"],
            rank=rank,
            source=info["source"],
            url=info["url"],
            title=info["title"],
            mineral_type=info["mineral_type"]
        )
        results.append(result)
    
    return results


# ==================== 混合检索器 ====================

class HybridSearcher:
    """
    混合检索器
    同时支持 BM25 稀疏检索 + 稠密向量检索 + RRF 融合
    """
    
    def __init__(
        self,
        embedder,
        milvus_client,
        bm25_encoder = None,
        bm25_index = None,
        top_k: int = 10,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5
    ):
        """
        初始化混合检索器
        
        Args:
            embedder: Qwen3-Embedding 向量化模型
            milvus_client: Milvus 客户端
            bm25_encoder: BM25 编码器 (可选，保留兼容)
            bm25_index: BM25Index 实例 (可选，稀疏通道真正生效需要它)
            top_k: 默认返回 Top-K 结果
            fusion_method: 融合方法 ('rrf' 或 'weighted')
            rrf_k: RRF 调整参数
            alpha: 加权融合权重
        """
        self.embedder = embedder
        self.milvus = milvus_client
        self.bm25_encoder = bm25_encoder
        self.bm25_index = bm25_index
        self.top_k = top_k
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.alpha = alpha
        
        logger.info(f"✅ 混合检索器初始化完成")
        logger.info(f"   - 融合方法: {fusion_method}")
        if fusion_method == "rrf":
            logger.info(f"   - RRF k 参数: {rrf_k}")
        else:
            logger.info(f"   - 加权系数 alpha: {alpha}")
    
    def search(
        self,
        query: str,
        visual_anchors: Optional[List[VisualAnchor]] = None,
        top_k: Optional[int] = None,
        fusion_method: Optional[str] = None,
        return_raw: bool = False
    ) -> Union[List[RetrievalResult], Tuple[List[Dict], List[Dict]]]:
        """
        执行混合检索
        
        Args:
            query: 文本查询
            visual_anchors: 视觉锚点列表 (可选)
            top_k: 返回 Top-K 结果
            fusion_method: 融合方法 (覆盖默认)
            return_raw: 是否返回原始检索结果
        
        Returns:
            融合后的结果，或 (dense_results, sparse_results) 元组
        """
        # 融合查询构建
        fused_query = self._build_fused_query(query, visual_anchors)
        
        # 双路并发检索
        dense_results, sparse_results = self._parallel_search(fused_query, top_k or self.top_k)
        
        # 融合
        method = fusion_method or self.fusion_method
        
        if method == "rrf":
            fused_results = rrf_fusion(
                dense_results,
                sparse_results,
                top_k or self.top_k,
                k=self.rrf_k
            )
        elif method == "weighted":
            fused_results = weighted_fusion(
                dense_results,
                sparse_results,
                top_k or self.top_k,
                alpha=self.alpha
            )
        else:
            raise ValueError(f"未知的融合方法: {method}")
        
        if return_raw:
            return dense_results, sparse_results
        
        return fused_results
    
    def _build_fused_query(
        self,
        query: str,
        visual_anchors: Optional[List[VisualAnchor]] = None
    ) -> str:
        """
        构建融合查询
        将文本查询与视觉锚点融合
        
        Args:
            query: 原始文本查询
            visual_anchors: 视觉锚点列表
        
        Returns:
            融合后的查询字符串
        """
        if not visual_anchors:
            return query
        
        # 提取主要矿物名称
        anchor_names = [anchor.mineral_name for anchor in visual_anchors[:3]]
        
        # 融合查询: "矿物A 矿物B 原始问题"
        fused_query = " ".join(anchor_names) + " " + query
        
        logger.info(f"🔗 融合查询: {fused_query}")
        
        return fused_query
    
    def _parallel_search(
        self,
        query: str,
        top_k: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        并发执行双路检索
        
        Args:
            query: 查询文本
            top_k: Top-K
        
        Returns:
            (dense_results, sparse_results)
        """
        dense_results = []
        sparse_results = []
        
        # 1. Dense 向量检索
        query_embedding = self.embedder.encode_queries(query)
        
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding[0]  # 取第一条
        
        try:
            dense_results = self.milvus.search_dense(
                query_vector=query_embedding,
                top_k=top_k * 2  # 多召回一些用于融合
            )
            logger.info(f"📡 Dense 检索: {len(dense_results)} 条结果")
        except Exception as e:
            logger.warning(f"⚠️ Dense 检索失败: {e}")
        
        # 2. Sparse 文本检索 (BM25)
        if self.bm25_index or self.bm25_encoder:
            try:
                sparse_results = self._bm25_search(query, top_k * 2)
                logger.info(f"📡 Sparse 检索: {len(sparse_results)} 条结果")
            except Exception as e:
                logger.warning(f"⚠️ Sparse 检索失败: {e}")
        
        return dense_results, sparse_results
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """
        BM25 文本检索
        
        Args:
            query: 查询文本
            top_k: Top-K
        
        Returns:
            BM25 检索结果
        """
        if self.bm25_index is not None:
            return self.bm25_index.search(query, top_k=top_k)

        # 没有 BM25 索引时稀疏通道为空，RRF 退化为稠密单路
        logger.warning("未提供 BM25 索引，稀疏通道为空")
        return []
    
    def search_with_filter(
        self,
        query: str,
        filter_expr: str,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        带过滤条件的检索
        
        Args:
            query: 查询文本
            filter_expr: 过滤表达式
            top_k: Top-K
        
        Returns:
            检索结果
        """
        # 获取 Dense 向量
        query_embedding = self.embedder.encode_queries(query)
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding[0]
        
        # 带过滤的 Dense 检索
        dense_results = self.milvus.search_dense(
            query_vector=query_embedding,
            top_k=top_k,
            filter_expr=filter_expr
        )
        
        # 转换为 RetrievalResult
        results = []
        for rank, doc in enumerate(dense_results, start=1):
            result = RetrievalResult(
                id=doc.get('id'),
                text=doc.get('text', ''),
                score=doc.get('distance', 0),
                rank=rank,
                source=doc.get('source', ''),
                url=doc.get('url', ''),
                title=doc.get('title', ''),
                mineral_type=doc.get('mineral_type', '')
            )
            results.append(result)
        
        return results


# ==================== 主函数 ====================

def create_hybrid_searcher(
    embedder,
    milvus_client,
    bm25_encoder = None,
    **kwargs
) -> HybridSearcher:
    """
    工厂函数：创建混合检索器
    
    Args:
        embedder: 向量化模型
        milvus_client: Milvus 客户端
        bm25_encoder: BM25 编码器
        **kwargs: 其他参数
    
    Returns:
        HybridSearcher 实例
    """
    return HybridSearcher(
        embedder=embedder,
        milvus_client=milvus_client,
        bm25_encoder=bm25_encoder,
        **kwargs
    )


# ==================== 示例用法 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("混合检索模块测试")
    print("=" * 60)
    
    # 测试 RRF 融合
    dense_results = [
        {"id": 1, "text": "文档1", "distance": 0.9},
        {"id": 2, "text": "文档2", "distance": 0.8},
        {"id": 3, "text": "文档3", "distance": 0.7},
    ]
    
    sparse_results = [
        {"id": 2, "text": "文档2", "distance": 0.85},
        {"id": 3, "text": "文档3", "distance": 0.75},
        {"id": 4, "text": "文档4", "distance": 0.65},
    ]
    
    # RRF 融合
    fused = rrf_fusion(dense_results, sparse_results, top_k=3)
    
    print("\n📊 RRF 融合结果:")
    for r in fused:
        print(f"  Rank {r.rank}: ID={r.id}, Score={r.score:.4f}")
        print(f"    Dense: score={r.dense_score:.4f}, rank={r.dense_rank}")
        print(f"    Sparse: score={r.sparse_score:.4f}, rank={r.sparse_rank}")
    
    # 加权融合
    weighted = weighted_fusion(dense_results, sparse_results, top_k=3, alpha=0.5)
    
    print("\n📊 加权融合结果 (alpha=0.5):")
    for r in weighted:
        print(f"  Rank {r.rank}: ID={r.id}, Score={r.score:.4f}")
    
    print("\n✅ 测试完成!")
