"""
Milvus 向量数据库客户端封装
支持 Dense 稠密向量 + Sparse 稀疏向量(BM25) 混合索引
实现双路混合检索的基础架构

支持两种模式:
- Milvus Lite (嵌入式): 使用本地文件存储，无需 Docker
- Milvus Server: 连接远程 Milvus 服务
"""
from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
from pathlib import Path
import json
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusClient:
    """
    Milvus 向量数据库客户端
    支持 Milvus Lite (嵌入式) 和 Milvus Server 两种模式
    """
    
    # 集合名称
    DEFAULT_COLLECTION = "mineral_knowledge"
    
    # 字段定义
    DENSE_FIELD = "dense_vector"
    SPARSE_FIELD = "sparse_vector"
    TEXT_FIELD = "text"
    ID_FIELD = "id"
    METADATA_FIELDS = ["title", "source", "url", "mineral_type"]
    
    def __init__(
        self,
        uri: str = "./milvus_lite.db",
        host: str = "localhost",
        port: int = 19530,
        user: str = "",
        password: str = "",
        collection_name: str = DEFAULT_COLLECTION,
        use_lite: bool = True
    ):
        """
        初始化 Milvus 客户端
        
        Args:
            uri: Milvus Lite 数据库文件路径 (如 "./milvus_lite.db")
            host: Milvus 服务器地址 (仅用于非 Lite 模式)
            port: Milvus 服务器端口 (仅用于非 Lite 模式)
            collection_name: 集合名称
            use_lite: 是否使用 Milvus Lite (嵌入式模式)
        """
        # 清除代理设置
        for proxy_key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
            os.environ.pop(proxy_key, None)
        
        try:
            # 尝试导入新版 MilvusClient API
            try:
                from pymilvus import MilvusClient as NewMilvusClient
                self._use_new_api = True
                logger.info("   使用 pymilvus 新版 API (MilvusClient)")
            except ImportError:
                self._use_new_api = False
                logger.info("   使用 pymilvus 旧版 API (connections)")
            
            # 保存配置
            self.uri = uri
            self.host = host
            self.port = port
            self.collection_name = collection_name
            self.use_lite = use_lite
            
            if self._use_new_api:
                # 新版 API
                if use_lite:
                    # 嵌入式模式 - 使用本地文件
                    self.client = NewMilvusClient(uri=uri)
                    self.is_lite = True
                    logger.info(f"✅ Milvus Lite 连接成功: {uri}")
                else:
                    # 远程模式
                    self.client = NewMilvusClient(uri=f"http://{host}:{port}")
                    self.is_lite = False
                    logger.info(f"✅ Milvus 服务器连接成功: {host}:{port}")
            else:
                # 旧版 API - 保留向后兼容
                from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
                self.DataType = DataType
                self.FieldSchema = FieldSchema
                self.CollectionSchema = CollectionSchema
                self.Collection = Collection
                
                alias = "default"
                connections.connect(
                    alias=alias,
                    host=host,
                    port=port,
                    user=user,
                    password=password
                )
                self.alias = alias
                self.is_lite = False
                logger.info(f"✅ Milvus 连接成功: {host}:{port}")
            
            self.collection = None
            
        except ImportError:
            raise ImportError("请安装 pymilvus: pip install pymilvus")
        except Exception as e:
            raise RuntimeError(f"Milvus 连接失败: {e}")
    
    def create_collection(
        self,
        dimension: int = 1024,
        description: str = "矿物知识库",
        with_sparse: bool = True,
        sparse_dim: int = 65536,
        auto_id: bool = True,
        overwrite: bool = False
    ) -> bool:
        """
        创建集合，支持混合向量索引
        
        Args:
            dimension: Dense 向量维度 (如 Qwen3-Embedding 输出 1024)
            description: 集合描述
            with_sparse: 是否创建 Sparse 向量字段 (BM25)
            sparse_dim: Sparse 向量维度
            auto_id: 是否自动生成 ID
            overwrite: 是否覆盖已存在的集合
        
        Returns:
            是否创建成功
        """
        from pymilvus import CollectionSchema, FieldSchema, DataType
        
        # 检查集合是否存在
        from pymilvus import utility
        if utility.has_collection(self.collection_name, using=self.alias):
            if overwrite:
                utility.drop_collection(self.collection_name, using=self.alias)
                logger.info(f"🗑️ 已删除旧集合: {self.collection_name}")
            else:
                logger.info(f"📁 集合已存在: {self.collection_name}")
                self.collection = Collection(self.collection_name, using=self.alias)
                return False
        
        # 定义字段
        fields = []
        
        # 1. ID 字段
        if auto_id:
            fields.append(FieldSchema(
                name=self.ID_FIELD,
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            ))
        else:
            fields.append(FieldSchema(
                name=self.ID_FIELD,
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False
            ))
        
        # 2. 文本字段
        fields.append(FieldSchema(
            name=self.TEXT_FIELD,
            dtype=DataType.VARCHAR,
            max_length=65535,
            description="原始文本"
        ))
        
        # 3. Dense 向量字段
        fields.append(FieldSchema(
            name=self.Dense_FIELD,
            dtype=DataType.FLOAT_VECTOR,
            dim=dimension,
            description="稠密向量"
        ))
        
        # 4. 元数据字段
        for meta_field in self.METADATA_FIELDS:
            fields.append(FieldSchema(
                name=meta_field,
                dtype=DataType.VARCHAR,
                max_length=512,
                description=f"{meta_field} 元数据"
            ))
        
        # 5. Sparse 向量字段 (BM25)
        if with_sparse:
            fields.append(FieldSchema(
                name=self.Sparse_FIELD,
                dtype=DataType.FLOAT_VECTOR,
                dim=sparse_dim,
                description="稀疏向量 (BM25)"
            ))
        
        # 创建 Schema
        schema = CollectionSchema(
            fields=fields,
            description=description
        )
        
        # 创建集合
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using=self.alias
        )
        
        logger.info(f"✅ 集合创建成功: {self.collection_name}")
        logger.info(f"   - Dense 向量维度: {dimension}")
        logger.info(f"   - Sparse 向量: {'启用' if with_sparse else '禁用'}")
        
        return True
    
    def create_hybrid_index(
        self,
        dense_index_type: str = "HNSW",
        sparse_index_type: str = "SPARSE_INVERTED_INDEX",
        metric_type: str = "IP",
        dense_params: Optional[Dict] = None,
        sparse_params: Optional[Dict] = None
    ) -> bool:
        """
        创建混合索引 (Dense + Sparse)
        
        Args:
            dense_index_type: Dense 向量索引类型 (HNSW, IVF_FLAT, etc.)
            sparse_index_type: Sparse 向量索引类型
            metric_type: 距离度量方式 (IP, L2, COSINE)
            dense_params: Dense 索引参数
            sparse_params: Sparse 索引参数
        
        Returns:
            是否创建成功
        """
        if self.collection is None:
            raise RuntimeError("集合未初始化，请先调用 create_collection")
        
        # 默认参数
        if dense_params is None:
            dense_params = {
                "M": 16,
                "efConstruction": 256
            }
        
        if sparse_params is None:
            sparse_params = {
                "inverted_index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP"
            }
        
        # 构建索引参数
        index_params = {
            "index_type": dense_index_type,
            "metric_type": metric_type,
            "params": dense_params
        }
        
        # 创建 Dense 向量索引
        self.collection.create_index(
            field_name=self.Dense_FIELD,
            index_params=index_params
        )
        
        # 如果存在 Sparse 字段，创建 Sparse 索引
        try:
            sparse_index_params = {
                "index_type": sparse_index_type,
                "metric_type": metric_type,
                "params": sparse_params
            }
            self.collection.create_index(
                field_name=self.Sparse_FIELD,
                index_params=sparse_index_params
            )
            logger.info("✅ 混合索引创建成功 (Dense + Sparse)")
        except Exception as e:
            logger.warning(f"⚠️ Sparse 索引创建失败: {e}")
        
        # 加载集合到内存
        self.collection.load()
        
        return True
    
    def insert(
        self,
        texts: List[str],
        dense_vectors: List[np.ndarray],
        sparse_vectors: Optional[List[np.ndarray]] = None,
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 100
    ) -> List[int]:
        """
        插入数据
        
        Args:
            texts: 文本列表
            dense_vectors: Dense 向量列表
            sparse_vectors: Sparse 向量列表 (可选)
            metadata: 元数据列表
            batch_size: 批次大小
        
        Returns:
            插入的 ID 列表
        """
        if self.collection is None:
            raise RuntimeError("集合未初始化")
        
        all_ids = []
        
        # 分批插入
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_dense = dense_vectors[i:i + batch_size]
            batch_sparse = sparse_vectors[i:i + batch_size] if sparse_vectors else None
            
            # 准备数据
            entities = []
            
            # 文本
            entities.append(batch_texts)
            
            # Dense 向量
            entities.append(batch_dense)
            
            # 元数据
            if metadata:
                batch_meta = metadata[i:i + batch_size]
                for field in self.METADATA_FIELDS:
                    field_data = [m.get(field, "") for m in batch_meta]
                    entities.append(field_data)
            else:
                for _ in batch_texts:
                    entities.append([""] * len(self.METADATA_FIELDS))
            
            # Sparse 向量
            if batch_sparse:
                entities.append(batch_sparse)
            
            # 插入
            insert_result = self.collection.insert(entities)
            all_ids.extend(insert_result.primary_keys)
            
            logger.info(f"📥 已插入 {len(batch_texts)} 条数据")
        
        # 刷新
        self.collection.flush()
        
        logger.info(f"✅ 插入完成，总计 {len(all_ids)} 条数据")
        
        return all_ids
    
    def search_dense(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Dense 向量检索
        
        Args:
            query_vector: 查询向量
            top_k: 返回 Top-K 结果
            filter_expr: 过滤表达式
            output_fields: 输出字段
        
        Returns:
            检索结果列表
        """
        if self.collection is None:
            raise RuntimeError("集合未初始化")
        
        # 确保向量格式正确
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector)
        
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # 默认输出字段
        if output_fields is None:
            output_fields = [self.TEXT_FIELD] + self.METADATA_FIELDS + [self.ID_FIELD]
        
        # 搜索参数
        search_params = {
            "metric_type": "IP",
            "params": {"ef": 64}
        }
        
        # 执行搜索
        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field=self.Dense_FIELD,
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields
        )
        
        # 解析结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.id,
                    "distance": hit.distance,
                    "text": hit.entity.get(self.TEXT_FIELD)
                }
                # 添加元数据
                for field in self.METADATA_FIELDS:
                    result[field] = hit.entity.get(field)
                
                formatted_results.append(result)
        
        return formatted_results
    
    def search_sparse(
        self,
        query_sparse_vector: np.ndarray,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Sparse 向量检索 (BM25)
        
        Args:
            query_sparse_vector: 查询稀疏向量
            top_k: 返回 Top-K 结果
            filter_expr: 过滤表达式
            output_fields: 输出字段
        
        Returns:
            检索结果列表
        """
        if self.collection is None:
            raise RuntimeError("集合未初始化")
        
        # 确保向量格式正确
        if isinstance(query_sparse_vector, list):
            query_sparse_vector = np.array(query_sparse_vector)
        
        if query_sparse_vector.ndim == 1:
            query_sparse_vector = query_sparse_vector.reshape(1, -1)
        
        # 默认输出字段
        if output_fields is None:
            output_fields = [self.TEXT_FIELD] + self.METADATA_FIELDS + [self.ID_FIELD]
        
        # Sparse 搜索参数
        search_params = {
            "metric_type": "IP",
            "params": {}
        }
        
        # 执行搜索
        results = self.collection.search(
            data=[query_sparse_vector.tolist()],
            anns_field=self.Sparse_FIELD,
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields
        )
        
        # 解析结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.id,
                    "distance": hit.distance,
                    "text": hit.entity.get(self.TEXT_FIELD)
                }
                for field in self.METADATA_FIELDS:
                    result[field] = hit.entity.get(field)
                
                formatted_results.append(result)
        
        return formatted_results
    
    def hybrid_search(
        self,
        query_dense: np.ndarray,
        query_sparse: Optional[np.ndarray],
        top_k: int = 10,
        alpha: float = 0.5,
        fusion_method: str = "rrf"
    ) -> List[Dict]:
        """
        混合检索接口 (供外部调用)
        
        Args:
            query_dense: Dense 查询向量
            query_sparse: Sparse 查询向量
            top_k: 返回 Top-K 结果
            alpha: 混合权重 (dense * alpha + sparse * (1-alpha))
            fusion_method: 融合方法 ('rrf', 'weighted')
        
        Returns:
            融合后的检索结果
        """
        # 双路检索
        dense_results = self.search_dense(query_dense, top_k * 2)
        sparse_results = []
        
        if query_sparse is not None:
            sparse_results = self.search_sparse(query_sparse, top_k * 2)
        
        # RRF 融合
        from .hybrid_search import rrf_fusion
        fused_results = rrf_fusion(
            dense_results,
            sparse_results,
            top_k,
            k=60
        )
        
        return fused_results
    
    def get_collection_info(self) -> Dict:
        """
        获取集合信息
        
        Returns:
            集合信息字典
        """
        if self.collection is None:
            return {}
        
        info = {
            "name": self.collection_name,
            "num_entities": self.collection.num_entities,
            "description": self.collection.description,
            "fields": []
        }
        
        # 获取字段信息
        try:
            schema = self.collection.schema
            for field in schema.fields:
                info["fields"].append({
                    "name": field.name,
                    "dtype": field.dtype,
                    "dim": getattr(field, "dim", None)
                })
        except:
            pass
        
        return info
    
    def delete_collection(self) -> bool:
        """
        删除集合
        
        Returns:
            是否删除成功
        """
        from pymilvus import utility
        
        if utility.has_collection(self.collection_name, using=self.alias):
            utility.drop_collection(self.collection_name, using=self.alias)
            logger.info(f"🗑️ 集合已删除: {self.collection_name}")
            return True
        
        return False
    
    def close(self):
        """关闭连接"""
        connections.disconnect(alias=self.alias)
        logger.info("🔌 Milvus 连接已关闭")


# ==================== BM25 工具类 ====================

class BM25Encoder:
    """
    BM25 稀疏向量编码器
    用于将文本转换为稀疏向量 (词袋模型)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化 BM25 编码器
        
        Args:
            k1: BM25 参数 k1
            b: BM25 参数 b
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = {}
        self.idf = {}
        self.doc_len = []
        self.corpus = []
    
    def fit(self, corpus: List[str]):
        """
        从语料库学习 IDF
        
        Args:
            corpus: 文档列表
        """
        import jieba
        from collections import Counter
        
        self.corpus = corpus
        self.corpus_size = len(corpus)
        
        # 分词并统计
        nd = {}
        for document in corpus:
            tokens = jieba.lcut(document)
            self.doc_len.append(len(tokens))
            
            frequencies = Counter(tokens)
            for word, freq in frequencies.items():
                nd[word] = nd.get(word, 0) + 1
        
        # 计算 IDF
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
        
        # 平均文档长度
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0
        
        logger.info(f"✅ BM25 模型训练完成，词汇量: {len(self.idf)}")
    
    def encode(self, query: str) -> np.ndarray:
        """
        将查询编码为稀疏向量
        
        Args:
            query: 查询文本
        
        Returns:
            稀疏向量 (dict 格式: {word_id: score})
        """
        import jieba
        
        tokens = jieba.lcut(query)
        
        # 计算 BM25 分数
        scores = {}
        for token in tokens:
            if token in self.idf:
                scores[token] = self.idf[token]
        
        return scores
    
    def encode_batch(self, queries: List[str]) -> List[Dict]:
        """
        批量编码
        
        Args:
            queries: 查询列表
        
        Returns:
            稀疏向量列表
        """
        return [self.encode(q) for q in queries]


import math


def create_milvus_client(
    host: str = "localhost",
    port: int = 19530,
    collection_name: str = "mineral_knowledge",
    dimension: int = 1024,
    with_sparse: bool = True,
    uri: str = "./milvus_lite.db",
    use_lite: bool = True
) -> MilvusClient:
    """
    工厂函数：创建 Milvus 客户端
    
    Args:
        host: 服务器地址
        port: 服务器端口
        collection_name: 集合名称
        dimension: 向量维度
        with_sparse: 是否启用 Sparse 向量
        uri: Milvus Lite 数据库文件路径
        use_lite: 是否使用 Milvus Lite (嵌入式模式)
    
    Returns:
        MilvusClient 实例
    """
    client = MilvusClient(
        uri=uri,
        host=host,
        port=port,
        collection_name=collection_name,
        use_lite=use_lite
    )
    
    # 创建集合和索引
    client.create_collection(
        dimension=dimension,
        with_sparse=with_sparse
    )
    
    client.create_hybrid_index()
    
    return client


# ==================== 示例用法 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("Milvus 客户端测试")
    print("=" * 60)
    
    # 注意: 需要先启动 Milvus 服务
    # docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus
    
    try:
        # 创建客户端
        client = MilvusClient(
            host="localhost",
            port=19530,
            collection_name="mineral_test"
        )
        
        # 创建集合 (Dense 维度 1024, 启用 Sparse)
        client.create_collection(
            dimension=1024,
            with_sparse=True,
            overwrite=True
        )
        
        # 创建索引
        client.create_hybrid_index()
        
        # 获取集合信息
        info = client.get_collection_info()
        print(f"\n集合信息: {info}")
        
        print("\n✅ Milvus 客户端测试完成")
        
    except Exception as e:
        print(f"⚠️ 测试失败 (请确保 Milvus 服务已启动): {e}")
