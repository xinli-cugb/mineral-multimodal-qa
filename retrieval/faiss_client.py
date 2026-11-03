"""
FAISS 向量数据库客户端封装
"""
import os
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FAISSClient:
    """
    FAISS 向量数据库客户端
    提供与 MilvusClient 相似的接口
    """

    DEFAULT_INDEX_FILE = "index.faiss"
    DEFAULT_DATA_FILE = "data.pkl"

    def __init__(
        self,
        index_path: str = "./faiss_mineral",
        dimension: int = 1024
    ):
        """
        初始化 FAISS 客户端

        Args:
            index_path: FAISS 索引文件目录
            dimension: 向量维度
        """
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.index = None
        self.data = []
        self._loaded = False

    def load(self):
        """加载索引和数据"""
        if not self._loaded:
            index_file = self.index_path / self.DEFAULT_INDEX_FILE
            data_file = self.index_path / self.DEFAULT_DATA_FILE

            if index_file.exists() and data_file.exists():
                self.index = faiss.read_index(str(index_file))
                with open(data_file, 'rb') as f:
                    self.data = pickle.load(f)
                self._loaded = True
                logger.info(f"   FAISS 索引加载成功: {len(self.data)} 条记录")
            else:
                logger.warning(f"   FAISS 索引文件不存在: {index_file}")

    def create_index(self, dimension: int = 1024):
        """
        创建新索引

        Args:
            dimension: 向量维度
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.data = []
        logger.info(f"   FAISS 索引创建成功, dimension={dimension}")

    def add(self, vectors: np.ndarray, metadata: List[Dict]):
        """
        添加向量和元数据

        Args:
            vectors: 向量数组 (N, dimension)
            metadata: 元数据列表
        """
        if self.index is None:
            self.create_index(vectors.shape[1])

        vectors = np.array(vectors).astype('float32')
        self.index.add(vectors)
        self.data.extend(metadata)
        logger.info(f"   添加 {len(metadata)} 条数据, 总计 {self.index.ntotal} 条")

    def save(self):
        """保存索引和数据"""
        self.index_path.mkdir(parents=True, exist_ok=True)

        index_file = self.index_path / self.DEFAULT_INDEX_FILE
        data_file = self.index_path / self.DEFAULT_DATA_FILE

        faiss.write_index(self.index, str(index_file))
        with open(data_file, 'wb') as f:
            pickle.dump(self.data, f)

        logger.info(f"   FAISS 索引已保存: {index_file}")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        向量检索

        Args:
            query_vector: 查询向量 (dimension,) 或 (1, dimension)
            top_k: 返回数量
            output_fields: 输出字段（暂不支持）

        Returns:
            检索结果列表
        """
        if self.index is None or not self._loaded:
            self.load()

        if self.index is None:
            return []

        query_vector = np.array(query_vector).astype('float32')
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        D, I = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results = []
        for i, idx in enumerate(I[0]):
            if idx >= 0 and idx < len(self.data):
                result = {"id": idx, "distance": float(D[0][i])}
                result.update(self.data[idx])
                results.append(result)

        return results

    def query(
        self,
        filter_expr: str = None,
        output_fields: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        查询（FAISS 不支持复杂过滤，仅返回前 N 条）

        Args:
            filter_expr: 过滤表达式（暂不支持）
            output_fields: 输出字段
            limit: 返回数量

        Returns:
            查询结果列表
        """
        if self.index is None or not self._loaded:
            self.load()

        if self.index is None:
            return []

        results = []
        for i, item in enumerate(self.data[:limit]):
            result = {"id": i}
            result.update(item)
            results.append(result)

        return results

    def delete(self, filter_expr: str = None) -> bool:
        """
        删除（FAISS 不支持删除，仅标记）

        Args:
            filter_expr: 过滤表达式

        Returns:
            是否成功
        """
        logger.warning("   FAISS 不支持删除操作")
        return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def close(self):
        """关闭客户端"""
        if self.index is not None:
            self.save()
        logger.info("   FAISS 客户端已关闭")


def create_faiss_client(
    index_path: str = "./faiss_mineral",
    dimension: int = 1024
) -> FAISSClient:
    """
    工厂函数：创建 FAISS 客户端

    Args:
        index_path: FAISS 索引文件目录
        dimension: 向量维度

    Returns:
        FAISSClient 实例
    """
    return FAISSClient(index_path=index_path, dimension=dimension)
