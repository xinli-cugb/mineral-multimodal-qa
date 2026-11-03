"""
Qwen3-Embedding 向量化模型封装
支持文本到稠密向量(Dense)的转换，用于语义检索
"""
import torch
import numpy as np
from typing import List, Union, Optional, Dict
from pathlib import Path
import json

# Qwen3-Embedding 官方查询模板：查询侧加 Instruct，文档侧不加
DEFAULT_QUERY_INSTRUCT = "给定一个矿物相关的查询，检索能够回答该查询的矿物资料。"
QUERY_TEMPLATE = "Instruct: {instruct}\nQuery: {query}"


class Qwen3Embedder:
    """
    Qwen3-Embedding 向量化编码器
    将文本转换为稠密向量，用于语义检索
    
    支持 HuggingFace 和 ModelScope 下载
    """
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = None,
        normalize: bool = True,
        max_length: int = 512
    ):
        """
        初始化 Qwen3-Embedding 编码器
        
        Args:
            model_path: 模型路径或 HuggingFace Hub 名称
            device: 运行设备
            normalize: 是否对输出向量进行 L2 归一化
            max_length: 最大输入长度
        """
        # 设置默认设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.normalize = normalize
        self.max_length = max_length
        
        # 加载模型和分词器
        self._load_model(model_path)
        
        print(f"✅ Qwen3-Embedding 模型加载成功")
        print(f"   模型路径: {model_path}")
        print(f"   设备: {self.device}")
        print(f"   向量维度: {self.embedding_dim}")
    
    def _load_model(self, model_path: str):
        """加载模型和分词器"""
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        self.model.to(self.device)
        self.model.eval()
        
        # 获取向量维度
        self.embedding_dim = self.model.config.hidden_size
    
    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        将文本编码为向量
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批次大小
            show_progress: 是否显示进度条
        
        Returns:
            numpy 数组，shape: [num_texts, embedding_dim]
        """
        # 统一为列表
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # 移动到设备
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # 前向传播
            outputs = self.model(**encoded)
            
            # Qwen3-Embedding 使用 last-token pooling（不是 [CLS]）
            last_hidden = outputs.last_hidden_state
            last_idx = encoded["attention_mask"].sum(dim=1) - 1
            last_idx = last_idx.clamp(min=0)
            row_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
            embeddings = last_hidden[row_idx, last_idx]
            
            # L2 归一化
            if self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.float().cpu().numpy())
            
            if show_progress:
                progress = min(i + batch_size, len(texts)) / len(texts) * 100
                print(f"\r编码进度: {progress:.1f}%", end="")
        
        if show_progress:
            print()
        
        return np.concatenate(all_embeddings, axis=0)
    
    @torch.no_grad()
    def encode_queries(self, queries: Union[str, List[str]]) -> np.ndarray:
        """
        编码查询文本（与 encode 等价，语义更明确）
        
        Args:
            queries: 查询文本
        
        Returns:
            查询向量
        """
        if isinstance(queries, str):
            queries = [queries]
        wrapped = [
            QUERY_TEMPLATE.format(instruct=DEFAULT_QUERY_INSTRUCT, query=q)
            for q in queries
        ]
        return self.encode(wrapped)
    
    @torch.no_grad()
    def encode_corpus(
        self, 
        corpus: List[Dict[str, str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        编码文档语料库
        
        Args:
            corpus: 文档列表，每个文档为 dict，需包含 'text' 或 'content' 字段
            batch_size: 批次大小
            show_progress: 是否显示进度条
        
        Returns:
            文档向量
        """
        # 提取文本
        texts = []
        for doc in corpus:
            if 'text' in doc:
                texts.append(doc['text'])
            elif 'content' in doc:
                texts.append(doc['content'])
            elif 'passage' in doc:
                texts.append(doc['passage'])
            else:
                # 尝试获取第一个字符串字段
                for key, value in doc.items():
                    if isinstance(value, str):
                        texts.append(value)
                        break
        
        return self.encode(texts, batch_size, show_progress)
    
    def compute_similarity(
        self,
        query_emb: np.ndarray,
        doc_embs: np.ndarray
    ) -> np.ndarray:
        """
        计算查询与文档的相似度
        
        Args:
            query_emb: 查询向量, shape: [embedding_dim] 或 [1, embedding_dim]
            doc_embs: 文档向量, shape: [num_docs, embedding_dim]
        
        Returns:
            相似度分数, shape: [num_docs]
        """
        if query_emb.ndim == 1:
            query_emb = query_emb.unsqueeze(0)
        
        # 点积或余弦相似度（因为已归一化，点积=余弦）
        similarities = np.dot(query_emb, doc_embs.T)
        
        return similarities.squeeze()
    
    def get_config(self) -> Dict:
        """获取模型配置信息"""
        return {
            "model_path": self.model.name_or_path,
            "device": self.device,
            "embedding_dim": self.embedding_dim,
            "normalize": self.normalize,
            "max_length": self.max_length
        }


class Qwen3EmbeddingWithHardNegatives(Qwen3Embedder):
    """
    支持难负样本的 Qwen3-Embedding
    用于 InfoNCE 对比学习微调
    """
    
    def __init__(
        self,
        model_path: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        normalize: bool = True,
        max_length: int = 512,
        margin: float = 0.5
    ):
        """
        初始化支持难负样本的编码器
        
        Args:
            margin: 正负样本相似度 margin，用于 InfoNCE 损失
        """
        super().__init__(model_path, device, normalize, max_length)
        self.margin = margin
    
    def compute_infoNCE_loss(
        self,
        query_emb: torch.Tensor,
        pos_embs: torch.Tensor,
        neg_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 InfoNCE 对比损失
        
        Args:
            query_emb: 查询向量, shape: [batch, dim]
            pos_embs: 正样本向量, shape: [batch, dim]
            neg_embs: 负样本向量, shape: [batch, num_neg, dim]
        
        Returns:
            损失值
        """
        import torch.nn.functional as F
        
        # 合并所有样本
        all_embs = torch.cat([pos_embs, neg_embs.view(pos_embs.size(0), -1, pos_embs.size(-1))], dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(query_emb, all_embs.transpose(-2, -1))
        
        # InfoNCE 损失
        logits = sim_matrix  # [batch, 1 + num_neg]
        labels = torch.zeros(query_emb.size(0), dtype=torch.long, device=query_emb.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def compute_triplet_loss(
        self,
        query_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        计算三元组损失 (Margin-based Triplet Loss)
        
        Args:
            query_emb: 查询向量
            pos_emb: 正样本向量
            neg_emb: 负样本向量
        
        Returns:
            损失值
        """
        import torch.nn.functional as F
        
        # 计算距离
        pos_dist = F.pairwise_distance(query_emb, pos_emb)
        neg_dist = F.pairwise_distance(query_emb, neg_emb)
        
        # Triplet loss with margin
        losses = F.relu(pos_dist - neg_dist + self.margin)
        
        return losses.mean()


# 添加缺失的 import
import torch.nn.functional as F


def load_embedder(
    model_path: str = "BAAI/bge-small-zh-v1.5",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    with_hard_negatives: bool = False,
    **kwargs
) -> Union[Qwen3Embedder, Qwen3EmbeddingWithHardNegatives]:
    """
    工厂函数：加载 Qwen3-Embedding 编码器
    
    Args:
        model_path: 模型路径
        device: 设备
        with_hard_negatives: 是否支持难负样本
    
    Returns:
        编码器实例
    """
    if with_hard_negatives:
        return Qwen3EmbeddingWithHardNegatives(model_path, device, **kwargs)
    else:
        return Qwen3Embedder(model_path, device, **kwargs)


# ==================== 示例用法 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("Qwen3-Embedding 向量化模型测试")
    print("=" * 60)
    
    # 加载模型
    embedder = load_embedder(
        model_path="BAAI/bge-small-zh-v1.5",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 测试编码
    test_queries = [
        "石英的化学成分是什么？",
        "黄铁矿的硬度是多少？",
        "方解石的光学特性"
    ]
    
    # 编码查询
    query_embeddings = embedder.encode(test_queries)
    print(f"\n查询向量 shape: {query_embeddings.shape}")
    
    # 编码文档
    corpus = [
        {"text": "石英是一种主要造岩矿物，主要成分是二氧化硅(SiO2)，属于三方晶系。"},
        {"text": "黄铁矿(FeS2)是自然界常见的金属硫化物矿物，硬度为6-6.5。"},
        {"text": "方解石(CaCO3)是碳酸盐矿物，在紫外光下具有荧光性。"}
    ]
    
    doc_embeddings = embedder.encode_corpus(corpus)
    print(f"文档向量 shape: {doc_embeddings.shape}")
    
    # 计算相似度
    similarities = embedder.compute_similarity(query_embeddings[0], doc_embeddings)
    print(f"\n第一个查询与各文档的相似度: {similarities}")
    
    print("\n✅ 测试完成!")
