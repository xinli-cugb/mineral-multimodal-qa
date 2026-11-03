"""
Qwen3-Reranker 交叉编码器模块
对混合检索的 Top-10 结果进行深度交互打分，重新排序截取 Top-3
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 数据结构 ====================

@dataclass
class RerankResult:
    """
    重排结果数据类
    """
    id: Union[int, str]
    text: str
    score: float
    rank: int
    original_rank: int
    source: str = ""
    url: str = ""
    title: str = ""
    mineral_type: str = ""
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "rank": self.rank,
            "original_rank": self.original_rank,
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "mineral_type": self.mineral_type
        }


# ==================== Reranker 模型 ====================

DEFAULT_INSTRUCT = "给定一个矿物学问题，判断文档是否包含回答该问题所需的信息。"
THINK_BLOCK = "<think>\n\n</think>\n\n"


class Qwen3Reranker:
    """
    Qwen3-Reranker 交叉编码器
    通过深度交互计算 Query-Document 相关性分数
    
    支持 HuggingFace 和 ModelScope 下载
    """
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Reranker-8x7b",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
        batch_size: int = 8,
        normalize: bool = True,
        instruction: Optional[str] = None
    ):
        """
        初始化 Qwen3-Reranker
        
        Args:
            model_path: 模型路径或 HuggingFace Hub 名称
            device: 运行设备
            max_length: 最大输入长度
            batch_size: 批次大小
            normalize: 是否对分数进行归一化
            instruction: 生成式 yes/no 打分时的任务指令
        """
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize = normalize
        self.model_path = model_path
        self.instruction = instruction or DEFAULT_INSTRUCT
        self.yes_no_mode = False
        self.yes_token_id = None
        self.no_token_id = None
        
        # 加载模型和分词器
        self._load_model(model_path)
        
        print(f"✅ Reranker 模型加载成功")
        print(f"   模型路径: {model_path}")
        print(f"   设备: {self.device}")
    
    def _load_model(self, model_path: str):
        """加载模型和分词器

        Qwen3-Reranker 微调后的 checkpoint 是 CausalLM 结构，需要用官方
        yes/no 生成式打分；旧式分类头模型仍走 SequenceClassification。
        """
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        architectures = " ".join(getattr(config, "architectures", []) or [])
        model_type = getattr(config, "model_type", "")
        self.yes_no_mode = "ForCausalLM" in architectures or model_type in ("qwen3", "qwen2")

        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        if self.yes_no_mode:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype
            )
            # 左侧 padding，保证 batch 中最后一个位置是真实文本的末 token
            self.tokenizer.padding_side = "left"
            self.yes_token_id = getattr(config, "yes_token_id", None)
            self.no_token_id = getattr(config, "no_token_id", None)
            if self.yes_token_id is None:
                self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
            if self.no_token_id is None:
                self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")
        else:
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype
                )
            except Exception:
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype
                    )
                except Exception:
                    self.model = AutoModel.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype
                    )

        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def score(
        self,
        query: str,
        documents: List[str],
        batch_size: Optional[int] = None
    ) -> List[float]:
        """
        计算查询与文档的相关性分数
        
        Args:
            query: 查询文本
            documents: 文档列表
            batch_size: 批次大小
        
        Returns:
            相关性分数列表
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        all_scores = []
        
        # 分批处理
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            # 构建输入
            inputs = self._build_inputs(query, batch_docs)
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 前向传播
            outputs = self.model(**inputs)
            
            # 获取分数
            scores = self._compute_scores(outputs, len(batch_docs))
            
            all_scores.extend(scores)
        
        # 归一化（yes/no 概率自带校准，不再做 min-max）
        if self.normalize and not self.yes_no_mode and all_scores:
            min_score = min(all_scores)
            max_score = max(all_scores)
            if max_score > min_score:
                all_scores = [(s - min_score) / (max_score - min_score) for s in all_scores]
        
        return all_scores
    
    def _build_inputs(
        self,
        query: str,
        documents: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        构建模型输入
        
        Args:
            query: 查询文本
            documents: 文档列表
        
        Returns:
            模型输入字典
        """
        if self.yes_no_mode:
            prefix = (
                "<|im_start|>system\n"
                "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
                'Note that the answer can only be "yes" or "no".<|im_end|>\n'
                "<|im_start|>user\n"
            )
            texts = [
                prefix
                + f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"
                + f"<|im_end|>\n<|im_start|>assistant\n{THINK_BLOCK}"
                for doc in documents
            ]
            return self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
                return_tensors="pt"
            )

        # 构建 [Query, Doc] 对
        pairs = [[query, doc] for doc in documents]
        
        # Tokenize
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return encoded
    
    def _compute_scores(
        self,
        outputs,
        num_docs: int
    ) -> List[float]:
        """
        从模型输出计算分数
        
        Args:
            outputs: 模型输出
            num_docs: 文档数量
        
        Returns:
            分数列表
        """
        # Qwen3-Reranker 生成式打分：取最后位置的 yes / no logits 做二分类
        if self.yes_no_mode:
            logits = outputs.logits[:, -1, :].float()
            pair_logits = logits[:, [self.no_token_id, self.yes_token_id]]
            probs = torch.softmax(pair_logits, dim=-1)
            return probs[:, 1].cpu().tolist()

        # 旧式分类头输出 logits，形状 [batch_size, 2]
        # logits[:, 1] 是正类的分数（相关性）
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            # 取正类（索引1）的分数
            scores = logits[:, 1]
        else:
            # 备用方案
            scores = outputs.last_hidden_state[:, 0, :].squeeze(-1)
        
        # 转换为 Python 列表
        return scores.cpu().tolist()
    
    @torch.no_grad()
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 3,
        batch_size: Optional[int] = None
    ) -> List[RerankResult]:
        """
        对候选文档进行重排
        
        Args:
            query: 查询文本
            candidates: 候选文档列表 (来自混合检索的 Top-10)
            top_k: 返回 Top-K 结果
            batch_size: 批次大小
        
        Returns:
            重排后的结果列表
        """
        if not candidates:
            return []
        
        # 提取文档文本
        documents = [doc.get('text', '') for doc in candidates]
        
        # 计算相关性分数
        scores = self.score(query, documents, batch_size)
        
        # 构建结果
        results = []
        for rank, (doc, score) in enumerate(zip(candidates, scores), start=1):
            result = RerankResult(
                id=doc.get('id', rank),
                text=doc.get('text', ''),
                score=score,
                rank=0,  # 稍后重新赋值
                original_rank=rank,
                source=doc.get('source', ''),
                url=doc.get('url', ''),
                title=doc.get('title', ''),
                mineral_type=doc.get('mineral_type', '')
            )
            results.append(result)
        
        # 按分数降序排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        # 重新赋值 rank
        for rank, result in enumerate(results[:top_k], start=1):
            result.rank = rank
        
        logger.info(f"🎯 重排完成: {len(candidates)} → Top-{len(results[:top_k])}")
        
        return results[:top_k]
    
    @torch.no_grad()
    def compute_pairwise_scores(
        self,
        query: str,
        doc_pos: str,
        doc_neg: str
    ) -> Tuple[float, float]:
        """
        计算成对分数 (用于训练)
        
        Args:
            query: 查询文本
            doc_pos: 正样本文档
            doc_neg: 负样本文档
        
        Returns:
            (正样本分数, 负样本分数)
        """
        scores = self.score(query, [doc_pos, doc_neg])
        
        return scores[0], scores[1]
    
    def get_config(self) -> Dict:
        """获取模型配置"""
        return {
            "model_path": self.model.name_or_path if hasattr(self.model, 'name_or_path') else "unknown",
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "normalize": self.normalize
        }


# ==================== 简化的 Cross-Encoder 实现 ====================

class CrossEncoderReranker:
    """
    通用的 Cross-Encoder 重排器
    兼容 HuggingFace Cross-Encoder 模型
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-8x7b",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
        batch_size: int = 4
    ):
        """
        初始化 Cross-Encoder 重排器
        
        Args:
            model_name: Cross-Encoder 模型名称
            device: 运行设备
            max_length: 最大输入长度
            batch_size: 批次大小
        """
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        
        # 加载模型
        try:
            from sentence_transformers import CrossEncoder
            
            self.model = CrossEncoder(
                model_name,
                max_length=max_length,
                device=device
            )
            
            logger.info(f"✅ Cross-Encoder 加载成功: {model_name}")
            
        except ImportError:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")
    
    def score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        计算查询-文档对的相关性分数
        
        Args:
            pairs: [(query, document), ...] 列表
        
        Returns:
            分数列表
        """
        scores = self.model.predict(pairs)
        
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        
        return scores
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 3
    ) -> List[RerankResult]:
        """
        对候选文档进行重排
        
        Args:
            query: 查询文本
            candidates: 候选文档列表
            top_k: 返回 Top-K
        
        Returns:
            重排结果
        """
        # 构建查询-文档对
        documents = [doc.get('text', '') for doc in candidates]
        pairs = [(query, doc) for doc in documents]
        
        # 批量计算分数
        scores = self.score(pairs)
        
        # 构建结果
        results = []
        for rank, (doc, score) in enumerate(zip(candidates, scores), start=1):
            result = RerankResult(
                id=doc.get('id', rank),
                text=doc.get('text', ''),
                score=float(score),
                rank=0,
                original_rank=rank,
                source=doc.get('source', ''),
                url=doc.get('url', ''),
                title=doc.get('title', ''),
                mineral_type=doc.get('mineral_type', '')
            )
            results.append(result)
        
        # 排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        # 重新赋值 rank
        for rank, result in enumerate(results[:top_k], start=1):
            result.rank = rank
        
        return results[:top_k]


# ==================== Reranker Pipeline ====================

class RerankerPipeline:
    """
    重排 Pipeline
    封装检索结果的重排流程
    """
    
    def __init__(
        self,
        reranker,
        top_k: int = 3,
        score_threshold: Optional[float] = None
    ):
        """
        初始化重排 Pipeline
        
        Args:
            reranker: 重排模型实例
            top_k: 返回 Top-K 结果
            score_threshold: 分数阈值，低于此分数的结果将被过滤
        """
        self.reranker = reranker
        self.top_k = top_k
        self.score_threshold = score_threshold
    
    def process(
        self,
        query: str,
        candidates: List[Dict]
    ) -> List[RerankResult]:
        """
        处理重排流程
        
        Args:
            query: 查询文本
            candidates: 候选文档列表
        
        Returns:
            重排后的结果
        """
        if not candidates:
            return []
        
        # 限制输入数量（避免过长）
        input_candidates = candidates[:20]
        
        # 重排
        results = self.reranker.rerank(
            query=query,
            candidates=input_candidates,
            top_k=self.top_k
        )
        
        # 分数过滤
        if self.score_threshold is not None:
            results = [r for r in results if r.score >= self.score_threshold]
        
        return results
    
    def process_with_details(
        self,
        query: str,
        candidates: List[Dict],
        original_query: str = None,
        visual_anchors: List[Dict] = None
    ) -> Dict:
        """
        带详细信息的处理
        
        Args:
            query: 查询文本
            candidates: 候选文档
            original_query: 原始查询
            visual_anchors: 视觉锚点
        
        Returns:
            包含详细信息的字典
        """
        results = self.process(query, candidates)
        
        return {
            "query": query,
            "original_query": original_query,
            "visual_anchors": visual_anchors,
            "input_candidates": len(candidates),
            "output_results": len(results),
            "results": [r.to_dict() for r in results]
        }


# ==================== 工厂函数 ====================

def load_reranker(
    model_type: str = "qwen3",
    model_path: str = "Qwen/Qwen3-Reranker-0.5B",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs
):
    """
    加载重排模型
    
    Args:
        model_type: 模型类型 ('qwen3' 或 'cross_encoder')
        model_path: 模型路径
        device: 设备
        **kwargs: 其他参数
    
    Returns:
        重排模型实例
    """
    if model_type == "qwen3":
        return Qwen3Reranker(
            model_path=model_path,
            device=device,
            **kwargs
        )
    elif model_type == "cross_encoder":
        return CrossEncoderReranker(
            model_name=model_path,
            device=device,
            **kwargs
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")


def create_reranker_pipeline(
    reranker,
    top_k: int = 3,
    score_threshold: float = None
) -> RerankerPipeline:
    """
    创建重排 Pipeline
    
    Args:
        reranker: 重排模型
        top_k: Top-K
        score_threshold: 分数阈值
    
    Returns:
        RerankerPipeline 实例
    """
    return RerankerPipeline(
        reranker=reranker,
        top_k=top_k,
        score_threshold=score_threshold
    )


# ==================== 示例用法 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Qwen3-Reranker 重排模块测试")
    print("=" * 60)
    
    # 注意: 实际使用时请替换为真实的模型路径
    # 这里演示使用 BGE-Reranker 作为替代
    
    try:
        # 使用 Cross-Encoder 作为重排器
        reranker = load_reranker(
            model_type="cross_encoder",
            model_path="Qwen/Qwen3-Reranker-8x7b",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # 测试候选文档
        candidates = [
            {"id": 1, "text": "石英是一种主要造岩矿物，化学成分为二氧化硅(SiO2)，属于三方晶系。", "source": "wiki"},
            {"id": 2, "text": "黄铁矿(FeS2)是自然界常见的金属硫化物矿物，硬度为6-6.5。", "source": "wiki"},
            {"id": 3, "text": "方解石(CaCO3)是碳酸盐矿物，在紫外光下具有荧光性。", "source": "baike"},
            {"id": 4, "text": "长石是地壳中最常见的矿物之一，主要分为钾长石和钠长石。", "source": "wiki"},
            {"id": 5, "text": "云母是一种硅酸盐矿物，具有良好的绝缘性能。", "source": "baike"},
        ]
        
        query = "石英的化学成分是什么？"
        
        # 重排
        results = reranker.rerank(query, candidates, top_k=3)
        
        print(f"\n🔍 查询: {query}")
        print(f"\n📊 重排结果 (Top-3):")
        for r in results:
            print(f"  Rank {r.rank} (原 Rank {r.original_rank}):")
            print(f"    分数: {r.score:.4f}")
            print(f"    文本: {r.text[:50]}...")
        
        # 创建 Pipeline
        pipeline = create_reranker_pipeline(reranker, top_k=3)
        
        # Pipeline 处理
        output = pipeline.process_with_details(
            query=query,
            candidates=candidates,
            original_query=query
        )
        
        print(f"\n📋 Pipeline 输出:")
        print(f"  输入候选数: {output['input_candidates']}")
        print(f"  输出结果数: {output['output_results']}")
        
    except Exception as e:
        print(f"⚠️ 测试失败: {e}")
        print("   (请确保模型文件已下载，或安装必要的依赖)")
    
    print("\n✅ 测试完成!")
