"""
矿物问答系统 Pipeline
端到端的 V2.0 旗舰版流水线
串联 Swin Transformer、Hybrid Search、Reranker 和 LLM
"""
import os
import sys
import json
import time
from typing import List, Dict, Optional, Union, Tuple, Any, Generator
from dataclasses import dataclass, field
from pathlib import Path
import logging
from PIL import Image
import numpy as np

# 确保项目根目录在 Python 路径中
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 数据结构 ====================

@dataclass
class VisualAnchor:
    """
    视觉锚点
    """
    mineral_name: str
    confidence: float
    rank: int
    is_primary: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "mineral_name": self.mineral_name,
            "confidence": self.confidence,
            "rank": self.rank,
            "is_primary": self.is_primary
        }


@dataclass
class RetrievedDocument:
    """
    检索到的文档
    """
    id: Union[int, str]
    text: str
    score: float
    rank: int
    source: str = ""
    url: str = ""
    title: str = ""
    dense_score: float = 0.0
    sparse_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "rank": self.rank,
            "source": self.source,
            "url": self.url,
            "title": self.title
        }


@dataclass
class QAResponse:
    """
    问答系统响应
    """
    answer: str
    query: str
    visual_anchors: List[Dict] = field(default_factory=list)
    retrieved_docs: List[Dict] = field(default_factory=list)
    reranked_docs: List[Dict] = field(default_factory=list)
    generation_time: float = 0.0
    total_time: float = 0.0
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "query": self.query,
            "visual_anchors": self.visual_anchors,
            "retrieved_docs": self.retrieved_docs,
            "reranked_docs": self.reranked_docs,
            "generation_time": self.generation_time,
            "total_time": self.total_time,
            "model": self.model,
            "usage": self.usage
        }


# ==================== Pipeline 主类 ====================

def _default_embedder_path() -> str:
    """优先使用微调后的 Qwen3-Embedding"""
    candidates = [
        "/root/autodl-tmp/outputs/embedding",
        "/root/V2_0_Mineral_QA/model/Qwen3-Embedding-0.6B",
    ]
    for path in candidates:
        if Path(path, "config.json").exists():
            return path
    return candidates[-1]


def _default_faiss_index_path() -> str:
    """优先使用 6408 篇全量索引"""
    for path in ("./faiss_mineral_6408", "./faiss_mineral_45"):
        if Path(path, "index.faiss").exists():
            return path
    return "./faiss_mineral_45"


def _default_reranker_path() -> str:
    """优先使用微调后的 Qwen3-Reranker-0.6B"""
    candidates = [
        "/root/autodl-tmp/outputs/reranker",
        "/root/V2_0_Mineral_QA/model/Qwen3-Reranker-0.6B",
    ]
    for path in candidates:
        if Path(path, "config.json").exists():
            return path
    return candidates[-1]


class MineralQAPipeline:
    """
    矿物问答系统 Pipeline
    整合视觉感知、混合检索、精排和 LLM 生成
    """
    
    def __init__(
        self,
        # Swin 模型配置
        swin_model_path: str = "/root/rag/models/checkpoints/swin_transformer/best_model_acc7886_ep10.pth",
        swin_id2label_path: str = "/root/V2_0_Mineral_QA/model/id2label.json",
        # Embedding 模型配置
        embedder_model_path: str = _default_embedder_path(),
        embedder_dimension: int = 1024,
        # Milvus 配置
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        milvus_collection: str = "mineral_knowledge",
        # Reranker 配置
        reranker_model_path: str = _default_reranker_path(),
        # LLM 配置
        llm_model_path: str = "/root/V2_0_Mineral_QA/model/Qwen2.5-7B-Instruct",
        llm_api_base: str = "http://localhost:8000/v1",
        # 其他配置
        device: str = "cuda",
        top_k_retrieval: int = 10,
        top_k_rerank: int = 3,
        enable_visual: bool = True,
        enable_rerank: bool = True,
        enable_bm25: bool = True,
        enable_query_fusion: bool = True,
        bm25_corpus_path: Optional[str] = None
    ):
        """
        初始化 Pipeline
        
        Args:
            swin_model_path: Swin 模型路径
            swin_id2label_path: Swin 标签映射文件
            embedder_model_path: Embedding 模型路径
            embedder_dimension: Embedding 向量维度
            milvus_host: Milvus 主机地址
            milvus_port: Milvus 端口
            milvus_collection: Milvus 集合名称
            reranker_model_path: Reranker 模型路径
            llm_model_path: LLM 模型路径
            llm_api_base: LLM API 地址
            device: 运行设备
            top_k_retrieval: 检索 Top-K
            top_k_rerank: 重排 Top-K
            enable_visual: 是否启用视觉模块
            enable_rerank: 是否启用重排模块
        """
        self.config = {
            "swin_model_path": swin_model_path,
            "swin_id2label_path": swin_id2label_path,
            "embedder_model_path": embedder_model_path,
            "embedder_dimension": embedder_dimension,
            "milvus_host": milvus_host,
            "milvus_port": milvus_port,
            "milvus_collection": milvus_collection,
            "reranker_model_path": reranker_model_path,
            "llm_model_path": llm_model_path,
            "llm_api_base": llm_api_base,
            "device": device,
            "top_k_retrieval": top_k_retrieval,
            "top_k_rerank": top_k_rerank,
            "enable_visual": enable_visual,
            "enable_rerank": enable_rerank,
            "enable_bm25": enable_bm25,
            "enable_query_fusion": enable_query_fusion,
            "bm25_corpus_path": bm25_corpus_path,
            "faiss_index_path": _default_faiss_index_path()
        }
        
        # 各模块实例
        self.swin_model = None
        self.embedder = None
        self.milvus_client = None
        self.hybrid_searcher = None
        self.reranker = None
        self.llm_client = None
        self.bm25_index = None
        self.query_fuser = None
        self.prompt_template = None
        
        # 初始化状态
        self.initialized = False
        
        logger.info("📦 Pipeline 配置完成")
    
    def initialize(self):
        """
        初始化所有模块
        """
        logger.info("🚀 开始初始化 Pipeline...")
        start_time = time.time()
        
        # 1. 初始化视觉模型 (Swin)
        if self.config["enable_visual"]:
            logger.info("📷 初始化 Swin Transformer...")
            try:
                from vision.swin_model import MineralSwinClassifier
                self.swin_model = MineralSwinClassifier(
                    model_path=self.config["swin_model_path"],
                    id2label_path=self.config["swin_id2label_path"],
                    device=self.config["device"]
                )
                logger.info("   ✅ Swin 模型加载成功")
            except Exception as e:
                logger.warning(f"   ⚠️ Swin 模型加载失败: {e}")
                self.config["enable_visual"] = False
        
        # 2. 初始化 Embedding 模型
        logger.info("🔢 初始化 Qwen3-Embedding...")
        try:
            from retrieval.qwen3_embedder import load_embedder
            self.embedder = load_embedder(
                model_path=self.config["embedder_model_path"],
                device=self.config["device"]
            )
            logger.info("   ✅ Embedding 模型加载成功")
        except Exception as e:
            logger.error(f"   ❌ Embedding 模型加载失败: {e}")
            raise
        
        # 3. 初始化向量数据库客户端 (优先 FAISS，回退 Milvus)
        self.faiss_client = None
        self.milvus_client = None

        # 尝试 FAISS
        logger.info("🗄️ 初始化 FAISS 客户端...")
        try:
            from retrieval.faiss_client import FAISSClient
            faiss_path = self.config.get("faiss_index_path", "./faiss_mineral_45")
            self.faiss_client = FAISSClient(index_path=faiss_path, dimension=1024)
            self.faiss_client.load()
            if self.faiss_client.is_loaded:
                logger.info(f"   ✅ FAISS 索引加载成功: {faiss_path}")
            else:
                logger.warning(f"   ⚠️ FAISS 索引未找到: {faiss_path}")
                self.faiss_client = None
        except Exception as e:
            logger.warning(f"   ⚠️ FAISS 加载失败: {e}")
            self.faiss_client = None

        # 如果 FAISS 不可用，尝试 Milvus
        if not self.faiss_client:
            logger.info("🗄️ 初始化 Milvus 客户端...")
            try:
                from retrieval.milvus_client import MilvusClient
                self.milvus_client = MilvusClient(
                    uri="./milvus_lite.db",
                    host=self.config["milvus_host"],
                    port=self.config["milvus_port"],
                    collection_name=self.config["milvus_collection"],
                    use_lite=True
                )
                mode = "Milvus Lite" if self.milvus_client.is_lite else "Milvus 服务器"
                logger.info(f"   ✅ {mode} 客户端连接成功")
            except Exception as e:
                logger.warning(f"   ⚠️ Milvus 连接失败: {e}")

        # 3.5 构建 BM25 稀疏索引（与稠密索引共用同一批文档，保证 id 对齐）
        self.bm25_index = None
        if self.config["enable_bm25"]:
            logger.info("🧮 构建 BM25 稀疏索引...")
            try:
                from retrieval.bm25_search import build_bm25_from_pipeline_sources
                faiss_data = getattr(self.faiss_client, "data", None) if self.faiss_client else None
                self.bm25_index = build_bm25_from_pipeline_sources(
                    faiss_data=faiss_data,
                    corpus_path=self.config.get("bm25_corpus_path"),
                )
                if self.bm25_index:
                    logger.info(f"   ✅ BM25 索引就绪: {len(self.bm25_index)} 篇")
                else:
                    logger.warning("   ⚠️ 未找到 BM25 语料，稀疏通道关闭")
            except Exception as e:
                logger.warning(f"   ⚠️ BM25 索引构建失败: {e}")
                self.bm25_index = None

        # 4. 初始化 Reranker (在 HybridSearcher 之前，因为可能需要)
        self.reranker = None
        self.reranker_pipeline = None
        if self.config["enable_rerank"]:
            logger.info("🎯 初始化 Qwen3-Reranker...")
            try:
                from retrieval.reranker import load_reranker, create_reranker_pipeline
                self.reranker = load_reranker(
                    model_type="qwen3",
                    model_path=self.config["reranker_model_path"],
                    device=self.config["device"]
                )
                self.reranker_pipeline = create_reranker_pipeline(
                    self.reranker,
                    top_k=self.config["top_k_rerank"]
                )
                logger.info("   ✅ Reranker 加载成功")
            except Exception as e:
                logger.warning(f"   ⚠️ Reranker 加载失败: {e}")
                self.config["enable_rerank"] = False

        # 5. 初始化检索器 (优先 FAISS，回退 HybridSearcher+Milvus)
        self.hybrid_searcher = None
        if self.faiss_client:
            logger.info("🔍 使用 FAISS 向量检索...")
            self.hybrid_searcher = None  # FAISS 直接在 pipeline 中使用
        elif self.milvus_client:
            logger.info("🔍 初始化 HybridSearcher...")
            try:
                from retrieval.hybrid_search import HybridSearcher
                self.hybrid_searcher = HybridSearcher(
                    embedder=self.embedder,
                    milvus_client=self.milvus_client,
                    bm25_index=self.bm25_index,
                    top_k=self.config["top_k_retrieval"]
                )
                logger.info("   ✅ HybridSearcher 初始化成功")
            except Exception as e:
                logger.warning(f"   ⚠️ HybridSearcher 初始化失败: {e}")
                self.hybrid_searcher = None

        # 6. 初始化 LLM 客户端 (直接加载模式)
        logger.info("🤖 初始化 LLM 客户端...")
        try:
            from generator.llm_client import create_llm_client
            self.llm_client = create_llm_client(
                model_type="qwen2.5",
                model_path=self.config["llm_model_path"],
                use_direct=True,
                device=self.config["device"]
            )
            logger.info("   ✅ LLM 客户端初始化成功")
        except Exception as e:
            logger.error(f"   ❌ LLM 客户端初始化失败: {e}")
            raise
        
        # 6.5 初始化查询融合改写器（论文 §2.2.2）
        if self.config["enable_query_fusion"]:
            try:
                from generator.query_fusion import QueryFuser
                self.query_fuser = QueryFuser(self.llm_client, max_anchors=3)
                logger.info("   ✅ 查询融合器初始化成功")
            except Exception as e:
                logger.warning(f"   ⚠️ 查询融合器初始化失败，将退化为拼接: {e}")
                self.query_fuser = None

        # 7. 初始化 Prompt 模板
        from generator.prompts import create_prompt_template, VisualAnchorPrompt, ContextDocument
        self.prompt_template = create_prompt_template(
            include_visual=self.config["enable_visual"]
        )
        
        self.initialized = True
        
        init_time = time.time() - start_time
        logger.info(f"✅ Pipeline 初始化完成，耗时: {init_time:.2f}s")
    
    def run(
        self,
        image: Union[str, Image.Image, None] = None,
        text_query: str = None,
        return_details: bool = True,
        stream: bool = False,
        **llm_kwargs
    ) -> Union[QAResponse, Generator[str, None, None]]:
        """
        运行 Pipeline
        
        Args:
            image: 输入图像 (路径、PIL Image 或 None)
            text_query: 文本问题
            return_details: 是否返回详细信息
            stream: 是否流式输出
            **llm_kwargs: LLM 生成参数
        
        Returns:
            QAResponse 或流式生成器
        """
        if not self.initialized:
            self.initialize()
        
        total_start = time.time()
        
        # 1. 验证输入
        if not text_query and not image:
            raise ValueError("必须提供 image 或 text_query")
        
        if not text_query:
            text_query = "请描述这个矿物标本"
        
        logger.info(f"🔍 处理查询: {text_query}")
        if image:
            logger.info(f"🖼️  收到图像输入")
        
        # 2. 视觉感知 (Swin)
        visual_anchors = []
        if image and self.config["enable_visual"] and self.swin_model:
            logger.info("📷 执行视觉识别...")
            visual_start = time.time()

            # 确保图像为 PIL Image
            if isinstance(image, str):
                image = Image.open(image)
            elif hasattr(image, 'read'):  # Streamlit UploadedFile
                image = Image.open(image)

            # 获取视觉锚点
            anchors_result = self.swin_model.get_visual_anchors(image, confidence_threshold=0.01)
            visual_anchors = anchors_result.get("visual_anchors", [])
            
            logger.info(f"   识别到 {len(visual_anchors)} 个候选矿物")
            for anchor in visual_anchors:
                logger.info(f"   - {anchor['mineral_name']}: {anchor['confidence']:.2%}")
        
        # 3. 查询融合改写（视觉锚点 + 用户问题 → 检索查询，论文 §2.2.2）
        fused_query = text_query
        if self.query_fuser:
            fusion = self.query_fuser.fuse(text_query, visual_anchors)
            fused_query = fusion.fused_query
            logger.info(
                f"🧩 融合查询: {fused_query} "
                f"(intent={fusion.intent}, llm={fusion.used_llm}, "
                f"clarify={fusion.needs_clarification})"
            )

        # 4. 双路检索：Dense + BM25 → RRF
        retrieved_docs = []
        if self.hybrid_searcher or self.faiss_client:
            logger.info("🔍 执行混合检索...")
            retrieval_start = time.time()
            
            # 转换视觉锚点格式
            anchor_objects = None
            if visual_anchors:
                from generator.prompts import VisualAnchorPrompt
                anchor_objects = [
                    VisualAnchorPrompt(
                        mineral_name=a["mineral_name"],
                        confidence=a["confidence"],
                        rank=a["rank"]
                    ) for a in visual_anchors
                ]
            
            # 执行检索
            if self.hybrid_searcher:
                results = self.hybrid_searcher.search(
                    query=fused_query,
                    visual_anchors=None,
                    top_k=self.config["top_k_retrieval"]
                )
                retrieved_docs = [r.to_dict() for r in results]
            elif self.faiss_client:
                # 使用 FAISS 检索
                retrieval_start = time.time()
                query_vec = self.embedder.encode_queries([fused_query])
                faiss_results = self.faiss_client.search(
                    query_vector=query_vec[0],
                    top_k=self.config["top_k_retrieval"]
                )
                retrieved_docs = []
                for r in faiss_results:
                    doc = {
                        "id": r.get("id"),
                        "name": r.get("name", ""),
                        "english_name": r.get("english_name", ""),
                        "text": r.get("text", ""),
                        "score": 1.0 / (1.0 + r.get("distance", 0))
                    }
                    retrieved_docs.append(doc)
                retrieval_time = time.time() - retrieval_start
                logger.info(f"   FAISS 检索完成, 耗时: {retrieval_time:.2f}s")
                logger.info(f"   FAISS 检索结果:")
                for i, doc in enumerate(retrieved_docs[:5]):
                    logger.info(f"      {i+1}. {doc.get('name', '')} ({doc.get('english_name', '')}) - score: {doc.get('score', 0):.4f}")

                # BM25 稀疏通道 + RRF 融合
                sparse_results = []
                if self.bm25_index:
                    sparse_results = self.bm25_index.search(
                        fused_query,
                        top_k=self.config["top_k_retrieval"] * 2
                    )
                if sparse_results:
                    from retrieval.hybrid_search import rrf_fusion
                    fused_results = rrf_fusion(
                        retrieved_docs,
                        sparse_results,
                        top_k=self.config["top_k_retrieval"],
                        k=60
                    )
                    retrieved_docs = [r.to_dict() for r in fused_results]
                logger.info(
                    f"   Dense={len(faiss_results)} / BM25={len(sparse_results)} "
                    f"→ RRF Top-{len(retrieved_docs)}"
                )
            else:
                retrieved_docs = []

            logger.info(f"   检索到 {len(retrieved_docs)} 条候选文档")
        else:
            if self.faiss_client:
                logger.info("🔍 使用 FAISS 向量检索")
            else:
                logger.warning("⚠️ 混合检索器未初始化")
        
        # 4. 重排 (Reranker)
        reranked_docs = []
        if retrieved_docs and self.config["enable_rerank"] and hasattr(self, 'reranker_pipeline'):
            logger.info("🎯 执行重排...")
            rerank_start = time.time()
            
            # 重排
            rerank_results = self.reranker_pipeline.process(
                query=fused_query,
                candidates=retrieved_docs
            )
            
            reranked_docs = [r.to_dict() for r in rerank_results]
            logger.info(f"   重排后 Top-{len(reranked_docs)} 文档")
        
        # 如果没有重排，直接使用检索结果
        if not reranked_docs:
            reranked_docs = retrieved_docs[:self.config["top_k_rerank"]]
        
        # 5. 构建 Prompt
        logger.info("📝 构建 Prompt...")
        
        # 转换数据格式
        anchor_prompts = None
        if visual_anchors:
            from generator.prompts import VisualAnchorPrompt
            anchor_prompts = [
                VisualAnchorPrompt(
                    mineral_name=a["mineral_name"],
                    confidence=a["confidence"],
                    rank=a["rank"]
                ) for a in visual_anchors
            ]
        
        context_prompts = None
        if reranked_docs:
            from generator.prompts import ContextDocument
            context_prompts = [
                ContextDocument(
                    text=d["text"],
                    score=d["score"],
                    rank=d["rank"],
                    source=d.get("source", ""),
                    title=d.get("title", "")
                ) for d in reranked_docs
            ]
        
        messages = self.prompt_template.build(
            query=text_query,
            visual_anchors=anchor_prompts,
            context_documents=context_prompts
        )
        
        # 6. LLM 生成
        logger.info("🤖 调用 LLM 生成回答...")
        generation_start = time.time()
        
        try:
            if stream:
                # 流式输出
                return self._stream_response(
                    messages=messages,
                    total_start=total_start,
                    visual_anchors=visual_anchors,
                    retrieved_docs=retrieved_docs,
                    reranked_docs=reranked_docs,
                    **llm_kwargs
                )
            else:
                # 非流式输出
                response = self.llm_client.chat(messages, stream=False, **llm_kwargs)
                
                generation_time = time.time() - generation_start
                total_time = time.time() - total_start
                
                answer = response.content if hasattr(response, 'content') else str(response)
                usage = response.usage if hasattr(response, 'usage') else {}
                
                result = QAResponse(
                    answer=answer,
                    query=text_query,
                    visual_anchors=visual_anchors,
                    retrieved_docs=retrieved_docs if return_details else [],
                    reranked_docs=reranked_docs if return_details else [],
                    generation_time=generation_time,
                    total_time=total_time,
                    model=self.config["llm_model_path"],
                    usage=usage
                )
                
                logger.info(f"✅ 生成完成，耗时: {generation_time:.2f}s")
                
                return result
                
        except Exception as e:
            logger.error(f"❌ LLM 生成失败: {e}")
            raise
    
    def _stream_response(
        self,
        messages: List[Dict],
        total_start: float,
        visual_anchors: List[Dict],
        retrieved_docs: List[Dict],
        reranked_docs: List[Dict],
        **llm_kwargs
    ) -> Generator[str, None, None]:
        """
        流式输出
        """
        response = self.llm_client.chat(messages, stream=True, **llm_kwargs)
        
        full_answer = ""
        generation_start = time.time()
        
        for chunk in response:
            full_answer += chunk
            yield chunk
        
        generation_time = time.time() - generation_start
        total_time = time.time() - total_start
    
    def run_simple(
        self,
        text_query: str,
        image_path: str = None,
        **kwargs
    ) -> str:
        """
        简单接口：只返回回答文本
        
        Args:
            text_query: 文本问题
            image_path: 图像路径
        
        Returns:
            回答文本
        """
        result = self.run(
            image=image_path,
            text_query=text_query,
            return_details=False,
            **kwargs
        )
        
        if hasattr(result, 'answer'):
            return result.answer
        else:
            return str(result)


# ==================== 工厂函数 ====================

def create_pipeline(config: Dict = None, **kwargs) -> MineralQAPipeline:
    """
    创建 Pipeline
    
    Args:
        config: 配置字典
        **kwargs: 额外配置
    
    Returns:
        MineralQAPipeline 实例
    """
    if config is None:
        config = {}
    
    config.update(kwargs)
    
    pipeline = MineralQAPipeline(**config)
    return pipeline


def load_pipeline_from_config(config_path: str) -> MineralQAPipeline:
    """
    从配置文件加载 Pipeline
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        MineralQAPipeline 实例
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return create_pipeline(config)


# ==================== 示例用法 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("矿物问答系统 Pipeline 测试")
    print("=" * 60)
    
    # 方式1: 使用默认配置
    print("\n📦 创建 Pipeline...")
    pipeline = create_pipeline(
        swin_model_path="/root/rag/models/checkpoints/swin_transformer/best_model_acc7886_ep10.pth",
        embedder_model_path="/root/V2_0_Mineral_QA/model/Qwen3-Embedding-0.6B",
        reranker_model_path=_default_reranker_path(),
        llm_model_path="/root/V2_0_Mineral_QA/model/Qwen2.5-7B-Instruct",
        device="cuda",
        enable_visual=False,  # 测试时关闭视觉模块
        enable_rerank=False    # 测试时关闭重排
    )
    
    # 注意: 实际运行时需要确保各服务已启动
    # 1. Milvus 服务
    # 2. LLM API 服务 (如 vLLM)
    
    print("\n⚠️ 注意: 完整测试需要:")
    print("   1. 启动 Milvus 服务")
    print("   2. 启动 LLM API 服务")
    print("   3. 确保模型文件已下载")
    
    print("\n✅ Pipeline 创建完成!")
    
    # 示例调用（需要服务启动后才能运行）
    # result = pipeline.run(
    #     text_query="石英的化学成分是什么？",
    #     image_path="./mineral_sample.jpg"
    # )
    # print(f"\n📝 回答: {result.answer}")
