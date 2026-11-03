"""
训练数据准备脚本
用于生成 Qwen3-Embedding 和 Qwen3-Reranker 的训练数据
支持多种数据格式和负样本采样策略
"""
import json
import random
from typing import List, Dict, Optional
from pathlib import Path
import logging
import argparse
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """训练数据生成器"""
    
    def __init__(
        self,
        data_path: str,
        output_dir: str = "./training_data",
        seed: int = 42
    ):
        """
        初始化数据生成器
        
        Args:
            data_path: 原始数据路径
            output_dir: 输出目录
            seed: 随机种子
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        random.seed(seed)
        
        # 加载原始数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        logger.info(f"加载原始数据: {len(self.raw_data)} 条记录")
    
    def generate_embedding_training_data(
        self,
        output_file: str = "embedding_train.json",
        num_negatives: int = 7,
        query_types: List[str] = None,
        hard_negative_ratio: float = 0.3
    ) -> str:
        """
        生成 Embedding 训练数据
        
        Args:
            output_file: 输出文件名
            num_negatives: 每个正样本的负样本数量
            query_types: 查询类型列表
            hard_negative_ratio: 难负样本比例
        
        Returns:
            输出文件路径
        """
        if query_types is None:
            query_types = [
                "definition",      # 定义类查询
                "composition",     # 成分类查询
                "property",        # 性质类查询
                "usage",           # 用途类查询
                "identification",  # 鉴别类查询
                "origin"           # 成因类查询
            ]
        
        training_data = []
        all_docs = []
        
        # 收集所有文档
        for item in self.raw_data:
            doc = self._create_document(item)
            all_docs.append(doc)
        
        # 生成训练样本
        for item in tqdm(self.raw_data, desc="生成 Embedding 训练数据"):
            name = item.get('name', '')
            english_name = item.get('english_name', '')
            text = item.get('text', '')
            
            # 正样本
            positive = self._create_document(item)
            
            # 生成查询
            queries = self._generate_queries(name, english_name, text, query_types)
            
            # 为每个查询创建样本
            for query in queries:
                # 采样负样本
                negatives = self._sample_negatives(
                    all_docs=all_docs,
                    exclude_name=name,
                    num_negatives=num_negatives,
                    hard_negative_ratio=hard_negative_ratio,
                    query=query
                )
                
                training_data.append({
                    'query': query,
                    'positive': positive,
                    'negatives': negatives
                })
        
        # 保存
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Embedding 训练数据已保存: {output_path}")
        logger.info(f"  样本数: {len(training_data)}")
        
        return str(output_path)
    
    def generate_reranker_training_data(
        self,
        output_file: str = "reranker_train.json",
        num_negatives: int = 4,
        query_types: List[str] = None
    ) -> str:
        """
        生成 Reranker 训练数据
        
        Args:
            output_file: 输出文件名
            num_negatives: 每个正样本的负样本数量
            query_types: 查询类型列表
        
        Returns:
            输出文件路径
        """
        if query_types is None:
            query_types = [
                "definition",
                "composition",
                "property",
                "usage",
                "identification",
                "origin"
            ]
        
        training_data = []
        all_docs = []
        
        # 收集所有文档
        for item in self.raw_data:
            doc = self._create_document(item)
            all_docs.append(doc)
        
        # 生成训练样本
        for item in tqdm(self.raw_data, desc="生成 Reranker 训练数据"):
            name = item.get('name', '')
            english_name = item.get('english_name', '')
            text = item.get('text', '')
            
            # 正样本文档
            positive_doc = self._create_document(item)
            
            # 生成查询
            queries = self._generate_queries(name, english_name, text, query_types)
            
            # 为每个查询创建样本
            for query in queries:
                # 正样本
                training_data.append({
                    'query': query,
                    'doc': positive_doc,
                    'label': 1
                })
                
                # 负样本
                negatives = self._sample_negatives(
                    all_docs=all_docs,
                    exclude_name=name,
                    num_negatives=num_negatives,
                    hard_negative_ratio=0.3,
                    query=query
                )
                
                for neg_doc in negatives:
                    training_data.append({
                        'query': query,
                        'doc': neg_doc,
                        'label': 0
                    })
        
        # 打乱顺序
        random.shuffle(training_data)
        
        # 保存
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Reranker 训练数据已保存: {output_path}")
        logger.info(f"  样本数: {len(training_data)}")
        
        # 统计正负样本比例
        pos_count = sum(1 for item in training_data if item['label'] == 1)
        neg_count = sum(1 for item in training_data if item['label'] == 0)
        logger.info(f"  正样本: {pos_count}, 负样本: {neg_count}, 比例: {pos_count/neg_count:.2f}")
        
        return str(output_path)
    
    def generate_qa_pairs(
        self,
        output_file: str = "qa_pairs.json",
        num_pairs_per_item: int = 5
    ) -> str:
        """
        生成问答对数据
        
        Args:
            output_file: 输出文件名
            num_pairs_per_item: 每条记录生成的问答对数量
        
        Returns:
            输出文件路径
        """
        qa_pairs = []
        
        for item in tqdm(self.raw_data, desc="生成问答对"):
            name = item.get('name', '')
            english_name = item.get('english_name', '')
            text = item.get('text', '')
            
            # 生成问答对
            pairs = self._extract_qa_pairs(name, english_name, text)
            
            # 限制数量
            if len(pairs) > num_pairs_per_item:
                pairs = random.sample(pairs, num_pairs_per_item)
            
            qa_pairs.extend(pairs)
        
        # 保存
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        
        logger.info(f"问答对数据已保存: {output_path}")
        logger.info(f"  问答对数: {len(qa_pairs)}")
        
        return str(output_path)
    
    def generate_triplet_data(
        self,
        output_file: str = "triplet_data.json",
        num_hard_negatives: int = 3
    ) -> str:
        """
        生成三元组数据 (query, positive, negative)
        
        Args:
            output_file: 输出文件名
            num_hard_negatives: 难负样本数量
        
        Returns:
            输出文件路径
        """
        triplet_data = []
        all_docs = []
        
        # 收集所有文档
        for item in self.raw_data:
            doc = self._create_document(item)
            all_docs.append(doc)
        
        # 生成三元组
        for item in tqdm(self.raw_data, desc="生成三元组数据"):
            name = item.get('name', '')
            english_name = item.get('english_name', '')
            text = item.get('text', '')
            
            # 正样本
            positive = self._create_document(item)
            
            # 生成查询
            queries = self._generate_queries(name, english_name, text)
            
            for query in queries:
                # 采样难负样本
                negatives = self._sample_negatives(
                    all_docs=all_docs,
                    exclude_name=name,
                    num_negatives=num_hard_negatives,
                    hard_negative_ratio=0.5,
                    query=query
                )
                
                for neg in negatives:
                    triplet_data.append({
                        'query': query,
                        'positive': positive,
                        'negative': neg
                    })
        
        # 保存
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(triplet_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"三元组数据已保存: {output_path}")
        logger.info(f"  样本数: {len(triplet_data)}")
        
        return str(output_path)
    
    def _create_document(self, item: Dict) -> str:
        """创建文档文本"""
        name = item.get('name', '')
        english_name = item.get('english_name', '')
        text = item.get('text', '')
        
        return f"{name} ({english_name}): {text}"
    
    def _generate_queries(
        self,
        name: str,
        english_name: str,
        text: str,
        query_types: List[str] = None
    ) -> List[str]:
        """生成查询"""
        if query_types is None:
            query_types = ["definition", "composition", "property", "usage", "identification"]
        
        queries = []
        
        # 定义类查询
        if "definition" in query_types:
            queries.append(f"{name}是什么？")
            queries.append(f"什么是{name}？")
            queries.append(f"{name}的定义是什么？")
        
        # 成分类查询
        if "composition" in query_types:
            queries.append(f"{name}的化学成分是什么？")
            queries.append(f"{name}由什么组成？")
            if '化学成分' in text:
                queries.append(f"{name}的主要成分是什么？")
        
        # 性质类查询
        if "property" in query_types:
            queries.append(f"{name}的物理性质是什么？")
            queries.append(f"{name}的硬度是多少？")
            queries.append(f"{name}的密度是多少？")
            queries.append(f"{name}是什么颜色的？")
            if '硬度' in text:
                queries.append(f"{name}的莫氏硬度是多少？")
            if '密度' in text or '相对密度' in text:
                queries.append(f"{name}的相对密度是多少？")
            if '颜色' in text:
                queries.append(f"{name}有哪些颜色？")
        
        # 用途类查询
        if "usage" in query_types:
            queries.append(f"{name}有什么用途？")
            queries.append(f"{name}的主要用途是什么？")
            queries.append(f"{name}可以用来做什么？")
            if '用途' in text:
                queries.append(f"{name}在工业上有什么用途？")
        
        # 鉴别类查询
        if "identification" in query_types:
            queries.append(f"{name}如何鉴别？")
            queries.append(f"{name}的鉴别特征是什么？")
            queries.append(f"如何区分{name}和其他矿物？")
            if '鉴别' in text:
                queries.append(f"{name}的核心鉴别特征是什么？")
        
        # 成因类查询
        if "origin" in query_types:
            queries.append(f"{name}是如何形成的？")
            queries.append(f"{name}的成因是什么？")
            if '成因' in text or '产状' in text:
                queries.append(f"{name}主要产在什么地方？")
        
        return queries
    
    def _extract_qa_pairs(
        self,
        name: str,
        english_name: str,
        text: str
    ) -> List[Dict]:
        """从文本中提取问答对"""
        qa_pairs = []
        
        # 化学成分
        if '化学成分' in text:
            import re
            match = re.search(r'化学成分[：:](.*?)(?=[。\n]|结晶学)', text)
            if match:
                answer = match.group(1).strip()
                qa_pairs.append({
                    'question': f"{name}的化学成分是什么？",
                    'answer': answer,
                    'mineral': name,
                    'type': 'composition'
                })
        
        # 硬度
        if '硬度' in text:
            import re
            match = re.search(r'硬度\s*(\d+(?:\.\d+)?(?:\s*[-~]\s*\d+(?:\.\d+)?)?)', text)
            if match:
                hardness = match.group(1)
                qa_pairs.append({
                    'question': f"{name}的硬度是多少？",
                    'answer': f"{name}的莫氏硬度为{hardness}",
                    'mineral': name,
                    'type': 'property'
                })
        
        # 密度
        if '相对密度' in text:
            import re
            match = re.search(r'相对密度\s*(\d+(?:\.\d+)?(?:\s*[-~]\s*\d+(?:\.\d+)?)?)', text)
            if match:
                density = match.group(1)
                qa_pairs.append({
                    'question': f"{name}的密度是多少？",
                    'answer': f"{name}的相对密度为{density}g/cm³",
                    'mineral': name,
                    'type': 'property'
                })
        
        # 晶系
        if '晶系' in text:
            import re
            match = re.search(r'(等轴晶系|三方晶系|四方晶系|六方晶系|单斜晶系|三斜晶系|斜方晶系)', text)
            if match:
                crystal_system = match.group(1)
                qa_pairs.append({
                    'question': f"{name}属于什么晶系？",
                    'answer': f"{name}属于{crystal_system}",
                    'mineral': name,
                    'type': 'property'
                })
        
        # 用途
        if '主要用途' in text:
            import re
            match = re.search(r'主要用途[：:](.*?)(?=[。\n]|核心鉴别)', text)
            if match:
                usage = match.group(1).strip()
                qa_pairs.append({
                    'question': f"{name}有什么用途？",
                    'answer': usage,
                    'mineral': name,
                    'type': 'usage'
                })
        
        return qa_pairs
    
    def _sample_negatives(
        self,
        all_docs: List[str],
        exclude_name: str,
        num_negatives: int,
        hard_negative_ratio: float,
        query: str = None
    ) -> List[str]:
        """
        采样负样本
        
        Args:
            all_docs: 所有文档
            exclude_name: 排除的矿物名称
            num_negatives: 负样本数量
            hard_negative_ratio: 难负样本比例
            query: 查询文本 (用于难负样本挖掘)
        
        Returns:
            负样本列表
        """
        # 过滤排除项
        candidates = [doc for doc in all_docs if exclude_name not in doc]
        
        if len(candidates) < num_negatives:
            # 候选不足，重复采样
            return random.choices(candidates, k=num_negatives)
        
        # 简单随机采样 (实际应用中可以使用 BM25 或 Embedding 相似度挖掘难负样本)
        negatives = random.sample(candidates, num_negatives)
        
        return negatives


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练数据准备脚本")
    parser.add_argument("--data_path", type=str, default="/root/V2_0_Mineral_QA/text_data/45.json",
                        help="原始数据路径")
    parser.add_argument("--output_dir", type=str, default="./training_data",
                        help="输出目录")
    parser.add_argument("--num_negatives", type=int, default=7,
                        help="负样本数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 创建数据生成器
    generator = TrainingDataGenerator(
        data_path=args.data_path,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # 生成各类训练数据
    logger.info("=" * 60)
    logger.info("开始生成训练数据...")
    logger.info("=" * 60)
    
    # 1. Embedding 训练数据
    generator.generate_embedding_training_data(
        output_file="embedding_train.json",
        num_negatives=args.num_negatives
    )
    
    # 2. Reranker 训练数据
    generator.generate_reranker_training_data(
        output_file="reranker_train.json",
        num_negatives=4
    )
    
    # 3. 问答对数据
    generator.generate_qa_pairs(
        output_file="qa_pairs.json",
        num_pairs_per_item=5
    )
    
    # 4. 三元组数据
    generator.generate_triplet_data(
        output_file="triplet_data.json",
        num_hard_negatives=3
    )
    
    logger.info("=" * 60)
    logger.info("训练数据生成完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()