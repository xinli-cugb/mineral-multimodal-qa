"""
基于本地 Embedding 模型计算矿物数据相似度，生成难负样本（优化版）
使用批量计算和矩阵运算大幅提升速度
"""
import json
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import argparse
import numpy as np


class FastSimilarityMatcher:
    """
    快速相似度匹配器（优化版）
    
    使用批量计算和矩阵运算，大幅提升相似度计算速度
    """
    
    def __init__(
        self,
        embedding_model_path: str = "/root/V2_0_Mineral_QA/model/Qwen3-Embedding-0.6B",
        device: str = "cuda",
        batch_size: int = 32
    ):
        """
        初始化快速相似度匹配器
        
        Args:
            embedding_model_path: 本地 Embedding 模型路径
            device: 运行设备 (cuda/cpu)
            batch_size: 批量计算大小
        """
        self.embedding_model_path = embedding_model_path
        self.device = device
        self.batch_size = batch_size
        
        print(f"加载 Embedding 模型: {embedding_model_path}")
        print(f"使用设备: {device}")
        print(f"批量大小: {batch_size}")
        
        # 加载 tokenizer 和模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            embedding_model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModel.from_pretrained(
            embedding_model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        print("✅ Embedding 模型加载完成！")
    
    def get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        批量获取文本的 Embedding 向量
        
        Args:
            texts: 文本列表
        
        Returns:
            Embedding 矩阵 (n x embedding_dim)
        """
        embeddings = []
        
        with torch.no_grad():
            # 分批处理
            for i in tqdm(range(0, len(texts), self.batch_size), desc="计算 Embedding"):
                batch_texts = texts[i:i + self.batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # 归一化
                batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                embeddings.append(batch_embeddings)
        
        # 合并所有批次
        all_embeddings = np.vstack(embeddings)
        return all_embeddings
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        计算相似度矩阵（批量矩阵运算）
        
        Args:
            embeddings: Embedding 矩阵 (n x embedding_dim)
        
        Returns:
            相似度矩阵 (n x n)
        """
        print("计算相似度矩阵...")
        # 使用矩阵乘法批量计算余弦相似度
        similarity_matrix = np.dot(embeddings, embeddings.T)
        return similarity_matrix
    
    def generate_query(self, name: str) -> str:
        """
        生成查询语句
        
        Args:
            name: 矿物名称
        
        Returns:
            查询语句
        """
        return f"{name}的是什么，它有什么性质？"
    
    def build_response_text(self, item: Dict) -> str:
        """
        构建响应文本
        
        Args:
            item: 矿物数据
        
        Returns:
            响应文本
        """
        name = item.get('name', '')
        english_name = item.get('English_name', '')
        description = item.get('描述', '')
        
        if english_name:
            return f"{name}（{english_name}），{description}"
        return f"{name}，{description}"
    
    def find_hard_negatives_batch(
        self,
        similarity_matrix: np.ndarray,
        top_k: int = 5
    ) -> List[List[int]]:
        """
        批量找出难负样本
        
        Args:
            similarity_matrix: 相似度矩阵 (n x n)
            top_k: 每个样本的难负样本数量
        
        Returns:
            难负样本索引列表的列表
        """
        print("提取难负样本...")
        n = similarity_matrix.shape[0]
        all_hard_negatives = []
        
        for i in tqdm(range(n), desc="提取难负样本"):
            # 获取与第i个样本的相似度
            similarities = similarity_matrix[i].copy()
            
            # 将自己设为最小值，避免选中自己
            similarities[i] = -np.inf
            
            # 找出相似度最高的top_k个索引
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            all_hard_negatives.append(top_indices)
        
        return all_hard_negatives
    
    def process_all(
        self,
        data_list: List[Dict],
        output_file: str,
        top_k: int = 5
    ):
        """
        处理所有数据，生成难负样本（优化版）
        
        Args:
            data_list: 所有矿物数据
            output_file: 输出文件路径
            top_k: 每个样本的难负样本数量
        """
        print(f"开始处理 {len(data_list)} 条数据...")
        
        # 1. 构建所有文本
        print("构建文本...")
        all_texts = [self.build_response_text(item) for item in data_list]
        
        # 2. 批量计算所有 Embedding
        embeddings = self.get_batch_embeddings(all_texts)
        print(f"✅ Embedding 计算完成，形状: {embeddings.shape}")
        
        # 3. 批量计算相似度矩阵
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        print(f"✅ 相似度矩阵计算完成，形状: {similarity_matrix.shape}")
        
        # 4. 批量提取难负样本索引
        hard_negative_indices = self.find_hard_negatives_batch(similarity_matrix, top_k)
        print(f"✅ 难负样本提取完成")
        
        # 5. 构建最终结果
        print("构建最终结果...")
        results = []
        
        for i, item in enumerate(tqdm(data_list, desc="构建结果")):
            # 生成查询
            query = self.generate_query(item.get('name', ''))
            
            # 生成响应
            response = all_texts[i]
            
            # 获取难负样本文本
            rejected_response = [all_texts[idx] for idx in hard_negative_indices[i]]
            
            # 构建结果
            result = {
                "response": response,
                "query": query,
                "rejected_response": rejected_response
            }
            
            results.append(result)
        
        # 保存结果
        self._save_data(results, output_file)
        print(f"\n✅ 处理完成！共生成 {len(results)} 条数据")
        print(f"结果已保存到: {output_file}")
    
    def _save_data(self, data: List[Dict], output_file: str):
        """保存数据到文件"""
        # 创建输出目录
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="矿物数据相似度匹配与难负样本生成脚本（优化版）")
    parser.add_argument("--input_file", type=str, 
                        default="/root/V2_0_Mineral_QA/text_data/all_v1_refactored.json",
                        help="输入文件路径（重构后的矿物数据）")
    parser.add_argument("--output_file", type=str, 
                        default="/root/rag/data/knowledge_base/qwen3_emb.json",
                        help="输出文件路径")
    parser.add_argument("--embedding_model_path", type=str, 
                        default="/root/V2_0_Mineral_QA/model/Qwen3-Embedding-0.6B",
                        help="本地 Embedding 模型路径")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="运行设备")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批量计算大小")
    parser.add_argument("--top_k", type=int, default=5,
                        help="难负样本数量")
    parser.add_argument("--limit", type=int, default=None,
                        help="处理数据数量限制（用于测试）")
    
    args = parser.parse_args()
    
    # 读取原始数据
    print(f"读取数据: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"共 {len(data)} 条矿物数据")
    
    # 限制处理数量（用于测试）
    if args.limit:
        data = data[:args.limit]
        print(f"限制处理数量为: {args.limit}")
    
    # 创建快速相似度匹配器
    matcher = FastSimilarityMatcher(
        embedding_model_path=args.embedding_model_path,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # 处理所有数据
    matcher.process_all(
        data_list=data,
        output_file=args.output_file,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()