"""
将矿物文本数据导入 FAISS 向量数据库
使用本地 Qwen3-Embedding 模型
"""
import json
import numpy as np
import faiss
import pickle
import torch
import torch.nn.functional as F
from pathlib import Path

def main():
    json_path = "/root/V2_0_Mineral_QA/text_data/45.json"
    
    # 1. 加载数据
    print(f"加载数据: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"共 {len(data)} 条记录")
    
    # 2. 初始化 embedding 函数
    print("加载 Embedding 模型...")
    from transformers import AutoTokenizer, AutoModel
    import torch

    model_path = "/root/V2_0_Mineral_QA/model/Qwen3-Embedding-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    embed_model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    embed_model = embed_model.cuda()
    embed_model.eval()

    dim = 1024  # Qwen3-Embedding-0.6B 输出 1024 维
    print(f"Embedding 模型加载成功, dimension={dim}")
    
    # 3. 准备数据
    print("准备数据...")
    docs = []
    for item in data:
        text = f"{item.get('name', '')} ({item.get('english_name', '')}): {item.get('text', '')}"
        docs.append(text)
    
    # 4. 生成向量 (使用与 qwen3_embedder 相同的方式: [CLS] token + L2 normalize)
    print(f"生成 {len(docs)} 条文本的向量...")
    vectors = []
    for text in docs:
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = embed_model(**inputs)
            # 使用 [CLS] token (与 qwen3_embedder 相同)
            embedding = outputs.last_hidden_state[:, 0, :]
            # L2 归一化 (与 qwen3_embedder 相同)
            embedding = F.normalize(embedding, p=2, dim=1)
            embedding = embedding.cpu().numpy()[0]
        vectors.append(embedding)
    vectors = np.array(vectors).astype('float32')
    print(f"向量生成完成, shape: {vectors.shape}")
    
    # 5. 创建 FAISS 索引
    print("创建 FAISS 索引...")
    index = faiss.IndexFlatL2(dim)  # L2 距离
    index.add(vectors)
    print(f"索引创建成功, 共 {index.ntotal} 条向量")
    
    # 6. 保存索引和数据
    db_dir = Path("./faiss_mineral_45")
    db_dir.mkdir(exist_ok=True)
    
    faiss.write_index(index, str(db_dir / "index.faiss"))
    
    with open(db_dir / "data.pkl", "wb") as f:
        pickle.dump(data, f)
    
    print("数据保存成功!")
    
    # 7. 验证 - 搜索测试
    print("\n验证数据 (搜索测试):")
    query_vec = vectors[0:1]
    D, I = index.search(query_vec, 3)
    print(f"搜索 '{(data[0].get('name', ''))}' 的结果:")
    for i, idx in enumerate(I[0]):
        print(f"  {i+1}. {data[idx].get('name', '')} ({data[idx].get('english_name', '')}) - 距离: {D[0][i]:.4f}")
    
    print("\n✅ 数据导入完成!")
    print(f"向量维度: {dim}")
    print(f"数据库目录: {db_dir}")

if __name__ == "__main__":
    main()
