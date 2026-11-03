"""
将矿物文本数据导入 Milvus 向量数据库
使用 pymilvus[model] 内置 embedding
"""
import json
from pymilvus import MilvusClient, model

def main():
    json_path = "/root/V2_0_Mineral_QA/text_data/45.json"
    
    # 1. 加载数据
    print(f"加载数据: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"共 {len(data)} 条记录")
    
    # 2. 初始化 Milvus
    db_path = "./milvus_mineral_45.db"
    client = MilvusClient(db_path)
    print(f"Milvus Lite 连接成功: {db_path}")
    
    # 3. 创建集合
    collection_name = "mineral_knowledge_45"
    
    # 检查并删除已存在的集合
    if client.has_collection(collection_name):
        print(f"删除已存在的集合: {collection_name}")
        client.drop_collection(collection_name)
    
    # 创建新集合 (dimension 由 embedding 模型决定)
    client.create_collection(
        collection_name=collection_name,
        dimension=768  # paraphrase-albert-small-v2 输出 768 维
    )
    print(f"集合创建成功: {collection_name}, dimension=768")
    
    # 4. 初始化 embedding 函数
    print("加载 Embedding 模型 (paraphrase-albert-small-v2)...")
    embedding_fn = model.DefaultEmbeddingFunction()
    dim = embedding_fn.dim
    print(f"Embedding 模型加载成功, dimension={dim}")
    
    # 5. 准备数据
    print("准备数据...")
    docs = []
    for item in data:
        # 使用 name + text 作为检索内容
        text = f"{item.get('name', '')} ({item.get('english_name', '')}): {item.get('text', '')}"
        docs.append(text)
    
    # 6. 生成向量
    print(f"生成 {len(docs)} 条文本的向量...")
    vectors = embedding_fn.encode_documents(docs)
    print(f"向量生成完成, shape: {vectors.shape}")
    
    # 7. 组装数据并插入
    print("插入数据到 Milvus...")
    entities = []
    for i, item in enumerate(data):
        entity = {
            "id": i,
            "vector": vectors[i].tolist(),
            "name": item.get('name', ''),
            "english_name": item.get('english_name', ''),
            "text": item.get('text', '')
        }
        entities.append(entity)
    
    # 分批插入
    batch_size = 50
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i+batch_size]
        client.insert(collection_name=collection_name, data=batch)
        print(f"  已插入 {min(i+batch_size, len(entities))}/{len(entities)} 条")
    
    print("数据插入完成!")
    
    # 8. 验证
    print("\n验证数据:")
    result = client.query(
        collection_name=collection_name,
        filter="id >= 0",
        output_fields=["name", "english_name"],
        limit=3
    )
    for r in result:
        print(f"  - {r['name']} ({r['english_name']})")
    
    print("\n✅ 数据导入完成!")
    print(f"集合: {collection_name}")
    print(f"记录数: {len(data)}")
    print(f"向量维度: {dim}")
    print(f"数据库文件: {db_path}")

if __name__ == "__main__":
    main()
