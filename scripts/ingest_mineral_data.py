"""
将矿物文本数据导入 Milvus 向量数据库
"""
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_mineral_data(json_path: str):
    """加载矿物数据 (JSON Lines 格式)"""
    print(f"加载数据: {json_path}")
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"共 {len(data)} 条记录")
    return data

def create_text_for_embedding(item):
    """为每条记录创建用于 embedding 的文本"""
    texts = []
    if item.get('name'):
        texts.append(f"矿物名称: {item['name']}")
    if item.get('english_name'):
        texts.append(f"英文名: {item['english_name']}")
    if item.get('brief_characterization'):
        texts.append(f"描述: {item['brief_characterization']}")
    if item.get('main_purpose'):
        texts.append(f"主要用途: {item['main_purpose']}")
    if item.get('specific_purpose'):
        texts.append(f"具体用途: {item['specific_purpose']}")
    if item.get('place_of_origin'):
        texts.append(f"产地: {item['place_of_origin']}")
    return " | ".join(texts)

def init_milvus():
    """初始化 Milvus Lite"""
    from pymilvus import MilvusClient

    client = MilvusClient("./milvus_mineral.db")
    print(f"Milvus Lite 连接成功")

    return client

def create_collection(client, dimension: int = 1024):
    """创建集合"""
    collection_name = "mineral_knowledge"

    # 检查是否已存在
    try:
        client.get_collection(collection_name)
        print(f"集合 {collection_name} 已存在，删除重建...")
        client.drop_collection(collection_name)
    except:
        pass

    # 创建集合
    client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
        description="矿物知识库",
        overwrite=True
    )
    print(f"集合 {collection_name} 创建成功，维度: {dimension}")
    return collection_name

def init_embedder():
    """初始化 embedding 模型"""
    from transformers import AutoTokenizer, AutoModel
    import torch

    model_path = "/root/V2_0_Mineral_QA/model/Qwen3-Embedding-0.6B"
    print(f"加载 Embedding 模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model = model.cuda()
    model.eval()

    return tokenizer, model

def get_embeddings(texts, tokenizer, model, batch_size: int = 32):
    """获取文本的 embedding"""
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="生成向量"):
        batch_texts = texts[i:i+batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # 使用 mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)

        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)

def insert_data(client, collection_name: str, data: list, embeddings: np.ndarray):
    """插入数据到 Milvus"""
    print(f"插入 {len(data)} 条数据...")

    entities = []
    for i, item in enumerate(data):
        entity = {
            "id": i,
            "vector": embeddings[i].tolist(),
            "name": item.get('name', ''),
            "english_name": item.get('english_name', ''),
            "brief_characterization": item.get('brief_characterization', ''),
            "specific_purpose": item.get('specific_purpose', ''),
            "place_of_province": item.get('place_of_province', ''),
            "storage_location": item.get('the_storage_location', ''),
            "photo_url": item.get('photo_url', ''),
        }
        entities.append(entity)

    # 分批插入
    batch_size = 100
    for i in tqdm(range(0, len(entities), batch_size), desc="插入数据"):
        batch = entities[i:i+batch_size]
        client.insert(collection_name=collection_name, data=batch)

    print(f"数据插入完成!")

def main():
    import sys
    print("开始执行...", flush=True)

    # 配置
    json_path = "/root/V2_0_Mineral_QA/text_data/all_v1.json"
    dimension = 1024  # Qwen3-Embedding 输出维度
    print(f"配置完成: dimension={dimension}", flush=True)

    # 1. 加载数据
    data = load_mineral_data(json_path)
    print(f"数据加载完成: {len(data)} 条", flush=True)

    # 2. 初始化 Milvus
    print("初始化 Milvus...", flush=True)
    client = init_milvus()
    print("Milvus 初始化完成", flush=True)

    # 3. 创建集合
    print("创建集合...", flush=True)
    collection_name = create_collection(client, dimension)
    print(f"集合创建完成: {collection_name}", flush=True)

    # 4. 初始化 embedder
    print("加载 Embedding 模型 (这可能需要几分钟)...", flush=True)
    tokenizer, model = init_embedder()
    print("Embedding 模型加载完成", flush=True)

    # 5. 为每条数据创建 embedding 文本
    texts = [create_text_for_embedding(item) for item in data]
    print(f"准备生成 {len(texts)} 条文本的向量...")

    # 6. 生成向量
    embeddings = get_embeddings(texts, tokenizer, model)

    # 7. 插入数据
    insert_data(client, collection_name, data, embeddings)

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
    print(f"向量维度: {dimension}")
    print(f"数据库文件: ./milvus_mineral.db")

if __name__ == "__main__":
    main()
