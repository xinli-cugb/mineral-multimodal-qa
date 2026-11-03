#!/bin/bash
# 微调 Qwen3-Embedding 和 Qwen3-Reranker 模型
# 使用生成的难负样本数据进行微调

set -e

# 配置参数
MODEL_PATH="/root/V2_0_Mineral_QA/model/Qwen3-Embedding-0.6B"
DATA_PATH="/root/rag/data/knowledge_base/qwen3_emb.json"
OUTPUT_DIR="/root/V2_0_Mineral_QA/output/finetuned_models"
DEVICE="cuda"

# 训练参数
NUM_EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=2e-5
MAX_LENGTH=512
WARMUP_RATIO=0.1

echo "========================================="
echo "开始微调 Qwen3-Embedding 和 Reranker 模型"
echo "========================================="
echo "模型路径: $MODEL_PATH"
echo "数据路径: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "设备: $DEVICE"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR/embedding"
mkdir -p "$OUTPUT_DIR/reranker"

# 1. 微调 Qwen3-Embedding
echo "========================================="
echo "1. 微调 Qwen3-Embedding 模型"
echo "========================================="
python3 /root/V2_0_Mineral_QA/scripts/finetune_qwen3_embedding.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR/embedding" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --warmup_ratio $WARMUP_RATIO \
    --device $DEVICE \
    --num_negatives 5 \
    --temperature 0.05 \
    --save_steps 100 \
    --eval_steps 100

echo ""
echo "✅ Qwen3-Embedding 微调完成!"
echo ""

# 2. 微调 Qwen3-Reranker
echo "========================================="
echo "2. 微调 Qwen3-Reranker 模型"
echo "========================================="
python3 /root/V2_0_Mineral_QA/scripts/finetune_qwen3_reranker.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR/reranker" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --warmup_ratio $WARMUP_RATIO \
    --device $DEVICE \
    --num_negatives 5 \
    --dropout 0.1 \
    --save_steps 100 \
    --eval_steps 100 \
    --mode train

echo ""
echo "✅ Qwen3-Reranker 微调完成!"
echo ""

echo "========================================="
echo "所有微调任务完成!"
echo "========================================="
echo "模型保存位置:"
echo "  - Embedding: $OUTPUT_DIR/embedding/final_model"
echo "  - Reranker: $OUTPUT_DIR/reranker/final_model"
echo ""