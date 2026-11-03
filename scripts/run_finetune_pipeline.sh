#!/usr/bin/env bash
# 串行执行 Embedding -> Reranker 微调，避免两个任务同时占用显存
set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cd /root/V2_0_Mineral_QA
LOG_DIR=/root/autodl-tmp/logs
mkdir -p "$LOG_DIR"

echo "[$(date '+%F %T')] pipeline start"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

echo "[$(date '+%F %T')] step 1/2: embedding"
python scripts/finetune_qwen3_embedding.py \
  --num_epochs 2 \
  --batch_size 2 \
  --grad_accum 8 \
  --learning_rate 2e-5 \
  --max_doc_len 256 \
  --max_query_len 64 \
  --output_dir /root/autodl-tmp/outputs/embedding \
  > "$LOG_DIR/embedding_train.log" 2>&1
echo "[$(date '+%F %T')] embedding done"

echo "[$(date '+%F %T')] step 2/2: reranker"
python scripts/finetune_qwen3_reranker.py \
  --num_epochs 2 \
  --batch_size 16 \
  --grad_accum 1 \
  --learning_rate 2e-5 \
  --max_length 320 \
  --output_dir /root/autodl-tmp/outputs/reranker \
  > "$LOG_DIR/reranker_train.log" 2>&1
echo "[$(date '+%F %T')] reranker done"
echo "[$(date '+%F %T')] pipeline finished"
