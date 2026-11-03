# Mineral Image–Text Question Answering System

A retrieval-augmented generation (RAG) system for mineral specimen question answering. It supports text questions and “image + question” inputs, integrating visual recognition, hybrid retrieval, reranking, and local large language model generation into a complete pipeline.

## Architecture

```
Text question / Mineral image
   ├─ Image → Swin Transformer → Top-3 visual anchors
   ├─ Query fusion (optional LLM-based rewriting)
   ├─ Qwen3-Embedding → FAISS dense retrieval
   ├─ jieba BM25 sparse retrieval
   ├─ RRF fusion
   ├─ Qwen3-Reranker reranking → Top-3
   └─ Qwen2.5-7B-Instruct answer generation
```

## Workflow

1. The user enters a text question or uploads a mineral image.
2. Swin Transformer classifies the image and returns the Top-3 candidate minerals.
3. For image–text inputs, the query fusion module rewrites the visual candidates and question into a retrieval query.
4. The embedding and BM25 channels retrieve documents in parallel, and RRF merges their results into a candidate list.
5. The reranker rescores the candidates and retains the Top-3 documents.
6. The documents and question are combined into a prompt, and the local Qwen2.5-7B model generates an answer.
7. The Streamlit page displays the answer, cited documents, and visual recognition results.

## Directory Structure

| Path | Description |
|---|---|
| `run_pipeline.py` | Entry point (Web / CLI / environment check) |
| `pipeline/mineral_qa_pipeline.py` | Full pipeline orchestration |
| `retrieval/` | Embedding, FAISS, BM25, hybrid retrieval, and reranking |
| `generator/` | LLM clients, prompts, and query fusion |
| `vision/` | Swin inference and training code |
| `scripts/` | Data construction, model fine-tuning, indexing, and evaluation scripts |
| `training_data/` | Corpus and training sets |
| `faiss_mineral_6408/` | FAISS index |

## Quick Start

```bash
pip install -r requirements.txt

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Check the environment
python run_pipeline.py --mode check

# Launch the Web interface
streamlit run run_pipeline.py
```

Model paths are configured in `pipeline/mineral_qa_pipeline.py`, including the generation model, embedding model, reranker, Swin weights, and label mapping. If fine-tuned weights are unavailable, the code falls back to the base models under `model/`.

## Data Construction

```bash
python scripts/build_retrieval_dataset.py \
  --input text_data/all_v1.json \
  --output_dir training_data \
  --num_negatives 7 --hard_ratio 0.7 \
  --val_ratio 0.1 --test_ratio 0.1 --seed 42
```

This generates the corpus, embedding training sets, and reranker training pairs, with grouped splitting by mineral name.

## Model Training

```bash
# Embedding fine-tuning
python scripts/finetune_qwen3_embedding.py \
  --model_path model/Qwen3-Embedding-0.6B \
  --train_data training_data/embedding/train.json \
  --val_data training_data/embedding/val.json \
  --test_data training_data/embedding/test.json \
  --output_dir outputs/embedding

# Reranker fine-tuning
python scripts/finetune_qwen3_reranker.py \
  --model_path model/Qwen3-Reranker-0.6B \
  --train_data training_data/reranker/train_pairs.jsonl \
  --val_data training_data/reranker/val_pairs.jsonl \
  --test_data training_data/reranker/test_pairs.jsonl \
  --output_dir outputs/reranker
```

## Evaluation

```bash
python scripts/eval_retrieval.py --split test
python scripts/eval_rerank.py --split test
python scripts/eval_judge.py --eval-set training_data/eval_100.jsonl
```

The image–text end-to-end evaluation set (100 test images + class labels + questions + gold documents) is located in `eval/vision_qa/`. Run:

```bash
python scripts/eval_vision_qa.py --dataset eval/vision_qa/vision_qa_100.jsonl
```

The fixed 100-example evaluation set is saved in `training_data/eval_100.jsonl`, randomly sampled from the test set using seed=42. It can be reproduced with:

```bash
python scripts/sample_eval_set.py --num 100 --seed 42
```

These are used for full-corpus retrieval evaluation, reranking ablation, and end-to-end LLM-as-a-Judge evaluation, respectively.

## Notes

- Python 3.8+ is required; a GPU is recommended.
- For GPU execution, setting `OMP_NUM_THREADS` and `MKL_NUM_THREADS` is recommended.


