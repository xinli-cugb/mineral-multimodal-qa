# Mineral Multimodal Question Answering System

A mineral question answering system that combines computer vision and retrieval-augmented generation (RAG). The project includes visual recognition, query fusion, hybrid retrieval, reranking, and local large language model generation.

The full pipeline follows this workflow:

```text
Mineral image + Question
          |
          v
Swin Transformer -> Top-3 visual candidates
          |
          v
Query fusion
          |
          v
Qwen3-Embedding + BM25 -> RRF fusion
          |
          v
Qwen3-Reranker -> Top-3 evidence documents
          |
          v
Qwen2.5-7B-Instruct -> Answer
```

## 1. Repository Structure

| Path | Description |
| --- | --- |
| `pipeline/mineral_qa_pipeline.py` | Full pipeline orchestration |
| `vision/` | Swin Transformer inference and training |
| `retrieval/` | Embedding, FAISS, Milvus, BM25, hybrid retrieval, and reranking |
| `generator/` | Language-model clients, prompt templates, and query fusion |
| `scripts/` | Data preparation, fine-tuning, indexing, evaluation, and demonstration scripts |
| `faiss_mineral_45/` | FAISS index and metadata for the 45-record knowledge collection |
| `training_data/` | Training artifacts, statistics, and fixed text evaluation examples |
| `eval/vision_qa/` | Image-question examples, annotations, and images |
| `requirements.txt` | Python dependency requirements |

## 2. Installation

Install the project dependencies in a Python environment:

```bash
pip install -r requirements.txt
```

For GPU execution, the original project uses the following thread settings:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

Model paths, data paths, and runtime parameters are configured in the corresponding scripts.

## 3. Data Preparation and Model Training

| Script | Purpose |
| --- | --- |
| `scripts/refactor_mineral_data.py` | Rewrite structured mineral records through a language-model API |
| `scripts/refactor_mineral_data_local.py` | Rewrite mineral records using a local language model |
| `scripts/prepare_training_data.py` | Prepare embedding and reranker training examples |
| `scripts/generate_hard_negatives.py` | Select negative examples using embedding similarity |
| `scripts/build_retrieval_dataset.py` | Build the corpus, queries, negative examples, and grouped text splits |
| `scripts/finetune_qwen3_embedding.py` | Fine-tune the embedding model |
| `scripts/finetune_qwen3_reranker.py` | Fine-tune the reranker |
| `vision/swin_trainer.py` | Train and validate the mineral image classifier |
| `scripts/ingest_45_faiss.py` | Build a FAISS index for the 45-record collection |
| `scripts/ingest_45.py` | Import the 45-record collection into Milvus |
| `scripts/ingest_mineral_data.py` | Import structured mineral data into Milvus |

The text dataset builder groups records by normalized mineral name when creating training, validation, and test splits.

## 4. Question Answering and Evaluation

`pipeline/mineral_qa_pipeline.py` connects visual recognition, query fusion, retrieval, reranking, and answer generation.

`scripts/mineral_qa_system.py` provides a simplified Streamlit demonstration that identifies a mineral, looks up its information by name, and generates an answer.

| Script | Purpose |
| --- | --- |
| `scripts/eval_retrieval.py` | Evaluate dense, BM25, and hybrid retrieval |
| `scripts/eval_rerank.py` | Compare retrieval rankings before and after reranking |
| `scripts/eval_judge.py` | Evaluate text question answering using an LLM judge |
| `scripts/eval_common.py` | Shared data-loading and metric functions |
| `scripts/test_refactor.py` | Test the output format of data rewriting |
| `scripts/test_api_response.py` | Inspect language-model API responses |

The retrieval evaluation functions calculate Recall@K, NDCG@K, and MRR@K. The Judge script supports rubric scoring and binary judgments. Its rubric covers accuracy, completeness, faithfulness, and non-hallucination.

## 5. Test Set Examples

### 5.1 Text Examples

`training_data/eval_100.jsonl` contains **100 fixed text examples**.

| Field | Description |
| --- | --- |
| `eval_id` | Example identifier |
| `query` | Question text |
| `query_type` | Question category |
| `positive_id` | Reference document identifier |
| `positive_name` | Reference mineral name |
| `positive_text` | Reference document text |
| `negatives` | Negative document texts |

The question categories are distributed as follows:

| Category | Number |
| --- | ---: |
| attribute | 29 |
| definition | 25 |
| usage | 18 |
| origin | 11 |
| english | 6 |
| attribute_only | 6 |
| identification | 5 |
| **Total** | **100** |

`attribute` asks about a property of a named mineral. `attribute_only` asks for a mineral identity from property conditions without providing its name.

### 5.2 Image-Question Examples

`eval/vision_qa/vision_qa_100.jsonl` contains **100 image-question examples covering 45 class labels**. The corresponding **100 images** are stored under `eval/vision_qa/images/`.

| Field | Description |
| --- | --- |
| `eval_id` | Example identifier |
| `image` | Image path relative to the annotation file |
| `label_en` | English class label |
| `label_zh` | Chinese mineral label |
| `question` | Question text |
| `gold_id` | Reference document identifier |
| `gold_name` | Reference mineral name |
| `gold_text` | Reference document text |
| `class_aliases` | Alternative class names |

The text examples and image-question examples are provided as separate JSONL files, with one record per line.
