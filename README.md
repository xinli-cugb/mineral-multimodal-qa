# Mineral Multimodal Question Answering System

A mineral question answering system that combines computer vision and retrieval-augmented generation (RAG). The repository includes mineral image recognition, query fusion, hybrid retrieval, reranking, and local large language model generation.

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

## 1. Repo Structure

```text
.
├── pipeline/             # End-to-end mineral QA pipeline
├── vision/               # Swin Transformer inference and training
├── retrieval/            # Dense, sparse, hybrid retrieval, and reranking
├── generator/            # Query fusion, prompt templates, and LLM clients
├── scripts/              # Data preparation, training, indexing, and evaluation
├── faiss_mineral_45/     # FAISS index and metadata for the mineral knowledge base
├── eval/vision_qa/       # Image-question examples and corresponding images
├── requirements.txt      # Python dependencies
└── README.md
```

Key entry points:

| File | Description |
| --- | --- |
| `pipeline/mineral_qa_pipeline.py` | Orchestrates visual recognition, retrieval, reranking, and answer generation |
| `scripts/mineral_qa_system.py` | Streamlit interface for interactive mineral image question answering |
| `scripts/build_retrieval_dataset.py` | Builds retrieval-model samples from structured mineral records |
| `vision/swin_trainer.py` | Fine-tunes the Swin Transformer mineral classifier |
| `scripts/finetune_qwen3_embedding.py` | Fine-tunes the Qwen3 embedding model |
| `scripts/finetune_qwen3_reranker.py` | Fine-tunes the Qwen3 reranker |
| `scripts/eval_retrieval.py` | Evaluates dense, BM25, and hybrid retrieval |
| `scripts/eval_rerank.py` | Evaluates reranking performance |
| `scripts/eval_judge.py` | Runs end-to-end answer evaluation with an LLM judge |

## 2. Installation

Python 3.9 or later is required. A CUDA-enabled GPU is recommended for model training and local inference.

```bash
git clone <repository-url>
cd <repository-directory>

conda create -n mineral-qa python=3.10 -y
conda activate mineral-qa
pip install -r requirements.txt
```

Milvus is optional. Install its Python client only when using the Milvus retrieval backend:

```bash
pip install "pymilvus>=2.4.0"
```

## 3. Dataset Preparation

The complete training datasets are not included in this repository. Prepare the image data and structured mineral records locally, and pass their paths to the corresponding scripts.

### 3.1 Visual Classification Data

`vision/swin_trainer.py` supports a class-directory layout:

```text
images/
├── class_1/
│   ├── image_001.jpg
│   └── image_002.jpg
└── class_2/
    └── image_003.jpg
```

It also accepts JSON or CSV annotation files. A JSON annotation file should contain records with `image_path` and `label` fields:

```json
[
  {"image_path": "/path/to/image_001.jpg", "label": "class_1"},
  {"image_path": "/path/to/image_003.jpg", "label": "class_2"}
]
```

A CSV annotation file should contain the same two columns:

```csv
image_path,label
/path/to/image_001.jpg,class_1
/path/to/image_003.jpg,class_2
```

### 3.2 Retrieval Training Data

`scripts/build_retrieval_dataset.py` accepts a JSON array or JSONL file containing structured mineral records. The fields used by the script include `name`, `english_name`, `brief_characterization`, `main_purpose`, `specific_purpose`, and the available origin fields such as `place_of_origin`, `place_of_province`, or `place_of_country`.

Prepare fixed training, validation, and test splits with:

```bash
python scripts/build_retrieval_dataset.py \
  --input <path-to-structured-mineral-records.json> \
  --output_dir <path-to-output-directory> \
  --embedding_model_path <path-to-Qwen3-Embedding-0.6B> \
  --seed 42
```

The script generates query-positive-document samples, selects negative documents, and groups records by normalized mineral name when creating the data splits.

## 4. How to Run

The original experiment used local model checkpoints and datasets. Before running a script, replace its default paths with the locations of your Swin Transformer checkpoint, Qwen3-Embedding model, Qwen3-Reranker model, Qwen2.5-7B-Instruct model, and mineral data.

### 4.1 Run the Interactive Demo

```bash
streamlit run scripts/mineral_qa_system.py
```

Set the model and database paths in the Streamlit sidebar, upload a mineral image, enter a question, and submit it to run the interactive question answering demo.

### 4.2 Train the Models

Train the visual classifier:

```bash
python vision/swin_trainer.py \
  --train_image_dir <path-to-training-images> \
  --train_annotations <path-to-training-annotations> \
  --val_image_dir <path-to-validation-images> \
  --val_annotations <path-to-validation-annotations> \
  --num_classes 45 \
  --output_dir <path-to-swin-output>
```

Fine-tune the embedding and reranking models with the prepared split files:

```bash
python scripts/finetune_qwen3_embedding.py \
  --model_path <path-to-Qwen3-Embedding-0.6B> \
  --train_data <path-to-embedding-train.json> \
  --val_data <path-to-embedding-val.json> \
  --test_data <path-to-embedding-test.json> \
  --output_dir <path-to-embedding-output>

python scripts/finetune_qwen3_reranker.py \
  --model_path <path-to-Qwen3-Reranker-0.6B> \
  --train_data <path-to-reranker-train.jsonl> \
  --val_data <path-to-reranker-val.jsonl> \
  --test_data <path-to-reranker-test.jsonl> \
  --output_dir <path-to-reranker-output>
```

### 4.3 Evaluate the System

After configuring the local split, index, and model paths, run:

```bash
python scripts/eval_retrieval.py --split test --methods dense,bm25,hybrid
python scripts/eval_rerank.py --split test --methods dense,hybrid
python scripts/eval_judge.py --split test --num-samples 100 --seed 42
```

## 5. Test Set Example

`eval/vision_qa/vision_qa_examples.jsonl` contains image-question examples. The corresponding images are stored under `eval/vision_qa/images/`.

The examples are stored in JSONL format, with one record per line.