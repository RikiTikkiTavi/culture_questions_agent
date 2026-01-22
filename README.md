# Cultural QA Agent - RAG System

A Retrieval-Augmented Generation (RAG) system for answering cultural questions about China, Iran, the United Kingdom, and the United States. Built with constraint of using only the `Meta-Llama-3-8B` model.

## ✨ Features

- 🔍 **Multi-Source Retrieval**: Wikipedia, Wikivoyage, web search, and training data
- 🎯 **Hybrid Search**: Dense (BGE-M3) + Sparse (BM25) retrieval
- 🔄 **Advanced Reranking**: Cross-encoder (BGE-reranker-v2-m3) or Late-Interaction (ColBERT)
- 🧠 **LLM-based Query Generation**: Semantic query expansion using Llama-3-8B
- 📊 **Two Question Types**: Multiple Choice Questions (MCQ) and Short Answer Questions (SAQ)
- ⚡ **Async Workflow**: Event-driven architecture using LlamaIndex Workflows
- 📈 **MLflow Integration**: Comprehensive experiment tracking and metrics
- 🚀 **Multi-Process Inference**: Optimized batch prediction with configurable concurrency

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage](#usage)
  - [Building the Index](#1-build-the-index)
  - [Running Evaluation](#2-run-evaluation)
  - [Generating Predictions](#3-generate-competition-submissions)
- [Configuration](#configuration)
- [Performance](#performance)
- [License](#license)

## 📁 Project Structure

```
.
├── conf/
│   └── config.yaml                  # Hydra configuration
├── data/
│   ├── train_dataset_mcq.csv        # MCQ training data
│   ├── train_dataset_saq.csv        # SAQ training data
│   ├── test_dataset_mcq.csv         # MCQ test data
│   ├── test_dataset_saq.csv         # SAQ test data
│   ├── mcq_prediction.tsv           # Generated MCQ predictions
│   └── saq_prediction.tsv           # Generated SAQ predictions
├── src/culture_questions_agent/
│   ├── ingestion/
│   │   ├── ingest.py                # Main ingestion pipeline
│   │   ├── wikipedia.py             # Wikipedia data loader
│   │   ├── wikivoyage.py            # Wikivoyage XML parser
│   │   ├── questions.py             # Training data reader
│   │   └── web.py                   # Web search & scraping
│   ├── predictor/
│   │   ├── discriminative_predictor.py  # NLL-based prediction
│   │   └── generative_predictor.py      # Text generation
│   ├── workflow.py                  # Event-driven QA workflow
│   ├── multi_retriever.py           # Multi-source retrieval orchestrator
│   ├── query_generator.py           # LLM query generation
│   ├── search_tools.py              # Web search integration
│   ├── inference.py                 # Competition submission generation
│   ├── evaluate.py                  # MLflow evaluation
│   ├── data.py                      # Data loading utilities
│   └── structures.py                # Data structures
├── prompts/
│   ├── query_generation_prompt.jinja    # Query generation template
│   ├── mcq_prompt.jinja                 # MCQ answering template
│   └── saq_prompt.jinja                 # SAQ answering template
├── storage/lancedb/                 # Vector database storage
├── tracking/                        # MLflow tracking
│   ├── mlruns.sqlite                # Experiment database
│   └── artifacts/                   # Experiment artifacts
├── notebooks/                       # Data exploration
└── pyproject.toml                   # Poetry dependencies
```

## 🚀 Installation

### Prerequisites

- Python 3.11+
- Poetry
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd culture_questions_agent

# Install dependencies
poetry install

# Set cache directory (optional)
export HF_HOME=/path/to/cache
```

## ⚡ Quick Start

```bash
# 1. Build the knowledge base
poetry run python -m culture_questions_agent.ingestion.ingest

# 2. Evaluate on MCQ
poetry run python -m culture_questions_agent.evaluate task_type="mcq"

# 3. Generate competition submissions
poetry run python -m culture_questions_agent.inference task_type="mcq"

# 4. View results in MLflow
mlflow ui --backend-store-uri sqlite:///tracking/mlruns.sqlite
```

## 🏗️ Architecture

### System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 1: DATA INGESTION                      │
└─────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
            Wikipedia      Wikivoyage    Training Data
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
                      Section-Aware Parsing
                                  ▼
                      Semantic Chunking (256 tokens)
                                  ▼
                    ┌─────────────────────────┐
                    │  LanceDB Vector Store   │
                    │  • Dense: BGE-M3        │
                    │  • Sparse: BM25         │
                    └─────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 2: QUESTION ANSWERING                      │
└─────────────────────────────────────────────────────────────────┘

Input Question (MCQ/SAQ)
         │
         ▼
┌─────────────────────────┐
│  1. Query Generation    │ ← Llama-3-8B (optional)
│  • Semantic expansion   │
│  • Or direct question   │
└─────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  2. Multi-Retriever Orchestration (Parallel) │
│  ┌────────────┬────────────┬─────────────┐  │
│  │ Wikipedia  │ Web Search │ Train Data  │  │
│  │ • Dense    │ • DDGS     │ • Dense     │  │
│  │ • Sparse   │            │ • Sparse    │  │
│  └────────────┴────────────┴─────────────┘  │
└──────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  3. Deduplication       │
│  • By content hash      │
│  • Preserve top-k       │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  4. Grouped Reranking               │
│  • Web + Wiki: top-6 (ColBERT)      │
│  • Training Data: top-4 (BGE v2-m3) │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  5. Prediction          │
│  • MCQ: Logits          │
│  • SAQ: Generate        │
└─────────────────────────┘
         │
         ▼
      Answer
```

### Key Components

1. **Ingestion Pipeline** ([`src/culture_questions_agent/ingestion/`](src/culture_questions_agent/ingestion/))
   - Wikipedia section parser with metadata extraction
   - Wikivoyage XML dump processor
   - Web search with query generation
   - Training data indexing

2. **Retrieval System** ([`multi_retriever.py`](src/culture_questions_agent/multi_retriever.py))
   - Multi-source parallel retrieval
   - Hybrid search (dense + sparse)
   - Content-based deduplication
   - Fusion strategies

3. **Reranking** ([`workflow.py`](src/culture_questions_agent/workflow.py))
   - ColBERT (late-interaction) for Wikipedia + Web Search
   - BGE-reranker-v2-m3 (cross-encoder) for Training Data
   - Grouped reranking by source

4. **Prediction** ([`src/culture_questions_agent/predictor/`](src/culture_questions_agent/predictor/))
   - Discriminative: Logits-based (for MCQ)
   - Generative: Text generation (for SAQ)

## 📖 Usage

### 1. Build the Index

Build the RAG knowledge base from multiple sources:

```bash
poetry run python -m culture_questions_agent.ingestion.ingest
```

**What happens:**
- Downloads Wikipedia pages for 4 countries (China, Iran, UK, US)
- Parses Wikivoyage XML dump
- Performs web searches based on training questions
- Creates semantic chunks (256 tokens, 50 overlap)
- Builds hybrid indices (Dense + Sparse)
- Saves to [`storage/lancedb/`](storage/lancedb/)

**Configuration:** Edit [`conf/config.yaml`](conf/config.yaml) to customize:
- `ingestion.country_filter_list`: Countries to include
- `ingestion.topic_templates`: Wikipedia page templates
- `ingestion.chunk_size`: Chunk size for splitting

### 2. Run Evaluation

Evaluate the system on training data:

**MCQ Evaluation:**
```bash
poetry run python -m culture_questions_agent.evaluate task_type="mcq"
```

**SAQ Evaluation:**
```bash
poetry run python -m culture_questions_agent.evaluate task_type="saq"
```

**Output:**
- Overall accuracy metrics
- Per-country accuracy breakdown
- MLflow experiment tracking

**View Results:**
```bash
mlflow ui --backend-store-uri sqlite:///tracking/mlruns.sqlite
# Open http://localhost:5000
```

### 3. Generate Competition Submissions

Generate prediction files for test datasets:

**MCQ Predictions:**
```bash
poetry run python -m culture_questions_agent.inference task_type="mcq"
# Output: data/mcq_prediction.tsv
```

**SAQ Predictions:**
```bash
poetry run python -m culture_questions_agent.inference task_type="saq"
# Output: data/saq_prediction.tsv
```

**Performance Optimization:**

Configure concurrency in [`conf/config.yaml`](conf/config.yaml):
```yaml
inference:
  max_concurrent: 10      # Concurrent predictions per process
  num_processes: 0        # Number of processes (0 = single process)
```

## ⚙️ Configuration

The system uses [Hydra](https://hydra.cc/) for configuration management. See [`conf/config.yaml`](conf/config.yaml) for all options.

### Key Configuration Sections

#### Model Settings
```yaml
model:
  llm_name: "meta-llama/Meta-Llama-3-8B"
  cache_dir: "/path/to/cache"
  reranker_name: "BAAI/bge-reranker-v2-m3"
  embedding_model_name: "BAAI/bge-m3"
  predictor_type: "generative"  # or "discriminative"
```

#### Retrieval Settings
```yaml
retrieval:
  use_colbert: true              # Late-interaction retrieval
  use_reranker: true             # Cross-encoder reranking
  use_wiki_retrieval: true       # Enable Wikipedia
  use_train_data_retrieval: true # Enable training data
  use_web_retrieval: true        # Enable web search
  num_queries: 3                 # Queries to generate
  use_direct_question: false     # Use question directly
```

#### Reranking Groups
```yaml
retrieval:
  reranking_groups:
    - sources: ["train_data"]
      top_k: 4
    - sources: ["wiki", "web"]
      top_k: 6
```

#### Ingestion Settings
```yaml
ingestion:
  chunk_size: 256
  chunk_overlap: 50
  skip_wiki: false
  skip_web: false
  skip_training_data: false
```

## 📊 Performance

**System Specifications:**
- Model: Meta-Llama-3-8B
- Embedding: BAAI/bge-m3
- Reranker: BAAI/bge-reranker-v2-m3
- Vector Store: LanceDB (hybrid search)

**Optimization Features:**
- Multi-process inference with configurable concurrency
- Async workflow for I/O operations
- Efficient batch processing
- GPU acceleration for embeddings and reranking

## 🔧 Development

### Project Layout

The project follows a modular architecture:

- **Ingestion** ([`src/culture_questions_agent/ingestion/`](src/culture_questions_agent/ingestion/)): Data loading and indexing
- **Retrieval** ([`multi_retriever.py`](src/culture_questions_agent/multi_retriever.py)): Multi-source retrieval orchestration
- **Workflow** ([`workflow.py`](src/culture_questions_agent/workflow.py)): Event-driven QA pipeline
- **Prediction** ([`src/culture_questions_agent/predictor/`](src/culture_questions_agent/predictor/)): Answer generation strategies

### Key Files

- [`workflow.py`](src/culture_questions_agent/workflow.py): Main QA workflow with 4 steps (query generation, retrieval, reranking, prediction)
- [`multi_retriever.py`](src/culture_questions_agent/multi_retriever.py): Orchestrates parallel retrieval from multiple sources
- [`query_generator.py`](src/culture_questions_agent/query_generator.py): LLM-based query expansion
- [`inference.py`](src/culture_questions_agent/inference.py): Competition submission generation
- [`evaluate.py`](src/culture_questions_agent/evaluate.py): MLflow-based evaluation

## 📝 License

MIT