# Cultural QA Agent - Offline Hybrid RAG System

A complete offline hybrid RAG system for cultural question answering using LlamaIndex Workflows, Hydra configuration, and MLflow tracking.

## Features

✅ **Hybrid Retrieval**: BGE-M3 (dense) + BM25 (sparse) with reciprocal reranking  
✅ **Base Model Completion**: Meta-Llama-3-8B with few-shot prompting  
✅ **Event-Driven Architecture**: LlamaIndex Workflows  
✅ **Configuration Management**: Hydra with YAML configs  
✅ **Logging**: Structured logging throughout  
✅ **MLflow Integration**: Tracing and evaluation  
✅ **Jinja2 Templates**: Prompt templates in `prompts/`  

## Project Structure

```
.
├── conf/
│   └── config.yaml              # Hydra configuration
├── data/
│   ├── test_dataset_mcq.csv     # MCQ evaluation dataset
│   └── test_dataset_saq.csv     # SAQ evaluation dataset
├── prompts/
│   └── completion_prompt.jinja  # Prompt template
├── src/culture_questions_agent/
│   ├── __init__.py
│   ├── builder.py               # Index builder (Wikipedia → Vector DB)
│   ├── workflow.py              # LlamaIndex Workflow implementation
│   ├── main.py                  # Main entry point
│   ├── evaluation.py            # MLflow evaluation utilities
│   └── run_evaluation.py        # Evaluation runner script
└── pyproject.toml               # Poetry dependencies
```

## Installation

```bash
poetry install
```

## Usage

### 1. Build Knowledge Atlas

```bash
poetry run python src/culture_questions_agent/main.py
```

This will:
- Generate topics from templates using pycountry (all countries)
- Download Wikipedia pages for Culture, Music, and Cuisine
- Build hybrid index (BGE-M3 + BM25)
- Save to `./storage/`

### 2. Run Interactive Queries

The main script runs example queries with MLflow tracking:

```bash
poetry run python src/culture_questions_agent/main.py
```

### 3. Run Evaluation

Evaluate on test datasets:

```bash
poetry run python src/culture_questions_agent/run_evaluation.py
```

Results are logged to MLflow and saved as CSV files.

## Configuration

Edit `conf/config.yaml`:

```yaml
model:
  llm_name: "meta-llama/Llama-3.1-8B"
  embed_name: "BAAI/bge-m3"
  cache_dir: "/path/to/cache"

storage:
  persist_dir: "./storage"

retrieval:
  top_k: 5
  mode: "reciprocal_rerank"

target_topic_templates:
  - "Culture of {}"
  - "Music of {}"
  - "Cuisine of {}"
```

### Override Configuration

```bash
poetry run python src/culture_questions_agent/main.py \
  model.llm_name=meta-llama/Meta-Llama-3-70B \
  retrieval.top_k=10
```

## MLflow Tracking

View experiments:

```bash
mlflow ui
```

Then open http://localhost:5000

All runs include:
- **Parameters**: Model names, retrieval config
- **Metrics**: Sample count, answer lengths
- **Artifacts**: Predictions JSON, results CSV
- **Traces**: Retrieval and generation spans

## Prompt Templates

Templates are in `prompts/` as Jinja2 files. Edit `completion_prompt.jinja`:

```jinja
Context:
{% for part in context_parts %}
[{{ part.index }}] {{ part.text }}
{% endfor %}

Question: {{ query }}
Answer:
```

## Architecture

### Workflow Steps

1. **Retrieve** (`retrieve` step):
   - Hybrid search using QueryFusionRetriever
   - Returns top-k nodes from both retrievers
   - MLflow span tracking

2. **Generate** (`generate` step):
   - Render Jinja2 prompt template
   - Base model completion (few-shot)
   - Clean and return answer
   - MLflow span tracking

### Data Flow

```
Query → StartEvent → retrieve() → RetrievalEvent → generate() → StopEvent → Answer
```

## Development

### Add New Prompt Template

1. Create `prompts/my_template.jinja`
2. Update `workflow.py` to load the template:
   ```python
   template = self.jinja_env.get_template("my_template.jinja")
   ```

### Add New Metrics

Edit `evaluation.py` and add to `run_mlflow_evaluation()`:

```python
metrics["new_metric"] = calculate_new_metric(predictions)
mlflow.log_metrics(metrics)
```

## License

MIT
