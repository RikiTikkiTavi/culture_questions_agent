# Cultural QA Agent - NLL-Based Search System

A cultural question answering system using LLM-based query generation, web search, reranking, and NLL-based prediction for multiple-choice questions.

## Features

✅ **LLM Query Generation**: Meta-Llama-3.1-8B generates contextual search queries  
✅ **Multi-Source Search**: Wikipedia + DDGS (multi-engine web search)  
✅ **Semantic Reranking**: BAAI/bge-reranker-v2-m3 selects top relevant results  
✅ **NLL-Based Prediction**: Direct model evaluation via negative log-likelihood  
✅ **Dual-Path Retrieval**: Question context + option verification strategies  
✅ **Event-Driven Architecture**: LlamaIndex Workflows  
✅ **Configuration Management**: Hydra with extensive experimental flags  
✅ **MLflow Integration**: SQLite tracking with autolog  
✅ **Jinja2 Templates**: Prompt templates in `prompts/`  

## Project Structure

```
.
├── conf/
│   └── config.yaml                  # Hydra configuration
├── data/
│   ├── test_dataset_mcq.csv         # MCQ evaluation dataset
│   ├── test_dataset_saq.csv         # SAQ evaluation dataset
│   ├── train_dataset_mcq.csv        # MCQ training dataset
│   └── train_dataset_saq.csv        # SAQ training dataset
├── prompts/
│   ├── query_generation_prompt.jinja  # LLM query generation template
│   ├── completion_prompt.jinja        # (Legacy) Completion template
│   └── mcq_prompt.jinja              # MCQ prompt template
├── src/culture_questions_agent/
│   ├── __init__.py
│   ├── query_generator.py           # LLM-based query generation
│   ├── search_tools.py              # Wikipedia + DDGS search
│   ├── reranker.py                  # BGE reranker for result selection
│   ├── nll_predictor.py             # NLL-based option prediction
│   ├── workflow.py                  # Dual-path retrieval workflow
│   ├── main.py                      # Main entry point
│   ├── data.py                      # Data loading utilities
│   ├── structures.py                # Data structures
│   ├── evaluate.py                  # Evaluation logic
│   └── utils.py                     # Utility functions
├── tracking/                        # MLflow artifacts and SQLite DB
└── pyproject.toml                   # Poetry dependencies
```

## Installation

```bash
poetry install
```

## Architecture Overview

### System Flow

```
MCQ Question
    ↓
[1. Query Generation]
    - Use question directly OR
    - Generate multiple queries using LLM
    ↓
[2. Dual-Path Search]
    - Path 1: Question context (DDGS web search)
    - Path 2: Option verification (Wikipedia definitions) [optional]
    ↓
[3. Reranking] [optional]
    - Score all snippets with BGE-reranker-v2-m3
    - Select top-k most relevant
    ↓
[4. NLL Prediction]
    - Evaluate each option: P(option | context + question)
    - Select option with minimal NLL loss
    ↓
Answer
```

### Key Components

1. **Query Generator** (`query_generator.py`)
   - Uses Llama-3.1-8B to generate contextual search queries
   - Shares model instance with NLL predictor (memory efficient)
   - Can be bypassed with `use_direct_question=true`

2. **Search Engine** (`search_tools.py`)
   - **Wikipedia**: Primary source for entity definitions
   - **DDGS**: Multi-engine web search aggregator (no API key required)
   - Configurable result length and title inclusion

3. **Reranker** (`reranker.py`)
   - BAAI/bge-reranker-v2-m3 for semantic relevance scoring
   - Selects top-k snippets instead of concatenating all results
   - Optional - can be disabled for faster inference

4. **NLL Predictor** (`nll_predictor.py`)
   - Direct HuggingFace transformers (AutoModelForCausalLM)
   - Computes P(option | context, question) via NLL loss
   - Lower loss = better fit

## Usage

### Basic Inference

```bash
poetry run python -m culture_questions_agent.main
```

### Run Evaluation

```bash
poetry run python -m culture_questions_agent.evaluate
```

Results are logged to MLflow SQLite database at `tracking/mlruns.sqlite`.

## Configuration

Edit `conf/config.yaml` for extensive experimental control:

### Model Configuration
```yaml
model:
  llm_name: "meta-llama/Llama-3.1-8B"      # LLM for query gen + NLL
  reranker_name: "BAAI/bge-reranker-v2-m3" # Semantic reranker
  cache_dir: "/path/to/.cache"             # HuggingFace cache
```

### Retrieval Configuration
```yaml
retrieval:
  # Query Generation
  num_queries: 3                    # Number of queries to generate per question
  use_direct_question: false        # true = use question directly, false = generate queries
  
  # Context Selection
  use_question_context: true        # Include question context in NLL prompt
  use_option_context: false         # Include option-specific context in NLL prompt
  
  # Search Parameters
  max_search_chars: 5000           # Max characters per search result
  max_web_search_results: 3        # Number of web results per query
  include_title: true              # Include page titles in snippets
  
  # Reranking
  use_reranker: true               # Enable semantic reranking
  reranker_top_k: 3                # Top results to keep after reranking
```

### MLflow Configuration
```yaml
mlflow:
  tracking_uri: "sqlite:///tracking/mlruns.sqlite"
  artifact_location: "tracking/artifacts"
```

### Experimental Combinations

**Zero-shot (no retrieval)**:
```yaml
use_question_context: false
use_option_context: false
```

**Question context only**:
```yaml
use_question_context: true
use_option_context: false
```

**Option verification only**:
```yaml
use_question_context: false
use_option_context: true
```

**Full dual-path**:
```yaml
use_question_context: true
use_option_context: true
```

### Override via CLI

```bash
poetry run python -m culture_questions_agent.main \
  retrieval.use_reranker=false \
  retrieval.use_direct_question=true
```

## MLflow Tracking

View experiments in the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///tracking/mlruns.sqlite
```

Then open http://localhost:5000

All runs automatically log:
- **Parameters**: Model names, all retrieval flags, reranker settings
- **Metrics**: Accuracy, sample counts
- **Artifacts**: Prediction results, evaluation CSVs
- **System Info**: Python version, Git commit (if available)

## Workflow Details

### Event-Driven Pipeline

```python
@step
async def generate_queries(self, ev: StartEvent) -> QueryGenerationEvent:
    """Generate search queries (or use direct question)"""
    
@step  
async def search_for_options(self, ev: QueryGenerationEvent) -> SearchEvent:
    """Execute dual-path search with optional reranking"""
    
@step
async def predict_with_nll(self, ev: SearchEvent) -> StopEvent:
    """Predict best option using NLL loss"""
```

### NLL Prediction Method

For each option, the system:
1. Builds prompt: `"Context: {context}\nQuestion: {question}\nAnswer: {option}"`
2. Computes token-level log probabilities
3. Calculates mean negative log-likelihood (NLL)
4. Selects option with **minimum NLL** (highest probability)

This approach leverages the model's internal probability distribution without requiring generation.

## Prompt Templates

Templates use Jinja2 syntax in `prompts/`:

**Query Generation** (`query_generation_prompt.jinja`):
```jinja
Generate {{ num_queries }} different search queries to find information about:
"{{ question }}"

Each query should be a concise search phrase (3-8 words).
Return only the queries, one per line.
```

**MCQ Format** (`mcq_prompt.jinja`):
```jinja
Context: {{ context }}

Question: {{ question }}

Options:
{% for key, text in options.items() %}
{{ key }}. {{ text }}
{% endfor %}

Answer: {{ answer }}
```
Key Dependencies

- **transformers**: HuggingFace models (Llama-3.1-8B, BGE-reranker-v2-m3)
- **llama-index-core**: Workflow orchestration
- **llama-index-tools-wikipedia**: Wikipedia search
- **ddgs**: Multi-engine web search (DuckDuckGo, Bing, etc.)
- **hydra-core**: Configuration management
- **mlflow**: Experiment tracking
- **jinja2**: Prompt templating

## Advantages Over RAG

1. ✅ **No Index Building**: No vector store maintenance
2. ✅ **Dynamic Information**: Fresh search results for each query
3. ✅ **Probabilistic Selection**: Direct model evaluation via NLL
4. ✅ **Flexible Retrieval**: Easy to swap/combine search sources
5. ✅ **Lower Memory**: No large embedding indices in memory
6. ✅ **Transparent Scoring**: NLL provides interpretable confidence scores

## 
## Development

### Add Custom Search Source

Edit `search_tools.py`:
```python
def search_custom_source(self, query: str) -> str:
    # Implement custom search logic
    return search_results
```

### Modify Query Generation Strategy

Edit `query_generator.py` to change the LLM prompt or extraction logic.

### Experiment Tracking

All configuration parameters are automatically logged to MLflow. Run experiments with different settings:

```bash
# Experiment 1: Direct question with reranking
poetry run python -m culture_questions_agent.evaluate \
  retrieval.use_direct_question=true \
  retrieval.use_reranker=true

# Experiment 2: Generated queries without reranking  
poetry run python -m culture_questions_agent.evaluate \
  retrieval.use_direct_question=false \
  retrieval.use_reranker=false
```

## License

MIT
