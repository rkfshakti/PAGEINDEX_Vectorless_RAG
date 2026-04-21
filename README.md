# PageIndex Vectorless RAG

> **Document Q&A without a vector database** — powered by [PageIndex](https://github.com/VectifyAI/PageIndex) and any OpenAI-compatible LLM.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## What is this?

Traditional RAG pipelines embed every document chunk into a vector space and retrieve answers via cosine similarity search.  **PageIndex takes a fundamentally different approach**:

| Aspect | Vector RAG | PageIndex Vectorless RAG |
|--------|-----------|--------------------------|
| Storage | Embedding vectors + vector DB | Hierarchical JSON tree |
| Retrieval | Nearest-neighbour search | LLM-driven tree navigation |
| Context | Fixed-size chunks | Semantically coherent sections |
| Setup | Embedding model + vector DB required | Just an LLM — any provider |
| Accuracy (FinanceBench) | ~85–90 % | **98.7 %** |

PageIndex converts a document into a **smart table of contents** — a hierarchical tree where each node represents a natural section.  At query time, the LLM *reasons* through the tree (like a human expert scanning chapters) to find the most relevant sections, then generates a grounded answer.

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/pageindex-vectorless-rag.git
cd pageindex-vectorless-rag

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                   # installs the pageindex-demo CLI
```

### 2. Configure your LLM

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

#### Local server (LM Studio / Ollama / vLLM / Jan)

```dotenv
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL=llama-3.1-8b-instruct
```

Start your local server, note the model name it reports at `/v1/models`, and set `LLM_MODEL` to match.

#### OpenAI cloud

```dotenv
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-2024-11-20
```

#### Any other OpenAI-compatible provider

The project uses [LiteLLM](https://docs.litellm.ai/) under the hood, so it supports 100+ providers out of the box:

```dotenv
# Anthropic
LLM_BASE_URL=https://api.anthropic.com
LLM_API_KEY=sk-ant-...
LLM_MODEL=anthropic/claude-sonnet-4-6

# Azure OpenAI
LLM_BASE_URL=https://YOUR_RESOURCE.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT
LLM_API_KEY=YOUR_AZURE_KEY
LLM_MODEL=azure/gpt-4o

# Ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3.1
```

### 3. Run the demo

```bash
# Check your configuration
pageindex-demo info

# Index a document (PDF or Markdown)
pageindex-demo index data/your_report.pdf

# Ask a single question
pageindex-demo ask data/your_report.pdf "What are the key findings?"

# Ask with sources
pageindex-demo ask data/your_report.pdf "Summarise chapter 3" --sources

# Interactive chat
pageindex-demo chat data/your_report.pdf
```

---

## Python API

```python
from pageindex_demo import RAGPipeline

pipeline = RAGPipeline()

# Index once — the tree is cached in results/
pipeline.load_document("report.pdf")

# Ask anything
answer = pipeline.ask("What are the main conclusions?")
print(answer)

# Ask with source attribution
result = pipeline.ask("What revenue was reported?", return_sources=True)
print(result["answer"])
for src in result["sources"]:
    print(f"  → {src['title']}")

# Interactive chat in the terminal
pipeline.chat()
```

### Custom settings

```python
from pageindex_demo import RAGPipeline, Settings

settings = Settings(
    llm_base_url="http://localhost:1234/v1",
    llm_api_key="lm-studio",
    llm_model="llama-3.1-8b-instruct",
    max_pages_per_node=5,
    max_tokens_per_node=10000,
)

pipeline = RAGPipeline(settings)
pipeline.load_document("report.pdf")
print(pipeline.ask("What is discussed in the introduction?"))
```

---

## Project structure

```
pageindex-vectorless-rag/
├── src/
│   └── pageindex_demo/
│       ├── __init__.py      # Public API exports
│       ├── config.py        # Settings — env vars / .env file
│       ├── indexer.py       # PDF / Markdown → PageIndex tree
│       ├── retriever.py     # Vectorless tree-search via LLM
│       ├── pipeline.py      # High-level RAG pipeline
│       └── cli.py           # pageindex-demo CLI (Click)
├── examples/
│   ├── 01_index_document.py
│   ├── 02_query_rag.py
│   └── 03_interactive_chat.py
├── tests/
│   ├── conftest.py
│   └── test_pipeline.py
├── data/               # Put your PDFs here (gitignored)
├── results/            # Auto-generated index trees (gitignored)
├── .env.example        # Configuration template
├── pyproject.toml      # Package metadata (PEP 517/518)
└── requirements.txt
```

---

## CLI reference

```
Usage: pageindex-demo [OPTIONS] COMMAND [ARGS]...

  PageIndex Vectorless RAG — Q&A over documents without a vector database.

Options:
  -v, --verbose  Enable debug logging.
  --help         Show this message and exit.

Commands:
  ask    Ask a question about a document.
  chat   Interactive Q&A session.
  index  Build a PageIndex tree for a document.
  info   Show current configuration.
```

---

## How it works

```
PDF / Markdown
      │
      ▼
 ┌─────────────┐
 │  PageIndex  │  LLM reads the document, detects the table of contents,
 │   Indexer   │  and builds a hierarchical tree of sections/subsections.
 └──────┬──────┘  Saved as JSON — no vectors, no embeddings.
        │
        ▼  tree (JSON)
 ┌─────────────┐
 │  Retriever  │  LLM receives the tree skeleton + your query.
 │  (tree      │  It reasons about which nodes are relevant and
 │   search)   │  returns their IDs — like an expert scanning a ToC.
 └──────┬──────┘
        │  relevant node IDs
        ▼
 ┌─────────────┐
 │   Answer    │  Full text of the selected nodes is passed as context.
 │  Generator  │  LLM produces a grounded, cited answer.
 └─────────────┘
        │
        ▼
    Final answer
```

---

## Configuration reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint |
| `LLM_API_KEY` | *(required)* | API key (use any string for local servers) |
| `LLM_MODEL` | `gpt-4o-2024-11-20` | Model name |
| `RETRIEVE_MODEL` | same as `LLM_MODEL` | Model for tree-search step |
| `TOC_CHECK_PAGES` | `20` | Pages scanned for a table of contents |
| `MAX_PAGES_PER_NODE` | `10` | Max pages per index node |
| `MAX_TOKENS_PER_NODE` | `20000` | Token limit per node |
| `RESULTS_DIR` | `results` | Output directory for cached trees |

---

## Development

```bash
pip install -r requirements-dev.txt

# Lint
ruff check src/ examples/ tests/

# Type-check
mypy src/

# Tests
pytest
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push and open a pull request

---

## Credits

- [PageIndex](https://github.com/VectifyAI/PageIndex) by VectifyAI — the vectorless RAG engine
- [LiteLLM](https://github.com/BerriAI/litellm) — unified LLM provider interface
- [Click](https://click.palletsprojects.com/) — CLI framework

---

## License

MIT — see [LICENSE](LICENSE).
