# Annual AI Research Report 2025

## Executive Summary

This report summarises the key developments in artificial intelligence research during 2025.
Large language models have continued to advance, with significant improvements in reasoning,
multimodal understanding, and factual accuracy. The vectorless retrieval paradigm, exemplified
by PageIndex, achieved state-of-the-art results on professional document benchmarks, surpassing
traditional embedding-based approaches.

## 1. Introduction

Artificial intelligence research in 2025 was marked by three defining themes:

1. **Reasoning over retrieval** — Models increasingly navigate structured knowledge rather than
   searching flat vector spaces.
2. **Efficient inference** — Quantised local models (4-bit, 8-bit GGUF) now rival cloud APIs for
   many document-QA tasks.
3. **Agentic pipelines** — LLMs are deployed as active agents that decompose tasks and call tools.

The following sections detail each theme with supporting data.

## 2. Vectorless Retrieval

### 2.1 Background

Traditional Retrieval-Augmented Generation (RAG) pipelines chunk documents into fixed-size
passages, embed them with a separate model, and store the vectors in a vector database.
At query time the nearest-neighbour chunks are retrieved and passed to the LLM as context.

This approach has several limitations:

- Chunk boundaries split semantically coherent passages.
- Embedding models introduce a second latency and cost point.
- Cosine similarity captures surface-form closeness, not logical relevance.

### 2.2 The PageIndex Approach

PageIndex converts a document into a hierarchical tree analogous to its table of contents.
Each node represents a natural section — chapter, section, or subsection — with a concise
summary and, optionally, the full text.

At retrieval time an LLM reasons over the tree skeleton to identify the nodes most likely
to contain the answer.  This mirrors how a domain expert rapidly locates relevant chapters
before reading in depth.

### 2.3 Performance

On FinanceBench — a benchmark of 150 financial document QA pairs — PageIndex achieved
**98.7 % accuracy**, compared with 85–90 % for the best vector-based baselines.

| Method | FinanceBench Accuracy |
|--------|-----------------------|
| BM25 + GPT-4 | 82 % |
| FAISS + GPT-4 | 88 % |
| PageIndex + GPT-4o | **98.7 %** |

## 3. Efficient Local Inference

### 3.1 Quantised Models

The release of Llama 3.1, Mistral 3, and Phi-4 in quantised GGUF format enabled high-quality
local inference on consumer hardware.  LM Studio and Ollama lowered the barrier to running
7B–13B models on laptops.

### 3.2 Local RAG

Combining PageIndex with a local LLM eliminates all external dependencies for document QA:

```
Document → PageIndex tree → Local LLM reasoning → Answer
```

This architecture is privacy-preserving, offline-capable, and cost-free at inference time.

## 4. Agentic Pipelines

### 4.1 Tool-Calling Frameworks

OpenAI Agents SDK, LangGraph, and CrewAI matured into production-grade frameworks for
multi-step agentic pipelines.  PageIndex integrates with these via a simple tool interface
that exposes `search(query)` as a callable.

### 4.2 Case Study: Financial Analysis

A tier-1 investment bank deployed a PageIndex-based agentic pipeline to process quarterly
earnings reports.  The system:

1. Ingests a PDF and builds the index in under two minutes.
2. Answers structured queries (revenue, EBITDA, guidance) with cited section references.
3. Generates an executive summary with no human intervention.

Analyst productivity improved by 60 % on initial document review tasks.

## 5. Conclusions

The 2025 AI landscape shifted decisively toward reasoning-first retrieval and local inference.
PageIndex demonstrated that removing the vector database from the RAG pipeline — replacing it
with LLM-driven tree navigation — both simplifies the architecture and improves accuracy.

Key takeaways:

- **No vector DB needed**: a hierarchical JSON tree and a capable LLM are sufficient.
- **Provider-agnostic**: any OpenAI-compatible endpoint (cloud or local) works out of the box.
- **Privacy-friendly**: fully local deployments are viable on modern consumer hardware.

## References

1. VectifyAI. *PageIndex: Document Index for Vectorless, Reasoning-based RAG*. GitHub, 2024.
2. Llamafile, LM Studio, Ollama — local inference runtimes, 2025.
3. FinanceBench: A New Financial Question Answering Benchmark. ICLR, 2025.
