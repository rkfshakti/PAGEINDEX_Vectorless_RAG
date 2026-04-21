"""PageIndex Vectorless RAG Demo — no vector DB, pure LLM reasoning."""

from pageindex_demo.config import Settings
from pageindex_demo.indexer import Indexer
from pageindex_demo.retriever import Retriever
from pageindex_demo.pipeline import RAGPipeline

__version__ = "0.1.0"
__all__ = ["Settings", "Indexer", "Retriever", "RAGPipeline"]
