"""Unit tests for config, retriever, and pipeline (no real LLM calls)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pageindex_demo.config import Settings
from pageindex_demo.engine.tree_utils import create_node_mapping, remove_fields, pretty_print_tree
from pageindex_demo.pipeline import RAGPipeline
from pageindex_demo.retriever import Retriever


# ── Settings ──────────────────────────────────────────────────────────────────

class TestSettings:
    def test_defaults_are_openai(self):
        s = Settings(
            llm_base_url="https://api.openai.com/v1",
            llm_api_key="sk-test",
            llm_model="gpt-4o",
        )
        assert not s.is_local
        assert s.index_model == "gpt-4o"

    def test_local_server_prefixes_model(self):
        s = Settings(
            llm_base_url="http://localhost:1234/v1",
            llm_api_key="lm-studio",
            llm_model="llama-3.1-8b",
        )
        assert s.is_local
        assert s.index_model == "openai/llama-3.1-8b"

    def test_already_prefixed_model_not_double_prefixed(self):
        s = Settings(
            llm_base_url="http://localhost:1234/v1",
            llm_api_key="key",
            llm_model="openai/llama-3.1-8b",
        )
        assert s.index_model == "openai/llama-3.1-8b"

    def test_retrieve_model_falls_back_to_llm_model(self):
        s = Settings(llm_model="gpt-4o", retrieve_model="")
        assert s.retrieve_model == "gpt-4o"

    def test_results_dir_created(self, tmp_path: Path):
        new_dir = tmp_path / "my_results"
        s = Settings(results_dir=new_dir)
        assert new_dir.exists()


# ── Tree utilities ────────────────────────────────────────────────────────────

class TestTreeUtils:
    def test_create_node_mapping(self, sample_tree):
        mapping = create_node_mapping(sample_tree)
        assert "root" in mapping
        assert "ch1" in mapping
        assert "ch2" in mapping
        assert "ch2_1" in mapping
        assert "ch3" in mapping

    def test_remove_fields(self, sample_tree):
        stripped = remove_fields(sample_tree, fields=["text"])
        mapping = create_node_mapping(stripped)
        for node in mapping.values():
            assert "text" not in node

    def test_pretty_print_tree(self, sample_tree):
        output = pretty_print_tree(sample_tree)
        assert "Sample Report" in output
        assert "Introduction" in output


# ── Retriever ─────────────────────────────────────────────────────────────────

class TestRetriever:
    def _make_retriever(self, node_ids: list[str]) -> Retriever:
        s = Settings(
            llm_base_url="http://localhost:1234/v1",
            llm_api_key="test",
            llm_model="test-model",
        )
        r = Retriever(s)
        r._llm_tree_search = MagicMock(return_value=node_ids)
        return r

    def test_search_returns_nodes(self, sample_tree):
        r = self._make_retriever(["ch3"])
        results = r.search("What are the conclusions?", sample_tree)
        assert len(results) == 1
        assert results[0]["id"] == "ch3"

    def test_search_unknown_node_skipped(self, sample_tree):
        r = self._make_retriever(["ch3", "does_not_exist"])
        results = r.search("question", sample_tree)
        assert all(n["id"] != "does_not_exist" for n in results)

    def test_get_context_joins_texts(self, sample_tree):
        r = self._make_retriever([])
        node_map = create_node_mapping(sample_tree)
        nodes = [node_map["ch1"], node_map["ch3"]]
        ctx = r.get_context(nodes)
        assert "introduction" in ctx.lower() or "Introduction" in ctx
        assert "40%" in ctx

    def test_top_k_respected(self, sample_tree):
        r = self._make_retriever(["ch1", "ch2", "ch3", "ch2_1"])
        results = r.search("anything", sample_tree, top_k=2)
        assert len(results) <= 2


# ── Pipeline ──────────────────────────────────────────────────────────────────

class TestRAGPipeline:
    def _make_pipeline(self, tmp_index: Path) -> RAGPipeline:
        s = Settings(
            llm_base_url="http://localhost:1234/v1",
            llm_api_key="test",
            llm_model="test-model",
            results_dir=tmp_index / "results",
        )
        return RAGPipeline(s)

    def test_load_tree_from_cache(self, tmp_index: Path):
        pipeline = self._make_pipeline(tmp_index)
        pipeline.load_tree("sample")
        assert pipeline._tree is not None
        assert pipeline._doc_name == "sample"

    def test_ask_without_document_raises(self, tmp_index: Path):
        pipeline = self._make_pipeline(tmp_index)
        with pytest.raises(RuntimeError, match="No document loaded"):
            pipeline.ask("any question")

    @patch("pageindex_demo.pipeline.RAGPipeline._generate_answer", return_value="mocked answer")
    def test_ask_returns_answer(self, mock_gen, tmp_index: Path, sample_tree):
        pipeline = self._make_pipeline(tmp_index)
        pipeline._tree = sample_tree
        pipeline._doc_name = "sample"
        pipeline.retriever._llm_tree_search = MagicMock(return_value=["ch3"])

        result = pipeline.ask("What are the findings?", return_sources=True)
        assert result["answer"] == "mocked answer"
        assert isinstance(result["sources"], list)

    @patch("pageindex_demo.pipeline.RAGPipeline._generate_answer", return_value="ok")
    def test_ask_with_no_nodes_returns_fallback(self, _mock, tmp_index: Path, sample_tree):
        pipeline = self._make_pipeline(tmp_index)
        pipeline._tree = sample_tree
        pipeline.retriever._llm_tree_search = MagicMock(return_value=[])

        answer = pipeline.ask("unrelated question")
        assert "could not find" in answer.lower()
