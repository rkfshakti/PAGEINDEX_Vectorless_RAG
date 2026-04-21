"""Retriever — vectorless tree search driven by LLM reasoning.

Instead of computing cosine similarity over embedding vectors, the retriever
asks the LLM to navigate the PageIndex tree — just like a human expert scans
a table of contents to find relevant sections.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import litellm

from pageindex_demo.config import Settings
from pageindex_demo.engine.tree_utils import create_node_mapping, remove_fields

logger = logging.getLogger(__name__)

_TREE_SEARCH_PROMPT = """\
You are an expert document navigator.

You are given:
  1. A user question.
  2. A hierarchical tree representing a document's structure.
     Each node has an "id", "title", and "summary". Leaf nodes also carry text.

Your task:
  Identify ALL node IDs whose content is likely to contain information
  needed to answer the question. Prefer specificity — pick the deepest
  relevant nodes rather than their ancestors when possible.

Return ONLY valid JSON in this exact schema (no markdown fences):
{{
  "thinking": "<brief chain-of-thought reasoning>",
  "node_list": ["node_id_1", "node_id_2"]
}}

Question:
{question}

Document tree (structure only, no full text):
{tree_structure}
"""


class Retriever:
    """Navigates a PageIndex tree to find sections relevant to a query."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.settings.apply_to_environment()

    # ── Public API ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        tree: dict[str, Any],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the most relevant nodes for *query* via LLM tree-search.

        Args:
            query:  Natural-language question.
            tree:   PageIndex tree dict (produced by Indexer).
            top_k:  Maximum number of nodes to return.

        Returns:
            List of node dicts with id, title, summary, and text.
        """
        node_map = create_node_mapping(tree)

        # Send only the skeleton (no full text) to the LLM — keeps prompt compact.
        tree_skeleton = remove_fields(
            json.loads(json.dumps(tree)),  # deep copy
            fields=["text"],
        )

        node_ids = self._llm_tree_search(query, tree_skeleton)

        results: list[dict[str, Any]] = []
        for nid in node_ids[:top_k]:
            node = node_map.get(nid)
            if node:
                results.append(node)
            else:
                logger.warning("LLM returned unknown node id '%s'; skipping.", nid)

        logger.info(
            "Tree search — query: '%s' | %d nodes selected",
            query[:60],
            len(results),
        )
        return results

    def get_context(self, nodes: list[dict[str, Any]]) -> str:
        """Concatenate node texts into a single context string for the LLM."""
        parts: list[str] = []
        for node in nodes:
            title = node.get("title", "")
            text = node.get("text", "").strip()
            if text:
                parts.append(f"[{title}]\n{text}")
        return "\n\n---\n\n".join(parts)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _llm_tree_search(
        self,
        query: str,
        tree_skeleton: dict[str, Any],
    ) -> list[str]:
        prompt = _TREE_SEARCH_PROMPT.format(
            question=query,
            tree_structure=json.dumps(tree_skeleton, indent=2),
        )

        response = litellm.completion(
            model=self.settings.search_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()

        # Strip accidental markdown fences.
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            parsed = json.loads(raw)
            thinking = parsed.get("thinking", "")
            node_ids: list[str] = parsed.get("node_list", [])
            if thinking:
                logger.debug("LLM reasoning: %s", thinking)
            return node_ids
        except json.JSONDecodeError:
            logger.error("Tree-search LLM returned invalid JSON:\n%s", raw)
            return []
