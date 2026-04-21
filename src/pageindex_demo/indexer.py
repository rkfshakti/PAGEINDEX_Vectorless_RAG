"""Indexer — converts PDF or Markdown documents into a PageIndex-style tree.

PageIndex builds a hierarchical tree (like a smart table of contents) entirely
through LLM reasoning.  No embeddings, no vector store, no chunking heuristics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pageindex_demo.config import Settings
from pageindex_demo.engine.parser import parse_pdf, parse_markdown
from pageindex_demo.engine.tree_builder import build_tree
from pageindex_demo.engine.tree_utils import pretty_print_tree

logger = logging.getLogger(__name__)


class Indexer:
    """Builds a vectorless PageIndex tree for PDF and Markdown documents."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.settings.apply_to_environment()

    # ── Public API ─────────────────────────────────────────────────────────────

    def index_pdf(self, pdf_path: str | Path) -> dict[str, Any]:
        """Build a PageIndex tree from a PDF file.

        Args:
            pdf_path: Path to a PDF document.

        Returns:
            Nested dict tree — each node has: id, title, summary, text, children.
        """
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("Parsing PDF: %s", pdf_path.name)
        sections = parse_pdf(pdf_path)
        logger.info("  → %d pages extracted", len(sections))

        logger.info("Building vectorless tree via LLM (model: %s)…", self.settings.index_model)
        tree = build_tree(sections, self.settings.index_model, add_summaries=True)

        self._save_tree(tree, pdf_path.stem)
        return tree

    def index_markdown(self, md_path: str | Path) -> dict[str, Any]:
        """Build a PageIndex tree from a Markdown file."""
        md_path = Path(md_path).resolve()
        if not md_path.exists():
            raise FileNotFoundError(f"Markdown not found: {md_path}")

        logger.info("Parsing Markdown: %s", md_path.name)
        sections = parse_markdown(md_path)
        logger.info("  → %d sections found", len(sections))

        logger.info("Building vectorless tree via LLM (model: %s)…", self.settings.index_model)
        tree = build_tree(sections, self.settings.index_model, add_summaries=True)

        self._save_tree(tree, md_path.stem)
        return tree

    def load_tree(self, name: str) -> dict[str, Any]:
        """Load a previously built tree from the results directory."""
        index_file = self.settings.results_dir / f"{name}_index.json"
        if not index_file.exists():
            raise FileNotFoundError(
                f"No cached index for '{name}'. "
                "Run index_pdf() or index_markdown() first."
            )
        with open(index_file, encoding="utf-8") as f:
            return json.load(f)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _save_tree(self, tree: dict[str, Any], stem: str) -> Path:
        out = self.settings.results_dir / f"{stem}_index.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)
        logger.info("Index saved → %s", out)
        logger.debug("\nTree structure:\n%s", pretty_print_tree(tree))
        return out
