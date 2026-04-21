"""Full RAG pipeline — index once, query many times.

Combines Indexer + Retriever into a single high-level interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import litellm

from pageindex_demo.config import Settings
from pageindex_demo.indexer import Indexer
from pageindex_demo.retriever import Retriever

logger = logging.getLogger(__name__)

_ANSWER_PROMPT = """\
You are a helpful assistant answering questions based on provided document excerpts.

Instructions:
- Answer concisely and accurately using only the provided context.
- Cite the section title in brackets when referencing specific content, e.g. [Introduction].
- If the context does not contain enough information, say so clearly.

Question:
{question}

Context (extracted document sections):
{context}
"""


class RAGPipeline:
    """End-to-end vectorless RAG: document → tree index → question answering.

    Example::

        pipeline = RAGPipeline()
        pipeline.load_document("report.pdf")
        answer = pipeline.ask("What are the key findings?")
        print(answer)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.indexer = Indexer(self.settings)
        self.retriever = Retriever(self.settings)
        self._tree: dict[str, Any] | None = None
        self._doc_name: str = ""

    # ── Document loading ──────────────────────────────────────────────────────

    def load_document(
        self,
        path: str | Path,
        *,
        force_reindex: bool = False,
    ) -> "RAGPipeline":
        """Index a PDF or Markdown document and make it ready for Q&A.

        Args:
            path:           Path to the document (PDF or .md).
            force_reindex:  Re-build the index even if a cached version exists.

        Returns:
            Self, for method chaining.
        """
        path = Path(path).resolve()
        stem = path.stem

        if not force_reindex:
            try:
                self._tree = self.indexer.load_tree(stem)
                self._doc_name = stem
                logger.info("Loaded cached index for '%s'.", stem)
                return self
            except FileNotFoundError:
                pass

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            self._tree = self.indexer.index_pdf(path)
        elif suffix in (".md", ".markdown"):
            self._tree = self.indexer.index_markdown(path)
        else:
            raise ValueError(f"Unsupported file type '{suffix}'. Use .pdf or .md")

        self._doc_name = stem
        return self

    def load_tree(self, name: str) -> "RAGPipeline":
        """Load a previously built index by document stem name."""
        self._tree = self.indexer.load_tree(name)
        self._doc_name = name
        return self

    # ── Q&A ───────────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        *,
        top_k: int = 5,
        return_sources: bool = False,
    ) -> str | dict[str, Any]:
        """Answer a natural-language question about the loaded document.

        Args:
            question:       The user's question.
            top_k:          Max number of document sections to retrieve.
            return_sources: When True, return a dict with ``answer`` and
                            ``sources`` (the retrieved nodes).

        Returns:
            Answer string, or a dict when *return_sources* is True.
        """
        if self._tree is None:
            raise RuntimeError("No document loaded. Call load_document() first.")

        nodes = self.retriever.search(question, self._tree, top_k=top_k)
        if not nodes:
            answer = "I could not find relevant sections in the document to answer this question."
            if return_sources:
                return {"answer": answer, "sources": []}
            return answer

        context = self.retriever.get_context(nodes)
        answer = self._generate_answer(question, context)

        if return_sources:
            return {
                "answer": answer,
                "sources": [
                    {"id": n.get("id"), "title": n.get("title")} for n in nodes
                ],
            }
        return answer

    def chat(self) -> None:
        """Start an interactive Q&A session in the terminal."""
        if self._tree is None:
            raise RuntimeError("No document loaded. Call load_document() first.")

        print(f"\nDocument: {self._doc_name}")
        print("Type your question and press Enter. Type 'quit' or 'exit' to stop.\n")

        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not question:
                continue
            if question.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break

            result = self.ask(question, return_sources=True)
            print(f"\nAssistant: {result['answer']}")
            if result["sources"]:
                titles = ", ".join(s["title"] for s in result["sources"] if s.get("title"))
                print(f"Sources: {titles}\n")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_answer(self, question: str, context: str) -> str:
        prompt = _ANSWER_PROMPT.format(question=question, context=context)
        response = litellm.completion(
            model=self.settings.index_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
