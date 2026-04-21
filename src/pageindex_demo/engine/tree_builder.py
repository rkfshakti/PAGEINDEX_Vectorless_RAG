"""Tree builder — uses an LLM to convert flat sections into a hierarchical tree.

This is the core of the vectorless approach: instead of embedding-based chunking,
the LLM reasons about document structure and produces a semantically coherent tree.

Inspired by VectifyAI/PageIndex (MIT License).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import litellm

from pageindex_demo.engine.parser import Section

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_BUILD_TREE_PROMPT = """\
You are an expert document analyst. You will receive a list of document sections
(each with a title and text excerpt) and you must organize them into a hierarchical
tree structure that mirrors the document's logical organization.

Sections:
{sections_json}

Return ONLY valid JSON matching this exact schema (no markdown, no explanation):
{{
  "id": "root",
  "title": "<document title>",
  "summary": "<1-2 sentence document overview>",
  "children": [
    {{
      "id": "<unique_id>",
      "title": "<section title>",
      "summary": "<1-2 sentence section summary>",
      "section_index": <original section index>,
      "children": [ ... ]
    }}
  ]
}}

Rules:
- Assign a unique short id to every node (e.g. "intro", "ch1", "ch1_2").
- "section_index" must match the 0-based index of the section in the input list.
  For parent nodes that span multiple sections, use the index of the first section.
- Nest subsections logically based on heading levels and content.
- Do NOT include a "text" field — it will be attached later.
- Limit nesting to 3 levels maximum.
"""

_SUMMARISE_NODE_PROMPT = """\
Write a concise 1-2 sentence summary of the following document section.
Return ONLY the summary text, nothing else.

Section title: {title}
Section text:
{text}
"""


# ── Public API ────────────────────────────────────────────────────────────────

def build_tree(
    sections: list[Section],
    model: str,
    *,
    add_summaries: bool = True,
) -> dict[str, Any]:
    """Convert a flat list of sections into a hierarchical PageIndex-style tree.

    Args:
        sections:      Parsed sections from parse_pdf() or parse_markdown().
        model:         LiteLLM model string (e.g. "openai/llama-3.1-8b").
        add_summaries: Whether to generate per-node summaries via LLM.

    Returns:
        Nested dict tree with id, title, summary, text, children per node.
    """
    if not sections:
        raise ValueError("No sections to build a tree from.")

    sections_for_prompt = [
        {"index": i, "title": s.title, "level": s.level, "text_preview": s.text[:300]}
        for i, s in enumerate(sections)
    ]

    logger.info("Building tree structure via LLM (%d sections)…", len(sections))
    tree = _llm_build_tree(sections_for_prompt, model)

    # Attach full text from the original sections to every leaf node.
    _attach_text(tree, sections)

    # Optionally generate richer summaries for each node.
    if add_summaries:
        logger.info("Generating node summaries…")
        _enrich_summaries(tree, model)

    return tree


# ── Private helpers ───────────────────────────────────────────────────────────

def _llm_build_tree(sections_for_prompt: list[dict], model: str) -> dict[str, Any]:
    prompt = _BUILD_TREE_PROMPT.format(
        sections_json=json.dumps(sections_for_prompt, indent=2)
    )
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    raw = _strip_fences(raw)

    try:
        tree = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM returned invalid JSON for tree; falling back to flat structure.")
        tree = _fallback_tree(sections_for_prompt)

    return tree


def _attach_text(node: dict[str, Any], sections: list[Section]) -> None:
    """Recursively attach section text to tree nodes."""
    idx = node.get("section_index")
    if idx is not None and 0 <= idx < len(sections):
        node["text"] = sections[idx].text
        node["page_start"] = sections[idx].page_start
        node["page_end"] = sections[idx].page_end
    else:
        node.setdefault("text", "")

    for child in node.get("children", []):
        _attach_text(child, sections)


def _enrich_summaries(node: dict[str, Any], model: str) -> None:
    """Regenerate summaries for nodes that have actual text content."""
    text = node.get("text", "").strip()
    if text and len(text) > 100:
        try:
            prompt = _SUMMARISE_NODE_PROMPT.format(
                title=node.get("title", ""),
                text=text[:2000],
            )
            resp = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            node["summary"] = resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.debug("Summary generation failed for '%s': %s", node.get("title"), exc)

    for child in node.get("children", []):
        _enrich_summaries(child, model)


def _fallback_tree(sections: list[dict]) -> dict[str, Any]:
    """Simple flat tree used when the LLM fails to produce valid JSON."""
    children = [
        {
            "id": f"node_{i}",
            "title": s["title"],
            "summary": "",
            "section_index": s["index"],
            "children": [],
        }
        for i, s in enumerate(sections)
    ]
    return {
        "id": "root",
        "title": "Document",
        "summary": "",
        "children": children,
    }


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if the LLM wraps its output."""
    if text.startswith("```"):
        parts = text.split("```")
        # parts[1] is the fenced block; strip language tag on first line.
        inner = parts[1]
        if inner and not inner.startswith("{"):
            inner = inner[inner.index("\n") + 1:] if "\n" in inner else inner
        return inner.strip()
    return text
