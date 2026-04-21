"""Document parsers — PDF and Markdown to a flat list of sections with text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Section:
    """A logical section extracted from a document."""

    title: str
    level: int  # heading depth: 0 = document root, 1 = H1, 2 = H2, …
    text: str
    page_start: int = 0
    page_end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── PDF parser ────────────────────────────────────────────────────────────────

def parse_pdf(path: str | Path) -> list[Section]:
    """Extract pages from a PDF and group them into sections.

    Uses PyMuPDF when available, falls back to PyPDF2.
    Each PDF page becomes a provisional section; the tree-builder LLM then
    merges and re-organises them into the final hierarchy.
    """
    path = Path(path)
    try:
        return _parse_pdf_pymupdf(path)
    except ImportError:
        return _parse_pdf_pypdf2(path)


def _parse_pdf_pymupdf(path: Path) -> list[Section]:
    import fitz  # PyMuPDF

    doc = fitz.open(str(path))
    sections: list[Section] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            sections.append(
                Section(
                    title=f"Page {page_num + 1}",
                    level=1,
                    text=text,
                    page_start=page_num + 1,
                    page_end=page_num + 1,
                )
            )
    doc.close()
    return sections


def _parse_pdf_pypdf2(path: Path) -> list[Section]:
    import PyPDF2

    sections: list[Section] = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                sections.append(
                    Section(
                        title=f"Page {i + 1}",
                        level=1,
                        text=text,
                        page_start=i + 1,
                        page_end=i + 1,
                    )
                )
    return sections


# ── Markdown parser ───────────────────────────────────────────────────────────

def parse_markdown(path: str | Path) -> list[Section]:
    """Parse a Markdown file into sections based on heading structure."""
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    return _split_by_headings(content)


def _split_by_headings(content: str) -> list[Section]:
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(heading_re.finditer(content))

    if not matches:
        # No headings — treat as single section.
        return [Section(title="Document", level=0, text=content.strip())]

    sections: list[Section] = []

    # Text before the first heading.
    preamble = content[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(title="Preamble", level=0, text=preamble))

    for idx, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.end() + 1
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        text = content[start:end].strip()
        sections.append(Section(title=title, level=level, text=text, page_start=idx + 1))

    return sections
