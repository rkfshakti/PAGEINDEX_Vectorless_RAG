"""Example 1 — Build a PageIndex tree from a PDF or Markdown document.

Run:
    python examples/01_index_document.py data/your_report.pdf
    python examples/01_index_document.py data/notes.md

The tree is saved to results/<doc_stem>_index.json and reused in later examples.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from the repo root without pip-installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pageindex_demo.config import Settings
from pageindex_demo.indexer import Indexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a document with PageIndex.")
    parser.add_argument("document", help="Path to a PDF or Markdown file.")
    parser.add_argument("--force", action="store_true", help="Re-index even if cached.")
    args = parser.parse_args()

    settings = Settings.from_env()
    print(f"\nEndpoint : {settings.llm_base_url}")
    print(f"Model    : {settings.index_model}")
    print(f"Local    : {settings.is_local}\n")

    indexer = Indexer(settings)
    doc_path = Path(args.document)

    suffix = doc_path.suffix.lower()
    if suffix == ".pdf":
        tree = indexer.index_pdf(doc_path)
    elif suffix in (".md", ".markdown"):
        tree = indexer.index_markdown(doc_path)
    else:
        print(f"Error: unsupported file type '{suffix}'.", file=sys.stderr)
        sys.exit(1)

    # Pretty-print the top-level structure.
    print("\nDocument tree (top 2 levels):")
    print(json.dumps(_trim_tree(tree, depth=2), indent=2))
    print(f"\nFull index saved → {settings.results_dir / (doc_path.stem + '_index.json')}")


def _trim_tree(node: dict, depth: int) -> dict:
    """Return a shallow copy of the tree limited to *depth* levels."""
    if depth == 0:
        return {"id": node.get("id"), "title": node.get("title"), "...": "..."}
    trimmed = {k: v for k, v in node.items() if k != "children"}
    if "children" in node and depth > 0:
        trimmed["children"] = [_trim_tree(c, depth - 1) for c in node["children"]]
    return trimmed


if __name__ == "__main__":
    main()
