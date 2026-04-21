"""Example 3 — Interactive multi-turn chat over a document.

Run (after Example 1):
    python examples/03_interactive_chat.py data/your_report.pdf

Type questions at the prompt.  Type 'quit' or press Ctrl-C to exit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pageindex_demo.pipeline import RAGPipeline


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python 03_interactive_chat.py <document>")
        sys.exit(1)

    doc_path = sys.argv[1]
    pipeline = RAGPipeline()
    pipeline.load_document(doc_path)
    pipeline.chat()


if __name__ == "__main__":
    main()
