"""Example 2 — Ask questions about an indexed document.

Run (after Example 1):
    python examples/02_query_rag.py data/your_report.pdf "What are the main conclusions?"
    python examples/02_query_rag.py data/notes.md "Summarise section 2" --sources
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pageindex_demo.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Q&A over an indexed document.")
    parser.add_argument("document", help="Path to the original PDF or Markdown file.")
    parser.add_argument("question", help="Your question.")
    parser.add_argument("--top-k", type=int, default=5, help="Max sections to retrieve.")
    parser.add_argument("--sources", action="store_true", help="Print source section titles.")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    pipeline.load_document(args.document)

    result = pipeline.ask(args.question, top_k=args.top_k, return_sources=True)

    print(f"\nQuestion: {args.question}")
    print(f"\nAnswer:\n{result['answer']}")

    if args.sources and result["sources"]:
        print("\nSections used:")
        for src in result["sources"]:
            print(f"  • {src['title']}")


if __name__ == "__main__":
    main()
