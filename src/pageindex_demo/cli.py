"""Command-line interface for the PageIndex Vectorless RAG demo."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from pageindex_demo.config import Settings
from pageindex_demo.pipeline import RAGPipeline


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """PageIndex Vectorless RAG — Q&A over documents without a vector database."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@cli.command("index")
@click.argument("document", type=click.Path(exists=True, dir_okay=False))
@click.option("--force", "-f", is_flag=True, help="Force re-index even if cached.")
@click.pass_context
def index_cmd(ctx: click.Context, document: str, force: bool) -> None:
    """Build a PageIndex tree for DOCUMENT (PDF or Markdown).

    The tree is saved to the results/ directory and can be reused for
    multiple queries without re-indexing.

    \b
    Examples:
      pageindex-demo index report.pdf
      pageindex-demo index notes.md --force
    """
    _setup_logging(ctx.obj.get("verbose", False))
    settings = Settings.from_env()

    click.echo(f"Indexing: {document}")
    click.echo(f"LLM endpoint: {settings.llm_base_url}")
    click.echo(f"Model: {settings.index_model}")

    pipeline = RAGPipeline(settings)
    pipeline.load_document(document, force_reindex=force)
    click.echo(f"Index ready → {settings.results_dir / (Path(document).stem + '_index.json')}")


@cli.command("ask")
@click.argument("document", type=click.Path(exists=True, dir_okay=False))
@click.argument("question")
@click.option("--top-k", default=5, show_default=True, help="Max sections to retrieve.")
@click.option("--sources", is_flag=True, help="Print source section titles.")
@click.option("--json-output", "json_out", is_flag=True, help="Output raw JSON.")
@click.pass_context
def ask_cmd(
    ctx: click.Context,
    document: str,
    question: str,
    top_k: int,
    sources: bool,
    json_out: bool,
) -> None:
    """Ask a QUESTION about DOCUMENT using vectorless retrieval.

    \b
    Examples:
      pageindex-demo ask report.pdf "What are the key findings?"
      pageindex-demo ask report.pdf "Summarise chapter 3" --sources
    """
    _setup_logging(ctx.obj.get("verbose", False))
    pipeline = RAGPipeline()
    pipeline.load_document(document)
    result = pipeline.ask(question, top_k=top_k, return_sources=True)

    if json_out:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"\n{result['answer']}\n")
        if sources and result["sources"]:
            click.echo("Sources:")
            for s in result["sources"]:
                click.echo(f"  • {s['title']}")


@cli.command("chat")
@click.argument("document", type=click.Path(exists=True, dir_okay=False))
@click.pass_context
def chat_cmd(ctx: click.Context, document: str) -> None:
    """Start an interactive Q&A session for DOCUMENT.

    \b
    Examples:
      pageindex-demo chat report.pdf
    """
    _setup_logging(ctx.obj.get("verbose", False))
    pipeline = RAGPipeline()
    pipeline.load_document(document)
    pipeline.chat()


@cli.command("info")
def info_cmd() -> None:
    """Show current configuration (endpoint, model, paths)."""
    s = Settings.from_env()
    click.echo("\n== PageIndex Vectorless RAG -- Configuration ==")
    click.echo(f"  LLM Base URL   : {s.llm_base_url}")
    click.echo(f"  Index model    : {s.index_model}")
    click.echo(f"  Search model   : {s.search_model}")
    click.echo(f"  Local server   : {'yes' if s.is_local else 'no'}")
    click.echo(f"  Results dir    : {s.results_dir}")
    click.echo(f"  ToC pages      : {s.toc_check_pages}")
    click.echo(f"  Max pages/node : {s.max_pages_per_node}")
    click.echo(f"  Max tokens/node: {s.max_tokens_per_node}\n")


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
