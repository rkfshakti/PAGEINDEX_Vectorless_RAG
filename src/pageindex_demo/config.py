"""Configuration management — reads from environment variables or .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """All runtime settings, driven purely by environment variables.

    Environment variables (set in .env or shell):
        LLM_BASE_URL   – Base URL of any OpenAI-compatible API endpoint.
                          Default: https://api.openai.com/v1
                          Local examples:
                            LM Studio  → http://localhost:1234/v1
                            Ollama     → http://localhost:11434/v1
                            vLLM       → http://localhost:8000/v1
        LLM_API_KEY    – API key. Use a dummy value for local servers.
        LLM_MODEL      – Model identifier served at that endpoint.
                          For local servers the value must match the model
                          name reported by /v1/models (e.g. "llama-3.1-8b").
        RETRIEVE_MODEL – Model used only for the tree-search step.
                          Falls back to LLM_MODEL when not set.
        TOC_CHECK_PAGES     – How many leading pages to scan for a ToC.
        MAX_PAGES_PER_NODE  – Max PDF pages collapsed into a single node.
        MAX_TOKENS_PER_NODE – Token ceiling per node.
        RESULTS_DIR         – Where index JSON files are stored.
    """

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-2024-11-20")
    )
    retrieve_model: str = field(
        default_factory=lambda: os.getenv("RETRIEVE_MODEL", "")
    )

    # ── PageIndex tuning ─────────────────────────────────────────────────────
    toc_check_pages: int = field(
        default_factory=lambda: int(os.getenv("TOC_CHECK_PAGES", "20"))
    )
    max_pages_per_node: int = field(
        default_factory=lambda: int(os.getenv("MAX_PAGES_PER_NODE", "10"))
    )
    max_tokens_per_node: int = field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS_PER_NODE", "20000"))
    )

    # ── Paths ────────────────────────────────────────────────────────────────
    results_dir: Path = field(
        default_factory=lambda: Path(os.getenv("RESULTS_DIR", "results"))
    )

    def __post_init__(self) -> None:
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if not self.retrieve_model:
            self.retrieve_model = self.llm_model

    # ── Helpers ──────────────────────────────────────────────────────────────

    @property
    def is_local(self) -> bool:
        """True when pointing at a local / self-hosted endpoint."""
        return "api.openai.com" not in self.llm_base_url

    def _litellm_model_name(self, model: str) -> str:
        """Prefix model name so litellm routes it to the right provider.

        For any non-OpenAI base URL we use the ``openai/`` prefix which tells
        litellm to treat the endpoint as an OpenAI-compatible server.
        """
        if self.is_local and not model.startswith(
            ("openai/", "anthropic/", "gemini/", "bedrock/")
        ):
            return f"openai/{model}"
        return model

    @property
    def index_model(self) -> str:
        return self._litellm_model_name(self.llm_model)

    @property
    def search_model(self) -> str:
        return self._litellm_model_name(self.retrieve_model)

    def apply_to_environment(self) -> None:
        """Push settings into env vars so PageIndex / litellm pick them up."""
        import litellm  # local import to keep startup fast

        os.environ["OPENAI_API_BASE"] = self.llm_base_url
        os.environ["OPENAI_API_KEY"] = self.llm_api_key or "not-needed"
        litellm.drop_params = True

    @classmethod
    def from_env(cls) -> "Settings":
        return cls()
