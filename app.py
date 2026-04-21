"""Streamlit front-end for PageIndex Vectorless RAG.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pageindex_demo.config import Settings
from pageindex_demo.engine.tree_utils import pretty_print_tree
from pageindex_demo.indexer import Indexer
from pageindex_demo.pipeline import RAGPipeline

logging.basicConfig(level=logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PageIndex Vectorless RAG",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Chat bubbles */
    .user-bubble {
        background: #1e3a5f;
        color: #e8f4fd;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 6px 0 6px 60px;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .assistant-bubble {
        background: #1a1a2e;
        color: #e8e8f0;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 6px 60px 6px 0;
        font-size: 0.95rem;
        line-height: 1.5;
        border-left: 3px solid #4a9eff;
    }
    .source-tag {
        display: inline-block;
        background: #2d2d4e;
        color: #8888cc;
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 10px;
        margin: 4px 3px 0 0;
    }
    .stat-card {
        background: #16213e;
        border: 1px solid #2d2d4e;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4a9eff;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #8888aa;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state defaults ────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "pipeline": None,
        "tree": None,
        "doc_name": "",
        "messages": [],          # list of {role, content, sources}
        "indexing": False,
        "indexed": False,
        "settings_changed": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _count_nodes(node: dict) -> int:
    if node is None:
        return 0
    return 1 + sum(_count_nodes(c) for c in node.get("children", []))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📑 PageIndex")
    st.caption("Vectorless RAG — no embeddings, no vector DB")
    st.divider()

    # ── LLM Configuration ────
    st.subheader("LLM Configuration")

    llm_base_url = st.text_input(
        "API Base URL",
        value=st.session_state.get("llm_base_url", Settings.from_env().llm_base_url),
        help="Any OpenAI-compatible endpoint: LM Studio, Ollama, vLLM, OpenAI, Azure…",
        key="llm_base_url",
    )
    llm_api_key = st.text_input(
        "API Key",
        value=st.session_state.get("llm_api_key", Settings.from_env().llm_api_key or "lm-studio"),
        type="password",
        help="Use any non-empty string for local servers",
        key="llm_api_key",
    )
    llm_model = st.text_input(
        "Model",
        value=st.session_state.get("llm_model", Settings.from_env().llm_model),
        help="Model name as reported by /v1/models",
        key="llm_model",
    )

    st.divider()

    # ── Advanced settings ────
    with st.expander("Advanced Settings"):
        toc_pages = st.slider("ToC scan pages", 5, 50, 20)
        max_pages = st.slider("Max pages / node", 1, 20, 10)
        max_tokens = st.slider("Max tokens / node", 5000, 40000, 20000, step=1000)
        top_k = st.slider("Sections to retrieve (top-k)", 1, 10, 5)

    st.divider()

    # ── Document upload ────
    st.subheader("Document")
    uploaded = st.file_uploader(
        "Upload PDF or Markdown",
        type=["pdf", "md", "markdown"],
        help="The document will be indexed once and cached for re-use.",
    )

    if uploaded:
        file_size_kb = round(len(uploaded.getvalue()) / 1024, 1)
        st.caption(f"📄 `{uploaded.name}` — {file_size_kb} KB")

        if st.button("🔍 Build Index", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.indexed = False
            st.session_state.indexing = True
            st.session_state.tree = None

            settings = Settings(
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                llm_model=llm_model,
                toc_check_pages=toc_pages,
                max_pages_per_node=max_pages,
                max_tokens_per_node=max_tokens,
                results_dir=Path("results"),
            )

            suffix = Path(uploaded.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)

            with st.spinner(f"Building vectorless index for **{uploaded.name}**…"):
                t0 = time.time()
                try:
                    indexer = Indexer(settings)
                    if suffix == ".pdf":
                        tree = indexer.index_pdf(tmp_path)
                    else:
                        tree = indexer.index_markdown(tmp_path)

                    # Copy tree to results under the original name
                    stem = Path(uploaded.name).stem
                    out = settings.results_dir / f"{stem}_index.json"
                    with open(out, "w", encoding="utf-8") as f:
                        json.dump(tree, f, indent=2, ensure_ascii=False)

                    pipeline = RAGPipeline(settings)
                    pipeline._tree = tree
                    pipeline._doc_name = stem

                    st.session_state.pipeline = pipeline
                    st.session_state.tree = tree
                    st.session_state.doc_name = stem
                    st.session_state.indexed = True
                    st.session_state.indexing = False
                    elapsed = round(time.time() - t0, 1)
                    st.success(f"Index built in {elapsed}s")
                except Exception as exc:
                    st.session_state.indexing = False
                    st.error(f"Indexing failed: {exc}")
                finally:
                    tmp_path.unlink(missing_ok=True)

    # Status badge
    if st.session_state.indexed:
        st.success(f"✅ Ready — `{st.session_state.doc_name}`")
    elif not uploaded:
        st.info("Upload a document to get started.")


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## 📑 PageIndex Vectorless RAG")
st.caption(
    "Document Q&A powered by LLM-driven tree search — **no embeddings, no vector database**."
)

tab_chat, tab_tree, tab_how = st.tabs(["💬 Chat", "🌳 Document Tree", "ℹ️ How It Works"])


# ── Tab: Chat ─────────────────────────────────────────────────────────────────
with tab_chat:
    if not st.session_state.indexed:
        st.markdown(
            """
<div style="text-align:center; padding: 60px 0; color: #666;">
    <div style="font-size:3rem;">📄</div>
    <div style="font-size:1.1rem; margin-top:12px;">Upload a document in the sidebar to start chatting</div>
    <div style="font-size:0.85rem; color:#555; margin-top:6px;">Supports PDF and Markdown files</div>
</div>""",
            unsafe_allow_html=True,
        )
    else:
        # Stats row
        tree = st.session_state.tree
        node_count = _count_nodes(tree) if tree else 0
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{node_count}</div>'
                '<div class="stat-label">Index Nodes</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{len(st.session_state.messages) // 2}</div>'
                '<div class="stat-label">Questions Asked</div></div>',
                unsafe_allow_html=True,
            )
        with col3:
            is_local = "api.openai.com" not in llm_base_url
            badge = "🟢 Local" if is_local else "🔵 Cloud"
            st.markdown(
                f'<div class="stat-card"><div class="stat-value" style="font-size:1.2rem">{badge}</div>'
                '<div class="stat-label">LLM Endpoint</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="user-bubble">🧑 {msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    sources_html = ""
                    if msg.get("sources"):
                        tags = "".join(
                            f'<span class="source-tag">📎 {s["title"]}</span>'
                            for s in msg["sources"]
                            if s.get("title")
                        )
                        sources_html = f"<div style='margin-top:8px'>{tags}</div>"
                    st.markdown(
                        f'<div class="assistant-bubble">🤖 {msg["content"]}{sources_html}</div>',
                        unsafe_allow_html=True,
                    )

        # Input
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("chat_form", clear_on_submit=True):
            cols = st.columns([5, 1])
            question = cols[0].text_input(
                "Ask anything about the document",
                placeholder="e.g. What are the key findings? Summarise section 2…",
                label_visibility="collapsed",
            )
            submitted = cols[1].form_submit_button("Send", type="primary", use_container_width=True)

        if submitted and question.strip():
            pipeline: RAGPipeline = st.session_state.pipeline
            settings = Settings(
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                llm_model=llm_model,
            )
            pipeline.settings = settings
            pipeline.retriever.settings = settings
            pipeline.retriever.settings.apply_to_environment()

            st.session_state.messages.append({"role": "user", "content": question})

            with st.spinner("Searching document tree…"):
                try:
                    result = pipeline.ask(question, top_k=top_k, return_sources=True)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result.get("sources", []),
                        }
                    )
                except Exception as exc:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Error: {exc}", "sources": []}
                    )
            st.rerun()

        if st.session_state.messages:
            if st.button("🗑️ Clear chat", use_container_width=False):
                st.session_state.messages = []
                st.rerun()


# ── Tab: Tree ─────────────────────────────────────────────────────────────────
with tab_tree:
    if not st.session_state.indexed or not st.session_state.tree:
        st.info("Index a document first to see its tree structure here.")
    else:
        tree = st.session_state.tree
        st.markdown(f"### Tree — `{st.session_state.doc_name}`")

        col_vis, col_raw = st.columns([2, 1])

        with col_vis:
            st.markdown("**Visual outline**")
            st.code(pretty_print_tree(tree), language=None)

        with col_raw:
            st.markdown("**Raw JSON** (first 120 lines)")
            raw = json.dumps(tree, indent=2, ensure_ascii=False)
            st.code("\n".join(raw.splitlines()[:120]), language="json")


# ── Tab: How It Works ─────────────────────────────────────────────────────────
with tab_how:
    st.markdown(
        """
## How PageIndex Vectorless RAG works

Traditional RAG **embeds every chunk** into a vector space and retrieves by cosine similarity.
PageIndex does something fundamentally different:

---

### Step 1 — Parse

The document (PDF or Markdown) is parsed into a flat list of sections, preserving
natural heading hierarchy and page boundaries.

---

### Step 2 — Build the Index (once per document)

An LLM reads the section list and organises it into a **hierarchical tree** —
like a smart table of contents.  Each node gets a title and summary.
The full text of every section is stored at its leaf node.

This tree is saved as a JSON file and **reused on every subsequent query**.

---

### Step 3 — Tree Search (every query)

When you ask a question, the LLM receives only the **tree skeleton** (titles +
summaries, no full text).  It reasons about which nodes are most likely to contain
the answer — just like a human expert scans chapter headings before reading.

The LLM returns a list of node IDs.

---

### Step 4 — Answer

The full text of the selected nodes is assembled as context.  The LLM generates
a grounded answer with section citations.

---

### Why it outperforms vector RAG

| | Vector RAG | PageIndex |
|---|---|---|
| Retrieval | Cosine similarity | LLM reasoning |
| Context | Fixed chunks | Whole sections |
| Requires embedding model | Yes | No |
| Requires vector DB | Yes | No |
| FinanceBench accuracy | 85–90% | **98.7%** |

---

### Works with any LLM

Set `LLM_BASE_URL` to any OpenAI-compatible endpoint:

```
Local  →  LM Studio / Ollama / vLLM / Jan
Cloud  →  OpenAI / Anthropic / Azure / Gemini (via LiteLLM)
```
"""
    )


