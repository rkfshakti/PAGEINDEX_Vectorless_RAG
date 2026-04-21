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
    page_title="PageIndex · Vectorless RAG",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Master CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ═══════════════════════════════════════════════════════
   GLOBAL RESET & BASE
═══════════════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Kill Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
.stDeployButton { display: none; }

/* ═══════════════════════════════════════════════════════
   SCROLLBAR
═══════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0d1a; }
::-webkit-scrollbar-thumb { background: #2d2d5e; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4a4aff; }

/* ═══════════════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080818 0%, #0d0d25 100%) !important;
    border-right: 1px solid rgba(74, 74, 255, 0.15) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 0 1rem 0;
}
.sidebar-logo-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #4a4aff, #8b5cf6);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    box-shadow: 0 0 20px rgba(74,74,255,0.4);
}
.sidebar-logo-text { line-height: 1.2; }
.sidebar-logo-title {
    font-size: 1.05rem; font-weight: 700;
    background: linear-gradient(90deg, #ffffff, #a5b4fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sidebar-logo-sub { font-size: 0.68rem; color: #6b6ba0; letter-spacing: 0.06em; text-transform: uppercase; }

.sidebar-section-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a4aff;
    margin: 1.2rem 0 0.5rem 0;
    display: flex; align-items: center; gap: 6px;
}
.sidebar-section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(74,74,255,0.4), transparent);
}

/* Sidebar inputs */
[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(74,74,255,0.25) !important;
    border-radius: 8px !important;
    color: #e0e0ff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #4a4aff !important;
    box-shadow: 0 0 0 3px rgba(74,74,255,0.15) !important;
}
[data-testid="stSidebar"] label {
    color: #8080b0 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}

/* Slider */
[data-testid="stSidebar"] .stSlider [data-testid="stSliderThumb"] {
    background: #4a4aff !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stSliderTrack"] > div:first-child {
    background: rgba(255,255,255,0.08) !important;
}

/* Upload area */
[data-testid="stFileUploadDropzone"] {
    background: rgba(74,74,255,0.04) !important;
    border: 1.5px dashed rgba(74,74,255,0.35) !important;
    border-radius: 10px !important;
    transition: all 0.2s;
}
[data-testid="stFileUploadDropzone"]:hover {
    background: rgba(74,74,255,0.08) !important;
    border-color: rgba(74,74,255,0.6) !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #4a4aff, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em;
    padding: 0.55rem 1rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(74,74,255,0.3) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(74,74,255,0.5) !important;
}

/* Expander */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(74,74,255,0.15) !important;
    border-radius: 10px !important;
}

/* ═══════════════════════════════════════════════════════
   HERO HEADER
═══════════════════════════════════════════════════════ */
.hero {
    background: linear-gradient(135deg, #080818 0%, #0f0f2e 50%, #0d0828 100%);
    border: 1px solid rgba(74,74,255,0.2);
    border-radius: 20px;
    padding: 2.5rem 2.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(74,74,255,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(139,92,246,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(74,74,255,0.12);
    border: 1px solid rgba(74,74,255,0.3);
    color: #a5b4fc;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
}
.hero-badge-dot {
    width: 6px; height: 6px;
    background: #4a4aff;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0 0 0.6rem 0;
    background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 70%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    font-size: 1rem;
    color: #7070a0;
    max-width: 580px;
    line-height: 1.6;
}
.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 1.4rem;
}
.hero-pill {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: #9090c0;
    font-size: 0.76rem;
    padding: 4px 12px;
    border-radius: 20px;
    display: inline-flex; align-items: center; gap: 5px;
}
.hero-pill-accent { color: #a5b4fc; }

/* ═══════════════════════════════════════════════════════
   TABS
═══════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #6060a0 !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(74,74,255,0.25), rgba(139,92,246,0.2)) !important;
    color: #e0e0ff !important;
    box-shadow: 0 2px 8px rgba(74,74,255,0.2) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ═══════════════════════════════════════════════════════
   STAT CARDS
═══════════════════════════════════════════════════════ */
.stat-grid { display: flex; gap: 14px; margin-bottom: 1.5rem; }
.stat-card {
    flex: 1;
    background: linear-gradient(135deg, rgba(14,14,35,0.9), rgba(20,20,50,0.9));
    border: 1px solid rgba(74,74,255,0.18);
    border-radius: 14px;
    padding: 16px 20px;
    display: flex; align-items: center; gap: 14px;
    transition: border-color 0.2s, transform 0.2s;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #4a4aff, #8b5cf6);
    opacity: 0.7;
}
.stat-card:hover { border-color: rgba(74,74,255,0.4); transform: translateY(-1px); }
.stat-icon {
    width: 44px; height: 44px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    flex-shrink: 0;
}
.stat-icon-blue  { background: rgba(74,74,255,0.15); }
.stat-icon-purple{ background: rgba(139,92,246,0.15); }
.stat-icon-green { background: rgba(34,197,94,0.15); }
.stat-icon-amber { background: rgba(245,158,11,0.15); }
.stat-body { min-width: 0; }
.stat-value {
    font-size: 1.65rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ffffff, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.stat-label {
    font-size: 0.72rem;
    color: #5050a0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-top: 3px;
}

/* ═══════════════════════════════════════════════════════
   CHAT
═══════════════════════════════════════════════════════ */
.chat-wrapper {
    background: rgba(8,8,24,0.6);
    border: 1px solid rgba(74,74,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    min-height: 300px;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}
.chat-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 1rem;
    text-align: center;
    color: #404068;
}
.chat-empty-icon {
    font-size: 3rem;
    margin-bottom: 0.8rem;
    opacity: 0.5;
}
.chat-empty-title { font-size: 1rem; font-weight: 600; color: #5050a0; margin-bottom: 0.3rem; }
.chat-empty-sub { font-size: 0.82rem; color: #383860; }

.msg-row {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    margin-bottom: 1.2rem;
    animation: fadeSlideIn 0.3s ease;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.msg-row-user  { flex-direction: row-reverse; }
.msg-avatar {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.msg-avatar-user { background: linear-gradient(135deg, #1e3a8a, #3b82f6); }
.msg-avatar-ai   { background: linear-gradient(135deg, #4a4aff, #8b5cf6); box-shadow: 0 0 12px rgba(74,74,255,0.35); }
.msg-body { max-width: 76%; }
.msg-bubble {
    padding: 12px 16px;
    border-radius: 16px;
    font-size: 0.91rem;
    line-height: 1.65;
    word-break: break-word;
}
.msg-bubble-user {
    background: linear-gradient(135deg, #1e3a8a, #1d4ed8);
    color: #e8f0fe;
    border-radius: 16px 16px 4px 16px;
    box-shadow: 0 4px 15px rgba(29,78,216,0.25);
}
.msg-bubble-ai {
    background: linear-gradient(135deg, rgba(14,14,40,0.95), rgba(20,20,55,0.95));
    color: #d0d0f0;
    border-radius: 16px 16px 16px 4px;
    border: 1px solid rgba(74,74,255,0.2);
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.msg-meta {
    font-size: 0.68rem;
    color: #3a3a6a;
    margin-top: 4px;
    padding: 0 4px;
}
.msg-row-user .msg-meta { text-align: right; }
.sources-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(74,74,255,0.12);
}
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(74,74,255,0.1);
    border: 1px solid rgba(74,74,255,0.25);
    color: #8888d0;
    font-size: 0.71rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    transition: all 0.15s;
}
.source-chip:hover { background: rgba(74,74,255,0.2); color: #b0b0ff; }

/* Chat input area */
.chat-input-area {
    background: rgba(10,10,28,0.8);
    border: 1px solid rgba(74,74,255,0.2);
    border-radius: 14px;
    padding: 1rem;
    backdrop-filter: blur(10px);
}

/* Streamlit form/input inside chat area */
.stForm { border: none !important; padding: 0 !important; }
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(74,74,255,0.3) !important;
    border-radius: 10px !important;
    color: #e0e0ff !important;
    font-size: 0.92rem !important;
    padding: 0.65rem 1rem !important;
    transition: all 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #4a4aff !important;
    box-shadow: 0 0 0 3px rgba(74,74,255,0.15) !important;
}
.stTextInput > div > div > input::placeholder { color: #404070 !important; }

/* Send button */
.stFormSubmitButton > button, .stButton > button {
    background: linear-gradient(135deg, #4a4aff, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(74,74,255,0.3) !important;
}
.stFormSubmitButton > button:hover, .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(74,74,255,0.5) !important;
}

/* Clear chat button — secondary style */
button[kind="secondary"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #6060a0 !important;
    font-size: 0.8rem !important;
    box-shadow: none !important;
}
button[kind="secondary"]:hover {
    background: rgba(255,60,60,0.1) !important;
    border-color: rgba(255,60,60,0.3) !important;
    color: #ff8080 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ═══════════════════════════════════════════════════════
   EMPTY / LANDING STATE
═══════════════════════════════════════════════════════ */
.landing {
    text-align: center;
    padding: 4rem 2rem;
    background: rgba(8,8,24,0.4);
    border: 1px dashed rgba(74,74,255,0.2);
    border-radius: 20px;
}
.landing-icon { font-size: 4rem; margin-bottom: 1rem; opacity: 0.6; }
.landing-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #4040a0;
    margin-bottom: 0.5rem;
}
.landing-steps {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 2rem;
    flex-wrap: wrap;
}
.landing-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    max-width: 130px;
}
.landing-step-num {
    width: 32px; height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4a4aff, #8b5cf6);
    color: white;
    font-size: 0.85rem;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 12px rgba(74,74,255,0.4);
}
.landing-step-text { font-size: 0.78rem; color: #3a3a70; text-align: center; }

/* ═══════════════════════════════════════════════════════
   TREE VIEW
═══════════════════════════════════════════════════════ */
.tree-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(74,74,255,0.12);
}
.tree-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e0e0ff;
}
.tree-doc-badge {
    background: rgba(74,74,255,0.12);
    border: 1px solid rgba(74,74,255,0.25);
    color: #8888d0;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
}

.tree-node {
    background: rgba(10,10,30,0.6);
    border: 1px solid rgba(74,74,255,0.12);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    transition: border-color 0.2s;
}
.tree-node:hover { border-color: rgba(74,74,255,0.35); }
.tree-node-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #4a4aff;
    font-weight: 600;
}
.tree-node-title { font-size: 0.9rem; font-weight: 600; color: #c0c0ff; margin: 2px 0; }
.tree-node-summary { font-size: 0.78rem; color: #5050a0; line-height: 1.5; }
.tree-children { margin-left: 20px; border-left: 2px solid rgba(74,74,255,0.15); padding-left: 12px; }

/* Code blocks */
.stCodeBlock > div {
    background: rgba(8,8,24,0.8) !important;
    border: 1px solid rgba(74,74,255,0.15) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ═══════════════════════════════════════════════════════
   HOW IT WORKS
═══════════════════════════════════════════════════════ */
.flow-container {
    display: flex;
    align-items: stretch;
    gap: 0;
    margin: 2rem 0;
    flex-wrap: wrap;
}
.flow-step {
    flex: 1;
    min-width: 180px;
    background: linear-gradient(135deg, rgba(12,12,35,0.9), rgba(18,18,50,0.9));
    border: 1px solid rgba(74,74,255,0.15);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    position: relative;
    transition: transform 0.2s, border-color 0.2s;
}
.flow-step:hover { transform: translateY(-3px); border-color: rgba(74,74,255,0.4); }
.flow-step-num {
    width: 30px; height: 30px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4a4aff, #8b5cf6);
    color: white;
    font-size: 0.8rem;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 0.8rem;
    box-shadow: 0 0 12px rgba(74,74,255,0.4);
}
.flow-step-icon { font-size: 1.6rem; margin-bottom: 0.5rem; }
.flow-step-title { font-size: 0.9rem; font-weight: 700; color: #c0c0ff; margin-bottom: 0.4rem; }
.flow-step-desc { font-size: 0.78rem; color: #5050a0; line-height: 1.55; }
.flow-arrow {
    display: flex;
    align-items: center;
    padding: 0 8px;
    color: rgba(74,74,255,0.4);
    font-size: 1.3rem;
    flex-shrink: 0;
}

.compare-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    margin: 1.5rem 0;
    border: 1px solid rgba(74,74,255,0.15);
}
.compare-table th {
    background: rgba(74,74,255,0.12);
    color: #a5b4fc;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 12px 16px;
    text-align: left;
    border-bottom: 1px solid rgba(74,74,255,0.2);
}
.compare-table td {
    background: rgba(8,8,24,0.6);
    color: #9090c0;
    font-size: 0.85rem;
    padding: 11px 16px;
    border-bottom: 1px solid rgba(74,74,255,0.07);
}
.compare-table tr:last-child td { border-bottom: none; }
.compare-table td.win {
    color: #86efac;
    font-weight: 600;
}
.compare-table td.lose { color: #f87171; }

.provider-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 10px;
    margin-top: 1rem;
}
.provider-card {
    background: rgba(10,10,30,0.6);
    border: 1px solid rgba(74,74,255,0.12);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    transition: all 0.2s;
}
.provider-card:hover { border-color: rgba(74,74,255,0.35); transform: translateY(-2px); }
.provider-card-icon { font-size: 1.4rem; margin-bottom: 4px; }
.provider-card-name { font-size: 0.75rem; color: #6060a0; font-weight: 600; }
.provider-card-type { font-size: 0.65rem; color: #3a3a60; margin-top: 2px; }

/* Status messages */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 3px !important;
}

/* Dividers */
hr { border-color: rgba(74,74,255,0.1) !important; margin: 1rem 0 !important; }

/* Metric */
[data-testid="metric-container"] {
    background: rgba(10,10,30,0.6) !important;
    border: 1px solid rgba(74,74,255,0.12) !important;
    border-radius: 12px !important;
    padding: 14px !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
def _init_state() -> None:
    defaults = {
        "pipeline": None,
        "tree": None,
        "doc_name": "",
        "messages": [],
        "indexed": False,
        "index_time": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _count_nodes(node: dict) -> int:
    if not node:
        return 0
    return 1 + sum(_count_nodes(c) for c in node.get("children", []))

def _count_depth(node: dict, d: int = 0) -> int:
    if not node or not node.get("children"):
        return d
    return max(_count_depth(c, d + 1) for c in node["children"])

def _render_tree_nodes(node: dict, depth: int = 0) -> str:
    indent = depth * 20
    children_html = "".join(_render_tree_nodes(c, depth + 1) for c in node.get("children", []))
    children_wrap = f'<div class="tree-children">{children_html}</div>' if children_html else ""
    summary = node.get("summary", "")
    summary_html = f'<div class="tree-node-summary">{summary[:120]}{"…" if len(summary)>120 else ""}</div>' if summary else ""
    return f"""
<div class="tree-node" style="margin-left:{indent}px">
  <div class="tree-node-id"># {node.get('id','?')}</div>
  <div class="tree-node-title">{node.get('title','Untitled')}</div>
  {summary_html}
</div>
{children_wrap}
"""

def _get_settings() -> Settings:
    return Settings(
        llm_base_url=st.session_state.get("llm_base_url", Settings.from_env().llm_base_url),
        llm_api_key=st.session_state.get("llm_api_key", Settings.from_env().llm_api_key or "lm-studio"),
        llm_model=st.session_state.get("llm_model", Settings.from_env().llm_model),
        toc_check_pages=st.session_state.get("toc_pages", 20),
        max_pages_per_node=st.session_state.get("max_pages", 10),
        max_tokens_per_node=st.session_state.get("max_tokens", 20000),
        results_dir=Path("results"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">📑</div>
        <div class="sidebar-logo-text">
            <div class="sidebar-logo-title">PageIndex</div>
            <div class="sidebar-logo-sub">Vectorless RAG</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── LLM Config ──
    st.markdown('<div class="sidebar-section-label">⚡ LLM Endpoint</div>', unsafe_allow_html=True)

    env = Settings.from_env()
    st.text_input("Base URL", value=env.llm_base_url,
                  help="Any OpenAI-compatible endpoint", key="llm_base_url")
    st.text_input("API Key", value=env.llm_api_key or "lm-studio",
                  type="password", help="Use any string for local servers", key="llm_api_key")
    st.text_input("Model", value=env.llm_model,
                  help="Model name from /v1/models", key="llm_model")

    # Quick preset buttons
    st.markdown('<div class="sidebar-section-label">🚀 Quick Presets</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("LM Studio", use_container_width=True):
        st.session_state["llm_base_url"] = "http://localhost:1234/v1"
        st.session_state["llm_api_key"] = "lm-studio"
        st.rerun()
    if c2.button("Ollama", use_container_width=True):
        st.session_state["llm_base_url"] = "http://localhost:11434/v1"
        st.session_state["llm_api_key"] = "ollama"
        st.rerun()
    c3, c4 = st.columns(2)
    if c3.button("OpenAI", use_container_width=True):
        st.session_state["llm_base_url"] = "https://api.openai.com/v1"
        st.rerun()
    if c4.button("vLLM", use_container_width=True):
        st.session_state["llm_base_url"] = "http://localhost:8000/v1"
        st.rerun()

    # ── Advanced ──
    st.markdown('<div class="sidebar-section-label">⚙️ Index Settings</div>', unsafe_allow_html=True)
    with st.expander("Tune Parameters"):
        st.slider("ToC scan pages", 5, 50, 20, key="toc_pages")
        st.slider("Max pages / node", 1, 20, 10, key="max_pages")
        st.slider("Max tokens / node", 5000, 40000, 20000, step=1000, key="max_tokens")
        st.slider("Retrieve top-k sections", 1, 10, 5, key="top_k")

    # ── Upload ──
    st.markdown('<div class="sidebar-section-label">📄 Document</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PDF or Markdown",
                                type=["pdf", "md", "markdown"],
                                label_visibility="collapsed")

    if uploaded:
        kb = round(len(uploaded.getvalue()) / 1024, 1)
        st.markdown(f"<div style='font-size:0.78rem;color:#5050a0;margin:-4px 0 8px 0;'>📄 <code style='color:#7070c0'>{uploaded.name}</code> &nbsp;·&nbsp; {kb} KB</div>", unsafe_allow_html=True)

        if st.button("🔍  Build Vectorless Index", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.indexed = False
            st.session_state.tree = None

            settings = _get_settings()
            suffix = Path(uploaded.name).suffix.lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)

            with st.spinner("Building tree index via LLM…"):
                t0 = time.time()
                try:
                    indexer = Indexer(settings)
                    tree = indexer.index_pdf(tmp_path) if suffix == ".pdf" else indexer.index_markdown(tmp_path)

                    stem = Path(uploaded.name).stem
                    out = settings.results_dir / f"{stem}_index.json"
                    with open(out, "w", encoding="utf-8") as f:
                        json.dump(tree, f, indent=2, ensure_ascii=False)

                    pipeline = RAGPipeline(settings)
                    pipeline._tree = tree
                    pipeline._doc_name = stem

                    st.session_state.pipeline  = pipeline
                    st.session_state.tree      = tree
                    st.session_state.doc_name  = stem
                    st.session_state.indexed   = True
                    st.session_state.index_time = round(time.time() - t0, 1)

                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")
                finally:
                    tmp_path.unlink(missing_ok=True)

    # Status
    st.markdown("")
    if st.session_state.indexed:
        st.markdown(f"""
        <div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.25);
                    border-radius:10px;padding:10px 14px;">
            <div style="color:#4ade80;font-size:0.8rem;font-weight:700;">✅ Index Ready</div>
            <div style="color:#166534;font-size:0.72rem;margin-top:2px;">
                {st.session_state.doc_name} &nbsp;·&nbsp; {st.session_state.index_time}s
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(74,74,255,0.06);border:1px solid rgba(74,74,255,0.15);
                    border-radius:10px;padding:10px 14px;text-align:center;">
            <div style="color:#4040a0;font-size:0.78rem;">Upload a document above to begin</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
is_local = "api.openai.com" not in st.session_state.get("llm_base_url", "")
endpoint_label = "Local Server" if is_local else "Cloud API"
endpoint_color = "#4ade80" if is_local else "#60a5fa"

st.markdown(f"""
<div class="hero">
    <div class="hero-badge">
        <span class="hero-badge-dot"></span>
        Vectorless RAG Engine
    </div>
    <div class="hero-title">PageIndex — No Vectors. Pure Reasoning.</div>
    <div class="hero-subtitle">
        Document Q&amp;A that thinks like a human expert: navigate a structured knowledge tree,
        not a sea of embeddings. Works with any LLM, any format, zero infrastructure.
    </div>
    <div class="hero-pills">
        <span class="hero-pill">📄 PDF &amp; Markdown</span>
        <span class="hero-pill">🌳 Hierarchical Tree Index</span>
        <span class="hero-pill">🧠 LLM-Driven Retrieval</span>
        <span class="hero-pill">🚫 No Vector DB</span>
        <span class="hero-pill">🚫 No Embeddings</span>
        <span class="hero-pill" style="color:{endpoint_color}">⚡ {endpoint_label}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_chat, tab_tree, tab_how = st.tabs(["💬  Chat", "🌳  Document Tree", "ℹ️  How It Works"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB — CHAT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    if not st.session_state.indexed:
        # ── Landing state ──
        st.markdown("""
        <div class="landing">
            <div class="landing-icon">📑</div>
            <div class="landing-title">Upload a document to start asking questions</div>
            <div style="color:#303060;font-size:0.85rem;">Supports PDF and Markdown files</div>
            <div class="landing-steps">
                <div class="landing-step">
                    <div class="landing-step-num">1</div>
                    <div style="font-size:1.5rem">📄</div>
                    <div class="landing-step-text">Upload your document in the sidebar</div>
                </div>
                <div class="landing-step">
                    <div class="landing-step-num">2</div>
                    <div style="font-size:1.5rem">🌳</div>
                    <div class="landing-step-text">Click Build Index — LLM creates a tree</div>
                </div>
                <div class="landing-step">
                    <div class="landing-step-num">3</div>
                    <div style="font-size:1.5rem">💬</div>
                    <div class="landing-step-text">Ask anything, get cited answers</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Stats row ──
        tree = st.session_state.tree
        node_count  = _count_nodes(tree)
        depth       = _count_depth(tree)
        q_count     = len([m for m in st.session_state.messages if m["role"] == "user"])

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-icon stat-icon-blue">🌳</div>
                <div class="stat-body">
                    <div class="stat-value">{node_count}</div>
                    <div class="stat-label">Index Nodes</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon stat-icon-purple">📐</div>
                <div class="stat-body">
                    <div class="stat-value">{depth}</div>
                    <div class="stat-label">Tree Depth</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon stat-icon-amber">💬</div>
                <div class="stat-body">
                    <div class="stat-value">{q_count}</div>
                    <div class="stat-label">Questions Asked</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon stat-icon-green">{"🟢" if is_local else "🔵"}</div>
                <div class="stat-body">
                    <div class="stat-value" style="font-size:1rem;padding-top:4px;">{"Local" if is_local else "Cloud"}</div>
                    <div class="stat-label">LLM Mode</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Chat history ──
        messages_html = ""
        if not st.session_state.messages:
            messages_html = """
            <div class="chat-empty">
                <div class="chat-empty-icon">💭</div>
                <div class="chat-empty-title">No messages yet</div>
                <div class="chat-empty-sub">Type your first question below</div>
            </div>"""
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    messages_html += f"""
                    <div class="msg-row msg-row-user">
                        <div class="msg-avatar msg-avatar-user">🧑</div>
                        <div class="msg-body">
                            <div class="msg-bubble msg-bubble-user">{msg["content"]}</div>
                        </div>
                    </div>"""
                else:
                    sources_html = ""
                    if msg.get("sources"):
                        chips = "".join(
                            f'<span class="source-chip">📎 {s["title"]}</span>'
                            for s in msg["sources"] if s.get("title")
                        )
                        if chips:
                            sources_html = f'<div class="sources-row">{chips}</div>'
                    messages_html += f"""
                    <div class="msg-row">
                        <div class="msg-avatar msg-avatar-ai">🤖</div>
                        <div class="msg-body">
                            <div class="msg-bubble msg-bubble-ai">
                                {msg["content"]}
                                {sources_html}
                            </div>
                        </div>
                    </div>"""

        st.markdown(f'<div class="chat-wrapper">{messages_html}</div>', unsafe_allow_html=True)

        # ── Input ──
        st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
        with st.form("chat_form", clear_on_submit=True):
            cols = st.columns([6, 1])
            question = cols[0].text_input(
                "q", label_visibility="collapsed",
                placeholder="Ask anything about the document…"
            )
            send = cols[1].form_submit_button("Send ➤", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if send and question.strip():
            pipeline: RAGPipeline = st.session_state.pipeline
            settings = _get_settings()
            pipeline.settings = settings
            pipeline.retriever.settings = settings
            pipeline.retriever.settings.apply_to_environment()

            st.session_state.messages.append({"role": "user", "content": question})
            with st.spinner("Navigating document tree…"):
                try:
                    result = pipeline.ask(question,
                                          top_k=st.session_state.get("top_k", 5),
                                          return_sources=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                    })
                except Exception as exc:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚠️ Error: {exc}",
                        "sources": [],
                    })
            st.rerun()

        # Clear button
        if st.session_state.messages:
            if st.button("🗑️ Clear conversation", type="secondary"):
                st.session_state.messages = []
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB — DOCUMENT TREE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_tree:
    if not st.session_state.indexed or not st.session_state.tree:
        st.markdown("""
        <div class="landing" style="padding:2.5rem">
            <div style="font-size:2.5rem;opacity:0.4">🌳</div>
            <div class="landing-title" style="margin-top:0.6rem">No document indexed yet</div>
            <div style="color:#303060;font-size:0.82rem">Build an index first to explore the document tree</div>
        </div>""", unsafe_allow_html=True)
    else:
        tree = st.session_state.tree
        doc  = st.session_state.doc_name
        nc   = _count_nodes(tree)
        dep  = _count_depth(tree)

        st.markdown(f"""
        <div class="tree-header">
            <div class="tree-title">Document Tree</div>
            <div class="tree-doc-badge">{doc}</div>
            <div style="margin-left:auto;display:flex;gap:8px">
                <span style="background:rgba(74,74,255,0.1);border:1px solid rgba(74,74,255,0.2);
                      color:#6060c0;font-size:0.72rem;padding:3px 10px;border-radius:20px;">
                    {nc} nodes
                </span>
                <span style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.2);
                      color:#9060c0;font-size:0.72rem;padding:3px 10px;border-radius:20px;">
                    depth {dep}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_l, col_r = st.columns([3, 2])

        with col_l:
            st.markdown("**Interactive Outline**")
            st.markdown(_render_tree_nodes(tree), unsafe_allow_html=True)

        with col_r:
            st.markdown("**Raw Index JSON**")
            raw = json.dumps(tree, indent=2, ensure_ascii=False)
            st.code("\n".join(raw.splitlines()[:150]), language="json")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**ASCII Tree**")
            st.code(pretty_print_tree(tree), language=None)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB — HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_how:

    st.markdown("""
    <h2 style="font-size:1.5rem;font-weight:800;
               background:linear-gradient(135deg,#ffffff,#a5b4fc);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               margin-bottom:0.3rem;">
        How Vectorless RAG Works
    </h2>
    <p style="color:#5050a0;font-size:0.9rem;margin-bottom:1.5rem;">
        Traditional RAG embeds every chunk into vector space and searches by cosine similarity.
        PageIndex replaces that with something far more intuitive — LLM reasoning over a structured document tree.
    </p>
    """, unsafe_allow_html=True)

    # ── Flow diagram ──
    st.markdown("""
    <div class="flow-container">
        <div class="flow-step">
            <div class="flow-step-num">1</div>
            <div class="flow-step-icon">📄</div>
            <div class="flow-step-title">Parse</div>
            <div class="flow-step-desc">PDF or Markdown is split into sections preserving natural heading hierarchy and page boundaries.</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-step">
            <div class="flow-step-num">2</div>
            <div class="flow-step-icon">🌳</div>
            <div class="flow-step-title">Build Tree</div>
            <div class="flow-step-desc">LLM organises sections into a hierarchical tree — like a smart table of contents. Saved as JSON, never rebuilt.</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-step">
            <div class="flow-step-num">3</div>
            <div class="flow-step-icon">🧭</div>
            <div class="flow-step-title">Navigate</div>
            <div class="flow-step-desc">At query time the LLM sees only the tree skeleton (titles + summaries) and reasons about which nodes to visit.</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-step">
            <div class="flow-step-num">4</div>
            <div class="flow-step-icon">💡</div>
            <div class="flow-step-title">Answer</div>
            <div class="flow-step-desc">Full text of selected nodes is assembled as context. LLM generates a grounded, cited answer.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Comparison table ──
    st.markdown("""
    <h3 style="font-size:1rem;font-weight:700;color:#c0c0ff;margin:2rem 0 0.8rem 0;">
        PageIndex vs Vector RAG
    </h3>
    <table class="compare-table">
        <thead>
            <tr>
                <th>Aspect</th>
                <th>Vector RAG</th>
                <th>PageIndex</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Retrieval mechanism</td><td class="lose">Cosine similarity</td><td class="win">LLM reasoning</td></tr>
            <tr><td>Context quality</td><td class="lose">Fixed-size chunks</td><td class="win">Whole semantic sections</td></tr>
            <tr><td>Embedding model</td><td class="lose">Required</td><td class="win">Not needed</td></tr>
            <tr><td>Vector database</td><td class="lose">Required</td><td class="win">Not needed</td></tr>
            <tr><td>Infrastructure</td><td class="lose">Complex</td><td class="win">Just an LLM</td></tr>
            <tr><td>FinanceBench accuracy</td><td class="lose">85–90%</td><td class="win">98.7% ⭐</td></tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    # ── Provider grid ──
    st.markdown("""
    <h3 style="font-size:1rem;font-weight:700;color:#c0c0ff;margin:2rem 0 0.5rem 0;">
        Works with any LLM Provider
    </h3>
    <div class="provider-grid">
        <div class="provider-card"><div class="provider-card-icon">🖥️</div><div class="provider-card-name">LM Studio</div><div class="provider-card-type">Local</div></div>
        <div class="provider-card"><div class="provider-card-icon">🦙</div><div class="provider-card-name">Ollama</div><div class="provider-card-type">Local</div></div>
        <div class="provider-card"><div class="provider-card-icon">⚡</div><div class="provider-card-name">vLLM</div><div class="provider-card-type">Local</div></div>
        <div class="provider-card"><div class="provider-card-icon">🌐</div><div class="provider-card-name">OpenAI</div><div class="provider-card-type">Cloud</div></div>
        <div class="provider-card"><div class="provider-card-icon">🔮</div><div class="provider-card-name">Anthropic</div><div class="provider-card-type">Cloud</div></div>
        <div class="provider-card"><div class="provider-card-icon">☁️</div><div class="provider-card-name">Azure OpenAI</div><div class="provider-card-type">Cloud</div></div>
        <div class="provider-card"><div class="provider-card-icon">💎</div><div class="provider-card-name">Gemini</div><div class="provider-card-type">Cloud</div></div>
        <div class="provider-card"><div class="provider-card-icon">🔧</div><div class="provider-card-name">Any OpenAI-compat.</div><div class="provider-card-type">Any</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick start ──
    st.markdown("""
    <h3 style="font-size:1rem;font-weight:700;color:#c0c0ff;margin:2rem 0 0.8rem 0;">
        Quick Start
    </h3>
    """, unsafe_allow_html=True)
    st.code("""# 1. Clone & install
git clone https://github.com/YOUR_USERNAME/pageindex-vectorless-rag
cd pageindex-vectorless-rag
pip install -r requirements.txt && pip install -e .

# 2. Configure (copy .env.example → .env and edit)
cp .env.example .env

# 3. Run the UI
streamlit run app.py

# 4. Or use the CLI
pageindex-demo index  data/your_report.pdf
pageindex-demo ask    data/your_report.pdf "What are the main findings?"
pageindex-demo chat   data/your_report.pdf""", language="bash")
