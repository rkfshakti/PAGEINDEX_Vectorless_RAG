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

st.set_page_config(
    page_title="PageIndex · Vectorless RAG",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CSS  — light, airy, professional
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ── base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background-color: #f5f4fb !important;
    color: #1e1b4b !important;
}
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
.block-container { padding-top: 1.4rem !important; padding-bottom: 3rem !important; max-width: 1200px !important; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #ece9f8; }
::-webkit-scrollbar-thumb { background: #c4b8f5; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #9f8de8; }

/* ════════════════════════
   SIDEBAR
════════════════════════ */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8e4f7 !important;
    box-shadow: 2px 0 12px rgba(99,102,241,0.06) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }

.sb-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 0 0 1.2rem 0; border-bottom: 1px solid #ede9fc; margin-bottom: 0.4rem;
}
.sb-logo-icon {
    width: 42px; height: 42px; border-radius: 12px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35);
}
.sb-logo-name {
    font-size: 1.05rem; font-weight: 800; color: #1e1b4b; letter-spacing: -0.01em;
}
.sb-logo-tag {
    font-size: 0.65rem; color: #7c74c9; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.sb-section {
    font-size: 0.63rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6366f1;
    margin: 1.1rem 0 0.45rem 0;
    display: flex; align-items: center; gap: 6px;
}
.sb-section::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, #e0dcfa, transparent);
}

/* sidebar inputs */
[data-testid="stSidebar"] .stTextInput input {
    background: #faf9ff !important;
    border: 1.5px solid #ddd8f8 !important;
    border-radius: 9px !important;
    color: #1e1b4b !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.81rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    background: #fff !important;
}
[data-testid="stSidebar"] label {
    color: #6b6ba8 !important; font-size: 0.77rem !important; font-weight: 500 !important;
}

/* sidebar preset buttons */
[data-testid="stSidebar"] .stButton > button {
    background: #f0eeff !important;
    color: #5b5bd6 !important;
    border: 1.5px solid #d4cefd !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 0.4rem 0.6rem !important;
    transition: all 0.18s !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #e3deff !important;
    border-color: #6366f1 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 10px rgba(99,102,241,0.18) !important;
}

/* upload zone */
[data-testid="stFileUploadDropzone"] {
    background: #faf9ff !important;
    border: 2px dashed #c8c2f5 !important;
    border-radius: 12px !important;
    transition: all 0.2s;
}
[data-testid="stFileUploadDropzone"]:hover {
    background: #f0eeff !important;
    border-color: #6366f1 !important;
}

/* primary sidebar CTA */
.stButton [data-testid="baseButton-primary"] > button,
[data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 0.88rem !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
    transform: translateY(-1px) !important;
}

/* expander */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #faf9ff !important;
    border: 1.5px solid #e8e3fc !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: #5b5bd6 !important; font-size: 0.82rem !important; font-weight: 600 !important;
}

/* slider accent */
[data-testid="stSidebar"] .stSlider [data-testid="stSliderThumb"] {
    background: #6366f1 !important;
}

/* ════════════════════════
   TABS
════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-radius: 12px !important;
    padding: 5px !important;
    gap: 3px !important;
    border: 1.5px solid #e8e3fc !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.06) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #9090c4 !important;
    padding: 9px 22px !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #ffffff !important;
    box-shadow: 0 3px 10px rgba(99,102,241,0.3) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.4rem !important; }

/* ════════════════════════
   HERO
════════════════════════ */
.hero {
    background: linear-gradient(135deg, #ffffff 0%, #faf8ff 50%, #f3f0ff 100%);
    border: 1.5px solid #e4defc;
    border-radius: 22px;
    padding: 2.6rem 3rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(99,102,241,0.08), 0 1px 4px rgba(99,102,241,0.06);
}
.hero::before {
    content: '';
    position: absolute; top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(139,92,246,0.12) 0%, transparent 65%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute; bottom: -100px; left: 60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(99,102,241,0.07) 0%, transparent 65%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(99,102,241,0.08);
    border: 1.5px solid rgba(99,102,241,0.22);
    color: #5b5bd6;
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 5px 14px; border-radius: 20px; margin-bottom: 1.1rem;
}
.hero-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #6366f1;
    animation: hpulse 2s infinite;
}
@keyframes hpulse {
    0%,100% { transform: scale(1); opacity: 1; }
    50%      { transform: scale(0.7); opacity: 0.5; }
}
.hero-title {
    font-size: 2.6rem; font-weight: 800; line-height: 1.15;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #1e1b4b 20%, #4f46e5 60%, #7c3aed 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem;
}
.hero-sub {
    font-size: 1rem; color: #6b6ba8; max-width: 560px;
    line-height: 1.7; font-weight: 400; margin-bottom: 1.5rem;
}
.hero-pills { display: flex; flex-wrap: wrap; gap: 8px; }
.hero-pill {
    background: #ffffff;
    border: 1.5px solid #e4defc;
    color: #6b6ba8;
    font-size: 0.76rem; font-weight: 500;
    padding: 5px 13px; border-radius: 20px;
    display: inline-flex; align-items: center; gap: 5px;
    box-shadow: 0 1px 4px rgba(99,102,241,0.06);
    transition: all 0.18s;
}
.hero-pill:hover { border-color: #9f8de8; color: #5b5bd6; transform: translateY(-1px); }
.hero-pill-hi { color: #5b5bd6 !important; border-color: #c4b8f5 !important; background: #f3f0ff !important; }

/* ════════════════════════
   STAT CARDS
════════════════════════ */
.stat-row { display: flex; gap: 14px; margin-bottom: 1.6rem; }
.stat-card {
    flex: 1;
    background: #ffffff;
    border: 1.5px solid #ede9fc;
    border-radius: 16px;
    padding: 18px 20px;
    display: flex; align-items: center; gap: 14px;
    box-shadow: 0 2px 12px rgba(99,102,241,0.07);
    transition: transform 0.18s, box-shadow 0.18s, border-color 0.18s;
    position: relative; overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    border-radius: 16px 16px 0 0;
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(99,102,241,0.14);
    border-color: #c4b8f5;
}
.stat-icon {
    width: 46px; height: 46px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.35rem; flex-shrink: 0;
}
.si-indigo { background: linear-gradient(135deg, #eef2ff, #e0e7ff); }
.si-violet { background: linear-gradient(135deg, #f5f3ff, #ede9fe); }
.si-amber  { background: linear-gradient(135deg, #fffbeb, #fef3c7); }
.si-green  { background: linear-gradient(135deg, #f0fdf4, #dcfce7); }
.stat-val {
    font-size: 1.75rem; font-weight: 800; line-height: 1;
    background: linear-gradient(135deg, #1e1b4b, #4f46e5);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.stat-lbl {
    font-size: 0.7rem; color: #9090c4; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 3px;
}

/* ════════════════════════
   LANDING
════════════════════════ */
.landing {
    text-align: center;
    background: linear-gradient(135deg, #ffffff, #faf8ff);
    border: 2px dashed #d4cefd;
    border-radius: 22px;
    padding: 4rem 2rem;
    box-shadow: 0 2px 16px rgba(99,102,241,0.05);
}
.landing-icon { font-size: 3.5rem; margin-bottom: 0.8rem; }
.landing-h { font-size: 1.25rem; font-weight: 700; color: #1e1b4b; margin-bottom: 0.4rem; }
.landing-p { font-size: 0.88rem; color: #9090c4; }
.step-row {
    display: flex; justify-content: center; gap: 2.5rem;
    margin-top: 2.4rem; flex-wrap: wrap;
}
.step-col { display: flex; flex-direction: column; align-items: center; gap: 8px; max-width: 130px; }
.step-num {
    width: 34px; height: 34px; border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white; font-size: 0.85rem; font-weight: 800;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 12px rgba(99,102,241,0.35);
}
.step-emoji { font-size: 1.6rem; }
.step-txt { font-size: 0.78rem; color: #6b6ba8; text-align: center; line-height: 1.4; }

/* ════════════════════════
   CHAT
════════════════════════ */
.chat-box {
    background: #ffffff;
    border: 1.5px solid #ede9fc;
    border-radius: 18px;
    padding: 1.6rem;
    min-height: 320px;
    box-shadow: 0 2px 16px rgba(99,102,241,0.07);
    margin-bottom: 1rem;
}
.chat-empty {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 3rem 1rem; color: #c4bef0; text-align: center;
}
.chat-empty-ico { font-size: 2.6rem; margin-bottom: 0.6rem; opacity: 0.5; }
.chat-empty-h { font-size: 0.95rem; font-weight: 600; color: #b0aad8; }
.chat-empty-s { font-size: 0.8rem; color: #cdc7ec; margin-top: 3px; }

.msg-row { display: flex; align-items: flex-end; gap: 10px; margin-bottom: 1.2rem;
           animation: fadein 0.28s ease; }
@keyframes fadein { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
.msg-row-user { flex-direction: row-reverse; }
.avatar {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem; flex-shrink: 0;
}
.av-user { background: linear-gradient(135deg, #3b82f6, #6366f1); box-shadow: 0 3px 10px rgba(99,102,241,0.25); }
.av-ai   { background: linear-gradient(135deg, #6366f1, #8b5cf6); box-shadow: 0 3px 10px rgba(139,92,246,0.3); }
.msg-body { max-width: 76%; }
.bubble {
    padding: 13px 17px; border-radius: 18px;
    font-size: 0.91rem; line-height: 1.65; word-break: break-word;
}
.bubble-user {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: #ffffff;
    border-radius: 18px 18px 4px 18px;
    box-shadow: 0 4px 16px rgba(99,102,241,0.28);
}
.bubble-ai {
    background: #faf8ff;
    color: #1e1b4b;
    border: 1.5px solid #ede9fc;
    border-radius: 18px 18px 18px 4px;
    box-shadow: 0 3px 12px rgba(99,102,241,0.08);
}
.src-row {
    display: flex; flex-wrap: wrap; gap: 6px;
    margin-top: 10px; padding-top: 10px;
    border-top: 1px solid #ede9fc;
}
.src-chip {
    display: inline-flex; align-items: center; gap: 5px;
    background: #f0eeff; border: 1.5px solid #d4cefd;
    color: #5b5bd6; font-size: 0.7rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    transition: all 0.15s;
}
.src-chip:hover { background: #e3deff; border-color: #9f8de8; }

/* chat input wrapper */
.input-wrap {
    background: #ffffff;
    border: 1.5px solid #e4defc;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    box-shadow: 0 2px 12px rgba(99,102,241,0.07);
    margin-bottom: 0.6rem;
}
.stTextInput > div > div > input {
    background: #faf8ff !important;
    border: 1.5px solid #ddd8f8 !important;
    border-radius: 10px !important;
    color: #1e1b4b !important;
    font-size: 0.92rem !important;
    padding: 0.65rem 1rem !important;
    transition: all 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    background: #fff !important;
}
.stTextInput > div > div > input::placeholder { color: #b0aad8 !important; }
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 0.88rem !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
    transition: all 0.2s !important;
    padding: 0.65rem 1.4rem !important;
}
.stFormSubmitButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
}
/* clear chat */
.stButton > button[kind="secondary"], .stButton > button {
    background: #f8f6ff !important;
    border: 1.5px solid #e0d9fc !important;
    color: #8080c0 !important;
    border-radius: 9px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    transition: all 0.18s !important;
}
.stButton > button:hover {
    background: #fee2e2 !important;
    border-color: #fca5a5 !important;
    color: #ef4444 !important;
    transform: none !important;
}

/* ════════════════════════
   TREE VIEW
════════════════════════ */
.tree-hdr {
    display: flex; align-items: center; gap: 12px;
    padding-bottom: 1rem; margin-bottom: 1.2rem;
    border-bottom: 1.5px solid #ede9fc;
}
.tree-title { font-size: 1.1rem; font-weight: 800; color: #1e1b4b; }
.tree-badge {
    background: #f0eeff; border: 1.5px solid #d4cefd;
    color: #5b5bd6; font-size: 0.73rem; font-weight: 700;
    padding: 3px 11px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
}
.tree-meta {
    margin-left: auto; display: flex; gap: 8px;
}
.tree-tag {
    background: #f5f3ff; border: 1.5px solid #e4defc;
    color: #7c74c9; font-size: 0.7rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
}
.tnode {
    background: #ffffff;
    border: 1.5px solid #ede9fc;
    border-radius: 11px;
    padding: 11px 15px;
    margin: 6px 0;
    transition: border-color 0.18s, box-shadow 0.18s;
    box-shadow: 0 1px 4px rgba(99,102,241,0.05);
}
.tnode:hover { border-color: #c4b8f5; box-shadow: 0 3px 12px rgba(99,102,241,0.1); }
.tnode-id { font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#6366f1; font-weight:700; }
.tnode-title { font-size:0.9rem; font-weight:700; color:#1e1b4b; margin:3px 0 2px; }
.tnode-sum { font-size:0.78rem; color:#8080b8; line-height:1.5; }
.tchildren { margin-left:18px; border-left:2px solid #e0dcfa; padding-left:12px; margin-top:4px; }

/* code blocks */
.stCodeBlock > div {
    background: #f8f6ff !important;
    border: 1.5px solid #e4defc !important;
    border-radius: 11px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #2d2b6b !important;
}

/* ════════════════════════
   HOW IT WORKS
════════════════════════ */
.flow-row { display:flex; align-items:stretch; gap:0; margin: 1.8rem 0; flex-wrap:wrap; }
.flow-card {
    flex:1; min-width:165px;
    background:#ffffff;
    border:1.5px solid #e8e3fc;
    border-radius:16px;
    padding:1.4rem 1.2rem;
    box-shadow:0 2px 12px rgba(99,102,241,0.07);
    transition: transform 0.18s, box-shadow 0.18s;
    position: relative;
}
.flow-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,#6366f1,#8b5cf6);
    border-radius:16px 16px 0 0;
}
.flow-card:hover { transform:translateY(-3px); box-shadow:0 8px 28px rgba(99,102,241,0.15); }
.flow-arrow { display:flex; align-items:center; padding:0 10px; color:#c4b8f5; font-size:1.4rem; flex-shrink:0; }
.flow-num {
    width:28px; height:28px; border-radius:50%;
    background:linear-gradient(135deg,#6366f1,#8b5cf6);
    color:white; font-size:0.78rem; font-weight:800;
    display:flex; align-items:center; justify-content:center;
    margin-bottom:0.7rem;
    box-shadow:0 3px 10px rgba(99,102,241,0.35);
}
.flow-ico { font-size:1.6rem; margin-bottom:0.5rem; }
.flow-h { font-size:0.9rem; font-weight:700; color:#1e1b4b; margin-bottom:0.35rem; }
.flow-p { font-size:0.78rem; color:#8080b8; line-height:1.55; }

/* compare table */
.cmp-tbl { width:100%; border-collapse:separate; border-spacing:0;
           border-radius:14px; overflow:hidden;
           border:1.5px solid #e8e3fc; margin:1.5rem 0;
           box-shadow:0 2px 12px rgba(99,102,241,0.07); }
.cmp-tbl th {
    background:linear-gradient(135deg,#6366f1,#8b5cf6);
    color:#ffffff; font-size:0.78rem; font-weight:700;
    letter-spacing:0.06em; text-transform:uppercase;
    padding:13px 18px; text-align:left;
}
.cmp-tbl td {
    background:#ffffff; color:#6b6ba8; font-size:0.86rem;
    padding:11px 18px; border-bottom:1px solid #f0ecfd;
}
.cmp-tbl tr:last-child td { border-bottom:none; }
.cmp-tbl tr:hover td { background:#faf8ff; }
.win { color:#16a34a !important; font-weight:700; }
.lose { color:#dc2626 !important; }

/* provider grid */
.prov-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(130px,1fr)); gap:10px; margin-top:1rem; }
.prov-card {
    background:#ffffff; border:1.5px solid #ede9fc;
    border-radius:12px; padding:14px 10px; text-align:center;
    box-shadow:0 2px 8px rgba(99,102,241,0.06);
    transition:all 0.18s;
}
.prov-card:hover { border-color:#c4b8f5; transform:translateY(-2px); box-shadow:0 5px 18px rgba(99,102,241,0.13); }
.prov-ico { font-size:1.5rem; margin-bottom:5px; }
.prov-name { font-size:0.75rem; color:#1e1b4b; font-weight:700; }
.prov-type { font-size:0.65rem; color:#a0a0c8; margin-top:2px; }

/* alert overrides */
[data-testid="stAlert"] { border-radius:10px !important; }

/* section headings */
.sec-h2 {
    font-size:1.45rem; font-weight:800; color:#1e1b4b;
    letter-spacing:-0.02em; margin-bottom:0.3rem;
}
.sec-sub { font-size:0.9rem; color:#8080b8; line-height:1.6; margin-bottom:1.4rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
def _init():
    defs = dict(pipeline=None, tree=None, doc_name="", messages=[],
                indexed=False, index_time=0.0)
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def _count(n):
    return 0 if not n else 1 + sum(_count(c) for c in n.get("children", []))

def _depth(n, d=0):
    if not n or not n.get("children"): return d
    return max(_depth(c, d+1) for c in n["children"])

def _tree_html(node, indent=0):
    ml = indent * 18
    children = node.get("children", [])
    ch_html = "".join(_tree_html(c, indent+1) for c in children)
    ch_wrap = f'<div class="tchildren">{ch_html}</div>' if ch_html else ""
    s = node.get("summary","")
    s_html = f'<div class="tnode-sum">{s[:130]}{"…" if len(s)>130 else ""}</div>' if s else ""
    return f"""
<div class="tnode" style="margin-left:{ml}px">
  <div class="tnode-id"># {node.get("id","?")}</div>
  <div class="tnode-title">{node.get("title","—")}</div>
  {s_html}
</div>{ch_wrap}"""

def _cfg():
    env = Settings.from_env()
    return Settings(
        llm_base_url  = st.session_state.get("llm_base_url",  env.llm_base_url),
        llm_api_key   = st.session_state.get("llm_api_key",   env.llm_api_key or "lm-studio"),
        llm_model     = st.session_state.get("llm_model",     env.llm_model),
        toc_check_pages   = st.session_state.get("toc_pg",  20),
        max_pages_per_node= st.session_state.get("max_pg",  10),
        max_tokens_per_node=st.session_state.get("max_tok", 20000),
        results_dir   = Path("results"),
    )


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
      <div class="sb-logo-icon">📑</div>
      <div>
        <div class="sb-logo-name">PageIndex</div>
        <div class="sb-logo-tag">Vectorless RAG</div>
      </div>
    </div>""", unsafe_allow_html=True)

    env = Settings.from_env()

    st.markdown('<div class="sb-section">⚡ LLM Endpoint</div>', unsafe_allow_html=True)
    st.text_input("Base URL", value=env.llm_base_url,  key="llm_base_url",
                  help="Any OpenAI-compatible API endpoint")
    st.text_input("API Key",  value=env.llm_api_key or "lm-studio",
                  type="password", key="llm_api_key",
                  help="Any non-empty string for local servers")
    st.text_input("Model",    value=env.llm_model, key="llm_model",
                  help="Exact model name from /v1/models")

    st.markdown('<div class="sb-section">🚀 Quick Presets</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("LM Studio", use_container_width=True):
        st.session_state.llm_base_url = "http://localhost:1234/v1"
        st.session_state.llm_api_key  = "lm-studio"; st.rerun()
    if c2.button("Ollama", use_container_width=True):
        st.session_state.llm_base_url = "http://localhost:11434/v1"
        st.session_state.llm_api_key  = "ollama"; st.rerun()
    c3, c4 = st.columns(2)
    if c3.button("OpenAI", use_container_width=True):
        st.session_state.llm_base_url = "https://api.openai.com/v1"; st.rerun()
    if c4.button("vLLM", use_container_width=True):
        st.session_state.llm_base_url = "http://localhost:8000/v1"; st.rerun()

    st.markdown('<div class="sb-section">⚙️ Index Tuning</div>', unsafe_allow_html=True)
    with st.expander("Parameters"):
        st.slider("ToC scan pages",         5,  50,   20, key="toc_pg")
        st.slider("Max pages / node",       1,  20,   10, key="max_pg")
        st.slider("Max tokens / node",   5000, 40000, 20000, step=1000, key="max_tok")
        st.slider("Retrieve top-k",         1,  10,    5, key="top_k")

    st.markdown('<div class="sb-section">📄 Document</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PDF or Markdown",
                                type=["pdf","md","markdown"],
                                label_visibility="collapsed")

    if uploaded:
        kb = round(len(uploaded.getvalue()) / 1024, 1)
        st.markdown(
            f"<p style='font-size:0.76rem;color:#9090c4;margin:-2px 0 8px;'>"
            f"📄 <code style='color:#5b5bd6'>{uploaded.name}</code> · {kb} KB</p>",
            unsafe_allow_html=True)

        if st.button("🔍  Build Vectorless Index", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.indexed  = False
            st.session_state.tree     = None
            settings = _cfg()
            suffix   = Path(uploaded.name).suffix.lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)

            with st.spinner("Building tree via LLM…"):
                t0 = time.time()
                try:
                    ix   = Indexer(settings)
                    tree = ix.index_pdf(tmp_path) if suffix == ".pdf" else ix.index_markdown(tmp_path)
                    stem = Path(uploaded.name).stem
                    out  = settings.results_dir / f"{stem}_index.json"
                    out.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")

                    pipe = RAGPipeline(settings)
                    pipe._tree = tree; pipe._doc_name = stem

                    st.session_state.update(
                        pipeline=pipe, tree=tree, doc_name=stem,
                        indexed=True, index_time=round(time.time()-t0, 1))
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
                finally:
                    tmp_path.unlink(missing_ok=True)

    st.markdown("")
    if st.session_state.indexed:
        st.markdown(f"""
        <div style="background:#f0fdf4;border:1.5px solid #bbf7d0;
                    border-radius:11px;padding:11px 15px;">
          <div style="color:#15803d;font-size:0.82rem;font-weight:700;">✅ Index Ready</div>
          <div style="color:#4ade80;font-size:0.72rem;margin-top:2px;font-family:'JetBrains Mono',monospace;">
            {st.session_state.doc_name} · {st.session_state.index_time}s
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#f5f3ff;border:1.5px dashed #c4b8f5;
                    border-radius:11px;padding:11px 15px;text-align:center;">
          <div style="color:#7c74c9;font-size:0.78rem;">Upload a document above to begin</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════
is_local = "api.openai.com" not in st.session_state.get("llm_base_url","")
ep_label = "Local Server" if is_local else "Cloud API"
ep_color = "#16a34a"      if is_local else "#2563eb"

st.markdown(f"""
<div class="hero">
  <div class="hero-badge"><span class="hero-dot"></span>Vectorless RAG Engine · Open Source</div>
  <div class="hero-title">PageIndex — No Vectors.<br>Pure Reasoning.</div>
  <div class="hero-sub">
    Document Q&amp;A that thinks like a human expert: navigate a structured
    knowledge tree instead of drowning in embeddings.
    Works with any LLM, any format, zero infrastructure overhead.
  </div>
  <div class="hero-pills">
    <span class="hero-pill">📄 PDF &amp; Markdown</span>
    <span class="hero-pill">🌳 Hierarchical Tree Index</span>
    <span class="hero-pill">🧠 LLM-Driven Retrieval</span>
    <span class="hero-pill">🚫 No Vector DB</span>
    <span class="hero-pill">🚫 No Embeddings</span>
    <span class="hero-pill hero-pill-hi" style="color:{ep_color}">⚡ {ep_label}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab_chat, tab_tree, tab_how = st.tabs(["💬  Chat", "🌳  Document Tree", "ℹ️  How It Works"])


# ──────────────────────────────────────────────
# TAB: CHAT
# ──────────────────────────────────────────────
with tab_chat:

    if not st.session_state.indexed:
        st.markdown("""
        <div class="landing">
          <div class="landing-icon">📑</div>
          <div class="landing-h">Upload a document to start asking questions</div>
          <div class="landing-p">Supports PDF and Markdown · Index is built once, reused forever</div>
          <div class="step-row">
            <div class="step-col">
              <div class="step-num">1</div>
              <div class="step-emoji">📄</div>
              <div class="step-txt">Upload your document in the sidebar</div>
            </div>
            <div class="step-col">
              <div class="step-num">2</div>
              <div class="step-emoji">🌳</div>
              <div class="step-txt">Click Build Index — LLM creates a tree</div>
            </div>
            <div class="step-col">
              <div class="step-num">3</div>
              <div class="step-emoji">💬</div>
              <div class="step-txt">Ask anything and get cited answers</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    else:
        # ── Stats ──
        tree = st.session_state.tree
        nc   = _count(tree)
        dep  = _depth(tree)
        nq   = sum(1 for m in st.session_state.messages if m["role"] == "user")
        ep_s = "🟢 Local" if is_local else "🔵 Cloud"

        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card">
            <div class="stat-icon si-indigo">🌳</div>
            <div><div class="stat-val">{nc}</div><div class="stat-lbl">Index Nodes</div></div>
          </div>
          <div class="stat-card">
            <div class="stat-icon si-violet">📐</div>
            <div><div class="stat-val">{dep}</div><div class="stat-lbl">Tree Depth</div></div>
          </div>
          <div class="stat-card">
            <div class="stat-icon si-amber">💬</div>
            <div><div class="stat-val">{nq}</div><div class="stat-lbl">Questions</div></div>
          </div>
          <div class="stat-card">
            <div class="stat-icon si-green">{"🟢" if is_local else "🔵"}</div>
            <div><div class="stat-val" style="font-size:1rem;padding-top:5px;">{"Local" if is_local else "Cloud"}</div>
                 <div class="stat-lbl">LLM Mode</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Messages ──
        if not st.session_state.messages:
            msgs_html = """
            <div class="chat-empty">
              <div class="chat-empty-ico">💭</div>
              <div class="chat-empty-h">No messages yet</div>
              <div class="chat-empty-s">Ask your first question below</div>
            </div>"""
        else:
            msgs_html = ""
            for m in st.session_state.messages:
                if m["role"] == "user":
                    msgs_html += f"""
                    <div class="msg-row msg-row-user">
                      <div class="avatar av-user">🧑</div>
                      <div class="msg-body">
                        <div class="bubble bubble-user">{m["content"]}</div>
                      </div>
                    </div>"""
                else:
                    chips = ""
                    if m.get("sources"):
                        chips = "".join(
                            f'<span class="src-chip">📎 {s["title"]}</span>'
                            for s in m["sources"] if s.get("title"))
                        if chips:
                            chips = f'<div class="src-row">{chips}</div>'
                    msgs_html += f"""
                    <div class="msg-row">
                      <div class="avatar av-ai">🤖</div>
                      <div class="msg-body">
                        <div class="bubble bubble-ai">{m["content"]}{chips}</div>
                      </div>
                    </div>"""

        st.markdown(f'<div class="chat-box">{msgs_html}</div>', unsafe_allow_html=True)

        # ── Input ──
        st.markdown('<div class="input-wrap">', unsafe_allow_html=True)
        with st.form("qform", clear_on_submit=True):
            cols = st.columns([6, 1])
            q    = cols[0].text_input("q", label_visibility="collapsed",
                                      placeholder="Ask anything about the document…")
            send = cols[1].form_submit_button("Send ➤", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if send and q.strip():
            pipe: RAGPipeline = st.session_state.pipeline
            cfg = _cfg()
            pipe.settings = cfg
            pipe.retriever.settings = cfg
            cfg.apply_to_environment()
            st.session_state.messages.append({"role":"user","content":q})
            with st.spinner("Navigating document tree…"):
                try:
                    res = pipe.ask(q, top_k=st.session_state.get("top_k",5), return_sources=True)
                    st.session_state.messages.append(
                        {"role":"assistant","content":res["answer"],"sources":res.get("sources",[])})
                except Exception as e:
                    st.session_state.messages.append(
                        {"role":"assistant","content":f"⚠️ {e}","sources":[]})
            st.rerun()

        if st.session_state.messages:
            if st.button("🗑️  Clear conversation"):
                st.session_state.messages = []; st.rerun()


# ──────────────────────────────────────────────
# TAB: TREE
# ──────────────────────────────────────────────
with tab_tree:
    if not st.session_state.indexed:
        st.markdown("""
        <div class="landing" style="padding:2.5rem">
          <div style="font-size:2.5rem">🌳</div>
          <div class="landing-h" style="margin-top:0.6rem">No document indexed yet</div>
          <div class="landing-p">Build an index first to explore the document tree</div>
        </div>""", unsafe_allow_html=True)
    else:
        tree = st.session_state.tree
        nc, dep = _count(tree), _depth(tree)
        st.markdown(f"""
        <div class="tree-hdr">
          <div class="tree-title">Document Tree</div>
          <div class="tree-badge">{st.session_state.doc_name}</div>
          <div class="tree-meta">
            <span class="tree-tag">🗂 {nc} nodes</span>
            <span class="tree-tag">📏 depth {dep}</span>
          </div>
        </div>""", unsafe_allow_html=True)

        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown("**Interactive Outline**")
            st.markdown(_tree_html(tree), unsafe_allow_html=True)
        with col_r:
            st.markdown("**Raw Index JSON**")
            raw = json.dumps(tree, indent=2, ensure_ascii=False)
            st.code("\n".join(raw.splitlines()[:150]), language="json")
            st.markdown("**ASCII Tree**")
            st.code(pretty_print_tree(tree), language=None)


# ──────────────────────────────────────────────
# TAB: HOW IT WORKS
# ──────────────────────────────────────────────
with tab_how:
    st.markdown("""
    <div class="sec-h2">How Vectorless RAG Works</div>
    <div class="sec-sub">
      Traditional RAG embeds every chunk into a vector space and retrieves by cosine similarity.
      PageIndex replaces all of that with structured LLM reasoning over a document tree.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="flow-row">
      <div class="flow-card">
        <div class="flow-num">1</div>
        <div class="flow-ico">📄</div>
        <div class="flow-h">Parse</div>
        <div class="flow-p">PDF or Markdown is split into sections that respect heading hierarchy and page boundaries.</div>
      </div>
      <div class="flow-arrow">→</div>
      <div class="flow-card">
        <div class="flow-num">2</div>
        <div class="flow-ico">🌳</div>
        <div class="flow-h">Build Tree</div>
        <div class="flow-p">LLM organises sections into a hierarchical tree — a smart table of contents. Saved as JSON, never rebuilt.</div>
      </div>
      <div class="flow-arrow">→</div>
      <div class="flow-card">
        <div class="flow-num">3</div>
        <div class="flow-ico">🧭</div>
        <div class="flow-h">Navigate</div>
        <div class="flow-p">At query time the LLM sees only the skeleton (titles + summaries) and reasons about which nodes to visit.</div>
      </div>
      <div class="flow-arrow">→</div>
      <div class="flow-card">
        <div class="flow-num">4</div>
        <div class="flow-ico">💡</div>
        <div class="flow-h">Answer</div>
        <div class="flow-p">Full text of selected nodes is assembled as context. LLM generates a grounded, cited answer.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sec-h2" style="font-size:1.1rem;margin-top:1.5rem">PageIndex vs Vector RAG</div>
    <table class="cmp-tbl">
      <thead><tr><th>Aspect</th><th>Vector RAG</th><th>PageIndex</th></tr></thead>
      <tbody>
        <tr><td>Retrieval</td>         <td class="lose">Cosine similarity</td><td class="win">LLM reasoning</td></tr>
        <tr><td>Context quality</td>   <td class="lose">Fixed-size chunks</td><td class="win">Whole semantic sections</td></tr>
        <tr><td>Embedding model</td>   <td class="lose">Required</td>         <td class="win">Not needed</td></tr>
        <tr><td>Vector database</td>   <td class="lose">Required</td>         <td class="win">Not needed</td></tr>
        <tr><td>Infrastructure</td>    <td class="lose">Complex stack</td>    <td class="win">Just an LLM</td></tr>
        <tr><td>FinanceBench</td>      <td class="lose">85 – 90 %</td>        <td class="win">98.7 % ⭐</td></tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sec-h2" style="font-size:1.1rem;margin-top:1.5rem">Works with any LLM Provider</div>
    <div class="prov-grid">
      <div class="prov-card"><div class="prov-ico">🖥️</div><div class="prov-name">LM Studio</div><div class="prov-type">Local</div></div>
      <div class="prov-card"><div class="prov-ico">🦙</div><div class="prov-name">Ollama</div><div class="prov-type">Local</div></div>
      <div class="prov-card"><div class="prov-ico">⚡</div><div class="prov-name">vLLM</div><div class="prov-type">Local</div></div>
      <div class="prov-card"><div class="prov-ico">🌐</div><div class="prov-name">OpenAI</div><div class="prov-type">Cloud</div></div>
      <div class="prov-card"><div class="prov-ico">🔮</div><div class="prov-name">Anthropic</div><div class="prov-type">Cloud</div></div>
      <div class="prov-card"><div class="prov-ico">☁️</div><div class="prov-name">Azure OpenAI</div><div class="prov-type">Cloud</div></div>
      <div class="prov-card"><div class="prov-ico">💎</div><div class="prov-name">Gemini</div><div class="prov-type">Cloud</div></div>
      <div class="prov-card"><div class="prov-ico">🔧</div><div class="prov-name">Any OAI-compat</div><div class="prov-type">Any</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""<div class="sec-h2" style="font-size:1.1rem">Quick Start</div>""", unsafe_allow_html=True)
    st.code("""# 1. Clone & install
git clone https://github.com/YOUR_USERNAME/pageindex-vectorless-rag
cd pageindex-vectorless-rag && pip install -r requirements.txt && pip install -e .

# 2. Configure — copy .env.example → .env and set your LLM endpoint
cp .env.example .env

# 3. Launch the UI
streamlit run app.py

# 4. Or use the CLI directly
pageindex-demo index  data/your_report.pdf
pageindex-demo ask    data/your_report.pdf "What are the key findings?"
pageindex-demo chat   data/your_report.pdf""", language="bash")
