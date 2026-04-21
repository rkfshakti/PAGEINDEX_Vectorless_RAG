"""
make_gif.py — Records an end-to-end demo GIF of PageIndex Vectorless RAG.

Usage:
    python make_gif.py

Output:
    demo/pageindex_demo.gif
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright, Page

# ── Config ──────────────────────────────────────────────────────────────────
APP_URL     = "http://localhost:8504"
OUT_DIR     = Path("demo")
OUT_GIF     = OUT_DIR / "pageindex_demo.gif"
VIEWPORT    = {"width": 1280, "height": 800}
DOC_PATH    = str(Path("data/sample_report.md").resolve())

# GIF timing (milliseconds per frame)
HOLD_LONG   = 3000   # pause on important moments
HOLD_MED    = 2000
HOLD_SHORT  = 1200

OUT_DIR.mkdir(exist_ok=True)


# ── Annotation helpers ───────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a font, gracefully falling back to default."""
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def annotate(img: Image.Image, label: str, step: str = "") -> Image.Image:
    """Add a bottom-bar annotation with step label and description."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    bar_h = 52
    # gradient-ish bar
    for y in range(h - bar_h, h):
        alpha = int(230 * (y - (h - bar_h)) / bar_h)
        draw.rectangle([0, y, w, y], fill=(15, 13, 50, alpha))

    # solid base
    draw.rectangle([0, h - bar_h, w, h], fill=(15, 13, 50, 230))

    # left accent stripe
    draw.rectangle([0, h - bar_h, 5, h], fill=(99, 102, 241))

    fn_bold   = _load_font(16)
    fn_normal = _load_font(13)

    if step:
        draw.text((18, h - bar_h + 8),  step,  font=fn_bold,   fill=(165, 180, 252))
    draw.text((18, h - bar_h + 28), label, font=fn_normal, fill=(199, 210, 254))

    # branding
    brand = "PageIndex · Vectorless RAG"
    bb = draw.textbbox((0, 0), brand, font=fn_normal)
    bw = bb[2] - bb[0]
    draw.text((w - bw - 16, h - bar_h + 19), brand, font=fn_normal, fill=(67, 65, 120))

    return img


def screenshot(page: Page, label: str, step: str = "",
               hold_ms: int = HOLD_MED, scroll_top: bool = False) -> list[Image.Image]:
    """Capture a page screenshot, annotate it, and return frames for the GIF."""
    if scroll_top:
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.3)
    page.wait_for_timeout(600)
    raw   = page.screenshot(full_page=False)
    img   = Image.open(io.BytesIO(raw)).convert("RGBA")
    frame = annotate(img, label, step)
    # repeat frame to control display duration
    n_frames = max(1, hold_ms // 100)
    return [frame] * n_frames


def scroll_shot(page: Page, label: str, step: str = "",
                to: int = 400, hold_ms: int = HOLD_MED) -> list[Image.Image]:
    """Scroll down smoothly then screenshot."""
    page.evaluate(f"window.scrollTo({{top:{to}, behavior:'smooth'}})")
    time.sleep(0.8)
    return screenshot(page, label, step, hold_ms=hold_ms)


# ── Main recording logic ─────────────────────────────────────────────────────

def record() -> None:
    frames: list[Image.Image] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=False,   # visible so you can see it happen
            slow_mo=120,
            args=["--disable-blink-features=AutomationControlled",
                  "--no-sandbox", "--disable-dev-shm-usage"],
        )
        ctx  = browser.new_context(viewport=VIEWPORT)
        page = ctx.new_page()

        # ── 1. Landing page ────────────────────────────────────────────────
        print("Step 1 — Landing page")
        page.goto(APP_URL, wait_until="networkidle", timeout=30_000)
        # Wait until Streamlit has fully rendered (hero title visible)
        page.wait_for_selector("text=PageIndex", timeout=30_000)
        page.wait_for_selector("text=No Vectors", timeout=30_000)
        time.sleep(3)
        frames += screenshot(page,
            "Clean landing state — upload a document to begin",
            "Step 1  ·  Landing", hold_ms=HOLD_LONG, scroll_top=True)

        # ── 2. Hero + feature pills ────────────────────────────────────────
        print("Step 2 — Hero")
        frames += screenshot(page,
            "Hero: No Vectors, No Embeddings — pure LLM reasoning over a tree index",
            "Step 2  ·  Hero Banner", hold_ms=HOLD_LONG)

        # ── 3. Sidebar config ──────────────────────────────────────────────
        print("Step 3 — Sidebar LLM config")
        page.hover("text=Base URL")
        time.sleep(0.4)
        frames += screenshot(page,
            "Sidebar: configure any OpenAI-compatible endpoint (local or cloud)",
            "Step 3  ·  LLM Configuration", hold_ms=HOLD_MED)

        # ── 4. Upload document ─────────────────────────────────────────────
        print("Step 4 — Upload document")
        upload_input = page.locator('input[type="file"]').first
        upload_input.set_input_files(DOC_PATH)
        time.sleep(1.5)
        frames += screenshot(page,
            f"Document uploaded: sample_report.md",
            "Step 4  ·  Upload Document", hold_ms=HOLD_MED)

        # ── 5. Click Build Index ───────────────────────────────────────────
        print("Step 5 — Build Index")
        btn = page.locator("text=Build Vectorless Index").first
        btn.scroll_into_view_if_needed()
        time.sleep(0.5)
        frames += screenshot(page,
            "Clicking 'Build Vectorless Index' — LLM will create a tree structure",
            "Step 5  ·  Start Indexing", hold_ms=HOLD_SHORT)
        btn.click()
        time.sleep(1.5)

        # spinner / building
        frames += screenshot(page,
            "LLM is parsing the document and building the hierarchical tree index…",
            "Step 5  ·  Indexing in Progress", hold_ms=HOLD_LONG)

        # ── 6. Wait for index to complete ──────────────────────────────────
        print("Step 6 — Waiting for index…")
        page.wait_for_selector("text=Index Ready", timeout=300_000)
        time.sleep(1.5)
        frames += screenshot(page,
            "Index built! Tree saved as JSON — reusable without re-indexing",
            "Step 6  ·  Index Ready", hold_ms=HOLD_LONG, scroll_top=True)

        # ── 7. Chat tab — stat cards ───────────────────────────────────────
        print("Step 7 — Stats cards")
        page.locator("text=💬  Chat").click()
        time.sleep(1)
        frames += screenshot(page,
            "Stats: node count, tree depth, questions asked, LLM mode",
            "Step 7  ·  Chat — Stats Overview", hold_ms=HOLD_MED, scroll_top=True)

        # ── 8. Ask question 1 ──────────────────────────────────────────────
        print("Step 8 — First question")
        q1 = "What accuracy did PageIndex achieve on FinanceBench?"
        page.locator('input[placeholder="Ask anything about the document…"]').fill(q1)
        time.sleep(0.6)
        frames += screenshot(page,
            f'Asking: "{q1}"',
            "Step 8  ·  First Question", hold_ms=HOLD_MED)
        page.keyboard.press("Enter")
        time.sleep(1)

        # waiting for answer
        frames += screenshot(page,
            "Retrieving: LLM navigates the tree skeleton to find relevant nodes…",
            "Step 8  ·  Tree Navigation", hold_ms=HOLD_LONG)
        page.wait_for_selector("text=98.7", timeout=120_000)
        time.sleep(1.5)
        frames += screenshot(page,
            "Answer with source citation — retrieved from the exact document section",
            "Step 8  ·  Answer + Sources", hold_ms=HOLD_LONG)

        # ── 9. Ask question 2 ──────────────────────────────────────────────
        print("Step 9 — Second question")
        q2 = "How does vectorless RAG compare to traditional vector-based methods?"
        page.locator('input[placeholder="Ask anything about the document…"]').fill(q2)
        time.sleep(0.5)
        frames += screenshot(page,
            f'Asking: "{q2}"',
            "Step 9  ·  Second Question", hold_ms=HOLD_SHORT)
        page.keyboard.press("Enter")
        time.sleep(1)
        page.wait_for_selector("text=cosine", timeout=120_000)
        time.sleep(1.5)
        frames += scroll_shot(page,
            "Detailed comparison answer — no vector DB, no embedding model needed",
            "Step 9  ·  Comparison Answer", to=500, hold_ms=HOLD_LONG)

        # ── 10. Switch to Tree tab ─────────────────────────────────────────
        print("Step 10 — Tree tab")
        page.locator("text=🌳  Document Tree").click()
        time.sleep(1.5)
        frames += screenshot(page,
            "Document Tree: interactive node outline + raw JSON index",
            "Step 10  ·  Tree View", hold_ms=HOLD_LONG, scroll_top=True)
        frames += scroll_shot(page,
            "Full JSON tree structure — hierarchically organised by LLM",
            "Step 10  ·  Raw JSON Index", to=400, hold_ms=HOLD_MED)

        # ── 11. How It Works tab ───────────────────────────────────────────
        print("Step 11 — How It Works")
        page.locator("text=ℹ️  How It Works").click()
        time.sleep(1.5)
        frames += screenshot(page,
            "4-step flow: Parse → Build Tree → Navigate → Answer",
            "Step 11  ·  How It Works", hold_ms=HOLD_LONG, scroll_top=True)
        frames += scroll_shot(page,
            "Comparison table: PageIndex 98.7% accuracy vs Vector RAG 85–90%",
            "Step 11  ·  Performance Comparison", to=550, hold_ms=HOLD_LONG)
        frames += scroll_shot(page,
            "Works with 8+ LLM providers: LM Studio, Ollama, OpenAI, Anthropic…",
            "Step 11  ·  Multi-Provider Support", to=1000, hold_ms=HOLD_LONG)

        # ── 12. Back to chat — outro ───────────────────────────────────────
        print("Step 12 — Final frame")
        page.locator("text=💬  Chat").click()
        time.sleep(1)
        frames += screenshot(page,
            "PageIndex Vectorless RAG — open source, provider-agnostic, production-ready",
            "✅  Demo Complete", hold_ms=HOLD_LONG, scroll_top=True)

        browser.close()

    # ── Assemble GIF ──────────────────────────────────────────────────────────
    print(f"\nAssembling {len(frames)} frames into GIF…")
    rgb_frames = [f.convert("RGB") for f in frames]

    # Each stored frame = 100 ms, so duration list matches hold_ms logic
    durations = [100] * len(rgb_frames)

    rgb_frames[0].save(
        OUT_GIF,
        format="GIF",
        save_all=True,
        append_images=rgb_frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )

    size_mb = OUT_GIF.stat().st_size / 1_048_576
    print(f"\n{'='*50}")
    print(f"  GIF saved  →  {OUT_GIF}")
    print(f"  Size       →  {size_mb:.1f} MB")
    print(f"  Frames     →  {len(rgb_frames)}")
    print(f"  Duration   →  ~{sum(durations)/1000:.0f}s")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    record()
