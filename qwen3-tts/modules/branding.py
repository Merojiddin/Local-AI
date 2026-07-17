"""Chang 日日向上 sunrise branding — logo, Gradio theme and CSS.

Extracted unchanged from the original TTS app; shared by every tab.
"""

from __future__ import annotations

import base64
from pathlib import Path

import gradio as gr

from .storage import BASE_DIR

ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

LOGO_FILES = ("logo.png", "logo.jpg", "logo.jpeg", "logo.webp", "logo.svg")
LOGO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}

FALLBACK_LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 430">
  <defs>
    <linearGradient id="ray" x1="0" y1="1" x2="0.55" y2="0">
      <stop offset="0" stop-color="#FDB913"/>
      <stop offset="0.45" stop-color="#F58220"/>
      <stop offset="1" stop-color="#E8380D"/>
    </linearGradient>
    <linearGradient id="word" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#E8380D"/>
      <stop offset="0.6" stop-color="#F58220"/>
      <stop offset="1" stop-color="#FDB913"/>
    </linearGradient>
  </defs>
  <g>
    <path d="M250 296 C214 208 212 106 250 40 C288 106 286 208 250 296 Z" fill="url(#ray)" transform="rotate(-66 250 300)"/>
    <path d="M250 296 C214 208 212 106 250 40 C288 106 286 208 250 296 Z" fill="url(#ray)" transform="rotate(-44 250 300)"/>
    <path d="M250 296 C214 208 212 106 250 40 C288 106 286 208 250 296 Z" fill="url(#ray)" transform="rotate(-22 250 300)"/>
    <path d="M250 296 C214 208 212 106 250 40 C288 106 286 208 250 296 Z" fill="url(#ray)"/>
    <path d="M250 296 C214 208 212 106 250 40 C288 106 286 208 250 296 Z" fill="url(#ray)" transform="rotate(22 250 300)"/>
    <path d="M250 296 C214 208 212 106 250 40 C288 106 286 208 250 296 Z" fill="url(#ray)" transform="rotate(44 250 300)"/>
    <path d="M250 296 C214 208 212 106 250 40 C288 106 286 208 250 296 Z" fill="url(#ray)" transform="rotate(66 250 300)"/>
  </g>
  <path d="M206 300 A44 44 0 0 1 294 300 Z" fill="#FFDF00"/>
  <text x="255" y="402" font-family="'Snell Roundhand','Brush Script MT',cursive" font-size="105" font-weight="700" fill="url(#word)" text-anchor="middle">Chang</text>
  <text x="420" y="348" font-family="'PingFang SC','Hiragino Sans GB',sans-serif" font-size="40" font-weight="600" letter-spacing="12" fill="#F5A623">日日向上</text>
</svg>"""


def find_logo_file() -> Path | None:
    for name in LOGO_FILES:
        p = ASSETS_DIR / name
        try:
            if p.exists() and p.stat().st_size > 0:
                return p
        except OSError:
            continue
    return None


def logo_data_uri() -> str:
    p = find_logo_file()
    if p is not None:
        mime = LOGO_MIME.get(p.suffix.lower(), "image/png")
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"
    data = base64.b64encode(FALLBACK_LOGO_SVG.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{data}"


CHANG_PRIMARY = gr.themes.Color(
    "#FFF8ED", "#FFEFD2", "#FFDFA8", "#FEC873", "#FBA43B", "#F58220",
    "#E86A12", "#C7530E", "#9E3F10", "#7F300F", "#5A210B",
)

THEME = gr.themes.Soft(
    primary_hue=CHANG_PRIMARY,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.stone,
    spacing_size=gr.themes.sizes.spacing_sm,
    text_size=gr.themes.sizes.text_sm,
    radius_size=gr.themes.sizes.radius_lg,
    font=["-apple-system", "BlinkMacSystemFont", "Segoe UI", "PingFang SC", "Helvetica Neue", "sans-serif"],
).set(
    body_background_fill="linear-gradient(180deg, #FFFBF4 0%, #FFF4E4 100%)",
    block_background_fill="#FFFFFF",
    block_border_color="#F3E0C7",
    block_border_width="1px",
    block_shadow="0 1px 4px rgba(214, 120, 30, 0.07)",
    block_title_text_color="#7A5C3E",
    body_text_color="#43301C",
    button_primary_background_fill="linear-gradient(90deg, #E8380D 0%, #F58220 55%, #FDB913 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #C92F0B 0%, #DE7212 55%, #E3A605 100%)",
    button_primary_text_color="#FFFFFF",
    slider_color="#F58220",
)

CHANG_CSS = """
:root {
  --chang-red: #E8380D;
  --chang-orange: #F58220;
  --chang-gold: #FDB913;
  --chang-line: #F3E0C7;
}
.gradio-container { max-width: 1500px !important; padding: 8px 18px 6px !important; }
footer { display: none !important; }

/* ---- header ---- */
#chang-topbar { align-items: center; margin-bottom: 2px; }
.chang-header { display: flex; align-items: center; gap: 14px; padding: 2px 0; }
.chang-logo { height: 48px; width: auto; }
.chang-row { display: flex; align-items: baseline; gap: 10px; }
.chang-name {
  font-size: 1.35rem; font-weight: 800; line-height: 1.1;
  background-image: linear-gradient(90deg, var(--chang-red) 0%, var(--chang-orange) 55%, var(--chang-gold) 100%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: var(--chang-orange);
}
.chang-cjk { font-size: 0.9rem; font-weight: 700; letter-spacing: 0.3em; color: var(--chang-gold); }
.chang-tagline { font-size: 0.75rem; color: #977a5c; margin-top: 1px; }
#lang-col { justify-content: center; }

/* ---- language pill ---- */
#lang-pick .wrap {
  display: flex; flex-wrap: nowrap; gap: 3px; padding: 3px;
  border: 1px solid var(--chang-line); border-radius: 999px;
  background: #fff; box-shadow: 0 1px 3px rgba(232, 106, 18, 0.06);
}
#lang-pick label {
  border: none !important; background: transparent !important; box-shadow: none !important;
  border-radius: 999px !important; padding: 3px 12px !important; font-size: 0.8rem !important;
}
#lang-pick label.selected, #lang-pick label:has(input:checked) {
  background: linear-gradient(90deg, var(--chang-red), var(--chang-orange) 60%, var(--chang-gold)) !important;
}
#lang-pick label.selected span, #lang-pick label:has(input:checked) span { color: #fff !important; }

/* ---- status bar ---- */
#status-bar { margin: 0 0 4px; }
.status-line {
  display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
  font-size: 0.76rem; color: #7A5C3E; padding: 4px 10px;
  border: 1px solid var(--chang-line); border-radius: 8px; background: #FFFDF8;
}
.status-line b { color: var(--chang-red); font-weight: 700; }
.status-warn { color: #B3261E; font-weight: 700; }

/* ---- compact blocks ---- */
.block, .form { padding-top: 4px !important; padding-bottom: 4px !important; }
.gap { gap: 6px !important; }
.gr-group, .styler { padding: 6px 8px !important; }
.section-head p {
  font-size: 0.8rem !important; margin: 0 !important; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--chang-orange) !important;
}
.section-head strong { color: var(--chang-red); }
span[data-testid="block-info"], label span { font-size: 0.78rem !important; }
.wrap.default { min-height: 0 !important; }
.hint-text p { font-size: 0.75rem !important; color: #a58969 !important; margin: 0 0 4px !important; }

/* ---- tabs ---- */
.tab-nav button, button.tab-item { font-weight: 600 !important; }
.tab-nav button.selected, button.selected.tab-item {
  color: var(--chang-red) !important;
  border-color: var(--chang-orange) !important;
}

/* ---- results footer ---- */
.result-info p { font-size: 0.8rem !important; margin: 2px 0 !important; }
.app-footer p { font-size: 0.72rem !important; color: #a58969 !important; margin: 2px 0 0 !important; }
"""

# Keep the whole app in light mode so the warm palette and logo always match.
FORCE_LIGHT_JS = """
() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.href);
  }
}
"""

TAGLINES = {
    "en": "Local AI toolbox — TTS, chat, vision, speech-to-text, OCR & document search. 100% on your Mac.",
    "vi": "Bộ công cụ AI cục bộ — TTS, chat, hình ảnh, chuyển giọng nói thành văn bản, OCR & tìm kiếm tài liệu. Chạy 100% trên máy của bạn.",
    "zh": "本地 AI 工具箱 — 语音合成、聊天、图像、语音转文字、OCR 与文档搜索，完全在你的电脑上运行。",
}


def header_html(lang: str) -> str:
    tagline = TAGLINES.get(lang, TAGLINES["en"])
    return f"""
<div class="chang-header">
  <img class="chang-logo" src="{logo_data_uri()}" alt="Chang 日日向上"/>
  <div class="chang-title">
    <div class="chang-row"><span class="chang-name">Chang</span><span class="chang-cjk">日日向上</span></div>
    <div class="chang-tagline">{tagline}</div>
  </div>
</div>"""
