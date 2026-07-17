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

# The official Chang logo lives in assets/. logo_header.png is a small
# pre-resized copy embedded inline in the page header; logo.png is the
# full-resolution original (also used as the favicon).
LOGO_FILES = ("logo.png", "logo.jpg", "logo.jpeg", "logo.webp", "logo.svg")
HEADER_LOGO = "logo_header.png"
LOGO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}

_LOGO_URI_CACHE: dict[tuple[str, float], str] = {}


def find_logo_file() -> Path | None:
    for name in LOGO_FILES:
        p = ASSETS_DIR / name
        try:
            if p.exists() and p.stat().st_size > 0:
                return p
        except OSError:
            continue
    return None


def _header_logo_file() -> Path | None:
    p = ASSETS_DIR / HEADER_LOGO
    try:
        if p.exists() and p.stat().st_size > 0:
            return p
    except OSError:
        pass
    return find_logo_file()


def logo_data_uri() -> str | None:
    p = _header_logo_file()
    if p is None:
        return None
    key = (str(p), p.stat().st_mtime)
    if key not in _LOGO_URI_CACHE:
        mime = LOGO_MIME.get(p.suffix.lower(), "image/png")
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        _LOGO_URI_CACHE.clear()
        _LOGO_URI_CACHE[key] = f"data:{mime};base64,{data}"
    return _LOGO_URI_CACHE[key]


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
.gradio-container { max-width: 1560px !important; padding: 6px 16px 4px !important; }
footer { display: none !important; }

/* ---- header ---- */
#chang-topbar { align-items: center; margin-bottom: 2px; }
.chang-header { display: flex; align-items: center; gap: 12px; padding: 2px 0; }
.chang-logo { height: 44px; width: auto; }
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
.block, .form { padding-top: 2px !important; padding-bottom: 2px !important; }
.gap { gap: 4px !important; }
.gr-group, .styler { padding: 4px 8px !important; }
.section-head p {
  font-size: 0.78rem !important; margin: 0 !important; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--chang-orange) !important;
}
.section-head strong { color: var(--chang-red); }
span[data-testid="block-info"], label span { font-size: 0.78rem !important; }
.wrap.default { min-height: 0 !important; }
.hint-text p { font-size: 0.75rem !important; color: #a58969 !important; margin: 0 0 4px !important; }

/* ---- compact radio pills (all radio groups except the language pill) ---- */
label[data-testid$="-radio-label"] {
  padding: 2px 10px !important; font-size: 0.8rem !important;
}
fieldset.block { gap: 2px !important; }

/* ---- audio players: shrink the empty placeholder ---- */
#tts-audio .empty { min-height: 56px !important; }
#tts-audio { min-height: 0 !important; }

/* ---- library: header icon buttons + per-row file list ---- */
#lib-head { align-items: center; gap: 4px !important; }
#lib-head .section-head { flex-grow: 1; }
.icon-btn {
  padding: 2px 6px !important; font-size: 0.95rem !important;
  min-width: 40px !important; max-width: 44px !important;
}
.lib-list {
  border: 1px solid var(--chang-line); border-radius: 8px; background: #fff;
  max-height: 320px; overflow-y: auto; font-size: 0.8rem;
}
.lib-row {
  display: grid; grid-template-columns: 1fr 58px 112px 104px;
  gap: 6px; align-items: center; padding: 4px 8px;
  border-bottom: 1px solid var(--chang-line);
}
.lib-row:last-child { border-bottom: none; }
.lib-head-row {
  font-weight: 700; position: sticky; top: 0; z-index: 1; background: #FFFDF8;
}
.lib-name { overflow-wrap: anywhere; }
.lib-size, .lib-date { color: #7A5C3E; font-size: 0.72rem; white-space: nowrap; }
.lib-actions { display: flex; gap: 4px; justify-content: flex-end; }
.lib-btn {
  border: 1px solid var(--chang-line); background: #FFFDF8; border-radius: 6px;
  padding: 1px 7px; cursor: pointer; font-size: 0.85rem; line-height: 1.4;
}
.lib-btn:hover { background: #FFF3E0; border-color: var(--chang-orange); }
.lib-row.playing { background: #FFF7E8; }
.lib-row.playing .lib-name { color: var(--chang-red); font-weight: 700; }
.lib-empty { padding: 10px; color: #a58969; }

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

# One shared Audio element for the library: clicking ▶ on a row stops whatever
# was playing and plays that row's file. Reveal/preview clicks are forwarded to
# Python by writing JSON into the hidden #lib-evt textbox.
LIBRARY_HEAD = """
<script>
window.changLibStop = () => {
  const a = window._changLibAudio;
  if (a) { a.pause(); a.currentTime = 0; }
  document.querySelectorAll('.lib-row.playing').forEach(r => r.classList.remove('playing'));
};
window.changLibPlay = (btn) => {
  const row = btn.closest('.lib-row');
  if (!row || !row.dataset.src) return;
  window.changLibStop();
  const a = window._changLibAudio = window._changLibAudio || new Audio();
  a.src = row.dataset.src;
  a.onended = () => row.classList.remove('playing');
  a.play();
  row.classList.add('playing');
};
window.changLibEvt = (action, btn) => {
  const row = btn.closest('.lib-row');
  const box = document.querySelector('#lib-evt textarea');
  if (!row || !box) return;
  box.value = JSON.stringify({action: action, name: row.dataset.name, t: Date.now()});
  box.dispatchEvent(new Event('input', {bubbles: true}));
};
</script>
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
    uri = logo_data_uri()
    logo_img = f'<img class="chang-logo" src="{uri}" alt="Chang 日日向上"/>' if uri else ""
    return f"""
<div class="chang-header">
  {logo_img}
  <div class="chang-title">
    <div class="chang-row"><span class="chang-name">Chang</span><span class="chang-cjk">日日向上</span></div>
    <div class="chang-tagline">{tagline}</div>
  </div>
</div>"""
