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
.gradio-container {
  max-width: 1560px !important; padding: 2px 14px 2px !important;
  --layout-gap: 4px; --form-gap-width: 1px;
}
main.app, main.fillable { padding: 2px 0 !important; }
/* Gradio pads every HTML block by 9px top and bottom — that is three wasted
   rows between the logo, the status line and the tabs. */
#chang-topbar .html-container, #status-bar .html-container,
.tts-panel .html-container { padding: 0 !important; }
.tab-wrapper { padding-bottom: 2px !important; }
footer { display: none !important; }

/* ---- header ---- */
#chang-topbar { align-items: center; margin-bottom: 0; }
.chang-header { display: flex; align-items: center; gap: 10px; padding: 0; }
.chang-logo { height: 32px; width: auto; }
.chang-row { display: flex; align-items: baseline; gap: 8px; }
.chang-name {
  font-size: 1.1rem; font-weight: 800; line-height: 1.1;
  background-image: linear-gradient(90deg, var(--chang-red) 0%, var(--chang-orange) 55%, var(--chang-gold) 100%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: var(--chang-orange);
}
.chang-cjk { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.24em; color: var(--chang-gold); }
.chang-tagline { font-size: 0.68rem; color: #977a5c; margin-top: 0; }
#lang-col { justify-content: center; }

/* ---- language pill ---- */
#lang-pick .wrap {
  display: flex; flex-wrap: nowrap; gap: 3px; padding: 3px;
  border: 1px solid var(--chang-line); border-radius: 999px;
  background: #fff; box-shadow: 0 1px 3px rgba(232, 106, 18, 0.06);
}
#lang-pick label {
  border: none !important; background: transparent !important; box-shadow: none !important;
  border-radius: 999px !important; padding: 2px 11px !important; font-size: 0.76rem !important;
}
#lang-pick label.selected, #lang-pick label:has(input:checked) {
  background: linear-gradient(90deg, var(--chang-red), var(--chang-orange) 60%, var(--chang-gold)) !important;
}
#lang-pick label.selected span, #lang-pick label:has(input:checked) span { color: #fff !important; }

/* ---- status bar ---- */
#status-bar { margin: 0 0 2px; }
.status-line {
  display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  font-size: 0.72rem; color: #7A5C3E; padding: 2px 9px;
  border: 1px solid var(--chang-line); border-radius: 8px; background: #FFFDF8;
}
.status-line b { color: var(--chang-red); font-weight: 700; }
.status-warn { color: #B3261E; font-weight: 700; }

/* ---- compact blocks ---- */
.block, .form { padding-top: 0 !important; padding-bottom: 0 !important; }
.gap { gap: 2px !important; }
.gr-group, .styler { padding: 2px 7px !important; }
.gr-group { margin-bottom: 3px !important; }
.section-head p {
  font-size: 0.74rem !important; margin: 0 !important; letter-spacing: 0.04em;
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

/* Sliders are the densest thing on the page — squeeze the label row, the
   number box and the track so three rows of them cost one row of height. */
.block .head { margin-bottom: 0 !important; align-items: center; gap: 2px !important; }
.block .head label { min-width: 0; }
.block .head span[data-testid="block-info"] {
  font-size: 0.7rem !important; line-height: 1.15 !important;
}
.block .head input[type="number"] {
  height: 19px !important; width: 40px !important; padding: 0 3px !important;
  font-size: 0.7rem !important;
}
.block .head .tab-like-container { gap: 1px !important; }
.reset-button { padding: 0 2px !important; font-size: 0.7rem !important; }
.slider_input_container { margin: 0 !important; }
.min_value, .max_value { font-size: 0.62rem !important; }
input[type="range"] { margin: 0 !important; }
.block > label > span, label > span[data-testid="block-info"] {
  margin-bottom: 1px !important; display: inline-block;
}
input[type="text"], input[type="number"], textarea, .gr-box {
  font-size: 0.84rem !important;
}

/* ---- audio players: keep every player to a couple of rows ---- */
#tts-audio .empty, .compact-audio .empty { min-height: 40px !important; }
#tts-audio, .compact-audio { min-height: 0 !important; }
#tts-audio .waveform-container, .compact-audio .waveform-container,
#tts-audio #waveform, .compact-audio #waveform {
  max-height: 46px !important; min-height: 0 !important; overflow: hidden;
}
#tts-audio .timestamps, .compact-audio .timestamps {
  font-size: 0.66rem !important; padding: 0 4px !important; margin: 0 !important;
}
#tts-audio .controls, .compact-audio .controls {
  padding: 1px 4px !important; margin: 0 !important; gap: 4px !important;
}
#tts-audio .component-wrapper, .compact-audio .component-wrapper { padding: 0 !important; }
#tts-audio .controls .icon, .compact-audio .controls .icon {
  height: 16px !important; width: 16px !important;
}
#tts-audio .source-selection button, .compact-audio .source-selection button {
  width: 22px !important; height: 22px !important;
}

/* ---- TTS single tab: three-column workspace that fits one window ---- */
#tts-main { align-items: stretch; gap: 8px !important; }
#tts-main > .column { gap: 5px !important; }
#tts-main .tts-panel { padding: 3px 8px 4px !important; flex: 1 1 auto; min-height: 0; }
#tts-main #tts-saved { flex: 0 0 auto; }
#tts-main .block { padding-top: 0 !important; padding-bottom: 0 !important; }
/* Text column: the textarea grows to fill the tallest sibling column. Don't set
   an explicit height on the column itself — that would cancel the row's
   align-items:stretch. Let it stretch, then fill down through the wrappers. */
#tts-col-text > * { height: 100%; }
#tts-col-text .tts-panel,
#tts-col-text .tts-panel > .styler {
  display: flex; flex-direction: column; min-height: 0;
}
#tts-col-text .tts-panel > .styler { flex: 1 1 auto !important; }
/* Gradio puts an inline flex-grow:0 on the .form wrapper — override it, or the
   textarea stops at its line count and leaves the panel half empty. */
#tts-col-text .tts-panel > .styler > .form {
  flex: 1 1 auto !important; display: flex; flex-direction: column; min-height: 0;
}
#tts-text {
  flex: 1 1 auto !important; min-height: 0 !important;
  display: flex; flex-direction: column;
}
#tts-text > label { flex: 1 1 auto; display: flex; flex-direction: column; min-height: 0; }
#tts-text .input-container { flex: 1 1 auto; display: flex; min-height: 0; }
#tts-text textarea {
  flex: 1 1 auto !important; height: 100% !important;
  min-height: 110px !important; resize: none; box-sizing: border-box;
}
/* Keep the reference-clip uploader/player from towering in clone mode: the
   upload drop zone is a .boundedheight button that defaults to ~240px. */
.compact-audio, .compact-audio .component-wrap { min-height: 0 !important; }
.compact-audio .audio-container button.boundedheight {
  min-height: 56px !important; font-size: 0.8rem !important;
}
.compact-audio .audio-container button.boundedheight > .wrap {
  flex-direction: row !important; flex-wrap: wrap; gap: 5px;
  align-items: center; justify-content: center;
}
.compact-audio .or { font-size: 0.72rem !important; }
.compact-audio .icon-wrap { width: 18px !important; margin: 0 !important; }
.compact-audio .audio-container .wrap { min-height: 0 !important; }
.compact-audio .audio-container button.boundedheight .icon-wrap { margin-bottom: 0 !important; }
.compact-audio .source-selection {
  padding: 0 !important; margin: 0 !important; min-height: 0 !important;
}
.compact-audio .source-selection button { padding: 1px 6px !important; }

/* Trim vertical bulk in the Voice & Model column so the whole tab clears a
   laptop viewport: one-line selector cards, no RAM subline, no chip. */
#tts-main .ms-section { margin: 0 0 2px !important; }
#tts-main .ms-title { font-size: 0.74rem; margin-bottom: 2px; }
#tts-main .ms-chip { display: none !important; }
#tts-main .ms-row { gap: 4px !important; }
#tts-main .ms-card { padding: 3px 9px !important; font-size: 0.76rem; }
#tts-main .ms-ram { display: none !important; }
#tts-main .ms-hint { margin-top: 1px !important; font-size: 0.66rem; line-height: 1.3; }
#tts-main .hint-text p { font-size: 0.68rem !important; line-height: 1.3 !important; margin: 1px 0 0 !important; }
#tts-main .section-head p { line-height: 1.5; }
/* Saved voices: label-less rows (the panel head names them) on two tight lines */
#tts-saved .row { gap: 4px !important; }
#tts-saved input[type="text"] { padding: 3px 7px !important; }
#tts-saved .wrap-inner, #tts-saved .secondary-wrap { padding: 1px 4px !important; }

/* ---- batch tab: two balanced columns + a one-row action bar ---- */
#tts-batch { align-items: stretch; gap: 8px !important; }
#tts-batch > .column { gap: 5px !important; }
#tts-batch .tts-panel { padding: 3px 8px 4px !important; }
#tts-batch-actions { align-items: center; gap: 10px !important; margin: 5px 0 4px; }
.compact-drop .center.boundedheight { min-height: 52px !important; }
.compact-drop .center.boundedheight > .wrap {
  flex-direction: row !important; flex-wrap: wrap; gap: 5px;
  align-items: center; justify-content: center; font-size: 0.8rem !important;
}
.compact-drop .icon-wrap { width: 18px !important; margin: 0 !important; }

/* ---- action bar: generate · output · preview · download ---- */
#tts-actions { align-items: center; gap: 8px !important; margin: 4px 0 3px; }
#tts-generate { height: 46px; min-height: 46px; font-size: 0.9rem !important; }
#tts-result-col { justify-content: center; gap: 2px !important; }
#tts-actions #tts-audio .empty { min-height: 38px !important; }
#tts-actions #tts-audio { min-height: 0 !important; }
/* inline Output (format + quality) — compact card so it rides the action bar */
#tts-output-inline {
  border: 1px solid var(--chang-line); border-radius: 10px;
  background: #FFFDF8; padding: 2px 9px 3px !important; gap: 0 !important;
  align-self: center;
}
#tts-output-inline .section-head p { margin-bottom: 1px !important; font-size: 0.68rem !important; }
#tts-output-inline .form, #tts-output-inline fieldset.block { gap: 1px !important; }
#tts-output-inline label[data-testid$="-radio-label"] {
  padding: 1px 7px !important; white-space: nowrap !important; font-size: 0.75rem !important;
}
/* keep Format | Quality side-by-side, pills on one line each */
#tts-output-row { flex-wrap: nowrap !important; gap: 8px !important; }
#tts-output-inline fieldset.block .wrap { flex-wrap: nowrap !important; }

/* ---- library: header icon buttons + per-row file list ---- */
#lib-head { align-items: center; gap: 4px !important; }
#lib-head .section-head { flex-grow: 1; }
.icon-btn {
  padding: 2px 6px !important; font-size: 0.95rem !important;
  min-width: 40px !important; max-width: 44px !important;
}
.lib-list {
  border: 1px solid var(--chang-line); border-radius: 8px; background: #fff;
  max-height: min(122px, 16vh); overflow-y: auto; font-size: 0.76rem;
}
.lib-row {
  display: grid; grid-template-columns: minmax(0, 1fr) 60px 112px 96px;
  gap: 8px; align-items: center; padding: 1px 10px;
  border-bottom: 1px solid var(--chang-line);
}
.lib-row:last-child { border-bottom: none; }
.lib-head-row {
  font-weight: 700; position: sticky; top: 0; z-index: 1; background: #FFFDF8;
}
/* min-width:0 lets the name track shrink instead of forcing overflow, and the
   2-line clamp stops long Chinese filenames from stacking one glyph per line. */
.lib-name {
  min-width: 0; overflow-wrap: anywhere; word-break: break-word;
  display: -webkit-box; -webkit-line-clamp: 1; -webkit-box-orient: vertical;
  overflow: hidden; line-height: 1.7;
}
.lib-size, .lib-date { color: #7A5C3E; font-size: 0.72rem; white-space: nowrap; }
.lib-actions { display: flex; gap: 4px; justify-content: flex-end; }
.lib-btn {
  border: 1px solid var(--chang-line); background: #FFFDF8; border-radius: 6px;
  padding: 0 6px; cursor: pointer; font-size: 0.8rem; line-height: 1.5;
}
.lib-btn:hover { background: #FFF3E0; border-color: var(--chang-orange); }
.lib-row.playing { background: #FFF7E8; }
.lib-row.playing .lib-name { color: var(--chang-red); font-weight: 700; }
.lib-empty { padding: 10px; color: #a58969; }

/* ---- model selector cards (shared by every AI category) ---- */
.ms-holder { padding: 0 !important; }
.ms-section { margin: 2px 0 6px; }
.ms-title {
  font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--chang-orange); margin-bottom: 3px;
}
.ms-chip {
  display: inline-block; font-size: 0.68rem; font-weight: 600; color: #7A5C3E;
  background: #FFF3E0; border: 1px solid var(--chang-line);
  border-radius: 999px; padding: 1px 9px; margin-bottom: 5px;
}
.ms-row { display: flex; flex-wrap: wrap; gap: 6px; }
.ms-card {
  display: flex; align-items: center; gap: 8px; padding: 6px 12px;
  background: #fff; border: 1px solid var(--chang-line); border-radius: 12px;
  box-shadow: 0 1px 3px rgba(232, 106, 18, 0.08);
  cursor: pointer; user-select: none; font-size: 0.8rem; color: #43301C;
  transition: box-shadow 0.12s, border-color 0.12s;
}
.ms-card:hover { border-color: var(--chang-orange); box-shadow: 0 2px 6px rgba(232, 106, 18, 0.16); }
.ms-card.selected {
  background: linear-gradient(90deg, var(--chang-red), var(--chang-orange) 60%, var(--chang-gold));
  border-color: transparent; color: #fff;
  box-shadow: 0 2px 6px rgba(232, 106, 18, 0.28);
}
.ms-card.disabled { opacity: 0.45; cursor: not-allowed; pointer-events: none; }
.ms-radio {
  width: 14px; height: 14px; min-width: 14px; display: inline-block;
  border: 2px solid #C9AE8C; border-radius: 50%; background: #fff; position: relative;
}
.ms-card.selected .ms-radio { border-color: #fff; background: transparent; }
.ms-card.selected .ms-radio::after {
  content: ""; position: absolute; inset: 2px; border-radius: 50%; background: #fff;
}
.ms-text { display: flex; flex-direction: column; line-height: 1.25; }
.ms-name { font-weight: 600; }
.ms-ram { font-size: 0.68rem; color: #a58969; }
.ms-card.selected .ms-ram { color: rgba(255, 255, 255, 0.88); }
.ms-tag {
  font-size: 0.64rem; font-weight: 700; border-radius: 999px;
  padding: 1px 7px; white-space: nowrap;
}
.ms-tag.ok { background: #E8F5E9; color: #2E7D32; }
.ms-tag.miss { background: #F5F0EA; color: #8a7660; }
.ms-tag.rec { background: #FFF3D6; color: #A05A00; border: 1px solid #FDB913; }
.ms-tag.mem { background: #FDECEA; color: #B3261E; }
.ms-card.selected .ms-tag { border-color: transparent; }
.ms-dl {
  border: 1px solid var(--chang-line); background: #FFFDF8; border-radius: 8px;
  padding: 1px 8px; cursor: pointer; font-size: 0.72rem; color: #43301C;
}
.ms-dl:hover { background: #FFF3E0; border-color: var(--chang-orange); }
.ms-hint { font-size: 0.7rem; color: #a58969; margin-top: 3px; }
#model-evt { display: none !important; }

/* ---- tabs ---- */
.tab-nav { margin-bottom: 3px !important; }
.tab-nav button, button.tab-item {
  font-weight: 600 !important; font-size: 0.85rem !important; padding: 5px 10px !important;
}
.tabitem { padding: 3px 0 0 !important; border: none !important; }
.tab-nav button.selected, button.selected.tab-item {
  color: var(--chang-red) !important;
  border-color: var(--chang-orange) !important;
}

/* ---- results footer ---- */
.result-info p { font-size: 0.74rem !important; margin: 1px 0 !important; line-height: 1.35; }
.result-info code { font-size: 0.7rem !important; overflow-wrap: anywhere; }
.app-footer p { font-size: 0.68rem !important; color: #a58969 !important; margin: 1px 0 0 !important; }
"""

# One shared Audio element for the library: clicking ▶ on a row stops whatever
# was playing and plays that row's file. Reveal/preview clicks are forwarded to
# Python by writing JSON into the hidden #lib-evt textbox.
LIBRARY_HEAD = """
<style>
/* ---- fit-to-window safety net ----
   The layout is sized so the TTS tab clears a ~860px-tall viewport with no page
   scroll. On shorter windows scale the page down a notch rather than hand the
   user a scrollbar — same layout, just smaller. This lives here rather than in
   CHANG_CSS because Gradio rewrites custom-CSS selectors and drops the queries. */
@media (max-height: 855px) { .gradio-container { zoom: 0.94; } }
@media (max-height: 800px) { .gradio-container { zoom: 0.88; } }
@media (max-height: 730px) { .gradio-container { zoom: 0.82; } }
@media (max-height: 660px) { .gradio-container { zoom: 0.75; } }
</style>
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
window.changModelPick = (el) => {
  if (el.classList.contains('disabled') || el.classList.contains('selected')) return;
  const box = document.querySelector('#model-evt textarea');
  if (!box) return;
  box.value = JSON.stringify({action: 'select', cat: el.dataset.cat, value: el.dataset.value, t: Date.now()});
  box.dispatchEvent(new Event('input', {bubbles: true}));
};
window.changModelInstall = (ev, btn) => {
  ev.stopPropagation();
  const box = document.querySelector('#model-evt textarea');
  if (!box) return;
  btn.textContent = '⏳';
  box.value = JSON.stringify({action: 'install', model: btn.dataset.model, t: Date.now()});
  box.dispatchEvent(new Event('input', {bubbles: true}));
};
document.addEventListener('keydown', (ev) => {
  const input = ev.target;
  if (!(input instanceof HTMLTextAreaElement) || !input.closest('#chat-message')) return;
  if (ev.key !== 'Enter' || ev.isComposing || ev.keyCode === 229) return;

  if (ev.metaKey) {
    ev.preventDefault();
    ev.stopImmediatePropagation();
    input.setRangeText('\\n', input.selectionStart, input.selectionEnd, 'end');
    input.dispatchEvent(new Event('input', {bubbles: true}));
    return;
  }

  if (!ev.shiftKey && !ev.ctrlKey && !ev.altKey) {
    ev.preventDefault();
    ev.stopImmediatePropagation();
    document.querySelector('#chat-send')?.click();
  }
}, true);
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
