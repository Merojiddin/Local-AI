"""Outputs library: browse the outputs folder, play audio, preview text files."""

from __future__ import annotations

import html
import json
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import gradio as gr

from .. import storage


# --------------------------------------------------------------------------- #
# Outputs library — browse the outputs folder, play audio, preview text files
# --------------------------------------------------------------------------- #
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
TEXT_EXTS = {".csv", ".txt", ".json", ".md", ".srt", ".vtt"}


def _fmt_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{max(1, n // 1024)} KB"


def _output_files() -> list[Path]:
    try:
        files = [
            p for p in storage.outputs_dir().iterdir()
            if p.is_file() and not p.name.startswith(".")
        ]
    except OSError:
        return []
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def lib_html() -> str:
    """Outputs listing with per-row buttons. Play/stop are pure JS against a
    shared Audio element (functions in branding.LIBRARY_HEAD); reveal/preview
    round-trip to Python through the hidden #lib-evt textbox."""
    rows = [
        '<div class="lib-row lib-head-row">'
        "<span>File</span><span>Size</span><span>Modified</span><span></span></div>"
    ]
    for p in _output_files():
        st = p.stat()
        ext = p.suffix.lower()
        name = html.escape(p.name)
        src = html.escape("/gradio_api/file=" + quote(str(p.resolve())))
        btns = []
        if ext in AUDIO_EXTS:
            btns.append('<button class="lib-btn" title="Play" '
                        'onclick="changLibPlay(this)">▶</button>')
            btns.append('<button class="lib-btn" title="Stop" '
                        'onclick="changLibStop()">⏹</button>')
        elif ext in TEXT_EXTS:
            btns.append('<button class="lib-btn" title="Preview" '
                        "onclick=\"changLibEvt('preview', this)\">📄</button>")
        btns.append('<button class="lib-btn" title="Show in Finder" '
                    "onclick=\"changLibEvt('reveal', this)\">📂</button>")
        rows.append(
            f'<div class="lib-row" data-name="{name}" data-src="{src}">'
            f'<span class="lib-name">{name}</span>'
            f'<span class="lib-size">{_fmt_size(st.st_size)}</span>'
            f'<span class="lib-date">'
            f'{datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")}</span>'
            f'<span class="lib-actions">{"".join(btns)}</span></div>'
        )
    if len(rows) == 1:
        rows.append('<div class="lib-empty">No files yet.</div>')
    return f'<div class="lib-list">{"".join(rows)}</div>'


def ui_files_refresh():
    return gr.update(value=lib_html())


def ui_lib_event(payload: str):
    """Reveal/preview clicks arriving from the library HTML via #lib-evt."""
    try:
        data = json.loads(payload or "{}")
    except json.JSONDecodeError:
        return gr.update()
    name = Path(str(data.get("name", ""))).name
    path = storage.outputs_dir() / name
    if not name or not path.is_file():
        return gr.update()
    if data.get("action") == "reveal":
        subprocess.run(["open", "-R", str(path)], check=False)
        return gr.update()
    if data.get("action") == "preview" and path.suffix.lower() in TEXT_EXTS:
        try:
            content = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            raise gr.Error(f"Could not read file: {exc}")
        if len(content) > 20000:
            content = content[:20000] + "\n…"
        return gr.update(value=content, visible=True)
    return gr.update()

