#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chang 日日向上 — Local AI Toolbox (Apple Silicon, 100% local)
=============================================================

One compact Gradio window with tabs:
  TTS · Chat · Vision · Transcribe · OCR · Documents · Models · Settings

Only one heavy model (TTS / chat / vision / whisper) is kept in memory at a
time — see modules/memory_manager.py. Models are installed from the Models
tab or manage_models.command; nothing is downloaded automatically.

Usage:
  python app.py             # start the web app on http://127.0.0.1:7860
  python app.py --download  # legacy: download both Qwen3-TTS models, then exit
"""

from __future__ import annotations

import os
import sys


def _fatal(message: str) -> None:
    print("\n" + "=" * 70)
    print("STARTUP ERROR")
    print("=" * 70)
    print(message)
    print("=" * 70 + "\n")
    sys.exit(1)


try:
    import gradio as gr
except ImportError:
    _fatal(
        "Gradio is not installed.\n"
        "Fix: run install.command again, or inside the project folder run:\n"
        "  ./.venv/bin/pip install -r requirements.txt"
    )

try:
    import mlx.core  # noqa: F401
    import mlx_audio  # noqa: F401
except ImportError:
    _fatal(
        "MLX-Audio is not installed (or not for Apple Silicon).\n"
        "Fix: run install.command again, or inside the project folder run:\n"
        "  ./.venv/bin/pip install -r requirements.txt\n"
        "MLX-Audio only works on Apple Silicon Macs (M1/M2/M3/M4)."
    )

from modules import branding, chat, documents, generator_tab, memory_manager as mm
from modules import model_manager as mgr
from modules import ocr, storage, transcription, tts, vision


# --------------------------------------------------------------------------- #
# Status bar
# --------------------------------------------------------------------------- #
def status_html() -> str:
    settings = storage.load_settings()
    used, total, _avail = mm.ram_status()
    warn = mm.memory_warning(float(settings.get("mem_warn_gb", 13.0)))
    warn_html = f'<span class="status-warn">{warn}</span>' if warn else ""
    return (
        '<div class="status-line">'
        f"<span>🧠 Model: <b>{mm.loaded_summary()}</b></span>"
        f"<span>📊 RAM: <b>{used:.1f} / {total:.0f} GB</b></span>"
        f"<span>⚙️ Task: <b>{mm.get_task()}</b></span>"
        f"{warn_html}"
        "</div>"
    )


def build_ui() -> "gr.Blocks":
    settings = storage.load_settings()
    force_light = bool(settings.get("force_light_theme", True))

    with gr.Blocks(
        title="Chang 日日向上 · Local AI Toolbox",
        theme=branding.THEME,
        css=branding.CHANG_CSS,
        head=branding.LIBRARY_HEAD,
        js=branding.FORCE_LIGHT_JS if force_light else None,
    ) as demo:
        lang_store = gr.BrowserState(tts.DEFAULT_LANG, storage_key="chang_tts_lang")

        # ---- Header: logo + brand + language toggle ----
        with gr.Row(elem_id="chang-topbar"):
            with gr.Column(scale=1, min_width=380):
                header = gr.HTML(branding.header_html(tts.DEFAULT_LANG))
            with gr.Column(scale=0, min_width=320, elem_id="lang-col"):
                lang = gr.Radio(
                    choices=tts.LANG_CHOICES, value=tts.DEFAULT_LANG, show_label=False,
                    container=False, elem_id="lang-pick",
                )

        # ---- Status bar ----
        with gr.Row(elem_id="status-bar"):
            with gr.Column(scale=8):
                status = gr.HTML(status_html())
            with gr.Column(scale=1, min_width=150):
                unload_btn = gr.Button("⏏️ Unload model", size="sm")

        # ---- Tabs ----
        with gr.Tab("🎧 TTS"):
            i18n_outputs, i18n_updates = tts.build_tts_tab(lang, settings)
        with gr.Tab("💬 Chat"):
            chat.build_chat_tab(settings)
        with gr.Tab("🖼 Vision"):
            vision.build_vision_tab(settings)
        with gr.Tab("🎙 Transcribe"):
            transcription.build_transcribe_tab(settings)
        with gr.Tab("🔎 OCR"):
            ocr.build_ocr_tab(settings)
        with gr.Tab("📚 Documents"):
            documents.build_documents_tab(settings)
        with gr.Tab("🏗 Generator"):
            generator_tab.build_generator_tab(settings)
        with gr.Tab("📦 Models"):
            mgr.build_models_tab()
        with gr.Tab("⚙️ Settings"):
            storage.build_settings_tab()

        # ---- Status refresh + idle auto-unload ----
        timer = gr.Timer(4)

        def on_tick():
            s = storage.load_settings()
            mm.auto_unload_check(
                bool(s.get("auto_unload", True)),
                float(s.get("auto_unload_minutes", 10)),
            )
            return status_html()

        timer.tick(on_tick, outputs=[status], show_progress="hidden")

        def on_unload():
            mm.unload_all()
            gr.Info("All models unloaded — memory released.")
            return status_html()

        unload_btn.click(on_unload, outputs=[status])

        # ---- Language switching (header + TTS tab) ----
        def on_lang_change(lg: str):
            lg = lg if lg in tts.T else tts.DEFAULT_LANG
            return [lg, gr.update(value=branding.header_html(lg)), *i18n_updates(lg)]

        lang.input(
            fn=on_lang_change, inputs=[lang],
            outputs=[lang_store, header, *i18n_outputs], show_progress="hidden",
        )

        def on_page_load(stored: str):
            lg = stored if stored in tts.T else tts.DEFAULT_LANG
            return [gr.update(value=lg), gr.update(value=branding.header_html(lg)),
                    *i18n_updates(lg)]

        demo.load(
            fn=on_page_load, inputs=[lang_store],
            outputs=[lang, header, *i18n_outputs], show_progress="hidden",
        )

    return demo


# --------------------------------------------------------------------------- #
# Legacy CLI: download both TTS models (kept for backwards compatibility)
# --------------------------------------------------------------------------- #
def download_tts_models() -> None:
    for key in ("tts-0.6b", "tts-1.7b"):
        info = mgr.MODELS[key]
        existing = mgr.installed_path(key)
        if existing is not None:
            print(f"{info['name']}: already installed at {existing}")
            continue
        print(f"Downloading {info['name']} ({info['size_gb']:.1f} GB)…")
        target = mgr.install(key)
        print(f"  OK -> {target}")


if __name__ == "__main__":
    if "--download" in sys.argv:
        download_tts_models()
        sys.exit(0)
    logo_file = branding.find_logo_file()
    build_ui().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=os.environ.get("QWEN3_TTS_NO_BROWSER") != "1",
        show_error=True,
        favicon_path=str(logo_file) if logo_file else None,
        # The library plays files straight from the outputs folder.
        allowed_paths=[str(storage.outputs_dir().resolve())],
    )
