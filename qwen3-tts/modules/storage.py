"""Paths, user settings, disk-space helpers and the Settings tab.

Layout (all separate, so removing model weights never touches user data):
  <models dir>        model weights   (default ~/Library/Application Support/LocalAIToolbox/models)
  outputs/            generated audio / transcripts (user data)
  data/indexes/       document-search indexes       (user data)
  cache/              TTS audio cache
  cache/temp/         disposable temporary files
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SETTINGS_FILE = BASE_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "models_dir": "~/Library/Application Support/LocalAIToolbox/models",
    "outputs_dir": "outputs",
    "indexes_dir": "data/indexes",
    "collections_dir": "data/collections",
    "generator_dir": "data/generator_projects",
    "default_tts_model": "Higher Quality — Qwen3-TTS 1.7B",
    # Per-category model selections (see modules/model_select.py). Values are
    # registry keys from models.json; unknown values fall back to these
    # defaults when read.
    "chat_model": "chat-8b",
    "coding_model": "kiwi-8b",
    "vision_model": "vision-4b",
    "embed_model": "embed-0.6b",
    "rerank_model": "rerank-0.6b",
    "stt_model": "whisper",
    "tts_voice_mode": "CustomVoice",
    "ocr_engine": "apple",
    "force_light_theme": True,
    "auto_unload": True,
    "auto_unload_minutes": 10,
    "show_advanced": True,
    "mem_warn_gb": 13.0,
}


def load_settings() -> dict:
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            settings.update({k: data[k] for k in DEFAULT_SETTINGS if k in data})
    except (OSError, ValueError):
        pass
    return settings


def save_settings(settings: dict) -> None:
    tmp = SETTINGS_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
    tmp.replace(SETTINGS_FILE)


def _resolve(value: str) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = BASE_DIR / p
    return p


def models_dir() -> Path:
    p = _resolve(load_settings()["models_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def outputs_dir() -> Path:
    p = _resolve(load_settings()["outputs_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def indexes_dir() -> Path:
    p = _resolve(load_settings()["indexes_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def collections_dir() -> Path:
    p = _resolve(load_settings()["collections_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def generator_dir() -> Path:
    p = _resolve(load_settings()["generator_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = BASE_DIR / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def temp_dir() -> Path:
    p = cache_dir() / "temp"
    p.mkdir(parents=True, exist_ok=True)
    return p


def free_disk_gb(path: Path | None = None) -> float:
    target = path or models_dir()
    while not target.exists():
        target = target.parent
    return shutil.disk_usage(target).free / 1e9


def dir_size_gb(path: Path) -> float:
    total = 0
    if path.exists():
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
    return total / 1e9


def clear_temp_cache() -> float:
    """Delete everything inside cache/temp only. Returns freed GB."""
    td = temp_dir()
    freed = dir_size_gb(td)
    for child in td.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError:
            continue
    return freed


def open_in_finder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["open", str(path)], check=False)


# --------------------------------------------------------------------------- #
# Settings tab
# --------------------------------------------------------------------------- #
def build_settings_tab():
    import gradio as gr

    s = load_settings()

    gr.Markdown("Changes to folders and theme take effect after a restart.", elem_classes="hint-text")
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("**Folders**", elem_classes="section-head")
                in_models = gr.Textbox(value=s["models_dir"], label="Model storage (can point to an external SSD)")
                in_outputs = gr.Textbox(value=s["outputs_dir"], label="Outputs folder")
                in_indexes = gr.Textbox(value=s["indexes_dir"], label="Document index folder")
                with gr.Row():
                    open_models_btn = gr.Button("Open models", size="sm")
                    open_outputs_btn = gr.Button("Open outputs", size="sm")
                    open_indexes_btn = gr.Button("Open indexes", size="sm")
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("**Behaviour**", elem_classes="section-head")
                in_default_tts = gr.Radio(
                    choices=["Higher Quality — Qwen3-TTS 1.7B"],
                    value="Higher Quality — Qwen3-TTS 1.7B", label="Default TTS model",
                )
                in_theme = gr.Checkbox(value=bool(s["force_light_theme"]), label="Force light theme (matches the Chang branding)")
                in_auto_unload = gr.Checkbox(value=bool(s["auto_unload"]), label="Automatically unload idle models")
                in_unload_min = gr.Slider(2, 60, value=float(s["auto_unload_minutes"]), step=1, label="Unload after idle (minutes)")
                in_advanced = gr.Checkbox(value=bool(s["show_advanced"]), label="Show advanced controls (accordions open by default)")
                in_warn = gr.Slider(8, 15.5, value=float(s["mem_warn_gb"]), step=0.5, label="Memory warning threshold (GB used)")
            with gr.Group():
                gr.Markdown("**Maintenance**", elem_classes="section-head")
                clear_btn = gr.Button("🧹 Clear temporary cache (cache/temp only)", size="sm")
                maint_info = gr.Markdown(elem_classes="result-info")

    save_btn = gr.Button("💾 Save settings", variant="primary")
    save_info = gr.Markdown(elem_classes="result-info")

    def ui_save(m, o, i, dtts, theme, au, aum, adv, warn):
        current = load_settings()
        current.update(
            models_dir=m.strip() or DEFAULT_SETTINGS["models_dir"],
            outputs_dir=o.strip() or DEFAULT_SETTINGS["outputs_dir"],
            indexes_dir=i.strip() or DEFAULT_SETTINGS["indexes_dir"],
            default_tts_model=dtts,
            force_light_theme=bool(theme),
            auto_unload=bool(au),
            auto_unload_minutes=int(aum),
            show_advanced=bool(adv),
            mem_warn_gb=float(warn),
        )
        try:
            for key in ("models_dir", "outputs_dir", "indexes_dir"):
                _resolve(current[key]).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise gr.Error(f"Cannot create folder: {exc}")
        save_settings(current)
        return "✅ Saved. Folder and theme changes apply after restarting the app."

    save_btn.click(
        ui_save,
        inputs=[in_models, in_outputs, in_indexes, in_default_tts, in_theme,
                in_auto_unload, in_unload_min, in_advanced, in_warn],
        outputs=[save_info],
    )

    def ui_clear_temp():
        freed = clear_temp_cache()
        return f"🧹 Temporary cache cleared — {freed * 1024:.0f} MB freed."

    clear_btn.click(ui_clear_temp, outputs=[maint_info])
    open_models_btn.click(lambda: open_in_finder(models_dir()))
    open_outputs_btn.click(lambda: open_in_finder(outputs_dir()))
    open_indexes_btn.click(lambda: open_in_finder(indexes_dir()))
