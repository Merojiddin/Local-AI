"""Reusable pill/card model selectors — one per AI category.

Every category (chat, coding, vision, embedding, reranker, speech-to-text,
TTS voice mode, OCR engine) renders through the same component: a section
title, a small rounded "Model" chip and a wrapping row of selectable cards
(radio circle · name · estimated RAM · status tags · install button), styled
to match the warm pill look of the existing TTS selector.

Clicking a card only SAVES the choice to settings.json — nothing is loaded
until the related feature is actually used. Loading then goes through
memory_manager's single HEAVY slot, which always releases the previous heavy
model first, so Qwen3 14B, Qwen3-VL 8B and Kiwi 8B can never be resident
together on a 16 GB Mac.

All selectors share one hidden event bridge (#model-evt): the card's JS
writes a JSON event, the Python handler saves the selection (or downloads
missing weights) and re-renders every selector.
"""

from __future__ import annotations

import html as html_lib
import json

import gradio as gr

from . import model_manager as mgr
from . import storage

# --------------------------------------------------------------------------- #
# Category definitions
#
# Option fields: value  — what is stored in settings.json (usually a registry
#                         key from models.json)
#                label  — speed/quality wording shown before the model name
#                name   — model display name
#                badge  — optional "rec" (Recommended) / "mem" (High memory)
# --------------------------------------------------------------------------- #
CATEGORIES: dict[str, dict] = {
    "chat": {
        "settings_key": "chat_model",
        "title": "🤖 Chat model",
        "hint": "Used for chat, reasoning, dictionary questions, document answers and the Generator.",
        "default": "chat-8b",
        "options": [
            {"value": "chat-4b", "label": "Fast", "name": "Qwen3 4B"},
            {"value": "chat-8b", "label": "Balanced", "name": "Qwen3 8B", "badge": "rec"},
            {"value": "chat-14b", "label": "Higher quality", "name": "Qwen3 14B Q4", "badge": "mem"},
        ],
    },
    "coding": {
        "settings_key": "coding_model",
        "title": "👨‍💻 Coding model",
        "hint": "Used automatically when files or a project folder are attached to the chat.",
        "default": "kiwi-8b",
        "options": [
            {"value": "chat-4b", "label": "Fast", "name": "Qwen3 4B"},
            {"value": "kiwi-8b", "label": "Balanced", "name": "Kiwi 8B Q4", "badge": "rec"},
            {"value": "chat-14b", "label": "Higher quality", "name": "Qwen3 14B Q4", "badge": "mem"},
        ],
    },
    "vision": {
        "settings_key": "vision_model",
        "title": "🖼 Vision model",
        "hint": None,
        "default": "vision-4b",
        "options": [
            {"value": "vision-4b", "label": "Fast", "name": "Qwen3-VL 4B"},
            {"value": "vision-8b", "label": "Higher quality", "name": "Qwen3-VL 8B Q4", "badge": "rec"},
        ],
    },
    "embedding": {
        "settings_key": "embed_model",
        "title": "🧭 Embedding model",
        "hint": "Indexes built with one embedding model must be searched with the same model.",
        "default": "embed-0.6b",
        "options": [
            {"value": "embed-0.6b", "label": "Fast", "name": "Qwen3-Embedding 0.6B"},
            {"value": "embed-4b", "label": "Higher quality", "name": "Qwen3-Embedding 4B", "badge": "rec"},
        ],
    },
    "reranker": {
        "settings_key": "rerank_model",
        "title": "🎯 Reranker model",
        "hint": None,
        "default": "rerank-0.6b",
        "options": [
            {"value": "rerank-0.6b", "label": "Fast", "name": "Qwen3-Reranker 0.6B"},
            {"value": "rerank-4b", "label": "Higher quality", "name": "Qwen3-Reranker 4B", "badge": "rec"},
        ],
    },
    "stt": {
        "settings_key": "stt_model",
        "title": "🎙 Speech-to-text model",
        "hint": None,
        "default": "whisper",
        "options": [
            {"value": "whisper", "label": "Fast", "name": "Whisper Large-v3-Turbo"},
            {"value": "whisper-v3", "label": "Higher accuracy", "name": "Whisper Large-v3"},
        ],
    },
    "tts_voice_mode": {
        "settings_key": "tts_voice_mode",
        "title": "🗣 Voice mode",
        "chip": "Voice mode",
        "hint": "Built-in voices use the CustomVoice models; voice cloning uses the Base models.",
        "default": "CustomVoice",
        "options": [
            {"value": "CustomVoice", "label": "Built-in voices", "name": "CustomVoice"},
            {"value": "Base", "label": "Clone a voice", "name": "Base"},
        ],
    },
    "ocr": {
        "settings_key": "ocr_engine",
        "title": "🔎 OCR engine",
        "chip": "Engine",
        "hint": "AI-assisted OCR reads the page with the selected vision model — slower but understands messy layouts.",
        "default": "apple",
        "options": [
            {"value": "apple", "label": "Fast", "name": "Apple Vision OCR"},
            {"value": "qwen-vl", "label": "AI-assisted", "name": "Qwen3-VL OCR"},
        ],
    },
}


# --------------------------------------------------------------------------- #
# Selection state (validated read + save)
# --------------------------------------------------------------------------- #
def selected_key(cat: str) -> str:
    """The saved selection for this category, validated.

    Falls back to the category default when the saved value is unknown or its
    model no longer exists in the registry.
    """
    spec = CATEGORIES[cat]
    value = storage.load_settings().get(spec["settings_key"], spec["default"])
    if value not in {o["value"] for o in spec["options"]}:
        return spec["default"]
    if cat not in ("ocr", "tts_voice_mode") and value not in mgr.MODELS:
        return spec["default"]
    return value


def save_selection(cat: str, value: str) -> bool:
    spec = CATEGORIES[cat]
    if value not in {o["value"] for o in spec["options"]}:
        return False
    settings = storage.load_settings()
    settings[spec["settings_key"]] = value
    storage.save_settings(settings)
    return True


def on_model_removed(key: str) -> None:
    """Reset any category whose saved selection points at removed weights."""
    settings = storage.load_settings()
    changed = False
    for spec in CATEGORIES.values():
        if settings.get(spec["settings_key"]) == key:
            settings[spec["settings_key"]] = spec["default"]
            changed = True
    if changed:
        storage.save_settings(settings)


def _tts_size_suffix() -> str:
    """'0.6b' or '1.7b' depending on the chosen default TTS model."""
    label = str(storage.load_settings().get("default_tts_model", ""))
    return "0.6b" if "0.6B" in label else "1.7b"


def option_model_key(cat: str, opt_value: str) -> str | None:
    """Registry key backing one option (None = nothing to install)."""
    if cat == "ocr":
        if opt_value == "apple":
            return "ocr"
        from_settings = selected_key("vision")
        return from_settings if from_settings in mgr.MODELS else None
    if cat == "tts_voice_mode":
        suffix = _tts_size_suffix()
        return f"tts-{suffix}-base" if opt_value == "Base" else f"tts-{suffix}"
    return opt_value if opt_value in mgr.MODELS else None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _esc(text: str) -> str:
    return html_lib.escape(str(text), quote=True)


def render(cat: str) -> str:
    spec = CATEGORIES[cat]
    current = selected_key(cat)
    cards = []
    for opt in spec["options"]:
        key = option_model_key(cat, opt["value"])
        info = mgr.MODELS.get(key) if key else None
        builtin = bool(info and info["kind"] == "ocr")
        installed = mgr.is_installed(key) if key else False
        selected = opt["value"] == current

        tags = []
        if opt.get("badge") == "rec":
            tags.append('<span class="ms-tag rec">Recommended</span>')
        if opt.get("badge") == "mem":
            tags.append('<span class="ms-tag mem">High memory</span>')
        if builtin:
            tags.append('<span class="ms-tag ok">Built-in</span>')
        elif installed:
            tags.append('<span class="ms-tag ok">Installed</span>')
        elif info:
            tags.append('<span class="ms-tag miss">Not installed</span>')

        ram = ""
        if info and info.get("ram_gb") and not builtin:
            ram = f'<span class="ms-ram">≈{info["ram_gb"]:.1f} GB RAM</span>'

        button = ""
        if info and not builtin and not installed:
            need = info.get("download_gb", info["size_gb"])
            button = (
                f'<button class="ms-dl" data-model="{_esc(key)}" '
                f'onclick="changModelInstall(event, this)" '
                f'title="Download {_esc(info["name"])} (≈{need:.1f} GB)">⬇ Get</button>'
            )

        cards.append(
            f'<div class="ms-card{" selected" if selected else ""}" role="radio" '
            f'aria-checked="{"true" if selected else "false"}" tabindex="0" '
            f'data-cat="{_esc(cat)}" data-value="{_esc(opt["value"])}" '
            f'onclick="changModelPick(this)" '
            f'onkeydown="if(event.key===\'Enter\'||event.key===\' \')changModelPick(this)">'
            f'<span class="ms-radio"></span>'
            f'<span class="ms-text"><span class="ms-name">{_esc(opt["label"])} — '
            f'{_esc(opt["name"])}</span>{ram}</span>'
            f'{"".join(tags)}{button}</div>'
        )

    hint = f'<div class="ms-hint">{_esc(spec["hint"])}</div>' if spec.get("hint") else ""
    return (
        f'<div class="ms-section"><div class="ms-title">{spec["title"]}</div>'
        f'<span class="ms-chip">{_esc(spec.get("chip", "Model"))}</span>'
        f'<div class="ms-row" role="radiogroup">{"".join(cards)}</div>{hint}</div>'
    )


# --------------------------------------------------------------------------- #
# Gradio wiring — selectors register here; app.py attaches the shared bridge
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, gr.HTML] = {}
_LAST_HTML: dict[str, str] = {}


def build_selector(cat: str) -> gr.HTML:
    comp = gr.HTML(render(cat), elem_classes="ms-holder")
    _REGISTRY[cat] = comp
    return comp


def components() -> list[gr.HTML]:
    return list(_REGISTRY.values())


def refresh_updates(force: bool = False) -> list:
    """One gr.update per registered selector; no-ops when nothing changed."""
    out = []
    for cat in _REGISTRY:
        html = render(cat)
        if force or _LAST_HTML.get(cat) != html:
            _LAST_HTML[cat] = html
            out.append(gr.update(value=html))
        else:
            out.append(gr.update())
    return out


def attach_bridge() -> None:
    """Create the hidden event textbox + dispatcher. Call once, after all tabs."""
    if not _REGISTRY:
        return
    evt = gr.Textbox(visible=False, elem_id="model-evt")

    def on_event(payload: str):
        from . import memory_manager as mm

        try:
            data = json.loads(payload or "{}")
        except ValueError:
            data = {}
        action = data.get("action")

        if action == "select":
            cat = data.get("cat")
            if cat in CATEGORIES and save_selection(cat, data.get("value", "")):
                spec = CATEGORIES[cat]
                opt = next(o for o in spec["options"] if o["value"] == selected_key(cat))
                gr.Info(f"Saved: {opt['name']} — it loads the next time you use this feature.")

        elif action == "install":
            key = data.get("model")
            if key in mgr.MODELS and not mgr.is_installed(key):
                info = mgr.MODELS[key]
                need = info.get("download_gb", info["size_gb"])
                gr.Info(f"Downloading {info['name']} (≈{need:.1f} GB)… watch the status bar.")
                try:
                    mgr.install(key, progress_cb=lambda f, t: mm.set_task(f"⬇ {t}"))
                except Exception as exc:  # noqa: BLE001
                    raise gr.Error(str(exc))
                finally:
                    mm.set_task("idle")
                gr.Info(f"{info['name']} installed.")

        return refresh_updates(force=True)

    evt.input(on_event, inputs=[evt], outputs=components(), show_progress="hidden")
