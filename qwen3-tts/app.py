#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Chinese Text-to-Speech (Qwen3-TTS + MLX-Audio)
====================================================

Runs 100% locally on Apple Silicon. No cloud, no login, no database.

Two models are supported:
  * Fast          -> mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit
  * Higher Quality-> mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit

Usage:
  python app.py            # start the web app on http://127.0.0.1:7860
  python app.py --download # download both models, then exit
"""

from __future__ import annotations

import base64
import csv
import filecmp
import gc
import hashlib
import inspect
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"  # models live inside the project, not ~/.cache
ASSETS_DIR = BASE_DIR / "assets"  # put your logo here as assets/logo.png
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

SAMPLE_RATE = 24000  # Qwen3-TTS native output rate (mono, pcm_s16le)
LANG = "zh"


# --------------------------------------------------------------------------- #
# Dependency imports with beginner-friendly errors
# --------------------------------------------------------------------------- #
def _fatal(message: str) -> "NoReturn":  # type: ignore[name-defined]
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
    import mlx.core as mx
    from mlx_audio.tts.generate import generate_audio
    from mlx_audio.tts.utils import load_model
except ImportError:
    _fatal(
        "MLX-Audio is not installed (or not for Apple Silicon).\n"
        "Fix: run install.command again, or inside the project folder run:\n"
        "  ./.venv/bin/pip install -r requirements.txt\n"
        "MLX-Audio only works on Apple Silicon Macs (M1/M2/M3/M4)."
    )

# Detect which optional features the installed MLX-Audio actually supports,
# instead of assuming argument names.
_GEN_PARAMS = inspect.signature(generate_audio).parameters
SUPPORTS_INSTRUCT = "instruct" in _GEN_PARAMS
SUPPORTS_SPEED = "speed" in _GEN_PARAMS
SUPPORTS_LANG = "lang_code" in _GEN_PARAMS


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODELS = {
    "Fast — Qwen3-TTS 0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
    "Higher Quality — Qwen3-TTS 1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
}
DEFAULT_MODEL = "Fast — Qwen3-TTS 0.6B"

MODEL_SHORT = {
    MODELS["Fast — Qwen3-TTS 0.6B"]: "0.6B",
    MODELS["Higher Quality — Qwen3-TTS 1.7B"]: "1.7B",
}

# Where each model is stored locally inside the project folder.
MODEL_LOCAL_DIRS = {
    MODELS["Fast — Qwen3-TTS 0.6B"]: MODELS_DIR / "Qwen3-TTS-0.6B",
    MODELS["Higher Quality — Qwen3-TTS 1.7B"]: MODELS_DIR / "Qwen3-TTS-1.7B",
}


def resolve_model_source(repo: str) -> str:
    """Return the local project folder if the model is present there,
    otherwise fall back to the Hugging Face repo id (which downloads)."""
    local = MODEL_LOCAL_DIRS.get(repo)
    if local and (local / "config.json").exists():
        return str(local)
    return repo

VOICES = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"]
DEFAULT_VOICE = "Vivian"

MODES = ["Single Word", "Vocabulary Phrase", "Sentence", "Paragraph"]
DEFAULT_MODE = "Sentence"

MODE_SHORT = {
    "Single Word": "word",
    "Vocabulary Phrase": "phrase",
    "Sentence": "sent",
    "Paragraph": "para",
}

# Per-mode pronunciation behaviour.
#   speed_factor : multiplies the user speed (Single Word = slower & clearer)
#   min_pad      : minimum silence before/after so short items are not abrupt
#   instruct     : default style hint used only when the user leaves the
#                  style box empty (and only if the model supports instruct)
MODE_CFG = {
    "Single Word": {
        "speed_factor": 0.82,
        "min_pad": 0.20,
        "instruct": "请放慢速度，逐字非常清晰地发音，不要急促。",
    },
    "Vocabulary Phrase": {
        "speed_factor": 0.90,
        "min_pad": 0.10,
        "instruct": "请清晰地朗读这个词语，字与字之间稍作停顿。",
    },
    "Sentence": {
        "speed_factor": 1.0,
        "min_pad": 0.0,
        "instruct": "请用自然的标准普通话语调朗读。",
    },
    "Paragraph": {
        "speed_factor": 1.0,
        "min_pad": 0.0,
        "instruct": "请用自然的语速朗读，句子之间自然停顿。",
    },
}

STYLE_EXAMPLES = [
    "像中文老师一样，慢速清晰地朗读 (Speak slowly and clearly like a Chinese teacher)",
    "使用中性的标准普通话 (Use neutral Standard Mandarin)",
    "自然地朗读 (Speak naturally)",
    "清晰地强调每一个音节 (Emphasize every syllable clearly)",
    "用温暖的女性教学嗓音 (Use a warm female teaching voice)",
]

# User-saved style-instruction presets, persisted to disk so they survive
# restarts. Stored as a simple {name: instruction} JSON file in the project.
PRESETS_FILE = BASE_DIR / "style_presets.json"


def load_presets() -> dict:
    """Return saved style-instruction presets as an ordered {name: text} dict."""
    try:
        with open(PRESETS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (OSError, ValueError):
        pass
    return {}


def write_presets(presets: dict) -> None:
    """Atomically write presets to disk (temp file + replace)."""
    tmp = PRESETS_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(presets, f, ensure_ascii=False, indent=2)
    tmp.replace(PRESETS_FILE)


FORMATS = ["MP3", "WAV"]
DEFAULT_FORMAT = "MP3"

QUALITY_KBPS = {"128 kbps": 128, "192 kbps": 192, "256 kbps": 256}
QUALITIES = list(QUALITY_KBPS.keys())
DEFAULT_QUALITY = "192 kbps"


# --------------------------------------------------------------------------- #
# Branding — "Chang 日日向上" sunrise identity (red -> orange -> gold)
# --------------------------------------------------------------------------- #
# The real logo is loaded from assets/logo.(png|jpg|jpeg|webp|svg) if present.
# Until that file exists, a built-in SVG in the same spirit is shown instead.
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


# --------------------------------------------------------------------------- #
# Internationalisation — Tiếng Việt / English / 中文
# --------------------------------------------------------------------------- #
DEFAULT_LANG = "en"
LANG_CHOICES = [("Tiếng Việt", "vi"), ("English", "en"), ("中文", "zh")]

T = {
    "en": {
        "tagline": "Local Chinese text-to-speech — runs 100% on your Mac, no cloud, no login.",
        "tab_single": "Single",
        "tab_batch": "Batch",
        "sec_text": "✍️ **Text**",
        "text_label": "Chinese text",
        "text_ph": "在这里输入中文…",
        "sec_voice": "🎤 **Voice & Model**",
        "model": "Model",
        "model_fast": "Fast — 0.6B",
        "model_hq": "Higher quality — 1.7B",
        "voice": "Voice",
        "sec_pron": "🔊 **Pronunciation**",
        "mode": "Reading mode",
        "mode_word": "Single word",
        "mode_phrase": "Phrase",
        "mode_sent": "Sentence",
        "mode_para": "Paragraph",
        "speed": "Speed (×)",
        "style": "Style instruction (optional)",
        "style_ph": "例如：像中文老师一样，慢速清晰地朗读",
        "acc_examples": "Style examples",
        "acc_presets": "Saved presets",
        "preset_dd": "Saved presets",
        "load": "Load",
        "delete": "Delete",
        "save": "Save",
        "preset_name": "Save current instruction as…",
        "preset_ph": "preset name",
        "p_before": "Pause before (s)",
        "p_after": "Pause after (s)",
        "p_between": "Pause between repeats (s)",
        "repeat": "Repeat count",
        "sec_out": "💾 **Output**",
        "format": "Format",
        "quality": "MP3 quality",
        "generate": "🎧 Generate audio",
        "audio": "Preview",
        "download": "⬇️ Download audio",
        "batch_desc": (
            "Paste **one Chinese word or sentence per line**, or upload a CSV with "
            "columns `text, voice, model, mode, speed` (only `text` is required). "
            "Rows use the defaults on the right."
        ),
        "batch_text": "One item per line",
        "batch_csv": "Or upload CSV",
        "b_defaults": "⚙️ **Defaults for all rows**",
        "b_style": "Style instruction for all rows (optional)",
        "acc_adv": "Pauses & repeats",
        "batch_btn": "🚀 Generate batch",
        "table": "Results",
        "zip": "⬇️ Download ZIP (audio + results CSV)",
        "footer": "Files are saved in",
        "res_cached": "♻️ From cache",
        "res_new": "🆕 Newly generated",
        "res_file": "Saved file",
        "loading": "Loading model: {model} … The first run downloads it (several GB) and can take a few minutes.",
        "item_of": "Item {n} of {total}",
        "packaging": "Packaging results",
        "batch_done": "**Done.** {ok} succeeded, {failed} failed, {total} total.",
    },
    "vi": {
        "tagline": "Chuyển văn bản tiếng Trung thành giọng nói — chạy 100% trên máy của bạn, không cần mạng hay đăng nhập.",
        "tab_single": "Tạo đơn",
        "tab_batch": "Tạo hàng loạt",
        "sec_text": "✍️ **Văn bản**",
        "text_label": "Văn bản tiếng Trung",
        "text_ph": "在这里输入中文…",
        "sec_voice": "🎤 **Giọng & Mô hình**",
        "model": "Mô hình",
        "model_fast": "Nhanh — 0.6B",
        "model_hq": "Chất lượng cao — 1.7B",
        "voice": "Giọng đọc",
        "sec_pron": "🔊 **Phát âm**",
        "mode": "Chế độ đọc",
        "mode_word": "Từ đơn",
        "mode_phrase": "Cụm từ",
        "mode_sent": "Câu",
        "mode_para": "Đoạn văn",
        "speed": "Tốc độ (×)",
        "style": "Chỉ dẫn phong cách (tuỳ chọn)",
        "style_ph": "例如：像中文老师一样，慢速清晰地朗读",
        "acc_examples": "Ví dụ phong cách",
        "acc_presets": "Mẫu đã lưu",
        "preset_dd": "Mẫu đã lưu",
        "load": "Tải",
        "delete": "Xoá",
        "save": "Lưu",
        "preset_name": "Lưu chỉ dẫn hiện tại với tên…",
        "preset_ph": "tên mẫu",
        "p_before": "Nghỉ trước (giây)",
        "p_after": "Nghỉ sau (giây)",
        "p_between": "Nghỉ giữa các lần lặp (giây)",
        "repeat": "Số lần lặp",
        "sec_out": "💾 **Đầu ra**",
        "format": "Định dạng",
        "quality": "Chất lượng MP3",
        "generate": "🎧 Tạo âm thanh",
        "audio": "Nghe thử",
        "download": "⬇️ Tải tệp âm thanh",
        "batch_desc": (
            "Dán **mỗi dòng một từ hoặc một câu tiếng Trung**, hoặc tải lên tệp CSV "
            "có các cột `text, voice, model, mode, speed` (chỉ bắt buộc cột `text`). "
            "Các dòng dùng cài đặt mặc định ở bên phải."
        ),
        "batch_text": "Mỗi dòng một mục",
        "batch_csv": "Hoặc tải lên CSV",
        "b_defaults": "⚙️ **Cài đặt mặc định cho mọi dòng**",
        "b_style": "Chỉ dẫn phong cách cho tất cả các dòng (tuỳ chọn)",
        "acc_adv": "Nghỉ & lặp lại",
        "batch_btn": "🚀 Tạo hàng loạt",
        "table": "Kết quả",
        "zip": "⬇️ Tải ZIP (âm thanh + CSV kết quả)",
        "footer": "Tệp được lưu tại",
        "res_cached": "♻️ Lấy từ bộ nhớ đệm",
        "res_new": "🆕 Vừa tạo mới",
        "res_file": "Tệp đã lưu",
        "loading": "Đang nạp mô hình: {model} … Lần chạy đầu sẽ tải mô hình về (vài GB) và có thể mất vài phút.",
        "item_of": "Mục {n} / {total}",
        "packaging": "Đang đóng gói kết quả",
        "batch_done": "**Hoàn tất.** {ok} thành công, {failed} thất bại, tổng cộng {total}.",
    },
    "zh": {
        "tagline": "本地中文语音合成 — 完全在你的电脑上运行，无需联网或登录。",
        "tab_single": "单条生成",
        "tab_batch": "批量生成",
        "sec_text": "✍️ **文本**",
        "text_label": "中文文本",
        "text_ph": "在这里输入中文…",
        "sec_voice": "🎤 **声音与模型**",
        "model": "模型",
        "model_fast": "快速 — 0.6B",
        "model_hq": "高质量 — 1.7B",
        "voice": "声音",
        "sec_pron": "🔊 **发音设置**",
        "mode": "朗读模式",
        "mode_word": "单词",
        "mode_phrase": "词组",
        "mode_sent": "句子",
        "mode_para": "段落",
        "speed": "语速 (×)",
        "style": "风格指令（可选）",
        "style_ph": "例如：像中文老师一样，慢速清晰地朗读",
        "acc_examples": "风格示例",
        "acc_presets": "已存预设",
        "preset_dd": "已存预设",
        "load": "载入",
        "delete": "删除",
        "save": "保存",
        "preset_name": "将当前指令另存为…",
        "preset_ph": "预设名称",
        "p_before": "前停顿（秒）",
        "p_after": "后停顿（秒）",
        "p_between": "重复间停顿（秒）",
        "repeat": "重复次数",
        "sec_out": "💾 **输出**",
        "format": "格式",
        "quality": "MP3 音质",
        "generate": "🎧 生成语音",
        "audio": "试听",
        "download": "⬇️ 下载音频",
        "batch_desc": (
            "每行粘贴**一个中文词语或句子**，或上传包含 "
            "`text, voice, model, mode, speed` 列的 CSV（仅 `text` 必填）。"
            "各行以右侧设置为默认值。"
        ),
        "batch_text": "每行一条",
        "batch_csv": "或上传 CSV",
        "b_defaults": "⚙️ **所有行的默认设置**",
        "b_style": "应用于所有行的风格指令（可选）",
        "acc_adv": "停顿与重复",
        "batch_btn": "🚀 批量生成",
        "table": "结果",
        "zip": "⬇️ 下载 ZIP（音频 + 结果 CSV）",
        "footer": "文件保存在",
        "res_cached": "♻️ 来自缓存",
        "res_new": "🆕 新生成",
        "res_file": "已保存文件",
        "loading": "正在加载模型：{model}…… 首次运行需要下载（数 GB），可能需要几分钟。",
        "item_of": "第 {n} / {total} 条",
        "packaging": "正在打包结果",
        "batch_done": "**完成。**成功 {ok} 条，失败 {failed} 条，共 {total} 条。",
    },
}


def tr(lang: str, key: str) -> str:
    table = T.get(lang) or T[DEFAULT_LANG]
    return table.get(key) or T[DEFAULT_LANG].get(key, key)


def model_choices(lang: str) -> list[tuple[str, str]]:
    """Translated display labels; values stay the internal English keys."""
    return [
        (tr(lang, "model_fast"), "Fast — Qwen3-TTS 0.6B"),
        (tr(lang, "model_hq"), "Higher Quality — Qwen3-TTS 1.7B"),
    ]


def mode_choices(lang: str) -> list[tuple[str, str]]:
    return [
        (tr(lang, "mode_word"), "Single Word"),
        (tr(lang, "mode_phrase"), "Vocabulary Phrase"),
        (tr(lang, "mode_sent"), "Sentence"),
        (tr(lang, "mode_para"), "Paragraph"),
    ]


def header_html(lang: str) -> str:
    return f"""
<div class="chang-header">
  <img class="chang-logo" src="{logo_data_uri()}" alt="Chang 日日向上"/>
  <div class="chang-title">
    <div class="chang-row"><span class="chang-name">Chang</span><span class="chang-cjk">日日向上</span></div>
    <div class="chang-tagline">{tr(lang, "tagline")}</div>
  </div>
</div>"""


# --------------------------------------------------------------------------- #
# UI — theme & CSS in the logo's sunrise palette, compact one-window layout
# --------------------------------------------------------------------------- #
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
.chang-logo { height: 56px; width: auto; }
.chang-row { display: flex; align-items: baseline; gap: 10px; }
.chang-name {
  font-size: 1.45rem; font-weight: 800; line-height: 1.1;
  background-image: linear-gradient(90deg, var(--chang-red) 0%, var(--chang-orange) 55%, var(--chang-gold) 100%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: var(--chang-orange);
}
.chang-cjk { font-size: 0.95rem; font-weight: 700; letter-spacing: 0.3em; color: var(--chang-gold); }
.chang-tagline { font-size: 0.78rem; color: #977a5c; margin-top: 1px; }
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


# --------------------------------------------------------------------------- #
# Model manager — keeps ONE model in memory at a time (16GB RAM friendly)
# --------------------------------------------------------------------------- #
class ModelManager:
    def __init__(self) -> None:
        self.repo: str | None = None
        self.model = None

    def needs_load(self, repo: str) -> bool:
        return not (self.repo == repo and self.model is not None)

    def release(self) -> None:
        """Release the current model cleanly before loading another one."""
        if self.model is not None:
            try:
                del self.model
            except Exception:
                pass
        self.model = None
        self.repo = None
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

    def get(self, repo: str):
        if self.repo == repo and self.model is not None:
            return self.model
        # Never hold two models at once.
        self.release()
        self.model = load_model(resolve_model_source(repo))
        self.repo = repo
        return self.model


MODEL_MANAGER = ModelManager()


# --------------------------------------------------------------------------- #
# FFmpeg helpers
# --------------------------------------------------------------------------- #
def _require_ffmpeg() -> None:
    if not FFMPEG:
        raise RuntimeError(
            "FFmpeg is not installed. It is required to build WAV/MP3 output.\n"
            "Fix: install Homebrew, then run:  brew install ffmpeg"
        )


def _run_ffmpeg(args: list[str]) -> None:
    _require_ffmpeg()
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg failed:\n" + (proc.stderr or "unknown error").strip())


def make_silence(duration: float, dst: Path) -> None:
    _run_ffmpeg(
        [
            "-f", "lavfi",
            "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
            "-t", f"{max(0.0, duration):.3f}",
            "-c:a", "pcm_s16le",
            str(dst),
        ]
    )


def normalize_wav(src: Path, dst: Path) -> None:
    """Re-encode to a canonical mono/24000/pcm_s16le WAV so pieces concat cleanly."""
    _run_ffmpeg(
        ["-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)]
    )


def concat_wavs(parts: list[Path], dst: Path) -> None:
    """Concatenate identical-format WAV files with the ffmpeg concat demuxer."""
    if len(parts) == 1:
        shutil.copy2(parts[0], dst)
        return
    listfile = dst.with_suffix(".txt")
    with open(listfile, "w", encoding="utf-8") as f:
        for p in parts:
            safe = str(p).replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
    try:
        _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(dst)])
    finally:
        listfile.unlink(missing_ok=True)


def to_mp3(src: Path, dst: Path, kbps: int) -> None:
    _run_ffmpeg(["-i", str(src), "-c:a", "libmp3lame", "-b:a", f"{kbps}k", str(dst)])


# --------------------------------------------------------------------------- #
# Filenames & cache
# --------------------------------------------------------------------------- #
def safe_stub(text: str) -> str:
    """Build a safe filename stub from Chinese text (keeps CJK & word chars)."""
    text = (text or "").strip()
    cleaned = re.sub(r"[^\w一-鿿]+", "_", text, flags=re.UNICODE).strip("_")
    if not cleaned:
        cleaned = "audio"
    return cleaned[:40]


def unique_output_path(text: str, ext: str, cache_file: Path) -> Path:
    """Pick the friendly output filename, named purely after the text
    (e.g. 你好.mp3). If that name is already taken by identical audio we
    reuse it; if it is taken by *different* audio we add a numeric suffix
    (你好_2.mp3) so nothing is silently overwritten."""
    stub = safe_stub(text)
    candidate = OUTPUTS_DIR / f"{stub}.{ext}"
    i = 2
    while candidate.exists():
        if (
            candidate.stat().st_size == cache_file.stat().st_size
            and filecmp.cmp(candidate, cache_file, shallow=False)
        ):
            return candidate  # same audio already saved under this name
        candidate = OUTPUTS_DIR / f"{stub}_{i}.{ext}"
        i += 1
    return candidate


def cache_hash(params: dict) -> str:
    blob = json.dumps(params, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Core generation (one item) — with deterministic cache
# --------------------------------------------------------------------------- #
def generate_one(
    text: str,
    model_label: str,
    voice: str,
    mode: str,
    speed: float,
    style: str,
    p_before: float,
    p_after: float,
    p_between: float,
    repeat: int,
    out_format: str,
    quality: str,
) -> dict:
    _require_ffmpeg()

    text = (text or "").strip()
    if not text:
        raise ValueError("Please enter some Chinese text.")
    if model_label not in MODELS:
        raise ValueError(f"Unknown model: {model_label}")
    if voice not in VOICES:
        raise ValueError(f"Unknown voice: {voice}. Allowed: {', '.join(VOICES)}")
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}. Allowed: {', '.join(MODES)}")

    repo = MODELS[model_label]
    ext = "mp3" if str(out_format).upper() == "MP3" else "wav"
    kbps = QUALITY_KBPS.get(quality, 192)
    repeat = max(1, min(5, int(repeat)))
    speed = float(speed)
    style = (style or "").strip()
    p_before = max(0.0, min(2.0, float(p_before)))
    p_after = max(0.0, min(2.0, float(p_after)))
    p_between = max(0.0, min(2.0, float(p_between)))

    # Deterministic cache key over the RAW user inputs (item 14).
    params = {
        "text": text,
        "repo": repo,
        "voice": voice,
        "mode": mode,
        "speed": round(speed, 3),
        "style": style,
        "p_before": round(p_before, 3),
        "p_after": round(p_after, 3),
        "p_between": round(p_between, 3),
        "repeat": repeat,
        "format": ext,
        "quality": kbps,
    }
    digest = cache_hash(params)
    cache_file = CACHE_DIR / f"{digest}.{ext}"

    # Cache hit: reuse without regenerating.
    if cache_file.exists():
        friendly = unique_output_path(text, ext, cache_file)
        if not friendly.exists():
            shutil.copy2(cache_file, friendly)
        return {
            "path": friendly,
            "cached": True,
            "gen_time": 0.0,
            "repo": repo,
            "voice": voice,
            "digest": digest,
        }

    t0 = time.time()

    # Effective (mode-adjusted) settings.
    cfg = MODE_CFG[mode]
    eff_speed = max(0.5, min(1.5, speed * cfg["speed_factor"]))
    eff_instruct = style or cfg["instruct"]
    eff_before = max(p_before, cfg["min_pad"])
    eff_after = max(p_after, cfg["min_pad"])

    model = MODEL_MANAGER.get(repo)  # loads & keeps in RAM

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        prefix = "seg"
        kwargs = dict(
            text=text,
            model=model,
            voice=voice,
            output_path=str(tdp),
            file_prefix=prefix,
            audio_format="wav",
            join_audio=True,
            verbose=False,
        )
        if SUPPORTS_SPEED:
            kwargs["speed"] = eff_speed
        if SUPPORTS_LANG:
            kwargs["lang_code"] = LANG
        if SUPPORTS_INSTRUCT and eff_instruct:
            kwargs["instruct"] = eff_instruct

        try:
            generate_audio(**kwargs)
        except TypeError:
            # An installed version rejected an optional kwarg: retry minimally.
            for opt in ("instruct", "speed", "lang_code"):
                kwargs.pop(opt, None)
            generate_audio(**kwargs)

        produced = sorted(tdp.glob(f"{prefix}*.wav"))
        if not produced:
            raise RuntimeError(
                "The model did not produce any audio. Try shorter text, another "
                "voice, or the other model."
            )

        # Normalise model output to a canonical WAV.
        base = tdp / "base.wav"
        if len(produced) == 1:
            normalize_wav(produced[0], base)
        else:
            norm_parts = []
            for i, p in enumerate(produced):
                np_ = tdp / f"n{i}.wav"
                normalize_wav(p, np_)
                norm_parts.append(np_)
            concat_wavs(norm_parts, base)

        # Assemble: [before] + base (x repeat, joined by [between]) + [after].
        parts: list[Path] = []
        if eff_before > 0:
            s = tdp / "sb.wav"
            make_silence(eff_before, s)
            parts.append(s)
        for i in range(repeat):
            parts.append(base)
            if i < repeat - 1 and p_between > 0:
                s = tdp / f"sm{i}.wav"
                make_silence(p_between, s)
                parts.append(s)
        if eff_after > 0:
            s = tdp / "sa.wav"
            make_silence(eff_after, s)
            parts.append(s)

        assembled = tdp / "assembled.wav"
        concat_wavs(parts, assembled)

        # Write canonical artifact into the cache in the requested format.
        if ext == "wav":
            shutil.copy2(assembled, cache_file)
        else:
            to_mp3(assembled, cache_file, kbps)

    friendly = unique_output_path(text, ext, cache_file)
    shutil.copy2(cache_file, friendly)
    return {
        "path": friendly,
        "cached": False,
        "gen_time": time.time() - t0,
        "repo": repo,
        "voice": voice,
        "digest": digest,
    }


# --------------------------------------------------------------------------- #
# Error mapping -> friendly gr.Error messages
# --------------------------------------------------------------------------- #
def to_friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if isinstance(exc, MemoryError) or "out of memory" in low or "metal" in low:
        return (
            "Insufficient memory. Close other apps and try the Fast (0.6B) model, "
            "or use shorter text.\n\nDetails: " + msg
        )
    if "ffmpeg" in low:
        return "FFmpeg problem. Install it with:  brew install ffmpeg\n\nDetails: " + msg
    if any(k in low for k in ("connection", "download", "http", "resolve host", "timed out", "offline")):
        return (
            "Model download / network problem. Check your internet connection and try "
            "again (the first run downloads several GB).\n\nDetails: " + msg
        )
    if "no module named" in low or isinstance(exc, ImportError):
        return (
            "A Python dependency is missing. Run install.command again, or:\n"
            "  ./.venv/bin/pip install -r requirements.txt\n\nDetails: " + msg
        )
    if isinstance(exc, ValueError):
        return msg
    return "Audio generation failed.\n\nDetails: " + msg


# --------------------------------------------------------------------------- #
# Single generation UI callback
# --------------------------------------------------------------------------- #
def ui_generate(
    text, model_label, voice, mode, speed, style,
    p_before, p_after, p_between, repeat, out_format, quality,
    lang=DEFAULT_LANG,
):
    try:
        repo = MODELS.get(model_label)
        if repo and MODEL_MANAGER.needs_load(repo):
            gr.Info(tr(lang, "loading").format(model=model_label))
        res = generate_one(
            text, model_label, voice, mode, speed, style,
            p_before, p_after, p_between, repeat, out_format, quality,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        raise gr.Error(to_friendly_error(exc))

    path = str(res["path"])
    status = tr(lang, "res_cached") if res["cached"] else tr(lang, "res_new")
    gen_time = "0.0s" if res["cached"] else f"{res['gen_time']:.1f}s"
    info = (
        f"{status} · **{MODEL_SHORT[res['repo']]}** · {res['voice']} · {gen_time}\n\n"
        f"**{tr(lang, 'res_file')}:** `{path}`"
    )
    return path, path, info


# --------------------------------------------------------------------------- #
# Batch helpers
# --------------------------------------------------------------------------- #
def _resolve_voice(value: str, default: str, lineno: int) -> str:
    if not value:
        return default
    for v in VOICES:
        if v.lower() == value.strip().lower():
            return v
    raise ValueError(
        f"CSV line {lineno}: unknown voice '{value}'. Allowed: {', '.join(VOICES)}"
    )


def _resolve_model(value: str, default: str, lineno: int) -> str:
    if not value:
        return default
    v = value.strip().lower()
    fast_keys = {"fast", "0.6b", "0.6", "small", MODELS["Fast — Qwen3-TTS 0.6B"].lower()}
    hq_keys = {"quality", "higher quality", "hq", "1.7b", "1.7", "large",
               MODELS["Higher Quality — Qwen3-TTS 1.7B"].lower()}
    if v in fast_keys or v == "fast — qwen3-tts 0.6b":
        return "Fast — Qwen3-TTS 0.6B"
    if v in hq_keys or v == "higher quality — qwen3-tts 1.7b":
        return "Higher Quality — Qwen3-TTS 1.7B"
    raise ValueError(
        f"CSV line {lineno}: unknown model '{value}'. Use '0.6B' (fast) or '1.7B' (quality)."
    )


def _resolve_mode(value: str, default: str, lineno: int) -> str:
    if not value:
        return default
    v = value.strip().lower()
    table = {
        "single word": "Single Word", "word": "Single Word",
        "vocabulary phrase": "Vocabulary Phrase", "phrase": "Vocabulary Phrase",
        "sentence": "Sentence", "paragraph": "Paragraph",
    }
    if v in table:
        return table[v]
    raise ValueError(
        f"CSV line {lineno}: unknown mode '{value}'. Allowed: {', '.join(MODES)}"
    )


def _resolve_speed(value: str, default: float, lineno: int) -> float:
    if not value:
        return default
    try:
        return max(0.5, min(1.5, float(value)))
    except ValueError:
        raise ValueError(f"CSV line {lineno}: speed '{value}' is not a number.")


def parse_batch(text_lines: str, csv_path: str | None, defaults: dict) -> list[dict]:
    rows: list[dict] = []
    if csv_path:
        try:
            f = open(csv_path, newline="", encoding="utf-8-sig")
        except OSError as exc:
            raise ValueError(f"Could not open CSV file: {exc}")
        with f:
            reader = csv.DictReader(f)
            headers = [(h or "").strip().lower() for h in (reader.fieldnames or [])]
            if "text" not in headers:
                raise ValueError(
                    "CSV must have a 'text' column. Allowed columns: "
                    "text, voice, model, mode, speed."
                )
            for lineno, raw in enumerate(reader, start=2):
                low = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
                text = low.get("text", "")
                if not text:
                    continue
                rows.append(
                    dict(
                        text=text,
                        voice=_resolve_voice(low.get("voice", ""), defaults["voice"], lineno),
                        model_label=_resolve_model(low.get("model", ""), defaults["model_label"], lineno),
                        mode=_resolve_mode(low.get("mode", ""), defaults["mode"], lineno),
                        speed=_resolve_speed(low.get("speed", ""), defaults["speed"], lineno),
                    )
                )
    else:
        for line in (text_lines or "").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(
                dict(
                    text=line,
                    voice=defaults["voice"],
                    model_label=defaults["model_label"],
                    mode=defaults["mode"],
                    speed=defaults["speed"],
                )
            )

    if not rows:
        raise ValueError(
            "No input rows found. Paste one item per line, or upload a CSV with a 'text' column."
        )
    return rows


def ui_batch(
    text_lines, csv_file, voice, model_label, mode, speed, style,
    p_before, p_after, p_between, repeat, out_format, quality,
    lang=DEFAULT_LANG,
    progress=gr.Progress(),
):
    try:
        defaults = dict(voice=voice, model_label=model_label, mode=mode, speed=float(speed))
        rows = parse_batch(text_lines, csv_file, defaults)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(to_friendly_error(exc))

    total = len(rows)
    # Group by model to minimise reloads, but report results in original order.
    order = sorted(range(total), key=lambda i: rows[i]["model_label"])
    results: dict[int, dict] = {}
    produced_paths: list[Path] = []

    for count, i in enumerate(order):
        r = rows[i]
        progress(count / total, desc=tr(lang, "item_of").format(n=count + 1, total=total))
        try:
            res = generate_one(
                r["text"], r["model_label"], r["voice"], r["mode"], r["speed"],
                style, p_before, p_after, p_between, repeat, out_format, quality,
            )
            results[i] = {
                "text": r["text"],
                "filename": res["path"].name,
                "path": str(res["path"]),
                "status": "cached" if res["cached"] else "generated",
                "error": "",
            }
            produced_paths.append(res["path"])
        except Exception as exc:  # noqa: BLE001 - continue on failure
            results[i] = {
                "text": r["text"],
                "filename": "",
                "path": "",
                "status": "failed",
                "error": str(exc),
            }

    progress(1.0, desc=tr(lang, "packaging"))
    ordered = [results[i] for i in range(total)]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = OUTPUTS_DIR / f"batch_results_{ts}.csv"
    with open(results_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "filename", "path", "status", "error"])
        writer.writeheader()
        writer.writerows(ordered)

    zip_path = OUTPUTS_DIR / f"batch_{ts}.zip"
    seen: set[str] = set()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in produced_paths:
            if p.exists() and str(p) not in seen:
                zf.write(p, arcname=p.name)
                seen.add(str(p))
        zf.write(results_csv, arcname=results_csv.name)

    ok = sum(1 for r in ordered if r["status"] in ("generated", "cached"))
    failed = sum(1 for r in ordered if r["status"] == "failed")
    summary = (
        tr(lang, "batch_done").format(ok=ok, failed=failed, total=total)
        + f"\n\n**ZIP:** `{zip_path}`\n\n**CSV:** `{results_csv}`"
    )
    table = [[r["text"], r["filename"], r["status"], r["error"]] for r in ordered]
    return table, str(zip_path), summary


# --------------------------------------------------------------------------- #
# Style-preset UI callbacks
# --------------------------------------------------------------------------- #
def ui_save_preset(name: str, style_text: str):
    name = (name or "").strip()
    if not name:
        raise gr.Error("Please enter a name for this instruction preset.")
    style_text = (style_text or "").strip()
    if not style_text:
        raise gr.Error("The style instruction box is empty — nothing to save.")
    presets = load_presets()
    existed = name in presets
    presets[name] = style_text
    write_presets(presets)
    gr.Info(f"{'Updated' if existed else 'Saved'} instruction preset: {name}")
    # Refresh the dropdown, select the just-saved name, clear the name box.
    return gr.update(choices=sorted(presets), value=name), ""


def ui_load_preset(name: str):
    if not name:
        raise gr.Error("Pick a saved preset to load.")
    text = load_presets().get(name)
    if text is None:
        raise gr.Error(f"Preset '{name}' was not found (it may have been deleted).")
    gr.Info(f"Loaded preset: {name}")
    return text


def ui_delete_preset(name: str):
    if not name:
        raise gr.Error("Pick a saved preset to delete.")
    presets = load_presets()
    if name in presets:
        del presets[name]
        write_presets(presets)
        gr.Info(f"Deleted preset: {name}")
    return gr.update(choices=sorted(presets), value=None)


def build_ui() -> "gr.Blocks":
    ffmpeg_ok = bool(FFMPEG)
    L0 = DEFAULT_LANG

    with gr.Blocks(
        title="Chang 日日向上 · Chinese TTS",
        theme=THEME,
        css=CHANG_CSS,
        js=FORCE_LIGHT_JS,
    ) as demo:
        lang_store = gr.BrowserState(DEFAULT_LANG, storage_key="chang_tts_lang")

        # ---- Header: logo + brand + language toggle ----
        with gr.Row(elem_id="chang-topbar"):
            with gr.Column(scale=1, min_width=380):
                header = gr.HTML(header_html(L0))
            with gr.Column(scale=0, min_width=320, elem_id="lang-col"):
                lang = gr.Radio(
                    choices=LANG_CHOICES, value=L0, show_label=False,
                    container=False, elem_id="lang-pick",
                )

        if not ffmpeg_ok:
            gr.Markdown(
                "> ⚠️ **FFmpeg was not found.** Audio output needs it. "
                "Install with `brew install ffmpeg`, then restart the app."
            )
        if not SUPPORTS_INSTRUCT:
            gr.Markdown(
                "> ℹ️ The installed MLX-Audio does not support style instructions; "
                "the style box is disabled."
            )

        with gr.Tab(tr(L0, "tab_single")) as tab_single:
            with gr.Row(equal_height=False):
                # ---- Left column: text, voice/model, output, generate ----
                with gr.Column(scale=5):
                    with gr.Group():
                        sec_text = gr.Markdown(tr(L0, "sec_text"), elem_classes="section-head")
                        text = gr.Textbox(
                            label=tr(L0, "text_label"),
                            placeholder=tr(L0, "text_ph"),
                            lines=3,
                            autofocus=True,
                        )
                    with gr.Group():
                        sec_voice = gr.Markdown(tr(L0, "sec_voice"), elem_classes="section-head")
                        model_label = gr.Radio(
                            choices=model_choices(L0), value=DEFAULT_MODEL, label=tr(L0, "model")
                        )
                        voice = gr.Dropdown(choices=VOICES, value=DEFAULT_VOICE, label=tr(L0, "voice"))
                    with gr.Group():
                        sec_out = gr.Markdown(tr(L0, "sec_out"), elem_classes="section-head")
                        with gr.Row():
                            out_format = gr.Radio(choices=FORMATS, value=DEFAULT_FORMAT, label=tr(L0, "format"))
                            quality = gr.Dropdown(choices=QUALITIES, value=DEFAULT_QUALITY, label=tr(L0, "quality"))
                    generate_btn = gr.Button(tr(L0, "generate"), variant="primary")
                    audio_out = gr.Audio(label=tr(L0, "audio"), type="filepath")
                    file_out = gr.DownloadButton(tr(L0, "download"), size="sm")
                    info_out = gr.Markdown(elem_classes="result-info")

                # ---- Right column: pronunciation settings ----
                with gr.Column(scale=6):
                    with gr.Group():
                        sec_pron = gr.Markdown(tr(L0, "sec_pron"), elem_classes="section-head")
                        mode = gr.Radio(choices=mode_choices(L0), value=DEFAULT_MODE, label=tr(L0, "mode"))
                        speed = gr.Slider(
                            minimum=0.5, maximum=1.5, value=1.0, step=0.05, label=tr(L0, "speed")
                        )
                        style = gr.Textbox(
                            label=tr(L0, "style"),
                            placeholder=tr(L0, "style_ph"),
                            lines=2,
                            interactive=SUPPORTS_INSTRUCT,
                        )
                        with gr.Accordion(tr(L0, "acc_examples"), open=False) as acc_examples:
                            gr.Examples(examples=[[s] for s in STYLE_EXAMPLES], inputs=[style], label="")
                        with gr.Accordion(tr(L0, "acc_presets"), open=False) as acc_presets:
                            with gr.Row():
                                preset_dd = gr.Dropdown(
                                    choices=sorted(load_presets()), value=None,
                                    label=tr(L0, "preset_dd"), scale=3,
                                )
                                load_preset_btn = gr.Button(tr(L0, "load"), scale=1, size="sm")
                                del_preset_btn = gr.Button(tr(L0, "delete"), scale=1, size="sm")
                            with gr.Row():
                                preset_name = gr.Textbox(
                                    label=tr(L0, "preset_name"), placeholder=tr(L0, "preset_ph"), scale=3
                                )
                                save_preset_btn = gr.Button(tr(L0, "save"), scale=1, size="sm")
                            save_preset_btn.click(
                                fn=ui_save_preset, inputs=[preset_name, style],
                                outputs=[preset_dd, preset_name],
                            )
                            load_preset_btn.click(fn=ui_load_preset, inputs=[preset_dd], outputs=[style])
                            del_preset_btn.click(fn=ui_delete_preset, inputs=[preset_dd], outputs=[preset_dd])
                        with gr.Row():
                            p_before = gr.Slider(0, 2, value=0.0, step=0.05, label=tr(L0, "p_before"))
                            p_after = gr.Slider(0, 2, value=0.0, step=0.05, label=tr(L0, "p_after"))
                        with gr.Row():
                            p_between = gr.Slider(0, 2, value=0.3, step=0.05, label=tr(L0, "p_between"))
                            repeat = gr.Slider(1, 5, value=1, step=1, label=tr(L0, "repeat"))

            generate_btn.click(
                fn=ui_generate,
                inputs=[text, model_label, voice, mode, speed, style,
                        p_before, p_after, p_between, repeat, out_format, quality, lang],
                outputs=[audio_out, file_out, info_out],
            )

        with gr.Tab(tr(L0, "tab_batch")) as tab_batch:
            batch_desc = gr.Markdown(tr(L0, "batch_desc"))
            with gr.Row(equal_height=False):
                # ---- Left column: input ----
                with gr.Column(scale=5):
                    batch_text = gr.Textbox(
                        label=tr(L0, "batch_text"), lines=9, placeholder="你好\n谢谢\n再见"
                    )
                    batch_csv = gr.File(
                        label=tr(L0, "batch_csv"), file_types=[".csv"], type="filepath", height=120
                    )
                # ---- Right column: defaults ----
                with gr.Column(scale=6):
                    with gr.Group():
                        b_defaults = gr.Markdown(tr(L0, "b_defaults"), elem_classes="section-head")
                        b_model = gr.Radio(choices=model_choices(L0), value=DEFAULT_MODEL, label=tr(L0, "model"))
                        with gr.Row():
                            b_voice = gr.Dropdown(choices=VOICES, value=DEFAULT_VOICE, label=tr(L0, "voice"))
                            b_speed = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label=tr(L0, "speed"))
                        b_mode = gr.Radio(choices=mode_choices(L0), value=DEFAULT_MODE, label=tr(L0, "mode"))
                        b_style = gr.Textbox(label=tr(L0, "b_style"), lines=2, interactive=SUPPORTS_INSTRUCT)
                        with gr.Accordion(tr(L0, "acc_adv"), open=False) as b_acc_adv:
                            with gr.Row():
                                b_before = gr.Slider(0, 2, value=0.0, step=0.05, label=tr(L0, "p_before"))
                                b_after = gr.Slider(0, 2, value=0.0, step=0.05, label=tr(L0, "p_after"))
                            with gr.Row():
                                b_between = gr.Slider(0, 2, value=0.3, step=0.05, label=tr(L0, "p_between"))
                                b_repeat = gr.Slider(1, 5, value=1, step=1, label=tr(L0, "repeat"))
                        with gr.Row():
                            b_format = gr.Radio(choices=FORMATS, value=DEFAULT_FORMAT, label=tr(L0, "format"))
                            b_quality = gr.Dropdown(choices=QUALITIES, value=DEFAULT_QUALITY, label=tr(L0, "quality"))

            batch_btn = gr.Button(tr(L0, "batch_btn"), variant="primary")
            with gr.Row():
                with gr.Column(scale=3):
                    batch_summary = gr.Markdown(elem_classes="result-info")
                with gr.Column(scale=1, min_width=260):
                    batch_zip = gr.DownloadButton(tr(L0, "zip"), size="sm")
            batch_table = gr.Dataframe(
                headers=["text", "filename", "status", "error"],
                label=tr(L0, "table"), wrap=True, max_height=240,
            )

            batch_btn.click(
                fn=ui_batch,
                inputs=[batch_text, batch_csv, b_voice, b_model, b_mode, b_speed, b_style,
                        b_before, b_after, b_between, b_repeat, b_format, b_quality, lang],
                outputs=[batch_table, batch_zip, batch_summary],
            )

        footer = gr.Markdown(
            f"{tr(L0, 'footer')} `{OUTPUTS_DIR}`", elem_classes="app-footer"
        )

        # ---- Language switching ----
        # i18n_outputs and the update list returned by i18n_updates() MUST stay
        # in the same order.
        i18n_outputs = [
            header, tab_single, tab_batch,
            sec_text, text, sec_voice, model_label, voice,
            sec_out, out_format, quality, generate_btn, audio_out, file_out,
            sec_pron, mode, speed, style, acc_examples, acc_presets,
            preset_dd, load_preset_btn, del_preset_btn, preset_name, save_preset_btn,
            p_before, p_after, p_between, repeat,
            batch_desc, batch_text, batch_csv, b_defaults, b_model, b_voice, b_speed,
            b_mode, b_style, b_acc_adv, b_before, b_after, b_between, b_repeat,
            b_format, b_quality, batch_btn, batch_zip, batch_table,
            footer,
        ]

        def i18n_updates(lg: str) -> list:
            return [
                gr.update(value=header_html(lg)),                                        # header
                gr.update(label=tr(lg, "tab_single")),                                   # tab_single
                gr.update(label=tr(lg, "tab_batch")),                                    # tab_batch
                gr.update(value=tr(lg, "sec_text")),                                     # sec_text
                gr.update(label=tr(lg, "text_label"), placeholder=tr(lg, "text_ph")),    # text
                gr.update(value=tr(lg, "sec_voice")),                                    # sec_voice
                gr.update(label=tr(lg, "model"), choices=model_choices(lg)),             # model_label
                gr.update(label=tr(lg, "voice")),                                        # voice
                gr.update(value=tr(lg, "sec_out")),                                      # sec_out
                gr.update(label=tr(lg, "format")),                                       # out_format
                gr.update(label=tr(lg, "quality")),                                      # quality
                gr.update(value=tr(lg, "generate")),                                     # generate_btn
                gr.update(label=tr(lg, "audio")),                                        # audio_out
                gr.update(label=tr(lg, "download")),                                     # file_out
                gr.update(value=tr(lg, "sec_pron")),                                     # sec_pron
                gr.update(label=tr(lg, "mode"), choices=mode_choices(lg)),               # mode
                gr.update(label=tr(lg, "speed")),                                        # speed
                gr.update(label=tr(lg, "style"), placeholder=tr(lg, "style_ph")),        # style
                gr.update(label=tr(lg, "acc_examples")),                                 # acc_examples
                gr.update(label=tr(lg, "acc_presets")),                                  # acc_presets
                gr.update(label=tr(lg, "preset_dd")),                                    # preset_dd
                gr.update(value=tr(lg, "load")),                                         # load_preset_btn
                gr.update(value=tr(lg, "delete")),                                       # del_preset_btn
                gr.update(label=tr(lg, "preset_name"), placeholder=tr(lg, "preset_ph")), # preset_name
                gr.update(value=tr(lg, "save")),                                         # save_preset_btn
                gr.update(label=tr(lg, "p_before")),                                     # p_before
                gr.update(label=tr(lg, "p_after")),                                      # p_after
                gr.update(label=tr(lg, "p_between")),                                    # p_between
                gr.update(label=tr(lg, "repeat")),                                       # repeat
                gr.update(value=tr(lg, "batch_desc")),                                   # batch_desc
                gr.update(label=tr(lg, "batch_text")),                                   # batch_text
                gr.update(label=tr(lg, "batch_csv")),                                    # batch_csv
                gr.update(value=tr(lg, "b_defaults")),                                   # b_defaults
                gr.update(label=tr(lg, "model"), choices=model_choices(lg)),             # b_model
                gr.update(label=tr(lg, "voice")),                                        # b_voice
                gr.update(label=tr(lg, "speed")),                                        # b_speed
                gr.update(label=tr(lg, "mode"), choices=mode_choices(lg)),               # b_mode
                gr.update(label=tr(lg, "b_style")),                                      # b_style
                gr.update(label=tr(lg, "acc_adv")),                                      # b_acc_adv
                gr.update(label=tr(lg, "p_before")),                                     # b_before
                gr.update(label=tr(lg, "p_after")),                                      # b_after
                gr.update(label=tr(lg, "p_between")),                                    # b_between
                gr.update(label=tr(lg, "repeat")),                                       # b_repeat
                gr.update(label=tr(lg, "format")),                                       # b_format
                gr.update(label=tr(lg, "quality")),                                      # b_quality
                gr.update(value=tr(lg, "batch_btn")),                                    # batch_btn
                gr.update(label=tr(lg, "zip")),                                          # batch_zip
                gr.update(label=tr(lg, "table")),                                        # batch_table
                gr.update(value=f"{tr(lg, 'footer')} `{OUTPUTS_DIR}`"),                  # footer
            ]

        def on_lang_change(lg: str):
            lg = lg if lg in T else DEFAULT_LANG
            return [lg, *i18n_updates(lg)]

        lang.input(
            fn=on_lang_change, inputs=[lang],
            outputs=[lang_store, *i18n_outputs], show_progress="hidden",
        )

        def on_page_load(stored: str):
            lg = stored if stored in T else DEFAULT_LANG
            return [gr.update(value=lg), *i18n_updates(lg)]

        demo.load(
            fn=on_page_load, inputs=[lang_store],
            outputs=[lang, *i18n_outputs], show_progress="hidden",
        )

    return demo


# --------------------------------------------------------------------------- #
# Model download (setup)
# --------------------------------------------------------------------------- #
def download_models() -> None:
    """Download both models INTO the project's models/ folder (not ~/.cache)."""
    from huggingface_hub import snapshot_download

    print("Downloading both Qwen3-TTS models into the project 'models' folder")
    print("(this can take several minutes)…")
    for label, repo in MODELS.items():
        local = MODEL_LOCAL_DIRS[repo]
        print(f"\n>>> {label}\n    {repo}\n    -> {local}")
        if (local / "config.json").exists():
            print("    Already present, skipping.")
            continue
        try:
            snapshot_download(repo_id=repo, local_dir=str(local))
            print(f"    OK -> {local}")
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {exc}")
            raise
    print("\nAll models are in:", MODELS_DIR)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if "--download" in sys.argv:
        download_models()
        sys.exit(0)
    logo_file = find_logo_file()
    build_ui().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
        favicon_path=str(logo_file) if logo_file else None,
    )
