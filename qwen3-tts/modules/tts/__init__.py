"""Chinese text-to-speech (Qwen3-TTS + MLX-Audio) tab.

Once a single ``tts.py``; now split into focused submodules — config, i18n,
audio, chunking, naming, engine, library and ui. This package re-exports the
public surface so callers such as ``from modules import tts`` and
``tts.build_tts_tab`` / ``tts.split_for_tts`` keep working unchanged.
"""

from __future__ import annotations

from .config import (
    BASE_MODELS,
    DEFAULT_FORMAT,
    DEFAULT_MODE,
    DEFAULT_MODEL,
    DEFAULT_QUALITY,
    DEFAULT_VOICE,
    FFMPEG,
    FISH_LABEL,
    FISH_REPO,
    FORMATS,
    LANG,
    MODE_CFG,
    MODEL_SHORT,
    MODELS,
    MODES,
    QUALITIES,
    QUALITY_KBPS,
    REPO_TO_KEY,
    SAMPLE_RATE,
    STYLE_EXAMPLES,
    SUPPORTS_INSTRUCT,
    SUPPORTS_LANG,
    SUPPORTS_REF,
    SUPPORTS_REF_TEXT,
    SUPPORTS_SPEED,
    VOICES,
    active_repo,
    get_model,
    is_fish,
    needs_load,
    resolve_model_source,
    voice_mode,
)
from .i18n import (
    DEFAULT_LANG,
    LANG_CHOICES,
    T,
    mode_choices,
    model_choices,
    tr,
)
from .audio import (
    concat_wavs,
    make_silence,
    normalize_wav,
    normalize_wav_loud,
    to_mp3,
)
from .naming import cache_hash, file_digest, safe_stub, unique_output_path
from .chunking import MAX_CHUNK_CHARS, chunk_max_tokens, split_for_tts
from .engine import (
    _resolve_mode,
    _resolve_model,
    _resolve_speed,
    _resolve_voice,
    generate_one,
    parse_batch,
    to_friendly_error,
)
from .library import (
    AUDIO_EXTS,
    TEXT_EXTS,
    lib_html,
    ui_files_refresh,
    ui_lib_event,
)
from .ui import build_tts_tab, ui_batch, ui_generate

__all__ = [
    "build_tts_tab",
    "DEFAULT_LANG",
    "LANG_CHOICES",
    "T",
    "tr",
    "model_choices",
    "mode_choices",
    "MODELS",
    "BASE_MODELS",
    "MODEL_SHORT",
    "REPO_TO_KEY",
    "FISH_LABEL",
    "FISH_REPO",
    "DEFAULT_MODEL",
    "VOICES",
    "MODES",
    "FORMATS",
    "QUALITIES",
    "is_fish",
    "active_repo",
    "voice_mode",
    "resolve_model_source",
    "get_model",
    "needs_load",
    "split_for_tts",
    "chunk_max_tokens",
    "MAX_CHUNK_CHARS",
    "generate_one",
    "parse_batch",
    "to_friendly_error",
    "lib_html",
    "ui_generate",
    "ui_batch",
]
