"""Model registry, capability detection and heavy-slot loading for the TTS tab.

Formerly the top of the single-file ``tts.py``; the package ``__init__``
re-exports these names so ``from modules import tts`` keeps working.
"""

from __future__ import annotations

import inspect
import shutil

from .. import memory_manager as mm
from .. import model_manager as mgr
from .. import model_select as ms

from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model


FFMPEG = shutil.which("ffmpeg")
SAMPLE_RATE = 24000  # Qwen3-TTS native output rate (mono, pcm_s16le)
LANG = "zh"

# Detect which optional features the installed MLX-Audio actually supports,
# instead of assuming argument names.
_GEN_PARAMS = inspect.signature(generate_audio).parameters
SUPPORTS_INSTRUCT = "instruct" in _GEN_PARAMS
SUPPORTS_SPEED = "speed" in _GEN_PARAMS
SUPPORTS_LANG = "lang_code" in _GEN_PARAMS
SUPPORTS_REF = "ref_audio" in _GEN_PARAMS
SUPPORTS_REF_TEXT = "ref_text" in _GEN_PARAMS
# Streaming decode needs BOTH flags: generate_audio only writes a file for a
# streamed run when save=True (`save_streamed_audio = stream and save`), so
# stream=True on its own would silently produce no output.
SUPPORTS_STREAM = "stream" in _GEN_PARAMS and "save" in _GEN_PARAMS

MODELS = {
    "Higher Quality — Qwen3-TTS 1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
}
# Base variant of the same size — used when Voice mode is "Clone a voice".
BASE_MODELS = {
    "Higher Quality — Qwen3-TTS 1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
}
DEFAULT_MODEL = "Higher Quality — Qwen3-TTS 1.7B"

# Fish Audio S2 Pro — a multilingual, zero-shot voice-cloning model. Unlike the
# Qwen sizes it has no named speakers and no CustomVoice/Base split: it clones
# the timbre of a reference clip the user supplies, and emotion/prosody are
# steered with inline [tag] markup in the text rather than a style instruction.
# It is selected like a "size" in the Model radio but drives a different UI and
# generation path (reference-audio upload instead of the named-voice picker).
FISH_LABEL = "Voice Clone — Fish S2 Pro 4B"
FISH_REPO = "mlx-community/fish-audio-s2-pro-8bit"

MODEL_SHORT = {
    MODELS["Higher Quality — Qwen3-TTS 1.7B"]: "1.7B",
    BASE_MODELS["Higher Quality — Qwen3-TTS 1.7B"]: "1.7B Base",
    FISH_REPO: "Fish S2 Pro",
}

# Registry keys used by the Models tab / installer.
REPO_TO_KEY = {
    MODELS["Higher Quality — Qwen3-TTS 1.7B"]: "tts-1.7b",
    BASE_MODELS["Higher Quality — Qwen3-TTS 1.7B"]: "tts-1.7b-base",
    FISH_REPO: "tts-fish-s2pro",
}


def voice_mode() -> str:
    """'CustomVoice' (built-in voices) or 'Base' (voice cloning), validated."""
    return ms.selected_key("tts_voice_mode")


def is_fish(model_label: str) -> bool:
    """True when the selected model is the Fish S2 Pro voice-cloning model."""
    return model_label == FISH_LABEL


def clone_active(model_label: str) -> bool:
    """True when this generation clones from an uploaded reference clip, i.e. the
    reference-clip uploader should be shown instead of the named-voice picker.

    Fish always clones. A Qwen size clones when Voice mode is "Clone a voice"
    (Base): the Qwen Base weights carry a speaker encoder that reproduces the
    timbre of a reference clip, so the uploader applies to them too."""
    return is_fish(model_label) or voice_mode() == "Base"


def active_repo(model_label: str) -> str:
    """The repo for this model. Fish has a single repo; the Qwen sizes depend on
    the currently saved voice mode (CustomVoice vs Base)."""
    if model_label == FISH_LABEL:
        return FISH_REPO
    table = BASE_MODELS if voice_mode() == "Base" else MODELS
    return table[model_label]


def resolve_model_source(repo: str) -> str:
    """Local weights path for this repo. Raises if not installed (no silent downloads)."""
    return mgr.model_path_or_error(REPO_TO_KEY[repo])


VOICES = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"]
DEFAULT_VOICE = "Vivian"

MODES = ["Single Word", "Vocabulary Phrase", "Sentence", "Paragraph"]
DEFAULT_MODE = "Sentence"

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

FORMATS = ["MP3", "WAV"]
DEFAULT_FORMAT = "MP3"

QUALITY_KBPS = {"128 kbps": 128, "192 kbps": 192, "256 kbps": 256}
QUALITIES = list(QUALITY_KBPS.keys())
DEFAULT_QUALITY = "192 kbps"


# --------------------------------------------------------------------------- #
# Model loading through the shared heavy slot
# --------------------------------------------------------------------------- #
def get_model(repo: str):
    return mm.HEAVY.get(
        f"tts:{repo}",
        f"Qwen3-TTS {MODEL_SHORT[repo]}",
        lambda: load_model(resolve_model_source(repo)),
    )


def needs_load(repo: str) -> bool:
    return mm.HEAVY.key != f"tts:{repo}"
