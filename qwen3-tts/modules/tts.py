"""Chinese text-to-speech (Qwen3-TTS + MLX-Audio) — ported unchanged from the
original single-purpose app: word/phrase/sentence/paragraph modes, style
instructions with saved presets, pauses & repeats, batch generation with CSV,
deterministic caching, WAV/MP3 export and vi/en/zh interface language.

Changes from the original: the model is held in the shared heavy-model slot
(one heavy model at a time) and weights are resolved through the model
registry — a missing model is reported instead of silently downloaded.
"""

from __future__ import annotations

import csv
import filecmp
import hashlib
import html
import inspect
import json
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import gradio as gr

from . import memory_manager as mm
from . import model_manager as mgr
from . import model_select as ms
from . import storage

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

MODELS = {
    "Fast — Qwen3-TTS 0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
    "Higher Quality — Qwen3-TTS 1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
}
# Base variants of the same sizes — used when Voice mode is "Clone a voice".
BASE_MODELS = {
    "Fast — Qwen3-TTS 0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
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
    MODELS["Fast — Qwen3-TTS 0.6B"]: "0.6B",
    MODELS["Higher Quality — Qwen3-TTS 1.7B"]: "1.7B",
    BASE_MODELS["Fast — Qwen3-TTS 0.6B"]: "0.6B Base",
    BASE_MODELS["Higher Quality — Qwen3-TTS 1.7B"]: "1.7B Base",
    FISH_REPO: "Fish S2 Pro",
}

# Registry keys used by the Models tab / installer.
REPO_TO_KEY = {
    MODELS["Fast — Qwen3-TTS 0.6B"]: "tts-0.6b",
    MODELS["Higher Quality — Qwen3-TTS 1.7B"]: "tts-1.7b",
    BASE_MODELS["Fast — Qwen3-TTS 0.6B"]: "tts-0.6b-base",
    BASE_MODELS["Higher Quality — Qwen3-TTS 1.7B"]: "tts-1.7b-base",
    FISH_REPO: "tts-fish-s2pro",
}


def voice_mode() -> str:
    """'CustomVoice' (built-in voices) or 'Base' (voice cloning), validated."""
    return ms.selected_key("tts_voice_mode")


def is_fish(model_label: str) -> bool:
    """True when the selected model is the Fish S2 Pro voice-cloning model."""
    return model_label == FISH_LABEL


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

PRESETS_FILE = storage.BASE_DIR / "style_presets.json"


def load_presets() -> dict:
    try:
        with open(PRESETS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (OSError, ValueError):
        pass
    return {}


def write_presets(presets: dict) -> None:
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
# Internationalisation — Tiếng Việt / English / 中文 (TTS tab)
# --------------------------------------------------------------------------- #
DEFAULT_LANG = "en"
LANG_CHOICES = [("Tiếng Việt", "vi"), ("English", "en"), ("中文", "zh")]

T = {
    "en": {
        "tab_single": "Single",
        "tab_batch": "Batch",
        "sec_text": "✍️ **Text**",
        "text_label": "Chinese text",
        "text_ph": "在这里输入中文…",
        "sec_voice": "🎤 **Voice & Model**",
        "model": "Model",
        "model_fast": "Fast — 0.6B",
        "model_hq": "Higher quality — 1.7B",
        "model_clone": "Voice clone — Fish S2 Pro",
        "ref_audio": "Reference voice clip (5–15s)",
        "ref_text": "Reference transcript (optional)",
        "ref_text_ph": "Type what is said in the reference clip — improves cloning",
        "clone_note": (
            "🎭 **Fish S2 Pro** clones the uploaded voice. Steer emotion with inline "
            "tags in your text, e.g. `[happy]`, `[whisper]`, `[shout]`. Voice/style "
            "presets above don't apply to this model."
        ),
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
        "tab_files": "Library",
        "files_hint": "Files in the outputs folder, newest first — click a row to play audio or preview text.",
        "refresh": "🔄 Refresh",
        "open_folder": "📂 Open folder",
        "files_table": "Output files",
        "file_player": "Play selected file",
        "file_text": "File contents",
        "res_cached": "♻️ From cache",
        "res_new": "🆕 Newly generated",
        "res_file": "Saved file",
        "loading": "Loading model: {model} … this can take a moment.",
        "item_of": "Item {n} of {total}",
        "packaging": "Packaging results",
        "batch_done": "**Done.** {ok} succeeded, {failed} failed, {total} total.",
    },
    "vi": {
        "tab_single": "Tạo đơn",
        "tab_batch": "Tạo hàng loạt",
        "sec_text": "✍️ **Văn bản**",
        "text_label": "Văn bản tiếng Trung",
        "text_ph": "在这里输入中文…",
        "sec_voice": "🎤 **Giọng & Mô hình**",
        "model": "Mô hình",
        "model_fast": "Nhanh — 0.6B",
        "model_hq": "Chất lượng cao — 1.7B",
        "model_clone": "Nhân bản giọng — Fish S2 Pro",
        "ref_audio": "Mẫu giọng tham chiếu (5–15 giây)",
        "ref_text": "Lời thoại của mẫu (tuỳ chọn)",
        "ref_text_ph": "Nhập nội dung nói trong mẫu — giúp nhân bản chính xác hơn",
        "clone_note": (
            "🎭 **Fish S2 Pro** nhân bản giọng đã tải lên. Điều khiển cảm xúc bằng thẻ "
            "trong văn bản, ví dụ `[happy]`, `[whisper]`, `[shout]`. Các thiết lập giọng/"
            "phong cách ở trên không áp dụng cho mô hình này."
        ),
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
        "tab_files": "Thư viện",
        "files_hint": "Tệp trong thư mục đầu ra, mới nhất trước — nhấp vào một dòng để phát âm thanh hoặc xem nội dung.",
        "refresh": "🔄 Làm mới",
        "open_folder": "📂 Mở thư mục",
        "files_table": "Tệp đầu ra",
        "file_player": "Phát tệp đã chọn",
        "file_text": "Nội dung tệp",
        "res_cached": "♻️ Lấy từ bộ nhớ đệm",
        "res_new": "🆕 Vừa tạo mới",
        "res_file": "Tệp đã lưu",
        "loading": "Đang nạp mô hình: {model}…",
        "item_of": "Mục {n} / {total}",
        "packaging": "Đang đóng gói kết quả",
        "batch_done": "**Hoàn tất.** {ok} thành công, {failed} thất bại, tổng cộng {total}.",
    },
    "zh": {
        "tab_single": "单条生成",
        "tab_batch": "批量生成",
        "sec_text": "✍️ **文本**",
        "text_label": "中文文本",
        "text_ph": "在这里输入中文…",
        "sec_voice": "🎤 **声音与模型**",
        "model": "模型",
        "model_fast": "快速 — 0.6B",
        "model_hq": "高质量 — 1.7B",
        "model_clone": "声音克隆 — Fish S2 Pro",
        "ref_audio": "参考声音片段（5–15 秒）",
        "ref_text": "参考文本（可选）",
        "ref_text_ph": "输入参考片段中所说的内容 — 可提升克隆效果",
        "clone_note": (
            "🎭 **Fish S2 Pro** 会克隆上传的声音。用文本中的标签控制情感，例如 "
            "`[happy]`、`[whisper]`、`[shout]`。上方的声音/风格设置对该模型无效。"
        ),
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
        "tab_files": "文件库",
        "files_hint": "输出文件夹中的文件（最新在前）— 点击一行即可播放音频或预览内容。",
        "refresh": "🔄 刷新",
        "open_folder": "📂 打开文件夹",
        "files_table": "输出文件",
        "file_player": "播放所选文件",
        "file_text": "文件内容",
        "res_cached": "♻️ 来自缓存",
        "res_new": "🆕 新生成",
        "res_file": "已保存文件",
        "loading": "正在加载模型：{model}……",
        "item_of": "第 {n} / {total} 条",
        "packaging": "正在打包结果",
        "batch_done": "**完成。**成功 {ok} 条，失败 {failed} 条，共 {total} 条。",
    },
}


def tr(lang: str, key: str) -> str:
    table = T.get(lang) or T[DEFAULT_LANG]
    return table.get(key) or T[DEFAULT_LANG].get(key, key)


def model_choices(lang: str) -> list[tuple[str, str]]:
    return [
        (tr(lang, "model_fast"), "Fast — Qwen3-TTS 0.6B"),
        (tr(lang, "model_hq"), "Higher Quality — Qwen3-TTS 1.7B"),
        (tr(lang, "model_clone"), FISH_LABEL),
    ]


def mode_choices(lang: str) -> list[tuple[str, str]]:
    return [
        (tr(lang, "mode_word"), "Single Word"),
        (tr(lang, "mode_phrase"), "Vocabulary Phrase"),
        (tr(lang, "mode_sent"), "Sentence"),
        (tr(lang, "mode_para"), "Paragraph"),
    ]


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
    text = (text or "").strip()
    cleaned = re.sub(r"[^\w一-鿿]+", "_", text, flags=re.UNICODE).strip("_")
    if not cleaned:
        cleaned = "audio"
    return cleaned[:40]


def unique_output_path(text: str, ext: str, cache_file: Path) -> Path:
    """Friendly output name (你好.mp3); numeric suffix if taken by different audio."""
    stub = safe_stub(text)
    outputs = storage.outputs_dir()
    candidate = outputs / f"{stub}.{ext}"
    i = 2
    while candidate.exists():
        if (
            candidate.stat().st_size == cache_file.stat().st_size
            and filecmp.cmp(candidate, cache_file, shallow=False)
        ):
            return candidate
        candidate = outputs / f"{stub}_{i}.{ext}"
        i += 1
    return candidate


def cache_hash(params: dict) -> str:
    blob = json.dumps(params, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_digest(path: str) -> str:
    """Content hash of a file (used to key the cache on the reference clip)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Long-text chunking
# --------------------------------------------------------------------------- #
# The model generates a bounded number of tokens per call (~12.5 tokens/sec of
# audio). A long paragraph fed in one shot can exceed that budget and trail off
# into silence when the cap is hit mid-utterance, so we split on sentence
# boundaries and synthesize each piece separately, then concatenate. Short text
# yields a single chunk and behaves exactly as before.
MAX_CHUNK_CHARS = 120
_SENT_END = "。！？!?；;…\n"
_SENT_SPLIT = re.compile(rf"[^{re.escape(_SENT_END)}]*[{re.escape(_SENT_END)}]?", re.UNICODE)
_CLAUSE_END = "，,、：:"
_CLAUSE_SPLIT = re.compile(rf"[^{re.escape(_CLAUSE_END)}]*[{re.escape(_CLAUSE_END)}]?", re.UNICODE)


def _hard_pieces(sentence: str, max_chars: int) -> list[str]:
    """Break an over-long sentence on clause punctuation, then on raw length."""
    pieces: list[str] = []
    for clause in _CLAUSE_SPLIT.findall(sentence):
        clause = clause.strip()
        if not clause:
            continue
        while len(clause) > max_chars:
            pieces.append(clause[:max_chars])
            clause = clause[max_chars:]
        if clause:
            pieces.append(clause)
    return pieces


def split_for_tts(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into synthesis chunks no longer than max_chars, preferring
    sentence boundaries and packing consecutive sentences together."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = [s.strip() for s in _SENT_SPLIT.findall(text) if s.strip()]
    chunks: list[str] = []
    cur = ""
    for s in sentences:
        parts = _hard_pieces(s, max_chars) if len(s) > max_chars else [s]
        for part in parts:
            if cur and len(cur) + len(part) > max_chars:
                chunks.append(cur)
                cur = ""
            cur += part
    if cur:
        chunks.append(cur)
    return chunks


def chunk_max_tokens(chunk: str) -> int:
    """Generous per-chunk token cap: comfortably above what a chunk this long
    needs (~2.8 tokens/char observed) so generation always reaches EOS."""
    return max(512, min(4096, len(chunk) * 8 + 128))


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
    ref_audio: str | None = None,
    ref_text: str = "",
) -> dict:
    _require_ffmpeg()

    fish = is_fish(model_label)

    text = (text or "").strip()
    if not text:
        raise ValueError("Please enter some Chinese text.")
    if model_label not in MODELS and not fish:
        raise ValueError(f"Unknown model: {model_label}")
    if not fish and voice not in VOICES:
        raise ValueError(f"Unknown voice: {voice}. Allowed: {', '.join(VOICES)}")
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}. Allowed: {', '.join(MODES)}")
    if fish and not ref_audio:
        raise ValueError(
            "Fish S2 Pro clones a voice from a reference clip — upload a short "
            "(5–15 second) audio sample of the target voice first."
        )

    repo = active_repo(model_label)
    ref_text = (ref_text or "").strip()
    # Fish clones from the reference clip; the Qwen Base voice mode relies on the
    # model's own default timbre. Either way there is no named speaker to pass.
    cloning = fish or voice_mode() == "Base"
    ext = "mp3" if str(out_format).upper() == "MP3" else "wav"
    kbps = QUALITY_KBPS.get(quality, 192)
    repeat = max(1, min(5, int(repeat)))
    speed = float(speed)
    style = (style or "").strip()
    p_before = max(0.0, min(2.0, float(p_before)))
    p_after = max(0.0, min(2.0, float(p_after)))
    p_between = max(0.0, min(2.0, float(p_between)))

    params = {
        "text": text,
        "repo": repo,
        "voice": "clone" if fish else voice,
        "mode": mode,
        "speed": round(speed, 3),
        "style": "" if fish else style,
        "p_before": round(p_before, 3),
        "p_after": round(p_after, 3),
        "p_between": round(p_between, 3),
        "repeat": repeat,
        "format": ext,
        "quality": kbps,
    }
    if fish:
        params["ref"] = file_digest(ref_audio)
        params["ref_text"] = ref_text
    digest = cache_hash(params)
    cache_file = storage.cache_dir() / f"{digest}.{ext}"

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

    cfg = MODE_CFG[mode]
    eff_speed = max(0.5, min(1.5, speed * cfg["speed_factor"]))
    # Fish is driven by inline [tags], not a Chinese style instruction.
    eff_instruct = "" if fish else (style or cfg["instruct"])
    eff_before = max(p_before, cfg["min_pad"])
    eff_after = max(p_after, cfg["min_pad"])

    model = get_model(repo)
    mm.set_task(f"TTS: generating ({MODEL_SHORT[repo]})…")

    try:
        with tempfile.TemporaryDirectory(dir=storage.temp_dir()) as td:
            tdp = Path(td)

            base_kwargs = dict(
                model=model,
                output_path=str(tdp),
                audio_format="wav",
                join_audio=True,
                verbose=False,
            )
            # Named speaker only for the Qwen CustomVoice models. Fish clones from
            # the uploaded reference clip; the Qwen Base mode has no named speaker
            # and falls back to its own default timbre.
            if fish:
                if SUPPORTS_REF and ref_audio:
                    base_kwargs["ref_audio"] = ref_audio
                if SUPPORTS_REF_TEXT and ref_text:
                    base_kwargs["ref_text"] = ref_text
            elif not cloning:
                base_kwargs["voice"] = voice
            if SUPPORTS_SPEED:
                base_kwargs["speed"] = eff_speed
            # Qwen is Chinese-only (force zh); Fish is multilingual — let it detect.
            if SUPPORTS_LANG and not fish:
                base_kwargs["lang_code"] = LANG
            if SUPPORTS_INSTRUCT and eff_instruct and not fish:
                base_kwargs["instruct"] = eff_instruct

            # Long text is synthesized in sentence-sized chunks so a single
            # over-long generation can't hit the token cap and trail off into
            # silence. Each chunk is written as seg000.wav, seg001.wav, … and
            # concatenated below in lexical (= reading) order.
            chunks = split_for_tts(text)
            for idx, chunk in enumerate(chunks):
                if len(chunks) > 1:
                    mm.set_task(
                        f"TTS: generating ({MODEL_SHORT[repo]}) — "
                        f"part {idx + 1}/{len(chunks)}…"
                    )
                out_name = f"seg{idx:03d}"
                kwargs = dict(
                    base_kwargs,
                    text=chunk,
                    file_prefix=out_name,
                    max_tokens=chunk_max_tokens(chunk),
                )
                try:
                    generate_audio(**kwargs)
                except TypeError:
                    for opt in ("instruct", "speed", "lang_code", "max_tokens"):
                        kwargs.pop(opt, None)
                    generate_audio(**kwargs)

                if not (tdp / f"{out_name}.wav").exists():
                    raise RuntimeError(
                        "The model did not produce any audio. Try shorter text, "
                        "another voice, or the other model."
                    )

            produced = sorted(tdp.glob("seg*.wav"))
            if not produced:
                raise RuntimeError(
                    "The model did not produce any audio. Try shorter text, another "
                    "voice, or the other model."
                )

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

            if ext == "wav":
                shutil.copy2(assembled, cache_file)
            else:
                to_mp3(assembled, cache_file, kbps)
    finally:
        mm.set_task("idle")

    friendly = unique_output_path(text, ext, cache_file)
    shutil.copy2(cache_file, friendly)
    return {
        "path": friendly,
        "cached": False,
        "gen_time": time.time() - t0,
        "repo": repo,
        "voice": "clone" if fish else voice,
        "digest": digest,
    }


# --------------------------------------------------------------------------- #
# Error mapping -> friendly gr.Error messages
# --------------------------------------------------------------------------- #
def to_friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "not installed" in low and "models tab" in low:
        return msg
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
            "again.\n\nDetails: " + msg
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
# UI callbacks (single + batch + presets)
# --------------------------------------------------------------------------- #
def ui_generate(
    text, model_label, voice, mode, speed, style,
    p_before, p_after, p_between, repeat, out_format, quality,
    ref_audio=None, ref_text="",
    lang=DEFAULT_LANG,
):
    try:
        repo = active_repo(model_label) if (model_label in MODELS or is_fish(model_label)) else None
        if repo and needs_load(repo):
            gr.Info(tr(lang, "loading").format(model=model_label))
        res = generate_one(
            text, model_label, voice, mode, speed, style,
            p_before, p_after, p_between, repeat, out_format, quality,
            ref_audio=ref_audio, ref_text=ref_text,
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
    clone_keys = {"fish", "clone", "s2", "s2 pro", "fish s2 pro", "voice clone",
                  FISH_LABEL.lower()}
    if v in fast_keys or v == "fast — qwen3-tts 0.6b":
        return "Fast — Qwen3-TTS 0.6B"
    if v in hq_keys or v == "higher quality — qwen3-tts 1.7b":
        return "Higher Quality — Qwen3-TTS 1.7B"
    if v in clone_keys:
        return FISH_LABEL
    raise ValueError(
        f"CSV line {lineno}: unknown model '{value}'. Use '0.6B' (fast), '1.7B' "
        "(quality), or 'fish' (voice clone)."
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
    ref_audio=None, ref_text="",
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
                ref_audio=ref_audio, ref_text=ref_text,
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

    outputs = storage.outputs_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = outputs / f"batch_results_{ts}.csv"
    with open(results_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "filename", "path", "status", "error"])
        writer.writeheader()
        writer.writerows(ordered)

    zip_path = outputs / f"batch_{ts}.zip"
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


# --------------------------------------------------------------------------- #
# UI — built inside the host app's "TTS" tab. Returns the i18n machinery so
# the host can wire the global language switch.
# --------------------------------------------------------------------------- #
def build_tts_tab(lang_component, settings: dict):
    L0 = DEFAULT_LANG
    default_model = settings.get("default_tts_model", DEFAULT_MODEL)
    if default_model not in MODELS:
        default_model = DEFAULT_MODEL
    show_adv = bool(settings.get("show_advanced", True))

    if not FFMPEG:
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
            with gr.Column(scale=5):
                with gr.Group():
                    sec_text = gr.Markdown(tr(L0, "sec_text"), elem_classes="section-head")
                    text = gr.Textbox(
                        label=tr(L0, "text_label"),
                        placeholder=tr(L0, "text_ph"),
                        lines=3,
                    )
                with gr.Group():
                    sec_voice = gr.Markdown(tr(L0, "sec_voice"), elem_classes="section-head")
                    model_label = gr.Radio(
                        choices=model_choices(L0), value=default_model, label=tr(L0, "model")
                    )
                    ms.build_selector("tts_voice_mode")
                    is_clone0 = default_model == FISH_LABEL
                    voice = gr.Radio(
                        choices=VOICES, value=DEFAULT_VOICE, label=tr(L0, "voice"),
                        visible=not is_clone0,
                    )
                    ref_audio = gr.Audio(
                        label=tr(L0, "ref_audio"), type="filepath",
                        sources=["upload", "microphone"], visible=is_clone0,
                    )
                    ref_text = gr.Textbox(
                        label=tr(L0, "ref_text"), placeholder=tr(L0, "ref_text_ph"),
                        lines=2, visible=is_clone0,
                    )
                    clone_note = gr.Markdown(
                        tr(L0, "clone_note"), visible=is_clone0, elem_classes="hint-text"
                    )

                    def _persist_model(label: str) -> None:
                        if label in MODELS:
                            s = storage.load_settings()
                            s["default_tts_model"] = label
                            storage.save_settings(s)

                    def _toggle_clone(label: str):
                        """Swap the visible controls when the model changes: named-voice
                        picker for Qwen, reference-clip uploader for Fish S2 Pro."""
                        clone = is_fish(label)
                        return (
                            gr.update(visible=not clone),  # voice
                            gr.update(visible=clone),      # ref_audio
                            gr.update(visible=clone),      # ref_text
                            gr.update(visible=clone),      # clone_note
                        )

                    model_label.change(_persist_model, inputs=[model_label],
                                       show_progress="hidden")
                    model_label.change(
                        _toggle_clone, inputs=[model_label],
                        outputs=[voice, ref_audio, ref_text, clone_note],
                        show_progress="hidden",
                    )
                    sec_out = gr.Markdown(tr(L0, "sec_out"), elem_classes="section-head")
                    with gr.Row():
                        out_format = gr.Radio(
                            choices=FORMATS, value=DEFAULT_FORMAT, label=tr(L0, "format"), scale=2
                        )
                        quality = gr.Radio(
                            choices=QUALITIES, value=DEFAULT_QUALITY, label=tr(L0, "quality"), scale=3
                        )
                generate_btn = gr.Button(tr(L0, "generate"), variant="primary")
                audio_out = gr.Audio(label=tr(L0, "audio"), type="filepath", elem_id="tts-audio")
                file_out = gr.DownloadButton(tr(L0, "download"), size="sm")
                info_out = gr.Markdown(elem_classes="result-info")

            with gr.Column(scale=6):
                with gr.Group():
                    sec_pron = gr.Markdown(tr(L0, "sec_pron"), elem_classes="section-head")
                    mode = gr.Radio(choices=mode_choices(L0), value=DEFAULT_MODE, label=tr(L0, "mode"))
                    style = gr.Textbox(
                        label=tr(L0, "style"),
                        placeholder=tr(L0, "style_ph"),
                        lines=2,
                        interactive=SUPPORTS_INSTRUCT,
                    )
                    with gr.Accordion(tr(L0, "acc_examples"), open=False, visible=show_adv) as acc_examples:
                        gr.Examples(examples=[[s] for s in STYLE_EXAMPLES], inputs=[style], label="")
                    with gr.Accordion(tr(L0, "acc_presets"), open=False, visible=show_adv) as acc_presets:
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
                        speed = gr.Slider(
                            minimum=0.5, maximum=1.5, value=1.0, step=0.05, label=tr(L0, "speed")
                        )
                        repeat = gr.Slider(1, 5, value=1, step=1, label=tr(L0, "repeat"))
                    with gr.Row():
                        p_before = gr.Slider(0, 2, value=0.0, step=0.05, label=tr(L0, "p_before"))
                        p_after = gr.Slider(0, 2, value=0.0, step=0.05, label=tr(L0, "p_after"))
                        p_between = gr.Slider(0, 2, value=0.3, step=0.05, label=tr(L0, "p_between"))

                with gr.Group():
                    with gr.Row(elem_id="lib-head"):
                        sec_files = gr.Markdown(
                            f"📁 **{tr(L0, 'tab_files')}**", elem_classes="section-head"
                        )
                        files_refresh_btn = gr.Button(
                            "🔄", size="sm", scale=0, min_width=40, elem_classes="icon-btn"
                        )
                        files_open_btn = gr.Button(
                            "📂", size="sm", scale=0, min_width=40, elem_classes="icon-btn"
                        )
                    files_list = gr.HTML(lib_html())
                    file_text = gr.Textbox(
                        label=tr(L0, "file_text"), lines=6, max_lines=12,
                        visible=False, show_copy_button=True,
                    )
                    lib_evt = gr.Textbox(visible=False, elem_id="lib-evt")

        generate_btn.click(
            fn=ui_generate,
            inputs=[text, model_label, voice, mode, speed, style,
                    p_before, p_after, p_between, repeat, out_format, quality,
                    ref_audio, ref_text, lang_component],
            outputs=[audio_out, file_out, info_out],
        ).then(ui_files_refresh, outputs=[files_list], show_progress="hidden")

        files_refresh_btn.click(ui_files_refresh, outputs=[files_list], show_progress="hidden")
        files_open_btn.click(lambda: storage.open_in_finder(storage.outputs_dir()))
        lib_evt.input(ui_lib_event, inputs=[lib_evt], outputs=[file_text], show_progress="hidden")
        tab_single.select(ui_files_refresh, outputs=[files_list], show_progress="hidden")

    with gr.Tab(tr(L0, "tab_batch")) as tab_batch:
        batch_desc = gr.Markdown(tr(L0, "batch_desc"))
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                batch_text = gr.Textbox(
                    label=tr(L0, "batch_text"), lines=9, placeholder="你好\n谢谢\n再见"
                )
                batch_csv = gr.File(
                    label=tr(L0, "batch_csv"), file_types=[".csv"], type="filepath", height=120
                )
            with gr.Column(scale=6):
                with gr.Group():
                    b_defaults = gr.Markdown(tr(L0, "b_defaults"), elem_classes="section-head")
                    b_model = gr.Radio(choices=model_choices(L0), value=default_model, label=tr(L0, "model"))
                    with gr.Row():
                        b_voice = gr.Radio(choices=VOICES, value=DEFAULT_VOICE, label=tr(L0, "voice"))
                        b_speed = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label=tr(L0, "speed"))
                    b_mode = gr.Radio(choices=mode_choices(L0), value=DEFAULT_MODE, label=tr(L0, "mode"))
                    b_style = gr.Textbox(label=tr(L0, "b_style"), lines=2, interactive=SUPPORTS_INSTRUCT)
                    b_ref_audio = gr.Audio(
                        label=tr(L0, "ref_audio"), type="filepath",
                        sources=["upload", "microphone"], visible=is_clone0,
                    )
                    b_ref_text = gr.Textbox(
                        label=tr(L0, "ref_text"), placeholder=tr(L0, "ref_text_ph"),
                        lines=2, visible=is_clone0,
                    )

                    def _toggle_bclone(label: str):
                        clone = is_fish(label)
                        return (
                            gr.update(visible=not clone),  # b_voice
                            gr.update(visible=clone),      # b_ref_audio
                            gr.update(visible=clone),      # b_ref_text
                        )

                    b_model.change(
                        _toggle_bclone, inputs=[b_model],
                        outputs=[b_voice, b_ref_audio, b_ref_text],
                        show_progress="hidden",
                    )
                    with gr.Accordion(tr(L0, "acc_adv"), open=False) as b_acc_adv:
                        with gr.Row():
                            b_before = gr.Slider(0, 2, value=0.0, step=0.05, label=tr(L0, "p_before"))
                            b_after = gr.Slider(0, 2, value=0.0, step=0.05, label=tr(L0, "p_after"))
                        with gr.Row():
                            b_between = gr.Slider(0, 2, value=0.3, step=0.05, label=tr(L0, "p_between"))
                            b_repeat = gr.Slider(1, 5, value=1, step=1, label=tr(L0, "repeat"))
                    with gr.Row():
                        b_format = gr.Radio(choices=FORMATS, value=DEFAULT_FORMAT, label=tr(L0, "format"))
                        b_quality = gr.Radio(choices=QUALITIES, value=DEFAULT_QUALITY, label=tr(L0, "quality"))

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
                    b_before, b_after, b_between, b_repeat, b_format, b_quality,
                    b_ref_audio, b_ref_text, lang_component],
            outputs=[batch_table, batch_zip, batch_summary],
        )

    footer = gr.Markdown(
        f"{tr(L0, 'footer')} `{storage.outputs_dir()}`", elem_classes="app-footer"
    )

    # i18n_outputs and the update list returned by i18n_updates() MUST stay
    # in the same order.
    i18n_outputs = [
        tab_single, tab_batch,
        sec_text, text, sec_voice, model_label, voice,
        sec_out, out_format, quality, generate_btn, audio_out, file_out,
        sec_pron, mode, speed, style, acc_examples, acc_presets,
        preset_dd, load_preset_btn, del_preset_btn, preset_name, save_preset_btn,
        p_before, p_after, p_between, repeat,
        sec_files, file_text,
        batch_desc, batch_text, batch_csv, b_defaults, b_model, b_voice, b_speed,
        b_mode, b_style, b_acc_adv, b_before, b_after, b_between, b_repeat,
        b_format, b_quality, batch_btn, batch_zip, batch_table,
        footer,
        ref_audio, ref_text, clone_note, b_ref_audio, b_ref_text,
    ]

    def i18n_updates(lg: str) -> list:
        return [
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
            gr.update(value=f"📁 **{tr(lg, 'tab_files')}**"),                        # sec_files
            gr.update(label=tr(lg, "file_text")),                                    # file_text
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
            gr.update(value=f"{tr(lg, 'footer')} `{storage.outputs_dir()}`"),        # footer
            gr.update(label=tr(lg, "ref_audio")),                                    # ref_audio
            gr.update(label=tr(lg, "ref_text"), placeholder=tr(lg, "ref_text_ph")),  # ref_text
            gr.update(value=tr(lg, "clone_note")),                                    # clone_note
            gr.update(label=tr(lg, "ref_audio")),                                    # b_ref_audio
            gr.update(label=tr(lg, "ref_text"), placeholder=tr(lg, "ref_text_ph")),  # b_ref_text
        ]

    return i18n_outputs, i18n_updates
