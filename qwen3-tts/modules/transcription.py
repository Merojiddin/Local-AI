"""Speech-to-text — Whisper via the STT engine already bundled with
MLX-Audio. The model (large-v3-turbo or large-v3) comes from the
speech-to-text selector. Accepts WAV/MP3/M4A/MP4 (decoded locally with
FFmpeg).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gradio as gr

from . import memory_manager as mm
from . import model_manager as mgr
from . import model_select as ms
from . import storage

LANG_OPTIONS = [
    ("Auto-detect", "auto"),
    ("中文 Chinese", "zh"),
    ("English", "en"),
    ("Tiếng Việt Vietnamese", "vi"),
]
TASK_OPTIONS = [
    ("Transcription", "transcribe"),
    ("Translate to English", "translate"),
]


def _load():
    # Import models before generate: mlx_audio 0.4.5 has a circular import
    # if mlx_audio.stt.generate is imported first.
    import mlx_audio.stt.models  # noqa: F401
    from mlx_audio.stt.models.whisper import Model

    key = ms.selected_key("stt")
    path = mgr.model_path_or_error(key)
    name = mgr.MODELS[key]["name"]
    return mm.HEAVY.get(
        f"whisper:{key}", name,
        lambda: Model.from_pretrained(path),
    )


def _seg_field(seg, name, default=None):
    if isinstance(seg, dict):
        return seg.get(name, default)
    return getattr(seg, name, default)


def _fmt_ts(seconds: float, srt: bool = False) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = seconds % 60
    if srt:
        return f"{h:02d}:{m:02d}:{int(s):02d},{int(round(s % 1 * 1000)):03d}"
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def _to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = _seg_field(seg, "start", 0.0)
        end = _seg_field(seg, "end", start)
        text = (_seg_field(seg, "text", "") or "").strip()
        if not text:
            continue
        lines.append(f"{i}\n{_fmt_ts(start, srt=True)} --> {_fmt_ts(end, srt=True)}\n{text}\n")
    return "\n".join(lines)


def transcribe(audio_file, language, task, timestamps):
    if not audio_file:
        raise gr.Error("Upload an audio or video file first (WAV, MP3, M4A or MP4).")
    if Path(audio_file).suffix.lower() not in (".wav", ".mp3", ".m4a", ".mp4", ".aac", ".flac", ".ogg", ".mov"):
        raise gr.Error("Unsupported file type. Use WAV, MP3, M4A or MP4.")

    try:
        model = _load()
    except RuntimeError as exc:
        raise gr.Error(str(exc))
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Could not load Whisper: {exc}")

    mm.set_task("Transcribing…")
    try:
        result = model.generate(
            str(audio_file),
            language=None if language == "auto" else language,
            task=task,
            return_timestamps=True,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Transcription failed: {exc}")
    finally:
        mm.HEAVY.touch()
        mm.set_task("idle")

    text = (getattr(result, "text", "") or "").strip()
    segments = getattr(result, "segments", None) or []
    detected = getattr(result, "language", None)

    if timestamps and segments:
        shown = "\n".join(
            f"[{_fmt_ts(_seg_field(s, 'start', 0.0))} → {_fmt_ts(_seg_field(s, 'end', 0.0))}] "
            f"{(_seg_field(s, 'text', '') or '').strip()}"
            for s in segments if (_seg_field(s, "text", "") or "").strip()
        )
    else:
        shown = text

    stem = Path(audio_file).stem[:40] or "transcript"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs = storage.outputs_dir()
    txt_path = outputs / f"{stem}_{ts}.txt"
    txt_path.write_text(text + "\n", encoding="utf-8")
    srt_update = gr.update(value=None, visible=False)
    if segments:
        srt_path = outputs / f"{stem}_{ts}.srt"
        srt_path.write_text(_to_srt(segments), encoding="utf-8")
        srt_update = gr.update(value=str(srt_path), visible=True)

    info = f"**Saved:** `{txt_path}`"
    if detected:
        info = f"**Detected language:** {detected} · " + info
    return shown, str(txt_path), srt_update, info


def build_transcribe_tab(settings: dict):
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            ms.build_selector("stt")
            audio_file = gr.File(
                label="Audio / video file (WAV, MP3, M4A, MP4)",
                file_types=[".wav", ".mp3", ".m4a", ".mp4"],
                type="filepath", height=110,
            )
            with gr.Group():
                language = gr.Dropdown(choices=LANG_OPTIONS, value="auto", label="Language")
                task = gr.Radio(choices=TASK_OPTIONS, value="transcribe", label="Task")
                timestamps = gr.Checkbox(value=False, label="Show timestamps")
            go_btn = gr.Button("📝 Transcribe", variant="primary")
        with gr.Column(scale=6):
            result = gr.Textbox(
                label="Transcript", lines=14, max_lines=20, show_copy_button=True,
            )
            with gr.Row():
                txt_dl = gr.DownloadButton("⬇️ TXT", size="sm")
                srt_dl = gr.DownloadButton("⬇️ SRT", size="sm", visible=False)
            info = gr.Markdown(elem_classes="result-info")

    go_btn.click(
        transcribe,
        inputs=[audio_file, language, task, timestamps],
        outputs=[result, txt_dl, srt_dl, info],
    )
