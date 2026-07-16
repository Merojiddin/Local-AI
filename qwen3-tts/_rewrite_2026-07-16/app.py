#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Chinese Text-to-Speech (Qwen3-TTS via MLX-Audio)
======================================================

Runs 100% locally on Apple Silicon. No cloud, no login, no database.

Models (both stored inside ./models):
  Fast           -> mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit
  Higher Quality -> mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit

Usage:
  python app.py             start the web app on http://127.0.0.1:7860
  python app.py --download  download both models, then exit
"""

from __future__ import annotations

import csv
import gc
import hashlib
import inspect
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
import zipfile
from datetime import datetime
from pathlib import Path

try:
    import gradio as gr
    import numpy as np
except ImportError as _exc:  # keep the message beginner-friendly
    print(f"❌ A Python dependency is missing: {_exc}")
    print("Please run install.command again to repair the installation.")
    sys.exit(1)

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
for _d in (OUTPUTS_DIR, CACHE_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FFMPEG = shutil.which("ffmpeg")

MODELS = {
    "Fast — Qwen3-TTS 0.6B": {
        "repo": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
        "dir": MODELS_DIR / "Qwen3-TTS-0.6B",
    },
    "Higher Quality — Qwen3-TTS 1.7B": {
        "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "dir": MODELS_DIR / "Qwen3-TTS-1.7B",
    },
}
DEFAULT_MODEL = "Fast — Qwen3-TTS 0.6B"

VOICES = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"]
DEFAULT_VOICE = "Vivian"

# Per-mode behavior: default style instruction (used only when the user leaves
# the instruction box empty) and extra edge silence so short clips are not abrupt.
MODES = {
    "Single Word": {
        "instruct": "请用很慢、非常清晰的语速朗读这个词语，每个音节都发音清楚。",
        "edge_pad": 0.25,
        "short": "word",
    },
    "Vocabulary Phrase": {
        "instruct": "请清晰地朗读这个词组，语速稍慢，发音清楚。",
        "edge_pad": 0.15,
        "short": "phrase",
    },
    "Sentence": {
        "instruct": "",  # natural Mandarin, no special instruction
        "edge_pad": 0.05,
        "short": "sentence",
    },
    "Paragraph": {
        "instruct": "请自然流畅地朗读，语气自然，句子之间自然停顿。",
        "edge_pad": 0.05,
        "short": "paragraph",
    },
}
DEFAULT_MODE = "Sentence"

INSTRUCTION_EXAMPLES = [
    "Speak slowly and clearly like a Chinese teacher",
    "Use neutral Standard Mandarin",
    "Speak naturally",
    "Emphasize every syllable clearly",
    "Use a warm female teaching voice",
]

MP3_BITRATES = ["128 kbps", "192 kbps", "256 kbps"]
DEFAULT_BITRATE = "192 kbps"

SEGMENT_GAP_SECONDS = 0.2  # silence between text segments (multi-line input)


# --------------------------------------------------------------------------- #
# Friendly errors
# --------------------------------------------------------------------------- #
class AppError(Exception):
    """An error with a message safe to show directly to the user."""


def _require_ffmpeg() -> str:
    if not FFMPEG:
        raise AppError(
            "FFmpeg is not installed. Open Terminal, run:  brew install ffmpeg  "
            "then restart the app (it is needed for MP3 output and the speed slider)."
        )
    return FFMPEG


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, AppError):
        return str(exc)
    text = f"{type(exc).__name__}: {exc}"
    lowered = str(exc).lower()
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return (
            f"A Python dependency is missing ({exc}). "
            "Please run install.command again to repair the installation."
        )
    if "out of memory" in lowered or "metal" in lowered and "alloc" in lowered:
        return (
            "The Mac ran out of memory while generating audio. Close other apps, "
            "switch to the Fast (0.6B) model, and try again."
        )
    return f"Audio generation failed: {text}"


# --------------------------------------------------------------------------- #
# Model download
# --------------------------------------------------------------------------- #
def download_models() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is missing. Run install.command first.")
        sys.exit(1)
    for label, spec in MODELS.items():
        target = spec["dir"]
        print(f"\nDownloading {label}\n  {spec['repo']}\n  -> {target}")
        try:
            snapshot_download(repo_id=spec["repo"], local_dir=str(target))
        except Exception as exc:  # noqa: BLE001
            print(
                f"\n❌ Download failed for {spec['repo']}: {exc}\n"
                "Check your internet connection and run install.command again."
            )
            sys.exit(1)
    print("\n✅ Both models are downloaded.")


def model_is_downloaded(model_key: str) -> bool:
    d = MODELS[model_key]["dir"]
    return d.is_dir() and any(d.glob("*.safetensors"))


# --------------------------------------------------------------------------- #
# Model manager — keeps one model in memory at a time (16 GB RAM friendly)
# --------------------------------------------------------------------------- #
class ModelManager:
    def __init__(self) -> None:
        self._key: str | None = None
        self._model = None
        self.supports_instruct = False
        self.lock = threading.Lock()  # serializes all generation

    def _release(self) -> None:
        self._model = None
        self._key = None
        gc.collect()
        try:
            import mlx.core as mx

            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except ImportError:
            pass

    def get(self, model_key: str, status=None):
        """Return the loaded model for model_key, swapping models if needed."""
        if model_key not in MODELS:
            raise AppError(f"Unknown model: {model_key}")
        if self._key == model_key and self._model is not None:
            return self._model
        if not model_is_downloaded(model_key):
            raise AppError(
                f"The model files for “{model_key}” were not found in the models folder. "
                "Please run install.command (or `python app.py --download`) first."
            )
        if self._model is not None:
            if status:
                status(f"Releasing {self._key}…")
            self._release()
        if status:
            status(f"Loading {model_key} (first use takes a moment)…")
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:
            raise AppError(
                f"MLX-Audio is not installed correctly ({exc}). Run install.command again."
            ) from exc
        try:
            model = load_model(MODELS[model_key]["dir"])
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                f"Could not load {model_key}: {exc}. "
                "If this keeps happening, delete the models folder and run install.command again."
            ) from exc
        # Verify the installed MLX-Audio generate() API before relying on options.
        params = inspect.signature(model.generate).parameters
        for required in ("text", "voice", "lang_code"):
            if required not in params:
                raise AppError(
                    "The installed MLX-Audio version has an unexpected API "
                    f"(model.generate has no “{required}” argument). "
                    "Run install.command again to restore the tested version (0.4.5)."
                )
        self.supports_instruct = "instruct" in params
        self._model = model
        self._key = model_key
        return model


MANAGER = ModelManager()


# --------------------------------------------------------------------------- #
# Audio helpers (16-bit mono WAV via stdlib; FFmpeg for tempo + MP3)
# --------------------------------------------------------------------------- #
def write_wav(path: Path, audio, sample_rate: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        frames = w.readframes(w.getnframes())
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def _run_ffmpeg(args: list[str]) -> None:
    cmd = [_require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AppError(f"FFmpeg failed: {result.stderr.strip()[:400]}")


def change_tempo(audio, sample_rate: int, speed: float):
    """Pitch-preserving speed change through FFmpeg's atempo filter."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        dst = Path(tmp) / "out.wav"
        write_wav(src, audio, sample_rate)
        _run_ffmpeg(
            ["-i", str(src), "-filter:a", f"atempo={speed:.3f}",
             "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le", str(dst)]
        )
        data, _ = read_wav(dst)
    return data


def wav_to_mp3(wav_path: Path, mp3_path: Path, bitrate: str) -> None:
    kbps = bitrate.split()[0]
    _run_ffmpeg(["-i", str(wav_path), "-c:a", "libmp3lame", "-b:a", f"{kbps}k", str(mp3_path)])


# --------------------------------------------------------------------------- #
# Cache + filenames
# --------------------------------------------------------------------------- #
def cache_key(settings: dict) -> str:
    payload = json.dumps(settings, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_filename(text: str, voice: str, model_key: str, mode: str, key: str, ext: str) -> Path:
    core = re.sub(r"[^\w]+", "", text, flags=re.UNICODE).strip("_")[:16] or "audio"
    model_short = "0.6B" if "0.6B" in model_key else "1.7B"
    base = f"{core}_{voice}_{model_short}_{MODES[mode]['short']}_{key[:8]}"
    candidate = OUTPUTS_DIR / f"{base}.{ext}"
    counter = 1
    while candidate.exists():
        candidate = OUTPUTS_DIR / f"{base}_{counter}.{ext}"
        counter += 1
    return candidate


# --------------------------------------------------------------------------- #
# Core synthesis (shared by single and batch generation)
# --------------------------------------------------------------------------- #
def synthesize(
    text: str,
    model_key: str,
    voice: str,
    mode: str,
    speed: float,
    instruction: str,
    pause_before: float,
    pause_after: float,
    pause_between: float,
    repeat: int,
    out_format: str,
    bitrate: str,
    status=None,
) -> dict:
    """Generate (or fetch from cache) one audio file. Returns metadata dict."""
    text = (text or "").strip()
    if not text:
        raise AppError("Please enter some Chinese text first. 请先输入中文文本。")
    if model_key not in MODELS:
        raise AppError(f"Unknown model: {model_key}")
    if voice not in VOICES:
        raise AppError(f"Unknown voice: {voice}. Choose one of: {', '.join(VOICES)}")
    if mode not in MODES:
        raise AppError(f"Unknown mode: {mode}. Choose one of: {', '.join(MODES)}")
    speed = float(min(1.5, max(0.5, speed)))
    repeat = int(min(5, max(1, repeat)))
    ext = "mp3" if out_format == "MP3" else "wav"
    if ext == "mp3" or abs(speed - 1.0) > 1e-3:
        _require_ffmpeg()  # fail early with a clear message

    instruction = (instruction or "").strip()
    effective_instruct = instruction or MODES[mode]["instruct"]

    settings = {
        "text": text,
        "model": model_key,
        "voice": voice,
        "mode": mode,
        "speed": round(speed, 3),
        "instruct": effective_instruct,
        "pause_before": round(float(pause_before), 2),
        "pause_after": round(float(pause_after), 2),
        "pause_between": round(float(pause_between), 2),
        "repeat": repeat,
        "format": ext,
        "bitrate": bitrate if ext == "mp3" else "",
    }
    key = cache_key(settings)
    sidecar = CACHE_DIR / f"{key}.json"

    # ---- cache hit -----------------------------------------------------------
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            cached_path = BASE_DIR / meta["output"]
            if cached_path.exists():
                meta["cached"] = True
                meta["path"] = str(cached_path)
                return meta
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt sidecar -> regenerate

    # ---- generate ------------------------------------------------------------
    started = time.time()
    with MANAGER.lock:
        model = MANAGER.get(model_key, status=status)
        use_instruct = effective_instruct if MANAGER.supports_instruct else None
        if status:
            status("Generating audio…")
        try:
            results = list(
                model.generate(
                    text=text,
                    voice=voice.lower(),
                    instruct=use_instruct or None,
                    lang_code="chinese",
                    verbose=False,
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise AppError(friendly_error(exc)) from exc

    if not results:
        raise AppError("The model returned no audio. Try shorter text or another voice.")
    sample_rate = int(results[0].sample_rate)

    def silence(seconds: float):
        return np.zeros(int(sample_rate * max(0.0, seconds)), dtype=np.float32)

    gap = silence(SEGMENT_GAP_SECONDS)
    pieces = []
    for i, r in enumerate(results):
        if i:
            pieces.append(gap)
        pieces.append(np.asarray(np.array(r.audio), dtype=np.float32).reshape(-1))
    speech = np.concatenate(pieces)

    if abs(speed - 1.0) > 1e-3:
        if status:
            status("Adjusting speed…")
        speech = change_tempo(speech, sample_rate, speed)

    edge = silence(MODES[mode]["edge_pad"])
    unit = np.concatenate([edge, speech, edge])
    parts = [silence(pause_before)]
    for i in range(repeat):
        if i:
            parts.append(silence(pause_between))
        parts.append(unit)
    parts.append(silence(pause_after))
    final = np.concatenate(parts)

    out_path = safe_filename(text, voice, model_key, mode, key, ext)
    if ext == "wav":
        write_wav(out_path, final, sample_rate)
    else:
        if status:
            status("Converting to MP3…")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_wav = Path(tmp) / "final.wav"
            write_wav(tmp_wav, final, sample_rate)
            wav_to_mp3(tmp_wav, out_path, bitrate)

    meta = {
        "output": str(out_path.relative_to(BASE_DIR)),
        "path": str(out_path),
        "model": model_key,
        "voice": voice,
        "mode": mode,
        "gen_seconds": round(time.time() - started, 2),
        "created": datetime.now().isoformat(timespec="seconds"),
        "cached": False,
    }
    sidecar.write_text(
        json.dumps({k: v for k, v in meta.items() if k not in ("cached", "path")},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return meta


# --------------------------------------------------------------------------- #
# Gradio handlers
# --------------------------------------------------------------------------- #
def generate_single(text, model_key, voice, mode, speed, instruction,
                    pause_before, pause_after, pause_between, repeat,
                    out_format, bitrate, progress=gr.Progress()):
    def status(msg: str) -> None:
        progress(0.4, desc=msg)

    try:
        meta = synthesize(text, model_key, voice, mode, speed, instruction,
                          pause_before, pause_after, pause_between, int(repeat),
                          out_format, bitrate, status=status)
    except AppError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(friendly_error(exc)) from exc

    state = "Loaded from cache ⚡" if meta["cached"] else "Newly generated ✅"
    summary = (
        f"**Saved to:** `{meta['output']}`  \n"
        f"**Model:** {meta['model']}  ·  **Voice:** {meta['voice']}  \n"
        f"**Generation time:** {meta['gen_seconds']} s  ·  **Status:** {state}"
    )
    return meta["path"], meta["path"], summary


def _normalize_batch_row(row: dict, defaults: dict, line_no: int) -> dict:
    """Validate one batch row; raises AppError with the row number on problems."""
    text = (row.get("text") or "").strip()
    if not text:
        raise AppError(f"Row {line_no}: empty text.")

    voice_raw = (row.get("voice") or "").strip()
    voice = defaults["voice"]
    if voice_raw:
        match = [v for v in VOICES if v.lower() == voice_raw.lower()]
        if not match:
            raise AppError(f"Row {line_no}: unknown voice “{voice_raw}”. Use: {', '.join(VOICES)}")
        voice = match[0]

    model_raw = (row.get("model") or "").strip()
    model_key = defaults["model"]
    if model_raw:
        if "1.7" in model_raw:
            model_key = "Higher Quality — Qwen3-TTS 1.7B"
        elif "0.6" in model_raw or model_raw.lower() == "fast":
            model_key = "Fast — Qwen3-TTS 0.6B"
        else:
            raise AppError(f"Row {line_no}: unknown model “{model_raw}”. Use 0.6B or 1.7B.")

    mode_raw = (row.get("mode") or "").strip()
    mode = defaults["mode"]
    if mode_raw:
        match = [m for m in MODES if m.lower() == mode_raw.lower()]
        if not match:
            raise AppError(f"Row {line_no}: unknown mode “{mode_raw}”. Use: {', '.join(MODES)}")
        mode = match[0]

    speed_raw = (row.get("speed") or "").strip() if isinstance(row.get("speed"), str) else row.get("speed")
    speed = defaults["speed"]
    if speed_raw not in (None, ""):
        try:
            speed = min(1.5, max(0.5, float(speed_raw)))
        except (TypeError, ValueError):
            raise AppError(f"Row {line_no}: invalid speed “{speed_raw}” (use 0.5–1.5).")

    return {"text": text, "voice": voice, "model": model_key, "mode": mode, "speed": speed}


def parse_batch_rows(batch_text: str, csv_path: str | None, defaults: dict):
    """Returns (rows, row_errors): rows are normalized dicts, errors are (text, message)."""
    rows, errors = [], []
    line_no = 0
    for line in (batch_text or "").splitlines():
        if not line.strip():
            continue
        line_no += 1
        try:
            rows.append(_normalize_batch_row({"text": line}, defaults, line_no))
        except AppError as exc:
            errors.append((line.strip(), str(exc)))
    if csv_path:
        try:
            raw = Path(csv_path).read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            raise AppError(f"Could not read the CSV file: {exc}. Save it as UTF-8 CSV.")
        reader = csv.DictReader(io.StringIO(raw))
        fields = [f.strip().lower() for f in (reader.fieldnames or [])]
        if "text" not in fields:
            raise AppError(
                "The CSV file needs a header row with a “text” column "
                "(optional columns: voice, model, mode, speed)."
            )
        for i, raw_row in enumerate(reader, start=2):  # header is line 1
            row = {(k or "").strip().lower(): (v or "") for k, v in raw_row.items()}
            try:
                rows.append(_normalize_batch_row(row, defaults, i))
            except AppError as exc:
                errors.append(((row.get("text") or "").strip(), str(exc)))
    return rows, errors


def generate_batch(batch_text, csv_file, model_key, voice, mode, speed, instruction,
                   pause_before, pause_after, pause_between, repeat,
                   out_format, bitrate, progress=gr.Progress()):
    defaults = {"voice": voice, "model": model_key, "mode": mode, "speed": float(speed)}

    try:
        rows, row_errors = parse_batch_rows(batch_text, csv_file, defaults)
    except AppError as exc:
        raise gr.Error(str(exc)) from exc
    if not rows and not row_errors:
        raise gr.Error("Batch is empty: paste one line per item or upload a CSV file.")

    results = [{"text": t, "filename": "", "path": "", "status": "error", "error": e}
               for t, e in row_errors]
    # Process grouped by model so the 16 GB Mac never swaps models per row.
    order = sorted(range(len(rows)), key=lambda i: rows[i]["model"] != model_key)
    generated, cached, failed = 0, 0, len(row_errors)
    row_results: dict[int, dict] = {}
    for done, i in enumerate(order):
        row = rows[i]
        progress((done + 1) / max(1, len(rows)),
                 desc=f"{done + 1}/{len(rows)}: {row['text'][:20]}")
        try:
            meta = synthesize(row["text"], row["model"], row["voice"], row["mode"],
                              row["speed"], instruction, pause_before, pause_after,
                              pause_between, int(repeat), out_format, bitrate)
            if meta["cached"]:
                cached += 1
            else:
                generated += 1
            row_results[i] = {"text": row["text"], "filename": Path(meta["path"]).name,
                              "path": meta["output"], "status": "cached" if meta["cached"] else "generated",
                              "error": ""}
        except Exception as exc:  # noqa: BLE001
            failed += 1
            row_results[i] = {"text": row["text"], "filename": "", "path": "",
                              "status": "error", "error": friendly_error(exc)}
    results.extend(row_results[i] for i in range(len(rows)))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUTS_DIR / f"batch_results_{stamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "filename", "path", "status", "error"])
        writer.writeheader()
        writer.writerows(results)

    zip_path = OUTPUTS_DIR / f"batch_audio_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        seen = set()
        for r in results:
            if r["status"] != "error" and r["path"] and r["filename"] not in seen:
                zf.write(BASE_DIR / r["path"], arcname=r["filename"])
                seen.add(r["filename"])
        zf.write(csv_path, arcname=csv_path.name)

    summary = (
        f"**Done.** {generated} generated · {cached} from cache · {failed} failed  \n"
        f"ZIP: `{zip_path.relative_to(BASE_DIR)}` · Results CSV: `{csv_path.relative_to(BASE_DIR)}`"
    )
    if failed:
        first_errors = [r for r in results if r["status"] == "error"][:3]
        summary += "  \n" + "  \n".join(f"⚠️ {r['error']}" for r in first_errors)
    return str(zip_path), str(csv_path), summary


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_ui():
    warnings = []
    if not FFMPEG:
        warnings.append("⚠️ **FFmpeg not found** — MP3 output and the speed slider will not work. "
                        "Install it with `brew install ffmpeg`.")
    missing = [k for k in MODELS if not model_is_downloaded(k)]
    if missing:
        warnings.append("⚠️ **Model files missing:** " + ", ".join(missing) +
                        ". Run **install.command** (or `python app.py --download`).")

    theme = gr.themes.Soft(primary_hue="orange", neutral_hue="stone")
    with gr.Blocks(theme=theme, title="Local Chinese TTS — Qwen3") as demo:
        gr.Markdown("# 🌅 Local Chinese Text-to-Speech\n"
                    "Qwen3-TTS running fully on this Mac — no internet needed after install.")
        if warnings:
            gr.Markdown("\n\n".join(warnings))

        gr.Markdown("## Text")
        text_in = gr.Textbox(label="Chinese text 中文文本", lines=6,
                             placeholder="例如：你好！今天天气很好。")

        gr.Markdown("## Voice and Model")
        with gr.Row():
            model_in = gr.Radio(list(MODELS), value=DEFAULT_MODEL, label="Model")
            voice_in = gr.Radio(VOICES, value=DEFAULT_VOICE, label="Voice",
                                info="Dylan = Beijing dialect, Eric = Sichuan dialect")

        gr.Markdown("## Pronunciation Settings")
        with gr.Row():
            mode_in = gr.Radio(list(MODES), value=DEFAULT_MODE, label="Generation mode")
            speed_in = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed",
                                 info="Pitch is preserved (FFmpeg)")
        instruct_in = gr.Dropdown(
            INSTRUCTION_EXAMPLES, value=None, allow_custom_value=True,
            label="Style instruction (optional)",
            info="Pick an example or type your own; leave empty for the mode's default style.")
        with gr.Row():
            pause_before_in = gr.Slider(0, 2, value=0.0, step=0.1, label="Pause before (s)")
            pause_after_in = gr.Slider(0, 2, value=0.0, step=0.1, label="Pause after (s)")
            pause_between_in = gr.Slider(0, 2, value=0.6, step=0.1, label="Pause between repeats (s)")
            repeat_in = gr.Slider(1, 5, value=1, step=1, label="Repeat count")

        gr.Markdown("## Output Settings")
        with gr.Row():
            format_in = gr.Radio(["MP3", "WAV"], value="MP3", label="Format")
            bitrate_in = gr.Radio(MP3_BITRATES, value=DEFAULT_BITRATE,
                                  label="MP3 quality (ignored for WAV)")

        gr.Markdown("## Generate")
        go_btn = gr.Button("Generate Audio", variant="primary")
        audio_out = gr.Audio(label="Result", type="filepath")
        file_out = gr.File(label="Download")
        info_out = gr.Markdown()

        shared_settings = [model_in, voice_in, mode_in, speed_in, instruct_in,
                           pause_before_in, pause_after_in, pause_between_in,
                           repeat_in, format_in, bitrate_in]
        go_btn.click(generate_single, inputs=[text_in, *shared_settings],
                     outputs=[audio_out, file_out, info_out])

        gr.Markdown("## Batch Generation")
        gr.Markdown("One Chinese word or sentence per line, and/or upload a CSV with a "
                    "`text` column (optional columns: `voice`, `model`, `mode`, `speed`). "
                    "Settings above are used as defaults.")
        batch_text_in = gr.Textbox(label="One item per line", lines=5,
                                   placeholder="你好\n谢谢\n再见")
        batch_file_in = gr.File(label="CSV file (optional)", file_types=[".csv"], type="filepath")
        batch_btn = gr.Button("Generate Batch")
        batch_zip_out = gr.File(label="ZIP of all audio")
        batch_csv_out = gr.File(label="Results CSV")
        batch_info_out = gr.Markdown()
        batch_btn.click(generate_batch,
                        inputs=[batch_text_in, batch_file_in, *shared_settings],
                        outputs=[batch_zip_out, batch_csv_out, batch_info_out])

        gr.Markdown(f"Generated files are saved in `{OUTPUTS_DIR}`.")
    return demo


def main() -> None:
    if "--download" in sys.argv:
        download_models()
        return
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    try:
        demo.launch(server_name="127.0.0.1", server_port=7860,
                    inbrowser=True, show_api=False)
    except OSError:
        print("Port 7860 is busy — starting on a free port instead.")
        demo.launch(server_name="127.0.0.1", inbrowser=True, show_api=False)


if __name__ == "__main__":
    main()
