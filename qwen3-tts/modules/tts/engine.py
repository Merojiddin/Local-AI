"""Core single-item generation (with cache), error mapping and batch parsing."""

from __future__ import annotations

import csv
import shutil
import tempfile
import time
from pathlib import Path

from .. import memory_manager as mm
from .. import storage

from mlx_audio.tts.generate import generate_audio

from .config import (
    FISH_LABEL,
    LANG,
    MODE_CFG,
    MODEL_SHORT,
    MODELS,
    MODES,
    QUALITY_KBPS,
    SUPPORTS_INSTRUCT,
    SUPPORTS_LANG,
    SUPPORTS_REF,
    SUPPORTS_REF_TEXT,
    SUPPORTS_SPEED,
    VOICES,
    active_repo,
    get_model,
    is_fish,
    voice_mode,
)
from .audio import (
    _require_ffmpeg,
    concat_wavs,
    make_silence,
    normalize_wav,
    normalize_wav_loud,
    to_mp3,
)
from .chunking import (
    FISH_MAX_CHUNK_CHARS,
    MAX_CHUNK_CHARS,
    chunk_max_tokens,
    split_for_tts,
)
from .naming import cache_hash, file_digest, unique_output_path


def _no_audio_message(fish: bool) -> str:
    """Message for a missing segment. mlx_audio's generate_audio swallows every
    exception (it just prints and returns without writing a file), so the real
    cause never reaches us. On this hardware a silently-missing Fish segment is
    almost always its 44.1 kHz vocoder overflowing the ~8.9 GiB single-buffer
    limit, so point Fish users at the lighter Qwen path that avoids it."""
    if fish:
        return (
            "The Fish S2 Pro model ran out of GPU memory decoding this audio "
            "(its 44.1 kHz vocoder needs more than macOS allows in one buffer). "
            "Try shorter text, or switch the model to Qwen3-TTS 1.7B with Voice "
            "mode set to “Clone a voice” — it clones from your "
            "reference clip too, but is far lighter on memory."
        )
    return (
        "The model did not produce any audio. Try shorter text, another voice, "
        "or the other model."
    )


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
            chunk_chars = FISH_MAX_CHUNK_CHARS if fish else MAX_CHUNK_CHARS
            chunks = split_for_tts(text, chunk_chars)
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
                    max_tokens=chunk_max_tokens(chunk, fish=fish),
                )
                try:
                    generate_audio(**kwargs)
                except TypeError:
                    for opt in ("instruct", "speed", "lang_code", "max_tokens"):
                        kwargs.pop(opt, None)
                    generate_audio(**kwargs)

                if not (tdp / f"{out_name}.wav").exists():
                    raise RuntimeError(_no_audio_message(fish))

            produced = sorted(tdp.glob("seg*.wav"))
            if not produced:
                raise RuntimeError(_no_audio_message(fish))

            base = tdp / "base.wav"
            if len(produced) == 1:
                normalize_wav(produced[0], base)
            else:
                # Loudness-match the separately-generated chunks so the joined
                # audio doesn't jump in volume from one piece to the next.
                norm_parts = []
                for i, p in enumerate(produced):
                    np_ = tdp / f"n{i}.wav"
                    normalize_wav_loud(p, np_)
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
            "Insufficient memory. Close other apps and try again with shorter text."
            "\n\nDetails: " + msg
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
    # The 0.6B "fast" size was removed; legacy fast aliases now map to the single
    # remaining Qwen size (1.7B) so old CSVs keep working.
    qwen_keys = {"fast", "0.6b", "0.6", "small", "quality", "higher quality", "hq",
                 "1.7b", "1.7", "large", MODELS["Higher Quality — Qwen3-TTS 1.7B"].lower()}
    clone_keys = {"fish", "clone", "s2", "s2 pro", "fish s2 pro", "voice clone",
                  FISH_LABEL.lower()}
    if v in qwen_keys or v in ("fast — qwen3-tts 0.6b", "higher quality — qwen3-tts 1.7b"):
        return "Higher Quality — Qwen3-TTS 1.7B"
    if v in clone_keys:
        return FISH_LABEL
    raise ValueError(
        f"CSV line {lineno}: unknown model '{value}'. Use '1.7B' (quality) or "
        "'fish' (voice clone)."
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
