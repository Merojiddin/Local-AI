"""FFmpeg helpers: silence, normalisation, concatenation and MP3 encoding."""

from __future__ import annotations

import re
import shutil
import subprocess
import wave
from pathlib import Path

from .config import FFMPEG, SAMPLE_RATE


# --------------------------------------------------------------------------- #
# Pause stretching
# --------------------------------------------------------------------------- #
# The model already pauses at most sentence ends — just briefly (0.2-0.9s
# observed), and it sometimes runs two sentences together. Rather than cut the
# text into one generation per sentence (which restarts prosody at every full
# stop), the pauses it *did* produce are found in the finished audio and padded
# out to the length the user asked for.
#
# The cut is made at the MIDPOINT of a detected silence, never at its edges, so
# no speech can be clipped by a slightly-early boundary: the operation only ever
# inserts silence and never removes a sample of audio.
SILENCE_NOISE_DB = -35.0   # below this is "silence" (audio is loudnorm'd to -16 LUFS)
SILENCE_MIN_GAP = 0.08     # shorter dips are stop consonants, not pauses
SILENCE_SENT_MIN = 0.30    # a gap at least this long is read as a sentence break
_SIL_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SIL_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


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


def normalize_wav_loud(src: Path, dst: Path) -> None:
    """Like normalize_wav, but also level each piece to a common loudness (EBU
    R128) so separately-generated chunks don't jump in volume when joined. Used
    only for multi-chunk long text; single-shot audio is left untouched."""
    _run_ffmpeg(
        [
            "-i", str(src),
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst),
        ]
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


def wav_duration(path: Path) -> float:
    """Seconds of audio in a PCM wav, read from its header (no ffprobe needed)."""
    with wave.open(str(path)) as w:
        return w.getnframes() / float(w.getframerate())


def detect_silences(src: Path) -> list[tuple[float, float]]:
    """(start, end) of every silent span, via ffmpeg's silencedetect filter."""
    _require_ffmpeg()
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-nostats", "-i", str(src),
         "-af", f"silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_GAP}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    spans: list[tuple[float, float]] = []
    start: float | None = None
    for line in (proc.stderr or "").splitlines():
        m = _SIL_START.search(line)
        if m:
            start = float(m.group(1))
            continue
        m = _SIL_END.search(line)
        if m and start is not None:
            spans.append((start, float(m.group(1))))
            start = None
    if start is not None:                      # file ended while still silent
        spans.append((start, wav_duration(src)))
    return spans


def stretch_silences(
    src: Path, dst: Path, workdir: Path, tag: str, sentence: float, comma: float
) -> int:
    """Pad the pauses already present in src out to the requested lengths.

    Gaps of at least SILENCE_SENT_MIN are treated as sentence breaks, shorter
    ones as clause breaks; each is topped up to its target (never shortened).
    Leading and trailing silence is left alone — that is the caller's
    pause-before/pause-after. Returns how many gaps were lengthened.
    """
    total = wav_duration(src)
    inserts: list[tuple[float, float]] = []
    for s_start, s_end in detect_silences(src):
        if s_start <= 0.01 or s_end >= total - 0.01:
            continue
        gap = s_end - s_start
        target = sentence if gap >= SILENCE_SENT_MIN else comma
        if target > gap:
            inserts.append(((s_start + s_end) / 2.0, target - gap))

    if not inserts:
        shutil.copy2(src, dst)
        return 0

    parts: list[Path] = []
    prev = 0.0
    for i, (pos, extra) in enumerate(inserts):
        seg = workdir / f"{tag}sp{i:03d}.wav"
        # -ss/-t (offset + duration) rather than -ss/-to: unambiguous about
        # whether the end time is relative to the cut or to the original file.
        _run_ffmpeg(["-i", str(src), "-ss", f"{prev:.4f}", "-t", f"{pos - prev:.4f}",
                     "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(seg)])
        sil = workdir / f"{tag}sg{i:03d}.wav"
        make_silence(extra, sil)
        parts += [seg, sil]
        prev = pos
    tail = workdir / f"{tag}sptail.wav"
    _run_ffmpeg(["-i", str(src), "-ss", f"{prev:.4f}",
                 "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(tail)])
    parts.append(tail)
    concat_wavs(parts, dst)
    return len(inserts)
