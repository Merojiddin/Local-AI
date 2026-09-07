"""FFmpeg helpers: silence, normalisation, concatenation and MP3 encoding."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import FFMPEG, SAMPLE_RATE


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
