"""Saved voice-clone profiles.

A voice clone is driven by an uploaded reference clip (+ optional transcript).
On its own that clip is ephemeral — it lives only in the browser upload and is
gone on the next reload, so the same voice has to be re-uploaded every session.

This module persists a loaded clone to disk under a user-given name so it can be
reloaded later. Each profile is two files in ``storage.voices_dir()``:

  <slug>.<ext>   the reference clip, copied in
  <slug>.json    metadata — name, clip filename, transcript, digest, created

Flat sidecar files (rather than one manifest) mean a corrupt/edited entry never
takes the whole library down: ``list_voices`` just skips it.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .. import storage
from .naming import file_digest

# Clip extensions we accept; anything else is copied in as .wav (mlx_audio reads
# the container, not the extension, but keeping a real suffix is friendlier).
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".aac"}


def _slug(name: str) -> str:
    """Filesystem-safe stem from a display name (keeps CJK, like safe_stub)."""
    cleaned = re.sub(r"[^\w一-鿿]+", "_", (name or "").strip(), flags=re.UNICODE).strip("_")
    return (cleaned or "voice")[:40]


def _manifest(slug: str) -> Path:
    return storage.voices_dir() / f"{slug}.json"


def list_voices() -> list[dict]:
    """All saved profiles, newest first. Entries whose clip is missing or whose
    metadata won't parse are skipped so the picker never points at a dead file."""
    out: list[dict] = []
    vdir = storage.voices_dir()
    for meta in vdir.glob("*.json"):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        clip = vdir / str(data.get("audio", ""))
        if not data.get("name") or not clip.is_file():
            continue
        data["slug"] = meta.stem
        data["path"] = str(clip)
        out.append(data)
    out.sort(key=lambda d: d.get("created", 0), reverse=True)
    return out


def voice_names() -> list[str]:
    """Display names for the dropdown, newest first."""
    return [v["name"] for v in list_voices()]


def get_voice(name: str) -> dict | None:
    name = (name or "").strip()
    for v in list_voices():
        if v["name"] == name:
            return v
    return None


def save_voice(name: str, ref_audio: str | None, ref_text: str = "") -> dict:
    """Persist the currently loaded reference clip + transcript under ``name``.

    Re-saving an existing name overwrites it (updates the clip/transcript). A new
    name that happens to slug the same as an existing one gets a numeric suffix so
    the two never share files.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("Give the voice a name before saving it.")
    if not ref_audio:
        raise ValueError("Upload or record a reference clip first, then save it.")
    src = Path(ref_audio)
    if not src.is_file():
        raise ValueError("The reference clip is no longer available — re-upload it.")

    vdir = storage.voices_dir()
    existing = {v["slug"]: v["name"] for v in list_voices()}
    slug = _slug(name)
    # Keep a stable slug when overwriting the same name; otherwise avoid colliding
    # with a different voice that slugs identically.
    if slug in existing and existing[slug] != name:
        i = 2
        while f"{slug}_{i}" in existing and existing[f"{slug}_{i}"] != name:
            i += 1
        slug = f"{slug}_{i}"

    ext = src.suffix.lower()
    if ext not in AUDIO_EXTS:
        ext = ".wav"
    # Drop any prior clip for this slug — its extension may differ from the new one.
    for old in vdir.glob(f"{slug}.*"):
        if old.suffix.lower() in AUDIO_EXTS:
            old.unlink()
    clip = vdir / f"{slug}{ext}"
    clip.write_bytes(src.read_bytes())

    data = {
        "name": name,
        "audio": clip.name,
        "ref_text": (ref_text or "").strip(),
        "digest": file_digest(str(clip)),
        "created": time.time(),
    }
    _manifest(slug).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    data["slug"] = slug
    data["path"] = str(clip)
    return data


def delete_voice(name: str) -> bool:
    """Remove a saved profile (clip + metadata). Returns False if it wasn't found."""
    v = get_voice(name)
    if not v:
        return False
    vdir = storage.voices_dir()
    _manifest(v["slug"]).unlink(missing_ok=True)
    for clip in vdir.glob(f"{v['slug']}.*"):
        clip.unlink(missing_ok=True)
    return True
