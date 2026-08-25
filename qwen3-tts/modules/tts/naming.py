"""Output filenames and the deterministic generation cache keys."""

from __future__ import annotations

import filecmp
import hashlib
import json
import re
from pathlib import Path

from .. import storage


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
