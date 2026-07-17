"""One-heavy-model-at-a-time memory management for a 16 GB Mac.

Heavy models (TTS, chat, vision, whisper) share a single slot: loading one
always releases the previous one first (delete refs -> gc -> clear the MLX
Metal cache). Small models (embedding, reranker) live in separate light
slots and may stay resident alongside one heavy model.
"""

from __future__ import annotations

import gc
import threading
import time

import psutil

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - MLX is a hard requirement at runtime
    mx = None

_LOCK = threading.RLock()
_TASK = "idle"


def set_task(text: str) -> None:
    global _TASK
    with _LOCK:
        _TASK = text or "idle"


def get_task() -> str:
    with _LOCK:
        return _TASK


def _clear_mlx_cache() -> None:
    if mx is not None:
        try:
            mx.clear_cache()
        except Exception:
            pass


def ram_status() -> tuple[float, float, float]:
    """(used_gb, total_gb, available_gb)."""
    vm = psutil.virtual_memory()
    return (vm.total - vm.available) / 1e9, vm.total / 1e9, vm.available / 1e9


class Slot:
    """Holds one loaded model (+ optional tokenizer/processor)."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.key: str | None = None
        self.label: str | None = None
        self.obj = None
        self.last_used = 0.0
        self._lock = threading.RLock()

    def get(self, key: str, label: str, loader):
        """Return the cached object for `key`, loading it if needed."""
        with self._lock:
            if self.key == key and self.obj is not None:
                self.last_used = time.time()
                return self.obj
            self.release()
            set_task(f"Loading {label}…")
            try:
                self.obj = loader()
            except Exception:
                set_task("idle")
                self.release()
                raise
            self.key = key
            self.label = label
            self.last_used = time.time()
            set_task("idle")
            return self.obj

    def touch(self) -> None:
        self.last_used = time.time()

    def release(self) -> None:
        with self._lock:
            if self.obj is not None:
                self.obj = None
            self.key = None
            self.label = None
            gc.collect()
            _clear_mlx_cache()


HEAVY = Slot("heavy")           # TTS / chat / vision / whisper — only ever one
LIGHT: dict[str, Slot] = {      # may stay resident next to one heavy model
    "embed": Slot("embed"),
    "rerank": Slot("rerank"),
}


def unload_all() -> None:
    HEAVY.release()
    for slot in LIGHT.values():
        slot.release()
    set_task("idle")


def loaded_summary() -> str:
    parts = []
    if HEAVY.label:
        parts.append(HEAVY.label)
    light = [s.label for s in LIGHT.values() if s.label]
    if light:
        parts.append("+ " + ", ".join(light))
    return " ".join(parts) if parts else "none"


def memory_warning(threshold_gb: float, need_gb: float = 0.0) -> str | None:
    """Return a warning string when memory pressure is high, else None."""
    used, total, avail = ram_status()
    if need_gb and avail < need_gb + 1.0:
        return (
            f"⚠️ Low memory: {avail:.1f} GB available but about {need_gb:.1f} GB is "
            "needed. Close other apps or unload the current model first."
        )
    if used > threshold_gb:
        return f"⚠️ High memory use: {used:.1f} / {total:.0f} GB."
    return None


def auto_unload_check(enabled: bool, idle_minutes: float) -> None:
    """Called periodically by the UI timer: release idle slots."""
    if not enabled:
        return
    cutoff = time.time() - idle_minutes * 60
    for slot in (HEAVY, *LIGHT.values()):
        if slot.obj is not None and slot.last_used < cutoff:
            slot.release()
