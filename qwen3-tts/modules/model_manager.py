"""Model registry (models.json), install / remove / verify, and the Models tab.

Weights are looked up in this order, so nothing is ever downloaded twice:
  1. the shared models directory from Settings
  2. legacy project locations (models/Qwen3-TTS-*) used by earlier versions
  3. the Hugging Face cache (~/.cache/huggingface/hub)
Downloads always go to the shared models directory.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

from . import storage

REGISTRY_FILE = storage.BASE_DIR / "models.json"


def _load_registry() -> dict:
    try:
        with open(REGISTRY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot read {REGISTRY_FILE}. Restore it from the repository."
        ) from exc


_REG = _load_registry()
MODELS: dict[str, dict] = _REG["models"]
PRESETS: dict[str, list[str]] = _REG["presets"]
PRESETS["6. Custom"] = []


def hf_cache_path(repo: str) -> Path | None:
    """Return a usable snapshot inside the HF cache, if one exists."""
    if not repo:
        return None
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    snap_root = cache / ("models--" + repo.replace("/", "--")) / "snapshots"
    if snap_root.is_dir():
        for snap in sorted(snap_root.iterdir(), reverse=True):
            if (snap / "config.json").exists():
                return snap
    return None


def installed_path(key: str) -> Path | None:
    """Where this model's weights live, or None if not installed."""
    info = MODELS[key]
    if info["kind"] == "ocr":  # built into macOS
        return Path("/System/Library/Frameworks/Vision.framework")
    if info["folder"]:
        p = storage.models_dir() / info["folder"]
        if (p / "config.json").exists():
            return p
    for rel in info.get("legacy_dirs", []):
        p = storage.BASE_DIR / rel
        if (p / "config.json").exists():
            return p
    if info.get("convert_from"):
        # The HF cache can only ever hold the *unconverted* source weights
        # (full precision — unusable on 16 GB); never treat those as installed.
        return None
    return hf_cache_path(info["repo"])


def is_installed(key: str) -> bool:
    return installed_path(key) is not None


def model_path_or_error(key: str) -> str:
    p = installed_path(key)
    if p is None:
        raise RuntimeError(
            f"{MODELS[key]['name']} is not installed. "
            "Open the Models tab (or run manage_models.command) to install it."
        )
    return str(p)


def actual_size_gb(key: str) -> float:
    p = installed_path(key)
    if p is None or MODELS[key]["kind"] == "ocr":
        return 0.0
    return storage.dir_size_gb(p)


def estimate_download_gb(keys: list[str]) -> float:
    return sum(
        MODELS[k].get("download_gb", MODELS[k]["size_gb"])
        for k in keys if not is_installed(k)
    )


def install(key: str, progress_cb=None) -> Path:
    """Download one model into the shared models directory.

    progress_cb, if given, is called with (fraction, text) roughly once a second.
    """
    from huggingface_hub import snapshot_download

    info = MODELS[key]
    if info["kind"] == "ocr":
        return installed_path(key)
    existing = installed_path(key)
    if existing is not None:
        return existing
    if info.get("convert_from"):
        return _install_convert(key, info, progress_cb)

    need = info["size_gb"]
    free = storage.free_disk_gb(storage.models_dir())
    if free < need + 1.0:
        raise RuntimeError(
            f"Not enough disk space for {info['name']}: needs about {need:.1f} GB, "
            f"only {free:.1f} GB free."
        )

    target = storage.models_dir() / info["folder"]
    holder: dict = {}

    def _download():
        try:
            snapshot_download(repo_id=info["repo"], local_dir=str(target))
        except Exception as exc:  # noqa: BLE001 - re-raised on the main thread
            holder["error"] = exc

    thread = threading.Thread(target=_download, daemon=True)
    thread.start()
    while thread.is_alive():
        if progress_cb:
            done = storage.dir_size_gb(target)
            frac = min(0.99, done / need) if need else 0.5
            progress_cb(frac, f"{info['name']}: {done:.2f} / {need:.1f} GB")
        time.sleep(1.0)
    thread.join()
    if "error" in holder:
        raise RuntimeError(f"Download of {info['name']} failed: {holder['error']}")
    if not (target / "config.json").exists():
        raise RuntimeError(f"Download of {info['name']} finished but config.json is missing.")
    if progress_cb:
        progress_cb(1.0, f"{info['name']}: done")
    return target


def _install_convert(key: str, info: dict, progress_cb=None) -> Path:
    """Download full-precision source weights and convert them to 4-bit locally.

    Used for models that have no ready-made MLX 4-bit build on Hugging Face
    (Kiwi-1 8B, Qwen3-Reranker 4B). The source lands in the HF cache on the
    home disk and is deleted after a successful conversion.
    """
    from huggingface_hub import snapshot_download

    from . import memory_manager as mm

    src_repo = info["convert_from"]
    download_gb = info.get("download_gb", info["size_gb"])
    home_free = storage.free_disk_gb(Path.home())
    models_free = storage.free_disk_gb(storage.models_dir())
    if home_free < download_gb + 1.0 or models_free < info["size_gb"] + 1.0:
        raise RuntimeError(
            f"Not enough disk space for {info['name']}: the download needs about "
            f"{download_gb:.1f} GB (home disk: {home_free:.1f} GB free) and the "
            f"converted model about {info['size_gb']:.1f} GB "
            f"(models folder: {models_free:.1f} GB free)."
        )

    target = storage.models_dir() / info["folder"]
    if target.exists():
        shutil.rmtree(target)  # leftovers from an interrupted conversion

    cache_root = (Path.home() / ".cache" / "huggingface" / "hub"
                  / ("models--" + src_repo.replace("/", "--")))
    holder: dict = {}

    def _download():
        try:
            snapshot_download(repo_id=src_repo)
        except Exception as exc:  # noqa: BLE001 - re-raised on the main thread
            holder["error"] = exc

    thread = threading.Thread(target=_download, daemon=True)
    thread.start()
    while thread.is_alive():
        if progress_cb:
            done = storage.dir_size_gb(cache_root)
            frac = min(0.79, done / download_gb * 0.8) if download_gb else 0.4
            progress_cb(frac, f"{info['name']}: downloading {done:.2f} / {download_gb:.1f} GB")
        time.sleep(1.0)
    thread.join()
    if "error" in holder:
        raise RuntimeError(f"Download of {info['name']} failed: {holder['error']}")

    # Conversion materialises the weights shard by shard — free the RAM first.
    mm.unload_all()
    if progress_cb:
        progress_cb(0.8, f"{info['name']}: converting to 4-bit (this takes a few minutes)…")
    from mlx_lm import convert as mlx_convert
    try:
        mlx_convert(hf_path=src_repo, mlx_path=str(target), quantize=True, q_bits=4)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(target, ignore_errors=True)
        raise RuntimeError(f"4-bit conversion of {info['name']} failed: {exc}")
    if not (target / "config.json").exists():
        raise RuntimeError(f"Conversion of {info['name']} finished but config.json is missing.")

    shutil.rmtree(cache_root, ignore_errors=True)  # reclaim the full-precision source
    if progress_cb:
        progress_cb(1.0, f"{info['name']}: done")
    return target


def _assert_model_dir(path: Path) -> None:
    """Refuse to delete anything that does not look like a model folder."""
    if not (path / "config.json").exists():
        raise RuntimeError(f"Refusing to delete {path}: no config.json found.")
    for user_dir in (storage.outputs_dir(), storage.indexes_dir()):
        if user_dir == path or user_dir in path.parents or path in user_dir.parents:
            raise RuntimeError(f"Refusing to delete {path}: overlaps user data.")


def remove(key: str) -> str:
    """Delete only this model's downloaded weights. Never touches user data."""
    info = MODELS[key]
    if info["kind"] == "ocr":
        return "OCR is built into macOS — nothing to remove."
    removed = []
    for candidate in (
        storage.models_dir() / info["folder"] if info["folder"] else None,
        *[storage.BASE_DIR / rel for rel in info.get("legacy_dirs", [])],
    ):
        if candidate and (candidate / "config.json").exists():
            _assert_model_dir(candidate)
            shutil.rmtree(candidate)
            removed.append(str(candidate))
    cache_snap = hf_cache_path(info["repo"])
    if cache_snap is not None:
        repo_root = cache_snap.parent.parent  # models--org--name
        shutil.rmtree(repo_root, ignore_errors=True)
        removed.append(str(repo_root))
    if not removed:
        return f"{info['name']} is not installed — nothing removed."
    return f"Removed {info['name']}:\n" + "\n".join(f"  {p}" for p in removed)


def verify(key: str) -> str:
    info = MODELS[key]
    if info["kind"] == "ocr":
        try:
            import Vision  # noqa: F401
            return "✅ OCR: Apple Vision framework is available."
        except ImportError:
            return "❌ OCR: pyobjc-framework-Vision is missing. Run install.command again."
    p = installed_path(key)
    if p is None:
        return f"❌ {info['name']}: not installed."
    problems = []
    try:
        json.loads((p / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        problems.append(f"config.json unreadable ({exc})")
    weights = list(p.glob("*.safetensors")) + list(p.glob("**/*.safetensors"))
    if not weights:
        problems.append("no .safetensors weight files found")
    else:
        from safetensors import safe_open
        for w in sorted(set(weights)):
            try:
                with safe_open(str(w), framework="numpy") as f:
                    next(iter(f.keys()), None)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{w.name} is corrupt ({exc})")
    if problems:
        return f"❌ {info['name']}: " + "; ".join(problems) + ". Remove and reinstall it."
    return f"✅ {info['name']}: OK ({actual_size_gb(key):.2f} GB at {p})"


def status_rows() -> list[list[str]]:
    rows = []
    for key, info in MODELS.items():
        p = installed_path(key)
        if info["kind"] == "ocr":
            status, where = "built-in", "macOS"
        elif p is None:
            status, where = "not installed", "—"
        else:
            status, where = f"installed ({actual_size_gb(key):.1f} GB)", str(p)
        size = "0 (built-in)" if info["kind"] == "ocr" else f"≈{info['size_gb']:.1f} GB"
        rows.append([info["name"], info["repo"] or "macOS Vision framework",
                     info["purpose"], size, status, where])
    return rows


# --------------------------------------------------------------------------- #
# Models tab
# --------------------------------------------------------------------------- #
def _choice_label(key: str) -> str:
    info = MODELS[key]
    mark = "✅ " if is_installed(key) else ""
    size = "built-in" if info["kind"] == "ocr" else f"{info['size_gb']:.1f} GB"
    return f"{mark}{info['name']} ({size})"


def _choices() -> list[tuple[str, str]]:
    return [(_choice_label(k), k) for k in MODELS]


def _disk_line() -> str:
    return (
        f"**Models folder:** `{storage.models_dir()}` · "
        f"**Free disk:** {storage.free_disk_gb():.1f} GB"
    )


def _estimate_line(selected: list[str]) -> str:
    selected = selected or []
    dl = estimate_download_gb(selected)
    already = [k for k in selected if is_installed(k)]
    free = storage.free_disk_gb()
    msg = f"**Estimated download:** {dl:.1f} GB · **Free disk:** {free:.1f} GB"
    if already:
        msg += f" · already installed, will be skipped: {len(already)}"
    if dl and free < dl + 1.0:
        msg += "\n\n⚠️ **Not enough free disk space for this selection.**"
    return msg


def build_models_tab():
    import gradio as gr

    gr.Markdown(
        "Model weights are separate from your generated files and document indexes — "
        "removing a model never deletes user data.",
        elem_classes="hint-text",
    )
    disk_info = gr.Markdown(_disk_line(), elem_classes="result-info")
    table = gr.Dataframe(
        headers=["Model", "ID", "Purpose", "Download", "Status", "Location"],
        value=status_rows(), interactive=False, wrap=True, max_height=260,
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("**Install**", elem_classes="section-head")
                preset = gr.Dropdown(choices=list(PRESETS), value="6. Custom", label="Preset")
                selection = gr.CheckboxGroup(choices=_choices(), label="Models to install")
                estimate = gr.Markdown(_estimate_line([]), elem_classes="result-info")
                install_btn = gr.Button("⬇️ Install selected", variant="primary")
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("**Manage one model**", elem_classes="section-head")
                manage_dd = gr.Dropdown(choices=_choices(), label="Model")
                with gr.Row():
                    verify_btn = gr.Button("Verify", size="sm")
                    open_btn = gr.Button("Open folder", size="sm")
                confirm_remove = gr.Checkbox(label="Yes, delete this model's weights", value=False)
                remove_btn = gr.Button("🗑 Remove model", size="sm", variant="stop")
            manage_info = gr.Markdown(elem_classes="result-info")

    def refresh():
        return (
            gr.update(value=_disk_line()),
            gr.update(value=status_rows()),
            gr.update(choices=_choices()),
            gr.update(choices=_choices()),
        )

    def on_preset(name):
        keys = PRESETS.get(name, [])
        return gr.update(value=keys), _estimate_line(keys)

    preset.change(on_preset, inputs=[preset], outputs=[selection, estimate])
    selection.change(lambda sel: _estimate_line(sel), inputs=[selection], outputs=[estimate])

    def ui_install(selected, progress=gr.Progress()):
        from . import memory_manager as mm
        if not selected:
            raise gr.Error("Select at least one model (or pick a preset) first.")
        todo = [k for k in selected if not is_installed(k)]
        if not todo:
            gr.Info("Everything selected is already installed.")
            return (*refresh(), gr.update(value=_estimate_line(selected)))
        need = estimate_download_gb(todo)
        if storage.free_disk_gb() < need + 1.0:
            raise gr.Error(
                f"Not enough disk space: the selection needs about {need:.1f} GB "
                f"but only {storage.free_disk_gb():.1f} GB is free."
            )
        errors = []
        for i, key in enumerate(todo):
            mm.set_task(f"Downloading {MODELS[key]['name']}…")
            try:
                install(key, progress_cb=lambda f, t, i=i: progress((i + f) / len(todo), desc=t))
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
            finally:
                mm.set_task("idle")
        if errors:
            raise gr.Error("Some downloads failed:\n" + "\n".join(errors))
        gr.Info(f"Installed {len(todo)} model(s).")
        return (*refresh(), gr.update(value=_estimate_line(selected)))

    install_btn.click(ui_install, inputs=[selection],
                      outputs=[disk_info, table, selection, manage_dd, estimate])

    def ui_verify(key):
        if not key:
            raise gr.Error("Pick a model first.")
        return verify(key)

    verify_btn.click(ui_verify, inputs=[manage_dd], outputs=[manage_info])

    def ui_open(key):
        if not key:
            raise gr.Error("Pick a model first.")
        p = installed_path(key)
        if p is None or MODELS[key]["kind"] == "ocr":
            storage.open_in_finder(storage.models_dir())
        else:
            storage.open_in_finder(p if p.is_dir() else p.parent)

    open_btn.click(ui_open, inputs=[manage_dd])

    def ui_remove(key, confirmed):
        from . import memory_manager as mm
        from . import model_select
        if not key:
            raise gr.Error("Pick a model first.")
        if not confirmed:
            raise gr.Error("Tick the confirmation box first — removal deletes the downloaded weights.")
        mm.unload_all()  # never delete files that are memory-mapped by a loaded model
        msg = remove(key)
        # Any category that had selected this model falls back to its default.
        model_select.on_model_removed(key)
        return (msg + f"\n\n**Free disk now:** {storage.free_disk_gb():.1f} GB",
                gr.update(value=False), *refresh())

    remove_btn.click(ui_remove, inputs=[manage_dd, confirm_remove],
                     outputs=[manage_info, confirm_remove, disk_info, table, selection, manage_dd])
