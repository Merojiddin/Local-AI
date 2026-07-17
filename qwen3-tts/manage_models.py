#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive model manager for the terminal.

Used by install.command (first-time preset selection) and
manage_models.command (install / remove / verify / storage later on).

  python manage_models.py                # interactive menu
  python manage_models.py --first-run    # preset selection only (installer)
"""

from __future__ import annotations

import sys

from modules import model_manager as mgr
from modules import storage


def show_status() -> None:
    print()
    print(f"{'Model':<32} {'Download':>10}  {'Status'}")
    print("-" * 78)
    for key, info in mgr.MODELS.items():
        p = mgr.installed_path(key)
        if info["kind"] == "ocr":
            status = "built into macOS"
        elif p is None:
            status = "not installed"
        else:
            status = f"installed ({mgr.actual_size_gb(key):.1f} GB) -> {p}"
        size = "0 GB" if info["kind"] == "ocr" else f"≈{info['size_gb']:.1f} GB"
        print(f"{info['name']:<32} {size:>10}  {status}")
    print("-" * 78)
    print(f"Models folder : {storage.models_dir()}")
    print(f"Free disk     : {storage.free_disk_gb():.1f} GB")


def _numbered_models() -> list[str]:
    keys = list(mgr.MODELS)
    print()
    for i, key in enumerate(keys, start=1):
        info = mgr.MODELS[key]
        mark = "✅" if mgr.is_installed(key) else "  "
        size = "built-in" if info["kind"] == "ocr" else f"{info['size_gb']:.1f} GB"
        print(f"  {i}. {mark} {info['name']:<32} {size:>9}  — {info['purpose']}")
    return keys


def _pick_models(prompt: str) -> list[str]:
    keys = _numbered_models()
    raw = input(f"\n{prompt} (numbers separated by spaces, or Enter to cancel): ").strip()
    if not raw:
        return []
    picked = []
    for token in raw.replace(",", " ").split():
        try:
            idx = int(token) - 1
            if 0 <= idx < len(keys):
                picked.append(keys[idx])
        except ValueError:
            continue
    return picked


def do_install(keys: list[str]) -> None:
    todo = [k for k in keys if not mgr.is_installed(k) and mgr.MODELS[k]["kind"] != "ocr"]
    skipped = [k for k in keys if k not in todo]
    for k in skipped:
        print(f"  {mgr.MODELS[k]['name']}: already available, skipping.")
    if not todo:
        print("Nothing to download.")
        return
    need = mgr.estimate_download_gb(todo)
    free = storage.free_disk_gb()
    print(f"\nEstimated download : {need:.1f} GB")
    print(f"Free disk space    : {free:.1f} GB")
    if free < need + 1.0:
        print("❌ Not enough free disk space — install fewer models or free up space.")
        return
    if input("Proceed? [y/N]: ").strip().lower() not in ("y", "yes"):
        print("Cancelled.")
        return
    for k in todo:
        info = mgr.MODELS[k]
        print(f"\n>>> {info['name']} ({info['size_gb']:.1f} GB)")
        try:
            target = mgr.install(k)
            print(f"    OK -> {target}  (actual size {mgr.actual_size_gb(k):.2f} GB)")
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {exc}")
    print(f"\nFree disk space now: {storage.free_disk_gb():.1f} GB")


def first_run() -> None:
    print("\nWhich models do you want to install now?")
    print("(You can always install or remove models later — Models tab or manage_models.command.)\n")
    presets = list(mgr.PRESETS)
    for i, name in enumerate(presets, start=1):
        keys = mgr.PRESETS[name]
        if name.startswith("6"):
            desc = "pick models yourself"
        else:
            total = sum(mgr.MODELS[k]["size_gb"] for k in keys)
            desc = f"{total:.1f} GB — " + ", ".join(mgr.MODELS[k]["name"] for k in keys)
        print(f"  {i}. {name:<24} {desc}")
    print("  0. Skip — install nothing now")

    raw = input("\nChoose a preset [0-6]: ").strip()
    if raw not in {"1", "2", "3", "4", "5", "6"}:
        print("Skipping model download. Install models later from the Models tab.")
        return
    name = presets[int(raw) - 1]
    keys = mgr.PRESETS[name] or _pick_models("Which models do you want to install?")
    if not keys:
        print("Nothing selected.")
        return
    do_install(keys)


def do_remove() -> None:
    keys = _pick_models("Which models do you want to REMOVE?")
    if not keys:
        return
    print("\nThis deletes ONLY the downloaded model weights.")
    print("Generated audio, documents and indexes are never touched.")
    for k in keys:
        info = mgr.MODELS[k]
        if input(f"Delete {info['name']}? [y/N]: ").strip().lower() in ("y", "yes"):
            print(" ", mgr.remove(k).replace("\n", "\n  "))
    print(f"\nFree disk space now: {storage.free_disk_gb():.1f} GB")


def do_verify() -> None:
    print()
    for key in mgr.MODELS:
        print(" ", mgr.verify(key))


def change_models_dir() -> None:
    print(f"\nCurrent models folder: {storage.models_dir()}")
    raw = input("New folder (e.g. /Volumes/SSD/LocalAIToolbox/models, Enter to keep): ").strip()
    if not raw:
        return
    settings = storage.load_settings()
    settings["models_dir"] = raw
    try:
        storage.save_settings(settings)
        print(f"Saved. Models folder is now: {storage.models_dir()}")
        print("Note: already-downloaded models stay where they are; move them manually")
        print("into the new folder, or install them again from the Models tab.")
    except OSError as exc:
        print(f"❌ Could not use that folder: {exc}")


def menu() -> None:
    while True:
        show_status()
        print(
            "\n  1. Install models"
            "\n  2. Remove models"
            "\n  3. Verify model files"
            "\n  4. Change the shared model folder"
            "\n  0. Quit"
        )
        choice = input("\nChoose: ").strip()
        if choice == "1":
            keys = _pick_models("Which models do you want to install?")
            if keys:
                do_install(keys)
        elif choice == "2":
            do_remove()
        elif choice == "3":
            do_verify()
        elif choice == "4":
            change_models_dir()
        else:
            return


if __name__ == "__main__":
    try:
        if "--first-run" in sys.argv:
            first_run()
        else:
            menu()
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
