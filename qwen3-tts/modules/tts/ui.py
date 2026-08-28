"""Gradio UI for the TTS tab: single + batch callbacks and the tab builder."""

from __future__ import annotations

import csv
import zipfile
from datetime import datetime

import gradio as gr

from .. import model_select as ms
from .. import storage

from .config import (
    DEFAULT_FORMAT,
    DEFAULT_MODE,
    DEFAULT_MODEL,
    DEFAULT_QUALITY,
    DEFAULT_VOICE,
    FFMPEG,
    FISH_LABEL,
    FORMATS,
    MODEL_SHORT,
    MODELS,
    QUALITIES,
    STYLE_EXAMPLES,
    SUPPORTS_INSTRUCT,
    VOICES,
    active_repo,
    clone_active,
    is_fish,
    needs_load,
)
from .i18n import DEFAULT_LANG, mode_choices, model_choices, tr
from .engine import generate_one, parse_batch, to_friendly_error
from .library import lib_html, ui_files_refresh, ui_lib_event
from . import voices as vx


# --------------------------------------------------------------------------- #
# UI callbacks (single + batch)
# --------------------------------------------------------------------------- #
def ui_generate(
    text, model_label, voice, mode, speed, style,
    p_before, p_after, p_between, repeat, out_format, quality,
    ref_audio=None, ref_text="",
    lang=DEFAULT_LANG,
):
    try:
        repo = active_repo(model_label) if (model_label in MODELS or is_fish(model_label)) else None
        if repo and needs_load(repo):
            gr.Info(tr(lang, "loading").format(model=model_label))
        res = generate_one(
            text, model_label, voice, mode, speed, style,
            p_before, p_after, p_between, repeat, out_format, quality,
            ref_audio=ref_audio, ref_text=ref_text,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        raise gr.Error(to_friendly_error(exc))

    path = str(res["path"])
    status = tr(lang, "res_cached") if res["cached"] else tr(lang, "res_new")
    gen_time = "0.0s" if res["cached"] else f"{res['gen_time']:.1f}s"
    info = (
        f"{status} · **{MODEL_SHORT[res['repo']]}** · {res['voice']} · {gen_time}\n\n"
        f"**{tr(lang, 'res_file')}:** `{path}`"
    )
    return path, path, info


def ui_batch(
    text_lines, csv_file, voice, model_label, mode, speed, style,
    p_before, p_after, p_between, repeat, out_format, quality,
    ref_audio=None, ref_text="",
    lang=DEFAULT_LANG,
    progress=gr.Progress(),
):
    try:
        defaults = dict(voice=voice, model_label=model_label, mode=mode, speed=float(speed))
        rows = parse_batch(text_lines, csv_file, defaults)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(to_friendly_error(exc))

    total = len(rows)
    # Group by model to minimise reloads, but report results in original order.
    order = sorted(range(total), key=lambda i: rows[i]["model_label"])
    results: dict[int, dict] = {}
    produced_paths: list[Path] = []

    for count, i in enumerate(order):
        r = rows[i]
        progress(count / total, desc=tr(lang, "item_of").format(n=count + 1, total=total))
        try:
            res = generate_one(
                r["text"], r["model_label"], r["voice"], r["mode"], r["speed"],
                style, p_before, p_after, p_between, repeat, out_format, quality,
                ref_audio=ref_audio, ref_text=ref_text,
            )
            results[i] = {
                "text": r["text"],
                "filename": res["path"].name,
                "path": str(res["path"]),
                "status": "cached" if res["cached"] else "generated",
                "error": "",
            }
            produced_paths.append(res["path"])
        except Exception as exc:  # noqa: BLE001 - continue on failure
            results[i] = {
                "text": r["text"],
                "filename": "",
                "path": "",
                "status": "failed",
                "error": str(exc),
            }

    progress(1.0, desc=tr(lang, "packaging"))
    ordered = [results[i] for i in range(total)]

    outputs = storage.outputs_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = outputs / f"batch_results_{ts}.csv"
    with open(results_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "filename", "path", "status", "error"])
        writer.writeheader()
        writer.writerows(ordered)

    zip_path = outputs / f"batch_{ts}.zip"
    seen: set[str] = set()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in produced_paths:
            if p.exists() and str(p) not in seen:
                zf.write(p, arcname=p.name)
                seen.add(str(p))
        zf.write(results_csv, arcname=results_csv.name)

    ok = sum(1 for r in ordered if r["status"] in ("generated", "cached"))
    failed = sum(1 for r in ordered if r["status"] == "failed")
    summary = (
        tr(lang, "batch_done").format(ok=ok, failed=failed, total=total)
        + f"\n\n**ZIP:** `{zip_path}`\n\n**CSV:** `{results_csv}`"
    )
    table = [[r["text"], r["filename"], r["status"], r["error"]] for r in ordered]
    return table, str(zip_path), summary


# --------------------------------------------------------------------------- #
# Saved voice-clone profiles (persist the loaded reference clip + transcript)
# --------------------------------------------------------------------------- #
def ui_save_voice(name, ref_audio, ref_text, lang=DEFAULT_LANG):
    """Persist the currently loaded clip under a name and refresh the picker."""
    try:
        v = vx.save_voice(name, ref_audio, ref_text)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        raise gr.Error(to_friendly_error(exc))
    return (
        gr.update(choices=vx.voice_names(), value=v["name"]),  # saved_voice
        gr.update(value=""),                                   # voice_name (clear)
        tr(lang, "saved_ok").format(name=v["name"]),           # saved_status
    )


def ui_load_voice(name, lang=DEFAULT_LANG):
    """Load a saved profile back into the reference-clip + transcript boxes."""
    v = vx.get_voice(name)
    if not v:
        return gr.update(), gr.update(), tr(lang, "pick_first")
    return (
        gr.update(value=v["path"]),                # ref_audio
        gr.update(value=v.get("ref_text", "")),    # ref_text
        tr(lang, "loaded_ok").format(name=v["name"]),
    )


def ui_delete_voice(name, lang=DEFAULT_LANG):
    """Delete a saved profile and refresh the picker."""
    if not name:
        return gr.update(), tr(lang, "pick_first")
    vx.delete_voice(name)
    return (
        gr.update(choices=vx.voice_names(), value=None),  # saved_voice
        tr(lang, "deleted_ok").format(name=name),         # saved_status
    )


# --------------------------------------------------------------------------- #
# UI — built inside the host app's "TTS" tab. Returns the i18n machinery so
# the host can wire the global language switch.
# --------------------------------------------------------------------------- #
def build_tts_tab(lang_component, settings: dict):
    L0 = DEFAULT_LANG
    default_model = settings.get("default_tts_model", DEFAULT_MODEL)
    if default_model not in MODELS:
        default_model = DEFAULT_MODEL
    show_adv = bool(settings.get("show_advanced", True))

    if not FFMPEG:
        gr.Markdown(
            "> ⚠️ **FFmpeg was not found.** Audio output needs it. "
            "Install with `brew install ffmpeg`, then restart the app."
        )
    if not SUPPORTS_INSTRUCT:
        gr.Markdown(
            "> ℹ️ The installed MLX-Audio does not support style instructions; "
            "the style box is disabled."
        )

    with gr.Tab(tr(L0, "tab_single")) as tab_single:
        # Three balanced columns — Text · Pronunciation · Voice & Model — so the
        # whole workspace is visible at once instead of stacking into one tall
        # scroll. The action bar (generate + preview) and the library sit below,
        # full width.
        with gr.Row(equal_height=True, elem_id="tts-main"):
            # ---- Column 1: Text ----
            with gr.Column(scale=5, min_width=280, elem_id="tts-col-text"):
                with gr.Group(elem_classes="tts-panel"):
                    sec_text = gr.Markdown(tr(L0, "sec_text"), elem_classes="section-head")
                    text = gr.Textbox(
                        label=tr(L0, "text_label"),
                        placeholder=tr(L0, "text_ph"),
                        lines=15, max_lines=24,
                        elem_id="tts-text",
                    )

            # ---- Column 2: Pronunciation ----
            with gr.Column(scale=4, min_width=280):
                with gr.Group(elem_classes="tts-panel"):
                    sec_pron = gr.Markdown(tr(L0, "sec_pron"), elem_classes="section-head")
                    mode = gr.Radio(choices=mode_choices(L0), value=DEFAULT_MODE, label=tr(L0, "mode"))
                    style = gr.Textbox(
                        label=tr(L0, "style"),
                        placeholder=tr(L0, "style_ph"),
                        lines=2,
                        interactive=SUPPORTS_INSTRUCT,
                    )
                    with gr.Accordion(tr(L0, "acc_examples"), open=False, visible=show_adv) as acc_examples:
                        gr.Examples(examples=[[s] for s in STYLE_EXAMPLES], inputs=[style], label="")
                    with gr.Row():
                        speed = gr.Slider(
                            minimum=0.5, maximum=1.5, value=1.0, step=0.05, label=tr(L0, "speed")
                        )
                        repeat = gr.Slider(1, 5, value=1, step=1, label=tr(L0, "repeat"))
                    with gr.Row():
                        p_before = gr.Slider(0, 2, value=0.0, step=0.05, label=tr(L0, "p_before"))
                        p_after = gr.Slider(0, 2, value=0.0, step=0.05, label=tr(L0, "p_after"))
                        p_between = gr.Slider(0, 2, value=0.3, step=0.05, label=tr(L0, "p_between"))

            # ---- Column 3: Voice & Model (+ Output) ----
            with gr.Column(scale=4, min_width=280):
                with gr.Group(elem_classes="tts-panel"):
                    sec_voice = gr.Markdown(tr(L0, "sec_voice"), elem_classes="section-head")
                    model_label = gr.Radio(
                        choices=model_choices(L0), value=default_model, label=tr(L0, "model")
                    )
                    ms.build_selector("tts_voice_mode")
                    is_clone0 = clone_active(default_model)
                    voice = gr.Radio(
                        choices=VOICES, value=DEFAULT_VOICE, label=tr(L0, "voice"),
                        visible=not is_clone0,
                    )
                    ref_audio = gr.Audio(
                        label=tr(L0, "ref_audio"), type="filepath",
                        sources=["upload", "microphone"], visible=is_clone0,
                        elem_id="tts-ref-audio",
                    )
                    ref_text = gr.Textbox(
                        label=tr(L0, "ref_text"), placeholder=tr(L0, "ref_text_ph"),
                        lines=2, visible=is_clone0,
                    )
                    clone_note = gr.Markdown(
                        tr(L0, "clone_note"), visible=is_clone0, elem_classes="hint-text"
                    )

                    # Persist a loaded clone so it survives the session: save the
                    # uploaded clip + transcript under a name, then reload it from
                    # the dropdown next time instead of re-uploading. The whole
                    # group toggles with the clone controls above.
                    with gr.Group(visible=is_clone0) as saved_voice_group:
                        saved_head = gr.Markdown(
                            tr(L0, "saved_head"), elem_classes="section-head"
                        )
                        with gr.Row():
                            saved_voice = gr.Dropdown(
                                choices=vx.voice_names(), value=None,
                                label=tr(L0, "saved_pick"), scale=4, min_width=140,
                                filterable=False,
                            )
                            load_voice_btn = gr.Button(
                                tr(L0, "load_voice"), size="sm", scale=0, min_width=90,
                            )
                            delete_voice_btn = gr.Button(
                                "🗑", size="sm", scale=0, min_width=44,
                                elem_classes="icon-btn",
                            )
                        with gr.Row():
                            voice_name = gr.Textbox(
                                label=tr(L0, "save_as"), placeholder=tr(L0, "save_as_ph"),
                                scale=4, min_width=140, max_lines=1,
                            )
                            save_voice_btn = gr.Button(
                                tr(L0, "save_voice"), size="sm", scale=0, min_width=90,
                            )
                        saved_status = gr.Markdown(
                            tr(L0, "saved_hint"), elem_classes="hint-text"
                        )

                    def _persist_model(label: str) -> None:
                        if label in MODELS:
                            s = storage.load_settings()
                            s["default_tts_model"] = label
                            storage.save_settings(s)

                    def _toggle_clone(label: str):
                        """Swap the visible controls: named-voice picker vs the
                        reference-clip uploader. The uploader shows for Fish (always
                        clones) and for a Qwen size when Voice mode is "Clone a
                        voice" — see clone_active()."""
                        clone = clone_active(label)
                        return (
                            gr.update(visible=not clone),  # voice
                            gr.update(visible=clone),      # ref_audio
                            gr.update(visible=clone),      # ref_text
                            gr.update(visible=clone),      # clone_note
                            gr.update(visible=clone),      # saved_voice_group
                        )

                    model_label.change(_persist_model, inputs=[model_label],
                                       show_progress="hidden")
                    model_label.change(
                        _toggle_clone, inputs=[model_label],
                        outputs=[voice, ref_audio, ref_text, clone_note, saved_voice_group],
                        show_progress="hidden",
                    )
                    # The Voice-mode card ("Built-in voices" / "Clone a voice") is
                    # saved through the shared selector bridge, not an ordinary
                    # Gradio input, so react to it there: re-run the same toggle so
                    # flipping to "Clone a voice" reveals the uploader for Qwen too.
                    ms.add_bridge_reactor(
                        inputs=[model_label],
                        outputs=[voice, ref_audio, ref_text, clone_note, saved_voice_group],
                        fn=lambda data, label: (
                            _toggle_clone(label)
                            if data.get("cat") == "tts_voice_mode"
                            else (gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
                        ),
                    )

                    save_voice_btn.click(
                        ui_save_voice,
                        inputs=[voice_name, ref_audio, ref_text, lang_component],
                        outputs=[saved_voice, voice_name, saved_status],
                        show_progress="hidden",
                    )
                    load_voice_btn.click(
                        ui_load_voice,
                        inputs=[saved_voice, lang_component],
                        outputs=[ref_audio, ref_text, saved_status],
                        show_progress="hidden",
                    )
                    delete_voice_btn.click(
                        ui_delete_voice,
                        inputs=[saved_voice, lang_component],
                        outputs=[saved_voice, saved_status],
                        show_progress="hidden",
                    )

        # ---- Action bar: generate · output · preview · download — full width.
        # Output format rides here rather than in the Voice column so the three
        # columns stay short enough to clear a laptop viewport.
        with gr.Row(equal_height=True, elem_id="tts-actions"):
            generate_btn = gr.Button(
                tr(L0, "generate"), variant="primary", scale=0,
                min_width=200, elem_id="tts-generate",
            )
            with gr.Column(scale=0, min_width=530, elem_id="tts-output-inline"):
                sec_out = gr.Markdown(tr(L0, "sec_out"), elem_classes="section-head")
                with gr.Row(elem_id="tts-output-row"):
                    # Labels hidden — MP3/WAV and the kbps values are self-evident,
                    # and dropping them keeps the whole card to a single pill row.
                    out_format = gr.Radio(
                        choices=FORMATS, value=DEFAULT_FORMAT, label=tr(L0, "format"),
                        show_label=False, scale=2, min_width=100,
                    )
                    quality = gr.Radio(
                        choices=QUALITIES, value=DEFAULT_QUALITY, label=tr(L0, "quality"),
                        show_label=False, scale=3, min_width=230,
                    )
            audio_out = gr.Audio(
                label=tr(L0, "audio"), type="filepath", elem_id="tts-audio", scale=4
            )
            with gr.Column(scale=2, min_width=200, elem_id="tts-result-col"):
                file_out = gr.DownloadButton(tr(L0, "download"), size="sm")
                info_out = gr.Markdown(elem_classes="result-info")

        # Library spans the full width below so long Chinese filenames have room
        # to lay out instead of stacking one glyph per line.
        with gr.Group():
            with gr.Row(elem_id="lib-head"):
                sec_files = gr.Markdown(
                    f"📁 **{tr(L0, 'tab_files')}**", elem_classes="section-head"
                )
                files_refresh_btn = gr.Button(
                    "🔄", size="sm", scale=0, min_width=40, elem_classes="icon-btn"
                )
                files_open_btn = gr.Button(
                    "📂", size="sm", scale=0, min_width=40, elem_classes="icon-btn"
                )
            files_list = gr.HTML(lib_html())
            file_text = gr.Textbox(
                label=tr(L0, "file_text"), lines=6, max_lines=12,
                visible=False, show_copy_button=True,
            )
            lib_evt = gr.Textbox(visible=False, elem_id="lib-evt")

        generate_btn.click(
            fn=ui_generate,
            inputs=[text, model_label, voice, mode, speed, style,
                    p_before, p_after, p_between, repeat, out_format, quality,
                    ref_audio, ref_text, lang_component],
            outputs=[audio_out, file_out, info_out],
        ).then(ui_files_refresh, outputs=[files_list], show_progress="hidden")

        files_refresh_btn.click(ui_files_refresh, outputs=[files_list], show_progress="hidden")
        files_open_btn.click(lambda: storage.open_in_finder(storage.outputs_dir()))
        lib_evt.input(ui_lib_event, inputs=[lib_evt], outputs=[file_text], show_progress="hidden")
        tab_single.select(ui_files_refresh, outputs=[files_list], show_progress="hidden")

    with gr.Tab(tr(L0, "tab_batch")) as tab_batch:
        batch_desc = gr.Markdown(tr(L0, "batch_desc"))
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                batch_text = gr.Textbox(
                    label=tr(L0, "batch_text"), lines=9, placeholder="你好\n谢谢\n再见"
                )
                batch_csv = gr.File(
                    label=tr(L0, "batch_csv"), file_types=[".csv"], type="filepath", height=120
                )
            with gr.Column(scale=6):
                with gr.Group():
                    b_defaults = gr.Markdown(tr(L0, "b_defaults"), elem_classes="section-head")
                    b_model = gr.Radio(choices=model_choices(L0), value=default_model, label=tr(L0, "model"))
                    with gr.Row():
                        b_voice = gr.Radio(choices=VOICES, value=DEFAULT_VOICE, label=tr(L0, "voice"))
                        b_speed = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label=tr(L0, "speed"))
                    b_mode = gr.Radio(choices=mode_choices(L0), value=DEFAULT_MODE, label=tr(L0, "mode"))
                    b_style = gr.Textbox(label=tr(L0, "b_style"), lines=2, interactive=SUPPORTS_INSTRUCT)
                    b_ref_audio = gr.Audio(
                        label=tr(L0, "ref_audio"), type="filepath",
                        sources=["upload", "microphone"], visible=is_clone0,
                    )
                    b_ref_text = gr.Textbox(
                        label=tr(L0, "ref_text"), placeholder=tr(L0, "ref_text_ph"),
                        lines=2, visible=is_clone0,
                    )

                    def _toggle_bclone(label: str):
                        clone = clone_active(label)
                        return (
                            gr.update(visible=not clone),  # b_voice
                            gr.update(visible=clone),      # b_ref_audio
                            gr.update(visible=clone),      # b_ref_text
                        )

                    b_model.change(
                        _toggle_bclone, inputs=[b_model],
                        outputs=[b_voice, b_ref_audio, b_ref_text],
                        show_progress="hidden",
                    )
                    ms.add_bridge_reactor(
                        inputs=[b_model],
                        outputs=[b_voice, b_ref_audio, b_ref_text],
                        fn=lambda data, label: (
                            _toggle_bclone(label)
                            if data.get("cat") == "tts_voice_mode"
                            else (gr.update(), gr.update(), gr.update())
                        ),
                    )
                    with gr.Accordion(tr(L0, "acc_adv"), open=False) as b_acc_adv:
                        with gr.Row():
                            b_before = gr.Slider(0, 2, value=0.0, step=0.05, label=tr(L0, "p_before"))
                            b_after = gr.Slider(0, 2, value=0.0, step=0.05, label=tr(L0, "p_after"))
                        with gr.Row():
                            b_between = gr.Slider(0, 2, value=0.3, step=0.05, label=tr(L0, "p_between"))
                            b_repeat = gr.Slider(1, 5, value=1, step=1, label=tr(L0, "repeat"))
                    with gr.Row():
                        b_format = gr.Radio(choices=FORMATS, value=DEFAULT_FORMAT, label=tr(L0, "format"))
                        b_quality = gr.Radio(choices=QUALITIES, value=DEFAULT_QUALITY, label=tr(L0, "quality"))

        batch_btn = gr.Button(tr(L0, "batch_btn"), variant="primary")
        with gr.Row():
            with gr.Column(scale=3):
                batch_summary = gr.Markdown(elem_classes="result-info")
            with gr.Column(scale=1, min_width=260):
                batch_zip = gr.DownloadButton(tr(L0, "zip"), size="sm")
        batch_table = gr.Dataframe(
            headers=["text", "filename", "status", "error"],
            label=tr(L0, "table"), wrap=True, max_height=240,
        )

        batch_btn.click(
            fn=ui_batch,
            inputs=[batch_text, batch_csv, b_voice, b_model, b_mode, b_speed, b_style,
                    b_before, b_after, b_between, b_repeat, b_format, b_quality,
                    b_ref_audio, b_ref_text, lang_component],
            outputs=[batch_table, batch_zip, batch_summary],
        )

    footer = gr.Markdown(
        f"{tr(L0, 'footer')} `{storage.outputs_dir()}`", elem_classes="app-footer"
    )

    # i18n_outputs and the update list returned by i18n_updates() MUST stay
    # in the same order.
    i18n_outputs = [
        tab_single, tab_batch,
        sec_text, text, sec_voice, model_label, voice,
        sec_out, out_format, quality, generate_btn, audio_out, file_out,
        sec_pron, mode, speed, style, acc_examples,
        p_before, p_after, p_between, repeat,
        sec_files, file_text,
        batch_desc, batch_text, batch_csv, b_defaults, b_model, b_voice, b_speed,
        b_mode, b_style, b_acc_adv, b_before, b_after, b_between, b_repeat,
        b_format, b_quality, batch_btn, batch_zip, batch_table,
        footer,
        ref_audio, ref_text, clone_note, b_ref_audio, b_ref_text,
    ]

    def i18n_updates(lg: str) -> list:
        return [
            gr.update(label=tr(lg, "tab_single")),                                   # tab_single
            gr.update(label=tr(lg, "tab_batch")),                                    # tab_batch
            gr.update(value=tr(lg, "sec_text")),                                     # sec_text
            gr.update(label=tr(lg, "text_label"), placeholder=tr(lg, "text_ph")),    # text
            gr.update(value=tr(lg, "sec_voice")),                                    # sec_voice
            gr.update(label=tr(lg, "model"), choices=model_choices(lg)),             # model_label
            gr.update(label=tr(lg, "voice")),                                        # voice
            gr.update(value=tr(lg, "sec_out")),                                      # sec_out
            gr.update(label=tr(lg, "format")),                                       # out_format
            gr.update(label=tr(lg, "quality")),                                      # quality
            gr.update(value=tr(lg, "generate")),                                     # generate_btn
            gr.update(label=tr(lg, "audio")),                                        # audio_out
            gr.update(label=tr(lg, "download")),                                     # file_out
            gr.update(value=tr(lg, "sec_pron")),                                     # sec_pron
            gr.update(label=tr(lg, "mode"), choices=mode_choices(lg)),               # mode
            gr.update(label=tr(lg, "speed")),                                        # speed
            gr.update(label=tr(lg, "style"), placeholder=tr(lg, "style_ph")),        # style
            gr.update(label=tr(lg, "acc_examples")),                                 # acc_examples
            gr.update(label=tr(lg, "p_before")),                                     # p_before
            gr.update(label=tr(lg, "p_after")),                                      # p_after
            gr.update(label=tr(lg, "p_between")),                                    # p_between
            gr.update(label=tr(lg, "repeat")),                                       # repeat
            gr.update(value=f"📁 **{tr(lg, 'tab_files')}**"),                        # sec_files
            gr.update(label=tr(lg, "file_text")),                                    # file_text
            gr.update(value=tr(lg, "batch_desc")),                                   # batch_desc
            gr.update(label=tr(lg, "batch_text")),                                   # batch_text
            gr.update(label=tr(lg, "batch_csv")),                                    # batch_csv
            gr.update(value=tr(lg, "b_defaults")),                                   # b_defaults
            gr.update(label=tr(lg, "model"), choices=model_choices(lg)),             # b_model
            gr.update(label=tr(lg, "voice")),                                        # b_voice
            gr.update(label=tr(lg, "speed")),                                        # b_speed
            gr.update(label=tr(lg, "mode"), choices=mode_choices(lg)),               # b_mode
            gr.update(label=tr(lg, "b_style")),                                      # b_style
            gr.update(label=tr(lg, "acc_adv")),                                      # b_acc_adv
            gr.update(label=tr(lg, "p_before")),                                     # b_before
            gr.update(label=tr(lg, "p_after")),                                      # b_after
            gr.update(label=tr(lg, "p_between")),                                    # b_between
            gr.update(label=tr(lg, "repeat")),                                       # b_repeat
            gr.update(label=tr(lg, "format")),                                       # b_format
            gr.update(label=tr(lg, "quality")),                                      # b_quality
            gr.update(value=tr(lg, "batch_btn")),                                    # batch_btn
            gr.update(label=tr(lg, "zip")),                                          # batch_zip
            gr.update(label=tr(lg, "table")),                                        # batch_table
            gr.update(value=f"{tr(lg, 'footer')} `{storage.outputs_dir()}`"),        # footer
            gr.update(label=tr(lg, "ref_audio")),                                    # ref_audio
            gr.update(label=tr(lg, "ref_text"), placeholder=tr(lg, "ref_text_ph")),  # ref_text
            gr.update(value=tr(lg, "clone_note")),                                    # clone_note
            gr.update(label=tr(lg, "ref_audio")),                                    # b_ref_audio
            gr.update(label=tr(lg, "ref_text"), placeholder=tr(lg, "ref_text_ph")),  # b_ref_text
        ]

    return i18n_outputs, i18n_updates
