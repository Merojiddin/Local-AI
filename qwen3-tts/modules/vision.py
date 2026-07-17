"""Image & homework analysis — Qwen3-VL 4B (4-bit) via MLX-VLM.

Accepts PNG/JPG images or one page of a PDF (rendered locally with pypdfium2).
"""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

from . import memory_manager as mm
from . import model_manager as mgr
from . import storage

MODEL_KEY = "vision-4b"

PROMPT_PRESETS = [
    "Extract the Chinese text",
    "Correct this student homework — point out every mistake and give the correct version",
    "Explain the grammar used in this text",
    "Describe the screenshot",
]


def _load():
    from mlx_vlm import load
    path = mgr.model_path_or_error(MODEL_KEY)
    return mm.HEAVY.get("vision", "Qwen3-VL 4B", lambda: load(path))


def render_pdf_page(pdf_path: str, page_number: int) -> str:
    """Render one PDF page to a PNG in the temp cache; returns the PNG path."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_path)
    try:
        n_pages = len(doc)
        page_idx = max(1, min(int(page_number), n_pages)) - 1
        page = doc[page_idx]
        bitmap = page.render(scale=2.0)
        image = bitmap.to_pil()
    finally:
        doc.close()
    out = storage.temp_dir() / f"vision_pdf_{Path(pdf_path).stem}_{page_idx + 1}_{int(time.time())}.png"
    image.save(out)
    return str(out)


def analyze(image_path, pdf_file, pdf_page, preset, custom_prompt, max_tokens, temperature):
    prompt = (custom_prompt or "").strip() or (preset or "").strip()
    if not prompt:
        raise gr.Error("Pick a preset prompt or type your own.")

    if image_path:
        img = image_path
    elif pdf_file:
        try:
            img = render_pdf_page(pdf_file, int(pdf_page))
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(f"Could not render the PDF page: {exc}")
    else:
        raise gr.Error("Upload an image or a PDF first.")

    try:
        model, processor = _load()
    except RuntimeError as exc:
        raise gr.Error(str(exc))
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Could not load the vision model: {exc}")

    mm.set_task("Vision: analyzing image…")
    try:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        config = getattr(model, "config", None)
        formatted = apply_chat_template(processor, config, prompt, num_images=1)
        result = generate(
            model, processor, formatted, image=[img],
            max_tokens=int(max_tokens), temperature=float(temperature),
            verbose=False,
        )
        text = getattr(result, "text", None) or str(result)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Analysis failed: {exc}")
    finally:
        mm.HEAVY.touch()
        mm.set_task("idle")
    return text.strip()


def build_vision_tab(settings: dict):
    show_adv = bool(settings.get("show_advanced", True))
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            image = gr.Image(label="Image (PNG / JPG)", type="filepath", height=260)
            with gr.Accordion("…or a PDF page", open=False):
                pdf = gr.File(label="PDF", file_types=[".pdf"], type="filepath", height=90)
                pdf_page = gr.Number(value=1, precision=0, minimum=1, label="Page number")
        with gr.Column(scale=6):
            preset = gr.Dropdown(
                choices=PROMPT_PRESETS, value=PROMPT_PRESETS[0],
                label="Task",
            )
            custom_prompt = gr.Textbox(
                label="Or your own prompt (overrides the task above)",
                placeholder="e.g. 把图片里的中文抄写下来并翻译成英文", lines=2,
            )
            with gr.Accordion("Advanced", open=False, visible=show_adv):
                max_tokens = gr.Slider(64, 2048, value=800, step=32, label="Maximum output length")
                temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
            analyze_btn = gr.Button("🔍 Analyze", variant="primary")
            result = gr.Textbox(
                label="Result", lines=12, max_lines=18, show_copy_button=True,
            )

    analyze_btn.click(
        analyze,
        inputs=[image, pdf, pdf_page, preset, custom_prompt, max_tokens, temperature],
        outputs=[result],
    )
