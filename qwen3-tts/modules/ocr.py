"""OCR — Apple's built-in Vision framework (VNRecognizeTextRequest).

Chosen instead of PaddleOCR: it is already on every Mac (no download, no extra
venv), is fast on Apple Silicon, and natively recognises Chinese, English and
Vietnamese. Accepts images and PDFs (pages rendered locally with pypdfium2).
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import gradio as gr

from . import memory_manager as mm
from . import storage

LANG_CHOICES = [
    ("中文 (Simplified)", "zh-Hans"),
    ("中文 (Traditional)", "zh-Hant"),
    ("English", "en-US"),
    ("Tiếng Việt", "vi"),
]
MAX_PDF_PAGES = 50


def _vision_langs(selected: list[str]) -> list[str]:
    """Map UI selections to the tags this macOS actually supports."""
    import Vision

    req = Vision.VNRecognizeTextRequest.alloc().init()
    supported, _ = req.supportedRecognitionLanguagesAndReturnError_(None)
    supported = list(supported or [])
    out = []
    for want in selected or [c[1] for c in LANG_CHOICES]:
        if want in supported:
            out.append(want)
        else:  # e.g. Vietnamese is listed under a regional tag like 'vi-VT'
            match = next((s for s in supported if s.split("-")[0] == want.split("-")[0]), None)
            if match and match not in out:
                out.append(match)
    return out


def ocr_image(image_path: str, languages: list[str]) -> str:
    import Vision
    from Foundation import NSURL

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    langs = _vision_langs(languages)
    if langs:
        request.setRecognitionLanguages_(langs)

    url = NSURL.fileURLWithPath_(str(image_path))
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    ok = handler.performRequests_error_([request], None)
    success = ok[0] if isinstance(ok, tuple) else ok
    if not success:
        raise RuntimeError(f"macOS Vision could not process the image: {ok}")

    observations = request.results() or []
    # Sort top-to-bottom (Vision uses a bottom-left origin), then left-to-right.
    def key(obs):
        box = obs.boundingBox()
        return (-(box.origin.y + box.size.height), box.origin.x)

    lines = []
    for obs in sorted(observations, key=key):
        candidates = obs.topCandidates_(1)
        if candidates and len(candidates):
            lines.append(str(candidates[0].string()))
    return "\n".join(lines)


def _pdf_page_images(pdf_path: str) -> list[str]:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_path)
    paths = []
    try:
        n = min(len(doc), MAX_PDF_PAGES)
        for i in range(n):
            bitmap = doc[i].render(scale=2.0)
            out = storage.temp_dir() / f"ocr_{Path(pdf_path).stem}_{i + 1}_{int(time.time())}.png"
            bitmap.to_pil().save(out)
            paths.append(str(out))
    finally:
        doc.close()
    return paths


def run_ocr(file_path, languages, progress=gr.Progress()):
    if not file_path:
        raise gr.Error("Upload an image (PNG/JPG) or a PDF first.")
    path = Path(file_path)

    mm.set_task("OCR: recognizing text…")
    try:
        if path.suffix.lower() == ".pdf":
            pages = _pdf_page_images(str(path))
            parts = []
            for i, img in enumerate(pages):
                progress((i + 1) / len(pages), desc=f"Page {i + 1} / {len(pages)}")
                text = ocr_image(img, languages)
                parts.append(f"--- Page {i + 1} ---\n{text}" if len(pages) > 1 else text)
            text = "\n\n".join(parts)
        else:
            text = ocr_image(str(path), languages)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"OCR failed: {exc}")
    finally:
        mm.set_task("idle")

    if not text.strip():
        return "(no text recognized)", gr.update(visible=False)

    stem = path.stem[:40] or "ocr"
    out = storage.outputs_dir() / f"{stem}_ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    out.write_text(text + "\n", encoding="utf-8")
    return text, gr.update(value=str(out), visible=True)


def build_ocr_tab(settings: dict):
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            file_in = gr.File(
                label="Image or PDF", file_types=[".png", ".jpg", ".jpeg", ".webp", ".tiff", ".pdf"],
                type="filepath", height=110,
            )
            languages = gr.CheckboxGroup(
                choices=LANG_CHOICES, value=["zh-Hans", "en-US", "vi"],
                label="Languages (priority order)",
            )
            go_btn = gr.Button("🔎 Extract text", variant="primary")
            gr.Markdown(
                "Uses the OCR engine built into macOS — no model download needed.",
                elem_classes="hint-text",
            )
        with gr.Column(scale=6):
            result = gr.Textbox(
                label="Extracted text", lines=14, max_lines=20, show_copy_button=True,
            )
            dl = gr.DownloadButton("⬇️ Download TXT", size="sm", visible=False)

    go_btn.click(run_ocr, inputs=[file_in, languages], outputs=[result, dl])
