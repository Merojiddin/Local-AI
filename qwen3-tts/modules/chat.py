"""Chat & coding assistant — Qwen3 4B (4-bit) via MLX-LM.

Project-folder questions use keyword retrieval: only the handful of most
relevant files are sent to the model. The model never reads a whole
repository at once.
"""

from __future__ import annotations

import re
from pathlib import Path

import gradio as gr

from . import memory_manager as mm
from . import model_manager as mgr

MODEL_KEY = "chat-4b"

TEXT_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml", ".toml",
    ".md", ".txt", ".html", ".css", ".sh", ".bash", ".zsh", ".c", ".h",
    ".cpp", ".hpp", ".java", ".kt", ".swift", ".go", ".rs", ".rb", ".php",
    ".sql", ".xml", ".ini", ".cfg", ".csv", ".command",
}
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".ruff_cache", "dist", "build", ".next", "outputs", "cache", "models",
}
MAX_FILE_BYTES = 200_000       # skip bigger files during retrieval
PER_FILE_CHARS = 6_000         # clip each included file
TOTAL_CONTEXT_CHARS = 24_000   # overall budget for retrieved code
UPLOAD_CHARS = 8_000           # clip each uploaded file

DEFAULT_SYSTEM = "You are a helpful assistant for writing, coding and Chinese-language teaching."


def _load():
    from mlx_lm import load
    path = mgr.model_path_or_error(MODEL_KEY)
    return mm.HEAVY.get("chat", "Qwen3 4B", lambda: load(path))


def _read_clipped(path: Path, limit: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"[could not read: {exc}]"
    if len(text) > limit:
        text = text[:limit] + f"\n… [clipped, file has {len(text)} characters]"
    return text


def retrieve_project_files(folder: str, query: str) -> list[tuple[Path, str]]:
    """Keyword-score files in the project and return the most relevant few."""
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ValueError(f"Not a folder: {folder}")
    terms = {w.lower() for w in re.findall(r"[\w一-鿿]{2,}", query)}
    if not terms:
        return []

    scored: list[tuple[float, Path]] = []
    count = 0
    for p in root.rglob("*"):
        if count > 5000:  # safety cap for huge trees
            break
        if p.is_dir() or any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in TEXT_EXTS:
            continue
        count += 1
        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
            content = p.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        name = str(p.relative_to(root)).lower()
        score = sum(3.0 for t in terms if t in name)
        score += sum(min(content.count(t), 10) * 0.5 for t in terms)
        if score > 0:
            scored.append((score, p))

    scored.sort(key=lambda x: -x[0])
    picked: list[tuple[Path, str]] = []
    used = 0
    for _, p in scored[:12]:
        text = _read_clipped(p, PER_FILE_CHARS)
        if used + len(text) > TOTAL_CONTEXT_CHARS:
            break
        picked.append((p.relative_to(root), text))
        used += len(text)
        if len(picked) >= 6:
            break
    return picked


def _build_context(files, folder: str, query: str) -> str:
    uploads = []
    for f in files or []:
        p = Path(f)
        uploads.append(f"### Uploaded file: {p.name}\n```\n{_read_clipped(p, UPLOAD_CHARS)}\n```")
    project = []
    if (folder or "").strip():
        picked = retrieve_project_files(folder.strip(), query)
        if picked:
            listing = "\n".join(
                f"### {rel}\n```\n{text}\n```" for rel, text in picked
            )
            project.append(
                f"Relevant files retrieved from {folder.strip()} "
                f"(only these {len(picked)} files were selected by keyword search — "
                f"this is not the whole project):\n{listing}"
            )
        else:
            project.append(
                f"(No files in {folder.strip()} matched the question's keywords.)"
            )
    return "\n\n".join(uploads + project)


def chat_fn(message, history, system_prompt, temperature, max_tokens, files, folder):
    message = (message or "").strip()
    if not message:
        raise gr.Error("Type a message first.")
    history = list(history or [])

    try:
        context = _build_context(files, folder, message)
    except ValueError as exc:
        raise gr.Error(str(exc))

    try:
        model, tokenizer = _load()
    except RuntimeError as exc:
        raise gr.Error(str(exc))
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Could not load the chat model: {exc}")

    messages = [{"role": "system", "content": (system_prompt or DEFAULT_SYSTEM).strip()}]
    for m in history:
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            messages.append({"role": m["role"], "content": m["content"]})
    user_content = f"{context}\n\n{message}" if context else message
    messages.append({"role": "user", "content": user_content})

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=float(temperature), top_p=0.95)

    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": ""}]
    mm.set_task("Chat: generating…")
    try:
        reply = ""
        for response in stream_generate(
            model, tokenizer, prompt,
            max_tokens=int(max_tokens), sampler=sampler,
        ):
            reply += response.text
            history[-1]["content"] = reply
            yield history, ""
    finally:
        mm.HEAVY.touch()
        mm.set_task("idle")


def build_chat_tab(settings: dict):
    show_adv = bool(settings.get("show_advanced", True))
    with gr.Row(equal_height=False):
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                type="messages", label="Qwen3 4B (local)", height=430,
                show_copy_button=True,
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask anything — coding, writing, Chinese…",
                    show_label=False, lines=2, scale=8,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            with gr.Row():
                clear_btn = gr.Button("🗑 Clear conversation", size="sm")
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("**Settings**", elem_classes="section-head")
                system_prompt = gr.Textbox(
                    label="System instruction", value=DEFAULT_SYSTEM, lines=2,
                )
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                max_tokens = gr.Slider(64, 4096, value=1024, step=64, label="Maximum output length (tokens)")
            with gr.Accordion("Files & project context", open=False, visible=show_adv):
                files = gr.File(
                    label="Attach text/code files", file_count="multiple",
                    type="filepath", height=110,
                )
                folder = gr.Textbox(
                    label="Project folder (optional)",
                    placeholder="/Users/you/Projects/my-app",
                )
                gr.Markdown(
                    "Only the few most relevant files (by keyword match with your "
                    "question) are sent to the model — it does not read the whole "
                    "repository at once.",
                    elem_classes="hint-text",
                )

    inputs = [msg, chatbot, system_prompt, temperature, max_tokens, files, folder]
    send_btn.click(chat_fn, inputs=inputs, outputs=[chatbot, msg])
    msg.submit(chat_fn, inputs=inputs, outputs=[chatbot, msg])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
