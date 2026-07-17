"""Document search — Qwen3-Embedding 0.6B (+ optional Qwen3-Reranker 0.6B).

Upload PDF / DOCX / TXT / Markdown files, build a local semantic index, then
ask questions over it. Indexes live in data/indexes/<name>/ — completely
separate from model weights, so removing models never touches them.

The embedding model is loaded through mlx-lm: Qwen3-Embedding is a standard
Qwen3 tower whose sentence vector is the last-token hidden state, L2-normalised
(the official usage), so no extra dependency is needed.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np

from . import memory_manager as mm
from . import model_manager as mgr
from . import storage

EMBED_KEY = "embed-0.6b"
RERANK_KEY = "rerank-0.6b"
CHAT_KEY = "chat-4b"

QUERY_INSTRUCT = "Given a search query, retrieve relevant passages that answer the query"
CHUNK_CHARS = 700
CHUNK_OVERLAP = 100
MAX_CHUNKS = 2000
MAX_EMBED_TOKENS = 1024


# --------------------------------------------------------------------------- #
# Embedding / reranking via mlx-lm
# --------------------------------------------------------------------------- #
def _load_embedder():
    from mlx_lm import load
    path = mgr.model_path_or_error(EMBED_KEY)
    return mm.LIGHT["embed"].get("embed", "Qwen3-Embedding 0.6B", lambda: load(path))


def _load_reranker():
    from mlx_lm import load
    path = mgr.model_path_or_error(RERANK_KEY)
    return mm.LIGHT["rerank"].get("rerank", "Qwen3-Reranker 0.6B", lambda: load(path))


def embed_texts(texts: list[str], is_query: bool = False, progress_cb=None) -> np.ndarray:
    import mlx.core as mx

    model, tokenizer = _load_embedder()
    eos_id = tokenizer.eos_token_id
    vectors = []
    for i, text in enumerate(texts):
        if is_query:
            text = f"Instruct: {QUERY_INSTRUCT}\nQuery:{text}"
        ids = tokenizer.encode(text)[: MAX_EMBED_TOKENS - 1]
        if not ids or ids[-1] != eos_id:
            ids = ids + [eos_id]
        hidden = model.model(mx.array([ids]))       # [1, L, D] last hidden states
        vec = hidden[0, -1, :].astype(mx.float32)   # last-token pooling
        vec = vec / mx.maximum(mx.linalg.norm(vec), 1e-9)
        mx.eval(vec)
        vectors.append(np.array(vec))
        if progress_cb and (i % 10 == 0 or i == len(texts) - 1):
            progress_cb((i + 1) / len(texts))
    mm.LIGHT["embed"].touch()
    return np.stack(vectors)


def rerank_scores(query: str, passages: list[str]) -> list[float]:
    import mlx.core as mx

    model, tokenizer = _load_reranker()
    try:
        yes_id = tokenizer.convert_tokens_to_ids("yes")
        no_id = tokenizer.convert_tokens_to_ids("no")
    except Exception:  # noqa: BLE001 - fall back to encoding
        yes_id = tokenizer.encode("yes")[-1]
        no_id = tokenizer.encode("no")[-1]

    prefix = (
        "<|im_start|>system\nJudge whether the Document meets the requirements "
        'based on the Query and the Instruct provided. Note that the answer can '
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    scores = []
    for passage in passages:
        text = (
            f"{prefix}<Instruct>: {QUERY_INSTRUCT}\n<Query>: {query}\n"
            f"<Document>: {passage[:4000]}{suffix}"
        )
        ids = tokenizer.encode(text)[-4096:]
        logits = model(mx.array([ids]))[0, -1]
        pair = mx.softmax(mx.stack([logits[no_id], logits[yes_id]]).astype(mx.float32))
        mx.eval(pair)
        scores.append(float(pair[1]))
    mm.LIGHT["rerank"].touch()
    return scores


# --------------------------------------------------------------------------- #
# Text extraction & chunking
# --------------------------------------------------------------------------- #
def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        import docx
        document = docx.Document(str(path))
        return "\n".join(p.text for p in document.paragraphs)
    if suffix in (".txt", ".md", ".markdown"):
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {path.name} (use PDF, DOCX, TXT or MD)")


def chunk_text(text: str) -> list[str]:
    """~700-char chunks with overlap, split at sentence boundaries when possible."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind(c) for c in "。！？.!?\n")
            if cut > CHUNK_CHARS // 2:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


# --------------------------------------------------------------------------- #
# Index storage (data/indexes/<name>/)
# --------------------------------------------------------------------------- #
def _index_dir(name: str) -> Path:
    safe = re.sub(r"[^\w一-鿿-]+", "_", name.strip()).strip("_")
    if not safe:
        raise ValueError("Please give the index a name.")
    return storage.indexes_dir() / safe


def list_indexes() -> list[str]:
    root = storage.indexes_dir()
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "meta.json").exists()
    ) if root.exists() else []


def build_index(name: str, files: list[str], progress_cb=None) -> str:
    if not files:
        raise ValueError("Upload at least one PDF, DOCX, TXT or Markdown file.")
    chunks, sources = [], []
    for f in files:
        p = Path(f)
        text = extract_text(p)
        for c in chunk_text(text):
            chunks.append(c)
            sources.append(p.name)
        if len(chunks) > MAX_CHUNKS:
            raise ValueError(
                f"Too much text (>{MAX_CHUNKS} chunks). Index fewer/shorter files at once."
            )
    if not chunks:
        raise ValueError("No text could be extracted from the uploaded files.")

    mm.set_task(f"Indexing {len(chunks)} passages…")
    try:
        vectors = embed_texts(chunks, is_query=False, progress_cb=progress_cb)
    finally:
        mm.set_task("idle")

    target = _index_dir(name)
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "embeddings.npy", vectors)
    (target / "chunks.json").write_text(
        json.dumps(
            [{"text": c, "source": s} for c, s in zip(chunks, sources)],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (target / "meta.json").write_text(
        json.dumps(
            {
                "name": target.name,
                "created": datetime.now().isoformat(timespec="seconds"),
                "files": sorted({Path(f).name for f in files}),
                "chunks": len(chunks),
                "model": mgr.MODELS[EMBED_KEY]["repo"],
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return f"✅ Index **{target.name}** built: {len(chunks)} passages from {len(files)} file(s)."


def delete_index(name: str) -> str:
    target = _index_dir(name)
    root = storage.indexes_dir()
    if root not in target.parents:
        raise ValueError("Refusing to delete outside the index folder.")
    if not target.exists():
        return f"Index '{name}' does not exist."
    shutil.rmtree(target)
    return f"🗑 Deleted index '{name}' (document files and models are untouched)."


def search_index(name: str, query: str, top_k: int, use_rerank: bool):
    target = _index_dir(name)
    if not (target / "embeddings.npy").exists():
        raise ValueError(f"Index '{name}' not found — build it first.")
    vectors = np.load(target / "embeddings.npy")
    chunks = json.loads((target / "chunks.json").read_text(encoding="utf-8"))

    q = embed_texts([query], is_query=True)[0]
    sims = vectors @ q
    n_candidates = min(len(chunks), max(top_k * 4, top_k) if use_rerank else top_k)
    order = np.argsort(-sims)[:n_candidates]
    results = [
        {"text": chunks[i]["text"], "source": chunks[i]["source"], "score": float(sims[i])}
        for i in order
    ]
    if use_rerank and len(results) > 1:
        mm.set_task("Re-ranking results…")
        try:
            rr = rerank_scores(query, [r["text"] for r in results])
        finally:
            mm.set_task("idle")
        for r, s in zip(results, rr):
            r["score"] = s
        results.sort(key=lambda r: -r["score"])
    return results[: int(top_k)]


def answer_with_chat(query: str, passages: list[dict]) -> str:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    path = mgr.model_path_or_error(CHAT_KEY)
    model, tokenizer = mm.HEAVY.get("chat", "Qwen3 4B", lambda: load(path))
    context = "\n\n".join(
        f"[{i + 1}] ({p['source']}) {p['text']}" for i, p in enumerate(passages)
    )
    messages = [
        {"role": "system", "content": "Answer using ONLY the provided passages. Cite passage numbers like [1]. If the passages do not contain the answer, say so."},
        {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {query}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    mm.set_task("Answering from documents…")
    try:
        text = generate(model, tokenizer, prompt, max_tokens=800,
                        sampler=make_sampler(temp=0.3))
    finally:
        mm.HEAVY.touch()
        mm.set_task("idle")
    return text.strip()


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_documents_tab(settings: dict):
    gr.Markdown(
        "Indexes are stored in `data/indexes/` — separate from model weights and outputs.",
        elem_classes="hint-text",
    )
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            with gr.Group():
                gr.Markdown("**Build an index**", elem_classes="section-head")
                idx_name = gr.Textbox(label="Index name", placeholder="e.g. hsk4-notes")
                idx_files = gr.File(
                    label="PDF / DOCX / TXT / Markdown",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt", ".md", ".markdown"],
                    type="filepath", height=110,
                )
                build_btn = gr.Button("📚 Build index", variant="primary")
                build_info = gr.Markdown(elem_classes="result-info")
            with gr.Accordion("Manage indexes", open=False):
                del_dd = gr.Dropdown(choices=list_indexes(), label="Index")
                confirm_del = gr.Checkbox(label="Yes, delete this index", value=False)
                del_btn = gr.Button("🗑 Delete index", size="sm", variant="stop")
                del_info = gr.Markdown(elem_classes="result-info")
        with gr.Column(scale=6):
            with gr.Group():
                gr.Markdown("**Ask your documents**", elem_classes="section-head")
                q_index = gr.Dropdown(choices=list_indexes(), label="Index")
                question = gr.Textbox(label="Question", lines=2,
                                      placeholder="What does the contract say about …?")
                with gr.Row():
                    top_k = gr.Slider(1, 10, value=4, step=1, label="Passages")
                    use_rerank = gr.Checkbox(
                        value=False, label="Re-rank (needs Qwen3-Reranker)",
                    )
                    use_llm = gr.Checkbox(
                        value=False, label="Answer with chat model (needs Qwen3 4B)",
                    )
                search_btn = gr.Button("🔍 Search", variant="primary")
            answer_md = gr.Markdown(elem_classes="result-info")
            results_md = gr.Markdown()

    def _dd_refresh():
        choices = list_indexes()
        return gr.update(choices=choices), gr.update(choices=choices)

    def ui_build(name, files, progress=gr.Progress()):
        try:
            msg = build_index(
                name, files,
                progress_cb=lambda f: progress(f, desc="Embedding passages…"),
            )
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(str(exc))
        return (msg, *_dd_refresh())

    build_btn.click(ui_build, inputs=[idx_name, idx_files],
                    outputs=[build_info, del_dd, q_index])

    def ui_delete(name, confirmed):
        if not name:
            raise gr.Error("Pick an index first.")
        if not confirmed:
            raise gr.Error("Tick the confirmation box first.")
        try:
            msg = delete_index(name)
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(str(exc))
        return (msg, gr.update(value=False), *_dd_refresh())

    del_btn.click(ui_delete, inputs=[del_dd, confirm_del],
                  outputs=[del_info, confirm_del, del_dd, q_index])

    def ui_search(name, query, k, do_rerank, do_answer):
        query = (query or "").strip()
        if not name:
            raise gr.Error("Pick an index (build one first if the list is empty).")
        if not query:
            raise gr.Error("Type a question first.")
        try:
            results = search_index(name, query, int(k), bool(do_rerank))
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(str(exc))
        if not results:
            return "", "_No matching passages found._"
        answer = ""
        if do_answer:
            try:
                answer = "### Answer\n" + answer_with_chat(query, results)
            except Exception as exc:  # noqa: BLE001
                answer = f"⚠️ Could not generate an answer: {exc}"
        body = "\n\n".join(
            f"**[{i + 1}] {r['source']}** · score {r['score']:.3f}\n\n> {r['text']}"
            for i, r in enumerate(results)
        )
        return answer, body

    search_btn.click(ui_search, inputs=[q_index, question, top_k, use_rerank, use_llm],
                     outputs=[answer_md, results_md])
