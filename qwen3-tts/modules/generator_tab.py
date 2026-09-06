"""Generator tab — structured JSON generation from document collections.

Compact layout (fits ~1440×900):
  top bar    project · mode · status · Start/Pause/Resume/Stop
  left       input words · source collections · prompt & schema (accordions)
  right      current word · evidence · generated JSON · validation · review
  bottom     queue · failures · results · exports · recovery (accordions)

All state lives on disk (see gen_projects.py) — the tab is a thin view over
it, refreshed by a timer while a queue is running.
"""

from __future__ import annotations

import html
import json
import re

import gradio as gr

from . import collections_store as cs
from . import gen_projects as gp
from . import generator_engine as ge
from . import storage

# Settings are gathered/applied positionally — keep this list and the
# component list created in build_generator_tab() in the same order.
SETTING_KEYS = [
    "mode", "description", "collections",
    "prompt_template", "template_description", "prompt_version",
    "schema_mode", "example_object", "required_keys", "json_schema", "key_field",
    "max_tokens", "temperature", "top_p", "repetition_penalty", "seed",
    "context_budget", "max_retries",
    "chunks_per_word", "max_chars_per_chunk", "max_retrieval_tokens",
    "use_rerank", "min_score", "query_expansion", "exact_first",
    "only_sources", "allow_model_knowledge", "omit_unsupported",
    "never_invent_citations",
    "validation_level", "pos_allowlist", "max_example_chars",
    "duplicate_mode", "field_sources", "authoritative", "collection_priority",
]


# --------------------------------------------------------------------------- #
# Config <-> UI conversion
# --------------------------------------------------------------------------- #
def _parse_json_field(text: str, label: str):
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError as exc:
        raise gr.Error(f"{label} is not valid JSON: {exc}")


def _csv_list(text: str) -> list[str]:
    return [t.strip() for t in re.split(r"[,\n，、]+", text or "") if t.strip()]


def _cfg_from_inputs(cfg: dict, vals: tuple) -> dict:
    v = dict(zip(SETTING_KEYS, vals))
    example = _parse_json_field(v["example_object"], "Example object")
    schema = _parse_json_field(v["json_schema"], "JSON schema")
    fields = _parse_json_field(v["field_sources"], "Field-specific sources") or {}
    if fields and not isinstance(fields, dict):
        raise gr.Error('Field-specific sources must be an object like '
                       '{"examples": ["Example sentences"]}')
    seed = str(v["seed"] or "").strip()
    required = _csv_list(v["required_keys"])
    checks = {}
    pos = _csv_list(v["pos_allowlist"])
    if pos:
        checks["pos_allowlist"] = pos
    if v["max_example_chars"] and float(v["max_example_chars"]) > 0:
        checks["max_example_chars"] = int(v["max_example_chars"])

    cfg = dict(cfg)
    cfg.update(
        mode=v["mode"], description=(v["description"] or "").strip(),
        collections=list(v["collections"] or []),
        prompt_template=v["prompt_template"] or gp.DEFAULT_PROMPT,
        template_description=(v["template_description"] or "").strip(),
        prompt_version=int(v["prompt_version"] or 1),
        schema_mode=v["schema_mode"],
        example_object=example if isinstance(example, dict) else None,
        required_keys=required or None,
        json_schema=schema if isinstance(schema, dict) else None,
        key_field=(v["key_field"] or "hanzi").strip(),
        max_tokens=int(v["max_tokens"]), temperature=float(v["temperature"]),
        top_p=float(v["top_p"]), repetition_penalty=float(v["repetition_penalty"]),
        seed=int(seed) if seed.lstrip("-").isdigit() else None,
        context_budget=int(v["context_budget"]), max_retries=int(v["max_retries"]),
        chunks_per_word=int(v["chunks_per_word"]),
        max_chars_per_chunk=int(v["max_chars_per_chunk"]),
        max_retrieval_tokens=int(v["max_retrieval_tokens"]),
        use_rerank=bool(v["use_rerank"]), min_score=float(v["min_score"]),
        query_expansion=bool(v["query_expansion"]), exact_first=bool(v["exact_first"]),
        only_sources=bool(v["only_sources"]),
        allow_model_knowledge=bool(v["allow_model_knowledge"]),
        omit_unsupported=bool(v["omit_unsupported"]),
        never_invent_citations=bool(v["never_invent_citations"]),
        validation_level=v["validation_level"], custom_checks=checks,
        duplicate_mode=v["duplicate_mode"],
        field_sources={k: list(val) for k, val in fields.items()},
        authoritative=list(v["authoritative"] or []),
        collection_priority=_csv_list(v["collection_priority"]),
    )
    return cfg


def _inputs_from_cfg(cfg: dict) -> list:
    checks = cfg.get("custom_checks") or {}
    return [
        cfg.get("mode", "Dictionary entry"), cfg.get("description", ""),
        [c for c in cfg.get("collections", []) if c in cs.collection_names()],
        cfg.get("prompt_template", gp.DEFAULT_PROMPT),
        cfg.get("template_description", ""), int(cfg.get("prompt_version", 1)),
        cfg.get("schema_mode", "example"),
        json.dumps(cfg["example_object"], ensure_ascii=False, indent=1)
        if cfg.get("example_object") else "",
        ", ".join(cfg.get("required_keys") or []),
        json.dumps(cfg["json_schema"], ensure_ascii=False, indent=1)
        if cfg.get("json_schema") else "",
        cfg.get("key_field", "hanzi"),
        int(cfg.get("max_tokens", ge.DEFAULT_OUTPUT_TOKENS)),
        float(cfg.get("temperature", 0.2)), float(cfg.get("top_p", 0.95)),
        float(cfg.get("repetition_penalty", 1.0)),
        "" if cfg.get("seed") is None else str(cfg.get("seed")),
        int(cfg.get("context_budget", 32768)), int(cfg.get("max_retries", 3)),
        int(cfg.get("chunks_per_word", 10)), int(cfg.get("max_chars_per_chunk", 1200)),
        int(cfg.get("max_retrieval_tokens", 6000)),
        bool(cfg.get("use_rerank", True)), float(cfg.get("min_score", 0.0)),
        bool(cfg.get("query_expansion", True)), bool(cfg.get("exact_first", True)),
        bool(cfg.get("only_sources", True)), bool(cfg.get("allow_model_knowledge", False)),
        bool(cfg.get("omit_unsupported", True)), bool(cfg.get("never_invent_citations", True)),
        cfg.get("validation_level", "strict"),
        ", ".join(checks.get("pos_allowlist") or []),
        int(checks.get("max_example_chars") or 0),
        cfg.get("duplicate_mode", "merge"),
        json.dumps(cfg.get("field_sources") or {}, ensure_ascii=False, indent=1)
        if cfg.get("field_sources") else "",
        [c for c in cfg.get("authoritative", []) if c in cs.collection_names()],
        ", ".join(cfg.get("collection_priority") or []),
    ]


# --------------------------------------------------------------------------- #
# Table / panel renderers
# --------------------------------------------------------------------------- #
def _queue_rows(pid: str) -> list[list]:
    if not pid:
        return []
    return [
        [i["id"], i["hanzi"], i.get("pinyin", ""), i["status"], i.get("attempts", 0),
         (i.get("error") or "")[:120]]
        for i in gp.load_queue(pid)
    ]


def _failure_rows(pid: str) -> list[list]:
    if not pid:
        return []
    return [
        [f["hanzi"], "; ".join(f.get("errors", []))[:200], f.get("attempts", 0),
         f.get("timestamp", "")]
        for f in gp.load_failures(pid)
    ]


def _results_summary(pid: str) -> str:
    if not pid:
        return ""
    results = gp.load_results(pid)
    cfg = gp.get_config(pid) or {}
    key = cfg.get("key_field", "hanzi")
    words = [str(o.get(key, "?")) for o in results if isinstance(o, dict)]
    meta = gp.load_json(gp._pfile(pid, "results_meta.json"), default={}) or {}
    counts = {"approved": 0, "unreviewed": 0, "needs_correction": 0, "rejected": 0}
    for w in words:
        counts[meta.get(w, {}).get("review_status", "unreviewed")] = \
            counts.get(meta.get(w, {}).get("review_status", "unreviewed"), 0) + 1
    tail = "、".join(words[-12:])
    return (f"**results.json: {len(words)} objects** · approved {counts['approved']} · "
            f"unreviewed {counts['unreviewed']} · needs correction "
            f"{counts['needs_correction']} · rejected {counts['rejected']}\n\n"
            f"Latest: {tail or '—'}")


def _status_html(pid: str) -> str:
    """Run status as a filled progress bar plus a one-line summary.

    The runner works through the queue on its own thread, so there is no Gradio
    progress event to hang a bar on — the 2.5s tick redraws this instead. The
    data-* attributes are what branding's window-top bar mirrors, and its
    data-run flip back to "idle" is what triggers the finish chime.
    """
    snap = ge.RUNNER.snapshot()
    running = snap["status"] != "idle" and snap.get("pid") == pid
    parts = []
    pct = 0.0
    label = ""

    if pid:
        counts = gp.queue_counts(pid)
        done = counts["completed"] + counts["failed"] + counts["skipped"]
        total = done + counts["queued"]
        pct = 100.0 * counts["completed"] / total if total else 0.0
        parts.append(f"queued <b>{counts['queued']}</b> · completed "
                     f"<b>{counts['completed']}</b> · failed <b>{counts['failed']}</b> "
                     f"· skipped <b>{counts['skipped']}</b>")

    if running:
        word = snap.get("current_word") or "…"
        pos, tot = snap.get("position", 0), snap.get("total", 0)
        if tot:
            pct = 100.0 * pos / tot
        label = f"{word} — {pos}/{tot}"
        parts.insert(0, f"<b>{html.escape(snap['status'].upper())}</b> — current: "
                        f"<b>{html.escape(str(word))}</b> ({pos}/{tot})")
        if snap.get("current_elapsed"):
            parts.append(f"current {snap['current_elapsed']:.0f}s")
        if snap.get("avg_seconds"):
            parts.append(f"avg {snap['avg_seconds']:.0f}s/word")
        if snap.get("eta_seconds"):
            m, sec = divmod(int(snap["eta_seconds"]), 60)
            parts.append(f"≈{m}m{sec:02d}s left")
            label += f" · ≈{m}m{sec:02d}s left"
        if snap.get("memory_note"):
            parts.append(html.escape(str(snap["memory_note"])))
    elif pid:
        prog = gp.load_progress(pid)
        if prog.get("last_word"):
            parts.append(f"last: {html.escape(str(prog['last_word']))} "
                         f"({html.escape(str(prog.get('last_status')))})")
        parts.insert(0, "<b>idle</b>")

    text = " · ".join(parts) if parts else "<b>idle</b> — create or select a project."
    return (
        f'<div id="gen-progress" class="gen-prog" '
        f'data-run="{"running" if running else "idle"}" '
        f'data-pct="{pct:.1f}" data-label="{html.escape(label, quote=True)}">'
        f'<div class="gen-prog-track">'
        f'<div class="gen-prog-fill" style="width:{max(0.0, min(100.0, pct)):.1f}%"></div>'
        f'</div>'
        f'<div class="gen-prog-text">{text}</div>'
        f'</div>'
    )


def _log_text(pid: str) -> str:
    if not pid:
        return ""
    return "\n".join(f"{e.get('time', '')[-8:]}  {e.get('kind', ''):16} {e.get('message', '')}"
                     for e in gp.read_log_tail(pid, 25))


def _evidence_md(evidence: list[dict]) -> str:
    if not evidence:
        return "_no evidence retrieved yet_"
    out = []
    for e in evidence[:10]:
        prov = " · ".join(str(b) for b in [
            e.get("collection"), e.get("source"),
            f"page {e['page']}" if e.get("page") else None,
            f"rows {e['row']}" if e.get("row") else None,
            e.get("json_path"),
            "exact" if e.get("exact_match") else None,
            f"score {e.get('rerank_score') if e.get('rerank_score') is not None else e.get('score')}",
        ] if b)
        out.append(f"**[{e.get('label')}]** {prov}\n> {(e.get('text') or '')[:400]}")
    return "\n\n".join(out)


def _last_outcome_panels(pid: str):
    snap = ge.RUNNER.snapshot()
    lo = snap.get("last_outcome")
    if not lo or snap.get("pid") != pid:
        return "_nothing generated yet in this session_", "", ""
    obj = json.dumps(lo["object"], ensure_ascii=False, indent=1) if lo.get("object") else ""
    lines = [f"**{lo['word']}** → {lo['status']} in {lo.get('seconds', 0):.1f}s"]
    if lo.get("errors"):
        lines.append("❌ " + " · ".join(lo["errors"]))
    if lo.get("warnings"):
        lines.append("⚠️ " + " · ".join(lo["warnings"]))
    if lo.get("notes"):
        lines.append("ℹ️ " + " · ".join(lo["notes"]))
    return _evidence_md(lo.get("evidence") or []), obj, "\n\n".join(lines)


def _collection_rows() -> list[list]:
    rows = []
    for m in cs.list_collections():
        rows.append([
            m["name"], len(m.get("files", [])),
            "✅" if m.get("indexed") else "✗ rebuild needed",
            m.get("chunk_count", 0), m.get("last_indexed") or "—",
            f"{cs.storage_usage_mb(m['name']):.1f} MB",
        ])
    return rows


def _proj_choices():
    return gp.project_ids()


def _completed_words(pid: str) -> list[str]:
    if not pid:
        return []
    return gp.results_keys(pid)


# --------------------------------------------------------------------------- #
# Tab
# --------------------------------------------------------------------------- #
def build_generator_tab(settings: dict):
    projects = _proj_choices()
    first_pid = projects[0][1] if projects else None
    colls = cs.collection_names()

    # ---- top bar ---------------------------------------------------------- #
    with gr.Row(elem_id="gen-topbar"):
        project_dd = gr.Dropdown(choices=projects, value=first_pid,
                                 label="Generator project", scale=3)
        status_md = gr.HTML(_status_html(first_pid), elem_classes="result-info")
    with gr.Row():
        start_btn = gr.Button("▶️ Start", variant="primary", size="sm")
        pause_btn = gr.Button("⏸ Pause after word", size="sm")
        resume_btn = gr.Button("▶ Resume", size="sm")
        stop_btn = gr.Button("⏹ Stop safely", size="sm")
        skip_btn = gr.Button("⏭ Skip word", size="sm")
        cancel_btn = gr.Button("❌ Cancel generation", size="sm")
        ctrl_info = gr.Markdown(elem_classes="hint-text")

    with gr.Row(equal_height=False):
        # ================= LEFT column ================= #
        with gr.Column(scale=5):
            with gr.Accordion("🗂 A· Project — create / manage", open=not projects):
                with gr.Row():
                    new_name = gr.Textbox(label="New project name",
                                          placeholder="e.g. HSK Dictionary", scale=2)
                    new_mode = gr.Dropdown(choices=list(gp.MODE_PRESETS),
                                           value="Dictionary entry", label="Mode", scale=1)
                new_desc = gr.Textbox(label="Description", lines=1)
                create_btn = gr.Button("➕ Create project", size="sm", variant="primary")
                with gr.Accordion("Rename · duplicate · delete · export · import", open=False):
                    ren_name = gr.Textbox(label="Rename to / duplicate as")
                    with gr.Row():
                        rename_btn = gr.Button("Rename", size="sm")
                        dup_btn = gr.Button("Duplicate", size="sm")
                        exp_cfg_btn = gr.Button("Export config", size="sm")
                    del_confirm = gr.Checkbox(label="Yes, delete this project and its results")
                    del_btn = gr.Button("🗑 Delete project", size="sm", variant="stop")
                    imp_cfg_file = gr.File(label="Import config (.json)", type="filepath",
                                           height=60)
                    imp_cfg_btn = gr.Button("Import as new project", size="sm")
                proj_info = gr.Markdown(elem_classes="result-info")

            with gr.Group():
                gr.Markdown("**C· Input words**", elem_classes="section-head")
                words_tb = gr.Textbox(
                    label="Words (one per line, or 、/, separated)", lines=4,
                    placeholder="爱\n喜欢\n学习",
                )
                words_file = gr.File(label="Or upload TXT / CSV / JSON word list",
                                     file_types=[".txt", ".csv", ".json"],
                                     type="filepath", height=60)
                with gr.Row():
                    add_words_btn = gr.Button("➕ Add to queue", size="sm", variant="primary")
                    allow_dup_cb = gr.Checkbox(label="Allow duplicates", value=False)
                input_report = gr.Markdown(elem_classes="result-info")
                with gr.Accordion("Edit queue", open=False):
                    sel_ids = gr.Textbox(label="Item IDs (comma-separated)",
                                         placeholder="3, 5, 8")
                    with gr.Row():
                        remove_ids_btn = gr.Button("Remove", size="sm")
                        top_ids_btn = gr.Button("Move to top", size="sm")
                        start_sel_btn = gr.Button("Process only these", size="sm")
                    with gr.Row():
                        retry_failed_btn = gr.Button("↻ Retry failed", size="sm")
                        requeue_unrev_btn = gr.Button("↻ Requeue unreviewed", size="sm")
                    queue_info = gr.Markdown(elem_classes="result-info")

            with gr.Group():
                gr.Markdown("**B· Source collections**", elem_classes="section-head")
                coll_cb = gr.CheckboxGroup(choices=colls, label="Use these collections")
                with gr.Accordion("Manage collections", open=False):
                    coll_table = gr.Dataframe(
                        headers=["Collection", "Files", "Indexed", "Chunks",
                                 "Last indexed", "Size"],
                        value=_collection_rows(), interactive=False, max_height=160)
                    with gr.Row():
                        c_new_name = gr.Textbox(label="New collection name", scale=2)
                        c_new_btn = gr.Button("➕ Create", size="sm")
                    c_pick = gr.Dropdown(choices=colls, label="Collection")
                    c_files = gr.File(
                        label="Add files (PDF/DOCX/TXT/MD/CSV/JSON/PNG/JPG)",
                        file_count="multiple", type="filepath", height=70,
                        file_types=list(cs.SUPPORTED_EXTS))
                    with gr.Row():
                        c_add_btn = gr.Button("Add files", size="sm")
                        c_rebuild_btn = gr.Button("🔄 Rebuild index", size="sm",
                                                  variant="primary")
                    c_file_name = gr.Textbox(label="File name (for remove / preview)")
                    with gr.Row():
                        c_rm_file_btn = gr.Button("Remove file", size="sm")
                        c_preview_btn = gr.Button("Preview extract", size="sm")
                    c_del_confirm = gr.Checkbox(label="Yes, delete this collection")
                    c_del_btn = gr.Button("🗑 Delete collection", size="sm", variant="stop")
                    with gr.Row():
                        c_query = gr.Textbox(label="Manual search", scale=2)
                        c_search_btn = gr.Button("🔍", size="sm")
                    c_info = gr.Markdown(elem_classes="result-info")
                    c_preview_md = gr.Textbox(label="Preview / search results", lines=6,
                                              max_lines=12)

            with gr.Accordion("📝 D· Prompt template", open=False):
                prompt_tb = gr.Textbox(label="Template (uses {{word}}, {{retrieved_sources}}, "
                                             "{{json_schema}}, …)",
                                       value=gp.DEFAULT_PROMPT, lines=12)
                with gr.Row():
                    tmpl_desc = gr.Textbox(label="Template description", scale=2)
                    tmpl_ver = gr.Number(label="Version", value=1, precision=0, scale=1)
                with gr.Row():
                    tmpl_reset_btn = gr.Button("Reset to default", size="sm")
                    preview_btn = gr.Button("👁 Preview final prompt", size="sm")
                prompt_preview = gr.Textbox(label="Prompt preview (first queued word)",
                                            lines=6, max_lines=14)

            with gr.Accordion("🧩 D· JSON schema / example", open=False):
                schema_mode = gr.Radio(choices=["example", "schema"], value="example",
                                       label="Mode")
                example_tb = gr.Code(label="Example object (example mode)", language="json",
                                     lines=6)
                required_tb = gr.Textbox(
                    label="Required keys (comma-separated; empty = all example keys)")
                schema_tb = gr.Code(label="JSON Schema (schema mode)", language="json",
                                    lines=6)
                key_field_tb = gr.Textbox(label="Unique key field", value="hanzi")

            with gr.Accordion("⚙️ E· Generation settings", open=False):
                quick_tokens = gr.Radio(
                    choices=[str(t) for t in ge.QUICK_TOKEN_OPTIONS],
                    value=str(ge.DEFAULT_OUTPUT_TOKENS), label="Quick output-token presets")
                max_tokens_sl = gr.Slider(ge.MIN_OUTPUT_TOKENS, ge.MAX_OUTPUT_TOKENS,
                                          value=ge.DEFAULT_OUTPUT_TOKENS, step=256,
                                          label="Maximum output tokens")
                with gr.Row():
                    temp_sl = gr.Slider(0.0, 1.5, value=0.2, step=0.05, label="Temperature")
                    top_p_sl = gr.Slider(0.05, 1.0, value=0.95, step=0.05, label="Top-p")
                with gr.Row():
                    rep_sl = gr.Slider(1.0, 1.5, value=1.0, step=0.05,
                                       label="Repetition penalty")
                    seed_tb = gr.Textbox(label="Random seed (empty = off)")
                ctx_sl = gr.Slider(8192, ge.model_max_context(), value=32768, step=4096,
                                   label=f"Context budget (model max "
                                         f"{ge.model_max_context():,}; high values use "
                                         "more RAM)")
                retries_sl = gr.Slider(1, 5, value=3, step=1, label="Max model retries/word")
                estimate_btn = gr.Button("📏 Estimate context usage", size="sm")
                estimate_md = gr.Markdown(elem_classes="result-info")

            with gr.Accordion("🔍 Advanced retrieval", open=False):
                with gr.Row():
                    chunks_sl = gr.Slider(3, 30, value=10, step=1,
                                          label="Retrieved chunks per word")
                    chunk_chars_sl = gr.Slider(200, 4000, value=1200, step=100,
                                               label="Max characters per chunk")
                ret_tokens_sl = gr.Slider(500, 20000, value=6000, step=500,
                                          label="Max total retrieval tokens")
                with gr.Row():
                    rerank_cb = gr.Checkbox(value=True, label="Use reranker")
                    minscore_sl = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
                                            label="Minimum relevance score")
                with gr.Row():
                    expansion_cb = gr.Checkbox(value=True, label="Query expansion")
                    exact_cb = gr.Checkbox(value=True, label="Exact word matches first")

            with gr.Accordion("🎯 Accuracy controls", open=False):
                only_src_cb = gr.Checkbox(value=True, label="Only use supplied sources")
                allow_knowledge_cb = gr.Checkbox(
                    value=False, label="Allow model knowledge when sources are missing")
                omit_cb = gr.Checkbox(value=True,
                                      label="Omit unsupported optional fields")
                no_cite_cb = gr.Checkbox(value=True, label="Never invent citations")

            with gr.Accordion("✅ Validation rules", open=False):
                valid_radio = gr.Radio(choices=["basic", "schema", "strict", "custom"],
                                       value="strict", label="Validation level")
                pos_tb = gr.Textbox(label="POS allowlist (comma-separated; empty = off)",
                                    placeholder="n, v, adj, adv")
                max_ex_nb = gr.Number(label="Max example length (chars, 0 = off)",
                                      value=0, precision=0)
                dup_radio = gr.Radio(choices=["merge", "skip", "replace", "review"],
                                     value="merge", label="When the word already exists")

            with gr.Accordion("🗺 Field-specific sources & priority", open=False):
                field_src_tb = gr.Code(
                    label='Field → collections, e.g. {"examples": ["Example sentences"]}',
                    language="json", lines=4)
                auth_cb = gr.CheckboxGroup(choices=colls,
                                           label="Authoritative collections")
                prio_tb = gr.Textbox(
                    label="Collection priority (highest first, comma-separated)")

            save_btn = gr.Button("💾 Save project settings", variant="primary")
            save_info = gr.Markdown(elem_classes="result-info")

        # ================= RIGHT column ================= #
        with gr.Column(scale=6):
            with gr.Group():
                gr.Markdown("**F· Current word**", elem_classes="section-head")
                evidence_md = gr.Markdown("_nothing generated yet in this session_")
                gen_json = gr.Code(label="Generated JSON", language="json", lines=10)
                valid_md = gr.Markdown(elem_classes="result-info")
            with gr.Accordion("📜 Live log", open=False):
                log_tb = gr.Textbox(show_label=False, lines=10, max_lines=16)

            with gr.Accordion("🔎 H· Review & edit", open=False):
                with gr.Row():
                    review_dd = gr.Dropdown(choices=_completed_words(first_pid),
                                            label="Word", scale=2)
                    review_load_btn = gr.Button("Load", size="sm")
                review_meta = gr.Markdown(elem_classes="hint-text")
                review_json = gr.Code(label="Object (editable)", language="json", lines=10)
                with gr.Row():
                    rv_validate_btn = gr.Button("Validate", size="sm")
                    rv_save_btn = gr.Button("💾 Save edit", size="sm", variant="primary")
                    rv_revert_btn = gr.Button("Revert", size="sm")
                with gr.Row():
                    rv_approve_btn = gr.Button("✅ Approve", size="sm")
                    rv_needs_btn = gr.Button("✏️ Needs correction", size="sm")
                    rv_reject_btn = gr.Button("🚫 Reject", size="sm")
                rv_fields_tb = gr.Textbox(label="Regenerate only these fields "
                                                "(comma-separated)",
                                          placeholder="examples, collocations")
                with gr.Row():
                    rv_regen_fields_btn = gr.Button("↻ Regenerate fields", size="sm")
                    rv_regen_word_btn = gr.Button("↻ Regenerate whole word", size="sm")
                review_info = gr.Markdown(elem_classes="result-info")
                review_sources = gr.Markdown()
                with gr.Accordion("Raw model response & final prompt", open=False):
                    review_raw = gr.Textbox(show_label=False, lines=10, max_lines=18)

    # ================= BOTTOM ================= #
    with gr.Row(equal_height=False):
        with gr.Column(scale=6):
            gr.Markdown("**Queue**", elem_classes="section-head")
            queue_df = gr.Dataframe(
                headers=["ID", "Word", "Pinyin", "Status", "Tries", "Error"],
                value=_queue_rows(first_pid), interactive=False, max_height=220)
        with gr.Column(scale=5):
            gr.Markdown("**G· Results & failures**", elem_classes="section-head")
            results_md = gr.Markdown(_results_summary(first_pid))
            fail_df = gr.Dataframe(headers=["Word", "Errors", "Tries", "When"],
                                   value=_failure_rows(first_pid), interactive=False,
                                   max_height=140)
            with gr.Row():
                exp_json_btn = gr.Button("⬇ JSON", size="sm")
                exp_jsonl_btn = gr.Button("⬇ JSONL", size="sm")
                exp_csv_btn = gr.Button("⬇ CSV", size="sm")
                exp_zip_btn = gr.Button("⬇ ZIP", size="sm")
                open_exports_btn = gr.Button("📂 Open folder", size="sm")
            export_info = gr.Markdown(elem_classes="result-info")
            with gr.Accordion("🛟 Recovery tools", open=False):
                with gr.Row():
                    rec_check_btn = gr.Button("Check consistency", size="sm")
                    rec_restore_btn = gr.Button("Restore latest backup", size="sm")
                    rec_rebuild_btn = gr.Button("Rebuild from object files", size="sm")
                rec_info = gr.Markdown(elem_classes="result-info")

    # ----------------------------------------------------------------------- #
    # Wiring
    # ----------------------------------------------------------------------- #
    setting_comps = [
        new_mode, new_desc, coll_cb,
        prompt_tb, tmpl_desc, tmpl_ver,
        schema_mode, example_tb, required_tb, schema_tb, key_field_tb,
        max_tokens_sl, temp_sl, top_p_sl, rep_sl, seed_tb,
        ctx_sl, retries_sl,
        chunks_sl, chunk_chars_sl, ret_tokens_sl,
        rerank_cb, minscore_sl, expansion_cb, exact_cb,
        only_src_cb, allow_knowledge_cb, omit_cb, no_cite_cb,
        valid_radio, pos_tb, max_ex_nb,
        dup_radio, field_src_tb, auth_cb, prio_tb,
    ]
    tables = [queue_df, results_md, fail_df, status_md]

    def _refresh_tables(pid):
        return (_queue_rows(pid), _results_summary(pid), _failure_rows(pid),
                _status_html(pid))

    def _save_cfg(pid, *vals) -> dict:
        if not pid:
            raise gr.Error("Create or select a project first.")
        cfg = gp.get_config(pid)
        if cfg is None:
            raise gr.Error(f"Project '{pid}' not found.")
        cfg = _cfg_from_inputs(cfg, vals)
        gp.save_config(pid, cfg)
        return cfg

    # ---- project selection / CRUD ---- #
    def ui_select(pid):
        if not pid:
            return ([gr.update()] * len(setting_comps)
                    + list(_refresh_tables(pid)) + [gr.update(choices=[])])
        cfg = gp.get_config(pid)
        if cfg is None:
            raise gr.Error(f"Project '{pid}' not found.")
        return (_inputs_from_cfg(cfg) + list(_refresh_tables(pid))
                + [gr.update(choices=_completed_words(pid))])

    project_dd.change(ui_select, inputs=[project_dd],
                      outputs=setting_comps + tables + [review_dd])

    def ui_create(name, desc, mode):
        try:
            pid = gp.create_project(name, desc, mode)
        except ValueError as exc:
            raise gr.Error(str(exc))
        return (gr.update(choices=_proj_choices(), value=pid),
                f"✅ Created project **{name}**. Select collections, add words, save "
                "settings, then press Start.")

    create_btn.click(ui_create, inputs=[new_name, new_desc, new_mode],
                     outputs=[project_dd, proj_info])

    def ui_rename(pid, name):
        if not pid:
            raise gr.Error("Select a project first.")
        gp.rename_project(pid, name)
        return gr.update(choices=_proj_choices(), value=pid), f"Renamed to **{name}**."

    rename_btn.click(ui_rename, inputs=[project_dd, ren_name],
                     outputs=[project_dd, proj_info])

    def ui_duplicate(pid, name):
        if not pid:
            raise gr.Error("Select a project first.")
        try:
            new_pid = gp.duplicate_project(pid, name or "Copy")
        except ValueError as exc:
            raise gr.Error(str(exc))
        return (gr.update(choices=_proj_choices(), value=new_pid),
                f"Duplicated into **{name or 'Copy'}** (settings copied, queue empty).")

    dup_btn.click(ui_duplicate, inputs=[project_dd, ren_name],
                  outputs=[project_dd, proj_info])

    def ui_delete(pid, confirmed):
        if not pid:
            raise gr.Error("Select a project first.")
        if not confirmed:
            raise gr.Error("Tick the confirmation box first.")
        snap = ge.RUNNER.snapshot()
        if snap["status"] != "idle" and snap.get("pid") == pid:
            raise gr.Error("Stop the running queue first.")
        msg = gp.delete_project(pid)
        choices = _proj_choices()
        return (gr.update(choices=choices, value=choices[0][1] if choices else None),
                gr.update(value=False), msg)

    del_btn.click(ui_delete, inputs=[project_dd, del_confirm],
                  outputs=[project_dd, del_confirm, proj_info])

    def ui_export_cfg(pid):
        if not pid:
            raise gr.Error("Select a project first.")
        return f"Exported to `{gp.export_config(pid)}`"

    exp_cfg_btn.click(ui_export_cfg, inputs=[project_dd], outputs=[proj_info])

    def ui_import_cfg(path, name):
        if not path:
            raise gr.Error("Upload a config file first.")
        try:
            pid = gp.import_config(path, name)
        except ValueError as exc:
            raise gr.Error(str(exc))
        return (gr.update(choices=_proj_choices(), value=pid),
                f"Imported as **{gp.get_config(pid)['name']}**.")

    imp_cfg_btn.click(ui_import_cfg, inputs=[imp_cfg_file, ren_name],
                      outputs=[project_dd, proj_info])

    # ---- settings save / template helpers ---- #
    def ui_save(pid, *vals):
        cfg = _save_cfg(pid, *vals)
        return (f"✅ Saved settings for **{cfg['name']}** "
                f"(prompt v{cfg['prompt_version']}, {len(cfg['collections'])} "
                "collection(s)).")

    save_btn.click(ui_save, inputs=[project_dd] + setting_comps, outputs=[save_info])

    quick_tokens.change(lambda v: gr.update(value=int(v)),
                        inputs=[quick_tokens], outputs=[max_tokens_sl])
    tmpl_reset_btn.click(lambda: gr.update(value=gp.DEFAULT_PROMPT), outputs=[prompt_tb])

    def _sample_item(pid):
        queue = [i for i in gp.load_queue(pid) if i["status"] == "queued"] or \
            gp.load_queue(pid)
        return queue[0] if queue else {"hanzi": "爱", "pinyin": "ài"}

    def ui_preview(pid, *vals):
        cfg = _save_cfg(pid, *vals)
        item = _sample_item(pid)
        chunks, _ = ge.retrieve_evidence(cfg, item) if cfg.get("collections") else ([], {})
        system, user = ge.build_word_prompt(cfg, item, chunks, 1,
                                            max(1, gp.queue_counts(pid)["total"]))
        return f"=== SYSTEM ===\n{system}\n\n=== USER ===\n{user}"

    preview_btn.click(ui_preview, inputs=[project_dd] + setting_comps,
                      outputs=[prompt_preview])

    def ui_estimate(pid, *vals):
        cfg = _save_cfg(pid, *vals)
        item = _sample_item(pid)
        chunks, _ = ge.retrieve_evidence(cfg, item) if cfg.get("collections") else ([], {})
        system, user = ge.build_word_prompt(cfg, item, chunks, 1, 1)
        rep = ge.context_report(system, "", user, cfg["max_tokens"],
                                cfg["context_budget"])
        state = "✅ fits" if rep["fits"] else f"⚠️ over budget by {rep['overflow']:,} tokens " \
            "(evidence will be trimmed automatically before the output allowance is cut)"
        return (f"Word **{item['hanzi']}** — system {rep['system']:,} + prompt&evidence "
                f"{rep['prompt']:,} + reserved output {rep['output_reserved']:,} "
                f"= **{rep['total']:,} tokens** of {rep['budget']:,} budget "
                f"(model max {rep['model_max']:,}). {state}")

    estimate_btn.click(ui_estimate, inputs=[project_dd] + setting_comps,
                       outputs=[estimate_md])

    # ---- words / queue ---- #
    def ui_add_words(pid, text, file, allow_dup):
        if not pid:
            raise gr.Error("Create or select a project first.")
        rows = gp.rows_from_text(text or "")
        if file:
            try:
                rows += gp.rows_from_file(file)
            except ValueError as exc:
                raise gr.Error(str(exc))
        if not rows:
            raise gr.Error("Type or upload some words first.")
        rep = gp.enqueue(pid, rows, allow_duplicates=bool(allow_dup))
        msg = (f"**{len(rep['valid'])} added** · queue total {rep['queued_total']}"
               + (f" · duplicates in input: {len(rep['duplicate_in_input'])}"
                  if rep["duplicate_in_input"] else "")
               + (f" · already queued/generated (skipped): "
                  f"{'、'.join(rep['duplicate_existing'][:10])}"
                  if rep["duplicate_existing"] else "")
               + (f" · invalid rows: {len(rep['invalid'])}" if rep["invalid"] else ""))
        return msg, gr.update(value=""), *_refresh_tables(pid)

    add_words_btn.click(ui_add_words,
                        inputs=[project_dd, words_tb, words_file, allow_dup_cb],
                        outputs=[input_report, words_tb] + tables)

    def _ids(text):
        return [int(t) for t in re.findall(r"\d+", text or "")]

    def ui_remove_ids(pid, text):
        n = gp.remove_from_queue(pid, _ids(text))
        return f"Removed {n} item(s).", *_refresh_tables(pid)

    remove_ids_btn.click(ui_remove_ids, inputs=[project_dd, sel_ids],
                         outputs=[queue_info] + tables)

    def ui_top_ids(pid, text):
        gp.reorder_queue(pid, _ids(text))
        return "Moved to top.", *_refresh_tables(pid)

    top_ids_btn.click(ui_top_ids, inputs=[project_dd, sel_ids],
                      outputs=[queue_info] + tables)

    def ui_retry_failed(pid):
        n = gp.reset_items(pid, statuses=("failed",))
        return f"Requeued {n} failed item(s) — press Start.", *_refresh_tables(pid)

    retry_failed_btn.click(ui_retry_failed, inputs=[project_dd],
                           outputs=[queue_info] + tables)

    def ui_requeue_unreviewed(pid):
        if not pid:
            raise gr.Error("Select a project first.")
        meta = gp.load_json(gp._pfile(pid, "results_meta.json"), default={}) or {}
        queue = gp.load_queue(pid)
        n = 0
        for item in queue:
            status = meta.get(item["hanzi"], {}).get("review_status", "unreviewed")
            if item["status"] == "completed" and status == "unreviewed":
                item.update(status="queued", error=None)
                n += 1
        gp.save_queue(pid, queue)
        return (f"Requeued {n} unreviewed item(s) — press Start to regenerate them.",
                *_refresh_tables(pid))

    requeue_unrev_btn.click(ui_requeue_unreviewed, inputs=[project_dd],
                            outputs=[queue_info] + tables)

    # ---- run controls ---- #
    def ui_start(pid, *vals):
        _save_cfg(pid, *vals)  # Start always runs with what is on screen
        return ge.RUNNER.start(pid), _status_html(pid)

    start_btn.click(ui_start, inputs=[project_dd] + setting_comps,
                    outputs=[ctrl_info, status_md])

    def ui_start_selected(pid, text, *vals):
        _save_cfg(pid, *vals)
        ids = _ids(text)
        if not ids:
            raise gr.Error("Type the queue item IDs first.")
        gp.reset_items(pid, ids=ids, statuses=("failed", "skipped", "completed"))
        return ge.RUNNER.start(pid, only_ids=ids), _status_html(pid)

    start_sel_btn.click(ui_start_selected,
                        inputs=[project_dd, sel_ids] + setting_comps,
                        outputs=[ctrl_info, status_md])

    for btn, fn in ((pause_btn, ge.RUNNER.pause), (resume_btn, ge.RUNNER.resume),
                    (stop_btn, ge.RUNNER.stop), (skip_btn, ge.RUNNER.skip_current),
                    (cancel_btn, ge.RUNNER.cancel_current)):
        btn.click(lambda pid, f=fn: (f(), _status_html(pid)), inputs=[project_dd],
                  outputs=[ctrl_info, status_md])

    # ---- timer refresh ---- #
    timer = gr.Timer(2.5)

    def ui_tick(pid):
        ev, obj, val = _last_outcome_panels(pid)
        running = ge.RUNNER.snapshot()["status"] != "idle"
        return (_status_html(pid), _queue_rows(pid), _results_summary(pid),
                _failure_rows(pid), _log_text(pid), ev, obj, val,
                gr.update(choices=_completed_words(pid)) if not running else gr.update())

    timer.tick(ui_tick, inputs=[project_dd],
               outputs=[status_md, queue_df, results_md, fail_df, log_tb,
                        evidence_md, gen_json, valid_md, review_dd],
               show_progress="hidden")

    # ---- collections management ---- #
    def _coll_refresh():
        names = cs.collection_names()
        return (gr.update(value=_collection_rows()), gr.update(choices=names),
                gr.update(choices=names), gr.update(choices=names))

    coll_outputs = [coll_table, coll_cb, c_pick, auth_cb]

    def ui_new_coll(name):
        try:
            cs.create_collection(name)
        except ValueError as exc:
            raise gr.Error(str(exc))
        return f"✅ Created collection **{name}** — now add files.", *_coll_refresh()

    c_new_btn.click(ui_new_coll, inputs=[c_new_name],
                    outputs=[c_info] + coll_outputs)

    def ui_coll_add(name, files):
        if not name:
            raise gr.Error("Pick a collection first.")
        if not files:
            raise gr.Error("Upload at least one file.")
        try:
            msg = cs.add_files(name, files)
        except ValueError as exc:
            raise gr.Error(str(exc))
        return msg, *_coll_refresh()

    c_add_btn.click(ui_coll_add, inputs=[c_pick, c_files],
                    outputs=[c_info] + coll_outputs)

    def ui_coll_rebuild(name, progress=gr.Progress()):
        if not name:
            raise gr.Error("Pick a collection first.")
        try:
            msg = cs.rebuild_index(name, progress_cb=lambda f, t: progress(f, desc=t))
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(str(exc))
        return msg, *_coll_refresh()

    c_rebuild_btn.click(ui_coll_rebuild, inputs=[c_pick],
                        outputs=[c_info] + coll_outputs)

    def ui_coll_rm_file(name, filename):
        if not name or not filename:
            raise gr.Error("Pick a collection and type the file name.")
        return cs.remove_file(name, filename), *_coll_refresh()

    c_rm_file_btn.click(ui_coll_rm_file, inputs=[c_pick, c_file_name],
                        outputs=[c_info] + coll_outputs)

    def ui_coll_preview(name, filename):
        if not name or not filename:
            raise gr.Error("Pick a collection and type the file name.")
        try:
            return cs.preview_file(name, filename)
        except ValueError as exc:
            raise gr.Error(str(exc))

    c_preview_btn.click(ui_coll_preview, inputs=[c_pick, c_file_name],
                        outputs=[c_preview_md])

    def ui_coll_delete(name, confirmed):
        if not name:
            raise gr.Error("Pick a collection first.")
        if not confirmed:
            raise gr.Error("Tick the confirmation box first.")
        msg = cs.delete_collection(name)
        return msg, gr.update(value=False), *_coll_refresh()

    c_del_btn.click(ui_coll_delete, inputs=[c_pick, c_del_confirm],
                    outputs=[c_info, c_del_confirm] + coll_outputs)

    def ui_coll_search(name, query):
        if not name:
            raise gr.Error("Pick a collection first.")
        if not (query or "").strip():
            raise gr.Error("Type a search query first.")
        res = cs.search([name], [query.strip()], top_k=5, exact_word=query.strip())
        if not res:
            return "(no matches)"
        return "\n\n".join(f"[{cs.format_source_line(r)}] score {r['score']:.3f}\n"
                           f"{r['text'][:300]}" for r in res)

    c_search_btn.click(ui_coll_search, inputs=[c_pick, c_query],
                       outputs=[c_preview_md])

    # ---- review & edit ---- #
    def ui_review_load(pid, word):
        if not pid or not word:
            raise gr.Error("Pick a completed word first.")
        cfg = gp.get_config(pid)
        key = cfg.get("key_field", "hanzi")
        obj = next((o for o in gp.load_results(pid)
                    if isinstance(o, dict) and str(o.get(key)) == word), None)
        if obj is None:
            raise gr.Error(f"'{word}' is not in results.json.")
        meta = gp.get_meta_for(pid, word)
        notes = gp.load_review_notes(pid, word)
        srcs = gp.sources_for(pid, word)
        meta_line = (f"status **{meta.get('review_status', 'unreviewed')}** · "
                     f"model {meta.get('generated_by_model', '?')} · prompt "
                     f"v{meta.get('prompt_version', '?')} · created "
                     f"{meta.get('created_at', '?')} · updated {meta.get('updated_at', '?')}")
        notes_md = "\n".join(f"- ⚠️ [{n['type']}] {n['detail']}" for n in notes[-8:])
        src_md = "**Sources used:**\n\n" + _evidence_md(srcs)
        return (json.dumps(obj, ensure_ascii=False, indent=1), meta_line,
                (notes_md + "\n\n" if notes_md else "") + src_md,
                gp.load_raw(pid, word), "")

    review_load_btn.click(ui_review_load, inputs=[project_dd, review_dd],
                          outputs=[review_json, review_meta, review_sources,
                                   review_raw, review_info])
    review_dd.change(ui_review_load, inputs=[project_dd, review_dd],
                     outputs=[review_json, review_meta, review_sources,
                              review_raw, review_info])

    def _parse_edit(pid, word, text):
        obj = _parse_json_field(text, "Edited object")
        if not isinstance(obj, dict):
            raise gr.Error("The edited text must be one JSON object.")
        cfg = gp.get_config(pid)
        errs, warns = ge.validate_result(obj, cfg, word, n_sources=99)
        return obj, errs, warns

    def ui_rv_validate(pid, word, text):
        _, errs, warns = _parse_edit(pid, word, text)
        if errs:
            return "❌ " + "\n\n❌ ".join(errs)
        return "✅ valid" + (" · ⚠️ " + " · ".join(warns) if warns else "")

    rv_validate_btn.click(ui_rv_validate, inputs=[project_dd, review_dd, review_json],
                          outputs=[review_info])

    def ui_rv_save(pid, word, text):
        obj, errs, warns = _parse_edit(pid, word, text)
        if errs:
            return "❌ Not saved:\n\n❌ " + "\n\n❌ ".join(errs), gr.update()
        gp.upsert_result(pid, word, obj, gp.get_meta_for(pid, word), mode="replace")
        return ("💾 Saved." + (" ⚠️ " + " · ".join(warns) if warns else ""),
                _results_summary(pid))

    rv_save_btn.click(ui_rv_save, inputs=[project_dd, review_dd, review_json],
                      outputs=[review_info, results_md])

    rv_revert_btn.click(ui_review_load, inputs=[project_dd, review_dd],
                        outputs=[review_json, review_meta, review_sources,
                                 review_raw, review_info])

    def _set_status(pid, word, status):
        if not pid or not word:
            raise gr.Error("Load a word first.")
        gp.set_review_status(pid, word, status)
        return f"Marked **{word}** as {status}.", _results_summary(pid)

    rv_approve_btn.click(lambda p, w: _set_status(p, w, "approved"),
                         inputs=[project_dd, review_dd],
                         outputs=[review_info, results_md])
    rv_needs_btn.click(lambda p, w: _set_status(p, w, "needs_correction"),
                       inputs=[project_dd, review_dd],
                       outputs=[review_info, results_md])
    rv_reject_btn.click(lambda p, w: _set_status(p, w, "rejected"),
                        inputs=[project_dd, review_dd],
                        outputs=[review_info, results_md])

    def ui_regen_fields(pid, word, fields_text):
        if not pid or not word:
            raise gr.Error("Load a word first.")
        fields = _csv_list(fields_text)
        if not fields:
            raise gr.Error("Type at least one field name.")
        if ge.RUNNER.snapshot()["status"] != "idle":
            raise gr.Error("Wait for the queue to finish (or pause it) first.")
        out = ge.regenerate_fields(pid, word, fields)
        if out["status"] != "completed":
            return "❌ " + "; ".join(out.get("errors", ["failed"])), gr.update()
        return (f"✅ Regenerated {', '.join(fields)} for **{word}**.",
                json.dumps(out["object"], ensure_ascii=False, indent=1))

    rv_regen_fields_btn.click(ui_regen_fields,
                              inputs=[project_dd, review_dd, rv_fields_tb],
                              outputs=[review_info, review_json])

    def ui_regen_word(pid, word):
        if not pid or not word:
            raise gr.Error("Load a word first.")
        if ge.RUNNER.snapshot()["status"] != "idle":
            raise gr.Error("Wait for the queue to finish (or pause it) first.")
        cfg = gp.get_config(pid)
        item = next((i for i in gp.load_queue(pid) if i["hanzi"] == word),
                    {"hanzi": word})
        out = ge.process_word(pid, cfg, item, 1, 1)
        if out["status"] != "completed":
            return "❌ " + "; ".join(out.get("errors", ["failed"])), gr.update()
        return (f"✅ Regenerated **{word}** ({out.get('outcome')}).",
                json.dumps(out["object"], ensure_ascii=False, indent=1))

    rv_regen_word_btn.click(ui_regen_word, inputs=[project_dd, review_dd],
                            outputs=[review_info, review_json])

    # ---- exports & recovery ---- #
    def _export(pid, fmt):
        if not pid:
            raise gr.Error("Select a project first.")
        return f"Exported: `{gp.export_results(pid, fmt)}`"

    exp_json_btn.click(lambda p: _export(p, "json"), inputs=[project_dd],
                       outputs=[export_info])
    exp_jsonl_btn.click(lambda p: _export(p, "jsonl"), inputs=[project_dd],
                        outputs=[export_info])
    exp_csv_btn.click(lambda p: _export(p, "csv"), inputs=[project_dd],
                      outputs=[export_info])
    exp_zip_btn.click(lambda p: _export(p, "zip"), inputs=[project_dd],
                      outputs=[export_info])
    open_exports_btn.click(
        lambda p: storage.open_in_finder(gp._pfile(p, "exports")) if p else None,
        inputs=[project_dd])

    rec_check_btn.click(lambda p: gp.consistency_check(p) if p else "",
                        inputs=[project_dd], outputs=[rec_info])
    rec_restore_btn.click(lambda p: gp.restore_results_backup(p) if p else "",
                          inputs=[project_dd], outputs=[rec_info])
    rec_rebuild_btn.click(lambda p: gp.rebuild_results_from_objects(p) if p else "",
                          inputs=[project_dd], outputs=[rec_info])
