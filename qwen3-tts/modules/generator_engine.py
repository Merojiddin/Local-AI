"""Generation engine for the Generator tab.

Pipeline per word (one word at a time — this Mac has 16 GB unified memory):

  queued word -> retrieval from the project's collections (evidence grouped by
  field, provenance kept) -> context-budget check (evidence is trimmed before
  the output-token allowance) -> the selected chat model via mlx-lm -> JSON extraction ->
  deterministic repair -> validation -> model repair with exact errors ->
  full regenerate -> results.json append (atomic) / failures.json.

The queue runner is a single background thread with pause / resume / stop /
skip / cancel controls; progress is checkpointed after every word so the queue
survives app restarts, browser closure and crashes. The chat model shares the
existing HEAVY memory slot, so it can never be loaded next to Vision/Whisper/TTS.
"""

from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from pathlib import Path

from . import collections_store as cs
from . import gen_projects as gp
from . import memory_manager as mm
from . import model_manager as mgr
from .safe_json import extract_and_parse, infer_schema_from_example, validate_schema

def chat_key() -> str:
    """The registry key of the currently selected chat model."""
    from . import model_select
    return model_select.selected_key("chat")

# Output-token policy (spec: min 256, max 16384, default 8192)
MIN_OUTPUT_TOKENS = 256
MAX_OUTPUT_TOKENS = 16384
DEFAULT_OUTPUT_TOKENS = 8192
QUICK_TOKEN_OPTIONS = [2048, 4096, 8192, 12288, 16384]

REPAIR_SYSTEM = (
    "You repair JSON objects. Return ONE corrected JSON object and nothing "
    "else — no markdown fences, no commentary. Repair this object only. "
    "Do not change correct factual content."
)

_TONE_MARKS = "āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜĀÁǍÀĒÉĚÈĪÍǏÌŌÓǑÒŪÚǓÙǕǗǙǛ"
_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")
_SOURCE_LABEL_RE = re.compile(r"\[S(\d+)\]")


# --------------------------------------------------------------------------- #
# Tokens & context budget
# --------------------------------------------------------------------------- #
_tokenizer = None
_tokenizer_key = None
_model_max_ctx = None
_model_max_ctx_key = None


def _get_tokenizer():
    """Tokenizer only (a few MB) — no model weights are loaded for counting."""
    global _tokenizer, _tokenizer_key
    key = chat_key()
    if _tokenizer is None or _tokenizer_key != key:
        from mlx_lm.utils import load_tokenizer
        _tokenizer = load_tokenizer(Path(mgr.model_path_or_error(key)))
        _tokenizer_key = key
    return _tokenizer


def model_max_context() -> int:
    """The installed model's real context window (from its config.json)."""
    global _model_max_ctx, _model_max_ctx_key
    key = chat_key()
    if _model_max_ctx is None or _model_max_ctx_key != key:
        try:
            cfg = json.loads(
                (Path(mgr.model_path_or_error(key)) / "config.json")
                .read_text(encoding="utf-8")
            )
            _model_max_ctx = int(cfg.get("max_position_embeddings", 32768))
        except Exception:  # noqa: BLE001
            _model_max_ctx = 32768
        _model_max_ctx_key = key
    return _model_max_ctx


def count_tokens(text: str) -> int:
    try:
        return len(_get_tokenizer().encode(text))
    except Exception:  # noqa: BLE001 - heuristic fallback
        cjk = len(_CJK_RE.findall(text))
        return cjk + (len(text) - cjk) // 4 + 1


def context_report(system: str, evidence_text: str, user_prompt: str,
                   max_tokens: int, context_budget: int) -> dict:
    """Estimated combined usage + whether it fits the effective window."""
    budget = min(int(context_budget), model_max_context())
    parts = {
        "system": count_tokens(system or ""),
        "evidence": count_tokens(evidence_text or ""),
        "prompt": count_tokens(user_prompt or ""),
        "output_reserved": int(max_tokens),
    }
    total = sum(parts.values()) + 32  # chat-template overhead
    return {
        **parts,
        "total": total,
        "budget": budget,
        "model_max": model_max_context(),
        "fits": total <= budget,
        "overflow": max(0, total - budget),
    }


def fit_to_budget(chunks: list[dict], system: str, prompt_wo_evidence: str,
                  max_tokens: int, context_budget: int,
                  render_evidence) -> tuple[list[dict], int, list[str]]:
    """Trim retrieved evidence first; only then reduce the output allowance.

    `render_evidence(chunks)` -> the evidence text as it will appear in the
    prompt. Returns (kept_chunks, effective_max_tokens, notes).
    """
    notes: list[str] = []
    chunks = list(chunks)
    report = context_report(system, render_evidence(chunks), prompt_wo_evidence,
                            max_tokens, context_budget)
    while not report["fits"] and chunks:
        # drop the least relevant chunk (list is kept sorted best-first)
        dropped = chunks.pop()
        notes.append(f"dropped evidence chunk from {dropped.get('source', '?')} "
                     f"(score {dropped.get('score', 0):.2f}) to fit the context window")
        report = context_report(system, render_evidence(chunks), prompt_wo_evidence,
                                max_tokens, context_budget)
    eff_max = int(max_tokens)
    if not report["fits"]:
        room = report["budget"] - (report["total"] - report["output_reserved"])
        eff_max = max(MIN_OUTPUT_TOKENS, room)
        notes.append(f"reduced output allowance {max_tokens} → {eff_max} tokens "
                     "(prompt alone nearly fills the context window)")
    return chunks, eff_max, notes


# --------------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------------- #
_VAR_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def render_template(template: str, mapping: dict) -> str:
    def sub(m):
        return str(mapping.get(m.group(1), ""))
    return _VAR_RE.sub(sub, template or "")


def _input_line(label: str, value) -> str:
    value = (str(value) if value is not None else "").strip()
    return f"{label}: {value}\n" if value else ""


def pinyin_without_tones(pinyin: str) -> str:
    decomposed = unicodedata.normalize("NFD", pinyin or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[1-5]", "", stripped).strip()


def build_queries(item: dict, expansion: bool) -> list[str]:
    """Retrieval queries from supplied input only — nothing is invented."""
    word = item.get("hanzi", "")
    queries = [word]
    pinyin = (item.get("pinyin") or "").strip()
    if expansion:
        queries.append(f"{word} 意思 meaning definition")
        if pinyin:
            queries.append(f"{word} {pinyin}")
            no_tones = pinyin_without_tones(pinyin)
            if no_tones and no_tones != pinyin:
                queries.append(f"{word} {no_tones}")
        for key in ("topic", "notes", "expected_pos"):
            v = (item.get(key) or "").strip()
            if v:
                queries.append(f"{word} {v}")
    # dedupe, keep order
    seen, out = set(), []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


# --------------------------------------------------------------------------- #
# Retrieval orchestration
# --------------------------------------------------------------------------- #
def retrieve_evidence(cfg: dict, item: dict) -> tuple[list[dict], dict]:
    """Evidence for one word: general collections + field-specific rules.

    Returns (chunks_sorted_best_first, notes) where every chunk carries
    'fields' (which output fields it was retrieved for) and provenance.
    """
    word = item.get("hanzi", "")
    queries = build_queries(item, bool(cfg.get("query_expansion", True)))
    top_k = int(cfg.get("chunks_per_word", 10))
    common = dict(
        use_rerank=bool(cfg.get("use_rerank", True)),
        min_score=float(cfg.get("min_score", 0.0)),
        exact_word=word,
        exact_first=bool(cfg.get("exact_first", True)),
        max_chars_per_chunk=int(cfg.get("max_chars_per_chunk", 1200)),
    )

    by_id: dict[str, dict] = {}

    def add(chunks, field):
        for c in chunks:
            cid = c.get("chunk_id") or f"{c.get('collection')}:{hash(c['text'])}"
            entry = by_id.setdefault(cid, {**c, "fields": []})
            if field and field not in entry["fields"]:
                entry["fields"].append(field)

    general = [n for n in cfg.get("collections", []) if n in cs.collection_names()]
    if general:
        add(cs.search(general, queries, top_k=top_k, **common), None)

    field_sources = cfg.get("field_sources") or {}
    per_field_k = max(2, top_k // 2)
    for field, colls in field_sources.items():
        colls = [n for n in colls if n in cs.collection_names()]
        if colls:
            add(cs.search(colls, [f"{word} {field}"] + queries[:2],
                          top_k=per_field_k, **common), field)

    # priority order: authoritative + priority list break score ties
    priority = {n: i for i, n in enumerate(cfg.get("collection_priority") or [])}
    authoritative = set(cfg.get("authoritative") or [])

    def rank(c):
        return (
            0 if c.get("exact") else 1,
            0 if c.get("collection") in authoritative else 1,
            priority.get(c.get("collection"), 999),
            -c.get("rerank_score", c.get("score", 0.0)),
        )

    chunks = sorted(by_id.values(), key=rank)

    # cap total evidence tokens (cheapest trim: drop worst-ranked)
    max_ret = int(cfg.get("max_retrieval_tokens", 6000))
    while chunks and sum(count_tokens(c["text"]) for c in chunks) > max_ret:
        chunks.pop()

    notes = _conflict_check(chunks, authoritative, word)
    return chunks, notes


def _conflict_check(chunks: list[dict], authoritative: set, word: str) -> dict:
    """When several authoritative collections give exact evidence for the same
    field, record both excerpts — never silently pick one."""
    conflicts = []
    exact_auth = [c for c in chunks if c.get("exact") and c.get("collection") in authoritative]
    by_field: dict[str, list[dict]] = {}
    for c in exact_auth:
        for f in (c.get("fields") or ["(general)"]):
            by_field.setdefault(f, []).append(c)
    for field, group in by_field.items():
        colls = {c["collection"] for c in group}
        if len(colls) > 1:
            conflicts.append({
                "field": field,
                "collections": sorted(colls),
                "excerpts": [
                    {"collection": c["collection"], "source": c.get("source"),
                     "text": c["text"][:400]} for c in group[:4]
                ],
            })
    return {"conflicts": conflicts}


def render_evidence_text(chunks: list[dict], fields_only: list[str] | None = None) -> str:
    """Numbered [S1]-style evidence block, grouped by requested output field."""
    if not chunks:
        return "(no evidence retrieved from the selected collections)"
    groups: dict[str, list[tuple[int, dict]]] = {}
    for i, c in enumerate(chunks, start=1):
        for f in (c.get("fields") or ["general"]):
            groups.setdefault(f, []).append((i, c))
    if fields_only:
        groups = {f: g for f, g in groups.items() if f in fields_only or f == "general"}
    lines = []
    for field in sorted(groups, key=lambda f: (f != "general", f)):
        header = "General evidence" if field == "general" else f"Evidence for field '{field}'"
        lines.append(f"## {header}")
        for i, c in groups[field]:
            lines.append(f"[S{i}] ({cs.format_source_line(c)})\n{c['text']}")
    return "\n\n".join(lines)


def evidence_provenance(chunks: list[dict]) -> list[dict]:
    """What gets written to sources_used.jsonl — the exact evidence used."""
    return [
        {
            "label": f"S{i}",
            "collection": c.get("collection"),
            "source": c.get("source"),
            "page": c.get("page"),
            "row": c.get("row"),
            "json_path": c.get("json_path"),
            "section": c.get("section"),
            "chunk_id": c.get("chunk_id"),
            "score": round(float(c.get("score", 0.0)), 4),
            "rerank_score": round(float(c["rerank_score"]), 4) if "rerank_score" in c else None,
            "fields": c.get("fields") or [],
            "exact_match": bool(c.get("exact")),
            "text": c.get("text"),
        }
        for i, c in enumerate(chunks, start=1)
    ]


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #
def effective_schema(cfg: dict) -> dict | None:
    if cfg.get("schema_mode") == "schema" and isinstance(cfg.get("json_schema"), dict):
        return cfg["json_schema"]
    if isinstance(cfg.get("example_object"), dict):
        return infer_schema_from_example(cfg["example_object"], cfg.get("required_keys"))
    return None


def system_prompt(cfg: dict) -> str:
    rules = ["You output exactly one valid JSON object and nothing else."]
    if cfg.get("only_sources") and not cfg.get("allow_model_knowledge"):
        rules.append("Use ONLY the supplied source evidence for factual content.")
    elif cfg.get("only_sources"):
        rules.append("Prefer the supplied source evidence; use your own knowledge "
                     "only for fields the evidence does not cover.")
    if cfg.get("omit_unsupported"):
        rules.append("Omit optional fields the evidence does not support instead of guessing.")
    if cfg.get("never_invent_citations"):
        rules.append("Never invent citations, source names or official exam references.")
    return " ".join(rules)


def build_word_prompt(cfg: dict, item: dict, chunks: list[dict],
                      queue_pos: int, total: int,
                      previous_summary: str = "") -> tuple[str, str]:
    """Returns (system, user_prompt) with all template variables filled."""
    schema = effective_schema(cfg)
    schema_text = json.dumps(schema, ensure_ascii=False, indent=1) if schema else \
        "(no schema set — follow the example structure exactly)"
    if cfg.get("schema_mode") == "example" and isinstance(cfg.get("example_object"), dict):
        schema_text += "\n\nExample object:\n" + json.dumps(
            cfg["example_object"], ensure_ascii=False, indent=1)

    auth = ", ".join(cfg.get("authoritative") or []) or "(none marked)"
    mapping = {
        "word": item.get("hanzi", ""),
        "word_index": queue_pos,
        "total_words": total,
        "hsk_level": item.get("hsk_new") or item.get("hsk_old") or "",
        "topic": item.get("topic", ""),
        "input_pinyin": _input_line("Pinyin (from input)", item.get("pinyin")),
        "input_hsk_old": _input_line("HSK 2.0 level (from input)", item.get("hsk_old")),
        "input_hsk_new": _input_line("HSK 3.0 level (from input)", item.get("hsk_new")),
        "input_notes": _input_line("Notes (from input)", item.get("notes")),
        "retrieved_sources": render_evidence_text(chunks),
        "authoritative_sources": auth,
        "json_schema": schema_text,
        "previous_result_summary": previous_summary,
    }
    user = render_template(cfg.get("prompt_template") or gp.DEFAULT_PROMPT, mapping)
    extra = (item.get("custom_instruction") or "").strip()
    if extra:
        user += f"\n\nAdditional instruction for this word: {extra}"
    return system_prompt(cfg), user


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_result(obj: dict, cfg: dict, word: str,
                    n_sources: int) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings). Errors block acceptance; warnings go to review."""
    errors: list[str] = []
    warnings: list[str] = []
    level = cfg.get("validation_level", "strict")
    key = cfg.get("key_field", "hanzi")

    if level in ("schema", "strict", "custom"):
        schema = effective_schema(cfg)
        if schema:
            errors.extend(validate_schema(obj, schema))

    if key in obj and str(obj.get(key)).strip() != word:
        errors.append(f"'{key}' is '{obj.get(key)}' but the queued word is '{word}'")
    elif key not in obj:
        errors.append(f"missing key field '{key}'")

    def walk_strings(node, path="$"):
        if isinstance(node, str):
            yield path, node
        elif isinstance(node, dict):
            for k, v in node.items():
                yield from walk_strings(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from walk_strings(v, f"{path}[{i}]")

    if level in ("strict", "custom"):
        checks = cfg.get("custom_checks") or {}
        for path, s in walk_strings(obj):
            if "```" in s:
                errors.append(f"{path}: contains a markdown fence")
            for m in _SOURCE_LABEL_RE.finditer(s):
                if int(m.group(1)) > n_sources:
                    errors.append(f"{path}: cites source [S{m.group(1)}] "
                                  f"but only {n_sources} sources were supplied")
        if "pinyin" in obj and isinstance(obj["pinyin"], str):
            p = obj["pinyin"].strip()
            if not p:
                errors.append("$.pinyin: empty")
            elif not any(c in _TONE_MARKS for c in p) and not re.search(r"[1-5]", p):
                warnings.append("$.pinyin: no tone marks or tone numbers found")
        senses = obj.get("senses")
        if isinstance(senses, list):
            n = len(senses)
            for path, node in _walk_dicts(obj):
                for ref_key in ("sense", "sense_ref", "sense_id", "sense_index"):
                    v = node.get(ref_key)
                    if isinstance(v, int) and not (0 <= v <= n):
                        errors.append(f"{path}.{ref_key}: {v} is outside 0–{n} "
                                      f"({n} senses defined)")
        pos_allow = checks.get("pos_allowlist")
        if pos_allow:
            for path, node in _walk_dicts(obj):
                v = node.get("pos")
                if isinstance(v, str) and v and v not in pos_allow:
                    warnings.append(f"{path}.pos: '{v}' not in the POS allowlist")
        max_ex = checks.get("max_example_chars")
        if max_ex:
            for path, s in walk_strings(obj):
                if "example" in path and len(s) > int(max_ex):
                    warnings.append(f"{path}: example longer than {max_ex} characters")
        for path, s in walk_strings(obj):
            if _CJK_RE.search(s) and len(s) > 8 and re.search(r"[,.!?]$", s):
                warnings.append(f"{path}: Chinese text ends with Western punctuation")
    return errors, warnings


def _walk_dicts(node, path="$"):
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk_dicts(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_dicts(v, f"{path}[{i}]")


# --------------------------------------------------------------------------- #
# Model calls
# --------------------------------------------------------------------------- #
def _load_chat():
    from mlx_lm import load
    key = chat_key()
    path = mgr.model_path_or_error(key)
    return mm.HEAVY.get(f"chat:{key}", mgr.MODELS[key]["name"], lambda: load(path))


def generate_text(system: str, user: str, cfg: dict,
                  cancel_event: threading.Event | None = None,
                  max_tokens_override: int | None = None) -> tuple[str, str]:
    """One generation call. Returns (text, finish_reason);
    finish_reason ∈ stop | length | cancelled."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    import mlx.core as mx

    model, tokenizer = _load_chat()
    if cfg.get("seed") is not None and str(cfg.get("seed")).strip() != "":
        mx.random.seed(int(cfg["seed"]))

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    sampler = make_sampler(temp=float(cfg.get("temperature", 0.2)),
                           top_p=float(cfg.get("top_p", 0.95)))
    rep = float(cfg.get("repetition_penalty", 1.0) or 1.0)
    processors = make_logits_processors(repetition_penalty=rep) if rep != 1.0 else None

    max_tokens = int(max_tokens_override or cfg.get("max_tokens", DEFAULT_OUTPUT_TOKENS))
    max_tokens = max(MIN_OUTPUT_TOKENS, min(MAX_OUTPUT_TOKENS, max_tokens))

    text = ""
    finish = "stop"
    try:
        for r in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens,
                                 sampler=sampler, logits_processors=processors):
            text += r.text
            finish = r.finish_reason or finish
            if cancel_event is not None and cancel_event.is_set():
                return text, "cancelled"
    finally:
        mm.HEAVY.touch()
    return text, finish or "stop"


# --------------------------------------------------------------------------- #
# One-word pipeline
# --------------------------------------------------------------------------- #
def process_word(pid: str, cfg: dict, item: dict, queue_pos: int, total: int,
                 cancel_event: threading.Event | None = None,
                 previous_summary: str = "") -> dict:
    """Full pipeline for one word. Returns an outcome dict:
    {status: completed|failed|cancelled, object, errors, warnings, raw,
     evidence, notes, seconds, outcome}."""
    word = item.get("hanzi", "")
    t0 = time.time()
    result: dict = {"word": word, "status": "failed", "object": None,
                    "errors": [], "warnings": [], "raw": "", "notes": []}

    chunks, retrieval_notes = retrieve_evidence(cfg, item)
    for conflict in retrieval_notes.get("conflicts", []):
        gp.add_review_note(
            pid, word, "authoritative_conflict",
            f"Authoritative collections disagree for field '{conflict['field']}': "
            f"{', '.join(conflict['collections'])} — higher-priority source preferred, "
            "both excerpts recorded.",
            {"excerpts": conflict["excerpts"]},
        )
        result["notes"].append(f"conflict recorded for field '{conflict['field']}'")

    if not chunks and cfg.get("only_sources") and not cfg.get("allow_model_knowledge"):
        result["errors"] = ["no evidence found in the selected collections and "
                            "'only use supplied sources' is enabled"]
        result["seconds"] = time.time() - t0
        return result

    system, user_full = build_word_prompt(cfg, item, chunks, queue_pos, total,
                                          previous_summary)
    user_wo_evidence = build_word_prompt(cfg, item, [], queue_pos, total,
                                         previous_summary)[1]
    chunks, eff_max, budget_notes = fit_to_budget(
        chunks, system, user_wo_evidence,
        int(cfg.get("max_tokens", DEFAULT_OUTPUT_TOKENS)),
        int(cfg.get("context_budget", 32768)),
        lambda cc: render_evidence_text(cc),
    )
    result["notes"].extend(budget_notes)
    if budget_notes:  # evidence changed — rebuild the final prompt
        system, user_full = build_word_prompt(cfg, item, chunks, queue_pos, total,
                                              previous_summary)
    result["evidence"] = evidence_provenance(chunks)
    result["prompt"] = user_full

    max_model_retries = int(cfg.get("max_retries", 3))
    model_calls = 0
    obj = None
    raw = ""
    errors: list[str] = []
    warnings: list[str] = []

    while model_calls < max(1, max_model_retries):
        model_calls += 1
        raw, finish = generate_text(system, user_full, cfg, cancel_event,
                                    max_tokens_override=eff_max)
        result["raw"] = raw
        if finish == "cancelled":
            result["status"] = "cancelled"
            result["seconds"] = time.time() - t0
            return result

        obj, parse_errors, truncated = extract_and_parse(raw)
        if truncated or finish == "length":
            errors = ["output was cut off at the token limit — increase "
                      "'Maximum output tokens' and regenerate"]
            result["notes"].append("incomplete output (token limit reached)")
            obj = None
            continue  # regenerate (it may finish within the limit next time)

        if obj is not None:
            errors, warnings = validate_result(obj, cfg, word, len(chunks))
            errors = parse_errors + errors if parse_errors else errors
            if not errors:
                break
            # targeted repair: send the object + exact errors back once per cycle
            if model_calls < max(1, max_model_retries):
                repair_user = (
                    "Repair this object only. Do not change correct factual content.\n\n"
                    "JSON object:\n" + json.dumps(obj, ensure_ascii=False, indent=1) +
                    "\n\nValidation errors to fix:\n- " + "\n- ".join(errors[:15])
                )
                model_calls += 1
                raw2, finish2 = generate_text(REPAIR_SYSTEM, repair_user, cfg,
                                              cancel_event, max_tokens_override=eff_max)
                if finish2 == "cancelled":
                    result["status"] = "cancelled"
                    result["seconds"] = time.time() - t0
                    return result
                obj2, parse2, trunc2 = extract_and_parse(raw2)
                if obj2 is not None and not trunc2:
                    errs2, warns2 = validate_result(obj2, cfg, word, len(chunks))
                    if not errs2:
                        obj, errors, warnings, raw = obj2, [], warns2, raw2
                        result["raw"] = raw2
                        break
                # repair failed — loop continues with a full regenerate
        else:
            errors = parse_errors or ["no JSON object found in the model output"]

    result["seconds"] = time.time() - t0
    result["errors"] = errors
    result["warnings"] = warnings
    result["model_calls"] = model_calls
    gp.save_raw(pid, word, result.get("prompt", ""), raw)

    if obj is not None and not errors:
        meta = {
            "generated_by_model": mgr.MODELS[chat_key()]["repo"],
            "prompt_version": cfg.get("prompt_version", 1),
            "source_collection_ids": sorted({c["collection"] for c in result["evidence"]})
            if result.get("evidence") else [],
        }
        outcome = gp.upsert_result(pid, word, obj, meta)
        gp.save_sources_used(pid, word, result["evidence"] or [])
        for w in warnings:
            gp.add_review_note(pid, word, "warning", w)
        result["status"] = "completed"
        result["object"] = obj
        result["outcome"] = outcome
        gp.clear_failure(pid, word)
    else:
        gp.add_failure(pid, word, errors, raw, model_calls)
    return result


# --------------------------------------------------------------------------- #
# Field-only regeneration
# --------------------------------------------------------------------------- #
def regenerate_fields(pid: str, word: str, fields: list[str]) -> dict:
    """Regenerate only `fields` of an existing object; everything else is kept."""
    cfg = gp.get_config(pid)
    results = gp.load_results(pid)
    key = cfg.get("key_field", "hanzi")
    idx = next((i for i, o in enumerate(results)
                if isinstance(o, dict) and str(o.get(key)) == word), None)
    if idx is None:
        return {"status": "failed", "errors": [f"'{word}' is not in results.json"]}
    current = results[idx]

    item = {"hanzi": word}
    chunks, _ = retrieve_evidence({**cfg, "field_sources": {
        f: (cfg.get("field_sources", {}).get(f) or cfg.get("collections", []))
        for f in fields}}, item)
    evidence = render_evidence_text(chunks, fields_only=fields)
    system = system_prompt(cfg)
    user = (
        f"Current JSON object for the word {word}:\n"
        + json.dumps(current, ensure_ascii=False, indent=1)
        + f"\n\nRegenerate ONLY these fields: {', '.join(fields)}.\n"
        "Return ONE JSON object containing ONLY those fields with their new values. "
        "Do not repeat the other fields.\n\nEvidence:\n" + evidence
    )
    raw, finish = generate_text(system, user, cfg)
    obj, errors, truncated = extract_and_parse(raw)
    if obj is None or truncated:
        return {"status": "failed", "errors": errors or ["output truncated"], "raw": raw}

    updated = dict(current)
    for f in fields:
        if f in obj:
            updated[f] = obj[f]
    errs, warns = validate_result(updated, cfg, word, len(chunks))
    if errs:
        return {"status": "failed", "errors": errs, "raw": raw, "object": obj}
    gp.upsert_result(pid, word, updated, gp.get_meta_for(pid, word), mode="replace")
    gp.save_sources_used(pid, word, evidence_provenance(chunks))
    return {"status": "completed", "object": updated, "warnings": warns, "raw": raw}


# --------------------------------------------------------------------------- #
# Queue runner (single background worker)
# --------------------------------------------------------------------------- #
class QueueRunner:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._cancel_current = threading.Event()
        self._skip_current = False
        self._pause = False
        self._stop = False
        self.state: dict = {"status": "idle", "pid": None, "current_word": None,
                            "position": 0, "total": 0, "seconds_per_word": [],
                            "current_started": None, "last_error": None,
                            "memory_note": None}

    # ---- controls -------------------------------------------------------- #
    def start(self, pid: str, only_ids: list[int] | None = None,
              only_statuses: tuple = ("queued",)) -> str:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return "A generation queue is already running — pause or stop it first."
            cfg = gp.get_config(pid)
            if cfg is None:
                return f"Project '{pid}' not found."
            if not cfg.get("collections") and not cfg.get("allow_model_knowledge"):
                if cfg.get("only_sources"):
                    return ("Select at least one source collection (or allow model "
                            "knowledge in the accuracy settings).")
            self._pause = False
            self._stop = False
            self._skip_current = False
            self._cancel_current.clear()
            self.state.update(status="running", pid=pid, last_error=None,
                              memory_note=None)
            self._thread = threading.Thread(
                target=self._run, args=(pid, only_ids, only_statuses), daemon=True)
            self._thread.start()
            return "▶️ Generation started."

    def pause(self) -> str:
        with self._lock:
            if self.state["status"] != "running":
                return "Nothing is running."
            self._pause = True
            return "⏸ Will pause after the current word."

    def resume(self) -> str:
        with self._lock:
            if self.state["status"] == "paused":
                self._pause = False
                self.state["status"] = "running"
                return "▶️ Resumed."
            if self.state["status"] == "running":
                self._pause = False
                return "Already running."
            pid = self.state.get("pid")
        if pid:
            return self.start(pid)
        return "Nothing to resume — pick a project and press Start."

    def stop(self) -> str:
        with self._lock:
            if self.state["status"] not in ("running", "paused"):
                return "Nothing is running."
            self._stop = True
            self._pause = False
            return "⏹ Will stop safely after the current word."

    def cancel_current(self) -> str:
        with self._lock:
            if self.state["status"] != "running":
                return "Nothing is generating."
            self._cancel_current.set()
            return "❌ Cancelling the current generation (the word stays queued)."

    def skip_current(self) -> str:
        with self._lock:
            if self.state["status"] != "running":
                return "Nothing is generating."
            self._skip_current = True
            self._cancel_current.set()
            return "⏭ Skipping the current word."

    def snapshot(self) -> dict:
        with self._lock:
            s = dict(self.state)
        secs = s.pop("seconds_per_word", [])
        s["avg_seconds"] = sum(secs) / len(secs) if secs else None
        if s.get("current_started"):
            s["current_elapsed"] = time.time() - s["current_started"]
        pid = s.get("pid")
        if pid:
            counts = gp.queue_counts(pid)
            s["counts"] = counts
            if s["avg_seconds"] and counts.get("queued"):
                s["eta_seconds"] = counts["queued"] * s["avg_seconds"]
        return s

    # ---- worker ---------------------------------------------------------- #
    def _next_item(self, pid, only_ids, only_statuses):
        queue = gp.load_queue(pid)
        for item in queue:
            if item["status"] in only_statuses and (only_ids is None or item["id"] in only_ids):
                return item
        return None

    def _set_item_status(self, pid, item_id, **updates):
        queue = gp.load_queue(pid)
        for item in queue:
            if item["id"] == item_id:
                item.update(updates)
                break
        gp.save_queue(pid, queue)

    def _run(self, pid: str, only_ids, only_statuses) -> None:
        cfg = gp.get_config(pid)
        gp.log(pid, "queue_start", f"queue started (model {mgr.MODELS[chat_key()]['name']})")
        completed_words: list[str] = []
        try:
            while True:
                if self._stop:
                    break
                if self._pause:
                    with self._lock:
                        self.state["status"] = "paused"
                        self.state["current_word"] = None
                    gp.log(pid, "queue_pause", "paused")
                    while self._pause and not self._stop:
                        time.sleep(0.3)
                    if self._stop:
                        break
                    with self._lock:
                        self.state["status"] = "running"
                    gp.log(pid, "queue_resume", "resumed")

                # memory safety: pause instead of crashing
                key = chat_key()
                need = mgr.MODELS[key].get("ram_gb", 4.0)
                if mm.HEAVY.key != f"chat:{key}":
                    warn = mm.memory_warning(threshold_gb=15.0, need_gb=need)
                    if warn:
                        with self._lock:
                            self.state.update(status="paused", memory_note=warn)
                        self._pause = True
                        gp.log(pid, "memory_pause", warn)
                        continue

                item = self._next_item(pid, only_ids, only_statuses)
                if item is None:
                    break

                counts = gp.queue_counts(pid)
                position = counts["completed"] + counts["failed"] + counts["skipped"] + 1
                with self._lock:
                    self.state.update(current_word=item["hanzi"], position=position,
                                      total=counts["total"], current_started=time.time())
                self._cancel_current.clear()
                self._skip_current = False
                self._set_item_status(pid, item["id"], status="running")
                gp.log(pid, "word_start", f"{item['hanzi']} ({position}/{counts['total']})")
                mm.set_task(f"Generator: {item['hanzi']} ({position}/{counts['total']})")

                try:
                    prev = ", ".join(completed_words[-3:])
                    outcome = process_word(pid, cfg, item, position, counts["total"],
                                           cancel_event=self._cancel_current,
                                           previous_summary=prev)
                except Exception as exc:  # noqa: BLE001 - keep the queue alive
                    outcome = {"status": "failed", "errors": [f"unexpected error: {exc}"],
                               "raw": "", "seconds": 0.0}
                    gp.add_failure(pid, item["hanzi"], outcome["errors"], "", 0)
                finally:
                    mm.set_task("idle")

                item_after = dict(attempts=item.get("attempts", 0) + 1)
                if outcome["status"] == "completed":
                    item_after.update(status="completed", error=None)
                    completed_words.append(item["hanzi"])
                elif outcome["status"] == "cancelled":
                    if self._skip_current:
                        item_after.update(status="skipped", error="skipped by user")
                    elif self._stop:
                        item_after.update(status="queued", error=None)
                    else:
                        item_after.update(status="queued", error=None)
                        self._pause = True  # cancel = hold the queue for the user
                else:
                    item_after.update(status="failed",
                                      error="; ".join(outcome.get("errors", []))[:500])
                self._set_item_status(pid, item["id"], **item_after)

                secs = outcome.get("seconds", 0.0)
                with self._lock:
                    self.state["seconds_per_word"] = (
                        self.state["seconds_per_word"][-19:] + [secs])
                    self.state["current_started"] = None
                    self.state["last_outcome"] = {
                        "word": item["hanzi"],
                        "status": item_after["status"],
                        "errors": outcome.get("errors", [])[:8],
                        "warnings": outcome.get("warnings", [])[:8],
                        "notes": outcome.get("notes", [])[:8],
                        "object": outcome.get("object"),
                        "evidence": (outcome.get("evidence") or [])[:12],
                        "seconds": secs,
                    }
                gp.log(pid, f"word_{item_after['status']}",
                       f"{item['hanzi']}: {item_after['status']} in {secs:.1f}s"
                       + (f" — {item_after.get('error')}" if item_after.get("error") else ""))
                gp.save_progress(pid, {
                    "last_word": item["hanzi"],
                    "last_status": item_after["status"],
                    "counts": gp.queue_counts(pid),
                    "avg_seconds": (sum(self.state["seconds_per_word"]) /
                                    len(self.state["seconds_per_word"]))
                    if self.state["seconds_per_word"] else None,
                })
        finally:
            with self._lock:
                self.state.update(status="idle", current_word=None, current_started=None)
            counts = gp.queue_counts(pid)
            gp.log(pid, "queue_end",
                   f"queue stopped — {counts['completed']} completed, "
                   f"{counts['failed']} failed, {counts['skipped']} skipped, "
                   f"{counts['queued']} still queued")


RUNNER = QueueRunner()
