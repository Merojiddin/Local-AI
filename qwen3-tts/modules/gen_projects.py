"""Generator projects: per-project config, queue, results and recovery.

Each project lives in its own folder, fully separate from documents and models:

  data/generator_projects/<project_id>/
    config.json            all settings (collections, prompt, schema, …)
    queue.json             word queue with per-item status (survives restarts)
    progress.json          checkpoint written after every word
    results.json           ONE valid JSON array of plain result objects
    results_meta.json      per-word metadata kept OUTSIDE the exported objects
    failures.json          words that could not be generated after retries
    review_report.json     conflicts / unsupported-field notes for review
    sources_used.jsonl     exact evidence used for every word
    logs/generation_log.jsonl
    objects/<word>.json    every valid word saved separately (rebuild source)
    backups/               timestamped results.json backups
    exports/

results.json is only ever replaced atomically after the new content has been
validated, with a timestamped backup taken first — it stays a valid JSON
array after every completed word.
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

from . import storage
from .safe_json import (
    append_jsonl,
    atomic_write_json,
    latest_valid_backup,
    load_json,
    read_jsonl,
    sanitize_filename,
    timestamped_backup,
)

DEFAULT_PROMPT = """You are a careful Chinese lexicographer. Create ONE dictionary entry for the word below, as ONE valid JSON object and nothing else — no markdown fences, no commentary.

Word: {{word}}  ({{word_index}} of {{total_words}})
{{input_pinyin}}{{input_hsk_old}}{{input_hsk_new}}{{input_notes}}

Follow this JSON schema exactly:
{{json_schema}}

Evidence from my source documents (grouped by field, with provenance):
{{retrieved_sources}}

Rules:
- Use ONLY the evidence above for facts it covers. Do not invent citations or exam sources.
- If the evidence does not support an optional field, omit that field.
- The "hanzi" value must be exactly {{word}}.
- Return exactly one JSON object.
"""

MODE_PRESETS = {
    "Dictionary entry": {},
    "Example sentences": {},
    "Grammar explanation": {},
    "Collocations": {},
    "Quiz questions": {},
    "Translation": {},
    "Custom structured JSON": {},
}

DEFAULT_CONFIG = {
    "name": "",
    "description": "",
    "mode": "Dictionary entry",
    "created": None,
    "updated": None,
    # sources
    "collections": [],
    "collection_priority": [],   # highest first
    "authoritative": [],
    "field_sources": {},         # {"examples": ["Example sentences", …]}
    # prompt & schema
    "prompt_template": DEFAULT_PROMPT,
    "prompt_version": 1,
    "template_description": "",
    "schema_mode": "example",    # "schema" | "example"
    "json_schema": None,
    "example_object": None,
    "required_keys": None,       # example mode: which keys are required
    "key_field": "hanzi",
    # generation
    "max_tokens": 8192,
    "temperature": 0.2,
    "top_p": 0.95,
    "repetition_penalty": 1.0,
    "seed": None,
    "context_budget": 32768,
    "max_retries": 3,
    # retrieval
    "chunks_per_word": 10,
    "max_chars_per_chunk": 1200,
    "max_retrieval_tokens": 6000,
    "use_rerank": True,
    "min_score": 0.0,
    "query_expansion": True,
    "exact_first": True,
    "include_related": True,
    # accuracy
    "only_sources": True,
    "allow_model_knowledge": False,
    "omit_unsupported": True,
    "never_invent_citations": True,
    # validation
    "validation_level": "strict",   # basic | schema | strict | custom
    "custom_checks": {},
    # results
    "duplicate_mode": "merge",      # skip | replace | merge | review | ask
}


# --------------------------------------------------------------------------- #
# Project CRUD
# --------------------------------------------------------------------------- #
def _proj_dir(pid: str) -> Path:
    return storage.generator_dir() / pid


def _pfile(pid: str, name: str) -> Path:
    return _proj_dir(pid) / name


def list_projects() -> list[dict]:
    root = storage.generator_dir()
    out = []
    if root.exists():
        for p in sorted(root.iterdir()):
            cfg = load_json(p / "config.json")
            if p.is_dir() and isinstance(cfg, dict):
                cfg["project_id"] = p.name
                out.append(cfg)
    return out


def project_ids() -> list[dict]:
    """[(display name, project_id)] for dropdowns."""
    return [(f"{c['name']}", c["project_id"]) for c in list_projects()]


def get_config(pid: str) -> dict | None:
    cfg = load_json(_pfile(pid, "config.json"))
    if not isinstance(cfg, dict):
        return None
    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)
    merged["project_id"] = pid
    return merged


def save_config(pid: str, cfg: dict) -> None:
    cfg = {k: v for k, v in cfg.items() if k != "project_id"}
    cfg["updated"] = datetime.now().isoformat(timespec="seconds")
    atomic_write_json(_pfile(pid, "config.json"), cfg)


def create_project(name: str, description: str = "", mode: str = "Dictionary entry") -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("Give the project a name.")
    pid = sanitize_filename(name)
    target = _proj_dir(pid)
    if (target / "config.json").exists():
        raise ValueError(f"A project folder '{pid}' already exists.")
    for sub in ("logs", "objects", "backups", "exports"):
        (target / sub).mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(name=name, description=description.strip(), mode=mode,
               created=datetime.now().isoformat(timespec="seconds"))
    save_config(pid, cfg)
    atomic_write_json(_pfile(pid, "queue.json"), [])
    atomic_write_json(_pfile(pid, "results.json"), [])
    atomic_write_json(_pfile(pid, "results_meta.json"), {})
    atomic_write_json(_pfile(pid, "failures.json"), [])
    atomic_write_json(_pfile(pid, "review_report.json"), [])
    return pid


def rename_project(pid: str, new_name: str) -> None:
    cfg = get_config(pid)
    if cfg is None:
        raise ValueError(f"Project '{pid}' not found.")
    new_name = (new_name or "").strip()
    if not new_name:
        raise ValueError("New name is empty.")
    cfg["name"] = new_name  # folder id stays stable so running queues never break
    save_config(pid, cfg)


def duplicate_project(pid: str, new_name: str) -> str:
    cfg = get_config(pid)
    if cfg is None:
        raise ValueError(f"Project '{pid}' not found.")
    new_pid = create_project(new_name, cfg.get("description", ""), cfg.get("mode", ""))
    fresh = get_config(new_pid)
    keep_new = {"name", "created", "updated", "project_id"}
    for k, v in cfg.items():
        if k not in keep_new:
            fresh[k] = v
    save_config(new_pid, fresh)
    return new_pid


def delete_project(pid: str) -> str:
    target = _proj_dir(pid)
    root = storage.generator_dir()
    if root not in target.parents:
        raise ValueError("Refusing to delete outside the generator folder.")
    if not target.exists():
        return f"Project '{pid}' does not exist."
    shutil.rmtree(target)
    return f"🗑 Deleted project '{pid}' (documents and models untouched)."


def export_config(pid: str) -> Path:
    cfg = get_config(pid)
    if cfg is None:
        raise ValueError(f"Project '{pid}' not found.")
    out = _pfile(pid, "exports") / f"{pid}_config_{_stamp()}.json"
    atomic_write_json(out, {k: v for k, v in cfg.items() if k != "project_id"})
    return out


def import_config(path: str, new_name: str | None = None) -> str:
    data = load_json(Path(path))
    if not isinstance(data, dict) or "prompt_template" not in data:
        raise ValueError("Not a generator project config file.")
    name = (new_name or "").strip() or f"{data.get('name', 'Imported')} (imported)"
    pid = create_project(name, data.get("description", ""), data.get("mode", ""))
    cfg = get_config(pid)
    for k, v in data.items():
        if k not in ("name", "created", "updated", "project_id"):
            cfg[k] = v
    save_config(pid, cfg)
    return pid


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# --------------------------------------------------------------------------- #
# Word-list input parsing
# --------------------------------------------------------------------------- #
INPUT_COLUMNS = ("hanzi", "pinyin", "hsk_old", "hsk_new", "topic", "notes",
                 "expected_pos", "custom_instruction")
_SPLIT_RE = re.compile(r"[\n,，、;；]+")
_HAS_HANZI = re.compile(r"[一-鿿㐀-䶿]")


def rows_from_text(text: str) -> list[dict]:
    """One word per line, or comma / Chinese-enumeration separated."""
    rows = []
    for token in _SPLIT_RE.split(text or ""):
        word = re.sub(r"\s+", " ", token).strip()
        if word:
            rows.append({"hanzi": word})
    return rows


def rows_from_file(path: str) -> list[dict]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".txt":
        return rows_from_text(p.read_text(encoding="utf-8", errors="replace"))
    if suffix == ".csv":
        with open(p, encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "hanzi" not in [c.strip().lower() for c in reader.fieldnames]:
                # no header — treat first column as hanzi
                f.seek(0)
                plain = csv.reader(f)
                return [{"hanzi": r[0].strip()} for r in plain if r and r[0].strip()]
            rows = []
            for r in reader:
                row = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
                row = {k: v for k, v in row.items() if k in INPUT_COLUMNS and v}
                if row.get("hanzi"):
                    rows.append(row)
            return rows
    if suffix == ".json":
        data = load_json(p)
        rows = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str) and item.strip():
                    rows.append({"hanzi": item.strip()})
                elif isinstance(item, dict) and str(item.get("hanzi", "")).strip():
                    rows.append({k: str(v).strip() for k, v in item.items()
                                 if k in INPUT_COLUMNS and str(v).strip()})
        return rows
    if suffix == ".xlsx":
        raise ValueError("XLSX is not supported (openpyxl is not installed). "
                         "Export the sheet as CSV instead.")
    raise ValueError(f"Unsupported word-list file: {p.name} (use TXT, CSV or JSON).")


def analyze_rows(rows: list[dict], existing_words: set[str]) -> dict:
    """Normalize, drop blanks, find duplicates; original order preserved."""
    valid, dup_input, dup_existing, invalid = [], [], [], []
    seen = set()
    for r in rows:
        word = re.sub(r"\s+", " ", str(r.get("hanzi", ""))).strip()
        if not word:
            invalid.append(r)
            continue
        row = {k: str(v).strip() for k, v in r.items() if k in INPUT_COLUMNS and str(v).strip()}
        row["hanzi"] = word
        if word in seen:
            dup_input.append(word)
            continue
        seen.add(word)
        if word in existing_words:
            dup_existing.append(word)
            continue
        valid.append(row)
    return {"valid": valid, "duplicate_in_input": dup_input,
            "duplicate_existing": dup_existing, "invalid": invalid}


# --------------------------------------------------------------------------- #
# Queue
# --------------------------------------------------------------------------- #
def load_queue(pid: str) -> list[dict]:
    q = load_json(_pfile(pid, "queue.json"), default=[])
    return q if isinstance(q, list) else []


def save_queue(pid: str, queue: list[dict]) -> None:
    atomic_write_json(_pfile(pid, "queue.json"), queue)


def enqueue(pid: str, rows: list[dict], allow_duplicates: bool = False) -> dict:
    queue = load_queue(pid)
    queued_words = {item["hanzi"] for item in queue}
    result_words = set(results_keys(pid))
    existing = set() if allow_duplicates else queued_words | result_words
    report = analyze_rows(rows, existing)
    next_id = max([item.get("id", 0) for item in queue], default=0)
    for row in report["valid"]:
        next_id += 1
        item = dict(row)
        item.update(id=next_id, status="queued", attempts=0, error=None,
                    added=datetime.now().isoformat(timespec="seconds"))
        queue.append(item)
    save_queue(pid, queue)
    report["queued_total"] = len(queue)
    return report


def remove_from_queue(pid: str, ids: list[int]) -> int:
    queue = load_queue(pid)
    keep = [i for i in queue if i["id"] not in set(ids) or i["status"] == "running"]
    removed = len(queue) - len(keep)
    save_queue(pid, keep)
    return removed


def reorder_queue(pid: str, ordered_ids: list[int]) -> None:
    queue = load_queue(pid)
    by_id = {i["id"]: i for i in queue}
    ordered = [by_id[i] for i in ordered_ids if i in by_id]
    rest = [i for i in queue if i["id"] not in set(ordered_ids)]
    save_queue(pid, ordered + rest)


def reset_items(pid: str, ids: list[int] | None = None,
                statuses: tuple = ("failed",)) -> int:
    """Put failed/skipped/completed items back to 'queued' for reprocessing."""
    queue = load_queue(pid)
    n = 0
    for item in queue:
        if (ids is None or item["id"] in set(ids)) and item["status"] in statuses:
            item.update(status="queued", error=None)
            n += 1
    save_queue(pid, queue)
    return n


def queue_counts(pid: str) -> dict:
    queue = load_queue(pid)
    counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0, "skipped": 0}
    for item in queue:
        counts[item.get("status", "queued")] = counts.get(item.get("status", "queued"), 0) + 1
    counts["total"] = len(queue)
    return counts


# --------------------------------------------------------------------------- #
# Results (always a valid JSON array on disk)
# --------------------------------------------------------------------------- #
def load_results(pid: str) -> list:
    path = _pfile(pid, "results.json")
    data = load_json(path)
    if isinstance(data, list):
        return data
    # corrupted or missing: recover from the newest valid backup
    backup, restored = latest_valid_backup(path, _pfile(pid, "backups"))
    if isinstance(restored, list):
        log(pid, "recovery", f"results.json unreadable — restored from {backup.name}")
        atomic_write_json(path, restored)
        return restored
    return []


def results_keys(pid: str, key_field: str | None = None) -> list[str]:
    cfg = get_config(pid) or {}
    key = key_field or cfg.get("key_field", "hanzi")
    return [str(o.get(key, "")) for o in load_results(pid) if isinstance(o, dict)]


def _merge_missing(existing: dict, new: dict) -> dict:
    """Fill fields missing/empty in `existing` from `new`; never overwrite."""
    merged = dict(existing)
    for k, v in new.items():
        cur = merged.get(k)
        if k not in merged or cur in (None, "", [], {}):
            merged[k] = v
    return merged


def upsert_result(pid: str, word: str, obj: dict, meta: dict,
                  mode: str | None = None) -> str:
    """Append or update one object in results.json — atomically, with backup.

    mode: skip | replace | merge | review (None = project setting).
    Returns what happened: "appended" | "skipped" | "replaced" | "merged" |
    "review_candidate".
    """
    cfg = get_config(pid) or {}
    key = cfg.get("key_field", "hanzi")
    mode = mode or cfg.get("duplicate_mode", "merge")
    path = _pfile(pid, "results.json")

    results = load_results(pid)
    if path.exists():
        timestamped_backup(path, _pfile(pid, "backups"))

    all_meta = load_json(_pfile(pid, "results_meta.json"), default={}) or {}
    now = datetime.now().isoformat(timespec="seconds")
    idx = next((i for i, o in enumerate(results)
                if isinstance(o, dict) and str(o.get(key)) == word), None)

    outcome = "appended"
    if idx is None:
        results.append(obj)
        all_meta[word] = {**meta, "created_at": now, "updated_at": now,
                          "review_status": "unreviewed"}
    else:
        prev_meta = all_meta.get(word, {})
        reviewed = prev_meta.get("review_status") == "approved"
        if mode == "skip":
            outcome = "skipped"
        elif mode == "replace" and not reviewed:
            results[idx] = obj
            outcome = "replaced"
        elif mode == "review" or (mode == "replace" and reviewed):
            add_review_note(pid, word, "duplicate",
                            "New generation kept as review candidate; existing entry untouched.",
                            {"candidate": obj})
            outcome = "review_candidate"
        else:  # merge (default): fill missing fields, keep reviewed content
            results[idx] = _merge_missing(results[idx], obj)
            outcome = "merged"
        if outcome in ("replaced", "merged"):
            all_meta[word] = {**prev_meta, **meta, "updated_at": now,
                              "created_at": prev_meta.get("created_at", now),
                              "review_status": prev_meta.get("review_status", "unreviewed")}

    if outcome != "skipped":
        atomic_write_json(path, results)
        atomic_write_json(_pfile(pid, "results_meta.json"), all_meta)
    if outcome in ("appended", "replaced", "merged"):
        save_object_file(pid, word, obj, all_meta.get(word, {}))
    return outcome


def save_object_file(pid: str, word: str, obj: dict, meta: dict) -> Path:
    out = _pfile(pid, "objects") / f"{sanitize_filename(word)}.json"
    atomic_write_json(out, {"key": word, "data": obj, "meta": meta})
    return out


def rebuild_results_from_objects(pid: str) -> str:
    """Recovery: reconstruct results.json from the per-word object files."""
    objects_dir = _pfile(pid, "objects")
    cfg = get_config(pid) or {}
    key = cfg.get("key_field", "hanzi")
    items = []
    for f in sorted(objects_dir.glob("*.json")):
        data = load_json(f)
        if isinstance(data, dict) and isinstance(data.get("data"), dict):
            items.append((data.get("meta", {}).get("created_at", ""), data["key"], data["data"]))
    items.sort()
    seen, results = set(), []
    for _, word, obj in items:
        k = str(obj.get(key, word))
        if k in seen:
            continue
        seen.add(k)
        results.append(obj)
    path = _pfile(pid, "results.json")
    timestamped_backup(path, _pfile(pid, "backups"))
    atomic_write_json(path, results)
    return f"Rebuilt results.json from {len(results)} object file(s)."


def restore_results_backup(pid: str) -> str:
    path = _pfile(pid, "results.json")
    backup, data = latest_valid_backup(path, _pfile(pid, "backups"))
    if backup is None:
        return "No valid backup found."
    atomic_write_json(path, data)
    return f"Restored results.json from {backup.name} ({len(data)} objects)."


def consistency_check(pid: str) -> str:
    cfg = get_config(pid) or {}
    key = cfg.get("key_field", "hanzi")
    results = load_results(pid)
    keys = [str(o.get(key, "")) for o in results if isinstance(o, dict)]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    completed = {i["hanzi"] for i in load_queue(pid) if i["status"] == "completed"}
    missing = sorted(completed - set(keys))
    lines = [f"results.json: {len(results)} objects, {len(set(keys))} unique '{key}' keys."]
    if dupes:
        lines.append(f"⚠️ duplicate keys in results: {', '.join(dupes[:20])}")
    if missing:
        lines.append(f"⚠️ completed in queue but missing from results: {', '.join(missing[:20])}")
    if not dupes and not missing:
        lines.append("✅ queue and results are consistent, no duplicate keys.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Failures / review / logs / sources
# --------------------------------------------------------------------------- #
def add_failure(pid: str, word: str, errors: list[str], raw: str, attempts: int) -> None:
    path = _pfile(pid, "failures.json")
    failures = load_json(path, default=[]) or []
    failures = [f for f in failures if f.get("hanzi") != word]
    failures.append({
        "hanzi": word, "errors": errors[:20], "raw_response": raw[-8000:],
        "attempts": attempts, "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    atomic_write_json(path, failures)


def load_failures(pid: str) -> list[dict]:
    return load_json(_pfile(pid, "failures.json"), default=[]) or []


def clear_failure(pid: str, word: str) -> None:
    path = _pfile(pid, "failures.json")
    failures = [f for f in (load_json(path, default=[]) or []) if f.get("hanzi") != word]
    atomic_write_json(path, failures)


def add_review_note(pid: str, word: str, kind: str, detail: str, extra: dict | None = None) -> None:
    path = _pfile(pid, "review_report.json")
    notes = load_json(path, default=[]) or []
    note = {"hanzi": word, "type": kind, "detail": detail,
            "timestamp": datetime.now().isoformat(timespec="seconds")}
    if extra:
        note.update(extra)
    notes.append(note)
    atomic_write_json(path, notes)


def load_review_notes(pid: str, word: str | None = None) -> list[dict]:
    notes = load_json(_pfile(pid, "review_report.json"), default=[]) or []
    return [n for n in notes if word is None or n.get("hanzi") == word]


def set_review_status(pid: str, word: str, status: str) -> None:
    path = _pfile(pid, "results_meta.json")
    meta = load_json(path, default={}) or {}
    meta.setdefault(word, {})["review_status"] = status
    meta[word]["updated_at"] = datetime.now().isoformat(timespec="seconds")
    atomic_write_json(path, meta)


def get_meta_for(pid: str, word: str) -> dict:
    return (load_json(_pfile(pid, "results_meta.json"), default={}) or {}).get(word, {})


def save_sources_used(pid: str, word: str, sources: list[dict]) -> None:
    append_jsonl(_pfile(pid, "sources_used.jsonl"),
                 {"hanzi": word, "timestamp": datetime.now().isoformat(timespec="seconds"),
                  "sources": sources})


def sources_for(pid: str, word: str) -> list[dict]:
    rows = [r for r in read_jsonl(_pfile(pid, "sources_used.jsonl")) if r.get("hanzi") == word]
    return rows[-1]["sources"] if rows else []


def log(pid: str, kind: str, message: str, **extra) -> None:
    entry = {"time": datetime.now().isoformat(timespec="seconds"), "kind": kind,
             "message": message}
    entry.update(extra)
    append_jsonl(_pfile(pid, "logs") / "generation_log.jsonl", entry)


def read_log_tail(pid: str, n: int = 40) -> list[dict]:
    return read_jsonl(_pfile(pid, "logs") / "generation_log.jsonl")[-n:]


def save_raw(pid: str, word: str, prompt: str, raw: str) -> None:
    """Keep the exact final prompt and raw model response for the review panel."""
    d = _pfile(pid, "logs") / "raw"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sanitize_filename(word)}.txt").write_text(
        "=== FINAL PROMPT ===\n" + (prompt or "") +
        "\n\n=== RAW MODEL RESPONSE ===\n" + (raw or ""),
        encoding="utf-8",
    )


def load_raw(pid: str, word: str) -> str:
    p = _pfile(pid, "logs") / "raw" / f"{sanitize_filename(word)}.txt"
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return "(no raw response recorded for this word)"


def save_progress(pid: str, progress: dict) -> None:
    progress["saved"] = datetime.now().isoformat(timespec="seconds")
    atomic_write_json(_pfile(pid, "progress.json"), progress)


def load_progress(pid: str) -> dict:
    return load_json(_pfile(pid, "progress.json"), default={}) or {}


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
def _flatten(obj: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key + "."))
        elif isinstance(v, list):
            flat[key] = json.dumps(v, ensure_ascii=False)
        else:
            flat[key] = v
    return flat


def export_results(pid: str, fmt: str) -> Path:
    results = load_results(pid)
    exports = _pfile(pid, "exports")
    exports.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    if fmt == "json":
        out = exports / f"results_{stamp}.json"
        atomic_write_json(out, results)
    elif fmt == "jsonl":
        out = exports / f"results_{stamp}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for obj in results:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    elif fmt == "csv":
        out = exports / f"results_{stamp}.csv"
        flat_rows = [_flatten(o) for o in results if isinstance(o, dict)]
        headers: list[str] = []
        for r in flat_rows:
            for k in r:
                if k not in headers:
                    headers.append(k)
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(flat_rows)
    elif fmt == "zip":
        out = exports / f"project_{stamp}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("results.json", "failures.json", "review_report.json",
                         "sources_used.jsonl", "config.json", "results_meta.json"):
                p = _pfile(pid, name)
                if p.exists():
                    z.write(p, name)
            logp = _pfile(pid, "logs") / "generation_log.jsonl"
            if logp.exists():
                z.write(logp, "generation_log.jsonl")
    else:
        raise ValueError(f"Unknown export format: {fmt}")
    return out
