"""Crash-safe JSON persistence and model-output JSON handling for the Generator.

Everything here is dependency-free (stdlib only) and side-effect free except
for the explicit file helpers. Used by the Generator tab for:

  * atomic writes with validation + timestamped backups (results.json never
    ends up half-written, even on crash / power loss)
  * extracting the first JSON object out of raw model output (fences,
    <think> blocks, leading/trailing commentary)
  * deterministic, string-aware syntax repair (trailing commas, Python
    literals) that never touches text inside string values
  * a compact JSON-Schema subset validator (jsonschema is not installed;
    this covers the subset needed for dictionary-style objects)
  * conservative schema inference from one example object
  * safe filenames for per-word object files (objects/爱.json)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

try:  # cross-process safety for the results file; filelock ships with hf-hub
    from filelock import FileLock
except ImportError:  # pragma: no cover
    FileLock = None


# --------------------------------------------------------------------------- #
# Atomic file IO
# --------------------------------------------------------------------------- #
def _lock(path: Path):
    class _Null:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    if FileLock is None:
        return _Null()
    return FileLock(str(path) + ".lock", timeout=30)


def atomic_write_json(path: Path, data, indent: int = 2) -> None:
    """Write JSON to a temp file, validate it, then atomically replace `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock(path):
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.stem + ".", suffix=".tmp", dir=str(path.parent)
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            json.loads(tmp.read_text(encoding="utf-8"))  # verify before replacing
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)


def load_json(path: Path, default=None):
    """Read JSON, returning `default` when missing or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def timestamped_backup(path: Path, backups_dir: Path, keep: int = 20) -> Path | None:
    """Copy `path` into backups_dir with a timestamp; prune old backups."""
    path = Path(path)
    if not path.exists():
        return None
    backups_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = backups_dir / f"{path.stem}.{stamp}{path.suffix}"
    if target.exists():  # same-second collision
        target = backups_dir / f"{path.stem}.{stamp}_{os.getpid()}{path.suffix}"
    shutil.copy2(path, target)
    old = sorted(backups_dir.glob(f"{path.stem}.*{path.suffix}"))
    for stale in old[:-keep]:
        stale.unlink(missing_ok=True)
    return target


def append_jsonl(path: Path, obj) -> None:
    """Append one JSON line (UTF-8, no escaping of CJK)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list:
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        pass
    return rows


def latest_valid_backup(path: Path, backups_dir: Path):
    """Return (backup_path, data) of the newest backup that parses, else (None, None)."""
    path = Path(path)
    if not backups_dir.is_dir():
        return None, None
    for cand in sorted(backups_dir.glob(f"{path.stem}.*{path.suffix}"), reverse=True):
        data = load_json(cand)
        if data is not None:
            return cand, data
    return None, None


# --------------------------------------------------------------------------- #
# Extracting JSON from raw model output
# --------------------------------------------------------------------------- #
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> tuple[str | None, str | None]:
    """Find the first complete top-level JSON object in `text`.

    Returns (json_string, error). error is "truncated" when an object opens
    but never closes (output was cut off), "not_found" when there is no '{'.
    """
    if not text:
        return None, "not_found"
    text = _THINK_RE.sub("", text)
    fenced = _FENCE_RE.search(text)
    if fenced and "{" in fenced.group(1):
        text = fenced.group(1)
    start = text.find("{")
    if start == -1:
        return None, "not_found"
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1], None
    return None, "truncated"


def _map_outside_strings(text: str, fn) -> str:
    """Rebuild `text`, passing non-string segments through fn, strings verbatim."""
    parts = []
    buf_start = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                parts.append(text[buf_start : i + 1])  # the whole string literal
                buf_start = i + 1
        else:
            if ch == '"':
                parts.append(fn(text[buf_start:i]))
                buf_start = i
                in_string = True
    parts.append(text[buf_start:] if in_string else fn(text[buf_start:]))
    return "".join(parts)


def repair_json_syntax(text: str) -> str:
    """Deterministic, content-preserving syntax repair.

    Only touches characters *outside* string values: trailing commas,
    Python/JS literals (True/False/None/undefined/NaN/Infinity). Never
    rewrites quotes or anything inside strings, so Chinese quotation marks
    in content are safe.
    """

    def fix(segment: str) -> str:
        segment = re.sub(r",\s*([}\]])", r"\1", segment)  # trailing commas
        segment = re.sub(r"\bTrue\b", "true", segment)
        segment = re.sub(r"\bFalse\b", "false", segment)
        segment = re.sub(r"\b(?:None|undefined|NaN)\b", "null", segment)
        segment = re.sub(r"-?\bInfinity\b", "null", segment)
        return segment

    return _map_outside_strings(text, fix)


def parse_object_strict(text: str) -> tuple[dict | None, list[str]]:
    """Parse one JSON object; report duplicate keys and non-object results."""
    dup: list[str] = []

    def hook(pairs):
        seen = {}
        for k, v in pairs:
            if k in seen:
                dup.append(str(k))
            seen[k] = v
        return seen

    try:
        obj = json.loads(text, object_pairs_hook=hook)
    except ValueError as exc:
        return None, [f"invalid JSON: {exc}"]
    errors = [f"duplicate key: {k}" for k in dup]
    if not isinstance(obj, dict):
        errors.append(f"expected one JSON object, got {type(obj).__name__}")
        return None, errors
    return obj, errors


def extract_and_parse(raw: str) -> tuple[dict | None, list[str], bool]:
    """Full pipeline: extract → parse → deterministic repair → parse.

    Returns (object, errors, truncated).
    """
    snippet, err = extract_json_object(raw)
    if snippet is None:
        return None, [f"no JSON object found ({err})"], err == "truncated"
    obj, errors = parse_object_strict(snippet)
    if obj is not None:
        return obj, errors, False
    repaired = repair_json_syntax(snippet)
    obj2, errors2 = parse_object_strict(repaired)
    if obj2 is not None:
        return obj2, errors2, False
    return None, errors, False


# --------------------------------------------------------------------------- #
# JSON-Schema subset validation (jsonschema package is not installed)
# --------------------------------------------------------------------------- #
_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def validate_schema(obj, schema: dict, path: str = "$") -> list[str]:
    """Validate against the common JSON-Schema subset.

    Supported: type, properties, required, items, enum, const, minLength,
    maxLength, minimum, maximum, minItems, maxItems, pattern,
    additionalProperties (boolean form). Unknown keywords are ignored.
    """
    errors: list[str] = []
    if not isinstance(schema, dict):
        return errors

    stype = schema.get("type")
    if stype is not None:
        types = stype if isinstance(stype, list) else [stype]
        ok = False
        for t in types:
            py = _TYPES.get(t)
            if py is None:
                continue
            if t == "number" and isinstance(obj, bool):
                continue
            if t == "integer" and isinstance(obj, bool):
                continue
            if isinstance(obj, py):
                ok = True
                break
        if not ok:
            errors.append(f"{path}: expected {stype}, got {type(obj).__name__}")
            return errors  # deeper checks are meaningless on the wrong type

    if "enum" in schema and obj not in schema["enum"]:
        errors.append(f"{path}: value {obj!r} not in enum {schema['enum']!r}")
    if "const" in schema and obj != schema["const"]:
        errors.append(f"{path}: value {obj!r} != const {schema['const']!r}")

    if isinstance(obj, str):
        if "minLength" in schema and len(obj) < schema["minLength"]:
            errors.append(f"{path}: string shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(obj) > schema["maxLength"]:
            errors.append(f"{path}: string longer than maxLength {schema['maxLength']}")
        if "pattern" in schema:
            try:
                if not re.search(schema["pattern"], obj):
                    errors.append(f"{path}: does not match pattern {schema['pattern']!r}")
            except re.error:
                pass

    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if "minimum" in schema and obj < schema["minimum"]:
            errors.append(f"{path}: {obj} < minimum {schema['minimum']}")
        if "maximum" in schema and obj > schema["maximum"]:
            errors.append(f"{path}: {obj} > maximum {schema['maximum']}")

    if isinstance(obj, list):
        if "minItems" in schema and len(obj) < schema["minItems"]:
            errors.append(f"{path}: fewer than minItems {schema['minItems']}")
        if "maxItems" in schema and len(obj) > schema["maxItems"]:
            errors.append(f"{path}: more than maxItems {schema['maxItems']}")
        items = schema.get("items")
        if isinstance(items, dict):
            for i, el in enumerate(obj):
                errors.extend(validate_schema(el, items, f"{path}[{i}]"))

    if isinstance(obj, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in obj:
                errors.append(f"{path}: missing required field '{key}'")
        for key, sub in props.items():
            if key in obj:
                errors.extend(validate_schema(obj[key], sub, f"{path}.{key}"))
        if schema.get("additionalProperties") is False:
            for key in obj:
                if key not in props:
                    errors.append(f"{path}: unexpected field '{key}'")
    return errors


def infer_schema_from_example(example: dict, required_keys: list[str] | None = None) -> dict:
    """Conservative schema from one example object.

    Only the caller-specified `required_keys` become required (None means all
    top-level keys — the UI passes the user's choice). Types come from the
    example; array item shape comes from the first element; nothing gains
    length/enum constraints.
    """

    def infer(value):
        if isinstance(value, dict):
            return {
                "type": "object",
                "properties": {k: infer(v) for k, v in value.items()},
            }
        if isinstance(value, list):
            schema: dict = {"type": "array"}
            if value:
                schema["items"] = infer(value[0])
            return schema
        if isinstance(value, bool):
            return {"type": "boolean"}
        if isinstance(value, int):
            return {"type": "integer"}
        if isinstance(value, float):
            return {"type": "number"}
        if value is None:
            return {}
        return {"type": "string"}

    schema = infer(example)
    keys = list(example.keys()) if required_keys is None else [
        k for k in required_keys if k in example
    ]
    schema["required"] = keys
    return schema


# --------------------------------------------------------------------------- #
# Safe filenames for per-word object files
# --------------------------------------------------------------------------- #
# \w in Python 3 already covers CJK ideographs, kana and hangul.
_SAFE_CHARS = re.compile(r"[^\w.\-]+")


def sanitize_filename(word: str, max_len: int = 60) -> str:
    """Filesystem-safe name for objects/<word>.json; CJK is kept as-is."""
    word = (word or "").strip()
    safe = _SAFE_CHARS.sub("_", word).strip("._")
    clipped = safe[:max_len]
    if not clipped or clipped != word or len(safe) > max_len:
        digest = hashlib.sha1(word.encode("utf-8")).hexdigest()[:8]
        clipped = f"{clipped or 'item'}_{digest}"
    return clipped
