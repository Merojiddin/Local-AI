"""Tests for modules/safe_json.py — run with:  .venv/bin/python tests/test_safe_json.py"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import safe_json as sj  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        print(f"FAIL  {name}  {detail}")
        FAILURES.append(name)


def test_atomic_write_and_load():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sub with space" / "results.json"
        data = [{"hanzi": "爱", "pinyin": "ài"}]
        sj.atomic_write_json(p, data)
        check("atomic write round-trip", sj.load_json(p) == data)
        check("no tmp leftovers", not list(p.parent.glob("*.tmp")))

        b = sj.timestamped_backup(p, Path(td) / "backups")
        check("backup created", b is not None and b.exists())
        p.write_text("{corrupt", encoding="utf-8")
        check("load_json default on corrupt", sj.load_json(p, default="x") == "x")
        found, restored = sj.latest_valid_backup(p, Path(td) / "backups")
        check("backup restore finds data", restored == data)


def test_jsonl():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "log.jsonl"
        sj.append_jsonl(p, {"a": 1})
        sj.append_jsonl(p, {"b": "喜欢"})
        rows = sj.read_jsonl(p)
        check("jsonl round-trip", rows == [{"a": 1}, {"b": "喜欢"}])
        check("jsonl keeps CJK unescaped", "喜欢" in p.read_text(encoding="utf-8"))


def test_extract():
    obj = {"hanzi": "爱", "senses": [{"def": 'He said "hi" {ok}'}]}
    raw = "Sure! Here is the entry:\n```json\n" + json.dumps(obj, ensure_ascii=False) + "\n```\nHope this helps!"
    s, err = sj.extract_json_object(raw)
    check("extract from fenced chatter", s is not None and json.loads(s) == obj)

    s, err = sj.extract_json_object('<think>{"x": 1}</think>{"y": 2}')
    check("think block skipped", s is not None and json.loads(s) == {"y": 2})

    s, err = sj.extract_json_object('{"a": "unterminated…')
    check("truncated detected", s is None and err == "truncated")

    s, err = sj.extract_json_object("no json here")
    check("not found detected", s is None and err == "not_found")

    s, err = sj.extract_json_object('{"a": "brace } in string", "b": 1} trailing')
    check("brace inside string ignored", s is not None and json.loads(s)["b"] == 1)


def test_repair():
    bad = '{"a": [1, 2,], "b": True, "c": None, "note": "保留 True 和 “中文引号”",}'
    fixed = sj.repair_json_syntax(bad)
    obj = json.loads(fixed)
    check("trailing commas fixed", obj["a"] == [1, 2])
    check("python literals fixed", obj["b"] is True and obj["c"] is None)
    check("string content untouched", obj["note"] == "保留 True 和 “中文引号”")


def test_parse_strict():
    obj, errs = sj.parse_object_strict('{"a": 1, "a": 2}')
    check("duplicate key reported", any("duplicate key: a" in e for e in errs))
    obj, errs = sj.parse_object_strict("[1, 2]")
    check("non-object rejected", obj is None and any("expected one JSON object" in e for e in errs))


def test_extract_and_parse():
    obj, errs, trunc = sj.extract_and_parse('Result: {"hanzi": "爱", "n": [1,2,],}')
    check("pipeline repairs and parses", obj == {"hanzi": "爱", "n": [1, 2]})
    obj, errs, trunc = sj.extract_and_parse('{"hanzi": "爱", "pinyin": "à')
    check("pipeline flags truncation", obj is None and trunc)


def test_schema():
    schema = {
        "type": "object",
        "required": ["hanzi", "pinyin", "senses"],
        "properties": {
            "hanzi": {"type": "string", "minLength": 1},
            "pinyin": {"type": "string", "minLength": 1},
            "hsk": {"type": "integer", "minimum": 1, "maximum": 9},
            "senses": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["meaning_en"],
                    "properties": {"meaning_en": {"type": "string"}},
                },
            },
        },
    }
    good = {"hanzi": "爱", "pinyin": "ài", "hsk": 1, "senses": [{"meaning_en": "to love"}]}
    check("valid object passes", sj.validate_schema(good, schema) == [])

    bad = {"hanzi": "", "hsk": 12, "senses": []}
    errs = sj.validate_schema(bad, schema)
    check("missing required", any("missing required field 'pinyin'" in e for e in errs))
    check("minLength", any("minLength" in e for e in errs))
    check("maximum", any("maximum" in e for e in errs))
    check("minItems", any("minItems" in e for e in errs))

    errs = sj.validate_schema({"hanzi": 5, "pinyin": "ài", "senses": [{"meaning_en": 1}]}, schema)
    check("type errors with paths", any(e.startswith("$.hanzi") for e in errs)
          and any("$.senses[0].meaning_en" in e for e in errs))


def test_infer():
    example = {"hanzi": "爱", "hsk": 1, "senses": [{"meaning_en": "love", "vi": "yêu"}], "note": None}
    schema = sj.infer_schema_from_example(example, required_keys=["hanzi", "senses"])
    check("inferred required respects choice", schema["required"] == ["hanzi", "senses"])
    check("inferred nested types",
          schema["properties"]["senses"]["items"]["properties"]["meaning_en"]["type"] == "string")
    check("optional fields not required", "note" not in schema["required"])
    check("example object validates", sj.validate_schema(example, schema) == [])


def test_sanitize():
    check("cjk preserved", sj.sanitize_filename("爱") == "爱")
    check("word kept verbatim", sj.sanitize_filename("喜欢") == "喜欢")
    a = sj.sanitize_filename("a/b:c")
    b = sj.sanitize_filename("a/b?c")
    check("slashes removed", "/" not in a and ":" not in a)
    check("different words differ", a != b, f"{a} vs {b}")
    check("empty handled", sj.sanitize_filename("") != "")
    long = sj.sanitize_filename("x" * 500)
    check("long clipped", len(long) < 80)


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        print(f"[{fn}]")
        globals()[fn]()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        sys.exit(1)
    print("All safe_json tests passed.")
