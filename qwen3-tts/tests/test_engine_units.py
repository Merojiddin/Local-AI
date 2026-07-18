"""Unit tests for modules/generator_engine.py (no chat-model weights loaded).

Run with: .venv/bin/python tests/test_engine_units.py
Loads only the tokenizer (a few MB) for token counting.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import generator_engine as ge  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        print(f"FAIL  {name}  {detail}")
        FAILURES.append(name)


def test_templates():
    out = ge.render_template("Word {{word}} ({{word_index}}/{{total_words}}) {{missing}}",
                             {"word": "爱", "word_index": 1, "total_words": 5})
    check("variables substituted", out == "Word 爱 (1/5) ")
    check("input line rendered", ge._input_line("Pinyin", "ài") == "Pinyin: ài\n")
    check("input line empty", ge._input_line("Pinyin", "") == "")


def test_pinyin():
    check("tone marks stripped", ge.pinyin_without_tones("xǐhuan") == "xihuan")
    check("tone digits stripped", ge.pinyin_without_tones("ai4") == "ai")
    check("ü kept without tone", ge.pinyin_without_tones("nǚ") == "nu"
          or ge.pinyin_without_tones("nǚ") == "nü")


def test_queries():
    q = ge.build_queries({"hanzi": "爱", "pinyin": "ài"}, expansion=True)
    check("hanzi first", q[0] == "爱")
    check("pinyin variants included", any("ài" in x for x in q)
          and any("ai" in x and "ài" not in x for x in q), q)
    check("no traditional invented", not any("愛" in x for x in q))
    q2 = ge.build_queries({"hanzi": "爱"}, expansion=False)
    check("expansion off = word only", q2 == ["爱"])


def test_tokens_and_budget():
    # The generator follows the selected chat model; when its weights are
    # installed the context comes from its config.json, otherwise the
    # conservative 32768 fallback is used.
    from modules import model_manager as mgr

    key = ge.chat_key()
    if mgr.is_installed(key):
        import json as _json
        from pathlib import Path as _Path
        cfg = _json.loads((_Path(mgr.model_path_or_error(key)) / "config.json")
                          .read_text(encoding="utf-8"))
        expected = int(cfg.get("max_position_embeddings", 32768))
    else:
        expected = 32768
    check("model max context read", ge.model_max_context() == expected,
          ge.model_max_context())
    n = ge.count_tokens("爱 means to love — yêu")
    check("token counting works", 3 <= n <= 30, n)

    rep = ge.context_report("sys", "evidence", "prompt", 8192, 32768)
    check("report fits small", rep["fits"] and rep["budget"] == 32768)
    rep = ge.context_report("sys", "x " * 40000, "prompt", 8192, 16384)
    check("overflow detected", not rep["fits"] and rep["overflow"] > 0)

    chunks = [{"text": "词 " * 800, "source": f"f{i}.txt", "score": 1.0 - i * 0.1}
              for i in range(8)]
    kept, eff_max, notes = ge.fit_to_budget(
        chunks, "sys", "prompt", 2048, 4096, lambda cc: "\n".join(c["text"] for c in cc))
    check("evidence trimmed before output", len(kept) < 8 and notes, len(kept))
    check("worst chunks dropped first", all(c["score"] >= kept[-1]["score"] for c in kept))
    check("output allowance kept when possible", eff_max == 2048
          or (eff_max >= ge.MIN_OUTPUT_TOKENS and any("reduced output" in n for n in notes)))

    kept2, eff2, notes2 = ge.fit_to_budget(
        [], "sys", "p " * 8000, 8192, 4096, lambda cc: "")
    check("output reduced when prompt alone too big",
          eff2 < 8192 and eff2 >= ge.MIN_OUTPUT_TOKENS, (eff2, notes2))


def test_evidence_rendering():
    chunks = [
        {"text": "爱 ài love", "source": "hsk.txt", "collection": "HSK", "page": 2,
         "fields": [], "score": 0.9, "exact": True, "chunk_id": "a#0"},
        {"text": "我爱你。", "source": "ex.docx", "collection": "Examples",
         "fields": ["examples"], "score": 0.8, "exact": False, "chunk_id": "b#1"},
    ]
    text = ge.render_evidence_text(chunks)
    check("numbered labels", "[S1]" in text and "[S2]" in text)
    check("grouped by field", "General evidence" in text and "field 'examples'" in text)
    check("provenance shown", "page 2" in text)
    prov = ge.evidence_provenance(chunks)
    check("provenance records", prov[0]["label"] == "S1"
          and prov[0]["exact_match"] and prov[1]["fields"] == ["examples"])
    check("empty evidence text", "no evidence" in ge.render_evidence_text([]))


def test_conflict_check():
    chunks = [
        {"text": "爱 HSK 1", "collection": "List A", "source": "a.txt",
         "exact": True, "fields": ["hsk_level"]},
        {"text": "爱 HSK 2", "collection": "List B", "source": "b.txt",
         "exact": True, "fields": ["hsk_level"]},
    ]
    notes = ge._conflict_check(chunks, {"List A", "List B"}, "爱")
    check("conflict detected", notes["conflicts"]
          and notes["conflicts"][0]["field"] == "hsk_level"
          and len(notes["conflicts"][0]["excerpts"]) == 2)
    notes2 = ge._conflict_check(chunks, {"List A"}, "爱")
    check("no conflict if one authoritative", notes2["conflicts"] == [])


def test_validation():
    cfg = {
        "validation_level": "strict", "key_field": "hanzi",
        "schema_mode": "example",
        "example_object": {"hanzi": "词", "pinyin": "cí", "senses": [{"en": "word"}]},
        "required_keys": ["hanzi", "pinyin", "senses"],
        "custom_checks": {"pos_allowlist": ["n", "v", "adj"]},
    }
    good = {"hanzi": "爱", "pinyin": "ài", "senses": [{"en": "to love"}]}
    errs, warns = ge.validate_result(good, cfg, "爱", n_sources=3)
    check("valid passes", errs == [], errs)

    errs, _ = ge.validate_result({"hanzi": "恨", "pinyin": "ài",
                                  "senses": [{"en": "x"}]}, cfg, "爱", 3)
    check("wrong hanzi caught", any("queued word" in e for e in errs))

    errs, _ = ge.validate_result({"hanzi": "爱", "pinyin": "ài",
                                  "senses": [{"en": "see ```code```"}]}, cfg, "爱", 3)
    check("markdown fence caught", any("fence" in e for e in errs))

    errs, _ = ge.validate_result({"hanzi": "爱", "pinyin": "ài",
                                  "senses": [{"en": "love [S9]"}]}, cfg, "爱", 3)
    check("fabricated citation caught", any("[S9]" in e for e in errs))

    errs, _ = ge.validate_result({"hanzi": "爱", "pinyin": "",
                                  "senses": [{"en": "x"}]}, cfg, "爱", 3)
    check("empty pinyin is error", any("pinyin" in e for e in errs))

    _, warns = ge.validate_result({"hanzi": "爱", "pinyin": "ai",
                                   "senses": [{"en": "x"}]}, cfg, "爱", 3)
    check("toneless pinyin is warning", any("tone" in w for w in warns))

    errs, _ = ge.validate_result(
        {"hanzi": "爱", "pinyin": "ài", "senses": [{"en": "x"}],
         "examples": [{"zh": "我爱你。", "sense_ref": 5}]}, cfg, "爱", 3)
    check("sense ref out of range", any("sense_ref" in e for e in errs))

    _, warns = ge.validate_result(
        {"hanzi": "爱", "pinyin": "ài", "senses": [{"en": "x", "pos": "verbo"}]},
        cfg, "爱", 3)
    check("pos allowlist warning", any("allowlist" in w for w in warns))

    errs, _ = ge.validate_result({"hanzi": "爱"}, cfg, "爱", 3)
    check("schema missing fields", any("pinyin" in e for e in errs))


def test_prompt_assembly():
    cfg = {
        "schema_mode": "example",
        "example_object": {"hanzi": "词", "pinyin": "cí"},
        "required_keys": ["hanzi", "pinyin"],
        "prompt_template": ("W={{word}} P={{input_pinyin}}S={{json_schema}} "
                            "E={{retrieved_sources}} A={{authoritative_sources}}"),
        "authoritative": ["HSK"],
        "only_sources": True, "omit_unsupported": True, "never_invent_citations": True,
        "key_field": "hanzi",
    }
    item = {"hanzi": "爱", "pinyin": "ài", "custom_instruction": "focus on family usage"}
    system, user = ge.build_word_prompt(cfg, item, [], 1, 3)
    check("system has accuracy rules", "ONLY the supplied source" in system
          and "Never invent citations" in system)
    check("word in prompt", "W=爱" in user)
    check("input pinyin in prompt", "Pinyin (from input): ài" in user)
    check("schema + example embedded", '"required"' in user and "Example object" in user)
    check("authoritative listed", "A=HSK" in user)
    check("custom instruction appended", "focus on family usage" in user)

    schema = ge.effective_schema(cfg)
    check("effective schema from example", schema["required"] == ["hanzi", "pinyin"])


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        print(f"[{fn}]")
        globals()[fn]()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        sys.exit(1)
    print("All engine unit tests passed.")
