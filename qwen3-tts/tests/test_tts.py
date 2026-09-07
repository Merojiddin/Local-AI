"""Tests for modules/tts.py long-text chunking — run with:
    .venv/bin/python tests/test_tts.py

Only the pure text-splitting helpers are exercised (no model / ffmpeg needed).
These guard the fix for long audio being clipped at the model's per-call token
cap and trailing off into silence: text is now split on sentence boundaries and
synthesized chunk-by-chunk.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import tts  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}")
    else:
        print(f"FAIL  {name}  {detail}")
        FAILURES.append(name)


ESSAY = (
    "你选择工资高但是要加班的工作还是工资少但压力少的工作？"
    "面对这两个工作机会，如果让我做出选择，就我个人而言，我更倾向于选择第二份工作，"
    "也就是工资比较少、但是压力也比较少的工作。做出这个决定，主要有以下两个极其重要的原因："
    "首先，身体健康是一切事情的基础，比赚很多钱更重要。现在的现代生活速度特别快，"
    "如果选择一份需要经常加班、压力很大的工作，身体和精神都会感觉特别疲劳。"
    "总之，轻松、稳定的工作最符合我对幸福生活的想法。"
)


def _content(s: str) -> str:
    """Comparable payload: drop whitespace the splitter strips at boundaries."""
    return "".join(s.split())


def test_short_text_single_chunk():
    check("empty -> no chunks", tts.split_for_tts("") == [])
    check("whitespace -> no chunks", tts.split_for_tts("   \n  ") == [])
    check("short -> one chunk", tts.split_for_tts("你好，谢谢！") == ["你好，谢谢！"])
    # A chunk exactly at the budget is not split.
    at_budget = "字" * tts.MAX_CHUNK_CHARS
    check("at-budget -> one chunk", tts.split_for_tts(at_budget) == [at_budget])


def test_long_text_chunked_within_budget():
    # Explicit budget so this exercises the splitter itself rather than the
    # module default, which is now large enough to hold the essay in one take.
    budget = 200
    chunks = tts.split_for_tts(ESSAY, max_chars=budget)
    check("essay splits into several chunks", len(chunks) >= 2, f"got {len(chunks)}")
    check(
        "every chunk within budget",
        all(len(c) <= budget for c in chunks),
        f"lens={[len(c) for c in chunks]}",
    )
    check("no empty chunks", all(c.strip() for c in chunks))
    check("content preserved", _content("".join(chunks)) == _content(ESSAY))


def test_sentence_boundaries_preferred():
    # Two sentences that together exceed the (explicit) budget must split
    # between them, not mid-sentence. Explicit max_chars keeps this independent
    # of the module default.
    a = "第一句话内容" * 12  # ~72 chars
    b = "第二句话内容" * 12
    chunks = tts.split_for_tts(a + "。" + b + "。", max_chars=100)
    check("splits between sentences", len(chunks) == 2, f"got {len(chunks)}")
    check("first chunk ends the first sentence", chunks[0].endswith("。"))


def test_giant_sentence_hard_split():
    # A single clause with no sentence-ending punctuation still gets broken up
    # so no chunk can blow the token budget.
    giant = "这是一个非常长的没有句号的句子" * 20
    budget = 200
    chunks = tts.split_for_tts(giant, max_chars=budget)
    check("giant sentence is split", len(chunks) >= 2, f"got {len(chunks)}")
    check(
        "hard-split chunks within budget",
        all(len(c) <= budget for c in chunks),
        f"lens={[len(c) for c in chunks]}",
    )
    check("giant content preserved", _content("".join(chunks)) == _content(giant))


def test_default_budget_holds_a_two_minute_read():
    # Every extra chunk is an independent generation and so an audible prosody
    # restart. The default budget must therefore keep an ordinary long read —
    # a full two-minute one included — as a SINGLE generation. Measured on the
    # 1.7B model: 625 chars of natural prose -> 121.5s of audio, i.e. 5.14
    # chars/sec, so two minutes is ~620 chars.
    two_minutes = "字" * 620
    chunks = tts.split_for_tts(two_minutes)
    check(
        "two-minute read stays one chunk",
        len(chunks) == 1,
        f"budget={tts.MAX_CHUNK_CHARS}, got {len(chunks)} chunks",
    )


def test_chunk_max_tokens_bounds():
    check("small chunk floored", tts.chunk_max_tokens("x" * 10) == 512)
    check("budget chunk headroom", tts.chunk_max_tokens("x" * tts.MAX_CHUNK_CHARS) >= 1000)
    check("very long clause capped", tts.chunk_max_tokens("x" * 5000) == 4096)
    check(
        "cap always above library default clip",
        all(tts.chunk_max_tokens("x" * n) >= 512 for n in (1, 50, 120, 999)),
    )


def test_fish_routing():
    # Fish is recognised and routes to its single repo regardless of voice mode.
    check("is_fish true for Fish label", tts.is_fish(tts.FISH_LABEL))
    check("is_fish false for Qwen", not tts.is_fish("Higher Quality — Qwen3-TTS 1.7B"))
    check("active_repo -> Fish repo", tts.active_repo(tts.FISH_LABEL) == tts.FISH_REPO)
    check("Fish in registry key map", tts.FISH_REPO in tts.REPO_TO_KEY)
    check("Fish has a short name", bool(tts.MODEL_SHORT.get(tts.FISH_REPO)))
    check(
        "Fish is a third model choice",
        tts.FISH_LABEL in [v for _, v in tts.model_choices("en")],
    )
    # Batch CSV model column accepts Fish aliases.
    for alias in ("fish", "clone", "Voice Clone — Fish S2 Pro 4B"):
        check(f"_resolve_model({alias!r})", tts._resolve_model(alias, "x", 1) == tts.FISH_LABEL)


def _write_wav(path: Path) -> None:
    import struct
    import wave

    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(struct.pack("<" + "h" * 2400, *([0] * 2400)))


def test_saved_voices():
    # Persist voice-clone profiles to an isolated temp dir so real user data is
    # never touched. Exercises save, list, get, overwrite and delete.
    import tempfile

    from modules import storage
    from modules.tts import voices as vx

    root = Path(tempfile.mkdtemp())
    vdir = root / "voices"
    orig = storage.load_settings
    storage.load_settings = lambda: {**orig(), "voices_dir": str(vdir)}
    try:
        clip = root / "ref.wav"
        _write_wav(clip)

        check("starts empty", vx.list_voices() == [])
        v = vx.save_voice("李老师 Li", str(clip), "你好")
        check("clip copied in", Path(v["path"]).is_file())
        check("clip lives under voices dir", Path(v["path"]).parent == vdir)
        check("name listed", vx.voice_names() == ["李老师 Li"])
        check("transcript round-trips", vx.get_voice("李老师 Li")["ref_text"] == "你好")

        # Re-saving the same name overwrites in place (no duplicate profile).
        vx.save_voice("李老师 Li", str(clip), "更新")
        check("overwrite keeps one entry", len(vx.list_voices()) == 1)
        check("overwrite updates text", vx.get_voice("李老师 Li")["ref_text"] == "更新")

        # A different name that slugs the same still gets its own files.
        vx.save_voice("!!!", str(clip), "a")
        vx.save_voice("???", str(clip), "b")
        check("slug collision keeps both", len(vx.list_voices()) == 3)

        # Empty name / missing clip are rejected.
        for bad in (lambda: vx.save_voice("", str(clip)),
                    lambda: vx.save_voice("x", None)):
            try:
                bad()
                check("invalid save rejected", False, "no error raised")
            except ValueError:
                check("invalid save rejected", True)

        check("delete removes it", vx.delete_voice("李老师 Li"))
        check("delete is idempotent", not vx.delete_voice("李老师 Li"))
        check("two profiles remain", len(vx.list_voices()) == 2)
    finally:
        storage.load_settings = orig


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        print(f"[{fn}]")
        globals()[fn]()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        sys.exit(1)
    print("All tts chunking tests passed.")
