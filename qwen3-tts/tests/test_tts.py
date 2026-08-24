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
    chunks = tts.split_for_tts(ESSAY)
    check("essay splits into several chunks", len(chunks) >= 2, f"got {len(chunks)}")
    check(
        "every chunk within budget",
        all(len(c) <= tts.MAX_CHUNK_CHARS for c in chunks),
        f"lens={[len(c) for c in chunks]}",
    )
    check("no empty chunks", all(c.strip() for c in chunks))
    check("content preserved", _content("".join(chunks)) == _content(ESSAY))


def test_sentence_boundaries_preferred():
    # Two sentences that together exceed the budget must split between them,
    # not mid-sentence.
    a = "第一句话内容" * 12  # ~72 chars
    b = "第二句话内容" * 12
    chunks = tts.split_for_tts(a + "。" + b + "。")
    check("splits between sentences", len(chunks) == 2, f"got {len(chunks)}")
    check("first chunk ends the first sentence", chunks[0].endswith("。"))


def test_giant_sentence_hard_split():
    # A single clause with no sentence-ending punctuation still gets broken up
    # so no chunk can blow the token budget.
    giant = "这是一个非常长的没有句号的句子" * 20
    chunks = tts.split_for_tts(giant)
    check("giant sentence is split", len(chunks) >= 2, f"got {len(chunks)}")
    check(
        "hard-split chunks within budget",
        all(len(c) <= tts.MAX_CHUNK_CHARS for c in chunks),
        f"lens={[len(c) for c in chunks]}",
    )
    check("giant content preserved", _content("".join(chunks)) == _content(giant))


def test_chunk_max_tokens_bounds():
    check("small chunk floored", tts.chunk_max_tokens("x" * 10) == 512)
    check("budget chunk headroom", tts.chunk_max_tokens("x" * tts.MAX_CHUNK_CHARS) >= 1000)
    check("very long clause capped", tts.chunk_max_tokens("x" * 5000) == 4096)
    check(
        "cap always above library default clip",
        all(tts.chunk_max_tokens("x" * n) >= 512 for n in (1, 50, 120, 999)),
    )


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        print(f"[{fn}]")
        globals()[fn]()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        sys.exit(1)
    print("All tts chunking tests passed.")
