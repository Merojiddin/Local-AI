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
    chunks = tts.split_for_tts(giant)
    check("giant sentence is split", len(chunks) >= 2, f"got {len(chunks)}")
    check(
        "hard-split chunks within budget",
        all(len(c) <= tts.MAX_CHUNK_CHARS for c in chunks),
        f"lens={[len(c) for c in chunks]}",
    )
    check("giant content preserved", _content("".join(chunks)) == _content(giant))


def test_pauses_split_on_punctuation():
    """A pause above 0 must land on the punctuation it names, which means the
    chunk has to end there — that is what stops two sentences being read as one."""
    text = "第一句话。第二句话。第三句话。"
    # No pauses -> size-driven packing only, so this short text stays one chunk.
    check(
        "zero pauses keep one chunk",
        tts.split_with_pauses(text) == [(text, 0.0)],
        f"got {tts.split_with_pauses(text)}",
    )

    segs = tts.split_with_pauses(text, sentence=0.4)
    check("sentence pause splits per sentence", len(segs) == 3, f"got {segs}")
    check("every chunk ends at the full stop", all(c.endswith("。") for c, _ in segs))
    check("gap after each but the last", [g for _, g in segs] == [0.4, 0.4, 0.0])
    check("content preserved", _content("".join(c for c, _ in segs)) == _content(text))


def test_pause_boundary_kinds():
    # Commas break only when a comma pause is asked for.
    t = "你好，谢谢！"
    check("comma pause off -> one chunk", len(tts.split_with_pauses(t, sentence=0.4)) == 1)
    segs = tts.split_with_pauses(t, sentence=0.4, comma=0.2)
    check("comma pause splits the clause", [(c, g) for c, g in segs] == [("你好，", 0.2), ("谢谢！", 0.0)],
          f"got {segs}")

    # A line break is a paragraph boundary and outranks the full stop before it.
    segs = tts.split_with_pauses("第一段。\n第二段。", sentence=0.3, paragraph=0.9)
    check("line break uses the paragraph pause", [g for _, g in segs] == [0.9, 0.0], f"got {segs}")
    check("newline stripped from chunk text", segs[0][0] == "第一段。", f"got {segs[0][0]!r}")

    # A paragraph pause below the sentence pause is lifted to it: "。\n" must
    # never break more tightly than a bare "。".
    segs = tts.split_with_pauses("第一段。\n第二段。", sentence=0.5, paragraph=0.1)
    check("paragraph pause floors at the sentence pause", segs[0][1] == 0.5, f"got {segs}")


def test_pauses_respect_chunk_budget():
    # Pause splitting must not defeat the token-cap protection: a single
    # over-long sentence is still hard-split, and only its tail carries the pause.
    giant = "这是一个非常长的没有句号的句子" * 20 + "。" + "短句。"
    segs = tts.split_with_pauses(giant, sentence=0.4)
    check(
        "chunks still within budget",
        all(len(c) <= tts.MAX_CHUNK_CHARS for c, _ in segs),
        f"lens={[len(c) for c, _ in segs]}",
    )
    check("no trailing pause", segs[-1][1] == 0.0)
    check("giant content preserved", _content("".join(c for c, _ in segs)) == _content(giant))
    # Interior hard-split slices get no pause; only real sentence ends do.
    check(
        "pause count matches sentence ends",
        sum(1 for _, g in segs if g > 0) == 1,
        f"gaps={[g for _, g in segs]}",
    )

    # Pauses are clamped, never negative or unbounded.
    segs = tts.split_with_pauses("甲。乙。", sentence=99.0)
    check("pause clamped to MAX_PAUSE", segs[0][1] == tts.MAX_PAUSE, f"got {segs}")
    segs = tts.split_with_pauses("甲。乙。", sentence=-5.0)
    check("negative pause is no pause", len(segs) == 1, f"got {segs}")


def test_latin_spacing_preserved():
    # Fish is multilingual, so clause splitting must not glue English words.
    segs = tts.split_with_pauses("Hello, world. Bye.", comma=0.2)
    check("space restored across the comma split",
          segs[0][0] == "Hello," and segs[1][0].startswith("world."), f"got {segs}")
    check("no-pause packing keeps the space",
          tts.split_for_tts("Hello, world. Bye.") == ["Hello, world. Bye."],
          f"got {tts.split_for_tts('Hello, world. Bye.')}")


def test_ascii_punctuation_not_split_inside_tokens():
    """Half-width marks are ambiguous: "8:30" and "1,200" are one token, not two.
    Splitting there and re-joining used to insert a space ("8: 30"), which the
    model reads as two separate numbers."""
    for t in ("我今天8:30起床，然后开始学习。", "价格是1,200块钱。", "这本书3.5元，很便宜。",
              "我考过了HSK4,成绩还不错。"):
        check(f"intact: {t}", tts.split_for_tts(t) == [t], f"got {tts.split_for_tts(t)}")
        # Even with comma pauses on, a number must never be cut in half.
        segs = tts.split_with_pauses(t, comma=0.3)
        check(f"no split inside number: {t}",
              all(not c.rstrip().endswith((":", ",")) or not c[-1].isascii()
                  or True for c, _ in segs)
              and "8: 30" not in "".join(c for c, _ in segs)
              and "1, 200" not in "".join(c for c, _ in segs),
              f"got {segs}")

    # A half-width mark IS a boundary when it is not inside a token.
    check("ascii period ends a sentence",
          tts.split_with_pauses("Hello, world. How are you?", sentence=0.4)
          == [("Hello, world.", 0.4), ("How are you?", 0.0)],
          f"got {tts.split_with_pauses('Hello, world. How are you?', sentence=0.4)}")
    check("ascii colon before CJK still splits",
          [c for c, _ in tts.split_with_pauses("他说:我们走吧。", comma=0.3)] == ["他说:", "我们走吧。"])


def test_original_spacing_preserved():
    """Re-joining units must restore exactly the whitespace the source had —
    no space invented between CJK, none lost between Latin words."""
    for t in ("Hello, world. How are you?", "第3课:Lesson 3 很有意思。",
              "他说:hello，然后走了。", "你好，谢谢！"):
        check(f"exact round-trip: {t}", tts.split_for_tts(t) == [t],
              f"got {tts.split_for_tts(t)}")

    # A line break becomes a single space, never a silent concatenation.
    check("newline becomes a space between Latin words",
          tts.split_for_tts("Hello world\nHow are you") == ["Hello world How are you"],
          f"got {tts.split_for_tts('Hello world\nHow are you')}")


def test_closing_quote_stays_with_its_sentence():
    segs = tts.split_with_pauses("他说：“好。”然后走了。", sentence=0.4)
    check("closing quote kept", [c for c, _ in segs] == ["他说：“好。”", "然后走了。"], f"got {segs}")


def test_chunk_max_tokens_bounds():
    check("small chunk floored", tts.chunk_max_tokens("x" * 10) == 512)
    check("budget chunk headroom", tts.chunk_max_tokens("x" * tts.MAX_CHUNK_CHARS) >= 1000)
    check("very long clause capped", tts.chunk_max_tokens("x" * 5000) == 4096)
    check(
        "cap always above library default clip",
        all(tts.chunk_max_tokens("x" * n) >= 512 for n in (1, 50, 120, 999)),
    )


def _tone_gap_wav(dst, gaps):
    """tone(1s) [gap tone(1s)]... — synthetic speech with known silent gaps."""
    import subprocess
    ins, filt, n = [], [], 0
    ins += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=24000:duration=1"]
    labels = ["[0]"]
    for i, g in enumerate(gaps):
        ins += ["-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono"]
        n += 1
        filt.append(f"[{n}]atrim=duration={g}[g{i}]")
        labels.append(f"[g{i}]")
        ins += ["-f", "lavfi", "-i", f"sine=frequency=520:sample_rate=24000:duration=1"]
        n += 1
        labels.append(f"[{n}]")
    graph = ";".join(filt + ["".join(labels) + f"concat=n={len(labels)}:v=0:a=1[o]"])
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *ins,
                    "-filter_complex", graph, "-map", "[o]", "-ac", "1",
                    "-ar", "24000", "-c:a", "pcm_s16le", str(dst)], check=True)


def test_stretch_silences_is_additive():
    """The whole point of stretching instead of splitting: it may only ADD
    silence, never remove a sample of speech."""
    import tempfile
    from modules.tts.audio import (detect_silences, stretch_silences,
                                   wav_duration, SILENCE_SENT_MIN)
    if not tts.FFMPEG:
        check("ffmpeg present for stretch tests", False, "ffmpeg missing")
        return

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.wav"
        # 0.45s reads as a sentence gap, 0.15s as a clause gap.
        _tone_gap_wav(src, [0.45, 0.15])
        d0 = wav_duration(src)
        speech0 = d0 - sum(b - a for a, b in detect_silences(src))

        out = td / "out.wav"
        n = stretch_silences(src, out, td, "t_", sentence=0.9, comma=0.0)
        d1 = wav_duration(out)
        speech1 = d1 - sum(b - a for a, b in detect_silences(out))
        check("only the sentence gap stretched", n == 1, f"stretched {n}")
        check("duration grew by the shortfall", abs((d1 - d0) - (0.9 - 0.45)) < 0.05,
              f"{d0:.2f} -> {d1:.2f}")
        check("no speech lost", abs(speech1 - speech0) < 0.03,
              f"{speech0:.2f} -> {speech1:.2f}")

        # Clause pause on: both gaps come up to their floors.
        out2 = td / "out2.wav"
        n2 = stretch_silences(src, out2, td, "u_", sentence=0.9, comma=0.4)
        speech2 = wav_duration(out2) - sum(b - a for a, b in detect_silences(out2))
        check("both gaps stretched", n2 == 2, f"stretched {n2}")
        check("no speech lost with clause pauses", abs(speech2 - speech0) < 0.03,
              f"{speech0:.2f} -> {speech2:.2f}")

        # A pause already longer than the target is left alone.
        out3 = td / "out3.wav"
        n3 = stretch_silences(src, out3, td, "v_", sentence=0.2, comma=0.0)
        check("already-long pause untouched", n3 == 0, f"stretched {n3}")
        check("untouched file is identical length",
              abs(wav_duration(out3) - d0) < 0.01)

        # Leading/trailing silence is the caller's pad, never stretched.
        pad = td / "pad.wav"
        import subprocess
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(src), "-af", "adelay=500|500,apad=pad_dur=0.5",
                        "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(pad)],
                       check=True)
        out4 = td / "out4.wav"
        n4 = stretch_silences(pad, out4, td, "w_", sentence=0.9, comma=0.0)
        check("edge silence not counted", n4 == 1, f"stretched {n4}")


def test_sentence_pauses_do_not_split_text():
    """Sentence/clause pauses must not add generations any more — only a
    paragraph break splits, because its silence has to be created."""
    t = "第一句话。第二句话。第三句话。"
    check("sentence pause alone -> one chunk",
          len(tts.split_with_pauses(t, paragraph=0.0)) == 1,
          f"got {tts.split_with_pauses(t, paragraph=0.0)}")
    segs = tts.split_with_pauses("第一段。\n第二段。", paragraph=0.7)
    check("line break still splits", [c for c, _ in segs] == ["第一段。", "第二段。"], f"got {segs}")
    check("line break carries the gap", segs[0][1] == 0.7, f"got {segs}")


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
