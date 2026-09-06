"""Long-text chunking: split text into synthesis-sized pieces on sentence
boundaries so a single over-long generation can't hit the token cap, and tag
each piece with the silence to splice in after it."""

from __future__ import annotations

import re


# --------------------------------------------------------------------------- #
# Long-text chunking
# --------------------------------------------------------------------------- #
# The model generates a bounded number of tokens per call (~12.5 tokens/sec of
# audio). A long paragraph fed in one shot can exceed that budget and trail off
# into silence when the cap is hit mid-utterance, so long text is split on
# sentence boundaries and each piece is synthesized separately, then joined.
#
# Each chunk is an independent generation, so its pace/intonation/loudness reset
# at the boundary — audible as the reading "restarting". Two things keep that in
# check: (1) the budget is large enough that a normal paragraph stays a SINGLE
# coherent generation, and chunks are balanced to roughly equal size (no tiny
# tail pieces); (2) chunks are loudness-matched before joining (normalize_wav_loud).
MAX_CHUNK_CHARS = 200

# Fish S2 Pro is a different beast from the light 24 kHz Qwen models: it is a 4B
# model at 44.1 kHz whose vocoder (codec.decode) materialises a whole chunk's
# waveform in a SINGLE Metal buffer, growing linearly with the tokens generated.
# On a 16 GB Mac the largest single GPU buffer is ~8.9 GiB, and a normal-length
# chunk overflows it — the model dies with "[metal::malloc] Attempting to
# allocate … greater than the maximum allowed buffer size". That ceiling is a
# fixed device property: it cannot be raised with set_memory_limit /
# set_wired_limit (those bound the *total* working set, not one allocation), so
# the only cure is to decode less audio per call. Fish therefore uses a much
# smaller chunk and a hard token cap that keeps every decode well under the
# buffer limit while coexisting with the ~7.5 GB resident model.
FISH_MAX_CHUNK_CHARS = 60
FISH_MAX_TOKENS = 480


# --------------------------------------------------------------------------- #
# Punctuation pauses
# --------------------------------------------------------------------------- #
# The model decides its own phrasing inside a single generation, and on long
# input it leaves only a short breath between sentences — sometimes none at all.
#
# Sentence and clause pauses are therefore NOT applied here: cutting the text at
# every full stop would make each sentence its own generation and restart the
# prosody at every boundary. Instead the finished audio is measured and the gaps
# the model already produced are padded out to the requested length
# (audio.stretch_silences), which adds no generations and cannot drop speech.
#
# Only a paragraph break still splits the text, because a line break is
# structural, rare, and its silence has to be created rather than lengthened.
DEFAULT_PAUSE_SENTENCE = 0.5     # minimum gap at 。！？；…
DEFAULT_PAUSE_COMMA = 0.0        # off by default: over-pausing at ，、 hurts flow
DEFAULT_PAUSE_PARAGRAPH = 0.7    # inserted at a line break
MAX_PAUSE = 3.0

# Full-width punctuation is unambiguous — in CJK text it is always a boundary.
# Its half-width twin is not: "8:30", "1,200" and "3.5" are one token, not two,
# so an ASCII mark only ends a unit when it is not sitting between two
# alphanumerics. Without that guard the splitter cuts inside a number and the
# rejoin puts a space there ("8: 30"), which the model then reads as two numbers.
_FW_SENT = "。！？；…"
_FW_CLAUSE = "，、："
_HW_SENT = "!?;."
_HW_CLAUSE = ",:"
_SENT_CHARS = _FW_SENT + _HW_SENT
_CLAUSE_CHARS = _FW_CLAUSE + _HW_CLAUSE
# Quotes/brackets that close *after* the punctuation ("他说：“好。”") belong to the
# sentence they end, not to the next one.
_CLOSERS = "”’」』）)》〉】]"

_FW = re.escape(_FW_SENT + _FW_CLAUSE)
_HW = re.escape(_HW_SENT + _HW_CLAUSE)
# A punctuation run: full-width marks freely, half-width ones only where they
# are not enclosed by alphanumerics on both sides.
_PUNCT_RUN = rf"(?:[{_FW}]|(?<![0-9A-Za-z])[{_HW}]|[{_HW}](?![0-9A-Za-z]))+"

# One terminator = a punctuation run, any closing quotes, and the whitespace up
# to and including the line break that follows it. The bare `\n+` alternative
# catches a line break with no punctuation before it.
_TERM = re.compile(
    rf"{_PUNCT_RUN}"
    rf"[{re.escape(_CLOSERS)}]*"
    r"[^\S\n]*\n*"
    r"|\n+"
)


def _classify(term: str) -> str:
    """Boundary kind for a matched terminator: paragraph > sentence > clause."""
    if "\n" in term:
        return "para"
    if any(c in _SENT_CHARS for c in term):
        return "sent"
    return "clause"


def _split_units(text: str, max_chars: int) -> list[tuple[str, str, bool]]:
    """Sentence/clause units in reading order as (text, boundary, space_before).

    `boundary` is 'para' | 'sent' | 'clause' | '' (the trailing fragment).
    `space_before` records whether whitespace separated this unit from the
    previous one in the source, so re-joining units restores the original
    spacing exactly — Chinese needs no separator, "Hello, world." does, and
    "8:30" must not gain one.

    A unit longer than max_chars has no punctuation to break on, so it is cut on
    raw length; only its final slice keeps the boundary (and therefore the pause).
    """
    pieces: list[tuple[str, str]] = []
    pos = 0
    for m in _TERM.finditer(text):
        pieces.append((text[pos:m.end()], _classify(m.group(0))))
        pos = m.end()
    if text[pos:]:
        pieces.append((text[pos:], ""))

    units: list[tuple[str, str, bool]] = []
    gap = False          # whitespace seen since the previous unit's last char
    for piece, kind in pieces:
        body = piece.strip()
        if not body:
            gap = gap or bool(piece)
            continue
        space = gap or piece[:1].isspace()
        gap = piece[-1:].isspace()
        if len(body) <= max_chars:
            units.append((body, kind, space))
            continue
        for i in range(0, len(body), max_chars):
            slice_ = body[i:i + max_chars]
            is_last = i + max_chars >= len(body)
            units.append((slice_, kind if is_last else "", space and i == 0))
    return units


def _pack(units: list[tuple[str, bool]], max_chars: int) -> list[str]:
    """Pack a run of (text, space_before) units into the fewest chunks that each
    stay within max_chars, keeping the pieces roughly equal in size (so there is
    no stubby final chunk that reads oddly)."""
    if not units:
        return []

    # Aim for the fewest chunks possible, then even them out: packing to an even
    # target avoids a long chunk followed by a stubby one (e.g. 182 + 24). The
    # total counts the separators that will be put back, or a run of Latin units
    # would be measured short and split when it did in fact fit.
    total = sum(len(t) for t, _ in units) + sum(1 for _, sp in units[1:] if sp)
    n_chunks = max(1, -(-total // max_chars))       # ceil(total / max_chars)
    target = -(-total // n_chunks)                   # ceil(total / n_chunks)

    chunks: list[str] = []
    leads: list[bool] = []       # space_before of each chunk's first unit
    cur = ""
    cur_lead = False
    for text_, space in units:
        if not cur:
            cur, cur_lead = text_, space
            continue
        # Flush before adding would overshoot the even target. target <= max_chars
        # and every unit <= max_chars, so no chunk can exceed max_chars.
        sep = " " if space else ""
        if len(cur) + len(sep) + len(text_) > target:
            chunks.append(cur)
            leads.append(cur_lead)
            cur, cur_lead = text_, space
        else:
            cur += sep + text_
    if cur:
        chunks.append(cur)
        leads.append(cur_lead)

    # Fold a stubby trailing chunk back into its neighbour when it still fits —
    # a lone 20-char tail generated on its own reads worse than a fuller chunk.
    # Pop first, then append to the new last chunk: writing `chunks[-2] +=
    # chunks.pop()` would resolve the -2 index *after* pop() shrinks the list and
    # corrupt the wrong chunk.
    while (
        len(chunks) >= 2
        and len(chunks[-1]) < target * 0.6
        and len(chunks[-2]) + (1 if leads[-1] else 0) + len(chunks[-1]) <= max_chars
    ):
        tail = chunks.pop()
        tail_lead = leads.pop()
        chunks[-1] += (" " if tail_lead else "") + tail
    return chunks


def split_with_pauses(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    *,
    sentence: float = 0.0,
    comma: float = 0.0,
    paragraph: float = 0.0,
) -> list[tuple[str, float]]:
    """Synthesis chunks paired with the silence (seconds) to splice in after each.

    A boundary whose pause is above zero ends its chunk, so the silence lands
    exactly on that punctuation mark. Boundaries with a zero pause are packed
    together as before, letting one generation cover several sentences and keep
    its prosody continuous. The final chunk never carries a pause — trailing
    silence is the caller's "pause after" setting.
    """
    text = (text or "").strip()
    if not text:
        return []

    pause = {
        "sent": min(MAX_PAUSE, max(0.0, float(sentence))),
        "clause": min(MAX_PAUSE, max(0.0, float(comma))),
        "para": min(MAX_PAUSE, max(0.0, float(paragraph))),
        "": 0.0,
    }
    # A line break is also a sentence end: never let it pause *less* than a plain
    # full stop, or "。\n" would break more tightly than "。".
    pause["para"] = max(pause["para"], pause["sent"])

    out: list[tuple[str, float]] = []
    run: list[tuple[str, bool]] = []

    def flush(gap: float) -> None:
        if not run:
            return
        chunks = _pack(run, max_chars)
        run.clear()
        out.extend((c, 0.0) for c in chunks)
        if chunks and gap > 0:
            out[-1] = (out[-1][0], gap)

    for body, kind, space in _split_units(text, max_chars):
        run.append((body, space))
        if pause[kind] > 0:
            flush(pause[kind])
    flush(0.0)

    if out:
        out[-1] = (out[-1][0], 0.0)
    return out


def split_for_tts(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Chunk text for synthesis, ignoring pauses (size-driven splitting only)."""
    return [c for c, _ in split_with_pauses(text, max_chars)]


def chunk_max_tokens(chunk: str, fish: bool = False) -> int:
    """Per-chunk token cap: comfortably above what a chunk this long needs
    (~2.8 tokens/char observed) so generation always reaches EOS.

    Fish gets a hard, low ceiling instead: its 44.1 kHz vocoder decodes the whole
    chunk in one Metal buffer, so an over-long generation would blow past the
    ~8.9 GiB single-buffer limit. FISH_MAX_CHUNK_CHARS keeps chunks short enough
    that this cap is never actually reached mid-utterance (which would clip the
    tail to silence)."""
    if fish:
        return max(256, min(FISH_MAX_TOKENS, len(chunk) * 8 + 128))
    return max(512, min(4096, len(chunk) * 8 + 128))
